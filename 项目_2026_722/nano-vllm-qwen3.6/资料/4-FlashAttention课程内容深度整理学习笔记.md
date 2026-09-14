# FlashAttention 课程内容深度整理学习笔记

> 本笔记根据课程转文字整理。原转写中存在较多语音识别错误，例如 “FlyShareTonson” 应理解为 **FlashAttention**，“Eutonson” 应理解为 **Attention**，“SouthMax” 应理解为 **Softmax**，“MIMORRY EFFECIENT” 应理解为 **Memory Efficient**。本文已按正确技术名词进行修正和重组。

---

## 一、这节课的核心宗旨

这节课主要讲的是 **FlashAttention 为什么能加速 Transformer 中的 Attention 计算，并且为什么它还能节省显存**。

传统 Attention 的计算公式本身并不复杂：

```text
Q, K, V = input hidden states 经过线性变换得到
S = QK^T
P = softmax(S)
O = PV
```

但是在真实 GPU 上运行时，瓶颈并不只来自矩阵乘法本身，而是来自大量中间结果在 **HBM 显存** 和 **片上 SRAM/cache** 之间来回读写。FlashAttention 的核心思想不是近似 Attention，也不是牺牲精度，而是从 **IO-aware** 的角度重新组织 Attention 的计算过程，尽量减少对 HBM 的访问。

一句话总结：

**FlashAttention 通过分块计算、算子融合、避免保存大规模中间矩阵、反向传播重计算中间结果，把标准 Attention 从“显存读写瓶颈”中解放出来，从而实现更快的训练速度和更低的显存占用。**

这节课的技术主线可以概括为：

1. 标准 Attention 会产生很大的中间矩阵 `S` 和 `P`；
2. `S` 和 `P` 的大小随序列长度平方增长；
3. 这些中间矩阵频繁读写 HBM，导致 Attention 成为 memory-bound 操作；
4. FlashAttention 通过 tiling 分块，把 Q/K/V 的小块放入 SRAM 内部计算；
5. 通过 fusion 融合 `QK^T`、Softmax、`PV`，避免把 `S` 和 `P` 写回 HBM；
6. Softmax 由于需要整行归一化，必须设计可分块的 online safe softmax；
7. 反向传播时不保存完整 `S` 和 `P`，而是保存少量统计量并重计算；
8. 最终实现 exact attention，同时提高速度、降低显存。

---

## 二、课程按时间推进梳理知识点

### 1. FlashAttention 的论文目标：Fast、Memory Efficient、Exact、IO-aware

课程一开始从 FlashAttention 的论文题目切入，指出它有几个关键词：

| 关键词 | 含义 |
|---|---|
| Fast | 加快 Attention 的计算速度，从而提升训练速度 |
| Memory Efficient | 减少显存占用，尤其减少随序列长度平方增长的中间矩阵 |
| Exact Attention | 计算结果和标准 Attention 一致，不是近似算法 |
| IO-aware | 关注 GPU 内存层级之间的数据读写开销，而不是只关注 FLOPs |

这里最容易误解的是 **Exact Attention**。

很多 Attention 优化算法是近似算法，例如通过低秩近似、稀疏化、局部窗口等方式减少计算量。这类方法通常会改变 Attention 的数学结果。FlashAttention 不一样，它仍然计算完整的标准 Attention，只是改变了计算顺序和内存访问方式。

所以 FlashAttention 的核心价值是：

**不牺牲模型效果，同时提升速度和节省显存。**

---

### 2. 标准 Attention 的计算流程

课程接着回顾标准 Attention：

```text
输入 token hidden states
  ↓
线性变换得到 Q、K、V
  ↓
S = QK^T
  ↓
P = softmax(S)
  ↓
O = PV
  ↓
输出 attention result
```

这里做了简化，没有展开：

1. scale，也就是除以 `sqrt(d_k)`；
2. 多头注意力；
3. dropout；
4. causal mask 或 padding mask。

但这个简化不影响理解 FlashAttention 的核心，因为 FlashAttention 最重要的问题不是公式复杂，而是 **中间矩阵如何存储和读写**。

如果序列长度为 `N`，head dimension 为 `D`，那么：

```text
Q: N x D
K: N x D
V: N x D
S = QK^T: N x N
P = softmax(S): N x N
O = PV: N x D
```

关键问题在于 `S` 和 `P` 都是 `N x N`。当序列长度变长时，它们的显存占用会按平方增长。

例如序列长度从 4096 增加到 8192，`S` 和 `P` 的元素数量不是变成 2 倍，而是变成 4 倍。

---

### 3. PyTorch 标准实现中的 HBM 读写过程

课程用 PyTorch 或普通 GPU Attention 实现解释了 Attention 的真实执行过程。

假设 Q、K、V 一开始都存储在 HBM 中，典型流程是：

| 步骤 | 操作 | 主要数据流 |
|---|---|---|
| 1 | 从 HBM 读取 Q、K | HBM -> SRAM/register |
| 2 | 计算 `S = QK^T` | GPU core 计算 |
| 3 | 将 S 写回 HBM | SRAM/register -> HBM |
| 4 | 再从 HBM 读取 S | HBM -> SRAM/register |
| 5 | 计算 `P = softmax(S)` | GPU core 计算 |
| 6 | 将 P 写回 HBM | SRAM/register -> HBM |
| 7 | 从 HBM 读取 P 和 V | HBM -> SRAM/register |
| 8 | 计算 `O = PV` | GPU core 计算 |
| 9 | 将 O 写回 HBM | SRAM/register -> HBM |

这个流程的问题是，中间矩阵 `S` 和 `P` 都非常大，而且要被写入 HBM、再从 HBM 读回来。

这就导致 Attention 中大量时间不是花在真正的矩阵乘法上，而是花在显存读写上。

---

### 4. 为什么 Attention 是 memory-bound，而不是单纯 compute-bound

课程区分了两类性能瓶颈：

| 类型 | 含义 | 典型操作 |
|---|---|---|
| Compute-bound | 计算量很大，瓶颈在 GPU 算力 | 大矩阵乘法、卷积 |
| Memory-bound | 数据读写很多，瓶颈在显存带宽 | Softmax、Dropout、归约操作 |

矩阵乘法通常是 compute-bound，因为它能进行大量乘加运算，GPU 算力利用率高。

但 Softmax、Dropout、reduce sum、reduce max 这类操作通常是 memory-bound，因为它们需要读写大量数据，但每个数据点上的计算很少。

Attention 里面虽然有 `QK^T` 和 `PV` 两个矩阵乘法，但整个 Attention 算子的实际瓶颈往往被 `S`、`P` 的读写拖慢。

所以 FlashAttention 的切入点不是“减少多少乘法”，而是：

**减少 HBM 读写次数，让更多计算在片上 SRAM 中完成。**

---

### 5. GPU 内存层级：HBM 与 SRAM

课程强调了 GPU 内存层级：

| 存储位置 | 特点 | 类比理解 |
|---|---|---|
| HBM | 容量大，但访问相对慢 | GPU 显存 |
| SRAM / shared memory / register | 容量小，但访问非常快 | GPU 芯片内部缓存 |

HBM 可以存储模型权重、激活、中间结果等大数据，但访问成本高。

SRAM 容量很小，不能放下完整的 `N x N` Attention 矩阵，但可以放下某个小块。

FlashAttention 的关键就是：

**不要试图一次性生成完整的 S 和 P，而是把矩阵分成小块，一块一块放到 SRAM 中计算。**

---

### 6. FlashAttention 的第一个核心：分块计算 Tiling

FlashAttention 会把 Q、K、V 分成 block。

普通 Attention 的思路是：

```text
先完整算出 S = QK^T
再完整算出 P = softmax(S)
再完整算出 O = PV
```

FlashAttention 的思路是：

```text
读入一小块 Q
读入一小块 K、V
在 SRAM 中计算当前块对 O 的贡献
更新 O 的局部结果
继续处理下一个 K、V 块
直到得到完整 O
```

这样做的好处是：

1. 不需要把完整 `S` 存到 HBM；
2. 不需要把完整 `P` 存到 HBM；
3. K、V 的一个分块可以被多个 Q 分块复用；
4. 中间计算结果尽量留在 SRAM 中。

这里的本质是把 Attention 从“先生成巨大中间矩阵”变成“边算边累计输出”。

---

### 7. FlashAttention 的第二个核心：算子融合 Fusion

课程提到，memory-bound 优化中常见方法是 **fusion**。

普通实现中，`QK^T`、Softmax、`PV` 可能是多个独立 kernel：

```text
Kernel 1: 计算 S = QK^T，写回 HBM
Kernel 2: 读取 S，计算 softmax，写回 P
Kernel 3: 读取 P 和 V，计算 O
```

FlashAttention 尽量把这些步骤融合在一起：

```text
读取 Q/K/V block
  ↓
计算局部 QK^T
  ↓
局部 Softmax 统计与归一化
  ↓
乘 V 并更新 O
  ↓
只把最终 O 写回 HBM
```

这样就把多次 HBM 访问压缩成尽量少的访问。

对于 AI Infra 面试来说，可以这样表达：

**Fusion 的目的不是改变数学公式，而是减少 kernel 之间中间结果落 HBM 的次数，提高数据局部性。**

---

### 8. 为什么 Softmax 分块比较困难

如果没有 Softmax，只计算类似：

```text
O = QK^T V
```

那么分块相对简单，因为矩阵乘法天然可以分块累加。

但 Attention 中间有 Softmax：

```text
P_i = exp(S_i) / sum(exp(S_i))
```

Softmax 是按行进行的。对某一行来说，必须知道整行所有元素，才能得到正确的分母。

这带来一个问题：

**如果只看当前 block，就不知道这一行在其他 block 里的最大值和总和，因此不能直接得到全局正确的 Softmax。**

所以 FlashAttention 必须解决：

1. Softmax 如何分块计算；
2. 分块 Softmax 如何保持数值稳定；
3. 局部结果如何合并为全局结果。

这就是课程后半部分重点讲的 online softmax / safe softmax。

---

### 9. Safe Softmax：解决 FP16 数值溢出

课程先讲了 Safe Softmax。

普通 Softmax 是：

```text
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

问题是，在 FP16 下，如果 `x` 稍微大一些，`exp(x)` 就可能溢出。

为了解决这个问题，Safe Softmax 会先减去这一行的最大值 `m`：

```text
m = max(x_1, x_2, ..., x_n)
softmax(x_i) = exp(x_i - m) / sum_j exp(x_j - m)
```

因为 `x_i - m <= 0`，所以指数项不会变得特别大，从而避免溢出。

注意，分子和分母同时除以 `exp(m)`，所以 Softmax 的数学结果不变。

Safe Softmax 的计算过程可以理解为：

1. 找到行最大值 `m`；
2. 计算 `p_i = exp(x_i - m)`；
3. 计算 `l = sum_i p_i`；
4. 得到 `softmax(x_i) = p_i / l`。

这里的 `m` 和 `l` 会成为 FlashAttention 分块合并的关键统计量。

---

### 10. Online Softmax：让 Softmax 可以分块合并

为了让 Softmax 可以分块计算，课程把一行数据拆成两个部分：

```text
block 1: x_1 ... x_n
block 2: x_{n+1} ... x_{2n}
```

对每个 block 分别计算：

```text
m_1 = max(block 1)
l_1 = sum(exp(x_i - m_1))

m_2 = max(block 2)
l_2 = sum(exp(x_i - m_2))
```

然后要合并成全局 Softmax。

全局最大值是：

```text
m = max(m_1, m_2)
```

但 `l_1` 和 `l_2` 是基于各自局部最大值算出来的，不能直接相加。需要把它们调整到同一个全局最大值 `m` 下：

```text
l = l_1 * exp(m_1 - m) + l_2 * exp(m_2 - m)
```

同理，每个 block 里的 `exp(x_i - m_block)` 也要乘上修正系数：

```text
exp(x_i - m_global)
= exp(x_i - m_block) * exp(m_block - m_global)
```

这就是 online softmax 的核心。

它说明：

**即使 Softmax 依赖整行数据，也可以通过保存每行的最大值 m 和归一化分母 l 来逐块合并。**

---

### 11. FlashAttention 前向传播的整体伪代码思想

课程中提到了伪代码。可以用更清晰的方式整理如下：

```text
输入：Q, K, V in HBM
输出：O

初始化：
  O = 0
  m = -inf
  l = 0

按 block 遍历 K, V：
  把 K_j, V_j 从 HBM 读入 SRAM

  按 block 遍历 Q：
    把 Q_i, O_i, m_i, l_i 读入 SRAM

    计算当前块分数：
      S_ij = Q_i K_j^T

    计算当前块最大值：
      m_ij = rowmax(S_ij)

    合并历史最大值：
      m_new = max(m_i, m_ij)

    计算当前块指数项：
      P_ij = exp(S_ij - m_new)

    更新归一化分母：
      l_new = exp(m_i - m_new) * l_i
              + sum(exp(S_ij - m_new))

    更新输出：
      O_i = [exp(m_i - m_new) * l_i * O_i + P_ij V_j] / l_new

    写回 O_i, m_new, l_new

返回 O
```

这里最重要的是 `O` 的更新不是简单相加，而是要根据新的全局最大值和新的归一化分母重新缩放。

因为旧的 `O_i` 是基于旧的 softmax 分母算出来的，当新 block 加进来以后，softmax 的分母变了，旧结果必须按比例调整。

---

### 12. FlashAttention 为什么能节省显存

标准 Attention 在训练时通常需要保存 `S` 和 `P`，因为反向传播需要它们来计算梯度。

FlashAttention 不保存完整的 `S` 和 `P`，只保存较小的统计量，例如每行的：

```text
m: row max
l: row sum of exp
```

或者等价的 log-sum-exp 统计量。

这样显存占用从原来的：

```text
O(N^2)
```

降低为接近：

```text
O(N)
```

其中 `N` 是序列长度。

对于长序列训练，节省非常明显。课程中提到，当序列长度越长，FlashAttention 相比普通实现节省的显存越多；在较长序列下，显存节省可以达到非常显著的倍数。

---

### 13. FlashAttention 反向传播：重计算中间结果

一个自然问题是：

**如果前向传播没有保存完整的 S 和 P，那反向传播怎么办？**

答案是：

**反向传播时重新计算需要的中间结果。**

这类似于 gradient checkpointing 的思想：

1. 前向传播不保存所有中间激活；
2. 只保存少量必要信息；
3. 反向传播时根据保存的信息重新计算中间结果；
4. 用额外计算换取显存节省和 IO 减少。

FlashAttention 保存了足够恢复 Softmax 的统计量，因此反向传播可以分块重算 `S` 和 `P`，再计算梯度。

这里有一个重要权衡：

| 方案 | 优点 | 缺点 |
|---|---|---|
| 保存完整 S/P | 反向传播不用重算 | 显存占用大，HBM IO 多 |
| 不保存 S/P，反向重算 | 显存小，IO 少 | 计算量略有增加 |

FlashAttention 的实践结果说明：虽然计算量略有增加，但减少 HBM IO 带来的收益更大，所以整体速度更快。

---

### 14. FlashAttention-2 的改进方向

课程最后提到 FlashAttention-2。

FlashAttention-2 的大方向和 FlashAttention-1 一致，仍然是：

1. 分块；
2. 融合；
3. 减少 HBM 读写；
4. 利用 online softmax；
5. 保持 exact attention。

但 FlashAttention-2 做了更多工程优化，例如：

1. 减少非矩阵乘法部分的计算开销；
2. 改变循环顺序，提高并行度；
3. 更充分利用 GPU 线程块和 warp；
4. 对 causal mask 的上三角无效区域进行跳过，减少无意义计算。

尤其是 causal attention 中，未来 token 对当前 token 不可见。Attention 矩阵的上三角区域会被 mask 掉。

如果某些 block 完全落在被 mask 的区域，就不需要计算这些 block。

这进一步减少了计算量。

---

## 三、整节课的完整技术主线：从输入到输出

现在把整节课串成一条完整流程。

### 1. 输入阶段：token hidden states 进入 Attention

Transformer 中每个 token 会先被表示成 hidden state。进入 Attention 层后，通过三个线性层得到：

```text
Q: query
K: key
V: value
```

其中：

1. Q 表示当前位置要查询什么信息；
2. K 表示每个位置能被匹配的特征；
3. V 表示每个位置真正提供给输出的信息。

---

### 2. 标准 Attention 阶段：生成完整 S 和 P

普通 Attention 会先计算：

```text
S = QK^T
```

`S` 表示每个 token 对其他 token 的注意力打分。

然后计算：

```text
P = softmax(S)
```

`P` 表示归一化后的注意力权重。

最后：

```text
O = PV
```

得到每个 token 聚合后的新表示。

---

### 3. 问题出现：S 和 P 太大，HBM IO 太多

当序列长度是 `N` 时，`S` 和 `P` 都是 `N x N`。

如果序列很长，这两个矩阵会非常大。

更关键的是，它们不仅占显存，还会被反复写入和读取 HBM：

```text
算 S -> 写 HBM
读 S -> 算 softmax -> 写 P 到 HBM
读 P 和 V -> 算 O
```

因此 Attention 的实际瓶颈变成显存带宽，而不是单纯计算能力。

---

### 4. FlashAttention 改造：不生成完整 S/P

FlashAttention 不再把完整的 `S` 和 `P` 作为中间矩阵落到 HBM。

它把 Q、K、V 分成小块：

```text
Q_i, K_j, V_j
```

每次只把一部分 block 放入 SRAM：

```text
S_ij = Q_i K_j^T
```

然后立刻做 Softmax 的局部统计和 `PV` 更新。

这样中间结果只在 SRAM 中短暂存在，不写回 HBM。

---

### 5. Softmax 难点：用 online softmax 合并 block

因为 Softmax 需要整行归一化，所以 FlashAttention 不能简单地对每个 block 独立 softmax。

它需要保存每一行到目前为止的：

```text
m: 当前已经看到的最大值
l: 当前已经看到的 exp 求和
O: 当前已经累计的输出
```

每看到一个新的 K/V block，就更新：

```text
m_new
l_new
O_new
```

通过这种方式，FlashAttention 可以一边扫描 K/V block，一边得到全局正确的 Softmax 结果。

---

### 6. 输出阶段：只写回最终 O

所有 K/V block 处理完后，每个 Q block 对应的输出 `O_i` 就是完整 Attention 的正确结果。

最终只需要把 `O` 写回 HBM。

完整流程变成：

```text
Q/K/V in HBM
  ↓
分块读入 SRAM
  ↓
局部计算 QK^T
  ↓
online safe softmax
  ↓
局部乘 V 并更新 O
  ↓
最终 O 写回 HBM
```

相比标准 Attention，它避免了完整 `S` 和 `P` 的 HBM 写读。

---

## 四、与 AI Infra / 大模型推理系统的关系

虽然课程主要从训练角度讲 FlashAttention，但它对 AI Infra 推理也非常重要。

### 1. Attention 是大模型推理的核心算子

无论训练还是推理，Transformer 的核心模块都是 Attention 和 MLP。

在推理阶段，尤其是 prefill 阶段，模型需要对 prompt 中所有 token 做 Attention：

```text
输入 prompt tokens
  ↓
prefill 并行计算所有 token 的 hidden states
  ↓
构建 KV cache
  ↓
decode 逐 token 生成
```

prefill 阶段的 Attention 仍然会处理较长序列，因此 FlashAttention 能显著影响 TTFT。

TTFT 即 time to first token，表示从请求进入到生成第一个 token 的时间。

---

### 2. FlashAttention 与 prefill

Prefill 阶段通常是 compute-heavy，因为要一次性处理 prompt 的全部 token。

如果 prompt 很长，Attention 的 `QK^T` 会形成较大的注意力矩阵。

FlashAttention 在这里的作用是：

1. 减少 Attention 中间矩阵显存占用；
2. 减少 HBM IO；
3. 提高长 prompt 的 prefill 速度；
4. 降低长上下文请求的显存压力。

所以 vLLM、TensorRT-LLM、SGLang 等推理系统通常都会关注高性能 Attention kernel。

---

### 3. FlashAttention 与 decode

Decode 阶段每次只生成一个新 token。

此时新 token 的 Q 只有一个位置，但它要和历史所有 K/V 做 Attention。

Decode 阶段的瓶颈经常来自：

1. KV cache 读取；
2. 显存带宽；
3. batch 调度；
4. kernel launch 和同步开销。

FlashAttention 对 decode 的帮助和 prefill 不完全一样，因为 decode 的计算形态更像单 query 对长 KV cache 的 attention。实际推理框架中通常还会使用专门的 paged attention、flash decoding 或其他 decode attention kernel。

可以这样理解：

| 阶段 | 主要特点 | Attention 优化重点 |
|---|---|---|
| Prefill | 一次处理整段 prompt | FlashAttention 类 block attention |
| Decode | 每次生成一个 token | KV cache 读取、paged attention、flash decoding |

---

### 4. FlashAttention 与 KV Cache

KV Cache 是推理阶段非常重要的机制。

在 decode 时，历史 token 的 K/V 不需要重复计算，而是保存在 KV cache 中。每生成一个新 token，只需要计算新 token 的 Q/K/V，并让新 Q 去 attend 历史 K/V。

FlashAttention 主要优化 Attention 计算过程中的 IO，而 KV Cache 主要避免重复计算历史 K/V。

两者关系如下：

| 技术 | 解决的问题 |
|---|---|
| KV Cache | 避免 decode 阶段重复计算历史 token 的 K/V |
| FlashAttention | 减少 Attention 内部中间矩阵和 HBM IO |
| PagedAttention | 更高效管理多请求 KV cache 的显存分页 |

所以在 AI Infra 面试中，不要把 FlashAttention 和 KV Cache 混为一谈。

---

## 五、关键概念表

| 概念 | 解释 | 在本节课中的作用 |
|---|---|---|
| Q/K/V | Attention 的 query、key、value 矩阵 | 标准 Attention 输入 |
| S | `QK^T` 得到的注意力分数矩阵 | 标准实现中的大中间矩阵 |
| P | `softmax(S)` 得到的注意力权重矩阵 | 标准实现中的大中间矩阵 |
| O | `PV` 得到的 Attention 输出 | 最终输出 |
| HBM | GPU 高带宽显存，容量大但访问慢于片上缓存 | FlashAttention 主要减少其读写 |
| SRAM | GPU 片上高速缓存，容量小但访问快 | FlashAttention 尽量把 block 放在这里计算 |
| Tiling | 分块计算 | 避免一次性生成完整 S/P |
| Fusion | 算子融合 | 减少 kernel 间中间结果落 HBM |
| Safe Softmax | 减去最大值避免 exp 溢出 | 保证 FP16 下数值稳定 |
| Online Softmax | 分块合并 Softmax 的统计量 | 让 FlashAttention 可以分块计算 |
| Gradient Checkpointing | 反向传播重算中间激活以节省显存 | FlashAttention 反向传播思想类似 |
| Memory-bound | 瓶颈在数据读写 | Attention 优化重点 |
| Compute-bound | 瓶颈在计算量 | 大矩阵乘法常见 |

---

## 六、面试常见问题与回答

### 1. FlashAttention 解决了什么问题？

FlashAttention 主要解决标准 Attention 中 HBM 读写开销大和中间矩阵显存占用高的问题。

标准 Attention 会显式生成并保存 `S = QK^T` 和 `P = softmax(S)`，它们都是 `N x N` 矩阵，显存占用随序列长度平方增长。FlashAttention 通过分块计算和算子融合，避免把完整的 `S` 和 `P` 写入 HBM，从而减少 IO、提升速度、降低显存。

---

### 2. FlashAttention 是近似 Attention 吗？

不是。

FlashAttention 是 exact attention。它和标准 Attention 的数学结果一致，只是改变了计算顺序和内存访问方式。它不像稀疏 Attention 或低秩 Attention 那样通过近似来减少计算。

---

### 3. 为什么标准 Attention 显存占用是 O(N^2)？

因为标准 Attention 会生成 `S = QK^T` 和 `P = softmax(S)`。

如果序列长度是 `N`，那么每个 attention head 的 `S` 和 `P` 都是 `N x N`，所以中间矩阵显存占用随 `N^2` 增长。

---

### 4. FlashAttention 为什么能把显存复杂度降到接近 O(N)？

因为它不保存完整的 `N x N` 注意力矩阵，而是分块计算，每次只在 SRAM 中保留一小块 `S_ij` 和临时统计量。

前向传播只需要保存输出 `O` 以及每行的 softmax 统计量，例如最大值 `m` 和归一化分母 `l`，这些都是随序列长度线性增长的。

---

### 5. FlashAttention 为什么能加速？

它减少了 HBM 读写。

标准 Attention 中，`S` 和 `P` 会被写回 HBM，然后再读出来继续计算。FlashAttention 通过分块和融合，在 SRAM 中完成 `QK^T`、Softmax 和 `PV` 的局部计算，只把最终结果写回 HBM。

虽然它可能增加少量额外计算，但大幅减少 IO 后，整体速度更快。

---

### 6. 什么是 IO-aware？

IO-aware 指算法设计时关注数据在不同存储层级之间搬运的代价。

在 GPU 中，HBM 容量大但访问相对慢，SRAM/cache 容量小但访问快。FlashAttention 不是只看 FLOPs，而是通过减少 HBM 访问来提升实际性能。

---

### 7. 为什么 Softmax 分块困难？

Softmax 需要对整行做归一化。

对于一行 `S_i`，只有知道整行所有元素的最大值和指数和，才能得到正确的 softmax 结果。如果只看一个 block，会缺少其他 block 的信息。

FlashAttention 使用 online softmax，通过保存和合并每行的最大值 `m` 和分母 `l` 来解决这个问题。

---

### 8. Safe Softmax 的作用是什么？

Safe Softmax 通过在指数运算前减去该行最大值，避免 `exp(x)` 在 FP16 下溢出。

公式是：

```text
softmax(x_i) = exp(x_i - m) / sum_j exp(x_j - m)
m = max(x)
```

这样 `x_i - m <= 0`，指数值不会过大，数值更稳定。

---

### 9. Online Softmax 如何合并两个 block？

假设两个 block 的最大值和分母分别是：

```text
m_1, l_1
m_2, l_2
```

全局最大值是：

```text
m = max(m_1, m_2)
```

全局分母是：

```text
l = l_1 * exp(m_1 - m) + l_2 * exp(m_2 - m)
```

这样就能把局部 softmax 统计量调整到同一个全局最大值下。

---

### 10. FlashAttention 前向传播为什么要保存 m 和 l？

因为反向传播和后续 block 合并都需要知道 softmax 的归一化信息。

`m` 表示每行当前已处理部分的最大值，`l` 表示基于该最大值的指数和。保存它们就可以在不保存完整 `S` 和 `P` 的情况下，重建 softmax 相关中间结果。

---

### 11. FlashAttention 反向传播为什么需要重计算？

因为前向传播没有保存完整的 `S` 和 `P`。

反向传播计算梯度时需要这些中间信息，所以 FlashAttention 利用保存的统计量重新计算局部 `S` 和 `P`，再完成梯度计算。这相当于用少量额外计算换取显存和 IO 的节省。

---

### 12. FlashAttention 和 gradient checkpointing 有什么相似之处？

两者都体现了“用计算换显存”的思想。

Gradient checkpointing 不保存所有中间激活，而是在反向传播时重算。FlashAttention 也不保存完整 `S` 和 `P`，而是在反向传播时根据 Q/K/V 和 softmax 统计量重新计算需要的中间结果。

---

### 13. FlashAttention 对推理有什么作用？

在推理的 prefill 阶段，模型需要对 prompt 中全部 token 做 Attention。如果 prompt 较长，Attention 的 IO 和显存压力很大。

FlashAttention 可以加速 prefill，降低长上下文下的显存占用，从而改善 TTFT。

在 decode 阶段，由于每次只生成一个 token，优化重点还包括 KV cache 读取、paged attention、batch 调度和专门的 decode kernel。

---

### 14. FlashAttention 和 PagedAttention 有什么区别？

FlashAttention 优化的是 Attention 算子内部的计算和 IO，重点是避免生成和保存完整注意力矩阵。

PagedAttention 优化的是推理系统中的 KV cache 管理，把 KV cache 按块分页，减少显存碎片，支持连续批处理和多请求调度。

简单说：

```text
FlashAttention: 算子级优化
PagedAttention: 推理系统级 KV cache 管理优化
```

---

### 15. 为什么说 FlashAttention 的收益在长序列下更明显？

因为标准 Attention 的中间矩阵 `S` 和 `P` 都是 `N x N`。

序列越长，`N^2` 的中间矩阵越大，HBM 读写越重。FlashAttention 避免保存这些完整中间矩阵，所以长序列下节省的显存和 IO 更明显。

---

## 七、学习这节课时最应该抓住的主线

这节课不要只记“FlashAttention 很快”，而要抓住它背后的推理逻辑：

```text
标准 Attention 需要生成 S 和 P
  ↓
S/P 是 N x N，长序列下非常大
  ↓
S/P 需要反复写入和读取 HBM
  ↓
Attention 变成 memory-bound
  ↓
FlashAttention 用 tiling 把计算拆成 block
  ↓
用 fusion 让 QK^T、Softmax、PV 在 SRAM 内完成
  ↓
用 online safe softmax 保证分块结果仍然正确
  ↓
前向不保存完整 S/P，反向重计算
  ↓
减少 HBM IO，降低显存，提高速度，并保持 exact attention
```

如果你准备 AI Infra / 推理框架面试，最重要的是能讲清楚三句话：

1. **FlashAttention 快，不是因为它少算了很多 FLOPs，而是因为它少访问了很多 HBM。**
2. **FlashAttention 省显存，是因为它不保存完整的 `N x N` 注意力矩阵。**
3. **FlashAttention 仍然是 exact attention，靠 online softmax 保证分块计算的结果和标准 Attention 一致。**

---

## 八、复习检查题

1. 标准 Attention 为什么要生成 `S` 和 `P`？
2. 为什么 `S` 和 `P` 的显存占用是 `O(N^2)`？
3. HBM 和 SRAM 的区别是什么？
4. 什么是 memory-bound？Attention 为什么容易 memory-bound？
5. FlashAttention 的 tiling 具体解决什么问题？
6. FlashAttention 的 fusion 具体减少了哪些 HBM 读写？
7. 为什么 Softmax 不能直接按 block 独立计算？
8. Safe Softmax 为什么要减去最大值？
9. Online Softmax 中 `m` 和 `l` 分别是什么？
10. FlashAttention 反向传播为什么要重计算？
11. FlashAttention 和 PagedAttention 分别属于什么层面的优化？
12. FlashAttention 对 prefill 和 decode 的帮助有什么不同？

---

## 九、一句话总结

FlashAttention 的本质是：**在不改变标准 Attention 数学结果的前提下，通过分块、融合和 online softmax，把 Attention 的主要瓶颈从频繁访问 HBM 转移到更高效的片上计算，从而实现更快、更省显存的 Transformer Attention。**


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
