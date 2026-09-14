# GPU 内部自注意力计算流程课程学习笔记

> 本笔记根据课程转文字整理。原文是语音识别结果，存在较多错别字和技术名词识别错误，例如 “秩序一例” 应理解为 **自注意力**，“日尔斯军名Multiple Sensor” 应理解为 **Streaming Multiprocessor / SM**，“LEDDataCache” 应理解为 **L1 Data Cache**，“些的memory” 应理解为 **shared memory**，“TENZER CALL” 应理解为 **Tensor Core**，“QTACO” 应理解为 **CUDA Core**。本文已按正确术语重新整理。

---

## 一、这节课的核心宗旨

这节课主要讲的是：

**一次 Transformer 自注意力计算，在 GPU 内部到底是怎样一步一步执行的。**

它不是单纯讲 Attention 公式，而是把公式放到 GPU 硬件结构里解释：

1. 数据最开始在 CPU 内存或磁盘中；
2. 通过 PCIe 传到 GPU 显存 HBM；
3. 计算时从 HBM 进入 L2 Cache；
4. 再进入每个 SM 内部的 SRAM 区域；
5. 最后由 Tensor Core、CUDA Core、SFU 等计算单元执行矩阵乘法、加法、指数、除法等操作；
6. 中间结果再沿着相反路径写回 HBM。

这节课的目的不是让你记住每个硬件部件的细节，而是让你理解一个关键问题：

**标准 Attention 在 GPU 上执行时，会频繁把中间矩阵从 HBM 读到 SM，再从 SM 写回 HBM。由于 HBM 访问速度远慢于 SM 内部计算速度，所以 Attention 很容易被显存读写拖慢。**

这也正是后续 FlashAttention 要解决的问题。

一句话总结：

**这节课用 GPU 硬件视角解释了标准多头自注意力的执行过程，并指出普通 Attention 的核心瓶颈来自 HBM 与 SM 之间反复搬运 Q/K/V、注意力分数矩阵 S、注意力权重矩阵 P 等数据。**

---

## 二、课程按时间推进梳理知识点

### 1. GPU 的基本结构：从 H100 到 SM

课程一开始用 H100 的内部结构图引入。

虽然现代 GPU 看起来非常复杂，但可以把它理解为由很多个基础计算单元组成，这些基础计算单元叫做：

```text
SM = Streaming Multiprocessor
```

中文常翻译为 **流式多处理器**。

在 NVIDIA GPU 中，SM 是非常重要的计算单元。一个 GPU 中有很多个 SM，每个 SM 可以并行处理一部分任务。

为了方便理解，课程把复杂的 H100 简化为一个只有 4 个 SM 的 GPU。

这种简化很重要，因为后面讲自注意力计算时，就可以把不同 token、不同 head、不同矩阵块分给不同 SM 并行计算。

---

### 2. 一个 SM 内部有什么

课程接着解释一个 SM 的内部组成。

一个 SM 里面可以简化理解为包含：

| 组成部分 | 作用 |
|---|---|
| Register | 寄存器，离计算核心最近，速度最快，用于存放当前线程正在使用的数据 |
| CUDA Core | 处理普通加法、乘法、除法等标量或向量计算 |
| Tensor Core | 专门加速矩阵乘法，尤其适合深度学习中的 GEMM |
| SFU | Special Function Unit，用于指数、对数、三角函数等特殊函数 |
| L1 Data Cache | SM 内部的一级缓存 |
| Shared Memory | 同一个 SM 内线程块共享的数据存储空间 |

课程中把 Register、L1 Cache、Shared Memory 这些靠近计算单元的高速存储区域合起来，简化称为：

```text
SRAM
```

这里的 SRAM 不是严格等同于某一个硬件模块，而是为了帮助理解：它代表 GPU 芯片内部速度很快、容量较小的存储空间。

---

### 3. HBM、L2 Cache、SRAM 的层级关系

课程随后讲 GPU 内部的数据流向。

可以把 GPU 的存储层级理解为：

```text
HBM 显存
  ↓
L2 Cache
  ↓
SM 内部 L1/shared/register，也就是简化意义上的 SRAM
  ↓
CUDA Core / Tensor Core / SFU
```

它们的特点是：

| 层级 | 容量 | 速度 | 作用 |
|---|---:|---:|---|
| HBM | 最大 | 相对最慢 | 存模型权重、输入、激活、中间结果 |
| L2 Cache | 中等 | 比 HBM 快 | 多个 SM 共享的缓存 |
| SRAM / L1 / shared / register | 很小 | 很快 | SM 内部计算直接使用 |
| 计算核心 | 无存储或极小 | 最快 | 真正执行运算 |

这里要抓住一个核心：

**数据不能凭空在 Tensor Core 上计算，必须先从 HBM 搬到更靠近计算核心的地方。**

如果数据搬运很慢，计算核心就会等待数据，导致 GPU 算力用不满。

---

### 4. CPU、内存、PCIe、NVLink 与 GPU 的关系

课程还讲了数据从 GPU 外部进入 GPU 的过程。

一般情况下：

```text
磁盘
  ↓
CPU 内存
  ↓
PCIe
  ↓
GPU HBM
```

也就是说，模型权重、输入数据等通常会先从磁盘加载到 CPU 内存，然后通过 PCIe 传输到 GPU 显存 HBM。

如果是多卡训练或多卡推理，还可能用到：

```text
NVLink
```

NVLink 是 GPU 和 GPU 之间的高速互联方式。相比 PCIe，NVLink 通常带宽更高、延迟更低，可以让多张 GPU 更高效地交换数据。

在 AI Infra 中，这些概念非常重要：

| 技术 | 在推理系统中的意义 |
|---|---|
| PCIe | CPU 和 GPU 之间传输输入、输出、权重或控制信息 |
| HBM | 存放模型权重、KV Cache、激活、中间结果 |
| NVLink | 多卡 tensor parallel / pipeline parallel / expert parallel 中通信 |
| L2 / SRAM | 高性能 kernel 设计需要重点利用的数据局部性 |

---

### 5. 自注意力计算前的数据准备：Token 和 Embedding

课程用一句诗作为例子，比如：

```text
蒹葭苍苍，白露为霜
```

为了让模型处理文本，第一步不是直接把汉字送进模型，而是先经过 tokenizer。

流程是：

```text
文本
  ↓
Tokenizer
  ↓
token id 序列
  ↓
Embedding 查表
  ↓
输入向量矩阵 X
```

假设每个字是一个 token，每个 token 都有一个编号。例如：

```text
白 -> token id = 2
```

Embedding 矩阵中第 2 行就是“白”这个 token 的向量表示。

所以 embedding 本质上是一次查表：

```text
X = embedding[token_ids]
```

这里得到的 `X` 是后续 Attention 的输入。

---

### 6. Q/K/V 参数矩阵与多头注意力

得到输入矩阵 `X` 之后，自注意力需要计算：

```text
Q = X W_Q
K = X W_K
V = X W_V
```

课程中的例子是两头注意力，也就是有两个 attention head。

多头注意力可以理解为：

1. 每个 head 都有自己的一套 Q/K/V；
2. 不同 head 可以从不同角度建模 token 之间的关系；
3. 最后把多个 head 的输出拼接起来，再通过 `W_O` 混合。

在实际实现中，`W_Q`、`W_K`、`W_V` 往往不是分开三次存储和计算，而是合并成一个大的权重矩阵：

```text
W_QKV = [W_Q, W_K, W_V]
```

这样一次矩阵乘法就可以得到拼接后的 Q/K/V：

```text
QKV = X W_QKV
```

这是很多 Transformer 实现中的常见优化。

---

### 7. 权重和输入如何进入 GPU

课程强调，`X`、`W_QKV` 等数据最初都需要被加载到 GPU 的 HBM 中。

大致流程是：

```text
CPU 内存中的 token ids / 参数
  ↓ PCIe
GPU HBM
  ↓ L2 Cache
SM 内部 SRAM
  ↓
Tensor Core 执行矩阵乘法
```

在真实推理系统中，模型权重通常会提前加载到 GPU HBM 中，不会每次请求都重新从 CPU 传入。

但是输入 token、输出结果、调度控制信息等仍然会涉及 CPU 和 GPU 之间的数据交互。

---

### 8. 第一步矩阵乘法：计算 Q/K/V

课程假设 GPU 有 4 个 SM，可以把输入 token 分成 4 份，每个 SM 处理一部分 token。

例如序列长度是 8，可以按每 2 个 token 切一块：

| SM | 负责 token |
|---|---|
| SM1 | 第 1-2 个 token |
| SM2 | 第 3-4 个 token |
| SM3 | 第 5-6 个 token |
| SM4 | 第 7-8 个 token |

每个 SM 都需要读取：

1. 自己负责的输入 `X_i`；
2. 共享的权重矩阵 `W_QKV`。

然后执行：

```text
QKV_i = X_i W_QKV
```

由于这是矩阵乘法，所以主要由 Tensor Core 完成。

计算完后，每个 SM 得到自己那一部分 token 对应的 Q/K/V 结果，再写回 L2 Cache 和 HBM。

---

### 9. 为什么 W_QKV 可以被多个 SM 复用

课程中特别强调，4 个 SM 都需要同一个 `W_QKV` 参数矩阵。

如果每个 SM 都从 HBM 重新读一遍，会浪费大量带宽。

更好的方式是：

```text
W_QKV 从 HBM 读到 L2 Cache
  ↓
多个 SM 从 L2 Cache 读取
```

这样可以减少重复访问 HBM。

这就是缓存的意义：

**把多个计算单元共同需要的数据尽量放在更近、更快的缓存中，减少访问远端 HBM 的次数。**

---

### 10. 拆分 Q/K/V：按 head 分组

得到大的 `QKV` 矩阵后，需要拆开：

```text
QKV -> Q_1, K_1, V_1, Q_2, K_2, V_2
```

其中：

```text
Q_1, K_1, V_1 -> 第一个 head
Q_2, K_2, V_2 -> 第二个 head
```

课程中用两头注意力举例，所以最后会得到两套 Q/K/V。

后续每个 head 的 attention 可以并行计算。

这也是多头注意力适合 GPU 并行的原因之一：

1. 不同 head 之间相对独立；
2. 同一个 head 内不同 token 或不同矩阵块也可以并行；
3. GPU 可以把这些计算分配给不同 SM。

---

### 11. 第二步矩阵乘法：计算注意力分数 S

自注意力的下一步是：

```text
S = Q K^T
```

如果是多头注意力，则每个 head 都单独计算：

```text
S_1 = Q_1 K_1^T
S_2 = Q_2 K_2^T
```

课程中的例子是：

1. SM1 和 SM2 负责第一个 head；
2. SM3 和 SM4 负责第二个 head。

对于第一个 head，SM1 和 SM2 可以继续按 token 行进行切分：

| SM | 负责内容 |
|---|---|
| SM1 | 第一个 head 中前 4 个 token 的 Q |
| SM2 | 第一个 head 中后 4 个 token 的 Q |

但是每个 SM 都需要完整的 `K_1^T`，因为每个 token 都要和所有 token 计算注意力分数。

这点非常关键：

**计算某个 token 的 attention 时，它的 Q 要和所有历史/上下文 token 的 K 做点积。**

所以每个 Q 分块都需要访问完整或相应范围内的 K。

---

### 12. Scale：除以 sqrt(d_k)

计算完：

```text
QK^T
```

之后，标准 Attention 还要除以：

```text
sqrt(d_k)
```

也就是：

```text
S = QK^T / sqrt(d_k)
```

这里的 `d_k` 是每个 head 的 key/query 维度。

为什么要除以 `sqrt(d_k)`？

因为如果维度很大，点积结果的方差会变大，Softmax 输入值可能过大，导致 Softmax 过于尖锐，训练不稳定。

这个 scale 操作是对矩阵中每个元素做除法，属于 element-wise 操作，不是大矩阵乘法。

---

### 13. 中间结果 S 会被写回 HBM

课程讲到，每个 SM 计算出自己负责的注意力分数块之后，会先写回 L2 Cache，再写回 HBM。

最后把多个 SM 计算出来的块拼起来，就得到完整的注意力分数矩阵：

```text
S_1, S_2
```

这里已经出现了标准 Attention 的第一个问题：

**S 是一个很大的中间矩阵，而且它会被写回 HBM。**

如果序列长度是 `N`，每个 head 的 `S` 是：

```text
N x N
```

当 `N` 很大时，这个矩阵会非常占显存。

---

### 14. 第三步：对 S 做 Softmax 得到 P

得到注意力分数矩阵 `S` 后，需要按行做 Softmax：

```text
P = softmax(S)
```

Softmax 是逐行执行的。每一行表示一个 token 对所有 token 的注意力分数。

对每一行：

```text
softmax(x_i) = exp(x_i - m) / sum_j exp(x_j - m)
```

其中：

```text
m = max(x_1, x_2, ..., x_n)
```

这里减去最大值 `m` 是为了数值稳定，防止 `exp(x)` 过大导致溢出。

---

### 15. Softmax 在 GPU 中涉及哪些计算单元

Softmax 不只是一个简单公式，它在 GPU 中会拆成几类操作：

| 操作 | 计算单元 |
|---|---|
| 求最大值 max | CUDA Core / reduction |
| 指数 `exp` | SFU |
| 求和 sum | CUDA Core / reduction |
| 除法 normalization | CUDA Core |

这说明 Softmax 不是 Tensor Core 最擅长的大矩阵乘法，而是很多 element-wise 和 reduction 操作。

因此 Softmax 往往更偏 memory-bound：

1. 需要读取整行数据；
2. 计算强度不如矩阵乘法高；
3. 中间结果可能还要写回 HBM；
4. 很容易被内存带宽限制。

---

### 16. P 矩阵：注意力权重矩阵

Softmax 计算完后得到：

```text
P_1 = softmax(S_1)
P_2 = softmax(S_2)
```

`P` 表示注意力权重。

例如 `P[i, j]` 表示：

```text
第 i 个 token 在聚合信息时，对第 j 个 token 分配了多少权重
```

课程中同样把 `P` 按行切给不同 SM 计算，最后再拼成完整的 `P_1` 和 `P_2`。

但是这里又出现一个问题：

**P 也是 N x N 的大矩阵，也会被写回 HBM。**

这就为 FlashAttention 的优化埋下伏笔。

---

### 17. 第四步：P 乘 V 得到每个 head 的输出

Attention 的下一步是：

```text
O = P V
```

对于两个 head：

```text
O_1 = P_1 V_1
O_2 = P_2 V_2
```

课程中仍然用 4 个 SM 并行处理：

1. SM1、SM2 处理第一个 head；
2. SM3、SM4 处理第二个 head；
3. 每个 SM 负责一部分 `P` 的行；
4. 但每个 SM 需要读取对应 head 的完整 `V`。

这是因为每个 token 的输出是对所有 token 的 value 做加权求和：

```text
O_i = sum_j P[i, j] V_j
```

所以每一行 `P_i` 都要和完整的 `V` 相乘。

---

### 18. 多头输出拼接

每个 head 都会得到自己的输出：

```text
O_1: N x d_head
O_2: N x d_head
```

然后把多个 head 在特征维度上拼接：

```text
Concat(O_1, O_2)
```

如果有 2 个 head，每个 head 维度是 2，那么拼接后维度就是 4。

课程例子中最终得到一个：

```text
8 x 4
```

的矩阵。

---

### 19. 输出投影 W_O：混合多个 head 的信息

多头输出拼接之后，还要乘一个输出投影矩阵：

```text
Y = Concat(O_1, O_2) W_O
```

为什么需要 `W_O`？

因为拼接后的不同 head 仍然是“分区明显”的，每个 head 的特征还没有充分混合。

`W_O` 的作用是：

1. 混合不同 head 的信息；
2. 保持输出维度和输入 hidden size 一致；
3. 为后续残差连接和 MLP 层提供统一形状。

如果拼接后的矩阵是：

```text
8 x 4
```

为了输出仍然是：

```text
8 x 4
```

那么 `W_O` 可以是：

```text
4 x 4
```

这一步也是矩阵乘法，主要由 Tensor Core 加速。

---

### 20. 课程最后指出的核心问题：反复读写 HBM

课程最后总结了一个非常重要的观察：

标准自注意力计算过程中，数据总是在：

```text
HBM -> L2 -> SM/SRAM -> 计算核心 -> SM/SRAM -> L2 -> HBM
```

之间来回流动。

尤其是中间矩阵：

```text
S = QK^T
P = softmax(S)
```

会被写回 HBM，然后又被后续步骤读出来。

这导致：

1. SM 内部的 Tensor Core / CUDA Core 可能算得很快；
2. 但它们经常要等数据从 HBM 搬过来；
3. HBM 读写速度成为瓶颈；
4. GPU 算力无法被充分利用。

这就是下一节 FlashAttention 的出发点：

**能不能不要把 S 和 P 这种大中间矩阵写回 HBM，而是在 SRAM 中边算边用？**

---

## 三、从输入到输出串联完整流程

下面把整节课讲到的内容串成一条完整技术主线。

### 1. 文本进入模型前：先变成 token id

输入文本：

```text
蒹葭苍苍，白露为霜
```

先经过 tokenizer：

```text
文本 -> token ids
```

每个 token id 是词表中的编号。

---

### 2. token id 经过 embedding 查表变成 X

模型不能直接处理 token id，需要把 token id 映射成向量：

```text
X = embedding[token_ids]
```

得到输入矩阵：

```text
X: N x hidden_size
```

---

### 3. X 和 W_QKV 从 HBM 进入 SM，计算 Q/K/V

输入矩阵 `X` 和权重矩阵 `W_QKV` 已经在 GPU HBM 中。

计算时：

```text
HBM -> L2 Cache -> SM SRAM -> Tensor Core
```

然后执行：

```text
QKV = X W_QKV
```

多个 SM 可以按 token 分块并行计算。

---

### 4. 拆分 Q/K/V 和不同 head

计算得到的大矩阵拆成：

```text
Q_1, K_1, V_1
Q_2, K_2, V_2
```

分别对应不同 attention head。

---

### 5. 每个 head 计算注意力分数 S

对每个 head：

```text
S_h = Q_h K_h^T / sqrt(d_k)
```

这一步是大矩阵乘法加 element-wise scale。

矩阵乘法由 Tensor Core 负责，scale 可由 CUDA Core 处理。

计算出的 `S_h` 会写回 HBM。

---

### 6. 对 S 做 Softmax 得到注意力权重 P

对每个 head：

```text
P_h = softmax(S_h)
```

Softmax 按行处理：

1. 求每行最大值；
2. 每个元素减最大值；
3. 做指数；
4. 求和；
5. 除以总和归一化。

这一步会用到 SFU 和 CUDA Core。

输出 `P_h` 也会写回 HBM。

---

### 7. P 乘 V 得到每个 head 的输出

对每个 head：

```text
O_h = P_h V_h
```

这又是矩阵乘法，主要由 Tensor Core 执行。

---

### 8. 拼接多个 head，并用 W_O 混合

多个 head 输出拼接：

```text
O = Concat(O_1, O_2, ..., O_h)
```

然后做输出投影：

```text
Y = O W_O
```

最终得到自注意力层输出。

---

### 9. 整体数据流总结

标准自注意力在 GPU 中的数据流大致是：

```text
文本
  ↓ tokenizer
token ids
  ↓ embedding 查表
X
  ↓ XW_QKV
Q/K/V
  ↓ QK^T / sqrt(d_k)
S 注意力分数
  ↓ softmax
P 注意力权重
  ↓ PV
每个 head 的输出
  ↓ concat
多头拼接结果
  ↓ W_O
最终 attention 输出
```

硬件层面对应：

```text
HBM
  ↓
L2 Cache
  ↓
SM 内部 SRAM
  ↓
Tensor Core / CUDA Core / SFU
  ↓
SRAM
  ↓
L2 Cache
  ↓
HBM
```

---

## 四、这节课和 FlashAttention 的关系

这节课其实是 FlashAttention 的铺垫。

标准 Attention 的执行方式中，最值得注意的是：

```text
QK^T 生成 S
  ↓
S 写回 HBM
  ↓
读 S 做 Softmax
  ↓
P 写回 HBM
  ↓
读 P 和 V 做 PV
```

问题在于：

1. `S` 是 `N x N`；
2. `P` 也是 `N x N`；
3. 序列越长，中间矩阵越大；
4. HBM 读写越多；
5. SM 计算单元越容易等待数据。

FlashAttention 的核心就是改写这个流程：

```text
不要完整生成 S/P
不要把 S/P 写回 HBM
把 Q/K/V 分块放入 SRAM
在 SRAM 中完成局部 QK^T、softmax、PV
只写回最终 O
```

所以这节课你应该和上一节 FlashAttention 结合起来理解：

| 本节课 | FlashAttention |
|---|---|
| 解释标准 Attention 在 GPU 中如何执行 | 解释如何优化这个执行过程 |
| 重点看到 S/P 被写回 HBM | 重点避免 S/P 写回 HBM |
| 展示 HBM 读写拖慢 SM | 用 tiling/fusion 减少 HBM IO |
| 讲 GPU 内存层级 | 利用内存层级进行 IO-aware 优化 |

---

## 五、与 AI Infra / 推理系统的关系

这节课虽然讲的是底层 GPU 执行过程，但和大模型推理系统非常相关。

### 1. 推理性能不只看模型结构，还要看硬件数据流

同一个 Attention 公式，不同 kernel 实现性能可能差很多。

原因在于：

1. 数据是否重复从 HBM 读取；
2. 中间结果是否写回 HBM；
3. 是否充分利用 Tensor Core；
4. 是否让多个 SM 高效并行；
5. 是否减少 kernel launch 和同步开销；
6. 是否利用 L2 Cache / shared memory 的数据复用。

AI Infra 推理优化的很多工作，本质上就是围绕这些问题展开。

---

### 2. Prefill 阶段为什么重视 Attention kernel

在大模型推理中：

```text
Prefill: 一次处理完整 prompt
Decode: 每次生成一个新 token
```

Prefill 阶段需要对 prompt 中全部 token 做 Attention。

如果 prompt 很长，`QK^T` 对应的注意力矩阵会很大。

因此 prefill 的性能很依赖高效 Attention kernel，例如 FlashAttention。

这会直接影响：

```text
TTFT = Time To First Token
```

也就是用户等第一个 token 出来的时间。

---

### 3. Decode 阶段为什么重视 KV Cache 和显存带宽

Decode 阶段每次只生成一个 token。

新 token 的 Q 需要和历史所有 token 的 K/V 计算 attention。

历史 K/V 保存在 KV Cache 中。

所以 decode 阶段常见瓶颈包括：

1. 读取 KV Cache 的 HBM 带宽；
2. 多请求 batch 下 KV Cache 的管理；
3. 不同请求长度不同导致的调度问题；
4. kernel launch 和小矩阵计算效率问题。

这就是为什么 vLLM 会提出 PagedAttention。

---

### 4. 为什么要理解 SM、HBM、L2、Tensor Core

如果你准备 AI Infra 面试，只会说公式是不够的。

面试官可能会追问：

1. 为什么 Attention 慢？
2. 慢在计算还是显存读写？
3. FlashAttention 为什么快？
4. Tensor Core 主要加速哪类操作？
5. Softmax 为什么不是 Tensor Core 的优势场景？
6. KV Cache 为什么容易带来显存带宽瓶颈？

理解这节课，就能把这些问题串起来。

---

## 六、关键概念表

| 概念 | 解释 | 在本节课中的作用 |
|---|---|---|
| GPU | 图形处理器，大规模并行计算设备 | 执行 Transformer 计算 |
| HBM | GPU 显存，容量大但访问慢于片上存储 | 存模型权重、输入、Q/K/V、中间矩阵 |
| L2 Cache | GPU 多个 SM 共享的缓存 | 减少重复访问 HBM |
| SM | Streaming Multiprocessor，GPU 基本计算单元 | 并行处理矩阵块、token 块、head |
| SRAM | 课程中对 SM 内高速存储的简化称呼 | 存放当前计算块 |
| Register | 寄存器，离计算最近 | 存当前线程数据 |
| Shared Memory | 同一个 SM 内线程块共享的高速存储 | kernel 优化常用 |
| CUDA Core | 普通计算核心 | 加减乘除、归约等 |
| Tensor Core | 矩阵乘法加速单元 | 加速 QKV、QK^T、PV、W_O |
| SFU | Special Function Unit | 计算 exp、log 等特殊函数 |
| PCIe | CPU 和 GPU 间通信通道 | 把数据从 CPU 内存传到 GPU |
| NVLink | GPU 和 GPU 间高速互联 | 多卡训练/推理通信 |
| Tokenizer | 文本转 token id | 模型输入第一步 |
| Embedding | token id 查表成向量 | 得到输入矩阵 X |
| Q/K/V | 自注意力的 query/key/value | 计算注意力分数与输出 |
| S | `QK^T / sqrt(d_k)` | 注意力分数矩阵 |
| P | `softmax(S)` | 注意力权重矩阵 |
| W_O | 输出投影矩阵 | 混合多头输出 |

---

## 七、面试常见问题与回答

### 1. GPU 中的 SM 是什么？

SM 是 Streaming Multiprocessor，流式多处理器，是 NVIDIA GPU 中的基本计算单元。

一个 GPU 由很多个 SM 组成。每个 SM 内部有 CUDA Core、Tensor Core、寄存器、shared memory、L1 cache 等资源，可以并行执行大量线程。

在 Transformer 计算中，不同 token、不同 head、不同矩阵块都可以被分配到不同 SM 上并行处理。

---

### 2. HBM、L2 Cache、SRAM 的区别是什么？

HBM 是 GPU 显存，容量最大，但相对访问最慢。

L2 Cache 是 GPU 芯片上的共享缓存，多个 SM 可以共享，速度比 HBM 快。

SRAM 在本节课中泛指 SM 内部更靠近计算核心的高速存储，例如 L1 cache、shared memory、register，容量小但速度快。

GPU 优化的核心之一就是尽量减少 HBM 访问，让数据尽可能在 L2 和 SM 内部复用。

---

### 3. Tensor Core 和 CUDA Core 有什么区别？

Tensor Core 专门用于加速矩阵乘法，特别适合深度学习中的 GEMM，比如：

```text
XW_QKV
QK^T
PV
OW_O
```

CUDA Core 更通用，可以执行普通加减乘除、比较、归约等操作。

Softmax 中的求最大值、求和、除法等更多依赖 CUDA Core 和 SFU。

---

### 4. SFU 是做什么的？

SFU 是 Special Function Unit，用于计算特殊数学函数，例如：

1. exp；
2. log；
3. sin/cos；
4. reciprocal 等。

在 Attention 中，Softmax 需要计算 `exp(x)`，所以会用到 SFU。

---

### 5. 自注意力的完整计算流程是什么？

标准自注意力流程是：

```text
X -> Q/K/V
S = QK^T / sqrt(d_k)
P = softmax(S)
O = PV
Y = Concat(heads) W_O
```

其中 `S` 是注意力分数，`P` 是注意力权重，`O` 是每个 head 的输出，最后通过 `W_O` 混合多个 head 的信息。

---

### 6. 为什么 Q/K/V 通常会合并计算？

因为：

```text
Q = XW_Q
K = XW_K
V = XW_V
```

这三次矩阵乘法输入都是同一个 `X`。

把 `W_Q`、`W_K`、`W_V` 拼成一个大矩阵：

```text
W_QKV = [W_Q, W_K, W_V]
```

就可以一次矩阵乘法得到 Q/K/V，减少 kernel 调用和数据读写开销。

---

### 7. 多头注意力为什么适合并行？

因为不同 head 的 Q/K/V 和 attention 计算相对独立。

例如两个 head 可以分别计算：

```text
S_1 = Q_1K_1^T
S_2 = Q_2K_2^T
```

不同 head 可以分配给不同 SM 或不同线程块执行。同一个 head 内部也可以按 token 行或矩阵块继续并行。

---

### 8. 为什么计算 QK^T 时每个 Q block 需要访问完整 K？

因为每个 token 都要和上下文中的所有 token 计算注意力分数。

对于某个 query token：

```text
score_i,j = q_i · k_j
```

这里的 `j` 会遍历所有 key token。

所以即使 Q 可以按行切分，每个 Q 分块仍然需要读取完整或对应上下文范围内的 K。

---

### 9. Softmax 为什么要减去最大值？

为了数值稳定。

如果直接计算：

```text
exp(x_i)
```

当 `x_i` 很大时，FP16/FP32 都可能出现溢出。

减去最大值：

```text
softmax(x_i) = exp(x_i - m) / sum_j exp(x_j - m)
```

不会改变 Softmax 结果，但可以保证指数项不会过大。

---

### 10. 标准 Attention 为什么会产生大量 HBM 读写？

因为它通常会显式生成中间矩阵：

```text
S = QK^T
P = softmax(S)
```

这些矩阵会被写回 HBM，后续计算又从 HBM 读出。

当序列长度为 `N` 时，`S` 和 `P` 都是 `N x N`，长序列下读写量非常大。

---

### 11. 为什么 Attention 容易成为 memory-bound？

Attention 中虽然有矩阵乘法，但还包含大量中间矩阵读写，以及 Softmax 这类 element-wise/reduction 操作。

如果 SM 内的计算单元已经算得很快，但数据从 HBM 搬运不过来，计算单元就会等待数据。

这时瓶颈不是 FLOPs，而是显存带宽，也就是 memory-bound。

---

### 12. 这节课如何引出 FlashAttention？

这节课展示了标准 Attention 的问题：

```text
S 写回 HBM
P 写回 HBM
后续又从 HBM 读出来
```

FlashAttention 正是为了解决这个问题：

1. 把 Q/K/V 分块；
2. 在 SRAM 中计算局部 `QK^T`；
3. 在 SRAM 中做 online softmax；
4. 直接乘 V 更新输出；
5. 不把完整 S/P 写回 HBM。

所以 FlashAttention 本质上是对标准 Attention 数据流的重排和优化。

---

### 13. 为什么说理解硬件有助于理解推理框架？

因为推理框架优化不是只写 Python 调度逻辑，还要理解 GPU kernel 的性能瓶颈。

例如：

1. prefill 阶段依赖高效 attention kernel；
2. decode 阶段依赖高效 KV cache 读取；
3. continuous batching 需要减少空转和提高 GPU 利用率；
4. tensor parallel 涉及多卡通信和 NVLink；
5. quantization 会改变显存带宽和计算吞吐的平衡。

这些都离不开对 GPU 内存层级和计算单元的理解。

---

### 14. W_O 的作用是什么？

多头 Attention 得到多个 head 的输出后，会先拼接。

但拼接只是把不同 head 的结果放在一起，并没有真正混合它们的信息。

`W_O` 是输出投影矩阵，用于：

1. 混合多个 head 的信息；
2. 保持输出 hidden size 不变；
3. 方便后续残差连接和 MLP 处理。

---

### 15. PCIe 和 NVLink 有什么区别？

PCIe 通常用于 CPU 和 GPU 之间通信。

NVLink 通常用于 GPU 和 GPU 之间高速通信。

在多卡训练或推理中，如果模型被切到多张 GPU 上，GPU 之间需要频繁交换数据。NVLink 的带宽和延迟通常优于 PCIe，因此对大模型并行计算很重要。

---

## 八、复习检查题

1. SM 是什么？它在 GPU 中承担什么角色？
2. HBM、L2 Cache、shared memory、register 的速度和容量有什么区别？
3. Tensor Core 主要加速什么操作？
4. Softmax 为什么会用到 SFU？
5. 文本如何变成模型可以处理的输入矩阵 X？
6. 为什么 Q/K/V 可以通过一次 `XW_QKV` 得到？
7. 多头注意力中不同 head 的 Q/K/V 有什么区别？
8. 为什么 `QK^T` 的结果是注意力分数矩阵？
9. 为什么要除以 `sqrt(d_k)`？
10. Softmax 为什么按行计算？
11. 为什么 S 和 P 都是 `N x N`？
12. 为什么标准 Attention 会频繁读写 HBM？
13. 为什么 HBM 读写慢会导致 SM 空等？
14. FlashAttention 针对这节课暴露出的哪个问题进行优化？
15. 在推理系统中，prefill 和 decode 对 Attention 优化的关注点有什么不同？

---

## 九、最重要的记忆主线

这节课最重要的不是背硬件名词，而是记住这条链路：

```text
文本进入模型
  ↓
Tokenizer 得到 token id
  ↓
Embedding 查表得到 X
  ↓
X 和 W_QKV 从 HBM 读入 SM
  ↓
Tensor Core 计算 Q/K/V
  ↓
Tensor Core 计算 QK^T 得到 S
  ↓
S 写回 HBM
  ↓
读取 S，用 SFU/CUDA Core 做 Softmax 得到 P
  ↓
P 写回 HBM
  ↓
读取 P 和 V，用 Tensor Core 计算 PV
  ↓
拼接多个 head，乘 W_O
  ↓
得到 Attention 输出
```

这条链路暴露出标准 Attention 的核心问题：

**中间矩阵 S 和 P 太大，而且会反复写入和读取 HBM，导致 GPU 计算单元经常等待数据。**

这就是为什么后面要学习 FlashAttention：

**FlashAttention 的本质，就是尽量让 Attention 的中间计算停留在 SM 内部高速存储中，减少 HBM 读写，从而提升速度并节省显存。**

---

## 十、一句话总结

标准自注意力在 GPU 上不是“公式一写就直接算完”，而是经历了从 HBM 到 L2、再到 SM 内部 SRAM、再到 Tensor Core/CUDA Core/SFU 的复杂数据流；理解这条数据流，就能理解为什么 Attention 会被 HBM 读写拖慢，也能自然理解 FlashAttention 为什么要通过分块和融合来减少中间矩阵落显存。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
