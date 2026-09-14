# vLLM 分块显存管理与 PagedAttention 学习笔记

> 课程主题：vLLM 中 KV Cache 的分页 / 分块显存管理  
> 适用方向：AI Infra / 大模型推理 / 推理引擎 / vLLM / nano-vLLM / Transformer 推理优化  
> 核心关键词：KV Cache、block_size、PagedAttention、显存碎片、吞吐量、batch size、prefill、decode、block table

---

## 0. 本节课转写中的术语修正

这份视频转文字是自动识别生成的，里面有一些技术词被识别错了。阅读时建议按下面方式理解：

| 转写中的词 | 正确理解 | 说明 |
|---|---|---|
| VLM / VLLM | vLLM | 这里讲的是大模型推理框架 vLLM，不是视觉语言模型 VLM。 |
| KVCatch / KVKG | KV Cache | Transformer 推理中缓存历史 token 的 Key 和 Value。 |
| BlogSythe / BLOGSIDE / BlogSize | block_size | 每个 KV Cache block 能容纳多少个 token 的 KV。 |
| Promote | prompt | 用户输入的原始文本。 |
| 偷肯 | token | tokenizer 切分后的基本单位。 |
| 形存 / 线存 / 战用显存 | 显存 / 占用显存 | GPU memory。 |
| PJITENSION / ITENSION | PagedAttention / Attention | vLLM 的核心机制：分页式 KV 管理 + 支持分页读取的 Attention。 |
| MexCquinLin | max_seq_len | 模型或服务允许的最大序列长度。 |
| NumberAdd | num_heads / num_kv_heads | 注意力头数量或 KV 头数量。 |
| App16 | FP16 | 半精度浮点格式。 |
| 中止符 | EOS token | 模型生成结束标志。 |

---

# 一、总体阐述：这节课到底想讲什么

本节课的核心目标是：**解释 vLLM 为什么要把 KV Cache 从“连续大块显存”改成“固定大小 block 的分页式显存管理”，以及这种机制如何提升推理系统的显存利用率和批处理能力。**

在普通 Transformer 推理中，每个请求都需要保存历史 token 的 Key/Value，也就是 KV Cache。问题在于，如果系统一开始按照 `max_seq_len` 预分配 KV Cache，比如一个请求最多可能生成 4096 个 token，那么即使这个请求实际只用了 10 个 token，也可能长期占着 4096 token 对应的显存空间。这会造成大量浪费，并压缩同时服务的请求数量。

vLLM 的思路是：**不要一次性给每个请求分配最长序列的连续 KV Cache，而是把 GPU 显存切成很多固定大小的 block，用多少分多少。** 这和操作系统中的分页内存管理很像：逻辑上一个序列是连续增长的，但物理显存上可以分散存放，只要有一张映射表把“逻辑 block”映射到“物理 block”即可。

这节课在 AI Infra 学习路线中的位置非常关键。它不是在讲 Transformer 的数学公式，而是在讲**大模型推理系统如何把 Transformer 放到真实 GPU 服务中高效运行**。如果你后面继续学 vLLM、nano-vLLM、KV Cache、continuous batching、scheduler、显存优化、prefix caching、speculative decoding，这节课的“分块 KV Cache 管理”都是底层基础。

一句话总结：

> **Transformer 解释了模型怎么算，vLLM 解释了很多请求同时来时，KV Cache 怎么存、怎么复用、怎么不浪费显存。**

---

# 二、按课程推进顺序梳理知识点

## 1. 课程开头：vLLM 的一个核心特点是分块显存管理

### 核心知识点

课程一开始指出，vLLM 的明显特点之一是引入了 **分块显存管理**。这里的分块管理主要针对的是 **KV Cache**，不是模型权重。

### 概念解释

在 Transformer 自回归推理中，模型每生成一个 token，都需要用注意力机制回看之前所有 token。为了避免每一步都重新计算历史 token 的 K 和 V，推理系统会把历史 token 的 K/V 保存下来，这就是 KV Cache。

对于一个 token，在每一层 Transformer 中都会产生对应的 K 和 V。对于长上下文、多层模型、多头注意力来说，KV Cache 会占用非常多显存。

### 在推理系统中的作用

KV Cache 是 LLM 推理服务中最重要的运行时显存开销之一。模型权重一般是固定占用，但 KV Cache 会随着：

- 请求数量增加；
- prompt 变长；
- 输出 token 变长；
- 并发 batch 变大；

而动态增长。

所以推理框架的显存管理能力，直接决定了它能同时服务多少用户、吞吐量有多高、长上下文支持得好不好。

---

## 2. 传统 KV Cache 的问题：按最大长度预分配

### 核心知识点

传统方式通常按 `max_seq_len` 给每个请求预留 KV Cache 空间。截图中给出的传统代码大致是：

```python
self.cache_k = torch.zeros((
    args.max_batch_size,
    args.max_seq_len,
    self.n_local_kv_heads,
    self.head_dim,
)).cuda()
```

这里最关键的是第二维：`args.max_seq_len`。

### 概念解释

假设系统允许一个对话最长 4096 token，那么传统方式可能会为每个 batch 位置预留 4096 token 的 KV Cache。即使真实请求只生成了十几个 token，也仍然可能占住完整的最大长度空间。

也就是说：

```text
理论最大长度：4096 token
实际使用长度：10 token
浪费空间：4096 - 10 = 4086 token 对应的 KV Cache
```

课程里强调的不是普通文本 token 本身占空间，而是**每个 token 在每一层、每个 KV head、每个 head_dim 上对应的 K/V 张量占空间**。

### 在推理系统中的作用

这种预分配方式的缺点是：

1. **显存占用大**：每个请求都按最大长度占空间；
2. **内部碎片严重**：已分配但未使用的位置不能给别人用；
3. **批处理能力下降**：显存被浪费后，能放进 GPU 的请求数减少；
4. **吞吐量下降**：batch size 变小后，GPU 并行能力无法充分发挥。

所以这不是单纯的“内存浪费”问题，而是会直接影响推理服务的核心指标：吞吐量、并发数、TTFT、TPOT 和服务稳定性。

---

## 3. 为什么会影响 batch size 和吞吐量

### 核心知识点

课程中提到：如果一个请求占用了 4096 token 的 KV Cache，即使它现在只用到 0～10 token，后面 11～4096 的位置也不能分配给其他请求。因此实际可并行处理的 batch size 会被压缩。

### 概念解释

batch size 在推理服务中不是简单地等于“请求数量”。它受多个因素限制：

- 模型权重占用；
- KV Cache 占用；
- 临时激活值 / workspace 占用；
- attention kernel 的计算限制；
- scheduler 当前能放入多少 token。

当 KV Cache 被过度预留时，GPU 看似还有计算能力，但显存已经被占满，新的请求无法进入 batch。

### 在推理系统中的作用

这就是为什么 vLLM 要重点优化 KV Cache，而不是只优化算子速度。真实线上推理中，很多时候瓶颈不只是“算得慢”，而是：

> **显存被 KV Cache 占住，batch 做不大，GPU 利用率上不去。**

---

## 4. vLLM 的解决思路：把连续 KV Cache 切成固定大小的 block

### 核心知识点

vLLM 不再为每个请求一次性分配一整条连续 KV Cache，而是把可用 KV 显存划分成很多固定大小的 block。每个 block 可以存放固定数量 token 的 KV Cache。

例如课程中为了便于理解，假设：

```text
block_size = 16
```

这表示：

```text
一个 block 可以容纳 16 个 token 对应的 KV Cache
```

### 概念解释

如果一个请求有 40 个 token，那么按 `block_size = 16` 来看，需要：

```text
第 1 个 block：token 0 ~ 15
第 2 个 block：token 16 ~ 31
第 3 个 block：token 32 ~ 39，有 8 个位置被使用，剩下 8 个位置暂时浪费
```

也就是说，请求需要多少 block，系统就分配多少 block。

### 在推理系统中的作用

这实现了课程中反复强调的思想：

> **吃多少拿多少，拿多少用多少。**

它的好处是：

- 不再按最大长度一次性占满；
- 短请求不会浪费长上下文空间；
- 长请求可以随着生成过程逐步申请新 block；
- 显存利用率提升；
- 能容纳更多并发请求；
- 更适合 continuous batching。

---

## 5. 分块方案仍然有浪费，但浪费上界很小

### 核心知识点

分块管理不是完全零浪费，因为最后一个 block 可能没有填满。但浪费最多是：

```text
block_size - 1 个 token 对应的 KV Cache
```

如果 `block_size = 16`，那么一个请求最后最多浪费 15 个 token 的 KV Cache 空间。

### 概念解释

假设一个请求长度是 17：

```text
第 1 个 block：16 个 token，完全用满
第 2 个 block：1 个 token，只用 1 个位置，浪费 15 个位置
```

虽然仍然有内部碎片，但和传统预分配 4096 个位置相比，浪费已经小得多。

### 在推理系统中的作用

这就是分块 KV Cache 的工程价值：

```text
传统方式最大浪费：max_seq_len - 实际长度
分块方式最大浪费：block_size - 1
```

只要 `block_size` 选得合理，就能在管理开销和显存浪费之间取得平衡。

---

## 6. PagedAttention：不只是“分块存”，还要“能分块算”

### 核心知识点

课程中特别强调：**分块显存管理本身只是 PagedAttention 的一部分。** 完整的 PagedAttention 还包括一种 attention 计算方式，使模型在计算注意力时能够从这些不连续的 block 中读取 KV Cache。

### 概念解释

传统 attention 假设一个序列的 K/V 在显存中是连续排列的。但分块管理后，一个请求的逻辑 token 序列可能分散在多个物理 block 中。

例如：

```text
逻辑上：block id1 -> block id2 -> block id3
物理上：mem1     -> mem5     -> mem2
```

这就需要一张映射表，告诉 attention kernel：

```text
请求的第 0 个逻辑 block 在哪个物理 block？
请求的第 1 个逻辑 block 在哪个物理 block？
请求的第 2 个逻辑 block 在哪个物理 block？
```

这个映射关系通常可以理解为 **block table**。

### 在推理系统中的作用

PagedAttention 的关键是：

1. **存储层面**：KV Cache 可以被切成多个 block；
2. **计算层面**：Attention kernel 能根据 block table 找到这些 block；
3. **调度层面**：请求进入、继续生成、结束时，block 可以动态申请和释放。

所以 PagedAttention 不是一个单独的数学 attention 公式，而是推理系统中的一整套 KV Cache 管理与读取机制。

---

## 7. prompt 如何被切分成 block

### 核心知识点

课程后半部分用一个 prompt 举例：

```text
prompt: hello, everybody from xhs, where are you?
```

为了便于演示，课程假设：

```text
block_size = 4
```

于是 prompt 被切成多个逻辑 block：

```text
block id1：前 4 个 token
block id2：接下来 4 个 token
block id3：再接下来 4 个 token
...
```

### 概念解释

这里要特别注意：真实系统里 block 中存的不是英文单词字符串，而是 tokenizer 编码后的 **token ids**。

例如课程中随手举了类似这样的数字：

```text
hello, every body -> 11, 13, 41, 56
from xhs, ...     -> 22, 23, 31, 44
```

这些数字只是示意，真实 token id 由模型的 tokenizer 词表决定。不同模型的 tokenizer 不同，同一句话切出来的 token id 也可能不同。

### 在推理系统中的作用

对于推理框架来说，文本只是入口，真正进入模型的是 token ids。KV Cache 也是围绕 token ids 对应的模型中间结果来组织的。

完整链路是：

```text
用户文本 prompt
    ↓ tokenizer.encode
input token ids
    ↓ 按 block_size 切分
logical blocks
    ↓ 申请物理 KV blocks
physical memory blocks
```

---

## 8. 每个 block 存的到底是什么

### 核心知识点

每个 block 存放的是一组 token 对应的 KV Cache，而不是 token id 本身。token id 只是模型输入，KV Cache 是模型 forward 过程中计算出来的 Key 和 Value。

课程中给出的概念公式可以理解为：

```text
一个 block 的 KV Cache 空间
≈ block_size × 2 × head_dim × num_heads × dtype_size
```

其中：

- `block_size`：这个 block 能放多少个 token；
- `2`：K 和 V 两份缓存；
- `head_dim`：每个注意力头的维度；
- `num_heads` 或 `num_kv_heads`：KV 头数量；
- `dtype_size`：每个元素占多少字节，例如 FP16 是 2 bytes。

更完整地看，如果考虑多层 Transformer，还要乘以层数：

```text
总 KV Cache 大小
≈ num_layers × token_count × 2 × num_kv_heads × head_dim × dtype_size
```

### 概念解释

需要区分三类东西：

| 名称 | 是什么 | 是否直接存进 KV Cache block |
|---|---|---|
| 原始文本 | 用户输入的字符串 | 否 |
| token ids | tokenizer 编码后的整数序列 | 否，作为模型输入使用 |
| KV Cache | 模型每层 attention 计算出的 K/V 张量 | 是 |

所以课程中“block 里放 token id”的说法是为了帮助理解逻辑切分；严格来说，物理显存 block 里保存的是这些 token 对应的 K/V 张量。

### 在推理系统中的作用

这个公式非常重要，因为它让你能估算显存：

```text
KV Cache 越大，能同时服务的请求越少；
block_size 越大，最后一个 block 的浪费可能越大；
dtype 越小，KV Cache 占用越少；
num_kv_heads 越小，KV Cache 占用越少。
```

这也解释了为什么 GQA / MQA、KV Cache 量化、PagedAttention 都是推理系统优化的核心方向。

---

## 9. 逻辑 block id 与物理 mem block 的关系

### 核心知识点

截图中左侧画了 `block id1、block id2、block id3...` 和上方 `mem1、mem2...` 的映射关系。这里要理解两层含义：

```text
logical block id：某个请求内部的第几个 block
physical mem block：GPU 上实际分配到的 KV Cache 显存块
```

### 概念解释

一个请求逻辑上有连续的 block：

```text
request A:
logical block 0 -> logical block 1 -> logical block 2
```

但物理显存可能不是连续的：

```text
logical block 0 -> physical mem 1
logical block 1 -> physical mem 4
logical block 2 -> physical mem 2
```

推理系统需要记录这种映射关系。这个结构就是理解 vLLM 源码里 block table、block manager、sequence、scheduler 的基础。

### 在推理系统中的作用

这个映射带来的好处是：

- 物理显存不要求连续；
- 任意空闲 block 都可以分给新请求；
- 请求结束后可以释放它占用的 block；
- 共享前缀时，多个请求可以指向相同的 block；
- speculative decoding 或 beam search 等场景下更容易复用前缀 KV。

---

## 10. 请求继续生成时如何申请新 block

### 核心知识点

课程最后用 “I am from Zhejiang” 举例说明：如果 prompt 已经处理完，模型继续生成回答，当新 token 增长到需要更多空间时，系统会再申请新的 block。

### 概念解释

推理过程分两个阶段理解最清楚：

#### 1）Prefill 阶段

模型一次性处理用户输入 prompt 的所有 token，计算出这些 token 的 KV Cache，并把它们写入对应 block。

```text
prompt token ids
    ↓ model forward
prompt 对应的 K/V
    ↓ 写入 block1、block2、block3...
```

#### 2）Decode 阶段

模型每次生成一个新 token。每生成一个 token，就需要：

```text
新 token 进入模型
    ↓ 计算这个 token 的 K/V
    ↓ 追加写入当前 block
    ↓ 如果当前 block 满了，就申请新 block
    ↓ attention 读取历史所有 block 中的 KV
    ↓ 生成下一个 token
```

### 在推理系统中的作用

这就是 vLLM 支持长文本连续生成和高并发服务的基础：请求不是一开始就占满最大长度，而是随着 decode 逐步扩容。请求结束后，比如生成 EOS，系统就可以释放这个请求的 block，把它们还给全局 block 池。

---

# 三、整节课的技术主线：从输入到输出完整串联

下面把整节课串成一条完整流程。

## 1. 用户输入 prompt

用户输入一句话：

```text
hello, everybody from xhs, where are you?
```

这句话是自然语言文本，模型不能直接处理。

---

## 2. tokenizer 把文本转成 token ids

模型先用 tokenizer 编码：

```text
prompt string
    ↓ tokenizer.encode
[11, 13, 41, 56, 22, 23, 31, 44, ...]
```

这里的数字只是示意。真实 token id 由模型词表决定。

---

## 3. 推理框架按 block_size 对 token ids 分组

假设：

```text
block_size = 4
```

那么 token ids 会被逻辑切分成：

```text
logical block 0：[11, 13, 41, 56]
logical block 1：[22, 23, 31, 44]
logical block 2：[...]
```

这些 logical block 只是请求内部的逻辑顺序，不代表 GPU 物理显存连续。

---

## 4. Block Manager 从空闲池中分配物理 KV block

vLLM 会维护一批空闲物理 block：

```text
free blocks: mem1, mem2, mem3, mem4, ...
```

当一个请求需要 3 个 block 时，系统就分配 3 个物理 block，并建立映射：

```text
request A logical block 0 -> mem1
request A logical block 1 -> mem2
request A logical block 2 -> mem3
```

如果物理显存中 `mem2` 被其他请求占用，也没关系，可以映射到 `mem5`：

```text
request A logical block 1 -> mem5
```

关键是：**逻辑连续，物理可以不连续。**

---

## 5. Prefill 计算 prompt 的 KV Cache

模型对 prompt token ids 做 forward。每一层 attention 都会产生这些 token 的 K/V。

```text
input token ids
    ↓ embedding + position
hidden states
    ↓ attention projection
K/V tensors
    ↓ 写入对应 physical block
```

写入后，系统不再只保存 token id，而是保存这些 token 对应的 KV Cache。

---

## 6. Attention 通过 block table 读取历史 KV

在后续计算中，attention 需要读取这个请求前面所有 token 的 K/V。由于 K/V 分散在不同物理 block 中，所以不能简单按连续地址读，而要通过 block table 找到每个 logical block 对应的 physical block。

```text
当前请求的 block table
logical block 0 -> mem1
logical block 1 -> mem5
logical block 2 -> mem3
```

PagedAttention kernel 根据这张表去读取 K/V，然后完成注意力计算。

---

## 7. Decode 阶段逐 token 生成

当模型开始生成回答时，每一步只生成一个 token：

```text
step 1: 生成 token A
step 2: 生成 token B
step 3: 生成 token C
...
```

每生成一个 token，就要把这个 token 对应的新 K/V 写入当前 block。如果当前 block 已满，就申请新的 physical block。

```text
当前 block 未满：直接追加 K/V
当前 block 已满：allocate new block -> 再写入 K/V
```

---

## 8. 请求结束后释放 block

当模型生成 EOS token，或者达到最大输出长度，或者用户中断请求时，这个 sequence 结束。推理系统会释放它占用的 physical blocks。

```text
request finished
    ↓ free physical blocks
    ↓ blocks 回到空闲池
    ↓ 可分配给其他请求
```

这一步对线上服务很重要，因为请求会不断进入和退出。高效释放与复用 block，才能支持持续高并发。

---

# 四、这节课和 vLLM / nano-vLLM 源码的对应关系

如果后面阅读 nano-vLLM 或 vLLM 源码，可以这样对应：

| 课程概念 | 源码中可能对应的模块 | 作用 |
|---|---|---|
| 请求 / prompt | Sequence / Request | 表示一个用户请求及其 token 状态。 |
| token ids | tokenizer 输出、Sequence.token_ids | 模型真正处理的输入。 |
| block_size | config 中的 block size | 每个 KV block 容纳的 token 数。 |
| logical block | Sequence 内部的 block 编号 | 描述请求自己的第几个 block。 |
| physical block / mem | KV Cache block pool | GPU 上真实的 KV Cache 存储块。 |
| block table | block mapping table | 记录逻辑 block 到物理 block 的映射。 |
| Block Manager | block_manager.py | 负责申请、释放、复用 block。 |
| Scheduler | scheduler.py | 决定哪些请求进入本轮 prefill/decode。 |
| Model Runner | model_runner.py | 负责实际执行模型 forward。 |
| PagedAttention | attention kernel | 根据 block table 读取离散 KV blocks 并计算 attention。 |

对于你正在学的 nano-vLLM，可以重点看这条主线：

```text
LLM.generate
  -> Scheduler 选择请求
  -> BlockManager 分配 KV blocks
  -> ModelRunner 执行 prefill/decode
  -> Attention 使用 block table 读取 KV Cache
  -> 生成 token
  -> 请求结束后释放 block
```

---

# 五、关键概念深度解释

## 1. KV Cache 为什么这么占显存

对每个 token，每一层 Transformer 都要保存 K 和 V。假设：

```text
num_layers = 32
num_kv_heads = 32
head_dim = 128
dtype = FP16 = 2 bytes
token_count = 4096
```

那么单个请求的 KV Cache 量级大约是：

```text
32 × 4096 × 2 × 32 × 128 × 2 bytes
≈ 2 GB
```

这只是一个量级估算，真实大小还受 GQA、张量并行、实现布局等影响。但它能说明：长上下文场景下，KV Cache 不是小开销。

---

## 2. block_size 不是 batch size

这两个概念很容易混淆：

| 概念 | 含义 |
|---|---|
| block_size | 一个 KV Cache block 可以存多少个 token。 |
| batch size | 一轮推理里同时处理多少请求或多少 token。 |

举例：

```text
block_size = 16
```

表示每个 KV block 存 16 个 token 的 K/V。

而 batch size 可能是：

```text
本轮同时处理 8 个请求
或本轮 prefill 总共处理 2048 个 token
或本轮 decode 处理 64 个 token
```

它们不是同一个维度。

---

## 3. block id 不是 token id

| 概念 | 示例 | 含义 |
|---|---|---|
| token id | 11、13、41、56 | tokenizer 输出的词表编号。 |
| logical block id | block id1、block id2 | 一个请求内部第几个 KV block。 |
| physical block id | mem1、mem2 | GPU 上真实的 KV Cache block。 |

token id 是模型输入；block id 是显存管理单位。

---

## 4. PagedAttention 和操作系统分页有什么相似点

PagedAttention 的思想和操作系统分页内存管理非常像：

| 操作系统分页 | vLLM PagedAttention |
|---|---|
| 虚拟页 virtual page | logical KV block |
| 物理页 physical page | physical KV block |
| 页表 page table | block table |
| 页面分配 / 回收 | KV block allocate / free |
| 内存碎片优化 | KV Cache 显存碎片优化 |

区别在于：操作系统分页管理的是 CPU 内存虚拟地址，vLLM 管理的是 GPU 上的 KV Cache，并且 attention kernel 要能利用这张映射表完成计算。

---

## 5. 为什么它能提升吞吐量

吞吐量提升的原因不是“单个 token 的数学计算变少了”，而是：

```text
显存浪费减少
    ↓
同一张 GPU 能放下更多请求 / 更多 token
    ↓
batch 更大
    ↓
GPU 并行度更高
    ↓
单位时间生成更多 token
```

所以 PagedAttention 的收益主要来自系统层面的显存利用率和调度效率提升。

---

# 六、学习者最容易误解的点

## 误解 1：block 里存的是 token id

不准确。token id 是输入，block 里真正存的是模型算出来的 K/V 张量。课程中把 token id 写进 block，是为了帮助理解“哪些 token 属于哪个 block”。

## 误解 2：PagedAttention 只是把显存切块

不完整。切块只是存储管理，还需要 attention kernel 能根据 block table 读取不连续的 KV Cache。

## 误解 3：分块后完全没有浪费

不对。最后一个 block 可能没填满，所以最多浪费 `block_size - 1` 个 token 的 KV 空间。

## 误解 4：block_size 越小越好

不一定。block_size 小可以减少内部碎片，但会增加 block table 规模、管理开销和 kernel 访问复杂度。工程中要折中。

## 误解 5：KV Cache 优化只影响长文本

不完全。长文本场景收益明显，但高并发短请求也会受益，因为短请求不再按最大长度浪费显存。

---

# 七、面试常见问题与参考答案

## 1. 什么是 KV Cache？为什么推理时需要它？

KV Cache 是 Transformer 自回归推理时缓存历史 token 的 Key 和 Value。生成第 t 个 token 时，当前 token 的 Query 需要和历史所有 token 的 Key 做 attention，并加权读取历史 Value。如果不缓存，每生成一步都要重新计算所有历史 token 的 K/V，复杂度和延迟会很高。KV Cache 通过保存历史 K/V，让 decode 阶段每步只计算新 token 的 K/V，从而显著降低重复计算。

---

## 2. 传统 KV Cache 管理有什么问题？

传统方式通常按 `max_batch_size × max_seq_len × num_kv_heads × head_dim` 预分配连续 KV Cache。问题是请求实际长度通常远小于最大长度，但未使用的位置也被占住，不能分配给其他请求。这会造成内部碎片、显存浪费、可容纳 batch size 下降，最终降低吞吐量。

---

## 3. vLLM 的 PagedAttention 解决了什么问题？

PagedAttention 解决的是大模型推理中 KV Cache 显存管理低效的问题。它把 KV Cache 划分为固定大小的 block，按需分配给请求，并通过 block table 记录逻辑 block 到物理 block 的映射。attention kernel 根据这张映射表读取不连续的 KV blocks，从而减少显存碎片，提高显存利用率和批处理能力。

---

## 4. block_size 表示什么？

block_size 表示一个 KV Cache block 能容纳多少个 token 的 K/V。例如 `block_size = 16` 表示每个 block 可以存 16 个 token 对应的 KV Cache。它不是 batch size，也不是 hidden size，而是 KV Cache 管理的分页粒度。

---

## 5. 分块管理是否完全消除了显存浪费？

没有。最后一个 block 可能没有填满，所以仍然存在内部碎片。但浪费上界从传统方式的 `max_seq_len - actual_len` 降低到 `block_size - 1` 个 token 对应的 KV Cache。相比按最大长度预分配，浪费大幅降低。

---

## 6. block table 是什么？

block table 是记录某个请求的逻辑 block 到物理 KV block 映射的数据结构。由于分块管理后，一个请求的 KV Cache 在物理显存上可以不连续，所以 attention kernel 需要通过 block table 找到每个逻辑位置对应的物理 block。

---

## 7. logical block 和 physical block 有什么区别？

logical block 是请求内部按 token 顺序划分出来的逻辑编号，例如一个请求的第 0、1、2 个 block。physical block 是 GPU 显存中真实分配出来的 KV Cache 块。二者通过 block table 映射。逻辑上连续，不代表物理上连续。

---

## 8. Prefill 和 Decode 阶段中 KV Cache 分别如何使用？

Prefill 阶段一次性处理 prompt 的所有 token，并把这些 token 在各层产生的 K/V 写入 KV Cache。Decode 阶段每次生成一个 token，只计算这个新 token 的 K/V，并追加写入 KV Cache；同时 attention 会读取之前所有 token 的 KV Cache 来生成下一个 token。

---

## 9. 为什么 PagedAttention 能提升吞吐量？

因为它减少了 KV Cache 显存浪费，使同一张 GPU 能容纳更多请求和更多 token。更高的并发和更大的 batch 可以提高 GPU 利用率，从而提升系统整体吞吐量。它的收益主要来自系统级显存管理优化，而不是改变 Transformer 的数学结构。

---

## 10. 为什么 attention kernel 需要为 PagedAttention 做特殊设计？

因为传统 attention 通常假设一个序列的 K/V 在内存中连续存放。PagedAttention 下，同一请求的 K/V 可能分散在多个 physical blocks 中，所以 kernel 必须根据 block table 进行间接寻址，读取多个不连续 block 中的 K/V，再完成 attention 计算。

---

## 11. block_size 选择过大或过小分别有什么问题？

block_size 过大时，最后一个 block 的内部碎片更大，短请求浪费更多显存。block_size 过小时，block 数量变多，block table 更大，调度和 kernel 访问的管理开销增加。因此实际系统中需要在碎片率和管理开销之间折中。

---

## 12. PagedAttention 和 prefix caching 有什么关系？

PagedAttention 提供了按 block 管理 KV Cache 的基础。prefix caching 可以在多个请求共享相同前缀时，让它们复用相同前缀对应的 KV blocks，而不必重复计算和重复存储。因此分块管理天然适合做前缀共享、树状请求复用、speculative decoding 等优化。

---

## 13. vLLM 的分块 KV Cache 和操作系统分页有什么类比？

操作系统把虚拟地址空间切成页，通过页表映射到物理内存页；vLLM 把一个请求的 KV Cache 切成 logical blocks，通过 block table 映射到 GPU 上的 physical KV blocks。两者都通过“逻辑连续、物理可不连续”的方式减少连续内存分配压力和碎片问题。

---

# 八、用一句话背下来

如果面试中需要快速概括这节课，可以这样说：

> vLLM 的 PagedAttention 把每个请求的 KV Cache 从传统的连续最大长度预分配，改成按固定 block 分页、按需申请、通过 block table 间接寻址读取，从而把 KV Cache 的浪费从 `max_seq_len` 级别降低到 `block_size` 级别，提高显存利用率、batch size 和推理吞吐量。

---

# 九、建议你接下来如何复习

建议按下面顺序复习：

1. 先搞懂 Transformer 推理为什么需要 KV Cache；
2. 再搞懂传统连续 KV Cache 为什么浪费；
3. 然后理解 `block_size`、logical block、physical block、block table；
4. 再把 prefill / decode 和 block 分配流程串起来；
5. 最后去读 nano-vLLM 的 `block_manager.py`、`scheduler.py`、`model_runner.py`，看这些概念在代码里怎么落地。

真正掌握这节课的标志是：你能不看图说清楚下面这句话：

```text
一个请求进来后，prompt 被 tokenizer 编成 token ids，按 block_size 切成逻辑 block，BlockManager 为这些逻辑 block 分配物理 KV blocks，模型 forward 计算出的 K/V 被写入这些 blocks，后续 decode 继续按需申请新 block，attention 通过 block table 读取历史 KV，直到请求结束释放 block。
```


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-vllm|模块-vllm]]
- 关联阅读：[[outputs/项目整理/专题-02-KV缓存与调度|专题-02-KV缓存与调度]]

%% 项目关联导航：结束 %%
