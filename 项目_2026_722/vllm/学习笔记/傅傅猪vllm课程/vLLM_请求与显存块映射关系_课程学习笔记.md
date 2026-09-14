# vLLM 请求与显存块映射关系：课程深度整理学习笔记

> 本笔记基于课程视频语音转文字内容整理。原文中存在较多自动识别误差，例如 “VLM” 应理解为 **vLLM**，“KVCatch” 应理解为 **KV Cache**，“Blog” 应理解为 **Block**，“talker” 应理解为 **token**。本笔记已按推理系统语境进行技术名词校正。

---

## 一、本节课的核心宗旨

这节课的主题是：

> **在 vLLM 的 PagedAttention / 分块 KV Cache 机制中，如何把一次调度中的多个请求、多个 token，映射到真实的 GPU KV Cache 显存块中。**

前面课程已经讲过：vLLM 不再像传统推理方式那样，为每个请求一次性申请一整段连续 KV Cache，而是把 KV Cache 显存切成很多固定大小的 **block**。每个 block 能容纳固定数量 token 的 K/V 缓存。例如 `block_size = 16` 表示一个 block 能存放 16 个 token 对应的 KV Cache。

本节课进一步解决一个更关键的问题：

> **显存已经被切成 block 了，那么当多个请求一起进入 GPU 执行时，系统怎么知道每个请求的 token 应该读写到哪个 block、哪个位置？**

因此，本节课实际上是从“显存管理”进入“模型执行前的数据准备”阶段。它对应的是推理引擎中非常核心的准备逻辑：

1. 多个请求被调度器选中；
2. 每个请求本轮可能只需要计算一部分 token；
3. 这些 token 会被打平成一个统一的一维 batch；
4. GPU attention kernel 必须知道：
   - 每个请求在这个一维 batch 里的边界在哪里；
   - 每个请求已经计算过多少 token；
   - 每个请求占用了哪些 KV Cache block；
   - 当前要计算的每个 token 应该写入 KV Cache 的哪个物理 slot；
5. 这些信息最终服务于 **PagedAttention** 的计算。

所以，这节课的技术主线可以概括为：

> **请求 token → 打平成 GPU batch → 用 `query_start_loc` 区分请求边界 → 用 `block_table` 找到请求占用的物理 block → 用 `slot_mapping` 找到每个 token 的 KV Cache 写入位置 → 供 PagedAttention kernel 使用。**

---

## 二、关键术语纠错与对应关系

| 转写中可能出现的词 | 正确技术名词 | 含义 |
|---|---|---|
| VLM | vLLM | 大模型推理框架 |
| KVCatch / TVCatch | KV Cache | Attention 中缓存历史 token 的 Key / Value |
| Blog / Bolck / 显存快 | Block | KV Cache 的固定大小物理块 |
| BlogSize / Bolishize | block_size | 每个 block 可容纳的 token 数 |
| talker / 偷肯 | token | 模型处理文本的基本单位 |
| ProMode / Promote | Prompt | 用户输入的提示词 |
| GPU Modular / Model Rona | GPU ModelRunner / ModelRunner | 模型在 GPU 上执行前后的调度与运行组件 |
| query start the lock / qstadlock | `query_start_loc` | 标记不同请求在扁平化 token batch 中的起始位置 |
| plotable | `block_table` | 请求的逻辑 block 到物理 block 的映射表 |
| slotmap / slot 卖品 | `slot_mapping` | 每个 token 对应的 KV Cache 物理写入位置 |
| numb computed talker | `num_computed_tokens` | 每个请求已经计算过的 token 数量 |

---

## 三、按照课程推进顺序梳理知识点

由于转文字文件没有给出精确时间戳，下面按照视频讲解的自然推进顺序整理。

---

### 1. 开头：本节课处于模型执行前的准备阶段

#### 核心知识点

课程一开始说明，本节内容讲的是：

> **请求和显存块的映射关系。**

这个阶段已经不是“为什么要分块管理 KV Cache”，而是进入到更靠近代码实现的位置，也就是 **GPU ModelRunner 中模型真正执行前的数据准备阶段**。

#### 相关概念解释

在大模型推理中，一次 forward 之前，系统不能只把 token IDs 扔给 GPU。对于 vLLM 这种使用 PagedAttention 的框架，还必须准备额外的元数据。

这些元数据包括：

1. 本轮执行了哪些请求；
2. 每个请求本轮执行多少个 token；
3. 这些 token 在 batch 中怎么排列；
4. 这些 token 的 KV Cache 应该写入哪里；
5. 每个请求过去已经写入了哪些 KV Cache block。

如果没有这些信息，GPU 上的 attention kernel 就无法正确访问历史 K/V，也无法把新 token 的 K/V 写入正确位置。

#### 在 AI Infra / 推理系统中的作用

这部分是推理引擎和普通 Transformer 代码最大的区别之一。

普通教学代码里，attention 通常认为输入是一个规则张量，例如：

```text
[batch_size, seq_len, hidden_size]
```

但在线推理服务中，每个请求长度不同，生成进度也不同：

```text
Request 1: 本轮算 3 个 token
Request 2: 本轮算 1 个 token
Request 3: 本轮算 3 个 token
```

为了让 GPU 高效并行执行，vLLM 需要把这些不规则请求组织成一个可执行的 batch，同时还要保存每个请求自己的边界和 KV Cache 地址。

---

### 2. 回顾：vLLM 为什么要把 KV Cache 分成 block

#### 核心知识点

前面课程已经讲过，vLLM 会把一个很长的 prompt 按 `block_size` 切成多个 block。每个 block 负责存放一段 token 对应的 KV Cache。

例如：

```text
block_size = 4

Prompt tokens:
[t0, t1, t2, t3, t4, t5, t6, t7, t8]

逻辑切分：
Block 0: [t0, t1, t2, t3]
Block 1: [t4, t5, t6, t7]
Block 2: [t8]
```

传统方式可能会为一个请求一次性申请最大长度的连续 KV Cache，例如 4096 个 token 的空间。即使这个请求最终只生成了 10 个 token，后面大量显存也会被提前占住，其他请求无法使用。

vLLM 的分块方式则是：

> **用多少，申请多少；需要新的 token 空间时，再申请新的 block。**

#### 相关概念解释

KV Cache 的作用是缓存每一层 attention 中历史 token 的 Key 和 Value。自回归生成时，当前 token 需要和历史 token 做 attention。如果每次都重新计算所有历史 token 的 K/V，计算成本会非常高。因此推理系统会缓存历史 K/V。

但是 KV Cache 的显存开销很大，并且和序列长度线性相关：

```text
KV Cache 大小 ≈ 层数 × token数 × KV head数 × head_size × 2(K和V) × dtype字节数
```

vLLM 把这段显存做成“分页式管理”，类似操作系统中的分页内存：

```text
逻辑序列 token
        ↓
逻辑 block
        ↓
物理 KV Cache block
```

#### 在 AI Infra / 推理系统中的作用

分块管理的核心收益是：

1. **减少显存浪费**：不用一次性按最大长度申请；
2. **降低内存碎片影响**：固定大小 block 更容易复用；
3. **提升并发能力**：释放出的 block 可以给其他请求用；
4. **支撑 continuous batching**：不同请求可以在不同时间加入或退出 batch；
5. **为 prefix caching / 共享前缀等优化打基础**。

---

### 3. 第一类核心元数据：`query_start_loc`

#### 核心知识点

课程中重点讲了一个变量：

```text
query_start_loc
```

它用于区分一次调度中，不同请求在扁平化 token batch 里的边界。

一次调度中可能有多个请求，每个请求本轮需要计算的 token 数量不同。为了让 GPU 高效计算，系统会把它们打平成一个一维 token 序列。

例如本轮调度：

```text
Request 1 本轮要算: [31, 55, 61]
Request 2 本轮要算: [99]
Request 3 本轮要算: [29, 29, 99]
```

打平成一维后：

```text
flat_tokens = [31, 55, 61, 99, 29, 29, 99]
```

但是打平成一维后，原来的请求边界丢失了。此时就需要 `query_start_loc` 记录每个请求的起始位置：

```text
query_start_loc = [0, 3, 4, 7]
```

含义是：

```text
Request 1: flat_tokens[0:3] = [31, 55, 61]
Request 2: flat_tokens[3:4] = [99]
Request 3: flat_tokens[4:7] = [29, 29, 99]
```

它本质上就是一个前缀和数组：

```text
query_start_loc[i] = 第 i 个请求在 flat_tokens 中的起始下标
```

最后一个元素表示所有 token 的总数。

#### 相关概念解释

`query_start_loc` 类似很多高性能 attention kernel 中的 `cu_seqlens`，也就是 cumulative sequence lengths。

它解决的问题是：

> **如何在一个扁平化的一维 token batch 中，恢复每个请求自己的 token 边界。**

在 GPU kernel 中，很多时候不希望保留复杂的 Python list 或变长数组，而是把所有 token 摊平成连续数组，再用前缀和数组告诉 kernel 每个请求在哪里开始、在哪里结束。

#### 在 AI Infra / 推理系统中的作用

`query_start_loc` 的作用非常关键：

1. 支持变长请求 batch；
2. 支持不同请求在同一次 forward 中并行执行；
3. 让 attention kernel 知道每个请求的 query token 范围；
4. 避免不同请求之间的 attention 串扰；
5. 是 PagedAttention / FlashAttention 类后端常见的输入元数据。

一个重要理解是：

> **`query_start_loc` 不是 KV Cache 的地址表，它只负责标记“本轮 query token 属于哪个请求”。**

KV Cache 的物理地址还需要依赖后面的 `block_table` 和 `slot_mapping`。

---

### 4. 第二类核心元数据：KV Cache 的物理 block 空间

#### 核心知识点

课程接着回顾了每一层 attention 中 KV Cache 的 block 化空间。

每个 attention 层都会有自己独立的一块 KV Cache 区域。这块区域由多个 block 组成：

```text
一层 KV Cache ≈ num_blocks × block_size × num_kv_heads × head_size × 2 × dtype
```

其中：

| 参数 | 含义 |
|---|---|
| `num_blocks` | 当前层可以使用多少个 KV Cache block |
| `block_size` | 每个 block 能存多少个 token 的 KV Cache |
| `num_kv_heads` | KV head 的数量 |
| `head_size` | 每个 head 的维度 |
| `2` | K 和 V 两份缓存 |
| `dtype` | 数据类型，例如 FP16 / BF16，占用 2 字节 |

#### 相关概念解释

对于一个 token 来说，在某一层 attention 中需要缓存：

```text
K: [num_kv_heads, head_size]
V: [num_kv_heads, head_size]
```

所以一个 token 的 KV Cache 大小约为：

```text
num_kv_heads × head_size × 2 × dtype字节数
```

一个 block 能放 `block_size` 个 token，所以一个 block 的大小约为：

```text
block_size × num_kv_heads × head_size × 2 × dtype字节数
```

一层有 `num_blocks` 个 block，所以一层 KV Cache 空间约为：

```text
num_blocks × block_size × num_kv_heads × head_size × 2 × dtype字节数
```

如果模型有 `num_layers` 层，那么总 KV Cache 大小约为：

```text
num_layers × num_blocks × block_size × num_kv_heads × head_size × 2 × dtype字节数
```

#### 在 AI Infra / 推理系统中的作用

这部分解释的是：

> **block_table 和 slot_mapping 最终要映射到哪里。**

前面的 `query_start_loc` 只是告诉系统“本轮有哪些 token 属于哪些请求”，但每个 token 计算出的 K/V 最终必须写入真实 GPU 显存。这个真实显存就是按层分配好的 KV Cache block 空间。

需要注意：

不同 attention 后端对 KV Cache 张量的 layout 可能不同。例如某些后端希望 K/V 分开存，有些后端希望 block 维度在前，有些后端会为了访存对齐做 reshape。因此，实际代码中的 shape 可能会因 backend 而变化，但本质容量公式不变。

---

### 5. 第三类核心元数据：`block_table`

#### 核心知识点

课程中提到，一个请求会占用哪些 block，需要有一张映射表来记录。这个映射表通常可以理解为：

```text
block_table
```

它记录的是：

> **每个请求的逻辑 block，对应到哪个物理 KV Cache block。**

例如：

```text
block_size = 4

Request 1 的 token:
[t0, t1, t2, t3, t4, t5, t6, t7, t8]

逻辑 block：
logical block 0: [t0, t1, t2, t3]
logical block 1: [t4, t5, t6, t7]
logical block 2: [t8]
```

假设系统给 Request 1 分配的物理 block 是：

```text
logical block 0 → physical block 10
logical block 1 → physical block 25
logical block 2 → physical block 7
```

那么可以表示为：

```text
block_table[Request 1] = [10, 25, 7]
```

#### 相关概念解释

这里要区分两个概念：

| 概念 | 含义 |
|---|---|
| 逻辑 block | 一个请求内部按 token 顺序切出来的第几个 block |
| 物理 block | GPU KV Cache 池中真实分配到的 block 编号 |

为什么需要这层映射？

因为 vLLM 使用分页式显存管理，一个请求的 KV Cache 不一定连续存放在物理显存中。它可能像这样分散：

```text
Request 1:
逻辑第 0 块 → 物理 block 10
逻辑第 1 块 → 物理 block 25
逻辑第 2 块 → 物理 block 7
```

只要有 `block_table`，attention kernel 就能通过逻辑位置找到真实物理 block。

#### 在 AI Infra / 推理系统中的作用

`block_table` 是 PagedAttention 的核心。

传统 attention 假设一个请求的 KV Cache 是连续的：

```text
Request 1 KV Cache:
[token0][token1][token2][token3][token4]...
```

PagedAttention 则允许它不连续：

```text
Request 1 KV Cache:
logical block 0 → physical block 10
logical block 1 → physical block 25
logical block 2 → physical block 7
```

这样做可以极大提高显存利用率，因为系统不需要为每个请求找一整段连续空间，只需要找到若干个空闲 block。

---

### 6. 第四类核心元数据：`slot_mapping`

#### 核心知识点

课程继续讲了更细粒度的映射关系：

> **每个 token 在 block 内部的偏移位置是什么。**

这就是 `slot_mapping` 要解决的问题。

假设：

```text
block_size = 4

Request 1 的 token 下标：
token position: 0 1 2 3 4 5 6 7 8
```

逻辑 block 切分为：

```text
logical block 0: token position 0, 1, 2, 3
logical block 1: token position 4, 5, 6, 7
logical block 2: token position 8
```

如果本轮要处理的是：

```text
token position 6 和 token position 7
```

那么它们属于：

```text
logical_block_id = position // block_size
```

计算得到：

```text
token position 6: logical_block_id = 6 // 4 = 1
token position 7: logical_block_id = 7 // 4 = 1
```

它们在 block 内部的偏移是：

```text
block_offset = position % block_size
```

计算得到：

```text
token position 6: block_offset = 6 % 4 = 2
token position 7: block_offset = 7 % 4 = 3
```

如果 `block_table[Request 1][1] = 25`，说明 Request 1 的逻辑 block 1 对应物理 block 25，那么这两个 token 的物理 slot 可以表示为：

```text
slot = physical_block_id × block_size + block_offset
```

所以：

```text
token position 6 → slot = 25 × 4 + 2
token position 7 → slot = 25 × 4 + 3
```

这就是 `slot_mapping` 的核心含义。

#### 相关概念解释

`slot_mapping` 不是简单的“第几个 token”，而是告诉系统：

> **当前这个 token 计算出来的 K/V，要写到 KV Cache 池里的哪个物理位置。**

可以把它理解为 KV Cache 的“写地址”。

在实际实现中，`slot_mapping` 通常是一个一维数组，它的长度等于本轮参与计算的 token 数。数组中的每个元素对应一个 token 的物理 slot。

例如本轮 flat_tokens 有 7 个 token：

```text
flat_tokens = [31, 55, 61, 99, 29, 29, 99]
```

那么：

```text
slot_mapping = [slot0, slot1, slot2, slot3, slot4, slot5, slot6]
```

`slot_mapping[i]` 表示 `flat_tokens[i]` 这个 token 的 K/V 应该写入哪里。

#### 在 AI Infra / 推理系统中的作用

`slot_mapping` 是连接“当前计算 token”和“KV Cache 物理写入地址”的关键变量。

它的作用包括：

1. 写入新 token 的 K/V；
2. 让 attention kernel 正确读取历史 K/V；
3. 支持请求 KV Cache 非连续存储；
4. 支持多个请求混合成一个 batch 后仍能互不干扰；
5. 是 PagedAttention 能够工作的关键地址映射。

---

### 7. 第五类核心元数据：`num_computed_tokens`

#### 核心知识点

课程还提到了：

```text
num_computed_tokens
```

它表示：

> **每个请求已经被计算过的 token 数量。**

例如一个请求原始 prompt 有 9 个 token：

```text
[t0, t1, t2, t3, t4, t5, t6, t7, t8]
```

如果当前已经完成了前 6 个 token 的 prefill，下一轮要从 token position 6 开始继续计算，那么：

```text
num_computed_tokens = 6
```

#### 相关概念解释

在 continuous batching 中，请求不是一次性固定执行完的。一个请求可能在某一轮只计算 prompt 的一部分，也可能在下一轮继续计算，也可能进入 decode 阶段每轮生成一个 token。

所以系统必须记录每个请求的进度：

```text
这个请求已经算到哪里了？
下一轮应该从哪个 token 开始算？
当前要算的 token 属于哪个逻辑 block？
```

这些都和 `num_computed_tokens` 有关。

#### 在 AI Infra / 推理系统中的作用

`num_computed_tokens` 主要用于：

1. 确定本轮从请求的哪个 token 位置开始计算；
2. 判断是否需要分配新的 block；
3. 计算当前 token 的逻辑 block id 和 block offset；
4. 区分 prefill 阶段和 decode 阶段；
5. 让请求可以被分多轮调度执行。

---

## 四、把整节课串成完整技术流程

下面用“从输入到输出”的方式，把本节内容串起来。

---

### Step 1：用户请求进入系统

用户输入 prompt，例如：

```text
hello everybody from xhs, where are you?
```

经过 tokenizer 后变成 token IDs：

```text
[11, 13, 41, 56, 22, 23, 31, 44, ...]
```

这些 token 不是直接一次性全部固定写入一整段连续 KV Cache，而是要按照 vLLM 的 block 管理方式处理。

---

### Step 2：调度器决定本轮执行哪些请求、哪些 token

在线推理系统中可能同时有多个请求：

```text
Request 1
Request 2
Request 3
...
```

每个请求的进度不同：

```text
Request 1: prompt 还没完全 prefill
Request 2: 已经进入 decode
Request 3: prompt 较长，本轮只处理其中一段
```

调度器会根据显存、最大 batch token 数、请求状态等因素，决定本轮执行哪些 token。

---

### Step 3：把本轮 token 打平成一维 batch

假设本轮调度结果是：

```text
Request 1: [31, 55, 61]
Request 2: [99]
Request 3: [29, 29, 99]
```

为了 GPU 高效执行，系统把它们拼成：

```text
flat_tokens = [31, 55, 61, 99, 29, 29, 99]
```

但这样做会丢失请求边界。

---

### Step 4：用 `query_start_loc` 保存请求边界

系统生成：

```text
query_start_loc = [0, 3, 4, 7]
```

于是 GPU kernel 可以知道：

```text
Request 1 对应 flat_tokens[0:3]
Request 2 对应 flat_tokens[3:4]
Request 3 对应 flat_tokens[4:7]
```

这一步解决的是：

> **本轮 query token 如何区分属于哪个请求。**

---

### Step 5：根据请求进度确定 token 的逻辑位置

对于每个请求，系统知道它已经计算过多少 token：

```text
num_computed_tokens
```

结合本轮 token 在请求内部的位置，可以得到每个 token 的全局 token position。

例如：

```text
Request 1 已经算过 6 个 token
本轮要算 2 个 token
```

那么本轮 token 的 position 可能是：

```text
6, 7
```

---

### Step 6：根据 `block_size` 计算逻辑 block 和 block 内偏移

假设：

```text
block_size = 4
```

则：

```text
logical_block_id = token_position // block_size
block_offset = token_position % block_size
```

对于 token position 6 和 7：

```text
position 6 → logical_block_id = 1, block_offset = 2
position 7 → logical_block_id = 1, block_offset = 3
```

这一步解决的是：

> **当前 token 属于请求内部的第几个逻辑 block，以及在 block 内第几个位置。**

---

### Step 7：用 `block_table` 找到物理 block

假设 Request 1 的 `block_table` 是：

```text
block_table[Request 1] = [10, 25, 7]
```

那么：

```text
logical_block_id = 1
physical_block_id = block_table[Request 1][1] = 25
```

这一步解决的是：

> **请求内部的逻辑 block 对应 GPU KV Cache 池中的哪个物理 block。**

---

### Step 8：生成 `slot_mapping`

有了：

```text
physical_block_id
block_offset
block_size
```

就可以计算物理 slot：

```text
slot = physical_block_id × block_size + block_offset
```

例如：

```text
position 6 → slot = 25 × 4 + 2
position 7 → slot = 25 × 4 + 3
```

所有本轮 token 都会生成一个对应的 slot，构成：

```text
slot_mapping
```

这一步解决的是：

> **每个 token 的 K/V 应该写入 KV Cache 的哪个物理位置。**

---

### Step 9：进入 PagedAttention 计算

最终，GPU attention kernel 会拿到：

```text
input_ids / token ids
positions
query_start_loc
block_table
slot_mapping
num_computed_tokens
KV Cache tensors
```

然后执行 attention：

1. 对本轮 token 计算 Q/K/V；
2. 根据 `slot_mapping` 把新 K/V 写入 KV Cache；
3. 根据 `block_table` 读取历史 token 的 K/V；
4. 根据 `query_start_loc` 区分不同请求的 query 范围；
5. 完成 attention 输出；
6. 继续后续 MLP、logits、采样等步骤。

---

## 五、用一个完整例子理解本节课

假设：

```text
block_size = 4
```

当前有 3 个请求，本轮要执行的 token 如下：

```text
Request 1: [31, 55, 61]
Request 2: [99]
Request 3: [29, 29, 99]
```

打平后：

```text
flat_tokens = [31, 55, 61, 99, 29, 29, 99]
```

得到：

```text
query_start_loc = [0, 3, 4, 7]
```

假设 Request 1 之前已经计算过 6 个 token，因此这次的 3 个 token 对应 position：

```text
6, 7, 8
```

计算逻辑 block 和 offset：

```text
position 6:
logical_block_id = 6 // 4 = 1
block_offset = 6 % 4 = 2

position 7:
logical_block_id = 7 // 4 = 1
block_offset = 7 % 4 = 3

position 8:
logical_block_id = 8 // 4 = 2
block_offset = 8 % 4 = 0
```

假设：

```text
block_table[Request 1] = [10, 25, 7]
```

则：

```text
position 6 → physical block 25, offset 2
position 7 → physical block 25, offset 3
position 8 → physical block 7, offset 0
```

对应 slot：

```text
position 6 → 25 × 4 + 2 = 102
position 7 → 25 × 4 + 3 = 103
position 8 → 7 × 4 + 0 = 28
```

所以 Request 1 本轮 token 的 `slot_mapping` 可以理解为：

```text
[102, 103, 28]
```

这说明：

```text
第一个 token 的 KV 写到物理 slot 102
第二个 token 的 KV 写到物理 slot 103
第三个 token 的 KV 写到物理 slot 28
```

注意这里第三个 token 跳到了另一个物理 block，这正是 PagedAttention 的特点：**逻辑上连续的 token，不要求物理显存连续。**

---

## 六、本节课最重要的理解点

### 1. `query_start_loc` 解决的是 batch 中请求边界问题

它告诉 GPU：

```text
flat_tokens 中哪些 token 属于同一个请求。
```

它不负责显存地址。

---

### 2. `block_table` 解决的是逻辑 block 到物理 block 的映射问题

它告诉 GPU：

```text
某个请求的第 n 个逻辑 block，实际存放在哪个物理 block。
```

它是 PagedAttention 的核心地址表。

---

### 3. `slot_mapping` 解决的是当前 token 的 KV Cache 写入地址问题

它告诉 GPU：

```text
当前这个 token 的 K/V 应该写到 KV Cache 池里的哪个 slot。
```

---

### 4. `num_computed_tokens` 解决的是请求执行进度问题

它告诉系统：

```text
这个请求已经算了多少 token，本轮应该从哪里继续。
```

---

### 5. 四者之间的关系

可以这样记：

```text
query_start_loc：
    当前 batch 中，请求和请求之间怎么分界？

num_computed_tokens：
    每个请求之前已经算到哪里？

block_table：
    每个请求占用了哪些物理 block？

slot_mapping：
    当前每个 token 的 KV 应该写到哪个物理位置？
```

最终它们一起服务于：

```text
PagedAttention kernel 正确读写 KV Cache
```

---

## 七、和 Transformer 原理的对应关系

普通 Transformer attention 中，我们通常只关心：

```text
Q = XWq
K = XWk
V = XWv
Attention(Q, K, V)
```

但在推理系统中，尤其是自回归生成时，K/V 不只是当前 token 的 K/V，还包括历史 token 的 K/V。

因此推理系统要解决：

```text
历史 K/V 存在哪里？
当前 token 的 K/V 写到哪里？
当前 token 能看到哪些历史 K/V？
不同请求之间如何隔离？
```

这就是 vLLM 中 `query_start_loc`、`block_table`、`slot_mapping` 这些变量存在的原因。

可以这样对应：

| Transformer 原理概念 | vLLM 推理系统中的实现问题 |
|---|---|
| token 序列 | 多个请求的 token 被打平成 flat_tokens |
| attention mask / 请求边界 | `query_start_loc` 帮助区分不同请求 |
| 历史 K/V | 存在 KV Cache block 中 |
| 序列位置 position | 用于计算 logical block id 和 block offset |
| K/V 写入 | 由 `slot_mapping` 指定 |
| K/V 读取 | 由 `block_table` + token 位置决定 |
| batch 推理 | 多个请求合并成一次 GPU forward |

---

## 八、面试高频问题与参考答案

---

### 问题 1：vLLM 中为什么需要 `query_start_loc`？

#### 参考答案

vLLM 在一次调度中会同时处理多个请求，而且每个请求本轮要计算的 token 数量可能不同。为了提高 GPU 执行效率，系统会把这些 token 打平成一个一维数组。但打平以后，请求之间的边界信息会丢失。

`query_start_loc` 是一个前缀和数组，用来记录每个请求在这个一维 token batch 中的起始位置。例如本轮 3 个请求 token 数分别是 3、1、3，那么：

```text
query_start_loc = [0, 3, 4, 7]
```

这样 attention kernel 就能知道：

```text
第 0 到 2 个 token 属于请求 1
第 3 个 token 属于请求 2
第 4 到 6 个 token 属于请求 3
```

所以，`query_start_loc` 的本质作用是标记变长请求在扁平化 batch 中的边界。

---

### 问题 2：`query_start_loc` 和 `slot_mapping` 有什么区别？

#### 参考答案

二者解决的问题不同。

`query_start_loc` 解决的是：

```text
当前 batch 中每个请求的 token 范围在哪里。
```

它关心的是请求边界。

`slot_mapping` 解决的是：

```text
当前每个 token 的 K/V 应该写入 KV Cache 的哪个物理 slot。
```

它关心的是 KV Cache 写入地址。

所以可以简单记：

```text
query_start_loc：区分请求
slot_mapping：定位显存
```

---

### 问题 3：什么是 `block_table`？

#### 参考答案

`block_table` 是请求的逻辑 block 到物理 KV Cache block 的映射表。

在 vLLM 中，一个请求的 KV Cache 按 `block_size` 切成多个逻辑 block。但这些逻辑 block 不一定连续存放在 GPU 显存中，它们可能分散在不同的物理 block 中。

例如：

```text
block_table[request] = [10, 25, 7]
```

表示：

```text
请求的逻辑 block 0 存在物理 block 10
请求的逻辑 block 1 存在物理 block 25
请求的逻辑 block 2 存在物理 block 7
```

PagedAttention kernel 通过 `block_table` 找到历史 token 的 K/V 所在物理位置。

---

### 问题 4：`slot_mapping` 是怎么计算的？

#### 参考答案

假设某个 token 在请求内部的位置是 `token_position`，每个 block 能容纳 `block_size` 个 token，那么：

```text
logical_block_id = token_position // block_size
block_offset = token_position % block_size
```

然后通过 `block_table` 找到物理 block：

```text
physical_block_id = block_table[request_id][logical_block_id]
```

最后计算物理 slot：

```text
slot = physical_block_id × block_size + block_offset
```

所有本轮 token 的 slot 组成 `slot_mapping`。

---

### 问题 5：`num_computed_tokens` 的作用是什么？

#### 参考答案

`num_computed_tokens` 表示一个请求已经被计算过的 token 数量。它主要用于记录请求进度。

在 continuous batching 中，一个请求可能不会一次性完成全部 prompt 计算，而是可能被拆成多轮执行。因此系统必须知道这个请求已经算到哪里，下一轮从哪个 token 继续。

它还会参与计算当前 token 的逻辑位置，从而进一步计算：

```text
logical_block_id
block_offset
slot_mapping
```

---

### 问题 6：为什么 PagedAttention 需要 `block_table`？

#### 参考答案

PagedAttention 的核心思想是允许一个请求的 KV Cache 分散存放在多个非连续物理 block 中。这样可以避免为每个请求申请大块连续显存，从而减少显存碎片，提高显存利用率和并发能力。

但是一旦 KV Cache 不连续存放，attention kernel 就必须有一张地址表来找到每个逻辑 block 对应的物理 block。这个地址表就是 `block_table`。

没有 `block_table`，kernel 就不知道一个请求的历史 K/V 分别存在哪里。

---

### 问题 7：为什么把多个请求 token 打平成一维数组？

#### 参考答案

因为 GPU 更适合处理规则、连续的批量数据。如果每个请求都保留成 Python 层面的变长 list，会导致 kernel 调度复杂、并行效率低。

vLLM 会把本轮所有请求要计算的 token 拼成一个一维 batch，然后用额外元数据恢复结构信息：

```text
query_start_loc：恢复请求边界
block_table：恢复请求的 KV Cache block 地址
slot_mapping：确定每个 token 的 KV 写入地址
```

这种方式兼顾了 GPU 执行效率和变长请求管理。

---

### 问题 8：`block_size` 会影响哪些东西？

#### 参考答案

`block_size` 会影响：

1. 每个 KV Cache block 能存多少 token；
2. block 内部偏移的计算；
3. `logical_block_id = token_position // block_size`；
4. `block_offset = token_position % block_size`；
5. 显存浪费上限；
6. block 管理粒度；
7. attention kernel 的访存模式和性能。

一般来说，`block_size` 越大，管理开销越小，但最后一个 block 的内部浪费可能越大；`block_size` 越小，显存利用更细，但 block 管理和 kernel 地址计算开销可能更高。

---

### 问题 9：为什么说这些变量是在模型执行前准备的？

#### 参考答案

因为 attention kernel 在 GPU 上执行时，需要提前知道本轮 batch 的结构和 KV Cache 地址。如果等到 kernel 内部再动态推断，会非常低效甚至不可行。

因此在模型 forward 前，ModelRunner 会准备好：

```text
input_ids
positions
query_start_loc
block_table
slot_mapping
```

然后把这些信息传给 attention backend。这样 GPU kernel 才能直接根据这些元数据读写 KV Cache。

---

### 问题 10：这节课和 vLLM 高吞吐有什么关系？

#### 参考答案

vLLM 的高吞吐来自多个方面，其中一个重要基础就是 PagedAttention 和分块 KV Cache 管理。

本节课讲的映射关系是 PagedAttention 能工作的前提：

1. `block_table` 让请求的 KV Cache 可以非连续存放；
2. `slot_mapping` 让新 token 的 K/V 能写到正确物理位置；
3. `query_start_loc` 让多个变长请求可以合并成一个 GPU batch；
4. `num_computed_tokens` 让请求可以被分多轮调度。

这些机制共同支撑了 continuous batching，提高了显存利用率和 GPU 利用率，因此能提升在线推理吞吐。

---

## 九、容易混淆的点

### 1. `block_size = 16` 不是 16 字节

`block_size = 16` 表示：

```text
一个 block 能存放 16 个 token 的 KV Cache
```

不是 16 字节，也不是 16 个浮点数。

一个 block 的真实显存大小还要乘上：

```text
num_kv_heads × head_size × 2 × dtype字节数 × 层数
```

如果只看单层，则不乘层数。

---

### 2. token ID 不等于 KV Cache

token ID 是 tokenizer 输出的整数编号，例如：

```text
hello → 15339
```

KV Cache 是模型 attention 层中计算出的 Key / Value 张量。

流程是：

```text
token ID
  ↓ embedding
hidden state
  ↓ attention linear projection
K / V
  ↓ 写入 KV Cache
```

所以 block 中真正存的是 token 对应的 K/V 张量，不是 token ID 本身。

---

### 3. `block_table` 不记录 block 里的具体 token 值

`block_table` 记录的是：

```text
逻辑 block → 物理 block
```

它不直接记录 token ID，也不直接记录 K/V 的数值。

---

### 4. `slot_mapping` 更像“写地址表”

当前 token 计算出 K/V 后，需要写入 KV Cache。写到哪里由 `slot_mapping` 指定。

---

### 5. `query_start_loc` 更像“请求边界表”

它告诉 GPU 本轮 flat token batch 中，每个请求从哪里开始、到哪里结束。

---

## 十、最终总结

本节课不是单纯讲 KV Cache 有多大，而是讲：

> **在 vLLM 的分块 KV Cache 机制中，多个请求、多个 token、多个显存 block 之间如何建立映射关系。**

最核心的四个变量是：

```text
query_start_loc
num_computed_tokens
block_table
slot_mapping
```

它们分别回答四个问题：

```text
query_start_loc：
    本轮 batch 里，每个请求的 token 边界在哪里？

num_computed_tokens：
    每个请求之前已经计算到哪里？

block_table：
    每个请求的逻辑 block 对应哪个物理 block？

slot_mapping：
    当前每个 token 的 K/V 应该写入哪个物理 slot？
```

这四个变量共同完成了从“请求级别逻辑序列”到“GPU 物理 KV Cache 地址”的转换。

因此，本节课可以看作是理解 vLLM PagedAttention 的关键前置内容。只有理解这些映射关系，后续再看 vLLM / nano-vLLM 中的 scheduler、block_manager、model_runner、attention backend，才不会只看到一堆变量名，而能明白它们背后的系统设计逻辑。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-vllm|模块-vllm]]
- 关联阅读：[[outputs/项目整理/专题-02-KV缓存与调度|专题-02-KV缓存与调度]]

%% 项目关联导航：结束 %%
