# 课程学习笔记：主题二_nano-vllm 之 PagedAttention 与内存管理_哔哩哔哩_bilibili

> 说明：本笔记基于课程语音转文字文件整理。原文存在明显自动识别误差，例如“Page Detention”应理解为 **PagedAttention**，“TVCAT / KVCatch / CAT”等应理解为 **KV Cache**，“SecretsLens”应理解为 **sequence length / 序列长度**，“NALA VLM / 当了未来”等应理解为 **nano-vLLM / vLLM**。以下内容已经按照 AI Infra / 大模型推理 / nano-vLLM 源码学习语境进行了技术校正、结构化整理和扩展解释。

---

## 目录

1. [视频宗旨与课程定位](#1-视频宗旨与课程定位)
2. [按时间进度梳理知识点](#2-按时间进度梳理知识点)
3. [核心概念详细解释](#3-核心概念详细解释)
4. [整节课技术流程串联](#4-整节课技术流程串联)
5. [我能从这节课中学到什么](#5-我能从这节课中学到什么)
6. [面试问题与回答思路](#6-面试问题与回答思路)
7. [和 nano-vLLM 源码文件的对应关系](#7-和-nano-vllm-源码文件的对应关系)
8. [学习建议与复习路线](#8-学习建议与复习路线)

---

# 1. 视频宗旨与课程定位

## 1.1 这节课主要讲什么

这节课的核心主题是：**nano-vLLM 中 PagedAttention 与 KV Cache 内存管理机制**。

更具体地说，这节课不是泛泛介绍 Transformer，也不是讲模型训练，而是在上一讲已经介绍 nano-vLLM 整体推理流程的基础上，进一步深入回答一个推理系统中的关键问题：

> 大模型推理时，每个请求都会持续生成 token，每个 token 都需要保存对应的 Key / Value。随着请求数增加、上下文长度变长，KV Cache 会占用大量显存。推理引擎如何高效管理这部分显存？

因此，这节课的技术主线是：

```text
为什么需要 KV Cache
  ↓
KV Cache 为什么会造成显存瓶颈
  ↓
普通连续显存分配为什么容易浪费和碎片化
  ↓
vLLM / nano-vLLM 如何用 PagedAttention 把 KV Cache 按 block 管理
  ↓
BlockManager / Scheduler / ModelRunner / Attention / Context 如何配合
  ↓
prefix caching 如何复用相同前缀的 KV Cache
  ↓
prefill / decode 阶段如何申请、写入、读取和释放 KV Cache
```

这节课真正想让学习者理解的，不只是“PagedAttention 是什么”，而是：

> 大模型推理引擎本质上也是一个资源管理系统，KV Cache 显存就像操作系统中的内存页，需要被动态分配、映射、共享和回收。

---

## 1.2 它在 AI Infra / 大模型推理学习路线中的位置

如果把大模型推理学习路线分成几层：

```text
第 1 层：Transformer 基础
知道 attention、Q/K/V、MLP、LayerNorm、logits、sampling 是什么。

第 2 层：推理流程基础
知道 prompt → prefill → decode → next token → EOS 的整体过程。

第 3 层：推理引擎核心
知道 continuous batching、KV Cache、request scheduling、block management。

第 4 层：性能优化
知道 PagedAttention、prefix caching、chunked prefill、CUDA Graph、FlashAttention、Tensor Parallel。

第 5 层：服务部署与工程化
知道 OpenAI-compatible API、benchmark、显存配置、吞吐量、TTFT、TPOT、并发优化。
```

本节课处在 **第 3 层和第 4 层之间**，属于推理引擎核心机制课程。它直接连接你前面学过的 `engine/` 源码与后续要理解的 `layers/attention.py`、vLLM Benchmark、推理服务性能调优。

---

## 1.3 它和 vLLM / nano-vLLM / Transformer 推理的关系

### 和 Transformer 推理的关系

Transformer Decoder 生成每个新 token 时，需要当前 token 的 Query 去关注历史 token 的 Key / Value。

如果没有 KV Cache，每次 decode 都要重新计算所有历史 token 的 K/V，计算量会非常大。

有了 KV Cache 后：

```text
历史 token 的 K/V 只算一次
  ↓
写入 KV Cache
  ↓
后续 decode 直接读取历史 K/V
  ↓
只需要为当前新 token 计算新的 Q/K/V
```

所以 KV Cache 是大模型自回归推理提速的基础。

### 和 vLLM 的关系

vLLM 的代表性创新之一就是 **PagedAttention**。PagedAttention 的思想是：不要为每个请求申请一整块连续 KV Cache，而是像操作系统分页一样，把 KV Cache 拆成固定大小的 block，由 block table 管理逻辑序列到物理显存 block 的映射。

### 和 nano-vLLM 的关系

nano-vLLM 是一个极简版本的 vLLM。它保留了 vLLM 推理引擎中最核心的思想，包括：

- Sequence 请求状态管理；
- Scheduler 调度 prefill / decode；
- BlockManager 分配和释放 KV Cache block；
- block table 记录 sequence 到 KV block 的映射；
- prefix caching 复用相同前缀；
- ModelRunner 准备 input、position、slot_mapping、block_tables；
- Attention 层真正写入和读取 KV Cache。

这节课就是围绕这些模块讲解 nano-vLLM 的内存管理主线。

---

## 1.4 这节课适合解决学习者的什么问题

这节课适合解决以下问题：

1. **为什么大模型推理显存不只是模型权重占用？**
   
   因为推理时还要保存每一层、每一个 token 的 K/V，这部分就是 KV Cache，长上下文和高并发时它可能非常大。

2. **为什么 vLLM 比普通 Hugging Face 推理更适合高并发服务？**
   
   因为 vLLM 不是简单为每个请求分配连续显存，而是用 block/page 方式动态管理 KV Cache，减少碎片和浪费。

3. **PagedAttention 到底解决了什么问题？**
   
   解决 KV Cache 动态增长、显存碎片、请求长度不确定、前缀复用等问题。

4. **BlockManager、Scheduler、ModelRunner、Attention 之间怎么配合？**
   
   Scheduler 决定什么时候运行请求，BlockManager 管理 KV block，ModelRunner 准备运行时数据，Attention 层真正使用 block table 和 slot_mapping 完成 KV Cache 读写。

5. **prefix caching 为什么只能复用前缀？**
   
   因为 Transformer 是自回归结构，某个位置的表示依赖它之前的全部上下文。只有前缀完全相同，后续层的 K/V 才能保证一致。

---

# 2. 按时间进度梳理知识点

原始转写文件没有稳定、清晰的时间戳，因此这里按照课程自然推进顺序分成若干部分。

---

## 第 1 部分：从 KV Cache 显存瓶颈讲起

### 本段核心知识点

课程开头先指出：大模型推理时，每一个 token 的推理都需要依赖前面所有 token 的 Key / Value。为了避免重复计算，推理引擎会把历史 token 的 K/V 缓存起来，这就是 **KV Cache**。

### 老师想表达什么

老师想强调：

> KV Cache 不是一个可有可无的小优化，而是大模型推理能够高效进行的基础。但是它同时也是推理显存占用的主要来源之一。

在 decode 阶段，当前新 token 只需要重新计算自己的 Q/K/V，但它的 Q 需要和历史所有 token 的 K 做 attention，然后对历史 V 加权求和。如果每次都重新算历史 K/V，代价会非常高，所以必须缓存。

### 相关概念解释

#### 1. KV Cache 是什么

Transformer Attention 中，每个 token 会产生：

```text
Q = Query
K = Key
V = Value
```

在自回归推理中，第 t 个 token 只能看见第 1 到第 t 个 token。decode 生成第 t+1 个 token 时，前 t 个 token 的 K/V 已经算过了，可以缓存下来。

所以 KV Cache 保存的是：

```text
每一层 Transformer
每一个历史 token
对应的 Key 向量和 Value 向量
```

#### 2. 为什么 KV Cache 占显存

KV Cache 的大小大致与以下因素成正比：

```text
2 × num_layers × sequence_length × num_kv_heads × head_dim × dtype_size
```

其中：

- `2`：因为要保存 K 和 V；
- `num_layers`：每一层 Attention 都有自己的 K/V；
- `sequence_length`：上下文越长，历史 token 越多；
- `num_kv_heads`：KV head 数量；
- `head_dim`：每个 head 的维度；
- `dtype_size`：例如 FP16/BF16 通常是 2 bytes。

当请求数很多时，总 KV Cache 还要乘以并发请求数。

### 在 AI Infra 中的作用

这部分知识对应推理系统的显存建模。做 vLLM / nano-vLLM / 推理服务部署时，不能只看模型权重大小，还必须考虑：

```text
总显存 ≈ 模型权重显存 + KV Cache 显存 + 临时计算显存 + 框架开销
```

这也是为什么同一个模型在短上下文、低并发下可以跑，但长上下文、高并发时会 OOM。

---

## 第 2 部分：普通 KV Cache 分配的问题

### 本段核心知识点

老师接着讲到，如果为每个请求直接申请连续 KV Cache 显存，会出现两个问题：

1. 显存浪费；
2. 显存碎片化。

### 老师想表达什么

请求的最终生成长度是不确定的。一个请求可能生成 10 个 token 就结束，也可能生成几千、几万 token。如果一开始为它预留最大长度的连续 KV Cache，会浪费大量显存；如果动态申请和释放连续显存，又会导致碎片化。

### 相关概念解释

#### 1. 显存浪费

假设系统为了保险，给每个请求预留 4096 个 token 的 KV Cache 空间，但某个请求只生成 100 个 token，那么剩下的空间就浪费了。

在高并发推理中，这种浪费会非常严重。

#### 2. 显存碎片化

如果不同请求不断进入和结束，显存中会出现很多小空洞。即使总空闲显存够，也可能因为没有足够大的连续空间而无法分配。

这类似操作系统中的内存碎片问题。

### 在推理系统中的作用

推理引擎必须解决这样的问题：

```text
请求长度未知
请求到达时间不同
请求结束时间不同
KV Cache 动态增长
GPU 显存有限
```

因此，推理引擎需要一种更灵活的 KV Cache 管理方式，这就是 PagedAttention 的动机。

---

## 第 3 部分：PagedAttention 的基本思想

### 本段核心知识点

PagedAttention 的核心思想是：

> 把 KV Cache 显存切成固定大小的 block，每个请求不再持有一整块连续显存，而是通过 block table 记录自己使用了哪些 block。

这非常类似操作系统中的分页机制。

### 老师想表达什么

老师用操作系统的 page / page table / process 类比 PagedAttention：

| 操作系统 | PagedAttention / nano-vLLM |
|---|---|
| Page 页 | KV Cache Block |
| Page Table 页表 | Block Table |
| Process 进程 | Sequence / Request |
| 物理内存 | GPU KV Cache Tensor |
| 虚拟地址到物理地址映射 | token 位置到 KV block 位置映射 |

也就是说，PagedAttention 的重点不是 Attention 公式变了，而是 **Attention 使用 KV Cache 的存储方式变了**。

### 相关概念解释

#### 1. Block

Block 是 KV Cache 管理的最小单位。例如 nano-vLLM 中默认一个 block 可以容纳 256 个 token 的 K/V。

一个请求如果有 600 个 token，可能需要：

```text
Block 0：存 token 0~255
Block 1：存 token 256~511
Block 2：存 token 512~599
```

#### 2. Block Table

Block table 是一个列表，记录某个 sequence 的逻辑 block 对应哪个物理 KV Cache block。

例如：

```text
sequence 的逻辑 block：0   1   2
实际物理 block id：      5   8   3
```

那么该 sequence 的 `block_table = [5, 8, 3]`。

#### 3. 按需分配

请求开始时，不需要一次性申请最大长度空间，而是根据当前 token 数量分配需要的 block。decode 过程中如果当前 block 满了，再申请新 block。

#### 4. 动态回收

请求生成结束后，释放它占用的 block，放回 free block 池，供其他请求使用。

#### 5. 共享复用

如果两个请求有完全相同的前缀，这些前缀对应的 KV Cache block 可以共享，不需要重复存储。这就是 prefix caching。

### 在 AI Infra 中的作用

PagedAttention 是 vLLM 提高吞吐量和显存利用率的关键机制之一。它让推理服务能够：

- 支持更多并发请求；
- 减少显存碎片；
- 避免为短请求浪费长上下文空间；
- 支持前缀缓存复用；
- 配合 continuous batching 做动态调度。

---

## 第 4 部分：nano-vLLM 中相关模块分工

### 本段核心知识点

老师开始把 PagedAttention 的思想对应到 nano-vLLM 的源码模块：

```text
BlockManager：负责 KV Cache block 的申请、释放、复用、hash 管理
Scheduler：负责调度 sequence，并决定何时申请/释放 block
ModelRunner：负责准备模型输入和 KV Cache 张量
Attention：负责真正写入和读取 KV Cache
Context：负责把本轮推理需要的 block table、slot mapping 等信息传给 attention
Sequence：表示一个请求，保存 token_ids、block_table、状态等信息
```

### 老师想表达什么

PagedAttention 不是某一个文件单独完成的，而是多个模块配合完成的：

```text
请求进入系统
  ↓
Sequence 保存请求状态
  ↓
Scheduler 选择本轮要运行哪些 Sequence
  ↓
BlockManager 为 Sequence 分配 block
  ↓
ModelRunner 准备 input_ids / positions / slot_mapping / block_tables
  ↓
Context 把运行时信息暴露给 Attention
  ↓
Attention 写入和读取 KV Cache
```

### 在系统中的作用

这部分是从“概念”进入“源码架构”的关键。你学完这节课后，应该知道每个文件的定位：

- `block_manager.py` 是内存管理器；
- `scheduler.py` 是请求调度器；
- `sequence.py` 是请求状态对象；
- `model_runner.py` 是模型执行器；
- `attention.py` 是真正使用 KV Cache 的模型层；
- `context.py` 是运行时上下文桥梁。

---

## 第 5 部分：Attention 中如何写入和读取 KV Cache

### 本段核心知识点

老师重点讲了 `attention.py` 中的 KV Cache 写入逻辑，尤其是 `store_kvcache` / `store_kvcache_kernel` 相关机制。

核心结论：

> Attention 层会先计算当前 token 的 K/V，然后根据 `slot_mapping` 把它们写入全局 KV Cache Tensor 中的正确位置。

### 老师想表达什么

在 PagedAttention 中，sequence 的 KV Cache 不是连续存储的，所以 Attention 层不能简单地说“把第 i 个 token 写到第 i 个位置”。它必须通过调度器和 block manager 提供的映射信息，计算真实物理位置。

### 相关概念解释

#### 1. slot_mapping 是什么

`slot_mapping` 告诉 Attention：

```text
当前这批输入 token 计算出来的 K/V，应该写到 KV Cache 的哪个全局 slot 位置。
```

例如：

```text
slot_mapping = [48, 49, 50, 51]
```

表示当前 4 个 token 的 K/V 分别写到全局 KV Cache 的第 48、49、50、51 个 token slot。

这里的 slot 是整个 KV Cache 大 tensor 中的全局位置，不只是某个 block 内部的相对位置。

#### 2. block_table 是什么

`block_table` 告诉 Attention：

```text
某个 sequence 的第 0、1、2... 个逻辑 block 分别对应哪些物理 block。
```

decode 时，Attention 需要根据 `block_table` 找到历史 token 的 K/V。

#### 3. prefill 和 decode 中 Attention 的区别

Prefill 阶段：

```text
处理 prompt 的一段 token
计算这一段 token 的 Q/K/V
把 K/V 写入 KV Cache
对 prompt 内部做 causal attention
可能还需要读取 prefix cache 命中的历史 block
```

Decode 阶段：

```text
每个 sequence 只输入当前最后一个 token
计算这个新 token 的 Q/K/V
把新 K/V 写入 KV Cache
当前 Q 通过 block_table 读取完整历史 K/V
得到当前 token 的输出 hidden state
```

### 在 AI Infra 中的作用

这部分解释了你之前在 `engine` 中学到的 block table 和 slot mapping 最终在哪里发挥作用：

```text
BlockManager 只是管理映射
ModelRunner 只是准备映射
Attention 才是真正使用映射读写 KV Cache 的地方
```

如果只看 `engine`，你会知道“有 block table”；看完这里，你才知道“block table 最终让 Attention 找到历史 K/V”。

---

## 第 6 部分：Context 的作用

### 本段核心知识点

老师提到 `Context` 是一个全局运行时上下文，用来保存本轮模型执行所需的信息，例如：

- 当前是否是 prefill；
- `cu_seqlens_q`；
- `cu_seqlens_k`；
- `max_seqlen_q`；
- `max_seqlen_k`；
- `slot_mapping`；
- `context_lens`；
- `block_tables`。

### 老师想表达什么

模型的 Attention 层本身不直接知道 Scheduler 调度了哪些 sequence，也不知道它们的 block table。ModelRunner 在执行前把这些运行时信息放进 Context，Attention forward 时再从 Context 中取出来。

### 相关概念解释

#### 1. 为什么需要 Context

Attention 层需要知道：

```text
当前是 prefill 还是 decode？
当前 token 的 K/V 写到哪里？
每个 sequence 的历史 KV block 在哪里？
每个 sequence 的上下文长度是多少？
```

这些信息不是模型权重的一部分，而是每一轮推理动态变化的运行时信息，所以需要一个 Context 传递。

#### 2. cu_seqlens_q / cu_seqlens_k 是什么

当一个 batch 中包含多个不同长度的 sequence 时，FlashAttention 需要知道每个 sequence 在扁平化 token tensor 中的边界。

例如两个 sequence 长度分别是 10 和 100，则累积长度可能是：

```text
cu_seqlens = [0, 10, 110]
```

表示：

```text
第 1 个 sequence 在 [0, 10)
第 2 个 sequence 在 [10, 110)
```

这用于 variable-length attention。

---

## 第 7 部分：Block 和 BlockManager 的作用

### 本段核心知识点

老师接着讲 `Block` 和 `BlockManager`。

`Block` 保存一个 KV block 的元信息：

- `block_id`：物理 block 编号；
- `ref_count`：引用计数；
- `hash`：当前 block 对应 token 前缀的哈希；
- `token_ids`：这个 block 对应的 token ids。

`BlockManager` 负责：

- 分配 block；
- 释放 block；
- 检查是否有足够空间；
- 根据 hash 做 prefix cache；
- 更新引用计数；
- 维护 free block 和 used block。

### 老师想表达什么

BlockManager 类似操作系统内存管理器，它不直接执行模型计算，但它决定 KV Cache 显存能否被高效使用。

### 相关概念解释

#### 1. free_block_ids

空闲 block 池。新请求需要空间时，从这里取 block。

#### 2. used_block_ids

当前正在被使用的 block 集合。

#### 3. hash_to_block_id

用于 prefix caching。它保存：

```text
某段 token 前缀的 hash → 对应的物理 block id
```

如果新请求的某些前缀 block 和已有 block 完全一样，就可以复用。

#### 4. ref_count

引用计数。一个 block 可能被多个 sequence 共享。

例如两个请求前缀相同，它们共享前两个 block。那么这两个 block 的 `ref_count = 2`。当其中一个请求结束时，不能直接释放这些 block，只能把引用计数减 1。只有 `ref_count == 0` 时，才能真正释放。

---

## 第 8 部分：Sequence 与 block_table

### 本段核心知识点

老师说明，一个 request 在 nano-vLLM 内部对应一个 `Sequence`。

Sequence 保存：

- token 序列；
- prompt token 数；
- completion token 数；
- 当前状态；
- block_table；
- 采样参数；
- 已缓存 token 数；
- 本轮调度 token 数。

### 老师想表达什么

Sequence 是请求在推理引擎中的运行时实体。它不是简单的一串 token，而是带有状态、缓存、调度信息的对象。

### block_table 的例子

假设一个 sequence 有 3 个逻辑 block：

```text
逻辑 block 0 → 物理 block 5
逻辑 block 1 → 物理 block 8
逻辑 block 2 → 物理 block 3
```

那么：

```text
seq.block_table = [5, 8, 3]
```

Attention 通过这个表，就能找到该 sequence 的历史 K/V 分别存在全局 KV Cache 的哪些物理 block 中。

---

## 第 9 部分：prefix caching 为什么只能复用前缀

### 本段核心知识点

老师提出一个重要问题：

> 如果两个 sequence 的第 0 个 block 和第 2 个 block 相同，但第 1 个 block 不同，能不能只复用第 0 和第 2 个 block？

答案是：**不能。prefix caching 只能复用连续前缀。**

### 老师想表达什么

Transformer 是自回归结构，一个 token 的 hidden state / K/V 不只取决于这个 token 本身，还取决于它前面的所有 token。

因此，即使第 2 个 block 的 token ids 表面上相同，只要它前面的上下文不同，这个 block 在模型每一层中计算出来的 K/V 就可能不同。

### 举例解释

假设有两个序列：

```text
序列 A：Block0 = 我爱你，Block1 = 中国，Block2 = 今天很好
序列 B：Block0 = 我爱你，Block1 = 母亲，Block2 = 今天很好
```

虽然 Block2 的 token ids 一样，但由于它前面的上下文不同，Block2 中 token 的 attention 结果不同，后续层的 K/V 也不同。

所以只能复用：

```text
从开头开始连续完全相同的 block
```

这就是 prefix caching 中 “prefix” 的含义。

### 在系统中的作用

这个问题非常适合面试，因为它考察的是你是否真正理解 Transformer 的上下文依赖，而不是只知道 hash 命中。

---

## 第 10 部分：prefill 阶段的完整内存流程

### 本段核心知识点

老师讲了 prefill 阶段如何处理 sequence：

1. Scheduler 遍历 waiting 队列；
2. 检查本轮 token 数是否超过 `max_num_batched_tokens`；
3. 检查 sequence 数是否超过 `max_num_seqs`；
4. 调用 BlockManager 判断能否分配 block；
5. 识别 prefix cache 命中的 block；
6. 为未命中的 token 分配新 block；
7. ModelRunner 准备 prefill 输入；
8. Attention 写入 KV Cache；
9. postprocess 更新 sequence 状态。

### 老师想表达什么

Prefill 不是简单地“跑一遍 prompt”。在推理引擎中，prefill 前必须先完成内存规划：

```text
哪些 token 已经有 cache？
哪些 token 需要重新计算？
需要几个新 block？
本轮最多能调度多少 token？
slot_mapping 怎么生成？
block_table 是否需要传给 attention？
```

### 在 nano-vLLM 中的意义

prefill 阶段主要完成两件事：

1. 为 prompt 建立 KV Cache；
2. 得到第一个生成 token 的 logits。

如果 prompt 很长，或者本轮 `max_num_batched_tokens` 不够，prefill 还可能被 chunked prefill 切成多轮。

---

## 第 11 部分：decode 阶段的完整内存流程

### 本段核心知识点

老师最后讲 decode：

1. Scheduler 从 running 队列中选择 sequence；
2. 判断当前 sequence 是否需要新 block；
3. 如果当前 block 已满，则申请新 block；
4. ModelRunner 准备 decode 输入；
5. 每个 sequence 只输入最后一个 token；
6. Attention 读取历史 KV Cache，写入当前新 K/V；
7. 模型输出 logits；
8. sampler 选出 next token；
9. postprocess 追加 token，并检查 EOS / max_tokens；
10. 如果完成，则释放 sequence 占用的 block。

### 老师想表达什么

Decode 阶段和 prefill 阶段最大的区别是：

```text
Prefill：一次处理一段 prompt token
Decode：每个 sequence 每轮通常只处理 1 个新 token
```

但是 decode 可以把多个 sequence 组成 batch，一起执行一次模型 forward，这就是 continuous batching 的基础。

---

# 3. 核心概念详细解释

## 3.1 KV Cache

KV Cache 是大模型推理时保存历史 Key / Value 的缓存。

### 是什么

在每一层 attention 中，每个 token 会产生 K/V。decode 时，历史 token 的 K/V 不需要反复计算，可以缓存。

### 为什么重要

没有 KV Cache，生成第 t 个 token 时要重新计算前 t-1 个 token 的 K/V，推理复杂度和延迟都会大幅增加。

### 在系统中起什么作用

KV Cache 是 decode 阶段能够高效逐 token 生成的基础。

### 和项目实践的关系

在 nano-vLLM 中，KV Cache 由 `ModelRunner.allocate_kv_cache()` 统一申请，Attention 层通过 block table 和 slot_mapping 读写它。

---

## 3.2 PagedAttention

PagedAttention 是一种 KV Cache 管理方式。

### 是什么

它把 KV Cache 切成固定大小的 block，并通过 block table 维护逻辑 token block 到物理 cache block 的映射。

### 为什么重要

它解决了长上下文和高并发场景下 KV Cache 显存浪费、碎片化和动态增长的问题。

### 在系统中起什么作用

它让每个请求不需要连续显存，只需要拿到若干 block，并通过 block table 找到它们。

### 和项目实践的关系

nano-vLLM 中的 `BlockManager`、`block_table`、`slot_mapping`、`attention.py` 都围绕这个机制工作。

---

## 3.3 Block

### 是什么

Block 是 KV Cache 分配的最小单位。

### 为什么重要

如果每个 token 都单独分配，管理成本太高；如果一次分配最大长度，又浪费显存。block 是两者之间的折中。

### 在系统中起什么作用

每个 sequence 的 token 被切成多个逻辑 block，每个逻辑 block 映射到一个物理 KV block。

---

## 3.4 Block Table

### 是什么

Block table 是 sequence 级别的映射表。

```text
逻辑 block index → 物理 block id
```

### 为什么重要

因为 sequence 的 KV Cache 在物理显存中不一定连续，必须通过 block table 找到历史 K/V。

### 在系统中起什么作用

Attention decode 时，通过 block table 读取该 sequence 的完整历史 KV Cache。

---

## 3.5 Slot Mapping

### 是什么

slot_mapping 记录当前这批 token 的 K/V 应该写到 KV Cache 的哪个全局 slot。

### 为什么重要

在 prefill / decode 中，当前新算出来的 K/V 必须准确写入 KV Cache，否则后续 attention 会读错历史上下文。

### 在系统中起什么作用

`ModelRunner.prepare_prefill()` 和 `ModelRunner.prepare_decode()` 会构造 slot_mapping，`Attention` 通过它写入 KV Cache。

---

## 3.6 Prefix Caching

### 是什么

Prefix caching 是指多个请求如果有相同前缀，就复用这部分前缀对应的 KV Cache。

### 为什么重要

在聊天、Agent、代码助手、系统 prompt 场景中，多个请求经常共享相同的 system prompt 或历史上下文。复用前缀可以减少计算和显存占用。

### 在系统中起什么作用

BlockManager 通过 hash 判断 block 是否命中缓存，并通过 ref_count 管理共享 block 的生命周期。

### 为什么只能复用前缀

因为 Transformer 中某个 token 的表示依赖它前面的所有 token。非连续位置即使 token ids 相同，只要前文不同，K/V 也不能保证相同。

---

## 3.7 Ref Count

### 是什么

ref_count 是引用计数，记录一个 block 被多少个 sequence 使用。

### 为什么重要

共享 block 不能在某一个 sequence 结束时立刻释放，否则其他 sequence 会读到无效 cache。

### 在系统中起什么作用

释放 block 时先 `ref_count -= 1`，只有变成 0 才真正回收到 free block 池。

---

## 3.8 Scheduler

### 是什么

Scheduler 是推理调度器，负责决定本轮运行哪些 sequence，是 prefill 还是 decode。

### 为什么重要

高并发推理不是一个请求跑完再跑下一个，而是多个请求动态组成 batch。Scheduler 决定 GPU 每一步干什么。

### 在系统中起什么作用

它维护 waiting / running 队列，调度 prefill 和 decode，并在必要时触发 block 分配、抢占、释放。

---

## 3.9 ModelRunner

### 是什么

ModelRunner 是模型执行器，负责把 Scheduler 选出来的 sequence 转成模型可以执行的 tensor 输入。

### 为什么重要

Scheduler 管逻辑，真正跑模型还需要 input_ids、positions、context_lens、block_tables、slot_mapping 等张量。

### 在系统中起什么作用

它连接调度层和模型层。

---

## 3.10 Attention

### 是什么

Attention 是 Transformer 的核心计算层。

### 为什么重要

KV Cache 最终是在 Attention 中被写入和读取的。

### 在系统中起什么作用

它使用当前 token 的 Q 查询历史 K/V，得到融合上下文后的 hidden state。

---

# 4. 整节课技术流程串联

下面用“从输入到输出”的方式串联整节课。

## 4.1 请求进入系统

用户输入 prompt，例如：

```text
请解释一下 PagedAttention
```

Tokenizer 把 prompt 转成 token ids：

```text
[101, 345, 778, ...]
```

nano-vLLM 为这个请求创建一个 Sequence。

Sequence 保存：

```text
token_ids
num_prompt_tokens
num_cached_tokens
num_scheduled_tokens
block_table
status = WAITING
sampling_params
```

---

## 4.2 Scheduler 选择 prefill 请求

请求一开始进入 waiting 队列。Scheduler 在一次 step 中遍历 waiting 队列，判断：

```text
本轮 batch 还能容纳多少 sequence？
本轮还能容纳多少 token？
该 sequence 需要多少 KV block？
是否有 prefix cache 命中？
当前 free block 是否足够？
```

如果资源足够，就调度这个 sequence 做 prefill。

---

## 4.3 BlockManager 分配 KV Cache block

BlockManager 为 sequence 分配 block。

如果没有 prefix cache：

```text
为 prompt token 对应的所有 block 申请新物理 block
```

如果有 prefix cache：

```text
前若干 block 直接复用已有 block
ref_count += 1
剩余未命中的 block 申请新 block
```

sequence 的 block_table 被填好。

---

## 4.4 ModelRunner 准备 prefill 输入

ModelRunner 根据 sequence 构造：

```text
input_ids：本轮要计算的 token ids
positions：这些 token 的位置编号
cu_seqlens_q / cu_seqlens_k：多个 sequence 的边界
slot_mapping：当前 token 的 K/V 写入位置
block_tables：prefix cache 或 decode 需要的历史 block 映射
```

然后调用 `set_context()`，把这些信息放入全局 Context。

---

## 4.5 Attention 执行 prefill

模型进入 Qwen3 的 decoder layer。

每层 Attention 会：

```text
hidden_states → q/k/v projection
  ↓
对 q/k 应用 RoPE 位置信息
  ↓
根据 slot_mapping 把 k/v 写入 KV Cache
  ↓
使用 FlashAttention 完成 causal attention
  ↓
输出新的 hidden_states
```

如果有 prefix cache，Attention 还会根据 block_table 读取已经缓存的历史 K/V。

---

## 4.6 模型输出 logits 并采样

经过多层 decoder 后，模型得到最后的 hidden state。

LM Head 把 hidden state 映射到词表大小的 logits：

```text
hidden_state → logits[vocab_size]
```

Sampler 根据 logits 和 temperature 得到下一个 token。

---

## 4.7 Scheduler postprocess

Scheduler 收到采样出来的 token 后：

```text
更新 block hash
更新 num_cached_tokens
把新 token append 到 Sequence
检查是否 EOS
检查是否达到 max_tokens
```

如果 sequence 完成：

```text
释放它占用的 KV block
从 running 队列移除
返回输出
```

如果没有完成：

```text
继续留在 running 队列，进入 decode 阶段
```

---

## 4.8 Decode 阶段继续生成

decode 阶段每轮通常每个 sequence 只处理一个 token。

Scheduler 从 running 队列中选择多个 sequence 组成 batch。

对于每个 sequence：

```text
检查当前 block 是否已满
如果满了，申请新 block
准备 last_token 作为 input_ids
准备 position
准备 context_lens
准备 block_table
准备 slot_mapping
```

Attention 会：

```text
为当前 token 计算新的 q/k/v
把新的 k/v 写入 KV Cache
用当前 q 通过 block_table 读取历史 k/v
计算 attention 输出
```

然后模型输出 logits，sampler 选择下一个 token，追加回 sequence。

这个过程反复执行，直到 EOS 或 max_tokens。

---

# 5. 我能从这节课中学到什么

## 5.1 概念层面

你应该掌握以下核心概念：

### 1. KV Cache

位置：Transformer Attention 推理中的历史 K/V 缓存。

作用：避免重复计算历史 token 的 K/V，是 decode 加速的基础。

### 2. PagedAttention

位置：KV Cache 显存管理机制。

作用：用 block + block table 管理 KV Cache，减少显存浪费和碎片。

### 3. Block

位置：KV Cache 的最小分配单位。

作用：让请求按需申请 KV Cache。

### 4. Block Table

位置：Sequence 内部维护的逻辑到物理映射表。

作用：让 Attention 能找到某个 sequence 的历史 KV Cache。

### 5. Slot Mapping

位置：ModelRunner 准备输入时生成，Attention 写 cache 时使用。

作用：告诉 Attention 当前新 K/V 写到哪里。

### 6. Prefix Caching

位置：BlockManager 中基于 hash 的缓存复用机制。

作用：复用相同前缀的 KV Cache，减少重复计算和显存占用。

### 7. Ref Count

位置：Block 元信息中。

作用：管理共享 block 的释放时机。

---

## 5.2 系统层面

你应该理解一个大模型推理系统至少包含这些模块：

```text
用户请求输入
  ↓
Tokenizer
  ↓
Sequence 创建
  ↓
Scheduler 调度
  ↓
BlockManager 分配 KV Cache
  ↓
ModelRunner 准备张量输入
  ↓
Context 传递运行时信息
  ↓
Transformer / Attention 执行
  ↓
Sampler 选择 token
  ↓
Scheduler 更新状态
  ↓
输出结果 / 继续 decode / 释放资源
```

其中每个模块的职责是：

| 模块 | 职责 |
|---|---|
| Tokenizer | 把文本转成 token ids |
| Sequence | 保存单个请求的运行状态 |
| Scheduler | 决定每一步运行哪些请求 |
| BlockManager | 管理 KV Cache block |
| ModelRunner | 准备模型输入并执行 forward |
| Context | 保存本轮 Attention 需要的动态信息 |
| Attention | 写入/读取 KV Cache 并计算 attention |
| Sampler | 从 logits 中选下一个 token |

---

## 5.3 工程层面

这节课对你后续做项目有直接帮助。

### 1. nano-vLLM 源码阅读

你会用到：

- `block_manager.py`：理解 block 分配、释放、prefix cache；
- `scheduler.py`：理解 prefill / decode 何时申请 block；
- `model_runner.py`：理解 slot_mapping、block_tables 怎么构造；
- `attention.py`：理解 KV Cache 真正如何读写；
- `context.py`：理解运行时上下文怎么传递。

### 2. vLLM Benchmark

你会用到：

- 为什么长 prompt 会增加 prefill 开销；
- 为什么长输出会增加 KV Cache 占用；
- 为什么 batch size 增大后显存线性增长；
- 为什么 `gpu_memory_utilization` 会影响可用 KV Cache blocks；
- 为什么 prefix cache 命中能降低 TTFT。

### 3. 推理服务部署

你会用到：

- 如何估算并发数和最大上下文长度；
- 如何理解 OOM 原因；
- 如何调整最大 batch token 数；
- 如何解释为什么高并发下需要 vLLM 而不是普通 transformers generate。

### 4. 性能优化

你会用到：

- KV Cache 显存是推理服务关键瓶颈；
- block 管理能提升显存利用率；
- prefix caching 能减少重复计算；
- continuous batching 能提高 GPU 利用率；
- chunked prefill 可以避免长 prompt 阻塞 decode。

---

# 6. 面试问题与回答思路

## 问题 1：什么是 KV Cache？为什么大模型推理需要 KV Cache？

回答思路：

KV Cache 是 Transformer decode 阶段保存历史 token 的 Key / Value 的缓存。自回归生成时，当前 token 的 Query 需要和历史 token 的 Key / Value 做 attention。如果每一步都重新计算历史 K/V，计算量非常大。KV Cache 让历史 K/V 只计算一次，后续 decode 直接读取，从而显著降低推理开销。

---

## 问题 2：KV Cache 显存大小和哪些因素有关？

回答思路：

KV Cache 大小主要与层数、序列长度、KV head 数量、head_dim、数据类型、并发请求数有关。大致公式是：

```text
2 × num_layers × sequence_length × num_kv_heads × head_dim × dtype_size × batch_size
```

其中 2 表示 K 和 V。

---

## 问题 3：PagedAttention 解决了什么问题？

回答思路：

PagedAttention 解决 KV Cache 显存管理问题。传统方式为每个请求分配连续 KV Cache，容易浪费和碎片化。PagedAttention 把 KV Cache 切成固定 block，通过 block table 做逻辑到物理映射，实现按需分配、动态回收和共享复用。

---

## 问题 4：PagedAttention 和操作系统分页有什么相似之处？

回答思路：

操作系统用 page 和 page table 管理虚拟地址到物理内存的映射。PagedAttention 用 KV block 和 block table 管理 sequence 的逻辑 token block 到 GPU 物理 KV Cache block 的映射。Sequence 类似进程，KV block 类似物理页。

---

## 问题 5：BlockManager 在 nano-vLLM 中负责什么？

回答思路：

BlockManager 负责 KV Cache block 的分配、释放、prefix cache 命中检查、hash 到 block id 的映射维护、引用计数管理。它是 nano-vLLM 中的 KV Cache 内存管理器。

---

## 问题 6：block_table 是什么？

回答思路：

block_table 是每个 sequence 的逻辑 block 到物理 KV Cache block 的映射。例如 sequence 的逻辑 block 0、1、2 分别存储在物理 block 5、8、3 中，则 block_table 是 `[5, 8, 3]`。Attention 通过它读取历史 K/V。

---

## 问题 7：slot_mapping 是什么？

回答思路：

slot_mapping 告诉 Attention 当前这批新 token 的 K/V 应该写入 KV Cache 的哪个全局 slot。它主要服务于 KV Cache 写入。block_table 主要服务于历史 KV Cache 读取。

---

## 问题 8：prefix caching 为什么只能复用前缀，而不能复用中间相同的片段？

回答思路：

Transformer 是自回归结构，某个 token 的 hidden state 和 K/V 依赖它之前的完整上下文。即使某个中间 block 的 token ids 相同，只要它前面的上下文不同，计算出来的 K/V 就可能不同。因此只能复用从开头开始连续完全相同的前缀。

---

## 问题 9：prefill 和 decode 在 KV Cache 使用上有什么区别？

回答思路：

prefill 阶段处理 prompt 的一段 token，计算并写入这些 token 的 K/V，同时得到第一个生成 token 的 logits。decode 阶段每个 sequence 每轮通常只处理一个新 token，计算当前 token 的 K/V 写入 KV Cache，并通过 block_table 读取历史 K/V 完成 attention。

---

## 问题 10：为什么需要 ref_count？

回答思路：

因为 prefix caching 会让多个 sequence 共享同一个 KV block。一个 sequence 结束时不能直接释放共享 block，否则其他 sequence 会读到无效缓存。ref_count 记录有多少 sequence 正在使用这个 block，只有 ref_count 变成 0 时才能真正释放。

---

## 问题 11：ModelRunner 在 PagedAttention 流程中起什么作用？

回答思路：

ModelRunner 连接调度层和模型执行层。Scheduler 选出本轮要运行的 sequence 后，ModelRunner 根据这些 sequence 构造 input_ids、positions、slot_mapping、context_lens、block_tables，并设置 Context，然后调用模型 forward。

---

## 问题 12：Attention 层怎么知道去哪里读写 KV Cache？

回答思路：

Attention 层从 Context 中读取 slot_mapping、block_tables、context_lens 等信息。slot_mapping 用于把当前 token 的 K/V 写入 KV Cache；block_tables 和 context_lens 用于 decode 时读取历史 K/V。

---

# 7. 和 nano-vLLM 源码文件的对应关系

| 课程概念 | nano-vLLM 文件 | 作用 |
|---|---|---|
| 请求状态 | `engine/sequence.py` | Sequence 保存 token、状态、block_table |
| 调度 | `engine/scheduler.py` | 选择 prefill / decode 请求，触发申请和释放 |
| KV block 管理 | `engine/block_manager.py` | 分配、释放、prefix cache、ref_count |
| 模型执行 | `engine/model_runner.py` | 准备 input_ids、positions、slot_mapping、block_tables |
| 上下文传递 | `utils/context.py` | 把本轮动态信息传给 Attention |
| KV Cache 读写 | `layers/attention.py` | 写入当前 K/V，读取历史 K/V |
| 输出采样 | `layers/sampler.py` | 从 logits 选择 next token |
| 模型结构 | `models/qwen3.py` | 组织 embedding、decoder layer、attention、MLP、lm_head |

---

# 8. 学习建议与复习路线

## 8.1 你应该先掌握的主线

建议你先把下面这条线背熟：

```text
Sequence 表示请求
Scheduler 选择本轮运行哪些 Sequence
BlockManager 给 Sequence 分配 KV Cache block
Sequence.block_table 记录逻辑 block 到物理 block 的映射
ModelRunner 根据 block_table 生成 slot_mapping 和 block_tables
Context 把这些信息传给 Attention
Attention 根据 slot_mapping 写入新 K/V
Attention 根据 block_tables 读取历史 K/V
Sampler 选择 next token
Scheduler 更新 Sequence 状态并在结束时释放 block
```

---

## 8.2 复习时重点看哪些问题

你复习这节课时，建议围绕下面几个问题：

1. 为什么 KV Cache 会成为显存瓶颈？
2. 为什么普通连续显存分配不适合高并发推理？
3. PagedAttention 如何借鉴操作系统分页？
4. block、block_table、slot_mapping 分别是什么？
5. prefix caching 为什么只能复用前缀？
6. prefill 和 decode 分别什么时候申请 block？
7. Attention 层如何读写 KV Cache？
8. 一个请求结束后，KV block 如何释放？

---

## 8.3 和你后续秋招项目的关系

如果你要把 nano-vLLM 写到简历里，可以重点突出：

```text
深入阅读 nano-vLLM 推理引擎核心模块，理解 PagedAttention 风格的 KV Cache block 管理机制，掌握 Sequence、Scheduler、BlockManager、ModelRunner 与 Attention 层之间的数据流关系，能够解释 prefill/decode 阶段 KV Cache 的申请、写入、读取、复用与释放流程。
```

这句话比泛泛写“学习 nano-vLLM 源码”更有含金量，因为它点出了推理引擎最核心的机制。

---

# 9. 本节课一句话总结

这节课的核心可以概括为：

> 大模型推理中，KV Cache 是 decode 加速的关键，但也是显存瓶颈；PagedAttention 通过把 KV Cache 切成 block，并用 block table 像操作系统页表一样管理映射，实现了按需分配、动态回收和前缀共享，从而支撑高并发、长上下文的大模型推理服务。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]

%% 项目关联导航：结束 %%
