# vLLM KV Cache 显存容量计算与初始化分配：课程深度学习笔记

> 本节课主题：在上一节“KV Cache 分块显存管理”的基础上，继续讲 vLLM 在真正启动推理服务时，**如何确定 KV Cache 可以使用多少显存、能切出多少个 block、每个 block 多大、这些 KV Cache 最终挂在哪里**。

---

## 0. 语音转写中的术语校正

视频转文字里有不少自动识别错误，学习时建议先统一成下面这些术语：

| 转写中可能出现的词 | 正确术语 | 含义 |
|---|---|---|
| VLM / VRM | vLLM | 大模型推理框架 |
| Promote / ProMode | Prompt | 用户输入提示词 |
| Blog / Bolg / BlogSize / Bolishize | Block / block_size | KV Cache 的分页块，以及每块能容纳的 token 数 |
| KVCatch / TVCatch / KVH | KV Cache | Attention 中缓存历史 token 的 Key / Value |
| NumberHeads / NumbKVHead | num_kv_heads | KV head 数量，GQA/MQA 下通常小于 attention head 数 |
| HeadSize / Dimension | head_size | 每个 head 的向量维度 |
| Ditip | dtype | 数据类型，如 FP16、BF16 |
| FreeBlock Queen | FreeBlockQueue | 空闲物理 block 队列 |
| ModelRona / model wrong | ModelRunner | 负责执行模型 forward 的组件 |
| RSHIP | reshape | 改变张量形状，以适配 Attention kernel |
| Xformer | xFormers | 一种 Attention 加速后端 |
| 张亮并行 | Tensor Parallel, TP | 张量并行，多 GPU 切分模型计算 |

---

## 1. 任务一：这节课的核心宗旨

这节课的核心目标是回答一个非常实际的问题：

> vLLM 知道要用分页方式管理 KV Cache，但 GPU 上到底能分出多少个 KV Cache block？这些 block 的大小怎么计算？它们最终存在模型的什么位置？

上一节课讲的是“为什么要分块”：传统连续 KV Cache 会按最大上下文长度提前申请一大段显存，哪怕请求只用了很短的 token，也会长期占住大块空间，导致显存浪费、内部碎片和 batch 能力下降。

本节课则从工程实现角度继续往下走：

1. vLLM 先根据 `gpu_memory_utilization` 得到用户允许使用的 GPU 显存上限。
2. 再扣除模型权重显存。
3. 对推理过程做一次模拟运行，估算中间激活值、临时张量等非 KV Cache 开销。
4. 剩下的显存才可以作为 KV Cache 池。
5. 根据每个 block 的字节大小，计算总共能切出多少个物理 block。
6. 将这些 block 初始化到空闲队列中，后续由调度器按需分配给不同请求。
7. KV Cache 实际上不是只有一份，而是每个 Attention 层都有独立的 KV Cache 空间，并挂在对应的 `ModelRunner` 上。

所以，这节课不是只讲公式，而是在解释 vLLM 启动阶段的一条关键链路：

```text
GPU 总显存
  ↓
乘以 gpu_memory_utilization 得到允许使用的显存预算
  ↓
扣除模型权重
  ↓
通过 profile / dummy run 估计非 KV Cache 临时开销
  ↓
得到 KV Cache 可用显存
  ↓
计算每个 block 的大小
  ↓
计算 num_blocks
  ↓
为每层 Attention 分配 KV Cache 张量
  ↓
reshape 成 Attention kernel 需要的布局
  ↓
挂到 ModelRunner，供调度和推理阶段使用
```

它在 AI Infra / 推理系统学习中的位置非常重要：

- 它连接了 **模型结构参数** 与 **GPU 显存管理**；
- 它解释了 vLLM 为什么能比普通推理脚本承载更多并发请求；
- 它是理解 PagedAttention、BlockManager、Scheduler、ModelRunner 的基础；
- 它直接对应推理系统面试中常见的“KV Cache 显存怎么算”“vLLM 如何确定最大并发能力”“为什么要 profile run”等问题。

---

## 2. 任务二：按照课程推进顺序梳理知识点

> 原始转写文件没有严格时间戳，因此这里按照老师讲解的先后顺序整理。

---

### 2.1 开头：回顾上一节的分块显存管理

#### 本段核心知识点

上一节已经讲过，vLLM 不再为每个请求一次性申请一整段连续 KV Cache，而是将 KV Cache 显存切分成多个固定大小的 block。每个 block 能存放固定数量 token 的 K/V。

例如：

```text
block_size = 16
```

这不是说 block 只有 16 字节，也不是说里面存 16 个 token id，而是说：

> 这个 block 能为 16 个 token 存放它们在各 Attention 层里的 K/V 张量。

如果一个 prompt 被切成 3 个逻辑块，就只需要给它分配 3 个物理 block；如果后续 decode 又生成了更多 token，当前 block 满了，再继续申请新 block。

#### 相关概念解释

**传统连续 KV Cache 管理：** 可能会按照 `max_seq_len` 或最大上下文长度提前申请。例如最大长度 4096，那么一个请求来了就直接为它预留 4096 个 token 的 KV Cache 空间。即使这个请求最后只用了几十个 token，剩余大部分空间也不能给其他请求使用。

**vLLM 分块管理：** 将长序列拆成多个 block。请求需要多少，就申请多少。它类似操作系统中的分页内存管理。

**内部碎片：** 分块管理仍然有浪费。例如 `block_size = 16`，最后一个 block 只用了 1 个 token，那么这个 block 剩下 15 个 token 的 KV Cache 空间暂时浪费。但浪费上限是 `block_size - 1` 个 token 的空间，远小于传统连续分配可能浪费几千个 token 的空间。

#### 在 AI Infra 中的作用

分块管理直接提升推理服务的显存利用率和并发能力。因为多个请求可以共享同一个全局 KV Cache block 池，每个请求按需申请，不必一开始占满最大上下文长度。

---

### 2.2 第一部分：GPU 总显存不能全部拿来存 KV Cache

#### 本段核心知识点

老师接着讲：系统显存要先分成多个部分。GPU 有 32GB 显存，不代表这 32GB 都能给 vLLM 的 KV Cache 使用。

通常会有一个参数：

```text
gpu_memory_utilization
```

它表示 vLLM 被允许使用 GPU 总显存的比例。

例如：

```text
GPU 总显存 = 32GB
gpu_memory_utilization = 0.8
vLLM 允许使用的显存预算 = 32GB × 0.8 = 25.6GB
```

视频转写里出现了“225.6G”，这里应理解为语音识别错误，正确结果是 **25.6GB**。

#### 相关概念解释

**GPU 总显存：** 显卡物理显存容量，例如 24GB、32GB、80GB。

**gpu_memory_utilization：** 推理框架允许自己占用的显存比例。它不是越大越好，因为还要给 CUDA runtime、通信库、临时 buffer、其他进程留空间。

**显存预算：**

```text
allowed_memory = total_gpu_memory × gpu_memory_utilization
```

这个预算包括：

```text
模型权重显存 + 推理过程临时开销 + KV Cache 显存
```

#### 在 AI Infra 中的作用

这个参数决定了推理服务的容量上限。设置过低，会导致可用 KV Cache block 少，最大并发和长上下文能力下降；设置过高，可能导致 OOM，因为系统或 CUDA 临时分配没有余量。

---

### 2.3 第二部分：模型权重显存可以预估

#### 本段核心知识点

KV Cache 可用显存不是直接等于 `allowed_memory`。首先要扣掉模型权重。

模型权重大小通常是可以预估的，因为模型加载前就知道参数量、dtype 和分片方式。

例如：

```text
模型参数量 = 7B
权重 dtype = FP16
单参数大小 = 2 bytes
理论权重大小 ≈ 7B × 2 bytes = 14GB
```

实际显存还会受到模型结构、padding、分布式切分、框架额外开销影响，但大体可以估计。

#### 相关概念解释

**模型权重：** Transformer 中的所有参数，例如 Embedding、Attention 的 Q/K/V/O 矩阵、MLP 的线性层参数、LayerNorm 参数等。

**dtype：** 数据类型。常见有 FP16、BF16、FP32、INT8、INT4。dtype 越小，权重和 KV Cache 占用越低。

**权重显存估算：**

```text
weight_memory ≈ 参数量 × 每个参数字节数
```

例如 FP16 / BF16 一般是 2 bytes，FP32 是 4 bytes。

#### 在 AI Infra 中的作用

模型权重是推理服务中的固定显存开销。模型越大，剩给 KV Cache 的显存越少；KV Cache 越少，可同时服务的请求数、长上下文能力和 batch 能力就越弱。

---

### 2.4 第三部分：临时激活值和中间张量不能直接手算，vLLM 用模拟执行估算

#### 本段核心知识点

除了模型权重，推理过程中还会产生一些临时显存开销，例如：

- forward 中间激活值；
- Attention 计算临时 buffer；
- CUDA kernel workspace；
- logits、采样相关临时张量；
- 框架自身的缓存和运行时开销。

这些开销很难完全靠公式静态计算，因此 vLLM 会做一次模拟运行，也可以理解为 profile run / dummy run。

流程大致是：

```text
加载模型权重后，记录当前显存占用
  ↓
跑一次模拟 forward
  ↓
再次记录显存占用或峰值显存
  ↓
两者差值近似表示非 KV Cache 临时开销
```

#### 相关概念解释

**激活值 activation：** 模型 forward 过程中每层产生的中间输出。在训练时激活值要保存用于反向传播；在推理时不需要保存全部历史激活，但仍会有当前 step 的中间张量和临时 buffer。

**profile run / dummy run：** 用虚拟输入或最大可能形状运行一次模型，观察真实显存峰值，而不是凭经验猜。

**为什么需要模拟执行：** 因为不同模型、不同 batch、不同并发、不同 attention backend、不同 CUDA kernel 都会影响临时显存。纯手算容易不准。

#### 在 AI Infra 中的作用

这一步决定了 vLLM 能安全分配多少 KV Cache。分少了，浪费显存；分多了，真实推理时可能 OOM。所以 vLLM 通过 profile run 在“尽量吃满显存”和“避免运行时爆显存”之间取平衡。

---

### 2.5 第四部分：KV Cache 可用显存的计算公式

#### 本段核心知识点

结合前面几部分，可以得到：

```text
allowed_memory = total_gpu_memory × gpu_memory_utilization
available_kv_cache_memory = allowed_memory - weight_memory - non_kv_runtime_memory
```

其中：

```text
non_kv_runtime_memory
```

主要由 profile run 估算，包括临时激活值、中间张量和 kernel workspace 等。

#### 更直观的例子

假设：

```text
GPU 总显存 = 32GB
gpu_memory_utilization = 0.8
allowed_memory = 25.6GB
模型权重占用 = 14GB
profile run 估计非 KV 临时开销 = 2GB
```

那么：

```text
available_kv_cache_memory = 25.6GB - 14GB - 2GB = 9.6GB
```

这 9.6GB 才是后续可以切成 KV Cache block 的空间。

#### 在 AI Infra 中的作用

KV Cache 可用显存直接决定系统容量。推理服务中常见的“最大并发数”“最大上下文长度”“能不能承载长 prompt”最终都和这个值强相关。

---

### 2.6 第五部分：每个 KV Cache block 的大小怎么计算

#### 本段核心知识点

老师接着讲：知道 KV Cache 总空间后，还要知道每个 block 多大，才能算出 `num_blocks`。

单层 Attention 中，一个 block 的 KV Cache 大小可以理解为：

```text
block_bytes_per_layer
= block_size × num_kv_heads × head_size × 2 × dtype_size
```

其中：

| 参数 | 含义 |
|---|---|
| `block_size` | 一个 block 能容纳多少个 token |
| `num_kv_heads` | KV head 数量 |
| `head_size` | 每个 head 的维度 |
| `2` | K 和 V 两份缓存 |
| `dtype_size` | 每个元素占多少字节，例如 FP16 为 2 bytes |

如果考虑所有 Transformer 层，则一个“全模型意义上的物理 block”总成本是：

```text
total_block_bytes
= num_layers × block_size × num_kv_heads × head_size × 2 × dtype_size
```

因为每个 token 经过每一层 Attention 时，都会产生这一层自己的 K/V cache。

#### 重要理解

很多初学者容易误会：

> 一个 token 只有一份 KV Cache。

更准确地说：

> 一个 token 在每个 Attention 层都会有一份 K 和一份 V。

所以如果模型有 32 层，那么同一个 token 会在 32 个 Attention 层中分别留下 K/V 缓存。

#### 数值例子

假设：

```text
block_size = 16
num_kv_heads = 8
head_size = 128
dtype = FP16, dtype_size = 2 bytes
num_layers = 32
```

单层一个 block 大小：

```text
16 × 8 × 128 × 2 × 2 bytes = 65,536 bytes = 64KB
```

所有 32 层合起来：

```text
64KB × 32 = 2048KB = 2MB
```

所以在这个假设下，一个 block 编号从全模型角度看，大约会消耗 2MB 的 KV Cache 显存。

如果可用 KV Cache 显存是 9.6GiB，则大约可以分出：

```text
9.6GiB ÷ 2MiB ≈ 4915 个 block
```

这些 block 可以容纳的 token block 容量约为：

```text
4915 × 16 ≈ 78,640 个 token 位置
```

注意：这个 token 位置不是给一个请求独占，而是由所有并发请求共享。

---

### 2.7 第六部分：num_blocks 与 FreeBlockQueue

#### 本段核心知识点

算出每个 block 的大小后，就可以计算：

```text
num_blocks = available_kv_cache_memory // total_block_bytes
```

得到的 `num_blocks` 表示当前 GPU KV Cache 池里最多能切出多少个物理 block。

这些 block 初始化时会放入一个空闲队列：

```text
FreeBlockQueue
```

当某个请求需要 KV Cache 空间时，调度器从空闲队列中取出 block；请求结束后，再把 block 归还给空闲队列。

#### 相关概念解释

**物理 block：** GPU 显存里真实存在的一块 KV Cache 存储区域。

**逻辑 block：** 某个 sequence 在语义上按 token 顺序切出来的第 0 块、第 1 块、第 2 块等。

**block table：** 用来记录逻辑 block 到物理 block 的映射。例如：

```text
sequence A logical block 0 → physical block 17
sequence A logical block 1 → physical block 203
sequence A logical block 2 → physical block 88
```

逻辑上连续，物理上可以不连续。

#### 在 AI Infra 中的作用

这是 vLLM 能够像操作系统分页一样管理 KV Cache 的核心。它让请求不再需要连续大块显存，降低碎片，提高 GPU 显存复用率。

---

### 2.8 第七部分：KV Cache 和 ModelRunner 的关系

#### 本段核心知识点

老师强调：KV Cache 是和 `ModelRunner` 相关的。

`ModelRunner` 可以理解为：

> 负责在某个设备上执行模型 forward 的执行器。

在单 GPU 场景下，通常有一个主要的 `ModelRunner`。在 Tensor Parallel 多 GPU 场景下，每个设备上都会有自己的 `ModelRunner`，每个 `ModelRunner` 也会维护自己那一份 KV Cache。

#### 相关概念解释

**ModelRunner：** 推理框架中的模型执行组件。调度器决定本轮要跑哪些请求后，实际的 forward 通常由 ModelRunner 完成。

**KV Cache 挂在 ModelRunner 上：** 也就是 ModelRunner 在执行 forward 时可以直接访问对应设备上的 KV Cache 张量。

**TP 场景：** 如果 `tensor_parallel_size = 4`，模型被切到 4 张 GPU 上，每张 GPU 只保存一部分权重，也需要保存这一部分计算对应的 KV Cache。因此每个设备上的 ModelRunner 都有独立 KV Cache。

#### 在 AI Infra 中的作用

这说明 KV Cache 不是一个抽象列表，而是真实和模型执行设备绑定的 GPU Tensor。理解这一点后，才能理解为什么 TP、PP、不同 Attention backend 都会影响 KV Cache 的形状和布局。

---

### 2.9 第八部分：KV Cache 是按 Attention 层分配的

#### 本段核心知识点

视频中特别提到：一个模型有多个 Attention 层，每一层都有独立的 KV Cache 空间。

例如一个简化版 LLaMA 模型有 4 个 Attention 层，那么就会有 4 份 KV Cache 张量。真实大模型可能有 24 层、32 层、40 层、80 层等。

常见形状可以抽象为：

```text
每层 KV Cache shape ≈ [num_blocks, 2, block_size, num_kv_heads, head_size]
```

有些实现或 Attention backend 可能使用：

```text
[2, num_blocks, block_size, num_kv_heads, head_size]
```

或者其他布局。维度顺序可能不同，但核心维度一定包括：

```text
num_blocks, K/V 两份, block_size, num_kv_heads, head_size
```

#### 相关概念解释

**为什么每层都有 KV Cache？**

Transformer 的每一层 Attention 都会基于当前层输入计算 K 和 V。第 1 层的 K/V 和第 20 层的 K/V 是不同的张量，不能共用。

**为什么有 `2` 这个维度？**

因为缓存里有两类内容：

```text
K cache
V cache
```

它们都需要保存历史 token 的信息。

**为什么是 `num_kv_heads` 而不是一定等于 `num_attention_heads`？**

很多现代模型使用 GQA 或 MQA。此时 Query head 数可能较多，而 Key/Value head 数更少。KV Cache 只缓存 K/V，所以受 `num_kv_heads` 影响。

#### 在 AI Infra 中的作用

这部分是 KV Cache 显存计算的核心。如果忘记乘 `num_layers`，就会把显存估算小很多；如果把 `num_attention_heads` 和 `num_kv_heads` 混淆，也会算错。

---

### 2.10 第九部分：reshape 是为了适配不同 Attention 后端

#### 本段核心知识点

老师提到，KV Cache 申请完后还会进行 `reshape`，最终形状和后续使用的 Attention backend 有关。

不同后端对输入 Tensor layout 的要求可能不同，例如：

- xFormers；
- vLLM 自带的 PagedAttention kernel；
- FlashAttention 类后端；
- 其他 CUDA kernel。

因此，初始化 KV Cache 时不仅要申请足够显存，还要把它整理成计算 kernel 方便访问的形状。

#### 相关概念解释

**reshape：** 改变 Tensor 的视图形状，使同一段内存按不同维度解释。多数情况下 reshape 不一定改变底层数据内容，但具体是否复制取决于内存是否连续和框架实现。

**layout：** 张量在内存中的维度排列方式。对 CUDA kernel 来说，layout 会影响访存连续性、coalescing、cache 命中和整体性能。

#### 在 AI Infra 中的作用

推理系统不是“能算就行”，还要“算得快”。KV Cache 的 layout 会影响 Attention kernel 的访存效率，进而影响 decode 阶段的 TPOT 和吞吐量。

---

### 2.11 结尾：本节课完整总结

老师最后总结了 vLLM KV Cache 初始化分配的大致步骤：

```text
1. 通过模拟运行计算可用显存
2. 获取每层 Attention 的 KV Cache 基本信息
3. 根据 block_size、num_kv_heads、head_size、dtype 计算 block 大小
4. 根据剩余显存计算 num_blocks
5. 按层分配 KV Cache Tensor
6. reshape 成 Attention 后端需要的形状
7. 挂到 ModelRunner 上
```

这个流程对应的是 vLLM 启动推理服务时的初始化阶段，不是每个请求来了才重新计算一次。

---

## 3. 任务三：从输入到输出串联整节课技术主线

下面用一次大模型推理服务的完整生命周期串起来。

---

### 3.1 服务启动阶段：先决定 KV Cache 池有多大

当你启动 vLLM 推理服务时，例如加载一个 Qwen 或 LLaMA 模型，系统首先会加载模型权重到 GPU。

这时 GPU 显存大致被分成：

```text
GPU 总显存
├── vLLM 不允许使用或预留给系统的部分
└── vLLM 允许使用的部分
    ├── 模型权重
    ├── 临时激活值 / 中间张量 / kernel workspace
    └── KV Cache 池
```

vLLM 根据 `gpu_memory_utilization` 决定自己最多能占用多少显存，然后扣除权重和 profile run 得到的非 KV 开销，得到 KV Cache 可用显存。

---

### 3.2 初始化阶段：把 KV Cache 池切成固定大小 block

得到 KV Cache 可用显存后，vLLM 要计算每个 block 的大小。

对于单层来说：

```text
block_bytes_per_layer
= block_size × num_kv_heads × head_size × 2 × dtype_size
```

对于整个模型来说：

```text
total_block_bytes
= num_layers × block_bytes_per_layer
```

于是：

```text
num_blocks = available_kv_cache_memory // total_block_bytes
```

然后系统会为每个 Attention 层申请 KV Cache 张量，并将 block id 初始化到 FreeBlockQueue 里。

---

### 3.3 请求进入阶段：Prompt 被 tokenizer 转成 token ids

用户输入 prompt：

```text
hello everybody from xhs, where are you?
```

模型不能直接处理自然语言文本，而是先通过 tokenizer 转成 token ids：

```text
[11, 13, 41, 56, 22, 23, 31, 44, ...]
```

然后按照 `block_size` 切成逻辑 block。

例如 `block_size = 4`：

```text
logical block 0: [11, 13, 41, 56]
logical block 1: [22, 23, 31, 44]
logical block 2: [...]
```

---

### 3.4 调度阶段：逻辑 block 映射到物理 block

调度器发现这个请求需要计算 prompt 的 KV Cache，于是从 FreeBlockQueue 中取出物理 block。

映射关系可能是：

```text
logical block 0 → physical block 100
logical block 1 → physical block 37
logical block 2 → physical block 205
```

对请求来说，token 顺序是连续的；对 GPU 显存来说，物理 block 可以不连续。

这就是“分页显存管理”的关键。

---

### 3.5 Prefill 阶段：计算 prompt 中所有 token 的 K/V 并写入 KV Cache

模型执行 prompt 的 forward。每一层 Attention 都会为这些 token 计算 K 和 V。

这些 K/V 不会简单存在 token ids 里，而是存到对应 Attention 层的 KV Cache Tensor 里。

对于每一层，写入位置由 block table 决定：

```text
第 0 个逻辑 block 的 K/V → 物理 block 100
第 1 个逻辑 block 的 K/V → 物理 block 37
第 2 个逻辑 block 的 K/V → 物理 block 205
```

Prefill 结束后，prompt 的历史信息已经被缓存下来。

---

### 3.6 Decode 阶段：每生成一个新 token，就复用历史 KV Cache

接下来模型开始逐 token 生成回答。

每一轮 decode：

```text
当前新 token
  ↓
计算当前 token 的 Q/K/V
  ↓
Q 与历史 K 做 attention
  ↓
根据 attention 权重聚合历史 V
  ↓
得到输出 hidden state
  ↓
经过 lm_head 得到 logits
  ↓
采样出下一个 token
  ↓
把当前 token 的 K/V 追加写入 KV Cache
```

如果当前 block 还没满，就继续写入当前 block；如果当前 block 满了，就从 FreeBlockQueue 申请新 block。

---

### 3.7 请求结束阶段：释放 block

当模型生成 EOS，或者达到最大输出长度，或者用户中断请求后，这个 sequence 结束。

它占用的物理 block 会被释放回 FreeBlockQueue，供后续请求复用。

这就是 vLLM 高并发能力的重要来源：

```text
请求来了 → 按需申请 block
请求增长 → 不够再申请 block
请求结束 → 释放 block
其他请求 → 复用这些 block
```

---

## 4. 本节课最重要的公式汇总

### 4.1 vLLM 允许使用的显存预算

```text
allowed_memory = total_gpu_memory × gpu_memory_utilization
```

---

### 4.2 KV Cache 可用显存

```text
available_kv_cache_memory
= allowed_memory - weight_memory - non_kv_runtime_memory
```

---

### 4.3 单层中一个 block 的 KV Cache 字节数

```text
block_bytes_per_layer
= block_size × num_kv_heads × head_size × 2 × dtype_size
```

---

### 4.4 全模型意义下一个 block 的总字节数

```text
total_block_bytes
= num_layers × block_size × num_kv_heads × head_size × 2 × dtype_size
```

---

### 4.5 可以切出的 block 数量

```text
num_blocks
= available_kv_cache_memory // total_block_bytes
```

---

### 4.6 理论 token 容量

```text
total_token_capacity ≈ num_blocks × block_size
```

注意：这只是 KV Cache 池能容纳的 token block 总位置，真实可服务的请求数还受调度策略、prompt 长度、输出长度、batch 限制、Attention kernel 性能等影响。

---

## 5. 和 nano-vLLM 源码的对应关系

如果你在读 nano-vLLM，可以这样对应：

| 课程概念 | 源码中可能对应的模块 | 作用 |
|---|---|---|
| `block_size` | 配置参数 / EngineArgs | 每个 block 可容纳的 token 数 |
| `num_blocks` | BlockManager 初始化 | GPU 上能切出的物理 block 数 |
| `FreeBlockQueue` | BlockManager 的 free blocks | 管理空闲物理 block |
| logical block → physical block | block table | 记录每个 sequence 的映射关系 |
| `ModelRunner` | model_runner.py | 执行模型 forward，持有 KV Cache |
| 每层 KV Cache | Attention 层 / model_runner 初始化 | 保存每层历史 K/V |
| profile run | 初始化阶段显存 profiling | 估算非 KV 显存占用 |
| reshape | Attention backend 适配 | 把 KV Cache 改成 kernel 需要的 shape |
| TP 独立 KV Cache | 多 GPU 执行器 | 每个设备维护自己的 KV Cache |

---

## 6. 初学者最容易混淆的点

### 6.1 block_size = 16 不是 16 字节

`block_size = 16` 表示一个 block 能容纳 16 个 token 的 KV Cache。

真正的物理大小还要乘：

```text
num_layers × num_kv_heads × head_size × 2 × dtype_size
```

---

### 6.2 KV Cache 里存的不是 token id

token id 是整数编号，例如：

```text
[11, 13, 41, 56]
```

KV Cache 存的是模型每层 Attention 计算出来的 Key Tensor 和 Value Tensor，是浮点数张量。

---

### 6.3 一个 token 不是只有一份 K/V

一个 token 在每一层 Attention 都有自己的 K/V。

如果模型有 32 层，那么这个 token 会在 32 层里各自留下 K/V 缓存。

---

### 6.4 num_kv_heads 不一定等于 num_attention_heads

在 MHA 中两者通常相等。

但在 GQA/MQA 中，KV head 数可能更少。KV Cache 显存计算用的是 `num_kv_heads`，不是 Query head 数。

---

### 6.5 profile run 不是在处理真实用户请求

profile run 是服务启动或初始化阶段用来估算显存峰值的模拟执行，不是给用户生成回答。

---

### 6.6 reshape 不是为了改变语义，而是为了适配计算布局

同样一块 KV Cache 显存，可以按不同维度顺序组织。不同 Attention kernel 对 layout 有不同要求，reshape 的目标是让后续计算更高效。

---

## 7. 面试问题与参考答案

### 问题 1：vLLM 为什么要做 KV Cache 分块管理？

**回答：**

传统推理框架可能为每个请求按照最大上下文长度提前申请连续 KV Cache。这样会导致大量内部碎片，因为请求实际长度往往远小于最大长度。vLLM 将 KV Cache 切成固定大小 block，请求按需申请，结束后释放，从而提高显存利用率，支持更高并发和更好的 continuous batching。

---

### 问题 2：block_size = 16 具体是什么意思？

**回答：**

它表示一个 KV Cache block 可以容纳 16 个 token 的 K/V 缓存，不是 16 字节，也不是 16 个 token id。实际显存大小还要乘以层数、KV head 数、head size、K/V 两份以及 dtype 字节数。

---

### 问题 3：一个 KV Cache block 的大小怎么计算？

**回答：**

单层 Attention 中：

```text
block_bytes_per_layer
= block_size × num_kv_heads × head_size × 2 × dtype_size
```

如果考虑整个模型所有层：

```text
total_block_bytes
= num_layers × block_size × num_kv_heads × head_size × 2 × dtype_size
```

其中 `2` 表示 K 和 V 两份缓存。

---

### 问题 4：vLLM 如何计算 num_blocks？

**回答：**

先计算 KV Cache 可用显存：

```text
available_kv_cache_memory
= total_gpu_memory × gpu_memory_utilization
  - weight_memory
  - non_kv_runtime_memory
```

再除以一个全模型 block 的大小：

```text
num_blocks = available_kv_cache_memory // total_block_bytes
```

这样得到 GPU 上可以切出的物理 block 数量。

---

### 问题 5：为什么不能把 GPU 总显存全部用来存 KV Cache？

**回答：**

因为 GPU 显存还要存模型权重、临时激活值、中间张量、CUDA workspace、通信 buffer 和运行时开销。如果全部分给 KV Cache，真实推理时很容易 OOM。所以 vLLM 先用 `gpu_memory_utilization` 限制预算，再扣掉权重和非 KV 临时开销。

---

### 问题 6：为什么 vLLM 要做 profile run？

**回答：**

模型权重显存可以根据参数量和 dtype 大致预估，但推理过程中的临时显存开销和具体模型、batch、Attention 后端、CUDA kernel 都有关，纯手算很难准确。profile run 通过模拟执行观察真实显存峰值，从而估算非 KV Cache 开销，避免 KV Cache 分配过多导致 OOM。

---

### 问题 7：KV Cache 是全模型一份，还是每层一份？

**回答：**

每个 Attention 层都有独立的 KV Cache。因为每一层的输入 hidden state 不同，计算出来的 K/V 也不同。一个 token 会在每一层产生自己的 K/V，因此计算 KV Cache 显存时必须乘以 `num_layers`。

---

### 问题 8：KV Cache 里存的是 token id 吗？

**回答：**

不是。token id 是 tokenizer 输出的整数编号，作为模型输入。KV Cache 存的是 Attention 层计算得到的 Key 和 Value 张量，是浮点数或量化后的张量，用来在 decode 阶段复用历史上下文。

---

### 问题 9：FreeBlockQueue 的作用是什么？

**回答：**

FreeBlockQueue 保存当前空闲的物理 block id。请求需要 KV Cache 时，从队列里取 block；请求结束后，把占用的 block 归还队列。它是 block 复用和动态显存管理的基础。

---

### 问题 10：逻辑 block 和物理 block 有什么区别？

**回答：**

逻辑 block 是某个 sequence 按 token 顺序切出来的第 0 块、第 1 块等；物理 block 是 GPU 显存中真实的 KV Cache 存储块。vLLM 通过 block table 维护逻辑 block 到物理 block 的映射，所以逻辑上连续的序列，物理上可以分散存储。

---

### 问题 11：Tensor Parallel 下 KV Cache 怎么处理？

**回答：**

在 Tensor Parallel 下，每个 GPU 有自己的 ModelRunner，并保存该设备负责计算部分对应的 KV Cache。也就是说 KV Cache 和设备、ModelRunner 是绑定的，不是所有 GPU 共用一份完整 KV Cache。

---

### 问题 12：reshape KV Cache 的目的是什么？

**回答：**

reshape 是为了把 KV Cache 整理成 Attention backend 需要的 Tensor layout。不同后端如 xFormers、PagedAttention kernel、FlashAttention 类 kernel 对维度顺序和内存布局要求不同。合适的 layout 可以提高访存效率和 decode 性能。

---

## 8. 这节课可以怎么复习

你可以按下面这条线来背：

```text
为什么分块？
  因为连续 KV Cache 浪费显存，降低 batch 能力。

怎么知道有多少显存可分？
  total_gpu_memory × gpu_memory_utilization - weights - profile 得到的临时开销。

一个 block 多大？
  block_size × num_kv_heads × head_size × 2 × dtype_size，再乘 num_layers。

能分多少 block？
  available_kv_cache_memory // total_block_bytes。

block 去哪里？
  初始化到 FreeBlockQueue，请求来了由调度器分配。

KV Cache 挂在哪里？
  挂在 ModelRunner 上，每个 Attention 层都有自己的 KV Cache。

为什么 reshape？
  为了适配不同 Attention kernel 的输入布局。
```

---

## 9. 本节课一句话总结

这节课讲的是：

> vLLM 不是随便把显存切成 block，而是在加载模型后，通过显存预算、权重占用、模拟运行得到的临时开销，精确估算可用于 KV Cache 的空间，再根据模型层数、KV head 数、head size、block_size 和 dtype 计算 block 大小与 block 数量，最终为每层 Attention 分配并 reshape KV Cache，挂到 ModelRunner 上，供后续调度器按需分配给请求。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-vllm|模块-vllm]]

%% 项目关联导航：结束 %%
