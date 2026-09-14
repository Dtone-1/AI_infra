# vLLM 技术分享与大模型推理框架学习/工作答疑课程学习笔记

> 说明：本笔记基于课程语音转写文件整理。原始转写中存在大量自动识别误差，例如“威严墨”应理解为 **vLLM**，“KVCatch/CubicCatch”应理解为 **KV Cache**，“XQ的”应理解为 **Executor**，“Woke/Walker”应理解为 **Worker**，“摩托鲁/Model Nogger”应理解为 **ModelRunner**，“配角担选/Page 的单线”应理解为 **PagedAttention**，“投机杰码”应理解为 **Speculative Decoding**。

---

## 一、这节课的核心宗旨

这节课不是单独讲某一个源码文件，而是一次 **vLLM 推理框架技术总览 + 工作/面试答疑**。它的目标是帮助学习者从“只知道 vLLM 能部署模型”提升到“知道 vLLM 为什么要分这么多模块、每类优化技术解决什么瓶颈、实际工作中哪些模块最值得优先学习”。

整节课可以概括为一句话：

> vLLM 不是一个简单的模型调用库，而是一个面向高并发、大模型、多 GPU、多后端、多量化、多调度策略的推理系统；学习 vLLM 要同时理解系统架构、内存管理、并行通信、量化压缩、投机解码和实际工程分工。

课程主要围绕六个模块展开：

1. **vLLM 基础资料与学习入口**：官方文档、官网、Release Notes、技术 Blog、examples 目录。
2. **vLLM 架构总览**：LLM / AsyncLLM、Engine、Scheduler、Executor、Worker、ModelRunner、模型对象之间的层级关系。
3. **内存管理优化**：KV Cache、PagedAttention、Prefix Caching、block table、slot mapping、KV Cache Manager。
4. **分布式并行优化**：DP、TP、PP、EP、CP，以及 all-reduce、all-gather、all-to-all 等 collective communication。
5. **量化压缩模块**：FP8、AWQ、GPTQ、SmoothQuant、KV Cache 量化、在线/离线量化、量化粒度、量化执行流程。
6. **投机解码组件**：draft model、Medusa、EAGLE、MTP，以及为什么它能缓解 decode 阶段串行瓶颈。

对于你目前学习 AI Infra / vLLM / nano-vLLM 的阶段，这节课的价值很大：它不是让你死记某个函数，而是让你知道 **vLLM 的知识地图**，也就是哪些模块在工业界最常见、哪些模块适合写简历、哪些模块容易被面试官连续追问。

---

## 二、课程知识点按内容推进顺序整理

原始转写没有严格时间戳，下面按照课程实际推进顺序整理。

---

### 1. vLLM 学习入口：先看官方资料，而不是只看零散教程

#### 本段核心知识点

课程一开始强调，vLLM 更新非常快，学习时不能只依赖旧博客或旧教程，而要定期关注官方资料。

推荐入口包括：

- vLLM 官方文档；
- vLLM GitHub 主仓库；
- vLLM 官方 Blog；
- vLLM Release Notes；
- `examples/` 目录中的示例代码。

#### 概念解释

**Release Notes** 是每个版本发布时记录新特性、性能优化、模型支持、bug fix 的说明。vLLM 是快速迭代框架，不同版本之间架构、参数、backend、模型支持差异可能很大，所以学习时要养成看 release 的习惯。

**examples 目录** 对新手很重要，因为它通常比源码主干更容易跑通。先跑 examples，再反向追源码，是学习大型框架更现实的方法。

#### 在 AI Infra 中的作用

AI Infra 工程不是只学“原理”，还要跟框架版本同步。例如某个版本开始支持新模型、新 backend、新量化方法、新调度模式，这些都会影响部署方案和面试表达。

#### 面试可说法

> 我学习 vLLM 时不会只看单篇教程，而是会先看官方文档、Release Notes 和 examples。因为 vLLM 迭代很快，很多模块比如 Engine V1/V2、ModelRunner、量化 backend、并行策略都会随版本变化。examples 适合作为入口，Release Notes 适合跟踪新特性。

---

### 2. vLLM 推理优化技术总览：计算、内存、调度、分布式、量化、投机解码

#### 本段核心知识点

课程将 vLLM 的优化技术分为几类：

| 优化类别 | 典型技术 | 主要解决的问题 |
|---|---|---|
| 计算优化 | CUDA Graph、算子融合、FlashAttention、编译优化 | 降低 kernel launch 开销、提升 attention/GEMM 性能 |
| 内存优化 | PagedAttention、KV Cache 管理、Prefix Caching | 提高显存利用率，减少重复计算 |
| 调度优化 | Continuous Batching、负载均衡、PD 分离、请求路由 | 提高吞吐量，降低 TTFT/TPOT |
| 分布式优化 | DP、TP、PP、EP、CP | 支持大模型和多 GPU 并行推理 |
| 量化压缩 | FP8、AWQ、GPTQ、KV Cache 量化 | 降低显存占用，提高吞吐 |
| 投机解码 | draft model、Medusa、EAGLE、MTP | 减少 decode 阶段串行 forward 次数 |

#### 概念解释

**计算优化** 关注的是一次模型 forward 本身怎么更快，例如 attention kernel、GEMM、算子融合、CUDA Graph。

**内存优化** 关注的是显存怎么更省、更少碎片、更高复用率。vLLM 最核心的创新之一就是 PagedAttention，用分页思想管理 KV Cache。

**调度优化** 关注的是多个请求怎么组织成 batch，哪些请求先执行，prefill 和 decode 怎么混合，如何降低 TTFT 和提高吞吐。

**分布式优化** 关注的是一个模型或一批请求怎么跨多 GPU / 多机执行。

**量化压缩** 关注的是权重、激活、KV Cache 的数据类型压缩，例如 BF16 → FP8 / INT4。

**投机解码** 关注的是 decode 阶段一次只能生成一个 token 的串行瓶颈，尝试让小模型先猜多个 token，再让大模型并行验证。

#### 在 AI Infra 中的作用

这几类优化分别对应不同岗位方向：

- 推理框架：调度、KV Cache、Engine、Executor、Worker；
- 算子优化：FlashAttention、GEMM、CUDA/Triton kernel；
- 分布式系统：TP/DP/EP/PD 分离、通信优化；
- 量化部署：FP8、INT4、权重加载、量化 kernel；
- 模型服务：OpenAI API、请求路由、负载均衡、流式返回。

---

### 3. vLLM 架构总览：从用户接口到模型对象的分层

#### 本段核心知识点

课程从自上而下的角度讲 vLLM 的模块分层：

```text
用户 / 客户端
  ↓
LLM / AsyncLLM / API Server
  ↓
Engine / EngineCore
  ↓
Scheduler
  ↓
Executor
  ↓
Worker
  ↓
ModelRunner
  ↓
Model Object
  ↓
Attention / MLP / Sampling / CUDA Kernel
```

不同版本的 vLLM 中类名和组织方式可能略有差异，但系统角色大体类似。

#### 关键模块解释

##### 1. LLM / AsyncLLM

`LLM` 通常用于离线批量推理，适合一次性提交一批 prompt，然后拿完整输出。

`AsyncLLM` 或在线 serving 相关模块用于在线服务，支持异步请求、流式返回、多客户端并发。

##### 2. Engine / EngineCore

Engine 是推理系统的核心入口层。它负责接收请求、维护请求队列、调用 Scheduler、把调度结果交给 Executor，并把模型输出返回给上层。

它不是最终执行模型 forward 的地方，更像是 **推理系统的大脑和中控层**。

##### 3. Scheduler

Scheduler 负责决定本轮 step 执行哪些请求、每个请求执行多少 token、是否需要分配/释放 KV Cache block。

它关心：

- waiting 队列；
- running 队列；
- prefill 请求；
- decode 请求；
- max_num_batched_tokens；
- max_num_seqs；
- KV Cache block 是否够用；
- 是否需要抢占或延迟某些请求。

##### 4. Executor

Executor 负责把 Scheduler 的调度结果派发给 Worker。它存在的关键原因之一是 vLLM 要支持多进程、多 GPU、张量并行、数据并行等复杂后端。

在 TP 场景中，一个 Executor 会管理多个 Worker，每个 Worker 对应一个 GPU rank。

##### 5. Worker

Worker 是 GPU 进程级别的执行单元。它持有某个 rank 上的模型分片或完整模型，并调用 ModelRunner 执行模型 forward。

注意：Worker 本身也不是最终计算函数，它通常会继续调用 ModelRunner。

##### 6. ModelRunner

ModelRunner 是真正组织模型执行输入、Attention Metadata、KV Cache、模型 forward 的模块。大多数和模型运行相关的逻辑都在这里。

##### 7. Model Object

Model Object 是具体模型结构，例如 Qwen、Llama、DeepSeek、Mistral 等。它最终执行 transformer layer、attention、MLP、lm_head 等。

#### 在 AI Infra 中的作用

vLLM 之所以分这么多层，是为了支持：

- 离线推理和在线服务；
- 多客户端请求；
- Continuous Batching；
- 多 GPU 并行；
- 多 backend；
- 多量化方法；
- 多模型架构；
- 复杂的内存管理和调度策略。

#### 面试可说法

> vLLM 的执行链路可以概括为：上层 LLM 或 API Server 接收请求，Engine 维护请求状态和主循环，Scheduler 决定本轮执行哪些 token 并分配 KV Cache，Executor 把调度结果派发给 Worker，Worker 调用 ModelRunner，ModelRunner 准备 attention metadata 和 KV Cache 信息后调用具体模型 forward，最后经过 sampling 得到输出 token，再返回给 Engine 做流式输出。

---

### 4. Attention Metadata：为什么 query_start_loc、slot_mapping、block_tables 很重要

#### 本段核心知识点

课程强调 vLLM Attention 模块中有几个关键元数据：

- `query_start_loc`
- `slot_mapping`
- `block_tables`
- KV Cache 相关 metadata

这些不是“附加信息”，而是 PagedAttention 能正确执行的基础。

#### 概念解释

##### 1. `query_start_loc`

在一次 batch 中，多个请求的 token 会被拼接成一个一维 token 序列送入模型。`query_start_loc` 记录每个请求在拼接后 query 张量中的起止位置。

例如本轮有 3 个请求：

```text
Request 1: 3 个 token
Request 2: 1 个 token
Request 3: 3 个 token
```

拼接后总 token 数是 7，那么：

```text
query_start_loc = [0, 3, 4, 7]
```

含义是：

- Request 1 使用 `[0, 3)`；
- Request 2 使用 `[3, 4)`；
- Request 3 使用 `[4, 7)`。

##### 2. `block_tables`

`block_tables` 是请求级别 / block 级别的映射表，记录每个请求的逻辑 block 对应哪个物理 KV Cache block。

它通常可以理解为二维结构：

```text
block_tables[request_index][logical_block_index] = physical_block_id
```

它解决的问题是：请求逻辑上是一段连续 token，但物理显存中的 KV Cache block 可以不连续。

##### 3. `slot_mapping`

`slot_mapping` 是 token 级别的映射。它告诉 attention kernel：当前这个 token 的 KV Cache 应该写入或读取哪个物理 slot。

可以理解为：

```text
slot_mapping[token_index] = physical_slot_id
```

其中 physical slot 是物理 block 内具体某个 token 位置。

##### 4. `block_tables` 与 `slot_mapping` 的区别

| 变量 | 粒度 | 作用 |
|---|---|---|
| `block_tables` | block 级别 | 记录一个请求的逻辑 block 到物理 block 的映射 |
| `slot_mapping` | token 级别 | 记录每个 token 对应的物理 KV slot |
| `query_start_loc` | request 级别 | 记录 batch 中每个请求 query token 的边界 |

#### 在 AI Infra 中的作用

PagedAttention 的关键不是“把 KV Cache 分块”这么简单，而是要让 attention kernel 在计算时知道：

- 当前 batch 中每个请求的 query 在哪里；
- 每个请求历史 token 的 KV Cache 存在哪些物理 block；
- 当前新 token 的 KV Cache 要写入哪个 slot。

这些信息就是通过 attention metadata 传入 attention backend 的。

---

### 5. KV Cache 与 PagedAttention：vLLM 内存管理的核心

#### 本段核心知识点

课程指出 vLLM 最早出圈的重要技术就是 **PagedAttention**。它借鉴操作系统虚拟内存分页思想，把 KV Cache 按固定大小的 block 管理，并通过 block table 建立逻辑块到物理块的映射。

#### KV Cache 为什么重要

在自回归生成中，每生成一个新 token，都需要关注前面所有 token。如果每一步都重新计算历史 token 的 K/V，会产生大量重复计算。

KV Cache 的作用是：

- prefill 阶段计算 prompt 所有 token 的 K/V，并缓存；
- decode 阶段每次只计算新 token 的 Q/K/V；
- attention 时新 token 的 Q 去和历史 K/V 做注意力计算；
- 历史 K/V 直接从 KV Cache 读取，不重新计算。

#### 没有 KV Cache vs 有 KV Cache

| 场景 | Decode 阶段每步计算 |
|---|---|
| 没有 KV Cache | 每步重新计算所有历史 token 的 K/V，重复计算严重 |
| 有 KV Cache | 每步只计算新 token 的 K/V，历史 K/V 直接读取 |

#### PagedAttention 的核心思想

传统 KV Cache 管理方式可能为每个请求预分配一大段连续显存，例如直接按最大长度 4096 分配。问题是：

- 请求实际可能只生成很短；
- 大量预留空间被浪费；
- 连续显存难以复用；
- 内部碎片严重；
- 并发吞吐下降。

PagedAttention 的思想是：

```text
逻辑 token 序列
  ↓ 按 block_size 切分
逻辑 block
  ↓ block table 映射
非连续物理 KV Cache block
```

它的优点是：

- 按需分配 block；
- 物理 block 不要求连续；
- 减少显存碎片；
- 提高 KV Cache 利用率；
- 支持更高并发；
- 方便 prefix cache 复用。

#### block、slot、block_size 的关系

假设 `block_size = 16`，则一个 block 最多存 16 个 token 的 KV Cache。

一个 slot 可以理解为 block 内一个 token 对应的 KV Cache 存储位置。

```text
Physical Block 7:
slot 0 → token A 的 KV
slot 1 → token B 的 KV
...
slot 15 → token P 的 KV
```

#### KV Cache Manager

课程中提到，Scheduler 初始化时会创建并持有 KV Cache Manager。它负责逻辑层与物理层之间的管理，包括：

- 当前请求需要多少 block；
- 哪些 block 已经分配；
- 哪些 block 仍然空闲；
- block table 如何维护；
- 请求结束后如何释放 block；
- prefix cache 如何复用 block。

#### 面试可说法

> vLLM 的 PagedAttention 借鉴了操作系统分页思想，把每个请求的 KV Cache 拆成固定大小 block。请求逻辑上是连续 token，但物理显存中的 block 可以不连续，通过 block table 维护逻辑 block 到物理 block 的映射。这样可以按需分配 KV Cache，降低内部碎片，提高并发吞吐。attention kernel 再通过 block_tables 和 slot_mapping 找到对应的 KV Cache 位置。

---

### 6. Prefix Caching：跨请求复用 KV Cache

#### 本段核心知识点

课程提到 Prefix Caching，即前缀缓存。它通过 block 级别的 KV Cache 复用，避免重复计算相同 prompt 前缀。

#### 概念解释

很多请求可能有相同前缀，例如：

```text
系统提示词：你是一个有帮助的助手……
用户问题 A：请解释 vLLM
用户问题 B：请解释 PagedAttention
```

系统提示词部分完全相同。如果每个请求都重新 prefill 系统提示词，就会浪费计算。

Prefix Caching 的做法是：

1. 对 prompt 前缀按 block 切分；
2. 对 block 中 token id 做 hash；
3. 如果新请求的某些 block 和已有缓存一致，就直接复用已有 KV Cache；
4. 只计算未命中的后续 block。

#### 在 AI Infra 中的作用

Prefix Caching 对以下场景特别有价值：

- 多轮对话；
- 长 system prompt；
- RAG 模板固定；
- agent 工具调用 prompt 固定；
- 批量请求共享相同上下文。

它可以显著降低 TTFT，因为 prefill 阶段重复计算减少了。

---

### 7. 分布式通信基础：collective communication 是并行推理的底座

#### 本段核心知识点

课程讲了分布式计算优化，并提到常见 collective communication：

- broadcast；
- scatter；
- gather；
- reduce；
- all-gather；
- all-reduce；
- reduce-scatter；
- all-to-all。

这些通信原语是 TP、DP、EP、CP 等并行策略的基础。

#### 主要通信原语解释

| 通信方式 | 含义 | 常见用途 |
|---|---|---|
| broadcast | 一个 rank 把数据发给所有 rank | 参数、调度信息广播 |
| scatter | 一个 rank 把不同数据分发给多个 rank | 数据切分分发 |
| gather | 多个 rank 把数据汇集到一个 rank | 收集结果 |
| reduce | 多个 rank 数据聚合到一个 rank | 求和/聚合 |
| all-gather | 每个 rank 收集所有 rank 的数据 | TP/CP 中拼接结果 |
| all-reduce | 每个 rank 都得到 reduce 后结果 | TP 中同步 partial output |
| reduce-scatter | reduce 后再切分给多个 rank | 大模型并行优化 |
| all-to-all | 每个 rank 给每个 rank 发送不同数据 | MoE expert dispatch/combine |

#### 在 AI Infra 中的作用

分布式推理的核心不是“多开几个 GPU”这么简单，而是不同并行策略对应不同通信模式。通信量、通信次数、是否能和计算 overlap，都会直接影响吞吐和延迟。

---

### 8. 数据并行 DP：复制模型，切请求

#### 本段核心知识点

课程强调：**数据并行不是切模型，而是切请求。**

DP 的做法是：

- 每个 DP rank 持有一份完整模型权重；
- 不同 DP rank 处理不同请求；
- 主要提升吞吐；
- 单个请求内部通常不需要跨 DP rank 通信。

#### 例子

假设有 100 个请求，`DP = 10`，可以理解为：

```text
DP rank 0 → 处理 10 个请求
DP rank 1 → 处理 10 个请求
...
DP rank 9 → 处理 10 个请求
```

每个 DP rank 都有完整模型副本。

#### DP 与 TP 的组合

如果 `DP = 2, TP = 4`，总 GPU 数通常是：

```text
总 GPU 数 = DP × TP = 2 × 4 = 8
```

可以理解为：

```text
DP rank 0: TP rank 0/1/2/3 组成一组，负责一份模型并行推理
DP rank 1: TP rank 0/1/2/3 组成另一组，负责另一份模型并行推理
```

#### 在 vLLM 中的意义

DP 常用于提高整体服务吞吐，并配合路由和负载均衡。课程中提到数据并行有内部负载均衡、混合负载均衡、外部负载均衡等模式。

#### 面试可说法

> DP 是请求级并行，每个 DP rank 通常持有完整模型副本，处理不同请求 batch。它主要提升吞吐，不解决单个模型太大放不下的问题。和 TP 组合时，每个 DP rank 内部还可以有多个 TP Worker 共同持有一个模型分片。

---

### 9. 张量并行 TP：切模型内部张量

#### 本段核心知识点

TP 是模型并行的一种，它把单层里的大矩阵计算切到多个 GPU 上。

常见切分位置包括：

- embedding；
- attention 中的 Q/K/V/O projection；
- MLP 中的 gate/up/down projection；
- lm_head。

#### Column Parallel 与 Row Parallel

面试中常问：**为什么有的线性层先列切，有的线性层行切？**

简单理解：

##### Column Parallel Linear

把权重矩阵按输出维度切分：

```text
Y = X · W
W = [W1, W2]
Y = [X·W1, X·W2]
```

每个 GPU 计算一部分输出特征，最后可能需要拼接/all-gather，或者后续计算可以继续保持切分状态。

##### Row Parallel Linear

把权重矩阵按输入维度切分：

```text
X = [X1, X2]
W = [W1; W2]
Y = X1·W1 + X2·W2
```

每个 GPU 计算 partial output，最后需要 all-reduce 求和。

#### 在 Transformer 中的典型组合

很多实现会采用类似：

```text
QKV / gate_up projection: Column Parallel
O projection / down projection: Row Parallel
```

这样可以让中间激活保持切分，减少不必要的 gather。

#### 在 AI Infra 中的作用

TP 解决的是：单个模型太大，或者单层计算太重，需要多个 GPU 共同完成一个请求的 forward。

#### 面试可说法

> TP 是模型内部并行，不是切请求。它通常切线性层的权重矩阵。Column parallel 按输出维切，得到部分输出；row parallel 按输入维切，每张卡得到 partial sum，最后 all-reduce。Transformer 中 QKV 和 MLP up/gate 常用 column parallel，O projection 和 down projection 常用 row parallel，以减少通信并保持张量切分一致。

---

### 10. 专家并行 EP：MoE 模型的核心并行方式

#### 本段核心知识点

EP 主要针对 MoE 层。MoE 模型中有多个 expert，每个 token 通过 gate/router 选择部分 expert。EP 把 expert 分布到不同 GPU 上。

#### 执行流程

MoE Expert Parallel 大致流程：

```text
hidden_states
  ↓
gate/router 计算每个 token 应该去哪些 expert
  ↓
all-to-all dispatch：把 token 发到对应 expert 所在 GPU
  ↓
每个 GPU 计算本地 expert
  ↓
all-to-all combine：把 expert 输出发回原位置
  ↓
聚合输出
```

#### 为什么 EP 用 all-to-all

因为每个 token 可能被路由到任意 expert，而 expert 又分布在不同 GPU 上，所以 token 需要在 GPU 间重新分发。这种多对多通信就是 all-to-all。

#### EP 与 TP 的区别

| 并行方式 | 切分对象 | 通信方式 | 适用模块 |
|---|---|---|---|
| TP | 单层矩阵/hidden dimension | all-reduce / all-gather | Attention、MLP |
| EP | expert 数量 | all-to-all | MoE expert layer |
| DP | 请求 batch | 通常请求间无通信 | 整体服务吞吐 |

#### 课程中的工程观点

课程中提到，现在大 MoE 模型部署通常会同时使用 TP 和 EP。Attention 层可能使用 TP/CP，MoE 层使用 EP，不同 module 使用不同并行策略。

#### 面试可说法

> Expert Parallel 主要用于 MoE 层，它不是把单个 expert 切开，而是把不同 expert 分配到不同 GPU。每个 token 通过 router 选择 expert，然后通过 all-to-all dispatch 到 expert 所在 GPU，本地计算后再 all-to-all combine 回来。TP 通常用 all-reduce，而 EP 的核心通信是 all-to-all。

---

### 11. 上下文并行 CP：长上下文 attention 的并行方式

#### 本段核心知识点

CP，即 Context Parallel，用于处理超长上下文 attention 中跨 token 依赖的问题。

#### 基本思想

把一个长序列按 sequence dimension 切成多个 chunk，每个 GPU 持有一部分 token。

例如序列长度 10k，切成 10 份：

```text
GPU 0 → token 0~999
GPU 1 → token 1000~1999
...
GPU 9 → token 9000~9999
```

每个 GPU 可以独立计算自己 chunk 的 Q/K/V。但 attention 需要看到全局 K/V，因此需要通信收集其他 GPU 的 K/V。

#### 执行流程

```text
输入长序列切成 N 个 chunk
  ↓
每个 GPU 计算本地 chunk 的 Q/K/V
  ↓
通过通信收集全局 K/V
  ↓
每个 GPU 用本地 Q 和全局 K/V 做 attention
  ↓
输出合并后进入下一层
```

#### 在 AI Infra 中的作用

CP 主要服务于长上下文模型推理，尤其当单 GPU 放不下完整上下文 KV 或 attention 计算过重时使用。

---

### 12. PD 分离：Prefill 与 Decode 的资源需求不同

#### 本段核心知识点

课程答疑中提到 PD 分离，即 Prefill-Decode Disaggregation。

重点纠正一个误区：

> PD 分离不是为了解决“模型权重太大放不下”，而是为了利用 prefill 和 decode 阶段不同的计算特性，提高整体吞吐和延迟表现。

#### Prefill 与 Decode 的差异

| 阶段 | 输入形态 | 主要瓶颈 | 特点 |
|---|---|---|---|
| Prefill | 一次处理 prompt 多个 token | 计算密集 | 大矩阵计算多，GPU 利用率高 |
| Decode | 每步生成 1 个 token | 显存带宽/访存受限 | 每步读大量 KV Cache，串行性强 |

#### 为什么要分离

如果 prefill 和 decode 混在同一批 GPU 上，二者资源需求不同，会互相干扰。

PD 分离可以：

- 用更适合大 batch 的资源处理 prefill；
- 用更适合低延迟、高频 decode 的资源处理 decode；
- 降低 TTFT；
- 提高 decode throughput；
- 避免 prefill 大请求阻塞 decode 小步生成。

#### 课程中的工程提醒

不是所有模型都需要 PD 分离。如果模型规模不大、并发不高、资源有限，PD 分离可能没有明显收益，甚至增加系统复杂度。

#### 面试可说法

> Prefill 通常是计算密集型，decode 通常更偏显存带宽受限。PD 分离不是为了解决模型放不下，而是因为两个阶段的最优并行配置、batch 形态、资源需求不同。把它们拆到不同 worker pool 或不同部署组，可以降低互相干扰，提高吞吐和延迟表现。

---

### 13. 量化压缩：FP8 是当前工业界重点之一

#### 本段核心知识点

课程把量化压缩作为重点模块，并在答疑中多次强调：如果做推理框架或模型部署，**FP8 量化是非常值得优先关注的方向**。

#### 常见量化方法

| 方法 | 主要量化对象 | 特点 |
|---|---|---|
| AWQ | 权重 | weight-only，常见 INT4 权重量化 |
| GPTQ | 权重 | weight-only，常见 INT4 权重量化 |
| SmoothQuant | 权重 + 激活 | 通过平滑激活/权重分布降低量化误差 |
| FP8 | 权重 + 激活，可扩展到 KV Cache | 当前大模型推理部署中越来越重要 |
| KV Cache Quantization | KV Cache | 降低长上下文显存占用 |

课程中提到 AWQ/GPTQ 通常属于只量化权重、不量化激活的方案；FP8 更接近 W8A8，即权重和激活都可进入低精度计算路径。

#### 离线量化与在线量化

##### 离线量化

模型上线前先把 BF16/FP16 权重转换成 FP8/INT4 等格式，并保存成量化后 checkpoint。

优点：

- 启动时更快；
- 运行时无需临时量化权重；
- 部署更稳定。

缺点：

- 需要额外量化工具；
- 需要校准/评测；
- 不同硬件 backend 可能要求不同格式。

##### 在线量化

加载 BF16/FP16 权重后，运行时转换为目标量化格式。

优点：

- 使用更灵活；
- 不一定需要提前准备量化 checkpoint。

缺点：

- 启动或运行时开销更高；
- 精度和性能需要验证；
- 工程路径更复杂。

#### 量化粒度

课程提到多种量化粒度，可以按从粗到细理解：

| 粒度 | 含义 | 特点 |
|---|---|---|
| per-tensor | 整个 tensor 共用一个 scale | 简单但精度可能差 |
| per-channel | 每个 channel 一个 scale | 精度更好 |
| per-token | 每个 token 一个 scale | 适合激活量化 |
| per-group / per-block | 按 group/block 共用 scale | 权衡精度和开销 |

一般来说，粒度越细，精度越好，但 scale 存储、计算和 kernel 实现更复杂。

#### vLLM 中量化执行流程

课程中提到，vLLM 的量化通常是模块化设计，每种量化方法有自己的 config、method、kernel/backend。

可以抽象成如下流程：

```text
模型加载
  ↓
根据参数创建 QuantizationConfig
  ↓
每个 layer 调用 get_quant_method
  ↓
创建量化权重 create_weights
  ↓
加载 checkpoint 权重
  ↓
process_weights_after_loading 做后处理
  ↓
推理时调用 quant_method.apply
  ↓
进入对应低精度 kernel / backend
```

#### 框架工程师和算子工程师的分工

课程答疑中提到：

- 框架侧通常负责量化配置、权重创建、权重加载、量化方法 dispatch、调用 kernel；
- 算子侧通常负责 FP8 GEMM、W8A8 kernel、反量化融合等底层实现；
- 框架工程师不一定亲自写所有 CUDA kernel，但要理解 kernel 的输入输出、数据格式、scale 组织方式，否则无法正确对接。

#### 面试可说法

> vLLM 的量化模块通常通过 QuantizationConfig 和 QuantMethod 组织。模型加载时根据配置为每层选择量化方法，创建量化权重，加载后做权重后处理，推理时通过 apply 调用对应 backend。AWQ/GPTQ 多数是 weight-only，FP8 更偏 W8A8，工业界现在很重视 FP8，因为它能兼顾显存、吞吐和硬件支持。

---

### 14. KV Cache 量化：长上下文显存优化的重要方向

#### 本段核心知识点

课程答疑中提到，KV Cache 量化在生产中会用，但前提是 attention backend 支持对应格式。

#### 为什么 KV Cache 量化重要

长上下文推理中，KV Cache 可能比权重更容易成为显存瓶颈。

KV Cache 显存大致与以下因素成正比：

```text
层数 × token 数 × KV head 数 × head size × 2(K和V) × dtype_size
```

如果把 KV Cache 从 BF16 压到 FP8，理论上 KV Cache 显存约减半。

#### 注意点

KV Cache 量化不能只改 dtype，还要考虑：

- attention kernel 是否支持 FP8 KV；
- scale 如何保存；
- 读 KV 时是否需要反量化；
- 反量化是否和 attention 融合；
- 精度是否可接受；
- 长上下文下误差是否累积。

---

### 15. 投机解码：解决 decode 串行瓶颈

#### 本段核心知识点

投机解码的核心目标是减少大模型 decode 阶段 forward 次数。

传统 decode：

```text
生成 K 个 token → 需要 K 次大模型 forward
```

投机解码：

```text
小模型先猜多个 token
  ↓
大模型一次并行验证多个 token
  ↓
接受其中一部分 token
  ↓
减少大模型 forward 次数
```

#### 为什么 decode 慢

decode 阶段每次只能基于已生成 token 生成下一个 token，因此天然串行。每一步只处理很少 token，但要读取大量权重和 KV Cache，容易受内存带宽限制。

#### draft model / proposer

draft model 是一个更小、更便宜的模型，负责快速生成候选 token。

大模型作为 target model，负责验证 draft token 是否可接受。

#### Medusa

Medusa 是一种单模型并行解码思路：在主模型顶部增加多个解码头，让模型一次预测多个未来 token。

它不一定需要单独的小模型，但需要额外 head 或训练。

#### EAGLE

EAGLE 可以理解为一种插件化的 speculative decoding 方案，它不直接预测 token，而是预测未来 hidden state，再映射到 token 空间。

#### MTP

MTP，即 Multi-Token Prediction。课程答疑中提到，现在一些新模型原生带 MTP，未来可能比传统“小模型引导大模型”的 speculative decoding 更值得关注。

#### 投机解码收益取决于什么

投机解码不一定总是加速。收益主要取决于：

- draft token 接受率；
- draft model 成本；
- 每次 draft 的 token 数；
- batch size；
- 任务类型；
- 中英文差异；
- 代码生成 vs 创作类任务；
- 多卡部署下 draft model 如何并行/放置；
- 验证过程是否高效。

课程中提到，代码生成等规律性强的任务接受率可能更高，创作类任务接受率可能较低。

#### 面试可说法

> Speculative decoding 用便宜的 draft model 或额外预测头先生成多个候选 token，再由大模型一次 forward 并行验证，从而把原本逐 token 串行 decode 转成“猜测 + 验证”的形式。它的收益取决于接受率和 draft 成本。如果接受率低，或者 batch 已经很大，收益可能不明显。Medusa、EAGLE、MTP 都是不同形式的多 token 预测或投机解码方案。

---

## 三、整节课串成一条“从输入到输出”的技术主线

下面把这节课讲到的所有模块串成一个完整推理流程。

---

### 1. 用户提交请求

用户可以通过两类入口使用 vLLM：

```text
离线批量推理：LLM.generate()
在线服务：OpenAI-compatible API / AsyncLLM
```

请求中包括：

- prompt；
- sampling 参数；
- request_id；
- priority；
- arrival_time；
- max_tokens；
- stop 条件等。

---

### 2. Tokenize

文本 prompt 被 tokenizer 转成 token ids：

```text
"介绍一下 vLLM" → [token_id_1, token_id_2, ...]
```

vLLM 后续调度和模型执行都基于 token ids。

---

### 3. Engine 接收请求并进入调度队列

Engine / EngineCore 接收请求后，将请求放入内部队列。

此时请求可能处于：

- waiting；
- running；
- prefill；
- decode；
- finished。

---

### 4. Scheduler 决定本轮执行哪些请求

Scheduler 根据资源约束做调度，例如：

- 本轮最多执行多少 token；
- 本轮最多容纳多少 seq；
- prefill 和 decode 如何混合；
- KV Cache block 是否够用；
- 是否需要为新请求分配 block；
- 是否有 prefix cache 命中；
- 是否有请求结束需要释放 block。

输出调度结果 `SchedulerOutput`。

---

### 5. KV Cache Manager 分配或复用 block

对于要执行的 token，KV Cache Manager 维护：

- 逻辑 block；
- 物理 block；
- block table；
- slot mapping；
- prefix cache；
- free block pool。

如果当前请求需要新的 KV Cache 空间，就分配新的物理 block。

如果命中 prefix cache，就复用已有 block。

---

### 6. Executor 分发调度结果

Scheduler 的输出交给 Executor。

Executor 根据并行配置把任务分发给 Worker：

- 单卡：一个 Worker；
- TP：多个 Worker 组成一个 TP group；
- DP + TP：多个 EngineCore / DP rank，每个 DP rank 内部有 TP workers；
- EP：MoE 层中 expert 分布在多个 GPU 上。

---

### 7. Worker 调用 ModelRunner

Worker 接收到执行命令后，调用 ModelRunner 执行模型。

ModelRunner 会准备：

- input_ids；
- positions；
- attention metadata；
- KV Cache tensors；
- block_tables；
- slot_mapping；
- query_start_loc；
- sampling metadata。

---

### 8. ModelRunner 调用具体模型 forward

具体模型执行 transformer forward：

```text
Embedding
  ↓
Transformer Layer 1
  ↓
Attention + MLP
  ↓
Transformer Layer N
  ↓
RMSNorm / LayerNorm
  ↓
lm_head
  ↓
logits
```

Attention 计算时通过 metadata 找到历史 KV Cache。

如果启用 TP/EP/CP，会在不同模块插入对应通信：

- TP：all-reduce / all-gather；
- EP：all-to-all dispatch/combine；
- CP：收集跨 chunk 的 K/V；
- DP：不同请求分发到不同副本。

---

### 9. Sampling 得到输出 token

模型输出 logits 后，根据 sampling 参数选择下一个 token：

- greedy；
- top-k；
- top-p；
- temperature；
- repetition penalty；
- stop token / EOS。

---

### 10. Decode 循环持续执行

如果请求未完成，新的 token 会进入下一轮 decode。

decode 阶段每轮通常只为每个请求生成一个 token，因此需要不断执行：

```text
调度 → KV Cache 读写 → forward → sampling → 返回 token
```

如果启用投机解码，则可能是：

```text
draft 生成多个 token → target model 验证 → 接受若干 token
```

---

### 11. 结果流式返回

对于在线服务，vLLM 会把生成 token 逐步返回给客户端，实现类似 ChatGPT 一个字/一个词逐渐出现的效果。

---

## 四、这节课对“找实习/面试”的重点提炼

### 1. 不是所有 vLLM 模块都要同等深度学习

课程答疑中非常重要的一点是：vLLM 模块太多，不可能短时间全部精通。面试和工作更看重你是否：

1. 理解整体架构；
2. 能讲清楚一个核心模块；
3. 项目真实跑过，有数据；
4. 能把源码、原理、性能指标串起来。

---

### 2. 新手优先级

对于你现在已经学过 nano-vLLM、准备学 vLLM 的阶段，建议优先级如下：

| 优先级 | 模块 | 为什么重要 |
|---|---|---|
| P0 | vLLM 总体架构 | 面试最容易先问“请求从哪里到哪里” |
| P0 | Scheduler / Continuous Batching | 推理框架核心，和吞吐/延迟直接相关 |
| P0 | KV Cache / PagedAttention | vLLM 最经典创新，几乎必问 |
| P0 | Executor / Worker / ModelRunner | vLLM 相比 nano-vLLM 的重要工程抽象 |
| P1 | TP / DP / EP | 大模型多卡部署必备 |
| P1 | FP8 / AWQ / GPTQ 量化流程 | 工业界高频方向 |
| P1 | Benchmark 指标 TTFT/TPOT/吞吐 | 项目落地必须有数据 |
| P2 | PD 分离 | 大规模 serving 场景加分 |
| P2 | Speculative Decoding / MTP | 新模型和 decode 优化加分 |
| P2 | CUDA/Triton Kernel | 如果投算子岗则必须深入 |

---

### 3. 简历项目怎么写才不像玩具

课程答疑中提到，面试官更看重项目真实性。只写“实现了一个推理框架”很空，应该写具体模块和指标。

更好的写法：

```text
基于 vLLM 部署 Qwen 系列模型，构建 OpenAI-compatible API 服务；
实现推理 benchmark，统计 TTFT、TPOT、吞吐量、显存占用；
分析不同并发数、输入长度、输出长度下 prefill/decode 性能瓶颈；
结合 vLLM 源码理解 Scheduler、KV Cache Manager、Executor、Worker、ModelRunner 的协作流程；
对比 FP16 与 AWQ/FP8 量化部署的显存与吞吐变化。
```

如果你做 nano-vLLM 项目，可以写：

```text
精读 nano-vLLM 源码，复现 LLM.generate、Scheduler、BlockManager、ModelRunner、PagedAttention 简化实现；
对比 nano-vLLM 与 vLLM 在 API Server、EngineCore、Executor、Worker、多进程通信、TP/DP、量化、prefix caching 等工程能力上的差异。
```

---

### 4. 面试官可能怎么追问

面试官不会只问“你知道 PagedAttention 吗”，更可能连续追问：

```text
PagedAttention 解决什么问题？
为什么传统 KV Cache 会浪费显存？
block table 存什么？
slot_mapping 和 block_table 区别是什么？
decode 阶段为什么每步只算一个 token？
如果请求结束，KV Cache 怎么释放？
prefix caching 如何判断是否命中？
Scheduler 在什么时候分配 block？
TP 下每个 Worker 的 KV Cache 是完整的吗还是分片的？
```

所以你学习时要把每个概念放回完整链路，而不是孤立背八股。

---

## 五、常见面试问题与参考答案

---

### Q1：vLLM 的整体架构你怎么理解？

**回答：**

vLLM 可以从用户接口到模型执行分成多层。上层是 LLM / AsyncLLM 或 OpenAI-compatible API，负责接收请求。请求进入 Engine 或 EngineCore 后，由 Scheduler 决定本轮执行哪些请求和 token，同时管理 KV Cache block。调度结果交给 Executor，Executor 负责把任务广播或分发给 Worker。Worker 是 GPU 进程级执行单元，它调用 ModelRunner。ModelRunner 准备 input ids、position、attention metadata、KV Cache、block table、slot mapping，然后调用具体模型 forward。模型输出 logits 后经过 sampling 得到新 token，再返回给 Engine，在线服务场景下会流式返回给客户端。

---

### Q2：vLLM 为什么需要 Executor 和 Worker？nano-vLLM 为什么可以省略？

**回答：**

nano-vLLM 是教学型简化实现，通常单进程、单卡，Engine 可以直接调用 ModelRunner。但真实 vLLM 要支持多进程、多 GPU、TP、DP、EP、不同 distributed backend、不同 worker 类型，所以需要 Executor 作为中间层统一管理 Worker。Worker 对应具体 GPU/rank，负责调用 ModelRunner 执行本地模型分片或完整模型。Executor/Worker 是工程复杂度带来的抽象，不是理论上必须存在，但工业系统中非常必要。

---

### Q3：PagedAttention 解决了什么问题？

**回答：**

PagedAttention 解决传统 KV Cache 连续预分配导致的显存浪费和碎片问题。传统方式可能为每个请求按最大上下文长度预留一大段连续显存，但请求实际生成长度不确定，很多空间没用也不能给别人用。PagedAttention 把 KV Cache 切成固定大小 block，按需分配物理 block，并通过 block table 维护逻辑 block 到物理 block 的映射。这样物理 block 可以不连续，显存利用率更高，并发吞吐更好。

---

### Q4：block_table 和 slot_mapping 有什么区别？

**回答：**

block_table 是 block 级别映射，记录每个请求的逻辑 block 对应哪个物理 KV Cache block。slot_mapping 是 token 级别映射，记录当前 batch 中每个 token 的 KV Cache 应该写入或读取哪个物理 slot。简单说，block_table 解决“这个请求的历史 KV 在哪些 block”，slot_mapping 解决“这个 token 具体对应哪个 KV 位置”。

---

### Q5：query_start_loc 是什么？

**回答：**

query_start_loc 记录 batch 中每个请求的 query token 在拼接后一维 query 张量中的起止位置。因为 vLLM 会把多个请求本轮要执行的 token 拼成一个大 tensor 送给模型，但 attention kernel 仍然需要知道哪些 token 属于同一个请求，所以需要 query_start_loc 作为请求边界。

---

### Q6：KV Cache 为什么能加速 decode？

**回答：**

自回归生成中，每个新 token 都要 attend 到历史 token。如果没有 KV Cache，每一步都要重新计算历史 token 的 K/V。KV Cache 在 prefill 阶段保存 prompt 的 K/V，decode 阶段只计算新 token 的 Q/K/V，历史 K/V 直接读取，因此避免重复计算。decode 阶段每步 query token 通常只有 1 个，但需要读取所有历史 K/V。

---

### Q7：Prefix Caching 和 KV Cache 是什么关系？

**回答：**

KV Cache 是单个请求内部复用历史 K/V，Prefix Caching 是跨请求复用相同前缀的 KV Cache。Prefix Caching 会对 prompt block 做 hash，如果新请求的前缀 block 和已有缓存一致，就可以直接复用已有 KV Cache block，减少 prefill 计算。

---

### Q8：DP、TP、EP 的区别是什么？

**回答：**

DP 是数据并行，每个 rank 有完整模型副本，处理不同请求，主要提升吞吐。TP 是张量并行，把模型内部大矩阵切到多个 GPU 上，一个请求需要多个 GPU 协同执行，常见通信是 all-reduce/all-gather。EP 是专家并行，主要用于 MoE 层，把不同 expert 放到不同 GPU，token 通过 gate 路由到 expert，核心通信是 all-to-all。

---

### Q9：TP 中 column parallel 和 row parallel 有什么区别？

**回答：**

Column parallel 是按权重输出维度切分，每张卡计算一部分输出特征；row parallel 是按输入维度切分，每张卡计算 partial output，最后通常 all-reduce 求和。Transformer 中 QKV projection 和 MLP up/gate 常用 column parallel，O projection 和 down projection 常用 row parallel，这样可以减少中间 gather 并保持张量切分一致。

---

### Q10：MoE 的 Expert Parallel 为什么需要 all-to-all？

**回答：**

MoE 中每个 token 会通过 router 选择 expert，而 expert 分布在不同 GPU 上。一个 GPU 上的 token 可能要发给其他 GPU 的 expert，其他 GPU 的 token 也可能发到本 GPU，所以这是多对多的数据交换，对应 all-to-all。计算完成后还要把结果 combine 回原来的 token 顺序，通常还需要一次 all-to-all。

---

### Q11：PD 分离是为了解决模型放不下吗？

**回答：**

不是。PD 分离主要是因为 prefill 和 decode 的计算特性不同。Prefill 一次处理多个 prompt token，通常计算密集；decode 每步只生成一个 token，但要读大量 KV Cache，通常更受显存带宽限制。把 prefill 和 decode 分离到不同资源池，可以针对不同阶段配置不同并行度和 batch 策略，提高吞吐和延迟表现。

---

### Q12：AWQ/GPTQ 和 FP8 量化有什么区别？

**回答：**

AWQ/GPTQ 多数是 weight-only 量化，主要压缩权重，激活仍可能保持 FP16/BF16。FP8 更常用于 W8A8，即权重和激活都进入 FP8 计算路径，也可以扩展到 KV Cache 量化。FP8 更依赖硬件和 kernel 支持，但工业界越来越重视，因为它在吞吐和显存之间有较好平衡。

---

### Q13：vLLM 量化模块的大致流程是什么？

**回答：**

模型加载时先根据参数创建 QuantizationConfig，然后每层通过 get_quant_method 选择对应量化方法。接着 create_weights 创建量化权重和 scale 等参数结构，加载 checkpoint 后通过 process_weights_after_loading 做后处理。推理时调用 quant_method.apply，进入对应量化 backend 或 kernel。

---

### Q14：KV Cache 量化有什么注意点？

**回答：**

KV Cache 量化可以显著降低长上下文显存占用，但必须 attention backend 支持对应低精度 KV 格式。还要处理 scale 保存、读取时反量化、是否与 attention kernel 融合、精度损失和长上下文误差累积等问题。不是简单把 KV tensor dtype 改成 FP8 就结束。

---

### Q15：投机解码为什么能加速？

**回答：**

普通 decode 生成 K 个 token 需要 K 次大模型 forward。投机解码用较小的 draft model 或额外预测头先生成多个候选 token，再用大模型一次 forward 并行验证这些 token。如果接受率较高，就能用更少的大模型 forward 生成更多 token，从而减少 decode 阶段串行开销。

---

### Q16：投机解码一定有收益吗？

**回答：**

不一定。收益取决于 draft model 成本、候选 token 接受率、任务类型、batch size 和验证开销。如果接受率低，或者 batch 已经很大、GPU 已充分利用，投机解码收益可能下降，甚至引入额外开销。

---

### Q17：Medusa、EAGLE、MTP 有什么区别？

**回答：**

Medusa 是在主模型上加多个预测头，直接预测多个未来 token。EAGLE 更像预测未来 hidden state，再映射到 token 空间。MTP 是 Multi-Token Prediction，一些新模型会原生带多 token 预测能力。它们都试图减少逐 token decode 的串行瓶颈，但实现方式和训练需求不同。

---

### Q18：如果面试问“你最推荐新手先学 vLLM 哪些模块”，怎么回答？

**回答：**

我会先学整体执行链路：LLM/API Server → Engine → Scheduler → Executor → Worker → ModelRunner → Model。然后重点学 Scheduler 和 KV Cache/PagedAttention，因为它们决定吞吐、显存和并发。接着学 Executor/Worker，因为这是 nano-vLLM 到真实 vLLM 的重要工程差异。再往后根据岗位方向选择 TP/DP/EP、量化、算子或 PD 分离。

---

### Q19：如果做推理框架岗位，Python 和 C++/CUDA 哪个更重要？

**回答：**

推理框架岗位通常 Python 代码很多，因为调度、模型组织、权重加载、服务逻辑主要在 Python 层。但 C++/CUDA 也要能看懂，尤其是 attention backend、GEMM、量化 kernel、通信后端。是否需要手写 CUDA 取决于岗位。如果是算子岗，CUDA/Triton 是核心；如果是框架岗，更重要的是理解系统链路和能对接 kernel。

---

### Q20：怎么证明自己的 vLLM 项目不是玩具？

**回答：**

要有可运行服务、benchmark 数据、源码理解和对比实验。比如部署 Qwen 模型，提供 OpenAI-compatible API，测 TTFT、TPOT、吞吐、显存；对比不同并发、输入长度、输出长度；分析 prefill/decode 差异；再结合源码说明 Scheduler、KV Cache Manager、Executor、Worker、ModelRunner 如何协作。如果还能做一个小修改或修 bug，就更有说服力。

---

## 六、对你当前学习路线的建议

结合你已经学过 nano-vLLM 的背景，这节课之后建议这样推进：

### 第一阶段：把 vLLM 的“请求链路”彻底串起来

目标是能闭眼说出：

```text
请求进入 → tokenization → Engine → Scheduler → KV Cache Manager → Executor → Worker → ModelRunner → Model.forward → logits → sampling → 流式返回
```

你要重点理解每层为什么存在，而不是只背类名。

---

### 第二阶段：把 nano-vLLM 和 vLLM 对照起来

| nano-vLLM | vLLM | 差异 |
|---|---|---|
| 简化 Engine | Engine / EngineCore | vLLM 支持在线服务、多进程、复杂调度 |
| 简化 Scheduler | Scheduler | vLLM 支持更多调度策略、抢占、prefix cache 等 |
| 简化 BlockManager | KV Cache Manager / Block Pool | vLLM 更复杂，支持多层、多设备、多缓存策略 |
| 直接调用 ModelRunner | Executor → Worker → ModelRunner | vLLM 为多 GPU、多进程抽象出 Executor/Worker |
| 单卡为主 | TP/DP/EP/CP | vLLM 支持工业多卡部署 |
| 基础 FP16/BF16 | 多种量化 backend | vLLM 支持 FP8/AWQ/GPTQ 等 |
| 简单输出 | Streaming + API Server | vLLM 面向在线服务 |

---

### 第三阶段：做一个能写进简历的 benchmark 项目

建议项目结构：

```text
project/
  server/        # vLLM OpenAI API 服务启动脚本
  benchmark/     # 压测脚本
  configs/       # 模型、并发、长度、量化配置
  results/       # CSV / JSON 结果
  analysis/      # 图表和分析
  notes/         # 源码学习笔记
```

核心指标：

- TTFT；
- TPOT；
- end-to-end latency；
- throughput tokens/s；
- requests/s；
- GPU memory；
- GPU utilization；
- prefill/decode 时间占比。

---

### 第四阶段：准备面试追问

你要能回答以下追问：

```text
为什么 decode 是 memory-bound？
为什么 prefill 是 compute-bound？
PagedAttention 和 OS paging 相似在哪里？
block_size 过大/过小分别有什么问题？
Scheduler 如何决定 prefill 和 decode 混 batch？
TP 中 all-reduce 发生在哪些层？
MoE 为什么是 all-to-all？
FP8 量化为什么现在重要？
投机解码接受率低会发生什么？
PD 分离什么时候值得做？
```

---

## 七、本节课最终总结

这节课的主线是：

> vLLM 通过分层架构承载复杂推理优化：上层负责请求接入，中层负责调度和 KV Cache 管理，底层通过 Executor/Worker/ModelRunner 执行模型，并结合 PagedAttention、Prefix Caching、TP/DP/EP/CP、FP8 量化、投机解码等技术解决大模型推理中的显存、吞吐、延迟和扩展性问题。

对面试来说，最重要的不是把所有模块都背下来，而是形成三种能力：

1. **链路能力**：一个请求从输入到输出经过哪些模块。
2. **瓶颈能力**：每种优化解决的是计算、显存、通信还是调度瓶颈。
3. **工程能力**：真实项目中如何部署、压测、分析和修改框架。

你现在已经学过 nano-vLLM，下一步学习 vLLM 时要重点补的是工业系统复杂度：

- API Server / AsyncLLM；
- Engine / EngineCore；
- Scheduler；
- KV Cache Manager；
- Executor / Worker / ModelRunner；
- ZMQ / 多进程通信；
- TP / DP / EP；
- FP8 / AWQ / GPTQ 量化；
- streaming output；
- benchmark 与性能分析。

只要你能把这些内容和自己的部署/源码项目结合起来，就已经能覆盖 AI Infra 推理框架实习面试中的大部分高频问题。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-vllm|模块-vllm]]

%% 项目关联导航：结束 %%
