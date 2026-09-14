# vLLM 源码全流程分析：Executor 与 Worker 组件协作学习笔记

> 适用对象：正在学习 AI Infra / 大模型推理 / vLLM 源码 / 推理引擎架构的初学者。  
> 本节主题：**vLLM 中 Engine、Executor、Worker、ModelRunner 之间如何协作，以及 Executor 如何通过 RPC 将任务分发给 Worker 执行。**  
> 说明：原视频转写存在较多 ASR 识别错误，本文已按 vLLM 语境进行术语校正，例如：`XQ/ExQ` 统一理解为 `Executor`，`Woke/Walker` 统一理解为 `Worker`，`Anding/NG` 统一理解为 `Engine`，`Queen/困` 统一理解为 `Queue`，`CMQ/SerialMQ` 统一理解为 `ZeroMQ`。

---

## 一、这节课的核心宗旨

这节课主要讲 **vLLM 推理框架中 Executor 与 Worker 组件的协作关系**。

前面课程已经讲过：

1. 客户端如何把请求发给 vLLM Engine；
2. Engine 如何接收请求、维护输入队列和输出队列；
3. Engine 如何支持流式返回；
4. vLLM 如何用 ZeroMQ 在前后端进程之间传输请求和结果。

本节课是在这个基础上继续向后走：

> 用户请求进入 Engine 之后，真正负责把请求分发到 GPU / Worker / ModelRunner 去执行的组件是谁？  
> 答案就是：**Executor**。

所以，本节课的技术主线可以概括为：

```text
客户端请求
  ↓
Engine 接收请求
  ↓
Engine 将请求放入 input_queue
  ↓
Executor 从 input_queue 取请求
  ↓
Executor 根据调度结果向 Worker 下发执行命令
  ↓
Worker 调用 ModelRunner / Model 执行 forward
  ↓
Worker 将执行结果返回给 Executor
  ↓
Executor 将结果写回 Engine output_queue
  ↓
Engine 再把结果流式返回给客户端
```

这节课真正想让你理解的是：

- **Engine 不直接跑模型**，它更像请求入口和结果出口；
- **Executor 是 Engine 和 Worker 之间的调度执行中枢**；
- **Worker 也不是最终模型计算本体**，它会进一步调用 `ModelRunner` 和模型的 `forward`；
- 多卡推理时会有多个 Worker，每个 Worker 通常对应一个 rank / GPU / 部分模型权重；
- Executor 与 Worker 之间通过 RPC 形式通信，本质是传递“要调用的方法名 + 方法参数”；
- vLLM 为了高效传输，内部结合了共享内存 / Ring Buffer 和 ZeroMQ，不同大小的数据走不同传输路径。

---

## 二、本节课在 AI Infra 学习路线中的位置

如果你把 vLLM 看成一个完整推理系统，可以粗略拆成下面几层：

```text
服务接口层：OpenAI API / HTTP / AsyncLLM / 客户端
  ↓
Engine 层：接收请求、维护 input_queue / output_queue、流式返回
  ↓
Executor 层：从队列取任务、调度执行、向 Worker 广播命令
  ↓
Worker 层：每个进程 / 每张 GPU / 每个 rank 的实际执行入口
  ↓
ModelRunner 层：准备输入、KV Cache、attention metadata、执行模型 forward
  ↓
Model / CUDA Kernel 层：Transformer 计算、PagedAttention、采样等
```

前几节课重点在 **Engine、ZeroMQ、流式返回、KV Cache 分块管理**。本节课进入 **Executor + Worker**，属于从“服务层”走向“执行层”的关键过渡。

对于 AI Infra / 推理引擎岗位来说，这一节非常重要，因为它对应真实推理系统中的几个核心能力：

1. **请求如何从服务层进入计算层**；
2. **多进程 / 多卡推理如何组织 Worker**；
3. **Executor 如何向多个 Worker 分发同一个执行命令**；
4. **RPC 在推理框架内部如何落地**；
5. **小数据和大数据如何选择不同 IPC 通信方式**；
6. **Engine、Executor、Worker、ModelRunner 的边界如何划分**。

这也是面试中非常容易考察的内容，因为它不是单纯 Transformer 原理，而是推理框架工程实现。

---

## 三、术语校正表

由于视频转写中有较多自动识别错误，先把核心术语统一：

| 转写中可能出现的词 | 正确理解 | 解释 |
|---|---|---|
| VR / VLM / VM | vLLM | 大模型推理框架 |
| XQ / ExQ / EQ / XQT | Executor | 执行器，负责将 Engine 的任务分发给 Worker |
| Woke / Walker / Woker | Worker | 工作进程 / 工作节点，负责调用 ModelRunner 执行模型 |
| Anding / NG | Engine | 引擎模块，负责接收请求和返回结果 |
| Queen / 困 / 堆列 | Queue | 队列，如 input_queue、output_queue |
| CMQ / SerialMQ | ZeroMQ / ZMQ | 进程间通信库 |
| BrotherCastMQ | Broadcast Message Queue | 广播消息队列，用于 Executor 向多个 Worker 下发命令 |
| ResponseMQ | Worker Response Message Queue | Worker 返回结果给 Executor 的队列 |
| Model Rona | ModelRunner | 模型执行封装层 |
| 张亮并行 / 张量变形 | Tensor Parallel, TP | 张量并行，多 GPU 拆分模型权重或计算 |
| Rank0 到 N | rank 0 到 rank N-1 | 分布式并行中的进程编号 |
| EQ model / Excode model | execute_model | 执行模型推理的方法 |
| schedule output | SchedulerOutput | 调度器输出，告诉本轮要执行哪些请求 / token |

---

## 四、按照课程推进顺序梳理知识点

### 1. 回顾整体架构：Engine 之后是 Executor

本节一开始先回顾 vLLM 的整体架构。

在上一节课中，重点讲的是 **Engine 模块**。Engine 的主要职责是：

- 接收客户端请求；
- 把请求放入自己的输入队列 `input_queue`；
- 等待后端推理完成；
- 从输出队列 `output_queue` 获取结果；
- 将结果返回给客户端，支持流式输出。

但是 Engine 本身并不直接执行模型计算。请求进入 Engine 后，还需要继续传给后面的执行组件。这个执行组件就是本节课的重点：**Executor**。

Executor 的位置如下：

```text
Client / AsyncLLM
  ↓
Engine
  ↓
Executor
  ↓
Worker
  ↓
ModelRunner
  ↓
Model.forward / CUDA Kernel
```

Executor 起到的是 **承上启下** 的作用：

- 向上连接 Engine；
- 向下连接 Worker；
- 从 Engine 获取请求或调度任务；
- 将任务进一步派发给 Worker；
- 汇总 Worker 的执行结果；
- 再把结果交回 Engine。

#### 在 AI Infra 中的作用

这对应推理系统中的 **执行调度层**。很多同学只知道“模型 forward 一下就生成 token”，但真正的在线推理服务不是这样简单。它需要：

- 多请求并发；
- 多 GPU 协同；
- Worker 进程管理；
- 通信队列；
- 调度结果下发；
- 执行结果回收；
- 流式返回。

Executor 正是把这些工程问题组织起来的关键模块。

---

### 2. Worker 并不是最终计算者，而是调用 ModelRunner

课程中强调：Worker 不是模型实际计算的最底层执行者。

Worker 的职责更像一个 **GPU 工作进程入口**，它接收 Executor 下发的命令，然后调用更底层的模块，例如：

- `ModelRunner`；
- 具体模型对象；
- `model.forward`；
- attention backend；
- PagedAttention；
- sampling 相关逻辑。

也就是说，Worker 的角色可以理解为：

```text
Worker = 接收执行命令 + 准备调用环境 + 调用 ModelRunner + 返回结果
```

Worker 收到类似下面的 RPC 命令：

```text
方法名：execute_model
参数：SchedulerOutput / 执行请求信息 / KV cache 信息等
```

然后 Worker 内部会执行类似：

```python
worker.execute_model(scheduler_output)
```

再进一步调用：

```python
model_runner.execute_model(...)
```

最后才真正进入模型前向计算。

#### 在 AI Infra 中的作用

这体现了推理框架的分层设计：

- Executor 不关心模型内部细节；
- Worker 不负责调度策略；
- ModelRunner 负责模型执行前的数据组织；
- Model / Kernel 负责数学计算。

这种分层让 vLLM 可以更容易支持：

- 单卡执行；
- 多卡 TP；
- 多进程 Worker；
- 不同模型结构；
- 不同 attention backend；
- 量化模块；
- KV Cache 管理模块。

---

### 3. 为什么会有多个 Worker：Tensor Parallel 与 rank

课程中提到，如果配置了张量并行，例如：

```text
tensor_parallel_size = 2
```

那么通常会有两个 Worker。

这是因为在 Tensor Parallel 中，模型权重和计算会被拆分到多个 GPU / 多个 rank 上。例如：

```text
Worker 0 / rank 0 → 持有模型的一部分权重
Worker 1 / rank 1 → 持有模型的另一部分权重
```

它们共同完成一次模型推理。

可以理解为：

```text
完整模型
  ↓ 按张量维度切分
GPU 0 上的模型分片 + GPU 1 上的模型分片
  ↓
两个 Worker 协同执行
```

在这种情况下，Executor 需要把同一个执行命令广播给多个 Worker：

```text
Executor
  ├── Worker rank 0
  └── Worker rank 1
```

每个 Worker 根据自己的 rank 执行自己负责的那部分计算。

#### 在 AI Infra 中的作用

这就是推理引擎支持大模型部署的基础。

如果模型太大，单张 GPU 放不下完整权重，就需要多卡并行。此时系统必须解决：

- Worker 数量如何创建；
- 每个 Worker 对应哪个 rank；
- 每个 Worker 加载哪部分权重；
- Executor 如何广播命令；
- Worker 如何同步执行；
- 执行结果如何汇总。

---

### 4. RPC Demo：Executor 如何远程调用 Worker 方法

课程中用一个简单 RPC Demo 帮助理解 Executor 和 Worker 的关系。

RPC 是 **Remote Procedure Call，远程过程调用**。

本质含义是：

> 调用方看起来像在调用一个普通函数，但这个函数实际在另一个进程、另一台机器、或者另一个 Worker 中执行。

例如 Executor 想让 Worker 执行一个函数：

```text
func = demo
args = (1, 2, "str")
```

Executor 不会真的在自己进程里执行 `demo(1, 2, "str")`，而是把：

```text
方法名：demo
参数：1, 2, "str"
```

通过通信队列发送给 Worker。

Worker 收到之后，在自己的进程中执行：

```python
demo(1, 2, "str")
```

然后把执行结果返回给 Executor。

在 vLLM 中，这个思想对应：

```text
Executor 发送：execute_model(scheduler_output)
Worker 接收：执行 execute_model
Worker 内部：调用 ModelRunner.execute_model
Worker 返回：模型执行结果
```

#### 为什么不用直接函数调用？

因为 Executor 和 Worker 可能不在同一个进程中，甚至可能对应不同 GPU。

普通函数调用只能在当前进程中执行，而 vLLM 需要跨进程 / 多 GPU 协同，因此要通过 RPC 思想传递命令。

---

### 5. Executor 类与派生类：多进程 Executor

课程中提到，Executor 有多个派生类。

在不同执行环境下，vLLM 可能使用不同 Executor，例如：

- 单进程 Executor；
- 多进程 Executor；
- Ray Executor；
- Multiprocessing Executor；
- GPU Executor 等。

本节课中使用的是多卡配置，所以关注的是类似：

```text
Multiprocessing Executor / MultiProcExecutor
```

它是 Executor 的一个派生类，用来管理多个 Worker 进程。

课程中还提到 `distributed_backend`，它用于决定使用哪种分布式后端。例如：

```text
distributed_backend = mp
```

这里的 `mp` 可以理解为 multiprocessing，多进程后端。

#### 在 AI Infra 中的作用

不同部署方式需要不同 Executor：

| 场景 | 可能使用的执行后端 |
|---|---|
| 单卡本地推理 | 单进程 / 本地 Executor |
| 单机多卡 TP | multiprocessing Executor |
| 多机多卡 | Ray / 分布式 Executor |
| 离线批处理 | 可能使用更简单的执行路径 |
| 在线服务 | 需要 Engine + Executor + Worker 协作 |

Executor 的抽象使 vLLM 能够在不同硬件和部署环境中复用上层逻辑。

---

### 6. Executor 和 Worker 之间的两个核心通信队列

课程中重点提到两个队列：

```text
RPCBroadcastMQ
WorkerResponseMQ
```

它们分别负责两个方向的通信。

#### 6.1 RPCBroadcastMQ：Executor → Worker

`RPCBroadcastMQ` 负责从 Executor 向 Worker 发送命令。

例如：

```text
Executor 发送：execute_model + scheduler_output
Worker 接收：方法名 + 参数
```

它叫 Broadcast，是因为 Executor 可能要把同一个命令广播给多个 Worker。

例如 TP=2 时：

```text
Executor
  ├── 向 Worker 0 发送 execute_model
  └── 向 Worker 1 发送 execute_model
```

#### 6.2 WorkerResponseMQ：Worker → Executor

`WorkerResponseMQ` 负责 Worker 把执行结果返回给 Executor。

例如：

```text
Worker 执行完模型 forward
  ↓
将输出结果写入 WorkerResponseMQ
  ↓
Executor 从 WorkerResponseMQ 获取结果
```

#### 总结

```text
Executor --RPCBroadcastMQ--> Worker
Executor <--WorkerResponseMQ-- Worker
```

这两个队列分别对应：

- 命令下发；
- 结果回收。

---

### 7. Executor 和 Worker 建立通信前需要握手

课程中讲到，Executor 和 Worker 之间不是一开始就能直接通信，需要先建立连接。

连接建立过程可以理解为一次“握手”。

大致流程如下：

```text
1. Worker 作为读端，向 Executor 写端发送订阅消息 SUB
2. Executor 收到 Worker 的订阅消息
3. Executor 向 Worker 返回 READY
4. Worker 收到 READY
5. 双方确认通信队列已经建立
6. 后续可以正式读写消息
```

用更直观的方式表示：

```text
Worker  ── SUB ──> Executor
Worker  <─ READY ─ Executor
Worker  ── 开始接收命令 ── Executor
```

为什么需要这个过程？

因为 Executor 向 Worker 广播命令之前，必须确认 Worker 已经准备好接收消息。如果 Worker 还没订阅完成，Executor 就发送命令，可能会导致消息丢失或阻塞。

#### 在 AI Infra 中的作用

这属于分布式系统中的基础问题：

- 进程启动顺序不确定；
- Worker 初始化可能比 Executor 慢；
- 通信通道必须先建立；
- 调用方必须知道接收方 ready；
- 否则任务下发可能失败。

推理框架不是只有模型计算，还要处理大量系统工程问题。

---

### 8. MessageQueue 的混合传输机制：共享内存 + ZeroMQ

课程中提到，Executor 和 Worker 之间的 `MessageQueue` 并不是单一通信方式，而是混合了：

1. **共享内存 / Ring Buffer**；
2. **ZeroMQ socket**。

核心策略是：

```text
如果消息较小 → 使用共享内存 / Ring Buffer
如果消息较大 → 使用 ZeroMQ socket
```

课程中提到一个阈值：

```text
16 MB
```

即：

```text
消息大小 <= 16 MB：走共享内存 / Ring Buffer
消息大小 > 16 MB：走 ZeroMQ
```

注意：不同版本实现细节可能会变化，但这节课强调的是这个设计思想。

#### 为什么小消息适合共享内存？

共享内存的优势是：

- 少一次数据复制；
- 进程间传递开销低；
- 适合频繁传递小型控制消息；
- 延迟更低。

例如方法名、少量参数、调度元信息等，通常可以走共享内存。

#### 为什么大消息可能走 ZeroMQ？

大数据传输可能涉及：

- 序列化；
- 跨进程传递；
- buffer 管理；
- 阻塞控制；
- 网络 / socket 语义。

ZeroMQ 更适合处理这类较大或更通用的消息传递。

#### 在 AI Infra 中的作用

推理系统中通信成本很重要。一次 token 生成可能很快，但如果每次调度都因为进程间通信浪费大量时间，就会降低吞吐量。

所以 vLLM 这类框架会尽量：

- 小消息走低开销路径；
- 大消息走稳定通信路径；
- 控制数据复制；
- 避免通信成为瓶颈。

---

### 9. `enqueue` / `dequeue`：Executor 下发命令，Worker 接收命令

课程中提到：

```text
enqueue / dequeue
```

可以简单理解为：

- `enqueue`：入队，发送消息；
- `dequeue`：出队，接收消息。

当 Executor 想让 Worker 执行模型时，会做类似：

```text
RPCBroadcastMQ.enqueue(
    method_name = "execute_model",
    args = scheduler_output,
    ...
)
```

Worker 侧则会：

```text
method_name, args = RPCBroadcastMQ.dequeue()
```

然后根据 `method_name` 找到对应方法并执行：

```python
getattr(worker, method_name)(*args)
```

如果方法名是 `execute_model`，那就执行：

```python
worker.execute_model(scheduler_output)
```

#### 这里传递的不是“函数本身”，而是“函数名 + 参数”

这是 RPC 的核心。

普通函数调用：

```python
worker.execute_model(scheduler_output)
```

RPC 形式：

```text
发送："execute_model" + scheduler_output
接收：根据字符串找到 execute_model 方法并执行
```

#### 在 AI Infra 中的作用

这使得 Executor 可以统一管理 Worker，而不需要知道 Worker 内部具体如何执行。Executor 只需要发命令：

```text
你执行 execute_model，这些是参数
```

Worker 负责真正执行。

---

### 10. `execute_model`：Worker 调用 ModelRunner 执行推理

当 Worker 收到 `execute_model` 命令后，会进入模型执行流程。

这里的参数通常包含调度器输出，例如：

```text
SchedulerOutput
```

它代表本轮调度要执行的内容，例如：

- 哪些请求进入本轮 batch；
- 每个请求执行 prefill 还是 decode；
- 每个请求本轮要处理多少 token；
- 需要使用哪些 KV Cache block；
- 需要的 attention metadata；
- 采样相关信息等。

Worker 接收后，会进一步调用 ModelRunner：

```text
Worker.execute_model
  ↓
ModelRunner.execute_model
  ↓
Model.forward
  ↓
Attention / MLP / Sampling
```

#### Worker 和 ModelRunner 的区别

| 组件 | 主要职责 |
|---|---|
| Worker | GPU 工作进程入口，接收 Executor 命令，管理本 rank 执行 |
| ModelRunner | 组织模型输入、KV Cache、attention metadata，并调用模型 forward |
| Model | Transformer 网络本体 |
| Kernel | CUDA 层面的具体算子执行 |

一句话区分：

> Worker 负责“接活”，ModelRunner 负责“组织活怎么干”，Model / Kernel 负责“真正干活”。

---

### 11. Worker 执行结束后，结果通过 WorkerResponseMQ 返回 Executor

Worker 执行完 `execute_model` 之后，会得到输出结果，例如：

- 生成 token；
- logprobs；
- 请求状态更新；
- 是否结束；
- 采样结果；
- 本轮输出结构。

然后 Worker 把这些结果写入：

```text
WorkerResponseMQ
```

Executor 通过类似 `get_response` 的逻辑从该队列中取回结果。

流程如下：

```text
Worker 执行模型
  ↓
Worker 将结果 enqueue 到 WorkerResponseMQ
  ↓
Executor 从 WorkerResponseMQ dequeue / get_response
  ↓
Executor 得到执行结果
```

得到结果后，Executor 再把结果交回 Engine 的输出队列：

```text
Executor
  ↓
Engine.output_queue
```

然后 Engine 再按上一节课讲过的流式返回链路，把 token 推回客户端。

---

## 五、从输入到输出串联完整流程

下面把整节课内容串成一条完整链路。

### 1. 用户输入 prompt

用户通过客户端输入：

```text
用 20 个字介绍 vLLM
```

客户端会经过：

```text
文本 prompt
  ↓ tokenizer
prompt_token_ids
```

然后通过 API / AsyncLLM / Engine 接口发送到 vLLM。

---

### 2. Engine 接收请求

Engine 负责接收请求，并把请求放入输入队列：

```text
Engine.input_queue
```

此时请求还没有真正执行模型，只是进入了 vLLM 后端执行流程。

---

### 3. Executor 从 Engine 获取任务

Executor 从 Engine 的输入队列中取出请求，并结合调度器结果决定本轮要执行哪些请求。

调度后得到类似：

```text
SchedulerOutput
```

它告诉后端：

```text
本轮哪些请求要执行？
每个请求执行哪些 token？
需要哪些 KV Cache block？
哪些请求是 prefill？哪些是 decode？
```

---

### 4. Executor 通过 RPCBroadcastMQ 向 Worker 下发命令

Executor 不直接执行模型，而是通过 RPC 方式给 Worker 发送命令。

发送内容可以理解为：

```text
方法名：execute_model
参数：SchedulerOutput
```

如果有多个 Worker，例如 TP=2，则 Executor 向多个 Worker 广播：

```text
Executor
  ├── execute_model → Worker rank 0
  └── execute_model → Worker rank 1
```

---

### 5. Worker 从 RPCBroadcastMQ 接收命令

Worker 通过 `dequeue` 获取消息。

它拿到的是：

```text
method_name = "execute_model"
args = scheduler_output
```

然后执行：

```python
worker.execute_model(scheduler_output)
```

---

### 6. Worker 调用 ModelRunner

Worker 自己不是最低层计算者，它会调用：

```text
ModelRunner.execute_model
```

ModelRunner 会准备模型 forward 所需的数据，例如：

- input_ids；
- positions；
- KV Cache；
- block table；
- slot mapping；
- attention metadata；
- sampling metadata。

然后真正调用模型。

---

### 7. Model / Kernel 执行计算

模型执行 Transformer forward：

```text
Embedding
  ↓
Transformer layers
  ↓
Attention + KV Cache
  ↓
MLP
  ↓
logits
  ↓
sampling
  ↓
next token
```

这一阶段才是真正占用 GPU 算力的部分。

---

### 8. Worker 将结果写入 WorkerResponseMQ

模型执行完后，Worker 拿到本轮结果，然后写入：

```text
WorkerResponseMQ
```

---

### 9. Executor 获取 Worker 执行结果

Executor 从 `WorkerResponseMQ` 中获取结果。

如果是多 Worker / 多 rank 场景，Executor 需要等待相应 Worker 返回，或者根据 rank 组织结果。

---

### 10. Executor 将结果交回 Engine

Executor 将执行结果放入 Engine 的输出队列：

```text
Engine.output_queue
```

---

### 11. Engine 流式返回客户端

Engine 根据请求 ID 找到对应客户端或对应请求的输出队列，然后逐步返回 token。

最终用户看到的是类似 ChatGPT 一样的效果：

```text
vLLM 是...
```

一个 token 或一个片段一个片段地返回。

---

## 六、核心流程图

### 1. 总体模块图

```text
┌────────────┐
│  Client    │
└─────┬──────┘
      │ HTTP / API / Async request
      ▼
┌────────────┐
│   Engine   │
│ input_queue│
│output_queue│
└─────┬──────┘
      │
      ▼
┌────────────┐
│  Executor  │
└─────┬──────┘
      │ RPCBroadcastMQ: execute_model(args)
      ▼
┌────────────┐
│   Worker   │
└─────┬──────┘
      │
      ▼
┌────────────┐
│ModelRunner │
└─────┬──────┘
      │
      ▼
┌────────────┐
│   Model    │
└────────────┘
```

### 2. Executor 与 Worker 通信图

```text
                 命令下发
Executor ─────────────────────> Worker
          RPCBroadcastMQ
          method = execute_model
          args = SchedulerOutput

                 结果返回
Executor <───────────────────── Worker
          WorkerResponseMQ
          output = model result
```

### 3. 多 Worker / TP 场景

```text
              ┌──────────────┐
              │   Executor   │
              └──────┬───────┘
                     │ broadcast execute_model
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   Worker 0     Worker 1      Worker N-1
   rank 0       rank 1        rank N-1
      │            │             │
      ▼            ▼             ▼
 GPU 0 shard   GPU 1 shard    GPU N shard
```

---

## 七、几个容易混淆的问题

### 1. Engine 和 Executor 有什么区别？

Engine 更靠近用户请求入口，负责请求接收、输入输出队列、流式返回。

Executor 更靠近后端执行，负责把 Engine 交来的任务分发给 Worker 执行。

```text
Engine：管请求入口和结果出口
Executor：管任务分发和执行协调
```

---

### 2. Executor 和 Worker 有什么区别？

Executor 是调度和分发中心，Worker 是具体执行入口。

```text
Executor：告诉 Worker 要干什么
Worker：接收命令并调用 ModelRunner 去干
```

---

### 3. Worker 和 ModelRunner 有什么区别？

Worker 是进程 / rank / GPU 级别的工作单元。

ModelRunner 是模型执行封装层，负责准备模型输入并调用 forward。

```text
Worker：进程级执行入口
ModelRunner：模型执行准备与调用封装
```

---

### 4. RPC 传递的是函数吗？

不是直接传递函数对象，而是传递：

```text
方法名 + 参数
```

Worker 收到后，根据方法名找到对应函数并执行。

---

### 5. 为什么需要两个队列？

因为通信有两个方向：

```text
Executor → Worker：下发命令
Worker → Executor：返回结果
```

所以需要：

```text
RPCBroadcastMQ：命令下发
WorkerResponseMQ：结果返回
```

---

### 6. 为什么要有握手过程？

因为 Worker 必须先准备好接收消息。Executor 不能在 Worker 未 ready 时直接广播任务，否则可能出现消息丢失或阻塞。

握手过程保证：

```text
Worker 已订阅
Executor 已确认
双方通信通道建立
```

---

### 7. 为什么小消息和大消息要走不同通信路径？

因为通信方式的性能特点不同：

| 消息类型 | 更适合的方式 | 原因 |
|---|---|---|
| 小消息 | 共享内存 / Ring Buffer | 低延迟，少拷贝 |
| 大消息 | ZeroMQ | 更适合复杂或大体量传输 |

推理系统中频繁传递调度元信息，如果每次都走高开销路径，会降低吞吐量。

---

## 八、结合源码理解的伪代码

下面用伪代码模拟本节课讲的核心逻辑。

### 1. Executor 下发执行命令

```python
class Executor:
    def execute_model(self, scheduler_output):
        # 1. 构造 RPC 消息
        method_name = "execute_model"
        args = (scheduler_output,)

        # 2. 广播给 Worker
        self.rpc_broadcast_mq.enqueue(method_name, args)

        # 3. 等待 Worker 返回结果
        outputs = self.worker_response_mq.dequeue()

        # 4. 返回给 Engine
        return outputs
```

### 2. Worker 接收命令并执行

```python
class Worker:
    def run(self):
        while True:
            # 1. 从 Executor 接收命令
            method_name, args = self.rpc_broadcast_mq.dequeue()

            # 2. 根据方法名找到函数
            method = getattr(self, method_name)

            # 3. 执行函数
            result = method(*args)

            # 4. 返回结果给 Executor
            self.worker_response_mq.enqueue(result)

    def execute_model(self, scheduler_output):
        return self.model_runner.execute_model(scheduler_output)
```

### 3. MessageQueue 根据大小选择传输方式

```python
class MessageQueue:
    def enqueue(self, data):
        serialized = serialize(data)

        if len(serialized) <= 16 * 1024 * 1024:
            self.ring_buffer.write(serialized)
        else:
            self.zmq_socket.send(serialized)

    def dequeue(self):
        flag = self.read_flag()

        if flag == "shared_memory":
            return self.ring_buffer.read()
        else:
            return self.zmq_socket.recv()
```

这段伪代码不是 vLLM 源码原文，而是为了帮助理解本节课讲到的设计思想。

---

## 九、和前几节课内容的衔接

### 1. 和 Engine / 流式推理的关系

上一节课讲：

```text
客户端 ↔ Engine ↔ input_queue / output_queue ↔ 流式返回
```

本节课补上了 Engine 后面的部分：

```text
input_queue → Executor → Worker → ModelRunner → output_queue
```

所以两节课合起来就是：

```text
客户端
  ↓
Engine 接收请求
  ↓
Executor 分发任务
  ↓
Worker 执行模型
  ↓
Executor 收集结果
  ↓
Engine 流式返回
  ↓
客户端收到输出
```

---

### 2. 和 KV Cache / PagedAttention 的关系

Executor 与 Worker 负责的是“谁来执行”和“如何下发执行”。

KV Cache / PagedAttention 负责的是“执行时显存如何组织”。

它们属于不同层次：

| 层次 | 关注点 |
|---|---|
| Executor / Worker | 多进程、多卡、任务分发、执行协调 |
| KV Cache / PagedAttention | 显存管理、block 映射、attention 读取 KV |
| ModelRunner | 把调度结果转换成模型可执行输入 |

当 Worker 执行 `execute_model` 时，ModelRunner 会使用前面课程讲过的：

- KV Cache block；
- block table；
- slot mapping；
- query_start_loc；
- num_computed_tokens；
- attention metadata。

也就是说：

```text
Executor / Worker 决定“任务交给谁执行”
PagedAttention / KV Cache 决定“执行时 KV 怎么存、怎么读”
```

---

### 3. 和调度器 Scheduler 的关系

Scheduler 负责决定：

```text
本轮哪些请求能执行
每个请求执行多少 token
哪些请求进入 prefill
哪些请求进入 decode
KV Cache block 是否足够
```

Executor 拿到 Scheduler 的结果后，把它下发给 Worker。

所以：

```text
Scheduler：做决策
Executor：派发决策
Worker：执行决策
ModelRunner：把决策转换成模型输入
```

---

## 十、面试常见问题与参考答案

### 问题 1：vLLM 中 Engine、Executor、Worker、ModelRunner 分别负责什么？

**参考答案：**

Engine 负责接收用户请求、维护输入输出队列，并把结果流式返回给客户端。Executor 位于 Engine 和 Worker 之间，负责从 Engine 获取任务，并将任务分发给一个或多个 Worker。Worker 是每个进程或 GPU rank 上的执行入口，接收 Executor 的 RPC 命令，然后调用 ModelRunner。ModelRunner 负责准备模型 forward 所需的输入、KV Cache、attention metadata 等，并最终调用模型执行。

---

### 问题 2：为什么 vLLM 不让 Engine 直接调用模型 forward？

**参考答案：**

因为在线推理系统不仅要执行模型，还要处理多请求并发、多进程、多 GPU、流式返回、调度、KV Cache 管理等问题。Engine 更适合作为请求入口和结果出口。如果让 Engine 直接执行模型，会导致服务层和执行层耦合严重，不利于多卡扩展和分布式部署。vLLM 使用 Executor 和 Worker 将执行逻辑解耦，使 Engine 专注请求管理，Executor 专注任务分发，Worker 专注模型执行。

---

### 问题 3：Executor 在 vLLM 中起什么作用？

**参考答案：**

Executor 是连接 Engine 和 Worker 的执行调度层。它从 Engine 或调度器获得本轮要执行的任务，然后通过 RPCBroadcastMQ 将方法名和参数广播给 Worker，例如 `execute_model(scheduler_output)`。Worker 执行完后，会通过 WorkerResponseMQ 将结果返回给 Executor。Executor 再把结果交给 Engine 输出队列。

---

### 问题 4：Worker 是不是直接完成 Transformer 计算？

**参考答案：**

不是完全准确。Worker 是执行入口，它接收 Executor 的命令，并调用 ModelRunner。ModelRunner 再组织输入、KV Cache、attention metadata 等，最后调用模型的 forward。真正的矩阵计算、attention、MLP、采样等发生在模型和 CUDA kernel 层。

---

### 问题 5：Executor 和 Worker 之间为什么要使用 RPC？

**参考答案：**

因为 Executor 和 Worker 可能处于不同进程，Worker 还可能对应不同 GPU / rank。普通函数调用无法跨进程执行。RPC 的方式是将“方法名 + 参数”发送给 Worker，Worker 收到后在自己的进程中调用对应方法。这样可以支持多进程、多卡和分布式执行。

---

### 问题 6：RPCBroadcastMQ 和 WorkerResponseMQ 分别有什么作用？

**参考答案：**

RPCBroadcastMQ 用于 Executor 向 Worker 下发执行命令，例如 `execute_model` 及其参数。WorkerResponseMQ 用于 Worker 执行完成后将结果返回给 Executor。前者是命令下发通道，后者是结果返回通道。

---

### 问题 7：为什么叫 BroadcastMQ？

**参考答案：**

因为在多 Worker 场景下，Executor 可能要把同一个执行命令广播给多个 Worker。例如 tensor parallel size 为 2 时，会有两个 Worker 分别对应 rank 0 和 rank 1，Executor 需要同时向它们下发 `execute_model` 命令。

---

### 问题 8：Tensor Parallel 下 Worker 数量和什么有关？

**参考答案：**

Worker 数量通常和并行配置有关，特别是 tensor parallel size。比如 TP=2 时，通常会有两个 Worker，分别对应不同 rank / GPU，每个 Worker 持有模型权重的一部分，并执行对应的计算分片。

---

### 问题 9：Executor 和 Worker 建立连接时为什么需要握手？

**参考答案：**

因为 Worker 需要先订阅并准备好接收 Executor 的消息。如果 Executor 在 Worker 未 ready 时发送命令，可能导致消息丢失或阻塞。握手过程通常包括 Worker 发送订阅消息，Executor 返回 READY，双方确认通信队列建立后再正式传输任务。

---

### 问题 10：vLLM 内部为什么要结合共享内存和 ZeroMQ？

**参考答案：**

因为不同大小的数据适合不同通信路径。小消息如方法名、调度元信息适合走共享内存或 Ring Buffer，开销低、延迟小；大消息更适合走 ZeroMQ socket，通信机制更通用。混合传输可以降低进程间通信成本，提高推理系统吞吐。

---

### 问题 11：`execute_model` 这个方法在整条链路中处于什么位置？

**参考答案：**

`execute_model` 是 Worker 侧接收 Executor 命令后执行模型推理的核心入口。Executor 会通过 RPC 下发 `execute_model` 和调度结果，Worker 收到后调用自己的 `execute_model`，内部再调用 ModelRunner 的 `execute_model`，最终进入模型 forward。

---

### 问题 12：SchedulerOutput 在 Executor 和 Worker 协作中有什么作用？

**参考答案：**

SchedulerOutput 是调度器对本轮执行的决策结果，包含哪些请求要执行、执行多少 token、prefill/decode 状态、KV Cache 相关信息等。Executor 把 SchedulerOutput 作为参数发送给 Worker，Worker 和 ModelRunner 根据它组织模型输入并执行本轮推理。

---

### 问题 13：Engine 的 output_queue 和 WorkerResponseMQ 有什么区别？

**参考答案：**

WorkerResponseMQ 是 Worker 返回执行结果给 Executor 的内部通信队列。Engine 的 output_queue 是 Executor 将结果交还给 Engine 后，由 Engine 用于向客户端返回结果的队列。前者属于 Executor-Worker 层，后者属于 Engine-Client 返回链路。

---

### 问题 14：为什么说 Executor 是承上启下的组件？

**参考答案：**

因为 Executor 向上连接 Engine，从 Engine 获取请求和调度任务；向下连接 Worker，通过 RPC 将执行命令发送给 Worker，并收集 Worker 的结果。它既不直接负责客户端通信，也不直接完成模型底层计算，而是负责执行层任务分发和协调。

---

### 问题 15：如果让你用一句话总结本节课，你会怎么说？

**参考答案：**

本节课讲的是：vLLM 中 Engine 接收请求后，Executor 如何通过 RPCBroadcastMQ 将执行命令分发给多个 Worker，Worker 再调用 ModelRunner 执行模型，并通过 WorkerResponseMQ 将结果返回给 Executor，最后结果再回到 Engine 进行流式输出。

---

## 十一、学习本节课时应该抓住的主线

不要把本节课理解成“讲了几个队列名字”。真正要抓住的是推理系统的分层执行逻辑：

```text
服务入口：Engine
执行中枢：Executor
进程 / GPU 执行单元：Worker
模型执行封装：ModelRunner
实际计算：Model.forward + CUDA Kernel
```

如果你能回答下面几个问题，就说明本节课基本理解了：

1. 用户请求进入 Engine 后，为什么还要经过 Executor？
2. Executor 为什么需要向 Worker 发送方法名和参数？
3. Worker 收到 `execute_model` 后到底做什么？
4. 多卡场景下为什么有多个 Worker？
5. RPCBroadcastMQ 和 WorkerResponseMQ 分别负责什么？
6. 为什么通信队列要有握手过程？
7. 为什么消息传输要区分共享内存和 ZeroMQ？
8. Executor / Worker 和前面学过的 KV Cache、PagedAttention、slot mapping 有什么关系？

---

## 十二、给初学者的理解方式

你可以把 vLLM 推理过程类比成一个餐厅系统：

| vLLM 组件 | 餐厅类比 | 作用 |
|---|---|---|
| Client | 顾客 | 发起点餐请求 |
| Engine | 前台 | 接单、记录订单、通知顾客结果 |
| Executor | 店长 / 调度员 | 决定把订单分给哪个后厨工位 |
| Worker | 后厨工位 | 接收任务，开始做菜 |
| ModelRunner | 厨师的操作流程 | 准备食材、安排步骤、实际做菜前组织流程 |
| Model / CUDA Kernel | 炉灶和厨具 | 真正完成计算 |
| RPCBroadcastMQ | 店长喊话系统 | 把任务下发给工位 |
| WorkerResponseMQ | 后厨反馈通道 | 工位告诉店长做好了 |
| output_queue | 出餐口 | 把结果交给前台返回顾客 |

这样你就能理解：

- 前台不做菜；
- 店长不直接炒菜；
- 工位接活后还要按具体流程做菜；
- 多个工位可以并行工作；
- 沟通通道必须稳定，否则餐厅就乱了。

---

## 十三、最终总结

本节课是 vLLM 源码全流程中的关键一环，主题是 **Executor 与 Worker 的协作**。

核心结论如下：

1. **Engine 负责请求入口和结果出口，不直接执行模型。**
2. **Executor 是 Engine 和 Worker 之间的执行中枢。**
3. **Worker 是每个进程 / GPU / rank 上的执行入口。**
4. **Worker 内部会进一步调用 ModelRunner，ModelRunner 再调用模型 forward。**
5. **Executor 通过 RPCBroadcastMQ 向 Worker 下发方法名和参数。**
6. **Worker 执行完成后通过 WorkerResponseMQ 返回结果。**
7. **多卡 TP 场景下会有多个 Worker，每个 Worker 对应不同 rank。**
8. **Executor 和 Worker 通信前需要握手，保证双方 ready。**
9. **vLLM 内部通信可能结合共享内存 / Ring Buffer 和 ZeroMQ，以降低 IPC 成本。**
10. **本节课连接了上一节 Engine 流式返回和后续 ModelRunner / Model 执行，是理解 vLLM 全链路架构的核心桥梁。**

一句话记忆：

> **Engine 接请求，Executor 派任务，Worker 接命令，ModelRunner 跑模型，结果再原路返回。**


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-vllm|模块-vllm]]

%% 项目关联导航：结束 %%
