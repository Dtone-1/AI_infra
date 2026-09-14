# vLLM 引擎架构与流式推理：课程深度整理学习笔记

> 本笔记根据课程视频转文字文件整理。原始转写中存在自动识别误差，例如“VLM”多处应理解为 **vLLM**，“cmq / SerialMQ / CaroMQ”多处应理解为 **ZeroMQ / ZMQ**，“安静”多处应理解为 **Engine / 引擎**，“流氏执行”应理解为 **流式执行 / streaming execution**。本笔记已在不改变课程主线的前提下做了术语校正和结构化整理。

---

## 0. 本节课在 vLLM / AI Infra 学习路线中的位置

前几节课主要围绕 vLLM 的 **KV Cache 分块显存管理、block 分配、请求与显存块映射关系** 展开，关注的是“模型执行时显存怎么管理”。

本节课的重点开始从 **显存管理层** 转向 **引擎通信与流式推理层**：

- 用户请求如何从客户端进入 vLLM；
- vLLM Engine 为什么像一个“中介进程”；
- Engine 如何通过 ZeroMQ 和后端推理进程通信；
- 在线推理为什么能做到“生成一个 token 返回一个 token”；
- 多客户端、多请求并发时，vLLM 如何区分不同请求的输出；
- `request_id`、输入队列、输出队列、流式读取之间是什么关系。

可以把本节课理解为：**vLLM 从“收到用户请求”到“把模型输出流式返回给用户”的通信主流程。**

---

# 任务一：总体阐述该视频的宗旨

## 1.1 这节课主要想讲什么？

这节课的核心目标是讲清楚 **vLLM Engine 模块与流式推理的源码执行链路**。

用户平时使用 vLLM 时，可能只看到类似下面的调用：

```python
llm.generate(...)
```

或者通过 OpenAI-compatible API 发送请求：

```text
client/chatbox/web 前端
    ↓ HTTP 请求
vLLM API Server
    ↓ 内部通信
vLLM Engine / Engine Process
    ↓ 调度、模型执行、KV Cache 管理
GPU ModelRunner
    ↓
输出 token
```

但是在源码内部，这个过程并不是“函数同步调用后一次性返回完整文本”这么简单。在线推理场景中，vLLM 要支持：

1. 多个客户端同时发请求；
2. 同一个客户端可以连续提交请求，而不是必须等上一条完全结束；
3. 模型每生成一小段结果，系统就能把这段结果及时返回；
4. 不同请求的输出不能混在一起；
5. API Server、Engine、后端推理进程之间要通过高效 IPC / socket 通信解耦。

因此，本节课重点解释：

- ZeroMQ 在 vLLM 内部通信中的作用；
- REQ/REP、DEALER/ROUTER、PUSH/PULL 三种通信模式的区别；
- Engine 如何接收请求；
- Engine 如何把请求放入输入队列；
- 后端推理引擎如何把结果写入输出队列；
- Engine 如何用 PUSH 把结果推回客户端；
- 客户端如何根据 `request_id` 找到属于自己的结果队列，并流式读取输出。

---

## 1.2 这节课的技术主线

本节课的主线可以概括为一句话：

> **用户请求通过 HTTP 到达 vLLM API Server，API Server 再通过 ZeroMQ 把请求发给 Engine；Engine 把请求放入输入队列交给后端推理进程；推理结果生成后写入输出队列；Engine 再通过 PUSH/PULL 通道把结果推回客户端；客户端根据 request_id 找到对应队列并逐步 yield 输出，从而形成流式返回。**

这一节课不是在讲 Transformer 的计算公式，也不是在讲 KV Cache 的显存布局，而是在讲 **推理服务系统中的“请求流转”和“输出流转”**。

---

# 任务二：按照课程推进顺序梳理知识点

由于转写文本没有严格时间戳，下面按照课程讲解的自然顺序整理。

---

## 2.1 开头：什么是 vLLM Engine？什么是流式执行？

### 核心知识点

课程开头先说明：本节课以 vLLM 为蓝本，讲解其中的 **引擎模块** 和 **流式执行**。

### 概念解释

#### 1. Engine / 引擎模块

在推理系统中，Engine 可以理解为一个“调度与通信中枢”。它不只是简单调用模型 forward，而是负责把用户请求接入系统，并协调后面的多个组件：

- 请求接收；
- 请求排队；
- 调度；
- 模型执行；
- KV Cache 管理；
- 量化模块；
- ModelRunner；
- 输出回传；
- 流式返回。

用户看到的是一个统一接口，但 Engine 内部连接了多个复杂模块。

#### 2. 流式执行

流式执行指的是：

> 模型不是等完整回答全部生成完再返回，而是每生成一小段 token，就可以把这段 token 返回给客户端。

例如 ChatGPT 或其他聊天产品中，回答会一个词、一个字、一个片段地“蹦出来”。这就是流式返回的用户体验。

### 在 AI Infra 中的作用

流式执行对大模型推理服务非常重要：

- **降低用户感知延迟**：即使完整回答需要 10 秒，只要第 1 秒能返回第一个 token，用户就会觉得系统响应快。
- **提高交互体验**：聊天机器人、代码助手、Agent 都需要边生成边展示。
- **便于中途取消**：用户可以在生成过程中停止请求，系统释放资源。
- **便于多请求调度**：推理引擎可以动态处理多个请求，而不是被一个长请求完全阻塞。

---

## 2.2 vLLM 整体通信链路：客户端、API Server、Engine、推理进程

### 核心知识点

课程接着解释：用户不是直接把请求发给 GPU ModelRunner，而是经过一条通信链路：

```text
ChatBox / Web / Client
    ↓ HTTP
vLLM API Server
    ↓ ZeroMQ
Engine Process
    ↓ 输入队列
Backend Inference Engine / ModelRunner
    ↓ 输出队列
Engine Process
    ↓ ZeroMQ
Client
```

### 概念解释

#### 1. HTTP 层

用户通过 ChatBox、网页、Python SDK 或 OpenAI-compatible API 发送请求，本质上是 HTTP 请求。请求中可能包含：

- prompt；
- model；
- max_tokens；
- temperature；
- top_p；
- stop；
- stream=True / False；
- API key；
- URL；
- 请求格式等。

HTTP 层面向外部用户，是服务接口。

#### 2. ZeroMQ 层

vLLM 内部组件之间不一定直接用 Python 函数调用，而会使用 ZeroMQ 进行进程间通信。

ZeroMQ 可以提供高效的 socket 通信模式，支持：

- 多进程之间传输消息；
- 一对一、一对多、多对一；
- 异步消息提交；
- 非阻塞 IO；
- 不同通信模式组合。

### 在 AI Infra 中的作用

大模型推理系统通常不是一个单线程单进程程序，而是多组件协作的服务系统。通信层的设计会直接影响：

- 并发处理能力；
- 请求排队能力；
- 流式返回能力；
- 多客户端隔离能力；
- 服务稳定性；
- 资源调度效率。

---

## 2.3 ZeroMQ 的 REQ/REP 模式：一问一答

### 核心知识点

课程先讲最基础的 ZeroMQ 通信模式：REQ/REP，也就是 Request/Response。

```text
Client -- request --> Server
Client <-- reply ---- Server
```

### 概念解释

REQ/REP 的特点：

1. 严格区分客户端和服务端；
2. 一问一答；
3. 客户端必须先发请求，再收响应；
4. 收到响应之后，才能发下一条请求；
5. 不能连续发两条请求而不接收响应。

课程中通过例子说明：如果客户端连续发送两条消息，而中间没有接收服务端回复，就会出问题。

### 为什么 REQ/REP 不适合 vLLM？

vLLM 在线推理场景中有很多并发请求：

```text
用户 1：发起聊天请求
用户 2：发起网页请求
用户 3：发起代码补全请求
...
```

如果使用 REQ/REP，一条请求必须等待完整响应后才能发送下一条，这会导致：

- 并发能力差；
- 请求提交阻塞；
- 不适合长文本生成；
- 不适合流式输出；
- 不适合高吞吐在线服务。

### 在 AI Infra 中的作用

REQ/REP 适合理解通信模式基础，但不是高并发 LLM Serving 的理想模式。面试中可以这样回答：

> REQ/REP 是同步一问一答模型，简单但阻塞；LLM Serving 需要异步提交、多请求并发、流式回传，因此更适合 DEALER/ROUTER 和 PUSH/PULL 这类模式组合。

---

## 2.4 ZeroMQ 的 DEALER/ROUTER 模式：异步、多客户端、可区分来源

### 核心知识点

课程重点讲了 DEALER/ROUTER，因为它更适合 vLLM 的请求接收场景。

```text
Client 1 / DEALER -- identity + payload --> ROUTER / Server
Client 2 / DEALER -- identity + payload --> ROUTER / Server
Client 3 / DEALER -- identity + payload --> ROUTER / Server
```

### 概念解释

#### 1. DEALER

DEALER 可以理解为一个更灵活的客户端 socket。它不要求“一发一收”严格交替，因此客户端可以连续发送多个请求。

#### 2. ROUTER

ROUTER 可以理解为一个能够识别客户端身份的服务端 socket。它接收消息时，通常会拿到多段数据：

```text
identity + payload
```

其中：

- `identity`：客户端身份标识；
- `payload`：真正的请求数据包。

#### 3. identity 的作用

假设有两个客户端：

```text
Client 1: ChatBox
Client 2: Web 页面
```

它们都向 Engine 发请求。Engine 必须知道：

- 这条请求是谁发来的；
- 后续输出应该返回给谁；
- 不同客户端的消息不能混淆。

`identity` 就是用来区分客户端来源的。

#### 4. payload 的内容

payload 不一定只是用户输入的文本，它可能是一个被序列化、压缩、封装后的数据包，其中包括：

- request_id；
- prompt token ids；
- sampling params；
- priority；
- stop strings；
- arrival time；
- 客户端类型；
- 其他元信息。

### DEALER/ROUTER 为什么适合 vLLM？

因为它支持：

- 客户端异步提交请求；
- 多客户端并发；
- 服务端根据 identity 区分请求来源；
- 不要求严格一问一答；
- 更适合作为 API Server 到 Engine 的请求通道。

### 在 AI Infra 中的作用

DEALER/ROUTER 解决的是 **请求入口侧的并发与路由问题**。

它回答的问题是：

> 多个客户端同时把请求打到 vLLM，Engine 怎么知道每条请求来自哪里？

---

## 2.5 Poller / epoll：为什么不能一直死循环监听？

### 核心知识点

课程接着讲 ZeroMQ socket 注册到 Poller 中，由内核或高效 IO 机制帮助监听事件。

### 概念解释

如果程序写成：

```python
while True:
    msg = socket.recv()
```

或者不断轮询某个 socket 是否有消息，就会浪费 CPU 时间片。

更好的方式是把 socket 注册到一个监听器，例如：

```python
poller.register(socket, zmq.POLLIN)
```

这样就相当于告诉系统：

> 这个 socket 如果有新消息可读，再通知我处理。

课程中用类比解释：

- 原来：你自己一直坐在门口等客人；
- 现在：你请了一个前台帮你盯门口，客人来了再通知你。

### epoll 的直观理解

在 Linux 网络编程中，epoll 是高性能 IO 多路复用机制。它可以帮助程序监听多个文件描述符 / socket，当某个 socket 可读或可写时再唤醒用户态程序。

在课程语境中，不需要把 epoll 内核细节讲得特别深，重点理解：

> Poller / epoll 的作用是避免用户态程序空转等待，提高 IO 监听效率。

### 在 AI Infra 中的作用

大模型推理服务往往需要同时处理：

- 多个用户请求；
- 多个输出通道；
- 多个 worker；
- 多个连接事件。

如果通信层一直忙等，会浪费 CPU，影响推理服务整体吞吐。因此 Poller / epoll 是高并发服务框架中的基础设施。

---

## 2.6 PUSH/PULL 模式：单向结果推送

### 核心知识点

课程讲完 DEALER/ROUTER 后，又讲了 PUSH/PULL 模式。

```text
PUSH 端  ---- message ---->  PULL 端
```

它和 DEALER/ROUTER 的区别是：

- DEALER/ROUTER 是双向通信；
- PUSH/PULL 是单向通信；
- PUSH 只负责推送；
- PULL 只负责拉取 / 接收。

### 概念解释

#### 1. PUSH

PUSH 端负责把消息推出去。

例如模型生成 token：

```text
Time 1: token1
Time 2: token2
Time 3: token3
```

每产生一个输出片段，就可以通过 PUSH 通道推回。

#### 2. PULL

PULL 端负责接收 PUSH 端推来的消息。

### 为什么输出侧适合 PUSH/PULL？

请求进入时，系统需要知道请求来自哪个客户端，所以适合 DEALER/ROUTER。

输出返回时，系统更像是“后端有结果了就往前推”，所以适合 PUSH/PULL。

可以理解为：

```text
请求入口：DEALER/ROUTER，强调多客户端身份识别
结果出口：PUSH/PULL，强调生成结果及时推送
```

### 在 AI Infra 中的作用

PUSH/PULL 支撑的是 **流式返回通道**。

它回答的问题是：

> 模型每生成一段输出，Engine 怎么及时把这段输出推回客户端？

---

## 2.7 Engine 接收请求：`process_input_socket`

### 核心知识点

课程进入 vLLM Engine 源码流程，讲 `process_input_socket` 的作用：

> 接收来自客户端 / API Server 的输入请求，解包后放入 Engine 的输入队列。

### 流程拆解

大致流程可以整理为：

```text
1. 创建 / 持有 ROUTER socket
2. 将 socket 注册到 Poller
3. 等待客户端请求
4. 收到 ZeroMQ data frames
5. 对请求数据解压缩 / 反序列化
6. 得到 Request 对象
7. 把 Request 放入 input_queue
```

### Request 中包含什么？

课程中提到，请求对象里可能包含：

- `request_id`：请求唯一标识；
- `sampling_params`：采样参数；
- `priority`：优先级；
- `stop`：停止条件；
- `arrival_time`：到达时间；
- `prompt_token_ids`：用户输入经过 tokenizer 后得到的 token id；
- 其他调度和生成相关信息。

其中最关键的是：

```text
prompt_token_ids
```

这是模型真正要处理的输入内容。

### `input_queue` 的作用

`input_queue` 是 Engine 与后端推理执行模块之间的缓冲区。

```text
process_input_socket
    ↓
input_queue
    ↓
backend inference engine
```

Engine 收到请求后，不是自己马上执行模型，而是先把请求放入输入队列，由后端推理组件读取队列并执行。

### 在 AI Infra 中的作用

这个流程体现了推理系统中的一个重要设计思想：

> 接收请求和执行模型解耦。

好处是：

- 网络接收不阻塞模型执行；
- 模型执行不阻塞新请求接入；
- 便于调度器统一管理请求；
- 便于支持 continuous batching；
- 便于多进程 / 多 worker 架构。

---

## 2.8 后端推理执行：Engine 只是中介，不直接等同于模型 forward

### 核心知识点

课程中明确说：Engine 模块更像一个“中介角色”。

Engine 负责通信和队列管理，而后端推理框架负责真正的模型推理。

### 完整关系

```text
Engine.input_queue
    ↓
后端推理引擎读取请求
    ↓
调度器选择本轮执行哪些请求
    ↓
ModelRunner 执行模型 forward
    ↓
生成 token / RequestOutput
    ↓
写入 Engine.output_queue
```

### Engine 不只是“模型调用函数”

初学者容易误解：

> Engine = 模型 forward。

实际上不是。

Engine 更准确的定位是：

```text
Engine = 请求管理 + 通信管理 + 队列桥接 + 输出回传
```

真正的模型执行通常由更底层的组件完成，例如：

- Scheduler；
- Executor；
- Worker；
- ModelRunner；
- Attention backend；
- KV Cache manager。

### 在 AI Infra 中的作用

这种拆分是服务化推理系统的常见架构：

- API Server 面向外部请求；
- Engine 负责内部请求生命周期；
- Scheduler 负责批处理和调度；
- ModelRunner 负责 GPU 前向计算；
- Cache Manager 负责 KV Cache；
- Output Processor 负责输出组装和返回。

---

## 2.9 Engine 返回结果：`process_output_socket`

### 核心知识点

课程接着讲结果如何从后端推理引擎返回给客户端。

相关函数是：

```text
process_output_socket / process_output_sockets
```

其作用可以概括为：

> 从输出队列中拿到后端推理结果，压缩 / 编码后，通过 PUSH socket 推回客户端。

### 流程拆解

```text
1. 后端推理引擎生成结果
2. 结果写入 Engine 的 output_queue
3. process_output_socket 从 output_queue 取结果
4. 根据客户端身份或请求信息区分输出目标
5. 对结果进行压缩 / 序列化
6. 通过 PUSH socket 推回客户端
```

### 为什么还要压缩？

课程中提到，通信过程中数据可能会压缩。原因是网络或进程间传输的数据成本较高，压缩可以减少传输量。

对于 LLM Serving 来说，输出对象可能包含：

- request_id；
- token id；
- token text；
- finish_reason；
- logprobs；
- usage；
- stop 信息；
- 当前是否完成。

如果并发量很大，通信开销不可忽视。

### 在 AI Infra 中的作用

`process_output_socket` 是流式推理的关键环节。

它保证：

- 后端生成结果能及时推出；
- 多个请求的输出能被正确回传；
- 输出不会阻塞请求接收；
- 客户端可以边生成边显示。

---

## 2.10 客户端侧：`add_request` 做了什么？

### 核心知识点

课程后半部分开始从客户端角度看流程。核心函数是：

```text
add_request
```

课程中强调：`add_request` 主要做两件事。

### `add_request` 的两个核心动作

#### 动作一：发送请求到 Engine

客户端通过 DEALER/ROUTER 通道，把请求发给 Engine。

```text
client.add_request(request)
    ↓
ZeroMQ DEALER socket
    ↓
Engine ROUTER socket
```

#### 动作二：创建该请求专属的结果队列

每个请求都有唯一的 `request_id`，客户端会为它创建一个专属的等待队列。

```text
request_id = xxx

request_id 对应的 output_queue:
    queue_for_request_xxx
```

后续返回结果时，系统会根据 `request_id` 把输出放入这个队列。

### 为什么要为每个请求创建独立队列？

因为在线推理时，不同请求的输出可能交错返回：

```text
request A 生成 token A1
request B 生成 token B1
request A 生成 token A2
request C 生成 token C1
request B 生成 token B2
...
```

如果只有一个总队列，客户端很难知道哪个 token 属于哪个请求。

因此必须根据 `request_id` 做分发：

```text
output.request_id == A  →  queue_A
output.request_id == B  →  queue_B
output.request_id == C  →  queue_C
```

### 在 AI Infra 中的作用

这是流式多请求系统的核心机制：

> 所有输出可以混合进入一个总通道，但最终必须按 request_id 分发到对应请求的结果队列。

这就是“多路复用输入，多路分发输出”。

---

## 2.11 客户端接收结果：`process_outputs` / output handler

### 核心知识点

课程最后解释了客户端收到后端推回结果之后，如何把结果放回对应请求队列。

### 流程拆解

```text
1. 客户端从 PULL socket 收到一批输出
2. 输出中包含 request_id 和 output
3. 遍历所有输出
4. 根据 request_id 找到对应请求的队列
5. 把 output 放入该队列
```

伪代码可以理解为：

```python
for request_output in outputs:
    request_id = request_output.request_id
    queue = request_id_to_queue[request_id]
    queue.put(request_output)
```

### `generate` 如何实现流式读取？

客户端的 `generate` 或异步生成接口，会等待这个请求对应的队列：

```python
async for output in llm.generate(...):
    yield output
```

底层逻辑可以理解为：

```text
queue_A 中一旦有新 output
    ↓
generate 立即读出来
    ↓
返回给上层用户
    ↓
前端显示一个 token / 一段文本
```

这样就形成了用户看到的流式返回。

### 在 AI Infra 中的作用

这部分体现了流式推理的本质：

- 后端不是一次性返回完整文本；
- 每个请求有自己的结果队列；
- 结果按 request_id 分发；
- 上层接口通过异步迭代持续读取队列；
- 每读取到一个输出片段，就返回给用户。

---

# 任务三：串联整节课的完整流程

下面用“从输入到输出”的方式串起本节课的完整技术主线。

---

## 3.1 从用户输入到模型输出的完整链路

### 第 1 步：用户在客户端输入 prompt

例如用户在 ChatBox 或网页中输入：

```text
用 20 个字介绍 vLLM
```

客户端通过 HTTP 请求把这个 prompt 发给 vLLM API Server。

---

### 第 2 步：API Server 把请求封装成内部 Request

API Server 会将请求转成内部结构，里面包含：

```text
request_id
prompt_token_ids
sampling_params
priority
arrival_time
stop
stream
其他元信息
```

其中：

- `request_id` 用来唯一标识请求；
- `prompt_token_ids` 是用户输入经过 tokenizer 后的 token id；
- `sampling_params` 控制生成策略；
- `stream=True` 表示需要流式返回。

---

### 第 3 步：客户端 / API Server 通过 DEALER 把请求发给 Engine

```text
API Server / Client
    ↓ DEALER socket
Engine
    ↑ ROUTER socket
```

DEALER/ROUTER 的作用是：

- 允许客户端异步发送请求；
- Engine 能识别不同客户端 identity；
- 不要求严格一问一答；
- 支持高并发请求入口。

---

### 第 4 步：Engine 的 `process_input_socket` 接收请求

Engine 通过 ROUTER socket 收到消息：

```text
identity + payload
```

然后执行：

```text
接收 data frames
    ↓
解压缩
    ↓
反序列化
    ↓
得到 Request 对象
    ↓
放入 input_queue
```

---

### 第 5 步：后端推理引擎从 `input_queue` 取请求

后端推理模块读取请求，进入模型推理流程：

```text
input_queue
    ↓
scheduler
    ↓
batch 构建
    ↓
KV Cache 准备
    ↓
ModelRunner forward
    ↓
生成 token
```

这里会涉及前面几节课讲过的内容：

- block 分配；
- KV Cache；
- `query_start_loc`；
- `block_table`；
- `slot_mapping`；
- prefill / decode；
- continuous batching。

本节课不展开模型 forward 细节，只把它看作“后端推理已经完成了一次输出”。

---

### 第 6 步：后端推理引擎把结果写入 `output_queue`

生成结果可能是：

```text
RequestOutput(
    request_id="xxx",
    text="vLLM 是...",
    token_ids=[...],
    finished=False
)
```

或者某个 token 片段。

结果被写入 Engine 的 `output_queue`。

---

### 第 7 步：Engine 的 `process_output_socket` 取出结果并 PUSH 回客户端

Engine 读取输出队列：

```text
output_queue
    ↓
process_output_socket
    ↓
压缩 / 序列化
    ↓
PUSH socket
```

然后把结果推回客户端侧的 PULL socket。

---

### 第 8 步：客户端接收输出并按 `request_id` 分发

客户端收到一批输出后，不会直接全部给用户，而是先根据 `request_id` 分发：

```text
output.request_id == request_A  → queue_A
output.request_id == request_B  → queue_B
output.request_id == request_C  → queue_C
```

每个请求都有自己的结果队列。

---

### 第 9 步：`generate` 从该请求队列中流式读取

对于某个请求 A：

```text
queue_A 有新结果
    ↓
generate 读取
    ↓
yield 给上层
    ↓
前端显示
```

只要模型继续生成，队列就继续有新输出，客户端就继续显示。

---

### 第 10 步：生成完成后结束流式返回

当某个输出标记 `finished=True` 或生成 EOS / stop string / 达到 max_tokens 后：

```text
该请求结束
    ↓
客户端停止等待
    ↓
释放请求状态
    ↓
后端释放相关资源
```

---

## 3.2 用一张 ASCII 图理解整节课

```text
┌──────────────────────────┐
│ ChatBox / Web / Python   │
│ 用户输入 prompt           │
└─────────────┬────────────┘
              │ HTTP
              ▼
┌──────────────────────────┐
│ vLLM API Server           │
│ 生成 request_id            │
│ tokenize prompt            │
│ 封装 Request               │
└─────────────┬────────────┘
              │ ZeroMQ DEALER
              ▼
┌──────────────────────────┐
│ Engine Process             │
│ ROUTER 接收请求             │
│ process_input_socket       │
│ 解压 / 反序列化             │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│ input_queue                │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│ Backend Inference Engine   │
│ Scheduler                  │
│ ModelRunner                │
│ KV Cache Manager           │
│ GPU forward                │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│ output_queue               │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│ process_output_socket      │
│ PUSH 输出结果               │
└─────────────┬────────────┘
              │ ZeroMQ PUSH/PULL
              ▼
┌──────────────────────────┐
│ Client output handler      │
│ 根据 request_id 分发         │
│ queue_A / queue_B / ...    │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│ generate / async generator │
│ 逐段 yield 输出             │
└──────────────────────────┘
```

---

# 4. 本节课关键概念总表

| 概念 | 作用 | 初学者理解 |
|---|---|---|
| Engine | vLLM 内部请求管理和通信中枢 | 像“调度前台 + 中转站” |
| 流式执行 | 生成一段返回一段 | 用户看到文字一个个蹦出来 |
| ZeroMQ | 内部进程通信工具 | API Server 和 Engine 之间的通信管道 |
| REQ/REP | 一问一答通信 | 简单但阻塞，不适合高并发 LLM |
| DEALER/ROUTER | 异步、多客户端通信 | 适合请求进入 Engine |
| identity | 客户端身份标识 | Engine 知道请求来自谁 |
| payload | 请求数据包 | 包含 request_id、prompt token、采样参数等 |
| Poller / epoll | 高效监听 socket | 避免死循环浪费 CPU |
| PUSH/PULL | 单向推送通信 | 适合把生成结果推回客户端 |
| process_input_socket | 接收请求 | 把客户端请求放入 input_queue |
| input_queue | 输入队列 | Engine 与推理后端之间的缓冲区 |
| output_queue | 输出队列 | 推理后端把结果写回 Engine |
| process_output_socket | 返回结果 | 把后端输出推回客户端 |
| add_request | 发送请求 + 创建结果队列 | 每个请求都有自己的等待队列 |
| request_id | 请求唯一标识 | 防止多请求输出混淆 |
| generate | 流式读取结果 | 队列有输出就 yield 给用户 |

---

# 5. 结合源码时应该重点看哪些函数？

根据课程内容，建议按下面顺序看源码：

## 5.1 请求入口侧

重点看：

```text
add_request
process_input_socket
recv_multipart
decode / decompress request
input_queue.put(...)
```

要带着问题看：

- 请求在哪里被创建？
- `request_id` 在哪里生成？
- 请求如何通过 ZeroMQ 发给 Engine？
- Engine 如何接收？
- 请求对象里有哪些字段？
- 请求最终放到了哪个队列？

---

## 5.2 输出返回侧

重点看：

```text
process_output_socket
output_queue.get(...)
socket.send(...)
process_outputs
request_state.output_queue.put(...)
generate
```

要带着问题看：

- 后端推理结果写到哪里？
- Engine 从哪里拿输出？
- 输出如何通过 PUSH/PULL 返回？
- 客户端如何按 `request_id` 分发？
- `generate` 为什么能流式 yield？

---

## 5.3 调试建议

课程中多次强调通过调试理解源码。建议调试时关注：

1. 在 Engine 进程打印 PID；
2. 用 VSCode attach 到对应进程；
3. 在 `process_input_socket` 打断点；
4. 发送一个简单请求；
5. 观察 Request 对象字段；
6. 在输出处理函数打断点；
7. 观察 output 中的 `request_id`；
8. 观察结果如何被 put 到对应 queue。

不要只看静态代码。vLLM 这种推理框架的源码必须结合运行时数据结构理解。

---

# 6. 和前几节课内容的串联

前几节课讲的是：

```text
KV Cache 分块
    ↓
block_size
    ↓
num_blocks
    ↓
block_table
    ↓
slot_mapping
    ↓
PagedAttention
```

本节课讲的是：

```text
用户请求
    ↓
Engine 接收
    ↓
输入队列
    ↓
后端推理
    ↓
输出队列
    ↓
流式返回
```

两者合起来就是完整 vLLM 推理流程：

```text
用户请求进入系统
    ↓
Engine 接收并排队
    ↓
Scheduler 选择本轮执行请求
    ↓
为请求分配 KV Cache block
    ↓
准备 query_start_loc / block_table / slot_mapping
    ↓
ModelRunner 执行 prefill / decode
    ↓
PagedAttention 读取/写入 KV Cache
    ↓
生成新 token
    ↓
输出写回 Engine
    ↓
Engine 流式推回客户端
```

因此，本节课是连接 **服务层** 和 **模型执行层** 的关键一环。

---

# 7. 面试问题与参考答案

## 问题 1：vLLM Engine 在推理系统中负责什么？

**参考答案：**

vLLM Engine 不是单纯的模型 forward 函数，而是请求生命周期的管理中枢。它负责接收用户请求、维护输入队列、与后端推理进程通信、接收模型输出、按请求 ID 分发结果，并支持流式返回。真正的模型执行通常由 Scheduler、Executor、Worker、ModelRunner 等模块完成，Engine 更像服务层和执行层之间的桥梁。

---

## 问题 2：什么是流式推理？为什么大模型服务需要流式推理？

**参考答案：**

流式推理指模型每生成一段 token，就立即返回给客户端，而不是等完整回答全部生成完再一次性返回。它可以降低用户感知延迟，提高聊天、代码补全等交互场景的体验，也方便用户中途取消请求并释放资源。对于在线 LLM Serving，流式推理几乎是标准能力。

---

## 问题 3：REQ/REP 和 DEALER/ROUTER 的区别是什么？

**参考答案：**

REQ/REP 是严格一问一答模式，客户端必须发送请求、等待响应，然后才能发送下一条请求；它简单但阻塞，不适合高并发流式推理。DEALER/ROUTER 更灵活，客户端可以异步发送请求，ROUTER 可以通过 identity 区分不同客户端，因此更适合 vLLM 这类多客户端、多请求并发的推理系统。

---

## 问题 4：为什么 vLLM 内部通信要用 ZeroMQ？

**参考答案：**

vLLM 是一个多组件、多进程协作的推理系统，API Server、Engine、后端推理进程之间需要高效通信。ZeroMQ 提供了多种通信模式，例如 DEALER/ROUTER 支持异步多客户端请求，PUSH/PULL 支持单向结果推送，Poller 支持高效 IO 监听。这些能力适合构建高并发、低阻塞的推理服务通信层。

---

## 问题 5：DEALER/ROUTER 中的 identity 有什么作用？

**参考答案：**

identity 用来标识客户端来源。多个客户端同时向 Engine 发送请求时，ROUTER 接收到的消息会包含客户端 identity 和 payload。Engine 可以根据 identity 区分请求来自哪个客户端，并在后续结果返回时保证输出不会发错对象。

---

## 问题 6：PUSH/PULL 在流式推理中负责什么？

**参考答案：**

PUSH/PULL 主要负责输出结果的单向推送。后端推理引擎生成 token 或 RequestOutput 后，Engine 可以通过 PUSH socket 把结果推回客户端，客户端通过 PULL socket 接收。这种模式适合“有结果就推”的流式输出场景。

---

## 问题 7：`process_input_socket` 的作用是什么？

**参考答案：**

`process_input_socket` 负责从 ZeroMQ socket 中接收客户端发来的请求数据，通常包括接收 multipart frames、解压缩、反序列化，得到内部 Request 对象，然后将请求放入 Engine 的 input_queue。后端推理引擎会从 input_queue 中读取请求并执行推理。

---

## 问题 8：`process_output_socket` 的作用是什么？

**参考答案：**

`process_output_socket` 负责从 Engine 的 output_queue 中读取后端推理结果，将结果序列化或压缩后，通过 PUSH socket 推回客户端。它是模型输出从后端返回到客户端的关键通道，也是实现流式返回的重要环节。

---

## 问题 9：为什么每个请求需要独立的结果队列？

**参考答案：**

在线推理中，不同请求的输出可能交错产生。例如请求 A 生成一个 token，请求 B 又生成一个 token，然后请求 A 继续生成。如果所有输出只放在一个队列里，客户端很难区分输出属于哪个请求。因此系统会根据 request_id 为每个请求维护独立结果队列，输出回来后按 request_id 分发，保证每个请求读取到自己的结果。

---

## 问题 10：`request_id` 在流式推理中的作用是什么？

**参考答案：**

`request_id` 是请求的唯一标识。它贯穿请求生命周期：请求发送时携带 request_id，Engine 接收后保留它，后端推理输出 RequestOutput 时也带上它，客户端收到输出后根据 request_id 找到对应的结果队列。没有 request_id，多请求并发时输出就会混乱。

---

## 问题 11：为什么 Poller / epoll 对推理服务有帮助？

**参考答案：**

如果程序一直用死循环监听 socket，会浪费 CPU 时间片。Poller / epoll 可以把 socket 注册到高效 IO 监听机制中，只有当 socket 有新消息可读时才唤醒程序处理。这样可以减少空转，提高高并发服务中的 CPU 利用效率。

---

## 问题 12：vLLM 中 HTTP、ZeroMQ、Engine、ModelRunner 分别属于哪一层？

**参考答案：**

HTTP 属于外部服务接口层，用于接收用户请求；ZeroMQ 属于内部通信层，用于 API Server、Engine 和后端推理进程之间通信；Engine 属于请求管理和中转层，负责队列和输出分发；ModelRunner 属于模型执行层，负责真正的 GPU forward 和 KV Cache 使用。

---

## 问题 13：从用户输入到流式输出，完整流程是什么？

**参考答案：**

用户通过 HTTP 发出 prompt；API Server 将 prompt token 化并封装成 Request；客户端通过 ZeroMQ DEALER 把请求发给 Engine 的 ROUTER；Engine 的 process_input_socket 接收并解包请求，放入 input_queue；后端推理引擎读取请求并执行模型生成；生成结果写入 output_queue；Engine 的 process_output_socket 通过 PUSH 把结果推回客户端；客户端 PULL 到结果后，根据 request_id 分发到对应请求队列；generate 从队列中不断读取并 yield 给用户，形成流式输出。

---

# 8. 初学者最容易混淆的点

## 8.1 Engine 不等于 ModelRunner

Engine 是服务和通信中枢，ModelRunner 是模型执行组件。

```text
Engine 管请求
ModelRunner 跑模型
```

---

## 8.2 request_id 和 identity 不完全一样

| 名称 | 用途 |
|---|---|
| identity | 标识客户端 / socket 来源 |
| request_id | 标识某一次具体请求 |

一个客户端可以发多个请求，因此一个 identity 下可能有多个 request_id。

---

## 8.3 input_queue 和 output_queue 不是同一个东西

```text
input_queue：放待推理请求
output_queue：放推理完成后的结果
```

---

## 8.4 PUSH/PULL 不负责请求入口

请求入口更需要区分客户端身份，所以适合 DEALER/ROUTER。

PUSH/PULL 更适合输出侧，因为输出侧主要是有结果就推回。

---

## 8.5 流式返回不是模型内部“每层返回一次”

流式返回一般是指 **每生成一个 token 或一段文本后返回一次**，不是 Transformer 每一层计算完都返回。

---

# 9. 复习时要抓住的一句话

> vLLM 的流式推理，本质上是：**请求通过 DEALER/ROUTER 异步进入 Engine，Engine 放入 input_queue 交给后端推理；后端生成结果后写入 output_queue，Engine 再通过 PUSH/PULL 推回客户端；客户端根据 request_id 把结果放入对应队列，generate 持续读取该队列并 yield 给用户。**

---

# 10. 建议你后续看源码时的学习顺序

1. 先看最外层 API 请求是怎么变成 Request 的；
2. 再看 `add_request` 如何发送请求；
3. 看 ZeroMQ DEALER/ROUTER 如何连接 Engine；
4. 看 `process_input_socket` 如何接收并放入 `input_queue`；
5. 暂时跳过复杂推理执行，把后端推理当成黑盒；
6. 看后端输出如何进入 `output_queue`；
7. 看 `process_output_socket` 如何 PUSH 回客户端；
8. 看客户端如何根据 `request_id` 分发结果；
9. 看 `generate` 为什么能实现流式 yield；
10. 最后再把这条链路和 KV Cache / Scheduler / ModelRunner 串起来。

---

# 11. 本节课最终知识框架

```text
vLLM 在线推理服务
│
├── 外部请求层
│   ├── ChatBox
│   ├── Web
│   └── OpenAI-compatible API
│
├── 内部通信层
│   ├── ZeroMQ
│   ├── DEALER/ROUTER：请求入口
│   ├── PUSH/PULL：输出返回
│   └── Poller/epoll：高效监听
│
├── Engine 层
│   ├── process_input_socket
│   ├── input_queue
│   ├── output_queue
│   └── process_output_socket
│
├── 后端推理层
│   ├── Scheduler
│   ├── ModelRunner
│   ├── KV Cache Manager
│   └── Attention Backend
│
└── 流式输出层
    ├── request_id
    ├── 每请求独立 output queue
    ├── process_outputs
    └── generate / yield
```

---

# 12. 结论

本节课讲的是 vLLM 的 **服务层与通信层主流程**。它解释了为什么 vLLM 不是简单地“输入 prompt、调用模型、返回字符串”，而是一个面向高并发在线服务的复杂推理系统。

学完这节课，你需要形成三个核心认识：

1. **Engine 是请求生命周期管理中心**：负责接收、排队、转发、回传，而不是单纯 forward。
2. **ZeroMQ 是内部组件通信基础**：DEALER/ROUTER 解决请求入口，PUSH/PULL 解决输出推送，Poller 解决高效监听。
3. **流式返回依赖 request_id 和结果队列**：多请求输出会交错产生，必须按 request_id 分发到对应队列，客户端才能持续读取并返回给用户。

这节课把“推理引擎源码”从模型计算扩展到了服务系统设计，是学习 AI Infra / LLM Serving 时必须掌握的一层。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-vllm|模块-vllm]]

%% 项目关联导航：结束 %%
