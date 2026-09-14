# scheduler.py 源码逐行精读解析

> 解析对象：`scheduler.py`  
> 所属项目：nano-vLLM  
> 核心主题：请求调度、Prefill / Decode 分离、KV Cache block 分配、抢占与后处理

---

## 1. 文件整体定位

### 1.1 这个文件负责什么

`scheduler.py` 定义了 nano-vLLM 的 `Scheduler` 类。它是推理引擎中的**请求调度层**，负责决定每一轮推理应该处理哪些 `Sequence`，以及本轮属于 **Prefill** 还是 **Decode**。

在 `LLMEngine.generate()` 的主循环中，`LLMEngine.step()` 会反复调用：

```python
seqs, is_prefill = self.scheduler.schedule()
token_ids = self.model_runner.call("run", seqs, is_prefill)
self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

因此，`Scheduler` 的职责可以概括为：

```text
1. 接收新请求 Sequence
2. 把请求放入 waiting 队列
3. 根据 max_num_seqs / max_num_batched_tokens / KV Cache 空间选择本轮要执行的请求
4. 区分本轮是 prefill 还是 decode
5. 在显存 block 不足时执行 preemption 抢占
6. 模型执行后更新 Sequence 状态
7. 请求完成后释放 KV Cache block
```

它不直接执行模型，也不直接做 attention 计算，而是负责“谁先跑、跑多少、跑完怎么更新状态”。

---

### 1.2 它属于哪一层

`scheduler.py` 属于：

```text
推理引擎调度层 / Request Scheduling 层
```

在 nano-vLLM 中可以这样分层：

```text
用户 API 层：
    example.py / bench.py / llm.py

推理引擎总控层：
    llm_engine.py

请求与调度层：
    sequence.py
    scheduler.py   ← 当前文件

KV Cache 管理层：
    block_manager.py

模型执行层：
    model_runner.py

模型结构层：
    qwen3.py
    layers/attention.py
    layers/linear.py
    layers/sampler.py
```

---

### 1.3 它和哪些文件有关

`scheduler.py` 和以下文件关系最密切：

| 文件 | 关系 |
|---|---|
| `llm_engine.py` | 调用 `Scheduler.add()`、`Scheduler.schedule()`、`Scheduler.postprocess()`、`Scheduler.is_finished()` |
| `sequence.py` | `Scheduler` 调度的基本对象就是 `Sequence` |
| `block_manager.py` | `Scheduler` 通过 `BlockManager` 分配、追加、释放 KV Cache block |
| `model_runner.py` | `Scheduler.schedule()` 返回的 `seqs` 和 `is_prefill` 会传给 `ModelRunner.run()` |
| `config.py` | `Scheduler.__init__()` 从 `Config` 读取最大并发数、最大 batch token 数、EOS、block size 等参数 |

---

### 1.4 它在完整推理流程中的位置

完整流程如下：

```text
用户输入 prompts
    ↓
LLMEngine.generate()
    ↓
LLMEngine.add_request()
    ↓
每条 prompt 被包装成 Sequence
    ↓
Scheduler.add(seq)
    ↓
Sequence 进入 waiting 队列
    ↓
LLMEngine.step()
    ↓
Scheduler.schedule()
    ↓
选出本轮 seqs，判断 is_prefill
    ↓
ModelRunner.run(seqs, is_prefill)
    ↓
模型 forward + sampler 产生 token_ids
    ↓
Scheduler.postprocess(seqs, token_ids, is_prefill)
    ↓
更新 Sequence 状态，释放或维护 KV Cache
    ↓
所有请求 finished 后 generate() 返回结果
```

所以 `Scheduler` 位于：

```text
Sequence 请求对象之后，ModelRunner 模型执行之前。
```

它是连接“请求管理”和“模型执行”的核心模块。

---

## 2. 代码结构总览

### 2.1 导入模块

```python
from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
```

导入内容分为两类：

| 导入项 | 来源 | 作用 |
|---|---|---|
| `deque` | Python 标准库 | 构造 waiting / running 双端队列 |
| `Config` | `config.py` | 提供调度参数 |
| `Sequence` | `sequence.py` | 表示一条推理请求 |
| `SequenceStatus` | `sequence.py` | 表示请求状态：WAITING / RUNNING / FINISHED |
| `BlockManager` | `block_manager.py` | 管理 KV Cache block |

---

### 2.2 定义了哪些类

本文件只定义了一个类：

```python
class Scheduler:
```

它负责维护两个请求队列：

```python
self.waiting: deque[Sequence] = deque()
self.running: deque[Sequence] = deque()
```

并负责调用 `BlockManager` 管理 KV Cache 资源。

---

### 2.3 定义了哪些函数 / 方法

`Scheduler` 中定义了 5 个核心方法：

| 方法 | 作用 |
|---|---|
| `__init__(self, config)` | 初始化调度器、读取配置、创建 BlockManager、创建 waiting/running 队列 |
| `is_finished(self)` | 判断所有请求是否都完成 |
| `add(self, seq)` | 添加新请求到 waiting 队列 |
| `schedule(self)` | 核心调度函数，决定本轮执行 prefill 还是 decode |
| `preempt(self, seq)` | 抢占某个 running 请求，释放它的 KV Cache，并放回 waiting 队列 |
| `postprocess(self, seqs, token_ids, is_prefill)` | 模型执行后更新 Sequence 状态，追加 token，处理 EOS 和释放 block |

---

### 2.4 重要变量和数据结构

| 变量 | 类型 | 含义 |
|---|---|---|
| `self.max_num_seqs` | `int` | 一轮最多调度多少条 Sequence |
| `self.max_num_batched_tokens` | `int` | Prefill 阶段一轮最多处理多少 prompt token |
| `self.eos` | `int` | 模型结束 token id |
| `self.block_size` | `int` | 一个 KV Cache block 能容纳多少 token |
| `self.block_manager` | `BlockManager` | KV Cache block 管理器 |
| `self.waiting` | `deque[Sequence]` | 等待 prefill 或被抢占后等待重新 prefill 的请求队列 |
| `self.running` | `deque[Sequence]` | 已完成 prefill、正在 decode 的请求队列 |
| `scheduled_seqs` | `list[Sequence]` | 本轮被选中执行的请求列表 |
| `is_prefill` | `bool` | 本轮是否为 Prefill 阶段 |
| `num_batched_tokens` | `int` | Prefill 本轮已经累计调度的 token 数 |
| `num_scheduled_tokens` | `int` | 某条 Sequence 本轮计划处理的 token 数 |

---

### 2.5 主流程和辅助逻辑

主流程集中在：

```python
def schedule(self) -> tuple[list[Sequence], bool]:
```

以及：

```python
def postprocess(self, seqs, token_ids, is_prefill):
```

辅助逻辑是：

```python
def is_finished(self):
def add(self, seq):
def preempt(self, seq):
```

---

### 2.6 文件组织结构图

```text
scheduler.py
│
├── 导入依赖
│   ├── deque
│   ├── Config
│   ├── Sequence / SequenceStatus
│   └── BlockManager
│
└── class Scheduler
    │
    ├── __init__()
    │   ├── 读取调度参数
    │   ├── 创建 BlockManager
    │   ├── 创建 waiting 队列
    │   └── 创建 running 队列
    │
    ├── is_finished()
    │   └── 判断 waiting 和 running 是否都为空
    │
    ├── add()
    │   └── 新请求进入 waiting
    │
    ├── schedule()
    │   ├── Prefill 调度逻辑
    │   ├── 如果有 prefill 请求，直接返回 prefill batch
    │   └── Decode 调度逻辑
    │       ├── 检查能否 append 新 token
    │       ├── 必要时 preempt 其他请求
    │       └── 返回 decode batch
    │
    ├── preempt()
    │   ├── 状态改回 WAITING
    │   ├── 释放 KV Cache block
    │   └── 放回 waiting 队头
    │
    └── postprocess()
        ├── hash 已完成 block
        ├── 更新 num_cached_tokens
        ├── 处理 chunked prefill
        ├── append 新 token
        ├── 检查 EOS / max_tokens
        └── 完成后释放 block
```

---

## 3. 逐行代码解释

### 3.1 导入 `deque`

```python
from collections import deque
```

#### 语法作用

从 Python 标准库 `collections` 中导入 `deque`。

`deque` 是 double-ended queue，即双端队列，可以高效地从队头或队尾插入、删除元素。

#### 工程作用

调度器需要频繁执行：

```text
队头取请求
队尾放请求
队头重新插入被抢占请求
队尾弹出低优先级 running 请求
```

这些操作用普通 list 也能做，但队头弹出 `list.pop(0)` 是 O(n)，而 `deque.popleft()` 是 O(1)。

#### 在推理流程中的意义

waiting / running 队列是调度器的核心数据结构。它们决定：

```text
哪些请求还没完成 prefill
哪些请求正在 decode
哪些请求应该被优先调度
```

---

### 3.2 导入 Config、Sequence、BlockManager

```python
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
```

#### 语法作用

从 nano-vLLM 的其他模块中导入调度器依赖的类。

#### 工程作用

| 类 | 在本文件中的作用 |
|---|---|
| `Config` | 提供调度器初始化参数 |
| `Sequence` | 调度器管理的请求对象 |
| `SequenceStatus` | 修改请求状态 |
| `BlockManager` | 分配、追加、释放 KV Cache block |

#### 在推理流程中的意义

这三类对象构成调度层的核心：

```text
Config 决定调度上限
Sequence 表示被调度的请求
BlockManager 决定是否有足够 KV Cache 显存资源
```

下一步应该结合看：

```text
config.py
sequence.py
block_manager.py
```

---

### 3.3 定义 Scheduler 类

```python
class Scheduler:
```

#### 语法作用

定义一个名为 `Scheduler` 的类。

#### 工程作用

`Scheduler` 是 nano-vLLM 中负责请求调度的组件。

它不直接生成 token，而是回答几个问题：

```text
本轮是否有 waiting 请求需要 prefill？
本轮能处理多少 prompt token？
哪些 running 请求可以 decode？
KV Cache block 是否够用？
不够用时抢占谁？
某条请求生成结束后如何释放资源？
```

#### 在推理流程中的意义

`LLMEngine.step()` 每次推进生成都会调用 `Scheduler.schedule()`，所以它是每轮推理的“入口裁判”。

---

### 3.4 初始化方法

```python
def __init__(self, config: Config):
```

#### 语法作用

定义类的初始化方法。`config: Config` 是类型注解，表示传入参数应是 `Config` 对象。

#### 工程作用

初始化调度器的配置参数、KV Cache block 管理器、waiting/running 队列。

#### 在推理流程中的意义

`LLMEngine.__init__()` 创建 `Scheduler(config)`，之后所有请求都会通过这个调度器管理。

---

### 3.5 保存最大并发请求数

```python
self.max_num_seqs = config.max_num_seqs
```

#### 语法作用

把配置对象中的 `max_num_seqs` 保存为调度器实例属性。

#### 工程作用

限制一轮最多调度多少条 Sequence。

例如：

```text
max_num_seqs = 512
```

表示一次模型执行中最多包含 512 条请求。

#### 在推理流程中的意义

这个参数控制 batch 中 sequence 数量，主要影响 Decode 阶段吞吐。

Decode 阶段通常每条请求只生成 1 个 token，所以：

```text
一轮 decode token 数 ≈ len(scheduled_seqs)
```

---

### 3.6 保存最大 batched token 数

```python
self.max_num_batched_tokens = config.max_num_batched_tokens
```

#### 语法作用

保存配置中的 `max_num_batched_tokens`。

#### 工程作用

限制 Prefill 阶段一轮最多处理多少 token。

例如：

```text
max_num_batched_tokens = 16384
```

表示本轮 prefill 的 prompt token 总数不能超过 16384。

#### 在推理流程中的意义

Prefill 阶段一次处理的是 prompt tokens，长度可能很长。如果不限制 token 总量，显存和计算量会爆炸。

它主要影响：

```text
Prefill 吞吐
TTFT
单轮显存占用
长 prompt 调度行为
```

---

### 3.7 保存 EOS token id

```python
self.eos = config.eos
```

#### 语法作用

保存模型的结束 token id。

#### 工程作用

后处理时用于判断请求是否结束：

```python
if token_id == self.eos:
    seq.status = SequenceStatus.FINISHED
```

#### 在推理流程中的意义

EOS 是自回归生成的自然停止条件。模型生成 EOS 后，该请求应释放 KV Cache，不再参与 decode。

---

### 3.8 保存 block size

```python
self.block_size = config.kvcache_block_size
```

#### 语法作用

保存 KV Cache block 的大小。

#### 工程作用

`block_size` 表示一个 KV Cache block 可以容纳多少 token。

例如：

```text
block_size = 256
```

那么一个长度为 600 的 sequence 至少需要：

```text
ceil(600 / 256) = 3 个 block
```

#### 在推理流程中的意义

调度器会用 `block_size` 计算 prefix cache 命中的 block 能覆盖多少 token：

```python
num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
```

这和 PagedAttention / KV Cache block 管理直接相关。

---

### 3.9 创建 BlockManager

```python
self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
```

#### 语法作用

创建一个 `BlockManager` 对象。

#### 工程作用

`BlockManager` 负责管理所有可用的 KV Cache block，包括：

```text
can_allocate(seq)：判断能否为新请求分配 block
allocate(seq, num_cached_blocks)：为请求分配 block
can_append(seq)：decode 新 token 前判断是否需要/能否追加 block
may_append(seq)：必要时为 decode 追加新 block
hash_blocks(seq)：把已完成 block 加入 prefix cache 哈希表
deallocate(seq)：请求结束或被抢占时释放 block
```

#### 在推理流程中的意义

Scheduler 本身不直接维护底层 KV Cache 张量，但它负责决定什么时候调用 BlockManager。

也就是说：

```text
Scheduler 决定调度策略
BlockManager 决定 KV Cache 资源是否满足调度要求
```

---

### 3.10 创建 waiting 队列

```python
self.waiting: deque[Sequence] = deque()
```

#### 语法作用

创建一个空的双端队列，并用类型注解标明队列元素是 `Sequence`。

#### 工程作用

`waiting` 保存等待 Prefill 的请求。

进入 waiting 的情况有两种：

```text
1. 新请求刚被 add(seq) 加入
2. running 请求因为 KV Cache 不足被 preempt 后重新放回 waiting
```

#### 在推理流程中的意义

waiting 队列中的 Sequence 通常还不能直接 decode，因为它们的 prompt KV Cache 还没有完整建立。

---

### 3.11 创建 running 队列

```python
self.running: deque[Sequence] = deque()
```

#### 语法作用

创建另一个空的 `deque[Sequence]`。

#### 工程作用

`running` 保存已经完成 Prefill、正在 Decode 的请求。

当某条 waiting 请求的 prompt token 全部处理完后，会执行：

```python
seq.status = SequenceStatus.RUNNING
self.waiting.popleft()
self.running.append(seq)
```

#### 在推理流程中的意义

running 队列是 decode 阶段的主要调度对象。

---

### 3.12 判断是否完成

```python
def is_finished(self):
    return not self.waiting and not self.running
```

#### 语法作用

定义方法 `is_finished()`，返回布尔值。

#### 工程作用

如果 waiting 和 running 都为空，说明没有待处理请求，也没有正在生成的请求。

#### 在推理流程中的意义

`LLMEngine.generate()` 用它控制主循环：

```python
while not self.is_finished():
    output, num_tokens = self.step()
```

所以它决定整个 generation 什么时候结束。

---

### 3.13 添加请求

```python
def add(self, seq: Sequence):
    self.waiting.append(seq)
```

#### 语法作用

定义 `add()` 方法，接收一个 `Sequence` 对象，并追加到 waiting 队列尾部。

#### 工程作用

新请求不会直接进入 running，而是先进入 waiting，等待 Prefill。

#### 在推理流程中的意义

`LLMEngine.add_request()` 会把 prompt 包装成 `Sequence`，然后调用：

```python
self.scheduler.add(seq)
```

也就是说，调度器接收到的不是原始字符串，而是已经编码好并封装了采样参数的 `Sequence`。

---

## 3.14 核心调度函数 `schedule()`

```python
def schedule(self) -> tuple[list[Sequence], bool]:
```

#### 语法作用

定义 `schedule()` 方法，返回值类型为：

```python
tuple[list[Sequence], bool]
```

也就是返回两个东西：

```text
1. 本轮要执行的 Sequence 列表
2. 本轮是否是 Prefill
```

#### 工程作用

这是整个 `scheduler.py` 最核心的方法。

它的整体逻辑是：

```text
先尝试调度 waiting 队列做 Prefill
如果有任何 Prefill 请求被调度，直接返回 Prefill batch
如果没有 Prefill 请求，再调度 running 队列做 Decode
```

这说明 nano-vLLM 当前策略是：

```text
Prefill 优先，并且 Prefill / Decode 不混合执行。
```

---

### 3.15 初始化本轮调度列表

```python
scheduled_seqs = []
num_batched_tokens = 0
```

#### 语法作用

创建空列表和计数器。

#### 工程作用

| 变量 | 类型 | 含义 |
|---|---|---|
| `scheduled_seqs` | `list[Sequence]` | 本轮被选中的请求 |
| `num_batched_tokens` | `int` | Prefill 阶段已调度 token 总数 |

#### 在推理流程中的意义

`scheduled_seqs` 最终会传给 `ModelRunner.run()`。

`num_batched_tokens` 用于限制 Prefill 本轮计算量不能超过 `max_num_batched_tokens`。

---

## 3.16 Prefill 调度循环

```python
# prefill
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
```

#### 语法作用

只要 waiting 队列非空，并且当前已调度 Sequence 数量小于上限，就持续从 waiting 队列挑选请求。

#### 工程作用

这个循环处理还没有完成 prompt 计算的请求。

#### 在推理流程中的意义

Prefill 阶段负责把 prompt token 输入模型，建立 KV Cache，并通常生成第一个 completion token。

---

### 3.17 取 waiting 队头请求

```python
seq = self.waiting[0]
```

#### 语法作用

读取 waiting 队列队头元素，但不弹出。

#### 工程作用

调度器优先处理最早进入 waiting 的请求。

不立即 `popleft()` 的原因是：

```text
只有当该 seq 的 prompt token 全部完成 prefill 后，才真正从 waiting 移到 running。
如果只是 chunked prefill 的一部分，它仍然留在 waiting 队头。
```

#### 在推理流程中的意义

这保证长 prompt 可以分块执行，而不会在没完成前进入 decode。

---

### 3.18 计算本轮剩余 token 容量

```python
remaining = self.max_num_batched_tokens - num_batched_tokens
if remaining == 0:
    break
```

#### 语法作用

计算本轮 Prefill 还能容纳多少 token。如果没有剩余容量，就退出 prefill 调度。

#### 工程作用

防止一轮 Prefill 处理过多 token，导致显存或计算量过大。

#### 在推理流程中的意义

Prefill 的计算成本和 prompt token 数强相关，因此需要 `max_num_batched_tokens` 限制。

---

### 3.19 判断是否首次分配 block

```python
if not seq.block_table:
```

#### 语法作用

判断当前 Sequence 是否还没有 KV Cache block 表。

`seq.block_table` 是列表，空列表在布尔判断中为 False。

#### 工程作用

如果 `block_table` 为空，说明该请求还没有分配 KV Cache block。

这种情况通常是：

```text
1. 新请求第一次 prefill
2. 被 preempt 后重新进入 waiting，并且原 block 已释放
```

#### 在推理流程中的意义

只有分配了 block，后续 Attention 才知道这条 Sequence 的 KV Cache 应该写到哪些物理 block。

---

### 3.20 检查能否分配 KV Cache block

```python
num_cached_blocks = self.block_manager.can_allocate(seq)
if num_cached_blocks == -1:
    break
```

#### 语法作用

调用 `BlockManager.can_allocate(seq)`，返回可复用的 prefix cache block 数，或者返回 `-1` 表示当前无法分配。

#### 工程作用

这一步做两件事：

```text
1. 判断可用 KV Cache block 是否足够
2. 检查当前 seq 是否能命中 prefix cache
```

`num_cached_blocks` 的含义是：

```text
当前请求前多少个完整 block 已经被缓存，可以复用，不需要重新 prefill
```

如果返回 `-1`，说明 KV Cache 空间不足，当前 waiting 队头请求无法调度。

#### 在推理流程中的意义

这是 PagedAttention / Prefix Caching 和 Scheduler 的交汇点。

调度器不是盲目把请求送去模型，而是先问 BlockManager：

```text
这个请求的 KV Cache block 能不能安排？
有多少前缀已经缓存？
```

---

### 3.21 计算本次还需要 Prefill 的 token 数

```python
num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
```

#### 语法作用

根据总 token 数减去已经缓存的完整 block token 数，得到还需要计算的 token 数。

#### 工程作用

如果 prefix cache 命中了一部分 block，就不需要重新计算这些 token。

例如：

```text
seq.num_tokens = 1000
block_size = 256
num_cached_blocks = 2

已缓存 token ≈ 512
还需要 prefill token = 1000 - 512 = 488
```

#### 在推理流程中的意义

这体现了 prefix caching 的性能价值：

```text
相同前缀越多，Prefill 需要计算的 token 越少，TTFT 越低。
```

---

### 3.22 已有 block_table 时的 num_tokens 计算

```python
else:
    num_tokens = seq.num_tokens - seq.num_cached_tokens
```

#### 语法作用

如果 `seq.block_table` 不为空，则说明该请求已经分配过 block，本轮只需要计算尚未缓存的 token。

#### 工程作用

这种情况通常发生在 **chunked prefill**：

```text
某个 prompt 太长，一轮 max_num_batched_tokens 不够处理完。
第一轮处理前半部分。
下一轮继续处理剩余部分。
```

这时 `seq.num_cached_tokens` 记录已经 prefill 完的 token 数。

#### 在推理流程中的意义

这让 nano-vLLM 支持长 prompt 分块 Prefill。

---

### 3.23 限制 chunked prefill 只给第一个 seq

```python
if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
    break
```

#### 语法作用

如果剩余 token 容量不足以完整处理当前 seq，并且本轮已经调度过其他 seq，就停止继续调度。

#### 工程作用

注释说明：

```text
only allow chunked prefill for the first seq
```

也就是说，只有本轮第一个被调度的 Sequence 可以被切分 Prefill。

例如：

```text
max_num_batched_tokens = 1000

情况 A：
第一个 seq 需要 1500 token
允许调度其中 1000 token，做 chunked prefill

情况 B：
已经调度了 seq1 800 token
剩余 200 token
seq2 需要 500 token
不允许只调度 seq2 的 200 token，直接 break
```

#### 在推理流程中的意义

这是一个简化调度策略，避免一个 batch 中出现多个被切碎的 prefill 请求，从而降低实现复杂度。

---

### 3.24 首次调度时真正分配 block

```python
if not seq.block_table:
    self.block_manager.allocate(seq, num_cached_blocks)
```

#### 语法作用

如果当前 Sequence 还没有 block_table，则调用 BlockManager 分配 KV Cache block。

#### 工程作用

`allocate()` 会为 Sequence 建立 `block_table`。

`block_table` 可以理解为：

```text
逻辑 block id → 物理 KV Cache block id 的映射
```

例如：

```text
seq 的逻辑 block：0, 1, 2
实际物理 block：5, 9, 13

seq.block_table = [5, 9, 13]
```

#### 在推理流程中的意义

后面的 `ModelRunner.prepare_prefill()` 和 `Attention` 会依赖 `block_table` 找到 KV Cache 写入位置。

---

### 3.25 设置本轮计划处理 token 数

```python
seq.num_scheduled_tokens = min(num_tokens, remaining)
```

#### 语法作用

设置当前 Sequence 本轮要处理的 token 数。

#### 工程作用

如果剩余容量足够：

```text
num_scheduled_tokens = num_tokens
```

如果当前 prompt 太长，只能处理一部分：

```text
num_scheduled_tokens = remaining
```

#### 在推理流程中的意义

`ModelRunner.prepare_prefill()` 会根据这个字段决定本轮实际送入模型的 token 范围。

---

### 3.26 更新本轮 batch token 计数

```python
num_batched_tokens += seq.num_scheduled_tokens
```

#### 语法作用

把当前 Sequence 本轮调度的 token 数加到总计数中。

#### 工程作用

用于判断后续还能不能继续加入更多 waiting 请求。

#### 在推理流程中的意义

它保证本轮 prefill 的总 token 数不超过 `max_num_batched_tokens`。

---

### 3.27 判断当前 Sequence 是否 Prefill 完成

```python
if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
```

#### 语法作用

判断当前已缓存 token 数加上本轮计划处理 token 数，是否等于当前 Sequence 的总 token 数。

#### 工程作用

如果等于，说明 prompt 部分在本轮执行后就会完整 prefill 完成。

注意：此处 `seq.num_tokens` 在 prefill 阶段主要代表 prompt token 数，因为 completion token 还没 append，或者还没正式进入后续 decode。

#### 在推理流程中的意义

只有完整完成 prefill 的请求才能进入 running 队列，参与后续 decode。

---

### 3.28 状态切换到 RUNNING

```python
seq.status = SequenceStatus.RUNNING
self.waiting.popleft()
self.running.append(seq)
```

#### 语法作用

把当前 Sequence 状态改为 `RUNNING`，从 waiting 队头弹出，并加入 running 队尾。

#### 工程作用

表示该请求即将完成 prompt prefill，可以进入 decode 阶段。

#### 在推理流程中的意义

这是请求生命周期中的关键状态转移：

```text
WAITING → RUNNING
```

对应：

```text
还没完成 prompt KV Cache
    ↓
已经完成 prompt KV Cache，可以逐 token 生成
```

---

### 3.29 加入本轮 scheduled_seqs

```python
scheduled_seqs.append(seq)
```

#### 语法作用

把当前 Sequence 加入本轮调度列表。

#### 工程作用

最终这些 `scheduled_seqs` 会被返回给 `LLMEngine.step()`，再传给 `ModelRunner.run()`。

#### 在推理流程中的意义

这一步正式确认该请求会参与本轮模型 forward。

---

### 3.30 Prefill 优先返回

```python
if scheduled_seqs:
    return scheduled_seqs, True
```

#### 语法作用

如果本轮已经调度了任何 prefill 请求，则直接返回：

```python
(scheduled_seqs, True)
```

#### 工程作用

这里的 `True` 表示本轮是 Prefill。

#### 在推理流程中的意义

这说明 nano-vLLM 当前调度策略是：

```text
只要有 Prefill batch，就不执行 Decode。
```

也就是说，它不把 Prefill 和 Decode 混合在同一个 batch 中。

这是一种简化实现。生产级 vLLM 的调度策略会更复杂。

---

## 3.31 Decode 调度循环

```python
# decode
while self.running and len(scheduled_seqs) < self.max_num_seqs:
```

#### 语法作用

如果没有 Prefill 被调度，就进入 Decode 阶段。

循环条件是：

```text
running 队列非空
并且本轮调度的 seq 数量小于 max_num_seqs
```

#### 工程作用

从 running 队列中选择请求，每条请求本轮生成一个 token。

#### 在推理流程中的意义

Decode 是自回归生成的主体阶段。

每次 decode 通常做：

```text
输入每条 seq 的 last_token
读取历史 KV Cache
生成下一个 token
```

---

### 3.32 从 running 队头取出请求

```python
seq = self.running.popleft()
```

#### 语法作用

从 running 队列左侧弹出一个 Sequence。

#### 工程作用

调度器按照 running 队列顺序选择请求。

#### 在推理流程中的意义

被取出的 seq 暂时离开 running 队列，等待检查是否有足够 block append 新 token。

---

### 3.33 判断是否可以追加 token

```python
while not self.block_manager.can_append(seq):
```

#### 语法作用

只要当前 Sequence 无法 append 新 token，就进入循环处理。

#### 工程作用

Decode 阶段每生成一个 token，Sequence 长度加 1，对应 KV Cache 也要存储这个新 token 的 K/V。

如果当前 token 仍在已有 block 空间内，不需要新 block。

如果新 token 会落到一个新 block 中，就必须有空闲 block。

`can_append(seq)` 用来判断：

```text
当前 seq 生成下一个 token 所需的 KV Cache block 是否足够
```

#### 在推理流程中的意义

Decode 不是只算 logits，还必须保证新 token 的 KV 能写入 cache。否则后续 attention 无法复用历史。

---

### 3.34 如果有其他 running 请求，抢占队尾请求

```python
if self.running:
    self.preempt(self.running.pop())
```

#### 语法作用

如果 running 队列里还有其他请求，就从队尾弹出一个请求并抢占。

#### 工程作用

当当前 seq 没有足够 block append 新 token 时，调度器尝试抢占其他 running 请求，释放它占用的 KV Cache block。

为什么抢占 `self.running.pop()`？

```text
popleft() 取的是较早进入 running 的请求
pop() 抢占的是队尾请求
```

这类似于保留前面的请求，牺牲后面的请求。

#### 在推理流程中的意义

抢占是为了在 KV Cache 空间不足时继续推进部分请求，而不是整个系统停住。

---

### 3.35 如果没有其他请求，抢占当前 seq

```python
else:
    self.preempt(seq)
    break
```

#### 语法作用

如果 running 队列已经没有其他请求可以抢占，就抢占当前 seq，并跳出 while 循环。

#### 工程作用

说明当前请求也无法 append 新 token，且没有其他请求能释放资源，只能把当前请求放回 waiting，释放其 block，等待以后重新 prefill。

#### 在推理流程中的意义

这是极端显存不足时的保护逻辑。

被抢占的请求会丢失当前 KV Cache block，之后需要重新 prefill。

---

### 3.36 `while ... else` 分支：成功可 append

```python
else:
    seq.num_scheduled_tokens = 1
    seq.is_prefill = False
    self.block_manager.may_append(seq)
    scheduled_seqs.append(seq)
```

#### 语法作用

这是 Python 的 `while ... else` 结构。

当 `while not self.block_manager.can_append(seq)` 没有被 `break` 打断，而是正常结束时，会执行 `else`。

也就是说：

```text
当前 seq 最终可以 append 新 token
```

#### 工程作用

这几行表示当前 seq 被成功调度进入 Decode batch：

```python
seq.num_scheduled_tokens = 1
```

Decode 阶段每条 Sequence 本轮只生成 1 个 token。

```python
seq.is_prefill = False
```

标记当前不是 Prefill，而是 Decode。

```python
self.block_manager.may_append(seq)
```

如果新 token 需要新 block，则真正追加 block。

```python
scheduled_seqs.append(seq)
```

加入本轮 decode batch。

#### 在推理流程中的意义

这一步完成 Decode 调度。

它告诉后面的 ModelRunner：

```text
这条 seq 本轮只需要输入 last_token，生成一个新 token。
```

---

### 3.37 确保 decode 至少调度了一个请求

```python
assert scheduled_seqs
```

#### 语法作用

断言 `scheduled_seqs` 非空。

如果为空，程序会抛出 AssertionError。

#### 工程作用

在进入 decode 逻辑时，按理说 running 队列不为空，应至少能调度一个请求。

如果无法调度任何请求，说明调度逻辑或 block 管理可能出现异常。

#### 在推理流程中的意义

这是一个调试保护，防止 `ModelRunner.run()` 收到空 batch。

---

### 3.38 把已调度的 Decode 请求放回 running 队列

```python
self.running.extendleft(reversed(scheduled_seqs))
```

#### 语法作用

把 `scheduled_seqs` 重新放回 running 队列左侧。

这里使用：

```python
reversed(scheduled_seqs)
```

是为了保持原来的顺序。

例如：

```text
scheduled_seqs = [A, B, C]
reversed(scheduled_seqs) = [C, B, A]

extendleft 会依次从左侧插入：
插入 C → [C]
插入 B → [B, C]
插入 A → [A, B, C]
```

最终顺序仍然是 `[A, B, C]`。

#### 工程作用

这些请求本轮会被模型执行，但还没有完成，所以需要继续保留在 running 队列。

如果某些请求在 `postprocess()` 后完成，后处理会把它们从 running 中移除。

#### 在推理流程中的意义

这维持了 running 队列的持续 decode 状态。

---

### 3.39 返回 Decode batch

```python
return scheduled_seqs, False
```

#### 语法作用

返回本轮调度出的请求列表，以及 `False` 表示 Decode 阶段。

#### 工程作用

`LLMEngine.step()` 会把它传给：

```python
self.model_runner.call("run", seqs, is_prefill)
```

#### 在推理流程中的意义

`ModelRunner` 看到 `is_prefill=False`，会走 decode 输入准备逻辑，例如只输入每条 Sequence 的 `last_token`。

---

## 3.40 抢占方法 `preempt()`

```python
def preempt(self, seq: Sequence):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```

#### 语法作用

定义 `preempt()` 方法，接收一个 `Sequence`，对其执行抢占处理。

#### 工程作用

抢占做了四件事：

```text
1. 状态改回 WAITING
2. 标记 is_prefill=True
3. 释放它占用的 KV Cache block
4. 放回 waiting 队头
```

#### 在推理流程中的意义

抢占意味着这条请求的 KV Cache 被丢弃，后续需要重新 prefill。

这是一种显存不足时的退让机制：

```text
释放部分 running 请求的 KV Cache
让当前或更优先的请求继续 decode
```

与 vLLM 原版相比，这里的抢占策略比较简单：

```text
直接释放整条 seq 的 KV Cache
重新放回 waiting
后续重新 prefill
```

生产级系统可能会有更复杂的 recompute / swap 策略。

---

## 3.41 后处理方法 `postprocess()`

```python
def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
```

#### 语法作用

定义后处理方法。

参数类型：

| 参数 | 类型 | 含义 |
|---|---|---|
| `seqs` | `list[Sequence]` | 本轮执行的请求 |
| `token_ids` | `list[int]` | ModelRunner 为每条请求采样出的新 token id |
| `is_prefill` | `bool` | 本轮是否是 Prefill |

#### 工程作用

模型执行后，调度器要根据结果更新每条 Sequence。

包括：

```text
更新 cached token 数
清空 scheduled token 数
必要时追加新 token
判断是否完成
完成后释放 KV Cache
```

#### 在推理流程中的意义

`postprocess()` 是模型执行结果回写到请求状态的地方。

---

### 3.42 遍历每条 seq 和生成 token

```python
for seq, token_id in zip(seqs, token_ids):
```

#### 语法作用

把本轮请求列表和模型输出 token 列表一一配对。

#### 工程作用

通常：

```text
seqs[i] 对应 token_ids[i]
```

每条 Sequence 得到一个新 token。

#### 在推理流程中的意义

这是从 batch 级模型输出回到单请求状态管理的过程。

---

### 3.43 哈希已完成 block

```python
self.block_manager.hash_blocks(seq)
```

#### 语法作用

调用 BlockManager 的 `hash_blocks()` 方法。

#### 工程作用

它会把当前 Sequence 中已经完整计算过的 block 做哈希登记，用于 prefix caching。

例如，如果某条请求的前 256 个 token 已经完成 KV Cache 计算，就可以把这个 block 的 token 内容哈希后记录起来。

后续如果新请求有相同前缀，就可以复用这个 block。

#### 在推理流程中的意义

这一步把已经完成 prefill/decode 的 block 加入 prefix cache 系统。

它是 Prefix Caching 的关键后处理。

---

### 3.44 更新 cached token 数

```python
seq.num_cached_tokens += seq.num_scheduled_tokens
```

#### 语法作用

把本轮已经执行过的 token 数加入 `num_cached_tokens`。

#### 工程作用

`num_cached_tokens` 表示：

```text
当前 Sequence 中已经完成模型计算，并且对应 KV Cache 已经可用的 token 数
```

#### 在推理流程中的意义

Prefill 分块时尤其重要。

例如：

```text
prompt 长度 3000
max_num_batched_tokens = 1024

第 1 轮后 num_cached_tokens = 1024
第 2 轮后 num_cached_tokens = 2048
第 3 轮后 num_cached_tokens = 3000
```

只有全部 prompt token cached 后，才能正式 append 生成 token。

---

### 3.45 清空本轮 scheduled token 数

```python
seq.num_scheduled_tokens = 0
```

#### 语法作用

把当前 Sequence 的本轮调度 token 数归零。

#### 工程作用

`num_scheduled_tokens` 只是“一轮临时状态”，执行完后要清空，避免影响下一轮调度。

#### 在推理流程中的意义

下一轮 `schedule()` 会重新设置它。

---

### 3.46 处理未完成的 chunked prefill

```python
if is_prefill and seq.num_cached_tokens < seq.num_tokens:
    continue
```

#### 语法作用

如果当前是 Prefill 阶段，并且当前 seq 还没有把所有 token 都缓存完，则跳过后续 append token 逻辑。

#### 工程作用

这是 chunked prefill 的关键判断。

如果一个长 prompt 只 prefill 了一部分，那么此时还不能把模型输出的 token 当作 completion token 追加，因为完整 prompt 还没处理完。

#### 在推理流程中的意义

只有当 prompt 全部 prefill 完成后，模型输出的 token 才能作为第一个生成 token 被 append。

---

### 3.47 追加生成 token

```python
seq.append_token(token_id)
```

#### 语法作用

调用 Sequence 的 `append_token()` 方法，把新 token 加到 `seq.token_ids` 末尾。

#### 工程作用

这一步把模型本轮生成的 token 真正写入请求状态。

内部会更新：

```text
token_ids
last_token
num_tokens
```

#### 在推理流程中的意义

自回归生成依赖 `last_token`。

本轮 append 的 token 会成为下一轮 decode 的输入。

```text
本轮生成 token_i
    ↓
append 到 Sequence
    ↓
下一轮用 token_i 作为 last_token 继续生成 token_{i+1}
```

---

### 3.48 判断是否生成结束

```python
if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
```

#### 语法作用

判断当前 Sequence 是否满足结束条件。

结束条件有两个：

```text
1. 没有忽略 EOS，并且当前 token 是 EOS
2. 已生成 token 数达到 max_tokens
```

#### 工程作用

这对应两种停止方式：

| 条件 | 含义 |
|---|---|
| `token_id == self.eos` | 模型主动结束 |
| `num_completion_tokens == max_tokens` | 达到用户设置的最大生成长度 |

`ignore_eos=True` 时，即使模型生成 EOS，也继续生成，直到 `max_tokens`。

#### 在推理流程中的意义

这是每条请求结束的最终判断。

---

### 3.49 标记完成、释放 block、移出 running

```python
seq.status = SequenceStatus.FINISHED
self.block_manager.deallocate(seq)
self.running.remove(seq)
```

#### 语法作用

把请求状态标记为完成，释放其 KV Cache block，并从 running 队列中移除。

#### 工程作用

完成的请求不应该继续占用显存，也不应该继续参与后续 decode。

#### 在推理流程中的意义

这是请求生命周期的终点：

```text
RUNNING → FINISHED
```

同时释放 KV Cache block，使其他请求可以使用这些显存资源。

---

## 4. 背后的框架性原理

### 4.1 request / sequence

在 nano-vLLM 中，一条用户请求会被包装成 `Sequence`。

`Scheduler` 不直接处理字符串 prompt，而是处理 `Sequence` 对象。

每个 `Sequence` 记录：

```text
seq_id
status
token_ids
last_token
num_tokens
num_prompt_tokens
num_cached_tokens
num_scheduled_tokens
block_table
temperature
max_tokens
ignore_eos
```

在本文件中体现为：

```python
def add(self, seq: Sequence):
    self.waiting.append(seq)
```

`Scheduler` 的核心任务就是调度这些 `Sequence`。

---

### 4.2 scheduler

Scheduler 是推理引擎中决定执行顺序的模块。

它必须同时考虑：

```text
请求数量上限
Prefill token 数上限
KV Cache block 是否足够
请求是否已经完成 prefill
请求是否达到 EOS / max_tokens
```

本文件中的调度策略是：

```text
1. 优先处理 waiting 队列的 Prefill
2. 如果本轮有 Prefill，就不做 Decode
3. 如果没有 Prefill，再处理 running 队列的 Decode
4. Decode 前检查能否 append KV Cache block
5. 如果 block 不够，抢占其他 running 请求
```

---

### 4.3 Prefill / Decode

Prefill：

```text
处理 prompt token，建立 KV Cache。
```

Decode：

```text
每轮输入 last_token，读取历史 KV Cache，生成一个新 token。
```

在本文件中：

```python
return scheduled_seqs, True
```

表示 Prefill。

```python
return scheduled_seqs, False
```

表示 Decode。

这是 `Scheduler` 和 `ModelRunner` 之间的重要协议。

---

### 4.4 KV Cache

KV Cache 保存历史 token 在每层 attention 中的 Key / Value。

它的重要性在于：

```text
Decode 阶段不需要重复计算全部历史 token 的 K/V。
```

在本文件中，Scheduler 不直接操作 KV Cache tensor，而是通过 BlockManager 间接管理：

```python
self.block_manager.can_allocate(seq)
self.block_manager.allocate(seq, num_cached_blocks)
self.block_manager.can_append(seq)
self.block_manager.may_append(seq)
self.block_manager.deallocate(seq)
```

---

### 4.5 block / block table / block manager

vLLM 的核心思想之一是把 KV Cache 按 block 管理。

不是每条请求都分配一整段连续 KV Cache，而是分配多个 block。

`block_table` 表示：

```text
当前 Sequence 的逻辑 token block 对应哪些物理 KV Cache block。
```

在本文件中：

```python
if not seq.block_table:
    self.block_manager.allocate(seq, num_cached_blocks)
```

说明新请求必须先分配 block_table，才能进入模型执行。

---

### 4.6 attention

Attention 在本文件中没有直接出现，但 Scheduler 的调度结果会影响 Attention 的执行方式。

Prefill 时：

```text
Attention 处理 prompt 的多个 token，并把 K/V 写入 KV Cache。
```

Decode 时：

```text
Attention 只处理当前 token 的 query，同时读取历史 KV Cache。
```

`is_prefill` 就是传递给 Attention 间接使用的重要信号。

---

### 4.7 model runner

`ModelRunner` 是真正执行模型的模块。

Scheduler 只返回：

```python
seqs, is_prefill
```

然后 `LLMEngine.step()` 传给 ModelRunner：

```python
token_ids = self.model_runner.call("run", seqs, is_prefill)
```

所以 Scheduler 和 ModelRunner 的分工是：

```text
Scheduler：决定跑谁、跑 prefill 还是 decode
ModelRunner：准备 tensor、执行 forward、采样 token
```

---

### 4.8 sampling / logits / temperature

本文件不直接处理 logits 和 temperature。

但 `postprocess()` 接收的 `token_ids` 是 ModelRunner 已经完成 logits 计算和采样之后的结果。

流程是：

```text
Scheduler.schedule()
    ↓
ModelRunner.run()
    ↓
模型输出 logits
    ↓
Sampler 根据 temperature 采样 token_id
    ↓
Scheduler.postprocess() append token_id
```

---

### 4.9 throughput / latency / TTFT / TPOT

`scheduler.py` 不直接计算性能指标，但它强烈影响这些指标。

| 指标 | 与 Scheduler 的关系 |
|---|---|
| TTFT | Prefill 调度越快，首 token 越快返回 |
| TPOT | Decode batch 越稳定，每 token 平均延迟越低 |
| Throughput | `max_num_seqs` 和 `max_num_batched_tokens` 决定批处理规模 |
| Latency | Prefill 优先策略可能让 decode 请求等待，从而影响部分请求延迟 |

本文件中的关键调度参数：

```python
self.max_num_seqs
self.max_num_batched_tokens
```

分别控制：

```text
Decode 并发请求数
Prefill 一轮 token 数
```

---

## 5. 和 vLLM 原版设计的关系

### 5.1 连续批处理 Continuous Batching

本文件体现了连续批处理思想。

请求不是固定成一个 batch 后一起开始、一起结束，而是：

```text
新请求进入 waiting
完成 prefill 后进入 running
decode 中请求完成后释放资源
其他请求继续执行
```

这体现在：

```python
self.waiting: deque[Sequence] = deque()
self.running: deque[Sequence] = deque()
```

以及：

```python
while self.waiting ...
while self.running ...
```

---

### 5.2 PagedAttention / KV Cache block 管理

vLLM 的核心是 PagedAttention，即把 KV Cache 切成 block 管理。

本文件通过 `BlockManager` 体现这一思想：

```python
self.block_manager = BlockManager(...)
self.block_manager.allocate(seq, num_cached_blocks)
self.block_manager.may_append(seq)
self.block_manager.deallocate(seq)
```

Scheduler 不直接读写 KV Cache tensor，但它决定 block 什么时候分配、追加、释放。

---

### 5.3 Request / Sequence 调度

本文件直接体现 request / sequence 调度。

每个用户请求在内部是 `Sequence`，并在以下状态之间流转：

```text
WAITING → RUNNING → FINISHED
```

抢占时还可能：

```text
RUNNING → WAITING
```

对应代码：

```python
seq.status = SequenceStatus.RUNNING
seq.status = SequenceStatus.WAITING
seq.status = SequenceStatus.FINISHED
```

---

### 5.4 Prefill / Decode 分离

本文件非常明确地区分 Prefill 和 Decode：

```python
if scheduled_seqs:
    return scheduled_seqs, True
```

```python
return scheduled_seqs, False
```

这正是大模型推理引擎的基础设计。

Prefill 和 Decode 的计算特征完全不同：

```text
Prefill：大矩阵计算，处理很多 prompt token
Decode：每条请求一个 token，高频小步循环
```

所以调度器必须把它们分开处理。

---

### 5.5 高吞吐推理服务

本文件通过批处理参数提高吞吐：

```python
self.max_num_seqs
self.max_num_batched_tokens
```

通过这两个参数，Scheduler 能够：

```text
一次 Prefill 处理多个请求的 prompt token
一次 Decode 处理多个 running 请求
请求完成后及时释放 KV Cache
```

这就是高吞吐推理服务的基础。

---

### 5.6 nano-vLLM 相比原版 vLLM 的简化点

从本文件可以看出，nano-vLLM 的 Scheduler 比生产级 vLLM 简化很多：

```text
1. Prefill 和 Decode 不混合执行
2. 调度策略比较简单，Prefill 优先
3. 没有复杂优先级队列
4. 没有复杂 SLA / latency-aware 调度
5. 抢占策略简单，直接 deallocate 后回到 waiting
6. 没有 swap 到 CPU 的复杂机制
7. chunked prefill 只允许本轮第一个 seq 使用
8. 没有 sequence group / beam search 等复杂结构
```

但这些简化反而让它非常适合学习 vLLM 的主线思想：

```text
Sequence 状态管理
waiting / running 队列
Prefill / Decode 分离
KV Cache block 管理
抢占与资源释放
```

---

## 6. 总结：这个文件应该怎么学

`scheduler.py` 是 nano-vLLM 里最重要的核心文件之一。

你应该重点抓住这条逻辑：

```text
新请求进入 waiting
    ↓
Scheduler 优先调度 Prefill
    ↓
BlockManager 分配 KV Cache block
    ↓
Prefill 完成后进入 running
    ↓
Scheduler 调度 Decode
    ↓
每条 running seq 每轮生成 1 个 token
    ↓
KV Cache 不足时 preempt
    ↓
ModelRunner 返回 token_id
    ↓
postprocess append token
    ↓
EOS 或 max_tokens 后 FINISHED
    ↓
释放 KV Cache block
```

最重要的几个字段是：

```text
waiting
running
max_num_seqs
max_num_batched_tokens
num_scheduled_tokens
num_cached_tokens
block_table
is_prefill
```

最重要的几个方法是：

```text
schedule()
postprocess()
preempt()
```

下一步建议阅读顺序：

```text
1. sequence.py
   理解 Sequence 保存了哪些请求状态

2. block_manager.py
   理解 can_allocate / allocate / can_append / may_append / deallocate

3. model_runner.py
   理解 Scheduler 返回的 seqs 和 is_prefill 如何变成模型输入 tensor

4. attention.py
   理解 block_table 和 slot_mapping 如何真正用于 KV Cache 读写
```


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]
- 关联阅读：[[outputs/项目整理/专题-01-请求从输入到输出|专题-01-请求从输入到输出]]

%% 项目关联导航：结束 %%
