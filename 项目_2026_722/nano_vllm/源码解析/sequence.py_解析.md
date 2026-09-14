# sequence.py 源码逐行精读与框架原理解析

> 解析对象：`sequence.py`  
> 解析目标：严格按照上传的“nano-vLLM 源码逐行精读提示词”结构，对该文件进行源码逐行精读 + 框架原理解释。

---

## 1. 文件整体定位

### 1.1 这个文件负责什么

`sequence.py` 定义了 nano-vLLM 内部表示单条推理请求的核心数据结构：`Sequence`。

用户在外层传入的是 prompt，例如字符串：

```text
introduce yourself
```

经过 tokenizer 编码后，会变成 token id 列表：

```python
[15496, 457, 1312, ...]
```

这些 token ids 进入推理引擎后，需要被包装成一个带状态的对象。这个对象不仅保存 prompt token，还保存已经生成的 completion token、请求状态、KV Cache block 映射表、调度进度、采样参数等。这个对象就是 `Sequence`。

一句话概括：

```text
Sequence = 一条请求在推理引擎内部的完整状态载体
```

它不是模型层，不负责 Attention 计算；也不是 Scheduler，不直接决定谁被调度；也不是 BlockManager，不直接分配 KV Cache。它负责保存一条请求在整个生命周期中被各模块共享和修改的状态。

### 1.2 它属于哪一层

该文件属于：

```text
请求状态层 / Sequence 管理层 / 推理引擎基础数据结构层
```

在 nano-vLLM 中的位置大致是：

```text
用户 API 层：LLM.generate()
    ↓
请求封装层：Sequence
    ↓
调度层：Scheduler
    ↓
KV Cache 管理层：BlockManager
    ↓
模型执行层：ModelRunner
    ↓
Attention / Sampler / CUDA 执行层
```

### 1.3 它和项目中哪些文件有关

| 文件 | 关系 |
|---|---|
| `llm_engine.py` | `add_request()` 中创建 `Sequence(prompt, sampling_params)`，并加入 Scheduler |
| `sampling_params.py` | `Sequence` 从 `SamplingParams` 读取 `temperature`、`max_tokens`、`ignore_eos` |
| `scheduler.py` | Scheduler 读取和修改 Sequence 的状态，例如 WAITING、RUNNING、FINISHED |
| `block_manager.py` | 根据 `num_blocks`、`block_table`、`block(i)` 分配、复用、释放 KV Cache block |
| `model_runner.py` | 根据 Sequence 的 `token_ids`、`last_token`、`block_table` 准备 prefill/decode 输入 |
| `attention.py` | 间接使用 Sequence 的 `block_table` 来定位 KV Cache 中的历史 K/V |

### 1.4 它在完整推理流程中的位置

```text
用户输入 prompt
    ↓
tokenizer.encode(prompt)
    ↓
token_ids: list[int]
    ↓
Sequence(token_ids, sampling_params)
    ↓
Scheduler.add(seq)
    ↓
Scheduler.schedule()
    ↓
ModelRunner.run(seqs, is_prefill)
    ↓
Scheduler.postprocess(seqs, token_ids, is_prefill)
    ↓
Sequence.append_token(token_id)
    ↓
Sequence.status = FINISHED
    ↓
LLMEngine 收集 seq.completion_token_ids
    ↓
tokenizer.decode()
    ↓
返回最终文本
```

所以，`Sequence` 贯穿请求从创建、prefill、decode 到结束的完整生命周期。

---

## 2. 代码结构总览

### 2.1 源码

```python
from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
```

### 2.2 导入了哪些模块

| 导入对象 | 作用 |
|---|---|
| `copy` | 浅拷贝传入的 `token_ids`，避免外部列表修改影响 Sequence 内部状态 |
| `Enum` | 定义请求状态枚举类 |
| `auto` | 自动给枚举成员分配值 |
| `count` | 创建自增计数器，用于生成唯一 `seq_id` |
| `SamplingParams` | 提供采样参数，如 `temperature`、`max_tokens`、`ignore_eos` |

### 2.3 定义了哪些类

本文件定义了两个类：

```text
SequenceStatus
Sequence
```

其中：

```text
SequenceStatus：表示请求状态
Sequence：表示一条具体推理请求
```

### 2.4 定义了哪些函数 / 方法

| 方法 / 属性 | 作用 |
|---|---|
| `__init__` | 初始化一条请求 |
| `__len__` | 返回当前 token 总数 |
| `__getitem__` | 允许像列表一样访问 token |
| `is_finished` | 判断请求是否完成 |
| `num_completion_tokens` | 计算已生成 completion token 数 |
| `prompt_token_ids` | 返回 prompt 部分 token |
| `completion_token_ids` | 返回 completion 部分 token |
| `num_blocks` | 计算当前请求需要多少 KV Cache block |
| `last_block_num_tokens` | 计算最后一个 block 内有多少 token |
| `block(i)` | 返回第 i 个逻辑 block 的 token 切片 |
| `append_token(token_id)` | decode 阶段追加新 token |
| `__getstate__` | 自定义序列化状态 |
| `__setstate__` | 自定义反序列化状态 |

### 2.5 重要变量或数据结构

| 字段 | 类型 | 含义 |
|---|---|---|
| `seq_id` | `int` | 请求唯一编号 |
| `status` | `SequenceStatus` | 当前请求状态 |
| `token_ids` | `list[int]` | prompt token + completion token |
| `last_token` | `int` | 当前最后一个 token |
| `num_tokens` | `int` | 当前 token 总数 |
| `num_prompt_tokens` | `int` | prompt token 数 |
| `num_cached_tokens` | `int` | 已经写入或命中 KV Cache 的 token 数 |
| `num_scheduled_tokens` | `int` | 本轮调度要处理的 token 数 |
| `is_prefill` | `bool` | 当前请求是否处于 prefill 相关阶段 |
| `block_table` | `list[int]` | 逻辑 block 到物理 KV Cache block 的映射 |
| `temperature` | `float` | 采样温度 |
| `max_tokens` | `int` | 最大生成 token 数 |
| `ignore_eos` | `bool` | 是否忽略 EOS |

### 2.6 哪些代码属于主流程

主流程相关代码：

```text
__init__
append_token
is_finished
prompt_token_ids
completion_token_ids
num_completion_tokens
```

这些直接参与请求创建、逐 token 生成、结束判断、输出收集。

### 2.7 哪些代码属于辅助逻辑

KV Cache / block 管理辅助：

```text
num_blocks
last_block_num_tokens
block(i)
block_table
num_cached_tokens
num_scheduled_tokens
```

多进程通信辅助：

```text
__getstate__
__setstate__
```

这些通常服务于 Scheduler、BlockManager、ModelRunner。

### 2.8 文件组织结构图

```text
sequence.py
├── import
│   ├── copy
│   ├── Enum / auto
│   ├── count
│   └── SamplingParams
│
├── SequenceStatus(Enum)
│   ├── WAITING
│   ├── RUNNING
│   └── FINISHED
│
└── Sequence
    ├── 类变量
    │   ├── block_size
    │   └── counter
    │
    ├── 初始化
    │   └── __init__
    │
    ├── 类列表行为
    │   ├── __len__
    │   └── __getitem__
    │
    ├── 请求状态属性
    │   ├── is_finished
    │   ├── num_completion_tokens
    │   ├── prompt_token_ids
    │   └── completion_token_ids
    │
    ├── KV Cache block 属性
    │   ├── num_blocks
    │   ├── last_block_num_tokens
    │   └── block(i)
    │
    ├── Decode 更新
    │   └── append_token
    │
    └── 多进程序列化
        ├── __getstate__
        └── __setstate__
```

---

## 3. 逐行代码解释

### 3.1 导入模块

```python
from copy import copy
```

**语法作用：** 从 Python 标准库 `copy` 中导入浅拷贝函数。

**工程作用：** 后面用于复制传入的 `token_ids`。

**推理流程意义：** 一条请求进入引擎后，Sequence 应该拥有自己的 token 状态。后续 decode 会不断追加 token，如果直接引用外部列表，外部修改可能污染内部状态。

**下一步结合文件：** 看 `llm_engine.py` 的 `add_request()`，它会把 tokenizer 编码后的 token ids 传给 `Sequence`。

---

```python
from enum import Enum, auto
```

**语法作用：** 导入枚举类基类和自动赋值工具。

**工程作用：** 用于定义请求状态 `SequenceStatus`。

**推理流程意义：** Scheduler 需要根据请求状态判断请求是否还在等待、是否正在运行、是否已经完成。

---

```python
from itertools import count
```

**语法作用：** 导入无限自增计数器。

**工程作用：** 用于生成全局递增的 `seq_id`。

**推理流程意义：** batch 中可能有很多请求，完成顺序不一定等于输入顺序。`seq_id` 可以保证最终输出按原始请求顺序还原。

---

```python
from nanovllm.sampling_params import SamplingParams
```

**语法作用：** 从 nano-vLLM 的采样参数模块导入 `SamplingParams`。

**工程作用：** Sequence 会从 SamplingParams 中读取生成控制参数。

**推理流程意义：** 每条请求都需要知道自己的 `temperature`、`max_tokens`、`ignore_eos`，这样 Scheduler 和 Sampler 才知道何时停止、如何采样。

**下一步结合文件：** 看 `sampling_params.py`。

---

### 3.2 SequenceStatus

```python
class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
```

**语法作用：** 定义枚举类，包含三个状态值。

**工程作用：** 表示一条请求的生命周期状态。

| 状态 | 含义 |
|---|---|
| `WAITING` | 等待调度，通常尚未完成 prefill |
| `RUNNING` | 正在运行，通常处于 decode 阶段 |
| `FINISHED` | 已生成结束，不再参与调度 |

**推理流程意义：** 请求通常按如下状态流转：

```text
WAITING
  ↓ prefill 完成
RUNNING
  ↓ EOS 或达到 max_tokens
FINISHED
```

如果资源不足，也可能出现运行中的请求被抢占，再回到 waiting 队列。具体要结合 `scheduler.py` 和 `block_manager.py` 看。

---

### 3.3 Sequence 类变量

```python
class Sequence:
    block_size = 256
    counter = count()
```

**语法作用：** 定义 `Sequence` 类，并设置两个类变量。

**工程作用：** `block_size` 表示 KV Cache 分块大小，默认 256；`counter` 用于给每条请求生成唯一 ID。

**推理流程意义：** KV Cache 不按一整条请求连续分配，而是按 block 分页管理。假设：

```text
block_size = 256
sequence 长度 = 600
```

则需要：

```text
ceil(600 / 256) = 3 个 block
```

`block_size` 是 PagedAttention / KV Cache block 管理的基础参数。

**下一步结合文件：** 看 `config.py` 中的 `kvcache_block_size`，以及 `llm_engine.py` 如何执行 `Sequence.block_size = config.kvcache_block_size`。

---

### 3.4 初始化方法

```python
def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
```

**语法作用：** 定义构造函数，`token_ids` 的类型标注为 `list[int]`。

**工程作用：** 创建一条新的推理请求。

**推理流程意义：** 当 `LLMEngine.add_request()` 收到 prompt 后，会创建：

```python
seq = Sequence(prompt, sampling_params)
```

从这一刻起，prompt 就变成了引擎内部的 Sequence 请求。

**注意点：** `sampling_params = SamplingParams()` 是默认参数对象。若该对象被修改，可能出现默认对象共享问题。当前代码只是读取字段，风险较低；更稳妥的写法是默认 `None`，函数内部再创建。

---

```python
self.seq_id = next(Sequence.counter)
```

**语法作用：** 从自增计数器中取下一个编号。

**工程作用：** 给请求分配唯一 ID。

**推理流程意义：** 方便最终按输入顺序排序输出，也方便调度器区分不同请求。

---

```python
self.status = SequenceStatus.WAITING
```

**语法作用：** 设置实例状态字段。

**工程作用：** 新请求默认处于等待状态。

**推理流程意义：** 请求刚创建时还没有经过 prefill，也还没有进入 decode，所以先进入 Scheduler 的 waiting 队列。

---

```python
self.token_ids = copy(token_ids)
```

**语法作用：** 浅拷贝 token id 列表。

**工程作用：** 保存当前请求完整 token 序列。

**推理流程意义：** 初始时它只包含 prompt token；decode 过程中会不断追加生成 token，最终变成：

```text
[prompt tokens] + [completion tokens]
```

---

```python
self.last_token = token_ids[-1]
```

**语法作用：** 取 token 列表最后一个元素。

**工程作用：** 保存当前请求的最后一个 token。

**推理流程意义：** Decode 阶段通常只输入上一轮的最后一个 token，再结合 KV Cache 生成下一个 token。这个字段能避免每次都从完整 token_ids 里取最后一个。

**隐含前提：** `token_ids` 不能为空，否则这里会报错。

---

```python
self.num_tokens = len(self.token_ids)
```

**语法作用：** 保存当前 token 总数。

**工程作用：** 记录上下文长度。

**推理流程意义：** 该字段会影响 position、block 数量、调度 token 数，以及是否达到上下文上限。

---

```python
self.num_prompt_tokens = len(token_ids)
```

**语法作用：** 记录原始 prompt token 数。

**工程作用：** 固定 prompt 和 completion 的分界线。

**推理流程意义：** 因为后续生成 token 会追加到 `token_ids` 后面，所以必须记录原始 prompt 长度。最终可以用 `token_ids[num_prompt_tokens:]` 取出 completion 部分。

---

```python
self.num_cached_tokens = 0
```

**语法作用：** 初始化已缓存 token 数。

**工程作用：** 表示已有多少 token 的 KV Cache 已经存在或被复用。

**推理流程意义：** Prefix caching 场景下，前缀 token 可能已经命中缓存，不需要重复 prefill。该字段记录命中进度。

**下一步结合文件：** 看 `block_manager.py` 如何判断 prefix cache 命中。

---

```python
self.num_scheduled_tokens = 0
```

**语法作用：** 初始化本轮被调度 token 数。

**工程作用：** 表示 Scheduler 当前 step 安排该请求处理多少 token。

**推理流程意义：** Prefill 阶段可能一次处理多个 token；decode 阶段通常每条请求一次处理一个 token。该字段也可用于统计 prefill throughput。

---

```python
self.is_prefill = True
```

**语法作用：** 初始化阶段标志。

**工程作用：** 新请求默认处于 prefill 相关阶段。

**推理流程意义：** Prefill 和 decode 需要传给 ModelRunner 的数据不同：prefill 需要完整 token_ids，decode 只需要 last_token 和 KV Cache。

---

```python
self.block_table = []
```

**语法作用：** 初始化空列表。

**工程作用：** 保存逻辑 block 到物理 KV Cache block 的映射。

**推理流程意义：** 这是 PagedAttention 的关键元数据。例如：

```text
block_table = [5, 8, 12]
```

表示当前 Sequence 的第 0、1、2 个逻辑 block 分别存放在物理 KV block 5、8、12 中。Attention 需要根据它找到历史 K/V。

**下一步结合文件：** 看 `block_manager.py` 如何分配 `block_table`，再看 `attention.py` 如何使用它。

---

```python
self.temperature = sampling_params.temperature
self.max_tokens = sampling_params.max_tokens
self.ignore_eos = sampling_params.ignore_eos
```

**语法作用：** 从 `sampling_params` 中读取字段并保存到实例中。

**工程作用：** 让每条请求拥有自己的采样和停止参数。

**推理流程意义：**

| 参数 | 作用 |
|---|---|
| `temperature` | 控制采样随机性 |
| `max_tokens` | 限制最多生成多少 completion token |
| `ignore_eos` | 是否忽略 EOS 结束符 |

这样同一个 batch 中不同 Sequence 可以有不同生成长度和停止策略。

---

### 3.5 列表行为

```python
def __len__(self):
    return self.num_tokens
```

**语法作用：** 定义 `len(seq)` 的行为。

**工程作用：** 让 Sequence 可以像列表一样返回长度。

**推理流程意义：** 其他模块可以直接用 `len(seq)` 表示当前上下文长度。

---

```python
def __getitem__(self, key):
    return self.token_ids[key]
```

**语法作用：** 定义索引行为，使 `seq[i]` 或 `seq[a:b]` 可用。

**工程作用：** 让 Sequence 可以像 token 列表一样访问。

**推理流程意义：** 简化上层模块访问 token 的写法。

---

### 3.6 请求是否完成

```python
@property
def is_finished(self):
    return self.status == SequenceStatus.FINISHED
```

**语法作用：** `@property` 把方法变成只读属性。

**工程作用：** 判断 Sequence 是否完成。

**推理流程意义：** Scheduler 和 LLMEngine 会用它决定是否继续调度、是否收集输出、是否释放 KV Cache。

---

### 3.7 completion token 数量

```python
@property
def num_completion_tokens(self):
    return self.num_tokens - self.num_prompt_tokens
```

**语法作用：** 动态计算属性。

**工程作用：** 计算当前已经生成了多少新 token。

**推理流程意义：** 判断是否达到 `max_tokens` 的核心依据：

```text
num_completion_tokens >= max_tokens
```

---

### 3.8 prompt_token_ids

```python
@property
def prompt_token_ids(self):
    return self.token_ids[:self.num_prompt_tokens]
```

**语法作用：** 返回 token_ids 的前半部分。

**工程作用：** 获取 prompt token。

**推理流程意义：** Prefill 阶段处理的是 prompt 部分，prefix caching 也常以 prompt block 为单位进行匹配。

---

### 3.9 completion_token_ids

```python
@property
def completion_token_ids(self):
    return self.token_ids[self.num_prompt_tokens:]
```

**语法作用：** 返回 token_ids 中 prompt 之后的部分。

**工程作用：** 获取模型生成的回答 token。

**推理流程意义：** 最终 `LLMEngine` 返回的是 completion，而不是 prompt + completion 全部内容。

---

### 3.10 num_blocks

```python
@property
def num_blocks(self):
    return (self.num_tokens + self.block_size - 1) // self.block_size
```

**语法作用：** 用整数除法实现向上取整。

**工程作用：** 计算当前请求需要多少 KV Cache block。

**推理流程意义：** KV Cache 按 block 管理，不是按请求整体连续分配。

例如：

```text
num_tokens = 600
block_size = 256
num_blocks = 3
```

---

### 3.11 last_block_num_tokens

```python
@property
def last_block_num_tokens(self):
    return self.num_tokens - (self.num_blocks - 1) * self.block_size
```

**语法作用：** 计算最后一个 block 中已有 token 数。

**工程作用：** 判断最后一个 KV block 的占用情况。

**推理流程意义：** Decode 追加新 token 时，需要知道当前最后一个 block 是否已满；满了就要申请新 block。

---

### 3.12 block(i)

```python
def block(self, i):
    assert 0 <= i < self.num_blocks
    return self.token_ids[i*self.block_size: (i+1)*self.block_size]
```

**语法作用：** 返回第 i 个 block 对应的 token 切片，并用 `assert` 检查范围。

**工程作用：** 把完整 token_ids 按 block_size 切分。

**推理流程意义：** BlockManager 可以用 `seq.block(i)` 做 block hash、prefix cache 匹配和 block 分配。

---

### 3.13 append_token

```python
def append_token(self, token_id: int):
    self.token_ids.append(token_id)
    self.last_token = token_id
    self.num_tokens += 1
```

**语法作用：** 定义追加 token 的方法，参数 `token_id` 是整数。

**工程作用：** Decode 阶段每生成一个 token，就更新 Sequence 状态。

**推理流程意义：** 自回归生成的推进过程就是不断：

```text
生成 token
  ↓
append_token
  ↓
下一轮 decode
```

这个方法同时更新：

```text
token_ids
last_token
num_tokens
```

---

### 3.14 __getstate__

```python
def __getstate__(self):
    last_state = self.last_token if not self.is_prefill else self.token_ids
    return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)
```

**语法作用：** 自定义对象被 pickle 序列化时保存哪些状态。

**工程作用：** 控制 Sequence 跨进程传输的数据量。

**推理流程意义：** ModelRunner 可能运行在独立进程中，尤其是 tensor parallel 场景。传输 Sequence 时不一定需要完整对象。

关键设计是：

| 阶段 | 传输内容 |
|---|---|
| prefill | 完整 `token_ids` |
| decode | 只传 `last_token` |

原因是：

```text
Prefill 需要处理整段 prompt
Decode 只需要当前 token + 历史 KV Cache
```

这体现了 prefill/decode 分离的推理引擎思想。

---

### 3.15 __setstate__

```python
def __setstate__(self, state):
    self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
    if isinstance(last_state, list):
        self.token_ids = last_state
        self.last_token = self.token_ids[-1]
    else:
        self.token_ids = []
        self.last_token = last_state
```

**语法作用：** 自定义对象从 pickle 状态恢复时的行为。

**工程作用：** 根据 `__getstate__()` 传来的轻量状态恢复 Sequence。

**推理流程意义：**

如果 `last_state` 是 list，说明这是 prefill 阶段，需要完整 token_ids：

```text
self.token_ids = last_state
self.last_token = self.token_ids[-1]
```

如果 `last_state` 是 int，说明这是 decode 阶段，只恢复 last_token：

```text
self.token_ids = []
self.last_token = last_state
```

这并不表示主进程中的 Sequence 丢失了完整 token_ids，而是模型执行进程中的轻量副本不再需要完整历史 token。历史上下文已经存入 KV Cache，并通过 `block_table` 访问。

---

## 4. 背后的框架性原理

### 4.1 prompt / token / tokenizer

`Sequence` 接收的不是字符串 prompt，而是 `list[int]` 形式的 token ids。说明在它之前，prompt 已经经过 tokenizer 编码。

```text
prompt(str)
  ↓ tokenizer.encode
token_ids(list[int])
  ↓ Sequence
```

模型推理系统内部真正处理的是 token id，而不是自然语言字符串。

### 4.2 request / sequence / sequence group

在本文件中，一条用户 request 被简化表示为一个 `Sequence`。

原版 vLLM 中还有更复杂的 SequenceGroup，用于管理 beam search、多候选输出等。nano-vLLM 这里没有显式实现复杂 SequenceGroup，而是用单个 Sequence 承载请求状态。

### 4.3 scheduler

Scheduler 依赖 Sequence 的状态字段做调度：

```text
status
num_tokens
num_cached_tokens
num_scheduled_tokens
num_completion_tokens
max_tokens
ignore_eos
block_table
```

Sequence 提供状态，Scheduler 负责决策和修改状态。

### 4.4 prefill / decode

本文件通过 `is_prefill` 和自定义序列化体现 prefill/decode 分离：

```text
Prefill：传完整 token_ids
Decode：传 last_token
```

这正是 LLM 推理优化的基本逻辑。

### 4.5 KV Cache

本文件不保存 K/V 张量，但保存 KV Cache 元数据：

```text
num_cached_tokens
block_table
num_blocks
last_block_num_tokens
```

这些字段告诉引擎哪些 token 已经缓存、需要多少 block、历史 KV 在哪里。

### 4.6 block / block table / block manager

`Sequence` 提供 block 管理的接口：

```text
block_size
num_blocks
last_block_num_tokens
block(i)
block_table
```

BlockManager 根据这些字段分配、复用和释放 KV Cache block。

### 4.7 attention

Attention 不在本文件中计算，但 decode attention 需要通过 block_table 找到历史 K/V。因此 Sequence 间接为 Attention 提供索引元数据。

### 4.8 model runner

ModelRunner 会接收 Sequence 列表，准备模型输入。`__getstate__` 和 `__setstate__` 说明 Sequence 可能跨进程传输，并且在 prefill/decode 阶段传输的数据不同。

### 4.9 GPU 执行

本文件没有 PyTorch Tensor 或 CUDA 操作。它主要保存 CPU 侧元数据。GPU 执行发生在 ModelRunner、Attention、Sampler 等模块中。

### 4.10 tensor parallel

本文件不直接实现 tensor parallel，但自定义序列化方法对多进程通信友好。tensor parallel 场景下，多个 ModelRunner 进程需要接收 Sequence 的轻量状态。

### 4.11 logits / sampling / temperature

本文件不计算 logits，也不执行 sampling，但保存了 `temperature`，用于后续 Sampler 采样。

```text
Sequence.temperature
  ↓
ModelRunner / Sampler
  ↓
logits / temperature
  ↓
sample next token
```

### 4.12 top_p / top_k

本文件没有保存 top_p / top_k。说明 nano-vLLM 当前采样参数是简化版，主要支持 temperature、max_tokens、ignore_eos。

### 4.13 throughput / latency / TTFT / TPOT

本文件不直接统计性能，但提供统计基础：

```text
num_prompt_tokens
num_completion_tokens
num_scheduled_tokens
```

例如 prefill throughput 可以根据本轮 `num_scheduled_tokens` 统计，decode throughput 可以根据生成 token 数统计。

---

## 5. 和 vLLM 原版设计的关系

### 5.1 连续批处理

每条请求独立保存状态，Scheduler 可以动态选择哪些 Sequence 进入本轮 batch。这是 continuous batching 的基础。

### 5.2 PagedAttention

本文件中的：

```text
block_size
block_table
num_blocks
block(i)
```

是 PagedAttention 的上层数据结构基础。它们让请求的 KV Cache 可以按 block 分页存储，而不是整段连续存储。

### 5.3 KV Cache block 管理

Sequence 不直接分配 KV Cache，但提供 BlockManager 需要的信息：

```text
当前需要多少 block
最后一个 block 是否满
每个逻辑 block 对应哪些 token
物理 block 映射表
```

### 5.4 request / sequence 调度

原版 vLLM 的请求管理更复杂，而 nano-vLLM 将请求状态压缩为 `Sequence` + `SequenceStatus`，更适合学习。

### 5.5 prefill / decode 分离

本文件非常直接地体现了 prefill/decode 分离：

```python
last_state = self.last_token if not self.is_prefill else self.token_ids
```

这说明 decode 阶段不再传完整历史 token，而是依赖 KV Cache。

### 5.6 高吞吐推理服务

高吞吐依赖：

```text
多请求并发
请求状态独立维护
KV Cache block 复用
减少跨进程通信
prefill/decode 分离
```

`Sequence` 正是这些机制的基础状态对象。

### 5.7 nano-vLLM 的简化点

| 原版 vLLM | nano-vLLM 中的简化 |
|---|---|
| 复杂 SequenceGroup | 这里只定义单个 Sequence |
| beam search / 多候选 | 本文件没有体现 |
| 复杂调度优先级 | 这里只保存基础状态 |
| 完整采样参数 | 这里只保存 temperature、max_tokens、ignore_eos |
| 生产级请求管理 | 更偏向源码教学和离线推理 |

---

## 6. 总结

`sequence.py` 是 nano-vLLM 中非常基础但非常关键的文件。它不是计算核心，却是调度、KV Cache、模型执行之间共享请求状态的核心数据结构。

核心主线是：

```text
prompt token ids
  ↓
Sequence
  ↓
WAITING
  ↓ prefill
RUNNING
  ↓ append_token 逐 token decode
FINISHED
  ↓ completion_token_ids
最终输出
```

这个文件最值得掌握的内容是：

```text
1. Sequence 如何表示一条请求
2. token_ids 如何同时保存 prompt 和 completion
3. num_prompt_tokens 如何划分 prompt / completion
4. block_table 如何服务 KV Cache 分页管理
5. append_token 如何推进逐 token 生成
6. __getstate__ / __setstate__ 如何体现 prefill/decode 分离
```

下一步建议阅读：

```text
scheduler.py
  看 Sequence 状态如何流转

block_manager.py
  看 block_table 如何分配和复用

model_runner.py
  看 prefill/decode 如何根据 Sequence 准备模型输入

attention.py
  看 block_table 如何参与 KV Cache 读写
```


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]
- 关联阅读：[[outputs/项目整理/专题-01-请求从输入到输出|专题-01-请求从输入到输出]]

%% 项目关联导航：结束 %%
