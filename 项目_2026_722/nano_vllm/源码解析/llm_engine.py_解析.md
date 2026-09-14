# llm_engine.py 源码解析

> 文件：`llm_engine.py`  
> 定位：nano-vLLM 推理引擎主控层 / 用户 API 到底层调度执行系统的桥梁  
> 关键词：LLMEngine、generate、request、Sequence、Scheduler、ModelRunner、Prefill、Decode、Sampling、Throughput、Tensor Parallel

---

## 1. 文件整体定位

`llm_engine.py` 是 nano-vLLM 中非常核心的文件。它不是模型结构文件，也不是底层 CUDA / Attention 文件，而是整个推理引擎的“总控制器”。

它负责把用户侧的调用：

```python
outputs = llm.generate(prompts, sampling_params)
```

转换成推理引擎内部的一整套流程：

```text
prompts
  ↓ tokenizer.encode
Sequence 请求对象
  ↓ Scheduler.add
waiting 队列
  ↓ Scheduler.schedule
Prefill / Decode 调度
  ↓ ModelRunner.run
模型前向 + KV Cache + 采样
  ↓ Scheduler.postprocess
更新 Sequence 状态
  ↓ tokenizer.decode
返回文本结果
```

所以这个文件属于：

```text
推理引擎主控层 / API 编排层 / 请求生命周期管理层
```

它在项目中的位置可以理解为：

```text
example.py / bench.py
  ↓
nanovllm.llm.LLM
  ↓
LLMEngine
  ↓
Scheduler + Sequence + ModelRunner
  ↓
BlockManager + Qwen3 Model + Attention + Sampler
```

### 它和哪些文件有关

| 相关文件 | 关系 |
|---|---|
| `nanovllm/llm.py` | `LLM` 类通常继承或包装 `LLMEngine`，对外暴露用户 API |
| `nanovllm/config.py` | `LLMEngine` 用 `Config` 统一管理模型路径、batch 上限、KV Cache block size、TP 大小等配置 |
| `nanovllm/sampling_params.py` | 每个请求的生成参数，例如 temperature、max_tokens、ignore_eos |
| `nanovllm/engine/sequence.py` | 把一个 prompt 包装成一个内部请求对象 `Sequence` |
| `nanovllm/engine/scheduler.py` | 负责调度 waiting / running 请求，决定本轮做 prefill 还是 decode |
| `nanovllm/engine/model_runner.py` | 真正执行模型前向、KV Cache 读写和采样 |
| `nanovllm/engine/block_manager.py` | 间接相关，由 Scheduler 使用，用于 KV Cache block 管理 |
| `nanovllm/layers/attention.py` | 间接相关，由 ModelRunner 调用模型前向时使用，完成 attention 和 KV Cache 访问 |

### 它在完整推理流程中的位置

```text
用户输入 prompt
  ↓
example.py / bench.py 调用 LLM.generate
  ↓
LLMEngine.generate
  ↓
LLMEngine.add_request
  ↓
Sequence(prompt, sampling_params)
  ↓
Scheduler.add
  ↓
while not finished:
      LLMEngine.step
        ↓
      Scheduler.schedule
        ↓
      ModelRunner.run
        ↓
      Scheduler.postprocess
  ↓
tokenizer.decode
  ↓
返回 output text
```

`llm_engine.py` 的核心作用可以总结为一句话：

> 它把“用户的一次 generate 调用”拆解成“多个 Sequence 请求的调度、执行、后处理和结果恢复”。

---

## 2. 代码结构总览

源码如下：

```python
import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        ...

    def exit(self):
        ...

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        ...

    def step(self):
        ...

    def is_finished(self):
        ...

    def generate(...):
        ...
```

### 2.1 导入模块

| 导入内容 | 作用 |
|---|---|
| `atexit` | 注册程序退出时自动执行的清理函数，防止多进程资源残留 |
| `fields` | 从 dataclass 类型中读取字段名，用于过滤 `Config` 支持的参数 |
| `perf_counter` | 高精度计时，用于统计 prefill / decode 吞吐量 |
| `tqdm` | 显示生成进度条和吞吐量 |
| `AutoTokenizer` | 加载 HuggingFace tokenizer，负责文本和 token id 的转换 |
| `torch.multiprocessing as mp` | 用于 Tensor Parallel 多进程启动 |
| `Config` | nano-vLLM 的全局配置类 |
| `SamplingParams` | 采样参数类 |
| `Sequence` | 请求对象，一个 prompt 对应一个 Sequence |
| `Scheduler` | 请求调度器，负责 prefill / decode 调度 |
| `ModelRunner` | 模型执行器，负责模型前向、KV Cache 和采样 |

### 2.2 定义的类

本文件只定义了一个类：

```python
class LLMEngine:
```

它是推理引擎的主控类。

### 2.3 定义的方法

| 方法 | 作用 |
|---|---|
| `__init__` | 初始化配置、Sequence block size、多进程 ModelRunner、tokenizer、Scheduler |
| `exit` | 退出时通知 ModelRunner 退出，并等待子进程结束 |
| `add_request` | 把 prompt 转成 token ids，再封装成 Sequence 加入 Scheduler |
| `step` | 执行一轮调度 + 模型运行 + 后处理 |
| `is_finished` | 判断所有请求是否完成 |
| `generate` | 对外暴露的完整生成接口，负责添加请求、循环 step、收集结果并 decode |

### 2.4 重要数据结构

| 变量 | 含义 |
|---|---|
| `config` | 引擎配置对象，管理模型路径、batch 限制、KV Cache block size 等 |
| `self.ps` | Tensor Parallel 子进程列表 |
| `self.events` | 主进程和子进程同步用的事件对象 |
| `self.model_runner` | rank 0 的模型执行器，主进程直接持有 |
| `self.tokenizer` | HuggingFace tokenizer，用于 encode / decode |
| `self.scheduler` | 请求调度器 |
| `seqs` | 当前 step 被调度出来的一批 Sequence |
| `is_prefill` | 本轮是否是 prefill 阶段 |
| `outputs` | 已完成请求的输出结果缓存 |

### 2.5 文件组织结构图

```text
llm_engine.py
├── import 部分
│   ├── Python 标准库：atexit / dataclasses / time
│   ├── 第三方库：tqdm / transformers / torch.multiprocessing
│   └── nano-vLLM 内部模块：Config / SamplingParams / Sequence / Scheduler / ModelRunner
│
└── class LLMEngine
    ├── __init__
    │   ├── 解析 Config 参数
    │   ├── 设置 Sequence.block_size
    │   ├── 启动 Tensor Parallel 子进程
    │   ├── 创建 rank 0 ModelRunner
    │   ├── 加载 tokenizer
    │   ├── 设置 eos token
    │   └── 创建 Scheduler
    │
    ├── exit
    │   └── 清理 ModelRunner 和子进程
    │
    ├── add_request
    │   ├── prompt 字符串编码为 token ids
    │   ├── 创建 Sequence
    │   └── 加入 Scheduler
    │
    ├── step
    │   ├── Scheduler.schedule
    │   ├── ModelRunner.run
    │   ├── Scheduler.postprocess
    │   └── 返回已完成请求
    │
    ├── is_finished
    │   └── 调用 Scheduler.is_finished
    │
    └── generate
        ├── 创建进度条
        ├── 统一 sampling_params
        ├── 添加所有请求
        ├── 循环 step
        ├── 统计吞吐量
        ├── 收集输出 token ids
        └── tokenizer.decode 得到最终文本
```

---

## 3. 逐行代码解释

### 3.1 导入退出清理模块

```python
import atexit
```

#### 语法作用

导入 Python 标准库 `atexit`。

#### 工程作用

`atexit` 可以注册程序退出时自动调用的函数。本文件后面有：

```python
atexit.register(self.exit)
```

意思是当 Python 程序正常退出时，会自动调用 `self.exit()` 清理资源。

#### 在 nano-vLLM 中的意义

nano-VLLM 可能会启动多个进程用于 Tensor Parallel。如果程序退出时不清理这些进程，可能造成：

```text
子进程残留
GPU 显存未释放
NCCL 通信资源未释放
程序卡死或僵尸进程
```

所以 `atexit` 是推理引擎资源管理的一部分。

---

### 3.2 导入 dataclass 字段读取工具

```python
from dataclasses import fields
```

#### 语法作用

从 Python `dataclasses` 模块中导入 `fields` 函数。

#### 工程作用

`fields(Config)` 可以读取 `Config` 这个 dataclass 中定义的所有字段，例如：

```text
model
max_num_batched_tokens
max_num_seqs
max_model_len
gpu_memory_utilization
tensor_parallel_size
enforce_eager
kvcache_block_size
...
```

后面代码会用它过滤 `kwargs`：

```python
config_fields = {field.name for field in fields(Config)}
config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
```

#### 在推理流程中的意义

用户初始化 LLM 时可能写：

```python
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
```

这些额外参数通过 `**kwargs` 传入。`fields(Config)` 的作用是只保留 `Config` 支持的参数，避免无关参数污染配置。

---

### 3.3 导入高精度计时器

```python
from time import perf_counter
```

#### 语法作用

导入高精度计时函数 `perf_counter`。

#### 工程作用

`perf_counter()` 常用于性能测量，精度比普通 `time.time()` 更适合统计代码片段耗时。

本文件用它统计每一次 `step()` 的耗时，从而计算：

```text
Prefill 吞吐量
Decode 吞吐量
```

#### 在推理流程中的意义

推理引擎非常关心性能指标：

```text
Prefill tokens/s
Decode tokens/s
TTFT
TPOT
总吞吐量
```

本文件虽然没有完整统计 TTFT / TPOT，但已经通过 `perf_counter()` 体现了推理引擎对吞吐量的关注。

---

### 3.4 导入 tqdm 进度条

```python
from tqdm.auto import tqdm
```

#### 语法作用

导入 `tqdm` 进度条工具。

#### 工程作用

用于在 `generate()` 过程中显示当前已经完成多少条请求，以及当前 prefill / decode 吞吐量。

#### 在 nano-vLLM 中的意义

在离线批量推理中，用户通常希望看到：

```text
当前完成多少请求
Prefill 速度多少 tok/s
Decode 速度多少 tok/s
```

这对 benchmark 和调试很有帮助。

---

### 3.5 导入 tokenizer

```python
from transformers import AutoTokenizer
```

#### 语法作用

从 HuggingFace Transformers 导入自动 tokenizer 加载类。

#### 工程作用

`AutoTokenizer.from_pretrained(config.model)` 会根据模型目录自动加载对应 tokenizer。

#### 在推理流程中的意义

文本模型推理的基本流程是：

```text
字符串 prompt
  ↓ tokenizer.encode
input token ids
  ↓ model forward
output token ids
  ↓ tokenizer.decode
输出文本
```

`LLMEngine` 同时负责 encode 和 decode，所以它必须持有 tokenizer。

---

### 3.6 导入 PyTorch 多进程模块

```python
import torch.multiprocessing as mp
```

#### 语法作用

导入 PyTorch 的 multiprocessing 模块，并命名为 `mp`。

#### 工程作用

用于启动多个 `ModelRunner` 进程。每个进程可以绑定到不同 GPU，用于 Tensor Parallel。

#### 在 nano-vLLM 中的意义

当：

```python
tensor_parallel_size > 1
```

时，模型权重和计算会分布在多张 GPU 上。每张 GPU 通常对应一个 rank 和一个进程。

本文件中的逻辑是：

```text
rank 0：主进程中创建 ModelRunner
rank 1 ~ TP-1：通过 multiprocessing 创建子进程 ModelRunner
```

---

### 3.7 导入 nano-vLLM 内部模块

```python
from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
```

#### 工程作用

这些是 `LLMEngine` 依赖的核心组件：

```text
Config：引擎配置
SamplingParams：生成参数
Sequence：请求对象
Scheduler：调度器
ModelRunner：模型执行器
```

#### 架构意义

这几行基本暴露了 nano-VLLM 的核心分层：

```text
LLMEngine：主控
  ├── Config：配置
  ├── Sequence：请求状态
  ├── Scheduler：调度策略
  └── ModelRunner：GPU 执行
```

这也是你阅读源码时最重要的主线。

---

## 3.8 定义 LLMEngine 类

```python
class LLMEngine:
```

#### 语法作用

定义一个 Python 类 `LLMEngine`。

#### 工程作用

`LLMEngine` 是推理引擎主类，负责管理完整请求生命周期。

#### 在 nano-vLLM 中的意义

它对应 vLLM 里的 Engine 层，负责把外部请求转化为内部调度执行。

你可以把它理解为：

```text
LLMEngine = 推理系统的中央调度控制器
```

不过要注意：它自己不直接做底层调度算法，也不直接执行 attention。它主要负责“编排”：

```text
配置初始化
请求接收
调度触发
模型执行触发
结果收集
资源清理
```

---

## 3.9 初始化方法 `__init__`

```python
def __init__(self, model, **kwargs):
```

#### 语法作用

定义类的初始化方法。

参数：

| 参数 | 含义 |
|---|---|
| `self` | 当前对象本身 |
| `model` | 模型路径 |
| `**kwargs` | 额外配置参数，例如 `enforce_eager=True`、`max_model_len=4096` |

#### 工程作用

当用户写：

```python
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
```

最终会进入这个初始化方法。

---

### 3.9.1 读取 Config 支持的字段

```python
config_fields = {field.name for field in fields(Config)}
```

#### 语法作用

这是一个集合推导式。

它会遍历 `fields(Config)` 返回的所有 dataclass 字段，然后取出字段名，形成一个集合。

结果类似：

```python
{
    "model",
    "max_num_batched_tokens",
    "max_num_seqs",
    "max_model_len",
    "gpu_memory_utilization",
    "tensor_parallel_size",
    "enforce_eager",
    "hf_config",
    "eos",
    "kvcache_block_size",
    "num_kvcache_blocks",
}
```

#### 工程作用

这一步是为了知道哪些参数可以传给 `Config`。

#### 为什么这样写

因为 `LLMEngine.__init__` 接收的是：

```python
**kwargs
```

用户可能传入很多参数，但并不一定都是 `Config` 支持的。先拿到合法字段集合，可以避免乱传参数。

---

### 3.9.2 过滤 kwargs

```python
config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
```

#### 语法作用

这是字典推导式。

它遍历用户传入的 `kwargs`，只保留 key 在 `config_fields` 中的参数。

例如用户调用：

```python
LLM(path, enforce_eager=True, tensor_parallel_size=1, unknown_arg=123)
```

那么过滤后：

```python
config_kwargs = {
    "enforce_eager": True,
    "tensor_parallel_size": 1,
}
```

`unknown_arg` 会被忽略。

#### 工程作用

这让 `LLMEngine` 初始化更稳健，避免 `Config(...)` 因未知参数报错。

#### 在推理系统中的意义

推理引擎通常有大量配置项，例如：

```text
最大上下文长度
最大 batch token 数
KV Cache block size
GPU 显存使用比例
Tensor Parallel 大小
是否启用 CUDA Graph
```

这些配置最终都会影响调度策略和显存管理。

---

### 3.9.3 创建 Config 对象

```python
config = Config(model, **config_kwargs)
```

#### 语法作用

实例化 `Config`。

`model` 作为第一个参数传入，其余配置通过 `**config_kwargs` 展开。

例如：

```python
Config(
    model="~/huggingface/Qwen3-0.6B/",
    enforce_eager=True,
    tensor_parallel_size=1,
)
```

#### 工程作用

创建统一配置对象。

`Config.__post_init__()` 中通常会做：

```text
检查模型路径是否存在
检查 kvcache_block_size 是否符合要求
检查 tensor_parallel_size 是否合理
加载 HuggingFace AutoConfig
限制 max_model_len 不超过模型最大位置编码长度
```

#### 在推理流程中的意义

`config` 后面会传给：

```text
ModelRunner
Scheduler
```

所以它是整个推理系统的全局配置来源。

---

### 3.9.4 设置 Sequence 的 block size

```python
Sequence.block_size = config.kvcache_block_size
```

#### 语法作用

修改 `Sequence` 类的类属性 `block_size`。

注意这不是给某一个 `Sequence` 对象设置属性，而是给整个 `Sequence` 类设置属性。

#### 工程作用

让所有 Sequence 都使用同一个 KV Cache block size。

例如：

```python
config.kvcache_block_size = 256
```

那么每个 Sequence 在切分 token blocks 时都使用 256：

```text
block 0: token 0 ~ 255
block 1: token 256 ~ 511
block 2: token 512 ~ 767
```

#### 在 nano-vLLM 推理流程中的意义

`Sequence` 需要知道 block size，才能维护自己的逻辑 block：

```text
Sequence token_ids
  ↓ 按 block_size 切分
逻辑 block 0 / 1 / 2 / ...
  ↓ BlockManager 分配
物理 KV Cache block id
  ↓ block_table 记录映射关系
```

这一步是连接 `Config` 和 `Sequence / BlockManager / Attention` 的关键。

---

### 3.9.5 初始化子进程列表和同步事件列表

```python
self.ps = []
self.events = []
```

#### 语法作用

定义两个实例属性：

```text
self.ps：process list
self.events：event list
```

#### 工程作用

它们用于 Tensor Parallel 多进程管理。

| 属性 | 含义 |
|---|---|
| `self.ps` | 保存 rank 1 到 rank N 的子进程对象 |
| `self.events` | 保存主进程和子进程之间的同步事件 |

#### 在推理流程中的意义

如果 `tensor_parallel_size=1`，这两个列表为空。

如果 `tensor_parallel_size>1`，则每个额外 rank 都会启动一个独立进程。

---

### 3.9.6 获取 spawn 上下文

```python
ctx = mp.get_context("spawn")
```

#### 语法作用

从 PyTorch multiprocessing 获取 `spawn` 启动上下文。

#### 工程作用

`spawn` 表示创建新进程时，会重新启动一个 Python 解释器，然后导入相关模块并执行目标函数。

#### 为什么不用 fork

GPU / CUDA / NCCL 场景下，`fork` 可能导致 CUDA 上下文继承异常。`spawn` 更安全，更适合多 GPU 推理。

#### 在 Tensor Parallel 中的意义

多 GPU 推理通常采用：

```text
一个 rank 一个进程
每个进程绑定一张 GPU
通过 NCCL 通信
```

所以这里使用 multiprocessing 启动子进程。

---

### 3.9.7 启动 Tensor Parallel 子进程

```python
for i in range(1, config.tensor_parallel_size):
    event = ctx.Event()
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()
    self.ps.append(process)
    self.events.append(event)
```

#### 语法作用

这是一个循环，从 rank 1 开始启动子进程。

如果：

```python
config.tensor_parallel_size = 1
```

那么：

```python
range(1, 1)
```

为空，不会启动子进程。

如果：

```python
config.tensor_parallel_size = 4
```

则会启动 rank 1、rank 2、rank 3 三个子进程，rank 0 留给主进程。

#### 工程作用

每个子进程都运行一个 `ModelRunner`：

```python
ctx.Process(target=ModelRunner, args=(config, i, event))
```

这里 `target=ModelRunner` 的含义是把 `ModelRunner` 类当作进程入口调用。

每个子进程参数：

| 参数 | 含义 |
|---|---|
| `config` | 全局配置 |
| `i` | 当前 rank 编号 |
| `event` | 同步事件 |

#### 在推理流程中的意义

Tensor Parallel 的大致工作方式是：

```text
rank 0 主进程：接收调度结果，发起 run 调用，负责部分模型计算和采样协调
rank 1~N 子进程：执行自己 rank 上的模型分片计算
```

这个文件不展开具体 TP 通信细节，真正逻辑在：

```text
model_runner.py
linear.py
embed_head.py
```

继续阅读建议：看 `model_runner.py` 中如何初始化 NCCL、绑定 GPU、加载模型分片和处理 `call("run")`。

---

### 3.9.8 创建 rank 0 ModelRunner

```python
self.model_runner = ModelRunner(config, 0, self.events)
```

#### 语法作用

创建一个 `ModelRunner` 实例，rank 为 0。

#### 工程作用

主进程本身负责 rank 0 的模型执行器。

这里和前面的子进程配合：

```text
rank 0：主进程中的 self.model_runner
rank 1~N：multiprocessing 子进程中的 ModelRunner
```

#### 在推理流程中的意义

后续每次执行模型时，`LLMEngine.step()` 会调用：

```python
token_ids = self.model_runner.call("run", seqs, is_prefill)
```

也就是说，`LLMEngine` 不直接 forward 模型，而是通过 `ModelRunner` 统一执行。

`ModelRunner` 负责：

```text
准备 input_ids / positions / slot_mapping / block_tables
设置 attention context
执行 Qwen3ForCausalLM.forward
执行 sampler
返回 token_ids
```

---

### 3.9.9 加载 tokenizer

```python
self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
```

#### 语法作用

通过 HuggingFace 加载 tokenizer。

`use_fast=True` 表示优先使用 Rust 实现的 fast tokenizer。

#### 工程作用

`LLMEngine` 需要 tokenizer 做两件事：

```text
1. add_request 中把字符串 prompt 编码为 token ids
2. generate 结束后把 completion token ids 解码为文本
```

#### 在推理流程中的意义

模型内部处理的是 token ids，而用户看到的是文本，所以 `LLMEngine` 必须负责文本和 token 的边界转换。

---

### 3.9.10 设置 EOS token

```python
config.eos = self.tokenizer.eos_token_id
```

#### 语法作用

把 tokenizer 的 EOS token id 写入 config。

#### 工程作用

EOS 是 End Of Sequence 的缩写，表示生成结束 token。

例如模型生成到：

```text
<|endoftext|>
```

或者某个 chat 模型的结束标记时，应该停止继续 decode。

#### 在推理流程中的意义

Scheduler / Sequence 后处理阶段需要根据 EOS 判断请求是否完成。

典型逻辑是：

```text
如果新生成 token == eos，并且 ignore_eos=False：
    Sequence 标记为 FINISHED
```

所以这行代码把 tokenizer 层的信息传给调度层。

---

### 3.9.11 创建 Scheduler

```python
self.scheduler = Scheduler(config)
```

#### 语法作用

创建调度器实例。

#### 工程作用

Scheduler 负责维护请求队列和调度策略。

它通常会管理：

```text
waiting 队列：还没有完成 prefill 的请求
running 队列：已经完成 prefill，正在 decode 的请求
finished 请求：已经生成结束的请求
BlockManager：KV Cache block 分配与释放
```

#### 在推理流程中的意义

`LLMEngine` 只负责调用：

```python
self.scheduler.add(seq)
self.scheduler.schedule()
self.scheduler.postprocess(...)
self.scheduler.is_finished()
```

真正的请求调度策略在 `scheduler.py`。

继续阅读建议：下一步重点看 `scheduler.py` 的 `schedule()` 和 `postprocess()`。

---

### 3.9.12 注册退出清理函数

```python
atexit.register(self.exit)
```

#### 语法作用

把 `self.exit` 注册为程序退出时自动调用的函数。

#### 工程作用

确保即使用户没有手动调用 `llm.exit()`，程序退出时也会尝试清理资源。

#### 在推理系统中的意义

多进程推理系统必须关注资源释放，否则可能出现：

```text
进程无法退出
GPU 显存未释放
NCCL 资源残留
下一次运行失败
```

---

## 3.10 退出清理方法 `exit`

```python
def exit(self):
    self.model_runner.call("exit")
    del self.model_runner
    for p in self.ps:
        p.join()
```

### 语法作用

定义实例方法 `exit()`。

### 逐行解释

#### 通知 ModelRunner 退出

```python
self.model_runner.call("exit")
```

这行调用 `ModelRunner` 的 `call` 方法，并传入命令字符串 `"exit"`。

从设计上看，`ModelRunner.call()` 应该类似一个命令分发接口：

```text
call("run", ...)：执行模型推理
call("exit")：通知模型执行器退出
```

对于 Tensor Parallel，rank 0 的 ModelRunner 可能还会通知其他 rank 的子进程退出。

#### 删除 rank 0 ModelRunner

```python
del self.model_runner
```

删除对象引用，触发 Python 回收。

这有助于释放模型权重、KV Cache 等 GPU 显存资源。

#### 等待子进程结束

```python
for p in self.ps:
    p.join()
```

`join()` 会阻塞等待子进程退出。

### 工程意义

这是典型的多进程清理流程：

```text
发送退出信号
删除本地执行器
等待所有子进程退出
```

### 注意点

如果 `exit()` 被重复调用，可能存在 `self.model_runner` 已被删除的问题。生产级系统通常需要更复杂的幂等保护，例如：

```python
if hasattr(self, "model_runner"):
    ...
```

nano-VLLM 为了简洁，做了简化。

---

## 3.11 添加请求 `add_request`

```python
def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
```

### 语法作用

定义添加请求的方法。

类型注解：

```python
prompt: str | list[int]
```

表示 prompt 可以是字符串，也可以是 token id 列表。

```python
sampling_params: SamplingParams
```

表示采样参数必须是 `SamplingParams` 对象。

### 工程作用

把用户输入的 prompt 转换成内部 `Sequence`，并加入调度器。

---

### 3.11.1 判断 prompt 类型

```python
if isinstance(prompt, str):
    prompt = self.tokenizer.encode(prompt)
```

#### 语法作用

判断 `prompt` 是否是字符串。如果是字符串，就调用 tokenizer 编码成 token ids。

#### 工程作用

支持两种输入形式：

```text
1. 文本字符串：需要 tokenizer.encode
2. 已经编码好的 token id list：直接使用
```

这也是为什么 `bench.py` 可以直接构造随机 token ids 做压测。

#### 在推理流程中的意义

在推理引擎内部，`Sequence` 管理的是 token ids，而不是原始字符串。

所以这里完成了：

```text
用户文本输入
  ↓
tokenizer.encode
  ↓
prompt token ids
```

---

### 3.11.2 创建 Sequence

```python
seq = Sequence(prompt, sampling_params)
```

#### 语法作用

实例化 `Sequence` 对象。

这里的 `prompt` 已经是：

```python
list[int]
```

#### 工程作用

把一个请求封装成推理引擎内部状态对象。

一个 Sequence 通常会维护：

```text
seq_id
status
token_ids
prompt token 数量
completion token ids
num_cached_tokens
num_scheduled_tokens
block_table
sampling params
```

#### 在 nano-VLLM 中的意义

`Sequence` 是 request 生命周期的核心对象。

用户看到的是 prompt；引擎看到的是 Sequence。

```text
prompt + sampling_params
  ↓
Sequence
  ↓
Scheduler 统一管理
```

继续阅读建议：看 `sequence.py` 中 `Sequence.__init__`、`is_finished`、`completion_token_ids`、`block()` 等字段和方法。

---

### 3.11.3 加入 Scheduler

```python
self.scheduler.add(seq)
```

#### 语法作用

调用调度器的 `add()` 方法。

#### 工程作用

把新请求加入 Scheduler 的 waiting 队列。

#### 在推理流程中的意义

一个新请求通常还没有计算 prompt 的 KV Cache，所以它最开始应该处于：

```text
WAITING 状态
```

然后等待 `Scheduler.schedule()` 安排 prefill。

---

## 3.12 执行一轮推理 `step`

```python
def step(self):
```

### 工程作用

`step()` 是 `LLMEngine` 中最核心的方法之一。

它表示执行“一轮”推理调度和模型运行。

一轮 step 不是完整生成，而是：

```text
调度一批请求
  ↓
执行一次模型 forward
  ↓
采样得到新 token
  ↓
更新请求状态
```

完整生成需要在 `generate()` 中反复调用 `step()`。

---

### 3.12.1 调度请求

```python
seqs, is_prefill = self.scheduler.schedule()
```

#### 语法作用

调用 Scheduler 的 `schedule()` 方法，返回两个值：

| 返回值 | 含义 |
|---|---|
| `seqs` | 本轮被调度出来的一批 Sequence |
| `is_prefill` | 本轮是否是 prefill 阶段 |

#### 工程作用

Scheduler 决定本轮模型应该处理哪些请求。

它会考虑：

```text
waiting 队列里有没有新请求
running 队列里有哪些请求需要 decode
KV Cache block 是否足够
max_num_batched_tokens 是否超限
max_num_seqs 是否超限
```

#### 在推理流程中的意义

这是推理引擎区别于普通 `model.generate()` 的核心。

普通 PyTorch 生成通常是固定 batch；而推理引擎要做动态调度：

```text
请求不断进入
请求长度不同
请求生成速度不同
请求结束时间不同
KV Cache 显存有限
```

所以需要 Scheduler。

继续阅读建议：重点看 `scheduler.py` 的 `schedule()`。

---

### 3.12.2 计算本轮 token 数

```python
num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
```

#### 语法作用

这是一个条件表达式：

```python
A if condition else B
```

如果本轮是 prefill：

```python
num_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
```

如果本轮是 decode：

```python
num_tokens = -len(seqs)
```

#### 工程作用

这个变量用于后面统计吞吐量。

为什么 prefill 是正数？

```text
Prefill 一次可能处理很多 prompt tokens
```

例如某一轮调度了 3 个请求：

```text
seq0 prefill 100 tokens
seq1 prefill 200 tokens
seq2 prefill 50 tokens
```

那么本轮 prefill token 数：

```text
350 tokens
```

为什么 decode 是负数？

```python
-len(seqs)
```

因为 decode 阶段每个 sequence 通常只生成 1 个 token，所以本轮 decode token 数就是被调度的 sequence 数。

用负数是为了在 `generate()` 里区分 prefill 和 decode：

```python
if num_tokens > 0:
    prefill_throughput = ...
else:
    decode_throughput = -num_tokens / ...
```

#### 在推理性能中的意义

Prefill 和 Decode 性能差异很大：

```text
Prefill：大矩阵计算，处理 prompt，全量 attention，吞吐量通常高
Decode：逐 token 生成，受内存访问和 KV Cache 读取影响，单步小 batch 时 GPU 利用率低
```

所以分别统计 prefill / decode throughput 是合理的。

---

### 3.12.3 执行模型

```python
token_ids = self.model_runner.call("run", seqs, is_prefill)
```

#### 语法作用

调用 `ModelRunner.call()`，命令是 `"run"`，参数是当前调度出来的 `seqs` 和 `is_prefill`。

#### 工程作用

这一步会真正触发模型执行。

内部通常包括：

```text
prepare_prefill 或 prepare_decode
设置 input_ids
设置 positions
设置 slot_mapping
设置 block_tables
设置 attention context
执行模型 forward
得到 logits
执行 sampler
返回本轮生成的 token ids
```

#### 在推理流程中的意义

`LLMEngine` 本身不直接接触 PyTorch Tensor。

它把底层执行全部委托给 `ModelRunner`：

```text
LLMEngine：管流程
ModelRunner：管 GPU 执行
Scheduler：管请求选择
BlockManager：管 KV Cache block
```

这种分层让代码更清晰。

继续阅读建议：看 `model_runner.py` 的 `run()`、`prepare_prefill()`、`prepare_decode()`、`run_model()`。

---

### 3.12.4 后处理请求状态

```python
self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

#### 语法作用

调用调度器的后处理方法。

输入包括：

| 参数 | 含义 |
|---|---|
| `seqs` | 本轮执行的请求 |
| `token_ids` | 模型采样得到的新 token |
| `is_prefill` | 本轮是否是 prefill |

#### 工程作用

后处理可能会做：

```text
更新 Sequence 的 token_ids
更新 num_cached_tokens
更新 num_scheduled_tokens
判断是否生成 EOS
判断是否达到 max_tokens
把完成 prefill 的请求从 waiting 移到 running
把完成生成的请求标记为 finished
释放完成请求的 KV Cache block
```

#### 在推理流程中的意义

模型 forward 只是算出下一个 token；推理系统还必须维护请求状态。

所以生成系统的核心不是单纯 forward，而是：

```text
forward + 状态更新 + 资源管理 + 调度循环
```

---

### 3.12.5 收集已完成输出

```python
outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
```

#### 语法作用

这是列表推导式。

它遍历本轮调度的 `seqs`，只保留已经完成的请求。

每个输出是一个二元组：

```python
(seq.seq_id, seq.completion_token_ids)
```

#### 工程作用

把本轮刚刚完成的请求结果返回给 `generate()`。

为什么只返回 completion token ids？

因为用户最终只关心模型新生成的部分，而不是 prompt 本身。

#### 在推理流程中的意义

不同 sequence 会在不同 step 完成。

例如：

```text
seq0 生成 20 token 后完成
seq1 生成 200 token 后完成
seq2 生成 50 token 后完成
```

所以 `generate()` 必须逐步收集完成的请求。

---

### 3.12.6 返回结果

```python
return outputs, num_tokens
```

#### 工程作用

返回两个信息：

| 返回值 | 用途 |
|---|---|
| `outputs` | 本轮完成的请求输出 |
| `num_tokens` | 本轮处理 token 数，用于吞吐量统计 |

---

## 3.13 判断是否完成 `is_finished`

```python
def is_finished(self):
    return self.scheduler.is_finished()
```

### 语法作用

定义方法，直接返回调度器的完成状态。

### 工程作用

`LLMEngine` 不自己维护 waiting / running 队列，所以是否全部完成要问 Scheduler。

### 在推理流程中的意义

在 `generate()` 中有：

```python
while not self.is_finished():
    ...
```

这表示只要还有请求没完成，就继续调度和执行。

---

## 3.14 完整生成接口 `generate`

```python
def generate(
    self,
    prompts: list[str] | list[list[int]],
    sampling_params: SamplingParams | list[SamplingParams],
    use_tqdm: bool = True,
) -> list[str]:
```

### 语法作用

定义对外的完整生成接口。

类型注解：

| 参数 | 类型 | 含义 |
|---|---|---|
| `prompts` | `list[str] | list[list[int]]` | 可以是字符串 prompt 列表，也可以是 token id 列表 |
| `sampling_params` | `SamplingParams | list[SamplingParams]` | 可以是一个公共采样参数，也可以每个请求一个采样参数 |
| `use_tqdm` | `bool` | 是否显示进度条 |
| 返回值 | `list[str]` | 注解写的是字符串列表，但实际返回是字典列表 |

### 注意：返回类型注解和实际返回不完全一致

函数最后返回：

```python
outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
return outputs
```

实际类型更像：

```python
list[dict[str, str | list[int]]]
```

而不是纯 `list[str]`。

这是 nano-VLLM 简化实现中的一个小问题。

---

### 3.14.1 创建进度条

```python
pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
```

#### 语法作用

创建一个 tqdm 进度条。

参数含义：

| 参数 | 含义 |
|---|---|
| `total=len(prompts)` | 总请求数 |
| `desc="Generating"` | 进度条描述 |
| `dynamic_ncols=True` | 根据终端宽度动态调整显示 |
| `disable=not use_tqdm` | 如果 `use_tqdm=False`，禁用进度条 |

#### 工程作用

显示生成进度。

每完成一个 Sequence，就调用：

```python
pbar.update(1)
```

---

### 3.14.2 统一 sampling_params 格式

```python
if not isinstance(sampling_params, list):
    sampling_params = [sampling_params] * len(prompts)
```

#### 语法作用

如果传入的 `sampling_params` 不是列表，就复制成和 prompts 等长的列表。

例如：

```python
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
```

如果有两个 prompts，就变成：

```python
[
    sampling_params,
    sampling_params,
]
```

#### 工程作用

支持两种使用方式：

```python
# 所有请求共用采样参数
llm.generate(prompts, SamplingParams(max_tokens=256))

# 每个请求单独设置采样参数
llm.generate(prompts, [sp1, sp2, sp3])
```

#### 注意点

这里使用的是列表重复：

```python
[sampling_params] * len(prompts)
```

如果 `SamplingParams` 是可变对象，并且后续会被修改，那么多个请求会共享同一个对象，可能有隐患。

不过在本项目中 `SamplingParams` 主要作为只读参数使用，所以问题不大。

---

### 3.14.3 添加所有请求

```python
for prompt, sp in zip(prompts, sampling_params):
    self.add_request(prompt, sp)
```

#### 语法作用

`zip(prompts, sampling_params)` 会把 prompt 和对应采样参数配对。

例如：

```text
(prompt0, sp0)
(prompt1, sp1)
(prompt2, sp2)
```

#### 工程作用

把所有用户请求加入 Scheduler。

每次循环内部会执行：

```text
prompt string → token ids
prompt token ids + sampling_params → Sequence
Sequence → Scheduler waiting queue
```

#### 在推理流程中的意义

这是请求进入推理引擎的入口。

---

### 3.14.4 初始化输出和吞吐量变量

```python
outputs = {}
prefill_throughput = decode_throughput = 0.
```

#### 语法作用

创建一个空字典 `outputs`。

同时把两个吞吐量变量初始化为浮点数 `0.0`。

#### 工程作用

`outputs` 用来按 `seq_id` 保存完成请求的 token ids。

为什么用字典？

因为请求完成顺序不一定等于输入顺序。

例如：

```text
输入顺序：seq0, seq1, seq2
完成顺序：seq1, seq2, seq0
```

最后需要按 `seq_id` 排序恢复原始顺序。

---

### 3.14.5 主生成循环

```python
while not self.is_finished():
```

#### 语法作用

只要 Scheduler 里还有未完成请求，就继续循环。

#### 工程作用

这是完整生成过程的主循环。

每一次循环会执行一轮 `step()`：

```text
schedule
  ↓
model_runner.run
  ↓
postprocess
  ↓
收集完成输出
```

#### 在 nano-VLLM 中的意义

自回归生成不是一次完成，而是一轮一轮 decode。

对于每个 token：

```text
输入当前上下文
  ↓
模型输出 logits
  ↓
采样下一个 token
  ↓
追加到 Sequence
  ↓
继续下一轮
```

因此必须有循环。

---

### 3.14.6 记录本轮开始时间

```python
t = perf_counter()
```

#### 工程作用

记录当前 step 开始时间，用于计算本轮耗时。

---

### 3.14.7 执行一轮 step

```python
output, num_tokens = self.step()
```

#### 工程作用

执行一轮完整推理步骤。

返回：

| 返回值 | 含义 |
|---|---|
| `output` | 本轮完成的请求 |
| `num_tokens` | 本轮处理 token 数；正数表示 prefill，负数表示 decode |

---

### 3.14.8 统计 prefill 吞吐量

```python
if num_tokens > 0:
    prefill_throughput = num_tokens / (perf_counter() - t)
```

#### 语法作用

如果 `num_tokens > 0`，说明本轮是 prefill。

#### 工程作用

计算 prefill tokens/s：

```text
prefill_throughput = 本轮 prefill token 数 / 本轮耗时
```

#### 在性能分析中的意义

Prefill 吞吐量主要反映模型处理 prompt 的能力。

它受以下因素影响：

```text
prompt 长度
batch token 数
FlashAttention 性能
GPU 计算能力
内存带宽
```

---

### 3.14.9 统计 decode 吞吐量

```python
else:
    decode_throughput = -num_tokens / (perf_counter() - t)
```

#### 语法作用

如果 `num_tokens <= 0`，说明本轮是 decode。

由于 decode 时 `num_tokens = -len(seqs)`，所以这里取负号还原成正数。

#### 工程作用

计算 decode tokens/s。

Decode 阶段通常每个 sequence 生成一个 token，所以：

```text
本轮 decode token 数 = 本轮参与 decode 的 sequence 数
```

#### 在性能分析中的意义

Decode 吞吐量非常关键，因为长输出任务的大部分时间花在 decode。

Decode 性能受以下因素影响：

```text
并发 sequence 数
KV Cache 读取效率
PagedAttention / FlashAttention with KV Cache
CUDA Graph
模型大小
显存带宽
```

---

### 3.14.10 更新进度条显示

```python
pbar.set_postfix({
    "Prefill": f"{int(prefill_throughput)}tok/s",
    "Decode": f"{int(decode_throughput)}tok/s",
})
```

#### 语法作用

给 tqdm 进度条添加后缀信息。

#### 工程作用

实时显示当前 prefill / decode 吞吐量。

#### 在 AI Infra 学习中的意义

这让你可以直观看到两类性能指标差别：

```text
Prefill tok/s：prompt 处理速度
Decode tok/s：生成速度
```

这和 benchmark 中常见指标对应。

---

### 3.14.11 收集本轮完成的输出

```python
for seq_id, token_ids in output:
    outputs[seq_id] = token_ids
    pbar.update(1)
```

#### 语法作用

遍历本轮完成的请求输出。

每个元素是：

```python
(seq_id, token_ids)
```

#### 工程作用

把完成请求的 completion token ids 存入字典。

同时更新进度条，表示又完成了一个请求。

#### 为什么用 seq_id

因为请求不一定按输入顺序完成。

所以需要：

```text
seq_id → token_ids
```

最后再按 seq_id 排序。

---

### 3.14.12 关闭进度条

```python
pbar.close()
```

#### 工程作用

生成结束后关闭 tqdm 进度条。

---

### 3.14.13 按原始请求顺序恢复输出

```python
outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
```

#### 语法作用

先对所有 `seq_id` 排序，然后按顺序取出对应 token ids。

#### 工程作用

确保返回结果顺序和输入 prompts 顺序一致。

例如完成顺序是：

```text
seq2, seq0, seq1
```

最终返回顺序仍然是：

```text
seq0, seq1, seq2
```

#### 在推理服务中的意义

高并发推理系统里，请求完成顺序经常与提交顺序不同，但 API 返回时通常要保持用户可理解的顺序。

---

### 3.14.14 解码 token ids

```python
outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
```

#### 语法作用

列表推导式。

对每个 completion token id 列表调用 tokenizer 解码，构造字典。

输出格式：

```python
{
    "text": "生成的文本",
    "token_ids": [123, 456, 789]
}
```

#### 工程作用

把模型内部的 token ids 转成人类可读文本，同时保留 token ids 方便调试。

#### 在推理流程中的意义

这是完整推理链路的最后一步：

```text
completion token ids
  ↓ tokenizer.decode
completion text
```

---

### 3.14.15 返回结果

```python
return outputs
```

#### 工程作用

把所有请求的生成结果返回给用户。

用户侧使用方式：

```python
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output["text"])
```

---

## 4. 背后的框架性原理

### 4.1 prompt / token / tokenizer

在本文件中，prompt 可以是两种形式：

```python
prompt: str | list[int]
```

如果是字符串：

```python
prompt = self.tokenizer.encode(prompt)
```

如果已经是 token ids：

```python
直接进入 Sequence
```

这体现了推理系统中的一个边界：

```text
用户侧：字符串文本
模型侧：整数 token ids
```

Tokenizer 的作用就是在这两者之间转换。

### 4.2 request / sequence

用户传入的每个 prompt 可以理解为一个 request。

在 nano-VLLM 内部，一个 request 被包装成：

```python
seq = Sequence(prompt, sampling_params)
```

`Sequence` 负责记录：

```text
当前请求的 token_ids
当前请求状态
生成了多少 token
已经缓存了多少 token
KV Cache block_table
采样参数
```

这就是推理引擎做调度的基本单位。

### 4.3 scheduler

`LLMEngine` 通过以下方法使用 Scheduler：

```python
self.scheduler.add(seq)
self.scheduler.schedule()
self.scheduler.postprocess(seqs, token_ids, is_prefill)
self.scheduler.is_finished()
```

Scheduler 的核心职责是：

```text
管理 waiting / running 队列
决定本轮处理哪些请求
决定是 prefill 还是 decode
更新请求状态
释放完成请求的 KV Cache
```

### 4.4 prefill / decode

本文件中最明显的 prefill / decode 标记是：

```python
seqs, is_prefill = self.scheduler.schedule()
```

以及：

```python
num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
```

Prefill：处理 prompt，可能一次处理很多 token。

Decode：逐 token 生成，每个 sequence 一轮通常生成一个 token。

这两个阶段的计算特征完全不同，所以单独统计吞吐量。

### 4.5 KV Cache

本文件没有直接分配 KV Cache，但它通过：

```python
Sequence.block_size = config.kvcache_block_size
self.scheduler = Scheduler(config)
self.model_runner = ModelRunner(config, 0, self.events)
```

把 KV Cache 相关配置传给 Sequence、Scheduler 和 ModelRunner。

KV Cache 的真实管理链路是：

```text
Config.kvcache_block_size
  ↓
Sequence.block_size
  ↓
Scheduler / BlockManager 分配 block
  ↓
ModelRunner.prepare_prefill / prepare_decode 生成 block_tables 和 slot_mapping
  ↓
Attention 读写 KV Cache
```

### 4.6 block / block table / block manager

`llm_engine.py` 只设置：

```python
Sequence.block_size = config.kvcache_block_size
```

但这行非常关键，因为 block size 决定了一个 Sequence 如何切分 token。

例如：

```text
block_size = 256
sequence length = 600
```

则需要 3 个逻辑 block：

```text
block 0: 0 ~ 255
block 1: 256 ~ 511
block 2: 512 ~ 599
```

BlockManager 再把逻辑 block 映射到物理 KV Cache block：

```text
block_table = [5, 8, 12]
```

### 4.7 model runner / GPU 执行

`LLMEngine` 通过：

```python
self.model_runner = ModelRunner(config, 0, self.events)
token_ids = self.model_runner.call("run", seqs, is_prefill)
```

间接触发 GPU 执行。

ModelRunner 负责：

```text
把 Sequence 转成 Tensor 输入
把数据放到 GPU
执行模型 forward
调用 attention 读写 KV Cache
调用 sampler 得到新 token
返回 token ids
```

### 4.8 tensor parallel

本文件中的 Tensor Parallel 入口是：

```python
for i in range(1, config.tensor_parallel_size):
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
```

如果 TP=1，不创建子进程。

如果 TP>1，启动多个 ModelRunner 进程。

每个进程对应一个 rank，用于多 GPU 模型并行。

### 4.9 logits / sampling / temperature

本文件不直接处理 logits，但它把 `SamplingParams` 放入 `Sequence`：

```python
seq = Sequence(prompt, sampling_params)
```

后续 ModelRunner 执行模型得到 logits 后，会根据 Sequence 中的 sampling params 做采样。

完整链路是：

```text
Model forward
  ↓
logits
  ↓
temperature 缩放
  ↓
softmax
  ↓
sampler 采样
  ↓
next token id
```

### 4.10 throughput / latency / TTFT / TPOT

本文件统计的是近似吞吐量：

```python
prefill_throughput = num_tokens / elapsed

decode_throughput = -num_tokens / elapsed
```

它没有完整统计：

```text
TTFT：Time To First Token，首 token 延迟
TPOT：Time Per Output Token，每输出 token 平均耗时
总延迟：单请求从提交到完成的时间
```

但它已经把 prefill 和 decode 分开统计，这和专业推理 benchmark 的思路一致。

---

## 5. 和 vLLM 原版设计的关系

### 5.1 request / sequence 调度

本文件通过：

```python
Sequence(prompt, sampling_params)
self.scheduler.add(seq)
```

体现了 vLLM 中 request / sequence 管理思想。

用户提交的是 prompt，推理引擎内部管理的是 Sequence。

### 5.2 prefill / decode 分离

本文件通过：

```python
seqs, is_prefill = self.scheduler.schedule()
```

明确把一次 step 分成 prefill 或 decode。

这是 vLLM 推理引擎的核心之一。

### 5.3 连续批处理

`generate()` 中把所有请求加入 Scheduler，然后反复：

```python
while not self.is_finished():
    output, num_tokens = self.step()
```

这体现了 continuous batching 的基本形式：

```text
多个请求不是一次固定 batch 完全同步执行
而是由 Scheduler 动态维护状态
谁完成就返回谁
```

nano-VLLM 这里是离线批量场景，没有生产服务中的动态在线请求接入，但结构上已经具备 continuous batching 的思想。

### 5.4 KV Cache block 管理

本文件通过：

```python
Sequence.block_size = config.kvcache_block_size
```

把 block size 传入 Sequence。

真正的 PagedAttention / block table 逻辑不在本文件，但本文件是它的配置入口。

### 5.5 高吞吐推理服务

本文件中体现高吞吐设计的点包括：

```text
支持多个 prompts 一起 generate
使用 Scheduler 批量调度
分离 prefill / decode throughput
支持 tensor_parallel_size
支持 ModelRunner 独立执行
```

### 5.6 nano-VLLM 相比原版 vLLM 的简化

从本文件可以看到一些简化点：

| 方向 | nano-VLLM 简化表现 |
|---|---|
| 服务接口 | 没有 OpenAI-compatible HTTP server，只是离线 generate |
| 请求管理 | 没有复杂 request priority、timeout、abort、streaming |
| 调度策略 | 由简单 Scheduler 控制，没有完整生产级调度策略 |
| 指标统计 | 只统计 prefill / decode tok/s，没有完整 TTFT / TPOT / P99 latency |
| 多进程管理 | 简单创建 ModelRunner 子进程，没有复杂 worker 管理系统 |
| 返回格式 | 简单返回 text 和 token_ids，没有完整 metadata |

但是这些简化正好适合源码学习，因为它保留了最重要的推理引擎主线：

```text
LLMEngine
  ↓
Sequence
  ↓
Scheduler
  ↓
ModelRunner
  ↓
Attention / KV Cache / Sampler
```

---

## 6. 本文件学习重点总结

`llm_engine.py` 是你学习 nano-VLLM 时必须精读的核心文件之一。

你要重点掌握 5 个问题：

### 6.1 `generate()` 为什么不是一次模型 forward？

因为大模型自回归生成需要循环 decode：

```text
每轮生成一个 token
追加到上下文
继续下一轮
```

同时多个请求长度不同、结束时间不同，所以必须通过 Scheduler 管理。

### 6.2 `Sequence` 是什么？

`Sequence` 是一个请求在推理引擎内部的状态表示。

```text
prompt + sampling_params + token 状态 + KV Cache block_table
```

### 6.3 `Scheduler` 负责什么？

Scheduler 决定本轮处理哪些请求，是 prefill 还是 decode，并负责后处理状态。

### 6.4 `ModelRunner` 负责什么？

ModelRunner 真正执行模型前向、KV Cache 读写和采样。

### 6.5 为什么要区分 prefill throughput 和 decode throughput？

因为两者性能瓶颈不同：

```text
Prefill：计算密集，处理 prompt tokens
Decode：内存访问密集，逐 token 读取 KV Cache
```

---

## 7. 推荐下一步阅读顺序

读完 `llm_engine.py` 后，下一步建议按这个顺序读：

```text
1. sequence.py
   理解 Sequence 如何保存请求状态。

2. scheduler.py
   理解 waiting / running 队列、prefill / decode 调度和 postprocess。

3. block_manager.py
   理解 KV Cache block 分配、释放和 block_table。

4. model_runner.py
   理解 prepare_prefill / prepare_decode / run_model / sampler。

5. attention.py
   理解 slot_mapping、block_tables、FlashAttention 和 KV Cache 读写。
```

其中，和本文件关系最紧密的是：

```text
sequence.py
scheduler.py
model_runner.py
```

---

## 8. 一句话总结

`llm_engine.py` 是 nano-VLLM 的推理主控文件。它不直接实现 Attention 或 KV Cache，但它把用户输入、请求封装、调度器、模型执行器、采样结果和最终文本输出串成了一条完整链路。理解这个文件，就等于理解了 nano-VLLM 的顶层执行流程。

核心调用链是：

```text
LLMEngine.generate
  ↓
add_request
  ↓
Sequence
  ↓
Scheduler.add
  ↓
while not finished:
      step
        ↓
      Scheduler.schedule
        ↓
      ModelRunner.call("run")
        ↓
      Scheduler.postprocess
  ↓
tokenizer.decode
  ↓
return outputs
```


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]
- 关联阅读：[[outputs/项目整理/专题-01-请求从输入到输出|专题-01-请求从输入到输出]]

%% 项目关联导航：结束 %%
