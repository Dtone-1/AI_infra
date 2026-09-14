# `model_runner.py` 源码逐行精读与框架原理解析

> 解析对象：`nanovllm/engine/model_runner.py`  
> 所属主题：nano-vLLM / vLLM / AI Infra / 大模型推理引擎 / 模型执行器 / KV Cache / CUDA Graph / Tensor Parallel

---

## 1. 文件整体定位

### 1.1 这个文件负责什么

`model_runner.py` 是 nano-vLLM 中非常核心的文件之一。它不是用户入口，也不是请求调度器，而是**模型执行层**。

它主要负责：

1. 初始化分布式执行环境；
2. 创建并加载 Qwen3 模型；
3. 根据剩余 GPU 显存分配 KV Cache；
4. 将 `Scheduler` 选出来的 `Sequence` 转换成模型可以执行的 Tensor 输入；
5. 区分 Prefill 和 Decode 两种执行路径；
6. 设置 Attention 层需要的运行时上下文；
7. 调用模型 forward；
8. 计算 logits；
9. 调用 sampler 采样下一个 token；
10. 在 Decode 阶段使用 CUDA Graph 加速；
11. 在 Tensor Parallel 多进程场景下协调 rank 0 和其他 rank 的执行。

如果说：

```text
LLMEngine = 推理流程总控
Scheduler = 请求调度器
BlockManager = KV Cache block 管理器
Sequence = 单条请求状态对象
ModelRunner = 真正把 batch 送进模型执行的执行器
```

那么 `model_runner.py` 就是从“调度决策”走向“GPU 实际计算”的桥梁。

---

### 1.2 它属于哪一层

它属于：

```text
模型执行层 / GPU 执行层 / Runtime 层 / Model Runner 层
```

它直接面向 CUDA、PyTorch Tensor、分布式通信、KV Cache 张量和模型 forward。

在 nano-vLLM 的分层中，可以这样看：

```text
用户 API 层：
    example.py / bench.py / llm.py

引擎总控层：
    llm_engine.py

请求状态层：
    sequence.py

调度层：
    scheduler.py

KV Cache 管理层：
    block_manager.py

模型执行层：
    model_runner.py   ← 当前文件

模型结构层：
    models/qwen3.py

算子与层实现：
    attention.py / linear.py / sampler.py / embed_head.py / layernorm.py
```

---

### 1.3 它和哪些文件有关

`model_runner.py` 和以下文件关系最密切：

| 文件 | 关系 |
|---|---|
| `llm_engine.py` | `LLMEngine.step()` 通过 `model_runner.call("run", seqs, is_prefill)` 调用它 |
| `scheduler.py` | Scheduler 产出 `seqs` 和 `is_prefill`，交给 ModelRunner 执行 |
| `sequence.py` | ModelRunner 读取每条 `Sequence` 的 token、block_table、num_cached_tokens 等状态 |
| `block_manager.py` | block_manager 维护的 `block_table` 被 ModelRunner 转成 `block_tables` Tensor |
| `models/qwen3.py` | ModelRunner 创建并调用 `Qwen3ForCausalLM` |
| `layers/attention.py` | Attention 通过 context 获取 `slot_mapping`、`block_tables`、`context_lens`，完成 KV Cache 读写 |
| `layers/sampler.py` | ModelRunner 使用 `Sampler` 根据 logits 采样 token |
| `utils/context.py` | ModelRunner 调用 `set_context()` 设置 Attention 运行时上下文 |
| `utils/loader.py` | ModelRunner 调用 `load_model()` 加载 HuggingFace 权重 |

---

### 1.4 在完整推理流程中的位置

完整推理链路如下：

```text
用户输入 prompt
   ↓
LLMEngine.generate()
   ↓
LLMEngine.add_request()
   ↓
Sequence(prompt_token_ids, sampling_params)
   ↓
Scheduler.add(seq)
   ↓
while not finished:
   Scheduler.schedule()
      ↓
   得到本轮 seqs 和 is_prefill
      ↓
   ModelRunner.run(seqs, is_prefill)   ← 当前文件核心位置
      ↓
   prepare_prefill() 或 prepare_decode()
      ↓
   set_context()
      ↓
   Qwen3ForCausalLM.forward()
      ↓
   Attention 读写 KV Cache
      ↓
   compute_logits()
      ↓
   Sampler 采样下一个 token
      ↓
   Scheduler.postprocess()
      ↓
   更新 Sequence / 释放 KV Cache / 判断结束
```

`model_runner.py` 处在 `Scheduler` 和 `Qwen3ForCausalLM` 之间。

它的核心任务是：

```text
把 Sequence 这种“调度层对象”转换成 torch.Tensor 这种“模型计算对象”。
```

---

## 2. 代码结构总览

### 2.1 文件导入了哪些模块

```python
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
```

这些导入可以分成四类：

#### 第一类：序列化和多进程通信

```python
import pickle
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
```

用于 Tensor Parallel 多进程场景：

- rank 0 把要执行的方法名和参数 pickle 序列化；
- 写入共享内存；
- 通过 `Event` 通知其他 rank；
- 其他 rank 从共享内存读取任务并同步执行。

#### 第二类：PyTorch / CUDA / 分布式

```python
import torch
import torch.distributed as dist
```

用于：

- 创建 Tensor；
- 设置 CUDA 设备；
- 申请 GPU 显存；
- 初始化 NCCL 通信组；
- 使用 CUDA Graph；
- 执行模型 forward。

#### 第三类：nano-vLLM 内部对象

```python
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
```

`Config` 是全局配置，`Sequence` 是单条请求状态。

#### 第四类：模型、采样、上下文和权重加载

```python
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
```

分别负责：

- 创建 Qwen3 模型；
- 对 logits 采样；
- 给 Attention 层传递运行时上下文；
- 加载 HuggingFace safetensors 权重。

---

### 2.2 定义了哪些类

本文件只定义了一个类：

```python
class ModelRunner:
```

它是模型执行器。

---

### 2.3 定义了哪些函数 / 方法

`ModelRunner` 里包含以下方法：

| 方法 | 作用 |
|---|---|
| `__init__()` | 初始化分布式环境、模型、采样器、KV Cache、CUDA Graph |
| `exit()` | 释放共享内存、CUDA Graph、同步并销毁进程组 |
| `loop()` | 非 rank 0 进程循环等待 rank 0 指令 |
| `read_shm()` | 子进程从共享内存读取方法名和参数 |
| `write_shm()` | rank 0 向共享内存写入方法名和参数 |
| `call()` | 统一方法调用入口，同时通知 TP 子进程 |
| `warmup_model()` | 预热模型，并为 KV Cache 显存估算提供 peak memory |
| `allocate_kv_cache()` | 根据显存情况申请 KV Cache 大 Tensor |
| `prepare_block_tables()` | 把每条 Sequence 的 block_table 整理成二维 Tensor |
| `prepare_prefill()` | 为 Prefill 阶段准备输入 Tensor 和 Attention 上下文 |
| `prepare_decode()` | 为 Decode 阶段准备输入 Tensor 和 Attention 上下文 |
| `prepare_sample()` | 准备每条请求的 temperature Tensor |
| `run_model()` | 执行模型 forward，必要时使用 CUDA Graph |
| `run()` | 一轮模型执行主入口：准备输入、跑模型、采样 token |
| `capture_cudagraph()` | 捕获 Decode 阶段的 CUDA Graph |

---

### 2.4 重要变量和数据结构

#### `self.config`

类型：`Config`

作用：保存模型路径、最大 batch token、最大序列数、最大上下文长度、KV Cache block size、TP size、是否强制 eager 等配置。

#### `self.block_size`

类型：`int`

作用：每个 KV Cache block 容纳多少 token，通常是 256。

#### `self.world_size`

类型：`int`

作用：Tensor Parallel 总进程数 / GPU 数。

#### `self.rank`

类型：`int`

作用：当前 ModelRunner 所在的 TP rank。

#### `self.model`

类型：`Qwen3ForCausalLM`

作用：实际执行 forward 的模型。

#### `self.kv_cache`

类型：`torch.Tensor`

形状：

```text
[2, num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads_per_rank, head_dim]
```

含义：

```text
第 0 维的 2 表示 K Cache 和 V Cache
每一层都有自己的 K/V Cache
每个 block 存 block_size 个 token 的 K/V
```

#### `block_tables`

类型：`torch.Tensor`

形状大致为：

```text
[num_seqs, max_num_blocks_per_seq]
```

作用：告诉 Attention 每条 Sequence 的逻辑 block 对应哪个物理 KV Cache block。

#### `slot_mapping`

类型：`torch.Tensor`

形状：

```text
Prefill: [本轮要写入 KV Cache 的 token 数]
Decode: [本轮 decode 的 seq 数]
```

作用：告诉 Attention 当前 token 的 K/V 应该写到 KV Cache 的哪个物理 slot。

#### `cu_seqlens_q / cu_seqlens_k`

类型：`torch.Tensor[int32]`

作用：FlashAttention varlen 接口需要的累计序列长度。

#### `context_lens`

类型：`torch.Tensor[int32]`

形状：

```text
[decode_batch_size]
```

作用：Decode 阶段每条请求当前上下文长度。

---

### 2.5 主流程代码

主流程集中在：

```python
def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
```

它调用链如下：

```text
run(seqs, is_prefill)
   ↓
prepare_prefill() 或 prepare_decode()
   ↓
prepare_sample()
   ↓
run_model()
   ↓
sampler()
   ↓
reset_context()
   ↓
返回 token_ids
```

---

### 2.6 辅助逻辑

辅助逻辑主要包括：

```text
多进程通信：loop / read_shm / write_shm / call
显存规划：warmup_model / allocate_kv_cache
CUDA Graph：capture_cudagraph / run_model 中 graph.replay()
退出清理：exit
```

---

### 2.7 文件组织结构图

```text
model_runner.py
│
├── imports
│   ├── pickle / multiprocessing shared memory
│   ├── torch / torch.distributed
│   ├── Config / Sequence
│   ├── Qwen3ForCausalLM
│   ├── Sampler
│   ├── context utils
│   └── load_model
│
└── class ModelRunner
    │
    ├── 初始化与资源管理
    │   ├── __init__
    │   └── exit
    │
    ├── Tensor Parallel 多进程通信
    │   ├── loop
    │   ├── read_shm
    │   ├── write_shm
    │   └── call
    │
    ├── 显存和 KV Cache
    │   ├── warmup_model
    │   └── allocate_kv_cache
    │
    ├── 输入准备
    │   ├── prepare_block_tables
    │   ├── prepare_prefill
    │   ├── prepare_decode
    │   └── prepare_sample
    │
    ├── 模型执行
    │   ├── run_model
    │   └── run
    │
    └── CUDA Graph
        └── capture_cudagraph
```

---

## 3. 逐行 / 逐代码块解释

### 3.1 导入模块

```python
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
```

#### 语法作用

导入 Python 标准库和 PyTorch 相关模块。

#### 工程作用

- `pickle`：把 Python 对象序列化为 bytes，用于共享内存通信；
- `torch`：创建 Tensor、执行 CUDA 计算、使用 CUDA Graph；
- `torch.distributed as dist`：初始化 NCCL 进程组和执行多 GPU 通信；
- `Event`：多进程同步信号；
- `SharedMemory`：多进程共享内存区域。

#### 在推理流程中的意义

这个文件不只是跑模型，还要支持 Tensor Parallel。TP 场景下，每张 GPU 上有一个进程，每个进程都有一个 `ModelRunner`。rank 0 负责接收上层调用，然后通过共享内存和 Event 通知其他 rank 同步执行。

---

```python
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
```

#### 语法作用

导入 nano-vLLM 内部模块。

#### 工程作用

- `Config`：提供模型路径、显存比例、TP size、KV block size 等配置；
- `Sequence`：表示本轮要执行的请求；
- `Qwen3ForCausalLM`：具体模型；
- `Sampler`：从 logits 中采样 token；
- `set_context/get_context/reset_context`：向 Attention 层传递运行时信息；
- `load_model`：加载 HuggingFace 权重。

#### 下一步结合哪个文件看

- `Config`：看 `config.py`；
- `Sequence`：看 `sequence.py`；
- `Qwen3ForCausalLM`：看 `models/qwen3.py`；
- `Sampler`：看 `layers/sampler.py`；
- `set_context`：看 `utils/context.py`；
- KV Cache 实际使用：看 `layers/attention.py`。

---

### 3.2 定义 `ModelRunner`

```python
class ModelRunner:
```

#### 语法作用

定义一个类。

#### 工程作用

`ModelRunner` 是模型执行器，封装了从 `Sequence` 到 GPU 前向计算再到 token 采样的全过程。

#### 在 nano-vLLM 中的意义

`LLMEngine.step()` 不直接调用模型，而是调用：

```python
self.model_runner.call("run", seqs, is_prefill)
```

所以 `ModelRunner.run()` 是每一轮推理真正执行的地方。

---

### 3.3 初始化方法：保存配置和基础属性

```python
def __init__(self, config: Config, rank: int, event: Event | list[Event]):
    self.config = config
    hf_config = config.hf_config
    self.block_size = config.kvcache_block_size
    self.enforce_eager = config.enforce_eager
    self.world_size = config.tensor_parallel_size
    self.rank = rank
    self.event = event
```

#### 语法作用

定义构造函数，并接收三个参数：

| 参数 | 类型 | 含义 |
|---|---|---|
| `config` | `Config` | 全局配置对象 |
| `rank` | `int` | 当前 TP 进程编号 |
| `event` | `Event | list[Event]` | 多进程同步事件 |

`Event | list[Event]` 是 Python 类型注解，表示该参数可以是单个 `Event`，也可以是 `Event` 列表。

#### 工程作用

这些字段决定当前 runner 的运行环境：

```text
block_size      → KV Cache block 大小
enforce_eager   → 是否禁用 CUDA Graph
world_size      → Tensor Parallel 总进程数
rank            → 当前进程编号
event           → TP 通信同步信号
```

#### 在推理流程中的意义

一个 `ModelRunner` 对应一个 GPU rank。单卡时只有 rank 0；多卡 TP 时 rank 0 是主进程，rank 1、2、3 等是辅助执行进程。

---

### 3.4 初始化分布式环境和 CUDA 设备

```python
dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
torch.cuda.set_device(rank)
```

#### 语法作用

初始化 PyTorch 分布式进程组，并设置当前 CUDA 设备。

#### 工程作用

- `nccl` 是 NVIDIA GPU 间通信常用后端；
- `world_size` 表示总进程数；
- `rank` 表示当前进程编号；
- `torch.cuda.set_device(rank)` 让 rank 0 用 GPU 0，rank 1 用 GPU 1，以此类推。

#### 在推理流程中的意义

Tensor Parallel 下，每个 rank 只保存一部分权重和一部分 KV head。模型 forward 时，不同 rank 协同计算。

#### 下一步结合哪个文件看

看 `layers/linear.py` 和 `layers/embed_head.py`，理解 QKV / MLP / lm_head 是如何按 Tensor Parallel 切分的。

---

### 3.5 设置默认 dtype 和默认设备

```python
default_dtype = torch.get_default_dtype()
torch.set_default_dtype(hf_config.dtype)
torch.set_default_device("cuda")
```

#### 语法作用

读取当前默认 dtype，然后把默认 dtype 和默认 device 改为模型配置对应的类型和 CUDA。

#### 工程作用

后续创建模型参数或 Tensor 时，如果没有显式指定 dtype/device，就默认创建在：

```text
设备：cuda
类型：hf_config.dtype，例如 torch.float16 / torch.bfloat16
```

#### 在推理流程中的意义

大模型推理通常使用 FP16/BF16，而不是 FP32。这样可以节省显存并提高 Tensor Core 计算效率。

---

### 3.6 创建模型、加载权重、创建采样器

```python
self.model = Qwen3ForCausalLM(hf_config)
load_model(self.model, config.model)
self.sampler = Sampler()
```

#### 语法作用

创建模型对象、加载权重、创建采样器。

#### 工程作用

- `Qwen3ForCausalLM(hf_config)`：根据 HuggingFace 配置构造 Qwen3 模型结构；
- `load_model(self.model, config.model)`：从本地模型目录加载权重；
- `Sampler()`：后续把 logits 转成 token id。

#### 在推理流程中的意义

模型执行层需要两个核心组件：

```text
模型 forward：input_ids + positions → hidden_states → logits
Sampler：logits + temperature → next_token_id
```

---

### 3.7 预热模型、分配 KV Cache、捕获 CUDA Graph

```python
self.warmup_model()
self.allocate_kv_cache()
if not self.enforce_eager:
    self.capture_cudagraph()
```

#### 语法作用

依次调用三个成员方法。

#### 工程作用

1. `warmup_model()`：先跑一次模型，触发 CUDA kernel 初始化，并统计峰值显存；
2. `allocate_kv_cache()`：根据显存余量计算能放多少 KV Cache block；
3. `capture_cudagraph()`：如果允许优化，则为 Decode 阶段捕获 CUDA Graph。

#### 为什么先 warmup 再 allocate KV Cache

因为模型 forward 首次运行时会有额外显存开销，例如：

```text
CUDA kernel 初始化
cuBLAS workspace
FlashAttention workspace
临时张量峰值
```

如果不先 warmup，就可能高估可用于 KV Cache 的显存，导致后续 OOM。

---

### 3.8 恢复默认 device 和 dtype

```python
torch.set_default_device("cpu")
torch.set_default_dtype(default_dtype)
```

#### 语法作用

恢复 PyTorch 默认设备和默认数据类型。

#### 工程作用

避免后续普通 Tensor 被无意创建到 CUDA 上，也避免影响其他模块。

#### 在推理流程中的意义

这是一种工程安全处理：初始化模型阶段可以默认 CUDA，但初始化结束后恢复默认 CPU，避免隐藏的显存占用。

---

### 3.9 多进程共享内存初始化

```python
if self.world_size > 1:
    if rank == 0:
        self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
        dist.barrier()
    else:
        dist.barrier()
        self.shm = SharedMemory(name="nanovllm")
        self.loop()
```

#### 语法作用

判断是否启用 Tensor Parallel。如果 `world_size > 1`，则创建或连接共享内存。

#### 工程作用

- rank 0 创建名为 `nanovllm` 的共享内存；
- 其他 rank 等待 barrier 后打开同一块共享内存；
- 非 rank 0 进入 `loop()`，等待 rank 0 分发任务。

#### 在推理流程中的意义

`LLMEngine` 只直接调用 rank 0 的 `ModelRunner`。但是 TP 需要所有 rank 同步执行模型 forward。因此 rank 0 通过共享内存通知其他 rank：

```text
现在执行 run(seqs, is_prefill)
现在执行 exit()
```

#### 简化点

生产级推理系统通常会有更复杂的 RPC、worker 管理和异常处理。nano-vLLM 用共享内存 + Event 实现了一个很轻量的同步机制。

---

### 3.10 退出清理：`exit`

```python
def exit(self):
    if self.world_size > 1:
        self.shm.close()
        dist.barrier()
        if self.rank == 0:
            self.shm.unlink()
    if not self.enforce_eager:
        del self.graphs, self.graph_pool
    torch.cuda.synchronize()
    dist.destroy_process_group()
```

#### 语法作用

定义资源释放方法。

#### 工程作用

释放：

- shared memory；
- CUDA Graph 对象；
- 分布式进程组。

`torch.cuda.synchronize()` 确保所有 CUDA 操作完成后再销毁资源。

#### 在推理流程中的意义

推理引擎退出时必须清理 GPU 和进程资源，否则可能出现：

```text
共享内存残留
NCCL 进程未退出
CUDA kernel 尚未完成
显存未释放
```

---

### 3.11 子进程循环：`loop`

```python
def loop(self):
    while True:
        method_name, args = self.read_shm()
        self.call(method_name, *args)
        if method_name == "exit":
            break
```

#### 语法作用

定义一个无限循环，直到收到 `exit` 指令。

#### 工程作用

非 rank 0 的 ModelRunner 不直接被 `LLMEngine` 调用，而是在这里等待 rank 0 指令。

流程：

```text
等待 Event
   ↓
读取共享内存中的方法名和参数
   ↓
调用对应方法
   ↓
如果是 exit，则退出循环
```

#### 在推理流程中的意义

多 GPU Tensor Parallel 必须保证每个 rank 同步执行相同 forward。`loop()` 就是辅助 rank 的任务监听器。

---

### 3.12 从共享内存读取任务：`read_shm`

```python
def read_shm(self):
    assert self.world_size > 1 and self.rank > 0
    self.event.wait()
    n = int.from_bytes(self.shm.buf[0:4], "little")
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])
    self.event.clear()
    return method_name, args
```

#### 语法作用

从共享内存中读取 bytes，再反序列化为 Python 对象。

#### 工程作用

共享内存布局是：

```text
前 4 字节：数据长度 n
后 n 字节：pickle 序列化后的 [method_name, *args]
```

`self.event.wait()` 表示等待 rank 0 发信号。

#### 在推理流程中的意义

当 rank 0 调用：

```python
write_shm("run", seqs, is_prefill)
```

其他 rank 会在这里读到：

```text
method_name = "run"
args = [seqs, is_prefill]
```

然后执行同样的 `run()`。

---

### 3.13 向共享内存写任务：`write_shm`

```python
def write_shm(self, method_name, *args):
    assert self.world_size > 1 and self.rank == 0
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")
    self.shm.buf[4:n+4] = data
    for event in self.event:
        event.set()
```

#### 语法作用

把方法名和参数序列化后写入共享内存，并触发 Event。

#### 工程作用

rank 0 用这个方法广播任务给其他 rank。

#### 在推理流程中的意义

当上层调用 rank 0：

```python
self.model_runner.call("run", seqs, is_prefill)
```

rank 0 先把任务写入共享内存，通知其他 rank，然后自己也执行 `run()`。

---

### 3.14 统一调用入口：`call`

```python
def call(self, method_name, *args):
    if self.world_size > 1 and self.rank == 0:
        self.write_shm(method_name, *args)
    method = getattr(self, method_name, None)
    return method(*args)
```

#### 语法作用

根据字符串方法名获取对象方法并调用。

#### 工程作用

这是 rank 0 和非 rank 0 都会使用的统一入口。

例如：

```python
call("run", seqs, is_prefill)
```

等价于：

```python
self.run(seqs, is_prefill)
```

如果是 rank 0 且 world_size > 1，还会先通知其他 rank。

#### 在推理流程中的意义

`LLMEngine.step()` 调用的就是：

```python
token_ids = self.model_runner.call("run", seqs, is_prefill)
```

因此 `call()` 是总控层进入模型执行层的入口。

---

### 3.15 模型预热：`warmup_model`

```python
def warmup_model(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
    seq_len = min(max_num_batched_tokens, max_model_len)
    num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
    seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
    for seq in seqs:
        seq.num_scheduled_tokens = seq_len
    self.run(seqs, True)
    torch.cuda.empty_cache()
```

#### 语法作用

定义模型预热函数，构造若干假 `Sequence`，执行一次 Prefill。

#### 工程作用

预热的目的：

1. 触发模型首次 forward 的初始化开销；
2. 统计 peak memory；
3. 清理缓存；
4. 为后续 `allocate_kv_cache()` 估算可用显存做准备。

#### 关键参数解释

```python
seq_len = min(max_num_batched_tokens, max_model_len)
```

单条预热序列长度不能超过最大模型上下文长度，也不能超过本轮最大 batched token 数。

```python
num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
```

预热时总 token 数不超过 `max_num_batched_tokens`，seq 数不超过 `max_num_seqs`。

#### 在推理流程中的意义

这一步不产生真实用户输出，只是让 GPU 进入稳定状态，并帮助估算 KV Cache 空间。

#### 注意点

预热发生在 `allocate_kv_cache()` 之前，所以这些 warmup sequence 没有真实 `block_table`。后续 `prepare_prefill()` 中有：

```python
if not seq.block_table:    # warmup
    continue
```

这说明 warmup 不执行真实 KV Cache 写入。

---

### 3.16 分配 KV Cache：`allocate_kv_cache`

```python
def allocate_kv_cache(self):
    config = self.config
    hf_config = config.hf_config
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
    config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
    assert config.num_kvcache_blocks > 0
    self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

#### 语法作用

定义 KV Cache 分配函数。

#### 工程作用

它根据 GPU 显存估算可以放多少个 KV Cache block，然后创建一个大 Tensor，并把每一层 Attention 的 `k_cache` 和 `v_cache` 指向这块 Tensor 的对应切片。

#### 显存估算逻辑

```python
free, total = torch.cuda.mem_get_info()
used = total - free
peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
```

含义：

| 变量 | 含义 |
|---|---|
| `free` | 当前空闲显存 |
| `total` | GPU 总显存 |
| `used` | 当前已使用显存 |
| `peak` | 预热期间峰值分配显存 |
| `current` | 当前仍分配的显存 |

可用于 KV Cache 的显存大致为：

```text
total * gpu_memory_utilization - used - peak + current
```

意思是：

```text
目标可用显存预算
  - 当前已用显存
  - 预热时额外峰值开销
  + 当前已经算在 used 里的持久显存
```

这个估算是为了给模型运行时临时开销留余量。

#### 每个 block 占用多少显存

```python
block_bytes = 2 * num_hidden_layers * block_size * num_kv_heads * head_dim * dtype.itemsize
```

含义：

```text
2                    → K 和 V
num_hidden_layers    → 每一层都有 KV Cache
block_size           → 每个 block 存多少 token
num_kv_heads         → 当前 rank 上的 KV head 数
head_dim             → 每个 head 的维度
dtype.itemsize       → 每个元素占多少字节
```

因此 KV Cache 显存和下面因素成正比：

```text
层数 × token 数 × KV head 数 × head_dim × dtype 字节数 × 2
```

#### KV Cache Tensor 形状

```python
self.kv_cache = torch.empty(
    2,
    hf_config.num_hidden_layers,
    config.num_kvcache_blocks,
    self.block_size,
    num_kv_heads,
    head_dim
)
```

形状解释：

| 维度 | 含义 |
|---|---|
| `2` | K Cache 和 V Cache |
| `num_hidden_layers` | Transformer 层数 |
| `num_kvcache_blocks` | 物理 KV Cache block 数量 |
| `block_size` | 每个 block 的 token 容量 |
| `num_kv_heads` | 当前 rank 的 KV head 数 |
| `head_dim` | 每个 head 的维度 |

#### 分配给每层 Attention

```python
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id += 1
```

这说明模型中的每一层 Attention 都持有自己的 K/V Cache 视图。

#### 下一步结合哪个文件看

看 `layers/attention.py`，理解 `k_cache` 和 `v_cache` 是如何被写入和读取的。

---

### 3.17 准备 block_tables

```python
def prepare_block_tables(self, seqs: list[Sequence]):
    max_len = max(len(seq.block_table) for seq in seqs)
    block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
    block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    return block_tables
```

#### 语法作用

接收 `list[Sequence]`，返回一个 CUDA Tensor。

#### 工程作用

每条 Sequence 的 `block_table` 长度可能不同，因此需要 padding 成二维矩阵。

例如：

```text
seq1.block_table = [5, 8, 12]
seq2.block_table = [3, 7]
```

会变成：

```text
[
  [5, 8, 12],
  [3, 7, -1]
]
```

#### Tensor 类型和形状

```text
dtype: torch.int32
device: cuda
shape: [num_seqs, max_block_table_len]
```

#### 在推理流程中的意义

Attention 在 Decode 阶段需要根据 `block_tables` 找到每条请求历史 KV Cache 的物理 block。

这是 PagedAttention 的关键数据结构之一。

---

### 3.18 准备 Prefill 输入：初始化变量

```python
def prepare_prefill(self, seqs: list[Sequence]):
    input_ids = []
    positions = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping = []
    block_tables = None
```

#### 语法作用

定义 Prefill 输入准备函数，并初始化若干列表和变量。

#### 工程作用

这些变量最终会被转换成 CUDA Tensor，供模型和 Attention 使用。

| 变量 | 作用 |
|---|---|
| `input_ids` | 本轮要输入模型的 token id |
| `positions` | 每个 token 的 position id |
| `cu_seqlens_q` | FlashAttention varlen 的 query 累计长度 |
| `cu_seqlens_k` | FlashAttention varlen 的 key 累计长度 |
| `max_seqlen_q` | 本 batch 最大 query 长度 |
| `max_seqlen_k` | 本 batch 最大 key 长度 |
| `slot_mapping` | 当前 token 写入 KV Cache 的物理 slot |
| `block_tables` | prefix cache 场景需要的历史 block 映射 |

---

### 3.19 Prefill 遍历每条 Sequence

```python
for seq in seqs:
    start = seq.num_cached_tokens
    seqlen_q = seq.num_scheduled_tokens
    end = start + seqlen_q
    seqlen_k = end
    input_ids.extend(seq[start:end])
    positions.extend(range(start, end))
    cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
    cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
    max_seqlen_q = max(seqlen_q, max_seqlen_q)
    max_seqlen_k = max(seqlen_k, max_seqlen_k)
```

#### 语法作用

遍历本轮 Prefill 要执行的所有 seq。

#### 工程作用

对于每条 seq，确定本轮要计算哪一段 token：

```text
start = 已经缓存过的 token 数
seqlen_q = 本轮调度要计算的 token 数
end = start + seqlen_q
```

例如：

```text
seq 总长度 = 1000
num_cached_tokens = 256
num_scheduled_tokens = 512
```

则本轮处理：

```text
token[256:768]
```

#### 为什么 `seqlen_k = end`

Prefill 阶段中，当前 query token 可以看到从开头到当前位置的所有 key。对于这条 seq，本轮结束后可见的 key 长度是 `end`。

在 prefix cache 场景下：

```text
query 长度 = 本轮新算的 token 数
key 长度 = cached prefix + 本轮新算 token
```

所以 `seqlen_k` 可能大于 `seqlen_q`。

#### 在推理流程中的意义

这段代码支持 chunked prefill 和 prefix cache。

不是每次都从 token 0 开始算，而是可以从 `num_cached_tokens` 开始继续算。

---

### 3.20 Prefill 中计算 slot_mapping

```python
if not seq.block_table:    # warmup
    continue
start_block = start // self.block_size
end_block = (end + self.block_size - 1) // self.block_size
for i in range(start_block, end_block):
    slot_start = seq.block_table[i] * self.block_size
    if i == start_block:
        slot_start += start % self.block_size
    if i != end_block - 1:
        slot_end = seq.block_table[i] * self.block_size + self.block_size
    else:
        slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
    slot_mapping.extend(range(slot_start, slot_end))
```

#### 语法作用

根据 `seq.block_table` 和 token 范围，计算每个 token 应该写入 KV Cache 的物理位置。

#### 工程作用

`slot_mapping` 是 token 到 KV Cache 物理 slot 的映射。

例如：

```text
block_size = 256
seq.block_table = [5, 8, 12]
```

逻辑 token 0~255 对应物理 block 5；
逻辑 token 256~511 对应物理 block 8；
逻辑 token 512~767 对应物理 block 12。

物理 slot 计算方式：

```text
physical_slot = physical_block_id * block_size + offset_in_block
```

#### 在推理流程中的意义

Attention 计算出当前 token 的 K/V 后，需要知道写到 KV Cache 的哪个位置。`slot_mapping` 就是这个写入地址表。

#### 下一步结合哪个文件看

看 `layers/attention.py` 中的 KV Cache 写入 kernel，通常会用 `slot_mapping` 将 K/V 写入 `k_cache` 和 `v_cache`。

---

### 3.21 prefix cache 场景下准备 block_tables

```python
if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
    block_tables = self.prepare_block_tables(seqs)
```

#### 语法作用

判断是否存在 prefix cache。

#### 工程作用

如果总 key 长度大于总 query 长度，说明有一部分 K/V 来自已经缓存的 prefix，而不是本轮新算的 token。

这种情况下 Attention 需要 `block_tables` 找到历史 cached block。

#### 在推理流程中的意义

这是 Prefix Caching 的体现。

例如多个请求共享相同系统 prompt：

```text
请求 A: system prompt + 问题 A
请求 B: system prompt + 问题 B
```

如果 system prompt 的 KV Cache 已经存在，请求 B 的前缀可以复用，不必重新计算。

---

### 3.22 转换为 CUDA Tensor 并设置 context

```python
input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
return input_ids, positions
```

#### 语法作用

将 Python list 转为 CUDA Tensor，然后调用 `set_context()`。

#### 工程作用

Tensor 类型：

| Tensor | dtype | shape |
|---|---|---|
| `input_ids` | `int64` | `[total_query_tokens]` |
| `positions` | `int64` | `[total_query_tokens]` |
| `cu_seqlens_q` | `int32` | `[num_seqs + 1]` |
| `cu_seqlens_k` | `int32` | `[num_seqs + 1]` |
| `slot_mapping` | `int32` | `[total_query_tokens_to_write]` |

`pin_memory=True` 表示先创建 pinned CPU memory，配合 `.cuda(non_blocking=True)` 可以更高效地异步拷贝到 GPU。

#### 在推理流程中的意义

`set_context()` 将 Prefill 所需的 Attention 元信息保存到全局 context 中。后续模型 forward 到 Attention 层时，Attention 可以通过 `get_context()` 取到这些信息。

---

### 3.23 准备 Decode 输入

```python
def prepare_decode(self, seqs: list[Sequence]):
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []
    for seq in seqs:
        input_ids.append(seq.last_token)
        positions.append(len(seq) - 1)
        context_lens.append(len(seq))
        slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
```

#### 语法作用

定义 Decode 阶段输入准备函数。

#### 工程作用

Decode 阶段每条 seq 只输入最后一个 token。

对于每条 seq：

| 变量 | 含义 |
|---|---|
| `seq.last_token` | 本轮输入模型的 token |
| `len(seq) - 1` | 当前 token 的 position |
| `len(seq)` | 当前上下文长度 |
| `slot_mapping` | 当前 token 的 K/V 写入位置 |

#### 为什么 Decode 只输入 last_token

自回归生成时，如果已经有 KV Cache，历史 token 不需要重新计算。每轮只需要计算当前 token 的 Q/K/V，然后让当前 query attend 到历史 KV Cache。

所以 Decode 输入规模大致是：

```text
input_ids.shape = [decode_batch_size]
```

而不是：

```text
[所有历史 token 数]
```

#### slot_mapping 的含义

```python
seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
```

表示当前最后一个 token 应该写入最后一个物理 block 的某个 offset。

---

### 3.24 Decode 转 Tensor 并设置 context

```python
input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
block_tables = self.prepare_block_tables(seqs)
set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
return input_ids, positions
```

#### 工程作用

Decode 阶段需要的核心数据：

| 数据 | 用途 |
|---|---|
| `input_ids` | 当前每条 seq 的 last token |
| `positions` | 当前 token 的位置 |
| `slot_mapping` | 当前 K/V 写入位置 |
| `context_lens` | 每条 seq 当前上下文长度 |
| `block_tables` | 每条 seq 的历史 KV block 映射 |

#### 在 Attention 中如何使用

Decode Attention 会使用：

```text
当前 token 的 query
+ 历史 KV Cache
+ block_tables
+ context_lens
```

来完成当前 token 对全部历史上下文的注意力计算。

---

### 3.25 准备采样温度

```python
def prepare_sample(self, seqs: list[Sequence]):
    temperatures = [seq.temperature for seq in seqs]
    temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
    return temperatures
```

#### 语法作用

从每条 seq 中取出 temperature，并转成 CUDA Tensor。

#### 工程作用

每条请求可以有自己的 temperature。

Tensor 形状：

```text
temperatures.shape = [num_seqs]
dtype = float32
device = cuda
```

#### 在推理流程中的意义

采样器使用 temperature 对 logits 做缩放：

```text
logits / temperature
```

然后再 softmax 和随机采样。

---

### 3.26 模型执行：`run_model`

```python
@torch.inference_mode()
def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
```

#### 语法作用

定义模型执行方法，并使用 `@torch.inference_mode()` 禁用梯度。

#### 工程作用

推理阶段不需要反向传播，所以禁用 autograd 可以减少显存和计算开销。

#### 参数类型

| 参数 | 类型 | 含义 |
|---|---|---|
| `input_ids` | `torch.Tensor[int64]` | 输入 token ids |
| `positions` | `torch.Tensor[int64]` | position ids |
| `is_prefill` | `bool` | 当前是否是 Prefill |

---

### 3.27 直接执行模型路径

```python
if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
    return self.model.compute_logits(self.model(input_ids, positions))
```

#### 语法作用

如果满足条件，则直接调用模型 forward。

#### 工程作用

以下场景使用普通 eager 执行：

1. Prefill 阶段；
2. 用户强制 `enforce_eager=True`；
3. Decode batch size 超过 512。

#### 为什么 Prefill 不用 CUDA Graph

Prefill 的输入长度变化很大：

```text
不同 prompt 长度不同
chunked prefill 长度不同
cu_seqlens 不同
slot_mapping 长度不同
```

CUDA Graph 更适合 shape 相对固定的 Decode 阶段。

#### logits 形状说明

`self.model(input_ids, positions)` 返回 hidden states。`compute_logits()` 把 hidden states 投影到词表维度。

在 Prefill 阶段，`ParallelLMHead` 会利用 context 选择每条 seq 的最后一个 query 位置来计算 logits，因此 sampler 最终拿到的 logits 通常是：

```text
[num_seqs, vocab_size]
```

在 Decode 阶段，输入本来就是每条 seq 一个 token，因此 logits 也是：

```text
[decode_batch_size, vocab_size]
```

---

### 3.28 CUDA Graph Decode 路径

```python
else:
    bs = input_ids.size(0)
    context = get_context()
    graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
    graph_vars = self.graph_vars
```

#### 语法作用

Decode 且不强制 eager，并且 batch size 不超过 512 时，使用 CUDA Graph。

#### 工程作用

选择一个已经捕获好的 graph。`self.graph_bs` 中保存可用 batch size，例如：

```text
[1, 2, 4, 8, 16, 32, 48, ...]
```

如果当前 `bs=20`，则选择第一个 `>=20` 的 graph，例如 32。

#### 在推理流程中的意义

Decode 阶段每轮都会执行很多次，小 batch、短输入、kernel launch 开销明显。CUDA Graph 可以减少 Python 调度和 kernel launch overhead，提高 TPOT 性能。

---

### 3.29 将当前输入拷贝到 graph 静态 Tensor

```python
graph_vars["input_ids"][:bs] = input_ids
graph_vars["positions"][:bs] = positions
graph_vars["slot_mapping"].fill_(-1)
graph_vars["slot_mapping"][:bs] = context.slot_mapping
graph_vars["context_lens"].zero_()
graph_vars["context_lens"][:bs] = context.context_lens
graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
graph.replay()
return self.model.compute_logits(graph_vars["outputs"][:bs])
```

#### 语法作用

把动态输入拷贝到 CUDA Graph 捕获时使用的静态 buffer 中，然后 replay graph。

#### 工程作用

CUDA Graph 要求执行时使用固定地址的 Tensor。这里的 `graph_vars` 就是提前准备好的静态 Tensor。

每轮 Decode 时不是重新创建新计算图，而是：

```text
把新数据复制进旧 buffer
   ↓
graph.replay()
   ↓
从 outputs buffer 取结果
```

#### 在推理流程中的意义

这能减少 Decode 阶段每轮的 CPU overhead，对高并发小 token 生成很重要。

---

### 3.30 一轮执行主入口：`run`

```python
def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    logits = self.run_model(input_ids, positions, is_prefill)
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    reset_context()
    return token_ids
```

#### 语法作用

定义每一轮模型执行的总入口。

#### 工程作用

完整流程：

```text
1. 根据 is_prefill 选择输入准备函数
2. rank 0 准备 temperature
3. 执行模型 forward 得到 logits
4. rank 0 采样 token
5. 清空 context
6. 返回 token_ids
```

#### 为什么只有 rank 0 采样

Tensor Parallel 下，最终 logits 会在 rank 0 聚合。其他 rank 只是参与分片计算，不负责最终采样。

所以：

```python
token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
```

rank 0 返回 `list[int]`，其他 rank 返回 `None`。

#### 在推理流程中的意义

`run()` 是 `LLMEngine.step()` 触发的核心方法。

它完成：

```text
Sequence batch → Tensor input → model forward → logits → sampled token ids
```

---

### 3.31 捕获 CUDA Graph：初始化静态 Tensor

```python
@torch.inference_mode()
def capture_cudagraph(self):
    config = self.config
    hf_config = config.hf_config
    max_bs = min(self.config.max_num_seqs, 512)
    max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hf_config.hidden_size)
```

#### 语法作用

定义 CUDA Graph 捕获函数，创建静态 buffer。

#### 工程作用

这些 Tensor 会在 graph replay 时反复复用。

Tensor 形状：

| Tensor | shape | dtype |
|---|---|---|
| `input_ids` | `[max_bs]` | int64 |
| `positions` | `[max_bs]` | int64 |
| `slot_mapping` | `[max_bs]` | int32 |
| `context_lens` | `[max_bs]` | int32 |
| `block_tables` | `[max_bs, max_num_blocks]` | int32 |
| `outputs` | `[max_bs, hidden_size]` | 默认 dtype |

#### 为什么只适合 Decode

Decode 每条 seq 一轮只有一个 token，所以 `input_ids` 是一维 `[batch_size]`。这非常适合 CUDA Graph。

---

### 3.32 graph_bs 和 graphs 字典

```python
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
self.graphs = {}
self.graph_pool = None
```

#### 工程作用

预先捕获多个 batch size 的图。

例如：

```text
1, 2, 4, 8, 16, 32, 48, ..., 512
```

当前 Decode batch size 如果是 37，就可以选择 48 的 graph。

#### 在推理流程中的意义

不同 decode step 的 running seq 数可能不同，因此需要多个 graph 覆盖不同 batch size。

---

### 3.33 捕获每个 batch size 的 graph

```python
for bs in reversed(self.graph_bs):
    graph = torch.cuda.CUDAGraph()
    set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
    with torch.cuda.graph(graph, self.graph_pool):
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
    if self.graph_pool is None:
        self.graph_pool = graph.pool()
    self.graphs[bs] = graph
    torch.cuda.synchronize()
    reset_context()
```

#### 语法作用

循环捕获不同 batch size 的 CUDA Graph。

#### 工程作用

每个 `bs` 捕获一个 Decode forward 图。

注意这里捕获的是：

```python
outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
```

也就是模型主体 forward，不包含 sampler。

#### 为什么 reversed

从大 batch size 到小 batch size 捕获，有助于 graph memory pool 复用。

#### 在推理流程中的意义

后续 `run_model()` 可以直接：

```python
graph.replay()
```

而不是每轮重新走 Python 调度。

---

### 3.34 保存 graph 静态变量

```python
self.graph_vars = dict(
    input_ids=input_ids,
    positions=positions,
    slot_mapping=slot_mapping,
    context_lens=context_lens,
    block_tables=block_tables,
    outputs=outputs,
)
```

#### 工程作用

保存 CUDA Graph replay 时要复用的静态 Tensor。

#### 在推理流程中的意义

`run_model()` 会将真实输入拷贝到这些 Tensor 中，然后 replay 对应 graph。

---

## 4. 背后的框架性原理

### 4.1 prompt / token / tokenizer

#### 是什么

prompt 是用户输入的文本；token 是模型内部处理的整数编号；tokenizer 负责文本和 token id 的转换。

#### 为什么重要

模型不能直接处理字符串，只能处理 token ids。

#### 在本文件中如何体现

`model_runner.py` 不负责 tokenizer。它接收的 `Sequence` 已经包含 token ids。

具体输入：

```python
input_ids
```

就是从 `Sequence.token_ids` 中截取出来的 token id Tensor。

#### 和整体架构关系

```text
LLMEngine.add_request()
    负责 tokenizer.encode()
ModelRunner.prepare_prefill/decode()
    负责把 token ids 转成 CUDA Tensor
```

---

### 4.2 request / sequence

#### 是什么

一条用户请求在引擎内部被表示成一个 `Sequence`。

#### 为什么重要

推理引擎需要跟踪每条请求的状态，包括：

```text
token_ids
last_token
num_cached_tokens
num_scheduled_tokens
block_table
temperature
max_tokens
```

#### 在本文件中如何体现

所有准备输入的方法都接收：

```python
seqs: list[Sequence]
```

`prepare_prefill()` 和 `prepare_decode()` 读取 `Sequence` 的状态，生成模型输入。

---

### 4.3 scheduler

#### 是什么

Scheduler 决定本轮执行哪些 Sequence，以及本轮是 Prefill 还是 Decode。

#### 为什么重要

推理引擎不是简单一条请求跑到底，而是动态调度多个请求，提高吞吐量。

#### 在本文件中如何体现

`ModelRunner` 不主动调度。它只执行 Scheduler 给它的：

```python
run(seqs, is_prefill)
```

#### 和整体架构关系

```text
Scheduler.schedule()
   ↓
ModelRunner.run(seqs, is_prefill)
```

---

### 4.4 prefill / decode

#### 是什么

- Prefill：处理 prompt，建立 KV Cache；
- Decode：每轮输入 last token，生成下一个 token。

#### 为什么重要

两者计算形态不同，优化方式也不同。

#### 在本文件中如何体现

```python
input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
```

Prefill 使用：

```text
cu_seqlens_q / cu_seqlens_k / max_seqlen_q / max_seqlen_k
```

Decode 使用：

```text
context_lens / block_tables / CUDA Graph
```

#### 和 vLLM 关系

Prefill / Decode 分离是 vLLM 推理引擎的核心设计之一。

---

### 4.5 KV Cache

#### 是什么

KV Cache 保存每层 Attention 中历史 token 的 Key 和 Value。

#### 为什么重要

没有 KV Cache，Decode 每生成一个 token 都要重新计算全部历史 token，成本极高。

#### 在本文件中如何体现

```python
self.kv_cache = torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
```

并将每层 Attention 的 `k_cache` / `v_cache` 指向其中一部分。

---

### 4.6 block / block table / block manager

#### 是什么

vLLM 把 KV Cache 按 block 分页管理，而不是给每条请求连续分配一整段显存。

#### 为什么重要

这样可以减少显存碎片，提高并发请求下的显存利用率。

#### 在本文件中如何体现

`ModelRunner` 不分配 block，但使用 `seq.block_table`：

```python
block_tables = self.prepare_block_tables(seqs)
```

并计算：

```python
slot_mapping
```

#### 和整体架构关系

```text
BlockManager 分配 block
   ↓
Sequence.block_table 保存映射
   ↓
ModelRunner 转成 block_tables Tensor
   ↓
Attention 使用 block_tables 读取 KV Cache
```

---

### 4.7 attention

#### 是什么

Attention 根据 Q/K/V 计算当前 token 对历史上下文的注意力。

#### 为什么重要

大模型生成质量和上下文建模都依赖 Attention。

#### 在本文件中如何体现

ModelRunner 通过 `set_context()` 向 Attention 传递：

```text
is_prefill
cu_seqlens_q
cu_seqlens_k
slot_mapping
context_lens
block_tables
```

Attention 层通过 `get_context()` 获取这些信息。

---

### 4.8 model runner

#### 是什么

ModelRunner 是模型执行器。

#### 为什么重要

它把调度层的 Sequence 转成 GPU 模型输入，并实际执行 forward。

#### 在本文件中如何体现

整个文件就是 `ModelRunner`。

核心方法：

```python
run()
run_model()
prepare_prefill()
prepare_decode()
allocate_kv_cache()
```

---

### 4.9 GPU 执行

#### 是什么

GPU 执行包括 Tensor 创建、CUDA 拷贝、模型 forward、CUDA Graph replay 等。

#### 在本文件中如何体现

```python
torch.cuda.set_device(rank)
torch.set_default_device("cuda")
.cuda(non_blocking=True)
torch.cuda.CUDAGraph()
graph.replay()
```

---

### 4.10 tensor parallel

#### 是什么

Tensor Parallel 是把模型权重按张量维度切到多张 GPU 上。

#### 为什么重要

模型太大时单卡放不下，或者需要多卡提升吞吐。

#### 在本文件中如何体现

```python
self.world_size = config.tensor_parallel_size
dist.init_process_group(...)
num_kv_heads = hf_config.num_key_value_heads // self.world_size
```

KV heads 也按 TP rank 切分。

---

### 4.11 logits

#### 是什么

logits 是模型对词表中每个 token 的原始打分。

#### 为什么重要

采样器根据 logits 决定下一个 token。

#### 在本文件中如何体现

```python
logits = self.run_model(input_ids, positions, is_prefill)
```

`run_model()` 内部调用：

```python
self.model.compute_logits(...)
```

---

### 4.12 sampling / temperature

#### 是什么

sampling 是从 logits 中选择下一个 token。temperature 控制随机性。

#### 为什么重要

不同 temperature 会改变输出的保守程度和随机程度。

#### 在本文件中如何体现

```python
temperatures = self.prepare_sample(seqs)
token_ids = self.sampler(logits, temperatures).tolist()
```

---

### 4.13 throughput / latency / TTFT / TPOT

#### 是什么

- throughput：单位时间生成 token 数；
- latency：请求总延迟；
- TTFT：Time To First Token，首 token 时间；
- TPOT：Time Per Output Token，每个输出 token 时间。

#### 在本文件中如何体现

`model_runner.py` 不直接统计这些指标，但它决定这些指标的性能基础：

- Prefill 性能影响 TTFT；
- Decode 性能影响 TPOT；
- KV Cache 和 CUDA Graph 影响吞吐；
- Tensor Parallel 影响大模型可运行性和并发性能。

---

## 5. 和 vLLM 原版设计的关系

### 5.1 连续批处理

`ModelRunner` 本身不调度请求，但它接收 Scheduler 动态选出来的 `seqs`。

这体现了 Continuous Batching 的执行端：

```text
不同请求可以在不同 step 动态加入 batch
ModelRunner 每轮只执行 Scheduler 给出的 batch
```

---

### 5.2 PagedAttention

PagedAttention 的核心是 KV Cache 分 block 管理。

本文件体现为：

```python
block_tables = self.prepare_block_tables(seqs)
slot_mapping.append(...)
```

`block_tables` 和 `slot_mapping` 是 PagedAttention 访问 KV Cache 的关键元数据。

---

### 5.3 KV Cache block 管理

本文件不负责 block 分配，但负责将 block 信息转成 GPU Tensor。

关系如下：

```text
BlockManager.allocate(seq)
   ↓
seq.block_table
   ↓
ModelRunner.prepare_block_tables()
   ↓
Attention 使用 block_tables
```

---

### 5.4 request / sequence 调度

本文件执行的是 Scheduler 已经调度出的 sequence batch。

```python
run(seqs, is_prefill)
```

这体现了 vLLM 中“调度和执行分离”的思想。

---

### 5.5 prefill / decode 分离

本文件非常明确地区分了：

```python
prepare_prefill()
prepare_decode()
```

并且只对 Decode 使用 CUDA Graph：

```python
if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
    eager path
else:
    CUDA Graph path
```

这符合大模型推理优化中的常见策略。

---

### 5.6 高吞吐推理服务

本文件体现的高吞吐设计包括：

1. KV Cache 预分配；
2. block table 间接寻址；
3. Prefix Cache 支持；
4. Tensor Parallel；
5. CUDA Graph；
6. pinned memory + non_blocking H2D copy；
7. Prefill / Decode 分离。

---

### 5.7 nano-vLLM 的简化点

相比原版 vLLM，这个文件做了很多简化：

1. 只支持特定模型结构 Qwen3；
2. 多进程通信用共享内存和 Event，比较轻量；
3. 没有复杂的 worker executor / RPC 框架；
4. CUDA Graph 只覆盖 Decode 的模型 forward；
5. 不包含复杂的异步调度和服务端队列；
6. 采样参数较少；
7. 没有完整的生产级错误处理和资源回收机制。

但这些简化使它非常适合学习推理引擎核心原理。

---

## 6. 总结：这个文件应该怎么掌握

`model_runner.py` 是 nano-vLLM 中难度较高的核心文件。你学习时应该抓住四条主线：

### 6.1 执行主线

```text
run()
   ↓
prepare_prefill() / prepare_decode()
   ↓
set_context()
   ↓
run_model()
   ↓
sampler()
   ↓
reset_context()
```

### 6.2 KV Cache 主线

```text
allocate_kv_cache()
   ↓
self.kv_cache = [K/V, layer, block, token, kv_head, head_dim]
   ↓
module.k_cache / module.v_cache 指向对应层
   ↓
slot_mapping 决定写入位置
   ↓
block_tables 决定读取位置
```

### 6.3 Prefill / Decode 主线

```text
Prefill:
    多 token 输入
    使用 cu_seqlens_q/k
    可支持 prefix cache

Decode:
    每条 seq 输入 last_token
    使用 context_lens + block_tables
    可使用 CUDA Graph
```

### 6.4 Tensor Parallel 主线

```text
rank 0 接收 LLMEngine 调用
   ↓
rank 0 通过共享内存通知其他 rank
   ↓
所有 rank 同步执行 run()
   ↓
rank 0 聚合 logits 并采样 token
```

一句话总结：

> `model_runner.py` 是 nano-vLLM 中把调度结果真正送进 GPU 模型执行的核心模块，它连接了 Sequence、block_table、KV Cache、Attention、Qwen3 模型、Sampler、Tensor Parallel 和 CUDA Graph，是理解推理引擎运行时的关键文件。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]
- 关联阅读：[[outputs/项目整理/专题-01-请求从输入到输出|专题-01-请求从输入到输出]]

%% 项目关联导航：结束 %%
