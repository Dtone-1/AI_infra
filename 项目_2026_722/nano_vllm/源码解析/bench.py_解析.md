# bench.py 源码逐行精读解析

> 文件：`bench.py`  
> 定位：nano-vLLM 的离线推理性能 Benchmark 脚本  
> 学习重点：并发请求、输入长度、输出长度、吞吐量、warmup、Prefill / Decode 性能压力

---

# 1. 文件整体定位

## 1.1 这个文件负责什么

`bench.py` 是 nano-vLLM 仓库中的**性能测试入口文件**，它不是模型结构文件，也不是调度器或 KV Cache 的核心实现文件，而是一个用于压测推理引擎吞吐量的脚本。

它主要做 5 件事：

1. 构造大量随机 prompt token ids；
2. 为每条请求构造不同的 `SamplingParams`；
3. 初始化 nano-vLLM 的 `LLM` 推理引擎；
4. 先执行一次 warmup 推理；
5. 正式调用 `llm.generate()`，统计总生成 token 数、耗时和吞吐量。

简化理解：

```text
bench.py = 用随机 token 构造大量请求，然后测试 nano-vLLM 一次能生成多快
```

---

## 1.2 它属于哪一层

它属于：

```text
Benchmark 层 / 性能测试层 / 用户调用层
```

它不直接实现：

```text
Scheduler
Sequence
BlockManager
ModelRunner
Attention
KV Cache
Sampler
```

但它会通过 `LLM.generate()` 间接触发这些模块。

---

## 1.3 它和哪些文件有关

`bench.py` 的调用链大致是：

```text
bench.py
  ↓
nanovllm/__init__.py
  ↓
nanovllm/llm.py
  ↓
nanovllm/engine/llm_engine.py
  ↓
nanovllm/engine/sequence.py
  ↓
nanovllm/engine/scheduler.py
  ↓
nanovllm/engine/block_manager.py
  ↓
nanovllm/engine/model_runner.py
  ↓
nanovllm/models/qwen3.py
  ↓
nanovllm/layers/attention.py
  ↓
nanovllm/layers/sampler.py
```

其中最相关的文件是：

| 文件 | 关系 |
|---|---|
| `nanovllm/llm.py` | 暴露 `LLM` 用户 API |
| `nanovllm/sampling_params.py` | 定义 `SamplingParams` |
| `engine/llm_engine.py` | `generate()` 主流程 |
| `engine/scheduler.py` | 多请求调度、Prefill / Decode |
| `engine/sequence.py` | 每条请求的内部状态 |
| `engine/block_manager.py` | KV Cache block 分配 |
| `engine/model_runner.py` | 模型执行、KV Cache 分配、CUDA Graph |
| `layers/attention.py` | FlashAttention 和 KV Cache 读写 |
| `layers/sampler.py` | 根据 logits 采样下一个 token |

---

## 1.4 它在完整推理流程中的位置

完整推理链路如下：

```text
bench.py 构造随机 prompt_token_ids
   ↓
bench.py 构造 sampling_params
   ↓
LLM.generate(prompt_token_ids, sampling_params)
   ↓
LLMEngine.add_request()
   ↓
Sequence 保存 token_ids 和生成参数
   ↓
Scheduler 调度 waiting / running 请求
   ↓
BlockManager 分配 KV Cache blocks
   ↓
ModelRunner 执行 Prefill / Decode
   ↓
Attention 写入 / 读取 KV Cache
   ↓
Sampler 根据 logits 采样 token
   ↓
循环 Decode，直到每条请求达到 max_tokens
   ↓
bench.py 统计 throughput
```

所以，`bench.py` 的核心意义是：

> 它把 nano-vLLM 当成一个黑盒推理系统，用大量请求和长输入 / 长输出测试系统吞吐能力。

---

# 2. 代码结构总览

## 2.1 文件导入了哪些模块

```python
import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams
```

导入内容可以分成三类：

| 导入 | 类型 | 作用 |
|---|---|---|
| `os` | Python 标准库 | 处理本地模型路径 |
| `time` | Python 标准库 | 统计 benchmark 耗时 |
| `randint`, `seed` | Python 标准库 `random` | 构造随机输入长度、随机 token、随机输出长度 |
| `LLM`, `SamplingParams` | nano-vLLM 用户 API | 创建推理引擎和采样参数 |
| `vllm` 注释行 | 对比测试入口 | 可以切换到原版 vLLM 做对照 |

---

## 2.2 定义了哪些类

该文件没有定义类。

---

## 2.3 定义了哪些函数

该文件只定义了一个函数：

```python
def main():
```

`main()` 中包含完整 benchmark 流程。

---

## 2.4 重要变量和数据结构

| 变量 | 类型 | 作用 |
|---|---|---|
| `num_seqs` | `int` | 请求数量 / 并发 sequence 数 |
| `max_input_len` | `int` | 随机输入 token 长度上限 |
| `max_ouput_len` | `int` | 随机输出 token 长度上限，变量名中 `ouput` 疑似拼写错误 |
| `path` | `str` | 本地模型目录 |
| `llm` | `LLM` | nano-VLLM 推理引擎对象 |
| `prompt_token_ids` | `list[list[int]]` | 每条请求的输入 token ids |
| `sampling_params` | `list[SamplingParams]` | 每条请求的生成参数 |
| `t` | `float` | 计时变量 |
| `total_tokens` | `int` | 总生成 token 数 |
| `throughput` | `float` | 输出 token 吞吐量，单位 tok/s |

---

## 2.5 主流程和辅助逻辑

主流程：

```text
1. 固定随机种子
2. 设置 benchmark 参数
3. 初始化 LLM
4. 构造随机 prompt_token_ids
5. 构造每条请求的 SamplingParams
6. warmup
7. 正式计时 generate
8. 统计 throughput
9. 打印结果
```

辅助逻辑：

```text
os.path.expanduser()
seed(0)
randint()
time.time()
sum()
print()
```

---

## 2.6 文件组织结构图

```text
bench.py
├── import 区域
│   ├── os
│   ├── time
│   ├── random.randint / random.seed
│   ├── nanovllm.LLM
│   └── nanovllm.SamplingParams
│
├── main()
│   ├── 设置随机种子
│   ├── 设置压测参数
│   ├── 初始化模型路径
│   ├── 创建 LLM 引擎
│   ├── 构造随机 prompt_token_ids
│   ├── 构造 sampling_params
│   ├── warmup 推理
│   ├── 正式推理计时
│   ├── 计算 throughput
│   └── 打印结果
│
└── if __name__ == "__main__"
    └── main()
```

---

# 3. 逐行代码解释

## 3.1 导入模块

```python
import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams
```

### 语法作用

- `import os`：导入操作系统路径相关功能。
- `import time`：导入计时函数。
- `from random import randint, seed`：从 Python 随机数库中导入整数随机函数和随机种子函数。
- `from nanovllm import LLM, SamplingParams`：从 nano-vLLM 包中导入推理入口类和采样参数类。
- 注释掉的 `from vllm import LLM, SamplingParams` 表示可以把 nano-vLLM 替换为原版 vLLM 做性能对照。

### 工程作用

这几行决定了这个文件的用途：

```text
os      → 找模型目录
time    → 测时间
random  → 构造随机压测请求
LLM     → 启动推理引擎
SamplingParams → 控制每条请求的输出长度和采样行为
```

### 推理流程意义

`bench.py` 不使用真实 tokenizer，而是直接构造 `prompt_token_ids`。这说明这个文件测试的重点不是文本预处理，而是推理引擎本身：

```text
调度能力
Prefill 能力
Decode 能力
KV Cache 管理能力
吞吐量
```

---

## 3.2 定义 main 函数

```python
def main():
```

### 语法作用

定义主函数，封装整个 benchmark 流程。

### 工程作用

便于通过：

```python
if __name__ == "__main__":
    main()
```

控制脚本入口。

### 推理流程意义

所有推理压测逻辑都在 `main()` 内部执行。

---

## 3.3 固定随机种子

```python
    seed(0)
```

### 语法作用

设置 Python `random` 模块的随机种子为 `0`。

### 工程作用

让下面通过 `randint()` 生成的随机输入长度、随机 token id、随机输出长度具有可复现性。

也就是说，每次运行时：

```text
随机 prompt 长度基本一致
随机 prompt token 内容基本一致
随机 max_tokens 基本一致
```

### 注意点

这里只固定了 Python 标准库 `random` 的种子，没有固定 PyTorch / CUDA 的随机种子。

因此它保证的是：

```text
benchmark 输入请求形态可复现
```

但不一定保证：

```text
模型采样结果完全可复现
```

如果要进一步保证模型采样结果可复现，通常还需要设置：

```python
torch.manual_seed(...)
torch.cuda.manual_seed_all(...)
```

不过对吞吐量测试来说，真正重要的是请求长度分布可复现。

---

## 3.4 设置请求数量

```python
    num_seqs = 256
```

### 语法作用

定义整数变量 `num_seqs`，值为 `256`。

### 工程作用

表示本次 benchmark 构造 256 条请求。

可以理解为：

```text
num_seqs = 请求数量 = sequence 数量
```

### 在推理系统中的意义

每一条请求进入引擎后，通常会被封装成一个 `Sequence`。

所以这里大致对应：

```text
256 条 prompt
   ↓
256 个 Sequence
   ↓
Scheduler 调度 256 个请求
```

这会测试：

```text
Scheduler 的批处理能力
BlockManager 的 KV Cache 分配能力
ModelRunner 的 batch 执行能力
Decode 阶段的持续批处理能力
```

---

## 3.5 设置最大输入长度

```python
    max_input_len = 1024
```

### 语法作用

定义最大输入 token 长度为 1024。

### 工程作用

后面每条请求的输入长度会随机落在：

```text
100 ~ 1024 token
```

### 推理流程意义

输入长度主要影响 Prefill 阶段。

Prefill 阶段需要一次性处理 prompt 中的所有 token：

```text
输入越长
  ↓
Prefill 计算量越大
  ↓
KV Cache 初始占用越大
  ↓
TTFT 通常越高
```

在大模型推理中，长 prompt 会显著增加 Prefill 压力。

---

## 3.6 设置最大输出长度

```python
    max_ouput_len = 1024
```

### 语法作用

定义最大输出 token 长度为 1024。

### 工程作用

后面每条请求的 `max_tokens` 会随机落在：

```text
100 ~ 1024 token
```

### 注意：变量名疑似拼写错误

这里变量名写成了：

```python
max_ouput_len
```

更标准的写法应该是：

```python
max_output_len
```

但因为后续代码也使用了同一个变量名，所以程序可以正常运行。

这是一个拼写问题，不是运行错误。

### 推理流程意义

输出长度主要影响 Decode 阶段。

Decode 的特点是：

```text
每条请求每轮通常生成 1 个 token
生成 1024 个 token 就需要约 1024 轮 decode
```

所以输出长度越大，Decode 阶段越长，越能体现推理引擎的持续批处理能力和 KV Cache 访问效率。

---

## 3.7 设置模型路径

```python
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
```

### 语法作用

将 `~` 展开成当前用户的 home 目录。

例如：

```text
~/huggingface/Qwen3-0.6B/
```

可能变成：

```text
/home/username/huggingface/Qwen3-0.6B/
```

### 工程作用

指定本地 HuggingFace 格式模型目录。

该目录通常应该包含：

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
generation_config.json
```

### 推理流程意义

后续 `LLM(path, ...)` 会根据这个路径：

```text
读取模型配置
加载模型权重
初始化 tokenizer
初始化 Qwen3ForCausalLM
分配 KV Cache
```

---

## 3.8 初始化 LLM 推理引擎

```python
    llm = LLM(path, enforce_eager=False, max_model_len=4096)
```

### 语法作用

创建 `LLM` 对象，传入模型路径和配置参数。

### 参数解释

| 参数 | 含义 |
|---|---|
| `path` | 模型目录 |
| `enforce_eager=False` | 不强制 eager mode，允许使用优化路径 |
| `max_model_len=4096` | 模型最大上下文长度设置为 4096 token |

### 工程作用

这一步会初始化整个推理引擎。

内部大致包括：

```text
创建 Config
加载 HuggingFace config
初始化 tokenizer
初始化 Scheduler
初始化 ModelRunner
加载模型权重
分配 KV Cache
准备 CUDA Graph / torch.compile 等优化
```

### 为什么 benchmark 中使用 `enforce_eager=False`

在 `example.py` 这种学习入口中，常见写法是：

```python
enforce_eager=True
```

这样更方便调试。

但在 benchmark 中，代码使用：

```python
enforce_eager=False
```

表示允许使用更高性能的路径，例如：

```text
CUDA Graph
torch.compile
预热后的静态 decode graph
更低 Python 调度开销
```

因此这个设置更适合测性能。

### `max_model_len=4096` 的意义

每条请求的总长度大致满足：

```text
prompt length + generated length <= max_model_len
```

本文件中：

```text
prompt 最大 1024
output 最大 1024
理论最大总长约 2048
```

所以设置 `4096` 是安全的。

它也会影响 KV Cache 的规划，因为引擎需要为最大上下文长度和并发请求预留显存资源。

---

## 3.9 构造随机 prompt_token_ids

```python
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
```

### 语法作用

这是一个嵌套列表推导式。

可以拆开理解为：

```python
prompt_token_ids = []

for _ in range(num_seqs):
    input_len = randint(100, max_input_len)
    one_prompt = []

    for _ in range(input_len):
        token_id = randint(0, 10000)
        one_prompt.append(token_id)

    prompt_token_ids.append(one_prompt)
```

最终得到的数据结构是：

```python
[
    [token_id, token_id, token_id, ...],
    [token_id, token_id, token_id, ...],
    ...
]
```

共有 `num_seqs = 256` 个子列表。

### 数据结构形状

不是标准矩阵，因为每条 prompt 长度不同。

可以理解为：

```text
prompt_token_ids: list[list[int]]

第 0 条请求: 长度可能是 964
第 1 条请求: 长度可能是 318
第 2 条请求: 长度可能是 1020
...
第 255 条请求: 长度可能是 701
```

### 工程作用

这行代码直接构造 token ids，而不是构造自然语言文本。

这样做有几个好处：

1. 避免 tokenizer 成为 benchmark 的性能干扰项；
2. 可以直接控制输入 token 长度；
3. 可以模拟不同长度请求；
4. 更适合测推理引擎吞吐量。

### 为什么随机 token id 是 `0 ~ 10000`

这是假设模型词表大小远大于 10000。

Qwen 系列模型词表通常很大，因此 `0 ~ 10000` 一般是合法 token id 范围。

不过从严格角度说，最稳妥的写法应该根据模型配置读取 `vocab_size`，然后生成：

```python
randint(0, vocab_size - 1)
```

这里写 `10000` 是简化 benchmark。

### 在 nano-vLLM 推理流程中的意义

因为输入已经是 token ids，所以 `LLM.generate()` 内部可以跳过文本 tokenize 过程，直接把 token ids 封装为 `Sequence`。

这意味着 benchmark 更聚焦于：

```text
Sequence 管理
Prefill 计算
KV Cache 写入
Decode 计算
Sampler
吞吐统计
```

而不是 tokenizer 性能。

---

## 3.10 构造每条请求的 SamplingParams

```python
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
```

### 语法作用

这是一个列表推导式，生成 256 个 `SamplingParams` 对象。

等价于：

```python
sampling_params = []

for _ in range(num_seqs):
    sp = SamplingParams(
        temperature=0.6,
        ignore_eos=True,
        max_tokens=randint(100, max_ouput_len),
    )
    sampling_params.append(sp)
```

### 数据结构

最终：

```text
sampling_params: list[SamplingParams]
```

每条请求有自己的生成参数。

例如：

```text
Sequence 0: max_tokens = 583
Sequence 1: max_tokens = 1001
Sequence 2: max_tokens = 247
...
```

### 参数解释

#### `temperature=0.6`

控制采样随机性。

推理时模型输出 logits：

```text
logits → temperature 缩放 → softmax → 采样 token
```

temperature 越低，概率分布越尖锐，输出越稳定。

#### `ignore_eos=True`

表示即使模型生成了 EOS token，也不要提前停止。

这对 benchmark 非常重要。

如果不设置 `ignore_eos=True`，有些请求可能提前结束：

```text
请求 A 计划生成 1024 token，但生成 80 token 后遇到 EOS
请求 A 提前结束
```

这样会导致每次 benchmark 实际生成 token 数不稳定。

设置 `ignore_eos=True` 后：

```text
每条请求都会尽量生成到 max_tokens
```

这样吞吐统计更可控。

#### `max_tokens=randint(100, max_ouput_len)`

每条请求的输出长度随机在：

```text
100 ~ 1024
```

这模拟真实服务场景中不同用户请求有不同生成长度。

### 推理流程意义

不同 `max_tokens` 会让不同 sequence 在 Decode 阶段不同时间结束：

```text
Sequence A 生成 120 token 后结束
Sequence B 生成 900 token 后仍在 running
Sequence C 生成 1024 token 后结束
```

这会考验 scheduler 的动态调度能力。

在真实 vLLM 中，这对应 continuous batching 场景：

```text
有的请求结束
释放 KV Cache
新的请求进入 batch
GPU 尽量保持忙碌
```

nano-VLLM 的 benchmark 虽然是离线固定请求集，但仍能体现多 sequence 不同长度的调度压力。

---

## 3.11 vLLM 对照测试提示

```python
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]
```

### 语法作用

这是注释代码，不会执行。

如果改用原版 vLLM，需要把 `prompt_token_ids` 包装成字典列表：

```python
prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]
```

### 工程作用

这说明作者希望这个 benchmark 能在 nano-VLLM 和原版 vLLM 之间切换。

nano-VLLM 可能支持直接传：

```python
list[list[int]]
```

而原版 vLLM 的接口可能需要：

```python
[
    {"prompt_token_ids": [...]},
    {"prompt_token_ids": [...]},
]
```

### 推理流程意义

这体现了同一个 benchmark 可以用于横向比较：

```text
nano-vLLM throughput
vs
vLLM throughput
```

但需要注意：公平比较时应尽量保持相同配置，例如：

```text
模型相同
dtype 相同
GPU 相同
max_model_len 相同
tensor_parallel_size 相同
采样策略相同
输入输出长度分布相同
```

---

## 3.12 warmup 推理

```python
    llm.generate(["Benchmark: "], SamplingParams())
```

### 语法作用

调用一次 `llm.generate()`，输入一个简单字符串 prompt 和默认采样参数。

### 工程作用

这行是 warmup。

warmup 的目的不是测结果，而是让系统提前完成一些首次运行开销，例如：

```text
CUDA kernel 初始化
模型首次 forward
torch.compile 编译
CUDA Graph 捕获
显存缓存分配
FlashAttention 内部初始化
PyTorch CUDA context 初始化
```

### 为什么 benchmark 前必须 warmup

如果不 warmup，第一次正式计时会混入大量初始化开销。

例如：

```text
第一次 generate:
  模型真正第一次跑
  CUDA kernel 第一次加载
  graph 第一次 capture
  编译第一次触发
  显存 allocator 第一次申请

后续 generate:
  这些开销显著减少
```

所以 benchmark 一般写法都是：

```text
先 warmup，不计时
再正式推理，计时
```

### 在 nano-vLLM 推理流程中的意义

如果 `enforce_eager=False`，warmup 尤其重要。

因为非 eager 模式可能启用：

```text
CUDA Graph
torch.compile
固定 batch size 图捕获
decode 路径优化
```

这些优化通常需要第一次运行时准备。

---

## 3.13 开始计时

```python
    t = time.time()
```

### 语法作用

记录当前时间戳。

### 工程作用

作为 benchmark 正式计时的起点。

`time.time()` 返回当前 Unix 时间，单位是秒。

例如：

```text
1718000000.12345
```

---

## 3.14 正式执行 benchmark

```python
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
```

### 语法作用

调用 `llm.generate()`，传入：

1. 256 条随机 prompt token ids；
2. 256 个采样参数对象；
3. `use_tqdm=False` 关闭进度条。

### 工程作用

这是 benchmark 的核心执行语句。

它会触发 nano-VLLM 完整推理流程：

```text
1. 接收 prompt_token_ids
2. 为每条请求创建 Sequence
3. Scheduler 把请求放入 waiting 队列
4. Prefill 阶段处理所有 prompt
5. BlockManager 分配 KV Cache block
6. Attention 写入 prompt 的 K/V Cache
7. Decode 阶段逐 token 生成
8. Sampler 采样下一个 token
9. Scheduler 更新每条请求状态
10. 请求达到 max_tokens 后释放资源
```

### 为什么 `use_tqdm=False`

`tqdm` 是 Python 进度条库。

如果打开进度条，会产生额外的输出刷新开销。

benchmark 中关闭它，是为了让耗时更接近纯推理耗时。

### 推理系统意义

这一行代码会同时测试：

```text
Prefill 性能：
  长 prompt 的批量计算能力

Decode 性能：
  多请求逐 token 生成能力

KV Cache 性能：
  block 分配、写入、读取、释放

Scheduler 性能：
  多 sequence 状态管理

Sampler 性能：
  logits 转 token 的采样开销
```

---

## 3.15 计算耗时

```python
    t = (time.time() - t)
```

### 语法作用

用当前时间减去开始时间，得到 benchmark 总耗时。

### 工程作用

`t` 从开始时间戳变成了耗时秒数。

例如：

```text
t = 24.37
```

表示正式推理耗时 24.37 秒。

### 注意点

这个耗时包括：

```text
Prefill 时间
Decode 时间
Scheduler 时间
BlockManager 时间
Sampler 时间
Python 调度开销
```

不包括前面的 warmup 时间。

---

## 3.16 统计总生成 token 数

```python
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
```

### 语法作用

遍历所有 `SamplingParams`，把每条请求的 `max_tokens` 相加。

等价于：

```python
total_tokens = 0
for sp in sampling_params:
    total_tokens += sp.max_tokens
```

### 工程作用

因为前面设置了：

```python
ignore_eos=True
```

所以理论上每条请求都会生成 `sp.max_tokens` 个 token。

因此总生成 token 数可以直接用：

```text
sum(max_tokens)
```

### 注意：这里只统计输出 token

这里的 `total_tokens` 是：

```text
总输出 token 数
```

它没有统计输入 prompt token 数。

也就是说：

```text
total_tokens = completion tokens
```

不是：

```text
prompt tokens + completion tokens
```

因此后面算出来的 throughput 是：

```text
输出 token 吞吐量 tok/s
```

不是总 token 吞吐量。

---

## 3.17 计算吞吐量

```python
    throughput = total_tokens / t
```

### 语法作用

计算平均每秒生成多少 token。

### 工程作用

得到 benchmark 的核心指标：

```text
throughput = total output tokens / elapsed seconds
```

单位：

```text
tok/s
```

### 推理性能意义

吞吐量越高，说明推理系统在同样时间内能生成更多 token。

在大模型推理系统中，常见指标包括：

| 指标 | 含义 |
|---|---|
| Throughput | 每秒输出 token 数 |
| Latency | 单个请求整体耗时 |
| TTFT | Time To First Token，首 token 延迟 |
| TPOT | Time Per Output Token，平均每个输出 token 时间 |
| GPU utilization | GPU 利用率 |
| Memory usage | 显存占用 |

这个脚本只统计了最简单的：

```text
总体输出 token throughput
```

没有单独统计 TTFT / TPOT。

---

## 3.18 打印结果

```python
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
```

### 语法作用

使用 f-string 格式化输出结果。

`{t:.2f}` 表示保留两位小数。

### 工程作用

输出格式类似：

```text
Total: 142337tok, Time: 31.25s, Throughput: 4554.78tok/s
```

### 推理系统意义

这是 benchmark 的最终结果。

但要注意，它是宏观吞吐量，不包含更多细粒度分析。

如果你想更深入分析 AI Infra 推理性能，后续可以扩展统计：

```text
Prefill tokens/s
Decode tokens/s
TTFT
TPOT
每个请求 latency
峰值显存
不同并发数下 throughput 曲线
不同输入长度下 TTFT 曲线
不同输出长度下 TPOT 曲线
```

---

## 3.19 脚本入口

```python
if __name__ == "__main__":
    main()
```

### 语法作用

判断当前文件是否作为主脚本运行。

如果直接执行：

```bash
python bench.py
```

则调用：

```python
main()
```

如果被其他文件导入：

```python
import bench
```

则不会自动执行 benchmark。

### 工程作用

这是 Python 项目常见入口写法。

---

# 4. 背后的框架性原理

## 4.1 prompt / token / tokenizer

### 是什么

prompt 是用户输入给模型的内容。

token 是模型真正处理的整数编号。

tokenizer 负责：

```text
自然语言文本 ↔ token ids
```

### 为什么重要

大模型内部不能直接处理字符串，只能处理 token ids。

### 在本文件中如何体现

`bench.py` 没有用 tokenizer，而是直接构造：

```python
prompt_token_ids = [[randint(0, 10000) ...] ...]
```

这说明它绕过了文本到 token 的转换过程。

### 和 nano-vLLM / vLLM 的关系

推理引擎内部最核心的对象通常都是 token ids，而不是字符串。

因此 benchmark 直接传 token ids，可以更准确测试推理引擎本身，而不是 tokenizer。

---

## 4.2 request / sequence

### 是什么

request 是用户的一次生成请求。

sequence 是推理引擎内部表示请求状态的数据结构。

### 为什么重要

每条请求在生成过程中都有状态：

```text
当前 token_ids
已经生成多少 token
是否结束
占用了哪些 KV Cache block
当前处于 waiting / running / finished
```

这些都需要由 sequence 管理。

### 在本文件中如何体现

本文件构造了：

```python
num_seqs = 256
```

表示 256 条请求。

进入 `LLM.generate()` 后，每条请求会被包装成一个 sequence。

### 和整体架构的关系

Scheduler 调度的不是字符串，而是 sequence。

BlockManager 也是根据 sequence 分配 KV Cache。

---

## 4.3 scheduler

### 是什么

Scheduler 是推理引擎中的调度器。

它决定：

```text
这一轮执行哪些请求
哪些请求做 Prefill
哪些请求做 Decode
哪些请求结束
哪些请求释放 KV Cache
```

### 为什么重要

在多请求推理中，不同请求长度不同、结束时间不同。如果没有调度器，GPU 很容易空转或者显存被浪费。

### 在本文件中如何体现

本文件通过：

```python
num_seqs = 256
```

和随机输出长度制造调度压力。

### 和整体架构的关系

vLLM 的高吞吐能力很大程度来自 continuous batching，而 continuous batching 的核心就是 scheduler。

---

## 4.4 prefill / decode

### 是什么

Prefill：处理输入 prompt，计算并写入 KV Cache。  
Decode：逐 token 生成输出，复用历史 KV Cache。

### 为什么重要

两者性能特征不同：

| 阶段 | 特点 |
|---|---|
| Prefill | 一次处理很多 token，计算密集 |
| Decode | 每轮每请求生成 1 个 token，访存和调度开销明显 |

### 在本文件中如何体现

随机输入长度：

```python
randint(100, max_input_len)
```

主要影响 Prefill。

随机输出长度：

```python
randint(100, max_ouput_len)
```

主要影响 Decode。

### 和整体架构的关系

nano-vLLM 的 Scheduler / ModelRunner 需要根据 Prefill 和 Decode 使用不同的准备逻辑和 attention kernel。

---

## 4.5 KV Cache

### 是什么

KV Cache 保存每层 attention 中历史 token 的 Key 和 Value。

### 为什么重要

没有 KV Cache，Decode 每生成一个 token 都要重新计算整个上下文，成本极高。

有 KV Cache 后：

```text
历史 K/V 复用
只计算新 token
```

### 在本文件中如何体现

本文件设置：

```python
max_model_len=4096
num_seqs=256
max_input_len=1024
max_ouput_len=1024
```

这些参数都会影响 KV Cache 需求。

### 和整体架构的关系

BlockManager 负责分配 KV Cache block，Attention 负责读写 KV Cache，Scheduler 负责在请求结束后释放 KV Cache。

---

## 4.6 block / block table / block manager

### 是什么

vLLM 不直接为每条请求分配一整段连续 KV Cache，而是把 KV Cache 拆成固定大小的 block。

每条 sequence 有一个 block table，记录逻辑 block 到物理 block 的映射。

### 为什么重要

这种方式可以减少显存碎片，提高 KV Cache 利用率。

### 在本文件中如何体现

`bench.py` 没有直接操作 block，但它制造了大量不同长度请求，间接测试 BlockManager 的能力：

```text
不同输入长度
不同输出长度
256 条请求
长时间 Decode
```

### 和整体架构的关系

PagedAttention 的基础就是 block table。nano-VLLM 的 `block_manager.py` 和 `attention.py` 会使用这些结构。

---

## 4.7 model runner / GPU 执行

### 是什么

ModelRunner 是真正把 batch 送入 GPU 执行模型 forward 的模块。

### 为什么重要

它负责：

```text
准备 input_ids
准备 positions
准备 block_tables
准备 slot_mapping
调用模型 forward
调用 sampler
管理 CUDA Graph
```

### 在本文件中如何体现

`bench.py` 通过：

```python
enforce_eager=False
```

允许 ModelRunner 使用更高性能执行路径。

### 和整体架构的关系

Scheduler 决定“跑哪些请求”，ModelRunner 决定“怎么在 GPU 上跑”。

---

## 4.8 logits / sampling / temperature

### 是什么

模型每一步输出 logits，表示词表中每个 token 的原始分数。

Sampler 根据 logits 采样出下一个 token。

temperature 控制采样分布的尖锐程度。

### 为什么重要

采样策略会影响输出内容，也会影响 benchmark 是否稳定。

### 在本文件中如何体现

```python
SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=...)
```

这里设置 temperature 为 0.6，并通过 `ignore_eos=True` 保证生成长度接近 `max_tokens`。

### 和整体架构的关系

Sampler 是推理链路末端模块：

```text
hidden states
  ↓
lm_head
  ↓
logits
  ↓
sampler
  ↓
next token
```

---

## 4.9 throughput / latency / TTFT / TPOT

### 是什么

| 指标 | 含义 |
|---|---|
| Throughput | 单位时间生成 token 数 |
| Latency | 单请求总延迟 |
| TTFT | 首 token 延迟 |
| TPOT | 每个输出 token 平均耗时 |

### 为什么重要

不同指标对应不同优化目标。

例如：

```text
聊天机器人更关心 TTFT
批量离线生成更关心 Throughput
长文本生成更关心 TPOT
```

### 在本文件中如何体现

本文件只计算：

```python
throughput = total_tokens / t
```

它没有细分 TTFT / TPOT。

### 和整体架构的关系

vLLM / nano-VLLM 的目标之一是提升高并发吞吐量，而这个脚本正是用来验证吞吐量的。

---

# 5. 和 vLLM 原版设计的关系

## 5.1 连续批处理

本文件一次提交 256 条请求，并且每条请求输入输出长度不同。

这对应连续批处理的典型场景：

```text
不同请求长度不同
不同请求完成时间不同
调度器需要动态维护 batch
```

nano-VLLM 可能没有完整在线服务场景下的动态请求注入，但这个 benchmark 仍能测试多 sequence 批处理能力。

---

## 5.2 PagedAttention

本文件没有直接出现 PagedAttention，但通过大量变长请求间接触发 KV Cache block 管理。

PagedAttention 的核心是：

```text
逻辑上连续的 token 序列
底层映射到非连续 KV Cache block
```

这种设计适合处理本文件中的变长请求。

---

## 5.3 KV Cache block 管理

随机输入长度和随机输出长度会造成不同 sequence 需要不同数量的 KV Cache block。

这会测试：

```text
block 分配
block 追加
block 释放
block table 更新
```

这些逻辑主要在：

```text
engine/block_manager.py
engine/scheduler.py
layers/attention.py
```

---

## 5.4 request / sequence 调度

`num_seqs=256` 是对调度器的压力测试。

每条请求都会进入：

```text
WAITING → RUNNING → FINISHED
```

随机输出长度会让不同请求在不同时间结束。

---

## 5.5 prefill / decode 分离

本文件同时设置了较长输入和较长输出：

```python
max_input_len = 1024
max_ouput_len = 1024
```

这意味着 benchmark 同时覆盖：

```text
Prefill 压力
Decode 压力
```

如果只测短输入长输出，主要测 Decode。  
如果只测长输入短输出，主要测 Prefill。  
当前设置二者都有。

---

## 5.6 高吞吐推理服务

本文件的最终指标是：

```text
Throughput tok/s
```

这正是高吞吐离线推理服务最关心的指标之一。

不过它是一个简化 benchmark，不是完整线上服务压测。

---

## 5.7 nano-VLLM 相比原版 vLLM 的简化

从这个 benchmark 可以看出，nano-VLLM 更偏教学和源码学习，可能简化了：

```text
OpenAI-compatible Server
复杂请求队列
在线动态请求注入
复杂采样参数 top_p / top_k / penalty
Prefix cache 管理策略
多租户调度
Metrics 体系
完整 tracing
生产级异常处理
```

但它保留了最重要的推理引擎骨架：

```text
LLM API
Sequence
Scheduler
KV Cache block
ModelRunner
Prefill / Decode
Attention
Sampler
Benchmark
```

---

# 6. 这个文件的学习重点

你读 `bench.py` 时，不要只把它当作一个“测速脚本”。

它真正对应 AI Infra 推理方向中的这些问题：

```text
1. 如何构造可控的压测请求？
2. 为什么要直接传 prompt_token_ids？
3. 为什么要设置不同输入长度和输出长度？
4. 为什么 benchmark 前要 warmup？
5. 为什么要 ignore_eos=True？
6. 为什么 throughput 只统计输出 token？
7. 为什么 enforce_eager=False 更适合性能测试？
8. 为什么这个脚本能间接测试 Scheduler / KV Cache / ModelRunner？
```

最关键的一句话：

```text
bench.py 不是推理引擎核心实现，但它是观察推理引擎性能表现的入口。
```

---

# 7. 建议下一步阅读

读完 `bench.py` 后，建议按下面顺序继续：

```text
1. nanovllm/sampling_params.py
   理解 SamplingParams 到底保存哪些生成参数。

2. nanovllm/llm.py
   理解 LLM 类如何继承 LLMEngine。

3. engine/llm_engine.py
   精读 generate()、step()、add_request()，这是主流程核心。

4. engine/sequence.py
   理解每条请求如何在内部表示。

5. engine/scheduler.py
   理解 256 条请求如何被调度。

6. engine/block_manager.py
   理解这些请求如何分配 KV Cache blocks。

7. engine/model_runner.py
   理解 enforce_eager=False 对模型执行路径的影响。

8. layers/attention.py
   理解 Prefill / Decode 的 attention kernel 和 KV Cache 读写。
```

---

# 8. 用一句话总结 bench.py

`bench.py` 通过构造 256 条随机 token 请求，使用不同输入长度和输出长度压测 nano-VLLM 的离线推理吞吐量，重点观察 Scheduler、KV Cache、ModelRunner、Prefill / Decode 和采样流程在高并发长序列场景下的整体性能表现。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]

%% 项目关联导航：结束 %%
