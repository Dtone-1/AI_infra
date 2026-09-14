# example.py 源码解析

> 解析对象：`example.py`  
> 文件类型：nano-vLLM 使用示例 / 推理入口脚本  
> 学习重点：从用户侧 API 理解一次 prompt 如何进入 nano-vLLM 推理引擎，并最终生成 completion。

---

## 1. 文件整体定位

### 1.1 这个文件负责什么

`example.py` 是 nano-vLLM 项目的**最外层使用示例文件**。它不实现调度器、KV Cache、Attention 或模型结构，而是演示用户如何调用 nano-vLLM 完成一次离线大模型推理。

它主要完成以下事情：

1. 指定本地模型路径；
2. 加载 HuggingFace tokenizer；
3. 初始化 nano-vLLM 的 `LLM` 推理引擎；
4. 设置采样参数 `SamplingParams`；
5. 构造两个输入 prompt；
6. 使用 tokenizer 的 chat template 将普通文本包装成聊天模型格式；
7. 调用 `llm.generate()` 生成文本；
8. 打印 prompt 和 completion。

### 1.2 它属于哪一层

`example.py` 属于：

```text
入口层 / Demo 层 / 用户调用层
```

它不是 nano-vLLM 的核心引擎代码，但它是理解整个项目调用链的第一站。

### 1.3 它和项目中哪些文件有关

从 `example.py` 出发，后续会关联到以下文件：

```text
example.py
  ↓
nanovllm/__init__.py
  ↓
nanovllm/llm.py
  ↓
nanovllm/engine/llm_engine.py
  ↓
nanovllm/engine/sequence.py
nanovllm/engine/scheduler.py
nanovllm/engine/block_manager.py
nanovllm/engine/model_runner.py
  ↓
nanovllm/models/qwen3.py
nanovllm/layers/attention.py
nanovllm/layers/sampler.py
```

这些文件大致分工如下：

| 文件 | 作用 |
|---|---|
| `llm.py` | 暴露用户侧 `LLM` 类 |
| `llm_engine.py` | 推理引擎主循环，负责 `generate()`、`step()` |
| `sequence.py` | 表示一个请求的生命周期和 token 状态 |
| `scheduler.py` | 调度 waiting/running 请求，区分 prefill/decode |
| `block_manager.py` | 管理 KV Cache block 和 block table |
| `model_runner.py` | 准备模型输入，执行模型前向，管理 GPU 推理 |
| `qwen3.py` | 实现 Qwen3 模型结构 |
| `attention.py` | 执行 Attention，并读写 KV Cache |
| `sampler.py` | 根据 logits 和 temperature 采样下一个 token |

### 1.4 它在完整推理流程中的位置

完整流程可以理解为：

```text
用户输入 prompt
   ↓
example.py 构造 prompts 和 sampling_params
   ↓
LLM.generate(prompts, sampling_params)
   ↓
LLMEngine.add_request()
   ↓
Sequence 封装请求
   ↓
Scheduler 调度 prefill / decode
   ↓
BlockManager 分配 KV Cache block
   ↓
ModelRunner 准备输入并执行模型
   ↓
Qwen3ForCausalLM.forward()
   ↓
Attention 写入/读取 KV Cache
   ↓
Sampler 从 logits 中采样 token
   ↓
返回输出文本
   ↓
example.py 打印 completion
```

因此，`example.py` 的核心学习价值是：**从最外层 API 看清楚推理引擎的入口形态**。

---

## 2. 代码结构总览

### 2.1 源码全文

```python
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
```

### 2.2 导入了哪些模块

```python
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
```

导入模块说明：

| 模块 / 类 | 来源 | 作用 |
|---|---|---|
| `os` | Python 标准库 | 处理本地路径，例如展开 `~` |
| `LLM` | nano-vLLM | 用户侧推理引擎入口 |
| `SamplingParams` | nano-vLLM | 控制生成采样参数 |
| `AutoTokenizer` | HuggingFace Transformers | 加载模型对应 tokenizer |

### 2.3 定义了哪些函数

本文件只定义了一个函数：

```python
def main():
```

`main()` 封装了完整的推理调用流程。

### 2.4 定义了哪些类

本文件没有定义新类。它只是使用了外部类：

```text
LLM
SamplingParams
AutoTokenizer
```

### 2.5 重要变量或数据结构

| 变量 | 类型 | 作用 |
|---|---|---|
| `path` | `str` | 本地模型目录 |
| `tokenizer` | HuggingFace tokenizer 对象 | 将文本包装成 chat prompt，后续可编码成 token |
| `llm` | `LLM` 对象 | nano-vLLM 推理引擎实例 |
| `sampling_params` | `SamplingParams` 对象 | 控制 temperature、max_tokens 等生成参数 |
| `prompts` | `list[str]` | 输入请求列表 |
| `outputs` | `list[dict]` | 生成结果列表 |

### 2.6 主流程和辅助逻辑

主流程：

```text
设置模型路径
  ↓
加载 tokenizer
  ↓
初始化 LLM
  ↓
设置采样参数
  ↓
构造 prompts
  ↓
应用 chat template
  ↓
调用 generate
  ↓
打印输出
```

辅助逻辑：

```text
os.path.expanduser()
if __name__ == "__main__":
zip(prompts, outputs)
f-string + !r
```

### 2.7 文件组织结构图

```text
example.py
├── import 区域
│   ├── os
│   ├── LLM / SamplingParams
│   └── AutoTokenizer
│
├── main()
│   ├── 指定模型路径
│   ├── 加载 tokenizer
│   ├── 创建 LLM 引擎
│   ├── 创建 SamplingParams
│   ├── 构造 prompts
│   ├── 应用 chat template
│   ├── 调用 llm.generate()
│   └── 打印生成结果
│
└── 脚本入口
    └── if __name__ == "__main__": main()
```

---

## 3. 逐行代码解释

### 3.1 导入 `os`

```python
import os
```

#### 语法作用

导入 Python 标准库 `os`。`os` 提供操作系统相关接口，例如路径处理、环境变量读取、进程信息等。

#### 工程作用

本文件中只使用了：

```python
os.path.expanduser(...)
```

用来把 `~` 转换为用户 home 目录。

#### 在 nano-vLLM 推理流程中的意义

它不参与推理计算，只是为了让模型路径写法更方便。

#### 下一步关联文件

无。它是 Python 标准库功能。

---

### 3.2 导入 nano-vLLM 的用户侧 API

```python
from nanovllm import LLM, SamplingParams
```

#### 语法作用

从 `nanovllm` 包中导入两个对象：

```text
LLM
SamplingParams
```

#### 工程作用

`LLM` 是 nano-vLLM 对用户暴露的主入口，负责创建推理引擎并执行生成。

`SamplingParams` 是生成参数类，用于控制生成过程，例如：

```text
temperature
max_tokens
ignore_eos
```

#### 在 nano-vLLM 推理流程中的意义

这行代码把用户脚本和 nano-vLLM 引擎连接起来。

表面上用户调用：

```python
llm.generate(...)
```

底层实际会进入：

```text
LLMEngine.generate()
  → add_request()
  → step()
  → scheduler.schedule()
  → model_runner.run()
  → scheduler.postprocess()
```

#### 下一步关联文件

建议后续阅读：

```text
nanovllm/__init__.py
nanovllm/llm.py
nanovllm/sampling_params.py
nanovllm/engine/llm_engine.py
```

---

### 3.3 导入 HuggingFace tokenizer

```python
from transformers import AutoTokenizer
```

#### 语法作用

从 HuggingFace Transformers 库中导入 `AutoTokenizer`。

`AutoTokenizer` 会根据模型目录中的 tokenizer 配置自动加载对应分词器。

#### 工程作用

模型不能直接处理字符串，只能处理 token id。tokenizer 负责完成：

```text
文本字符串 → token ids
token ids → 文本字符串
```

在本文件中，它主要用于：

```python
tokenizer.apply_chat_template(...)
```

也就是把普通 prompt 转换成聊天模型输入格式。

#### 在 nano-vLLM 推理流程中的意义

对于 Qwen3 这类 Chat Model，直接输入裸文本通常不是最佳形式。模型训练时看到的是带角色标记的对话格式，因此需要 tokenizer 的 chat template。

#### 下一步关联文件

后续可结合：

```text
LLMEngine.generate()
```

看 nano-vLLM 内部如何继续把 formatted prompt 编码成 token ids。

---

### 3.4 定义主函数

```python
def main():
```

#### 语法作用

定义一个名为 `main` 的函数。冒号后面缩进的代码都属于这个函数。

#### 工程作用

将脚本主逻辑封装起来，避免文件被其他模块导入时自动执行推理。

#### 在 nano-vLLM 推理流程中的意义

`main()` 是用户侧推理流程的外部封装，不属于引擎内部。

---

### 3.5 设置模型路径

```python
path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
```

#### 语法作用

`os.path.expanduser()` 会把路径中的 `~` 展开成当前用户 home 目录。

例如：

```text
~/huggingface/Qwen3-0.6B/
```

可能被展开成：

```text
/home/username/huggingface/Qwen3-0.6B/
```

#### 工程作用

指定本地模型目录。该目录通常需要包含：

```text
config.json
tokenizer.json / tokenizer.model
model.safetensors
generation_config.json
```

#### 在 nano-vLLM 推理流程中的意义

这个路径会被传给两处：

```python
AutoTokenizer.from_pretrained(path)
LLM(path, ...)
```

也就是说：

```text
tokenizer 从该目录加载分词器
LLM 从该目录加载模型配置和权重
```

#### 下一步关联文件

后续阅读：

```text
nanovllm/config.py
nanovllm/utils/loader.py
nanovllm/models/qwen3.py
```

因为这些文件涉及模型配置读取和权重加载。

---

### 3.6 加载 tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(path)
```

#### 语法作用

调用类方法 `from_pretrained()`，从本地模型目录加载 tokenizer。

#### 工程作用

得到一个 tokenizer 对象，用来处理文本和 token 之间的转换。

例如：

```text
"introduce yourself"
  ↓ tokenizer
[若干整数 token id]
```

#### 在 nano-vLLM 推理流程中的意义

虽然本文件中没有直接调用 `tokenizer.encode()`，但后面的 chat template 会用到 tokenizer 配置。

对于 chat 模型来说，prompt 需要带上角色结构：

```text
user: ...
assistant:
```

否则模型可能无法按照对话方式回答。

#### 下一步关联文件

后续阅读：

```text
LLMEngine.generate()
```

看内部如何将 prompt 进一步编码成 token ids。

---

### 3.7 初始化 LLM 推理引擎

```python
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
```

#### 语法作用

调用 `LLM` 类的构造函数，创建一个 `llm` 对象。

参数含义：

| 参数 | 含义 |
|---|---|
| `path` | 模型路径 |
| `enforce_eager=True` | 强制使用 PyTorch eager mode |
| `tensor_parallel_size=1` | 张量并行大小为 1，即单 GPU 推理 |

#### 工程作用

这是 nano-vLLM 引擎真正初始化的地方。

内部大致会做：

```text
读取模型配置
初始化 tokenizer
初始化 scheduler
初始化 model runner
加载模型权重
分配 KV Cache
准备 CUDA / PyTorch 执行环境
```

#### 为什么设置 `enforce_eager=True`

`eager mode` 是 PyTorch 默认执行模式，操作会立即执行。

优点：

```text
方便调试
报错位置直观
适合学习源码
```

缺点：

```text
性能一般低于 CUDA Graph / torch.compile 优化路径
```

因此学习源码时建议先使用 `enforce_eager=True`。

#### 为什么设置 `tensor_parallel_size=1`

这表示不启用张量并行，模型完整放在一张 GPU 上。

如果设置为大于 1，就会涉及：

```text
多进程
NCCL 通信
权重切分
ColumnParallelLinear
RowParallelLinear
```

这会增加学习难度。

#### 在 nano-vLLM 推理流程中的意义

这一行是引擎启动点，后续所有生成请求都会交给这个 `llm` 对象处理。

#### 下一步关联文件

重点阅读：

```text
nanovllm/llm.py
nanovllm/engine/llm_engine.py
nanovllm/engine/model_runner.py
```

---

### 3.8 创建采样参数

```python
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
```

#### 语法作用

创建一个 `SamplingParams` 对象，并设置两个字段：

```text
temperature = 0.6
max_tokens = 256
```

#### 工程作用

控制模型生成时的采样行为。

| 参数 | 作用 |
|---|---|
| `temperature` | 控制生成随机性 |
| `max_tokens` | 限制最多生成多少个新 token |

#### 在 nano-vLLM 推理流程中的意义

推理引擎内部会把该参数绑定到每个 `Sequence` 上。

每条请求在 decode 过程中都会根据这些参数决定：

```text
每一步如何从 logits 中选 token
最多生成多少个 token
是否提前结束
```

#### temperature 的直观理解

模型每一步输出的是 logits，即词表中每个 token 的原始分数。

采样过程大致为：

```text
logits
  ↓ 除以 temperature
softmax
  ↓
概率分布
  ↓
采样一个 token
```

`temperature=0.6` 会让分布更尖锐，输出更稳定、更保守。

#### 下一步关联文件

建议阅读：

```text
nanovllm/sampling_params.py
nanovllm/layers/sampler.py
```

---

### 3.9 构造原始 prompts

```python
prompts = [
    "introduce yourself",
    "list all prime numbers within 100",
]
```

#### 语法作用

定义一个列表，里面包含两个字符串。

#### 工程作用

这表示一次向推理引擎提交两个请求。

| 请求编号 | 原始 prompt |
|---|---|
| 1 | `introduce yourself` |
| 2 | `list all prime numbers within 100` |

#### 在 nano-vLLM 推理流程中的意义

每个 prompt 后续会对应一个内部请求对象，也就是一个 `Sequence`。

可以理解为：

```text
prompt 1 → Sequence 1
prompt 2 → Sequence 2
```

调度器会将这些请求放入 waiting 队列，然后安排 prefill 和 decode。

#### 下一步关联文件

建议阅读：

```text
nanovllm/engine/sequence.py
nanovllm/engine/scheduler.py
```

---

### 3.10 使用 chat template 包装 prompt

```python
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for prompt in prompts
]
```

#### 语法作用

这是 Python 列表推导式。

它等价于：

```python
new_prompts = []
for prompt in prompts:
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    new_prompts.append(formatted_prompt)

prompts = new_prompts
```

#### 工程作用

把普通字符串 prompt 转换成 chat model 需要的输入格式。

普通输入：

```text
introduce yourself
```

可能会被转换成类似：

```text
<|im_start|>user
introduce yourself
<|im_end|>
<|im_start|>assistant
```

具体格式由 Qwen3 tokenizer 的 chat template 决定。

#### 参数解释

```python
[{"role": "user", "content": prompt}]
```

表示构造一个单轮对话，其中角色是 `user`。

```python
tokenize=False
```

表示返回字符串，而不是 token ids。

```python
add_generation_prompt=True
```

表示在末尾添加 assistant 开始回答的提示标记，让模型知道接下来应该生成助手回复。

#### 在 nano-vLLM 推理流程中的意义

这一步决定模型看到的 prompt 格式。

对于 Chat Model，如果不应用 chat template，模型可能出现：

```text
回答风格异常
不会正确进入 assistant 角色
结束符处理异常
输出格式不稳定
```

#### 下一步关联文件

建议阅读：

```text
LLMEngine.generate()
```

因为 formatted prompt 会在那里变成 token ids 并进入推理流程。

---

### 3.11 调用生成接口

```python
outputs = llm.generate(prompts, sampling_params)
```

#### 语法作用

调用 `llm` 对象的 `generate()` 方法，输入 prompts 和采样参数，返回 outputs。

#### 工程作用

这是本文件最核心的一行。

从用户角度看：

```text
输入 prompts
输出 completions
```

从推理引擎角度看，内部会触发完整流程：

```text
1. 将 prompt 编码成 token ids
2. 为每条请求创建 Sequence
3. Scheduler 接收请求
4. Prefill 阶段处理 prompt
5. 写入 KV Cache
6. Decode 阶段逐 token 生成
7. 每一步通过 Sampler 采样 token
8. 判断 EOS 或 max_tokens
9. 返回最终文本
```

#### 在 nano-vLLM 推理流程中的意义

这是用户侧 API 和内部推理引擎的分界线。

它背后的核心链路是：

```text
LLM.generate()
  ↓
LLMEngine.generate()
  ↓
LLMEngine.add_request()
  ↓
Scheduler.add()
  ↓
while not finished:
      LLMEngine.step()
        ↓
      Scheduler.schedule()
        ↓
      ModelRunner.run()
        ↓
      Scheduler.postprocess()
```

#### 下一步关联文件

最应该阅读：

```text
nanovllm/engine/llm_engine.py
```

---

### 3.12 遍历 prompt 和 output

```python
for prompt, output in zip(prompts, outputs):
```

#### 语法作用

`zip(prompts, outputs)` 会把两个列表按位置配对。

例如：

```text
prompts[0] ↔ outputs[0]
prompts[1] ↔ outputs[1]
```

#### 工程作用

保证每个输入 prompt 和它对应的生成结果一起打印。

#### 在 nano-vLLM 推理流程中的意义

说明 `llm.generate()` 返回结果的顺序与输入 prompt 顺序对应。

---

### 3.13 打印空行

```python
print("\n")
```

#### 语法作用

打印一个换行字符串。

#### 工程作用

让多个输出之间更容易区分。

#### 在 nano-vLLM 推理流程中的意义

无。它只是输出格式控制。

---

### 3.14 打印 prompt

```python
print(f"Prompt: {prompt!r}")
```

#### 语法作用

这是 f-string 格式化字符串。

`!r` 表示使用 `repr()` 形式打印变量。

例如普通打印可能显示：

```text
hello
```

而 `repr()` 会显示：

```text
'hello'
```

如果字符串里有换行符或特殊 token，也会更容易看清楚。

#### 工程作用

打印经过 chat template 处理后的 prompt。

#### 在 nano-vLLM 推理流程中的意义

这可以帮助你观察真正送给模型的 prompt 长什么样，而不是只看原始用户输入。

---

### 3.15 打印 completion

```python
print(f"Completion: {output['text']!r}")
```

#### 语法作用

从 `output` 字典中取出键为 `'text'` 的值，并用 `repr()` 格式打印。

#### 工程作用

打印模型生成的文本结果。

这也暗示 `llm.generate()` 返回的数据结构中，每个输出至少包含：

```python
{
    "text": "生成文本"
}
```

可能还会包含 token ids 等其他信息，具体要看 `LLMEngine.generate()` 的实现。

#### 在 nano-vLLM 推理流程中的意义

这是最终 completion 从引擎返回到用户侧脚本的位置。

---

### 3.16 脚本入口判断

```python
if __name__ == "__main__":
    main()
```

#### 语法作用

当文件被直接运行时，`__name__` 的值是 `"__main__"`，于是执行 `main()`。

如果该文件被其他文件导入，则不会自动执行。

#### 工程作用

这是 Python 脚本标准入口写法。

可以支持：

```bash
python example.py
```

直接运行，同时也允许其他文件安全导入 `example.py`。

#### 在 nano-vLLM 推理流程中的意义

它是用户启动推理示例的入口。

---

## 4. 背后的框架性原理

### 4.1 prompt / token / tokenizer

#### 是什么

`prompt` 是用户输入的自然语言文本。

`token` 是模型真正处理的基本单位，通常是整数 id。

`tokenizer` 负责：

```text
文本 → token ids
token ids → 文本
```

#### 为什么重要

大语言模型不是直接处理字符串，而是处理 token id。tokenizer 决定文本如何被切分，也决定 chat template 如何包装对话格式。

#### 在本文件中如何体现

```python
tokenizer = AutoTokenizer.from_pretrained(path)
```

加载 tokenizer。

```python
tokenizer.apply_chat_template(...)
```

将普通 prompt 包装成聊天模型格式。

#### 和 nano-vLLM / vLLM 的关系

nano-vLLM 内部推理流程最终处理的是 token ids。外部 prompt 进入 `generate()` 后，会被编码成 token ids，再交给 Sequence、Scheduler 和 ModelRunner。

---

### 4.2 request / sequence

#### 是什么

一个 request 是用户提交的一条生成请求。

一个 sequence 是推理引擎内部管理请求的对象，通常包含：

```text
token_ids
status
block_table
sampling_params
已生成 token 数
是否结束
```

#### 为什么重要

推理引擎需要同时处理多个请求，必须用统一的数据结构记录每个请求的状态。

#### 在本文件中如何体现

```python
prompts = [
    "introduce yourself",
    "list all prime numbers within 100",
]
```

这两个 prompt 会在引擎内部变成两个 sequence。

#### 和 nano-vLLM / vLLM 的关系

vLLM 的调度器管理的不是裸字符串，而是 request / sequence 级别的对象。nano-vLLM 对这一思想做了简化实现。

---

### 4.3 scheduler

#### 是什么

Scheduler 是调度器，负责决定每一轮处理哪些请求，是执行 prefill 还是 decode。

#### 为什么重要

多请求推理时，每个请求长度不同、到达时间不同、结束时间不同。没有 scheduler，就无法实现高吞吐 continuous batching。

#### 在本文件中如何体现

本文件没有直接调用 scheduler，但：

```python
outputs = llm.generate(prompts, sampling_params)
```

内部会调用 scheduler。

#### 和 nano-vLLM / vLLM 的关系

Scheduler 是 vLLM 推理引擎的核心模块之一，负责把多个 sequence 动态组织成 batch。

---

### 4.4 prefill / decode

#### 是什么

Prefill：处理 prompt 的阶段，一次性计算 prompt 的 hidden states，并写入 KV Cache。

Decode：逐 token 生成阶段，每次只输入上一个 token，并复用历史 KV Cache。

#### 为什么重要

大模型推理的两个阶段性能特征完全不同：

| 阶段 | 特点 |
|---|---|
| Prefill | 计算密集，处理整段 prompt |
| Decode | 显存访问密集，每次生成一个 token |

#### 在本文件中如何体现

本文件没有显式区分 prefill/decode，但 `llm.generate()` 内部一定会分成这两个阶段。

#### 和 nano-vLLM / vLLM 的关系

prefill/decode 分离是推理引擎设计的基础。vLLM 通过调度策略和 KV Cache 管理提高这两个阶段的整体吞吐。

---

### 4.5 KV Cache

#### 是什么

KV Cache 保存每一层 attention 中历史 token 的 Key 和 Value。

#### 为什么重要

没有 KV Cache，decode 每生成一个 token 都要重新计算所有历史 token，代价极高。

有 KV Cache 后：

```text
历史 token 的 K/V 直接复用
当前步只计算新 token 的 Q/K/V
```

#### 在本文件中如何体现

本文件没有直接出现 KV Cache，但初始化：

```python
llm = LLM(path, ...)
```

和生成：

```python
llm.generate(...)
```

都会触发底层 KV Cache 分配和使用。

#### 和 nano-vLLM / vLLM 的关系

vLLM 的核心创新之一就是高效管理 KV Cache，尤其是 PagedAttention。nano-vLLM 用更少的代码复现了 block 级 KV Cache 管理思想。

---

### 4.6 block / block table / block manager

#### 是什么

block 是 KV Cache 的分页单位。

block table 记录某条 sequence 的逻辑 token block 对应哪些物理 KV Cache block。

block manager 负责分配、释放和复用 block。

#### 为什么重要

如果 KV Cache 必须连续分配，显存碎片和浪费会很严重。分页管理可以显著提高显存利用率。

#### 在本文件中如何体现

本文件没有显式出现 block，但每个 prompt 进入 `llm.generate()` 后，底层都会为它分配 KV Cache block。

#### 和 nano-vLLM / vLLM 的关系

PagedAttention 的核心思想就是 block 化管理 KV Cache。nano-vLLM 的 `block_manager.py` 是学习这个思想的重点文件。

---

### 4.7 attention

#### 是什么

Attention 是 Transformer 的核心模块。每个 token 会根据 query、key、value 计算对历史上下文的关注权重。

#### 为什么重要

大模型生成下一个 token 时，需要通过 attention 读取前文信息。

#### 在本文件中如何体现

本文件没有直接调用 attention，但 `llm.generate()` 内部模型前向会进入 Qwen3 的 attention 层。

#### 和 nano-vLLM / vLLM 的关系

nano-vLLM 的 attention 层不仅计算注意力，还负责使用 KV Cache，并根据 prefill/decode 选择不同的 FlashAttention 路径。

---

### 4.8 model runner / GPU 执行

#### 是什么

ModelRunner 是负责真正执行模型前向计算的模块。

它通常负责：

```text
准备 input_ids
准备 positions
准备 slot_mapping
设置上下文
调用模型 forward
调用 sampler
```

#### 为什么重要

Scheduler 只决定“跑哪些请求”，ModelRunner 负责“怎么把这些请求变成 GPU 可执行的张量”。

#### 在本文件中如何体现

```python
llm = LLM(path, ...)
```

初始化时会创建 ModelRunner。

```python
llm.generate(...)
```

生成时会调用 ModelRunner 执行模型。

#### 和 nano-vLLM / vLLM 的关系

ModelRunner 是连接调度系统和模型计算图的桥梁。

---

### 4.9 tensor parallel

#### 是什么

Tensor Parallel 是把模型中的大矩阵按维度切到多张 GPU 上并行计算。

#### 为什么重要

当模型太大，一张 GPU 放不下，或者需要更高吞吐时，就需要张量并行。

#### 在本文件中如何体现

```python
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
```

这里设置 `tensor_parallel_size=1`，表示不启用多卡张量并行。

#### 和 nano-vLLM / vLLM 的关系

nano-vLLM 支持简化版张量并行，相关代码主要在：

```text
layers/linear.py
layers/embed_head.py
engine/model_runner.py
```

---

### 4.10 logits / sampling / temperature

#### 是什么

logits 是模型对词表中每个 token 的原始打分。

sampling 是从 logits 转成的概率分布中选择下一个 token。

temperature 控制分布的平滑程度。

#### 为什么重要

生成质量和随机性主要由采样策略决定。

#### 在本文件中如何体现

```python
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
```

设置了 temperature 和最大生成长度。

#### 和 nano-vLLM / vLLM 的关系

nano-vLLM 的 sampler 根据 logits 和 temperature 采样 token。生产级 vLLM 还会支持更复杂的 top_p、top_k、repetition penalty 等参数。

---

### 4.11 throughput / latency / TTFT / TPOT

#### 是什么

| 指标 | 含义 |
|---|---|
| throughput | 单位时间生成 token 数 |
| latency | 单个请求总延迟 |
| TTFT | Time To First Token，首 token 延迟 |
| TPOT | Time Per Output Token，平均每个输出 token 耗时 |

#### 为什么重要

推理引擎的目标不是“能生成”就行，而是要高吞吐、低延迟、稳定并发。

#### 在本文件中如何体现

本文件不是 benchmark 文件，但它一次传入多个 prompts：

```python
outputs = llm.generate(prompts, sampling_params)
```

这已经体现了批量请求入口。

#### 和 nano-vLLM / vLLM 的关系

vLLM 通过 continuous batching、PagedAttention、KV Cache 管理等技术提高 throughput，同时降低高并发下的平均延迟。

---

## 5. 和 vLLM 原版设计的关系

### 5.1 体现的 vLLM 核心思想

虽然 `example.py` 是入口脚本，但它间接体现了以下 vLLM 思想。

#### 统一 LLM API

用户只需要写：

```python
llm = LLM(...)
outputs = llm.generate(...)
```

不需要直接操作模型 forward、KV Cache 或调度器。

#### 多请求批处理

```python
prompts = [
    "introduce yourself",
    "list all prime numbers within 100",
]
```

这说明 API 层天然支持多个请求一起提交。

#### prefill / decode 分离

用户看不到该过程，但 `generate()` 内部一定需要将 prompt 处理和逐 token 生成拆开。

#### KV Cache block 管理

用户不需要手动管理 KV Cache。底层会由 BlockManager 分配和释放。

#### 高吞吐推理服务思想

`example.py` 虽然是离线 demo，但背后的结构服务于高吞吐推理：多个请求进入引擎，由调度器统一处理。

---

### 5.2 nano-vLLM 相比原版 vLLM 的简化

从该示例可以推测 nano-vLLM 做了很多教学化简化：

| 方向 | nano-vLLM 可能的简化 |
|---|---|
| 服务接口 | 没有复杂 OpenAI-compatible Server |
| 采样参数 | 主要演示 temperature、max_tokens，未完整覆盖 top_p/top_k 等 |
| 调度策略 | 简化 continuous batching 策略 |
| KV Cache | 保留 block 管理核心思想，但策略更简单 |
| 并行系统 | 支持 TP，但没有生产级复杂部署逻辑 |
| 监控指标 | 没有完整 metrics / tracing / observability |
| 容错机制 | 没有生产级请求取消、超时、优先级等机制 |

这也是它适合学习的原因：核心思想保留，但工程复杂度降低。

---

## 6. 学习建议与下一步阅读路线

`example.py` 是入口文件，不需要在这里停留太久。你真正要抓住的是这条调用链：

```text
example.py
  ↓
LLM(path, ...)
  ↓
LLMEngine.__init__()
  ↓
ModelRunner 初始化模型和 KV Cache
  ↓
Scheduler 初始化请求队列

llm.generate(prompts, sampling_params)
  ↓
LLMEngine.generate()
  ↓
add_request()
  ↓
Sequence(prompt, sampling_params)
  ↓
Scheduler.add(seq)
  ↓
while not finished:
      step()
        ↓
      scheduler.schedule()
        ↓
      model_runner.run()
        ↓
      scheduler.postprocess()
```

推荐下一步阅读顺序：

```text
1. nanovllm/llm.py
2. nanovllm/sampling_params.py
3. nanovllm/engine/llm_engine.py
4. nanovllm/engine/sequence.py
5. nanovllm/engine/scheduler.py
```

尤其是 `llm_engine.py`，因为 `example.py` 中最关键的一行：

```python
outputs = llm.generate(prompts, sampling_params)
```

真正的实现逻辑就在 `LLMEngine.generate()` 里。

---

## 7. 一句话总结

`example.py` 是 nano-vLLM 的用户入口示例，它展示了如何加载 Qwen3 tokenizer、初始化 `LLM` 推理引擎、设置采样参数、构造 chat prompt，并调用 `llm.generate()` 完成生成。它本身不实现推理引擎核心逻辑，但它连接了用户输入和内部的 `LLMEngine → Scheduler → BlockManager → ModelRunner → Attention → Sampler` 主流程，是学习 nano-vLLM 源码的第一站。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]

%% 项目关联导航：结束 %%
