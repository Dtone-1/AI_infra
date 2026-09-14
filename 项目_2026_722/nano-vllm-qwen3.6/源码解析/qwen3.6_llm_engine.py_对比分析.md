# qwen3.6_llm_engine.py_对比分析

> 对比对象：  
> 1. 原版 `nano-vLLM` 的 `llm_engine.py`  
> 2. `nano-vllm-qwen3.6` 中对应的 `llm_engine.py`  
>
> 分析目标：从整体工程角度理解 qwen3.6 版本相对原版做了哪些改造、为什么要改、这些改动在推理入口层发挥什么作用。

---

## 1. 文件整体定位

`llm_engine.py` 是 nano-vLLM 推理系统中最外层的 Python Engine 入口之一。

它主要负责：

```text
用户输入 prompt
  ↓
tokenizer / 输入预处理
  ↓
构造 Sequence
  ↓
交给 Scheduler
  ↓
Scheduler 每轮选出要运行的 seqs
  ↓
ModelRunner 执行模型 forward
  ↓
Sampler 采样 token
  ↓
Scheduler 更新序列状态
  ↓
输出最终文本和 token_ids
```

所以这个文件不是模型结构本体，也不是 Attention / MLP / GatedDeltaNet 的实现位置。

它更像是：

```text
用户接口层 + 请求入口层 + 推理主循环封装层
```

在原版 nano-vLLM 中，它主要面向纯文本输入；而 qwen3.6 版本在这个文件中最明显的变化，是把 Engine 的输入能力从“文本 / token id”扩展到了“多模态 messages 格式”。

---

## 2. 整体变化概览

qwen3.6 版本的 `llm_engine.py` 相比原版主要有四类变化。

| 改动类别 | 原版 nano-vLLM | qwen3.6 版本 | 作用 |
|---|---|---|---|
| 多模态处理入口 | 只支持 `str` 或 `list[int]` | 支持 `str`、`list[int]`、`list[dict]` | 支持带图片的 messages 输入 |
| 图像预处理 | 无 | 引入 `process_messages` | 将多模态消息转成 token、图像张量、图像网格信息 |
| 配置字段 | 只使用 KV cache、tp、model 等基础配置 | 新增读取 `image_token_id`、`vision_start_token_id`、`vision_end_token_id` | 让 Engine 知道图像 token 的特殊标记 |
| 退出逻辑 | 直接调用 `model_runner.exit` 并 join 子进程 | 增加 `_exited` 防重复退出、`hasattr` 判断、清空进程列表 | 防止重复析构、异常退出、资源释放不干净 |

可以一句话总结：

**这个文件的改动重点不是直接实现 Qwen3.6 的 hybrid / GatedDeltaNet 计算，而是把推理入口从纯文本请求扩展为可以接收多模态消息，并增强 Engine 生命周期管理。**

---

## 3. 改动一：新增 `process_messages` 导入

### 3.1 原版代码

原版导入部分主要是：

```python
from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
```

它只关心：

```text
配置
采样参数
请求序列
调度器
模型执行器
```

### 3.2 qwen3.6 版本新增

qwen3.6 版本多了一行：

```python
from nanovllm.utils.image_processing import process_messages
```

### 3.3 这说明什么

这说明 qwen3.6 版本的 Engine 入口开始承担一种新的输入预处理任务：

```text
多模态 messages
  ↓
process_messages
  ↓
token_ids + pixel_values + image_grid_thw
```

也就是说，用户输入不再只能是：

```python
"你好，请介绍一下你自己"
```

或者：

```python
[151644, 872, 198, ...]
```

还可以是类似：

```python
[
    {"role": "user", "content": [
        {"type": "image", "image": "..."},
        {"type": "text", "text": "请描述这张图片"}
    ]}
]
```

当然，具体 messages 格式取决于 `process_messages` 的实现。

### 3.4 工程意义

这个导入虽然只有一行，但意义很大：

1. Engine 层开始支持多模态输入。
2. 文本 token 和图像 tensor 的构造被统一放在请求入口。
3. 下游的 Scheduler 和 ModelRunner 可以继续围绕 `Sequence` 工作，而不是直接处理原始图片。
4. 模型 forward 前就能准备好视觉相关张量。

这属于典型的推理框架设计：

```text
入口层处理用户格式
执行层处理张量格式
模型层处理神经网络计算
```

---

## 4. 改动二：初始化时新增视觉特殊 token id

### 4.1 原版初始化逻辑

原版 `__init__` 中主要做：

```python
config = Config(model, **config_kwargs)
Sequence.block_size = config.kvcache_block_size
self.ps = []
self.events = []
...
self.model_runner = ModelRunner(config, 0, self.events)
self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
config.eos = self.tokenizer.eos_token_id
self.scheduler = Scheduler(config)
```

这个流程说明原版 Engine 只需要知道：

```text
模型路径
KV Cache block size
tensor parallel size
tokenizer
EOS token
scheduler
model_runner
```

### 4.2 qwen3.6 版本新增字段

qwen3.6 版本在初始化时新增：

```python
self.image_token_id = config.image_token_id
self.vision_start_token_id = config.vision_start_token_id
self.vision_end_token_id = config.vision_end_token_id
self._exited = False
```

前三个字段和多模态有关：

| 字段 | 作用 |
|---|---|
| `image_token_id` | 表示图像占位 token 的编号 |
| `vision_start_token_id` | 表示视觉内容开始的特殊 token |
| `vision_end_token_id` | 表示视觉内容结束的特殊 token |

### 4.3 为什么需要这些 token id

多模态大模型通常不会把图片直接塞进文本序列。

它会先把图片变成视觉特征，然后在文本 token 序列中插入一些特殊标记，例如：

```text
<vision_start> <image_pad/image_token> ... <vision_end>
```

这些特殊 token 的作用是告诉模型：

```text
这里开始是视觉区域
这里有图像特征占位
这里视觉区域结束
```

所以 Engine 在处理 messages 时必须知道这些特殊 token 的编号，否则它无法正确构造多模态输入序列。

### 4.4 在推理系统中的意义

这一步让 Engine 具备了从“自然用户输入”到“模型可执行输入”的转换能力。

对于纯文本模型：

```text
prompt -> tokenizer.encode -> token_ids
```

对于多模态模型：

```text
messages
  ↓
解析文本和图片
  ↓
插入 vision_start / image_token / vision_end
  ↓
生成 token_ids
  ↓
生成 pixel_values
  ↓
生成 image_grid_thw
```

这就是 qwen3.6 版本在入口层新增视觉 token 配置的原因。

---

## 5. 改动三：新增 `_exited`，让 `exit()` 具备幂等性

### 5.1 原版退出逻辑

原版：

```python
def exit(self):
    self.model_runner.call("exit")
    del self.model_runner
    for p in self.ps:
        p.join()
```

这个逻辑很直接：

1. 通知 `model_runner` 退出；
2. 删除主进程中的 `model_runner`；
3. 等待 tensor parallel 子进程结束。

### 5.2 原版潜在问题

这个写法有一个隐患：

```text
如果 exit() 被调用多次，可能重复访问已经删除的 model_runner。
```

因为 `exit()` 会被：

```python
atexit.register(self.exit)
```

注册到 Python 进程退出阶段。

同时用户也可能主动调用：

```python
engine.exit()
```

如果主动调用一次，进程结束时 `atexit` 又调用一次，就可能出现：

```text
AttributeError
重复 join
重复释放资源
```

### 5.3 qwen3.6 版本修改

qwen3.6 版本改成：

```python
def exit(self):
    if self._exited:
        return
    self._exited = True
    if hasattr(self, "model_runner"):
        self.model_runner.call("exit")
        del self.model_runner
    for p in self.ps:
        p.join()
    self.ps.clear()
```

### 5.4 这个改动解决什么问题

这个改动让 `exit()` 变成幂等函数。

所谓幂等，就是：

```text
调用一次和调用多次，最终效果一致。
```

具体体现在：

| 设计 | 作用 |
|---|---|
| `self._exited` | 防止重复退出 |
| `hasattr(self, "model_runner")` | 防止访问已删除属性 |
| `self.ps.clear()` | 子进程 join 后清空列表，避免后续重复处理 |

### 5.5 工程意义

这个改动和模型结构无关，但对推理服务很重要。

因为实际推理服务可能发生：

1. 用户主动停止服务；
2. Python 解释器退出；
3. 异常触发析构；
4. 多进程启动失败后进入清理；
5. notebook / 脚本重复创建和销毁 engine。

如果退出逻辑不稳，容易出现：

```text
僵尸进程
重复释放
进程卡住
异常日志污染
GPU 资源未释放
```

所以 `_exited` 是一个小改动，但属于工程稳定性增强。

---

## 6. 改动四：`add_request()` 支持多模态 messages

这是整个文件最重要的改动。

### 6.1 原版 `add_request`

原版：

```python
def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
    if isinstance(prompt, str):
        prompt = self.tokenizer.encode(prompt)
    seq = Sequence(prompt, sampling_params)
    self.scheduler.add(seq)
```

原版只支持两种输入：

| 输入类型 | 含义 |
|---|---|
| `str` | 原始文本 prompt，需要 tokenizer 编码 |
| `list[int]` | 已经编码好的 token ids |

也就是说，原版输入路径是：

```text
文本 prompt
  ↓ tokenizer.encode
token ids
  ↓ Sequence
Scheduler
```

或者：

```text
token ids
  ↓ Sequence
Scheduler
```

### 6.2 qwen3.6 版本 `add_request`

qwen3.6 版本改成：

```python
def add_request(self, prompt: str | list[int] | list[dict], sampling_params: SamplingParams):
    if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
        # Multimodal messages format
        token_ids, pixel_values, image_grid_thw = process_messages(
            prompt, self.tokenizer,
            image_token_id=self.image_token_id,
            vision_start_id=self.vision_start_token_id,
            vision_end_id=self.vision_end_token_id,
        )
        seq = Sequence(token_ids, sampling_params)
        seq.pixel_values = pixel_values
        seq.image_grid_thw = image_grid_thw
    else:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
    self.scheduler.add(seq)
```

### 6.3 新增输入类型：`list[dict]`

新增类型：

```python
list[dict]
```

表示一种结构化 messages 输入。

为什么用 `list[dict]` 判断？

因为多模态聊天输入通常不是单纯字符串，而是一组 message：

```python
[
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
]
```

多模态场景中 content 里还可能包含图片。

所以判断条件是：

```python
isinstance(prompt, list)
and len(prompt) > 0
and isinstance(prompt[0], dict)
```

含义：

```text
只要 prompt 是非空 list，并且第一个元素是 dict，就认为它是 messages 格式。
```

### 6.4 `process_messages()` 输出了什么

qwen3.6 版本调用：

```python
token_ids, pixel_values, image_grid_thw = process_messages(...)
```

这三个输出分别对应：

| 输出 | 含义 | 下游用途 |
|---|---|---|
| `token_ids` | 文本和视觉占位符组成的 token 序列 | 构造 `Sequence`，进入调度器 |
| `pixel_values` | 图片预处理后的张量 | 送给视觉编码器或多模态模型部分 |
| `image_grid_thw` | 图像 patch 网格信息，通常表示 time/height/width | 帮助模型知道视觉 token 的空间结构 |

这说明 qwen3.6 版本的请求不再只有 token 序列，还带着图像张量和图像结构元信息。

### 6.5 为什么要挂到 `Sequence` 上

代码：

```python
seq = Sequence(token_ids, sampling_params)
seq.pixel_values = pixel_values
seq.image_grid_thw = image_grid_thw
```

这里没有新建一个 `MultimodalSequence`，而是直接给 `Sequence` 动态挂属性。

这样做的好处是：

```text
尽量复用原来的 Scheduler / Sequence / Engine 流程
```

原来的 Scheduler 仍然只需要围绕 token 数、KV Cache block、状态机进行调度。

多模态额外信息则随着 `Sequence` 一起传给后面的 `ModelRunner`。

可以理解为：

```text
Sequence 原来只背 token_ids
现在 Sequence 还可以背 pixel_values 和 image_grid_thw
```

### 6.6 工程意义

这一步是“多模态推理入口改造”的核心。

它将原来的纯文本请求：

```text
prompt -> token ids -> Sequence
```

扩展为：

```text
messages
  ↓
文本 tokenize + 图片 preprocess + 视觉特殊 token 插入
  ↓
token_ids + pixel_values + image_grid_thw
  ↓
Sequence
  ↓
Scheduler
```

这让下游可以继续沿用大部分原有推理框架，只在必要位置增加图像张量处理。

---

## 7. 改动五：`generate()` 的输入类型扩展

### 7.1 原版类型标注

原版：

```python
prompts: list[str] | list[list[int]]
```

含义：

```text
一批文本 prompt
或者
一批 token ids
```

### 7.2 qwen3.6 版本类型标注

qwen3.6 版本：

```python
prompts: list[str] | list[list[int]] | list[list[dict]]
```

新增：

```python
list[list[dict]]
```

也就是：

```text
一批 messages 格式的多模态请求
```

单个 prompt 是：

```python
list[dict]
```

一批 prompts 就是：

```python
list[list[dict]]
```

### 7.3 为什么只改类型标注，主循环不用改

`generate()` 的主循环仍然是：

```python
for prompt, sp in zip(prompts, sampling_params):
    self.add_request(prompt, sp)
```

因为真正区分输入类型的逻辑已经放到了 `add_request()` 里。

所以 `generate()` 不需要关心：

```text
这是文本？
这是 token ids？
这是多模态 messages？
```

它只负责：

```text
把每个 prompt 加入请求队列
```

这体现了一个比较好的分层：

```text
generate(): 管批处理
add_request(): 管单请求输入格式
process_messages(): 管多模态预处理
```

### 7.4 工程意义

这种改法侵入性比较低。

原版文本推理路径完全保留：

```text
str -> tokenizer.encode -> Sequence
list[int] -> Sequence
```

新增多模态路径只在必要时触发：

```text
list[dict] -> process_messages -> Sequence + 图像属性
```

所以它对已有文本推理功能影响较小。

---

## 8. 保持不变的部分

除了上面几处变化，`llm_engine.py` 的核心推理循环基本保持不变。

### 8.1 多进程 tensor parallel 启动逻辑基本不变

仍然是：

```python
ctx = mp.get_context("spawn")
for i in range(1, config.tensor_parallel_size):
    event = ctx.Event()
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()
```

说明 qwen3.6 版本没有在这个文件里改变 tensor parallel 的基本启动方式。

### 8.2 `step()` 基本不变

仍然是：

```text
Scheduler.schedule()
  ↓
ModelRunner.run()
  ↓
Scheduler.postprocess()
  ↓
返回已完成请求
```

这说明 prefill / decode 调度主流程没有在 `llm_engine.py` 中发生大改。

### 8.3 吞吐统计逻辑不变

仍然通过：

```python
num_tokens > 0
```

区分 prefill 吞吐，反之用负数表示 decode batch size：

```python
prefill_throughput = num_tokens / time
decode_throughput = -num_tokens / time
```

说明这个文件仍然沿用原版的性能统计方式。

### 8.4 输出格式不变

最终仍然返回：

```python
{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
```

这意味着无论输入是文本还是多模态 messages，输出仍然是语言模型生成的文本 token。

---

## 9. 从推理流程角度看 qwen3.6 版本的新路径

### 9.1 原版纯文本路径

```text
用户输入 str
  ↓
tokenizer.encode
  ↓
Sequence(token_ids, sampling_params)
  ↓
Scheduler.add
  ↓
Scheduler.schedule
  ↓
ModelRunner.run
  ↓
模型 forward
  ↓
Sampler
  ↓
输出 token_ids / text
```

### 9.2 qwen3.6 多模态路径

```text
用户输入 list[dict] messages
  ↓
process_messages
  ↓
得到 token_ids / pixel_values / image_grid_thw
  ↓
Sequence(token_ids, sampling_params)
  ↓
seq.pixel_values = pixel_values
seq.image_grid_thw = image_grid_thw
  ↓
Scheduler.add
  ↓
Scheduler.schedule
  ↓
ModelRunner.run
  ↓
模型 forward 时结合文本 token 和视觉特征
  ↓
Sampler
  ↓
输出 token_ids / text
```

### 9.3 最关键变化

原版 Sequence 中核心信息是：

```text
token_ids
sampling_params
KV Cache 状态
请求状态
```

qwen3.6 版本中，Sequence 可能额外携带：

```text
pixel_values
image_grid_thw
```

这意味着 Sequence 从“纯文本请求对象”扩展成了“可携带多模态附加张量的请求对象”。

---

## 10. 这个文件和 Qwen3.6 / hybrid 架构的关系

需要特别注意：

**仅从这个 `llm_engine.py` 文件本身，看不到 GatedDeltaNet、hybrid layer、recurrent state、conv state、MoE routing 等 Qwen3.6 模型结构层面的改造。**

原因是 `llm_engine.py` 位于推理入口层，职责是：

```text
接收请求
预处理输入
管理调度主循环
调用 ModelRunner
管理退出
```

而 Qwen3.6 如果要支持 hybrid 架构，真正可能需要改的文件通常是：

```text
config.py
engine/model_runner.py
models/qwen3.py 或 qwen3_6.py
layers/attention.py
layers/gated_deltanet.py
utils/context.py
KV cache / state manager 相关文件
```

也就是说：

```text
llm_engine.py 不是 hybrid 计算发生的位置。
```

在这个文件里看到的是“入口层为更复杂输入形态做准备”。

如果 qwen3.6 项目包含视觉或多模态能力，那么这里的改动是合理的；如果讨论的是 Qwen3.6 hybrid 文本模型，那么这个文件只能说明 Engine 接口被扩展，并不能证明 hybrid 架构本身已经实现。

---

## 11. 每个改动的作用总结

| 改动 | 作用 | 对推理系统的意义 |
|---|---|---|
| 引入 `process_messages` | 支持 messages 格式输入预处理 | 为多模态请求进入推理系统提供入口 |
| 保存 `image_token_id` 等配置 | 识别视觉特殊 token | 正确构造文本-图像混合 token 序列 |
| `add_request` 支持 `list[dict]` | 支持多模态 messages | 用户接口从纯文本扩展到结构化消息 |
| `seq.pixel_values` | 把图像张量挂到请求对象上 | 下游 ModelRunner 可以拿到视觉输入 |
| `seq.image_grid_thw` | 保存图像 patch 网格结构 | 视觉模型可恢复图像空间布局 |
| `generate` 类型扩展 | 批量多模态输入类型合法化 | 保持批处理接口统一 |
| `_exited` | 防止重复退出 | 提升服务稳定性 |
| `hasattr(model_runner)` | 防止重复删除对象 | 避免析构阶段异常 |
| `self.ps.clear()` | 清空已 join 子进程列表 | 避免重复 join 和状态污染 |

---

## 12. 对初学者最重要的理解

这个文件的改动可以用一句话理解：

```text
原版 llm_engine.py：文本 prompt 入口
qwen3.6 llm_engine.py：文本 prompt + 多模态 messages 入口
```

它没有改变核心推理循环：

```text
Scheduler -> ModelRunner -> Model forward -> Sampler
```

而是在请求进入 Scheduler 之前，增加了一个分支：

```text
如果输入是 list[dict] messages：
    走多模态预处理
否则：
    走原来的文本/token 路径
```

所以它属于：

```text
输入接口层改造
```

而不是：

```text
模型计算层改造
```

---

## 13. 面试角度应该怎么回答

如果面试官问：

> qwen3.6 版本的 `llm_engine.py` 相比原版改了什么？

可以这样回答：

`llm_engine.py` 是推理入口层，原版主要支持文本字符串或 token id 输入，然后构造 `Sequence` 加入 `Scheduler`。qwen3.6 版本在这个文件中主要扩展了多模态 messages 输入：新增 `process_messages`，从 config 中读取 `image_token_id`、`vision_start_token_id`、`vision_end_token_id`，当 prompt 是 `list[dict]` 时，会把 messages 预处理成 `token_ids`、`pixel_values` 和 `image_grid_thw`，再把图像张量和网格信息挂到 `Sequence` 上，供后续 `ModelRunner` 和模型 forward 使用。同时它还增强了 `exit()`，通过 `_exited` 和 `hasattr` 防止重复退出导致异常。整体来看，这个文件没有直接实现 Qwen3.6 的 hybrid / GatedDeltaNet 结构，而是把 Engine 输入接口从纯文本扩展到可支持多模态请求，并提升了进程资源释放的稳定性。

---

## 14. Mermaid 对比流程图

```mermaid
flowchart TD
    A["用户输入 prompt"] --> B{"输入类型?"}

    B -->|str| C["tokenizer.encode"]
    C --> D["Sequence(token_ids)"]

    B -->|list[int]| D

    B -->|list[dict] messages| E["process_messages"]
    E --> F["token_ids"]
    E --> G["pixel_values"]
    E --> H["image_grid_thw"]

    F --> I["Sequence(token_ids)"]
    G --> J["seq.pixel_values"]
    H --> K["seq.image_grid_thw"]
    J --> L["Scheduler.add(seq)"]
    K --> L
    I --> L
    D --> L

    L --> M["Scheduler.schedule"]
    M --> N["ModelRunner.run"]
    N --> O["Model forward"]
    O --> P["Sampler"]
    P --> Q["输出 text / token_ids"]
```

---

## 15. 最终结论

qwen3.6 版本的 `llm_engine.py` 相比原版主要完成了两件事：

第一，**扩展输入形态**。原版只支持文本和 token ids；新版本支持多模态 messages，并通过 `process_messages` 生成 `token_ids`、`pixel_values`、`image_grid_thw`，把图像相关信息挂到 `Sequence` 上，为后续模型执行提供输入。

第二，**增强资源释放稳定性**。新版本通过 `_exited`、`hasattr` 和 `self.ps.clear()` 让 `exit()` 更安全，避免重复调用导致异常或进程状态污染。

因此，这个文件的核心意义是：

```text
让 Engine 层具备多模态请求入口能力，同时保持原有 Scheduler / ModelRunner / generate 主循环基本不变。
```

需要注意的是：

```text
Qwen3.6 的 hybrid / GatedDeltaNet / recurrent state 等模型结构改造，不是在这个文件中体现的。
```

要完整理解 Qwen3.6 的工程支持，还需要继续对比 `config.py`、`model_runner.py`、模型文件、attention/state 相关文件。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
