# loader.py 源码宏观解析

## 1. 文件整体定位

`loader.py` 是 nano-vLLM 中负责 **从 HuggingFace safetensors 权重文件加载模型参数** 的工具文件。

它不在模型 forward 中执行，也不属于 Transformer 的某个计算层。它发生在模型初始化阶段：

```text
ModelRunner 初始化
  -> 创建 Qwen3ForCausalLM
  -> load_model(model, config.model)
  -> 从 safetensors 读取权重
  -> 写入 nano-vLLM 模型参数
  -> warmup / allocate KV Cache / capture CUDA Graph
```

在 `model_runner.py` 中调用位置是：

```python
self.model = Qwen3ForCausalLM(hf_config)
load_model(self.model, config.model)
```

所以 `loader.py` 的作用是：把磁盘上的模型权重，正确放进 nano-vLLM 自己定义的模型结构里。

它尤其重要的一点是：nano-vLLM 的模型结构和 HuggingFace 原始权重结构并不完全一一对应。

例如 HuggingFace 里可能有：

```text
q_proj.weight
k_proj.weight
v_proj.weight
gate_proj.weight
up_proj.weight
```

但 nano-vLLM 里为了推理性能和张量并行，会合并成：

```text
qkv_proj.weight
gate_up_proj.weight
```

`loader.py` 就负责处理这种“原始权重名”和“运行时参数名”之间的映射。

## 2. 这个文件要解决的核心问题

`loader.py` 主要解决的是 **权重加载问题**，同时也间接服务于 **并行切分问题** 和 **合并权重问题**。

| 问题 | 说明 |
|---|---|
| 权重加载问题 | 从 `.safetensors` 文件中读取 tensor 并写入模型参数 |
| 权重名映射问题 | 把 HF 的 `q_proj/k_proj/v_proj` 映射到 nano-vLLM 的 `qkv_proj` |
| 并行切分问题 | 调用参数自带的 `weight_loader`，让每个 TP rank 只加载自己的切片 |
| 模型结构适配问题 | 适配 nano-vLLM 合并 QKV、合并 gate/up 的模型结构 |
| 张量计算问题 | 不涉及 forward 计算 |
| 采样输出问题 | 不涉及 |

它和 Transformer 结构的关系是：它不定义 Transformer 的数学计算，但它决定 Transformer 每一层用的权重是否被正确加载。

如果 `loader.py` 出错，后果通常不是 shape 小问题，而是模型完全无法正确推理：

- q/k/v 权重放错位置，Attention 会错；
- gate/up 权重放错位置，MLP 会错；
- TP 切片加载错误，多卡推理会错；
- embedding/lm_head 切片错误，token 输入输出会错。

所以这个文件虽然短，但它是模型能否正常运行的前置条件。

## 3. 代码结构总览

### 3.1 文件导入

```python
import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
```

含义如下：

| 导入 | 作用 |
|---|---|
| `os` | 拼接模型路径 |
| `glob` | 找到目录下所有 `.safetensors` 文件 |
| `torch` | Tensor 类型 |
| `nn` | Parameter 类型 |
| `safe_open` | 读取 safetensors 权重文件 |

这里使用的是 `safetensors`，不是传统的 PyTorch `.bin`。

`safetensors` 的优势是：

- 加载更安全，不执行任意 pickle 代码；
- 适合大模型分片权重；
- HuggingFace 模型常用这种格式。

### 3.2 `default_weight_loader`

```python
def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)
```

这是默认权重加载方式。

含义很直接：

```text
把 loaded_weight 原样复制到 param.data
```

适用于不需要特殊切分或合并的参数，例如某些 RMSNorm 权重：

```text
model.layers.0.input_layernorm.weight
```

如果某个参数没有自定义 `weight_loader`，就用这个默认函数。

### 3.3 `load_model`

```python
def load_model(model: nn.Module, path: str):
```

这是整个文件的核心函数。

输入：

| 参数 | 含义 |
|---|---|
| `model` | 已经构造好的 nano-vLLM 模型，例如 `Qwen3ForCausalLM` |
| `path` | HuggingFace 模型目录，里面包含 `.safetensors` 文件 |

核心流程可以概括为：

```text
读取 packed_modules_mapping
遍历所有 safetensors 文件
  遍历文件里的每个 weight_name
    如果 weight_name 属于需要合并的模块
      映射到 nano-vLLM 参数名
      调用目标参数的特殊 weight_loader
    否则
      按原始名字找到参数
      调用参数自己的 loader 或默认 copy
```

## 4. 权重加载流程和数据流

### 4.1 获取 packed_modules_mapping

代码：

```python
packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
```

在 `Qwen3ForCausalLM` 中定义了：

```python
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
```

这个映射的含义是：

| HF 原始模块名 | nano-vLLM 目标模块名 | shard_id |
|---|---|---|
| `q_proj` | `qkv_proj` | `"q"` |
| `k_proj` | `qkv_proj` | `"k"` |
| `v_proj` | `qkv_proj` | `"v"` |
| `gate_proj` | `gate_up_proj` | `0` |
| `up_proj` | `gate_up_proj` | `1` |

它告诉 loader：

```text
遇到 q_proj.weight，不要找 q_proj.weight 参数；
要把它放进 qkv_proj.weight 的 q 分片。
```

这就是权重打包加载的关键。

### 4.2 遍历 safetensors 文件

代码：

```python
for file in glob(os.path.join(path, "*.safetensors")):
    with safe_open(file, "pt", "cpu") as f:
```

这会找到模型目录下所有 `.safetensors` 文件。

很多大模型不是一个权重文件，而是多个分片文件，例如：

```text
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
...
```

`glob` 会逐个打开它们。

`safe_open(file, "pt", "cpu")` 表示：

- 用 PyTorch tensor 格式读取；
- 先把权重读到 CPU；
- 后面再 copy 到模型参数所在设备。

因为 `ModelRunner` 初始化时设置了：

```python
torch.set_default_device("cuda")
self.model = Qwen3ForCausalLM(hf_config)
```

所以模型参数通常已经在当前 GPU 上，而 safetensors 先从 CPU 读出，再复制到参数中。

### 4.3 遍历权重名

代码：

```python
for weight_name in f.keys():
```

`weight_name` 是 safetensors 中保存的参数名。

例如可能是：

```text
model.embed_tokens.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.down_proj.weight
model.layers.0.input_layernorm.weight
```

loader 需要判断这些名字是否能直接对应 nano-vLLM 参数，还是需要先改名/合并。

### 4.4 处理 packed 权重

代码：

```python
for k in packed_modules_mapping:
    if k in weight_name:
        v, shard_id = packed_modules_mapping[k]
        param_name = weight_name.replace(k, v)
        param = model.get_parameter(param_name)
        weight_loader = getattr(param, "weight_loader")
        weight_loader(param, f.get_tensor(weight_name), shard_id)
        break
```

这段是 `loader.py` 最关键的逻辑。

以 QKV 为例。

原始权重名：

```text
model.layers.0.self_attn.q_proj.weight
```

匹配到：

```python
k = "q_proj"
v = "qkv_proj"
shard_id = "q"
```

替换后：

```text
model.layers.0.self_attn.qkv_proj.weight
```

然后：

```python
param = model.get_parameter(param_name)
```

找到 nano-vLLM 中的目标参数。

最后调用：

```python
weight_loader(param, loaded_weight, "q")
```

这个 `weight_loader` 不是普通 copy，而是 `QKVParallelLinear.weight_loader`，它会把 q/k/v 权重放到合并矩阵的正确位置，并按 TP rank 切片。

MLP 的 gate/up 也是类似：

```text
gate_proj.weight -> gate_up_proj.weight 的第 0 段
up_proj.weight   -> gate_up_proj.weight 的第 1 段
```

对应：

```python
weight_loader(param, loaded_weight, 0)
weight_loader(param, loaded_weight, 1)
```

### 4.5 处理普通权重

如果没有命中 packed mapping，就走 `else` 分支：

```python
else:
    param = model.get_parameter(weight_name)
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, f.get_tensor(weight_name))
```

这里表示权重名可以直接对应 nano-vLLM 模型中的参数。

例如：

```text
model.layers.0.self_attn.o_proj.weight
model.layers.0.mlp.down_proj.weight
model.layers.0.input_layernorm.weight
```

不过即使是普通名字，也不一定是简单 copy。

例如：

- `RowParallelLinear.weight` 有自己的 `weight_loader`，会按输入维度切片；
- `VocabParallelEmbedding.weight` 有自己的 `weight_loader`，会按 vocab 维度切片；
- `RMSNorm.weight` 没有特殊 loader，就用 `default_weight_loader`。

这说明 loader 的设计是“分发式”的：

```text
loader.py 只负责找到参数
具体怎么加载由参数自己的 weight_loader 决定
```

### 4.6 和 tensor parallel 的关系

`loader.py` 自己不直接调用 `dist.get_rank()`，也不直接写切片逻辑。

但它调用的参数 `weight_loader` 会处理 TP：

| 参数类型 | loader 行为 |
|---|---|
| `ColumnParallelLinear` | 沿 output 维度切分 |
| `RowParallelLinear` | 沿 input 维度切分 |
| `QKVParallelLinear` | 合并 q/k/v，并按 head 分片 |
| `MergedColumnParallelLinear` | 合并 gate/up，并按 intermediate 分片 |
| `VocabParallelEmbedding` | 沿 vocab 维度切分 |
| 普通参数 | 直接 copy |

所以多卡加载时，每个 rank 都会运行 `load_model`，但每个 rank 的 `weight_loader` 会只取自己需要的那一片权重。

这就是 nano-vLLM 支持 tensor parallel 的关键机制之一。

## 5. 和 Transformer / Qwen 模型结构的关系

### 5.1 和 Qwen3Attention 的关系

Qwen3Attention 中实际模块是：

```python
self.qkv_proj = QKVParallelLinear(...)
```

但 HuggingFace 原始权重通常是：

```text
q_proj.weight
k_proj.weight
v_proj.weight
```

所以 loader 必须把三份原始权重装进一份合并权重：

```text
q_proj.weight
k_proj.weight
v_proj.weight
  -> qkv_proj.weight
```

其中 `"q" / "k" / "v"` 这个 `shard_id` 告诉 `QKVParallelLinear.weight_loader` 当前加载的是哪一段。

如果不做这个映射，`model.get_parameter("...q_proj.weight")` 会找不到参数，因为 nano-vLLM 模型里没有单独的 `q_proj`。

### 5.2 和 Qwen3MLP 的关系

Qwen3MLP 中实际模块是：

```python
self.gate_up_proj = MergedColumnParallelLinear(...)
```

但 HuggingFace 原始权重通常是：

```text
gate_proj.weight
up_proj.weight
```

所以 loader 要做：

```text
gate_proj.weight -> gate_up_proj.weight 的第 0 段
up_proj.weight   -> gate_up_proj.weight 的第 1 段
```

这正好对应 `activation.py` 中的：

```python
x, y = x.chunk(2, -1)
return F.silu(x) * y
```

也就是说：

- loader 保证 `gate_proj` 权重在前半段；
- loader 保证 `up_proj` 权重在后半段；
- `SiluAndMul` 才能正确把前半段当 gate、后半段当 up。

### 5.3 和 embedding / LM Head 的关系

`embed_head.py` 中：

```python
self.weight.weight_loader = self.weight_loader
```

因此 embedding 和 LM Head 权重加载时，会按 vocab 维度切分。

对于：

```text
model.embed_tokens.weight
lm_head.weight
```

如果它们没有进入 packed mapping，loader 会按原名找到参数，然后调用参数自己的 `weight_loader`。

这让每个 tensor parallel rank 只持有一部分 vocab 权重。

### 5.4 和 RMSNorm 等普通参数的关系

`RMSNorm.weight` 没有自定义 `weight_loader`，因此使用：

```python
default_weight_loader
```

也就是直接复制完整权重。

这是合理的，因为 RMSNorm 的权重 shape 通常是：

```text
[hidden_size]
```

在当前 nano-vLLM 实现中，它不做单独切分。

### 5.5 和推理系统初始化的关系

`load_model` 在 KV Cache 分配之前执行：

```text
创建模型
加载权重
warmup_model
allocate_kv_cache
capture_cudagraph
```

这个顺序很重要：

1. 先创建模型结构；
2. 再加载权重；
3. warmup 时用真实模型权重跑一遍；
4. 根据显存情况分配 KV Cache；
5. 如果启用 CUDA Graph，再 capture decode 图。

所以 `loader.py` 属于推理服务启动阶段的关键环节，不属于每个请求的热路径。

## 6. 学习总结

`loader.py` 的核心价值可以概括为一句话：

> 它把 HuggingFace safetensors 中的原始权重，按照 nano-vLLM 的合并模块和 tensor parallel 切分规则，加载到运行时模型参数中。

学习这个文件要抓住三条主线：

1. **权重名映射主线**：`q_proj/k_proj/v_proj -> qkv_proj`，`gate_proj/up_proj -> gate_up_proj`；
2. **参数分发主线**：`loader.py` 找参数，具体怎么 copy/切片/合并由参数自己的 `weight_loader` 决定；
3. **初始化流程主线**：模型必须先正确加载权重，后续 warmup、KV Cache 分配、CUDA Graph 才有意义。

它和前面几个文件的关系可以这样理解：

| 文件 | 负责内容 | 和 `loader.py` 的关系 |
|---|---|---|
| `qwen3.py` | 定义模型结构和 `packed_modules_mapping` | 告诉 loader 哪些权重要改名合并 |
| `linear.py` | 定义并行 Linear 的 `weight_loader` | 决定 QKV/MLP/o_proj 权重如何切分加载 |
| `embed_head.py` | 定义 vocab parallel embedding/head | 决定词表权重如何切分加载 |
| `layernorm.py` | 定义 RMSNorm 权重 | 通常走默认 copy |
| `model_runner.py` | 调用 `load_model` | 在模型初始化阶段触发权重加载 |

对 AI Infra 推理学习来说，`loader.py` 很适合帮你理解一个工程事实：推理框架不是“定义模型结构然后直接跑”这么简单，真实权重文件的命名、合并、切片、并行分布都必须和运行时模型结构严格对齐。



%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]

%% 项目关联导航：结束 %%
