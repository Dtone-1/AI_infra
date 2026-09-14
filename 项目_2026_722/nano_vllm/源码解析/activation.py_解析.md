# activation.py 源码宏观解析

## 1. 文件整体定位

`activation.py` 是 nano-vLLM 中负责 **MLP 激活函数计算** 的 layer 文件。

它在 `models/qwen3.py` 中被 Qwen3 的 MLP 模块使用：

```python
from nanovllm.layers.activation import SiluAndMul

self.act_fn = SiluAndMul()
```

在 Qwen3MLP 的 forward 中：

```python
gate_up = self.gate_up_proj(x)
x = self.act_fn(gate_up)
x = self.down_proj(x)
```

所以它位于 Transformer Decoder Layer 的 **Feed Forward Network / MLP 子模块** 中，负责完成从 `gate_up_proj` 到 `down_proj` 之间的非线性变换。

整体位置可以理解为：

```text
hidden_states
  -> RMSNorm
  -> Self-Attention
  -> RMSNorm
  -> MLP:
       gate_up_proj
       -> SiluAndMul
       -> down_proj
  -> next layer
```

这个文件不是调度器、不是 KV Cache、不是采样器，也不负责权重加载。它只做一个非常核心的小算子：

```text
silu(gate) * up
```

这就是 Qwen/LLaMA 等模型中常见的 SwiGLU 类 MLP 激活结构。

## 2. 这个文件要解决的核心问题

`activation.py` 主要解决的是 **模型结构问题** 和 **张量计算问题**。

| 问题 | 说明 |
|---|---|
| 模型结构问题 | 实现 Qwen3 MLP 中的 gated activation |
| 张量计算问题 | 把一个合并张量切成两半，并执行 `SiLU(x) * y` |
| 权重加载问题 | 不涉及，这个文件没有参数 |
| 并行切分问题 | 不直接通信，但会处理张量并行后的局部 MLP 张量 |
| 采样输出问题 | 不涉及 |

它对应原始 Transformer 中 FFN/MLP 的激活函数部分。

原始 Transformer FFN 通常是：

```text
x -> Linear -> ReLU/GELU -> Linear
```

而 Qwen3 这类现代大语言模型常用 gated MLP，形式更接近：

```text
x -> gate_proj, up_proj
gate = silu(gate_proj(x))
up = up_proj(x)
output = gate * up
output -> down_proj
```

nano-vLLM 中为了减少线性层调用次数，会用 `MergedColumnParallelLinear` 把 `gate_proj` 和 `up_proj` 合并成一个 `gate_up_proj`：

```text
x -> gate_up_proj -> [gate, up] -> silu(gate) * up -> down_proj
```

`activation.py` 负责的正是中间这一步：

```text
[gate, up] -> silu(gate) * up
```

它和高性能推理的关系在于：

1. MLP 是大模型 decoder layer 中计算量很大的部分；
2. `gate_proj` 和 `up_proj` 合并后，需要一个高效的激活模块处理合并输出；
3. `@torch.compile` 可以让 PyTorch 尝试编译优化这个小算子；
4. 这个激活函数虽然代码短，但每层都会执行，属于高频路径。

## 3. 代码结构总览

### 3.1 文件导入

```python
import torch
from torch import nn
import torch.nn.functional as F
```

含义如下：

| 导入 | 作用 |
|---|---|
| `torch` | 张量类型与编译装饰器 |
| `nn` | 定义 PyTorch Module |
| `F` | 使用 `F.silu` 激活函数 |

这个文件没有导入 `torch.distributed`，说明它本身不负责跨 GPU 通信。

### 3.2 `SiluAndMul`

文件中只定义了一个类：

```python
class SiluAndMul(nn.Module):
```

它是一个无参数模块，没有 `__init__`，只有 `forward`。

这说明它不像 `RMSNorm`、`Linear`、`Embedding` 那样有可学习权重。它只是对输入张量做固定数学变换。

### 3.3 `forward`

核心代码：

```python
@torch.compile
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x, y = x.chunk(2, -1)
    return F.silu(x) * y
```

逐步理解：

1. `@torch.compile`：让 PyTorch 对这个 forward 进行编译优化；
2. `x.chunk(2, -1)`：沿最后一维把输入切成两半；
3. `F.silu(x)`：对第一半做 SiLU 激活；
4. `* y`：和第二半逐元素相乘；
5. 返回相乘后的结果。

SiLU 的公式是：

```text
silu(x) = x * sigmoid(x)
```

因此整个模块的公式是：

```text
output = silu(x_left) * x_right
```

在 MLP 语义中：

```text
x_left  = gate
x_right = up
output  = silu(gate) * up
```

## 4. 张量流和 forward 流程

### 4.1 输入是什么

`SiluAndMul` 的输入来自 Qwen3MLP：

```python
gate_up = self.gate_up_proj(x)
x = self.act_fn(gate_up)
```

其中 `gate_up_proj` 是：

```python
MergedColumnParallelLinear(
    hidden_size,
    [intermediate_size] * 2,
    bias=False,
)
```

所以它输出的是 `gate_proj` 和 `up_proj` 合并后的结果。

在单卡情况下：

```text
gate_up: [num_tokens, 2 * intermediate_size]
```

在 tensor parallel 多卡情况下，因为 `MergedColumnParallelLinear` 是列并行，每个 rank 只得到一部分 intermediate 维度：

```text
gate_up: [num_tokens, 2 * intermediate_size / tp_size]
```

### 4.2 `chunk(2, -1)` 后 shape 如何变化

代码：

```python
x, y = x.chunk(2, -1)
```

这里第二个 `x` 是局部变量，会覆盖输入变量名。为了理解清楚，可以把它改写成概念形式：

```python
gate, up = gate_up.chunk(2, dim=-1)
```

单卡情况下：

```text
gate_up: [num_tokens, 2 * intermediate_size]
gate:    [num_tokens, intermediate_size]
up:      [num_tokens, intermediate_size]
```

多卡 tensor parallel 情况下：

```text
gate_up: [num_tokens, 2 * intermediate_size / tp_size]
gate:    [num_tokens, intermediate_size / tp_size]
up:      [num_tokens, intermediate_size / tp_size]
```

也就是说，它始终沿最后一维切成两半。

### 4.3 激活和相乘

代码：

```python
return F.silu(x) * y
```

对应：

```text
output = silu(gate) * up
```

输出 shape 和 `gate`、`up` 相同。

单卡：

```text
output: [num_tokens, intermediate_size]
```

多卡：

```text
output: [num_tokens, intermediate_size / tp_size]
```

这个输出会继续进入 `down_proj`：

```python
x = self.down_proj(x)
```

而 `down_proj` 是 `RowParallelLinear`，会把多卡上的局部 intermediate 结果通过 `all_reduce` 聚合回：

```text
[num_tokens, hidden_size]
```

### 4.4 batch、sequence、hidden size 在哪里体现

nano-vLLM 的模型 forward 通常把 batch 内 token 展平成一维 token 列表，所以激活层看到的是：

```text
[num_tokens, feature_dim]
```

这里：

| 维度 | 含义 |
|---|---|
| `num_tokens` | 当前调度步参与计算的 token 数 |
| `feature_dim` | MLP 的中间维度，通常是 `intermediate_size` 或其 TP 分片 |

prefill 阶段：

```text
num_tokens = 本轮所有 prompt token 数
```

decode 阶段：

```text
num_tokens = 当前 batch 中 sequence 数
```

`activation.py` 不关心 token 属于哪条 sequence，也不关心 position。它只对每个 token 的 MLP 中间向量逐元素计算。

## 5. 和 Transformer / Qwen 模型结构的关系

### 5.1 和原始 FFN 的关系

原始 Transformer FFN 可以简化理解为：

```text
FFN(x) = Linear2(Activation(Linear1(x)))
```

比如：

```text
x -> Linear -> ReLU/GELU -> Linear
```

Qwen3 的 MLP 更像 gated FFN：

```text
MLP(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
```

`activation.py` 实现的是中间的：

```text
silu(gate_proj(x)) * up_proj(x)
```

### 5.2 和 SwiGLU 的关系

`SiluAndMul` 对应的是 SwiGLU 类结构。

从直觉上理解：

```text
up_proj(x)
```

提供主要的中间特征；

```text
silu(gate_proj(x))
```

提供一个门控系数；

两者相乘：

```text
silu(gate) * up
```

表示模型可以动态控制哪些中间特征更重要。

所以它不是普通的“激活一下”，而是带门控的 MLP 表达方式。

### 5.3 和 Qwen3MLP 的关系

Qwen3MLP 的完整代码结构是：

```python
self.gate_up_proj = MergedColumnParallelLinear(
    hidden_size,
    [intermediate_size] * 2,
    bias=False,
)
self.down_proj = RowParallelLinear(
    intermediate_size,
    hidden_size,
    bias=False,
)
assert hidden_act == "silu"
self.act_fn = SiluAndMul()
```

forward：

```python
gate_up = self.gate_up_proj(x)
x = self.act_fn(gate_up)
x = self.down_proj(x)
return x
```

可以对应成：

```text
hidden_states
  -> gate_up_proj
       = gate_proj + up_proj 的合并版本
  -> SiluAndMul
       = silu(gate) * up
  -> down_proj
       = intermediate_size -> hidden_size
```

这正是 Qwen3 Decoder Layer 中 MLP 的核心计算路径。

### 5.4 和 tensor parallel 的关系

`activation.py` 自己不做通信，但它处理的是已经被 `MergedColumnParallelLinear` 切分后的局部张量。

多卡情况下：

```text
rank 0: 负责一部分 intermediate channels
rank 1: 负责另一部分 intermediate channels
```

每个 rank 都在自己的局部张量上执行：

```text
silu(local_gate) * local_up
```

然后交给 `RowParallelLinear` 的 `down_proj`：

```text
local_output -> all_reduce -> full hidden_states
```

所以它虽然不是并行通信模块，但它位于张量并行 MLP 的中间路径上。

### 5.5 和推理性能的关系

这个文件只有几行代码，但它在每个 decoder layer 的 MLP 中都会被调用。

性能相关点有三个：

1. `chunk(2, -1)` 避免分别计算 `gate_proj` 和 `up_proj` 后再手动组织；
2. `F.silu(x) * y` 是逐元素操作，主要受显存读写和 kernel 调度影响；
3. `@torch.compile` 尝试把这个小函数编译优化，减少 Python 层开销。

在大模型推理中，MLP 的大矩阵乘法是主计算量，但高频逐元素激活也不能完全忽略。尤其 decode 阶段 batch 较小时，小算子调度开销会更明显。

## 6. 学习总结

`activation.py` 的核心价值可以概括为一句话：

> 它实现了 Qwen3 MLP 中的 `silu(gate) * up` 门控激活，把合并线性层 `gate_up_proj` 的输出转换成 `down_proj` 可以继续处理的中间特征。

学习这个文件要抓住三条主线：

1. **模型结构主线**：它属于 Qwen3 MLP，不属于 Attention；
2. **张量 shape 主线**：输入最后一维是两倍 intermediate 分片，切成 `gate` 和 `up` 后再相乘；
3. **推理优化主线**：它配合 `MergedColumnParallelLinear` 减少线性层调用，并用 `torch.compile` 优化高频小算子。

它和前面几个文件的关系可以这样理解：

| 文件 | 负责内容 | 在模型中的位置 |
|---|---|---|
| `layernorm.py` | RMSNorm 与 residual 融合 | Attention/MLP 之前 |
| `linear.py` | `gate_up_proj`、`down_proj` 等线性层 | MLP 主体 |
| `activation.py` | `silu(gate) * up` | `gate_up_proj` 和 `down_proj` 中间 |
| `embed_head.py` | token embedding 与 LM Head | 模型输入/输出 |

对 AI Infra 推理学习来说，这个文件能帮助你把 MLP 的结构彻底串起来：`MergedColumnParallelLinear` 不是随便把两个输出拼起来，而是为了后续 `SiluAndMul` 一次性完成 gated activation，然后再通过 `RowParallelLinear` 回到 hidden size。



%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]

%% 项目关联导航：结束 %%
