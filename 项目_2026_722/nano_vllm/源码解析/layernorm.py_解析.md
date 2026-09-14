# layernorm.py 源码宏观解析

## 1. 文件整体定位

`layernorm.py` 是 nano-vLLM 中负责 **RMSNorm 归一化层** 的 layer 文件。

它在模型 forward 中主要出现在 Qwen3 Decoder Layer 的几个关键位置：

```python
self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

在 Attention 内部，如果没有 `qkv_bias`，还会对 Q/K 做单独 RMSNorm：

```python
self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
```

所以这个文件负责的是 Transformer/Qwen 模型中的 **归一化计算子模块**。

它不负责调度、不负责 KV Cache、不负责采样，也不负责张量并行切分。它的核心任务是：在每一层 Transformer Block 内部稳定 hidden states 的数值分布，让后续 Attention 和 MLP 计算更稳定。

在 Qwen3 的整体流程中，它大致处在这里：

```text
hidden_states
  -> RMSNorm
  -> Self-Attention
  -> RMSNorm
  -> MLP
  -> RMSNorm(final)
  -> LM Head
```

## 2. 这个文件要解决的核心问题

`layernorm.py` 主要解决两个问题：

| 问题 | 说明 |
|---|---|
| 模型结构问题 | 实现 Qwen3 使用的 RMSNorm 层 |
| 张量计算问题 | 对 hidden states 按最后一维做归一化，并乘以可学习权重 |

它和原始 Transformer 的关系是：原始 Transformer 常用 LayerNorm，而 LLaMA/Qwen 这类模型常用 RMSNorm。

LayerNorm 通常会做：

```text
x_norm = (x - mean) / sqrt(var + eps)
```

RMSNorm 不减均值，只按均方根缩放：

```text
x_norm = x / sqrt(mean(x^2) + eps)
```

因此 RMSNorm 计算更简单，常见于现代大语言模型。

它和高性能推理的关系在于：

1. RMSNorm 出现在每一层 decoder block 中，调用频率很高；
2. 它会处理所有参与推理的 token；
3. nano-vLLM 用 `torch.compile` 编译归一化函数，减少 Python 调度开销；
4. `add_rms_forward` 把 residual add 和 RMSNorm 合在一起，减少中间张量和额外操作。

这个文件虽然短，但属于模型 forward 中的高频小算子。

## 3. 代码结构总览

### 3.1 文件导入

```python
import torch
from torch import nn
```

这里只依赖 PyTorch。

和 `linear.py`、`embed_head.py` 不同，这个文件没有使用 `torch.distributed`，说明 RMSNorm 本身不做 tensor parallel 通信。

每个 rank 只对自己本地持有的 hidden states 做归一化即可。

### 3.2 `RMSNorm`

文件中只定义了一个类：

```python
class RMSNorm(nn.Module):
```

它是一个 PyTorch module，用来实现 RMSNorm。

初始化函数：

```python
def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(hidden_size))
```

核心变量：

| 变量 | 作用 |
|---|---|
| `hidden_size` | 要归一化的最后一维大小 |
| `eps` | 防止除零的小常数 |
| `weight` | 可学习缩放参数，shape 为 `[hidden_size]` |

`weight` 初始为全 1，训练后会变成每个 hidden 维度对应的缩放系数。推理时它从模型权重中加载。

### 3.3 `rms_forward`

```python
@torch.compile
def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
```

这是普通 RMSNorm 路径，用于没有额外 residual 输入的情况。

计算流程：

```python
orig_dtype = x.dtype
x = x.float()
var = x.pow(2).mean(dim=-1, keepdim=True)
x.mul_(torch.rsqrt(var + self.eps))
x = x.to(orig_dtype).mul_(self.weight)
return x
```

逐步理解：

1. 记录原始 dtype，例如 `float16` 或 `bfloat16`；
2. 转成 `float32` 做归一化，提高数值稳定性；
3. 沿最后一维计算 `mean(x^2)`；
4. 用 `rsqrt` 计算 `1 / sqrt(var + eps)`；
5. 原地乘上缩放因子；
6. 转回原 dtype；
7. 乘以可学习参数 `weight`。

其中：

```python
var = x.pow(2).mean(dim=-1, keepdim=True)
```

这里的 `var` 严格来说不是 LayerNorm 里的方差，而是 RMSNorm 中的均方值。

### 3.4 `add_rms_forward`

```python
@torch.compile
def add_rms_forward(
    self,
    x: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
```

这是带 residual 的 RMSNorm 路径。

它做两件事：

```text
先把 x 和 residual 相加
再对相加后的结果做 RMSNorm
```

代码：

```python
x = x.float().add_(residual.float())
residual = x.to(orig_dtype)
```

这里先得到新的 residual：

```text
residual = x + old_residual
```

然后对这个结果做 RMSNorm：

```python
var = x.pow(2).mean(dim=-1, keepdim=True)
x.mul_(torch.rsqrt(var + self.eps))
x = x.to(orig_dtype).mul_(self.weight)
return x, residual
```

返回值有两个：

| 返回值 | 含义 |
|---|---|
| `x` | 归一化后的 hidden states，送入下一个子模块 |
| `residual` | 未归一化的残差累积值，供后续残差连接继续使用 |

这就是 nano-vLLM 中常见的 fused residual add + RMSNorm 写法。

### 3.5 `forward`

```python
def forward(
    self,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
```

`forward` 根据是否传入 `residual` 选择不同路径：

```python
if residual is None:
    return self.rms_forward(x)
else:
    return self.add_rms_forward(x, residual)
```

也就是说：

- 没有 residual：只做 RMSNorm；
- 有 residual：先做 residual add，再做 RMSNorm，并返回新的 residual。

这个设计是为了配合 `qwen3.py` 中 decoder layer 的残差流动。

## 4. 张量流和 forward 流程

### 4.1 输入输出 shape

RMSNorm 主要处理 hidden states。

在 nano-vLLM 中，hidden states 常见形状是：

```text
[num_tokens, hidden_size]
```

其中：

| 维度 | 含义 |
|---|---|
| `num_tokens` | 当前推理步参与计算的 token 数 |
| `hidden_size` | 模型隐藏维度 |

prefill 阶段：

```text
num_tokens = 本轮所有 prompt token 数
```

decode 阶段：

```text
num_tokens = 当前 batch 中 sequence 数
```

RMSNorm 沿最后一维做归一化：

```python
mean(dim=-1, keepdim=True)
```

所以每个 token 的 hidden vector 单独归一化，不会跨 token 混合信息。

### 4.2 普通 RMSNorm 流程

输入：

```text
x: [num_tokens, hidden_size]
```

计算：

```text
var: [num_tokens, 1]
x_norm: [num_tokens, hidden_size]
weight: [hidden_size]
output: [num_tokens, hidden_size]
```

公式可以写成：

```text
output = weight * x / sqrt(mean(x^2) + eps)
```

这里的 `mean(x^2)` 是对每个 token 的 hidden 维度求平均。

### 4.3 带 residual 的 RMSNorm 流程

输入：

```text
x:        [num_tokens, hidden_size]
residual: [num_tokens, hidden_size]
```

先做残差相加：

```text
residual_new = x + residual
```

再对 `residual_new` 做 RMSNorm：

```text
x_norm = RMSNorm(residual_new)
```

输出：

```text
x_norm:       [num_tokens, hidden_size]
residual_new: [num_tokens, hidden_size]
```

这对应 Qwen3 Decoder Layer 中的写法：

```python
hidden_states, residual = self.input_layernorm(hidden_states, residual)
```

这里 `hidden_states` 是归一化后送入 Attention 或 MLP 的输入，而 `residual` 是保留下来的残差主线。

### 4.4 在 Qwen3DecoderLayer 中的完整流动

`Qwen3DecoderLayer.forward` 中的核心代码是：

```python
if residual is None:
    hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
else:
    hidden_states, residual = self.input_layernorm(hidden_states, residual)

hidden_states = self.self_attn(positions, hidden_states)
hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
hidden_states = self.mlp(hidden_states)
return hidden_states, residual
```

可以拆成三段理解。

第一层刚开始时，`residual is None`：

```text
hidden_states from embedding
  -> RMSNorm
  -> Attention
```

同时把原始 embedding 输出保存为 residual：

```text
residual = hidden_states_before_norm
```

Attention 结束后：

```text
attention_output + residual
  -> RMSNorm
  -> MLP
```

MLP 输出暂时不立刻加 residual，而是返回给下一层。下一层开头再执行：

```text
previous_mlp_output + residual
  -> RMSNorm
  -> Attention
```

所以这个实现不是每一步都显式写：

```text
x = x + sublayer(x)
x = norm(x)
```

而是通过 `hidden_states, residual` 两条变量线，把 residual add 和 RMSNorm 融合到下一次 norm 调用里。

### 4.5 Q/K RMSNorm 的 shape

在 `Qwen3Attention` 中还可能出现：

```python
q = self.q_norm(q)
k = self.k_norm(k)
```

此时输入 shape 是：

```text
q: [num_tokens, num_heads, head_dim]
k: [num_tokens, num_kv_heads, head_dim]
```

对应的 `RMSNorm` 初始化维度是：

```python
RMSNorm(self.head_dim, eps=rms_norm_eps)
```

因此它沿最后一维 `head_dim` 做归一化：

```text
每个 token、每个 head 内部单独做 RMSNorm
```

这说明 `RMSNorm` 不只可以处理 `[num_tokens, hidden_size]`，也可以处理 `[num_tokens, num_heads, head_dim]`。关键是最后一维必须等于 `weight` 的长度。

## 5. 和 Transformer / Qwen 模型结构的关系

### 5.1 和 LayerNorm 的区别

传统 Transformer 常见 LayerNorm：

```text
x_norm = (x - mean(x)) / sqrt(var(x) + eps)
```

RMSNorm：

```text
x_norm = x / sqrt(mean(x^2) + eps)
```

区别是：

| 对比项 | LayerNorm | RMSNorm |
|---|---|---|
| 是否减均值 | 是 | 否 |
| 是否计算方差 | 是 | 否，计算均方值 |
| 参数 | 通常有 weight 和 bias | 这里仅有 weight |
| 计算复杂度 | 略高 | 略低 |
| 常见模型 | 原始 Transformer、BERT 等 | LLaMA、Qwen 等 |

Qwen3 使用 RMSNorm，所以 nano-vLLM 实现的是 `RMSNorm`，不是普通 `LayerNorm`。

### 5.2 和 Pre-Norm Decoder 的关系

现代大语言模型通常使用 Pre-Norm 结构，也就是先归一化，再进入 Attention/MLP：

```text
RMSNorm -> Attention
RMSNorm -> MLP
```

这和原始 Transformer 论文中的 Post-Norm 有差别。

在 Qwen3DecoderLayer 中：

```python
hidden_states = self.input_layernorm(...)
hidden_states = self.self_attn(...)
hidden_states = self.post_attention_layernorm(...)
hidden_states = self.mlp(...)
```

可以看到 Attention 和 MLP 前面都有 RMSNorm。

### 5.3 和残差连接的关系

Transformer Block 中残差连接非常重要。概念上可以理解为：

```text
x = x + Attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

nano-vLLM 的实现把这件事拆成 `hidden_states` 和 `residual` 两条线：

| 变量 | 作用 |
|---|---|
| `hidden_states` | 当前子模块输出，准备进入下一步计算 |
| `residual` | 保存残差累积主线 |

`add_rms_forward` 做的就是：

```text
hidden_states + residual
  -> 得到新的 residual
  -> 对新的 residual 做 RMSNorm
  -> 得到下一个子模块输入
```

这样写的好处是减少显式的加法和中间张量，让推理 forward 更紧凑。

### 5.4 和推理性能的关系

RMSNorm 本身不是 FLOPs 最大的部分，真正重的是 Attention 和 MLP 的大矩阵乘法。但 RMSNorm 有几个特点：

1. 每层都会调用，累计次数多；
2. 每次都要读写 hidden states，受显存带宽影响；
3. 小算子太多会增加 kernel launch 或 Python 调度开销；
4. residual add + norm 如果分开写，会产生更多中间读写。

因此 nano-vLLM 使用：

```python
@torch.compile
```

让 PyTorch 尽量编译优化这些小函数。

这体现了推理优化中的一个常见思路：不仅要优化大矩阵乘法，也要减少高频小算子的额外开销。

## 6. 学习总结

`layernorm.py` 的核心价值可以概括为一句话：

> 它实现了 Qwen3 推理中的 RMSNorm，并把 residual add 与 RMSNorm 融合在一起，为 Attention、MLP 和最终输出提供稳定的 hidden states。

学习这个文件要抓住三条主线：

1. **模型结构主线**：RMSNorm 是 Qwen3 Decoder Layer 中 Attention 和 MLP 前的归一化层；
2. **张量计算主线**：它沿最后一维计算 `mean(x^2)`，再用 `rsqrt` 缩放；
3. **推理优化主线**：`torch.compile` 和 `add_rms_forward` 用来减少小算子开销与中间张量。

它和前面几个文件的关系可以这样理解：

| 文件 | 负责内容 | 在模型中的位置 |
|---|---|---|
| `embed_head.py` | token id 和 logits 两端转换 | 模型输入/输出 |
| `linear.py` | Attention/MLP 的线性投影 | 模型主体计算 |
| `layernorm.py` | hidden states 归一化与 residual 融合 | 每个 Decoder Layer 内部 |

对 AI Infra 推理学习来说，这个文件提醒你：推理引擎的性能不只取决于大算子，像 RMSNorm、残差连接、dtype 转换、内存读写这类高频细节，也会影响整体 decode latency 和吞吐表现。



%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]

%% 项目关联导航：结束 %%
