# rotary_embedding.py 源码宏观解析

## 1. 文件整体定位

`rotary_embedding.py` 是 nano-vLLM 中负责 **RoPE 旋转位置编码** 的 layer 文件。

RoPE 全称是 Rotary Position Embedding，中文一般叫“旋转位置编码”。它的作用是：把 token 的位置信息注入到 Attention 的 Query 和 Key 中，让模型在计算注意力分数时知道 token 的相对/绝对位置关系。

在 `models/qwen3.py` 中，它被 Qwen3Attention 使用：

```python
self.rotary_emb = get_rope(
    self.head_dim,
    rotary_dim=self.head_dim,
    max_position=max_position,
    base=rope_theta,
)
```

forward 中调用：

```python
q, k = self.rotary_emb(positions, q, k)
```

所以它位于 Attention 子模块中，具体位置是：

```text
hidden_states
  -> qkv_proj
  -> split q/k/v
  -> reshape q/k/v
  -> RMSNorm(q/k, optional)
  -> RoPE(q/k)
  -> Attention
```

注意：RoPE 只作用在 `q` 和 `k` 上，不作用在 `v` 上。因为 Attention 分数由 `q @ k^T` 决定，位置信息需要进入 Query/Key 的匹配过程；Value 主要承载被聚合的内容。

## 2. 这个文件要解决的核心问题

`rotary_embedding.py` 主要解决的是 **模型结构问题** 和 **张量计算问题**。

| 问题 | 说明 |
|---|---|
| 模型结构问题 | 实现 Qwen3 Attention 中使用的 RoPE 位置编码 |
| 张量计算问题 | 根据 `positions` 取出 cos/sin，并对 q/k 做旋转变换 |
| 权重加载问题 | 不涉及，RoPE 没有可训练权重 |
| 并行切分问题 | 不直接通信，但作用于每个 TP rank 本地的 attention heads |
| 采样输出问题 | 不涉及 |

它和原始 Transformer 位置编码的关系是：

```text
原始 Transformer: token embedding + absolute positional encoding
Qwen/LLaMA 类模型: q/k 上应用 RoPE
```

原始 Transformer 常见做法是把位置编码直接加到 embedding 上：

```text
hidden_states = token_embedding + position_embedding
```

而 RoPE 不是简单相加，而是在 Attention 里对 Q/K 向量做旋转：

```text
q_positioned = rotate(q, position)
k_positioned = rotate(k, position)
```

它和高性能推理的关系在于：

1. RoPE 每层 Attention 都要执行；
2. prefill 和 decode 都需要根据 token 的 `positions` 应用 RoPE；
3. decode 阶段每次只处理新 token，但仍然要用正确 position；
4. 代码提前缓存所有位置的 cos/sin，避免每次 forward 重算三角函数；
5. 使用 `@torch.compile` 尝试减少高频小算子开销。

## 3. 代码结构总览

### 3.1 文件导入

```python
from functools import lru_cache
import torch
from torch import nn
```

含义如下：

| 导入 | 作用 |
|---|---|
| `lru_cache` | 缓存 `get_rope` 返回的 RoPE 模块 |
| `torch` | 张量计算、三角函数、einsum |
| `nn` | 定义 PyTorch Module |

这个文件没有 `torch.distributed`，说明 RoPE 本身不做跨 GPU 通信。每个 tensor parallel rank 对自己本地的 q/k heads 做 RoPE 即可。

### 3.2 `apply_rotary_emb`

```python
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
```

这是实际执行旋转变换的函数。

核心代码：

```python
x1, x2 = torch.chunk(x.float(), 2, dim=-1)
y1 = x1 * cos - x2 * sin
y2 = x2 * cos + x1 * sin
return torch.cat((y1, y2), dim=-1).to(x.dtype)
```

它做了几件事：

1. 把输入 `x` 转成 float32；
2. 沿最后一维把 head 向量切成两半；
3. 用 cos/sin 做二维旋转；
4. 把旋转后的两半拼回去；
5. 转回原始 dtype。

如果把最后一维中的两个分量看成二维平面坐标：

```text
[x1, x2]
```

旋转后的结果就是：

```text
y1 = x1 * cos - x2 * sin
y2 = x2 * cos + x1 * sin
```

这就是 RoPE 里的“旋转”。

### 3.3 `RotaryEmbedding`

```python
class RotaryEmbedding(nn.Module):
```

这是 RoPE 模块本体，负责初始化 cos/sin 缓存，并在 forward 时根据 positions 取出对应位置的 cos/sin。

初始化参数：

| 参数 | 含义 |
|---|---|
| `head_size` | 每个 attention head 的维度 |
| `rotary_dim` | 应用 RoPE 的维度 |
| `max_position_embeddings` | 最大支持的位置长度 |
| `base` | RoPE 频率基数，也就是常见的 `rope_theta` |

代码里有一个限制：

```python
assert rotary_dim == head_size
```

说明 nano-vLLM 这个实现要求整个 `head_dim` 都应用 RoPE，不支持只对部分 head_dim 应用 RoPE。

### 3.4 cos/sin 缓存构造

初始化中的关键代码：

```python
inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
```

`inv_freq` 表示不同维度对应的旋转频率。

然后：

```python
t = torch.arange(max_position_embeddings, dtype=torch.float)
freqs = torch.einsum("i,j -> ij", t, inv_freq)
```

这里构造出：

```text
freqs: [max_position_embeddings, rotary_dim / 2]
```

含义是：每个位置、每个频率维度对应一个旋转角度。

接着：

```python
cos = freqs.cos()
sin = freqs.sin()
cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
```

得到缓存：

```text
cos_sin_cache: [max_position_embeddings, 1, rotary_dim]
```

中间的 `1` 是为了后续和 attention heads 维度广播。

最后：

```python
self.register_buffer("cos_sin_cache", cache, persistent=False)
```

这表示 `cos_sin_cache` 是模块的一部分，会跟随模型移动到 GPU，但不是可训练参数。

`persistent=False` 表示它不需要作为模型权重持久保存，因为它可以根据配置重新计算。

### 3.5 `RotaryEmbedding.forward`

```python
@torch.compile
def forward(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
```

输入：

| 输入 | shape | 含义 |
|---|---|---|
| `positions` | `[num_tokens]` | 每个 token 的位置编号 |
| `query` | `[num_tokens, num_heads, head_dim]` | 当前 rank 的 Q |
| `key` | `[num_tokens, num_kv_heads, head_dim]` | 当前 rank 的 K |

核心流程：

```python
cos_sin = self.cos_sin_cache[positions]
cos, sin = cos_sin.chunk(2, dim=-1)
query = apply_rotary_emb(query, cos, sin)
key = apply_rotary_emb(key, cos, sin)
return query, key
```

`positions` 用来从缓存中取出当前 token 对应的 cos/sin：

```text
cos_sin: [num_tokens, 1, head_dim]
cos:     [num_tokens, 1, head_dim / 2]
sin:     [num_tokens, 1, head_dim / 2]
```

然后分别对 query/key 做旋转。

中间维度为 `1`，所以它可以广播到所有 heads：

```text
query: [num_tokens, num_heads, head_dim]
cos:   [num_tokens, 1, head_dim / 2]
```

这表示同一个 token position 的 cos/sin 会作用到该 token 的所有 attention heads。

### 3.6 `get_rope`

```python
@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
```

这个函数用于创建并缓存 RoPE 模块。

`@lru_cache(1)` 表示最多缓存一个调用结果。对于当前 nano-vLLM 的简单模型加载场景，通常一个模型只需要一套 RoPE 配置，所以缓存一个就够。

这样可以避免重复创建 `RotaryEmbedding` 和重复构造 cos/sin cache。

## 4. 张量流和 forward 流程

### 4.1 `positions` 从哪里来

在 `model_runner.py` 中，prefill 和 decode 都会构造 `positions`。

prefill 阶段：

```python
positions.extend(range(start, end))
```

也就是 prompt 中每个 token 的位置。

decode 阶段：

```python
positions.append(len(seq) - 1)
```

也就是当前新 token 在整条 sequence 中的位置。

因此 RoPE 不自己决定位置，它只消费上游准备好的 `positions`。

### 4.2 在 Attention 中的输入 shape

在 `Qwen3Attention.forward` 中：

```python
qkv = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
q = q.view(-1, self.num_heads, self.head_dim)
k = k.view(-1, self.num_kv_heads, self.head_dim)
v = v.view(-1, self.num_kv_heads, self.head_dim)
```

此时：

```text
q: [num_tokens, num_heads, head_dim]
k: [num_tokens, num_kv_heads, head_dim]
v: [num_tokens, num_kv_heads, head_dim]
```

然后：

```python
q, k = self.rotary_emb(positions, q, k)
```

RoPE 输出 shape 不变：

```text
q: [num_tokens, num_heads, head_dim]
k: [num_tokens, num_kv_heads, head_dim]
```

只是 q/k 的数值被位置相关的 cos/sin 旋转过。

### 4.3 `apply_rotary_emb` 内部 shape

以 query 为例：

```text
query: [num_tokens, num_heads, head_dim]
```

切成两半：

```python
x1, x2 = torch.chunk(query.float(), 2, dim=-1)
```

得到：

```text
x1: [num_tokens, num_heads, head_dim / 2]
x2: [num_tokens, num_heads, head_dim / 2]
```

cos/sin：

```text
cos: [num_tokens, 1, head_dim / 2]
sin: [num_tokens, 1, head_dim / 2]
```

广播后做旋转：

```text
y1 = x1 * cos - x2 * sin
y2 = x2 * cos + x1 * sin
```

拼接：

```text
output: [num_tokens, num_heads, head_dim]
```

key 的过程完全一样，只是第二维是 `num_kv_heads`。

### 4.4 prefill 和 decode 中的差异

RoPE 本身的计算逻辑在 prefill 和 decode 中没有区别，区别只在 `positions` 的内容。

prefill 阶段：

```text
positions = [0, 1, 2, ..., prompt_len - 1]
```

如果一个 batch 内有多条 sequence，positions 会按 token 展平后的顺序组织，每条 sequence 内部仍然是自己的位置编号。

decode 阶段：

```text
positions = [seq1_len - 1, seq2_len - 1, ...]
```

每条 sequence 当前只处理一个 token，但这个 token 的 position 必须等于它在完整上下文中的位置。

这点非常重要：decode 不是每次都从位置 0 重新开始，而是随着 sequence 长度递增。

### 4.5 和 KV Cache 的关系

RoPE 发生在 K 写入 KV Cache 之前。

在 Attention 流程中：

```text
hidden_states
  -> qkv_proj
  -> q/k/v
  -> RoPE(q/k)
  -> store k/v into KV Cache
  -> attention computation
```

因此 KV Cache 中保存的 K 已经带有对应 position 的 RoPE 信息。

decode 阶段，新 token 的 q/k 会用当前 position 做 RoPE，然后和历史 cache 中已经旋转好的 K 一起计算 attention。

## 5. 和 Transformer / Qwen 模型结构的关系

### 5.1 为什么需要位置编码

Attention 本身看的是 token 之间的相似度：

```text
attention_score = q @ k^T
```

如果没有位置信息，模型只知道 token 内容，不知道 token 的顺序。

例如：

```text
我 喜欢 你
你 喜欢 我
```

token 集合相似，但语义不一样。位置编码的目的就是让模型知道顺序。

### 5.2 RoPE 和传统位置编码的区别

传统绝对位置编码通常加在 embedding 上：

```text
hidden_states = token_embedding + position_embedding
```

RoPE 是在 Attention 的 q/k 上做旋转：

```text
q' = rotate(q, position)
k' = rotate(k, position)
```

它的一个重要直觉是：旋转后的 q/k 点积能够体现相对位置关系。

因此 RoPE 非常适合自回归大语言模型，也常见于 LLaMA、Qwen 等模型。

### 5.3 和 Qwen3Attention 的关系

Qwen3Attention 的核心顺序是：

```text
hidden_states
  -> QKVParallelLinear
  -> q/k/v
  -> optional q_norm/k_norm
  -> RoPE
  -> Attention
  -> o_proj
```

`rotary_embedding.py` 就负责其中的：

```text
q/k + positions -> position-aware q/k
```

没有这一步，Attention 仍然可以算，但模型会丢失训练时依赖的位置编码机制，生成质量会严重受影响。

### 5.4 和长上下文的关系

初始化时有：

```python
max_position_embeddings
base
```

它们决定了 RoPE cache 支持的位置范围和频率分布。

在 Qwen3Attention 中：

```python
max_position=config.max_position_embeddings
rope_theta=getattr(config, "rope_theta", 1000000)
```

如果配置里有 `rope_scaling`：

```python
if isinstance(rope_scaling, dict):
    rope_theta = rope_scaling.get("rope_theta", rope_theta)
```

这说明 nano-vLLM 会从模型配置中读取 RoPE 相关参数。长上下文模型通常会通过不同的 `rope_theta` 或 RoPE scaling 策略支持更长 context。

不过这个文件里的实现比较简化：它只读取 `rope_theta`，并且要求 `rotary_dim == head_size`。

### 5.5 和 tensor parallel 的关系

在 tensor parallel 下，Q/K/V 的 heads 会分布到不同 rank。

例如：

```text
rank 0: 一部分 q heads / kv heads
rank 1: 另一部分 q heads / kv heads
```

RoPE 对每个 rank 本地的 q/k 做同样的位置旋转，不需要跨 rank 通信。

所以它和 `linear.py` 的区别是：

| 文件 | 是否切分/通信 | 作用 |
|---|---|---|
| `linear.py` | 负责张量并行切分和 all_reduce | 大矩阵线性层 |
| `rotary_embedding.py` | 不通信，只处理本地 q/k | 注入位置信息 |

## 6. 学习总结

`rotary_embedding.py` 的核心价值可以概括为一句话：

> 它根据 token 的 `positions` 为 Attention 中的 Query 和 Key 应用 RoPE 旋转位置编码，让模型在 q/k 匹配时感知 token 顺序和位置关系。

学习这个文件要抓住三条主线：

1. **模型结构主线**：RoPE 属于 Attention 内部，作用在 q/k 上，不作用在 v 上；
2. **张量 shape 主线**：`positions [num_tokens]` 取出 cos/sin，再广播到 `[num_tokens, num_heads, head_dim]`；
3. **推理优化主线**：提前缓存 cos/sin，使用 `lru_cache` 和 `torch.compile` 降低重复计算和小算子开销。

它和前面几个文件的关系可以这样理解：

| 文件 | 负责内容 | 在模型中的位置 |
|---|---|---|
| `linear.py` | 生成 q/k/v，执行 o_proj 等线性层 | Attention/MLP 主体 |
| `layernorm.py` | 对 hidden_states 或 q/k 做 RMSNorm | Attention/MLP 前 |
| `rotary_embedding.py` | 对 q/k 注入位置信息 | Attention 计算前 |
| `attention.py` | 使用 q/k/v 和 KV Cache 计算注意力输出 | Attention 核心 |

对 AI Infra 推理学习来说，这个文件很适合帮助你建立一个关键认识：推理时不仅要喂 token id，还必须为每个 token 维护正确的 position；否则即使 KV Cache、Attention、Linear 都正常，模型也会因为位置信息错误而生成异常结果。



%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]

%% 项目关联导航：结束 %%
