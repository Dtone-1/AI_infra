# qwen3.py 源码宏观解析

## 1. 文件整体定位

`qwen3.py` 是 nano-vLLM 中负责 **定义 Qwen3 模型结构和前向推理流程** 的核心 model 文件。

前面解析过的 layer 文件，例如：

- `embed_head.py`
- `linear.py`
- `layernorm.py`
- `activation.py`
- `rotary_embedding.py`
- `attention.py`
- `sampler.py`
- `loader.py`

都更像是局部模块；而 `qwen3.py` 的作用是把这些局部模块组装成完整的 Qwen3 CausalLM。

它在 nano-vLLM 推理流程中的位置是：

```text
ModelRunner
  -> 创建 Qwen3ForCausalLM
  -> load_model 加载权重
  -> model(input_ids, positions)
  -> compute_logits(hidden_states)
  -> Sampler
  -> next token
```

在 `model_runner.py` 中：

```python
self.model = Qwen3ForCausalLM(hf_config)
load_model(self.model, config.model)
```

运行模型时：

```python
logits = self.model.compute_logits(self.model(input_ids, positions))
```

所以 `qwen3.py` 是 nano-vLLM 中“模型本体”的定义文件。

它负责：

1. 根据 `Qwen3Config` 创建 Qwen3 模型结构；
2. 定义 Attention、MLP、Decoder Layer、Model、CausalLM 的层级关系；
3. 指定 Q/K/V、Gate/Up 权重如何合并加载；
4. 描述 `input_ids -> hidden_states -> logits` 的完整模型 forward。

## 2. 这个文件要解决的核心问题

`qwen3.py` 主要解决的是 **模型结构问题**，同时也涉及 **张量计算组织、张量并行适配、权重加载映射**。

| 问题 | 说明 |
|---|---|
| 模型结构问题 | 定义 Qwen3 Attention、MLP、DecoderLayer、Model、CausalLM |
| 张量计算问题 | 串联 embedding、attention、MLP、norm、LM Head |
| 并行切分问题 | 使用并行 Linear、并行 Embedding、并行 LM Head |
| 权重加载问题 | 提供 `packed_modules_mapping`，让 loader 正确合并权重 |
| KV Cache 问题 | 间接通过 `Attention` 模块接入 KV Cache |
| 采样输出问题 | 不直接采样，只输出 logits 给 sampler |

它和原始 Transformer 的关系是：

```text
Decoder-only Transformer / Causal LM
  = Token Embedding
  + N 个 Decoder Layer
  + Final Norm
  + LM Head
```

在 Qwen3 中，一个 Decoder Layer 主要由：

```text
RMSNorm
Self-Attention
RMSNorm
MLP/SwiGLU
Residual
```

组成。

`qwen3.py` 的高性能推理目标体现在：

1. QKV 合并：`QKVParallelLinear` 把 q/k/v 投影合并；
2. MLP 合并：`MergedColumnParallelLinear` 把 gate/up 投影合并；
3. 张量并行：Attention heads、MLP intermediate、vocab embedding/head 都能按 TP rank 切分；
4. RoPE cache：通过 `get_rope` 使用旋转位置编码缓存；
5. KV Cache：通过 `Attention` 模块进入 prefill/decode 的高效注意力路径；
6. 权重映射：通过 `packed_modules_mapping` 适配 HuggingFace 原始权重。

也就是说，它不是单纯照抄 HuggingFace 模型，而是为 nano-vLLM 推理框架改写后的轻量 Qwen3 实现。

## 3. 代码结构总览

### 3.1 文件导入

```python
import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config
```

这些是基础依赖：

| 导入 | 作用 |
|---|---|
| `torch` | 张量类型 |
| `nn` | 定义模型模块 |
| `dist` | 获取 tensor parallel world size |
| `Qwen3Config` | 从 HuggingFace 配置中读取模型超参数 |

然后导入 nano-vLLM 自己实现的 layer：

```python
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
```

这些模块分别对应：

| 模块 | 作用 |
|---|---|
| `SiluAndMul` | Qwen3 MLP 的 gated activation |
| `Attention` | FlashAttention + KV Cache |
| `RMSNorm` | 归一化和 residual 融合 |
| `QKVParallelLinear` | Q/K/V 合并投影 |
| `MergedColumnParallelLinear` | Gate/Up 合并投影 |
| `RowParallelLinear` | 输出投影和 down projection |
| `get_rope` | RoPE 旋转位置编码 |
| `VocabParallelEmbedding` | 输入 token embedding |
| `ParallelLMHead` | 输出 vocab logits |

### 3.2 类结构总览

文件中主要定义了 5 个类：

| 类 | 作用 |
|---|---|
| `Qwen3Attention` | 单层 self-attention 子模块 |
| `Qwen3MLP` | 单层 MLP/SwiGLU 子模块 |
| `Qwen3DecoderLayer` | 一个完整 Decoder Layer |
| `Qwen3Model` | embedding + 多层 decoder + final norm |
| `Qwen3ForCausalLM` | Qwen3Model + LM Head，用于生成 logits |

它们的层级关系是：

```text
Qwen3ForCausalLM
  -> Qwen3Model
       -> VocabParallelEmbedding
       -> ModuleList[Qwen3DecoderLayer]
            -> Qwen3Attention
            -> Qwen3MLP
       -> Final RMSNorm
  -> ParallelLMHead
```

## 4. 各类详解

### 4.1 `Qwen3Attention`

`Qwen3Attention` 定义 Qwen3 Decoder Layer 中的 self-attention 子模块。

初始化参数：

| 参数 | 含义 |
|---|---|
| `hidden_size` | 模型隐藏维度 |
| `num_heads` | 总 query attention heads |
| `num_kv_heads` | 总 key/value heads |
| `max_position` | 最大位置长度 |
| `head_dim` | 每个 head 的维度 |
| `rms_norm_eps` | RMSNorm epsilon |
| `qkv_bias` | q/k/v projection 是否有 bias |
| `rope_theta` | RoPE 频率参数 |
| `rope_scaling` | RoPE scaling 配置 |

#### 4.1.1 TP head 切分

```python
tp_size = dist.get_world_size()
self.total_num_heads = num_heads
assert self.total_num_heads % tp_size == 0
self.num_heads = self.total_num_heads // tp_size
self.total_num_kv_heads = num_kv_heads
assert self.total_num_kv_heads % tp_size == 0
self.num_kv_heads = self.total_num_kv_heads // tp_size
```

这里把总 heads 按 tensor parallel 数切分。

例如：

```text
total_num_heads = 32
total_num_kv_heads = 8
tp_size = 2
```

则每个 rank：

```text
num_heads = 16
num_kv_heads = 4
```

这说明每个 GPU rank 只计算一部分 attention heads。

#### 4.1.2 head_dim、q_size、kv_size

```python
self.head_dim = head_dim or hidden_size // self.total_num_heads
self.q_size = self.num_heads * self.head_dim
self.kv_size = self.num_kv_heads * self.head_dim
self.scaling = self.head_dim ** -0.5
```

含义：

| 变量 | 作用 |
|---|---|
| `head_dim` | 每个 head 的特征维度 |
| `q_size` | 当前 rank 的 Q 总维度 |
| `kv_size` | 当前 rank 的 K/V 总维度 |
| `scaling` | Attention softmax 缩放因子 |

`scaling` 对应标准 Attention 里的：

```text
1 / sqrt(head_dim)
```

#### 4.1.3 QKV 投影

```python
self.qkv_proj = QKVParallelLinear(
    hidden_size,
    self.head_dim,
    self.total_num_heads,
    self.total_num_kv_heads,
    bias=qkv_bias,
)
```

这一步把原本的：

```text
q_proj
k_proj
v_proj
```

合并成一个：

```text
qkv_proj
```

输出维度是：

```text
(total_num_heads + 2 * total_num_kv_heads) * head_dim / tp_size
```

这里使用 `QKVParallelLinear`，所以它既做 QKV 合并，也做 tensor parallel 切分。

#### 4.1.4 输出投影

```python
self.o_proj = RowParallelLinear(
    self.total_num_heads * self.head_dim,
    hidden_size,
    bias=False,
)
```

Attention 输出先在每个 rank 上得到局部 heads 的结果，然后通过 RowParallelLinear 聚合回 hidden size。

#### 4.1.5 RoPE

```python
self.rotary_emb = get_rope(
    self.head_dim,
    rotary_dim=self.head_dim,
    max_position=max_position,
    base=rope_theta,
)
```

RoPE 用于给 Q/K 注入位置信息。

如果配置里有 `rope_scaling`：

```python
if isinstance(rope_scaling, dict):
    rope_theta = rope_scaling.get("rope_theta", rope_theta)
```

这个实现只读取 `rope_theta`，整体比较简化。

#### 4.1.6 Attention kernel

```python
self.attn = Attention(
    self.num_heads,
    self.head_dim,
    self.scaling,
    self.num_kv_heads,
)
```

这里接入前面解析过的 `attention.py`，里面负责：

- prefill 的 `flash_attn_varlen_func`；
- decode 的 `flash_attn_with_kvcache`；
- KV Cache 写入；
- prefix cache 和 block table。

#### 4.1.7 Q/K Norm

```python
if not self.qkv_bias:
    self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
    self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
```

如果没有 qkv bias，就对 Q/K 做 head_dim 级别的 RMSNorm。

输入 shape：

```text
q: [num_tokens, num_heads, head_dim]
k: [num_tokens, num_kv_heads, head_dim]
```

RMSNorm 沿最后一维 `head_dim` 做归一化。

### 4.2 `Qwen3Attention.forward`

代码：

```python
qkv = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
q = q.view(-1, self.num_heads, self.head_dim)
k = k.view(-1, self.num_kv_heads, self.head_dim)
v = v.view(-1, self.num_kv_heads, self.head_dim)
if not self.qkv_bias:
    q = self.q_norm(q)
    k = self.k_norm(k)
q, k = self.rotary_emb(positions, q, k)
o = self.attn(q, k, v)
output = self.o_proj(o.flatten(1, -1))
return output
```

整体数据流：

```text
hidden_states
  -> qkv_proj
  -> split q/k/v
  -> reshape to heads
  -> optional q_norm/k_norm
  -> RoPE(q/k)
  -> FlashAttention + KV Cache
  -> flatten heads
  -> o_proj
```

shape 变化：

```text
hidden_states: [num_tokens, hidden_size]
qkv:           [num_tokens, q_size + 2 * kv_size]
q:             [num_tokens, num_heads, head_dim]
k:             [num_tokens, num_kv_heads, head_dim]
v:             [num_tokens, num_kv_heads, head_dim]
o:             [num_tokens, num_heads, head_dim]
flatten(o):    [num_tokens, num_heads * head_dim]
output:        [num_tokens, hidden_size]
```

这里的 `num_heads` 和 `num_kv_heads` 都是当前 TP rank 上的局部 head 数。

### 4.3 `Qwen3MLP`

`Qwen3MLP` 定义 Qwen3 Decoder Layer 中的 FFN/MLP 子模块。

初始化：

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

Qwen3 MLP 不是普通的：

```text
Linear -> GELU -> Linear
```

而是 gated MLP：

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

nano-vLLM 中：

- `gate_proj` 和 `up_proj` 合并成 `gate_up_proj`；
- `SiluAndMul` 做 `silu(gate) * up`；
- `down_proj` 投影回 hidden size。

forward：

```python
gate_up = self.gate_up_proj(x)
x = self.act_fn(gate_up)
x = self.down_proj(x)
return x
```

shape：

```text
x:       [num_tokens, hidden_size]
gate_up: [num_tokens, 2 * intermediate_size / tp_size]
act:     [num_tokens, intermediate_size / tp_size]
output:  [num_tokens, hidden_size]
```

其中 `down_proj` 是 RowParallelLinear，会通过 all_reduce 聚合各 rank 的局部 intermediate 结果。

### 4.4 `Qwen3DecoderLayer`

`Qwen3DecoderLayer` 表示一个完整的 Qwen3 decoder block。

初始化中包含：

```python
self.self_attn = Qwen3Attention(...)
self.mlp = Qwen3MLP(...)
self.input_layernorm = RMSNorm(...)
self.post_attention_layernorm = RMSNorm(...)
```

结构可以理解为：

```text
RMSNorm
  -> Self-Attention
  -> RMSNorm
  -> MLP
```

forward：

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

这里最需要注意的是 `hidden_states` 和 `residual` 两条线。

概念上，Transformer block 常写成：

```text
x = x + Attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

nano-vLLM 这里用 `RMSNorm` 的 fused residual add 写法：

```text
hidden_states + residual
  -> new residual
  -> RMSNorm(new residual)
  -> next sublayer input
```

所以：

| 变量 | 含义 |
|---|---|
| `hidden_states` | 当前子模块的输出，准备进入下一步 |
| `residual` | 残差主线，保存未归一化的累积状态 |

第一层开始时 `residual is None`，因此：

```python
hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
```

这表示：

- `hidden_states`：embedding 输出经过 RMSNorm，送入 Attention；
- `residual`：保存原始 embedding 输出。

### 4.5 `Qwen3Model`

`Qwen3Model` 是不带 LM Head 的主体模型。

初始化：

```python
self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

它包括：

1. token embedding；
2. 多层 decoder layer；
3. final RMSNorm。

forward：

```python
hidden_states = self.embed_tokens(input_ids)
residual = None
for layer in self.layers:
    hidden_states, residual = layer(positions, hidden_states, residual)
hidden_states, _ = self.norm(hidden_states, residual)
return hidden_states
```

整体流程：

```text
input_ids
  -> VocabParallelEmbedding
  -> DecoderLayer 0
  -> DecoderLayer 1
  -> ...
  -> DecoderLayer N-1
  -> Final RMSNorm
  -> hidden_states
```

输入：

```text
input_ids: [num_tokens]
positions: [num_tokens]
```

输出：

```text
hidden_states: [num_tokens, hidden_size]
```

其中 `num_tokens` 取决于当前推理阶段：

- prefill：本轮调度的 prompt token 总数；
- decode：当前 batch 中 sequence 数。

### 4.6 `Qwen3ForCausalLM`

`Qwen3ForCausalLM` 是用于语言模型生成的最外层模型。

它包含：

```python
self.model = Qwen3Model(config)
self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
```

也就是：

```text
Qwen3ForCausalLM = Qwen3Model + LM Head
```

#### 4.6.1 `packed_modules_mapping`

```python
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
```

这是给 `loader.py` 使用的权重映射表。

含义：

| HF 原始权重 | nano-vLLM 目标参数 | shard_id |
|---|---|---|
| `q_proj` | `qkv_proj` | `"q"` |
| `k_proj` | `qkv_proj` | `"k"` |
| `v_proj` | `qkv_proj` | `"v"` |
| `gate_proj` | `gate_up_proj` | `0` |
| `up_proj` | `gate_up_proj` | `1` |

这使得 HuggingFace 原始权重可以正确加载到 nano-vLLM 合并后的模块中。

#### 4.6.2 权重共享

```python
if config.tie_word_embeddings:
    self.lm_head.weight.data = self.model.embed_tokens.weight.data
```

如果配置启用 `tie_word_embeddings`，则输入 embedding 和输出 LM Head 共享权重。

这在语言模型中很常见，可以减少参数量，并让输入/输出词表空间一致。

#### 4.6.3 forward 和 compute_logits

```python
def forward(self, input_ids, positions):
    return self.model(input_ids, positions)
```

`forward` 只返回 hidden states。

```python
def compute_logits(self, hidden_states):
    return self.lm_head(hidden_states)
```

`compute_logits` 单独把 hidden states 转成 logits。

为什么要拆开？

因为 `model_runner.py` 在 CUDA Graph decode 场景下可能先 replay 模型主体输出 hidden states，再单独调用：

```python
self.model.compute_logits(...)
```

这也让模型 forward 和 logits 计算在代码结构上更清晰。

## 5. 张量流和 forward 流程

### 5.1 完整推理数据流

从 `ModelRunner` 看，一轮模型计算大致是：

```text
prepare_prefill / prepare_decode
  -> input_ids, positions
  -> Qwen3ForCausalLM.forward
  -> Qwen3Model.forward
  -> hidden_states
  -> Qwen3ForCausalLM.compute_logits
  -> logits
  -> Sampler
  -> token_ids
```

在 `model_runner.py` 中：

```python
return self.model.compute_logits(self.model(input_ids, positions))
```

这正好对应：

```text
model(input_ids, positions) -> hidden_states
compute_logits(hidden_states) -> logits
```

### 5.2 prefill 阶段 shape

prefill 阶段输入 prompt token。

假设当前调度了多条请求，总 token 数是：

```text
num_tokens = sum(seq.num_scheduled_tokens)
```

则：

```text
input_ids: [num_tokens]
positions: [num_tokens]
```

进入 Qwen3Model：

```text
embedding output: [num_tokens, hidden_size]
```

每一层 decoder：

```text
hidden_states: [num_tokens, hidden_size]
q:             [num_tokens, num_heads, head_dim]
k/v:           [num_tokens, num_kv_heads, head_dim]
attention out: [num_tokens, hidden_size]
mlp out:       [num_tokens, hidden_size]
```

最后：

```text
final hidden_states: [num_tokens, hidden_size]
```

但是 `ParallelLMHead` 在 prefill 阶段通常只取每条 sequence 的最后一个 token 计算 logits：

```text
logits: [num_seqs, vocab_size]
```

### 5.3 decode 阶段 shape

decode 阶段每条 sequence 只输入一个 last token。

如果当前 batch 有 `num_seqs` 条请求：

```text
input_ids: [num_seqs]
positions: [num_seqs]
```

模型主体输出：

```text
hidden_states: [num_seqs, hidden_size]
```

LM Head 输出：

```text
logits: [num_seqs, vocab_size]
```

Sampler 输出：

```text
next_token_ids: [num_seqs]
```

decode 阶段 Attention 会通过 `attention.py` 使用 KV Cache：

```text
当前 token q/k/v + 历史 KV Cache -> attention output
```

### 5.4 hidden size、heads、head dim 的关系

在 Attention 里：

```text
hidden_size = total_num_heads * head_dim
```

但由于 tensor parallel，每个 rank 上是：

```text
num_heads = total_num_heads / tp_size
num_kv_heads = total_num_kv_heads / tp_size
```

所以局部 Q/K/V shape 是：

```text
q: [num_tokens, num_heads, head_dim]
k: [num_tokens, num_kv_heads, head_dim]
v: [num_tokens, num_kv_heads, head_dim]
```

`o_proj` 之后回到：

```text
[num_tokens, hidden_size]
```

### 5.5 batch 维度在哪里

nano-vLLM 的模型层通常不保留传统：

```text
[batch_size, seq_len, hidden_size]
```

而是把 batch 内 token 展平成：

```text
[num_tokens, hidden_size]
```

不同 sequence 的边界不由 shape 表示，而是由 `context.py` 中的：

- `cu_seqlens_q`
- `cu_seqlens_k`
- `block_tables`
- `context_lens`

表示。

因此 `qwen3.py` 本身只关心：

```text
input_ids [num_tokens]
positions [num_tokens]
hidden_states [num_tokens, hidden_size]
```

具体哪些 token 属于哪条请求，由 `model_runner.py` 和 `attention.py` 的 context 信息处理。

## 6. 和 Transformer / Qwen 模型结构的关系

### 6.1 Decoder-only Causal LM

Qwen3 是 decoder-only causal language model。

整体结构是：

```text
Token IDs
  -> Token Embedding
  -> Decoder Layers
  -> Final Norm
  -> LM Head
  -> Logits
```

`qwen3.py` 对应的就是这条主线。

### 6.2 Attention 子结构

Qwen3Attention 对应 Transformer 中的 self-attention：

```text
Q = xWq
K = xWk
V = xWv
Attention = softmax(QK^T / sqrt(d))V
Output = Attention Wo
```

nano-vLLM 实现中变成：

```text
QKVParallelLinear
  -> split q/k/v
  -> q/k norm
  -> RoPE
  -> FlashAttention + KV Cache
  -> RowParallelLinear(o_proj)
```

### 6.3 MLP 子结构

Qwen3MLP 对应 Transformer 的 FFN：

```text
MLP(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
```

nano-vLLM 实现中变成：

```text
MergedColumnParallelLinear(gate_up_proj)
  -> SiluAndMul
  -> RowParallelLinear(down_proj)
```

### 6.4 RMSNorm 和 residual

Qwen3DecoderLayer 使用 RMSNorm 和 residual：

```text
RMSNorm -> Attention
RMSNorm -> MLP
```

代码中用 `hidden_states, residual` 两条变量线实现 residual add + norm 的融合。

这和前面 `layernorm.py` 的 `add_rms_forward` 直接对应。

### 6.5 RoPE 位置编码

Qwen3Attention 不在 embedding 上直接加 position embedding，而是在 q/k 上应用 RoPE：

```python
q, k = self.rotary_emb(positions, q, k)
```

`positions` 来自 `model_runner.py`，prefill 和 decode 都必须传正确位置。

### 6.6 GQA / KV heads

`num_heads` 和 `num_kv_heads` 分开，说明 Qwen3 支持 GQA：

```text
num_attention_heads >= num_key_value_heads
```

这样可以减少 KV Cache 占用，因为 cache 保存的是 K/V heads，而不是 Q heads。

## 7. 和 nano-vLLM 推理系统的关系

### 7.1 和 `model_runner.py`

`model_runner.py` 负责准备：

- `input_ids`
- `positions`
- context 信息
- KV Cache
- CUDA Graph
- sampler temperature

`qwen3.py` 负责把 `input_ids/positions` 算成 hidden states 和 logits。

两者关系是：

```text
ModelRunner 管调度和运行环境
Qwen3ForCausalLM 管模型数学计算
```

### 7.2 和 `loader.py`

`Qwen3ForCausalLM` 提供：

```python
packed_modules_mapping
```

`loader.py` 使用它把 HuggingFace 权重映射到 nano-vLLM 的合并模块中。

也就是说：

```text
qwen3.py 定义模型结构和映射规则
loader.py 执行权重加载
```

### 7.3 和 `attention.py`

`qwen3.py` 创建 Attention 模块：

```python
self.attn = Attention(...)
```

但具体 prefill/decode、KV Cache、FlashAttention 逻辑都在 `attention.py`。

所以：

```text
qwen3.py 决定 Attention 在模型里怎么接
attention.py 决定 Attention 推理时怎么高效算
```

### 7.4 和 tensor parallel

`qwen3.py` 通过以下模块接入 tensor parallel：

| 模块 | 并行方式 |
|---|---|
| `VocabParallelEmbedding` | vocab 维度切分 |
| `QKVParallelLinear` | Q/K/V 输出维度按 head 切分 |
| `MergedColumnParallelLinear` | MLP intermediate 维度切分 |
| `RowParallelLinear` | 输入维度切分后 all_reduce |
| `ParallelLMHead` | vocab logits 分片后 gather |

因此 `qwen3.py` 是模型并行结构的总装文件。

## 8. 学习总结

`qwen3.py` 的核心价值可以概括为一句话：

> 它把 nano-vLLM 中所有 layer 组件组装成完整的 Qwen3 CausalLM，并适配 tensor parallel、KV Cache、RoPE、SwiGLU、权重合并加载和 logits 输出。

学习这个文件要抓住五条主线：

1. **模型层级主线**：`Qwen3ForCausalLM -> Qwen3Model -> Qwen3DecoderLayer -> Attention/MLP`；
2. **forward 数据流主线**：`input_ids -> embedding -> decoder layers -> norm -> hidden_states -> logits`；
3. **Attention 主线**：`QKVParallelLinear -> q/k/v -> RoPE -> Attention -> o_proj`；
4. **MLP 主线**：`gate_up_proj -> SiluAndMul -> down_proj`；
5. **推理系统主线**：`packed_modules_mapping`、tensor parallel、KV Cache、prefill/decode 都通过这个模型结构被接入。

它和其他文件的关系可以这样理解：

| 文件 | 在 `qwen3.py` 中的角色 |
|---|---|
| `embed_head.py` | 输入 embedding 和输出 LM Head |
| `linear.py` | Attention/MLP 中的并行线性层 |
| `layernorm.py` | Decoder Layer 中的 RMSNorm 和 residual 融合 |
| `activation.py` | MLP 中的 `silu(gate) * up` |
| `rotary_embedding.py` | Attention 中 q/k 的 RoPE 位置编码 |
| `attention.py` | FlashAttention + KV Cache 的核心 attention |
| `loader.py` | 使用 `packed_modules_mapping` 加载合并权重 |
| `model_runner.py` | 调用 Qwen3ForCausalLM 执行推理 |

对 AI Infra 推理学习来说，`qwen3.py` 是你理解 nano-vLLM 模型侧源码的收束点。读懂它之后，前面那些分散的 layer 文件就不再是孤立模块，而是能串成一条完整的推理链路：

```text
token id
  -> embedding
  -> RMSNorm
  -> QKV projection
  -> RoPE
  -> FlashAttention + KV Cache
  -> MLP/SwiGLU
  -> final norm
  -> LM Head
  -> logits
```



%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]
- 关联阅读：[[outputs/项目整理/专题-03-Qwen模型适配与MTP|专题-03-Qwen模型适配与MTP]]

%% 项目关联导航：结束 %%
