# qwen3.6_qwen3_5.py_分析

> 分析对象：`nano-vllm-qwen3.6` 新增源码文件 `qwen3_5.py`  
>
> 本文目标：从整体工程角度分析该文件为什么需要新增、它相对原版 nano-vLLM 的 `qwen3.py` 解决了什么问题，以及它在 Qwen3.5 / Qwen3.6 / hybrid 架构 / 多模态推理系统中的作用。

---

## 1. 文件整体定位

`qwen3_5.py` 是 `nano-vllm-qwen3.6` 中新增的模型结构文件。

原版 nano-vLLM 主要支持的是普通 Qwen3 dense decoder-only 模型，也就是：

```text
Embedding
  ↓
DecoderLayer × N
  ↓
Final RMSNorm
  ↓
LM Head
```

每个 DecoderLayer 基本是：

```text
RMSNorm
  ↓
Self-Attention
  ↓
RMSNorm
  ↓
MLP
```

而 `qwen3_5.py` 的目标明显更复杂。它新增了一套 Qwen3.5 / Qwen3.6 风格的模型结构，核心特点包括：

```text
1. full attention 层和 GatedDeltaNet 层混合
2. full attention 使用 gated query / output gating
3. 使用 GemmaRMSNorm，而不是普通 RMSNorm
4. 使用 InterleavedMRoPE，支持文本和多模态 3D 位置编码
5. 支持可选视觉编码器，把 image embeddings 填入 image token 位置
6. 支持可选 MTP 模块
7. 适配新的权重命名前缀和加载映射
```

所以，这个文件不是简单把 `qwen3.py` 改个名字，而是新增了一套用于 hybrid / multimodal / Qwen3.5 类模型的模型主体实现。

---

## 2. 为什么需要新增这个文件

原版 nano-vLLM 的 `qwen3.py` 更适合普通 dense Transformer：

```text
所有 layer 都是 attention + MLP
所有 attention 层都使用 KV Cache
所有 token 都走同一类 decoder block
位置编码是标准 RoPE
Norm 是普通 RMSNorm
输入主要是文本 token
```

但是 Qwen3.5 / Qwen3.6 类模型可能不再是纯 attention-only 结构，而是 hybrid 架构：

```text
一部分层是 full_attention
一部分层是 GatedDeltaNet / linear attention / recurrent 类层
```

这会带来几个原版 `qwen3.py` 无法自然处理的问题：

```text
1. 每一层不一定都是 Attention
2. 有些层不需要 KV Cache，但需要 recurrent state / conv state
3. 模型层类型需要根据 config.layer_types 动态选择
4. full attention 里 Q projection 可能带 gate
5. 位置编码可能是 partial MRoPE，而不是标准 RoPE
6. 多模态输入需要视觉 encoder 和 image token 替换
7. norm 参数定义可能不是普通 RMSNorm
```

因此新增 `qwen3_5.py` 的意义是：

```text
把 Qwen3.5 / Qwen3.6 这类 hybrid、多模态、带门控结构的模型从原版 Qwen3 dense 实现中拆出来，形成单独模型实现。
```

---

## 3. 文件中的主要类

该文件主要定义了五个类：

| 类名 | 作用 |
|---|---|
| `Qwen3_5Attention` | full attention 层，带 query gate、GemmaRMSNorm、InterleavedMRoPE 和 output gating |
| `Qwen3_5MLP` | SwiGLU MLP，使用 `gate_up_proj + down_proj` |
| `Qwen3_5DecoderLayer` | 单个 decoder layer，根据 `layer_types` 选择 full attention 或 GatedDeltaNet |
| `Qwen3_5Model` | 模型主干：Embedding + DecoderLayer × N + Final GemmaRMSNorm |
| `Qwen3_5ForCausalLM` | 最外层 causal LM：主干模型 + LM Head + 可选视觉 encoder + 可选 MTP |

整体结构可以概括为：

```text
Qwen3_5ForCausalLM
  ├── Qwen3_5Model
  │     ├── VocabParallelEmbedding
  │     ├── Qwen3_5DecoderLayer × N
  │     │     ├── Qwen3_5Attention 或 GatedDeltaNet
  │     │     ├── Qwen3_5MLP
  │     │     ├── GemmaRMSNorm
  │     │     └── GemmaRMSNorm
  │     └── Final GemmaRMSNorm
  ├── ParallelLMHead
  ├── optional Qwen3MTP
  └── optional Qwen3VLVisionEncoder
```

---

## 4. 新增模块一：`Qwen3_5Attention`

`Qwen3_5Attention` 是 full attention 层的实现。

它和原版 Qwen3 attention 的差异非常大。

### 4.1 原版 Qwen3 attention 的典型结构

原版 Qwen3 dense 通常是：

```text
hidden_states
  ↓
qkv_proj
  ↓
split q/k/v
  ↓
q_norm / k_norm
  ↓
RoPE
  ↓
Attention
  ↓
o_proj
```

其中 Q/K/V 通常通过一个合并后的 `QKVParallelLinear` 统一计算。

### 4.2 Qwen3_5Attention 的结构

`Qwen3_5Attention` 使用的是：

```text
q_proj
k_proj
v_proj
o_proj
```

而不是合并的 qkv_proj。

其中最特殊的是：

```python
self.q_proj = ColumnParallelLinear(
    config.hidden_size,
    self.total_num_heads * self.head_dim * 2,
    bias=False,
)
```

也就是说，q_proj 的输出不是普通的 Q，而是：

```text
Q + gate
```

forward 中：

```python
q_gate = self.q_proj(hidden_states)
q_gate = q_gate.view(-1, self.num_heads, self.head_dim * 2)
q, gate = q_gate.chunk(2, dim=-1)
```

这表示每个 head 的 q 向量旁边还有一个 gate 向量。

---

## 5. Qwen3_5Attention 的 output gating

full attention 输出后，代码有：

```python
o = o * torch.sigmoid(gate)
```

这一步是 output gating。

普通 attention 是：

```text
o = Attention(q, k, v)
output = o_proj(o)
```

这里变成：

```text
q, gate = q_proj(hidden_states).split()
o = Attention(q, k, v)
o = o * sigmoid(gate)
output = o_proj(o)
```

它的意义是：

```text
Attention 计算出来的信息不是直接全部通过，而是被 gate 控制每个 head_dim 通道的强弱。
```

可以理解为：

```text
Attention 负责从上下文取信息；
gate 负责决定这些信息通过多少。
```

这是一种更强的动态控制机制。

---

## 6. Qwen3_5Attention 使用 GemmaRMSNorm

该文件中 full attention 内部使用：

```python
self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
```

这和普通 Qwen3 使用的 RMSNorm 不完全一样。

GemmaRMSNorm 的特点是：

```text
output = norm(x) * (1 + weight)
```

而普通 RMSNorm 是：

```text
output = norm(x) * weight
```

所以这个文件依赖了前面新增的 `GemmaRMSNorm`。

这说明 `qwen3_5.py` 的模型结构不是简单沿用原版 Qwen3 dense 的 norm 形式，而是适配了新的 norm 参数定义。

---

## 7. Qwen3_5Attention 使用 InterleavedMRoPE

该文件中：

```python
self.rotary_emb = InterleavedMRoPE(...)
```

而不是标准 `RotaryEmbedding`。

它从 config 中读取：

```text
rope_theta
partial_rotary_factor
mrope_section
```

这说明它支持：

```text
partial RoPE
interleaved multi-dimensional RoPE
文本 1D positions
多模态 3D positions
```

在 forward 中：

```python
q, k = self.rotary_emb(positions, q, k)
```

如果 positions 是普通 `[N]`，它可以退化成文本 RoPE；如果 positions 是 `[3, N]`，它会走多模态 MRoPE。

这和前面 `model_runner.py` 中为图像 token 计算 3D positions 的逻辑是配套的。

---

## 8. 新增模块二：`Qwen3_5MLP`

`Qwen3_5MLP` 仍然是典型 SwiGLU MLP：

```python
self.gate_up_proj = MergedColumnParallelLinear(
    config.hidden_size,
    [config.intermediate_size] * 2,
    bias=False,
)
self.down_proj = RowParallelLinear(
    config.intermediate_size,
    config.hidden_size,
    bias=False,
)
self.act_fn = SiluAndMul()
```

forward 非常简洁：

```python
return self.down_proj(self.act_fn(self.gate_up_proj(x)))
```

它对应公式：

```text
MLP(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
```

这一部分和原版 Qwen3 dense 的 MLP 思路接近，仍然使用：

```text
gate_proj + up_proj 合并成 gate_up_proj
```

不过在权重加载映射中，`Qwen3_5ForCausalLM` 只定义了：

```python
packed_modules_mapping = {
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
```

没有 q/k/v 到 qkv_proj 的映射。

原因是本文件中的 attention 不再使用合并的 qkv_proj，而是显式定义了：

```text
q_proj
k_proj
v_proj
```

所以 gate/up 仍然合并，q/k/v 不合并。

---

## 9. 新增模块三：`Qwen3_5DecoderLayer`

`Qwen3_5DecoderLayer` 是该文件体现 hybrid 架构的核心。

初始化时：

```python
layer_type = config.layer_types[layer_idx]

if layer_type == "full_attention":
    self.self_attn = Qwen3_5Attention(config, layer_idx)
    self.linear_attn = None
else:
    self.linear_attn = GatedDeltaNet(config, layer_idx)
    self.self_attn = None
```

这说明每一层的类型不是固定的，而是由：

```text
config.layer_types[layer_idx]
```

决定。

如果该层是：

```text
full_attention
```

就使用 full attention。

否则使用：

```text
GatedDeltaNet
```

这就是 hybrid 架构的直接体现。

---

## 10. full attention 层和 GatedDeltaNet 层的区别

在这个文件中，每个 decoder layer 的主干结构仍然是：

```text
input norm
  ↓
attention-like 子层
  ↓
post norm
  ↓
MLP
```

但 attention-like 子层可能有两种：

| layer_type | 使用模块 | 历史状态 |
|---|---|---|
| `full_attention` | `Qwen3_5Attention` | KV Cache |
| 其他 | `GatedDeltaNet` | recurrent state / conv state |

这和原版 Qwen3 dense 最大的不同是：

```text
原版每一层都是 full attention；
qwen3_5.py 中每一层可以是 full attention，也可以是 GatedDeltaNet。
```

所以推理系统必须配合修改：

```text
KV Cache 不能再按总层数分配
GDN state 需要单独分配
Scheduler 需要维护 state_slot_id
Context 需要传 state_indices
CUDA Graph 需要包含 state_indices
```

这就是为什么前面多个文件都围绕 `is_hybrid`、`state_slot_id`、`state_indices` 做了改造。

---

## 11. DecoderLayer 中的 residual / norm 流程

`Qwen3_5DecoderLayer.forward()` 保留了 nano-vLLM 原有的 fused add RMSNorm 风格：

```python
if residual is None:
    hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
else:
    hidden_states, residual = self.input_layernorm(hidden_states, residual)
```

然后：

```python
if self.self_attn is not None:
    hidden_states = self.self_attn(positions, hidden_states)
else:
    hidden_states = self.linear_attn(hidden_states)

hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
hidden_states = self.mlp(hidden_states)
```

也就是说，不论子层是 full attention 还是 GatedDeltaNet，外层 block 的 residual/norm/MLP 框架保持一致。

这是一种很好的工程抽象：

```text
DecoderLayer 外壳统一；
中间的 token-mixing 模块根据 layer_type 替换。
```

---

## 12. 新增模块四：`Qwen3_5Model`

`Qwen3_5Model` 是模型主干。

它包含：

```python
self.embed_tokens = VocabParallelEmbedding(...)
self.layers = nn.ModuleList([...])
self.norm = GemmaRMSNorm(...)
```

forward 流程是：

```text
input_ids
  ↓
Embedding
  ↓
如果有 image_embeds，把视觉 embedding scatter 到 image token 位置
  ↓
DecoderLayer × N
  ↓
Final GemmaRMSNorm
  ↓
hidden_states
```

最关键新增点是：

```python
if image_embeds is not None and image_token_mask is not None:
    hidden_states[image_token_mask] = image_embeds.to(hidden_states.dtype)
```

这说明该模型支持多模态输入：

```text
文本 token 先走 embedding
图像 token 位置被视觉 encoder 输出替换
```

也就是说，图像不是直接作为 token id 查 embedding，而是先经过视觉编码器得到 image_embeds，再填入对应的 image token 位置。

---

## 13. image token 替换机制

多模态输入大致可以理解为：

```text
input_ids:
    [文本 token, image_token, image_token, ..., 文本 token]

embed_tokens(input_ids):
    得到每个 token 的初始 hidden_states

visual(pixel_values, image_grid_thw):
    得到 image_embeds

hidden_states[image_token_mask] = image_embeds
```

这样模型后续 decoder layer 看到的是同一个 hidden_states 序列，其中：

```text
文本位置是文本 embedding
图像位置是视觉 encoder 输出
```

这是一种常见的 VLM 融合方式。

---

## 14. 新增模块五：`Qwen3_5ForCausalLM`

`Qwen3_5ForCausalLM` 是最外层 causal language model。

它包含：

```python
self.model = Qwen3_5Model(config)
self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
```

它的职责是：

```text
模型主干 forward 得到 hidden_states
LM Head 把 hidden_states 转成 logits
```

这和原版 `Qwen3ForCausalLM` 的外壳思路一致。

但 qwen3_5 版本额外支持：

```text
1. 权重前缀
2. 视觉模型
3. MTP
4. tie_word_embeddings
```

---

## 15. 权重前缀：`weight_prefix` 和 `visual_prefix`

文件中定义：

```python
weight_prefix = "model.language_model."
visual_prefix = "model.visual."
```

这说明 checkpoint 中的权重命名可能不是原版 Qwen3 那种简单结构，而是把语言模型和视觉模型分开：

```text
model.language_model.xxx
model.visual.xxx
```

因此 loader 需要知道：

```text
语言模型权重应该映射到 Qwen3_5Model
视觉模型权重应该映射到 visual encoder
```

这也是多模态模型常见的权重组织方式。

---

## 16. packed_modules_mapping 的变化

该文件定义：

```python
packed_modules_mapping = {
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
```

这里只打包 MLP 的 gate/up。

原版 Qwen3 dense 往往还有：

```text
q_proj/k_proj/v_proj -> qkv_proj
```

但这个文件没有 qkv 合并映射。

原因是：

```text
Qwen3_5Attention 中 q_proj、k_proj、v_proj 是单独的 ColumnParallelLinear；
q_proj 还输出 q + gate，结构已经不同于原版 qkv_proj。
```

所以 q/k/v 不再适合使用原版的 QKVParallelLinear 打包逻辑。

---

## 17. tie_word_embeddings 支持

文件中：

```python
if getattr(config, 'tie_word_embeddings', False):
    self.lm_head.weight.data = self.model.embed_tokens.weight.data
```

这表示如果配置要求共享词嵌入和 LM Head 权重，则：

```text
embedding weight 和 lm_head weight 指向同一份数据
```

这种设计可以减少参数量，也符合部分模型的 checkpoint 结构。

---

## 18. 可选 MTP 支持

文件中：

```python
self.mtp = None
if getattr(config, "enable_mtp", False):
    from nanovllm.models.qwen3_mtp import Qwen3MTP
    self.mtp = Qwen3MTP(config)
```

这说明 qwen3_5 模型可以挂载 MTP 模块。

MTP 通常可以理解为 multi-token prediction / draft token 相关能力，用于投机解码或一次预测多个未来 token 的辅助路径。

该文件只是把 MTP 模块挂到模型上，真正的 MTP 运行逻辑在 `model_runner.py` 中，例如：

```text
run_mtp_probe
run_mtp_draft_step
run_mtp_draft_fast_step
```

所以这里的意义是：

```text
模型结构层暴露 MTP 子模块；
执行层负责调用它。
```

---

## 19. 可选视觉编码器支持

文件中：

```python
self.visual = None
if vision_config is not None:
    from nanovllm.models.vision_encoder import Qwen3VLVisionEncoder
    self.visual = Qwen3VLVisionEncoder(vision_config)
```

这说明视觉 encoder 是可选的。

如果 `vision_config` 存在，就创建：

```text
Qwen3VLVisionEncoder
```

forward 中：

```python
if pixel_values is not None and self.visual is not None:
    image_embeds = self.visual(pixel_values, image_grid_thw)
```

然后传给 `Qwen3_5Model`，由模型主干把 image_embeds 替换到 image token 位置。

这和前面几个文件形成完整链路：

```text
llm_engine.py:
    process_messages 得到 pixel_values / image_grid_thw

sequence.py:
    Sequence 临时携带 pixel_values / image_grid_thw

model_runner.py:
    prepare_prefill 收集多模态张量
    计算 MRoPE positions
    传给 model

qwen3_5.py:
    visual encoder 生成 image_embeds
    scatter 到 image token hidden_states
```

---

## 20. forward 流程总览

`Qwen3_5ForCausalLM.forward()` 的整体流程是：

```text
输入:
    input_ids
    positions
    pixel_values
    image_grid_thw
    image_token_mask

如果有图像:
    image_embeds = visual(pixel_values, image_grid_thw)

进入语言模型主干:
    hidden_states = embed_tokens(input_ids)

如果有图像 embedding:
    hidden_states[image_token_mask] = image_embeds

逐层执行:
    full_attention 层:
        Qwen3_5Attention

    GDN 层:
        GatedDeltaNet

每层后接:
    GemmaRMSNorm
    Qwen3_5MLP

最后:
    Final GemmaRMSNorm
    返回 hidden_states

compute_logits:
    lm_head(hidden_states)
```

---

## 21. 和原版 Qwen3 dense 的核心差异

| 维度 | 原版 Qwen3 dense | qwen3_5.py |
|---|---|---|
| 层类型 | 每层 full attention | `layer_types` 决定 full attention 或 GatedDeltaNet |
| Attention 投影 | 常见 qkv 合并 | q/k/v 分开，q_proj 输出 q + gate |
| Attention 输出 | 直接 o_proj | `o * sigmoid(gate)` 后再 o_proj |
| Norm | RMSNorm | GemmaRMSNorm |
| RoPE | 标准 RoPE | InterleavedMRoPE / partial MRoPE |
| 多模态 | 通常无 | 可选 visual encoder + image token scatter |
| MTP | 通常无 | 可选 `Qwen3MTP` |
| 权重前缀 | 较简单 | `model.language_model.` 和 `model.visual.` |
| KV Cache | 所有层通常都有 KV | 只有 full attention 层需要 KV，GDN 层需要 state |

---

## 22. 对推理系统的影响

新增 `qwen3_5.py` 不只是模型文件变化，它会影响整个推理系统。

### 22.1 KV Cache 分配

因为不是所有层都是 full attention：

```text
只有 Qwen3_5Attention 层需要 KV Cache
GatedDeltaNet 层不需要 KV Cache
```

所以 `model_runner.py` 中需要：

```text
只统计有 k_cache/v_cache 的层数
```

而不是按 `num_hidden_layers` 分配 KV Cache。

### 22.2 GDN state 分配

GatedDeltaNet 层需要：

```text
conv_states
recurrent_states
```

所以需要：

```text
allocate_gdn_state()
Sequence.state_slot_id
Scheduler.StateSlotManager
Context.state_indices
```

### 22.3 CUDA Graph

decode 阶段如果使用 CUDA Graph，除了 input_ids、positions、slot_mapping、block_tables，还需要传：

```text
state_indices
```

否则 graph replay 时无法知道每个请求对应哪个 GDN state slot。

### 22.4 prefix cache

hybrid 模型中，如果只复用 KV Cache 而不复用 GDN state，会导致历史状态不一致。

所以 scheduler/block_manager 需要在 hybrid 下禁用或谨慎处理 prefix cache。

---

## 23. 和前面新增/修改文件的关系

这个文件是模型结构中心，前面很多文件的改造都是为它服务的。

| 文件 | 和 qwen3_5.py 的关系 |
|---|---|
| `config.py` | 提供 `is_hybrid`、`layer_types`、vision_config、rope 参数等 |
| `model_runner.py` | 自动创建 `Qwen3_5ForCausalLM`，准备多模态输入，分配 KV/GDN state |
| `sequence.py` | 为每个请求保存 `state_slot_id` 和多模态临时数据 |
| `scheduler.py` | 分配 GDN state slot，并在结束/抢占时释放 |
| `block_manager.py` | hybrid 下禁用 prefix cache，避免 KV 和 GDN state 不一致 |
| `context.py` | 传递 `state_indices` 给 GDN 层 |
| `layernorm.py` | 提供 `GemmaRMSNorm` |
| `rotary_embedding.py` | 提供 `InterleavedMRoPE` |
| `attention.py` | 为 full attention 层提供 KV Cache attention |
| `gated_delta_net.py` | 为非 full_attention 层提供 GDN 计算 |
| `vision_encoder.py` | 为多模态图像输入提供 image embeddings |
| `qwen3_mtp.py` | 为 MTP / draft token 路径提供模型模块 |

---

## 24. 为什么这个文件是项目核心新增文件

前面很多文件的改动都属于“系统层支持”：

```text
Sequence 增加 state_slot_id
Scheduler 增加 StateSlotManager
ModelRunner 分配 GDN state
BlockManager 禁用 prefix cache
Context 传 state_indices
```

但这些改动最终都是为了让模型结构可以执行。

`qwen3_5.py` 是真正把模型结构定义出来的地方：

```text
哪一层是 full attention
哪一层是 GatedDeltaNet
full attention 怎么算
GDN 层怎么接入
多模态 image embeddings 怎么进入 hidden_states
最后怎么接 LM Head
```

所以它是 `nano-vllm-qwen3.6` 支持 Qwen3.5 / Qwen3.6 hybrid 模型的核心模型文件。

---

## 25. 常见误区

### 误区一：qwen3_5.py 只是 qwen3.py 改名

不是。

它新增了 hybrid layer 选择、多模态视觉入口、MTP、GemmaRMSNorm、InterleavedMRoPE 和 gated attention。

### 误区二：所有层仍然都有 KV Cache

不是。

只有 full attention 层需要 KV Cache。GatedDeltaNet 层需要 recurrent/conv state。

### 误区三：GatedDeltaNet 是 MLP 的替代品

不是。

在这个文件里，GatedDeltaNet 替代的是 attention-like token mixing 子层。

每层后面仍然有 MLP。

### 误区四：多模态只是多传了 pixel_values

不只是。

pixel_values 需要经过 visual encoder 变成 image_embeds，然后 scatter 到 image token 的 hidden_states 位置，并且 positions 还需要支持 MRoPE。

### 误区五：MTP 是主 forward 必须执行的

不是。

MTP 是可选模块，只有 `enable_mtp` 时创建，执行逻辑由 `ModelRunner` 的 MTP 路径触发。

---

## 26. 面试角度应该怎么回答

如果面试官问：

> `qwen3_5.py` 这个新增文件解决了什么问题？

可以这样回答：

`qwen3_5.py` 是 nano-vllm-qwen3.6 中新增的模型结构文件，用来支持 Qwen3.5 / Qwen3.6 这类 hybrid 架构。原版 nano-vLLM 的 `qwen3.py` 更偏向普通 dense decoder-only Transformer，每层都是 full attention + MLP；而 `qwen3_5.py` 根据 `config.layer_types` 在每一层选择 `Qwen3_5Attention` 或 `GatedDeltaNet`，因此可以支持 full attention 层和 recurrent/linear attention 层混合。它的 full attention 也不是原版 qkv 合并结构，而是 q/k/v 分开，q_proj 输出 query 和 gate，attention 输出后再乘 `sigmoid(gate)` 做 output gating。同时它使用 `GemmaRMSNorm` 和 `InterleavedMRoPE`，支持 partial MRoPE 和多模态 3D positions。模型外壳还支持可选视觉 encoder，把 image embeddings scatter 到 image token 位置，以及可选 MTP 模块。整体来看，这个文件是 Qwen3.5/Qwen3.6 hybrid、多模态、MTP 能力在模型层的核心实现；系统层的 state_slot、GDN state、prefix cache 禁用等改造都是为了让这个模型文件能正确推理。

---

## 27. 初学者最应该抓住的主线

这个文件可以用一条主线理解：

```text
原版 qwen3.py:
    所有层都是 full attention
    历史状态主要是 KV Cache
    输入主要是文本 token

qwen3_5.py:
    有些层是 full attention
    有些层是 GatedDeltaNet
    attention 层用 KV Cache
    GDN 层用 recurrent/conv state
    输入可以是文本 + 图像
    还可以挂 MTP
```

最关键的三个代码点是：

```text
1. layer_types 决定 full_attention 或 GatedDeltaNet
2. q_proj 输出 q + gate，attention 输出做 gating
3. image_embeds scatter 到 image_token_mask 对应位置
```

---

## 28. 最终结论

`qwen3_5.py` 是 `nano-vllm-qwen3.6` 中非常核心的新增模型文件。

它的意义可以概括为：

```text
把 nano-vLLM 从只支持 Qwen3 dense 文本 Transformer，
扩展到可以支持 Qwen3.5 / Qwen3.6 hybrid、多模态和 MTP 模型。
```

具体来说，它完成了：

1. **Hybrid 架构支持**  
   通过 `config.layer_types` 在每层选择 `Qwen3_5Attention` 或 `GatedDeltaNet`。

2. **Gated full attention**  
   q_proj 同时产生 query 和 gate，attention 输出乘 `sigmoid(gate)`。

3. **GemmaRMSNorm 支持**  
   模型主干和 q/k norm 使用 Gemma 风格 RMSNorm。

4. **MRoPE 支持**  
   使用 `InterleavedMRoPE`，支持 partial RoPE 和多模态 3D positions。

5. **多模态支持**  
   可选视觉 encoder 生成 image_embeds，并替换 image token hidden states。

6. **MTP 支持**  
   可选加载 `Qwen3MTP`，为 draft / multi-token prediction 路径预留模型结构。

7. **权重加载适配**  
   通过 `weight_prefix`、`visual_prefix` 和 `packed_modules_mapping` 适配新的 checkpoint 命名和 MLP gate/up 打包。

因此，这个文件是理解 `nano-vllm-qwen3.6` 项目最关键的新增源码之一。

如果把前面分析过的文件串起来，可以得到完整工程逻辑：

```text
qwen3_5.py 定义 hybrid 模型结构
sequence.py 记录每个请求的 GDN state_slot_id
scheduler.py 分配和释放 state slot
model_runner.py 分配 KV Cache 和 GDN state，并准备多模态输入
context.py 把 state_indices 传给模型层
block_manager.py 在 hybrid 下避免不安全 prefix cache
rotary_embedding.py 提供 MRoPE
layernorm.py 提供 GemmaRMSNorm
```

这条链路就是 nano-vLLM 为支持 Qwen3.5 / Qwen3.6 hybrid 推理所做的核心工程改造。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
