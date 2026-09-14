# nano-vLLM-qwen3.6 整体改动宏观分析

> 目标读者：已经学习过原版 nano-vLLM，并逐个看过 `nano-vllm-qwen3.6` 源码对比文件，但还没有把分散在多个文件中的改动串成一套完整工程逻辑的学习者。
>
> 本文不再按“单个文件”逐行解释，而是按“大的工程方向”重新组织：每个方向为什么要改、涉及哪些源码文件、这些文件之间怎么配合、在推理流程中发挥什么作用。

---

## 0. 一句话总览

原版 nano-vLLM 可以理解为：

```text
面向 Qwen3 dense 文本模型的轻量推理框架
```

它的核心链路比较清晰：

```text
文本 prompt
  ↓ tokenizer
input_ids
  ↓ Scheduler 分配 KV Cache blocks
ModelRunner 准备 prefill/decode 输入
  ↓ Qwen3 dense model forward
Attention 写入/读取 KV Cache
  ↓ LM Head
Sampler 采样下一个 token
  ↓ Scheduler postprocess
循环 decode
```

`nano-vllm-qwen3.6` 的核心变化不是某个文件局部加几行代码，而是把这个轻量框架扩展成了一个实验型 Qwen3.5/Qwen3.6 推理工程：

```text
Qwen3 dense 文本推理
  ↓ 扩展为
Qwen3.5/Qwen3.6 hybrid + FP8 权重加载 + 多模态入口 + MTP/spec decode 实验
```

所以你看到的细碎改动，本质上服务于几条大主线：

| 大方向 | 要解决的问题 | 典型涉及文件 |
|---|---|---|
| 1. 新模型结构支持 | 原版只适合 Qwen3 dense，不能表达 Qwen3.5/Qwen3.6 hybrid | `qwen3_5.py`、`layernorm.py`、`rotary_embedding.py`、`model_runner.py` |
| 2. Hybrid 运行时状态管理 | GatedDeltaNet 层不使用 KV Cache，而需要 recurrent/conv state | `sequence.py`、`scheduler.py`、`block_manager.py`、`model_runner.py` |
| 3. FP8 checkpoint 加载 | Qwen3.6 FP8 权重不能直接 copy，需要 scale 反量化 | `loader.py`、`quant.py`、`linear.py`、`embed_head.py` |
| 4. 多模态输入支持 | 原版只处理文本，Qwen-VL 类模型需要 image preprocessing + vision encoder | `llm_engine.py`、`sequence.py`、`model_runner.py`、`image_processing.py`、`vision_encoder.py`、`qwen3_5.py`、`rotary_embedding.py` |
| 5. MTP / speculative decoding 原型 | 原版 decode 一次只生成一个 token；MTP 尝试 draft 多个 token 并 verify | `qwen3_mtp.py`、`model_runner.py`、`sampler.py`、`sequence.py`、`scheduler.py`、`block_manager.py`、根目录 `test_*.py/run_*.py` |
| 6. Tensor Parallel 输出与采样链路调整 | vocab / logits 分片后，LM Head 不再总是 gather 完整 logits | `embed_head.py`、`sampler.py`、`model_runner.py` |
| 7. 实验脚本与验证体系 | 新功能需要 smoke test、benchmark、rollback correctness test | `run_text_qwen35_v2.py`、`run_text_qwen36_fp8.py`、`bench_qwen35_fixed.py`、`bench_mtp_draft_sweep.py`、`test_state_rollback.py` 等 |

---

## 1. 先建立原版 nano-vLLM 的基准心智模型

在理解 qwen3.6 fork 之前，先把原版 nano-vLLM 的核心假设记住。

原版更像一个小型 vLLM：

```text
LLM / LLMEngine
  ↓
Sequence
  ↓
Scheduler
  ↓
BlockManager 管 KV Cache blocks
  ↓
ModelRunner
  ↓
Qwen3ForCausalLM
  ↓
Attention / MLP / RMSNorm / LM Head / Sampler
```

原版隐含了几个非常关键的假设：

```text
1. 模型是普通 decoder-only dense Transformer。
2. 每一层 decoder block 都有 full attention。
3. 每一层 attention 都需要 KV Cache。
4. 请求的历史状态基本可以用 KV Cache + token_ids 表示。
5. 输入主要是文本 token ids。
6. checkpoint 权重通常是 FP16/BF16/FP32，可以直接加载。
7. decode 是标准自回归，一次 forward 生成一个 token。
```

这些假设在 Qwen3 dense 上基本成立。

但到 Qwen3.5/Qwen3.6 后，问题变成：

```text
1. 有些层不再是 full attention，而是 GatedDeltaNet。
2. GatedDeltaNet 不写 KV Cache，而维护 recurrent/conv state。
3. KV Cache 不能再按总层数简单分配。
4. prefix cache 只复用 KV 会和 GDN state 不一致。
5. Qwen3.6 FP8 checkpoint 需要 scale 反量化。
6. 多模态模型需要图片预处理、vision encoder 和 MRoPE。
7. MTP/spec decode 需要保存、验证、回滚 decode 状态。
```

因此，qwen3.6 fork 的所有改动都可以理解为：

```text
把原版 nano-vLLM 的这些隐含假设逐个打破，然后补上对应工程机制。
```

---

## 2. 大方向一：新模型结构支持 —— 从 Qwen3 dense 到 Qwen3.5/Qwen3.6 hybrid

### 2.1 这个方向解决什么问题

原版 `qwen3.py` 更适合普通 dense Transformer：

```text
每层都是：
RMSNorm → Self Attention → RMSNorm → MLP
```

而 qwen3.5/qwen3.6 类模型引入 hybrid 结构：

```text
一部分层是 full_attention
另一部分层是 GatedDeltaNet / linear attention / recurrent 类模块
```

这意味着模型文件不能再简单写死为：

```text
for each layer:
    self_attn + mlp
```

而要变成：

```text
for each layer:
    if layer_type == full_attention:
        use Qwen3_5Attention
    else:
        use GatedDeltaNet
```

### 2.2 涉及到哪些学习源码对比文件

这个方向主要涉及：

```text
qwen3.6_qwen3_5.py_分析.md
qwen3.6_layernorm.py_对比分析.md
qwen3.6_rotary_embedding.py_对比分析.md
qwen3.6_model_runner.py_对比分析.md
qwen3.6_scheduler.py_对比分析.md
qwen3.6_sequence.py_对比分析.md
```

### 2.3 `qwen3_5.py` 是这个方向的核心

`qwen3_5.py` 是新模型结构的中心文件。

它新增的不是一个简单模型别名，而是一个新的模型家族实现：

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

这里最关键的是 `Qwen3_5DecoderLayer`：

```text
layer_type = config.layer_types[layer_idx]

if layer_type == "full_attention":
    self.self_attn = Qwen3_5Attention(...)
else:
    self.linear_attn = GatedDeltaNet(...)
```

这就是 hybrid 的本质。

原版 Qwen3 dense：

```text
所有层都是 attention 层
```

qwen3.6：

```text
有些层是 attention
有些层是 GDN
```

### 2.4 full attention 层本身也变了

`Qwen3_5Attention` 不是原版 Qwen3 attention 的简单复制。

它有几个明显变化：

```text
1. q_proj / k_proj / v_proj 分开，而不是简单 qkv_proj 合并。
2. q_proj 输出 query + gate，即输出维度是普通 query 的 2 倍。
3. Attention 输出后乘 sigmoid(gate)，形成 output gating。
4. q/k 使用 GemmaRMSNorm。
5. RoPE 使用 InterleavedMRoPE，而不是只支持标准 1D RoPE。
```

可以概括为：

```text
原版 Attention:
    QKV projection → q/k norm → RoPE → Attention → o_proj

Qwen3.5/3.6 Attention:
    q_proj 得到 q + gate
    k_proj / v_proj 单独计算
    q/k 用 GemmaRMSNorm
    q/k 加 InterleavedMRoPE
    Attention 输出乘 sigmoid(gate)
    再 o_proj
```

这说明 qwen3.6 fork 支持的新模型结构不仅是“有 GDN 层”，连 full attention 层内部也更复杂。

### 2.5 `layernorm.py` 为什么也属于模型结构支持

你之前单独看 `layernorm.py` 时可能觉得只是新增两个 norm 类。

但从整体看，它是 `qwen3_5.py` 能跑起来的基础。

qwen3.6 版本新增：

```text
GemmaRMSNorm
RMSNormGated
```

`qwen3_5.py` 中大量使用 `GemmaRMSNorm`：

```text
q_norm
k_norm
input_layernorm
post_attention_layernorm
final norm
```

所以 `layernorm.py` 的意义是：

```text
原版只有普通 RMSNorm，不足以表达 Qwen3.5/Qwen3.6 checkpoint 的 norm 定义。
新增 GemmaRMSNorm 是为了让模型参数语义和 checkpoint 对齐。
新增 RMSNormGated 则为 gated/hybrid 模块提供基础算子。
```

### 2.6 `rotary_embedding.py` 为什么也属于模型结构支持

原版 RoPE 假设很简单：

```text
positions 是一维 [N]
rotary_dim == head_size
整个 head 都旋转
```

qwen3.6 版本扩展为：

```text
1. partial RoPE：只旋转 head 的一部分维度。
2. InterleavedMRoPE：支持文本 1D positions 和多模态 3D positions。
```

它直接服务于 `Qwen3_5Attention`：

```text
Qwen3_5Attention 使用 InterleavedMRoPE 给 q/k 加位置编码。
```

所以 `rotary_embedding.py` 的宏观意义是：

```text
把 Attention 位置编码从“纯文本标准 RoPE”，扩展为“Qwen3.5/Qwen3.6 所需的 partial RoPE / MRoPE”。
```

### 2.7 这一方向的结论

这条主线可以这样记：

```text
qwen3_5.py 定义新模型结构；
layernorm.py 提供新 norm；
rotary_embedding.py 提供新位置编码；
model_runner.py 负责创建和运行这个新模型。
```

原版 nano-vLLM 只需要支持：

```text
Qwen3ForCausalLM
```

qwen3.6 需要支持：

```text
Qwen3 dense
Qwen3.5/Qwen3.6 hybrid
optional vision
optional MTP
```

---

## 3. 大方向二：Hybrid 运行时状态管理 —— KV Cache 之外新增 GDN state

### 3.1 这个方向解决什么问题

原版 nano-vLLM 的运行时历史状态主要是 KV Cache。

对普通 attention 层来说，decode 阶段每次只输入一个新 token，但它需要看到所有历史 token，于是使用：

```text
K Cache
V Cache
```

但 GatedDeltaNet 这类 recurrent/hybrid 层不是 full attention，它不使用 KV Cache，而是维护类似：

```text
recurrent state
conv state
```

因此，hybrid 模型的历史状态变成两套：

```text
full attention 层:
    KV Cache

GatedDeltaNet 层:
    recurrent/conv state
```

这会影响整个推理系统。

### 3.2 涉及到哪些学习源码对比文件

这个方向主要涉及：

```text
qwen3.6_sequence.py_对比分析.md
qwen3.6_scheduler.py_对比分析.md
qwen3.6_block_manager.py_对比分析.md
qwen3.6_model_runner.py_对比分析.md
qwen3.6_qwen3_5.py_分析.md
qwen3.6_llm_engine.py_对比分析.md
```

### 3.3 `sequence.py`：每个请求多了 `state_slot_id`

原版 `Sequence` 主要记录：

```text
token_ids
block_table
num_cached_tokens
num_scheduled_tokens
status
sampling params
```

这些足够描述一个普通 attention-only 请求。

qwen3.6 版本新增：

```text
state_slot_id
```

这个字段的意义是：

```text
当前请求在 GDN state 池中占用哪个 slot。
```

可以类比：

```text
block_table:
    记录这个请求的 token 历史对应哪些 KV Cache block

state_slot_id:
    记录这个请求的 GDN recurrent/conv state 存在哪个 state slot
```

所以 `Sequence` 从“KV Cache 请求状态对象”扩展为：

```text
KV Cache + GDN state 的统一请求状态对象
```

### 3.4 `scheduler.py`：从只分配 KV block 到同时分配 state slot

原版 Scheduler 的核心资源是：

```text
KV Cache blocks
```

qwen3.6 版本增加了类似：

```text
StateSlotManager
```

它负责管理 GDN state slot 的空闲和分配。

调度一个请求时，不仅要判断：

```text
KV blocks 够不够
```

还要判断：

```text
GDN state slot 够不够
```

所以调度逻辑变成：

```text
prefill 请求进入运行队列前：
    分配 KV blocks
    如果 hybrid，则分配 state_slot_id

请求结束或被抢占时：
    释放 KV blocks
    释放 state_slot_id
```

这就是 hybrid 模型对 Scheduler 的核心影响。

### 3.5 `block_manager.py`：hybrid 下 prefix cache 需要谨慎

原版 `BlockManager` 支持 prefix cache：

```text
如果两个请求有相同前缀 token，后一个请求可以复用前一个请求的 KV block。
```

这对 attention-only 模型通常是成立的，因为历史状态主要就是 KV Cache。

但 hybrid 模型中，如果只复用 KV Cache，而不复用 GDN recurrent/conv state，就会出现：

```text
Attention 层历史状态复用了；
GDN 层历史状态没复用；
两类层看到的“历史”不一致。
```

所以 qwen3.6 版本引入：

```text
disable_prefix_cache
```

Scheduler 在 hybrid 模型下会让 BlockManager 禁用 prefix cache。

这不是因为 prefix cache 不好，而是因为：

```text
KV-only prefix cache 与 GDN state 一致性之间存在冲突。
```

如果未来想重新支持 hybrid prefix cache，就必须同时缓存和恢复：

```text
KV Cache prefix
GDN recurrent state prefix
GDN conv state prefix
```

否则会造成隐蔽错误。

### 3.6 `model_runner.py`：真正分配和传递 GDN state

`model_runner.py` 是运行时核心。

qwen3.6 版本中，它要负责：

```text
1. 判断模型是不是 hybrid。
2. 只给 attention 层分配 KV Cache。
3. 给 GDN 层分配 recurrent/conv state tensors。
4. 根据 seq.state_slot_id 构造 state_indices。
5. forward 时把 state_indices 传给模型/GDN 层。
6. CUDA Graph replay 时也要保证 state_indices 正确。
```

这说明 GDN state 的生命周期贯穿：

```text
Sequence 记录 slot id
Scheduler 分配 slot
ModelRunner 分配 state tensor
Model forward 使用 state_indices 读写 state
请求结束释放 slot
```

### 3.7 这一方向的完整链路

可以把 hybrid state 链路画成：

```text
LLMEngine.add_request
  ↓
Sequence(state_slot_id=-1)
  ↓
Scheduler.schedule
  ├── BlockManager.allocate(seq)        # KV Cache blocks
  └── StateSlotManager.allocate()       # GDN state slot
          ↓
       seq.state_slot_id = slot
          ↓
ModelRunner.prepare_input
  ↓
state_indices = [seq.state_slot_id, ...]
  ↓
Qwen3_5DecoderLayer
  ├── full_attention 层 → Attention/KV Cache
  └── GatedDeltaNet 层 → recurrent/conv state[state_indices]
```

### 3.8 这一方向的结论

这一方向是 qwen3.6 相比原版最重要的系统层改造之一。

它解决的是：

```text
原版推理系统只知道 KV Cache；
hybrid 模型还需要管理 GDN recurrent/conv state。
```

涉及文件之间的分工是：

```text
sequence.py:
    每个请求记录 state_slot_id

scheduler.py:
    分配/释放 state slot

block_manager.py:
    hybrid 下避免不安全 prefix cache

model_runner.py:
    分配 GDN state tensors，并在 forward/CUDA graph 中传递 state_indices

qwen3_5.py:
    某些 decoder layer 真的调用 GatedDeltaNet
```

---

## 4. 大方向三：FP8 checkpoint 加载 —— 不是 runtime FP8，而是加载时反量化

### 4.1 这个方向解决什么问题

Qwen3.6-27B-FP8 checkpoint 的权重可能是：

```text
FP8 weight + weight_scale_inv
```

原版 nano-vLLM 的 loader 假设：

```text
读出来的权重可以直接 copy 到 nn.Parameter
```

这对 FP16/BF16 checkpoint 没问题，但对 FP8 不成立。

FP8 权重要结合 scale 才能还原：

```text
真实权重 ≈ FP8 weight × scale
```

因此 qwen3.6 增加了一条 FP8 加载链路。

### 4.2 涉及到哪些学习源码对比文件

这个方向主要涉及：

```text
qwen3.6_loader.py_对比分析.md
qwen3.6_quant.py_分析.md
qwen3.6_linear.py_对比分析.md
qwen3.6_embed_head.py_对比分析.md
qwen3.6_model_runner.py_对比分析.md
qwen3.6_qwen3_5.py_分析.md
```

### 4.3 `loader.py`：读取 FP8 weight 和 scale

qwen3.6 的 `loader.py` 增加了：

```text
LoadResult
loaded_names
skipped_names
loaded_scale
weight_prefix
visual_prefix
log_fn
maybe_dequant_fp8_weight
```

它在读取 safetensors 时会跳过：

```text
*.weight_scale_inv
```

因为 scale 不是普通模型参数。

然后对每个 weight：

```text
if loaded_weight.dtype 是 FP8 且存在 weight_name + "_scale_inv":
    loaded_scale = scale tensor
else:
    loaded_scale = None
```

再把：

```text
loaded_weight
loaded_scale
```

传给具体参数自己的 `weight_loader`。

### 4.4 `quant.py`：真正做 block-wise FP8 反量化

`quant.py` 是底层工具文件。

它的核心逻辑是：

```text
输入:
    weight: 当前权重，可能已经是 TP shard
    scale_inv: 完整权重对应的 block-wise scale
    row_start / col_start: 当前 shard 在完整矩阵中的偏移

处理:
    根据 row_start/col_start 找到对应 scale block
    repeat_interleave 展开 scale
    裁剪到和 weight 同 shape
    weight.float() * scale.float()
    转成 BF16

输出:
    BF16 weight
```

注意，这里非常关键：

```text
qwen3.6 当前实现的是 load-time dequantization。
```

也就是说：

```text
磁盘 checkpoint 是 FP8；
加载进模型参数后是 BF16；
forward 仍然主要走普通 BF16 F.linear / attention。
```

它不是：

```text
runtime FP8 GEMM
FP8 activation quantization
FP8 KV Cache
```

### 4.5 `linear.py`：TP 切片后再反量化

为什么 FP8 反量化不只改 `loader.py` 和 `quant.py`，还要改 `linear.py`？

因为 tensor parallel 下，每张卡只加载完整权重的一部分。

例如：

```text
ColumnParallelLinear:
    按输出维切，也就是按行切

RowParallelLinear:
    按输入维切，也就是按列切

QKVParallelLinear:
    q/k/v 分片更加复杂

MergedColumnParallelLinear:
    gate/up 合并权重也要按 shard 放入
```

FP8 scale 是按完整权重矩阵的 block 组织的。

所以反量化当前 shard 时，必须知道：

```text
这个 shard 在完整矩阵中的 row_start / col_start
```

qwen3.6 的 `linear.py` 做的就是：

```text
先根据 TP rank 切出当前 shard
再计算 row_start / col_start
再调用 maybe_dequant_fp8_weight
最后 copy 到 param
```

这叫：

```text
rank-local FP8 dequantization
```

优点是每张卡只反量化自己拥有的 shard，不需要先反量化完整权重。

### 4.6 `embed_head.py`：embedding / lm_head 也要支持 FP8

词表并行下：

```text
VocabParallelEmbedding / ParallelLMHead
```

按 vocab 维度切分。

所以它也要处理：

```text
row_start = 当前 TP rank 的 vocab 起始位置
loaded_scale = 对应 FP8 scale
```

qwen3.6 的 `embed_head.py` 不仅支持 FP8 反量化，还改变了 LM Head logits 的汇聚方式，这一点后面会单独讲。

### 4.7 `qwen3_5.py` 和 `loader.py` 的前缀映射关系

Qwen3.5/Qwen3.6 checkpoint 的权重命名可能不是原版那种简单参数名。

`qwen3_5.py` 中定义了：

```text
weight_prefix = "model.language_model."
visual_prefix = "model.visual."
packed_modules_mapping = {
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
```

于是 `loader.py` 要负责：

```text
model.language_model.*  → model.*
model.visual.*          → visual.*
gate_proj/up_proj       → gate_up_proj
```

所以 FP8 加载不只是数值格式问题，也是 checkpoint 命名空间映射问题。

### 4.8 这一方向的完整链路

```text
run_text_qwen36_fp8.py
  ↓
LLM(model=Qwen3.6-27B-FP8, tp=4, enable_vision=False)
  ↓
ModelRunner 创建 Qwen3_5ForCausalLM
  ↓
loader.py 遍历 safetensors
  ↓
读取 xxx.weight 和 xxx.weight_scale_inv
  ↓
根据 weight_prefix / visual_prefix / packed_modules_mapping 找到参数
  ↓
linear.py / embed_head.py 的 weight_loader 根据 TP rank 切 shard
  ↓
quant.py 根据 row_start / col_start 做 block-wise dequant
  ↓
BF16 权重 copy 到模型参数
  ↓
普通 BF16 forward
```

### 4.9 这一方向的结论

这条主线可以总结为：

```text
loader.py 负责读 scale 和映射名字；
linear.py/embed_head.py 负责 TP shard 切片和 offset；
quant.py 负责 block-wise FP8 → BF16 反量化。
```

面试或简历中要说准确：

```text
实现 FP8 checkpoint 加载兼容 / rank-local load-time dequantization。
```

不要误写成：

```text
实现 runtime FP8 推理
实现 FP8 KV Cache
实现 FP8 GEMM
```

---

## 5. 大方向四：多模态输入支持 —— 从纯文本 prompt 到 messages + image

### 5.1 这个方向解决什么问题

原版 nano-vLLM 的输入基本是：

```text
prompt: str
  ↓ tokenizer.encode
input_ids
```

多模态模型输入可能是：

```text
messages = [
  {role: user, content: [text part, image part, text part]}
]
```

这时不仅要生成 token ids，还要准备：

```text
pixel_values
image_grid_thw
image_token_mask
3D MRoPE positions
```

所以 qwen3.6 增加了一条完整多模态 prefill 链路。

### 5.2 涉及到哪些学习源码对比文件

这个方向主要涉及：

```text
qwen3.6_image_processing.py_分析.md
qwen3.6_vision_encoder.py_分析.md
qwen3.6_llm_engine.py_对比分析.md
qwen3.6_sequence.py_对比分析.md
qwen3.6_model_runner.py_对比分析.md
qwen3.6_qwen3_5.py_分析.md
qwen3.6_rotary_embedding.py_对比分析.md
qwen3.6_loader.py_对比分析.md
```

### 5.3 `image_processing.py`：多模态入口预处理

这个文件负责把 messages 中的图片转成模型能用的输入。

它做几件事：

```text
1. 解析 messages 中的 text/image part。
2. 对图片 smart_resize，使尺寸对齐 patch_size * merge_size。
3. RGB 转换、to_tensor、normalize。
4. 补 temporal 维度，适配 3D Conv patch embedding。
5. 按 Qwen-VL 风格 reshape/permute 成 pixel_values。
6. 生成 image_grid_thw = [grid_t, grid_h, grid_w]。
7. 在文本中插入：
   <|vision_start|> + <|image_pad|> * num_image_tokens + <|vision_end|>
8. 最终 tokenizer.encode 得到 token_ids。
```

输出是：

```text
token_ids
pixel_values
image_grid_thw
```

### 5.4 `llm_engine.py`：入口支持 str / token ids / messages

原版 `LLMEngine.add_request()` 主要处理：

```text
str prompt
list[int] token ids
```

qwen3.6 增加对：

```text
list[dict] messages
```

的支持。

当输入是多模态 messages 时，它会调用：

```text
process_messages()
```

得到：

```text
token_ids
pixel_values
image_grid_thw
```

然后把多模态张量挂到 `Sequence` 上。

### 5.5 `sequence.py`：临时携带图片张量

qwen3.6 的 `Sequence` 增加：

```text
pixel_values
image_grid_thw
```

它们用于首次 prefill。

注意：图片 tensor 不应该每轮 decode 都反复携带。

合理流程是：

```text
prefill 阶段：
    使用 pixel_values/image_grid_thw
    vision encoder 生成 image_embeds
    image_embeds 进入语言模型上下文

decode 阶段：
    不再重新处理图片
    依靠 KV Cache / GDN state 继续生成
```

所以 `Sequence` 只是多模态数据从入口到 ModelRunner 的临时载体。

### 5.6 `model_runner.py`：组织多模态 batch 和 MRoPE positions

ModelRunner 在 prefill 时要做：

```text
1. 收集 batch 中所有 seq.pixel_values。
2. 收集 image_grid_thw。
3. 构造 image_token_mask，标记哪些 token 是 image_pad。
4. 根据 image_grid_thw 计算 3D positions。
5. 把 pixel_values / image_grid_thw / image_token_mask / positions 传给模型。
```

这一步连接了输入预处理和模型 forward。

### 5.7 `vision_encoder.py`：图片内容编码

`vision_encoder.py` 是视觉塔。

它的作用是：

```text
pixel_values + image_grid_thw
  ↓
VisionPatchEmbed
  ↓
absolute position embedding interpolation
  ↓
VisionRotaryEmbedding
  ↓
VisionBlock × N
  ↓
VisionPatchMerger
  ↓
image_embeds
```

视觉 encoder 和语言模型不同：

```text
视觉 attention 是 non-causal。
语言 decoder attention 是 causal。
```

视觉 encoder 输出的 `image_embeds` 要和语言模型 hidden size 对齐。

### 5.8 `qwen3_5.py`：把 image_embeds 填入 token 序列

在语言模型主干中，先正常做：

```text
hidden_states = embed_tokens(input_ids)
```

然后：

```text
hidden_states[image_token_mask] = image_embeds
```

这表示：

```text
文本 token 位置：使用文本 embedding
图像 token 位置：使用视觉 encoder 输出
```

之后所有 token 一起进入 decoder layers。

### 5.9 `rotary_embedding.py`：语言模型侧的 MRoPE

多模态不只是把图片 embedding 塞进去，还需要正确的位置编码。

文本 token 只有一维位置：

```text
position = token index
```

图像 token 有 3D 位置：

```text
temporal / height / width
```

所以 `rotary_embedding.py` 中新增 `InterleavedMRoPE`，能处理：

```text
positions: [N]
positions: [3, N]
```

### 5.10 多模态完整链路

```text
用户输入 messages
  ↓
llm_engine.py
  ↓
image_processing.process_messages
  ├── token_ids
  ├── pixel_values
  └── image_grid_thw
  ↓
Sequence 保存 pixel_values / image_grid_thw
  ↓
Scheduler 调度 prefill
  ↓
ModelRunner.prepare_prefill
  ├── image_token_mask
  ├── MRoPE 3D positions
  ├── pixel_values batch
  └── image_grid_thw batch
  ↓
Qwen3_5ForCausalLM.forward
  ↓
vision_encoder.py 生成 image_embeds
  ↓
qwen3_5.py scatter 到 image token hidden_states
  ↓
hybrid decoder layers
```

### 5.11 这一方向的结论

多模态改造不是一个文件完成的，而是一条端到端链路：

```text
image_processing.py:
    图片预处理和 placeholder token 插入

llm_engine.py:
    接收 messages 输入

sequence.py:
    临时保存图片张量

model_runner.py:
    batch 化图片张量，生成 image mask 和 3D positions

vision_encoder.py:
    把 pixel_values 编码成 image_embeds

qwen3_5.py:
    把 image_embeds 填到 image token 位置

rotary_embedding.py:
    给 text/image token 提供 MRoPE
```

---

## 6. 大方向五：MTP / speculative decoding 原型 —— 从单 token decode 到 draft/verify/rollback

### 6.1 这个方向解决什么问题

原版 nano-vLLM 是标准自回归：

```text
每次 decode forward 生成 1 个 token
```

这在 decode 阶段有强串行依赖：

```text
token_t 生成后，才能生成 token_{t+1}
```

MTP 的思路是：

```text
主模型生成当前 token；
MTP 模块尝试预测后面若干 draft token；
再用主模型 verify；
如果 draft 和 target 一致，则一次接受多个 token；
如果不一致，则回滚状态并提交正确 token。
```

### 6.2 涉及到哪些学习源码对比文件

这个方向主要涉及：

```text
qwen3.6_qwen3_mtp.py_分析.md
qwen3.6_qwen3_5.py_分析.md
qwen3.6_model_runner.py_对比分析.md
qwen3.6_sampler.py_对比分析.md
qwen3.6_sequence.py_对比分析.md
qwen3.6_scheduler.py_对比分析.md
qwen3.6_block_manager.py_对比分析.md
nano-vllm-qwen3.6_根目录文件作用与源码改造分析.md
```

### 6.3 `qwen3_mtp.py`：MTP 模型结构

`qwen3_mtp.py` 新增：

```text
Qwen3MTPDecoderLayer
Qwen3MTP
```

MTP 输入：

```text
主模型 hidden_states
当前 token inputs_embeds
positions
```

MTP 流程：

```text
inputs_embeds → GemmaRMSNorm
hidden_states → GemmaRMSNorm
  ↓
concat([inputs_embeds, hidden_states])
  ↓
ReplicatedLinear(2H → H)
  ↓
Qwen3MTPDecoderLayer × num_layers
  ↓
Final GemmaRMSNorm
  ↓
mtp_hidden
  ↓
lm_head
  ↓
draft logits
```

它是 draft token 预测模块。

但要注意：

```text
qwen3_mtp.py 本身不是完整 speculative decoding 系统。
```

它只是提供 MTP forward 模块。

完整 spec decode 还需要：

```text
draft 生成
verify
accept/reject
KV/GDN state rollback
scheduler 状态恢复
benchmark
```

这些主要在 `model_runner.py` 和根目录测试脚本里。

### 6.4 `model_runner.py`：MTP 调用、verify、状态保存恢复

qwen3.6 的 `model_runner.py` 增加了大量 MTP/verify 相关接口，例如概念上包括：

```text
run_mtp_probe
run_mtp_draft_step
run_mtp_draft_fast_step
run_step_probe
run_verify_auto_probe
run_verify_batch_fast
save_decode_state
save_decode_state_range
restore_decode_state
drop_decode_state
reset_gdn_state_slots
```

它负责：

```text
1. 主模型 forward 得到 main token。
2. 调用 Qwen3MTP 生成 draft token。
3. 对 draft token 做 verify。
4. 保存和恢复 decode state。
5. 支持 eager/graph/chunk verify 模式。
6. 给测试脚本返回 token、score、top-k、logit diff、状态信息。
```

### 6.5 `sampler.py`：采样器从只返回 token 到返回 token + score

MTP/spec decode 里只知道 token 不够，还经常需要：

```text
token score
local rank score
top-k
verify score
```

qwen3.6 的 `sampler.py` 增加了：

```text
forward_with_scores
greedy_with_scores
```

这使得 ModelRunner 可以拿到：

```text
sample_tokens
sample_scores
```

尤其在 tensor parallel vocab sharding 下，每个 rank 只看到局部 logits，需要用 token+score 做跨 rank 比较和汇聚。

### 6.6 根目录测试脚本：把 MTP 原型跑成实验闭环

MTP 相关脚本承担不同层次验证：

```text
test_mtp_forward.py:
    验证 MTP 单步 forward、draft top-k、hidden/logits shape、权重加载情况

test_mtp1_verify.py:
    验证 draft_len=1 时，MTP draft token 与主模型 verify token 是否一致

test_mtp1_spec_decode.py:
    验证 draft_len=1 的 accept/reject 和 rollback

test_mtp_spec_decode.py:
    验证多 token draft、batch verify、greedy alignment、reject rerun、性能统计

run_mtp_fast_decode.py:
    去掉大量 debug probe，更接近 fast path benchmark

bench_mtp_draft_sweep.py:
    扫 draft_len，观察 accept_rate、tok/s、target_fw_per_tok、reject_reruns

test_state_rollback.py:
    专门验证 decode state rollback 是否正确
```

这些脚本不是“多余文件”，而是把 MTP 从模型模块变成可验证实验系统的关键。

### 6.7 为什么 state rollback 是 MTP/spec decode 的核心

speculative decoding 最大的问题是：

```text
你必须先假设 draft token 可能成立，去验证它；
但如果它不成立，不能让这次验证污染真实请求状态。
```

对普通 attention 模型，要回滚：

```text
Sequence token_ids
KV Cache blocks
scheduler queues
block_table
prefix hash
```

对 hybrid 模型，还要回滚：

```text
GDN recurrent state
GDN conv state
state_slot_id 对应 state tensor
```

所以 qwen3.6 的 `test_state_rollback.py` 非常重要。

它不是普通单元测试，而是在验证：

```text
同一个 decode state 保存后，恢复再跑，token 和 logits 是否一致。
```

如果 rollback 不正确，MTP reject 分支就会造成后续生成错误。

### 6.8 MTP/spec decode 完整链路

```text
当前请求处于 decode
  ↓
Scheduler.schedule
  ↓
ModelRunner.run_mtp_draft_step / run_mtp_draft_fast_step
  ├── 主模型 forward → main token
  └── Qwen3MTP forward → draft tokens
  ↓
Scheduler.postprocess 提交 main token
  ↓
保存 scheduler snapshot + decode state
  ↓
verify draft tokens
  ↓
比较 draft token 和 target token
  ↓
如果全部匹配:
    commit draft tokens

如果遇到 reject:
    restore scheduler
    restore decode state
    rerun trusted verify
    commit 正确 token
  ↓
drop saved state
```

### 6.9 这一方向的结论

这一方向可以这样理解：

```text
qwen3_mtp.py 提供 draft 模型结构；
model_runner.py 提供 draft/verify/save/restore 执行接口；
sampler.py 提供 token+score 采样；
test/run/bench 脚本把它们组合成可验证的 speculative decoding 原型。
```

它还不是生产级 spec decode，但已经覆盖了学习和简历项目中很有价值的关键点：

```text
draft token 预测
accept rate 测量
batch verify
CUDA Graph verify
reject rerun
state rollback
与 greedy 输出对齐
```

---

## 7. 大方向六：Tensor Parallel 输出与采样链路调整

### 7.1 这个方向解决什么问题

原版 `ParallelLMHead` 可能在 LM Head 内部把所有 rank 的 logits gather 成完整 vocab logits。

这种方式直观，但在更复杂 TP / FP8 / MTP 路径下不一定最合适。

qwen3.6 的 `embed_head.py` 做了一个关键变化：

```text
ParallelLMHead.forward() 返回当前 rank 的 local logits，
不再在 LM Head 内部 gather 成完整 vocab logits。
```

然后在 `ModelRunner.sample()` 中完成跨 rank token/score 汇聚。

### 7.2 涉及到哪些学习源码对比文件

这个方向主要涉及：

```text
qwen3.6_embed_head.py_对比分析.md
qwen3.6_sampler.py_对比分析.md
qwen3.6_model_runner.py_对比分析.md
qwen3.6_linear.py_对比分析.md
qwen3.6_loader.py_对比分析.md
```

### 7.3 为什么要这么改

当 vocab 被 tensor parallel 切分时：

```text
rank0 持有 vocab 的一部分
rank1 持有 vocab 的另一部分
...
```

每个 rank 只能算出自己的 local logits。

原版方式：

```text
LM Head 内部 gather 所有 local logits → full logits → sampler
```

qwen3.6 方式：

```text
LM Head 返回 local logits
Sampler 在本 rank 上选 local best/sample token 和 score
ModelRunner all_gather token+score
选全局 token
```

这种设计的好处是：

```text
1. LM Head 职责更单一，只负责 local projection。
2. 分布式采样逻辑集中在 ModelRunner。
3. MTP/verify/probe 等路径可以复用 token+score 汇聚机制。
4. 避免在 LM Head 中强制构造完整 vocab logits。
```

### 7.4 和 `sampler.py` 的关系

由于 LM Head 只返回 local logits，sampler 不能只返回 token id。

它还要返回：

```text
score
```

这样不同 rank 的候选 token 才能比较谁是全局最优或谁被采样到。

因此 `sampler.py` 新增：

```text
forward_with_scores
greedy_with_scores
```

这也是为什么 sampler 的改动看起来小，但在整体上很关键。

### 7.5 这一方向的结论

这条主线可以概括为：

```text
embed_head.py:
    LM Head 从“输出完整 logits”变成“输出 local logits”

sampler.py:
    从“只返回 token”变成“返回 token + score”

model_runner.py:
    负责跨 rank all_gather token/score 并确定最终 token
```

这服务于：

```text
TP vocab sharding
MTP draft sampling
verify/probe 输出分析
```

---

## 8. 大方向七：实验脚本与验证体系

### 8.1 为什么根目录脚本很重要

源码改造完成后，需要证明：

```text
模型能加载
文本能生成
FP8 能反量化
MTP 能 forward
verify 能对齐
rollback 能恢复
性能指标能统计
```

所以 qwen3.6 根目录新增了很多脚本。

这些脚本不是核心库代码，但它们是项目可展示、可验证、可写进简历的入口。

### 8.2 涉及到哪些学习文件

```text
nano-vllm-qwen3.6_根目录文件作用与源码改造分析.md
qwen3.6_qwen3_mtp.py_分析.md
qwen3.6_model_runner.py_对比分析.md
qwen3.6_scheduler.py_对比分析.md
qwen3.6_block_manager.py_对比分析.md
qwen3.6_quant.py_分析.md
qwen3.6_loader.py_对比分析.md
```

### 8.3 脚本如何对应功能

| 文件 | 验证对象 | 项目意义 |
|---|---|---|
| `run_text_qwen35_v2.py` | Qwen3.5 text-only | 验证 hybrid 模型能正常 generate |
| `run_text_qwen36_fp8.py` | Qwen3.6 FP8 text-only | 验证 FP8 checkpoint loading + TP text inference |
| `bench_qwen35_fixed.py` | Qwen3.5 固定性能 | 分离 prefill/decode 性能 |
| `test_mtp_forward.py` | MTP 单步 forward | 验证 MTP 权重、hidden/logits shape、draft top-k |
| `test_mtp1_verify.py` | MTP-1 draft/verify | 测最小 draft accept rate |
| `test_mtp1_spec_decode.py` | MTP-1 spec decode | 测 accept/reject 和 rollback |
| `test_mtp_spec_decode.py` | 多 token spec decode | 测 batch verify、greedy alignment、reject rerun |
| `run_mtp_fast_decode.py` | MTP fast path | 更接近性能 benchmark 的 fast decode 路径 |
| `bench_mtp_draft_sweep.py` | draft_len sweep | 比较不同 draft_len 的收益与开销 |
| `test_state_rollback.py` | 状态回滚 | 验证 KV/GDN/scheduler state 恢复一致性 |
| `bench.py` | 原版 throughput baseline | 保留原版 Qwen3 dense benchmark |

### 8.4 这一方向的结论

根目录脚本把内部源码能力串成实验闭环：

```text
run_*.py:
    跑通功能

bench_*.py:
    测性能

test_*.py:
    验证正确性
```

对你学习项目来说，根目录脚本非常适合帮助你“反向理解源码”：

```text
脚本调用哪个 LLM 参数？
调用了 model_runner 的哪个方法？
为什么需要 snapshot？
为什么要 compare greedy？
为什么要统计 accept_rate？
```

这些问题能把细碎源码串起来。

---

## 9. 把所有改动串成 4 条端到端主链路

前面按方向讲了很多，现在把它们压缩成 4 条真正的推理链路。

---

## 9.1 链路一：Qwen3.5 hybrid text-only 推理

```text
run_text_qwen35_v2.py / bench_qwen35_fixed.py
  ↓
LLM(... Qwen3.5-9B ...)
  ↓
LLMEngine.add_request(prompt)
  ↓
Sequence(token_ids)
  ↓
Scheduler.schedule
  ├── 分配 KV Cache blocks
  └── 如果 hybrid，分配 state_slot_id
  ↓
ModelRunner.prepare_input
  ├── input_ids
  ├── positions
  ├── slot_mapping
  ├── block_tables
  └── state_indices
  ↓
Qwen3_5ForCausalLM
  ↓
Qwen3_5DecoderLayer × N
  ├── full_attention 层 → Attention + KV Cache
  └── GatedDeltaNet 层 → recurrent/conv state
  ↓
LM Head local logits
  ↓
Sampler token+score
  ↓
ModelRunner 汇聚 TP 结果
  ↓
Scheduler.postprocess
```

这条链路对应的核心改动：

```text
qwen3_5.py
layernorm.py
rotary_embedding.py
sequence.py
scheduler.py
block_manager.py
model_runner.py
embed_head.py
sampler.py
```

---

## 9.2 链路二：Qwen3.6 FP8 text-only 推理

```text
run_text_qwen36_fp8.py
  ↓
LLM(... Qwen3.6-27B-FP8, tp=4, enable_vision=False ...)
  ↓
ModelRunner 创建模型
  ↓
loader.py 加载 safetensors
  ├── 跳过 weight_scale_inv 单独参数
  ├── 找到 loaded_scale
  ├── weight_prefix/visual_prefix 映射
  └── packed_modules_mapping 映射
  ↓
linear.py / embed_head.py 根据 TP rank 切 shard
  ↓
quant.py 根据 row_start/col_start 做 block-wise dequant
  ↓
BF16 参数进入模型
  ↓
后续推理走 hybrid text-only 链路
```

这条链路对应的核心改动：

```text
loader.py
quant.py
linear.py
embed_head.py
qwen3_5.py
model_runner.py
```

重点记忆：

```text
这是加载时 FP8 → BF16，不是 runtime FP8 GEMM。
```

---

## 9.3 链路三：多模态 prefill 推理

```text
用户传入 messages，其中包含 image part
  ↓
LLMEngine.add_request(messages)
  ↓
image_processing.process_messages
  ├── 文本拼接 ChatML
  ├── 图片 resize/normalize/patchify
  ├── 生成 pixel_values
  ├── 生成 image_grid_thw
  └── 插入 image_pad placeholder token
  ↓
Sequence
  ├── token_ids
  ├── pixel_values
  └── image_grid_thw
  ↓
Scheduler.schedule prefill
  ↓
ModelRunner.prepare_prefill
  ├── 收集 pixel_values
  ├── 收集 image_grid_thw
  ├── 构造 image_token_mask
  └── 计算 MRoPE 3D positions
  ↓
Qwen3_5ForCausalLM.forward
  ↓
vision_encoder.py
  └── pixel_values/image_grid_thw → image_embeds
  ↓
qwen3_5.py
  └── hidden_states[image_token_mask] = image_embeds
  ↓
后续 text/image token 一起进入 hybrid decoder layers
```

这条链路对应的核心改动：

```text
image_processing.py
llm_engine.py
sequence.py
model_runner.py
vision_encoder.py
qwen3_5.py
rotary_embedding.py
loader.py
```

---

## 9.4 链路四：MTP speculative decoding 原型

```text
test_mtp_spec_decode.py / run_mtp_fast_decode.py
  ↓
LLM(enable_mtp=True)
  ↓
普通 prefill 建立上下文
  ↓
decode round
  ↓
ModelRunner.run_mtp_draft_step / run_mtp_draft_fast_step
  ├── 主模型生成 main token
  └── Qwen3MTP 生成 draft tokens
  ↓
Scheduler.postprocess 提交 main token
  ↓
保存 scheduler snapshot + decode state
  ↓
ModelRunner.run_verify_auto_probe / run_verify_batch_fast
  ↓
比较 draft tokens 与 target tokens
  ↓
accept:
    commit draft tokens

reject:
    restore_scheduler
    restore_decode_state
    rerun trusted verify
    commit 正确 token
  ↓
统计 accept_rate / tok_s / target_fw_per_tok / reject_reruns
```

这条链路对应的核心改动：

```text
qwen3_mtp.py
model_runner.py
sampler.py
sequence.py
scheduler.py
block_manager.py
根目录 test/run/bench 脚本
```

---

## 10. 用“模块责任”重新理解所有文件

下面这张表把所有你学过的文件按责任重新分类。

| 模块责任 | 文件 | 一句话作用 |
|---|---|---|
| 请求入口 | `llm_engine.py` | 支持文本/token_ids/messages，接入多模态 process_messages |
| 请求状态 | `sequence.py` | 保存 token 状态、KV block_table、GDN state_slot_id、多模态临时张量 |
| 调度 | `scheduler.py` | 调度 waiting/running，请求进入运行前分配 KV blocks 和 state slots |
| KV block 管理 | `block_manager.py` | 管理 KV Cache block，hybrid 下禁用/限制 prefix cache |
| 执行核心 | `model_runner.py` | 准备输入、运行模型、采样、分配 KV/GDN state、CUDA Graph、MTP/verify/save/restore |
| 模型结构 | `qwen3_5.py` | 定义 Qwen3.5/Qwen3.6 hybrid 主模型，支持 attention/GDN/vision/MTP |
| MTP 模型 | `qwen3_mtp.py` | 定义 draft token prediction 的 MTP 辅助模块 |
| 视觉模型 | `vision_encoder.py` | 把 pixel_values/image_grid_thw 编码成 image_embeds |
| 图片预处理 | `image_processing.py` | messages 图片转 pixel_values/image_grid_thw，并插入 image_pad token |
| 线性层 | `linear.py` | TP linear 权重切片加载，支持 FP8 scale 反量化 |
| Embedding/LM Head | `embed_head.py` | vocab parallel embedding/head，支持 FP8 和 local logits |
| Norm | `layernorm.py` | 普通 RMSNorm、GemmaRMSNorm、RMSNormGated |
| RoPE/MRoPE | `rotary_embedding.py` | 标准 RoPE、partial RoPE、多模态 InterleavedMRoPE |
| Sampler | `sampler.py` | token 采样，qwen3.6 返回 token+score，支持 greedy 快路径 |
| Loader | `loader.py` | checkpoint 名字映射、FP8 scale 读取、visual/language prefix 处理 |
| Quant | `quant.py` | block-wise FP8 权重加载时反量化为 BF16 |
| 根目录脚本 | `run_*.py / test_*.py / bench_*.py` | 跑通、测试和评估 qwen3.6 新功能 |

---

## 11. 你最容易混淆的几个点

### 11.1 Qwen3.6 支持 FP8，不等于 runtime FP8 推理

当前项目中的 FP8 主要是：

```text
checkpoint 是 FP8
加载时反量化成 BF16
推理时权重常驻 BF16
```

不是：

```text
FP8 GEMM kernel
FP8 activation
FP8 KV Cache
```

涉及文件：

```text
loader.py
quant.py
linear.py
embed_head.py
run_text_qwen36_fp8.py
```

### 11.2 GDN state 不等于 KV Cache

Attention 层的历史状态是：

```text
K/V tensors
```

GatedDeltaNet 层的历史状态是：

```text
recurrent/conv state
```

所以 hybrid 模型必须同时管理：

```text
KV block_table
state_slot_id
```

涉及文件：

```text
sequence.py
scheduler.py
model_runner.py
block_manager.py
qwen3_5.py
```

### 11.3 prefix cache 在 hybrid 下不能简单复用

原版 prefix cache 复用 KV block。

hybrid 下如果只复用 KV，不复用 GDN state，会造成状态不一致。

所以 qwen3.6 对 hybrid 禁用或限制 prefix cache。

涉及文件：

```text
scheduler.py
block_manager.py
sequence.py
```

### 11.4 MTP 模块不等于完整 speculative decoding

`qwen3_mtp.py` 只是 MTP 模型结构。

完整 spec decode 还需要：

```text
ModelRunner draft/verify
Sampler token+score
Scheduler/KV/GDN state rollback
根目录脚本验证 accept/reject
```

涉及文件：

```text
qwen3_mtp.py
model_runner.py
sampler.py
test_mtp_spec_decode.py
run_mtp_fast_decode.py
test_state_rollback.py
```

### 11.5 多模态不是只多传 pixel_values

多模态至少包括：

```text
图片预处理
placeholder token 插入
image_grid_thw
vision encoder
image_embeds scatter
MRoPE 3D positions
```

涉及文件：

```text
image_processing.py
llm_engine.py
sequence.py
model_runner.py
vision_encoder.py
qwen3_5.py
rotary_embedding.py
```

---

## 12. 如果面试官问“这个项目到底改了什么”，可以这样回答

可以按 5 段回答。

第一段，模型结构：

> 我基于 nano-vLLM 原版 Qwen3 dense 文本推理框架，扩展了 Qwen3.5/Qwen3.6 hybrid 模型结构。原版默认每层都是 full attention + MLP，而 qwen3.6 版本新增 `qwen3_5.py`，根据 `config.layer_types` 在每层选择 full attention 或 GatedDeltaNet。full attention 层本身也改成带 query/output gate、GemmaRMSNorm 和 InterleavedMRoPE 的结构。

第二段，运行时状态：

> 为了支持 GatedDeltaNet，推理状态不能只依赖 KV Cache。我在 Sequence/Scheduler/ModelRunner 链路中引入了 `state_slot_id` 和 StateSlotManager，为每个请求分配 GDN recurrent/conv state slot；ModelRunner 只给 attention 层分配 KV Cache，同时维护 GDN state tensors，并在 decode/CUDA Graph 中传递 state_indices。hybrid 下还需要禁用或限制 KV-only prefix cache，避免 KV Cache 与 GDN state 不一致。

第三段，FP8 权重加载：

> qwen3.6 支持 FP8 checkpoint 加载，但不是 runtime FP8 GEMM。loader 读取 safetensors 中的 `weight_scale_inv`，linear/embed/lm_head 的 weight_loader 根据 tensor parallel rank 切出当前 shard，并把 row/col offset 传给 quant.py 做 block-wise FP8 反量化，最终加载为 BF16 参数。这样每个 TP rank 只反量化自己拥有的 shard。

第四段，多模态：

> 多模态方面新增了 image_processing 和 vision_encoder。入口侧把 messages 中的图片 resize、normalize、patchify，生成 pixel_values 和 image_grid_thw，并在文本中插入 image_pad tokens；ModelRunner prefill 阶段构造 image_token_mask 和 MRoPE 3D positions；vision encoder 将图片编码成 image_embeds，qwen3_5 模型再把 image_embeds scatter 到 image token 对应的 hidden_states 位置。

第五段，MTP/spec decode：

> 项目还新增了 Qwen3.6 MTP prototype。qwen3_mtp.py 定义 MTP 辅助预测模块，用当前 token embedding 和主模型 hidden state 预测 draft hidden；ModelRunner 接入 run_mtp_probe、draft_step、verify、save/restore decode state 等接口；sampler 支持返回 token+score；根目录测试脚本验证 MTP forward、MTP-1 verify、多 token spec decode、accept/reject、greedy alignment 和 state rollback。

---

## 13. 从简历项目角度提炼

这个项目可以提炼成一个比较完整的 AI Infra 推理项目：

```text
基于 nano-vLLM 扩展 Qwen3.5/Qwen3.6 推理支持：实现 hybrid 模型结构接入、GatedDeltaNet state 管理、FP8 checkpoint rank-local dequant、Qwen-VL 多模态输入链路和 MTP/speculative decoding prototype，并通过 run/bench/test 脚本验证 text inference、prefill/decode 性能、MTP accept-rate、draft_len sweep 与 KV/GDN state rollback correctness。
```

可以拆成 4 个简历 bullet：

```text
1. 模型结构与状态管理：新增 Qwen3.5/Qwen3.6 hybrid 模型，支持 full attention 与 GatedDeltaNet 混合；改造 Sequence/Scheduler/ModelRunner，为请求分配 KV Cache blocks 与 GDN state slots，并在 CUDA Graph decode 中传递 state_indices。

2. FP8 checkpoint 加载：改造 loader/linear/embed_head/quant，支持 safetensors 中 FP8 weight + weight_scale_inv，按 TP rank 切片后进行 block-wise dequant，加载为 BF16 参数，支持 Qwen3.6-27B-FP8 text-only 推理。

3. 多模态链路：新增 image_processing 与 vision_encoder，支持 messages 中图片预处理、image_pad token 对齐、image_grid_thw、vision encoder image_embeds 生成，以及 qwen3_5 模型中的 image token embedding 替换和 MRoPE positions。

4. MTP/spec decode 原型：新增 Qwen3MTP 模块与 ModelRunner draft/verify/save/restore 接口，支持 MTP forward probe、MTP-1 verify、多 token batch verify、accept/reject、greedy alignment 和 KV/GDN state rollback 测试。
```

指标可以从这些脚本中收集：

```text
bench_qwen35_fixed.py:
    prefill tok/s
    decode tok/s
    total out_tok/s

run_text_qwen36_fp8.py:
    Qwen3.6 FP8 text-only 是否跑通
    decode tok/s

bench_mtp_draft_sweep.py:
    draft_len
    accept_rate
    tok/s
    target_forwards_per_token
    mtp_forwards_per_token
    reject_reruns

test_state_rollback.py:
    rollback_ok
    max_logit_diff
```

---

## 14. 你后续复习时应该怎么抓主线

不要再孤立地看某个文件“改了几行”。

建议按下面 5 个问题复习：

### 问题 1：Qwen3.5/Qwen3.6 为什么不能直接用原版 qwen3.py？

看：

```text
qwen3_5.py
layernorm.py
rotary_embedding.py
```

答案：

```text
因为模型结构变成 hybrid，attention 层带 gate，norm/RoPE 定义也不同。
```

### 问题 2：为什么 Scheduler/Sequence/ModelRunner 都要改？

看：

```text
sequence.py
scheduler.py
block_manager.py
model_runner.py
```

答案：

```text
因为 GDN 层需要 recurrent/conv state，不能只靠 KV Cache。
```

### 问题 3：FP8 支持到底在哪一层实现？

看：

```text
loader.py
quant.py
linear.py
embed_head.py
```

答案：

```text
loader 读 scale，linear/embed 计算 TP shard offset，quant 做 block-wise dequant。
```

### 问题 4：图片输入怎么进入语言模型？

看：

```text
image_processing.py
llm_engine.py
sequence.py
model_runner.py
vision_encoder.py
qwen3_5.py
rotary_embedding.py
```

答案：

```text
图片先变 pixel_values，再经 vision encoder 变 image_embeds，最后替换 image_pad token 的 hidden_states。
```

### 问题 5：MTP/spec decode 为什么一定要 rollback？

看：

```text
qwen3_mtp.py
model_runner.py
sampler.py
test_state_rollback.py
test_mtp_spec_decode.py
run_mtp_fast_decode.py
```

答案：

```text
因为 draft token 可能被拒绝，验证过程不能污染真实 KV Cache / GDN state / scheduler 状态。
```

---

## 15. 最终总结

`nano-vllm-qwen3.6` 相比原版 `nano-vLLM` 的改动，可以归纳为一句话：

```text
它把一个只需要管理 KV Cache 的 Qwen3 dense 文本推理小框架，扩展成了一个能够实验 Qwen3.5/Qwen3.6 hybrid、FP8 checkpoint、多模态输入和 MTP/spec decode 的推理研究框架。
```

最核心的系统变化是：

```text
1. 模型不再全是 attention，而是 attention + GDN hybrid。
2. 状态不再只有 KV Cache，而是 KV Cache + GDN state。
3. 输入不再只有文本，而是可以包含图片。
4. 权重不再都是直接 copy 的 BF16/FP16，而是可以是 FP8 + scale。
5. decode 不再只考虑单 token 自回归，还加入 MTP draft/verify/rollback 原型。
```

把这些变化对应到文件：

```text
模型结构:
    qwen3_5.py / layernorm.py / rotary_embedding.py

运行时状态:
    sequence.py / scheduler.py / block_manager.py / model_runner.py

FP8 加载:
    loader.py / quant.py / linear.py / embed_head.py

多模态:
    image_processing.py / vision_encoder.py / llm_engine.py / sequence.py / model_runner.py / qwen3_5.py

MTP/spec decode:
    qwen3_mtp.py / model_runner.py / sampler.py / test_*.py / run_mtp_fast_decode.py

验证与 benchmark:
    bench_qwen35_fixed.py / bench_mtp_draft_sweep.py / test_state_rollback.py / run_text_qwen36_fp8.py
```

真正理解这个项目的关键，不是记住每个文件改了哪些行，而是记住这些改动背后的工程因果链：

```text
因为模型变 hybrid
  → 所以需要 GDN state
  → 所以 Sequence/Scheduler/ModelRunner 要改
  → 所以 prefix cache 不能只按 KV 复用
  → 所以 rollback 也要覆盖 KV + GDN state

因为 checkpoint 是 FP8
  → 所以 loader 要读 scale
  → 所以 linear/embed 要传 shard offset
  → 所以 quant 要按 block 反量化

因为支持多模态
  → 所以入口要处理 messages/image
  → 所以 Sequence 要携带 pixel_values
  → 所以 ModelRunner 要构造 image mask 和 MRoPE positions
  → 所以模型要接 vision encoder 并 scatter image_embeds

因为尝试 MTP/spec decode
  → 所以要新增 MTP 模块
  → 所以 sampler 要返回 token+score
  → 所以 ModelRunner 要支持 draft/verify/save/restore
  → 所以根目录测试脚本要验证 accept/reject/rollback/greedy alignment
```

这就是从宏观上理解 `nano-vllm-qwen3.6` 的主线。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
