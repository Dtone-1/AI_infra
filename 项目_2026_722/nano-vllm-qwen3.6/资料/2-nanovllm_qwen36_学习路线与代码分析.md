# Nano-vLLM Qwen3.6 项目学习路线与代码级分析

> 目标：在已经熟悉原始 Nano-VLLM `engine/` 目录的基础上，系统补齐 `layers/`、`models/`、`utils/`，再理解 `nano-vllm-qwen3.6` 相比原项目的新增模型架构、状态管理、FP8 checkpoint 加载、CUDA Graph decode 和 MTP 实验逻辑。

---

## 0. 你的当前状态与最合理的学习策略

你现在已经学习并熟悉了原始 Nano-VLLM 的 `engine/` 文件夹，这说明你已经理解了推理引擎中最上层的执行链路：

```text
LLM / LLMEngine
  -> Scheduler
  -> Sequence / BlockManager
  -> ModelRunner
  -> prefill / decode
  -> KV cache / block table
  -> sampler
```

但是你目前还没有系统学习：

```text
nanovllm/layers/
nanovllm/models/
nanovllm/utils/
```

所以不能直接跳进 `nano-vllm-qwen3.6` 的 `GatedDeltaNet`、`FP8`、`MTP`。原因是：

```text
Qwen3.6 fork 的新增内容，全部建立在原 Nano-VLLM 的 layer/model/utils 机制之上。
```

因此学习路线不应该是：

```text
直接读 qwen3.6 新增代码
```

而应该是：

```text
原始 Nano-VLLM 剩余模型相关代码
  -> 原始 Qwen3 dense 推理链路
  -> Qwen3.5/Qwen3.6 hybrid 架构差异
  -> fork 新增状态管理与 FP8/MTP 实验
```

最推荐的学习顺序是：

```text
第一阶段：补齐原始 Nano-VLLM 的 layers/models/utils
第二阶段：完整串起 Qwen3 dense 模型的一次 forward
第三阶段：理解 Qwen3.5/Qwen3.6 相比 Qwen3 的模型结构变化
第四阶段：学习 fork 的 runtime 改造：KV cache + GDN state
第五阶段：学习 FP8 checkpoint 加载，不要误认为是原生 FP8 推理
第六阶段：学习 CUDA Graph decode 在 hybrid 模型下的变化
第七阶段：最后再看 MTP / speculative decode 原型
第八阶段：整理成简历项目与面试问答
```

---

## 1. 项目整体定位：这个 fork 到底做了什么

`nano-vllm-qwen3.6` 不是单纯把模型名从 Qwen3 改成 Qwen3.6。它本质上是把原始 Nano-VLLM 从一个“小型 Qwen3 dense attention 推理引擎”，扩展成一个“支持 Qwen3.5/Qwen3.6 hybrid 模型实验的学习型推理引擎”。

原始 Nano-VLLM 的核心路径是：

```text
Qwen3ForCausalLM
  -> Qwen3Model
    -> Qwen3DecoderLayer
      -> Qwen3Attention
      -> Qwen3MLP
```

也就是每一层都是：

```text
RMSNorm
  -> full attention
  -> RMSNorm
  -> MLP
```

而 `nano-vllm-qwen3.6` 新增了：

```text
Qwen3_5ForCausalLM
  -> Qwen3_5Model
    -> Qwen3_5DecoderLayer
      -> full_attention 层：Qwen3_5Attention
      -> linear-attention 层：GatedDeltaNet
      -> Qwen3_5MLP
```

所以新项目的核心变化是：

```text
从纯 full attention 模型
变成 full attention + GatedDeltaNet 的 hybrid 模型
```

这会进一步引出一系列推理框架层面的变化：

```text
1. KV cache 不能再按总层数分配，只能给 full attention 层分配。
2. GatedDeltaNet 层不使用 KV cache，而是维护 recurrent state 和 conv state。
3. Scheduler 除了管理 KV block，还要管理 GDN state slot。
4. Decode CUDA Graph 里除了 token、position、slot_mapping、block_table，还要传 state_indices。
5. MTP speculative decode 发生 reject 时，不仅要回滚 KV cache，还要回滚 GDN state。
6. Qwen3.6-FP8 checkpoint 需要支持 FP8 权重 + scale_inv 的加载和反量化。
```

---

## 2. 代码目录对比：原始 Nano-VLLM vs Qwen3.6 fork

### 2.1 原始 Nano-VLLM 的 `nanovllm/` 目录

原始项目核心文件如下：

```text
nanovllm/
  config.py
  llm.py
  sampling_params.py
  engine/
    block_manager.py
    llm_engine.py
    model_runner.py
    scheduler.py
    sequence.py
  layers/
    activation.py
    attention.py
    embed_head.py
    layernorm.py
    linear.py
    rotary_embedding.py
    sampler.py
  models/
    qwen3.py
  utils/
    context.py
    loader.py
```

这个版本只包含一个主模型定义：

```text
models/qwen3.py
```

也就是 Qwen3 dense text 模型。

---

### 2.2 Qwen3.6 fork 新增的关键文件

`nano-vllm-qwen3.6` 在原始基础上新增了：

```text
nanovllm/layers/gated_delta_net.py
nanovllm/models/qwen3_5.py
nanovllm/models/qwen3_mtp.py
nanovllm/models/vision_encoder.py
nanovllm/utils/image_processing.py
nanovllm/utils/quant.py
```

同时新增了多个运行脚本和实验脚本：

```text
run_text_qwen35_v2.py
run_text_qwen36_fp8.py
run_mtp_fast_decode.py
bench_qwen35_fixed.py
bench_mtp_draft_sweep.py
test_mtp_forward.py
test_mtp1_verify.py
test_mtp1_spec_decode.py
test_mtp_spec_decode.py
test_state_rollback.py
```

其中最重要的是：

```text
gated_delta_net.py       # Qwen3.5/Qwen3.6 hybrid 模型的线性注意力层
qwen3_5.py               # Qwen3.5/Qwen3.6 主模型定义
qwen3_mtp.py             # MTP draft 模型原型
quant.py                 # FP8 block dequant 工具
model_runner.py          # 推理运行时大幅扩展
scheduler.py             # 新增 GDN state slot 管理
sequence.py              # Sequence 新增 state_slot_id 和多模态字段
```

---

## 3. 第一阶段：先补齐原始 Nano-VLLM 的 `layers/`、`models/`、`utils/`

这一阶段的目标不是看 Qwen3.6，而是把原始 Nano-VLLM 剩余代码补齐。你已经看懂 engine，那么现在要补齐 engine 下面真正执行模型计算的部分。

推荐学习顺序：

```text
utils/context.py
  -> layers/linear.py
  -> layers/embed_head.py
  -> layers/layernorm.py
  -> layers/activation.py
  -> layers/rotary_embedding.py
  -> layers/attention.py
  -> layers/sampler.py
  -> utils/loader.py
  -> models/qwen3.py
```

---

### 3.1 `utils/context.py`：理解运行时上下文

原始 Nano-VLLM 通过全局 `Context` 保存一次 forward 所需的运行时信息。

核心字段包括：

```text
is_prefill
cu_seqlens_q
cu_seqlens_k
max_seqlen_q
max_seqlen_k
slot_mapping
context_lens
block_tables
```

你需要理解：

```text
prefill 阶段：需要 cu_seqlens_q / cu_seqlens_k / slot_mapping / block_tables

decode 阶段：需要 slot_mapping / context_lens / block_tables
```

学习重点：

```text
1. ModelRunner.prepare_prefill() 如何 set_context()
2. ModelRunner.prepare_decode() 如何 set_context()
3. Attention.forward() 如何 get_context()
4. 为什么这些信息不作为 attention.forward() 的显式参数传入
```

你应该能画出：

```text
ModelRunner.prepare_xxx()
  -> set_context(...)
  -> model(input_ids, positions)
  -> Qwen3Attention.forward()
  -> Attention.forward()
  -> get_context()
```

---

### 3.2 `layers/linear.py`：理解 Tensor Parallel 线性层

这是原始 Nano-VLLM 里最重要的 layer 文件之一。

需要重点理解这些类：

```text
ReplicatedLinear
ColumnParallelLinear
MergedColumnParallelLinear
QKVParallelLinear
RowParallelLinear
```

#### Column Parallel

Column parallel 是按输出维度切分权重。

原始权重：

```text
W: [out_features, in_features]
```

TP=2 时：

```text
rank0: W[:out/2, :]
rank1: W[out/2:, :]
```

前向时每个 rank 计算一部分输出，不需要马上 all-reduce。

典型用于：

```text
q_proj / k_proj / v_proj
gate_proj / up_proj
vocab embedding / lm_head
```

#### Row Parallel

Row parallel 是按输入维度切分权重。

TP=2 时：

```text
rank0: W[:, :in/2]
rank1: W[:, in/2:]
```

每个 rank 计算部分结果，最后需要：

```text
dist.all_reduce(y)
```

典型用于：

```text
o_proj
down_proj
```

#### MergedColumnParallelLinear

用于把两个矩阵合并加载，例如：

```text
gate_proj + up_proj -> gate_up_proj
```

这是 Qwen / LLaMA 类 MLP 常见优化。

#### QKVParallelLinear

用于把：

```text
q_proj
k_proj
v_proj
```

合并成一个：

```text
qkv_proj
```

你要重点理解 `packed_modules_mapping` 为什么能把 checkpoint 里的 `q_proj.weight`、`k_proj.weight`、`v_proj.weight` 加载进同一个 `qkv_proj.weight`。

---

### 3.3 `layers/attention.py`：理解 FlashAttention + KV Cache

`Attention.forward()` 是原始 Nano-VLLM 模型计算中最关键的逻辑之一。

它做三件事：

```text
1. 把当前 token 的 K/V 写入 KV cache。
2. prefill 阶段调用 flash_attn_varlen_func。
3. decode 阶段调用 flash_attn_with_kvcache。
```

你需要区分：

```text
prefill：一次处理 prompt 的多个 token

decode：一次处理每个序列的最后一个 token
```

在 prefill 中：

```text
q/k/v 是当前 prompt 的整段张量
cu_seqlens_q / cu_seqlens_k 用于告诉 FlashAttention 每条序列的边界
slot_mapping 用于把 K/V 写到物理 KV cache slot
```

在 decode 中：

```text
q 是当前 token 的 query
k_cache/v_cache 是历史 K/V
context_lens 是每条序列当前长度
block_tables 是逻辑 block 到物理 block 的映射
```

---

### 3.4 `models/qwen3.py`：把模型结构完整串起来

原始 Qwen3 模型结构是：

```text
Qwen3ForCausalLM
  -> Qwen3Model
    -> embed_tokens
    -> 多层 Qwen3DecoderLayer
      -> input_layernorm
      -> Qwen3Attention
      -> post_attention_layernorm
      -> Qwen3MLP
    -> final norm
  -> lm_head
```

你需要跟一遍 forward：

```text
input_ids
  -> VocabParallelEmbedding
  -> hidden_states
  -> Qwen3DecoderLayer × N
  -> norm
  -> lm_head
  -> logits
```

重点问题：

```text
1. Qwen3Attention 里 q/k/v 的 shape 如何变化？
2. num_attention_heads 和 num_key_value_heads 有什么区别？
3. q_norm / k_norm 为什么只作用在 head_dim 上？
4. rotary_embedding 作用在哪里？
5. MLP 里的 gate_up_proj 为什么要合并？
6. RowParallelLinear 为什么 forward 后要 all_reduce？
7. compute_logits() 为什么单独放在 ForCausalLM 中？
```

这一阶段完成后，你应该能说清楚：

```text
原始 Nano-VLLM 是如何从 token ids 经过 embedding、attention、MLP、lm_head 得到 logits 的。
```

---

## 4. 第二阶段：理解 Qwen3、Qwen3.5、Qwen3.6 的区别

### 4.1 Qwen3：原始 dense full attention 路径

在原始 Nano-VLLM 里，Qwen3 是标准 decoder-only dense 模型。

每一层都是：

```text
full attention + MLP
```

因此每一层都需要 KV cache。

它的推理状态主要是：

```text
KV cache
block table
slot mapping
context lens
```

这就是你在原始 `engine/` 里学到的内容。

---

### 4.2 Qwen3.5：hybrid 架构

Qwen3.5 在这个 fork 中通过 `models/qwen3_5.py` 实现。

关键变化是 `config.layer_types`：

```python
layer_type = config.layer_types[layer_idx]

if layer_type == "full_attention":
    self.self_attn = Qwen3_5Attention(config, layer_idx)
    self.linear_attn = None
else:
    self.linear_attn = GatedDeltaNet(config, layer_idx)
    self.self_attn = None
```

也就是说，不同层的计算模块不同：

```text
full_attention 层：仍然走普通 attention，需要 KV cache
非 full_attention 层：走 GatedDeltaNet，需要 recurrent state / conv state
```

这就是 hybrid 模型的核心。

---

### 4.3 Qwen3.6：在本项目中的定位

这个项目里的 Qwen3.6 重点是：

```text
Qwen3.6-27B-FP8 text-only inference
```

但必须注意：

```text
它不是原生 FP8 matmul 推理。
```

当前实现是：

```text
FP8 checkpoint
  -> 根据 weight_scale_inv 做 block dequant
  -> 转成 BF16 resident weights
  -> 用普通 BF16 计算路径推理
```

所以简历或面试里不能说：

```text
实现了原生 FP8 高性能推理算子
```

更准确的说法是：

```text
支持 Qwen3.6-FP8 checkpoint 的 TP rank-local 反量化加载，将 FP8 safetensors 权重按 rank 切片后转换为 BF16 resident weights。
```

---

## 5. 第三阶段：重点理解新术语

下面这些术语是你学习 `nano-vllm-qwen3.6` 前必须搞懂的。

### 5.1 Hybrid model

Hybrid model 在这里指：

```text
同一个 decoder-only 模型里，部分层是 full attention，部分层是 linear attention / GatedDeltaNet。
```

它不是 MoE，也不是简单的多模型融合。

核心区别：

```text
full attention 层：依赖 KV cache
GatedDeltaNet 层：依赖 recurrent state + conv state
```

---

### 5.2 GatedDeltaNet

`GatedDeltaNet` 是这个项目新增的核心层。

它可以理解为一种带门控的线性注意力 / recurrent state 层。

它的主要状态有两个：

```text
conv_states
recurrent_states
```

其中：

```text
conv_states：用于 causal depthwise conv1d 的滑动窗口状态。
recurrent_states：用于 gated delta rule 的递推记忆状态。
```

在 decode 阶段，每个序列每生成一个 token，都要更新它对应的：

```text
conv_state
recurrent_state
```

所以它不像 full attention 那样只追加 K/V，而是会不断更新状态。

---

### 5.3 State slot

由于 batch 中有多条 sequence，每条 sequence 都需要一份 GDN 状态。

项目用：

```text
state_slot_id
```

表示某个 sequence 绑定到哪个状态槽。

对应代码在 `Sequence` 中：

```python
self.state_slot_id = -1
```

调度器中新增了：

```text
StateSlotManager
```

作用类似 KV cache 的 block allocator，只不过它分配的是 GDN state slot。

---

### 5.4 state_indices

`state_indices` 是 forward 时传给 GatedDeltaNet 的状态索引。

在 `utils/context.py` 中，Qwen3.6 fork 新增了：

```python
state_indices: torch.Tensor | None = None
```

在 decode 阶段：

```text
state_indices[i] = 第 i 条 sequence 的 state_slot_id
```

GatedDeltaNet 根据它找到该 sequence 的：

```text
conv_states[state_indices]
recurrent_states[state_indices]
```

---

### 5.5 MRoPE

原始 Qwen3 主要使用普通 RoPE。

Qwen3.5/Qwen3.6 fork 中新增：

```text
InterleavedMRoPE
```

MRoPE 可以理解为多维 rotary position embedding，在多模态场景下可以处理：

```text
temporal position
height position
width position
```

在 text-only 推理中，可以先把它理解成兼容 1D position 的 RoPE 扩展。

---

### 5.6 FP8 checkpoint 与 weight_scale_inv

Qwen3.6-FP8 checkpoint 里权重可能是：

```text
torch.float8_e4m3fn
```

并且会配套：

```text
xxx.weight_scale_inv
```

项目中的 `utils/quant.py` 做的是 block-wise dequant：

```text
FP8 weight * scale_inv -> BF16 weight
```

关键点：

```text
这个项目支持 FP8 checkpoint 加载，但最终常驻权重是 BF16。
```

---

### 5.7 MTP

MTP 可以理解为 Multi-Token Prediction，用于 speculative decoding 实验。

项目中的 MTP 当前做的是：

```text
1. 加载 MTP 权重
2. 做 single-step forward probe
3. 做 MTP-1 draft/verify
4. 做多 token draft/verify 原型
5. 测 accept_rate、verify overhead、greedy alignment
```

但它目前不是成熟加速路径。

你应该把它理解成：

```text
学习 speculative decoding 状态控制的实验模块
```

不要把它包装成生产级加速。

---

## 6. 第四阶段：学习 Qwen3.6 fork 的代码改动

### 6.1 `config.py` 的改动

原始 `Config` 主要包含：

```text
model
max_num_batched_tokens
max_num_seqs
max_model_len
gpu_memory_utilization
tensor_parallel_size
enforce_eager
hf_config
eos
kvcache_block_size
num_kvcache_blocks
```

Qwen3.6 fork 新增：

```text
full_config
is_hybrid
max_state_slots
enable_vision
enable_mtp
vision_config
image_token_id
vision_start_token_id
vision_end_token_id
```

关键逻辑是：

```python
self.full_config = AutoConfig.from_pretrained(self.model)

if hasattr(self.full_config, 'text_config'):
    self.hf_config = self.full_config.text_config
else:
    self.hf_config = self.full_config

self.is_hybrid = hasattr(self.hf_config, 'layer_types')
```

这说明：

```text
1. 对多模态模型，full_config 可能包含 vision_config 和 text_config。
2. 真正用于语言模型推理的是 text_config。
3. 如果 text_config 中存在 layer_types，就认为是 hybrid 模型。
```

学习重点：

```text
Config 不只是参数集合，它决定了后面 ModelRunner 创建哪个模型、是否分配 GDN state、是否启用 vision/MTP。
```

---

### 6.2 `model_runner.py` 的改动

这是 fork 中改动最大的文件。

原始 `model_runner.py` 大约 257 行；Qwen3.6 fork 扩展到 1500 多行。

新增或强化的能力包括：

```text
1. 自动选择 Qwen3 或 Qwen3_5 模型。
2. 加载 FP8 checkpoint 并记录 loaded/skipped names。
3. 分配 runtime buffers。
4. 只给 full attention 层分配 KV cache。
5. 为 GatedDeltaNet 分配 conv_states / recurrent_states。
6. 支持 state save / restore / drop。
7. 支持 multimodal prefill 张量准备。
8. 支持 hybrid decode 的 state_indices。
9. 支持 MTP draft/verify 相关 runner call。
10. 支持 verify CUDA Graph bucket。
```

#### 模型选择

```python
def _create_model(hf_config, vision_config=None):
    model_type = getattr(hf_config, 'model_type', '')
    if 'qwen3_5' in model_type:
        return Qwen3_5ForCausalLM(hf_config, vision_config=vision_config)
    else:
        return Qwen3ForCausalLM(hf_config)
```

含义：

```text
Qwen3 走原始 Qwen3ForCausalLM。
Qwen3.5/Qwen3.6 hybrid 走 Qwen3_5ForCausalLM。
```

---

### 6.3 KV cache 分配逻辑改动

原始 Nano-VLLM 可以认为所有层都有 KV cache。

但 hybrid 模型中不是所有层都是 full attention。

所以 fork 中改成：

```python
num_kv_layers = sum(
    1 for m in self.model.modules()
    if hasattr(m, "k_cache") and hasattr(m, "v_cache")
)
```

这表示：

```text
只统计真正有 k_cache/v_cache 的模块。
```

然后按 `num_kv_layers` 分配：

```python
self.kv_cache = torch.empty(
    2,
    num_kv_layers,
    config.num_kvcache_blocks,
    self.block_size,
    num_kv_heads,
    head_dim,
)
```

这个改动非常关键。

如果还按总层数分配 KV cache，会产生：

```text
1. 显存浪费
2. layer_id 与 full attention 层不对应
3. GDN 层没有 KV cache 却被分配 cache
```

面试表述：

```text
原始 Nano-VLLM 默认每层 attention 都需要 KV cache；Qwen3.5/Qwen3.6 hybrid 架构中只有 full attention 层需要 KV cache，因此我将 KV cache 分配从 num_hidden_layers 改为动态统计具有 k_cache/v_cache 的 attention 模块。
```

---

### 6.4 GDN state 分配逻辑

fork 中新增：

```python
def allocate_gdn_state(self):
    ...
```

它会遍历模型中的：

```text
GatedDeltaNet layers
```

然后为每个 GDN 层分配：

```text
conv_states:      [max_slots, conv_dim, kernel_size - 1]
recurrent_states: [max_slots, num_v_heads, head_k_dim, head_v_dim]
```

核心含义：

```text
max_slots 表示最多支持多少条并发 sequence 拥有自己的 GDN 状态。
```

与 KV cache 的区别：

```text
KV cache 是按 token block 增长的。
GDN state 是按 sequence slot 绑定的。
```

因此每条 sequence 在 scheduler 中需要：

```text
block_table    # attention 层 KV cache 物理 block 映射
state_slot_id  # GDN 层 recurrent/conv state 槽位
```

---

### 6.5 `scheduler.py` 的改动

fork 新增：

```python
class StateSlotManager:
    def __init__(self, num_slots: int):
        self.free_slots = deque(range(num_slots))

    def can_allocate(self) -> bool:
        return len(self.free_slots) > 0

    def allocate(self) -> int:
        return self.free_slots.popleft()

    def deallocate(self, slot_id: int):
        self.free_slots.append(slot_id)
```

调度时的逻辑变成：

```text
prefill 新请求时：
  1. 检查 KV block 是否够。
  2. 如果是 hybrid，检查 state slot 是否够。
  3. 分配 block_table。
  4. 分配 state_slot_id。
```

结束或 preempt 时：

```text
释放 KV block
释放 state slot
```

另外 fork 中对 hybrid 模型禁用 prefix cache：

```python
self.block_manager.allocate(seq, disable_prefix_cache=self.is_hybrid)
```

原因是：

```text
hybrid 模型不能只复用 attention 的 KV cache。
如果 prefix cache 复用了 KV，但 GDN recurrent/conv state 没有同步复用，状态就不一致。
```

所以当前实现选择保守禁用 hybrid prefix cache。

---

### 6.6 `sequence.py` 的改动

新增字段：

```python
self.state_slot_id = -1
self.pixel_values = None
self.image_grid_thw = None
```

其中最关键的是：

```text
state_slot_id
```

它的含义是：

```text
这条 sequence 当前绑定的 GDN state slot。
```

`-1` 表示还没有分配。

你要把它和 `block_table` 对比理解：

```text
block_table：管理 attention KV cache 位置。
state_slot_id：管理 GatedDeltaNet recurrent/conv state 位置。
```

---

### 6.7 `context.py` 的改动

fork 中 `Context` 新增：

```python
state_indices: torch.Tensor | None = None
```

作用：

```text
把当前 batch 中每条 sequence 的 state_slot_id 传给 GatedDeltaNet。
```

在 `prepare_decode()` 中：

```python
self.decode_cpu_state_indices[i] = seq.state_slot_id
...
set_context(..., state_indices=state_indices_t)
```

在 `GatedDeltaNet._forward_decode()` 中：

```python
state_indices = context.state_indices
conv_state = self.conv_states[state_indices]
rec_state = self.recurrent_states[state_indices]
```

这条链路必须完整掌握。

---

## 7. 第五阶段：学习 `models/qwen3_5.py`

`qwen3_5.py` 是新项目模型适配的核心。

学习顺序：

```text
Qwen3_5ForCausalLM
  -> Qwen3_5Model
  -> Qwen3_5DecoderLayer
  -> Qwen3_5Attention
  -> Qwen3_5MLP
```

---

### 7.1 Qwen3_5DecoderLayer

这是判断 hybrid layer 的入口：

```python
layer_type = config.layer_types[layer_idx]

if layer_type == "full_attention":
    self.self_attn = Qwen3_5Attention(config, layer_idx)
    self.linear_attn = None
else:
    self.linear_attn = GatedDeltaNet(config, layer_idx)
    self.self_attn = None
```

forward 时：

```python
if self.self_attn is not None:
    hidden_states = self.self_attn(positions, hidden_states)
else:
    hidden_states = self.linear_attn(hidden_states)
```

你要理解：

```text
同一个 DecoderLayer 抽象下，attention 子层可能是 full attention，也可能是 GatedDeltaNet。
```

---

### 7.2 Qwen3_5Attention

它相比原始 Qwen3Attention 有几个变化：

```text
1. q_proj 输出 2 倍，用于 query + gate。
2. 使用 GemmaRMSNorm。
3. 使用 InterleavedMRoPE。
4. attention 输出后乘 sigmoid(gate)。
```

核心代码逻辑：

```text
q_gate = q_proj(hidden_states)
q, gate = q_gate.chunk(2, dim=-1)

k = k_proj(hidden_states)
v = v_proj(hidden_states)

q = q_norm(q)
k = k_norm(k)
q, k = mrope(positions, q, k)

o = attention(q, k, v)
o = o * sigmoid(gate)
output = o_proj(o)
```

你要对比原始 Qwen3Attention：

```text
原始 Qwen3：qkv_proj 合并 Q/K/V。
Qwen3.5：q_proj/k_proj/v_proj 分开，并且 q_proj 包含 gate。
```

---

### 7.3 Qwen3_5ForCausalLM

它新增了：

```text
weight_prefix = "model.language_model."
visual_prefix = "model.visual."
self.mtp = Qwen3MTP(config) if enable_mtp
self.visual = Qwen3VLVisionEncoder(vision_config) if vision_config is not None
```

这说明它需要适配 checkpoint 中的命名：

```text
model.language_model.* -> model.*
model.visual.*         -> visual.*
```

所以它和 `utils/loader.py` 是强相关的。

---

## 8. 第六阶段：学习 `layers/gated_delta_net.py`

这是整个新项目最难、最有价值的文件。

建议你分三层学习，不要一口气读完。

---

### 8.1 第一层：先不看数学，只看状态和接口

先只看：

```text
class GatedDeltaNet
__init__
forward
_forward_prefill
_forward_decode
```

你要先回答：

```text
1. GatedDeltaNet 的输入输出 shape 是什么？
2. 它如何区分 prefill 和 decode？
3. 它在 decode 阶段从哪里拿 state_indices？
4. 它维护了哪些状态？
5. 它的状态在哪里分配？
```

核心结论：

```text
GatedDeltaNet.forward(hidden_states)
  -> if context.is_prefill: _forward_prefill(hidden_states)
  -> else: _forward_decode(hidden_states)
```

---

### 8.2 第二层：理解 conv state

GatedDeltaNet 里有 causal depthwise conv1d：

```text
causal_conv1d_prefill
causal_conv1d_decode
```

decode 阶段每次只有一个 token，所以需要保存过去 `kernel_size - 1` 个 token 的卷积状态：

```text
conv_states[state_indices]
```

这类似一个滑动窗口。

你可以这样理解：

```text
full attention 用 KV cache 保存历史 token 的 K/V。
GatedDeltaNet 的 causal conv 用 conv_state 保存卷积所需的历史局部窗口。
```

---

### 8.3 第三层：理解 recurrent state

GatedDeltaNet 的第二个状态是：

```text
recurrent_states
```

shape 大致是：

```text
[max_slots, num_v_heads, head_k_dim, head_v_dim]
```

decode 阶段调用：

```python
recurrent_gated_delta_rule(q, k, v, g, beta, rec_state)
```

它会：

```text
1. 根据当前 token 的 q/k/v/g/beta 更新 recurrent state。
2. 根据更新后的 state 计算当前 token 输出。
```

这个状态的本质是：

```text
linear attention 的递推记忆。
```

与 KV cache 最大区别：

```text
KV cache 保存所有历史 K/V，长度随 token 增长。
recurrent state 是固定大小状态，每步递推更新。
```

---

## 9. 第七阶段：学习 FP8 checkpoint 加载

相关文件：

```text
utils/quant.py
utils/loader.py
layers/linear.py
layers/gated_delta_net.py
```

学习顺序：

```text
quant.py
  -> loader.py
  -> linear.py 的 weight_loader
  -> gated_delta_net.py 的 qkv_weight_loader/head_weight_loader
```

---

### 9.1 `quant.py`

核心函数：

```python
def dequant_fp8_weight(weight, scale_inv, row_start=0, col_start=0, block_size=(128, 128)):
    ...
    return (weight.float() * scale.float()).to(torch.bfloat16)
```

它做的是：

```text
1. 根据当前 shard 的 row_start / col_start 找到对应 scale block。
2. 把 scale_inv 扩展到 weight 的二维 shape。
3. FP8 weight 转 float 后乘 scale。
4. 输出 BF16。
```

关键点：

```text
row_start / col_start 很重要，因为 TP rank 只加载完整矩阵的一部分。
```

---

### 9.2 `loader.py`

fork 里的 `load_model()` 支持：

```text
1. 读取 safetensors。
2. 跳过 .weight_scale_inv 本身。
3. 如果当前 weight 是 FP8 且存在 weight_scale_inv，则一并传给 weight_loader。
4. 处理 visual_prefix。
5. 处理 weight_prefix。
6. 处理 packed_modules_mapping。
7. 返回 LoadResult(loaded_names, skipped_names)。
```

核心逻辑：

```python
scale_name = weight_name + "_scale_inv"
loaded_scale = (
    f.get_tensor(scale_name)
    if loaded_weight.dtype == torch.float8_e4m3fn and scale_name in weight_names
    else None
)
```

然后：

```python
weight_loader(param, loaded_weight, loaded_scale)
```

或 packed module：

```python
weight_loader(param, loaded_weight, shard_id, loaded_scale)
```

---

### 9.3 `linear.py` 中的 rank-local dequant

例如 `ColumnParallelLinear.weight_loader()`：

```python
loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
loaded_weight = maybe_dequant_fp8_weight(
    loaded_weight, loaded_scale, row_start, col_start
)
param_data.copy_(loaded_weight)
```

顺序是：

```text
先按 TP rank 切片
再对当前 shard 做 dequant
再 copy 到 param
```

而不是：

```text
完整 FP8 weight 全部 dequant 成 BF16
再切片
```

这样做的价值：

```text
节省内存，更接近真实推理框架的 checkpoint loading 方式。
```

---

## 10. 第八阶段：学习 CUDA Graph decode 的变化

原始 Nano-VLLM 已经有 decode CUDA Graph。

fork 中的变化主要是：

```text
1. 预分配 runtime buffers。
2. decode graph 输入中增加 state_indices。
3. capture_cudagraph() 时 hybrid 模型需要 state_indices。
4. MTP verify 额外 capture verify-length buckets。
```

你要重点看：

```text
allocate_runtime_buffers()
prepare_decode()
run_model()
capture_cudagraph()
capture_verify_cudagraph()
capture_verify_chunk_cudagraph()
```

decode graph 的输入包括：

```text
input_ids
positions
slot_mapping
context_lens
block_tables
state_indices  # hybrid 新增
```

为什么要加 `state_indices`？

```text
因为 GatedDeltaNet decode 时必须知道当前 batch 每条 sequence 使用哪个 recurrent/conv state slot。
```

---

## 11. 第九阶段：最后学习 MTP / speculative decode

相关文件：

```text
models/qwen3_mtp.py
test_mtp_forward.py
test_mtp1_verify.py
test_mtp1_spec_decode.py
test_mtp_spec_decode.py
run_mtp_fast_decode.py
bench_mtp_draft_sweep.py
test_state_rollback.py
```

不要一开始就看 MTP。它应该放在最后。

推荐顺序：

```text
1. qwen3_mtp.py：理解 MTP 模型结构。
2. test_mtp_forward.py：只验证 MTP forward 能不能跑。
3. test_mtp1_verify.py：理解 draft token 和 target verify。
4. test_state_rollback.py：理解状态保存和回滚。
5. test_mtp1_spec_decode.py：理解 accept/reject。
6. test_mtp_spec_decode.py：理解多 token draft。
7. run_mtp_fast_decode.py：看 fast path 统计指标。
8. bench_mtp_draft_sweep.py：看 draft_len sweep。
```

---

### 11.1 MTP 的核心流程

MTP speculative decoding 的基本流程：

```text
1. 主模型生成当前 token。
2. MTP draft head 猜后续 token。
3. target model 验证 draft token。
4. 如果 draft token 和 target token 一致，accept。
5. 如果不一致，reject。
6. reject 时恢复 decode state，再提交 target token。
```

在普通 full attention 模型中，回滚主要是 KV cache。

在这个 hybrid 模型中，回滚必须同时恢复：

```text
KV cache
GDN conv_states
GDN recurrent_states
```

这就是 `save_decode_state()` / `restore_decode_state()` 的价值。

---

### 11.2 你要关注的指标

不要只看“速度有没有提升”。这个项目 README 已说明 MTP 目前是原型，不保证带来 decode speedup。

更合理的学习指标是：

```text
accept_rate
draft_len
greedy_match
target_forwards_per_token
mtp_forwards_per_token
verify_graph_replays
verify_eager_calls
verify_chunk_calls
reject_reruns
decode_tok_s
```

其中最重要的是：

```text
greedy_match == True
```

也就是 speculative decode 输出要和 greedy baseline 对齐。

如果输出不一致，速度再快也没有意义。

---

## 12. 你的完整学习顺序安排

下面是最推荐的实操学习路线。

---

### 第 1 步：复习原始 engine 与剩余模块接口

目标：建立原始 Nano-VLLM 总图。

读：

```text
nanovllm/engine/model_runner.py
nanovllm/engine/scheduler.py
nanovllm/engine/block_manager.py
nanovllm/engine/sequence.py
nanovllm/utils/context.py
```

产出：

```text
一张 prefill/decode 流程图：
Scheduler.schedule()
  -> ModelRunner.prepare_prefill/prepare_decode
  -> set_context
  -> model forward
  -> Attention.forward
  -> sampler
  -> Scheduler.postprocess
```

---

### 第 2 步：学习原始 Nano-VLLM 的 `layers/`

顺序：

```text
linear.py
embed_head.py
layernorm.py
activation.py
rotary_embedding.py
attention.py
sampler.py
```

你必须重点掌握：

```text
ColumnParallelLinear
RowParallelLinear
QKVParallelLinear
MergedColumnParallelLinear
VocabParallelEmbedding
ParallelLMHead
RMSNorm
RoPE
FlashAttention with KV cache
Sampler
```

产出：

```text
每个 layer 的输入 shape、输出 shape、是否 TP、是否通信。
```

---

### 第 3 步：完整串起 `models/qwen3.py`

顺序：

```text
Qwen3ForCausalLM
Qwen3Model
Qwen3DecoderLayer
Qwen3Attention
Qwen3MLP
```

产出：

```text
Qwen3 forward 数据流图。
```

你要能讲清楚：

```text
input_ids -> embedding -> decoder layers -> norm -> lm_head -> logits
```

以及：

```text
prefill 和 decode 的区别不在模型结构本身，而在 Attention.forward 读取的 Context 不同。
```

---

### 第 4 步：开始看 Qwen3.6 fork 的目录差异

先不要深入代码，先做宏观对比。

命令：

```bash
find nanovllm -type f | sort
```

重点记录新增文件：

```text
gated_delta_net.py
qwen3_5.py
qwen3_mtp.py
vision_encoder.py
image_processing.py
quant.py
```

产出：

```text
原始 Nano-VLLM 与 Qwen3.6 fork 文件对照表。
```

---

### 第 5 步：学习 `qwen3_5.py`

目标：理解 hybrid 模型结构。

顺序：

```text
Qwen3_5ForCausalLM
Qwen3_5Model
Qwen3_5DecoderLayer
Qwen3_5Attention
Qwen3_5MLP
```

重点问题：

```text
1. layer_types 如何决定每层走 full attention 还是 GatedDeltaNet？
2. Qwen3_5Attention 和 Qwen3Attention 有什么区别？
3. 为什么 q_proj 输出包含 gate？
4. 为什么使用 InterleavedMRoPE？
5. visual 和 mtp 是如何挂到 Qwen3_5ForCausalLM 上的？
```

---

### 第 6 步：学习 `gated_delta_net.py`

目标：先理解工程状态管理，再理解数学计算。

顺序：

```text
GatedDeltaNet.__init__
GatedDeltaNet.forward
_forward_prefill
_forward_decode
causal_conv1d_prefill / causal_conv1d_decode
chunk_gated_delta_rule
recurrent_gated_delta_rule
```

产出：

```text
一张 GatedDeltaNet prefill/decode 状态流图。
```

重点结论：

```text
GDN 层没有 KV cache。
GDN 层维护 conv_states 和 recurrent_states。
```

---

### 第 7 步：学习 runtime 状态改造

重点文件：

```text
sequence.py
scheduler.py
context.py
model_runner.py
```

顺序：

```text
Sequence.state_slot_id
  -> StateSlotManager
  -> Scheduler.schedule()
  -> allocate_gdn_state()
  -> prepare_prefill()/prepare_decode()
  -> set_context(... state_indices=...)
  -> GatedDeltaNet._forward_decode()
```

产出：

```text
一张 sequence 到 state slot 的绑定图。
```

最重要的理解：

```text
block_table 解决 full attention 的 KV cache 定位。
state_slot_id 解决 GatedDeltaNet 的 recurrent/conv state 定位。
```

---

### 第 8 步：学习 FP8 checkpoint loading

重点文件：

```text
utils/quant.py
utils/loader.py
layers/linear.py
layers/gated_delta_net.py
```

学习目标：

```text
1. FP8 weight 如何通过 weight_scale_inv 反量化？
2. 为什么 row_start/col_start 对 TP shard 很重要？
3. 为什么先切片再 dequant？
4. ColumnParallel / RowParallel / QKVParallel / MergedColumnParallel 如何分别加载 FP8 权重？
5. GatedDeltaNet 的 q/k/v 权重如何切分？
```

产出：

```text
一张 FP8 checkpoint -> TP shard -> BF16 param 的流程图。
```

---

### 第 9 步：学习 CUDA Graph decode 改造

重点文件：

```text
model_runner.py
```

重点函数：

```text
allocate_runtime_buffers
prepare_decode
run_model
capture_cudagraph
capture_verify_cudagraph
capture_verify_chunk_cudagraph
```

学习目标：

```text
1. CUDA Graph 为什么需要固定 shape 和固定 tensor 地址？
2. graph_vars 里保存了哪些 buffer？
3. decode 时为什么只 copy 数据再 graph.replay()？
4. hybrid 模型为什么 graph_vars 里要增加 state_indices？
5. verify graph bucket 1-4 是为 MTP verify 服务的。
```

---

### 第 10 步：最后学习 MTP 原型

重点文件：

```text
qwen3_mtp.py
test_mtp_forward.py
test_mtp1_verify.py
test_state_rollback.py
test_mtp1_spec_decode.py
test_mtp_spec_decode.py
run_mtp_fast_decode.py
bench_mtp_draft_sweep.py
```

学习目标：

```text
1. MTP draft token 是怎么产生的？
2. target verify 怎么做？
3. accept/reject 逻辑怎么判断？
4. reject 时为什么要 restore_decode_state？
5. 为什么 hybrid 模型回滚比普通 transformer 更复杂？
6. 为什么当前 MTP 不一定能带来速度提升？
```

产出：

```text
一张 speculative decoding 状态机图。
```

---

## 13. 建议你最终整理出的四张图

为了真正学会这个项目，建议你最终画出四张图。

### 图 1：原始 Nano-VLLM 推理链路图

```text
LLM.generate
  -> LLMEngine.add_request
  -> Scheduler.schedule
  -> ModelRunner.run
  -> prepare_prefill / prepare_decode
  -> Qwen3ForCausalLM.forward
  -> Attention.forward
  -> sampler
  -> Scheduler.postprocess
```

---

### 图 2：Qwen3 vs Qwen3.5/Qwen3.6 模型结构对比图

```text
Qwen3:
  Every layer = full attention + MLP

Qwen3.5/Qwen3.6 hybrid:
  Some layers = full attention + MLP
  Other layers = GatedDeltaNet + MLP
```

---

### 图 3：KV cache 与 GDN state 双状态管理图

```text
Sequence
  -> block_table
      -> KV cache blocks
      -> full attention layers

  -> state_slot_id
      -> conv_states[state_slot_id]
      -> recurrent_states[state_slot_id]
      -> GatedDeltaNet layers
```

---

### 图 4：MTP speculative decode 回滚图

```text
save_decode_state
  -> draft tokens
  -> target verify
      -> accept: commit draft tokens
      -> reject: restore KV cache + restore GDN states + commit target token
```

---

## 14. 你应该重点写进学习笔记的对比表

| 模块 | 原始 Nano-VLLM | Qwen3.6 fork |
|---|---|---|
| 模型 | Qwen3 dense text | Qwen3 + Qwen3.5/Qwen3.6 hybrid |
| Attention | 每层 full attention | full attention 与 GatedDeltaNet 混合 |
| KV cache | 默认每层 attention 都需要 | 只给 full attention 层分配 |
| 额外状态 | 无 | GDN conv_states + recurrent_states |
| Sequence | block_table | block_table + state_slot_id |
| Scheduler | 管理 KV block | 管理 KV block + state slot |
| Context | attention runtime 信息 | 新增 state_indices |
| ModelRunner | prefill/decode/CUDA Graph | 新增 GDN state、FP8、vision、MTP、verify graph |
| Loader | 普通 safetensors 加载 | FP8 weight + scale_inv 反量化加载 |
| CUDA Graph | decode graph | decode graph + state_indices + verify graph buckets |
| MTP | 无 | draft/verify/rollback 原型 |
| Vision | 无 | Qwen3.5 local vision encoder smoke test |
| 生产程度 | 学习型小推理引擎 | 研究/学习型 hybrid 推理实验，不是生产 serving |

---

## 15. 实操复现建议

### 15.1 没有 4 张 4090 的情况下

如果你没有 4 张 RTX 4090，不建议一开始就跑 Qwen3.6-27B-FP8。

你可以先做：

```text
1. python -m compileall nanovllm examples ...
2. 跑原始 Qwen3 dense 小模型路径。
3. 用 fake tensor 单独测试 FP8 dequant 逻辑。
4. 阅读并注释 qwen3_5.py / gated_delta_net.py。
5. 画出状态管理图。
6. 只分析 MTP 脚本流程，不强求完整跑通。
```

---

### 15.2 有多卡环境的情况下

如果你有多卡环境，可以按 README 的顺序跑：

```bash
python run_text_qwen35_v2.py \
  --model ~/huggingface/Qwen3.5-9B \
  --tp 1
```

再跑：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_text_qwen35_v2.py \
  --model ~/huggingface/Qwen3.5-9B \
  --tp 4
```

最后再尝试：

```bash
python run_text_qwen36_fp8.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --tp 4
```

再按顺序测试：

```bash
python test_mtp_forward.py
python test_mtp1_verify.py
python test_state_rollback.py
python test_mtp1_spec_decode.py
python run_mtp_fast_decode.py
python bench_mtp_draft_sweep.py
```

---

## 16. 面试视角：学到什么程度才算真正掌握

你至少要能回答下面这些问题。

### 16.1 原始 Nano-VLLM 相关

```text
1. prefill 和 decode 在 ModelRunner 里有什么区别？
2. slot_mapping 是什么？
3. block_table 是什么？
4. KV cache 的物理布局是什么？
5. ColumnParallelLinear 和 RowParallelLinear 有什么区别？
6. QKVParallelLinear 如何把 q/k/v 权重合并？
7. FlashAttention 在 prefill 和 decode 中分别怎么调用？
8. Sampler 如何从 logits 采样 token？
```

---

### 16.2 Qwen3.5/Qwen3.6 fork 相关

```text
1. Qwen3 和 Qwen3.5/Qwen3.6 最大区别是什么？
2. 什么是 hybrid model？
3. 为什么 KV cache 不能按总层数分配？
4. GatedDeltaNet 为什么需要 conv_state 和 recurrent_state？
5. state_slot_id 和 block_table 分别解决什么问题？
6. state_indices 为什么要进入 Context？
7. hybrid 模型为什么要谨慎使用 prefix cache？
8. Qwen3.6-FP8 checkpoint 加载做了什么？
9. 当前项目有没有实现 native FP8 matmul？
10. MTP reject 时为什么要回滚状态？
11. 为什么 hybrid 模型的 speculative decode 回滚更复杂？
12. CUDA Graph decode 为什么要新增 state_indices？
```

---

## 17. 推荐的最终项目包装方式

如果你后面要把它写进简历，不建议写成：

```text
实现 Qwen3.6 FP8 高性能推理系统
```

这个说法太大，而且不准确。

更准确的写法是：

```text
基于 Nano-VLLM 扩展 Qwen3.5/Qwen3.6 hybrid 推理路径，支持 full attention 与 GatedDeltaNet 混合层；重构 KV cache 分配逻辑，仅为 full attention 层分配 cache，并新增 GDN recurrent/conv state slot 管理；实现 FP8 checkpoint 的 TP rank-local BF16 反量化加载；实现 MTP draft/verify/rollback 原型，并通过 greedy alignment、accept rate、verify graph replay 等指标验证推理状态一致性。
```

这个描述更符合项目真实情况，也更容易通过面试追问。

---

## 18. 最终学习路线总结

你现在最应该做的是：

```text
先补齐原始 Nano-VLLM 的 layers/models/utils，尤其是 linear、attention、loader、qwen3。
```

然后再进入：

```text
qwen3_5.py + gated_delta_net.py
```

接着理解：

```text
Sequence.state_slot_id
StateSlotManager
ModelRunner.allocate_gdn_state
Context.state_indices
```

最后再看：

```text
FP8 checkpoint loading
CUDA Graph hybrid decode
MTP speculative decode
```

一句话总结：

```text
原始 Nano-VLLM 解决的是 full attention 模型的 KV cache 推理；
nano-vllm-qwen3.6 进一步解决 hybrid 模型下 KV cache 与 recurrent/conv state 并存时的模型适配、状态管理、权重加载和实验性 speculative decode 问题。
```



%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
