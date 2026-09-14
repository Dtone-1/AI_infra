# qwen3.6_model_runner.py_对比分析

> 对比对象：  
> 1. 原版 `nano-vLLM` 的 `model_runner.py`  
> 2. `nano-vllm-qwen3.6` 中对应的 `model_runner.py`
>
> 分析目标：从整体工程角度理解 qwen3.6 版本相对原版做了哪些核心改造，以及这些改造在 Qwen3.6 / hybrid 架构 / GatedDeltaNet / 多模态 / CUDA Graph / 推理执行流程中的作用。

---

## 1. 文件整体定位

`model_runner.py` 是 nano-vLLM 推理系统中真正负责 **执行模型 forward** 的核心文件。

如果把前面几个文件串起来：

```text
llm_engine.py      负责用户入口和 generate 主循环
sequence.py        保存每个请求的状态
scheduler.py       决定本轮跑哪些请求
block_manager.py   管理 KV Cache block
model_runner.py    准备张量、设置 Context、调用模型、采样输出
```

那么 `ModelRunner` 就是：

```text
Scheduler 和模型 forward 之间的执行中枢
```

它的职责包括：

```text
1. 初始化模型
2. 加载权重
3. 初始化 tensor parallel 分布式环境
4. warmup 模型
5. 分配 KV Cache
6. 准备 prefill 输入张量
7. 准备 decode 输入张量
8. 设置 Context
9. 调用模型 forward
10. 计算 logits
11. 采样 next token
12. 捕获和 replay CUDA Graph
13. 多进程 rank 间通信
```

原版 `model_runner.py` 主要服务于普通 Qwen3 dense / decoder-only attention 模型。

qwen3.6 版本则把它扩展成了一个更复杂的执行层，开始支持：

```text
Qwen3.5 / hybrid 模型自动创建
GatedDeltaNet recurrent/conv state 管理
只给 full attention 层分配 KV Cache
多模态图像输入
MRoPE 3D position
多卡视觉张量广播
decode runtime buffer 预分配
更复杂的 CUDA Graph
验证探针 / logits 对比
MTP draft token 探针
```

---

## 2. qwen3.6 版本整体变化概览

相比原版，qwen3.6 版本 `model_runner.py` 的变化可以分为九大类。

| 类别 | 原版 nano-vLLM | qwen3.6 版本 | 核心意义 |
|---|---|---|---|
| 模型创建 | 固定 `Qwen3ForCausalLM` | `_create_model()` 自动选择 Qwen3 / Qwen3_5 | 支持 hybrid 模型族 |
| 权重加载 | `load_model()` 无返回值 | 保存 `load_result`，检查 MTP 权重 | 支持扩展模块权重校验 |
| KV Cache 分配 | 按总层数分配 | 只统计有 `k_cache/v_cache` 的 attention 层 | 适配 hybrid：不是每层都有 KV Cache |
| GDN state | 无 | `allocate_gdn_state()` 分配 conv/recurrent state 池 | 支持 GatedDeltaNet 层 decode 状态 |
| Context 输入 | 只有 KV 相关字段 | 增加 `state_indices` | 把请求映射到 GDN state slot |
| 多模态 | 无 | `pixel_values / image_grid_thw / image_token_mask / MRoPE` | 支持图文输入 |
| Decode 准备 | 每轮新建 tensor | 预分配 CPU pinned + GPU buffer | 降低 decode 阶段小张量开销 |
| CUDA Graph | 只捕获普通 decode | 捕获普通 decode + verify graph + chunk graph，并带 state_indices | 支持 hybrid state 安全 replay |
| 调试/验证/MTP | 无 | decode state snapshot、topk、logit compare、verify、MTP draft | 支持正确性验证和投机/多 token 预测实验 |

一句话总结：

**原版 `ModelRunner` 是“普通 Qwen3 dense 文本推理执行器”；qwen3.6 版本把它改造成了“同时支持 hybrid recurrent state、多模态输入、CUDA Graph state replay、验证探针和 MTP 草稿预测的复杂执行器”。**

---

## 3. 改动一：模型创建从固定 Qwen3 改为自动识别模型类型

### 3.1 原版写法

原版直接导入并创建：

```python
from nanovllm.models.qwen3 import Qwen3ForCausalLM

self.model = Qwen3ForCausalLM(hf_config)
load_model(self.model, config.model)
```

这说明原版 `ModelRunner` 默认只支持一种模型主体：

```text
Qwen3 dense / Qwen3ForCausalLM
```

---

### 3.2 qwen3.6 版本写法

qwen3.6 版本新增：

```python
def _create_model(hf_config, vision_config=None):
    model_type = getattr(hf_config, 'model_type', '')
    if 'qwen3_5' in model_type:
        from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
        return Qwen3_5ForCausalLM(hf_config, vision_config=vision_config)
    else:
        from nanovllm.models.qwen3 import Qwen3ForCausalLM
        return Qwen3ForCausalLM(hf_config)
```

初始化时改成：

```python
self.model = _create_model(hf_config, vision_config=config.vision_config)
```

---

### 3.3 改动意义

这个改动说明：

```text
ModelRunner 不再假设模型一定是 Qwen3 dense。
```

它开始根据 `hf_config.model_type` 自动选择模型类。

这对 Qwen3.5 / Qwen3.6 很重要，因为从 Qwen3 dense 到 Qwen3.5/3.6 hybrid，不只是模型名字变了，而是模型结构发生了变化：

```text
Qwen3 dense:
    full attention + MLP

Qwen3.5 / Qwen3.6 hybrid:
    GatedDeltaNet / Gated Attention / MoE 或 FFN 混合
```

因此执行层不能再硬编码：

```text
一定创建 Qwen3ForCausalLM
```

而要让模型创建逻辑根据配置分流。

---

## 4. 改动二：初始化流程新增日志、torch._dynamo 配置、load_result 和 MTP 权重检查

### 4.1 新增 `torch._dynamo`

qwen3.6 版本新增：

```python
import torch._dynamo
torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
```

这说明 qwen3.6 版本可能使用了更多 `torch.compile` 或更复杂的动态图编译路径。

当模型结构、verify 路径、MTP 路径增多后，Dynamo 需要缓存更多编译图，否则可能频繁触发重编译或缓存溢出。

---

### 4.2 新增 `_log()`

qwen3.6 版本新增：

```python
def _log(self, message: str):
    if self.rank == 0:
        print(f"[ModelRunner rank 0] {message}", flush=True)
```

初始化过程中会打印：

```text
creating model
allocating runtime buffers
warming up model
allocating kv cache
allocating gated delta state
capturing cuda graphs
model runner ready
```

这个改动不是模型结构改造，但对调试复杂推理系统很重要。

因为 qwen3.6 的初始化明显更复杂：

```text
模型创建
权重加载
runtime buffer
warmup
KV Cache
GDN state
CUDA Graph
verify graph
```

如果中间出错，日志能定位卡在哪一步。

---

### 4.3 load_model 返回值被保存

原版：

```python
load_model(self.model, config.model)
```

qwen3.6：

```python
self.load_result = load_model(self.model, config.model, log_fn=self._log)
```

这说明 qwen3.6 版本的权重加载不只是“加载完即可”，还要记录：

```text
哪些权重 loaded
哪些权重 skipped
```

这对扩展模型很重要，因为新增模块可能包括：

```text
GDN 权重
vision 权重
MTP 权重
MoE 权重
```

如果某些权重被跳过，模型可能能初始化但结果错误。

---

### 4.4 MTP 权重检查

qwen3.6 版本新增：

```python
if config.enable_mtp:
    mtp_skipped = [name for name in self.load_result.skipped_names if name.startswith("mtp.")]
    mtp_loaded = [name for name in self.load_result.loaded_names if name.startswith("mtp.")]
    assert not mtp_skipped
```

这说明该版本支持一个叫 MTP 的扩展模块。

从后续代码看，MTP 用于根据当前 hidden state 和 token embedding 生成 draft token，类似多 token 预测或投机解码相关实验路径。

这个检查的作用是：

```text
只要 enable_mtp=True，就必须确保 MTP 权重没有被漏加载。
```

否则后面 MTP draft 输出没有意义。

---

## 5. 改动三：KV Cache 分配从“所有层”改为“只给 attention 层”

### 5.1 原版 KV Cache 分配

原版：

```python
block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, ...)
```

这表示原版默认：

```text
每一层都有 attention
每一层都需要 k_cache / v_cache
```

这对于 Qwen3 dense 是合理的。

---

### 5.2 qwen3.6 版本 KV Cache 分配

qwen3.6 版本改成：

```python
num_kv_layers = sum(1 for m in self.model.modules() if hasattr(m, "k_cache") and hasattr(m, "v_cache"))
block_bytes = 2 * num_kv_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
self.kv_cache = torch.empty(2, num_kv_layers, ...)
```

也就是说：

```text
不是按 hf_config.num_hidden_layers 分配 KV Cache
而是扫描模型模块，只统计真正有 k_cache/v_cache 的层
```

---

### 5.3 为什么这是 hybrid 架构的关键改动

在 Qwen3.5 / Qwen3.6 hybrid 架构中，不是每一层都是 full attention。

可能存在：

```text
GatedDeltaNet 层
Gated Attention 层
MoE/FFN 层
```

其中只有 full attention 层需要 KV Cache。

如果仍然按照总层数分配：

```text
num_hidden_layers
```

会产生两个问题：

1. 给不需要 KV Cache 的 GDN 层浪费显存；
2. KV Cache layer index 和真实 attention 层对应关系可能错位。

qwen3.6 改成扫描 `hasattr(k_cache, v_cache)` 后，语义变成：

```text
只有真正实现 KV Cache 接口的模块，才被分配 KV Cache。
```

这就是支持 hybrid 模型的基础。

---

## 6. 改动四：新增 GatedDeltaNet state 池分配

### 6.1 原版没有 GDN state

原版只有：

```text
KV Cache
```

没有任何 recurrent state 或 conv state。

这是因为普通 decoder-only Transformer 的 decode 历史状态主要是 KV Cache。

---

### 6.2 qwen3.6 新增 `allocate_gdn_state()`

qwen3.6 版本新增：

```python
def allocate_gdn_state(self):
    from nanovllm.layers.gated_delta_net import GatedDeltaNet
    gdn_layers = [m for m in self.model.modules() if isinstance(m, GatedDeltaNet)]
```

然后为每个 GDN 层分配：

```python
layer.conv_states = torch.zeros(max_slots, conv_dim, kernel_size - 1, ...)
layer.recurrent_states = torch.zeros(max_slots, num_v_heads, head_k_dim, head_v_dim, dtype=torch.float32)
```

---

### 6.3 GDN state 是什么

对于普通 attention：

```text
历史信息 = 每个历史 token 的 K/V
```

对于 GatedDeltaNet：

```text
历史信息 = 随时间递推更新的 recurrent state + conv state
```

所以 hybrid 模型 decode 时，不仅要知道：

```text
这个请求的 KV Cache 在哪些 block
```

还要知道：

```text
这个请求的 GDN recurrent state 在哪个 slot
```

前面 `Sequence.state_slot_id` 和 `Scheduler.StateSlotManager` 解决的是“请求分配哪个 slot”。

这里 `ModelRunner.allocate_gdn_state()` 解决的是：

```text
GPU 上真正的 state tensor 池在哪里。
```

---

### 6.4 为什么 recurrent_states 用 float32

代码里：

```python
layer.recurrent_states = torch.zeros(..., dtype=torch.float32)
```

即使模型权重可能是 FP16/BF16，recurrent state 仍然用 FP32。

这通常是为了数值稳定性。

因为 recurrent state 会在 decode 中不断递推更新，如果用低精度长期累积，误差可能更明显。

---

### 6.5 max_state_slots 如何确定

qwen3.6 版本根据剩余 GPU 显存计算：

```python
max_slots = int(free * 0.9) // bytes_per_slot
max_slots = min(max_slots, config.max_num_seqs)
config.max_state_slots = max_slots
```

这说明 state slot 数不是固定写死的，而是受 GPU 剩余显存和最大并发请求数约束。

它和 scheduler 中的 `StateSlotManager(config.max_state_slots)` 对应：

```text
ModelRunner 负责分配真实 GPU state 池
Scheduler 负责给每个 Sequence 分配 slot id
Sequence 负责保存自己的 state_slot_id
Context 负责把 state_indices 传给模型层
```

这是一条完整链路。

---

## 7. 改动五：新增 GDN state 重置、保存和恢复

### 7.1 `reset_gdn_state_slots`

qwen3.6 新增：

```python
def reset_gdn_state_slots(self, slot_ids: list[int]):
    ...
    layer.conv_states.index_fill_(0, indices, 0)
    layer.recurrent_states.index_fill_(0, indices, 0)
```

作用是：

```text
把指定请求槽位的 GDN state 清零。
```

这在以下场景重要：

```text
CUDA Graph 捕获后恢复干净状态
请求结束后复用 slot
验证/探针前后避免状态污染
```

---

### 7.2 decode state snapshot

qwen3.6 新增：

```python
save_decode_state
save_decode_state_range
restore_decode_state
drop_decode_state
```

这些函数会保存：

```text
1. 当前 decode 相关 KV slots
2. 当前 seq 的 state_slot_ids
3. KV Cache 中对应位置的内容
4. GDN 层 conv/recurrent states
```

然后可以恢复。

---

### 7.3 为什么需要 snapshot

这不是普通在线推理的最小必需功能，更像是为了：

```text
验证 speculative / MTP / verify 路径
对比 logits
回滚 decode 状态
测试某个输入 token 序列对模型输出的影响
```

对于普通 attention 模型，保存 KV Cache 部分就够了。

但 hybrid 模型必须同时保存：

```text
KV Cache + GDN state
```

否则恢复后 attention 状态恢复了，但 GDN recurrent 状态没有恢复，输出仍然会变。

---

## 8. 改动六：新增 runtime buffer 预分配

### 8.1 原版 decode 准备方式

原版 `prepare_decode()` 每轮都会构造 Python list，然后转成 tensor：

```python
input_ids = []
positions = []
slot_mapping = []
context_lens = []
...
torch.tensor(..., pin_memory=True).cuda(non_blocking=True)
```

这种写法简单，但 decode 阶段每轮都执行，容易有小张量分配和 CPU->GPU 拷贝开销。

---

### 8.2 qwen3.6 版本预分配 buffer

qwen3.6 新增：

```python
self.decode_cpu_input_ids
self.decode_cpu_positions
self.decode_cpu_slot_mapping
self.decode_cpu_context_lens
self.decode_cpu_block_tables
self.decode_gpu_input_ids
self.decode_gpu_positions
self.decode_gpu_slot_mapping
self.decode_gpu_context_lens
self.decode_gpu_block_tables
self.sample_cpu_temperatures
self.sample_gpu_temperatures
```

hybrid 时还新增：

```python
self.decode_cpu_state_indices
self.decode_gpu_state_indices
```

---

### 8.3 改动意义

decode 阶段每次只生成一个 token，但会重复很多轮。

如果每轮都新建很多小 tensor，会引入：

```text
CPU 分配开销
pin memory 开销
GPU allocator 开销
kernel launch 前准备开销
```

qwen3.6 版本改成预分配后，每轮只做：

```text
填 CPU pinned buffer
copy 到 GPU buffer slice
set_context
```

这更接近高性能推理框架的做法。

对 hybrid 模型来说，还需要每轮把：

```text
state_indices
```

一并拷到 GPU，告诉 GDN 层使用哪个 state slot。

---

## 9. 改动七：prefill 输入准备支持 hybrid state 和多模态

### 9.1 原版 `prepare_prefill`

原版只准备文本 token：

```text
input_ids
positions
cu_seqlens_q
cu_seqlens_k
slot_mapping
block_tables
```

返回：

```python
return input_ids, positions
```

---

### 9.2 qwen3.6 版本新增 state_indices

qwen3.6 版本在 prefill 中新增：

```python
state_indices = []
state_indices.append(seq.state_slot_id)
...
state_indices_t = torch.tensor(state_indices, ...)
set_context(..., state_indices=state_indices_t)
```

这表示 prefill 阶段不仅要写 KV Cache，还要让 GDN 层知道：

```text
当前 batch 中每个 sequence 对应哪个 recurrent state slot。
```

---

### 9.3 qwen3.6 版本调整 start / seqlen_k

原版：

```python
start = seq.num_cached_tokens
seqlen_k = end
```

qwen3.6：

```python
seqlen = len(seq)
start = min(seq.num_cached_tokens, seqlen - 1)
seqlen_k = seqlen
```

这个改动反映了更复杂的缓存和 re-prefill 逻辑。

在 hybrid、preemption、chunk verify 或 prefix 相关场景中，当前要计算的 query token 范围和 key 的总上下文长度可能不完全等价于 `end`。

qwen3.6 用完整 `seqlen` 作为 `seqlen_k`，更强调：

```text
当前 query 要面对的 key 上下文长度是整个序列长度。
```

---

### 9.4 多模态数据收集

qwen3.6 新增：

```python
has_images = any(seq.pixel_values is not None for seq in seqs)
pixel_values_list = []
image_grid_thw_list = []
all_positions_3d = []
image_token_mask_parts = []
```

最终返回：

```python
return input_ids, positions, pixel_values, image_grid_thw, image_token_mask
```

这说明 `ModelRunner` 不再只处理文本 token，还要处理图像张量。

---

## 10. 改动八：新增 MRoPE 3D position 计算

### 10.1 原版 position

原版 position 很简单：

```python
positions.extend(range(start, end))
```

也就是普通一维位置：

```text
0, 1, 2, 3, ...
```

---

### 10.2 qwen3.6 版本 `_compute_mrope_positions`

qwen3.6 新增：

```python
def _compute_mrope_positions(self, token_ids, image_grid_thw):
    positions = torch.zeros(3, n, dtype=torch.long)
```

返回的是：

```text
(3, seq_len)
```

也就是：

```text
temporal position
height position
width position
```

这通常对应多模态 RoPE / MRoPE 的位置编码需求。

---

### 10.3 为什么图像需要 3D position

文本 token 是一维序列：

```text
第 0 个 token
第 1 个 token
第 2 个 token
```

但图像 patch 本来具有二维空间结构：

```text
第 h 行，第 w 列
```

如果是视频，还可能有时间维：

```text
第 t 帧，第 h 行，第 w 列
```

所以多模态模型需要让视觉 token 带有：

```text
时间 + 高度 + 宽度
```

qwen3.6 版本用 `image_grid_thw` 和 `spatial_merge_size` 计算这些 position。

---

### 10.4 mixed batch 兼容

如果一个 batch 里有图像请求，也有纯文本请求，qwen3.6 会把纯文本 position 也扩展成 3D：

```python
pos_1d.unsqueeze(0).expand(3, -1)
```

这样整个 batch 的 positions 形状统一。

这属于多模态 batching 的必要兼容处理。

---

## 11. 改动九：多卡 tensor parallel 下广播图像数据

### 11.1 原版没有图像广播

原版只需要广播方法调用和 `Sequence` 状态。

文本 token、positions、block_tables 等每个 rank 都可以根据传入 seqs 构造。

---

### 11.2 qwen3.6 新增 `_broadcast_image_data`

qwen3.6 在 `run()` 中：

```python
if self.world_size > 1:
    pixel_values, image_grid_thw, image_token_mask, positions = \
        self._broadcast_image_data(...)
```

这个函数会广播：

```text
是否有图像
positions 是否是 3D
positions shape
pixel_values shape 和 dtype
pixel_values
image_grid_thw
image_token_mask
```

---

### 11.3 为什么需要广播

在多进程 tensor parallel 中，rank 0 通常持有完整的 Python 请求对象和多模态数据。

其他 rank 也要执行模型 forward 的一部分权重。

如果模型里包含视觉编码器或视觉 token 替换逻辑，那么每个 TP rank 都需要同样的：

```text
pixel_values
image_grid_thw
image_token_mask
positions
```

否则各 rank 的 forward 输入不一致，tensor parallel 结果会错误。

因此 qwen3.6 新增图像数据广播，是多模态 + tensor parallel 同时存在时必须做的工程改造。

---

## 12. 改动十：decode 输入准备从动态 tensor 构造改为预分配 buffer + state_indices

### 12.1 原版 decode

原版 decode 每轮：

```python
input_ids.append(seq.last_token)
positions.append(len(seq) - 1)
context_lens.append(len(seq))
slot_mapping.append(...)
block_tables = self.prepare_block_tables(seqs)
set_context(False, ...)
```

---

### 12.2 qwen3.6 decode

qwen3.6 版本：

```python
self.decode_cpu_input_ids[i] = seq.last_token
self.decode_cpu_positions[i] = len(seq) - 1
self.decode_cpu_context_lens[i] = len(seq)
self.decode_cpu_slot_mapping[i] = ...
if self.config.is_hybrid:
    self.decode_cpu_state_indices[i] = seq.state_slot_id
```

然后拷贝到 GPU buffer：

```python
input_ids.copy_(...)
positions.copy_(...)
slot_mapping.copy_(...)
context_lens.copy_(...)
block_tables.copy_(...)
state_indices_t.copy_(...)
```

最后：

```python
set_context(False, ..., state_indices=state_indices_t)
```

---

### 12.3 工程意义

这个改动同时解决两类问题：

第一，性能：

```text
减少 decode 阶段反复分配小 tensor 的开销。
```

第二，hybrid state：

```text
decode 每个请求都必须携带 state_slot_id，让 GDN 层在对应 slot 上读写 recurrent/conv state。
```

---

## 13. 改动十一：采样逻辑增强，支持 greedy、temperature、TP all-gather 和 top-k

### 13.1 原版采样

原版：

```python
temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
```

采样逻辑比较简单，且主要在 rank 0 上处理。

---

### 13.2 qwen3.6 版本采样

qwen3.6 新增：

```python
def sample(self, logits, temperatures, greedy):
    if greedy:
        token_ids, scores = self.sampler.greedy_with_scores(logits)
    else:
        token_ids, scores = self.sampler.forward_with_scores(logits, temperatures)
```

并在 TP 下做：

```python
dist.all_gather(all_scores, scores)
dist.all_gather(all_token_ids, token_ids)
rank_ids = scores.argmax(...)
```

还新增：

```python
topk_tokens()
```

---

### 13.3 改动意义

这说明 qwen3.6 的采样模块不只是为了正常 generate，还要服务于：

```text
logit probe
verify probe
MTP draft top-k 检查
跨 TP vocab partition 的全局最优 token 选择
```

在 tensor parallel 的 vocab parallel LM Head 下，每个 rank 只持有词表的一部分 logits。

如果要做 greedy 或 top-k，需要把各 rank 的候选 token 和 score 聚合起来，才能得到全局结果。

---

## 14. 改动十二：run_model 支持多模态参数和 hybrid CUDA Graph state_indices

### 14.1 原版 run_model

原版：

```python
return self.model.compute_logits(self.model(input_ids, positions))
```

decode graph replay 时只填：

```text
input_ids
positions
slot_mapping
context_lens
block_tables
```

---

### 14.2 qwen3.6 run_model

qwen3.6：

```python
self.model(
    input_ids,
    positions,
    pixel_values=pixel_values,
    image_grid_thw=image_grid_thw,
    image_token_mask=image_token_mask
)
```

decode CUDA Graph replay 时还会填：

```python
if self.config.is_hybrid:
    graph_vars["state_indices"][:bs] = context.state_indices
```

---

### 14.3 改动意义

这个函数是模型 forward 的最后入口。

qwen3.6 版本在这里把两条新能力接进模型：

```text
多模态输入 -> pixel_values / image_grid_thw / image_token_mask
hybrid state -> state_indices
```

同时要保证 CUDA Graph replay 时，这些运行时变量也能被更新。

否则 eager 模式能跑，但 graph replay 模式会因为 state_indices 仍是捕获时的旧值，导致 GDN state 读写错槽位。

---

## 15. 改动十三：run() 主流程增加多模态广播、greedy 判断和统一 sample

### 15.1 原版 run

原版：

```python
input_ids, positions = self.prepare_prefill(...) if is_prefill else self.prepare_decode(...)
temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
logits = self.run_model(input_ids, positions, is_prefill)
token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
reset_context()
return token_ids
```

---

### 15.2 qwen3.6 run

qwen3.6：

```python
if is_prefill:
    input_ids, positions, pixel_values, image_grid_thw, image_token_mask = self.prepare_prefill(seqs)
    if self.world_size > 1:
        self._broadcast_image_data(...)
else:
    input_ids, positions = self.prepare_decode(seqs)
    pixel_values = image_grid_thw = image_token_mask = None

greedy = all(seq.temperature <= 1e-10 for seq in seqs)
temperatures = None if greedy else self.prepare_sample(seqs)
logits = self.run_model(..., pixel_values=..., image_grid_thw=..., image_token_mask=...)
token_ids = self.sample(logits, temperatures, greedy)
```

---

### 15.3 改动意义

`run()` 从原来的简单文本路径，变成了统一调度：

```text
prefill:
    可能有图像
    可能有 3D positions
    可能需要 TP 广播图像张量

decode:
    无图像输入
    需要 KV Cache + GDN state

sample:
    greedy 和 sampling 分流
    TP 下聚合 score/token
```

这说明 qwen3.6 的执行层已经不仅是一个最小 demo，而是开始具备复杂推理框架的形态。

---

## 16. 改动十四：新增 probe / verify / logits 对比功能

qwen3.6 新增了大量调试和验证函数：

```text
run_step_probe
run_verify_batch_probe
run_verify_batch_fast
run_verify_auto_probe
run_verify_chunk_probe
run_verify_chunk_fast
_save_decode_state_slots
restore_decode_state
topk_tokens
_compare_logits
```

这些功能主要用于：

```text
1. 查看某一步输出 token
2. 查看 top-k token 和 score
3. 保存某次 logits 作为参考
4. 比较两次 logits 最大差异
5. 验证一段候选 token 序列
6. 在 eager / graph / chunk 模式之间对比
7. 支持 speculative / MTP 路径的正确性调试
```

---

### 16.1 为什么 hybrid 下验证更复杂

普通 attention-only 模型验证一段 token，主要要保证：

```text
KV Cache 正确
positions 正确
block_tables 正确
```

hybrid 模型还要保证：

```text
GDN recurrent state 正确
GDN conv state 正确
state_indices 正确
```

因此 qwen3.6 的 verify graph / chunk graph 都要把 `state_indices` 接进去。

如果不这样做，验证路径和真实 decode 路径就不是同一个状态逻辑，logits 对比没有意义。

---

## 17. 改动十五：新增 MTP draft token 路径

qwen3.6 中新增：

```text
_run_mtp_draft
run_mtp_probe
run_mtp_draft_step
run_mtp_draft_fast_step
```

核心逻辑是：

```text
1. 先跑主模型得到 hidden_states 和主 token
2. 取最后 token 对应的 hidden
3. 用主 token embedding 作为 MTP 输入
4. 调用 self.model.mtp(...)
5. 得到 draft logits
6. 采样 draft token
7. 重复 draft_len 次
```

这类路径通常用于：

```text
多 token prediction
草稿 token 生成
投机解码相关实验
```

它不是原版 nano-vLLM 必需功能，而是 qwen3.6 版本额外加入的模型能力验证/实验接口。

---

## 18. 改动十六：CUDA Graph 捕获支持 state_indices、verify graph 和 chunk graph

### 18.1 原版 CUDA Graph

原版只捕获 decode：

```python
set_context(False, slot_mapping=..., context_lens=..., block_tables=...)
outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
```

graph_vars 包括：

```text
input_ids
positions
slot_mapping
context_lens
block_tables
outputs
```

---

### 18.2 qwen3.6 CUDA Graph

qwen3.6 增加：

```python
state_indices = torch.arange(max_bs, dtype=torch.int32) if config.is_hybrid else None
set_context(..., state_indices=state_indices[:bs])
```

graph_vars 中也保存：

```python
self.graph_vars["state_indices"] = state_indices
```

并在捕获后：

```python
if config.is_hybrid:
    self.reset_gdn_state_slots(list(range(max_bs)))
```

此外，若启用 MTP，还会捕获：

```python
capture_verify_cudagraph()
capture_verify_chunk_cudagraph()
```

---

### 18.3 为什么 CUDA Graph 必须处理 state_indices

CUDA Graph 的特点是：

```text
捕获时固定执行图，replay 时只更新输入 buffer 内容。
```

对于普通 attention decode，运行时变化的是：

```text
input_ids
positions
slot_mapping
context_lens
block_tables
```

对于 hybrid decode，还多了：

```text
state_indices
```

如果 graph replay 时没有更新 `state_indices`，模型会继续读写捕获时的 state slot，导致不同请求之间 state 串扰。

所以 qwen3.6 修改 CUDA Graph 是支持 hybrid 的关键点之一。

---

## 19. 和前面几个文件的联动关系

`model_runner.py` 不是孤立改动，它和前面几个文件连成一条完整链路。

### 19.1 GDN state 链路

```text
sequence.py:
    seq.state_slot_id 保存请求状态槽位

scheduler.py:
    StateSlotManager 分配 / 释放 state_slot_id

model_runner.py:
    allocate_gdn_state 分配真实 GPU state tensor
    prepare_prefill / prepare_decode 把 state_slot_id 转成 state_indices
    set_context(..., state_indices=...)
    CUDA Graph replay 更新 state_indices
    reset/save/restore GDN state
```

### 19.2 多模态链路

```text
llm_engine.py:
    process_messages 得到 token_ids / pixel_values / image_grid_thw

sequence.py:
    seq.pixel_values / seq.image_grid_thw 临时挂载图像数据

model_runner.py:
    prepare_prefill 收集图像 tensor
    _compute_mrope_positions 计算 3D position
    _broadcast_image_data 在 TP 多卡间同步图像数据
    run_model 把图像参数传给模型
```

### 19.3 KV Cache 链路

```text
block_manager.py:
    分配 block_table

scheduler.py:
    调度 prefill/decode，may_append/deallocate

model_runner.py:
    prepare_prefill / prepare_decode 根据 block_table 构造 slot_mapping
    allocate_kv_cache 只给 attention 层分配 KV Cache
    Attention 层根据 Context 写入/读取 KV Cache
```

---

## 20. 从推理流程看 qwen3.6 ModelRunner 的新 forward 路径

### 20.1 Prefill 路径

```text
Scheduler.schedule 选出 waiting seqs
  ↓
ModelRunner.prepare_prefill
  ↓
收集 input_ids
收集 positions 或 MRoPE 3D positions
收集 slot_mapping
收集 cu_seqlens_q / cu_seqlens_k
收集 state_indices
收集 pixel_values / image_grid_thw / image_token_mask
  ↓
set_context(is_prefill=True, ..., state_indices=...)
  ↓
TP>1 时广播图像数据
  ↓
run_model
  ↓
model(input_ids, positions, pixel_values, image_grid_thw, image_token_mask)
  ↓
compute_logits
  ↓
sample
  ↓
返回 token_ids
```

---

### 20.2 Decode 路径

```text
Scheduler.schedule 选出 running seqs
  ↓
ModelRunner.prepare_decode
  ↓
从预分配 CPU buffer 填入 last_token / position / slot_mapping / context_lens / block_tables / state_indices
  ↓
拷贝到预分配 GPU buffer
  ↓
set_context(is_prefill=False, ..., state_indices=...)
  ↓
run_model
  ↓
如果 eager 或 batch 太大:
      直接 model forward
  否则:
      更新 graph_vars
      CUDA Graph replay
  ↓
compute_logits
  ↓
greedy/sample
  ↓
返回 next token
```

---

## 21. 原版与 qwen3.6 的核心差异表

| 模块 | 原版 | qwen3.6 | 影响 |
|---|---|---|---|
| 模型创建 | 固定 Qwen3 | 自动识别 Qwen3 / Qwen3_5 | 支持 hybrid 模型 |
| 权重加载 | 不保存结果 | 保存 loaded/skipped | 支持 MTP 权重校验 |
| KV Cache | 按所有层分配 | 只按 attention 层分配 | 避免 hybrid 层浪费和错位 |
| GDN state | 无 | conv/recurrent state 池 | 支持 GatedDeltaNet decode |
| Runtime buffer | 每轮构造 tensor | 预分配 CPU/GPU buffer | 降低 decode 开销 |
| Prefill | 文本 1D positions | 文本/图像，支持 MRoPE | 支持多模态 |
| Decode | KV Cache context | KV Cache + state_indices | 支持 hybrid recurrent state |
| TP 多卡 | 只广播方法调用 | 额外广播图像 tensors | 支持多模态 TP |
| Sampling | 简单 sampler | greedy/sampling + scores + TP gather | 支持 probe/top-k/全局采样 |
| CUDA Graph | 普通 decode graph | decode + state_indices + verify/chunk graphs | 支持 hybrid graph replay |
| 调试能力 | 基本无 | step/verify/logit/MTP probe | 支持正确性验证和实验 |

---

## 22. 面试角度应该怎么回答

如果面试官问：

> qwen3.6 版本的 `model_runner.py` 相比原版改了什么？

可以这样回答：

`model_runner.py` 是推理执行层，原版主要负责初始化 Qwen3ForCausalLM、加载权重、分配按层 KV Cache、准备 prefill/decode 输入、设置 Context、调用模型 forward、采样 token 和捕获普通 decode CUDA Graph。qwen3.6 版本的改动非常大，核心是把执行层从普通 Qwen3 dense 文本推理扩展到 hybrid / 多模态 / 验证实验路径。第一，它通过 `_create_model()` 根据 `model_type` 自动创建 Qwen3 或 Qwen3_5 模型，并保存权重加载结果用于 MTP 权重校验。第二，KV Cache 不再按总层数分配，而是扫描模型中真正有 `k_cache/v_cache` 的 attention 层，只给 full attention 层分配 KV Cache。第三，新增 `allocate_gdn_state()`，为 GatedDeltaNet 层分配 conv state 和 recurrent state，并通过 `state_indices` 把每个请求映射到对应 state slot。第四，prefill/decode 的 Context 都扩展了 `state_indices`，CUDA Graph replay 也要更新 `state_indices`，避免 hybrid decode 读写错状态槽。第五，它支持多模态输入：prefill 会收集 `pixel_values`、`image_grid_thw`、`image_token_mask`，计算 MRoPE 3D positions，并在 tensor parallel 多卡间广播图像数据。第六，它预分配 decode 和 sample runtime buffers，减少 decode 小张量分配开销。最后，它还新增了 verify、logit compare、decode state snapshot 和 MTP draft token 相关接口，用于验证 hybrid/MTP 路径的正确性。整体来看，qwen3.6 的 `model_runner.py` 是支持 Qwen3.6 hybrid 推理最核心的执行层改造文件。

---

## 23. 初学者最应该抓住的主线

这个文件非常长，不建议一开始逐行背。

应该先抓住三条主线。

### 23.1 普通 Qwen3 dense 主线

```text
prepare_prefill / prepare_decode
  ↓
set_context
  ↓
model(input_ids, positions)
  ↓
compute_logits
  ↓
sample
```

这是原版已有的主线。

---

### 23.2 Qwen3.6 hybrid 主线

```text
只给 attention 层分配 KV Cache
  ↓
给 GatedDeltaNet 层分配 conv/recurrent state 池
  ↓
Sequence 保存 state_slot_id
  ↓
Scheduler 分配 state_slot_id
  ↓
ModelRunner 把 state_slot_id 变成 state_indices
  ↓
set_context(..., state_indices=...)
  ↓
GDN 层按 state_indices 读写自己的 recurrent state
```

这是 qwen3.6 支持 hybrid 的核心。

---

### 23.3 多模态主线

```text
Sequence 携带 pixel_values / image_grid_thw
  ↓
prepare_prefill 收集图像张量
  ↓
_compute_mrope_positions 计算 3D position
  ↓
TP 多卡广播图像数据
  ↓
model forward 接收 pixel_values / image_grid_thw / image_token_mask
```

这是 qwen3.6 支持多模态的核心。

---

## 24. 最终结论

qwen3.6 版本的 `model_runner.py` 是整个项目中改动最重、最能体现工程升级的文件之一。

它的核心意义不是简单地“多传了几个参数”，而是把执行层从：

```text
普通 Qwen3 dense 文本推理执行器
```

升级为：

```text
支持 hybrid GatedDeltaNet state、多模态输入、MTP/verify 实验路径和更复杂 CUDA Graph replay 的执行器
```

最关键的工程改造是：

```text
1. 只给 attention 层分配 KV Cache
2. 给 GDN 层分配 recurrent/conv state 池
3. 通过 state_indices 把请求和 GDN state slot 对齐
4. CUDA Graph replay 时同步更新 state_indices
5. prefill 支持多模态图像张量和 MRoPE 3D positions
6. decode 阶段预分配 runtime buffers，降低小张量开销
7. 增加 verify/MTP/logit probe，用于复杂路径正确性验证
```

因此，如果要理解 qwen3.6 相比原版 nano-vLLM 的核心工程难点，`model_runner.py` 必须重点看。它是前面 `sequence.py`、`scheduler.py`、`block_manager.py` 中新增字段和资源管理真正落到模型执行的地方。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
