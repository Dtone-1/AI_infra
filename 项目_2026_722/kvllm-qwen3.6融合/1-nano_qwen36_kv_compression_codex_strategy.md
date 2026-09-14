# nano-vLLM-Qwen3.6 融合 nano-kvllm 的完整修改策略与 Codex 分阶段任务书

## 0. 分析对象与结论

本报告基于以下三个本地仓库快照进行源码级对比：

- 原始参考：`nano-vllm`，当前提交 `bb823b3`
- KV 压缩来源：`nano-kvllm`，当前提交 `a7d8069`（v0.2.0）
- 融合主体：`nano-vllm-qwen3.6`，当前提交 `c468d63`

三个仓库的 Python 源码均已通过 `python -m compileall` 静态语法检查。

最终结论不是“把 nano-kvllm 的若干文件复制到 Qwen3.6 项目”，而是：

> 保留 Qwen3.6 项目的调度、分块 Prefill、混合 GDN 状态、视觉、多卡、CUDA Graph、MTP 与 FP8 权重加载架构；仅移植 nano-kvllm 的压缩策略、SnapKV 选择算法和 KV 内存压实思想，并围绕 Qwen3.6 的接口重新实现元数据管理。

直接覆盖文件会破坏 Qwen3.6 项目的核心语义。尤其不能直接覆盖：

- `engine/sequence.py`
- `engine/scheduler.py`
- `engine/model_runner.py`
- `layers/attention.py`
- `models/qwen3.py`

---

# 1. 两个分支各自在做什么

## 1.1 nano-vllm-qwen3.6 的主体结构

Qwen3.6 分支不是只替换了模型类，而是对整个推理生命周期做了扩展。

### 配置层：`nanovllm/config.py`

在原始 nano-vLLM 配置之上增加了：

- `full_config` 与 `text_config` 分离
- `is_hybrid`
- `max_state_slots`
- 视觉配置与图像特殊 token
- `enable_mtp`
- Qwen3.5/Qwen3.6 模型配置适配

这意味着融合时必须继续以该文件为准，只能追加 KV 压缩配置，不能用 nano-kvllm 的 `Config` 替换。

### 请求状态层：`engine/sequence.py`

Qwen3.6 分支中的 `Sequence` 同时维护：

- `num_tokens`：真实、完整的逻辑 token 数量
- `num_cached_tokens`：已经完成模型计算、无需再次 Prefill 的逻辑 token 数量
- `num_scheduled_tokens`：当前 Chunked Prefill 实际调度的 token 数量
- `block_table`：KV Cache 的物理块映射
- `state_slot_id`：Gated DeltaNet 的 recurrent/conv 状态槽
- 多模态输入数据
- 面向张量并行进程通信的紧凑序列化

这些语义被 Scheduler、ModelRunner、GDN、MTP 状态回滚共同依赖。

### 调度层：`engine/scheduler.py`

Qwen3.6 分支已支持：

- Chunked Prefill
- Prefix Cache 与 Hybrid 模型的兼容处理
- KV Block 与 GDN State Slot 两套资源管理
- 抢占后 KV 与 GDN 状态重算
- Prefill 和 Decode 两种不同的后处理逻辑

### 执行层：`engine/model_runner.py`

这是融合冲突最大的文件，包含：

- Qwen3/Qwen3.5 模型自动创建
- TP 多进程共享内存调用
- 预分配 CPU/GPU Decode staging buffer
- Chunked Prefill
- 视觉编码与 MRoPE
- 只为 full-attention 层分配 KV Cache
- 为 GDN 层分配 conv/recurrent state
- CUDA Graph Decode
- MTP draft/verify、状态快照和恢复
- 多种验证路径

因此，nano-kvllm 中约 300 行的旧版 `model_runner.py` 绝对不能覆盖该文件。

### 模型层

Qwen3.5/Qwen3.6 是 hybrid 架构：

- `full_attention` 层使用标准 KV Cache
- 其他层使用 `GatedDeltaNet`
- GDN 层维护 recurrent state / conv state，不使用 KV Cache

`allocate_kv_cache()` 已经通过扫描具有 `k_cache/v_cache` 的模块，只给 full-attention 层分配 KV Cache。

### FP8 的实际含义

当前 Qwen3.6 分支的 README 和代码表明：

- FP8 checkpoint 在加载阶段按 rank 切片并反量化
- 模型运行权重最终为 BF16
- KV Cache 也按 `hf_config.dtype` 分配，当前不是原生 FP8 KV Cache

因此，本次融合是“KV 稀疏压缩”，不是“FP8 KV Cache 量化”。二者应分为两个独立阶段。

---

## 1.2 nano-kvllm 的压缩链路

nano-kvllm v0.2.0 的主要思路是：

1. 全局每经过 `kv_compress_period` 个 Decode Step 进行一次检查。
2. 从当前 batch 中选择最多 `kv_compress_topk` 条候选序列。
3. 每条候选序列必须积累至少一个未压缩窗口。
4. 取末尾 `kv_compress_window_blocks` 个完整块作为压缩窗口。
5. 使用当前 Decode token 在每个 Attention 层产生的 Q，计算该层历史 K 的重要性。
6. 通过 SnapKV 保留重要 token，并保留窗口首 token 与最新 token。
7. 在每层 KV Cache 内执行原地搬移。
8. 最后一个 Attention 层产生压缩事件。
9. Scheduler 根据事件缩短 block table，释放多余物理块。

核心文件包括：

- `nanokvllm/layers/CompressMethod.py`
- `nanokvllm/layers/compress_utils.py`
- `nanokvllm/layers/attention.py`
- `nanokvllm/utils/context.py`
- `nanokvllm/engine/model_runner.py`
- `nanokvllm/engine/scheduler.py`
- `nanokvllm/engine/block_manager.py`
- `nanokvllm/engine/sequence.py`

其中真正值得移植的是：

- 周期性、窗口化压缩策略
- SnapKV 的 Q-K 重要性计算
- 每层独立选取 token 的设计
- 压缩后统一释放 block 的事件机制

---

# 2. 为什么不能直接复制 nano-kvllm

## 2.1 最严重的语义冲突：`num_tokens`

nano-kvllm 压缩后执行：

```python
seq.num_tokens = new_context_len
```

同时保留完整 `token_ids`，再额外维护：

- `rope_pos`
- `generated_completion_tokens`
- `tail_uncompressed_len`

也就是说，它把 `num_tokens` 从“真实逻辑长度”改成了“压缩后的物理 KV 长度”。

这在旧版纯 Qwen3 流程中勉强可以工作，但与 Qwen3.6 分支直接冲突：

- Chunked Prefill 用 `num_tokens` 计算剩余 token
- `num_completion_tokens` 用它判断最大生成长度
- Decode 的 position 使用 `len(seq)-1`
- GDN state 对应完整真实时间线
- MTP 快照使用逻辑位置映射 KV slot
- 抢占重算需要完整 token 序列长度

### 正确设计

必须分离两个长度：

- `num_tokens`：真实逻辑 token 长度，永远不因压缩减小
- `kv_num_tokens`：当前物理 KV 上下文长度，压缩后减小

Decode 时：

```text
RoPE / MRoPE position = num_tokens - 1
FlashAttention cache_seqlens = kv_num_tokens
写入 KV 的 slot = kv_num_tokens - 1 对应的物理槽
```

这样，模型的真实时间位置不缩短，但 FlashAttention 只读取保留下来的 KV。

---

## 2.2 Prefix Cache 与原地压缩冲突

Prefix Cache 允许多个 Sequence 引用同一个物理 KV block，即 `ref_count > 1`。

nano-kvllm 会直接修改压缩窗口中的 K/V 数据。如果压缩块是共享块，就会同时篡改其他请求的前缀缓存。

### 第一版必须采用的安全策略

当 `kv_compress_enabled=True` 时：

- 全局禁用 Prefix Cache 复用
- 每条序列拥有独占 KV block
- 压缩前断言窗口内 block 的 `ref_count == 1`

后续若要同时支持 Prefix Cache 与压缩，需要额外实现 Copy-on-Write，不应放进第一版。

---

## 2.3 nano-kvllm 的物理槽位搬移隐含连续 block ID 假设

原实现中尾部搬移使用类似：

```python
dst_tail_start = dst_keep[:, -1] + 1
```

这隐含了“下一个逻辑块的物理 block ID 与当前 block ID 连续”的假设。

Paged KV Cache 的 block table 可能是：

```text
[7, 23, 2, 41, 9]
```

相邻逻辑块通常不是相邻物理 block。不能通过绝对 slot 加一跨块。

### 正确实现

所有源和目标位置都应经过以下映射：

```text
逻辑 KV 位置
  -> logical_block_index = position // block_size
  -> offset = position % block_size
  -> physical_block_id = block_table[logical_block_index]
  -> physical_slot = physical_block_id * block_size + offset
```

建议新增统一函数：

```python
logical_positions_to_slots(block_table, logical_positions, block_size)
```

并用非连续 block ID 的单元测试覆盖。

---

## 2.4 CUDA Graph 下原版压缩可能根本不执行

nano-kvllm 中 `run_model()` 想通过 `compress_any` 决定是否回退 Eager，但当前代码没有在选中序列时正确把它设为 `True`。

CUDA Graph Capture 时没有进入动态压缩分支，Graph Replay 也不会重新执行 Python 分支。因此若不修复：

- `enforce_eager=True` 时可能压缩
- 默认 CUDA Graph Decode 时可能完全没有压缩

### 正确策略

- 普通 Decode Step：继续使用 CUDA Graph
- 实际包含压缩请求的 Step：强制 Eager
- 压缩结束后的下一步：恢复 CUDA Graph

判定应直接使用：

```python
need_eager_decode = bool(context.compress_selected_batch_indices)
```

不能依赖未维护的冗余标记。

---

## 2.5 “最后一层”在 Hybrid 模型中不是最后一个 KV 层

nano-kvllm 把模型总层数传入 Attention，使用：

```python
if layer_id + 1 >= num_layers:
    record_event()
```

Qwen3.5/Qwen3.6 中最后一层可能是 GDN 层，没有 KV Cache，也不会调用 `Attention.forward()`。

结果是：

- 所有 full-attention 层完成了 GPU KV 搬移
- 但没有任何层记录 compression event
- Scheduler 不释放 block，CPU 元数据与 GPU 内容失配

### 正确策略

给每个 `Attention` 模块分配：

- `kv_layer_index`
- `num_kv_layers`

在：

```python
kv_layer_index == num_kv_layers - 1
```

时记录事件。

该编号最好在 `ModelRunner.allocate_kv_cache()` 扫描 Attention 模块时设置，不需要修改 Qwen3/Qwen3.5 模型 forward 参数。

---

## 2.6 MTP 与压缩暂时不兼容

Qwen3.6 分支的 MTP/verify 代码使用：

```python
_kv_slots_for_positions(seq, logical_position, num_slots)
```

它默认“逻辑 token 位置”和“物理 KV 位置”一一对应。压缩后该关系不再成立。

同时 MTP 还涉及：

- speculative token 的临时 KV 写入
- accept/reject
- KV 与 GDN state 快照
- rollback
- verify graph/chunk graph

### 第一版处理方式

在配置中显式禁止：

```python
if kv_compress_enabled and enable_mtp:
    raise ValueError(...)
```

先完成普通 autoregressive Decode 的可靠融合。MTP 兼容应单独立项。

---

# 3. 应该保留、修改和新增哪些文件

## 3.1 必须修改的目标文件

| 文件 | 修改内容 | 是否可从 nano-kvllm 覆盖 |
|---|---|---:|
| `nanovllm/config.py` | 追加压缩配置、参数校验、MTP 互斥 | 否 |
| `nanovllm/engine/sequence.py` | 增加物理 KV 长度和压缩状态，扩展序列化 | 否 |
| `nanovllm/engine/block_manager.py` | 使用物理 KV 长度追加和截断 block | 否 |
| `nanovllm/engine/scheduler.py` | 应用压缩事件，抢占时恢复完整 KV 状态 | 否 |
| `nanovllm/engine/llm_engine.py` | 接收 `(token_ids, compression_events)` | 否 |
| `nanovllm/engine/model_runner.py` | 选择压缩请求、准备物理 context、配置 Attention、Eager 回退 | 否 |
| `nanovllm/layers/attention.py` | store 后、FlashAttention 前调用压缩 | 否 |
| `nanovllm/utils/context.py` | 增加单步压缩上下文 | 否 |

## 3.2 建议新增文件

| 文件 | 来源与用途 |
|---|---|
| `nanovllm/layers/compress_methods.py` | 从 `CompressMethod.py` 重构，保存 SnapKV |
| `nanovllm/layers/compress_utils.py` | 按目标 block table 语义重写 KV 选择和压实 |
| `nanovllm/engine/compression_policy.py` | 纯 Python 的周期和候选选择逻辑，便于单测 |
| `tests/test_sequence_kv_state.py` | 逻辑/物理长度状态测试 |
| `tests/test_block_manager_compression.py` | block 释放与 ref_count 测试 |
| `tests/test_compression_policy.py` | 周期与 Top-K 序列选择测试 |
| `tests/test_snapkv.py` | GQA/MHA 选择算法测试 |
| `tests/test_kv_compaction.py` | 非连续 block ID 的压实测试 |
| `tests/test_compression_disabled_regression.py` | 关闭压缩时行为不变 |
| `bench_kv_compression.py` | 吞吐、TPOT、block 和压缩耗时评测 |

## 3.3 第一版不应修改的文件

- `nanovllm/models/qwen3.py`
- `nanovllm/models/qwen3_5.py`
- `nanovllm/models/qwen3_mtp.py`
- `nanovllm/layers/gated_delta_net.py`
- `nanovllm/models/vision_encoder.py`
- `nanovllm/utils/quant.py`
- `nanovllm/utils/loader.py`

nano-kvllm 修改 `models/qwen3.py` 的主要原因是向 Attention 传 `Layer` 和 `num_layers`。目标工程可以在 `allocate_kv_cache()` 中直接给 Attention 模块编号，因此不需要污染模型定义。

---

# 4. 融合后的核心状态设计

## 4.1 Sequence 新增字段

建议在 Qwen3.6 的 `Sequence` 中新增：

```python
self.kv_num_tokens = self.num_tokens
self.kv_tail_uncompressed_tokens = 0
self.kv_cache_compressed = False
```

新增属性：

```python
@property
def num_kv_blocks(self):
    return ceil(self.kv_num_tokens / block_size)

@property
def last_kv_block_num_tokens(self):
    ...
```

不要引入 nano-kvllm 的 `rope_pos` 和 `generated_completion_tokens`，因为目标工程继续让 `num_tokens` 表示真实逻辑长度。

## 4.2 必须持续成立的不变量

Codex 每个阶段都应围绕这些不变量写断言和测试：

1. Scheduler rank 0 上：`num_tokens == len(token_ids)`。
2. `0 <= num_cached_tokens <= num_tokens`。
3. 未发生压缩或抢占重算后：`kv_num_tokens == num_tokens`。
4. 压缩后：`0 < kv_num_tokens <= num_tokens`。
5. `len(block_table) == ceil(kv_num_tokens / block_size)`。
6. Decode 的 RoPE/MRoPE position 使用 `num_tokens - 1`。
7. FlashAttention 的 `context_lens` 使用 `kv_num_tokens`。
8. Decode KV 写入 slot 使用 `kv_num_tokens - 1` 对应的物理位置。
9. 压缩事件必须在 append 新采样 token 之前应用。
10. GDN state slot 不因 KV 压缩改变。
11. 开启压缩时所有可被压缩 block 必须 `ref_count == 1`。
12. 抢占后将 `kv_num_tokens` 恢复为 `num_tokens`，并从完整 token_ids 重算。

## 4.3 append_token 的语义

正常采样得到一个新 token 后：

```python
num_tokens += 1
kv_num_tokens += 1
kv_tail_uncompressed_tokens += 1
```

这里的 `kv_num_tokens` 与当前 nano-vLLM 一样，包含“下一步将被写入 KV Cache 的 last_token”。

## 4.4 压缩事件的语义

建议事件格式：

```python
{
    "batch_index": int,
    "seq_id": int,
    "old_kv_num_tokens": int,
    "new_kv_num_tokens": int,
    "keep_block_count": int,
    "freed_block_ids": list[int],
    "compressed_window_tokens": int,
    "retained_window_tokens": int,
}
```

Scheduler 应先：

```text
更新 block table、释放 block
更新 kv_num_tokens
重置 kv_tail_uncompressed_tokens
设置 kv_cache_compressed=True
```

再调用 `append_token()`。

---

# 5. 一次压缩 Decode Step 的正确时间线

假设某条请求当前：

```text
真实逻辑长度 num_tokens = 1800
物理 KV 长度 kv_num_tokens = 1289
block_size = 256
block_table = [7, 23, 2, 41, 9, 15]
```

当前 last token 的真实位置为 1799，但它即将写入物理 KV 的位置为 1288。

完整流程：

1. Scheduler 调度该序列。
2. `BlockManager.may_append()` 按 `kv_num_tokens` 判断是否需要新 block。
3. `ModelRunner.prepare_decode()`：
   - `input_id = seq.last_token`
   - `position = seq.num_tokens - 1 = 1799`
   - `context_len = seq.kv_num_tokens = 1289`
   - `slot_mapping` 按物理 KV 位置 1288 查 block table
4. Compression Policy 判断当前是否为周期点，并选中该序列。
5. `set_context()` 保存：
   - selected batch indices
   - 压缩前 base context lengths
   - 当前是否必须 Eager
6. 进入第一个 full-attention 层：
   - 先将当前 token 的 K/V 写入旧物理尾部
   - 使用本层当前 Q 对本层压缩窗口 K 评分
   - 将保留 K/V 和未压缩 tail 搬到较前逻辑位置
   - 更新本步 `context.context_lens`
   - FlashAttention 读取压缩后的本层 KV
7. GDN 层正常更新自己的 recurrent/conv state，不参与 KV 压缩。
8. 后续 full-attention 层分别用本层 Q 选择本层要保留的 KV。
9. 最后一个 KV 层生成一次 compression event。
10. LM Head 与 Sampler 生成新 token。
11. ModelRunner 在 reset context 前取出 compression events。
12. Scheduler 根据 event 截断 block table 并释放物理块。
13. Scheduler append 新 token：
    - 真实逻辑长度加一
    - 压缩后的物理 KV 长度加一
14. 下一 Decode Step 可重新使用 CUDA Graph，直到下一个实际压缩 Step。

---

# 6. Codex 的工作方式

## 6.1 仓库与分支

在 Qwen3.6 主体仓库中：

```bash
git checkout -b feature/kv-compression-integration
git tag baseline-qwen36-c468d63
```

将 nano-kvllm 作为只读参考仓库放在相邻目录，例如：

```text
workspace/
├── nano-vllm-qwen3.6/    # Codex 允许修改
└── nano-kvllm/            # 只读参考
```

不要让 Codex 在两个仓库之间自动执行整目录复制。

## 6.2 每次只允许一个阶段

每个阶段都要求 Codex：

1. 先分析目标文件和参考文件。
2. 列出计划修改的文件、接口和不变量。
3. 只修改本阶段允许的文件。
4. 添加对应测试。
5. 运行测试和 compileall。
6. 输出完整变更摘要。
7. 未通过验收时不得进入下一阶段。

## 6.3 建议创建 AGENTS.md

在目标仓库根目录创建 `AGENTS.md`，写入：

```markdown
# KV Compression Integration Rules

- The Qwen3.6 repository is the source of truth.
- Do not replace target files with nano-kvllm files wholesale.
- Compression must be disabled by default.
- With compression disabled, outputs and scheduling behavior must remain unchanged.
- Keep logical token length and physical KV length separate.
- `num_tokens` always means logical sequence length.
- Compression only affects full-attention KV cache; never GDN recurrent/conv state.
- Disable prefix-cache sharing while KV compression is enabled.
- Compression steps must use eager execution; non-compression decode may use CUDA Graph.
- Do not modify MTP, vision, FP8 loader, or GDN implementation in the first integration.
- Reject `enable_mtp=True` together with `kv_compress_enabled=True` in the first version.
- Add tests for every state transition and block-table mutation.
- Never assume physical block IDs are contiguous.
```

---

# 7. 分阶段任务与可直接交给 Codex 的提示词

## 阶段 0：建立基线与融合契约

### 目标

不实现功能，只固定当前 Qwen3.6 行为、测试命令和接口约束。

### 允许修改

- `AGENTS.md`
- `docs/kv_compression_integration_contract.md`
- 基线测试或测试脚本

### Codex 提示词

```text
你正在修改 nano-vllm-qwen3.6 仓库，旁边的 nano-kvllm 仓库只作为只读参考。

本阶段不要实现 KV 压缩，也不要修改现有推理核心代码。请完成：
1. 阅读目标仓库的 config.py、sequence.py、block_manager.py、scheduler.py、llm_engine.py、model_runner.py、attention.py、qwen3.py、qwen3_5.py、gated_delta_net.py。
2. 阅读参考仓库中对应文件和 CompressMethod.py、compress_utils.py。
3. 创建 AGENTS.md，写明：目标仓库是唯一主体；不能整文件覆盖；num_tokens 始终表示逻辑长度；必须新增独立物理 KV 长度；压缩只处理 full-attention；压缩时禁用 prefix sharing；压缩 Step 强制 eager；第一版禁止与 MTP 同时开启。
4. 创建 docs/kv_compression_integration_contract.md，记录现有 Prefill、Decode、GDN state、CUDA Graph 和 MTP 的接口边界。
5. 运行并记录 python -m compileall 的基线结果。
6. 如果本机有模型，记录 compression 不存在时的固定 greedy 输出、首步 logits top-k 或现有 smoke test 结果，作为后续回归基线。

只提交文档和基线测试，不得修改推理行为。最后列出所有检查结果和后续风险。
```

### 验收

- 核心 Python 文件无行为修改。
- compileall 通过。
- 有明确的基线输出或至少静态基线。

---

## 阶段 1：配置与 Sequence 双长度模型

### 目标

只建立状态模型，不触发压缩。

### 修改文件

- `nanovllm/config.py`
- `nanovllm/engine/sequence.py`
- `tests/test_sequence_kv_state.py`

### 建议配置

```python
kv_compress_enabled: bool = False
kv_compress_period: int = 1024
kv_compress_topk: int = 20  # 实际含义：每个压缩周期最多选择多少条 sequence
kv_compress_window_blocks: int = 4
kv_compress_keep_blocks: int = 2
kv_compress_keep_extra_tokens: int = 1
```

配置校验至少包括：

- period >= 1
- topk >= 1
- window_blocks >= 1
- 0 <= keep_blocks < window_blocks
- `2 <= keep_blocks * block_size + keep_extra_tokens < window_blocks * block_size`
- 第一版 `kv_compress_enabled and enable_mtp` 直接报错

### Codex 提示词

```text
只完成配置和 Sequence 状态扩展，不实现任何压缩触发或 GPU 搬移。

要求：
1. 在目标 config.py 的 slots dataclass 中追加 KV 压缩配置，默认 kv_compress_enabled=False，保持现有 Qwen3.5/Qwen3.6、视觉、MTP 配置逻辑不变。
2. 在 __post_init__ 中增加严格参数校验，并在第一版禁止 kv_compress_enabled 与 enable_mtp 同时开启。
3. 在目标 Sequence 中新增 kv_num_tokens、kv_tail_uncompressed_tokens、kv_cache_compressed。
4. num_tokens 的现有语义完全不变；不要加入参考项目的 rope_pos 或 generated_completion_tokens。
5. 新增 num_kv_blocks 和 last_kv_block_num_tokens 属性。
6. append_token 同时增加逻辑长度和物理 KV 长度，并增加未压缩尾部计数。
7. 扩展 __getstate__/__setstate__，兼容目标仓库现有 7/8 项旧序列化格式，并序列化所有新增字段。
8. 添加纯 CPU 单元测试，验证初始化、append、序列化往返和旧格式兼容。
9. 压缩关闭时所有现有字段和行为必须保持不变。

禁止修改 scheduler、model_runner、attention、模型文件。
```

### 验收

- 单测通过。
- `num_tokens` 从未被物理压缩语义污染。
- 旧序列化格式仍能读取。

---

## 阶段 2：BlockManager 支持物理 KV 长度

### 目标

让 block 分配、追加和截断基于 `kv_num_tokens`，但仍不执行 GPU 压缩。

### 修改文件

- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/scheduler.py` 中仅允许传递 prefix-cache 禁用参数和 preempt reset
- `tests/test_block_manager_compression.py`

### 关键设计

- `can_append()` 和 `may_append()` 使用物理 KV 长度。
- 开启压缩时 allocate 禁止 Prefix Cache 复用。
- 新增明确方法，例如：

```python
apply_compression(seq, new_kv_num_tokens, keep_block_count)
```

- 该方法负责：
  - 检查 keep count
  - 释放尾部 blocks
  - 更新 `seq.block_table`
  - 更新 `seq.kv_num_tokens`
  - 清理不再可信的 hash
- 抢占后：
  - 释放当前物理 KV blocks
  - `kv_num_tokens = num_tokens`
  - `kv_tail_uncompressed_tokens = 0`
  - `kv_cache_compressed = False`
  - 从完整 token_ids 重算

### Codex 提示词

```text
本阶段只修改 BlockManager 的物理 KV 元数据生命周期，不实现压缩算法。

要求：
1. 所有 Decode append 所需 block 的判断改为基于 seq.kv_num_tokens，而不是 len(seq)。
2. 保持 num_tokens 和 num_cached_tokens 的逻辑语义不变。
3. 当 config.kv_compress_enabled=True 时，Scheduler 分配请求必须禁用 Prefix Cache 复用，确保每个被压缩 block 的 ref_count 为 1。
4. 新增 apply_compression 或等价单一入口，输入 new_kv_num_tokens 和 keep_block_count，安全释放尾部 block 并更新 block_table。
5. 不要依赖 token_ids 为压缩后的物理 KV 提供哈希，因为压缩后逻辑 token 与物理槽不再一一对应。
6. preempt 时恢复完整重算状态：kv_num_tokens=num_tokens、清零尾部计数、清除 compressed 标志。
7. 添加 block_size=4 的纯 CPU 测试，覆盖：跨 block append、压缩释放、重复释放保护、ref_count、抢占重置。
8. 压缩关闭时保留当前 Prefix Cache 行为。

禁止修改 model_runner、attention 和压缩算法文件。
```

### 验收

- 对非压缩请求，block 分配结果与原版一致。
- 压缩后 `len(block_table) == ceil(kv_num_tokens/B)`。
- 无 block 重复释放。

---

## 阶段 3：独立实现压缩策略与 SnapKV

### 目标

先写纯函数与单元测试，不挂入真实 Attention。

### 修改文件

- 新增 `nanovllm/engine/compression_policy.py`
- 新增 `nanovllm/layers/compress_methods.py`
- 新增 `nanovllm/layers/compress_utils.py`
- 对应 tests

### Compression Policy

输入：

- decode step counter
- 当前 seq 列表
- block size
- period
- max selected sequences
- window blocks

输出：

- `is_compress_step`
- `selected_batch_indices`
- `selected_seq_ids`

必须使用 `seq.kv_num_tokens` 和 `seq.kv_tail_uncompressed_tokens`。

### 压实实现建议

第一版优先正确性，允许对最多 20 条 selected sequence 做小规模 Python 循环，每条内部使用 GPU tensor gather/index_copy。不要为了完全向量化复制参考代码中的脆弱槽位计算。

实现逻辑位置到物理槽位的统一映射，不得假设 block IDs 连续。

### Codex 提示词

```text
本阶段实现独立的 compression policy、SnapKV 和 KV compaction 纯逻辑，但不要接入 Attention 或 ModelRunner。

参考 nano-kvllm 的算法思想，不要逐行复制。

要求：
1. compression_policy.py 提供可独立测试的周期和候选选择函数。kv_compress_topk 表示一次最多压缩多少条 sequence，不是保留 token 的 top-k 数量。
2. compress_methods.py 实现 SnapKV，支持 MHA 和 GQA；输出每条 sequence 共用的一组有序 token index。
3. compress_utils.py 实现逻辑 KV 位置到 Paged KV physical slot 的映射。
4. 所有目标槽位都必须通过 block_table 映射，不得使用绝对 slot + 1 跨越逻辑 block。
5. 提供一个单层压实函数，输入 q、k_cache、v_cache、block_table、old_context_len 和参数，返回 new_context_len、keep_block_count、保留 index 等结果。
6. 第一版可以对 selected sequence 循环，以正确性为优先；每条内部使用 index_select/index_copy_，搬移前 clone 源数据，避免重叠写破坏。
7. 删除参考实现中未使用的 nn、triton 等导入。
8. 添加测试：
   - MHA 与 GQA SnapKV shape
   - 保留 index 有序且不重复
   - 非连续 block IDs，例如 [7, 2, 11, 4, 9]
   - K/V 用逻辑位置编码，压实后逐槽核对
   - tail_len 为 0、1、B-1
   - keep_extra_tokens 不同取值
   - 多层允许选择不同 token，但新长度一致

禁止修改现有 Attention、ModelRunner、Scheduler。
```

### 验收

- 非连续 block ID 测试通过。
- 无越界、重复 index 和重叠写错误。
- CPU 测试可运行；若函数限定 CUDA，至少把位置映射与索引选择拆成 CPU 可测部分。

---

## 阶段 4：扩展 Context 与 ModelRunner Decode 准备

### 目标

产生压缩选择元数据，并确保逻辑位置与物理 KV 长度正确，但 Attention 暂时仍不搬移。

### 修改文件

- `nanovllm/utils/context.py`
- `nanovllm/engine/model_runner.py`
- policy tests / prepare_decode tests

### Context 新字段

建议：

```python
compression_enabled_for_step: bool
compress_selected_batch_indices: tuple[int, ...]
compress_selected_seq_ids: tuple[int, ...]
compress_base_context_lens: torch.Tensor | None
compression_events: list | None
```

继续保留：

- `state_indices`
- Prefill 的 cu_seqlens
- block_tables

### prepare_decode 必须改为

```text
input_id       = last_token
position       = num_tokens - 1
context_len    = kv_num_tokens
slot_mapping   = physical slot of kv_num_tokens - 1
block_tables   = physical block table
state_indices  = unchanged
```

### Codex 提示词

```text
本阶段把压缩选择元数据接入 Context 和 prepare_decode，但不要修改 Attention，不要真正搬移 KV。

要求：
1. 在 slots Context 中追加 compression step 所需字段，保留 state_indices 和所有现有字段。
2. set_context/reset_context 必须完整初始化和清理新增字段。
3. ModelRunner 增加 decode_step_counter，并调用独立 compression policy。
4. prepare_decode 的 position 必须继续使用逻辑 seq.num_tokens - 1。
5. context_lens 与 slot_mapping 改为使用 seq.kv_num_tokens 和 last_kv_block_num_tokens。
6. 保存 compression 前的 context_lens clone，供所有 KV 层使用同一基准长度。
7. 只有普通 run() 的 Decode 才允许选择压缩；所有 MTP probe、verify、chunk verify 路径必须关闭 compression metadata。
8. 当前阶段不返回 compression event，也不修改 block table。
9. 为 prepare_decode 增加可检查的状态测试，验证逻辑 position 与物理 context_len 可以不同。

禁止修改 attention.py 和模型文件。
```

### 验收

- 压缩关闭时 prepare_decode 行为与基线相同。
- 人为设置 `num_tokens != kv_num_tokens` 时，position 和 context length 分别正确。

---

## 阶段 5：接入 Attention，每个 full-attention 层执行压缩

### 目标

真正修改每层 KV Cache，但暂不让 Scheduler 释放 blocks。

### 修改文件

- `nanovllm/layers/attention.py`
- `nanovllm/engine/model_runner.py` 的 `allocate_kv_cache()` 与 `run_model()`
- Context / tests

### 最小侵入方案

在 `allocate_kv_cache()` 扫描 KV 模块时：

```python
kv_modules = [...]
for kv_layer_index, module in enumerate(kv_modules):
    module.k_cache = ...
    module.v_cache = ...
    module.configure_kv_compression(config, kv_layer_index, len(kv_modules))
```

这样：

- 不修改 `models/qwen3.py`
- 不修改 `models/qwen3_5.py`
- 不给模型 forward 新增 Layer 参数
- Hybrid 模型自然只枚举 full-attention 层

Attention 顺序：

```text
store current K/V
-> selected 时压实本层 KV
-> FlashAttention 读取压缩后的 KV
```

### Codex 提示词

```text
本阶段把已测试的单层 KV compaction 接入目标 Attention。

要求：
1. 不修改 qwen3.py、qwen3_5.py 的 forward 签名。
2. 在 ModelRunner.allocate_kv_cache() 枚举实际拥有 KV cache 的 Attention 模块，并给它们配置 kv_layer_index 和 num_kv_layers。
3. Qwen3.5/Qwen3.6 的 GDN 层不得参与压缩，也不得修改 recurrent/conv state。
4. Attention.forward 中先 store 当前 token 的 K/V，再进行压缩，最后调用 flash_attn_with_kvcache。
5. 所有 KV 层使用 context.compress_base_context_lens 作为压缩前基准；每层可基于自己的 Q 选择不同 token，但 new_context_len 必须相同。
6. 第一/后续 KV 层应更新 context.context_lens，使本步 FlashAttention 读取压缩后的长度。
7. 只有最后一个 KV 层记录一次 compression event；判断依据是 kv_layer_index == num_kv_layers - 1。
8. 在实际选中压缩请求的 Decode Step，run_model 必须绕过 CUDA Graph 强制 eager；其他 Decode Step 继续使用 graph。
9. 压缩前断言涉及 block 的 ref_count 独占条件应在 CPU 元数据层完成；GPU 层不要猜测共享关系。
10. 添加测试或 instrumentation，证明 hybrid 模型只配置 full-attention 模块。

本阶段 Scheduler 可以暂不消费事件，但 ModelRunner 必须能在 reset_context 前读取事件。
```

### 验收

- 关闭压缩：仍走原 CUDA Graph 路径。
- 选中压缩：明确进入 Eager。
- 最后一个 KV 层产生且只产生一条每 sequence event。
- GDN state 未改变。

---

## 阶段 6：ModelRunner、LLMEngine、Scheduler 完成事件闭环

### 目标

完成 GPU 压实与 CPU block 元数据释放的原子闭环。

### 修改文件

- `engine/model_runner.py`
- `engine/llm_engine.py`
- `engine/scheduler.py`
- `engine/block_manager.py`
- tests

### 正确顺序

```text
ModelRunner forward + sample
-> 读取 context.compression_events
-> reset_context
-> 返回 token_ids, events
-> Scheduler 先 apply_compression
-> 再 append sampled token
```

### Codex 提示词

```text
本阶段完成 compression event 从 ModelRunner 到 Scheduler 的闭环。

要求：
1. 普通 ModelRunner.run() 返回 token_ids 和 compression_events；在 Prefill 或无压缩时 events 为空。
2. 不改变 MTP/probe/verify 方法现有返回协议。
3. LLMEngine.step() 兼容新返回值，并把 is_prefill 和 events 传给 Scheduler.postprocess()。
4. Scheduler 在 Decode 后处理中必须先按 batch_index/seq_id 校验和去重 event，再调用 BlockManager.apply_compression，然后才 append 新采样 token。
5. 压缩事件应用后更新 kv_num_tokens、kv_tail_uncompressed_tokens=0、kv_cache_compressed=True，但不得修改 num_tokens、num_cached_tokens 和 token_ids 历史。
6. append 后逻辑长度和物理 KV 长度各自加一。
7. Prefill、Chunked Prefill、EOS、max_tokens、GDN state slot 的现有逻辑不得改变。
8. finish/deallocate 与 preempt 不能双重释放压缩后已经移除的 blocks。
9. TP worker 也必须执行相同 GPU 压实，但只有 rank 0 的 event 被 Scheduler 使用。
10. 添加端到端状态机测试，至少模拟：
   - Prefill -> 多次 Decode -> 压缩 -> 继续 Decode
   - batch 中只有部分 sequence 被压缩
   - 压缩后 EOS
   - 压缩后 preempt 并全量重算
   - 多次压缩
```

### 验收

- CPU block table 与 GPU context length 在 Step 结束后一致。
- block 数实际减少。
- 下一 Decode Step slot_mapping 正确。
- 逻辑生成长度判断仍正确。

---

## 阶段 7：回归、CUDA Graph、TP 与 Hybrid 验证

### 目标

证明融合没有破坏 Qwen3.6 主体能力。

### 必测矩阵

| 模式 | 压缩 | 执行方式 | 预期 |
|---|---:|---|---|
| Qwen3 | 关 | Eager | 与基线一致 |
| Qwen3 | 关 | CUDA Graph | 与基线一致 |
| Qwen3 | 开但未触发 | CUDA Graph | 与关闭压缩一致 |
| Qwen3 | 触发 | Eager fallback | block 减少、继续生成 |
| Qwen3.5 hybrid | 关 | Graph | 与基线一致 |
| Qwen3.5 hybrid | 触发 | Eager fallback | 仅 full-attention KV 压缩 |
| TP=1 | 触发 | Eager | 正常 |
| TP>1 | 触发 | Eager | 各 rank 元数据一致 |
| Vision Prefill | 开 | Prefill 不压缩 | 正常 |
| MTP | 开压缩 | 配置阶段拒绝 | 明确报错 |

### Codex 提示词

```text
本阶段不增加新功能，只做回归修复和多模式验证。

要求：
1. compression disabled 时固定 seed 的 greedy token IDs、logits top-k、Prefill/Decode 调度与基线一致。
2. compression enabled 但参数设置为不会触发时，结果必须与 disabled 完全一致。
3. 实际触发 compression 时记录：logical length、physical KV length、active block count、freed block count、compression latency。
4. 验证普通 Decode Step 使用 CUDA Graph，只有实际压缩 Step 回退 eager。
5. 验证 Qwen3.5/Qwen3.6 hybrid 中 GDN state 不被压缩或清零。
6. TP>1 时所有 rank 必须使用相同 selected batch indices 和 base context lengths；必要时从 rank 0 广播选择结果，不要依赖潜在不同步的隐藏状态。
7. 运行 compileall 和全部 CPU tests。
8. 对无法在本机运行的大模型测试，提供明确命令和预期日志字段，不要伪造结果。
```

---

## 阶段 8：Benchmark 与质量评估

### 目标

证明项目“融合成功”和“压缩有价值”，而不只是能运行。

### 系统指标

至少记录：

- Prefill throughput
- Decode throughput
- TTFT
- TPOT
- 总生成 latency
- 峰值 GPU memory
- KV Cache active block 数
- 每次压缩释放 block 数
- 每次压缩耗时
- 压缩 Step TPOT 抖动
- 同显存下最大并发请求数
- OOM 前最大上下文或最大 batch

### 质量指标

压缩会改变模型输出，不能要求触发压缩后 token 完全一致。建议：

- Math500 / GSM8K：准确率
- LongBench 子集：任务得分
- WikiText 或自建长文本：perplexity
- 固定长对话：关键事实保持率
- greedy 输出与 baseline 的 token agreement / edit distance 仅作为辅助

### 对照组

```text
A. compression disabled
B. compression enabled, but never triggered
C. periodic window compression
D. 不同 window_blocks / keep_blocks / period / selected sequence count
```

### Codex 提示词

```text
请新增 bench_kv_compression.py，不修改核心算法。

要求：
1. 同一脚本支持 compression on/off 和所有压缩参数。
2. 使用 torch.cuda.Event 或同步后的 perf_counter 分别统计普通 Decode Step 与 compression Step 耗时。
3. 每步记录 active/free KV blocks、logical tokens、physical KV tokens、被选 sequence 数。
4. 汇总 TTFT、TPOT、Decode throughput、峰值显存、压缩事件数、平均释放 blocks、最大并发。
5. 输出 JSON/CSV，便于后续画图。
6. 提供小模型本地验证命令和服务器 Qwen3.6-27B-FP8 TP=4 命令。
7. 不得把单次 smoke test 吞吐写成稳定性能结论；至少 warmup 后重复多次。
```

---

# 8. 推荐的 Git 提交序列

每个提交只做一个可验证主题：

```text
1. docs: add kv compression integration contract
2. feat: add kv compression config and sequence physical length state
3. feat: make block manager track physical kv length
4. feat: add compression policy and snapkv utilities
5. feat: add context metadata and decode preparation
6. feat: integrate per-kv-layer cache compaction
7. feat: apply compression events in scheduler
8. test: add hybrid graph and tensor-parallel regression coverage
9. bench: add kv compression benchmark and metrics
10. docs: document limitations and experiment protocol
```

每次提交前至少执行：

```bash
python -m compileall nanovllm tests
pytest -q tests/test_sequence_kv_state.py
pytest -q tests/test_block_manager_compression.py
pytest -q tests/test_compression_policy.py
pytest -q tests/test_snapkv.py
pytest -q tests/test_kv_compaction.py
```

有模型环境时再执行 smoke/regression。

---

# 9. Codex 每阶段完成后的统一审查提示词

不要直接相信 Codex 的“已完成”。每阶段可继续给它以下审查任务：

```text
请以代码审查者身份检查你刚才的改动，不要继续增加功能。

逐项回答并提供对应代码位置和测试：
1. num_tokens 是否仍只表示逻辑长度？是否存在任何压缩路径修改它？
2. Decode position 是否来自逻辑长度，而 context_lens 和 slot_mapping 是否来自物理 KV 长度？
3. block_table 长度是否始终与物理 KV 长度一致？
4. 是否有任何代码假设 block IDs 连续？
5. 压缩是否可能修改 ref_count>1 的共享块？
6. Hybrid 模型最后一个 full-attention 层是否一定记录 event？
7. 压缩 Step 是否确实绕过 CUDA Graph？非压缩 Step 是否仍使用 Graph？
8. GDN recurrent/conv state 是否完全未被压缩路径修改？
9. Prefill、Chunked Prefill、preempt、EOS 和 max_tokens 是否仍使用逻辑状态？
10. MTP 是否被明确禁用，而不是静默产生错误？
11. TP 各 rank 是否使用相同压缩选择？
12. compression disabled 是否有零行为回归？

如果发现问题，先给出最小修复方案和失败测试，再修改代码。不要用描述代替测试。
```

---

# 10. 人工检查 Codex 改动时最应关注的红线

看到以下情况应立即退回：

1. 把 nano-kvllm 的 `sequence.py`、`scheduler.py` 或 `model_runner.py` 整体复制进目标。
2. 压缩后执行 `seq.num_tokens = new_context_len`。
3. 用 `len(seq)` 同时代表 RoPE position 和 FlashAttention context length。
4. 修改 GDN state 来“配合 KV 压缩”。
5. 使用模型总层数判断最后一个压缩层。
6. 在 Attention forward 中新增模型层号参数并一路改所有模型 forward，虽然可以通过模块编号避免。
7. 压缩时仍允许 Prefix Cache 共享 block。
8. 通过 `physical_slot + 1` 跨 Paged block。
9. CUDA Graph 打开时没有明确 Eager fallback。
10. 直接声称 MTP 兼容，但没有重写 speculative KV 映射和 rollback。
11. 只检查“程序不报错”，没有检查 block table、context length、slot mapping 和 ref_count。
12. 默认开启压缩，导致基线行为无法稳定回归。

---

# 11. 第一版合理的功能边界

建议第一版定义为：

> 在 nano-vllm-qwen3.6 的普通 autoregressive Decode 路径中，为 Qwen3 和 Qwen3.5/Qwen3.6 hybrid 模型的 full-attention 层加入周期性窗口 SnapKV 压缩；压缩 Step 使用 Eager，其他 Step 保留 CUDA Graph；支持 TP，保留视觉 Prefill；暂不支持 MTP 与 Prefix Cache 共享。

第一版不应承诺：

- MTP speculative decode 下压缩
- Prefix Cache + compression 同时生效
- 原生 FP8 KV Cache
- 自动突破模型最大 RoPE/位置长度
- 所有压缩配置任意组合都高效

KV 压缩主要首先证明：

- 活跃 KV blocks 减少
- 同显存可容纳更多并发或更长生成
- 吞吐/TPOT 在合理压缩周期下可接受或改善
- 质量下降在可控范围内

---

# 12. 最终建议

最重要的策略只有三条：

1. **以 Qwen3.6 分支的状态机为准，不移植 nano-kvllm 的 Sequence 语义。**
2. **以逻辑长度/物理 KV 长度双轨设计为融合核心。**
3. **先完成普通 Decode 的最小可靠闭环，再逐步恢复 CUDA Graph、TP、Hybrid；MTP 单独处理。**

这样拆分后，Codex 每次只处理有限文件和明确不变量。你也可以通过单元测试、状态日志和 Git 小提交逐阶段判断它改得是否正确，而不是等数千行代码一次性改完后再排查。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-kvllm-qwen3.6融合|模块-kvllm-qwen3.6融合]]
- 关联阅读：[[outputs/项目整理/专题-02-KV缓存与调度|专题-02-KV缓存与调度]]

%% 项目关联导航：结束 %%
