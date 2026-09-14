# Qwen3.6 + KV Cache 压缩融合：Codex 单次完整任务提示词

> 使用方法：在 Codex App / Codex CLI 中打开 `nano-vllm-qwen3.6` 作为主工作区，并保证原始 `nano-vllm` 与 `nano-kvllm` 仓库可从同一工作区或相邻目录只读访问。把下面“完整提示词”整体发送给 Codex。不要拆开逐条发送。

---

## 完整提示词

你现在负责完成一个完整的源码融合工程。请不要只给分析、计划或伪代码，也不要停在某个中间阶段；你必须在当前工作区内实际修改代码、补充测试、运行验证并形成可审查的 Git 提交记录，直到达到本文定义的最终验收条件，或者明确记录由于当前机器缺少 GPU、模型权重或系统依赖而无法执行的硬件验证项。

## 一、仓库角色

当前存在三个来源仓库：

1. **目标主体仓库：`nano-vllm-qwen3.6`**
   - 所有最终代码修改只落在该仓库。
   - 以它当前的架构、接口、功能和代码风格为主。

2. **KV 压缩参考仓库：`nano-kvllm`**
   - 只读参考。
   - 提取其中的周期性压缩策略、候选请求选择、SnapKV 重要性计算、每层 KV 选择及压缩事件思想。
   - 禁止直接覆盖目标仓库同名核心文件。

3. **原始基线仓库：`nano-vllm`**
   - 只读参考。
   - 用于识别两个派生分支各自改变了什么，以及确认原始语义。

开始前自行发现三个仓库路径；若仓库名称略有差异，通过 `.git`、`pyproject.toml`、包目录和 README 判断。除非仓库确实不存在，否则不要询问用户路径。

## 二、最终工程目标

以 `nano-vllm-qwen3.6` 为主体，将 `nano-kvllm` 的 KV Cache 动态稀疏压缩能力重新适配并融合进来，使目标工程具备：

- 周期性检查是否触发 KV Cache 压缩；
- 按请求选择压缩候选；
- 对 full-attention 层使用当前 Query 与历史 Key 计算重要性；
- 使用 SnapKV 类策略保留重要 KV token；
- 在保持真实逻辑位置不变的情况下，压缩物理 KV 上下文；
- 压实 KV Cache 后释放不再需要的 Paged KV blocks；
- 保持 Qwen3.6 分支已有的 Chunked Prefill、Hybrid GDN、视觉输入、MRoPE、张量并行和普通 CUDA Graph Decode 能力不被无关破坏；
- 对暂时无法安全兼容的功能实施显式限制，而不是静默产生错误结果。

该任务只实现“KV 稀疏压缩”。不要把 FP8 KV Cache 量化混入本次改动。

## 三、执行方式：一个任务，内部里程碑连续完成

这是一次完整任务，不要在每个阶段等待用户确认。你应在同一个 Codex 线程、同一个目标分支和同一个工作树中连续完成全部工作。

但禁止把所有修改堆成一个不可审查的大提交。你必须在内部按里程碑执行：

1. 先检查和记录基线；
2. 完成一个逻辑闭环；
3. 运行该阶段相关测试；
4. 修复失败；
5. 创建本地 Git commit；
6. 更新进度和设计记录；
7. 自动进入下一里程碑。

中途不要要求用户逐阶段确认。只有出现以下真正阻塞项时才停下说明：

- 三个仓库中有关键仓库完全不存在；
- 工作区无写权限；
- 关键代码已损坏且无法恢复；
- 必需依赖无法安装且不存在可用的静态/CPU 替代验证方式。

缺少 GPU、模型权重或多卡环境不属于停止实施的理由。此时继续完成代码、静态检查、CPU 单元测试和可执行测试脚本，并在最终报告中把硬件验证列为“未在当前环境执行”，不得伪造通过结果。

## 四、开始实施前必须创建的工程控制文件

先在目标仓库创建并持续维护：

### 1. `AGENTS.md`

写入本任务的永久约束、核心不变量、允许修改范围、禁止事项和验证命令。后续所有实现必须遵守它。

### 2. `docs/kv_compression_integration_spec.md`

作为设计事实来源，至少记录：

- 三个仓库的当前 commit；
- 两个分支相对原始 nano-vLLM 的改动边界；
- 状态模型；
- 压缩时间线；
- Prefix Cache、MTP、CUDA Graph、Hybrid 层的兼容策略；
- 文件级修改清单；
- 已知限制。

### 3. `PLANS.md`

列出本文全部里程碑、当前状态、验收条件和测试命令。它是执行计划，实施过程中实时更新状态。

### 4. `docs/kv_compression_implementation_log.md`

持续记录：

- 已完成内容；
- 修改文件；
- 关键设计决策及原因；
- 运行过的命令和真实结果；
- 遇到的问题与修复；
- 尚未验证的项目。

不要等到最后才补写这些文件。

## 五、不可违反的架构原则

### 原则 1：保留 Qwen3.6 主体，禁止直接覆盖核心文件

以下目标文件可能需要局部修改，但禁止用 `nano-kvllm` 版本整体替换：

- `config.py`
- `engine/sequence.py`
- `engine/block_manager.py`
- `engine/scheduler.py`
- `engine/model_runner.py`
- `layers/attention.py`
- `models/qwen3.py`
- 与 Qwen3.5/Qwen3.6 Hybrid、视觉、MTP、TP、CUDA Graph 相关的文件

必须先理解目标文件现有语义，再做最小侵入式融合。

### 原则 2：真实逻辑长度与物理 KV 长度必须分离

不得采用 `nano-kvllm` 中压缩后缩短 `Sequence.num_tokens` 的做法。

必须建立双长度模型：

- `num_tokens`：完整真实 token 时间线长度；压缩绝不能减小它；
- `kv_num_tokens`：当前实际存储在 full-attention KV Cache 中的物理上下文长度；压缩后允许减小。

相关语义必须满足：

- RoPE / MRoPE 的真实 position 基于逻辑时间线；
- FlashAttention 的 `cache_seqlens` 基于物理 KV 长度；
- 新 Decode token 的 KV 写入位置基于当前物理 KV 尾部；
- 最大生成长度、EOS、Chunked Prefill、GDN state、视觉位置和请求完成判断继续依据真实逻辑长度；
- 抢占和重算不能把已压缩 KV 当成原始完整 token 序列。

优先使用清晰命名，例如：

```python
num_tokens          # 逻辑长度
kv_num_tokens       # 物理 KV 长度
num_cached_tokens   # 已完成模型计算的逻辑 token 数
```

若目标仓库已有更合适的等价字段，应保持现有接口风格，但语义必须明确分离并有测试。

### 原则 3：所有 KV 位置必须通过 block table 映射

Paged KV Cache 的相邻逻辑 block 不保证拥有连续物理 block ID。

禁止通过以下方式跨逻辑 block 计算目标位置：

```python
dst_slot = previous_absolute_slot + 1
```

必须实现和复用统一映射：

```text
logical_position
  -> logical_block_index = logical_position // block_size
  -> block_offset = logical_position % block_size
  -> physical_block_id = block_table[logical_block_index]
  -> physical_slot = physical_block_id * block_size + block_offset
```

必须有非连续物理 block table 的单元测试，例如 `[7, 23, 2, 41, 9]`。

### 原则 4：压缩只作用于 full-attention KV 层

Qwen3.5/Qwen3.6 Hybrid 架构中的 Gated DeltaNet 层维护 recurrent/conv state，不使用标准 KV Cache。

- 不得压缩 GDN state；
- 不得把模型总层数当作 KV 层数；
- 给 full-attention 层建立明确的 `kv_layer_index` 和 `num_kv_layers`；
- 只在最后一个实际 KV 层完成后生成一次请求级压缩事件；
- 不得使用 `layer_id == num_hidden_layers - 1` 判断最后 KV 层。

### 原则 5：第一版压缩与 Prefix Cache 共享不兼容

原地修改共享 KV block 会破坏其他请求。

第一版必须安全处理：

- 当 KV 压缩启用时，禁止 Prefix Cache 命中/共享，或者在配置阶段明确拒绝两者同时启用；
- 压缩前验证相关 block 为该请求独占；
- 对 `ref_count > 1` 的 block 不得原地压缩；
- 暂不实现 Copy-on-Write，除非现有架构已经提供可靠且测试充分的机制。

### 原则 6：第一版压缩与 MTP 显式互斥

MTP 的 draft、verify、状态快照和回滚依赖逻辑位置与 KV slot 的稳定映射。

第一版应在配置校验中明确拒绝：

```text
kv_compress_enabled == True and enable_mtp == True
```

必须报出清晰异常，不得静默关闭其中一项，也不得声称支持但缺少完整测试。

### 原则 7：实际发生压缩的 Decode Step 必须走 Eager

动态 KV 选择、搬移和 Python 控制流不能依赖已经捕获的固定 CUDA Graph Replay 自动重新执行。

- 普通 Decode step 可以继续使用 CUDA Graph；
- 本轮实际包含任何压缩请求时，整个相关执行路径必须强制 Eager；
- 压缩完成后的普通 step 应恢复原 CUDA Graph 路径；
- 确保用于判断压缩的 batch 标志真实赋值和传播，不能存在定义但从未设置的 `compress_any` 类错误；
- 为 Eager/CUDA Graph 路径选择添加可测试的纯逻辑或诊断信息。

### 原则 8：每层可以有不同保留索引，但请求级新物理长度必须一致

SnapKV 可以为每个 attention layer 选择不同的历史 token，但每层压实后的 KV 序列长度必须一致，以便统一使用请求级 `kv_num_tokens` 和 block table。

### 原则 9：压缩事件必须在全部 KV 层成功压实后才释放 blocks

正确顺序：

1. Scheduler/ModelRunner 决定本轮候选和窗口；
2. `prepare_decode()` 生成逻辑位置、物理槽位和压缩元数据；
3. 每个 full-attention 层独立计算重要性并压实本层 K/V；
4. 最后一个实际 KV 层产生一次完成事件；
5. ModelRunner 返回结构化事件；
6. Engine/Scheduler 更新 `kv_num_tokens`；
7. BlockManager 缩短 `block_table` 并释放尾部 blocks；
8. 下一轮 Decode 使用新的物理长度。

任何 attention 层尚未完成时都不能提前释放 blocks。

### 原则 10：新增功能默认关闭

`kv_compress_enabled` 默认必须为 `False`。关闭时：

- 原有代码路径行为保持不变；
- 原有测试应继续通过；
- 不应引入不必要的性能损失；
- 生成结果在相同采样设置下应与修改前基线一致或满足项目已有容差。

## 六、建议配置项

根据目标项目现有配置风格设计，至少需要表达：

- `kv_compress_enabled: bool = False`
- `kv_compress_period`
- `kv_compress_topk`
- `kv_compress_window_blocks`
- `kv_compress_keep_ratio` 或等价保留预算
- SnapKV 池化/平滑窗口参数（如算法实际需要）
- 最小上下文或最小未压缩尾部长度

要求：

- 配置有范围校验；
- 参数之间有一致性校验；
- block/window/keep 数量必须可以映射为合法物理长度；
- 参数写入 README 或设计文档；
- 不照搬无法解释或未被代码使用的参数。

## 七、实现里程碑

你必须自动连续完成以下里程碑，并在每个里程碑测试成功后创建本地 commit。

### 里程碑 A：基线、差异地图和契约

- 记录三个仓库 commit 和工作树状态；
- 对比三者目录与核心文件；
- 运行目标仓库现有静态检查/单元测试/可用 smoke test；
- 创建四个工程控制文档；
- 明确受影响文件和状态不变量；
- 不在该里程碑实现压缩功能。

建议 commit：

```text
docs: define qwen36 kv compression integration contract
```

### 里程碑 B：配置与 Sequence 双长度状态

- 在目标配置中添加压缩参数和互斥校验；
- 扩展 Sequence 及其序列化/跨进程元数据；
- 保证 append token 时逻辑长度与物理 KV 长度的更新时机正确；
- 保证 Chunked Prefill 不把未计算 token 算入物理 KV；
- 添加字段语义和不变量测试；
- 关闭压缩时行为不变。

建议 commit：

```text
feat(kv-compress): add configuration and dual-length sequence state
```

### 里程碑 C：BlockManager 适配物理 KV 长度

- block 分配、追加、释放和 `block_table` 截断依据正确的物理 KV 长度；
- 不改变真实 token 历史；
- 新增压缩完成后的尾部 block 释放接口；
- 加入独占 block 检查；
- 禁止 Prefix Cache 共享冲突；
- 添加边界测试，包括压缩后长度位于块中间、刚好块边界、非连续 block ID。

建议 commit：

```text
feat(kv-compress): make block management use physical kv length
```

### 里程碑 D：独立压缩策略、SnapKV 和槽位压实工具

建议新增独立模块，例如：

```text
nanovllm/kv_compression/policy.py
nanovllm/kv_compression/snapkv.py
nanovllm/kv_compression/slots.py
nanovllm/kv_compression/events.py
```

实际路径遵循目标仓库包结构。

要求：

- 将“是否压缩、压缩哪些请求、压缩哪个窗口、保留多少 token”与 Attention 内核解耦；
- 尽量将候选选择和位置计算实现为纯函数；
- SnapKV 输入输出形状有明确断言；
- 正确处理 GQA/MQA 的 Query head 到 KV head 聚合；
- 明确保留窗口边界、最新 token 和必要 sink token；
- 每层可产生不同 keep indices，但输出长度一致；
- logical position 到 physical slot 映射不假设 block 连续；
- 若源目标区域重叠，压实操作必须避免覆盖尚未读取的数据，可通过临时 gather buffer 或安全顺序实现；
- 添加 CPU 可执行单元测试和适当 CUDA 测试（环境允许时）。

建议 commit：

```text
feat(kv-compress): add snapkv policy and paged slot compaction
```

### 里程碑 E：Context 与 Decode 准备元数据

- 扩展 context 以携带每请求逻辑 position、物理 KV 长度、block table、压缩窗口、目标长度和事件标识；
- `prepare_decode()` 同时正确准备：
  - 真实 RoPE/MRoPE position；
  - 新 token 写入物理 slot；
  - FlashAttention `cache_seqlens`；
  - 被选中请求的压缩元数据；
- TP worker 之间元数据序列化一致；
- staging buffer 大小、dtype 和 device 正确；
- 添加纯元数据构造测试。

建议 commit：

```text
feat(kv-compress): prepare decode metadata for compressed contexts
```

### 里程碑 F：Attention 层融合

- 只修改 full-attention 路径；
- 保持关闭压缩时原 fast path；
- 在启用且本轮选中时：
  - 使用当前层 Query 与历史 Key 计算重要性；
  - 获取当前层 keep indices；
  - 在当前层 K/V Cache 中按 block table 映射压实；
- 不修改 GDN state；
- 使用 `kv_layer_index/num_kv_layers` 判断最后 KV 层；
- 最后 KV 层生成结构化完成事件；
- 添加多层不同 keep indices、统一新长度测试；
- 添加 GQA/MQA 维度测试。

建议 commit：

```text
feat(kv-compress): integrate per-layer compression into full attention
```

### 里程碑 G：ModelRunner、Engine、Scheduler 事件闭环

- 调度周期、候选请求和 top-k 策略以目标 Scheduler 架构重新实现；
- 不直接照搬旧 Scheduler；
- 实际压缩 step 强制 Eager，普通 step 保留 CUDA Graph；
- ModelRunner 返回结构化压缩事件；
- Engine/Scheduler 在模型执行成功后应用事件；
- 更新 `kv_num_tokens`、未压缩尾部/窗口状态；
- BlockManager 释放 blocks；
- 防止重复应用、部分层失败后提前释放或事件丢失；
- TP 下仅由正确的控制 rank 应用一次请求级状态变化，并保证所有 worker KV 层已完成；
- 添加事件生命周期测试。

建议 commit：

```text
feat(kv-compress): close compression event loop and release blocks
```

### 里程碑 H：回归、兼容限制与文档

完成以下验证：

1. 压缩关闭：
   - 原测试通过；
   - 基线生成 smoke test 通过；
   - Chunked Prefill 通过；
   - Hybrid Qwen3.5/Qwen3.6 路径不受影响；
   - 视觉路径至少完成静态/元数据回归；
   - CUDA Graph 普通 Decode 路径仍可使用。

2. 压缩开启：
   - 达到周期后触发；
   - 物理 `kv_num_tokens` 减小而真实 `num_tokens` 单调增加；
   - block 数减少；
   - 下一个 Decode 正确读取压缩后的 KV；
   - RoPE/MRoPE position 不回退；
   - 非连续 block table 正确；
   - 多请求 batch 中只压缩符合条件的 top-k 请求；
   - 压缩 step 使用 Eager，下一普通 step 恢复 Graph 路径；
   - Prefix Cache 冲突被显式禁止；
   - MTP 冲突被显式禁止；
   - GDN state 未被压缩或错误释放。

3. 异常和边界：
   - 窗口不足；
   - keep 数无效；
   - 长度不足一个 block；
   - 压缩后仍位于同一 block；
   - 共享 block；
   - batch 中请求结束/抢占；
   - 无 full-attention KV 层时给出合理行为；
   - 环境没有 CUDA 时测试合理 skip，而不是误报通过。

更新 README、设计文档和实现日志。

建议 commit：

```text
test(kv-compress): add integration regression coverage and docs
```

### 里程碑 I：Benchmark 与可复现实验入口

添加独立 benchmark 或扩展已有 bench，至少可以对比：

- compression disabled；
- compression enabled with one or more keep ratios；

输出：

- 峰值 KV Cache 占用或已使用 block 数；
- 可支持上下文长度/生成长度；
- TTFT；
- TPOT；
- token throughput；
- 总延迟；
- 压缩事件次数；
- 每次压缩前后物理 KV token 数和 blocks；
- 压缩额外耗时；
- 生成质量的最低限度对照（固定 prompt 与 greedy token 差异、困惑度或项目可执行的代理指标）。

不得伪造 benchmark 数字。若当前环境不能运行模型，提供可执行脚本、参数说明和预期输出字段，并在日志中明确未运行。

建议 commit：

```text
bench: add reproducible kv compression evaluation
```

## 八、必须实现的核心单元测试

测试文件命名遵循现有项目风格，至少覆盖：

1. `test_logical_positions_map_through_noncontiguous_block_table`
2. `test_sequence_logical_length_never_shrinks_after_compression`
3. `test_kv_length_shrinks_and_append_grows_physical_tail`
4. `test_block_manager_releases_only_unused_tail_blocks`
5. `test_compression_rejects_shared_prefix_blocks`
6. `test_mtp_and_compression_are_mutually_exclusive`
7. `test_policy_selects_only_eligible_topk_sequences`
8. `test_snapkv_keep_budget_and_required_tokens`
9. `test_gqa_query_heads_reduce_to_kv_heads_correctly`
10. `test_each_kv_layer_may_keep_different_positions_with_same_length`
11. `test_last_kv_layer_not_last_model_layer_emits_event`
12. `test_compression_event_applied_exactly_once`
13. `test_compression_step_forces_eager_execution`
14. `test_normal_decode_can_return_to_cuda_graph_path`
15. `test_disabled_compression_preserves_existing_behavior`

对于 GPU 专属测试使用明确的 `skipif`，同时尽可能提供 CPU 纯逻辑测试。

## 九、代码质量要求

- 不保留未使用字段、死代码或只写不读的配置；
- 不使用全局可变状态传递压缩事件；
- 不让 Attention 直接修改 Scheduler 的 Sequence 对象；
- 事件应为显式 dataclass/结构体；
- 纯策略、槽位映射和 block 计算应与 CUDA kernel 调用解耦；
- 关键张量形状有断言和注释；
- 所有异常信息应说明冲突配置或不变量；
- 避免为了融合进行无关重构；
- 不删除目标分支已有功能；
- 不把模型层和调度层通过隐式 side effect 紧耦合；
- 不以“代码可编译”代替行为验证。

## 十、每个里程碑后的自动审查

每次 commit 前执行：

1. 查看 `git diff --check`；
2. 运行修改范围相关单元测试；
3. 运行 `python -m compileall` 或项目等价静态检查；
4. 搜索新增字段是否存在“只写不读”；
5. 检查是否误改 GDN、视觉、TP、Chunked Prefill 和 MTP 逻辑；
6. 检查 `num_tokens` 是否在任何压缩路径中被减小；
7. 检查所有跨 block 位置是否经过 block table；
8. 检查 block 是否仅在最后 KV 层完成后释放；
9. 更新 `PLANS.md` 和 implementation log；
10. 然后创建本地 commit。

若测试失败，先修复，不得把失败状态带入下一个里程碑。

## 十一、最终总验收

完成所有可执行工作后，运行项目可用的完整测试集合，并给出最终报告。最终回复必须包含：

1. 最终分支名和 commit 列表；
2. 修改/新增文件清单；
3. 完整调用链：Scheduler → ModelRunner → Context → Attention → Event → Scheduler/BlockManager；
4. 双长度模型说明；
5. 一次压缩 Decode step 的时间线；
6. Prefix Cache、MTP、Hybrid、视觉、TP、CUDA Graph 的实际处理；
7. 所有运行过的命令及真实结果摘要；
8. 未运行测试及原因；
9. 已知限制；
10. 如何启动普通生成、如何开启压缩、如何运行 benchmark；
11. `git status --short`，确保没有未解释的临时文件；
12. 对最终 diff 再做一次独立代码审查，重点寻找：
    - 逻辑/物理长度混淆；
    - 非连续 block ID 错误；
    - 共享 block 被原地修改；
    - 最后一模型层与最后 KV 层混淆；
    - CUDA Graph 中动态压缩未执行；
    - TP 多 rank 重复应用事件；
    - block 过早释放；
    - GDN state、MRoPE 或视觉位置被错误缩短。

若最终仍存在无法证明正确的部分，必须明确标注，不得用“应该可以”“大概通过”作为验收结论。

## 十二、现在开始

立即从检查仓库、Git 状态和现有测试入口开始。先建立设计事实来源和基线，然后连续实施全部里程碑。不要只返回计划；必须实际修改代码、运行验证并形成提交。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-kvllm-qwen3.6融合|模块-kvllm-qwen3.6融合]]

%% 项目关联导航：结束 %%
