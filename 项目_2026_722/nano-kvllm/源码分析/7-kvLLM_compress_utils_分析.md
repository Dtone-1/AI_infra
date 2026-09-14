# nano-kvLLM 新增源码分析：`compress_utils.py`

## 1. 文件定位与新增意义

`compress_utils.py` 是 nano-kvLLM 相比原版 nano-vLLM 新增的 **KV Cache 压缩执行层**。

此前几个文件的职责是：

```text
ModelRunner       决定本轮是否压缩、选择哪些请求
Attention         在 KV 写入后调用压缩函数
CompressMethod    根据 Query–Key 相关性选出要保留的位置
compress_utils    把“保留位置”落实为真实的 KV 搬移和长度更新
Scheduler         接收压缩事件
BlockManager      释放压缩后不再需要的物理 Blocks
```

因此，本文件解决的不是“哪些 token 更重要”这一算法问题，而是更偏 AI Infra 的工程问题：

> 如何把逻辑 token 索引转换为 Paged KV Cache 的物理槽位，并安全地原地压紧 K/V，使现有 FlashAttention 可以继续读取，同时向上层报告可以释放多少 Block。

它是连接“压缩算法”和“推理系统资源管理”的桥梁。

---

## 2. 文件整体结构

文件包含三个核心部分：

| 函数 | 主要职责 |
|---|---|
| `get_tail_window_and_tail_slots()` | 从 Block Table 中定位待压缩窗口和尾部残块 |
| `gather_kv_by_slots()` | 按物理槽位批量提取待评分、待搬移的 K/V |
| `MyCompressCompact()` | 执行选择、原地 Compact、长度更新和事件生成 |

整体数据流：

```text
选中的 Sequence
    ↓
根据 context_lens 和 block_tables 找到尾部窗口
    ↓
把逻辑 Block 展开为绝对 KV Slot
    ↓
Gather 窗口内 K/V
    ↓
SnapKV 生成 keep_idx
    ↓
把保留 KV 搬到窗口前部
    ↓
把未满尾块紧接着搬到保留区域之后
    ↓
缩短 context.context_lens
    ↓
最后一层生成 compression_events
```

---

## 3. 该文件实现的压缩模型

当前实现不是对整条序列重新筛选，而是周期性压缩尾部的一段完整 Block 窗口。

假设：

```text
P = 窗口之前、不参与本次压缩的前缀
W = 最后 window_blocks 个完整 Blocks
T = W 后面可能存在的未满尾块
```

压缩前：

```text
[P][W][T]
```

算法只对 `W` 做重要性选择，将其压缩为 `K`，而 `P` 和 `T` 保留：

```text
[P][K][T]
```

因此新上下文长度为：

```text
new_context_len
= old_context_len
- window_tokens
+ keep_tokens
```

其中：

```text
window_tokens = window_blocks × block_size
keep_tokens   = keep_blocks × block_size + keep_extra_tokens
```

这种局部周期压缩有几个工程优势：

- 不需要每次扫描全部历史 KV；
- 压缩成本受固定窗口限制；
- 旧前缀不反复搬移；
- 未满尾块可以继续写入；
- 容易换算压缩后可释放的 Block 数。

---

## 4. `get_tail_window_and_tail_slots()`：从逻辑长度定位物理槽位

### 4.1 输入

```python
block_tables
context_lens
seq_idxs
block_size
window_blocks
```

其中：

- `block_tables`：Batch 中每条 Sequence 的逻辑 Block → 物理 Block 映射；
- `context_lens`：压缩前实际可见的 KV 长度；
- `seq_idxs`：本轮被 ModelRunner 选中压缩的 Batch Index；
- `block_size`：每个 KV Block 容纳的 token 数；
- `window_blocks`：本次压缩窗口包含多少个完整 Block。

### 4.2 计算完整块与尾部长度

```python
full_blocks = old_context_lens // B
tail_lens   = old_context_lens % B
```

例如：

```text
context_len = 1100
block_size  = 256

full_blocks = 4
tail_len    = 76
```

表示前四个 Block 已满，第五个 Block 有 76 个有效 KV。

### 4.3 定位最后若干完整 Blocks

压缩窗口的逻辑 Block 下标为：

```text
full_blocks - window_blocks
到
full_blocks - 1
```

这保证未满尾块不进入 SnapKV 选择，而是单独保护并搬移。

### 4.4 将物理 Block 展开成绝对 Slot

一个物理 Block ID 对应的全局槽位范围为：

```text
block_id × block_size
到
block_id × block_size + block_size - 1
```

函数最终返回：

```python
window_src_slots  # [m, window_blocks * block_size]
```

它已经不再是逻辑 token 下标，而是可直接索引扁平 K/V Cache 的物理地址。

### 4.5 单独识别未满尾块

如果 `tail_lens > 0`，逻辑下标 `full_blocks` 对应未满尾块。

若没有尾块，则强制：

```python
tail_block_id = -1
```

这样后续代码不会把填充或越界 Block 当作真实缓存使用。

---

## 5. 为什么必须依赖 `compress_base_context_lens`

`MyCompressCompact()` 没有直接用已被前层修改的 `context.context_lens` 定位窗口，而是使用：

```python
context.compress_base_context_lens
```

这是跨层压缩正确性的关键。

一轮模型 Forward 会依次经过所有 Transformer 层：

```text
Layer 0 压缩并把 context_len 从 2048 改成 1024
Layer 1 开始执行
Layer 2 开始执行
...
```

如果后续层直接读取已缩短的 `context.context_lens`，就会把已经压缩后的 1024 再当作原长度重新压缩，导致每层窗口位置不同。

因此 ModelRunner 在所有层执行前保存统一基线：

```text
compress_base_context_lens = 压缩前长度
```

每一层都按同一个旧长度定位自己的 K/V 窗口，但把本层 FlashAttention 使用的 `context.context_lens` 更新成同一个新长度。

---

## 6. `gather_kv_by_slots()`：从 Paged Cache 提取连续算法输入

KV Cache 原始形状：

```text
[num_blocks, block_size, num_kv_heads, head_dim]
```

函数先扁平化为：

```text
[total_slots, num_kv_heads, head_dim]
```

再根据 `src_slots` Gather：

```text
[m, S, Hk, D]
```

最后转置为 SnapKV 需要的：

```text
[m, Hk, S, D]
```

其中：

- `m`：本轮选中的请求数量；
- `S`：压缩窗口 token 数；
- `Hk`：KV Head 数；
- `D`：Head Dimension。

这一步实现了从 Paged KV 的非连续物理布局，到压缩算法所需的规则 Batch Tensor 的转换。

---

## 7. `MyCompressCompact()` 的完整执行流程

### 7.1 前置检查

以下情况直接跳过：

```text
当前为 Prefill
Context Lens 不存在
Block Table 不存在
本轮没有选中任何请求
```

说明本文件只支持 Decode 阶段的 Paged KV Cache 压缩。

### 7.2 选取当前 Query

```python
q_sub = q_current.index_select(0, seq_idxs).unsqueeze(2)
```

形状从：

```text
[batch, Hq, D]
```

变为：

```text
[m, Hq, 1, D]
```

当前实现只使用本轮一个 Query 位置进行重要性判断。

### 7.3 Gather 压缩窗口 K/V

调用前两个辅助函数，得到：

```text
window_src_slots
old_context_lens
tail_lens
tail_block_ids
k_sub
v_sub
```

### 7.4 调用 SnapKV 选择保留位置

```python
keep_idx = SnapKV(
    q_sub,
    k_sub,
    v_sub,
    num_keep=keep_tokens - 2,
    window=1,
)
```

SnapKV 最终会额外加入：

- 压缩窗口第一个位置；
- 窗口最后一个位置。

所以：

```text
最终索引数量
= 1 + (keep_tokens - 2) + 1
= keep_tokens
```

### 7.5 将窗口内相对索引转换为物理 Slot

```python
src_keep = torch.gather(window_src_slots, 1, keep_idx)
```

至此，算法输出的逻辑相对位置变成了真实 K/V Cache 源地址。

### 7.6 构造目标位置

保留的 KV 被写入压缩窗口最前面的 `keep_tokens` 个槽位：

```python
dst_keep = window_src_slots[:, :keep_tokens]
```

因此压缩区域会从：

```text
[窗口内分散的重要 KV]
```

变成：

```text
[窗口前部连续排列的重要 KV]
```

这是后续 FlashAttention 能继续使用普通 `context_lens` 的前提。

### 7.7 搬移未满尾块

SnapKV 不处理窗口后的未满尾块。

这些尾部 KV 会原样搬到保留区域之后：

```text
目标位置 = 最后一个保留槽位 + 1 + 尾部偏移
```

因此最终布局为：

```text
未参与压缩前缀
+ 压缩保留 KV
+ 原样保留的尾部 KV
```

### 7.8 安全的重叠搬移

源区域与目标区域可能重叠。

代码先执行：

```python
vals_k = index_select(...).clone()
vals_v = index_select(...).clone()
```

再执行：

```python
index_copy_(...)
```

这样先保存所有源值，再覆盖目标位置，避免原地前移过程中提前覆盖尚未读取的数据。

### 7.9 立即更新本轮 Context Length

```python
context.context_lens[seq_idxs] = new_context_lens
```

这一更新发生在 `MyCompressCompact()` 返回前。

由于 `attention.py` 随后立即调用：

```python
flash_attn_with_kvcache(...)
```

FlashAttention 本轮就会只读取压缩后的有效长度。

如果只生成事件、等 Scheduler 在模型运行结束后再更新长度，本轮 Attention 会按旧长度读取残留区域，因此这里的即时更新是必须的。

### 7.10 最后一层生成事件

只有：

```python
layer_id + 1 >= num_layers
```

时才生成事件。

这避免每个 Transformer 层都向 Scheduler 上报重复事件。

事件包含：

```python
{
    "batch_index": ...,
    "layer": ...,
    "new_context_len": ...,
    "keep_blocks": ...,
    "freed_block_ids": ...,
    "tail_uncompressed_len_after": 0,
}
```

ModelRunner 在 `reset_context()` 前收集这些事件，再由 Scheduler 和 BlockManager 释放尾部 Blocks。

---

## 8. 为什么每层可以选择不同 token，但必须保留相同数量

`MyCompressCompact()` 在每个 Attention 层都会重新调用 SnapKV。

由于各层的 Query 和 Key 不同，各层得到的 `keep_idx` 也可能不同。

这并不一定是错误：

- 每层拥有独立 K/V Cache；
- 每层 Attention 可以保留不同的历史 token；
- 压缩后的 K/V 只供本层使用。

真正必须一致的是：

```text
所有层压缩后的有效长度相同
所有层使用相同数量的物理 Blocks
所有层把有效 KV 紧凑放在各自缓存的前部区域
```

当前保留数量由固定配置计算，因此各层 `new_context_len` 相同。

Tensor Parallel 各 Rank 也可以基于本地 Head 选择不同 token，只要压缩后的长度和 Block 数一致。因为每个 Rank 保存自己的 Head 分片，其 K/V 内容本就独立。

---

## 9. 物理 Block 回收为何延迟到 Scheduler

`MyCompressCompact()` 只改写 GPU K/V 内容和本轮 Context，不直接修改 Python 侧：

```text
Sequence.block_table
Block.ref_count
free_block_ids
used_block_ids
```

原因是这些资源元数据由主进程的 Scheduler 和 BlockManager 统一管理。

执行顺序：

```text
Attention 内完成所有层 KV Compact
    ↓
最后一层生成事件
    ↓
ModelRunner 返回事件
    ↓
Scheduler.postprocess()
    ↓
BlockManager.truncate_blocks()
```

这样可以避免模型 Forward 中直接操作调度器状态，也保证所有层处理完成后再统一提交物理 Block 释放。

---

## 10. `freed_block_ids` 的实际作用

事件中计算了：

```python
freed_block_ids
```

但当前 Scheduler 主要使用：

```text
keep_blocks
```

调用 `truncate_blocks()`，没有直接消费 `freed_block_ids`。

因此该字段目前主要用于：

- 调试；
- 日志；
- 验证 Scheduler 释放的 Block 是否一致；
- 将来实现更精细的释放协议。

建议后续要么删除冗余字段，要么在调试模式中加入一致性校验：

```text
事件中的 freed_block_ids
==
truncate_blocks() 实际释放的 Block IDs
```

---

## 11. 算法复杂度与性能开销

设：

```text
m  = 本轮压缩请求数
S  = window_blocks × block_size
Hq = Query Head 数
Hk = KV Head 数
D  = Head Dimension
K  = 保留 token 数
```

主要开销包括：

### 11.1 Gather K/V

```text
O(m × S × Hk × D)
```

### 11.2 Query-Key 评分

```text
O(m × Hq × S × D)
```

当前 Query Window 为 1。

### 11.3 Top-K

近似：

```text
O(m × Hq × S)
```

加上 Top-K 选择成本。

### 11.4 KV Compact

```text
O(m × (K + tail_len) × Hk × D)
```

所以压缩不是免费操作。设置：

- 压缩周期；
- 固定窗口；
- 每步最多压缩请求数；

就是为了把该开销控制在可接受范围内。

---

## 12. 关键正确性不变量

### 12.1 `keep_tokens` 必须小于窗口长度

应满足：

```text
0 < keep_tokens < window_tokens
```

否则：

- 等于时没有实际压缩；
- 大于时 SnapKV 会跳过或 Top-K 参数无效；
- 事件可能重置周期但不释放 Block。

当前代码没有显式断言，建议补充。

### 12.2 新长度与 Block 数必须一致

```text
keep_blocks_after = ceil(new_context_len / block_size)
```

Scheduler 和 BlockManager 必须使用同样结果。

### 12.3 K/V 使用同一搬移索引

当前代码对 K 和 V 共用 `src_flat`、`dst_flat`，满足该条件。

### 12.4 压缩后有效区域必须连续

FlashAttention 只接收长度和 Block Table，没有额外稀疏索引，因此有效 K/V 必须紧凑排列。

### 12.5 原始 RoPE 语义不能重算

这里只搬移已经编码过位置的 K，不重新按紧凑位置做 RoPE，因此物理位置变化不会重置逻辑位置。

---

## 13. 潜在问题一：共享 Prefix Cache Block 可能被原地改写

Compact 是直接写入现有物理 Block。

如果压缩窗口中的某个 Block：

```text
ref_count > 1
```

说明它正被其他 Sequence 共享。

当前请求原地改写该 Block，会同时破坏其他请求的 KV Cache。

这种情况可能出现在：

- 压缩窗口延伸到共享 Prompt 前缀；
- 多请求共享较长前缀；
- 缺少 Copy-on-Write。

当前设计需要保证至少一项成立：

1. 压缩窗口只覆盖 Decode 后独占 Blocks；
2. 压缩前对共享 Block 做 Copy-on-Write；
3. 被选请求的窗口 Blocks 全部验证 `ref_count == 1`。

这是系统正确性层面的重要检查。

---

## 14. 潜在问题二：多个 Sequence 的目标 Slot 可能发生冲突

正常情况下，每条 Sequence 的尾部 Blocks 应独占，因此不同请求的 `dst_flat` 不重叠。

如果 Block 共享或 Block Table 状态错误，多个 Sequence 可能写入相同物理槽位。`index_copy_()` 对重复目标索引的结果可能不符合预期。

建议在调试模式检查：

```text
dst_flat 是否全局唯一
```

或至少检查选中窗口的物理 Blocks 是否独占。

---

## 15. 潜在问题三：大量 `.item()` 和 Python 循环会触发同步

文件中存在：

```python
tail_lens.sum().item()
t.item()
keep_idx.min().item()
keep_idx.max().item()
src_flat.min().item()
.tolist()
```

这些操作会把 GPU 结果同步到 CPU，造成：

- CUDA Pipeline 停顿；
- 压缩步 TPOT 尖峰；
- 无法自然进入 CUDA Graph；
- Batch 越大时 Python 开销增加。

尤其尾部偏移构造使用 Python List：

```python
[torch.arange(int(t.item())) for t in tail_lens]
```

更高性能的实现可以完全 Tensor 化，或使用 Triton/CUDA Kernel 完成 Gather 与 Compact。

当前实现更偏正确性验证和原型实现，而非最终生产优化。

---

## 16. 潜在问题四：`tail_uncompressed_len_after = 0` 的语义不完全准确

本次压缩只处理完整窗口，未满尾块被原样保留。

但事件直接设置：

```python
tail_uncompressed_len_after = 0
```

如果该字段严格表示“压缩后仍未被压缩的新 token 数”，那么未满尾块长度可能应该保留为：

```text
tail_lens
```

当前清零会使下一次压缩必须再新增完整 `window_tokens` 才触发，未满尾部不计入增长。

这可能是有意的周期策略，但字段名容易误导。建议明确两种语义之一：

- “距上次压缩后的新增 token 数”：可清零；
- “当前未参与压缩的尾部 token 数”：应设为 `tail_lens`。

---

## 17. 潜在问题五：`bos` 实际是局部窗口首 token

SnapKV 返回的第一个固定索引为 0。

但传入的 K 只是尾部压缩窗口，所以这里的索引 0 表示：

```text
压缩窗口的第一个 token
```

而不是整条 Sequence 的真实 BOS。

真实全局 BOS 位于未参与本次压缩的前缀中，本来就没有被删除。

因此代码中的“BOS policy”更准确地说是：

```text
保留压缩窗口的首个锚点 token
```

建议修改注释，避免把局部窗口索引与全局 token 位置混淆。

---

## 18. 潜在问题六：压缩后的 Prefix Cache 哈希失效

本文件会改写压缩窗口内的物理 KV 内容，但 BlockManager 中这些 Blocks 可能还保存旧的：

```text
hash
token_ids
hash_to_block_id
```

如果继续把它们当作原始 Prefix Cache 复用，就可能命中错误 KV。

压缩后至少应使所有被改写的 Blocks 失去普通 Prefix Cache 资格，或者为压缩布局建立新的独立缓存语义。

前面 `block_manager.py` 只对最后一个保留 Block 做有限哈希失效，可能不足以覆盖这里实际改写的多个 Blocks。

---

## 19. 潜在问题七：事件只描述长度，不描述层内 token 身份

这本身可以接受，因为：

- Scheduler 只管理物理 Block 数；
- 每层可以保留不同 token；
- 上层不需要知道层内具体索引。

但前提是后续任何功能都不能假设：

```text
压缩后第 i 个缓存位置在所有层对应同一个原始 token
```

例如需要恢复 token 级映射、可视化或跨层共享索引时，当前事件信息不够。

---

## 20. 未使用的导入

文件导入了：

```python
from torch import nn
import triton
import triton.language as tl
```

当前代码没有使用这些对象。

这说明可能原计划实现 Triton Compact Kernel，但现在使用的是纯 PyTorch：

```text
index_select + clone + index_copy_
```

正式代码可删除无用导入；性能优化时也可以把当前搬移过程融合成 Triton Kernel。

---

## 21. 与其他文件的完整联动

```text
model_runner.py
  周期性选择 Sequence
  保存 compress_base_context_lens
        ↓
attention.py
  store_kvcache()
  调用 MyCompressCompact()
        ↓
compress_utils.py
  定位窗口
  Gather K/V
  调用 SnapKV
  原地 Compact
  更新 context_lens
  最后一层生成事件
        ↓
model_runner.py
  收集 compression_events
        ↓
scheduler.py
  更新 seq.num_tokens
  重置尾部计数
        ↓
block_manager.py
  truncate_blocks()
  回收物理 Block
```

---

## 22. 最终总结

`compress_utils.py` 是 nano-kvLLM 的 KV Cache 压缩执行核心。

它完成了四个关键转换：

```text
Block Table
→ 绝对物理 Slot

重要 token 相对索引
→ K/V Cache 真实源地址

分散保留 KV
→ 前部连续紧凑布局

GPU 层压缩结果
→ Scheduler 可处理的 compression_event
```

该文件最重要的工程价值是：

> 让 SnapKV 输出的不再只是“应该保留哪些 token”，而是真正变成 FlashAttention 可读取的压缩 KV Cache，并最终转化为可释放的物理 Blocks。

当前实现适合作为功能验证版本，但要走向高性能和高并发，还需要重点优化：

- 避免修改共享 Prefix Blocks；
- 完善压缩后哈希失效；
- 消除 `.item()` 和 Python 循环带来的同步；
- 使用 Triton/CUDA 融合 Gather 与 Compact；
- 明确尾部计数语义；
- 验证 `keep_tokens < window_tokens`；
- 确保压缩步正确退出普通 CUDA Graph。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-kvllm|模块-nano-kvllm]]

%% 项目关联导航：结束 %%
