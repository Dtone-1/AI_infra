# nano-vLLM 与 nano-kvLLM：`model_runner.py` 源码对比分析

## 1. 对比对象与核心结论

本次对比的两个文件分别是：

- 原版 nano-vLLM 的 `model_runner.py`
- nano-kvLLM 的 `model_runner.py`

`ModelRunner` 是推理系统中真正把 Scheduler 选出的请求转化为 GPU 模型执行的模块。它负责：

```text
Sequence 状态
    ↓
构造 input_ids / positions
    ↓
构造 slot_mapping / context_lens / block_tables
    ↓
运行 Qwen3 模型
    ↓
采样下一个 token
    ↓
将运行结果返回 LLMEngine
```

在原版 nano-vLLM 中，`ModelRunner` 主要解决：

- Prefill 与 Decode 两种输入组织方式；
- Prefix Cache；
- Paged KV Cache 的 Block Table 和 Slot Mapping；
- Tensor Parallel 多进程执行；
- CUDA Graph 加速 Decode；
- 模型输出采样。

nano-kvLLM 在此基础上增加了 KV Cache 压缩控制链。它不仅执行模型，还负责：

1. 判断本轮是否为周期性压缩步；
2. 从当前 Decode Batch 中选择需要压缩的 Sequence；
3. 把压缩选择信息写入全局 Context；
4. 让各 Attention 层在写入新 KV 后执行压缩；
5. 在模型运行后收集 `compression_events`；
6. 将压缩事件返回 LLMEngine 和 Scheduler；
7. 将 RoPE 的逻辑位置与压缩后的 KV 上下文长度分离。

因此，nano-kvLLM 中的 `ModelRunner` 可以概括为：

> KV Cache 压缩的运行时控制中心：它决定何时压缩、压缩哪些请求、给 Attention 层提供哪些压缩元数据，并把压缩结果向上层返回。

---

# 2. 整体变化概览

| 改动方向 | 原版 nano-vLLM | nano-kvLLM | 工程意义 |
|---|---|---|---|
| 包命名空间 | `nanovllm` | `nanokvllm` | 使用独立改造工程 |
| Hugging Face dtype | `hf_config.dtype` | `hf_config.torch_dtype` | 适配新的配置字段命名 |
| 模型构造 | `Qwen3ForCausalLM(hf_config)` | `Qwen3ForCausalLM(hf_config, config)` | 将压缩配置传入模型和 Attention 层 |
| 压缩开关 | 无 | `kv_compress_enabled` | 支持运行时启用或关闭压缩 |
| 压缩周期 | 无 | `decode_step_counter` | 周期性触发压缩 |
| 共享内存大小 | `2**20` | `2**24` | 容纳更大的 Sequence 和压缩元数据 |
| Warmup | 支持按 token budget 截断序列，并设置 `num_scheduled_tokens` | 直接使用 `max_model_len`，不再设置调度 token 数 | 配合完整 Prefill，但存在边界风险 |
| Prefill 输入 | 支持 Chunked Prefill | 一次处理剩余完整 Prompt | 与 Scheduler 简化一致 |
| Decode Position | `len(seq) - 1` | `seq.rope_pos` | 压缩后保持 RoPE 逻辑位置连续 |
| Decode Context Length | `len(seq)` | 仍为 `len(seq)`，但此时表示压缩后缓存长度 | 将物理缓存长度与逻辑位置解耦 |
| 压缩候选选择 | 无 | 周期、窗口、Top-K Request 筛选 | 控制压缩频率和单步压缩开销 |
| Context 压缩字段 | 无 | selected batch、seq id、基础长度等 | 向所有 Attention 层广播压缩决策 |
| CUDA Graph | 普通 Decode 使用 Graph | 试图在压缩步切回 eager | 动态压缩通常不能直接依赖固定 Graph |
| 返回协议 | 只返回 `token_ids` | 返回 `(token_ids, compression_events)` | 将压缩结果送回 Scheduler |
| 退出清理 | 清理 Graph 和进程组 | 额外尝试清理查询窗口 Buffer | 释放压缩辅助状态 |
| Tokenizer | 无 | 导入但未使用 | 调试残留 |

本文件最核心的两条设计线是：

```text
逻辑位置：rope_pos 单调递增，不受压缩影响
物理上下文：context_lens 随 KV 压缩缩短
```

以及：

```text
ModelRunner 选择压缩对象
→ Context 将决策传入 Attention
→ Attention 执行压缩并记录事件
→ ModelRunner 收集事件
→ Scheduler 截断物理 Blocks
```

---

# 3. 原版 ModelRunner 在推理链路中的位置

原版推理路径可以抽象为：

```text
Scheduler.schedule()
        ↓
ModelRunner.run(seqs, is_prefill)
        ↓
prepare_prefill() / prepare_decode()
        ↓
set_context(...)
        ↓
model(input_ids, positions)
        ↓
Attention 根据 Context 写入和读取 KV Cache
        ↓
compute_logits()
        ↓
Sampler
        ↓
token_ids
```

其中 `Context` 是 `ModelRunner` 和 Attention 层之间的临时通信对象，保存：

- 当前是否为 Prefill；
- Query/Key 的变长序列信息；
- KV 写入位置 `slot_mapping`；
- 每条请求的上下文长度；
- 每条请求的 Block Table。

nano-kvLLM 正是利用这条已有的 Context 通道，将压缩控制信息继续传入所有 Attention 层。

---

# 4. 详细改动一：模型构造函数接收完整 Config

原版：

```python
self.model = Qwen3ForCausalLM(hf_config)
```

nano-kvLLM：

```python
self.model = Qwen3ForCausalLM(hf_config, config)
```

## 4.1 为什么需要把 Config 传入模型

原版模型只需要 Hugging Face 配置，例如：

- hidden size；
- attention heads；
- KV heads；
- layer count；
- dtype；
- RoPE 参数。

KV Cache 压缩还需要工程配置，例如：

```text
是否启用压缩
压缩周期
压缩窗口 Block 数
每轮选择多少请求
压缩阈值
查询窗口或重要性统计参数
```

这些参数不是标准 Qwen3 模型结构的一部分，因此不能只依赖 `hf_config`。

将完整 `config` 传入模型后，模型内部的 Attention 层可以：

- 初始化压缩算法；
- 创建 Query Window Manager；
- 读取窗口和选择参数；
- 在 Decode Forward 中根据 Context 执行压缩；
- 生成 `compression_events`。

这说明压缩算法的实际实现很可能位于：

```text
models/qwen3.py
或
layers/attention.py
```

而 `ModelRunner` 负责运行时调度。

---

# 5. 详细改动二：新增压缩开关和全局 Decode Step 计数器

nano-kvLLM：

```python
self.kv_compress_enabled = config.kv_compress_enabled
if config.kv_compress_enabled:
    self.decode_step_counter = 0
```

## 5.1 `kv_compress_enabled`

该字段允许同一套代码运行两种模式：

```text
关闭压缩：作为普通 nano-vLLM 使用
开启压缩：周期性选择请求并触发 KV Cache 压缩
```

这对实验非常重要，因为可以在相同模型、相同输入和相同调度配置下对比：

- 显存占用；
- Decode 吞吐；
- TPOT；
- 最大并发；
- 输出质量；
- 压缩额外开销。

## 5.2 `decode_step_counter`

每执行一次 Decode Batch：

```python
self.decode_step_counter += 1
```

然后根据：

```python
decode_step_counter % kv_compress_period == 0
```

决定本轮是否允许压缩。

注意它是 ModelRunner 级全局计数器，而不是每条 Sequence 独立计数器。

因此：

```text
第 N 个全局 Decode Batch
→ 所有当前运行请求共同进入一次压缩候选检查
```

某条请求是否真正被压缩，还要继续满足自己的：

- 未压缩尾部长度；
- 当前完整 Block 数；
- Block 边界条件。

---

# 6. 详细改动三：共享内存从 1 MiB 扩大到 16 MiB

原版：

```python
size = 2**20
```

nano-kvLLM：

```python
size = 2**24
```

即：

```text
1 MiB → 16 MiB
```

## 6.1 为什么需要扩大

前面对 `sequence.py` 的分析已经确认，nano-kvLLM 在多进程通信时会序列化：

- 完整 `token_ids`；
- `seq_id`；
- `rope_pos`；
- `generated_completion_tokens`；
- `tail_uncompressed_len`；
- Block Table；
- 其他缓存状态。

原版 Decode 阶段通常只需要传递最后一个 token，通信数据较小。

nano-kvLLM 为支持压缩状态恢复和跨进程一致性，传输内容明显增大，因此扩大共享内存是配套修改。

## 6.2 固定大小仍存在溢出风险

`write_shm()` 直接执行：

```python
data = pickle.dumps(...)
n = len(data)
self.shm.buf[4:n+4] = data
```

没有检查：

```text
n + 4 <= SharedMemory 大小
```

在以下情况下仍可能超过 16 MiB：

- 很长上下文；
- 很大 Batch；
- Tensor Parallel 多 Worker；
- 每个 Sequence 都保存完整 token 历史；
- Block Table 很长。

建议加入显式检查，并在超限时：

- 抛出清晰异常；
- 使用动态共享内存；
- 改成共享 Tensor/元数据；
- Decode 只传递增量状态。

---

# 7. 详细改动四：Warmup 不再依赖 `num_scheduled_tokens`

## 7.1 原版 Warmup

原版先限制单条 Warmup 序列长度：

```python
seq_len = min(max_num_batched_tokens, max_model_len)
```

然后计算 Batch 数：

```python
num_seqs = min(
    max_num_batched_tokens // seq_len,
    max_num_seqs
)
```

并设置：

```python
seq.num_scheduled_tokens = seq_len
```

这与原版 Chunked Prefill 协议一致。

## 7.2 nano-kvLLM Warmup

nano-kvLLM：

```python
num_seqs = min(
    max_num_batched_tokens // max_model_len,
    max_num_seqs
)
seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
self.run(seqs, True)
```

不再设置 `num_scheduled_tokens`，因为新的 `prepare_prefill()` 会自动处理完整剩余序列。

## 7.3 与 Scheduler 简化保持一致

nano-kvLLM 已取消原版的 Chunked Prefill 状态，因此 Warmup 也改成完整序列模式。

## 7.4 潜在边界问题

如果：

```text
max_model_len > max_num_batched_tokens
```

则：

```python
max_num_batched_tokens // max_model_len == 0
```

从而：

```text
num_seqs = 0
seqs = []
```

后续 `run([], True)` 很可能产生空输入错误。

原版使用：

```python
seq_len = min(max_num_batched_tokens, max_model_len)
```

可以避免这一问题。

此外，直接使用 `max_model_len` 构造 Warmup 序列，可能比原版消耗更多显存和计算，尤其当模型最大上下文很长时。

更稳健的写法应恢复：

```python
seq_len = min(max_num_batched_tokens, max_model_len)
num_seqs = max(1, min(max_num_batched_tokens // seq_len, max_num_seqs))
```

---

# 8. 详细改动五：KV Cache dtype 字段改为 `torch_dtype`

原版使用：

```python
hf_config.dtype
```

nano-kvLLM 使用：

```python
hf_config.torch_dtype
```

包括：

```python
torch.set_default_dtype(hf_config.torch_dtype)
block_bytes = ... * hf_config.torch_dtype.itemsize
```

这主要属于 Hugging Face 配置兼容性调整，不是 KV Cache 压缩核心逻辑。

但它很重要，因为 KV Cache Block 数计算直接依赖 dtype 大小：

```text
每个 Block 字节数
=
2
× 层数
× block_size
× 本 rank 的 KV heads
× head_dim
× dtype bytes
```

如果 dtype 字段取错，可能导致：

- Block 数量估算错误；
- 显存分配过量；
- 显存利用率不足；
- 初始化报错。

---

# 9. KV Cache 初始分配本身没有缩小

nano-kvLLM 仍然通过：

```python
self.kv_cache = torch.empty(
    2,
    num_hidden_layers,
    num_kvcache_blocks,
    block_size,
    num_kv_heads,
    head_dim,
)
```

一次性建立整个 KV Cache Block Pool。

这说明当前压缩机制并不是：

```text
启动时分配更小的 KV Cache Tensor
```

而是：

```text
运行过程中让单条请求更早释放已占用 Blocks，
使固定大小的全局 Block Pool 被更多请求复用。
```

因此压缩的主要系统收益是：

- 同一块 GPU 上提高可并发 Sequence 数；
- 降低 Decode 过程中 Block 耗尽概率；
- 减少 Scheduler 抢占；
- 延长可运行上下文；
- 提高固定 KV Pool 的利用率。

查看 `nvidia-smi` 时，总预分配显存不一定明显下降，因为全局 KV Pool 仍然按配置预分配。

更合理的观测指标是：

```text
每条请求占用 Block 数
空闲 Block 数
最大并发数
抢占次数
可完成的上下文长度
```

---

# 10. 详细改动六：Prefill 准备逻辑取消 Chunked Prefill

## 10.1 原版

原版为每条 Sequence 使用：

```python
start = seq.num_cached_tokens
seqlen_q = seq.num_scheduled_tokens
end = start + seqlen_q
```

所以本轮可以只处理 Prompt 的一部分。

`slot_mapping` 也只覆盖：

```text
[start, end)
```

对应的缓存槽位。

## 10.2 nano-kvLLM

nano-kvLLM 使用：

```python
seqlen = len(seq)
input_ids.extend(seq[seq.num_cached_tokens:])
positions.extend(range(seq.num_cached_tokens, seqlen))
seqlen_q = seqlen - seq.num_cached_tokens
seqlen_k = seqlen
```

即：

```text
从 Prefix Cache 命中结束位置开始，
一次处理所有剩余 Prompt token。
```

这与前面 `scheduler.py` 的变化一致：

```text
只有完整 Prompt 能进入本轮预算时，
才调度该请求。
```

## 10.3 Slot Mapping 的变化

原版根据精确的 `start_block` 和 `end_block` 构造本轮写入位置。

nano-kvLLM 直接从：

```python
seq.num_cached_blocks
```

遍历到：

```python
seq.num_blocks
```

其中 `num_cached_blocks` 只计算完整缓存 Block 数。

这依赖一个重要前提：

> Prefix Cache 只复用完整 Block，不存在部分 Block 命中。

这与 BlockManager 的哈希策略一致，因为只有完整 Block 才拥有稳定哈希。

---

# 11. 详细改动七：Decode 的 RoPE 位置改用 `seq.rope_pos`

原版：

```python
positions.append(len(seq) - 1)
```

nano-kvLLM：

```python
positions.append(seq.rope_pos)
```

这是本文件中最重要的正确性改造之一。

## 11.1 原版为什么可以使用 `len(seq)-1`

在普通推理中：

```text
Sequence 逻辑长度
=
当前 KV Cache 上下文长度
```

假设已经有 1,000 个 token，则当前最后一个 token 的位置是：

```text
999
```

所以：

```python
len(seq) - 1
```

就是正确 RoPE 位置。

## 11.2 压缩后为什么不再成立

假设模型逻辑上已经处理 1,500 个 token，但 KV 压缩后只保留 700 个：

```text
逻辑时间轴最后位置 = 1499
实际 KV 上下文长度 = 700
```

如果继续使用：

```python
len(seq) - 1
```

就会把下一个 Decode token 的位置错误地放到压缩后的短上下文位置附近。

这会导致：

- Query 使用错误的旋转角度；
- 保留 Key 与新 Query 的相对位置关系被破坏；
- 模型输出质量下降；
- 长上下文语义混乱。

## 11.3 `rope_pos` 的作用

`Sequence.append_token()` 每生成一个 token：

```python
rope_pos += 1
```

压缩发生时不回退。

因此：

```text
KV Cache 可以从 1500 压缩到 700，
rope_pos 仍然保持 1499。
```

下一个 token 继续使用真实逻辑位置。

---

# 12. 详细改动八：`context_lens` 继续使用压缩后的 `len(seq)`

nano-kvLLM：

```python
context_lens.append(len(seq))
```

代码注释明确指出：

```text
len(seq) now denotes the actual compressed seq length
```

这说明当前设计将两个概念拆开：

| 输入 | 使用字段 | 含义 |
|---|---|---|
| `positions` | `seq.rope_pos` | token 在完整逻辑时间轴中的位置 |
| `context_lens` | `len(seq)` | 当前 Attention 实际可读取的 KV 数量 |

这是 KV Cache 压缩正确工作的核心。

可以概括为：

```text
RoPE 回答：这个 token 逻辑上是第几个？
context_lens 回答：当前物理缓存里有多少 KV 可读？
```

两者在普通推理中相等或高度相关，在压缩推理中必须分离。

---

# 13. 详细改动九：压缩后 Slot Mapping 依赖新的缓存长度

Decode 写入位置仍然是：

```python
seq.block_table[-1] * block_size
+ seq.last_block_num_tokens
- 1
```

其中：

```text
last_block_num_tokens
```

由压缩后的 `seq.num_tokens` 计算。

压缩事件经过 Scheduler 后：

```text
truncate_blocks(seq, keep_blocks)
seq.num_tokens = new_context_len
append_token(new_token)
```

下一轮 Decode 前，Scheduler 再调用：

```text
may_append(seq)
```

因此 ModelRunner 根据新的：

- `block_table`；
- `num_tokens`；
- 最后 Block 有效 token 数；

计算当前 `last_token` 应写入的物理槽位。

这条链路要求压缩事件满足：

```text
keep_blocks == ceil(new_context_len / block_size)
```

或至少保证保留 Block 容量足以容纳新的上下文。

否则 Slot Mapping 可能：

- 指向不存在的 Block；
- 写入错误偏移；
- 覆盖其他请求的 KV；
- 产生越界访问。

---

# 14. 详细改动十：周期性压缩调度

nano-kvLLM 在 `prepare_decode()` 中加入：

```python
self.decode_step_counter += 1
is_compress_step = (
    self.decode_step_counter
    % self.config.kv_compress_period
    == 0
)
```

## 14.1 为什么不是每个 Decode Step 都压缩

压缩本身可能包含：

- 计算 Query/Key 重要性；
- Token 选择；
- KV Gather/Compact；
- Block 内容搬移；
- 同步；
- 生成事件；
- 更新 Block Table。

如果每生成一个 token 都执行，额外开销可能超过节省的 Attention 成本。

周期性压缩实现了折中：

```text
连续生成一段 token
→ 尾部逐渐增长
→ 到固定周期再集中压缩
```

## 14.2 全局周期的特征

该周期是 ModelRunner 级别，而不是 Sequence 级别。

优点：

- 实现简单；
- 同一 Batch 中可统一压缩；
- 方便控制单步延迟尖峰。

缺点：

- 新加入的请求与老请求共享同一全局时钟；
- 某条请求达到窗口阈值后，仍需等到下一个全局压缩步；
- 不同请求的压缩间隔并不完全相同。

---

# 15. 详细改动十一：压缩候选条件

压缩候选需要满足：

```python
seq.tail_uncompressed_len >= window_tokens
and
full_blocks >= window_blocks
```

其中：

```python
window_tokens = window_blocks * block_size
full_blocks = current_context_len // block_size
```

## 15.1 `tail_uncompressed_len`

表示最近一次压缩后新增长度。

只有新增长度达到一个完整窗口，才值得再次压缩。

这避免：

```text
刚压缩完又立即压缩
```

## 15.2 `full_blocks >= window_blocks`

要求当前上下文至少拥有足够多的完整 Blocks。

这保证压缩窗口在物理 Block 层面有意义，避免对过短请求压缩。

## 15.3 为什么同时需要两个条件

假设请求总上下文很长，但刚刚压缩过：

```text
full_blocks 很多
tail_uncompressed_len 很小
```

不应立即再次压缩。

反之，请求新增长度很多，但总上下文仍不足一个窗口，也不适合执行压缩。

---

# 16. 详细改动十二：Block 边界保护

代码：

```python
if tail_len == B - 1:
    continue
```

即当：

```text
current_context_len % block_size == block_size - 1
```

时，本轮不选择该请求压缩。

从本文件只能确认这是一个 Block 边界保护条件，具体原因需要结合 Attention 压缩实现判断。

可能目的是避免：

- 当前 Decode 写入与尾 Block 即将填满的状态冲突；
- 压缩后 Block 数和新 token 写入位置发生边界歧义；
- 下一轮 `may_append()` 的哈希或新 Block 分配状态不一致；
- 压缩搬移时处理只剩一个槽位的尾块。

该条件应在代码中增加明确注释，否则很难验证是否存在 Off-by-One 问题。

---

# 17. 详细改动十三：每个压缩步只选择部分请求

代码：

```python
selected = candidates[:topk]
```

随后保存：

```python
selected_batch_indices
selected_seq_ids
```

这里的 `kv_compress_topk` 实际含义是：

```text
每个压缩步最多选择多少条 Sequence
```

不是常见的“每条 Sequence 保留 Top-K token”。

建议配置命名更明确，例如：

```text
kv_compress_max_seqs_per_step
```

## 17.1 为什么限制请求数量

同一 Batch 中同时压缩太多请求，可能造成：

- 单步 TPOT 突增；
- 大量 KV 数据搬移；
- 更多同步；
- Attention 内核延迟不稳定；
- GPU 临时内存压力。

因此限制每步压缩请求数，可以把压缩成本摊到多个 Decode Step。

## 17.2 选择顺序

当前直接取候选列表前 `topk` 条，即按 Batch 顺序选择。

这是一种简单策略。若需要更稳定的收益，可以按以下指标排序：

- 当前 KV Block 数最多；
- `tail_uncompressed_len` 最大；
- 预计可释放 Block 最多；
- 请求剩余生成长度；
- 等待时间或公平性。

---

# 18. 详细改动十四：通过 Context 广播压缩决策

nano-kvLLM 为 Context 设置：

```python
ctx.is_compress_step
ctx.compress_selected_batch_indices
ctx.compress_selected_seq_ids
ctx.compress_base_context_lens
```

这些字段使每个 Attention 层能够知道：

```text
本轮是否允许压缩；
Batch 中哪些请求需要压缩；
这些请求的身份是什么；
压缩前的上下文长度是多少。
```

## 18.1 为什么使用 Batch Index 和 Seq ID 两种身份

- `batch_index`：方便在 GPU Batch Tensor 中定位请求；
- `seq_id`：提供稳定的请求身份，便于生成事件和跨模块校验。

## 18.2 `compress_base_context_lens`

```python
context_lens.clone()
```

保存压缩前的基础长度。

Attention 层在压缩过程中可能修改当前上下文信息，因此需要保留原始长度，用于：

- 计算压缩前后差值；
- 生成 `new_context_len`；
- 计算释放 Block 数；
- 构造事件；
- 多层之间共享统一基线。

---

# 19. 压缩为什么放在 Attention 层执行

代码注释表明，压缩计划是：

```text
当前 Decode token 写入 KV Cache
→ 在 Flash Attention 读取 KV 之前
→ 紧凑/压缩当前 KV
→ 使用压缩后的上下文计算 Attention
```

这通常需要访问：

- 当前层的 K Cache；
- 当前层的 V Cache；
- 当前 Query；
- 可能的 Attention Score 或 Query Window；
- 当前层实际 KV Tensor 布局。

这些数据都位于 Attention 层，因此 ModelRunner 只负责“做决策和广播”，真正的数据压缩必须在 Attention Forward 中执行。

---

# 20. 跨层压缩必须满足统一布局约束

整个模型的 `Sequence.block_table` 只有一份，所有 Transformer 层共享相同的逻辑 Block 布局。

因此，如果各 Attention 层独立压缩，必须保证：

```text
每层最终保留相同数量和相同逻辑位置的 token。
```

否则：

```text
第 1 层 Block 0 对应 token A
第 2 层 Block 0 对应 token B
```

而上层只有一份 `block_table` 和 `context_lens`，无法正确描述不同层布局。

这意味着压缩选择策略通常需要：

- 使用跨层共享的 token 索引；
- 由某一层或统一管理器决定保留位置；
- 所有层按相同索引 Compact；
- 只允许层内 KV 数值不同，布局不能不同。

`compress_selected_seq_ids` 只决定压缩哪些请求，并不能单独保证各层保留 token 一致。该一致性需要在 Attention 或 Query Window Manager 中实现。

---

# 21. 详细改动十五：CUDA Graph 与动态压缩

原版：

```python
if is_prefill
or enforce_eager
or batch_size > 512:
    eager
else:
    CUDA Graph replay
```

nano-kvLLM 新增：

```python
need_eager_decode = (
    not is_prefill
    and getattr(context, "compress_any", False)
)
```

并把它加入 eager 条件。

设计意图是：

```text
普通 Decode → CUDA Graph
发生压缩的 Decode → eager
```

这是合理方向，因为压缩通常包含：

- 动态请求选择；
- 动态保留长度；
- 数据依赖分支；
- KV Gather/Compact；
- Python Context 中的动态列表；
- 不固定的压缩事件。

这些操作很难直接放入固定 CUDA Graph。

---

# 22. 当前 CUDA Graph 切换逻辑存在明显风险

在 `prepare_decode()` 中，代码先执行：

```python
ctx.compress_need_mask = None
ctx.compress_any = False
```

随后虽然计算了：

```text
is_compress_step
selected_batch_indices
```

但没有在这里把：

```python
ctx.compress_any
```

设置为 `True`。

`run_model()` 在模型 Forward 之前读取：

```python
need_eager_decode = context.compress_any
```

此时 Attention 层尚未执行，不可能在 Forward 内部先把该字段改为 True 再影响已经做出的 eager/graph 选择。

因此，基于本文件显示的执行顺序：

```text
prepare_decode：compress_any = False
        ↓
run_model：读取 compress_any
        ↓
仍选择 CUDA Graph
        ↓
Attention Forward 才有机会发现压缩请求
```

这意味着压缩步很可能没有按设计切换到 eager。

## 22.1 更合理的判断

应在 `prepare_decode()` 完成候选选择后直接设置：

```python
ctx.compress_any = bool(selected_batch_indices)
```

或者在 `run_model()` 中判断：

```python
need_eager_decode = (
    not is_prefill
    and bool(context.compress_selected_batch_indices)
)
```

## 22.2 为什么这是关键问题

CUDA Graph Replay 不会重新执行普通 Python 控制流。压缩选择信息目前是 Python List 和 Context 属性，如果没有显式 Tensor 化并纳入 Graph，动态压缩逻辑通常无法在 Replay 中正确变化。

可能结果包括：

- 压缩根本没有执行；
- Graph 捕获时固定的分支被重复使用；
- `compression_events` 为空；
- 动态 KV Compact 没有进入 Graph；
- 不同压缩步行为不一致。

除非 Attention 实现使用了额外的 Graph 安全 Tensor 控制路径，否则当前代码需要修正 eager 触发条件。

---

# 23. CUDA Graph 捕获还需要额外验证

`capture_cudagraph()` 基本沿用原版，只设置：

```python
slot_mapping
context_lens
block_tables
```

没有显式设置：

```text
is_compress_step
selected_batch_indices
compress_base_context_lens
compression_events
```

因此有两种合理架构：

## 架构 A：压缩步全部 eager

普通 Graph 不包含压缩逻辑，压缩步跳过 Graph。

这是最简单、最安全的实现。

## 架构 B：Graph 内支持动态压缩

需要把压缩控制状态改成固定形状 GPU Tensor，并捕获：

- 选择 Mask；
- 每条请求的压缩长度；
- Compact 操作；
- 事件输出 Buffer。

当前文件更接近架构 A，但 `compress_any` 判断可能没有正确实现。

---

# 24. 详细改动十六：`run()` 返回压缩事件

原版：

```python
return token_ids
```

nano-kvLLM：

```python
return token_ids, compression_events
```

模型执行结束后、`reset_context()` 之前：

```python
ctx = get_context()
compression_events = ctx.compression_events
```

## 24.1 为什么必须在 reset 前收集

`Context` 是单轮模型执行的临时状态。

`reset_context()` 后：

- 当前 Batch 的 Slot Mapping 被清除；
- Context Lens 被清除；
- 压缩选择信息被清除；
- Attention 层写入的事件也会消失。

所以压缩事件必须在 Reset 前复制出来。

## 24.2 为什么只在 Rank 0 返回

在 Tensor Parallel 模式中：

- 所有 Rank 都执行自己的模型分片；
- 只有 Rank 0 负责采样和主控制；
- Scheduler 只存在于主进程控制链中。

因此只有 Rank 0 需要把事件返回 LLMEngine。

其他 Rank 仍必须对本地 KV Cache 执行完全一致的压缩，否则不同 GPU 的 KV 布局会失去同步。

---

# 25. Tensor Parallel 下的压缩一致性要求

每个 Rank 拥有一部分 KV Heads，但共享：

- Sequence 顺序；
- Block Table；
- Context Lens；
- 压缩后的长度。

因此所有 Rank 必须：

```text
在同一 Decode Step
选择相同 Sequence
保留相同 token 位置
得到相同 new_context_len
执行相同 Block Compact 布局
```

当前 `decode_step_counter`、Batch Index 和 Sequence 状态会在各 Rank 上独立计算，但只要输入完全一致，结果应一致。

真正需要注意的是：

> 如果压缩选择依据各 Rank 本地 KV Head 的 Attention Score，不同 Rank 可能得到不同的重要 token 排名。

此时必须：

- 在 Rank 间聚合 Score；
- 由 Rank 0 决定保留索引并广播；
- 或采用所有 Rank 都能确定的统一策略。

否则 Scheduler 虽然只收到一份事件，不同 GPU 上的实际 KV 内容却可能不同。

这一点需要结合 Attention 压缩实现继续核查。

---

# 26. 详细改动十七：退出时尝试清理 Query Window Buffer

nano-kvLLM：

```python
if hasattr(self, "query_window_manager"):
    self.query_window_manager.buffers.clear()
```

这表明压缩算法可能维护跨 Decode Step 的查询窗口或统计 Buffer，例如：

- 最近若干 Query；
- Token 重要性累计值；
- Attention Score；
- 每个 Sequence 的压缩历史。

但从本文件看，`ModelRunner` 没有显式创建：

```python
self.query_window_manager
```

它可能：

- 在其他方法中动态挂载；
- 实际属于模型或 Attention；
- 当前清理代码没有真正访问到对象。

如果管理器实际位于：

```python
self.model
```

则这里的 `hasattr(self, ...)` 会始终为 False，清理逻辑无效。

需要继续检查管理器的创建位置。

---

# 27. AutoTokenizer 导入未使用

nano-kvLLM 新增：

```python
from transformers import AutoTokenizer
```

但本文件没有实例化或调用。

这不是压缩功能所需，应属于调试残留，可以删除。

---

# 28. 与前面四个文件的完整压缩链路

现在已经可以把五个核心文件串起来。

## 28.1 Sequence

保存：

```text
完整 token_ids
逻辑 rope_pos
压缩后当前 num_tokens
generated_completion_tokens
tail_uncompressed_len
block_table
```

## 28.2 Scheduler

调度 Decode，并在模型返回后：

```text
处理 compression_events
截断 Blocks
更新 num_tokens
重置 tail_uncompressed_len
```

## 28.3 ModelRunner

在 Decode 前：

```text
使用 rope_pos 构造 Position
使用压缩后 len(seq) 构造 Context Lens
选择本轮压缩请求
把压缩决策写入 Context
```

模型运行后：

```text
收集 compression_events
```

## 28.4 Attention / Model

根据 Context：

```text
写入当前 token KV
执行重要性计算
Compact/Compress KV
记录 new_context_len 和 keep_blocks
```

## 28.5 BlockManager

根据 Scheduler 提交的事件：

```text
truncate_blocks()
释放尾部物理 Blocks
```

完整路径：

```text
tail_uncompressed_len 达到窗口
        ↓
ModelRunner 在周期步选择 Sequence
        ↓
Context 广播压缩对象
        ↓
Attention 压缩每层 KV
        ↓
Context 记录 compression_events
        ↓
ModelRunner 返回事件
        ↓
LLMEngine 转交 Scheduler
        ↓
Scheduler 更新 Sequence
        ↓
BlockManager 回收物理 Blocks
```

---

# 29. 一次压缩 Decode Step 的完整示例

假设：

```text
block_size = 256
当前压缩后 Context Length = 2048
rope_pos = 4095
tail_uncompressed_len = 1024
window_blocks = 4
period 到期
```

## 29.1 Prepare Decode

ModelRunner 构造：

```text
input_id = 当前 last_token
position = 4095
context_len = 2048
slot_mapping = 当前尾 Block 的写入槽位
```

## 29.2 选择压缩

因为：

```text
tail_uncompressed_len >= 4 × 256
full_blocks >= 4
```

该 Sequence 成为候选。

Context 保存：

```text
selected_batch_index
seq_id
base_context_len = 2048
```

## 29.3 Attention Forward

每层 Attention：

```text
先写入当前 token 的 K/V
→ 根据统一选择结果 Compact KV
→ 假设压缩到 1280 token
→ 形成 compression_event
```

事件可能包含：

```text
new_context_len = 1280
keep_blocks = 5
tail_uncompressed_len_after = 0
```

## 29.4 ModelRunner 返回

```text
sampled token
compression_events
```

## 29.5 Scheduler 提交

```text
truncate_blocks(seq, 5)
seq.num_tokens = 1280
seq.tail_uncompressed_len = 0
seq.append_token(sampled_token)
```

此后：

```text
num_tokens = 1281
rope_pos = 4096
```

可以看到：

```text
物理上下文：2048 → 1280 → 1281
逻辑位置：4095 → 4096
```

压缩改变了 Attention 可读 KV 数量，但没有让逻辑时间倒退。

---

# 30. 当前实现中最值得关注的潜在问题

## 30.1 CUDA Graph eager 触发变量可能错误

这是本文件最重要的问题。

当前 `run_model()` 读取 `compress_any`，但该字段在 Forward 前被固定为 False。

建议直接根据：

```text
compress_selected_batch_indices
```

判断压缩步并切回 eager。

---

## 30.2 Warmup 可能生成空 Batch

当：

```text
max_model_len > max_num_batched_tokens
```

时，`num_seqs` 可能为 0。

应恢复原版 `seq_len=min(...)` 的保护。

---

## 30.3 Warmup 长度可能过大

直接使用 `max_model_len` 可能导致不必要的长序列 Warmup和显存峰值。

---

## 30.4 SharedMemory 没有长度检查

即使扩展到 16 MiB，长上下文仍可能溢出。

---

## 30.5 `kv_compress_topk` 命名含义不清

当前表示每步最多压缩的 Sequence 数，而不是保留 token 数。

---

## 30.6 Batch 顺序选择可能不够公平

候选直接取前 N 条。可以根据预计释放 Block 数或等待时间排序。

---

## 30.7 Block 边界跳过条件缺少解释

`tail_len == B-1` 的准确不变量需要结合 Attention Compact 实现验证。

---

## 30.8 多层压缩布局必须完全一致

否则单一 Block Table 无法描述各层不同布局。

---

## 30.9 Tensor Parallel Rank 必须统一保留索引

若各 Rank 根据本地 Attention Score 独立选择 token，可能产生跨 GPU KV 布局不一致。

---

## 30.10 Query Window Manager 清理对象可能不在 ModelRunner

应确认实际对象归属。

---

## 30.11 CUDA Graph 捕获上下文没有压缩控制 Tensor

这进一步说明压缩步应使用 eager，除非另有 Graph-safe 实现。

---

# 31. 哪些改动真正服务于 KV Cache 压缩

## 31.1 核心正确性改动

- `positions` 使用 `seq.rope_pos`；
- `context_lens` 使用压缩后实际长度；
- 将完整 Config 传入模型；
- 压缩周期和候选选择；
- Context 中传递压缩对象和基础长度；
- 模型运行后收集 `compression_events`；
- 返回 `(token_ids, compression_events)`。

## 31.2 压缩性能控制

- 周期性而非逐 token 压缩；
- 每步限制压缩请求数量；
- 尝试在压缩步切回 eager；
- Query Window Buffer 管理。

## 31.3 配套架构变化

- SharedMemory 扩大；
- Prefill 取消 Chunked 模式；
- Warmup 不再使用 `num_scheduled_tokens`；
- 使用 `torch_dtype`；
- 模型构造函数增加 Config。

## 31.4 非核心或调试残留

- 未使用的 AutoTokenizer；
- `#!!!` 注释；
- 可能无效的 Query Window 清理位置。

---

# 32. 初学者理解方式

可以把 ModelRunner 想成“GPU 推理现场的总指挥”。

原版总指挥负责：

```text
把请求整理成 GPU 输入；
告诉 Attention 去哪里写 KV；
告诉 Attention 去哪些 Blocks 读历史；
运行模型；
采样下一个 token。
```

nano-kvLLM 的总指挥还要负责：

```text
判断今天是否到了整理仓库的时间；
挑选哪些请求需要整理；
告诉每一层 Attention 采用相同的整理计划；
模型运行后收集整理报告；
把报告交给 Scheduler 和 BlockManager。
```

其中：

- `rope_pos` 是请求真实走过的时间；
- `context_lens` 是仓库当前实际保留的货物数量；
- `slot_mapping` 是新货物要放入的槽位；
- `compression_events` 是整理后释放货架的报告。

最关键的原则是：

> 仓库可以缩小，但请求真实经历过的时间不能倒退。

---

# 33. 最终总结

nano-kvLLM 对 `model_runner.py` 的核心改造，是把普通 GPU 模型执行器扩展成 KV Cache 压缩的运行时控制器。

它完成了三项关键工作。

## 第一：分离逻辑位置和物理上下文

```text
positions = seq.rope_pos
context_lens = len(seq)
```

压缩后：

- RoPE 仍沿完整生成时间轴递增；
- Attention 只读取压缩后保留的 KV。

这是压缩正确性的基础。

## 第二：建立周期性压缩调度

```text
全局 Decode Step 到达 period
→ 检查 tail_uncompressed_len
→ 检查完整窗口 Blocks
→ 每步选择部分 Sequence
→ 将决策写入 Context
```

它控制压缩频率和单步额外开销。

## 第三：把 Attention 压缩结果送回系统资源层

```text
Attention 产生 compression_events
→ ModelRunner 在 reset_context 前收集
→ LLMEngine 传递
→ Scheduler 更新 Sequence
→ BlockManager 回收 Blocks
```

这样压缩才真正影响全局 KV Cache Block Pool。

当前最需要优先修正或验证的问题是 CUDA Graph 路径：

```text
compress_any 在 Forward 前被设为 False，
run_model 又在 Forward 前依赖它判断是否切换 eager。
```

从本文件的执行顺序看，压缩步可能仍进入 CUDA Graph Replay，导致动态压缩逻辑无法正确执行。更合理的实现应根据已经计算出的：

```text
compress_selected_batch_indices
```

直接决定压缩步使用 eager。

后续最值得继续对比的文件是：

```text
utils/context.py
layers/attention.py
models/qwen3.py
压缩选择或 query_window_manager 相关文件
```

需要重点确认：

1. Attention 如何根据 Context 选择和搬移 KV；
2. 所有层是否使用同一组保留 token 索引；
3. Tensor Parallel 各 Rank 如何同步压缩选择；
4. `compression_events` 在哪一层创建；
5. `new_context_len` 与 `keep_blocks` 如何计算；
6. 压缩步是否真的绕过 CUDA Graph；
7. 被压缩后的 KV 是否仍保持正确 RoPE 语义和 Block 布局。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-kvllm|模块-nano-kvllm]]

%% 项目关联导航：结束 %%
