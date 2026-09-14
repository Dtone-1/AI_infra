# nano-vLLM 与 nano-kvLLM：`scheduler.py` 源码对比分析

## 1. 对比对象与核心结论

本次对比的两个文件分别是：

- 原版 nano-vLLM 的 `scheduler.py`
- nano-kvLLM 的 `scheduler.py`

`Scheduler` 是推理引擎的中心控制器之一。它位于请求状态、KV Cache Block 管理和模型执行之间，负责回答三个问题：

1. 本轮应该运行哪些请求？
2. 这些请求能否获得足够的 KV Cache 空间？
3. 模型运行结束后，如何更新请求状态并释放或调整缓存？

原版 nano-vLLM 的 Scheduler 主要围绕以下能力设计：

```text
连续批处理
+ Prefix Cache 复用
+ Chunked Prefill
+ Decode 阶段 Block 扩容
+ 显存不足时抢占并重算
```

nano-kvLLM 的 Scheduler 则进一步承担：

```text
接收 compression_events
+ 按请求定位压缩结果
+ 截断物理 KV Block
+ 重写当前缓存上下文长度
+ 重置未压缩尾部长度
+ 保持压缩后继续 Decode
```

因此，nano-kvLLM 中 Scheduler 的核心变化可以概括为：

> Scheduler 不再只是“决定谁运行”，还负责把模型侧产生的 KV Cache 压缩结果提交到请求状态和 Block Manager。

---

# 2. 整体变化概览

| 改动方向 | 原版 nano-vLLM | nano-kvLLM | 工程意义 |
|---|---|---|---|
| 包命名空间 | `nanovllm` | `nanokvllm` | 使用独立改造工程 |
| Prefill 策略 | 支持 Prefix Cache 和 Chunked Prefill | 只调度可一次完整执行的请求 | 简化 Prefill，将重点放在 Decode 压缩 |
| Prefill 缓存检查 | `can_allocate()` 返回可复用缓存块数 | `can_allocate()` 只返回布尔值 | Block Manager 接口和职责发生变化 |
| 单轮调度 token 数 | 使用 `num_scheduled_tokens` | 删除该状态 | 不再支持原版细粒度分块 Prefill协议 |
| Prefill/Decode 标记 | `seq.is_prefill` | 删除 | 阶段由 Scheduler 的返回值和执行路径决定 |
| Decode Block 申请 | `may_append(seq)` | 保留 | 每轮 Decode 前仍需保证写入空间 |
| 抢占恢复 | 设置 `is_prefill=True` 后释放缓存 | 恢复逻辑序列长度后释放缓存 | 压缩后重算必须恢复完整 token 历史 |
| 后处理输入 | `token_ids + is_prefill` | `token_ids + compression_events` | 后处理从阶段感知变为压缩事件感知 |
| 压缩事件处理 | 无 | 去重、定位请求、截断 Block、更新长度 | 建立压缩结果提交链路 |
| 停止条件 | `num_completion_tokens == max_tokens` | `generated_completion_tokens >= max_tokens` | 停止条件与被压缩缓存长度解耦 |
| Tokenizer | 无 | Scheduler 内加载 tokenizer | 目前仅用于注释调试，生产路径无必要 |

最重要的结构变化是：

```text
原版：
调度 → 模型运行 → 更新缓存进度 → 追加 token

nano-kvLLM：
调度 → 模型运行/压缩 → 提交压缩事件 → 截断缓存 → 追加 token
```

---

# 3. 原版 Scheduler 在推理系统中的作用

原版 Scheduler 内部维护两个队列：

```python
self.waiting
self.running
```

它们构成请求生命周期：

```text
新请求
  ↓
WAITING 队列
  ↓ Prefill 完成
RUNNING 队列
  ↓ Decode 多轮运行
FINISHED
```

每一轮 `schedule()` 优先尝试 Prefill；如果没有可运行的等待请求，再调度 Decode。

这一优先级意味着：

```text
只要当前能够接纳新的 Prefill 请求，
Scheduler 就先运行 Prefill；
否则再运行已有请求的 Decode。
```

原版 `postprocess()` 则负责：

- 标记本轮 KV 已经写入；
- 处理 Chunked Prefill 是否完成；
- 追加采样出的 token；
- 判断 EOS 或达到最大生成长度；
- 完成后释放 KV Block。

---

# 4. nano-kvLLM Scheduler 的总体职责变化

引入 KV Cache 压缩后，模型执行不再只返回采样 token，而可能返回：

```text
token_ids
compression_events
```

压缩事件描述某条序列的 KV Cache 在本轮执行后发生了什么，例如：

- 压缩后上下文还剩多少 token；
- 应保留多少物理 Block；
- 未压缩尾部还剩多长；
- 事件对应批次中的哪条请求。

Scheduler 必须把这些信息应用到主进程中的真实 Sequence 和 Block Manager。

因此后处理链路变为：

```text
ModelRunner
   │
   ├── token_ids
   └── compression_events
             ↓
Scheduler.postprocess()
             ↓
按 batch_index 找到 Sequence
             ↓
BlockManager.truncate_blocks()
             ↓
更新当前缓存长度和压缩尾部
             ↓
追加新生成 token
```

这使 Scheduler 成为压缩机制的“状态提交点”。

---

# 5. 详细改动一：Prefill 调度从复杂分块策略改为完整请求调度

## 5.1 原版逻辑

原版 Prefill 会计算本轮剩余 token 预算：

```python
remaining = self.max_num_batched_tokens - num_batched_tokens
```

随后分两种情况：

### 首次调度

```python
num_cached_blocks = self.block_manager.can_allocate(seq)
```

`can_allocate()` 不只是返回能否分配，还返回可以复用多少个缓存 Block。

然后计算真正需要执行的 Prompt token：

```python
num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
```

这体现了 Prefix Cache：

```text
Prompt 总长度
- 已命中的前缀缓存
= 实际需要 Prefill 的 token 数
```

### 已经执行过部分 Prefill

如果 `seq.block_table` 非空：

```python
num_tokens = seq.num_tokens - seq.num_cached_tokens
```

说明该 Sequence 可能经历过 Chunked Prefill，本轮只需继续处理尚未缓存的部分。

### 分块调度

原版允许第一条请求进行 Chunked Prefill：

```python
seq.num_scheduled_tokens = min(num_tokens, remaining)
```

因此即使长 Prompt 超过本轮 token 预算，也可以先执行一部分。

---

## 5.2 nano-kvLLM 逻辑

nano-kvLLM 的条件变为：

```python
if num_batched_tokens + len(seq) > self.max_num_batched_tokens \
        or not self.block_manager.can_allocate(seq):
    break
```

满足条件后：

```python
self.block_manager.allocate(seq)
num_batched_tokens += len(seq) - seq.num_cached_tokens
```

它不再设置：

```text
num_scheduled_tokens
is_prefill
```

也不再将一条 Prompt 拆成多轮 Prefill。

换句话说，一条请求只有在以下条件同时满足时才进入本轮 Prefill：

```text
完整请求可以放进 token budget
且
Block Manager 能为它分配缓存
```

否则它继续停留在 waiting 队首，后面的请求也不会越过它。

---

## 5.3 这一变化的意义

这项改动显著简化了 Prefill 状态机：

```text
原版：
未执行 → 部分 Prefill → 继续 Prefill → 完成

nano-kvLLM：
未执行 → 一次完整 Prefill → 完成
```

这样做可能是因为 nano-kvLLM 当前实验重点是 Decode 阶段的 KV Cache 动态压缩，而不是保留原版所有调度优化。

优点：

- 更容易保证压缩状态与序列状态一致；
- 不需要维护每轮 `num_scheduled_tokens`；
- 避免 Chunked Prefill 与压缩逻辑同时存在时的复杂边界；
- 更容易定位性能和正确性问题。

代价：

- 长 Prompt 必须完整放入单轮 token budget；
- 可能降低长上下文请求的接纳能力；
- 失去原版 Chunked Prefill 带来的调度灵活性；
- waiting 队首的超长请求可能阻塞后续短请求；
- 当前 Scheduler 的可扩展性弱于原版。

因此，不能把这一变化理解成 KV 压缩天然要求取消 Chunked Prefill。更准确地说：

> 这是 nano-kvLLM 当前实现为了降低系统复杂度而做出的工程取舍。

---

# 6. 详细改动二：Prefix Cache 接口被弱化或移除

原版：

```python
num_cached_blocks = self.block_manager.can_allocate(seq)
```

返回值包含：

```text
-1：不能分配
其他值：可复用的前缀缓存 Block 数量
```

随后：

```python
self.block_manager.allocate(seq, num_cached_blocks)
```

这说明原版 Scheduler 明确参与 Prefix Cache 命中数量的计算和分配。

nano-kvLLM：

```python
self.block_manager.can_allocate(seq)
self.block_manager.allocate(seq)
```

这里 `can_allocate()` 看起来只返回布尔值，`allocate()` 也不再接收 `num_cached_blocks`。

基于本文件可以判断：

- 原版的 Prefix Cache 协议已不再由 Scheduler 显式驱动；
- Block Manager 的接口已经被重写；
- nano-kvLLM 更关注“当前请求能否分配物理 KV 空间”，而不是前缀哈希复用。

但是否完全删除 Prefix Cache，还需要继续查看 nano-kvLLM 的 `block_manager.py`。仅凭 Scheduler 能确定的是：

> 原版 Prefix Cache 的调用协议在这里已经消失。

---

# 7. 详细改动三：删除 `block_size` 成员

原版 Scheduler 保存：

```python
self.block_size = config.kvcache_block_size
```

主要用于：

```python
num_cached_blocks * self.block_size
```

计算 Prefix Cache 已覆盖多少 token。

nano-kvLLM 删除该字段，原因是当前 Scheduler 不再直接进行缓存块数到 token 数的换算。

这说明 Block 细节更集中到 Block Manager 内部，Scheduler 只通过以下高层接口操作：

```text
can_allocate
allocate
can_append
may_append
truncate_blocks
deallocate
```

这是一个更清晰的职责分层方向：

```text
Scheduler：决定运行和提交状态
Block Manager：决定物理 Block 如何变化
```

不过 nano-kvLLM 又在 `postprocess()` 中直接读取 `keep_blocks` 并调用 `truncate_blocks()`，因此 Scheduler 仍然知道压缩后的 Block 数量，而不是完全对 Block 布局无感知。

---

# 8. 详细改动四：Decode 调度删除每轮 token 状态写入

原版 Decode 成功调度后：

```python
seq.num_scheduled_tokens = 1
seq.is_prefill = False
self.block_manager.may_append(seq)
```

nano-kvLLM：

```python
self.block_manager.may_append(seq)
scheduled_seqs.append(seq)
```

删除这些字段后，Decode 语义被简化为：

```text
只要 Sequence 被加入 scheduled_seqs，
本轮就为它生成一个 token。
```

这与 nano-kvLLM 的 `Sequence.append_token()` 配合：

```text
每次 append_token()
→ generated_completion_tokens += 1
→ rope_pos += 1
→ tail_uncompressed_len += 1
```

原版依赖显式的 `num_scheduled_tokens=1` 来统一 Prefill 和 Decode 后处理；nano-kvLLM 则直接根据调用路径处理，不再需要这个中间状态。

---

# 9. 详细改动五：抢占时必须恢复逻辑序列长度

## 9.1 原版抢占

```python
seq.status = SequenceStatus.WAITING
seq.is_prefill = True
self.block_manager.deallocate(seq)
self.waiting.appendleft(seq)
```

原版被抢占后：

- 释放全部 KV Cache；
- 回到 waiting；
- 后续重新 Prefill 整条序列。

由于原版 `num_tokens` 始终表示逻辑 token 总数，因此无需恢复。

---

## 9.2 nano-kvLLM 抢占

nano-kvLLM 新增：

```python
seq.num_tokens = len(seq.token_ids)
if len(seq.token_ids) > 0:
    seq.last_token = seq.token_ids[-1]
```

随后才释放缓存并重新进入 waiting。

代码注释明确写道：

```text
restore logical sequence length before recompute
```

这揭示了 nano-kvLLM 中一个极其重要的设计：

> 压缩发生后，`seq.num_tokens` 在运行阶段可能不再表示完整逻辑 token 数，而是表示当前 KV Cache 的上下文长度。

例如：

```text
完整 token_ids 长度 = 1500
压缩后 KV 上下文长度 = 700
seq.num_tokens = 700
```

此时如果请求被抢占并释放全部 KV Cache，后续需要重新 Prefill 完整历史，必须恢复：

```text
seq.num_tokens = len(seq.token_ids) = 1500
```

否则重算只会处理 700 个 token，丢失逻辑历史。

---

## 9.3 为什么还要恢复 `last_token`

压缩期间，`token_ids` 保存完整逻辑历史，而 `last_token` 应始终指向最新生成 token。

抢占前再次执行：

```python
seq.last_token = seq.token_ids[-1]
```

可以避免压缩或反序列化流程导致 `last_token` 与完整 token 历史失去同步。

---

## 9.4 工程意义

该修改证明 nano-kvLLM 采用的是一种“字段复用”设计：

```text
运行/压缩阶段：
num_tokens → 当前缓存上下文长度

抢占/重算阶段：
num_tokens → 完整逻辑序列长度
```

这种设计减少了新增字段，但也增加了理解和维护难度。

更清晰的生产级设计通常会明确拆分：

```text
logical_num_tokens
cached_context_len
```

而不是让同一个 `num_tokens` 在不同阶段拥有不同语义。

---

# 10. 详细改动六：后处理接口从阶段感知变为压缩事件感知

原版：

```python
postprocess(seqs, token_ids, is_prefill)
```

nano-kvLLM：

```python
postprocess(seqs, token_ids, compression_events=None)
```

这一变化与 `llm_engine.py` 中的改动直接对应。

原版 Scheduler 需要知道本轮是不是 Prefill，因为：

- Prefill 可能只执行一个 Chunk；
- 只有完整 Prefill 后才能追加采样 token；
- 要更新 `num_cached_tokens` 和 `num_scheduled_tokens`。

nano-kvLLM 已取消 Chunked Prefill，因此不再需要 `is_prefill` 来决定是否继续等待。

新的核心问题变成：

```text
本轮是否发生了 KV Cache 压缩？
如果发生，应该怎样修改主进程状态？
```

所以 `compression_events` 替代了 `is_prefill` 成为后处理的关键输入。

---

# 11. 详细改动七：压缩事件按 `batch_index` 映射到 Sequence

nano-kvLLM：

```python
bidx = ev["batch_index"]
if 0 <= bidx < len(seqs):
    dedup[bidx] = ev
```

`batch_index` 表示压缩事件属于本轮批次中的第几条 Sequence。

例如：

```text
seqs = [seq_A, seq_B, seq_C]

event.batch_index = 1
→ 事件属于 seq_B
```

Scheduler 通过：

```python
seq = seqs[bidx]
```

找到主进程中的真实请求对象。

这是压缩事件从模型批处理维度返回请求维度的关键映射。

为什么不能只依赖事件顺序？

- 并不是每条 Sequence 都会触发压缩；
- 一个批次中可能只有部分请求达到压缩阈值；
- 某条 Sequence 可能产生多个中间事件；
- 压缩事件数量不一定等于批大小。

因此必须携带明确索引或 `seq_id`。

当前实现使用 `batch_index`，而上一轮 `sequence.py` 又新增了 `seq_id` 序列化。说明系统可能同时具备两种身份信息，但本文件实际使用的是 `batch_index`。

---

# 12. 详细改动八：对同一请求的压缩事件去重

代码：

```python
dedup = {}
for ev in compression_events:
    bidx = ev["batch_index"]
    if 0 <= bidx < len(seqs):
        dedup[bidx] = ev
```

字典后写覆盖前写，因此：

```text
同一个 batch_index 出现多个事件
→ 只保留最后一个事件
```

注释也说明：

```text
only keep the last event for each seq
```

## 12.1 为什么需要去重

可能的场景包括：

- 模型内部多个层都上报压缩；
- 同一 Sequence 在一次运行中发生多个阶段性压缩；
- 不同组件产生重复事件；
- 只希望提交最终压缩结果。

如果每个事件都调用一次 `truncate_blocks()`，可能重复释放 Block 或反复修改长度。因此保留最终状态更安全。

## 12.2 该设计隐含的前提

最后一个事件必须代表该 Sequence 的最终压缩状态。

如果事件顺序不稳定，或者不同事件描述不同层的独立缓存，那么简单按 Sequence 去重可能丢失信息。

因此需要结合 ModelRunner 的事件定义确认：

```text
事件是“全模型统一缓存状态”
还是
“某一层的局部压缩状态”
```

如果所有层共享同一 Block Table，当前设计合理；如果每层独立压缩，则事件模型需要更细粒度的标识。

---

# 13. 详细改动九：通过 `truncate_blocks()` 回收物理缓存

每条有效压缩事件包含：

```python
keep_blocks = ev["keep_blocks"]
```

随后：

```python
self.block_manager.truncate_blocks(seq, keep_blocks)
```

这一步是 Scheduler 层真正执行显存回收的接口。

可以理解为：

```text
压缩算法决定保留哪些历史 KV
        ↓
模型侧生成 keep_blocks
        ↓
Scheduler 提交结果
        ↓
Block Manager 截断 seq.block_table
        ↓
多余 Block 返回空闲池
```

压缩算法只有在这一步之后，才真正转化为可供其他请求使用的显存容量。

如果只在张量内部做稀疏选择，却不释放 Block：

```text
逻辑上压缩了
但显存池没有新增可分配空间
```

那么对并发容量没有实际帮助。

因此 `truncate_blocks()` 是连接“算法压缩率”和“系统显存收益”的关键操作。

具体如何截断、是否搬移 KV、如何处理尾块，需要继续查看 `block_manager.py` 和模型侧缓存布局。

---

# 14. 详细改动十：用 `new_context_len` 重写 `seq.num_tokens`

压缩事件提供：

```python
new_context_len = ev["new_context_len"]
```

Scheduler 执行：

```python
seq.num_tokens = new_context_len
```

代码注释明确指出：

```text
num_tokens still denotes current cache length in your current design
```

这说明当前 nano-kvLLM 将 `num_tokens` 重新定义为：

```text
当前模型继续 Decode 时可见的缓存上下文长度
```

而不是始终表示：

```text
Prompt + 已生成 token 的完整逻辑长度
```

---

## 14.1 压缩后的长度关系

压缩后可能出现：

```text
len(seq.token_ids) = 1500
seq.num_tokens = 700
seq.generated_completion_tokens = 500
seq.rope_pos = 1499
```

各字段分别表示：

| 字段 | 含义 |
|---|---|
| `len(token_ids)` | 完整逻辑 token 历史 |
| `num_tokens` | 当前物理/有效缓存上下文长度 |
| `generated_completion_tokens` | 实际生成 token 数 |
| `rope_pos` | 逻辑位置时间轴 |
| `block_table` | 当前实际占用的物理 KV Blocks |

这是 nano-kvLLM 支持压缩的核心状态分离。

---

## 14.2 为什么压缩后仍能继续生成

随后 `seq.append_token(token_id)` 会执行：

```text
num_tokens += 1
rope_pos += 1
tail_uncompressed_len += 1
```

因此压缩后上下文从 700 开始继续增长：

```text
压缩后：700
下一轮：701
再下一轮：702
```

但 RoPE 位置则从完整逻辑位置继续：

```text
1499 → 1500 → 1501
```

这实现：

```text
物理上下文变短
但逻辑位置不倒退
```

---

## 14.3 这种设计的风险

`Sequence` 中仍然存在：

```python
@property
def num_completion_tokens(self):
    return self.num_tokens - self.num_prompt_tokens
```

当 `num_tokens` 被改成压缩后缓存长度时，这个属性可能变成负数或失真。

nano-kvLLM 因此改用：

```python
generated_completion_tokens
```

判断最大生成长度。

这也证明 `num_completion_tokens` 在压缩模式下已不能作为可靠停止指标。

不过同一个字段拥有两种语义仍容易导致其他模块误用。必须全面检查所有 `seq.num_tokens` 调用点，否则可能出现：

- 错误计算逻辑长度；
- 错误计算 Block 数；
- 错误切分 Prompt 和 Completion；
- 错误统计吞吐；
- 抢占重算前没有恢复完整长度。

---

# 15. 详细改动十一：重置 `tail_uncompressed_len`

代码：

```python
seq.tail_uncompressed_len = ev.get(
    "tail_uncompressed_len_after", 0
)
```

在 `Sequence.append_token()` 中，每生成一个 token：

```python
tail_uncompressed_len += 1
```

当压缩发生时，Scheduler 根据事件重置它。

典型周期：

```text
压缩刚完成
tail_uncompressed_len = 0

生成 1 个 token
tail_uncompressed_len = 1

生成到阈值
tail_uncompressed_len = N

触发压缩
tail_uncompressed_len = 0 或剩余未处理长度
```

为什么事件允许提供 `tail_uncompressed_len_after`，而不是强制清零？

因为某次压缩可能：

- 只压缩部分新尾部；
- 保留一小段最近 token 不参与压缩；
- 按滑动窗口保留保护区；
- 存在无法整块处理的剩余 token。

因此由压缩算法返回处理后的剩余尾部长度，比 Scheduler 固定设为 0 更通用。

---

# 16. 详细改动十二：删除原版缓存进度更新

原版 `postprocess()`：

```python
self.block_manager.hash_blocks(seq)
seq.num_cached_tokens += seq.num_scheduled_tokens
seq.num_scheduled_tokens = 0
```

nano-kvLLM 删除了这些代码。

## 16.1 删除 `hash_blocks(seq)`

原版通过哈希已完成的 Block，为 Prefix Cache 复用建立索引。

删除它进一步说明：

```text
nano-kvLLM 当前 Scheduler 不再维护原版 Prefix Cache 哈希链路。
```

## 16.2 删除 `num_cached_tokens` 累加

原版每轮根据本轮实际执行 token 数更新：

```text
已经进入 KV Cache 的 token 数
```

nano-kvLLM 不再在 Scheduler 中显式更新它。

可能原因包括：

- Block Manager 内部负责更新；
- ModelRunner 或其他模块负责更新；
- 当前实现不再依赖 `num_cached_tokens` 的精细进度；
- 完整 Prefill 后直接认为请求已缓存；
- `num_tokens` 被用作当前缓存长度的主要状态。

但仅从本文件看，存在需要核查的潜在一致性问题：

```text
压缩后修改了 num_tokens 和 block_table，
却没有显式修改 num_cached_tokens。
```

如果其他模块仍读取 `num_cached_tokens`，它可能保留压缩前的旧值。

因此后续对比 `block_manager.py` 时必须确认：

```text
truncate_blocks() 是否同步更新 seq.num_cached_tokens？
allocate() 和 may_append() 是否更新该字段？
```

如果没有，那么这是明显的状态一致性风险。

---

# 17. 详细改动十三：Prefill 不再单独跳过 token 追加

原版：

```python
if is_prefill and seq.num_cached_tokens < seq.num_tokens:
    continue
seq.append_token(token_id)
```

这段逻辑用于 Chunked Prefill：

```text
Prompt 尚未全部 Prefill
→ 本轮不把模型输出当作最终采样 token
```

nano-kvLLM 取消 Chunked Prefill，因此后处理直接：

```python
seq.append_token(token_id)
```

这隐含模型执行协议：

- 完整 Prefill 后会返回第一个生成 token；
- 每轮 Decode 后返回一个新 token；
- Scheduler 不需要区分两种情况。

这与 nano-kvLLM 的简化 Prefill 调度保持一致。

---

# 18. 详细改动十四：停止条件与缓存压缩解耦

原版：

```python
seq.num_completion_tokens == seq.max_tokens
```

而 `num_completion_tokens` 由：

```python
num_tokens - num_prompt_tokens
```

计算。

nano-kvLLM 改为：

```python
seq.generated_completion_tokens >= seq.max_tokens
```

这是必须的，因为压缩后：

```text
num_tokens = 当前缓存长度
```

它可能小于 `num_prompt_tokens`，不能再推导实际生成长度。

新的判断保证：

```text
无论历史 KV 被删除多少，
真正生成过的 token 数不会减少。
```

同时从 `==` 改为 `>=` 更稳健，可以避免某些状态跳跃或恢复异常导致越过阈值后无法结束。

EOS 判断保持不变：

```python
not seq.ignore_eos and token_id == self.eos
```

压缩不影响 EOS token 的语义。

---

# 19. 详细改动十五：Scheduler 中新增 Tokenizer，但当前未参与有效逻辑

nano-kvLLM 初始化：

```python
self.tokenizer = AutoTokenizer.from_pretrained(
    config.model,
    use_fast=True
)
```

实际只在被注释的调试代码中使用：

```python
# print(self.tokenizer.decode(seq.token_ids))
```

因此当前它会：

- 再次加载 tokenizer；
- 增加初始化耗时；
- 增加 CPU 内存占用；
- 与 LLMEngine 中已经加载的 tokenizer 重复。

但它不参与 Scheduler 的正式调度或压缩逻辑。

这是典型的调试残留，应在正式版本中删除，或通过依赖注入复用已有 tokenizer。

它不是 KV Cache 压缩的核心改动。

---

# 20. nano-kvLLM Scheduler 的完整工作流程

## 20.1 新请求进入

```text
Sequence
  ↓
Scheduler.add()
  ↓
waiting 队列
```

## 20.2 Prefill 调度

Scheduler 检查：

```text
完整 Sequence 是否能放入本轮 token budget？
Block Manager 是否能一次分配所需缓存？
```

如果可以：

```text
allocate()
→ status = RUNNING
→ waiting 移到 running
→ 本轮执行完整 Prefill
```

## 20.3 Decode 调度

对于 running 中的请求：

```text
检查是否还能追加一个 KV token
```

如果空间不足：

```text
抢占其他请求
或
抢占当前请求
```

如果空间足够：

```text
may_append()
→ 加入本轮 batch
```

## 20.4 模型执行

ModelRunner 返回：

```text
token_ids
compression_events（可选）
```

## 20.5 提交压缩事件

对于每条发生压缩的 Sequence：

```text
按 batch_index 定位
→ 保留最后一条事件
→ truncate_blocks()
→ num_tokens = new_context_len
→ 更新 tail_uncompressed_len
```

## 20.6 追加新 token

```text
append_token()
→ 保存 token
→ 当前缓存长度 +1
→ 生成计数 +1
→ RoPE 位置 +1
→ 未压缩尾部 +1
```

## 20.7 判断结束

满足以下任一条件：

```text
生成 EOS
或
generated_completion_tokens >= max_tokens
```

则：

```text
status = FINISHED
→ deallocate()
→ 从 running 删除
```

---

# 21. 与 `llm_engine.py`、`sequence.py` 的联动

前面三个文件的改动已经形成完整压缩控制链：

```text
Sequence
保存：
- generated_completion_tokens
- rope_pos
- tail_uncompressed_len
- 完整 token_ids
- 当前缓存 num_tokens
        ↓
ModelRunner
运行模型并产生：
- token_ids
- compression_events
        ↓
LLMEngine.step()
兼容拆分普通返回值或 tuple
        ↓
Scheduler.postprocess()
- 对事件去重
- 截断 Block
- 更新缓存长度
- 重置尾部
- 追加新 token
```

三个文件的职责可以概括为：

| 文件 | 压缩链路职责 |
|---|---|
| `sequence.py` | 保存逻辑历史和压缩状态 |
| `llm_engine.py` | 传递 token 与压缩事件 |
| `scheduler.py` | 提交压缩结果并更新系统资源 |

这说明 nano-kvLLM 的 KV Cache 压缩不是 Attention 层的孤立优化，而是跨越：

```text
模型执行
→ 引擎编排
→ 调度
→ 请求状态
→ 显存 Block 管理
```

的一套系统级改造。

---

# 22. 当前设计中最需要关注的状态语义

## 22.1 `token_ids`

表示完整逻辑 token 历史：

```text
Prompt + 所有已生成 token
```

压缩不能删除它，否则抢占后无法完整重算，也无法返回完整生成结果。

## 22.2 `num_tokens`

在当前设计中可能表示：

```text
未压缩/重算阶段：完整逻辑长度
压缩运行阶段：当前缓存上下文长度
```

这是双重语义字段。

## 22.3 `generated_completion_tokens`

表示实际 Decode 生成次数，不受压缩影响。

## 22.4 `rope_pos`

表示原始逻辑时间轴位置，不受缓存长度缩短影响。

## 22.5 `tail_uncompressed_len`

表示最近一次压缩后仍未处理的新增长度。

## 22.6 `block_table`

表示当前物理 KV Cache Block 布局，由压缩事件和 Block Manager 修改。

理解 Scheduler 的关键就是记住：

```text
完整 token 历史、逻辑位置、生成数量、物理缓存长度
已经不再是同一个数值。
```

---

# 23. 潜在问题与工程优化建议

## 23.1 `num_tokens` 双重语义风险较高

当前代码需要在抢占时手动恢复：

```python
seq.num_tokens = len(seq.token_ids)
```

这说明字段语义依赖运行阶段。

建议拆分为：

```text
logical_num_tokens
cache_context_len
```

这样可以减少其他模块误用。

---

## 23.2 `num_cached_tokens` 是否同步未知

Scheduler 在压缩后只修改：

```text
block_table
num_tokens
tail_uncompressed_len
```

未显式修改 `num_cached_tokens`。

必须核查 `truncate_blocks()` 是否负责同步，否则 Sequence 内部缓存统计可能不一致。

---

## 23.3 取消 Chunked Prefill 降低长 Prompt 能力

当前条件要求完整 `len(seq)` 不超过剩余预算。

建议未来恢复：

- Chunked Prefill；
- 对逻辑长度和物理缓存长度分别建模；
- 明确压缩仅作用于 Decode，或定义 Prefill 压缩边界。

---

## 23.4 队首阻塞

Prefill 遇到第一个无法分配的请求就 `break`，不会继续查看后续短请求。

这保证 FIFO 公平，但可能降低吞吐。可考虑：

- 有界跳过；
- 按长度分组；
- aging 防止饥饿；
- 预估压缩后可用空间。

---

## 23.5 压缩事件只校验 `batch_index`

无效索引会被静默忽略：

```python
if 0 <= bidx < len(seqs):
```

更稳健的实现应：

- 记录告警；
- 同时校验 `seq_id`；
- 检查事件数量和字段完整性；
- 防止错误事件作用于错误请求。

---

## 23.6 简单保留最后事件可能丢失层级信息

如果压缩事件按层独立产生，按 `batch_index` 去重可能错误。

事件协议应明确包含：

```text
seq_id
layer_id
event_version
new_context_len
keep_blocks
```

并规定事件是局部状态还是最终状态。

---

## 23.7 `keep_blocks` 和 `new_context_len` 必须一致

Scheduler 当前分别使用两个字段：

```text
truncate_blocks(seq, keep_blocks)
seq.num_tokens = new_context_len
```

需要验证：

```text
keep_blocks 对应的容量
是否足以容纳 new_context_len
```

还要考虑尾块：

```text
ceil(new_context_len / block_size)
```

如果不一致，可能造成越界写入或浪费 Block。

---

## 23.8 压缩事件应用顺序

当前顺序是：

```text
先压缩旧缓存
再 append 新 token
```

因此 `new_context_len` 应表示：

```text
追加本轮新 token 之前的压缩后长度
```

随后 `append_token()` 将其加一。

如果模型侧事件已经把当前新 token 计入 `new_context_len`，这里会重复加一。事件协议必须明确长度的时间点。

---

## 23.9 Tokenizer 重复加载

Scheduler 内 tokenizer 只用于注释调试，应删除，避免重复资源占用。

---

## 23.10 `postprocess()` 声明返回 `list[bool]`，实际没有返回值

nano-kvLLM 定义：

```python
def postprocess(...) -> list[bool]:
```

函数末尾没有 `return`，实际返回 `None`。

这是类型注解错误，应改为：

```python
-> None
```

或者真正返回所需布尔列表。

它不影响当前调用方，因为 `LLMEngine.step()` 未使用返回值，但会误导静态检查和维护者。

---

# 24. 哪些改动真正服务于 KV Cache 压缩

## 24.1 核心压缩改动

- `compression_events` 后处理接口；
- 按 `batch_index` 映射事件；
- 同一请求事件去重；
- `truncate_blocks()` 释放多余物理 Block；
- 用 `new_context_len` 更新当前缓存长度；
- 更新 `tail_uncompressed_len`；
- 抢占时恢复完整逻辑长度；
- 使用 `generated_completion_tokens` 判断停止。

## 24.2 为压缩简化的原有功能

- 删除 Chunked Prefill；
- 删除 `num_scheduled_tokens`；
- 删除 `is_prefill`；
- 弱化或删除 Prefix Cache；
- 删除 Block 哈希更新。

这些不是压缩算法的必然要求，而是当前实现为了降低耦合和调试难度所做的范围缩减。

## 24.3 与压缩无关或属于调试残留

- Scheduler 内重复加载 tokenizer；
- 注释中的 decode 打印；
- `waiting`、`running` 后面的 `#!!!`；
- 错误的 `list[bool]` 返回类型注解。

---

# 25. 初学者理解方式

可以把原版 Scheduler 想象成普通仓库调度员：

```text
哪些订单先入库？
每张订单占多少货架？
货架不够时暂停谁？
订单完成后释放哪些货架？
```

nano-kvLLM 的 Scheduler 还要处理“仓库压缩整理报告”：

```text
某张订单的旧货物已经被筛选
→ 只保留重要部分
→ 多余货架可以释放
→ 订单完整清单仍必须保留
→ 下次继续按原始时间顺序处理
```

其中：

- `token_ids` 是订单完整清单；
- `num_tokens` 是当前货架上实际保留的上下文长度；
- `rope_pos` 是订单真实处理进度；
- `generated_completion_tokens` 是已经新增的货物数量；
- `tail_uncompressed_len` 是最近还没整理的新货物；
- `block_table` 是当前占用的货架编号；
- `compression_events` 是整理报告。

Scheduler 的任务是把整理报告真正落实到货架管理系统中。

---

# 26. 最终总结

nano-kvLLM 对 `scheduler.py` 的核心改造，是让 Scheduler 能够接收并提交 KV Cache 压缩结果。

最关键的新增链路是：

```text
compression_events
→ 按 batch_index 找到 Sequence
→ 对事件去重
→ truncate_blocks()
→ 更新 new_context_len
→ 重置 tail_uncompressed_len
→ append_token()
```

这条链路让 KV Cache 压缩不再只是模型内部的张量操作，而能够真正：

- 缩短请求当前缓存上下文；
- 释放物理 KV Block；
- 将显存重新提供给其他请求；
- 在压缩后继续正确 Decode。

同时，nano-kvLLM 为降低实现复杂度，取消或弱化了原版的：

- Chunked Prefill；
- Prefix Cache；
- 每轮 token 调度状态；
- Prefill 阶段标记；
- Block 哈希更新。

当前实现最值得警惕的是 `num_tokens` 的双重语义：

```text
有时表示完整逻辑长度，
有时表示压缩后的当前缓存长度。
```

它依赖抢占时手动恢复完整长度，容易与其他模块产生状态不一致。后续阅读时应优先追踪：

```text
block_manager.truncate_blocks()
model_runner 中 compression_events 的生成
attention 中压缩后的 KV 索引
rope_pos 的实际使用
num_cached_tokens 的更新位置
```

只有这些部分全部对应起来，才能确认：

1. 压缩后实际释放了多少显存；
2. Block Table 是否与缓存张量一致；
3. RoPE 位置是否保持正确；
4. 抢占重算是否恢复完整上下文；
5. 压缩是否真正提升并发容量，而不是只改变逻辑长度。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-kvllm|模块-nano-kvllm]]

%% 项目关联导航：结束 %%
