# nano-vLLM 与 nano-kvLLM：`block_manager.py` 源码对比分析

## 1. 对比对象与核心结论

本次对比的两个文件分别是：

- 原版 nano-vLLM 的 `block_manager.py`
- nano-kvLLM 的 `block_manager.py`

`BlockManager` 是推理引擎中真正管理 KV Cache 物理空间的模块。前面的几个文件分别承担：

```text
Sequence       保存请求和压缩状态
Scheduler      决定请求何时运行，并提交压缩事件
LLMEngine      在 ModelRunner 与 Scheduler 之间传递结果
BlockManager   真正分配、复用、追加、截断和回收 KV Block
```

因此，只有当压缩结果最终进入 `BlockManager`，并把多余 Block 归还到 `free_block_ids` 时，KV Cache 压缩才真正转化为：

- 更低的物理显存占用；
- 更多可接纳请求；
- 更高的并发容量；
- 更少的抢占和重算。

nano-kvLLM 对该文件最核心的改造是新增：

```python
truncate_blocks(seq, keep_blocks)
```

该接口根据压缩事件缩短 `seq.block_table`，降低尾部 Block 的引用计数，并将引用计数归零的 Block 归还空闲池。

但对比还表明，nano-kvLLM 不只是增加一个截断函数，而是同时重写了：

1. Block 分配接口；
2. Prefix Cache 命中检查；
3. Block 哈希建立时机；
4. Decode 阶段完整 Block 的哈希更新；
5. 空闲缓存 Block 的重新启用方式。

这些改动使代码更直接地适配 KV Cache 压缩实验，但也引入了若干需要重点核查的状态一致性风险，尤其是：

```text
旧 hash_to_block_id 映射没有及时删除；
压缩后只将最后一个保留 Block 的 hash 设为 -1；
已失效的 token_ids 和哈希索引可能仍被 Prefix Cache 使用；
Prefill 分配阶段可能在 KV 尚未计算完成时提前登记缓存。
```

---

# 2. 整体变化概览

| 改动方向 | 原版 nano-vLLM | nano-kvLLM | 工程意义 |
|---|---|---|---|
| 包命名空间 | `nanovllm` | `nanokvllm` | 使用独立改造工程 |
| 底层 Block 分配 | `_allocate_block()` 自己从队首取空闲 Block | `_allocate_block(block_id)` 分配指定 Block | 支持重新激活哈希命中的任意空闲 Block |
| 旧哈希清理 | 复用 Block 前删除其旧哈希映射 | 未删除旧映射 | 可能产生陈旧哈希索引 |
| `can_allocate()` 返回值 | 返回命中 Prefix Cache 的 Block 数，失败返回 `-1` | 只返回布尔值 | Scheduler 不再显式获得前缀命中长度 |
| 分配容量估算 | 扣除可共享的已使用缓存 Block | 直接要求空闲 Block 数不小于逻辑 Block 总数 | 更保守，可能降低请求接纳率 |
| `allocate()` 接口 | 接收 `num_cached_blocks` | 内部重新逐块判断命中 | Prefix Cache 判断从“检查阶段”移动到“分配阶段” |
| Prefix Cache 哈希建立 | KV 计算完成后由 `hash_blocks()` 建立 | 在 `allocate()` 中提前建立 | 简化后处理，但存在过早登记风险 |
| Decode 新 Block 分配 | 只在需要时追加 Block | 保留，并增加哈希状态断言 | 显式维护完整块和可写尾块状态 |
| Decode 完整块哈希 | `postprocess()` 调用 `hash_blocks()` | `may_append()` 中直接完成 | 哈希更新职责前移 |
| 压缩后 Block 回收 | 无 | 新增 `truncate_blocks()` | 真正释放压缩后的尾部物理 Block |
| 压缩后缓存统计 | 无 | 截断 `block_table` 并限制 `num_cached_tokens` | 同步物理缓存元数据 |
| 压缩后哈希失效 | 无对应逻辑 | 仅将新的最后一个 Block 的 `hash=-1` | 意图防止继续把可写尾块当成不可变缓存，但处理可能不完整 |

从功能层面，可以把变化概括为：

```text
原版 BlockManager
= Paged KV Block 分配
+ Prefix Cache
+ 引用计数
+ Decode 扩容

nano-kvLLM BlockManager
= 原有 Block 管理
+ 压缩后尾部 Block 截断
+ 更直接的哈希状态维护
```

---

# 3. 原版 BlockManager 的基础模型

## 3.1 Block 是什么

每个 `Block` 保存：

```python
self.block_id
self.ref_count
self.hash
self.token_ids
```

它并不直接保存 GPU 上的 K/V Tensor，而是保存物理 KV Cache Block 的管理元数据。

可以理解为：

| 字段 | 作用 |
|---|---|
| `block_id` | 物理 KV Cache Block 编号 |
| `ref_count` | 当前有多少 Sequence 引用该 Block |
| `hash` | 该 Block 及其完整前缀的哈希 |
| `token_ids` | 用于验证哈希命中的 token 内容 |

实际 GPU KV Cache 通常可抽象为：

```text
k_cache[layer, block_id, ...]
v_cache[layer, block_id, ...]
```

`block_id` 决定 Sequence 的 token 应从哪个物理缓存槽读取。

---

## 3.2 `block_table` 是什么

每条 Sequence 保存：

```python
seq.block_table = [block_id_0, block_id_1, ...]
```

它描述逻辑 Block 到物理 Block 的映射：

```text
Sequence 的第 0 个逻辑块 → 物理 Block 17
Sequence 的第 1 个逻辑块 → 物理 Block 3
Sequence 的第 2 个逻辑块 → 物理 Block 42
```

注意力层通过该表访问该请求的历史 KV。

因此，KV Cache 压缩后不仅要改变 Tensor 内容，还必须保证：

```text
压缩后的有效上下文长度
与
block_table 中保留的物理 Block 数量
保持一致。
```

---

## 3.3 空闲池与已使用集合

```python
self.free_block_ids
self.used_block_ids
```

- `free_block_ids`：当前没有 Sequence 引用，可以重新分配；
- `used_block_ids`：至少被一条 Sequence 引用。

Block 被释放时不一定立即清空其哈希和 token 元数据。原版利用这一点保留 Prefix Cache：

```text
Block 当前没有被请求使用
但其 GPU KV 内容仍然存在
→ 后续相同前缀可直接重新激活
```

---

## 3.4 引用计数

多个请求可能共享同一个前缀 Block：

```text
Sequence A ─┐
            ├→ Block 12，ref_count = 2
Sequence B ─┘
```

释放其中一条请求时：

```text
ref_count: 2 → 1
```

Block 仍不能进入空闲池。

只有：

```text
ref_count: 1 → 0
```

才调用 `_deallocate_block()`。

这使 Prefix Cache 共享不会因为单个请求结束而错误释放。

---

# 4. 原版 Prefix Cache 的工作机制

原版通过链式哈希识别相同前缀：

```python
h = compute_hash(current_block_token_ids, previous_block_hash)
```

所以当前 Block 的哈希不仅取决于本块 token，还取决于之前所有 Block 的前缀。

例如：

```text
Block 0 hash = H(tokens[0:256])
Block 1 hash = H(Block 0 hash, tokens[256:512])
Block 2 hash = H(Block 1 hash, tokens[512:768])
```

即使两个请求的某个局部 Block token 完全相同，只要前缀不同，哈希也不同。

这很重要，因为 Transformer 的 KV 不只与局部 token 有关，还与之前上下文有关。

哈希命中后还会检查：

```python
self.blocks[block_id].token_ids == token_ids
```

用于降低哈希碰撞风险。

---

# 5. 详细改动一：`_allocate_block()` 改为分配指定 Block

## 5.1 原版实现

原版：

```python
def _allocate_block(self) -> int:
    block_id = self.free_block_ids.popleft()
    ...
    return block_id
```

它始终获取空闲队列的第一个 Block。

此外，原版在重新使用一个空闲 Block 前，会清除其旧哈希索引：

```python
if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
    del self.hash_to_block_id[block.hash]
```

因为一个空闲 Block 虽然可能保留旧 Prefix Cache 内容，但如果它现在将被用于另一段 KV，就必须删除旧映射。

---

## 5.2 nano-kvLLM 实现

nano-kvLLM：

```python
def _allocate_block(self, block_id: int) -> Block:
    block = self.blocks[block_id]
    assert block.ref_count == 0
    block.reset()
    self.free_block_ids.remove(block_id)
    self.used_block_ids.add(block_id)
    return self.blocks[block_id]
```

调用方可以指定：

- 空闲队首的新 Block；
- 哈希命中的某个空闲缓存 Block。

这让同一个底层函数同时处理：

```text
普通新分配
和
重新激活空闲 Prefix Cache Block
```

---

## 5.3 为什么压缩项目可能采用指定 Block 分配

nano-kvLLM 的 `allocate()` 在内部判断哈希命中，因此它需要：

```text
命中哪个 block_id
→ 就重新激活哪个 block_id
```

指定 Block 的接口比原版“统一从队首取一个”更直接。

此外，压缩后大量 Block 会重新进入空闲池，其中部分仍带有旧缓存元数据。指定分配接口使管理器能够根据哈希选择特定空闲 Block。

---

## 5.4 重要风险：未删除旧哈希映射

nano-kvLLM 的 `_allocate_block(block_id)` 在 `reset()` 前没有执行原版的：

```python
del self.hash_to_block_id[block.hash]
```

因此，如果一个带有旧哈希的空闲 Block 被作为普通新空间复用：

```text
旧 hash → block_id
```

这条字典映射可能继续存在，即使 Block 已经：

- 被重置；
- 写入新 token；
- 建立了新哈希；
- 保存了不同的 KV 内容。

### 一般情况下为何不立即崩溃

后续查找旧 hash 时还会检查：

```python
self.blocks[block_id].token_ids != token_ids
```

如果新旧 token 不同，会判定 Cache Miss。

所以部分陈旧映射只是造成：

- 无效查找；
- 哈希表增长；
- 额外比较开销。

### 但仍可能造成错误命中

假设：

```text
旧前缀 P1 + 局部 token 块 T → old_hash
新前缀 P2 + 同样的局部 token 块 T → new_hash
```

Block 被复用后：

```text
hash = new_hash
token_ids = T
```

但字典中仍保留：

```text
old_hash → 同一个 block_id
```

以后查询 `old_hash` 时：

- 能找到该 block_id；
- `token_ids` 仍然等于 T；
- 当前代码没有验证 `block.hash == old_hash`；
- 可能把属于前缀 P2 的 KV 错当成前缀 P1 的 KV。

这会造成错误的 Prefix Cache 复用，属于模型正确性问题，而不仅是性能问题。

### 建议

重新分配前应删除旧映射：

```python
old_hash = block.hash
if old_hash != -1 and self.hash_to_block_id.get(old_hash) == block_id:
    del self.hash_to_block_id[old_hash]
```

查找命中时也应增加：

```python
block.hash == h
```

的显式验证。

---

# 6. 详细改动二：`can_allocate()` 从前缀感知变为保守布尔判断

## 6.1 原版逻辑

原版逐块检查 Prefix Cache：

```python
num_cached_blocks = 0
num_new_blocks = seq.num_blocks
```

每命中一个前缀 Block：

```python
num_cached_blocks += 1
```

如果命中的 Block 已经被其他请求使用，可以共享引用，不需要消耗一个新的空闲 Block：

```python
if block_id in self.used_block_ids:
    num_new_blocks -= 1
```

最终：

```python
if len(self.free_block_ids) < num_new_blocks:
    return -1
return num_cached_blocks
```

因此原版 `can_allocate()` 同时回答：

1. 能否接纳该请求；
2. 命中了多少 Prefix Cache Block；
3. 实际需要多少新物理 Block。

---

## 6.2 nano-kvLLM 逻辑

nano-kvLLM：

```python
def can_allocate(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= seq.num_blocks
```

它不检查 Prefix Cache，只要求：

```text
空闲 Block 数量 ≥ 该 Sequence 的全部逻辑 Block 数
```

---

## 6.3 工程意义

接口更简单：

```text
原版：返回 -1 或缓存命中数量
nano-kvLLM：返回 True / False
```

Scheduler 不再处理 Prefix Cache 细节，所有命中判断移动到 `allocate()`。

---

## 6.4 这是一个保守但可能浪费容量的判断

假设请求需要 10 个 Block，其中前 8 个可与正在运行的请求共享：

```text
实际新增 Block 需求 = 2
```

但 nano-kvLLM 仍要求：

```text
free_block_ids >= 10
```

如果当前只有 5 个空闲 Block：

- 实际完全可以接纳；
- nano-kvLLM 却会拒绝。

因此它不会造成显存超分配，但可能造成：

- 请求接纳率下降；
- waiting 队列阻塞；
- 并发能力低于实际可用容量；
- Prefix Cache 收益无法体现在调度准入阶段。

这与前一个 `scheduler.py` 的结论一致：nano-kvLLM 保留了部分 Prefix Cache 行为，但弱化了原版精细的前缀感知调度。

---

# 7. 详细改动三：Prefix Cache 判断从 `can_allocate()` 移入 `allocate()`

## 7.1 原版两阶段协议

原版：

```text
can_allocate()
    ↓
计算 num_cached_blocks
    ↓
allocate(seq, num_cached_blocks)
```

检查和执行分离。

优点：

- Scheduler 可以准确计算本轮实际 Prefill token 数；
- 申请前就知道缓存命中长度；
- 容量判断考虑共享缓存；
- `allocate()` 只执行已确定的方案。

---

## 7.2 nano-kvLLM 单阶段协议

nano-kvLLM：

```text
can_allocate()
    ↓
只检查最保守的空闲容量
    ↓
allocate(seq)
    ↓
重新逐块寻找 Prefix Cache
```

`allocate()` 内部维护：

```python
cache_miss = False
```

一旦某个 Block 未命中：

```python
cache_miss = True
```

后续 Block 即使哈希存在，也不再复用。

这符合 Prefix Cache 必须是连续前缀的要求：

```text
只能复用从第 0 块开始连续命中的前缀，
不能跳过中间缺失块后继续复用后面的块。
```

---

## 7.3 命中一个 Block 时的两种状态

### 命中的 Block 正在被使用

```python
block.ref_count += 1
```

表示多个 Sequence 共享它，不消耗空闲 Block。

### 命中的 Block 当前空闲

```python
block = self._allocate_block(block_id)
```

将该 Block 从 `free_block_ids` 移回 `used_block_ids`。

之后：

```python
seq.num_cached_tokens += self.block_size
```

告诉模型该请求有多少 Prompt token 已经存在于 KV Cache 中，可以跳过计算。

---

# 8. 详细改动四：nano-kvLLM 在 `allocate()` 中提前建立哈希

nano-kvLLM 对每个完整 Block 执行：

```python
block.update(h, token_ids)
self.hash_to_block_id[h] = block_id
```

这一动作发生在：

```text
Scheduler 调度 Prefill
→ BlockManager.allocate()
→ ModelRunner 真正执行 Prefill
```

之前。

---

## 8.1 原版为何不这样做

原版在 `allocate()` 中只分配物理 Block，不会立即宣称这些 Block 已有有效缓存。

只有模型运行完成后，Scheduler 的 `postprocess()` 才调用：

```python
self.block_manager.hash_blocks(seq)
```

此时 Block 对应的 K/V 已经真实写入 GPU 缓存，才建立：

```text
hash → block_id
```

因此原版遵循：

```text
先计算 KV
再发布缓存
```

---

## 8.2 nano-kvLLM 的前移设计

nano-kvLLM 删除了 `hash_blocks()`，把哈希登记分散到：

- Prefill 的 `allocate()`；
- Decode 的 `may_append()`。

这简化了 Scheduler 后处理，不再依赖：

```text
num_scheduled_tokens
num_cached_tokens 的逐轮累加
hash_blocks()
```

---

## 8.3 潜在正确性风险：缓存可能在 KV 尚未写入时被发布

考虑同一个 Prefill batch 中有两个请求：

```text
Sequence A 和 Sequence B 具有相同前缀。
```

Scheduler 可能依次执行：

```text
allocate(A)
allocate(B)
ModelRunner.run([A, B], is_prefill=True)
```

`allocate(A)` 已经将 A 的完整块写入 `hash_to_block_id`，但此时 A 的 GPU KV 尚未计算。

随后 `allocate(B)` 可能把 A 的 Block 识别为 Prefix Cache 命中，并增加：

```python
seq_B.num_cached_tokens
```

结果 B 可能跳过这部分 Prefill 计算，但对应 KV 还没有真正生成。

除非 ModelRunner 对同批请求有非常特殊的执行顺序和依赖保证，否则这是潜在的错误复用。

更稳健的原则应是：

```text
只有在 KV 计算完成后，
Block 才能进入全局 Prefix Cache 索引。
```

因此建议保留原版“运行后发布”的机制，或者为 Block 增加状态：

```text
ALLOCATED
COMPUTING
READY
```

只有 `READY` Block 才允许其他请求复用。

---

# 9. 详细改动五：重新设计 Decode 阶段的 `may_append()`

## 9.1 原版逻辑

原版只处理一种情况：

```python
if len(seq) % self.block_size == 1:
    seq.block_table.append(self._allocate_block())
```

原因是：

- 新采样 token 已经追加到 `token_ids`；
- 下一轮 Decode 才会计算该 token 的 KV；
- 当它是新 Block 的第一个 token 时，需要提前分配新的物理 Block。

完整 Block 的哈希由后处理阶段 `hash_blocks()` 更新。

---

## 9.2 nano-kvLLM 逻辑

nano-kvLLM 将其扩展为三个分支：

```python
if len(seq) % self.block_size == 1:
    分配新 Block
elif len(seq) % self.block_size == 0:
    为刚填满的 Block 建立哈希
else:
    验证最后一个 Block 仍可写
```

### 情况一：当前 token 是新块第一个 token

```python
assert last_block.hash != -1
```

要求上一块已经完整、不可再写，并且已有哈希。

随后新建一个 `hash=-1` 的可写 Block。

### 情况二：当前 token 即将把最后一块填满

```python
assert last_block.hash == -1
```

该块之前是未完成、可写状态。

代码根据完整 token 内容计算哈希：

```python
token_ids = seq.block(seq.num_blocks - 1)
prefix = previous_block.hash
h = compute_hash(token_ids, prefix)
```

再把它加入 Prefix Cache。

### 情况三：仍在同一个未满块内部

要求：

```python
last_block.hash == -1
```

因为未满 Block 仍会继续写入，不应当作为不可变 Prefix Cache 共享。

---

## 9.3 这里体现的 Block 状态机

nano-kvLLM 使用 `hash` 同时表达两种含义：

```text
hash == -1
→ Block 未完成或仍可能被写入

hash != -1
→ Block 已完成，可作为不可变 Prefix Cache
```

状态转换近似为：

```text
新 Block
hash = -1
   ↓ 不断追加 token
填满 Block
   ↓
计算 hash
   ↓
成为可共享的只读 Prefix Cache Block
```

---

## 9.4 哈希更新时机仍需要谨慎

当 `len(seq) % block_size == 0` 时，`token_ids` 已经包含使该块完整的最新采样 token，但这个 token 的 KV 将在即将执行的 Decode step 中计算。

`may_append()` 在模型运行前就登记哈希。

与 Prefill 的问题类似，它在 K/V 完成前短暂发布缓存。

Decode 调度阶段通常不会同时进行新的 Prefix Cache 分配，因此正常流程中其他请求可能直到本轮模型执行完成后才查询该哈希，风险较 Prefill 低。

但如果：

- 模型执行失败；
- 存在异步调度；
- 未来支持 Prefill/Decode 混合批处理；
- 多线程同时访问 BlockManager；

就可能留下标记为有效但实际未写完的缓存。

生产级实现仍应把“发布缓存”放在执行成功之后。

---

# 10. 详细改动六：新增 `truncate_blocks()`，让压缩真正释放显存

新增方法：

```python
def truncate_blocks(self, seq, keep_blocks):
```

它接收：

```text
keep_blocks = 压缩后应该保留的物理 Block 数
```

并完成四步操作。

---

## 10.1 第一步：判断是否需要截断

```python
if keep_blocks >= len(seq.block_table):
    return
```

如果要求保留的数量不少于当前数量，不做任何处理。

---

## 10.2 第二步：释放尾部 Block 引用

```python
tail = seq.block_table[keep_blocks:]
```

然后逆序处理：

```python
block.ref_count -= 1
if block.ref_count == 0:
    self._deallocate_block(block_id)
```

这保持了原有引用计数语义：

- 只被当前 Sequence 使用的 Block 会真正进入空闲池；
- 与其他请求共享的 Block 不会被错误释放。

---

## 10.3 第三步：截断 Sequence 的 Block Table

```python
seq.block_table = seq.block_table[:keep_blocks]
```

此后注意力层只能通过该 Sequence 访问保留的物理 Blocks。

这是压缩结果从算法层落到缓存寻址层的关键步骤。

---

## 10.4 第四步：限制缓存 token 统计

```python
seq.num_cached_tokens = min(
    seq.num_cached_tokens,
    keep_blocks * self.block_size
)
```

这一点回应了前面对 Scheduler 的疑问：

> `truncate_blocks()` 确实会同步缩小 `num_cached_tokens`，但它只执行上限截断，不一定将其精确设置为压缩后的有效 token 数。

例如：

```text
keep_blocks = 3
block_size = 256
new_context_len = 600
```

物理容量为 768 token，但有效上下文只有 600。

这里最多保证：

```text
num_cached_tokens <= 768
```

并不能自动保证：

```text
num_cached_tokens == 600
```

Scheduler 会单独设置：

```python
seq.num_tokens = new_context_len
```

所以当前系统中：

- `num_tokens`：压缩后有效上下文长度；
- `num_cached_tokens`：缓存统计值，最多受物理容量限制；
- `len(block_table) * block_size`：物理容量上限。

这三个数不一定相等。

---

# 11. `truncate_blocks()` 如何把压缩率转化为系统收益

假设压缩前：

```text
block_table 长度 = 20
每块 256 token
物理容量 = 5120 token
```

压缩事件返回：

```text
keep_blocks = 8
new_context_len = 1900
```

Scheduler 调用：

```text
truncate_blocks(seq, 8)
```

BlockManager 释放 12 个尾部 Block。

若这些 Block 都只被当前请求引用：

```text
free_block_ids 增加 12
used_block_ids 减少 12
```

系统立即可以把它们分给：

- 新 Prefill 请求；
- 其他 Decode 请求的新 Block；
- 被抢占后重新计算的请求。

因此：

```text
算法压缩率
只有经过 truncate_blocks()
才变成物理 Block 释放率。
```

若算法只缩短 Attention 的逻辑索引，却不截断 `block_table`，显存池不会获得新的可用 Block，并发容量也不会提高。

---

# 12. 详细改动七：压缩后将最后一个保留 Block 的 hash 设为 `-1`

代码：

```python
if seq.block_table:
    last_block = self.blocks[seq.block_table[-1]]
    last_block.hash = -1
```

其意图是：

```text
压缩后的最后一个 Block 可能未填满，
后续 Decode 还要继续写入，
所以不能继续视为不可变的完整 Prefix Cache Block。
```

这与 `may_append()` 的状态约定一致：

```text
可写尾块 → hash == -1
完整只读块 → hash != -1
```

---

## 12.1 仅修改 `hash` 还不够

代码没有同步删除：

```python
hash_to_block_id[old_hash]
```

也没有清空或重建：

```python
last_block.token_ids
```

于是可能出现：

```text
last_block.hash = -1
但
hash_to_block_id[old_hash] = last_block_id
且
last_block.token_ids 仍是旧值
```

而 `allocate()` 判断命中时没有检查：

```python
block.hash == h
```

只检查：

```python
block_id 是否存在
token_ids 是否相同
```

因此一个已经被标记为无效的 Block 仍可能通过旧映射被当作缓存命中。

这是该文件中最明确的哈希一致性问题之一。

### 建议

失效时应执行：

```python
old_hash = last_block.hash
if old_hash != -1 and self.hash_to_block_id.get(old_hash) == last_block_id:
    del self.hash_to_block_id[old_hash]
last_block.hash = -1
last_block.token_ids = []
```

具体是否清空 `token_ids` 取决于后续是否仍需调试或重建哈希，但至少不能保留可被误命中的旧索引。

---

# 13. 更深层问题：压缩可能让所有保留 Block 的哈希失效

当前 `truncate_blocks()` 只使最后一个保留 Block 失效。

这只有在以下前提下才足够：

```text
压缩只是删除完整尾部 Block，
保留的前缀 Block 内容完全不变。
```

但许多 KV Cache 压缩算法不是简单截断尾部，而是：

- 从整个历史中选择重要 token；
- 保留窗口内最近 token；
- 保留 Attention Score 较高的历史 token；
- 将离散保留 token 紧凑搬移到前面的缓存槽；
- 重新组织 Position/Slot Mapping。

在这种情况下，压缩后前 N 个物理 Block 的 KV 内容可能已被重写。

例如：

```text
压缩前 Block 0：token 0~255
压缩前 Block 1：token 256~511
压缩后 Block 0：选中的 token 0、8、31、...
```

此时原有：

```text
block.hash
block.token_ids
hash_to_block_id
```

已经不再描述真实 KV 内容。

只设置最后一块 `hash=-1` 不够，所有被改写的保留 Block 都应：

- 从 Prefix Cache 哈希表移除；
- 清除旧 token 元数据；
- 或根据压缩后的新逻辑 token 序列重新计算元数据。

如果压缩后的缓存无法再对应连续原始 token 前缀，最安全的方式通常是：

```text
压缩过的 Sequence 不再参与普通 Prefix Cache 共享。
```

因此必须结合压缩内核确认：

```text
compression_events 是单纯尾部截断，
还是对 KV 进行选择与重排。
```

这决定当前哈希失效逻辑是否正确。

---

# 14. 详细改动八：尾部释放 Block 的哈希元数据仍被保留

`_deallocate_block()` 与原版一样，只做：

```python
used_block_ids.remove(block_id)
free_block_ids.append(block_id)
```

不会清除：

```text
block.hash
block.token_ids
hash_to_block_id
```

在普通 Prefix Cache 中，这是有意设计：

```text
请求虽然结束，
GPU Block 内容仍在，
可以作为空闲 Prefix Cache 保留。
```

但压缩释放的尾部 Block 是否仍可保留为 Prefix Cache，需要具体判断。

### 如果压缩前没有改写这些尾部 Block

它们仍包含原来的有效 KV，可以继续作为前缀缓存候选。

### 如果压缩过程将保留 token搬移或覆盖了尾部 Block

尾部 Block 中的物理 KV 可能不再与旧 `token_ids` 对应。

此时将它们直接放回空闲池并保留旧哈希，会造成错误复用。

因此压缩内核和 BlockManager 必须建立明确契约：

```text
被释放 Block 的 KV 内容是否仍然保持原样？
```

若不能保证，就应在压缩释放路径中清除哈希元数据，而不能复用普通 `deallocate` 语义。

---

# 15. `num_cached_tokens` 在 nano-kvLLM 中的实际变化

该字段在不同路径中被修改：

## 15.1 Prefix Cache 命中

```python
seq.num_cached_tokens += self.block_size
```

表示有完整 Prompt Block 可以跳过 Prefill。

## 15.2 全部释放

```python
seq.num_cached_tokens = 0
```

抢占或完成请求后清零。

## 15.3 压缩截断

```python
seq.num_cached_tokens = min(
    seq.num_cached_tokens,
    keep_blocks * block_size
)
```

限制它不能超过保留物理容量。

---

## 15.4 与 `seq.num_tokens` 的区别

在 nano-kvLLM 当前设计中：

```text
seq.num_tokens
→ Scheduler 根据 new_context_len 设置的有效缓存上下文长度

seq.num_cached_tokens
→ 主要表示 Prefix Cache 已命中的 token 数或缓存进度统计

len(block_table) * block_size
→ 实际分配的物理容量
```

因此压缩后不能简单假设：

```text
num_tokens == num_cached_tokens
```

也不能假设：

```text
num_cached_tokens == len(block_table) * block_size
```

后续所有调用点都必须明确自己需要的是：

- 逻辑历史长度；
- 压缩后有效上下文长度；
- 已复用 Prefix Cache token 数；
- 物理 Block 容量。

---

# 16. 与前面三个文件的完整联动链路

目前已经可以把四个文件串成一条完整控制路径。

## 16.1 Sequence 层

保存：

```text
token_ids                    完整逻辑历史
generated_completion_tokens 真实生成数量
rope_pos                     逻辑位置
tail_uncompressed_len        未压缩尾部
num_tokens                   当前有效缓存长度
block_table                  物理 Block 映射
```

## 16.2 ModelRunner 层

执行模型和压缩算法，返回：

```text
token_ids
compression_events
```

事件包含：

```text
batch_index
new_context_len
keep_blocks
tail_uncompressed_len_after
```

## 16.3 LLMEngine 层

识别 ModelRunner 返回值是否带有压缩事件，并转交 Scheduler。

## 16.4 Scheduler 层

执行：

```text
compression_events 去重
→ 找到对应 Sequence
→ BlockManager.truncate_blocks()
→ seq.num_tokens = new_context_len
→ 更新 tail_uncompressed_len
→ append_token()
```

## 16.5 BlockManager 层

执行：

```text
释放 block_table 尾部
→ ref_count--
→ 引用归零的 Block 进入 free_block_ids
→ 缩短 block_table
→ 限制 num_cached_tokens
→ 将新尾块标为可写
```

完整链路为：

```text
压缩算法选择保留 KV
        ↓
ModelRunner 形成 compression_event
        ↓
LLMEngine 传递事件
        ↓
Scheduler 提交事件
        ↓
BlockManager 截断物理 Block
        ↓
空闲 Block 增加
        ↓
系统可接纳更多请求
```

---

# 17. `truncate_blocks()` 与 `new_context_len` 必须满足的关系

Scheduler 分别使用：

```text
keep_blocks
new_context_len
```

二者必须一致。

设：

```text
B = block_size
K = keep_blocks
L = new_context_len
```

至少应满足：

```text
0 < L <= K × B
```

并且通常应满足：

```text
K = ceil(L / B)
```

除非系统故意预留额外空块。

若：

```text
K < ceil(L / B)
```

则分配的 Block 容量不足，后续 Attention 或写入可能越界。

若：

```text
K >> ceil(L / B)
```

则压缩后仍保留过多空闲容量，降低显存收益。

当前代码没有验证二者关系，建议在调试模式加入断言。

---

# 18. 需要关注的边界条件

## 18.1 `keep_blocks == 0`

`truncate_blocks()` 会释放全部 Block，并将表置空。

之后 `may_append()` 立即访问：

```python
block_table[-1]
```

会触发索引错误。

如果压缩策略保证至少保留一个 Block，应显式断言：

```python
assert keep_blocks >= 1
```

否则需要支持空表后的重新分配。

---

## 18.2 `keep_blocks < 0`

Python 切片会产生非预期结果：

```python
seq.block_table[:negative]
```

当前没有参数校验。

应要求：

```python
0 <= keep_blocks <= len(seq.block_table)
```

---

## 18.3 最后一个 Block 被多个请求共享

`truncate_blocks()` 直接执行：

```python
last_block.hash = -1
```

如果该 Block 的 `ref_count > 1`，意味着其他 Sequence 也在引用它。

修改共享 Block 的元数据会影响所有引用者。

正常设计下，可写尾块不应共享；但如果 Prefix Cache 复用了完整块，而压缩后该块变成新的可写尾块，就需要 Copy-on-Write：

```text
共享只读 Block
→ 当前 Sequence 需要修改
→ 分配新 Block 并复制
→ 当前 Sequence 改用新 Block
```

当前代码没有显式处理这一点。

---

## 18.4 哈希字典可能长期积累陈旧项

由于：

- 重分配时不删除旧映射；
- `truncate_blocks()` 失效 hash 时不删除映射；
- 一个 Block 可能先后对应多个 hash；

`hash_to_block_id` 可能不断增长。

即使多数陈旧映射被 token 比较挡住，也会：

- 增加内存占用；
- 增加无效命中；
- 增加错误复用风险；
- 使调试时哈希状态难以理解。

---

## 18.5 `free_block_ids.remove(block_id)` 的复杂度

按指定 Block 从 deque 中删除通常需要线性搜索。

普通队首分配最好使用：

```python
popleft()
```

Prefix Cache 命中空闲 Block 时才需要按值删除。

可以拆分为：

```text
_allocate_new_block()
_reactivate_cached_block(block_id)
```

让职责和复杂度更明确。

---

# 19. 代码层面可以改进的设计

## 19.1 区分 Block 生命周期状态

不要仅用 `hash == -1` 隐式表达全部状态，可以增加：

```text
FREE_CACHED
ALLOCATED
COMPUTING
READY_SHARED
MUTABLE_TAIL
```

这样可以防止尚未计算完成的 Block 被复用。

---

## 19.2 建立统一的哈希失效函数

例如：

```python
def _invalidate_hash(self, block):
    old_hash = block.hash
    if old_hash != -1 and self.hash_to_block_id.get(old_hash) == block.block_id:
        del self.hash_to_block_id[old_hash]
    block.hash = -1
    block.token_ids = []
```

所有以下操作统一调用：

- Block 被重新用于新内容；
- 压缩重写保留 Block；
- 尾块重新变为可写；
- 发生异常回滚。

---

## 19.3 哈希命中时验证完整一致性

至少检查：

```python
block.hash == h
block.token_ids == token_ids
```

而不是只检查 token。

---

## 19.4 恢复“计算完成后发布缓存”

更稳健的流程：

```text
allocate 只分配
ModelRunner 写入 KV
postprocess/hash_blocks 发布哈希
```

压缩后的新布局也应在物理搬移完成后再更新元数据。

---

## 19.5 压缩 Block 与 Prefix Cache 解耦

若压缩后的 KV 不再对应连续原始 token 前缀，可以为 Sequence 增加：

```text
is_compressed
```

压缩后：

```text
禁止其 Block 进入普通 Prefix Cache
```

或建立独立的压缩缓存键和位置映射机制。

---

## 19.6 明确三个长度

建议不要继续混用：

```text
logical_num_tokens
active_context_len
allocated_cache_capacity
```

分别对应：

```text
len(token_ids)
new_context_len
len(block_table) * block_size
```

---

# 20. 哪些改动真正服务于 KV Cache 压缩

## 20.1 核心压缩改动

- 新增 `truncate_blocks()`；
- 根据 `keep_blocks` 释放尾部 Block；
- 缩短 `seq.block_table`；
- 调整 `seq.num_cached_tokens`；
- 将压缩后的新尾块标记为可写；
- 让释放后的 Block 重新进入全局空闲池。

这些改动直接决定压缩是否产生物理显存收益。

---

## 20.2 为简化新调度链路而做的改动

- `can_allocate()` 改为布尔判断；
- `allocate()` 内部自行检测 Prefix Cache；
- 删除外部传入的 `num_cached_blocks`；
- 删除独立 `hash_blocks()`；
- 把完整块哈希维护前移到 `allocate()` 和 `may_append()`。

这些改动不是 KV Cache 压缩的理论必需条件，而是当前 nano-kvLLM 对原版 Block 管理协议的重构。

---

## 20.3 需要重点审查的潜在问题

- 复用 Block 前没有删除旧哈希映射；
- 哈希命中不验证 `block.hash == h`；
- Prefill KV 尚未计算完成就可能发布哈希；
- 压缩后只使最后一个保留 Block 失效；
- 哈希失效时没有删除 `hash_to_block_id` 中的旧条目；
- 压缩释放的 Block 是否仍保留有效 KV 未明确；
- 共享 Block 变为可写尾块时没有 Copy-on-Write；
- `keep_blocks` 与 `new_context_len` 缺少一致性检查。

这些问题不一定在当前实验输入下全部触发，但从代码契约看都值得继续验证。

---

# 21. 初学者理解方式

可以把 BlockManager 想成一个仓库管理员。

## 原版

仓库管理员负责：

```text
给请求分配货架；
相同前缀请求共享已经摆好的货架；
记录每个货架被几张订单使用；
订单结束后把货架归还；
保留货架上的旧货物，供以后相同订单复用。
```

## nano-kvLLM

现在又增加了“压缩整理”：

```text
一个请求原本占 20 个货架；
压缩算法说只需保留 8 个；
Scheduler 把报告交给 BlockManager；
BlockManager 释放后 12 个货架；
其他请求马上可以使用这些货架。
```

但仓库还有一本“货架内容索引表”：

```text
hash_to_block_id
```

如果货架内容已经改变，却没有删除旧索引，之后就可能按照旧标签拿到错误货物。

这就是 nano-kvLLM 当前 Block 哈希维护中最需要警惕的问题。

---

# 22. 最终总结

nano-kvLLM 对 `block_manager.py` 的核心贡献，是新增了一条真正释放压缩后物理 KV Cache Block 的路径：

```text
compression_event.keep_blocks
→ truncate_blocks()
→ 尾部 Block ref_count--
→ ref_count 为 0 的 Block 回到 free_block_ids
→ block_table 缩短
→ 系统获得新的可分配显存空间
```

这是 KV Cache 压缩从“算法层减少有效 token”转化为“推理系统层提高并发容量”的关键一步。

与此同时，nano-kvLLM 还重写了 Prefix Cache 和哈希维护协议：

```text
原版：
can_allocate 计算命中
→ allocate 执行分配
→ 模型完成计算
→ hash_blocks 发布缓存

nano-kvLLM：
can_allocate 只做保守容量检查
→ allocate 内部判断命中并立即登记哈希
→ may_append 维护完整块哈希
```

该方案减少了对 `num_scheduled_tokens` 和 `hash_blocks()` 的依赖，但当前实现存在明显的哈希元数据一致性隐患，尤其是：

```text
旧哈希映射未清理；
压缩后失效 Block 仍可能被旧索引命中；
压缩若重排 KV，所有被改写 Block 的旧哈希都应失效；
Prefill Block 可能在 KV 未计算完成前被发布为缓存。
```

因此，从整体项目角度看：

> `truncate_blocks()` 已经建立了 KV Cache 压缩的物理 Block 回收机制，但 Prefix Cache 与压缩缓存之间的边界和哈希失效规则仍需要进一步完善。

后续阅读最应该继续核查：

```text
1. ModelRunner 如何生成 keep_blocks 和 new_context_len；
2. 压缩内核是否把保留 KV 搬移到前部 Blocks；
3. 被释放的尾部 Blocks 是否被压缩过程覆盖；
4. Attention 的 slot_mapping 如何适配压缩后的布局；
5. 压缩后 Prefix Cache 是否仍被允许；
6. BlockManager 的哈希映射是否会在实际测试中出现陈旧命中。
```

把这些问题与前面的 `sequence.py`、`scheduler.py`、`llm_engine.py` 串起来后，可以得到 nano-kvLLM 当前 KV Cache 压缩系统的完整核心思想：

```text
逻辑 token 历史保留不变
        +
模型侧压缩有效 KV 上下文
        +
Scheduler 接收压缩事件
        +
BlockManager 截断物理 Block
        +
RoPE 和生成计数继续沿原逻辑时间轴推进
```


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-kvllm|模块-nano-kvllm]]

%% 项目关联导航：结束 %%
