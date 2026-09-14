# qwen3.6_block_manager.py_对比分析

> 对比对象：  
> 1. 原版 `nano-vLLM` 的 `block_manager.py`  
> 2. `nano-vllm-qwen3.6` 中对应的 `block_manager.py`
>
> 本文目标：从整体工程角度分析 qwen3.6 版本相比原版 `BlockManager` 做了哪些修改、为什么这样改，以及这些修改在 Qwen3.6 / hybrid 架构 / 推理系统中的作用。

---

## 1. 文件整体定位

`block_manager.py` 是 nano-vLLM 推理系统中负责 **KV Cache block 管理** 的核心文件。

在 decoder-only 大模型推理中，Attention 层需要保存历史 token 的 Key / Value，也就是 KV Cache。为了避免为每个请求分配一整块连续显存，nano-vLLM 会把 KV Cache 切成固定大小的 block，然后每个 `Sequence` 通过 `block_table` 记录自己占用了哪些 block。

所以 `BlockManager` 的核心职责是：

```text
管理 KV Cache block 的分配、释放、复用、追加和 prefix cache hash 映射
```

它在推理流程中的位置大致是：

```text
LLMEngine.add_request
  ↓
Scheduler.schedule
  ↓
BlockManager.can_allocate / allocate
  ↓
ModelRunner.prepare_prefill/decode
  ↓
Attention 写入或读取 KV Cache
  ↓
Scheduler.postprocess
  ↓
BlockManager.may_append / deallocate
```

注意：

`block_manager.py` 不直接做 Attention 计算，也不直接保存真正的 K/V 张量。它管理的是 **block 元数据**：

```text
哪些 block 空闲？
哪些 block 正在使用？
某个 block 对应哪些 token？
某个 prefix hash 对应哪个 block？
某个 block 有多少请求共享？
```

---

## 2. qwen3.6 版本整体变化概览

相比原版 `block_manager.py`，qwen3.6 版本的主要变化如下：

| 改动点 | 原版 nano-vLLM | qwen3.6 版本 | 作用 |
|---|---|---|---|
| `_allocate_block` | 不接收参数，只从 free 队列头部分配 | 接收指定 `block_id` | 既能分配新 block，也能激活 prefix cache 命中的 free block |
| `can_allocate` 返回值 | 返回 `num_cached_blocks` 或 `-1` | 返回 `bool` | 调度器只判断是否有足够 block，不再在这里返回 prefix cache 命中块数 |
| prefix cache 判断位置 | `can_allocate()` 先精确判断可复用块数 | `allocate()` 内部边分配边判断命中 | 逻辑从“预检查 + 分配”改为“分配时处理” |
| `allocate` 参数 | `allocate(seq, num_cached_blocks)` | `allocate(seq, disable_prefix_cache=False)` | 增加 hybrid 模型下禁用 prefix cache 的入口 |
| hash 写入时机 | 通过 `hash_blocks()` 在 postprocess 后更新 | 在 `allocate()` 和 `may_append()` 中更新 | 去掉单独的 `hash_blocks()` |
| `may_append` | 只在需要新 block 时追加 | 同时负责新 block 追加和满 block hash 更新 | decode 阶段 block 追加和 hash 维护合并 |
| `hash_blocks` | 存在 | 删除 | scheduler 不再显式调用 hash_blocks |
| prefix cache 策略 | 更精细、更节省 block | 更保守、更适合 hybrid 一致性 | 为 Qwen3.6 hybrid / GDN state 避免不安全复用 |

一句话总结：

**qwen3.6 版本的 `BlockManager` 不再把 prefix cache 命中数量作为调度前置返回值，而是把 prefix cache 处理内聚到 `allocate()` 中，并新增 `disable_prefix_cache` 开关，服务于 hybrid 模型下 KV Cache 与 GDN recurrent state 的一致性。**

---

## 3. 原版 BlockManager 的核心设计

原版 `BlockManager` 的设计目标是：

```text
在普通 decoder-only attention 模型中，尽可能复用已有 prefix KV Cache，减少 prefill 计算和 KV Cache 显存占用。
```

它的核心流程是：

```text
can_allocate(seq)
  ↓
计算该请求有多少完整 prefix block 已经存在
  ↓
返回 num_cached_blocks
  ↓
Scheduler 根据 num_cached_blocks 决定还要 prefill 多少 token
  ↓
allocate(seq, num_cached_blocks)
  ↓
复用命中的 block，并为未命中部分分配新 block
  ↓
prefill 后 hash_blocks(seq)
  ↓
把新完成的完整 block 写入 hash_to_block_id
```

原版中比较关键的点有两个。

第一，`can_allocate()` 不只是判断显存够不够，它还会计算 prefix cache 能命中多少块：

```python
def can_allocate(self, seq: Sequence) -> int:
    ...
    return num_cached_blocks
```

如果资源不够，则返回 `-1`。

第二，`hash_blocks()` 是单独存在的。它会在 prefill 之后，根据本轮已经真正计算完成的 token 范围，把完整 block 写入 hash 表：

```python
def hash_blocks(self, seq: Sequence):
    ...
    block.update(h, token_ids)
    self.hash_to_block_id[h] = block.block_id
```

这说明原版更强调：

```text
只有已经被真正 prefill 完成的完整 block，才应该进入 prefix cache hash 表。
```

---

## 4. 改动一：`_allocate_block` 从“弹出队首”变成“指定 block_id 分配”

### 4.1 原版写法

原版：

```python
def _allocate_block(self) -> int:
    block_id = self.free_block_ids.popleft()
    block = self.blocks[block_id]
    ...
    block.reset()
    self.used_block_ids.add(block_id)
    return block_id
```

它只做一种事情：

```text
从 free_block_ids 队首拿一个空闲 block。
```

### 4.2 qwen3.6 版本写法

qwen3.6 版本：

```python
def _allocate_block(self, block_id: int) -> Block:
    block = self.blocks[block_id]
    assert block.ref_count == 0
    block.reset()
    self.free_block_ids.remove(block_id)
    self.used_block_ids.add(block_id)
    return block
```

它接收一个指定的 `block_id`。

这意味着它不再只能拿 free 队列头部 block，而是可以把任意一个空闲 block 激活为 used 状态。

### 4.3 这个改动的意义

为什么需要指定 `block_id`？

因为 prefix cache 命中时，hash 表可能指向某个当前不在 used 集合中的 free block：

```text
hash_to_block_id[h] = 某个 block_id
```

如果这个 block 当前是 free 状态，但它的内容仍然对应某个 prefix，那么复用它时就不能简单从 free 队列头部拿一个新块，而是应该重新激活这个特定的 `block_id`。

所以 qwen3.6 把 `_allocate_block` 改成接收 `block_id`，可以统一处理：

```text
1. 分配全新的空闲 block
2. 激活 prefix cache 命中的空闲 block
```

---

## 5. 改动二：`can_allocate()` 从返回缓存块数变成只返回 bool

### 5.1 原版 `can_allocate`

原版：

```python
def can_allocate(self, seq: Sequence) -> int:
    ...
    if len(self.free_block_ids) < num_new_blocks:
        return -1
    return num_cached_blocks
```

它返回的是：

```text
-1：不能分配
非负整数：可以分配，并且有多少个 prefix block 命中
```

因此原版 scheduler 可以这样使用：

```text
num_cached_blocks = can_allocate(seq)
num_tokens = seq.num_tokens - num_cached_blocks * block_size
allocate(seq, num_cached_blocks)
```

也就是说，原版调度器在真正 allocate 前，就知道：

```text
这个请求还有多少 token 需要 prefill。
```

### 5.2 qwen3.6 版本 `can_allocate`

qwen3.6 版本：

```python
def can_allocate(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= seq.num_blocks
```

它只做一件事：

```text
当前空闲 block 数量是否至少覆盖该请求需要的 block 数。
```

不再返回 `num_cached_blocks`。

### 5.3 这个改动的意义

这说明 qwen3.6 版本把 `can_allocate()` 的职责简化了：

```text
原版 can_allocate：
    资源准入检查 + prefix cache 命中分析

qwen3.6 can_allocate：
    只做资源准入检查
```

这种改法让 scheduler 的逻辑更简单，也让 prefix cache 的实际处理下沉到 `allocate()` 内部。

但它也有一个明显取舍：

```text
原版能够更精细地利用 prefix cache，提前扣除可复用 block；
qwen3.6 版本更保守，需要空闲 block 数至少覆盖整个请求的 block 数。
```

这在普通文本模型中可能降低 prefix cache 的资源利用效率，但在 hybrid 模型中反而更安全，因为 hybrid 模型不仅有 KV Cache，还有 GDN recurrent state。只复用 KV Cache 而不复用对应的 recurrent state，可能导致模型状态不一致。

---

## 6. 改动三：`allocate()` 新增 `disable_prefix_cache`

### 6.1 原版 `allocate`

原版：

```python
def allocate(self, seq: Sequence, num_cached_blocks: int):
    ...
```

原版需要 scheduler 提前传入：

```text
num_cached_blocks
```

然后它先复用这些缓存块，再为剩余部分分配新 block。

### 6.2 qwen3.6 版本 `allocate`

qwen3.6 版本：

```python
def allocate(self, seq: Sequence, disable_prefix_cache: bool = False):
    ...
    cache_miss = False or disable_prefix_cache
```

新增了：

```text
disable_prefix_cache
```

这个参数非常关键。

它允许上层调度器告诉 BlockManager：

```text
这次分配不要尝试复用 prefix cache。
```

### 6.3 为什么 Qwen3.6 / hybrid 需要禁用 prefix cache

Qwen3 dense 这类普通 attention-only 模型中，历史状态主要是 KV Cache，所以 prefix cache 复用比较自然：

```text
相同 prefix token
  ↓
复用历史 K/V
```

但 Qwen3.5 / Qwen3.6 hybrid 架构中，除了 full attention 层，还有 GatedDeltaNet / recurrent 类层。它们会维护类似：

```text
recurrent state
conv state
delta state
```

这类状态不是简单按 token block 存在 KV Cache 里。

如果只复用 KV Cache，而没有同步复用或重建 GDN recurrent state，就会出现：

```text
Attention 层认为 prefix 已经缓存
GDN 层却没有对应的 recurrent state
```

最终导致模型 forward 的历史状态不一致。

因此 qwen3.6 版本通过：

```python
allocate(seq, disable_prefix_cache=self.is_hybrid)
```

让 hybrid 模型可以禁用 prefix cache。

这不是为了让系统更快，而是为了让系统更正确。

---

## 7. 改动四：prefix cache 判断从 `can_allocate` 转移到 `allocate`

### 7.1 原版逻辑

原版是两阶段：

```text
can_allocate:
    计算 num_cached_blocks

allocate:
    根据 num_cached_blocks 复用前缀块
```

也就是说，prefix cache 命中分析发生在 `can_allocate()`。

### 7.2 qwen3.6 版本逻辑

qwen3.6 中，prefix cache 判断发生在 `allocate()`：

```python
h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
block_id = self.hash_to_block_id.get(h, -1)
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    cache_miss = True
```

如果命中：

```python
seq.num_cached_tokens += self.block_size
```

如果 miss：

```python
block_id = self.free_block_ids[0]
block = self._allocate_block(block_id)
```

### 7.3 这个改动的意义

这相当于把逻辑从：

```text
先算能复用多少，再分配
```

改成：

```text
分配过程中逐块判断是否还能复用
```

更直观地说：

```text
从第 0 个 block 开始看：
    如果前缀一直命中，就继续复用；
    一旦某个 block miss，后面的 block 全部按新 block 分配。
```

这和 prefix cache 的语义一致：prefix cache 只能复用连续前缀，不能跳着复用中间某些块。

---

## 8. 改动五：删除 `hash_blocks()`，把 hash 更新合并进 `allocate()` / `may_append()`

### 8.1 原版 `hash_blocks`

原版中有专门函数：

```python
def hash_blocks(self, seq: Sequence):
    ...
```

它会根据：

```text
seq.num_cached_tokens
seq.num_scheduled_tokens
```

计算这次新完成了哪些完整 block，然后把这些 block 写入 hash 表。

这个设计和 chunked prefill 很匹配：

```text
本轮只 prefill 一部分 token
  ↓
只有完整完成的 block 才能 hash
  ↓
未完整 block 不进入 prefix cache
```

### 8.2 qwen3.6 版本删除了 `hash_blocks`

qwen3.6 版本没有 `hash_blocks()`。

取而代之的是两处 hash 更新：

第一，在 `allocate()` 中：

```python
if h != -1:
    block.update(h, token_ids)
    self.hash_to_block_id[h] = block_id
```

第二，在 `may_append()` 中，当 decode 追加 token 后刚好填满一个 block 时：

```python
elif len(seq) % self.block_size == 0:
    ...
    h = self.compute_hash(token_ids, prefix)
    last_block.update(h, token_ids)
    self.hash_to_block_id[h] = last_block.block_id
```

### 8.3 这个改动的意义

qwen3.6 版本把 hash 维护变得更内聚：

```text
allocate 阶段：处理 prompt / prefill 初始 block 的 hash
may_append 阶段：处理 decode 过程中刚填满 block 的 hash
```

这样 scheduler 不需要再额外调用：

```python
block_manager.hash_blocks(seq)
```

从模块职责上看：

```text
BlockManager 自己维护 block hash
Scheduler 不需要知道什么时候 hash block
```

这让调度器更干净，但也意味着 `BlockManager` 对“什么时候一个 block 可以被 hash”承担了更多责任。

---

## 9. 改动六：`may_append()` 从只追加 block 变成同时维护 hash

### 9.1 原版 `may_append`

原版：

```python
def may_append(self, seq: Sequence):
    if len(seq) % self.block_size == 1:
        seq.block_table.append(self._allocate_block())
```

含义是：

```text
decode 阶段如果刚进入一个新 block，就分配一个新的 KV Cache block。
```

它只做追加，不做 hash。

### 9.2 qwen3.6 版本 `may_append`

qwen3.6 版本：

```python
def may_append(self, seq: Sequence):
    block_table = seq.block_table
    last_block = self.blocks[block_table[-1]]
    if len(seq) % self.block_size == 1:
        ...
        block_table.append(block_id)
    elif len(seq) % self.block_size == 0:
        ...
        last_block.update(h, token_ids)
        self.hash_to_block_id[h] = last_block.block_id
    else:
        assert last_block.hash == -1
```

它分三种情况：

| 条件 | 含义 | 操作 |
|---|---|---|
| `len(seq) % block_size == 1` | 新 token 进入一个新 block 的第一个位置 | 分配新 block |
| `len(seq) % block_size == 0` | 当前 block 刚好被填满 | 计算 hash 并写入 hash 表 |
| 其他 | 当前 block 未满 | 保持 hash 为 -1 |

### 9.3 为什么未满 block 的 hash 必须是 `-1`

prefix cache 只能复用完整 block。

如果一个 block 还没满：

```text
它对应的 token 内容还会继续变化。
```

此时如果提前写入 hash 表，后续追加 token 后 hash 就不再代表完整 block 内容了。

所以 qwen3.6 版本通过：

```python
assert last_block.hash == -1
```

强调：

```text
未满 block 不应该进入 prefix cache hash 表。
```

这和原版设计是一致的，只是 hash 更新位置不同。

---

## 10. 改动七：`can_allocate()` 更保守

qwen3.6 版本：

```python
return len(self.free_block_ids) >= seq.num_blocks
```

这个判断没有考虑已有 prefix cache 命中。

例如一个请求需要 10 个 block，但前 8 个 block 可以 prefix cache 复用。

原版可能只需要 2 个新 block：

```text
num_new_blocks = 2
```

qwen3.6 的 `can_allocate()` 会要求：

```text
free_block_ids >= 10
```

这明显更保守。

为什么可以接受？

因为 qwen3.6 的重点是支持 hybrid 模型，而 scheduler 在 hybrid 下会禁用 prefix cache。既然 hybrid 下本来就不复用 prefix cache，那么保守判断是合理的：

```text
所有 block 都按新分配考虑
```

对于非 hybrid 模型，这个设计可能牺牲一部分 prefix cache 的极致利用率。可以理解为：

```text
为了统一 hybrid 路径和降低复杂度，qwen3.6 版本弱化了原版 prefix cache 的精细调度能力。
```

---

## 11. 改动八：对旧 hash 映射的处理方式不同

原版 `_allocate_block()` 中有：

```python
if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
    del self.hash_to_block_id[block.hash]
```

也就是说，如果一个 free block 之前有 hash，并且 hash 表还指向它，那么在重新分配这个 block 前，原版会删除旧 hash 映射。

qwen3.6 版本 `_allocate_block(block_id)` 没有这段删除逻辑，而是直接：

```python
block.reset()
```

然后后续在 `allocate()` 或 `may_append()` 中写入新的 hash。

这有一个潜在影响：

```text
hash_to_block_id 里可能暂时保留指向已被 reset block 的旧 hash。
```

不过 qwen3.6 版本在判断 prefix cache 命中时，还会检查：

```python
self.blocks[block_id].token_ids != token_ids
```

因此即使 hash 表里有旧映射，只要 block 的 token 内容已经变化，也不会被误认为命中。

换句话说：

```text
正确性主要靠 hash + token_ids 双重校验保证。
```

但从工程洁净度来说，原版对旧 hash 映射的清理更主动；qwen3.6 版本更依赖命中时的 token_ids 校验。

---

## 12. 和 Scheduler 的联动变化

前面你已经对比过 `scheduler.py`。这两个文件要放在一起看。

原版 scheduler 调用链是：

```text
num_cached_blocks = block_manager.can_allocate(seq)
num_tokens = seq.num_tokens - num_cached_blocks * block_size
block_manager.allocate(seq, num_cached_blocks)
...
postprocess:
    block_manager.hash_blocks(seq)
```

qwen3.6 scheduler 调用链是：

```text
if not block_manager.can_allocate(seq):
    break

block_manager.allocate(seq, disable_prefix_cache=self.is_hybrid)
...
postprocess:
    不再调用 hash_blocks
```

这说明 qwen3.6 版本将三件事从 scheduler 中拿掉或弱化：

```text
1. scheduler 不再拿 num_cached_blocks
2. scheduler 不再显式传 num_cached_blocks 给 allocate
3. scheduler 不再显式调用 hash_blocks
```

对应地，BlockManager 自己承担了更多内部状态维护。

---

## 13. 和 Qwen3.6 hybrid / GDN state 的关系

`block_manager.py` 本身只管理 KV Cache block，不管理 GDN state slot。

GDN state slot 是在你前面看到的 `sequence.py` 和 `scheduler.py` 中体现的：

```text
Sequence.state_slot_id
Scheduler.StateSlotManager
```

但是 `block_manager.py` 和 hybrid 的关系体现在：

```python
disable_prefix_cache
```

也就是：

```text
hybrid 模型可以禁用 prefix cache。
```

为什么这个开关重要？

因为 hybrid 模型的历史状态不是只有 KV Cache：

| 层类型 | 历史状态 |
|---|---|
| Attention 层 | KV Cache |
| GatedDeltaNet 层 | recurrent state / conv state |

如果 prefix cache 只复用 Attention 的 KV Cache，但没有同步复用 GDN state，那么模型状态是不完整的。

因此 qwen3.6 版本没有贸然沿用原版 prefix cache，而是在 BlockManager 层提供禁用入口，让 Scheduler 在 hybrid 模型中强制走完整 prefill / state 重建路径。

这体现了推理系统中一个重要原则：

```text
缓存优化必须服从模型状态一致性。
```

---

## 14. 原版与 qwen3.6 的流程对比

### 14.1 原版 prefix cache 流程

```text
新请求进入 waiting 队列
  ↓
Scheduler 调用 BlockManager.can_allocate(seq)
  ↓
BlockManager 计算 prefix cache 命中多少完整 block
  ↓
返回 num_cached_blocks
  ↓
Scheduler 根据命中块数减少 prefill token 数
  ↓
allocate(seq, num_cached_blocks)
  ↓
prefill 执行
  ↓
postprocess 调用 hash_blocks(seq)
  ↓
新完成的完整 block 加入 prefix cache
```

### 14.2 qwen3.6 流程

```text
新请求进入 waiting 队列
  ↓
Scheduler 调用 BlockManager.can_allocate(seq)
  ↓
只判断 free block 是否足够
  ↓
allocate(seq, disable_prefix_cache=self.is_hybrid)
  ↓
如果 disable_prefix_cache=True：
      不复用 prefix cache，直接分配新 block
  如果 disable_prefix_cache=False：
      allocate 内部尝试逐块命中 prefix cache
  ↓
prefill / decode 执行
  ↓
decode 中 may_append 负责新 block 追加和满 block hash 更新
```

---

## 15. 对初学者最重要的理解

这个文件可以用三句话记住。

第一：

```text
BlockManager 管的是 KV Cache 的“页表”和“内存分配”，不是模型计算。
```

第二：

```text
原版 BlockManager 更关注 prefix cache 复用效率，所以 can_allocate 会返回 num_cached_blocks，hash_blocks 会在 postprocess 后精确更新。
```

第三：

```text
qwen3.6 BlockManager 更关注 hybrid 模型状态一致性，所以提供 disable_prefix_cache，并把 hash 维护合并进 allocate / may_append，整体逻辑更直接但 prefix cache 调度更保守。
```

---

## 16. 改动总结表

| 位置 | 原版设计 | qwen3.6 设计 | 工程意义 |
|---|---|---|---|
| `_allocate_block` | 从 free 队首分配 | 指定 block_id 分配 | 支持激活任意 free block |
| `can_allocate` | 返回 `num_cached_blocks` 或 `-1` | 返回 bool | 简化 scheduler 对 prefix cache 的依赖 |
| `allocate` | 根据外部传入 cached blocks 分配 | 内部判断 cache hit/miss | prefix cache 逻辑下沉 |
| `disable_prefix_cache` | 无 | 新增 | hybrid 模型下关闭不安全 KV-only prefix cache |
| `hash_blocks` | 独立函数 | 删除 | hash 维护内聚到 BlockManager 分配/追加流程 |
| `may_append` | 只追加新 block | 追加新 block + 满 block hash | decode 阶段自动维护 block hash |
| 资源判断 | 考虑 cached block 节省新分配 | 要求 free block 覆盖整个请求 | 更保守，更适合禁用 prefix cache 的 hybrid 路径 |
| 旧 hash 清理 | 分配时主动删除旧 hash 映射 | 依赖 token_ids 校验避免误命中 | 正确性仍可保证，但清理策略不同 |

---

## 17. 面试角度回答

如果面试官问：

> qwen3.6 版本的 `block_manager.py` 相比原版改了什么？

可以这样回答：

`block_manager.py` 负责 KV Cache block 的分配、释放、引用计数和 prefix cache hash 管理。原版 BlockManager 的 prefix cache 逻辑更精细，`can_allocate()` 会先计算能命中多少个 prefix block，并返回 `num_cached_blocks`，scheduler 再根据这个值减少 prefill token 数；prefill 完成后再通过 `hash_blocks()` 把新完成的完整 block 写入 hash 表。qwen3.6 版本做了简化和重构：`can_allocate()` 只返回是否有足够 free block，`allocate()` 内部自己判断 prefix cache 命中，并新增 `disable_prefix_cache` 参数。这个参数很关键，因为 Qwen3.6 hybrid 模型除了 Attention KV Cache，还有 GatedDeltaNet 的 recurrent/conv state，如果只复用 KV Cache 而没有同步复用 GDN state，会造成状态不一致，所以 hybrid 下需要禁用 prefix cache。此外，qwen3.6 删除了单独的 `hash_blocks()`，把 hash 更新合并进 `allocate()` 和 `may_append()`，让 BlockManager 自己维护 block hash。整体来看，这个文件的改动不是新增 GDN 计算，而是让 KV Cache block 管理适配 hybrid 模型的状态一致性要求。

---

## 18. 最终结论

qwen3.6 版本 `block_manager.py` 的核心改造可以概括为：

```text
从“精细 prefix cache 复用优先”
转向
“支持 hybrid 状态一致性优先”
```

它保留了原版的基本 KV Cache block 管理能力：

```text
block 分配
block 释放
ref_count
hash_to_block_id
free / used block 集合
```

但修改了 prefix cache 相关的关键路径：

```text
can_allocate 不再返回 num_cached_blocks
allocate 新增 disable_prefix_cache
hash_blocks 被移除
may_append 负责满 block hash 更新
```

因此，这个文件在 Qwen3.6 项目中的意义是：

```text
为 hybrid 模型提供更安全的 KV Cache 管理策略，避免 Attention KV Cache 和 GatedDeltaNet recurrent state 之间出现状态不一致。
```

要完整理解它，需要和前面几个文件连起来看：

```text
sequence.py:
    state_slot_id 记录 GDN state 槽位

scheduler.py:
    StateSlotManager 管理 GDN state slot
    hybrid 下调用 allocate(..., disable_prefix_cache=True)

block_manager.py:
    根据 disable_prefix_cache 控制是否复用 KV prefix cache
```

这三者合起来，才构成 qwen3.6 推理系统对 hybrid 架构的基础状态管理改造。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
