# block_manager.py 源码解析

> 解析对象：`nanovllm/engine/block_manager.py`  
> 主题：KV Cache block 管理、Prefix Caching、block_table 映射、显存块分配与释放

---

## 1. 文件整体定位

### 1.1 这个文件负责什么

`block_manager.py` 是 nano-vLLM 中负责 **KV Cache block 管理** 的核心文件。

它不直接执行模型计算，也不直接做 Attention，而是负责管理 Attention 所依赖的历史 KV Cache 存储块。它主要解决以下问题：

1. 每条 `Sequence` 需要多少个 KV Cache block；
2. 当前显存中还有没有足够的空闲 block；
3. 新请求进入 Prefill 时，应该分配哪些 block；
4. Decode 阶段生成新 token 时，是否需要追加新的 block；
5. 请求结束或被抢占时，如何释放 block；
6. 如果多个请求具有相同前缀，是否可以复用已经计算好的 prefix KV block；
7. 如何维护 `seq.block_table`，让每条逻辑 sequence 映射到底层物理 KV block。

一句话概括：

> `BlockManager` 是 nano-vLLM 的 KV Cache “内存管理器”，它把每条请求的 token 序列映射到一组固定大小的物理 cache block 上。

---

### 1.2 它属于哪一层

它属于：

```text
KV Cache 管理层 / Block 管理层 / PagedAttention 支撑层
```

在推理引擎中，它处于 `Scheduler` 和 `Attention` 之间：

```text
Scheduler 决定哪些 Sequence 本轮要运行
        ↓
BlockManager 分配 / 释放 / 复用 KV Cache block
        ↓
ModelRunner 根据 block_table 准备 block_tables、slot_mapping
        ↓
Attention 根据 block_tables 读写 KV Cache
```

---

### 1.3 它和哪些文件有关

| 相关文件 | 关系 |
|---|---|
| `sequence.py` | `BlockManager` 直接操作 `Sequence.block_table`、`Sequence.num_cached_tokens`、`Sequence.num_blocks` |
| `scheduler.py` | 调度器调用 `can_allocate()`、`allocate()`、`can_append()`、`may_append()`、`deallocate()` |
| `model_runner.py` | 根据 `seq.block_table` 准备 `block_tables` 和 `slot_mapping` |
| `attention.py` | 通过 `block_tables` 和 `slot_mapping` 访问真实 KV Cache |
| `config.py` | 提供 `num_kvcache_blocks` 和 `kvcache_block_size` |
| `llm_engine.py` | 间接通过 `Scheduler` 使用该文件 |

---

### 1.4 它在完整推理流程中的位置

完整流程可以抽象成：

```text
用户 prompt
   ↓
Tokenizer 编码为 token ids
   ↓
LLMEngine 创建 Sequence
   ↓
Scheduler.add(seq) 加入 waiting 队列
   ↓
Scheduler.schedule()
   ↓
BlockManager.can_allocate(seq)
   ↓
BlockManager.allocate(seq)
   ↓
ModelRunner.prepare_prefill()
   ↓
Attention 写入 KV Cache
   ↓
Scheduler.postprocess()
   ↓
BlockManager.hash_blocks(seq)
   ↓
Decode 阶段：
   BlockManager.can_append(seq)
   BlockManager.may_append(seq)
   Attention 继续读写 KV Cache
   ↓
请求结束：
   BlockManager.deallocate(seq)
```

所以 `block_manager.py` 是连接 **请求调度** 和 **KV Cache 真实存储** 的关键桥梁。

---

## 2. 代码结构总览

### 2.1 导入模块

源码开头：

```python
from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence
```

导入内容可以分为三类：

| 模块 | 作用 |
|---|---|
| `deque` | 维护空闲 block id 队列，支持高效从左侧取出、右侧归还 |
| `xxhash` | 计算 token block 的快速哈希，用于 prefix caching |
| `numpy` | 将 token id 列表转成字节流，作为 hash 输入 |
| `Sequence` | 读取 sequence 的 token、block 数量和 block_table |

---

### 2.2 定义了哪些类

本文件定义两个类：

```text
Block
BlockManager
```

#### `Block`

表示一个物理 KV Cache block 的元信息。

它不直接保存 GPU 上的 K/V Tensor，而是保存该 block 的管理信息：

```text
block_id
ref_count
hash
token_ids
```

#### `BlockManager`

负责管理所有 block：

```text
blocks
free_block_ids
used_block_ids
hash_to_block_id
```

它提供分配、释放、复用、追加、哈希登记等功能。

---

### 2.3 定义了哪些函数 / 方法

#### `Block` 中的方法

| 方法 | 作用 |
|---|---|
| `__init__()` | 初始化一个 block 的元信息 |
| `update()` | 记录该 block 对应的 hash 和 token_ids |
| `reset()` | 将 block 重置为新分配状态 |

#### `BlockManager` 中的方法

| 方法 | 作用 |
|---|---|
| `__init__()` | 初始化所有 block 和管理结构 |
| `compute_hash()` | 对 token block 计算带 prefix 的哈希 |
| `_allocate_block()` | 从 free 队列中分配一个物理 block |
| `_deallocate_block()` | 释放一个物理 block |
| `can_allocate()` | 判断一个新 sequence 是否能分配 block，同时检查 prefix cache |
| `allocate()` | 为 sequence 分配 block_table |
| `deallocate()` | 释放 sequence 占用的 block |
| `can_append()` | Decode 阶段判断是否有空间追加 token |
| `may_append()` | Decode 阶段必要时分配新 block |
| `hash_blocks()` | 将已经完整计算的 block 加入 prefix cache 哈希表 |

---

### 2.4 重要变量和数据结构

| 变量 | 类型 | 作用 |
|---|---|---|
| `block_size` | `int` | 每个 block 容纳多少 token |
| `blocks` | `list[Block]` | 所有物理 block 的元信息数组 |
| `hash_to_block_id` | `dict[int, int]` | prefix hash 到 block id 的映射 |
| `free_block_ids` | `deque[int]` | 当前空闲的 block id |
| `used_block_ids` | `set[int]` | 当前正在被使用的 block id |
| `seq.block_table` | `list[int]` | 每条 sequence 的逻辑 block 到物理 block 映射 |
| `Block.ref_count` | `int` | 当前 block 被多少条 sequence 引用 |
| `Block.hash` | `int` | 该 block 内容和前缀形成的哈希值 |
| `Block.token_ids` | `list[int]` | 该 block 对应的 token 内容 |

---

### 2.5 主流程和辅助逻辑

主流程方法：

```text
can_allocate()
allocate()
can_append()
may_append()
hash_blocks()
deallocate()
```

辅助方法：

```text
compute_hash()
_allocate_block()
_deallocate_block()
Block.update()
Block.reset()
```

---

### 2.6 文件组织结构图

```text
block_manager.py
├── import
│   ├── deque
│   ├── xxhash
│   ├── numpy
│   └── Sequence
│
├── class Block
│   ├── __init__()
│   ├── update()
│   └── reset()
│
└── class BlockManager
    ├── __init__()
    ├── compute_hash()
    ├── _allocate_block()
    ├── _deallocate_block()
    ├── can_allocate()
    ├── allocate()
    ├── deallocate()
    ├── can_append()
    ├── may_append()
    └── hash_blocks()
```

---

## 3. 逐行 / 逐代码块解释

---

### 3.1 导入依赖

```python
from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence
```

#### 语法作用

这几行导入本文件需要使用的标准库、第三方库和项目内部类。

#### 工程作用

`deque` 用于维护空闲 block id。相比普通 list，`deque.popleft()` 是 O(1)，适合频繁从队列头部取出 block。

`xxhash` 是高速非加密哈希库，用来快速给 token block 计算指纹。这里不是做安全加密，而是为了快速判断两个 token block 是否可能相同。

`numpy` 用来把 `list[int]` 转成连续字节流，方便传给 `xxhash`。

`Sequence` 是推理引擎中单条请求的状态对象。`BlockManager` 需要从 `Sequence` 里读取：

```text
seq.num_blocks
seq.block(i)
seq.block_table
seq.num_cached_tokens
seq.num_scheduled_tokens
```

#### 在 nano-vLLM 中的意义

这个文件管理的是 Sequence 到 KV Cache block 的映射，所以必须依赖 `Sequence`。

下一步需要结合：

```text
sequence.py
scheduler.py
attention.py
```

---

## 3.2 `Block` 类

### 3.2.1 类定义

```python
class Block:
```

#### 语法作用

定义一个普通 Python 类。

#### 工程作用

`Block` 表示一个物理 KV Cache block 的管理元信息。

注意：这里的 `Block` 不是 GPU Tensor 本身。真正的 KV Cache Tensor 通常在 `model_runner.py` 中分配，形状类似：

```text
[num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

本类只是记录“这个 block 当前是否空闲、被谁引用、对应哪些 token”。

---

### 3.2.2 初始化 Block

```python
def __init__(self, block_id):
    self.block_id = block_id
    self.ref_count = 0
    self.hash = -1
    self.token_ids = []
```

#### 语法作用

构造一个 `Block` 实例，并保存 block 的编号、引用计数、哈希值和 token 内容。

#### 工程作用

字段含义如下：

| 字段 | 类型 | 初始值 | 含义 |
|---|---|---|---|
| `block_id` | `int` | 传入值 | 物理 block 编号 |
| `ref_count` | `int` | `0` | 有多少条 Sequence 正在引用该 block |
| `hash` | `int` | `-1` | 该 block 的 prefix-aware hash |
| `token_ids` | `list[int]` | `[]` | 该 block 对应的 token 内容 |

`ref_count = 0` 表示当前 block 没有被任何请求使用。

`hash = -1` 表示当前 block 还没有登记到 prefix cache 中。

#### 在推理流程中的意义

当某条请求进入 Prefill 或 Decode 时，调度器会通过 `BlockManager` 为它申请 block。申请到的 block id 会写入：

```text
seq.block_table
```

而 `Block` 对象记录这个 block 的管理状态。

---

### 3.2.3 更新 Block 元信息

```python
def update(self, hash: int, token_ids: list[int]):
    self.hash = hash
    self.token_ids = token_ids
```

#### 语法作用

定义一个带类型注解的方法：

```text
hash: int
token_ids: list[int]
```

表示输入参数分别是整型哈希值和 token id 列表。

#### 工程作用

当某个 block 已经完成计算，并且可以作为 prefix cache 被复用时，需要记录：

```text
该 block 的 hash
该 block 对应的 token_ids
```

后续新请求如果前缀 token 与该 block 一致，就可以复用这个 block。

#### 为什么不仅保存 hash，还保存 token_ids

因为哈希理论上存在碰撞。虽然 `xxhash64` 碰撞概率很低，但代码仍然做了双重校验：

```text
先比 hash
再比 token_ids
```

这在 `can_allocate()` 中体现：

```python
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    break
```

这保证 prefix cache 的复用更加稳妥。

---

### 3.2.4 重置 Block

```python
def reset(self):
    self.ref_count = 1
    self.hash = -1
    self.token_ids = []
```

#### 语法作用

将 block 重置为“刚被分配给某个 Sequence”的状态。

#### 工程作用

`reset()` 在 `_allocate_block()` 中调用。含义是：

```text
这个 block 现在被新请求占用
引用计数设为 1
清空旧 hash
清空旧 token_ids
```

注意这里 `ref_count` 不是重置为 0，而是 1。

原因是 `_allocate_block()` 表示已经把这个 block 分配给某个 sequence，所以此时至少有一个引用者。

#### 在 KV Cache 管理中的意义

block 释放后可能还保留旧 hash 和 token_ids，用于 prefix cache 查找。但如果它后来被重新分配给新内容，那么旧 hash 就必须删除或失效，否则会把旧 token 前缀错误映射到新 block。

---

## 3.3 `BlockManager` 类

### 3.3.1 类定义

```python
class BlockManager:
```

#### 语法作用

定义 KV Cache block 管理器类。

#### 工程作用

`BlockManager` 统一管理所有物理 KV Cache block。它负责：

```text
分配 block
释放 block
复用 prefix cache block
维护 seq.block_table
判断 decode 是否需要新 block
登记已完成 block 的 hash
```

它是 `Scheduler` 进行请求调度时最重要的依赖之一。

---

### 3.3.2 初始化 BlockManager

```python
def __init__(self, num_blocks: int, block_size: int):
    self.block_size = block_size
    self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
    self.hash_to_block_id: dict[int, int] = dict()
    self.free_block_ids: deque[int] = deque(range(num_blocks))
    self.used_block_ids: set[int] = set()
```

#### 语法作用

构造函数接收两个参数：

| 参数 | 类型 | 含义 |
|---|---|---|
| `num_blocks` | `int` | 总共有多少个 KV Cache block |
| `block_size` | `int` | 每个 block 能容纳多少 token |

字段带有类型注解：

```python
self.blocks: list[Block]
self.hash_to_block_id: dict[int, int]
self.free_block_ids: deque[int]
self.used_block_ids: set[int]
```

#### 工程作用

假设：

```text
num_blocks = 100
block_size = 256
```

那么这个 BlockManager 会管理 100 个物理 KV Cache block，每个 block 可以容纳 256 个 token 的 K/V cache。

各字段含义：

| 字段 | 含义 |
|---|---|
| `self.block_size` | 每个 block 的 token 容量 |
| `self.blocks` | 所有 block 的元信息列表，`blocks[i]` 表示 block id 为 i 的块 |
| `self.hash_to_block_id` | prefix hash 到 block_id 的映射，用于 prefix caching |
| `self.free_block_ids` | 当前空闲 block id 队列 |
| `self.used_block_ids` | 当前正在使用的 block id 集合 |

#### 为什么同时有 `free_block_ids` 和 `used_block_ids`

`free_block_ids` 适合快速分配：

```text
从队列头部 popleft 一个空闲 block
```

`used_block_ids` 适合快速判断某个 block 是否正在使用：

```text
block_id in used_block_ids
```

两者组合能同时满足：

```text
快速取空闲块
快速判断块是否被占用
```

#### 在推理流程中的意义

这个初始化通常在 `Scheduler.__init__()` 中发生：

```text
Scheduler 创建 BlockManager
        ↓
BlockManager 管理 KV Cache block
        ↓
调度 prefill/decode 时申请和释放 block
```

---

## 3.4 `compute_hash()`：计算 prefix-aware block hash

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

#### 语法作用

这是一个类方法，用于根据 token ids 和前缀 hash 计算当前 block 的 hash。

参数：

| 参数 | 类型 | 含义 |
|---|---|---|
| `token_ids` | `list[int]` | 当前 block 中的 token id |
| `prefix` | `int` | 前一个 block 的 hash，默认 `-1` 表示没有前缀 |

返回值：

```text
int 类型的 64-bit hash digest
```

#### 工程作用

这个函数的关键不是只对当前 block 做 hash，而是把前一个 block 的 hash 也加入进来。

也就是说，第 i 个 block 的 hash 依赖于：

```text
前面所有 block 的内容
+
当前 block 的内容
```

形成链式哈希：

```text
h0 = hash(block0)
h1 = hash(h0, block1)
h2 = hash(h1, block2)
...
```

这样做的好处是：即使两个 block 的 token 内容相同，但如果前缀不同，它们的 hash 也不同。

#### 为什么要 prefix-aware

假设有两个 sequence：

```text
seq A:
block0 = [1, 2, 3]
block1 = [4, 5, 6]

seq B:
block0 = [9, 9, 9]
block1 = [4, 5, 6]
```

虽然 `block1` 内容都是 `[4, 5, 6]`，但它们前面的上下文不同，Attention 的 KV Cache 不能只按当前 block 内容随便复用。

因此 hash 必须和前缀绑定。

#### 在 nano-vLLM 推理流程中的意义

这个函数支撑 prefix caching：

```text
相同系统提示词 / 相同长前缀
        ↓
对应完整 block 的 hash 相同
        ↓
可以复用已有 KV Cache block
        ↓
减少 Prefill 计算
```

---

## 3.5 `_allocate_block()`：分配一个物理 block

```python
def _allocate_block(self) -> int:
    block_id = self.free_block_ids.popleft()
    block = self.blocks[block_id]
    assert block.ref_count == 0
    if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
        del self.hash_to_block_id[block.hash]
    block.reset()
    self.used_block_ids.add(block_id)
    return block_id
```

#### 语法作用

定义一个内部方法，返回值类型为 `int`，表示分配出来的 block id。

#### 工程作用

分配流程：

```text
1. 从 free_block_ids 左侧取出一个空闲 block_id
2. 找到对应 Block 对象
3. 确认 ref_count == 0
4. 如果它还残留在 hash_to_block_id 中，则删除旧 hash 映射
5. reset block，让 ref_count = 1
6. 加入 used_block_ids
7. 返回 block_id
```

#### 为什么要删除旧 hash 映射

一个 block 释放后可能仍然保留旧的 `hash` 和 `token_ids`，用于 prefix cache。

但一旦这个 block 被重新分配给新 sequence，它的物理 KV 内容即将被覆盖。此时旧 hash 不能继续指向它，否则后续请求可能错误复用已经被覆盖的 KV Cache。

所以这里执行：

```python
if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
    del self.hash_to_block_id[block.hash]
```

这表示：

```text
如果 hash 表中仍然认为这个 hash 对应当前 block
那么这个映射必须失效
```

#### 在推理流程中的意义

`_allocate_block()` 通常由以下方法间接调用：

```text
allocate()
may_append()
```

分别对应：

```text
Prefill 阶段初始化 block_table
Decode 阶段追加新 token 时可能新增 block
```

---

## 3.6 `_deallocate_block()`：释放一个物理 block

```python
def _deallocate_block(self, block_id: int):
    assert self.blocks[block_id].ref_count == 0
    self.used_block_ids.remove(block_id)
    self.free_block_ids.append(block_id)
```

#### 语法作用

内部方法，接收一个整数类型的 `block_id`。

#### 工程作用

释放流程：

```text
1. 确认该 block 的引用计数已经为 0
2. 从 used_block_ids 中移除
3. 放回 free_block_ids 队列尾部
```

#### 为什么要求 `ref_count == 0`

一个 block 可能被多个 sequence 复用，特别是 prefix cache 场景。

例如：

```text
seq A 和 seq B 共享同一个系统 prompt 的前几个 block
```

那么这些 block 的 `ref_count` 可能大于 1。

只有当所有引用它的 sequence 都结束或释放后，`ref_count` 才能降到 0，此时才允许真正放回空闲队列。

#### 在 KV Cache 管理中的意义

这是避免“提前释放共享 KV Cache”的核心保护。

---

## 3.7 `can_allocate()`：判断新 sequence 是否能分配 block

```python
def can_allocate(self, seq: Sequence) -> int:
    h = -1
    num_cached_blocks = 0
    num_new_blocks = seq.num_blocks
    for i in range(seq.num_blocks - 1):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break
        num_cached_blocks += 1
        if block_id in self.used_block_ids:
            num_new_blocks -= 1
    if len(self.free_block_ids) < num_new_blocks:
        return -1
    return num_cached_blocks
```

#### 语法作用

方法接收一个 `Sequence`，返回一个整数：

| 返回值 | 含义 |
|---|---|
| `-1` | 当前空闲 block 不够，不能分配 |
| `>= 0` | 可以分配，其中数值表示可复用的 cached blocks 数量 |

#### 工程作用

这个方法同时做两件事：

```text
1. 检查这个 sequence 前面有多少个完整 block 可以复用 prefix cache
2. 检查当前 free_block_ids 是否足够分配剩余新 block
```

#### 关键逻辑解释

初始化：

```python
h = -1
num_cached_blocks = 0
num_new_blocks = seq.num_blocks
```

含义：

```text
h = 当前前缀 hash
num_cached_blocks = 已经命中的 prefix cache block 数
num_new_blocks = 默认认为所有 block 都需要新分配
```

循环：

```python
for i in range(seq.num_blocks - 1):
```

这里只检查到 `seq.num_blocks - 1`，也就是不检查最后一个 block。

原因是最后一个 block 可能是不完整 block。Prefix cache 一般只缓存完整 block，因为不完整 block 后续还会追加 token，内容未稳定。

例如：

```text
block_size = 256
seq 长度 = 600

block0: 256 token，完整，可缓存
block1: 256 token，完整，可缓存
block2: 88 token，不完整，不缓存
```

取当前 block token：

```python
token_ids = seq.block(i)
```

计算链式 hash：

```python
h = self.compute_hash(token_ids, h)
```

查找是否已有相同 prefix block：

```python
block_id = self.hash_to_block_id.get(h, -1)
```

判断是否命中：

```python
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    break
```

命中条件是：

```text
hash 存在
并且 token_ids 完全一致
```

命中后：

```python
num_cached_blocks += 1
```

如果这个 block 当前正在被使用：

```python
if block_id in self.used_block_ids:
    num_new_blocks -= 1
```

说明这个 block 可以共享，不需要额外消耗一个新 free block。

#### 一个例子

假设：

```text
seq.num_blocks = 4
前 2 个完整 block 命中 prefix cache
第 3 个不命中
第 4 个是不完整 block
```

那么：

```text
num_cached_blocks = 2
```

如果这 2 个 cached block 当前仍在 used_block_ids 中，就可以直接增加 ref_count 共享，不需要新申请。

#### 为什么返回 `num_cached_blocks`

因为 `allocate()` 需要知道：

```text
前多少个 block 直接复用
后多少个 block 重新分配
```

#### 在 Scheduler 中的意义

`Scheduler.schedule()` 中会调用：

```text
num_cached_blocks = block_manager.can_allocate(seq)
```

如果返回 `-1`，表示 KV Cache 显存不够，本轮不能调度这个 sequence。

如果返回非负数，则可以继续 `allocate()`。

---

## 3.8 `allocate()`：为 Sequence 建立 block_table

```python
def allocate(self, seq: Sequence, num_cached_blocks: int):
    assert not seq.block_table
    h = -1
    for i in range(num_cached_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id[h]
        block = self.blocks[block_id]
        if block_id in self.used_block_ids:
            block.ref_count += 1
        else:
            block.ref_count = 1
            self.free_block_ids.remove(block_id)
            self.used_block_ids.add(block_id)
        seq.block_table.append(block_id)
    for i in range(num_cached_blocks, seq.num_blocks):
        seq.block_table.append(self._allocate_block())
    seq.num_cached_tokens = num_cached_blocks * self.block_size
```

#### 语法作用

方法接收：

| 参数 | 类型 | 含义 |
|---|---|---|
| `seq` | `Sequence` | 要分配 block 的请求 |
| `num_cached_blocks` | `int` | 可以复用的 prefix cache block 数量 |

无显式返回值，直接修改 `seq.block_table` 和 block 管理结构。

#### 工程作用

`allocate()` 真正执行 block 分配。它分成两段：

```text
1. 前 num_cached_blocks 个 block：复用 prefix cache
2. 剩余 block：新分配 free block
```

#### 第一段：复用 cached block

```python
assert not seq.block_table
```

要求这个 sequence 还没有 block_table。也就是说，不能重复初始化分配。

```python
for i in range(num_cached_blocks):
```

遍历可以复用的 prefix block。

```python
token_ids = seq.block(i)
h = self.compute_hash(token_ids, h)
block_id = self.hash_to_block_id[h]
```

重新计算 hash，并从 hash 表中找到对应 block id。

```python
if block_id in self.used_block_ids:
    block.ref_count += 1
```

如果该 block 正在被其他 sequence 使用，则增加引用计数。

```python
else:
    block.ref_count = 1
    self.free_block_ids.remove(block_id)
    self.used_block_ids.add(block_id)
```

如果该 block 当前不在 used 中，但仍然通过 hash 表保留着缓存信息，则需要从 free 队列移到 used 集合。

这说明 nano-vLLM 保留了一种缓存状态：

```text
block 已经不被使用
但 hash/token_ids 还在
如果新请求命中，可以重新激活
```

```python
seq.block_table.append(block_id)
```

将这个物理 block id 写入 sequence 的 block table。

#### 第二段：新分配 block

```python
for i in range(num_cached_blocks, seq.num_blocks):
    seq.block_table.append(self._allocate_block())
```

从缓存命中结束的位置开始，到 sequence 需要的总 block 数为止，全部新分配。

#### 更新 cached token 数

```python
seq.num_cached_tokens = num_cached_blocks * self.block_size
```

这表示：

```text
前 num_cached_blocks 个完整 block 已经有 KV Cache
这些 token 不需要重新 prefill
```

例如：

```text
num_cached_blocks = 3
block_size = 256

seq.num_cached_tokens = 768
```

#### 在推理流程中的意义

分配结束后：

```text
seq.block_table = [物理block0, 物理block1, 物理block2, ...]
```

后面的 `ModelRunner` 会根据 `seq.block_table` 准备 `block_tables` 和 `slot_mapping`，让 Attention 知道每个 token 的 K/V 应该写到哪里。

---

## 3.9 `deallocate()`：释放 Sequence 占用的 block

```python
def deallocate(self, seq: Sequence):
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

#### 语法作用

接收一个 `Sequence`，释放其 `block_table` 中引用的所有 block。

#### 工程作用

释放流程：

```text
1. 反向遍历 seq.block_table
2. 每个 block 的 ref_count 减 1
3. 如果 ref_count 降到 0，则真正释放 block
4. 清空 seq.num_cached_tokens
5. 清空 seq.block_table
```

#### 为什么 reversed

反向释放更符合 sequence token 的逻辑顺序：后面的 block 通常是最新分配、最新生成的 block。

虽然这段代码中反向释放不是绝对必要，但在 KV Cache 管理中反向释放通常更符合“尾部追加、尾部释放”的直觉。

#### 为什么不是直接释放所有 block

因为 prefix cache 下，一个 block 可能被多个 sequence 共享。

例如：

```text
seq A block_table = [1, 2, 3]
seq B block_table = [1, 2, 8]
```

block 1、2 被两个请求共享。

释放 seq A 时：

```text
block 1 ref_count: 2 → 1，不释放
block 2 ref_count: 2 → 1，不释放
block 3 ref_count: 1 → 0，释放
```

这就是 `ref_count` 的作用。

#### 在推理流程中的意义

`deallocate()` 会在两种情况下被调用：

```text
1. 请求生成结束
2. 请求被 preempt 抢占
```

在 `scheduler.py` 中：

```text
preempt(seq) → block_manager.deallocate(seq)
postprocess() 中 sequence finished → block_manager.deallocate(seq)
```

---

## 3.10 `can_append()`：Decode 阶段是否能追加新 token

```python
def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

#### 语法作用

接收一个 `Sequence`，返回布尔值。

#### 工程作用

Decode 阶段每轮会给 sequence 追加 1 个 token。追加 token 时可能出现两种情况：

```text
1. 当前最后一个 block 还没满
   不需要新 block

2. 当前 token 正好进入一个新 block
   需要新分配一个 block
```

这句代码的核心是：

```python
len(seq) % self.block_size == 1
```

为什么是 `== 1`？

需要结合调用时机理解。

在 `Scheduler.schedule()` 的 Decode 阶段，`can_append(seq)` 是在当前 sequence 已经包含上一次生成结果之后、准备本轮 decode 之前调用的。此时如果：

```text
len(seq) % block_size == 1
```

意味着当前 token 长度已经进入了一个新 block 的第一个位置，需要确认是否已经有或能分配对应 block。

这里的实现和 `may_append()` 配合：

```python
if len(seq) % self.block_size == 1:
    seq.block_table.append(self._allocate_block())
```

也就是说，当 sequence 长度来到新 block 的第一个 token 位置时，需要追加一个物理 block。

#### 返回值如何理解

表达式：

```python
(len(seq) % self.block_size == 1)
```

本身是布尔值，在 Python 中：

```text
True  等价于 1
False 等价于 0
```

所以整句可以理解为：

```text
如果需要新 block，则要求 free_block_ids 至少有 1 个
如果不需要新 block，则要求 free_block_ids 至少有 0 个
```

#### 在 Scheduler 中的意义

如果 `can_append(seq)` 返回 False，说明没有空闲 block 可以支撑这条请求继续 decode。

这时 scheduler 会抢占其他 running sequence，释放它们的 block：

```text
while not can_append(seq):
    preempt(...)
```

---

## 3.11 `may_append()`：必要时追加新 block

```python
def may_append(self, seq: Sequence):
    if len(seq) % self.block_size == 1:
        seq.block_table.append(self._allocate_block())
```

#### 语法作用

接收一个 `Sequence`，无返回值。

#### 工程作用

如果当前 sequence 已经进入一个新 block 的第一个 token 位置，则为它分配一个新的物理 block，并追加到 `seq.block_table`。

#### 和 `can_append()` 的关系

两者是成对使用的：

```text
can_append(seq)
    先判断有没有空闲 block

may_append(seq)
    真正分配新 block
```

为什么要分开？

因为调度器需要先判断资源是否够。如果不够，它可能要先抢占其他请求释放资源。

流程是：

```text
Scheduler decode 阶段
    ↓
can_append(seq)
    ↓
如果不够，preempt 其他 seq
    ↓
资源够了
    ↓
may_append(seq)
    ↓
把 seq 加入本轮 scheduled_seqs
```

---

## 3.12 `hash_blocks()`：把完整 block 登记到 prefix cache

```python
def hash_blocks(self, seq: Sequence):
    start = seq.num_cached_tokens // self.block_size
    end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
    if start == end: return
    h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
    for i in range(start, end):
        block = self.blocks[seq.block_table[i]]
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block.update(h, token_ids)
        self.hash_to_block_id[h] = block.block_id
```

#### 语法作用

接收一个 `Sequence`，根据本轮新完成的 token 范围，给其中完整的 block 计算 hash 并登记。

#### 工程作用

这一步发生在 `Scheduler.postprocess()` 中，通常是在模型已经完成本轮 Prefill/Decode 计算之后。

它负责：

```text
把已经完整计算好的 block 加入 prefix cache
```

#### start 和 end 的含义

```python
start = seq.num_cached_tokens // self.block_size
```

表示本轮之前已经缓存到第几个 block。

```python
end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
```

表示本轮计算完成后，完整缓存到了第几个 block。

例如：

```text
block_size = 256
seq.num_cached_tokens = 256
seq.num_scheduled_tokens = 512

start = 1
end = 3
```

说明本轮让 block1 和 block2 变成完整可缓存状态。

#### 为什么 `start == end` 就直接返回

如果本轮没有新增任何完整 block，就不用登记 hash。

例如：

```text
block_size = 256
seq.num_cached_tokens = 300
seq.num_scheduled_tokens = 1

start = 1
end = 1
```

这说明本轮只是继续填充某个未满 block，没有形成新的完整 block。

#### 取前缀 hash

```python
h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
```

如果不是从第 0 个 block 开始，就要接上前一个 block 的 hash，继续形成链式 hash。

#### 登记每个完整 block

```python
for i in range(start, end):
    block = self.blocks[seq.block_table[i]]
    token_ids = seq.block(i)
    h = self.compute_hash(token_ids, h)
    block.update(h, token_ids)
    self.hash_to_block_id[h] = block.block_id
```

含义是：

```text
1. 找到第 i 个逻辑 block 对应的物理 block
2. 取出这个逻辑 block 的 token_ids
3. 计算 prefix-aware hash
4. 更新 Block 元信息
5. 在 hash_to_block_id 中登记 hash → block_id
```

#### 在推理流程中的意义

这一步使得后续请求可以复用已经计算好的 KV Cache block。

例如：

```text
请求 A:
"你是一个助手，请回答：..."

请求 B:
"你是一个助手，请总结：..."
```

如果前面的系统 prompt 完全相同，并且刚好覆盖若干完整 block，那么 B 的 Prefill 可以复用 A 的这些 block。

---

## 4. 背后的框架性原理

---

### 4.1 KV Cache

#### 是什么

KV Cache 是 Transformer Decode 阶段保存历史 Key / Value 的缓存。

在自回归生成中，每生成一个新 token，都需要让当前 token attend 到前面所有 token。如果每一步都重新计算所有历史 token 的 K/V，成本会非常高。

因此推理引擎会把历史 token 的 K/V 保存下来：

```text
Prefill 阶段：
    计算 prompt 所有 token 的 K/V，写入 KV Cache

Decode 阶段：
    只计算当前 token 的 Q/K/V
    历史 K/V 从 KV Cache 读取
```

#### 为什么重要

KV Cache 是大模型推理中最重要的性能优化之一，同时也是显存大户。

显存占用大致与下面因素成正比：

```text
层数 × token 数 × KV head 数 × head_dim × dtype大小 × 2(K和V)
```

#### 在本文件中如何体现

本文件虽然不直接保存 K/V Tensor，但它管理 KV Cache 的“物理页”：

```text
BlockManager 管理 block id
Sequence.block_table 记录逻辑 token 到物理 block 的映射
```

#### 和整体架构的关系

`BlockManager` 负责决定 KV Cache 放在哪里，`Attention` 负责真正读写 KV Cache。

---

### 4.2 block / block table

#### 是什么

`block` 是固定大小的一段 KV Cache 空间。

`block_table` 是一条 sequence 的逻辑 block 到物理 block 的映射表。

例如：

```text
Sequence token 长度 = 600
block_size = 256

逻辑 block:
0, 1, 2

物理 block:
[7, 13, 2]

seq.block_table = [7, 13, 2]
```

#### 为什么重要

如果所有请求都要求连续显存，显存碎片会非常严重。

使用 block table 后：

```text
逻辑上 token 是连续的
物理上 KV Cache 可以分散在不同 block 中
```

这就是 PagedAttention 的基本思想。

#### 在本文件中如何体现

`allocate()` 会写入：

```python
seq.block_table.append(block_id)
```

`deallocate()` 会清空：

```python
seq.block_table.clear()
```

`hash_blocks()` 会根据：

```python
seq.block_table[i]
```

找到物理 block。

---

### 4.3 Prefix Caching

#### 是什么

Prefix Caching 是指多个请求如果有相同前缀，可以复用前缀对应的 KV Cache，避免重复 Prefill。

#### 为什么重要

大模型服务中，很多请求会共享系统提示词：

```text
你是一个有帮助的助手...
请遵守以下规则...
```

如果每个请求都重新计算这些前缀，会浪费大量 Prefill 计算。

#### 在本文件中如何体现

核心数据结构：

```python
self.hash_to_block_id: dict[int, int]
```

核心函数：

```text
compute_hash()
can_allocate()
allocate()
hash_blocks()
```

核心流程：

```text
hash_blocks() 登记已经计算好的完整 block
can_allocate() 检查新请求是否命中相同前缀
allocate() 复用命中的 block
```

---

### 4.4 request / sequence

#### 是什么

一条用户请求在内部会被表示成一个 `Sequence`。

Sequence 保存：

```text
token_ids
block_table
num_tokens
num_cached_tokens
num_scheduled_tokens
status
```

#### 为什么重要

推理引擎不是一次只处理一个请求，而是同时处理多个请求。每个请求必须有独立状态。

#### 在本文件中如何体现

`BlockManager` 直接操作 `Sequence`：

```text
seq.num_blocks
seq.block(i)
seq.block_table
seq.num_cached_tokens
```

---

### 4.5 Scheduler

#### 是什么

Scheduler 决定每一轮推理处理哪些 Sequence。

#### 为什么重要

LLM 推理是动态的：

```text
有些请求刚进入 Prefill
有些请求正在 Decode
有些请求结束释放 KV Cache
有些请求因显存不足被抢占
```

#### 在本文件中如何体现

`Scheduler` 会调用本文件的方法：

```text
can_allocate()
allocate()
can_append()
may_append()
deallocate()
hash_blocks()
```

也就是说，调度策略在 `scheduler.py`，资源管理在 `block_manager.py`。

---

### 4.6 Prefill / Decode

#### Prefill

Prefill 是处理 prompt 的阶段。需要为 prompt token 分配 KV Cache block。

本文件中的体现：

```text
can_allocate()
allocate()
hash_blocks()
```

#### Decode

Decode 是逐 token 生成阶段。每轮每条 sequence 通常追加一个 token。

本文件中的体现：

```text
can_append()
may_append()
deallocate()
```

---

### 4.7 Attention

#### 是什么

Attention 需要访问历史 token 的 K/V。

#### 为什么重要

Decode 阶段当前 token 的 query 需要和所有历史 K/V 计算注意力。

#### 在本文件中如何体现

本文件不计算 Attention，但它维护 Attention 所需的 block 映射：

```text
seq.block_table
```

后续 `model_runner.py` 会把它整理为 `block_tables` Tensor，传给 `attention.py`。

---

## 5. 和 vLLM 原版设计的关系

### 5.1 PagedAttention

本文件体现了 PagedAttention 的核心思想：

```text
把 KV Cache 切成固定大小的 block
每条请求维护 block_table
逻辑连续，物理不连续
```

对应代码：

```text
BlockManager.blocks
Sequence.block_table
allocate()
may_append()
deallocate()
```

相比普通连续 KV Cache，这种设计可以减少显存碎片，提高并发能力。

---

### 5.2 KV Cache block 管理

vLLM 的核心设计之一是高效管理 KV Cache block。

本文件实现了简化版：

```text
free_block_ids
used_block_ids
ref_count
block_table
```

它已经具备了基本的 block 分配、释放和共享机制。

---

### 5.3 Prefix Caching

本文件通过链式 hash 实现了 prefix caching：

```text
compute_hash(token_ids, prefix)
hash_to_block_id
hash_blocks()
can_allocate()
```

这和 vLLM 中复用相同前缀 KV Cache 的思想一致。

---

### 5.4 request / sequence 调度

本文件本身不调度请求，但为调度器提供资源判断接口：

```text
can_allocate()
can_append()
```

这使得 scheduler 可以根据 KV Cache 资源做调度决策。

---

### 5.5 Prefill / Decode 分离

本文件的方法天然对应两个阶段：

| 阶段 | 相关方法 |
|---|---|
| Prefill | `can_allocate()`、`allocate()`、`hash_blocks()` |
| Decode | `can_append()`、`may_append()`、`deallocate()` |

这体现了推理引擎中 Prefill 和 Decode 的不同资源需求。

---

### 5.6 nano-vLLM 的简化点

相比原版 vLLM，这个文件做了很多简化：

1. 没有复杂的 block eviction 策略；
2. 没有更完整的 prefix cache 生命周期管理；
3. 没有 CPU/GPU 分层缓存；
4. 没有滑动窗口 attention 的复杂管理；
5. 没有多租户服务中的优先级和公平性策略；
6. hash 映射为简单 `dict[int, int]`，没有处理多个相同 hash 候选 block；
7. block 分配策略非常简单，基本是 FIFO free list。

但它保留了最重要的学习主线：

```text
KV Cache 分块
block_table 映射
引用计数
prefix cache
decode 追加 block
请求结束释放 block
```

---

## 6. 总结：这个文件最应该掌握什么

`block_manager.py` 是理解 nano-vLLM 和 vLLM 的关键文件之一。你需要抓住以下主线：

```text
Sequence 有 token_ids
        ↓
Sequence 按 block_size 切成多个逻辑 block
        ↓
BlockManager 为这些逻辑 block 分配物理 block_id
        ↓
block_id 写入 seq.block_table
        ↓
ModelRunner / Attention 根据 block_table 读写 KV Cache
        ↓
Prefill 完成后 hash_blocks() 登记完整 block
        ↓
后续相同前缀请求可复用这些 block
        ↓
请求结束或被抢占时 deallocate() 释放 block
```

最关键的几个概念：

| 概念 | 在代码中的体现 |
|---|---|
| 物理 KV block | `Block` |
| 全局 block 管理器 | `BlockManager` |
| 空闲 block | `free_block_ids` |
| 正在使用的 block | `used_block_ids` |
| 逻辑到物理映射 | `seq.block_table` |
| prefix cache | `hash_to_block_id` |
| 共享 block | `ref_count` |
| prefill 分配 | `can_allocate()` / `allocate()` |
| decode 追加 | `can_append()` / `may_append()` |
| 完整 block 登记 | `hash_blocks()` |
| 释放资源 | `deallocate()` |

一句话总结：

> `block_manager.py` 是 nano-vLLM 的 KV Cache 内存管理核心，它通过 block table、free list、引用计数和 prefix hash，把每条请求的逻辑 token 序列映射到可复用、可释放、可分页管理的物理 KV Cache block 上。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]
- 关联阅读：[[outputs/项目整理/专题-02-KV缓存与调度|专题-02-KV缓存与调度]]

%% 项目关联导航：结束 %%
