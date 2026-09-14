# nano-vLLM Chunked Prefill 规则完整详解

> 目标：这份笔记专门解释 nano-vLLM 源码里的 **chunked prefill** 规则。重点不是抠每一行代码，而是从整体推理流程、调度规则、状态变化、KV Cache 变化几个角度，把它讲到真正能理解。

---

## 0. 先给结论：chunked prefill 到底是什么？

**chunked prefill** 可以翻译成：**分块预填充**、**分段 prefill**、**把一个很长 prompt 的 prefill 拆成多轮执行**。

在大模型推理中，一个请求通常分为两个阶段：

1. **prefill 阶段**：处理用户输入的 prompt，把 prompt 中所有 token 送进模型，计算并保存 KV Cache。
2. **decode 阶段**：在 KV Cache 的基础上，每轮生成 1 个新 token。

普通 prefill 的做法是：

```text
一个 prompt 有 10000 个 token
↓
一次 prefill 直接把 10000 个 token 全部送进模型
↓
得到第一个输出 token
↓
进入 decode 阶段
```

chunked prefill 的做法是：

```text
一个 prompt 有 10000 个 token
max_num_batched_tokens = 2048
↓
第 1 轮 prefill：处理 token 0 ~ 2047
第 2 轮 prefill：处理 token 2048 ~ 4095
第 3 轮 prefill：处理 token 4096 ~ 6143
...
最后一轮 prefill：处理剩余 token
↓
整个 prompt prefill 完成后，才采样第一个输出 token
↓
进入 decode 阶段
```

所以你可以先记住一句话：

> **chunked prefill 不是把 prompt 拆成多个独立问题，而是把同一个 prompt 的“预处理计算”分成多轮做；前面轮次产生的 KV Cache 会保留下来，后面轮次继续接着算。**

---

## 1. 为什么需要 chunked prefill？

### 1.1 普通 prefill 的问题

假设有两个请求：

```text
请求 A：prompt 长度 20000 token
请求 B：prompt 长度 20 token
```

如果没有 chunked prefill，请求 A 的 prefill 一次性非常长，会占用大量计算资源。请求 B 虽然很短，但可能要等 A 的大 prefill 做完之后才能被处理。

这会导致两个问题：

| 问题 | 解释 |
|---|---|
| 长 prompt 阻塞短 prompt | 一个超长 prompt 会让后面的短请求等待很久 |
| 单轮 token 数过大 | GPU 一次处理太多 token，显存和计算压力都变大 |

### 1.2 chunked prefill 的目的

chunked prefill 的核心目的有两个：

1. **限制每一轮 prefill 的 token 数量**，避免单轮 prefill 太大。
2. **让超长 prompt 可以分多轮处理**，不要一次性吃掉整个调度 step。

在 nano-vLLM 中，限制单轮 token 数量的核心参数是：

```text
max_num_batched_tokens
```

它表示：

> **一次调度 step 里，prefill 阶段最多处理多少个 prompt token。**

注意，这里说的是 **本轮所有被调度请求的 prefill token 总数**，不是单个请求的最大长度。

---

## 2. chunked prefill 出现在 nano-vLLM 的哪个位置？

chunked prefill 的核心逻辑在：

```text
nanovllm/engine/scheduler.py
Scheduler.schedule()
```

`schedule()` 是调度器最核心的方法，它决定每一轮到底执行：

1. prefill；还是
2. decode。

它的大体结构是：

```text
schedule()
├── 先尝试调度 waiting 队列中的 prefill 请求
│   ├── 如果成功调度了至少一个 prefill 请求
│   │   └── 直接返回 scheduled_seqs, True
│   └── 如果没有任何 prefill 请求能调度
│       └── 进入 decode 调度
└── 调度 running 队列中的 decode 请求
    └── 返回 scheduled_seqs, False
```

这里非常关键：

> **nano-vLLM 的这个实现中，只要本轮调度到了 prefill，就不会再调度 decode。**

也就是说，这个项目的调度是比较简化的：

```text
本轮有 prefill 可跑 → 跑 prefill
本轮没有 prefill 可跑 → 跑 decode
```

它不是完整 vLLM 那种更复杂的“decode 和 prefill 混合调度”。

---

## 3. 参与 chunked prefill 的几个核心变量

理解 chunked prefill 前，先把几个变量弄清楚。

---

### 3.1 `waiting`

```text
waiting 队列 = 还没有完成 prefill 的请求队列
```

新来的 prompt 会被包装成 `Sequence`，然后放入 `waiting`。

一个请求只要 prompt 还没有完整 prefill 完成，就仍然属于 waiting 队列。

---

### 3.2 `running`

```text
running 队列 = 已经完成 prefill，正在 decode 的请求队列
```

一个请求只有当整个 prompt 都已经 prefill 完成后，才会从 `waiting` 移到 `running`。

---

### 3.3 `scheduled_seqs`

```text
scheduled_seqs = 本轮被选中执行的请求列表
```

如果本轮是 prefill，那么里面放的是本轮要执行 prefill 的请求。

如果本轮是 decode，那么里面放的是本轮要生成下一个 token 的请求。

---

### 3.4 `num_batched_tokens`

```text
num_batched_tokens = 本轮已经安排了多少个 prefill token
```

比如：

```text
max_num_batched_tokens = 16
已经安排请求 A 的 6 个 token
已经安排请求 B 的 4 个 token
↓
num_batched_tokens = 10
remaining = 16 - 10 = 6
```

---

### 3.5 `remaining`

```text
remaining = max_num_batched_tokens - num_batched_tokens
```

它表示：

> **本轮 prefill 还剩多少 token 预算可以用。**

chunked prefill 的发生，正是因为：

```text
当前请求还需要 prefill 的 token 数 > remaining
```

---

### 3.6 `num_tokens`

在调度 prefill 时，源码中会计算：

```text
num_tokens = 当前 sequence 还需要 prefill 的 token 数
```

但它分两种情况。

#### 情况一：这个请求还没有分配过 KV Cache block

也就是：

```text
seq.block_table 为空
```

说明这是这个请求第一次被调度 prefill。

这时调度器会先问 `BlockManager`：

```text
这个请求是否能分配 KV Cache？
有没有可复用的 prefix cache block？
```

如果有 prefix cache 命中，已经缓存的 prefix 部分就不需要重新算。

所以：

```text
num_tokens = prompt 总 token 数 - 已命中的 prefix cache token 数
```

#### 情况二：这个请求已经被 chunked prefill 处理过一部分

也就是：

```text
seq.block_table 不为空
```

说明它之前已经开始 prefill 了，只是还没做完。

这时：

```text
num_tokens = prompt 总 token 数 - 已经 prefill 过的 token 数
```

也就是：

```text
num_tokens = seq.num_tokens - seq.num_cached_tokens
```

---

### 3.7 `num_scheduled_tokens`

```text
num_scheduled_tokens = 当前 sequence 本轮实际要 prefill 的 token 数
```

它由下面这个规则决定：

```text
num_scheduled_tokens = min(num_tokens, remaining)
```

也就是说：

| 情况 | 结果 |
|---|---|
| 当前请求剩余 token 数 <= 本轮剩余预算 | 本轮把它全部 prefill 完 |
| 当前请求剩余 token 数 > 本轮剩余预算 | 本轮只 prefill 一部分，也就是 chunked prefill |

---

## 4. nano-vLLM 的 chunked prefill 核心规则

源码里最核心的规则可以概括为一句话：

> **每一轮 prefill batch 中，只有本轮第一个被调度的 sequence 允许被 chunk；如果已经调度了别的 sequence，那么后续 sequence 必须完整放入本轮，否则就留到下一轮。**

这句话非常重要，下面详细拆开。

---

## 5. 调度器的 prefill 规则逐步解释

下面用接近源码逻辑的伪代码表示。

```text
scheduled_seqs = []
num_batched_tokens = 0

while waiting 不为空 and scheduled_seqs 数量 < max_num_seqs:

    seq = waiting 队首请求
    remaining = max_num_batched_tokens - num_batched_tokens

    if remaining == 0:
        break

    if seq 第一次被调度:
        num_cached_blocks = block_manager.can_allocate(seq)
        if KV Cache 不够:
            break
        num_tokens = seq 总 token 数 - prefix cache 命中的 token 数
    else:
        num_tokens = seq 总 token 数 - 已经 prefill 过的 token 数

    if remaining < num_tokens and scheduled_seqs 不是空:
        break

    if seq 第一次被调度:
        block_manager.allocate(seq)

    seq.num_scheduled_tokens = min(num_tokens, remaining)
    num_batched_tokens += seq.num_scheduled_tokens

    if seq 这轮之后 prefill 完成:
        seq 从 waiting 移到 running

    scheduled_seqs.append(seq)

if scheduled_seqs 不为空:
    return scheduled_seqs, is_prefill=True
```

这个流程里，最关键的是这条规则：

```text
if remaining < num_tokens and scheduled_seqs 不是空:
    break
```

它的意思是：

```text
如果当前请求放不进本轮剩余 token 预算，
并且本轮前面已经调度过请求，
那么当前请求不能被切分，直接留到下一轮。
```

反过来，如果：

```text
remaining < num_tokens
但是 scheduled_seqs 是空
```

就说明：

```text
当前请求是本轮第一个被调度的请求
```

这时不会 break，而是执行：

```text
seq.num_scheduled_tokens = min(num_tokens, remaining)
```

因为 `remaining < num_tokens`，所以结果就是：

```text
seq.num_scheduled_tokens = remaining
```

这就是 chunked prefill。

---

## 6. “only allow chunked prefill for the first seq” 到底是什么意思？

源码注释中有一句：

```text
only allow chunked prefill for the first seq
```

这句话不要误解成“只有第一个用户请求永远可以 chunk”。

它真正的意思是：

> **在某一轮 schedule() 调度中，只有本轮 prefill batch 的第一个 sequence 可以被切块。**

注意关键词是：

```text
本轮
```

不是整个程序生命周期。

举个例子：

```text
max_num_batched_tokens = 10
waiting = [A, B, C]
```

如果本轮开始时，A 在 waiting 队首，那么 A 是本轮第一个 sequence。

如果 A 很长：

```text
A 还需要 30 token
remaining = 10
```

因为本轮 `scheduled_seqs` 还是空，所以 A 可以被 chunk：

```text
本轮只处理 A 的 10 个 token
```

下一轮开始时，如果 A 还没完成，它仍然在 waiting 队首。那么它又是下一轮的第一个 sequence，所以还可以继续被 chunk。

等 A 完成后，B 成为 waiting 队首。此时 B 在某一轮中也可以成为“本轮第一个 sequence”，所以 B 也可以被 chunk。

所以正确理解是：

```text
不是只有请求 A 可以 chunk
而是每一轮 batch 里最多只能有一个被 chunk 的请求
并且这个被 chunk 的请求必须是本轮第一个被选中的请求
```

---

## 7. 为什么后面的 sequence 不允许 chunk？

假设：

```text
max_num_batched_tokens = 10
waiting = [A, B]
A 剩余 prefill token = 6
B 剩余 prefill token = 20
```

调度过程：

```text
先调度 A，占用 6 个 token
remaining = 4
然后看 B，B 需要 20 个 token
```

这时如果允许 B 也被 chunk，那么本轮会变成：

```text
A 完整 prefill 6 token
B 部分 prefill 4 token
```

但是 nano-vLLM 没这么做。它会让 B 留到下一轮。

原因可以从实现简化角度理解：

1. **保持调度逻辑简单**：一轮中最多只处理一个“未完成 prefill 的长请求”。
2. **减少复杂状态组合**：如果一个 batch 里很多请求都是半截 prefill，postprocess、hash_blocks、prefix cache 状态会更复杂。
3. **保持 FIFO 队列语义清晰**：长请求只有当它排到队首时才会被拆分处理。
4. **避免本轮 batch 过于碎片化**：后续放不下的长请求不硬塞，等下一轮独立处理。

所以 nano-vLLM 的策略不是“尽可能把剩余预算塞满”，而是：

> **在简单、可控的前提下做 chunked prefill。**

---

## 8. 例子一：单个超长 prompt 如何被 chunked prefill

假设参数如下：

```text
max_num_batched_tokens = 10
block_size = 4
请求 A prompt 长度 = 25 token
```

初始状态：

```text
waiting = [A]
running = []
A.num_tokens = 25
A.num_cached_tokens = 0
A.block_table = []
```

---

### 第 1 轮 schedule

```text
remaining = 10
A 还需要 prefill token = 25
scheduled_seqs 为空
```

因为 A 是本轮第一个 sequence，所以允许 chunk：

```text
A.num_scheduled_tokens = min(25, 10) = 10
```

执行模型时，`ModelRunner.prepare_prefill()` 会准备：

```text
input_ids = A 的 token[0:10]
positions = 0, 1, 2, ..., 9
```

模型会计算这 10 个 token 的 hidden states，并把对应 K/V 写入 KV Cache。

postprocess 后：

```text
A.num_cached_tokens = 10
A 还没有完成 prefill，因为 10 < 25
A 继续留在 waiting 队列
不会 append 输出 token
```

注意：

> **第 1 轮不会生成回答 token。因为 prompt 还没完整 prefill 完。**

---

### 第 2 轮 schedule

此时：

```text
A.num_cached_tokens = 10
A.num_tokens = 25
A 还需要 prefill token = 15
remaining = 10
scheduled_seqs 为空
```

A 仍然是本轮第一个 sequence，所以继续 chunk：

```text
A.num_scheduled_tokens = 10
```

这轮模型输入：

```text
input_ids = A 的 token[10:20]
positions = 10, 11, ..., 19
```

模型会接着把 token 10 到 token 19 的 K/V 写入 KV Cache。

postprocess 后：

```text
A.num_cached_tokens = 20
A 仍然没完成 prefill，因为 20 < 25
A 继续留在 waiting
仍然不会 append 输出 token
```

---

### 第 3 轮 schedule

此时：

```text
A.num_cached_tokens = 20
A.num_tokens = 25
A 还需要 prefill token = 5
remaining = 10
```

这次 A 剩余的 5 个 token 可以完整放入本轮：

```text
A.num_scheduled_tokens = 5
```

模型输入：

```text
input_ids = A 的 token[20:25]
positions = 20, 21, 22, 23, 24
```

这轮结束后：

```text
A.num_cached_tokens + A.num_scheduled_tokens = 25
```

说明 A 的 prompt 已经完整 prefill 完成。

调度器会把 A 从：

```text
waiting → running
```

然后 `postprocess()` 会对这轮模型输出的 logits 进行采样，得到第一个回答 token。

也就是：

```text
A.append_token(token_id)
```

此时 A 进入 decode 阶段。

---

### 这个例子的完整时间线

```text
第 1 轮：prefill A[0:10]      不生成回答 token
第 2 轮：prefill A[10:20]     不生成回答 token
第 3 轮：prefill A[20:25]     prefill 完成，生成第 1 个回答 token
第 4 轮：decode A             生成第 2 个回答 token
第 5 轮：decode A             生成第 3 个回答 token
...
```

重点：

> **chunked prefill 的中间轮次只是在补 KV Cache，不会产生最终回答 token。只有最后一段 prefill 完成后，才会采样第一个输出 token。**

---

## 9. 例子二：短 prompt 在前，长 prompt 在后

假设：

```text
max_num_batched_tokens = 10
waiting = [A, B]
A prompt 长度 = 6
B prompt 长度 = 20
```

---

### 第 1 轮 schedule

先看 A：

```text
remaining = 10
A 需要 6 token
A 可以完整放入
```

于是：

```text
scheduled_seqs = [A]
num_batched_tokens = 6
remaining = 4
```

接着看 B：

```text
B 需要 20 token
remaining = 4
scheduled_seqs 已经不是空
```

此时触发规则：

```text
remaining < num_tokens and scheduled_seqs 非空
```

所以 B 不能在本轮被 chunk，调度器直接 break。

第 1 轮只执行：

```text
prefill A 的 6 个 token
```

A 完成 prefill 后进入 running，并生成第一个回答 token。

B 仍然留在 waiting。

---

### 第 2 轮 schedule

此时：

```text
waiting = [B]
running = [A]
```

因为 waiting 不为空，调度器优先尝试 prefill。

B 现在是本轮第一个 sequence：

```text
remaining = 10
B 需要 20 token
scheduled_seqs 为空
```

所以 B 可以被 chunk：

```text
B.num_scheduled_tokens = 10
```

这一轮执行：

```text
prefill B[0:10]
```

注意：

> **虽然 A 已经在 running 里等待 decode，但因为本轮调度到了 B 的 prefill，所以本轮不会 decode A。**

这是 nano-vLLM 简化调度策略的结果。

---

### 这个例子说明什么？

它说明：

```text
后面的长请求如果本轮剩余预算不够，不能被顺手切一小段塞进来。
它必须等到下一轮成为本轮第一个 sequence，才允许 chunk。
```

---

## 10. 例子三：长 prompt 在前，短 prompt 在后

假设：

```text
max_num_batched_tokens = 16
waiting = [A, B]
A prompt 长度 = 20
B prompt 长度 = 3
```

---

### 第 1 轮

A 是本轮第一个 sequence：

```text
A 需要 20 token
remaining = 16
scheduled_seqs 为空
```

A 可以 chunk：

```text
A.num_scheduled_tokens = 16
```

此时：

```text
remaining = 0
```

所以 B 没机会被调度。

第 1 轮：

```text
prefill A[0:16]
```

A 还没完成 prefill，不生成输出 token。

---

### 第 2 轮

A 还剩：

```text
20 - 16 = 4 token
```

第 2 轮开始：

```text
remaining = 16
A 需要 4 token
```

A 可以完整放入：

```text
A.num_scheduled_tokens = 4
remaining = 12
A prefill 完成，移动到 running
```

接着看 B：

```text
B 需要 3 token
remaining = 12
```

B 也可以完整放入本轮。

所以第 2 轮可以调度：

```text
scheduled_seqs = [A, B]
```

这一轮执行：

```text
prefill A[16:20]
prefill B[0:3]
```

A 和 B 都完成 prefill，因此这轮结束后，A 和 B 都可以采样出各自的第一个回答 token。

---

### 这个例子说明什么？

它说明：

> **第一个 sequence 可以是 partial，也可以完整；如果它完整之后还剩 token 预算，后面的 sequence 只要能完整放入，也可以一起 prefill。**

但是如果后面的 sequence 放不完整，就不能 chunk，只能等下一轮。

---

## 11. chunked prefill 和 prefix cache 的关系

nano-vLLM 的 prefill 调度还和 prefix cache 有关系。

prefix cache 的意思是：

> **如果多个请求前缀 token 一样，那么之前已经算过的完整 block 可以复用，不用重新计算。**

比如：

```text
请求 A prompt：系统提示词 + 用户问题 1
请求 B prompt：系统提示词 + 用户问题 2
```

如果“系统提示词”很长，并且已经以完整 block 的形式缓存过，那么 B 的前面部分可以直接复用 A 的 KV Cache block。

---

### 11.1 第一次调度时先检查 prefix cache

当一个 sequence 第一次进入 prefill 调度时：

```text
seq.block_table 为空
```

调度器会调用：

```text
block_manager.can_allocate(seq)
```

它会返回：

```text
num_cached_blocks
```

也就是命中了多少个完整 prefix block。

然后计算：

```text
num_tokens = seq.num_tokens - num_cached_blocks * block_size
```

这表示：

```text
本请求真正需要重新 prefill 的 token 数
= prompt 总长度 - prefix cache 已经命中的 token 数
```

---

### 11.2 prefix cache 命中后，chunked prefill 从未缓存部分开始

假设：

```text
block_size = 4
prompt 总长度 = 20
命中 prefix cache blocks = 3
```

那么已经缓存的 token 数为：

```text
3 * 4 = 12
```

此时：

```text
seq.num_cached_tokens = 12
```

prefill 实际从 token 12 开始：

```text
第 1 个需要计算的 token = prompt[12]
```

如果：

```text
max_num_batched_tokens = 5
```

那么本轮只会 prefill：

```text
prompt[12:17]
```

下一轮继续：

```text
prompt[17:20]
```

所以 chunked prefill 和 prefix cache 是可以同时存在的：

```text
先跳过命中的 prefix cache
再对剩余未计算 token 做 chunked prefill
```

---

### 11.3 prefix cache 只缓存完整 block

从 `BlockManager.can_allocate()` 的逻辑看，它只检查：

```text
seq.num_blocks - 1
```

也就是最后一个 block 不参与 prefix cache 命中检查。

为什么？

因为最后一个 block 可能是不满的。

比如：

```text
block_size = 4
prompt 长度 = 10
blocks:
block0 = token 0,1,2,3
block1 = token 4,5,6,7
block2 = token 8,9        不满 4 个 token
```

prefix cache 通常只安全复用完整 block：

```text
block0、block1 可以缓存
block2 不作为完整 prefix block 命中
```

这样可以避免不完整 block 带来的边界问题。

---

## 12. chunked prefill 和 KV Cache block 分配的关系

这里有一个很容易忽略、但非常重要的点：

> **在 nano-vLLM 中，chunked prefill 只是把“计算”分块，不是把 KV Cache block 的分配也按 chunk 慢慢分配。**

为什么？

因为当一个 sequence 第一次被调度时，如果 `seq.block_table` 为空，调度器会调用：

```text
block_manager.allocate(seq, num_cached_blocks)
```

而 `allocate()` 会为这个 sequence 的所有 block 建好 `block_table`。

也就是说，哪怕这个 prompt 很长，本轮只 prefill 其中一小段，nano-vLLM 也会先给整个 prompt 对应的 KV Cache block 建好映射。

举例：

```text
block_size = 4
prompt 长度 = 25
需要 block 数 = ceil(25 / 4) = 7
```

即使本轮只 prefill 前 10 个 token，第一次调度时也会给这个 sequence 准备 7 个 block 的 `block_table`。

所以在 nano-vLLM 里：

| 项目 | 是否分块 |
|---|---|
| prefill 计算 | 是，按 `max_num_batched_tokens` 分 chunk |
| KV Cache block 映射 | 第一次调度时基本一次性分配 |
| prompt 的逻辑上下文 | 不被切断，仍然是完整连续上下文 |

这点很关键。

如果你以为 chunked prefill 可以让一个超长 prompt 在 KV Cache 不足时也慢慢跑，那在这个实现里是不对的。

因为第一次调度时，`can_allocate()` 仍然要求有足够的 block 给这个 sequence 建立 block table。KV Cache 不够时，它会返回失败，调度器会停止 prefill 调度。

---

## 13. chunked prefill 后，模型输入到底是什么？

调度器只决定：

```text
本轮哪些 seq 要跑
每个 seq 跑多少 token
```

真正把 sequence 变成模型输入的是：

```text
ModelRunner.prepare_prefill()
```

对于每个 sequence，它会计算：

```text
start = seq.num_cached_tokens
seqlen_q = seq.num_scheduled_tokens
end = start + seqlen_q
```

然后取：

```text
input_ids = seq[start:end]
positions = start, start+1, ..., end-1
```

这说明：

> **chunked prefill 的每一段不是从 position 0 重新开始，而是接着原 prompt 的真实位置继续计算。**

比如 prompt 长度 25，第一轮处理 0 到 9，第二轮处理 10 到 19。

第二轮的 position 不是：

```text
0,1,2,3,4,5,6,7,8,9
```

而是：

```text
10,11,12,13,14,15,16,17,18,19
```

这保证了模型仍然知道这些 token 在原始 prompt 中的位置。

---

## 14. 前面 chunk 的上下文怎么保留？

一个常见疑问是：

> 如果第二轮只输入 token 10 到 19，那模型怎么知道 token 0 到 9 的内容？

答案是：

```text
前面 token 的 K/V 已经写入 KV Cache
```

第一轮 prefill token 0 到 9 时，attention 层会把这些 token 的 key/value 写入 KV Cache。

第二轮 prefill token 10 到 19 时，模型可以通过 KV Cache 看到前面 token 0 到 9 的上下文。

所以 chunked prefill 并不是“丢掉前文”，而是：

```text
第 1 段：计算 token 0~9，并写入 KV Cache
第 2 段：计算 token 10~19，同时利用前面 KV Cache
第 3 段：计算 token 20~24，同时利用前面 KV Cache
```

最终效果等价于处理完整 prompt，只是分多轮完成。

---

## 15. postprocess 阶段如何处理 chunked prefill？

模型执行完之后，会进入：

```text
Scheduler.postprocess()
```

对于 prefill，它做几件事：

1. `hash_blocks(seq)`：把已经完整写好的 block 做 hash，供 prefix cache 复用。
2. 更新 `seq.num_cached_tokens`。
3. 判断这个 sequence 的 prefill 是否完成。
4. 如果没完成，直接 continue，不追加输出 token。
5. 如果完成，append 第一个输出 token。

核心逻辑可以理解为：

```text
seq.num_cached_tokens += seq.num_scheduled_tokens
seq.num_scheduled_tokens = 0

if 当前是 prefill 且 seq.num_cached_tokens < seq.num_tokens:
    说明 prompt 还没 prefill 完
    不生成回答 token
    继续留在 waiting
else:
    说明 prefill 完成，或者当前是 decode
    append 新 token
```

这解释了一个关键现象：

> **chunked prefill 的中间 chunk 即使模型也算出了 logits，也不会把采样 token 追加到回答里。只有最后一个 chunk 完成整个 prompt 后，才会真正追加第一个输出 token。**

为什么？

因为大模型生成第一个回答 token 的条件是：

```text
模型已经看完整个 prompt
```

如果 prompt 还只看了一半，就生成回答 token，那么语义上就错了。

---

## 16. chunked prefill 与 decode 的关系

decode 阶段的规则是：

```text
每个 running sequence 每轮通常只调度 1 个 token
```

但是在 nano-vLLM 的 `schedule()` 里，decode 只有在没有任何 prefill 被调度时才会执行。

所以调度优先级是：

```text
waiting 中有可以调度的 prefill → 执行 prefill
没有 prefill 可以调度 → 执行 decode
```

这意味着：

```text
如果 waiting 队列里一直有长 prompt 在做 chunked prefill，running 队列里的 decode 请求可能会被延迟。
```

这是 nano-vLLM 简化实现的一个特点。

完整工业级 vLLM 中，调度会更复杂，通常会考虑 decode 优先、chunked prefill 插空、延迟和吞吐的平衡。但 nano-vLLM 的实现更适合学习：逻辑清晰，容易读懂。

---

## 17. chunked prefill 与 `max_num_batched_tokens` 的关系

`max_num_batched_tokens` 是 chunked prefill 的直接触发条件。

如果：

```text
prompt 剩余 prefill token 数 <= remaining
```

就不需要 chunk，完整处理。

如果：

```text
prompt 剩余 prefill token 数 > remaining
```

是否 chunk 要看它是不是本轮第一个 sequence。

总结成表格：

| 当前 sequence 情况 | 是否允许 chunk | 结果 |
|---|---:|---|
| 本轮第一个 sequence，剩余 token > remaining | 允许 | 本轮处理 remaining 个 token |
| 本轮第一个 sequence，剩余 token <= remaining | 不需要 chunk | 本轮完整处理 |
| 本轮不是第一个 sequence，剩余 token > remaining | 不允许 | 留到下一轮 |
| 本轮不是第一个 sequence，剩余 token <= remaining | 不需要 chunk | 本轮完整处理 |

---

## 18. chunked prefill 与 `max_num_seqs` 的关系

`max_num_seqs` 限制的是：

```text
本轮最多调度多少个 sequence
```

而 `max_num_batched_tokens` 限制的是：

```text
本轮最多调度多少个 prefill token
```

两个限制同时生效。

举例：

```text
max_num_seqs = 4
max_num_batched_tokens = 100
```

如果有很多短 prompt：

```text
A = 10 token
B = 10 token
C = 10 token
D = 10 token
E = 10 token
```

虽然 token 预算还有很多，但是最多只能调度 4 个 sequence：

```text
本轮调度 A,B,C,D
E 留到下一轮
```

如果有一个超长 prompt：

```text
A = 1000 token
```

虽然 sequence 数量只占 1 个，但 token 预算不够，所以 A 会被 chunk。

所以：

```text
max_num_seqs 管请求数量
max_num_batched_tokens 管 token 数量
chunked prefill 主要由 token 数量限制触发
```

---

## 19. chunked prefill 的完整状态变化

一个长 prompt 在 chunked prefill 过程中的状态变化如下：

```text
新请求进入
↓
Sequence.status = WAITING
Sequence.is_prefill = True
Sequence.num_cached_tokens = 0
Sequence.block_table = []
↓
第一次被调度 prefill
↓
分配 block_table
设置 num_scheduled_tokens
执行部分 prompt token
↓
postprocess
更新 num_cached_tokens
如果 prefill 未完成：
    继续 WAITING
    不 append token
如果 prefill 完成：
    status = RUNNING
    从 waiting 移到 running
    append 第一个输出 token
↓
后续 decode
每轮 append 一个 token
↓
遇到 EOS 或达到 max_tokens
↓
status = FINISHED
释放 KV Cache block
```

用更短的流程图表示：

```text
WAITING
  ↓
chunked prefill 第 1 段
  ↓
仍然 WAITING
  ↓
chunked prefill 第 2 段
  ↓
仍然 WAITING
  ↓
最后一段 prefill 完成
  ↓
RUNNING + 生成第一个回答 token
  ↓
decode 逐 token 生成
  ↓
FINISHED
```

---

## 20. 关键细节：为什么 partial prefill 不 append token？

因为大模型的第一个输出 token 应该基于完整 prompt 的最后位置来预测。

假设用户 prompt 是：

```text
请阅读下面这篇文章，然后总结最后一段的观点：
[很长的文章]
```

如果 prompt 被切成三段：

```text
第 1 段：请阅读下面这篇文章...
第 2 段：文章中间内容...
第 3 段：文章最后一段...
```

模型只有看完第 3 段之后，才应该开始回答。

如果第 1 段刚 prefill 完就生成回答 token，那模型还没看到文章后面内容，回答一定是不完整的。

因此 nano-vLLM 的逻辑是：

```text
prefill 没完成 → 只更新 KV Cache，不追加输出 token
prefill 完成 → 才追加第一个输出 token
```

---

## 21. chunked prefill 和 `seq.num_cached_tokens` 的关系

`seq.num_cached_tokens` 是理解 chunked prefill 的核心状态变量。

它表示：

```text
当前 sequence 已经有多少个 prompt token 被处理过，并且对应 KV 已经在 KV Cache 中
```

它有两个来源：

1. prefix cache 命中的 token。
2. chunked prefill 已经执行过的 token。

比如：

```text
prompt 长度 = 100
prefix cache 命中 = 32 token
```

第一次调度前：

```text
seq.num_cached_tokens = 32
```

如果本轮又 chunked prefill 了 16 个 token：

```text
seq.num_cached_tokens = 48
```

下一轮 prefill 就从 token 48 开始。

所以：

> **`num_cached_tokens` 决定下一次 prefill 从 prompt 的哪个位置继续。**

---

## 22. chunked prefill 和 `block_table` 的关系

`block_table` 表示：

```text
这个 sequence 的逻辑 token block 对应到哪些物理 KV Cache block
```

例如：

```text
seq.block_table = [5, 2, 9]
```

表示：

```text
sequence 的第 0 个逻辑 block 存在物理 block 5
sequence 的第 1 个逻辑 block 存在物理 block 2
sequence 的第 2 个逻辑 block 存在物理 block 9
```

chunked prefill 时，`prepare_prefill()` 会根据：

```text
start
end
block_table
block_size
```

计算 `slot_mapping`。

`slot_mapping` 告诉 attention 层：

```text
本轮每一个 input token 的 K/V 应该写入 KV Cache 的哪个物理位置
```

所以：

```text
block_table 决定大方向：属于哪些物理 block
slot_mapping 决定精确位置：写到 block 内哪个 offset
```

---

## 23. chunked prefill 中 `slot_mapping` 怎么理解？

假设：

```text
block_size = 4
seq.block_table = [10, 11, 12]
```

那么逻辑位置和物理位置关系为：

```text
token 0 → block 10 offset 0 → 物理 slot = 10*4 + 0 = 40
token 1 → block 10 offset 1 → 物理 slot = 41
token 2 → block 10 offset 2 → 物理 slot = 42
token 3 → block 10 offset 3 → 物理 slot = 43

token 4 → block 11 offset 0 → 物理 slot = 44
token 5 → block 11 offset 1 → 物理 slot = 45
...
```

如果当前 chunk 是：

```text
start = 5
end = 10
```

那么本轮 token 是：

```text
token 5,6,7,8,9
```

对应 slot 可能是：

```text
token 5 → block 11 offset 1
token 6 → block 11 offset 2
token 7 → block 11 offset 3
token 8 → block 12 offset 0
token 9 → block 12 offset 1
```

`slot_mapping` 就是把这些位置整理成一个数组，让 attention 写 KV Cache 时知道该写哪里。

---

## 24. chunked prefill 和 `cu_seqlens_q / cu_seqlens_k`

在 `prepare_prefill()` 中，会构造两个重要数组：

```text
cu_seqlens_q
cu_seqlens_k
```

它们通常用于变长 FlashAttention。

可以简单理解为：

```text
cu_seqlens_q：本轮实际输入 query token 的累积长度
cu_seqlens_k：当前可见 key/value 上下文的累积长度
```

对于普通 prefill：

```text
q 长度 = k 长度
```

因为本轮输入的 token 就是当前全部上下文。

但是对于 prefix cache 或 chunked prefill：

```text
q 长度可能只是新计算的 chunk
k 长度包含已经缓存的前文 + 当前 chunk
```

比如：

```text
已经缓存 10 token
本轮新算 5 token
```

那么：

```text
seqlen_q = 5
seqlen_k = 15
```

这表示：

```text
当前 5 个新 token 作为 query，
它们可以 attend 到前面 10 个缓存 token + 当前 5 个新 token。
```

这就是 chunked prefill 能保持完整上下文的关键。

---

## 25. chunked prefill 是否减少总计算量？

一般来说：

> **chunked prefill 不减少总计算量，它只是把计算拆成多轮。**

一个 10000 token 的 prompt，不管一次性 prefill，还是分成 5 次 prefill，总体上都要处理这 10000 个 token。

它的主要作用不是减少总 FLOPs，而是：

1. 限制每轮 token 数，避免单轮过大。
2. 改善调度粒度。
3. 让超长 prompt 不会一次性占满整个 step。
4. 更好地控制吞吐和延迟。

但如果结合 prefix cache，则可以减少重复计算。

也就是说：

```text
chunked prefill：拆分计算，不天然减少总计算
prefix cache：复用已有 KV，能减少重复计算
```

两者经常一起出现，但作用不同。

---

## 26. chunked prefill 是否降低显存占用？

要分情况。

从理论上说，chunked prefill 可以降低某一轮计算中的临时激活开销，因为一次处理的 token 少了。

但是在 nano-vLLM 这个实现里，要特别注意：

> **它并不会按 chunk 慢慢分配 prompt 的 KV Cache block，而是在第一次调度时就为 sequence 建好 block_table。**

所以对于 KV Cache 长期占用来说：

```text
chunked prefill 不会显著减少这个请求最终需要的 KV Cache 总量
```

它主要减少的是：

```text
单轮 prefill 的计算规模和临时压力
```

而不是把整个请求的 KV Cache 需求变小。

---

## 27. chunked prefill 规则完整总结

nano-vLLM 中 chunked prefill 可以总结为以下 10 条规则：

### 规则 1：调度器每轮先看 waiting 队列

如果 waiting 队列里有请求可以 prefill，调度器优先做 prefill。

---

### 规则 2：prefill 受 `max_num_batched_tokens` 限制

本轮所有 prefill 请求的 token 总数不能超过：

```text
max_num_batched_tokens
```

---

### 规则 3：prefill 也受 `max_num_seqs` 限制

本轮最多调度：

```text
max_num_seqs
```

个 sequence。

---

### 规则 4：如果 sequence 是第一次调度，要先检查 KV Cache block

如果 KV Cache block 不够，调度器无法启动这个 sequence 的 prefill。

---

### 规则 5：第一次调度时会检查 prefix cache

如果有完整 prefix block 命中，则这些 token 不需要重新 prefill。

---

### 规则 6：只有本轮第一个 sequence 可以被 chunk

如果当前 sequence 放不进剩余 token 预算，并且它是本轮第一个 sequence，那么允许：

```text
只处理 remaining 个 token
```

---

### 规则 7：后续 sequence 不允许被 chunk

如果本轮已经调度了至少一个 sequence，而下一个 sequence 放不进剩余预算，那么直接停止本轮 prefill 调度，把它留到下一轮。

---

### 规则 8：partial prefill 不会生成回答 token

如果一个 sequence 的 prompt 还没有完整 prefill 完，postprocess 不会 append 采样 token。

---

### 规则 9：最后一个 prefill chunk 完成后，才生成第一个回答 token

当：

```text
seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens
```

说明整个 prompt 处理完了。

此时 sequence 从 waiting 进入 running，并 append 第一个输出 token。

---

### 规则 10：有 prefill 被调度时，本轮不会 decode

只要 `scheduled_seqs` 里有 prefill 请求，`schedule()` 就直接返回，不进入 decode 逻辑。

---

## 28. 最容易混淆的几个问题

### 问题 1：chunked prefill 是不是把 prompt 拆成多个独立 prompt？

不是。

它只是把同一个 prompt 的 prefill 计算拆成多轮。前面的 chunk 会写入 KV Cache，后面的 chunk 仍然能看到前文。

---

### 问题 2：每个 chunk 都会生成一个 token 吗？

不是。

中间 chunk 只补 KV Cache，不生成最终回答 token。

只有最后一个 chunk 把完整 prompt 处理完后，才生成第一个回答 token。

---

### 问题 3：chunked prefill 会不会改变模型回答？

理想情况下不会。

因为 positions 是连续的，前文 KV Cache 也保留，所以模型看到的上下文逻辑上仍然是完整 prompt。

---

### 问题 4：为什么短 prompt 后面的长 prompt 不能顺手切一段？

因为 nano-vLLM 的规则是：

```text
一轮 batch 中只有第一个 sequence 可以 chunk
```

如果前面已经调度了短 prompt，那么后面的长 prompt 就不是本轮第一个 sequence，不能 chunk，只能等下一轮。

---

### 问题 5：chunked prefill 会不会减少 KV Cache 占用？

在 nano-vLLM 这个实现里，对长期 KV Cache 总占用帮助不大，因为第一次调度时就会为 sequence 建好 block table。

它主要控制的是每轮 prefill 的计算 token 数。

---

### 问题 6：如果 KV Cache 不够，会不会先 chunk 一点跑起来？

在 nano-vLLM 这个实现里，不会。

第一次调度时 `can_allocate()` 如果发现 block 不够，会返回失败，调度器直接停止 prefill 调度。

所以这里的 chunked prefill 不是“内存不够就先跑一小段”，而是“token 预算不够就先算一小段”。

---

## 29. 用一句完整的话描述 nano-vLLM 的 chunked prefill

可以这样描述：

> 在 nano-vLLM 中，chunked prefill 是 Scheduler 在 prefill 调度阶段基于 `max_num_batched_tokens` 做的 token 级切分策略；当 waiting 队首 sequence 剩余 prompt token 数超过本轮 token 预算时，如果它是本轮第一个被调度的 sequence，就只调度其中 `remaining` 个 token，本轮执行后更新 KV Cache 和 `num_cached_tokens`，但在整个 prompt prefill 完成之前不追加输出 token；当最后一个 chunk 完成后，该 sequence 才进入 running 队列并采样第一个回答 token。后续 sequence 如果放不进剩余预算则不会被 chunk，而是留到下一轮。

---

## 30. 你读源码时应该抓住的主线

读 `scheduler.py` 时，不要只盯着代码。你应该脑子里一直有这条主线：

```text
waiting 里的请求还没完成 prompt 预处理
↓
每一轮 prefill 有 token 预算 max_num_batched_tokens
↓
队首请求如果太长，可以先处理一段
↓
处理过的 token 数记录在 num_cached_tokens
↓
KV 写入 block_table 指向的 KV Cache 位置
↓
prompt 没处理完就继续 waiting
↓
prompt 处理完才进入 running
↓
进入 running 后才开始 decode 逐 token 生成
```

如果你能把这条线讲清楚，chunked prefill 就已经理解了。

---

## 31. 最后给一个极简记忆版

```text
chunked prefill = 长 prompt 的 prefill 分多轮算

触发条件：
    当前 seq 剩余 prefill token > 本轮 remaining token 预算

允许条件：
    当前 seq 必须是本轮第一个被调度的 seq

本轮处理：
    num_scheduled_tokens = remaining

中间 chunk：
    只写 KV Cache
    更新 num_cached_tokens
    不生成回答 token
    继续留在 waiting

最后 chunk：
    prompt prefill 完成
    seq 从 waiting 进入 running
    采样第一个输出 token

后续：
    decode 阶段每轮生成 1 个 token
```

---

## 32. 和整个推理流程放在一起理解

最后把 chunked prefill 放回完整推理流程里：

```text
用户 prompt
↓
tokenizer.encode 得到 prompt token ids
↓
创建 Sequence，进入 waiting 队列
↓
Scheduler.schedule()
↓
如果 prompt 很短：
    一轮 prefill 完成
    生成第一个输出 token
    进入 running

如果 prompt 很长：
    第 1 轮 chunked prefill，写入部分 KV Cache，不输出
    第 2 轮 chunked prefill，继续写入 KV Cache，不输出
    ...
    最后一轮 chunked prefill，prompt 完整处理完，输出第一个 token
    进入 running
↓
decode 阶段
↓
每轮根据已有 KV Cache 和最后一个 token 预测下一个 token
↓
不断 append token
↓
遇到 EOS 或 max_tokens
↓
释放 KV Cache，返回完整回答
```

这就是 nano-vLLM 中 chunked prefill 的完整规则和运行过程。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]

%% 项目关联导航：结束 %%
