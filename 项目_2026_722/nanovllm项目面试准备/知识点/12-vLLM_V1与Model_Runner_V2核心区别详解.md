# vLLM V1 与“V2”核心差异详解：从 Engine V1 到 Model Runner V2

> 更新时间：2026-08-11  
> 面向对象：希望理解 vLLM 推理框架架构演进、准备 AI Infra / 推理优化面试的初学者。
>
> **先给出最重要的术语纠正：**
>
> 当前 vLLM 官方正式使用的核心引擎架构仍称为 **V1 Engine**。官方所说的 **V2**，主要是指 **Model Runner V2（MRV2）**，也就是 V1 Engine 内部“GPU 模型执行器”的第二代实现，而不是一套完全独立的“vLLM V2 Engine”。
>
> 因此，严格来说不应该说：
>
> ```text
> vLLM V1 Engine  →  vLLM V2 Engine
> ```
>
> 更准确的是：
>
> ```text
> vLLM V1 Engine
> ├── Scheduler
> ├── KV Cache Manager
> ├── Engine Core
> └── GPU Model Runner
>       ├── Model Runner V1
>       └── Model Runner V2（MRV2，新执行路径）
> ```
>
> MRV2 的目标不是重新推翻整个 V1 Engine，而是重点重构 **模型执行热路径**，进一步降低 CPU 开销、强化异步执行、优化输入准备/采样/CUDA Graph，并降低 V1 Model Runner 随功能扩展产生的复杂度。

---

## 1. 先理解 vLLM 为什么要从旧架构走向 V1

vLLM 最早的核心目标是：

```text
让大模型推理：
更高吞吐
更省 KV Cache
支持动态请求
支持多 GPU
```

早期版本已经有：

- PagedAttention；
- Continuous Batching；
- Tensor Parallel；
- Prefix Cache；
- CUDA Graph；
- 各类量化和采样功能。

但是随着功能越来越多：

```text
Chunked Prefill
Speculative Decode
Multi-modal
Prefix Cache
LoRA
Structured Output
复杂采样
Hybrid Attention
KV Offloading
```

原来的核心结构越来越难扩展。

因此 V1 做了一次较大的核心重构。

V1 并不是把模型 Kernel 全部重新写一遍，而是保留许多已经成熟的：

```text
模型实现
GPU Kernel
工具代码
```

重点重新设计：

```text
Scheduler
KV Cache Manager
Worker
Sampler
API / Engine Core
```

可以把它理解成：

> V1 首先解决的是“整个推理引擎怎样更统一、更容易扩展”。

而 MRV2 继续解决：

> “V1 Engine 已经统一以后，GPU 每一步 ModelRunner 还能不能进一步减少 CPU 和状态管理开销？”

---

# 2. 当前 V1 Engine 的整体架构

当前官方架构可以粗略画成：

```text
客户端
  ↓
API Server
  ↓
Engine Core
  ├── Scheduler
  ├── KV Cache Manager / Coordinator
  └── Request State
  ↓
Executor
  ↓
GPU Worker
  ↓
Model Runner
  ↓
模型 Forward / Sampling
```

多卡 Tensor Parallel 时：

```text
1 个 Engine Core
      ↓
多个 GPU Worker
      ↓
每张 GPU 一个 Worker 进程
```

例如：

```text
TP = 4

API Server
    ↓
Engine Core
    ↓
┌────────┬────────┬────────┬────────┐
GPU0     GPU1     GPU2     GPU3
Worker0  Worker1  Worker2  Worker3
```

其中：

### Engine Core

负责：

```text
请求生命周期
调度
KV Cache 分配
Preemption
模型执行协调
```

### GPU Worker / Model Runner

负责：

```text
准备这一轮模型输入
组织 Attention Metadata
执行模型 Forward
执行 CUDA Graph
计算 Logits
采样 Token
```

所以理解 MRV2 时必须先记住：

> **MRV2 改的是 Model Runner 这一层，不是把 V1 Scheduler、Engine Core、KV Cache Manager 全部重新换成另一套 V2。**

---

# 3. V1 Scheduler 的重要设计：统一 Token 调度

V1 的一个核心思想是：

> Scheduler 不再把 Prefill 和 Decode 看成两种完全割裂的任务，而是统一成“这个请求本轮要计算多少 Token”。

概念上 Scheduler 输出类似：

```python
{
    "request_A": 1,
    "request_B": 128,
    "request_C": 1,
}
```

其中：

```text
A：Decode 1 Token
B：Chunked Prefill 128 Token
C：Decode 1 Token
```

这样可以用统一的：

```text
Token Budget
```

调度不同请求。

因此 V1 很自然地支持：

- Continuous Batching；
- Chunked Prefill；
- Prefix Caching；
- Speculative Decoding；
- 动态 Token Budget。

这是 **Engine V1 层面的设计**。

MRV2 并没有推翻这个 Scheduler 思想。

---

# 4. 为什么有了 V1，还需要 Model Runner V2

V1 Engine 已经比老架构统一很多，但官方在实际开发中发现 Model Runner 仍然存在几个问题：

```text
① Persistent Batch 状态和实际模型输入耦合太紧
② Async Scheduling 是后补进去的，不够自然
③ CPU↔GPU 同步点仍然较多
④ Block Table 等大元数据更新复杂
⑤ 输入准备仍有较多 Python/CPU 开销
⑥ Sampling 路径内存与数值控制仍有优化空间
⑦ gpu_model_runner.py 功能越来越集中
⑧ dummy_run 承担职责太多
⑨ CUDA Graph 管理过于隐式
```

于是官方从更底层重新设计 Model Runner，形成：

```text
Model Runner V2
简称 MRV2
```

可以概括为：

> **V1 是“引擎核心架构重构”，MRV2 是“V1 Engine 内 GPU 执行热路径的二次重构”。**

---

# 5. 最大区别一：Persistent Batch 的状态组织方式改变

这是 MRV2 最核心的变化之一。

## 5.1 为什么需要 Persistent Batch

Continuous Batching 下，相邻两轮 Batch 通常变化很小。

例如：

```text
Step N:
[A, B, C, D, E]

Step N+1:
[A, B, C, D, F]
```

只有：

```text
E 完成
F 加入
```

如果每一步都重新从 Python List 构造完整 GPU Tensor：

```text
block_tables
temperature
seq_lens
sampling metadata
...
```

会浪费很多 CPU 时间。

所以 V1 引入 Persistent Batch：

> 保留上一轮 Batch 状态，只增量修改变化的部分。

---

# 6. V1 Persistent Batch 的问题

V1 的做法是：

```text
Persistent State Tensor
≈
实际 Model / Sampler Input Tensor
```

也就是长期状态和当前模型输入绑定得很紧。

假设：

```text
row0 → A
row1 → B
row2 → C
row3 → D
```

B 完成后，为保持 Batch 连续，可能要把其他请求挪到空出来的位置。

于是很多关联 Tensor 都需要同步重新排列：

```text
block_table row
temperature row
sampling params row
request state row
logits processor state
...
```

这会产生复杂的：

```text
add
remove
move
reorder
```

逻辑。

另外 V1 还需要维护：

```text
CachedRequestState
```

来防止 Persistent Tensor 某些行被覆盖后请求状态丢失。

所以 V1 虽然已经避免“每一步全量重建”，但代码复杂度比较高。

---

# 7. MRV2 的解决方案：持久状态与每步输入彻底解耦

MRV2 的设计是：

```text
Persistent Request State
          ↓
      Gather
          ↓
Per-Step Model Input
```

不再让二者是同一个东西。

MRV2 会提前建立固定大小的状态表，例如：

```text
max_num_reqs = 1024

Persistent State:

row0
row1
row2
...
row1023
```

一个请求在其活跃生命周期里被分配一个固定 Row。

例如：

```text
A → row 37
B → row 5
C → row 901
```

这个 Row 不需要等于：

```text
当前 Model Batch 第几行
```

模型这一轮真正需要：

```text
[C, A, B]
```

则 GPU 根据索引 Gather：

```text
row901
row37
row5
```

形成当前 Step 输入。

核心变化：

```text
V1：
Persistent State 本身就是 Batch Input

MRV2：
Persistent State 是数据库
当前 Batch Input 是从数据库 Gather 出来的视图
```

---

# 8. 为什么这种解耦非常重要

假设 B 完成。

V1 可能需要：

```text
删除 B
↓
移动其他 Row
↓
修改所有相关 Batch Tensor
```

MRV2：

```text
释放 B 的固定 Row
```

即可。

下一轮当前执行顺序由：

```text
gather index
```

决定。

这样可以减少：

- Tensor-wide Reorder；
- Request State 备份；
- Persistent Batch 的复杂 bookkeeping。

同时更适合异步执行。

---

# 9. Preemption 在 MRV2 中也更简单

MRV2 将：

```text
Preemption
```

在 Model Runner 状态层面近似看成：

```text
当前请求结束
```

释放其固定 State Row。

之后请求 Resume：

```text
重新作为一个新 Active State 加入
```

因此不必让一个已经离开执行 Batch 的 Request 永久占住旧 Row。

---

# 10. 最大区别二：MRV2 是 Async-First

现代 vLLM 希望做到：

```text
GPU 正在执行 Step N
        同时
CPU/Scheduler 准备 Step N+1
```

形成：

```text
CPU:
Prepare N+1     Prepare N+2
     ↓               ↓

GPU:
Execute N   Execute N+1   Execute N+2
```

如果 CPU 和 GPU 能重叠：

```text
总执行时间
<
CPU准备 + GPU执行简单相加
```

---

# 11. V1 的异步执行为什么比较别扭

V1 最初并不是完全以 Async Scheduling 为第一设计原则。

后来为了异步化，需要防止：

```text
CPU 正在修改一个 Pinned Buffer
```

而 GPU 同时：

```text
还在异步读取这个 Buffer
```

产生 Race Condition。

因此 V1 需要：

```text
Async Barrier
```

保护某些临界区。

问题是：

```text
Barrier容易漏
同步逻辑复杂
限制CPU/GPU重叠
后续功能越多越难维护
```

---

# 12. MRV2：从设计上消除 Async Barrier

MRV2 的思路不是继续增加 Barrier，而是让 CPU 和 GPU 尽量不要同时访问同一份可写 Buffer。

例如：

```text
CPU Persistent State
↓
复制到临时 Pinned Buffer
↓
GPU异步读取临时Buffer
```

与此同时 CPU 可以继续修改原 Persistent State。

于是：

```text
CPU写A
GPU读B
```

没有竞争。

这使 MRV2 更接近真正的 CPU/GPU 流水执行。

---

# 13. 最大区别三：StagedWriteTensor

Block Table 是推理系统中典型的大元数据 Tensor。

Continuous Batching 中每一步通常只有少量变化：

```text
某请求增加1个Block
某请求完成
某请求被Preempt
```

如果每一步都把完整 Block Table：

```text
CPU构造
↓
完整传GPU
```

浪费很大。

---

# 14. MRV2 的增量更新方式

MRV2 引入：

```text
StagedWriteTensor
```

思想是：

```text
完整Base Tensor长期留在GPU

CPU只记录diff：
“row25的第8个位置改成123”
“row91新增[7,8,9]”

↓
把diff打包
↓
一次传GPU
↓
一个Kernel应用diff
```

即：

```text
全量复制
→
增量更新
```

特别适合：

```text
block_tables
num_computed_tokens
其他大型Persistent Metadata
```

---

# 15. KV Cache 管理：不要误解成 V2 换了一套 PagedAttention

不能简单说：

```text
V1 用一种 KV Cache
V2 换成另一种 KV Cache
```

因为 MRV2 不是新的 Engine Core。

当前 KV Cache 的：

```text
Block 分配
Prefix Cache
KV Cache Groups
Hybrid KV Cache
Request Free / Preemption
```

主要仍由 V1 Engine Core 中的：

```text
KVCacheManager
KVCacheCoordinator
HybridKVCacheCoordinator
```

负责。

---

# 16. MRV2 真正改变的是“KV 元数据如何送进模型执行”

例如：

```text
block_tables
num_computed_tokens
attention metadata
```

V1 Model Runner：

```text
Persistent Batch 内直接维护和重排
```

MRV2：

```text
Persistent State与Step Input解耦
+
StagedWriteTensor增量更新
+
GPU Gather
+
GPU Native Metadata Preparation
```

因此更准确地说：

> **KV Cache 的资源管理仍然属于 V1 Engine Core；MRV2 优化的是 GPU 执行侧如何高效维护和消费这些 KV Cache 元数据。**

---

# 17. Hybrid KV Cache 为什么越来越重要

现在模型不一定每层都是同一种 Attention。

例如：

```text
Full Attention
Sliding Window
Mamba / State Space
Hybrid Attention
```

当前 V1 Core 已有：

```text
HybridKVCacheCoordinator
```

支持多个 KV Cache Group。

MRV2 也逐步加入：

```text
Qwen3.5
Mamba Hybrid
```

等模型支持。

所以现代 vLLM 的趋势是：

```text
Engine Core：
管理异构Cache资源

MRV2：
高效准备异构模型执行所需Metadata
```

---

# 18. 最大区别四：更多输入元数据在 GPU 上直接生成

模型每一步需要：

```text
input_ids
positions
query_start_loc
seq_lens
slot mapping
attention metadata
sampling mapping
```

V1 中很多元数据准备工作依赖：

```text
Python
CPU Tensor
CPU→GPU Copy
```

这在 Decode 小 Batch 或高并发时可能成为 CPU Bottleneck。

---

# 19. MRV2 使用 Triton 做 GPU-Native Input Preparation

MRV2 更倾向让 GPU Kernel 直接生成：

```text
input_ids
positions
query_start_loc
seq_lens
```

好处：

### ① 减少 Python/CPU 开销

不必在 Python 中循环拼 Tensor。

### ② 更适合异步

Speculative Decode 中，一些状态本来就是 GPU 刚计算出来的。

如果：

```text
GPU结果
↓
同步回CPU
↓
CPU算Metadata
↓
再传GPU
```

会制造同步屏障。

GPU-native 则可以：

```text
GPU结果
↓
GPU直接生成下一步Metadata
```

---

# 20. UVA：进一步减少某些大数据复制

MRV2 某些路径使用：

```text
Universal Virtual Addressing
```

例如较大的 Prompt Token 数据可以保存在 CPU，在合适路径中由 GPU Kernel 直接访问。

目标是减少某些：

```text
大而低频数据
```

不必要的重复 GPU 拷贝。

---

# 21. 最大区别五：Sampler 被大幅重写

模型 Forward 后还有：

```text
hidden state
↓
LM Head
↓
logits
↓
temperature
top-k
top-p
penalty
random sampling
logprobs
```

复杂请求下 Sampler 本身也会形成明显开销。

MRV2 将大量 Sampling 路径改成 Triton Kernel。

---

# 22. Gumbel Sampling

MRV2 引入 Triton Gumbel Sampling Kernel，可以避免显式物化完整 Softmax Probability Tensor。

收益包括：

```text
减少中间Tensor
减少显存流量
增强随机数与数值控制
```

---

# 23. Top-K Logprobs 更省显存

V1 的典型思路：

```text
完整Vocabulary logits
↓
完整Vocabulary logprobs
↓
Top-K
```

MRV2：

```text
先从logits定位Top-K Token
↓
只处理必要Logprob
```

在大词表和大 Batch 下可降低峰值显存。

---

# 24. Speculative Decoding 在 MRV2 中更自然

投机解码中：

```text
一个Request
```

可能在同一步产生：

```text
多个Logits Row
```

MRV2 使用：

```text
idx_mapping
```

在 Kernel 内把：

```text
logits row
→
request sampling state
```

映射起来。

这样比把每请求 Sampling State 复制成多个 Row 更简单，也更适合复杂投机解码。

---

# 25. 最大区别六：CUDA Graph 管理更显式

V1 已经支持 CUDA Graph，但官方认为它的 Graph 管理比较隐式。

MRV2 引入：

```text
CUDAGraphManager
```

显式负责：

```text
Capture
Graph生命周期
执行模式选择
Replay
```

这样更容易支持：

```text
普通Decode
Spec Decode
多步Draft
不同Graph模式
```

---

# 26. 多步 Draft 可以一起捕获

MRV2 可以把多个连续 Draft Forward 整体捕获进一张 CUDA Graph，而不是每个 Draft Token 都单独进行一次 Graph Replay。

概念上：

```text
旧：
Draft1 → replay
Draft2 → replay
Draft3 → replay

MRV2：
[Draft1 → Draft2 → Draft3]
           ↓
一次更大的Graph
```

目标是减少：

```text
Python循环
Graph Launch次数
Metadata重复构造
```

---

# 27. 最大区别七：`dummy_run` 职责拆分

V1 的 `dummy_run` 曾承担：

```text
Memory Profiling
torch.compile
CUDA Graph Capture
Warmup
DP Empty Forward
```

MRV2 改为：

```text
execute_model：
支持dummy执行

dummy_run：
复用execute_model做profiling/warmup

CUDA Graph：
独立Capture路径
```

这样减少：

```text
dummy路径
和
真实执行路径
```

之间行为不一致的问题。

---

# 28. 最大区别八：代码更模块化

V1 的 `gpu_model_runner.py` 随着功能增长越来越集中。

MRV2 更倾向把逻辑拆成：

```text
InputBatch
mrope_utils
penalties
sampling
metadata
CUDA Graph manager
...
```

目的：

```text
Feature低耦合
更容易维护
更容易扩展
```

---

# 29. V1 与 MRV2 的完整对比表

| 维度 | Model Runner V1 | Model Runner V2 |
|---|---|---|
| 所属引擎 | V1 Engine | **仍属于 V1 Engine** |
| 核心目标 | V1 初代 GPU 执行路径 | 更低 CPU 开销、更异步、更模块化 |
| Persistent Batch | State 与 Model Input 紧耦合 | Persistent State 与 Step Input 解耦 |
| 请求 Row | Batch 重排可能造成 Row 搬移 | 活跃期拥有固定 State Row |
| 每步输入 | 直接使用 Persistent Batch | 按执行顺序 Gather |
| CachedRequestState | 需要额外备份 | 设计上可去除冗余备份 |
| 异步设计 | 后续补充 Async 支持 | Async-First |
| CPU/GPU Race | Async Barrier | 通过 Buffer 所有权/副本降低 Race |
| Block Table | Persistent Batch 中维护 | `StagedWriteTensor` 增量更新 |
| Input Metadata | 较多 CPU/Python 构造 | 更多 Triton GPU-Native |
| Sampler | V1 Sampler | Triton-Native Sampler |
| Top-K Logprobs | 可先物化完整 logprobs | 先找 Top-K，再处理必要值 |
| Spec Decode | 状态映射较复杂 | `idx_mapping` 更适合多 logits |
| CUDA Graph | 相对隐式 | `CUDAGraphManager` 显式管理 |
| Multi-step Draft | 多步单独执行较多 | 可融合多步 Draft Graph |
| dummy_run | 职责较重 | 功能拆分 |
| 模块化 | ModelRunner 较集中 | 更细粒度模块化 |
| Hybrid Model | V1 路径已有能力 | MRV2 持续扩展 Qwen3.5/Mamba 等 |

---

# 30. 调度机制到底有没有“V1 Scheduler → V2 Scheduler”

严格来说没有这种简单替换。

当前官方 Scheduler 仍在：

```text
vllm/v1/core/sched/
```

MRV2 主要负责消费：

```text
SchedulerOutput
```

然后高效准备 GPU 模型输入。

所以如果面试官问：

> V1 和 V2 的 Scheduler 有什么区别？

更准确的回答应是：

> 如果这里的 V2 指官方 Model Runner V2，那么它不是新的 V2 Scheduler。Scheduler 仍属于 V1 Engine Core，MRV2 主要重构 SchedulerOutput 到 GPU Model Forward 之间的数据准备、状态维护和执行路径。

---

# 31. KV Cache 到底有没有“V1 KV Manager → V2 KV Manager”

同样不能这么简单描述。

当前 KV 资源管理主要仍然位于：

```text
vllm/v1/core/
```

包括：

```text
KVCacheManager
KVCacheCoordinator
HybridKVCacheCoordinator
```

MRV2 重点优化：

```text
Block Table怎么维护
num_computed_tokens怎么更新
Attention Metadata怎么构造
GPU如何消费这些信息
```

因此：

```text
KV资源调度：
V1 Engine Core

KV执行侧Metadata：
MRV2重点优化
```

---

# 32. 用一次请求理解 V1 Engine + MRV2

请求进入：

```text
“介绍一下 vLLM”
```

### ① API Server

```text
文本 → Token IDs
```

### ② Engine Core

创建 Request State。

### ③ Scheduler

决定：

```text
本轮计算多少Token
```

### ④ KV Cache Manager

决定：

```text
Prefix命中
KV Block分配
```

### ⑤ SchedulerOutput

告诉 Worker：

```text
哪些请求运行
每请求多少Token
Block信息
```

### ⑥ MRV2 更新 Persistent State

新请求获得固定 State Row。

只增量修改：

```text
Block Table
Sampling State
Token State
```

### ⑦ GPU 准备 Step Input

根据当前执行顺序 Gather 状态，并用 GPU Kernel 生成：

```text
positions
seq_lens
query_start_loc
...
```

### ⑧ Model Forward

执行模型。

### ⑨ Triton Sampler

生成 Token。

### ⑩ CPU/GPU 并行准备下一步

GPU 执行当前 Step 时，CPU 可以继续准备下一步。

---

# 33. 性能层面 MRV2 真正优化什么

MRV2 不是主要减少：

```text
Transformer FLOPs
```

而是在减少模型外围的“系统税”：

```text
Python开销
Persistent Batch重排
CPU输入构造
CPU→GPU元数据复制
CPU/GPU同步
Sampler中间Tensor
CUDA Graph Launch
Spec Decode控制开销
```

尤其在：

```text
小Batch Decode
高并发Serving
Speculative Decode
短Kernel密集场景
```

更重要。

---

# 34. 为什么模型越快，系统开销反而越明显

例如：

### 原来

```text
GPU计算 = 20 ms
CPU准备 = 1 ms
```

CPU 只占约 5%。

### Kernel 优化后

```text
GPU计算 = 4 ms
CPU准备 = 1 ms
```

CPU 已经占约 20%。

所以模型 Kernel 越快：

> CPU、Metadata、Sampling、Launch 等系统开销越容易成为下一阶段瓶颈。

MRV2 正是在进一步消除这些开销。

---

# 35. 当前 MRV2 的发展状态

截至 2026 年当前官方发布说明：

- MRV2 曾先在 Qwen3 Dense 上成为默认路径；
- 随后扩展到 Llama、Mistral 等 Dense 模型；
- 后续发布中已成为所有 Dense 模型的默认执行路径；
- Qwen3.5/Mamba Hybrid、量化模型、更多 Spec Decode 和非生成任务也在继续迁移到 MRV2。

但不同：

```text
模型
硬件Backend
特殊功能
```

仍可能具有不同支持程度，框架在某些情况下仍可能使用其他路径或回退。

---

# 36. 与 nano-vLLM 项目怎样建立联系

nano-vLLM 已经实现了：

```text
Scheduler
Continuous Batching
Paged KV Cache
BlockManager
Chunked Prefill
CUDA Graph
GDN State
KV Cache Compression
MTP
```

这些能帮助理解：

```text
推理系统“有什么模块”
```

而正式 vLLM MRV2 进一步关注：

```text
这些模块每一步怎样少做CPU工作
怎样异步
怎样减少同步
怎样让状态增量更新
怎样把更多Metadata/Sampling搬到GPU
```

所以可以理解成：

> nano-vLLM 更适合学习推理引擎基本骨架，MRV2 展示了生产级系统如何进一步优化控制面和数据面开销。

---

# 37. 面试时最重要的术语纠正

如果面试官问：

> “vLLM V1 和 V2 有什么区别？”

建议先回答：

> 严格来说，当前官方的核心 Engine 还是叫 V1，所谓 V2 主要是 Model Runner V2，不是一整套独立的 V2 Engine。V1 是对 Scheduler、KV Cache Manager、Worker、Sampler、API 等整个核心系统的重构；MRV2 则是在 V1 Engine 内进一步重构 GPU Model Runner，主要解决 Persistent Batch 状态耦合、异步调度、CPU 输入准备、Sampling 和 CUDA Graph 管理等热路径问题。

再展开细节。

---

# 38. 一段完整面试回答

> vLLM 官方现在更准确的说法不是 V1 Engine 和 V2 Engine，而是 V1 Engine 里面进一步引入了 Model Runner V2。V1 是一次比较大的核心架构升级，它把 Scheduler、KV Cache Manager、Worker、Sampler 和 API Server 等重新组织，统一用 Token Budget 调度 Prefill、Decode、Chunked Prefill 和 Spec Decode。后来的 MRV2 主要针对 GPU 执行热路径继续优化。V1 Model Runner 已经用了 Persistent Batch，避免每一步从头构造 Block Table 和采样参数，但它把持久请求状态和实际模型输入耦合得比较紧，请求增删时需要复杂的 Row 重排，还要维护 CachedRequestState。MRV2 则给每个活跃请求一个固定状态 Row，然后每一步根据真正的执行顺序从这些 Persistent State 中 Gather 当前输入，这样请求状态和 Batch 顺序解耦。第二个大的变化是 MRV2 从一开始就按异步执行设计，让 CPU 准备下一步时 GPU 可以同时执行当前步，并通过 StagedWriteTensor 只增量更新 Block Table 等大元数据，同时使用 Triton 在 GPU 上准备 positions、seq_lens 等输入。Sampler 也更多改成 Triton Kernel，并优化 Top-K Logprobs 和投机解码。CUDA Graph 方面 V1 的管理比较隐式，MRV2 用独立的 CUDAGraphManager 显式管理，还可以把多步 Draft Decode 一起捕获。需要注意 KV Cache 的 Block 分配、Prefix Cache 和 Hybrid KV Cache Manager 仍属于 V1 Engine Core，MRV2 主要优化的是这些 KV 元数据怎么高效送到 GPU 和 Attention，而不是重新发明一套 PagedAttention。总体来说，V1 解决的是整个引擎架构统一，MRV2 解决的是执行热路径进一步去 CPU 化、异步化和模块化。

---

# 39. 最终总结：只记住这 8 点

1. **当前官方没有简单意义上的完整“V2 Engine”；V2 主要是 Model Runner V2。**
2. **V1 是整个推理核心架构升级；MRV2 是 V1 Engine 内模型执行热路径升级。**
3. **V1 Persistent Batch 将状态和输入绑定较紧；MRV2 将 Persistent State 与每步 Input 解耦。**
4. **MRV2 为活跃 Request 分配固定 State Row，再按每步执行顺序 Gather。**
5. **MRV2 是 Async-First，通过减少 CPU/GPU Race 和 Barrier 增强流水执行。**
6. **StagedWriteTensor、GPU-native Metadata 和 Triton Sampler 都是在降低 CPU/内存开销。**
7. **KV Cache Manager 仍属于 V1 Engine Core；MRV2主要优化 Block Table 等 KV 元数据的执行侧消费。**
8. **CUDA Graph 在 MRV2 中变得显式且更易扩展，尤其适合复杂 Speculative Decode。**

---

# 附录：推荐官方资料阅读顺序

1. **Architecture Overview**  
   先理解当前 V1 Engine 的 API Server、Engine Core、Scheduler、GPU Worker 进程结构。

2. **vLLM V1 User Guide**  
   理解统一 Scheduler、Chunked Prefill、KV Cache 与 V1 核心变化。

3. **Model Runner V2 Design Document**  
   重点阅读 Persistent Batch、Async-First、StagedWriteTensor、GPU-Native Metadata、Triton Sampler 和 CUDAGraphManager。

4. **KV Cache Coordinator / Scheduler API**  
   理解为什么 KV Cache Manager 仍属于 V1 Engine Core。

5. **vLLM Releases**  
   查看 MRV2 当前默认模型与功能覆盖范围。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nanovllm项目面试准备|模块-nanovllm项目面试准备]]

%% 项目关联导航：结束 %%
