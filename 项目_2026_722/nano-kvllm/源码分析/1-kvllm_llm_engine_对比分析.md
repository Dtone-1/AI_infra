# nano-vLLM 与 nano-kvLLM：`llm_engine.py` 源码对比分析

## 1. 对比对象与结论先行

本次对比的两个文件分别为：

- 原版 nano-vLLM：导入路径为 `nanovllm.*` 的 `llm_engine.py`；
- nano-kvLLM：导入路径为 `nanokvllm.*` 的对应文件。

`LLMEngine` 是整个推理系统最外层的编排器。它本身不直接执行 Attention 或 KV Cache 压缩，而是负责把以下模块串起来：

1. 接收并分词用户请求；
2. 将请求交给 `Scheduler`；
3. 调用 `ModelRunner` 在 GPU 上执行 Prefill 或 Decode；
4. 将模型执行结果交回 `Scheduler` 更新请求状态；
5. 收集已完成序列并返回文本结果；
6. 统计 Prefill 和 Decode 吞吐。

nano-kvLLM 对该文件最重要的改造，可以概括为一句话：

> `LLMEngine` 从“只传递生成 token 的普通推理编排器”，变成了“同时传递 token 结果与 KV Cache 压缩事件的压缩感知编排器”。

真正执行压缩算法的主体并不在这个文件中，但这个文件建立了压缩结果从 `ModelRunner` 返回到 `Scheduler` 的关键控制链路。

---

## 2. nano-kvLLM 文件的整体变化概览

| 改动方向 | 原版 nano-vLLM | nano-kvLLM | 工程意义 |
|---|---|---|---|
| 项目命名空间 | `nanovllm.*` | `nanokvllm.*` | 使用改造后的配置、序列、调度器和模型执行器 |
| Sequence 块大小初始化 | `Sequence.block_size = config.kvcache_block_size` | 删除该设置 | `LLMEngine` 不再负责向 `Sequence` 注入固定 KV Cache 块大小，相关职责被迁移或重构 |
| `ModelRunner.run()` 返回值 | 仅返回 `token_ids` | 可返回 `(token_ids, compression_events)` | 模型执行层能够向调度层上报 KV Cache 压缩产生的状态变化 |
| 调度器后处理接口 | `postprocess(seqs, token_ids, is_prefill)` | `postprocess(seqs, token_ids, compression_events)` | 后处理重点从“当前是否 Prefill”转为“如何同步压缩事件” |
| 返回值兼容处理 | 无 | 兼容 tuple 与普通返回值 | 没有压缩事件时仍可沿用只返回 token 的执行路径 |
| Prefill token 统计 | 使用 `seq.num_scheduled_tokens` | 使用 `len(seq)` | 指标统计口径发生变化，需要结合新 `Sequence` 与 `Scheduler` 进一步确认 |
| GPU 计时 | 直接用 `perf_counter()` | 计时前后调用 `torch.cuda.synchronize()` | 避免 CUDA 异步执行导致吞吐统计虚高 |
| Decode 指标 | 只显示当前 step 吞吐 | 额外计算 step 吞吐平均值 | 便于评估压缩前后 Decode 性能，但当前平均方式仍有改进空间 |
| `use_tqdm` 处理 | 总是创建进度条，禁用时设置 `disable=True` | 仅启用时创建和更新进度条 | 避免禁用进度条时执行不必要操作 |

这些变化中，真正与 KV Cache 压缩主链路直接相关的是：

1. 删除 `Sequence.block_size` 的外部固定初始化；
2. `ModelRunner` 返回 `compression_events`；
3. `Scheduler.postprocess()` 接收并处理 `compression_events`。

其余变化主要服务于性能测试、调试和代码适配。

---

## 3. 原版 `LLMEngine` 的执行链路

原版 `step()` 的核心流程可以抽象为：

```text
Scheduler.schedule()
        ↓
得到本轮运行的 seqs 和 is_prefill
        ↓
ModelRunner.run(seqs, is_prefill)
        ↓
只返回 token_ids
        ↓
Scheduler.postprocess(seqs, token_ids, is_prefill)
        ↓
更新序列状态、追加生成 token、结束已完成请求
```

原版系统默认：

- Scheduler 在调度前已经掌握序列对应的 KV Cache 块；
- ModelRunner 只需要根据这些调度结果完成模型前向；
- 前向结束后，Engine 只需把新生成的 token 交回 Scheduler；
- KV Cache 的物理布局不会因为一次模型前向而出现需要额外上报的结构性变化。

因此，原版 `ModelRunner.run()` 的返回值只包含 `token_ids` 就足够了。

---

## 4. nano-kvLLM 的新执行链路

nano-kvLLM 的 `step()` 变为：

```text
Scheduler.schedule()
        ↓
得到本轮运行的 seqs 和 is_prefill
        ↓
ModelRunner.run(seqs, is_prefill)
        ↓
返回 token_ids
或返回 (token_ids, compression_events)
        ↓
LLMEngine 拆分模型输出与压缩事件
        ↓
Scheduler.postprocess(seqs, token_ids, compression_events)
        ↓
同时更新生成状态与压缩后的缓存管理状态
```

这里形成了一个非常重要的分工：

- `ModelRunner` 属于执行侧，接近 GPU 和模型计算，适合实际执行 KV Cache 压缩并产生结果；
- `Scheduler` 属于控制侧，维护请求、序列和缓存资源的全局状态；
- `LLMEngine` 是二者之间的桥梁，将执行侧产生的 `compression_events` 传回控制侧。

这正是推理框架中常见的“数据面与控制面协同”：

- 数据面负责真正计算和移动数据；
- 控制面负责记录资源属于谁、还有多少空间、下一轮应该调度谁。

KV Cache 压缩不能只在 GPU 上把张量压缩完就结束。只要压缩改变了缓存中的有效 token、位置映射、块占用或序列长度表示，Scheduler 所维护的元数据就必须同步更新，否则下一轮 Decode 仍可能按照旧状态读取 KV Cache，造成错读、越界或缓存资源泄漏。

---

## 5. 详细改动分析

### 5.1 导入路径由 `nanovllm` 切换为 `nanokvllm`

原版：

```python
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
```

改造后：

```python
from nanokvllm.config import Config
from nanokvllm.engine.sequence import Sequence
from nanokvllm.engine.scheduler import Scheduler
from nanokvllm.engine.model_runner import ModelRunner
```

这不是简单改包名。`LLMEngine` 调用的四个关键组件已经全部切换到 nano-kvLLM 的实现：

- `Config`：可能增加压缩阈值、保留比例、压缩策略等配置；
- `Sequence`：需要表达压缩前后的逻辑序列状态；
- `Scheduler`：需要根据压缩结果维护缓存资源；
- `ModelRunner`：负责实际模型执行，并上报压缩事件。

因此，`LLMEngine` 虽然改动行数不多，但它接入的是一整套压缩感知的下层实现。

### 5.2 删除 `Sequence.block_size = config.kvcache_block_size`

原版初始化时存在：

```python
Sequence.block_size = config.kvcache_block_size
```

nano-kvLLM 删除了这一行。

在原版中，这一写法把 KV Cache 的 block size 设置为 `Sequence` 类的全局类属性。Sequence 可以据此根据 token 数量计算：

- 当前占用多少个 block；
- 最后一个 block 是否填满；
- 是否需要申请新的 block。

删除后，至少可以确定一件事：

> nano-kvLLM 不再由 `LLMEngine` 统一向 `Sequence` 注入固定 block size，块大小或缓存占用计算的职责已经被迁移或重构。

这与 KV Cache 压缩存在合理联系。压缩后：

- 逻辑上下文长度不一定等于物理保留的 KV 数量；
- “序列有多少 token”不能再简单等价为“缓存占用多少 block”；
- 缓存资源计算可能需要压缩后的有效长度、保留索引或映射信息。

但是，仅凭这两个文件不能断言 nano-kvLLM 已完全取消块式管理，也不能确定新逻辑具体放在哪个类中。需要继续对比 `sequence.py`、`block_manager.py` 或 `scheduler.py` 才能确认。

### 5.3 `ModelRunner.run()` 从单一返回值变为复合返回协议

原版：

```python
token_ids = self.model_runner.call("run", seqs, is_prefill)
```

nano-kvLLM：

```python
ret = self.model_runner.call("run", seqs, is_prefill)
if isinstance(ret, tuple):
    token_ids, compression_events = ret
else:
    token_ids, compression_events = ret, None
```

这是本文件最核心的改动。

原版执行层只上报新生成的 token。nano-kvLLM 执行层还可以上报 `compression_events`。从系统设计角度看，压缩事件通常用于表达：

- 哪些请求或序列发生了压缩；
- 压缩发生在哪一层或哪一次 Decode；
- 原 KV Cache 长度与压缩后长度；
- 哪些 token 的 KV 被保留或淘汰；
- 缓存块、槽位或位置映射发生了什么变化；
- Scheduler 下一轮调度前需要修正哪些元数据。

上述字段的准确内容必须以 `ModelRunner` 和 `Scheduler` 的具体定义为准，但接口层面的意义已经很明确：

> KV Cache 压缩不再是 ModelRunner 内部不可见的局部操作，而被提升为需要跨模块同步的系统事件。

### 5.4 为无压缩路径保留兼容性

代码没有强制要求 `ModelRunner` 每次都返回 tuple，而是做了兼容判断：

```python
if isinstance(ret, tuple):
    token_ids, compression_events = ret
else:
    token_ids, compression_events = ret, None
```

这意味着以下情况都可以被统一处理：

- 本轮发生压缩：返回 token 和压缩事件；
- 本轮未发生压缩：可能只返回 token；
- 某种关闭压缩的配置：继续沿用普通推理路径；
- 部分模型或执行模式暂未接入压缩：不必立刻修改所有返回协议。

这种设计降低了改造侵入性，使压缩功能可以作为可选能力接入原有推理主循环。

不过，从接口规范角度，更稳定的做法通常是始终返回结构一致的对象，例如始终返回 `(token_ids, compression_events)`，其中没有事件时令后者为空列表或 `None`。当前通过 `isinstance(ret, tuple)` 判断虽然简单，但返回类型不完全统一，会增加调用方理解和类型检查成本。

### 5.5 `Scheduler.postprocess()` 的职责发生变化

原版：

```python
self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

nano-kvLLM：

```python
self.scheduler.postprocess(seqs, token_ids, compression_events)
```

原版后处理显式接收 `is_prefill`，说明 Scheduler 需要根据 Prefill/Decode 阶段采用不同的状态更新逻辑。

改造后，第三个参数换成了 `compression_events`。这说明 nano-kvLLM 的 Scheduler 后处理至少增加了一个更关键的任务：

1. 追加生成 token；
2. 判断 EOS 或达到最大生成长度；
3. 结束并回收已完成请求；
4. 根据压缩事件修正序列与 KV Cache 管理状态。

这一接口变化把压缩功能真正接入了调度闭环：

```text
调度缓存资源
→ GPU 执行并可能压缩
→ 返回压缩事件
→ Scheduler 修正资源状态
→ 下一轮基于新状态继续调度
```

如果缺少最后一步，压缩只能减少某个 GPU 张量的局部数据量，却无法可靠释放或复用系统层面的缓存资源。

另外，`is_prefill` 不再显式传入，并不代表 Scheduler 完全不区分 Prefill 与 Decode。可能的情况包括：

- 新版 `postprocess()` 只处理 Decode；
- Scheduler 可以从 Sequence 状态中自行判断阶段；
- 阶段相关逻辑已被移到其他方法；
- `compression_events` 本身携带了足够信息。

其准确机制需结合新版 `scheduler.py` 判断。

### 5.6 Prefill token 统计口径发生变化

原版在执行模型前统计：

```python
num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
```

nano-kvLLM 在后处理后统计：

```python
num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
```

两者并不完全等价：

- `seq.num_scheduled_tokens` 表示本轮实际被调度执行的 token 数；
- `len(seq)` 通常表示序列当前总长度。

在“每个请求只进行一次完整 Prefill”的简单情况下，两者可能得到相同结果。但如果系统支持以下能力，就可能产生偏差：

- Chunked Prefill；
- Prefix Cache 命中，只计算未命中的后缀；
- 被抢占后恢复 Prefill；
- 一个 Sequence 分多轮完成 Prefill；
- 后处理改变了 Sequence 长度。

因此，这项修改更像是 nano-kvLLM 为实验统计做出的简化，不应直接理解成压缩算法所必需的逻辑。评测时需要确认其统计的是“本轮实际计算 token”还是“序列总 token”。

### 5.7 加入 `torch.cuda.synchronize()`，修正 GPU 异步计时

nano-kvLLM 新增：

```python
import torch
```

并在每轮计时前后调用：

```python
torch.cuda.synchronize()
t = perf_counter()
output, num_tokens = self.step()
torch.cuda.synchronize()
```

CUDA Kernel 默认异步执行。CPU 调用 GPU 后，Python 代码可能很快返回，但 GPU 计算还没有真正结束。如果直接计算：

```python
perf_counter() - t
```

测到的可能主要是 CPU 提交 Kernel 的时间，而不是 GPU 完成推理的真实时间，最终导致吞吐率虚高。

同步后的计时区间更接近：

```text
本轮所有先前 GPU 工作完成
→ 开始计时
→ 执行本轮推理与压缩
→ 等待本轮 GPU 工作真正完成
→ 停止计时
```

这对 nano-kvLLM 很重要，因为项目需要比较：

- 不压缩与压缩的 Decode 吞吐；
- 压缩操作本身带来的额外时延；
- 上下文变短后 Attention 计算节省的时间；
- 显存节省与吞吐变化之间的权衡。

需要注意：`torch.cuda.synchronize()` 会强制 CPU 等待 GPU，破坏流水并增加同步开销。因此它适合 Benchmark，不适合直接保留在追求最大线上吞吐的生产热路径中。

### 5.8 新增平均 Decode 吞吐统计

nano-kvLLM 新增：

```python
decode_tp_sum = 0.0
decode_tp_steps = 0
```

循环结束后输出：

```python
avg_decode_tp = decode_tp_sum / decode_tp_steps
```

设计目的很明确：单步 Decode 吞吐会随以下因素波动：

- 当前活跃请求数量；
- 部分请求结束导致 batch 变小；
- 不同 step 是否触发 KV Cache 压缩；
- 压缩后上下文长度变化；
- GPU Kernel 调度和同步抖动。

因此，只显示最新一步吞吐不足以评价压缩方案，增加平均指标可以提供更稳定的观察值。

但当前实现存在两个统计问题。

#### 问题一：Prefill step 也被计入 `decode_tp_steps`

当前代码无论 `num_tokens` 正负，都会执行：

```python
decode_tp_sum += decode_throughput
decode_tp_steps += 1
```

在 Prefill step 中，`decode_throughput` 通常仍为 0，因此最终平均值会被 Prefill step 拉低。更合理的写法是只在 `num_tokens < 0` 时累计。

#### 问题二：对每步吞吐做算术平均并非严格的总体吞吐

若每一步 batch 大小或耗时不同，直接计算：

```text
(step1 tok/s + step2 tok/s + …) / step 数
```

会让每个 step 拥有相同权重。严格的总体 Decode 吞吐应计算：

```text
所有 Decode step 生成的 token 总数
÷
所有 Decode step 的总耗时
```

后者更适合写入项目 Benchmark 和简历指标。

### 5.9 `use_tqdm` 分支更加显式

原版始终创建进度条：

```python
pbar = tqdm(..., disable=not use_tqdm)
```

nano-kvLLM 改为只在启用时创建、更新和关闭：

```python
if use_tqdm:
    pbar = tqdm(...)
```

并为 `set_postfix()`、`update()` 和 `close()` 增加保护。

这是代码组织和测试适配上的改动，不属于 KV Cache 压缩核心逻辑。其好处是禁用进度条时不会创建对象，也不会反复执行进度条更新代码。

### 5.10 调试注释

nano-kvLLM 在 `add_request()` 中加入：

```python
# print("prompt len is", len(prompt))
```

这说明开发者曾关注输入长度。KV Cache 压缩实验通常高度依赖 prompt 长度，因为上下文越长：

- KV Cache 显存占用越大；
- Decode 读取 KV 的带宽开销越高；
- 压缩可能获得的收益越明显。

但该行已被注释，不会影响实际功能，只能视为实验调试痕迹。

---

## 6. 哪些改动真正支撑了 KV Cache 压缩

可以把本文件中的改动分为三层。

### 第一层：压缩主链路改造

```text
ModelRunner 返回 compression_events
        ↓
LLMEngine 接收并转发
        ↓
Scheduler 根据事件更新状态
```

这是本文件最核心的价值。它使 KV Cache 压缩从一个局部张量操作，变成推理引擎各模块能够共同感知的系统行为。

### 第二层：缓存模型职责重构

删除：

```python
Sequence.block_size = config.kvcache_block_size
```

表明原版“序列长度按固定 block size 映射为缓存占用”的部分职责正在调整。压缩后，逻辑 token 长度、有效 KV 长度和物理缓存占用可能不再完全一致，因此缓存状态不能只依赖一个全局固定类属性完成表达。

### 第三层：压缩效果 Benchmark

新增：

- CUDA 同步计时；
- Decode step 吞吐记录；
- 平均 Decode 吞吐输出。

这些代码不执行压缩，但用于回答项目最关键的实验问题：

> KV Cache 压缩虽然节省显存，但是否引入了额外时延？在长上下文 Decode 中，压缩后的 Attention 是否足以抵消压缩操作本身的成本？

---

## 7. 这份文件没有完成什么

需要特别注意，`llm_engine.py` 只是压缩控制链路的一部分。仅从本文件看不到：

- 采用什么压缩算法；
- 如何计算 token 重要性；
- 保留哪些 KV、淘汰哪些 KV；
- 是逐层压缩还是统一压缩；
- 压缩发生在 Prefill 后还是 Decode 过程中；
- 如何重排 KV Cache 张量；
- RoPE 位置与压缩后索引如何处理；
- 释放的物理块如何重新加入空闲池；
- 压缩是否影响模型输出精度。

这些问题应继续在以下文件中寻找答案：

1. `model_runner.py`：压缩何时触发、`compression_events` 如何产生；
2. `scheduler.py`：压缩事件如何改变序列和缓存资源状态；
3. `sequence.py`：逻辑长度、缓存长度、压缩状态如何表达；
4. `block_manager.py` 或缓存管理模块：物理块如何释放与复用；
5. Attention/KV Cache 相关层：压缩后的索引与张量如何参与后续 Decode。

---

## 8. 建议继续追踪的调用关系

下一步阅读时，可以围绕 `compression_events` 做反向和正向追踪：

```text
compression_events 在哪里定义？
        ↓
ModelRunner 的哪个函数创建它？
        ↓
什么条件触发压缩？
        ↓
事件中记录了哪些字段？
        ↓
Scheduler.postprocess() 如何解析这些字段？
        ↓
Sequence 或 BlockManager 的哪些状态被修改？
        ↓
下一轮 ModelRunner 如何读取压缩后的新状态？
```

只要把这条闭环追通，就能理解 nano-kvLLM 的压缩机制如何从算法落到推理系统工程实现。

---

## 9. 需要警惕的实现与评测问题

### 9.1 `compression_events` 返回类型不固定

`ModelRunner.run()` 有时返回 token，有时返回 tuple。建议后续统一返回一个固定结构，降低接口歧义。

### 9.2 Prefill token 数可能被高估

`sum(len(seq) for seq in seqs)` 统计的是序列长度，不一定等于本轮真正执行的 token 数。对 Chunked Prefill 或 Prefix Cache 场景尤其需要核对。

### 9.3 平均 Decode 吞吐计算不够严格

当前实现把 Prefill step 也计入平均步数，并采用 step 吞吐算术平均。正式实验应改为“Decode token 总数 / Decode 总时间”。

### 9.4 Benchmark 同步会降低真实服务吞吐

同步计时适合离线测量，但生产服务不应在每个 step 强制同步，否则会阻断 CPU-GPU 异步流水。

### 9.5 只测吞吐不能证明压缩方案有效

完整实验至少还应同时测量：

- KV Cache 显存占用或峰值 GPU 显存；
- 最大可支持上下文长度；
- 最大并发请求数；
- Decode 吞吐和 TPOT；
- TTFT；
- 压缩触发耗时；
- 输出质量或困惑度变化；
- 不同输入长度、输出长度和压缩率下的结果。

---

## 10. 最终总结

nano-kvLLM 对 `llm_engine.py` 的代码改动不算多，但架构意义较大。

原版 `LLMEngine` 的核心闭环是：

```text
调度请求 → 执行模型 → 返回 token → 更新序列
```

nano-kvLLM 将其扩展为：

```text
调度请求与缓存状态
→ 执行模型并可能压缩 KV Cache
→ 返回 token 与 compression_events
→ Scheduler 同步生成状态和压缩后的缓存状态
→ 基于新缓存状态进入下一轮调度
```

因此，这个文件并不是 KV Cache 压缩算法的实现位置，而是压缩功能接入推理引擎主循环的“总线”和“桥梁”。它解决的关键工程问题是：

> GPU 执行侧发生 KV Cache 压缩后，如何让调度侧及时、正确地知道缓存布局已经变化，并在下一轮推理中继续使用一致的状态。

此外，nano-kvLLM 加入 CUDA 同步计时和 Decode 吞吐统计，说明项目不仅要实现压缩，还要量化压缩对性能的影响。不过当前统计口径仍有若干不严谨之处，正式 Benchmark 前应进一步修正。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-kvllm|模块-nano-kvllm]]

%% 项目关联导航：结束 %%
