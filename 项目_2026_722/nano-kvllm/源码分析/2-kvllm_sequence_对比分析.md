# nano-vLLM 与 nano-kvLLM：`sequence.py` 源码对比分析

## 1. 对比对象与结论

本次对比的两个文件分别是：

- 原版 nano-vLLM 的 `sequence.py`
- nano-kvLLM 的 `sequence.py`

`Sequence` 可以理解为推理引擎中“一条请求当前运行状态的内存档案”。它不仅保存 token，还连接着调度器、KV Cache 管理器、模型执行器和多进程通信。

原版 nano-vLLM 中，`Sequence` 的核心目标是回答：

> 这条请求有多少 token、当前处于 Prefill 还是 Decode、调度器这轮准备处理多少 token、对应哪些 KV Cache Block？

nano-kvLLM 中，`Sequence` 的核心目标进一步扩展为：

> 在部分历史 KV Cache 已经被压缩或删除后，这条请求逻辑上生成到了哪里、RoPE 应使用什么位置、还有多少新 token 没有参与压缩、压缩事件应该对应哪条序列？

因此，这个文件的变化不是简单增加几个计数器，而是在建立一套能够区分以下两种长度的状态模型：

1. **逻辑序列长度**：模型真实读入和生成过多少 token；
2. **物理缓存长度**：GPU 的 KV Cache 当前实际保留多少 token。

这是 KV Cache 压缩能够正确工作的基础。

---

## 2. nano-kvLLM 文件的整体变化概览

| 改动方向 | 原版 nano-vLLM | nano-kvLLM | 工程意义 |
|---|---|---|---|
| 包命名空间 | `nanovllm` | `nanokvllm` | 使用改造后的独立工程模块 |
| 调度状态 | `num_scheduled_tokens`、`is_prefill` | 删除 | 调度阶段信息不再长期存放在 `Sequence` 中 |
| 生成进度 | 通过 `num_tokens - num_prompt_tokens` 间接计算 | 新增 `generated_completion_tokens` | 将生成进度与缓存压缩后的物理长度解耦 |
| RoPE 位置 | 默认可由 token 长度推导 | 新增 `rope_pos` | 压缩缓存后仍保持原始逻辑位置 |
| 压缩尾部 | 无 | 新增 `tail_uncompressed_len` | 记录尚未参与压缩的新生成后缀 |
| 已缓存块数量 | 无显式属性 | 新增 `num_cached_blocks` | 方便按 Block 管理和释放压缩后的缓存 |
| token 追加 | 只更新 token 列表和总长度 | 同时更新三个压缩相关计数器 | 每生成一个 token，同步推进逻辑时间轴 |
| 多进程序列化 | 固定位置的 tuple | 带字段名的 dict | 更适合扩展压缩元数据 |
| Decode 通信内容 | Decode 时只传 `last_token` | 始终传完整 `token_ids` | 压缩相关组件可以访问完整 token 历史 |
| 序列身份 | 不序列化 `seq_id` | 序列化 `seq_id` | 压缩事件可准确映射回请求 |

其中最核心的改动可以概括为：

> nano-kvLLM 将 `Sequence` 从“普通调度状态对象”改造成“逻辑序列状态与压缩缓存状态的统一载体”。

---

## 3. 原版 `Sequence` 在推理系统中的职责

在原版 nano-vLLM 中，一条请求进入系统后大致经历：

```text
Prompt token
    ↓
创建 Sequence
    ↓
Scheduler 选择本轮运行的 Sequence
    ↓
ModelRunner 执行 Prefill 或 Decode
    ↓
生成新 token
    ↓
Sequence.append_token()
    ↓
判断 EOS 或 max_tokens
```

原版 `Sequence` 主要保存四类信息。

### 3.1 请求身份与生命周期

```python
self.seq_id
self.status
```

- `seq_id`：请求唯一编号；
- `status`：请求当前处于 `WAITING`、`RUNNING` 或 `FINISHED`。

### 3.2 token 信息

```python
self.token_ids
self.last_token
self.num_tokens
self.num_prompt_tokens
```

这些字段可以支持：

- 获取完整 Prompt；
- 获取生成结果；
- 判断当前序列长度；
- 在 Decode 阶段取最后一个 token 作为下一轮输入。

### 3.3 KV Cache 映射

```python
self.num_cached_tokens
self.block_table
```

- `num_cached_tokens`：已经写入 KV Cache 的 token 数；
- `block_table`：该 Sequence 占用了哪些 KV Cache Block。

### 3.4 调度阶段信息

```python
self.num_scheduled_tokens
self.is_prefill
```

- `num_scheduled_tokens`：本轮调度准备处理多少 token；
- `is_prefill`：该 Sequence 是否还在 Prefill 阶段。

原版设计中，序列 token 数量、RoPE 位置和 KV Cache 长度通常保持单调同步，因此很多状态可以由 `num_tokens` 间接推导。

引入 KV Cache 压缩后，这种假设不再成立。

---

## 4. 为什么 KV Cache 压缩会破坏原版状态假设

假设一条请求：

- Prompt 有 1,000 个 token；
- Decode 已生成 500 个 token；
- 逻辑总长度为 1,500；
- 压缩算法只在 KV Cache 中保留 700 个重要 token。

此时系统中同时存在：

```text
逻辑总长度                 = 1500
生成 token 数              = 500
下一 token 的逻辑位置      = 1500
物理 KV Cache 中保留长度   = 700
```

如果仍然用物理缓存长度推导所有信息，就可能出现以下错误：

1. 把下一个 token 的 RoPE 位置错误地设置为 700；
2. 错误地认为只生成了部分 token；
3. 破坏 `max_tokens` 停止条件；
4. 压缩后重新申请或释放错误数量的 KV Block；
5. 无法判断哪些新 token 尚未参与下一轮压缩。

因此，压缩系统必须把“模型逻辑时间轴”和“GPU 缓存占用”拆开记录。

nano-kvLLM 新增的三个字段正是为此服务：

```python
generated_completion_tokens
rope_pos
tail_uncompressed_len
```

---

## 5. 详细改动一：删除 `num_scheduled_tokens`

原版：

```python
self.num_scheduled_tokens = 0
```

nano-kvLLM 删除了这个字段。

### 5.1 原版字段的作用

在支持 Chunked Prefill 或细粒度调度时，一条长 Prompt 不一定一轮全部执行完。调度器可以设置：

```text
本轮只处理该 Sequence 的一部分 token
```

`num_scheduled_tokens` 就是这轮真正送入模型的 token 数。

### 5.2 nano-kvLLM 删除它意味着什么

结合 nano-kvLLM 的 `LLMEngine.step()`，Prefill 吞吐统计改为直接根据 `len(seq)` 计算，而不是读取 `seq.num_scheduled_tokens`。

这说明 nano-kvLLM 当前实现至少在这一条调用链中，不再依赖 `Sequence` 保存每轮局部调度长度。可能的设计方向包括：

- Prefill 按完整序列处理；
- 调度长度被移动到其他对象或临时变量；
- 当前版本弱化或取消了原版的 Chunked Prefill 状态；
- KV 压缩实验更聚焦 Decode 阶段，而不是复杂的 Prefill 分块调度。

### 5.3 工程影响

优点：

- `Sequence` 状态更简单；
- 避免调度临时状态和压缩长期状态混杂；
- 减少跨进程序列化字段。

代价：

- 若后续重新支持 Chunked Prefill，必须在其他位置恢复“本轮调度 token 数”；
- 不能仅靠当前 `Sequence` 判断一条 Prompt 的分块 Prefill 进度。

因此，这项修改本身不是 KV Cache 压缩算法的核心，但反映了 nano-kvLLM 对原有调度链路进行了简化或重构。

---

## 6. 详细改动二：删除 `is_prefill`

原版：

```python
self.is_prefill = True
```

nano-kvLLM 删除了该字段。

### 6.1 原版中的意义

原版 `Sequence.__getstate__()` 会根据 `is_prefill` 决定跨进程传输内容：

```python
last_state = self.last_token if not self.is_prefill else self.token_ids
```

也就是说：

- Prefill：模型需要完整 Prompt，因此发送完整 `token_ids`；
- Decode：模型通常只需要最新 token，因此只发送 `last_token`。

这是一个典型的 IPC 数据量优化。

### 6.2 nano-kvLLM 为什么不再这样做

nano-kvLLM 的 `__getstate__()` 始终序列化完整 `token_ids`，因此不再需要 `is_prefill` 决定传输格式。

压缩系统可能需要完整 token 历史来完成：

- 根据 token 位置解释缓存压缩结果；
- 维护压缩前后的 token 映射；
- 根据 `seq_id` 处理压缩事件；
- 在压缩策略中访问历史 token 或逻辑位置；
- 恢复序列的完整逻辑状态。

### 6.3 代价

原版 Decode 阶段只传最后一个 token，而 nano-kvLLM 传完整 token 列表，序列越长，进程间序列化与复制开销越大。

因此这里存在清晰的权衡：

```text
更完整、可恢复的压缩状态
            ↕
更大的 CPU 内存和 IPC 开销
```

对于教学或实验项目，这种实现更直观、更安全；对于生产级长上下文推理框架，则需要进一步优化，例如只传增量 token、共享元数据或使用独立压缩状态结构。

---

## 7. 详细改动三：新增 `generated_completion_tokens`

nano-kvLLM：

```python
self.generated_completion_tokens = 0
```

每追加一个 token：

```python
self.generated_completion_tokens += 1
```

### 7.1 为什么已有 `num_completion_tokens` 还要新增它

文件中原本已有：

```python
@property
def num_completion_tokens(self):
    return self.num_tokens - self.num_prompt_tokens
```

从当前文件看，两者在正常情况下数值相同。

但新增独立计数器体现了一个重要意图：

> 生成进度不应该依赖缓存压缩后的长度或 token 容器的具体组织方式。

未来如果压缩实现进一步修改：

- `num_tokens` 的定义；
- `token_ids` 的保存方式；
- Prompt 与生成 token 的映射；
- 物理缓存中的 token 数量；

`generated_completion_tokens` 仍然可以直接表示真实 Decode 次数。

### 7.2 它解决什么问题

最直接的用途是保证停止条件正确：

```text
generated_completion_tokens >= max_tokens
```

KV Cache 压缩只允许改变“保存多少历史 KV”，不能让系统误以为已经生成的 token 消失了。

### 7.3 与 `num_completion_tokens` 的关系

当前实现中存在一定信息重复：

```text
generated_completion_tokens
≈ num_tokens - num_prompt_tokens
```

这不是必然错误，而是为了显式区分概念：

- `num_completion_tokens`：由序列长度推导；
- `generated_completion_tokens`：独立记录生成过程。

但后续代码应统一规定哪个字段是权威来源，否则两个字段更新不一致时会产生状态漂移。

---

## 8. 详细改动四：新增 `rope_pos`

nano-kvLLM 初始化：

```python
self.rope_pos = self.num_tokens - 1
```

追加 token 时：

```python
self.rope_pos += 1
```

### 8.1 RoPE 位置为什么不能等于压缩后的 KV Cache 长度

RoPE 将 token 的逻辑位置编码进 Query 和 Key。假设序列原本生成到了位置 1,499，即使 KV Cache 压缩后只保留 700 个 token，下一个 token 的位置仍然应是 1,500，而不是 700。

错误做法：

```text
next_position = 当前缓存中保留的 token 数
```

正确做法：

```text
next_position = 原始逻辑时间轴上的位置
```

`rope_pos` 就是用于保存这一逻辑位置。

### 8.2 为什么这是压缩正确性的关键

如果压缩后重置位置编号：

- 新 token 的 Q 使用错误 RoPE 相位；
- 保留下来的历史 K 使用旧位置；
- Q 和 K 的相对位置关系被破坏；
- 注意力分数不再对应模型训练时的位置语义；
- 输出质量会明显下降。

因此：

> KV Cache 可以删除，逻辑位置不能倒退。

### 8.3 初始化语义

Prompt 长度为 `N` 时：

```python
rope_pos = N - 1
```

表示当前最后一个 Prompt token 的位置是 `N - 1`。

每生成一个新 token 后位置加一，始终保持：

```text
rope_pos = 当前序列最后一个 token 的逻辑位置
```

需要注意：模型执行时究竟使用当前 `rope_pos`，还是使用 `rope_pos + 1` 作为新 token 的位置，要结合 ModelRunner 中的具体调用判断。仅凭本文件不能最终确定调用端的加一时机。

---

## 9. 详细改动五：新增 `tail_uncompressed_len`

nano-kvLLM：

```python
self.tail_uncompressed_len = 0
```

每生成一个 token：

```python
self.tail_uncompressed_len += 1
```

### 9.1 “未压缩尾部”是什么

KV Cache 压缩一般不会在每生成一个 token 后立即执行，否则压缩计算、数据搬移和 Block 重排开销可能过大。

更常见的方式是：

```text
已压缩的历史区域 + 最近新生成、暂未压缩的尾部区域
```

例如：

```text
[压缩后保留的历史 KV] [新生成 64 个 token 的完整 KV]
                         ↑
                tail_uncompressed_len = 64
```

### 9.2 它可能参与哪些判断

典型逻辑可能是：

```text
如果 tail_uncompressed_len 达到压缩间隔
    触发一次压缩
    更新缓存映射
    将 tail_uncompressed_len 清零或减去已处理长度
```

本文件只负责递增，真正的触发、压缩和重置应在 Scheduler、ModelRunner 或压缩管理模块中完成。

### 9.3 为什么不能直接用总长度取模

单独保存尾部长度更可靠，因为：

- 压缩可能不是固定间隔；
- 某次压缩可能只处理部分尾部；
- 不同序列可能在不同 Decode step 触发；
- 压缩失败或跳过时需要保留未处理长度；
- 批处理中不同请求的压缩进度可能不同。

因此 `tail_uncompressed_len` 是连接“逐 token Decode”和“周期性压缩”的桥梁。

---

## 10. 详细改动六：新增 `num_cached_blocks`

nano-kvLLM：

```python
@property
def num_cached_blocks(self):
    return self.num_cached_tokens // self.block_size
```

### 10.1 该属性表达什么

它将缓存 token 数换算成完整 KV Block 数：

```text
完整已缓存 Block 数 = 已缓存 token 数 // Block 大小
```

KV Cache 管理器通常按 Block 分配、引用和释放显存，而不是按单个 token 操作，因此这个属性能让调度器或缓存管理器更方便地进行：

- Block 数量检查；
- 压缩后释放 Block；
- 重新构建 Block Table；
- 估算一条序列的物理缓存占用。

### 10.2 与 `num_blocks` 的区别

文件中已有：

```python
num_blocks = ceil(num_tokens / block_size)
```

两者不能混淆：

| 属性 | 计算依据 | 含义 |
|---|---|---|
| `num_blocks` | `num_tokens` | 逻辑完整序列需要多少 Block |
| `num_cached_blocks` | `num_cached_tokens` | 当前已缓存 token 中包含多少完整 Block |

在压缩前，两者可能接近；压缩后，它们可能明显不同。

### 10.3 需要注意的边界

该属性使用向下取整：

```python
num_cached_tokens // block_size
```

如果 `num_cached_tokens` 包含一个未满 Block，这个属性不会把它计入。

这可能是有意的，因为它表示“完整缓存块数”；也可能要求调用端额外处理尾块。必须结合 Block Manager 的逻辑判断，不能直接把它理解为当前实际占用的全部 Block 数。

---

## 11. 详细改动七：`append_token()` 同时推进三条状态轴

原版：

```python
self.token_ids.append(token_id)
self.last_token = token_id
self.num_tokens += 1
```

nano-kvLLM 增加：

```python
self.generated_completion_tokens += 1
self.rope_pos += 1
self.tail_uncompressed_len += 1
```

这意味着每生成一个 token，系统同时推进：

```text
逻辑序列长度轴：num_tokens
生成进度轴：generated_completion_tokens
位置编码轴：rope_pos
压缩周期轴：tail_uncompressed_len
```

这四个字段虽然当前都以 1 为步长增长，但语义完全不同：

- `num_tokens`：序列总长度；
- `generated_completion_tokens`：已生成多少 completion token；
- `rope_pos`：逻辑位置；
- `tail_uncompressed_len`：距离上次压缩后新增多少 token。

压缩发生时，一般只应修改与物理缓存相关的状态，例如：

- `num_cached_tokens`；
- `block_table`；
- `tail_uncompressed_len`。

不应回退：

- `num_tokens`；
- `generated_completion_tokens`；
- `rope_pos`。

这就是 nano-kvLLM 中最重要的状态不变量。

---

## 12. 详细改动八：序列化格式从 tuple 改为 dict

### 12.1 原版 tuple 格式

原版：

```python
return (
    self.num_tokens,
    self.num_prompt_tokens,
    self.num_cached_tokens,
    self.num_scheduled_tokens,
    self.block_table,
    last_state,
)
```

优点：

- 紧凑；
- 创建和读取成本低；
- 字段少时实现简单。

缺点：

- 强依赖字段顺序；
- 增加字段时必须同步修改拆包顺序；
- 不直观；
- 不适合快速扩展大量压缩元数据。

### 12.2 nano-kvLLM dict 格式

nano-kvLLM：

```python
return {
    "num_tokens": ...,
    "num_prompt_tokens": ...,
    "num_cached_tokens": ...,
    "block_table": ...,
    "token_ids": ...,
    "last_token": ...,
    "generated_completion_tokens": ...,
    "rope_pos": ...,
    "seq_id": ...,
    "tail_uncompressed_len": ...,
}
```

优势：

1. 字段有明确名称；
2. 增加压缩状态时不依赖固定位置；
3. `__setstate__()` 可以为缺失字段设置默认值；
4. 调试多进程状态更方便；
5. 更适合压缩功能继续演进。

### 12.3 兼容性需要准确理解

新实现使用：

```python
state.get("field", default)
```

这可以兼容“新 dict 状态中缺少某些字段”的情况。

但它不能直接读取原版的 tuple 状态，因为 tuple 没有 `.get()` 方法。因此它不是完整的跨版本兼容设计。

更准确地说：

> 新实现具备同一 dict 协议内部的字段缺省兼容能力，但不兼容原版 tuple 序列化协议。

主进程和 Worker 必须使用同一版本代码。

---

## 13. 详细改动九：始终序列化完整 `token_ids`

原版根据阶段选择：

```text
Prefill：发送完整 token_ids
Decode：只发送 last_token
```

nano-kvLLM 始终发送：

```python
"token_ids": self.token_ids
```

### 13.1 功能收益

Worker 或压缩模块能够获得：

- 完整 Prompt；
- 完整生成结果；
- token 下标与逻辑位置；
- 压缩前后的历史映射基础。

这降低了压缩状态恢复的复杂度。

### 13.2 性能代价

对于长度为 `L` 的序列，每个 Decode step 都序列化完整列表，可能形成近似累积开销：

```text
1 + 2 + 3 + ... + L
```

在极端情况下，CPU 侧数据处理量可能随生成长度呈二次增长趋势。实际开销还取决于：

- 进程通信机制；
- Python 对象是否被完整复制；
- tensor parallel 进程数量；
- Sequence 是否每轮都经过 pickle；
- 批大小和上下文长度。

这会削弱 KV Cache 压缩带来的部分性能收益，因为 GPU 显存下降不代表 CPU/IPC 开销也下降。

生产级优化方向包括：

- Decode 只发送新增 token；
- 独立维护压缩元数据；
- 使用共享内存；
- 使用 tensor 而不是 Python list；
- 仅在压缩触发时发送必要历史信息。

---

## 14. 详细改动十：新增 `seq_id` 序列化

nano-kvLLM：

```python
"seq_id": getattr(self, "seq_id", None)
```

反序列化：

```python
self.seq_id = state.get("seq_id", None)
```

### 14.1 为什么原版可以不传

原版 Worker 主要执行模型计算，返回结果通常可以按照输入批次顺序与 Sequence 对齐，因此不一定需要在反序列化对象中恢复 `seq_id`。

### 14.2 为什么压缩系统更需要它

nano-kvLLM 的模型执行结果除了 `token_ids`，还可能返回 `compression_events`。

压缩事件可能包含：

- 哪条 Sequence 触发压缩；
- 删除或保留了哪些位置；
- 释放了哪些 Block；
- 新的缓存长度；
- Block Table 如何变化。

此时稳定的 `seq_id` 可以让主进程将压缩结果准确映射回原请求。

因此，`seq_id` 从一个只在调度器内部使用的身份字段，变成跨进程压缩协议的一部分。

---

## 15. 详细改动十一：更健壮但也更宽松的反序列化

nano-kvLLM 对 `token_ids` 和 `last_token` 增加了多种兜底逻辑：

```python
if token_ids is not None:
    ...
else:
    ...
```

同时还处理：

- `token_ids` 不是 list；
- `token_ids` 为空；
- `last_token` 缺失；
- 新字段缺失。

### 15.1 好处

- 更容易调试和兼容不同调用路径；
- 压缩状态部分缺失时不一定立即崩溃；
- 可以恢复更完整的 Sequence 对象；
- 为后续新增字段预留空间。

### 15.2 风险

下面的兜底：

```python
self.last_token = 0
```

可能掩盖上游状态损坏。Token ID 0 不一定是合理的输入 token，静默继续运行可能比立即报错更难排查。

更严格的工程实现通常会：

- 对必需字段使用断言或显式异常；
- 只对真正可选字段设置默认值；
- 对序列化协议增加版本号；
- 验证 `num_tokens`、`token_ids` 和 `last_token` 的一致性。

---

## 16. nano-kvLLM 中应维持的关键状态不变量

理解这个文件最重要的不是记住字段，而是理解字段之间应该满足哪些关系。

### 16.1 生成状态关系

在未发生异常修改时：

```text
num_tokens
= num_prompt_tokens + generated_completion_tokens
```

当前代码中也应近似满足：

```text
generated_completion_tokens
= num_completion_tokens
```

### 16.2 位置关系

对于从 0 开始的位置编号：

```text
rope_pos = num_tokens - 1
```

即使 KV Cache 被压缩，这个关系仍不应被物理缓存长度替代。

### 16.3 缓存关系

压缩后通常可能出现：

```text
num_cached_tokens < num_tokens
```

这是正常现象，不应被当成状态错误。

### 16.4 尾部关系

应满足：

```text
0 <= tail_uncompressed_len <= 最近一次压缩后新生成的 token 数
```

压缩执行后，应由其他模块重置或扣减该值。

### 16.5 Block 关系

```text
block_table
```

描述物理 KV Block，而：

```text
num_blocks
```

是根据逻辑 token 总数计算。

压缩后不能简单假设：

```text
len(block_table) == num_blocks
```

更可能需要根据压缩后的实际缓存布局解释。

---

## 17. 一条 Sequence 在 nano-kvLLM 中的完整生命周期

### 阶段一：创建请求

假设 Prompt 长度为 1,000：

```text
num_tokens = 1000
num_prompt_tokens = 1000
generated_completion_tokens = 0
rope_pos = 999
tail_uncompressed_len = 0
```

### 阶段二：Prefill

模型为 Prompt 建立 KV Cache：

```text
num_cached_tokens → 1000
block_table → 对应的物理 Block
```

### 阶段三：逐 token Decode

生成一个 token 后：

```text
num_tokens = 1001
generated_completion_tokens = 1
rope_pos = 1000
tail_uncompressed_len = 1
```

连续生成 64 个 token 后：

```text
num_tokens = 1064
generated_completion_tokens = 64
rope_pos = 1063
tail_uncompressed_len = 64
```

### 阶段四：触发压缩

假设算法把 1,064 个逻辑位置对应的 KV 压缩为 600 个保留项。

压缩后合理的状态应类似：

```text
num_tokens = 1064                    # 不变
generated_completion_tokens = 64     # 不变
rope_pos = 1063                      # 不变
num_cached_tokens = 600              # 物理缓存减少
block_table = 新的缓存块映射
tail_uncompressed_len = 0 或剩余未处理长度
```

### 阶段五：继续 Decode

下一个 token 仍然应使用逻辑位置 1,064，而不是位置 600。

这正是 `rope_pos` 独立存在的原因。

---

## 18. 与上一轮 `llm_engine.py` 改动的联动关系

`sequence.py` 的新增字段不是孤立存在的，它们与 `llm_engine.py` 中的改动构成一条完整链路：

```text
Sequence 保存压缩相关状态
        ↓
ModelRunner 执行推理并可能触发压缩
        ↓
返回 token_ids + compression_events
        ↓
LLMEngine.step() 拆分返回值
        ↓
Scheduler.postprocess() 处理生成结果与压缩事件
        ↓
更新 Sequence 的缓存长度、Block Table 和压缩进度
```

对应关系如下：

| `sequence.py` 改动 | `llm_engine.py` 改动 | 作用 |
|---|---|---|
| 新增 `seq_id` 序列化 | 传递 `compression_events` | 将事件映射回具体请求 |
| 新增 `tail_uncompressed_len` | ModelRunner 可能触发压缩 | 判断和记录压缩周期 |
| 新增 `rope_pos` | 压缩后继续 Decode | 保持位置编码连续 |
| 新增 `generated_completion_tokens` | Scheduler 后处理 | 保持停止条件独立 |
| 删除 `num_scheduled_tokens` | Prefill 统计改用 `len(seq)` | 简化原版分块调度状态 |
| 完整序列化 `token_ids` | Worker 返回压缩信息 | 为压缩决策和状态恢复提供上下文 |

因此，nano-kvLLM 的整体设计不是在 Attention 层单独删除 KV，而是让压缩事件沿着：

```text
模型执行层 → 引擎层 → 调度层 → Sequence 状态层
```

完整传播。

---

## 19. 值得注意的潜在问题与优化点

### 19.1 `generated_completion_tokens` 与现有属性重复

当前它与：

```python
num_tokens - num_prompt_tokens
```

理论上相等。

建议明确一个权威字段，并在调试模式下加入一致性检查：

```python
assert self.generated_completion_tokens == self.num_completion_tokens
```

压缩实现若需要二者不同，应在注释中明确语义。

### 19.2 完整 `token_ids` 的 IPC 成本

每个 Decode step 发送完整列表，长上下文和高并发下可能成为 CPU 瓶颈。

建议后续 Benchmark 不只测 GPU 显存，还测：

- Decode TPOT；
- CPU 使用率；
- 进程间通信耗时；
- 序列长度增长时的吞吐下降；
- 压缩开启与关闭的端到端性能差异。

### 19.3 序列化字段不完整

新 dict 没有保存：

```text
status
temperature
max_tokens
ignore_eos
```

这与原版类似，可能是因为 Worker 只需要模型执行相关字段。

但如果反序列化后的 Sequence 被用于停止判断或完整调度，这些字段缺失会产生问题。应明确主进程对象与 Worker 副本的职责边界。

### 19.4 缺少协议版本

建议加入：

```python
"state_version": 1
```

这样后续增加字段或修改语义时，可以明确处理不同版本，而不是依赖大量默认值猜测。

### 19.5 `last_token = 0` 可能掩盖错误

必需状态缺失时更适合抛出异常，而不是生成一个可能无效的 token。

### 19.6 `num_cached_blocks` 对尾块的定义需明确

使用向下取整可能只表示完整 Block 数。如果缓存实际占用还包括部分尾块，调用端不能直接用它进行全部显存释放。

### 19.7 固定 `block_size = 256` 的一致性风险

当前 `Sequence` 中的 Block 大小仍是类变量：

```python
block_size = 256
```

而上一轮对比中，nano-kvLLM 的 `LLMEngine` 不再像原版一样显式执行：

```python
Sequence.block_size = config.kvcache_block_size
```

这意味着必须进一步确认：

- nano-kvLLM 是否始终强制使用 256；
- 是否在其他模块设置 `Sequence.block_size`；
- Config 中的 Block 大小是否也固定为 256；
- Sequence、Block Manager 与 KV Cache 张量的 Block 大小是否一致。

如果这些位置不一致，`num_blocks`、`num_cached_blocks`、`last_block_num_tokens` 和 `block()` 都可能计算错误。这是后续阅读 `config.py`、`block_manager.py` 和 `scheduler.py` 时应重点核查的问题。

---

## 20. 哪些改动是真正为了 KV Cache 压缩

### 20.1 核心压缩改动

以下改动直接服务于压缩正确性：

- `rope_pos`
- `tail_uncompressed_len`
- `generated_completion_tokens`
- `num_cached_blocks`
- `seq_id` 跨进程传递
- 完整压缩状态的序列化
- `append_token()` 同步更新压缩相关状态

### 20.2 配套架构改动

以下改动主要是为了适配新的调用链：

- tuple 序列化改为 dict；
- 更灵活的 `__setstate__()`；
- 完整传输 `token_ids`；
- 删除 `is_prefill`。

### 20.3 调度简化或行为变化

以下改动不能简单归类为压缩算法本身：

- 删除 `num_scheduled_tokens`；
- Prefill 阶段信息不再保存在单条 Sequence 中。

它们更像是 nano-kvLLM 为实现压缩实验而对原版调度模型做出的简化或重构。

---

## 21. 初学者应如何理解这个文件

可以把原版 `Sequence` 想成一张普通快递单：

```text
包裹是谁的？
现在运到哪了？
占用了哪个仓位？
```

nano-kvLLM 的 `Sequence` 则像一张经过“仓库压缩存储”改造后的快递单：

```text
包裹原本经过了多少站？              → rope_pos
总共新增了多少货物？                 → generated_completion_tokens
仓库目前实际保留多少货物？           → num_cached_tokens
最近又积累多少未整理货物？           → tail_uncompressed_len
货物现在放在哪些仓位？               → block_table
本次整理结果属于哪张订单？           → seq_id
```

压缩可以减少仓库占用，但不能篡改包裹真实经过的路线和已经产生的历史。

---

## 22. 最终总结

nano-kvLLM 对 `sequence.py` 的核心改造，是建立“逻辑序列状态”和“物理 KV Cache 状态”相互分离但能够协同更新的状态模型。

最关键的三项新增状态是：

```text
generated_completion_tokens
rope_pos
tail_uncompressed_len
```

它们分别保证：

1. KV Cache 压缩后，生成长度与停止条件仍然正确；
2. KV Cache 压缩后，RoPE 位置不会被错误重置；
3. 系统能够按周期识别尚未压缩的新生成尾部。

同时，nano-kvLLM 将序列化协议从固定 tuple 改成可扩展 dict，并传递完整 token 历史和 `seq_id`，使压缩事件可以跨进程传播并准确映射回具体请求。

从整个推理系统看，这个文件承担的是：

```text
请求逻辑历史
    +
生成进度
    +
位置编码进度
    +
物理缓存映射
    +
压缩周期状态
```

它本身不执行 KV Cache 压缩算法，但为 Scheduler、ModelRunner 和压缩模块提供了正确执行压缩所必需的状态基础。

后续最值得继续对比的文件是：

```text
scheduler.py
block_manager.py
model_runner.py
attention.py
config.py
```

阅读这些文件时，应重点追踪：

```text
tail_uncompressed_len 在哪里触发和重置？
rope_pos 在哪里传入 RoPE？
compression_events 如何修改 block_table？
num_cached_tokens 在压缩后如何更新？
被删除的 KV Block 如何归还空闲池？
```

只有把这些调用链连起来，才能完整理解 nano-kvLLM 的 KV Cache 压缩机制。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-kvllm|模块-nano-kvllm]]

%% 项目关联导航：结束 %%
