# nano-vLLM 与 nano-kvLLM：`attention.py` 源码对比分析

## 1. 对比对象与核心结论

本次对比的两个文件分别是：

- 原版 nano-vLLM 的 `attention.py`
- nano-kvLLM 的 `attention.py`

`Attention` 是 KV Cache 压缩真正落到 GPU 数据层的位置。

前面已经分析过：

```text
Sequence        保存压缩状态
ModelRunner     判断何时压缩、选择哪些请求
Scheduler       接收压缩事件并更新请求状态
BlockManager    回收压缩后多余的物理 Block
```

而本文件负责的是：

```text
把当前 token 的 K/V 写入缓存
        ↓
对选中的请求执行 KV Cache Compact/Compression
        ↓
让 FlashAttention 读取压缩后的 KV Cache
        ↓
产生本层 Attention 输出
```

nano-kvLLM 相比原版最核心的改造，是在 Decode Attention 路径中插入：

```python
MyCompressCompact(...)
```

并将执行顺序设计为：

```text
store_kvcache()
→ MyCompressCompact()
→ flash_attn_with_kvcache()
```

这说明压缩不是在生成结束后离线处理，而是在当前 Decode Step 内：

1. 先写入当前 token 的 K/V；
2. 再压缩历史 KV Cache；
3. 最后使用压缩后的 KV 计算本轮 Attention。

因此，`attention.py` 是整个 nano-kvLLM 项目中最接近“压缩算法本体”的核心接入点。

但仅从本文件看，`MyCompressCompact` 必须同时满足多个严格条件：

- 所有 Transformer 层必须采用一致的 token 保留布局；
- 压缩后必须更新当前 Attention 使用的有效上下文长度；
- 压缩后的 K/V 必须位于 `block_table` 能访问的前部槽位；
- 只有一个合适位置可以生成全局 `compression_events`；
- Tensor Parallel 各 Rank 必须得到一致的压缩选择；
- CUDA Graph 压缩步必须正确退回 eager，或压缩代码必须 Graph-safe。

---

# 2. 整体变化概览

| 改动方向 | 原版 nano-vLLM | nano-kvLLM | 工程意义 |
|---|---|---|---|
| 包命名空间 | `nanovllm` | `nanokvllm` | 使用独立改造工程 |
| 新增压缩模块 | 无 | `MyCompressCompact` | 在 Attention 层执行实际 KV Compact |
| Attention 构造参数 | Heads、Head Dim、Scale、KV Heads | 额外接收完整配置与层数 | 让每层获得压缩参数 |
| 压缩开关 | 无 | `kv_compress_enabled` | 支持压缩开关和基线对比 |
| 压缩参数 | 无 | Window Blocks、Keep Blocks、Extra Tokens | 控制压缩窗口和保留规模 |
| Forward 参数 | `q, k, v` | `q, k, v, Layer` | 向压缩函数传递层编号 |
| KV 写入 | 先写当前 token KV | 保持不变 | 当前 token 先进入 Cache |
| 压缩时机 | 无 | 写入 KV 后、Attention 前 | 本轮直接使用压缩后缓存 |
| 压缩触发条件 | 无 | Decode、启用压缩、周期步、请求被选中 | 避免 Prefill 和普通 Decode 执行压缩 |
| Prefill Attention | Varlen FlashAttention | 基本不变 | Prefill 路径不参与动态压缩 |
| Decode Attention | `flash_attn_with_kvcache` | 压缩后调用同一算子 | 保留原有高性能 KV Cache Attention |
| 事件与长度更新 | 无 | 依赖 `MyCompressCompact` 修改 Context | 把算法结果送回上层控制链 |
| 额外导入 | 无 | `os` | 当前未使用，属于残留 |

本文件的总体设计可概括为：

> nano-kvLLM 没有替换原版 FlashAttention，而是在 FlashAttention 之前对其即将读取的 KV Cache 做原地压缩，从而最大程度复用原有推理路径。

---

# 3. 原版 Attention 的完整职责

原版 `Attention.forward()` 的流程非常简洁：

```text
获取本轮 Context
        ↓
将新计算出的 K/V 写入 Paged KV Cache
        ↓
如果是 Prefill：
    使用 flash_attn_varlen_func
如果是 Decode：
    使用 flash_attn_with_kvcache
        ↓
返回 Attention 输出
```

## 3.1 Prefill 路径

Prefill 使用：

```python
flash_attn_varlen_func(...)
```

支持：

- 多条不同长度 Prompt 的变长 Batch；
- Causal Attention；
- Prefix Cache；
- Block Table。

如果存在 Prefix Cache：

```python
k, v = k_cache, v_cache
```

然后通过：

```python
block_table=context.block_tables
```

让 FlashAttention 从 Paged KV Cache 中读取已经命中的前缀。

## 3.2 Decode 路径

Decode 使用：

```python
flash_attn_with_kvcache(...)
```

输入包括：

- 当前 Query；
- 全局 K Cache；
- 全局 V Cache；
- 每条请求的 `cache_seqlens`；
- 每条请求的 `block_table`。

它只为每条请求处理一个或少量新 Query，同时从 KV Cache 读取全部历史。

## 3.3 `store_kvcache()`

无论 Prefill 还是 Decode，Attention 都先通过 Triton Kernel：

```python
store_kvcache(...)
```

将本轮产生的 K/V 写到由 `slot_mapping` 指定的物理缓存位置。

因此原版顺序是：

```text
写入新 KV
→ 用包含新 KV 的缓存计算 Attention
```

nano-kvLLM 保留了这一基本顺序，并在二者中间插入压缩。

---

# 4. 详细改动一：引入 `MyCompressCompact`

nano-kvLLM 新增：

```python
from nanokvllm.layers.compress_utils import MyCompressCompact
```

这是本文件最关键的新增依赖。

从调用参数可以推断，`MyCompressCompact` 至少需要负责：

```text
读取当前 Query
访问当前层 K/V Cache
识别当前层编号
读取压缩配置
读取全局 Context
对选中的 Sequence 执行 KV 选择和紧凑搬移
记录压缩后的长度或事件
```

它不是简单返回一个压缩后的 Tensor，而是接收原始缓存对象：

```python
k_cache
v_cache
```

这意味着它很可能对全局 KV Cache 做原地修改。

这种设计的优点是：

- 不需要替换 FlashAttention 接口；
- 不需要为压缩后 KV 额外分配长期缓存；
- 后续 `flash_attn_with_kvcache()` 可以继续读取原缓存对象；
- Scheduler 只需在模型执行后更新 Block 元数据。

---

# 5. 详细改动二：Attention 构造函数接收压缩配置

原版构造函数：

```python
Attention(
    num_heads,
    head_dim,
    scale,
    num_kv_heads,
)
```

nano-kvLLM 增加：

```python
vllm_config,
num_layers
```

并保存：

```python
self.kv_compress_enabled
self.num_layers
self.kv_compress_window_blocks
self.kv_compress_keep_blocks
self.kv_compress_keep_extra_tokens
```

## 5.1 `kv_compress_enabled`

控制本层是否启用压缩逻辑。

关闭时，Attention 应尽量退化为原版路径：

```text
store KV
→ FlashAttention
```

这便于进行基线实验。

## 5.2 `kv_compress_window_blocks`

表示压缩观察或处理窗口的 Block 数。

在 ModelRunner 中：

```text
window_tokens = window_blocks × block_size
```

用于判断一条请求是否积累了足够长的未压缩尾部。

在 Attention 中，它可能决定：

- 从多少个最近 Block 中收集 Query/Key 信息；
- 哪一段 KV 参与压缩；
- 哪一段近期窗口受到保护。

## 5.3 `kv_compress_keep_blocks`

表示压缩后固定保留多少个完整 Block，或算法中的目标保留块数。

具体语义仍需结合 `compress_utils.py` 判断，可能是：

- 旧历史区域保留 Block 数；
- 每次压缩目标 Block 数；
- 重要 token 压缩后占用的完整块数。

## 5.4 `kv_compress_keep_extra_tokens`

表示除完整保留 Blocks 外，再保留多少额外 token。

它通常用于：

- 保留最近不满一个 Block 的尾部；
- 保留局部窗口；
- 避免压缩结果必须严格对齐 Block；
- 提高模型质量。

## 5.5 `num_layers`

压缩函数需要知道模型总层数，可能用于：

- 判断当前是否为最后一层；
- 只在最后一层生成一次全局事件；
- 管理所有层都完成压缩后的统一状态；
- 分配跨层统计 Buffer；
- 保证同一 Sequence 的压缩索引在所有层一致。

这说明压缩不是单层局部操作，而是需要跨层协调。

---

# 6. 详细改动三：Forward 新增层编号参数

原版：

```python
forward(self, q, k, v)
```

nano-kvLLM：

```python
forward(self, q, k, v, Layer)
```

并传给：

```python
layer_id=Layer
```

## 6.1 为什么压缩需要层编号

每层都有独立 K/V Cache：

```text
Layer 0 K/V
Layer 1 K/V
...
Layer N-1 K/V
```

压缩时所有层都要对各自缓存执行 Compact。

层编号可以用于：

- 访问对应层的压缩统计；
- 使用跨层共享保留索引；
- 在第一层初始化压缩；
- 在最后一层生成事件；
- 判断所有层是否处理完成；
- 调试每层压缩状态。

## 6.2 命名问题

参数使用大写：

```python
Layer
```

不符合常见 Python 命名习惯。

更清晰的命名应为：

```python
layer_id
```

这样也可以避免与类名或模块概念混淆。

---

# 7. 详细改动四：压缩触发条件

nano-kvLLM 只有在以下条件全部成立时才调用压缩：

```python
not context.is_prefill
and self.kv_compress_enabled
and context.is_compress_step
and context.compress_selected_batch_indices
```

可以拆成四层保护。

## 7.1 只在 Decode 压缩

```python
not context.is_prefill
```

说明当前实现不压缩 Prompt Prefill 阶段。

这符合项目重点：

- Prefill 一次性建立完整 Prompt KV；
- Decode 过程中 KV 持续增长；
- 动态压缩主要解决长生成阶段的缓存膨胀。

## 7.2 配置启用

```python
self.kv_compress_enabled
```

用于基线对比和关闭算法。

## 7.3 当前为周期压缩步

```python
context.is_compress_step
```

由 ModelRunner 根据全局 Decode Step 计数器设置。

## 7.4 Batch 中至少有一个被选中请求

```python
context.compress_selected_batch_indices
```

空列表时不执行压缩函数。

这避免虽然到了周期步，但没有任何请求满足窗口条件时仍启动压缩逻辑。

---

# 8. 最关键的执行顺序：Store → Compress → Attend

nano-kvLLM 的顺序是：

```python
store_kvcache(...)
MyCompressCompact(...)
flash_attn_with_kvcache(...)
```

这是整个压缩设计中最重要的执行语义。

---

## 8.1 第一步：写入当前 token 的 K/V

Decode 时输入的 `k` 和 `v` 对应当前 `last_token`。

`store_kvcache()` 将其写入：

```text
slot_mapping 指定的当前尾部位置
```

此时缓存包含：

```text
之前保留的 KV
+
压缩后新增长的尾部 KV
+
当前 token 的 KV
```

---

## 8.2 第二步：执行压缩

`MyCompressCompact` 可以让当前 token 参与：

- Query 重要性计算；
- 新旧 KV 的选择；
- 查询窗口更新；
- 尾部保护；
- 压缩触发后的 Compact。

如果压缩在 Store 之前执行，那么当前 token 的 KV 不在缓存中，后续还需要处理新的写入位置和长度，逻辑更复杂。

---

## 8.3 第三步：读取压缩后的 KV

之后调用：

```python
flash_attn_with_kvcache(...)
```

理论上应直接使用压缩后的：

- `k_cache`；
- `v_cache`；
- `context.context_lens`；
- `context.block_tables`。

因此本轮 Attention 输出就已经基于压缩后的历史，而不是下一轮才生效。

---

## 8.4 这一顺序的收益

- 压缩立即降低本轮 Attention 读取长度；
- 当前 token 可以参与压缩决策；
- 不需要在两个 Decode Step 之间额外执行压缩；
- 可以复用同一 Attention Forward 调用；
- 压缩后不需要重新运行模型。

---

# 9. `MyCompressCompact` 必须修改哪些状态

由于后续 FlashAttention 仍使用：

```python
cache_seqlens=context.context_lens
block_table=context.block_tables
```

因此压缩函数不能只移动 K/V Tensor，还必须保证相关元数据同步。

至少需要处理以下内容。

## 9.1 K Cache

被保留的 Key 必须被紧凑搬移到 FlashAttention 可访问的有效前部槽位。

## 9.2 V Cache

Value 必须使用与 Key 完全相同的保留索引和搬移位置。

## 9.3 Context Length

压缩后，当前被压缩请求的：

```python
context.context_lens[batch_index]
```

必须变成新的有效长度。

否则 FlashAttention 仍会读取压缩前长度，可能访问：

- 已经失效的 KV；
- 重复或残留数据；
- 应被释放的尾部区域。

## 9.4 Block Table

如果压缩只在原有前部 Blocks 内做 Compact，且物理 Block ID 不变，那么本轮 Attention 可以继续使用原 Block Table。

如果压缩会改变物理 Block 顺序或映射，则必须同步修改：

```python
context.block_tables
```

## 9.5 Compression Events

压缩函数还必须记录：

```text
batch_index
seq_id
new_context_len
keep_blocks
tail_uncompressed_len_after
```

供 ModelRunner 在 Forward 后收集。

---

# 10. 为什么压缩后长度必须在本层 Attention 前更新

当前代码调用顺序为：

```text
MyCompressCompact
→ flash_attn_with_kvcache
```

所以：

```python
context.context_lens
```

必须在 `MyCompressCompact` 返回前已经更新。

不能只生成事件，等 Scheduler 在整个模型运行后再更新长度。

原因是 Scheduler 的更新发生在：

```text
所有 Transformer 层都完成 Forward
→ Logits 计算
→ Sampling
→ ModelRunner 返回
```

如果 Attention 本层仍使用旧长度：

```text
物理 KV 已经 Compact
但 FlashAttention 仍按旧长度读取
```

会导致错误。

因此存在两个不同层次的长度更新：

```text
GPU 本轮临时 Context Length
→ 必须在 Attention 内立即更新

主进程 Sequence.num_tokens
→ 模型运行结束后由 Scheduler 更新
```

二者必须得到相同的 `new_context_len`。

---

# 11. 跨层压缩的一致性问题

每个 Transformer 层都会调用自己的 `Attention.forward()`，也就会调用：

```python
MyCompressCompact(...)
```

这意味着同一个 Decode Step 中：

```text
Layer 0 压缩一次
Layer 1 压缩一次
...
Layer N-1 压缩一次
```

这不是重复压缩同一份缓存，因为每层拥有独立的 K/V Cache。

但所有层必须采用相同的逻辑保留布局。

---

## 11.1 为什么布局必须一致

系统只有一份：

```text
Sequence.block_table
Context.context_lens
compression_event.new_context_len
```

如果不同层保留不同数量的 token，单一长度无法描述。

更严重的是，如果不同层把不同逻辑 token 搬移到同一个紧凑位置：

```text
Layer 0 槽位 100 → 原 token 20
Layer 1 槽位 100 → 原 token 50
```

模型各层看到的历史序列语义不一致。

这会破坏 Transformer 层间表示传递。

---

## 11.2 可能的正确实现方式

### 方式一：统一索引

先由某个共享管理器计算：

```text
每条 Sequence 要保留哪些逻辑位置
```

所有层使用同一组索引搬移自己的 K/V。

### 方式二：固定结构压缩

如果压缩策略只是：

```text
保留固定前 N Blocks
+
最近窗口
```

则所有层天然使用相同布局。

### 方式三：跨层聚合重要性

先聚合多个层的 Attention/Query 统计，再生成一套全局保留索引。

---

## 11.3 不应采用的方式

每层根据自己的局部 Attention Score 独立选择不同 token，同时仍共用一个 Context Length 和 Block Table。

除非系统支持每层独立 Block Table，否则这种设计不可行。

---

# 12. `num_layers` 可能承担的协调作用

`MyCompressCompact` 接收：

```python
num_layers=self.num_layers
```

这很可能用于管理跨层生命周期，例如：

```text
Layer 0：
    初始化本轮压缩状态
    计算或读取统一保留索引

中间层：
    按统一索引 Compact 各自 K/V

最后一层：
    生成一次 compression_event
    完成本轮状态提交
```

如果每层都向：

```python
context.compression_events
```

追加相同事件，那么 Scheduler 前面实现的“按 batch_index 只保留最后一个事件”可以消除重复。

但更合理的设计仍是：

```text
只在最后一层产生全局事件
```

这样事件语义更清晰，也减少 Python 对象操作。

---

# 13. 当前 Query `q_current` 的作用

压缩函数接收：

```python
q_current=q
```

说明压缩策略很可能是 Query-aware 的。

可能的思路包括：

- 根据当前 Query 与历史 Key 的相关性选择重要 KV；
- 维护最近多个 Query 的窗口；
- 累积 token 重要性；
- 选择对近期生成最有帮助的历史 token；
- 保护近期局部窗口，同时压缩远端历史。

这种压缩不同于简单滑动窗口，因为它可能保留远处但重要的 KV。

## 13.1 Query Shape

原版 Decode 的 `q` 通常形状类似：

```text
[batch_size, num_query_heads, head_dim]
```

在调用 FlashAttention 时：

```python
q.unsqueeze(1)
```

变成单 token Query Length。

`MyCompressCompact` 必须正确区分：

- Batch 维度；
- Query Head；
- Tensor Parallel 下本 Rank 的 Heads；
- 被选择的 Batch Index。

---

# 14. Tensor Parallel 下的重要性选择问题

每个 Tensor Parallel Rank 只拥有部分 Attention Heads 或 KV Heads。

如果压缩函数根据本 Rank 的 `q` 和 `k_cache` 独立计算重要 token，不同 Rank 可能得到不同结果。

例如：

```text
Rank 0 认为 token 100 最重要
Rank 1 认为 token 300 最重要
```

如果两个 Rank 分别 Compact 不同 token：

- 各 GPU 的 KV Cache 逻辑布局不一致；
- 后续 Tensor Parallel 聚合失去统一语义；
- Scheduler 只有一份压缩事件，无法描述差异。

因此正确实现需要至少满足一种机制：

1. 在 Rank 间聚合 Importance Score；
2. Rank 0 计算保留索引并广播；
3. 使用与 Head 无关、所有 Rank 一致的固定压缩规则；
4. 对各 Rank Score 做确定性的 All-Reduce。

本文件没有显示跨 Rank 同步逻辑，因此必须在 `MyCompressCompact` 中继续核查。

---

# 15. 压缩函数与 CUDA Graph 的关系

`MyCompressCompact` 由普通 Python 条件控制：

```python
if context.is_compress_step
and context.compress_selected_batch_indices:
```

其中：

```text
compress_selected_batch_indices
```

是动态 Python List。

这通常不适合直接放入 CUDA Graph Replay。

原因包括：

- Graph 捕获时分支固定；
- Python List 不会作为 GPU Graph 输入变化；
- Compact 长度和选择请求数量动态变化；
- 可能有动态 Tensor 索引、搬移和事件创建；
- `context.compression_events` 是 Python 侧状态。

因此最自然的设计是：

```text
普通 Decode Step → CUDA Graph
压缩 Decode Step → Eager
```

前一轮 `model_runner.py` 已经指出，当前 `compress_any` 的设置顺序可能导致压缩步没有正确切回 eager。

`attention.py` 的新增动态 Python 分支进一步说明：

> 如果 ModelRunner 仍使用 CUDA Graph Replay，这里的压缩分支很可能不会按每轮动态状态正常执行。

---

# 16. Prefill 路径为何保持不变

nano-kvLLM 没有在：

```python
if context.is_prefill:
```

分支中执行压缩。

原因通常包括：

- Prompt KV 只建立一次；
- Prefill 计算以大矩阵为主，压缩逻辑更复杂；
- Prompt token 可能全部重要；
- 项目主要验证 Decode 增长阶段的缓存控制；
- Prefix Cache 与 Prefill 压缩容易产生冲突；
- 压缩前需要 Query Window 或历史统计，Prefill 阶段不一定具备。

因此当前系统的生命周期更像：

```text
完整 Prompt Prefill
→ 建立原始 KV Cache
→ Decode 一段时间
→ 周期性压缩
→ 继续 Decode
```

---

# 17. Prefix Cache 与压缩 Cache 的边界

Prefill 分支仍保留原版 Prefix Cache：

```python
if context.block_tables is not None:
    k, v = k_cache, v_cache
```

这意味着 nano-kvLLM 仍尝试兼容 Prefix Cache。

但压缩后的 KV Cache 可能不再对应连续原始 token 前缀。

如果压缩算法会：

- 选择离散 token；
- 重排 KV；
- 将历史 token Compact 到前部；
- 改变 Block 内 token 内容；

那么压缩后的 Block 不应继续使用普通 Prefix Cache 哈希语义。

前一轮 `block_manager.py` 已经发现，当前哈希失效处理可能不完整。

从 Attention 层看，需要明确：

```text
MyCompressCompact 修改了哪些物理 Blocks？
```

凡是被改写的 Block，都不应继续被当作原始连续前缀缓存共享，除非重新建立符合新语义的缓存键。

---

# 18. `store_kvcache()` 本身没有改变

Triton Kernel 完全沿用原版。

它对每个输入 token：

```text
读取 slot_mapping
→ 读取对应 K/V 向量
→ 写入扁平化 KV Cache 槽位
```

这说明 nano-kvLLM 没有改变“新 token 如何进入 Paged KV Cache”的基本方式。

压缩只发生在写入之后。

这种最小侵入式设计的优点是：

- 原版 Paged KV 写入逻辑可直接复用；
- Prefill 和普通 Decode 路径变化较小；
- 压缩关闭时容易恢复原性能；
- 压缩算法与存储 Kernel 解耦。

---

# 19. 压缩后的物理布局必须满足 FlashAttention 接口

后续仍调用：

```python
flash_attn_with_kvcache(
    q.unsqueeze(1),
    k_cache,
    v_cache,
    cache_seqlens=context.context_lens,
    block_table=context.block_tables,
)
```

这意味着压缩后必须满足 FlashAttention 的 Paged KV Cache 约束。

## 19.1 有效 token 必须紧凑

对于一条请求，前：

```text
context_len
```

个逻辑缓存位置必须都对应有效 KV。

不能出现：

```text
有效、空洞、有效、空洞
```

除非 FlashAttention 有额外的稀疏索引接口，而当前调用没有。

## 19.2 Block Table 必须有效

每个逻辑 Block 都必须映射到当前仍被该 Sequence 持有的物理 Block。

## 19.3 最后一块允许不满

由 `cache_seqlens` 指定有效长度，FlashAttention 不会读取最后 Block 中超出长度的槽位。

## 19.4 K/V 布局必须一致

Key 和 Value 必须使用完全相同的 Compact 索引。

---

# 20. 当前 Attention 调用中没有显式传递 RoPE 位置

`attention.py` 只接收已经经过模型投影和 RoPE 处理后的：

```text
q
k
v
```

RoPE 位置由前面的模型层使用 `positions` 处理。

因此本文件不直接使用：

```python
seq.rope_pos
```

但它依赖 ModelRunner 已经正确构造逻辑位置。

压缩后保留的旧 Key 已经包含原位置的 RoPE 信息。

只要 Compact 是移动 K/V 向量，而不是重新对其按紧凑位置编码，就可以保留原始逻辑位置信息。

这是重要原则：

> KV 被搬到新的物理槽位，不代表其 RoPE 逻辑位置应该改变。

物理索引和位置编码必须分离。

---

# 21. 压缩后 Key 的 RoPE 信息如何保持

假设原 token 位于逻辑位置 1000。

它的 Key 已经经过：

```text
RoPE(position=1000)
```

压缩后该 Key 被搬到紧凑缓存的物理位置 100。

正确行为是：

```text
物理槽位变成 100
Key 向量仍保留 position=1000 的旋转结果
```

不能重新将其解释为逻辑位置 100。

当前方案直接搬移 K Cache Tensor，理论上可以保持这一点。

新 Query 则由：

```python
seq.rope_pos
```

使用真实逻辑位置编码。

因此 Query-Key Attention 仍反映原始逻辑位置关系。

---

# 22. Compression Event 应在哪一层生成

`MyCompressCompact` 在每层都会执行，但全局 Scheduler 只需要一份事件。

事件描述的是：

```text
该 Sequence 压缩后的统一上下文长度和 Block 数
```

而不是某一层局部 K/V 数值。

合理策略包括：

## 22.1 第一层决定、最后一层提交

- 第一层计算统一保留索引；
- 所有层执行 Compact；
- 最后一层确认完成并写入事件。

## 22.2 每层写相同事件，上层去重

Scheduler 当前会按 Batch Index 保留最后一个事件，因此可以容忍重复。

但这会：

- 增加 Context 事件数量；
- 隐藏层间结果不一致；
- 让“最后事件”覆盖前面错误。

## 22.3 统一管理器生成一次事件

由共享压缩管理器追踪所有层完成状态，最后写一次。

从工程可维护性看，第三种或第一种更好。

---

# 23. `context.context_lens` 的跨层修改必须谨慎

假设 Layer 0 压缩后将：

```text
context_len: 2048 → 1024
```

Layer 1 进入 `MyCompressCompact` 时读取到的 Context Length 已经是 1024。

如果 Layer 1 的压缩逻辑仍需要知道压缩前长度 2048，就必须使用 ModelRunner 保存的：

```python
context.compress_base_context_lens
```

而不能使用已经修改后的 `context.context_lens`。

这正是 ModelRunner 在压缩步提前 Clone 基础长度的原因。

正确的跨层状态可能是：

```text
compress_base_context_lens
→ 所有层共同使用的压缩前长度

context.context_lens
→ 第一层压缩后更新，用于本轮 FlashAttention
```

每层 Compact 都必须依据同一基础长度和同一保留索引，避免后续层重复压缩已经缩短的缓存。

---

# 24. 当前代码中的潜在问题

## 24.1 `MyCompressCompact` 使用函数名大写

Python 中大写名称通常表示类。

如果它是普通函数，更规范的命名是：

```python
my_compress_compact
```

如果它是可调用类，则当前直接调用可能是实例化，而不是执行实际压缩，需要检查实现。

---

## 24.2 `Layer` 参数命名不规范

建议改为：

```python
layer_id
```

---

## 24.3 `os` 导入未使用

```python
import os
```

当前文件没有使用，属于残留。

---

## 24.4 未显式验证压缩后长度

Attention 在调用 FlashAttention 前没有检查：

```text
context_lens <= block_table capacity
context_lens > 0
```

应由压缩函数保证，并建议加入调试断言。

---

## 24.5 未显式验证所有层使用同一压缩索引

本文件传递了层编号和总层数，但一致性完全依赖压缩工具实现。

---

## 24.6 CUDA Graph 路径风险

动态 Python List 和事件逻辑不适合普通 Graph Replay。

---

## 24.7 压缩函数每层都执行

需要确认：

- 统一保留索引只计算一次；
- 每层只搬移自己的 K/V；
- 事件只生成一次或严格一致；
- `context_lens` 不会被重复缩短。

---

## 24.8 Tensor Parallel 缺少显式同步

若重要性选择依赖本地 Head，必须在压缩工具中同步。

---

## 24.9 压缩失败后的回滚

当前没有异常处理。

若某层 Compact 失败，而前面层已完成搬移：

```text
模型各层 KV 布局可能部分更新
```

生产级实现需要：

- 预先验证；
- 原子式状态提交；
- 或失败时终止请求，而不是继续运行。

---

# 25. 当前设计的优势

尽管存在需要核查的细节，该设计有明显工程优点。

## 25.1 对原版 Attention 侵入较小

只在原路径中插入一个压缩函数。

## 25.2 保留 FlashAttention

没有重新实现完整 Attention Kernel。

## 25.3 压缩关闭时容易回退

条件不满足时基本走原路径。

## 25.4 压缩结果本轮立即生效

压缩后马上进入 FlashAttention。

## 25.5 与 Paged KV Cache 兼容

只要 Compact 后布局满足 Block Table 约束，就能继续复用现有缓存系统。

## 25.6 支持 Query-aware 压缩

当前 Query 被直接传给压缩算法。

---

# 26. 与前面五个文件的完整联动

现在可以把六个核心文件串联起来。

## 26.1 ModelRunner 选择请求

```text
到达压缩周期
→ 检查 tail_uncompressed_len
→ 选出 Batch Index 和 Seq ID
→ 写入 Context
```

## 26.2 Attention 写入当前 KV

```text
store_kvcache()
```

## 26.3 Attention 执行 Compact

```text
MyCompressCompact()
→ 选择保留 KV
→ 搬移 K/V
→ 更新 Context Length
→ 记录 Compression Event
```

## 26.4 Attention 使用压缩缓存

```text
flash_attn_with_kvcache()
```

## 26.5 ModelRunner 收集事件

```text
Context.compression_events
→ 返回给 LLMEngine
```

## 26.6 Scheduler 提交事件

```text
new_context_len
keep_blocks
tail_uncompressed_len_after
```

## 26.7 BlockManager 回收尾部 Block

```text
truncate_blocks()
→ free_block_ids 增加
```

完整链路为：

```text
Sequence 的未压缩尾部增长
        ↓
ModelRunner 周期性选择压缩请求
        ↓
Attention 写入当前 token KV
        ↓
MyCompressCompact 原地紧凑 K/V
        ↓
FlashAttention 读取压缩后缓存
        ↓
事件返回 Scheduler
        ↓
BlockManager 释放尾部 Blocks
```

---

# 27. 一个完整的压缩示例

假设：

```text
block_size = 256
压缩前有效 Context Length = 2048
Block 数量 = 8
逻辑 RoPE 位置 = 4095
```

压缩配置：

```text
window_blocks = 4
keep_blocks = 3
keep_extra_tokens = 100
```

## 27.1 当前 token 写入

当前 token K/V 写入第 2048 个逻辑缓存位置对应槽位。

缓存临时长度可能变为：

```text
2049
```

## 27.2 Compact

压缩算法可能决定保留：

```text
3 个完整块 + 100 个额外 token
```

得到：

```text
new_context_len = 868
keep_blocks = ceil(868 / 256) = 4
```

## 27.3 物理搬移

选中的 868 个 K/V 被紧凑写入前 4 个 Blocks。

## 27.4 更新 Context

```text
context_lens = 868
```

## 27.5 FlashAttention

当前 Query 使用逻辑 RoPE 位置 4095，但只读取 868 个保留 KV。

## 27.6 事件

```text
new_context_len = 868
keep_blocks = 4
tail_uncompressed_len_after = 0
```

## 27.7 上层回收

Scheduler 和 BlockManager 释放原来第 4～7 号逻辑 Block 对应的物理 Blocks。

---

# 28. 哪些改动真正服务于 KV Cache 压缩

## 28.1 核心算法接入

- 引入 `MyCompressCompact`；
- 把压缩插入 KV 写入和 FlashAttention 之间；
- 将当前 Query 传入压缩算法；
- 传入 K/V Cache；
- 传入层编号和层数；
- 传入窗口、保留 Blocks 和额外 token 参数；
- 使用 Context 中的压缩请求选择。

## 28.2 正确性保障相关

- 只在 Decode 压缩；
- 所有层执行各自 K/V Compact；
- 本轮 Attention 理论上读取压缩后长度；
- 保留原始 RoPE 编码后的 K/V；
- 依赖统一 Context 保证跨层布局一致。

## 28.3 非核心变化

- `os` 导入；
- 参数命名风格；
- 包路径修改。

---

# 29. 初学者理解方式

可以把每层 Attention 的 KV Cache 想成一排仓库货架。

原版流程：

```text
把当前 token 的新货物放进货架
→ 从全部货架读取历史
→ 计算 Attention
```

nano-kvLLM 流程：

```text
把当前 token 的新货物放进货架
→ 根据当前 Query 挑选重要历史货物
→ 把重要货物紧凑搬到前面
→ 从整理后的短货架读取历史
→ 计算 Attention
```

每一层都有自己的仓库，但所有层必须使用同一份“保留货物清单”。

否则不同层会认为同一个紧凑位置对应不同历史 token，模型语义就会混乱。

---

# 30. 最终总结

nano-kvLLM 对 `attention.py` 的核心改造，是把 KV Cache 压缩真正插入 Attention 的数据执行路径：

```text
store_kvcache
→ MyCompressCompact
→ flash_attn_with_kvcache
```

这使压缩后的 KV Cache 能够在当前 Decode Step 立即被 Attention 使用。

该文件体现了 nano-kvLLM 的关键思想：

> 不替换原版 FlashAttention，而是在 FlashAttention 读取之前，将需要保留的 K/V 紧凑整理到兼容 Paged KV Cache 的布局中。

新增的 `MyCompressCompact` 必须完成：

```text
统一选择保留 token
搬移当前层 K/V
更新本轮 Context Length
维护跨层一致性
生成压缩事件
```

其中最关键的不变量是：

1. 所有层必须采用相同逻辑保留布局；
2. Key 和 Value 必须使用相同索引；
3. 压缩后有效 KV 必须在前部连续可访问；
4. `context.context_lens` 必须在本层 FlashAttention 前更新；
5. 被搬移的旧 Key 保留原 RoPE 语义；
6. Tensor Parallel 各 Rank 必须使用统一压缩选择；
7. 压缩步应正确退出普通 CUDA Graph，或使用专门的 Graph-safe 实现。

从目前已分析的文件看，完整系统已经形成：

```text
ModelRunner 决定何时压缩
        ↓
Attention 执行实际 KV Compact
        ↓
ModelRunner 收集压缩事件
        ↓
Scheduler 更新 Sequence
        ↓
BlockManager 释放物理 Blocks
```

下一步最值得继续分析的是：

```text
compress_utils.py
utils/context.py
models/qwen3.py
query_window_manager 相关文件
```

其中 `compress_utils.py` 是确认以下问题的关键：

- 实际如何选择重要 KV；
- 是否所有层共享同一保留索引；
- 如何原地搬移 K/V；
- 是否同步更新 `context_lens`；
- `compression_events` 如何创建；
- Tensor Parallel 如何保持一致；
- 压缩后哪些 Blocks 被改写；
- 是否会与 Prefix Cache 哈希产生冲突。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-kvllm|模块-nano-kvllm]]

%% 项目关联导航：结束 %%
