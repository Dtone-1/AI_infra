# nano-kvLLM 新增源码分析：`CompressMethod.py`

## 1. 文件定位与新增意义

`CompressMethod.py` 是 nano-kvLLM 新增的 **KV Cache 保留索引选择层**。

它不负责：

- 分配或释放 KV Blocks；
- 修改 Block Table；
- 搬移 K/V Tensor；
- 更新 Sequence；
- 调度压缩周期。

它只回答一个算法问题：

> 给定当前 Query 和一段历史 Key，哪些历史位置对当前生成更重要，应该在压缩后继续保留？

当前文件实现的函数是：

```python
SnapKV(Q, K, V, num_keep, window)
```

输出：

```text
每条 Sequence 应保留的位置索引 final_idx
```

随后 `compress_utils.py` 才会根据这些索引真正 Gather 和 Compact K/V。

因此两者的职责边界是：

```text
CompressMethod.py
决定“保留谁”

compress_utils.py
决定“从哪里取、搬到哪里、释放多少”
```

---

## 2. 算法总体思想

当前实现采用 Query-aware 的重要性评分。

直观过程：

```text
当前 Query
    ↓
与历史 Key 做点积
    ↓
Softmax 得到关注概率
    ↓
跨 Query Window 和 Attention Heads 聚合
    ↓
选出重要性最高的 num_keep 个位置
    ↓
再固定保留一个首部锚点和最近 window 个位置
```

最终保留集合：

```text
[首部锚点]
+
[Top-K 重要历史 token]
+
[最近 window 个 token]
```

最终索引数量：

```text
1 + num_keep + window
```

---

## 3. 输入输出张量

### 3.1 Query

```text
Q: [B, Hq, window, D]
```

- `B`：Batch 中待压缩请求数；
- `Hq`：Query Head 数；
- `window`：用于评分的最近 Query 数；
- `D`：Head Dimension。

在当前 nano-kvLLM 调用中：

```text
window = 1
```

即只使用当前 Decode Query。

### 3.2 Key

```text
K: [B, Hk, L, D]
```

- `Hk`：KV Head 数；
- `L`：待压缩窗口长度。

### 3.3 Value

```text
V: [B, Hk, L, D]
```

当前函数不使用 V，只为接口统一保留。

### 3.4 输出

```text
final_idx: [B, 1 + num_keep + window]
```

每行表示该请求在输入 K 窗口内需要保留的位置。

---

## 4. 为什么使用 Query–Key 相关性

Transformer Attention 的核心打分是：

```text
score = QKᵀ / √D
```

如果当前 Query 对某个历史 Key 给出较高注意力概率，可以认为该历史 KV 对当前生成更有价值。

所以该算法用当前或最近 Query 作为“探针”，估计历史 token 的重要性。

相比简单滑动窗口：

- 可以保留远处但重要的信息；
- 不必只保留最近 token；
- 对长上下文中的关键实体或指令更友好。

相比随机或固定间隔采样：

- 选择与当前生成状态相关；
- 更符合 Attention 的实际使用模式。

---

## 5. 跳过压缩的条件

```python
if L - window <= 0 or L <= num_keep + window:
    return False
```

含义：

1. 除去最近保护窗口后，没有可供选择的历史；
2. 当前总长度不大于目标保留数量，没有压缩空间。

这样可以防止：

- 对过短窗口执行 Top-K；
- `torch.topk()` 的 K 大于候选数；
- 压缩后长度不降反升。

需要注意，调用方当前还固定额外加入一个首部锚点，因此更严格的配置检查应在上层保证最终保留数小于输入长度。

---

## 6. 排除最近窗口后计算历史重要性

```python
K_cut = K[:, :, :-window, :]
```

最近 `window` 个 Key 不进入 Top-K 竞争，而是在最后无条件保留。

这体现“近期窗口保护”：

```text
较远历史：按重要性竞争
最近历史：直接保留
```

当前调用中 `window=1`，表示始终保留压缩窗口最后一个 Key。

另外，`compress_utils.py` 还会把完整压缩窗口之后的未满尾块全部保留，所以系统实际上有两层近期保护：

- SnapKV 保留窗口内最后一个位置；
- Compact 层保留窗口外的整个尾部残块。

---

## 7. 标准多头注意力路径

当：

```python
Hq == Hk
```

时，直接执行：

```python
attn_scores = Q @ K_cut.transpose(-1, -2) / sqrt(D)
```

结果形状：

```text
[B, Hq, query_window, L-window]
```

每个 Query Head 分别评价历史 Key。

---

## 8. GQA 路径

Qwen 等模型常使用 Grouped Query Attention：

```text
Query Head 数 Hq
>
KV Head 数 Hk
```

多个 Query Heads 共享同一个 KV Head。

代码要求：

```python
Hq % Hk == 0
```

并计算：

```text
group_size = Hq / Hk
```

然后把 Query 重新组织为：

```text
[B, Hk, group_size, window, D]
```

Key 扩展一个 Group 维度后进行批量矩阵乘法，最后恢复到：

```text
[B, Hq, window, L-window]
```

这一改造使同一个算法同时支持：

- MHA：`Hq == Hk`；
- GQA：`Hq > Hk` 且整除。

这是该文件对 nano-vLLM 所用 Qwen 模型非常关键的适配。

---

## 9. “Attention Sink”处理

代码执行：

```python
attn_scores[:, :, :, 0] = -inf
```

使候选区域第一个位置不会通过 Top-K 被再次选中。

随后又显式加入：

```python
bos_idx = 0
```

其目的不是删除该位置，而是：

```text
固定保留第一个位置
同时避免它占用 Top-K 的普通重要 token 名额
```

### 重要语义说明

在当前调用链中，K 不是整条序列，而是尾部压缩窗口。

因此索引 0 实际表示：

```text
压缩窗口的第一个 token
```

并不是真实的全局 BOS。

全局 BOS 位于压缩窗口之前的未动前缀中，本来就会保留。

所以变量名 `bos_idx` 更准确应改为：

```text
anchor_idx
或
window_first_idx
```

---

## 10. Softmax 与重要性聚合

### 10.1 Softmax

```python
attn_probs = softmax(attn_scores, dim=-1)
```

把每个 Query Head 对历史 Key 的分数转换成概率分布。

### 10.2 跨 Query Window 聚合

```python
key_importance = attn_probs.sum(dim=2)
```

如果使用多个近期 Query，则一个历史 Key 被多个 Query 持续关注时，重要性更高。

当前 `window=1`，这一求和没有实际聚合作用。

### 10.3 跨 Heads 聚合

```python
key_importance.sum(dim=1, keepdim=True)
```

最终得到：

```text
[B, 1, L-window]
```

即每条 Sequence 只生成一套 token 保留索引，供当前 Rank 的所有 KV Heads 共享。

优点：

- Compact 布局简单；
- 不需要每个 Head 单独的稀疏索引；
- 可以继续使用普通连续 KV Cache 和 FlashAttention。

代价：

- 某个只对少数 Head 重要的 token，可能被跨 Head 求和稀释；
- 无法做到 Head-wise 独立保留。

---

## 11. Top-K 选择与顺序恢复

```python
torch.topk(..., largest=True, sorted=False)
```

先选择重要性最高的 `num_keep` 个位置。

随后：

```python
torch.sort(idx_keep)
```

按原始 token 位置重新排序。

为什么不能按重要性顺序直接写入缓存？

因为 Attention 中 Key 的 RoPE 信息和逻辑时序仍对应原序列。虽然 K 已包含位置编码，但保持原有时间顺序：

- 更符合序列语义；
- 便于调试；
- 避免其他位置相关逻辑误解；
- 让压缩缓存布局稳定。

因此算法是：

```text
按重要性选
按原序列顺序存
```

---

## 12. 最终保留集合

```python
final_idx = cat([
    bos_idx,
    base_idx,
    tail_idx
])
```

三部分互不重复：

1. 首部锚点索引 0；
2. Top-K 从排除最近窗口后的区域选出，且位置 0 被设为负无穷；
3. `tail_idx` 来自最后 `window` 个位置，不参与 Top-K。

所以最终长度稳定为：

```text
1 + num_keep + window
```

当前 `compress_utils.py` 传入：

```text
num_keep = keep_tokens - 2
window   = 1
```

最终正好得到 `keep_tokens` 个位置。

---

## 13. 当前实现与“完整 SnapKV”概念的关系

从代码本身看，这是一种 **SnapKV 思路的简化实现**：

- 使用近期 Query 对历史 Key 评分；
- 保留高重要性历史；
- 保护近期窗口；
- 支持 GQA。

但当前实现没有体现更复杂的：

- 多步 Query Observation Window 管理；
- 池化或平滑；
- Head-wise 容量分配；
- 跨层共享策略；
- 专用 CUDA Kernel。

尤其当前调用只传入一个 Query：

```text
query window = 1
```

因此更准确的工程描述是：

> 基于当前 Decode Query 的 SnapKV-style Top-K 历史 KV 选择。

不宜仅凭函数名宣称完全复现某一论文或官方实现。

---

## 14. Value 为什么没有参与评分

函数签名保留 V，但实际只用 Q 和 K。

这是符合 Attention 评分机制的：

```text
QKᵀ 决定关注哪个位置
V 决定从该位置读取什么内容
```

选择位置时通常只需 Q–K 相关性。

V 会在 `compress_utils.py` 中根据同一 `keep_idx` 被搬移，确保 K/V 对齐。

若长期不使用 V，可以：

- 从函数签名删除；
- 或在注释中明确仅为统一压缩接口保留。

---

## 15. 算法复杂度

设：

```text
B  = 压缩请求数
Hq = Query Heads
L  = 压缩窗口长度
D  = Head Dimension
W  = Query Window
K  = num_keep
```

### 15.1 QK 打分

```text
O(B × Hq × W × L × D)
```

当前 `W=1`。

### 15.2 Softmax 和聚合

```text
O(B × Hq × W × L)
```

### 15.3 Top-K

随实现不同，近似与：

```text
B × L
```

和 K 相关。

因此窗口长度越大，重要性选择越贵。这也是上层只压缩固定尾部窗口，而不每次扫描整条上下文的原因。

---

## 16. 当前实现的优势

### 16.1 逻辑清晰

选择算法与缓存搬移完全解耦。

### 16.2 同时支持 MHA 和 GQA

适合 Qwen 的 KV Head 结构。

### 16.3 保留时间顺序

Top-K 后按索引排序。

### 16.4 固定保留锚点和近期窗口

避免全部容量只被高分远端 token 占用。

### 16.5 输出统一索引

容易与 Paged KV Cache Compact 对接。

### 16.6 可替换性强

未来可以在不修改 Scheduler 和 BlockManager 的情况下，把 `SnapKV()` 替换为其他选择策略，只要仍返回固定数量的位置索引。

---

## 17. 潜在问题一：只使用当前一个 Query，选择可能抖动

当前调用：

```text
window = 1
```

某个 token 是否被保留只取决于当前一步 Query。

这可能导致：

- 对瞬时 Query 过拟合；
- 不同 Decode Step 的重要位置波动；
- 忽略过去若干步持续重要的信息；
- 压缩质量对当前 token 敏感。

可优化为：

- 维护最近多个 Query；
- 对多步分数做累积或平均；
- 使用 Query Window Manager；
- 给长期重要 token 建立衰减统计。

---

## 18. 潜在问题二：跨 Head 求和可能偏向某些模式

当前所有 Query Heads 的概率直接求和。

可能的替代方式：

- 平均；
- 最大值；
- 分组聚合；
- 每个 KV Head 独立评分；
- 为不同 Head 分配部分容量。

当前求和简单稳定，但会让所有 Head 共用一套保留索引。

---

## 19. 潜在问题三：没有显式处理 Padding 或无效位置

传入 K 来自完整物理窗口，正常情况下全部有效。

如果上层错误地把无效 Slot Gather 进来，本函数没有 Mask，会把它们当作正常 Key 评分。

所以有效性完全依赖 `compress_utils.py` 正确构造窗口。

---

## 20. 潜在问题四：配置边界缺少验证

应显式保证：

```text
num_keep >= 0
window >= 1
L > num_keep + window
Hq % Hk == 0
```

当前只覆盖部分条件。

若上层配置：

```text
keep_tokens < 2
```

则传入的 `num_keep` 可能为负数。

---

## 21. 潜在问题五：`math` 和 `V` 未使用

```python
import math
```

未被使用。

`V` 参数也没有参与计算。

这些不会影响正确性，但说明代码仍有原型痕迹。

---

## 22. 潜在问题六：局部首 token 被称为 BOS

前面已经说明，这会造成对算法语义的误解。

建议重命名：

```python
anchor_idx = torch.zeros(...)
```

注释改为：

```text
固定保留压缩窗口首位置，避免其占用普通 Top-K 名额
```

---

## 23. 与 `compress_utils.py` 的组合关系

```text
CompressMethod.SnapKV
输入：
  当前 Query
  压缩窗口 K/V
输出：
  窗口内保留位置 keep_idx
        ↓
compress_utils.MyCompressCompact
  keep_idx → 物理 src_slots
  构造 dst_slots
  搬移 K/V
  更新 context_lens
  生成 compression_event
```

因此 `CompressMethod.py` 是算法策略层，`compress_utils.py` 是系统执行层。

这种分层很好，因为未来可以替换为：

- 固定滑动窗口；
- 随机或均匀采样；
- Attention Score 累积；
- Heavy-Hitter；
- Layer-wise 不同策略；

而不需要重写上层调度和 Block 回收协议。

---

## 24. 最终总结

`CompressMethod.py` 为 nano-kvLLM 提供了“哪些 KV 值得保留”的决策能力。

当前算法执行：

```text
QKᵀ / √D
→ Softmax
→ 跨 Query Window 聚合
→ 跨 Heads 聚合
→ Top-K
→ 固定加入窗口首锚点
→ 固定加入近期窗口
→ 按原位置排序后返回
```

它的工程意义是：

> 将 KV Cache 压缩从简单截断，升级为与当前生成状态相关的内容感知选择。

同时，这个文件保持了与系统层的清晰解耦：

- 不关心 Block ID；
- 不关心物理显存；
- 不关心 Scheduler；
- 不直接修改 K/V；
- 只输出保留索引。

当前实现仍属于简化、易验证的版本。后续改进重点包括：

- 使用多步 Query Window；
- 减少逐步选择抖动；
- 评估不同 Head 聚合策略；
- 补充配置断言；
- 明确局部锚点而非全局 BOS；
- 使用更高效的专用 Kernel；
- 对压缩率与模型质量进行系统 Benchmark。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-kvllm|模块-nano-kvllm]]

%% 项目关联导航：结束 %%
