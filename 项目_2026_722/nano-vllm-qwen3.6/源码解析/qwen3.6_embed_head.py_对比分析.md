# qwen3.6_embed_head.py_对比分析

> 对比对象：  
> 1. 原版 `nano-vLLM` 的 `embed_head.py`  
> 2. `nano-vllm-qwen3.6` 中对应的 `embed_head.py`
>
> 目标：从整体工程角度分析 qwen3.6 版本相比原版做了哪些修改、为什么要这样改，以及这些修改在 Qwen3.6 / 推理系统 / FP8 权重加载 / Tensor Parallel 采样路径中的作用。

---

## 1. 文件整体定位

`embed_head.py` 负责模型 forward 的开头和结尾，主要定义两个模块：

```text
VocabParallelEmbedding
ParallelLMHead
```

在 decoder-only 大模型推理流程中，它们的位置是：

```text
input_ids
  ↓
VocabParallelEmbedding
  ↓
hidden_states
  ↓
DecoderLayer × N
  ↓
Final RMSNorm
  ↓
ParallelLMHead
  ↓
logits
  ↓
Sampler
  ↓
next_token_id
```

其中：

| 模块 | 作用 | 在 forward 中的位置 |
|---|---|---|
| `VocabParallelEmbedding` | 将 token id 查表成 hidden state | 模型开头 |
| `ParallelLMHead` | 将 hidden state 投影成词表 logits | 模型结尾 |

这个文件不实现 Attention、MLP、GatedDeltaNet，也不管理 KV Cache。它解决的是 **词表维度并行切分、Embedding 查表、LM Head 输出 logits、prefill 阶段只算最后 token logits** 等问题。

---

## 2. qwen3.6 版本整体变化概览

相比原版，qwen3.6 版本主要有两类变化：

| 改动点 | 原版 nano-vLLM | qwen3.6 版本 | 作用 |
|---|---|---|---|
| 权重加载 | 只接收 `loaded_weight` 并直接切片 copy | 新增 `loaded_scale`，并调用 `maybe_dequant_fp8_weight` | 支持 FP8 embedding / lm_head 权重加载 |
| LM Head 多卡输出 | `ParallelLMHead.forward()` 内部 gather 各 rank logits，并在 rank 0 拼成完整 vocab logits | `ParallelLMHead.forward()` 只返回当前 rank 的局部 logits | 减少完整 vocab logits 跨卡传输，将跨 rank 采样合并放到 `ModelRunner.sample` |
| Embedding forward | mask + embedding + all_reduce | 基本不变 | 仍然保持 vocab parallel embedding 语义 |
| Prefill logits 优化 | 只取每个请求最后一个 token 的 hidden state 算 logits | 保持不变 | 避免 prompt 中所有 token 都做 vocab 投影 |

一句话总结：

```text
qwen3.6 版本的 embed_head.py 主要做了 FP8 权重加载适配，并把 LM Head 内部的 full-vocab logits gather 去掉，改为返回本 rank 的局部 logits。
```

---

## 3. 保持不变：词表并行 Embedding 的核心设计

原版和 qwen3.6 版本都保留了相同的 vocab parallel embedding 思路。

初始化时，每个 rank 只保存一段词表权重：

```python
self.tp_rank = dist.get_rank()
self.tp_size = dist.get_world_size()
self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
```

例如：

```text
vocab_size = 150000
tp_size = 2

rank 0 负责 token id [0, 75000)
rank 1 负责 token id [75000, 150000)
```

这样做的意义是：

```text
把 embedding / lm_head 这种巨大词表权重切到多张 GPU 上，降低单卡显存占用。
```

---

## 4. Embedding forward 逻辑基本不变

两个版本的 `VocabParallelEmbedding.forward()` 基本一致：

```python
if self.tp_size > 1:
    mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
    x = mask * (x - self.vocab_start_idx)

y = F.embedding(x, self.weight)

if self.tp_size > 1:
    y = mask.unsqueeze(1) * y
    dist.all_reduce(y)
return y
```

含义是：

1. 每个 rank 判断哪些 token id 属于自己的词表分片；
2. 属于当前 rank 的 token 正常查 embedding；
3. 不属于当前 rank 的 token 输出置零；
4. 所有 rank 对 embedding 结果做 `all_reduce`；
5. 因为每个 token 只会在一个 rank 上有真实 embedding，所以求和后得到完整 hidden state。

这部分没有变化，说明 qwen3.6 没有改变：

```text
input_ids -> hidden_states
```

这条路径的基本语义。

---

## 5. 改动一：Embedding / LM Head 权重加载支持 FP8 反量化

### 5.1 原版写法

原版 `weight_loader` 很直接：

```python
def weight_loader(self, param, loaded_weight):
    param_data = param.data
    shard_size = param_data.size(0)
    start_idx = self.tp_rank * shard_size
    loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
    param_data.copy_(loaded_weight)
```

它默认 `loaded_weight` 已经是可以直接复制进参数的浮点权重，例如 FP16、BF16 或 FP32。

### 5.2 qwen3.6 版本写法

qwen3.6 版本新增：

```python
from nanovllm.utils.quant import maybe_dequant_fp8_weight
```

并把 `weight_loader` 改成：

```python
def weight_loader(
    self,
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    loaded_scale: torch.Tensor | None = None,
):
    param_data = param.data
    shard_size = param_data.size(0)
    start_idx = self.tp_rank * shard_size
    loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
    loaded_weight = maybe_dequant_fp8_weight(
        loaded_weight, loaded_scale, row_start=start_idx
    )
    param_data.copy_(loaded_weight)
```

### 5.3 作用和意义

这个改动说明 qwen3.6 版本允许 checkpoint 中的 embedding / lm_head 权重是 FP8 格式。

FP8 权重一般不能直接作为最终计算权重使用，而是需要结合 scale 做反量化：

```text
FP8 weight + scale
  ↓
maybe_dequant_fp8_weight
  ↓
FP16 / BF16 / FP32 weight
  ↓
copy 到模型参数
```

这和前面 `linear.py` 的改动是一致的：

```text
linear.py:
    QKV / MLP / o_proj / down_proj 等普通线性层支持 FP8 权重反量化

embed_head.py:
    embedding / lm_head 这种 vocab parallel 词表权重也支持 FP8 权重反量化
```

---

## 6. 为什么需要 `row_start=start_idx`

`VocabParallelEmbedding` 是按词表维度切分的。权重形状是：

```text
[num_embeddings, embedding_dim]
```

在 tensor parallel 下：

```text
rank 0: weight[0 : shard_size]
rank 1: weight[shard_size : 2 * shard_size]
rank 2: weight[2 * shard_size : 3 * shard_size]
```

所以当前 rank 的权重 shard 在完整权重矩阵里的行偏移就是：

```python
row_start = start_idx
```

如果 FP8 scale 是按行、按 block 或按二维分块组织的，反量化函数必须知道当前 shard 对应原矩阵的哪一段。否则可能出现：

```text
权重切片来自 rank 1 的词表行，
但反量化时使用了 rank 0 对应的 scale。
```

这会导致权重数值错误。

因此：

```python
maybe_dequant_fp8_weight(loaded_weight, loaded_scale, row_start=start_idx)
```

的核心意义是：

```text
在 vocab parallel 切片后，仍然让 FP8 scale 和原始词表行位置对齐。
```

---

## 7. 改动二：ParallelLMHead 去掉完整 logits gather

这是本文件最重要的推理路径变化。

### 7.1 原版 LM Head

原版 `ParallelLMHead.forward()` 中有：

```python
logits = F.linear(x, self.weight)
if self.tp_size > 1:
    all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
    dist.gather(logits, all_logits, 0)
    logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
return logits
```

含义是：

```text
每个 rank 只计算自己 vocab shard 的 logits
  ↓
把所有 rank 的局部 logits gather 到 rank 0
  ↓
rank 0 按 vocab 维度 concat 成完整 logits
```

最终：

```text
rank 0: logits [num_seqs, full_vocab_size]
rank 1..N: logits None
```

这种实现简单，因为后续 sampler 只需要在 rank 0 的完整 logits 上采样。

缺点是：

```text
每一步都要跨卡传输完整 vocab logits shard。
```

当 vocab 很大、decode 每步都要执行时，这个通信开销会比较明显。

### 7.2 qwen3.6 LM Head

qwen3.6 版本变成：

```python
logits = F.linear(x, self.weight)
return logits
```

也就是说：

```text
每个 rank 只返回自己的局部 vocab logits，
不在 LM Head 内部 gather 成完整 vocab logits。
```

如果：

```text
tp_size = 2
```

那么输出变成：

```text
rank 0 logits: [num_seqs, vocab_size / 2]
rank 1 logits: [num_seqs, vocab_size / 2]
```

---

## 8. 为什么去掉 LM Head 内部 gather

qwen3.6 版本把跨 rank 采样合并逻辑转移到了 `ModelRunner.sample()`。

你前面上传的 qwen3.6 `model_runner.py` 中，采样路径大致是：

```python
if greedy:
    token_ids, scores = self.sampler.greedy_with_scores(logits)
else:
    token_ids, scores = self.sampler.forward_with_scores(logits, temperatures)

if self.world_size == 1:
    return token_ids.tolist()

token_ids = token_ids + self.model.lm_head.vocab_start_idx
all_scores = [torch.empty_like(scores) for _ in range(self.world_size)]
all_token_ids = [torch.empty_like(token_ids) for _ in range(self.world_size)]
dist.all_gather(all_scores, scores)
dist.all_gather(all_token_ids, token_ids)
...
return token_ids.gather(0, rank_ids).squeeze(0).tolist()
```

也就是说，新路径是：

```text
每个 rank 在自己的 vocab shard 上先得到候选 token 和 score
  ↓
跨 rank 只 all_gather token_ids 和 scores
  ↓
rank 0 选择全局 token
```

原版路径是：

```text
gather 完整 logits
  ↓
rank 0 在完整 vocab 上采样
```

qwen3.6 路径是：

```text
每个 rank 本地处理局部 logits
  ↓
只跨卡传少量候选 token / score
```

通信量更小，尤其适合 decode 阶段。

---

## 9. 通信量变化

假设：

```text
batch_size = B
vocab_size = V
tp_size = T
```

原版 LM Head gather 的通信对象大致是：

```text
每个 rank 传 [B, V/T] logits
rank 0 拼成 [B, V]
```

qwen3.6 的采样合并通信对象大致是：

```text
每个 rank 传 [B] token_ids 和 [B] scores
```

这就是为什么 qwen3.6 要把完整 logits gather 从 `ParallelLMHead` 中移出去。

它把职责拆成：

```text
ParallelLMHead:
    只负责本 rank 的局部 logits 计算

ModelRunner.sample:
    负责跨 rank 的采样结果合并
```

这种分层更符合推理执行器的职责边界。

---

## 10. Prefill 阶段只取最后 token 的优化保持不变

两个版本都保留：

```python
context = get_context()
if context.is_prefill:
    last_indices = context.cu_seqlens_q[1:] - 1
    x = x[last_indices].contiguous()
```

这表示：

```text
prefill 阶段模型主干会处理 prompt 的所有 token，
但 LM Head 只对每个请求最后一个 token 的 hidden state 计算 logits。
```

原因是：

```text
prompt 中间 token 的 hidden state 主要用于写 KV Cache，
真正预测下一个 token 只需要最后位置的 logits。
```

这能避免：

```text
[prompt_total_tokens, hidden_size]
  ↓ LM Head
[prompt_total_tokens, vocab_size]
```

这种巨大开销。

qwen3.6 保持该优化，说明它没有改变 prefill logits 的基本策略。

---

## 11. 这不是完整 FP8 runtime kernel

需要注意：qwen3.6 版本虽然支持 FP8 权重加载，但 forward 仍然是：

```python
F.embedding(x, self.weight)
F.linear(x, self.weight)
```

所以这里的 FP8 支持更准确地说是：

```text
load-time FP8 weight dequantization
```

而不是：

```text
runtime FP8 embedding lookup / runtime FP8 GEMM
```

也就是说：

```text
磁盘权重可以是 FP8，
加载时反量化到模型参数 dtype，
推理时仍然使用普通 PyTorch embedding / linear。
```

如果要实现真正运行时 FP8 推理，还需要专门的 FP8 GEMM kernel、activation scale、output scale 等配套。

---

## 12. 和 Qwen3.6 hybrid 架构的关系

`embed_head.py` 本身不直接实现：

```text
GatedDeltaNet
recurrent state
conv state
state_slot_id
MRoPE
hybrid layer pattern
```

这些内容更多体现在：

```text
model_runner.py
sequence.py
scheduler.py
context.py
gated_delta_net.py
qwen3_5.py / qwen3_6.py
```

但 `embed_head.py` 仍然是 Qwen3.6 支持链路的重要组成部分：

1. **FP8 权重兼容**：embedding 和 lm_head 也可能是大权重，必须支持 FP8 checkpoint 加载。  
2. **更轻的 TP 采样路径**：不再 gather 完整 logits，有利于降低 decode 阶段通信压力。  
3. **保留 prefill logits 优化**：只对每个请求最后 token 计算 logits，避免 prompt 全 token 做 vocab 投影。

所以它不是 hybrid 状态管理文件，而是：

```text
模型输入输出层的量化适配 + tensor parallel 输出路径改造文件。
```

---

## 13. 与 linear.py / model_runner.py 的关系

### 13.1 和 linear.py 的关系

前面 `linear.py` 的 qwen3.6 改造是：

```text
QKV / MLP / o_proj / down_proj 等线性层支持 FP8 权重反量化。
```

这里 `embed_head.py` 的改造是：

```text
embedding / lm_head 这种词表并行权重也支持 FP8 权重反量化。
```

两者共同构成：

```text
模型主要权重加载路径的 FP8 checkpoint 兼容能力。
```

### 13.2 和 model_runner.py 的关系

`embed_head.py` 去掉 LM Head 内部 gather 后，`model_runner.py` 必须承担跨 rank 采样合并工作。

整体路径变成：

```text
ParallelLMHead:
    每个 rank 计算局部 vocab logits

ModelRunner.sample:
    每个 rank 从局部 logits 中得到候选 token 和 score
    all_gather 所有 rank 的候选 token / score
    rank 0 选出最终 token
```

因此，这两个文件要一起看：

```text
embed_head.py 改局部 logits 输出，
model_runner.py 改分布式采样合并。
```

---

## 14. 改动总结表

| 位置 | 原版 | qwen3.6 | 意义 |
|---|---|---|---|
| import | 只导入 `get_context` | 新增 `maybe_dequant_fp8_weight` | 引入 FP8 反量化能力 |
| `weight_loader` 参数 | `loaded_weight` | `loaded_weight + loaded_scale` | 支持带 scale 的 FP8 权重 |
| 权重加载 | 切片后直接 copy | 切片后按 `row_start` 反量化再 copy | 保证 vocab shard 和 scale 对齐 |
| Embedding forward | mask + all_reduce | 不变 | 保持词表并行 embedding 语义 |
| LM Head prefill 优化 | 只取每个请求最后 token | 不变 | 减少 prefill logits 计算 |
| LM Head logits gather | 内部 gather 完整 logits 到 rank 0 | 删除 gather，返回本 rank 局部 logits | 降低完整 vocab logits 跨卡通信 |
| 跨 rank 采样 | sampler 面对完整 logits | `ModelRunner.sample` 汇聚 token / score | 分布式采样路径更轻量 |

---

## 15. 面试角度回答

如果面试官问：

> qwen3.6 版本的 `embed_head.py` 相比原版改了什么？

可以这样回答：

`embed_head.py` 负责 vocab parallel 的 Embedding 和 LM Head。原版中，每张卡只保存一段词表权重，Embedding forward 通过 mask 查本 rank 的 embedding，再 all_reduce 得到完整 hidden state；LM Head 也按 vocab 切分，但原版会在 `ParallelLMHead.forward()` 里 gather 各 rank 的局部 logits，在 rank 0 拼成完整词表 logits。qwen3.6 版本主要有两个变化：第一，`VocabParallelEmbedding.weight_loader` 新增了 `loaded_scale`，并调用 `maybe_dequant_fp8_weight`，支持 FP8 embedding / lm_head 权重在加载时反量化，同时用 `row_start` 保证词表分片和 scale 对齐；第二，删除了 LM Head 内部的完整 logits gather，让每个 rank 只返回局部 logits，后续由 `ModelRunner.sample()` 汇聚 token 和 score 完成全局采样。整体来看，这个文件不是直接实现 GatedDeltaNet，而是让模型输入输出层适配 FP8 权重格式和更轻量的 tensor parallel 采样路径。

---

## 16. 最终结论

qwen3.6 版本 `embed_head.py` 的核心改造是：

```text
在 vocab parallel embedding / lm_head 上支持 FP8 权重加载，并重构多卡 logits 输出路径。
```

它的工程意义主要体现在三点：

1. **FP8 checkpoint 兼容**  
   通过 `loaded_scale` 和 `maybe_dequant_fp8_weight`，支持 embedding / lm_head 词表权重的加载时反量化。

2. **Tensor Parallel scale 对齐**  
   通过 `row_start=start_idx`，确保每个 rank 的 vocab shard 使用正确的 FP8 scale。

3. **减少 LM Head 后的通信压力**  
   qwen3.6 不再在 LM Head 内部 gather 完整 vocab logits，而是返回局部 logits，让采样层只汇聚候选 token 和 score，降低 decode 阶段跨卡通信量。

因此，这个文件可以理解为：

```text
Qwen3.6 项目中模型输入输出层的量化适配 + 分布式采样路径改造。
```


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
