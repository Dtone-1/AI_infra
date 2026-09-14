# embed_head.py 源码宏观解析

## 1. 文件整体定位

`embed_head.py` 是 nano-vLLM 中负责 **输入词表 Embedding** 和 **输出 LM Head** 的 layer 文件。

它处在模型 forward 的两端：

```text
input_ids
  -> VocabParallelEmbedding
  -> hidden_states
  -> Qwen3 Decoder Layers
  -> final hidden_states
  -> ParallelLMHead
  -> logits
  -> Sampler
  -> next token
```

在 `models/qwen3.py` 中，它被这样使用：

```python
self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
```

所以这个文件负责两个关键入口/出口：

| 类 | 作用 | 所在位置 |
|---|---|---|
| `VocabParallelEmbedding` | 把 token id 转成 hidden state | 模型最开始 |
| `ParallelLMHead` | 把 hidden state 转成词表 logits | 模型最后输出 |

如果把 `linear.py` 理解成 Attention/MLP 内部的矩阵乘法基础设施，那么 `embed_head.py` 就是模型输入端和输出端的词表并行基础设施。

## 2. 这个文件要解决的核心问题

这个文件主要解决四类问题：

| 问题 | 说明 |
|---|---|
| 模型结构问题 | 实现 Transformer/Qwen 的 token embedding 和 LM head |
| 张量计算问题 | 使用 `F.embedding` 和 `F.linear` 完成查表与 logits 计算 |
| 权重加载问题 | 从完整词表权重中加载当前 TP rank 负责的词表切片 |
| 并行切分问题 | 按 vocab 维度切分 embedding / lm_head 权重 |

它和原始 Transformer 的对应关系是：

```text
token ids -> token embedding -> Transformer blocks -> lm_head -> logits
```

其中：

- `VocabParallelEmbedding` 对应输入端的 token embedding；
- `ParallelLMHead` 对应输出端的 language modeling head；
- 如果模型启用 `tie_word_embeddings`，两者可以共享同一份权重。

它和高性能推理的关系也很直接：大模型词表通常很大，例如几万到十几万 token。如果 embedding 和 lm_head 都完整复制到每张 GPU 上，显存会浪费；如果按 vocab 维度切开，每张 GPU 只保存一部分词表权重，就能降低单卡显存压力。

同时，输出 logits 的维度是 `vocab_size`，通常很大。`ParallelLMHead` 通过每个 rank 只计算部分 vocab logits，再在 rank 0 上 gather 拼接，减少了每个 rank 的权重规模。

## 3. 代码结构总览

### 3.1 文件导入

```python
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context
```

这些导入分别服务于：

| 模块 | 作用 |
|---|---|
| `torch` | 张量操作 |
| `nn` | 定义 PyTorch Module 和 Parameter |
| `F` | 调用 `F.embedding`、`F.linear` |
| `dist` | 获取 TP rank / TP size，并进行通信 |
| `get_context` | 获取当前推理阶段是 prefill 还是 decode |

这里最重要的是 `torch.distributed` 和 `get_context`。

- `torch.distributed` 说明这个文件支持多卡 tensor parallel；
- `get_context` 说明 LM Head 的行为会根据 prefill/decode 阶段发生变化。

### 3.2 `VocabParallelEmbedding`

`VocabParallelEmbedding` 的作用是：把输入 token id 转成 hidden state，并且词表维度按 TP rank 切分。

普通 embedding 权重形状是：

```text
[vocab_size, hidden_size]
```

在 vocab parallel 中，每个 rank 只保存一部分词表：

```text
[vocab_size / tp_size, hidden_size]
```

初始化中的关键变量：

| 变量 | 含义 |
|---|---|
| `tp_rank` | 当前 GPU / 进程编号 |
| `tp_size` | tensor parallel 总进程数 |
| `num_embeddings` | 总词表大小，也就是 `vocab_size` |
| `embedding_dim` | embedding 维度，也就是 `hidden_size` |
| `num_embeddings_per_partition` | 当前 rank 负责的 token 数 |
| `vocab_start_idx` | 当前 rank 负责的词表起始 token id |
| `vocab_end_idx` | 当前 rank 负责的词表结束 token id |
| `weight` | 当前 rank 持有的局部 embedding 权重 |

例如 `vocab_size = 100000`，`tp_size = 2`：

| rank | 负责 token id 范围 | 权重形状 |
|---|---|---|
| rank 0 | `[0, 50000)` | `[50000, hidden_size]` |
| rank 1 | `[50000, 100000)` | `[50000, hidden_size]` |

这就是 vocab parallel embedding。

### 3.3 `VocabParallelEmbedding.weight_loader`

```python
def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
    param_data = param.data
    shard_size = param_data.size(0)
    start_idx = self.tp_rank * shard_size
    loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
    param_data.copy_(loaded_weight)
```

这个函数负责权重加载。

HuggingFace 原始 embedding 权重是完整词表：

```text
loaded_weight: [vocab_size, hidden_size]
```

当前 rank 只需要其中一段：

```text
param_data: [vocab_size / tp_size, hidden_size]
```

所以它沿第 0 维切分：

```python
loaded_weight.narrow(0, start_idx, shard_size)
```

这里第 0 维就是 vocab 维度。

这和 `linear.py` 里的权重加载思想一样：原始模型权重是完整的，nano-vLLM 运行时只给每个 rank 加载自己负责的切片。

### 3.4 `VocabParallelEmbedding.forward`

forward 的输入是：

```python
x: torch.Tensor
```

在模型中它对应 `input_ids`：

```text
x: [num_tokens]
```

这里的 `num_tokens` 不是固定 batch size，而是当前调度步中被拉平的 token 数。

在 prefill 阶段：

```text
num_tokens = 本轮所有 prompt 中被调度的 token 总数
```

在 decode 阶段：

```text
num_tokens = 本轮参与 decode 的 sequence 数
```

如果 `tp_size == 1`，逻辑非常简单：

```python
y = F.embedding(x, self.weight)
```

输出：

```text
y: [num_tokens, hidden_size]
```

如果 `tp_size > 1`，则需要先判断每个 token id 是否属于当前 rank 的词表分片：

```python
mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
```

然后把全局 token id 转成当前 rank 内部的局部 token id：

```python
x = mask * (x - self.vocab_start_idx)
```

为什么要转成局部 id？因为当前 rank 的 `weight` 只有局部词表大小。例如 rank 1 负责 `[50000, 100000)`，全局 token id `60000` 在本 rank 内部应该变成 `10000`。

接着查 embedding：

```python
y = F.embedding(x, self.weight)
```

对于不属于当前 rank 的 token，`mask` 为 False，后面会把对应 embedding 置零：

```python
y = mask.unsqueeze(1) * y
```

最后所有 rank 做 `all_reduce`：

```python
dist.all_reduce(y)
```

因为每个 token 只会在一个 rank 上得到非零 embedding，其余 rank 是零，所以 all_reduce 求和后，每个 rank 都得到完整的 embedding 输出。

整体逻辑可以理解为：

```text
每个 rank 只查自己负责的 token
不负责的 token 输出 0
所有 rank 相加
得到完整 hidden_states
```

### 3.5 `ParallelLMHead`

`ParallelLMHead` 继承自 `VocabParallelEmbedding`：

```python
class ParallelLMHead(VocabParallelEmbedding):
```

这说明 LM Head 和 Embedding 的权重切分方式是一样的：都按 vocab 维度切分。

区别在于：

- Embedding 是根据 token id 查表；
- LM Head 是把 hidden state 投影到 vocab logits。

普通 LM Head 的权重形状也是：

```text
[vocab_size, hidden_size]
```

计算公式是：

```text
logits = hidden_states @ weight.T
```

在代码中就是：

```python
logits = F.linear(x, self.weight)
```

由于当前 rank 只持有部分 vocab 权重，所以每个 rank 只能算部分 logits：

```text
local_logits: [num_sequences, vocab_size / tp_size]
```

最后 rank 0 收集所有 rank 的 logits 并拼接：

```python
dist.gather(logits, all_logits, 0)
logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
```

最终 rank 0 得到完整词表 logits：

```text
full_logits: [num_sequences, vocab_size]
```

然后 sampler 在 rank 0 上采样 next token。

## 4. 张量流和 forward 流程

### 4.1 输入端：`input_ids -> hidden_states`

在 `Qwen3Model.forward` 中：

```python
hidden_states = self.embed_tokens(input_ids)
```

输入：

```text
input_ids: [num_tokens]
```

输出：

```text
hidden_states: [num_tokens, hidden_size]
```

这里的 `num_tokens` 由 `model_runner.py` 决定。

prefill 阶段，`prepare_prefill` 会把多个 sequence 的 prompt token 展平成一个列表：

```text
seq1 tokens + seq2 tokens + ...
```

decode 阶段，`prepare_decode` 每个 sequence 只放入一个 `last_token`：

```text
[seq1_last_token, seq2_last_token, ...]
```

所以 embedding 层不直接关心原始 batch 结构，它只看到一维的 token ids。

### 4.2 中间：`hidden_states -> Transformer layers`

Embedding 输出后进入多层 Qwen3 Decoder Layer：

```python
for layer in self.layers:
    hidden_states, residual = layer(positions, hidden_states, residual)
```

此时 shape 仍然是：

```text
hidden_states: [num_tokens, hidden_size]
```

Attention 内部会进一步把它变成：

```text
q: [num_tokens, num_heads, head_dim]
k: [num_tokens, num_kv_heads, head_dim]
v: [num_tokens, num_kv_heads, head_dim]
```

但 `embed_head.py` 自己只负责前后的 `[num_tokens, hidden_size]` 和 `[num_sequences, vocab_size]`。

### 4.3 输出端：`hidden_states -> logits`

在 `Qwen3ForCausalLM.compute_logits` 中：

```python
return self.lm_head(hidden_states)
```

输入：

```text
hidden_states: [num_tokens, hidden_size]
```

但是 LM Head 并不总是对所有 token 都算 logits。

#### prefill 阶段

prefill 阶段模型会一次性处理 prompt 的多个 token，例如：

```text
prompt: [t0, t1, t2, t3]
```

理论上每个位置都能输出一个 logits：

```text
logits for t0
logits for t1
logits for t2
logits for t3
```

但推理服务生成第一个新 token 时，只需要 prompt 最后一个位置的 logits，也就是 `t3` 对应的 logits。

所以代码中：

```python
if context.is_prefill:
    last_indices = context.cu_seqlens_q[1:] - 1
    x = x[last_indices].contiguous()
```

`cu_seqlens_q` 是多个 sequence 的累积长度。例如两条 prompt 长度分别是 3 和 5：

```text
cu_seqlens_q = [0, 3, 8]
```

则每条 sequence 最后一个 token 的 index 是：

```text
last_indices = [2, 7]
```

于是 LM Head 只对这些位置算 logits：

```text
x: [num_sequences, hidden_size]
```

这可以减少无用的 logits 计算。

#### decode 阶段

decode 阶段每条 sequence 本来就只输入一个 last token：

```text
input_ids: [num_sequences]
hidden_states: [num_sequences, hidden_size]
```

所以不需要再筛选最后位置，直接算 logits：

```text
logits: [num_sequences, vocab_size / tp_size]
```

### 4.4 vocab 并行 logits 汇总

每个 rank 只计算局部词表 logits：

```text
rank 0: logits for vocab [0, V/T)
rank 1: logits for vocab [V/T, 2V/T)
...
```

然后：

```python
dist.gather(logits, all_logits, 0)
```

rank 0 得到所有局部 logits：

```text
all_logits = [rank0_logits, rank1_logits, ...]
```

再拼接：

```python
torch.cat(all_logits, -1)
```

得到完整 logits：

```text
[num_sequences, vocab_size]
```

`model_runner.py` 中也能看到，采样只在 rank 0 上发生：

```python
token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
```

所以非 rank 0 的 `ParallelLMHead.forward` 返回 `None` 是可以接受的，因为它们不负责最终采样。

## 5. 和 Transformer / Qwen 模型结构的关系

### 5.1 输入 Embedding 的关系

Transformer 不能直接处理字符串，也不能直接处理 token id。token id 只是离散编号，必须先转成连续向量。

流程是：

```text
字符串 prompt
  -> tokenizer.encode
  -> input_ids
  -> embedding lookup
  -> hidden_states
```

`VocabParallelEmbedding` 对应的就是：

```text
input_ids -> hidden_states
```

它和普通 Transformer embedding 的区别是：普通 embedding 保存完整词表，而 nano-vLLM 在多卡情况下按词表维度切开。

### 5.2 输出 LM Head 的关系

Transformer Decoder 最后一层输出的是 hidden state，不是文字。

要生成下一个 token，需要把 hidden state 映射回词表空间：

```text
hidden_states -> logits over vocabulary -> sample / argmax -> next token id
```

`ParallelLMHead` 对应的就是：

```text
hidden_states -> logits
```

logits 的每一列对应一个词表 token 的分数。采样器根据这些分数生成下一个 token。

### 5.3 和 Qwen3ForCausalLM 的关系

在 `Qwen3ForCausalLM` 中：

```python
self.model = Qwen3Model(config)
self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
```

`self.model` 负责从 token id 算到最终 hidden state。

`self.lm_head` 负责从 final hidden state 算到 logits。

整体可以理解为：

```text
Qwen3ForCausalLM
  = Qwen3Model + ParallelLMHead
  = token ids -> hidden states -> logits
```

如果配置中启用了：

```python
config.tie_word_embeddings
```

则：

```python
self.lm_head.weight.data = self.model.embed_tokens.weight.data
```

这表示输入 embedding 和输出 LM Head 共享权重。

这个设计在很多语言模型里都存在，目的通常是减少参数量，并让输入 token 表示和输出 token 分类使用同一套词表空间。

### 5.4 和 prefill/decode 的关系

`embed_head.py` 虽然不是调度器，但它知道当前是 prefill 还是 decode：

```python
context = get_context()
if context.is_prefill:
    ...
```

这说明它和推理系统的执行阶段有关。

prefill 阶段：

- 输入是 prompt 的多个 token；
- Transformer 会计算所有 prompt token 的 hidden states；
- LM Head 只需要最后一个 token 的 hidden state；
- 目的是生成首个新 token。

decode 阶段：

- 每条 sequence 输入一个新 token；
- Transformer 结合 KV Cache 计算当前步 hidden state；
- LM Head 对每条 sequence 算 logits；
- 采样得到下一个 token。

这就是为什么 `ParallelLMHead` 要读取 `context.is_prefill`。

## 6. 学习总结

`embed_head.py` 的核心价值可以概括为一句话：

> 它实现了 nano-vLLM 中按词表维度切分的输入 embedding 和输出 LM Head，让模型能够在多卡 tensor parallel 下完成 token id 到 hidden state、hidden state 到 logits 的转换。

学习这个文件时要抓住三条主线：

1. **模型结构主线**：Embedding 在模型最前面，LM Head 在模型最后面；
2. **张量 shape 主线**：`input_ids [num_tokens] -> hidden_states [num_tokens, hidden_size] -> logits [num_sequences, vocab_size]`；
3. **并行切分主线**：Embedding 和 LM Head 都按 vocab 维度切分，每个 rank 只保存部分词表权重。

它和 `linear.py` 的关系也很清楚：

| 文件 | 切分对象 | 典型位置 |
|---|---|---|
| `linear.py` | hidden/intermediate/head 相关线性层 | Attention / MLP 内部 |
| `embed_head.py` | vocab 相关权重 | 模型输入端 / 输出端 |

对于 AI Infra 推理学习来说，这个文件能帮助你理解一个重要事实：推理引擎中的模型并行不只发生在 Attention 和 MLP 中，词表 embedding 和 logits 输出同样需要考虑并行切分、权重加载和跨 rank 通信。



%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]

%% 项目关联导航：结束 %%
