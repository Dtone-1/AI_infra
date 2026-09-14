# context.py_解析

## 1. 文件整体定位

`context.py` 是 nano-vLLM 中的一个 **推理上下文管理工具文件**。它本身不是模型层，也不直接完成矩阵乘法、Attention 计算、MLP 计算或采样；它的核心作用是保存“当前这一轮模型 forward 所需要的运行时元信息”，让模型内部的 layer 在 forward 时能够知道当前处于 **Prefill** 还是 **Decode**，以及应该如何访问 KV Cache。

在 nano-vLLM 推理系统中，请求从 Scheduler 进入 ModelRunner 后，ModelRunner 会先根据本轮调度结果构造一批张量，例如 `input_ids`、`positions`、`slot_mapping`、`block_tables`、`context_lens` 等。随后模型开始执行 forward。问题是：Attention 层除了拿到 `q/k/v` 以外，还必须知道 KV Cache 的物理位置、每条 sequence 的长度、batch 内不同请求的边界等信息。`context.py` 就是为了解决这个问题而存在的。

从推理流程位置看，它位于：

```text
Scheduler 调度请求
    ↓
ModelRunner.prepare_prefill / prepare_decode
    ↓
set_context(...) 设置本轮 forward 的上下文
    ↓
Qwen3ForCausalLM.forward(...)
    ↓
Attention / LMHead 等模块通过 get_context() 读取上下文
    ↓
reset_context() 清空上下文
```

所以，`context.py` 可以理解为 **ModelRunner 和模型内部 layer 之间传递运行时调度信息的桥梁**。

它不是 layer 文件；如果非要把它和模型 forward 联系起来，它主要服务于：

1. Attention 层的 KV Cache 写入与读取；
2. FlashAttention / Paged Attention 所需的变长序列信息；
3. LMHead 在 prefill 阶段只取每条 prompt 最后一个 token 的 hidden state；
4. CUDA Graph decode replay 时传递固定 buffer 对应的元信息。

---

## 2. 这个文件要解决的核心问题

### 2.1 它解决的不是模型结构问题，而是推理运行时上下文传递问题

`context.py` 不定义 Transformer 的新结构，也不加载权重，不做采样，也不直接做张量计算。它解决的是 **高性能推理中，模型 forward 需要额外运行时信息的问题**。

在普通 PyTorch Transformer 训练代码中，Attention 层通常只需要：

```text
hidden_states
attention_mask
position_ids
```

但在 vLLM / nano-vLLM 这类推理引擎中，Attention 层还需要知道：

1. 当前是 prefill 还是 decode；
2. 本 batch 里有几条 sequence；
3. 每条 sequence 的 token 边界在哪里；
4. 新产生的 K/V 应该写到 KV Cache 的哪个物理槽位；
5. decode 时每条 sequence 已经有多长；
6. 每条 sequence 的逻辑 block 对应哪些物理 KV block；
7. 是否存在 prefix cache；
8. CUDA Graph replay 时要使用哪一批静态 buffer。

这些信息并不是模型参数，也不是 token embedding，而是 **推理引擎调度层产生的元信息**。因此需要一个地方把这些信息保存起来，让 Attention / LMHead 在 forward 过程中读取。

---

### 2.2 它和原始 Transformer 的对应关系

原始 Transformer 结构大致是：

```text
input_ids
  ↓
Embedding
  ↓
Transformer Blocks
  ├── Self-Attention
  ├── MLP / FFN
  └── Residual / Norm
  ↓
LMHead
  ↓
logits
```

`context.py` 不对应其中某一个数学模块，但它主要影响两个位置：

```text
Self-Attention  ←  读取 Context，决定怎么使用 KV Cache
LMHead          ←  读取 Context，决定 prefill 阶段取哪些 hidden states 算 logits
```

也就是说，它不是 Transformer 原理中的“Attention 公式”本身，而是推理系统为了高效执行 Attention 而增加的 **运行时控制信息层**。

---

### 2.3 它和高性能推理目标的关系

大模型推理的核心性能问题之一是：如何高效处理大量不同长度的请求，并且让 KV Cache 复用、调度、显存管理和 Attention kernel 配合起来。

`context.py` 对高性能推理的意义主要体现在：

1. **支持 Prefill / Decode 分流**

   Prefill 阶段一次处理 prompt 的多个 token，适合使用变长 FlashAttention；Decode 阶段每条请求通常只新增一个 token，适合使用带 KV Cache 的 attention kernel。`is_prefill` 字段就是区分这两种执行路径的关键标志。

2. **支持变长 batch**

   一个 batch 里的不同请求 prompt 长度可能不同。`cu_seqlens_q`、`cu_seqlens_k` 和 `max_seqlen_q/k` 用来告诉 FlashAttention：多个 sequence 被拼成一个扁平 token 张量后，每条 sequence 的边界在哪里。

3. **支持 Paged KV Cache**

   vLLM / nano-vLLM 不希望为每个请求连续分配一整段最大长度 KV Cache，而是把 KV Cache 切成 block。`slot_mapping` 和 `block_tables` 就是逻辑 token / 逻辑 block 到物理 KV Cache 位置的映射信息。

4. **支持 CUDA Graph replay**

   Decode 阶段 batch size 较小且形状相对固定，适合 CUDA Graph 优化。CUDA Graph replay 期间不能频繁重新构造动态计算图，所以需要把 `slot_mapping`、`context_lens`、`block_tables` 等信息提前写入静态 buffer，再通过 Context 给模型层读取。

---

## 3. 代码结构总览

源码如下：

```python
from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
```

---

### 3.1 `Context` 类

`Context` 是一个 dataclass，用来保存当前 forward 所需的元信息。

```python
@dataclass(slots=True)
class Context:
```

这里有两个关键点：

1. `@dataclass`：自动生成初始化函数，避免手写 `__init__`；
2. `slots=True`：限制对象只能拥有声明过的字段，不再使用普通对象的 `__dict__`，可以减少对象开销，也避免误加不存在的属性。

虽然这个文件代码很短，但 `Context` 中每一个字段都对应推理系统中的一个重要概念。

---

### 3.2 `is_prefill`

```python
is_prefill: bool = False
```

这个字段表示当前 forward 是 prefill 还是 decode。

两种阶段的核心区别是：

| 阶段 | 输入 token 数 | 主要任务 | Attention 方式 |
|---|---:|---|---|
| Prefill | 每条请求可能有多个 prompt token | 处理 prompt，建立 KV Cache | 变长 FlashAttention |
| Decode | 每条请求通常只新增 1 个 token | 逐 token 生成下一个 token | 读取历史 KV Cache |

在 Attention 层中，通常会根据这个字段走不同逻辑：

```text
if context.is_prefill:
    使用 flash_attn_varlen_func
else:
    使用 flash_attn_with_kvcache
```

所以，`is_prefill` 是连接调度阶段和模型执行阶段的分支控制信号。

---

### 3.3 `cu_seqlens_q`

```python
cu_seqlens_q: torch.Tensor | None = None
```

`cu_seqlens_q` 表示 query 序列的 cumulative sequence lengths，也就是 batch 内每条 sequence 的 query token 累计边界。

假设一个 prefill batch 里有 3 条请求，它们本轮参与 attention 的 query token 数分别是：

```text
seq0: 5 tokens
seq1: 3 tokens
seq2: 4 tokens
```

那么所有 token 会被拼成一个扁平张量：

```text
total_q_tokens = 5 + 3 + 4 = 12
```

对应的 `cu_seqlens_q` 是：

```text
[0, 5, 8, 12]
```

含义是：

```text
seq0 的 token 范围: [0, 5)
seq1 的 token 范围: [5, 8)
seq2 的 token 范围: [8, 12)
```

它通常是 `torch.int32` 类型，shape 为：

```text
[num_seqs + 1]
```

FlashAttention 的变长接口需要它来知道扁平 token 张量中每条 sequence 的边界。

---

### 3.4 `cu_seqlens_k`

```python
cu_seqlens_k: torch.Tensor | None = None
```

`cu_seqlens_k` 表示 key/value 序列的 cumulative sequence lengths。

在没有 prefix cache 的普通 prefill 中，query 和 key 的长度通常一致。例如 prompt 长度是 5，那么：

```text
q 长度 = 5
k/v 长度 = 5
```

但如果存在 prefix cache，情况就不同：

```text
已经缓存的 prefix token 数 = 10
本轮新计算 query token 数 = 3
当前 key/value 可见长度 = 10 + 3 = 13
```

所以：

```text
seqlen_q = 本轮新算的 token 数
seqlen_k = 历史可见 token 数 + 本轮新 token 数
```

因此 `cu_seqlens_k` 可能大于 `cu_seqlens_q`。这正是 prefix caching / KV Cache 复用的体现。

---

### 3.5 `max_seqlen_q`

```python
max_seqlen_q: int = 0
```

`max_seqlen_q` 表示当前 batch 中 query 序列的最大长度。

例如：

```text
query lengths = [5, 3, 4]
max_seqlen_q = 5
```

FlashAttention 需要这个值来决定 kernel 内部 block 的上界和调度规模。

---

### 3.6 `max_seqlen_k`

```python
max_seqlen_k: int = 0
```

`max_seqlen_k` 表示当前 batch 中 key/value 序列的最大长度。

例如：

```text
key lengths = [5, 13, 4]
max_seqlen_k = 13
```

它通常用于 prefill 阶段的变长 attention。对于 decode 阶段，更多依赖 `context_lens` 和 `block_tables` 来定位历史 KV。

---

### 3.7 `slot_mapping`

```python
slot_mapping: torch.Tensor | None = None
```

`slot_mapping` 是 nano-vLLM 中非常关键的字段。它表示 **本轮产生的 K/V token 应该写入 KV Cache 的哪个物理 slot**。

可以把 KV Cache 理解成一个大的物理存储池：

```text
KV Cache = 很多 block
每个 block = block_size 个 token 槽位
每个 token 槽位 = 存一份 K 和一份 V
```

如果某个 token 要写入：

```text
physical_block_id = 7
offset_in_block = 3
block_size = 16
```

那么它的物理 slot 可以表示为：

```text
slot = physical_block_id * block_size + offset_in_block
     = 7 * 16 + 3
     = 115
```

`slot_mapping` 就是把本轮 token 映射到这些物理 slot。

在 prefill 阶段：

```text
slot_mapping shape = [本轮所有新计算 token 总数]
```

在 decode 阶段：

```text
slot_mapping shape = [batch_size]
```

因为 decode 通常每条 sequence 只新增一个 token。

Attention 层写 KV Cache 时，会根据 `slot_mapping` 把新生成的 `k/v` 写入正确位置。

---

### 3.8 `context_lens`

```python
context_lens: torch.Tensor | None = None
```

`context_lens` 表示 decode 阶段每条 sequence 当前的上下文长度。

例如 batch 中有 4 条正在 decode 的请求：

```text
seq0 当前长度 = 32
seq1 当前长度 = 128
seq2 当前长度 = 7
seq3 当前长度 = 64
```

那么：

```text
context_lens = [32, 128, 7, 64]
shape = [batch_size]
```

decode attention 需要知道每条请求当前能看到多少历史 token。否则它不知道应该从 KV Cache 中读取多少 K/V。

注意：`context_lens` 主要用于 decode 阶段；prefill 阶段通常通过 `cu_seqlens_q/k` 表示变长边界。

---

### 3.9 `block_tables`

```python
block_tables: torch.Tensor | None = None
```

`block_tables` 表示每条 sequence 的逻辑 block 到物理 block 的映射。

在 PagedAttention 中，每条 sequence 的 KV Cache 不一定连续存放，而是由多个物理 block 拼起来。例如：

```text
seq0 逻辑 block 0 -> 物理 block 8
seq0 逻辑 block 1 -> 物理 block 3
seq0 逻辑 block 2 -> 物理 block 20
```

那么 `block_tables` 中可能记录为：

```text
[8, 3, 20]
```

如果 batch 中有多条 sequence，那么：

```text
block_tables shape = [batch_size, max_num_blocks_per_seq]
```

它的作用是告诉 attention kernel：

```text
当前 sequence 的第 i 个逻辑 KV block，实际存储在哪个物理 KV block 中
```

这就是 vLLM / nano-vLLM 能够进行 KV Cache 分块管理、复用和动态调度的关键数据结构之一。

---

### 3.10 `_CONTEXT`

```python
_CONTEXT = Context()
```

`_CONTEXT` 是一个模块级全局变量，保存当前正在使用的 Context。

它的默认值是一个空 Context：

```text
is_prefill = False
cu_seqlens_q = None
cu_seqlens_k = None
...
```

模型执行前，ModelRunner 会调用 `set_context(...)` 把它替换成当前 batch 的上下文。模型执行结束后，再调用 `reset_context()` 清空。

---

### 3.11 `get_context()`

```python
def get_context():
    return _CONTEXT
```

`get_context()` 用于在模型内部读取当前上下文。

典型使用位置是：

```text
Attention.forward(...)
ParallelLMHead.forward(...)
```

它的设计好处是：不需要在每一层 forward 的参数中反复传入 `slot_mapping`、`block_tables`、`context_lens` 等参数。否则模型结构的 forward 签名会变得非常复杂。

普通写法可能是：

```python
attention.forward(q, k, v, slot_mapping, block_tables, context_lens, cu_seqlens_q, ...)
```

nano-vLLM 通过全局 Context 简化为：

```python
attention.forward(q, k, v)
```

Attention 内部再调用：

```python
context = get_context()
```

这让模型代码更接近普通 Transformer 结构，同时又能拿到推理引擎所需的额外信息。

---

### 3.12 `set_context(...)`

```python
def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables
    )
```

`set_context()` 的作用是在一次模型 forward 前设置上下文。

在 prefill 阶段，它一般会设置：

```text
is_prefill = True
cu_seqlens_q
cu_seqlens_k
max_seqlen_q
max_seqlen_k
slot_mapping
block_tables
```

在 decode 阶段，它一般会设置：

```text
is_prefill = False
slot_mapping
context_lens
block_tables
```

这体现了 prefill 和 decode 的不同需求：

| 字段 | Prefill | Decode |
|---|---|---|
| `is_prefill` | True | False |
| `cu_seqlens_q` | 需要 | 通常不需要 |
| `cu_seqlens_k` | 需要 | 通常不需要 |
| `max_seqlen_q` | 需要 | 通常不需要 |
| `max_seqlen_k` | 需要 | 通常不需要 |
| `slot_mapping` | 需要 | 需要 |
| `context_lens` | 通常不需要 | 需要 |
| `block_tables` | prefix cache / paged attention 需要 | 需要 |

---

### 3.13 `reset_context()`

```python
def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
```

`reset_context()` 用于在一次模型运行结束后清空上下文。

这是一个很重要的工程细节。因为 `_CONTEXT` 是全局变量，如果不清空，下一轮 forward 可能误用上一轮 batch 的 `slot_mapping`、`block_tables` 或 `context_lens`，导致：

1. KV Cache 写错位置；
2. Attention 读取错误历史 token；
3. batch 内 sequence 边界错误；
4. CUDA Graph replay 读到旧状态；
5. 输出 token 异常。

所以一次推理 step 的典型生命周期是：

```text
set_context(...)
model.forward(...)
sampler(...)
reset_context()
```

---

## 4. 张量流和 forward 流程

`context.py` 本身没有定义模型 forward，但它服务于模型 forward。理解它的关键，是把它放进 prefill 和 decode 两条路径中看。

---

### 4.1 Prefill 阶段的数据流

Prefill 阶段处理 prompt token。假设一个 batch 中有多条请求，每条请求长度不同：

```text
seq0 prompt: 5 tokens
seq1 prompt: 3 tokens
seq2 prompt: 4 tokens
```

ModelRunner 会把这些 token 拼成一个扁平输入：

```text
input_ids shape = [total_tokens]
positions shape = [total_tokens]

total_tokens = 5 + 3 + 4 = 12
```

进入模型后：

```text
input_ids
  ↓ embedding
hidden_states shape = [total_tokens, hidden_size]
  ↓ attention qkv projection
q shape = [total_tokens, num_heads, head_dim]
k shape = [total_tokens, num_kv_heads, head_dim]
v shape = [total_tokens, num_kv_heads, head_dim]
```

由于 batch 中每条 sequence 长度不同，不能简单当成规则的 `[batch_size, seq_len, hidden_size]`。所以需要：

```text
cu_seqlens_q = [0, 5, 8, 12]
cu_seqlens_k = [0, 5, 8, 12]
max_seqlen_q = 5
max_seqlen_k = 5
```

Attention 层通过 `get_context()` 读取这些信息，然后调用变长 attention kernel：

```text
flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    causal=True
)
```

输出：

```text
attention_output shape = [total_tokens, num_heads, head_dim]
```

经过合并 head 和输出投影后：

```text
hidden_states shape = [total_tokens, hidden_size]
```

最后进入 LMHead 时，prefill 阶段通常不需要为 prompt 中每一个 token 都采样下一个 token，只需要每条 sequence 最后一个 token 的 hidden state 来预测下一个 token。

因此 LMHead 可以通过：

```text
last_indices = cu_seqlens_q[1:] - 1
```

取出每条 sequence 的最后一个 token：

```text
last_indices = [4, 7, 11]
```

然后只对这些 hidden states 计算 logits：

```text
selected_hidden_states shape = [batch_size, hidden_size]
logits shape = [batch_size, vocab_size]
```

这也是 `Context` 服务 LMHead 的地方。

---

### 4.2 Decode 阶段的数据流

Decode 阶段每条请求通常只新增一个 token。

假设 batch size 为 4：

```text
input_ids shape = [4]
positions shape = [4]
```

进入模型后：

```text
hidden_states shape = [4, hidden_size]
q shape = [4, num_heads, head_dim]
k shape = [4, num_kv_heads, head_dim]
v shape = [4, num_kv_heads, head_dim]
```

新生成的 k/v 会写入 KV Cache。写入位置由 `slot_mapping` 决定：

```text
slot_mapping shape = [batch_size]
```

例如：

```text
slot_mapping = [115, 32, 401, 78]
```

含义是：

```text
第 0 条请求的新 token 写入 KV Cache slot 115
第 1 条请求的新 token 写入 KV Cache slot 32
第 2 条请求的新 token 写入 KV Cache slot 401
第 3 条请求的新 token 写入 KV Cache slot 78
```

然后 decode attention 需要读取历史 KV Cache。它需要两个核心信息：

```text
context_lens shape = [batch_size]
block_tables shape = [batch_size, max_num_blocks_per_seq]
```

其中：

```text
context_lens = 每条 sequence 当前总长度
block_tables = 每条 sequence 的逻辑 block 到物理 block 的映射
```

Attention 层可以据此调用带 KV Cache 的 attention kernel：

```text
flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    cache_seqlens=context_lens,
    block_table=block_tables,
    causal=True
)
```

输出：

```text
attention_output shape = [batch_size, num_heads, head_dim]
hidden_states shape = [batch_size, hidden_size]
logits shape = [batch_size, vocab_size]
```

然后 Sampler 根据 logits 采样下一个 token。

---

### 4.3 Prefill 和 Decode 中 Context 字段对比

| 字段 | Prefill 阶段含义 | Decode 阶段含义 |
|---|---|---|
| `is_prefill` | 标记走 prefill attention | 标记走 decode attention |
| `cu_seqlens_q` | query token 的变长边界 | 一般不用 |
| `cu_seqlens_k` | key/value token 的变长边界 | 一般不用 |
| `max_seqlen_q` | batch 内最大 query 长度 | 一般不用 |
| `max_seqlen_k` | batch 内最大 key/value 长度 | 一般不用 |
| `slot_mapping` | 多个 prompt token 写入 KV Cache 的位置 | 每条 sequence 新 token 写入 KV Cache 的位置 |
| `context_lens` | 一般不用 | 每条 sequence 当前上下文长度 |
| `block_tables` | prefix cache / paged attention 时使用 | decode 读取历史 KV Cache 时使用 |

---

### 4.4 batch、sequence、hidden size、head 维度在哪里体现

`context.py` 不直接保存 `hidden_size`、`num_heads` 或 `head_dim`，这些由模型配置和 Attention 层本身决定。但它保存了和 batch / sequence 相关的运行时信息。

| 概念 | 在哪里体现 |
|---|---|
| batch size | `cu_seqlens_q.numel() - 1` 或 `context_lens.shape[0]` |
| sequence 长度 | `cu_seqlens_q/k`、`context_lens` |
| hidden size | 不在 Context 中，由模型层 hidden_states 决定 |
| num heads | 不在 Context 中，由 Attention 初始化参数决定 |
| head dim | 不在 Context 中，由 Attention 初始化参数决定 |
| KV Cache 物理位置 | `slot_mapping`、`block_tables` |
| Prefill / Decode 分支 | `is_prefill` |

也就是说，`Context` 关心的是 **这一轮 batch 的序列组织方式和 KV Cache 映射方式**，而不是模型权重维度本身。

---

## 5. 和 Transformer / Qwen 模型结构的关系

### 5.1 它不是 Qwen 特有结构，但 Qwen 推理需要它

`context.py` 并不是 Qwen3 模型结构中的一层。无论是 Qwen、LLaMA，还是其他 decoder-only Transformer，只要使用类似 vLLM 的 paged KV Cache 推理方式，都需要类似的上下文信息。

Qwen3 的模型结构中仍然有：

```text
Embedding
Transformer Decoder Layers
RMSNorm
LMHead
```

每个 decoder layer 中有：

```text
Self-Attention
MLP
Residual
Norm
```

`context.py` 主要服务于 Self-Attention 和 LMHead。

---

### 5.2 它和 Self-Attention 的关系

标准自注意力公式是：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

训练时，`Q/K/V` 通常都来自当前 batch 的完整序列。

推理时，尤其是 decode 阶段：

```text
Q = 当前新 token 的 query
K/V = 历史 token 的 KV Cache + 当前新 token 的 K/V
```

因此 Attention 层必须知道：

1. 当前新 token 的 K/V 写到哪里；
2. 历史 K/V 从哪些 block 读取；
3. 每条 sequence 的历史长度是多少；
4. 当前 batch 中不同 sequence 的边界在哪里。

这些都不是 Attention 公式本身能表达的，所以由 `Context` 提供。

---

### 5.3 它和 KV Cache 的关系

KV Cache 的目的，是避免 decode 阶段重复计算历史 token 的 K/V。

没有 KV Cache 时，每生成一个 token 都要重新计算整个上下文：

```text
第 1 步：计算 token 0..n 的 K/V
第 2 步：计算 token 0..n+1 的 K/V
第 3 步：计算 token 0..n+2 的 K/V
```

这样会造成大量重复计算。

有 KV Cache 后：

```text
Prefill：一次性计算 prompt 的 K/V，并写入 cache
Decode：每步只计算新 token 的 K/V，然后追加到 cache
```

`context.py` 中和 KV Cache 最相关的字段是：

```text
slot_mapping
context_lens
block_tables
```

其中：

1. `slot_mapping`：负责写入；
2. `context_lens`：负责告诉 kernel 每条 sequence 读多长；
3. `block_tables`：负责告诉 kernel 去哪些物理 block 读取。

---

### 5.4 它和 PagedAttention 的关系

PagedAttention 的核心思想是：不要给每条 sequence 分配一整段连续 KV Cache，而是像操作系统分页一样，把 KV Cache 分成 block。

逻辑上，一条 sequence 的 KV 可能是：

```text
logical block 0
logical block 1
logical block 2
```

物理上，它们可能存放在：

```text
physical block 8
physical block 3
physical block 20
```

`block_tables` 保存的就是这个映射。

这让系统可以：

1. 更灵活地分配和释放 KV Cache；
2. 减少显存碎片；
3. 支持不同长度请求混合 batch；
4. 支持 prefix cache 复用；
5. 更接近 vLLM 的高吞吐推理机制。

---

### 5.5 它和 LMHead 的关系

Prefill 阶段，如果 prompt 有很多 token，模型 forward 会产生每个 token 的 hidden state：

```text
hidden_states shape = [total_prompt_tokens, hidden_size]
```

但是采样下一个 token 时，只需要每条请求最后一个 token 的 hidden state：

```text
每条 sequence 的最后一个位置
```

`cu_seqlens_q` 可以快速给出这些位置：

```text
last_indices = cu_seqlens_q[1:] - 1
```

例如：

```text
cu_seqlens_q = [0, 5, 8, 12]
last_indices = [4, 7, 11]
```

这样 LMHead 只对：

```text
hidden_states[4]
hidden_states[7]
hidden_states[11]
```

计算 logits，而不是对所有 prompt token 都算 logits。

这可以减少无用计算，尤其是长 prompt prefill 时非常重要。

---

## 6. 工程理解：为什么用全局 Context

### 6.1 好处

使用全局 `_CONTEXT` 的主要好处是让模型层接口保持简洁。

如果不用全局 Context，那么很多 layer 的 forward 都要传入一堆参数：

```python
forward(
    hidden_states,
    positions,
    slot_mapping,
    context_lens,
    block_tables,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    is_prefill,
)
```

这样会导致：

1. 模型代码不够接近 HuggingFace 原始结构；
2. 每一层都要传递大量推理引擎参数；
3. Attention 和 LMHead 以外的层也被迫感知这些参数；
4. CUDA Graph capture / replay 时代码更复杂。

全局 Context 可以让模型主干仍然像普通 Transformer 一样：

```python
model(input_ids, positions)
```

而 Attention / LMHead 在需要时自行读取上下文。

---

### 6.2 代价

全局 Context 也有代价：

1. 它是全局状态，必须确保每次 forward 前正确设置；
2. 每次 forward 后必须 reset；
3. 如果多线程共享同一个进程执行不同 forward，可能出现上下文污染；
4. 调试时需要注意隐式依赖，因为 Attention 的行为不只由显式参数 `q/k/v` 决定，还依赖当前 Context。

nano-vLLM 的实现相对简洁，通常由 ModelRunner 控制一次 forward 的生命周期，因此这种设计是可以接受的。

---

## 7. 初学者应该怎么理解这个文件

对 AI Infra 推理方向初学者来说，`context.py` 的重点不是 Python 语法，而是理解它背后的推理系统设计。

你可以这样记：

```text
context.py = 当前推理 step 的“运行时说明书”
```

它告诉模型内部的 Attention：

```text
现在是 prefill 还是 decode？
本 batch 的 sequence 边界在哪里？
新 K/V 写到 KV Cache 哪里？
历史 K/V 从哪些 block 读？
每条 sequence 当前长度是多少？
```

它告诉 LMHead：

```text
prefill 阶段应该取每条 prompt 的最后一个 token 来算 logits。
```

它告诉 CUDA Graph 相关逻辑：

```text
decode replay 时用哪批静态 buffer 和 batch 元信息。
```

因此，这个文件虽然代码很短，但它连接了：

```text
Scheduler
ModelRunner
Attention
KV Cache
PagedAttention
LMHead
CUDA Graph
```

是理解 nano-vLLM 推理流程时非常关键的工具文件。

---

## 8. 一句话总结

`context.py` 是 nano-vLLM 在一次模型 forward 期间保存 prefill/decode 状态、变长序列边界、KV Cache 写入位置、上下文长度和 block 映射的全局运行时上下文模块；它不直接计算模型结果，但决定 Attention 和 LMHead 如何在高性能推理场景下正确、高效地使用 KV Cache。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]

%% 项目关联导航：结束 %%
