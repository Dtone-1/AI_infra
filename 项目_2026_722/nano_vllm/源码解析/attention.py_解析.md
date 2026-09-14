# attention.py 源码宏观解析

## 1. 文件整体定位

`attention.py` 是 nano-vLLM 中负责 **Attention 核心计算、KV Cache 写入、prefill/decode 分支执行** 的 layer 文件。

它在 `models/qwen3.py` 中被 `Qwen3Attention` 使用：

```python
self.attn = Attention(
    self.num_heads,
    self.head_dim,
    self.scaling,
    self.num_kv_heads,
)
```

在 Qwen3Attention 的 forward 中：

```python
q, k = self.rotary_emb(positions, q, k)
o = self.attn(q, k, v)
output = self.o_proj(o.flatten(1, -1))
```

所以它位于 Attention 子模块的核心位置：

```text
hidden_states
  -> qkv_proj
  -> split q/k/v
  -> q/k RMSNorm(optional)
  -> RoPE(q/k)
  -> Attention(q/k/v + KV Cache)
  -> o_proj
```

它不是普通手写的 `q @ k^T` attention，而是调用 FlashAttention 相关接口：

- `flash_attn_varlen_func`：用于 prefill 阶段的变长序列 attention；
- `flash_attn_with_kvcache`：用于 decode 阶段结合 KV Cache 的 attention；
- Triton kernel：用于把当前 step 的 K/V 写入 Paged KV Cache。

这个文件是 nano-vLLM 从“模型结构”走向“推理引擎”的关键连接点。

## 2. 这个文件要解决的核心问题

`attention.py` 主要解决四类问题：

| 问题 | 说明 |
|---|---|
| 模型结构问题 | 实现 Transformer/Qwen 的 Self-Attention 计算 |
| 张量计算问题 | 使用 FlashAttention 计算 q/k/v 注意力输出 |
| KV Cache 问题 | 将新的 K/V 写入 cache，decode 时复用历史 K/V |
| 推理阶段分支问题 | prefill 和 decode 使用不同 attention kernel |

它不负责：

- Q/K/V 线性投影，这部分在 `linear.py` 的 `QKVParallelLinear`；
- RoPE 位置编码，这部分在 `rotary_embedding.py`；
- 输出投影 `o_proj`，这部分是 `RowParallelLinear`；
- 请求调度和 block 分配，这部分在 engine 层。

它和原始 Transformer 的对应关系是：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

但在推理引擎中，Attention 不能只按这个公式写。原因是：

1. prompt 长度不同，需要支持变长 batch；
2. decode 阶段不能每步重复计算全部历史 token 的 K/V；
3. KV Cache 用 block/page 方式管理，不是简单连续大矩阵；
4. prefix cache 时，部分 prefix 已经在 KV Cache 中；
5. 高性能推理要使用 FlashAttention，而不是普通 PyTorch attention。

所以 `attention.py` 的真正目标是：在保持 Transformer Attention 语义的前提下，用适合推理服务的方式高效执行 prefill 和 decode。

## 3. 代码结构总览

### 3.1 文件导入

```python
import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context
```

含义如下：

| 导入 | 作用 |
|---|---|
| `torch` / `nn` | PyTorch 张量与 Module |
| `triton` / `tl` | 编写自定义 GPU kernel |
| `flash_attn_varlen_func` | prefill 变长序列 FlashAttention |
| `flash_attn_with_kvcache` | decode 阶段使用 KV Cache 的 FlashAttention |
| `get_context` | 获取当前推理批次的元信息 |

这里最重要的是 `get_context()`。

`attention.py` 的 forward 只接收 `q/k/v`，但它还需要知道：

- 当前是 prefill 还是 decode；
- 每条 sequence 的长度；
- KV Cache 写入位置；
- block table；
- prefix cache 是否存在。

这些信息都不从函数参数传入，而是通过全局 context 读取。

### 3.2 `store_kvcache_kernel`

```python
@triton.jit
def store_kvcache_kernel(...):
```

这是一个 Triton GPU kernel，用来把当前 token 的 `key/value` 写入 KV Cache。

核心逻辑：

```python
idx = tl.program_id(0)
slot = tl.load(slot_mapping_ptr + idx)
if slot == -1: return
```

每个 program 负责一个 token。`idx` 表示当前是第几个 token，`slot` 表示这个 token 的 K/V 应该写入 KV Cache 的哪个物理位置。

然后读取当前 token 的 key/value：

```python
key_offsets = idx * key_stride + tl.arange(0, D)
value_offsets = idx * value_stride + tl.arange(0, D)
key = tl.load(key_ptr + key_offsets)
value = tl.load(value_ptr + value_offsets)
```

再写入 cache：

```python
cache_offsets = slot * D + tl.arange(0, D)
tl.store(k_cache_ptr + cache_offsets, key)
tl.store(v_cache_ptr + cache_offsets, value)
```

其中：

```text
D = num_kv_heads * head_dim
```

所以它每次写入的是某个 token 的完整 K 向量和 V 向量。

### 3.3 `store_kvcache`

```python
def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
```

这是 Python 包装函数，负责做 shape/stride 检查，然后启动 Triton kernel。

输入 shape 通常是：

```text
key:   [N, num_kv_heads, head_dim]
value: [N, num_kv_heads, head_dim]
```

其中：

```text
N = 当前 step 需要写入 cache 的 token 数
```

KV Cache 的逻辑形状来自 `model_runner.py`：

```python
self.kv_cache = torch.empty(
    2,
    num_hidden_layers,
    num_kvcache_blocks,
    block_size,
    num_kv_heads,
    head_dim,
)
```

分配给某一层后：

```text
k_cache: [num_kvcache_blocks, block_size, num_kv_heads, head_dim]
v_cache: [num_kvcache_blocks, block_size, num_kv_heads, head_dim]
```

`slot_mapping` 的含义是：

```text
第 i 个 token -> KV Cache 中第 slot_mapping[i] 个物理 token slot
```

物理 slot 可以理解为：

```text
slot = block_id * block_size + block_offset
```

这样 Triton kernel 就可以把 `[N, num_kv_heads, head_dim]` 的 K/V 写入分块 KV Cache 中。

### 3.4 `Attention`

文件中最重要的类是：

```python
class Attention(nn.Module):
```

初始化参数：

| 参数 | 含义 |
|---|---|
| `num_heads` | 当前 TP rank 上的 query heads 数 |
| `head_dim` | 每个 head 的维度 |
| `scale` | attention softmax 缩放因子，通常是 `head_dim ** -0.5` |
| `num_kv_heads` | 当前 TP rank 上的 key/value heads 数 |

初始化中：

```python
self.k_cache = self.v_cache = torch.tensor([])
```

一开始 cache 是空 tensor。真正的 KV Cache 在 `ModelRunner.allocate_kv_cache()` 中分配，然后挂到每一层 Attention 模块上：

```python
module.k_cache = self.kv_cache[0, layer_id]
module.v_cache = self.kv_cache[1, layer_id]
```

也就是说，Attention 模块本身不负责申请整块 KV Cache，它只负责使用已经分配好的 cache。

## 4. 张量流和 forward 流程

### 4.1 Attention.forward 的输入输出

函数签名：

```python
def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
```

输入来自 Qwen3Attention：

```text
q: [num_tokens, num_heads, head_dim]
k: [num_tokens, num_kv_heads, head_dim]
v: [num_tokens, num_kv_heads, head_dim]
```

输出：

```text
o: [num_tokens, num_heads, head_dim]
```

随后在 Qwen3Attention 中：

```python
output = self.o_proj(o.flatten(1, -1))
```

`o.flatten(1, -1)` 会把 heads 和 head_dim 合并：

```text
[num_tokens, num_heads, head_dim]
  -> [num_tokens, num_heads * head_dim]
```

再进入 `o_proj` 投影回 hidden size。

### 4.2 forward 的第一步：获取 context

```python
context = get_context()
k_cache, v_cache = self.k_cache, self.v_cache
```

`context` 由 `model_runner.py` 在每次模型运行前设置。

prefill 阶段设置的信息包括：

| context 字段 | 含义 |
|---|---|
| `is_prefill=True` | 当前是 prefill |
| `cu_seqlens_q` | 每条 sequence 的 query 累积长度 |
| `cu_seqlens_k` | 每条 sequence 的 key 累积长度 |
| `max_seqlen_q` | 本 batch 最大 query 长度 |
| `max_seqlen_k` | 本 batch 最大 key 长度 |
| `slot_mapping` | 当前 token 写入 KV Cache 的物理位置 |
| `block_tables` | prefix cache 情况下的 block 映射表 |

decode 阶段设置的信息包括：

| context 字段 | 含义 |
|---|---|
| `is_prefill=False` | 当前是 decode |
| `slot_mapping` | 当前新 token 写入 KV Cache 的位置 |
| `context_lens` | 每条 sequence 当前上下文长度 |
| `block_tables` | 每条 sequence 对应的 KV Cache block 列表 |

### 4.3 forward 的第二步：写入 KV Cache

```python
if k_cache.numel() and v_cache.numel():
    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
```

只要 KV Cache 已经分配，就把当前传入的 `k/v` 写入 cache。

这一步在 prefill 和 decode 中都会发生：

- prefill：把 prompt token 的 K/V 写入 KV Cache；
- decode：把当前新 token 的 K/V 写入 KV Cache。

这样后续 decode 就不需要重新计算历史 token 的 K/V。

### 4.4 prefill 路径

代码：

```python
if context.is_prefill:
    if context.block_tables is not None:    # prefix cache
        k, v = k_cache, v_cache
    o = flash_attn_varlen_func(
        q, k, v,
        max_seqlen_q=context.max_seqlen_q,
        cu_seqlens_q=context.cu_seqlens_q,
        max_seqlen_k=context.max_seqlen_k,
        cu_seqlens_k=context.cu_seqlens_k,
        softmax_scale=self.scale,
        causal=True,
        block_table=context.block_tables,
    )
```

prefill 阶段的特点是：一次性处理 prompt 中的一段或全部 token。

因为不同请求 prompt 长度不同，不能简单拼成固定矩形 batch，所以使用 `flash_attn_varlen_func`，通过 `cu_seqlens_q/cu_seqlens_k` 表示每条 sequence 的边界。

例如两条 sequence 的 prefill token 长度分别是 3 和 5：

```text
q 被展平成 8 个 token
cu_seqlens_q = [0, 3, 8]
```

FlashAttention 根据 `cu_seqlens_q` 知道：

```text
第 0 条 sequence: token [0, 3)
第 1 条 sequence: token [3, 8)
```

`causal=True` 表示因果 mask，每个 token 只能看见自己和之前的 token，不能看未来 token。

### 4.5 prefill 中的 prefix cache 分支

```python
if context.block_tables is not None:    # prefix cache
    k, v = k_cache, v_cache
```

当存在 prefix cache 时，说明当前请求的一部分前缀 token 已经在 KV Cache 中，不需要重新把这部分 K/V 当作普通输入张量传给 FlashAttention。

这时 `k/v` 改为整个 `k_cache/v_cache`，再配合：

```python
block_table=context.block_tables
```

让 FlashAttention 根据 block table 从 KV Cache 里读取历史 prefix 的 K/V。

这就是 prefix caching 和 paged KV Cache 结合的地方。

在 `model_runner.py` 中可以看到：

```python
if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
    block_tables = self.prepare_block_tables(seqs)
```

含义是：如果 K 的总长度大于当前 Q 的总长度，说明有一部分历史 prefix K 已经缓存，需要通过 block table 访问。

### 4.6 decode 路径

代码：

```python
else:    # decode
    o = flash_attn_with_kvcache(
        q.unsqueeze(1),
        k_cache,
        v_cache,
        cache_seqlens=context.context_lens,
        block_table=context.block_tables,
        softmax_scale=self.scale,
        causal=True,
    )
```

decode 阶段的特点是：每条 sequence 每次只生成一个新 token。

此时输入：

```text
q: [num_sequences, num_heads, head_dim]
```

`flash_attn_with_kvcache` 需要 query 带 sequence length 维度，所以代码做：

```python
q.unsqueeze(1)
```

变成：

```text
q: [num_sequences, 1, num_heads, head_dim]
```

这里的 `1` 表示每条 sequence 当前只有一个 query token。

decode 的 K/V 不再来自当前 `k/v` 输入，而是来自完整 KV Cache：

```text
k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
```

同时使用：

```python
cache_seqlens=context.context_lens
block_table=context.block_tables
```

来告诉 kernel：

- 每条 sequence 当前上下文长度是多少；
- 每条 sequence 的逻辑 token 对应哪些物理 KV Cache blocks。

最终输出仍然是：

```text
o: [num_sequences, num_heads, head_dim]
```

### 4.7 prefill 和 decode 的核心区别

| 对比项 | prefill | decode |
|---|---|---|
| 处理 token | prompt 中的一段或全部 token | 每条 sequence 当前 1 个新 token |
| q shape | `[total_tokens, num_heads, head_dim]` | `[num_seqs, num_heads, head_dim]` |
| attention kernel | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |
| 是否用 KV Cache | 会写入 cache；prefix cache 时也会读取 cache | 主要从 cache 读取历史 K/V |
| 长度信息 | `cu_seqlens_q/k`, `max_seqlen_q/k` | `context_lens`, `block_tables` |
| 目标 | 处理 prompt，得到首 token logits | 逐 token 生成后续输出 |

这也是你之前问的 prefill/decode 边界：`attention.py` 里这两个分支就是非常直接的代码边界。

## 5. 和 Transformer / Qwen 模型结构的关系

### 5.1 和标准 Self-Attention 的关系

标准 Attention 公式是：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

`attention.py` 的 `Attention.forward` 仍然在做这件事，只是没有手写矩阵乘法和 softmax，而是交给 FlashAttention：

```python
flash_attn_varlen_func(...)
flash_attn_with_kvcache(...)
```

这样可以减少显存读写，提高 attention 计算效率。

### 5.2 和 Qwen3Attention 的关系

Qwen3Attention 的完整流程可以写成：

```text
hidden_states
  -> QKVParallelLinear
  -> q/k/v
  -> q/k RMSNorm(optional)
  -> RoPE(q/k)
  -> Attention(q/k/v)
  -> RowParallelLinear(o_proj)
```

`attention.py` 只负责其中：

```text
Attention(q, k, v)
```

但在推理系统中，这一步不仅是数学 Attention，还包含：

```text
KV Cache 写入
prefill/decode 分支
prefix cache block table
FlashAttention kernel 选择
```

所以它比普通模型定义里的 `self_attn` 更接近推理引擎核心。

### 5.3 和 KV Cache 的关系

KV Cache 的意义是：decode 阶段每次只计算新 token 的 K/V，历史 token 的 K/V 直接从 cache 读。

如果没有 KV Cache，生成第 1000 个 token 时，模型可能需要重新计算前 999 个 token 的 K/V，代价非常高。

有 KV Cache 后：

```text
prefill:
  prompt tokens -> compute K/V -> store KV Cache

decode step t:
  new token -> compute new K/V -> store KV Cache
  query attends to all cached K/V
```

`attention.py` 中：

```python
store_kvcache(...)
```

负责写入；

```python
flash_attn_with_kvcache(...)
```

负责读取并计算 decode attention。

### 5.4 和 Paged Attention 的关系

nano-vLLM 的 KV Cache 不是每条 sequence 一个连续大数组，而是分成固定大小 block：

```text
[num_blocks, block_size, num_kv_heads, head_dim]
```

每条 sequence 通过 `block_table` 记录自己使用了哪些物理 blocks。

这和 vLLM 的 PagedAttention 思想是一致的：把 KV Cache 像分页内存一样管理，避免不同长度请求造成大量连续内存浪费。

在 `attention.py` 中，Paged KV Cache 主要通过两个变量体现：

| 变量 | 作用 |
|---|---|
| `slot_mapping` | 当前 token 的 K/V 写入哪个物理 slot |
| `block_tables` | 每条 sequence 的逻辑 blocks 对应哪些物理 blocks |

`store_kvcache` 使用 `slot_mapping` 写入；

FlashAttention 使用 `block_table` 读取。

### 5.5 和 GQA/MQA 的关系

Attention 初始化时有：

```python
num_heads
num_kv_heads
```

这说明 query heads 和 key/value heads 可以不同。

在 Qwen3 中常见的是 GQA，也就是：

```text
num_attention_heads >= num_key_value_heads
```

这可以减少 KV Cache 显存占用，因为 K/V heads 更少。

在 shape 上体现为：

```text
q: [num_tokens, num_heads, head_dim]
k: [num_tokens, num_kv_heads, head_dim]
v: [num_tokens, num_kv_heads, head_dim]
```

FlashAttention 内部会处理 query heads 到 kv heads 的对应关系。

这也是 KV Cache 显存公式里为什么使用 `num_kv_heads`，而不是 `num_heads`：

```text
KV Cache size ∝ 2 * num_layers * num_blocks * block_size * num_kv_heads * head_dim
```

### 5.6 和推理性能的关系

`attention.py` 直接影响推理性能：

1. prefill 阶段：prompt 很长，attention 计算量大，使用 FlashAttention varlen；
2. decode 阶段：每步 token 少，但要频繁访问 KV Cache，显存带宽和 cache 访问很关键；
3. KV Cache 写入：每个新 token 都要写 K/V；
4. block table：决定 cache 读取是否能正确映射到物理 blocks；
5. prefix cache：复用已有 prefix，减少重复 prefill。

这就是为什么 AI Infra 面试里经常问：

- prefill 和 decode 有什么区别；
- KV Cache 怎么组织；
- PagedAttention 是什么；
- 为什么 decode 往往受显存带宽影响；
- FlashAttention 在推理中解决什么问题。

`attention.py` 把这些问题都压缩进了一个很短但很关键的文件。

## 6. 学习总结

`attention.py` 的核心价值可以概括为一句话：

> 它把 Qwen3 的 Self-Attention 计算接入 FlashAttention 和 Paged KV Cache，实现了 prefill、decode、prefix cache 三种推理场景下的高效注意力计算。

学习这个文件要抓住四条主线：

1. **模型结构主线**：它接收已经投影、归一化、RoPE 后的 `q/k/v`，输出 attention result；
2. **KV Cache 主线**：`store_kvcache` 把当前 K/V 写入 block 化 cache，decode 从 cache 读历史 K/V；
3. **prefill/decode 主线**：prefill 用 `flash_attn_varlen_func`，decode 用 `flash_attn_with_kvcache`；
4. **PagedAttention 主线**：`slot_mapping` 负责写入位置，`block_tables` 负责读取映射。

它和前面几个文件的关系可以这样理解：

| 文件 | 负责内容 | 和 `attention.py` 的关系 |
|---|---|---|
| `linear.py` | 生成 q/k/v 和 o_proj | Attention 前后线性层 |
| `rotary_embedding.py` | 对 q/k 注入位置信息 | Attention 前处理 |
| `context.py` | 保存当前批次元信息 | Attention 读取 prefill/decode/cache 信息 |
| `block_manager.py` | 管理 KV Cache blocks | 为 block_tables/slot_mapping 提供基础 |
| `attention.py` | FlashAttention + KV Cache | Attention 核心执行 |

对 AI Infra 推理学习来说，这个文件是必须重点理解的，因为它已经不只是“模型层代码”，而是模型计算、KV Cache、调度元信息、FlashAttention kernel 之间的交汇点。



%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]

%% 项目关联导航：结束 %%
