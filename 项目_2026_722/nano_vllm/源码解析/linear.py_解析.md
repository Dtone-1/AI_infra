# linear.py 源码宏观解析

## 1. 文件整体定位

`linear.py` 是 nano-vLLM 中负责 **线性层与张量并行切分** 的核心 layer 文件。

普通 Transformer 里的 `Linear` 层本质上就是矩阵乘法：

```python
y = x @ W.T + b
```

但在大模型推理中，权重矩阵通常非常大，单张 GPU 放不下或者计算压力太大，所以 vLLM / nano-vLLM 会把部分线性层的权重切到多个 GPU 上。`linear.py` 做的就是这件事：它把普通线性层封装成适合 **Tensor Parallelism** 的并行线性层。

它位于模型 forward 的底层计算模块中，被 `models/qwen3.py` 直接使用：

- `QKVParallelLinear`：用于 Attention 里的 `q_proj / k_proj / v_proj`；
- `MergedColumnParallelLinear`：用于 MLP 里的 `gate_proj / up_proj`；
- `RowParallelLinear`：用于 Attention 的 `o_proj` 和 MLP 的 `down_proj`；
- `ReplicatedLinear`：用于不切分、每个 rank 都完整保存权重的普通线性层。

所以这个文件不是调度器、KV Cache 管理器，也不是模型结构定义文件，而是模型 forward 中高频出现的 **矩阵乘法算子封装层**。

## 2. 这个文件要解决的核心问题

`linear.py` 主要解决三个问题：

| 问题 | 说明 |
|---|---|
| 张量计算问题 | 统一封装 `F.linear(x, weight, bias)` |
| 权重加载问题 | 从 HuggingFace 原始权重中取出当前 GPU rank 需要的切片 |
| 并行切分问题 | 支持列并行、行并行、QKV 合并并行、MLP 合并并行 |

它对应原始 Transformer 结构中的所有线性投影部分：

- Attention 中的 `Q/K/V` 投影；
- Attention 输出投影 `O`；
- MLP 中的 `gate_proj / up_proj / down_proj`；
- 某些不需要并行切分的普通线性层。

它和高性能推理的关系非常直接：大模型推理中大部分计算都来自矩阵乘法，尤其是 Attention 和 MLP 中的线性层。如果这些线性层不能并行切分，模型就很难扩展到多卡推理。

在 nano-vLLM 里，张量并行的基本思路是：

1. 每个 GPU rank 只保存一部分权重；
2. 每个 rank 只计算自己负责的局部输出；
3. 必要时通过 `all_reduce` 或 `gather` 汇总结果；
4. 对上层模型来说，接口仍然像普通 `Linear` 一样使用。

这也是为什么 `linear.py` 看起来代码不长，但它支撑的是整个模型并行推理能力。

## 3. 代码结构总览

### 3.1 `divide`

```python
def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator
```

这是一个安全整除函数，用来确保 hidden size、head 数量、intermediate size 等维度可以被 `tp_size` 整除。

例如：

```python
num_heads = divide(total_num_heads, tp_size)
```

如果总 attention heads 数不能均匀分给每张 GPU，就会直接报错，避免后续 shape 错乱。

### 3.2 `LinearBase`

`LinearBase` 是所有并行线性层的基类。

核心变量：

| 变量 | 作用 |
|---|---|
| `tp_dim` | 当前层权重要沿哪个维度切分 |
| `tp_rank` | 当前进程 / GPU 在 tensor parallel group 中的编号 |
| `tp_size` | tensor parallel 的总进程数 / GPU 数 |
| `weight` | 当前 rank 持有的局部权重 |
| `bias` | 可选偏置 |

其中最关键的是：

```python
self.weight.weight_loader = self.weight_loader
```

这行代码给参数对象动态挂了一个 `weight_loader` 方法。模型加载 HuggingFace 权重时，可以通过这个方法把完整权重切成当前 rank 需要的局部权重。

`LinearBase.forward` 没有实现，因为不同并行方式的 forward 逻辑不同。

### 3.3 `ReplicatedLinear`

`ReplicatedLinear` 表示不做张量切分的线性层。

每个 rank 都保存完整权重：

```python
param.data.copy_(loaded_weight)
```

forward 就是普通线性层：

```python
return F.linear(x, self.weight, self.bias)
```

它适合权重较小、没有必要切分，或者某些需要在每个 rank 上完整复制的场景。

### 3.4 `ColumnParallelLinear`

`ColumnParallelLinear` 是 **列并行线性层**。

从 PyTorch 权重形状看，线性层权重是：

```text
weight: [output_size, input_size]
```

列并行的意思在 Megatron-LM / vLLM 语境中通常是：把输出维度切开，也就是每个 rank 负责一部分 output channels。

在代码中体现为：

```python
super().__init__(input_size, divide(output_size, tp_size), bias, 0)
```

也就是当前 rank 的权重形状变成：

```text
[output_size / tp_size, input_size]
```

权重加载时沿 `tp_dim = 0` 切：

```python
loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
```

forward 时每个 rank 都计算自己的局部输出：

```python
return F.linear(x, self.weight, self.bias)
```

注意这里没有 `all_reduce`，因为列并行得到的是输出维度的一部分。上层通常会让后续层继续消费这个局部结果，或者在特定位置再汇总。

### 3.5 `MergedColumnParallelLinear`

`MergedColumnParallelLinear` 是 Column Parallel 的增强版本，用于把多个线性层合并成一个权重矩阵。

典型场景是 Qwen3 MLP：

```python
self.gate_up_proj = MergedColumnParallelLinear(
    hidden_size,
    [intermediate_size] * 2,
    bias=False,
)
```

原始结构中本来有两个线性层：

```text
gate_proj: hidden_size -> intermediate_size
up_proj:   hidden_size -> intermediate_size
```

合并后变成一个线性层：

```text
gate_up_proj: hidden_size -> 2 * intermediate_size
```

这样可以减少 kernel launch 次数，也更适合一次矩阵乘法完成多个投影。

`loaded_shard_id` 用来说明当前加载的是合并权重中的哪一段：

- `0`：对应 `gate_proj`；
- `1`：对应 `up_proj`。

加载时先根据 `loaded_shard_id` 定位到当前投影在合并权重中的偏移，再按 TP rank 取切片。

### 3.6 `QKVParallelLinear`

`QKVParallelLinear` 专门用于 Attention 的 Q/K/V 合并投影。

在原始 Transformer 中通常有三个线性层：

```text
q_proj: hidden_size -> num_heads * head_dim
k_proj: hidden_size -> num_kv_heads * head_dim
v_proj: hidden_size -> num_kv_heads * head_dim
```

nano-vLLM 把它们合并成一个：

```text
qkv_proj: hidden_size -> (num_heads + 2 * num_kv_heads) * head_dim
```

这对应 `qwen3.py` 里的代码：

```python
qkv = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
```

`QKVParallelLinear` 内部会计算当前 rank 拥有多少个 Q heads 和 KV heads：

```python
self.num_heads = divide(total_num_heads, tp_size)
self.num_kv_heads = divide(total_num_kv_heads, tp_size)
```

加载权重时，`loaded_shard_id` 是字符串：

- `"q"`：加载 q_proj 的当前 rank 切片；
- `"k"`：加载 k_proj 的当前 rank 切片；
- `"v"`：加载 v_proj 的当前 rank 切片。

这个类非常关键，因为它把模型结构中的 Q/K/V 三个投影，和推理系统中的张量并行权重切分连接起来。

### 3.7 `RowParallelLinear`

`RowParallelLinear` 是 **行并行线性层**。

它和 Column Parallel 正好相反：不是切输出维度，而是切输入维度。

初始化时：

```python
super().__init__(divide(input_size, tp_size), output_size, bias, 1)
```

当前 rank 的权重形状为：

```text
[output_size, input_size / tp_size]
```

权重加载时沿 `tp_dim = 1` 切：

```python
loaded_weight = loaded_weight.narrow(1, start_idx, shard_size)
```

forward 时，每个 rank 先计算局部输出：

```python
y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
```

然后通过 `all_reduce` 把所有 rank 的局部输出相加：

```python
if self.tp_size > 1:
    dist.all_reduce(y)
```

为什么是相加？因为完整矩阵乘法可以拆成多个输入分片的局部乘法之和：

```text
x @ W.T
= [x0, x1] @ [W0, W1].T
= x0 @ W0.T + x1 @ W1.T
```

每个 rank 算其中一项，最后 `all_reduce(sum)` 得到完整输出。

在 Qwen3 中，`RowParallelLinear` 主要用于：

- Attention 的 `o_proj`；
- MLP 的 `down_proj`。

这正好对应常见的并行模式：

```text
ColumnParallelLinear -> RowParallelLinear
```

先把中间维度切开并行计算，再用 Row Parallel 把结果聚合回来。

## 4. 张量流和 forward 流程

### 4.1 普通线性层公式

所有类底层最终都调用：

```python
F.linear(x, weight, bias)
```

其含义是：

```text
x:      [..., input_size]
weight: [output_size, input_size]
output: [..., output_size]
```

前面的 `...` 可以是 token 维度，也可以是 batch/token 展平后的维度。

在 nano-vLLM 中，模型 forward 通常会把 batch 内 token 拉平成一维 token 列表，所以常见形状是：

```text
hidden_states: [num_tokens, hidden_size]
```

其中：

- `num_tokens`：当前推理步参与计算的 token 数；
- prefill 阶段通常是多个 prompt token；
- decode 阶段通常是每个 sequence 的一个新 token；
- `hidden_size`：模型隐藏层维度。

### 4.2 Column Parallel 的 shape

假设：

```text
hidden_states: [num_tokens, hidden_size]
weight:        [output_size / tp_size, hidden_size]
```

则每个 rank 输出：

```text
local_output: [num_tokens, output_size / tp_size]
```

不同 rank 拥有不同的输出通道。多个 rank 拼起来才是完整的：

```text
full_output: [num_tokens, output_size]
```

在 QKV 投影中，Column Parallel 会让每个 rank 拥有一部分 attention heads。

### 4.3 QKV Parallel 的 shape

假设：

```text
hidden_states: [num_tokens, hidden_size]
total_num_heads: H
total_num_kv_heads: H_kv
head_dim: D
tp_size: T
```

每个 rank 上：

```text
num_heads = H / T
num_kv_heads = H_kv / T
q_size = num_heads * D
kv_size = num_kv_heads * D
```

`qkv_proj` 输出：

```text
qkv: [num_tokens, q_size + 2 * kv_size]
```

然后切成：

```text
q: [num_tokens, q_size]
k: [num_tokens, kv_size]
v: [num_tokens, kv_size]
```

再 reshape：

```text
q: [num_tokens, num_heads, head_dim]
k: [num_tokens, num_kv_heads, head_dim]
v: [num_tokens, num_kv_heads, head_dim]
```

这个形状会继续传入 Attention 模块，与 RoPE、KV Cache、FlashAttention 结合。

### 4.4 Merged Column Parallel 的 shape

以 MLP 的 `gate_up_proj` 为例：

```text
input: [num_tokens, hidden_size]
```

完整输出原本是：

```text
gate: [num_tokens, intermediate_size]
up:   [num_tokens, intermediate_size]
```

合并后：

```text
gate_up: [num_tokens, 2 * intermediate_size / tp_size]
```

然后 `SiluAndMul` 会把它拆成两半：

```text
silu(gate) * up
```

得到：

```text
[num_tokens, intermediate_size / tp_size]
```

最后进入 `RowParallelLinear` 的 `down_proj` 聚合回：

```text
[num_tokens, hidden_size]
```

### 4.5 Row Parallel 的 shape

假设输入已经按 hidden/intermediate 维度分片：

```text
x:      [num_tokens, input_size / tp_size]
weight: [output_size, input_size / tp_size]
```

每个 rank 计算：

```text
local_y: [num_tokens, output_size]
```

然后所有 rank 做 `all_reduce(sum)`：

```text
y = sum(local_y across ranks)
```

最终每个 rank 都得到完整输出：

```text
[num_tokens, output_size]
```

## 5. 和 Transformer / Qwen 模型结构的关系

`linear.py` 可以直接映射到 Qwen3 Decoder Layer 的两个核心子模块。

### 5.1 Attention 部分

Qwen3Attention 中使用：

```python
self.qkv_proj = QKVParallelLinear(...)
self.o_proj = RowParallelLinear(...)
```

对应 Transformer：

```text
hidden_states
  -> q_proj / k_proj / v_proj
  -> attention
  -> o_proj
```

在 nano-vLLM 里变成：

```text
hidden_states
  -> QKVParallelLinear
  -> split q/k/v
  -> RoPE + Attention + KV Cache
  -> RowParallelLinear
```

这里的并行策略是：

- Q/K/V 投影按 head 维度切到不同 rank；
- 每个 rank 计算自己负责的 attention heads；
- `o_proj` 使用 Row Parallel 汇总多卡计算结果。

### 5.2 MLP 部分

Qwen3MLP 中使用：

```python
self.gate_up_proj = MergedColumnParallelLinear(...)
self.down_proj = RowParallelLinear(...)
```

对应 Transformer FFN / SwiGLU：

```text
hidden_states
  -> gate_proj, up_proj
  -> silu(gate) * up
  -> down_proj
```

在 nano-vLLM 里变成：

```text
hidden_states
  -> MergedColumnParallelLinear
  -> SiluAndMul
  -> RowParallelLinear
```

这里的优化点是：

- `gate_proj` 和 `up_proj` 合并成一次线性计算；
- 中间层 `intermediate_size` 很大，适合按列切分；
- `down_proj` 再通过行并行把局部结果加和回 hidden size。

### 5.3 和权重加载的关系

这个文件还有一个容易被初学者忽略的重点：它不只是 forward 计算，还负责权重如何从原始模型文件加载到并行权重中。

例如 HuggingFace 权重中可能是：

```text
q_proj.weight: [num_heads * head_dim, hidden_size]
k_proj.weight: [num_kv_heads * head_dim, hidden_size]
v_proj.weight: [num_kv_heads * head_dim, hidden_size]
```

但 nano-vLLM 运行时需要的是合并后的：

```text
qkv_proj.weight: [(num_heads + 2 * num_kv_heads) / tp_size * head_dim, hidden_size]
```

所以 `QKVParallelLinear.weight_loader` 要根据 `"q" / "k" / "v"` 把原始权重放进合并矩阵的正确位置。

同理，MLP 中的：

```text
gate_proj.weight
up_proj.weight
```

要放入：

```text
gate_up_proj.weight
```

这就是 `MergedColumnParallelLinear.weight_loader` 的作用。

## 6. 学习总结

`linear.py` 的核心价值可以概括为一句话：

> 它把 Transformer 中普通的线性层，改造成适合 nano-vLLM 多卡推理的张量并行线性层。

学习这个文件时要抓住三个主线：

1. **计算主线**：所有层本质上都是 `F.linear`；
2. **切分主线**：Column Parallel 切输出维度，Row Parallel 切输入维度；
3. **加载主线**：HuggingFace 原始权重需要被切片、合并并放入当前 rank 的局部参数。

对 AI Infra 推理方向来说，这个文件非常重要，因为它连接了三层知识：

| 层次 | 对应内容 |
|---|---|
| Transformer 原理 | QKV 投影、输出投影、MLP 投影 |
| PyTorch 实现 | `nn.Parameter`、`F.linear`、shape 变换 |
| 推理系统优化 | Tensor Parallel、权重切分、`all_reduce`、多卡扩展 |

如果只从模型结构角度看，它只是几个 Linear 类；但从推理引擎角度看，它是模型能够在多 GPU 上运行的基础设施之一。



%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]

%% 项目关联导航：结束 %%
