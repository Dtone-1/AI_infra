# qwen3.6_linear.py_对比分析

> 对比对象：  
> 1. 原版 `nano-vLLM` 的 `linear.py`  
> 2. `nano-vllm-qwen3.6` 中对应的 `linear.py`
>
> 本文目标：从整体工程角度分析 qwen3.6 版本相比原版 `linear.py` 做了哪些修改、为什么这样改，以及这些修改在 Qwen3.6 / 推理系统 / 权重量化加载中的作用。

---

## 1. 文件整体定位

`linear.py` 是 nano-vLLM 中所有核心线性层的基础实现文件。

在 Transformer / Qwen3 / Qwen3.6 推理中，大量计算本质都是线性层：

```text
Embedding 后的 hidden_states
  ↓
QKV projection
  ↓
Attention output projection
  ↓
MLP gate/up projection
  ↓
MLP down projection
  ↓
LM Head
```

其中 `linear.py` 主要负责下面这些工程问题：

```text
1. 普通线性层如何执行
2. Tensor Parallel 下权重如何切分
3. HuggingFace 原始权重如何加载到 nano-vLLM 合并后的权重矩阵里
4. QKV / gate-up 这种合并线性层如何处理分片
5. Row Parallel 输出后如何 all_reduce
```

所以这个文件不是直接实现 Attention 公式，也不是直接实现 GatedDeltaNet。但它是模型 forward 中所有大矩阵乘法的底层支撑模块。

---

## 2. qwen3.6 版本整体变化概览

相比原版 `linear.py`，qwen3.6 版本的核心变化可以概括为一句话：

```text
在线性层权重加载阶段新增 FP8 权重反量化支持，并保证 tensor parallel 切片后 scale 对齐正确。
```

具体变化如下：

| 改动点 | 原版 nano-vLLM | qwen3.6 版本 | 作用 |
|---|---|---|---|
| 新增导入 | 无 | `from nanovllm.utils.quant import maybe_dequant_fp8_weight` | 引入 FP8 权重反量化工具 |
| `weight_loader` 参数 | 只接收 `loaded_weight` | 新增 `loaded_scale: torch.Tensor | None = None` | 支持加载带 scale 的 FP8 权重 |
| ReplicatedLinear | 直接 copy 权重 | copy 前调用 `maybe_dequant_fp8_weight` | 支持非 TP 普通线性层的 FP8 权重 |
| ColumnParallelLinear | 先按输出维切片，再 copy | 切片后根据 row/col offset 反量化 | 支持列并行权重分片的 FP8 scale 对齐 |
| MergedColumnParallelLinear | 用 `chunk(tp_size)` 取当前 rank 权重 | 改为 `narrow` 切片并反量化 | 支持 gate/up 合并权重的 FP8 分片加载 |
| QKVParallelLinear | 用 `chunk(tp_size)` 取 q/k/v 当前 rank | 改为 `narrow` 切片并反量化 | 支持 Q/K/V 合并权重的 FP8 分片加载 |
| RowParallelLinear | 按输入维切片后 copy | 切片后根据 row/col offset 反量化 | 支持行并行权重分片的 FP8 scale 对齐 |
| forward 计算 | `F.linear` / all_reduce | 基本不变 | FP8 只影响加载，不改变推理 forward 代码路径 |

---

## 3. 最关键新增：`maybe_dequant_fp8_weight`

### 3.1 原版没有量化处理

原版 `linear.py` 中，权重加载基本都是：

```python
param.data.copy_(loaded_weight)
```

或者先根据 tensor parallel 切出当前 rank 的 shard：

```python
loaded_weight = loaded_weight.narrow(...)
param_data.copy_(loaded_weight)
```

这说明原版默认 HuggingFace / safetensors 中读出来的权重就是可以直接复制到模型参数里的浮点权重，例如：

```text
FP16
BF16
FP32
```

### 3.2 qwen3.6 版本新增导入

qwen3.6 版本新增：

```python
from nanovllm.utils.quant import maybe_dequant_fp8_weight
```

这说明 qwen3.6 版本考虑了另一种情况：

```text
磁盘上的 loaded_weight 可能是 FP8 量化权重
```

FP8 权重不能总是直接 copy 到模型参数中。它通常需要配合 scale 做反量化：

```text
FP8 weight + scale
  ↓
dequant
  ↓
FP16 / BF16 / FP32 weight
  ↓
copy 到模型参数
```

所以 `maybe_dequant_fp8_weight` 的作用可以理解为：

```text
如果 loaded_weight 是 FP8 格式，就根据 loaded_scale 反量化；
如果不是 FP8，就原样返回或做最小处理。
```

---

## 4. `weight_loader` 统一新增 `loaded_scale`

### 4.1 原版接口

原版各类 `weight_loader` 只接收权重本身。

例如：

```python
def weight_loader(self, param, loaded_weight):
    ...
```

对于合并层，额外接收 shard id：

```python
def weight_loader(self, param, loaded_weight, loaded_shard_id):
    ...
```

### 4.2 qwen3.6 版本接口

qwen3.6 版本统一增加：

```python
loaded_scale: torch.Tensor | None = None
```

例如：

```python
def weight_loader(
    self,
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    loaded_scale: torch.Tensor | None = None,
):
```

以及：

```python
def weight_loader(
    self,
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    loaded_shard_id: str,
    loaded_scale: torch.Tensor | None = None,
):
```

### 4.3 这个改动的意义

这说明权重加载器不再只加载一个 tensor，而是可以加载：

```text
weight tensor
scale tensor
```

在 FP8 权重量化中，scale 是必需元数据。因为 FP8 的数值范围和精度都很有限，实际权重通常不是简单地：

```text
真实权重 = FP8 数值
```

而是类似：

```text
真实权重 ≈ FP8 数值 × scale
```

不同实现可能按 tensor、按 channel、按 block 使用不同粒度的 scale。

因此 `linear.py` 中所有权重加载路径都必须能接收 `loaded_scale`，否则即使 loader 读到了 scale，也无法传给线性层做正确反量化。

---

## 5. 改动一：ReplicatedLinear 支持 FP8 反量化

### 5.1 原版

原版：

```python
def weight_loader(self, param, loaded_weight):
    param.data.copy_(loaded_weight)
```

`ReplicatedLinear` 是每张卡都保存完整权重的普通线性层。

### 5.2 qwen3.6 版本

qwen3.6：

```python
loaded_weight = maybe_dequant_fp8_weight(loaded_weight, loaded_scale)
param.data.copy_(loaded_weight)
```

### 5.3 作用

如果某个普通线性层权重是 FP8 格式，则在 copy 到模型参数前先反量化。

这保证普通非并行线性层也能加载 FP8 checkpoint。

不过在 Qwen 类大模型中，最关键的通常还是下面几类并行线性层：

```text
ColumnParallelLinear
MergedColumnParallelLinear
QKVParallelLinear
RowParallelLinear
```

---

## 6. 改动二：ColumnParallelLinear 支持切片后反量化

### 6.1 原版逻辑

原版 `ColumnParallelLinear` 会按输出维度切分权重：

```python
shard_size = param_data.size(self.tp_dim)
start_idx = self.tp_rank * shard_size
loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
param_data.copy_(loaded_weight)
```

`tp_dim = 0`，也就是按权重矩阵的第 0 维切：

```text
weight: [output_size, input_size]
按 output_size 切分
```

每张 GPU 只保存一部分输出通道。

### 6.2 qwen3.6 版本新增逻辑

qwen3.6 版本在切片之后增加：

```python
row_start = start_idx if self.tp_dim == 0 else 0
col_start = start_idx if self.tp_dim == 1 else 0
loaded_weight = maybe_dequant_fp8_weight(
    loaded_weight, loaded_scale, row_start, col_start
)
```

### 6.3 为什么需要 row_start / col_start

FP8 scale 可能不是整个权重一个 scale，而是可能按块、按行、按列或二维 block 存储。

当 tensor parallel 切出当前 rank 的权重 shard 后，当前 shard 在原始完整权重矩阵中的位置不是从 `(0, 0)` 开始，而是有偏移的。

例如 Column Parallel：

```text
完整权重: [output_size, input_size]
rank 0: 第 0 到第 shard_size-1 行
rank 1: 第 shard_size 到第 2*shard_size-1 行
```

所以 rank 1 的权重 shard 在完整矩阵中的 row offset 是：

```text
row_start = shard_size
```

如果反量化 scale 是按原始矩阵坐标组织的，那么反量化函数必须知道：

```text
当前 shard 对应原矩阵的哪一段行/列
```

这就是 `row_start` / `col_start` 的意义。

---

## 7. 改动三：MergedColumnParallelLinear 支持 FP8 分片加载

### 7.1 MergedColumnParallelLinear 的作用

`MergedColumnParallelLinear` 用于把多个 column parallel 线性层合并成一个大线性层。

典型场景是 MLP 中的：

```text
gate_proj + up_proj -> gate_up_proj
```

概念上：

```text
gate = gate_proj(x)
up   = up_proj(x)
```

工程上合并成：

```text
gate_up = gate_up_proj(x)
```

这样可以减少 kernel launch，并更方便 tensor parallel 切分。

### 7.2 原版加载方式

原版使用：

```python
loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
```

也就是说：

```text
先把完整 loaded_weight 按 tp_size chunk
再取当前 rank 的那一块
```

### 7.3 qwen3.6 版本加载方式

qwen3.6 改成：

```python
start_idx = self.tp_rank * shard_size
loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
...
loaded_weight = maybe_dequant_fp8_weight(
    loaded_weight, loaded_scale, row_start, col_start
)
```

### 7.4 为什么从 chunk 改成 narrow

`chunk()` 对普通 FP16/BF16 权重通常没问题。

但 FP8 反量化需要知道当前 shard 在原始矩阵中的精确 offset。

`narrow()` 更直接保留了：

```text
start_idx
shard_size
```

方便传给：

```python
maybe_dequant_fp8_weight(..., row_start, col_start)
```

这说明 qwen3.6 版本不只是“能切片”，还需要“知道切片来自原矩阵哪里”。

### 7.5 对 Qwen MLP 的意义

Qwen MLP 中 `gate_up_proj` 对应两个投影：

```text
gate_proj
up_proj
```

原始 HuggingFace 权重通常是分开的：

```text
gate_proj.weight
up_proj.weight
```

nano-vLLM 会把它们加载到一个合并参数里。

qwen3.6 版本让这个加载过程支持：

```text
gate_proj FP8 weight + scale
up_proj FP8 weight + scale
```

并且在 tensor parallel 下正确放入当前 rank 对应的 shard。

---

## 8. 改动四：QKVParallelLinear 支持 FP8 分片加载

### 8.1 QKVParallelLinear 的作用

`QKVParallelLinear` 用于 Attention 中的：

```text
q_proj
k_proj
v_proj
```

概念上它们是三个线性层：

```text
Q = X Wq
K = X Wk
V = X Wv
```

工程上合并成一个：

```text
qkv = qkv_proj(x)
```

再 split 成 q/k/v。

### 8.2 原版加载方式

原版根据 `loaded_shard_id` 判断当前加载的是 q、k、v：

```python
if loaded_shard_id == "q":
    shard_size = self.num_heads * self.head_size
    shard_offset = 0
elif loaded_shard_id == "k":
    shard_size = self.num_kv_heads * self.head_size
    shard_offset = self.num_heads * self.head_size
else:
    shard_size = self.num_kv_heads * self.head_size
    shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
```

然后再按 tensor parallel rank 取当前 shard。

### 8.3 qwen3.6 版本新增

qwen3.6 在 q/k/v 当前 rank shard 切出来后，调用：

```python
loaded_weight = maybe_dequant_fp8_weight(
    loaded_weight, loaded_scale, row_start, col_start
)
```

### 8.4 对推理系统的意义

Attention 的 QKV projection 是大模型推理中非常核心的矩阵乘法之一。

支持 FP8 checkpoint 加载后，Qwen3.6 项目可以处理如下形式的权重：

```text
q_proj.weight: FP8
q_proj.weight_scale: scale
k_proj.weight: FP8
k_proj.weight_scale: scale
v_proj.weight: FP8
v_proj.weight_scale: scale
```

然后在加载时反量化成实际计算使用的 dtype。

这为模型压缩、显存节省、加载 FP8 权重格式提供了基础。

---

## 9. 改动五：RowParallelLinear 支持 FP8 反量化

### 9.1 RowParallelLinear 的作用

`RowParallelLinear` 通常用于：

```text
Attention o_proj
MLP down_proj
```

它按输入维度切分权重。

也就是每张卡拿输入特征的一部分，计算局部输出，然后所有卡做：

```python
dist.all_reduce(y)
```

得到完整输出。

### 9.2 原版逻辑

原版中，如果是 bias：

```python
if param_data.ndim == 1:
    param_data.copy_(loaded_weight)
    return
```

否则按输入维度切权重：

```python
loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
param_data.copy_(loaded_weight)
```

这里 `tp_dim = 1`，表示按权重矩阵的列方向切分：

```text
weight: [output_size, input_size]
按 input_size 切分
```

### 9.3 qwen3.6 版本变化

qwen3.6 在切片后加入：

```python
row_start = start_idx if self.tp_dim == 0 else 0
col_start = start_idx if self.tp_dim == 1 else 0
loaded_weight = maybe_dequant_fp8_weight(
    loaded_weight, loaded_scale, row_start, col_start
)
```

对于 Row Parallel，`tp_dim = 1`，所以通常是：

```text
row_start = 0
col_start = start_idx
```

### 9.4 为什么 Row Parallel 更需要 col_start

RowParallelLinear 按输入维切分，也就是按列切。

如果 FP8 scale 是按二维 block 对齐的，那么 rank 1/2/3 对应的权重 shard 在原始矩阵中的列 offset 不同。

所以必须告诉反量化函数：

```text
当前 shard 从原始矩阵第几列开始
```

否则可能用错 scale，导致反量化后的权重数值错误。

---

## 10. forward 计算基本没有变化

一个很重要的点是：

**qwen3.6 版本没有把 forward 改成真正的 FP8 GEMM。**

例如：

```python
def forward(self, x):
    return F.linear(x, self.weight, self.bias)
```

以及 RowParallelLinear：

```python
y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
if self.tp_size > 1:
    dist.all_reduce(y)
return y
```

这些 forward 路径和原版基本一致。

因此这个文件里的 FP8 支持更准确地说是：

```text
FP8 权重加载兼容 / load-time dequantization
```

而不是：

```text
真正运行时 FP8 matmul kernel
```

也就是说：

```text
磁盘上可以是 FP8 权重
加载进模型参数前被反量化
推理计算仍然走普通 F.linear
```

如果要实现真正 FP8 推理，还需要：

```text
FP8 activation
FP8 GEMM kernel
scale 管理
动态量化或静态量化路径
CUDA/Triton kernel 支持
```

仅凭这个 `linear.py` 的改动，还不能说项目已经实现了完整 FP8 runtime inference。

---

## 11. 这些改动和 Qwen3.6 的关系

从这个文件本身看，qwen3.6 版本的变化主要是：

```text
支持 FP8 权重格式加载
```

它不是直接支持：

```text
GatedDeltaNet
hybrid layer pattern
recurrent state
conv state
MoE routing
多模态 MRoPE
```

这些内容在其他文件中体现，例如：

```text
model_runner.py
scheduler.py
sequence.py
context.py
models/qwen3_5.py
layers/gated_delta_net.py
```

但是 `linear.py` 的 FP8 改造仍然很重要，因为无论是 dense attention、MLP、GDN 投影还是 MoE expert，底层都会大量用到线性层。

换句话说：

```text
Qwen3.6 的结构支持靠模型层和状态管理层；
Qwen3.6 的权重格式兼容靠 linear.py 和 loader/quant 相关文件。
```

---

## 12. 和 loader.py / quant.py 的关系

这个文件新增的 `loaded_scale` 参数只有在上游 loader 能把 scale 传进来时才有意义。

整体链路应该是：

```text
safetensors / HF checkpoint
  ↓
loader.py 读取 weight 和可能存在的 scale
  ↓
调用 param.weight_loader(param, loaded_weight, loaded_scale=...)
  ↓
linear.py 中 maybe_dequant_fp8_weight(...)
  ↓
反量化后的权重 copy 到 nn.Parameter
```

所以 `linear.py` 只是 FP8 权重支持链路的一环。

完整支持还依赖：

```text
nanovllm.utils.quant.maybe_dequant_fp8_weight
nanovllm.utils.loader.load_model
模型权重命名和 scale 命名映射
```

如果 loader 没有传 `loaded_scale`，那么 `maybe_dequant_fp8_weight` 很可能会走普通权重路径。

---

## 13. 对 Tensor Parallel 的意义

这次改动最容易被忽略的点是：

```text
FP8 反量化必须和 tensor parallel 切片顺序配合。
```

不能简单地：

```text
先对完整权重反量化，再切片
```

因为这样会占用更多临时显存。

也不能简单地：

```text
切片后直接反量化，但不知道原始 offset
```

因为 scale 可能按原始矩阵位置组织。

qwen3.6 版本采用的是：

```text
先切出当前 rank 需要的 shard
再带着 row_start / col_start 做反量化
再 copy 到 param
```

这是一种更适合大模型 TP 加载的设计。

不同并行线性层的 offset 关系如下：

| 线性层 | TP 切分方向 | offset 主要体现在 |
|---|---|---|
| ColumnParallelLinear | 输出维 / 行方向 | `row_start` |
| MergedColumnParallelLinear | 输出维 / 行方向 | `row_start` |
| QKVParallelLinear | 输出维 / 行方向 | `row_start` |
| RowParallelLinear | 输入维 / 列方向 | `col_start` |

---

## 14. 对 QKV / MLP 合并权重的意义

nano-vLLM 为了推理效率做了很多权重合并：

```text
q_proj + k_proj + v_proj -> qkv_proj
gate_proj + up_proj -> gate_up_proj
```

这些合并在普通 FP16/BF16 权重下已经比较复杂。

加入 FP8 后，还要处理：

```text
每个子权重自己的 scale
每个子权重在合并矩阵中的 offset
tensor parallel 后的 rank shard
```

qwen3.6 版本在 `MergedColumnParallelLinear` 和 `QKVParallelLinear` 中都加入了：

```text
loaded_scale
row_start / col_start
maybe_dequant_fp8_weight
```

说明它不仅支持普通单个线性层 FP8，还考虑到了 Qwen 模型中最核心的合并投影结构。

---

## 15. 原版与 qwen3.6 加载流程对比

### 15.1 原版加载流程

```text
读取 loaded_weight
  ↓
根据 TP rank 切片
  ↓
param.data.copy_(loaded_weight)
```

对于 QKV / gate-up：

```text
读取 q/k/v 或 gate/up 原始权重
  ↓
定位到合并参数中对应区域
  ↓
按 TP rank 切片
  ↓
copy 到 param
```

### 15.2 qwen3.6 加载流程

```text
读取 loaded_weight
读取 optional loaded_scale
  ↓
根据 TP rank 切片
  ↓
计算 row_start / col_start
  ↓
maybe_dequant_fp8_weight(loaded_weight, loaded_scale, row_start, col_start)
  ↓
copy 到 param
```

对于 QKV / gate-up：

```text
读取 q/k/v 或 gate/up FP8 权重和 scale
  ↓
定位到合并参数中对应区域
  ↓
按 TP rank 切出当前 shard
  ↓
根据 shard offset 反量化
  ↓
copy 到合并参数对应区域
```

---

## 16. 这个文件没有改变什么

为了避免误解，需要明确：

qwen3.6 版本 `linear.py` 没有明显改变下面这些内容：

```text
1. LinearBase 的基本参数结构
2. ColumnParallelLinear 的 forward 计算
3. RowParallelLinear 的 all_reduce 行为
4. QKVParallelLinear 的 q/k/v 输出大小计算
5. MergedColumnParallelLinear 的整体合并思想
6. Tensor Parallel 的基本切分方式
```

也就是说：

```text
模型计算图没有因为这个文件发生大变化。
```

变化集中在：

```text
权重加载阶段
```

---

## 17. 面试角度应该怎么回答

如果面试官问：

> qwen3.6 版本的 `linear.py` 相比原版改了什么？

可以这样回答：

`linear.py` 是 nano-vLLM 中 tensor parallel 线性层的基础实现，原版主要负责普通 FP16/BF16 权重的切片加载和 forward 计算。qwen3.6 版本最核心的变化是引入了 `maybe_dequant_fp8_weight`，并且给各类 `weight_loader` 增加了 `loaded_scale` 参数，使得 ReplicatedLinear、ColumnParallelLinear、MergedColumnParallelLinear、QKVParallelLinear、RowParallelLinear 都可以加载带 scale 的 FP8 权重。对于 tensor parallel 线性层，新版本会先根据当前 rank 切出权重 shard，再根据切片在原始矩阵中的 `row_start` 或 `col_start` 调用反量化函数，这样可以保证 FP8 scale 和切片位置对齐。需要注意的是，这个文件的 forward 计算基本没有变，仍然是 `F.linear` 和必要的 `all_reduce`，所以这里实现的是加载阶段的 FP8 反量化兼容，而不是真正的运行时 FP8 GEMM。整体意义是让 qwen3.6 项目能够兼容 FP8 checkpoint，同时保持原有 tensor parallel 线性层结构。

---

## 18. 初学者最应该抓住的主线

学习这个文件时，可以抓住一条主线：

```text
原版 linear.py:
    假设 loaded_weight 已经是可直接 copy 的浮点权重

qwen3.6 linear.py:
    允许 loaded_weight 是 FP8 权重
    并通过 loaded_scale 在加载时反量化
```

然后再记住：

```text
ColumnParallel -> 按行切，要关注 row_start
RowParallel    -> 按列切，要关注 col_start
QKV/gate_up    -> 既有合并权重，又有 TP 切片，所以必须更小心处理 scale
```

---

## 19. 最终结论

qwen3.6 版本 `linear.py` 的核心改造是：

```text
为所有主要线性层的 weight_loader 增加 FP8 权重反量化能力。
```

它的工程意义主要有三点：

1. **支持 FP8 checkpoint 加载**  
   通过 `loaded_scale` 和 `maybe_dequant_fp8_weight`，模型可以读取 FP8 权重并在加载时反量化到实际参数 dtype。

2. **适配 Tensor Parallel 切片**  
   对 Column / Row / QKV / Merged 线性层，在切片后传入 `row_start` / `col_start`，保证 scale 对齐原始权重矩阵位置。

3. **保持 forward 路径不变**  
   推理时仍然走 `F.linear` 和 `all_reduce`，所以这是加载兼容层改造，不是运行时 FP8 kernel 改造。

因此，这个文件在 qwen3.6 项目中的作用可以概括为：

```text
让 nano-vLLM 的并行线性层体系具备 FP8 权重格式兼容能力，为更大模型、更低显存权重加载和量化部署打基础。
```

它和前面几个文件的关系是：

```text
sequence.py / scheduler.py / model_runner.py:
    负责 hybrid state、多模态、调度和运行时状态

linear.py:
    负责底层线性层权重加载，尤其是 FP8 权重反量化与 TP 切片对齐
```


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
