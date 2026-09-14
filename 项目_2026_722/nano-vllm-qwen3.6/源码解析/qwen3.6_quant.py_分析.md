# qwen3.6_quant.py_分析

> 分析对象：`nano-vllm-qwen3.6` 新增源码文件 `quant.py`
>
> 本文目标：从整体工程角度分析该文件为什么需要新增、它在 Qwen3.6 / FP8 权重加载 / tensor parallel 权重切片中的作用，以及它和 `loader.py`、`linear.py`、`embed_head.py` 的关系。

---

## 1. 文件整体定位

`quant.py` 是 `nano-vllm-qwen3.6` 中新增的 **量化权重处理工具文件**。

它目前只实现了一类功能：

```text
FP8 权重加载时的反量化
```

更具体地说，它支持：

```text
FP8 weight + scale_inv
  ↓
根据 row_start / col_start 找到当前权重 shard 对应的 scale block
  ↓
将 FP8 weight 还原成 BF16 weight
  ↓
交给模型参数加载
```

所以这个文件不是 attention、MLP、GDN、KV Cache 或 scheduler 文件，而是：

```text
checkpoint 权重格式兼容层
```

它配合前面分析过的：

```text
loader.py
linear.py
embed_head.py
```

共同实现 qwen3.6 项目对 FP8 checkpoint 的加载兼容。

---

## 2. 为什么需要新增 quant.py

原版 nano-vLLM 主要假设 checkpoint 权重已经是：

```text
FP16 / BF16 / FP32
```

也就是说，loader 读到 tensor 后可以直接 copy 到模型参数中。

但是 qwen3.6 项目中开始出现 FP8 权重支持。FP8 权重的特点是：

```text
1. 单个数值占用更少显存
2. 数值范围和精度更有限
3. 需要 scale 才能还原近似真实权重
```

因此加载 FP8 权重时不能简单：

```python
param.data.copy_(loaded_weight)
```

而是要：

```text
loaded_weight: FP8
scale_inv: 反量化 scale
  ↓
weight.float() * scale.float()
  ↓
BF16 weight
  ↓
copy 到 param
```

这就是新增 `quant.py` 的原因。

---

## 3. 文件中的主要函数

该文件只有三个函数：

| 函数 | 作用 |
|---|---|
| `_ceil_div` | 整数向上除法，用于计算 block scale 覆盖范围 |
| `dequant_fp8_weight` | 真正执行 FP8 block-wise 反量化 |
| `maybe_dequant_fp8_weight` | 包装函数：有 scale 就反量化，没有 scale 就原样返回 |

整体结构非常轻量：

```text
quant.py
  ├── _ceil_div
  ├── dequant_fp8_weight
  └── maybe_dequant_fp8_weight
```

---

## 4. `_ceil_div` 的作用

源码：

```python
def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator
```

它做的是整数向上取整除法。

例如：

```text
ceil(129 / 128) = 2
ceil(256 / 128) = 2
ceil(257 / 128) = 3
```

在本文件中，它用于计算：

```text
当前权重 shard 覆盖了多少个 scale block
```

因为 FP8 scale 是按 block 存储的，默认 block 大小是：

```text
128 × 128
```

如果某个权重 shard 跨越多个 block，就需要取出多个 scale 值。

---

## 5. `dequant_fp8_weight` 的整体作用

`dequant_fp8_weight()` 是该文件的核心函数。

它的输入是：

```python
weight: torch.Tensor
scale_inv: torch.Tensor
row_start: int = 0
col_start: int = 0
block_size: tuple[int, int] = (128, 128)
```

含义如下：

| 参数 | 含义 |
|---|---|
| `weight` | 当前要加载的 FP8 权重，可能是完整权重，也可能是 TP 切片后的 shard |
| `scale_inv` | 与完整权重对应的 scale tensor |
| `row_start` | 当前 weight shard 在原始完整矩阵中的起始行 |
| `col_start` | 当前 weight shard 在原始完整矩阵中的起始列 |
| `block_size` | scale 的 block 粒度，默认 128×128 |

它的输出是：

```text
反量化后的 BF16 权重
```

源码最后返回：

```python
return (weight.float() * scale.float()).to(torch.bfloat16)
```

所以可以明确：

```text
当前实现是加载时反量化到 BF16。
```

---

## 6. 为什么需要 row_start / col_start

这是理解该文件的关键。

在 tensor parallel 下，模型不会把完整权重加载到每张卡上，而是会先切片：

```text
ColumnParallelLinear:
    按输出维 / 行方向切

RowParallelLinear:
    按输入维 / 列方向切

VocabParallelEmbedding:
    按 vocab 行方向切
```

假设完整权重矩阵是：

```text
weight_full: [4096, 4096]
```

FP8 scale 是按：

```text
128 × 128 block
```

存储的。

如果当前 rank 只加载：

```text
weight_full[2048:3072, :]
```

那么当前 shard 在原始矩阵中并不是从第 0 行开始，而是从：

```text
row_start = 2048
```

开始。

如果反量化时不知道这个偏移，就会用错 scale block。

所以 qwen3.6 在 `linear.py` 和 `embed_head.py` 中传入：

```text
row_start
col_start
```

而 `quant.py` 根据这些 offset 取正确的 scale 区域。

---

## 7. block-wise scale 的计算流程

`dequant_fp8_weight()` 中最重要的逻辑是：

```python
scale_row_start = row_start // block_n
scale_col_start = col_start // block_k
scale_row_end = _ceil_div(row_start + rows, block_n)
scale_col_end = _ceil_div(col_start + cols, block_k)
scale = scale_inv[scale_row_start:scale_row_end, scale_col_start:scale_col_end]
```

这一步做的是：

```text
根据当前权重 shard 在原矩阵中的坐标范围，
计算它覆盖哪些 scale block。
```

假设：

```text
block_size = 128 × 128
row_start = 256
rows = 128
```

那么：

```text
scale_row_start = 256 // 128 = 2
scale_row_end = ceil((256 + 128) / 128) = 3
```

说明当前 shard 在行方向只覆盖第 2 个 scale block。

如果：

```text
row_start = 100
rows = 100
```

那么它覆盖：

```text
row 100 到 row 199
```

跨越 block 0 和 block 1，所以：

```text
scale_row_start = 0
scale_row_end = ceil(200 / 128) = 2
```

需要取两个 block 的 scale。

---

## 8. scale repeat_interleave 的意义

取出 block 级 scale 后，源码执行：

```python
scale = scale.repeat_interleave(block_n, 0).repeat_interleave(block_k, 1)
```

这一步把 block 级 scale 展开成逐元素可广播的 scale。

例如，原始 scale shape 可能是：

```text
[num_scale_rows, num_scale_cols]
```

每个 scale 对应一个：

```text
128 × 128
```

权重 block。

`repeat_interleave` 后，每个 scale 被复制成对应 block 大小：

```text
scale: [num_scale_rows * 128, num_scale_cols * 128]
```

这样就可以和当前 `weight` 做逐元素乘法：

```python
weight.float() * scale.float()
```

---

## 9. row_offset / col_offset 的意义

由于当前权重 shard 不一定正好从 block 边界开始，所以还需要二次裁剪。

源码：

```python
row_offset = row_start % block_n
col_offset = col_start % block_k
scale = scale[row_offset:row_offset + rows, col_offset:col_offset + cols]
```

这一步解决的是：

```text
当前 shard 位于某个 scale block 中间，而不是从 block 左上角开始。
```

例如：

```text
block_size = 128
row_start = 64
rows = 128
```

当前 shard 覆盖：

```text
row 64 ~ row 191
```

它跨越两个 block，但第一个 block 只用后 64 行，第二个 block 只用前 64 行。

所以先取 block 0 和 block 1 的 scale，展开后再从：

```text
row_offset = 64
```

裁剪出刚好对应当前 weight 的 scale。

这就是该实现能正确处理任意切片 offset 的原因。

---

## 10. 反量化公式

最后一步：

```python
return (weight.float() * scale.float()).to(torch.bfloat16)
```

可以理解为：

```text
dequant_weight = fp8_weight × scale
```

然后输出 BF16。

这里有两个细节：

### 10.1 先转 float 再乘

`weight.float()` 和 `scale.float()` 表示乘法用 FP32 进行。

这样可以避免直接在低精度格式里计算造成更大误差。

### 10.2 输出 BF16

最后 `.to(torch.bfloat16)` 表示：

```text
加载进模型的权重 dtype 是 BF16。
```

所以当前实现不是 runtime FP8 GEMM，而是：

```text
load-time FP8 -> BF16 dequantization
```

---

## 11. `maybe_dequant_fp8_weight` 的作用

源码：

```python
def maybe_dequant_fp8_weight(weight, scale_inv, row_start=0, col_start=0):
    if scale_inv is None:
        return weight
    return dequant_fp8_weight(weight, scale_inv, row_start, col_start)
```

这个函数是一个非常实用的包装器。

它让上游代码可以统一写：

```python
loaded_weight = maybe_dequant_fp8_weight(loaded_weight, loaded_scale, row_start, col_start)
```

而不用每次都判断：

```text
这个权重是不是 FP8？
有没有 scale？
```

如果 `loaded_scale` 为 None，就说明：

```text
普通 FP16/BF16 权重
```

函数直接返回原权重。

如果有 scale，就走 FP8 反量化。

这让 `loader.py`、`linear.py`、`embed_head.py` 的代码更简洁。

---

## 12. 和 loader.py 的关系

`loader.py` 负责从 safetensors 中读取：

```text
weight
weight_scale_inv
```

并传给 weight_loader：

```text
loaded_weight
loaded_scale
```

而 `quant.py` 负责真正执行：

```text
loaded_weight + loaded_scale -> BF16 weight
```

完整链路：

```text
safetensors:
    xxx.weight
    xxx.weight_scale_inv

loader.py:
    loaded_weight = f.get_tensor(xxx.weight)
    loaded_scale = f.get_tensor(xxx.weight_scale_inv)

linear.py / embed_head.py / default_weight_loader:
    maybe_dequant_fp8_weight(loaded_weight, loaded_scale, row_start, col_start)

quant.py:
    dequant_fp8_weight(...)
```

所以 `quant.py` 是 FP8 加载链路的底层数学实现。

---

## 13. 和 linear.py 的关系

`linear.py` 中 qwen3.6 版本给各类线性层的 `weight_loader` 增加了：

```text
loaded_scale
row_start
col_start
```

例如：

```text
ColumnParallelLinear:
    按行切，所以传 row_start

RowParallelLinear:
    按列切，所以传 col_start

QKVParallelLinear:
    q/k/v 每个 shard 都要根据原矩阵位置传 offset

MergedColumnParallelLinear:
    gate/up 合并权重也要正确传 offset
```

这些 offset 最终都交给：

```python
maybe_dequant_fp8_weight(...)
```

然后由本文件根据 block size 找正确 scale。

因此可以说：

```text
linear.py 负责切片位置；
quant.py 负责根据切片位置找 scale 并反量化。
```

---

## 14. 和 embed_head.py 的关系

`embed_head.py` 中 qwen3.6 版本的 `VocabParallelEmbedding.weight_loader()` 也调用了：

```python
maybe_dequant_fp8_weight(loaded_weight, loaded_scale, row_start=start_idx)
```

Embedding / LM Head 是按 vocab 维度切片的。

也就是：

```text
不同 TP rank 持有不同 vocab 范围
```

所以它需要传：

```text
row_start = vocab_start
```

让 `quant.py` 取正确的 scale block。

因此 `quant.py` 不只服务线性层，也服务词表并行 embedding / lm_head。

---

## 15. 和 Qwen3.6 的关系

从这个文件本身看，它不直接涉及：

```text
GatedDeltaNet
hybrid layer_types
state_slot_id
vision encoder
MRoPE
MTP
scheduler
KV Cache
```

它的直接目标是：

```text
支持 FP8 checkpoint 权重加载。
```

但是在 qwen3.6 项目中，模型变复杂后，权重体积更大，加载 FP8 checkpoint 的需求更强。

所以它可以理解为 qwen3.6 工程改造中的：

```text
权重量化兼容基础设施
```

而不是：

```text
hybrid 架构本身的实现。
```

---

## 16. 这个文件实现的是权重量化，不是 KV Cache 量化

需要特别区分：

```text
FP8 weight dequantization
```

和：

```text
FP8 KV cache quantization
```

本文件做的是前者：

```text
磁盘上的模型权重是 FP8
加载时反量化成 BF16 参数
```

它没有做：

```text
KV Cache 存储成 FP8
decode 时对 K/V 动态量化/反量化
attention kernel 读取 FP8 KV
```

所以如果简历中写这个文件相关内容，应该准确表述为：

```text
支持 FP8 checkpoint 的加载时反量化
```

不能直接说：

```text
实现了 FP8 KV Cache 量化
```

除非你后续真的改了 KV Cache 存储和 attention kernel。

---

## 17. 这个文件也不是运行时 FP8 GEMM

当前实现最后输出 BF16：

```python
.to(torch.bfloat16)
```

说明权重进入模型参数后是 BF16。

forward 计算仍然可能是：

```text
F.linear(x, BF16 weight)
```

而不是：

```text
FP8 GEMM kernel
```

真正运行时 FP8 GEMM 需要：

```text
FP8 weight 常驻
activation quantization
scale management
FP8 matmul kernel
输出反量化
```

本文件没有这些内容。

所以它是：

```text
load-time dequantization
```

不是：

```text
runtime FP8 inference kernel
```

---

## 18. block size 128×128 的意义

默认：

```python
block_size = (128, 128)
```

这说明 scale 是按 128 行 × 128 列的 block 粒度组织的。

相比 per-tensor scale：

```text
整个权重一个 scale
```

block-wise scale 更精细，可以减少量化误差。

相比 per-element scale：

```text
每个元素一个 scale
```

block-wise scale 又更省元数据。

因此 128×128 block 是一种折中：

```text
量化精度
scale 存储开销
实现复杂度
```

都相对平衡。

---

## 19. 原版 nano-vLLM 为什么没有 quant.py

原版 nano-vLLM 没有这个文件，因为它不需要处理 FP8 权重：

```text
safetensors 读出来的权重可以直接 copy
```

qwen3.6 新增这个文件，说明项目开始支持：

```text
更复杂的 checkpoint 格式
带 scale 元数据的 FP8 权重
```

这也是从 toy/minimal inference engine 向更真实模型部署靠近的重要一步。

---

## 20. 这个实现的优点

### 20.1 简洁

文件很短，逻辑清楚：

```text
有 scale 就反量化
没有 scale 就原样返回
```

### 20.2 和 TP 切片兼容

通过：

```text
row_start / col_start
```

支持切片后的权重 shard 正确对齐 scale。

### 20.3 不侵入 forward

它只在加载时处理权重，不影响模型 forward 代码。

这使得集成成本低：

```text
不用改 attention / MLP forward
不用写 FP8 kernel
```

### 20.4 兼容普通权重

`maybe_dequant_fp8_weight` 在 `scale_inv is None` 时直接返回 weight，所以同一套 loader 可以同时支持：

```text
普通 BF16 checkpoint
FP8 checkpoint
```

---

## 21. 这个实现的局限

### 21.1 会失去 FP8 权重常驻显存优势

因为最终返回 BF16，模型参数还是 BF16。

所以它不能减少推理时权重显存到 FP8 级别。

它节省的主要可能是：

```text
checkpoint 磁盘体积
加载兼容性
```

而不是 runtime weight memory。

### 21.2 反量化会产生加载时额外开销

加载时要：

```text
读取 FP8
读取 scale
展开 scale
乘法反量化
转 BF16
```

这会增加启动阶段开销。

### 21.3 scale repeat_interleave 可能产生临时大 tensor

`repeat_interleave` 会把 block scale 展开到和 weight shard 一样大的 scale tensor。

这实现简单，但可能带来临时显存/内存开销。

对大型权重加载，如果要进一步优化，可以考虑：

```text
按 block 分块反量化
避免完整展开 scale
使用 fused dequant kernel
```

### 21.4 只支持当前 scale 格式

它假设 scale shape 和 block_size 组织方式固定为 block-wise 2D scale。

如果某些 checkpoint 使用不同量化格式，例如：

```text
per-channel scale
per-tensor scale
不同 block size
不同 FP8 dtype
```

就需要扩展。

---

## 22. 对简历项目的准确表述

如果把这个功能写到简历里，建议写成：

```text
实现 FP8 checkpoint 权重加载兼容：在 loader 中解析 weight_scale_inv，并在 Linear / Embedding weight_loader 中根据 TP shard 的 row/col offset 调用 block-wise dequant，将 FP8 权重加载时反量化为 BF16 参数，支持 Column/Row/QKV/Merged/Embedding 等并行权重的 scale 对齐。
```

不要写成：

```text
实现 FP8 推理
实现 FP8 KV Cache
实现 FP8 GEMM
```

除非你确实补充了运行时量化 kernel 和 KV cache 存储改造。

---

## 23. 面试角度应该怎么回答

如果面试官问：

> qwen3.6 新增的 `quant.py` 有什么作用？

可以这样回答：

`quant.py` 是 qwen3.6 版本新增的 FP8 权重反量化工具文件。它主要服务于 checkpoint 加载阶段，而不是模型 forward。loader 在读取 safetensors 时，如果发现权重是 FP8，并且存在对应的 `weight_scale_inv`，就把这个 scale 传给各参数的 weight_loader。weight_loader 会根据当前 tensor parallel shard 在完整权重矩阵中的 `row_start` 和 `col_start` 调用 `maybe_dequant_fp8_weight`。这个函数如果没有 scale 就原样返回，如果有 scale 就调用 `dequant_fp8_weight`。`dequant_fp8_weight` 按默认 128×128 block 计算当前 shard 覆盖的 scale block，展开 scale 后根据 offset 裁剪到和 weight 同形状，再执行 `weight.float() * scale.float()`，最后转成 BF16。它的意义是让 nano-vLLM-qwen3.6 能加载 FP8 checkpoint，并且在 tensor parallel 切片场景下正确对齐 block-wise scale。需要注意的是，它实现的是加载时 FP8 到 BF16 的反量化，不是运行时 FP8 GEMM，也不是 FP8 KV Cache 量化。

---

## 24. 初学者最应该抓住的主线

这个文件可以用一句话记住：

```text
quant.py 负责把 FP8 权重 + scale 还原成 BF16 权重。
```

最关键的是：

```text
row_start / col_start
```

因为 tensor parallel 下每张卡只加载完整权重的一部分。

完整链路是：

```text
loader.py 读 weight_scale_inv
  ↓
linear.py / embed_head.py 计算 shard offset
  ↓
quant.py 根据 offset 找 scale block
  ↓
FP8 weight × scale
  ↓
BF16 weight
```

---

## 25. 最终结论

`quant.py` 是 `nano-vllm-qwen3.6` 为支持 FP8 checkpoint 加载新增的底层工具文件。

它的核心意义是：

```text
让模型可以加载带 block-wise scale 的 FP8 权重，并在 tensor parallel 切片后正确反量化。
```

它完成了：

1. **整数向上除法**  
   用 `_ceil_div` 计算当前 shard 覆盖的 scale block 范围。

2. **block-wise scale 选择**  
   根据 `row_start / col_start` 从 `scale_inv` 中选出当前 shard 对应的 scale blocks。

3. **scale 展开与裁剪**  
   使用 `repeat_interleave` 展开 128×128 block scale，再根据 offset 裁剪到 weight 同形状。

4. **FP8 到 BF16 反量化**  
   执行 `weight.float() * scale.float()`，最后转成 BF16。

5. **统一包装接口**  
   `maybe_dequant_fp8_weight` 让普通权重和 FP8 权重共用同一套加载路径。

因此，它在整个 qwen3.6 项目中的位置是：

```text
loader.py:
    读取 FP8 weight 和 scale

linear.py / embed_head.py:
    计算 TP shard 的 row/col offset

quant.py:
    根据 offset 做 block-wise FP8 反量化

model forward:
    使用反量化后的 BF16 权重继续普通推理
```

如果说 `loader.py` 是 FP8 checkpoint 支持的入口，那么 `quant.py` 就是 FP8 权重反量化的核心实现。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
