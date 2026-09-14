# qwen3.6_loader.py_对比分析

> 对比对象：  
> 1. 原版 `nano-vLLM` 的 `loader.py`  
> 2. `nano-vllm-qwen3.6` 中对应的 `loader.py`
>
> 本文目标：从整体工程角度分析 qwen3.6 版本相比原版 `loader.py` 做了哪些修改、为什么这样改，以及这些修改在 Qwen3.6 / Qwen3.5 / hybrid / 多模态 / FP8 权重加载中的作用。

---

## 1. 文件整体定位

`loader.py` 是 nano-vLLM 中负责 **模型权重加载** 的工具文件。

在推理系统启动阶段，整体流程大致是：

```text
Config 读取模型路径和 HF config
  ↓
ModelRunner 创建模型结构
  ↓
load_model(model, path)
  ↓
读取 safetensors 权重
  ↓
把 checkpoint 中的 tensor 映射到 model 的 nn.Parameter
  ↓
调用各参数自己的 weight_loader
  ↓
模型 ready，开始 warmup / allocate kv cache / capture cuda graph
```

因此，`loader.py` 处在：

```text
模型结构定义
  和
checkpoint 权重文件
```

之间。

它要解决的问题不是 forward 怎么算，而是：

```text
磁盘上的权重名字、权重格式、权重切片方式，
如何正确加载到 nano-vLLM 自己定义的模型参数里。
```

原版 nano-vLLM 的 `loader.py` 很短，因为原版只支持比较标准的 Qwen3 dense 权重加载。qwen3.6 版本的 `loader.py` 明显变复杂，因为它要支持：

```text
1. FP8 权重和 scale
2. Qwen3.5/Qwen3.6 checkpoint 的权重名前缀
3. 多模态 visual encoder 权重
4. packed modules 的 loaded_scale 传递
5. 不匹配权重的跳过和记录
6. 加载日志
7. 返回加载结果给 ModelRunner 检查
```

---

## 2. qwen3.6 版本整体变化概览

相比原版 `loader.py`，qwen3.6 版本主要做了这些改造：

| 改动点 | 原版 nano-vLLM | qwen3.6 版本 | 作用 |
|---|---|---|---|
| FP8 支持 | 无 | 引入 `maybe_dequant_fp8_weight` | 支持 FP8 checkpoint 加载时反量化 |
| `default_weight_loader` | 只接收 `loaded_weight` | 新增 `loaded_scale` | 支持普通参数加载 FP8 scale |
| 加载结果 | 无返回值 | 新增 `LoadResult` | 记录 loaded/skipped 权重 |
| 日志 | 无 | `log_fn` 可选回调 | ModelRunner 可打印加载进度 |
| 文件顺序 | `glob()` 原始顺序 | `sorted(glob())` | 提高加载顺序可复现性 |
| scale tensor | 当普通权重处理 | 跳过 `.weight_scale_inv` | scale 作为辅助元数据，不当作参数直接加载 |
| FP8 scale 查找 | 无 | 查找 `weight_name + "_scale_inv"` | 支持 FP8 权重反量化 |
| 模型前缀 | 直接按 weight_name 找参数 | 支持 `weight_prefix` | 适配 `model.language_model.` 等 checkpoint 前缀 |
| 视觉前缀 | 无 | 支持 `visual_prefix` | 加载 `model.visual.` 到 `visual.` |
| packed modules | 支持简单映射 | 支持映射 + scale + try/except | 适配 gate/up 等合并参数和 FP8 |
| 错误处理 | 找不到参数直接异常 | 记录 `skipped_names` 并继续 | 支持部分权重缺失/额外模块 |
| 返回值 | `None` | `LoadResult` | 供 MTP/visual/hybrid 检查加载情况 |

一句话总结：

**qwen3.6 版本的 `loader.py` 从一个极简 safetensors 权重加载器，升级为支持 FP8、权重前缀映射、多模态 visual 权重、加载日志和加载结果统计的更通用 checkpoint loader。**

---

## 3. 原版 loader.py 的设计特点

原版 `loader.py` 的逻辑非常直接：

```text
遍历 path/*.safetensors
  ↓
遍历每个 weight_name
  ↓
如果命中 packed_modules_mapping:
      把 gate_proj/up_proj/q_proj/k_proj/v_proj 等映射到合并参数
      调用 param.weight_loader(param, tensor, shard_id)
  ↓
否则:
      model.get_parameter(weight_name)
      调用默认或自定义 weight_loader
```

它假设：

```text
1. checkpoint 权重名和模型参数名基本一致
2. 没有额外的前缀需要处理
3. 没有 visual encoder 权重
4. 没有 FP8 scale tensor
5. 每个 checkpoint tensor 都应该能找到对应参数
6. 找不到参数就是错误
```

这种写法对原版 nano-vLLM 很合理，因为原版模型结构简单、权重格式简单、模块数量少。

但对 qwen3.6 来说，这些假设都不够用了。

---

## 4. 改动一：新增 `LoadResult`

qwen3.6 版本新增：

```python
@dataclass
class LoadResult:
    loaded_names: list[str]
    skipped_names: list[str]
```

原版 `load_model()` 没有返回值。

新增 `LoadResult` 的意义是：

```text
把“哪些权重成功加载、哪些权重被跳过”显式返回给上层。
```

这对 qwen3.6 很重要，因为模型可能包含：

```text
语言模型权重
视觉模型权重
MTP 权重
FP8 scale tensor
checkpoint 中多余权重
当前配置未启用的模块权重
```

如果 loader 只是一遇到不匹配就报错，很多实验性功能会很难调试。

有了 `LoadResult`，上层可以做检查，例如 `model_runner.py` 中对 MTP 权重加载情况进行断言：

```text
如果 enable_mtp，mtp.* 权重不应该被 skipped
```

这比原版“静默加载或直接报错”更适合复杂模型。

---

## 5. 改动二：默认加载器支持 FP8 反量化

### 5.1 原版

原版：

```python
def default_weight_loader(param, loaded_weight):
    param.data.copy_(loaded_weight)
```

也就是说，读出来什么 tensor，就直接 copy 到参数里。

### 5.2 qwen3.6 版本

qwen3.6：

```python
def default_weight_loader(param, loaded_weight, loaded_scale=None):
    loaded_weight = maybe_dequant_fp8_weight(loaded_weight, loaded_scale)
    param.data.copy_(loaded_weight)
```

新增了两个点：

```text
1. loaded_scale 参数
2. maybe_dequant_fp8_weight
```

### 5.3 工程意义

这说明 qwen3.6 版本的 checkpoint 可能包含 FP8 权重。

FP8 权重通常不能直接 copy 到 FP16/BF16 参数中，需要结合 scale 做反量化：

```text
FP8 weight + scale
  ↓
maybe_dequant_fp8_weight
  ↓
FP16/BF16/FP32 tensor
  ↓
param.data.copy_
```

这和前面 `linear.py`、`embed_head.py` 的改造是同一条链路：

```text
loader.py:
    读取 loaded_scale

linear.py/embed_head.py:
    使用 loaded_scale 对切片权重反量化

quant.py:
    实际执行 maybe_dequant_fp8_weight
```

---

## 6. 改动三：读取并跳过 FP8 scale tensor

qwen3.6 版本在遍历权重名时增加：

```python
if weight_name.endswith(".weight_scale_inv"):
    continue
```

这说明 checkpoint 中可能有类似：

```text
xxx.weight
xxx.weight_scale_inv
```

其中：

```text
xxx.weight
    是 FP8 权重

xxx.weight_scale_inv
    是反量化所需 scale
```

scale tensor 不是模型参数本身，不能按普通权重加载到 `model.get_parameter()` 中。

所以 loader 的正确逻辑是：

```text
遍历到 weight_scale_inv:
    跳过，不单独加载

遍历到 weight:
    查找 weight + "_scale_inv"
    如果存在且 weight 是 FP8，就把它作为 loaded_scale 传给 weight_loader
```

qwen3.6 正是这么做的：

```python
scale_name = weight_name + "_scale_inv"
loaded_scale = (
    f.get_tensor(scale_name)
    if loaded_weight.dtype == torch.float8_e4m3fn and scale_name in weight_names
    else None
)
```

---

## 7. 为什么判断 dtype 是 `torch.float8_e4m3fn`

qwen3.6 并不是对所有权重都找 scale，而是判断：

```python
loaded_weight.dtype == torch.float8_e4m3fn
```

这表示：

```text
只有当 weight 本身是 FP8 e4m3fn 格式时，才需要 scale。
```

如果权重是：

```text
float16
bfloat16
float32
```

则：

```text
loaded_scale = None
```

这样可以兼容普通 checkpoint 和 FP8 checkpoint。

---

## 8. 改动四：支持加载日志 `log_fn`

qwen3.6 版本 `load_model()` 新增参数：

```python
def load_model(model, path, log_fn=None):
```

并在关键节点调用：

```python
log_fn(f"loading weights from {len(files)} safetensors files")
log_fn(f"loading weights {idx}/{len(files)}: {os.path.basename(file)}")
log_fn("weights loaded")
```

原版没有任何加载日志。

这个改动的意义是：

```text
复杂模型权重文件很多，加载时间更长，需要知道当前加载进度。
```

尤其 qwen3.6 可能包含：

```text
language model
visual encoder
MTP
FP8 scale
```

如果启动很慢，没有日志很难判断是卡住了还是正在加载。

---

## 9. 改动五：safetensors 文件排序

原版：

```python
for file in glob(os.path.join(path, "*.safetensors")):
```

qwen3.6：

```python
files = sorted(glob(os.path.join(path, "*.safetensors")))
```

这个改动很小，但有工程意义。

`glob()` 返回顺序不一定稳定，`sorted()` 可以保证：

```text
不同机器、不同文件系统下加载顺序更可复现。
```

在调试 `skipped_names`、`loaded_names` 或定位某个权重加载问题时，固定顺序更友好。

---

## 10. 改动六：支持 `weight_prefix`

qwen3.6 版本新增：

```python
weight_prefix = getattr(model, "weight_prefix", "")
```

并在加载语言模型权重时做：

```python
if weight_prefix and mapped_name.startswith(weight_prefix):
    mapped_name = "model." + mapped_name[len(weight_prefix):]
```

### 10.1 为什么需要 weight_prefix

在 `qwen3_5.py` 中，模型定义了：

```python
weight_prefix = "model.language_model."
```

这说明 checkpoint 中语言模型权重名可能是：

```text
model.language_model.layers.0.xxx
```

但 nano-vLLM 模型对象里的参数名可能是：

```text
model.layers.0.xxx
```

如果不处理前缀，原版 loader 会直接：

```python
model.get_parameter("model.language_model.layers.0.xxx")
```

但模型里没有这个参数，于是报错。

qwen3.6 的映射把它变成：

```text
model.language_model.layers.0.xxx
  ↓ strip weight_prefix
model.layers.0.xxx
```

这就是 checkpoint 命名空间到 nano-vLLM 模型命名空间的桥接。

---

## 11. 改动七：支持 `visual_prefix`

qwen3.6 版本新增：

```python
visual_prefix = getattr(model, "visual_prefix", "")
```

并专门处理：

```python
if visual_prefix and weight_name.startswith(visual_prefix):
    mapped_name = weight_name.replace(visual_prefix, "visual.")
    ...
```

在 `qwen3_5.py` 中，视觉前缀是：

```python
visual_prefix = "model.visual."
```

这意味着 checkpoint 里的视觉塔权重可能长这样：

```text
model.visual.patch_embed.proj.weight
model.visual.blocks.0.attn.qkv.weight
...
```

而 nano-vLLM 模型对象里视觉模块挂在：

```text
self.visual
```

所以参数名应该是：

```text
visual.patch_embed.proj.weight
visual.blocks.0.attn.qkv.weight
...
```

qwen3.6 loader 做的就是：

```text
model.visual.xxx
  ↓
visual.xxx
```

这让多模态模型的视觉 encoder 权重能够正确加载进 `Qwen3VLVisionEncoder`。

---

## 12. visual weights 的 try/except 设计

qwen3.6 处理 visual 权重时：

```python
try:
    param = model.get_parameter(mapped_name)
except (AttributeError, KeyError):
    skipped_names.append(weight_name)
    continue
```

原版没有这种容错。

为什么需要？

因为视觉模型可能是可选的：

```text
vision_config is None:
    self.visual = None
```

或者 checkpoint 中可能有一些当前实现暂不支持的视觉权重。

如果原版那样直接 `get_parameter()`，就会启动失败。

qwen3.6 的处理方式是：

```text
找得到就加载
找不到就记录 skipped_names 并继续
```

这对多模态/实验性支持很重要。

---

## 13. 改动八：packed modules 加载支持 loaded_scale

原版 packed modules 逻辑是：

```python
weight_loader(param, f.get_tensor(weight_name), shard_id)
```

qwen3.6 版本改成：

```python
weight_loader(param, loaded_weight, shard_id, loaded_scale)
```

这和 `linear.py` 中的改动对应。

例如 MLP 中：

```text
gate_proj.weight
up_proj.weight
```

会被加载到：

```text
gate_up_proj.weight
```

如果这些权重是 FP8，那么每个原始权重都可能有自己的 scale。

loader 必须把：

```text
loaded_weight
loaded_scale
shard_id
```

一起传给 packed parameter 的 weight_loader。

否则 `MergedColumnParallelLinear` 无法正确反量化和放置对应 shard。

---

## 14. 改动九：packed modules 加载增加跳过机制

原版 packed modules 如果找不到参数，会直接异常。

qwen3.6 版本：

```python
try:
    param = model.get_parameter(param_name)
except (AttributeError, KeyError):
    skipped_names.append(weight_name)
    break
```

这对于 qwen3.6 很重要，因为当前模型可能启用或禁用不同模块，例如：

```text
MTP
visual
language-only
multimodal
不同 Qwen3.5/Qwen3.6 checkpoint 变体
```

如果 checkpoint 有某些权重但当前模型结构没有对应模块，loader 不应该直接崩溃，而是记录 skipped，交给上层决定是否可接受。

---

## 15. 改动十：普通参数加载也支持跳过

原版普通路径：

```python
param = model.get_parameter(weight_name)
...
```

找不到直接报错。

qwen3.6：

```python
try:
    param = model.get_parameter(mapped_name)
except (AttributeError, KeyError):
    skipped_names.append(weight_name)
    continue
```

这进一步增强了 loader 的鲁棒性。

但要注意，这种容错是一把双刃剑：

```text
优点:
    支持复杂 checkpoint，不因额外权重失败

风险:
    如果关键权重被跳过，模型可能数值错误
```

所以 qwen3.6 又通过 `LoadResult` 把 `skipped_names` 返回，让上层可以检查关键模块是否加载完整。

---

## 16. 改动十一：返回 `LoadResult`

qwen3.6 `load_model()` 结尾：

```python
return LoadResult(loaded_names, skipped_names)
```

这让 ModelRunner 可以拿到：

```text
self.load_result.loaded_names
self.load_result.skipped_names
```

这对调试非常有用。

例如：

```text
打印 loaded tensor 数量
检查某类权重是否全部加载
查看哪些权重没有映射到当前模型
```

原版 loader 完全没有这类可观测性。

---

## 17. 和 linear.py / embed_head.py 的关系

前面 `linear.py`、`embed_head.py` 的 qwen3.6 版本都做了类似改动：

```text
weight_loader(..., loaded_scale=None)
maybe_dequant_fp8_weight(...)
row_start / col_start
```

但如果 `loader.py` 不读取 scale，这些改动就没有意义。

完整链路是：

```text
loader.py:
    从 safetensors 中读取 weight 和 weight_scale_inv

linear.py / embed_head.py:
    接收 loaded_scale
    根据 TP shard offset 做 FP8 反量化

quant.py:
    实现 maybe_dequant_fp8_weight
```

所以 `loader.py` 是 FP8 checkpoint 支持的入口。

---

## 18. 和 qwen3_5.py 的关系

`qwen3_5.py` 中定义了：

```text
weight_prefix = "model.language_model."
visual_prefix = "model.visual."
packed_modules_mapping = {
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
```

qwen3.6 的 `loader.py` 正是读取这些属性：

```python
packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
weight_prefix = getattr(model, "weight_prefix", "")
visual_prefix = getattr(model, "visual_prefix", "")
```

然后根据这些信息决定：

```text
语言模型权重怎么映射
视觉模型权重怎么映射
gate/up 权重怎么打包
```

也就是说，`qwen3_5.py` 定义“模型需要什么名字”，`loader.py` 负责“checkpoint 里的名字怎么对上模型里的名字”。

---

## 19. 和 vision_encoder.py 的关系

`vision_encoder.py` 定义了：

```text
Qwen3VLVisionEncoder
```

它挂载在模型外壳中：

```python
self.visual = Qwen3VLVisionEncoder(vision_config)
```

checkpoint 中视觉权重通常在：

```text
model.visual.*
```

而模型参数名是：

```text
visual.*
```

因此 `loader.py` 中 `visual_prefix` 的映射是视觉 encoder 能正确加载权重的关键。

如果没有这段逻辑，多模态模型虽然创建了 visual encoder，但权重加载会失败。

---

## 20. 和 model_runner.py 的关系

在 qwen3.6 的 `model_runner.py` 中，加载模型时会保存结果：

```text
self.load_result = load_model(...)
```

并且可能传入：

```text
log_fn=self._log
```

这样启动时会输出：

```text
loading weights from N safetensors files
loading weights i/N: xxx.safetensors
weights loaded
```

如果启用了 MTP，`model_runner.py` 还会检查：

```text
mtp_skipped
mtp_loaded
```

这说明 `loader.py` 的返回值已经成为上层执行器判断模型是否加载完整的重要依据。

---

## 21. 原版与 qwen3.6 加载流程对比

### 21.1 原版加载流程

```text
for each safetensors file:
    for each weight_name:
        if weight_name matches packed_modules_mapping:
            param_name = weight_name.replace(k, v)
            param = model.get_parameter(param_name)
            param.weight_loader(param, tensor, shard_id)
        else:
            param = model.get_parameter(weight_name)
            weight_loader(param, tensor)
```

特点：

```text
简单
严格
无前缀处理
无 FP8
无 visual
无返回结果
```

---

### 21.2 qwen3.6 加载流程

```text
读取 sorted safetensors files
记录 loaded_names / skipped_names

for each file:
    weight_names = set(f.keys())

    for each weight_name:
        if weight_name endswith ".weight_scale_inv":
            continue

        loaded_weight = f.get_tensor(weight_name)
        loaded_scale = FP8 时查找 weight_name + "_scale_inv"

        if visual_prefix 命中:
            model.visual.* 映射
            加载或 skipped
            continue

        if weight_prefix 命中:
            model.language_model.* -> model.*

        if 命中 packed_modules_mapping:
            映射到 packed param
            weight_loader(param, loaded_weight, shard_id, loaded_scale)
            加载或 skipped
            continue

        普通参数:
            model.get_parameter(mapped_name)
            weight_loader(param, loaded_weight, loaded_scale)
            加载或 skipped

返回 LoadResult
```

这条流程明显更适合复杂 Qwen3.6 checkpoint。

---

## 22. 这个文件没有实现什么

需要避免误解，`loader.py` 虽然支持 FP8 scale，但它没有实现：

```text
1. 真正的 FP8 GEMM
2. runtime FP8 activation quantization
3. KV Cache FP8 量化
4. GDN state 量化
5. 权重下载
6. safetensors 分布式并行读取优化
7. 异步加载
8. 权重 shape 自动修复
```

它主要是：

```text
checkpoint 参数名映射 + 权重 tensor 加载 + scale 传递 + 加载结果记录。
```

真正的 FP8 反量化逻辑在：

```text
quant.py / maybe_dequant_fp8_weight
```

真正的 tensor parallel 切片逻辑在：

```text
linear.py / embed_head.py 的 weight_loader
```

---

## 23. qwen3.6 loader.py 的工程意义

qwen3.6 的 loader 改造解决了三个核心问题。

### 23.1 权重格式更复杂

支持：

```text
FP16/BF16 普通权重
FP8 weight + weight_scale_inv
```

### 23.2 权重命名空间更复杂

支持：

```text
model.language_model.* -> model.*
model.visual.* -> visual.*
```

### 23.3 模型结构更复杂

支持：

```text
language model
visual encoder
MTP
packed MLP
可选模块
```

因此它是 qwen3.6 项目从“单一文本模型加载”走向“复杂 multimodal/hybrid checkpoint 加载”的关键基础设施。

---

## 24. 常见误区

### 误区一：loader.py 实现了完整 FP8 推理

不是。

它只是读取 FP8 权重和 scale，并传给 weight_loader 做加载时反量化。forward 仍可能是普通 dtype 计算。

### 误区二：skipped_names 一定是错误

不一定。

如果 checkpoint 中包含当前配置未启用的模块，跳过可能是合理的。

但如果关键模块权重被跳过，例如启用 MTP 时 MTP 权重被跳过，就可能是严重错误。

### 误区三：visual_prefix 是给语言模型用的

不是。

`visual_prefix` 专门把 checkpoint 中的 `model.visual.*` 映射到模型对象的 `visual.*`。

### 误区四：weight_prefix 只是字符串替换，无关紧要

不是。

如果不做这个映射，Qwen3.5/Qwen3.6 checkpoint 中的语言模型权重可能根本找不到对应参数。

---

## 25. 面试角度应该怎么回答

如果面试官问：

> qwen3.6 版本的 `loader.py` 相比原版改了什么？

可以这样回答：

`loader.py` 是模型权重加载器。原版 nano-vLLM 的实现很简单，遍历 safetensors 文件后按权重名直接找参数，如果命中 packed_modules_mapping 就把 gate/up 或 q/k/v 之类的权重映射到合并参数中，否则直接 copy。qwen3.6 版本做了明显增强：首先新增 `LoadResult`，记录 loaded_names 和 skipped_names，并支持 `log_fn` 打印加载进度；其次引入 `maybe_dequant_fp8_weight`，让默认加载器和各类 weight_loader 都能接收 `loaded_scale`，并且在遍历 safetensors 时跳过 `.weight_scale_inv`，只在对应 FP8 weight 加载时把 scale 作为辅助 tensor 传下去；第三，它支持 `weight_prefix` 和 `visual_prefix`，可以把 checkpoint 里的 `model.language_model.*` 映射到模型里的 `model.*`，把 `model.visual.*` 映射到 `visual.*`，从而适配 Qwen3.5/Qwen3.6 多模态 checkpoint；最后，它对找不到的参数不再直接崩溃，而是记录 skipped_names 并继续，方便兼容 MTP、视觉塔等可选模块。整体来看，这个文件把原版的简单权重加载器升级成了支持 FP8、多模态、模型前缀映射和加载可观测性的复杂 checkpoint loader。

---

## 26. 初学者最应该抓住的主线

这个文件可以这样记：

```text
原版 loader.py:
    checkpoint 权重名 ≈ 模型参数名
    直接 get_parameter + copy

qwen3.6 loader.py:
    checkpoint 权重名可能有前缀
    权重可能是 FP8 + scale
    模型可能有 visual / mtp / optional modules
    所以需要映射、反量化、跳过和记录
```

最关键的三条链路是：

```text
FP8:
    weight + weight_scale_inv -> loaded_scale -> weight_loader

语言模型:
    model.language_model.* -> model.*

视觉模型:
    model.visual.* -> visual.*
```

---

## 27. 最终结论

qwen3.6 版本 `loader.py` 的核心改造是：

```text
把原版极简权重加载器，扩展成适配 Qwen3.6 复杂 checkpoint 的通用加载器。
```

它完成了：

1. **FP8 checkpoint 支持**  
   读取 `weight_scale_inv`，并把 `loaded_scale` 传给 weight_loader。

2. **加载时反量化入口**  
   默认加载器调用 `maybe_dequant_fp8_weight`。

3. **语言模型前缀映射**  
   支持 `model.language_model.* -> model.*`。

4. **视觉模型前缀映射**  
   支持 `model.visual.* -> visual.*`，让视觉 encoder 权重能加载。

5. **packed modules 兼容 FP8**  
   gate/up 等打包权重加载时也传递 scale。

6. **加载鲁棒性增强**  
   找不到参数时记录 skipped_names，而不是直接终止。

7. **加载可观测性增强**  
   返回 `LoadResult`，并支持 `log_fn` 输出加载进度。

因此，这个文件在整个 qwen3.6 项目中的位置是：

```text
qwen3_5.py / vision_encoder.py / qwen3_mtp.py:
    定义复杂模型结构

linear.py / embed_head.py:
    定义参数如何切片和反量化

loader.py:
    把 checkpoint 权重正确映射并加载到这些结构中

model_runner.py:
    调用 loader，并根据 LoadResult 检查加载结果
```

如果说 `qwen3_5.py` 定义了“模型长什么样”，那么 `loader.py` 就负责“checkpoint 里的权重如何正确装进这个模型”。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
