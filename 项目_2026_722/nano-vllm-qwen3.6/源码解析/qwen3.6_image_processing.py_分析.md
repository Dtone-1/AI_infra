# qwen3.6_image_processing.py_分析

> 分析对象：`nano-vllm-qwen3.6` 新增源码文件 `image_processing.py`
>
> 本文目标：从整体工程角度分析该文件为什么需要新增、它在 Qwen3.6 / Qwen-VL / 多模态推理入口中的作用，以及它和 `llm_engine.py`、`sequence.py`、`model_runner.py`、`vision_encoder.py`、`qwen3_5.py` 的关系。

---

## 1. 文件整体定位

`image_processing.py` 是 `nano-vllm-qwen3.6` 中新增的 **多模态输入预处理文件**。

原版 nano-vLLM 主要面向纯文本推理，入口流程是：

```text
prompt: str
  ↓
tokenizer.encode(prompt)
  ↓
token_ids
  ↓
Sequence
  ↓
Scheduler / ModelRunner / Model
```

而 qwen3.6 版本开始支持类似 Qwen-VL 的多模态 messages 输入，用户输入不再只是字符串，也可能包含图片：

```python
[
  {
    "role": "user",
    "content": [
      {"type": "image", "image": ...},
      {"type": "text", "text": "请描述这张图"}
    ]
  }
]
```

这时推理入口不能只做 tokenizer.encode，还需要额外完成：

```text
1. 解析 messages
2. 找到 image part
3. 读取或接收 PIL Image
4. resize / normalize / patchify
5. 生成 pixel_values
6. 生成 image_grid_thw
7. 在文本中插入 image placeholder token
8. 最终 tokenizer.encode 成 token_ids
```

`image_processing.py` 就是做这件事的文件。

它不是模型层，也不是视觉编码器本身，而是：

```text
多模态请求进入推理系统之前的预处理桥梁。
```

---

## 2. 为什么需要新增这个文件

原版 nano-vLLM 的输入只有文本：

```text
字符串 prompt
  ↓ tokenizer
token_ids
```

但是多模态模型需要同时准备两类输入：

```text
文本输入:
    token_ids

图像输入:
    pixel_values
    image_grid_thw
```

而且这两类输入必须保持对齐：

```text
文本里有多少个 <|image_pad|> token
就必须和视觉 encoder 最终输出的 image embeds 数量匹配。
```

如果只把图片转成 tensor，但没有在文本 token 序列里插入对应 image token，语言模型不知道图像 embedding 应该插入哪里。

反过来，如果只插入 image token，但没有 pixel_values，模型也无法看到图像内容。

因此新增 `image_processing.py` 的意义是：

```text
把多模态 messages 同时转换成语言模型需要的 token_ids 和视觉模型需要的 pixel_values/image_grid_thw，并保证二者数量对齐。
```

---

## 3. 文件中的主要函数

该文件主要包含三个核心函数：

| 函数 | 作用 |
|---|---|
| `smart_resize` | 将图片尺寸调整到 patch/merge 对齐的倍数，并控制像素预算 |
| `process_image` | 将单张 PIL Image 转成 `pixel_values` 和 `image_grid_thw` |
| `process_messages` | 将多模态 messages 转成 `token_ids + pixel_values + image_grid_thw` |

此外还定义了图像归一化参数：

```python
IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)
```

这意味着图像 tensor 会从 `[0, 1]` 归一化到大致 `[-1, 1]` 区间。

---

## 4. 输入输出总览

### 4.1 输入

`process_messages()` 接收：

```python
messages: list[dict]
tokenizer
patch_size
temporal_patch_size
merge_size
image_token_id
vision_start_id
vision_end_id
```

其中最重要的是：

```text
messages:
    OpenAI / ChatML 风格的多模态消息结构

tokenizer:
    Qwen tokenizer，用于最终把拼接好的文本编码成 token_ids

patch_size / temporal_patch_size / merge_size:
    和视觉 encoder 的 patch embedding / patch merger 对齐

image_token_id / vision_start_id / vision_end_id:
    图像特殊 token 的 id
```

需要注意的是，在当前文件中 `image_token_id / vision_start_id / vision_end_id` 参数虽然传入了，但 `process_messages()` 构造文本 placeholder 时实际使用的是特殊 token 字符串：

```text
<|vision_start|>
<|image_pad|>
<|vision_end|>
```

然后由 tokenizer 将这些特殊 token 字符串转成对应 token id。

---

### 4.2 输出

`process_messages()` 返回：

```python
token_ids, pixel_values, image_grid_thw
```

其中：

| 返回值 | 含义 |
|---|---|
| `token_ids` | 插入 image placeholder 后的完整文本 token ids |
| `pixel_values` | 所有图片 patch 展平后的视觉输入 tensor |
| `image_grid_thw` | 每张图片的 `[T, H, W]` 网格信息 |
| `None` | 如果没有图片，则 `pixel_values` 和 `image_grid_thw` 为 None |

这三个输出会继续进入：

```text
llm_engine.py -> Sequence -> Scheduler -> ModelRunner -> Qwen3_5ForCausalLM
```

---

## 5. 函数一：smart_resize

`smart_resize()` 的职责是：

```text
把输入图片尺寸调整到符合 patch/merge 要求的大小，并控制像素数量范围。
```

它接收：

```python
height
width
factor
min_pixels
max_pixels
```

其中默认：

```text
factor = 32
min_pixels = 3136
max_pixels = 1003520
```

在 `process_image()` 中，factor 被设置为：

```python
factor = patch_size * merge_size
```

默认情况下：

```text
patch_size = 16
merge_size = 2
factor = 32
```

这意味着最终图片高宽必须是 32 的倍数。

---

## 6. 为什么图片尺寸要对齐 factor

视觉模型后面会做两步：

```text
1. patchify:
    按 patch_size 切 patch

2. merge:
    按 merge_size × merge_size 合并 patch
```

假设：

```text
patch_size = 16
merge_size = 2
```

那么语言模型最终看到的一个 image token，对应视觉侧：

```text
2 × 2 个 patch
```

每个 patch 是：

```text
16 × 16 像素
```

所以一个 merged image token 对应：

```text
32 × 32 像素区域
```

因此图片高宽必须能被：

```text
patch_size * merge_size
```

整除，否则后续 patch reshape / merge 会对不上。

这就是 `smart_resize()` 要把图片尺寸调整到 factor 倍数的原因。

---

## 7. smart_resize 的三步逻辑

`smart_resize()` 主要做三件事。

### 7.1 检查最小边

```python
if height < factor or width < factor:
    raise ValueError(...)
```

如果图片太小，小于一个最小 patch/merge 单元，就无法正常切 patch。

---

### 7.2 四舍五入到 factor 倍数

```python
h_bar = max(factor, round(height / factor) * factor)
w_bar = max(factor, round(width / factor) * factor)
```

这一步把高宽调整到最接近的 factor 倍数。

例如：

```text
height = 501
factor = 32
501 / 32 ≈ 15.66
round -> 16
new height = 512
```

---

### 7.3 像素预算控制

如果像素太多：

```python
if h_bar * w_bar > max_pixels:
    scale = sqrt(max_pixels / (h_bar * w_bar))
    ...
```

就缩小。

如果像素太少：

```python
if h_bar * w_bar < min_pixels:
    scale = sqrt(min_pixels / (h_bar * w_bar))
    ...
```

就放大。

这一步非常重要，因为视觉 token 数和图像像素数强相关。

图像太大，会导致：

```text
pixel_values 变大
vision encoder 计算变重
image token 数变多
language model prefill 变慢
KV Cache / GDN state 压力变大
```

图像太小，则可能丢失有效视觉信息。

所以 `smart_resize()` 是多模态推理入口的第一道性能/质量平衡控制。

---

## 8. 函数二：process_image

`process_image()` 是单张图片预处理的核心函数。

它输入：

```text
PIL Image
```

输出：

```text
pixel_values
image_grid_thw
```

它的流程是：

```text
PIL Image
  ↓ RGB
resize 到 factor 对齐尺寸
  ↓
to_tensor
  ↓
normalize
  ↓
补 temporal 维度
  ↓
计算 grid_t / grid_h / grid_w
  ↓
reshape + permute 成 Qwen VL patch 顺序
  ↓
flatten 成 pixel_values
  ↓
返回 image_grid_thw
```

---

## 9. process_image 第一步：RGB 转换和 resize

源码中：

```python
image = image.convert("RGB")
w, h = image.size
factor = patch_size * merge_size
new_h, new_w = smart_resize(h, w, factor, min_pixels, max_pixels)
image = image.resize((new_w, new_h), Image.BICUBIC)
```

这保证：

```text
1. 输入一定是 3 通道 RGB
2. 高宽满足 patch/merge 对齐
3. 尺寸在 min_pixels/max_pixels 预算内
```

这里使用 `Image.BICUBIC` 做插值，属于常见图像 resize 方式。

---

## 10. process_image 第二步：to_tensor 和 normalize

源码中：

```python
pixels = TF.to_tensor(image)
pixels = TF.normalize(pixels, IMAGE_MEAN, IMAGE_STD)
```

`TF.to_tensor(image)` 会把 PIL Image 转成：

```text
(C, H, W)
float32
数值范围 [0, 1]
```

随后使用：

```text
mean = 0.5
std = 0.5
```

归一化：

```text
x_norm = (x - 0.5) / 0.5
```

所以数值范围大约变成：

```text
[-1, 1]
```

这通常要和视觉 encoder checkpoint 的训练预处理保持一致。

如果这里的 mean/std 和模型训练时不一致，视觉 encoder 输出会偏移，影响图像理解效果。

---

## 11. process_image 第三步：增加 temporal 维度

源码中：

```python
pixels = pixels.unsqueeze(0)
if pixels.shape[0] < temporal_patch_size:
    pad = pixels.repeat(temporal_patch_size, 1, 1, 1)[:temporal_patch_size]
    pixels = pad
```

原始图片是：

```text
(C, H, W)
```

转换后变成：

```text
(1, C, H, W)
```

然后如果 `temporal_patch_size = 2`，就复制帧：

```text
(1, C, H, W)
  ↓
(2, C, H, W)
```

这是因为视觉 encoder 使用的是 3D Conv patch embedding：

```text
temporal_patch_size × patch_size × patch_size
```

即使输入是单张图片，也要伪造成满足 temporal patch 的 T 维度。

对图片而言，这相当于：

```text
把同一帧复制到 temporal_patch_size 个时间步
```

---

## 12. process_image 第四步：计算 image_grid_thw

源码中：

```python
T, C, H, W = pixels.shape
grid_t = T // temporal_patch_size
grid_h = H // patch_size
grid_w = W // patch_size
```

`image_grid_thw` 记录的是：

```text
视觉 patch 网格:
    T 方向有多少个 temporal patch
    H 方向有多少个 patch
    W 方向有多少个 patch
```

默认单张图片：

```text
T = 2
temporal_patch_size = 2
grid_t = 1
```

如果 H=448，W=448，patch_size=16：

```text
grid_h = 28
grid_w = 28
```

最终：

```text
image_grid_thw = [[1, 28, 28]]
```

这个值后面非常重要：

```text
vision_encoder.py:
    用它计算视觉 patch 位置和 cu_seqlens

model_runner.py:
    用它计算 MRoPE 3D positions

process_messages.py:
    用它计算需要插入多少个 image_pad token
```

---

## 13. process_image 第五步：reshape / permute 成 Qwen VL patch 顺序

源码中：

```python
patches = pixels.view(
    grid_t, temporal_patch_size,
    C,
    grid_h // merge_size, merge_size, patch_size,
    grid_w // merge_size, merge_size, patch_size,
)
```

然后：

```python
patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
```

最后：

```python
pixel_values = patches.reshape(
    -1,
    C * temporal_patch_size * patch_size * patch_size
)
```

这部分是整个文件最容易看不懂的地方。

它做的是：

```text
把图片按照 Qwen VL 约定的 patch 顺序重新排列，
并把每个 patch group flatten 成一行。
```

最终 `pixel_values` 形状是：

```text
(total_patches, patch_dim)
```

其中：

```text
patch_dim = C * temporal_patch_size * patch_size * patch_size
```

默认：

```text
C = 3
temporal_patch_size = 2
patch_size = 16
patch_dim = 3 * 2 * 16 * 16 = 1536
```

也就是说，每一行 pixel_values 表示一个 temporal-spatial patch 的原始像素向量。

这些 pixel_values 会送入 `vision_encoder.py` 的 `VisionPatchEmbed`，再通过 3D Conv 变成视觉 hidden states。

---

## 14. process_image 输出的 pixel_values 和 image_grid_thw 如何配合

`pixel_values` 保存真正的图像内容：

```text
patch 像素值
```

`image_grid_thw` 保存图像 patch 的结构：

```text
这些 patch 原本在 T/H/W 上如何排列
```

二者必须同时存在。

如果只有 pixel_values，没有 image_grid_thw：

```text
vision encoder 不知道 patch 的空间结构
MRoPE 也不知道图像 token 的 T/H/W 坐标
```

如果只有 image_grid_thw，没有 pixel_values：

```text
模型知道有多少 patch，但没有图像内容
```

所以 `process_image()` 返回二者：

```python
return pixel_values, image_grid_thw
```

---

## 15. 函数三：process_messages

`process_messages()` 是多模态入口最重要的函数。

它负责把 messages 转成：

```text
token_ids
pixel_values
image_grid_thw
```

它处理两类 content：

```text
1. content 是字符串
2. content 是 list，里面包含 text part 和 image part
```

---

## 16. process_messages 对纯文本消息的处理

如果：

```python
content = "你好"
```

则构造：

```text
<|im_start|>user
你好<|im_end|>
```

对应代码：

```python
text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
```

这是一种 ChatML 风格对话模板。

最后还会追加：

```text
<|im_start|>assistant
```

表示让模型从 assistant 回答位置开始生成。

---

## 17. process_messages 对图片消息的处理

如果 content 是 list，例如：

```python
[
  {"type": "image", "image": img},
  {"type": "text", "text": "描述这张图"}
]
```

它会遍历每个 part。

当遇到：

```python
part["type"] == "image"
```

时：

```python
img = part["image"]
if isinstance(img, str):
    img = Image.open(img)
pv, gt = process_image(img, ...)
pixel_values_list.append(pv)
grid_thw_list.append(gt)
```

也就是说，图片可以是：

```text
PIL Image
或图片路径字符串
```

如果是字符串路径，就用 PIL 打开。

然后调用 `process_image()` 得到：

```text
pv = pixel_values
gt = image_grid_thw
```

---

## 18. 为什么要插入 image placeholder

处理图片后，代码计算：

```python
t, h, w = gt[0].tolist()
num_tokens = t * (h // merge_size) * (w // merge_size)
placeholder = "<|vision_start|>" + "<|image_pad|>" * num_tokens + "<|vision_end|>"
parts_text.append(placeholder)
```

这一步非常关键。

它表示：

```text
每张图片在文本 token 序列中占据若干个 image_pad token。
```

这些 image_pad token 后续会对应视觉 encoder 输出的 image embeddings。

---

## 19. num_tokens 为什么这样计算

`image_grid_thw` 中：

```text
t = grid_t
h = grid_h
w = grid_w
```

其中 h/w 是 patch 级别的网格大小。

但视觉 encoder 后面会进行 spatial merge：

```text
merge_size × merge_size 个 patch 合并成 1 个视觉 token
```

所以语言模型最终看到的 image token 数量是：

```text
t * (h // merge_size) * (w // merge_size)
```

例如：

```text
grid_t = 1
grid_h = 28
grid_w = 28
merge_size = 2
```

则：

```text
num_tokens = 1 * 14 * 14 = 196
```

也就是说，文本中要插入 196 个：

```text
<|image_pad|>
```

后续 `vision_encoder.py` 输出的 image_embeds 数量也应该是 196。

这就是多模态对齐的核心。

---

## 20. vision_start / image_pad / vision_end 的作用

一张图片在文本中被表示为：

```text
<|vision_start|>
<|image_pad|> × num_tokens
<|vision_end|>
```

这三个特殊 token 的作用是：

| token | 作用 |
|---|---|
| `<|vision_start|>` | 标记视觉内容开始 |
| `<|image_pad|>` | 占位视觉 token，后续被 image_embeds 对齐替换 |
| `<|vision_end|>` | 标记视觉内容结束 |

语言模型通过这些特殊 token 知道：

```text
这里是一段图像内容
```

但真正的图像语义不是来自 `<|image_pad|>` 这个 token 的 embedding，而是后续：

```text
hidden_states[image_token_mask] = image_embeds
```

把视觉 encoder 输出替换进去。

---

## 21. process_messages 最终构造 full_text

所有消息处理完后：

```python
full_text = "".join(text_parts) + "<|im_start|>assistant\n"
token_ids = tokenizer.encode(full_text, add_special_tokens=False)
```

这说明：

```text
messages 先被转成一个 ChatML 字符串
再统一 tokenizer.encode
```

而不是逐段分别 encode。

这样可以保证特殊 token、换行、role 格式都按 tokenizer 的规则处理。

---

## 22. 多张图片如何处理

如果 messages 中有多张图片：

```text
image1
image2
...
```

每张图片都会产生：

```text
pv_i
gt_i
placeholder_i
```

最后：

```python
pixel_values = torch.cat(pixel_values_list, dim=0)
image_grid_thw = torch.cat(grid_thw_list, dim=0)
```

也就是说：

```text
所有图片的 pixel_values 在 patch 维度拼接
所有图片的 image_grid_thw 在 image 维度拼接
```

`image_grid_thw` 形状是：

```text
(num_images, 3)
```

这使得后面的 vision encoder 可以一次处理多张图片，同时用 `grid_thw` 知道每张图的边界和大小。

---

## 23. 没有图片时的返回

如果没有图片：

```python
pixel_values = None
image_grid_thw = None
```

这很重要，因为同一个入口函数可以同时支持：

```text
纯文本 messages
多模态 messages
```

后续 `ModelRunner.prepare_prefill()` 会检查：

```python
has_images = any(seq.pixel_values is not None for seq in seqs)
```

如果没有图片，就走普通文本路径。

---

## 24. 和 llm_engine.py 的关系

前面 `llm_engine.py` 的 qwen3.6 版本中新增了：

```python
from nanovllm.utils.image_processing import process_messages
```

当 prompt 是 list[dict] 形式的 messages 时，会调用：

```text
process_messages(...)
```

然后得到：

```text
token_ids
pixel_values
image_grid_thw
```

再构造 `Sequence`：

```text
seq = Sequence(token_ids, sampling_params)
seq.pixel_values = pixel_values
seq.image_grid_thw = image_grid_thw
```

所以 `image_processing.py` 是 `llm_engine.py` 支持多模态 messages 的底层工具。

---

## 25. 和 sequence.py 的关系

`image_processing.py` 生成：

```text
pixel_values
image_grid_thw
```

`sequence.py` 中新增字段：

```text
seq.pixel_values
seq.image_grid_thw
```

就是为了临时保存这些数据。

它们不是 decode 阶段每轮都需要的状态，而是：

```text
首次 prefill 使用
用完后清空
```

这样可以避免大图像 tensor 长期占用内存或被反复跨进程传输。

---

## 26. 和 model_runner.py 的关系

`model_runner.py` 在 `prepare_prefill()` 中会：

```text
收集 seq.pixel_values
收集 seq.image_grid_thw
构造 image_token_mask
计算 MRoPE 3D positions
把 pixel_values/image_grid_thw/image_token_mask 传给模型
```

也就是说：

```text
image_processing.py:
    负责把原始图片转成 pixel_values/image_grid_thw
    并在文本中插入 image_pad token

model_runner.py:
    负责在 prefill 时把这些张量整理成 batch
    并构造模型 forward 所需的 mask / positions
```

二者是多模态推理入口的上下游关系。

---

## 27. 和 vision_encoder.py 的关系

`vision_encoder.py` 接收的输入正是：

```text
pixel_values
image_grid_thw
```

也就是说：

```text
image_processing.py:
    生产 pixel_values/image_grid_thw

vision_encoder.py:
    消费 pixel_values/image_grid_thw
    输出 image_embeds
```

完整链路：

```text
PIL Image
  ↓ image_processing.process_image
pixel_values + image_grid_thw
  ↓ Qwen3VLVisionEncoder
image_embeds
```

所以 `image_processing.py` 是视觉 encoder 的前处理部分。

---

## 28. 和 qwen3_5.py 的关系

`qwen3_5.py` 中：

```text
Qwen3_5ForCausalLM.forward(...)
```

会接收：

```text
pixel_values
image_grid_thw
image_token_mask
```

然后：

```text
image_embeds = self.visual(pixel_values, image_grid_thw)
hidden_states[image_token_mask] = image_embeds
```

这里的 `pixel_values/image_grid_thw` 最初就是由 `image_processing.py` 生成的。

同时，`image_processing.py` 插入的 `<|image_pad|>` token 数必须和 `vision_encoder.py` 输出的 image_embeds 数量一致，否则 `image_token_mask` 和 `image_embeds` 无法正确对齐。

---

## 29. 和 rotary_embedding.py / MRoPE 的关系

`image_processing.py` 输出的 `image_grid_thw` 不只用于视觉 encoder，也用于语言模型侧 MRoPE。

前面 `model_runner.py` 中会根据：

```text
token_ids
image_grid_thw
image_token_id
```

计算：

```text
positions_3d
```

然后 `rotary_embedding.py` 中的 `InterleavedMRoPE` 使用这些 3D positions 给语言模型 attention 的 q/k 加位置编码。

因此 `image_grid_thw` 同时服务两条路径：

```text
vision_encoder.py:
    图像 patch 位置 / cu_seqlens / patch merger

language model MRoPE:
    image token 的 temporal/height/width positions
```

---

## 30. 该文件体现的工程取舍

### 30.1 简化版 messages 模板

该文件手工拼接：

```text
<|im_start|>{role}
...
<|im_end|>
```

而不是调用 tokenizer 的高级 chat_template 接口。

优点：

```text
实现简单
依赖少
便于 nano-vLLM 轻量化
```

缺点：

```text
对不同 tokenizer/chat_template 的兼容性较弱
如果模型特殊 token 模板变化，需要手动维护
```

---

### 30.2 图片路径和 PIL Image 都支持

```python
if isinstance(img, str):
    img = Image.open(img)
```

这让入口既支持：

```text
直接传图片路径
直接传 PIL Image
```

优点是方便测试。

但工程上还可以继续扩展：

```text
URL 图片
base64 图片
bytes 图片
多帧视频
```

---

### 30.3 单图复制 temporal 维度

对静态图片使用：

```text
repeat 到 temporal_patch_size
```

这是一种兼容 3D Conv patch embed 的简单做法。

它不表示图片真的有多个时间帧，而是为了满足视觉 encoder 的输入形状。

---

### 30.4 当前没有处理 video

虽然变量名里有 temporal_patch_size / grid_t，但当前 `process_messages()` 只处理：

```text
part["type"] == "image"
```

没有看到视频解码、多帧采样等逻辑。

所以它更准确地说是：

```text
图片多模态预处理
```

而不是完整视频处理。

---

## 31. 这个文件没有实现什么

需要避免误解，`image_processing.py` 没有实现：

```text
1. vision encoder
2. language model forward
3. image_embeds scatter
4. image_token_mask 构造的完整逻辑
5. MRoPE positions 计算
6. KV Cache / GDN state 管理
7. 多卡 image data broadcast
8. 视频读取和抽帧
9. URL/base64 图片输入
10. 完整 OpenAI API 多模态协议兼容
```

它只做：

```text
messages 解析 + 图片预处理 + placeholder token 插入。
```

---

## 32. 推理流程中的位置

从端到端推理看，它处在最前面：

```text
用户输入 messages
  ↓
image_processing.process_messages
  ↓
token_ids + pixel_values + image_grid_thw
  ↓
Sequence
  ↓
Scheduler
  ↓
ModelRunner.prepare_prefill
  ↓
Qwen3VLVisionEncoder
  ↓
Qwen3_5Model
  ↓
LM Head
  ↓
Sampler
```

所以它属于：

```text
输入预处理层
```

而不是：

```text
调度层
模型执行层
算子层
采样层
```

---

## 33. 和原版 nano-vLLM 的区别

原版 nano-vLLM 没有这个文件，因为它只需要：

```text
prompt 字符串 -> token_ids
```

qwen3.6 新增该文件，是为了支持：

```text
messages 中包含 image part
图片转 pixel_values
图片网格信息 image_grid_thw
文本 token 序列中插入 image placeholder
```

对比可以概括为：

| 能力 | 原版 nano-vLLM | qwen3.6 image_processing.py |
|---|---|---|
| 文本 prompt | 支持 | 支持 |
| ChatML messages | 较弱/无 | 手工拼接支持 |
| 图片输入 | 无 | 支持 image part |
| 图片 resize | 无 | smart_resize |
| 图片 normalize | 无 | mean/std normalize |
| patchify | 无 | Qwen VL 风格 reshape/permute |
| image_grid_thw | 无 | 生成 |
| image placeholder | 无 | 插入 `<|image_pad|>` |
| 多图拼接 | 无 | cat pixel_values / grid_thw |

---

## 34. 面试角度应该怎么回答

如果面试官问：

> `image_processing.py` 这个新增文件有什么作用？

可以这样回答：

`image_processing.py` 是 nano-vllm-qwen3.6 为支持多模态输入新增的预处理文件。原版 nano-vLLM 只需要把文本 prompt tokenizer 成 token ids，而多模态 Qwen-VL 类模型需要同时准备文本 token 和图像 tensor。这个文件中的 `process_messages()` 会解析 messages，如果遇到图片，就调用 `process_image()` 把 PIL Image resize 到 patch/merge 对齐尺寸，转 tensor，按 mean/std 归一化，补 temporal 维度，再按照 Qwen-VL 的 patch 顺序 reshape/permute 成 `pixel_values`，同时生成 `image_grid_thw`。然后它根据 `image_grid_thw` 和 `merge_size` 计算视觉 token 数，在文本中插入 `<|vision_start|> + <|image_pad|> * num_tokens + <|vision_end|>`，最后统一 tokenizer.encode 得到 token_ids。这样返回的 token_ids、pixel_values、image_grid_thw 可以交给 LLMEngine/Sequence/ModelRunner，后续 vision encoder 生成 image_embeds，并在 qwen3_5 模型中替换 image token 对应的 hidden states。它是多模态推理入口侧的关键桥梁。

---

## 35. 初学者最应该抓住的主线

这个文件可以用一句话理解：

```text
image_processing.py 把“带图片的 messages”
变成“token_ids + pixel_values + image_grid_thw”。
```

最重要的三件事是：

```text
1. 图片要 resize 到 patch_size * merge_size 的倍数
2. 图片要转成 vision encoder 需要的 pixel_values
3. 文本里要插入和 image_embeds 数量一致的 image_pad token
```

其中最核心的对齐公式是：

```text
num_image_tokens = grid_t * (grid_h // merge_size) * (grid_w // merge_size)
```

---

## 36. 最终结论

`image_processing.py` 是 `nano-vllm-qwen3.6` 多模态支持链路中的入口预处理文件。

它的核心意义是：

```text
让 nano-vLLM 可以从纯文本 prompt 输入，扩展到支持包含图片的 messages 输入。
```

它完成了：

1. **智能 resize**  
   将图像高宽调整到 `patch_size * merge_size` 的倍数，并控制像素预算。

2. **图像 tensor 化和归一化**  
   将 PIL Image 转成 RGB tensor，并归一化到视觉 encoder 期望的分布。

3. **时间维度适配**  
   对静态图片复制 temporal 维度，以适配 3D Conv patch embedding。

4. **Qwen-VL 风格 patchify**  
   按照 temporal/spatial patch 与 merge 排列，把图像转成 `pixel_values`。

5. **生成 image_grid_thw**  
   记录每张图片的 `[grid_t, grid_h, grid_w]`，供 vision encoder 和 MRoPE 使用。

6. **插入 image placeholder tokens**  
   根据 merged image token 数，在文本中插入 `<|image_pad|>`，保证视觉 embedding 和语言 token 序列对齐。

7. **输出统一多模态输入**  
   返回 `token_ids + pixel_values + image_grid_thw`，供后续 `Sequence`、`ModelRunner`、`vision_encoder.py` 和 `qwen3_5.py` 使用。

因此，它在整个 qwen3.6 项目中的位置是：

```text
image_processing.py:
    多模态输入预处理

vision_encoder.py:
    图像内容编码

qwen3_5.py:
    图像 embedding 和语言 token 融合

model_runner.py:
    prefill 阶段组织 batch、MRoPE positions 和 forward 参数
```

如果说 `vision_encoder.py` 是“看图”的模块，那么 `image_processing.py` 就是“把图整理成模型能看的格式”的模块。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
