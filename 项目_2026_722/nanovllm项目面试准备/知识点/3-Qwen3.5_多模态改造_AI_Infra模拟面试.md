# nano-vLLM Qwen3.5 多模态改造：AI Infra 模拟面试问答

> 面试背景：候选人在 nano-vLLM 上适配 Qwen3.5 Hybrid 架构，并补齐图片多模态推理链路。以下问题按照真实面试中的追问顺序组织，重点考察候选人是否理解从输入处理、视觉编码、特征融合、位置编码到推理执行的完整链路。

---

## 一、简历表述优化

### 原表述

> 支持 messages + image 多模态输入、Vision Encoder 特征注入及 Interleaved MRoPE。

### 建议表述

> **打通 messages + image 多模态推理链路，完成图像预处理、Vision Encoder 编码、视觉特征注入及 Interleaved MRoPE 适配。**

这句话比“支持多模态输入”更准确，因为本项目并不是简单增加一个图片参数，而是改造了输入预处理、请求数据传递、视觉编码、特征融合和位置编码等一整条 prefill 链路。

---

# 二、模拟技术面试

## 问题 1：请介绍一下你在 Qwen3.5 多模态部分做了哪些工作。

### 参考回答

我做的核心工作，是把原来只支持纯文本的 nano-vLLM，扩展成能够处理“文本加图片”的多模态推理框架。

原版流程比较简单，文本经过 tokenizer 变成 token ids，然后直接进入语言模型。我改造后，入口可以接收 `messages + image`。文本仍然会转成 token ids；图片则先经过尺寸调整、归一化和 patch 化，生成视觉模型需要的像素张量和网格信息。同时，我会在文本 token 序列里插入与视觉 token 数量一致的图片占位符。

到了 prefill 阶段，图片先经过 Vision Encoder，编码成与语言模型 hidden size 一致的视觉 embedding；然后把这些视觉 embedding 填到图片占位符对应的位置。文本和图片由此变成一条统一的 hidden states 序列，再一起进入 Qwen3.5 的 Hybrid Decoder。

另外，图像 token 不能只使用普通的一维位置，所以我还适配了 Interleaved MRoPE，为图像 token 构造时间、高度和宽度三个方向的位置。图片只在首次 prefill 时编码，后续 decode 不会重复运行视觉塔，而是复用已经建立好的 KV Cache 和 GDN state。

---

## 问题 2：听起来不只是多传了一个图片张量。为什么不能直接把图片作为额外参数传给模型？

### 参考回答

因为语言模型最终处理的是一条 token 序列，而图片本身不是 token id。只把图片张量传进去，模型并不知道图像信息应该出现在文本序列的哪个位置，也不知道图片最终会对应多少个视觉 token。

所以这条链路至少要同时解决三个问题。

第一，要把图片转换成 Vision Encoder 能处理的输入。第二，要在文本序列中提前预留对应数量的图片位置。第三，要保证 Vision Encoder 输出的 embedding 数量和这些预留位置严格一致。

例如一张图片经过 patch 划分和空间合并后，最终可能对应 196 个视觉 token，那么文本中也必须有 196 个图片占位 token。之后才能把 196 行视觉 embedding 一一填入这些位置。

因此，多模态改造的关键不是“增加一个图片参数”，而是让图像内容、图片占位符、视觉 embedding 和位置编码在整个 prefill 链路中保持一致。

---

## 问题 3：那一张原始图片进入系统后，具体会经过哪些预处理？

### 参考回答

图片进入系统后，我主要做了五步处理。

第一步是统一转成 RGB，避免灰度图或者带透明通道的图片导致输入通道不一致。

第二步是调整图片尺寸。尺寸不能随意缩放，而是要对齐视觉模型的 patch 划分和空间合并粒度。例如 patch 大小和 merge 大小相乘后是 32，那么图片的高和宽就需要调整成 32 的倍数，否则后面的 reshape 和 patch 合并会对不上。同时还要限制总像素数，避免大图产生过多视觉 token，导致 prefill 时间和显存占用过高。

第三步是把图片转成浮点 tensor，并按照视觉模型训练时的分布做归一化。

第四步是补齐时间维度。这个视觉塔使用三维卷积做 patch embedding，即使输入是静态图片，也需要满足时间维度的输入格式，所以会复制静态帧来适配 temporal patch。

最后，再按照模型约定的顺序做 reshape 和 permute，把图片整理成展平后的 patch 像素向量，同时生成这张图片的 `T、H、W` 网格信息。最终输出的核心就是 `pixel_values` 和 `image_grid_thw`。

---

## 问题 4：`pixel_values` 和 `image_grid_thw` 分别代表什么？为什么两个都需要？

### 参考回答

可以把它们理解成“图像内容”和“图像结构”。

`pixel_values` 保存的是每个图像 patch 的实际像素内容。经过预处理后，每一行可以看成一个时空 patch 的展平像素向量，Vision Encoder 会把这些像素向量投影成视觉 hidden states。

`image_grid_thw` 保存的是这些 patch 原来如何排列，其中 T 表示时间方向，H 和 W 表示图片在高度和宽度方向上的 patch 数量。它不保存图像内容，而是告诉后续模块这些 patch 的空间结构和每张图片的边界。

两个都需要，是因为只有 `pixel_values`，模型虽然有像素数据，但不知道 patch 原来位于哪一行、哪一列，也无法为多张不同尺寸的图片划分 attention 边界。只有 `image_grid_thw`，模型又只有形状，没有真实图片内容。

在我的实现中，`image_grid_thw` 还被复用了三次：计算图片占位 token 的数量、组织视觉编码器内部的变长 attention，以及生成语言模型侧的三维 MRoPE 位置。

---

## 问题 5：图片占位 token 的数量是怎么确定的？如果数量不一致会发生什么？

### 参考回答

图片占位 token 的数量由 Vision Encoder 最终输出多少个视觉 embedding 决定。

图片先被切成 patch，视觉塔末尾还会把相邻的多个 patch 做空间合并。假设网格大小是 `T × H × W`，空间合并比例是 `merge`，那么最终进入语言模型的视觉 token 数大致是：

```text
N = T × (H / merge) × (W / merge)
```

例如一张图片的 patch 网格是 `1 × 28 × 28`，合并比例是 2，那么最终视觉 token 数就是 `1 × 14 × 14 = 196`。文本中就需要插入 196 个图片占位 token，并用视觉开始和视觉结束标记包住它们。

prefill 时会根据这些占位 token 构造一个布尔 mask，Vision Encoder 输出 196 行 image embeddings 后，再按照 mask 把它们填入统一的 hidden states。

如果两边数量不一致，最直接的结果就是 scatter 或赋值时发生 shape mismatch。即使某些情况下没有立即报错，只要位置错位，图片特征也会被放到错误的 token 位置，模型输出就不可信。所以这是整条多模态链路中最重要的正确性约束之一。

---

## 问题 6：Vision Encoder 是怎么把图片转换成语言模型可以使用的 embedding 的？

### 参考回答

整体上可以分为 patch embedding、视觉 Transformer 编码和 patch merger 三个阶段。

首先，预处理后的 patch 像素会经过三维卷积。这个卷积的 kernel 和 stride 都对应 temporal patch 和空间 patch 大小，因此它会把每个原始像素 patch 投影成一个视觉 hidden state。

然后，视觉 hidden states 会加入位置相关信息，再通过多层视觉 Transformer。这里的 attention 是非因果的，因为编码整张图片时，每个 patch 都应该能够看到同一张图片中的其他 patch，不存在语言生成中的“不能看未来 token”。为了支持不同尺寸图片和多张图片，视觉 attention 使用变长序列边界，避免不同图片的 patch 互相做 attention。

最后，视觉塔通过 patch merger 把相邻 patch 合并。一方面可以减少进入语言模型的视觉 token 数，降低后续 prefill 成本；另一方面会把视觉 hidden size 投影到语言模型的 hidden size。

所以 Vision Encoder 最终输出的每一行 embedding，都正好对应文本序列中的一个图片占位位置。

---

## 问题 7：视觉 embedding 最后是怎么和文本融合的？为什么采用这种方式？

### 参考回答

我的实现采用的是占位替换式融合，而不是在语言模型外部单独维护一条视觉分支。

首先，文本 token 会正常经过词嵌入层，得到整条序列的 hidden states。图片占位 token 此时也有一个普通的 token embedding，但它只负责占位置，不代表真实图片内容。

Vision Encoder 生成 image embeddings 后，会根据图片 token mask，把占位位置上的普通 embedding 替换成真实的视觉 embedding。替换完成后，序列大致可以理解为：

```text
文本 hidden state
→ 多个图像 hidden state
→ 文本 hidden state
```

之后，语言模型的 Attention、GDN 和 MLP 层看到的都是统一维度的 hidden states，不需要再区分某个位置最初来自文字还是图片。

这样做的好处是，图像和文本可以复用同一套序列调度、位置编码和状态缓存机制；同时，图片在对话中的相对位置也可以由占位 token 明确表达。它本质上属于在语言模型输入 embedding 层完成的早期融合。

---

## 问题 8：Vision Encoder 内部已经有位置编码了，为什么语言模型侧还需要 Interleaved MRoPE？

### 参考回答

因为两套位置编码服务的阶段不同。

Vision Encoder 内部的位置编码，是为了让图像 patch 在视觉塔中知道彼此的空间关系，例如哪个 patch 在上方、哪个在右侧。它解决的是“视觉塔如何理解图片内部结构”。

但是图片经过 Vision Encoder 后，会作为一组视觉 token 插入语言模型的文本序列。进入语言模型后，这些视觉 token 还要和问题文本一起参与 Attention。此时语言模型不仅要知道它们在整条序列中的位置，还要保留图像 token 对应的时间、高度和宽度坐标。

普通文本可以使用一维 position，图像 token 则需要 T、H、W 三个方向的位置。Interleaved MRoPE 会分别生成这三个方向的旋转频率，再按模型规定的方式交错应用到 Attention 的 Query 和 Key 上。

所以视觉塔的位置编码和语言模型侧的 MRoPE 并不重复：前者负责图片内部的视觉编码，后者负责图像 token 和文本 token 进入统一 Decoder 后的位置建模。

---

## 问题 9：多模态请求在 prefill 和 decode 两个阶段有什么区别？这对性能有什么影响？

### 参考回答

图片主要只在首次 prefill 阶段处理一次。

prefill 时，系统需要准备完整的文本和图片输入，运行 Vision Encoder，生成 image embeddings，把它们注入 token 序列，然后让完整上下文通过 Hybrid Decoder。这个过程会写入 Full Attention 层的 KV Cache，也会更新 GDN 层的 recurrent state 和 convolution state。

prefill 完成后，原始图片 tensor 就没有必要继续保留，所以我会清理请求中临时保存的 `pixel_values` 和网格信息。后续 decode 每次只输入新生成的一个 token，不再重复做图片预处理，也不会再次运行 Vision Encoder，而是依赖 prefill 已经建立好的 KV Cache 和 GDN state 延续图片上下文。

因此，多模态输入对 TTFT 的影响比较明显，因为视觉编码和更多的 prompt token 都发生在 prefill；但进入 decode 后，单步开销主要仍由语言模型本身决定，图片不会在每一步重复编码。

从 AI Infra 角度看，图像分辨率最终会影响视觉 token 数，所以尺寸控制和 patch merge 不只是模型正确性问题，也是 TTFT、显存和吞吐之间的工程权衡。

---

## 问题 10：这次改造对调度、批处理和多卡执行有什么影响？你会怎么验证链路是正确的？

### 参考回答

调度器本身不需要理解图片语义，它看到的仍然是一条 token 序列。图片占位 token 会增加 prompt 长度，因此会影响 prefill token budget、KV Block 分配以及 Hybrid 状态初始化，但正常的 waiting、running 和 decode 调度逻辑可以继续复用。

真正需要改造的是请求数据传递和 prefill 准备。每个请求除了 token ids，还要临时携带图片 tensor 和网格信息。batch 执行时，需要把不同请求的图片数据拼接起来，根据每张图片的网格生成变长 attention 边界、图片 token mask 和三维位置。如果采用 Tensor Parallel，还要把图片张量、网格、mask 和位置从调度侧广播给所有 rank，保证各个模型分片看到一致的输入。

我会从五个层面验证。

第一，做纯文本回归，确保开启多模态支持后原有文本推理结果不受影响。第二，检查图片占位 token 数量是否严格等于 Vision Encoder 输出行数。第三，分别测试单图、多图、不同分辨率以及文本和图片交错出现的情况。第四，确认视觉塔只在首次 prefill 执行一次，decode 阶段不会重复处理图片。第五，在多卡下检查各 rank 的输入形状、位置和输出一致性，同时记录 TTFT、显存峰值和视觉 token 数之间的关系。

目前这套实现主要支持静态图片输入，还没有补齐视频抽帧、URL、Base64 和完整 OpenAI 多模态协议兼容。因此我在面试中会把它描述为“图片多模态推理链路”，而不是声称已经支持完整的视频多模态。

---

# 三、面试回答时需要始终抓住的主线

面试中不需要一开始就讲大量代码细节，可以始终围绕下面这条链路展开：

```text
messages + image
    ↓
解析文本和图片
    ↓
token_ids + pixel_values + image_grid_thw
    ↓
插入与视觉输出数量一致的图片占位 token
    ↓
prefill 阶段组织 mask 和 T/H/W 三维位置
    ↓
Vision Encoder 生成 image embeddings
    ↓
替换图片占位位置的 hidden states
    ↓
文本与图像共同进入 Hybrid Decoder
    ↓
decode 阶段复用 KV Cache 和 GDN state
```

一句话概括这项工作的技术价值：

> **把 nano-vLLM 从“只能执行纯文本 token 序列”，扩展成“能够组织、编码并执行图文统一序列”的多模态推理引擎。**

---

# 四、面试中容易说错的地方

1. 不要说 `pixel_values` 就是最终输入语言模型的图片 embedding。它只是视觉塔的像素 patch 输入，经过 Vision Encoder 后才会变成 image embeddings。
2. 不要说图片占位 token 自己包含图像语义。它只用于占位和对齐，真实图像语义来自 Vision Encoder。
3. 不要说视觉塔每次 decode 都会运行。它主要在首次 prefill 运行一次。
4. 不要把 `image_grid_thw` 只解释成图片尺寸。它描述的是 patch 网格，并同时用于占位数量、视觉 attention 边界和 MRoPE 位置。
5. 不要把当前实现描述成完整视频支持。虽然存在 temporal 维度和 3D patch embedding，但现有输入处理链路主要实现的是静态图片。
6. 不要说 Scheduler 直接处理图片。Scheduler 主要按 token 和资源进行调度，多模态 tensor 的整理发生在 prefill 执行准备阶段。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nanovllm项目面试准备|模块-nanovllm项目面试准备]]

%% 项目关联导航：结束 %%
