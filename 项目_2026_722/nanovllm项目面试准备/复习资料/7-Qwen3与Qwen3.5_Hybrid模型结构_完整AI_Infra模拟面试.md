# Qwen3 与 Qwen3.5 Hybrid 模型结构：AI Infra 模拟技术面试

> **面试背景**：候选人基于 nano-vLLM 适配 Qwen3.5 Hybrid 推理架构，并在此基础上扩展多模态输入、MTP 投机解码和 KV Cache 压缩能力。  
> **面试目标**：不是背诵模型名词，而是确认候选人是否真正理解“模型结构为什么这样设计、一次推理如何执行、结构变化为什么会牵动推理引擎，以及多模态数据如何进入语言模型”。
>
> **资料范围**：本文结合两份学习材料，以及仓库 `Dtone-1/nano-vllm-qwen3.6` 的 `feat/qwen36-kv-compression` 分支源码整理。重点核对了 `qwen3.py`、`qwen3_5.py`、`gated_delta_net.py`、`model_runner.py`、`scheduler.py`、`sequence.py` 和 `context.py`。
>
> **命名说明**：仓库中用于承载 Qwen3.5/Qwen3.6 Hybrid 语言模型的实现文件名为 `qwen3_5.py`。本文按照简历口径统一称为“Qwen3.5 Hybrid”，所有回答均限定在该仓库的实际实现范围内，不把项目实现无限外推为所有官方模型版本的通用结论。

---

# 问题 1：请介绍一下 Qwen3 的模型结构。

## 参考回答

先说结论，Qwen3 Dense 是一个 Decoder-only 的自回归语言模型。它没有独立的 Encoder，输入是一串 Token ID，模型按照从左到右的因果关系处理这串 Token，并根据前面的上下文预测下一个 Token。

从整体结构看，输入 Token ID 首先经过词嵌入层，变成 Hidden States；然后依次经过多层 Decoder Layer；所有层处理完成后，再经过最终的 RMSNorm，接 LM Head 映射到词表维度，得到每个候选 Token 的 Logits；最后由 Sampler 根据 Greedy 或温度等采样策略选出下一个 Token。生成出来的 Token 会追加到当前序列中，下一轮再作为 Decode 输入，因此整个生成过程是自回归的。

单个 Decoder Layer 仍然是标准的 Pre-Norm Transformer 结构，主要包含两个子层。第一个子层是 Causal Self-Attention，第二个子层是 SwiGLU MLP，两部分外面都有残差连接。进入 Attention 前先做 RMSNorm，Attention 计算完成后把结果加回残差主干；然后再次做 RMSNorm，进入 MLP，MLP 输出再通过残差进入下一层。

Attention 内部会把输入 Hidden States 投影成 Query、Key 和 Value。Query 和 Key 会做归一化，并通过 RoPE 注入位置信息；然后执行因果 Attention，保证当前位置只能看到自己及其之前的 Token，不能看到未来 Token。Qwen3 使用 GQA，也就是 Query Head 的数量可以多于 Key/Value Head 的数量。这样做的工程意义是，模型仍然可以保留较多 Query Head 的表达能力，但需要缓存的 K/V Head 更少，因此能够减少 KV Cache 的显存占用和 Decode 阶段的访存压力。

MLP 部分使用 SwiGLU。可以理解为输入同时经过 Gate 分支和 Up 分支，Gate 分支先做 SiLU 激活，再和 Up 分支逐元素相乘，最后经过 Down Projection 回到 Hidden Size。nano-vLLM 为了提高推理效率，会把 Gate Projection 和 Up Projection 合并成一次线性投影，但数学语义仍然是 SwiGLU。

从推理引擎角度看，Qwen3 的模型结构只负责定义“每个 Token 要经过哪些计算”，而 Prefill、Decode、KV Cache 的物理位置和批处理边界由 ModelRunner 和 Context 负责组织。Prefill 阶段一次处理 Prompt 中的多个 Token，并把各层新产生的 K/V 写入 KV Cache；Decode 阶段每个请求通常只输入最新的一个 Token，再从 KV Cache 中读取历史 K/V。也就是说，完整的 Qwen3 推理不能只看模型文件，还要把模型结构、运行时 Context、Attention Kernel 和 KV Cache 管理连在一起理解。

---

# 问题 2：Qwen3.5 的结构在此基础上有什么改变？

## 参考回答

先说最核心的变化：Qwen3.5 不再是“所有 Decoder Layer 都使用 Full Attention”的结构，而是变成了 Full Attention 和 Gated DeltaNet 混合的 Hybrid Decoder。

在 Qwen3 Dense 中，每一层的 Token Mixing 模块都是 Causal Full Attention，因此每一层都需要保存历史 Token 的 Key 和 Value。Qwen3.5 则会根据每层的类型配置，决定这一层走 Full Attention 还是 Gated DeltaNet。无论走哪一种 Token Mixing，后面仍然会接 MLP，所以单层外壳仍然可以理解为“Norm、Token Mixing、Norm、MLP”，只是中间的 Token Mixing 不再固定为 Attention。

这个变化首先改变了历史状态的保存方式。Full Attention 层仍然使用 KV Cache，保存历史 Token 级别的 K/V，而且缓存大小会随着上下文长度增加。Gated DeltaNet 层不保存每个历史 Token 的完整 K/V，而是维护 Convolution State 和 Recurrent State，通过递推方式压缩历史信息。因此，Qwen3.5 Hybrid 同时存在两套历史状态：一套是 Full Attention 的 KV Cache，另一套是 GDN 的 Conv/Recurrent State。

第二个变化是，Qwen3.5 的 Full Attention 本身也和 Qwen3 不同。原版 Qwen3 在工程上把 Q、K、V 合并到一个 QKV Projection 中；Qwen3.5 将 q_proj、k_proj 和 v_proj 分开，其中 q_proj 的输出不是只有 Query，而是同时包含 Query 和 Gate。Attention 计算完成后，输出还要乘上 `sigmoid(Gate)`，再经过输出投影。可以把 Query 理解为“从历史中读取什么”，把 Gate 理解为“读取回来的信息允许通过多少”。

第三个变化是 Norm 和位置编码。Qwen3.5 的主干以及 Query/Key 归一化采用 GemmaRMSNorm，它和普通 RMSNorm 的权重语义不同，推理实现必须与 Checkpoint 对齐。位置编码使用 Interleaved MRoPE，它既能处理纯文本的一维位置，也能处理图片 Token 的 Temporal、Height、Width 三维位置，因此为多模态输入提供了语言模型侧的位置建模能力。

第四个变化是模型外壳增加了可选分支。仓库中的 Qwen3.5 模型可以按配置挂载 Vision Encoder 和 MTP。Vision Encoder 把图片转换成与语言模型 Hidden Size 对齐的视觉 Embedding，再替换文本序列中图片占位位置的普通 Token Embedding；MTP 则使用主模型 Hidden 和当前 Token Embedding 生成 Draft Hidden。但这两个分支都是可选能力，不是每次普通文本 Forward 都必须执行。

从 AI Infra 角度看，真正困难的不是把一个 GDN 类加入模型，而是要让整个推理系统理解两套状态。KV Cache 不能再按照总层数分配，而要只按 Full Attention 层数分配；每个活跃请求还要动态获得一个 GDN State Slot；Scheduler、Sequence、ModelRunner、Context、Chunked Prefill、抢占重算和 CUDA Graph 都必须携带并正确更新这个 Slot。只有这些系统层都一致，才能说完成了 Qwen3.5 Hybrid 推理适配。

---

# 问题 3：你刚才说 Qwen3 是 Dense、Decoder-only、Causal LM，还使用了 GQA。能分别解释这几个概念吗？它们之间是什么关系？

## 参考回答

这几个词描述的是模型的不同维度，不能混在一起。

Decoder-only 描述的是模型拓扑。它表示模型只有 Transformer Decoder Block，没有像传统 Encoder-Decoder 模型那样单独先用 Encoder 编码输入。所有输入 Token 和后续生成 Token 都放在同一条序列中，由 Decoder 按因果顺序处理。这种结构非常适合自回归文本生成。

Causal LM 描述的是训练和推理目标。模型在某个位置只能利用当前及之前的 Token，预测下一个 Token，不能看到未来内容。在 Attention 中，这个约束通过 Causal Mask 实现。也正因为每个新 Token 都依赖前面已经生成的 Token，Decode 过程天然是串行的。

Dense 描述的是参数使用方式。Dense 模型中，每个 Token 都经过同一套 MLP 参数，不会像 MoE 那样先经过 Router，再只选择少数专家。这里的 Dense 不是简单说“参数很多”，也不是说所有矩阵从存储形式上都必须是稠密矩阵，而是和专家路由结构相对。

GQA 描述的是 Attention Head 的组织方式。普通 Multi-Head Attention 可以让每个 Query Head 都有独立的 K/V Head；Multi-Query Attention 则让所有 Query Head 共享很少的 K/V Head；GQA 位于两者之间，让多个 Query Head 共享一组 K/V Head。推理时，KV Cache 的大小主要由 K/V Head 数量决定，因此 GQA 可以明显降低缓存和带宽成本，同时保留较多 Query Head。

它们之间没有互相替代关系。一个模型可以同时是 Decoder-only、Causal LM、Dense，并在 Attention 中使用 GQA。Qwen3 Dense 就可以用这四个维度共同描述。到了 Qwen3.5，Decoder-only 和 Causal LM 的整体范式仍然保留，MLP 仍然可以是 Dense，但 Token Mixing 从全层 Full Attention 变成 Full Attention 与 GDN 的 Hybrid。

面试时我会特别说明，Hybrid 不等于 MoE。Hybrid 在这个项目中主要指不同层使用不同类型的 Token Mixing；MoE 则是 MLP 专家路由，两者描述的不是同一件事。

---

# 问题 4：不要只讲宏观结构，请你按一次真实 Forward 的顺序，把 Qwen3 的一个 Decoder Layer 讲清楚。

## 参考回答

可以。我从一个已经完成 Embedding 的 Hidden States 开始讲。

进入 Decoder Layer 后，首先走 Pre-Norm。也就是先对输入做 RMSNorm，再送进 Self-Attention。nano-vLLM 在代码中维护了一条单独的 Residual 主干，并把部分 Residual Add 和 RMSNorm 融合到同一个算子接口中，所以代码写法和教科书图不完全一样，但数学语义仍然是 Pre-Norm 加残差。

经过 Norm 后，Hidden States 进入 QKV Projection。原版 Qwen3 的工程实现把 Q、K、V 三个投影合并为一个较大的并行线性层，一次计算后再按各自维度切分。切分出来的 Q、K、V 会进一步 reshape 成多头形式。由于使用 GQA，Query Head 数可能多于 K/V Head 数，因此 Q 和 K/V 的 Head 维度不一定相同。

接下来，Query 和 Key 会分别做 QK Norm，再应用 RoPE。RoPE 不是把一个位置向量直接加到 Hidden States 上，而是根据 Position 对 Q/K 的部分维度进行旋转，使 Attention Score 自然包含相对位置信息。Value 不参与 RoPE，因为位置主要通过 Q 和 K 的匹配关系影响 Attention 权重。

之后进入真正的 Attention Kernel。这里 Prefill 和 Decode 走不同路径。Prefill 中，一个 Batch 可能包含多个长度不同的 Prompt，ModelRunner 会把 Token 打平成一维序列，并通过累计长度描述各请求边界；Attention 使用 Varlen FlashAttention，同时保持 Causal 约束。每个新 Token 的 K/V 还会按照 Slot Mapping 写入对应层的 KV Cache。

Decode 时，每个请求通常只输入一个最新 Token。模型只需要计算这个 Token 的 Q、K、V；当前 K/V 写入新的 Cache Slot，而 Attention 会根据 Context Length 和 Block Table 读取历史 KV Cache，计算当前 Query 对全部历史上下文的 Attention。这样就不用为每个新 Token 重新计算整段历史的 K/V。

Attention 输出仍然是多头形式，先把 Head 维度展平，再经过 o_proj 回到模型 Hidden Size。这个输出通过残差连接加入主干，然后再做一次 RMSNorm，进入 SwiGLU MLP。

MLP 中，输入同时经过 Gate 和 Up 两个投影，Gate 分支做 SiLU，再和 Up 分支逐元素相乘，最后通过 Down Projection 回到 Hidden Size。MLP 输出继续沿残差主干传给下一层。

全部 Decoder Layer 完成后，还要做一次 Final RMSNorm。模型主体输出的是 Hidden States，而不是直接输出 Token。之后 LM Head 才把最后需要采样的位置映射到词表 Logits，Sampler 再选择下一个 Token。把主干 Forward 和 LM Head 分开，可以避免 Prefill 时为所有 Prompt Token 都计算完整词表 Logits，通常只需要为每个请求最后一个有效位置计算即可。

---

# 问题 5：Qwen3.5 的 Full Attention 为什么要把 q_proj、k_proj、v_proj 分开？Query 和 Gate 到底怎么工作？

## 参考回答

先说结论，q_proj、k_proj 和 v_proj 分开不是简单的代码风格变化，而是因为 Qwen3.5 的 Query 分支已经不再是普通 Query 投影，它还要同时产生一个 Gate。

在 Qwen3 中，Q、K、V 都是比较标准的线性投影，所以可以把三份权重在工程上打包成一个 QKV Projection。一次矩阵乘后，再按照 Query 和 K/V 的各自大小切分，能够减少算子调用，也方便 Tensor Parallel 实现。

Qwen3.5 中，q_proj 的输出维度是普通 Query 的两倍。投影结果先按 Head reshape，每个 Head 的最后一维再被切成两半，一半是真正用于 Attention 的 Query，另一半是 Gate。k_proj 和 v_proj 仍然分别产生 Key 和 Value。由于 q_proj 的语义和输出形状已经与 K/V 明显不同，继续强行放进原来的统一 QKV 模块会增加切分、权重映射和并行逻辑的复杂度，因此仓库直接使用三个独立投影。

Query 会经过 GemmaRMSNorm 和 Interleaved MRoPE，然后与 Key 计算 Attention Score。Gate 不参与 Query-Key 点积，也不决定 Attention 权重。Attention 根据 Q、K、V 得到多头输出以后，才把输出与 `sigmoid(Gate)` 逐元素相乘，然后再经过 o_proj。

直观地说，Query 负责提出问题：“我当前应该从历史上下文中读取哪些信息？”Attention 根据 Q/K 匹配把信息从 Value 中聚合回来；Gate 再决定：“这些已经取回来的信息，在当前 Token、当前 Head、当前通道上允许通过多少？”

因为 `sigmoid` 把 Gate 映射到零到一之间，所以 Gate 较小时，对应通道的 Attention Output 会被抑制；Gate 较大时，信息大部分保留。这提供了比普通 Attention 更细粒度的动态信息流控制。

在 Tensor Parallel 下，q_proj 和 k/v_proj 都按输出维度切分，每张卡负责一部分 Head；o_proj 再按输入维度切分并做跨卡规约。虽然投影拆开会增加独立线性调用，但这是模型结构和 Checkpoint 本身要求的语义，首先必须保证正确性，不能为了复用原 QKV 打包而错误改变权重布局。

---

# 问题 6：Gated DeltaNet 层具体取代了 Attention 的哪部分？它为什么需要 Conv State 和 Recurrent State？

## 参考回答

Gated DeltaNet 取代的是 Decoder Layer 中的 Token Mixing 模块，也就是原本 Full Attention 所在的位置。它不是取代整个 Decoder Layer，因为 GDN 计算完成后，后面仍然会经过 Post-Norm、SwiGLU MLP 和残差路径。

Full Attention 处理历史的方法，是显式保存每个历史 Token 的 Key 和 Value。当前 Token 的 Query 可以直接和所有历史 Key 计算相似度，再聚合对应 Value。这种方式保留了细粒度历史，但缓存会随着上下文线性增长。

Gated DeltaNet 的思路不同。它不把完整历史 Token 全部留在 Cache 中，而是通过递推规则把历史压缩进状态。仓库实现中主要有两类状态。

第一类是 Convolution State。GDN 输入会先投影出混合的 Q/K/V 特征，然后经过因果 Depthwise Conv1D。卷积核只需要最近有限长度的输入，因此状态中保存的是卷积窗口尚需延续的历史特征。Prefill 时可以对一段 Token 做因果卷积，同时更新尾部状态；Decode 时只输入一个新 Token，就把旧窗口左移并写入新值。

第二类是 Recurrent State。它可以理解为 Gated Delta Rule 维护的压缩记忆矩阵。每个新 Token 会根据当前 Key、Value、衰减门和更新强度，先衰减旧状态，再用当前信息修正状态，最后用 Query 从更新后的状态中读取输出。它不是简单把所有历史平均到一个向量，而是一种按 Head 维护的结构化递推记忆。

Prefill 阶段，仓库可以用 Chunk 形式处理一段序列并得到最终 State；对于已经存在历史状态的继续 Prefill，也可以按 Token 递推更新。Decode 阶段则严格使用单 Token Recurrent 路径，根据当前请求对应的 State Index，读出该请求在每个 GDN Layer 上的 Conv/Recurrent State，完成更新后写回同一 Slot。

因此，GDN State 的大小主要由模型维度和 GDN 层数决定，而不是由上下文 Token 数直接决定。它能够减少长上下文下逐 Token KV 的增长，但也带来一个重要约束：状态更新是有顺序的。你不能随意跳过某个 Token，也不能在 Reject 后只删除逻辑 Token 而不恢复 State，否则后续输出会从错误的递推状态继续。

Hybrid 结构的目的可以理解为，让部分 Full Attention 层保留直接访问 Token 级历史的能力，让部分 GDN 层用固定大小递推状态降低长上下文成本。但它不是“GDN 一定在所有方面优于 Attention”，而是在建模能力、状态大小和推理复杂度之间做结构性折中。

---

# 问题 7：你提到了 Interleaved MRoPE 和可选 Vision Encoder。请把一张图片进入 Qwen3.5 Hybrid 主干的完整逻辑讲清楚。

## 参考回答

先说结论，图片不能直接作为 Token ID 送进语言模型。多模态链路需要同时准备文本序列、图像像素特征、图片占位位置和多维位置编码，最后把图像编码结果注入统一的 Hidden States 序列。

用户输入通常是包含文本和图片的 Messages。入口侧会把文本部分组织成对话模板并 Tokenize；图片则经过尺寸调整、归一化和 Patch 化，生成 Vision Encoder 需要的 Pixel Values，同时生成 `image_grid_thw`，描述图片在时间、高度和宽度方向上的 Patch 网格。

文本序列中还会插入一段图片占位 Token。占位数量不能随便设置，而要与 Vision Encoder 最终输出的视觉 Token 数严格一致。这样语言模型序列就提前知道图像内容应该放在哪些位置。

到了首次 Prefill，ModelRunner 会把 Pixel Values、网格信息、图片位置 Mask 和 Position 一起组织进 Batch。Vision Encoder 接收 Pixel Values 和 Grid 信息，把原始 Patch 投影成视觉 Hidden，经过视觉 Transformer 和 Patch Merge，最终输出与语言模型 Hidden Size 对齐的 Image Embeddings。

语言模型先正常对全部 Input IDs 查词嵌入，得到一条文本 Hidden States 序列。图片占位 Token 此时也会有普通词嵌入，但它只负责预留位置，不代表真实图像内容。随后模型根据 Image Token Mask，把这些占位位置上的普通 Embedding 替换成 Vision Encoder 输出的 Image Embeddings。替换完成后，文本 Token 和图像 Token 已经具有相同 Hidden Size，会一起进入后续的 Qwen3.5 Hybrid Decoder Layer。

Interleaved MRoPE 解决的是语言模型侧的位置问题。纯文本 Token 只需要一个一维序列位置；图像 Token 则来自二维空间，框架还保留了 Temporal 维度，因此会为图像 Token 构造 T、H、W 三组位置。Interleaved MRoPE 按模型规定的维度区间，把三类位置频率交错应用到 Query 和 Key 上。这样语言模型不仅知道这些视觉 Token 位于整条对话的哪个片段，也能保留它们在图像网格中的相对空间关系。

需要区分两套位置建模。Vision Encoder 内部也有自己的视觉位置编码，用来理解 Patch 之间的空间关系；Interleaved MRoPE 则发生在图像 Embedding 已经进入语言模型以后，用于图像 Token 和文本 Token 共同参与 Decoder Attention 时的位置建模，两者服务的阶段不同。

图片通常只在首次 Prefill 运行一次 Vision Encoder。Prefill 结束后，图像信息已经通过各层计算写入 Full Attention 的 KV Cache，并影响 GDN 的递推状态。后续 Decode 每轮只输入新生成的文本 Token，不会重新编码同一张图片，而是依靠已经建立的 KV Cache 和 GDN State 延续图像上下文。

---

# 问题 8：Qwen3 和 Qwen3.5 在 Prefill 与 Decode 阶段分别维护什么？为什么 Qwen3.5 更容易出现状态一致性问题？

## 参考回答

Prefill 和 Decode 的分界，可以理解为完整 Prompt 是否已经被模型处理完。

对于 Qwen3 Dense，Prefill 会一次处理一段 Prompt Token。每个 Full Attention 层为这些 Token 计算 K/V，并根据 Slot Mapping 写入 KV Cache；同时通过 Varlen Attention 完成 Prompt 内部的因果计算。Prefill 结束后，取每个请求最后一个有效位置的 Hidden，经过 LM Head 和 Sampler 得到第一个生成 Token。

Decode 阶段，每个请求通常每轮只输入最新一个 Token。当前 Token 产生新的 K/V，写入下一个 Cache Slot；Query 通过 Block Table 和 Context Length 读取此前所有物理 KV，得到当前输出。也就是说，Qwen3 的核心历史状态是每层 KV Cache，以及 Scheduler 用来描述这些 Cache 的 Block Table 和逻辑长度。

Qwen3.5 Hybrid 在 Prefill 中要同时推进两类状态。Full Attention 层仍然写 KV Cache；GDN 层则要按 Prompt 顺序更新 Conv State 和 Recurrent State。Decode 时同样如此：Full Attention 读取并追加 KV，GDN 根据请求的 State Index 读取旧状态，计算后写回新状态。

因此 Qwen3.5 有两条必须同步的时间线。第一条是逻辑 Token 序列，表示请求已经接受了哪些 Token；第二条是物理模型状态，包含 Full Attention 的 KV 长度和 GDN 的递推状态。只要其中一条比另一条多走或少走一步，后续结果就可能错误。

Chunked Prefill 是一个典型场景。Prompt 可能因为 Token Budget 被分成多段执行。第一段 Prefill 结束后，Full Attention 已经写入部分 KV，GDN 也已经形成部分 State；下一段必须在这些状态基础上继续，而不能重新从零开始，也不能重复更新已经处理过的 Token。

抢占重算也是典型场景。如果请求因为资源不足被抢占，KV Block 可能被释放，GDN State Slot 也会回收。请求重新进入 Prefill 时，需要根据完整逻辑历史重算，并先清空新分配的 State Slot，否则会混入上一个请求留下的数据。

MTP Reject Rollback 更能体现这个问题。Target Verify 可能已经为多个候选 Token 写入 KV 并更新 GDN State；如果中途发现 Draft 错误，只从 Sequence 删除错误 Token 并不够，必须同时恢复被覆盖的 KV Slots 和各层 Conv/Recurrent State，再重跑接受前缀。

所以 Qwen3.5 的难点不是单个算子更难，而是所有执行路径都必须保证“Token 进度、KV 进度、GDN State 进度”一致。

---

# 问题 9：结合你的项目，具体说说为了支持 Hybrid 架构，你对推理引擎做了哪些系统级改造。

## 参考回答

我会把改造分成模型识别、状态分配、调度传递和执行一致性四个方面。

首先是模型识别。ModelRunner 创建模型时，会根据配置中的模型类型选择原版 Qwen3 Dense 实现，或者 Qwen3.5 Hybrid 实现。这样同一套 Engine 可以保留原 Qwen3 路径，同时为 Hybrid 模型创建对应的 Decoder Layer、可选 Vision Encoder 和可选 MTP。

第二是 KV Cache 分配逻辑。原版可以近似按总 Decoder 层数分配 KV Cache；Hybrid 中只有真正包含 K/V Cache 的 Full Attention 层需要分配。因此项目会遍历模型模块，统计实际拥有 K/V Cache 的层，并只给这些层绑定物理缓存。这样避免为 GDN 层错误分配无用的 KV。

第三是 GDN State Pool。模型初始化并加载权重后，ModelRunner 会收集所有 GDN 层，根据 Conv State 和 Recurrent State 的形状估算单个请求、所有 GDN 层需要的显存，再结合剩余显存和最大并发数计算可用 State Slot 数。每个 GDN Layer 都分配同样 Slot 数的 Conv State Pool 和 Recurrent State Pool。Recurrent State 使用更高精度保存，是为了保持递推数值稳定。

第四是请求级状态管理。Sequence 中新增 State Slot ID，用来表示这个请求在所有 GDN 层上使用哪一行状态，同时还记录该 Slot 是否需要清零。Scheduler 增加 State Slot Manager，维护空闲 Slot。请求首次进入 Prefill时，除了分配 KV Block，还会分配 State Slot，并标记需要 Reset。请求结束时释放 Slot；请求被抢占时也释放 Slot，因为重新调度后会通过 Re-Prefill 重算状态。

第五是运行时 Context。ModelRunner 在准备 Prefill 或 Decode Batch 时，不仅生成 Input IDs、Positions、Slot Mapping、Block Tables 和 Context Length，还要生成 State Indices。底层 Attention 通过 Context 读取 KV 相关信息，GDN 则通过同一个 Context 读取请求对应的 State Index。这样模型层不需要在每层 Forward 参数中反复传递整套调度元数据。

第六是 CUDA Graph。Decode Graph 捕获时，State Indices 也必须是静态输入 Buffer 的一部分；Replay 前要把当前请求的 State Index 复制到对应 Buffer。否则即使 Input Token 和 KV Block 正确，GDN 仍可能读写错误请求的状态。

第七是状态 Reset 与回滚。新分配或抢占后复用的 State Slot 必须在 Prefill 前清零；MTP Reject 等实验路径还需要保存和恢复指定请求的 KV Slot、Conv State 和 Recurrent State。只有逻辑控制面和 GPU 数据面同时恢复，才能和可信的普通 Greedy 路径保持一致。

因此，我认为这项工作的价值不只是“实现了一个 Qwen3.5 模型类”，而是让 Scheduler、缓存管理、运行时 Context 和模型执行都理解 Hybrid 状态的生命周期。

---

# 问题 10：你如何证明 Qwen3.5 Hybrid 适配成功？当前项目还有哪些边界不能夸大？

## 参考回答

我不会把“程序能运行并生成一段看起来合理的文本”当作适配成功的充分证据。我会从结构、数值、状态、功能回归和性能五个层面验证。

第一是结构验证。检查配置中的每个 Layer Type 是否正确映射到 Full Attention 或 GDN；检查 Full Attention 的 q_proj 是否正确拆分 Query 和 Gate，Gate 是否在 Attention 输出后生效；检查 q/k 使用的 Norm、Interleaved MRoPE 参数、MLP 权重打包和 Checkpoint 参数名是否对应。还要确认 Vision Encoder 与 MTP 只在开启对应配置时创建，Text-only 主路径不会被强制引入额外开销。

第二是权重与数值验证。固定 Prompt、固定 Greedy 采样，与可信实现比较生成 Token；必要时比较关键位置的 Hidden 或 Logits。分别测试短 Prompt、长 Prompt、单次完整 Prefill、Chunked Prefill 和多轮 Decode。结构错误有时不会立即报 Shape Error，但会表现为 Logits 持续偏离，因此数值对齐比“能跑”更可靠。

第三是状态验证。检查 KV Cache 实际层数只等于 Full Attention 层数；检查每个请求获得独立 State Slot，不会串状态。可以在同一状态上执行一次 Decode，保存 Token 和 Logits，然后恢复 KV 与 GDN State，再执行一次，要求结果一致。还要验证 Eager 与 CUDA Graph Replay 一致，Chunked Prefill 与一次性 Prefill 的最终状态和输出一致。

第四是资源生命周期验证。测试请求结束后 KV Block 和 State Slot 是否都被释放；测试多请求并发时 Slot 是否重复分配；测试抢占后请求重新 Prefill 是否先清空状态。多模态请求还要确认 Vision Encoder 只在首次 Prefill 执行，图片 Embedding 数量和占位位置一致，后续 Decode 不重复处理图片。

第五是性能验证。模型正确后，再评估 TTFT、TPOT、Decode Throughput、KV Cache 占用、GDN State 占用和最大并发。Hybrid 的理论优势之一是部分层不再让状态随上下文线性增长，但整个系统还会受到 Full Attention 层、GDN 算子效率、Tensor Parallel 通信和调度开销影响，所以不能只看某一项显存下降就断言整体一定更快。

项目边界也需要准确表达。第一，当前仓库实现的是 Qwen3.5/Qwen3.6 Hybrid 推理适配，不等于重现了完整训练流程。第二，多模态能力是图片输入链路，不能因为 Vision Encoder 使用 Temporal 维度就宣称已经实现完整视频处理。第三，MTP 是包含 Draft、Verify、Accept/Reject 和回滚的实验原型，当前主要面向单请求、Text-only 和 Greedy 路径，不能表述成生产级 Continuous Batching 投机解码。第四，FP8 部分如果实际做的是加载量化 Checkpoint 并分片反量化，就不能说自己实现了运行时 FP8 GEMM。第五，KV Cache 压缩只作用于 Full Attention 层，GDN State 是另一类状态，不能混称为所有层统一压缩。

我在面试中会把成果总结为：基于 nano-vLLM 打通了 Qwen3.5 Hybrid 的模型执行与双状态运行时，使 Full Attention 的 KV Cache 和 GDN 的 Conv/Recurrent State 能在 Prefill、Decode、Chunked Prefill、抢占重算和 CUDA Graph 场景下保持一致，并在这个正确基础上继续扩展多模态、MTP 和 KV Cache 压缩实验能力。

---

# 面试结束时的两分钟总述

如果面试官要求我把前面的内容压缩成一段，我会这样回答：

Qwen3 Dense 是标准的 Decoder-only Causal LM，整体由 Embedding、多层 Full Attention 加 SwiGLU MLP、Final RMSNorm 和 LM Head 组成。每层都通过 Causal Attention 建模历史，并在推理时使用 KV Cache；Qwen3 还使用 GQA，通过减少 K/V Head 降低 KV Cache 和 Decode 访存成本。

Qwen3.5 在这个基础上改成 Hybrid Decoder，每层根据配置选择 Full Attention 或 Gated DeltaNet。Full Attention 仍使用 KV Cache，但其内部变成独立 q/k/v Projection，q_proj 同时产生 Query 和 Gate，Attention 输出乘 `sigmoid(Gate)` 后再投影；Q/K 和主干使用 GemmaRMSNorm，位置编码改为兼容文本一维和多模态 T/H/W 三维位置的 Interleaved MRoPE。GDN 层则不保存逐 Token K/V，而是维护 Conv State 和 Recurrent State。

因此，适配 Qwen3.5 不能只修改模型层。推理引擎必须只为 Full Attention 层分配 KV Cache，同时为每个请求分配 GDN State Slot；Scheduler 负责 Slot 生命周期，Sequence 记录 Slot ID，ModelRunner 在 Prefill、Decode 和 CUDA Graph 中传递 State Index。多模态时，图片先经过 Vision Encoder 得到 Image Embeddings，再替换图片占位 Token 的普通 Embedding，并通过 MRoPE 保留空间位置。最终项目的核心价值，是让 nano-vLLM 从仅管理全层 KV Cache 的 Qwen3 Dense 推理，扩展为能够正确管理 KV Cache 与 GDN 双状态的 Qwen3.5 Hybrid 推理框架。

---

# 面试中最容易出现的错误表述

第一，不要把 Dense 和 Decoder-only 当成同一个概念。前者描述参数路由，后者描述模型拓扑。

第二，不要说 Qwen3.5 完全取消了 Attention。它是 Full Attention 与 GDN 混合。

第三，不要说 GDN State 是另一种逐 Token KV Cache。KV Cache 保存每个历史 Token 的 K/V，GDN 保存压缩递推状态。

第四，不要说 q_proj 只产生 Query。该仓库的 Qwen3.5 Full Attention 中，q_proj 同时产生 Query 和 Gate。

第五，不要说 Gate 参与 QK Score。Gate 在 Attention 已经得到输出后进行逐元素调制。

第六，不要把 GemmaRMSNorm 和普通 RMSNorm 当成完全相同的公式与权重语义。

第七，不要说 Interleaved MRoPE 只在图片输入时存在。纯文本也走这一模块，只是一维位置路径。

第八，不要说所有 Hybrid 层仍然分配 KV Cache。只有 Full Attention 层需要 K/V Cache，GDN 层使用 Conv/Recurrent State。

第九，不要把 Vision Encoder 输出说成 Token ID。它输出的是与语言模型 Hidden Size 对齐的视觉 Embedding。

第十，不要把“生成结果看起来正常”当成唯一正确性证明。还必须验证 Logits、KV/GDN State、Chunked Prefill、抢占、CUDA Graph 和资源回收的一致性。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nanovllm项目面试准备|模块-nanovllm项目面试准备]]

%% 项目关联导航：结束 %%
