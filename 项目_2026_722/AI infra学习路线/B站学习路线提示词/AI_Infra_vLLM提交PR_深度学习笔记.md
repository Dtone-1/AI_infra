# AI-Infra 规划课与模拟面：怎么给 vLLM 提交 PR  
## 长视频字幕整理稿 + 技术主线分析 + 开源贡献路线 + 面试问答

> 输入材料：  
> 1. `AI-Infra规划课和模拟面_2026年6月6日-怎么给vLLM提交pr_哔哩哔哩_bilibili.txt`  
> 2. `课程内容深度整理学习笔记生成提示词.md`  
>
> 处理说明：原始字幕来自自动识别，存在错别字、断句错误、技术名词误识别和大量口语重复。本笔记不是逐字转写，而是基于字幕内容进行技术名词纠错、语义复原、结构化整理和学习路线分析。  
>
> 重要提醒：vLLM 开源贡献流程可能随社区演进而变化。本文中的 PR 方法论以课程内容为主，同时补充通用开源协作原则；正式提交 PR 前应再检查 vLLM 官方 Contributing 文档、PR 模板和 CI 要求。

---

## 目录

1. [这节课一句话结论](#一这节课一句话结论)  
2. [课程核心宗旨](#二课程核心宗旨)  
3. [字幕技术名词纠错表](#三字幕技术名词纠错表)  
4. [按课程推进顺序整理知识点](#四按课程推进顺序整理知识点)  
5. [整节课的技术主线：从面试暴露问题到 vLLM PR 闭环](#五整节课的技术主线从面试暴露问题到-vllm-pr-闭环)  
6. [vLLM 开源贡献的完整方法论](#六vllm-开源贡献的完整方法论)  
7. [vLLM 重点特性学习地图](#七vllm-重点特性学习地图)  
8. [结合你的 AI Infra 学习目标：如何吸收这节课](#八结合你的-ai-infra-学习目标如何吸收这节课)  
9. [简历项目如何写得更像 AI Infra](#九简历项目如何写得更像-ai-infra)  
10. [面试高频问题与参考答案](#十面试高频问题与参考答案)  
11. [最终执行清单](#十一最终执行清单)

---

# 一、这节课一句话结论

这节课的核心不是单纯讲“GitHub 怎么点按钮提 PR”，而是讲：

> **AI Infra 实习生想通过 vLLM 开源贡献提升简历辨识度，不能只停留在会用框架或复现 nano-vLLM；必须先选定一个 vLLM 模块，理解它的原理、使用场景和源码实现，再从真实 issue、日常使用问题、社区新特性不完善处切入，完成可验证、可解释、可维护的小 PR。**

课程中反复强调几个判断：

1. **简历被反馈“技术深度不足”，很多时候不是项目名不够好，而是项目细节回答不扎实。**  
   比如多卡并行、FlashAttention 版本差异、prefix caching、MLA 并行、CUDA Graph、AWQ 等，如果写在简历里，就要能回答清楚。

2. **开源贡献最靠谱的切入口不是凭空想一个大 feature，而是从具体模块的小问题开始。**  
   典型来源包括：自己使用 vLLM 时发现的 bug、社区 issue 里的报错、新模型或新推理方法刚接入时的不完善点。

3. **对实习生来说，不建议一开始横向铺 AI 编译器、训练框架、云原生、Agent 等太多方向。**  
   推理框架本身知识量已经足够大。与其面很广但都不深，不如先把 vLLM 核心特性学扎实。

4. **AI 编程工具可以辅助写代码，但不能替代工程判断。**  
   真正的差异在于：能不能提出清晰需求、拆解问题、理解 AI 生成代码、验证行为、积累实现模式、保证可维护性。

5. **HPC / 物理计算 / CPU 并行背景可以转 AI Infra，但要补大模型结构和 CUDA / GPU 侧知识。**  
   HPC 的性能优化思路和 AI 算子优化高度相通，但需要把场景迁移到 LLM 模型结构、GPU kernel、KV Cache、推理服务上。

---

# 二、课程核心宗旨

这节课主要围绕三类学生的问题展开：

## 1. 已经做过 vLLM / nano-vLLM 项目，但面试被反馈技术深度不足

这类学生的问题通常不是“完全没项目”，而是：

- 项目能跑，但对 vLLM 细节理解不够；
- 简历写了 AWQ、CUDA Graph、Tensor Parallel、prefix caching 等关键词，但面试追问时说不深；
- 对多卡并行、FlashAttention、MLA、cache 机制等高频点掌握不稳定；
- 只知道功能名，不清楚为什么需要这个功能、在什么场景下生效、源码里如何实现、如何验证效果。

课程给出的解决方向是：

> **不要只背八股，要围绕一个 vLLM 特性建立“原理 → 使用 → 源码 → 实验 → PR”的闭环。**

## 2. 想给 vLLM 提交 PR，但不知道从哪里入手

主讲人的建议是：

- 先选择一个自己感兴趣且相对熟悉的方向；
- 不要上来就做巨大功能；
- 先从 issue、bug、文档、小模型支持、小特性完善入手；
- 对某个模块足够熟悉后，再跟进新方法、新论文、新模型、新推理策略；
- 提 PR 前要能复现问题、解释问题、给出测试和文档。

也就是说，vLLM PR 不是“为了简历硬凑一个 commit”，而是一个完整的工程能力证明。

## 3. HPC / 非科班 / Agent 背景学生想转 AI Infra

课程中对这类学生的核心建议是：

- HPC 优化经验有价值，因为性能瓶颈分析、并行计算、CPU/GPU 加速思路是相通的；
- 但必须补 AI 背景知识，至少知道大模型由哪些层组成，Qwen / DeepSeek / LLaMA 等模型大概有什么结构；
- 如果目标是算子开发，vLLM 不需要学到特别深，但需要了解基本推理流程；
- 如果目标是推理框架，需要系统理解 vLLM 的请求调度、KV Cache、batching、多卡并行、模型执行流程。

---

# 三、字幕技术名词纠错表

| 字幕识别结果 | 建议修正 | 说明 |
|---|---|---|
| 技术深度不足 / 技术深度不够 | 技术深度不足 | 面试反馈的核心问题 |
| 潜坠缓存 / 显追缓存 / 显坠缓存 | Prefix Caching / 前缀缓存 | vLLM / SGLang 中常见 KV Cache 复用机制 |
| 位置无关的潜追缓存 | 位置无关的 Prefix Caching | 一类更复杂的前缀复用问题，涉及 prompt 结构变化后缓存是否仍可复用 |
| MLA | Multi-head Latent Attention | DeepSeek 系列相关注意力机制 |
| 多卡兵型 / 五迪兵型 | 多卡并行 / 5D 并行 | TP、PP、DP、EP、CP 等并行策略组合 |
| Flasher Tenshin | FlashAttention | 注意力优化经典算子 |
| OnlineSautomus | Online Softmax | FlashAttention 的关键基础 |
| AIB&E器 | AI 编译器 | TVM、MLIR、IREE、Triton compiler 等方向 |
| 讯联优化 | 训练优化 / 训练 Infra | 分布式训练、并行训练、通信优化等 |
| 云原生 | Cloud Native | K8s、容器、调度、服务治理等 |
| 医术 / 一球 | Issue | GitHub issue |
| PR | Pull Request | 开源代码贡献请求 |
| AZN / A&J / AI Ending | AI Coding / AI 编程助手 | Codex、Claude Code、Cursor、Copilot 等 |
| Duwag / Dewag | Debug | 调试 |
| AWQ | Activation-aware Weight Quantization | 常见 LLM 权重量化方法 |
| Tesla Parallel | Tensor Parallel | 张量并行 |
| Cudagraph / 库达国亚福 | CUDA Graph | 减少 kernel launch overhead 的 CUDA 执行图 |
| 扛派了 | torch.compile | PyTorch 2.x 编译优化入口 |
| VAM / 威拉马 / 维拉马 | vLLM | 大模型推理服务框架 |
| Nano VR / Nano VAM | nano-vLLM | vLLM 简化学习项目 |
| SFlap / Swap | Swap | KV Cache / block 在 CPU 与 GPU 之间换入换出 |
| Reconpute | Recompute | 重算策略 |
| Beam Sircing | Beam Search | 多候选序列搜索解码方法 |
| 生疼 | 昇腾 / Ascend | 华为 NPU 生态 |
| NPI | MPI | Message Passing Interface，HPC 多进程通信 |
| NKG / NCL | NCCL | NVIDIA 集合通信库 |
| SomeX / Softarmark | Softmax | 常见 CUDA 面试算子 |
| Transport | Transpose | 矩阵转置算子 |
| MathMal | MatMul / GEMM | 矩阵乘法算子 |

---

# 四、按课程推进顺序整理知识点

## 4.1 第一位同学：面试反馈“技术深度不足”到底是什么意思

### 核心情节

第一位同学已经参加过小红书等公司的面试，HR 反馈“技术深度和岗位有 gap”。同学复盘后提到几个没有答好的问题：

- 位置无关的 prefix caching；
- MLA 如何并行；
- 多卡并行是否了解；
- FlashAttention 各版本差异；
- vLLM 相关 feature 的源码和使用细节不够熟。

### 核心知识点

面试官所谓的“技术深度不足”，通常不是指你不知道某个名词，而是指你不能把一个技术点讲成完整闭环：

```text
这个技术点解决什么问题？
它为什么在推理系统里重要？
它在 vLLM 中大概怎么实现？
它和哪些模块交互？
你有没有实际用过？
打开或关闭它后性能/显存/吞吐有什么变化？
它有什么限制？
```

如果只能回答“听说过”“大概知道”，就会被认为深度不足。

### 相关概念解释

#### Prefix Caching

Prefix Caching 指多个请求共享相同 prompt 前缀时，可以复用已经计算出的 KV Cache，减少重复 prefill 计算。

常见场景：

- 系统 prompt 固定；
- 多轮对话；
- RAG 模板固定；
- Agent 工具调用模板固定；
- 批量评测时 prompt 前半部分相同。

难点在于：真实请求不一定完全连续、完全顺序一致。如果 prompt 结构变化、片段顺序变化、共享部分不是简单前缀，普通 prefix caching 可能失效。因此面试官问“位置无关的 prefix caching”，本质是在考察你是否理解 cache key、token 位置、RoPE、KV Cache 复用条件之间的关系。

#### MLA 并行

MLA，即 Multi-head Latent Attention，是 DeepSeek 系列中用于降低 KV Cache 压力的重要注意力设计。面试问“MLA 如何并行”，已经不是普通八股，而是进一步考察：

- MLA 的 Q/K/V 或 latent 表示如何切分；
- tensor parallel 下哪些维度能切；
- 哪些阶段需要通信；
- cache 如何保存；
- 和普通 MHA/GQA 的并行差异。

对于实习生，如果没有专门看过实现，答不出来是正常的。但如果简历里写了 DeepSeek、MLA、模型适配，就要准备。

#### FlashAttention 版本差异

面试中常见问法：

- FlashAttention v1 解决了什么问题？
- v2 相比 v1 改了什么？
- v3 为什么和 Hopper 相关？
- Online Softmax 在其中起什么作用？
- 为什么 FlashAttention 是 IO-aware？

回答时不能只说“更快”。要围绕 HBM 访问、分块计算、避免显式 materialize attention matrix、warp/block 划分、Hopper 的 TMA/WGMMA/FP8 等讲。

### 在 AI Infra 中的作用

这些问题都指向一个核心：

> **大模型推理框架不是 model.forward 的简单封装，而是一个围绕显存、调度、并发、算子、并行和模型结构共同优化的系统。**

如果你要做推理框架岗，就不能只懂单个算子或只懂 nano-vLLM 的主流程。你需要知道 vLLM 里每个 feature 为什么存在，它对应的性能瓶颈是什么。

---

## 4.2 面试高频点：多卡并行、FlashAttention、Prefix Caching

### 核心知识点

主讲人认为：

- 多卡并行是比较常规的问题；
- FlashAttention 版本差异也是常规问题；
- Prefix Caching 如果没有加“位置无关”限定，也是重要考点；
- MLA 并行属于更深入、更偏的追问。

### 多卡并行为什么重要

大模型通常无法只靠单卡高效运行，尤其是：

- 模型参数量大；
- KV Cache 显存压力大；
- 并发请求多；
- MoE 模型专家分布复杂；
- 长上下文导致单卡显存不足。

常见并行策略：

| 并行方式 | 英文 | 主要切分对象 | 常见通信 |
|---|---|---|---|
| 张量并行 | Tensor Parallelism, TP | 权重矩阵维度 | AllReduce / AllGather |
| 流水线并行 | Pipeline Parallelism, PP | 模型层 | Send / Recv |
| 数据并行 | Data Parallelism, DP | 请求或 batch | 通常推理中更多用于多副本 |
| 专家并行 | Expert Parallelism, EP | MoE expert | All-to-All |
| 上下文并行 | Context Parallelism, CP | 序列长度 | Ring / AllGather 等 |

面试回答要注意：不要只背缩写，要能说出“切什么、为什么切、通信发生在哪里”。

---

## 4.3 AI 编程工具对工程能力的影响

### 核心情节

同学问：AI 编程工具现在已经可以写代码、优化算子，那人还需要什么能力？

主讲人给出两个关键建议：

1. **提问方式和需求表达能力很重要。**  
   同样是让 AI 写代码，需求是否清晰会显著影响结果质量。

2. **不能只合入 AI 生成代码，要积累 AI 的实现思路。**  
   要看它怎么解决问题，把实现方式沉淀成文档，这相当于“看高手棋谱”。

### 核心知识点

AI 编程工具可以做：

- 生成第一版代码；
- 修复语法错误；
- 根据报错定位问题；
- 辅助写单元测试；
- 解释已有代码；
- 帮助查找调用链；
- 改写部分实现。

但它不能完全替代：

- 判断需求是否合理；
- 判断修改是否会破坏架构；
- 判断性能瓶颈是否真实；
- 设计实验验证；
- 对代码可维护性负责；
- 对 PR 质量和测试结果负责。

### 在 AI Infra 中的作用

AI Infra 项目代码复杂，尤其是 vLLM 这种大型项目。AI 生成的 patch 可能能通过局部测试，但可能带来：

- 隐式行为变化；
- 与 scheduler / KV cache manager / worker 的交互问题；
- 特定模型或硬件后端上的兼容问题；
- CI 难以覆盖的边界问题；
- 可维护性下降。

所以正确方式是：

```text
AI 辅助生成
  → 人类理解代码
  → 人类补充测试
  → 人类验证行为
  → 人类写清楚 PR 描述
  → 人类承担责任
```

这也是开源社区越来越强调 AI-assisted contribution 透明度和人工审查的原因。

---

## 4.4 是否要扩展到训练优化、AI 编译器、云原生

### 核心情节

同学担心推理框架岗位不多，想问是否应该学习训练优化、AI 编译器、云原生，以扩大可投岗位范围。

主讲人的建议比较明确：

- AI 编译器和训练优化都不容易学；
- 推理框架本身知识量已经很大；
- 实习阶段应先把推理相关知识学扎实；
- 云原生可以等实习或项目有需求时再补；
- 不建议在求职初期过度横向扩张。

### 为什么不建议过早横向扩张

AI Infra 的几个方向都很深：

| 方向 | 学习难点 |
|---|---|
| 推理框架 | KV Cache、调度、batching、多卡并行、量化、spec decoding、源码复杂 |
| 算子开发 | CUDA、Triton、GPU 架构、性能分析、GEMM、Attention |
| AI 编译器 | IR、Pass、MLIR、TVM、图优化、lowering、codegen |
| 训练优化 | 分布式训练、ZeRO/FSDP、反向传播、通信、显存优化 |
| 云原生 | K8s、容器、服务治理、资源调度、监控、弹性伸缩 |

对于实习求职，最危险的状态是：

> 每个方向都能说几个名词，但没有一个方向能经得住面试官深入追问。

---

## 4.5 如何给 vLLM 提交 PR：从哪里找需求

### 核心情节

同学问：完全没有参与过 vLLM 开发，第一步该怎么做？

主讲人给出的路线是：

1. 先确定自己想做哪个方向；
2. 选择一个 vLLM 模块或 feature；
3. 深入理解它的实现；
4. 从日常使用、issue 列表、新特性支持不完善三个来源寻找贡献点。

### PR 需求来源一：自己日常使用发现问题

这通常来自真实业务或实验场景：

- 某个模型跑不起来；
- 某个参数组合报错；
- 某个后端不兼容；
- 某个 feature 打开后行为异常；
- 某个 benchmark 表现不符合预期；
- 某个 warning / error 信息不清晰。

优点：

- 问题真实；
- 容易复现；
- PR 描述有说服力；
- 修改范围通常较小。

### PR 需求来源二：社区 issue

如果没有公司业务场景，可以看其他开发者的 issue：

- 查找与你熟悉模块相关的 issue；
- 尝试复现报错；
- 缩小最小复现代码；
- 判断是用户使用问题、文档问题还是 vLLM bug；
- 如果是 bug，再尝试修复。

适合初学者的 issue 类型：

- 文档不清楚；
- error message 不友好；
- 某个参数组合缺少校验；
- 某个模型支持缺失；
- 测试覆盖不足；
- 边界条件报错；
- small bug / good first issue。

### PR 需求来源三：新特性刚接入时的不完善点

例如：

- 新的 speculative decoding 方法；
- MTP；
- 新模型架构支持；
- 新量化方式；
- 多模态模型支持；
- 新硬件后端；
- 新的 attention backend；
- 新版本 torch / CUDA / Python 兼容问题。

这类贡献难度更高，但含金量也更高。

---

## 4.6 简历修改：标准实习生简历不是问题，问题在基础深度

### 核心情节

主讲人看第一位同学简历后认为：简历本身没有明显问题，是一份标准实习生简历。问题更可能出在：

- 推理基础不够扎实；
- 简历中写的 feature 没有准备好；
- AWQ、CUDA Graph、Tensor Parallel 等技术点被问深时答不上来；
- 缺少实际使用和实验验证。

### 简历写法原则

不要只写：

> 支持 AWQ 量化，支持 CUDA Graph，支持 Tensor Parallel。

更好的写法是：

> 在 nano-vLLM / vLLM 学习项目中实现 AWQ W4A16 权重量化推理路径，补充权重加载、反量化计算和精度对齐逻辑；基于相同 prompt / output length 设计 benchmark，对比 FP16 与 AWQ 在显存占用、TTFT、TPOT 和 tokens/s 上的差异。

核心是写清楚：

```text
做了什么
为什么做
改了哪里
如何验证
效果如何
有什么限制
```

---

## 4.7 三年研究生规划：实习、开源、论文、比赛的优先级

### 核心情节

同学问：三年研究生阶段应该如何规划？是否要做项目、开源、比赛、论文？

主讲人建议：

- 如果实验室有推理/系统方向积累，可以尝试论文；
- 如果没有相关积累，自己硬发高水平系统论文难度较大；
- 竞赛可以关注国产芯片厂、昇腾、沐曦等组织的算子或推理优化比赛；
- 开源贡献可以作为简历亮点；
- 实习是最有效的项目经历来源；
- 要持续强化 vLLM feature 的原理、使用、源码和实验。

### 推荐优先级

对 AI Infra 求职而言，优先级大致是：

```text
相关实习 > 高质量开源 PR / 项目 > 系统化技术博客 / benchmark 报告 > 普通比赛 > 与方向弱相关论文
```

但如果实验室本身能做系统顶会或高质量论文，那论文会非常有价值。

---

## 4.8 第二位同学：HPC / 物理计算背景如何转 AI Infra

### 核心情节

第二位同学是物理 / 计算方向，平时用计算解决科研问题，项目里有 CPU 并行、MPI、祖传 C 代码优化，但 CUDA 改造不多。

主讲人的建议：

- HPC 背景转 AI Infra 是可行的；
- 传统 HPC 性能优化和 AI 算子优化思路相通；
- 需要补大模型结构和 AI 背景；
- 可以把科研项目改造成 CUDA / 多卡版本；
- 如果目标算子岗，vLLM 不必学得特别深。

### HPC 和 AI Infra 的共通点

| HPC 优化 | AI Infra 对应能力 |
|---|---|
| CPU 并行 | GPU 并行 / CUDA |
| MPI 多进程通信 | NCCL 多卡通信 |
| 性能瓶颈定位 | Nsight / roofline / profiling |
| 数值计算 kernel 优化 | GEMM / attention / norm 算子优化 |
| 祖传代码改造 | 推理框架模块改造 |
| 多节点计算 | 分布式训练 / 推理 |

### 需要补的短板

HPC 同学常见短板不是性能优化思路，而是 AI 背景：

- Transformer 是什么；
- LLM 有哪些层；
- attention、MLP、RMSNorm、RoPE、MoE 在做什么；
- Qwen / LLaMA / DeepSeek 模型结构；
- 推理为什么分 prefill / decode；
- KV Cache 为什么重要；
- 量化如何影响显存和速度。

---

## 4.9 第三位同学：Agent 背景转 AI Infra，nano-vLLM 项目如何包装

### 核心情节

第三位同学之前做 Agent / 搜索 / 应用相关实习，想转 AI Infra。其项目包括：

- nano-vLLM 二次开发；
- Beam Search；
- Swap；
- Recompute；
- 调度策略；
- KV Cache 换入换出；
- GM / GEMM 相关 CUDA 项目。

主讲人认为：

- nano-vLLM 项目做得比较扎实；
- 但简历需要从问题出发，而不是罗列功能；
- Agent 实习与 Infra 相关性不强，是否保留取决于简历空间；
- 后续应该转向 vLLM，而不是继续深挖 nano-vLLM。

### nano-vLLM 项目如何写

不要写成：

> 实现 Beam Search、Swap、Recompute、Scheduler。

应该写成：

> 针对长序列请求在显存不足时被频繁抢占、重算成本高的问题，在 nano-vLLM 中实现 KV Cache Swap 与 Recompute 策略：设计 GPU/CPU block 换入换出机制，并在 scheduler 中增加保护策略，降低长请求被反复 preempt 的概率；实现 Beam Search 分支序列管理和 copy-on-write 逻辑，支持多候选序列共享 prefix cache。

这类写法更能体现：

- 问题背景；
- 系统瓶颈；
- 设计思路；
- 数据结构；
- 与 scheduler / block manager 的交互；
- 工程复杂度。

---

# 五、整节课的技术主线：从面试暴露问题到 vLLM PR 闭环

这节课可以串成一条完整主线：

```text
面试反馈技术深度不足
  ↓
复盘被问住的问题
  ↓
发现问题集中在 vLLM feature 的原理、源码、实验闭环不足
  ↓
选择一个具体方向：prefix caching / AWQ / CUDA Graph / speculative decoding / TP / scheduler
  ↓
先学原理和使用场景
  ↓
再看 vLLM 源码实现
  ↓
实际跑模型和 benchmark
  ↓
从 issue / bug / 新特性不完善处找小切口
  ↓
复现问题
  ↓
设计修复方案
  ↓
补测试和文档
  ↓
提交 PR
  ↓
应对 review
  ↓
将 PR 和技术报告沉淀到简历
```

这个流程的本质是：

> **把“我学过 vLLM”升级成“我理解 vLLM 的一个模块，并且能基于真实问题对它做工程改造”。**

对求职来说，后者的说服力远强于前者。

---

# 六、vLLM 开源贡献的完整方法论

## 6.1 第一步：不要先问“我要提什么 PR”，先问“我要深挖哪个模块”

适合实习生切入的模块：

| 方向 | 难度 | 适合程度 | 说明 |
|---|---:|---:|---|
| 文档 / 示例 / how-to | 低 | 高 | 容易合并，但简历含金量有限 |
| 错误信息 / 参数校验 | 低 | 高 | 适合第一个 PR |
| 新模型支持的小修复 | 中 | 高 | 含金量较好 |
| 量化相关修复 | 中 | 中高 | 需要理解权重加载和 kernel 路径 |
| Prefix Caching / KV Cache | 中高 | 高 | 和推理框架核心强相关 |
| Scheduler / request 管理 | 中高 | 高 | 面试价值高，但更容易引入复杂 bug |
| Speculative Decoding | 中高 | 高 | 新特性多，机会多 |
| CUDA / Triton kernel | 高 | 中 | 要求 profiling 和测试充分 |
| PD 分离 / 大架构改造 | 很高 | 低 | 不适合第一个 PR |

建议选择：

```text
一个你简历里已经写过的模块
+
一个你能本地跑通实验的场景
+
一个改动范围可控的问题
```

---

## 6.2 第二步：建立本地开发环境

通用流程：

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
# 创建 Python 环境，安装开发依赖
# 按官方文档选择 CUDA / CPU / 其他硬件后端的安装方式
# 安装 pre-commit
# 跑最小测试
```

需要注意：

- 开发 Python 代码和开发 CUDA/C++ 代码的环境复杂度不同；
- CUDA/C++ 改动需要更长编译时间；
- 本地 GPU 不足时，可以先做 Python 层小修复；
- 对大型测试无法本地全跑时，至少要跑相关单测和最小复现脚本。

---

## 6.3 第三步：找到小而真实的问题

寻找方式：

1. GitHub issue 搜索关键词：
   - `prefix caching`
   - `AWQ`
   - `CUDA graph`
   - `speculative decoding`
   - `Qwen`
   - `DeepSeek`
   - `TP`
   - `scheduler`
   - `KV cache`
   - `good first issue`

2. 用自己熟悉的模型跑：
   - Qwen；
   - LLaMA；
   - DeepSeek Distill；
   - 小尺寸多模态模型；
   - AWQ / GPTQ 量化模型。

3. 改参数组合：
   - `max_model_len`;
   - `tensor_parallel_size`;
   - `enable_prefix_caching`;
   - `enforce_eager`;
   - `quantization`;
   - `speculative_config`;
   - `max_num_batched_tokens`;
   - `gpu_memory_utilization`.

4. 观察：
   - 报错；
   - warning；
   - 性能异常；
   - 文档与实际行为不一致；
   - 某模型特殊结构不支持；
   - 参数组合缺少校验。

---

## 6.4 第四步：复现和最小化

一个合格 PR 的前提是能复现问题。

复现材料最好包括：

```text
vLLM commit / version
CUDA / PyTorch / Python 版本
GPU 型号
模型名称
启动命令
请求样例
报错日志
预期行为
实际行为
最小复现脚本
```

如果不能稳定复现，就不要急着改代码。

---

## 6.5 第五步：定位源码路径

常见路径思路：

| 问题类型 | 可能关注模块 |
|---|---|
| API 参数行为异常 | entrypoints / OpenAI server / arg parsing |
| 请求调度问题 | scheduler / engine core |
| KV Cache 分配问题 | block manager / kv cache manager |
| 模型 forward 问题 | model runner / worker / model executor |
| 新模型支持问题 | model definitions / registry / config |
| 量化问题 | quantization / weight loading / kernel dispatch |
| CUDA Graph 问题 | compilation / cudagraph capture / replay |
| kernel 问题 | csrc / attention backend / Triton kernels |
| 分布式问题 | distributed / parallel state / worker |

源码定位方法：

```text
先从报错栈定位入口
  ↓
找到关键函数
  ↓
用日志/断点确认数据流
  ↓
找最近的测试文件
  ↓
看相似 feature 的实现
  ↓
只做最小必要修改
```

---

## 6.6 第六步：实现、测试、文档

PR 不应只包含代码，还要考虑：

- 单元测试；
- 集成测试；
- 文档更新；
- benchmark；
- backward compatibility；
- error message；
- 类型标注；
- 对多硬件后端的影响；
- 对已有 feature 的兼容。

如果是性能优化 PR，最好提供：

| 指标 | 说明 |
|---|---|
| TTFT | 首 token 延迟 |
| TPOT | 每输出 token 延迟 |
| throughput | tokens/s 或 requests/s |
| GPU memory | 显存占用 |
| benchmark config | 模型、输入长度、输出长度、并发数 |
| before / after | 修改前后对比 |

---

## 6.7 第七步：PR 描述应该怎么写

推荐结构：

```markdown
## What does this PR do?
说明这个 PR 修改了什么。

## Why is this needed?
说明原来的问题、影响范围、触发场景。

## How is it implemented?
说明核心实现思路、涉及模块、关键逻辑。

## Tests
列出本地跑过的测试、benchmark 或最小复现。

## Limitations
说明当前没有覆盖的情况或后续工作。

## Notes
如果使用 AI 辅助生成代码，应按社区要求说明。
```

简历价值最高的不是“我提过 PR”，而是你能解释：

> 这个 PR 解决了什么真实问题，为什么原来有问题，我怎么定位的，我怎么验证没有引入回归。

---

# 七、vLLM 重点特性学习地图

## 7.1 PagedAttention

### 解决什么问题

KV Cache 动态增长、多请求并发时容易造成显存碎片和浪费。PagedAttention 借鉴操作系统分页思想，把 KV Cache 拆成 block 管理。

### 面试要点

- 为什么 KV Cache 会占大量显存；
- block table 是什么；
- logical block 和 physical block 的区别；
- prefill 和 decode 阶段 KV Cache 如何增长；
- prefix caching 如何复用 block；
- 和传统连续内存分配相比有什么优势。

---

## 7.2 Prefix Caching

### 解决什么问题

多个请求共享前缀时，避免重复 prefill。

### 面试要点

- cache key 如何设计；
- token ID、位置编码、RoPE 是否影响复用；
- 什么情况下不能复用；
- 多轮对话 / RAG / Agent 中为什么常见；
- 与 PagedAttention 的关系；
- prefix cache 命中率如何影响 TTFT。

---

## 7.3 Tensor Parallel

### 解决什么问题

模型太大或单卡计算不够时，把权重矩阵切到多卡上。

### 面试要点

- attention 的 QKV projection 怎么切；
- MLP 的 gate/up/down projection 怎么切；
- Row Parallel 和 Column Parallel 区别；
- 哪些地方需要 AllReduce；
- TP 和 DP 的区别；
- TP 对显存和通信的影响。

---

## 7.4 AWQ 量化

### 解决什么问题

通过权重量化降低显存占用，提高部署效率。

### 面试要点

- AWQ 和 GPTQ 的区别；
- W4A16 是什么意思；
- 为什么通常权重量化比激活量化更容易；
- 反量化发生在哪里；
- 量化后为什么可能不一定更快；
- 精度如何验证；
- 在 vLLM 中量化模型如何加载和执行。

---

## 7.5 CUDA Graph

### 解决什么问题

减少 CPU 端反复 launch kernel 的开销，适合 shape 相对稳定的推理片段。

### 面试要点

- capture 和 replay 是什么；
- 为什么动态 shape 会影响 CUDA Graph；
- vLLM 如何选择可 capture 的 batch size；
- enforce eager 对 CUDA Graph 有什么影响；
- CUDA Graph 和 torch.compile 有什么区别；
- 什么情况下 CUDA Graph 加速明显。

---

## 7.6 Speculative Decoding

### 解决什么问题

用 draft model 或其他候选生成方式一次提出多个 token，再由 target model 验证，减少大模型逐 token 调用次数。

### 面试要点

- draft model 和 target model 怎么协作；
- accept / reject 逻辑；
- 为什么分布接近时收益更大；
- 和 MTP、EAGLE 等方法的关系；
- 推理框架中调度和 KV Cache 如何配合；
- 哪些场景可能没有收益。

---

## 7.7 FlashAttention

### 解决什么问题

标准 attention 会产生巨大的 attention score 矩阵，并带来大量 HBM 读写。FlashAttention 通过分块和 Online Softmax 减少 HBM 访问。

### 面试要点

- safe softmax 和 online softmax；
- 为什么不 materialize 完整 attention matrix；
- FA1 / FA2 / FA3 的差异；
- Hopper 上 TMA / WGMMA 的作用；
- 为什么 FlashAttention 是 IO-aware；
- decode attention 和 prefill attention 的差异。

---

# 八、结合你的 AI Infra 学习目标：如何吸收这节课

你当前目标是 AI Infra / 大模型推理实习，并且正在学习 nano-vLLM、vLLM、CUDA、Transformer。对你来说，这节课最值得吸收的是以下几点。

## 8.1 不要只做“学习型项目”，要做“问题型项目”

学习型项目写法：

> 学习 nano-vLLM，了解调度、KV Cache、PagedAttention。

问题型项目写法：

> 针对 nano-vLLM 长 prompt 请求在显存不足时被频繁抢占导致重复 prefill 的问题，设计基于 block 访问分数的 KV Cache 压缩/淘汰策略，并在 scheduler 中加入长请求保护机制，降低重复计算开销。

后者更像 AI Infra 项目。

## 8.2 你接下来应该选一个 vLLM 模块深挖

结合你的背景，优先级建议：

1. **Scheduler + KV Cache Manager**  
   最贴近 nano-vLLM，迁移学习成本低。

2. **Prefix Caching / PagedAttention**  
   面试高频，和你已经学过的 block manager 有联系。

3. **Benchmark / Profiling**  
   最适合做简历数据，能形成 TTFT、TPOT、吞吐、显存对比。

4. **AWQ / GPTQ 量化推理路径**  
   如果你后续想做量化部署或推理优化，可以作为项目亮点。

5. **CUDA Graph / torch.compile**  
   有一定深度，但需要实际跑实验，不要只背概念。

## 8.3 你暂时不应该发散太多

不建议现在主攻：

- AI 编译器；
- 训练框架；
- 云原生；
- Agent；
- 大型 PD 分离；
- 从零写复杂 kernel。

这些方向都可以了解，但短期求实习最重要的是：

```text
vLLM 推理主线
+
CUDA 高频算子
+
简历项目实验数据
+
面试八股与手撕
```

---

# 九、简历项目如何写得更像 AI Infra

## 9.1 vLLM / nano-vLLM 项目简历模板

```text
基于 nano-vLLM 的 LLM 推理调度与 KV Cache 优化实验
- 阅读并复现 nano-vLLM 的 Engine、Scheduler、BlockManager、ModelRunner 等核心模块，梳理请求从 tokenization、prefill、decode 到 streaming output 的端到端流程；
- 针对长 prompt 请求在显存不足时被 preempt 后重复 prefill 的问题，设计 KV Cache block 评分与压缩策略，保留高相关历史 block，降低重复计算开销；
- 在 scheduler 中加入请求状态管理和长请求保护策略，支持 WAITING / RUNNING / SWAPPED / FINISHED 等状态转换；
- 设计 benchmark 脚本统计 TTFT、TPOT、吞吐量、显存占用和 preemption 次数，对比优化前后的长上下文推理表现；
- 结合 vLLM 的 PagedAttention、Prefix Caching 和 Continuous Batching 机制，整理源码阅读笔记和面试问答。
```

## 9.2 vLLM PR 简历模板

```text
vLLM 开源贡献与推理特性调试
- 选取 vLLM 中 [Prefix Caching / AWQ / Speculative Decoding / Scheduler] 模块作为开源贡献方向，阅读相关源码并复现社区 issue；
- 针对 [具体问题] 设计最小复现脚本，定位到 [具体模块/函数] 中的边界条件处理缺失；
- 提交 PR 修复 [问题]，补充单元测试与文档说明，确保相关测试通过；
- 在 Qwen / LLaMA 模型上验证修改前后行为一致性，并记录性能和显存变化。
```

## 9.3 CUDA 算子项目简历模板

```text
CUDA 高频算子实现与性能分析
- 使用 CUDA C++ 实现 Reduce、Softmax、Transpose、MatMul 等高频算子，掌握 block/grid 划分、shared memory、warp-level reduction、coalesced memory access 等优化方法；
- 对 Softmax 实现 naive、block-level reduction、online softmax 等版本，使用 Nsight Compute 分析 memory throughput、occupancy、warp stall 和 bank conflict；
- 对 MatMul 实现 shared memory tiling 版本，并与 PyTorch / cuBLAS baseline 对比不同矩阵规模下的性能；
- 总结常见面试手撕版本，能够解释每个优化点对应的硬件瓶颈。
```

---

# 十、面试高频问题与参考答案

## 1. 面试官说你“技术深度不足”，一般指什么？

答：通常不是指完全不会，而是指技术点没有形成闭环。比如写了 AWQ、CUDA Graph、Tensor Parallel，但只能解释名词，不能说清楚它解决什么问题、在什么场景下使用、源码大概怎么实现、如何验证效果、有什么限制。AI Infra 面试更看重“原理 + 工程实现 + 实验验证”的组合能力。

---

## 2. Prefix Caching 解决什么问题？

答：Prefix Caching 解决多个请求共享相同 prompt 前缀时的重复 prefill 计算问题。LLM prefill 阶段会为 prompt 计算 KV Cache，如果不同请求前缀相同，就可以复用这部分 KV Cache，从而降低 TTFT、减少计算开销。它常见于系统 prompt 固定、多轮对话、RAG 和 Agent 场景。

---

## 3. Prefix Caching 为什么不是简单字符串匹配？

答：因为真正复用的是 token 对应的 KV Cache。需要考虑 tokenizer 结果、token 位置、RoPE 或位置编码、模型配置、cache block 边界等。如果 prompt 片段顺序变化或位置变化，普通 KV Cache 未必还能直接复用。位置无关 prefix caching 更复杂，需要重新定义可复用条件。

---

## 4. PagedAttention 和 Prefix Caching 有什么关系？

答：PagedAttention 解决 KV Cache 的 block 化管理问题，把 KV Cache 拆成物理 block，用 block table 映射请求的逻辑 token block。Prefix Caching 可以基于这些 block 复用已有前缀 KV Cache。前者提供高效的内存管理基础，后者利用共享前缀减少重复计算。

---

## 5. vLLM 为什么需要 Continuous Batching？

答：LLM 请求长度和生成长度都不固定。静态 batch 会被长请求拖慢，短请求结束后 batch 位置浪费。Continuous Batching 在每个 decode step 动态加入新请求、移除完成请求，使 GPU 持续保持较高利用率，从而提升吞吐。

---

## 6. Prefill 和 Decode 的性能特征有什么区别？

答：Prefill 处理整个 prompt，通常计算密集，attention 可以并行处理多个 token；Decode 每次生成一个 token，通常更受 KV Cache 读取、显存带宽、kernel launch、调度开销影响。长 prompt 会放大 prefill 成本，高并发长输出会放大 decode 阶段的显存带宽压力。

---

## 7. Tensor Parallel 的基本思想是什么？

答：Tensor Parallel 把大矩阵权重按维度切分到多张 GPU 上并行计算。例如 QKV projection 或 MLP projection 可以按列或按行切分。切分后某些位置需要 AllReduce 或 AllGather 合并结果。它用于模型单卡放不下或单卡算力不足的场景。

---

## 8. TP 和 DP 有什么区别？

答：TP 是把一个模型的单层权重切到多卡上，一个请求的同一次 forward 需要多卡协同；DP 是复制多份模型，不同 GPU 处理不同请求或 batch。TP 主要解决模型太大或单请求算力不足，DP 主要提升并发吞吐。

---

## 9. FlashAttention 为什么快？

答：FlashAttention 不是减少理论 FLOPs，而是减少 HBM 读写。它通过分块计算和 Online Softmax 避免显式保存完整 attention matrix，让更多中间计算留在 SRAM/shared memory/register 层级，从而降低内存 IO，提升实际速度。

---

## 10. Online Softmax 是什么？

答：Online Softmax 是一种分块计算 softmax 的方法。它在逐块读取 attention score 时维护当前最大值和归一化分母，使 softmax 在不一次性拿到完整向量的情况下仍能保持数值稳定。这是 FlashAttention 能够分块计算的关键。

---

## 11. FlashAttention v1、v2、v3 大概有什么区别？

答：v1 主要提出 IO-aware attention，通过 tiling 和 online softmax 减少 HBM 访问；v2 改进并行划分和计算组织，提高 GPU 利用率；v3 面向 Hopper 架构，利用 TMA、WGMMA、FP8 等新硬件能力进一步优化。面试回答应围绕算法分块、并行粒度和硬件特性展开。

---

## 12. CUDA Graph 在推理中解决什么问题？

答：CUDA Graph 通过捕获一段 GPU 执行图并重复 replay，减少 CPU 端频繁 kernel launch 的开销。它适合 shape 较稳定、执行路径重复的推理场景。LLM 推理中由于 batch size 和 sequence length 动态变化，需要通过 bucket 或固定 shape 策略提升 graph 复用率。

---

## 13. AWQ 是什么？

答：AWQ 是 Activation-aware Weight Quantization，即激活感知的权重量化方法。它通过分析激活分布，保护对输出更敏感的权重通道，使 W4A16 等低比特权重量化在降低显存占用的同时尽量保持精度。

---

## 14. 量化后为什么不一定更快？

答：量化降低了权重读取带宽和显存占用，但可能引入反量化开销、kernel 不够优化、batch/shape 不适配、访存模式变化等问题。如果没有高效 int4/int8 kernel 或融合反量化，实际速度未必提升。

---

## 15. Speculative Decoding 的核心思想是什么？

答：用小模型或 draft 机制先生成多个候选 token，再用目标大模型一次性验证。如果候选 token 被接受，就相当于减少了目标模型逐 token forward 的次数，从而加速 decode。收益取决于 draft token 接受率和验证开销。

---

## 16. vLLM PR 应该从什么类型的问题开始？

答：适合从小而真实的问题开始，比如文档补充、错误信息改善、参数校验、某个模型的小兼容问题、issue 中可复现的小 bug、测试覆盖不足等。不要一开始做 PD 分离、大规模 scheduler 重构或复杂 kernel 重写。

---

## 17. 如何判断一个 issue 适不适合作为第一个 PR？

答：看四点：能否稳定复现，改动范围是否可控，是否有明确预期行为，是否能补测试。如果只能大概猜测原因，或者涉及多个复杂模块交互，不适合作为第一个 PR。

---

## 18. 提 vLLM PR 前应该准备哪些信息？

答：应准备 vLLM 版本或 commit、环境信息、模型名称、启动命令、请求样例、实际报错、预期行为、最小复现脚本、修改思路、测试结果。如果是性能优化，还要提供 benchmark 配置和修改前后指标。

---

## 19. AI 编程工具生成的代码可以直接提交 PR 吗？

答：不应该直接提交。人需要审查所有代码、理解实现、验证端到端行为、补测试，并按社区要求披露 AI 辅助情况。AI 可以辅助实现，但 PR 责任仍然由提交者承担。

---

## 20. 为什么不要过早横向学习 AI 编译器和训练框架？

答：这些方向本身都很深。实习阶段如果同时铺开推理、训练、编译器、云原生，很容易每个方向都浅。推理框架本身已经包括调度、KV Cache、多卡、量化、算子、服务化等大量内容，短期内先把推理方向做深更有效。

---

## 21. HPC 背景转 AI Infra 的优势是什么？

答：HPC 背景通常熟悉性能优化、并行计算、瓶颈定位、数值计算和通信，这些与 AI 算子优化高度相通。短板是对大模型结构、GPU kernel、LLM 推理流程、KV Cache 和推理框架不熟，需要补 AI 背景。

---

## 22. 算子岗需要把 vLLM 学到多深？

答：算子岗不一定需要深入 vLLM 源码，但至少要了解 LLM 推理基本流程、Transformer 结构、常见算子来自哪里、KV Cache 是什么、PagedAttention 大概解决什么问题。如果投推理框架岗，则需要更深入理解 vLLM 的 scheduler、engine、worker、model runner 等模块。

---

## 23. nano-vLLM 项目会不会太普通？

答：单纯“看过 nano-vLLM”确实普通。但如果在 nano-vLLM 上做了自己的 feature，例如 KV Cache 压缩、Swap/Recompute、Beam Search、Prefix Caching、AWQ、Benchmark，并能讲清楚问题背景、实现和实验，就不普通。

---

## 24. 简历里写 Nsight Systems / Nsight Compute 会被问什么？

答：面试官可能问：你用它看过哪些指标？怎么定位瓶颈？看到了哪些现象？如何根据 profiling 结果修改代码？如果只是写了工具名但没有实际定位问题的经历，会被认为不扎实。

---

## 25. 如何把开源 PR 写进简历？

答：不要只写“给 vLLM 提交 PR”。应该写 PR 解决的问题、涉及模块、核心修改、测试和效果。例如：“针对 vLLM 在某量化模型加载时的参数校验缺失问题，复现社区 issue，定位到 quantization config 解析逻辑，补充边界检查和单元测试，提交 PR 修复。”

---

# 十一、最终执行清单

## 1. 一周内完成

- 选定 vLLM 深挖方向：Scheduler / KV Cache / Prefix Caching / AWQ / Speculative Decoding 中选一个；
- 跑通 vLLM 本地开发环境；
- 用一个小模型跑通 OpenAI-compatible API；
- 记录一次 TTFT、TPOT、吞吐、显存；
- 阅读对应模块 2~3 个核心文件。

## 2. 两周内完成

- 找 5 个相关 issue；
- 复现其中 1~2 个；
- 写最小复现脚本；
- 对照源码定位问题；
- 做一个小修复或文档补充；
- 跑相关测试。

## 3. 一个月内完成

- 提交 1 个小 PR；
- 写一篇技术博客或 Markdown 报告；
- 把 PR / issue 复现过程整理到简历；
- 准备该模块 10 个面试问答；
- 对比 nano-vLLM 和 vLLM 中对应模块的实现差异。

## 4. 简历检查清单

每个项目都必须能回答：

```text
为什么做？
原来有什么问题？
你改了哪里？
涉及哪些模块？
怎么验证正确性？
性能指标是什么？
有没有和 baseline 对比？
有什么局限？
面试官如果让我现场画流程图，我能不能画出来？
```

## 5. 最终建议

对于 AI Infra 推理方向，最有效的成长路径不是“资料越看越多”，而是：

```text
跑通一个系统
  → 读懂一个模块
  → 复现一个问题
  → 修掉一个 bug
  → 补一个测试
  → 提一个 PR
  → 写一篇报告
  → 把它讲成面试项目
```

这也是这节课“怎么给 vLLM 提交 PR”背后的真正含义。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-AI infra学习路线|模块-AI infra学习路线]]

%% 项目关联导航：结束 %%
