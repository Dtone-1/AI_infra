# AI-Infra规划课：怎么学习大模型推理框架  
## 长视频字幕整理稿 + AI Infra 深度分析 + 2027 秋招路线建议

> 生成时间：2026-06-03  
> 输入材料：自动识别字幕 txt 文件《AI-Infra规划课_怎么学习大模型推理框架_哔哩哔哩_bilibili.txt》  
> 说明：原字幕来自自动识别，存在大量错别字、断句错误、技术词识别错误。本文不是逐字转写，而是基于字幕内容进行“技术名词纠错 + 语义复原 + 结构化整理 + 学习路线分析”。

---

## 目录

1. [这期视频一句话结论](#一这期视频一句话结论)  
2. [字幕技术名词纠错表](#二字幕技术名词纠错表)  
3. [视频整体结构](#三视频整体结构)  
4. [前半部分：接近原视频表达的整理稿](#四前半部分接近原视频表达的整理稿)  
5. [后半部分：AI Infra 深度分析](#五后半部分ai-infra-深度分析)  
6. [大模型推理框架应该学什么](#六大模型推理框架应该学什么)  
7. [CUDA 算子开发应该学什么](#七cuda-算子开发应该学什么)  
8. [模型结构为什么也要学](#八模型结构为什么也要学)  
9. [结合你的背景：你应该怎么转 AI Infra](#九结合你的背景你应该怎么转-ai-infra)  
10. [2026-2027 秋招准备路线](#十2026-2027-秋招准备路线)  
11. [可落地项目建议](#十一可落地项目建议)  
12. [面试问题清单](#十二面试问题清单)  
13. [最终建议](#十三最终建议)  
14. [参考资料](#十四参考资料)

---

# 一、这期视频一句话结论

这期视频的核心不是“泛泛介绍 AI Infra”，而是一个非常具体的校招规划咨询：

> **想找大模型推理 / AI Infra 岗位，不能只学 CUDA 算子。校招阶段至少要同时补三块：CUDA 算子基础、大模型推理框架 vLLM、大模型结构基础。**

视频里老师/前辈反复强调几个判断：

1. **只学 CUDA 会把路走窄**。  
   只会写算子，可以投算子开发岗，但推理框架开发、大模型部署、模型适配类岗位可能面试时会很尴尬。

2. **大模型推理方向论文不是最重要的点**。  
   这个方向更偏工程，校招更看重代码实战、工程理解、项目表达和知识点扩展能力。

3. **vLLM 是校招推理框架面试的高频入口**。  
   学 vLLM 不一定要求你完全改源码，但至少要部署过模型服务，理解核心模块和关键技术点。

4. **算子开发和推理框架开发待遇差不多，很多公司甚至在同一个大组里**。  
   因此更建议把路线走宽，而不是一开始就把自己限定成纯算子方向。

5. **大模型结构也要懂**。  
   面试可能问 Qwen、DeepSeek、MoE、MLA、Fused MoE 等结构。做推理优化虽然不是训练算法岗，但你必须知道模型由哪些模块组成、这些模块会落到哪些算子和通信流程上。

---

# 二、字幕技术名词纠错表

| 字幕识别 | 建议修正 | 说明 |
|---|---|---|
| 检理 | 简历 | 老师在看同学简历 |
| 非科班 | 非计算机科班 | 学生本科是电子工程/电子信息方向 |
| 库达 / 苦大 / 哭大 | CUDA | GPU 编程基础 |
| VIOM / VAM / VLAM / 比奥M | vLLM | 大模型推理服务框架 |
| Page Attaching / Page Attension | PagedAttention | vLLM 代表性 KV Cache 管理技术 |
| Continue Expression | Continuous Batching | 连续批处理，推理服务吞吐优化核心机制 |
| Channel Profile | Chunked Prefill | 可能指 vLLM 中的分块 Prefill 机制 |
| Kutagraph / Kutagno | CUDA Graph | 减少 kernel launch overhead 的图执行机制 |
| Gbtq / GbtqAWQ | GPTQ / AWQ | 常见大模型量化方式 |
| 投机捷码 / 投机结码 | Speculative Decoding | 投机解码 |
| LM Engine | LLMEngine | vLLM 旧架构/接口中常见概念 |
| Walker | Worker | 推理框架中的执行进程/执行单元 |
| Model Runner | ModelRunner | 模型执行模块 |
| 千万三 | Qwen3 | 通义千问 Qwen3，大模型结构面试中可能被问 |
| DPC / Dipsick | DeepSeek | DeepSeek 系列模型 |
| MLA | Multi-head Latent Attention | DeepSeek 系列代表性注意力机制优化 |
| Moe / M.O.E | MoE, Mixture of Experts | 混合专家模型 |
| Fuse Moe | Fused MoE | MoE 推理中常见融合算子/融合实现 |
| outdoor 通性 | All-to-All 通信 | MoE expert parallel 中常见通信模式 |
| FlyStyles 探讯 | FlashAttention | 注意力优化经典方向 |
| NCU | Nsight Compute | NVIDIA kernel 性能分析工具 |
| Warp 分化 | Warp divergence | CUDA 分支发散 |
| Band conflict | Bank conflict | shared memory bank conflict |
| GEMM / GAML | GEMM | 矩阵乘法核心算子 |
| 阿布主义用 | 不太确定 | 可能是某种岗位/公司/项目名，技术分析中不依赖该词 |
| 华为生 / 华为生成 | 华为昇腾 / Ascend / CANN | 国产 AI 芯片生态 |
| 缺点 | Triton / 自定义算子框架，可能不确定 | 字幕中语境是“先用某种方式快速实现第一版算子”，可能是 Triton，也可能是企业内部算子 DSL |

---

# 三、视频整体结构

这期视频是一次规划咨询/简历点评，主要有两位同学。

## 第一位同学

背景大致是：

- 非计算机科班，本科电子工程，研究生电子信息/控制方向；
- 实验室课题偏传统数据采集、传感器数据处理、传统机器学习；
- 想转 AI Infra / 大模型推理方向；
- 自己规划里主要想学 CUDA；
- 担心没有论文、没有实习、工程能力薄弱、和头部同学差距大。

老师/前辈的建议核心是：

- 这个阶段没有论文、没有实习、工程能力弱都正常；
- 实验室课题如果和工业 AI Infra 无关，不必投入过多；
- CUDA 可以学，但不能只学 CUDA；
- 三个月做 CUDA 入门和经典算子优化入门是可行的；
- 同时要补 vLLM 推理框架，把就业方向从纯算子开发扩展到推理框架开发和部署方向。

## 第二位同学

背景大致是：

- 已经开始做 CUDA 算子项目；
- 有一些实习/项目经历；
- 秋招开始较晚，面试反馈包括：不会推理框架、不了解 Qwen3 模型结构、算术强度算错；
- 项目中写过 Reduce、GEMM、Fused MoE 等算子练习；
- 想投芯片厂或算子开发方向。

老师/前辈的建议核心是：

- 仅有算子项目还不够，校招更希望候选人能力全面；
- 推理框架 vLLM 也要了解，即使投算子岗也可能被问；
- 大模型结构必须补，比如 Qwen3、DeepSeek、MoE、MLA；
- 算子项目表达要具体：看了哪些性能指标，发现了什么瓶颈，怎么改代码解决；
- 如果做国产芯片生态，华为昇腾/CANN 有价值，但泛化性不如 NVIDIA/CUDA/vLLM 技术栈；
- 模型适配通常要从论文、HuggingFace 代码、vLLM 实现、实际运行报错和缺失算子几个角度入手。

---

# 四、前半部分：接近原视频表达的整理稿

以下是对字幕前半部分的“语义复原稿”。为了便于阅读，我将原来的碎片化 ASR 字幕整理为对话式结构。

---

## 4.1 第一位同学：背景、课题和当前规划

老师先看第一位同学的简历。他问：

> 我看到你说自己是非科班，那你本科是什么专业？

同学回答大意是：本科是电子工程，现在研究生是电子信息类，具体偏控制方向。

老师继续问：

> 你现在在实验室里，导师给你的课题主要涉及哪方面？你之前具体做过什么工作？

同学解释说，项目本身不算特别复杂，主要是用多种传感器采集数据，然后对采集到的数据做综合处理，再通过课题组的一些算法提取更高级的特征，用于评估某些对象或状态。

老师追问：

> 这个工作是不是传统机器学习就能做？还是一定要更高级的大模型？

同学回答：比较简单，传统机器学习就可以。

老师接着问：

> 那你现在对大模型有什么认知？你学了 CS336，对 Transformer、MoE 这些东西有没有概念？你知不知道大模型里面各个模块在做什么？

同学说有了解，但还不算很深入。

老师看完规划后给出第一判断：

> 我看你的规划，如果能按照这个规划执行下来，最后应该会有一个不错的结果。主线是学习 CUDA，跟课程走，这没什么问题；主线再配合面试八股，也没什么问题。总体看，你目前规划制定得还可以，只要能实施下来，结果应该不会太差。

---

## 4.2 没有论文、没有实习，是不是大问题？

同学担心自己现在“零实习、零论文”。

老师的回答比较直接：

> 你现在这个阶段，做大模型推理方向没有论文是很正常的。这个方向本来就不是特别好发论文，除非你从算法方向去发。结合去年和今年秋招经验来看，论文在这个方向上不是最重要的点，所以不用太担心。

对于实习，老师也说：

> 你明年才开始秋招，现在没有实习也正常。工程能力和代码实战薄弱也正常，毕竟你现在还是学生，不可能要求一个在校生工程能力特别强。后面如果有实习，工程能力会慢慢提升。

这部分的核心意思是：

- AI Infra / 推理框架方向不完全等于学术算法岗；
- 没论文不是致命问题；
- 校招阶段更重要的是工程能力、项目表达、基础知识和学习能力；
- 学生工程能力弱是正常的，但要通过项目逐步补上。

---

## 4.3 实验室方向和工业方向不一致怎么办？

同学担心自己实验室方向偏传统，和工业 AI Infra 不匹配，同时还要投入时间完成毕业要求。

老师的建议是：

> 这个是客观条件，你已经在这个实验室里了，就没办法完全绕开。但如果这个课题对你以后找 AI Infra 方向帮助不大，你自己要把握好投入精力。该完成的完成，但不要把全部时间都压在一个和目标岗位关系不大的方向上。

这里其实非常适合你当前的情况。你现在的 IMS 课题本身有硬件、嵌入式、采集、信号处理价值，但如果你想转 AI Infra，就不能只围着实验室科研需求打转。你要把 IMS 项目尽量包装成：

- 嵌入式系统工程能力；
- 传感器数据采集与处理；
- 边缘设备部署；
- AI/模型部署结合；
- 工程联调和性能优化经验。

也就是说，实验室课题不一定直接等于 AI Infra 项目，但可以成为你的工程背景资产。

---

## 4.4 三个月能不能学会 CUDA 入门？

老师判断，同学从 12 月底到寒假结束大概有 3 个月时间。老师说：

> 你至少还有三个月时间，把 CUDA 入门的东西学一下，然后去优化一些算子。不是说让你自己凭空发明优化方法，而是把别人的优化套路熟悉一下。三个月做 CUDA 入门肯定是够的。当然这只是入门。

这句话非常关键：

> **校招阶段的 CUDA 算子学习，不是要求你成为顶级 kernel 专家，而是要求你掌握经典算子、经典优化套路和基本性能分析方法。**

三个月合理目标应该是：

- 跑通 CUDA 开发环境；
- 会写 vector add、reduce、transpose、matmul；
- 理解 thread/block/grid、shared memory、warp、coalesced access；
- 理解 shared memory bank conflict、warp divergence；
- 会用 Nsight Compute 看一些基础指标；
- 对 GEMM、FlashAttention、LayerNorm、Softmax 等算子有基本认知；
- 面试中能手写基础版本，并能说出优化方向。

---

## 4.5 算子开发 vs 推理框架开发，应该选哪个？

同学问：以后到底从事算子方向，还是大模型推理加速/部署方向？

老师的回答是：

> 从待遇来说，这两个方向差不多。有些大厂、芯片公司或者独角兽公司，写算子和写大模型推理框架的同事可能就在同一个大组里，所以待遇也类似。

接着老师给出建议：

> 我还是比较建议你学习一下 vLLM。你这三个月时间相对比较充裕，vLLM 也要跟进去学。否则你以后找工作只能找算子开发类岗位，大模型推理框架类岗位面试时你不了解，就会很尴尬。要把路走宽一点。

这就是整期视频最重要的结论之一：

> **AI Infra 求职不能只会 CUDA。CUDA 是底层能力，但 vLLM / 推理框架是你理解大模型服务化和工程系统的入口。**

---

## 4.6 vLLM 应该怎么学？

老师给出 vLLM 学习顺序：

### 第一步：先跑起来

> 学 vLLM 的第一步，是把这个框架运行起来。你要自己部署一个大模型推理服务，自己尝试调用。

也就是说，不要一上来读源码。第一步要先做：

- 安装 vLLM；
- 下载一个小模型；
- 启动 OpenAI-compatible API server；
- 用 curl / Python client 调用；
- 观察显存、吞吐、延迟；
- 改 batch size、max model len、并发数；
- 感受推理服务到底是什么。

### 第二步：分模块理解代码

老师说：

> 后续才是分模块看里面的实现，再结合八股或者相关文章分析代码。

他提到一些重点模块/知识点：

- LLMEngine；
- Worker；
- ModelRunner；
- PagedAttention；
- Continuous Batching；
- Quantization；
- GPTQ / AWQ；
- Speculative Decoding；
- Chunked Prefill；
- CUDA Graph；
- Scheduler；
- KV Cache 管理。

老师还强调：

> 校招阶段问 vLLM，不会问你太深入，一般集中在这些高频知识点。

所以校招对 vLLM 的掌握目标不是“完全吃透源码”，而是：

1. 能部署；
2. 能解释核心架构；
3. 能说出 PagedAttention 解决什么问题；
4. 能说出 Continuous Batching 为什么提升吞吐；
5. 能说出 KV Cache 为什么重要；
6. 能说出量化、投机解码、Chunked Prefill、CUDA Graph 各自作用；
7. 能结合代码位置或模块说出大概执行流程。

---

## 4.7 面试会不会问 vLLM？

同学问：

> 这些 vLLM 的东西，是投推理框架岗位会问，还是投算子岗位也会问？

老师回答：

> 一般来说投框架会问，但投算子岗位也保不齐会被问一下。如果校招面试中问到 vLLM，90% 会集中在刚才那些知识点里。

这说明现在 AI Infra 岗位边界并不完全清晰。算子、框架、部署、模型适配之间有重叠。一个做算子的同学，如果完全不懂推理框架，也会给面试官留下“知识面太窄”的印象。

---

## 4.8 学历、双非、非科班是不是硬伤？

字幕里有同学提到双非本科、非科班等问题。老师的态度是：

> 如果技术到位，你现在的学历已经够你找到一份不错的工作。关键是你能不能把知识点掌握好，CUDA 能不能掌握常见优化方式，面试时能不能把常见算子写出来。

这句话对你也很重要。你的学校和专业背景并不差，电子信息/嵌入式背景反而可以成为优势。关键是你要把技术路线做成闭环，而不是停留在“我想转 AI Infra”。

---

## 4.9 实习选择：大厂边缘业务要不要去？

同学问：如果有些大厂岗位平台好，但业务比较边缘，要不要去？

老师反问：

> 你说的边缘，是工作内容边缘，还是部门做的事情和目标方向无关？

如果部门做的事情和 AI Infra 完全无关，或者只是交付类、边缘业务，老师不太建议去。因为实习阶段最重要的是积累方向相关的产出。如果只是大厂 title，但工作和 AI Infra 没关系，对后续求职帮助有限。

这对你后续实习选择非常关键：

> **如果你的目标是 AI Infra，那么实习最好优先选择模型部署、推理服务、端侧 AI、AI 系统、算子优化、Linux/驱动/硬件加速相关岗位，而不是单纯看公司名气。**

---

# 五、后半部分：AI Infra 深度分析

## 5.1 这期视频讲的是 AI Infra 哪一层？

AI Infra 是一个很大的概念。可以粗略分成下面几层：

| 层级 | 典型方向 | 代表技术 |
|---|---|---|
| 应用层 | RAG、Agent、AI 应用开发 | LangChain、LlamaIndex、Spring AI |
| 部署服务层 | 模型服务、推理服务、API 服务 | vLLM、Triton Inference Server、TGI、Ray Serve |
| 推理框架层 | LLM 推理引擎、调度、KV Cache、Batching | vLLM、TensorRT-LLM、SGLang、llama.cpp |
| 算子层 | CUDA kernel、GEMM、Attention、LayerNorm | CUDA、Triton、CUTLASS、FlashAttention |
| 编译器层 | 图优化、算子融合、IR、硬件后端 | TVM、MLIR、XLA、TorchInductor |
| 芯片/硬件层 | GPU/NPU/ASIC 适配 | NVIDIA GPU、昇腾、寒武纪、壁仞等 |

这期视频主要讲的是中间两层：

1. **推理框架层：vLLM、大模型部署、KV Cache、PagedAttention、Continuous Batching；**
2. **算子层：CUDA、GEMM、FlashAttention、Reduce、Fused MoE、性能分析。**

它没有系统展开 AI 编译器，也没有深入训练框架。因此它更准确的标题是：

> **大模型推理框架与 CUDA 算子校招路线规划。**

---

## 5.2 为什么老师反复强调 vLLM？

vLLM 的价值在于：它把“大模型推理服务”这个问题工程化了。

大模型推理不是简单 `model.generate()`。真实服务要处理：

- 多用户并发请求；
- 输入长度不一样；
- 输出长度动态增长；
- KV Cache 动态分配；
- GPU 显存碎片；
- 吞吐和延迟平衡；
- Prefill 和 Decode 两个阶段；
- 长上下文；
- 多卡并行；
- OpenAI API 兼容；
- 量化、投机解码、CUDA Graph 等加速。

vLLM 官方文档把它描述为高吞吐、内存高效的 LLM 推理与服务引擎，核心特性包括 PagedAttention、Continuous Batching、Chunked Prefill、Prefix Caching、CUDA/HIP Graph、量化等。  
参考：vLLM 官方文档 https://docs.vllm.ai/en/latest/ ，vLLM 官网 https://vllm.ai/

所以校招学习 vLLM 的意义不是“背一个框架名字”，而是借它理解大模型推理系统的基本问题。

---

## 5.3 vLLM 的核心问题：KV Cache

大模型自回归生成时，每生成一个 token，都要基于前面已经生成的 token 继续计算。为了避免每一步重复计算历史 token 的 Key/Value，推理框架会缓存历史 token 的 K/V，这就是 KV Cache。

KV Cache 的问题是：

- 每个请求长度不同；
- 每个请求还在不断生成新 token；
- KV Cache 会动态增长；
- 多请求并发时显存容易碎片化；
- 显存利用率低会限制 batch size；
- batch size 上不去，吞吐就上不去。

PagedAttention 的思想是借鉴操作系统分页，把 KV Cache 按 block/page 管理，减少碎片和重复拷贝。PagedAttention 原论文《Efficient Memory Management for Large Language Model Serving with PagedAttention》指出，KV Cache 会随请求动态增长和收缩，低效管理会造成显著显存浪费，而 PagedAttention 通过类似虚拟内存分页的机制提升内存管理效率。  
参考：PagedAttention/vLLM 论文 https://arxiv.org/abs/2309.06180

所以面试中如果问：

> PagedAttention 解决什么问题？

你不能只说“提升推理速度”。更准确的回答是：

> 它主要解决大模型服务中 KV Cache 动态分配造成的显存碎片和显存浪费问题，使框架可以在相同显存下容纳更多并发请求，从而提升吞吐。

---

## 5.4 Continuous Batching 为什么重要？

普通 batching 是等一批请求凑齐后一起跑，但大模型生成过程有明显问题：

- 每个请求 prompt 长度不同；
- 每个请求生成长度不同；
- 有的请求很快结束，有的请求还在继续；
- 如果静态 batch，短请求结束后位置空着，GPU 利用率下降。

Continuous Batching 的思路是：

> 推理服务持续接收新请求，在每个 decode step 动态维护 batch，把已完成请求移除，把新请求加入，从而让 GPU 尽量保持忙碌。

所以它解决的是：

- 动态请求调度；
- GPU 利用率；
- 吞吐；
- 延迟和吞吐之间的平衡。

面试回答可以这样说：

> Continuous Batching 是在线推理服务中的动态批处理机制。它不是一次性固定 batch，而是在生成过程中持续插入新请求、移除完成请求，从而提高 GPU 利用率和服务吞吐。它通常要和 KV Cache 管理、调度器、请求状态管理一起理解。

---

## 5.5 Chunked Prefill 是什么？

LLM 推理有两个阶段：

1. **Prefill**：处理输入 prompt，计算初始 KV Cache；
2. **Decode**：逐 token 生成输出。

长 prompt 的 Prefill 可能非常耗时，还会阻塞其他 decode 请求。Chunked Prefill 的思路是把长 prompt 分块处理，让长输入不要一次性霸占 GPU 调度资源。

面试中可以这样回答：

> Chunked Prefill 把长 prompt 的 prefill 阶段拆成多个 chunk，使长上下文请求不会长期阻塞 decode 请求，有利于降低在线服务中的尾延迟，并改善调度公平性。

---

## 5.6 Speculative Decoding 是什么？

Speculative Decoding 通常用一个小模型作为 draft model，先快速生成多个候选 token，再用大模型一次性验证这些候选 token。它的目标是减少大模型逐 token 调用次数，提高生成速度。

面试中可以这样回答：

> 投机解码用小模型预测候选 token，然后用目标大模型并行验证。如果候选 token 被接受，就相当于一次大模型 forward 产生多个有效 token；如果不接受，则回退或修正。它适合在大小模型输出分布比较接近、验证开销低于逐 token 解码开销时加速生成。

TensorRT-LLM 官方文档中也把 speculative decoding 作为生成式 AI 推理优化的重要能力之一。  
参考：TensorRT-LLM 文档 https://nvidia.github.io/TensorRT-LLM/

---

## 5.7 CUDA Graph 是什么？

在推理服务中，频繁 launch kernel 会有 CPU 端调度开销。CUDA Graph 可以把一段固定形状、固定执行路径的 GPU 操作捕获成图，然后重复执行，减少 launch overhead。

但大模型推理有动态 batch、动态 sequence length，所以 CUDA Graph 的使用需要考虑 shape 稳定性和图复用策略。

面试中可以这样说：

> CUDA Graph 通过捕获并复用 GPU 执行图，减少 CPU launch overhead，适合重复执行、shape 较稳定的推理片段。在 LLM 推理框架里，它通常要和 batch size、sequence length、padding 或 bucket 策略结合，否则动态图形态会降低复用效果。

---

# 六、大模型推理框架应该学什么

## 6.1 推荐学习顺序

根据视频里的建议，vLLM 学习顺序应该是：

### 阶段 1：部署体验

目标：知道推理服务长什么样。

任务：

- 安装 vLLM；
- 部署 Qwen / Llama / DeepSeek Distill 等小模型；
- 启动 OpenAI-compatible API；
- 写 Python client 或 curl 调用；
- 观察 GPU 显存；
- 测输入长度、输出长度、并发数对延迟和吞吐的影响。

你需要记录：

- TTFT：Time To First Token；
- TPOT：Time Per Output Token；
- Throughput：tokens/s；
- QPS；
- GPU memory usage；
- 并发数；
- max_model_len；
- batch size 相关配置。

### 阶段 2：看文档和架构

重点看：

- vLLM Architecture Overview；
- Engine Core；
- Scheduler；
- KV Cache Manager；
- GPU Worker；
- Model Runner；
- Attention Backend；
- Quantization；
- Speculative Decoding；
- Prefix Caching；
- Chunked Prefill。

vLLM 新版架构文档中提到，engine core process 负责调度器、KV Cache 管理，并协调 GPU workers 执行模型。  
参考：vLLM Architecture Overview https://docs.vllm.ai/en/latest/design/arch_overview.html

### 阶段 3：按模块读源码

不要一上来从 main 函数一路读到底。建议按问题读源码：

| 问题 | 对应模块 |
|---|---|
| 请求从 API 进入后怎么变成推理任务？ | entrypoints / engine |
| 多个请求怎么调度？ | scheduler |
| KV Cache 怎么分配和回收？ | kv cache manager |
| 模型 forward 在哪里执行？ | model runner / worker |
| Attention kernel 怎么调用？ | attention backend |
| 量化模型怎么加载和执行？ | quantization |
| 投机解码怎么验证 draft token？ | speculative decoding |

### 阶段 4：做一个小修改

校招阶段如果能做到这一步，非常加分：

- 增加一个简单日志统计；
- 打印每个请求的 prefill/decode 时间；
- 加一个简单 benchmark 脚本；
- 对某个配置做实验对比；
- 对某个模块写源码阅读笔记；
- 尝试支持一个小模型或修复一个小 bug。

---

## 6.2 vLLM 校招高频知识点

| 知识点 | 必须会回答的问题 |
|---|---|
| PagedAttention | 为什么 KV Cache 会碎片化？PagedAttention 怎么解决？ |
| KV Cache | 为什么自回归生成需要 KV Cache？显存占用如何估算？ |
| Continuous Batching | 和普通 batching 有什么区别？为什么提升吞吐？ |
| Prefill / Decode | 两阶段计算特点有什么不同？ |
| Chunked Prefill | 为什么长 prompt 会影响服务延迟？ |
| Prefix Caching | 相同前缀为什么可以复用 KV Cache？ |
| Quantization | GPTQ、AWQ、FP8、INT4 大概解决什么问题？ |
| Speculative Decoding | draft model 和 target model 怎么配合？ |
| CUDA Graph | 为什么能减少 launch overhead？有什么限制？ |
| Scheduler | 如何平衡吞吐、延迟和公平性？ |
| ModelRunner | 模型 forward 的执行入口在哪里？ |
| Worker | 多进程/多卡执行单元如何协作？ |

---

# 七、CUDA 算子开发应该学什么

## 7.1 视频里的核心建议

老师对 CUDA 算子的要求很实际：

> 校招不会要求你在有限时间内写出极致优化的 kernel，但要求你能写出基础版本，并且在面试官提示优化点时，说出思路，最好能动手改。

这说明校招算子考察重点是：

1. 基础 kernel 会不会写；
2. CUDA 执行模型懂不懂；
3. 常见性能瓶颈能不能识别；
4. 优化套路是否熟悉；
5. 能不能用性能指标支撑自己的优化；
6. 能不能把项目讲清楚。

---

## 7.2 必会基础

你至少要掌握：

- thread / block / grid；
- warp；
- global memory / shared memory / register；
- memory coalescing；
- shared memory bank conflict；
- warp divergence；
- occupancy；
- kernel launch overhead；
- synchronization；
- atomic operation；
- CUDA stream；
- CUDA event；
- Nsight Compute / Nsight Systems 基础使用。

---

## 7.3 必练经典算子

建议练习顺序：

1. Vector Add  
2. Reduce  
3. Prefix Sum / Scan  
4. Matrix Transpose  
5. Softmax  
6. LayerNorm / RMSNorm  
7. GEMM / MatMul  
8. Batched GEMM  
9. Attention 简化版  
10. FlashAttention 原理版  
11. Fused MoE 简化版

其中 **GEMM** 是重中之重。CUTLASS 官方文档也强调其围绕不同层级的矩阵乘累加操作构建统一编程模型，包括 device-level、threadblock-level、warp-level、thread-level 和 instruction-level GEMM。  
参考：CUTLASS GEMM API https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html

---

## 7.4 面试表达方式

不要这样讲：

> 我写了一个 Reduce，然后做了一些优化。

应该这样讲：

> 我先写了一个 naive reduce，每个 block 处理一段输入，用 shared memory 做 block 内归约。然后我用 Nsight Compute 看了 memory throughput、warp stall、shared memory bank conflict 等指标。第一版存在访存不连续和同步开销较大的问题。后面我通过调整访问模式、减少分支、展开循环、优化 shared memory 布局来减少 warp divergence 和 bank conflict，最后对比了优化前后的耗时和吞吐。

重点是：

- 初始版本是什么；
- 性能瓶颈是什么；
- 你看了哪些指标；
- 你如何改代码；
- 优化后效果如何；
- 这个优化为什么有效。

---

## 7.5 Nsight Compute 应关注哪些指标？

对于初学者，可以先关注：

- Duration；
- GPU Throughput；
- Memory Throughput；
- SM Occupancy；
- Achieved Occupancy；
- Warp Stall Reasons；
- Global Load/Store Efficiency；
- Shared Memory Bank Conflicts；
- DRAM Throughput；
- L2 Cache Hit Rate；
- Tensor Core Utilization；
- Arithmetic Intensity。

如果你投算子岗，尤其要会解释：

> 这个 kernel 是 compute-bound 还是 memory-bound？

算术强度 Arithmetic Intensity 就和这个问题有关。它大致表示单位访存对应多少计算量。算术强度高的 kernel 更可能受计算能力限制，算术强度低的 kernel 更可能受内存带宽限制。

---

# 八、模型结构为什么也要学

## 8.1 视频里的面试反馈

第二位同学面试反馈中有一条是：

> 不会 Qwen3 的模型结构。

老师认为这确实是问题。因为做推理优化虽然不是训练算法岗，但你后续要支持模型、拆解算子、理解模型执行流程。如果你不知道模型结构，就很难知道某个算子在整个模型中的作用。

---

## 8.2 面试问“Qwen3 结构”应该怎么答？

不要只说：

> 它是一个大语言模型。

更好的答法是：

> Qwen3 整体属于 Transformer Decoder-only 架构的大语言模型系列。回答时可以先讲主干结构：token embedding、多层 decoder block、attention、MLP/MoE、norm、lm head。然后再讲它相对前代或其他模型的创新点，例如是否包含 dense 和 MoE 版本、thinking/non-thinking 模式、长上下文、多语言能力、推理模式等。最后结合推理优化，说这些结构会落到 attention、MLP/GEMM、MoE routing、KV Cache、量化和并行执行上。

Qwen 官方博客介绍 Qwen3 时提到，Qwen3 包含 dense 和 MoE 模型，并强调 thinking mode 与 non-thinking mode 的统一，以及训练和推理成本方面的改进。  
参考：Qwen3 官方博客 https://qwenlm.github.io/blog/qwen3/ ，Qwen3 Technical Report https://arxiv.org/abs/2505.09388

---

## 8.3 面试问“DeepSeek 结构”应该怎么答？

可以这样答：

> DeepSeek-V3 是 MoE 架构大模型。它的关键点包括 MLA、DeepSeekMoE、辅助损失-free 的负载均衡策略、多 token prediction 等。对于推理优化，重点不是背论文指标，而是理解 MLA 如何影响 KV Cache 和注意力计算，MoE 如何影响 expert routing、Fused MoE kernel、专家并行和 All-to-All 通信。

DeepSeek-V3 技术报告介绍其为 671B 总参数、每 token 激活 37B 参数的 MoE 模型，并采用 MLA 和 DeepSeekMoE 架构。  
参考：DeepSeek-V3 Technical Report https://arxiv.org/abs/2412.19437

---

## 8.4 MoE 应该怎么答？

视频里老师指出，同学回答 MoE 时不能只背概念。

差的回答：

> MoE 是混合专家模型，由门控网络和专家网络组成。

这个回答太浅。

更好的回答：

> MoE 的核心思想是增加总参数量，但每个 token 只激活部分专家，因此在扩大模型容量的同时控制每 token 的计算量。推理时，每个 token 先经过 router/gate 选择 top-k experts，然后把 token 分发给对应专家计算，最后把专家输出按权重合并。做分布式推理时，如果专家分布在不同 GPU 或节点上，就会涉及 All-to-All 通信。工程优化上常见问题包括 expert load balance、token dispatch/combine、Fused MoE kernel、通信与计算重叠等。

这样回答就把模型结构和推理系统联系起来了。

---

## 8.5 MLA 应该怎么答？

可以这样答：

> MLA 是 DeepSeek 系列里用于降低 attention KV Cache 显存占用、提高推理效率的注意力机制设计。传统 Multi-Head Attention 需要为每层、每个 head 存储 K/V，长上下文下 KV Cache 显存压力很大。MLA 通过 latent 表示压缩 key/value 相关状态，使推理时 KV Cache 更经济。面试中要能把它和 KV Cache、长上下文、显存占用、attention 计算联系起来。

---

# 九、结合你的背景：你应该怎么转 AI Infra

结合你的情况：

- 你是电子信息/新一代电子信息技术方向研一；
- 当前研究是 IMS 离子迁移谱仪相关，包含嵌入式控制、采集、显示、远程 Web、仪器联调；
- 你有 ESP32-S3、LVGL、TFT、传感器/采样、系统集成经验；
- 你正在补 Linux/OS；
- 你预计 2027 暑期秋招；
- 你想从嵌入式转 AI Infra，尤其是推理侧。

我的判断是：

> **你不应该直接照搬纯 CS 同学的“CUDA + vLLM + 算子”路线，也不应该只做普通 RAG/Agent 应用。你最适合走“嵌入式/Linux/系统工程 + 模型部署/推理框架 + 端侧或边缘 AI”的交叉路线。**

你相比普通 AI 应用开发者的优势是：

- 你接触过真实硬件；
- 你做过嵌入式系统；
- 你理解采集、控制、通信、显示、联调；
- 你有实际科研设备场景；
- 你可以把 AI 部署到边缘设备/仪器系统里。

你相比纯 AI Infra 科班强手的短板是：

- C++ 需要加强；
- Linux 系统编程需要加强；
- CUDA 需要从零补；
- 深度学习和大模型结构需要补；
- 工程项目需要更贴近工业岗位；
- 需要 GitHub 项目和面试表达。

所以你的目标不应该是“马上变成顶级 CUDA 算子工程师”，而应该是：

> **到 2027 秋招前，成为一个有嵌入式系统背景、懂 Linux/C++、能部署和分析大模型推理服务、了解 CUDA 算子基础、能做边缘 AI/模型部署项目的候选人。**

---

# 十、2026-2027 秋招准备路线

## 阶段 1：2026 年 6-8 月  
### 目标：补齐基础，不急着啃 vLLM 源码

重点：

1. C++ 基础  
   - 指针、引用、类、模板、STL、RAII；
   - 多线程、锁、条件变量；
   - CMake；
   - 基础项目组织。

2. Linux 基础  
   - 进程、线程、文件 IO；
   - socket；
   - 内存管理；
   - shell；
   - gdb；
   - perf 基础。

3. Python + PyTorch  
   - Tensor；
   - autograd；
   - module；
   - dataloader；
   - 简单训练；
   - 模型保存和加载。

4. 深度学习基础  
   - MLP、CNN、Transformer；
   - Attention；
   - Tokenizer；
   - Embedding；
   - Decoder-only LM；
   - KV Cache。

阶段产出：

> 跑通一个小模型：PyTorch 推理 → ONNX 导出 → ONNX Runtime 推理 → 简单性能对比。

---

## 阶段 2：2026 年 9-12 月  
### 目标：进入模型部署和推理服务

重点：

1. ONNX Runtime；
2. TensorRT 基础；
3. Triton Inference Server；
4. vLLM 部署；
5. llama.cpp 部署；
6. 模型量化基础；
7. 推理 benchmark；
8. TTFT、TPOT、tokens/s、QPS、显存统计。

阶段产出：

> 做一个“LLM 推理服务性能分析项目”：同一个模型用不同推理后端部署，比较延迟、吞吐、显存、并发能力。

项目可以包含：

- vLLM 部署 Qwen 小模型；
- llama.cpp 部署量化模型；
- ONNX Runtime 部署 BERT/小模型；
- TensorRT 部署视觉模型或小型 Transformer；
- 写 benchmark 脚本；
- 画出性能表；
- 写技术报告。

---

## 阶段 3：2027 年 1-3 月  
### 目标：深入 vLLM + CUDA 算子入门

重点：

1. vLLM 核心模块  
   - Scheduler；
   - KV Cache；
   - PagedAttention；
   - ModelRunner；
   - Worker；
   - Continuous Batching；
   - Chunked Prefill；
   - Quantization。

2. CUDA 算子  
   - vector add；
   - reduce；
   - transpose；
   - matmul；
   - softmax；
   - layernorm；
   - attention 简化版。

3. 性能分析  
   - Nsight Compute；
   - Nsight Systems；
   - roofline 思想；
   - memory-bound / compute-bound；
   - arithmetic intensity。

阶段产出：

> vLLM 源码阅读笔记 + CUDA 算子优化笔记 + 一个可展示 benchmark 项目。

---

## 阶段 4：2027 年 4-6 月  
### 目标：简历包装与面试准备

重点：

1. 重构简历项目；
2. 准备 C++ 八股；
3. 准备 Linux/OS；
4. 准备计算机网络；
5. 准备 CUDA 面试题；
6. 准备 vLLM 高频题；
7. 准备大模型结构题；
8. 准备 IMS 项目和 AI Infra 的联系讲法。

你的简历可以形成三个项目组合：

1. **IMS 嵌入式控制系统**  
   体现硬件系统、嵌入式、采集、联调能力。

2. **LLM 推理服务性能分析平台**  
   体现模型部署、vLLM、推理服务、benchmark 能力。

3. **CUDA 算子优化练习 / IMS 谱图模型部署**  
   体现底层优化或边缘 AI 能力。

---

# 十一、可落地项目建议

## 项目 1：LLM 推理服务性能分析平台

### 项目定位

AI Infra / 推理部署项目。

### 项目内容

- 使用 vLLM 部署 Qwen / Llama / DeepSeek Distill 小模型；
- 提供 OpenAI-compatible API；
- 编写压测脚本；
- 统计 TTFT、TPOT、tokens/s、QPS、显存；
- 分析并发数、输入长度、输出长度、max_model_len 对性能的影响；
- 对比不同量化模型；
- 总结 PagedAttention、Continuous Batching、Chunked Prefill 的影响。

### 简历表述示例

> 基于 vLLM 搭建大模型推理服务性能分析平台，支持 OpenAI-compatible API 调用与多并发压测。实现 TTFT、TPOT、吞吐、显存占用等指标统计，分析输入长度、输出长度、并发数、量化方式对推理性能的影响，并结合 vLLM 的 PagedAttention、Continuous Batching、KV Cache 管理机制进行性能解释。

---

## 项目 2：CUDA 经典算子优化项目

### 项目定位

算子开发 / GPU 编程项目。

### 项目内容

- 实现 Reduce、Softmax、LayerNorm、GEMM；
- 对比 naive 版本和优化版本；
- 使用 Nsight Compute 分析；
- 记录 bank conflict、warp divergence、memory coalescing；
- 尝试 shared memory、tiling、loop unrolling；
- 对 GEMM 学习 CUTLASS 分层思想。

### 简历表述示例

> 实现 CUDA Reduce / Softmax / GEMM 等经典算子，并基于 Nsight Compute 分析 kernel 性能瓶颈。针对全局内存访问不连续、shared memory bank conflict、warp divergence 等问题进行优化，记录优化前后耗时、memory throughput、occupancy 等指标变化，形成算子优化实验报告。

---

## 项目 3：IMS 谱图智能识别与边缘部署

### 项目定位

你的差异化项目：嵌入式 + AI + 部署。

### 项目内容

- 对 IMS 谱图做峰检测；
- 提取漂移时间、峰值、峰宽等特征；
- 用传统机器学习或轻量神经网络做物质分类；
- 训练 Python 模型；
- 导出 ONNX；
- 在边缘设备或本地服务上推理；
- 和你的 IMS 控制系统结合；
- 后续可以扩展到“仪器数据采集 → 模型识别 → 结果显示 → 远程 Web 展示”。

### 简历表述示例

> 面向离子迁移谱仪的谱图识别任务，构建从谱图采集、峰检测、特征提取到轻量模型推理的端到端流程。基于漂移时间、峰值强度和峰宽等特征训练分类模型，并尝试导出 ONNX 进行边缘侧部署，实现仪器采集数据的实时分析与本地显示。

这个项目是你区别于普通 AI Infra 候选人的关键。

---

# 十二、面试问题清单

## 12.1 vLLM / 推理框架

1. vLLM 是什么？解决什么问题？  
2. PagedAttention 为什么能减少 KV Cache 显存浪费？  
3. KV Cache 的显存占用和哪些因素有关？  
4. Prefill 和 Decode 的计算特征有什么不同？  
5. Continuous Batching 和普通 batching 区别是什么？  
6. Chunked Prefill 解决什么问题？  
7. Prefix Caching 适合什么场景？  
8. Speculative Decoding 的流程是什么？  
9. CUDA Graph 在推理服务中有什么作用？  
10. vLLM 中 scheduler 的作用是什么？  
11. ModelRunner / Worker 大概负责什么？  
12. 如果一个请求特别长，会如何影响其他请求？  
13. 如何评价推理服务性能？  
14. TTFT 和 TPOT 分别代表什么？  
15. 吞吐和延迟之间如何权衡？

---

## 12.2 CUDA / 算子

1. CUDA 的 thread/block/grid 是什么？  
2. warp 是什么？  
3. 什么是 warp divergence？  
4. 什么是 shared memory bank conflict？  
5. 如何优化 Reduce？  
6. 如何优化 Matrix Transpose？  
7. GEMM 为什么重要？  
8. 什么是 memory coalescing？  
9. 如何判断 kernel 是 memory-bound 还是 compute-bound？  
10. 算术强度是什么？  
11. Nsight Compute 常看哪些指标？  
12. FlashAttention 为什么省显存？  
13. Softmax 怎么并行实现？  
14. LayerNorm/RMSNorm 的计算流程是什么？  
15. Fused kernel 有什么好处？

---

## 12.3 大模型结构

1. Transformer Decoder-only 结构是什么？  
2. Attention 的 Q/K/V 分别是什么？  
3. MHA、MQA、GQA 有什么区别？  
4. MoE 是什么？为什么能扩大参数但控制计算量？  
5. MoE 推理中为什么会涉及 All-to-All 通信？  
6. Fused MoE 大概融合了哪些步骤？  
7. DeepSeek 的 MLA 解决什么问题？  
8. Qwen3 的模型结构应该怎么介绍？  
9. DeepSeek-V3 的关键结构有哪些？  
10. KV Cache 和模型结构有什么关系？  
11. 为什么长上下文会带来显存压力？  
12. 量化对模型推理有什么影响？  
13. GPTQ 和 AWQ 大概区别是什么？  
14. INT4/INT8/FP8 各自适合什么场景？  
15. 模型适配时如何从 HuggingFace 代码定位算子？

---

## 12.4 项目表达

1. 你的项目解决什么问题？  
2. 为什么这个项目属于 AI Infra / 模型部署？  
3. 你做了哪些工程实现？  
4. 你看了哪些性能指标？  
5. 遇到过什么瓶颈？  
6. 如何定位瓶颈？  
7. 如何优化？  
8. 优化前后数据是多少？  
9. 如果重做，你会怎么改？  
10. 这个项目和目标岗位有什么关系？

---

# 十三、最终建议

对你来说，这期视频最值得吸收的不是某一个知识点，而是这套判断：

> **AI Infra 校招不是只看你会不会一个工具，而是看你是否能把“模型结构 → 推理框架 → 算子执行 → 部署服务 → 性能指标”串起来。**

你现在最应该避免三个误区：

## 误区 1：只学嵌入式，不碰 AI 系统

这样你会回到普通嵌入式/Linux 岗位，和 AI Infra 的距离仍然比较远。

## 误区 2：只学 RAG/Agent 应用

这样你会变成普通 AI 应用开发候选人，和你的硬件/嵌入式优势脱节。

## 误区 3：只学 CUDA 算子

这样路线太窄，而且对你来说起步成本高，短期内很难和科班强手硬拼。

你更合理的路线是：

> **Linux/C++ 打底 → PyTorch/Transformer 入门 → ONNX/TensorRT/vLLM 部署 → CUDA 算子入门 → IMS 边缘 AI 项目结合 → 2027 秋招投模型部署/推理框架/端侧 AI/AI Infra 相关岗位。**

最后给你一个非常具体的学习优先级：

1. **最高优先级：C++ + Linux + Python/PyTorch 基础**  
2. **第二优先级：vLLM 部署与推理服务 benchmark**  
3. **第三优先级：Transformer / Qwen / DeepSeek / MoE / MLA 模型结构**  
4. **第四优先级：CUDA 经典算子入门与 Nsight 分析**  
5. **第五优先级：把 IMS 项目改造成“边缘 AI + 模型部署”项目**  
6. **第六优先级：RAG/Agent 只作为应用包装，不作为主线**

如果你能在 2027 秋招前形成下面这个组合，你的竞争力会明显提升：

> **一个真实嵌入式系统项目 + 一个推理服务性能分析项目 + 一个 CUDA/模型部署实验项目 + 一套清楚的 AI Infra 知识体系。**

---

# 十四、参考资料

以下资料用于校正视频中涉及的 AI Infra 技术概念，建议后续按顺序阅读：

1. vLLM 官方文档  
   https://docs.vllm.ai/en/latest/

2. vLLM 官网  
   https://vllm.ai/

3. vLLM Architecture Overview  
   https://docs.vllm.ai/en/latest/design/arch_overview.html

4. PagedAttention / vLLM 论文  
   Efficient Memory Management for Large Language Model Serving with PagedAttention  
   https://arxiv.org/abs/2309.06180

5. TensorRT-LLM 官方文档  
   https://nvidia.github.io/TensorRT-LLM/

6. CUTLASS 官方文档  
   https://docs.nvidia.com/cutlass/

7. CUTLASS GEMM API  
   https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html

8. Qwen3 官方博客  
   https://qwenlm.github.io/blog/qwen3/

9. Qwen3 Technical Report  
   https://arxiv.org/abs/2505.09388

10. DeepSeek-V3 Technical Report  
    https://arxiv.org/abs/2412.19437

---

# 附录：原始字幕基本信息

- 原始字幕字符数：12331
- 原始字幕特点：对话式咨询，技术词识别错误较多，未按段落断句。
- 本文处理方式：保留视频核心表达和咨询逻辑，但对错别字、技术词、语义断裂进行了整理和复原。



%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-AI infra学习路线|模块-AI infra学习路线]]

%% 项目关联导航：结束 %%
