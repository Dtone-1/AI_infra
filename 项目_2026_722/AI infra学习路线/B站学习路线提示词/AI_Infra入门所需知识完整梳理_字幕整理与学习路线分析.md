# AI Infra 入门所需知识完整梳理  
## 长视频字幕整理稿 + 深度学习分析 + 结合个人背景的秋招路线

> 生成时间：2026-06-03  
> 输入文件：`llm氪普课-ai infra入门所需知识完整梳理_哔哩哔哩_bilibili.txt`  
> 处理方式：自动字幕纠错、技术名词复原、原视频逻辑重组、AI Infra 知识体系扩展、结合个人背景制定学习路线。  
> 注意：原字幕来自自动识别，存在大量错别字、断句错误和技术名词误识别。本文不是逐字转写，而是“尽量接近原视频表达的整理稿 + 系统化学习分析”。

---

## 目录

1. [这期视频到底讲了什么](#一这期视频到底讲了什么)  
2. [字幕技术名词纠错表](#二字幕技术名词纠错表)  
3. [前半部分：接近原视频表达的整理稿](#三前半部分接近原视频表达的整理稿)  
4. [视频中的 AI Infra 三大知识板块](#四视频中的-ai-infra-三大知识板块)  
5. [第一部分：算法基础应该学什么](#五第一部分算法基础应该学什么)  
6. [第二部分：框架层应该学什么](#六第二部分框架层应该学什么)  
7. [第三部分：底层原理应该学什么](#七第三部分底层原理应该学什么)  
8. [推理框架核心知识点详解](#八推理框架核心知识点详解)  
9. [训练框架与强化学习框架应该怎么理解](#九训练框架与强化学习框架应该怎么理解)  
10. [硬件、通信和分布式系统知识](#十硬件通信和分布式系统知识)  
11. [这期视频和前几期视频的关系](#十一这期视频和前几期视频的关系)  
12. [结合你的背景：你应该怎么学 AI Infra](#十二结合你的背景你应该怎么学-ai-infra)  
13. [2026-2027 秋招学习路线](#十三2026-2027-秋招学习路线)  
14. [可落地项目建议](#十四可落地项目建议)  
15. [面试问题清单](#十五面试问题清单)  
16. [最终结论](#十六最终结论)  
17. [参考资料](#十七参考资料)

---

# 一、这期视频到底讲了什么

这期视频是一个 **AI Infra 入门知识体系梳理课**。它不是单纯讲某一个工具，也不是只讲 CUDA、vLLM 或训练框架，而是试图回答：

> **如果一个人想入门 AI Infra，需要系统掌握哪些知识点？这些知识点之间的关系是什么？哪些词是面试和学习中经常出现的关键名词？**

视频中的主线可以概括为三大类：

1. **算法层**  
   包括大模型基础结构、Transformer、Attention、MoE、模型评测、量化、性能优化等。

2. **框架层**  
   包括推理框架、训练框架、强化学习框架、KV Cache 管理、Serving、Batching、Prefill/Decode、PD 分离、推理引擎等。

3. **底层原理层**  
   包括 CUDA 编程、算子实现、FlashAttention、Online Softmax、通信原理、GPU 硬件、显存、带宽、NCCL、RDMA、InfiniBand、并行策略等。

这期视频最重要的价值是：

> **它把 AI Infra 从“到处都是零散名词”的状态，整理成了一个由算法、框架、底层系统组成的知识树。**

这对你很重要，因为你现在正在从嵌入式/Linux方向考虑转 AI Infra。你需要的不是只知道几个热门词，而是知道这些知识点如何组织成路线。

---

# 二、字幕技术名词纠错表

| 字幕识别错误 | 建议修正 | 说明 |
|---|---|---|
| ANEFRA / ANEFRA知识书里 | AI Infra / AI Infrastructure | 视频主题 |
| 宝丝 / 保存 | 粉丝 / 同学 | 口语称呼 |
| 知识书里 / 真理书里 | 知识梳理 | 视频标题语义 |
| 故事可以统成为算法 | 可以统称为算法 | 第一大类 |
| 带模型 / 大远模型 | 大模型 | LLM |
| 侵略化 | 量化 | Quantization |
| 优换 | 优化 | Optimization |
| 幻存 / KVCASH | KV Cache | Key-Value Cache |
| Surread / Serverread | Serving / Server | 推理服务端 |
| 浸球端 | 请求端 / Client | Client 侧 |
| 病情策略 | 并行策略 / batching 策略 | 视上下文而定 |
| KVCache架边成 | KV Cache 加速 / CUDA 编程 | 语义上是底层实现 |
| CHA和Passen | C++ 和 Python | 基础语言 |
| 算字 | 算子 | Operator / Kernel |
| 扩大编程 | CUDA 编程 | GPU 编程 |
| 穿普方面的结构 / 川普结构 | Transformer 结构 | 大模型主干 |
| 图疼 | 图腾 | 这里指“核心主线/基础图谱” |
| 添线 | Attention | 注意力机制 |
| Learnon | LayerNorm | 层归一化 |
| Pray / Post | Pre / Post | Pre-LN / Post-LN |
| 线存 | 显存 | GPU Memory |
| 随纸添线 | Sparse Attention | 稀疏注意力 |
| Lightening添线 | Lightning Attention | 一类高效注意力 |
| MQA这QA MLA | MQA / GQA / MLA | Attention 变体 |
| 阿姆异 | MoE | Mixture of Experts |
| Flash的添线 | FlashAttention | 高性能 Attention |
| Alonger's of the Max | Online Softmax | FlashAttention 相关基础技巧 |
| Decal layer | Decoder layer | 解码器层 |
| Gating网络 | Gate / Router | MoE 路由网络 |
| VIO / I / M / SG | vLLM / TGI / SGLang 等 | 推理框架，字幕不完全可靠 |
| Mini的 | Mini vLLM / Tiny framework | 简化版推理框架 |
| 腰隔板 / 腰缩板 | 阉割版 / 压缩版 | 简化实现，便于学习 |
| Lama Ferry | LLaMA-Factory | 常见微调/训练框架 |
| V2L / T2L | veRL / TRL | 强化学习/后训练框架，字幕不可靠但语义接近 |
| mega串 | Megatron | 大模型训练框架 |
| 串丑门库 | torch 基础库 / 从 PyTorch 写 | 训练框架底层 |
| Page的 attention | PagedAttention | vLLM KV Cache 管理 |
| Redix attention | RadixAttention | SGLang 前缀复用机制 |
| PrefixCansion | Prefix Caching | 前缀缓存 |
| Continuous batching | Continuous Batching | 连续批处理 |
| Trunk the perfect | Chunked Prefill | 分块预填充 |
| Prefeal / Prefeal阶段 | Prefill | 预填充阶段 |
| Decal阶段 | Decode | 解码阶段 |
| PD分离 | Prefill-Decode Disaggregation | 预填充/解码分离 |
| 投机节码 | Speculative Decoding | 投机解码 |
| EPLB | Expert Parallel Load Balancing | 专家并行负载均衡 |
| Experse elastic Expert parallel low the balancing | Expert Parallel Load Balancing | 字幕误识别 |
| 专家崇牌 | 专家重排 | Expert rebalancing / expert placement |
| MTP / MultiTokenPornicSync | Multi-Token Prediction | 多 token 预测 |
| TPTPTP | TP / PP / DP 等 | 张量并行/流水线并行/数据并行 |
| Infinite Band | InfiniBand | 高速网络 |
| RDMA | RDMA | 远程直接内存访问 |
| NCCL | NCCL | NVIDIA 集合通信库 |
| TOP结构 | Topology / 拓扑结构 | GPU/网络拓扑 |
| AKHK | A100 / H100 等卡型 | 字幕不确定，语义是不同 GPU 型号 |

---

# 三、前半部分：接近原视频表达的整理稿

下面是对视频前半部分的“接近原意整理稿”。为了阅读顺畅，我会把原字幕中错误断句和错词修正，但尽量保留讲课的表达顺序。

---

## 3.1 开场：为什么要做 AI Infra 知识梳理

视频开头老师说：

> 大家好，今天给大家讲一下 AI Infra 的知识梳理。这个课主要是为了比较系统地说明：AI Infra 主要有哪些知识点？这些知识点对应哪些关键名词？之前出过一些 AI Infra 的初始课程，大家反馈希望有更系统的指导，想知道这门课应该怎么学，知识点应该怎么掌握。所以今天就来整体梳理一下。

也就是说，这不是一节深入某个算法或框架的课，而是一节“总览课”。它的目标是让初学者先知道地图长什么样，避免刚开始就陷入零散名词。

---

## 3.2 AI Infra 有三大类知识

老师把 AI Infra 的知识分为三大类：

1. **算法**
2. **框架**
3. **底层原理**

他说：

> 作为 AI Infra 来说，可以认为有三大类知识。第一部分是算法，第二部分是框架，第三部分是底层原理。整体上可以认为是这三大类。

其中算法部分包括：

- 大模型基础知识；
- 相关评测指标；
- 模型量化；
- 性能优化。

框架部分包括：

- 常见推理框架；
- 训练框架；
- 强化学习训练框架；
- KV Cache；
- Serving；
- 请求端和服务端；
- batching 策略；
- 推理架构。

底层原理部分包括：

- CUDA 编程；
- 高性能算子实现；
- 通信原理；
- 推理硬件架构；
- 显存、带宽、GPU 结构；
- 分布式并行。

---

## 3.3 算法层：大模型基础是所有内容的主干

老师强调，大模型知识点很多，很容易学乱：

> 因为大模型知识点很多，大家很容易学得很乱。你会感觉自己到处都是不会的东西，而且这些东西彼此之间好像没有关系。这样越学越多，最后人肯定会学崩。

所以他建议要抓住一个核心主干：

> Transformer 结构可以认为是大模型的图腾。无论有多少输入、多少输出，以及中间有多少复杂关系，核心都要围绕 Transformer 结构去理解。

也就是说，AI Infra 虽然最终是系统工程，但不能完全绕开模型结构。你要知道模型由哪些模块组成，这些模块对应哪些计算、显存和通信问题。

---

## 3.4 Transformer 结构应该怎么学

老师说，Transformer 结构里至少要理解：

- 它有哪些组成部分；
- 各组成部分之间有什么公式；
- Attention 是怎么计算的；
- LayerNorm 是 Pre-LN 还是 Post-LN；
- Pre-LN 和 Post-LN 的区别；
- 不同归一化方式对训练稳定性有什么帮助；
- 这些内容在面试中会不会考；
- 如果是偏算法或训练岗，可能会考得更深。

其中 Attention 是核心。他提到，不仅要学 Attention 的计算公式，还要学复杂度：

- 时间复杂度；
- 空间复杂度；
- 参数量；
- 显存计算量。

他特别提醒：

> 参数量计算、显存计算、空间复杂度不是互相无关的概念。大家不要把这些概念学散了。

这句话很重要。AI Infra 里很多问题本质上都回到：

- 模型有多少参数；
- 每一步计算量是多少；
- KV Cache 占多少显存；
- batch size 能开多大；
- 长上下文为什么贵；
- 为什么要量化；
- 为什么要优化 Attention。

---

## 3.5 Attention 家族

老师接着讲 Attention 的变体：

> Attention 不可避免会有一些分支，比如 Sparse Attention、Lightning Attention，还有 MQA、GQA、MLA 等。

这些都是面试和学习中常见的关键词。

可以这样理解：

- **MHA**：Multi-Head Attention，标准多头注意力；
- **MQA**：Multi-Query Attention，多个 query head 共享较少的 key/value head；
- **GQA**：Grouped-Query Attention，在 MHA 和 MQA 之间折中；
- **MLA**：Multi-head Latent Attention，DeepSeek 系列中用于降低 KV Cache 压力的注意力设计；
- **Sparse Attention**：通过稀疏模式降低长序列注意力计算；
- **Lightning Attention**：一类高效 attention 方案；
- **FlashAttention**：通过 IO-aware 思想优化 attention 计算；
- **Online Softmax**：FlashAttention 中非常关键的数值稳定和分块计算技巧。

老师还提到，面试里可能会让你手写一些模块：

- MHA；
- Decoder Layer；
- MoE；
- Attention 变体。

他说 MHA 听起来复杂，但本质就是矩阵乘法、reshape、softmax 和投影。MoE 听起来复杂，但本质上是 gate/router 网络加专家选择。

---

## 3.6 框架层：不要只把推理当成 model.generate

进入框架部分，老师说：

> 如果你之前搞训练，可能会觉得推理只是用大模型跑一下，这是很天然的事情，不会意识到它要分服务端和请求端。但如果把它当成一个服务来理解，就好理解很多。一个服务肯定有提供商，也有请求方。

这句话是理解推理框架的关键。

在真实工业场景中，大模型推理不是简单地调用 `model.generate()`。它是一个在线服务系统：

- 有客户端请求；
- 有服务端调度；
- 有多用户并发；
- 有请求排队；
- 有动态 batch；
- 有 KV Cache 管理；
- 有 Prefill 和 Decode；
- 有显存和吞吐限制；
- 有延迟指标；
- 有量化和硬件加速；
- 有多卡分布式部署。

所以，AI Infra 中的“推理框架”本质上是一个 **大模型在线服务系统**。

---

## 3.7 常见推理框架

老师提到工业上常见的推理框架：

- vLLM；
- TGI；
- SGLang；
- 以及一些 Mini 版/简化版框架。

他说如果觉得完整框架不好理解，可以先看 Mini 版或压缩版。因为它们代码更少，原理更清晰，更适合入门。

这点非常适合初学者。直接读 vLLM/SGLang 源码可能会被工程细节淹没，而 Mini 版实现能帮你先理解：

- 请求如何进入；
- scheduler 如何组织；
- KV Cache 如何分配；
- 模型 forward 如何执行；
- batch 如何动态更新；
- 输出 token 如何返回。

---

## 3.8 训练框架和强化学习框架

老师接着讲训练框架。他说：

> 如果是入门，可以看 LLaMA-Factory。但它的缺点是封装太死，如果想做一点改动，会受很多限制。小白可以用它入门感受一下，进阶就不太推荐了。

然后他提到强化学习和大模型后训练框架，比如 veRL、TRL 之类。对于正常大规模训练，可能会涉及 Megatron、DeepSpeed，甚至有人会自己从更底层写训练框架。

他还提醒：

> 你不仅要知道自己用了什么框架，还要知道这个框架的优势是什么。面试时如果说自己用了 LLaMA-Factory，但不知道它支持什么、不支持什么，就容易被问住。

这句话对应面试中的一个常见问题：

> 你为什么选这个框架？它相比其他框架有什么优势和限制？

如果回答不出来，说明只是“用了工具”，没有真正理解框架。

---

## 3.9 KV Cache 相关考点

老师在框架部分列出 KV Cache 相关的常见考点：

- PagedAttention；
- RadixAttention；
- Prefix Caching。

这些都是推理框架里非常核心的技术。

简单理解：

- **KV Cache**：保存历史 token 的 key/value，避免每一步重复计算；
- **PagedAttention**：像操作系统分页一样管理 KV Cache，减少显存碎片；
- **RadixAttention**：SGLang 中围绕前缀复用的机制；
- **Prefix Caching**：相同 prompt 前缀的 KV Cache 可以复用，减少重复计算。

---

## 3.10 Serving：Continuous Batching

老师用“等大巴”和“等地铁”的比喻解释 Continuous Batching。

普通 Static Batching 就像大巴：

> 要凑够一车人，或者要等这一批所有人都处理完，才能处理下一批。如果某个请求很长，其他短请求也要等它。

Continuous Batching 就像地铁：

> 已经完成的请求立刻腾出位置，新来的请求可以补位，不需要等整个 batch 一起结束。

它的核心是：

- 生成阶段动态选择未完成请求组成 batch；
- 请求完成后立即移除；
- 新请求可以加入；
- 减少等待；
- 提高 GPU 利用率；
- 提高吞吐；
- 降低排队延迟。

---

## 3.11 Chunked Prefill 和 PD 分离

老师接着讲 Chunked Prefill。他先说明 LLM 推理有两个阶段：

1. **Prefill**：处理输入 prompt，构建 KV Cache；
2. **Decode**：逐 token 生成输出。

长输入的 Prefill 可能非常慢。如果一个长 prompt 一直占着 GPU，其他请求会被阻塞。所以 Chunked Prefill 的做法是：

> 把长输入的 Prefill 拆成多个小块，一段一段处理。这样 GPU 有空时可以穿插处理其他 Decode 请求，而不是必须等整个长 prompt 处理完。

它的作用是：

- 降低排队时间；
- 提高 GPU 利用率；
- 改善显存效率；
- 让长 prompt 和短 decode 请求更好地混合调度。

老师还提到 **PD 分离**，也就是 Prefill-Decode Disaggregation。它的意思是把 Prefill 和 Decode 两个阶段在系统架构上分离，让不同资源处理不同阶段，从而优化吞吐和延迟。

---

## 3.12 Decode 阶段：Speculative Decoding、EPLB、MTP

老师把一部分内容称为“生成/解码阶段”的知识。

### Speculative Decoding

投机解码名字听起来很高级，但基本思想是：

> 用小模型先猜几个 token，再让大模型验证。如果猜对，就能一次接受多个 token，从而加速生成。

### EPLB

EPLB 大致是 **Expert Parallel Load Balancing**。它和 MoE 专家并行相关。

老师解释说，不必死记缩写，核心是知道场景：

> 在 MoE 专家并行场景下，需要做负载均衡和专家重排。工程上有静态和动态两类，可以理解为提前做一次布局，或者在线持续调整布局。

### MTP

MTP 是 **Multi-Token Prediction**，多 token 预测。

普通自回归生成一次预测一个 token。MTP 的想法是一次预测多个 token，从而提高生成速度。

---

## 3.13 并行策略

老师接着提到一堆并行策略，例如：

- TP：Tensor Parallelism，张量并行；
- PP：Pipeline Parallelism，流水线并行；
- DP：Data Parallelism，数据并行；
- EP：Expert Parallelism，专家并行；
- CP：Context Parallelism，长上下文并行。

这些策略经常组合使用。难点不是背名字，而是知道：

> 一个特别大的模型，到底应该怎么切分到多张卡、多台机器上？

这涉及：

- 模型参数怎么切；
- batch 怎么切；
- 层怎么切；
- 序列长度怎么切；
- MoE 专家怎么切；
- 通信量怎么估算；
- 哪种并行会带来什么通信开销。

---

## 3.14 硬件和通信

最后老师讲硬件和通信。

他说，这部分面试可能没那么频繁，但实际工作中非常重要。比如：

- CUDA 基本关键字；
- 线程、block、grid；
- GPU 显存大小和带宽；
- HBM；
- 量化和显存之间的关系；
- checkpoint；
- NCCL；
- GPU 间通信；
- 拓扑结构；
- RDMA；
- InfiniBand；
- 不同 GPU 卡型的性能、显存、带宽；
- 多机多卡集群的通信协议。

他还说，实际工作中可能需要你帮公司算成本、算资源消耗、算某个模型需要多少卡、吞吐能达到多少。

这说明 AI Infra 不是纯算法，也不是纯代码。它是算法、系统、硬件和工程成本之间的综合优化。

---

# 四、视频中的 AI Infra 三大知识板块

根据视频内容，可以把 AI Infra 入门知识整理成下面这棵树：

```text
AI Infra
├── 1. 算法层
│   ├── Transformer
│   ├── Attention / MHA / MQA / GQA / MLA
│   ├── MoE
│   ├── FlashAttention / Online Softmax
│   ├── 模型评测
│   ├── 量化
│   └── 性能优化
│
├── 2. 框架层
│   ├── 推理框架
│   │   ├── vLLM
│   │   ├── SGLang
│   │   ├── TGI
│   │   └── TensorRT-LLM
│   ├── 训练框架
│   │   ├── Megatron
│   │   ├── DeepSpeed
│   │   ├── LLaMA-Factory
│   │   └── PyTorch FSDP
│   ├── 强化学习/后训练框架
│   │   ├── veRL
│   │   ├── TRL
│   │   └── OpenRLHF
│   └── Serving 机制
│       ├── KV Cache
│       ├── PagedAttention
│       ├── RadixAttention
│       ├── Prefix Caching
│       ├── Continuous Batching
│       ├── Chunked Prefill
│       ├── PD 分离
│       ├── Speculative Decoding
│       └── MTP
│
└── 3. 底层原理
    ├── CUDA 编程
    ├── 高性能算子
    ├── GEMM / Attention / Softmax / LayerNorm
    ├── GPU 架构
    ├── 显存 / 带宽 / HBM
    ├── NCCL
    ├── RDMA / InfiniBand
    ├── 多机多卡通信
    └── 并行策略 TP / PP / DP / EP / CP
```

---

# 五、第一部分：算法基础应该学什么

## 5.1 为什么 AI Infra 也要学算法

很多人误以为 AI Infra 是系统方向，只要会 C++、CUDA、Linux 就够了。但视频里强调，算法基础仍然是第一层。

原因是：

1. 推理框架服务的是模型；
2. 算子优化优化的是模型中的计算模块；
3. 显存优化绕不开模型结构；
4. KV Cache 来自自回归 Transformer；
5. MoE 并行来自模型架构；
6. 量化要理解权重、激活、精度损失；
7. 性能优化要知道瓶颈来自 Attention 还是 MLP。

所以 AI Infra 不要求你像算法研究员一样发明新模型，但必须能读懂模型结构。

---

## 5.2 Transformer 是图腾

视频里说 Transformer 是整个大模型知识体系的“图腾”。这句话可以理解为：

> 只要你理解 Transformer，后面的 Attention 变体、KV Cache、FlashAttention、MoE、量化、并行策略，都能找到位置。

Transformer Decoder-only LLM 的基本结构：

```text
Input Token IDs
    ↓
Token Embedding
    ↓
N × Decoder Block
    ├── RMSNorm / LayerNorm
    ├── Self-Attention
    │   ├── Q/K/V Projection
    │   ├── RoPE / Position Encoding
    │   ├── Attention Score
    │   ├── Softmax
    │   └── Output Projection
    ├── Residual Connection
    ├── RMSNorm / LayerNorm
    ├── MLP / FFN / MoE
    └── Residual Connection
    ↓
Final Norm
    ↓
LM Head
    ↓
Next Token Logits
```

你要能把每一层和工程问题对应起来：

| 模型模块 | 工程问题 |
|---|---|
| Attention | O(n²) 复杂度、KV Cache、FlashAttention |
| MLP / FFN | GEMM、算子融合、量化 |
| MoE | Expert routing、All-to-All、Fused MoE、负载均衡 |
| LayerNorm/RMSNorm | 小算子优化、融合 |
| RoPE | 长上下文扩展、位置编码 |
| LM Head | vocab projection、大矩阵计算 |
| KV Cache | 显存占用、PagedAttention、Prefix Caching |

---

## 5.3 Attention 家族

### MHA

MHA 是标准 Multi-Head Attention。每个 head 有自己的 Q/K/V。

优点：

- 表达能力强；
- 经典结构；
- 容易理解。

缺点：

- KV Cache 显存大；
- 长上下文下成本高。

### MQA

MQA 是 Multi-Query Attention。多个 query heads 共享较少的 key/value head。

优点：

- 显著减少 KV Cache；
- 推理更省显存；
- decode 更快。

缺点：

- 表达能力可能受影响；
- 相比 MHA 有一定折中。

### GQA

GQA 是 Grouped-Query Attention。它介于 MHA 和 MQA 之间，一组 query heads 共享一个 key/value head。

优点：

- 比 MHA 更省 KV Cache；
- 比 MQA 更保留表达能力；
- 现在很多 LLM 使用 GQA。

### MLA

MLA 是 DeepSeek 系列中非常重要的注意力设计。它的核心价值是：

> 降低 KV Cache 显存压力，并提升长上下文推理效率。

你不需要一开始完全推导 MLA，但要知道它和 KV Cache、显存优化、长上下文推理高度相关。

---

## 5.4 MoE

MoE 是 Mixture of Experts，混合专家模型。

浅层理解：

> 模型里有多个专家网络，每个 token 只选择其中一部分专家参与计算。

更工程化的理解：

> MoE 通过增加总参数量提升模型容量，但每个 token 只激活 Top-k 专家，因此每 token 计算量没有按总参数量同比例增加。

MoE 在 AI Infra 中带来几个系统问题：

1. **Router/Gate**：token 如何选择专家；
2. **Token Dispatch**：token 如何分发到专家；
3. **Expert Computation**：专家内部通常是 MLP/GEMM；
4. **Token Combine**：专家输出如何合并；
5. **Load Balance**：不同专家负载不均怎么办；
6. **Expert Parallelism**：专家分布在不同 GPU 时如何通信；
7. **All-to-All Communication**：token 跨卡分发带来通信成本；
8. **Fused MoE Kernel**：如何融合路由、排序、GEMM、合并等步骤。

所以面试问 MoE，不要只回答“门控网络 + 专家网络”。更好的回答是：

> MoE 的核心是用稀疏激活的方式扩大模型容量。每个 token 通过 router 选择 top-k experts，只在对应专家上计算，所以总参数可以很大，但单 token 计算量可控。推理系统中 MoE 会引入 token dispatch/combine、专家负载均衡、All-to-All 通信、Fused MoE kernel 和专家并行等问题。

---

## 5.5 FlashAttention 和 Online Softmax

FlashAttention 是 Attention 优化中的核心知识。

普通 Attention 的瓶颈：

- Attention score 矩阵很大；
- 显存读写开销高；
- 长序列下显存压力巨大。

FlashAttention 的核心思路：

> 不把完整 Attention 矩阵写回 HBM，而是通过分块计算和 Online Softmax，在 SRAM/shared memory 层级完成更多计算，减少 HBM 访问。

Online Softmax 的作用是：

> 在分块计算时保持 softmax 数值稳定，不需要一次性拿到完整序列的全部 attention scores。

对 AI Infra 来说，FlashAttention 不是单纯算法，而是算法和硬件存储层次结合的典型案例。

---

# 六、第二部分：框架层应该学什么

## 6.1 推理框架

视频里提到的推理框架主要包括：

- vLLM；
- SGLang；
- TGI；
- 其他 mini 版教学框架；
- 还可以扩展到 TensorRT-LLM、llama.cpp、MLC LLM 等。

推理框架要解决的问题：

1. 如何加载模型；
2. 如何接收请求；
3. 如何调度请求；
4. 如何管理 KV Cache；
5. 如何动态 batching；
6. 如何做 prefill/decode；
7. 如何支持流式输出；
8. 如何支持量化；
9. 如何支持多卡并行；
10. 如何暴露 API 服务；
11. 如何评估吞吐和延迟。

vLLM 官方文档中列出的核心能力包括 PagedAttention、Continuous Batching、Chunked Prefill、Prefix Caching、CUDA/HIP Graph、量化、优化的 Attention/GEMM/MoE kernel、Speculative Decoding 以及分布式推理并行能力。  
这和视频中提到的知识点高度吻合。

---

## 6.2 SGLang 和 RadixAttention

SGLang 也是重要推理服务框架。它的特点是：

- 面向大语言模型和多模态模型的高性能 serving；
- 支持低延迟和高吞吐；
- 支持 RadixAttention；
- 支持 PD 分离；
- 支持 Speculative Decoding；
- 支持 Continuous Batching；
- 支持 PagedAttention；
- 支持 TP/PP/EP/DP；
- 支持结构化输出和多 LoRA batching。

RadixAttention 可以理解为围绕 prefix 复用的一套机制。对于 Agent、多轮对话、长上下文、重复 prompt 场景，前缀复用能显著减少重复计算。

---

## 6.3 训练框架

训练框架和推理框架关注点不同。

推理框架关注：

- 在线服务；
- 并发请求；
- 延迟；
- 吞吐；
- KV Cache；
- 显存碎片；
- decoding。

训练框架关注：

- 分布式训练；
- 反向传播；
- optimizer state；
- gradient；
- activation；
- checkpoint；
- 数据并行；
- 张量并行；
- 流水线并行；
- ZeRO/FSDP；
- 混合精度；
- 大规模容错。

常见训练框架：

- PyTorch DDP/FSDP；
- DeepSpeed；
- Megatron / Megatron-Core；
- LLaMA-Factory；
- torchtune；
- Colossal-AI；
- Lightning Fabric；
- verl / TRL / OpenRLHF 等后训练框架。

---

## 6.4 为什么 LLaMA-Factory 适合入门但不适合深入

视频中说 LLaMA-Factory 封装较重，入门可以，进阶不推荐完全依赖。

这句话可以理解为：

- 入门时，它能帮你快速跑通 SFT、LoRA、DPO 等流程；
- 但如果你想深入训练系统、改 loss、改数据流、改并行策略、改 rollout 或 RL 算法，它的封装会变成限制；
- 对 AI Infra 求职来说，只会用框架不够，最好知道框架底层如何组织训练流程。

所以你的学习策略应该是：

> 先用高级框架跑通流程，再逐步下沉到 PyTorch、Megatron、DeepSpeed、verl 等更底层或更工程化的框架。

---

# 七、第三部分：底层原理应该学什么

## 7.1 底层原理不是只会 CUDA

底层原理包括：

- CUDA 编程；
- C++；
- Python；
- 高性能算子；
- GPU 架构；
- 显存层次；
- 访存优化；
- 通信库；
- 多机多卡；
- 分布式并行；
- 网络拓扑；
- 硬件资源估算。

视频里说：

> 高性能算子本质上也可以算是框架的一部分，因为它最终服务于框架。但如果作为考点，它也属于底层原理。

也就是说，算子不是孤立存在的。你写的 GEMM、Attention、LayerNorm、Fused MoE，最终都要服务于：

- 推理框架；
- 训练框架；
- 模型结构；
- 硬件加速；
- 显存节省；
- 吞吐提升。

---

## 7.2 CUDA 编程

CUDA 入门至少要会：

- thread / block / grid；
- warp；
- global memory；
- shared memory；
- register；
- memory coalescing；
- bank conflict；
- warp divergence；
- occupancy；
- CUDA stream；
- CUDA event；
- atomic；
- synchronization；
- kernel launch；
- Nsight Compute；
- Nsight Systems。

经典算子：

1. Vector Add；
2. Reduce；
3. Matrix Transpose；
4. GEMM；
5. Softmax；
6. LayerNorm / RMSNorm；
7. Attention；
8. FlashAttention 简化版；
9. Fused MoE 简化版。

---

## 7.3 高性能算子要关注什么

不能只关注“代码能跑”，还要关注：

- 是否减少 HBM 访问；
- 是否提高 shared memory 利用；
- 是否减少 bank conflict；
- 是否减少 warp divergence；
- 是否提高 occupancy；
- 是否使用 Tensor Core；
- 是否做了 tiling；
- 是否做了 fusion；
- 是否避免重复读写；
- 是否改善 memory throughput；
- 是否真的提升 end-to-end latency，而不是只提升 micro benchmark。

---

## 7.4 通信原理

AI Infra 的通信包括：

- 单机多卡通信；
- 多机多卡通信；
- AllReduce；
- AllGather；
- ReduceScatter；
- Broadcast；
- All-to-All；
- Send/Recv；
- NCCL；
- RDMA；
- InfiniBand；
- NVLink；
- PCIe；
- 网络拓扑。

NCCL 是 NVIDIA 针对多 GPU / 多节点通信优化的集合通信库，支持 AllReduce、AllGather、Broadcast、ReduceScatter、点对点 Send/Recv 等通信原语。它在分布式训练和推理中都非常重要。

---

# 八、推理框架核心知识点详解

## 8.1 KV Cache

自回归生成时，模型每次生成一个 token。为了避免每一步重复计算历史 token 的 K/V，需要把历史 token 的 K/V 保存下来，这就是 KV Cache。

KV Cache 的显存占用大致和这些因素有关：

- batch size；
- sequence length；
- layer 数；
- hidden size；
- attention head 数；
- KV head 数；
- dtype；
- 是否 MHA/MQA/GQA/MLA。

所以长上下文和大并发都会迅速推高显存压力。

---

## 8.2 PagedAttention

PagedAttention 主要解决 KV Cache 管理问题。

传统 KV Cache 管理可能出现：

- 显存碎片；
- 预留过多；
- 动态增长困难；
- 多请求并发下浪费明显。

PagedAttention 借鉴操作系统分页思想，把 KV Cache 拆成 block/page 管理，从而提高显存利用率，让同样显存可以服务更多并发请求。

---

## 8.3 Prefix Caching

Prefix Caching 的作用是复用相同前缀的 KV Cache。

适用场景：

- 系统 prompt 固定；
- 多轮对话；
- Agent 工具调用；
- RAG 中固定 instruction；
- 批量评测；
- 长文档重复提问。

核心价值：

> 避免重复计算相同前缀，降低 TTFT，节省显存和计算。

---

## 8.4 RadixAttention

RadixAttention 可以理解为更系统化地管理和复用 prompt 前缀。它常与 SGLang 关联。

它适合：

- 共享前缀多；
- 多轮对话；
- Agent 程序；
- 结构化 prompt；
- 树状请求；
- 长上下文复用。

---

## 8.5 Continuous Batching

Continuous Batching 解决在线推理中的动态调度问题。

静态 batching 的问题：

- 一批请求一起开始；
- 要等整批完成；
- 长请求拖累短请求；
- GPU 利用率不稳定。

Continuous Batching 的方法：

- 每一步动态组成 batch；
- 完成的请求立刻移除；
- 新请求及时加入；
- 让 GPU 尽量持续工作。

---

## 8.6 Chunked Prefill

Chunked Prefill 把长 prompt 的 prefill 拆成多个 chunk。

解决问题：

- 长 prompt 阻塞 decode；
- prefill 计算密集；
- decode 访存密集；
- 二者混合调度可以提升 GPU 利用率；
- 降低等待时间和尾延迟。

---

## 8.7 PD 分离

PD 分离是 Prefill-Decode Disaggregation。

Prefill 和 Decode 特点不同：

| 阶段 | 主要特点 |
|---|---|
| Prefill | 处理输入 prompt，计算密集，适合大 batch |
| Decode | 每次生成一个 token，访存密集，强依赖 KV Cache |

把二者分离后，可以分别调度、分别部署、分别优化资源。

---

## 8.8 Speculative Decoding

投机解码流程：

1. 小模型 draft model 先生成多个候选 token；
2. 大模型 target model 并行验证；
3. 如果候选 token 被接受，则一次推进多个 token；
4. 如果不接受，则回退或修正。

适用条件：

- 小模型足够快；
- 小模型和大模型输出分布接近；
- 验证成本低于逐 token 生成；
- 接受率较高。

---

## 8.9 MTP

MTP 是 Multi-Token Prediction。

普通自回归模型一次预测一个 token。MTP 让模型一次预测多个未来 token，用于提升训练或推理效率。

在面试中，你可以把 MTP 和 speculative decoding 关联起来：

> 它们都试图减少“每次只推进一个 token”的低效率问题，只是具体实现路径不同。

---

## 8.10 EPLB

EPLB 可以理解为 Expert Parallel Load Balancing。

它出现在 MoE 专家并行场景中。

问题：

- 不同专家被 token 选中的频率不同；
- 热门专家负载高；
- 冷门专家空闲；
- 多卡之间通信和计算不平衡；
- 会影响吞吐和延迟。

解决方向：

- 静态专家布局；
- 动态专家重排；
- token routing 优化；
- expert parallel balance；
- All-to-All 优化。

---

# 九、训练框架与强化学习框架应该怎么理解

## 9.1 训练框架和推理框架的区别

| 维度 | 训练框架 | 推理框架 |
|---|---|---|
| 核心目标 | 训练大模型 | 服务大模型 |
| 关键指标 | MFU、吞吐、收敛、显存 | TTFT、TPOT、QPS、吞吐、延迟 |
| 关键对象 | 参数、梯度、优化器状态、激活 | KV Cache、请求、batch、token |
| 典型技术 | DP/TP/PP/ZeRO/FSDP/Checkpoint | PagedAttention/Continuous Batching/Prefix Cache |
| 代表框架 | Megatron、DeepSpeed、FSDP | vLLM、SGLang、TGI、TensorRT-LLM |

---

## 9.2 Megatron

Megatron/Megatron-Core 主要服务于大规模训练。

它的核心价值：

- 张量并行；
- 流水线并行；
- 序列并行；
- 上下文并行；
- 专家并行；
- 分布式 checkpoint；
- activation checkpoint；
- 面向大规模 GPU 集群。

Megatron-Core 官方文档把 DP、TP、PP、CP、EP 等作为可组合的并行策略，用于从数十亿到数万亿参数模型训练。

---

## 9.3 DeepSpeed

DeepSpeed 的核心关键词：

- ZeRO；
- Pipeline Parallelism；
- Offload；
- 大规模训练；
- 显存优化；
- 分布式优化。

DeepSpeed 的 PipelineModule 将模型 forward 表示成顺序层，从而可以按 stage 做流水线划分。

---

## 9.4 LLaMA-Factory

适合：

- 快速入门；
- 跑通 SFT；
- 跑通 LoRA；
- 跑通 DPO；
- 做小规模微调实验。

不适合：

- 深度改训练流程；
- 深入研究分布式训练；
- 深入掌握框架底层；
- 作为 AI Infra 主项目。

---

## 9.5 veRL / TRL / OpenRLHF

这些框架偏后训练和强化学习。

常见关键词：

- PPO；
- DPO；
- GRPO；
- RLHF；
- RLVR；
- reward model；
- rollout；
- actor；
- critic；
- reference model；
- KL penalty；
- advantage。

如果你的目标是推理框架/模型部署，这部分可以先了解概念，不必作为主线。  
如果你后续想做大模型后训练 Infra，就需要深入。

---

# 十、硬件、通信和分布式系统知识

## 10.1 GPU 关键指标

你需要知道：

- 显存容量；
- 显存带宽；
- FP16/BF16/FP8/INT8/INT4 算力；
- Tensor Core；
- NVLink；
- PCIe；
- HBM；
- SM 数量；
- shared memory；
- L2 cache；
- 功耗；
- 多卡拓扑。

面试和工作中可能需要估算：

- 一个模型需要多少显存；
- KV Cache 占多少；
- 一台机器能跑多大模型；
- batch size 能开多大；
- 用 FP16 / INT8 / INT4 差多少；
- TP/PP/DP 该怎么配置；
- 通信是否成为瓶颈。

---

## 10.2 并行策略

### DP：Data Parallelism

切 batch。每张卡一份模型，处理不同数据。

### TP：Tensor Parallelism

切张量/矩阵。适合单层太大，一张卡放不下或算不动。

### PP：Pipeline Parallelism

按层切模型。不同 GPU 负责不同层。

### CP：Context Parallelism

按序列长度切。适合长上下文。

### EP：Expert Parallelism

MoE 专家并行。不同专家放到不同 GPU。

这些策略不是孤立的。实际大模型训练/推理经常组合使用。

---

## 10.3 通信模式

| 通信模式 | 常见场景 |
|---|---|
| AllReduce | 数据并行梯度同步 |
| AllGather | 参数/激活聚合 |
| ReduceScatter | ZeRO/FSDP/张量并行 |
| Broadcast | 参数同步 |
| All-to-All | MoE expert parallel、序列并行 |
| Send/Recv | Pipeline parallel |

其中 MoE 特别容易引入 All-to-All，因为 token 要被分发到不同专家所在 GPU。

---

## 10.4 RDMA 和 InfiniBand

RDMA 允许一台机器直接访问另一台机器内存，减少 CPU 参与和拷贝开销。  
InfiniBand 是高性能集群常用网络。

在大模型训练和推理集群中，它们关系到：

- 多机通信延迟；
- 带宽；
- AllReduce 性能；
- All-to-All 性能；
- 集群扩展效率；
- 训练成本；
- 推理吞吐。

---

# 十一、这期视频和前几期视频的关系

结合你之前上传的几期字幕，可以形成如下关系：

## 第一类：AI 应用开发路线

之前“后端开发转 AI 的学习路线”更偏：

- LangChain；
- Spring AI；
- Eino；
- RAG；
- Agent；
- AI 应用开发。

这类适合后端开发转 AI 应用。

## 第二类：大模型推理框架校招规划

之前“怎么学习大模型推理框架”更偏：

- CUDA；
- vLLM；
- PagedAttention；
- Continuous Batching；
- Qwen/DeepSeek 模型结构；
- 算子开发；
- 校招面试。

这类适合 AI Infra 推理侧求职。

## 第三类：AI Infra 总体系梳理

本期视频更偏：

- 算法；
- 框架；
- 底层原理；
- 推理框架；
- 训练框架；
- 强化学习框架；
- 硬件通信；
- 分布式并行。

它是前两类内容的上位知识地图。

---

# 十二、结合你的背景：你应该怎么学 AI Infra

你的背景特点：

- 电子信息/新一代电子信息技术研一；
- 当前做 IMS 离子迁移谱仪相关研究；
- 有 ESP32-S3、LVGL、TFT、Web AP、嵌入式控制系统经历；
- 接触过采集、显示、远程控制、系统联调；
- 正在学 Linux/OS；
- 目标是 2027 年暑期秋招；
- 对 AI Infra、推理侧、模型部署、端侧/边缘 AI 感兴趣。

基于这期视频，我对你的判断是：

> 你应该把 AI Infra 当成一个“系统工程方向”来学，而不是只学 AI 应用，也不是直接硬刚纯 CUDA 算子。

更适合你的路线是：

```text
嵌入式/Linux基础
    ↓
C++ / Python / PyTorch
    ↓
Transformer / Attention / MoE / 量化
    ↓
ONNX / TensorRT / vLLM / SGLang
    ↓
模型部署与推理服务性能分析
    ↓
CUDA 经典算子入门
    ↓
IMS 谱图识别 / 边缘 AI 部署项目
    ↓
AI Infra / 模型部署 / 端侧 AI / 推理框架相关岗位
```

---

# 十三、2026-2027 秋招学习路线

## 阶段 1：2026 年 6-8 月  
### 目标：打基础

重点：

1. C++  
   - 指针、引用、类、模板、STL；
   - RAII；
   - 多线程；
   - CMake；
   - gdb。

2. Linux  
   - 进程/线程；
   - 文件 IO；
   - socket；
   - 内存；
   - shell；
   - perf。

3. Python / PyTorch  
   - Tensor；
   - autograd；
   - module；
   - dataloader；
   - 简单模型训练；
   - 模型保存/加载。

4. Transformer 基础  
   - Attention；
   - Decoder-only；
   - KV Cache；
   - MHA/MQA/GQA；
   - MoE；
   - 量化基本概念。

阶段产出：

> 写一篇“Transformer 推理流程笔记”，并跑通一个小模型的 PyTorch 推理。

---

## 阶段 2：2026 年 9-12 月  
### 目标：进入模型部署

重点：

- ONNX Runtime；
- TensorRT；
- Triton Inference Server；
- vLLM；
- llama.cpp；
- SGLang 入门；
- 量化；
- benchmark；
- TTFT / TPOT / QPS / tokens/s / 显存。

阶段项目：

> LLM 推理服务性能分析平台。

你要能对比：

- 不同模型；
- 不同输入长度；
- 不同输出长度；
- 不同并发；
- 不同量化；
- vLLM vs llama.cpp；
- GPU 显存和吞吐变化。

---

## 阶段 3：2027 年 1-3 月  
### 目标：深入推理框架和 CUDA

重点：

- vLLM 架构；
- PagedAttention；
- Continuous Batching；
- Chunked Prefill；
- Prefix Caching；
- PD 分离；
- Speculative Decoding；
- CUDA 基础；
- Reduce / GEMM / Softmax / LayerNorm；
- Nsight Compute。

阶段项目：

> CUDA 经典算子优化 + vLLM 源码阅读笔记。

---

## 阶段 4：2027 年 4-6 月  
### 目标：项目整合和面试准备

重点：

- 简历项目重构；
- C++ 八股；
- Linux/OS；
- 计算机网络；
- CUDA 面试；
- vLLM 面试；
- 模型结构面试；
- IMS 项目包装；
- 投递提前批和实习。

最终简历项目组合建议：

1. IMS 嵌入式控制系统；
2. LLM 推理服务性能分析平台；
3. IMS 谱图智能识别与边缘部署；
4. CUDA 算子优化实验。

---

# 十四、可落地项目建议

## 项目 1：LLM 推理服务性能分析平台

### 项目定位

AI Infra / 推理部署。

### 内容

- 使用 vLLM 部署 Qwen/DeepSeek Distill/Llama 小模型；
- 支持 OpenAI-compatible API；
- 编写压测脚本；
- 统计 TTFT、TPOT、tokens/s、QPS、显存；
- 测试不同并发和上下文长度；
- 分析 PagedAttention、Continuous Batching、Chunked Prefill；
- 写项目报告。

### 简历描述

> 基于 vLLM 搭建大模型推理服务性能分析平台，支持 OpenAI-compatible API 调用和多并发压测。实现 TTFT、TPOT、tokens/s、QPS、GPU 显存等指标统计，分析输入长度、输出长度、并发数、量化方式对推理性能的影响，并结合 PagedAttention、Continuous Batching、Chunked Prefill 等机制解释性能变化。

---

## 项目 2：CUDA 经典算子优化

### 项目定位

算子开发 / GPU 编程。

### 内容

- Reduce；
- Matrix Transpose；
- Softmax；
- LayerNorm；
- GEMM；
- 简化 Attention；
- 使用 Nsight Compute；
- 分析 memory throughput、occupancy、bank conflict、warp divergence；
- 写优化前后对比。

### 简历描述

> 实现 Reduce、Softmax、LayerNorm、GEMM 等 CUDA 经典算子，并基于 Nsight Compute 分析性能瓶颈。针对访存不连续、shared memory bank conflict、warp divergence 等问题进行优化，记录优化前后 latency、memory throughput、occupancy 等指标变化。

---

## 项目 3：IMS 谱图智能识别与边缘部署

### 项目定位

你的差异化项目：嵌入式 + AI + 模型部署。

### 内容

- IMS 谱图采集；
- 峰检测；
- 漂移时间提取；
- 峰值、峰宽、面积等特征；
- 传统机器学习分类；
- 小型神经网络分类；
- ONNX 导出；
- 边缘设备或本地推理服务部署；
- 显示端和 Web 端展示识别结果。

### 简历描述

> 面向离子迁移谱仪谱图识别任务，构建从数据采集、峰检测、特征提取到轻量模型推理的端到端流程。基于漂移时间、峰值强度、峰宽等特征训练分类模型，并导出 ONNX 进行边缘侧部署，实现仪器采集数据的实时分析与本地/远程显示。

---

## 项目 4：AI Infra 知识库 + 面试助手

### 项目定位

AI 应用层包装，不作为主线，但可辅助学习。

### 内容

- 把 AI Infra 论文、博客、源码笔记做成 RAG 知识库；
- 支持查询 vLLM、CUDA、MoE、KV Cache；
- 每个回答带引用；
- 作为学习工具；
- 不要把它包装成唯一核心项目。

---

# 十五、面试问题清单

## 15.1 算法层

1. Transformer Decoder-only 结构是什么？  
2. Attention 的 Q/K/V 是什么？  
3. MHA、MQA、GQA、MLA 有什么区别？  
4. KV Cache 为什么能加速推理？  
5. KV Cache 显存占用怎么估算？  
6. MoE 为什么可以增加参数量但不同比例增加计算量？  
7. MoE 推理中为什么会有 All-to-All？  
8. FlashAttention 为什么快？  
9. Online Softmax 解决什么问题？  
10. Pre-LN 和 Post-LN 有什么区别？  
11. RMSNorm 和 LayerNorm 区别是什么？  
12. 什么是模型量化？  
13. GPTQ 和 AWQ 大概区别是什么？  
14. FP16、BF16、FP8、INT8、INT4 有什么区别？  
15. 长上下文为什么会带来推理压力？

---

## 15.2 推理框架

1. vLLM 解决什么问题？  
2. PagedAttention 解决什么问题？  
3. Continuous Batching 和 Static Batching 区别是什么？  
4. Chunked Prefill 为什么能改善延迟？  
5. Prefill 和 Decode 的计算特点有什么不同？  
6. PD 分离是什么？  
7. Prefix Caching 适合什么场景？  
8. RadixAttention 是什么？  
9. Speculative Decoding 流程是什么？  
10. MTP 和普通自回归生成有什么区别？  
11. EPLB 解决什么问题？  
12. 推理服务中 TTFT、TPOT、QPS、tokens/s 分别是什么？  
13. 如何设计一个推理 benchmark？  
14. 如何判断推理瓶颈是算力、显存还是通信？  
15. vLLM、SGLang、TGI 的定位有什么区别？

---

## 15.3 训练框架

1. DP、TP、PP、EP、CP 分别是什么？  
2. Megatron 的核心并行策略有哪些？  
3. DeepSpeed ZeRO 解决什么问题？  
4. FSDP 和 ZeRO 有什么关系？  
5. Pipeline Parallelism 为什么会有 bubble？  
6. Activation Checkpointing 用来解决什么问题？  
7. LLaMA-Factory 适合什么，不适合什么？  
8. PPO、DPO、GRPO 大概是什么？  
9. RLHF 的基本流程是什么？  
10. 后训练框架和预训练框架有什么区别？

---

## 15.4 底层原理

1. CUDA thread/block/grid 是什么？  
2. warp 是什么？  
3. 什么是 warp divergence？  
4. 什么是 shared memory bank conflict？  
5. memory coalescing 是什么？  
6. GEMM 为什么重要？  
7. Tensor Core 是什么？  
8. Nsight Compute 常看哪些指标？  
9. 什么是 memory-bound 和 compute-bound？  
10. 算术强度是什么？  
11. NCCL 支持哪些通信原语？  
12. AllReduce 和 All-to-All 区别是什么？  
13. RDMA 解决什么问题？  
14. InfiniBand 在大模型集群中有什么作用？  
15. GPU 显存带宽为什么重要？

---

# 十六、最终结论

这期视频的核心价值是把 AI Infra 的知识树讲清楚：

> **AI Infra = 算法结构理解 + 框架系统实现 + 底层硬件/通信/算子原理。**

对你来说，最重要的不是一次性学完所有内容，而是按优先级建立路线。

你的最佳路线不是纯 AI 应用，也不是一开始就硬冲顶级算子，而是：

```text
C++ / Linux / Python / PyTorch
    ↓
Transformer / Attention / MoE / KV Cache
    ↓
vLLM / SGLang / TensorRT / ONNX Runtime
    ↓
推理服务 benchmark
    ↓
CUDA 经典算子入门
    ↓
IMS 边缘 AI / 模型部署项目
    ↓
2027 秋招：模型部署 / 推理框架 / 端侧 AI / AI Infra
```

你真正应该形成的竞争力是：

> **懂嵌入式系统，懂 Linux/C++，能做模型部署和推理服务分析，理解大模型结构和 CUDA 基础，并能把 AI 落到真实仪器/边缘设备场景中。**

这比只做一个普通 RAG 项目更有区分度，也比一上来硬拼纯 CUDA 算子更适合你的背景。

---

# 十七、参考资料

以下资料用于校正本视频中的 AI Infra 技术名词，也建议作为后续学习参考：

1. vLLM 官方文档  
   https://docs.vllm.ai/en/stable/

2. SGLang 官方文档  
   https://sgl-project.github.io/

3. Megatron-Core 并行策略文档  
   https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html

4. NVIDIA NCCL 官方文档  
   https://docs.nvidia.com/deeplearning/nccl/

5. NVIDIA NCCL Developer 页面  
   https://developer.nvidia.com/nccl

6. DeepSpeed Pipeline Parallelism 文档  
   https://deepspeed.readthedocs.io/en/latest/pipeline.html

7. FlashAttention 论文  
   https://arxiv.org/abs/2205.14135

8. vLLM / PagedAttention 论文  
   https://arxiv.org/abs/2309.06180

9. SARATHI / Chunked Prefill 相关论文  
   https://arxiv.org/abs/2308.16369

---

# 附录：原始字幕信息

- 原始字幕字符数：6304
- 原始字幕主题：AI Infra 入门知识完整梳理
- 字幕主要问题：大量技术词误识别、缺少断句、部分英文框架名不准确
- 本文处理原则：保留原视频讲课逻辑，修正技术词，扩展成可复习的 AI Infra 学习笔记



%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-AI infra学习路线|模块-AI infra学习路线]]

%% 项目关联导航：结束 %%
