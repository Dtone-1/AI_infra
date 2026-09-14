# AI Infra 规划课与模拟面：2026 年 5 月 30 日职业规划实录  
## 课程内容深度整理 + 学习笔记 + 面试问答

> 输入材料：`AI-Infra规划课和模拟面_2026年5月30日职业规划实录_哔哩哔哩_bilibili.txt`  
> 整理目标：基于自动字幕进行技术名词纠错、语义复原、结构化学习笔记生成，并结合 AI Infra / 大模型推理 / CUDA 算子 / 推理框架求职准备进行深度分析。  
> 说明：原始字幕存在错别字、重复口语、断句错误和技术词误识别，本文不是逐字稿，而是按课程主线重构后的学习笔记。

---

## 目录

1. [本节课一句话结论](#1-本节课一句话结论)
2. [视频整体宗旨](#2-视频整体宗旨)
3. [字幕技术名词纠错表](#3-字幕技术名词纠错表)
4. [按课程推进顺序梳理知识点](#4-按课程推进顺序梳理知识点)
5. [核心技术点深度解释](#5-核心技术点深度解释)
6. [整节课的“从输入到输出”技术主线](#6-整节课的从输入到输出技术主线)
7. [结合 AI Infra 求职的路线建议](#7-结合-ai-infra-求职的路线建议)
8. [面试高频问题与参考答案](#8-面试高频问题与参考答案)
9. [可直接执行的学习与项目打磨清单](#9-可直接执行的学习与项目打磨清单)
10. [最终总结](#10-最终总结)

---

# 1. 本节课一句话结论

这节课的核心不是单纯讲某个技术点，而是围绕几位同学的背景与项目，讨论 **AI Infra 求职中如何选择方向、如何打磨项目、如何补齐 CUDA / 推理框架 / 大模型基础，以及如何把已有 Linux、HPC、科研项目转化成 AI Infra 简历竞争力**。

最重要的结论是：

> **AI Infra 不是只靠“学几个热门词”就能投递的方向。你需要根据目标岗位拆分能力：如果偏 CUDA/算子，就要能手写常见算子并讲清优化；如果偏推理框架，就要理解端到端推理流程、调度、KV Cache、资源管理；如果已有 Linux/HPC/嵌入式背景，要把这些经历翻译成“系统能力 + 性能优化能力 + 工程落地能力”。**

---

# 2. 视频整体宗旨

本节课主要围绕三类学生的问题展开。

第一类同学有 Linux / 系统 / 驱动背景，正在考虑从传统系统开发转向 AI Infra。他的问题集中在：

- 是否必须学 CS336；
- 是否需要学 Agent；
- CUDA 算子要学到什么程度；
- AI Infra 是否真的还有岗位需求；
- 端侧推理、机器人、车企、NPU / TensorRT 是否值得投；
- K8S / Go 推理平台项目和真正 AI Infra 岗位的关系。

第二类同学有 HPC / 科研算子优化背景，已经在算子相关岗位实习或入职。他的问题集中在：

- HPC 项目能否迁移到 AI Infra；
- 算子开发和推理框架哪个前景更广；
- 是否要补 CUTLASS；
- 是否要补训练框架；
- 简历里如何体现 CUDA 能力和性能优化数据。

第三类同学以 nano-vLLM / KV Cache 优化 / 视觉部署项目为主，正在准备找 AI Infra 实习。他的问题集中在：

- nano-vLLM 项目是否过于常见；
- 自己的 KV Cache 压缩优化如何讲；
- YOLO / Jetson / 相机部署项目是否太浅；
- 是否需要新做项目；
- 面试官会从项目里问什么；
- 学历和无实习经历如何弥补；
- 应该先冲大厂还是先找中小厂实习。

这节课的真正目标是：

> **帮助 AI Infra 初学者把“方向选择、项目包装、技术补短板、面试准备”串成一条可执行路线。**

---

# 3. 字幕技术名词纠错表

| 字幕识别 | 推荐修正 | 说明 |
|---|---|---|
| 演艺 / 言一 | 研一 | 学生年级 |
| 孤大 / 苦大 / 哭大 | CUDA | NVIDIA GPU 编程模型 |
| 算字 / 擅自 | 算子 | Operator / Kernel |
| Transport 算字 | Transpose 算子 | CUDA 高频手撕算子之一 |
| MathMal / Mamory | MatMul / Memory | 矩阵乘法 / 存储 |
| CAS336 | CS336 | Stanford CS336，大语言模型相关课程 |
| Agent | Agent | 大模型应用层智能体 |
| 推理框架 | Inference Framework / LLM Serving Engine | vLLM、SGLang、TensorRT-LLM 等 |
| 短侧 / 断策 | 端侧 | Edge / On-device inference |
| 支架 | 智驾 | 自动驾驶 / 智能驾驶 |
| 车起 | 车企 | 汽车公司 |
| 新的人车 | 新能源车 | 车载智能模型部署场景 |
| 推行平台 | 推理平台 | 推理服务或部署平台 |
| K-8S | K8s / Kubernetes | 容器编排系统 |
| 运为 | 运维 | DevOps / MLOps 相关 |
| 世界家 | C++ | 字幕将 C++ 误识别 |
| 新盘 / 缺盘 | Triton | 可能指 Triton 算子开发语言，结合上下文判断 |
| 酷特拉斯 / 库特拉斯 | CUTLASS | NVIDIA CUDA Templates for Linear Algebra Subroutines |
| KVCash | KV Cache | 大模型推理缓存 |
| 钱质匹配 | 前缀匹配 / Prefix Caching | 复用相同 prompt 前缀的 KV Cache |
| PagedAttention | PagedAttention | vLLM 核心 KV Cache 管理机制 |
| Preview | Prefill | prompt 预填充阶段 |
| Decode | Decode | 自回归解码阶段 |
| Nsise System / ensize system | Nsight Systems | NVIDIA 性能分析工具 |
| Nsight Compute | Nsight Compute | NVIDIA kernel 性能分析工具 |
| Hugface / Hackface | Hugging Face | 模型与 Transformers 生态 |
| Transformers | Transformers 库 | Hugging Face 模型推理库 |
| YOLO / 幽露 | YOLO | 目标检测模型 |
| Jetson / 杰斯 | Jetson | NVIDIA 边缘计算平台 |
| TensorRT / TNT | TensorRT | NVIDIA 推理优化引擎 |
| ONEX | ONNX | 开放神经网络交换格式 |

---

# 4. 按课程推进顺序梳理知识点

## 4.1 第一段：学习 AI Infra 是否要看兴趣

### 核心知识点

课程开头讨论了一个很关键的问题：**选择 AI Infra 是否只是因为热门，还是自己真的对系统、底层、性能优化有兴趣**。

主讲人的判断是：个人兴趣很重要。如果只是因为方向热门而硬学，长期成长会比较困难；但如果本来对 Linux、系统、驱动、底层开发有兴趣，那么 CUDA 和 AI Infra 反而是比较自然的延伸。

### 相关概念解释

CUDA 虽然属于 GPU 编程，但本质上仍然是底层并行编程。它涉及：

- 线程组织；
- 内存层级；
- 并行计算；
- 显存访问；
- 性能瓶颈分析；
- 和硬件架构相关的优化。

所以对 Linux、驱动、嵌入式、系统开发有兴趣的人，通常更容易接受 CUDA 的思维方式。

### 在 AI Infra 中的作用

AI Infra 的很多问题不是单纯写业务逻辑，而是面对有限硬件资源做性能优化。比如：

- 同样显存如何服务更多请求；
- 同样 GPU 如何提高吞吐；
- 同样模型如何降低延迟；
- 如何减少 kernel launch 开销；
- 如何提升 memory bandwidth 利用率。

这和系统方向的“资源管理”和“性能优化”思维是相通的。

---

## 4.2 第二段：CS336 和 Agent 是否必须学

### 核心知识点

课程中讨论了 Stanford CS336 和 Agent 的学习必要性。

主讲人的观点是：

- **CS336 可以作为大模型算法、推理与系统的通识课，有时间可以学。**
- **Agent 不是 AI Infra 推理岗位的强制技能。**
- 如果目标是大模型推理框架、CUDA 算子、推理优化，不学 Agent 也可以找到相关工作。
- 如果个人感兴趣，Agent 可以作为拓展知识，帮助理解上层应用如何调用推理系统。

### 相关概念解释

CS336 这类课程通常覆盖：

- Transformer；
- Tokenizer；
- 训练数据；
- Scaling Law；
- 训练流程；
- 推理基础；
- 模型评测；
- 部分系统问题。

它适合作为“理解大模型整体生态”的入口，但不等于求职 AI Infra 的全部技能。

Agent 主要属于 LLM 应用层，关注：

- 任务规划；
- 工具调用；
- 多轮推理；
- 记忆管理；
- 外部 API 交互；
- 多智能体协作。

### 在 AI Infra 中的作用

Infra 与 Agent 的关系是上下游关系：

```text
Agent / RAG / 应用层
        ↓ 调用
LLM 推理服务 API
        ↓ 承载
vLLM / SGLang / TensorRT-LLM
        ↓ 执行
CUDA / Triton / GPU kernel
```

做 Infra 不一定要懂 Agent 细节，但如果知道 Agent 的请求模式，就能更好理解为什么推理框架需要：

- Prefix Caching；
- 长上下文优化；
- 多轮对话缓存；
- 高并发 serving；
- 流式输出；
- 延迟与吞吐平衡。

---

## 4.3 第三段：CUDA 算子学习要到什么程度

### 核心知识点

主讲人明确指出，CUDA 算子面试不是要求你现场写出极致优化版本，而是要掌握一个合理的面试策略：

1. 先写出基础版本；
2. 解释基础版本的执行流程；
3. 在面试官提示下逐步加入优化；
4. 对常见算子准备一个能熟练手写的优化版本。

尤其是 Transpose、MatMul 等算子，面试时通常不会要求直接手写最复杂版本，但至少要知道常见优化路线。

### 相关概念解释

常见 CUDA 算子包括：

- Vector Add；
- Reduce；
- Transpose；
- Softmax；
- LayerNorm / RMSNorm；
- MatMul / GEMM；
- Attention 相关 kernel。

这些算子的优化通常围绕：

- 合并访存；
- shared memory；
- tiling；
- 减少 bank conflict；
- 减少 warp divergence；
- 提高 occupancy；
- 减少全局内存读写；
- 使用 Tensor Core。

### 在 AI Infra 中的作用

大模型推理的底层计算最终会落到算子上。比如：

| 大模型模块 | 对应算子 |
|---|---|
| Attention QKV 投影 | GEMM |
| Attention score | MatMul |
| Softmax | Softmax kernel |
| MLP / FFN | GEMM |
| RMSNorm / LayerNorm | Norm kernel |
| KV Cache 写入/读取 | memory copy / cache kernel |
| MoE expert 计算 | Grouped GEMM / Fused MoE |

所以即使目标是推理框架，CUDA 也能帮助你理解底层性能瓶颈。

---

## 4.4 第四段：AI Infra 是否“快没需求了”

### 核心知识点

课程中有人担心 AI Infra 是否需求变少。主讲人的判断是：

> AI Infra 仍处在行业上升期，不是“快没需求”。只要大模型应用请求增长，而硬件资源有限，就需要软件层面的推理优化。

### 相关概念解释

大模型服务的成本主要来自：

- GPU 显存；
- GPU 算力；
- 网络通信；
- 请求并发；
- token 生成速度；
- 长上下文成本；
- 多轮对话缓存成本。

当用户量上升时，公司不可能无限买 GPU，因此需要通过软件优化提高单位硬件的服务能力。

### 在 AI Infra 中的作用

AI Infra 的价值可以概括为：

> **让同样的 GPU 服务更多请求、生成更多 token、占用更少显存、保持更低延迟。**

这也是 vLLM、SGLang、TensorRT-LLM、CUDA kernel、量化、KV Cache 优化持续存在的原因。

---

## 4.5 第五段：端侧推理、车企、机器人和云端 Infra 的区别

### 核心知识点

课程讨论了端侧推理是否值得投。结论是：值得投，但要理解它和互联网云端大模型推理不是同一个场景。

端侧推理常见于：

- 智驾；
- 车载座舱；
- 机器人；
- Jetson 平台；
- NPU / TensorRT / TensorRT Lite / ONNX Runtime；
- 视觉模型、小模型、本地大模型。

云端推理常见于：

- 多机多卡集群；
- vLLM / SGLang；
- 高并发 LLM API；
- 多租户模型服务；
- KV Cache 管理；
- 长上下文推理。

### 相关概念解释

端侧场景通常算力有限，例如车载芯片、机器人平台、边缘 GPU 等。它更关注：

- 模型压缩；
- TensorRT 部署；
- INT8 / FP16 量化；
- 单卡或小规模多卡推理；
- 延迟稳定性；
- 功耗限制；
- 与传感器、控制系统集成。

云端场景更关注：

- 批处理；
- 请求调度；
- Prefix Cache；
- PagedAttention；
- 分布式推理；
- 多机多卡通信；
- 吞吐 / TTFT / TPOT。

### 在 AI Infra 中的作用

二者都属于 AI Infra，但侧重点不同：

| 维度 | 端侧推理 | 云端 LLM 推理 |
|---|---|---|
| 硬件 | Jetson、车载 NPU、边缘 GPU | A100/H100/A800/H800 等 GPU 集群 |
| 模型 | CV、小模型、轻量 LLM | 大语言模型、多模态大模型 |
| 优化目标 | 低延迟、低功耗、稳定部署 | 高吞吐、低 TTFT、显存利用率 |
| 框架 | TensorRT、ONNX Runtime、NCNN、MNN | vLLM、SGLang、TensorRT-LLM |
| 项目表达 | 部署、量化、算子插件、前后处理优化 | 调度、KV Cache、Batching、Serving |

---

## 4.6 第六段：K8S / Go 推理平台项目和 AI Infra 的关系

### 核心知识点

有同学提到之前做过 K8S + Go 的推理平台。主讲人判断这类项目更像推理平台运维或 MLOps / DevOps，不等同于大模型推理框架开发、CUDA 算子开发、AI 编译器。

### 相关概念解释

K8S 推理平台通常涉及：

- 模型服务容器化；
- 服务编排；
- 负载均衡；
- 监控；
- 自动扩缩容；
- GPU 资源调度；
- HTTP/gRPC API；
- CI/CD。

它更偏平台层，而非模型执行引擎层。

### 在 AI Infra 中的作用

K8S 项目不是没用，但它要和目标岗位匹配：

- 如果投 MLOps / AI 平台工程，K8S + Go 很有价值；
- 如果投 vLLM 推理框架，项目相关性较弱；
- 如果投 CUDA 算子，项目基本不直接相关；
- 如果投推理服务工程，可以作为工程能力补充，但不能替代推理引擎项目。

---

## 4.7 第七段：HPC 背景如何转 AI Infra

### 核心知识点

第二位同学有 HPC / 科研算子优化背景。主讲人认为，这类项目和 AI 算子开发有很强迁移性，因为它们解决的是同一类问题：

> **针对特定计算任务，分析性能瓶颈，然后通过并行化、访存优化、减少冗余计算等方式提升性能。**

### 相关概念解释

HPC 项目常见优化方向：

- 并行化；
- 线程组织；
- 内存访问优化；
- 减少重复计算；
- 降低通信开销；
- 负载均衡；
- 算法级剪枝或稀疏化；
- 使用 CUDA / OpenMP / MPI 等。

AI 算子开发常见优化方向：

- tiling；
- shared memory；
- Tensor Core；
- fusion；
- memory coalescing；
- reduce kernel；
- softmax kernel；
- GEMM 优化；
- attention kernel。

两者方法论高度相似。

### 在 AI Infra 中的作用

HPC 背景可以转化为 AI Infra 的优势，但简历表达必须调整：

不要只写：

> 优化某科研算法，提升性能。

应该写清楚：

- 任务背景是什么；
- 计算瓶颈在哪里；
- 用什么编程模型实现；
- 是否使用 CUDA；
- 哪些 kernel 做了优化；
- 优化前后性能变化；
- 性能指标是什么；
- 对 AI 算子开发有什么迁移价值。

---

## 4.8 第八段：算子开发 vs 推理框架开发

### 核心知识点

课程中讨论了算子开发和推理框架开发哪个前景更广。

主讲人的判断是：

- 算子开发更专精；
- 推理框架覆盖面更广；
- 两者都是端到端推理优化链路中的一环；
- 不存在绝对谁更有前景，更多取决于个人兴趣、项目经历和岗位机会。

### 相关概念解释

算子开发通常关注：

- 某一类 kernel；
- 对业务模型的新算子支持；
- 针对不同 shape / dtype / GPU 架构调优；
- 和 PyTorch / Triton / CUDA / CUTLASS 对齐；
- micro benchmark 和 end-to-end benchmark。

推理框架开发通常关注：

- API server；
- 请求调度；
- KV Cache 管理；
- batching；
- streaming；
- model runner；
- worker；
- 分布式执行；
- 量化支持；
- speculative decoding；
- prefix caching。

### 在 AI Infra 中的作用

可以把端到端 LLM 推理拆成：

```text
用户请求
  → 服务入口
  → 请求调度
  → KV Cache 分配
  → 模型执行
  → CUDA/Triton 算子
  → token 采样
  → 流式输出
```

算子主要负责“模型执行”中的底层计算。推理框架负责把请求、资源和计算组织起来。

---

## 4.9 第九段：CUTLASS 是否值得补

### 核心知识点

对于已有 CUDA / HPC 背景并且未来想做算子开发的同学，主讲人建议补 CUTLASS。

### 相关概念解释

CUTLASS 是 NVIDIA 提供的 CUDA C++ 模板库，用于高性能矩阵计算。它封装了 GEMM、卷积、Tensor Core、tiling、pipeline 等高性能计算模式。

学习 CUTLASS 的价值在于：

- 理解工业级 GEMM 的分层结构；
- 理解 threadblock / warp / instruction 级别的矩阵计算；
- 理解 Tensor Core 使用方式；
- 理解高性能 kernel 的模板化实现；
- 为写复杂算子打基础。

### 在 AI Infra 中的作用

大模型里的绝大多数计算最终都与 GEMM 相关：

- QKV projection；
- attention output projection；
- MLP up/down/gate projection；
- lm_head；
- MoE expert GEMM。

因此 CUTLASS 对算子开发非常有价值。

---

## 4.10 第十段：nano-vLLM 项目是否过于常见

### 核心知识点

第三位同学做了 nano-vLLM 项目。主讲人承认：nano-vLLM 很常见，很多同学都会写。但关键区别在于：

- 大多数人只是看了一遍；
- 少数人会真正添加功能或解决问题；
- 如果你有自己的优化点，项目仍然有价值。

### 相关概念解释

nano-vLLM 适合作为 vLLM 入门项目，因为它简化了：

- 请求对象；
- 调度器；
- KV Cache 管理；
- block manager；
- model runner；
- prefill / decode；
- token sampling。

但如果只是“读源码”，简历辨识度不够。需要加入自己的工作，例如：

- KV Cache 压缩；
- Prefix Cache；
- Chunked Prefill；
- 简化版 PagedAttention；
- benchmark 统计；
- 调度策略改进；
- 支持新模型；
- 支持量化或 KV Cache 量化。

### 在 AI Infra 中的作用

nano-vLLM 的作用是帮你建立端到端推理框架认知。它不是最终工业项目，但能帮助你理解 vLLM 的主干逻辑。

---

## 4.11 第十一段：KV Cache 压缩优化如何讲

### 核心知识点

第三位同学在 nano-vLLM 中做了 KV Cache 压缩。思路是：当 prompt 很长、显存不够时，不是直接撤回请求重新 prefill，而是对历史窗口中的 KV Cache 进行打分，保留 Top-K 重要 block，淘汰低分 block。

### 相关概念解释

KV Cache 是自回归 LLM 推理中保存历史 token 的 Key / Value 缓存。长上下文会导致 KV Cache 显存占用巨大。

KV Cache 压缩的核心问题是：

> 哪些历史 token 对后续生成更重要？哪些可以被丢弃或近似？

一种简单思路是基于当前 query 与历史 key 的相关性打分，保留得分高的历史 KV，淘汰得分低的 KV。

### 在 AI Infra 中的作用

KV Cache 压缩可以缓解：

- 长上下文显存压力；
- block 不足导致请求被抢占；
- 长 prompt 的 prefill 成本；
- 多请求并发时的显存占用。

但它也会引入问题：

- 精度是否下降；
- 压缩策略是否稳定；
- 压缩本身是否带来额外开销；
- 需要如何 benchmark；
- 对 TTFT / TPOT / throughput 的影响。

---

## 4.12 第十二段：YOLO / Jetson / 相机部署项目为什么容易显得浅

### 核心知识点

课程中指出，YOLO 部署、Jetson 平台、相机拉流、TensorRT 推理这类项目很常见，容易被认为偏部署应用，而不是 AI Infra 深度项目。

原因是：

- 模型结构成熟；
- 工具链成熟；
- PyTorch → ONNX → TensorRT 可一键转换；
- 如果没有算子、前处理、插件或性能分析，工作量不够深入。

### 相关概念解释

这类项目通常包括：

```text
摄像头采集
  → 解码 / 拉流
  → resize / normalize / letterbox
  → TensorRT 推理
  → NMS 后处理
  → 结构化输出
```

如果只是跑通流程，偏工程集成。

要提升深度，可以做：

1. 前处理 CUDA 化；
2. 自定义 TensorRT Plugin；
3. NMS CUDA 优化；
4. Batch 推理优化；
5. FP16 / INT8 对比；
6. Nsight 分析；
7. 端到端延迟拆解；
8. GPU / CPU 负载对比；
9. 多路流调度。

### 在 AI Infra 中的作用

部署项目可以证明工程落地能力，但要想匹配 AI Infra，必须体现“性能优化”和“底层理解”。

---

## 4.13 第十三段：面试官会怎么问 nano-vLLM 项目

### 核心知识点

对于做了 KV Cache 压缩的项目，面试官可能重点问：

- 压缩策略具体是什么；
- 如何打分；
- 为什么这样打分合理；
- 压缩对精度的影响；
- 压缩对显存的影响；
- 是否影响 TTFT / TPOT；
- 是否与 PagedAttention / block manager 冲突；
- 如何处理被淘汰 block；
- 长上下文场景下是否稳定。

### 相关概念解释

面试官关心的不只是“你加了功能”，而是：

- 你是否真的理解系统瓶颈；
- 你是否有实验数据支撑；
- 你是否考虑过副作用；
- 你是否知道该优化在真实系统中的位置。

### 在 AI Infra 中的作用

项目表达要从“我实现了一个功能”升级到：

> 我发现长上下文下 KV Cache 显存占用大、block 不足会导致请求抢占，于是设计了基于 query 相关性的 block 级压缩策略，在保留关键上下文的同时降低显存占用，并通过 benchmark 对比压缩前后的吞吐、延迟和输出质量。

---

## 4.14 第十四段：Nsight Systems 写在简历上会被问什么

### 核心知识点

如果简历里写“熟悉 Nsight Systems / Nsight Compute”，面试官会问你具体怎么用过。

### 相关概念解释

Nsight Systems 更偏系统级 profiling，能看：

- CPU / GPU 时间线；
- kernel launch；
- CUDA memcpy；
- stream 并发；
- CPU 调度开销；
- GPU idle；
- 多线程调用关系。

Nsight Compute 更偏单 kernel profiling，能看：

- memory throughput；
- occupancy；
- warp stall；
- global load/store efficiency；
- shared memory bank conflict；
- L2 cache hit rate；
- Tensor Core utilization。

### 在 AI Infra 中的作用

性能优化必须有 profile 数据。否则简历里的“优化”容易变成空话。

更好的项目表达是：

> 使用 Nsight Systems 定位端到端流程中 CPU 预处理与 GPU kernel launch 开销，使用 Nsight Compute 分析自定义 CUDA kernel 的访存效率与 warp stall，针对瓶颈做 shared memory tiling 和访存合并优化。

---

## 4.15 第十五段：学历和无实习如何弥补

### 核心知识点

课程最后讨论了学历和无实习的问题。主讲人指出：

- 学历背景无法再改变；
- 只能通过技术深度、项目质量、面试表现和实习经历弥补；
- 现在最重要的是尽快投递、积累面试反馈；
- 不要只在学校里闭门造车。

### 相关概念解释

简历筛选通常看：

- 学校；
- 实习；
- 项目；
- 技术栈；
- 开源贡献；
- 论文 / 比赛；
- 岗位匹配度。

如果学校和实习弱，就要增强：

- 项目真实性；
- 项目数据；
- 面试可讲深度；
- 代码能力；
- CUDA 手撕；
- 推理框架八股；
- 广投和面试经验。

### 在 AI Infra 中的作用

AI Infra 是工程实践很强的方向。很多能力不是看课能完全获得，必须通过：

- 做项目；
- 跑 benchmark；
- 读源码；
- 调试；
- 面试；
- 修改简历；
- 再投递。

---

# 5. 核心技术点深度解释

## 5.1 CUDA 算子学习的正确目标

不要把目标设成“所有算子都写到极致”。更合理的目标是：

1. 能手写基础版本；
2. 能解释瓶颈；
3. 能说出优化路径；
4. 对 3–5 个高频算子准备熟练版本；
5. 能结合 profiling 数据讲优化效果。

建议优先准备：

| 优先级 | 算子 | 面试价值 |
|---|---|---|
| P0 | Reduce | 简单但高频，考察 shared memory / warp reduce |
| P0 | Softmax | LLM 高频算子，考察 reduce + 数值稳定 |
| P0 | MatMul | GEMM 是核心，考察 tiling / shared memory |
| P1 | Transpose | 考察合并访存、bank conflict |
| P1 | LayerNorm / RMSNorm | LLM 高频小算子 |
| P2 | Attention 简化版 | 面试可能问原理，不一定手撕完整 |
| P2 | NMS | CV / 部署岗位可能问 |
| P2 | TopK | 部分岗位可能问 |

---

## 5.2 算子项目怎么写简历

低质量写法：

> 使用 CUDA 优化某算法，提升性能。

高质量写法：

> 针对 XXX 算法中 XXX 阶段计算密集、访存不连续的问题，使用 CUDA 实现核心 kernel，并通过 shared memory tiling、合并访存、减少线程分支和避免重复计算等方式优化；使用 Nsight Compute 分析 memory throughput、warp stall 和 occupancy，最终在输入规模 XXX 下相较 CPU baseline 提升 X 倍，相较 naive CUDA 提升 X%。

关键是要有：

- 背景；
- 瓶颈；
- 方法；
- 工具；
- 数据；
- 对比对象。

---

## 5.3 推理框架项目怎么写简历

低质量写法：

> 阅读 nano-vLLM 源码，理解 PagedAttention 和 KV Cache。

高质量写法：

> 基于 nano-vLLM 实现长上下文场景下的 KV Cache block 压缩策略：在 block manager 中引入基于当前 query 与历史 key 相关性的 Top-K block 保留机制，降低长 prompt 下 KV Cache 显存占用；补充 benchmark 脚本统计压缩前后的显存占用、TTFT、TPOT 和生成质量变化，并分析压缩阈值对吞吐和输出稳定性的影响。

关键是要体现：

- 你修改了哪个模块；
- 为什么改；
- 怎么改；
- 有什么实验；
- 有什么 trade-off。

---

## 5.4 端侧部署项目怎么升级成 AI Infra 项目

普通端侧部署：

```text
YOLO → ONNX → TensorRT → Jetson → 摄像头推理
```

升级为 AI Infra 项目：

```text
多路视频流
  → CUDA 前处理
  → TensorRT FP16/INT8 engine
  → 自定义 Plugin / NMS kernel
  → batch 调度
  → Nsight profiling
  → latency / throughput / GPU utilization 对比
```

简历要强调：

- 端到端延迟拆解；
- 前处理和后处理占比；
- 自定义 CUDA / TensorRT Plugin；
- FP16 / INT8 量化对比；
- batch size 与吞吐关系；
- 多路流调度策略；
- 是否满足实时帧率要求。

---

## 5.5 AI 编程工具不是替代工程师，而是改变工作流

课程中隐含提到 CodeX / Claude Code 等工具。关键判断是：

- AI 可以生成基础代码；
- AI 可以辅助实现算子或框架 feature；
- 但工程师仍然要做需求拆解、性能判断、profiling、实验设计、验收和维护。

算子或框架开发的真实流程更像：

```text
需求拆解
  → 判断缺哪些算子 / feature
  → 设计实现方案
  → AI 辅助写初版
  → 跑通 correctness
  → profiling
  → 判断优化方向
  → 迭代优化
  → benchmark
  → 合入与维护
```

所以不会因为 AI 能写 kernel，就不需要算子工程师。真正的壁垒在于：

- 知道该写什么；
- 知道为什么慢；
- 知道怎么验证；
- 知道怎么和系统集成；
- 知道怎么维护线上正确性和性能。

---

# 6. 整节课的“从输入到输出”技术主线

这节课虽然是职业规划课，但可以串成一条非常清晰的 AI Infra 求职技术主线。

## 6.1 用户输入阶段：从岗位目标开始

你首先要明确自己投的是哪类岗位：

```text
算子开发
推理框架开发
端侧模型部署
AI 平台 / MLOps
AI 编译器
```

不同岗位要求不同：

- 算子开发：CUDA、C++、profiling、GEMM、Softmax、Reduce；
- 推理框架：vLLM、调度、KV Cache、PagedAttention、服务化；
- 端侧部署：TensorRT、ONNX、量化、Jetson、NPU；
- 平台工程：K8S、Go、服务治理、资源调度；
- 编译器：IR、Pass、图优化、MLIR、TVM。

## 6.2 数据进入阶段：项目如何进入推理系统

以一个真实推理请求为例：

```text
用户输入文本 / 图像 / 视频流
  → 预处理
  → tokenizer / image processor
  → 推理框架接收请求
  → scheduler 排队
  → KV Cache / 显存资源分配
  → 模型 forward
  → CUDA/Triton kernel 执行
  → 采样 / 后处理
  → 输出 token / 检测框 / 结构化结果
```

你的项目应该能落到其中至少一个关键环节。

## 6.3 计算执行阶段：算子是模型执行的核心

模型 forward 内部会调用大量算子：

```text
Embedding
  → QKV GEMM
  → Attention
  → Softmax
  → Output GEMM
  → RMSNorm
  → MLP GEMM
  → Sampling
```

算子开发负责让这些 kernel 更快、更省显存。

## 6.4 资源管理阶段：推理框架负责组织请求和显存

大模型推理框架负责：

- 多请求调度；
- batch 动态更新；
- KV Cache 分配；
- block 回收；
- prefix 复用；
- prefill / decode 切换；
- 流式输出。

nano-vLLM / vLLM 的学习价值就在这里。

## 6.5 性能反馈阶段：Profiling 决定优化方向

不能凭感觉优化，要通过：

- Nsight Systems；
- Nsight Compute；
- nvidia-smi；
- benchmark 脚本；
- TTFT / TPOT / throughput；
- latency breakdown；
- memory usage；
- GPU utilization。

找到瓶颈后再优化。

## 6.6 求职输出阶段：项目必须能被面试官追问

最终输出不是代码本身，而是简历和面试表达：

```text
项目背景
  → 发现瓶颈
  → 设计方案
  → 修改模块
  → 性能数据
  → trade-off
  → 面试可复述
```

这就是从“学习”转化为“求职竞争力”的完整流程。

---

# 7. 结合 AI Infra 求职的路线建议

## 7.1 如果你偏 CUDA / 算子开发

优先级如下：

1. CUDA 基础；
2. Reduce / Softmax / MatMul / Transpose；
3. Nsight Compute；
4. CUTLASS 入门；
5. Triton 基础；
6. 项目中补充性能数据；
7. LeetCode 中等题；
8. C++ 八股。

项目要体现：

- kernel 实现；
- 优化策略；
- profile 数据；
- CPU / naive CUDA / optimized CUDA 对比；
- shape / dtype / batch size 变化。

## 7.2 如果你偏推理框架

优先级如下：

1. Transformer / KV Cache / Prefill / Decode；
2. nano-vLLM；
3. vLLM 核心机制；
4. PagedAttention；
5. Continuous Batching；
6. Prefix Caching；
7. Scheduler；
8. Worker / ModelRunner；
9. Benchmark；
10. CUDA 基础算子。

项目要体现：

- 请求生命周期；
- 调度逻辑；
- KV Cache 管理；
- block manager；
- 你加的 feature；
- benchmark 数据。

## 7.3 如果你偏端侧推理 / 机器人 / 车企

优先级如下：

1. ONNX；
2. TensorRT；
3. FP16 / INT8；
4. Jetson / NPU 平台；
5. CUDA 前后处理；
6. 自定义 Plugin；
7. 多路流调度；
8. 端到端延迟拆解；
9. Linux / C++ / Python 工程能力。

项目要体现：

- 实时性；
- 功耗 / 算力限制；
- 延迟；
- 帧率；
- batch；
- 多模型部署；
- 硬件资源占用。

---

# 8. 面试高频问题与参考答案

## Q1：AI Infra 推理岗位必须学 Agent 吗？

**参考答案：**

不必须。Agent 更多属于大模型应用层，核心是工具调用、任务规划、多轮交互和外部系统协作。AI Infra 推理岗位更关注模型如何高效运行，包括 vLLM / SGLang、KV Cache、调度、Batching、CUDA 算子、显存管理和吞吐优化。不过了解 Agent 的请求模式有帮助，因为 Agent 往往有长上下文、多轮对话、重复前缀和工具调用，这些会影响推理框架对 Prefix Cache、长上下文优化和流式输出的设计。

---

## Q2：CS336 对 AI Infra 求职有必要吗？

**参考答案：**

可以学，但不是替代工程项目的核心材料。CS336 适合作为大模型通识课，帮助理解 Transformer、训练、推理、模型结构和系统问题。对于 AI Infra 求职，更关键的是把课程知识落实到项目中，比如跑通 vLLM、理解 KV Cache、做 benchmark、实现 CUDA 算子或修改推理框架功能。

---

## Q3：CUDA 算子面试要写到什么程度？

**参考答案：**

一般不要求现场写出极致优化版本，但要能写基础版本，并说明优化路径。比如 Reduce、Softmax、Transpose、MatMul 这些高频算子，要能先写 naive 版本，再解释如何用 shared memory、tiling、合并访存、避免 bank conflict、减少 warp divergence 等方式优化。对 MatMul 这种复杂算子，可以准备一个熟练版本，不一定覆盖最先进 Tensor Core 优化，但基本 tiling 和 shared memory 要会。

---

## Q4：算子开发和推理框架开发哪个更有前景？

**参考答案：**

两者都是端到端推理优化链路的重要环节。算子开发更专精，主要优化某类 kernel，如 GEMM、Attention、MoE、Norm；推理框架更广，涉及请求调度、KV Cache、Batching、Serving、分布式、模型执行等。不能简单说谁更有前景，关键看个人兴趣和项目匹配。算子岗技术深、面试硬；框架岗系统面广、工程复杂度高。

---

## Q5：HPC 项目如何包装成 AI Infra 项目？

**参考答案：**

HPC 项目和 AI 算子开发的方法论相通，都是分析瓶颈并做并行优化。包装时要讲清任务背景、计算瓶颈、使用的并行编程模型、CUDA 实现细节、优化策略和性能数据。最好明确写出使用 CUDA，实现了哪些 kernel，如何通过 shared memory、减少重复计算、改善访存或降低线程发散提升性能。

---

## Q6：CUTLASS 值得学吗？

**参考答案：**

如果目标是算子开发，CUTLASS 值得学。它是 NVIDIA 的高性能线性代数模板库，能帮助理解 GEMM 的分层实现，包括 threadblock、warp、instruction 级别的 tiling 和 Tensor Core 使用。大模型中 Attention、MLP、MoE 都大量依赖 GEMM，因此 CUTLASS 对算子开发非常有价值。

---

## Q7：nano-vLLM 项目是不是太常见？

**参考答案：**

只读源码确实常见，辨识度不够。但如果在 nano-vLLM 上加入自己的功能，比如 KV Cache 压缩、Prefix Cache、Chunked Prefill、benchmark、调度优化、量化支持或新模型适配，就会有项目价值。面试官看重的是你是否真的理解端到端流程，并有自己的工作量和实验数据。

---

## Q8：KV Cache 压缩优化怎么讲？

**参考答案：**

可以从问题出发：长上下文请求会占用大量 KV Cache，导致显存压力和 block 不足。我的策略是在 block manager 中对历史 KV block 进行重要性评估，根据当前 query 与历史 key 的相关性打分，保留 Top-K 重要 block，淘汰低分 block，从而降低显存占用。实验上需要对比压缩前后的显存占用、TTFT、TPOT、throughput 和生成质量，说明压缩比例和质量损失之间的 trade-off。

---

## Q9：YOLO + TensorRT + Jetson 项目为什么容易被认为浅？

**参考答案：**

因为这类部署流程工具链已经很成熟，很多工作可以通过 PyTorch → ONNX → TensorRT 一键完成。如果只是跑通摄像头输入和模型推理，更多是工程集成。要提升为 AI Infra 项目，需要加入 CUDA 前处理、自定义 TensorRT Plugin、NMS 优化、量化对比、batch 调度、端到端 latency breakdown 和 Nsight profiling 等工作。

---

## Q10：Nsight Systems 和 Nsight Compute 有什么区别？

**参考答案：**

Nsight Systems 是系统级 profiler，主要看 CPU/GPU 时间线、kernel launch、CUDA memcpy、stream 并发和 GPU idle，用于定位端到端流程瓶颈。Nsight Compute 是 kernel 级 profiler，主要看单个 CUDA kernel 的 memory throughput、occupancy、warp stall、bank conflict、L2 cache hit rate 等指标，用于分析具体 kernel 为什么慢。

---

## Q11：学历不占优势怎么弥补？

**参考答案：**

学历背景无法改变，只能用技术深度、项目质量、面试表现和实习经历弥补。具体做法是：项目必须真实、有数据、有源码修改、有可讲的技术细节；尽快投递中小厂和相关实习，通过面试反馈发现短板；同时补 CUDA 手撕、推理框架八股、C++/数据结构等基础。

---

## Q12：为什么要尽快从 YOLO / CV 部署转向 LLM 推理？

**参考答案：**

如果目标是大模型推理框架或 AI Infra 推理侧，长期停留在 YOLO / CV 部署会和目标岗位不够匹配。CV 部署可以作为工程能力补充，但应尽快转入 LLM 推理的核心知识，包括 Transformer、KV Cache、Prefill/Decode、PagedAttention、Continuous Batching、vLLM / nano-vLLM、LLM benchmark 等。

---

# 9. 可直接执行的学习与项目打磨清单

## 9.1 两周内必须补齐

- [ ] CUDA thread / block / grid / warp；
- [ ] global memory / shared memory / register；
- [ ] Reduce 手写；
- [ ] Softmax 手写；
- [ ] Transpose 手写；
- [ ] MatMul shared memory tiling；
- [ ] Nsight Systems 基础；
- [ ] Nsight Compute 基础；
- [ ] Transformer Decoder-only 流程；
- [ ] KV Cache / Prefill / Decode。

## 9.2 一个月内完成

- [ ] 完整复盘 nano-vLLM；
- [ ] 画出请求从输入到输出的流程图；
- [ ] 写 benchmark 脚本统计 TTFT / TPOT / throughput；
- [ ] 为 nano-vLLM 增加一个真实 feature；
- [ ] 为项目补充实验数据；
- [ ] 简历中删除无法讲清楚的内容；
- [ ] 每个项目准备 5 个面试追问答案。

## 9.3 项目修改建议

### nano-vLLM 项目

需要补充：

- 压缩策略伪代码；
- block manager 修改点；
- 显存占用对比；
- TTFT / TPOT 对比；
- 生成质量变化；
- 压缩比例对性能影响；
- trade-off 分析。

### CUDA / HPC 项目

需要补充：

- 使用 CUDA 明确说明；
- 输入规模；
- CPU baseline；
- naive CUDA baseline；
- optimized CUDA 性能；
- Nsight 指标；
- 优化前后数据；
- bottleneck 分析。

### YOLO / TensorRT 项目

若保留，需要补充：

- CUDA 前处理；
- NMS 优化；
- TensorRT Plugin；
- FP16 / INT8 对比；
- 多路流；
- batch 推理；
- 端到端延迟拆解。

---

# 10. 最终总结

本节课的核心价值在于：它没有停留在“AI Infra 学什么”的抽象层面，而是结合具体同学的背景，说明了如何把不同经历转化成可求职的能力。

对于有 Linux / 系统背景的同学，AI Infra 是自然延伸，但不要把 K8S / Go 平台项目误认为推理框架项目。

对于有 HPC 背景的同学，算子优化能力可以迁移到 AI Infra，但必须在简历中明确 CUDA、性能瓶颈和优化数据。

对于做 nano-vLLM 的同学，项目常见不是问题，问题是有没有自己的真实 feature 和实验支撑。

对于做 YOLO / Jetson / TensorRT 部署的同学，单纯部署偏浅，必须加入 CUDA 前后处理、插件、profiling 和端到端性能分析。

最终求职策略可以概括为：

```text
明确目标岗位
  → 补齐岗位核心能力
  → 打磨一个能讲深的项目
  → 用数据证明优化效果
  → 广投积累面试反馈
  → 反复修改简历和项目表达
```

AI Infra 的门槛不在于背几个名词，而在于你能否真正解释：

- 为什么慢；
- 慢在哪里；
- 如何测；
- 如何改；
- 改完效果如何；
- 对真实推理系统有什么意义。

这也是后续学习和秋招准备最应该围绕的主线。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-AI infra学习路线|模块-AI infra学习路线]]

%% 项目关联导航：结束 %%
