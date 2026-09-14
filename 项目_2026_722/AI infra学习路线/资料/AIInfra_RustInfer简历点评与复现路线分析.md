# AI Infra 简历点评视频分析报告：RustInfer 推理框架项目复盘与个人复现路线

> 说明：本文基于你上传的视频转文字文件、两张简历截图和分析提示词整理。由于截图局部存在模糊，凡是无法确认的内容均以“截图不清 / 疑似 / 待补充”标注，不把推断内容伪装成原简历内容。

---

# 任务 1：复现简历内容

## 1.1 教育背景

```text
教育背景
双非本硕深圳 c9 硕                     计算机科学与技术
研究方向：AI for Science
```

说明：

- “深圳 c9 硕”是根据截图可见文字复现，具体学校名称截图未展示。
- “双非本硕深圳 c9 硕”这句话本身表述有些不规范，可能原意是“本科双非，硕士为深圳某 C9 高校/研究院相关项目”，需要原简历进一步确认。

---

## 1.2 专业技能

```text
专业技能

- 推理框架：
  熟悉 sglang / vllm 主流推理框架，掌握 Rust / C++ 混合异步编程；
  深入理解 LLM 推理架构（PagedAttention / Continuous Batching / RadixAttention）；
  掌握 CUDA Graph 机制与显存池化技术。

- 高性能计算：
  熟悉编写 CUDA Kernel，能使用 C++ / CuTe 开发高性能 CUDA 算子；
  熟悉算子融合技术；掌握 nsight systems / nsight compute 性能分析与瓶颈定位；
  了解不同架构显卡的差异，并能根据其特性设计高性能算子；
  能写 BF16 混合精度算子。

- 算法与框架：
  掌握 PyTorch 及其底层机制（Dispatcher, Autograd）；
  熟悉常用架构（Transformer, Diffusion, UNet, KAN）；
  了解 PINNs（物理信息神经网络）；
  熟练使用 BF16 / FP16 混合精度训练、推理与数据存储。

- 工程能力：
  具备全栈开发能力，熟悉 WASM 前端与 Axum 后端开发；
  掌握 Docker 容器化部署与 Git 工作流。
```

简要判断：

这份专业技能覆盖面非常广，从推理框架、CUDA、PyTorch 底层、AI for Science 到 WASM 全栈都有涉及。优点是“强综合能力标签”明显；风险是如果面试时任何一项都讲不深，容易被认为“堆关键词”。因此真正核心应收敛到：

```text
Rust / C++ 推理框架 + KV Cache / 调度 + CUDA Graph / 算子 + vLLM/SGLang 对照理解
```

---

## 1.3 项目经历一：基于 Rust 高性能推理框架

```text
项目经历

基于 Rust 高性能推理框架
独立开发者 | Infra 方向                         2025 年 08 月 - 至今

项目地址：
https://github.com/Vinci-hit/RustInfer

项目背景：
在学习主流 Python 推理系统（vLLM / SGLang）的过程中发现都有 GIL 瓶颈，
利用科研业务时间，从零构建的基于 Rust + CUDA 的异步推理框架。
```

### 主要工作内容复现

```text
- Rust 异步调度架构：
  基于 Async 和 Actor 模型构建推理调度器，实现了 Scheduler、BlockManager
  与 Model Worker 的完全解耦；设计了高并发任务队列，为支持 Continuous
  Batching 打下高扩展性基础。

- 显存管理与 RadixAttention：
  在 Rust 中复现 RadixTree 前缀树缓存机制，实现基于 LRU 引用计数的 Block
  自动回收；设计了 PagedAttention 风格的非连续显存池，有效解决显存碎片化
  问题，大幅提升 Decode 阶段的显存带宽利用率。

- 高性能算子与 CUDA Graph：
  实现 Decoding 阶段中 CUDA Graph 的自动录制与重放，减少 Kernel Launch
  的 CPU 开销，性能提高 10%。

- 混合精度：
  全链路支持 BF16 混合精度计算，实现 CuTe 优化 FlashAttention 算子性能；
  在 Batch Size = 1 的低延迟场景下，得益于 Rust 的零运行时开销，端到端
  延迟与 vLLM 持平或更优。

- 全栈可观测性与接口：
  实现了基于 WASM 的高性能可视化前端，支持 Token 生成流的实时渲染与公式显示；
  开发了兼容 OpenAI API 协议的 Rust Server，通过 ZeroMQ 实现计算与服务层的
  高效通信，具备完整的全链路监控能力。
```

### 截图与字幕互相印证的信息

视频转文字中，主讲人重点追问了这个框架是否独立完成、是否参考其他项目。被点评同学回答最开始参考了某个项目，后续功能不断新增。主讲人认为“自己琢磨出这么一套东西来真的很不错”，并建议继续完善。

主讲人还指出，PagedAttention 如果是在已有推理框架基础上后接入，会非常难，因为它会影响算子、调度层和设计模式。也就是说，这个项目真正的难点不只是“写一个 Rust 服务”，而是要把 KV Cache、BlockManager、Scheduler、CUDA Graph 和推理执行链路真正串起来。

---

## 1.4 项目经历二：某 AI 算法方向科研课题

```text
项目经历

某 AI 算法方向的科研课题
核心研究员 | 算法方向                         2025 年 08 月 - 至今

截图可见补充：
做得好的话未来有机会工程落地
```

说明：

- 该项目截图只显示了标题、角色、方向和一句简略说明。
- 未看到具体研究问题、模型结构、数据集、实验指标、论文/专利/工程落地情况。
- 因此不能凭空补充项目内容。

可确认定位：

```text
这是一个 AI for Science / 算法科研方向项目，和推理框架项目相比，
更偏算法应用或科研落地，不是纯 AI Infra 项目。
```

---

# 任务 2：对项目进行详细分析

# 2.1 项目一：基于 Rust 高性能推理框架

## 2.1.1 项目定位

这个项目本质上是：

```text
大模型推理引擎 / LLM Serving Runtime / 高性能推理框架自研项目
```

它不是普通“模型部署 Demo”，也不是简单“调用 vLLM API”。它试图从框架层复现和改造 vLLM / SGLang 的核心能力，包括：

- 请求调度；
- Scheduler；
- BlockManager；
- Model Worker；
- KV Cache block 管理；
- RadixTree / Prefix Cache；
- PagedAttention 风格显存池；
- Continuous Batching；
- CUDA Graph；
- OpenAI-compatible Server；
- Token 流式输出；
- 性能可观测。

如果内容真实、代码可运行、指标可复现，这个项目在校招 AI Infra / 推理引擎方向属于非常高含金量项目。

---

## 2.1.2 属于 AI Infra 哪一层

| 项目 | AI Infra 层级 | 判断依据 | 与目标岗位的相关性 |
|---|---|---|---|
| 基于 Rust 高性能推理框架 | 推理框架层 / Runtime 层 / 部分算子层 | 包含调度、KV Cache、BlockManager、Model Worker、CUDA Graph、Server API | 极高，直接对应大模型推理引擎、AI Runtime、LLM Serving 岗位 |
| CUDA Graph 与 CuTe FlashAttention | 算子与 GPU 执行优化层 | 涉及 Kernel Launch overhead、BF16、CuTe、FlashAttention | 很高，对算子岗和推理框架底层岗都有价值 |
| WASM 前端与 OpenAI API Server | 模型服务层 / 工程展示层 | 提供可视化、OpenAI 协议兼容、流式输出 | 中高，增强工程闭环和可展示性 |
| ZeroMQ 计算与服务层通信 | 系统工程 / 进程间通信层 | 将服务层与计算层解耦 | 中高，可体现系统设计能力 |

---

## 2.1.3 技术栈拆解

| 技术栈 | 在项目中的作用 | AI Infra 价值 | 面试可能问题 | 学习优先级 |
|---|---|---|---|---|
| Rust Async | 构建异步服务和调度框架 | 体现高并发系统能力 | Rust async runtime 原理？Future 如何调度？tokio 和 actor 模型关系？ | 高 |
| Actor 模型 | 解耦 Scheduler、BlockManager、Model Worker | 对复杂推理系统模块化有价值 | 为什么用 Actor？和多线程队列相比有什么优劣？ | 高 |
| Scheduler | 管理请求进入、执行和退出 | 推理框架核心 | Prefill / Decode 如何调度？Continuous Batching 怎么实现？ | 极高 |
| BlockManager | 管理 KV Cache block 分配回收 | vLLM 核心机制之一 | block size 怎么定？如何处理碎片？如何回收？ | 极高 |
| RadixTree / Prefix Cache | 复用相同 prompt 前缀 | SGLang / LLM serving 高频考点 | 前缀复用如何降低 TTFT？LRU 和引用计数如何结合？ | 高 |
| PagedAttention 风格显存池 | 管理非连续 KV Cache | 解决显存碎片和并发瓶颈 | PagedAttention 为什么像操作系统分页？和连续 KV Cache 有何区别？ | 极高 |
| CUDA Graph | 降低 decode 阶段 kernel launch 开销 | 低延迟推理优化重点 | CUDA Graph 适合什么 shape？动态图怎么复用 graph？ | 高 |
| CuTe / FlashAttention | 优化 attention kernel | 算子开发核心能力 | FlashAttention 如何减少 HBM 访问？CuTe 的抽象是什么？ | 中高 |
| BF16 / FP16 | 混合精度推理 | 降低显存、提高吞吐 | BF16 和 FP16 区别？为什么 BF16 训练更稳？ | 高 |
| OpenAI-compatible API | 对外服务协议 | 工程落地能力 | 如何实现 stream response？如何兼容 chat/completions？ | 中 |
| ZeroMQ | 服务层与计算层通信 | 系统解耦和跨语言通信 | 为什么不用 HTTP/gRPC？ZeroMQ 模式如何选？ | 中 |
| WASM 前端 | 可视化展示 token 流 | 提升项目可展示性 | WASM 为什么高性能？和普通前端有什么区别？ | 低到中 |

---

## 2.1.4 项目含金量评价

| 维度 | 分数 | 评价 |
|---|---:|---|
| AI Infra 相关性 | 5 | 直接命中大模型推理框架、Serving、Runtime、KV Cache、调度、CUDA Graph 等核心方向 |
| 技术深度 | 5 | 如果真实实现，已超过普通校招项目，涉及框架设计、异步系统、显存管理和 GPU 执行优化 |
| 工程完整性 | 4.5 | 有 GitHub 链接、Server、前端、通信、可观测性，闭环很强；但仍需要确认代码成熟度和测试覆盖 |
| 简历辨识度 | 5 | Rust + CUDA + LLM 推理框架非常少见，主讲人也明确表示“眼前一亮” |
| 复现价值 | 3.5 | 对强基础同学价值极高；对初学者直接完整复现难度过大，应拆成 mini 版本复现 |

总评：

> 这个项目适合你“参考思想和拆分复现”，但不适合你现在直接照搬做完整 RustInfer。你更适合先完成 Python mini-vLLM / nano-vLLM 源码学习，再做 vLLM benchmark，最后选一个小模块用 Rust 或 C++ 重写。

---

## 2.1.5 项目短板与简历优化建议

```text
原始问题：
- 技术点覆盖过宽，Rust、CUDA、CuTe、FlashAttention、WASM、ZeroMQ、OpenAI API、RadixAttention 全都写上，容易被面试官逐项深挖。
- “发现 GIL 瓶颈”这一表述风险较高，因为真实 vLLM 的瓶颈通常不简单等同于 Python GIL，更多在 GPU 执行、调度、KV Cache、通信和服务架构。
- “端到端延迟与 vLLM 持平或更优”是强结论，必须有完整 benchmark，否则容易被质疑。
- “大幅提升 Decode 阶段显存带宽利用率”需要具体指标支撑，例如 tokens/s、TPOT、P95 latency、显存占用、Nsight 指标。
- “CuTe 优化 FlashAttention”是高风险高价值表述，面试官可能会直接问 CuTe layout、tiling、MMA、shared memory、寄存器使用等细节。
```

```text
优化思路：
- 将简历主线收敛为“Rust 异步推理框架 + KV Cache 管理 + CUDA Graph 低延迟优化”。
- 所有性能结论必须给出测试环境和对比基线。
- 将“不确定/仍在实现”的模块从“已实现”中分离。
- 将“GIL 瓶颈”改成“Python 调度与跨层调用开销 / 服务层异步调度开销”，避免过度简化。
- 开源链接之外，最好加 README、架构图、benchmark 表格和在线 demo。
```

优化后的简历表述示例：

```text
基于 Rust + CUDA 实现轻量级 LLM 推理框架 RustInfer，参考 vLLM / SGLang 的
请求调度与 KV Cache 管理思想，完成 Scheduler、BlockManager、ModelWorker
等核心模块拆分，支持请求队列、流式生成和 OpenAI-compatible API。

设计非连续 KV Block 管理机制，基于引用计数与 LRU 策略实现 KV Block 分配、
复用与回收，并引入 RadixTree 管理共享前缀，为 Prefix Cache 和 Continuous
Batching 提供基础。

针对 Decode 阶段小 batch、多 kernel launch 的低延迟场景，实现 CUDA Graph
录制与重放流程，在固定 batch/shape 配置下减少 CPU launch overhead；构建
benchmark 脚本统计 TTFT、TPOT、tokens/s、显存占用，并与 vLLM 在相同模型、
相同输入输出长度和相同硬件环境下进行对比。
```

---

## 2.1.6 如果你要复现，学习过程是什么

这个项目不能一上来完整复现。正确做法是拆成四个可落地阶段。

### 阶段 1：先做 Python 版 mini 推理框架

目标：理解推理框架基本数据流。

需要完成：

```text
Request
  -> Tokenizer
  -> Scheduler
  -> Prefill
  -> Decode loop
  -> KV Cache
  -> Streaming output
```

学习内容：

- nano-vLLM / mini-vLLM；
- PyTorch 推理；
- Tokenizer；
- Prefill / Decode；
- KV Cache；
- batch 概念；
- EOS 停止条件。

产出：

- 一篇《mini-vLLM 执行流程笔记》；
- 一张请求生命周期流程图；
- 一个能跑小模型的最小推理 demo。

---

### 阶段 2：做 vLLM 部署与 Benchmark 项目

目标：先理解工业推理系统的性能问题。

需要完成：

- 用 vLLM 部署 Qwen 小模型；
- 写 OpenAI API 调用脚本；
- 写 benchmark 脚本；
- 统计 TTFT、TPOT、tokens/s、总延迟、显存；
- 对比不同输入长度、输出长度、并发数；
- 写出 Prefill / Decode 性能差异分析；
- 对比 FP16 / INT4 或 AWQ 量化。

产出：

- 一个可写入简历的 AI Infra 项目；
- 一份 benchmark 报告；
- 一份 vLLM 核心机制学习笔记。

---

### 阶段 3：用 C++ 或 Rust 重写一个小模块

目标：建立系统语言和推理框架之间的联系。

建议只选一个模块，不要全写：

- 选项 A：Rust 实现异步请求队列 + Scheduler；
- 选项 B：C++ 实现 KV BlockManager；
- 选项 C：Rust 实现 OpenAI-compatible Server；
- 选项 D：Python + C++ extension 实现一个简化 KV Cache 管理器。

产出：

- 一个独立小仓库；
- README 讲清楚模块职责；
- 单元测试；
- 与 Python 版本的对比说明。

---

### 阶段 4：再考虑 CUDA Graph / PagedAttention

目标：做一个高价值但可控的深入点。

优先顺序：

```text
CUDA 基础
  -> Softmax / RMSNorm / Reduce
  -> Nsight Compute
  -> CUDA Graph demo
  -> vLLM 中 CUDA Graph 机制学习
  -> KV Cache block 管理
  -> PagedAttention 原理笔记
```

不建议你现在直接写完整 PagedAttention kernel。先把原理、数据结构、block table、KV block、调度关系讲清楚，比强行写一个跑不通的 kernel 更有价值。

---

# 2.2 项目二：某 AI 算法方向科研课题

## 2.2.1 项目定位

这个项目目前只能判断为：

```text
AI for Science / 深度学习算法科研项目 / 可能具备工程落地潜力
```

它和 AI Infra 的关系取决于后续具体内容：

- 如果只是训练/改进算法模型：更偏算法；
- 如果涉及模型压缩、部署、推理加速：可转向模型部署；
- 如果涉及科研系统落地：可作为工程场景补充；
- 如果能形成“算法模型 + 推理服务 + 性能优化”闭环，则可与 AI Infra 产生联系。

---

## 2.2.2 属于 AI Infra 哪一层

| 项目 | AI Infra 层级 | 判断依据 | 与目标岗位的相关性 |
|---|---|---|---|
| 某 AI 算法方向科研课题 | 算法层 / AI for Science 应用层 | 截图显示研究方向是算法，未展示部署和推理优化 | 中等，除非补充部署、量化、推理优化 |
| 若未来工程落地 | 模型部署层 / 应用服务层 | “做得好的话未来有机会工程落地” | 中高，但需要有实际落地形态 |
| 若加入推理优化 | AI Infra 边缘部署 / 模型服务层 | 例如 ONNX、TensorRT、vLLM、FastAPI 服务化 | 高，可与岗位更贴合 |

---

## 2.2.3 技术栈拆解

| 技术栈 | 在项目中的作用 | AI Infra 价值 | 面试可能问题 | 学习优先级 |
|---|---|---|---|---|
| 深度学习模型 | 完成科研任务建模 | 理解模型结构和推理计算图 | 模型结构是什么？为什么选它？ | 高 |
| PyTorch | 训练和实验 | AI Infra 基础框架 | Dataset、autograd、AMP、模型保存加载 | 高 |
| AI for Science | 提供真实科研场景 | 增强项目差异化 | 科研问题如何转成模型问题？ | 中 |
| 工程落地 | 将算法用于真实系统 | 可转向模型部署 | 如何服务化？如何评估延迟和吞吐？ | 中高 |
| 模型部署 | 若补充 ONNX/TensorRT | 与 AI Infra 强相关 | ONNX 导出、动态 shape、精度校验 | 高 |

---

## 2.2.4 项目含金量评价

| 维度 | 分数 | 评价 |
|---|---:|---|
| AI Infra 相关性 | 2.5 | 当前信息显示更偏算法，和推理框架关系不直接 |
| 技术深度 | 3 | 取决于具体算法，截图无法确认 |
| 工程完整性 | 2 | 未看到数据、训练、部署、指标、系统闭环 |
| 简历辨识度 | 3 | AI for Science 有一定特色，但表达过于模糊 |
| 复现价值 | 2.5 | 如果与你研究方向相关可做，否则不如优先做推理项目 |

总评：

> 这个项目可以作为“科研背景”和“AI for Science 场景”补充，但不适合作为 AI Infra 求职主项目。要想提高价值，必须补充模型部署、推理服务、性能指标或工程落地内容。

---

## 2.2.5 项目短板与简历优化建议

```text
原始问题：
- 项目名过于模糊，“某 AI 算法方向科研课题”无法体现具体任务。
- 没有说明数据来源、模型结构、实验指标和结果。
- 没有体现工程落地方式。
- 和 AI Infra 岗位关系不清晰。
```

```text
优化思路：
- 明确科研任务是什么。
- 写清楚你负责的数据、模型、训练、评估或部署环节。
- 如果目标是 AI Infra，要增加“推理部署 / 模型压缩 / 服务化 / 端侧运行”内容。
```

优化后的简历表述模板：

```text
面向 XXX 科研场景，构建基于 PyTorch 的深度学习建模流程，完成数据预处理、
模型训练、指标评估与实验复现；针对模型推理部署需求，完成 ONNX 导出与
推理结果一致性校验，并基于 FastAPI / ONNX Runtime 封装推理服务，统计
平均延迟、P95 延迟和吞吐等指标，为后续工程落地提供基础。
```

---

## 2.2.6 如果你要复现，学习过程是什么

如果你想把自己的科研项目转向 AI Infra，应按下面顺序：

1. 先把科研问题讲清楚：输入是什么、输出是什么、指标是什么。
2. 用 PyTorch 训练或加载一个轻量模型。
3. 导出 ONNX。
4. 用 ONNX Runtime 做推理。
5. 对比 PyTorch 和 ONNX Runtime 的结果一致性。
6. 用 FastAPI 封装推理接口。
7. 统计延迟、吞吐、CPU/GPU 占用。
8. 再考虑 TensorRT、量化或边缘部署。

---

# 任务 3：总结视频内容

## 3.1 这个视频真正想说明什么

这个视频最重要的结论不是“Rust 一定比 Python 好”，也不是“所有人都应该从零写一个推理框架”。

真正的核心是：

```text
一个能让面试官眼前一亮的 AI Infra 项目，必须能证明你真的理解了推理框架的核心问题：
请求如何调度、KV Cache 如何管理、显存如何复用、Decode 如何降低延迟、
性能如何度量、工程如何闭环。
```

被点评者的 RustInfer 项目之所以强，是因为它不是普通应用层项目，而是深入到了推理框架核心层：

- Scheduler；
- BlockManager；
- Model Worker；
- PagedAttention / RadixAttention；
- Continuous Batching；
- CUDA Graph；
- BF16；
- OpenAI-compatible API；
- 可视化与监控。

---

## 3.2 视频对 AI Infra 求职的启发

### 启发一：有 GitHub 链接的开源项目是优势，但前提是代码真实能打

视频里主讲人的意思很明确：

```text
没有链接的项目，面试官会怀疑你是不是编的；
有链接的项目，至少能证明你真的写过。
```

但开源也有风险：

- 代码质量太差会暴露问题；
- README 写得差会降低可信度；
- commit 很假会被看穿；
- 只是 fork 后少量修改，反而会扣分；
- 写了强性能结论但 benchmark 不完整，会被质疑。

所以最好的状态不是单纯“开源”，而是：

```text
GitHub 仓库 + 清晰 README + 架构图 + benchmark + 可运行 demo + 关键模块文档
```

如果还能部署一个简单网页，让面试官打开后直接对话，并说明“这个服务完全跑在我自己的 Rust 推理框架上”，项目说服力会进一步提高。

---

### 启发二：面试前突击有用，但不能代替长期项目积累

视频里主讲人说的“做到哪里就会哪里”可以理解为：

- 真实做过的内容，面试前突击复盘非常有效；
- 被面试打到的点，复盘后下次可以解决；
- 但完全没做过的内容，靠面试前几天突击很难讲深。

因此你的策略应该是：

```text
平时靠项目建立真实理解；
面试前靠八股和项目复盘提高表达稳定性。
```

对你来说，不能等到面试前再补 vLLM、KV Cache、CUDA。至少要提前形成：

- 一个能运行的推理服务项目；
- 一个能讲清楚的源码阅读项目；
- 一套核心八股笔记；
- 一份项目 benchmark 报告。

---

### 启发三：没有实习不是绝对硬伤，但强项目必须足够强

视频中主讲人对这份简历的评价是“比较拔尖”“眼前一亮”。原因不是学历，也不是实习，而是项目足够稀缺。

但是对于大厂筛选来说，没有实习仍然有风险：

- 部分岗位简历筛选会偏好实习；
- 部分团队更信任真实业务经历；
- 没有实习时，项目必须足够完整、可验证、可运行；
- 需要通过 GitHub、技术博客、benchmark 报告补可信度。

所以结论是：

```text
没有实习可以靠强项目弥补，但不能靠普通项目弥补。
```

如果项目只是“部署 vLLM 跑了一个模型”，没有实习会比较弱；如果项目能做到 RustInfer 这种深度，就算没实习也有明显竞争力。

---

### 启发四：校招不要求你每个方向都是专家，但要求你有一个主深度

视频里主讲人提到，校招生不一定要在每个方向都是专家，更看重学习能力和培养潜质。但这不等于可以“什么都写一点”。

正确做法是：

```text
一主线 + 两辅助
```

以这份简历为例：

- 主线：Rust LLM 推理框架；
- 辅助一：CUDA / CuTe / CUDA Graph；
- 辅助二：WASM / OpenAI API / 工程展示；
- 科研项目：作为 AI for Science 场景补充。

对你来说，更适合的主线是：

```text
vLLM 推理服务 Benchmark + nano-vLLM 源码理解 + 一个小型推理框架模块改造
```

辅助线可以是：

- CUDA Softmax / RMSNorm / GEMM 入门；
- IMS 谱图识别与边缘部署；
- Linux / C++ / Docker 工程能力。

---

# 截图底部四个问题直接回答

## 问题 1：能看到代码的开源项目，比没有链接的项目是优势还是劣势？

结论：

```text
只要代码真实、README 清楚、项目能运行，开源链接是明显优势。
```

原因：

1. 有链接能证明你确实做过，不是简历编故事。
2. 面试官可以看到代码结构、commit、文档和工程完整性。
3. 对 AI Infra 项目来说，代码比口头描述更有可信度。
4. 如果再加 benchmark 和 demo，可信度会非常高。

但开源项目也会暴露问题：

- 代码大量复制但没有理解；
- README 写得像空壳；
- 项目不能运行；
- benchmark 造假或不可复现；
- 简历写的功能和代码不一致。

建议：

```text
开源，但不要裸奔。至少准备 README、架构图、运行命令、benchmark、关键模块说明。
```

---

## 问题 2：知识储备属于做到哪里就会到哪里，打算如果有面试再突击一下来得及吗？

结论：

```text
面试前突击可以补表达和八股，但不能替代真实项目积累。
```

来得及的部分：

- 项目复盘；
- 八股整理；
- 常见问题回答模板；
- vLLM / SGLang 高频知识点；
- CUDA 基础问答；
- LeetCode 高频题；
- 简历每句话的追问准备。

来不及的部分：

- 从零理解 KV Cache；
- 从零实现调度器；
- 从零补 CUDA；
- 从零读懂 vLLM；
- 从零写出可信 benchmark；
- 从零形成工程闭环。

正确策略：

```text
平时：持续做项目和笔记。
面试前 2-3 周：集中突击项目表达、八股、手撕题和简历追问。
每次面试后：把不会的问题补进复盘文档。
```

---

## 问题 3：没有实习经历，这种简历在字节这类大厂岗位面试中，能排在什么水平？面试官可能会问什么？

### 3.1 能排在什么水平？

如果 RustInfer 项目真实、代码完整、能跑通、指标可信，那么在校招/实习简历中属于：

```text
强项目型候选人，上限可到 SP / SSP 讨论区间。
```

但实际是否能进字节等大厂，还会受这些因素影响：

- 学历筛选；
- 岗位 HC；
- 投递时机；
- 是否有内推；
- 项目是否与岗位强匹配；
- 面试中是否能讲深；
- 是否有基础八股和算法题能力。

没有实习的影响：

```text
对普通项目是明显短板；
对强开源推理框架项目是可被弥补的短板。
```

### 3.2 面试官可能问什么？

#### A. 项目真实性

- 这个框架哪些模块是你独立写的？
- 哪些参考了 vLLM / SGLang / TGI / llama.cpp？
- 为什么选择 Rust？
- GitHub 里哪几个 commit 最关键？
- 现在项目能跑哪些模型？
- 支持哪些 tokenizer 和模型结构？
- 如何验证输出正确？

#### B. 推理框架核心

- Prefill 和 Decode 的区别是什么？
- Continuous Batching 怎么实现？
- Scheduler 如何决定请求进入 batch？
- BlockManager 如何分配和回收 KV block？
- PagedAttention 解决什么问题？
- RadixAttention 和 Prefix Cache 的关系是什么？
- EOS 后如何释放 KV Cache？
- 长上下文下显存如何估算？

#### C. Rust 系统设计

- Rust async 原理是什么？
- 为什么用 Actor 模型？
- tokio runtime 如何调度任务？
- 如何避免锁竞争？
- 如何做跨线程通信？
- Rust 如何调用 CUDA / C++？
- FFI 边界如何管理内存安全？

#### D. CUDA / GPU

- CUDA Graph 为什么能减少开销？
- 什么场景适合 CUDA Graph？
- Decode 阶段为什么容易被 kernel launch overhead 影响？
- BF16 和 FP16 区别？
- FlashAttention 为什么快？
- CuTe 中 layout / tiling 的基本思想是什么？
- Nsight Systems 和 Nsight Compute 分别看什么？

#### E. 性能指标

- 你的 10% 提升是怎么测的？
- 对比基线是什么？
- 模型多大？
- 输入输出长度是多少？
- batch size 和并发是多少？
- 显卡型号是什么？
- TTFT、TPOT、P95 latency 怎么统计？
- 为什么 Batch Size=1 能和 vLLM 持平或更优？

#### F. 多卡与分布式

- 单卡和多卡设计差异是什么？
- TP / PP / DP / EP 分别是什么？
- 多卡下 KV Cache 怎么切？
- Actor 模型如何扩展到多 GPU？
- ZeroMQ 在多进程通信中的作用是什么？

---

## 问题 4：项目覆盖范围很广，没有哪样特别精通，怎么规划从现在到暑期实习/正式秋招？

结论：

```text
不要继续横向堆技术栈，要把主线压缩到一个能打穿的方向。
```

推荐路线：

```text
主线：大模型推理框架 / LLM Serving
辅助：CUDA 高频算子 + Linux/C++ 工程基础
差异化：你的 IMS / 嵌入式 / AI for Science 场景
```

### 阶段 A：现在到 1 个月内

目标：打通使用闭环。

任务：

- 跑通 vLLM 部署；
- 部署 Qwen 小模型；
- 写 OpenAI-compatible API 调用脚本；
- 记录 TTFT、TPOT、tokens/s、显存；
- 整理 Prefill / Decode / KV Cache / PagedAttention 笔记；
- 复习 Python、PyTorch、Tokenizer、Transformer 基础。

产出：

- 一篇 vLLM 部署报告；
- 一个 benchmark 脚本；
- 一份 20 个 vLLM 高频问题清单。

---

### 阶段 B：第 2-3 个月

目标：形成第一个简历项目。

任务：

- 完成 LLM 推理服务 Benchmark 项目；
- 对比输入长度、输出长度、并发数、量化方式；
- 画表格和曲线；
- 写清楚性能瓶颈；
- 读 nano-vLLM / mini-vLLM 核心源码；
- 形成源码阅读笔记。

产出：

```text
项目一：基于 vLLM 的 LLM 推理服务部署与性能分析平台
```

这是你当前最应该优先完成的项目。

---

### 阶段 C：第 4-5 个月

目标：形成底层能力证明。

任务：

- 学 CUDA 基础；
- 写 Reduce / Softmax / RMSNorm；
- 用 Nsight Compute 看指标；
- 对比 naive 与优化版本；
- 整理 bank conflict、coalescing、occupancy、warp divergence；
- 不要一上来硬啃 FlashAttention。

产出：

```text
项目二：CUDA 高频算子优化实验
```

---

### 阶段 D：第 6 个月以后

目标：做差异化模块。

可选路线：

1. Rust 实现 mini Scheduler；
2. C++ 实现 KV Cache BlockManager；
3. 给 nano-vLLM 加 benchmark 和可视化；
4. 把 IMS 谱图识别模型导出 ONNX 并做边缘部署；
5. 做一个面向 IMS 的轻量推理服务。

产出：

```text
项目三：mini 推理框架模块改造 / IMS 谱图智能识别与边缘部署
```

---

## 你的最终项目组合建议

不要直接照搬 RustInfer。你更适合下面这个组合：

```text
项目 1：基于 vLLM 的 LLM 推理服务部署与性能 Benchmark
定位：AI Infra / LLM Serving / 推理服务

项目 2：nano-vLLM 源码阅读与推理引擎核心模块分析
定位：推理框架源码理解 / Scheduler / KV Cache / BlockManager

项目 3：CUDA 高频算子优化实验
定位：GPU 编程 / 算子基础 / 性能分析

项目 4：IMS 谱图智能识别与边缘部署
定位：你的差异化项目，体现嵌入式 + AI 部署
```

正式秋招简历里建议放 2-3 个，不要全堆：

- AI Infra 岗位：项目 1 + 项目 2 + 项目 3；
- 模型部署 / 边缘 AI 岗位：项目 1 + 项目 3 + 项目 4；
- 嵌入式 AI / Runtime 岗位：IMS 项目 + LLM Benchmark + CUDA 算子。

---

# 最终结论

这份被点评简历的核心竞争力不是“写了很多技术名词”，而是它抓住了 AI Infra 推理方向最核心的系统问题：

```text
调度、KV Cache、显存管理、低延迟 Decode、CUDA Graph、服务化接口、工程展示。
```

对你来说，最大的启发是：

```text
不要把学习目标定成“看懂所有 AI Infra 名词”；
要把目标定成“做出一个能跑、能测、能解释瓶颈、能对比指标的推理系统项目”。
```

你现在最应该做的不是复刻完整 RustInfer，而是：

```text
先跑通 vLLM Benchmark，
再精读 nano-vLLM，
再补 CUDA 高频算子，
最后选一个小模块做深入改造。
```

这样既符合你当前基础，也更容易在秋招前形成可信项目闭环。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-AI infra学习路线|模块-AI infra学习路线]]

%% 项目关联导航：结束 %%
