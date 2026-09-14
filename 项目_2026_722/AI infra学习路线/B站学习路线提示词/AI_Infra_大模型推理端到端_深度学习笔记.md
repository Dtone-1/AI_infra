# AI-Infra规划课和模拟面：大模型推理端到端是必须掌握的  
## 字幕整理 + 深度学习笔记 + 推理框架面试路线

> 输入材料：`AI-Infra规划课和模拟面_2026年6月20日_大模型推理端到端是必须掌握的_哔哩哔哩_bilibili.txt`  
> 整理方式：自动字幕纠错、语义复原、技术主线重组、AI Infra 推理侧知识补充、面试问答整理。  
> 说明：原始字幕存在大量自动识别错误，例如“VM”多处应理解为 **vLLM**，“苦大/孤单”应理解为 **CUDA**，“圣子/散贵/算值”应理解为 **算子**，“PGTens”应理解为 **PagedAttention**。本文不是逐字稿，而是面向学习和面试准备的结构化笔记。

---

## 目录

1. [这节课一句话结论](#一这节课一句话结论)
2. [这节课的核心宗旨](#二这节课的核心宗旨)
3. [字幕技术名词纠错表](#三字幕技术名词纠错表)
4. [按照课程推进顺序梳理知识点](#四按照课程推进顺序梳理知识点)
5. [整节课的技术主线：从输入文本到输出 token](#五整节课的技术主线从输入文本到输出-token)
6. [为什么“端到端推理流程”对算子岗也重要](#六为什么端到端推理流程对算子岗也重要)
7. [CUDA 算子项目应该如何改写简历](#七cuda-算子项目应该如何改写简历)
8. [大模型结构应该怎么补](#八大模型结构应该怎么补)
9. [vLLM 端到端推理框架应该学什么](#九vllm-端到端推理框架应该学什么)
10. [传统计算机基础应该准备哪些](#十传统计算机基础应该准备哪些)
11. [面试高频问题与参考答案](#十一面试高频问题与参考答案)
12. [学习行动清单](#十二学习行动清单)
13. [最终总结](#十三最终总结)

---

# 一、这节课一句话结论

这节课的核心结论是：

> **即使目标是 CUDA / 算子开发岗位，也不能只懂单个算子怎么运行；必须补齐大模型推理的端到端流程，至少要知道一个用户输入的句子如何经过 tokenizer、请求调度、KV Cache 管理、模型 forward、sampling、decode 循环，最终变成输出 token。**

老师在模拟面中指出，该同学目前的主要问题不是“完全不会 CUDA”，而是：

1. **CUDA 项目表达不够工程化**：只说“比 CPU 快十几倍”不够，必须说明输入规模、瓶颈阶段、为什么适合 GPU、做了哪些优化、优化前后分别是多少。
2. **大模型结构知识不足**：只知道 Transformer 基础还不够，要补 LLaMA、Qwen、DeepSeek 等典型开源模型的结构。
3. **端到端推理框架缺失**：不能只知道 encoding / decoding 或单算子执行，要知道 vLLM 这类推理框架如何从请求入口到输出 token。
4. **面试准备要有优先级**：普通论文性价比低于实习；传统八股里计算机网络和操作系统相对低频，数据结构/算法和计算机组成原理更值得优先准备。

---

# 二、这节课的核心宗旨

这节课表面上是在给一位想找 CUDA 算子开发实习的同学做模拟面和简历查漏补缺，实际讲的是一个更普遍的问题：

> **AI Infra / 大模型推理岗位不是“只写 kernel”或“只背 Transformer”，而是要求候选人能把模型结构、算子实现、推理框架、性能指标和简历表达串起来。**

这节课主要解决四个问题。

## 2.1 实习和论文，哪个更重要？

如果没有强毕业要求，只是发普通期刊或普通会议，那么对 AI Infra 求职来说，实习的性价比明显更高。原因是 AI Infra 推理侧是强工程岗位，面试官更关心你是否真的做过系统、算子、性能分析、部署和调优。

## 2.2 CUDA 项目如何证明“真的有工作量”？

不能只写“使用 CUDA 加速某控制算法，较 CPU 加速十几倍”。这类描述太弱。你需要回答：

- 输入规模有多大？
- 算法分为哪些计算阶段？
- 哪些阶段适合 GPU 并行？
- 优化前的 GPU naive 版本是多少？
- 优化后的 GPU 版本是多少？
- 相比 CPU、相比 naive CUDA、相比库函数分别提升多少？
- 用了哪些优化手段？
- 性能瓶颈是访存、计算、同步、分支还是数据搬运？

## 2.3 大模型结构从哪里开始补？

老师建议不要一上来找杂乱资料，而是围绕几个典型开源模型学习：

```text
LLaMA 系列 → Qwen 系列 → DeepSeek 系列
```

学习方法是：

```text
论文 → 解读文章/大模型辅助理解 → HuggingFace 或官方仓库模型结构代码 → 自己调试
```

这样可以快速建立对现代 LLM 结构的工程认知。

## 2.4 为什么必须掌握端到端推理？

老师特别强调：现在该同学只对“单算子怎么运行”有概念，但对“大模型端到端怎么推理”缺少整体认识。这个问题在算子岗面试中也会吃亏，因为推理框架岗和算子岗并不是完全割裂的。大模型推理引擎里的高性能算子最终服务于完整推理系统。

---

# 三、字幕技术名词纠错表

| 字幕识别结果 | 建议修正 | 说明 |
|---|---|---|
| 思目 / 世界家开发 | 可能是某公司/实习岗位，语义不影响主线 | 指当前实习或工作背景 |
| 抽招 | 秋招 | 求职场景 |
| 英文 / 印法 | Infra / AI Infra | 目标方向 |
| 乐文 | 论文 | 讨论论文与实习性价比 |
| 普刊 | 普通期刊 | 老师认为求职价值有限 |
| 圣子 / 散贵 / 算值 | 算子 | CUDA 算子开发 |
| 苦大 / 孤单 | CUDA | GPU 编程与算子开发 |
| NV | NVIDIA / GPU | 语境是用 GPU 加速 |
| Transfermer | Transformer | 大模型基础结构 |
| Lemma / 莱马 | LLaMA | Meta 开源模型系列 |
| 千万 / 千万系列 | Qwen / 通义千问 | 国内主流开源模型 |
| 丢细里 | DeepSeek | 国内主流大模型系列 |
| 跟号部 / 跟它号部 | GitHub | 官方或开源代码仓库 |
| 跟费子 | HuggingFace | 模型和模型结构代码来源 |
| VM / VAM | vLLM | 大模型推理框架 |
| token ID | token id | tokenizer 输出的离散 token 编号 |
| encoding / decoding | 编码 / 解码，或 prefill / decode 的误解 | 老师指出它只是模型推理的一部分 |
| PGTens | PagedAttention | vLLM 高频考点 |
| 调土 | 调度 | 请求调度 / scheduler |
| E 的 code | LeetCode | 算法题准备 |

---

# 四、按照课程推进顺序梳理知识点

原始字幕没有明确时间戳，因此这里按照课程对话的推进顺序整理。

---

## 4.1 第一段：确认求职目标——金融量化不是主线，AI Infra 才是目标

### 核心内容

开头老师先确认同学当前的实习或项目背景，并询问秋招目标。该同学表示金融量化只是第一份实习或临时选择，自己真正想做的是 AI Infra，尤其是 CUDA / 算子开发方向。

### 相关概念解释

**AI Infra** 指支撑 AI 模型训练、推理、部署、调度、加速的基础设施方向。它包括：

- 大模型推理框架；
- CUDA / Triton 算子开发；
- AI 编译器；
- 模型部署与推理服务；
- 分布式训练和推理；
- GPU / NPU Runtime；
- 量化、并行、显存优化。

**金融量化** 和 AI Infra 的技术栈有交叉，例如 C++、Python、性能优化、并行计算，但业务方向和面试关注点差异很大。金融量化更看数学、统计、策略研究、交易系统；AI Infra 更看系统、模型、GPU、框架和算子。

### 在 AI Infra 求职中的作用

目标必须尽早明确。如果简历主线太分散，面试官会怀疑你只是“什么都投”，而不是对 AI Infra 有明确投入。对于你这种时间紧张、想速成推理侧的情况，简历最好围绕：

```text
CUDA 算子能力 + 大模型推理框架理解 + 可量化性能优化项目
```

---

## 4.2 第二段：普通论文 vs 实习——求职性价比判断

### 核心内容

同学问普通期刊或普通论文是否值得投入。老师判断：如果只是普通期刊，而且毕业没有硬性要求，那么不如把时间投入到实习和项目中。

### 相关概念解释

AI Infra 推理侧岗位本质是工程岗位。面试官通常更关心：

- 是否写过 CUDA / Triton；
- 是否做过性能 benchmark；
- 是否懂 vLLM / SGLang / TensorRT-LLM；
- 是否理解 KV Cache / PagedAttention / continuous batching；
- 是否能解释项目中的瓶颈和优化；
- 是否能手撕常见算子或算法题。

普通论文如果不在系统、HPC、AI 编译器、机器学习系统、推理优化等方向，对工程面试帮助有限。

### 在 AI Infra 求职中的作用

优先级应该是：

```text
对口实习 > 高质量工程项目 > 开源贡献/技术博客 > 普通论文
```

除非论文质量高、方向对口、能体现系统优化能力，否则不要把大量时间压在普通论文上。

---

## 4.3 第三段：CUDA 控制算法项目的问题——“比 CPU 快十几倍”不够

### 核心内容

该同学的第一个项目是实验室控制算法，用 CUDA 实现并加速。加速效果相比 CPU 大概十几倍。老师指出这个表达还不够，需要写清楚问题规模、性能指标、优化策略和实验数据。

### 相关概念解释

CUDA 项目的性能表达至少要区分三种对比对象：

| 对比对象 | 说明 | 面试价值 |
|---|---|---|
| CPU baseline | 原始 CPU 实现 | 能说明 GPU 加速收益，但说服力有限 |
| naive CUDA | 未优化 GPU 实现 | 能说明你的优化真正有效 |
| 优化 CUDA / 库函数 | 经过 shared memory、coalescing、tiling 等优化后的版本 | 能体现算子优化能力 |

老师强调：只和 CPU 比不够，因为当输入规模足够大时，哪怕写得一般的 GPU kernel 也可能比 CPU 快很多。真正有说服力的是你能说明：

- 为什么这个问题适合 GPU；
- 哪些部分可以并行；
- naive CUDA 哪里慢；
- 你做了什么优化；
- 优化后指标如何变化。

### 在 AI Infra / 推理系统中的作用

大模型推理优化也是同样逻辑。你不能只说“使用 vLLM 后吞吐提升”，而要说明：

- 是 TTFT 降低了，还是 TPOT 降低了？
- 是 prefill 快了，还是 decode 快了？
- 是显存利用率提高了，还是 batch size 变大了？
- 是 kernel launch overhead 减少了，还是 HBM 读写减少了？
- 是 PagedAttention、continuous batching、prefix cache、量化还是 CUDA Graph 起作用？

---

## 4.4 第四段：项目简历表达应该从“问题特点”出发

### 核心内容

老师建议同学在简历中先解释控制算法的问题特点，再说明哪些阶段适合 GPU 加速，最后描述优化策略和优化效果。

### 相关概念解释

一个好的 CUDA 项目描述应该符合下面这个逻辑：

```text
问题背景
  → 输入规模和计算模式
  → 原始 CPU 或 naive GPU 性能瓶颈
  → 为什么适合 GPU 并行
  → kernel 设计
  → 优化策略
  → benchmark 数据
  → 工程收益
```

例如：

> 针对控制算法中大规模状态矩阵更新与代价函数评估阶段计算密集、样本间相互独立的特点，将核心循环迁移到 CUDA kernel，并通过合并访存、减少主机设备数据拷贝、使用 shared memory 缓存中间结果、优化线程块划分等方式提升吞吐。

### 在 AI Infra 求职中的作用

面试官不一定懂你的控制算法，但他一定懂性能优化方法。因此你需要把项目从“领域项目”翻译成“计算问题”：

```text
控制算法问题 → 并行计算问题 → CUDA kernel 设计问题 → 性能优化问题
```

这才是算子岗面试官能快速理解的表达方式。

---

## 4.5 第五段：大模型结构知识缺失——不能只停留在 Transformer

### 核心内容

同学表示自己只懂比较基础的 Transformer，对 LLaMA、Qwen、DeepSeek 等现代大模型结构不熟悉。老师建议补齐这一块。

### 相关概念解释

Transformer 是基础，但现代大模型面试通常还会问：

- LLaMA 和原始 Transformer 有什么区别？
- 为什么很多 LLM 使用 RMSNorm 而不是 LayerNorm？
- RoPE 是什么？
- SwiGLU / Gated MLP 是什么？
- MHA、MQA、GQA 的区别是什么？
- Qwen 系列有什么结构特点？
- DeepSeek 的 MoE、MLA、MTP 等关键词是什么意思？
- MoE 为什么能降低推理成本？
- KV Cache 和模型结构有什么关系？

### 在 AI Infra 求职中的作用

做推理框架或算子开发时，模型结构不是“算法岗才需要懂”的东西。因为模型结构决定了：

- 需要哪些算子；
- 哪些算子是瓶颈；
- KV Cache 怎么组织；
- 能否使用 GQA / MQA 减少 KV Cache；
- MoE 是否需要专家并行和 All-to-All；
- 量化 kernel 如何设计；
- 推理框架如何适配模型。

---

## 4.6 第六段：学习大模型结构的方法——论文 + 解读 + 代码

### 核心内容

老师建议按以下方式补模型结构：

1. 看典型开源模型论文；
2. 看别人对论文的解读；
3. 用大模型辅助理解；
4. 看 GitHub / HuggingFace 上的模型结构定义代码；
5. 自己调试运行。

### 相关概念解释

论文通常会把一个结构或优化方法讲得很“宏大”，但代码会让你看到它在工程上到底是什么。例如：

- RMSNorm 在代码里就是一个归一化层；
- SwiGLU 在代码里就是两个线性投影加激活再相乘；
- GQA 在代码里体现为 `num_attention_heads` 和 `num_key_value_heads` 不相等；
- RoPE 在代码里体现为对 Q/K 做旋转位置编码；
- MoE 在代码里体现为 router 选择 expert，然后对 token 分组计算。

### 在 AI Infra 求职中的作用

对于工程岗，最终要落到代码。你不一定要像算法研究员一样推导所有理论，但必须能看懂：

```text
模型配置 config.json
  → 模型结构代码 modeling_xxx.py
  → forward 流程
  → Q/K/V、MLP、Norm、MoE 的具体实现
  → 这些模块对应哪些 kernel
```

---

## 4.7 第七段：复现项目已经能满足部分实习要求，但创新不足要用深度补

### 核心内容

同学表示自己做项目主要是看完后自己写一遍，能复现，但创新有困难。老师认为，如果能独立复现，对算子岗位实习已经有一定价值，不必过度焦虑。

### 相关概念解释

校招 / 实习阶段的项目一般分为三层：

| 层级 | 说明 | 面试价值 |
|---|---|---|
| 跑通使用 | 按教程运行项目 | 价值较低 |
| 独立复现 | 理解后自己写出可运行版本 | 有价值 |
| 改进优化 | 对性能、功能或工程结构做改进 | 最有价值 |

如果暂时不能创新，就要把“复现深度”做扎实：

- 是否理解每个 kernel 的职责？
- 是否能解释优化路径？
- 是否能自己改 block size / grid size？
- 是否能分析性能瓶颈？
- 是否有 benchmark 表格？
- 是否能和 PyTorch / cuBLAS / Triton 对比？

### 在 AI Infra 求职中的作用

算子岗位不要求每个学生都发明新算法，但要求你真的会写、会测、会解释。能把一个经典算子从 naive 到优化版本复现出来，并能讲清楚为什么快，本身就是有效项目。

---

## 4.8 第八段：端到端推理流程缺失——这是本节课最重要的问题

### 核心内容

老师问同学是否端到端跑过 vLLM，是否跟过“一个请求文本进去，到最后输出 token”的完整流程。该同学表示没有。老师指出：这是目前最大短板。

### 相关概念解释

很多初学者把“大模型推理”理解成：

```text
输入文本 → 模型 forward → 输出文本
```

但真实推理框架的流程更复杂：

```text
用户请求
  → API Server
  → tokenizer
  → request / sequence 对象
  → scheduler 调度
  → KV Cache 分配
  → prefill
  → sampling
  → decode 循环
  → streaming output
  → 请求结束与资源释放
```

其中模型 forward 只是中间一部分。更关键的是：

- 请求如何排队；
- batch 如何动态组成；
- KV Cache 如何分配；
- prefill 和 decode 如何调度；
- token 如何逐步生成；
- 完成请求如何释放显存；
- 多请求如何共享前缀；
- 多卡并行如何组织。

### 在 AI Infra 求职中的作用

即使是算子岗，面试官也可能问：

> 你写的算子在完整大模型推理系统中处于什么位置？

如果你只能回答“这个 kernel 输入一个矩阵输出一个矩阵”，但不知道它在 prefill 还是 decode 阶段、不知道它影响 TTFT 还是 TPOT、不知道它和 KV Cache 的关系，项目就会显得割裂。

---

## 4.9 第九段：encoding / decoding 不是完整推理框架

### 核心内容

同学把推理过程理解成 encoding / decoding。老师纠正说，这只是模型内部的一部分；vLLM 这类推理框架还包括 tokenizer、请求调度、资源管理、并行、采样等内容。

### 相关概念解释

在 LLM 推理中，更常用的阶段划分是：

| 阶段 | 作用 | 特点 |
|---|---|---|
| Tokenization | 文本转 token ids | CPU 侧，和 tokenizer 相关 |
| Prefill | 处理 prompt，构建初始 KV Cache | 计算密集，矩阵乘较大，影响 TTFT |
| Decode | 每次生成一个新 token | 访存密集，受 KV Cache 读取和显存带宽影响，影响 TPOT |
| Sampling | 从 logits 中选择下一个 token | temperature、top-k、top-p 等 |
| Detokenization | token ids 转文本 | CPU 侧，流式输出时持续执行 |

**encoding / decoding** 在不同语境下含义不同。对于 decoder-only LLM，严格说没有传统 encoder-decoder 结构里的 encoder。在线推理通常更关注 prefill / decode。

### 在 AI Infra 求职中的作用

面试时要避免把概念混用。更专业的表达是：

> 对 decoder-only 大模型推理而言，输入文本先经过 tokenizer 变成 token ids，然后进入请求调度；调度器安排 prefill 构建 KV Cache，之后进入 decode loop，每一步执行模型 forward 得到 logits，再经过 sampling 得到下一个 token，直到 EOS 或达到最大长度。

---

## 4.10 第十段：vLLM 必须了解的模块

### 核心内容

老师提到即使目标是算子岗，也推荐了解 vLLM 中有哪些模块，因为它能帮助建立大模型整体推理流程。

### 相关概念解释

vLLM 的学习可以先抓几个核心模块：

| 模块 | 作用 |
|---|---|
| API Server | 接收 OpenAI-compatible 请求 |
| Tokenizer | 文本和 token id 互转 |
| Engine | 推理请求的入口和结果出口 |
| Scheduler | 决定哪些请求进入本轮执行 |
| KV Cache Manager / Block Manager | 分配和释放 KV Cache blocks |
| Worker | 负责模型执行的进程/线程/设备侧执行单元 |
| ModelRunner | 真正组织模型 forward |
| Attention Backend | 调用 PagedAttention / FlashAttention 等底层 kernel |
| Sampling | 从 logits 生成下一个 token |
| Output Processor | 流式返回或聚合输出 |

### 在 AI Infra 求职中的作用

这些模块把“模型结构”和“工程系统”连接起来。你读懂这些之后，才能回答：

- vLLM 为什么吞吐高？
- PagedAttention 解决什么问题？
- Continuous batching 怎么做？
- Prefill 和 decode 在调度上有什么差异？
- KV Cache 怎么分配和释放？
- 算子优化对端到端性能有什么影响？

---

## 4.11 第十一段：传统计算机基础准备优先级

### 核心内容

同学问算子岗是否需要准备传统计算机基础，例如操作系统、计算机网络、计算机组成、数据结构等。老师认为计算机网络和操作系统相对低频，计算机组成原理和数据结构/算法更重要。

### 相关概念解释

AI Infra 面试中传统基础的优先级可以这样理解：

| 科目 | 优先级 | 原因 |
|---|---|---|
| 数据结构与算法 | 高 | LeetCode 高频，框架岗更常问 |
| 计算机组成原理 | 高 | GPU 架构、存储层次、访存、指令、并行计算都和它相关 |
| 操作系统 | 中 | 调度、内存管理、进程线程、虚拟内存对理解推理框架有帮助 |
| 计算机网络 | 中低 | 只有通信、分布式、服务端岗位更容易问；纯算子岗较少问 TCP/UDP 细节 |

### 在 AI Infra 求职中的作用

对于时间有限的算子 / 推理岗准备者，建议优先：

```text
LeetCode 中等题
  + CUDA / GPU 存储层次
  + 计算机组成原理中的 cache、内存、流水线、并行
  + 操作系统中的进程线程、虚拟内存、调度
```

计算机网络不要完全不看，但不必把 TCP 三次握手、拥塞控制这类传统八股放在最高优先级。

---

# 五、整节课的技术主线：从输入文本到输出 token

本节课最核心的技术主线是：**大模型推理端到端流程**。

下面按照“从输入到输出”的方式，把完整流程串起来。

---

## 5.1 用户输入阶段：自然语言请求进入服务

用户在网页、API、命令行或客户端里输入一句话，例如：

```text
请解释一下 PagedAttention 是什么。
```

这句话首先不是直接进入 GPU。它会进入推理服务的入口，例如 vLLM 的 OpenAI-compatible API server。

这一层通常处理：

- HTTP 请求；
- 模型名称；
- prompt / messages；
- temperature、top_p、max_tokens 等采样参数；
- 是否 stream；
- 用户请求校验。

---

## 5.2 Tokenization：文本转 token ids

大模型不能直接处理中文或英文字符串。推理系统会调用 tokenizer，把文本切成 token，并映射成整数 id：

```text
"请解释一下 PagedAttention 是什么"
  → [token_1, token_2, token_3, ...]
  → [10123, 4567, 9982, ...]
```

这一步通常在 CPU 侧完成。

需要理解：

- token id 是模型词表中的编号；
- 不同模型 tokenizer 不同；
- 同一句话在不同模型中可能对应不同 token ids；
- token 数量直接影响 prefill 计算量和 KV Cache 占用。

---

## 5.3 Request / Sequence 构造：请求进入引擎内部

token ids 会被封装成内部请求对象，例如：

- request id；
- prompt token ids；
- 已生成 token ids；
- sampling params；
- 状态：waiting / running / finished；
- KV Cache block 信息；
- 最大输出长度；
- EOS 条件。

推理框架需要维护大量请求的状态，因为在线服务不是只有一个请求。

---

## 5.4 Scheduler：请求调度

调度器决定当前这一轮 GPU 要执行哪些请求。

它要考虑：

- 当前 GPU 还能容纳多少 token；
- 当前 KV Cache block 是否够用；
- 哪些请求处于 prefill；
- 哪些请求处于 decode；
- 是否有长 prompt 阻塞；
- 是否使用 continuous batching；
- 是否要做 chunked prefill；
- 是否要保证公平性和低延迟。

调度器是推理框架区别于普通 `model.generate()` 的关键部分。

---

## 5.5 KV Cache 分配：为请求申请显存块

LLM 自回归生成时，每个 token 都会产生每一层的 Key / Value。为了避免重复计算历史 token，框架会保存这些 K/V，这就是 KV Cache。

KV Cache 的问题在于：

- 请求长度不同；
- 输出长度动态增长；
- 多请求并发；
- 显存容易碎片化；
- 长上下文占用巨大。

vLLM 的 PagedAttention / block manager 就是为了解决这个问题：把 KV Cache 拆成 block，按需分配和释放，而不是一次性给每个请求预留最大长度。

---

## 5.6 Prefill：处理 prompt，建立初始 KV Cache

调度器选中一个新请求后，会先执行 prefill。

Prefill 的作用：

```text
输入 prompt tokens
  → 经过所有 Transformer decoder layers
  → 为每一层生成 prompt 部分的 K/V
  → 写入 KV Cache
  → 得到最后一个位置的 logits
```

Prefill 特点：

- 一次处理多个 prompt token；
- 矩阵乘规模较大；
- 计算密集；
- 对 TTFT 影响明显；
- 长 prompt 会拖慢首 token 输出。

---

## 5.7 Sampling：从 logits 选择下一个 token

模型 forward 输出的是 logits，即词表上每个 token 的分数。Sampling 模块根据采样参数选择下一个 token：

- greedy：选最大概率 token；
- temperature：调节分布平滑程度；
- top-k：只在概率最高 k 个 token 中采样；
- top-p：只在累计概率达到 p 的 token 集合中采样；
- repetition penalty：惩罚重复 token。

输出的 token 会追加到该请求的生成序列中。

---

## 5.8 Decode loop：逐 token 生成

之后请求进入 decode 阶段。每一轮只输入上一步生成的新 token，并读取历史 KV Cache：

```text
上一步生成的新 token
  + 历史 KV Cache
  → 模型 forward
  → logits
  → sampling
  → 下一个 token
```

Decode 特点：

- 每轮通常只处理每个请求的一个新 token；
- 需要频繁读取 KV Cache；
- 更容易受显存带宽、kernel launch、调度开销影响；
- 对 TPOT 和整体吞吐影响大。

---

## 5.9 Continuous Batching：动态加入和移除请求

在线服务中，不同请求完成时间不同。如果使用固定 batch，短请求会被长请求拖住。

Continuous batching 的思想是：

```text
每一轮 decode 后：
  已完成请求 → 移出 batch，释放 KV Cache
  新请求 → 加入 batch，开始 prefill 或 decode
  未完成请求 → 继续下一轮 decode
```

这样 GPU 可以持续保持较高利用率。

---

## 5.10 结束条件与资源释放

当满足以下条件之一时，请求结束：

- 生成 EOS token；
- 达到 max_tokens；
- 用户中断；
- 超时；
- stop words 命中。

请求结束后，框架需要：

- 释放 KV Cache blocks；
- 更新请求状态；
- 返回最终文本；
- 对 stream 模式持续返回增量 token；
- 清理调度队列中的状态。

这一步非常重要，因为如果 KV Cache 不及时释放，显存会越来越紧张，吞吐会下降。

---

# 六、为什么“端到端推理流程”对算子岗也重要

很多准备 CUDA 算子岗的同学容易误解：

> 我只要会写 MatMul、Softmax、Reduce、LayerNorm 就行，vLLM 端到端流程不重要。

这节课明确否定了这种想法。原因如下。

## 6.1 算子必须放在系统里才有意义

一个 kernel 本身可能很快，但它是否真的提升端到端性能，要看它处于哪个阶段：

| 算子 | 常见位置 | 主要影响 |
|---|---|---|
| GEMM | Attention QKV、MLP、LM Head | prefill 和 decode 计算 |
| Softmax | Attention score、sampling | attention 和采样 |
| RMSNorm / LayerNorm | 每层 block 前后 | 小算子开销、融合机会 |
| PagedAttention kernel | decode attention | KV Cache 读取效率、TPOT |
| Fused MoE | MoE expert 计算 | MoE 模型吞吐和延迟 |
| Quantized GEMM | 量化模型推理 | 显存、带宽、吞吐 |

如果你不知道完整推理流程，就很难说明你优化的算子影响 TTFT、TPOT、吞吐还是显存。

## 6.2 面试官会问“你这个优化在端到端里有什么收益？”

单算子 benchmark 只是 micro-benchmark。工业中更关心 end-to-end：

- 单算子快 30%，端到端是否也快？
- 优化是否增加额外数据搬运？
- 是否影响动态 batch？
- 是否适配不同 batch / seq length？
- 是否适配 prefill 和 decode 两种 shape？
- 是否支持量化模型？
- 是否能接入 vLLM / TensorRT-LLM / SGLang？

## 6.3 推理框架是算子的上层调用者

算子开发不是独立存在的。上层框架需要：

- 根据请求 shape 选择 kernel；
- 管理 KV Cache；
- 安排 batch；
- 调用 attention backend；
- 处理多卡并行；
- 做 sampling 和输出。

因此算子岗也要知道：

```text
框架什么时候调用我的 kernel？
输入输出 shape 从哪里来？
数据在 GPU 显存中怎么组织？
kernel 运行前后还有哪些开销？
```

---

# 七、CUDA 算子项目应该如何改写简历

本节课对简历的建议非常重要。下面给出可直接套用的项目表达模板。

---

## 7.1 不推荐写法

```text
使用 CUDA 实现实验室控制算法加速，相比 CPU 版本提升十几倍。
```

这个写法的问题：

- 没有输入规模；
- 没有说明算法瓶颈；
- 没有说明为什么适合 GPU；
- 没有说明具体优化；
- 没有对比 naive CUDA；
- 没有性能指标；
- 面试官无法判断真实工作量。

---

## 7.2 推荐写法结构

```text
项目背景：
针对 xxx 控制算法中 xxx 阶段计算量大、样本间独立、循环结构规则的问题，设计 CUDA 并行加速方案。

工作内容：
1. 分析算法计算流程，将 xxx 阶段抽象为 xxx 并行计算问题；
2. 实现 CPU baseline、naive CUDA 和优化 CUDA 三个版本；
3. 针对 xxx 输入规模，设计线程块划分和数据布局；
4. 通过 coalesced memory access / shared memory / 减少同步 / 减少 host-device copy 等方式优化；
5. 使用 CUDA Event / Nsight Compute 统计 kernel 耗时、吞吐和内存访问效率。

效果：
在 xxx 输入规模下，优化 CUDA 版本相比 CPU 加速 xx 倍，相比 naive CUDA 加速 xx%，端到端耗时由 xx ms 降至 xx ms。
```

---

## 7.3 可写进简历的版本示例

> **基于 CUDA 的控制算法并行加速与性能优化**  
> - 针对实验室控制算法中大规模状态更新与代价函数评估阶段计算密集、样本间独立的特点，将核心循环迁移至 CUDA kernel，并实现 CPU baseline、naive CUDA 与优化 CUDA 三个版本；  
> - 设计基于输入规模的 grid/block 划分策略，优化 global memory 合并访问，减少 host-device 数据拷贝，并通过 shared memory 缓存中间结果降低重复访存；  
> - 使用 CUDA Event / Nsight Compute 对 kernel latency、memory throughput、occupancy 等指标进行分析，在 xxx 规模输入下相比 CPU 版本加速 xx 倍，相比 naive CUDA 版本提升 xx%。

---

## 7.4 面试时必须准备的数据

你至少要提前准备下面这张表：

| 输入规模 | CPU 耗时 | naive CUDA 耗时 | optimized CUDA 耗时 | CPU 加速比 | naive CUDA 提升 |
|---|---:|---:|---:|---:|---:|
| small | xx ms | xx ms | xx ms | xx × | xx % |
| medium | xx ms | xx ms | xx ms | xx × | xx % |
| large | xx ms | xx ms | xx ms | xx × | xx % |

还要准备：

- 数据规模为什么选这几个；
- small 规模下 GPU 是否可能不如 CPU；
- large 规模下为什么加速比提高；
- 数据搬运时间是否计入；
- kernel 时间和端到端时间是否区分；
- 误差或数值一致性如何验证。

---

# 八、大模型结构应该怎么补

## 8.1 学习目标

你不需要把所有模型论文研究到算法岗水平，但要达到工程岗要求：

```text
能看懂模型结构
能看懂 HuggingFace forward
能知道每个模块对应什么算子
能知道结构变化对推理性能有什么影响
```

## 8.2 推荐学习顺序

### 第一阶段：LLaMA 系列

重点看：

- Decoder-only Transformer；
- RMSNorm；
- RoPE；
- SwiGLU；
- GQA；
- KV Cache；
- causal mask；
- tokenizer 与模型配置。

学习价值：

> LLaMA 是现代开源 LLM 的基础模板。看懂 LLaMA，后面看 Qwen、DeepSeek 会轻松很多。

### 第二阶段：Qwen 系列

重点看：

- Qwen 的模型配置；
- GQA / attention head 与 kv head；
- RoPE 长上下文扩展；
- tokenizer 特点；
- dense / MoE 版本差异；
- thinking / non-thinking 相关设计。

学习价值：

> Qwen 是国内岗位面试常见模型。很多推理部署项目会用 Qwen 系列小模型做 benchmark。

### 第三阶段：DeepSeek 系列

重点看：

- MoE；
- MLA；
- MTP；
- expert routing；
- DeepSeek 系列为什么推理成本低；
- MoE 对推理框架和算子的要求。

学习价值：

> DeepSeek 相关结构会引出 MoE、Fused MoE、专家并行、KV Cache 压缩、长上下文等 AI Infra 高频问题。

---

## 8.3 具体学习方法

每个模型按四步走：

```text
第 1 步：看论文摘要、模型结构章节、实验设置
第 2 步：看中文解读或技术博客
第 3 步：打开 HuggingFace modeling_xxx.py 看 forward
第 4 步：用小模型跑一次推理，并打断点观察 tensor shape
```

重点观察：

- `config.json` 中的 hidden_size、num_layers、num_attention_heads、num_key_value_heads；
- attention forward 中 Q/K/V shape 如何变化；
- RoPE 在哪里加；
- KV Cache 如何传入和返回；
- MLP / MoE 如何计算；
- logits 如何生成。

---

# 九、vLLM 端到端推理框架应该学什么

## 9.1 先掌握整体流程，不要一上来死读源码

最短学习路径：

```text
部署 vLLM 服务
  → 调用 OpenAI-compatible API
  → 观察输入输出
  → 学 tokenizer / request / scheduler / KV cache / model runner
  → debug 一次请求从进入到输出
  → 再看 PagedAttention 和 continuous batching
```

## 9.2 必学知识点

| 知识点 | 必须会回答的问题 |
|---|---|
| vLLM 是什么 | 它解决 LLM serving 的什么问题？ |
| OpenAI-compatible API | 请求如何进入服务？ |
| Tokenizer | 文本如何变成 token ids？ |
| Request / Sequence | 框架如何表示一个请求？ |
| Scheduler | 多请求如何调度？ |
| Prefill / Decode | 两阶段有什么区别？ |
| KV Cache | 为什么要缓存 K/V？ |
| PagedAttention | 如何解决显存碎片和动态分配？ |
| Continuous Batching | 为什么能提升吞吐？ |
| Chunked Prefill | 为什么长 prompt 要拆分？ |
| Sampling | logits 如何变成 token？ |
| Worker / ModelRunner | 模型 forward 在哪里执行？ |
| Attention backend | 底层 attention kernel 如何被调用？ |
| Quantization | 量化模型如何影响显存和算子？ |
| Parallelism | TP / PP / DP / EP 分别是什么？ |

## 9.3 推荐调试路线

以一个简单 prompt 为例：

```python
prompt = "介绍一下 vLLM 的 PagedAttention"
```

调试时观察：

1. prompt 如何进入 API server；
2. tokenizer 输出多少 token；
3. request 对象如何创建；
4. scheduler 什么时候把它从 waiting 放到 running；
5. KV Cache block 如何申请；
6. prefill 输入 shape 是什么；
7. 模型 forward 返回 logits；
8. sampling 得到第一个 token；
9. decode 每轮输入 shape 如何变化；
10. token 如何流式返回；
11. EOS 后 block 如何释放。

---

# 十、传统计算机基础应该准备哪些

## 10.1 数据结构与算法

必须准备。尤其是：

- 数组；
- 哈希表；
- 链表；
- 栈和队列；
- 二叉树；
- 图；
- 堆；
- 二分；
- 滑动窗口；
- 动态规划基础；
- 拓扑排序。

算子岗也可能考 LeetCode，推理框架岗更可能考。

## 10.2 计算机组成原理

优先级高。重点是：

- cache；
- 内存层次；
- 局部性；
- 流水线；
- SIMD / SIMT；
- 指令级并行；
- 浮点数；
- 带宽和延迟；
- 并行计算基础。

这些和 GPU 架构、CUDA 访存优化高度相关。

## 10.3 操作系统

优先级中等。重点是：

- 进程 / 线程；
- 调度；
- 虚拟内存；
- 页表；
- mmap；
- 共享内存；
- 锁；
- 上下文切换；
- 文件系统基础。

PagedAttention 类似操作系统分页思想，所以 OS 基础会帮助理解 KV Cache block 管理。

## 10.4 计算机网络

优先级相对低，但不能完全不懂。重点是：

- HTTP；
- RPC；
- TCP / UDP 基本区别；
- 带宽与延迟；
- 分布式通信概念；
- RDMA / NCCL / InfiniBand 可以作为 AI Infra 拓展。

纯算子岗不必把传统网络八股放在最高优先级。

---

# 十一、面试高频问题与参考答案

## 1. 为什么普通论文不如实习有价值？

**答：** AI Infra 推理侧是工程岗位，普通论文如果不在系统、HPC、推理优化、AI 编译器等强相关方向，对面试帮助有限。实习能证明真实工程能力，包括代码、调试、性能分析、协作和项目落地，所以如果没有毕业硬性要求，优先做对口实习。

---

## 2. 你的 CUDA 项目为什么不能只写“比 CPU 快十几倍”？

**答：** 因为只和 CPU 比说明不了优化能力。输入规模足够大时，很多 naive GPU 实现也可能比 CPU 快。更有说服力的是同时给出 CPU baseline、naive CUDA、optimized CUDA 的对比，并说明优化前瓶颈、优化策略、具体指标和端到端收益。

---

## 3. CUDA 项目中应该如何证明性能优化有效？

**答：** 需要准备不同输入规模下的 benchmark，至少包括 CPU 耗时、naive CUDA 耗时、优化 CUDA 耗时、加速比、kernel 时间和端到端时间。同时说明是否计入 host-device copy，使用 CUDA Event 或 Nsight Compute 统计指标，并分析优化前后瓶颈变化。

---

## 4. 为什么面试官关心输入规模？

**答：** GPU 加速收益和输入规模强相关。小规模问题可能被 kernel launch 和数据搬运开销抵消，大规模问题才能发挥并行吞吐优势。如果不说明输入规模，加速比没有可解释性，也无法判断项目是否真实。

---

## 5. 为什么算子岗也要懂大模型结构？

**答：** 算子是服务模型结构的。Attention、MLP、RMSNorm、MoE、LM Head 等模块会落到不同 kernel 上。只有懂模型结构，才能知道自己优化的 kernel 位于哪个模块，影响 prefill 还是 decode，影响 TTFT 还是 TPOT，以及如何适配不同模型。

---

## 6. 学习 LLaMA、Qwen、DeepSeek 的工程方法是什么？

**答：** 先看论文中的模型结构章节，再看技术解读，用大模型辅助理解概念，最后看 HuggingFace 或官方仓库的模型定义代码，重点调试 forward 流程和 tensor shape。工程岗最终要落到代码，而不是只停留在论文概念。

---

## 7. 大模型端到端推理流程是什么？

**答：** 用户输入文本后，API server 接收请求，tokenizer 将文本转成 token ids，框架构造 request / sequence 对象，scheduler 进行请求调度，KV Cache manager 分配缓存块，模型执行 prefill 构建 KV Cache，然后进入 decode loop，每步读取 KV Cache 并生成 logits，sampling 得到下一个 token，最后 detokenizer 转回文本并返回给用户。

---

## 8. Prefill 和 Decode 有什么区别？

**答：** Prefill 处理输入 prompt 的全部 token，用于构建初始 KV Cache，计算更密集，主要影响 TTFT。Decode 每次只处理新生成的一个 token，同时读取历史 KV Cache，通常更受显存带宽、KV Cache 读取、kernel launch 和调度开销影响，主要影响 TPOT 和吞吐。

---

## 9. 为什么不能把推理过程简单理解为 encoding / decoding？

**答：** 对 decoder-only LLM 来说，在线推理更常用的划分是 tokenization、prefill、decode、sampling、detokenization。encoding / decoding 只描述了模型内部或传统 seq2seq 的一部分，不能覆盖推理框架中的请求调度、KV Cache 管理、batching、并行和资源释放。

---

## 10. vLLM 主要解决什么问题？

**答：** vLLM 是大模型推理服务框架，主要解决在线 serving 中的高吞吐、低延迟和高显存利用率问题。它通过 PagedAttention 管理 KV Cache，通过 continuous batching 动态调度请求，并支持量化、并行、streaming output 等能力。

---

## 11. PagedAttention 解决什么问题？

**答：** PagedAttention 解决 KV Cache 动态分配中的显存碎片和浪费问题。它借鉴操作系统分页思想，把 KV Cache 拆成 block/page，按需分配和释放，使框架在相同显存下支持更多并发请求，从而提升吞吐。

---

## 12. KV Cache 为什么重要？

**答：** 自回归生成中，每生成一个 token 都要关注历史上下文。如果不缓存历史 K/V，每一步都要重新计算所有历史 token，成本很高。KV Cache 保存每层历史 token 的 Key 和 Value，可以显著减少重复计算，但会占用大量显存，因此是推理优化核心。

---

## 13. Continuous batching 为什么能提升吞吐？

**答：** 普通 static batching 要等一批请求一起完成，长请求会拖住短请求。Continuous batching 每轮 decode 后动态移除完成请求、加入新请求，使 GPU 持续保持较高利用率，从而提升吞吐并降低排队延迟。

---

## 14. Sampling 在推理流程中做什么？

**答：** 模型 forward 输出 logits，sampling 根据 temperature、top-k、top-p、repetition penalty 等参数从 logits 中选择下一个 token。它决定生成文本的随机性、多样性和稳定性。

---

## 15. 你写的算子如何影响端到端性能？

**答：** 要看算子处于哪个模块。例如 GEMM 影响 Attention 和 MLP，PagedAttention kernel 影响 decode 阶段 KV Cache 读取，RMSNorm 影响每层小算子开销，Fused MoE 影响 MoE 模型专家计算。需要分别分析它对 TTFT、TPOT、吞吐和显存的影响。

---

## 16. 为什么优化单算子不一定提升端到端性能？

**答：** 单算子可能只占总耗时的一小部分，或者优化后引入额外数据搬运、同步、调度开销。端到端性能还受 tokenizer、scheduler、KV Cache、batching、sampling、通信和其他 kernel 影响。因此需要同时做 micro benchmark 和 end-to-end benchmark。

---

## 17. 面试中计算机基础哪些更重要？

**答：** 对算子和推理框架岗来说，数据结构与算法、计算机组成原理优先级较高。数据结构与算法对应 LeetCode 和框架代码能力；计算机组成对应 GPU 存储层次、cache、访存、并行计算。操作系统中进程线程、虚拟内存、调度也有帮助。计算机网络相对低频，除非岗位偏通信或分布式服务。

---

## 18. 如何从零开始补大模型结构？

**答：** 先从 LLaMA 这类基础开源模型开始，理解 decoder-only、RMSNorm、RoPE、SwiGLU、GQA、KV Cache；再看 Qwen 这种国内常见模型；最后看 DeepSeek 的 MoE、MLA、MTP 等结构。每个模型都按论文、解读、代码、调试四步学习。

---

## 19. 为什么 HuggingFace 代码很重要？

**答：** 因为论文会抽象描述模型结构，但 HuggingFace 代码会直接展示模型 forward 如何实现，Q/K/V shape 如何变化，KV Cache 如何传递，MLP 或 MoE 如何执行。工程岗要能把论文概念落到代码和算子上。

---

## 20. 算子岗是否需要准备 LeetCode？

**答：** 需要。虽然算子岗更常手撕 CUDA kernel，但 LeetCode 仍可能出现，尤其是框架岗或混合岗位。至少要熟练掌握中等难度常见题型，如数组、哈希、双指针、栈队列、树、图、二分、动态规划基础。

---

# 十二、学习行动清单

## 12.1 一周内完成

- 整理现有 CUDA 项目，补充输入规模、CPU baseline、naive CUDA、optimized CUDA 数据；
- 写清楚该算法哪些阶段适合 GPU 加速；
- 准备一张性能对比表；
- 复习 CUDA 基础：thread、block、grid、warp、global memory、shared memory、coalescing、bank conflict。

## 12.2 两周内完成

- 跑通一次 vLLM OpenAI-compatible API 服务；
- 用一个小模型，例如 Qwen 小模型，完成文本生成；
- 跟踪一次 prompt 从输入到输出 token 的流程；
- 画出 tokenizer、scheduler、KV Cache、prefill、decode、sampling 的流程图；
- 背熟 PagedAttention、Continuous Batching、Prefill/Decode 的面试回答。

## 12.3 一个月内完成

- 阅读 LLaMA / Qwen / DeepSeek 中至少两个模型的结构介绍；
- 打开 HuggingFace 模型代码，调试 forward；
- 整理 20 道大模型结构 + 推理框架问答；
- 刷 LeetCode 高频中等题；
- 补计算机组成原理中的 cache、内存层次、流水线、并行计算基础。

## 12.4 简历修改重点

你的简历项目应该体现：

```text
问题分析能力
  + CUDA 实现能力
  + 性能 benchmark 能力
  + 推理框架端到端理解
  + 大模型结构基础
```

不要只写“会 CUDA”“会 Transformer”“了解 vLLM”，而要写出实验数据和工程链路。

---

# 十三、最终总结

这节课真正想告诉求职者的是：

> **AI Infra 面试不是孤立考 CUDA、孤立考 Transformer、孤立考 LeetCode，而是看你能不能把模型结构、推理流程、算子优化、性能指标和项目表达串成一个完整工程闭环。**

对于准备算子开发或推理框架岗位的人，最关键的补短板顺序是：

```text
第一，现有 CUDA 项目要量化、工程化、可解释；
第二，补 LLaMA / Qwen / DeepSeek 等现代大模型结构；
第三，补 vLLM 端到端推理流程；
第四，准备 PagedAttention、KV Cache、Prefill/Decode、Continuous Batching 等高频八股；
第五，按优先级准备数据结构算法和计算机组成原理。
```

这节课标题里的“端到端是必须掌握的”非常准确。你可以只把某个算子写得很熟，但如果不知道它在大模型推理系统中的位置，就很难在 AI Infra 面试里形成完整竞争力。

最终目标不是“背很多名词”，而是做到：

```text
用户输入一句话
  → 我知道它如何变成 token ids
  → 我知道请求如何被 scheduler 调度
  → 我知道 KV Cache 如何申请和复用
  → 我知道 prefill 和 decode 分别跑什么
  → 我知道哪些 kernel 在其中发挥作用
  → 我知道我的 CUDA 优化影响哪个性能指标
  → 我能用数据证明优化确实有效
```

做到这个程度，才算真正建立了 AI Infra 推理侧的工程理解。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-AI infra学习路线|模块-AI infra学习路线]]

%% 项目关联导航：结束 %%
