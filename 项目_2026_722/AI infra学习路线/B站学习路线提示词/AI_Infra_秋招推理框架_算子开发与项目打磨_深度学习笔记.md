# AI Infra 规划课与模拟面学习笔记  
## 2026 年 6 月 27 日：秋招推理框架、算子开发与项目打磨指导

> 输入材料：`AI-Infra规划课和模拟面_2026年6月27日_秋招推理框架_算子开发与项目打磨指导_哔哩哔哩_bilibili.txt`  
> 处理方式：自动字幕纠错、语义复原、技术主线重组、面试视角分析。  
> 说明：原字幕来自自动识别，存在大量错别字、断句错误和技术名词误识别。本文不是逐字稿，而是面向 AI Infra / 大模型推理 / CUDA 算子 / 秋招面试准备的深度整理笔记。

---

## 目录

1. [这节课的核心结论](#1-这节课的核心结论)  
2. [字幕技术名词纠错表](#2-字幕技术名词纠错表)  
3. [课程整体结构](#3-课程整体结构)  
4. [按课程推进顺序整理知识点](#4-按课程推进顺序整理知识点)  
5. [推理框架端到端流程：从输入到输出](#5-推理框架端到端流程从输入到输出)  
6. [项目打磨方法论](#6-项目打磨方法论)  
7. [CUDA / Triton 算子学习与面试准备](#7-cuda--triton-算子学习与面试准备)  
8. [vLLM / nano-vLLM / mini-vLLM 学习路线](#8-vllm--nano-vllm--mini-vllm-学习路线)  
9. [多模态推理框架项目分析](#9-多模态推理框架项目分析)  
10. [端侧 / 机器人 / 云端 AI Infra 的区别](#10-端侧--机器人--云端-ai-infra-的区别)  
11. [面试高频问题与参考答案](#11-面试高频问题与参考答案)  
12. [给秋招准备者的执行路线](#12-给秋招准备者的执行路线)  
13. [最终总结](#13-最终总结)

---

# 1. 这节课的核心结论

这节课本质上是一场 **AI Infra 秋招模拟面 + 简历项目打磨课**。主讲人围绕几位同学的真实项目，重点讲了三个问题：

1. **推理框架项目如何从“实现了功能”打磨成“有优化、有实验、有面试说服力”的项目。**
2. **算子开发岗位到底要准备哪些 CUDA / Triton / C++ / LeetCode 内容。**
3. **AI Infra 学习不能只停留在单算子或单模型，而要理解大模型推理端到端流程。**

这节课最重要的判断是：

> **秋招 AI Infra 项目不是写几个功能就够了，而是必须能解释：为什么做、怎么做、优化了什么、实验数据如何、瓶颈在哪里、下一步还能怎么改。**

另一个重要结论是：

> **无论是推理框架岗还是算子开发岗，候选人都不能只懂局部。只懂单个 CUDA kernel 不够，只会部署模型也不够，必须理解从请求进入推理服务到 token 输出的完整链路。**

---

# 2. 字幕技术名词纠错表

| 字幕识别结果 | 建议修正 | 说明 |
|---|---|---|
| OC / OCM | OCR | 多处语境是图像文字识别 |
| 英勇 / 英佛 / Info | Infra / AI Infra | 指 AI 基础设施方向 |
| 拍头 / 拍头C | Paddle / PaddleOCR / PyTorch，需按上下文判断 | 字幕不稳定，OCR 项目语境更像 PaddleOCR |
| 避刑推理 | 并行推理 | 多 GPU 并行执行 |
| 第四月 / 地匹并行 | DP 并行 / Data Parallel | 多卡上复制模型、分发请求 |
| 一批 | EP / Expert Parallel 或 Batch，按上下文判断 | MoE 场景中可能是 EP |
| PVCache / PVCatch / KVCatch | KV Cache | 大模型推理缓存 |
| Profeo / Purefield / Profill | Prefill | 处理 prompt 的阶段 |
| Decode 阶段突补货 | Decode 阶段吞吐优化 / CUDA Graph / Speculative Decoding，需按上下文判断 | 字幕无法完全确认 |
| Channel Profeo | Chunked Prefill | 分块预填充 |
| APP / P8 | FP8 | FP8 KV Cache |
| GPT扣 / AW扣 | GPTQ / AWQ | 量化方法 |
| PD分离 | Prefill-Decode Disaggregation | 预填充和解码分离 |
| 钱质匹配 | 前缀匹配 / Prefix Caching | 复用相同前缀 KV Cache |
| 确诊 / Tritan / 拆盘 | Triton | GPU kernel DSL |
| 私家家 | C++ | 多处语境是 C++ 八股 |
| SALTMAX / Sautomus | Softmax | 高频手撕算子 |
| AMSLOM / RMsnom | RMSNorm | 归一化算子 |
| RDOS / Reduce | Reduce | 归约算子 |
| Mathomal / MathemaFlexion | MatMul / GEMM | 矩阵乘 |
| Flyce Tenshin | FlashAttention | Attention 优化算法 |
| 叉板 / 差点 | TensorRT Plugin | TensorRT 自定义插件 |
| 森马娃 | 可能是算能 / Sophon / 某嵌入式平台 | OCR 项目中的端侧平台 |
| 杰斯 | Jetson | NVIDIA Jetson 边缘计算平台 |
| 哈根费子 | Hugging Face | 模型代码和权重生态 |
| 加码3 / 伽马3 | Gemma 3 或同类多模态模型，ASR 不确定 | 多模态模型适配语境 |
| 千万三 / 千万三点五 | Qwen3 / Qwen2.5-VL / Qwen 系列，ASR 不确定 | 多模态模型适配语境 |
| Gating / 基丁网络 | Gated DeltaNet / Gated 网络，ASR 不确定 | 新型替代 Attention 的结构 |
| 线性增长 | 线性复杂度 | 相比 Attention 的二次复杂度 |

---

# 3. 课程整体结构

这节课按模拟面试同学可以分成五大段。

## 3.1 第一位同学：OCR / mini-VLM / 推理框架项目打磨

核心讨论点：

- OCR 或图像处理项目如何和 AI Infra 关联；
- 多 GPU 推理服务不能只写“信号量 + 轮询”，要体现调度优化；
- mini-VLM / mini-vLLM 项目可以继续加功能，但要选对方向；
- Chunked Prefill、KV Cache 量化、FP8 KV Cache、AWQ、Prefix Cache、PD 分离等功能的工作量和性价比；
- 已实现功能必须补实验数据，否则面试说服力不足。

## 3.2 第二位同学：研一/研二初学者如何入门推理框架

核心讨论点：

- 刚开始不建议直接啃完整 vLLM；
- 可以先看 nano-vLLM / mini-vLLM；
- 先吃透 PagedAttention、Continuous Batching、Prefix Caching 等核心机制；
- Transformer 和主流模型结构需要补；
- CUDA 不能只看懂，要能手写常见算子。

## 3.3 第三位同学：算子开发实习 / Triton 算子库 / 简历问题

核心讨论点：

- 简历不能把别人做的算子都写成自己的；
- 写太多算子等于给面试官立很多靶子；
- Softmax 这种基础 Triton/CUDA 算子手撕不出来非常危险；
- C++ 八股和 CUDA 能力是联动的；
- 算子项目要写清楚单算子性能提升和实验数据。

## 3.4 第四位同学：YOLO / TensorRT / 端侧机器人项目

核心讨论点：

- TensorRT Plugin 的合理性必须解释清楚；
- TensorRT 原生支持的算子，为什么要自定义实现，必须有实验支撑；
- YOLO 部署项目要写清楚平台、模型、吞吐、延迟、显存、插件优化；
- 边缘推理和云端推理侧重点不同；
- AI 替代算子开发不是简单替代，而是人机协同。

## 3.5 第五位同学：多模态 mini-vLLM 项目

核心讨论点：

- 多模态模型适配的工作量和价值较高；
- 视觉编码器、投影层、图像 token 替换、KV Cache 分组、Prefix Cache 图像哈希等是关键；
- 多模态推理框架项目比普通“跑通 vLLM”更有差异化；
- 新型 Gated / Linear Attention 类结构可能会挑战传统 Attention，但短期无法判断完全替代时间。

---

# 4. 按课程推进顺序整理知识点

## 4.1 OCR / 图像处理项目：只做应用还不够，要挖推理优化点

### 核心内容

第一位同学的实习项目是 OCR / 图像处理相关，做了多 GPU 并行推理调度。主讲人指出，这类项目容易偏应用层，和 AI Infra / 推理框架岗位的关联不够强，需要继续挖掘推理服务、调度、batch、负载均衡等系统优化点。

### 关键概念解释

#### 多 GPU 并行推理

多 GPU 推理可以有几种形式：

1. **Data Parallel / DP**
   - 每张 GPU 部署一份完整模型；
   - 不同请求分发到不同 GPU；
   - 适合模型能放进单卡的场景。

2. **Tensor Parallel / TP**
   - 模型权重切到多张 GPU；
   - 一个请求需要多卡协同完成；
   - 适合单卡放不下的大模型。

3. **Expert Parallel / EP**
   - MoE 模型中不同专家放到不同 GPU；
   - token 根据 router 分发到专家；
   - 会引入 All-to-All 通信。

OCR 项目中模型较小，因此更像是 DP 式多卡请求分发，不是 vLLM 里常见的大模型 TP/EP。

#### 信号量调度

同学实现的是类似：

```text
GPU 数量 = 信号量初值
请求到来 → 尝试获取空闲 GPU → 成功后修改 GPU 状态 → 向后端微服务发请求 → 完成后释放 GPU
```

这个方案能跑通，但推理优化味道不够强，因为它只是“资源占用控制”，没有深入考虑：

- 每张 GPU 当前排队长度；
- 每张 GPU 当前显存占用；
- 请求输入大小差异；
- batch 合并机会；
- 不同 GPU 实际处理速度；
- P95 / P99 延迟。

### 在 AI Infra 中的作用

真实推理服务的调度不是简单找空闲 GPU，而是要在吞吐、延迟、公平性、显存、负载均衡之间取平衡。项目要更像 Infra，需要从“能调度”升级成“能优化调度”。

### 项目打磨建议

可以增加两个优化点：

1. **基于负载预测的 GPU 分发**
   - 不只看 GPU 是否空闲；
   - 还看当前排队长度、平均处理耗时、显存占用；
   - 将请求发给预计完成时间最短的 GPU。

2. **请求合并成 batch**
   - 多个单图请求如果短时间内到达，可以合并成 batch；
   - 对 CNN/OCR/视觉模型，batch 推理通常能提升 GPU 利用率；
   - 需要实验对比 batch=1、2、4、8 的吞吐和延迟。

---

## 4.2 推理框架项目：实现功能之后必须补实验

### 核心内容

同学在 mini-VLM / mini-vLLM 项目中加入了功能，例如 KV Cache 量化、Chunked Prefill、TP/EP 支持、模型支持等。主讲人强调：只写“实现了某功能”不够，必须补实验数据。

### 为什么实验重要

面试官会问：

- 你为什么实现这个功能？
- 这个功能解决什么瓶颈？
- 优化前后指标是多少？
- 对 TTFT / TPOT / 吞吐 / 显存有什么影响？
- 有没有副作用？
- 适用于什么场景，不适用于什么场景？

如果没有实验，只能现场编，很容易露怯。

### 应该补哪些实验

#### Decode 阶段优化实验

如果做了 Decode 阶段优化，需要测试：

| 指标 | 说明 |
|---|---|
| TPOT | 每个输出 token 的平均耗时 |
| Decode throughput | 每秒输出 token 数 |
| GPU 利用率 | 是否提高 SM 利用率 |
| 显存占用 | 是否有额外开销 |
| batch size 敏感性 | 并发变化下是否有效 |

#### Chunked Prefill 实验

如果做了 Chunked Prefill，需要测试：

| 场景 | 测试重点 |
|---|---|
| 短 prompt | 是否没有明显额外开销 |
| 长 prompt | 是否降低排队阻塞 |
| prefill + decode 混合负载 | 是否改善 decode 请求尾延迟 |
| 不同 chunk size | chunk 太大/太小的影响 |

#### KV Cache 量化实验

需要测试：

| 指标 | 说明 |
|---|---|
| KV Cache 显存下降比例 | 例如 FP16 → INT8 / FP8 |
| 精度影响 | perplexity、输出一致性、任务准确率 |
| 解码速度 | 反量化是否引入额外开销 |
| 长上下文收益 | 长序列下显存收益更明显 |

---

## 4.3 mini-vLLM 项目功能选择：PD 分离不适合短期硬做

### 核心内容

同学想继续在 mini-vLLM 项目中加入 PD 分离、AWQ、Prefix Cache 命中率优化等功能。主讲人认为：**PD 分离工作量太大，不建议短期秋招前硬做；AWQ 相对更合适。**

### PD 分离是什么

PD 分离是 Prefill-Decode Disaggregation，即把 Prefill 和 Decode 阶段在系统层面拆开：

```text
Prefill 节点：处理长 prompt，生成 KV Cache
Decode 节点：接收 KV Cache，逐 token 生成
```

### 为什么 PD 分离工作量大

因为它不是加一个小模块，而是侵入整个推理系统：

1. **调度层要改**
   - 请求要先分配到 prefill 节点；
   - 再把 decode 阶段调度到 decode 节点。

2. **KV Cache 传输要做**
   - prefill 结果 KV Cache 需要跨节点或跨进程传输；
   - 需要处理 KV Cache 地址映射和生命周期。

3. **状态管理要改**
   - decode 节点必须知道 prefill 已经处理到哪里；
   - 请求状态跨节点同步。

4. **性能验证复杂**
   - 要证明 PD 分离真的带来收益，需要多机/多卡/混合负载实验。

### 为什么 AWQ 更适合短期项目

AWQ 是权重量化方法，通常是 W4A16 这类模式。相比 PD 分离，它更适合在已有量化模块基础上扩展：

```text
已有 W8A16 量化
  → 增加 W4A16 / AWQ 权重加载
  → 增加反量化路径
  → 做精度和显存实验
```

它工作边界更清晰，也更容易在简历中讲明白。

---

## 4.4 初学者路线：先 nano-vLLM，再完整 vLLM

### 核心内容

第二位同学基础较弱，主讲人建议不要一上来直接读完整 vLLM，而是先学习 nano-vLLM / mini-vLLM。

### 为什么不要直接啃 vLLM

完整 vLLM 的复杂度来自：

- API Server；
- Engine / EngineCore；
- Scheduler；
- KV Cache Manager；
- Worker；
- ModelRunner；
- Attention Backend；
- 分布式并行；
- 量化；
- CUDA Graph；
- Speculative Decoding；
- 多模型支持；
- 大量工程兼容逻辑。

初学者一上来读完整项目，很容易被工程细节淹没，反而抓不住核心。

### nano-vLLM / mini-vLLM 的价值

简化版项目通常保留核心机制：

- 请求状态管理；
- PagedAttention；
- KV Cache block 管理；
- Continuous Batching；
- Prefill / Decode；
- Scheduler；
- ModelRunner；
- 采样；
- 输出 token 回传。

学习顺序应该是：

```text
先跑通 demo
  → 理解请求对象 Sequence / Request
  → 理解 Scheduler 如何组 batch
  → 理解 BlockManager 如何分配 KV Cache
  → 理解 ModelRunner 如何执行 prefill/decode
  → 理解采样如何生成下一个 token
  → 再看完整 vLLM 架构
```

---

## 4.5 大模型结构学习：论文 + Hugging Face 代码 + 调试

### 核心内容

很多同学只知道 Transformer 的大概结构，但不了解 LLaMA、Qwen、DeepSeek、MoE、GQA、MLA 等现代模型。主讲人建议用“论文 + 开源代码 + 调试”的方式补模型结构。

### 为什么 AI Infra 也要懂模型结构

推理优化服务的是模型。如果不知道模型结构，就不知道：

- 哪些层是 Attention；
- 哪些层是 MLP；
- 是否使用 GQA/MQA；
- 是否使用 MoE；
- KV Cache 结构如何；
- RoPE 如何应用；
- 量化应该作用在哪些权重；
- 哪些算子是瓶颈。

### 推荐学习方法

```text
第一步：读论文结构部分
第二步：看博客/解读辅助理解
第三步：看 Hugging Face modeling_xxx.py
第四步：用 debugger 跑一遍 forward
第五步：把每个模块和推理框架执行流程对应起来
```

重点模型：

- LLaMA 系列；
- Qwen 系列；
- DeepSeek 系列；
- MoE 模型；
- 多模态模型，例如 Qwen-VL、Gemma 多模态版本等。

---

## 4.6 CUDA / Triton 算子：看懂不够，必须能手写

### 核心内容

主讲人反复强调：如果想投算子开发，不能只“看懂代码”，而要能流畅手写常见算子。

### 高频手撕算子

优先级最高：

1. **Softmax**
2. **Reduce**
3. **RMSNorm / LayerNorm**
4. **MatMul / GEMM**
5. **Transpose**

有一定概率被问：

6. **TopK**
7. **FlashAttention 简化版**
8. **NMS**
9. **MoE routing / combine 简化版**

### 为什么 Softmax 非常危险

Softmax 是 Triton 官方教程和 CUDA 面试中的高频基础题。若面试中写不出来，会让面试官怀疑：

- 是否真的写过 kernel；
- 是否只会用 AI 生成代码；
- 是否只做过表层项目；
- 是否缺乏 GPU 编程基本训练。

### MatMul 要掌握到什么程度

不一定要求写出 CUTLASS 级别性能，但至少要会：

- naive MatMul；
- shared memory tiling；
- block tiling；
- coalesced memory access；
- 减少 global memory 访问；
- 简单向量化；
- Tensor Core 原理知道；
- 能解释 compute-bound / memory-bound。

---

## 4.7 简历中算子不要写太多，否则会被反向攻击

### 核心内容

有同学把自己参与项目中的 30 个算子都写到简历上。主讲人指出：这非常危险。

### 为什么危险

简历写什么，面试官就可以问什么。如果写了 30 个算子：

```text
面试官随便抽一个你不熟的算子
  → 你回答不上来
  → 面试官认为项目不真实或参与不深
```

这就是“给自己立靶子”。

### 正确写法

应该筛选 2–4 个最核心、最熟悉、有优化数据的算子：

- Softmax；
- RMSNorm；
- MatMul；
- MoE aux loss / MoE routing；
- 某个融合算子；
- 某个实际大模型推理中高频算子。

每个算子都要准备：

1. 原始实现；
2. 优化版本；
3. 优化策略；
4. benchmark 结果；
5. Nsight / profiling 分析；
6. 和 PyTorch / Triton / CUDA baseline 的对比；
7. 精度校验。

---

## 4.8 C++ 八股和 CUDA 能力有关联

### 核心内容

有同学 C++ 八股回答不好，CUDA 手撕也不稳。主讲人指出：C++ 和 CUDA 能力通常是相关的。

### 为什么相关

CUDA kernel 本身通常是 CUDA C/C++。如果候选人 C++ 基础很弱，面试官会担心：

- 不能维护复杂 GPU 工程；
- 不理解内存模型；
- 不理解指针、引用、RAII；
- 不理解模板和编译期机制；
- 不理解性能敏感代码的写法。

### 推荐资料

主讲人推荐了《深入探索 C++ 对象模型》。这本书适合补：

- 虚函数表；
- 继承；
- 多态；
- 对象内存布局；
- 构造析构；
- C++ 编译模型。

但秋招时间紧时，建议以高频八股为主：

- 指针和引用；
- const；
- static；
- 虚函数；
- 多态；
- 内存布局；
- new/delete；
- 智能指针；
- STL vector/map/unordered_map；
- 左值右值；
- move 语义；
- 线程基础。

---

## 4.9 AI 会不会替代算子开发

### 核心内容

多位同学担心：AI 已经能生成 CUDA / Triton 算子，算子开发是不是没价值了？

主讲人的观点是：

> **AI 会显著提高算子开发效率，但短期不是完全替代。算子工程师的价值从“手写代码”转向“需求拆解、性能分析、优化决策、验收维护”。**

### 公司真实算子开发流程

可以抽象成：

```text
1. 需求拆解
   ↓
2. 开发周期评估
   ↓
3. 初版 kernel 开发
   ↓
4. 精度校验
   ↓
5. profiling 分析
   ↓
6. 选择优化方向
   ↓
7. 迭代优化
   ↓
8. 验收与维护
```

AI 最擅长的是第 3 步中的“生成初版代码”，也能辅助第 6 步给建议。但以下内容仍需要人：

- 判断哪些算子真的要重写；
- 判断哪些可以复用已有 kernel；
- 设计 benchmark；
- 解释 profiling 指标；
- 在多个优化方向中做取舍；
- 解决边界 case；
- 承担线上稳定性责任；
- 精度验收和长期维护。

所以不要把 AI 看成“替代你的人”，更应该把它看成“提高你产出的工具”。

---

## 4.10 YOLO / TensorRT 项目：自定义 Plugin 必须有合理动机

### 核心内容

有同学做 YOLO 模型在边缘平台部署，并写了 TensorRT 自定义插件。主讲人指出：TensorRT 本身支持 Conv2D，如果你重写基础算子，必须解释原因。

### 面试官会问

- TensorRT 原生就支持卷积，为什么要写 plugin？
- 你的 plugin 比 TensorRT 原生实现快吗？
- 你的 plugin 针对 YOLO 的哪个特殊结构？
- 是否做了算子融合？
- 是否减少了显存访问？
- 是否支持 dynamic shape？
- FP16 / INT8 下性能如何？
- 精度是否一致？

### 项目应补数据

| 指标 | 说明 |
|---|---|
| FPS | 峰值吞吐，例如 100 FPS / 140 FPS |
| latency | 单帧延迟 |
| P95 latency | 稳定性 |
| GPU memory | 显存占用 |
| TensorRT native baseline | 原生实现性能 |
| custom plugin | 自定义插件性能 |
| accuracy | mAP / 识别准确率 |

### 简历写法建议

不要只写：

> 实现 TensorRT 自定义 Conv2D Plugin。

应该写成：

> 针对 YOLO 模型在 Jetson / 嵌入式平台部署中的特定计算瓶颈，设计 TensorRT 自定义 Plugin，并与 TensorRT 原生算子进行 benchmark 对比；在 FP16 模式下对比单帧 latency、吞吐和显存占用，验证 plugin 在特定输入 shape 下的收益。

---

# 5. 推理框架端到端流程：从输入到输出

这节课反复强调：AI Infra 候选人必须理解大模型推理端到端流程。下面用 vLLM / mini-vLLM 风格串起来。

## 5.1 用户请求进入服务

```text
用户输入文本 / 对话消息
  ↓
HTTP 请求
  ↓
API Server / OpenAI-compatible Server
```

这一层负责：

- 接收请求；
- 解析 prompt；
- 解析 sampling 参数；
- 处理 stream / non-stream；
- 鉴权、限流、错误处理等工程逻辑。

## 5.2 Tokenizer 编码

```text
文本 prompt
  ↓
Tokenizer
  ↓
token ids
```

模型不能直接处理字符串，只能处理 token id。Tokenizer 把文本映射为整数序列。

## 5.3 构造请求对象

```text
token ids + sampling params
  ↓
Request / Sequence / SequenceGroup
```

请求对象通常包含：

- prompt token ids；
- 已生成 token；
- max tokens；
- temperature；
- top-p；
- top-k；
- stop token；
- 当前状态：waiting / running / finished；
- KV Cache block 信息。

## 5.4 Scheduler 调度

```text
waiting queue
  ↓
Scheduler
  ↓
本轮要执行的 batch
```

Scheduler 要决定：

- 哪些请求进入 prefill；
- 哪些请求进入 decode；
- prefill token 数是否超过上限；
- KV Cache block 是否足够；
- 长 prompt 是否 chunked prefill；
- decode 请求是否优先；
- 已完成请求是否释放资源。

## 5.5 KV Cache 分配

```text
请求 token
  ↓
BlockManager / KV Cache Manager
  ↓
分配 KV blocks
```

KV Cache 是推理框架的核心显存资源。PagedAttention 类机制把 KV Cache 拆成 block 管理，避免为每个请求预留最大长度显存。

## 5.6 Prefill 阶段

```text
prompt tokens
  ↓
model forward
  ↓
生成 prompt 部分 KV Cache
  ↓
得到首 token logits
```

Prefill 特点：

- 一次处理多个 prompt token；
- 计算密集；
- 长 prompt 可能非常耗时；
- 影响 TTFT；
- 适合矩阵乘并行。

Chunked Prefill 会把长 prompt 拆分，避免长请求阻塞 decode。

## 5.7 Decode 阶段

```text
上一步生成的 token
  ↓
读取历史 KV Cache
  ↓
model forward
  ↓
得到下一个 token logits
```

Decode 特点：

- 每轮通常只为每个请求生成一个 token；
- 强依赖 KV Cache；
- 访存压力大；
- 容易受 kernel launch、调度、显存带宽影响；
- 影响 TPOT 和输出吞吐。

## 5.8 Sampling

```text
logits
  ↓
temperature / top-k / top-p / repetition penalty
  ↓
next token id
```

Sampling 决定模型输出的随机性和多样性。

## 5.9 输出回传

```text
next token id
  ↓
detokenizer
  ↓
文本 token
  ↓
stream 返回给用户
```

如果是流式输出，每生成一个 token 或若干 token 就返回给客户端。

## 5.10 资源释放

```text
请求完成
  ↓
释放 KV Cache block
  ↓
从 running 队列移除
  ↓
新请求补位
```

这就是 Continuous Batching 能提升吞吐的原因：完成的请求释放资源，新请求持续加入。

---

# 6. 项目打磨方法论

## 6.1 一个 AI Infra 项目的标准叙述结构

简历和面试中讲项目，建议按下面结构：

```text
1. 背景：为什么要做这个项目？
2. 问题：原始系统有什么瓶颈？
3. 分析：怎么定位瓶颈？
4. 方案：你做了什么设计？
5. 实现：关键代码/模块怎么实现？
6. 实验：优化前后数据如何？
7. 反思：局限和后续优化方向是什么？
```

## 6.2 不要只写“支持了某功能”

错误写法：

> 支持 Chunked Prefill，支持 KV Cache 量化，支持 AWQ。

正确写法：

> 针对长 prompt prefill 阶段阻塞 decode 请求的问题，实现 Chunked Prefill，将长 prompt 拆分为固定 token 数的 chunk，并在 scheduler 中与 decode 请求混合调度；在长短请求混合负载下，统计 TTFT、TPOT 和 P95 latency，对比 chunk size 对尾延迟的影响。

## 6.3 实验数据优先级

建议至少准备以下数据：

| 类型 | 指标 |
|---|---|
| 延迟 | average latency、P95、P99 |
| 首 token | TTFT |
| 输出速度 | TPOT、tokens/s |
| 吞吐 | QPS、requests/s |
| 显存 | peak memory、KV Cache memory |
| 精度 | 输出一致性、任务准确率、perplexity |
| 可扩展性 | batch size、并发数、prompt length、output length |

## 6.4 项目要避免的问题

1. **没有 baseline**
   - 不知道和谁比，就无法证明优化。

2. **只有端到端数据，没有模块数据**
   - 如果优化的是单算子，要补单算子耗时；
   - 如果优化的是调度，要补排队时间、P95 延迟。

3. **只写功能，不写收益**
   - 面试官无法判断你做得深不深。

4. **写了不熟悉的内容**
   - 简历内容必须能经得起追问。

5. **把工具生成内容当成自己的理解**
   - 面试中很容易被反问击穿。

---

# 7. CUDA / Triton 算子学习与面试准备

## 7.1 算子岗必备能力层级

### 第一层：基础 CUDA

必须掌握：

- thread / block / grid；
- warp；
- global memory；
- shared memory；
- register；
- memory coalescing；
- bank conflict；
- warp divergence；
- occupancy；
- atomic；
- synchronization；
- CUDA stream；
- CUDA event。

### 第二层：基础算子手写

必须能写：

- vector add；
- reduce；
- softmax；
- layernorm / rmsnorm；
- transpose；
- matmul。

### 第三层：优化思路

能说清：

- tiling；
- shared memory 缓存；
- 向量化加载；
- 减少访存；
- 减少同步；
- 避免 bank conflict；
- 使用 warp-level primitive；
- 使用 Tensor Core；
- 算子融合。

### 第四层：性能分析

能用：

- Nsight Compute；
- Nsight Systems；
- CUDA event；
- roofline 分析；
- compute-bound / memory-bound 判断。

### 第五层：工程落地

能完成：

- PyTorch extension；
- Triton kernel；
- 单元测试；
- 精度对齐；
- benchmark；
- 接入推理框架。

---

## 7.2 Softmax 手撕要点

Softmax 公式：

```text
softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
```

优化要点：

1. 先做 row max，保证数值稳定；
2. 再做 exp 和 sum；
3. 最后归一化；
4. 一行通常由一个 block 或一个 program 处理；
5. 使用 shared memory 或 warp reduce；
6. 尽量 coalesced load/store；
7. 避免多次访问 global memory。

面试回答重点：

> Softmax 是典型 memory-bound 算子，核心优化方向是减少 global memory 读写、利用 shared memory / warp-level reduce 完成 max 和 sum，并保证数值稳定。

---

## 7.3 Reduce 手撕要点

Reduce 是很多算子的基础，例如 sum、max、norm。

优化要点：

- block 内归约；
- shared memory tree reduction；
- warp-level reduction；
- 减少分支；
- 循环展开；
- 多元素 per thread；
- 避免 bank conflict。

面试官喜欢问 Reduce，因为它简单但能考察 CUDA 基础是否扎实。

---

## 7.4 RMSNorm / LayerNorm 手撕要点

RMSNorm 公式：

```text
y = x / sqrt(mean(x^2) + eps) * weight
```

优化要点：

1. 对一行 hidden dimension 做平方和；
2. reduce 求 mean；
3. rsqrt；
4. 乘 weight；
5. 尽量向量化读取；
6. hidden size 固定时可以做模板优化。

RMSNorm 在 LLM 中非常常见，因此是大模型推理算子岗位高频题。

---

## 7.5 MatMul / GEMM 手撕要点

naive 版本：

```text
C[i, j] = sum_k A[i, k] * B[k, j]
```

优化路径：

1. naive global memory；
2. shared memory tiling；
3. block tiling；
4. thread tile；
5. register reuse；
6. vectorized load；
7. Tensor Core / WMMA；
8. pipeline / double buffering；
9. CUTLASS 思想。

面试中不一定要写最高性能版本，但必须能写出 shared memory tiling 版本并解释为什么减少 global memory 访问。

---

## 7.6 Triton 面试准备

Triton 常见问题：

1. Triton 和 CUDA 的区别是什么？
2. Triton program id 对应 CUDA 中什么概念？
3. block pointer / mask 是什么？
4. `tl.load` 和 `tl.store` 如何处理越界？
5. 为什么 Triton 写 Softmax 很方便？
6. Triton 性能为什么可能接近 CUDA？
7. Triton 不能完全替代 CUDA 的原因是什么？

Triton 的定位：

> Triton 是 Python-like GPU kernel DSL，适合快速实现和迭代大模型算子；CUDA 对底层控制更强，适合极致性能优化。

---

# 8. vLLM / nano-vLLM / mini-vLLM 学习路线

## 8.1 学习优先级

对于秋招准备，建议按下面顺序：

```text
1. Transformer / LLM 基础
2. nano-vLLM / mini-vLLM 核心机制
3. vLLM 官方文档和架构
4. vLLM 高频八股
5. 做一个可讲清楚的小功能
6. 补 benchmark 和实验
```

## 8.2 nano-vLLM 必学模块

| 模块 | 要理解的问题 |
|---|---|
| LLM / Engine | 用户请求如何进入系统 |
| Sequence | 请求状态如何表示 |
| Scheduler | waiting/running 如何调度 |
| BlockManager | KV Cache block 如何分配 |
| ModelRunner | prefill/decode 如何执行 |
| Sampler | logits 如何变成 token |
| Tokenizer | 文本和 token 如何互转 |
| KV Cache | 为什么需要缓存历史 K/V |

## 8.3 vLLM 必学机制

1. PagedAttention；
2. Continuous Batching；
3. Chunked Prefill；
4. Prefix Caching；
5. Speculative Decoding；
6. CUDA Graph；
7. Quantization；
8. Tensor Parallel；
9. Expert Parallel；
10. Scheduler；
11. Worker / GPU Worker；
12. ModelRunner；
13. KV Cache Manager。

## 8.4 不建议短期硬做的功能

秋招时间紧时，不建议直接做：

- 完整 PD 分离；
- 完整多机多卡调度；
- 完整高性能 FlashAttention；
- 大规模 MoE EP；
- 从零实现完整 vLLM。

原因是工作量太大，容易做成半成品。

## 8.5 更适合短期做的功能

更推荐：

- 增加 benchmark；
- 打印 prefill/decode 时间；
- 实现简单 Prefix Cache；
- 实现 KV Cache 量化；
- 支持一个小模型；
- 支持 AWQ / GPTQ 的加载路径；
- 实现 Chunked Prefill 简化版；
- 对 scheduler 做一个小优化；
- 对 BlockManager 做统计和可视化。

---

# 9. 多模态推理框架项目分析

## 9.1 多模态模型推理流程

多模态模型通常不是只输入文本，而是输入：

```text
图像 + 文本 prompt
```

端到端流程：

```text
图像
  ↓
图像预处理 resize / normalize
  ↓
Vision Encoder
  ↓
视觉特征
  ↓
Projector 映射到语言模型 hidden size
  ↓
替换 prompt 中的 image tokens
  ↓
进入 LLM decoder
  ↓
生成文本
```

## 9.2 多模态适配难点

### 视觉编码器适配

需要理解：

- 图像如何切 patch；
- vision encoder 输出什么；
- 输出维度如何对齐 LLM hidden size。

### 图像 token 替换

prompt 中会有特殊 image token：

```text
<image> ... <image>
```

这些 token 对应的位置需要替换成视觉特征 embedding。

### KV Cache 管理

多模态模型中，KV Cache 可能包含：

1. 文本 token 的 KV；
2. 图像 token 的 KV；
3. 特殊结构产生的额外状态缓存。

如果模型引入 Gated / Linear Attention 类结构，还可能需要额外 cache，不再是传统 Attention 的 K/V。

### Prefix Cache 图像哈希

文本 prefix 可以通过 token ids 做哈希。图像 prefix 更复杂：

- 同一张图像重复请求可以复用；
- 不同图像不能误复用；
- 图像原始数据很大，不适合在进程间反复拷贝；
- 可以对预处理后的图像或图像特征做 hash；
- hash 计算要避免引入过高开销。

## 9.3 为什么多模态 mini-vLLM 项目有价值

相比普通“跑通 vLLM”，多模态适配更有差异化，因为它涉及：

- 模型结构理解；
- vision encoder；
- projector；
- multimodal token merge；
- prefix caching 改造；
- KV Cache 结构变化；
- 新模型适配；
- Hugging Face 源码阅读；
- 框架扩展能力。

这类项目如果做扎实，面试含金量高。

---

# 10. 端侧 / 机器人 / 云端 AI Infra 的区别

## 10.1 端侧 / 机器人 AI Infra

典型场景：

- Jetson；
- 嵌入式 GPU；
- 机器人视觉；
- OCR；
- YOLO；
- 传感器融合；
- 低功耗设备；
- 单卡或少量 GPU。

关注点：

- 单帧延迟；
- FPS；
- 模型压缩；
- TensorRT；
- ONNX；
- INT8 量化；
- 端侧显存；
- 功耗；
- 稳定性；
- 与传感器/控制系统联动。

## 10.2 云端大模型 AI Infra

典型场景：

- 多机多卡集群；
- vLLM / SGLang；
- LLM Serving；
- 高并发请求；
- 长上下文；
- 多租户；
- KV Cache 管理；
- TP / PP / DP / EP；
- 大模型量化；
- 在线服务监控。

关注点：

- TTFT；
- TPOT；
- tokens/s；
- QPS；
- P95/P99；
- GPU 利用率；
- KV Cache 显存；
- 调度公平性；
- 集群成本。

## 10.3 两者的核心区别

| 维度 | 端侧/机器人 | 云端大模型 |
|---|---|---|
| 部署规模 | 单设备/单卡/少量卡 | 多卡/多机/集群 |
| 模型类型 | YOLO、OCR、小模型、多模态感知 | LLM、MoE、多模态大模型 |
| 指标 | FPS、单帧延迟、功耗 | TTFT、TPOT、吞吐、P99 |
| 框架 | TensorRT、ONNX Runtime、OpenVINO | vLLM、SGLang、TensorRT-LLM |
| 调度 | 任务级/设备级 | 请求级/token级/KV级 |
| 优化重点 | 模型压缩、算子融合、端侧部署 | KV Cache、batching、并行、调度 |

---

# 11. 面试高频问题与参考答案

## 11.1 为什么推理框架候选人必须理解端到端流程？

**回答：**

因为真实 LLM 推理不是简单调用 `model.forward()`，而是一个在线服务系统。请求进入后要经过 API Server、Tokenizer、Scheduler、KV Cache Manager、ModelRunner、Sampler 和 Detokenizer。性能瓶颈可能出现在调度、KV Cache、Prefill、Decode、采样、通信或底层算子。只有理解端到端流程，才能定位瓶颈并解释优化收益。

---

## 11.2 PagedAttention 解决什么问题？

**回答：**

PagedAttention 主要解决 KV Cache 动态分配中的显存浪费和碎片问题。传统方式可能为每个请求预留最大上下文长度的连续显存，短请求会浪费大量空间。PagedAttention 借鉴操作系统分页思想，将 KV Cache 拆成 block/page 管理，需要多少分配多少，并通过 block table 做逻辑到物理映射，从而提高显存利用率，支持更多并发请求。

---

## 11.3 Continuous Batching 和 Static Batching 有什么区别？

**回答：**

Static Batching 是固定一批请求一起执行，通常要等整个 batch 完成后才能处理下一批。LLM 生成中不同请求输出长度不同，短请求会被长请求拖累。Continuous Batching 在每个 decode step 动态维护 batch，已完成请求移除，新请求补位，使 GPU 持续保持较高利用率，从而提升吞吐并降低排队延迟。

---

## 11.4 Chunked Prefill 为什么有用？

**回答：**

长 prompt 的 Prefill 阶段会占用大量计算资源，如果一次性处理，会阻塞其他 decode 请求，导致尾延迟升高。Chunked Prefill 将长 prompt 拆成多个 chunk，在调度中和 decode 请求交错执行，可以改善在线服务的公平性和 P95/P99 延迟，特别适合长短请求混合负载。

---

## 11.5 PD 分离为什么工作量大？

**回答：**

PD 分离不是简单把 prefill 和 decode 写成两个函数，而是要在系统层面拆分资源。它涉及调度器改造、KV Cache 跨节点传输、请求状态同步、远端 KV 地址管理、prefill 节点和 decode 节点通信以及性能评估。它会侵入推理框架的调度和 KV Cache 生命周期管理，因此短期项目不建议轻易硬做。

---

## 11.6 KV Cache 量化有什么收益和代价？

**回答：**

收益是减少 KV Cache 显存占用，使同样显存能支持更长上下文或更多并发。代价是需要反量化或低精度 attention kernel 支持，可能带来额外计算开销和精度损失。实验上要比较显存下降比例、输出质量、TPOT 和长上下文场景下的吞吐收益。

---

## 11.7 AWQ 和 GPTQ 分别是什么？

**回答：**

GPTQ 和 AWQ 都是大模型权重量化方法，常用于 INT4 权重量化。GPTQ 更强调基于二阶信息或近似 Hessian 的逐层量化误差补偿；AWQ 强调保护重要激活通道，通过权重缩放降低量化误差。推理工程中它们都需要对应的量化权重加载、反量化或 fused GEMM kernel 支持。

---

## 11.8 Softmax CUDA kernel 怎么优化？

**回答：**

Softmax 通常按 row 处理，先做 max reduce 保证数值稳定，再做 exp 和 sum reduce，最后归一化。优化重点是减少 global memory 访问、使用 shared memory 或 warp-level primitive 做归约、保证 coalesced memory access、减少同步和分支。Softmax 多数情况下偏 memory-bound。

---

## 11.9 RMSNorm 为什么是大模型推理高频算子？

**回答：**

很多 LLM 使用 RMSNorm 替代 LayerNorm。它出现在每个 decoder block 中，调用频率高。虽然单次计算量不大，但小算子频繁执行会带来 kernel launch 和访存开销，因此常见优化包括融合、向量化加载、warp/block reduce 和与其他算子合并。

---

## 11.10 MatMul 优化的基本思路是什么？

**回答：**

MatMul 的核心优化是 tiling。naive 实现会重复从 global memory 读取 A/B。优化版本将 A/B 子块加载到 shared memory，在 block 内复用，减少 global memory 访问。进一步可以做 register tiling、vectorized load、double buffering、Tensor Core 和 pipeline。性能分析上通常要判断是否 compute-bound，并关注 Tensor Core 利用率和 memory throughput。

---

## 11.11 简历中为什么不能写太多算子？

**回答：**

因为简历上写的每个算子都可能被面试官追问。写太多不熟悉的算子会增加暴露风险。更好的方式是选择少数核心算子，准备好实现细节、优化策略、benchmark、profiling 和精度验证，让项目更可信。

---

## 11.12 TensorRT 原生支持 Conv，为什么还要写 Plugin？

**回答：**

只有在原生算子不能满足需求时才应该写 Plugin。例如：TensorRT 不支持某个特殊算子、需要融合多个操作、特定 shape 下自定义实现更快、需要特殊数据布局、或者需要处理动态 shape/后处理逻辑。若重写 TensorRT 已经高度优化的基础 Conv，必须用实验数据证明收益，否则面试官会质疑动机。

---

## 11.13 端侧 YOLO 推理项目怎么讲得更像 AI Infra？

**回答：**

不能只讲“部署成功”。要讲模型转换、TensorRT engine 构建、FP16/INT8、dynamic shape、Plugin、吞吐 FPS、单帧延迟、显存、精度变化、平台限制和优化策略。最好和 PyTorch / ONNX Runtime / TensorRT native 做 baseline 对比。

---

## 11.14 多 GPU OCR 服务只用信号量调度有什么问题？

**回答：**

信号量只能表示资源是否可用，不能表示 GPU 当前负载、排队时间、显存占用和预计完成时间。更好的调度应该结合请求大小、GPU 当前队列、历史平均耗时、batch 合并机会等因素，选择预计完成最快的 GPU。

---

## 11.15 为什么 batch 合并可能提升视觉模型推理吞吐？

**回答：**

单张图片推理可能无法充分利用 GPU。将短时间内到达的多个图片请求合并成 batch，可以提高矩阵计算规模和 GPU 利用率，从而提升吞吐。但 batch 会引入等待时间，所以要在吞吐和延迟之间权衡。

---

## 11.16 AI 会不会替代算子开发？

**回答：**

AI 会提高编码效率，但短期不会完全替代算子工程师。算子开发不只是写代码，还包括需求拆解、判断是否需要新算子、设计 benchmark、做 profiling、判断优化方向、精度验收和线上维护。AI 可以辅助生成初版 kernel 和优化建议，但最终判断、验证和责任仍需要工程师承担。

---

## 11.17 学 vLLM 为什么建议先看 nano-vLLM？

**回答：**

完整 vLLM 工程复杂，初学者容易被各种兼容逻辑和工程细节淹没。nano-vLLM / mini-vLLM 保留了 PagedAttention、Scheduler、KV Cache、Prefill/Decode、Sampling 等核心机制，代码量更小，更适合建立整体理解。理解简化版后再看完整 vLLM，学习曲线更平滑。

---

## 11.18 多模态模型适配的核心难点是什么？

**回答：**

核心难点是把图像信息接入语言模型推理流程。通常需要实现图像预处理、vision encoder、projector、image token 替换、视觉 embedding 和文本 embedding 拼接，同时还要处理多模态场景下的 prefix cache、KV Cache 管理和模型结构差异。

---

## 11.19 图像 prefix cache 如何避免误匹配？

**回答：**

文本 prefix 可以对 token ids 做 hash。图像 prefix 需要将图像预处理后的内容、尺寸、patch 信息或视觉特征纳入 hash，确保相同图像可以复用，不同图像不能误复用。同时要避免大图像数据在进程间反复拷贝，可以缓存图像特征或使用轻量哈希标识。

---

## 11.20 机器人 AI Infra 和云端大模型 AI Infra 最大区别是什么？

**回答：**

机器人和端侧场景多是单机单卡或少量卡，重点是 FPS、单帧延迟、功耗、TensorRT、模型压缩和传感器/控制联动。云端大模型场景是多机多卡集群，重点是 TTFT、TPOT、吞吐、KV Cache、Continuous Batching、并行策略和服务稳定性。两者都属于推理优化，但系统规模和性能指标不同。

---

# 12. 给秋招准备者的执行路线

## 12.1 时间紧张版本：两个月冲刺

### 第 1–2 周：补推理框架主线

- 跑通 vLLM；
- 跑通 nano-vLLM；
- 画出请求到 token 输出流程；
- 理解 PagedAttention、KV Cache、Continuous Batching；
- 写一篇自己的源码笔记。

### 第 3–4 周：补 CUDA / Triton 高频算子

每天至少手写一个：

- reduce；
- softmax；
- rmsnorm；
- matmul；
- transpose。

要求：

- CUDA 版至少能写基础实现；
- Triton 版至少能写 Softmax / MatMul；
- 每个算子有 benchmark；
- 能解释瓶颈和优化方向。

### 第 5 周：项目补实验

对简历项目补：

- baseline；
- 优化前后数据；
- 显存；
- 延迟；
- 吞吐；
- 精度；
- 适用场景。

### 第 6 周：面试表达训练

准备：

- 项目 3 分钟介绍；
- 项目 10 分钟深挖；
- 每个简历关键词至少准备 5 个追问；
- vLLM 高频八股；
- CUDA 高频八股；
- C++ 高频八股；
- LeetCode hot 100 中等题。

### 第 7–8 周：投递和复盘

- 边投边面；
- 每场面试后记录问题；
- 对不会的问题当天补；
- 简历根据反馈持续修改；
- 不要等“完全准备好”再投。

---

## 12.2 项目优先级建议

如果你现在有多个方向，不建议全部铺开。优先级如下：

1. **已有实习/科研项目深挖**
   - 最真实，最容易讲；
   - 补数据和优化点。

2. **nano-vLLM / mini-vLLM 改造**
   - 最贴近推理框架；
   - 适合 AI Infra 面试。

3. **CUDA / Triton 算子项目**
   - 最适合算子岗；
   - 需要手写能力支撑。

4. **vLLM 部署 benchmark**
   - 作为入门项目可以；
   - 单独作为核心项目略弱，需要源码或优化补充。

5. **RAG / Agent 应用**
   - 如果投 AI 应用有用；
   - 对推理框架/算子岗优先级较低。

---

## 12.3 简历修改原则

1. **删掉不熟的内容**
   - 不熟的算子不要写；
   - 不熟的框架特性不要写。

2. **每个项目必须有数据**
   - 延迟、吞吐、显存、精度至少选两个。

3. **把“功能”改成“问题-方案-效果”**
   - 不写“支持了 KV Cache 量化”；
   - 写“为降低长上下文显存占用，实现 KV Cache 量化，显存下降 X%，输出一致性保持 X”。

4. **避免大而空**
   - “掌握 vLLM”不如“实现简化版 BlockManager 并支持 prefix cache”。

5. **项目名称要专业**
   - “mini-vLLM 学习项目”不如“轻量级 LLM 推理引擎的 KV Cache 管理与调度优化”。

---

# 13. 最终总结

这节课的核心价值不是讲某一个固定技术点，而是从模拟面试中总结出 AI Infra 秋招准备的真实要求：

1. **项目要真实、可解释、有数据。**  
   只写实现了功能是不够的，必须能说明优化动机、实验设计和性能收益。

2. **算子开发不能只会看懂代码。**  
   Softmax、Reduce、RMSNorm、MatMul 这些高频算子必须能手写，并能讲清优化路径。

3. **推理框架必须理解端到端。**  
   从 HTTP 请求、tokenizer、scheduler、KV Cache、prefill、decode、sampler 到 streaming output 都要能串起来。

4. **短期项目要选工作量可控的功能。**  
   AWQ、KV Cache 量化、Prefix Cache、Chunked Prefill 简化版比完整 PD 分离更适合秋招前做。

5. **简历不要贪多。**  
   写得越多，被追问的靶子越多。少写但写深，比堆关键词更有竞争力。

6. **AI 不是算子工程师的替代品，而是放大器。**  
   未来更重要的能力是提出问题、拆解需求、做 profiling、判断优化方向和完成工程验收。

对于正在准备 AI Infra 推理侧秋招的人，最应该形成的能力闭环是：

```text
理解模型结构
  → 理解推理框架端到端流程
  → 能写常见 CUDA/Triton 算子
  → 能做 benchmark 和 profiling
  → 能把项目讲成“问题-方案-数据-反思”
```

只要这个闭环形成，简历和面试的可信度会明显提升。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-AI infra学习路线|模块-AI infra学习路线]]

%% 项目关联导航：结束 %%
