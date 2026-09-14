# AI Infra 推理方向学习路线与项目计划

> 适用对象：电子信息/嵌入式/Linux 背景，目标转向 AI Infra 推理工程方向。  
> 核心原则：不从完整机器学习体系开始学，而是围绕“大模型推理服务如何跑起来、如何测性能、如何理解瓶颈”建立最小闭环。  
> 主线定位：LLM 推理服务与性能优化。  
> 副线定位：CUDA 算子优化与端侧推理可作为后续增强，但当前不作为第一主线。

---

## 0. 总体路线

当前学习路线建议按以下顺序推进：

```text
阶段 1：一周补基础概念
    ↓
阶段 2：LLM 推理服务与 Benchmark
    ↓
阶段 3：推理引擎源码理解
    ↓
阶段 4：CUDA 基础算子与性能分析
    ↓
阶段 5：TensorRT-LLM / 端侧推理扩展
    ↓
阶段 6：面试准备与项目包装
```

其中，前两个项目最重要：

1. **项目一：基于 vLLM 与 GuideLLM 的大语言模型推理服务性能评测**
2. **项目二：基于 mini-SGLang / nano-vLLM 的轻量级推理引擎源码分析与插桩实验**

这两个项目分别解决两个核心问题：

```text
项目一：推理服务外部怎么部署、怎么压测、怎么分析性能
项目二：推理引擎内部怎么调度、怎么管理 KV Cache、怎么执行 prefill/decode
```

---

# 阶段 1：一周补基础概念

## 1.1 阶段定位

这一阶段是整个 AI Infra 推理路线的入口，目标不是系统学习机器学习和深度学习，而是补足后续做推理项目所必须理解的最小概念集。

在 AI Infra 中，它属于：

```text
基础认知层：理解模型是什么、推理是什么、LLM 为什么需要推理优化
```

你不需要在这一阶段深入训练算法、反向传播、优化器、传统机器学习算法。你只需要知道训练和推理的区别，以及大模型推理服务为什么会有延迟、吞吐、显存等工程问题。

---

## 1.2 需要学习的内容

### 1. 机器学习、神经网络、深度学习的基本概念

需要理解：

```text
机器学习：让模型从数据中学习规律
神经网络：机器学习中的一种模型结构
深度学习：多层神经网络
Transformer：当前大语言模型的主流网络结构
推理：使用训练好的模型对新输入进行前向计算
```

重点是区分：

```text
训练 = 学习模型参数
推理 = 使用模型参数生成结果
```

对 AI Infra 推理来说，训练不是当前主线。你主要关注的是：模型已经训练好了，如何把它高效部署出来服务用户。

---

### 2. Transformer 推理流程

需要理解：

```text
token
embedding
decoder-only Transformer
Self-Attention
Q / K / V
Multi-Head Attention
MLP / FFN
LayerNorm
logits
next token prediction
```

你要能讲清楚一次 LLM 推理的大概流程：

```text
用户输入文本
    ↓
tokenizer 将文本变成 token
    ↓
token 进入 Transformer
    ↓
模型输出 logits
    ↓
根据 logits 选择下一个 token
    ↓
把新 token 接回输入，继续生成
```

这一部分是所有推理优化的基础。只有理解模型为什么逐 token 生成，后面才能理解 decode 为什么慢、KV Cache 为什么重要。

---

### 3. Prefill、Decode 与 KV Cache

这是推理方向最核心的基础概念。

需要理解：

```text
Prefill：一次性处理用户输入 prompt，建立初始 KV Cache
Decode：模型一个 token 一个 token 地生成输出
KV Cache：缓存历史 token 的 Key/Value，避免重复计算
```

你要能解释：

```text
为什么输入越长，首 token 延迟 TTFT 越高？
为什么输出越长，decode 阶段越耗时？
为什么 KV Cache 会占用大量显存？
为什么上下文越长，并发能力越容易下降？
```

在 LLM 推理系统中，KV Cache 管理是 vLLM、SGLang、TensorRT-LLM 等推理框架的重要优化对象。

---

### 4. PyTorch 推理基础

只需要学习推理相关的 PyTorch。

重点内容：

```text
Tensor
shape
dtype
device
cuda()
to("cuda")
torch.no_grad()
torch.inference_mode()
model.eval()
forward()
generate()
```

你需要能看懂如下代码背后的含义：

```python
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=128)
```

这段代码代表：

```text
文本 → token → tensor → GPU → 模型前向计算 → logits → 输出 token
```

暂时不需要深入：

```text
反向传播
优化器
损失函数
训练循环
学习率调参
```

这些更偏训练方向，不是当前最短路径。

---

### 5. 推理性能指标

需要重点理解：

```text
TTFT：Time To First Token，首 token 延迟
TPOT：Time Per Output Token，每个输出 token 平均耗时
Latency：总延迟
Throughput：吞吐量，单位时间处理多少 token 或请求
QPS：每秒请求数
GPU Utilization：GPU 利用率
Memory Usage：显存占用
```

推理系统优化不是简单追求“快”，而是要同时平衡：

```text
低延迟：单个用户体验好
高吞吐：单位 GPU 服务更多用户，成本更低
低显存：能部署更大模型或支持更多并发
高稳定性：服务不崩溃、不 OOM
```

这些指标会直接用于第一个项目。

---

### 6. vLLM 与 SGLang 的基本概念

需要理解：

```text
vLLM：高吞吐 LLM 推理服务框架
SGLang：面向复杂 LLM 程序和高效推理服务的框架
PagedAttention：vLLM 中用于高效管理 KV Cache 的机制
Continuous Batching：持续批处理，提高多用户并发吞吐
Prefix Cache：复用相同前缀的 KV Cache
Chunked Prefill：将长 prompt 的 prefill 拆分，降低调度和显存压力
Speculative Decoding：推测解码，用小模型或 draft 机制加速生成
```

这个阶段不要求读源码，只要求知道它们在推理系统中解决什么问题。

---

## 1.3 阶段产出

这一阶段结束后，建议整理一份短文档：

```text
LLM 推理基础概念速查.md
```

内容包括：

```text
Transformer 推理流程
Prefill / Decode
KV Cache
TTFT / TPOT / Throughput
vLLM / SGLang 基本作用
量化的基本意义
CUDA 在推理中的作用
```

目标不是“完全学懂”，而是能带着这些概念进入项目实践。

---

# 阶段 2：LLM 推理服务与 Benchmark

## 2.1 阶段定位

这是你真正进入 AI Infra 推理的第一阶段实战。

在整个 AI Infra 中，它属于：

```text
推理服务层：把训练好的模型部署成服务，并评估它的性能
```

这一阶段的重点不是看源码，也不是写 CUDA，而是建立最重要的工程闭环：

```text
模型部署 → API 服务 → 并发请求 → 性能测试 → 指标分析 → 项目报告
```

这是最适合你当前阶段的第一主线。

---

## 2.2 需要学习的内容

### 1. vLLM 服务部署

需要学习：

```text
vLLM serve
OpenAI-compatible API Server
模型加载
gpu_memory_utilization
max_model_len
tensor_parallel_size
sampling 参数
```

你要知道：

```text
vLLM 的作用不是训练模型，而是把训练好的模型高效部署成推理服务。
```

例如部署 Qwen 或 Llama 小模型，并通过 HTTP API 调用它。

---

### 2. Benchmark 工具使用

建议使用：

```text
GuideLLM
vLLM 官方 benchmark 工具
```

你需要学习如何设置：

```text
输入长度
输出长度
并发数
请求速率
模型大小
测试轮数
输出格式
```

这一部分的目标是：不要自己从零写复杂 benchmark，而是基于开源工具做可复现实验。

---

### 3. 推理性能分析

需要围绕以下变量做实验：

```text
输入长度：128 / 512 / 1024 / 2048
输出长度：64 / 128 / 256
并发数：1 / 2 / 4 / 8 / 16
模型大小：0.5B / 1.5B / 3B
精度配置：FP16 / INT4 或 AWQ
```

需要记录：

```text
TTFT
TPOT
Throughput
Latency
显存占用
GPU 利用率
OOM 情况
```

你要能够分析：

```text
输入越长，prefill 计算量越大，TTFT 越高
输出越长，decode 阶段占比越高
并发增加时，吞吐量通常先提升，过高后延迟恶化
max_model_len 增大，会影响 KV Cache 和显存压力
量化可以降低显存占用，但可能影响输出质量
```

---

## 2.3 阶段产出

这一阶段结束后，你应该得到一个完整 GitHub 项目：

```text
vllm-guidellm-benchmark/
├── README.md
├── scripts/
│   ├── start_vllm.sh
│   ├── run_benchmark.sh
│   └── collect_gpu_info.sh
├── results/
│   ├── fp16_concurrency.csv
│   ├── long_context.csv
│   └── quantization_compare.csv
├── docs/
│   ├── inference_concepts.md
│   ├── benchmark_analysis.md
│   └── problems_and_solutions.md
└── figures/
    ├── throughput_vs_concurrency.png
    ├── ttft_vs_prompt_len.png
    └── memory_usage.png
```

---

# 阶段 3：推理引擎源码理解

## 3.1 阶段定位

如果阶段 2 是从外部理解推理服务，那么阶段 3 是从内部理解推理引擎。

在整个 AI Infra 中，它属于：

```text
推理引擎层：理解请求调度、KV Cache 管理、prefill/decode 执行过程
```

这一阶段最重要的是把你从“会使用推理框架”提升到“理解推理框架内部机制”。

---

## 3.2 需要学习的内容

建议选择：

```text
mini-SGLang
或
nano-vLLM
```

这两个都是轻量级项目，适合源码学习。相比直接读 vLLM 或 SGLang 主仓库，它们代码更少、结构更清楚。

重点学习模块：

```text
Engine：推理引擎入口
Scheduler：请求调度
ModelRunner：模型执行
KV Cache Manager / Block Manager：KV Cache 管理
Tokenizer / Detokenizer：文本与 token 转换
Sampling：采样生成 token
```

需要理解的核心流程：

```text
用户请求进入系统
    ↓
tokenize
    ↓
加入调度队列
    ↓
执行 prefill
    ↓
分配 KV Cache block
    ↓
进入 decode loop
    ↓
sampling 生成 token
    ↓
输出 token
    ↓
请求结束并释放 KV Cache
```

这一阶段不要追求看完全部源码，而是重点抓住：

```text
请求如何被调度
batch 如何形成
KV Cache 如何分配和释放
prefill 和 decode 如何切换
每一步 decode 如何生成新 token
```

---

## 3.3 阶段产出

建议输出：

```text
轻量级 LLM 推理引擎源码分析.md
```

内容包括：

```text
源码目录结构
核心模块职责
请求处理流程图
KV Cache 生命周期
Scheduler 调度逻辑
prefill/decode 执行流程
插桩日志分析
```

还可以做一个小实验：

```text
不同输入长度下，KV Cache block 分配情况
不同 batch size 下，decode 调度情况
普通 transformers.generate 与轻量推理引擎性能对比
```

---

# 阶段 4：CUDA 基础算子与性能分析

## 4.1 阶段定位

这一阶段是从推理系统走向底层性能优化的关键。

在整个 AI Infra 中，它属于：

```text
算子优化层：理解 GPU 如何执行模型计算，以及如何优化 kernel 性能
```

这部分是 AIInfraGuide 中非常重要的模块。它决定你能不能从“会部署模型”进一步提升到“懂底层优化”。

---

## 4.2 需要学习的内容

### 1. CUDA 基础

重点学习：

```text
kernel
thread / block / grid
warp
global memory
shared memory
register
memory coalescing
bank conflict
synchronization
atomic
occupancy
```

需要理解：

```text
GPU 为什么适合并行计算
线程如何映射到数据
显存访问为什么可能成为瓶颈
shared memory 为什么能加速
warp divergence 为什么会降低效率
```

---

### 2. 基础算子实现

推荐顺序：

```text
Vector Add
Reduce
Matrix Transpose
Softmax
LayerNorm / RMSNorm
Naive GEMM
Tiled GEMM
```

不要一开始直接做：

```text
FlashAttention
CUTLASS
Tensor Core MMA
高性能 GEMM
```

原因是这些内容门槛较高，容易导致长时间卡住，反而影响主线项目推进。

---

### 3. Nsight Compute 性能分析

需要学习：

```text
kernel 执行时间
memory throughput
occupancy
warp divergence
global memory load/store efficiency
shared memory bank conflict
roofline 思想
```

目标是能够解释：

```text
这个 kernel 慢在哪里？
是计算瓶颈还是访存瓶颈？
优化后性能为什么提升？
还有什么限制？
```

---

## 4.3 阶段产出

建议输出一个 CUDA 小项目：

```text
cuda-basic-ops/
├── reduce/
├── softmax/
├── layernorm/
├── gemm/
├── benchmark/
└── docs/
    ├── reduce_optimization.md
    ├── softmax_analysis.md
    └── nsight_notes.md
```

这个项目可以作为第二阶段求职时的底层能力证明。

---

# 阶段 5：TensorRT-LLM / 端侧推理扩展

## 5.1 阶段定位

这一阶段用于扩大你的推理部署能力边界。

在整个 AI Infra 中，它属于：

```text
部署后端与硬件适配层：让模型在不同硬件和推理后端上高效运行
```

根据你的目标，可以分成两个方向。

---

## 5.2 方向 A：服务器端 AI Infra 推理

建议学习：

```text
TensorRT-LLM
TensorRT
ONNX Runtime
LLM 量化
多 GPU 推理
```

适合目标：

```text
大厂 AI Infra 推理
GPU 推理服务
NVIDIA 生态
高性能 LLM serving
```

需要理解：

```text
TensorRT-LLM 如何构建 engine
TensorRT-LLM 和 vLLM 的区别
FP16 / INT8 / INT4 如何影响性能
Paged KV Cache、in-flight batching 等机制
```

---

## 5.3 方向 B：端侧/边缘 AI 推理

建议学习：

```text
llama.cpp
ncnn
ONNX Runtime
QNN
Core ML
TFLite
```

适合目标：

```text
嵌入式 AI
端侧 AI
机器人
车载
手机端
边缘设备推理部署
```

这一方向和你的嵌入式/Linux背景更贴近，可以作为你的特色路线。

需要理解：

```text
模型转换
端侧量化
CPU/GPU/NPU/DSP 后端差异
内存占用分析
低功耗推理
端侧部署限制
```

建议你当前优先服务器端主线，后续用端侧项目强化差异化优势。

---

# 阶段 6：面试准备与项目包装

## 6.1 阶段定位

这一阶段用于把学习内容转化成求职竞争力。

在整个 AI Infra 中，它属于：

```text
岗位匹配层：用面经反推知识短板，用项目证明工程能力
```

AIInfraGuide 的面经模块对你很有价值，建议长期并行使用。

---

## 6.2 需要整理的面试模块

建议建立文档：

```text
AI Infra 面试高频问题整理.md
```

按以下模块整理：

```text
C++ / Linux
CUDA
GPU 架构
Transformer
Attention
KV Cache
Prefill / Decode
vLLM / SGLang
量化
推理性能指标
分布式训练基础
项目复盘
```

你每完成一个项目，都要整理：

```text
项目背景
解决了什么问题
技术路线
遇到的问题
如何定位
如何解决
实验结果
性能指标
不足与改进方向
```

面试时，项目不是简单说“我做过”，而是要能讲清楚：

```text
为什么做这个项目？
系统架构是什么？
核心瓶颈在哪里？
指标怎么测？
结果说明了什么？
如果继续优化，你会怎么做？
```

---

# 两个最重要的项目

下面只保留当前阶段最重要的两个项目。

---

# 项目一：基于 vLLM 与 GuideLLM 的大语言模型推理服务性能评测

## 1. 项目定位

这是你的第一个主项目，定位是：

```text
LLM 推理服务部署与性能评测项目
```

在 AI Infra 中属于：

```text
推理服务层 + Benchmark 层
```

它解决的问题是：

```text
训练好的大模型如何部署成服务？
服务性能如何评估？
输入长度、输出长度、并发数、量化配置如何影响性能？
```

这个项目最适合你当前阶段，因为它能快速建立完整工程闭环。

---

## 2. 项目目标

项目目标：

```text
基于开源 vLLM 部署 Qwen / Llama 等大语言模型，使用 GuideLLM 或 vLLM 官方 benchmark 工具进行性能评测，分析不同输入长度、输出长度、并发数和精度配置下的推理性能变化。
```

最终你需要能够回答：

```text
什么是 TTFT？
什么是 TPOT？
什么是 Throughput？
为什么输入长度会影响首 token 延迟？
为什么输出长度会影响 decode 阶段耗时？
为什么并发提高后吞吐量会上升，但延迟也可能恶化？
为什么 KV Cache 会限制长上下文和高并发？
量化为什么能降低推理成本？
```

---

## 3. 技术栈

建议技术栈：

```text
Python
Linux
CUDA
PyTorch / Hugging Face
vLLM
GuideLLM
nvidia-smi / nvitop
Qwen2.5-0.5B / Qwen2.5-1.5B / TinyLlama
```

---

## 4. 实验设计

建议实验变量：

```text
模型大小：0.5B / 1.5B / 3B
输入长度：128 / 512 / 1024 / 2048
输出长度：64 / 128 / 256
并发数：1 / 2 / 4 / 8 / 16
精度配置：FP16 / INT4 或 AWQ
```

记录指标：

```text
TTFT
TPOT
End-to-end Latency
Throughput
QPS
GPU Utilization
显存占用
OOM 情况
```

---

## 5. 项目产出

建议仓库结构：

```text
vllm-guidellm-benchmark/
├── README.md
├── scripts/
│   ├── start_vllm.sh
│   ├── run_benchmark.sh
│   └── collect_gpu_info.sh
├── results/
│   ├── concurrency_test.csv
│   ├── prompt_length_test.csv
│   ├── output_length_test.csv
│   └── quantization_test.csv
├── docs/
│   ├── inference_concepts.md
│   ├── benchmark_analysis.md
│   └── problems_and_solutions.md
└── figures/
    ├── throughput_vs_concurrency.png
    ├── ttft_vs_prompt_len.png
    ├── tpot_vs_output_len.png
    └── memory_usage.png
```

---

## 6. 简历表达

```text
基于 vLLM 与 GuideLLM 的大语言模型推理服务性能评测

- 基于 vLLM 部署 Qwen 系列开源模型，构建 OpenAI-compatible API 推理服务；
- 使用 GuideLLM 设计推理 benchmark，测试不同输入长度、输出长度和并发数下的推理性能；
- 统计并分析 TTFT、TPOT、Throughput、Latency、GPU 利用率和显存占用；
- 分析 Prefill/Decode 阶段性能差异，理解 KV Cache 对长上下文和高并发的影响；
- 对比 FP16 与 INT4/AWQ 量化配置下的显存占用和推理吞吐变化。
```

---

# 项目二：基于 mini-SGLang / nano-vLLM 的轻量级推理引擎源码分析与插桩实验

## 1. 项目定位

这是你的第二个核心项目，定位是：

```text
轻量级 LLM 推理引擎源码理解项目
```

在 AI Infra 中属于：

```text
推理引擎层
```

项目一让你知道“推理服务外部怎么跑”，项目二让你知道“推理引擎内部怎么实现”。

---

## 2. 项目目标

项目目标：

```text
基于 mini-SGLang 或 nano-vLLM，阅读轻量级推理引擎源码，梳理请求从输入、调度、prefill、decode、KV Cache 分配到 token 输出的完整链路，并通过日志插桩观察调度和 KV Cache 管理过程。
```

你要能够回答：

```text
一个请求进入推理引擎后经历了哪些阶段？
Scheduler 的作用是什么？
prefill 和 decode 如何被调度？
KV Cache block 如何分配和释放？
batch 是如何形成的？
为什么 continuous batching 能提高吞吐？
prefix cache / chunked prefill 有什么作用？
```

---

## 3. 推荐选择

优先推荐：

```text
mini-SGLang
```

原因：

```text
更接近现代推理服务框架
包含 Radix Cache、Chunked Prefill、overlap scheduling 等机制
更适合理解 SGLang 生态
```

备选：

```text
nano-vLLM
```

原因：

```text
代码更轻量
更适合理解 vLLM 的基本思想
适合入门 KV Cache、调度器、模型执行流程
```

建议顺序：

```text
先根据文档跑通 example
再阅读核心模块
最后做插桩和流程图
```

---

## 4. 重点阅读模块

重点关注：

```text
Engine：推理引擎入口
Scheduler：请求调度
ModelRunner：模型执行
KV Cache Manager / Block Manager：KV Cache 管理
Tokenizer / Detokenizer：文本与 token 转换
Sampling：采样生成 token
```

不要一开始追求看完所有源码。

---

## 5. 插桩实验设计

可以添加日志观察：

```text
请求输入长度
请求进入队列的时间
请求进入 prefill 的时间
请求进入 decode 的时间
每一步 decode 的 batch size
每个请求生成的 token 数
KV Cache block 分配数量
KV Cache block 释放时间
请求总耗时
```

可以做的小实验：

```text
不同输入长度下 KV Cache 分配变化
不同并发数下 batch 调度变化
不同 max_new_tokens 下 decode loop 次数变化
普通 transformers.generate 与轻量推理引擎的速度对比
```

---

## 6. 项目产出

建议仓库结构：

```text
mini-sglang-source-analysis/
├── README.md
├── notes/
│   ├── architecture.md
│   ├── scheduler.md
│   ├── kv_cache.md
│   └── prefill_decode.md
├── logs/
│   ├── request_trace_short_prompt.log
│   ├── request_trace_long_prompt.log
│   └── batch_schedule.log
├── figures/
│   ├── engine_flow.png
│   ├── kv_cache_lifecycle.png
│   └── scheduler_flow.png
└── patches/
    └── logging_patch.diff
```

---

## 7. 简历表达

```text
基于 mini-SGLang / nano-vLLM 的轻量级推理引擎源码分析与插桩实验

- 阅读轻量级 LLM 推理引擎源码，梳理请求从输入、调度、Prefill、Decode 到 token 输出的完整链路；
- 分析 Engine、Scheduler、ModelRunner、KV Cache Manager 等核心模块职责；
- 通过日志插桩观察不同输入长度和 batch size 下的 KV Cache 分配与释放过程；
- 总结 Continuous Batching、Prefix Cache、Chunked Prefill 等机制的工程意义；
- 对比普通 Transformers generate 与轻量级推理引擎在吞吐和延迟上的差异。
```

---

# 最终执行顺序

建议你的实际执行顺序如下：

```text
第 1 周：
补 Transformer / PyTorch / KV Cache / Prefill / Decode / 推理指标

第 2-4 周：
完成项目一：vLLM + GuideLLM 推理 Benchmark

第 5-8 周：
完成项目二：mini-SGLang / nano-vLLM 源码插桩分析

第 9 周以后：
补 CUDA 基础算子、TensorRT-LLM、端侧推理
```

---

# 最终总结

你当前最适合的路线是：

```text
主线：LLM 推理服务与性能优化
副线：推理引擎源码理解
后续增强：CUDA 算子优化与端侧推理
```

最重要的两个项目是：

```text
1. vLLM + GuideLLM 推理服务 Benchmark
2. mini-SGLang / nano-vLLM 源码插桩分析
```

第一项目保证你能把模型服务跑起来，并用指标说清性能；第二项目保证你不只是会调用框架，而是能理解推理引擎内部的调度、KV Cache 和 prefill/decode 机制。

完成这两个项目后，你在 AI Infra 推理方向就不再只是“概念了解”，而是具备了可展示、可复盘、可写进简历的工程经历。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-AI infra学习路线|模块-AI infra学习路线]]

%% 项目关联导航：结束 %%
