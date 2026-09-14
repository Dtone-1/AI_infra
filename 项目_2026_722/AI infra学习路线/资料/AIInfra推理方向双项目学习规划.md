# AI Infra 推理方向双项目规划文档

> 适用目标：2027 暑期实习 / 秋招 AI Infra、LLM 推理工程、模型部署、推理框架、异构计算、端侧 AI 等方向。  
> 项目组合定位：**项目一负责证明你理解大模型推理服务、vLLM、KV Cache、Prefill/Decode、Benchmark 与推理框架机制；项目二负责证明你具备 CUDA/Triton 算子实现、性能分析和底层优化能力。**  
> 最终目标不是“学过很多名词”，而是形成两个能写进简历、能被面试官深挖、能用数据和代码支撑的 AI Infra 项目。

---

# 第一部分：两个项目的正式名称、实现内容与简历呈现

## 项目一：基于 vLLM 与 nano-vLLM 的大语言模型推理服务性能分析与核心机制复现

### 1. 项目定位

该项目是你的 **AI Infra 推理框架主项目**，核心目标是打通从真实推理服务部署到轻量推理框架机制复现的完整链路。

它不是单纯“部署一个模型”，也不是单纯“读 nano-vLLM 源码”，而是把两者合并成一个完整项目：

```text
vLLM：真实工业框架，负责真实模型部署、OpenAI API 服务、Benchmark、量化对比
nano-vLLM：轻量源码框架，负责理解 Scheduler、KV Cache、Prefill/Decode、ModelRunner 等核心机制
```

项目要证明你具备以下能力：

```text
1. 能把 Qwen 系列开源模型部署成 OpenAI-compatible API 推理服务；
2. 能设计并实现 LLM 推理 Benchmark 脚本；
3. 能统计 TTFT、TPOT、吞吐、总延迟、P95 延迟和显存占用；
4. 能分析输入长度、输出长度、并发数、vLLM 参数配置对性能的影响；
5. 能理解 Prefill / Decode 阶段的性能差异；
6. 能解释 KV Cache 对长上下文和高并发推理显存占用的影响；
7. 能对比 FP16 与 INT4/AWQ 量化模型的显存、吞吐和延迟差异；
8. 能通过 nano-vLLM 复现推理框架核心链路；
9. 能在 nano-vLLM 中加入 profiling / KV Cache 统计 / 简化优化机制；
10. 能把真实 vLLM Benchmark 中观察到的性能现象和 nano-vLLM 中的源码机制对应起来。
```

### 2. 项目正式简历描述

#### 简历项目名称

**基于 vLLM 与 nano-vLLM 的大语言模型推理服务性能分析与核心机制复现**

#### 简历项目描述

- 基于 vLLM 部署 Qwen 系列开源大语言模型，构建 OpenAI-compatible API 推理服务，支持普通请求与流式生成调用；
- 设计并实现 LLM 推理 Benchmark 脚本，系统统计 TTFT、TPOT、总延迟、P50/P95 延迟、吞吐量、QPS、GPU 显存占用与 OOM 情况；
- 构建多维度实验矩阵，对比不同输入长度、输出长度、并发数、`max_num_seqs`、`max_num_batched_tokens`、`max_model_len`、`gpu_memory_utilization` 等参数下的推理性能变化；
- 分析 Prefill 与 Decode 阶段的性能差异，结合输入长度与输出长度变化解释 TTFT、TPOT 与端到端延迟的变化趋势；
- 实现 KV Cache 显存估算脚本，结合模型层数、KV head 数、head_dim、上下文长度、batch/concurrency 与 dtype 估算 KV Cache 显存占用，并与 vLLM 实际显存表现进行对比分析；
- 对比 FP16 与 INT4/AWQ 量化模型在模型权重显存、KV Cache 占用、吞吐量和生成速度上的差异，分析量化对推理成本和服务能力的影响；
- 基于 nano-vLLM 梳理并复现 LLM 推理框架核心流程，包括 Tokenizer、Scheduler、KV Cache Manager、ModelRunner、Sampler 与请求生命周期；
- 在 nano-vLLM 中实现请求级 Prefill/Decode profiling、TTFT/TPOT 统计、KV Cache block 使用量统计，并通过实验解释真实 vLLM Benchmark 中观察到的性能现象；
- 尝试实现简化版 Prefix Cache / Chunked Prefill / Continuous Batching 机制之一，分析其对吞吐、延迟和显存利用率的影响；
- 输出完整实验报告、性能曲线、源码阅读笔记和可复现脚本，形成从推理服务部署、性能测试到框架机制分析的完整闭环。

### 3. 项目关键词

```text
vLLM
nano-vLLM
Qwen
OpenAI-compatible API
LLM Serving
Benchmark
TTFT
TPOT
Prefill
Decode
KV Cache
PagedAttention
Prefix Cache
Chunked Prefill
Continuous Batching
AWQ
INT4
FP16
Scheduler
ModelRunner
GPU Memory
Throughput
Latency
```

### 4. 项目适配岗位

该项目适合投递：

```text
AI Infra 实习生
大模型推理工程实习生
LLM Serving 实习生
模型部署实习生
推理框架研发实习生
AI 平台研发实习生
机器学习系统实习生
大模型平台实习生
端侧/云侧模型推理优化实习生
```

---

## 项目二：CUDA/Triton 大模型核心算子实现与性能分析

### 1. 项目定位

该项目是你的 **底层算子与性能优化项目**，用于补足第一个项目中“框架与服务”之外的底层能力。

项目一证明你理解：

```text
推理服务、vLLM、Benchmark、KV Cache、Prefill/Decode、推理框架机制
```

项目二证明你具备：

```text
CUDA/Triton 算子开发、GPU 并行编程、性能分析、memory-bound / compute-bound 判断、Nsight profiling 能力
```

这两个项目组合起来，能形成比较完整的 AI Infra 推理侧能力画像：

```text
我既能部署和分析真实 LLM 推理服务，
也理解推理框架内部机制，
还具备 CUDA/Triton 底层算子实现和性能分析能力。
```

### 2. 项目正式简历描述

#### 简历项目名称

**CUDA/Triton 大模型核心算子实现与性能分析**

#### 简历项目描述

- 面向 LLM 推理场景，实现 CUDA GEMM、Triton RMSNorm、Softmax、简化 Attention 等核心算子，并与 PyTorch 原生算子、cuBLAS 或 Triton baseline 进行性能对比；
- 从 naive GEMM 出发，逐步实现 shared memory tiling、register blocking、vectorized load 等优化策略，分析不同矩阵规模下的运行时间、吞吐、TFLOPS 与相对 cuBLAS 性能比例；
- 使用 Nsight Compute 对 CUDA kernel 进行 profiling，分析 memory throughput、SM occupancy、warp stall、global memory load/store efficiency、shared memory bank conflict 等指标；
- 实现 Triton 版本的 RMSNorm、Softmax 等 LLM 高频小算子，比较不同 hidden size、sequence length、batch size 下的性能表现；
- 实现 naive Attention，分析标准 Attention 中 attention score matrix 的显存读写开销，进一步学习 FlashAttention 的 IO-aware 优化思想和 Online Softmax 原理；
- 对比不同算子的 memory-bound / compute-bound 特征，总结 GEMM、Softmax、RMSNorm、Attention 在 GPU 上的性能瓶颈与优化方向；
- 输出完整性能测试脚本、实验数据、Nsight 分析截图、优化过程记录和技术报告，形成面向 AI Infra 推理方向的底层算子实践项目。

### 3. 项目关键词

```text
CUDA
Triton
GEMM
MatMul
RMSNorm
Softmax
Attention
FlashAttention
Online Softmax
cuBLAS
Nsight Compute
Nsight Systems
shared memory
register blocking
tiling
warp
bank conflict
coalesced memory access
Tensor Core
memory-bound
compute-bound
GPU profiling
```

### 4. 项目适配岗位

该项目适合投递：

```text
AI Infra 实习生
异构计算实习生
高性能计算实习生
CUDA/Triton 算子开发实习生
推理优化实习生
模型部署优化实习生
端侧 AI 推理优化实习生
AI 芯片 Runtime / 算子库相关实习
```

---

# 第二部分：两个项目的详细学习流程

## 总体学习顺序

建议你按下面顺序推进：

```text
阶段 0：基础准备
    ↓
阶段 1：vLLM 服务部署
    ↓
阶段 2：vLLM Benchmark 与性能实验
    ↓
阶段 3：Prefill / Decode / KV Cache 机制分析
    ↓
阶段 4：nano-vLLM 源码学习与 profiling 修改
    ↓
阶段 5：CUDA GEMM 入门与优化
    ↓
阶段 6：Triton LLM 小算子实现
    ↓
阶段 7：Attention / FlashAttention 原理实验
    ↓
阶段 8：项目整理、报告、简历与面试准备
```

两个项目不是完全割裂的，可以交叉推进，但总体建议先完成项目一的前半部分，再做项目二：

```text
先做 vLLM 部署和 Benchmark，让自己知道真实推理服务长什么样；
再看 nano-vLLM，理解 Benchmark 现象背后的框架机制；
然后补 CUDA/Triton，理解模型执行背后的底层算子。
```

---

## 阶段 0：基础准备

### 目标

先补齐做这两个项目所需的最低基础，不要求一开始深入，但必须知道每个概念在项目中处于什么位置。

### 需要完成的内容

```text
1. Linux 基础命令；
2. Python 基础与脚本编写；
3. PyTorch Tensor 基础；
4. HuggingFace Transformers 基本使用；
5. Transformer Decoder-only 架构；
6. Attention、MLP、RMSNorm、Tokenizer 基本概念；
7. KV Cache、Prefill、Decode 基本概念；
8. CUDA 环境和 NVIDIA GPU 基础；
9. Git / GitHub 基本操作。
```

### 阶段产出

```text
1. 能在 Linux / WSL / 云服务器中创建 Python 环境；
2. 能加载一个 HuggingFace 模型并执行简单推理；
3. 能解释 Tokenizer、Attention、KV Cache、Prefill、Decode；
4. 能使用 nvidia-smi 查看显存和 GPU 利用率；
5. 能用 Git 管理项目代码。
```

---

## 阶段 1：vLLM 服务部署

### 目标

把 Qwen 系列模型部署成真实 API 服务，建立大模型推理服务的第一感知。

### 推荐模型

如果显存较小：

```text
Qwen2.5-0.5B-Instruct
Qwen2.5-1.5B-Instruct
Qwen2.5-Coder-1.5B
```

如果显存较大：

```text
Qwen2.5-7B-Instruct
Qwen2.5-Coder-7B
```

### 需要完成的内容

```text
1. 安装 CUDA / PyTorch / vLLM；
2. 下载 Qwen 模型；
3. 启动 vLLM OpenAI-compatible API server；
4. 用 curl 调用接口；
5. 用 Python client 调用接口；
6. 测试普通输出和流式输出；
7. 记录模型加载显存和基本推理显存；
8. 整理启动命令和环境配置。
```

### 需要理解的问题

```text
1. vLLM 为什么提供 OpenAI-compatible API？
2. API 服务和 model.generate 有什么区别？
3. 请求进入服务后大概经过哪些环节？
4. 流式输出为什么适合大模型推理？
5. 模型加载显存和推理显存有什么区别？
```

### 阶段产出

```text
1. launch_vllm.sh 启动脚本；
2. client.py 调用脚本；
3. README 中记录环境、模型、启动方式；
4. 第一版服务运行截图和显存记录。
```

---

## 阶段 2：vLLM Benchmark 与性能实验

### 目标

设计可复现的 Benchmark，而不是简单“跑一下模型”。

### 实验变量

建议设计实验矩阵：

```text
输入长度：128 / 512 / 2048 / 4096 tokens
输出长度：64 / 256 / 512 tokens
并发数：1 / 4 / 8 / 16 / 32
模型精度：FP16 / AWQ-INT4
vLLM 参数：
    max_num_seqs
    max_num_batched_tokens
    max_model_len
    gpu_memory_utilization
    enable_prefix_caching
    enable_chunked_prefill
```

### 指标统计

需要统计：

```text
TTFT：Time To First Token，首 token 延迟
TPOT：Time Per Output Token，每 token 生成时间
Total Latency：总延迟
P50 / P95 Latency：中位延迟和尾延迟
Throughput：tokens/s
QPS：requests/s
GPU Memory Usage：显存占用
GPU Utilization：GPU 利用率
OOM / error rate：失败率
```

### Benchmark 脚本要求

你的 benchmark 脚本至少应该支持：

```text
1. 指定模型服务地址；
2. 指定输入长度；
3. 指定输出长度；
4. 指定并发数；
5. 控制请求数量；
6. 统计 TTFT；
7. 统计每个输出 token 间隔；
8. 输出 CSV / JSON 结果；
9. 支持多轮重复实验；
10. 支持 warmup，避免冷启动影响结果。
```

### 阶段产出

```text
1. benchmark.py；
2. results_fp16.csv；
3. results_awq.csv；
4. plot_results.py；
5. 并发数-吞吐曲线；
6. 并发数-P95 延迟曲线；
7. 输入长度-TTFT 曲线；
8. 输出长度-总延迟曲线；
9. 显存占用曲线。
```

---

## 阶段 3：Prefill / Decode / KV Cache 机制分析

### 目标

从实验现象上升到机制解释，让项目从“压测项目”变成“AI Infra 项目”。

### 需要理解的核心问题

```text
1. 为什么输入长度增加，TTFT 会明显增加？
2. 为什么输出长度增加，TPOT 和总延迟更重要？
3. 为什么 Prefill 通常更偏 compute-bound？
4. 为什么 Decode 通常更偏 memory-bound？
5. 为什么长上下文会显著增加 KV Cache 显存？
6. 为什么量化能降低权重显存，但不能同比例降低 KV Cache？
7. 为什么并发增加后吞吐提高，但单请求延迟也会上升？
8. 为什么 prefix cache 能降低共享前缀场景下的 TTFT？
9. 为什么 chunked prefill 能改善长 prompt 对 decode 的阻塞？
```

### KV Cache 显存估算脚本

建议写一个 `kv_cache_estimator.py`，输入：

```text
num_layers
num_kv_heads
head_dim
seq_len
batch_size / concurrency
dtype_bytes
```

估算公式：

```text
KV Cache Memory ≈ 2 × num_layers × seq_len × num_kv_heads × head_dim × dtype_bytes × batch_size
```

其中：

```text
2 代表 K 和 V；
num_layers 是模型层数；
seq_len 是上下文长度；
num_kv_heads 是 K/V head 数；
head_dim 是每个 head 的维度；
dtype_bytes 是每个元素占用字节数；
batch_size / concurrency 代表同时存在的请求数量。
```

### 阶段产出

```text
1. kv_cache_estimator.py；
2. KV Cache 理论估算表；
3. vLLM 实际显存占用对比表；
4. Prefill / Decode 分析文档；
5. 长上下文显存增长解释；
6. 量化后显存变化解释。
```

---

## 阶段 4：nano-vLLM 源码学习与 profiling 修改

### 目标

通过轻量框架理解 vLLM 的核心机制，避免停留在“会用 vLLM”。

### 学习顺序

建议按以下模块学习：

```text
1. 项目入口：请求从哪里进入；
2. Tokenizer：文本如何变成 token；
3. Scheduler：请求如何排队和组成 batch；
4. KV Cache Manager：KV Cache 如何分配和释放；
5. ModelRunner：模型 forward 在哪里执行；
6. Sampler：如何从 logits 采样出 token；
7. Request 生命周期：一个请求从进入到结束经历哪些状态；
8. Prefill / Decode：两阶段在代码里如何体现。
```

### 需要做的修改

建议先做低风险但有价值的修改：

```text
1. 打印每个请求的 TTFT；
2. 打印每个请求的 TPOT；
3. 打印每个请求的 total latency；
4. 打印每轮 scheduler 处理的请求数；
5. 打印当前 KV Cache block 使用量；
6. 打印请求结束后 KV Cache block 回收情况；
7. 统计 prefill 和 decode 阶段耗时。
```

进阶可以选择一个机制实现：

```text
1. 简化版 Prefix Cache；
2. 简化版 Chunked Prefill；
3. 简化版 Continuous Batching；
4. 简化版 KV Cache block allocator 可视化。
```

### 阶段产出

```text
1. nano_vllm_notes/architecture.md；
2. nano_vllm_notes/scheduler.md；
3. nano_vllm_notes/kv_cache.md；
4. nano_vllm_notes/prefill_decode.md；
5. 修改后的 nano-vLLM 代码；
6. profiling 输出日志；
7. 小优化前后对比实验。
```

---

## 阶段 5：CUDA GEMM 入门与优化

### 目标

通过 GEMM 学会 CUDA kernel 的基本写法和优化路径。

### 学习顺序

```text
1. CUDA 编程模型：grid / block / thread；
2. 内存层次：global memory / shared memory / register；
3. naive GEMM；
4. shared memory tiled GEMM；
5. register blocking；
6. vectorized load；
7. bank conflict 分析；
8. coalesced memory access；
9. Nsight Compute profiling；
10. 和 torch.matmul / cuBLAS 对比。
```

### 需要记录的指标

```text
运行时间
TFLOPS
相对 cuBLAS 性能比例
global memory throughput
shared memory bank conflict
SM occupancy
warp stall reason
不同矩阵规模下的性能变化
```

### 阶段产出

```text
1. cuda_gemm_naive.cu；
2. cuda_gemm_tiled.cu；
3. cuda_gemm_optimized.cu；
4. benchmark_gemm.py / benchmark_gemm.cpp；
5. Nsight Compute 分析记录；
6. GEMM 优化报告。
```

---

## 阶段 6：Triton LLM 小算子实现

### 目标

用 Triton 快速实现 LLM 高频小算子，形成比纯 CUDA 更容易展示的项目成果。

### 推荐实现算子

```text
1. RMSNorm；
2. Softmax；
3. SiLU / SwiGLU；
4. 简化 Attention；
5. Top-k / Sampling 可选。
```

### 学习重点

```text
1. Triton kernel 编写方式；
2. block / program_id 的理解；
3. mask 处理；
4. tl.load / tl.store；
5. 向量化计算；
6. 和 PyTorch 原生算子对比；
7. 不同 shape 下的性能变化。
```

### 阶段产出

```text
1. triton_rmsnorm.py；
2. triton_softmax.py；
3. triton_attention.py；
4. benchmark_triton_ops.py；
5. PyTorch vs Triton 性能对比表；
6. Triton 小算子实现报告。
```

---

## 阶段 7：Attention / FlashAttention 原理实验

### 目标

不要求完整手写 FlashAttention v2，但要理解 Attention 为什么是大模型推理的关键算子，以及 FlashAttention 为什么能优化。

### 学习顺序

```text
1. 标准 Attention 公式；
2. QK^T、scale、softmax、乘 V；
3. naive Attention 实现；
4. attention score matrix 的显存开销；
5. safe softmax；
6. online softmax；
7. FlashAttention 的分块计算思想；
8. 为什么不保存完整 attention matrix；
9. 为什么减少 HBM 读写；
10. 为什么长序列收益明显。
```

### 阶段产出

```text
1. naive_attention.py；
2. triton_attention.py；
3. online_softmax_notes.md；
4. flashattention_principle.md；
5. Attention 显存开销分析；
6. naive Attention vs 优化 Attention 性能对比。
```

---

## 阶段 8：项目整理、报告、简历与面试准备

### 目标

把“学习过程”变成“求职项目”。

### 项目一最终仓库结构建议

```text
llm-inference-analysis/
├── README.md
├── scripts/
│   ├── launch_vllm.sh
│   ├── benchmark.py
│   ├── kv_cache_estimator.py
│   └── plot_results.py
├── experiments/
│   ├── fp16_results.csv
│   ├── awq_results.csv
│   └── figures/
├── nano_vllm_notes/
│   ├── architecture.md
│   ├── scheduler.md
│   ├── kv_cache.md
│   └── prefill_decode.md
├── nano_vllm_modified/
│   └── modified_code/
└── report/
    └── llm_inference_report.md
```

### 项目二最终仓库结构建议

```text
llm-kernel-optimization/
├── README.md
├── cuda/
│   ├── gemm_naive.cu
│   ├── gemm_tiled.cu
│   └── gemm_optimized.cu
├── triton/
│   ├── rmsnorm.py
│   ├── softmax.py
│   └── attention.py
├── benchmarks/
│   ├── benchmark_gemm.py
│   └── benchmark_triton_ops.py
├── profiling/
│   ├── nsight_gemm_report.md
│   └── figures/
└── report/
    └── kernel_optimization_report.md
```

### 最终求职材料

你最终要准备：

```text
1. 两个 GitHub 仓库；
2. 两份 README；
3. 两份项目技术报告；
4. 性能图表；
5. 实验数据；
6. 简历项目描述；
7. 项目面试讲稿；
8. 常见追问问题答案。
```

---

# 第三部分：按学习流程列出所需技术栈及其作用

## 技术栈总览

| 学习阶段 | 技术栈 | 作用 |
|---|---|---|
| 基础准备 | Linux | 提供模型部署、GPU 调试、脚本运行的基础环境 |
| 基础准备 | Python | 编写推理 client、benchmark、数据分析和 Triton kernel |
| 基础准备 | PyTorch | 理解模型推理、Tensor 运算和算子 baseline |
| 基础准备 | HuggingFace Transformers | 加载开源模型、理解 tokenizer 和模型结构 |
| vLLM 服务 | vLLM | 部署高吞吐 LLM 推理服务 |
| vLLM 服务 | OpenAI-compatible API | 让本地模型服务兼容 OpenAI API 调用方式 |
| Benchmark | asyncio / aiohttp / httpx | 实现并发请求压测 |
| Benchmark | pandas / matplotlib | 处理实验数据并绘制性能图表 |
| 性能分析 | nvidia-smi | 查看 GPU 显存、利用率、功耗等 |
| 性能分析 | Nsight Systems | 分析整体推理时间线和 CPU/GPU 调度 |
| 性能分析 | Nsight Compute | 分析单个 CUDA kernel 的性能瓶颈 |
| 框架机制 | nano-vLLM | 学习轻量 LLM 推理框架核心流程 |
| 显存分析 | KV Cache estimator | 估算长上下文推理显存占用 |
| 量化 | AWQ / INT4 | 降低模型权重显存，分析量化推理收益 |
| CUDA 算子 | CUDA C++ | 编写 GPU kernel，理解底层并行计算 |
| CUDA 算子 | cuBLAS | 作为 GEMM 性能 baseline |
| Triton 算子 | Triton | 用 Python-like 方式快速实现高性能 GPU kernel |
| 工程管理 | Git / GitHub | 管理代码、展示项目、支撑简历 |
| 文档输出 | Markdown | 编写 README、实验报告、源码阅读笔记 |

---

## 1. Linux

### 作用

Linux 是 AI Infra 项目的基础运行环境。大多数大模型推理框架、CUDA 工具链、NVIDIA 驱动、vLLM、SGLang、TensorRT 都更适合在 Linux 环境中运行。

### 在项目中的用途

```text
1. 安装 CUDA、PyTorch、vLLM；
2. 管理 Python 环境；
3. 运行推理服务；
4. 执行 benchmark 脚本；
5. 查看进程和端口；
6. 监控 GPU 资源；
7. 编译 CUDA 程序；
8. 使用 Nsight / nvidia-smi / shell 脚本。
```

### 需要掌握

```text
cd / ls / cp / mv / rm
vim / nano
grep / find
ps / top / htop
kill
tmux
ssh
scp
bash 脚本
环境变量
conda / venv
```

---

## 2. Python

### 作用

Python 是两个项目的主要工程语言之一。

在项目一中，它用于：

```text
1. 写 API client；
2. 写 benchmark；
3. 处理实验数据；
4. 画图；
5. 写 KV Cache 显存估算脚本；
6. 阅读 nano-vLLM / vLLM 上层代码。
```

在项目二中，它用于：

```text
1. 编写 Triton kernel；
2. 编写 PyTorch baseline；
3. 编写性能测试脚本；
4. 调用 CUDA extension 或外部可执行程序；
5. 处理实验结果。
```

### 需要掌握

```text
基础语法
函数和类
文件读写
argparse
json / csv
time / statistics
asyncio
httpx / aiohttp
numpy
pandas
matplotlib
```

---

## 3. PyTorch

### 作用

PyTorch 是理解模型推理和算子行为的基础。

它在项目中的作用包括：

```text
1. 加载和运行模型；
2. 理解 Tensor shape；
3. 作为算子 correctness baseline；
4. 作为算子性能 baseline；
5. 理解 torch.matmul、softmax、layer_norm 等算子；
6. 对比自写 CUDA/Triton 算子的输出是否正确。
```

### 需要掌握

```text
Tensor 创建与操作
shape / reshape / view / transpose
matmul
softmax
LayerNorm / RMSNorm 原理
torch.cuda.synchronize()
torch.no_grad()
torch.inference_mode()
torch.compile 可选
```

---

## 4. HuggingFace Transformers

### 作用

HuggingFace Transformers 是加载 Qwen、Llama、DeepSeek 等开源模型的重要工具，也是理解模型结构和 tokenizer 的入口。

### 在项目中的用途

```text
1. 下载和加载 Qwen 模型；
2. 查看 config.json；
3. 理解 hidden_size、num_layers、num_attention_heads、num_key_value_heads；
4. 使用 tokenizer；
5. 生成固定长度输入；
6. 对比 vLLM 输出结果。
```

### 需要掌握

```text
AutoTokenizer
AutoModelForCausalLM
model.config
tokenizer.encode
tokenizer.decode
safetensors
模型目录结构
config.json
generation_config.json
```

---

## 5. vLLM

### 作用

vLLM 是项目一的核心工业级推理框架，用于部署真实 LLM 推理服务。

### 在项目中的用途

```text
1. 启动 OpenAI-compatible API server；
2. 部署 Qwen 模型；
3. 测试不同并发请求；
4. 对比 FP16 / AWQ；
5. 观察 prefix cache、chunked prefill 等配置；
6. 理解 LLM Serving 的真实工程形态。
```

### 需要掌握

```text
vllm serve
OpenAI API server
max_model_len
max_num_seqs
max_num_batched_tokens
gpu_memory_utilization
enable_prefix_caching
enable_chunked_prefill
quantization
tensor_parallel_size
```

---

## 6. OpenAI-compatible API

### 作用

OpenAI-compatible API 是大模型服务常见接口形式。vLLM、SGLang、TGI 等框架通常都支持类似 OpenAI 的接口。

### 在项目中的用途

```text
1. 用统一方式调用本地模型服务；
2. 支持 chat/completions；
3. 支持 streaming；
4. 便于 benchmark；
5. 便于未来接入上层应用。
```

### 需要掌握

```text
/v1/chat/completions
/v1/completions
messages
max_tokens
temperature
stream=True
SSE 流式返回
HTTP request / response
```

---

## 7. Benchmark 工具链

### 作用

Benchmark 是 AI Infra 项目的核心。没有性能数据，项目就很难体现工程价值。

### 在项目中的用途

```text
1. 产生并发请求；
2. 控制输入长度和输出长度；
3. 统计 TTFT；
4. 统计 TPOT；
5. 统计总延迟；
6. 统计吞吐和 QPS；
7. 输出 CSV / JSON；
8. 为后续画图和分析提供数据。
```

### 推荐技术

```text
Python asyncio
httpx / aiohttp
time.perf_counter()
pandas
numpy
matplotlib
csv / json
```

---

## 8. TTFT / TPOT / Throughput / Latency

### 作用

这些是 LLM 推理服务的核心指标。

### 含义

```text
TTFT：Time To First Token，首 token 延迟，主要反映 prefill 和排队开销；
TPOT：Time Per Output Token，每个输出 token 的平均生成时间，主要反映 decode 效率；
Throughput：单位时间生成 token 数，反映系统吞吐能力；
Latency：请求总延迟，反映用户体验；
P95 Latency：95% 请求的延迟上界，反映尾延迟。
```

### 在项目中的用途

```text
1. 判断服务性能；
2. 比较不同配置；
3. 分析 Prefill / Decode；
4. 评估并发能力；
5. 评估量化收益；
6. 支撑简历和面试表达。
```

---

## 9. KV Cache

### 作用

KV Cache 是 LLM 推理显存优化的核心对象。

### 在项目中的用途

```text
1. 解释长上下文显存增长；
2. 解释 Decode 阶段为什么依赖显存带宽；
3. 分析 batch/concurrency 对显存的影响；
4. 理解 PagedAttention；
5. 理解 nano-vLLM 中 KV Cache 管理。
```

### 需要掌握

```text
为什么需要 KV Cache
K/V 分别是什么
Prefill 阶段如何生成 KV Cache
Decode 阶段如何读取和追加 KV Cache
KV Cache 显存估算公式
MHA / MQA / GQA 对 KV Cache 的影响
```

---

## 10. nano-vLLM

### 作用

nano-vLLM 是理解推理框架内部机制的轻量入口。

完整 vLLM 工程较复杂，初学者直接读容易被工程细节淹没。nano-vLLM 的作用是帮助你先理解最小闭环。

### 在项目中的用途

```text
1. 学习请求生命周期；
2. 理解 Scheduler；
3. 理解 KV Cache Manager；
4. 理解 ModelRunner；
5. 理解 Prefill / Decode；
6. 添加 profiling；
7. 实现小型优化机制；
8. 对照真实 vLLM 的设计。
```

### 需要掌握

```text
项目入口
请求对象
Scheduler
KV Cache block
ModelRunner
Sampler
Prefill
Decode
batch 组织方式
```

---

## 11. CUDA C++

### 作用

CUDA C++ 是 NVIDIA GPU 编程的核心工具，也是 AI Infra 底层算子开发的重要基础。

### 在项目中的用途

```text
1. 编写 GEMM kernel；
2. 理解 GPU 并行执行模型；
3. 理解 thread / block / grid；
4. 理解 shared memory；
5. 理解 memory coalescing；
6. 分析 bank conflict；
7. 使用 Nsight Compute 分析 kernel。
```

### 需要掌握

```text
__global__ kernel
threadIdx / blockIdx / blockDim / gridDim
global memory
shared memory
register
__syncthreads()
warp
memory coalescing
bank conflict
occupancy
CUDA event
cudaMemcpy
cudaMalloc
```

---

## 12. cuBLAS

### 作用

cuBLAS 是 NVIDIA 官方高性能 BLAS 库，GEMM 性能非常强。自写 GEMM 时需要用它作为性能 baseline。

### 在项目中的用途

```text
1. 对比自写 GEMM 性能；
2. 判断自己的优化程度；
3. 理解工业级 GEMM 的性能水平；
4. 作为面试中解释差距的参照。
```

### 需要掌握

```text
torch.matmul 背后可能调用 cuBLAS
cublasSgemm / cublasGemmEx 概念
TFLOPS 计算
相对 cuBLAS 性能比例
```

---

## 13. Triton

### 作用

Triton 是一种 Python-like GPU kernel 编程语言，比 CUDA 更容易快速实现自定义算子。

### 在项目中的用途

```text
1. 实现 RMSNorm；
2. 实现 Softmax；
3. 实现简化 Attention；
4. 快速验证 LLM 算子；
5. 与 PyTorch baseline 对比；
6. 提升项目工程效率。
```

### 需要掌握

```text
@triton.jit
tl.program_id
tl.arange
tl.load
tl.store
mask
block size
num_warps
grid
benchmark
```

---

## 14. RMSNorm

### 作用

RMSNorm 是当前许多 LLM 中常见的归一化算子，例如 Llama、Qwen 等模型中都会使用类似结构。

### 在项目中的用途

```text
1. 作为 Triton 小算子练习；
2. 理解 LLM 中小算子的优化；
3. 比较 PyTorch 和 Triton 实现性能；
4. 学习 memory-bound 算子的特点。
```

### 需要掌握

```text
RMSNorm 公式
hidden size 维度归一化
elementwise operation
为什么它通常是 memory-bound
如何用 Triton 实现
```

---

## 15. Softmax

### 作用

Softmax 是 Attention 中的关键操作，也是 CUDA/Triton 面试高频小算子。

### 在项目中的用途

```text
1. 实现 Triton softmax；
2. 理解 safe softmax；
3. 理解 online softmax；
4. 为 FlashAttention 学习做准备。
```

### 需要掌握

```text
safe softmax
max trick
exp
sum
normalization
数值稳定性
行级并行
memory access pattern
```

---

## 16. Attention

### 作用

Attention 是 Transformer 和 LLM 的核心计算模块，也是推理优化的核心对象。

### 在项目中的用途

```text
1. 实现 naive Attention；
2. 分析 attention score matrix 显存开销；
3. 理解 KV Cache；
4. 理解 FlashAttention；
5. 理解 Attention 在 Prefill / Decode 中的不同表现。
```

### 需要掌握

```text
Q / K / V
QK^T
scale
softmax
乘 V
causal mask
MHA / MQA / GQA
KV Cache
attention score matrix
```

---

## 17. FlashAttention

### 作用

FlashAttention 是大模型推理和训练中非常重要的 Attention 优化方法，核心是减少 HBM 读写。

### 在项目中的用途

```text
1. 作为 Attention 优化理论学习；
2. 解释为什么 naive Attention 显存开销大；
3. 理解 IO-aware 优化；
4. 作为面试高频知识点准备。
```

### 需要掌握

```text
不显式保存完整 attention matrix
分块计算
online softmax
减少 HBM 读写
SRAM / shared memory
长序列收益
```

---

## 18. Nsight Compute

### 作用

Nsight Compute 是分析单个 CUDA kernel 性能的核心工具。

### 在项目中的用途

```text
1. 分析 GEMM kernel；
2. 查看 memory throughput；
3. 查看 SM occupancy；
4. 查看 warp stall；
5. 查看 shared memory bank conflict；
6. 判断优化方向。
```

### 需要掌握

```text
Duration
SM Occupancy
Memory Throughput
L2 Hit Rate
Global Load Efficiency
Shared Memory Bank Conflict
Warp Stall Reasons
Roofline 简单理解
```

---

## 19. Nsight Systems

### 作用

Nsight Systems 更适合分析整体程序时间线，包括 CPU 调度、GPU kernel launch、CUDA memcpy、多个 kernel 的执行顺序。

### 在项目中的用途

```text
1. 分析 vLLM 推理整体 timeline；
2. 观察 CPU 和 GPU 是否存在空泡；
3. 分析 kernel launch overhead；
4. 判断瓶颈在框架调度还是 GPU kernel。
```

### 需要掌握

```text
timeline
CUDA kernel launch
CPU/GPU overlap
CUDA memcpy
stream
整体耗时分布
```

---

## 20. AWQ / INT4 量化

### 作用

AWQ 是常见大模型权重量化方法之一，INT4 量化可以显著降低模型权重显存。

### 在项目中的用途

```text
1. 对比 FP16 和 INT4 显存占用；
2. 分析量化对吞吐和延迟的影响；
3. 理解量化不能同比例降低 KV Cache 显存；
4. 评估量化推理的成本收益。
```

### 需要掌握

```text
权重量化
激活量化
INT4
FP16
显存节省
精度损失
为什么量化不一定加速
为什么 KV Cache 仍可能占大量显存
```

---

## 21. Git / GitHub

### 作用

GitHub 是项目展示和秋招简历的重要支撑。AI Infra 项目必须能被复现、被阅读、被面试官快速理解。

### 在项目中的用途

```text
1. 管理代码；
2. 保存实验脚本；
3. 写 README；
4. 展示性能图表；
5. 记录提交历史；
6. 形成可展示项目。
```

### 需要掌握

```text
git init
git add
git commit
git push
branch
README.md
.gitignore
release / tag 可选
```

---

## 22. Markdown

### 作用

Markdown 用于编写 README、实验报告、源码阅读笔记和面试复盘。

### 在项目中的用途

```text
1. 写项目说明；
2. 记录环境配置；
3. 记录启动命令；
4. 记录 Benchmark 结果；
5. 记录源码流程图；
6. 记录优化过程；
7. 形成可读性强的项目报告。
```

### 需要掌握

```text
标题
代码块
表格
图片引用
链接
目录
项目结构展示
```

---

# 第四部分：最终学习与项目完成标准

## 项目一完成标准

你做到以下程度，就可以写进简历：

```text
1. vLLM 能成功部署 Qwen 模型；
2. OpenAI-compatible API 能正常调用；
3. Benchmark 脚本能测 TTFT、TPOT、吞吐、延迟、显存；
4. 至少完成 3 组以上实验变量对比；
5. 有 FP16 vs AWQ 的对比；
6. 有 Prefill / Decode 分析；
7. 有 KV Cache 显存估算；
8. 有 nano-vLLM 源码阅读笔记；
9. 有 nano-vLLM profiling 修改；
10. 有完整 README 和实验报告。
```

## 项目二完成标准

你做到以下程度，就可以写进简历：

```text
1. 能写 naive CUDA GEMM；
2. 能写 shared memory tiled GEMM；
3. 能用 Nsight Compute 分析 GEMM；
4. 能说明 bank conflict、coalesced access、occupancy；
5. 能用 Triton 写 RMSNorm；
6. 能用 Triton 写 Softmax；
7. 能实现 naive Attention；
8. 能解释 FlashAttention 的核心思想；
9. 有 PyTorch/cuBLAS/Triton 性能对比；
10. 有完整 README 和优化报告。
```

## 两个项目合并后的能力画像

当你完成这两个项目后，你的简历能力画像可以概括为：

```text
具备 Linux/Python/PyTorch 基础，能够基于 vLLM 部署大语言模型推理服务并进行系统 Benchmark；
理解 LLM 推理中的 Prefill、Decode、KV Cache、Scheduler、量化和长上下文显存问题；
能够基于 nano-vLLM 复现轻量推理框架核心机制，并进行请求级 profiling 和 KV Cache 分析；
具备 CUDA/Triton 核心算子实现基础，能够实现 GEMM、RMSNorm、Softmax、Attention 等算子并使用 Nsight 进行性能分析。
```

这就是比较完整的 AI Infra 推理实习候选人画像。

---

# 第五部分：建议时间安排

如果每天能投入 3～5 小时，可以按 10～12 周推进。

## 第 1～2 周：基础准备 + vLLM 跑通

```text
Linux / Python 环境
PyTorch / Transformers 基础
vLLM 安装
Qwen 模型部署
OpenAI API 调用
```

## 第 3～4 周：Benchmark 脚本与实验

```text
benchmark.py
TTFT / TPOT 统计
并发实验
输入输出长度实验
FP16 / AWQ 实验
性能曲线
```

## 第 5～6 周：KV Cache 分析 + nano-vLLM

```text
KV Cache 显存估算
Prefill / Decode 分析
nano-vLLM 源码阅读
Scheduler / KV Cache / ModelRunner 笔记
profiling 修改
```

## 第 7～8 周：CUDA GEMM

```text
CUDA 基础
naive GEMM
shared memory tiled GEMM
Nsight Compute
cuBLAS 对比
```

## 第 9～10 周：Triton 小算子

```text
Triton 基础
RMSNorm
Softmax
简化 Attention
PyTorch 对比
```

## 第 11～12 周：报告、简历、面试准备

```text
整理 GitHub 仓库
写 README
写实验报告
整理图表
准备简历项目描述
准备面试问答
```

---

# 第六部分：最终简历项目组合推荐

最终简历上建议呈现两个项目：

## 项目 1

**基于 vLLM 与 nano-vLLM 的大语言模型推理服务性能分析与核心机制复现**

核心标签：

```text
vLLM / nano-vLLM / Qwen / Benchmark / TTFT / TPOT / KV Cache / Prefill / Decode / AWQ / Scheduler
```

## 项目 2

**CUDA/Triton 大模型核心算子实现与性能分析**

核心标签：

```text
CUDA / Triton / GEMM / RMSNorm / Softmax / Attention / Nsight Compute / shared memory / bank conflict / memory-bound / compute-bound
```

两个项目合起来，能够支撑你投递：

```text
AI Infra 实习
大模型推理工程实习
LLM Serving 实习
模型部署实习
推理框架实习
异构计算实习
CUDA/Triton 算子实习
端侧 AI 推理优化实习
```

---

# 第七部分：最终提醒

这两个项目不要做成“学习笔记项目”，而要做成“工程实验项目”。

每个项目都必须有：

```text
1. 可运行代码；
2. 可复现实验；
3. 性能数据；
4. 图表；
5. 分析报告；
6. 源码理解；
7. 面试可讲的技术细节。
```

面试官真正关心的不是你是否“听说过 vLLM、CUDA、Triton”，而是：

```text
你是否真的跑过？
你是否测过？
你是否改过？
你是否知道为什么快或慢？
你是否能用数据证明自己的判断？
```

最终目标：

**项目一让你讲清楚 LLM 推理服务和框架机制；项目二让你讲清楚 GPU 算子和性能优化。两个项目共同构成 AI Infra 推理方向的求职闭环。**


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-AI infra学习路线|模块-AI infra学习路线]]

%% 项目关联导航：结束 %%
