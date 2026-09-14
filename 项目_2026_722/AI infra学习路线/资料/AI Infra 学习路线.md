# AI Infra 学习路线

目标：把面经里的零散问题变成能面试、能追问、能写代码的能力。

## 总体策略

不要按资料顺序学，按面试命中率学。

学习顺序：

1. 先掌握推理系统主线：KV Cache、PagedAttention、Continuous Batching、Prefill/Decode。
2. 再补 CUDA 和算子：GEMM、FlashAttention、Online Softmax、LayerNorm/RMSNorm。
3. 同步准备量化：INT8/INT4/FP8/FP4、AWQ、GPTQ、SmoothQuant。
4. 再学分布式和平台：DP/TP/PP/EP、MoE、ZeRO、K8s GPU 调度。
5. 每周都练手撕和项目表达。

## 第 0 阶段：建立个人基线

用 1 天完成。

产出：

- 写出你自己的 2 个项目，每个项目 10 行以内。
- 标出项目是否涉及：推理、训练、量化、算子、CUDA、分布式、K8s、性能优化。
- 列出当前不会的高频点，分成“完全不会”“知道概念但讲不深”“能讲但写不出”。

自测问题：

1. 你能不能 3 分钟讲清楚 AI Infra 和算法岗的区别？
2. 你能不能用自己的项目回答“你到底优化了什么”？
3. 你能不能说出一个性能指标，并解释它为什么变好了？

## 第 1 阶段：LLM 推理系统主线

建议 4-5 天。

必须掌握：

- KV Cache 原理和显存估算。
- MHA/MQA/GQA/MLA 对 KV Cache 的影响。
- Prefill 和 Decode 的差异。
- PagedAttention。
- Continuous Batching。
- Prefix Cache。
- P/D 分离。
- vLLM 请求调度生命周期。

学习顺序：

1. 先手算 KV Cache 显存。
2. 再理解为什么 Decode 是状态密集/带宽敏感。
3. 再学 PagedAttention 如何解决碎片和分配问题。
4. 再学 Continuous Batching 如何让请求按 step 插队。
5. 最后把它们串成一个简化推理引擎。

练习：

- 画图：用户请求进入推理服务后，从 prefill 到 decode 到释放 KV block 的流程。
- 口述：Prefill/Decode 分别怎么优化 TTFT/TBT。
- 设计：给 System Prompt 很多的业务，如何做 Prefix Cache 和路由。

通过标准：

- 能解释“为什么 vLLM 快”，不只说用了 PagedAttention。
- 能回答“为什么 P/D 分离后还要考虑 KV Cache 传输和路由”。
- 能把长会话卡顿归因到 KV 访问、上下文长度、调度和缓存管理。

## 第 2 阶段：CUDA / 算子基础

建议 5-7 天。

必须掌握：

- CUDA 编程模型：grid、block、thread、warp、SM。
- 内存层级：register、shared memory、L1/L2、global memory、HBM。
- 访存合并、bank conflict、occupancy、warp divergence。
- GEMM 优化套路。
- FlashAttention 和 Online Softmax。
- LayerNorm/RMSNorm 手写思路。

学习顺序：

1. 先学 CUDA 执行模型和内存层级。
2. 再写 Matrix Transpose 和 Reduction，理解访存和同步。
3. 再写 LayerNorm/RMSNorm。
4. 再学 GEMM tiling。
5. 最后学 FlashAttention。

练习：

- 写 Online Softmax 伪代码。
- 写 RMSNorm CUDA kernel 伪代码。
- 讲 GEMM 的 tile 怎么放 shared memory。
- 用 roofline 解释一个 kernel 是 compute bound 还是 memory bound。

通过标准：

- 面试官问“CUDA kernel 怎么优化”，你能按访存、并行、计算、调度、profiling 分层讲。
- 面试官问“shared memory 怎么选”，你能讲 tile size、occupancy、寄存器压力、bank conflict。
- 面试官问 FlashAttention，你能说清为什么少写 HBM，以及 online softmax 如何保证数值稳定。

## 第 3 阶段：量化

建议 3-4 天。

必须掌握：

- PTQ、QAT。
- 对称/非对称量化。
- per-tensor、per-channel、per-group。
- INT8、INT4、FP8、FP4。
- AWQ、GPTQ、SmoothQuant、LLM.int8。
- 量化掉点排查。

学习顺序：

1. 先理解 scale、zero point、clipping、calibration。
2. 再比较 INT 和 FP 低精度格式。
3. 再看 outlier 对激活量化的影响。
4. 最后学 AWQ/GPTQ/SmoothQuant 的动机和适用场景。

练习：

- 解释为什么异常值会导致量化误差变大。
- 设计一个量化掉点排查流程。
- 对比 AWQ、GPTQ、SmoothQuant。
- 回答“为什么 FP4 下均匀分布不一定更好”。

通过标准：

- 能从“精度、显存、吞吐、硬件支持”四个角度选量化方案。
- 能解释 group size 越小和越大的 trade-off。
- 能把量化和 GEMM、Tensor Core、反量化流程联系起来。

## 第 4 阶段：分布式训练 / MoE

建议 4-5 天。

必须掌握：

- DP、TP、PP、SP、EP。
- ZeRO Stage 1/2/3。
- FSDP/HSDP。
- 3D 并行。
- MoE router、dispatch、combine、All-to-All。
- AllReduce、ReduceScatter、Broadcast。

学习顺序：

1. 先掌握每种并行“切什么维度”。
2. 再学它们分别节省什么显存或通信。
3. 再学通信算子的使用场景。
4. 最后学 MoE 的负载均衡和通信 overlap。

练习：

- 手算 70B 模型训练显存粗估。
- 画 DP/TP/PP/EP 的通信图。
- 解释 TP 下 Transformer 线性层怎么切。
- 解释 MoE 的 All-to-All 为什么容易成为瓶颈。

通过标准：

- 面试官给出机器数和卡数，你能提出合理并行策略。
- 面试官问显存估算，你能拆成参数、梯度、优化器、激活、buffer。
- 面试官追问 MoE 通信，你能讲 token dispatch、expert 负载和 overlap。

## 第 5 阶段：平台工程 / K8s / 监控

建议 3 天。重点准备阿里云、MaaS、平台方向。

必须掌握：

- Pod 创建流程。
- GPU 资源声明、节点标签、亲和性、污点和容忍。
- PV/PVC、emptyDir、hostPath、OSS、NAS。
- HPA/VPA、蓝绿、金丝雀。
- DCGM exporter、cAdvisor、KSM、NVML。
- Nsight Systems / Nsight Compute。
- GPU 调度指标设计。

练习：

- 设计一个推理服务的调度器：输入请求，输出实例选择。
- 说明如何把 KV Cache 命中率、模型亲和性、显存余量放入调度策略。
- 说明为什么 nvidia-smi 不足以判断 LLM 推理是否跑得好。

通过标准：

- 能把 K8s 八股转成 AI 推理平台语言。
- 能讲清监控指标从业务层到 kernel 层的层级。
- 能解释 MIG、整卡、混部的选择。

## 第 6 阶段：项目表达和模拟面试

每周都做，最后集中 3 天。

项目材料结构：

1. 背景：为什么做。
2. 问题：瓶颈在哪里。
3. 方案：你做了什么。
4. 指标：提升多少。
5. 复盘：有什么 trade-off。

每个项目都要准备：

- 1 分钟版：面试开场。
- 3 分钟版：常规深挖。
- 10 分钟版：技术细节。
- 3 个质疑点：为什么不用别的方案、指标是否可信、线上是否能落地。
- 3 个迁移场景：阿里云、字节、快手/腾讯。

模拟面试安排：

- 第一次：只问项目，看表达是否具体。
- 第二次：项目 + 推理系统。
- 第三次：CUDA/量化。
- 第四次：并行/MoE/K8s。
- 第五次：全流程压力面。

## 两周冲刺版

适合已经有基础、近期要面试。

Day 1：整理项目，写 1/3/10 分钟版本。

Day 2：KV Cache、MHA/MQA/GQA/MLA、显存估算。

Day 3：PagedAttention、Continuous Batching、vLLM 生命周期。

Day 4：Prefill/Decode、P/D 分离、Prefix Cache、调度。

Day 5：FlashAttention、Online Softmax。

Day 6：CUDA 基础、GEMM、shared memory、bank conflict。

Day 7：手撕 LayerNorm/RMSNorm、LRU、滑动窗口。

Day 8：量化基础、INT8/INT4/FP8/FP4。

Day 9：AWQ、GPTQ、SmoothQuant、掉点排查。

Day 10：DP/TP/PP/EP、ZeRO、FSDP。

Day 11：MoE、All-to-All、通信 overlap。

Day 12：K8s GPU 调度、监控、Nsight/DCGM。

Day 13：公司风格模拟面试：阿里云/字节/快手。

Day 14：错题回炉，补最容易被追问打穿的点。

## 每日固定动作

- 30 分钟：复述一个高频题，录音或写下来。
- 60 分钟：学一个主题，做结构化笔记。
- 45 分钟：写一道 LeetCode/C++ 或 CUDA 伪代码。
- 30 分钟：把当天内容和自己的项目关联起来。

## 暂不优先

这些不是不重要，而是当前面经命中率较低，放在主线之后：

- 过深的 TVM 自动调优细节。
- 非主流芯片 ISA 适配细节。
- 扩散模型完整推理系统。
- VLA 端侧全链路优化。
- FlashAttention v4 细节。

## 当前最推荐的下一步

先从“KV Cache + PagedAttention + Continuous Batching”开始。它们在多家公司反复出现，而且能自然串起 vLLM、长会话、显存、调度、P/D 分离和系统设计。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-AI infra学习路线|模块-AI infra学习路线]]

%% 项目关联导航：结束 %%
