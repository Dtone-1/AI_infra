# AI Infra 高频题库索引

更新时间：2026-07-02

说明：本索引来自当前目录中的 AI Infra 面经 Markdown。频次是基于关键词的粗统计，用来判断优先级，不等同于严格题目去重计数。

## 高频信号

| 关键词 | 粗频次 | 判断 |
|---|---:|---|
| 量化 | 20 | 高频必会 |
| CUDA | 20 | 高频必会 |
| 并行 | 20 | 高频必会 |
| Prefill | 16 | 高频必会 |
| Decode | 15 | 高频必会 |
| KV Cache | 12 | 高频必会 |
| vLLM | 11 | 高频必会 |
| FlashAttention | 8 | 高频必会 |
| PagedAttention | 7 | 高频必会 |
| MoE | 7 | 高频必会 |
| Continuous Batching | 6 | 高频必会 |
| 张量并行 | 6 | 高频必会 |
| C++ | 6 | 基础筛选 |
| GEMM | 5 | 高频进阶 |
| LoRA | 5 | 中频 |
| K8s | 5 | 阿里云/平台岗重点 |
| 流水线并行 | 4 | 中高频 |
| TensorRT | 3 | 中频 |
| Online Softmax | 3 | 手撕高频 |
| LRU | 3 | 手撕基础 |

## 1. LLM 推理系统

优先级：最高。

核心题：

1. KV Cache 的存储原理是什么？显存占用如何估算？
2. 为什么长会话会越聊越卡？从 Decode、KV Cache、调度、上下文压缩角度解释。
3. PagedAttention 解决了什么问题？逻辑块和物理块如何映射？和 OS paging 类比时哪些地方相同、哪些地方不同？
4. Continuous Batching 和 Static/Dynamic Batching 的区别是什么？为什么能提升 GPU 利用率？
5. Prefill 和 Decode 的计算特征有什么不同？分别优化 TTFT 和 TBT 时应该看哪些指标？
6. P/D 分离为什么有效？Prefill 池和 Decode 池的并行度、路由、扩缩容策略有什么不同？
7. Prefix Cache 缓存的是什么？如何做 Cache-Aware Routing？公共 System Prompt 如何复用？
8. vLLM 相比传统推理框架的优势是什么？PagedAttention、continuous batching、chunked prefill、prefix caching 分别解决什么问题？
9. vLLM 怎么做异步调度？请求进入、调度、执行、回收 KV block 的生命周期是什么？
10. 投机解码解决什么问题？accept/reject 策略是什么？吞吐收益受哪些因素影响？
11. Multi-LoRA 推理、LoRA 亲和性路由如何设计？
12. 分布式 KV Cache 系统如何设计？一致性哈希、淘汰策略、一致性保障怎么做？

回答框架：

- 先讲瓶颈：显存容量、HBM 带宽、GPU 利用率、TTFT/TBT。
- 再讲机制：KV block 管理、调度策略、路由策略、缓存复用。
- 最后讲 trade-off：吞吐 vs 延迟、命中率 vs 负载均衡、碎片率 vs 管理开销、单机优化 vs 多机传输。

必须形成的产出：

- 能手画一个 vLLM 简化推理引擎流程图。
- 能手算 KV Cache 显存：`layers * 2 * seq_len * hidden_or_kv_dim * dtype_bytes`，并解释 MHA/GQA/MQA/MLA 的差异。
- 能解释一个请求从 prefill 到 decode 到释放 KV block 的生命周期。

## 2. CUDA / 算子 / GPU 底层

优先级：最高。

核心题：

1. CUDA 编程模型：grid、block、thread、warp、SM、shared memory、global memory、L2、HBM 分别是什么？
2. Warp 是什么？warp divergence、occupancy、SM active、tensor core active 分别说明什么？
3. 优化 CUDA Kernel 通常从哪些方面入手？
4. CUDA GEMM 的常见优化方法有哪些？
5. Shared Memory 大小如何选？如何考虑 tile size、bank conflict、occupancy、寄存器压力？
6. FlashAttention 的核心原理是什么？为什么能减少 HBM 读写？v1/v2/v3 的主要差异是什么？
7. Online Softmax 为什么是 FlashAttention 的基础？如何写伪代码？
8. Attention 算子和普通矩阵乘算子的优化差异是什么？
9. Prefix Cache 融合算子融合了哪些操作？算子访存效率和计算利用率如何评估？
10. Triton 和 CUDA 写算子的主要区别是什么？什么时候用 Triton，什么时候用 CUDA？
11. TensorRT 做了哪些优化？图优化、算子融合、精度选择、kernel auto-tuning 如何串起来？
12. Nsight Systems 和 Nsight Compute 分别看什么？如何定位 kernel launch、通信等待、访存瓶颈、tensor core 未打满？

回答框架：

- 算子优化按四层讲：算法重排、访存优化、并行划分、硬件指令/张量核利用。
- GEMM 按五点讲：tiling、shared memory staging、coalesced access、register blocking、tensor core / wmma。
- 性能分析按三步讲：看 timeline 找空泡，看 kernel 指标判断 compute/memory bound，看 roofline 判断优化方向。

必须形成的产出：

- 手写或伪代码：LayerNorm/RMSNorm、Online Softmax、Matrix Transpose、简化 GEMM、FlashAttention 关键循环。
- 能用 roofline 解释一个 kernel 是算力瓶颈还是带宽瓶颈。
- 能说明 shared memory bank conflict 的产生和规避方式。

## 3. 量化

优先级：最高。

核心题：

1. 常见量化方法有哪些？PTQ、QAT、LLM.int8、SmoothQuant、AWQ、GPTQ 各自解决什么问题？
2. INT8、INT4、FP8、FP4 的表示能力和适用场景有什么差异？
3. 对称量化和非对称量化的区别是什么？
4. per-tensor、per-channel、per-group 量化如何选择 group size？
5. Calibration 的核心作用是什么？校准集如何选？
6. 量化后精度下降如何排查？异常值、激活分布、敏感层、混合精度怎么处理？
7. 为什么异常值会影响量化？SmoothQuant 如何把激活异常值迁移到权重侧？
8. 为什么数据分布越均匀通常量化效果越好？为什么到 FP4 时这个直觉不一定适用？
9. 量化矩阵乘的维度和反量化流程是什么？硬件上需要哪些支持？
10. 端侧、云端、跨境低延迟场景下，量化方案如何取舍？

回答框架：

- 先讲目标：省显存、降带宽、提升吞吐，但必须控制精度损失。
- 再讲方法：权重量化、激活量化、KV Cache 量化、混合精度。
- 最后讲排查：先定位掉点层，再看分布和异常值，最后决定是否回退精度或做 QAT。

必须形成的产出：

- 能对比 AWQ、GPTQ、SmoothQuant 的核心思想。
- 能解释 scale/zero point、group size、outlier 对量化误差的影响。
- 能给出一套“量化掉点排查流程”。

## 4. 并行训练 / 分布式 / MoE

优先级：高。

核心题：

1. DP、TP、PP、SP、EP 分别是什么？各自切什么维度？
2. ZeRO Stage 1/2/3 分别切分 optimizer state、gradient、parameter 的哪些部分？
3. FSDP、HSDP 的通信模式有什么差异？4 机 32 卡 H200 从 FSDP 换 HSDP，通信时间为什么可能下降？
4. 训练 70B 模型如何粗略估算单卡显存？
5. 3D 并行如何设计？训练显存在 3D 并行下如何估计？
6. TP 在 Transformer 线性层中如何切分？为什么 attention 和 FFN 的切分方式不同？
7. PP 使用和不使用时，显存峰值是否相同？bubble 如何估算？
8. MoE 的运行过程是什么？router、top-k expert、dispatch、combine 的张量维度怎么变化？
9. MoE 并行如何通信？All-to-All、expert parallel、计算通信 overlap 如何做？
10. AllReduce、Broadcast、ReduceScatter 的适用场景是什么？NCCL 通信瓶颈如何定位？
11. RDMA 在 P/D 分离、分布式 KV Cache、跨机训练中解决什么问题？

回答框架：

- 并行题先讲“切什么”：样本、张量、层、序列、专家。
- 再讲“省什么”：显存、通信量、计算重复、流水线空泡。
- 最后讲“代价”：通信、同步、负载均衡、实现复杂度。

必须形成的产出：

- 能画出 DP/TP/PP/EP 的切分图。
- 能手算一个 70B 模型训练显存的粗略组成：参数、梯度、优化器状态、激活、通信 buffer。
- 能解释 MoE All-to-All 为什么容易成为瓶颈，以及如何 overlap。

## 5. 云原生 / GPU 调度 / 监控

优先级：阿里云、平台工程、MaaS 岗高；纯算子岗中。

核心题：

1. K8s Pod 创建流程是什么？如何结合推理服务讲 GPU 资源声明、节点标签、拓扑感知、镜像预热？
2. PV/PVC、emptyDir、hostPath、OSS、NAS、Longhorn 的区别是什么？Pod 漂移时数据怎么办？
3. Docker、containerd、rkt、轻量容器有什么区别？高版本 K8s 为什么不再直接依赖 Docker？
4. HPA、VPA、蓝绿发布、金丝雀发布如何用于推理服务？
5. GPU 调度器如何设计？显存、显存带宽、KV Cache 命中率、模型亲和性、LoRA 亲和性如何进入调度指标？
6. MIG、整卡、混部如何取舍？如何避免长短请求互相拖慢？
7. 多卡集群 GPU 指标监测如何设计？nvidia-smi 的显存占用为什么不够？
8. DCGM exporter、cAdvisor、KSM、NVML 分别提供什么信息？
9. Nsight Systems / Nsight Compute 如何补足 DCGM 指标？

回答框架：

- 平台题要贴 AI 场景，不要只背 K8s 组件。
- 调度题按资源维度讲：CPU、内存、显存、带宽、拓扑、缓存命中、模型亲和性。
- 监控题按层级讲：业务指标、框架指标、GPU 指标、kernel 指标。

必须形成的产出：

- 能设计一个 MaaS 推理平台的调度指标表。
- 能解释一个请求如何从网关路由到具体 GPU 实例。
- 能说清楚如何发现 tensor core 没打满或精度退化。

## 6. 模型结构 / 训练微调

优先级：中高。

核心题：

1. Transformer 相比 MLP 的核心优势是什么？
2. MHA、MQA、GQA、MLA 的设计思路与 KV Cache 占用差异是什么？
3. MLA 的矩阵吸收是什么？Prefill 和 Decode 阶段是否都要做？
4. RoPE 相对传统位置编码的优点是什么？
5. Llama 1-3、Qwen、DeepSeek MTP、MoE 模型有哪些结构变化？
6. LoRA 的核心思想是什么？为什么低秩分解能减少训练参数？
7. LoRA 中 A、B 矩阵为什么初始化方式不同？
8. 微调 Qwen 时训练参数怎么设？如何评估训练出来的模型？
9. VLM/VLA 推理和纯 LLM 推理有什么不同？多模态 continuous batching 怎么处理？
10. 扩散模型推理、多模态架构、文生图 feature cache 的基础思路是什么？

回答框架：

- 模型结构题要落到 infra 影响：显存、访存、并行、cache、调度。
- 微调题要讲数据、参数、训练稳定性、评估、部署。

必须形成的产出：

- 能从 KV Cache 角度解释 MHA/MQA/GQA/MLA。
- 能把 LoRA 和 Multi-LoRA 推理路由联系起来。
- 能解释 VLM 请求为什么比纯文本请求更难 batch。

## 7. 项目深挖

优先级：最高。很多面试 50%-70% 时间都在问项目。

高频追问：

1. 你项目解决的核心问题是什么？
2. 业务价值是什么？数据规模、QPS、延迟、显存、吞吐提升多少？
3. 最大技术难点是什么？为什么难？
4. 你具体负责哪部分？不是团队做了什么，而是你做了什么。
5. 性能瓶颈如何定位？为什么判断瓶颈在计算、访存、通信、调度或数据传输？
6. 为什么选择这个方案？替代方案是什么？trade-off 是什么？
7. 线上线下效果是否一致？如果不一致怎么解释？
8. 有没有真实落地？如何监控、回滚、灰度？
9. 如果迁移到阿里/字节/腾讯/快手场景，你会怎么改？
10. 如果让你重新做一次，最先优化哪里？

回答模板：

1. 背景：业务/系统目标是什么。
2. 瓶颈：原系统卡在哪里，用什么指标证明。
3. 方案：核心技术路径和关键实现。
4. 结果：量化指标，最好有延迟、吞吐、显存、成本。
5. 反思：局限、可扩展方向、下一步。

必须形成的产出：

- 准备 2 个项目的 1 分钟版、3 分钟版、10 分钟深挖版。
- 每个项目至少准备 5 个可量化指标。
- 每个项目准备 3 个被质疑时的防守点。

## 8. 编程 / 手撕

优先级：高。Infra 岗代码能力筛选强，C++ 和 CUDA 都要准备。

LeetCode / C++ 高频：

1. LRU，带过期时间的 LRU。
2. 滑动窗口最大值。
3. 最大子数组和。
4. 二叉树平衡路径统计。
5. 二维 DP。
6. 单词接龙。
7. 字母异位词变体。
8. 链表相交，可能有环。
9. 删除单链表节点，只给 node 指针。
10. 三线程顺序打印递增数字。
11. 时间字符串求时针分针夹角。

CUDA / Triton 高频：

1. LayerNorm / RMSNorm。
2. Online Softmax。
3. Matrix Transpose。
4. 大矩阵规约求和。
5. 简化 GEMM，不调用 cuBLAS。
6. FlashAttention 伪代码。
7. KV Cache allocator。
8. 手写 Multi-Head Attention。

C++ 八股：

1. `std::forward`、`std::move`、`enable_if`。
2. `emplace_back` 和 `push_back`。
3. 常量指针和指针常量。
4. 协程：有栈/无栈，C++20 协程特点。
5. STL 优化：为什么 `set` 慢，为什么 array + hash 更快？
6. 内存局部性：时间局部性 vs 空间局部性。
7. 多线程写竞争、锁、无锁设计。

准备标准：

- LeetCode：能 20 分钟内写出 mid 题，并讲复杂度。
- C++：能解释语言机制背后的性能影响。
- CUDA：至少能写出伪代码，能讲清 block/thread 划分、访存、同步和边界处理。

## 第一优先级清单

如果只给两周，先攻这些：

1. KV Cache 显存估算 + PagedAttention。
2. Prefill/Decode 差异 + P/D 分离。
3. Continuous Batching + vLLM 调度生命周期。
4. FlashAttention + Online Softmax。
5. CUDA GEMM 优化 + shared memory / bank conflict。
6. INT8/INT4/FP8/FP4 + AWQ/GPTQ/SmoothQuant。
7. DP/TP/PP/EP + ZeRO/FSDP。
8. MoE 运行流程 + All-to-All。
9. 项目深挖模板。
10. LRU、滑动窗口、LayerNorm/RMSNorm、Online Softmax 手撕。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-AI infra学习路线|模块-AI infra学习路线]]

%% 项目关联导航：结束 %%
