# AI Infra 面经源文件索引

更新时间：2026-07-02

用途：记录当前目录面经的主要价值，后续做答案库或模拟面试时方便回溯原文。

## 推理系统重点

| 文件 | 主要内容 |
|---|---|
| `面壁智能2026年 AI Infra 面经.md` | KV Cache、长会话、PagedAttention、Continuous Batching、Prefix Cache、量化、Prefill/Decode |
| `阿里云PAI AI Infra 岗位面试记录.md` | P/D 分离、Prefill 池/Decode 池、RDMA 传 KV Cache、MoE、PagedAttention、Radix Tree、LRU |
| `阿里云2026暑期实习Infra面经.md` | GPU 调度、P/D 分离、Prefix Cache、Cache-Aware Routing、LoRA 亲和性 |
| `字节跳动2026 AI Infra 算法岗面经.md` | KV Cache、PagedAttention、Continuous Batching、CUDA GEMM、FlashAttention、分布式 KV Cache |
| `字节infra一面，校招infra真的很吃香.md` | vLLM、PagedAttention、KV Cache、量化、DeepSeek 671B 负载均衡 |
| `腾讯infra一面面经.md` | vLLM 异步调度、投机解码、MoE、Continuous/Dynamic/Steady/Persistent Batching |
| `群核infra一面二面面经.md` | VLM continuous batching、Multi-LoRA、Cascade Attention、投机解码、KV Cache allocator |
| `快手 AI Infra 一面面经.md` | PD/AF 分离、FlashAttention V2、TP 切分、Ray、CUDA GEMM |
| `AI Infra 推理方向日常实习投递&#x2F;面试总结.md` | vLLM、sglang、RadixAttention、FlashDecoding、Chunked Prefill、Prefix Caching |

## CUDA / 算子 / GPU 底层

| 文件 | 主要内容 |
|---|---|
| `快手AIInfra校招面试.md` | H100/A100、warp、DP/TP-SP overlap、FlashAttention、PP 显存、CUDA 环境变量、Online Softmax |
| `快手 AI Infra 实习面经 （不含答案）.md` | 量化、CUDA GEMM、Shared Memory、CUDA Norm 算子 |
| `英伟达 AI Infra 实习面经.md` | GPU/并行计算项目、CUDA Kernel 优化、多线程手撕 |
| `阿里淘天AI-infra一面（已挂）.md` | FA、算子融合、Nsight 工具、KV Cache、GEMM 手撕 |
| `AI Infra人才计划面经 第一弹.md` | Online Softmax、Matrix Transpose、RMSNorm、GEMM、CUDA 手撕 |
| `ic转infra第一战：混元 infra面经（惨败）.md` | FP4、量化硬件、GPU 瓶颈分析、FlashAttention 维度推导 |
| `拼多多 AI Infra 面经.md` | Triton/CUDA、训练访存优化、KL 散度算子优化、手写 MHA |
| `百度infra一二面面经.md` | Prefix Cache 融合算子、DSA、Triton、访存效率、C++/Python 八股 |

## 量化重点

| 文件 | 主要内容 |
|---|---|
| `快手 AI Infra 实习面经 （不含答案）.md` | 量化方法、量化误差、异常值、INT8/FP8/INT4/FP4 |
| `AI infra暑期实习面经-阿里云.md` | AWQ、GPTQ、CUDA、VLM/VLA 推理 |
| `阿里国际AI Infra实习凉经.md` | INT8/INT4、显存优化、FlashAttention、推理优化分层 |
| `字节infra一面，校招infra真的很吃香.md` | 常见量化方法、对称/非对称量化 |
| `ic转infra第一战：混元 infra面经（惨败）.md` | FP4、量化矩阵乘、硬件量化流程 |
| `AI Infra 推理方向日常实习投递&#x2F;面试总结.md` | LLM.int8、SmoothQuant、AWQ、GPTQ |

## 分布式 / 并行 / MoE

| 文件 | 主要内容 |
|---|---|
| `Minimax ai infra一面面经.md` | ZeRO、DP/TP/PP、70B 显存估算、LoRA |
| `AI Infra 腾讯 烤面筋 面试题  BASE深圳.md` | AllReduce/Broadcast/ReduceScatter、并行策略、显存优化、AI 编译 |
| `AI Infra人才计划面经 第一弹.md` | FSDP/HSDP、3D 并行、MLA、MoE 蒸馏 |
| `腾讯infra一面面经.md` | EP、MoE 张量维度、MoE 通信方式 |
| `阿里ai infra一面.md` | 序列并行、Attention/FFN 差异、长序列、推理性能分析 |
| `快手AIInfra校招面试.md` | DP 与 TP-SP 计算通信 overlap、PP 显存峰值 |

## 云原生 / 平台工程

| 文件 | 主要内容 |
|---|---|
| `阿里云暑期实习AI infra面经.md` | K8s、PV/PVC、容器运行时、GPU 指标、DCGM、Nsight、vLLM/FlashAttention |
| `阿里云2026暑期实习Infra面经.md` | K8s Pod、GPU 调度、MIG、混部、KV Cache 命中率、智能路由 |
| `京东 AI Infra 实习一面面经（已入职）.md` | 高性能系统、C++、SIMD、Ray、RDMA、任务调度 |

## 项目深挖 / 求职复盘

| 文件 | 主要内容 |
|---|---|
| `阿里国际AI Infra实习凉经.md` | 项目表达缺陷、显存/算子/线上落地、推理优化分层 |
| `某大厂AI infra二面面经转发分享.md` | 项目深挖、异构计算、底层性能优化、图优化、推理优化 |
| `京东 AI Infra 实习一面面经（已入职）.md` | 项目价值、性能优化、多线程、锁、C++ 底层、SIMD |
| `华为ICTBG云软件研发部ai infra ai训推系统.md` | Qwen 微调、训练参数、模型评估、GPU 底层、KV Cache |
| `28届华为AI暑期实习timeline.md` | 项目效果、项目细节、实习经历、LeetCode 最大子数组和 |
| `6.2华为 ai 岗面经撕了2道，感觉不行了.md` | 手撕两道、项目困难与解决 |

## 手撕 / 编程题

| 文件 | 题目 |
|---|---|
| `AI Infra人才计划面经 第一弹.md` | LRU、带过期时间 LRU、链表、DP、Online Softmax、Matrix Transpose、RMSNorm、GEMM |
| `快手AIInfra校招面试.md` | LRU、Online Softmax、FlashAttention 伪代码 |
| `快手 AI Infra 实习面经 （不含答案）.md` | CUDA Norm 算子 |
| `英伟达 AI Infra 实习面经.md` | 时针分针夹角、三线程顺序打印 |
| `快手 AI Infra 一面面经.md` | LeetCode 单词接龙 |
| `群核infra一面二面面经.md` | KV Cache allocator、字母异位词变体、手撕 MHA |
| `百度infra一二面面经.md` | 非 hot100 mid，C++/Python 八股 |
| `阿里淘天AI-infra一面（已挂）.md` | GEMM shared memory 版本，不准调 cuBLAS |
| `快手实习AIInfra一面面经.md` | 有序数组转 BST |
| `28届华为AI暑期实习timeline.md` | 最大子数组和 |
| `6.2华为 ai 岗面经撕了2道，感觉不行了.md` | 二叉树平衡路径统计、二维 DP |

## Timeline / 参考价值较低

这些文件主要记录投递和流程，技术题较少：

- `华为ai岗实习offer➕timeline.md`
- `28届华为AI暑期实习timeline.md`
- `⭐华为暑期实习Ai infra工程师岗位面经 华为暑期实习招聘整理了网上常见的华为手机实习AI in.md`

## 后续处理建议

1. 先从推理系统重点文件开始，抽取可答题库。
2. CUDA/算子文件用于做手撕训练，不要只背概念。
3. 阿里云相关文件单独维护一套“K8s + GPU 调度 + MaaS 平台”答案。
4. 每次模拟面试后，把答不上来的问题追加到错题本。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-AI infra学习路线|模块-AI infra学习路线]]

%% 项目关联导航：结束 %%
