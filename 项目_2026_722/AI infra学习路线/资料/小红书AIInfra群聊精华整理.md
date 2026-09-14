# 小红书 AI Infra 群聊精华整理：实习、求职、项目与学习路线

> 整理说明：本文基于用户提供的小红书 AI Infra 群聊截图整理。为了保护隐私，已弱化具体昵称和头像信息，重点保留对 AI Infra 求职、实习、项目设计、CUDA/Triton、vLLM/nano-vLLM、量化、端侧与云侧推理等有价值的信息。  
> 整理目标：把零散群聊内容转化为可复习、可执行、可用于规划项目和面试准备的学习资料。

---

## 0. 总结论

这批群聊的核心结论可以概括为：

**AI Infra 找实习/找工作，不能只停留在“我学了 vLLM / CUDA / 量化 / SGLang”这种概念层面。真正有价值的是：能跑通框架、能改框架、能写或理解关键算子、能做 benchmark/profile、能解释优化前后的性能变化，最好还能把论文里的优化点复现进自己的项目里。**

也就是说，项目不能只是“调库”或“部署成功”，而要能体现：

- 你理解了系统内部流程；
- 你知道瓶颈在哪里；
- 你做过性能测试；
- 你改过代码；
- 你能解释优化逻辑；
- 你能用数据证明优化有效；
- 你能把论文、源码、工具和工程问题联系起来。

最重要的一句话：

**不要只做“能跑”的项目，要做“能解释为什么快、为什么慢、怎么优化、优化多少”的项目。AI Infra 面试真正看的就是这个。**

---

# 一、群聊中最有价值的核心结论

## 1. 项目不能只是“调库”，否则面试官会觉得没亮点

群聊中有群友提到自己做过：

```text
1. 用 CUDA 写了一个 MoE 算子；
2. 在 SGLang 平台上做了一个混合模型的量化；
3. 做了一些敏感层分析。
```

但他的反馈是：

```text
很多面试官认为项目没有很大的亮点，
因为确实算是比较基础的调库。
```

这个反馈非常关键。它说明 AI Infra 项目不是“我部署了 vLLM”“我跑了 SGLang”“我用了量化工具”就够了。

面试官会继续追问：

```text
你具体改了哪里？
为什么这么改？
优化了哪个指标？
性能提升多少？
显存降低多少？
profile 结果是什么？
这个优化是否有实际工程意义？
```

所以群友建议：

```text
找论文复现一下，把论文里的优化加入自己的项目模块中。
```

这句话可以作为你后续做项目的核心原则。

普通项目：

```text
我用 vLLM 部署了 Qwen 模型，并测试了吞吐。
```

更有亮点的项目：

```text
我基于 nano-vLLM 实现了某个论文中的 KV Cache 优化 / Attention 优化 / 量化策略，
并对比了优化前后的 TTFT、TPOT、显存占用和吞吐。
```

后者更像真正的 AI Infra 项目。

---

## 2. 小厂、大厂、实习面试都很重视项目细节

群聊中有人提到：

```text
小米一面没有算法题，
而是手撕项目中设计的代码，
重点拷打项目。
```

这说明 AI Infra 面试里，项目不是简历装饰，而是真会被逐行追问。

你以后做项目时，不能只会讲“项目背景”，必须准备：

```text
1. 项目为什么做；
2. 技术路线为什么这么选；
3. 你具体写了哪些代码；
4. 哪些模块是你自己实现的；
5. 遇到什么 bug；
6. 性能瓶颈在哪；
7. 如何用 profile 定位；
8. 优化前后数据；
9. 和现有框架/论文/库相比有什么不同；
10. 如果继续做，还能怎么提升。
```

这对你后面做 nano-vLLM 或 vLLM benchmark 项目非常重要。

---

## 3. “nano-vLLM + 论文优化”比单纯 SGLang 调库更适合做简历项目

群聊中有一个判断非常适合你：

```text
现在感觉把第二个项目换成 nano-vLLM 加一些论文的优化可能好一些。
```

原因是：如果你现在直接啃 vLLM / SGLang 源码，工程复杂度太高，容易陷入“看不懂、改不动、讲不清”的状态。

更合理的路线是：

```text
mini-vLLM / nano-vLLM
    ↓
理解最小推理框架
    ↓
实现自己的小优化
    ↓
做 benchmark
    ↓
再迁移到 vLLM / SGLang 的真实机制
```

群聊中也有人建议：

```text
mini-vLLM 看 GitHub；
看完后再看 vLLM；
在 mini 基础上实现点自己的东西；
框架是这样的。
```

这基本就是你当前最适合的路线。

---

# 二、AI Infra 推理岗位真实工作内容

群聊里有群友对 AI Infra 工作内容做了比较真实的描述：

```text
框架并不能支持所有模型在所有卡上很好地跑起来。
大部分时候做的就是适配新的模型、新的卡。
先做到跑起来，再把性能优化达标。
量化、多卡通信、PD 分离、算子融合等等，能做的东西太多了。
```

这段话可以拆成 AI Infra 推理岗位的真实工作流。

---

## 1. 第一步：适配模型和硬件

公司里不是所有模型都能在所有卡上直接跑。

可能要适配的新模型包括：

```text
Qwen
DeepSeek
Llama
MoE 模型
混合模型
多模态模型
```

可能要适配的新硬件包括：

```text
NVIDIA GPU
国产 GPU
NPU
端侧芯片
车端芯片
机器人芯片
```

可能涉及的框架包括：

```text
vLLM
SGLang
TensorRT-LLM
llama.cpp
Triton Inference Server
公司内部推理框架
```

很多工作不是发明新理论，而是：

```text
让模型在目标硬件和目标框架上先跑起来。
```

这就是模型适配、硬件适配、backend 适配。

---

## 2. 第二步：性能优化达标

跑起来只是第一步，后面还要看性能：

```text
吞吐是否达标；
TTFT 是否够低；
TPOT 是否够低；
显存是否够省；
多卡通信是否成为瓶颈；
长上下文是否能支撑；
量化后精度是否能接受；
是否能支持线上并发。
```

所以 AI Infra 不是“能跑就行”，而是：

```text
跑起来
    ↓
跑得快
    ↓
跑得稳
    ↓
跑得省
    ↓
能服务真实业务
```

---

## 3. 可以做的优化方向很多

群聊中提到的优化方向包括：

```text
量化
多卡通信
PD 分离
算子融合
新模型适配
新卡适配
敏感层分析
MoE 算子
Attention 算子
性能 profiling
```

可以进一步分成几类：

| 类别 | 具体内容 | 是否适合当前阶段 |
|---|---|---|
| 框架适配 | 模型在 vLLM/SGLang 上跑起来 | 适合 |
| 显存优化 | KV Cache、PagedAttention、Prefix Cache | 适合 |
| 性能测试 | TTFT、TPOT、吞吐、显存 | 非常适合 |
| 量化 | INT8/INT4/FP8/FP4、敏感层分析 | 可以逐步学 |
| 算子优化 | Attention、MoE、GEMM、FlashAttention | 后续重点 |
| 多卡通信 | TP、PP、EP、NCCL、All-to-All | 先了解，后深入 |
| PD 分离 | Prefill / Decode 分离部署 | 进阶内容 |
| 端侧适配 | 交叉编译、端侧芯片部署 | 和嵌入式背景相关 |

---

# 三、CUDA / Triton / 算子相关结论

## 1. 面试中 CUDA 仍然很重要

群聊中多次提到：

```text
面试手写 CUDA 的还挺多；
GEMM 最常见；
GEMM 搞明白了 CUDA 基本上也就入门了；
有人一面考了 CUDA 的 GEMM 变种；
还会问做了什么优化。
```

这说明：即使目标是推理框架岗，也不能完全不学 CUDA。

尤其是如果投递方向是：

```text
AI Infra
推理引擎
高性能计算
异构计算
模型部署优化
国产芯片 Runtime
端侧推理加速
```

CUDA/Triton 基础都会加分。

---

## 2. 但框架岗不一定每天写 CUDA

群聊中也有人说：

```text
框架岗进去大部分不会写算子；
我手撕是准备的 Triton，应付面试的。
```

这说明：

```text
工作内容 ≠ 面试内容。
```

真实工作中，推理框架岗可能更多写：

```text
Python/C++ 框架逻辑；
调度；
KV Cache 管理；
模型适配；
benchmark；
服务接口；
多卡并行；
框架 bug 修复。
```

但面试时仍可能问：

```text
GEMM；
Softmax；
Attention；
RMSNorm；
CUDA memory hierarchy；
shared memory；
bank conflict；
warp；
Tensor Core；
Triton kernel。
```

所以你的策略应该是：

```text
不要把自己训练成纯 CUDA 算子选手；
但要具备足够应对面试和理解框架底层的 CUDA/Triton 能力。
```

---

## 3. CUDA 学到什么程度算入门？

群聊中有几个判断：

```text
GEMM 搞明白了，CUDA 基本上就入门了；
一般会从朴素实现开始；
然后分块、流水线、向量化；
再问 bank conflict 等八股；
优化 GEMM 是一个过程，不是只看结果；
每一步用 NCU profile，再决定下一步如何优化。
```

所以 CUDA 项目不是“写出一个 GEMM”就结束，而是要体现优化路径：

```text
naive GEMM
    ↓
shared memory tiling
    ↓
register tiling
    ↓
向量化加载
    ↓
减少 bank conflict
    ↓
warp-level 优化
    ↓
pipeline
    ↓
Tensor Core
    ↓
对比 cuBLAS 性能
    ↓
用 Nsight Compute 分析瓶颈
```

面试时不要只说：

```text
我优化了 GEMM。
```

更好的表达是：

```text
我先写了 naive 版本，用 NCU 看 memory throughput 和 SM occupancy；
发现 global memory 访问重复严重；
然后做 block tiling，把数据搬到 shared memory；
再做 register blocking，减少 shared memory 访问；
之后处理 bank conflict 和访存合并；
最后和 cuBLAS 做性能对比。
```

这才像 AI Infra 候选人。

---

## 4. 算子重点不是所有都学，优先 Attention 和 MoE

群聊中有人说：

```text
算子主要还是 Attention 和 MoE；
这两个最关键；
写一写 Attention，还有 MoE；
bf16 精度就行；
量化什么的都不用太管；
然后要会用 Tensor Core。
```

这个判断适合大模型推理方向。

LLM 里最核心的计算主要集中在：

```text
Attention
MLP / GEMM
MoE
RMSNorm / LayerNorm
Softmax
RoPE
Sampling
```

其中最容易体现大模型特色的是：

```text
Attention / FlashAttention
MoE / Fused MoE
```

时间有限时，推荐顺序是：

```text
1. GEMM：CUDA 入门核心；
2. Softmax / LayerNorm：经典小算子；
3. Attention：理解大模型核心；
4. FlashAttention：体现 IO-aware 优化；
5. MoE：适配 DeepSeek/Qwen-MoE 等模型；
6. Fused MoE：进阶。
```

---

## 5. naive Attention 还是 FlashAttention？

群聊中有人问：

```text
naive attention 还是 flash attention？
```

回答倾向于：

```text
FlashAttention 肯定更有价值。
```

但对于初学者，不应该直接跳到 FlashAttention。合理路线应该是：

```text
先写 naive attention
    ↓
理解 QK^T、scale、softmax、V
    ↓
理解显存读写为什么浪费
    ↓
理解 online softmax
    ↓
再学 FlashAttention
```

如果直接背 FlashAttention，不理解 naive attention，面试很容易被追问穿。

---

## 6. “写到 cuBLAS 90%”是什么水平？

群聊里有人说：

```text
你要对于 cuBLAS 库的性能至少写到 90 吧；
CUDA 这个程度算很厉害了，校招来看。
```

这句话要正确理解。

对校招来说，如果自己写的 GEMM 能接近 cuBLAS 90%，已经是很强的算子能力。大多数普通 AI Infra 推理岗不一定要求达到这个水平。

当前更现实的目标是：

```text
能写出 naive GEMM；
能做 shared memory tiling；
能用 Nsight Compute 看指标；
能解释优化方向；
能说出和 cuBLAS 差距在哪里。
```

如果未来主攻算子岗，再追求 cuBLAS 级别性能。

---

## 7. Triton 是更容易上手的面试工具

群聊里多次提到 Triton：

```text
Triton 写起来跟 Torch 差不多；
一个周期就会写常见算子；
Triton 比较简单，找日常没啥问题；
框架岗手撕可以准备 Triton；
Triton 和 Torch 差不多，找日常没啥问题。
```

这说明 Triton 对初学者比较友好。

推荐策略：

```text
CUDA：理解底层原理和面试八股；
Triton：快速写出可运行的算子项目；
CUTLASS/CuTe：进阶了解，不要一开始深陷。
```

对你来说，Triton 可能比 CUDA 更适合快速做出简历项目。例如：

```text
用 Triton 实现 RMSNorm / Softmax / Attention 小算子；
和 PyTorch 原生实现对比；
用 benchmark 测速度；
写技术报告。
```

---

## 8. CUTLASS / CuTe / CuTeDSL 是进阶内容

群聊中有人问：

```text
CUTLASS 和 CuTe 校招会吗？
```

回答大意是：

```text
CuTe 有点难；
太抽象；
模板元编程；
CuTeDSL 还好一点。
```

所以：

```text
CUTLASS / CuTe 是算子岗进阶内容，不适合当前第一阶段死磕。
```

推荐顺序：

```text
CUDA 基础
    ↓
GEMM / Softmax / Attention
    ↓
Triton
    ↓
FlashAttention 原理
    ↓
再浅看 CUTLASS / CuTe
```

不要一上来就被 CuTe 的模板元编程劝退。

---

## 9. TileLang 是值得关注的新方向

群聊中提到：

```text
性能不够可以看看 TileLang；
国内很多大厂在推；
DS 用的也是 TileLang。
```

这属于新工具趋势信息。当前不用马上学，但可以记下来。

优先级建议：

```text
CUDA 基础 > Triton > vLLM/nano-vLLM > Nsight/profile > TileLang/CUTLASS/CuTe
```

TileLang 可以作为以后做算子项目时的拓展点。

---

# 四、模型压缩、量化、剪枝相关结论

## 1. 模型压缩主要还是量化

群聊中有人问：

```text
你们平时模型压缩主要做量化吗？
会做剪枝吗？
我看模型剪枝也有很多内容，不知道要不要学。
```

回答中比较明确的观点是：

```text
剪枝感觉没什么用；
剪枝很多论文不考虑内存读取；
没有实际加速效果；
剪枝要用结构化剪枝；
那些不拿延迟说事、只吹效果的论文不用看；
现在全连接层剪了有什么用，除非整层剪了。
```

这对你很有指导意义。

如果要学模型压缩，不建议优先学剪枝。更建议学：

```text
量化
敏感层分析
INT8 / INT4
AWQ / GPTQ
FP8 / FP4 趋势
量化后精度和性能对比
量化在 vLLM/SGLang 中的接入
```

---

## 2. 剪枝要看实际延迟，不要只看论文指标

群聊观点：

```text
剪枝论文如果不考虑内存读取，不考虑实际加速效果，就不值得看；
不拿延迟说事的不用看。
```

AI Infra 和算法论文的区别在于：

算法论文可能关注：

```text
参数少了多少；
FLOPs 少了多少；
准确率掉了多少。
```

AI Infra 更关注：

```text
真实延迟是否降低；
吞吐是否提升；
显存是否减少；
kernel 是否更快；
硬件是否真的能利用稀疏性；
端到端是否加速。
```

看论文时要重点看：

```text
有没有真实硬件测试；
有没有 latency；
有没有 throughput；
有没有 memory bandwidth 分析；
有没有 end-to-end speedup；
是不是只在理论 FLOPs 上变小。
```

---

## 3. FP4 是值得关注的趋势，但不能只追热点

群聊中有人问：

```text
现在业界 FP4 已经很流行了吗？
我问面试官，他们说现在都在做这方面相关的。
```

这可以作为一个信号：低精度量化是面试和工业界会关注的方向。

但当前不建议直接从 FP4 开始。合理顺序是：

```text
FP32 / FP16 / BF16
    ↓
INT8
    ↓
INT4
    ↓
AWQ / GPTQ
    ↓
FP8
    ↓
FP4
```

当前重点应放在：

```text
为什么量化能省显存；
为什么量化可能加速；
为什么量化会掉精度；
什么是敏感层；
为什么有些层不能量化太狠；
如何做量化前后 benchmark。
```

---

# 五、算法和 AI Infra 的关系

群聊中有一组讨论非常关键：

```text
Infra 也要懂部分算法；
Infra 属于主流算法要了解，但更重工程落地；
会 Transformer 和 MoE 够用了。
```

这对你很准确。

你不需要像算法岗一样深入研究：

```text
强化学习理论；
复杂训练算法；
大规模预训练 recipe；
数学推导；
论文创新。
```

但必须懂：

```text
Transformer 架构；
Attention；
KV Cache；
MoE；
MLP；
RMSNorm / LayerNorm；
Tokenizer；
量化；
推理阶段 prefill/decode；
主流模型结构变化。
```

原因是这些直接决定：

```text
算子怎么写；
显存怎么算；
KV Cache 怎么管理；
MoE 怎么并行；
Attention 怎么优化；
量化怎么做；
框架为什么这么设计。
```

所以你的学习策略应该是：

```text
算法不作为主线；
但 Transformer / Attention / MoE 是必须补的底层认知。
```

---

# 六、ACM / LeetCode / 408 的求职价值

群聊中有人说：

```text
ACM 没啥用；
我是 ACM 银牌，还是挂挂挂；
银牌遍地走，一场比赛可能就是很多块银牌，一点也不值钱。
```

这不是说算法题完全没用，而是说：

```text
AI Infra 面试不是只靠 ACM / LeetCode 就能过。
```

AI Infra 更重：

```text
项目；
系统理解；
CUDA/Triton；
推理框架；
性能分析；
模型部署；
工程落地。
```

408 / OS / C++ / Linux 对你有用，但不能只停留在基础课。

适合你的组合是：

```text
OS / Linux / C++
    +
PyTorch / Transformer
    +
nano-vLLM / vLLM
    +
CUDA/Triton 基础
    +
benchmark/profile 项目
```

这比单纯刷算法或单纯背 408 更适合 AI Infra。

---

# 七、端侧、云侧、车企和硬件适配

群聊中有人讨论端侧和云侧：

```text
你做哪些卡？
端侧还是云侧的卡？
我在做端侧机器人上的卡；
卡不熟悉，还要编译才能跑起来；
端侧是要交叉编译；
直接在端侧编译太慢了。
```

这对有嵌入式背景的人很有价值。

---

## 1. 云侧 AI Infra

云侧更偏：

```text
A100/H100/国产 GPU；
vLLM/SGLang/TensorRT-LLM；
多卡推理；
KV Cache；
高并发服务；
大模型吞吐；
PD 分离；
量化；
分布式通信。
```

---

## 2. 端侧 AI Infra

端侧更偏：

```text
机器人；
车企；
自动驾驶；
边缘设备；
NPU；
交叉编译；
模型转换；
ONNX / TensorRT / NCNN / MNN / RKNN；
端侧芯片适配；
功耗和延迟。
```

群聊中还有人说：

```text
你要去广州那些端侧车企，多写写 CUDA 就好了。
```

这说明端侧车企可能更看：

```text
CUDA / TensorRT / 模型部署 / 端侧优化 / 性能分析
```

而不一定一开始要求深挖 vLLM。

所以未来可以准备两个求职分支：

```text
分支 A：云侧大模型推理
nano-vLLM / vLLM / KV Cache / CUDA / benchmark

分支 B：端侧边缘 AI
TensorRT / ONNX / CUDA / 交叉编译 / 模型部署 / 嵌入式系统
```

结合你的背景，两个都可以准备，但主线建议仍然放在：

```text
推理服务 + 模型部署 + CUDA/Triton 基础
```

---

# 八、学习资源和路线建议

## 1. mini-vLLM / nano-vLLM

群友建议：

```text
mini-vLLM 看 GitHub；
看完后再看 vLLM；
在 mini 基础上实现点自己的东西；
框架是这样的。
```

推荐执行方式：

```text
第一步：跑通 nano-vLLM / mini-vLLM；
第二步：画出请求流转图；
第三步：理解 tokenizer、scheduler、KV cache、model runner；
第四步：加日志统计 prefill/decode 时间；
第五步：实现一个小优化；
第六步：写 benchmark 报告；
第七步：再看 vLLM 真实源码。
```

---

## 2. 知乎李少侠 CUDA 入门帖

群聊中有人建议：

```text
可以看知乎上李少侠的帖子；
那篇文章搞明白，CUDA 就算入门了；
然后写一写 FAv2、MoE；
看看 CuTe；
接着转框架。
```

这是一条“算子 → 框架”的路线。

你可以吸收，但不必完全照搬。更适合你的版本是：

```text
CUDA 入门文章
    ↓
GEMM / Softmax / Attention
    ↓
Triton 写几个算子
    ↓
nano-vLLM 框架
    ↓
vLLM 关键模块
```

---

## 3. 热门项目容易同质化，要加入自己的分析

群聊中有人说：

```text
很多人推荐 ffz；
感觉这个群里人手一个他项目；
面试官里都有 ffz 的兵。
```

这句话的意思是：热门项目如果大家都做，就容易同质化。

所以做开源项目时要注意：

```text
不要只复现热门项目；
要在项目上加自己的实验、修改、分析和报告。
```

比如大家都做 nano-vLLM，你可以加入：

```text
1. prefill/decode 性能日志；
2. KV Cache 显存估算；
3. 不同 batch/concurrency benchmark；
4. 一个论文优化点；
5. 一个简化版 speculative decoding；
6. 一个 prefix cache 复现实验；
7. 一个可视化性能报告。
```

这样项目就和别人不一样。

---

## 4. WSL 是 Windows 下学 CUDA 的推荐环境

群聊中有人问 Windows 上 CUDA 怎么跑，回答包括：

```text
用 WSL；
Windows 也能装 CUDA；
WSL 装 CUDA 方便一点。
```

如果现在是 Windows 笔记本，可以考虑：

```text
Windows + WSL2 + Ubuntu + CUDA Toolkit
```

但如果后续要跑 vLLM，最好还是用：

```text
Linux 服务器 / AutoDL / 实验室 GPU / 云服务器
```

因为 vLLM、CUDA、NVIDIA 驱动、PyTorch 版本匹配在原生 Linux 下更顺。

---

# 九、群聊隐含的项目优先级

## 第一优先级：nano-vLLM / mini-vLLM 魔改项目

这是最适合当前阶段的 AI Infra 项目。

项目目标：

```text
理解大模型推理框架的最小闭环，并加入自己的优化或分析。
```

可以做的模块：

```text
1. 请求调度；
2. Prefill / Decode 分离统计；
3. KV Cache 分配和释放；
4. 简化版 PagedAttention 思路；
5. Continuous Batching 模拟；
6. Prefix Cache；
7. Benchmark 脚本；
8. 性能可视化；
9. 论文优化复现。
```

简历表达可以写：

```text
基于 nano-vLLM 实现轻量级 LLM 推理框架分析与优化，梳理请求从 API 到 Scheduler、KV Cache、ModelRunner 的执行链路；实现 prefill/decode 阶段耗时统计与 KV Cache 显存占用分析，并对不同 batch size、上下文长度和并发数下的 TTFT、TPOT、吞吐进行 benchmark。
```

适合投：

```text
AI Infra 实习
推理框架实习
模型部署实习
大模型推理工程实习
```

---

## 第二优先级：CUDA/Triton 算子项目

不建议现在做特别深的纯 CUDA 项目，但可以做一个“面试可讲”的基础算子项目。

建议内容：

```text
1. CUDA 实现 naive GEMM；
2. shared memory tiling；
3. 用 Nsight Compute profile；
4. 分析 memory throughput / occupancy / bank conflict；
5. Triton 实现 softmax / RMSNorm / attention；
6. 对比 PyTorch / cuBLAS / Triton 性能。
```

不要一上来追求极致性能。当前重点是：

```text
能讲清楚优化过程。
```

---

## 第三优先级：量化 + 敏感层分析项目

群聊中有人做过：

```text
SGLang 平台上的混合模型量化；
敏感层分析。
```

这类项目有价值，但容易变成调库。要做出亮点，需要加入：

```text
1. 不同层量化对精度影响；
2. 不同量化策略对显存影响；
3. 不同量化策略对吞吐影响；
4. 为什么某些层敏感；
5. 是否有实际推理速度提升；
6. 在 vLLM/SGLang 中如何接入。
```

否则面试官可能会认为只是“用了现成量化工具”。

---

## 第四优先级：IMS + 边缘 AI 项目

结合你的背景，可以做一个差异化项目：

```text
IMS 谱图识别 + ONNX/TensorRT/端侧推理部署
```

这不是群聊里直接提到的，但非常适合你。因为你的背景不是纯 CS，你需要利用嵌入式、仪器、信号采集经验形成差异化。

---

# 十、对你个人路线的启发

结合你目前的情况，不要被群聊里“CUDA、MoE、FAv2、CuTe、TileLang”这些词吓到。你现在最重要的是建立一个求职闭环。

---

## 1. 当前不应该做的事

```text
1. 一上来死磕 CuTe；
2. 一上来追求 GEMM 写到 cuBLAS 90%；
3. 一上来啃完整 vLLM 源码；
4. 只看论文不写代码；
5. 只部署模型不做 benchmark；
6. 做和别人一样的热门项目，没有自己的分析；
7. 只学 RAG/Agent，把自己做成应用开发候选人。
```

---

## 2. 当前应该做的事

```text
1. 先补 PyTorch / Transformer / KV Cache；
2. 跑通 nano-vLLM；
3. 理解 prefill、decode、scheduler、KV cache；
4. 做 benchmark；
5. 写一篇项目报告；
6. 补 CUDA GEMM 基础；
7. 用 Triton 写几个简单算子；
8. 找一个论文优化点加入 nano-vLLM；
9. 尽快形成可投实习的项目。
```

---

# 十一、建议最终形成的两个核心项目

## 项目一：基于 nano-vLLM 的大模型推理框架学习与优化

核心亮点：

```text
不是简单部署，而是理解和修改框架。
```

项目内容：

```text
1. 跑通 nano-vLLM；
2. 梳理请求生命周期；
3. 加入 prefill/decode profiling；
4. 分析 KV Cache 显存占用；
5. 对比不同 batch size / context length / output length；
6. 实现一个小型优化，例如 prefix cache / chunked prefill 模拟 / 简化版 continuous batching；
7. 输出实验报告。
```

适合投：

```text
AI Infra 实习
推理框架实习
模型部署实习
大模型推理工程实习
```

---

## 项目二：CUDA/Triton LLM 核心算子实现与性能分析

核心亮点：

```text
不是只会调框架，也懂底层算子和 profile。
```

项目内容：

```text
1. CUDA 实现 GEMM；
2. shared memory tiling；
3. Nsight Compute profile；
4. Triton 实现 RMSNorm / Softmax / Attention；
5. 对比 PyTorch / cuBLAS 性能；
6. 分析 memory-bound / compute-bound；
7. 写优化报告。
```

适合投：

```text
推理优化
异构计算
模型部署
高性能计算
端侧推理
```

---

# 十二、面试准备清单

## 1. 项目面试准备

你要能回答：

```text
你的项目亮点是什么？
为什么不是调库？
你改了哪些源码？
为什么这么改？
有没有复现论文？
有没有性能数据？
profile 怎么做？
瓶颈在哪里？
优化前后差多少？
还有哪些不足？
```

---

## 2. CUDA 面试准备

你要能回答：

```text
GEMM 怎么写？
naive GEMM 有什么问题？
shared memory tiling 怎么做？
什么是 bank conflict？
什么是 coalesced memory access？
什么是 warp？
什么是 Tensor Core？
怎么用 Nsight Compute？
怎么判断 memory-bound 还是 compute-bound？
```

---

## 3. 推理框架面试准备

你要能回答：

```text
vLLM 为什么快？
KV Cache 是什么？
PagedAttention 解决什么问题？
Prefill 和 Decode 有什么区别？
Continuous Batching 为什么提高吞吐？
Chunked Prefill 解决什么问题？
Prefix Cache 有什么用？
SGLang 和 vLLM 有什么区别？
为什么新模型新卡需要适配？
```

---

## 4. 量化面试准备

你要能回答：

```text
量化为什么能省显存？
INT8 / INT4 有什么区别？
什么是敏感层？
为什么有些层不能量化？
量化后怎么评估精度？
量化是否一定加速？
剪枝为什么很多时候没有实际加速？
```

---

# 十三、最终行动路线

结合这批聊天，建议你接下来按这个顺序走：

```text
PyTorch / Transformer / KV Cache 基础
    ↓
mini-vLLM / nano-vLLM 跑通
    ↓
理解 prefill / decode / scheduler / KV Cache
    ↓
加 profiling 和 benchmark
    ↓
复现一个论文优化点，加入 nano-vLLM
    ↓
CUDA GEMM 入门
    ↓
Triton 写 Attention / RMSNorm / Softmax
    ↓
整理两个项目报告
    ↓
开始投日常实习
```

---

# 十四、最终判断

这批群聊真正有价值的地方，不是给出了一个“唯一正确路线”，而是暴露了 AI Infra 求职的真实评价标准：

```text
1. 只会调库不够；
2. 项目必须能被深挖；
3. CUDA/Triton 仍然重要；
4. GEMM、Attention、MoE 是高频核心；
5. nano-vLLM / mini-vLLM 更适合入门做项目；
6. 论文优化 + benchmark 能显著提升项目含金量；
7. 量化比剪枝更值得优先学习；
8. 端侧和云侧路线不同，但都需要性能意识；
9. 热门项目要做出自己的差异化；
10. AI Infra 更重工程落地，不是单纯算法或单纯应用。
```

一句话总结：

**你的目标不是“学会很多名词”，而是做出两个能被面试官深挖、能讲清楚性能指标、能体现工程修改的 AI Infra 项目。**


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-AI infra学习路线|模块-AI infra学习路线]]

%% 项目关联导航：结束 %%
