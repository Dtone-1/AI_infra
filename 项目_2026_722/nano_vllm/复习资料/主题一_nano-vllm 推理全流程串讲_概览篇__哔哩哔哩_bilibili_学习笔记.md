# 课程学习笔记：主题一_nano-vllm 推理全流程串讲_概览篇__哔哩哔哩_bilibili.txt

> 本笔记基于视频语音转文字文件整理。原文存在自动识别错误、重复口语、断句混乱等问题，本文已按 AI Infra / 大模型推理语境进行术语纠正和结构化重组。  
> 由于原文件没有清晰时间戳，下面按照课程内容自然推进顺序分段：背景引入 → vLLM 要解决的问题 → PagedAttention / Continuous Batching → nano-vLLM 架构 → 请求生命周期 → Prefill / Decode 差异。

---

## 1. 视频宗旨与课程定位

### 1.1 这节课主要想讲什么

这节课是一次 **nano-vLLM / vLLM 推理全流程概览课**。老师的核心目标不是直接带着读某一行代码，而是先从系统设计角度回答一个大问题：

> 为什么大模型推理不能只靠 PyTorch 或 Hugging Face Transformers 的 `generate()` 循环完成，而需要 vLLM / nano-vLLM 这样的推理引擎？

围绕这个问题，课程主要讲了四件事：

1. **传统推理方式的瓶颈**  
   使用 PyTorch 或 Transformers 直接推理虽然能跑通模型，但在服务化场景下容易出现显存占用高、GPU 利用率低、吞吐量上不去的问题。

2. **vLLM 的核心设计思想**  
   vLLM 主要通过 **PagedAttention** 和 **Continuous Batching** 解决显存管理和批处理效率问题。

3. **nano-vLLM 的整体架构**  
   nano-vLLM 是一个轻量级 vLLM 实现，代码抽象与 vLLM 0.9.0 版本高度相似，适合初学者从源码层面理解推理引擎。

4. **一个请求从 prompt 到输出 token 的完整生命周期**  
   用户输入 prompt 后，会经过 tokenization、request / sequence 构造、scheduler 调度、block manager 分配 KV Cache block、prefill、decode、停止条件判断、资源释放等步骤。

### 1.2 它在 AI Infra / 大模型推理学习路线中的位置

这节课处在 AI Infra 推理方向学习路线中的 **“推理引擎系统认知阶段”**。

如果把 AI Infra 推理方向拆成几个层次，可以这样理解：

| 层次 | 主要学习内容 | 本节课的位置 |
|---|---|---|
| 模型层 | Transformer、Attention、模型权重、前向计算 | 本节课只作为背景涉及 |
| 推理流程层 | Tokenization、Prefill、Decode、KV Cache | 本节课重点讲 |
| 系统调度层 | Request、Sequence、Scheduler、Batch、Continuous Batching | 本节课重点讲 |
| 显存管理层 | KV Cache、Block、Block Manager、PagedAttention | 本节课重点讲 |
| GPU 执行层 | CUDA kernel、FlashAttention、GPU/HBM、compute bound / memory bound | 本节课初步引入 |
| 服务部署层 | OpenAI-compatible API、Benchmark、QPS、TTFT、TPOT | 本节课没有展开，但为后续学习铺垫 |

因此，这节课不是单纯讲 Transformer，也不是单纯讲 CUDA，而是在讲 **“模型推理如何变成一个高吞吐、低延迟、可服务化的系统”**。

### 1.3 它和 vLLM、nano-vLLM、Transformer 推理、KV Cache、调度、显存优化的关系

本节课的技术主线可以概括为：

> Transformer 自回归推理会产生大量 KV Cache；KV Cache 占用显存且长度动态变化；多用户请求长度不同、到达时间不同；因此推理系统需要调度器和显存管理器来高效组织请求，让 GPU 尽可能持续工作。vLLM / nano-vLLM 就是在解决这些问题。

具体关系如下：

- **Transformer 推理** 是底层计算对象。模型每次前向计算都会生成下一 token 的概率分布。
- **Prefill / Decode** 是 Transformer 推理在生成式服务中的两个阶段。
- **KV Cache** 是减少重复计算的关键，但也带来显存管理问题。
- **PagedAttention** 用类似操作系统虚拟内存分页的方式管理 KV Cache，降低显存碎片和预分配浪费。
- **Scheduler** 决定哪些 request / sequence 在当前 step 被执行。
- **Continuous Batching** 让 batch 可以动态加入新请求，避免静态 batch 中短序列提前结束后 GPU 出现空泡。
- **nano-vLLM** 用较少代码复现 vLLM 的核心抽象，适合做源码导读和项目入门。
- **vLLM Benchmark / 推理服务部署** 会用到这节课提到的吞吐、延迟、显存、batch、prefill/decode 等概念。

### 1.4 这节课适合解决学习者的什么问题

这节课适合解决以下疑问：

1. 为什么大模型推理服务不是简单调用 `model.generate()`？
2. 为什么同样的模型，推理框架不同，吞吐量和显存占用差距很大？
3. vLLM 的核心价值到底是什么？
4. PagedAttention 和操作系统虚拟内存有什么相似之处？
5. Continuous Batching 为什么能提高 GPU 利用率？
6. 一个请求进入推理系统后，会经过哪些模块？
7. Prefill 和 Decode 为什么性能特征不同？
8. nano-vLLM 源码应该优先看哪些模块？

### 1.5 初学者应该带着什么问题学习

初学者学习这节课时，建议带着五个问题：

1. **服务化推理的目标是什么？**  
   不是单次生成结果，而是在多用户并发下尽可能提高吞吐、降低延迟、控制显存。

2. **为什么 KV Cache 是推理系统的核心资源？**  
   因为大模型自回归生成时，历史 token 的 K/V 会被后续 token 重复使用，缓存它们可以节省大量重复计算。

3. **为什么显存管理比普通 PyTorch 推理复杂？**  
   因为每个请求的输出长度未知，KV Cache 会动态增长，直接预分配会浪费，动态申请又会碎片化。

4. **Scheduler 到底在调度什么？**  
   它不是调度线程，而是在调度 request / sequence，决定当前 GPU step 该执行哪些 token。

5. **nano-vLLM 源码应该从哪里看？**  
   先看 LLMEngine / generate 入口，再看 Scheduler、BlockManager、Worker/ModelRunner、Model forward、KV Cache 数据结构。

---

## 2. 按时间进度梳理知识点

原始转文字文件没有稳定时间戳，因此按课程推进顺序整理为五个部分。

---

### 第 1 部分：课程背景与学习方法 —— 不要一上来陷入 low-level 代码

#### 1. 核心知识点

课程开头强调，学习 LLM serving / 推理引擎时，不能只从零散代码细节入手，而应当先从框架整体设计理念出发，采用 **自上而下** 的学习方式。

老师提到，这一系列可能会分多个主题讲，本节先讲推理全流程，目标包括：

- 搞懂 LLM serving 的服务设计理念；
- 理清一个 prompt 进入推理系统后经历的完整生命周期；
- 大致了解 vLLM / nano-vLLM 的核心架构；
- 以 nano-vLLM 为主线，因为它是 vLLM 的轻量级实现，抽象与 vLLM 0.9.0 版本相似。

#### 2. 老师想表达什么

老师想表达的是：**推理引擎不是单个函数或单个 CUDA kernel，而是一个系统。**

如果只看某个函数，例如 attention forward、block allocation、scheduler step，很容易迷失在细节中。正确学习路径应该是：

1. 先理解为什么需要推理引擎；
2. 再理解推理系统由哪些模块组成；
3. 再看请求在模块之间如何流转；
4. 最后再进入具体源码和 CUDA kernel。

#### 3. 相关概念解释

- **LLM Serving**  
  指将大语言模型部署成在线服务，对外接收用户请求，返回生成结果。它关注的不只是模型能不能跑，还关注并发、吞吐、延迟、显存、稳定性和接口形式。

- **推理引擎**  
  指专门负责大模型高效推理的系统组件，例如 vLLM、TensorRT-LLM、SGLang、TGI 等。推理引擎通常包含模型加载、调度、KV Cache 管理、GPU 执行、输出采样、服务接口等模块。

- **nano-vLLM**  
  轻量版 vLLM，用更少代码复现 vLLM 的关键业务逻辑，适合入门理解 request、sequence、scheduler、block manager、KV Cache 等抽象。

#### 4. 在 AI Infra 中的作用

AI Infra 推理方向的核心能力不是“会调一个模型 API”，而是理解：

- 请求如何被组织；
- 显存如何被管理；
- GPU 如何被喂满；
- batch 如何动态变化；
- 推理服务如何在吞吐和延迟之间取舍。

这节课正是在建立这些系统级认知。

#### 5. 初学者容易误解的地方

- 误解 1：认为会 PyTorch 就等于会推理引擎。  
  实际上 PyTorch 是通用深度学习框架，推理引擎则面向高并发服务化场景，两者关注点不同。

- 误解 2：认为 vLLM 只是把 Transformer forward 写得更快。  
  实际上 vLLM 的核心价值不只是算子优化，更重要的是 **KV Cache 显存管理 + 动态请求调度**。

- 误解 3：认为源码阅读应该从 CUDA kernel 开始。  
  初学者更应该先看系统主流程，否则很容易看懂局部代码却不知道它在整个系统中起什么作用。

#### 6. 和实际项目的联系

- 做 **nano-vLLM 源码阅读** 时，应先画出 LLMEngine、Scheduler、BlockManager、Worker、Model 的调用关系。
- 做 **vLLM Benchmark** 时，要理解吞吐、延迟、batch、prefill/decode 为什么会影响性能。
- 做 **推理服务部署** 时，要知道 OpenAI-compatible API 背后不是简单调用模型，而是完整推理引擎在工作。

---

### 第 2 部分：为什么不能只用 PyTorch / Transformers generate —— 传统推理方式的三个瓶颈

#### 1. 核心知识点

老师提出一个关键问题：

> 为什么不直接用 PyTorch 或 Hugging Face Transformers 的 `generate()` 循环完成推理？

答案是：直接推理可以完成单个请求，但在服务化、多并发、高吞吐场景下存在明显瓶颈。课程中总结了三个主要问题：

1. **显存碎片化**
2. **静态批处理低效**
3. **预分配内存浪费**

#### 2. 老师想表达什么

老师想说明：传统推理方式不是不能用，而是不适合做高性能 LLM serving。  
当请求数量变多、prompt 长度和输出长度差异变大时，简单 generate 循环会出现：

- GPU 显存占用高；
- 实际 GPU 利用率低；
- batch 中存在大量空泡；
- KV Cache 动态增长导致显存管理困难；
- 吞吐量难以提升。

#### 3. 相关概念详细解释

##### 3.1 显存碎片化

GPU 显存不像操作系统虚拟内存那样天然有完善的页表抽象。CUDA / GPU 内存分配通常更接近直接申请一段连续物理显存。

如果显存中有两块各 500 MB 的空闲空间，但它们不连续，那么申请一个 800 MB 连续 Tensor 可能失败。即使总空闲显存足够，也可能因为不连续而无法分配。

这就是显存碎片化问题。

在 LLM 推理中，KV Cache 会随着序列长度增长而动态变化，请求不断进入和结束，显存分配和释放非常频繁，因此碎片化问题会更加严重。

##### 3.2 静态批处理低效

传统 batch 通常是静态的：一批请求一起进入，一起执行。

问题在于，不同请求长度不同：

- 有的请求 prompt 很短，输出也短；
- 有的请求 prompt 很长，输出也长；
- 有的请求很快生成 EOS；
- 有的请求要生成到 max tokens。

在静态 batch 中，短请求结束后，batch 里对应的位置就空了，但长请求还在继续跑。这样 GPU 在后续 step 中会出现“空泡”，导致硬件利用率下降。

##### 3.3 预分配内存浪费

推理系统在请求进入时，通常不知道这个请求最终会生成多少 token。

如果采用动态扩容：

- 不够时再申请新显存；
- 可能导致频繁分配、拷贝、碎片化；
- 系统开销高。

如果采用最大长度预分配：

- 假设系统最大支持 65536 token；
- 每个请求都按最大长度申请 KV Cache；
- 大量请求实际用不到这么长；
- 显存严重浪费。

因此，推理系统需要一种更细粒度、更灵活的 KV Cache 管理方式。

#### 4. 在 AI Infra / 推理系统中的作用

这三个问题本质上对应推理系统最核心的两个目标：

| 问题 | 对应系统目标 | vLLM 的解决思路 |
|---|---|---|
| 显存碎片化 | 提高显存可用性 | PagedAttention / block-based KV Cache |
| 预分配浪费 | 提高显存利用率 | 按 block 动态分配 |
| 静态 batch 低效 | 提高 GPU 利用率和吞吐 | Continuous Batching |

#### 5. 初学者容易误解的地方

- 误解 1：显存够大就不会有问题。  
  实际上显存不仅要看总量，还要看是否能连续分配、是否被碎片化、KV Cache 管理是否高效。

- 误解 2：batch 越大越好。  
  batch 大可以提高吞吐，但也会增加等待时间、显存压力和调度复杂度。推理系统要在吞吐和延迟之间平衡。

- 误解 3：预分配最大长度最简单，所以最好。  
  它确实实现简单，但在高并发服务中会极大浪费显存，降低可服务请求数。

#### 6. 和实际项目的联系

- 在 **nano-vLLM** 中，应重点关注 block、block manager、KV Cache 的分配逻辑。
- 在 **vLLM Benchmark** 中，如果并发数升高但吞吐上不去，可能与 batch 调度、KV Cache 显存占用有关。
- 在 **推理服务部署** 中，max model len、max num seqs、gpu memory utilization 等配置都会直接影响显存使用和可并发请求数量。

---

### 第 3 部分：vLLM 的核心思想 —— PagedAttention 与 Continuous Batching

#### 1. 核心知识点

课程指出，vLLM 主要解决上述三个问题：

1. 用 **PagedAttention** 解决显存碎片化和预分配浪费；
2. 用 **Continuous Batching** 解决静态 batch 低效；
3. 用调度器和 block manager 把请求生命周期、KV Cache、GPU 执行串起来。

#### 2. 老师想表达什么

老师想表达的是：vLLM 的关键不是“又写了一个模型 forward”，而是借鉴操作系统思想，把 KV Cache 管理成一个个 block，使显存使用更细粒度、更灵活。

同时，vLLM 不再要求一批请求同时开始、同时结束，而是允许请求动态进入和退出 batch，从而提高 GPU 持续利用率。

#### 3. PagedAttention 详细解释

##### 3.1 是什么

**PagedAttention** 可以理解为把 KV Cache 按固定大小的 block 管理，而不是为每个 sequence 预分配一整段连续显存。

它的思想类似操作系统虚拟内存：

- 操作系统把虚拟地址映射到物理页；
- vLLM 把 sequence 中的 token 位置映射到 KV Cache block；
- sequence 逻辑上是连续的，但物理显存中的 block 不必连续。

##### 3.2 为什么重要

大模型推理中，KV Cache 是显存大户。每一层、每个 attention head、每个历史 token 都会保存 K/V。随着并发请求增加，KV Cache 会迅速消耗显存。

如果每个请求都预分配最大长度，显存会浪费；如果不断动态申请连续内存，会碎片化。PagedAttention 用 block 作为基本分配单位，使系统可以按需追加 block。

##### 3.3 在系统中的作用

PagedAttention 让推理系统可以：

- 以 block 为单位分配和释放 KV Cache；
- 避免为短请求浪费大量 max length 空间；
- 降低连续显存申请压力；
- 支持更高并发；
- 让 BlockManager 能统一管理请求对应的物理 block。

##### 3.4 和 nano-vLLM 的关系

nano-vLLM 中会出现 `Block`、`BlockManager`、`Sequence`、`KV Cache` 等抽象。理解 PagedAttention 后，读这些类就会清楚：

- `Sequence` 是逻辑序列；
- `Block` 是 KV Cache 的物理管理单位；
- `BlockManager` 负责把逻辑 token 位置映射到实际 block；
- attention forward 时需要根据 block table 找到对应历史 K/V。

#### 4. Continuous Batching 详细解释

##### 4.1 是什么

**Continuous Batching** 指推理系统在每个 step 动态维护 batch：

- 已完成的 sequence 可以退出；
- 新到来的 request 可以加入；
- batch 不再是一次性固定不变；
- GPU 尽量一直有活干。

##### 4.2 为什么重要

自回归生成每次 decode 只生成一个 token。不同请求完成时间不同，如果使用静态 batch，短请求结束后 batch 内会留下空位。Continuous Batching 可以把新请求插入这些空位，从而提高 GPU 利用率。

##### 4.3 在系统中的作用

Continuous Batching 依赖 Scheduler：

- Scheduler 维护 waiting queue 和 running sequence；
- 每个 step 检查哪些 sequence 可以继续执行；
- 检查显存容量是否允许加入新请求；
- 结合 BlockManager 分配 KV Cache block；
- 组成当前 step 的实际 batch 送到 Worker / ModelRunner 执行。

#### 5. 初学者容易误解的地方

- 误解 1：PagedAttention 是一种新的 Attention 数学公式。  
  它不是改变 attention 数学含义，而是改变 KV Cache 的显存组织方式和读取方式。

- 误解 2：Continuous Batching 就是普通 batch size 调大。  
  不是。普通 batch 是静态集合，Continuous Batching 是动态集合。

- 误解 3：block 越小越好。  
  block 太小会降低浪费，但管理开销和索引开销增加；block 太大管理简单，但内部浪费增加。实际系统需要折中。

#### 6. 和实际项目的联系

- **nano-vLLM 源码阅读**：重点看 BlockManager 如何分配 block、释放 block、维护 block table。
- **vLLM Benchmark**：Continuous Batching 会影响吞吐和延迟。高并发场景下，它通常比静态 batch 更能提高吞吐。
- **推理服务部署**：并发请求越多、长度差异越大，PagedAttention 和 Continuous Batching 的收益越明显。

---

### 第 4 部分：nano-vLLM 整体架构 —— LLMEngine、Scheduler、BlockManager、Worker、Model

#### 1. 核心知识点

课程介绍了 nano-vLLM 的主要系统抽象。虽然转文字中很多词识别错误，但根据语境可整理为以下模块：

| 转写中的可能错误 | 正确术语 | 作用 |
|---|---|---|
| RM serving / WRM / VLM | LLM serving / vLLM | 大模型推理服务 / 推理框架 |
| NanoWRM / NalviaM / downloadvrm | nano-vLLM | 轻量级 vLLM 实现 |
| rm 英诊 | LLMEngine | 对外暴露 generate 接口的总控模块 |
| sched up / specialty | Scheduler | 请求调度器 |
| block manager | BlockManager | KV Cache block 分配与管理 |
| walker / model walker | Worker / ModelRunner | GPU 执行进程，负责模型前向 |
| cubic catch | KV Cache | 历史 token 的 K/V 缓存 |
| prove / prop / ProP | prompt | 用户输入文本 |
| opput | output | 模型输出 |
| HCCR / SSR | 可能是 HCCL / NCCL | 多卡通信库，转写不确定 |

#### 2. 老师想表达什么

老师希望学习者先抓住 nano-vLLM 的五个关键类或模块：

1. **LLMEngine**：对外接口和总控；
2. **Scheduler**：决定哪些请求当前执行；
3. **BlockManager**：管理 KV Cache block；
4. **Worker / ModelRunner**：执行模型 forward；
5. **Model**：真正的 Transformer 模型结构和 forward 逻辑。

#### 3. 架构模块解释

##### 3.1 LLMEngine

LLMEngine 可以理解为推理系统的总控入口。用户侧通常只需要调用类似 `generate()` 的接口，把 prompt 输入进去，然后等待模型持续产生输出 token。

LLMEngine 背后会协调：

- tokenization；
- request 创建；
- scheduler 调度；
- worker 执行；
- sampling；
- 结果返回；
- 资源释放。

##### 3.2 Scheduler

Scheduler 负责决定当前 step 运行哪些请求。它需要考虑：

- waiting queue 中有哪些新请求；
- running queue 中有哪些旧请求还没结束；
- 当前 GPU / KV Cache 容量是否允许加入新请求；
- 哪些 sequence 已经达到 EOS 或 max tokens；
- prefill 请求和 decode 请求如何混合调度。

它是推理系统的“交通调度中心”。

##### 3.3 BlockManager

BlockManager 负责 KV Cache 的显存分配。它要回答：

- 某个 sequence 需要多少 block；
- 还能不能分配新的 block；
- 某个 token 的 K/V 存在哪个 block 的哪个 slot；
- 请求结束后如何释放 block；
- block table 如何传给 attention kernel 使用。

##### 3.4 Worker / ModelRunner

Worker 或 ModelRunner 是执行层。它通常绑定一张 GPU 或一个 GPU 进程，负责：

- 持有模型权重；
- 接收 scheduler 组织好的 batch；
- 准备 input ids、positions、block tables 等张量；
- 调用 model forward；
- 写入或读取 KV Cache；
- 返回 logits；
- 进行采样或把 logits 交给上层采样。

在多卡推理中，每个 worker 可能持有模型的一部分，通过通信库协同完成推理。

##### 3.5 Model

Model 是 Transformer 模型本体，包括 embedding、attention、MLP、norm、lm_head 等结构。课程中提到在 `model.py` 的 forward 中，Prefill 可能使用 FlashAttention forward，Decode 主要使用 KV Cache 方式。

#### 4. 在 AI Infra 中的作用

这些模块构成了推理引擎的最小系统闭环：

```text
用户请求
  ↓
LLMEngine 接收
  ↓
Tokenizer 转 token
  ↓
Request / Sequence
  ↓
Scheduler 调度
  ↓
BlockManager 分配 KV Cache block
  ↓
Worker / ModelRunner 执行模型 forward
  ↓
Sampling 得到新 token
  ↓
更新 Sequence 和 KV Cache
  ↓
返回输出 / 继续 Decode / 释放资源
```

#### 5. 初学者容易误解的地方

- 误解 1：LLMEngine 就是模型。  
  LLMEngine 是系统控制层，不是 Transformer 模型本身。

- 误解 2：Scheduler 只是简单 FIFO 队列。  
  实际 scheduler 要考虑显存容量、prefill/decode 阶段、sequence 状态、batch 组织等。

- 误解 3：BlockManager 只是申请显存。  
  它还负责逻辑 token 到物理 block 的映射，是 PagedAttention 的关键支撑。

- 误解 4：Worker 只是调用 `model.forward()`。  
  在推理引擎里，Worker 还涉及多卡、输入张量准备、KV Cache 管理、通信、采样等问题。

---

### 第 5 部分：一个请求的完整生命周期 —— Tokenization、Sequence、Scheduler、Prefill、Decode

#### 1. 核心知识点

课程后半段重点讲一个请求从 prompt 到 output 的完整过程：

1. 用户输入 prompt；
2. Tokenizer 把字符串转成 token ids；
3. 构造 Request / Sequence；
4. Sequence 进入 waiting queue；
5. Scheduler 检查容量；
6. BlockManager 分配 KV Cache block；
7. 进入 running 状态；
8. 执行 Prefill；
9. 执行 Decode；
10. 达到 EOS 或 max tokens 后结束；
11. 释放资源。

#### 2. Tokenization

##### 是什么

Tokenization 是把文本字符串转换成模型可处理的整数 token ids。

例如：

```text
输入文本：请解释什么是深度学习
Tokenizer 输出：[token_id_1, token_id_2, token_id_3, ...]
```

模型本身不能直接处理字符串，只能处理数字张量。因此 tokenization 是推理流程的第一步。

##### 常见算法

课程中提到的主流 tokenizer 算法包括：

- **BPE（Byte Pair Encoding）**
- **SentencePiece**

实际 vLLM / nano-vLLM 通常不会自己重新实现 tokenizer，而是复用 Hugging Face tokenizers。该库底层常用 Rust 实现，性能较高。

##### 在系统中的作用

Tokenizer 决定：

- prompt 会变成多少 token；
- 输入长度是多少；
- KV Cache 初始需要多少空间；
- max token、上下文长度、截断策略如何生效；
- 输出 token 如何 detokenize 回文本。

#### 3. Request 与 Sequence

##### Request

Request 更偏用户请求维度，包含：

- prompt；
- sampling parameters；
- request id；
- max tokens；
- stop condition；
- 当前生成状态。

##### Sequence

Sequence 更偏模型执行维度，表示一个正在推理的 token 序列。推理引擎内部通常以 sequence 为调度单位。

一个 request 可能对应一个或多个 sequence，例如 beam search 或多样本采样时，一个 request 可能扩展成多个 sequence。

本节课主要按单 request 对应单 sequence 的简单情形理解即可。

#### 4. Scheduler 调度

Sequence 进入调度器后，一开始通常处于 waiting queue。Scheduler 每一轮会检查：

- 是否有空闲 KV Cache block；
- 当前 batch 是否还能加入新 sequence；
- 哪些 running sequence 需要继续 decode；
- 哪些 sequence 已经结束；
- 哪些 waiting sequence 可以进入 running。

调度器决定当前 step 的执行集合。

#### 5. BlockManager 分配 KV Cache block

当 sequence 被调度执行时，BlockManager 会为它分配 KV Cache block。

在 Prefill 阶段，需要为 prompt 中的 token 准备 KV Cache 空间。  
在 Decode 阶段，每生成一个新 token，可能需要追加新的 KV Cache slot；当当前 block 填满，就要申请新的 block。

#### 6. Prefill 阶段

##### 是什么

Prefill 是对完整 prompt 做一次前向推理，计算 prompt 中所有 token 的 hidden states，并为这些 token 生成 K/V Cache。

例如 prompt 是：

```text
请解释什么是深度学习
```

Prefill 会一次性处理这整个 prompt。

##### 为什么重要

Prefill 通常计算量很大，因为它要处理完整输入序列，并进行大量矩阵乘法和 attention 计算。prompt 越长，Prefill 的计算量越大。

##### 输出什么

Prefill 结束后，系统会得到：

- prompt 对应的 KV Cache；
- 最后一个位置的 logits；
- 根据 logits 采样出的第一个输出 token。

#### 7. Decode 阶段

##### 是什么

Decode 是自回归生成阶段。每次只生成一个新 token，然后把这个 token 追加到 sequence 后面，再进入下一轮 decode。

例如：

```text
Prompt: 请解释什么是深度学习
第 1 次 Decode: 生成“深度”
第 2 次 Decode: 生成“学习”
第 3 次 Decode: 生成“是”
...
```

##### KV Cache 如何参与

Decode 时不需要重新计算所有历史 token 的 K/V，而是：

- 当前新 token 计算新的 Q/K/V；
- 当前 token 的 K/V 追加到 KV Cache；
- 历史 token 的 K/V 从 KV Cache 读取；
- 当前 token 的 Q 与所有历史 K 做 attention；
- 得到当前 token 的输出表示；
- 生成下一个 token。

注意：**缓存的是 K 和 V，不是 Q。**  
Q 通常是当前 step 根据当前 token 重新计算得到的。

##### 停止条件

Decode 会一直持续，直到：

- 生成 EOS token；
- 达到 max tokens；
- 命中 stop words / stop sequence；
- 请求被取消；
- 服务端策略终止。

#### 8. 初学者容易误解的地方

- 误解 1：Prefill 和 Decode 都只是 forward，没有区别。  
  两者都是 forward，但输入形态、计算量、访存模式、性能瓶颈完全不同。

- 误解 2：KV Cache 会缓存 Q/K/V 三者。  
  通常缓存 K/V，Q 是当前 token 临时计算的。

- 误解 3：Decode 一次生成完整回答。  
  Decode 是一个 token 一个 token 地生成。

- 误解 4：Tokenizer 是推理引擎的核心优化点。  
  Tokenizer 很重要，但通常复用成熟实现；推理引擎的核心优化更多在调度、KV Cache、GPU 执行和显存管理。

---

### 第 6 部分：Prefill 与 Decode 的性能差异 —— compute-bound 与 memory-bound

#### 1. 核心知识点

课程最后重点解释 Prefill 和 Decode 的性能差异：

- **Prefill 通常是计算密集型（compute-bound）**
- **Decode 通常是访存密集型（memory-bound）**

老师还提到了 Roofline Model，用来理解不同计算任务受算力还是带宽限制。

#### 2. Prefill 为什么是计算密集型

Prefill 要处理整个 prompt。假设 prompt 长度为 `N`，模型每层都要对这些 token 做矩阵乘法、attention、MLP 等操作。

尤其是矩阵乘法部分通常能较好利用 GPU Tensor Core，因此计算量大，GPU 算力利用率较高。

在 Prefill 阶段：

- 输入 token 多；
- 矩阵乘法规模较大；
- 并行度较高；
- 更容易把 GPU 算力打满；
- 常用 FlashAttention 等高效 attention kernel。

因此 Prefill 更偏 compute-bound。

#### 3. Decode 为什么是访存密集型

Decode 每次只处理一个新 token。单步计算量较小，但需要读取历史所有 token 的 KV Cache。

随着上下文变长，Decode 每一步都要访问更多历史 K/V：

```text
第 t 步 Decode：读取前 t 个 token 的 K/V Cache
```

因此 Decode 的瓶颈往往不是算力，而是：

- KV Cache 读取带宽；
- HBM 访问；
- 多卡场景下的通信；
- attention kernel 对小 batch / 单 token 的效率；
- batch 中 sequence 长度不一致带来的调度开销。

课程中特别提到一种情况：Prefill 和 Decode 可能不在同一张卡上，KV Cache 可能涉及卡间传输；而模型权重在本地 HBM 上读取速度更高，因此 Decode 更容易受访存和通信限制。

#### 4. Roofline Model

Roofline Model 用来分析一个算子是受算力限制还是受带宽限制。

- 横轴通常是算术强度：每读取一个字节数据能做多少计算；
- 纵轴是性能；
- 越靠左，越可能 memory-bound；
- 越靠右，越可能 compute-bound。

把它放到推理系统中理解：

| 阶段 | 典型特征 | 性能瓶颈 |
|---|---|---|
| Prefill | prompt 长、矩阵乘法大、并行度高 | 算力 / compute-bound |
| Decode | 每步一个 token、频繁读 KV Cache | 显存带宽 / memory-bound |
| 长上下文 Decode | 读取历史 K/V 更多 | KV Cache 访存更重 |
| 多卡 Decode | 可能有 KV Cache 或中间结果通信 | 通信带宽 + 同步开销 |

#### 5. FlashAttention 与 KV Cache

课程提到：

- Prefill 中可能使用 FlashAttention forward；
- Decode 中主要使用 with KV Cache 的路径；
- Prefill 在部分场景下也可能复用 KV Cache，但本节没有展开。

可以这样理解：

- **FlashAttention** 主要优化 attention 的计算和显存读写，尤其适合较长序列的 attention 计算。
- **KV Cache decode kernel** 主要面向单 token 增量生成，需要高效读取历史 K/V，并把新 K/V 写入缓存。

#### 6. 初学者容易误解的地方

- 误解 1：Decode 计算量小，所以一定快。  
  单步计算量小，但每一步都要读历史 KV Cache，而且要循环很多步，总体可能很慢。

- 误解 2：Prefill 长，所以一定比 Decode 慢。  
  Prefill 虽然一次计算量大，但并行度高；Decode 串行依赖强，每个 token 必须等上一个 token 生成后才能继续。

- 误解 3：GPU 性能只看 FLOPS。  
  推理性能还强烈依赖 HBM 带宽、KV Cache 布局、通信、batch 组织和 kernel 效率。

---

## 3. 核心概念详细解释

### 3.1 Transformer

Transformer 是当前大语言模型的主流基础架构。它由多层堆叠的 Transformer block 组成，每层通常包含：

- Self-Attention；
- MLP / FFN；
- LayerNorm / RMSNorm；
- Residual connection。

在推理系统中，Transformer 是被执行的模型本体；vLLM / nano-vLLM 的任务是高效组织请求，让 Transformer forward 在 GPU 上高效执行。

### 3.2 Attention / Self-Attention

Self-Attention 让每个 token 根据上下文中其他 token 的信息更新自身表示。

它的核心过程是：

1. 输入 hidden states 乘以权重矩阵得到 Q、K、V；
2. Q 与 K 做相似度计算；
3. 经过 softmax 得到 attention 权重；
4. 权重乘以 V 得到输出。

在自回归推理中，当前 token 只能关注它之前的 token，不能看到未来 token。

### 3.3 Token

Token 是模型处理文本的最小单位。它可能是一个字、一个词、一个子词，也可能是字节片段。

推理系统所有后续操作都基于 token ids，而不是原始字符串。

### 3.4 Request

Request 是用户层面的请求，例如：

```text
请解释什么是深度学习
```

它包含 prompt、采样参数、最大输出长度、停止条件等。

### 3.5 Sequence

Sequence 是推理引擎内部的执行单位，表示一个正在生成的 token 序列。Scheduler 通常调度 sequence，而不是直接调度原始字符串。

### 3.6 Batch

Batch 是一次送入模型执行的一组 sequence。  
在推理中，batch 不只是为了训练时那种矩阵并行，更是为了提高 GPU 利用率。

### 3.7 Continuous Batching

Continuous Batching 是动态批处理。它允许新请求在推理过程中加入 batch，已完成请求退出 batch。

它解决的是静态 batch 中短序列提前结束导致 GPU 空泡的问题。

### 3.8 Prefill

Prefill 是处理完整 prompt 的阶段。它计算 prompt 中所有 token 的 K/V Cache，并产生第一个输出 token 的 logits。

它通常 compute-bound，长 prompt 会显著增加 Prefill 时间。Benchmark 中的 **TTFT（Time To First Token）** 与 Prefill 关系很大。本节课没有重点讲 TTFT，但后续做 vLLM Benchmark 时必须关注它。

### 3.9 Decode

Decode 是逐 token 生成阶段。每轮 decode 生成一个 token，并把新 token 的 K/V 追加到 KV Cache。

它通常 memory-bound，输出越长，Decode 循环越多。Benchmark 中的 **TPOT（Time Per Output Token）** 与 Decode 关系很大。本节课没有重点讲 TPOT，但它是推理性能分析的关键指标。

### 3.10 KV Cache

KV Cache 是把历史 token 的 Key 和 Value 保存下来，供后续 token 复用。

没有 KV Cache 时，每生成一个新 token 都要重新计算整个历史序列的 K/V，代价极高。  
有 KV Cache 后，历史 K/V 直接读取，只计算当前 token 的新 K/V。

### 3.11 PagedAttention

PagedAttention 是 vLLM 的核心机制之一。它把 KV Cache 划分为固定大小 block，用类似操作系统分页的方式管理。

它解决：

- KV Cache 显存碎片；
- 最大长度预分配浪费；
- 动态增长的显存管理问题。

### 3.12 Block

Block 是 KV Cache 的物理管理单位。一个 sequence 的 token 逻辑上连续，但它们的 K/V 可以存放在不同 block 中。

### 3.13 Block Manager

Block Manager 负责分配、释放、映射 KV Cache block。它是 PagedAttention 能工作的关键组件。

### 3.14 Scheduler

Scheduler 负责请求调度。它决定：

- 哪些 waiting request 可以进入 running；
- 哪些 running sequence 继续 decode；
- 当前 batch 包含哪些 sequence；
- KV Cache 容量是否足够；
- 结束的 request 何时释放资源。

### 3.15 GPU / CUDA / 显存

GPU 是执行大模型矩阵计算的硬件。CUDA 是 NVIDIA GPU 编程平台。显存主要存储：

- 模型权重；
- 激活中间结果；
- KV Cache；
- 输入输出张量；
- 临时 buffer。

在推理服务中，显存瓶颈通常来自模型权重和 KV Cache，尤其是高并发、长上下文场景。

### 3.16 吞吐量、延迟、TTFT、TPOT、QPS

这些指标本节课没有系统展开，但与课程内容直接相关：

| 指标 | 含义 | 和本节课的关系 |
|---|---|---|
| 吞吐量 | 单位时间生成 token 数或处理请求数 | Continuous Batching 可提升吞吐 |
| 延迟 | 一个请求从进入到完成的时间 | Scheduler、Prefill、Decode 都会影响 |
| TTFT | Time To First Token，第一个 token 延迟 | 主要受排队和 Prefill 影响 |
| TPOT | Time Per Output Token，平均每个输出 token 时间 | 主要受 Decode 影响 |
| QPS | 每秒请求数 | 受请求长度、batch、显存和调度影响 |

### 3.17 FlashAttention

FlashAttention 是高效 Attention 实现，通过优化显存读写和计算组织，提高 attention 计算效率。  
本节课提到 Prefill 路径中会使用 FlashAttention forward，而 Decode 更多走 KV Cache 路径。

### 3.18 Tensor Parallel

课程中提到大模型单卡放不下时，需要把模型切分到不同卡上，每个 worker 持有模型一部分，这对应 Tensor Parallel 或类似并行方式。

Tensor Parallel 的目标是让多张 GPU 合作执行同一个模型层的计算，但会引入卡间通信成本。

### 3.19 Quantization、FP16 / BF16 / INT4 / INT8

本节课没有展开量化和数据类型，但后续做推理优化必须学习：

- FP16 / BF16：常见半精度推理格式；
- INT8 / INT4：量化推理格式，可降低显存和带宽压力；
- 量化会影响模型精度、显存占用和推理速度。

---

## 4. 整节课技术流程串联

下面用“从输入到输出”的方式，把整节课串成完整流程。

### 4.1 用户输入 prompt

用户向推理服务提交 prompt，例如：

```text
请解释什么是深度学习
```

在服务化系统中，用户可能通过 HTTP API、OpenAI-compatible API、Python SDK 或命令行提交请求。本节课主要讨论推理引擎内部流程，没有展开 API 层。

### 4.2 文本变成 token

Tokenizer 把字符串转成 token ids：

```text
"请解释什么是深度学习"
  ↓
[10123, 3456, 7890, ...]
```

这个过程使用模型配套 tokenizer，例如 BPE 或 SentencePiece。

### 4.3 构造 Request / Sequence

推理引擎把 token ids 和采样参数封装成 Request，再转成内部调度使用的 Sequence。

Sequence 记录：

- 已有 token；
- 当前生成到哪里；
- 对应的 KV Cache block；
- 是否 finished；
- 采样参数；
- 最大输出长度。

### 4.4 Sequence 进入 Scheduler

新 sequence 先进入 waiting queue。Scheduler 每轮检查资源：

- 当前 running sequence 有多少；
- GPU batch 容量是否足够；
- KV Cache block 是否足够；
- 是否可以接纳新 request；
- 已结束 sequence 是否需要释放。

### 4.5 BlockManager 分配 KV Cache 空间

当 Scheduler 决定执行某个 sequence 时，BlockManager 为它分配 KV Cache block。

这一步非常重要，因为 prompt 长度和输出长度动态变化，不能简单一次性预分配最大长度。

### 4.6 Prefill：完整 prompt 前向计算

Prefill 对 prompt 的所有 token 做一次模型 forward。

此时 Transformer 会执行：

1. embedding；
2. 多层 attention；
3. MLP；
4. norm；
5. lm_head；
6. 输出 logits。

在每一层 attention 中，prompt token 的 K/V 会被写入 KV Cache。

Prefill 结束后，系统通常得到第一个可采样输出 token。

### 4.7 Decode：逐 token 增量生成

Decode 阶段循环执行：

```text
读取上一步生成 token
  ↓
计算当前 token 的 Q/K/V
  ↓
读取历史 KV Cache
  ↓
执行 attention 和后续层
  ↓
得到 logits
  ↓
采样下一个 token
  ↓
追加到 sequence
  ↓
把新 K/V 写入 KV Cache
  ↓
判断是否结束
```

每一轮 decode 只能生成一个 token，因为自回归模型的第 t+1 个 token 依赖第 t 个 token。

### 4.8 KV Cache 为什么能加速

如果没有 KV Cache，第 t 次 decode 需要重新计算前 t 个 token 的 K/V。  
有 KV Cache 后，历史 K/V 已经保存，后续只需计算当前 token 的 K/V。

这样可以把大量重复计算转化为读取缓存。

代价是：显存占用增加，而且长上下文下 Decode 需要频繁读取大量历史 K/V，因此会变成 memory-bound。

### 4.9 多用户请求如何被调度

多个用户请求会同时进入系统。Scheduler 维护 waiting 和 running 状态。

每个 step：

- 一些旧请求继续 decode；
- 一些新请求进入 prefill；
- 一些结束请求释放 block；
- batch 动态变化。

这就是 Continuous Batching 的基础。

### 4.10 batch 和 Continuous Batching 为什么重要

GPU 适合大规模并行计算。如果一个请求一个请求地跑，GPU 很难被喂满。  
batch 可以把多个 sequence 组织到一起，提高并行度。

但普通静态 batch 会有空泡。Continuous Batching 可以让新请求动态补位，提高吞吐。

### 4.11 显存为什么会成为瓶颈

显存主要被三类东西占用：

1. 模型权重；
2. KV Cache；
3. 临时张量和中间 buffer。

其中 KV Cache 会随以下因素增长：

- batch size / 并发请求数；
- prompt 长度；
- output 长度；
- 模型层数；
- attention head 数；
- hidden size；
- 数据类型。

因此长上下文和高并发会让 KV Cache 成为显存瓶颈。

### 4.12 vLLM / nano-vLLM 主要解决什么问题

vLLM / nano-vLLM 主要解决：

- 如何高效管理 KV Cache；
- 如何减少显存碎片；
- 如何避免最大长度预分配浪费；
- 如何动态调度多用户请求；
- 如何提高 GPU 利用率；
- 如何把 Transformer 推理变成稳定高效的服务。

### 4.13 最终如何落到 AI Infra 项目实践

这节课最终会落到三个项目方向：

1. **nano-vLLM 源码阅读**  
   理解 LLMEngine、Scheduler、BlockManager、Worker、Model、KV Cache 的调用链。

2. **vLLM Benchmark**  
   通过实验观察并发数、输入长度、输出长度、batch 策略对 TTFT、TPOT、吞吐、显存的影响。

3. **推理服务部署**  
   用 vLLM 部署 Qwen / Llama 等模型，构建 OpenAI-compatible API 服务，并分析线上服务性能瓶颈。

---

## 5. 我从这节课中能学到什么

### 5.1 概念层面

你应该掌握以下概念：

| 概念 | 在系统中的位置 |
|---|---|
| Token | 文本进入模型前的数字化单位 |
| Request | 用户侧请求 |
| Sequence | 推理引擎内部调度单位 |
| Prefill | 处理 prompt，生成初始 KV Cache |
| Decode | 自回归逐 token 生成 |
| KV Cache | 保存历史 token 的 K/V，减少重复计算 |
| PagedAttention | vLLM 管理 KV Cache 的核心机制 |
| Block | KV Cache 的分配单位 |
| BlockManager | block 分配、释放、映射管理模块 |
| Scheduler | 请求调度和 batch 组织模块 |
| Continuous Batching | 动态批处理，提高 GPU 利用率 |
| Worker / ModelRunner | GPU 执行层 |
| Model | Transformer 模型本体 |
| compute-bound | 性能主要受算力限制 |
| memory-bound | 性能主要受带宽/访存限制 |

### 5.2 系统层面

你应该理解一个大模型推理系统至少包括：

1. **接口层**  
   接收用户请求，可能是 HTTP API / OpenAI-compatible API。

2. **Tokenizer 层**  
   把 prompt 转为 token ids。

3. **请求管理层**  
   创建 request / sequence，维护状态。

4. **调度层**  
   Scheduler 决定哪些 sequence 进入当前 batch。

5. **显存管理层**  
   BlockManager 管理 KV Cache block。

6. **执行层**  
   Worker / ModelRunner 调用模型 forward，在 GPU 上执行。

7. **模型层**  
   Transformer 计算 logits。

8. **采样层**  
   根据 logits 采样下一个 token。

9. **输出层**  
   detokenize 并返回给用户。

10. **资源回收层**  
   请求结束后释放 block 和状态。

### 5.3 工程层面

后续做项目时，这节课知识会直接用到：

#### nano-vLLM 源码阅读

建议阅读顺序：

1. 找 `LLMEngine` 或外部 `generate` 入口；
2. 看 request 如何变成 sequence；
3. 看 Scheduler 如何维护 waiting / running；
4. 看 BlockManager 如何分配 block；
5. 看 Worker / ModelRunner 如何准备输入；
6. 看 model forward 如何区分 prefill / decode；
7. 看 KV Cache 如何读写；
8. 看生成 token 如何追加回 sequence。

#### vLLM Benchmark

你可以设计实验观察：

- 输入长度变长，TTFT 如何变化；
- 输出长度变长，TPOT 和总延迟如何变化；
- 并发数增加，吞吐和显存如何变化；
- max model len 增加，KV Cache 空间如何变化；
- batch 策略不同，GPU 利用率如何变化。

#### 推理服务部署

部署 vLLM 服务时，你会遇到：

- 模型权重加载；
- GPU 显存不足；
- max model len 配置；
- 并发请求排队；
- TTFT 太高；
- TPOT 太慢；
- GPU 利用率不稳定；
- 长 prompt 抢占资源；
- 输出长度差异导致调度问题。

这些都和本节课内容直接相关。

### 5.4 面试层面

这节课可以支撑你回答推理引擎方向很多基础面试题。你不需要一开始就讲 CUDA kernel 细节，而是先把系统主线讲清楚。

---

## 6. 面向 AI Infra 的项目实践启发

### 6.1 做 nano-vLLM 源码阅读时的实践方式

不要直接打开源码从第一行开始看。建议采用“问题驱动”方式：

1. `generate()` 调用后进入哪个模块？
2. Request 是在哪里创建的？
3. Sequence 的状态有哪些？
4. Scheduler 每一步返回什么？
5. BlockManager 如何判断是否有空间？
6. KV Cache 的 block table 是如何传给模型的？
7. Prefill 和 Decode 的 forward 输入有什么不同？
8. Decode 后新 token 如何写回 sequence？
9. 请求结束后 block 如何释放？

你可以边读源码边画一张调用链图：

```text
LLMEngine.generate()
  → add_request()
  → tokenizer
  → Sequence
  → Scheduler.schedule()
  → BlockManager.allocate()
  → Worker.execute_model()
  → Model.forward()
  → Attention with KV Cache
  → sample()
  → update Sequence
```

### 6.2 做 vLLM Benchmark 时的实验设计

建议设计四组实验：

#### 实验 1：固定输出长度，改变输入长度

目的：观察 Prefill 对 TTFT 的影响。

预期：

- prompt 越长；
- Prefill 计算越重；
- TTFT 越高；
- KV Cache 初始占用越大。

#### 实验 2：固定输入长度，改变输出长度

目的：观察 Decode 对 TPOT 和总延迟的影响。

预期：

- 输出越长；
- Decode step 越多；
- 总延迟越高；
- KV Cache 持续增长。

#### 实验 3：改变并发数

目的：观察 Continuous Batching 和调度效果。

预期：

- 并发增加初期吞吐提升；
- 到达显存或算力瓶颈后吞吐增长放缓；
- 排队延迟增加；
- TTFT 可能升高。

#### 实验 4：改变 max model len / GPU memory utilization

目的：观察 KV Cache 显存管理。

预期：

- max model len 越大，单请求潜在 KV Cache 空间越大；
- 可并发请求数量可能下降；
- PagedAttention 能缓解但不能消除显存瓶颈。

### 6.3 做推理服务部署时的排查思路

如果你部署 vLLM 后发现性能不理想，可以按以下顺序排查：

1. 模型权重是否占满显存？
2. KV Cache 剩余空间是否不足？
3. max model len 是否设置过大？
4. 请求输入长度是否过长？
5. 输出 max tokens 是否过高？
6. batch 是否太小导致 GPU 利用率低？
7. 并发是否太高导致排队延迟大？
8. TTFT 高是排队问题还是 Prefill 问题？
9. TPOT 高是 Decode 访存问题还是 kernel 问题？
10. 是否存在多卡通信瓶颈？

---

## 7. 面试可能考点

### 7.1 为什么需要 vLLM 这样的推理引擎，不能直接用 PyTorch generate？

回答思路：

PyTorch / Transformers 的 generate 适合单机单请求或简单推理，但 LLM serving 面临多用户并发、动态长度、KV Cache 显存增长、batch 调度和吞吐优化等问题。vLLM 通过 PagedAttention 管理 KV Cache，通过 Continuous Batching 动态组织请求，从而提高显存利用率和 GPU 利用率。

### 7.2 vLLM 主要解决了哪些问题？

回答思路：

主要解决三类问题：

1. KV Cache 显存碎片化；
2. 最大长度预分配导致的显存浪费；
3. 静态 batch 中短请求提前结束导致 GPU 空泡。

对应方案是 PagedAttention 和 Continuous Batching。

### 7.3 什么是 KV Cache？为什么它能加速推理？

回答思路：

在 Transformer 自回归推理中，后续 token 的 attention 需要访问历史 token 的 K/V。如果每步都重新计算历史 K/V，会产生大量重复计算。KV Cache 把历史 token 的 K/V 保存下来，后续 decode 直接读取，只计算当前 token 的 K/V，因此显著加速生成。

### 7.4 KV Cache 缓存的是 Q、K、V 全部吗？

回答思路：

通常缓存 K 和 V，不缓存 Q。Q 是当前 token 在当前 step 计算出来的查询向量；历史 token 的 K/V 会被后续 token 反复使用，所以需要缓存。

### 7.5 Prefill 和 Decode 有什么区别？

回答思路：

Prefill 处理完整 prompt，一次性计算所有输入 token 的 KV Cache，计算量大、并行度高，通常 compute-bound。Decode 每次只生成一个 token，需要读取历史 KV Cache，单步计算小但访存多，通常 memory-bound，并且自回归依赖导致串行性强。

### 7.6 为什么 Decode 通常是 memory-bound？

回答思路：

Decode 每步只处理一个 token，矩阵计算规模较小，但需要读取历史所有 token 的 KV Cache。上下文越长，读取的 K/V 越多，因此性能受显存带宽和缓存布局影响很大。在多卡场景下，还可能受通信带宽限制。

### 7.7 什么是 PagedAttention？

回答思路：

PagedAttention 是 vLLM 的 KV Cache 管理机制。它借鉴操作系统分页思想，把 KV Cache 划分为固定大小 block，让 sequence 的逻辑 token 位置映射到物理 block。这样可以减少连续显存需求，降低碎片化，并避免最大长度预分配浪费。

### 7.8 PagedAttention 改变了 Attention 的数学计算吗？

回答思路：

没有。Attention 的数学含义不变，改变的是 KV Cache 的物理存储和访问方式。它让 attention kernel 可以通过 block table 找到历史 K/V。

### 7.9 什么是 Continuous Batching？

回答思路：

Continuous Batching 是动态批处理。推理过程中，已完成的 sequence 可以退出，新来的 request 可以加入当前执行批次，从而减少静态 batch 的空泡，提高 GPU 利用率和吞吐。

### 7.10 Scheduler 在推理引擎里做什么？

回答思路：

Scheduler 负责选择当前 step 要执行哪些 sequence。它需要管理 waiting/running 状态，检查 KV Cache 容量，决定 prefill/decode 的执行集合，并处理请求完成后的资源释放。

### 7.11 BlockManager 做什么？

回答思路：

BlockManager 负责 KV Cache block 的分配、释放和映射。它维护 sequence 的逻辑 token 到物理 block 的关系，是 PagedAttention 的核心支撑模块。

### 7.12 Request 和 Sequence 有什么区别？

回答思路：

Request 是用户请求，包含 prompt 和采样参数；Sequence 是推理引擎内部执行和调度的 token 序列。一个 request 可能对应一个或多个 sequence。

### 7.13 TTFT 和 TPOT 分别受什么影响？

回答思路：

TTFT 主要受排队时间和 Prefill 时间影响；TPOT 主要受 Decode 阶段影响，包括 KV Cache 读取、batch 组织、GPU 显存带宽和 kernel 效率。

### 7.14 为什么长上下文会增加显存压力？

回答思路：

因为每个历史 token 都需要在每层保存 K/V。上下文长度越长，KV Cache 越大；并发请求越多，KV Cache 总量越大。因此长上下文和高并发会显著增加显存压力。

### 7.15 nano-vLLM 适合怎么学习？

回答思路：

nano-vLLM 是 vLLM 的轻量级实现，适合先从 LLMEngine、Scheduler、BlockManager、Worker/ModelRunner、Model forward 这几个核心模块入手，理解一个请求从输入到输出的完整路径，再逐步深入 KV Cache 和 attention kernel。

---

## 8. 本节课复习提纲

### 8.1 一句话总结

这节课讲的是：**为什么需要 vLLM / nano-vLLM 这样的推理引擎，以及一个 prompt 在推理系统中如何经过 tokenization、调度、KV Cache 管理、Prefill、Decode，最终生成输出。**

### 8.2 必须掌握的主线

```text
为什么需要推理引擎
  ↓
传统 PyTorch/Transformers generate 的问题
  ↓
显存碎片化、静态 batch 空泡、预分配浪费
  ↓
vLLM 用 PagedAttention 管理 KV Cache
  ↓
vLLM 用 Continuous Batching 动态调度请求
  ↓
nano-vLLM 用 LLMEngine/Scheduler/BlockManager/Worker/Model 复现核心流程
  ↓
请求从 prompt → token → sequence → prefill → decode → output
  ↓
Prefill compute-bound，Decode memory-bound
```

### 8.3 复习时优先记住的概念

1. vLLM 的核心不是“换了一个 generate”，而是系统级推理优化。
2. KV Cache 是大模型推理的核心资源。
3. PagedAttention 用 block 管理 KV Cache。
4. Continuous Batching 让 batch 动态变化。
5. Scheduler 决定当前 step 执行哪些 sequence。
6. BlockManager 决定 KV Cache block 怎么分配。
7. Prefill 处理完整 prompt，Decode 逐 token 生成。
8. Prefill 通常 compute-bound，Decode 通常 memory-bound。
9. nano-vLLM 适合作为 vLLM 源码学习入口。
10. Benchmark 指标 TTFT、TPOT、吞吐、显存都能从这条主线解释。

### 8.4 建议画出的图

复习时建议自己画三张图：

1. **推理系统架构图**

```text
API / User
  ↓
LLMEngine
  ↓
Scheduler
  ↓
BlockManager
  ↓
Worker / ModelRunner
  ↓
Model / Attention / KV Cache
```

2. **请求生命周期图**

```text
Prompt
  ↓
Tokenizer
  ↓
Request
  ↓
Sequence
  ↓
Waiting Queue
  ↓
Running
  ↓
Prefill
  ↓
Decode Loop
  ↓
Finished
  ↓
Release Blocks
```

3. **PagedAttention 映射图**

```text
Sequence token positions:
[0, 1, 2, 3] [4, 5, 6, 7] [8, 9, ...]

Physical KV blocks:
Block 7      Block 1      Block 12
```

---

## 9. 后续学习建议

### 9.1 下一步先补 Transformer 推理基础

如果你对 Transformer 还不熟，建议补：

1. token embedding；
2. attention 中 Q/K/V 的来源；
3. causal mask；
4. logits 和 sampling；
5. 自回归生成；
6. KV Cache 原理。

目标不是先推公式，而是能说清楚：为什么生成下一个 token 要依赖历史 token，为什么历史 K/V 可以缓存。

### 9.2 再读 nano-vLLM 源码

源码阅读建议分三轮：

#### 第一轮：只看主流程

目标：跑通“请求怎么流转”。

重点：

- LLMEngine；
- generate；
- request；
- sequence；
- scheduler；
- worker。

#### 第二轮：看 KV Cache 和 block

目标：搞懂 PagedAttention 的数据结构。

重点：

- block；
- block table；
- block manager；
- allocate/free；
- append token；
- cache slot。

#### 第三轮：看模型 forward

目标：搞懂 Prefill / Decode 输入差异。

重点：

- input ids；
- positions；
- attention metadata；
- FlashAttention；
- KV Cache read/write；
- logits；
- sampling。

### 9.3 再做 vLLM Benchmark

建议实验顺序：

1. 单请求短 prompt；
2. 单请求长 prompt；
3. 多请求低并发；
4. 多请求高并发；
5. 固定输入变输出；
6. 固定输出变输入；
7. 观察显存变化；
8. 记录 TTFT、TPOT、吞吐。

### 9.4 最后做推理服务部署项目

可以设计一个正式项目：

**项目名称：基于 vLLM 的大模型推理服务部署与性能优化实验**

核心内容：

- 部署 Qwen / Llama 模型；
- 提供 OpenAI-compatible API；
- 编写 Benchmark 脚本；
- 统计 TTFT、TPOT、吞吐、总延迟、显存占用；
- 对比不同输入长度、输出长度、并发数；
- 分析 Prefill / Decode 性能差异；
- 总结 KV Cache 和 batch 调度对性能的影响。

这个项目正好可以把本节课知识转化为简历上的 AI Infra 推理项目。

---

## 10. 转文字错误纠正表

| 转文字内容 | 建议纠正 | 说明 |
|---|---|---|
| RM serving / WRM / VLM | LLM serving / vLLM | 根据上下文是在讲大模型推理服务和 vLLM |
| NanoWRM / NalviaM / downloadvrm | nano-vLLM | 课程主题是 nano-vLLM |
| Petro / Pathor | PyTorch | 指 PyTorch 直接推理 |
| HackingFace | Hugging Face | 指 Hugging Face Transformers |
| Transformers | Transformers | 有时指库，有时指模型架构，要结合语境 |
| prove / prop / ProP | prompt | 用户输入提示词 |
| opput | output | 输出 |
| tokennyler | tokenizer | 分词器 |
| BPEBPE | BPE | Byte Pair Encoding |
| sensants piece | SentencePiece | 常见 tokenizer 算法 |
| 配置的天神 | PagedAttention | 结合上下文，指 vLLM 核心机制 |
| 讯营内存 | 虚拟内存 | 操作系统类比 |
| 业表 | 页表 | 虚拟内存映射结构 |
| prog / prolog | block | 结合 PagedAttention 语境，指 KV Cache block |
| 经贷批处理 | 静态批处理 | 指 static batching |
| 空炮 / 空号 | 空泡 | batch 中无效计算位置 |
| sched up / specialty | Scheduler | 调度器 |
| rm 英诊 | LLMEngine | 推理引擎入口 |
| walker / model walker | Worker / ModelRunner | 执行模型前向的进程/模块 |
| cubic catch / kpcatch / kvcatch | KV Cache | K/V 缓存 |
| qcatch / vcatch | Q/K/V 或 K/V Cache | 原文混乱，注意 Q 通常不缓存 |
| prefield / persiode / Presue | Prefill | prompt 预填充阶段 |
| dcode / 底扣 / 抵扣 | Decode | 自回归解码阶段 |
| flasherTension | FlashAttention | 高效 attention 实现 |
| HCCR / SSR | 可能是 HCCL / NCCL | 多卡通信库，转写不确定 |
| compute bound | compute-bound | 算力受限 |
| memory bound | memory-bound | 访存受限 |
| roofline model | Roofline Model | 算力/带宽性能分析模型 |

---

## 11. 本节课最终应形成的理解

学完这节课后，你应该能用自己的话说清楚：

> 大模型推理服务的难点不是单次 forward，而是多请求、多长度、动态输出条件下的调度和显存管理。Transformer 自回归推理需要 KV Cache 加速，但 KV Cache 又带来显存碎片和动态增长问题。vLLM 通过 PagedAttention 以 block 方式管理 KV Cache，通过 Continuous Batching 动态组织 batch，提高显存利用率和 GPU 利用率。nano-vLLM 则用更轻量的代码呈现 vLLM 的核心架构，适合作为 AI Infra 推理引擎源码学习入口。



%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]

%% 项目关联导航：结束 %%
