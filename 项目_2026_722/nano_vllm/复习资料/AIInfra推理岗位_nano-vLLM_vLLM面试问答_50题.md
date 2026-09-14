# AI Infra 推理岗位面试问答：nano-vLLM / vLLM / LLM Serving 50 题

> 生成目的：把你学习 nano-vLLM 过程中的自制笔记，整理成更接近真实 AI Infra 推理岗位面试的问答材料。  
> 适用场景：暑期实习 / 秋招 AI Infra、LLM Inference、推理引擎、模型部署与性能优化相关岗位。  
> 重点主线：不要把 vLLM 理解成“一个用了 PagedAttention 的库”，而要理解成一个围绕 **动态请求调度、KV Cache 管理、GPU 执行、吞吐/延迟权衡** 构建的在线推理服务系统。

---

## 0. 本文件如何使用

建议按三轮复习：

1. **第一轮：能说清楚主线**  
   重点掌握：为什么需要推理引擎、Prefill/Decode、KV Cache、PagedAttention、Continuous Batching、Scheduler、BlockManager。

2. **第二轮：能回答追问**  
   重点掌握：为什么这样设计、解决什么问题、代价是什么、在什么 workload 下有效。

3. **第三轮：能结合项目表达**  
   重点练习：把 nano-vLLM 源码学习和 vLLM 工程系统联系起来，不要只停留在 API 调用或概念背诵。

---

## 1. 面试高频主线总览

真实面试中，nano-vLLM / vLLM 相关问题通常会围绕下面这条链展开：

```text
在线 LLM serving 为什么难
  ↓
PyTorch / Transformers generate 为什么不够
  ↓
自回归生成、Prefill、Decode、KV Cache
  ↓
KV Cache 动态增长导致显存压力和碎片问题
  ↓
PagedAttention / block table / BlockManager
  ↓
多请求动态到达与不同长度请求混跑
  ↓
Scheduler / Continuous Batching / token budget
  ↓
TTFT、TPOT、ITL、吞吐、显存、GPU 利用率
  ↓
nano-vLLM 与 vLLM 的功能差异
  ↓
真实服务调优与问题定位
```

---

# 一、推理引擎整体架构与请求生命周期

## Q1：为什么大模型推理不能只用 Hugging Face Transformers 的 `generate()`，还需要 vLLM / nano-vLLM 这样的推理引擎？

### 参考回答

`Transformers.generate()` 解决的是“单个模型怎么加载、怎么前向、怎么生成文本”的问题，适合实验和低并发场景。但在线 LLM serving 的主要矛盾不是单次 forward 能不能跑，而是多用户请求同时到来时，如何高效利用 GPU、控制显存、降低延迟并提高吞吐。

在服务化推理中，请求有几个特点：

- prompt 长度不同；
- 输出长度未知；
- 请求动态到达和结束；
- KV Cache 会随上下文和生成长度动态增长；
- 如果静态 batch，短请求结束后会留下 GPU 空泡；
- 如果按最大上下文预分配 KV Cache，会严重浪费显存。

vLLM / nano-vLLM 这类推理引擎额外做的是：请求管理、调度、Continuous Batching、KV Cache block 管理、PagedAttention、prefix cache、采样、流式输出和性能监控。

一句话：

> `generate()` 是模型推理接口；vLLM 是面向在线服务的推理系统。

### 面试追问

如果面试官追问“那 PyTorch 不能自己写 batch 吗？”可以回答：可以写简单 batch，但难点在于动态请求、变长输出、KV Cache 动态分配、抢占、prefix cache、chunked prefill、流式输出和多卡执行，这些不是普通 batch loop 能自然解决的。

---

## Q2：你怎么用一句话概括 vLLM 的核心价值？

### 参考回答

vLLM 是一个面向在线大模型服务的高吞吐推理引擎，它把 LLM 推理从“单次模型 forward”提升成“围绕动态请求集合、KV Cache 显存、token budget 和 GPU 执行链路的资源编排问题”。

如果展开说，它的核心价值包括：

1. 用 PagedAttention / paged KV cache 管理动态增长的 KV Cache；
2. 用 Continuous Batching 让请求在每个 step 动态进出 batch；
3. 用 Scheduler 在吞吐、TTFT、ITL、TPOT 之间做权衡；
4. 用 prefix caching、chunked prefill、CUDA graph、优化 kernel、量化、多卡执行等机制进一步提升服务能力。

不要只答“vLLM 快是因为 PagedAttention”。更好的表达是：

> PagedAttention 是显存管理基础，但 vLLM 的系统优势来自调度、KV Cache 管理、batching、prefix reuse、chunked prefill 和 GPU 执行优化的组合。

---

## Q3：nano-vLLM 和 vLLM 的关系是什么？面试中应该怎么介绍你学 nano-vLLM 的价值？

### 参考回答

nano-vLLM 是一个轻量化的 vLLM 风格实现，它保留了推理引擎最核心的抽象，例如 LLMEngine、Sequence、Scheduler、BlockManager、ModelRunner、KV Cache block、Prefill、Decode、Sampler 等。它适合初学者从源码层面理解推理引擎的主流程。

但 nano-vLLM 不等于生产级 vLLM。vLLM 还包含更复杂的功能：

- API Server 和 OpenAI-compatible serving；
- 多进程 / 多卡 Worker 架构；
- 更复杂的 Scheduler；
- Prefix caching、chunked prefill、preemption 的完整生产级实现；
- Tensor Parallel / Pipeline Parallel / Expert Parallel 等分布式能力；
- CUDA graph、FlashAttention / FlashInfer、量化、LoRA、多模型服务、监控指标等工程能力。

面试中可以这样说：

> 我通过 nano-vLLM 建立了推理引擎主链路认知：prompt 如何变成 sequence，scheduler 如何调度，BlockManager 如何管理 KV Cache block，ModelRunner 如何准备 tensor 并执行 forward，最后 sampler 如何产出 token。后续学习 vLLM 时，我会把它看成 nano-vLLM 主链路的生产级扩展。

---

## Q4：一个请求从进入推理引擎到生成完整回答，完整生命周期是什么？

### 参考回答

完整链路可以按下面顺序说明：

```text
用户 prompt
  ↓
Tokenizer 编码成 token ids
  ↓
创建 Request / Sequence
  ↓
进入 Scheduler waiting 队列
  ↓
Scheduler 根据 token budget、KV block、max_num_seqs 等限制选择本轮执行对象
  ↓
BlockManager 分配 KV Cache block
  ↓
ModelRunner 准备 input_ids、positions、slot_mapping、block_tables 等 tensor
  ↓
模型执行 Prefill forward
  ↓
生成 prompt KV Cache，并根据最后一个 prompt 位置 logits 采样第一个输出 token
  ↓
Sequence 进入 running 队列
  ↓
Decode 阶段每轮输入 last_token，读取历史 KV Cache，生成一个新 token
  ↓
追加 token，判断 EOS / max_tokens / stop sequence
  ↓
完成后释放 KV Cache block
  ↓
Tokenizer decode completion token ids，返回文本
```

核心理解：

- Prefill 负责读完整 prompt、建立 KV Cache、生成第一个 token；
- Decode 负责基于已有 KV Cache 逐 token 生成后续内容；
- Scheduler 决定每一轮谁上 GPU；
- BlockManager 管理每条 sequence 的 KV Cache 存储位置；
- ModelRunner 真正组织 tensor 并调用模型 forward。

---

## Q5：LLMEngine、Scheduler、BlockManager、ModelRunner、Model 分别做什么？

### 参考回答

可以把它们看成推理引擎中的五个层次：

| 模块 | 作用 |
|---|---|
| LLMEngine | 总控入口，接收 prompt，创建 sequence，循环 step，收集输出 |
| Scheduler | 调度中心，决定当前 step 执行哪些 waiting / running sequence |
| BlockManager | KV Cache 显存管理员，分配、释放、映射物理 block |
| ModelRunner | GPU 执行员，把 sequence 转成 tensor，调用模型 forward 和 sampler |
| Model | Transformer 本体，包括 embedding、attention、MLP、norm、lm_head |

面试表达时不要说 Scheduler 只是“队列”，它更准确的职责是：

> 把有限的 token budget、KV Cache 空间和 GPU 时间片分配给动态变化的请求集合。

BlockManager 也不只是“申请显存”，它还负责逻辑 token 到物理 KV block 的映射，是 PagedAttention 能工作的基础。

---

## Q6：Request 和 Sequence 有什么区别？为什么推理引擎内部更常调度 Sequence？

### 参考回答

Request 是用户层面的请求，包含 prompt、request id、采样参数、stop 条件等。Sequence 是推理引擎内部的执行单位，表示一个正在生成的 token 序列。

在简单场景中，一个 request 对应一个 sequence。但在 beam search、best-of、多样本采样等场景中，一个 request 可能扩展成多条 sequence。因此推理引擎内部更自然地调度 sequence。

Sequence 通常保存：

- token_ids；
- prompt token 数；
- 已生成 token 数；
- sampling params；
- status：WAITING / RUNNING / FINISHED；
- block_table；
- num_cached_tokens / num_scheduled_tokens 等运行时状态。

一句话：

> Request 是用户语义层的请求；Sequence 是模型执行层的 token 序列。

---

## Q7：`generate()` 在 nano-vLLM 中大概做了什么？

### 参考回答

`generate()` 不是简单调用一次 model forward，而是推理引擎的主循环入口。大致包括：

1. 接收 prompts 和 SamplingParams；
2. 如果 prompt 是字符串，调用 tokenizer 编码成 token ids；
3. 为每个 prompt 创建 Sequence；
4. 把 Sequence 加入 Scheduler 的 waiting 队列；
5. 不断执行 `step()`，直到所有 sequence finished；
6. 每个 step 中调度 prefill 或 decode，调用 ModelRunner 执行模型；
7. Sampler 采样 token；
8. 更新 sequence 状态，判断 EOS / max_tokens；
9. 完成后 decode token ids 为文本，返回 outputs。

可以用一句话概括：

> `generate()` 是一个调度循环，不是一次性生成完整回答的函数。

---

## Q8：为什么大模型回答不是一次性生成，而是逐 token 生成？

### 参考回答

主流 Decoder-only LLM 使用自回归生成机制，目标是不断建模：

```text
P(next_token | previous_tokens)
```

也就是每一步根据当前已经出现的所有 token，预测下一个 token。生成第 t+1 个 token 时，必须先知道第 t 个 token 是什么。因此输出 token 之间存在严格的串行依赖，不能一次性并行生成完整回答。

流程是：

```text
prompt → 预测 token1
prompt + token1 → 预测 token2
prompt + token1 + token2 → 预测 token3
...
```

这也是为什么 Decode 阶段通常难以完全并行化：每一步都依赖上一步的采样结果。

---

## Q9：logits、softmax、sampling、EOS 在生成链路中分别是什么？

### 参考回答

模型 forward 后输出的是 logits，也就是词表中每个 token 成为下一个 token 的原始分数。logits 不是概率，不要求在 0 到 1 之间，也不要求和为 1。

典型链路：

```text
hidden state → lm_head → logits → temperature/top-p/top-k 等处理 → softmax → probability distribution → sample next token
```

EOS 是 End Of Sequence，即结束 token。它和普通 token 一样参与采样。当模型某一步生成 EOS，推理引擎通常认为这条 sequence 已完成，然后释放 KV Cache。

注意：模型不是提前知道“回答要结束了”，而是每一步都让 EOS 参与竞争。如果当前上下文下 EOS 概率高，并且被采样出来，就结束。

---

# 二、Prefill、Decode 与 Transformer 推理机制

## Q10：Prefill 和 Decode 的本质区别是什么？

### 参考回答

Prefill 和 Decode 都会调用模型 forward，但输入形态、计算特征和系统瓶颈不同。

| 阶段 | 输入 | 输出 | 主要任务 | 性能特征 |
|---|---|---|---|---|
| Prefill | 完整 prompt token 序列 | prompt KV Cache + 第一个 token logits | 读完整 prompt，建立 KV Cache | 通常 compute-bound |
| Decode | 每条 sequence 的 last token | 下一个 token logits | 利用历史 KV Cache 逐 token 生成 | 通常 memory-bound |

Prefill 要一次处理很多 prompt token，矩阵乘法规模较大，GPU 并行度高。Decode 每轮只处理一个新 token，但要读取历史所有 KV Cache，单步计算小、访存压力大，而且生成过程串行。

一句话：

> Prefill 是“读题并建缓存”；Decode 是“拿缓存逐字写答案”。

---

## Q11：Prefill 阶段为什么用 prompt 最后一个位置的 logits 生成第一个输出 token？

### 参考回答

Decoder-only 自回归模型在每个位置都会预测“下一个 token”。对于完整 prompt：

```text
token0 token1 token2 ... tokenN
```

最后一个 prompt token 的 hidden state 通过 causal attention 已经聚合了前面所有 prompt 信息。因此它对应的 logits 表示：

```text
P(next_token | token0, token1, ..., tokenN)
```

这正是第一个输出 token 所需要的概率分布。所以 Prefill 虽然会计算 prompt 中所有 token 的 hidden states，但真正用于生成第一个回答 token 的是 prompt 最后位置的 logits。

---

## Q12：Decode 阶段为什么每次只输入一个 token？这样模型不会忘记 prompt 吗？

### 参考回答

不会忘记。因为 prompt 和历史生成 token 的 K/V 已经存在 KV Cache 中。

Decode 阶段生成下一个 token 时，只需要输入当前 sequence 的 last token。模型会：

1. 对 last token 计算当前层的 Q/K/V；
2. 将当前 token 的 K/V 写入 KV Cache；
3. 用当前 token 的 Q 去查询历史 KV Cache 中的 K；
4. 根据 attention 权重汇总历史 V；
5. 经过后续层和 lm_head 得到 logits；
6. 采样下一个 token。

所以 Decode 不是丢掉 prompt，而是通过 KV Cache 读取 prompt 的历史信息。

---

## Q13：为什么说 Prefill 通常 compute-bound，而 Decode 通常 memory-bound？

### 参考回答

Prefill 处理完整 prompt，输入 token 多，矩阵乘法规模较大，更容易利用 GPU Tensor Core，算术强度较高，因此常表现为 compute-bound。

Decode 每步只处理一个 token，矩阵计算规模小，但需要访问越来越长的历史 KV Cache。上下文越长，每一步读取的 K/V 越多，瓶颈更容易落在 HBM 带宽、KV Cache 布局、attention kernel 访存效率和多卡通信上，因此常表现为 memory-bound。

但要注意这不是绝对的：具体瓶颈还与 batch size、模型结构、硬件、上下文长度、kernel 实现和并行方式有关。

面试中可以这样说：

> Prefill 并行度高，主要看算力；Decode 单步计算小但频繁读 KV Cache，主要看带宽和调度效率。

---

## Q14：Decode 阶段一轮 batch size = 512 是什么意思？是不是 512 个完整 prompt？

### 参考回答

不是。Decode 阶段 batch size = 512，表示当前有 512 条 running sequence 同时参与这一轮 decode。每条 sequence 通常只输入自己的 last token，并结合自己的 KV Cache 生成一个新 token。

也就是说，这一轮输入大致是：

```text
[seq0.last_token, seq1.last_token, ..., seq511.last_token]
```

模型会输出 512 组 logits，Sampler 分别为 512 条 sequence 采样一个 next token。

完整 prompt 不会反复输入，因为 prompt 的 K/V 已经在 Prefill 中写入 KV Cache。

---

## Q15：attention head、KV head、GQA/MQA 和 KV Cache 有什么关系？

### 参考回答

Attention head 是 Transformer 中并行计算注意力的通道。每个 head 会有自己的 Q/K/V 表示。传统 MHA 中，query heads 和 KV heads 数量相同。

但 KV Cache 的显存占用与 KV heads 数量直接相关：

```text
KV Cache ∝ layers × kv_heads × head_dim × tokens × 2(K,V) × dtype_bytes
```

因此现代模型常用 GQA 或 MQA 降低 KV Cache 压力：

- MHA：每个 query head 有独立 K/V；
- GQA：多个 query head 共享一组 KV head；
- MQA：所有 query head 共享较少甚至 1 个 KV head。

GQA/MQA 不只是模型结构优化，也直接影响推理服务中的 KV Cache 显存占用和 Decode 访存压力。

---

# 三、KV Cache、PagedAttention 与 BlockManager

## Q16：什么是 KV Cache？为什么它能加速推理？

### 参考回答

KV Cache 是在自回归推理中保存历史 token 在每层 attention 中计算得到的 Key 和 Value 向量。

没有 KV Cache 时，每生成一个新 token，都要重新计算完整历史序列的 K/V，重复计算巨大。有了 KV Cache 后，历史 token 的 K/V 只计算一次，后续 decode 直接读取，只计算当前 token 的新 K/V。

它的本质是：

> 用显存空间换取重复计算的减少。

代价是显存占用随并发数、上下文长度、生成长度、层数、KV heads、head_dim 和 dtype 增长。

---

## Q17：KV Cache 缓存的是 Q、K、V 全部吗？为什么通常不缓存 Q？

### 参考回答

通常只缓存 K 和 V，不缓存 Q。

原因是：生成当前 token 时，需要用当前 token 的 Q 去查询历史 token 的 K，并加权汇总历史 token 的 V。历史 token 的 K/V 会被后续每一步反复使用，所以值得缓存。

历史 token 的 Q 只在它自己作为“当前 token”时用过，后续 token 不需要拿历史 Q 来做 attention。因此缓存 Q 没有明显收益，还会浪费显存。

一句话：

> 后续 token 会反复用历史 K/V，但不会反复用历史 Q。

---

## Q18：KV Cache 显存大小如何估算？

### 参考回答

常用粗略公式：

```text
KV Cache bytes ≈ 2 × num_layers × num_kv_heads × head_dim × total_tokens × dtype_bytes
```

其中：

- `2` 表示 K 和 V；
- `num_layers` 是 Transformer 层数；
- `num_kv_heads` 是 KV head 数量，不一定等于 query head 数量；
- `head_dim` 是每个 head 的维度；
- `total_tokens` 是所有 active sequence 的上下文 token 总数；
- `dtype_bytes` 例如 FP16/BF16 是 2 字节。

如果加上 batch / 并发维度，本质上就是所有正在服务的 sequence 的 token 数总和。

面试时要强调：

> 权重显存是固定成本，KV Cache 是动态成本；高并发和长上下文下，KV Cache 可能成为主要瓶颈。

---

## Q19：什么是 PagedAttention？它改变了 attention 的数学公式吗？

### 参考回答

PagedAttention 是 vLLM 中用于管理 KV Cache 的机制。它借鉴操作系统分页思想，把每条 sequence 逻辑上连续的 KV Cache 切成固定大小 block，并映射到物理显存中的 KV blocks。

它解决的是：

- 为每个请求按最大长度预分配会浪费显存；
- 频繁动态申请连续显存容易碎片化；
- 不同长度请求混跑时 KV Cache 难以高效管理。

PagedAttention 不改变 attention 的数学公式。Attention 仍然是 QK^T、softmax、乘 V。改变的是 K/V 在显存中的物理组织方式和 attention kernel 读取 K/V 的方式。

一句话：

> PagedAttention 不是新的注意力数学，而是 KV Cache 的分页式内存管理方案。

---

## Q20：block、block table、slot_mapping 分别是什么？

### 参考回答

- **block**：KV Cache 的物理分配单位，一个 block 存固定数量 token 的 K/V；
- **block table**：每条 sequence 的逻辑 block 到物理 KV block 的映射表；
- **slot_mapping**：本次输入 token 的 K/V 应写入 KV Cache 的具体物理 slot。

例子：

```text
block_size = 256
sequence_len = 600
需要 3 个逻辑 block
block_table = [5, 17, 23]
```

表示：

```text
逻辑 block 0 → 物理 block 5
逻辑 block 1 → 物理 block 17
逻辑 block 2 → 物理 block 23
```

`slot_mapping` 更细，它告诉 kernel 每个 token 的 K/V 写到哪个 block 的哪个 offset。

---

## Q21：sequence 数量、token 数量、KV Cache block 数量是什么关系？

### 参考回答

这三个层级不能混淆：

```text
sequence：请求执行单位
 token：模型处理和生成的基本单位
 block：KV Cache 显存分配单位
```

一条 sequence 可能包含很多 token，也可能占用多个 KV Cache block。例如：

```text
block_size = 256
sequence_len = 600
block 数 = ceil(600 / 256) = 3
```

如果有多条 sequence，总 block 数大致是：

```text
sum(ceil(sequence_len_i / block_size))
```

所以不能说“一个 sequence 对应一个 KV Cache”。更准确地说：

> 每条 sequence 逻辑上拥有自己的历史 KV Cache，但物理上由多个 block 承载。

---

## Q22：block_size 越小越好吗？为什么 vLLM / nano-vLLM 要设置 block_size？

### 参考回答

block_size 不是越小越好，它是显存利用率和管理开销之间的 trade-off。

block_size 小的优点：

- 内部浪费少；
- 对变长 sequence 更灵活；
- 请求结束后释放更细粒度。

block_size 小的缺点：

- block table 更长；
- 索引和调度开销增加；
- attention kernel 访存可能更碎；
- 管理复杂度提升。

block_size 大的优点是管理简单、索引开销低，但短请求或尾部 block 更容易浪费空间。

面试中可以回答：

> block_size 是分页粒度，决定了 KV Cache 的内部碎片和元数据/索引开销之间的平衡。

---

## Q23：Prefix Caching 是什么？它适合什么场景？

### 参考回答

Prefix caching 是对相同 prompt 前缀对应的 KV Cache block 做复用。如果多个请求共享相同 system prompt、工具描述、few-shot 示例或固定模板，那么前缀部分的 prefill 结果可以复用，不必每个请求都重新计算。

典型场景：

- 企业客服统一 system prompt；
- Agent 工具说明很长但重复；
- RAG 模板固定；
- 代码助手固定上下文模板；
- 多用户请求共享相同 few-shot 示例。

本质：

> 把重复的前缀 prefill 计算结果缓存为 KV blocks，后续请求命中时直接复用。

边界：如果请求前缀复用率低，prefix caching 收益就不明显，还可能增加哈希查找和缓存管理成本。

---

## Q24：为什么 prefix caching 通常只能复用前缀，而不是任意中间片段？

### 参考回答

自回归模型中，每个 token 的表示依赖它之前的完整上下文和位置。如果中间某段 token 内容相同，但它之前的上下文不同，那么这段 token 的 K/V 通常也不同，不能直接复用。

前缀之所以可以复用，是因为从序列开头开始，token 内容、顺序、位置和之前上下文完全一致，因此对应 K/V 可以一致。

所以 prefix caching 的关键条件是：

```text
从序列开头开始的连续 token 完全相同，并且位置也一致。
```

这也是为什么系统会对 token block 做 hash，用于判断前缀块是否命中。

---

## Q25：xxhash / hash 在 prefix caching 中有什么作用？

### 参考回答

hash 的作用是给一段 token block 生成一个快速可比较的“指纹”。

流程大致是：

```text
prompt token ids → 按 block 切分 → 对每个 block 计算 hash → 查询缓存表 → 命中则复用 KV block
```

`xxhash` 是一种速度很快的非加密哈希算法，适合做高性能查找。

注意：hash 主要用于快速定位候选缓存块。严谨实现中还要避免哈希冲突导致错误复用，通常需要结合 token 内容或其他元信息校验。

一句话：

> hash 是 token block 的身份证；prefix caching 是相同身份证的前缀 KV block 复用。

---

## Q26：KV Cache block 不够时怎么办？什么是 preemption？

### 参考回答

当高并发或长上下文导致可用 KV Cache block 不够时，推理系统不能继续给新 token 分配缓存。此时可能触发 preemption，即抢占。

典型流程：

1. 选择某些 running sequence；
2. 释放它们占用的 KV Cache block；
3. 把这些 sequence 放回 waiting 队列；
4. 等后续资源足够时重新 prefill 或恢复执行。

这类似操作系统内存不足时把一部分进程换出。代价是被抢占请求可能需要重新计算部分 KV Cache，增加延迟和计算开销。

面试中要说清楚 trade-off：

> preemption 可以避免 OOM、提高系统整体可服务性，但会牺牲被抢占请求的延迟和重复计算成本。

---

# 四、Scheduler、Continuous Batching 与 Chunked Prefill

## Q27：什么是 Continuous Batching？它和普通 batch 有什么区别？

### 参考回答

普通静态 batch 是一批请求一起开始，通常要等这一批都完成后再处理下一批。问题是请求输出长度不同，短请求先结束后，batch 里对应位置空掉，长请求继续跑，GPU 利用率下降。

Continuous Batching 是动态批处理：

```text
每一轮 step 都重新调度；
完成的 sequence 退出；
新来的 sequence 加入；
未完成的 sequence 继续 decode。
```

它的核心收益是减少静态 batch 空泡，让 GPU 尽量保持有足够 token 可处理，提高吞吐。

但它也会影响延迟，因为请求什么时候进入 batch、prefill 和 decode 如何混排，都由 Scheduler 决定。

---

## Q28：Scheduler 每一轮到底在调度什么？

### 参考回答

Scheduler 调度的不是线程，而是 sequence/token。它每一轮要决定：

- 哪些 waiting sequence 做 prefill；
- 哪些 running sequence 做 decode；
- 本轮 token budget 如何分配；
- KV Cache block 是否足够；
- 是否需要 chunked prefill；
- 是否需要 preemption；
- 是否优先保证 decode 的 inter-token latency；
- 完成请求何时释放 block。

可以用一句话概括：

> Scheduler 是把有限 GPU step、token budget 和 KV Cache 空间分配给动态请求集合的中枢。

---

## Q29：`max_num_batched_tokens` 和 `max_num_seqs` 分别限制什么？

### 参考回答

- `max_num_batched_tokens`：限制本轮调度总 token 数；
- `max_num_seqs`：限制本轮最多调度多少条 sequence。

在 Prefill 阶段，`max_num_batched_tokens` 通常限制的是所有 prompt token 加起来的数量，而不是单条请求的 prompt 长度。

例如：

```text
max_num_batched_tokens = 16384
seq A prompt 4000 tokens
seq B prompt 6000 tokens
seq C prompt 8000 tokens
```

A+B=10000 可以放进一轮，A+B+C=18000 超过预算，可能需要推迟 C 或对长 prompt 做 chunked prefill。

在 Decode 阶段，每条 sequence 通常每轮只处理一个 token，所以 decode batch 的 token 数大致接近 running sequence 数。

---

## Q30：什么是 Chunked Prefill？为什么生产级 vLLM 很重视它？

### 参考回答

Chunked prefill 是把很长的 prompt prefill 拆成多个小块分多轮处理，而不是一次占满整轮 token budget。

它解决的问题是：长 prompt prefill 计算量大，如果一次性执行，可能阻塞正在 decode 的请求，导致用户流式输出卡顿、ITL 变差。

Chunked prefill 的价值是：

- 降低长 prefill 对 decode 的阻塞；
- 让 compute-bound prefill 和 memory-bound decode 更细粒度混排；
- 改善在线服务中的交互体验；
- 在长 prompt workload 下提升调度灵活性。

一句话：

> Chunked prefill 不是简单“切小一点”，而是为了让长 prompt 不要独占本轮预算，避免拖住已经在生成的请求。

---

## Q31：为什么 vLLM V1 的调度策略通常会优先 decode？

### 参考回答

Decode 请求已经在向用户流式输出。如果 decode 长时间得不到调度，用户会感觉输出卡顿，ITL / TPOT 变差。Prefill 虽然影响 TTFT，但对已经开始输出的请求来说，decode 的连续性更直接影响交互体验。

因此生产级系统常倾向于优先调度 pending decode，再把剩余 token budget 分给 prefill，必要时把大 prefill chunk 掉。

这体现了吞吐与延迟的 trade-off：

- 优先 prefill：可能提高新请求进入速度，但会拖慢已有请求输出；
- 优先 decode：改善流式体验，但新请求 TTFT 可能受影响；
- chunked prefill：尝试在二者之间折中。

---

## Q32：为什么 batch size 不是越大越好？

### 参考回答

batch size 增大通常能提高 GPU 利用率和吞吐，但也会带来代价：

- 请求排队时间可能增加，TTFT 变高；
- KV Cache 显存占用增加，OOM / preemption 风险上升；
- 长 prompt 可能拖慢 decode，ITL 变差；
- 调度和 CPU 开销增加；
- batch 过大后 GPU 已经饱和，吞吐边际收益下降。

所以推理系统调优不是“batch 越大越好”，而是根据 workload 在吞吐、TTFT、TPOT、ITL、显存和稳定性之间取平衡。

---

## Q33：nano-vLLM 的调度相对 vLLM 有哪些简化？

### 参考回答

nano-vLLM 为了教学和源码可读性，通常会简化调度逻辑。典型简化包括：

- 调度策略更清晰，常见逻辑是 waiting 有可调度 prefill 时优先 prefill，否则 decode；
- prefill/decode 混排没有生产级 vLLM 那么复杂；
- chunked prefill、preemption、prefix cache 等实现可能更简单；
- 不包含完整 API server、多进程通信、复杂多卡拓扑和生产级 metrics；
- 对各种模型结构、采样模式、LoRA、量化、多租户的支持有限。

面试中要主动说明：

> nano-vLLM 适合学习主流程，但不能把它的简化实现当成生产级 vLLM 的完整实现。

---

## Q34：多个不同长度 prompt 如何组成一个 prefill batch？会不会相互 attention？

### 参考回答

ModelRunner 通常会把多个 prompt 的 token 拼成一个扁平的 `input_ids` tensor。例如：

```text
seq A: [A1, A2]
seq B: [B1, B2, B3]
拼接后: [A1, A2, B1, B2, B3]
```

但模型还需要知道每条序列的边界，否则 A 可能错误地 attend 到 B。因此还会准备：

- `cu_seqlens_q / cu_seqlens_k`：记录序列边界；
- `positions`：每个 token 的位置；
- `slot_mapping`：每个 token 的 KV 写入位置；
- `block_tables`：每条 sequence 的 KV block 映射。

这些 metadata 会传给 attention kernel，确保每条 sequence 只在自己的上下文内做 causal attention。

---

# 五、性能指标、Benchmark 与问题定位

## Q35：TTFT、TPOT、ITL、Throughput、QPS 分别是什么？

### 参考回答

| 指标 | 含义 | 主要受什么影响 |
|---|---|---|
| TTFT | Time To First Token，从请求进入到第一个 token 输出 | 排队时间 + prefill 时间 |
| TPOT | Time Per Output Token，平均每个输出 token 时间 | decode 性能 |
| ITL | Inter-Token Latency，相邻输出 token 间隔 | decode 调度、KV Cache 读取、kernel、batch |
| Throughput | 单位时间处理 token 数，常见 tokens/s | batch、并发、GPU 利用率、调度 |
| QPS | 每秒完成请求数 | 请求长度、输出长度、并发、调度 |

面试中要注意：吞吐和延迟不是总能同时优化。提高 throughput 的配置可能导致 TTFT 或 ITL 变差。

---

## Q36：如果线上 TTFT 很高，你会从哪些方面排查？

### 参考回答

TTFT = 排队时间 + prefill 时间 + 调度/系统开销。排查顺序可以是：

1. **请求长度分布**：prompt 是否变长，是否有大量长上下文请求；
2. **排队时间**：并发是否过高，waiting queue 是否堆积；
3. **prefill token budget**：`max_num_batched_tokens` 是否太小或被长 prompt 占满；
4. **chunked prefill**：是否启用，长 prompt 是否阻塞；
5. **prefix cache**：是否有可复用前缀，cache hit rate 是否低；
6. **KV Cache 空间**：是否频繁 preemption；
7. **CPU/Engine 开销**：tokenizer、scheduler、HTTP server 是否成为瓶颈；
8. **GPU 利用率**：prefill 是否有效吃满 GPU。

一句话：

> TTFT 高不一定是模型慢，可能是排队、长 prompt、prefill 调度、prefix cache 未命中或 CPU 侧开销导致。

---

## Q37：如果 TPOT / ITL 很差，你会从哪些方面排查？

### 参考回答

TPOT / ITL 主要对应 decode 阶段。排查方向：

1. **上下文长度**：历史 KV Cache 是否很长，导致每步读取大量 K/V；
2. **running sequence 数**：decode batch 是否太小，GPU 没吃满；
3. **KV Cache 布局**：block 分散、cache miss、访存效率是否差；
4. **调度策略**：decode 是否被大 prefill 阻塞；
5. **chunked prefill**：是否有效避免长 prompt 干扰 decode；
6. **kernel 效率**：attention decode kernel 是否适合当前 batch/context；
7. **多卡通信**：Tensor Parallel 是否引入过多 all-reduce；
8. **采样开销**：top-p、logprobs、复杂 stop 规则是否增加 CPU/GPU 开销。

一句话：

> TPOT / ITL 差通常要重点看 decode 的 KV Cache 读取、调度阻塞、batch 规模、kernel 和通信。

---

## Q38：Benchmark 推理系统时应该统计哪些指标？为什么要 warmup？

### 参考回答

常见指标包括：

- 延迟：E2E latency、TTFT、TPOT、ITL；
- 吞吐：output tokens/s、total tokens/s、requests/s；
- 显存：peak memory、KV Cache 使用量、block 使用率；
- 并发：concurrency、running / waiting 数量；
- 稳定性：OOM、超时、preemption 次数、吞吐波动；
- GPU/CPU：GPU utilization、SM occupancy、HBM 带宽、CPU 利用率。

warmup 是正式统计前先跑几轮不计入结果，目的是排除首次运行开销，例如 CUDA context 初始化、模型权重加载、显存分配、kernel 加载/编译、CUDA graph capture、缓存初始化等。

如果不 warmup，第一次请求会异常慢，污染 benchmark 结果。

---

## Q39：如何设计实验验证 Prefill 和 Decode 的性能差异？

### 参考回答

可以设计两组控制变量实验：

### 实验 1：固定输出长度，改变输入长度

- 输入长度：128、512、2048、8192 tokens；
- 输出长度固定：例如 64 tokens；
- 观察：TTFT、prefill time、初始 KV Cache 占用。

预期：输入越长，Prefill 越重，TTFT 越高。

### 实验 2：固定输入长度，改变输出长度

- 输入长度固定：例如 512 tokens；
- 输出长度：32、128、512、1024 tokens；
- 观察：TPOT、总 decode 时间、KV Cache 增长。

预期：输出越长，decode step 越多，总延迟越高，KV Cache 持续增长。

如果并发增加，还可以观察 throughput 先升后趋于饱和，TTFT 可能因排队变高。

---

## Q40：最大上下文长度越大，回答一定越好吗？对推理系统有什么影响？

### 参考回答

不一定。最大上下文长度大，只表示模型一次推理能容纳更多 token，不代表回答一定更准确。

好处：

- 能处理长文档；
- 能保留更多历史对话；
- RAG 可以放更多检索片段；
- 复杂任务可携带更多约束。

代价：

- Prefill 更慢，TTFT 增加；
- KV Cache 更大，显存压力上升；
- Decode 每步读取更长历史 K/V，TPOT/ITL 可能变差；
- 并发能力下降；
- 无关上下文可能干扰模型。

面试中要强调：

> 大上下文是容量能力，不是免费能力；它会直接改变系统的显存、带宽和调度压力。

---

## Q41：推理阶段 GPU 显存主要由哪些部分组成？什么最吃显存？

### 参考回答

推理显存大致包括：

```text
模型权重显存
KV Cache 显存
中间 activation / temporary buffer
logits / sampling buffer
CUDA context / framework reserve
通信 buffer / NCCL buffer
显存碎片和 allocator 预留
```

权重显存是固定成本，例如 FP16/BF16 下 7B 模型权重大约 14GB。KV Cache 是动态成本，随并发 sequence 数、上下文长度、生成长度、层数、KV heads、head_dim 和 dtype 增长。

低并发短上下文时，权重通常最吃显存；高并发长上下文时，KV Cache 可能成为主要瓶颈。量化模型会降低权重显存，但不一定降低 KV Cache，因此量化后 KV Cache 占比可能更突出。

---

# 六、源码阅读与 Python 工程问题

## Q42：`@dataclass(slots=True)` 在 nano-vLLM 这类项目里有什么价值？

### 参考回答

`@dataclass` 适合定义参数类或状态类，可以自动生成 `__init__`、`__repr__`、`__eq__` 等方法，减少模板代码。例如 `SamplingParams` 可以保存 `temperature`、`max_tokens`、`ignore_eos` 等采样配置。

`slots=True` 的价值是：

- 限制对象只能拥有预声明字段，防止误写属性；
- 减少每个对象的内存开销；
- 对大量小对象更友好；
- 让配置类更像固定结构的数据容器。

在推理引擎中，Sequence、SamplingParams、Config 等对象可能很多，轻量化和可读性都很重要。

---

## Q43：`__post_init__` 和 `assert` 在参数校验中有什么作用？有什么风险？

### 参考回答

`__post_init__` 是 dataclass 自动初始化字段后执行的补充初始化函数，常用于参数校验、派生字段计算、类型转换等。

例如：

```python
assert self.temperature > 1e-10, "greedy sampling is not permitted"
```

表示 SamplingParams 创建完成后检查 temperature 是否太小。如果不满足条件，就抛出 AssertionError。

但 `assert` 更适合开发期验证程序假设，不适合严肃的用户参数校验。因为 Python 在 `-O` 优化模式下可能忽略 assert。生产代码中更稳妥的写法是显式抛异常：

```python
if self.temperature <= 1e-10:
    raise ValueError("greedy sampling is not permitted")
```

面试中这样回答能体现你不仅看懂语法，也知道工程风险。

---

## Q44：为什么 Sequence 对象可以放进 deque？队列里不是只能放数字吗？

### 参考回答

Python 容器可以存放任意对象，不只数字。`deque.append(seq)` 是把一个 Sequence 对象作为整体加入队列。

推理引擎调度器管理的是请求状态对象，而不是简单 token id。每个 Sequence 内部保存 token_ids、status、sampling_params、block_table、生成进度等信息。

因此 waiting 队列更像：

```python
deque([seq1, seq2, seq3])
```

Scheduler 从队列中取出 sequence 后，可以直接访问和修改它的状态。

这体现了源码阅读中的一个重点：

> 推理系统调度的是带状态的对象，不是裸数据。

---

## Q45：如果面试官让你现场讲 nano-vLLM 源码阅读顺序，你会怎么说？

### 参考回答

我会按主流程自上而下读，而不是从某个 CUDA kernel 或模型层细节开始。

推荐顺序：

1. 看入口：`LLM` / `LLMEngine.generate()`；
2. 看配置：`Config`、`SamplingParams`；
3. 看请求状态：`Sequence`，理解 token_ids、status、block_table；
4. 看调度：`Scheduler.add()`、`Scheduler.schedule()`、`Scheduler.postprocess()`；
5. 看显存管理：`BlockManager` 如何 allocate/free、维护 block table、prefix cache；
6. 看执行层：`ModelRunner` 如何准备 input_ids、positions、slot_mapping、block_tables；
7. 看模型：Qwen / Transformer forward，区分 prefill 和 decode；
8. 看 sampler：logits 如何通过 temperature / softmax / sampling 变成 next token；
9. 最后把输出 token decode 成文本。

这样读的好处是，先建立“请求如何流动”的系统图，再进入局部实现细节，避免只看懂函数却不知道它在系统中起什么作用。

---

# 七、vLLM 相比 nano-vLLM 的生产级扩展

## Q46：vLLM 相比 nano-vLLM，生产级系统里更重要的功能有哪些？

### 参考回答

可以从六个维度回答：

1. **服务入口层**：OpenAI-compatible API Server、流式输出、请求取消、认证、metrics；
2. **进程架构层**：API Server、Engine、Executor、Worker 之间解耦，使用 IPC / socket / RPC 通信；
3. **调度层**：更复杂的 unified scheduler、decode 优先、chunked prefill、preemption、多队列策略；
4. **显存层**：更完整的 paged KV cache、prefix caching、KV transfer、分布式 KV 管理；
5. **执行层**：多卡 tensor parallel、pipeline parallel、expert parallel、CUDA graph、优化 attention kernel；
6. **模型与功能层**：量化、LoRA、多模态、speculative decoding、logprobs、guided decoding、结构化输出等。

nano-vLLM 的价值是帮你看懂骨架；vLLM 的难点是把这个骨架扩展成高并发、可观测、可部署、可扩展的生产系统。

---

## Q47：vLLM 中 Engine、Executor、Worker、ModelRunner 的层次关系怎么理解？

### 参考回答

可以用分层方式理解：

```text
API Server：接收 HTTP / OpenAI-compatible 请求
Engine：请求入口、调度和结果出口
Executor：执行组织层，管理一个或多个 Worker
Worker：通常绑定具体 GPU / rank，持有模型分片或执行上下文
ModelRunner：把调度结果转换成 tensor，调用模型 forward
Model：Transformer 计算本体
```

nano-vLLM 往往会把很多层合并或简化，所以看起来像 LLMEngine 直接协调 Scheduler 和 ModelRunner。而生产级 vLLM 为了多进程、多卡、异步服务、故障隔离和扩展性，会把这些职责拆开。

面试中要强调：

> Engine 更像控制平面，Worker / ModelRunner 更像执行平面。

---

## Q48：rank 是什么意思？为什么多卡推理中会说 worker 0 / rank 0？

### 参考回答

rank 是分布式系统中进程或设备的编号。在多 GPU 推理中，每个 worker 进程通常绑定一个 GPU，并拥有一个 rank。

例如 tensor parallel size = 2：

```text
rank 0 / worker 0：持有模型某些权重分片
rank 1 / worker 1：持有模型另一些权重分片
```

它们协同执行同一个模型层的计算，并通过 NCCL 等通信库交换中间结果。

rank 的作用是让每个进程知道：

- 自己是谁；
- 应该加载哪部分权重；
- 和哪些进程通信；
- 在 collective communication 中处于什么位置。

---

## Q49：Tensor Parallel 对推理性能有什么收益和代价？

### 参考回答

Tensor Parallel 是把模型层内的大矩阵计算切分到多张 GPU 上执行。收益是：

- 单卡放不下的大模型可以多卡加载；
- 大矩阵计算可以并行；
- 单请求或大 batch 下可能提高吞吐。

代价是：

- 每层可能需要 all-reduce / all-gather 等通信；
- batch 小或 decode 阶段计算量小的时候，通信开销可能占比很高；
- 多卡同步增加复杂度；
- GPU 间带宽和拓扑会明显影响性能。

所以 TP 不是越大越好。模型越大、单卡越放不下、计算量越高时 TP 更有必要；如果模型能单卡放下且 decode batch 较小，TP 可能因为通信开销导致收益有限。

---

## Q50：如果面试官问“你没真正改过 vLLM 源码，凭什么说你懂 vLLM”，你怎么回答？

### 参考回答

可以坦诚但不自降价值：

> 我不会把自己包装成 vLLM 核心贡献者。我目前的重点是通过 nano-vLLM 建立推理引擎主流程认知，再对照 vLLM 学习生产级系统扩展。我理解的重点不是只会调用 API，而是知道 vLLM 为什么要围绕 KV Cache、PagedAttention、Scheduler、Continuous Batching、Prefix Caching、Chunked Prefill 来设计，也能说明这些机制解决什么问题、有什么代价、在什么 workload 下有效。

然后补一句项目化表达：

> 如果后续做工程贡献，我会优先从文档、benchmark、scheduler 配置实验、prefix cache / chunked prefill 的性能分析、小 bugfix 或测试用例入手，而不是一上来改核心 kernel。

这种回答真实、可信，也能体现你有系统理解和工程落点。

---

# 八、最后的口头表达模板

## 1 分钟版：介绍你对 vLLM / nano-vLLM 的理解

我把 vLLM 理解成一个面向在线 LLM serving 的推理系统，而不只是一个模型调用库。在线推理的难点是多用户请求动态到达、prompt 和输出长度不同、KV Cache 随上下文动态增长，如果只是用普通 `generate()` 或静态 batch，很容易出现显存浪费、碎片化和 GPU 空泡。vLLM 的核心是用 paged KV cache / PagedAttention 管理 KV Cache，用 continuous batching 和 scheduler 动态组织请求，并结合 prefix caching、chunked prefill 等机制在吞吐和延迟之间做权衡。我学习 nano-vLLM 的价值是先把这条主链路看清楚：prompt 变成 sequence，scheduler 决定每轮执行对象，BlockManager 分配 KV block，ModelRunner 执行 prefill/decode，sampler 生成 token，最后释放资源。

## 3 分钟版：如果面试官让你展开讲 vLLM 为什么快

vLLM 快不是单靠某一个 kernel，而是多个系统机制叠加。首先，LLM 推理中 KV Cache 是核心资源，随着并发和上下文长度动态增长。传统连续分配或最大长度预分配会造成碎片和浪费。vLLM 用 PagedAttention，把 KV Cache 切成固定大小 block，通过 block table 实现逻辑连续、物理非连续的管理，提高显存利用率。其次，在线服务中请求不是静态 batch，而是不断进入和结束，所以 vLLM 用 continuous batching，让每个 step 动态调整 batch，减少短请求结束后的 GPU 空泡。第三，Prefill 和 Decode 性能特征不同，Prefill 更偏 compute-bound，Decode 更偏 memory-bound。为了避免长 prompt prefill 阻塞 decode，vLLM 引入 chunked prefill 和 decode 优先调度。第四，在大量共享 system prompt 或模板场景下，prefix caching 可以复用前缀 KV blocks，减少重复 prefill。综合来看，vLLM 的核心是把在线 LLM 推理变成围绕 token budget、KV Cache、动态请求集合和 GPU 执行效率的资源编排系统。

---

# 九、参考资料

## 你的自制学习笔记

- `nano-vLLM基础概念问答-2026-06-12.md`
- `nano-vLLM基础概念问答-2026-06-14.md`
- `nanovllm_推理全过程.md`
- `主题一_nano-vllm 推理全流程串讲_概览篇__哔哩哔哩_bilibili_学习笔记.md`
- `源码代码问题问答-2026-06-13.md`
- `源码代码问题问答-2026-06-14.md`
- `源码代码问题问答-2026-06-17.md`

## 公开资料与面试高频方向参考

- vLLM 官方文档：Optimization and Tuning / Chunked Prefill  
  https://docs.vllm.ai/en/stable/configuration/optimization/
- vLLM 官方文档：Paged Attention  
  https://docs.vllm.ai/en/latest/design/paged_attention/
- vLLM 论文：Efficient Memory Management for Large Language Model Serving with PagedAttention  
  https://arxiv.org/abs/2309.06180
- vLLM Blog：Inside vLLM: Anatomy of a High-Throughput LLM Inference System  
  https://vllm.ai/blog/2025-09-05-anatomy-of-vllm
- BentoML LLM Inference Handbook：Prefill-decode disaggregation  
  https://bentoml.com/llm/inference-optimization/prefill-decode-disaggregation
- AI Infra / vLLM 面试公开资料方向：KV Cache、PagedAttention、Continuous Batching、Prefix Caching、Chunked Prefill、吞吐/延迟 trade-off、性能定位。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]

%% 项目关联导航：结束 %%
