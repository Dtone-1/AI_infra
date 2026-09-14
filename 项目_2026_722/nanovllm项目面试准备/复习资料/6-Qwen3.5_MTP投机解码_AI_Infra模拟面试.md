# nano-vLLM Qwen3.5 MTP 投机解码：AI Infra 模拟技术面试

> 面试背景：候选人在 nano-vLLM 上适配 Qwen3.5 Hybrid 架构，并实现了 MTP 投机解码原型。以下问题按真实面试中的自然追问顺序组织，重点考察候选人是否真正理解“主模型生成、MTP Draft、Target Verify、Accept/Reject、状态回滚、Tensor Parallel 采样和性能收益”这一完整链路。

## 项目边界说明

这部分工作更准确的定位是：

> **完成 Qwen3.5 MTP 投机解码原型，打通 Main/Draft 生成、Target Verify、Accept/Reject、KV Cache 与 GDN State 回滚，并适配 Tensor Parallel 下的全局采样。**

当前实现属于可验证的工程原型，而不是已经接入默认生成主循环的生产级 Speculative Decoding Serving 系统。现有路径主要面向单请求、纯文本和 Greedy 解码；Chunk Verify 仍处于实验验证阶段。

---

# 模拟技术面试

## 问题 1：请介绍一下你在 Qwen3.5 MTP 部分的工作。

### 参考回答

我在这部分做的核心工作，是在 Qwen3.5 Hybrid 推理引擎中打通一套 MTP 投机解码原型。

整体流程是，先让完整主模型正常执行 Prefill 或 Decode，得到当前上下文对应的 `main hidden`，再经过共享的 LM Head 和采样器生成一个主模型确认的 Main Token。随后，MTP 使用这个 `main hidden` 和 Main Token 的 embedding 作为输入，预测后续 Draft Token。

如果 Draft 长度是 3，当前实现不是一次前向同时输出 3 个 token，而是连续执行 3 次较轻量的 MTP Forward：上一步的 MTP Hidden 和上一步 Draft Token 的 embedding，会作为下一步输入，递推生成多个候选 token。

候选生成后，再由完整主模型进行 Target Verify，比较 Draft 和主模型真正会输出的 Token，只接受最长连续匹配前缀。如果出现 Reject，就同时恢复调度器状态、KV Cache 和 GDN 的卷积及循环状态，再重新执行正确路径。

另外，因为项目支持 Tensor Parallel，主模型、MTP 和 Verify 得到的都是词表分片上的 Local Logits，所以我还改造了输出采样链路：每个 Rank 先选本地候选 Token 和 Score，再跨 Rank 选择全局 Token，并广播给所有 Rank，保证下一步 MTP 递推一致。

---

## 问题 2：你刚才一直提到 MTP 和 Speculative Decoding，这两个概念是一回事吗？

### 参考回答

不是一回事。

MTP 在这个项目中主要是候选预测模块。它根据主模型提供的上下文 Hidden 和当前 Token Embedding，生成一个新的 MTP Hidden，再通过共享 LM Head 得到 Draft Token。它解决的是“怎么低成本猜后续 Token”。

Speculative Decoding 则是一整套执行协议，除了 Draft 生成，还包括目标模型验证、最长前缀接受、拒绝后的状态恢复、正确 Token 提交以及下一轮执行。

可以概括为：

```text
MTP：负责猜。
Target Model：负责验证。
Speculative Decoding：负责把猜测、验证、提交和回滚组织成正确的生成流程。
```

仅仅实现一个 MTP 模块，并不等于完成了投机解码。真正复杂的部分反而在执行层，因为需要保证 Draft 不会污染真实请求状态，Reject 后也能恢复到与普通 Greedy Decode 完全一致的位置。

---

## 问题 3：从 Prompt 开始，Main Token 和第一个 Draft Token 是怎么生成的？

### 参考回答

先说 Main Token。

Prompt 经过 Tokenizer 后形成 Token IDs，主模型执行完整 Prefill。这个过程会让 Full Attention 层写入 KV Cache，同时让 Gated DeltaNet 层更新卷积状态和循环状态。

主模型经过所有 Decoder Layer 和最终归一化后，会得到每个 Prompt 位置的 Hidden States。系统取最后一个有效 Prompt 位置的 Hidden，作为 `main hidden`。它还没有进入 LM Head，而是主模型理解完整上下文后的最终隐藏表示。

然后：

```text
main hidden
→ LM Head
→ logits
→ Sampler
→ Main Token
```

得到 Main Token 后，系统会用主模型的 Embedding Table 查询这个 Token 对应的 Token Embedding，再和 `main hidden` 一起输入 MTP：

```text
main hidden + embedding(Main Token)
→ MTP
→ mtp hidden
→ 共享 LM Head
→ Draft Token 1
```

这里 Main Token 是完整目标模型确认的结果，而 Draft Token 只是辅助分支给出的候选，后续必须验证。

---

## 问题 4：`main hidden` 和 Token Embedding 有什么区别？为什么 MTP 两个都要？

### 参考回答

两者表达的信息完全不同。

`main hidden` 来自完整主模型，已经经过所有 Decoder Layer 和 Final Norm，包含整个历史上下文的信息。可以把它理解成：“模型读完前文后，当前对上下文形成了什么高级语义表示。”

Token Embedding 则只是用刚生成的 Token ID 去 Embedding Table 查出的一行初始向量。它表达的是：“刚刚生成的具体 Token 是什么。”它本身没有经过完整主模型，也不包含整段上下文。

之所以两个都需要，是因为 MTP 要同时知道：

```text
前文整体语义是什么；
主模型刚刚选择了哪个 Token。
```

实际结构中，会先分别对两个向量做归一化，再把它们拼接成 `2H` 维向量，通过线性层压回 `H`，然后进入少量 MTP Decoder Layer。

还要注意一个时间关系：`main hidden` 是用于预测 Main Token 的前一个位置 Hidden，而 Main Token 是由它经过 LM Head 选出来的。MTP 把“上一个位置的上下文状态”和“刚生成的 Token 内容”结合起来，近似推演下一个位置的 Hidden。

---

## 问题 5：如果配置生成 3 个 Draft Token，MTP 是一次输出 3 个，还是怎么生成的？

### 参考回答

当前项目中是递推生成，不是一次 Forward 并行输出 3 个位置。

假设 Draft Length 是 3，流程是：

```text
main hidden + embedding(Main Token)
→ MTP Step 1
→ mtp hidden 1
→ Draft 1

mtp hidden 1 + embedding(Draft 1)
→ MTP Step 2
→ mtp hidden 2
→ Draft 2

mtp hidden 2 + embedding(Draft 2)
→ MTP Step 3
→ mtp hidden 3
→ Draft 3
```

因此，生成 3 个 Draft 需要 3 次 MTP Forward。

MTP 的 Decoder Layer 也不是只有 Attention，它仍然是一个精简 Transformer Layer，包含归一化、Full Attention、MLP 和残差结构。它比完整主模型轻，主要是因为层数少，而且不走主模型的 Full Attention 与 GDN 混合层选择。

当前 MTP Draft 路径还做了一个重要隔离：它不会读写主模型真实的 KV Cache，也不会更新 GDN State。Draft 只是候选，不应该在验证前推进真实模型状态。

---
## 问题 6：Draft Token 生成后，目标模型具体怎么验证？为什么验证输入会有一个位置偏移？

### 参考回答

因为自回归模型的语义是“输入当前位置 Token，预测下一个 Token”，所以验证时存在一个 One-Token Shift。

假设主模型先生成：

```text
Main = M
```

MTP 生成：

```text
Draft = [D1, D2, D3]
```

为了验证这三个 Draft，目标模型的输入应当是：

```text
[M, D1, D2]
```

目标模型分别预测：

```text
输入 M  → Target 1
输入 D1 → Target 2
输入 D2 → Target 3
```

再比较：

```text
D1 是否等于 Target 1
D2 是否等于 Target 2
D3 是否等于 Target 3
```

假设：

```text
Draft  = [量, 并, 行]
Target = [量, 并, 化]
```

那么前两个匹配，第三个不匹配，`accept length` 就是 2。最终提交的是：

```text
已经确认的 Main Token
+ 两个被接受的 Draft
+ 第一个不匹配位置上目标模型给出的正确 Token
```

一旦出现第一个不匹配，后面的 Draft 即使碰巧相同也不能继续接受，因为它们建立在错误前缀上。

---

## 问题 7：为什么 Reject 时需要同时恢复 Scheduler、KV Cache 和 GDN State？只删除错误 Token 不行吗？

### 参考回答

不行，因为一次 Target Verify 不只是临时算出 Logits，它会真实推进模型和推理引擎的状态。

首先是控制面状态，包括 Sequence 中已经记录的 Token、请求状态、Block Table、Scheduler 的 Waiting/Running 队列、KV Block 的占用和 State Slot 的分配。

其次是 GPU 数据面状态。Full Attention 层会把 Verify 路径写入 KV Cache；GDN 层会递推更新 Convolution State 和 Recurrent State。

如果只删除 Sequence 里的错误 Token，GPU 上的 KV 和 GDN State 仍然保留错误路径，下一轮 Decode 会从被污染的状态继续执行。

反过来，只恢复 GPU Tensor 也不够，因为 Scheduler 和 Block Manager 仍可能认为那些位置已经提交或资源已经占用。

所以 Reject 必须同时恢复：

```text
控制面：
Sequence、Scheduler、Block Manager、State Slot Manager。

数据面：
被覆盖的 KV Slots、各 GDN Layer 的 Conv State 和 Recurrent State。
```

恢复后，再从 Verify 前的正确状态重新执行“最后一个已提交 Token加已接受 Draft 前缀”，最终提交目标模型给出的正确 Token。这也是 Hybrid 模型投机解码比纯 Transformer 更复杂的地方。

---

## 问题 8：你实现了 Eager、CUDA Graph 和 Chunk 三种 Verify，它们有什么区别？哪一种才真正可能加速？

### 参考回答

三种路径解决的问题不同。

### Eager Verify

Eager 路径会对每个待验证位置逐 Token 调用一次完整主模型 Decode。语义最直接，也最适合做正确性基线，但它没有减少目标模型 Forward 次数。

如果有 3 个 Draft，就仍然需要 3 次完整主模型验证。因此它通常不会带来真正加速，反而增加了 MTP 和状态快照的额外成本。

### CUDA Graph Verify

Graph 路径把多个顺序 Decode Step 捕获到一个 CUDA Graph 中。它内部依然是多个目标模型 Step，只是减少了 Python 调度、CPU 到 GPU 的 Kernel Launch 和固定输入准备开销。

所以它优化的是 Launch Overhead，不是减少目标模型主体计算量。

### Chunk Verify

Chunk 路径尝试把：

```text
[Main, Draft1, Draft2]
```

作为一个多 Token Causal Chunk，一次主模型 Forward 同时得到多个验证结果。它最接近投机解码理论上的收益来源，因为有机会把多次单 Token Decode 合并为一次 Multi-Token Forward。

但当前主模型是 Full Attention 和 GDN 的 Hybrid 架构。Attention 可以用 Causal Mask 处理 Chunk，而 GDN 的 Conv/Recurrent State 必须严格按 Token 顺序递推。要证明 Chunk 与逐 Token Decode 的状态更新完全一致比较困难。

所以当前 Chunk Verify 仍是实验路径，需要恢复状态后再用可信的 Eager 或 Graph 路径复核 Token 和 Logits，不能把它直接描述成已经成熟的加速实现。

---

## 问题 9：MTP 理论上为什么能加速？你这个项目当前已经证明加速了吗？

### 参考回答

理论收益来自两个条件同时成立。

第一，MTP 比完整主模型轻，可以低成本预测多个 Draft。

第二，目标模型能够用一次高效的 Multi-Token Forward 验证多个 Draft，而不是仍然逐个 Token 验证。

普通自回归生成 4 个 Token，大致需要 4 次完整主模型执行。理想的 MTP 路径可能是：

```text
1 次主模型执行生成 Main
+ 3 次轻量 MTP Forward 生成 Draft
+ 1 次主模型 Chunk Forward 验证多个 Draft
```

如果接受率高，就可能用大约 2 次目标模型执行推进多个 Token。

但是当前项目不能简单宣称已经实现稳定加速。因为 Eager Verify 不减少目标模型 Forward，Graph Verify主要减少 Launch 开销，而 Chunk Verify 由于 GDN 状态一致性问题仍需要可信路径复核。

另外还存在 MTP Forward、TP 通信、状态快照、Rollback 和 Reject Rerun 等额外成本。

所以我会通过以下指标判断是否真正加速：

```text
Draft Accept Rate
平均接受长度
Target Forward / Token
MTP Forward / Token
Reject Rerun 次数
最终 Decode Tokens/s
与 Greedy Baseline 的 Token 一致性
```

项目当前更准确的成果，是完成了正确性和性能评估原型，而不是已经获得经过充分证明的生产级加速。

---

## 问题 10：Tensor Parallel 下，主模型和 MTP 的 Token 是怎么在多个 Rank 之间正确选出来的？当前方案还有什么限制？

### 参考回答

在 Tensor Parallel 下，词表权重会按 Rank 切分，所以每个 Rank 的 LM Head 只能得到自己的 Local Logits。如果每次都把完整词表 Logits Gather 到 Rank 0，通信量比较大，而且主模型、MTP 和 Verify 都会重复走这条链路。

我的改造是让每个 Rank 先在自己的词表分片中选一个本地候选 Token 和对应 Score，然后把少量的 Token ID 和 Score 做跨 Rank 汇聚。Rank 0 比较各个 Rank 的 Score，选出全局赢家。

对于 Greedy，这和在完整词表上直接做 Argmax 完全等价，因为全局最大值一定是某个分片的局部最大值。

选出的 Token 还要立即 Broadcast 给所有 Rank。原因是同一次 MTP 调用中，所有 Rank 接下来都要用同一个 Token 去查 Embedding，并继续下一次 MTP Forward；如果各 Rank Token 不一致，后续通信顺序和模型计算都会出错。

这套方案的优点是，普通采样只需要交换每个请求、每个 Rank 的 Token 和 Score，而不是完整 Vocabulary Logits。

它当前也有限制：主要适合 Greedy 和温度采样。Top-p、规范化 Logprobs、复杂重复惩罚和严格跨 TP 配置随机可复现，还需要更多全局统计或候选汇聚逻辑。

此外，当前 MTP 原型主要限制在单请求、纯文本和 Greedy 模式，还没有完整接入 Continuous Batching 和默认 Engine 生成主循环。这些是后续要解决的工程问题。

---

# 面试中的完整口述主线

```text
Prompt
  ↓
完整 Qwen3.5 Hybrid 主模型 Prefill / Decode
  ↓
main hidden
  ↓
共享 LM Head + TP 全局采样
  ↓
Main Token
  ↓
main hidden + Main Token Embedding
  ↓
MTP 递推生成多个 Draft Token
  ↓
保存 Scheduler、KV Cache 和 GDN State
  ↓
Target Model Verify
  ↓
计算最长连续接受前缀
  ↓
全部匹配：直接提交
中途拒绝：恢复状态并重新执行正确前缀
  ↓
从最后一个已提交 Token 重新进入完整主模型
```

---

# 面试中容易说错的地方

1. **不要说 MTP 一次 Forward 同时生成多个 Token。** 当前实现是执行多次 MTP Forward，逐个递推生成 Draft。
2. **不要说 MTP 只有 Attention。** MTP Decoder Layer 仍包含 Full Attention、MLP、Norm 和残差结构，只是层数远少于完整主模型。
3. **不要说 `main hidden` 是 Main Token 已经经过主模型后的 Hidden。** 它是用于预测 Main Token 的前一个位置 Hidden，位于 Final Norm 之后、LM Head 之前。
4. **不要说 Draft 可以直接提交。** Draft 必须经过完整目标模型 Verify，只能接受最长连续匹配前缀。
5. **不要说 Eager Verify 已经带来投机解码加速。** Eager 仍逐 Token 执行完整主模型，主要用于正确性验证。
6. **不要说 CUDA Graph 减少了目标模型计算步数。** 它主要降低 Python 调度和 Kernel Launch 开销，内部仍是顺序 Decode Step。
7. **不要说 Chunk Verify 已经成熟可用。** Hybrid GDN 状态的 Chunk 等价性仍需可信路径复核。
8. **不要只恢复 Sequence Token。** Reject 后必须同时恢复控制面状态以及 KV Cache、GDN Conv/Recurrent State。
9. **不要把当前实现描述成生产级投机解码 Serving。** 当前主要是单请求、Text-only、Greedy 的正确性和性能原型。
10. **不要只看 Accept Rate 判断加速。** 还要综合 Target Forward/Token、MTP Forward/Token、Rollback、TP 通信和最终 Tokens/s。

---

# 一句话总结项目价值

> **在 Qwen3.5 Hybrid 推理引擎中建立了从 MTP 候选生成、TP 全局采样、目标模型验证到 KV/GDN 状态回滚的完整投机解码实验闭环，并明确验证了正确性路径与潜在加速路径的边界。**


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nanovllm项目面试准备|模块-nanovllm项目面试准备]]

%% 项目关联导航：结束 %%
