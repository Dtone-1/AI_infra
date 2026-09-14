# Qwen3 / nano-vLLM 模型结构面试题 30+ 问

> 基于以下学习文件整理：  
> 1. `Qwen3_DecoderOnly架构与nano-vLLM模型代码对应总览.md`  
> 2. `Qwen3_MLP_FFN_gate_up_down线性层详解.md`  
> 3. `Qwen3_RMSNorm_LayerNorm与残差连接详解.md`  
>
> 使用方式：先背熟“完整 forward 流程回答”，再逐题练习。题目尽量贴近 AI Infra / 推理框架 / vLLM / nano-vLLM 相关面试中常见的追问方式。

---

## 0. 开场必会题：请完整讲一下 Qwen3 dense 的一次 forward 流程

- Qwen3 dense 是一个 decoder-only 自回归语言模型，一次 forward 大致是：输入文本先经过 tokenizer 变成 token ids，再通过 embedding 变成 hidden states；随后 hidden states 进入多层 decoder layer，每层先做 RMSNorm 稳定数值，再通过 QKV projection 得到 Q、K、V，对 Q/K 加 RoPE 位置信息后做 causal self-attention，让每个 token 只能看自己和之前的 token，并通过 residual 保留原始主干信息；attention 输出后再经过 RMSNorm 和 SwiGLU MLP，其中 MLP 通过 gate/up/down 三个投影对每个 token 的表示做非线性特征加工，输出再通过残差连接回到主干；所有层结束后做 final RMSNorm，最后经过 LM Head 把 hidden states 映射成词表大小的 logits，用 logits 采样得到下一个 token。推理时 prefill 阶段一次处理 prompt 并建立 KV Cache，decode 阶段每次只输入新 token，并复用历史 KV Cache 继续生成。
- Qwen3 dense 是一个仅解码器结构的自回归语言模型，一次前向传播大致是：输入文本先经过分词器变成 token 编号，再通过词嵌入层变成隐藏状态；随后隐藏状态进入多层解码器层，每层先做 RMS 归一化来稳定数值，再通过 QKV 线性投影得到查询 Q、键 K、值 V，对 Q/K 加入旋转位置编码后做因果自注意力，让每个 token 只能看自己和之前的 token，并通过残差连接保留原始主干信息；注意力输出后再经过 RMS 归一化和 SwiGLU 前馈网络，其中前馈网络通过门控投影、升维投影、降维投影对每个 token 的表示做非线性特征加工，输出再通过残差连接回到主干；所有层结束后做最终 RMS 归一化，最后经过语言模型头把隐藏状态映射成词表大小的 logits 分数，用这些分数采样得到下一个 token。推理时，预填充阶段一次处理 prompt 并建立 KV 缓存，解码阶段每次只输入新 token，并复用历史 KV 缓存继续生成。

如果面试官问：

> 请你完整讲一下 Qwen3 dense 模型一次 forward 是怎么走的，最好结合 nano-vLLM 代码说一下。

可以这样回答：

Qwen3 dense 是一个 decoder-only causal language model，一次 forward 可以从 `ModelRunner.run()` 开始讲。nano-vLLM 不是直接把 `[batch, seq_len]` 喂给模型，而是根据当前阶段先准备输入。如果是 prefill，`ModelRunner.prepare_prefill()` 会把多个请求的 prompt token 拼成一维 `input_ids`，同时准备 `positions`、`cu_seqlens_q/k`、`slot_mapping`、`block_tables` 等信息；如果是 decode，`prepare_decode()` 会为每个仍在生成的请求取当前最后一个 token，所以 `input_ids` 形状通常是 `[batch_size]`。

接着 `ModelRunner` 会通过 `set_context()` 设置全局 `Context`，告诉底层 attention 当前是 prefill 还是 decode，以及 KV Cache 该如何写入或读取。然后进入 `run_model()`，调用：

```python
self.model.compute_logits(self.model(input_ids, positions))
```

这里 `self.model` 是 `Qwen3ForCausalLM`。它的 `forward()` 不直接返回 logits，而是先调用 `Qwen3Model.forward()` 返回 final hidden states。`Qwen3Model` 先用 `VocabParallelEmbedding` 把 `input_ids` 查表变成 `hidden_states`，形状从 `[T]` 变成 `[T, hidden_size]`。然后依次经过 `num_hidden_layers` 个 `Qwen3DecoderLayer`。

每个 decoder layer 是 pre-norm 结构，包含 attention 和 MLP。进入一层时，先做 `input_layernorm`。如果有 residual，`RMSNorm(x, residual)` 会把当前子层输出加到 residual 上，再做 RMSNorm；如果是第一层，先对 embedding 输出做普通 RMSNorm，并把 embedding 作为 residual 主干。接着进入 `Qwen3Attention`。

在 attention 中，先通过 `QKVParallelLinear` 一次性做 QKV projection，把 `hidden_states` 投影成拼接在一起的 q/k/v。然后通过 `split` 把 q、k、v 切开，再通过 `view` 把二维张量改成多头格式，比如 q 从 `[T, num_heads * head_dim]` 变成 `[T, num_heads, head_dim]`。随后对 q/k 做 RMSNorm 和 RoPE，注入位置信息。真正 attention 计算在 `layers/attention.py` 里：它会先按照 `slot_mapping` 把当前 token 的 k/v 写入 KV Cache；如果是 prefill，就调用 `flash_attn_varlen_func` 处理变长 prompt；如果是 decode，就调用 `flash_attn_with_kvcache`，用当前 q 读取历史 KV Cache 做 attention。attention 输出再经过 `o_proj` 回到 `[T, hidden_size]`。

Attention 后，代码执行 `post_attention_layernorm(hidden_states, residual)`，这一步等价于把 attention 输出加回 residual，再做 RMSNorm，作为 MLP 输入。MLP 使用 SwiGLU 结构：源码中 `gate_proj` 和 `up_proj` 被合并成 `gate_up_proj`，一次线性层输出 `[T, 2 * intermediate_size]`，再由 `SiluAndMul` 切成 gate 和 up，计算 `SiLU(gate) * up`，最后通过 `down_proj` 把中间维度压回 `hidden_size`。这个 MLP 输出不会在当前层末尾立刻加回 residual，而是在下一层开头的 fused add RMSNorm 中加回；最后一层的 MLP 输出则在 `Qwen3Model.norm(hidden_states, residual)` 这个 final RMSNorm 中加回。

所有 decoder layer 结束后，final RMSNorm 得到最终 hidden states。然后 `compute_logits()` 调用 `ParallelLMHead`，把 hidden states 映射成 vocab logits。prefill 阶段只取每个请求最后一个 token 的 hidden state 计算 logits，decode 阶段每个请求本来就只有一个 token。最后 `Sampler` 对 logits 做 temperature、softmax 和采样，得到下一步 token id。

一句话总结：

```text
ModelRunner 准备 token 和 Context
  -> Qwen3ForCausalLM / Qwen3Model
  -> Embedding
  -> 多层 DecoderLayer
  -> 每层 RMSNorm + QKV + FlashAttention/KV Cache + MLP/SwiGLU
  -> Final RMSNorm
  -> LM Head
  -> Sampler
  -> next token
```

---

## 1. Qwen3 dense 为什么叫 decoder-only causal LM？

**参考回答：**

Qwen3 dense 是 decoder-only，因为它只有 Transformer decoder block，没有 encoder-decoder 结构。它是 causal LM，因为生成时每个位置只能看到自己和之前的 token，不能看到未来 token。这个 causal 约束体现在 attention kernel 里 `causal=True`，也体现在自回归生成流程里：每次根据历史上下文预测下一个 token。

**追问方向：**

如果面试官继续问 decoder-only 和 encoder-decoder 的区别，可以回答：encoder-decoder 常见于翻译任务，encoder 读完整输入，decoder 逐步生成；decoder-only 则把 prompt 和生成统一成一个自回归序列。

---

## 2. nano-vLLM 中一次模型 forward 为什么不是只看 `qwen3.py`？

**参考回答：**

因为真实推理 forward 不只是模型结构，还包括请求调度、输入整理、KV Cache、Context、logits 和 sampling。`qwen3.py` 定义 Qwen3 模型结构，但 `ModelRunner` 负责准备 `input_ids/positions`，`context.py` 传递 prefill/decode 元信息，`attention.py` 根据 Context 选择 flash attention kernel，`embed_head.py` 负责 embedding 和 lm head，`sampler.py` 负责从 logits 采样 token。所以完整 forward 要从 `ModelRunner.run()` 一直讲到 `Sampler`。

---

## 3. `ModelRunner.run()` 在一次推理中负责什么？

**参考回答：**

`ModelRunner.run()` 是模型执行入口。它根据 `is_prefill` 选择 `prepare_prefill()` 或 `prepare_decode()`，构造本轮要送入模型的 `input_ids` 和 `positions`，同时设置 Context。然后调用 `run_model()` 执行模型 forward 和 compute_logits，最后调用 sampler 得到 next token。

核心链路是：

```python
input_ids, positions = prepare_prefill/decode(seqs)
logits = run_model(input_ids, positions, is_prefill)
token_ids = sampler(logits, temperatures)
```

---

## 4. prefill 和 decode 的区别是什么？

**参考回答：**

prefill 是处理 prompt 的阶段，一次会处理每个请求的多个 token，目标是计算 prompt 的 hidden states，并把 K/V 写入 KV Cache。decode 是逐 token 生成阶段，每个请求每轮通常只输入一个新 token，目标是利用已有 KV Cache 生成下一个 token。

| 阶段 | 输入 | Attention kernel | KV Cache |
|---|---|---|---|
| prefill | 多个 prompt token | `flash_attn_varlen_func` | 写入 prompt 的 K/V |
| decode | 每请求 1 个 token | `flash_attn_with_kvcache` | 读取历史 K/V，并写入当前 K/V |

---

## 5. decode 阶段的 `input_ids: [batch_size]` 中 batch_size 是什么？

**参考回答：**

decode 阶段每个请求每轮只输入一个 token。如果当前调度器选中了 8 个还没结束的请求，那么这一轮 decode 的 `input_ids` 就有 8 个 token，shape 是 `[8]`，这里 `batch_size = 8`。在 continuous batching 中，batch_size 是动态变化的，因为有些请求结束，有些请求新加入。

---

## 6. 为什么 prefill 不直接用 `[batch, seq_len]`，而是把 token 拼成一维？

**参考回答：**

因为不同请求的 prompt 长度不同。如果 padding 到统一长度，会浪费大量计算。nano-vLLM 使用 flash-attn varlen，把多个请求的 token 拼成一维 `[total_num_tokens]`，再用 `cu_seqlens_q/k` 记录每个请求的边界。这样可以高效处理变长 prompt。

---

## 7. Context 在 nano-vLLM 中解决什么问题？

**参考回答：**

模型的 `forward(input_ids, positions)` 参数很简单，但 attention kernel 还需要知道当前是 prefill 还是 decode、slot_mapping、block_tables、cu_seqlens、context_lens 等信息。如果把这些都传进每层 forward，接口会很复杂。nano-vLLM 用全局 Context 保存这些运行时信息，`ModelRunner` 设置 Context，`Attention.forward()` 读取 Context。

---

## 8. Qwen3ForCausalLM 为什么 forward 不直接返回 logits？

**参考回答：**

`Qwen3ForCausalLM.forward()` 返回的是模型主干的 hidden states，logits 由 `compute_logits()` 调用 LM Head 计算。这样做便于推理框架控制什么时候计算 logits。特别是 prefill 阶段，prompt 中所有 token 都要经过模型主干，但只有每个请求最后一个 token 需要 logits，所以分开可以减少不必要的 vocab projection。

---

## 9. LM Head 是什么？

**参考回答：**

LM Head 是把最终 hidden states 映射成词表 logits 的线性层。输入形状是 `[num_tokens, hidden_size]`，输出是 `[num_tokens, vocab_size]`。logits 表示模型对词表中每个 token 作为下一个 token 的原始分数。sampler 会基于 logits 做 temperature、softmax 和采样。

---

## 10. Final RMSNorm 的作用是什么？

**参考回答：**

Final RMSNorm 是所有 decoder layer 结束后、LM Head 之前的最后一次 RMSNorm。它在 nano-vLLM 里还会把最后一层 MLP 输出加回 residual，然后对最终 hidden states 做归一化，使送入 LM Head 的表示尺度稳定。它不负责生成 token，只负责整理最终 hidden states。

---

## 11. RMSNorm 和 LayerNorm 有什么区别？

**参考回答：**

LayerNorm 会减均值并除以标准差：

```text
(x - mean) / sqrt(var + eps)
```

RMSNorm 不减均值，只按均方根缩放：

```text
x / sqrt(mean(x^2) + eps)
```

Qwen3 使用的是 RMSNorm。它计算更简单，主要控制 hidden 向量的尺度，提升数值稳定性。

---

## 12. `RMSNorm(x, residual)` 和普通 `RMSNorm(x)` 有什么区别？

**参考回答：**

普通 `RMSNorm(x)` 只做归一化。`RMSNorm(x, residual)` 在 nano-vLLM 中表示 fused add RMSNorm，等价于：

```python
residual = residual + x
x = RMSNorm(residual)
```

它同时完成残差相加和 RMSNorm，减少中间张量和显存读写。

---

## 13. residual 是不是每层结束后的输出？

**参考回答：**

不是。nano-vLLM 中 `residual` 更像残差主干，保存已经累积的信息；`hidden_states` 常常是当前子层新产生的输出。Attention 输出会马上加进 residual，但 MLP 输出不会在当前层末尾立刻加，而是在下一层开头的 `input_layernorm(hidden_states, residual)` 中加回；最后一层 MLP 输出在 final norm 中加回。

---

## 14. hidden_states 是不是原始输入信息？

**参考回答：**

一开始接近是。Embedding 后的 `hidden_states` 是 token id 对应的初始向量表示。但经过每一层 attention 和 MLP 后，它会融合上下文并被不断加工，所以后面的 hidden_states 是模型内部当前阶段的 token 表示，不再是原始输入。

---

## 15. QKV projection 是什么？为什么要做这一步？

**参考回答：**

Attention 需要 Q、K、V 三个向量。Q 表示当前 token 要查询什么，K 表示每个 token 的匹配特征，V 表示真正要被聚合的内容。QKV projection 就是从 hidden_states 通过线性层生成 Q/K/V：

```text
Q = XWq
K = XWk
V = XWv
```

nano-vLLM 工程上用 `QKVParallelLinear` 把三次 projection 合成一次矩阵乘法，减少 kernel launch 并方便 tensor parallel。

---

## 16. QKV projection 后为什么要 split？

**参考回答：**

因为 `qkv_proj` 一次性输出的是拼接后的大张量：

```text
qkv: [T, q_size + k_size + v_size]
```

`split` 的作用是按最后一维把它切成：

```text
q: [T, q_size]
k: [T, kv_size]
v: [T, kv_size]
```

也就是把工程上合并计算的结果重新拆回概念上的 Q/K/V。

---

## 17. split 后为什么还要 view？

**参考回答：**

split 后的 q/k/v 还是二维张量，例如：

```text
q: [T, num_heads * head_dim]
```

但 attention 要按多头计算，所以需要 view 成：

```text
q: [T, num_heads, head_dim]
k: [T, num_kv_heads, head_dim]
v: [T, num_kv_heads, head_dim]
```

`view` 不改变数据内容，只改变张量形状，让后续 attention kernel 能按 head 维度处理。

---

## 18. hidden_size 和 head_dim 分别是什么？

**参考回答：**

`hidden_size` 是每个 token 在模型内部的总向量维度，常用 `H` 表示。`head_dim` 是每个 attention head 的维度，常用 `D` 表示。通常有：

```text
hidden_size = num_attention_heads * head_dim
```

例如 hidden_size 是 1024，num_heads 是 16，那么 head_dim 通常是 64。

---

## 19. GQA 中 num_attention_heads 和 num_key_value_heads 为什么可以不同？

**参考回答：**

GQA 是 grouped-query attention。它允许 Q 的头数多于 K/V 的头数。这样可以保持较强的 query 表达能力，同时减少 KV Cache 的存储和读取成本。因为 decode 阶段显存压力很大一部分来自历史 K/V，减少 KV heads 能降低 KV Cache 显存和带宽压力。

---

## 20. RoPE 在 Qwen3Attention 中起什么作用？

**参考回答：**

RoPE 是旋转位置编码，作用在 q 和 k 上，不作用在 v 上。因为 q/k 决定 token 之间的匹配分数，位置应该影响“谁和谁相关”；v 是被聚合的内容本身。源码中是：

```python
q, k = self.rotary_emb(positions, q, k)
```

---

## 21. FlashAttention 发生在 forward 的什么位置？

**参考回答：**

FlashAttention 发生在 Q/K/V 已经生成、split、view、q/k norm 和 RoPE 之后，`o_proj` 之前。也就是 attention 子层真正计算注意力输出的位置。源码里是 `Qwen3Attention.forward()` 调用 `self.attn(q, k, v)`，而 `self.attn` 在 `layers/attention.py` 中根据 prefill/decode 选择 FlashAttention kernel。

---

## 22. `flash_attn_varlen_func` 是什么？

**参考回答：**

它是 prefill 阶段用来处理变长 prompt 的 FlashAttention kernel。因为不同请求 prompt 长度不同，nano-vLLM 会把所有 token 拼成一维，然后用 `cu_seqlens_q/k` 标记每个请求边界。`flash_attn_varlen_func` 就能在不 padding 的情况下高效处理这些变长序列。

---

## 23. `flash_attn_with_kvcache` 是什么？

**参考回答：**

它是 decode 阶段使用的 attention kernel。decode 每个请求当前只有一个新 token，但要 attend 历史上下文。历史 K/V 保存在 KV Cache 中，所以 `flash_attn_with_kvcache` 会用当前 q 去读取历史 k_cache/v_cache，并计算当前 token 的 attention 输出。

---

## 24. KV Cache 是在哪里分配、在哪里写入的？

**参考回答：**

KV Cache 在 `ModelRunner.allocate_kv_cache()` 中统一分配，形状大致包含 K/V 两份、层数、block 数、block_size、kv heads 和 head_dim。分配后，`ModelRunner` 会遍历模型模块，把每个 Attention 模块的 `k_cache/v_cache` 指向对应层的 cache。写入发生在 `layers/attention.py` 中的 `store_kvcache()`，它根据 `context.slot_mapping` 把当前 token 的 k/v 写到对应 slot。

---

## 25. `slot_mapping` 和 `block_tables` 分别有什么作用？

**参考回答：**

`slot_mapping` 表示当前 step 新产生的 K/V 应该写入 KV Cache 的哪个具体 slot。`block_tables` 表示每个请求的逻辑 token block 对应到物理 KV Cache block 的映射。prefill 和 decode 都需要 slot_mapping 写入当前 K/V，decode 还依赖 block_tables 找到历史 K/V。

---

## 26. MLP / FFN 在 Transformer 里负责什么？

**参考回答：**

Attention 负责 token 和 token 之间的信息交互，MLP/FFN 负责对每个 token 自己的 hidden state 做非线性特征加工。Qwen3 的 MLP 使用 SwiGLU 结构，输入和输出都是 `[T, hidden_size]`，中间会扩展到 `intermediate_size`。

---

## 27. SwiGLU 的公式是什么？为什么需要 gate？

**参考回答：**

Qwen3 的 SwiGLU 公式是：

```text
down_proj(SiLU(gate_proj(x)) * up_proj(x))
```

`up_proj` 生成候选特征，`gate_proj` 生成门控信号，SiLU 后和 up 分支逐元素相乘，表示哪些中间特征应该通过。gate 让 FFN 不只是简单激活，而是具备门控筛选能力。

---

## 28. 为什么 nano-vLLM 中没有单独的 `gate_proj` 和 `up_proj`？

**参考回答：**

概念上有 `gate_proj` 和 `up_proj`，但工程上把它们合并成 `gate_up_proj`。`gate_up_proj` 一次输出 `[T, 2 * intermediate_size]`，然后 `SiluAndMul` 切成两半，一半作为 gate，一半作为 up。这样可以减少一次线性层调用和 kernel launch，也方便 tensor parallel。

---

## 29. `gate_up_proj` 和 `down_proj` 在 tensor parallel 下怎么切？

**参考回答：**

`gate_up_proj` 用 `MergedColumnParallelLinear`，按输出维度切分，因此每张卡只计算一部分 intermediate features。`down_proj` 用 `RowParallelLinear`，按输入维度切分，每张卡用自己的 intermediate shard 计算局部输出 `[T, hidden_size]`，最后通过 `all_reduce` 把局部输出相加，得到完整的 MLP 输出。

---

## 30. `ColumnParallelLinear` 和 `RowParallelLinear` 的区别是什么？

**参考回答：**

`ColumnParallelLinear` 按输出维度切权重，每张卡得到一部分输出特征，适合 QKV projection、gate/up projection 这种输出很宽的线性层。`RowParallelLinear` 按输入维度切权重，每张卡计算局部输出，然后 all_reduce 合并，适合 attention 的 `o_proj` 和 MLP 的 `down_proj`。

---

## 31. 为什么 prefill 阶段 LM Head 只取最后 token？

**参考回答：**

prefill 阶段所有 prompt token 都要经过模型主干，因为要建立上下文和 KV Cache。但预测下一个 token 只需要每个请求最后一个 prompt token 的 hidden state。如果对 prompt 中每个 token 都算 vocab logits，会浪费大量计算和显存。因此 `ParallelLMHead.forward()` 在 prefill 时用 `cu_seqlens_q[1:] - 1` 找到每个请求最后 token，只对这些位置算 logits。

---

## 32. Sampler 在整个推理流程中的作用是什么？

**参考回答：**

Sampler 接收 LM Head 输出的 logits，根据 temperature 做缩放，再做 softmax 得到概率分布，然后采样下一个 token id。采样得到的 token 会追加到对应 Sequence 中，下一轮 decode 时作为新的输入 token。

---

## 33. 如果让你说 nano-vLLM 中模型相关文件的调用关系，你怎么说？

**参考回答：**

可以从执行入口开始说：

```text
ModelRunner.run()
  -> prepare_prefill/decode()
  -> set_context()
  -> Qwen3ForCausalLM.forward()
  -> Qwen3Model.forward()
  -> Qwen3DecoderLayer.forward()
  -> Qwen3Attention / Qwen3MLP
  -> Attention 读取 Context 并操作 KV Cache
  -> Final RMSNorm
  -> compute_logits / LM Head
  -> Sampler
```

各文件职责是：`model_runner.py` 负责执行和输入准备，`context.py` 负责传递运行时信息，`qwen3.py` 负责模型结构，`attention.py` 负责 FlashAttention 和 KV Cache，`linear.py` 负责并行线性层，`layernorm.py` 负责 RMSNorm 和 fused add RMSNorm，`activation.py` 负责 SwiGLU，`embed_head.py` 负责 embedding 和 LM Head，`sampler.py` 负责采样。

---

## 34. 面试官追问：你觉得 nano-vLLM 的 Qwen3 实现和原始 Transformer 讲法最大的不同是什么？

**参考回答：**

最大的不同是工程实现会为了推理效率做很多融合和状态管理。比如 Q/K/V 概念上是三个 projection，但代码里合并成 `qkv_proj`；gate/up 概念上是两个 projection，但代码里合并成 `gate_up_proj`；residual add 和 RMSNorm 概念上是两步，但代码里合并成 `RMSNorm(x, residual)`；prefill/decode 差异也不写在模型结构里，而是通过 Context 传给 Attention kernel。这些不是改变模型数学结构，而是为了减少 kernel launch、减少显存读写、适配 KV Cache 和 continuous batching。

---

## 35. 最后复习：高频考点清单

如果时间有限，优先掌握下面这些：

1. 完整 Qwen3 dense forward 流程。
2. prefill 和 decode 的输入、目标、attention kernel 区别。
3. KV Cache 的分配、写入、读取。
4. QKV projection、split、view、GQA。
5. FlashAttention varlen 和 FlashAttention with KV Cache。
6. RMSNorm、fused add RMSNorm、residual 流。
7. MLP/SwiGLU、gate_up_proj、down_proj。
8. LM Head 为什么和 forward 分开。
9. ColumnParallelLinear 和 RowParallelLinear 的区别。
10. 各个模型相关文件的调用关系。

---

## 36. 一句话结尾模板

如果面试最后让你总结 nano-vLLM 中 Qwen3 模型 forward，你可以说：

> 我理解的 Qwen3 forward 不只是 `qwen3.py` 里的模型结构，而是从 `ModelRunner` 准备 prefill/decode 输入和 Context 开始，经过 embedding、decoder layers、RMSNorm、QKV projection、FlashAttention/KV Cache、SwiGLU MLP、final norm、LM Head，最后由 sampler 得到 next token。模型数学结构是 decoder-only Transformer，nano-vLLM 的工程重点是把 QKV、gate/up、add+RMSNorm 等操作做合并，并通过 Context 和 KV Cache 支持高效自回归推理。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
