# Decoder-only LLM 架构、逐层计算与训练推理流程学习笔记

> 适用对象：正在学习 Transformer、LLM 推理、vLLM / nano-vLLM / AI Infra 的同学。  
> 核心目标：搞清楚 Decoder-only LLM 从文本输入到 logits 输出的完整链路，理解每一层在做什么、张量形状如何变化、训练和推理为什么不同，以及这些概念如何对应到推理框架中的 Prefill、Decode、KV Cache、Sampling、Batching 等模块。

---

## 1. Decoder-only LLM 是什么

Decoder-only LLM 是目前主流大语言模型最常用的结构之一，例如 GPT 系列、LLaMA / Qwen / DeepSeek / Mistral 等大多数生成式语言模型都属于这一类。

它的核心特点是：

> **只使用 Transformer 的 Decoder 部分，通过自回归方式，从左到右逐 token 预测下一个 token。**

所谓自回归，也就是模型生成文本时遵循：

```text
给定前面的 token：x1, x2, ..., xt
预测下一个 token：x(t+1)
```

数学上，模型学习的是条件概率：

```text
P(x1, x2, ..., xT)
= P(x1) · P(x2 | x1) · P(x3 | x1, x2) · ... · P(xT | x1, ..., x(T-1))
```

也就是说，Decoder-only LLM 不会一次性“理解完整答案”，而是每一步根据当前上下文预测下一个 token。

---

## 2. Decoder-only LLM 的整体结构

一个典型 Decoder-only LLM 的结构可以概括为：

```text
输入文本
  ↓
Tokenizer
  ↓
Token IDs
  ↓
Token Embedding
  ↓
Position Encoding / RoPE
  ↓
N 层 Decoder Block
  ↓
Final Norm
  ↓
LM Head
  ↓
Logits
  ↓
Softmax / Sampling
  ↓
下一个 Token
```

更具体地，一个 Decoder Block 通常包含：

```text
输入 hidden states
  ↓
RMSNorm / LayerNorm
  ↓
Causal Self-Attention
  ↓
Residual Add
  ↓
RMSNorm / LayerNorm
  ↓
MLP / FFN / SwiGLU
  ↓
Residual Add
  ↓
输出 hidden states
```

现代 LLM 通常使用 **Pre-Norm** 结构：先做归一化，再进入 Attention 或 MLP。这与早期 Transformer 的 Post-Norm 略有不同。

---

## 3. 输入层：Tokenizer 与 Token IDs

### 3.1 Tokenizer 的作用

Tokenizer 负责把人类文本切分成模型可以处理的离散编号。

例如：

```text
文本："vLLM is fast"
Tokenizer 输出：[318, 11234, 374, 5043]
```

这里的数字就是 **token id**。每个 token id 对应词表中的一个 token。

需要注意：

- token 不一定等于一个汉字或一个英文单词；
- token 可能是一个字、一个词、一个词片段，甚至是空格加词；
- 不同模型的 tokenizer 不同，同一句话切出来的 token ids 可能不同；
- 模型本身只能处理 token id，不能直接处理字符串。

### 3.2 Token IDs 的张量形状

对于一个 batch 输入，token ids 通常形状为：

```text
input_ids: [batch_size, seq_len]
```

例如 batch_size = 2，seq_len = 6：

```text
[
  [101, 203, 404, 505, 606, 707],
  [111, 222, 333, 444, 555, 666]
]
```

含义是：

- 第 0 维：不同请求 / 不同样本；
- 第 1 维：每个样本中的 token 序列。

---

## 4. Embedding 层：把 token id 变成向量

### 4.1 为什么需要 Embedding

token id 只是离散编号，本身没有语义。模型不能直接对 id 做语义计算，所以需要一个 Embedding 表，把每个 token id 映射成一个连续向量。

Embedding 表可以理解为一个大矩阵：

```text
Embedding Weight: [vocab_size, hidden_size]
```

其中：

- `vocab_size`：词表大小，例如 32K、100K、150K；
- `hidden_size`：模型隐藏层维度，例如 4096、5120、8192。

当输入 token id 为 `318` 时，模型会取出 Embedding 表第 318 行作为这个 token 的向量。

### 4.2 Embedding 后的形状

输入：

```text
input_ids: [batch_size, seq_len]
```

经过 Embedding 后：

```text
hidden_states: [batch_size, seq_len, hidden_size]
```

例如：

```text
[2, 6] → [2, 6, 4096]
```

含义是：每个 token 都变成了一个 4096 维向量。

### 4.3 Embedding 向量包含什么

Embedding 向量不是人工定义的，而是在训练中学出来的。它会逐渐编码：

- 词义信息；
- 语法信息；
- 常见搭配关系；
- 与其他 token 的潜在语义距离。

例如“猫”和“狗”的 embedding 可能在向量空间中比较接近，而“猫”和“显卡”的 embedding 可能相对更远。

---

## 5. 位置编码：让模型知道 token 的顺序

### 5.1 为什么需要位置信息

Self-Attention 本身对输入顺序不敏感。如果只给模型一组 token 向量，模型不知道哪个 token 在前、哪个在后。

例如：

```text
我 爱 你
你 爱 我
```

这两句话 token 类似，但顺序不同，含义不同。所以模型必须知道每个 token 的位置。

### 5.2 常见位置编码方式

Decoder-only LLM 中常见的位置编码方式包括：

1. **绝对位置编码**
   - 早期 Transformer 使用；
   - 给每个位置一个位置向量；
   - 与 token embedding 相加。

2. **Learned Position Embedding**
   - GPT 类模型早期常用；
   - 位置编码也是训练出来的参数。

3. **RoPE：Rotary Position Embedding**
   - LLaMA、Qwen、Mistral 等主流模型常用；
   - 不是简单加位置向量，而是在 Attention 的 Q、K 上施加旋转变换；
   - 更适合长上下文扩展。

### 5.3 RoPE 的直观理解

RoPE 的核心思想是：

> 不直接给 hidden state 加位置，而是在计算注意力前，把不同位置的 Q 和 K 旋转到不同角度，使得点积时天然包含相对位置信息。

在 Attention 中，注意力分数来自：

```text
score = Q · K^T
```

RoPE 会修改 Q 和 K：

```text
Q_pos = RoPE(Q, position)
K_pos = RoPE(K, position)
```

然后再计算：

```text
score = Q_pos · K_pos^T
```

这样模型就能感知 token 之间的相对距离。

---

## 6. Decoder Block 总览

一个 Decoder-only LLM 通常由很多层 Decoder Block 堆叠而成。

例如：

```text
LLaMA-7B: 32 layers
Qwen2.5-7B: 28 layers
大模型可能有 60、80、100+ layers
```

每一层都在不断更新 token 的 hidden state。

第 0 层输入是 embedding：

```text
h0 = token_embedding(input_ids)
```

第 1 层输出：

```text
h1 = DecoderBlock1(h0)
```

第 n 层输出：

```text
hn = DecoderBlockN(h(n-1))
```

最后再送入 LM Head 得到 logits。

---

## 7. Normalization 层：LayerNorm / RMSNorm

### 7.1 为什么需要 Norm

深层网络训练时，hidden states 的数值分布可能不稳定。如果每一层输出的均值、方差变化很大，训练会变得困难，甚至梯度爆炸或梯度消失。

Norm 层的作用是稳定数值分布，让模型更容易训练。

### 7.2 LayerNorm

LayerNorm 会对每个 token 的 hidden 向量做归一化。

输入：

```text
x: [batch_size, seq_len, hidden_size]
```

对最后一维 hidden_size 计算均值和方差：

```text
mean = average(x)
var = variance(x)
```

归一化：

```text
LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta
```

其中：

- `gamma`：可学习缩放参数；
- `beta`：可学习平移参数；
- `eps`：防止除零的小数。

### 7.3 RMSNorm

现代 LLM 更常使用 RMSNorm，例如 LLaMA / Qwen。

RMSNorm 不减均值，只用均方根归一化：

```text
RMS(x) = sqrt(mean(x^2) + eps)
RMSNorm(x) = x / RMS(x) * weight
```

相比 LayerNorm：

- 计算更简单；
- 参数更少；
- 推理效率更高；
- 在大模型中效果足够好。

### 7.4 Pre-Norm 结构

现代 Decoder-only LLM 通常使用：

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

而不是：

```text
x = Norm(x + Attention(x))
```

Pre-Norm 的好处是训练更稳定，尤其适合堆叠很多层。

---

## 8. Causal Self-Attention：Decoder-only 的核心

### 8.1 Self-Attention 在做什么

Self-Attention 的作用是：

> 对每个 token，根据上下文中其他 token 的信息，重新计算它的表示。

例如句子：

```text
苹果 发布 了 新 手机，它 很 受欢迎
```

当模型处理“它”时，需要知道“它”大概率指代“新手机”或“苹果发布的产品”。Self-Attention 就负责建立这种 token 之间的关系。

### 8.2 为什么叫 Causal

Decoder-only LLM 是自回归模型，预测当前位置时不能看到未来 token。

例如训练时输入：

```text
我 今天 很 开心
```

当模型预测“很”时，只能看：

```text
我 今天
```

不能看未来的“开心”。

所以需要 **causal mask**，也叫因果 mask。

Causal mask 的规则：

```text
第 i 个 token 只能 attend 到位置 <= i 的 token
不能 attend 到位置 > i 的 token
```

注意力矩阵形状为 `[seq_len, seq_len]`，允许区域是下三角：

```text
位置 0: 只能看 0
位置 1: 可以看 0,1
位置 2: 可以看 0,1,2
位置 3: 可以看 0,1,2,3
```

矩阵示意：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

其中 1 表示可以看，0 表示不能看。

---

## 9. Q、K、V 的含义与计算

### 9.1 QKV 从哪里来

Attention 会把输入 hidden states 线性变换成 Q、K、V：

```text
Q = X Wq
K = X Wk
V = X Wv
```

其中：

```text
X:  [batch_size, seq_len, hidden_size]
Wq: [hidden_size, num_heads * head_dim]
Wk: [hidden_size, num_kv_heads * head_dim]
Wv: [hidden_size, num_kv_heads * head_dim]
```

### 9.2 Q、K、V 分别是什么

可以直观理解为：

- **Q：Query，当前 token 想找什么信息**；
- **K：Key，每个 token 提供什么索引特征**；
- **V：Value，每个 token 真正携带的信息内容**。

注意力分数由 Q 和 K 决定：

```text
attention_score = Q · K^T
```

然后用分数加权 V：

```text
attention_output = softmax(score) · V
```

### 9.3 一个 token 如何通过 Attention 更新自己

对于第 i 个 token：

1. 取出它的 Query：`Qi`；
2. 与前面所有 token 的 Key 做点积；
3. 得到它对每个历史 token 的关注程度；
4. softmax 归一化成权重；
5. 用这些权重加权所有历史 token 的 Value；
6. 得到新的 token 表示。

公式：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_head) + mask) V
```

其中：

- `sqrt(d_head)` 用来缩放点积，避免数值过大；
- `mask` 用来屏蔽未来 token；
- `softmax` 把分数变成概率分布。

---

## 10. Multi-Head Attention / MQA / GQA

### 10.1 Multi-Head Attention

单个 attention head 只能从一个子空间理解 token 关系。多头注意力会把 hidden_size 拆成多个 head，每个 head 学不同的关系。

例如：

```text
hidden_size = 4096
num_heads = 32
head_dim = 128
```

那么：

```text
4096 = 32 * 128
```

每个 head 独立计算 Attention，最后拼接起来。

形状变化大致是：

```text
X: [batch_size, seq_len, hidden_size]
Q: [batch_size, seq_len, num_heads, head_dim]
K: [batch_size, seq_len, num_heads, head_dim]
V: [batch_size, seq_len, num_heads, head_dim]
```

Attention 输出：

```text
O: [batch_size, seq_len, num_heads, head_dim]
```

拼接后：

```text
O: [batch_size, seq_len, hidden_size]
```

再经过输出投影：

```text
Output = O Wo
```

### 10.2 MHA 的问题

标准 MHA 中，Q、K、V 都有 `num_heads` 个 head。

推理时，K 和 V 需要保存到 KV Cache。如果 `num_heads` 很大，KV Cache 会非常占显存。

KV Cache 显存大致与以下因素相关：

```text
layers * seq_len * num_kv_heads * head_dim * 2 * dtype_size
```

其中 `2` 表示 K 和 V。

### 10.3 MQA：Multi-Query Attention

MQA 的思想是：

> Q 仍然有很多个 head，但所有 Q head 共享同一组 K/V head。

也就是：

```text
num_heads 很大
num_kv_heads = 1
```

优点：显著减少 KV Cache。

缺点：表达能力可能下降。

### 10.4 GQA：Grouped-Query Attention

GQA 是 MHA 和 MQA 的折中：

```text
num_heads = 32
num_kv_heads = 8
```

也就是每 4 个 Q head 共享一组 K/V head。

很多现代 LLM 使用 GQA，因为它在效果和推理效率之间比较平衡。

---

## 11. Attention 的详细计算流程

假设：

```text
batch_size = B
seq_len = S
hidden_size = H
num_heads = Nh
num_kv_heads = Nkv
head_dim = D
```

### 11.1 输入 hidden states

```text
X: [B, S, H]
```

### 11.2 线性投影得到 QKV

```text
Q = X Wq → [B, S, Nh * D]
K = X Wk → [B, S, Nkv * D]
V = X Wv → [B, S, Nkv * D]
```

reshape：

```text
Q → [B, S, Nh, D]
K → [B, S, Nkv, D]
V → [B, S, Nkv, D]
```

### 11.3 对 Q/K 应用 RoPE

```text
Q = RoPE(Q, position_ids)
K = RoPE(K, position_ids)
```

V 通常不加 RoPE。

### 11.4 处理 GQA 的 K/V 共享

如果 `num_heads > num_kv_heads`，则 K/V 会被多个 Q head 共享。

例如：

```text
num_heads = 32
num_kv_heads = 8
每 4 个 Q head 共享 1 个 K/V head
```

### 11.5 计算 attention score

为了便于矩阵乘法，通常把维度调整成：

```text
Q: [B, Nh, S, D]
K: [B, Nkv, S, D]
V: [B, Nkv, S, D]
```

经过共享/广播后：

```text
K: [B, Nh, S, D]
V: [B, Nh, S, D]
```

计算：

```text
Score = Q @ K^T / sqrt(D)
```

形状：

```text
Score: [B, Nh, S, S]
```

第一个 S 表示 query 位置，第二个 S 表示 key/value 位置。

### 11.6 加 causal mask

未来位置被加上一个极小值：

```text
Score[future_position] = -inf
```

softmax 后这些位置概率变成 0。

### 11.7 softmax 得到注意力权重

```text
P = softmax(Score)
```

形状：

```text
P: [B, Nh, S, S]
```

### 11.8 加权 Value

```text
O = P @ V
```

形状：

```text
O: [B, Nh, S, D]
```

转置并拼接：

```text
O → [B, S, Nh * D] = [B, S, H]
```

### 11.9 输出投影

```text
AttentionOutput = O Wo
```

输出形状仍然是：

```text
[B, S, H]
```

---

## 12. Residual Connection：残差连接

每个 Attention 和 MLP 后面都有残差连接。

形式：

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

残差连接的作用：

1. 保留原始信息；
2. 缓解深层网络梯度消失；
3. 让模型每一层学习“增量修改”，而不是完全重写表示；
4. 使得几十层甚至上百层 Transformer 可以稳定训练。

如果没有残差连接，深层 LLM 很难训练。

---

## 13. MLP / FFN 层：对每个 token 做非线性变换

### 13.1 MLP 在做什么

Attention 负责 token 之间的信息交互，MLP 负责对每个 token 的表示做非线性变换。

可以粗略理解为：

- Attention：不同 token 之间交流；
- MLP：每个 token 自己内部思考和加工。

MLP 不混合不同 token 的位置，它对每个 token 独立执行同样的前馈网络。

### 13.2 传统 FFN

原始 Transformer 使用：

```text
FFN(x) = W2 · activation(W1 · x)
```

形状：

```text
x: [B, S, H]
W1: [H, intermediate_size]
W2: [intermediate_size, H]
```

通常：

```text
intermediate_size ≈ 4 * hidden_size
```

### 13.3 SwiGLU

现代 LLM 常用 SwiGLU 结构，例如 LLaMA / Qwen。

SwiGLU 大致形式：

```text
MLP(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
```

其中：

```text
gate = gate_proj(x)
up   = up_proj(x)
hidden = SiLU(gate) * up
out = down_proj(hidden)
```

涉及三个线性层：

```text
gate_proj: [H, intermediate_size]
up_proj:   [H, intermediate_size]
down_proj: [intermediate_size, H]
```

### 13.4 SwiGLU 为什么有用

SwiGLU 中的 `gate_proj` 起到门控作用，决定哪些特征通过，哪些特征被抑制。

相比普通 FFN，它通常能带来更好的表达能力和训练效果。

---

## 14. Final Norm 与 LM Head

### 14.1 Final Norm

所有 Decoder Block 结束后，模型会对最终 hidden states 再做一次 Norm：

```text
final_hidden = final_norm(hidden_states)
```

形状：

```text
[B, S, H]
```

### 14.2 LM Head

LM Head 把 hidden state 映射回词表空间：

```text
logits = final_hidden @ W_vocab^T
```

其中：

```text
W_vocab: [vocab_size, hidden_size]
logits: [B, S, vocab_size]
```

每个位置都会得到一个长度为 `vocab_size` 的向量，表示下一个 token 是词表中每个 token 的分数。

例如：

```text
logits[0, 5, :] 表示第 0 个样本第 5 个位置预测下一个 token 的分数
```

### 14.3 Weight Tying

很多模型会让 token embedding 和 LM Head 共享权重：

```text
lm_head.weight = embedding.weight
```

这样可以：

- 减少参数量；
- 提升输入 token 表示和输出 token 分类之间的一致性。

但并不是所有模型都必须共享。

---

## 15. Logits、Softmax 与下一个 token 概率

LM Head 输出的是 logits，不是概率。

logits 经过 softmax 后变成概率分布：

```text
P(token_i) = exp(logit_i) / sum(exp(logit_j))
```

如果词表大小为 100000，那么每个位置会输出 100000 个分数。

在训练时，模型用这些 logits 和真实下一个 token 计算 loss。

在推理时，模型根据这些 logits 选择或采样下一个 token。

---

## 16. Decoder-only LLM 的训练流程

### 16.1 训练目标：预测下一个 token

训练 Decoder-only LLM 的核心任务是：

```text
给定前文，预测下一个 token
```

例如文本：

```text
大模型 推理 框架 vLLM 很 高效
```

训练样本可以看成：

```text
输入：大模型
目标：推理

输入：大模型 推理
目标：框架

输入：大模型 推理 框架
目标：vLLM

输入：大模型 推理 框架 vLLM
目标：很
```

实际训练时不会真的拆成这么多条，而是通过 shift 一次性完成。

### 16.2 Input 和 Label 的 shift

假设 token ids：

```text
[10, 20, 30, 40, 50]
```

训练时输入：

```text
input_ids = [10, 20, 30, 40]
```

标签：

```text
labels = [20, 30, 40, 50]
```

也就是：

```text
位置 0 输入 10，预测 20
位置 1 输入 20，预测 30
位置 2 输入 30，预测 40
位置 3 输入 40，预测 50
```

在实际代码中，通常输入整段：

```text
input_ids: [10, 20, 30, 40, 50]
```

模型输出：

```text
logits: [seq_len, vocab_size]
```

然后内部做 shift：

```text
shift_logits = logits[:, :-1, :]
shift_labels = input_ids[:, 1:]
```

### 16.3 Cross Entropy Loss

对于每个位置，模型输出词表概率分布，真实标签是下一个 token id。

交叉熵损失：

```text
loss = -log P(real_next_token)
```

如果模型给真实 token 的概率越高，loss 越小。

整个 batch 的 loss 是所有有效 token 的平均值。

### 16.4 Causal Mask 在训练中的作用

训练时虽然一次性把整个序列输入模型，但每个位置只能看到它前面的 token。

例如输入：

```text
[10, 20, 30, 40]
```

模型并行计算所有位置的输出，但由于 causal mask：

```text
位置 0 只能看 10
位置 1 只能看 10,20
位置 2 只能看 10,20,30
位置 3 只能看 10,20,30,40
```

所以训练不会作弊。

这也是 Transformer 训练高效的原因：

> 训练时可以并行计算所有 token 的预测；推理时必须逐 token 生成。

### 16.5 一次训练 step 的流程

```text
1. 读取文本数据
2. Tokenizer 编码成 token ids
3. 拼接 / 截断 / padding / packing 成固定长度序列
4. 输入模型 forward
5. 得到 logits
6. logits 和 labels shift 后计算 cross entropy loss
7. 反向传播 backward
8. 优化器更新参数
9. 清空梯度，进入下一步
```

伪代码：

```python
input_ids = batch["input_ids"]

logits = model(input_ids)

shift_logits = logits[:, :-1, :]
shift_labels = input_ids[:, 1:]

loss = cross_entropy(
    shift_logits.reshape(-1, vocab_size),
    shift_labels.reshape(-1),
)

loss.backward()
optimizer.step()
optimizer.zero_grad()
```

### 16.6 预训练、SFT、RLHF / DPO 的关系

Decoder-only LLM 的训练通常分阶段：

#### 阶段一：Pretraining

目标：学习语言建模能力。

数据：海量网页、书籍、代码、论文等。

训练任务：next token prediction。

模型学到：

- 语言规律；
- 世界知识；
- 基础推理能力；
- 代码和数学模式。

#### 阶段二：SFT

SFT 是 Supervised Fine-Tuning，有监督微调。

目标：让模型学会按照指令回答。

数据形式：

```text
用户：请解释 vLLM
助手：vLLM 是一个高性能大模型推理框架……
```

训练任务本质仍然是 next token prediction，只是数据变成了问答格式。

#### 阶段三：偏好对齐

常见方法：

- RLHF；
- DPO；
- IPO；
- ORPO。

目标：让模型回答更符合人类偏好，例如更安全、更有帮助、更遵循指令。

---

## 17. Decoder-only LLM 的推理流程

推理和训练最大的不同是：

> 训练时一次性并行预测所有位置；推理时需要一个 token 一个 token 生成。

推理可以分为两个阶段：

```text
Prefill 阶段：处理用户 prompt，建立上下文和 KV Cache
Decode 阶段：逐 token 生成回答
```

---

## 18. Prefill 阶段

### 18.1 Prefill 是什么

用户输入 prompt 后，模型需要先读完整个 prompt，计算每个 prompt token 的 hidden states，并为每一层 Attention 保存 K/V。

例如用户输入：

```text
请介绍一下 vLLM 的核心原理
```

tokenizer 后得到：

```text
[101, 208, 309, 415, 516, 617, 718]
```

Prefill 阶段会一次性处理这 7 个 token。

### 18.2 Prefill 的输出是什么

Prefill 主要产生两类结果：

1. 最后一个位置的 logits，用来采样第一个输出 token；
2. 每一层的 KV Cache，用于后续 decode。

### 18.3 Prefill 为什么计算量大

Prefill 的 Attention 需要处理完整 prompt。

如果 prompt 长度为 S，那么 Attention 矩阵是：

```text
[S, S]
```

所以 prefill 更偏向计算密集，尤其长 prompt 时非常重。

---

## 19. KV Cache

### 19.1 KV Cache 是什么

在自回归生成中，每生成一个新 token，都需要 attend 到所有历史 token。

如果没有 KV Cache，每一步都要重新计算所有历史 token 的 K/V。

这会非常浪费。

KV Cache 的做法是：

> 历史 token 的 K/V 一旦算出来，就缓存起来；下一步 decode 只计算新 token 的 Q/K/V，并复用历史 K/V。

### 19.2 KV Cache 保存什么

每一层 Attention 都有自己的 K Cache 和 V Cache。

形状大致为：

```text
K Cache: [num_layers, batch, seq_len, num_kv_heads, head_dim]
V Cache: [num_layers, batch, seq_len, num_kv_heads, head_dim]
```

实际推理框架中可能会为了性能调整 layout，例如 vLLM 会采用分页块管理。

### 19.3 KV Cache 的显存占用

KV Cache 显存约为：

```text
num_layers
* num_tokens
* num_kv_heads
* head_dim
* 2
* dtype_size
```

其中：

- `num_layers`：模型层数；
- `num_tokens`：上下文 token 数；
- `num_kv_heads`：K/V head 数；
- `head_dim`：每个 head 的维度；
- `2`：K 和 V；
- `dtype_size`：FP16/BF16 通常是 2 bytes。

这解释了为什么长上下文推理非常吃显存。

---

## 20. Decode 阶段

### 20.1 Decode 是什么

Decode 阶段从生成第一个输出 token 开始。

每一步：

```text
输入上一步生成的新 token
  ↓
计算这个 token 的 Q/K/V
  ↓
把新 K/V 写入 KV Cache
  ↓
新 token 的 Q attend 到历史所有 K/V
  ↓
得到 logits
  ↓
采样下一个 token
```

### 20.2 Decode 的特点

Decode 每次只处理一个新 token，所以 query 长度通常是 1。

Attention 计算形状大致为：

```text
Q: [1, head_dim]
K Cache: [past_seq_len, head_dim]
V Cache: [past_seq_len, head_dim]
```

计算：

```text
score = Q @ K_cache^T
output = softmax(score) @ V_cache
```

### 20.3 Decode 为什么容易成为性能瓶颈

Decode 阶段每次只生成一个 token，矩阵规模小，GPU 并行度不足，且需要不断读 KV Cache。

所以 decode 往往是：

- memory bandwidth bound；
- 对 batch size 很敏感；
- 需要 continuous batching 提升吞吐。

这也是 vLLM、SGLang、TensorRT-LLM 等推理框架重点优化的地方。

---

## 21. Sampling：从 logits 到输出 token

模型输出 logits 后，需要决定下一个 token。

常见策略：

### 21.1 Greedy Decoding

选择 logits 最大的 token：

```text
next_token = argmax(logits)
```

优点：稳定、确定。

缺点：可能死板、缺少多样性。

### 21.2 Temperature

temperature 控制分布平滑程度：

```text
logits = logits / temperature
```

- temperature 越低，输出越确定；
- temperature 越高，输出越随机。

### 21.3 Top-k

只保留概率最高的 k 个 token，然后重新归一化采样。

### 21.4 Top-p / Nucleus Sampling

保留累计概率达到 p 的最小 token 集合。

例如 `top_p = 0.9`，表示保留累计概率 90% 的候选 token。

### 21.5 Repetition Penalty

降低已经出现过的 token 的概率，减少重复输出。

### 21.6 EOS 和 Stop

如果生成 EOS token，或者匹配用户设置的 stop sequence，推理结束。

---

## 22. Decoder-only LLM 端到端推理主线

完整推理流程可以串起来看：

```text
1. 用户输入 prompt
2. Tokenizer 把 prompt 转成 input_ids
3. 模型进入 prefill
4. Embedding 把 token ids 转成 hidden states
5. 多层 Decoder Block 处理 prompt
6. 每层 Attention 计算并保存 prompt 的 KV Cache
7. LM Head 输出最后一个 prompt 位置的 logits
8. Sampler 采样第一个输出 token
9. 进入 decode 循环
10. 新 token 再次经过 embedding 和各层 Decoder Block
11. 每层只计算新 token 的 Q/K/V
12. 新 K/V 追加写入 KV Cache
13. 新 token 的 Q attend 到历史 KV Cache
14. LM Head 输出下一个 token 的 logits
15. Sampler 采样下一个 token
16. 重复 decode，直到 EOS / stop / max_tokens
17. Tokenizer decode 输出 token ids 为文本
```

这就是 Decoder-only LLM 推理的核心链路。

---

## 23. 训练和推理的核心区别

| 对比项 | 训练 | 推理 |
|---|---|---|
| 目标 | 学会预测下一个 token | 根据 prompt 生成新 token |
| 输入 | 大量文本序列 | 用户 prompt + 已生成 token |
| 计算方式 | 一次性并行计算整段序列 | Prefill 并行，Decode 串行逐 token |
| 是否使用 label | 使用 | 不使用 |
| 是否反向传播 | 是 | 否 |
| 是否更新参数 | 是 | 否 |
| 是否使用 KV Cache | 一般不用或训练中不作为核心优化 | 大量使用 |
| 性能瓶颈 | 算力、显存、通信 | KV Cache、batching、decode latency |
| 输出 | loss | token / text |

---

## 24. 为什么训练能并行，推理不能完全并行

训练时，真实完整文本已经存在。模型可以通过 causal mask 并行计算所有位置的预测。

例如训练句子：

```text
A B C D E
```

模型可以同时计算：

```text
A → B
A B → C
A B C → D
A B C D → E
```

因为 B、C、D、E 都已经在训练数据里。

但是推理时，未来 token 不存在。

```text
Prompt → ?
```

必须先生成第一个 token，才能把它作为上下文继续生成第二个 token。

所以 decode 天然串行。

这也是投机解码、Medusa、MTP 等技术想解决的问题：

> 尽量让一次 forward 产生或验证多个 token，从而减少串行步数。

---

## 25. Decoder-only LLM 与 vLLM / 推理框架的对应关系

| LLM 概念 | 推理框架中的对应模块 |
|---|---|
| Tokenizer | 输入预处理、prompt tokenization |
| Embedding | Model forward 的第一步 |
| Decoder Block | 模型主体执行 |
| Attention | Attention backend / FlashAttention / PagedAttention |
| Q/K/V | Attention 中间张量 |
| KV Cache | KV cache manager / block manager |
| Causal Mask | attention metadata |
| Prefill | prompt 阶段执行 |
| Decode | 逐 token 生成阶段 |
| Logits | 模型输出分数 |
| Sampler | sampling module |
| EOS | 请求结束条件 |
| Batch | scheduler 组织的一轮请求集合 |
| Sequence | 一个请求生成过程中的状态 |
| Block Table | 逻辑 token block 到物理 KV block 的映射 |
| Slot Mapping | token 到 KV cache 物理槽位的映射 |
| Continuous Batching | 动态加入/移除请求，提高 decode 吞吐 |

---

## 26. 从源码视角理解一次 forward

以 PyTorch 风格伪代码表示 Decoder-only LLM：

```python
class DecoderOnlyLLM(nn.Module):
    def __init__(self):
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            DecoderLayer() for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, position_ids, attention_mask=None, past_key_values=None):
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states, past_key_values = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, past_key_values
```

Decoder Layer：

```python
class DecoderLayer(nn.Module):
    def forward(self, x, position_ids, attention_mask, past_key_values):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, position_ids, attention_mask, past_key_values)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x
```

Attention：

```python
class SelfAttention(nn.Module):
    def forward(self, x, position_ids, attention_mask, past_key_values):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = reshape_to_heads(q)
        k = reshape_to_kv_heads(k)
        v = reshape_to_kv_heads(v)

        q, k = apply_rope(q, k, position_ids)

        if past_key_values is not None:
            k = concat(past_k, k)
            v = concat(past_v, v)

        attn_scores = q @ k.transpose(-1, -2) / sqrt(head_dim)
        attn_scores = attn_scores + attention_mask
        attn_probs = softmax(attn_scores)
        out = attn_probs @ v

        out = merge_heads(out)
        out = self.o_proj(out)
        return out
```

---

## 27. 常见易混点澄清

### 27.1 Decoder-only 里的 Decoder 和原始 Transformer Decoder 一样吗

不完全一样。

原始 Transformer 是 Encoder-Decoder 架构，Decoder 中有两类 attention：

1. masked self-attention；
2. cross-attention，attend 到 encoder 输出。

Decoder-only LLM 没有 encoder，所以通常没有 cross-attention，只保留 causal self-attention。

### 27.2 为什么叫 Decoder-only，不叫 Encoder-only

因为它是生成式模型，需要从左到右生成下一个 token。

Encoder-only 模型，例如 BERT，更适合理解类任务，它可以双向看上下文，但不能天然自回归生成。

### 27.3 Attention 层输出的维度为什么和输入一样

因为 Attention 之后还要接残差连接：

```text
x = x + Attention(x)
```

所以 Attention 输出必须和 x 形状一致，都是 `[B, S, H]`。

### 27.4 MLP 会改变 seq_len 吗

不会。

MLP 只改变每个 token 向量内部的表示，输入输出都是：

```text
[B, S, H]
```

中间会扩展到 intermediate_size，但最后会投影回 hidden_size。

### 27.5 KV Cache 保存的是最终 hidden state 吗

不是。

KV Cache 保存的是每一层 Attention 中投影得到的 K 和 V，不是整层输出的 hidden state。

### 27.6 Decode 阶段为什么只输入一个 token 还能利用完整上下文

因为历史上下文的 K/V 已经保存在 KV Cache 中。

新 token 的 Q 会和历史所有 K 做 attention，所以它仍然能看到完整上下文。

---

## 28. 面试问题与参考答案

### Q1：什么是 Decoder-only LLM？

Decoder-only LLM 是只使用 Transformer Decoder 堆叠而成的自回归语言模型。它通过 causal self-attention 从左到右建模文本序列，每一步根据已有上下文预测下一个 token。GPT、LLaMA、Qwen 等大多数生成式大语言模型都属于 Decoder-only 架构。

### Q2：Decoder-only 和 Encoder-only 有什么区别？

Encoder-only 模型通常使用双向 attention，可以看到完整上下文，适合理解任务，例如分类、检索、表示学习。Decoder-only 使用 causal mask，只能看到当前位置之前的 token，适合自回归生成任务。Decoder-only 可以自然地逐 token 生成文本。

### Q3：Decoder-only 模型为什么需要 causal mask？

因为训练目标是预测下一个 token。如果当前位置能看到未来 token，模型就会作弊，训练目标失效。Causal mask 会屏蔽未来位置，使第 i 个 token 只能 attend 到位置小于等于 i 的 token。

### Q4：Q、K、V 分别代表什么？

Q 是 Query，表示当前 token 想查询什么信息；K 是 Key，表示每个 token 可被匹配的索引特征；V 是 Value，表示每个 token 真正提供的信息内容。Attention 先用 Q 和 K 点积计算相关性，再用相关性权重加权 V。

### Q5：为什么 Attention 要除以 sqrt(head_dim)？

如果 head_dim 较大，QK 点积的数值方差会变大，softmax 可能进入饱和区，导致梯度变小、训练不稳定。除以 sqrt(head_dim) 可以稳定分数尺度。

### Q6：Multi-Head Attention 的意义是什么？

多头注意力允许模型在多个子空间中并行学习不同类型的 token 关系。有的 head 可能关注语法，有的关注实体指代，有的关注局部上下文，有的关注长距离依赖。多个 head 的结果拼接后再投影回 hidden_size。

### Q7：MHA、MQA、GQA 有什么区别？

MHA 中 Q/K/V 都有相同数量的 head，表达力强但 KV Cache 大。MQA 中多个 Q head 共享一组 K/V head，显著减少 KV Cache，但可能损失表达能力。GQA 是折中方案，多个 Q head 分组共享 K/V head，兼顾效果和推理效率。

### Q8：KV Cache 是什么，为什么推理需要它？

KV Cache 是推理过程中缓存历史 token 在每一层 Attention 中的 K 和 V。自回归 decode 时，新 token 需要 attend 到所有历史 token。如果不缓存，每一步都要重新计算历史 K/V，非常浪费。KV Cache 可以把每步计算量降低为只计算新 token 的 Q/K/V，并复用历史 K/V。

### Q9：Prefill 和 Decode 有什么区别？

Prefill 是处理用户 prompt 的阶段，一次性计算 prompt 所有 token 的 hidden states 和 KV Cache，并输出第一个生成 token 的 logits。Decode 是逐 token 生成阶段，每次只输入一个新 token，复用历史 KV Cache，生成下一个 token。Prefill 更偏计算密集，Decode 更偏内存带宽瓶颈。

### Q10：训练时为什么通常不需要像推理那样逐 token decode？

训练时完整文本已经存在，模型可以通过 causal mask 并行计算所有位置的 next-token prediction。推理时未来 token 尚未生成，必须先生成上一个 token 才能继续生成下一个 token，因此 decode 天然串行。

### Q11：Embedding 和 LM Head 有什么关系？

Embedding 把 token id 映射到 hidden 向量，LM Head 把 hidden 向量映射回词表 logits。很多模型会使用 weight tying，让 LM Head 权重和 Embedding 权重共享，以减少参数量并提升输入输出表示的一致性。

### Q12：RMSNorm 和 LayerNorm 有什么区别？

LayerNorm 会减均值并除以标准差，RMSNorm 只用均方根做缩放，不减均值。RMSNorm 计算更简单、参数更少，在现代 LLM 中广泛使用。

### Q13：MLP 层的作用是什么？

Attention 负责 token 之间的信息交互，MLP 负责对每个 token 的表示做非线性变换。MLP 通常先把 hidden_size 扩展到 intermediate_size，再通过激活函数和 down projection 投影回 hidden_size。

### Q14：SwiGLU 相比普通 FFN 有什么特点？

SwiGLU 使用 gate_proj 和 up_proj 两条分支，通过 SiLU(gate) 与 up 分支逐元素相乘形成门控机制，再用 down_proj 投影回 hidden_size。它通常比普通 FFN 表达能力更强，是现代 LLM 常用结构。

### Q15：Decoder-only LLM 的 loss 是怎么计算的？

模型输出每个位置的 logits，然后与右移一位的 labels 计算交叉熵。比如输入 `[10,20,30,40]`，模型在位置 0 预测 20，在位置 1 预测 30，以此类推。loss 是所有有效位置 next-token prediction 的平均交叉熵。

---

## 29. 一句话总结

Decoder-only LLM 的本质是：

> 把文本转成 token ids，经过 embedding 和多层 causal self-attention + MLP，不断更新每个 token 的 hidden state，最后用 LM Head 输出下一个 token 的概率；训练时并行学习 next-token prediction，推理时通过 prefill 建立 KV Cache，再 decode 阶段逐 token 生成文本。

如果你学习 vLLM / nano-vLLM，这套架构就是所有推理优化的基础：KV Cache、PagedAttention、Continuous Batching、Prefill/Decode 拆分、Sampling、Tensor Parallel、Quantization，本质上都是围绕这个 Decoder-only forward 流程做工程优化。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-vllm|模块-vllm]]
- 关联阅读：[[outputs/项目整理/专题-01-请求从输入到输出|专题-01-请求从输入到输出]]

%% 项目关联导航：结束 %%
