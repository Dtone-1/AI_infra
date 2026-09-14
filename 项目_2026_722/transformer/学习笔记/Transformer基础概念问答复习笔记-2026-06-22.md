# 基础概念问答复习笔记

## 整理范围说明

本文件整理的是从最近一次【复习整理起点】之后，到当前对话结束之间的基础概念问答内容。整理方式不是照搬原回答，而是按提问顺序压缩成适合快速复习的版本，重点保留概念关系、易混淆点和核心结论。

---

## 问答复习整理

## 1. 问题：词嵌入矩阵的大小 d=512 怎么理解？每个输入的 token 都会用很多个维度表示吗？d 的大小不同有什么影响？

### 复习版答案

可以理解为：**d 就是每个 token 被表示成多少个数字**。

词嵌入矩阵形状通常是：

```text
V × d
```

其中：

```text
V = 词表大小
d = 每个 token 的向量维度
```

如果 `d=512`，表示每个 token 会被查表变成一个 512 维向量。例如一句话有 4 个 token，则 embedding 后形状是：

```text
4 × 512
```

d 的影响：

- **d 越大**：表达能力越强，可以承载更复杂的语义信息；
- **d 越大**：参数量、计算量、显存占用也越大；
- **d 太小**：模型表达能力不足；
- **d 太大**：模型更重，训练和推理成本更高，也可能浪费。

注意：512 维不是每一维都有明确中文含义，而是模型训练出来的高维语义空间。

一句话记忆：

```text
d 是每个 token 向量的长度；d 越大，表达能力越强，但代价越高。
```

---

## 2. 问题：embedding 和 nano-vLLM 中的 tokenizer 好像有点像，但 tokenizer 好像只是把每个词变成一个数字，这是为什么？

### 复习版答案

两者不是一回事，而是**前后两步**：

```text
文本
  ↓ tokenizer
token ids
  ↓ embedding
token vectors
  ↓ Transformer
```

Tokenizer 的作用：

```text
文字 → token → token ID
```

Embedding 的作用：

```text
token ID → 高维向量
```

例如：

```text
文本：我喜欢篮球
tokenizer 输出：[31, 205, 918]
embedding 输出：3 × 512 的向量矩阵
```

token ID 只是词表编号，类似“学号”，本身没有语义大小关系；embedding 向量才是模型真正拿来计算的语义表示。

一句话记忆：

```text
Tokenizer 负责编号，Embedding 负责把编号查表变成向量。
```

---

## 3. 问题：Transformer 的这一套东西是不是就在 nano-vLLM 的 layers 文件夹里？两者没有区别，只是前后关系？

### 复习版答案

大方向可以这么理解，但要更准确：

```text
Transformer = 模型结构原理
nano-vLLM = 把 Transformer 推理跑起来的工程系统
layers/ = Transformer 核心层的代码实现之一
```

nano-vLLM 的 `layers/` 通常对应 Transformer 里的底层模块，例如：

```text
Attention
MLP / FFN
LayerNorm / RMSNorm
Linear
RoPE
activation
```

但完整模型不一定全部都在 `layers/` 中。一般还会有更上层的模型文件负责把这些层组装起来：

```text
input_ids
  ↓ embed_tokens
hidden_states
  ↓ 多层 Transformer block
hidden_states
  ↓ lm_head
logits
  ↓ sampler
next_token_id
```

所以：

```text
Tokenizer / Scheduler / KV Cache / Block Manager 是推理系统部分
Attention / MLP / Norm / RoPE 是 Transformer 模型层部分
```

一句话记忆：

```text
Transformer 是数学结构；nano-vLLM 是工程实现和推理优化；layers/ 是其中的核心神经网络零件库。
```

---

## 4. 问题：我这样理解注意力机制对不对：每个 token 都有自己的 Q/K/V，然后 Q/K/V 之间运算得到一个和原 token 向量矩阵大小一样的新矩阵，新矩阵中每个 token 和前后所有 token 都有联系。

### 复习版答案

基本正确，但要补两个精确点。

注意力机制流程是：

```text
X
  ↓ 乘 Wq/Wk/Wv
Q, K, V
  ↓ QK^T
注意力分数
  ↓ Softmax
注意力权重 A
  ↓ A × V
新的上下文表示
```

每个 token 都会由自己的输入向量生成 Q、K、V，但不是 Q/K/V 随便互相算，而是：

```text
Q 和 K 算相关性
相关性权重再去加权汇总 V
```

输出矩阵通常和输入矩阵形状一样：

```text
输入：L × d
输出：L × d
```

区别是：输入中每一行主要是单个 token 的表示；输出中每一行已经融合了上下文信息。

需要注意“能看谁”取决于 mask：

- Encoder self-attention：可以看前后所有 token；
- Decoder / GPT / Qwen / nano-vLLM：有 causal mask，只能看自己和之前 token，不能看未来。

一句话记忆：

```text
Attention 的本质是：用 Q 找相关性，用权重汇总 V，得到融合上下文的新 token 表示。
```

---

## 5. 问题：多头注意力中，是不是每个 token 用不同的 QKV 矩阵得到 QKV，不同矩阵代表不同关注方面，最后拼接再乘一个矩阵，输出大小和原始 token 向量矩阵一样？注意力机制输出矩阵大小不是和原始矩阵一样吗？

### 复习版答案

核心修正：

```text
不是每个 token 一套 QKV 矩阵，
而是每个 head 一套 QKV 矩阵。
```

也就是：

```text
Head 1：所有 token 共用 Wq1/Wk1/Wv1
Head 2：所有 token 共用 Wq2/Wk2/Wv2
...
Head h：所有 token 共用 Wqh/Wkh/Wvh
```

每个 token 都会经过所有 head。不同 head 的参数不同，所以可以学习不同关注角度，例如：

```text
有的头关注短距离关系
有的头关注长距离关系
有的头关注语法
有的头关注实体
```

以 `d_model=512, head=8` 为例：

```text
输入 X：L × 512
每个 head 输出：L × 64
8 个 head 拼接：L × 512
再乘 Wo：L × 512
```

所以：

- 单个 head 的输出不一定和原始输入一样大；
- 注意力权重矩阵 `A` 是 `L × L`，也不是原输入大小；
- **多头拼接并经过输出投影 Wo 后，最终 attention 输出通常回到 `L × d_model`**。

为什么要保持一样大？

因为后面要做残差连接：

```text
X + Attention(X)
```

形状必须一致。

一句话记忆：

```text
多头注意力是每个 head 用一套 QKV 从不同角度看同一批 token，最后拼接并映射回原 hidden_size。
```

---

## 6. 问题：embedding 向量再加上 positional encoding，位置信息和词嵌入向量相加后位置信息不就没了吗？怎么还能体现位置？

### 复习版答案

不会没。**相加不是覆盖，而是叠加。**

假设某个词向量是：

```text
embedding("我") = [3.0, 5.0]
```

第 0 个位置的位置编码是：

```text
pos0 = [0.1, 0.2]
```

相加后：

```text
[3.0, 5.0] + [0.1, 0.2] = [3.1, 5.2]
```

如果“我”出现在第 3 个位置，位置编码不同，最终向量也不同：

```text
"我"在第0位 → [3.1, 5.2]
"我"在第3位 → [2.6, 5.7]
```

所以同一个 token 出现在不同位置，送入 Transformer 的向量不同，这就体现了位置。

为什么不用拼接？

```text
embedding: 512维
position: 512维
拼接后: 1024维
相加后: 512维
```

相加可以保持维度不变，便于后续 attention、MLP、残差连接继续使用相同的 `d_model`。

一句话记忆：

```text
Embedding 表示“这个 token 是什么”，Position Encoding 表示“它在哪里”，相加后得到“这个位置上的这个 token”。
```

补充：现代大模型很多不用原始正余弦位置编码相加，而是用 RoPE 等位置编码方式，但目的仍然是让模型知道顺序和相对位置。

---

## 7. 问题：Padding mask 和 causal mask 的区别是什么？

### 复习版答案

二者都是让注意力“不看某些位置”，但屏蔽对象不同。

一句话区分：

```text
Padding mask：屏蔽假的 pad token
Causal mask：屏蔽未来 token
```

### Padding mask

解决 batch 中句子长度不同的问题。

例如：

```text
句子1：我 喜欢 篮球
句子2：我 喜欢 <pad>
```

`<pad>` 只是为了补齐长度，不是真内容，所以模型不应该关注它。

```text
真实 token：可以看
pad token：不能看
```

### Causal mask

解决自回归生成不能偷看未来的问题。

例如：

```text
<bos> I like playing ball
```

预测 `like` 时，只能看：

```text
<bos> I
```

不能看：

```text
playing ball
```

所以 causal mask 通常是遮住注意力矩阵右上角未来区域。

### 对比表

| 对比项 | Padding mask | Causal mask |
|---|---|---|
| 目的 | 忽略补齐 token | 防止偷看未来 |
| 屏蔽对象 | `<pad>` | 当前 token 后面的 token |
| 常见场景 | batch 对齐 | Decoder / GPT / Qwen 自回归生成 |
| 依据 | 哪些位置是补齐的 | 时间顺序 / 因果顺序 |
| 矩阵特点 | pad 在哪遮哪 | 通常遮右上三角 |

两者可以同时存在：

```text
最终 mask = padding mask + causal mask
```

一句话记忆：

```text
Padding mask 管“哪些 token 是假的”；causal mask 管“哪些 token 还没发生”。
```

---

# 最终速记版

```text
Tokenizer：文本 → token ID
Embedding：token ID → 向量
Position Encoding：给向量加入位置信息
Attention：让 token 汇总允许看到的其他 token 信息
Multi-Head：多组 attention 从不同角度看同一句话
Causal Mask：不能看未来
Padding Mask：不能看 pad
nano-vLLM layers：Transformer 核心层的工程实现
```


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-transformer|模块-transformer]]

%% 项目关联导航：结束 %%
