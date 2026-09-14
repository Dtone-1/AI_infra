# 图中 Transformer 架构详细分析与面试问答

## 0. 先看这张图在表达什么

这张图是**原始 Transformer 的 Encoder-Decoder 架构**，最早主要用于机器翻译等序列到序列任务，例如：

```text
输入：我 是 一条 狗
输出：I am a dog
```

整张图可以分成三大部分：

```text
左边：Encoder 编码器，负责理解输入句子
右边：Decoder 解码器，负责生成输出句子
顶部：Linear + Softmax，把 Decoder 的输出变成下一个 token 的概率
```

图中的 `N×` 表示：Encoder 层和 Decoder 层不是只做一次，而是重复堆叠 N 层。原始 Transformer 论文中常见设置是 `N=6`。

从宏观上看，完整流程是：

```text
输入文本
  ↓
Input Embedding
  ↓
Positional Encoding
  ↓
Encoder 多层处理
  ↓
得到编码信息 memory
  ↓
Decoder 根据已生成内容 + Encoder 编码信息
  ↓
Linear
  ↓
Softmax
  ↓
输出下一个 token 的概率
```

---

# 1. 图中每个环节详细分析

## 1.1 Inputs：输入文本

图中最底部左侧的 `Inputs` 指的是源语言输入，例如：

```text
我 是 一条 狗
```

但 Transformer 不能直接处理文字。模型内部只能处理数字，所以输入文本需要先经过 tokenizer 转换成 token id：

```text
文本：我 是 一条 狗
token ids：[101, 2769, 3221, 671, 3340, 4318, 102]
```

注意：token id 只是词表编号，本身不是语义向量。真正进入 Transformer 计算之前，还要经过 embedding。

---

## 1.2 Input Embedding：输入端词嵌入

`Input Embedding` 的作用是：

```text
token id → token 向量
```

假设输入序列长度是 `L_src`，每个 token 的向量维度是 `d_model`，那么 embedding 后得到：

```text
X_src.shape = L_src × d_model
```

例如：

```text
输入有 4 个 token
d_model = 512
则输入 embedding 矩阵形状 = 4 × 512
```

Embedding 的本质是一张可训练的查表矩阵：

```text
Embedding Matrix.shape = vocab_size × d_model
```

每个 token id 会取出其中一行，作为该 token 的初始向量。

### 关键理解

Embedding 向量主要表达：

```text
这个 token 是什么
```

但它本身通常不直接表达：

```text
这个 token 在句子中的第几个位置
```

所以后面需要位置编码。

---

## 1.3 Positional Encoding：位置编码

图中 `Positional Encoding` 表示给输入向量加入位置信息。

因为 Transformer 的自注意力机制是并行计算的，它不像 RNN 那样天然按顺序处理 token。如果只给模型 embedding，模型很难知道：

```text
“我 是 一条 狗”
```

和

```text
“狗 一条 是 我”
```

在顺序上的区别。

所以需要位置编码告诉模型：

```text
第 0 个 token 是谁
第 1 个 token 是谁
第 2 个 token 是谁
...
```

最终输入 Encoder 的不是单纯 embedding，而是：

```text
Encoder 输入 = Input Embedding + Positional Encoding
```

如果 embedding 是：

```text
L_src × d_model
```

那么 positional encoding 也必须是：

```text
L_src × d_model
```

这样才能相加，并且相加后形状不变：

```text
L_src × d_model
```

### 为什么相加后位置信息不会消失？

因为相加不是覆盖，而是叠加。相同 token 出现在不同位置，会加上不同位置编码，因此最终向量不同。

例如：

```text
“我”在第 0 位 → embedding("我") + pos_0
“我”在第 3 位 → embedding("我") + pos_3
```

这两个最终向量不同，所以模型可以区分位置。

---

# 2. Encoder 编码器部分

图中左侧大框是 Encoder。Encoder 的作用是：

```text
读取输入序列，提取每个 token 的上下文表示，形成源句子的编码信息。
```

一个 Encoder Layer 包含两个核心子层：

```text
1. Multi-Head Attention
2. Feed Forward
```

每个子层后面都有：

```text
Add & Norm
```

也就是残差连接和层归一化。

---

## 2.1 Encoder 中的 Multi-Head Attention

图中 Encoder 里的 `Multi-Head Attention` 是**多头自注意力机制**。

### 2.1.1 什么是自注意力？

自注意力的特点是：

```text
Q、K、V 都来自同一个输入序列
```

对于 Encoder 来说，输入是源句子 embedding + positional encoding 后的矩阵 `X`。

模型会通过三个可学习矩阵生成 Q、K、V：

```text
Q = XWq
K = XWk
V = XWv
```

然后计算注意力：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

它的含义是：

```text
每个 token 都去看输入句子中的所有 token，
判断哪些 token 对自己更重要，
再按重要程度汇总信息。
```

例如输入：

```text
我 喜欢 打 篮球
```

经过 Encoder self-attention 后：

```text
“我”的新表示     = 融合了“我、喜欢、打、篮球”的信息
“喜欢”的新表示   = 融合了“我、喜欢、打、篮球”的信息
“打”的新表示     = 融合了“我、喜欢、打、篮球”的信息
“篮球”的新表示   = 融合了“我、喜欢、打、篮球”的信息
```

### 2.1.2 为什么是 Multi-Head？

单个注意力头只能从一个角度看关系，多头注意力相当于让模型从多个角度同时看句子。

例如：

```text
Head 1：关注主谓关系
Head 2：关注动宾关系
Head 3：关注长距离依赖
Head 4：关注局部短距离关系
...
```

每个 head 有自己的一套参数：

```text
Head 1: Wq1, Wk1, Wv1
Head 2: Wq2, Wk2, Wv2
...
```

注意：不是每个 token 一套 QKV 矩阵，而是**每个 head 一套 QKV 矩阵；同一个 head 内所有 token 共享这一套矩阵**。

### 2.1.3 多头注意力输出形状

假设：

```text
X.shape = L_src × d_model
head 数量 = h
每个 head 维度 = d_head
通常 d_model = h × d_head
```

每个 head 输出：

```text
L_src × d_head
```

多个 head 拼接：

```text
L_src × (h × d_head) = L_src × d_model
```

再乘一个输出矩阵 `Wo`，得到：

```text
L_src × d_model
```

所以 Encoder 中的多头注意力最终输出形状通常和输入形状一致，方便后续做残差连接。

---

## 2.2 Add & Norm：残差连接 + 层归一化

图中每个子层后面都有 `Add & Norm`。

它可以拆成两部分：

```text
Add：残差连接
Norm：LayerNorm
```

### 2.2.1 Add：残差连接

残差连接的形式大致是：

```text
输出 = 输入 X + 子层输出 Sublayer(X)
```

例如注意力层之后：

```text
X1 = X + MultiHeadAttention(X)
```

为什么要加回原始输入？

因为深层网络训练时容易出现信息丢失、梯度传播困难。残差连接相当于给信息提供一条“直通通道”，让模型即使堆很多层也更稳定。

### 2.2.2 Norm：层归一化

LayerNorm 的作用是稳定每一层输出的数值分布，让训练更稳定。

形式上：

```text
LayerNorm(x) = γ · (x - μ) / σ + β
```

其中：

```text
μ、σ：当前样本特征维度上的均值和标准差
γ、β：可学习参数
```

在 Transformer 中常用 LayerNorm，而不是 BatchNorm，因为序列长度可能变化，LayerNorm 更适合序列模型。

---

## 2.3 Feed Forward：前馈神经网络

Encoder 中第二个核心子层是 `Feed Forward`，也叫 FFN。

它通常对每个 token 的表示单独做非线性变换：

```text
FFN(x) = Linear2(Activation(Linear1(x)))
```

原始 Transformer 中可以理解为：

```text
d_model → d_ff → d_model
```

例如：

```text
512 → 2048 → 512
```

### 注意力和 FFN 的分工

注意力机制负责：

```text
让不同 token 之间交换信息
```

FFN 负责：

```text
对每个 token 已经融合上下文后的表示做进一步加工
```

也就是说：

```text
Attention：跨 token 交流
FFN：单个 token 内部特征变换
```

### 为什么 FFN 输出还要回到 d_model？

因为后面还要做残差连接：

```text
X + FFN(X)
```

如果输入是 `L × d_model`，FFN 最终输出也必须是 `L × d_model`，否则无法相加。

---

## 2.4 Encoder 的整体输出

一个 Encoder Layer 是：

```text
X
  ↓
Multi-Head Self-Attention
  ↓
Add & Norm
  ↓
Feed Forward
  ↓
Add & Norm
```

这个结构重复 `N` 次后，得到最终 Encoder 输出：

```text
Encoder Output / Memory
```

它的含义是：

```text
源句子每个 token 的上下文表示
```

这些表示会被送到 Decoder 的 cross-attention 中，让 Decoder 生成目标语言时参考源句子信息。

---

# 3. Decoder 解码器部分

图中右侧大框是 Decoder。Decoder 的作用是：

```text
根据已经生成的目标 token，以及 Encoder 提供的源句子信息，预测下一个 token。
```

Decoder 比 Encoder 多一个注意力子层，因此一个 Decoder Layer 有三个核心子层：

```text
1. Masked Multi-Head Attention
2. Multi-Head Attention，也就是 Cross-Attention
3. Feed Forward
```

每个子层后面也都有 `Add & Norm`。

---

## 3.1 Outputs shifted right：右移后的输出序列

图中 Decoder 底部写着：

```text
Outputs shifted right
```

这是训练阶段非常重要的概念。

假设目标句子是：

```text
I am a dog <eos>
```

训练时 Decoder 的输入不是直接完整对齐目标，而是右移一位：

```text
Decoder 输入：<bos> I am a dog
预测目标：    I    am a dog <eos>
```

意思是：

```text
看到 <bos>，预测 I
看到 <bos> I，预测 am
看到 <bos> I am，预测 a
看到 <bos> I am a，预测 dog
看到 <bos> I am a dog，预测 <eos>
```

这样训练过程和推理过程保持一致：都是根据前面的 token 预测下一个 token。

---

## 3.2 Output Embedding：输出端词嵌入

Decoder 输入的目标端 token id 也要先经过 embedding：

```text
target token ids → output embedding
```

得到目标端向量矩阵：

```text
Y.shape = L_tgt × d_model
```

注意：在原始 Transformer 图里有 Input Embedding 和 Output Embedding。它们分别用于源语言输入和目标语言输入。

在很多现代大语言模型中，由于通常是 decoder-only 架构，不再有 Encoder 输入和 Decoder 输出两个独立端，但 token embedding 的思想仍然相同。

---

## 3.3 Decoder 里的 Positional Encoding

Decoder 也需要位置编码，因为目标端生成序列同样有顺序。

例如：

```text
<bos> I am a dog
```

每个 token 的位置不同，会加上不同位置编码：

```text
Decoder 输入 = Output Embedding + Positional Encoding
```

这样 Decoder 才知道目标端 token 的顺序。

---

## 3.4 Masked Multi-Head Attention：带因果掩码的自注意力

Decoder 的第一个子层是：

```text
Masked Multi-Head Attention
```

它也是自注意力，因为：

```text
Q、K、V 都来自 Decoder 当前输入序列
```

但它和 Encoder self-attention 的最大区别是：**Decoder 不能看未来 token**。

### 3.4.1 为什么不能看未来？

训练时 Decoder 输入是完整的右移标签：

```text
<bos> I am a dog
```

如果没有 mask，模型在预测 `I` 时可能直接看到后面的 `am a dog`，这相当于考试偷看答案。

所以需要 causal mask。

### 3.4.2 causal mask 的含义

对于目标序列：

```text
<bos> I am a dog
```

注意力可见关系是：

```text
<bos> 只能看 <bos>
I     可以看 <bos>, I
am    可以看 <bos>, I, am
a     可以看 <bos>, I, am, a
dog   可以看 <bos>, I, am, a, dog
```

不能看右侧未来 token。

矩阵上可以理解为只允许看下三角区域：

```text
          <bos>   I    am    a    dog
<bos>      ✓      ×     ×     ×     ×
I          ✓      ✓     ×     ×     ×
am         ✓      ✓     ✓     ×     ×
a          ✓      ✓     ✓     ✓     ×
dog        ✓      ✓     ✓     ✓     ✓
```

### 3.4.3 Masked Attention 的输出

它的输出仍然通常是：

```text
L_tgt × d_model
```

但每个目标 token 的表示只融合了自己和之前 token 的信息，不包含未来 token 信息。

---

## 3.5 Decoder 中的第二个 Multi-Head Attention：交叉注意力

图中 Decoder 中间的 `Multi-Head Attention` 不是普通自注意力，而是**交叉注意力 Cross-Attention**。

### 3.5.1 Cross-Attention 的作用

它负责让 Decoder 读取 Encoder 的输出信息。

机器翻译中，Decoder 生成英文时必须参考中文源句子。例如生成：

```text
dog
```

时，需要知道源句子里有：

```text
狗
```

这就靠 cross-attention。

### 3.5.2 Q、K、V 的来源

Cross-Attention 和 Self-Attention 的核心区别是 Q/K/V 来源不同。

Self-Attention：

```text
Q、K、V 都来自同一个序列
```

Cross-Attention：

```text
Q 来自 Decoder
K、V 来自 Encoder
```

即：

```text
Q = Decoder hidden states × Wq
K = Encoder output × Wk
V = Encoder output × Wv
```

然后仍然计算：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

含义是：

```text
Decoder 当前每个位置，用自己的 Q 去匹配 Encoder 每个位置的 K，
再按匹配程度从 Encoder 的 V 中读取源句子信息。
```

### 3.5.3 Cross-Attention 的形状

假设：

```text
源句子长度 = L_src
目标句子长度 = L_tgt
```

则：

```text
Q.shape = L_tgt × d_k
K.shape = L_src × d_k
V.shape = L_src × d_k
QK^T.shape = L_tgt × L_src
```

注意力矩阵 `L_tgt × L_src` 表示：

```text
目标端每个 token 分别关注源端每个 token 的程度
```

---

## 3.6 Decoder 中的 Feed Forward

Cross-Attention 之后，再进入 Feed Forward。

作用和 Encoder 中一样：

```text
对每个目标 token 的表示做进一步非线性加工
```

流程是：

```text
Masked Self-Attention
  ↓
Add & Norm
  ↓
Cross-Attention
  ↓
Add & Norm
  ↓
Feed Forward
  ↓
Add & Norm
```

这个 Decoder Layer 也重复 `N` 次。

---

# 4. Linear 与 Softmax 输出部分

Decoder 最上方是：

```text
Linear
  ↓
Softmax
  ↓
Outputs Probabilities
```

---

## 4.1 Linear：映射到词表大小

经过 Decoder 多层处理后，每个位置都有一个 hidden state：

```text
H.shape = L_tgt × d_model
```

但是模型最终要预测的是：

```text
下一个 token 是词表中哪个 token
```

假设词表大小是 `Vocab_size`，Linear 层会把每个位置的 hidden state 从 `d_model` 维映射到词表大小：

```text
L_tgt × d_model
  ↓ Linear
L_tgt × vocab_size
```

这个输出通常叫：

```text
logits
```

logits 是每个 token 的未归一化分数。

---

## 4.2 Softmax：变成概率分布

Softmax 把 logits 变成概率：

```text
logits → probabilities
```

例如某一步输出：

```text
I: 0.70
am: 0.10
dog: 0.05
...
```

模型就可以根据概率选择下一个 token。

选择方式可以是：

```text
greedy：选择概率最大的 token
sampling：按概率采样
top-k / top-p：限制候选范围后采样
temperature：控制概率分布尖锐程度
```

---

# 5. 整体流程：训练阶段

以机器翻译为例：

```text
源句子：我 是 一条 狗
目标句子：I am a dog
```

训练时完整流程：

```text
1. 源句子经过 tokenizer 得到 input_ids
2. input_ids 经过 Input Embedding 得到源端向量
3. 源端向量加 Positional Encoding
4. 进入 Encoder，经过 N 层编码，得到 Encoder Output
5. 目标句子右移一位，得到 Decoder 输入：
   <bos> I am a dog
6. Decoder 输入经过 Output Embedding
7. 加 Positional Encoding
8. 进入 Masked Multi-Head Attention，防止看未来
9. 进入 Cross-Attention，读取 Encoder Output
10. 进入 Feed Forward
11. Decoder 层重复 N 次
12. Linear 映射到词表大小
13. Softmax 得到每个位置预测下一个 token 的概率
14. 和真实目标：
   I am a dog <eos>
   计算 loss
15. 反向传播，更新模型参数
```

训练阶段的核心特点是：

```text
Decoder 输入是正确答案右移后的序列
每个位置都可以并行训练
但必须用 causal mask 防止偷看未来
```

---

# 6. 整体流程：推理阶段

推理阶段没有真实答案，只能一个 token 一个 token 生成。

仍以翻译为例：

```text
输入：我 是 一条 狗
```

推理过程：

```text
1. Encoder 先处理源句子，得到 Encoder Output
2. Decoder 初始输入 <bos>
3. Decoder 根据 <bos> 和 Encoder Output 预测第一个 token
4. 假设输出 I
5. 下一步 Decoder 输入 <bos> I
6. 输出 am
7. 下一步 Decoder 输入 <bos> I am
8. 输出 a
9. 下一步 Decoder 输入 <bos> I am a
10. 输出 dog
11. 下一步 Decoder 输入 <bos> I am a dog
12. 输出 <eos>，生成结束
```

可以写成：

```text
<bos> → I
<bos> I → am
<bos> I am → a
<bos> I am a → dog
<bos> I am a dog → <eos>
```

推理阶段的核心特点是：

```text
没有标签
不能一次知道完整输出
必须逐 token 生成
生成出的 token 会作为下一步输入
```

---

# 7. 原始 Transformer 和现代大模型的关系

这张图是原始 Encoder-Decoder Transformer，更适合机器翻译等 seq2seq 任务。

现代 GPT/Qwen/LLaMA 这类大语言模型大多是：

```text
Decoder-only Transformer
```

也就是主要保留右侧 Decoder 中的：

```text
Masked Multi-Head Self-Attention
Feed Forward
LayerNorm / RMSNorm
Linear 输出
```

通常没有左侧独立 Encoder，也没有 Cross-Attention。

它们的核心任务是：

```text
根据前文预测下一个 token
```

所以你在 nano-vLLM 中看到的很多模型层，如 attention、MLP、RMSNorm、RoPE、KV Cache，本质上都是围绕 decoder-only Transformer 推理展开的。

---

# 8. 面试官提问与完整答案

下面的问题按从基础到深入的顺序设计，用来考察你是否真正理解这张 Transformer 架构图。

---

## 问题 1：Transformer 为什么不能直接输入文字？为什么要先做 embedding？

### 标准答案

Transformer 内部只能进行数值计算，不能直接处理“我、是、一条、狗”这样的文字。

文本需要先经过 tokenizer 转成 token id，但 token id 只是词表编号，没有语义大小关系，不能直接当成语义向量使用。因此需要 embedding 层把 token id 映射成高维稠密向量。

完整链路是：

```text
文本 → token → token id → embedding 向量
```

Embedding 的作用是把离散的 token 编号转换成连续向量，使模型可以通过矩阵运算学习 token 之间的语义关系。

---

## 问题 2：Embedding 和 Positional Encoding 为什么要相加？相加后位置信息不会丢失吗？

### 标准答案

Embedding 表示 token 的语义信息，Positional Encoding 表示 token 的位置信息。二者相加后得到同时包含语义和位置的输入表示：

```text
最终输入 = Embedding + Positional Encoding
```

相加不是覆盖，而是叠加。相同 token 出现在不同位置时，会加上不同的位置编码，因此最终输入向量不同。

例如：

```text
"我"在第0位：embedding("我") + pos0
"我"在第3位：embedding("我") + pos3
```

这两个结果不同，所以模型可以区分位置。

相加还有一个好处：保持维度不变。如果拼接，维度会翻倍，后续计算成本更高。

---

## 问题 3：Encoder 中的 Multi-Head Attention 是什么注意力？Q/K/V 分别来自哪里？

### 标准答案

Encoder 中的 Multi-Head Attention 是**多头自注意力机制**。

自注意力的特点是：

```text
Q、K、V 都来自同一个输入序列
```

在 Encoder 中，输入是 embedding 加位置编码后的源句子矩阵 `X`：

```text
Q = XWq
K = XWk
V = XWv
```

然后计算：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

它的作用是让源句子中每个 token 都能关注其他 token，并融合上下文信息。

---

## 问题 4：注意力机制中 Q、K、V 分别可以怎么理解？

### 标准答案

可以用信息检索来理解：

```text
Q：Query，表示当前 token 想找什么信息
K：Key，表示每个 token 提供什么匹配标签
V：Value，表示每个 token 真正携带的内容
```

当前 token 用自己的 Q 去和所有 token 的 K 计算相似度，得到注意力权重，然后用这些权重去加权汇总所有 token 的 V。

所以注意力机制本质是：

```text
用 Q/K 决定关注谁，用注意力权重汇总 V 中的信息。
```

---

## 问题 5：为什么注意力公式里要除以 sqrt(d_k)？

### 标准答案

注意力分数来自点积：

```text
QK^T
```

如果 `d_k` 很大，点积结果可能数值过大。数值过大会让 softmax 输出过于极端，例如一个位置接近 1，其他位置接近 0，这会导致梯度变小，训练不稳定。

除以 `sqrt(d_k)` 是为了缩放分数，使 softmax 更稳定：

```text
softmax(QK^T / sqrt(d_k))
```

主要作用：

```text
1. 防止 softmax 饱和
2. 减少梯度消失风险
3. 提升训练稳定性
```

---

## 问题 6：多头注意力是不是每个 token 用不同的 QKV 矩阵？

### 标准答案

不是。

多头注意力中是：

```text
每个 head 有自己的一套 Wq/Wk/Wv
同一个 head 内，所有 token 共享同一套 Wq/Wk/Wv
```

例如：

```text
Head 1：所有 token 共用 Wq1, Wk1, Wv1
Head 2：所有 token 共用 Wq2, Wk2, Wv2
...
```

每个 token 都会经过所有 head。不同 head 的参数不同，所以可以从不同角度建模 token 之间的关系。

---

## 问题 7：多头注意力最后为什么要 Concat 后再乘 Wo？

### 标准答案

每个注意力头都会输出一份上下文表示。如果有多个 head，就需要把它们合并。

流程是：

```text
head_1, head_2, ..., head_h
  ↓ Concat
拼接后的多头表示
  ↓ 乘 Wo
映射回 d_model
```

`Wo` 的作用是：

```text
1. 融合不同 head 的信息
2. 把维度映射回 d_model
3. 保证输出能和输入做残差连接
```

如果输入是：

```text
L × d_model
```

最终多头注意力输出通常也要是：

```text
L × d_model
```

---

## 问题 8：Attention 输出矩阵的大小和输入矩阵大小一定一样吗？

### 标准答案

要分层次看。

中间矩阵不一样：

```text
Q/K/V：L × d_head
注意力权重 A：L × L
单个 head 输出：L × d_head
```

但标准 Transformer 的多头注意力最终输出通常会回到：

```text
L × d_model
```

和输入矩阵大小一样。

原因是后面要做残差连接：

```text
X + Attention(X)
```

如果形状不同，就无法相加。

所以准确说：

```text
注意力中间矩阵不一定和输入一样；
但完整 attention layer 的最终输出通常和输入一样。
```

---

## 问题 9：Add & Norm 中 Add 是什么？Norm 是什么？为什么需要？

### 标准答案

`Add & Norm` 包含两部分：

```text
Add：残差连接
Norm：层归一化 LayerNorm
```

Add 的作用是把子层输入加回输出：

```text
X + Sublayer(X)
```

这样可以保留原始信息，缓解深层网络训练困难和梯度消失问题。

Norm 的作用是对每层输出做归一化，使数值分布更稳定，从而让训练更稳定、更容易收敛。

Transformer 反复堆叠很多层，所以 Add & Norm 对稳定训练非常重要。

---

## 问题 10：Feed Forward 层的作用是什么？它和 Attention 有什么区别？

### 标准答案

Attention 负责 token 之间的信息交互：

```text
让每个 token 从其他 token 汇总上下文信息
```

Feed Forward 负责对每个 token 的表示做进一步非线性变换：

```text
对已经融合上下文的信息进行加工
```

区别是：

```text
Attention：跨 token 交流
FFN：每个 token 独立进行特征变换
```

FFN 通常结构是：

```text
Linear → Activation → Linear
```

为了做残差连接，FFN 最终输出维度通常仍然是 `d_model`。

---

## 问题 11：Decoder 为什么要有 Masked Multi-Head Attention？

### 标准答案

Decoder 是自回归生成结构，预测当前位置 token 时不能看到未来 token。

训练时 Decoder 输入的是目标句子右移后的序列，例如：

```text
<bos> I am a dog
```

如果没有 mask，模型在预测 `I` 时可能看到后面的 `am a dog`，这相当于偷看答案。

Masked Multi-Head Attention 使用 causal mask，保证：

```text
当前位置只能看自己和之前的位置
不能看未来位置
```

这样训练逻辑才和推理时逐 token 生成保持一致。

---

## 问题 12：Outputs shifted right 是什么意思？为什么训练时要右移？

### 标准答案

`Outputs shifted right` 表示训练时把目标序列右移一位作为 Decoder 输入。

例如目标输出是：

```text
I am a dog <eos>
```

Decoder 输入是：

```text
<bos> I am a dog
```

预测目标是：

```text
I am a dog <eos>
```

这样模型学习的是：

```text
根据前面的 token 预测下一个 token
```

这和推理阶段一致，因为推理时模型也是根据已经生成的 token 预测下一个 token。

---

## 问题 13：Decoder 中第二个 Multi-Head Attention 和第一个有什么区别？

### 标准答案

Decoder 第一个是：

```text
Masked Self-Attention
```

它的 Q/K/V 都来自 Decoder 当前输入，用于目标端内部建模，并且有 causal mask，不能看未来。

Decoder 第二个是：

```text
Cross-Attention
```

它的 Q 来自 Decoder，K/V 来自 Encoder：

```text
Q = Decoder hidden states × Wq
K = Encoder output × Wk
V = Encoder output × Wv
```

它的作用是让 Decoder 在生成目标 token 时读取源句子信息。

---

## 问题 14：Cross-Attention 为什么能连接 Encoder 和 Decoder？

### 标准答案

Cross-Attention 的 Q 来自 Decoder，代表当前目标端位置想查询什么信息；K/V 来自 Encoder，代表源句子中每个位置可供匹配的信息和实际内容。

计算过程：

```text
Decoder Q 与 Encoder K 计算相关性
得到目标端对源端的注意力权重
再用权重汇总 Encoder V
```

这样 Decoder 每生成一个 token，都可以根据当前上下文去源句子里查找相关信息。

所以 Cross-Attention 是 Encoder-Decoder 之间的信息桥梁。

---

## 问题 15：Linear 和 Softmax 在图中起什么作用？

### 标准答案

Decoder 输出的 hidden state 维度是：

```text
L_tgt × d_model
```

但最终要预测下一个 token 属于词表中的哪一个，所以 Linear 层把 hidden state 映射到词表大小：

```text
L_tgt × d_model → L_tgt × vocab_size
```

这个结果叫 logits。

Softmax 再把 logits 转成概率分布：

```text
logits → probabilities
```

最终模型根据概率选择下一个 token。

---

## 问题 16：训练过程和推理过程有什么区别？

### 标准答案

训练时：

```text
Decoder 输入 = 正确答案右移后的序列
目标 = 正确答案
可以并行计算多个位置
用 loss 和反向传播更新参数
需要 causal mask 防止偷看未来
```

推理时：

```text
没有正确答案
从 <bos> 开始
模型生成一个 token，就把它接回输入
逐 token 生成
直到生成 <eos> 或达到最大长度
不更新参数
```

一句话总结：

```text
训练是在老师给正确前缀的情况下学习预测下一个 token；
推理是在没有答案的情况下自己一步步生成 token。
```

---

## 问题 17：Padding mask 和 Causal mask 有什么区别？

### 标准答案

二者都用于屏蔽注意力中的某些位置，但目的不同。

Padding mask 用于屏蔽补齐 token：

```text
我 喜欢 <pad>
```

`<pad>` 不是真实内容，所以不能被关注。

Causal mask 用于屏蔽未来 token：

```text
预测当前位置时，不能看到右边未来位置
```

对比：

| 类型 | 屏蔽对象 | 目的 |
|---|---|---|
| Padding mask | `<pad>` | 忽略假的补齐 token |
| Causal mask | 未来 token | 防止自回归生成偷看答案 |

一句话：

```text
Padding mask 管“哪些 token 是假的”；
Causal mask 管“哪些 token 还没发生”。
```

---

## 问题 18：Encoder 输出的是什么？为什么 Decoder 需要它？

### 标准答案

Encoder 输出的是源句子每个位置的上下文表示，也可以叫 memory。

它包含：

```text
源句子 token 语义信息
token 之间的上下文关系
位置信息
多层 self-attention 提取出的特征
```

Decoder 需要它，因为生成目标语言时必须参考源语言内容。

例如输入中文：

```text
我 是 一条 狗
```

Decoder 生成英文 `dog` 时，需要从 Encoder 输出中读取“狗”对应的信息。这通过 Cross-Attention 完成。

---

## 问题 19：为什么 Transformer 能比 RNN 更好并行？

### 标准答案

RNN 按时间步递推：

```text
第 1 个 token 算完 → 第 2 个 token 才能算 → 第 3 个 token 才能算
```

很难并行。

Transformer 的 self-attention 可以把整个序列组成矩阵，一次性计算所有 token 之间的关系：

```text
QK^T → L × L 注意力矩阵
```

所以训练阶段可以高效利用 GPU 并行矩阵计算。

不过推理阶段对于自回归生成，Decoder 仍然通常需要逐 token 输出，因为下一个 token 依赖前面已经生成的 token。

---

## 问题 20：原始 Transformer 和 GPT/Qwen 这类大模型有什么关系？

### 标准答案

原始 Transformer 是 Encoder-Decoder 架构，适合机器翻译等 seq2seq 任务。

GPT/Qwen/LLaMA 等大模型通常是 Decoder-only Transformer，主要保留：

```text
Masked Self-Attention
FFN / MLP
LayerNorm / RMSNorm
Position Encoding / RoPE
Linear 输出
```

它们没有独立 Encoder，也通常没有 Cross-Attention，核心目标是：

```text
根据前文预测下一个 token
```

所以现代大模型和原始 Transformer 不是完全不同，而是继承了 Transformer 的核心思想，并在结构上做了简化和工程优化。

---

# 9. 最终总结

这张图可以压缩成一句话：

```text
Encoder 负责把输入序列编码成上下文表示；
Decoder 根据已生成的目标端 token 和 Encoder 编码信息，通过 masked self-attention、cross-attention 和 FFN 逐步预测下一个 token。
```

再压缩成核心链路：

```text
文本
→ token id
→ embedding
→ 加位置编码
→ Encoder self-attention + FFN
→ Encoder memory
→ Decoder masked self-attention
→ Decoder cross-attention 读取 memory
→ Decoder FFN
→ Linear
→ Softmax
→ 下一个 token 概率
```

最应该掌握的 6 个关键点：

```text
1. Embedding：把 token id 变成向量
2. Positional Encoding：加入顺序信息
3. Self-Attention：让同一序列内部 token 互相建立联系
4. Multi-Head：从多个角度建模 token 关系
5. Masked Attention：防止 Decoder 偷看未来
6. Cross-Attention：让 Decoder 读取 Encoder 信息
```

如果你能把这 6 点讲清楚，并能区分训练和推理，那么就说明你已经基本理解了这张 Transformer 架构图。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-transformer|模块-transformer]]

%% 项目关联导航：结束 %%
