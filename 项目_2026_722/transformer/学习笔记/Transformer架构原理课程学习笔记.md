# Transformer 架构原理课程学习笔记（2.1—2.18）

> 适用对象：之前没有系统学过深度学习，但想把 Transformer 架构、注意力机制、训练过程和推理过程真正串起来的学习者。  
> 使用方式：打开课程 PDF，对照每一节对应的课件页阅读。本文不是简单摘要，而是把“课件图 + 老师讲解 + 小白补充概念”合并成一份可以复习的听课式笔记。

---

## 0. 先建立一条主线：Transformer 到底在做什么？

Transformer 最早是为机器翻译这类序列任务设计的。比如输入中文：

```text
我 是 一条 狗
```

模型要输出英文：

```text
I am a dog
```

但模型不能直接理解“我、是、一条、狗”这些文字。模型内部只能处理数字，所以整个过程可以先粗略理解成：

```text
自然语言文本
  ↓ 分词 Tokenize
Token / Token ID
  ↓ 词嵌入 Embedding
词向量矩阵
  ↓ 加上位置编码 Positional Encoding
带语义 + 位置信息的输入矩阵
  ↓ Encoder 提取输入句子的整体信息
编码信息
  ↓ Decoder 根据编码信息逐个生成目标语言 token
输出 token 概率
  ↓ 选出概率最大的 token 或按采样策略选 token
输出文本
```

这门课的核心不是让你背公式，而是让你理解：

1. **文字怎么变成模型能计算的数字**；
2. **自注意力机制怎么计算 token 之间的关联关系**；
3. **多头注意力为什么比单个注意力更强**；
4. **掩码为什么能防止模型“偷看未来答案”**；
5. **Encoder 和 Decoder 如何组合成完整 Transformer**；
6. **训练时和推理时为什么不一样**。

---

## 0.1 课程章节与课件页对应关系

| 课程 | 主题 | 对应课件页 | 本节在整体中的位置 |
|---|---|---:|---|
| 2.1 | Transformer 算法背景 | 第 1 页 | 为什么需要 Transformer |
| 2.2 | Transformer 模型整体架构 | 第 2 页 | 先看完整架构图 |
| 2.3 | 输入文本数据数值转换 | 第 3 页 | 文本变数字的第一步 |
| 2.4 | 词嵌入方法 | 第 4 页 | Token ID 变成稠密向量 |
| 2.5 | 位置编码 | 第 5 页 | 给词向量补充顺序信息 |
| 2.6 | 自注意力机制先导 | 第 6 页 | 引出注意力在做什么 |
| 2.7 | 自注意力机制算法流程 | 第 6—7 页 | 按 Q/K/V 逐步理解注意力 |
| 2.8 | 自注意力机制矩阵理解方式 | 第 8 页 | 用矩阵公式统一表达 |
| 2.9 | 多头注意力机制 | 第 9 页 | 多组注意力并行计算再融合 |
| 2.10 | 掩码注意力机制：填充掩码 | 第 10 页 | 处理 batch 中不同长度句子 |
| 2.11 | 层归一化 | 第 11 页 | 稳定训练，配合残差连接 |
| 2.12 | 前馈神经网络 | 第 12 页 | Transformer block 中的 FFN |
| 2.13 | Transformer 推理过程 | 第 13 页 | 模型如何一个 token 一个 token 输出 |
| 2.14 | Transformer 训练过程 | 第 14 页 | 模型如何用标签和 loss 学参数 |
| 2.15 | 因果掩码注意力：训练过程 | 第 15 页 | 训练时防止看到未来 token |
| 2.16 | 因果掩码注意力：推理过程 | 第 16 页 | 推理时逐步增长序列 |
| 2.17 | 交叉注意力机制 | 第 17—19 页 | Decoder 如何读取 Encoder 信息 |
| 2.18 | Transformer 架构总结 | 第 20 页 | 把所有模块拼成完整模型 |

---

## 0.2 小白必须先知道的几个基础概念

### 0.2.1 Token 是什么？

Token 可以理解为模型处理文本的最小单位。它可能是一个字、一个词、一个词的一部分，甚至是标点。课程里为了方便讲解，把“我 是 一条 狗”切成：

```text
[我, 是, 一条, 狗]
```

真实大模型里会用更复杂的 tokenizer，但核心思想一样：**先把文本切成 token，再把 token 变成数字 ID**。

### 0.2.2 向量是什么？

向量就是一串数字。例如一个 token 可以被表示成：

```text
[0.12, -0.35, 0.88, ..., 0.07]
```

如果这个向量有 512 个数字，就说它是 512 维向量。Transformer 原论文中常用的词向量维度是 `d_model = 512`。课程中也多次用 512 这个数来帮助理解。

### 0.2.3 矩阵是什么？

矩阵可以理解为“很多行很多列的数字表”。一句话有多个 token，每个 token 又有一个向量，所以一句话整体就可以表示成一个矩阵：

```text
序列长度 L × 向量维度 d
```

比如句子有 4 个 token，每个 token 是 512 维，那么输入矩阵形状就是：

```text
4 × 512
```

### 0.2.4 权重 W 是什么？

课程里反复出现 `Wq`、`Wk`、`Wv`、`Wo` 等符号。它们都是模型中的**可学习参数**。你可以先把它理解成“模型内部的一张数字表”。

一开始这些数字可能是随机初始化的，训练时模型根据预测错误的程度，也就是 loss，通过反向传播不断调整这些 W，最后这些 W 就能把输入映射成更有用的表示。

### 0.2.5 Softmax 是什么？

Softmax 的作用是把一组分数变成一组概率，并且所有概率加起来等于 1。

例如原始分数：

```text
[2.0, 1.0, 0.1]
```

Softmax 后可能变成：

```text
[0.66, 0.24, 0.10]
```

在注意力机制里，Softmax 把 token 之间的相关性分数变成“注意力权重”。在模型最终输出时，Softmax 把词表上每个 token 的分数变成“下一个 token 的概率”。

### 0.2.6 Loss 和反向传播是什么？

训练模型时，模型会输出预测结果，真实标签是正确答案。预测结果和正确答案之间的差距叫做 loss。loss 越大，说明模型错得越厉害。

反向传播就是根据 loss 去调整模型参数，让下一次预测更接近正确答案。你可以暂时不用掌握数学推导，只需要知道：

```text
预测结果错了 → 计算 loss → 反向传播 → 更新 W、b 等参数 → 下次预测更准
```

---

# 2.1 Transformer 算法背景

对应课件：第 1 页《Transformer算法背景》

## 本节核心

本节是在回答：**为什么 Transformer 会出现？它解决了什么问题？为什么它后来能成为大模型的基础？**

Transformer 由 Google 研究团队提出，论文发表于 2017 年，标题是非常有名的：

```text
Attention Is All You Need
```

这句话的意思不是“只有注意力就够了”这么简单，而是强调：在序列建模任务中，以注意力机制为核心的架构可以摆脱传统循环神经网络 RNN/LSTM 对时间步递推的依赖。

## 课件内容对应

课件第 1 页强调 Transformer 的几个关键点：

- 使用 **Encoder-Decoder** 结构；
- 使用三类注意力：编码器多头自注意力、解码器多头自注意力、交叉注意力；
- 使用位置编码补充顺序信息；
- 使用残差连接、层归一化、前馈网络组成稳定的层块；
- 影响了 BERT、大模型、ViT、多模态大模型等后续模型。

## 为什么之前的 RNN/LSTM 不够好？

自然语言是有顺序的。例如：

```text
我 喜欢 打 篮球
```

如果顺序乱掉：

```text
篮球 打 喜欢 我
```

人会觉得不通顺。早期处理这种序列任务常用 RNN 或 LSTM。它们的特点是：

```text
第 1 个 token → 模型处理 → 得到状态
第 2 个 token → 依赖第 1 步状态继续处理
第 3 个 token → 依赖第 2 步状态继续处理
...
```

这会带来两个问题：

第一，**并行性差**。因为第 2 步要等第 1 步算完，第 3 步要等第 2 步算完，很难一次性并行处理整个句子。

第二，**长距离依赖建模困难**。如果一个句子很长，前面的重要信息可能在不断递推中被弱化或遗忘。

Transformer 的自注意力机制可以让一个 token 直接和句子中所有 token 计算关系，因此更适合建模长距离依赖，并且矩阵计算可以在 GPU 上高效并行。

## 小白理解

你可以把 RNN 想成“排队逐个读词”，把 Transformer 想成“一次把整句话摊开看，每个词都可以直接看其他词”。

例如翻译一句话时，某个词的含义可能依赖很远处的另一个词。Transformer 不需要一步步把信息传过去，而是通过注意力直接建立关联。

## 本节结论

Transformer 的出现，本质上是为了解决序列任务中的两个核心问题：

```text
1. 如何更并行地处理序列？
2. 如何更好地捕捉远距离 token 之间的关系？
```

它的答案就是：

```text
以注意力机制为核心，配合位置编码、残差连接、层归一化和前馈网络，构建可堆叠的 Encoder-Decoder 架构。
```

---

# 2.2 Transformer 模型的整体架构

对应课件：第 2 页《Transformer整体架构图》

## 本节核心

本节先让你从宏观上看到完整 Transformer 长什么样。不要一开始就被图吓到。整张图可以拆成两大部分：

```text
左边：Encoder 编码器
右边：Decoder 解码器
```

机器翻译任务中，Encoder 负责理解输入句子，Decoder 负责生成输出句子。

## Encoder 做什么？

以中文翻译英文为例：

```text
输入：我 是 一条 狗
```

Encoder 接收这句话的词向量和位置编码，然后通过多层网络提取信息。你可以把 Encoder 理解成：

```text
把原始输入句子压缩/转换成一组包含语义、关系、上下文的信息表示。
```

Encoder 的基本层块包括：

```text
输入词向量 + 位置编码
  ↓
多头自注意力机制
  ↓
Add & Norm（残差连接 + 层归一化）
  ↓
前馈神经网络 FFN
  ↓
Add & Norm
```

这个层块会重复 N 次。原始 Transformer 论文中 N=6。

## Decoder 做什么？

Decoder 负责生成目标语言。例如输出英文：

```text
I am a dog
```

它不是一次性把整个句子吐出来，而是一个 token 一个 token 生成：

```text
<bos> → I
<bos> I → am
<bos> I am → a
<bos> I am a → dog
<bos> I am a dog → <eos>
```

其中：

- `<bos>` 表示开始符，告诉 Decoder “开始生成”；
- `<eos>` 表示结束符，告诉模型“生成结束”。

Decoder 的基本层块比 Encoder 多一层注意力：

```text
输出端已生成 token 的词向量 + 位置编码
  ↓
带因果掩码的多头自注意力机制
  ↓
Add & Norm
  ↓
交叉注意力机制：读取 Encoder 输出的信息
  ↓
Add & Norm
  ↓
前馈神经网络 FFN
  ↓
Add & Norm
  ↓
Linear + Softmax 输出词表概率
```

## 三种注意力分别在哪里？

第 2 页课件图里有三处注意力：

1. **Encoder 的 Multi-Head Attention**：编码器多头自注意力。输入来自同一句源语言句子。
2. **Decoder 的 Masked Multi-Head Attention**：解码器带因果掩码的多头自注意力。只能看当前位置之前的目标 token，不能看未来。
3. **Decoder 的 Multi-Head Attention**：交叉注意力。Query 来自 Decoder，Key/Value 来自 Encoder，用来让 Decoder 读取源语言信息。

## 为什么 Decoder 需要 shifted right？

课件图中 Decoder 输入下方写着 `Outputs shifted right`。意思是训练时 Decoder 的输入不是完整目标句子原样对齐，而是右移一位：

```text
正确输出：I    am    a    dog    <eos>
Decoder输入：<bos> I     am   a      dog
```

也就是说，模型看到 `<bos>` 时要预测 `I`，看到 `<bos> I` 时要预测 `am`。这和真实推理时“根据前面生成的词预测下一个词”的过程一致。

## 本节结论

完整 Transformer 可以先记成一句话：

```text
Encoder 负责理解输入句子，Decoder 负责在 Encoder 信息的帮助下，自回归地逐个生成输出 token。
```

---

# 2.3 输入文本数据数值转换

对应课件：第 3 页《Transformer模型的输入数据》

## 本节核心

本节回答：**文字怎样变成模型能计算的数字？**

课件强调两点：

```text
1. Transformer 里的计算都是数值。
2. 文本进入模型前要数字化，模型输出后再把数字还原成文字。
```

## 从文本到 token

以课程中的例子：

```text
我 是 一条 狗
```

第一步是收集大量文本，第二步是把句子切成 token：

```text
[我, 是, 一条, 狗, 篮球, 鸡, ...]
```

然后统计每个 token 出现频率，建立词表。词表可以理解成一个映射表：

```text
我   → 1
是   → 2
一条 → 3
狗   → 4
篮球 → 5
鸡   → 6
...
```

真实模型中 token ID 不一定按词频这样简单排序，但课程里这样讲是为了让你理解：**每个 token 最终都会对应一个数字编号**。

## 为什么不能直接用 1、2、3、4 表示词？

假设：

```text
我 = 1
是 = 2
一条 = 3
```

如果直接把这些数字输入模型，就会出现语义上的荒谬：

```text
1 + 2 = 3
```

从数学上成立，但从语言上不代表：

```text
我 + 是 = 一条
```

所以 token ID 只是编号，不能直接当作有语义的数值来做加减乘除。

## 独热向量 One-hot

为了解决“数字编号有大小关系”的问题，可以把 token 转成独热向量。例如词表里有 6 个词：

```text
[我, 是, 一条, 狗, 篮球, 鸡]
```

那么：

```text
我   → [1,0,0,0,0,0]
是   → [0,1,0,0,0,0]
一条 → [0,0,1,0,0,0]
狗   → [0,0,0,1,0,0]
```

独热向量的好处是：每个词只是某个位置为 1，没有人为引入“我比是小”“狗比一条大”这种没有意义的数值关系。

## 独热向量的问题

课件中提到，如果词表大小 `V = 50k`，也就是 5 万个 token，那么每个 token 的独热向量都是 5 万维，而且只有一个位置是 1，其余全是 0。

这会带来两个问题：

1. **维度巨大，计算浪费**；
2. **无法表达语义相似性**。

例如“篮球”和“足球”语义上比“篮球”和“鸡”更接近，但在 one-hot 表示里，它们之间没有自然的相似关系。

## 本节结论

文本进入 Transformer 前必须数字化，但直接用 token ID 不合理，one-hot 又太稀疏、维度太高。因此后面要引出更好的表示方法：**词嵌入 Word Embedding**。

---

# 2.4 词嵌入方法 Word Embedding

对应课件：第 4 页《Transformer模型的输入——词嵌入》

## 本节核心

本节回答：**如何把高维稀疏的 one-hot 向量变成低维稠密、有语义关系的词向量？**

词嵌入可以理解成一张表：

```text
Embedding Matrix: V × d
```

其中：

- `V` 是词表大小；
- `d` 是每个 token 的向量维度；
- 原始 Transformer 中常用 `d = 512`。

## 词嵌入矩阵是什么？

假设词表：

```text
我, 是, 一条, 狗, 篮球, 鸡, 跳舞, ...
```

词嵌入矩阵可以理解成：

```text
第 1 行：我 的向量
第 2 行：是 的向量
第 3 行：一条 的向量
第 4 行：狗 的向量
第 5 行：篮球 的向量
...
```

如果 `d = 512`，那么每一行就是 512 个数字。

## 从 one-hot 到 embedding 的本质

课程中用矩阵乘法解释：one-hot 向量乘以词嵌入矩阵，最后相当于“取出某一行”。

例如：

```text
篮球 的 one-hot = [0,0,0,0,1,0,...]
```

它乘以 embedding matrix 后，只会保留“篮球”对应那一行向量。

所以在代码实现里，embedding 本质上经常就是一次查表：

```text
token_id → embedding_matrix[token_id]
```

## 词向量为什么有语义？

词嵌入矩阵里的数值不是人工手写出来的，而是训练出来的。经过大量数据训练后，语义相近的词会在向量空间里更接近。

例如课件右侧用二维图示意：

- “篮球、足球、乒乓球、网球”会聚在一起；
- “鸡、鸭、猪、羊、牛、鱼”会聚在一起；
- “我、你、他、她、我们”会聚在一起；
- “个、只、条、件、辆、位”这类量词也会聚在一起。

这只是把高维向量降到二维后的可视化，真实模型里的向量通常是几百维甚至几千维。

## 小白理解

one-hot 像身份证号，只能区分“谁是谁”；embedding 像个人画像，不仅能区分每个词，还能表达词之间的关系。

比如：

```text
篮球 和 足球 都是运动项目 → 向量更接近
篮球 和 鸡 关系较远 → 向量更远
```

## 本节结论

词嵌入解决了 one-hot 的两个问题：

```text
1. 把高维稀疏向量变成低维稠密向量；
2. 让词向量具有可学习的语义结构。
```

Transformer 的输入不是原始文字，而是每个 token 对应的 embedding 向量。

---

# 2.5 位置编码 Positional Encoding

对应课件：第 5 页《位置编码》

## 本节核心

本节回答：**既然 Transformer 一次性并行处理整句话，它怎么知道词的顺序？**

RNN 是按顺序一个一个读 token，所以天然知道顺序。Transformer 的自注意力是并行计算，如果只输入词向量，模型并不知道“我 是 一条 狗”中每个词的位置。

因此需要加上位置编码。

## 为什么词向量本身不包含顺序？

假设有四个 token：

```text
我 是 一条 狗
```

词嵌入只告诉模型每个 token 的语义，但不告诉模型它处在第几个位置。也就是说，词向量本身不知道：

```text
“我”在第 0 个位置
“是”在第 1 个位置
“一条”在第 2 个位置
“狗”在第 3 个位置
```

所以要额外构造一个位置向量，并与词向量相加：

```text
最终输入 = 词向量 + 位置编码
```

## 位置编码公式

课程中使用原始 Transformer 的正弦/余弦位置编码：

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

其中：

- `pos` 表示当前 token 在序列中的位置；
- `i` 表示向量维度中的索引；
- `d` 表示词向量维度。

公式看起来复杂，但你只需要先理解：**它会根据 token 的位置生成一个与词向量同维度的位置向量**。

## 为什么位置编码要和词向量同维度？

因为它们要相加。如果词向量是 512 维，位置编码也必须是 512 维：

```text
embedding(token)      : 1 × 512
position_encoding(pos): 1 × 512
相加后仍然是          : 1 × 512
```

这样每个 token 的最终输入向量同时包含：

```text
语义信息 + 位置信息
```

## 课程例子

课件中举例：

```text
“我 是 一条 狗”
pos = 0, 1, 2, 3
```

维度很小时可以算出具体位置编码数值。真实模型中维度可能是 512，位置编码就是一个 512 维向量。

## 小白理解

词向量像“这个词是什么意思”，位置编码像“这个词在第几个位置”。

最终送进 Transformer 的不是单纯的“词”，而是：

```text
这个词是什么意思 + 它在句子中的位置
```

## 本节结论

Transformer 没有 RNN 那样的天然顺序结构，所以必须通过位置编码把顺序信息注入输入向量。

---

# 2.6 自注意力机制先导

对应课件：第 6 页《自注意力机制》

## 本节核心

本节开始进入 Transformer 最重要的部分：**自注意力机制 Self-Attention**。

课程反复强调：Transformer 论文叫《Attention Is All You Need》，足以说明注意力机制在整个架构中的核心地位。

## 自注意力到底想解决什么问题？

一句话里，不同 token 之间不是孤立的。例如：

```text
我 喜欢 打 篮球
```

“篮球”和“打”的关系很强，因为“打篮球”是一个常见组合；“我”和“喜欢”也有关系，因为“我”是动作或情感的主体。

自注意力机制要做的事情就是：

```text
计算每个 token 和其他 token 的关联程度，
再根据关联程度汇聚其他 token 的信息，
生成新的上下文表示。
```

## 什么叫“自”注意力？

“自”表示 Query、Key、Value 都来自同一个输入序列。

比如 Encoder 中输入：

```text
[我, 喜欢, 打, 篮球]
```

每个 token 都会看同一句话里的所有 token，包括它自己。因此叫 self-attention。

## 一句话理解

自注意力机制不是简单地逐词处理，而是让每个词都问一遍：

```text
在理解我这个词时，句子里的其他词分别有多重要？
```

例如理解“篮球”时，“打”可能权重更高；理解“我”时，“喜欢”可能权重更高。

## 本节结论

自注意力机制的核心目标是：

```text
为每个 token 重新生成一个融合上下文信息的新表示。
```

这就是 Transformer 能理解上下文关系的关键。

---

# 2.7 自注意力机制的算法流程

对应课件：第 6—7 页《自注意力机制》

## 本节核心

本节详细解释自注意力机制如何计算。核心符号是：

```text
Q, K, V
```

分别叫：

- `Q`：Query，查询向量；
- `K`：Key，键向量；
- `V`：Value，值向量。

## Q、K、V 怎么来的？

输入 token 向量记作：

```text
a1, a2, a3, a4
```

它们分别代表：

```text
我, 喜欢, 打, 篮球
```

模型会用三个不同的权重矩阵对输入做线性映射：

```text
q_i = Wq · a_i
k_i = Wk · a_i
v_i = Wv · a_i
```

注意：`Wq`、`Wk`、`Wv` 是三组不同的可学习参数。它们会在训练中被更新。

## 第一步：用 Q 和 K 计算相关性分数

以第一个 token `a1 = 我` 为例，先得到它的 query：

```text
q1
```

再让 `q1` 分别和所有 token 的 key 做点积：

```text
q1 · k1
q1 · k2
q1 · k3
q1 · k4
```

这些分数表示“我”这个 token 对句子中各个 token 的关注程度。

## 第二步：Softmax 变成注意力权重

原始分数还不是概率，需要经过 Softmax：

```text
[score1, score2, score3, score4]
  ↓ softmax
[α1,1, α1,2, α1,3, α1,4]
```

这些 `α` 就是注意力权重，并且加起来等于 1。

例如可能是：

```text
[0.4, 0.3, 0.1, 0.2]
```

表示当前 token 在生成新表示时，分别从四个 token 那里吸收多少信息。

## 第三步：用注意力权重加权求和 V

有了注意力权重后，把它们乘到 value 向量上：

```text
b1 = α1,1 · v1 + α1,2 · v2 + α1,3 · v3 + α1,4 · v4
```

`b1` 就是第一个 token 的新表示。它不再只包含“我”本身的信息，而是融合了整句话中其他 token 的信息。

同理可以得到：

```text
b2, b3, b4
```

最终输出序列：

```text
[b1, b2, b3, b4]
```

## 小白类比

你可以把 Q/K/V 理解成一次“信息检索”：

- Query：我现在想找什么信息？
- Key：每个位置能提供什么标签供我匹配？
- Value：真正要被取走的信息内容。

当前 token 用自己的 Query 去和所有 Key 匹配，得到权重，再按权重从 Value 中汇总信息。

## 本节结论

自注意力机制的流程是：

```text
输入 X
  ↓ 乘 Wq/Wk/Wv
得到 Q/K/V
  ↓ QK 计算相关性
得到 score
  ↓ Softmax
得到注意力权重 A
  ↓ A 与 V 加权求和
得到融合上下文的新表示
```

---

# 2.8 自注意力机制：矩阵方式理解

对应课件：第 8 页《自注意力机制（矩阵方式理解）》

## 本节核心

前面是从单个 token 角度理解注意力。本节把它统一成矩阵公式。

## 输入矩阵形状

假设输入句子是：

```text
我 喜欢 打 篮球
```

序列长度 `L = 4`，每个 token 的维度是 `d`，则输入矩阵为：

```text
X: L × d
```

例如：

```text
X: 4 × 512
```

## 线性投影

用三个权重矩阵得到 Q、K、V：

```text
Q = XWq
K = XWk
V = XWv
```

如果每个注意力头的维度是 `d_k`，则：

```text
Wq: d × d_k
Wk: d × d_k
Wv: d × d_k
Q : L × d_k
K : L × d_k
V : L × d_k
```

## 注意力公式

完整的缩放点积注意力公式是：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

对应步骤：

```text
QK^T             → 得到 L × L 的相关性分数矩阵
除以 sqrt(d_k)   → 防止数值过大
softmax          → 得到注意力权重矩阵 A
A V              → 加权汇聚 Value 信息
```

## A 矩阵是什么？

`A` 是注意力权重矩阵，形状是：

```text
L × L
```

每一行表示“某个 token 看所有 token 的注意力分布”。

例如课件中的权重矩阵可以理解为：

```text
          我   喜欢   打   篮球
我       0.45 0.25 0.15 0.15
喜欢     0.20 0.35 0.30 0.15
打       0.12 0.26 0.32 0.30
篮球     0.08 0.16 0.36 0.40
```

这表示：

- “我”更多关注自己和“喜欢”；
- “篮球”更多关注“打”和自己。

## 为什么要除以 sqrt(d_k)？

如果 `d_k` 很大，`QK^T` 的点积结果可能很大。Softmax 遇到特别大的数时，输出会非常接近 one-hot：一个位置接近 1，其他位置接近 0。

这会导致梯度很小，训练不稳定。

所以除以 `sqrt(d_k)` 是为了：

```text
1. 防止 Softmax 饱和；
2. 减少梯度消失风险；
3. 提升训练稳定性和收敛速度。
```

## 本节结论

自注意力最重要的公式就是：

```text
A = softmax(QK^T / sqrt(d_k))
Output = A V
```

理解这个公式，就理解了 Transformer 中最核心的计算。

---

# 2.9 多头注意力机制 Multi-Head Attention

对应课件：第 9 页《多头注意力机制》

## 本节核心

本节回答：**为什么不只做一次注意力，而要做多头注意力？**

多头注意力就是做多组独立的注意力计算，然后把结果拼接并融合。

## 单头注意力的问题

一个注意力头可能只能学到一种关注方式。例如：

- 它可能主要关注近距离关系；
- 也可能主要关注语法结构；
- 也可能主要关注实体词。

但一句话里关系很多，仅靠一个头不一定够。

## 多头注意力的直觉

课件中说：不同头可以在不同子空间、不同相对位置上做对齐。

你可以理解成一个团队分工：

```text
头 1：看短距离依赖
头 2：看长距离依赖
头 3：关注实体
头 4：关注标点和句法
头 5：关注主谓关系
...
```

最后把这些头的结果融合起来，得到更丰富的表示。

## 公式

第 `i` 个头：

```text
head_i = Attention(XWq_i, XWk_i, XWv_i)
```

多个头拼接后再乘一个输出权重矩阵：

```text
MHA(X) = Concat(head_1, ..., head_h) Wo
```

其中：

- `h` 是头数；
- `Wo` 用来把拼接后的结果映射回模型维度；
- 这样输出维度才能继续和输入做残差连接。

## 为什么最后要用 Wo？

假设有 8 个头，每个头输出维度是 `d_k`，拼接后维度就是：

```text
8 × d_k
```

但 Transformer 后续模块希望维度仍然是 `d_model`。所以需要 `Wo` 做一次线性映射：

```text
L × (8d_k) → L × d_model
```

这就是课件里说的：

```text
把多头拼接变成融合，把维度拉回，让后续网络能用，还能和输入做残差。
```

## 注意力机制可以处理不同序列长度

课件第 9 页还强调：注意力机制的输入序列长度可以不同。

比如：

```text
I love you       → 3 个 token
I am a dog       → 4 个 token
```

在单个样本层面，Transformer 可以处理变长序列。但在同一个 batch 中，为了 GPU 并行计算，通常需要 padding 到相同长度，这就引出下一节的填充掩码。

## 本节结论

多头注意力的本质是：

```text
多组 Q/K/V 独立学习不同关系 → 拼接 → 线性融合回模型维度。
```

它让模型能从多个角度理解同一句话。

---

# 2.10 掩码注意力机制：填充掩码 Padding Mask

对应课件：第 10 页《掩码注意力机制——填充掩码》

## 本节核心

本节回答：**同一个 batch 中句子长度不一样时，如何并行计算又不让 padding 干扰模型？**

## 为什么需要 padding？

假设一个 batch 里有三句话：

```text
句子 A：我 喜欢 篮球       长度 3
句子 B：我 是 狗           长度 3
句子 C：我 爱              长度 2
```

GPU 更适合处理规则矩阵。同一个 batch 里通常要求序列长度一致，所以长度 2 的句子要补齐：

```text
我 爱 <pad>
```

这样所有句子长度都是 3，可以组成一个整齐的 batch 矩阵。

## padding 的问题

`<pad>` 是为了凑长度加进去的，它不是真实文本。如果模型在注意力计算中关注 `<pad>`，就会引入无意义信息。

所以需要 padding mask。

## mask 怎么起作用？

注意力分数原本是：

```text
Scores = QK^T / sqrt(d_k)
```

加上 mask 后：

```text
Scores = QK^T / sqrt(d_k) + Mask
```

对于真实 token，mask 位置加 0；对于 padding token，mask 位置加负无穷：

```text
真实位置：+ 0
pad 位置：+ (-∞)
```

然后经过 Softmax：

```text
softmax(-∞) = 0
```

这样 padding 位置的注意力权重就是 0，模型就不会从 `<pad>` 中取信息。

## 小白理解

padding mask 就像告诉模型：

```text
这些位置只是为了凑长度，不是真内容。计算注意力时不要看它们。
```

## padding mask 和因果 mask 的区别

这里讲的是 padding mask，只是屏蔽补齐位置。后面还会讲 causal mask，也就是因果掩码。两者目的不同：

| 掩码类型 | 解决的问题 | 屏蔽对象 |
|---|---|---|
| Padding Mask | batch 中句子长度不同 | `<pad>` 补齐位置 |
| Causal Mask | 生成任务不能偷看未来 | 当前 token 后面的未来 token |

## 本节结论

填充掩码的核心是：

```text
用 -∞ 把 padding 位置的注意力权重变成 0，使模型忽略无意义的补齐 token。
```

---

# 2.11 层归一化 Layer Normalization

对应课件：第 11 页《transformer模型——层归一化》

## 本节核心

本节回答：**为什么 Transformer 中需要 LayerNorm？它和 BatchNorm 有什么区别？**

## 为什么要归一化？

训练神经网络时，如果每层输出的数值分布变化很大，模型优化会更困难。归一化的作用是把数据调整到更稳定的分布，帮助模型更快、更稳定地训练。

课件用 loss 曲面的图来说明：不归一化时，优化路径可能很扭曲；归一化后，优化更容易找到较低 loss 的方向。

## LayerNorm 的公式

课件给出：

```text
LayerNorm(x) = γ · (x - μ) / σ + β
```

其中：

- `μ` 是当前样本在某一层特征维度上的均值；
- `σ` 是当前样本在某一层特征维度上的标准差；
- `γ` 是可学习缩放参数；
- `β` 是可学习偏置参数。

## LayerNorm 归一化的对象

LayerNorm 是对**单个样本的特征维度**做归一化。

假设一个样本在某层输出是：

```text
[x1, x2, x3, x4]
```

LayerNorm 就在这几个特征之间计算均值和方差，然后归一化。

## BatchNorm 和 LayerNorm 的区别

课程中强调：BatchNorm 和 LayerNorm 公式类似，但归一化对象不同。

| 对比项 | BatchNorm | LayerNorm |
|---|---|---|
| 归一化对象 | 一个 batch 中不同样本在同一特征上的值 | 单个样本内部的所有特征维度 |
| 常见场景 | CNN、图像任务 | RNN、Transformer、序列任务 |
| 对 batch 大小敏感吗 | 比较敏感 | 不太依赖 batch 大小 |
| 适合变长序列吗 | 不如 LayerNorm 方便 | 更适合 |

## Add & Norm 是什么？

课件第 2 页架构图里反复出现 `Add & Norm`。其中：

- `Add` 是残差连接；
- `Norm` 是层归一化。

残差连接可以理解为：

```text
输出 = 原始输入 + 子层输出
```

比如：

```text
X + MultiHeadAttention(X)
```

这样做的好处是信息不容易在深层网络中丢失，梯度也更容易传播。

## 本节结论

LayerNorm 负责稳定每一层的数值分布，残差连接负责保留原始信息和稳定深层训练。二者一起构成 Transformer 里的 `Add & Norm`。

---

# 2.12 前馈神经网络 Feed Forward Network

对应课件：第 12 页《transformer模型——前馈神经网络》

## 本节核心

本节回答：**Transformer 中注意力之后为什么还要接前馈神经网络？**

## 什么是前馈神经网络？

课件中定义：前馈神经网络是一种无循环的多层神经网络，信息从输入到输出单向流动。

关键点：

```text
它是一种结构类型，不是某一个固定网络。
```

只要信息从前往后流，没有循环反馈，就可以叫前馈网络。

## Transformer 里的 FFN 在做什么？

注意力机制主要负责 token 之间的信息交互：

```text
这个 token 应该关注哪些 token？
```

FFN 更像是对每个 token 的表示做进一步非线性变换：

```text
把注意力汇聚后的信息再加工、提炼、变换。
```

原始 Transformer 中常见形式是：

```text
FFN(x) = max(0, xW1 + b1) W2 + b2
```

也就是两层线性层，中间加激活函数。现在很多大模型会用 GELU、SwiGLU 等变体，但核心思想仍然是：**对每个位置的表示做非线性变换**。

## 为什么输入输出维度通常保持一致？

Transformer block 中要做残差连接：

```text
x + FFN(x)
```

如果 `x` 是 `L × d`，那么 `FFN(x)` 也需要回到 `L × d`，否则无法相加。

因此 FFN 内部可以先升维再降维，但最终输出维度要和输入一致。

## 小白理解

注意力机制像“把上下文信息拿过来”，FFN 像“对拿过来的信息做进一步加工”。

一个完整 Encoder block 可以理解为：

```text
先让 token 之间互相交流 → 再对每个 token 的信息做加工 → 保持维度不变，方便继续堆叠下一层。
```

## 本节结论

FFN 是 Transformer block 中注意力之后的非线性加工层。它不负责跨 token 交流，跨 token 交流主要由注意力完成；它主要负责增强每个 token 表示的表达能力。

---

# 2.13 Transformer 模型的推理过程

对应课件：第 13 页《Transformer模型的推理过程》

## 本节核心

本节回答：**模型训练好之后，如何真正把一句中文翻译成一句英文？**

推理就是模型实际使用时的生成过程。

## Encoder 先处理输入

以课程中的例子：

```text
输入：我 是 一条 狗
```

先经过：

```text
Tokenize → Embedding → Positional Encoding → Encoder 多层计算
```

得到 Encoder 输出的编码信息。这个编码信息包含了源语言句子的语义、上下文关系和位置信息。

## Decoder 一个 token 一个 token 生成

推理时 Decoder 一开始没有英文输出，所以先输入开始符：

```text
<bos>
```

模型结合 Encoder 编码信息，输出词表上每个 token 的概率。假设概率最大的是：

```text
I
```

于是第一个输出 token 是 `I`。

下一步，把已经生成的内容重新作为 Decoder 输入：

```text
<bos> I
```

模型继续预测下一个 token，假设是：

```text
am
```

继续：

```text
<bos> I am → a
<bos> I am a → dog
<bos> I am a dog → <eos>
```

当模型输出 `<eos>`，说明生成结束。

## 推理过程完整展开

可以写成：

```text
源句子：我 是 一条 狗

Encoder 输出编码信息 H

Step 1: Decoder 输入 <bos>               → 输出 I
Step 2: Decoder 输入 <bos> I             → 输出 am
Step 3: Decoder 输入 <bos> I am          → 输出 a
Step 4: Decoder 输入 <bos> I am a        → 输出 dog
Step 5: Decoder 输入 <bos> I am a dog    → 输出 <eos>
```

最终结果：

```text
I am a dog
```

## 为什么说它是自回归生成？

因为当前输出依赖前面已经生成的 token：

```text
下一个 token = f(源句子编码信息, 前面已经生成的 token)
```

这就叫自回归生成。

## 和大模型生成有什么关系？

你使用 ChatGPT 或其他大模型时，看到文字逐步输出，本质上也是类似思想：模型根据已有上下文不断预测下一个 token。区别是现代大模型通常是 decoder-only Transformer，不再使用完整的 Encoder-Decoder 翻译结构，但“根据前文预测下一个 token”的自回归思想一致。

## 本节结论

推理时，Encoder 对输入句子只编码一次；Decoder 从 `<bos>` 开始，基于 Encoder 信息和已有输出，逐个生成 token，直到生成 `<eos>`。

---

# 2.14 Transformer 模型的训练过程

对应课件：第 14 页《Transformer模型的训练过程》

## 本节核心

本节回答：**Transformer 是怎么通过数据和标签学会翻译的？训练过程和推理过程有什么不同？**

## 训练需要输入和标签

训练机器翻译模型时，需要成对数据：

```text
输入：我 是 一条 狗
标签：I am a dog
```

模型根据输入做预测，然后和标签计算 loss，再通过反向传播更新参数。

## 训练时 Decoder 输入什么？

这是本节最容易混淆的点。

推理时，Decoder 的输入来自模型自己上一步生成的 token：

```text
<bos> → 生成 I
<bos> I → 生成 am
...
```

但训练时不是这样。训练时 Decoder 输入的是**正确标签右移后的序列**：

```text
Decoder 输入：<bos> I am a dog
预测目标：   I     am a dog <eos>
```

这种训练方式叫 teacher forcing。也就是说，训练时模型不会把自己前一步可能生成错的 token 再喂回来，而是直接喂正确答案的前缀。

## 为什么训练时要喂正确标签？

如果训练一开始模型还很差，第一步就生成错了。假设应该生成 `I`，模型却生成了别的词。如果下一步继续把这个错误词喂进去，后面的输入也全错了，训练会变得很混乱。

所以训练时用正确标签前缀，让模型在标准输入条件下学习每一步应该输出什么。

## 既然喂了正确标签，模型会不会偷看答案？

这就是后面因果掩码要解决的问题。

虽然训练时把完整标签序列输入 Decoder，但在计算每个位置时，模型不能看到它后面的 token。

例如预测 `I` 时，模型只能看到 `<bos>`，不能看到 `I am a dog`；预测 `am` 时，只能看到 `<bos> I`，不能看到 `am a dog`。

这靠 causal mask 实现。

## Loss 怎么计算？

每个输出位置都会有一个 loss：

```text
L(I)
L(am)
L(a)
L(dog)
L(<eos>)
```

最终把这些 loss 汇总，然后反向传播更新模型参数，包括：

- Embedding 参数；
- Encoder 中注意力、FFN、LayerNorm 参数；
- Decoder 中 masked attention、cross attention、FFN、LayerNorm 参数；
- Linear 输出层参数。

## 训练和推理对比

| 对比项 | 训练 Training | 推理 Inference |
|---|---|---|
| Decoder 输入 | 正确标签右移后的序列 | 模型自己已经生成的 token |
| 是否并行 | 可以并行计算多个位置 | 通常逐 token 生成 |
| 是否知道正确答案 | 有标签用于算 loss | 没有标签，只能预测 |
| 关键机制 | teacher forcing + causal mask | autoregressive generation |

## 本节结论

训练时模型使用正确标签右移作为 Decoder 输入，并用因果掩码防止偷看未来；推理时模型只能根据自己已经生成的 token 继续生成下一个 token。

---

# 2.15 因果掩码注意力机制：训练过程

对应课件：第 15 页《掩码注意力机制——因果掩码（训练过程中）》

## 本节核心

本节回答：**训练时 Decoder 明明输入了完整正确标签，为什么模型仍然不能看到未来答案？**

答案是：因果掩码 causal mask。

## 什么是因果掩码？

在生成任务里，第 `t` 个位置只能依赖第 `t` 个位置之前的信息，不能依赖未来信息。

例如目标序列：

```text
<bos> I like playing ball
```

预测时应满足：

```text
位置 <bos>：只能看 <bos>
位置 I    ：只能看 <bos>, I
位置 like ：只能看 <bos>, I, like
位置 playing：只能看 <bos>, I, like, playing
位置 ball：可以看前面所有 token
```

不能出现：

```text
预测 I 时看到了 like / playing / ball
```

否则模型就相当于考试偷看答案。

## 因果 mask 如何实现？

和 padding mask 类似，注意力分数先计算：

```text
Scores = QK^T / sqrt(d_k)
```

然后加上 mask：

```text
Scores = QK^T / sqrt(d_k) + Mask
```

因果 mask 是一个上三角 mask：

```text
允许看当前位置及之前：加 0
禁止看未来位置：加 -∞
```

经过 Softmax 后，未来位置的权重变成 0。

## 矩阵直观理解

假设序列长度为 5，注意力矩阵是 `5 × 5`。每一行表示当前 token 看所有 token。

因果 mask 后，大致是：

```text
第 1 行：只能看第 1 个
第 2 行：只能看第 1、2 个
第 3 行：只能看第 1、2、3 个
第 4 行：只能看第 1、2、3、4 个
第 5 行：能看第 1、2、3、4、5 个
```

也就是一个下三角可见区域。

## 训练时为什么仍能并行？

虽然每个位置不能看未来，但这不等于必须一个一个算。因为 mask 矩阵可以一次性加到整个 `L × L` 注意力分数矩阵上，所以训练时仍然能并行计算所有位置的输出。

这是 Transformer 训练效率高的重要原因。

## 本节结论

因果掩码在训练中的作用是：

```text
即使 Decoder 输入了完整正确标签，也强制每个位置只能看到自己和过去，不能看到未来 token。
```

这样训练目标才和推理时的自回归生成逻辑一致。

---

# 2.16 因果掩码注意力机制：推理过程

对应课件：第 16 页《掩码注意力机制——因果掩码（推理过程中）》

## 本节核心

本节回答：**推理时因果掩码怎么工作？和训练时有什么区别？**

## 推理时序列逐步增长

推理开始时，Decoder 输入只有：

```text
<bos>
```

这时序列长度 `L = 1`，没有未来 token，所以 mask 很简单。

生成 `I` 后，下一步输入变成：

```text
<bos> I
```

此时 `L = 2`，第一个位置不能看第二个位置，第二个位置可以看前两个位置。

再生成 `like` 后：

```text
<bos> I like
```

此时 `L = 3`，注意力矩阵变成 `3 × 3`，仍然是下三角可见。

## 为什么推理时看起来更“自然”？

训练时虽然输入完整标签，但要靠 mask 遮住未来。推理时未来 token 根本还不存在，所以从直觉上更容易理解。

但是从实现上，为了统一计算方式，推理时仍然使用 causal mask，保证当前位置不能访问未来位置。

## 推理过程展开

```text
Step 1:
输入：<bos>
能看：<bos>
输出：I

Step 2:
输入：<bos> I
<bos> 只能看 <bos>
I 可以看 <bos>, I
输出：like

Step 3:
输入：<bos> I like
like 可以看 <bos>, I, like
输出：playing
```

每一步都只使用已有 token 的信息。

## 和 AI Infra 中 prefill/decode 的关系

如果你后面学习 vLLM、nano-vLLM，会看到两个词：

- **Prefill**：把已有 prompt 一次性送入模型，计算所有上下文 token 的表示和 KV Cache；
- **Decode**：之后每次只生成一个新 token，并复用前面缓存的 K/V。

这和本节推理过程高度相关。Transformer 自回归生成时，序列不断增长，如果每次都从头算所有 Q/K/V 会很浪费，所以现代推理引擎会用 KV Cache 加速。

## 本节结论

推理时 Decoder 输入由模型已生成 token 构成，序列一步步增长；因果掩码保证模型只能利用过去和当前位置的信息，不能访问未来。

---

# 2.17 交叉注意力机制 Cross-Attention

对应课件：第 17—19 页《交叉注意力机制》

## 本节核心

本节回答：**Decoder 如何读取 Encoder 提取出的源句子信息？**

答案是交叉注意力。

## 自注意力和交叉注意力的区别

自注意力中：

```text
Q, K, V 都来自同一个序列
```

交叉注意力中：

```text
Q 来自 Decoder
K, V 来自 Encoder
```

这是二者最关键的区别。

## 为什么需要交叉注意力？

翻译时，Decoder 正在生成英文，但它必须知道中文源句子的内容。

例如输入中文：

```text
我 喜欢 打 篮球
```

Decoder 当前已经生成：

```text
<bos> I like
```

它要继续生成 `playing` 或 `basketball`，就需要回头读取 Encoder 对中文句子的编码信息。

交叉注意力就是让 Decoder 的每个位置根据自己的 Query，去 Encoder 输出中查找相关信息。

## 交叉注意力计算流程

Encoder 输出记为：

```text
H_encoder
```

Decoder 前一层输出记为：

```text
H_decoder
```

则：

```text
Q = H_decoder Wq
K = H_encoder Wk
V = H_encoder Wv
```

然后仍然使用注意力公式：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

## 矩阵形状理解

假设：

- Encoder 输入长度是 `L_e`；
- Decoder 当前序列长度是 `L_d`；
- 每个头维度是 `d_k`。

则：

```text
Q: L_d × d_k
K: L_e × d_k
V: L_e × d_k
QK^T: L_d × L_e
注意力权重 A: L_d × L_e
AV: L_d × d_k
```

这表示 Decoder 的每个位置都会对 Encoder 的所有位置分配注意力权重。

## 小白理解

自注意力像“同一句话内部互相看”。

交叉注意力像“生成英文时，拿当前英文上下文去中文句子里查资料”。

例如 Decoder 生成 `basketball` 时，它可能会对 Encoder 中的“篮球”给予更高注意力。

## 交叉注意力在 Decoder 中的位置

Decoder 每层通常是：

```text
Masked Self-Attention
  ↓
Add & Norm
  ↓
Cross-Attention
  ↓
Add & Norm
  ↓
FFN
  ↓
Add & Norm
```

第一层 masked self-attention 让目标端 token 之间交流；第二层 cross-attention 让目标端读取源端信息。

## 本节结论

交叉注意力的本质是：

```text
Decoder 用自己的 Q 去匹配 Encoder 的 K，并从 Encoder 的 V 中加权读取源句子信息。
```

它是 Encoder-Decoder 翻译结构中连接源语言和目标语言的关键模块。

---

# 2.18 Transformer 架构总结

对应课件：第 20 页《transformer模型》

## 本节核心

本节把前面所有模块拼成完整 Transformer。

## 完整流程重新串起来

以中文翻译英文为例：

```text
输入：我 是 一条 狗
输出：I am a dog
```

完整过程如下：

```text
1. 输入中文句子
2. 切成 token
3. token 转成 token ID
4. token ID 查 embedding 表，得到词向量
5. 词向量 + 位置编码，得到 Encoder 输入 X
6. X 进入 Encoder Layer
   - 多头自注意力
   - Add & Norm
   - FFN
   - Add & Norm
7. Encoder Layer 重复 N 次，得到编码信息 H
8. Decoder 输入 <bos> 或训练时的 shifted right 标签
9. Decoder 先做 masked multi-head self-attention
10. Decoder 再做 cross-attention，读取 Encoder 信息 H
11. Decoder 再做 FFN
12. 经过 Linear + Softmax 得到词表概率
13. 选择下一个 token
14. 推理时重复 8—13，直到输出 <eos>
```

## Encoder Layer 总结

```text
X
 ↓
Multi-Head Self-Attention
 ↓
Add & Norm
 ↓
Feed Forward Network
 ↓
Add & Norm
 ↓
输出到下一层 Encoder
```

Encoder 的核心是：

```text
让输入句子内部 token 之间充分交互，提取源句子的上下文表示。
```

## Decoder Layer 总结

```text
Y
 ↓
Masked Multi-Head Self-Attention
 ↓
Add & Norm
 ↓
Cross-Attention with Encoder Output
 ↓
Add & Norm
 ↓
Feed Forward Network
 ↓
Add & Norm
 ↓
Linear + Softmax
```

Decoder 的核心是：

```text
根据已经生成的目标 token 和 Encoder 提供的源句子信息，预测下一个 token。
```

## 三种注意力最后再对比一次

| 注意力类型 | Q 来自哪里 | K/V 来自哪里 | 作用 |
|---|---|---|---|
| Encoder Self-Attention | Encoder 输入 | Encoder 输入 | 源句子内部 token 互相交流 |
| Decoder Masked Self-Attention | Decoder 输入 | Decoder 输入 | 目标句子前缀内部交流，不能看未来 |
| Cross-Attention | Decoder | Encoder | Decoder 读取源句子信息 |

## 本节结论

Transformer 不是一堆孤立模块，而是一套有明确分工的系统：

```text
Embedding 负责把 token 变成向量；
Position Encoding 负责加入顺序；
Self-Attention 负责 token 之间的信息交互；
Multi-Head 负责多角度建模关系；
Mask 负责控制哪些位置能看、哪些不能看；
LayerNorm + Residual 负责稳定训练；
FFN 负责进一步加工每个 token 表示；
Cross-Attention 负责连接 Encoder 和 Decoder；
Linear + Softmax 负责输出下一个 token 的概率。
```

---

# 3. 全课程终极总结：从零彻底理解 Transformer

## 3.1 Transformer 的一句话定义

Transformer 是一种以注意力机制为核心的神经网络架构，它把文本 token 转成向量，利用自注意力计算 token 之间的关系，通过多层 Encoder/Decoder 模块提取和生成序列信息。

## 3.2 最重要的一条计算链路

如果只记一条链路，就记这个：

```text
文本
 → token
 → token ID
 → embedding 向量
 → 加 positional encoding
 → Q/K/V
 → attention 权重
 → 加权汇聚 V
 → 多头拼接融合
 → Add & Norm
 → FFN
 → Decoder 输出概率
 → 选出下一个 token
```

## 3.3 Transformer 为什么强？

### 第一，它能并行处理序列

RNN 要一步一步读，Transformer 可以把整句话组成矩阵一次性计算注意力。训练时尤其高效。

### 第二，它能建模长距离依赖

任意两个 token 都可以通过注意力直接建立联系，不需要像 RNN 那样一层层传递。

### 第三，它可以堆叠很多层

残差连接和层归一化让深层训练更稳定，所以可以堆出更大的模型。

### 第四，它的结构非常通用

不仅可以做翻译，也可以做文本生成、文本理解、图像任务、多模态任务等。

## 3.4 训练过程和推理过程最重要的区别

| 问题 | 训练时 | 推理时 |
|---|---|---|
| 输入源句子 | 有 | 有 |
| 目标端输入 | 正确标签右移 | 模型已经生成的 token |
| 是否计算 loss | 是 | 否 |
| 是否更新参数 | 是 | 否 |
| 是否能并行算多个目标位置 | 是，靠 causal mask 防偷看 | 通常逐 token 生成 |
| 什么时候停止 | 一个训练样本算完 | 生成 `<eos>` 或达到最大长度 |

用一句话理解：

```text
训练是在老师给正确前缀的情况下学习预测下一个词；推理是在没有老师的情况下自己一步步生成。
```

## 3.5 Mask 的两个核心用途

### Padding Mask

用于屏蔽 batch 中补齐出来的 `<pad>`：

```text
我 爱 <pad>
```

`<pad>` 不是内容，所以注意力权重应为 0。

### Causal Mask

用于屏蔽未来 token：

```text
预测第 t 个 token 时，不能看 t 后面的 token。
```

它保证模型生成时符合从左到右的因果顺序。

## 3.6 Q/K/V 再用最简单的话解释

不要一开始死记 Q/K/V。可以这样理解：

```text
Q：我想找什么？
K：我有什么标签能被你匹配？
V：我真正携带的信息内容。
```

一个 token 用自己的 Q 去匹配其他 token 的 K，得到注意力权重，再从 V 中按权重取信息。

## 3.7 为什么注意力输出是“上下文表示”？

原始词向量只表示这个词本身。例如“打”的 embedding 只是“打”这个词的向量。

经过注意力后，“打”的新表示会融合“我、喜欢、篮球”等上下文信息，所以它不再是孤立的“打”，而是“在这句话中和篮球相关的打”。

这就是上下文表示。

## 3.8 为什么大模型大多是 Decoder-only？

本课程讲的是原始 Transformer 的 Encoder-Decoder 架构，适合机器翻译。

现在很多大语言模型，例如 GPT 类模型，主要使用 Decoder-only 架构。它们省去了独立 Encoder，只保留带因果掩码的 Decoder 堆叠，用来做：

```text
根据前文预测下一个 token
```

所以你学习这门课后，再学大模型推理时，要重点关注：

- token embedding；
- positional encoding / rotary embedding；
- causal self-attention；
- multi-head attention / grouped-query attention；
- FFN / MLP；
- LayerNorm / RMSNorm；
- KV Cache；
- prefill 和 decode。

---

# 4. 公式速查表

## 4.1 Embedding

```text
token_id → embedding_matrix[token_id]
```

输入：token ID  
输出：词向量

## 4.2 位置编码

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

最终输入：

```text
X = Embedding + PositionalEncoding
```

## 4.3 Q/K/V

```text
Q = XWq
K = XWk
V = XWv
```

## 4.4 Scaled Dot-Product Attention

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

如果有 mask：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + Mask) V
```

## 4.5 Multi-Head Attention

```text
head_i = Attention(XWq_i, XWk_i, XWv_i)
MHA(X) = Concat(head_1, ..., head_h) Wo
```

## 4.6 LayerNorm

```text
LayerNorm(x) = γ · (x - μ) / σ + β
```

## 4.7 Feed Forward Network

```text
FFN(x) = max(0, xW1 + b1) W2 + b2
```

---

# 5. 术语表：小白版

| 术语 | 中文 | 小白解释 |
|---|---|---|
| Token | 词元 | 模型处理文本的基本单位 |
| Token ID | 词元编号 | token 在词表里的数字编号 |
| Vocabulary | 词表 | 所有 token 的集合 |
| One-hot | 独热向量 | 只有一个位置为 1 的超长向量 |
| Embedding | 词嵌入 | token 对应的稠密语义向量 |
| Positional Encoding | 位置编码 | 告诉模型 token 在第几个位置 |
| Sequence Length | 序列长度 | 一句话里有多少个 token |
| d_model | 模型维度 | 每个 token 向量的主维度 |
| Q/K/V | 查询/键/值 | 注意力机制中的三类投影向量 |
| Attention Score | 注意力分数 | token 之间的相关性分数 |
| Attention Weight | 注意力权重 | Softmax 后的关注比例 |
| Multi-Head Attention | 多头注意力 | 多组注意力并行看不同关系 |
| Mask | 掩码 | 强制某些位置不能被注意到 |
| Padding | 填充 | 为了 batch 对齐补出来的 token |
| Causal Mask | 因果掩码 | 防止当前位置看到未来 token |
| LayerNorm | 层归一化 | 稳定每层输出分布 |
| Residual Connection | 残差连接 | 把输入直接加到子层输出上 |
| FFN | 前馈网络 | 对每个 token 表示做进一步加工 |
| Encoder | 编码器 | 理解输入序列 |
| Decoder | 解码器 | 生成输出序列 |
| Cross-Attention | 交叉注意力 | Decoder 读取 Encoder 信息 |
| Linear | 线性层 | 把隐藏表示映射到词表维度 |
| Softmax | 概率归一化 | 把分数转成概率 |
| Loss | 损失函数 | 衡量预测和正确答案差多少 |
| Backpropagation | 反向传播 | 根据 loss 更新模型参数 |
| Inference | 推理 | 模型训练好后实际生成结果 |
| Training | 训练 | 用数据和标签更新模型参数 |
| Autoregressive | 自回归 | 根据前面 token 生成下一个 token |

---

# 6. 最容易混淆的问题集中解释

## 6.1 Token ID 和 Embedding 是一回事吗？

不是。

```text
Token ID 是编号，例如 “狗” → 4。
Embedding 是向量，例如 “狗” → [0.21, -0.18, ..., 0.73]。
```

Token ID 只是查表用的索引，真正进入模型计算的是 embedding 向量。

## 6.2 位置编码为什么是加到词向量上，而不是拼接？

课程讲的是原始 Transformer 的做法：相加。

相加的好处是维度不变：

```text
词向量：512 维
位置编码：512 维
相加后：512 维
```

这样后续网络结构不需要改变。如果拼接，维度会变成 1024，需要额外处理。

## 6.3 注意力权重是人工设置的吗？

不是。注意力权重是由 Q 和 K 计算出来的，而 Q/K 又是由可学习参数 `Wq/Wk` 生成的。训练会调整这些参数，使注意力权重更有用。

## 6.4 多头注意力是不是简单重复 8 次？

形式上像重复，但每个头有自己独立的 `Wq/Wk/Wv`，所以它们学习到的关注方式可以不同。

## 6.5 训练时输入正确答案，为什么不算作弊？

训练时确实输入了正确标签前缀，但因果掩码会遮住未来位置。

例如预测 `am` 时，可以看到 `<bos> I`，但不能看到 `am a dog` 后面的内容。因此不算作弊。

## 6.6 Encoder 和 Decoder 的注意力有什么区别？

Encoder self-attention 可以让每个源语言 token 看整句源语言。  
Decoder masked self-attention 只能看已经生成的目标语言前缀。  
Cross-attention 让 Decoder 从 Encoder 输出中读取源语言信息。

## 6.7 Linear + Softmax 输出的是什么？

输出的是词表中每个 token 作为下一个 token 的概率。

假设词表大小是 50k，那么每一步输出就是一个 50k 维概率分布：

```text
P(token_1), P(token_2), ..., P(token_50000)
```

模型会根据解码策略选择下一个 token。

---

# 7. 面向 AI Infra / 推理引擎学习的衔接理解

你后面如果学习 vLLM、nano-vLLM、KV Cache、prefill/decode，本课程最相关的是以下内容。

## 7.1 Prefill 对应什么？

Prefill 可以理解为推理开始时，把 prompt 里的所有 token 一次性送进 Transformer，计算它们的隐藏状态以及 K/V 缓存。

这对应本课程中的：

```text
输入已有序列 → 做 masked self-attention → 得到上下文表示
```

## 7.2 Decode 对应什么？

Decode 是每次生成一个新 token。生成后把这个 token 接到序列后面，再预测下一个 token。

这对应本课程 2.13 和 2.16 的自回归推理过程：

```text
<bos> → I → am → a → dog → <eos>
```

## 7.3 KV Cache 为什么能加速？

注意力中每一步都需要 K 和 V。自回归生成时，前面 token 的 K/V 不会变。如果每生成一个 token 都重新计算所有历史 token 的 K/V，会很浪费。

KV Cache 就是把历史 token 的 K/V 存起来。下一步 decode 时，只计算新 token 的 Q/K/V，然后复用历史 K/V。

这和本课程 Q/K/V 的理解直接相关。

## 7.4 为什么理解 causal mask 很重要？

大语言模型生成文本时必须从左到右，不能提前看未来 token。causal mask 是 decoder-only LLM 的核心机制之一。

如果你理解了第 15、16 节，后面看 GPT、Qwen、LLaMA 的模型结构会轻松很多。

---

# 8. 自测题：检验是否真正学懂

## 8.1 基础题

1. 为什么 Transformer 不能直接输入文字？
2. Token ID 和 embedding 有什么区别？
3. 为什么 one-hot 向量不适合直接作为模型输入？
4. 位置编码解决了什么问题？
5. 自注意力机制中的 Q、K、V 分别起什么作用？
6. 为什么注意力公式里要除以 `sqrt(d_k)`？
7. 多头注意力为什么要 concat 后再乘 `Wo`？
8. Padding mask 和 causal mask 的区别是什么？
9. LayerNorm 和 BatchNorm 的归一化对象有什么区别？
10. Cross-attention 中 Q、K、V 分别来自哪里？

## 8.2 流程题

请你尝试不看答案，完整说出下面流程：

```text
“我 是 一条 狗” 如何经过 Transformer 输出 “I am a dog”？
```

标准回答应包含：

```text
tokenize → token ID → embedding → positional encoding → encoder self-attention → encoder output → decoder shifted right / <bos> → masked self-attention → cross-attention → FFN → linear → softmax → token 概率 → 逐步生成到 <eos>
```

## 8.3 关键理解题

### 问题：训练时 Decoder 输入完整标签，为什么模型没有偷看未来？

答案：因为 Decoder 的 masked self-attention 使用因果掩码。虽然标签序列整体作为输入，但每个位置的注意力只能看到当前位置及之前的位置，未来位置被加上 `-∞`，经过 Softmax 后权重为 0。

### 问题：为什么 Encoder 不需要 causal mask？

答案：Encoder 的任务是理解完整输入句子，不是自回归生成。翻译时源句子已经完整给出，所以 Encoder 中每个 token 可以看完整源句子。但如果同一个 batch 中有 padding，Encoder 仍然需要 padding mask。

### 问题：为什么 Decoder 需要 cross-attention？

答案：Decoder 只看目标端前缀并不知道源语言句子具体内容。cross-attention 让 Decoder 根据当前生成状态去 Encoder 输出中读取源句子信息，从而生成与源句子对应的目标 token。

---

# 9. 最终记忆版：一页背诵

Transformer 的输入是文本，但模型只能算数字，所以先把文本切成 token，再转成 token ID，再通过 embedding 表查到词向量。词向量只包含语义，不包含顺序，因此要加位置编码。加完后得到输入矩阵 X。

Encoder 中，X 通过多头自注意力计算句子内部 token 之间的关系。每个头都会用 `Q=XWq`、`K=XWk`、`V=XWv` 得到 Q/K/V，再计算 `softmax(QK^T/sqrt(d_k))V`，得到融合上下文的信息。多个头的结果拼接后通过 `Wo` 映射回模型维度。随后经过残差连接、层归一化和前馈神经网络。这样的 Encoder Layer 重复 N 次后，得到源句子的编码信息。

Decoder 中，目标端输入先经过带因果掩码的多头自注意力，保证每个位置只能看自己和之前的 token，不能看未来。然后 Decoder 用自己的 Q 去匹配 Encoder 输出的 K/V，这就是交叉注意力，用来读取源句子信息。再经过 FFN、Add & Norm、Linear 和 Softmax，得到词表上每个 token 的概率。

训练时，Decoder 输入是正确标签右移后的序列，用 causal mask 防止偷看未来，并通过 loss 和反向传播更新参数。推理时，Decoder 从 `<bos>` 开始，根据前面已经生成的 token 一个一个预测下一个 token，直到生成 `<eos>`。

一句话总结：

```text
Transformer = 文本数字化 + 词嵌入 + 位置编码 + 多头注意力 + 掩码控制 + 残差归一化 + 前馈网络 + 自回归生成。
```



%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-transformer|模块-transformer]]

%% 项目关联导航：结束 %%
