# nano-vLLM 推理全过程小白版梳理

> 目标：不追求逐行代码实现，而是从整体理解角度，把 nano-vLLM 从 `prompt` 到完整回答的推理过程讲清楚。重点解释：prefill、decode、下一个 token 怎么选出来、多个 prompt 高并发时怎么一起处理。

---

# 0. 先用一句话理解 nano-vLLM 在做什么

nano-vLLM 做的事情可以概括为：

> 把用户输入的 prompt 转成 token id，交给大模型计算下一个 token 的概率分布，按照采样规则选出下一个 token，再把这个 token 接到原序列后面，继续重复这个过程，直到遇到 EOS 或达到最大生成长度。

大模型推理不是一次性把整段回答“吐出来”，而是：

```text
prompt
  ↓
预测第 1 个回答 token
  ↓
prompt + 第 1 个回答 token
  ↓
预测第 2 个回答 token
  ↓
prompt + 第 1 个回答 token + 第 2 个回答 token
  ↓
继续预测……
```

也就是说，大模型回答的本质是：

```text
根据前面已经出现的所有 token，预测下一个 token。
```

这就是自回归生成。

---

# 1. 推理过程中的几个核心概念

在看完整流程前，先把几个词讲清楚。

## 1.1 prompt

`prompt` 就是用户输入的文本，例如：

```text
介绍一下 vLLM 的作用
```

在模型内部，模型不能直接处理中文、英文字符串，而是要先经过 tokenizer。

---

## 1.2 tokenizer

tokenizer 的作用是：

```text
文本字符串 → token id 列表
```

例如：

```text
"介绍一下 vLLM 的作用"
```

可能会被转换成：

```text
[151644, 104198, 100181, 364, 53, 43, 44, ...]
```

这里的数字就是 token id。

你可以理解为：

```text
每个模型都有自己的词表。
词表里的每个 token 都有一个编号。
模型真正处理的是这些编号，不是原始文字。
```

---

## 1.3 token

token 不一定等于一个汉字，也不一定等于一个英文单词。

它可能是：

```text
一个汉字
一个英文单词
一个英文单词的一部分
一个标点符号
一个空格组合
一个特殊符号，例如 EOS
```

模型生成回答时，本质上是在生成一个个 token id。

最后再由 tokenizer decode：

```text
token id 列表 → 文本字符串
```

---

## 1.4 logits

模型每次 forward 后，不是直接输出一个 token，而是输出一组分数。

这组分数叫做 logits。

如果词表大小是 150000，那么模型每次会输出大约 150000 个分数：

```text
第 0 个 token 的分数
第 1 个 token 的分数
第 2 个 token 的分数
……
第 149999 个 token 的分数
```

分数越高，说明模型认为这个 token 越可能成为下一个 token。

---

## 1.5 概率分布

logits 还不是概率。

模型会通过 softmax 把 logits 转成概率分布：

```text
logits → softmax → 每个 token 的概率
```

例如模型看到：

```text
中国的首都是
```

它可能认为：

```text
北京：0.80
上海：0.05
南京：0.03
中国：0.01
其他 token：若干小概率
```

然后采样器根据这个概率分布选出下一个 token。

---

## 1.6 EOS

EOS 是 End Of Sequence 的缩写，也就是结束 token。

模型不是提前知道回答什么时候结束，而是在每一步预测下一个 token 时，EOS 也参与竞争。

当模型某一步生成 EOS 时，系统认为：

```text
这条回答结束了。
```

nano-vLLM 中如果 `ignore_eos=False`，生成 EOS 后这个请求就会被标记为 `FINISHED`，并释放对应的 KV Cache。

---

## 1.7 KV Cache

Transformer 每一层 attention 都会计算 Key 和 Value。

如果每生成一个新 token，都重新计算整个 prompt 和所有历史回答 token 的 K/V，会非常慢。

所以推理引擎会把历史 token 的 K/V 保存下来，这个保存区域就叫 KV Cache。

KV Cache 的作用是：

```text
已经算过的历史 token 的 K/V 不重复计算。
生成新 token 时，只需要算新 token 的 K/V，然后和历史 KV Cache 做 attention。
```

这就是为什么 decode 阶段可以每次只输入一个 token。

---

## 1.8 Sequence

在 nano-vLLM 里，一个用户请求会被封装成一个 `Sequence`。

一个 Sequence 保存：

```text
这个请求的 token_ids
prompt 有多长
已经生成了多少 token
当前状态是 WAITING / RUNNING / FINISHED
KV Cache block 映射表 block_table
采样参数 temperature / max_tokens / ignore_eos
```

可以把 Sequence 理解成：

```text
一个请求在推理引擎内部的运行档案。
```

---

# 2. prefill 和 decode 的本质区别

大模型推理分为两个核心阶段：

```text
prefill 阶段：处理 prompt，建立 KV Cache，并生成第一个回答 token。
decode 阶段：基于已有 KV Cache，逐 token 继续生成后续回答。
```

---

# 3. Prefill 阶段详细过程

## 3.1 prefill 解决什么问题？

用户输入 prompt 后，模型需要先完整“读一遍” prompt。

例如 prompt 是：

```text
请解释一下 KV Cache 的作用
```

模型不能直接开始凭空生成答案。

它要先把 prompt 中的所有 token 送进 Transformer，计算这些 prompt token 的隐藏状态，同时为每一层 attention 生成对应的 K/V，并保存进 KV Cache。

所以 prefill 的核心任务是：

```text
1. 处理 prompt token
2. 建立 prompt 对应的 KV Cache
3. 根据 prompt 最后一个位置的 hidden state，预测第一个输出 token
```

---

## 3.2 prefill 输入的是什么？

prefill 输入的是 prompt 的 token id。

例如：

```text
prompt = "介绍一下 vLLM"
```

经过 tokenizer 后：

```text
token_ids = [101, 2456, 9345, 18, 43, 44]
```

这些 token id 会被放进 Sequence：

```text
Sequence.token_ids = [101, 2456, 9345, 18, 43, 44]
Sequence.status = WAITING
Sequence.is_prefill = True
```

然后 Scheduler 会把这个 Sequence 放入 waiting 队列。

---

## 3.3 prefill 之前为什么要分配 KV Cache block？

模型处理 prompt 时，每个 token 都会产生 K/V。

这些 K/V 需要保存到 GPU 显存里的 KV Cache 中。

但是不同请求的 prompt 长度不同，回答长度也不同，如果直接连续分配显存，会很容易产生碎片。

所以 nano-vLLM 像操作系统分页一样，把 KV Cache 切成一块一块的 block。

例如：

```text
block_size = 256
```

意思是一个 block 可以存 256 个 token 的 KV Cache。

如果一个 Sequence 有 600 个 token，那么它大约需要：

```text
ceil(600 / 256) = 3 个 block
```

Sequence 里会有一个 `block_table`：

```text
block_table = [5, 17, 23]
```

含义是：

```text
这个请求的第 0 个逻辑 block 存在物理 KV block 5
这个请求的第 1 个逻辑 block 存在物理 KV block 17
这个请求的第 2 个逻辑 block 存在物理 KV block 23
```

这就是 PagedAttention 的基本思想。

---

## 3.4 prefix cache 是什么？

如果多个请求的 prompt 前缀一样，就可以复用已经算过的 KV Cache。

例如：

```text
请求 A：请你作为 AI Infra 面试官，解释 KV Cache
请求 B：请你作为 AI Infra 面试官，解释 PagedAttention
```

这两个请求前面一大段是一样的。

如果请求 A 已经计算过前缀部分的 KV Cache，请求 B 就可以直接复用这部分，不需要重复计算。

nano-vLLM 通过 token block 的 hash 来判断某个 block 是否已经存在。

可以理解为：

```text
把一段 token block 做哈希。
如果另一个请求的 token block 哈希相同，并且 token 内容也相同，就认为可以复用。
```

这样 prefill 时可能只需要计算没有命中的后半部分。

---

## 3.5 prefill 的调度过程

Scheduler 会先看 waiting 队列。

如果 waiting 队列中有请求，它会优先调度 prefill。

调度时会考虑几个限制：

```text
1. 本轮最多处理多少个请求：max_num_seqs
2. 本轮最多处理多少个 prompt token：max_num_batched_tokens
3. KV Cache block 是否够用
4. prefix cache 命中了多少 block
```

假设：

```text
max_num_batched_tokens = 16384
max_num_seqs = 512
```

含义是：

```text
这一轮 prefill 最多处理 16384 个 token
这一轮最多同时调度 512 条 Sequence
```

注意：

```text
max_num_batched_tokens 限制的是本轮所有 prefill 请求加起来的 token 数，不是一条请求的 token 数。
```

---

## 3.6 chunked prefill 是什么？

如果一个 prompt 太长，一轮 token 预算不够完整处理它，nano-vLLM 允许把它切成几段处理。

例如：

```text
一个 prompt 有 20000 个 token
max_num_batched_tokens = 16384
```

那么第一轮可能只处理前 16384 个 token。

下一轮继续处理剩下的 token。

这就叫 chunked prefill。

在 nano-vLLM 这份代码中，有一个简化策略：

```text
只有当前 batch 的第一条 sequence 允许 chunked prefill。
```

这样可以避免调度逻辑过于复杂。

---

## 3.7 ModelRunner 在 prefill 里准备了什么？

Scheduler 选好本轮要跑的 Sequence 后，会交给 ModelRunner。

ModelRunner 会把多个 Sequence 的 prompt token 拼成一个大的一维 tensor。

例如有三个 prompt：

```text
seq1: [A, B, C]
seq2: [D, E]
seq3: [F, G, H, I]
```

会被拼成：

```text
input_ids = [A, B, C, D, E, F, G, H, I]
```

但是模型还必须知道每条序列的边界在哪里，所以还会准备：

```text
cu_seqlens_q
cu_seqlens_k
max_seqlen_q
max_seqlen_k
positions
slot_mapping
block_tables
```

这些信息不是给用户看的，而是给 attention 和 flash-attn 用的。

它们解决的问题是：

```text
虽然多个 prompt 被拼成了一个大 tensor，
但是 attention 不能让 seq1 看到 seq2，
也不能让 seq2 看到 seq3。
```

所以需要边界信息，让 flash-attn 知道每条序列各自的长度。

---

## 3.8 slot_mapping 是什么？

模型计算出每个 token 的 K/V 后，需要把它们写入 KV Cache。

问题是：

```text
每个 token 应该写到 KV Cache 的哪个物理位置？
```

`slot_mapping` 就是这个映射表。

它告诉 attention 层：

```text
第 0 个输入 token 的 K/V 写到 KV Cache 的 slot X
第 1 个输入 token 的 K/V 写到 KV Cache 的 slot Y
第 2 个输入 token 的 K/V 写到 KV Cache 的 slot Z
```

slot 的计算依赖 Sequence 的 block_table。

简单理解：

```text
block_table 决定这个请求占用哪些 KV block
slot_mapping 决定本次输入 token 写到这些 block 的具体哪个位置
```

---

## 3.9 prefill 中模型 forward 做了什么？

prefill 阶段，Qwen3 模型大致做：

```text
input_ids
  ↓
Embedding：token id 变成向量
  ↓
多层 Transformer Decoder Layer
  ↓
每层里面做 Attention + MLP + RMSNorm
  ↓
得到每个输入 token 的 hidden state
  ↓
LM Head 把 hidden state 转成 vocab logits
```

在 prefill 阶段，虽然 prompt 中每个 token 都会经过模型，但最终用于预测第一个回答 token 的，是每条 prompt 最后一个 token 位置的 hidden state。

为什么？

因为自回归模型要预测的是：

```text
P(下一个 token | 前面所有 prompt token)
```

prompt 最后一个位置的 hidden state 已经通过 causal attention 汇聚了前面所有 token 的信息。

所以用最后一个位置预测下一个 token。

---

## 3.10 prefill 的输出是什么？

prefill 的输出不是完整回答。

prefill 的输出通常是：

```text
每条 Sequence 的第一个生成 token
```

例如：

```text
prompt = "介绍一下 KV Cache"
```

prefill 后模型可能采样出：

```text
"KV"
```

此时 Sequence 变成：

```text
prompt token ids + 第一个生成 token id
```

然后这条 Sequence 进入 running 队列，后续进入 decode 阶段。

---

# 4. Decode 阶段详细过程

## 4.1 decode 解决什么问题？

prefill 已经把 prompt 处理完了，也已经建立了 prompt 的 KV Cache。

接下来模型要继续生成第 2 个、第 3 个、第 4 个……回答 token。

decode 的核心任务是：

```text
每轮只输入当前序列的最后一个 token，利用历史 KV Cache，预测下一个 token。
```

例如：

```text
prompt: 介绍一下 KV Cache
已经生成: KV
```

decode 下一轮输入的不是完整的：

```text
介绍一下 KV Cache KV
```

而只是最后一个 token：

```text
KV
```

历史内容的信息通过 KV Cache 提供。

---

## 4.2 为什么 decode 每次只处理一个 token？

因为历史 token 的 K/V 已经在 prefill 和之前的 decode 中保存过了。

假设当前序列是：

```text
[prompt tokens] + [已生成 token1, token2, token3]
```

生成下一个 token 时，只需要：

```text
1. 计算 token3 的 Q/K/V
2. 把 token3 的 K/V 写入 KV Cache
3. 用 token3 的 Q 去和历史所有 K/V 做 attention
4. 得到 logits
5. 采样 token4
```

历史 prompt 和 token1、token2 的 K/V 不需要重新计算。

---

## 4.3 decode 的输入是什么？

对每条 running Sequence，decode 输入是：

```text
last_token
position
context_len
block_table
slot_mapping
```

含义分别是：

```text
last_token：当前序列最后一个 token id
position：这个 token 在序列中的位置
context_len：当前序列总长度
block_table：这条序列对应的 KV Cache block 映射
slot_mapping：last_token 的 K/V 应该写入 KV Cache 的哪个 slot
```

---

## 4.4 decode 中 attention 怎么看见历史内容？

decode 时只输入一个 token，但 attention 需要看到完整历史。

它是通过 KV Cache 看到历史的。

具体来说：

```text
当前 token 产生 query：Q_current
历史所有 token 的 K/V 存在 KV Cache 中
Attention 用 Q_current 去查询历史 K/V
```

所以 decode 不是“忘了 prompt”，而是：

```text
prompt 的信息已经存在 KV Cache 里。
```

---

## 4.5 decode 的输出是什么？

每轮 decode 对每条 running Sequence 输出一个新 token。

例如：

```text
第 1 轮 decode 输出：Cache
第 2 轮 decode 输出：是
第 3 轮 decode 输出：大模型
第 4 轮 decode 输出：推理
……
```

每输出一个 token，就追加到 Sequence.token_ids 后面。

然后下一轮 decode 再用这个新 token 作为输入，继续预测下一个 token。

---

# 5. 从 prompt 到完整回答的完整流程

下面用完整链路把整个过程串起来。

---

## 5.1 用户调用 generate

用户代码大概是：

```python
outputs = llm.generate(prompts, sampling_params)
```

这里的 `prompts` 可以是一条，也可以是多条。

例如：

```python
prompts = [
    "介绍一下 KV Cache",
    "介绍一下 PagedAttention",
    "介绍一下 prefill 和 decode"
]
```

---

## 5.2 prompt 被 tokenizer 编码

如果 prompt 是字符串，nano-vLLM 会调用 tokenizer：

```text
字符串 prompt → token id 列表
```

例如：

```text
"介绍一下 KV Cache"
```

变成：

```text
[101, 205, 307, 408, 509]
```

---

## 5.3 创建 Sequence

每个 prompt 会创建一个 Sequence。

例如有 3 个 prompt，就创建 3 个 Sequence：

```text
Sequence 0: token_ids = prompt0_token_ids
Sequence 1: token_ids = prompt1_token_ids
Sequence 2: token_ids = prompt2_token_ids
```

每个 Sequence 初始状态都是：

```text
status = WAITING
is_prefill = True
num_cached_tokens = 0
num_scheduled_tokens = 0
block_table = []
```

---

## 5.4 Sequence 进入 waiting 队列

Scheduler 维护两个核心队列：

```text
waiting 队列：还没有完成 prefill 的请求
running 队列：已经完成 prefill，正在 decode 的请求
```

刚加入的请求会进入 waiting。

---

## 5.5 主循环开始执行 step

`generate()` 内部会反复执行：

```text
while not scheduler.is_finished():
    step()
```

每一个 step 大致包括：

```text
1. Scheduler 选择本轮要跑哪些 Sequence
2. ModelRunner 执行模型 forward
3. Sampler 采样下一个 token
4. Scheduler 更新 Sequence 状态
5. 如果某条 Sequence 完成，就收集输出
```

---

## 5.6 第一次 step 通常是 prefill

因为刚开始所有请求都在 waiting 队列，所以 Scheduler 会优先做 prefill。

Scheduler 会尽量把多个 prompt 放进同一轮 batch。

例如：

```text
seq0 prompt 长度 100
seq1 prompt 长度 300
seq2 prompt 长度 80
```

如果 token 预算足够，可能一轮同时处理：

```text
seq0 + seq1 + seq2
```

这一轮 prefill 的总 token 数是：

```text
100 + 300 + 80 = 480
```

---

## 5.7 BlockManager 分配 KV Cache

在真正跑模型前，BlockManager 会给每条 Sequence 分配 KV Cache block。

例如：

```text
seq0.block_table = [3]
seq1.block_table = [5, 6]
seq2.block_table = [9]
```

这表示每条 Sequence 的 KV Cache 存储位置。

---

## 5.8 ModelRunner 准备模型输入

ModelRunner 把多个 Sequence 的 token 拼成 tensor：

```text
input_ids = [seq0 tokens, seq1 tokens, seq2 tokens]
```

同时准备：

```text
positions：每个 token 的位置编号
cu_seqlens：每条序列的边界
slot_mapping：每个 token 的 KV Cache 写入位置
block_tables：每条序列的 block 表
```

然后设置全局 context，供 attention 层使用。

---

## 5.9 Qwen3 模型执行 forward

模型 forward 大致过程：

```text
input_ids
  ↓
Embedding
  ↓
第 1 层 Transformer Decoder
  ↓
第 2 层 Transformer Decoder
  ↓
……
  ↓
最后一层 Transformer Decoder
  ↓
RMSNorm
  ↓
LM Head
  ↓
logits
```

每一层 decoder 内部大致是：

```text
RMSNorm
  ↓
Self-Attention
  ↓
Residual
  ↓
RMSNorm
  ↓
MLP
  ↓
Residual
```

在 attention 中，模型会：

```text
1. 计算 Q/K/V
2. 对 Q/K 做 RoPE 位置编码
3. 把 K/V 写入 KV Cache
4. 用 causal attention 计算当前 token 对历史 token 的注意力
```

---

## 5.10 LM Head 输出 logits

模型最后会输出 logits。

对于每条 Sequence，logits 表示：

```text
在当前上下文后面，词表中每一个 token 成为下一个 token 的分数。
```

例如：

```text
"KV Cache 是"
```

模型可能给出：

```text
一种：8.2
用于：7.9
大：2.1
苹果：-3.4
EOS：-5.0
……
```

这些还不是概率，只是分数。

---

## 5.11 Sampler 选择下一个 token

Sampler 会先做 temperature 缩放：

```text
logits / temperature
```

temperature 的作用是控制随机性：

```text
temperature 较低：高分 token 更容易被选中，输出更稳定
temperature 较高：低分 token 也有更多机会被选中，输出更随机
```

然后做 softmax：

```text
logits → probability distribution
```

得到每个 token 的概率。

最后按照概率采样一个 token。

需要注意：

```text
采样不一定永远选择概率最高的 token。
```

如果某个 token 概率最高，它只是“更可能被选中”，不是“一定被选中”。

nano-vLLM 的 Sampler 使用了一种指数分布采样技巧，本质上仍然是在按照概率分布抽样。

---

## 5.12 采样出的 token 被追加到 Sequence

假设采样出的 token id 是：

```text
token_id = 888
```

Scheduler 会执行类似逻辑：

```text
seq.append_token(888)
```

于是 Sequence 从：

```text
[prompt tokens]
```

变成：

```text
[prompt tokens, 888]
```

这个 888 就是回答的第一个 token。

---

## 5.13 判断是否结束

每次追加 token 后，Scheduler 会判断：

```text
1. token_id 是否等于 EOS
2. 是否达到 max_tokens
```

如果满足任意一个结束条件：

```text
seq.status = FINISHED
释放 KV Cache block
从 running 队列移除
```

否则继续 decode。

---

## 5.14 进入 decode 循环

prefill 结束后，Sequence 进入 running 队列。

后续每一轮 decode：

```text
1. Scheduler 从 running 队列选择一批 Sequence
2. 每条 Sequence 只输入 last_token
3. ModelRunner 准备 decode 输入
4. Attention 使用已有 KV Cache
5. 模型输出 logits
6. Sampler 采样下一个 token
7. 追加 token
8. 判断 EOS / max_tokens
```

这个过程会一直重复。

---

## 5.15 最终 decode 成文本

当所有 Sequence 都完成后，`generate()` 会收集每个 Sequence 的 completion_token_ids。

注意：completion_token_ids 不包括 prompt token，只包括生成出来的回答 token。

然后调用 tokenizer.decode：

```text
completion_token_ids → 文本回答
```

最后返回类似：

```python
[
    {
        "text": "KV Cache 是大模型推理中用于保存历史 Key 和 Value 的缓存机制……",
        "token_ids": [...]
    }
]
```

---

# 6. “下一个 token”到底是怎么来的？

这是最关键的问题。

大模型不是查数据库，也不是模板匹配。

它每一步都在计算：

```text
在当前上下文条件下，词表中每个 token 作为下一个 token 的可能性。
```

可以写成：

```text
P(next_token | previous_tokens)
```

例如当前上下文是：

```text
KV Cache 的作用是
```

模型会给整个词表打分：

```text
保存：高分
缓存：高分
提高：中高分
香蕉：低分
EOS：可能低分
```

然后经过 temperature 和 softmax，得到概率：

```text
保存：0.35
缓存：0.22
提高：0.12
减少：0.06
……
```

采样器从这个概率分布中抽一个。

如果抽到“保存”，序列就变成：

```text
KV Cache 的作用是保存
```

下一步再重新计算：

```text
P(next_token | KV Cache 的作用是保存)
```

如此循环，直到结束。

---

# 7. 为什么模型可以“理解”前文？

从推理过程看，模型并没有人类意义上的理解。

它的机制是：

```text
1. token 变成向量
2. attention 让每个 token 和历史 token 发生信息交互
3. 多层 Transformer 不断提取上下文特征
4. 最后输出下一个 token 的概率分布
```

在 prefill 阶段，prompt 中每个位置会通过 causal attention 看见自己之前的 token。

在 decode 阶段，新 token 通过 KV Cache 看见所有历史 token。

所以模型生成的每个 token 都依赖之前的上下文。

---

# 8. 多个 prompt 是怎么高并发处理的？

## 8.1 高并发不是每个请求一个线程

在大模型推理中，高并发的核心不是：

```text
一个请求开一个 Python 线程
```

而是：

```text
把多个请求组成 batch，一起送到 GPU 上计算。
```

GPU 擅长并行计算。

如果一次只处理一个请求，GPU 可能吃不满，吞吐量低。

如果把多个请求合成 batch，GPU 可以一次处理更多 token，吞吐量更高。

---

## 8.2 nano-vLLM 中的两个队列

Scheduler 维护两个队列：

```text
waiting：等待 prefill 的请求
running：正在 decode 的请求
```

新请求进入：

```text
waiting
```

完成 prefill 后进入：

```text
running
```

生成结束后离开：

```text
FINISHED
```

---

## 8.3 多个 prompt 的 prefill batching

如果同时来了多个 prompt：

```text
prompt A
prompt B
prompt C
```

它们会分别变成：

```text
seq A
seq B
seq C
```

然后进入 waiting 队列。

Scheduler 会尽量把它们一起调度进同一个 prefill batch。

限制条件是：

```text
1. 不能超过 max_num_seqs
2. 不能超过 max_num_batched_tokens
3. KV Cache block 必须够
```

如果都满足，就可以一轮处理多个 prompt。

---

## 8.4 不同长度 prompt 怎么放进一个 batch？

不同 prompt 长度不一样，例如：

```text
seq A 长度 10
seq B 长度 200
seq C 长度 35
```

nano-vLLM 会把它们的 token 拼成一个平坦 tensor：

```text
[A 的 10 个 token, B 的 200 个 token, C 的 35 个 token]
```

同时用 `cu_seqlens` 记录边界：

```text
A 从 0 到 10
B 从 10 到 210
C 从 210 到 245
```

这样 flash-attn 就知道：

```text
A 只能在 A 内部做 attention
B 只能在 B 内部做 attention
C 只能在 C 内部做 attention
```

不会出现 A 的 prompt 看到 B 的 prompt 的问题。

---

## 8.5 多个请求的 decode batching

decode 阶段，每条 running Sequence 每轮只需要处理一个 token。

假设当前有 4 条正在生成的请求：

```text
seq A 当前最后 token 是 a
seq B 当前最后 token 是 b
seq C 当前最后 token 是 c
seq D 当前最后 token 是 d
```

那么一轮 decode 的输入就是：

```text
[a, b, c, d]
```

这就是一个 decode batch。

模型会一次性输出 4 组 logits：

```text
seq A 的下一个 token 概率分布
seq B 的下一个 token 概率分布
seq C 的下一个 token 概率分布
seq D 的下一个 token 概率分布
```

Sampler 再分别为每条 Sequence 采样一个 token。

---

## 8.6 每条请求长度不同，decode 怎么处理？

虽然每条 Sequence decode 时都只输入一个 token，但它们的历史长度可能不同：

```text
seq A 已经有 100 个 token
seq B 已经有 800 个 token
seq C 已经有 30 个 token
```

decode attention 需要知道每条 Sequence 的历史上下文长度。

所以 ModelRunner 会准备：

```text
context_lens = [100, 800, 30]
block_tables = 每条 Sequence 的 KV Cache block 映射
```

flash-attn 会根据这些信息，从 KV Cache 中找到每条 Sequence 自己的历史 K/V。

---

## 8.7 continuous batching 是什么？

传统 batching 可能是：

```text
等一批请求全部生成完成，再处理下一批请求。
```

这样的问题是：

```text
有些请求很短，很快结束；
有些请求很长，拖住整个 batch；
GPU 利用率会下降。
```

continuous batching 的思想是：

```text
每一轮 step 都重新调度。
已经结束的请求立刻退出。
新来的请求可以加入后续 step。
正在生成的请求继续 decode。
```

nano-vLLM 的 Scheduler 每次 step 都会重新决定：

```text
本轮跑哪些 waiting 请求做 prefill？
或者本轮跑哪些 running 请求做 decode？
```

这就比固定 batch 更灵活。

---

## 8.8 nano-vLLM 的一个重要简化

在完整 vLLM 中，调度逻辑会更复杂，可能支持更细粒度的 prefill/decode 混合。

nano-vLLM 这份代码为了简洁，调度策略可以概括为：

```text
如果 waiting 队列中可以调度 prefill，本轮优先跑 prefill。
如果本轮没有 prefill 可跑，再跑 decode。
```

也就是说，它不是在同一个 step 里复杂混合 prefill 和 decode，而是每个 step 选择一种主要阶段。

这对于学习非常友好，因为逻辑更清晰。

---

# 9. KV Cache block 不够时怎么办？

高并发时，很多请求都要占用 KV Cache。

如果 GPU 显存里的 KV Cache block 不够，Scheduler 会触发 preemption，也就是抢占。

抢占的大致过程是：

```text
1. 选择某条 running Sequence
2. 释放它当前占用的 KV Cache block
3. 把它重新放回 waiting 队列
4. 以后再重新 prefill 恢复它的状态
```

这听起来浪费，但可以保证系统不会因为 KV Cache 不够直接崩掉。

可以类比操作系统内存不够时，把一部分进程换出去。

---

# 10. prefix cache 如何提高高并发性能？

如果很多请求有相同前缀，prefix cache 可以明显减少重复计算。

例如系统提示词相同：

```text
你是一个专业的 AI Infra 助教，请回答以下问题：
```

后面用户问题不同：

```text
问题 A：什么是 KV Cache？
问题 B：什么是 PagedAttention？
问题 C：什么是 continuous batching？
```

这些请求前面共享同一个 system prompt。

prefix cache 可以复用 system prompt 的 KV Cache block。

这样每个请求不必重复 prefill 同一段前缀。

在多用户服务里，这对吞吐量和 TTFT 都有帮助。

---

# 11. TTFT 和 TPOT 怎么理解？

虽然你这次主要问流程，但这两个指标和 prefill/decode 强相关。

## 11.1 TTFT

TTFT 是 Time To First Token。

意思是：

```text
从用户提交 prompt，到模型生成第一个回答 token 的时间。
```

TTFT 主要受 prefill 影响。

prompt 越长，prefill 要处理的 token 越多，TTFT 通常越大。

---

## 11.2 TPOT

TPOT 是 Time Per Output Token。

意思是：

```text
生成阶段中，每生成一个 token 平均需要多少时间。
```

TPOT 主要受 decode 影响。

decode 每轮生成一个 token，所以 decode 性能直接决定回答流式输出的速度。

---

# 12. 一个完整例子：单个 prompt

假设用户输入：

```text
介绍一下 KV Cache
```

完整过程是：

```text
1. tokenizer 编码
   "介绍一下 KV Cache" → [10, 20, 30, 40]

2. 创建 Sequence
   token_ids = [10, 20, 30, 40]
   status = WAITING

3. Scheduler 调度 prefill
   发现 waiting 队列有请求
   分配 KV Cache block

4. ModelRunner prepare_prefill
   input_ids = [10, 20, 30, 40]
   positions = [0, 1, 2, 3]
   slot_mapping = 对应 KV Cache 写入位置

5. Qwen3 forward
   embedding → attention → MLP → logits

6. Sampler 采样第一个生成 token
   例如采样出 token 50

7. Scheduler append_token
   token_ids = [10, 20, 30, 40, 50]

8. 进入 decode
   输入 last_token = 50
   利用 KV Cache 看到历史 [10, 20, 30, 40]
   预测 token 51

9. 继续 decode
   token_ids = [10, 20, 30, 40, 50, 51]
   输入 last_token = 51
   预测 token 52

10. 重复直到 EOS 或 max_tokens

11. tokenizer.decode
   [50, 51, 52, ...] → "KV Cache 是一种用于保存……"
```

---

# 13. 一个完整例子：多个 prompt 高并发

假设同时来了三个请求：

```text
A: 介绍一下 KV Cache
B: 介绍一下 PagedAttention
C: 介绍一下 continuous batching
```

## 第一步：全部进入 waiting

```text
waiting = [A, B, C]
running = []
```

## 第二步：调度 prefill batch

如果资源足够：

```text
scheduled = [A, B, C]
is_prefill = True
```

三个 prompt 被拼成一个 batch，一次送进 GPU。

## 第三步：分别得到第一个生成 token

模型输出：

```text
A 的第一个 token
B 的第一个 token
C 的第一个 token
```

然后：

```text
waiting = []
running = [A, B, C]
```

## 第四步：decode batch

下一轮 decode：

```text
input_ids = [A.last_token, B.last_token, C.last_token]
```

模型一次输出：

```text
A 的下一个 token
B 的下一个 token
C 的下一个 token
```

## 第五步：某些请求先结束

假设 C 很短，先生成 EOS：

```text
C.status = FINISHED
释放 C 的 KV Cache
running = [A, B]
```

A 和 B 继续 decode。

如果此时来了新请求 D：

```text
waiting = [D]
running = [A, B]
```

下一轮 Scheduler 又会重新调度，D 可以开始 prefill。

这就是 continuous batching 的动态性。

---

# 14. 用一张总流程图理解

```text
用户 prompts
   ↓
Tokenizer 编码
   ↓
每个 prompt 创建一个 Sequence
   ↓
加入 Scheduler.waiting 队列
   ↓
┌─────────────────────────────┐
│ while 还有未完成请求          │
│                             │
│  Scheduler.schedule()        │
│     ↓                       │
│  选择 prefill 或 decode       │
│     ↓                       │
│  BlockManager 分配/追加 block │
│     ↓                       │
│  ModelRunner 准备输入         │
│     ↓                       │
│  Qwen3 模型 forward          │
│     ↓                       │
│  LM Head 输出 logits          │
│     ↓                       │
│  Sampler 采样 token           │
│     ↓                       │
│  Scheduler.postprocess()     │
│     ↓                       │
│  追加 token / 判断结束 / 释放缓存│
└─────────────────────────────┘
   ↓
收集 completion_token_ids
   ↓
Tokenizer decode
   ↓
返回文本回答
```

---

# 15. 用一句话区分 engine 中几个文件在流程里的作用

虽然这份笔记不重点讲代码，但你已经看完 engine，所以这里把它们和推理流程对应起来。

| 文件 | 在完整推理过程中的角色 |
|---|---|
| `llm_engine.py` | 总入口，负责接收 prompts、创建 Sequence、循环 step、返回最终文本 |
| `sequence.py` | 定义每个请求的运行时状态，保存 token_ids、block_table、生成进度 |
| `scheduler.py` | 决定每一轮跑哪些 Sequence，是 prefill 还是 decode |
| `block_manager.py` | 管理 KV Cache block，负责分配、释放、prefix cache、引用计数 |
| `model_runner.py` | 把调度结果变成 GPU tensor，执行模型 forward，并调用 sampler 得到 token |

更直观地说：

```text
LLMEngine：老板，负责总流程
Scheduler：调度员，决定这一轮谁上 GPU
Sequence：每个请求的档案袋
BlockManager：KV Cache 显存管理员
ModelRunner：GPU 执行员，真正跑模型
```

---

# 16. 初学者最容易混淆的几个点

## 16.1 prefill 不是生成完整回答

prefill 只是处理 prompt，并通常产生第一个回答 token。

完整回答主要是在 decode 的循环中逐 token 生成出来的。

---

## 16.2 decode 不是不看 prompt

decode 每次虽然只输入一个 last_token，但它通过 KV Cache 看到完整历史。

所以 decode 不是丢掉了 prompt，而是复用了 prompt 的缓存。

---

## 16.3 batch 不是固定不变的

在 continuous batching 里，batch 每一轮都可能变化。

```text
有的请求结束退出
有的新请求加入
有的请求继续 decode
有的请求被抢占回 waiting
```

---

## 16.4 KV Cache 保存的不是 token id

token id 是输入编号。

KV Cache 保存的是模型每一层 attention 计算出来的 Key/Value 向量。

BlockManager 里的 block token_ids 主要是为了 prefix cache 哈希校验，不等于真正的 K/V 张量本身。

真正的大块 K/V 张量在 ModelRunner 分配的 `kv_cache` 里。

---

## 16.5 EOS 不是人为提前知道的

EOS 是模型词表中的特殊 token。

每一步采样时，EOS 都可能被选中。

如果模型认为当前回答应该结束，EOS 概率可能变高，被采样出来后系统结束该请求。

---

# 17. 最终总结

nano-vLLM 的完整推理过程可以压缩成下面这段话：

> 用户输入 prompt 后，tokenizer 先把文本转成 token id。每个 prompt 被封装成一个 Sequence，进入 waiting 队列。Scheduler 先调度 prefill，BlockManager 为每个 Sequence 分配 KV Cache block，ModelRunner 把 token 拼成 batch，送入 Qwen3 模型。模型在 prefill 中处理 prompt，写入 KV Cache，并根据最后一个 prompt token 的 hidden state 输出 logits，Sampler 根据 logits 的概率分布采样第一个回答 token。之后 Sequence 进入 running 队列，decode 阶段每轮只输入每条 Sequence 的 last_token，通过 KV Cache 访问完整历史，再输出 logits 并采样下一个 token。这个过程不断循环，直到采样到 EOS 或达到 max_tokens。多个 prompt 通过 waiting/running 队列、batching、KV Cache block 管理和 continuous batching 在 GPU 上并发处理。

最核心的理解是：

```text
大模型回答不是一次性生成的，
而是每次根据已有上下文预测一个下一个 token。

prefill 负责读 prompt、建缓存、出第一个 token；
decode 负责利用缓存继续一个 token 一个 token 地生成；
Scheduler 负责把很多请求动态组成 batch；
BlockManager 负责管理 KV Cache 显存；
ModelRunner 负责真正跑 GPU 模型；
Sampler 负责从 logits 概率分布中选出下一个 token。
```


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]

%% 项目关联导航：结束 %%
