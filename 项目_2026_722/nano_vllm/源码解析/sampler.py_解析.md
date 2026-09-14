# sampler.py 源码宏观解析

## 1. 文件整体定位

`sampler.py` 是 nano-vLLM 中负责 **从 logits 采样生成 next token** 的文件。

它位于推理流程的最后一步：

```text
input_ids
  -> Qwen3 model forward
  -> hidden_states
  -> LM Head
  -> logits
  -> Sampler
  -> next token id
```

在 `model_runner.py` 中，它被这样使用：

```python
self.sampler = Sampler()
```

每轮推理运行时：

```python
logits = self.run_model(input_ids, positions, is_prefill)
token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
```

所以 `sampler.py` 不属于 Transformer Decoder Block 内部，也不是 Attention、MLP、KV Cache 的一部分。它负责的是 **模型输出阶段**：把 LM Head 输出的词表分数 `logits` 转换成真正要追加到 sequence 里的 token id。

如果前面的 `attention.py`、`linear.py`、`embed_head.py` 负责“算模型”，那么 `sampler.py` 负责“根据模型分数选下一个 token”。

## 2. 这个文件要解决的核心问题

`sampler.py` 主要解决的是 **采样输出问题** 和 **张量计算问题**。

| 问题 | 说明 |
|---|---|
| 采样输出问题 | 从每条 sequence 的词表 logits 中采样 next token |
| 张量计算问题 | 温度缩放、softmax、随机采样 |
| 模型结构问题 | 不涉及模型层结构 |
| 权重加载问题 | 不涉及，没有参数 |
| 并行切分问题 | 不直接并行切分；通常只在 rank 0 上执行 |

它和原始 Transformer 的关系是：

```text
Transformer 输出 hidden_states
  -> LM Head 得到 logits
  -> Sampling / Greedy / Beam Search 等解码策略
  -> 得到 next token
```

原始 Transformer 本身只定义了如何计算 logits，并不固定必须怎么选 token。采样策略属于推理系统的输出解码阶段。

在 nano-vLLM 中，这个文件实现的是一种简化采样：

```text
temperature scaling -> softmax -> multinomial sampling
```

但它没有实现：

- greedy sampling；
- top-k；
- top-p / nucleus sampling；
- repetition penalty；
- presence penalty；
- frequency penalty；
- beam search；
- logprobs 返回。

因此它是一个非常轻量的 sampler，适合 nano-vLLM 教学和简化推理框架，但功能上比完整 vLLM 的 sampler 少很多。

## 3. 代码结构总览

### 3.1 文件导入

```python
import torch
from torch import nn
```

这里只依赖 PyTorch：

| 导入 | 作用 |
|---|---|
| `torch` | 张量、softmax、随机指数分布 |
| `nn` | 定义 PyTorch Module |

它没有导入 `torch.distributed`，说明 sampler 自己不负责跨 GPU 通信。

在 tensor parallel 场景下，`ParallelLMHead` 会在 rank 0 上 gather 并拼接完整 logits，然后 `model_runner.py` 只在 rank 0 上执行采样：

```python
token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
```

所以 sampler 接收到的通常已经是完整词表 logits。

### 3.2 `Sampler`

文件中只定义了一个类：

```python
class Sampler(nn.Module):
```

它没有 `__init__`，也没有参数。说明它不保存模型权重，只是一个纯计算模块。

### 3.3 `forward`

核心代码：

```python
@torch.compile
def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1)
    sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
    return sample_tokens
```

它做了四步：

1. 把 logits 转成 float32；
2. 根据 temperature 对 logits 做缩放；
3. 对词表维度做 softmax，得到概率分布；
4. 用随机噪声采样，得到每条 sequence 的 token id。

`@torch.compile` 的作用是让 PyTorch 尝试编译优化这个小函数，减少 Python 层调度开销。

## 4. 张量流和 forward 流程

### 4.1 输入是什么

Sampler 的输入是：

```python
logits: torch.Tensor
temperatures: torch.Tensor
```

典型 shape：

```text
logits:       [num_seqs, vocab_size]
temperatures: [num_seqs]
```

其中：

| 维度 | 含义 |
|---|---|
| `num_seqs` | 当前推理步中需要采样的 sequence 数 |
| `vocab_size` | 模型词表大小 |

`logits` 来自：

```python
self.model.compute_logits(...)
```

也就是：

```text
hidden_states -> ParallelLMHead -> logits
```

`temperatures` 来自每条 sequence 的采样参数：

```python
def prepare_sample(self, seqs: list[Sequence]):
    temperatures = [seq.temperature for seq in seqs]
    temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
    return temperatures
```

每条 sequence 可以有自己的 temperature。

### 4.2 温度缩放

代码：

```python
logits = logits.float().div_(temperatures.unsqueeze(dim=1))
```

先把 logits 转成 float32：

```python
logits.float()
```

这样 softmax 更稳定。

然后：

```python
temperatures.unsqueeze(dim=1)
```

把：

```text
temperatures: [num_seqs]
```

变成：

```text
temperatures: [num_seqs, 1]
```

这样可以广播到每一行 logits：

```text
logits:       [num_seqs, vocab_size]
temperatures: [num_seqs, 1]
```

温度缩放公式：

```text
scaled_logits = logits / temperature
```

temperature 的直觉：

| temperature | 效果 |
|---|---|
| 小于 1 | 分布更尖锐，更偏向高分 token |
| 等于 1 | 不改变 logits |
| 大于 1 | 分布更平滑，更随机 |

nano-vLLM 的 `SamplingParams` 中限制：

```python
assert self.temperature > 1e-10, "greedy sampling is not permitted"
```

所以它不允许 temperature 接近 0，也就是不支持 greedy sampling。

### 4.3 softmax 得到概率

代码：

```python
probs = torch.softmax(logits, dim=-1)
```

沿最后一维，也就是词表维度做 softmax：

```text
probs: [num_seqs, vocab_size]
```

每一行都是一条 sequence 的 next token 概率分布：

```text
sum(probs[i]) = 1
```

例如：

```text
logits[0] -> probs[0] -> 第 0 条 sequence 的词表概率
logits[1] -> probs[1] -> 第 1 条 sequence 的词表概率
```

### 4.4 随机采样：Gumbel-Max / exponential trick

最关键的一行是：

```python
sample_tokens = probs.div_(
    torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
).argmax(dim=-1)
```

先生成和 `probs` 同 shape 的指数分布随机噪声：

```python
torch.empty_like(probs).exponential_(1)
```

shape：

```text
[num_seqs, vocab_size]
```

然后防止除以非常接近 0 的数：

```python
clamp_min_(1e-10)
```

接着：

```python
probs / exponential_noise
```

最后对词表维度取最大：

```python
argmax(dim=-1)
```

得到：

```text
sample_tokens: [num_seqs]
```

这是一种按概率分布采样的技巧，常被称为 exponential race / Gumbel-Max trick 的变体。

直觉上：

- `probs` 越大，被选中的概率越高；
- 随机噪声让结果不是永远选最大概率 token；
- `argmax` 最终为每条 sequence 选出一个 token id。

等价目标可以理解为：

```python
torch.multinomial(probs, num_samples=1)
```

但这里用张量操作和 `argmax` 实现，便于编译和批量执行。

### 4.5 输出是什么

输出：

```text
sample_tokens: [num_seqs]
```

每个元素是一个 token id。

例如：

```text
sample_tokens = [3187, 1024, 151643]
```

表示当前 batch 中 3 条 sequence 分别采样到了 3 个 next token。

这些 token id 后续会由 engine/scheduler 追加到对应的 `Sequence` 中，并在需要返回文本时由 tokenizer decode 成字符串。

## 5. 和 Transformer / Qwen 模型结构的关系

### 5.1 和 LM Head 的关系

Qwen3ForCausalLM 中：

```python
hidden_states = self.model(input_ids, positions)
logits = self.lm_head(hidden_states)
```

Sampler 接在 `logits` 后面：

```text
hidden_states
  -> LM Head
  -> logits
  -> Sampler
  -> next token id
```

LM Head 只是给每个词表 token 打分，Sampler 才真正决定选哪个 token。

### 5.2 和 prefill/decode 的关系

prefill 阶段：

- 模型处理 prompt；
- LM Head 通常只对每条 prompt 的最后 token 计算 logits；
- Sampler 根据这些 logits 采样第一个生成 token。

decode 阶段：

- 每条 sequence 输入上一步生成的 token；
- 模型结合 KV Cache 计算当前 logits；
- Sampler 采样下一个 token；
- 这个 token 被 append 到 sequence；
- 下一轮 decode 继续。

所以 sampler 每轮都会被调用一次，但它自己并不知道 prefill/decode。它只关心：

```text
这一轮有多少条 sequence
每条 sequence 的 logits 是什么
每条 sequence 的 temperature 是多少
```

### 5.3 和自回归生成的关系

自回归生成的核心是：

```text
已有 token -> 预测下一个 token -> 追加 -> 再预测下一个
```

Sampler 对应“预测下一个 token”里的最后一步：

```text
概率分布 -> 具体 token id
```

如果没有 sampler，模型只能给出分数，不能形成实际生成序列。

### 5.4 和 EOS 的关系

`sampler.py` 只负责采样 token id，并不判断 EOS。

也就是说，如果采样结果刚好是 EOS token，`sampler.py` 不会自己停止 sequence。

停止逻辑通常在 engine/scheduler 层根据：

- 采样到的 token 是否等于 EOS；
- 是否达到 `max_tokens`；
- 是否设置 `ignore_eos`；

来决定是否把 sequence 标记为 finished。

这和你之前问的 EOS 问题可以对上：EOS 是模型词表中的一个特殊 token，Sampler 可能采样到它，但“采样到之后是否结束请求”不是 sampler 文件负责。

### 5.5 和 vLLM 完整采样器的区别

nano-vLLM 的 sampler 很简洁，只支持 temperature sampling。

完整 vLLM 采样链路通常会更复杂，可能包括：

| 功能 | nano-vLLM sampler 是否实现 |
|---|---|
| temperature | 是 |
| greedy | 否 |
| top-k | 否 |
| top-p | 否 |
| min-p | 否 |
| repetition penalty | 否 |
| presence/frequency penalty | 否 |
| stop token 处理 | 否 |
| logprobs | 否 |
| beam search | 否 |

所以如果你未来想把 nano-vLLM 改造成更接近 vLLM 的项目，sampler 是一个很适合扩展的模块。

## 6. 学习总结

`sampler.py` 的核心价值可以概括为一句话：

> 它把 LM Head 输出的 logits 经过温度缩放和概率采样，转换成每条 sequence 的下一个 token id。

学习这个文件要抓住三条主线：

1. **生成流程主线**：`hidden_states -> logits -> probs -> token id`；
2. **张量 shape 主线**：`logits [num_seqs, vocab_size]`，输出 `sample_tokens [num_seqs]`；
3. **采样策略主线**：temperature 控制随机性，exponential noise + argmax 实现按概率采样。

它和前面几个文件的关系可以这样理解：

| 文件 | 负责内容 | 和 `sampler.py` 的关系 |
|---|---|---|
| `embed_head.py` | LM Head 生成 logits | Sampler 的输入来源 |
| `attention.py` | 计算 hidden states 中的 attention 部分 | 间接影响 logits |
| `model_runner.py` | 准备 temperatures 并调用 sampler | Sampler 的上层调用者 |
| `sampling_params.py` | 定义 temperature/max_tokens/ignore_eos | Sampler 使用 temperature |
| `sampler.py` | logits 到 next token | 生成链路最后一步 |

对 AI Infra 推理学习来说，这个文件提醒你：推理系统不仅要“跑模型 forward”，还必须把 logits 变成可追加到请求状态中的 token。采样策略越复杂，推理服务的输出控制能力越强；nano-vLLM 这里保留的是最小可用版本。



%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]

%% 项目关联导航：结束 %%
