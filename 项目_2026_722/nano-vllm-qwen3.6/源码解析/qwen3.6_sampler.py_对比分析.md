# qwen3.6_sampler.py_对比分析

> 对比对象：  
> 1. 原版 `nano-vLLM` 的 `sampler.py`  
> 2. `nano-vllm-qwen3.6` 中对应的 `sampler.py`
>
> 本文目标：从整体工程角度分析 qwen3.6 版本相比原版 `sampler.py` 做了哪些修改、为什么这样改，以及这些修改在 Qwen3.6 / 推理系统 / 分布式采样中的作用。

---

## 1. 文件整体定位

`sampler.py` 是 nano-vLLM 推理链路的最后一个核心计算模块。

模型 forward 的输出不是文本，而是 logits：

```text
hidden_states
  ↓
LM Head
  ↓
logits: [batch_size, vocab_size]
```

`Sampler` 的作用就是：

```text
根据 logits 和采样参数，选出每个请求的下一个 token id。
```

它在整体推理流程中的位置是：

```text
ModelRunner.prepare_prefill / prepare_decode
  ↓
模型 forward
  ↓
LM Head 得到 logits
  ↓
Sampler
  ↓
next_token_ids
  ↓
Scheduler.postprocess
  ↓
Sequence.append_token
```

所以这个文件不负责模型结构，也不负责 KV Cache / GDN state / 多模态输入。它负责的是：

```text
从 logits 到下一个 token 的选择策略。
```

---

## 2. qwen3.6 版本整体变化概览

相比原版 `sampler.py`，qwen3.6 版本主要做了以下变化：

| 改动点 | 原版 nano-vLLM | qwen3.6 版本 | 作用 |
|---|---|---|---|
| `forward()` 返回值 | 只返回 `sample_tokens` | 仍只返回 `sample_tokens`，但内部调用 `forward_with_scores()` | 保持原接口兼容 |
| 随机采样实现 | softmax 后用指数噪声采样 | logits 加 Gumbel 形式噪声后 max | 数学上等价，但更容易返回 score |
| 是否返回分数 | 不返回 score | 新增 `forward_with_scores()` 返回 token 和 score | 支持分布式 TP 下跨 rank 选全局最优 token |
| 贪心采样 | 无单独方法 | 新增 `greedy_with_scores()` | 支持 temperature≈0 的 greedy 快路径 |
| softmax | 显式计算 `probs = softmax(logits)` | 不显式计算 softmax | 减少一次概率张量构造，更适合 score 比较 |
| 和 ModelRunner 联动 | `self.sampler(logits, temperatures)` | `sample()` 中可调用 `greedy_with_scores` / `forward_with_scores` | 配合 qwen3.6 的多卡采样和 probe 功能 |

一句话总结：

**qwen3.6 版本的 `Sampler` 从“只负责采样 token”扩展成了“既能采样 token，也能返回对应 score，并支持 greedy 快路径”的采样模块。**

---

## 3. 原版 Sampler 的逻辑

原版 `Sampler` 只有一个 `forward()`：

```python
@torch.compile
def forward(self, logits, temperatures):
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1)
    sample_tokens = probs.div_(
        torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
    ).argmax(dim=-1)
    return sample_tokens
```

它的流程是：

```text
logits
  ↓
除以 temperature
  ↓
softmax 得到 probs
  ↓
用指数分布噪声做随机采样
  ↓
argmax 得到 sample_tokens
```

这里的采样方式可以理解为一种等价的 categorical sampling 技巧。

它没有直接用：

```python
torch.multinomial(probs, 1)
```

而是使用指数噪声实现采样。

---

## 4. qwen3.6 版本的结构变化

qwen3.6 版本 `Sampler` 变成三个方法：

```python
class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits, temperatures):
        sample_tokens, _ = self.forward_with_scores(logits, temperatures)
        return sample_tokens

    @torch.compile
    def forward_with_scores(self, logits, temperatures):
        ...

    @torch.compile
    def greedy_with_scores(self, logits):
        ...
```

也就是说：

```text
forward()
    仍然只返回 token，保持兼容

forward_with_scores()
    返回 token 和 score

greedy_with_scores()
    贪心返回 token 和 score
```

这说明 qwen3.6 版本不再只满足单卡、只取 token 的简单场景，而是要支持更复杂的执行层需求。

---

## 5. 改动一：新增 `forward_with_scores()`

### 5.1 qwen3.6 代码逻辑

qwen3.6 新增：

```python
def forward_with_scores(self, logits, temperatures):
    scores = logits.float().div_(temperatures.unsqueeze(dim=1))
    noise = torch.empty_like(scores).exponential_(1).clamp_min_(1e-10).log_()
    scores.sub_(noise)
    sample_scores, sample_tokens = scores.max(dim=-1)
    return sample_tokens, sample_scores
```

它返回两个东西：

```text
sample_tokens
sample_scores
```

### 5.2 sample_scores 是什么

`sample_scores` 不是原始 logits，也不是 softmax 概率。

它是加入随机噪声后的采样分数：

```text
score = logits / temperature - log(Exponential(1))
```

然后对 vocab 维度取最大值：

```text
sample_token = argmax(score)
sample_score = max(score)
```

因此 `sample_scores` 可以理解为：

```text
本 rank 当前选中 token 的采样竞争分数。
```

---

## 6. 为什么 qwen3.6 要返回 score

这个改动要和前面 `embed_head.py`、`model_runner.py` 一起看。

qwen3.6 版本中，`ParallelLMHead.forward()` 不再在 LM Head 内部 gather 全词表 logits，而是每个 tensor parallel rank 返回自己的局部 logits。之后 `ModelRunner.sample()` 会在各 rank 上分别采样，再把 token 和 score all_gather 到 rank 0，由 rank 0 选 score 最大的那个 token。

整体路径是：

```text
每个 TP rank:
    local logits: [batch, vocab_shard]
    Sampler.forward_with_scores()
    得到 local sample_token + local sample_score

rank 0:
    all_gather 所有 rank 的 sample_score / sample_token
    对 rank 维度取最大
    得到全局 sample_token
```

所以 `sample_scores` 是分布式采样必需的。

如果 Sampler 只返回 token，rank 0 无法判断：

```text
哪个 rank 采出来的 token 才是全局应该选中的 token？
```

因此 qwen3.6 必须让 sampler 返回 score。

---

## 7. 改动二：从 softmax 概率采样改为 logits score 采样

### 7.1 原版

原版先显式算 softmax：

```text
probs = softmax(logits / temperature)
```

再用：

```text
probs / exponential_noise
```

取 argmax。

### 7.2 qwen3.6 版本

qwen3.6 不显式计算 softmax，而是：

```text
scores = logits / temperature - log(exponential_noise)
argmax(scores)
```

### 7.3 两者为什么等价

原版：

```text
argmax( probs / E )
```

其中：

```text
probs = softmax(logits)
```

因为 `log()` 是单调函数，所以：

```text
argmax(probs / E)
= argmax(log(probs) - log(E))
```

而：

```text
log(softmax(logits)) = logits - logsumexp(logits)
```

对于同一行 logits 来说，`logsumexp(logits)` 是常数，不影响 argmax。

所以：

```text
argmax(log(probs) - log(E))
= argmax(logits - log(E))
```

这就是 qwen3.6 版本直接在 logits score 上采样的原因。

它避免了显式构造 softmax 概率，同时保留了可比较的采样 score。

---

## 8. 改动三：新增 `greedy_with_scores()`

qwen3.6 新增：

```python
def greedy_with_scores(self, logits):
    sample_scores, sample_tokens = logits.float().max(dim=-1)
    return sample_tokens, sample_scores
```

这表示：

```text
不加随机噪声
不做 temperature
直接选 logits 最大的 token
```

也就是贪心解码。

### 8.1 为什么需要 greedy_with_scores

在 qwen3.6 的 `ModelRunner.run()` 中，有类似逻辑：

```text
greedy = all(seq.temperature <= 1e-10 for seq in seqs)
```

如果所有请求 temperature 接近 0，就可以走 greedy 路径。

这有两个好处：

```text
1. 不需要生成随机噪声
2. 不需要 softmax 或 Gumbel 采样
3. 逻辑更简单，结果确定
4. 分布式 TP 下仍然可以通过 score 比较选全局最大 token
```

所以 `greedy_with_scores()` 是为了支持确定性解码和多卡局部 logits 合并。

---

## 9. forward() 保持兼容

qwen3.6 的 `forward()` 写成：

```python
def forward(self, logits, temperatures):
    sample_tokens, _ = self.forward_with_scores(logits, temperatures)
    return sample_tokens
```

这说明它保留了原版接口：

```text
输入 logits / temperatures
输出 sample_tokens
```

因此旧代码如果仍然调用：

```python
self.sampler(logits, temperatures)
```

依然可以工作。

但新代码可以选择更强的接口：

```python
self.sampler.forward_with_scores(...)
self.sampler.greedy_with_scores(...)
```

这是一种比较好的兼容式扩展：

```text
不破坏原来的调用方式
同时给新执行层提供更多信息
```

---

## 10. 和 tensor parallel 采样的关系

这个文件最关键的工程意义，是配合 qwen3.6 中 LM Head 的分布式采样改造。

原版中，LM Head 会把各 rank 的 logits gather 到 rank 0，得到完整 vocab logits，然后 rank 0 采样。

路径是：

```text
各 rank local logits
  ↓
LM Head 内部 gather full logits 到 rank 0
  ↓
rank 0 Sampler 采样
```

qwen3.6 版本更倾向于：

```text
各 rank local logits
  ↓
各 rank 本地 Sampler 采样并返回 token + score
  ↓
ModelRunner all_gather token + score
  ↓
rank 0 根据 score 选全局 token
```

这样可以避免在 LM Head 内部拼完整 vocab logits，降低通信和内存压力。

`Sampler.forward_with_scores()` 和 `greedy_with_scores()` 正是这个设计的基础。

---

## 11. 和 Qwen3.6 hybrid / GDN 的关系

从这个文件本身看，`sampler.py` 不直接涉及：

```text
GatedDeltaNet
recurrent state
conv state
KV Cache
state_slot_id
MRoPE
多模态图像输入
```

它和 Qwen3.6 hybrid 架构的关系是间接的。

无论模型内部是普通 dense Transformer，还是 attention + GDN 的 hybrid 结构，最后都会输出 logits：

```text
hybrid model forward
  ↓
LM Head
  ↓
logits
  ↓
Sampler
```

所以 `Sampler` 不需要知道模型内部结构。

它只需要适配 qwen3.6 执行层的采样需求，尤其是：

```text
分布式局部 logits 采样
贪心快路径
返回 score 供 rank 间比较
```

---

## 12. 和 MTP / verify probe 的关系

前面 `model_runner.py` 中 qwen3.6 版本有很多 probe / verify / MTP 相关函数。

这些函数常常需要：

```text
1. 采样 token
2. 查看 top-k
3. 保存 logits
4. 比较 logits 差异
5. 得到 token 对应 score
```

`Sampler` 返回 score 后，执行层可以更方便地做：

```text
分布式采样合并
贪心验证
MTP draft token 生成
probe 输出分析
```

所以 `sampler.py` 的改动虽然很短，但它支撑了更复杂的调试和投机/验证路径。

---

## 13. 性能角度分析

### 13.1 少一次显式 softmax

qwen3.6 版本不再显式计算：

```python
probs = torch.softmax(...)
```

而是直接：

```text
logits / temperature - log(exp_noise)
```

对采样而言，这可以减少一次概率张量构造。

不过它仍然会生成和 logits 同形状的随机噪声张量，所以它不是完全免费的。

### 13.2 更适合局部 logits

当 vocab 被 tensor parallel 切分时，每个 rank 只有一部分 vocab logits。

qwen3.6 的 score-based 采样天然适合：

```text
先在每个 shard 内取 max
再跨 rank 比 max score
```

这比先 gather 完整 logits 再采样更符合分布式执行思路。

### 13.3 greedy 路径更轻

`greedy_with_scores()` 只做：

```text
max(logits)
```

不需要 temperature，不需要 softmax，不需要随机噪声。

对于 temperature 接近 0 的确定性推理请求，性能和数值行为都更直接。

---

## 14. 原版与 qwen3.6 流程对比

### 14.1 原版采样流程

```text
logits
  ↓
logits / temperature
  ↓
softmax -> probs
  ↓
probs / Exponential(1)
  ↓
argmax
  ↓
sample_tokens
```

只返回：

```text
sample_tokens
```

---

### 14.2 qwen3.6 随机采样流程

```text
logits
  ↓
logits / temperature
  ↓
生成 Exponential(1) noise
  ↓
scores = logits / temperature - log(noise)
  ↓
max(scores)
  ↓
sample_tokens + sample_scores
```

返回：

```text
sample_tokens
sample_scores
```

---

### 14.3 qwen3.6 贪心采样流程

```text
logits
  ↓
max(logits)
  ↓
sample_tokens + sample_scores
```

---

## 15. 这个文件没有改变什么

qwen3.6 版本 `sampler.py` 没有改变以下内容：

```text
1. 采样仍然发生在 logits 之后
2. 每个请求每步仍然生成一个 token
3. temperature 仍然影响随机采样分布
4. Sampler 不关心模型结构
5. Sampler 不管理 KV Cache 或 GDN state
6. Sampler 不负责 EOS 判断
```

EOS 判断仍然是在 Scheduler / Sequence 状态更新中完成，而不是在 Sampler 中完成。

---

## 16. 面试角度应该怎么回答

如果面试官问：

> qwen3.6 版本的 `sampler.py` 相比原版做了什么改动？

可以这样回答：

`sampler.py` 负责从 LM Head 输出的 logits 中选出下一个 token。原版只有一个 `forward()`，先对 logits 除以 temperature，再做 softmax，然后通过指数噪声采样返回 token id。qwen3.6 版本把采样器扩展成支持返回 score：新增 `forward_with_scores()`，直接在 `logits / temperature` 上减去 `log(Exponential noise)`，再取 max，返回 `sample_tokens` 和 `sample_scores`；同时新增 `greedy_with_scores()`，用于 temperature 接近 0 时直接取 logits 最大值。这个改动的核心意义是配合 tensor parallel 下的分布式采样：每个 rank 只持有局部 vocab logits，可以本地采样出 token 和 score，再由 `ModelRunner.sample()` 跨 rank 汇聚 score，选出全局最终 token。它不直接涉及 GDN 或 KV Cache，但支撑了 qwen3.6 版本更复杂的分布式执行、贪心解码和 probe/MTP 路径。

---

## 17. 初学者最应该抓住的主线

这个文件可以这样记：

```text
原版 Sampler:
    logits -> softmax -> 随机采样 -> token

qwen3.6 Sampler:
    logits -> score-based 随机采样 -> token + score
    logits -> greedy max -> token + score
```

最关键的是：

```text
qwen3.6 多返回了 score。
```

这个 score 不是为了给用户看，而是为了让多卡 tensor parallel 推理时，可以在不同 rank 的局部 vocab 结果之间比较，最终选出全局 token。

---

## 18. 最终结论

qwen3.6 版本 `sampler.py` 的核心改造是：

```text
从“只返回 token 的采样器”
扩展为
“返回 token + score，并支持 greedy 快路径的采样器”。
```

它的工程意义主要有三点：

1. **支持分布式 vocab 采样**  
   每个 TP rank 可以基于局部 logits 得到局部 token 和 score，再跨 rank 汇聚选择全局 token。

2. **支持 greedy 快路径**  
   temperature 接近 0 时直接走 `greedy_with_scores()`，避免随机噪声和 softmax。

3. **支撑 probe / MTP / verify 等执行层功能**  
   qwen3.6 版本执行层需要更多采样中间信息，score-based sampler 为这些功能提供了基础。

因此，这个文件虽然代码量很小，但它和 `embed_head.py`、`model_runner.py` 的改动是配套的：

```text
embed_head.py:
    返回局部 logits

sampler.py:
    对局部 logits 采样并返回 token + score

model_runner.py:
    all_gather 各 rank token + score
    选出全局 token
```

这体现了 qwen3.6 版本在推理执行层对 tensor parallel 采样流程的重构。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
