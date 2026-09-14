# nano-vLLM 代码学习路线（精简版）

## 1. 项目定位

`nano-vLLM` 是一个极简版的大模型推理引擎。它不是训练框架，重点不是教你如何训练模型，而是帮助你理解：

- LLM 推理请求如何进入系统；
- Prefill / Decode 如何调度；
- KV Cache 如何分配、复用和释放；
- 模型前向如何执行；
- Attention 如何读写 KV Cache；
- vLLM 类推理引擎为什么比普通 `transformers.generate()` 更高效。

核心主线可以概括为：

```text
LLM.generate()
  → LLMEngine
  → Scheduler
  → BlockManager
  → ModelRunner
  → Qwen3 Model
  → Attention / KV Cache
  → Sampler
  → 输出文本
```

---

## 2. 总体阅读顺序

不要按照文件夹从上到下读，建议按“从外到内、从主流程到底层优化”的顺序阅读：

```text
1. example.py
2. bench.py
3. nanovllm/llm.py
4. nanovllm/config.py
5. nanovllm/sampling_params.py
6. engine/llm_engine.py
7. engine/sequence.py
8. engine/scheduler.py
9. engine/block_manager.py
10. engine/model_runner.py
11. models/qwen3.py
12. layers/layernorm.py
13. layers/activation.py
14. layers/rotary_embedding.py
15. layers/linear.py
16. layers/embed_head.py
17. layers/sampler.py
18. layers/attention.py
19. utils/context.py
20. utils/loader.py
```

---

## 3. 第一阶段：先看如何调用

### 3.1 `example.py`

学习目标：搞清楚用户如何使用这个推理引擎。

重点关注：

```python
llm = LLM(...)
sampling_params = SamplingParams(...)
outputs = llm.generate(prompts, sampling_params)
```

你需要搞懂：

- `LLM` 是什么；
- `SamplingParams` 控制什么；
- `generate()` 背后为什么不是简单一次 forward；
- prompt 是如何进入推理引擎的。

### 3.2 `bench.py`

学习目标：理解推理性能测试的基本方式。

重点关注：

- `num_seqs`：并发请求数；
- `max_input_len`：输入 prompt 长度；
- `max_output_len`：输出 token 长度；
- throughput：吞吐量。

这对应 AI Infra 中常见的性能指标：并发、延迟、吞吐、输入长度、输出长度。

---

## 4. 第二阶段：理解 API 和配置层

### 4.1 `nanovllm/llm.py`

这个文件很简单：

```python
class LLM(LLMEngine):
    pass
```

说明真正的核心逻辑在：

```text
engine/llm_engine.py
```

### 4.2 `nanovllm/config.py`

学习目标：理解推理引擎的全局配置。

重点参数：

| 参数 | 作用 |
|---|---|
| `max_num_batched_tokens` | 一轮最多处理多少 token |
| `max_num_seqs` | 最大并发请求数 |
| `max_model_len` | 最大上下文长度 |
| `gpu_memory_utilization` | GPU 显存使用比例 |
| `tensor_parallel_size` | 张量并行 GPU 数量 |
| `kvcache_block_size` | 每个 KV Cache block 的 token 数 |
| `num_kvcache_blocks` | KV Cache block 总数 |

### 4.3 `nanovllm/sampling_params.py`

学习目标：理解生成参数。

主要字段：

| 参数 | 作用 |
|---|---|
| `temperature` | 控制采样随机性 |
| `max_tokens` | 最大生成 token 数 |
| `ignore_eos` | 是否忽略结束符 |

---

## 5. 第三阶段：精读推理引擎主流程

### 5.1 `engine/llm_engine.py`

这是第一核心文件。

重点函数：

| 函数 | 作用 |
|---|---|
| `__init__()` | 初始化 tokenizer、scheduler、model runner |
| `add_request()` | 把 prompt 包装成 Sequence 并加入调度器 |
| `step()` | 执行一轮调度、模型推理和后处理 |
| `generate()` | 完整生成循环 |

核心流程：

```text
generate()
  → add_request()
  → while not finished:
        step()
  → tokenizer.decode()
```

你要重点理解：

- 推理不是一次性完成，而是循环 `step()`；
- 每次 `step()` 都会调度一批请求；
- 请求可能处于 Prefill、Decode 或 Finished 状态。

---

## 6. 第四阶段：理解请求对象和调度

### 6.1 `engine/sequence.py`

学习目标：理解一条用户请求在引擎内部如何表示。

重点字段：

| 字段 | 含义 |
|---|---|
| `token_ids` | prompt token + 已生成 token |
| `last_token` | Decode 阶段输入的最后一个 token |
| `num_prompt_tokens` | prompt 长度 |
| `num_cached_tokens` | 已经写入 KV Cache 的 token 数 |
| `num_scheduled_tokens` | 本轮要计算的 token 数 |
| `block_table` | 当前请求占用的 KV Cache block |
| `status` | WAITING / RUNNING / FINISHED |

一句话理解：

```text
Sequence = 一条正在推理的请求状态对象
```

### 6.2 `engine/scheduler.py`

这是第二核心文件。

学习目标：理解 Prefill / Decode 调度。

调度器维护两个队列：

| 队列 | 含义 |
|---|---|
| `waiting` | 等待 Prefill 的请求 |
| `running` | 已完成 Prefill，正在 Decode 的请求 |

核心逻辑：

```text
新请求进入 waiting
  → Prefill 完成后进入 running
  → Decode 逐 token 生成
  → 达到 EOS 或 max_tokens 后进入 finished
  → 释放 KV Cache
```

重点掌握：

- Prefill：一次性处理 prompt；
- Decode：每轮生成一个 token；
- Continuous Batching：请求可以动态加入和退出 batch；
- 调度器决定本轮处理哪些请求。

---

## 7. 第五阶段：理解 KV Cache 和 block 管理

### 7.1 `engine/block_manager.py`

这是第三核心文件。

学习目标：理解 vLLM 最核心的显存管理思想。

重点概念：

| 概念 | 含义 |
|---|---|
| Block | KV Cache 的基本分配单位 |
| block_id | 物理 block 编号 |
| block_table | Sequence 的逻辑 block 到物理 block 的映射 |
| ref_count | block 引用计数 |
| prefix caching | 相同前缀请求复用 KV Cache |

示例：

```text
block_size = 256
一个请求有 600 个 token

逻辑 block：0, 1, 2
物理 block：[5, 8, 12]

block_table = [5, 8, 12]
```

你需要重点看：

| 函数 | 作用 |
|---|---|
| `can_allocate()` | 判断新请求能否分配 KV Cache |
| `allocate()` | 为请求分配 block |
| `can_append()` | Decode 阶段判断能否追加 token |
| `may_append()` | 需要时申请新 block |
| `deallocate()` | 请求结束后释放 block |
| `hash_blocks()` | 支持 prefix caching |

---

## 8. 第六阶段：理解模型执行器

### 8.1 `engine/model_runner.py`

这是第四核心文件，也是整个仓库最复杂的文件之一。

学习目标：理解模型如何真正执行推理。

建议第一次只看这些函数：

| 函数 | 作用 |
|---|---|
| `__init__()` | 初始化分布式、加载模型、准备 KV Cache |
| `allocate_kv_cache()` | 根据显存分配 KV Cache |
| `prepare_prefill()` | 准备 Prefill 输入 |
| `prepare_decode()` | 准备 Decode 输入 |
| `run_model()` | 执行模型 forward |
| `run()` | 推理主入口，返回采样 token |

重点理解 KV Cache 形状：

```text
[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

其中：

```text
2 = K 和 V
num_layers = Transformer 层数
num_blocks = KV Cache block 数量
block_size = 每个 block 存多少 token
num_kv_heads = KV head 数量
head_dim = 每个 head 的维度
```

---

## 9. 第七阶段：理解 Qwen3 模型结构

### 9.1 `models/qwen3.py`

学习目标：理解 Decoder-only Transformer 的前向过程。

整体结构：

```text
Qwen3ForCausalLM
  → Qwen3Model
      → Embedding
      → 多层 Qwen3DecoderLayer
          → RMSNorm
          → Self-Attention
          → RMSNorm
          → MLP
      → Final RMSNorm
  → LM Head
  → Logits
```

重点类：

| 类 | 作用 |
|---|---|
| `Qwen3ForCausalLM` | 完整因果语言模型 |
| `Qwen3Model` | Transformer 主体 |
| `Qwen3DecoderLayer` | 单层 Decoder Block |
| `Qwen3Attention` | Self-Attention |
| `Qwen3MLP` | Feed Forward / SwiGLU MLP |

这一阶段重点不是优化，而是搞清楚：

```text
input_ids → embedding → attention/mlp 多层堆叠 → hidden_states → logits
```

---

## 10. 第八阶段：阅读基础 layers

建议按由易到难阅读。

### 10.1 `layers/layernorm.py`

实现 RMSNorm，以及 Add + RMSNorm 融合。

重点理解：

```text
RMSNorm = x / sqrt(mean(x^2) + eps) * weight
```

### 10.2 `layers/activation.py`

实现 SwiGLU 激活：

```text
silu(x1) * x2
```

对应 Qwen3 MLP 中的 gate/up 结构。

### 10.3 `layers/rotary_embedding.py`

实现 RoPE 旋转位置编码。

重点理解：

- RoPE 作用在 Q 和 K 上；
- 不作用在 V 上；
- 通过 cos/sin cache 加速计算。

### 10.4 `layers/linear.py`

理解 Tensor Parallel 的关键文件。

重点类：

| 类 | 作用 |
|---|---|
| `ColumnParallelLinear` | 按输出维度切分 |
| `RowParallelLinear` | 按输入维度切分，最后 all-reduce |
| `QKVParallelLinear` | 合并 Q/K/V 投影 |
| `MergedColumnParallelLinear` | 合并 gate/up 投影 |

### 10.5 `layers/embed_head.py`

实现词表并行：

| 类 | 作用 |
|---|---|
| `VocabParallelEmbedding` | 并行 embedding |
| `ParallelLMHead` | 并行输出 logits |

### 10.6 `layers/sampler.py`

负责从 logits 中采样下一个 token。

流程：

```text
logits → temperature 缩放 → softmax → 采样 token
```

---

## 11. 第九阶段：最后读 Attention

### 11.1 `layers/attention.py`

这是最难但最关键的底层文件。

学习目标：理解 Attention 如何读写 KV Cache。

主要功能：

```text
1. 把当前 token 的 K/V 写入 KV Cache
2. Prefill 阶段调用 flash_attn_varlen_func
3. Decode 阶段调用 flash_attn_with_kvcache
```

重点概念：

| 概念 | 含义 |
|---|---|
| `slot_mapping` | 当前 token 应该写入 KV Cache 的哪个物理位置 |
| `block_tables` | 每个 Sequence 的 block 映射表 |
| `context_lens` | 每条请求当前上下文长度 |
| `flash_attn_varlen_func` | Prefill 阶段处理不同长度 prompt |
| `flash_attn_with_kvcache` | Decode 阶段读取历史 KV Cache |

建议最后读这个文件，因为它依赖前面所有知识：

```text
Sequence
Scheduler
BlockManager
ModelRunner
Context
KV Cache
FlashAttention
```

---

## 12. 第十阶段：阅读工具文件

### 12.1 `utils/context.py`

作用：在 `ModelRunner` 和 `Attention` 之间传递运行时上下文。

保存的信息包括：

- 当前是否是 Prefill；
- `slot_mapping`；
- `block_tables`；
- `context_lens`；
- FlashAttention 需要的序列长度信息。

一句话理解：

```text
context.py = Attention 层读取推理运行时信息的全局通道
```

### 12.2 `utils/loader.py`

作用：加载 HuggingFace safetensors 权重，并处理权重映射。

重点理解：

```text
HF 原始权重：q_proj, k_proj, v_proj
nano-vLLM 内部：qkv_proj
```

以及：

```text
gate_proj + up_proj → gate_up_proj
```

这是推理框架中常见的权重融合方式。

---

## 13. 最重要的 6 个文件

如果时间有限，优先精读这 6 个：

```text
1. engine/llm_engine.py
2. engine/sequence.py
3. engine/scheduler.py
4. engine/block_manager.py
5. engine/model_runner.py
6. layers/attention.py
```

这 6 个文件体现的是推理引擎本身。

其次再读：

```text
1. models/qwen3.py
2. layers/linear.py
3. layers/embed_head.py
4. layers/rotary_embedding.py
5. layers/layernorm.py
6. layers/activation.py
7. layers/sampler.py
```

这些文件体现的是模型结构和基础算子。

---

## 14. 推荐学习节奏

### 第 1 遍：只看主流程

阅读：

```text
example.py
llm.py
llm_engine.py
sequence.py
scheduler.py
```

目标：搞懂请求如何从输入走到输出。

### 第 2 遍：专攻 KV Cache

阅读：

```text
sequence.py
block_manager.py
scheduler.py
model_runner.py
attention.py
```

目标：搞懂 block、block_table、slot_mapping、prefix caching。

### 第 3 遍：专攻模型结构

阅读：

```text
qwen3.py
layernorm.py
activation.py
rotary_embedding.py
linear.py
embed_head.py
sampler.py
```

目标：搞懂 Qwen3 的 Transformer forward 过程。

### 第 4 遍：专攻性能优化

阅读：

```text
model_runner.py
attention.py
linear.py
embed_head.py
```

目标：理解 FlashAttention、Tensor Parallel、CUDA Graph、KV Cache 预分配。

---

## 15. 最终记忆主线

学完这个项目后，你应该能完整讲出这条链路：

```text
example.py 调用 LLM.generate()
  ↓
LLMEngine.add_request() 创建 Sequence
  ↓
Scheduler 把请求加入 waiting 队列
  ↓
Scheduler.schedule() 决定本轮 Prefill 或 Decode
  ↓
BlockManager 分配 KV Cache block
  ↓
ModelRunner 准备 input_ids、positions、slot_mapping、block_tables
  ↓
set_context() 把运行时信息传给 Attention
  ↓
Qwen3ForCausalLM.forward()
  ↓
Attention 写入或读取 KV Cache
  ↓
lm_head 得到 logits
  ↓
Sampler 采样下一个 token
  ↓
Scheduler.postprocess() 更新 Sequence 状态
  ↓
循环 Decode，直到 EOS 或 max_tokens
  ↓
释放 KV Cache
  ↓
tokenizer.decode() 输出文本
```

一句话总结：

```text
nano-vLLM 的核心不是模型结构，而是：请求调度 + KV Cache 管理 + 模型执行 + Attention 优化。
```


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]

%% 项目关联导航：结束 %%
