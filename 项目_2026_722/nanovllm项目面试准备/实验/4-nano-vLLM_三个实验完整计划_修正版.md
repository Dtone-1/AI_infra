# nano-vLLM 融合项目实验计划

本文档统一规定三个正式实验的测试内容、固定配置和结果记录格式。所有实验均使用 **4×RTX 3090、TP=4**，各实验只回答自身范围内的问题。

# 实验一

## Qwen3.5-27B Hybrid Eager 推理效率实验

## 一、实验描述

### 1.1 实验名称

**Qwen3.5-27B Hybrid Eager 推理效率实验**

### 1.2 实验目的

本实验在 **4 张 RTX 3090、Tensor Parallel Size（TP）=4、Eager 模式**下，验证项目适配后的 Qwen3.5-27B Hybrid 文本推理路径能够稳定运行

### 1.4 工作负载定义

准备 8 条固定的纯文本 Prompt，要求：

- 每条 Prompt 在应用 Chat Template 和 Generation Prompt 后，实际长度严格等于 **1024 Token**；
- 8 条 Prompt 的内容互不相同，避免重复输入造成测试偏差；
- 不使用图片，不触发 Vision Encoder；
- 不使用过短的重复字符构造输入；固定并发 8：使用第 1～8 条。

每个请求固定生成 **1024 Token**。设置 `temperature=0` 和 `ignore_eos=True`，避免采样随机性或提前生成 EOS 导致不同请求的输出长度不一致

### 1.6 实验边界

本实验明确不测试以下内容：

- 不测试 CUDA Graph；
- 不测试 KV Cache 压缩；
- 不测试 MTP 投机解码；
- 不测试图片输入或 Vision Encoder；

因此，本实验最终能够支持的结论是：

> 在 4×RTX 3090、TP=4、Eager 模式下，项目适配后的 Qwen3.5-27B 能够完成 1K 输入、1K 输出的文本推理；在并发数8 下测得 TTFT、TPOT、Prefill Throughput、Decode Throughput 和端到端吞吐。

## 二、测试配置

### 2.2 模型与引擎配置

| 配置项 | 固定值 | 说明 |
|---|---:|---|
| 模型 | Qwen3.5-27B | 所有并发档位使用同一 Checkpoint |
| `tensor_parallel_size` | 4 | 四张 3090 |
| `enforce_eager` | `True` | 不捕获或回放 CUDA Graph |
| `max_model_len` | 2048 | 1024 输入 + 1024 输出 |
| `max_num_batched_tokens` | 8192 | 允许并发 8 的全部 Prompt 一次 Prefill |
| `max_num_seqs` | 8 | 本实验并发 |
| `gpu_memory_utilization` | 0.88 | 为运行时、NCCL 和 GDN state 留余量 |
| `enable_vision` | `False` | 纯文本实验 |
| `enable_mtp` | `False` | 关闭 MTP |
| `kv_compress_enabled` | `False` | 关闭 KV 压缩 |
| Prefix Cache | 自动关闭 | Hybrid 模型配置下保持关闭 |
| KV Block Size | 256 Token | 使用项目默认值 |

### 2.3 输入与采样配置

| 项目 | 固定值 |
|---|---:|
| 输入格式 | 纯文本 Chat Template |
| 实际输入长度 | 每条严格为 1024 Token |
| 输出长度 | 每条严格为 1024 Token |
| `temperature` | 0.0 |
| `max_tokens` | 1024 |
| `ignore_eos` | `True` |
| 随机种子 | 固定并写入结果文件 |
| Prompt 数量 | 固定 8 条 |

采样参数：

```python
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=1024,
    ignore_eos=True,
)
```

脚本必须在加入请求前断言：

```python
assert len(prompt_token_ids) == 1024
```

这里的 1024 指经过 Chat Template 后的实际 Token 数，不是字符数，也不是原始文本分词前的长度。

### 2.5 预热与重复次数

每个并发档位先进行 1 次预热，再进行 5 次正式测量：

| 阶段 | 次数 | 输出长度 | 是否计入结果 |
|---|---:|---:|---|
| 并发档位预热 | 1 | 64 Token | 否 |
| 正式测试 | 5 | 1024 Token | 是 |

---

# 实验二

## Qwen3.5-27B 动态 KV Cache 压缩实验

## 一、实验描述

### 1.1 实验名称

**Qwen3.5-27B 动态 KV Cache 压缩实验**

### 1.2 实验目的

本实验在 **4 张 RTX 3090、TP=4、Eager 模式**下，对同一 Qwen3.5-27B 分别关闭和开启 KV Cache 压缩，在完全相同的长上下文负载中比较：

1. Decode Throughput 和端到端吞吐量变化；
2. 不同并发下的请求完成率、抢占次数及零抢占并发；
3. 峰值活跃 KV Block、物理 KV Token 和累计释放 Block 数量。

### 1.4 工作负载定义

准备 8 条固定纯文本 Prompt，每条经过 Chat Template 后严格为 **16384 Token**，每个请求固定生成 **2048 Token**。脚本依次扫描并发数 **1、2、4、8**，每个并发档位均运行压缩关闭和压缩开启两种模式。

压缩开启时固定使用：

- 压缩周期：64 个 Decode Step；
- 压缩窗口：4 个 KV Block，即 1024 Token；
- 保留比例：0.5，即每个窗口保留 512 Token；
- Sink Token：1；
- Top-k：8，允许同一压缩周期处理全部活跃请求。

每组有效结果必须满足 `compression_event_count > 0`。若未发生真实压缩，该组结果不计入压缩收益。

### 1.6 实验边界

本实验只压缩 Full Attention 层的 KV Cache，不压缩 Gated DeltaNet 的 recurrent/conv state。底层 KV Tensor 在初始化时预分配，因此主要报告**活跃 KV Block 和物理 KV Token 的减少**，不直接表述为 `nvidia-smi` 显存下降。

本实验不测试 MTP、多模态输入和模型质量；压缩开启与关闭的输出 Token 仅保存用于后续检查，不在本实验中评价任务精度。

## 二、测试配置

### 2.2 模型与引擎配置

| 配置项 | 固定值 | 说明 |
|---|---:|---|
| 模型 | Qwen3.5-27B | 两种模式使用同一 Checkpoint |
| `tensor_parallel_size` | 4 | 四张 3090 |
| `enforce_eager` | `True` | 避免 Graph 路径干扰对照 |
| `max_model_len` | 18432 | 16K 输入 + 2K 输出 |
| `max_num_batched_tokens` | 8192 | 使用 Chunked Prefill |
| `max_num_seqs` | 8 | 测试并发 |
| `gpu_memory_utilization` | 0.88 | 两种模式保持一致 |
| `enable_vision` | `False` | 纯文本实验 |
| `enable_mtp` | `False` | MTP 与压缩互斥 |
| `kv_compress_enabled` | `False / True` | 唯一核心对照变量 |
| KV Block Size | 256 Token | 使用项目默认值 |

### 2.3 输入与采样配置

| 项目 | 固定值 |
|---|---:|
| 实际输入长度 | 每条严格为 16384 Token |
| 输出长度 | 每条严格为 2048 Token |
| 并发数 | 1、2、4、8 |
| `temperature` | 0.0 |
| `max_tokens` | 2048 |
| `ignore_eos` | `True` |
| Prompt 数量 | 固定 8 条 |
| 正式重复次数 | 每种模式、每个并发档位 3 次 |

### 2.4 Scheduler 计数器

在 Scheduler 中增加以下计数器，每轮测试开始前清零，结束后写入结果：

| 计数器 | 统计内容 |
|---|---|
| `preemption_count` | `Scheduler.preempt()` 被调用的次数 |
| `re_prefill_count` | 请求被抢占后重新进入 Prefill 的次数 |
| `re_prefill_tokens` | 抢占重算过程中累计重新计算的 Token 数 |

### 2.5 预热与执行顺序

模型初始化后先执行一次短预热，不计入结果。脚本按并发数 **1、2、4、8** 依次扫描，每个并发档位交错运行压缩关闭和开启两种模式。

两种模式除 `kv_compress_enabled` 及压缩参数外，其余配置、Prompt 和执行顺序保持一致。

| 汇总指标 | 计算方式 | 结果 |
|---|---|---:|
| 吞吐量变化 | 开启相对关闭的 Decode Throughput 变化率 | 待测 |
| KV Block 收益 | `1 - 开启峰值 Block / 关闭峰值 Block` | 待测 |
| 零抢占最高并发 | 在 1、2、4、8 中完成率 100% 且无抢占的最大值 | 待测 |

---

# 实验三

## Qwen3.6-27B-FP8 MTP-1 接受率实验

## 一、实验描述

### 1.1 实验名称

**Qwen3.6-27B-FP8 MTP-1 接受率实验**

### 1.2 实验目的

本实验在 **4 张 RTX 3090、TP=4、Eager 模式**下，测量 Qwen3.6-27B-FP8 的单 Draft Token 接受率，验证 MTP 预测的下一个 Token 与主模型 Verify 结果的一致程度。

接受率定义为：

\[
\text{Accept Rate}
=
\frac{\text{Accepted Draft Tokens}}
{\text{Verify Attempts}}
\times 100\%
\]

### 1.4 工作负载定义

准备 50 条固定文本 Prompt，中文生成任务。每条 Prompt 经过 Chat Template 后严格为 **512 Token**，每条固定生成 **128 Token**。

测试采用 MTP-1：

1. 主模型生成当前正式 Token；
2. MTP 预测下一个 Draft Token；
3. 主模型生成真实的下一个 Token；
4. 两者相同记为 Accepted，不同记为 Rejected。

先使用 5 条额外 Prompt 预热，再对 50 条正式 Prompt 顺序测试。模型只加载一次。

### 1.6 实验边界

本实验只测量 **MTP-1 Top-1 接受率**，不测试多 Token Draft Length Sweep，也不以接受率直接宣称获得推理加速。

本实验不启用 KV Cache 压缩和视觉输入；`top_k=5` 只用于调试输出，接受判断仍以 Draft Top-1 与主模型 Token 是否相同为准。

## 二、测试配置

### 2.2 模型与引擎配置

| 配置项 | 固定值 | 说明 |
|---|---:|---|
| 模型 | Qwen3.6-27B-FP8 | 使用仓库 MTP 对应 Checkpoint |
| `tensor_parallel_size` | 4 | 四张 3090 |
| `enforce_eager` | `True` | 使用基础验证路径 |
| `max_model_len` | 1024 | 512 输入 + 128 输出并保留余量 |
| `max_num_batched_tokens` | 512 | Prompt 一次完成 Prefill |
| `max_num_seqs` | 1 | 顺序测量接受率 |
| `gpu_memory_utilization` | 0.85 | 为 MTP 与运行时留余量 |
| `enable_vision` | `False` | 纯文本实验 |
| `enable_mtp` | `True` | 开启 MTP |
| `kv_compress_enabled` | `False` | 与 MTP 互斥 |
| `top_k` | 5 | 仅记录调试信息 |

### 2.3 输入与采样配置

| 项目 | 固定值 |
|---|---:|
| 正式 Prompt 数量 | 50 条 |
| 实际输入长度 | 每条严格为 512 Token |
| 输出长度 | 每条固定为 128 Token |
| `temperature` | 0.0 |
| `max_tokens` | 128 |
| `ignore_eos` | `True` |
| 预热 Prompt | 5 条，不计入结果 |
| 正式测试次数 | 固定数据集运行 1 次 |


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nanovllm项目面试准备|模块-nanovllm项目面试准备]]

%% 项目关联导航：结束 %%
