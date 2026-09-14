# Qwen3.5-27B Hybrid Eager 推理效率实验

# 一、实验描述

## 1.1 实验名称

**Qwen3.5-27B Hybrid Eager 推理效率实验**

## 1.2 实验目的

本实验在 **4 张 RTX 3090、Tensor Parallel Size（TP）=4、Eager 模式**下，验证项目适配后的 Qwen3.5-27B Hybrid 文本推理路径能够稳定运行

## 1.4 工作负载定义

准备 8 条固定的纯文本 Prompt，要求：

- 每条 Prompt 在应用 Chat Template 和 Generation Prompt 后，实际长度严格等于 **1024 Token**；
- 8 条 Prompt 的内容互不相同，避免重复输入造成测试偏差；
- 不使用图片，不触发 Vision Encoder；
- 不使用过短的重复字符构造输入；固定并发 8：使用第 1～8 条。

每个请求固定生成 **1024 Token**。设置 `temperature=0` 和 `ignore_eos=True`，避免采样随机性或提前生成 EOS 导致不同请求的输出长度不一致

## 1.6 实验边界

本实验明确不测试以下内容：

- 不测试 CUDA Graph；
- 不测试 KV Cache 压缩；
- 不测试 MTP 投机解码；
- 不测试图片输入或 Vision Encoder；

因此，本实验最终能够支持的结论是：

> 在 4×RTX 3090、TP=4、Eager 模式下，项目适配后的 Qwen3.5-27B 能够完成 1K 输入、1K 输出的文本推理；在并发数8 下测得 TTFT、TPOT、Prefill Throughput、Decode Throughput 和端到端吞吐。

# 二、测试配置

## 2.2 模型与引擎配置

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

## 2.3 输入与采样配置

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
|               |                       |

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

## 2.5 预热与重复次数

每个并发档位先进行 1 次预热，再进行 5 次正式测量：

| 阶段 | 次数 | 输出长度 | 是否计入结果 |
|---|---:|---:|---|
| 并发档位预热 | 1 | 64 Token | 否 |
| 正式测试 | 5 | 1024 Token | 是 |


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nanovllm项目面试准备|模块-nanovllm项目面试准备]]

%% 项目关联导航：结束 %%
