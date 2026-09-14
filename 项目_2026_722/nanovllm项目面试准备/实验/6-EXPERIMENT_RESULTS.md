# 四卡 TP=4 实验结果报告

## 1. 共同环境与计量口径

| 项目 | 值 |
|---|---|
| GPU | 4 × NVIDIA GeForce RTX 3090，单卡 24576 MiB |
| NVIDIA Driver | 595.71.05 |
| Python | 3.12.3，独立环境 `/root/autodl-tmp/envs/qwenkv` |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime / cuDNN | 12.8 / 91002 |
| Triton | 3.4.0 |
| Transformers | 5.14.1 |
| FlashAttention | 2.8.3 官方预编译 wheel |
| Safetensors | 0.8.0 |
| 随机种子 | 20260806 |
| 通用推理配置 | `CUDA_VISIBLE_DEVICES=0,1,2,3`、TP=4、Eager、temperature=0、ignore_eos=true |

计时区间是同步后的推理循环，包含 scheduler、模型执行、采样和后处理；不包含模型加载、提示词构造/分词、请求入队和引擎销毁。引擎初始化耗时单独记录。每条输入在运行前按 tokenizer 的 chat template 构造并严格断言 token 长度；每条输出也严格断言指定长度。

## 2. 实验一：Qwen3.5-27B Eager 性能

配置：并发 8，每请求输入 1024 token、输出 1024 token；1 次 64-token 预热；5 次正式重复。模型位于远端 `/root/autodl-tmp/huggingface/Qwen3.5-27B`。引擎初始化耗时 `32.098 s`。

### 2.1 五次正式运行聚合

| 指标 | 均值 | 样本标准差 | 最小值 | 最大值 |
|---|---:|---:|---:|---:|
| Wall time (s) | 146.5560 | 1.5584 | 144.8639 | 148.8205 |
| Prefill time (s) | 11.3325 | 0.0634 | 11.2507 | 11.4271 |
| Decode time (s) | 135.2090 | 1.5112 | 133.5111 | 137.3786 |
| Prefill throughput (tok/s) | 722.8960 | 4.0368 | 716.8911 | 728.1327 |
| Decode throughput (tok/s) | 60.5346 | 0.6745 | 59.5726 | 61.2983 |
| E2E output throughput (tok/s) | 55.9018 | 0.5924 | 55.0462 | 56.5496 |
| E2E total-token throughput (tok/s) | 111.8035 | 1.1848 | 110.0924 | 113.0993 |
| Request throughput (req/s) | 0.054592 | 0.000579 | 0.053756 | 0.055224 |

### 2.2 请求延迟分布

| 指标 | 均值 | 最小值 | P50 | P95 | 最大值 |
|---|---:|---:|---:|---:|---:|
| TTFT (s) | 11.3325 | 11.2507 | 11.3334 | 11.4271 | 11.4271 |
| TPOT (s/token) | 0.132183 | 0.130523 | 0.132452 | 0.134304 | 0.134304 |
| Request latency (s) | 146.5560 | 144.8639 | 146.8317 | 148.8205 | 148.8205 |

校验结果：5/5 批次完整；40/40 正式请求均生成精确 1024 token；无压缩模式保持关闭。

## 3. 实验二：Qwen3.5-27B KV Cache 压缩开关对照

最终方案按用户更新后的要求执行：只测并发 8，每请求输入 2048 token、输出 2048 token；disabled 和 enabled 分别新建引擎，各执行 1 次 66-token 预热和 1 次正式运行。旧的并发 1/2/4/8 长扫描已主动终止，未恢复，也不参与本报告。

压缩配置为 period=64、window=4 blocks（1024 token）、keep=2 blocks（512 token）、keep ratio=0.5、top-k=8、smoothing window=5、sink token=1、min tokens=1025、block size=256。除 `kv_compress_enabled` 外，两组核心引擎配置一致。

### 3.1 正式运行绝对值

| 指标 | Disabled | Enabled |
|---|---:|---:|
| 引擎初始化 (s) | 30.8885 | 30.3330 |
| Wall time (s) | 299.4788 | 299.0877 |
| Prefill time (s) | 19.3723 | 19.2839 |
| Decode time (s) | 280.0724 | 279.7707 |
| 压缩步骤累计时间 (s) | 0 | 1.4536 |
| Prefill throughput (tok/s) | 845.7446 | 849.6206 |
| Decode throughput (tok/s) | 58.4706 | 58.5337 |
| E2E output throughput (tok/s) | 54.7084 | 54.7799 |
| E2E total-token throughput (tok/s) | 109.4167 | 109.5598 |
| Request throughput (req/s) | 0.026713 | 0.026748 |
| TTFT mean / P50 / P95 (s) | 14.5325 / 14.5325 / 19.3723 | 14.4599 / 14.4599 / 19.2840 |
| TPOT mean / P50 / P95 (s/token) | 0.139202 / 0.139202 / 0.141566 | 0.139046 / 0.139046 / 0.141403 |
| 请求延迟均值 (s) | 299.4788 | 299.0877 |
| 峰值活跃 KV blocks | 128 | 80 |
| 峰值物理 KV tokens | 32760 | 20472 |
| 峰值逻辑 tokens | 32760 | 32760 |
| Preemption / re-prefill | 0 / 0 | 0 / 0 |
| 压缩事件 | 0 | 24 |
| 累计释放 KV blocks | 0 | 48 |
| 累计丢弃物理 KV tokens | 0 | 12288 |

### 3.2 开启压缩后的相对变化

- Decode throughput：`+0.1079%`。
- E2E output throughput：`+0.1308%`。
- 峰值活跃 KV blocks：降低 `37.5000%`。
- 峰值物理 KV tokens：降低 `37.5092%`。
- 逻辑长度保持 32760，而物理 KV 长度降至 20472，证明压缩没有篡改逻辑历史长度。
- 两组 preemption 均为 0；enabled 组确实发生 24 次压缩并释放 48 blocks，结果有效性标记为 true。

这里的吞吐差异接近噪声量级，主要结论是本配置下取得约 37.5% 的 KV 存储峰值下降，而未观察到明显的吞吐损失。由于最终方案每组只有一次正式运行，不能把微小的正向吞吐变化解释为稳定加速。

## 4. 实验三：Qwen3.6-27B-FP8 MTP-1 接受率

配置：Qwen3.6-27B-FP8、TP=4、Eager、MTP=1；输入 512 token、输出 128 token；5 条预热提示、50 条正式提示；每个验证回合使用一个 draft token。模型位于远端 `/root/autodl-tmp/huggingface/Qwen3.6-27B-FP8`。权重加载审计显示 MTP 权重 `loaded=15, skipped=0`。

### 4.1 接受率

| 指标 | 结果 |
|---|---:|
| 正式提示数 | 50 |
| 正式输出 token | 6400 |
| Verify attempts | 3200 |
| Accepted draft tokens | 2039 |
| Rejected draft tokens | 1161 |
| 总接受率 | 63.71875% |
| 每提示接受率 mean / min / P50 / P95 / max | 63.7188% / 46.8750% / 64.0625% / 77.4219% / 81.2500% |

接受率定义为 `accepted_draft_tokens / verify_attempts`。每个 128-token 输出包含 64 次单 draft 验证，因此 50 条提示应有 3200 次；实际计数完全一致。

### 4.2 辅助性能指标

| 指标 | 均值 | 最小值 | P50 | P95 | 最大值 |
|---|---:|---:|---:|---:|---:|
| TTFT (s) | 0.9541 | 0.9290 | 0.9503 | 0.9780 | 1.0044 |
| 单请求 wall time (s) | 18.4174 | 17.9073 | 18.3448 | 18.9782 | 19.1419 |

正式数据集 wall time 为 `920.883 s`，按数据集总时长计算的输出吞吐为 `6.9499 tok/s`。所有 50 条输出长度均为 128；accepted+rejected 与 verify attempts 对账一致。top-k=5 仅用于调试记录，本实验不声称 MTP 推理加速。

## 5. 复现命令

在远端 shell 中执行：

```bash
source /root/autodl-tmp/envs/qwenkv/bin/activate
cd /root/autodl-tmp/nano-vllm-qwen3.6
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -u experiments/tp4/qwen35_eager_tp4.py \
  --model /root/autodl-tmp/huggingface/Qwen3.5-27B \
  --devices 0,1,2,3 \
  --output outputs/tp4_experiments/qwen35_eager_tp4.json

python -u experiments/tp4/qwen35_kv_compression_tp4.py \
  --model /root/autodl-tmp/huggingface/Qwen3.5-27B \
  --devices 0,1,2,3 \
  --output outputs/tp4_experiments/qwen35_kv_compression_tp4.json

python -u experiments/tp4/qwen36_mtp1_acceptance_tp4.py \
  --model /root/autodl-tmp/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --output outputs/tp4_experiments/qwen36_mtp1_acceptance_tp4.json
```

JSON 保存了每个请求、每个重复、逐步计时、token ID、输入哈希、调度器/压缩计数和硬件环境；本报告中的数值均可由这些原始字段复算。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nanovllm项目面试准备|模块-nanovllm项目面试准备]]

%% 项目关联导航：结束 %%
