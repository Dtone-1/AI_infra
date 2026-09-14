# Nano-vLLM Qwen3.5/Qwen3.6

这是一个基于 `nano-vllm` 的小型、可读性较强的推理引擎，在原项目基础上扩展了 Qwen3.5 混合架构模型支持，以及 Qwen3.6 FP8 文本推理实验。

本仓库的目标是帮助学习 LLM 推理引擎的工作原理。张量并行、KV Cache 分配、CUDA Graph 解码、混合线性注意力状态以及量化 checkpoint 加载等内容，都被实现到了一个小型代码库中。

## 已支持功能

- 通过原始 Nano-vLLM 路径支持 Qwen3 dense 文本模型。
- 支持 Qwen3.5-9B BF16 文本推理。
- 通过本地视觉编码器路径支持 Qwen3.5-9B 多模态 smoke test。
- 在 4 张 RTX 4090 上支持 Qwen3.6-27B-FP8 文本推理。
- 支持 attention、MLP、词表 embedding/head 以及 GatedDeltaNet 的张量并行。
- 当 `enforce_eager=False` 时支持 CUDA Graph decode。
- 支持 rank-local FP8 checkpoint 加载：每个 TP rank 只切分并反量化自己负责的权重 shard。
- 实验性支持 Qwen3.6 MTP 权重加载、单步 forward 探测，以及 MTP-1 draft/verify 原型。

## 当前限制

- Qwen3.6-27B-FP8 在 FP8 block 反量化后会以 BF16 权重形式加载。目前还没有实现原生 FP8 matmul kernel。
- Qwen3.6-27B-FP8 当前只面向文本推理。请使用 `enable_vision=False`。
- Qwen3.6-27B-FP8 已在 4 张 RTX 4090 上以 `tensor_parallel_size=4` 验证通过。在当前 BF16 常驻权重实现下，TP=2 无法放入 24GB 显存卡中。
- Qwen3.6-27B-FP8 加载较慢，因为原始 FP8 checkpoint 会在启动时转换。如果预先转换为 TP-sharded checkpoint，启动会更快。
- MTP 当前只是一个原型，用于权重加载、单步 draft token 探测和 MTP-1 接受率测量。它目前还不能带来 decode 加速。
- 这不是一个生产级 serving stack，而是一个研究/学习型实现。

## 仓库结构

```text
nanovllm/
  engine/          scheduler、model runner、block/state managers
  layers/          attention、linear layers、GatedDeltaNet、sampler
  models/          Qwen3 和 Qwen3.5 模型定义
  utils/           checkpoint loader、FP8 dequant helpers、context utilities
examples/          原始示例
run_text_qwen35_v2.py
run_text_qwen36_fp8.py
test_mtp_forward.py
test_mtp1_verify.py
test_mtp1_spec_decode.py
test_mtp_spec_decode.py
test_state_rollback.py
bench_qwen35_fixed.py
```

## 安装

使用 Python 3.10-3.12，并安装支持 CUDA 的 PyTorch。真实推理需要 GPU，以及 `torch`、`triton` 和 `flash-attn`。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

如果你的环境中已经安装了 PyTorch、Triton 和 FlashAttention，那么只需要执行 editable install 即可。

## 模型下载

请将模型权重放在仓库之外。示例默认模型位于 `~/huggingface`。

```bash
hf download Qwen/Qwen3.5-9B \
  --local-dir ~/huggingface/Qwen3.5-9B \
  --max-workers 8

hf download Qwen/Qwen3.6-27B-FP8 \
  --local-dir ~/huggingface/Qwen3.6-27B-FP8 \
  --max-workers 8
```

不要将模型权重、生成的缓存、本地图片或 benchmark 日志提交到 git。

## 快速开始

Qwen3.5-9B 文本 smoke test：

```bash
python run_text_qwen35_v2.py \
  --model ~/huggingface/Qwen3.5-9B \
  --tp 1
```

使用 4 路张量并行运行 Qwen3.5-9B：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_text_qwen35_v2.py \
  --model ~/huggingface/Qwen3.5-9B \
  --tp 4
```

在 4 张 RTX 4090 上运行 Qwen3.6-27B-FP8 文本推理：

```bash
python run_text_qwen36_fp8.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --tp 4
```

调试时禁用 CUDA Graph：

```bash
python run_text_qwen36_fp8.py --eager
```

自定义 prompt：

```bash
python run_text_qwen36_fp8.py \
  --prompt "你好，请用三句话介绍你自己。然后讲一个简短笑话。"
```

Qwen3.6 MTP 单步 forward 探测：

```bash
python test_mtp_forward.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --tp 4 \
  --top-k 5
```

Qwen3.6 MTP-1 draft/verify 原型：

```bash
python test_mtp1_verify.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --tp 4 \
  --max-tokens 64
```

带有 accept/reject 状态控制的 Qwen3.6 MTP-1 speculative decode 原型：

```bash
python test_mtp1_spec_decode.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --tp 4 \
  --max-tokens 64
```

使用 `--force-reject-attempt 1` 可以强制触发一次 reject 路径，并验证状态回滚。

带有 batch verify、greedy 对齐和开销统计的 Qwen3.6 多 token MTP speculative decode 原型：

```bash
python test_mtp_spec_decode.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --tp 4 \
  --draft-len 4 \
  --verify-mode graph \
  --max-tokens 64
```

该原型将 draft 验证组织为 batch 级别的 accept/reject 统计。`--verify-mode graph` 会将 verify length 1-4 的 bucket 捕获为 CUDA graph，因此每次 verify 调用都会通过一个 graph replay 内部的顺序 decode 步骤。这可以减少 Python/launch 开销，但它不是 fused parallel GDN verify kernel。

`--verify-mode chunk` 是一个实验性的 continuation-prefill verifier。它会先执行 chunk probe，然后恢复 decode state，并使用可信的 graph/eager verify 路径做 accept/reject 和 state-commit 决策，以保证 greedy 输出保持对齐。当 CUDA Graph 启用时，chunk verify length 1-4 的 bucket 会被捕获为 CUDA graph。原始 chunk logits 仍会与可信路径进行比较，并且可能存在差异，因此该模式主要用于在实现 fused GDN chunk kernel 之前研究 chunk-verify 语义和开销。

不带 top-k/logit-diff probe 的 fast-path MTP decode benchmark：

```bash
python run_mtp_fast_decode.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --tp 4 \
  --draft-len 2 \
  --verify-mode graph \
  --max-tokens 128 \
  --compare-greedy
```

Draft length sweep：

```bash
python bench_mtp_draft_sweep.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --tp 4 \
  --draft-lens 1,2,3,4 \
  --verify-mode graph \
  --max-tokens 128
```

Decode-state rollback smoke test：

```bash
python test_state_rollback.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --tp 4
```

## API 示例

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/path/to/model",
    tensor_parallel_size=1,
    enforce_eager=True,
)
params = SamplingParams(temperature=0.7, max_tokens=128)
outputs = llm.generate(["Hello, Nano-vLLM."], params)
print(outputs[0]["text"])
```

## Benchmarks

Qwen3.5 单 prompt 计时辅助脚本：

```bash
python bench_qwen35_fixed.py \
  --model ~/huggingface/Qwen3.5-9B \
  --tp 4 \
  --max-tokens 256 \
  --repeats 3
```

最近在 RTX 4090 硬件上的本地 smoke-test 结果：

```text
Qwen3.6-27B-FP8, TP=4, CUDA Graph decode: Decode ~= 41 tok/s
Qwen3.5-9B, TP=4, CUDA Graph decode:       Decode ~= 98 tok/s
```

这些只是简单的单请求 smoke test，不是完整的 serving benchmark。

## 开发检查

在没有模型权重的情况下进行语法检查：

```bash
python -m compileall nanovllm examples run_text_qwen35_v2.py run_text_qwen36_fp8.py test_mtp_forward.py test_mtp1_verify.py test_mtp1_spec_decode.py test_mtp_spec_decode.py test_state_rollback.py
```

有用的运行时检查：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

## 开源说明

- 本项目使用 MIT 许可证。请保留现有的 `LICENSE` 文件。
- `pyproject.toml` 指向该 fork，并保留了到原始 upstream 项目的链接。
- 请将大型 artifact 保留在 git 仓库之外。`.gitignore` 已排除常见模型/checkpoint 文件。
- 在 issues/PRs 中报告推理结果时，应包含硬件假设。

## 致谢

本工作基于 Xingkai Yu 的原始 Nano-vLLM 项目，并保留 MIT 许可证。Qwen 模型权重由 Qwen 单独分发，并受其自身模型条款约束。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
