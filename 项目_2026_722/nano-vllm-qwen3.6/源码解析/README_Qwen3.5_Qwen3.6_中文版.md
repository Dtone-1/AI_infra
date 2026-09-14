# Nano-vLLM Qwen3.5/Qwen3.6

一个基于 `nano-vllm` 的紧凑、易读推理引擎，并进一步扩展以支持 Qwen3.5 混合架构模型，以及 Qwen3.6 FP8 纯文本推理实验。

本仓库主要用于学习大语言模型推理引擎的工作原理。张量并行、KV Cache 分配、CUDA Graph 解码、混合线性注意力状态以及量化检查点加载等功能，均在一个较小的代码库中实现。

## 已支持的功能

- 通过原始 Nano-vLLM 路径支持 Qwen3 Dense 纯文本模型。
- 支持 Qwen3.5-9B BF16 纯文本推理。
- 支持通过本地视觉编码器路径执行 Qwen3.5-9B 多模态冒烟测试。
- 支持在 4 张 RTX 4090 上运行 Qwen3.6-27B-FP8 纯文本推理。
- 支持 Attention、MLP、词表 Embedding/输出头以及 GatedDeltaNet 的张量并行。
- 当 `enforce_eager=False` 时支持 CUDA Graph 解码。
- 支持各 Rank 本地加载 FP8 检查点：每个 TP Rank 只切分并反量化自己持有的权重分片。
- 实验性支持 Qwen3.6 MTP 权重加载、单步前向探测，以及 MTP-1 Draft/Verify 原型。

## 当前限制

- Qwen3.6-27B-FP8 的 FP8 Block 在反量化后，会以 BF16 权重形式加载。当前尚未实现原生 FP8 矩阵乘 Kernel。
- Qwen3.6-27B-FP8 当前仅面向纯文本推理，请设置 `enable_vision=False`。
- Qwen3.6-27B-FP8 已在 4 张 RTX 4090、`tensor_parallel_size=4` 的环境下完成验证。由于当前实现会将权重常驻为 BF16，TP=2 无法装入单卡 24 GB 显存。
- Qwen3.6-27B-FP8 加载速度较慢，因为启动时需要转换原始 FP8 检查点。若预先生成经过转换的 TP 分片检查点，可以加快启动速度。
- 当前 MTP 仅是用于权重加载、单步 Draft Token 探测和 MTP-1 接受率测量的原型，暂时不能带来解码加速。
- 本项目不是生产级在线服务框架，而是一个面向研究和学习的实现。

## 仓库结构

```text
nanovllm/
  engine/          调度器、模型运行器、Block/状态管理器
  layers/          Attention、线性层、GatedDeltaNet、采样器
  models/          Qwen3 和 Qwen3.5 模型定义
  utils/           检查点加载器、FP8 反量化辅助函数、上下文工具
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

请使用 Python 3.10～3.12，并安装支持 CUDA 的 PyTorch。执行真实推理还需要 GPU，以及 `torch`、`triton` 和 `flash-attn`。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

如果当前环境已经安装 PyTorch、Triton 和 FlashAttention，只需执行可编辑安装即可。

## 模型下载

请将模型权重保存在仓库目录之外。以下示例默认使用 `~/huggingface` 目录。

```bash
hf download Qwen/Qwen3.5-9B \
  --local-dir ~/huggingface/Qwen3.5-9B \
  --max-workers 8

hf download Qwen/Qwen3.6-27B-FP8 \
  --local-dir ~/huggingface/Qwen3.6-27B-FP8 \
  --max-workers 8
```

不要将模型权重、生成的缓存、本地图像或性能测试日志提交到 Git 仓库中。

## 快速开始

Qwen3.5-9B 纯文本冒烟测试：

```bash
python run_text_qwen35_v2.py \
  --model ~/huggingface/Qwen3.5-9B \
  --tp 1
```

使用四路张量并行运行 Qwen3.5-9B：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_text_qwen35_v2.py \
  --model ~/huggingface/Qwen3.5-9B \
  --tp 4
```

在 4 张 RTX 4090 上运行 Qwen3.6-27B-FP8 纯文本推理：

```bash
python run_text_qwen36_fp8.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --tp 4
```

调试时关闭 CUDA Graph：

```bash
python run_text_qwen36_fp8.py --eager
```

自定义 Prompt：

```bash
python run_text_qwen36_fp8.py \
  --prompt "你好，请用三句话介绍你自己。然后讲一个简短笑话。"
```

Qwen3.6 MTP 单步前向探测：

```bash
python test_mtp_forward.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --tp 4 \
  --top-k 5
```

Qwen3.6 MTP-1 Draft/Verify 原型：

```bash
python test_mtp1_verify.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --tp 4 \
  --max-tokens 64
```

带 Accept/Reject 状态控制的 Qwen3.6 MTP-1 投机解码原型：

```bash
python test_mtp1_spec_decode.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --tp 4 \
  --max-tokens 64
```

使用 `--force-reject-attempt 1` 可以强制触发一次 Reject 路径，并验证状态回滚是否正确。

支持批量 Verify、贪心输出对齐和额外开销统计的 Qwen3.6 多 Token MTP 投机解码原型：

```bash
python test_mtp_spec_decode.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --tp 4 \
  --draft-len 4 \
  --verify-mode graph \
  --max-tokens 64
```

该原型将 Draft Token 的验证过程组织为 Batch 级别的 Accept/Reject 统计。`--verify-mode graph` 会将长度为 1～4 的 Verify Bucket 捕获为 CUDA Graph，因此每次 Verify 调用都可以通过一次 Graph Replay，执行内部连续的顺序 Decode 步骤。这样可以减少 Python 调度和 Kernel Launch 开销，但它并不是融合后的并行 GDN Verify Kernel。

`--verify-mode chunk` 是一个实验性的 Continuation-Prefill 验证器。它会先执行一次 Chunk 探测，然后恢复 Decode 状态，并使用可信的 Graph/Eager Verify 路径做出 Accept/Reject 和状态提交决策，从而保持贪心输出一致。当启用 CUDA Graph 时，长度为 1～4 的 Chunk Verify Bucket 也会被捕获为 CUDA Graph。原始 Chunk Logits 仍会与可信路径进行对比，并且二者可能存在差异。因此，该模式主要用于在融合 GDN Chunk Kernel 实现之前，研究 Chunk Verify 的语义和额外开销。

不执行 Top-K/Logit 差异探测的快速路径 MTP Decode Benchmark：

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

Draft 长度扫描测试：

```bash
python bench_mtp_draft_sweep.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --tp 4 \
  --draft-lens 1,2,3,4 \
  --verify-mode graph \
  --max-tokens 128
```

Decode 状态回滚冒烟测试：

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

## 性能测试

Qwen3.5 单 Prompt 计时工具：

```bash
python bench_qwen35_fixed.py \
  --model ~/huggingface/Qwen3.5-9B \
  --tp 4 \
  --max-tokens 256 \
  --repeats 3
```

近期在 RTX 4090 硬件上的本地冒烟测试结果：

```text
Qwen3.6-27B-FP8，TP=4，CUDA Graph Decode：约 41 tok/s
Qwen3.5-9B，TP=4，CUDA Graph Decode：      约 98 tok/s
```

这些结果仅来自简单的单请求冒烟测试，并非完整的在线服务 Benchmark。

## 开发检查

在没有模型权重的情况下执行语法检查：

```bash
python -m compileall nanovllm examples run_text_qwen35_v2.py run_text_qwen36_fp8.py test_mtp_forward.py test_mtp1_verify.py test_mtp1_spec_decode.py test_mtp_spec_decode.py test_state_rollback.py
```

常用运行环境检查：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

## 开源说明

- 本项目采用 MIT 许可证，请保留现有的 `LICENSE` 文件。
- `pyproject.toml` 指向当前 Fork，并保留了原始上游项目的链接。
- 请将大型文件保存在 Git 仓库之外。`.gitignore` 已排除常见的模型和检查点文件。
- 在 Issue 或 Pull Request 中报告推理结果时，应同时说明所使用的硬件条件。

## 致谢

本项目基于 Xingkai Yu 的原始 Nano-vLLM 项目开发，并继续采用 MIT 许可证。Qwen 模型权重由 Qwen 团队单独发布，并遵循其各自的模型使用条款。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
