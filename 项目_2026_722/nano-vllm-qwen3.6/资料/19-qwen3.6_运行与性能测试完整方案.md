# nano-vLLM-qwen3.6 成功运行与性能测试完整方案

> 适用对象：已经读完原版 nano-vLLM，并准备把 `nano-vllm-qwen3.6` 跑通、测出可复现实验数据，后续继续实现动态 KV Cache 压缩和 FP8 KV Cache 的学习者。  
> 分析依据：本次提供的两个仓库压缩包、仓库 Git 历史、项目内既有讨论，以及代码中的真实运行入口。  
> 本文不把仓库 README 中的示例数字当作你的实验结果；简历中的所有数字必须由你在自己的硬件和固定实验协议下重新测量。

---

## 0. 最重要的结论

### 0.1 推荐的实际执行路线

不要一上来就在笔记本上运行 `Qwen3.6-27B-FP8`。最稳妥的路线是：

1. 在本地 5070 Ti 笔记本上先用 `Qwen3-0.6B` 验证环境、原版链路和 fork 回归正确性。
2. 再用 `Qwen3.5-2B` 验证 hybrid 架构、Gated DeltaNet 状态、eager/CUDA Graph 和并发行为。
3. 在多卡服务器上运行 `Qwen3.5-9B`，完成正式的 hybrid 性能实验。
4. 使用仓库 README 已验证过的等级——`4 × RTX 4090 24 GB、TP=4`——运行 `Qwen3.6-27B-FP8`。
5. 基础生成完全正确之后，再依次运行 MTP forward、rollback、draft/verify、强制 reject、draft-length sweep。
6. 以上结果作为后续“动态 KV Cache 压缩”和“FP8 KV Cache”改造的 baseline；两项新功能尚未包含在当前仓库中。

### 0.2 当前仓库中三种“FP8”概念必须严格区分

| 概念 | 当前仓库是否实现 | 真实含义 |
|---|---:|---|
| FP8 checkpoint 读取 | 是 | 从 FP8 权重文件读取分块量化权重和 scale |
| 原生 FP8 GEMM / FP8 推理 | 否 | 当前没有 FP8 matmul kernel |
| FP8 KV Cache | 否 | 当前 `kv_cache` 跟随模型 dtype，实际仍是 BF16 |

`nanovllm/utils/quant.py` 会把 FP8 shard 乘 scale 后转换为 BF16；`linear.py`、`embed_head.py`、`gated_delta_net.py` 再把各 rank 所属 BF16 shard 写入参数。因此 `Qwen3.6-27B-FP8` 在这个仓库里是“FP8 checkpoint → rank-local 反量化 → BF16 常驻和计算”，不能宣传为原生 FP8 推理，也不能宣传为 FP8 KV Cache。

### 0.3 当前仓库的定位

它是离线推理与研究/学习实现，不是完整的生产 serving 系统：

- 有 `LLM.generate()`、Scheduler、Continuous Batching、Paged KV Cache、TP、CUDA Graph；
- 没有 HTTP/OpenAI API Server、SSE 流式输出、多租户和完整线上压测入口；
- 因此本文测得的 TTFT、TPOT 和吞吐是“离线 Engine 指标”，不包含网络和 API Server 开销；
- 和 vLLM 对比时，应优先使用 vLLM 的离线 `LLM` 接口，不能把 nano-vLLM 离线数据与 vLLM 在线 serving 数据直接相减。

---

## 1. 仓库分析结论

### 1.1 本次附件版本

| 仓库 | 附件中的 HEAD | 日期 | 说明 |
|---|---|---|---|
| 原版 nano-vLLM | `bb823b3` | 2026-04-26 | 已含 chunked prefill 相关修复 |
| nano-vLLM-qwen3.6 | `c468d63` | 2026-05-27 | 已含 MTP chunk graph verify 原型 |

复现实验时必须保存 `git rev-parse HEAD`。附件中 qwen3.6 仓库的 Git remote 指向 `Dtone-1/nano-vllm-qwen3.6`，但 `README.md`/`pyproject.toml` 又写了 `DaveByteAI/nano-vllm-qwen3.6`。在公开 README 或简历中引用项目前，建议先统一并确认真实上游地址；本轮实验以附件 commit `c468d63` 为准。

### 1.2 主推理调用链

```mermaid
flowchart TD
    A["LLM.generate / add_request"] --> B["Scheduler.schedule"]
    B --> C["BlockManager + StateSlotManager"]
    C --> D["ModelRunner.prepare_prefill / prepare_decode"]
    D --> E["Qwen3 dense 或 Qwen3.5 hybrid forward"]
    E --> F["Full Attention: KV Cache"]
    E --> G["Gated DeltaNet: conv + recurrent state"]
    F --> H["LM Head + TP token selection"]
    G --> H
    H --> I["Scheduler.postprocess"]
```

### 1.3 关键代码与实验的对应关系

| 文件 | 真实作用 | 必须验证的实验 |
|---|---|---|
| `config.py` | 读取 full/text/vision config，识别 hybrid，设置模型长度和开关 | config 识别、模型类型、最大长度 |
| `llm_engine.py` | 请求入口、tokenizer、多进程 TP、generate/step | TTFT、E2E、并发 |
| `scheduler.py` | prefill/decode 调度、state slot 分配、preemption | 并发、抢占、state slot 正确性 |
| `block_manager.py` | KV block 分配、释放、prefix cache | block 使用率、cache hit、preemption |
| `model_runner.py` | 模型创建/加载、KV/GDN 状态分配、输入准备、CUDA Graph、MTP verify/rollback | 几乎全部核心实验 |
| `qwen3_5.py` | full attention 与 GDN 层按 `layer_types` 交替执行 | hybrid 输出正确性 |
| `gated_delta_net.py` | prefill chunk rule、decode recurrent rule、两类 state 更新 | chunked prefill、batch decode、state 隔离 |
| `loader.py`、`quant.py` | safetensors 映射、rank-local FP8 反量化 | 加载时间、权重覆盖、显存 |
| `qwen3_mtp.py` | MTP head/layer forward | MTP 权重加载、draft token |
| `run_mtp_fast_decode.py` | draft/verify/accept/reject/rollback 快速路径 | accept rate、greedy match、吞吐 |
| `vision_encoder.py`、`image_processing.py` | 图像预处理、vision encoder、MRoPE | 可选多模态 smoke test |

### 1.4 从源码直接得到的限制

1. `kvcache_block_size` 必须满足 `% 256 == 0`，默认是 256；这个仓库不能直接做 vLLM 常见的 16/32-token block 对比。
2. hybrid 模型会在 `Scheduler` 中调用 `allocate(..., disable_prefix_cache=True)`，所以 Qwen3.5/Qwen3.6 当前主动禁用了 prefix caching。prefix cache 实验只能先在 Qwen3 dense 上做。
3. 只有 full-attention 层会分配 KV Cache；GDN 层使用单独的 conv state 和 FP32 recurrent state。
4. `_create_model()` 只区分 Qwen3 和 `qwen3_5`，Qwen3.6-27B 复用 Qwen3.5 hybrid 实现。
5. `Qwen3_5MLP` 是 dense MLP，没有 MoE expert/router 实现。不要直接拿 Qwen3.5/3.6 的 A3B MoE checkpoint 运行；即使加载过程没有立刻报错，也可能有大量 expert 权重被跳过。
6. `load_model()` 对普通权重不会强制要求 `skipped_names == 0`；如果映射错误，部分参数可能保持未初始化。因此“程序能输出文本”不等于权重完整加载。
7. TP 使用 NCCL、固定 TCP 端口 `2333`、固定共享内存名 `nanovllm`，更适合 Linux/WSL2；普通 Windows Python 环境无法提供 NCCL。
8. 现有 `bench.py` 只统计总 completion throughput；`bench_qwen35_fixed.py` 能区分 prefill/decode，但仍没有 TTFT、TPOT 分位数、QPS、峰值显存和并发曲线。
9. `run_mtp_fast_decode.py` 的 `decode_tok_s` 使用 `model_call_seconds`，排除了部分 Python/Scheduler wall time，可能高估用户实际可见吞吐；正式报告要同时输出 `wall_seconds`。
10. 本次已对附件 qwen3.6 仓库执行 `compileall`，所有 Python 文件通过语法检查；这不替代 GPU、模型权重和算子运行检查。

---

## 2. 需要什么条件和环境

## 2.1 操作系统

### 你的笔记本

你当前是 Windows + VMware Ubuntu 的使用方式。普通 VMware Ubuntu 通常看不到宿主机 NVIDIA CUDA GPU，而此仓库又依赖 NCCL、FlashAttention 和 CUDA Graph。因此建议：

- 首选：Windows 11 + WSL2 Ubuntu；
- 次选：原生 Ubuntu 双系统；
- 不建议：普通 VMware Ubuntu 中直接跑 GPU benchmark；
- Windows 原生 Python 也不适合作为主环境，因为代码硬编码了 NCCL backend。

NVIDIA 提供了 WSL2 CUDA 支持说明：[CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。

### 服务器

- Ubuntu 22.04/24.04；
- 同一节点内 NVIDIA GPU；
- TP 多卡最好是同型号 GPU；
- 记录 GPU 拓扑：`nvidia-smi topo -m`；
- 不要让其他任务共享 GPU，否则延迟和显存数据不可复现。

## 2.2 软件条件

| 项目 | 建议 |
|---|---|
| Python | 3.11，仓库要求 `>=3.10,<3.13` |
| PyTorch | 安装与显卡驱动兼容的 CUDA build；先装 PyTorch，再装 flash-attn |
| CUDA Toolkit | FlashAttention 需要编译时必须有 `nvcc`；官方当前说明要求 CUDA 12.0+ |
| Transformers | Qwen3.5 官方说明要求最新版本；仓库的 `transformers>=4.51.0` 下限并不足以保证识别 `qwen3_5` |
| FlashAttention | 安装后必须真实调用 kernel 验证，不能只验证 `import` |
| Triton | 与 PyTorch CUDA build 匹配 |
| 额外依赖 | `huggingface_hub`、`safetensors`、`xxhash`；多模态还需 `torchvision`、`Pillow` |

官方资料：

- [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- [FlashAttention 官方安装说明](https://github.com/Dao-AILab/flash-attention#installation-and-features)
- [Qwen3.5-9B 模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Hugging Face `hf download` 文档](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-download)

> 注意：FlashAttention 官方 README 当前明确列出的 FA2 NVIDIA 架构是 Ampere、Ada、Hopper，没有明确列出 Blackwell。你的 5070 Ti 属于较新的架构，因此必须把 FlashAttention build/kernel 验证放在最前面；本地失败时优先使用已验证的 4090/A100/H100 服务器环境，不要先怀疑 nano-vLLM 业务代码。

## 2.3 硬件分级

以下是“能否完成本项目实验”的工程建议，不是模型厂商的最低承诺。

| 等级 | 模型与任务 | GPU 建议 | 系统内存/磁盘建议 |
|---|---|---|---|
| 本地环境验证 | Qwen3-0.6B BF16 | 单卡 8 GB+ | RAM 16 GB+，空闲磁盘 10 GB+ |
| 本地 hybrid 验证 | Qwen3.5-2B，text-only | 你的 5070 Ti；先以实测 VRAM 为准 | RAM 32 GB+，空闲磁盘 15 GB+ |
| Qwen3.5 正式实验 | Qwen3.5-9B BF16 | 单卡 24 GB text-only 可能较紧；`2×24 GB` 或 `4×24 GB` 更稳 | RAM 64 GB+，空闲磁盘 30 GB+ |
| Qwen3.6 正式实验 | Qwen3.6-27B-FP8，仓库内部 BF16 常驻 | README 已验证 `4×RTX 4090 24 GB、TP=4` | RAM 64 GB 最低，建议 128 GB；空闲磁盘 60 GB+ |
| MTP 完整实验 | Qwen3.6-27B-FP8 + MTP | 与上一行相同，并预留 graph/snapshot 显存 | 同上 |

Qwen 官方模型卡显示：Qwen3.5-2B 为 2B 参数，Qwen3.5-9B 页面显示约 10B 参数，Qwen3.6-27B-FP8 页面显示约 28B 参数。当前仓库会把 Qwen3.6 FP8 shard 反量化为 BF16，所以不能按“28 GB FP8 权重能塞进单卡”来估算运行显存。

---

## 3. 从零搭建环境的详细步骤

以下命令默认在 WSL2/原生 Ubuntu 中执行。

## 3.1 第一步：确认 GPU 没有被 VMware 隔离

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
```

如果 `nvidia-smi` 不存在或看不到 5070 Ti，先解决 WSL2/驱动，不要继续安装项目。

## 3.2 第二步：创建独立环境

```bash
cd /path/to/nano-vllm-qwen3.6

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip setuptools wheel packaging ninja
```

原版与 qwen3.6 fork 的 Python 包名都叫 `nano-vllm`。做 A/B 对比时最好建立两个 venv，避免 editable install 相互覆盖。

## 3.3 第三步：先安装 CUDA 版 PyTorch

先去 PyTorch 官方 selector 选择 Linux、Pip、Python 和与你驱动兼容的 CUDA 版本。以下仅是 CUDA 12.8 wheel 的示例；如果官方 selector 已推荐更新版本，以官方命令为准。

```bash
python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
```

验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("name:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    print("BF16 matmul:", y.shape, torch.isfinite(y).all().item())
PY
```

只有 `available=True`、设备名正确、BF16 matmul 成功，才继续。

## 3.4 第四步：安装 Qwen3.5 所需 Transformers 和基础依赖

Qwen3.5 官方模型卡当前要求最新 Transformers。为了减少“model type `qwen3_5` 未识别”的问题，可以先使用官方建议的 main 版本；正式复现实验后再把成功版本冻结到 requirements/lock 文件。

```bash
python -m pip install -U \
  "transformers @ git+https://github.com/huggingface/transformers.git@main" \
  huggingface_hub safetensors xxhash pillow
```

## 3.5 第五步：安装并验证 FlashAttention

```bash
nvcc --version
ninja --version

MAX_JOBS=4 python -m pip install flash-attn --no-build-isolation
```

验证导入和关键 API：

```bash
python - <<'PY'
import flash_attn
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
print("flash_attn:", flash_attn.__version__)
print("varlen:", flash_attn_varlen_func)
print("kvcache:", flash_attn_with_kvcache)
PY
```

然后至少做一次小 tensor kernel smoke test；若在 5070 Ti 上出现 `no kernel image`、unsupported architecture 或 ABI symbol error：

1. 确认 PyTorch、CUDA Toolkit、flash-attn 是在同一个 venv 中安装；
2. 确认 flash-attn 是在安装完当前 PyTorch 后重新编译；
3. 尝试最新 flash-attn release/source；
4. 仍失败则转到 4090/A100/H100 环境完成主实验。

## 3.6 第六步：安装本仓库

```bash
python -m pip install -e .

python -m pip check
python -m compileall -q \
  nanovllm examples \
  run_text_qwen35_v2.py run_text_qwen36_fp8.py \
  run_mtp_fast_decode.py bench.py bench_qwen35_fixed.py \
  bench_mtp_draft_sweep.py test_mtp_forward.py \
  test_mtp1_verify.py test_mtp1_spec_decode.py \
  test_mtp_spec_decode.py test_state_rollback.py
```

记录最终环境：

```bash
mkdir -p results/env results/logs results/raw
python -m pip freeze > results/env/pip-freeze.txt
nvidia-smi -q > results/env/nvidia-smi-q.txt
nvidia-smi topo -m > results/env/gpu-topology.txt
git rev-parse HEAD > results/env/commit.txt
git status --short > results/env/git-status.txt
```

## 3.7 第七步：下载模型

先用 `--dry-run` 查看大小，再下载；模型放在仓库外。

```bash
hf download Qwen/Qwen3-0.6B --dry-run
hf download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B \
  --max-workers 8

hf download Qwen/Qwen3.5-2B --dry-run
hf download Qwen/Qwen3.5-2B \
  --local-dir ~/huggingface/Qwen3.5-2B \
  --max-workers 8
```

多卡服务器再下载：

```bash
hf download Qwen/Qwen3.5-9B \
  --local-dir ~/huggingface/Qwen3.5-9B \
  --max-workers 8

hf download Qwen/Qwen3.6-27B-FP8 \
  --local-dir ~/huggingface/Qwen3.6-27B-FP8 \
  --max-workers 8
```

验证文件完整性与 config 识别：

```bash
hf cache verify Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B

python - <<'PY'
import os
from transformers import AutoConfig

paths = [
    "~/huggingface/Qwen3-0.6B",
    "~/huggingface/Qwen3.5-2B",
]
for p in paths:
    p = os.path.expanduser(p)
    c = AutoConfig.from_pretrained(p)
    tc = getattr(c, "text_config", c)
    print(p)
    print("  full:", type(c).__name__)
    print("  model_type:", tc.model_type)
    print("  dtype:", getattr(tc, "dtype", None))
    print("  layers:", getattr(tc, "num_hidden_layers", None))
    print("  has layer_types:", hasattr(tc, "layer_types"))
PY
```

若 Qwen3.5 报 `model type qwen3_5 not recognized`，优先升级 Transformers；这不是 nano-vLLM 的模型 forward 报错。

---

## 4. 成功跑通项目的顺序

## 4.1 阶段 A：Qwen3-0.6B 环境烟雾测试

仓库脚本默认路径正是 `~/huggingface/Qwen3-0.6B`：

```bash
CUDA_VISIBLE_DEVICES=0 python examples/qwen3.py
```

检查：

- 能创建 NCCL process group；
- 权重加载完成；
- 两个 prompt 都得到非空输出；
- 没有 NaN、乱码和无限循环；
- `nvidia-smi` 能看到显存占用。

随后运行仓库原始吞吐 benchmark：

```bash
CUDA_VISIBLE_DEVICES=0 python bench.py 2>&1 \
  | tee results/logs/qwen3_06b_bench.log
```

这个数字只能当环境 smoke benchmark，不能作为最终性能报告，因为脚本：

- 固定 256 个请求；
- 输入/输出长度随机；
- 只给 aggregate output tok/s；
- 没有 TTFT、TPOT、P95 和显存峰值。

## 4.2 阶段 B：Qwen3.5-2B 本地 hybrid 烟雾测试

`run_text_qwen35_v2.py` 虽然名字是 text-only，但当前 `LLM(...)` 没有传 `enable_vision=False`，会按默认值创建 vision encoder。做文本实验前，应在该脚本的 `LLM(` 参数中加入：

```python
enable_vision=False,
```

先以 eager、短上下文、短输出运行：

```bash
CUDA_VISIBLE_DEVICES=0 python run_text_qwen35_v2.py \
  --model ~/huggingface/Qwen3.5-2B \
  --devices 0 \
  --tp 1 \
  --eager \
  --max-model-len 512 \
  --max-batched-tokens 128 \
  --max-tokens 32 \
  --gpu-memory-utilization 0.70
```

成功后再打开 CUDA Graph：

```bash
CUDA_VISIBLE_DEVICES=0 python run_text_qwen35_v2.py \
  --model ~/huggingface/Qwen3.5-2B \
  --devices 0 \
  --tp 1 \
  --max-model-len 1024 \
  --max-batched-tokens 256 \
  --max-tokens 128 \
  --gpu-memory-utilization 0.75
```

必须分别记录 eager 与 graph 的 greedy token IDs。性能测试用 `temperature=0.0` 保证可比；自然语言质量演示再用模型推荐的采样参数。

## 4.3 阶段 C：Qwen3.5-9B 多卡正式运行

仍然先确保 text-only 脚本传入 `enable_vision=False`。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_text_qwen35_v2.py \
  --model ~/huggingface/Qwen3.5-9B \
  --devices 0,1,2,3 \
  --tp 4 \
  --eager \
  --max-model-len 1024 \
  --max-batched-tokens 128 \
  --max-tokens 32 \
  --gpu-memory-utilization 0.75
```

eager 成功后删除 `--eager`，逐步把 `max-tokens` 提到 128/256，再运行现成 helper：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python bench_qwen35_fixed.py \
  --model ~/huggingface/Qwen3.5-9B \
  --tp 4 \
  --max-tokens 256 \
  --repeats 5 \
  --temperature 0.0 \
  2>&1 | tee results/logs/qwen35_9b_tp4_graph.log
```

注意：`bench_qwen35_fixed.py` 本身也没有传 `enable_vision=False`，正式 text-only 数据前应做相同修正。

## 4.4 阶段 D：Qwen3.6-27B-FP8 多卡运行

首先确认四张卡都空闲：

```bash
nvidia-smi
nvidia-smi topo -m
ss -ltnp | grep ':2333' || true
```

第一遍必须 eager、小输出：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_text_qwen36_fp8.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --tp 4 \
  --eager \
  --max-model-len 1024 \
  --max-batched-tokens 128 \
  --max-tokens 16 \
  --temperature 0.0 \
  --gpu-memory-utilization 0.75 \
  2>&1 | tee results/logs/qwen36_eager_smoke.log
```

成功标准：

- 四个 rank 都成功进入 NCCL；
- loader 遍历所有 safetensors；
- 文本模型权重没有异常 skip；
- 生成 token 合理且没有 NaN；
- 退出时四个进程都结束，没有残留共享内存。

随后打开 CUDA Graph：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_text_qwen36_fp8.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --tp 4 \
  --max-model-len 2048 \
  --max-batched-tokens 128 \
  --max-tokens 128 \
  --temperature 0.0 \
  --gpu-memory-utilization 0.80 \
  2>&1 | tee results/logs/qwen36_graph_smoke.log
```

若 graph capture OOM：先回到 `--eager` 确认逻辑；再降低 `gpu-memory-utilization`、`max-model-len`、`max-batched-tokens` 和 `max_num_seqs`。这个仓库先分配 KV Cache/GDN state，再 capture graph，所以不能把所有空闲显存都交给 cache。

## 4.5 阶段 E：MTP 必须按以下顺序运行

### 1. 权重与单步 forward

```bash
python test_mtp_forward.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 --tp 4 --top-k 5
```

要求：`mtp_loaded_count > 0`、`mtp_skipped_count == 0`，hidden/logits shape 正常。

### 2. 状态回滚

```bash
python test_state_rollback.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 --tp 4 \
  --tolerance 0.0
```

要求：`rollback_ok: True`、重复运行 token 相同、`max_logit_diff == 0`。

### 3. MTP-1 draft/verify

```bash
python test_mtp1_verify.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 --tp 4 \
  --max-tokens 64
```

### 4. 强制 reject，验证回滚分支

```bash
python test_mtp1_spec_decode.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 --tp 4 \
  --max-tokens 64 \
  --force-reject-attempt 1
```

### 5. fast path 与 greedy 对齐

```bash
python run_mtp_fast_decode.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 --tp 4 \
  --draft-len 2 \
  --verify-mode graph \
  --max-tokens 128 \
  --compare-greedy
```

要求：`greedy_match: True`。如果不一致，先停止性能结论，排查状态提交/回滚和 TP token 选择。

### 6. draft length sweep

```bash
python bench_mtp_draft_sweep.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 --tp 4 \
  --draft-lens 1,2,3,4 \
  --verify-mode graph \
  --max-tokens 128 \
  2>&1 | tee results/logs/mtp_draft_sweep.log
```

当前 MTP 是研究原型，README 已明确“尚未提供 decode speedup”。即使某次 `tok_s` 较高，也必须结合 wall time、accept rate、target forwards/token 和 reject rerun 共同解释。

---

## 5. 应该做哪些实验、测什么性能

## 5.1 总体实验地图

| ID | 实验 | 模型 | 核心变量 | 主要指标 | 目的 |
|---|---|---|---|---|---|
| E0 | 环境/算子验证 | 小 tensor | torch/FA/CUDA | 是否成功 | 排除环境问题 |
| E1 | 文本正确性 | Qwen3-0.6B、Qwen3.5-2B | prompt | token IDs、非空输出 | 基础正确 |
| E2 | fork 回归 | Qwen3-0.6B | 原版 vs fork | greedy match、性能差 | 证明新增功能没破坏 dense |
| E3 | hybrid state 正确性 | Qwen3.5 | eager/graph、chunked/unchunked、batch | token match、logit 差 | 验证 GDN state |
| E4 | 长度扩展 | Qwen3.5/3.6 | input/output length | prefill tok/s、TPOT、显存 | 找瓶颈 |
| E5 | 并发扩展 | Qwen3.5/3.6 | concurrency | output tok/s、QPS、P50/P95 | 验证 batching |
| E6 | CUDA Graph | Qwen3.5/3.6 | eager vs graph | TPOT、decode tok/s | 测 launch 优化 |
| E7 | TP scaling | Qwen3.5/3.6 | TP=1/2/4 | 吞吐、显存/卡、效率 | 测多卡收益 |
| E8 | Prefix cache | Qwen3 dense | 共享前缀比例 | TTFT、prefill tok/s、hit rate | 验证 KV 复用 |
| E9 | 冷启动/FP8 loader | Qwen3.6 FP8 | cold/warm OS cache | load time、CPU/GPU memory | 量化加载代价 |
| E10 | MTP | Qwen3.6 | draft=1–4、verify mode | accept rate、greedy match、wall tok/s | 判断原型价值 |
| E11 | 多模态 smoke | Qwen3.5 | 图像分辨率 | TTFT、图像 token、显存 | 可选功能验证 |
| E12 | vLLM 对照 | 同一模型/输入 | engine | 同口径指标 | 建立行业 baseline |

## 5.2 必测性能指标和计算方法

### 请求级

- **TTFT**：从请求加入 Engine 到生成第一个 completion token。nano-vLLM 的第一个 token 在 prefill `step()` 结束时产生。
- **E2E latency**：从请求加入到 EOS 或 `max_tokens` 完成。
- **TPOT**：

$$
TPOT = \frac{E2E-TTFT}{N_{output}-1}
$$

- **P50/P95 latency**：不能只报告平均值；并发实验至少运行 100 个请求。

### 阶段级

$$
Prefill\ Throughput = \frac{\sum input\ tokens}{\sum prefill\ time}
$$

$$
Decode\ Throughput = \frac{\sum decode\ tokens}{\sum decode\ time}
$$

注意第一个生成 token 属于 prefill step 的产出，不能在 prefill 和 decode 中重复计数。

### 系统级

- completion tokens/s；
- total tokens/s，必须和 completion tokens/s 分开写；
- QPS = 完成请求数 / 总 wall time；
- 每张 GPU 的 peak memory、平均/峰值 utilization、功耗；
- KV Cache 实际 blocks、free/used blocks、利用率；
- GDN state slot 数、每 slot 字节数；
- preemption 次数和 re-prefill token 数；
- model load time、CUDA Graph capture time、首个请求前初始化时间。

### TP 扩展效率

$$
Scaling\ Efficiency(N) = \frac{Throughput_{TP=N}}{N\times Throughput_{TP=1}}
$$

如果 TP=1 因显存无法运行，就只报告 TP2→TP4 的 speedup，不要伪造 TP1 基线。

## 5.3 KV Cache 和 GDN state 的显存口径

当前每个 rank 的 KV Cache 理论值为：

$$
M_{KV}=2\times L_{attn}\times N_{blocks}\times B_{size}\times
\frac{H_{kv}}{TP}\times D_{head}\times bytes(dtype)
$$

其中 `L_attn` 只统计 full-attention 层，不是总层数。

每个请求的 GDN state 近似为：

$$
M_{state/seq}=L_{gdn}\times
[conv\_dim\times(K-1)\times bytes(BF16)+H_vD_kD_v\times4]
$$

第二项固定用 FP32。正式报告必须把 `KV Cache memory` 和 `GDN state memory` 分开，不要都叫 KV Cache。

## 5.4 本地最小实验矩阵

用于你的 5070 Ti 笔记本：

| 模型 | input tokens | output tokens | concurrency | mode |
|---|---|---|---|---|
| Qwen3-0.6B | 128/512/1024 | 64/128 | 1/4/8 | eager/graph |
| Qwen3.5-2B text-only | 128/512/1024 | 64/128 | 1/2/4 | eager/graph |

每个配置：

1. 1 次不计入统计的 warmup；
2. 至少 5 次独立重复，正式延迟分位数实验用 100+ 请求；
3. 报告 median，同时报告 min/max 或标准差；
4. 同一对比中固定 prompt token IDs、output length、temperature、GPU power mode；
5. 每个配置用独立进程运行，避免上一次 prefix cache、state 或 CUDA allocator 污染。

## 5.5 多卡正式实验矩阵

| 维度 | 建议取值 |
|---|---|
| 模型 | Qwen3.5-9B；Qwen3.6-27B-FP8 |
| input length | 128、512、2048、4096 |
| output length | 64、128、256 |
| concurrency | 1、2、4、8、16，直到 OOM/频繁 preemption |
| TP | Qwen3.5 测 1/2/4（能装下才测）；Qwen3.6 以 TP4 为主 |
| execution | eager、CUDA Graph |
| temperature | 性能/正确性对比用 0.0；质量演示另测推荐采样 |
| EOS | 性能定长实验用 `ignore_eos=True`；真实请求实验用正常 EOS |

不要第一轮就测 262K 上下文。当前项目目标是先证明 hybrid state、调度和 cache 正确；动态 KV Cache 压缩完成后，再扩展到 8K/16K/32K 作为项目主实验。

---

## 6. 正确性测试必须先于性能测试

## 6.1 Dense 回归：原版 vs qwen3.6 fork

两个仓库分别创建 venv，在同一张 GPU、同一 Qwen3-0.6B checkpoint 上：

- `temperature=0.0`；
- 同一 chat template；
- 同一 input/output length；
- 同一 eager/graph 模式；
- 比较前 32/128 个 completion token IDs；
- 再比较 TTFT、TPOT、吞吐和显存。

如果 token 不一致，先比较首 token 和最长公共前缀；不要在 correctness 未解释前发布性能结论。

## 6.2 Hybrid 四组等价性

对 Qwen3.5-2B/9B 使用 20 条固定 prompt，逐项比较 greedy token IDs：

1. eager vs CUDA Graph；
2. 一次性 prefill vs chunked prefill；
3. concurrency=1 顺序运行 vs concurrency>1 同批运行；
4. TP1 vs TP2/TP4。

这里主要验证：

- `state_slot_id` 没有串请求；
- chunked continuation 正确更新 conv/recurrent state；
- CUDA Graph replay 使用稳定的 state buffer 地址；
- TP shard 和全局 token 选择正确。

## 6.3 Preemption 正确性

通过较低 cache 容量和多个长请求触发 preemption，比较：

- 无 preemption 单独运行的 token IDs；
- 有 preemption 后重新 prefill 的 token IDs；
- block 是否全部释放；
- state slot 是否释放、重新分配且没有沿用旧状态。

## 6.4 权重覆盖检查

建议让 rank0 在加载结束后输出：

- `len(loaded_names)`；
- `len(skipped_names)`；
- skipped 前 50 个名称；
- 按前缀分类的 skip 数量。

text-only + `enable_vision=False` 时，visual 权重被 skip 是预期行为；`enable_mtp=False` 时 MTP 权重 skip 也可能是预期行为。任何语言模型主干的 attention、GDN、MLP、embedding、norm、lm_head 权重被 skip 都应视为失败。

---

## 7. 性能采集方法

## 7.1 时间测量

使用 `time.perf_counter()`；GPU 区间前后必须同步：

```python
torch.cuda.synchronize()
start = perf_counter()
outputs, num_tokens = llm.step()
torch.cuda.synchronize()
elapsed = perf_counter() - start
```

需要在 benchmark 中保存：

- request enqueue timestamp；
- 第一次 prefill step 完成 timestamp；
- 每次 decode step 完成 timestamp；
- request finish timestamp；
- 本 step 的 scheduled tokens、active sequences、is_prefill。

现成 `bench_qwen35_fixed.py` 可以作为起点，但应补充 JSONL/CSV 输出和 TTFT/TPOT 分位数。

## 7.2 GPU 监控

benchmark 启动前在另一个终端运行：

```bash
nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm \
  --format=csv \
  -lms 200 > results/raw/gpu-monitor.csv
```

结束后停止监控。TP 场景必须报告所有 rank 对应 GPU，而不是只报告 rank0 的 `torch.cuda.max_memory_allocated()`。

## 7.3 冷启动与热启动

- **冷启动**：新进程、模型文件不在 OS page cache 或机器重启后的首次加载；
- **热启动**：新进程，但文件大概率仍在 page cache；
- 分别统计 checkpoint 读取、反量化、模型 ready、graph capture 时间；
- 不建议为“制造冷启动”随意执行系统 drop-cache 命令；如果无法可靠清 cache，就如实标成 warm-cache load time。

## 7.4 结果文件建议

```text
results/
  env/
    commit.txt
    pip-freeze.txt
    nvidia-smi-q.txt
    gpu-topology.txt
  configs/
    exp-e06-qwen35-tp4-graph.json
  raw/
    requests.jsonl
    steps.csv
    gpu-monitor.csv
  summary/
    summary.csv
  logs/
    *.log
```

`summary.csv` 至少包含：

```text
commit,model,model_revision,gpu,tp,eager,input_len,output_len,concurrency,
max_model_len,max_batched_tokens,gpu_memory_utilization,repeat,
load_s,graph_capture_s,ttft_ms,tpot_ms,e2e_ms,prefill_tok_s,
decode_tok_s,output_tok_s,qps,peak_gpu_mem_mb,preemptions,kv_blocks_used
```

---

## 8. 各专题实验应怎样解释

## 8.1 Eager vs CUDA Graph

预期：

- prefill 仍走 eager，主要收益应出现在 decode；
- batch 小、decode 长时更容易看到 TPOT 改善；
- graph 首次 capture 会增加启动时间和显存；
- 必须同时报告初始化时间与 steady-state decode 性能。

成功标准不是“graph 一定更快”，而是：token IDs 对齐，且 steady-state TPOT 的改善超过重复实验噪声。

## 8.2 TP 扩展

分别记录：

- 模型是否能装下；
- 每卡权重显存、KV/state 显存；
- prefill/decode tok/s；
- NCCL 通信开销；
- TP2→TP4 speedup；
- scaling efficiency。

小模型在多卡上可能更慢，因为 collective/进程开销大于矩阵切分收益。这不是测试失败。

## 8.3 Prefix cache

只在 Qwen3 dense 测：构造 0%、50%、90% 共享前缀请求，记录 cache hit blocks、TTFT 和 prefill tokens。Qwen3.5/3.6 hybrid 当前禁用 prefix cache，因为仅复用 KV block 不能恢复 GDN recurrent/conv state。

## 8.4 MTP

除吞吐外必须报告：

- `greedy_match`；
- `accept_rate`；
- 平均 accepted length；
- `target_forwards_per_token`；
- `mtp_forwards_per_token`；
- `reject_reruns`；
- `verify_graph_replays` / eager calls；
- `model_call_seconds` 与 `wall_seconds` 两种 tok/s。

如果 accept rate 高但 wall throughput 没提升，说明 draft、verify、状态 snapshot/restore 和 Python 控制开销抵消了收益。

## 8.5 Qwen3.6 FP8 checkpoint

当前只应该声称并测试：

- FP8 checkpoint 是否能正确读取；
- rank-local slice 是否降低单 rank 临时反量化范围；
- 加载时间和 host/GPU 峰值内存；
- 反量化后 greedy 输出是否与可信 baseline 对齐；
- 常驻权重 dtype 是否为 BF16。

不应该用当前结果证明“FP8 比 BF16 matmul 更快”或“FP8 KV Cache 节省 50% cache 显存”。

## 8.6 多模态

多模态只作为可选 smoke test：

- 安装 `torchvision`、`Pillow`；
- `enable_vision=True`；
- 将 `examples/qwen3_5.py` 的 `max_tokens=4096` 先减到 64；
- 使用一张固定图片；
- 记录图像预处理时间、vision encoder 时间、图像 token 数、TTFT 和显存。

当前自定义 image preprocessing/vision 路径更适合验证链路，不适合直接宣称达到官方 Qwen3.5 多模态精度。

---

## 9. 为后续动态 KV Cache 压缩和 FP8 KV Cache 保留的 baseline

历史讨论已经确定：以 `nano-vllm-qwen3.6` 为主干，先跑通 hybrid，再移植动态压缩，最后做 FP8 KV Cache。因此本次 baseline 必须提前记录以下统一指标：

| 指标 | Baseline | 动态压缩后 | FP8 KV Cache 后 |
|---|---:|---:|---:|
| KV Cache 峰值显存 |  |  |  |
| GDN state 峰值显存 |  |  |  |
| 固定显存最大 resident sequences |  |  |  |
| preemption 次数 |  |  |  |
| re-prefill tokens |  |  |  |
| decode throughput |  |  |  |
| TPOT P50/P95 |  |  |  |
| E2E P50/P95 |  |  |  |
| greedy match / 质量退化 |  |  |  |

动态压缩必须只作用于 full-attention KV Cache，不能压缩或错误复用 GDN recurrent/conv state。后续比较至少设置：

- 无压缩 baseline；
- 不同 Top-K/保留比例；
- 不同最近窗口大小；
- 不同压缩周期；
- BF16 KV vs FP8 KV；
- 8K/16K/32K 上下文；
- 相同显存预算下的最大并发。

只有性能提升、状态正确和质量退化三者同时可解释，才能写入简历。

---

## 10. 常见错误与排查顺序

| 现象 | 优先检查 |
|---|---|
| `torch.cuda.is_available=False` | 是否在 VMware；WSL 驱动；PyTorch 是否 CPU wheel |
| `qwen3_5 not recognized` | Transformers 太旧 |
| `flash_attn` undefined symbol | torch/flash-attn ABI 不匹配，重装/重编译 |
| `no kernel image` | 5070 Ti/Blackwell kernel 架构支持 |
| NCCL 初始化失败 | `CUDA_VISIBLE_DEVICES` 数量、TP 值、驱动、端口 2333 |
| `Address already in use` | `ss -ltnp | grep 2333`，清理残留进程 |
| shared memory already exists | 确认无 nano-vLLM 进程后检查 `/dev/shm/nanovllm` |
| 创建 KV Cache 时 assert/OOM | 降低 utilization、模型长度、batch token budget |
| CUDA Graph capture OOM | 先 eager；降低 cache/state 预算与 max_num_seqs |
| 输出乱码/NaN | 权重 skip、FP8 scale mapping、dtype、TP shard |
| eager 正常、graph 错 | graph 输入 buffer/state_indices/state 地址 |
| 单请求正常、并发错 | state slot 串扰、block table、batch sampling |
| MTP reject 后输出错 | Scheduler、KV、GDN state 未同时 rollback |
| 程序退出卡住 | 某 rank 异常、NCCL barrier、子进程未结束 |

排错原则：

```text
环境 → 小模型 → eager → 单卡 → 短输入/短输出
     → CUDA Graph → 并发 → TP → 27B FP8 → MTP
```

一次只改变一个变量。

---

## 11. 最终应产出的实验报告结构

1. 项目与 commit、本人修改范围；
2. 硬件、OS、驱动、CUDA、PyTorch、FlashAttention、Transformers；
3. 模型 repo/revision、checkpoint dtype、文本/视觉开关；
4. correctness：dense regression、eager/graph、chunked、batch、TP、rollback；
5. 单请求：TTFT、TPOT、prefill/decode throughput；
6. 并发：QPS、output tok/s、P50/P95、显存、preemption；
7. TP scaling；
8. FP8 checkpoint 加载时间/内存；
9. MTP accept/reject 与真实 wall-time 收益；
10. 局限：offline-only、hybrid prefix cache disabled、native FP8/FP8 KV 未实现；
11. 下一阶段：动态 KV 压缩和 FP8 KV Cache。

---

## 12. 可直接执行的最短清单

### 本地一周内完成

- [ ] WSL2 中 `nvidia-smi`、BF16 matmul、FlashAttention kernel 通过
- [ ] qwen3.6 fork `compileall` 和 `pip check` 通过
- [ ] Qwen3-0.6B eager/graph 跑通
- [ ] 原版 vs fork dense greedy token 对齐
- [ ] Qwen3.5-2B text-only eager 跑通
- [ ] Qwen3.5-2B CUDA Graph 跑通
- [ ] 做 input length、output length、concurrency 小矩阵
- [ ] 得到 TTFT、TPOT、throughput、P95、显存 baseline

### 多卡服务器完成

- [ ] Qwen3.5-9B TP2/TP4 正确性与性能
- [ ] Qwen3.6-27B-FP8 TP4 eager 跑通
- [ ] Qwen3.6 CUDA Graph 跑通并与 eager 对齐
- [ ] 记录 FP8 checkpoint load time 与 BF16 常驻显存
- [ ] MTP forward、rollback、forced reject 全部通过
- [ ] draft 1–4 sweep，`greedy_match=True`
- [ ] 所有结果保存环境、commit、原始日志和 CSV

完成以上清单后，你才拥有一个可信的 nano-vLLM-qwen3.6 baseline。之后再实现动态 KV Cache 压缩和 FP8 KV Cache，才能把“支持新模型”升级为有统一指标、有对照组、有工程增量的 AI Infra 简历项目。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
