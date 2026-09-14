# nano-vLLM-qwen3.6 本地环境配置指南  
## RTX 5070 Ti Laptop + Qwen3.5-2B BF16 + 两项冒烟实验

> 本指南是在你已经成功配置并运行原版 nano-vLLM 环境的基础上编写的。  
> 本轮目标不是完整复现仓库中的 9B、27B、MTP 或多卡实验，而是在本地 5070 Ti 笔记本上，用 `Qwen3.5-2B` 验证：
>
> 1. **Qwen3.5-2B BF16 文本推理，单卡运行；**
> 2. **Qwen3.5-2B 图像 + 文本多模态冒烟测试。**

---

# 0. 本轮实验边界

## 0.1 要完成的内容

| 实验 | 原仓库目标 | 本地调整后的目标 |
|---|---|---|
| 文本推理 | Qwen3.5-9B BF16，单卡/多卡 TP | **Qwen3.5-2B BF16，单卡，`TP=1`** |
| 多模态冒烟测试 | Qwen3.5-9B，图像 + 文本 | **Qwen3.5-2B，图像 + 文本** |

## 0.2 本轮明确不做

- 不下载和运行 `Qwen3.5-9B`；
- 不下载和运行 `Qwen3.6-27B-FP8`；
- 不运行 MTP、投机解码和 rollback 实验；
- 不实现 FP8 GEMM 或 FP8 KV Cache；
- 不做多卡张量并行；
- 不做正式 TTFT、TPOT、吞吐量性能测试；
- 不测试超长上下文；
- 不把这次冒烟测试结果直接写成性能提升指标。

你的笔记本只有一张 5070 Ti，因此本地只能验证单卡路径：

```text
tensor_parallel_size = 1
CUDA_VISIBLE_DEVICES = 0
```

这不等于完成了多卡 TP 验证。多卡实验应留到租用服务器后再做。

---

# 1. 哪些环境可以复用，哪些必须重新配置

## 1.1 总结表

| 环境层级 | 是否重新安装 | 本轮操作 |
|---|---:|---|
| Windows NVIDIA 驱动 | 否 | 继续复用已能被 WSL2 识别的驱动 |
| WSL2 与 Ubuntu 22.04 | 否 | 继续使用原 nano-vLLM 所在的 WSL2 Ubuntu |
| GCC 11 | 否 | 已安装且原项目可用时直接复用 |
| CUDA Toolkit 12.8、`nvcc` | 否 | 已能编译 FlashAttention 时直接复用 |
| VS Code 与 WSL 扩展 | 否 | 继续复用 |
| 原 nano-vLLM 虚拟环境 | **不要直接复用** | 保留原环境，用于后续 A/B 对照 |
| 新项目 Python 虚拟环境 | **是** | 为 qwen3.6 fork 新建独立 venv |
| PyTorch、Triton | 是，安装到新 venv | 使用与原环境相同的工作版本 |
| FlashAttention | 是，安装到新 venv | 优先复用原环境已验证成功的精确版本或 wheel |
| Transformers | 是 | Qwen3.5 需要较新的 Transformers，建议安装 main 版本 |
| torchvision、Pillow | 是 | 多模态图像预处理必须安装 |
| nano-vLLM-qwen3.6 项目包 | 是 | 在新 venv 中执行 editable install |
| Qwen3.5-2B 模型文件 | 只下载一次 | 放在项目目录外，可被多个虚拟环境共享 |
| 测试图片 | 准备一张即可 | 建议先用 256×256 左右的普通 JPG/PNG |

---

## 1.2 为什么不建议直接复用原 nano-vLLM 虚拟环境

原版 nano-vLLM 与 nano-vLLM-qwen3.6 的 Python 包名都叫：

```text
nano-vllm
```

两个项目又都需要执行：

```bash
python -m pip install -e .
```

如果在同一个虚拟环境中分别安装两个仓库，最后一次 editable install 会改变 `nanovllm` 实际指向的源码目录，容易出现：

- 你以为在运行原版，实际导入了 qwen3.6 fork；
- 修改一个仓库后，另一个实验结果也发生变化；
- A/B 对照时无法确认加载的是哪份源码；
- Transformers 或 FlashAttention 升级后破坏原版已跑通环境。

因此推荐保留两个独立环境：

```text
nano-vllm/
└── .venv-nano

nano-vllm-qwen3.6/
└── .venv-q35
```

系统级 CUDA、GCC、WSL 不需要重复安装；真正重复的是两个 venv 中的 Python 包。

---

# 2. 推荐目录结构

```text
~/projects/
├── nano-vllm/
│   └── .venv-nano/
│
├── nano-vllm-qwen3.6/
│   └── .venv-q35/
│
~/huggingface/
└── Qwen3.5-2B/
│
~/images/
└── image_demo.jpg
```

模型不要放进 Git 仓库，也不要为两个虚拟环境各下载一份。

---

# 3. 第一步：确认旧系统环境仍然正常

进入 WSL2 Ubuntu，执行：

```bash
nvidia-smi
```

应能看到：

- NVIDIA GeForce RTX 5070 Ti Laptop GPU；
- 显存总量；
- 驱动版本；
- 当前没有其他程序大量占用显存。

继续检查：

```bash
nvcc --version
gcc --version
python3.10 --version
```

你此前已经成功运行原版 nano-vLLM，所以满足以下条件时不要重装系统组件：

```text
nvidia-smi 正常
nvcc 正常
gcc 正常
原 nano-vLLM 仍能运行
```

## 3.1 WSL2 下不要重新安装 Linux NVIDIA 驱动

WSL2 使用 Windows 主机上的 NVIDIA 驱动映射 CUDA 能力。不要在 WSL 中执行会安装 Linux 显卡驱动的命令，例如不要随意安装：

```bash
sudo apt install nvidia-driver-xxx
```

CUDA Toolkit 可以位于 WSL 内，但显卡驱动应由 Windows 侧管理。

---

# 4. 第二步：记录原 nano-vLLM 的工作版本

先找到原项目虚拟环境。以下路径需要根据你的实际目录修改：

```bash
OLD_ENV=~/projects/nano-vllm/.venv-nano
```

若你原环境叫 `.venv`，则改为：

```bash
OLD_ENV=~/projects/nano-vllm/.venv
```

查看关键版本：

```bash
$OLD_ENV/bin/python - <<'PY'
import torch
import triton
import flash_attn

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("triton:", triton.__version__)
print("flash_attn:", flash_attn.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

保存旧环境依赖：

```bash
$OLD_ENV/bin/python -m pip freeze > ~/nano-vllm-working-environment.txt
```

重点记录：

```bash
grep -Ei 'torch|torchvision|triton|flash-attn|transformers' \
  ~/nano-vllm-working-environment.txt
```

## 4.1 FlashAttention 的处理原则

RTX 5070 Ti Laptop 属于 Blackwell、计算能力通常为 `sm_120`。FlashAttention 官方 README 当前列出的 FlashAttention-2 NVIDIA 支持范围主要是 Ampere、Ada 和 Hopper，并未把消费级 Blackwell 明确列入稳定支持列表。

但你已经成功运行过原 nano-vLLM，这说明你当前使用的 FlashAttention 版本、wheel 或编译方式至少在原环境中可用。

因此：

> **不要在新环境中盲目升级到任意最新 FlashAttention。优先复制原环境中已经验证成功的精确版本和安装方式。**

若你之前安装的是本地 wheel，例如：

```text
flash_attn-*.whl
```

直接在新 venv 中重新安装同一个 wheel。

若你之前是源码编译安装，则记录版本后，在新 venv 中按同样方式重新编译。

---

# 5. 第三步：创建 nano-vLLM-qwen3.6 独立虚拟环境

进入新项目目录：

```bash
cd ~/projects/nano-vllm-qwen3.6
```

创建环境：

```bash
python3.10 -m venv .venv-q35
source .venv-q35/bin/activate
```

确认当前 Python 来自新环境：

```bash
which python
python --version
```

预期路径类似：

```text
/home/你的用户名/projects/nano-vllm-qwen3.6/.venv-q35/bin/python
```

升级基础工具：

```bash
python -m pip install -U \
  pip setuptools wheel packaging ninja psutil
```

---

# 6. 第四步：安装 PyTorch 2.8 CUDA 12.8

既然原 nano-vLLM 已使用以下组合成功运行，本轮优先保持一致：

```text
Python 3.10
PyTorch 2.8.0
torchvision 0.23.0
CUDA wheel cu128
CUDA Toolkit 12.8
```

安装：

```bash
python -m pip install \
  torch==2.8.0 \
  torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

本项目不需要音频功能，因此无需安装 `torchaudio`。

验证 GPU 与 BF16：

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

assert torch.cuda.is_available(), "PyTorch 没有识别到 CUDA"
assert torch.cuda.device_count() >= 1, "没有发现可用 GPU"

print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))

x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()

print("BF16 matmul shape:", y.shape)
print("finite:", torch.isfinite(y).all().item())
PY
```

成功标准：

```text
torch: 2.8.0+cu128
cuda available: True
gpu: NVIDIA GeForce RTX 5070 Ti Laptop GPU
finite: True
```

---

# 7. 第五步：安装 Qwen3.5 与多模态依赖

Qwen3.5 是新的模型类型，项目 `pyproject.toml` 中虽然只要求：

```text
transformers >= 4.51.0
```

但这个下限并不能保证当前安装的 Transformers 能识别 `qwen3_5`。Qwen 官方模型卡建议使用较新的 Transformers。

安装：

```bash
python -m pip install -U \
  "transformers @ git+https://github.com/huggingface/transformers.git@main"
```

安装其余依赖：

```bash
python -m pip install -U \
  huggingface_hub \
  safetensors \
  xxhash \
  pillow \
  tqdm
```

`torchvision` 已在上一节安装，它和 `Pillow` 是本项目图像预处理路径需要的依赖。

检查：

```bash
python - <<'PY'
import torch
import torchvision
import transformers
import PIL
import triton

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
print("Pillow:", PIL.__version__)
print("triton:", triton.__version__)
PY
```

---

# 8. 第六步：在新 venv 中安装 FlashAttention

## 8.1 首选：安装原环境的同一 wheel

假设你之前下载的 wheel 位于：

```text
~/wheels/flash_attn_xxx.whl
```

则执行：

```bash
python -m pip install ~/wheels/flash_attn_xxx.whl
```

这是最稳妥的方式。

## 8.2 次选：按原环境的精确版本重新编译

先查看旧版本：

```bash
$OLD_ENV/bin/python -c \
  "import flash_attn; print(flash_attn.__version__)"
```

假设输出为：

```text
2.x.x
```

则在当前 `.venv-q35` 中安装完全相同的版本：

```bash
MAX_JOBS=4 python -m pip install \
  "flash-attn==2.x.x" \
  --no-build-isolation
```

不要把示例中的 `2.x.x` 原样执行，必须替换为你的真实版本。

## 8.3 验证导入与真实 CUDA kernel

```bash
python - <<'PY'
import torch
import flash_attn
from flash_attn import flash_attn_varlen_func

print("flash_attn:", flash_attn.__version__)

dtype = torch.bfloat16
device = "cuda"

# 两条长度均为 32 的序列，总 token 数 64
q = torch.randn(64, 8, 64, device=device, dtype=dtype)
k = torch.randn(64, 8, 64, device=device, dtype=dtype)
v = torch.randn(64, 8, 64, device=device, dtype=dtype)

cu = torch.tensor([0, 32, 64], device=device, dtype=torch.int32)

out = flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q=cu,
    cu_seqlens_k=cu,
    max_seqlen_q=32,
    max_seqlen_k=32,
    causal=True,
)

torch.cuda.synchronize()
print("output shape:", out.shape)
print("finite:", torch.isfinite(out).all().item())
PY
```

成功标准：

```text
output shape: torch.Size([64, 8, 64])
finite: True
```

仅仅执行：

```python
import flash_attn
```

并不能证明 CUDA kernel 真正可用。

---

# 9. 第七步：安装 nano-vLLM-qwen3.6 源码

由于依赖已经手动安装，为避免 `pip install -e .` 自动改变 PyTorch 或 FlashAttention 版本，建议使用：

```bash
cd ~/projects/nano-vllm-qwen3.6
python -m pip install -e . --no-deps
```

检查依赖：

```bash
python -m pip check
```

语法检查：

```bash
python -m compileall -q \
  nanovllm \
  examples \
  run_text_qwen35_v2.py
```

确认当前导入的是新项目源码：

```bash
python - <<'PY'
import nanovllm
print(nanovllm.__file__)
PY
```

输出路径必须位于：

```text
~/projects/nano-vllm-qwen3.6/
```

而不能指向原 nano-vLLM 仓库。

---

# 10. 第八步：下载 Qwen3.5-2B

模型放在项目目录外：

```bash
mkdir -p ~/huggingface
```

下载：

```bash
hf download Qwen/Qwen3.5-2B \
  --local-dir ~/huggingface/Qwen3.5-2B \
  --max-workers 4
```

建议至少预留约 10 GB 空闲磁盘，避免模型、缓存和临时文件挤满系统盘。

验证配置是否被识别：

```bash
python - <<'PY'
import os
from transformers import AutoConfig

path = os.path.expanduser("~/huggingface/Qwen3.5-2B")
config = AutoConfig.from_pretrained(path)
text_config = getattr(config, "text_config", config)

print("full config:", type(config).__name__)
print("text config:", type(text_config).__name__)
print("model_type:", text_config.model_type)
print("layers:", text_config.num_hidden_layers)
print("dtype:", getattr(text_config, "dtype", None))
print("has layer_types:", hasattr(text_config, "layer_types"))
print("has vision_config:", hasattr(config, "vision_config"))
PY
```

重点检查：

```text
model_type 中包含 qwen3_5
layers 为 24
has layer_types 为 True
has vision_config 为 True
```

Qwen 官方模型卡将 Qwen3.5-2B描述为带 Vision Encoder 的 2B 模型，并采用 Gated DeltaNet 与 Gated Attention 组成的 hybrid 布局，因此这里既要识别文本子配置，也要识别视觉配置。

---

# 11. 第九步：修改文本测试脚本，避免加载视觉编码器

仓库中的：

```text
run_text_qwen35_v2.py
```

虽然名字是 text-only，但当前脚本没有显式传入：

```python
enable_vision=False
```

而项目 `Config` 中默认：

```python
enable_vision = True
```

这会导致纯文本实验也创建视觉编码器，浪费本地显存。

找到：

```python
llm = LLM(
    model_path,
    enforce_eager=args.eager,
    tensor_parallel_size=args.tp,
    max_model_len=args.max_model_len,
    max_num_batched_tokens=args.max_batched_tokens,
    max_num_seqs=1,
    gpu_memory_utilization=args.gpu_memory_utilization,
)
```

修改为：

```python
llm = LLM(
    model_path,
    enforce_eager=args.eager,
    tensor_parallel_size=args.tp,
    max_model_len=args.max_model_len,
    max_num_batched_tokens=args.max_batched_tokens,
    max_num_seqs=1,
    gpu_memory_utilization=args.gpu_memory_utilization,
    enable_vision=False,
)
```

这是本轮文本实验最重要的源码参数修改。

---

# 12. 实验一：Qwen3.5-2B BF16 单卡文本推理

## 12.1 第一遍必须使用 eager

执行：

```bash
cd ~/projects/nano-vllm-qwen3.6
source .venv-q35/bin/activate

CUDA_VISIBLE_DEVICES=0 python run_text_qwen35_v2.py \
  --model ~/huggingface/Qwen3.5-2B \
  --devices 0 \
  --tp 1 \
  --eager \
  --max-model-len 512 \
  --max-batched-tokens 128 \
  --max-tokens 32 \
  --temperature 0.0 \
  --gpu-memory-utilization 0.70
```

参数含义：

| 参数 | 本轮值 | 原因 |
|---|---:|---|
| `--devices` | `0` | 只使用笔记本唯一 GPU |
| `--tp` | `1` | 不进行多卡切分 |
| `--eager` | 开启 | 先排除 CUDA Graph 捕获问题 |
| `--max-model-len` | `512` | 先降低 KV Cache 预算 |
| `--max-batched-tokens` | `128` | 降低 warmup 和 prefill 峰值 |
| `--max-tokens` | `32` | 冒烟测试不需要长输出 |
| `--temperature` | `0.0` | 输出更稳定，便于复现 |
| `--gpu-memory-utilization` | `0.70` | 给权重、GDN state 和运行时留余量 |

## 12.2 成功标准

日志应按顺序出现类似阶段：

```text
creating model
loading weights
allocating runtime buffers
warming up model
allocating kv cache
allocating gated delta state
model runner ready
```

最终应满足：

- 没有 Python traceback；
- 没有 CUDA OOM；
- 没有 `nan`；
- 输出文本非空；
- 输出内容基本与 prompt 有关；
- `outputs[0]["token_ids"]` 非空；
- 进程结束后显存能够释放。

## 12.3 可选：验证 CUDA Graph decode

只有 eager 成功后再执行：

```bash
CUDA_VISIBLE_DEVICES=0 python run_text_qwen35_v2.py \
  --model ~/huggingface/Qwen3.5-2B \
  --devices 0 \
  --tp 1 \
  --max-model-len 512 \
  --max-batched-tokens 128 \
  --max-tokens 32 \
  --temperature 0.0 \
  --gpu-memory-utilization 0.65
```

与上一条命令相比，删除了：

```text
--eager
```

这一步不是本轮两项实验的硬性成功条件。若 CUDA Graph OOM，而 eager 能正常推理，本轮环境和文本路径仍可判定基本跑通。

---

# 13. 实验二：Qwen3.5-2B 图像 + 文本多模态冒烟测试

仓库自带的：

```text
examples/qwen3_5.py
```

存在三个不适合本地直接运行的问题：

1. 模型路径硬编码为 `Qwen3.5-9B`；
2. `max_tokens=4096`，对冒烟测试过大；
3. 没有限制 `max_num_seqs`、上下文长度和显存预算。

因此建议新建一个本地专用脚本。

## 13.1 准备图片

将一张普通 JPG 或 PNG 放到：

```text
~/images/image_demo.jpg
```

建议：

- 普通照片即可；
- 最长边先控制在约 256 像素；
- 不要第一遍使用 4K 照片；
- 图片宽高均不要小于 32 像素。

可以执行：

```bash
mkdir -p ~/images
```

## 13.2 创建脚本

在项目根目录新建：

```text
run_qwen35_2b_multimodal.py
```

内容如下：

```python
import os

from PIL import Image
from nanovllm import LLM, SamplingParams


MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3.5-2B")
IMAGE_PATH = os.path.expanduser("~/images/image_demo.jpg")


def main() -> None:
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
    if not os.path.isfile(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    image = Image.open(IMAGE_PATH).convert("RGB")
    image.thumbnail((256, 256))

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": "请用一句中文简要描述这张图片中的主要内容。",
                },
            ],
        }
    ]

    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.65,
        enable_vision=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=32,
    )

    outputs = llm.generate(
        [messages],
        sampling_params,
        use_tqdm=True,
    )

    print("\n=== RAW OUTPUT ===")
    print(outputs)

    print("\n=== TEXT ===")
    print(outputs[0]["text"].replace("<|im_end|>", ""))

    llm.exit()


if __name__ == "__main__":
    main()
```

## 13.3 运行

```bash
cd ~/projects/nano-vllm-qwen3.6
source .venv-q35/bin/activate

CUDA_VISIBLE_DEVICES=0 python run_qwen35_2b_multimodal.py
```

## 13.4 多模态成功标准

- Pillow 能正确读取图片；
- 不出现 `vision_config` 缺失；
- 不出现 `pixel_values`、`image_grid_thw` shape 错误；
- 视觉编码器权重能够加载；
- FlashAttention vision kernel 能执行；
- 模型输出非空；
- 输出内容与图片主体存在基本语义对应；
- 进程正常退出。

这是 **smoke test**，只能证明图像预处理、vision encoder、视觉 token 融合和文本生成链路能够执行，不能证明模型达到官方多模态 benchmark 精度。

---

# 14. 显存不足时如何调整

按以下顺序调整，一次只改一项。

## 14.1 文本实验 OOM

第一步：

```text
保持 --eager
```

第二步降低：

```text
--max-model-len 512
--max-batched-tokens 64
--max-tokens 16
```

第三步把：

```text
--gpu-memory-utilization 0.70
```

改为：

```text
--gpu-memory-utilization 0.65
```

或：

```text
--gpu-memory-utilization 0.60
```

但该参数不能无限降低。仓库会根据它计算 KV Cache block 数。如果出现：

```text
assert config.num_kvcache_blocks > 0
```

说明预算太低，应反向略微提高，例如从 `0.60` 提高到 `0.65` 或 `0.70`。

## 14.2 多模态实验 OOM

优先：

1. 保持 `enforce_eager=True`；
2. 将图片缩小到 224×224 或 256×256；
3. 将 `max_tokens` 降到 16；
4. 将 `max_model_len` 降到 512；
5. 将 `max_num_batched_tokens` 降到 256；
6. 将 `gpu_memory_utilization` 在 `0.60～0.70` 之间微调。

不要第一时间改模型源码或 GDN 实现。

---

# 15. 常见错误与解决方法

## 15.1 `model type qwen3_5 not recognized`

原因：

```text
Transformers 版本过旧
```

处理：

```bash
python -m pip uninstall -y transformers
python -m pip install \
  "transformers @ git+https://github.com/huggingface/transformers.git@main"
```

重新运行配置识别测试。

---

## 15.2 `No module named PIL` 或 torchvision 相关错误

处理：

```bash
python -m pip install -U pillow
python -m pip install \
  torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

---

## 15.3 FlashAttention `undefined symbol`

常见原因：

- FlashAttention 是在另一版 PyTorch 下编译的；
- 升级 PyTorch 后没有重新安装 FlashAttention；
- wheel 的 Python/CUDA/PyTorch ABI 不匹配。

处理顺序：

```bash
python -m pip uninstall -y flash-attn
```

确认 PyTorch 版本：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

然后重新安装与原 nano-vLLM 工作环境一致的 wheel 或版本。

---

## 15.4 `no kernel image is available`、`invalid device function`

这通常与 RTX 50 系列 `sm_120` 编译架构支持有关。

处理：

1. 不要先修改 nano-vLLM 业务代码；
2. 对照原 nano-vLLM 已成功运行环境的 FlashAttention 版本；
3. 使用完全相同的 wheel 或编译参数；
4. 确认当前是 CUDA 12.8 PyTorch wheel；
5. 确认不是在 Windows 原生 Python 中运行；
6. 若原环境仍能运行而新环境失败，说明问题在 Python 依赖组合，不在 GPU 或项目模型实现。

---

## 15.5 `CUDA out of memory` 出现在 `capturing cuda graphs`

处理：

```text
先加 --eager
```

本轮优先证明模型逻辑路径正确，不要求必须打开 CUDA Graph。

---

## 15.6 `assert config.num_kvcache_blocks > 0`

与普通 OOM 不同，这可能是：

```text
gpu_memory_utilization 设置得过低
```

尝试：

```text
0.60 → 0.65 → 0.70
```

同时保持：

```text
max_model_len=512
max_num_batched_tokens=64 或 128
max_num_seqs=1
```

---

## 15.7 纯文本实验也占用大量显存

检查 `run_text_qwen35_v2.py` 中是否加入：

```python
enable_vision=False
```

若未加入，项目会按默认值创建视觉编码器。

---

## 15.8 多模态脚本找不到图片

确认：

```bash
ls -lh ~/images/image_demo.jpg
```

Python 中要使用：

```python
os.path.expanduser("~/images/image_demo.jpg")
```

不能假设 Python 会自动展开字符串中的 `~`。

---

## 15.9 程序退出后端口或进程残留

检查：

```bash
ps aux | grep -E 'run_text_qwen35|run_qwen35_2b|nanovllm'
ss -ltnp | grep ':2333' || true
```

确认没有 nano-vLLM 进程后再重新运行。

---

# 16. VS Code 中选择新环境

在 WSL 终端进入项目：

```bash
cd ~/projects/nano-vllm-qwen3.6
code .
```

然后在 VS Code 中：

```text
Ctrl + Shift + P
→ Python: Select Interpreter
→ Enter interpreter path
→ ./.venv-q35/bin/python
```

在 VS Code 终端验证：

```bash
which python
python -c "import nanovllm; print(nanovllm.__file__)"
```

必须分别指向：

```text
nano-vllm-qwen3.6/.venv-q35/bin/python
nano-vllm-qwen3.6/nanovllm/
```

---

# 17. 保存本轮环境记录

两项实验成功后执行：

```bash
cd ~/projects/nano-vllm-qwen3.6
mkdir -p results/env results/logs
```

保存依赖：

```bash
python -m pip freeze > results/env/pip-freeze-qwen35-2b.txt
```

保存 GPU 信息：

```bash
nvidia-smi -q > results/env/nvidia-smi-q.txt
```

保存仓库 commit：

```bash
git rev-parse HEAD > results/env/commit.txt
git status --short > results/env/git-status.txt
```

保存文本日志：

```bash
CUDA_VISIBLE_DEVICES=0 python run_text_qwen35_v2.py \
  --model ~/huggingface/Qwen3.5-2B \
  --devices 0 \
  --tp 1 \
  --eager \
  --max-model-len 512 \
  --max-batched-tokens 128 \
  --max-tokens 32 \
  --temperature 0.0 \
  --gpu-memory-utilization 0.70 \
  2>&1 | tee results/logs/qwen35-2b-text-smoke.log
```

保存多模态日志：

```bash
CUDA_VISIBLE_DEVICES=0 python run_qwen35_2b_multimodal.py \
  2>&1 | tee results/logs/qwen35-2b-multimodal-smoke.log
```

---

# 18. 最终验收清单

## 18.1 环境

- [ ] 在 WSL2 中能通过 `nvidia-smi` 看到 5070 Ti；
- [ ] 没有重新安装 Windows、Ubuntu、GCC 或系统级 CUDA；
- [ ] 新建了独立 `.venv-q35`；
- [ ] PyTorch 2.8.0 + cu128 能执行 BF16 matmul；
- [ ] FlashAttention 真实 CUDA kernel 测试通过；
- [ ] Transformers 能识别 `qwen3_5`；
- [ ] `torchvision` 和 `Pillow` 可导入；
- [ ] `nanovllm.__file__` 指向 qwen3.6 fork。

## 18.2 实验一：文本

- [ ] 使用 `Qwen3.5-2B`；
- [ ] `tensor_parallel_size=1`；
- [ ] `enable_vision=False`；
- [ ] eager 模式成功；
- [ ] 输出 token 与文本非空；
- [ ] 无 NaN、OOM 和 traceback。

## 18.3 实验二：多模态

- [ ] 使用同一份 `Qwen3.5-2B` 权重；
- [ ] `enable_vision=True`；
- [ ] 使用缩小后的本地图片；
- [ ] 图片预处理、视觉编码器和语言模型都能执行；
- [ ] 输出内容与图片基本相关；
- [ ] 无 shape、dtype、FlashAttention 和 OOM 错误。

完成以上两项，即可认为：

> **nano-vLLM-qwen3.6 已在你的 RTX 5070 Ti 笔记本上，以 Qwen3.5-2B 成功验证文本 hybrid 推理链路和图文多模态推理链路。**

这只能证明本地功能链路跑通，暂时不能证明：

- Qwen3.5-9B 已支持；
- 多卡 TP 已支持；
- Qwen3.6-27B-FP8 已支持；
- MTP 已正确；
- 性能优于原 nano-vLLM 或 vLLM。

---

# 19. 推荐执行顺序

```text
1. 检查旧环境版本
2. 新建 .venv-q35
3. 安装 PyTorch 2.8.0 cu128
4. 安装最新 Transformers
5. 安装 torchvision、Pillow 等依赖
6. 安装原环境同版本 FlashAttention
7. pip install -e . --no-deps
8. 下载 Qwen3.5-2B
9. 验证 AutoConfig
10. 文本脚本加入 enable_vision=False
11. 运行文本 eager 冒烟测试
12. 创建并运行多模态冒烟脚本
13. 保存日志、依赖和 commit
```

---

# 20. 参考资料

1. Qwen3.5-2B 官方模型卡：  
   https://huggingface.co/Qwen/Qwen3.5-2B

2. PyTorch 2.8.0 + torchvision 0.23.0 + CUDA 12.8 官方安装组合：  
   https://pytorch.org/get-started/previous-versions/

3. FlashAttention 官方安装与支持说明：  
   https://github.com/Dao-AILab/flash-attention

4. NVIDIA CUDA on WSL User Guide：  
   https://docs.nvidia.com/cuda/wsl-user-guide/index.html


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
