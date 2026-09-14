# nano-vLLM-qwen3.6 本地配置步骤（按当前文件夹结构）

> 适用目录结构：
>
> ```text
> ~/benchmark_logs/
> ├── nano-vllm/
> └── nano-vllm-qwen3.6/
>
> ~/huggingface/
> └── Qwen3-0.6B/
>
> ~/models/
> └── Qwen3-0.6B/
>
> ~/venvs/
> └── nano-vllm-base/
>
> ~/workspace/
> ├── nano-vllm/
> └── nano-vllm-qwen3.6/
> ```
>
> 本轮只运行：
>
> 1. `Qwen3.5-2B` 单卡 BF16 文本推理；
> 2. `Qwen3.5-2B` 图像 + 文本多模态冒烟测试。
>
> 不运行 9B、27B、MTP 和多卡 TP。

---

# 1. 统一使用的目录

从现在开始统一使用以下位置：

```text
项目源码：
~/workspace/nano-vllm-qwen3.6

新虚拟环境：
~/venvs/nano-vllm-qwen3.6

已有原版环境：
~/venvs/nano-vllm-base

Qwen3.5-2B 模型：
~/huggingface/Qwen3.5-2B

实验日志：
~/benchmark_logs/nano-vllm-qwen3.6

测试图片：
~/images/image_demo.jpg
```

`~/models/Qwen3-0.6B` 和 `~/huggingface/Qwen3-0.6B` 当前存在重复。本轮先不要删除，后续建议统一把模型放在：

```text
~/huggingface/
```

---

# 2. 先确认这些目录确实位于 home 目录

执行：

```bash
echo "$HOME"

realpath ~/workspace/nano-vllm-qwen3.6
realpath ~/workspace/nano-vllm
realpath ~/venvs/nano-vllm-base
realpath ~/benchmark_logs/nano-vllm-qwen3.6
realpath ~/huggingface
```

只要都能输出真实绝对路径，就可以继续。

如果某条命令提示：

```text
No such file or directory
```

说明 VS Code 中显示的是多根工作区，不一定都位于 `~` 下。此时右键对应文件夹，复制绝对路径，并替换下文中的路径。

---

# 3. 设置临时路径变量

每次开始配置时，建议先执行：

```bash
export PROJECT="$HOME/workspace/nano-vllm-qwen3.6"
export OLD_ENV="$HOME/venvs/nano-vllm-base"
export NEW_ENV="$HOME/venvs/nano-vllm-qwen3.6"
export MODEL_DIR="$HOME/huggingface/Qwen3.5-2B"
export LOG_DIR="$HOME/benchmark_logs/nano-vllm-qwen3.6"
export IMAGE_PATH="$HOME/images/image_demo.jpg"
```

检查：

```bash
printf '%s\n' \
  "$PROJECT" \
  "$OLD_ENV" \
  "$NEW_ENV" \
  "$MODEL_DIR" \
  "$LOG_DIR" \
  "$IMAGE_PATH"
```

这些变量只在当前终端有效，关闭终端后需要重新执行。

---

# 4. 系统环境不需要重新安装

以下内容继续复用，不要重新安装：

```text
Windows NVIDIA 驱动
WSL2
Ubuntu 22.04
GCC 11
CUDA Toolkit 12.8
VS Code
Remote - WSL 扩展
```

检查：

```bash
nvidia-smi
nvcc --version
gcc --version
python3.10 --version
```

只要原 nano-vLLM 仍能运行，就不要重装系统级 CUDA 或显卡驱动。

特别注意：不要在 WSL 中随意安装 Linux NVIDIA 驱动，例如：

```bash
sudo apt install nvidia-driver-xxx
```

WSL 使用的是 Windows 主机侧 NVIDIA 驱动。

---

# 5. 不直接复用 nano-vllm-base

你目前已有：

```text
~/venvs/nano-vllm-base
```

它继续保留给原版：

```text
~/workspace/nano-vllm
```

不要直接在该环境中安装 qwen3.6 fork，因为两个项目的包名都叫：

```text
nano-vllm
```

在同一环境执行两次：

```bash
pip install -e .
```

会导致导入路径相互覆盖。

因此，本轮新建：

```text
~/venvs/nano-vllm-qwen3.6
```

---

# 6. 记录原环境的已验证版本

执行：

```bash
"$OLD_ENV/bin/python" - <<'PY'
import torch
import triton

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("triton:", triton.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

try:
    import flash_attn
    print("flash_attn:", flash_attn.__version__)
except Exception as exc:
    print("flash_attn import failed:", repr(exc))

try:
    import transformers
    print("transformers:", transformers.__version__)
except Exception as exc:
    print("transformers import failed:", repr(exc))
PY
```

保存完整依赖：

```bash
"$OLD_ENV/bin/python" -m pip freeze \
  > "$HOME/nano-vllm-base-pip-freeze.txt"
```

查看关键包：

```bash
grep -Ei \
  '^(torch|torchvision|triton|flash-attn|transformers|safetensors|huggingface-hub)' \
  "$HOME/nano-vllm-base-pip-freeze.txt"
```

这一步的作用是记录已在 5070 Ti 上验证成功的 PyTorch 与 FlashAttention 组合。

---

# 7. 创建 qwen3.6 独立虚拟环境

如果之前还没有创建：

```bash
python3.10 -m venv "$NEW_ENV"
```

激活：

```bash
source "$NEW_ENV/bin/activate"
```

确认：

```bash
which python
python --version
```

`which python` 应输出类似：

```text
/home/用户名/venvs/nano-vllm-qwen3.6/bin/python
```

升级基础工具：

```bash
python -m pip install -U \
  pip \
  setuptools \
  wheel \
  packaging \
  ninja \
  psutil
```

---

# 8. 安装 PyTorch 2.8 CUDA 12.8

按照你原项目已跑通的版本安装：

```bash
python -m pip install \
  torch==2.8.0 \
  torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

验证：

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 1

print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))

x = torch.randn(
    1024,
    1024,
    device="cuda",
    dtype=torch.bfloat16,
)
y = x @ x
torch.cuda.synchronize()

print("shape:", y.shape)
print("finite:", torch.isfinite(y).all().item())
PY
```

必须满足：

```text
cuda available: True
finite: True
```

---

# 9. 安装 Transformers 与基础依赖

先安装当前正式版：

```bash
python -m pip install -U \
  transformers \
  huggingface_hub \
  safetensors \
  xxhash \
  pillow \
  tqdm
```

检查：

```bash
python - <<'PY'
import transformers
import torch
import torchvision
import PIL
import triton

print("transformers:", transformers.__version__)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("Pillow:", PIL.__version__)
print("triton:", triton.__version__)
PY
```

后面下载模型后，如果出现：

```text
model type qwen3_5 not recognized
```

再执行：

```bash
python -m pip uninstall -y transformers
python -m pip install \
  "transformers @ git+https://github.com/huggingface/transformers.git@main"
```

不要在模型尚未下载前先判断 Transformers 是否支持，因为真正的验证方式是读取 `Qwen3.5-2B/config.json`。

---

# 10. 安装与原环境一致的 FlashAttention

先读取原环境版本：

```bash
FLASH_VERSION=$(
  "$OLD_ENV/bin/python" -c \
  "import flash_attn; print(flash_attn.__version__)"
)

echo "$FLASH_VERSION"
```

如果你此前使用的是专门适配 5070 Ti 的本地 wheel，应优先重新安装该 wheel，例如：

```bash
python -m pip install ~/wheels/flash_attn_xxx.whl
```

如果此前是源码编译安装，则使用同一版本：

```bash
MAX_JOBS=4 python -m pip install \
  "flash-attn==$FLASH_VERSION" \
  --no-build-isolation
```

验证导入：

```bash
python - <<'PY'
import flash_attn
from flash_attn import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)

print("flash_attn:", flash_attn.__version__)
print("varlen:", flash_attn_varlen_func)
print("kvcache:", flash_attn_with_kvcache)
PY
```

验证真实 kernel：

```bash
python - <<'PY'
import torch
from flash_attn import flash_attn_varlen_func

q = torch.randn(
    64,
    8,
    64,
    device="cuda",
    dtype=torch.bfloat16,
)
k = torch.randn_like(q)
v = torch.randn_like(q)

cu = torch.tensor(
    [0, 32, 64],
    device="cuda",
    dtype=torch.int32,
)

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

print("shape:", out.shape)
print("finite:", torch.isfinite(out).all().item())
PY
```

只有真实 kernel 成功，才进入下一步。

---

# 11. 安装 nano-vLLM-qwen3.6 项目

进入源码目录：

```bash
cd "$PROJECT"
```

确认关键文件存在：

```bash
ls
ls nanovllm
ls examples
ls run_text_qwen35_v2.py
```

安装项目：

```bash
python -m pip install -e . --no-deps
```

检查依赖：

```bash
python -m pip check
```

检查语法：

```bash
python -m compileall -q \
  nanovllm \
  examples \
  run_text_qwen35_v2.py
```

确认导入位置：

```bash
python - <<'PY'
import nanovllm
print(nanovllm.__file__)
PY
```

输出必须位于：

```text
~/workspace/nano-vllm-qwen3.6/
```

不能指向：

```text
~/workspace/nano-vllm/
```

---

# 12. 下载 Qwen3.5-2B

创建模型目录：

```bash
mkdir -p "$HOME/huggingface"
```

下载：

```bash
hf download Qwen/Qwen3.5-2B \
  --local-dir "$MODEL_DIR" \
  --max-workers 4
```

检查：

```bash
ls -lh "$MODEL_DIR"
```

至少应能看到：

```text
config.json
tokenizer.json
tokenizer_config.json
*.safetensors
```

---

# 13. 验证 Qwen3.5-2B 配置

执行：

```bash
python - <<'PY'
import os
from transformers import AutoConfig

path = os.path.expanduser(
    "~/huggingface/Qwen3.5-2B"
)

config = AutoConfig.from_pretrained(path)
text_config = getattr(config, "text_config", config)

print("full config:", type(config).__name__)
print("text config:", type(text_config).__name__)
print("model_type:", text_config.model_type)
print("layers:", text_config.num_hidden_layers)
print("dtype:", getattr(text_config, "dtype", None))
print(
    "has layer_types:",
    hasattr(text_config, "layer_types"),
)
print(
    "has vision_config:",
    hasattr(config, "vision_config"),
)
PY
```

如果出现：

```text
model type qwen3_5 not recognized
```

执行：

```bash
python -m pip uninstall -y transformers

python -m pip install \
  "transformers @ git+https://github.com/huggingface/transformers.git@main"
```

然后重新运行配置验证。

---

# 14. 修改文本脚本：关闭视觉编码器

打开：

```text
~/workspace/nano-vllm-qwen3.6/run_text_qwen35_v2.py
```

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

原因：

```text
Config.enable_vision 默认是 True
```

若不关闭，纯文本实验也会创建视觉编码器，浪费显存。

---

# 15. 创建日志目录

执行：

```bash
mkdir -p "$LOG_DIR"
mkdir -p "$LOG_DIR/env"
mkdir -p "$LOG_DIR/text"
mkdir -p "$LOG_DIR/multimodal"
```

最终为：

```text
~/benchmark_logs/nano-vllm-qwen3.6/
├── env/
├── text/
└── multimodal/
```

---

# 16. 实验一：Qwen3.5-2B 单卡文本推理

进入项目并激活环境：

```bash
cd "$PROJECT"
source "$NEW_ENV/bin/activate"
```

第一遍必须使用 eager：

```bash
CUDA_VISIBLE_DEVICES=0 \
python run_text_qwen35_v2.py \
  --model "$MODEL_DIR" \
  --devices 0 \
  --tp 1 \
  --eager \
  --max-model-len 512 \
  --max-batched-tokens 128 \
  --max-tokens 32 \
  --temperature 0.0 \
  --gpu-memory-utilization 0.70 \
  2>&1 | tee "$LOG_DIR/text/qwen35-2b-eager.log"
```

参数含义：

```text
--devices 0
只使用第 0 张 GPU。

--tp 1
单卡，不做张量并行。

--eager
先跳过 CUDA Graph。

--max-model-len 512
降低 KV Cache 预算。

--max-batched-tokens 128
限制单次 prefill token 数。

--max-tokens 32
冒烟测试只生成少量 token。

--temperature 0.0
便于重复对照。

--gpu-memory-utilization 0.70
为权重和运行时预留空间。
```

成功标准：

```text
权重加载完成
没有 CUDA OOM
没有 traceback
没有 NaN
输出 token_ids 非空
输出文本非空
文本与提示词相关
进程正常退出
```

---

# 17. 可选：验证 CUDA Graph

只有 eager 成功后才运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
python run_text_qwen35_v2.py \
  --model "$MODEL_DIR" \
  --devices 0 \
  --tp 1 \
  --max-model-len 512 \
  --max-batched-tokens 128 \
  --max-tokens 32 \
  --temperature 0.0 \
  --gpu-memory-utilization 0.65 \
  2>&1 | tee "$LOG_DIR/text/qwen35-2b-graph.log"
```

与 eager 命令相比，仅删除了：

```text
--eager
```

如果 eager 正常、CUDA Graph OOM，本轮文本功能仍可判定基本跑通。CUDA Graph 后续单独排查。

---

# 18. 准备多模态图片

创建目录：

```bash
mkdir -p "$HOME/images"
```

把一张普通 JPG 图片放到：

```text
~/images/image_demo.jpg
```

检查：

```bash
ls -lh "$IMAGE_PATH"
```

建议第一遍使用：

```text
JPG 或 PNG
RGB 图片
最长边约 256 像素
不是 4K 大图
```

---

# 19. 创建 Qwen3.5-2B 多模态脚本

在项目根目录执行：

```bash
cd "$PROJECT"
```

创建：

```bash
cat > run_qwen35_2b_multimodal.py <<'PY'
import os

from PIL import Image
from nanovllm import LLM, SamplingParams


MODEL_PATH = os.path.expanduser(
    "~/huggingface/Qwen3.5-2B"
)
IMAGE_PATH = os.path.expanduser(
    "~/images/image_demo.jpg"
)


def main() -> None:
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_PATH}"
        )

    if not os.path.isfile(IMAGE_PATH):
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

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
                    "text": (
                        "请用一句中文简要描述"
                        "这张图片中的主要内容。"
                    ),
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
    text = outputs[0]["text"]
    print(text.replace("<|im_end|>", ""))

    llm.exit()


if __name__ == "__main__":
    main()
PY
```

语法检查：

```bash
python -m py_compile \
  run_qwen35_2b_multimodal.py
```

---

# 20. 实验二：图像 + 文本多模态

运行：

```bash
cd "$PROJECT"
source "$NEW_ENV/bin/activate"

CUDA_VISIBLE_DEVICES=0 \
python run_qwen35_2b_multimodal.py \
  2>&1 | tee \
  "$LOG_DIR/multimodal/qwen35-2b-image-text.log"
```

成功标准：

```text
图片能够被 Pillow 读取
视觉编码器权重能够加载
pixel_values 和 image_grid_thw 不报错
FlashAttention vision kernel 能执行
模型输出非空
输出与图片主要内容基本相关
进程正常退出
```

这里只证明多模态代码链路能运行，不等于完成官方视觉 benchmark。

---

# 21. OOM 调整顺序

## 21.1 文本实验

先保持：

```text
--eager
--tp 1
```

依次降低：

```text
--max-model-len 512
--max-batched-tokens 64
--max-tokens 16
```

显存比例可尝试：

```text
0.60
0.65
0.70
```

注意：

```text
gpu_memory_utilization 不是越低越好
```

如果出现：

```text
assert config.num_kvcache_blocks > 0
```

说明为 KV Cache 分配的预算过低，应把：

```text
0.60 调回 0.65 或 0.70
```

## 21.2 多模态实验

依次调整：

```text
图片缩小到 224×224
max_tokens 改为 16
max_model_len 改为 512
max_num_batched_tokens 改为 256
gpu_memory_utilization 在 0.60～0.70 之间微调
```

保持：

```python
enforce_eager=True
max_num_seqs=1
tensor_parallel_size=1
```

---

# 22. 常见错误

## 22.1 导入了原项目源码

执行：

```bash
python - <<'PY'
import nanovllm
print(nanovllm.__file__)
PY
```

如果指向：

```text
~/workspace/nano-vllm/
```

说明环境或 editable install 错误。

重新执行：

```bash
source "$NEW_ENV/bin/activate"
cd "$PROJECT"
python -m pip install -e . --no-deps
```

---

## 22.2 `qwen3_5 not recognized`

升级 Transformers：

```bash
python -m pip uninstall -y transformers

python -m pip install \
  "transformers @ git+https://github.com/huggingface/transformers.git@main"
```

---

## 22.3 FlashAttention `undefined symbol`

原因通常是 PyTorch 与 FlashAttention ABI 不匹配。

处理：

```bash
python -m pip uninstall -y flash-attn
```

确认 PyTorch：

```bash
python -c \
  "import torch; print(torch.__version__, torch.version.cuda)"
```

然后安装与原 `nano-vllm-base` 一致的 FlashAttention wheel 或版本。

---

## 22.4 `no kernel image` 或 `invalid device function`

优先检查：

```text
是否使用原环境同一 FlashAttention 构建
是否使用 torch 2.8.0 cu128
是否在 WSL2 中运行
是否真实针对 RTX 50 系列架构编译
```

不要先改 nano-vLLM 模型代码。

---

## 22.5 文本实验显存异常高

检查：

```python
enable_vision=False
```

是否已经添加到 `run_text_qwen35_v2.py`。

---

## 22.6 CUDA Graph capture OOM

先回到：

```text
--eager
```

本轮以 eager 成功为主要验收条件。

---

## 22.7 退出后残留进程

检查：

```bash
ps aux | grep -E \
  'nanovllm|run_text_qwen35|run_qwen35_2b'
```

检查端口：

```bash
ss -ltnp | grep ':2333' || true
```

确认没有 nano-vLLM 进程后再重新运行。

---

# 23. VS Code 选择解释器

打开项目：

```bash
cd "$PROJECT"
code .
```

在 VS Code 中：

```text
Ctrl + Shift + P
Python: Select Interpreter
Enter interpreter path
```

输入完整路径：

```text
/home/你的用户名/venvs/nano-vllm-qwen3.6/bin/python
```

可在终端执行：

```bash
echo "$HOME/venvs/nano-vllm-qwen3.6/bin/python"
```

复制输出路径。

验证：

```bash
which python

python - <<'PY'
import nanovllm
print(nanovllm.__file__)
PY
```

---

# 24. 保存环境信息

执行：

```bash
mkdir -p "$LOG_DIR/env"
```

保存 Python 包：

```bash
python -m pip freeze \
  > "$LOG_DIR/env/pip-freeze.txt"
```

保存 GPU 信息：

```bash
nvidia-smi -q \
  > "$LOG_DIR/env/nvidia-smi-q.txt"
```

保存仓库版本：

```bash
cd "$PROJECT"

git rev-parse HEAD \
  > "$LOG_DIR/env/commit.txt"

git status --short \
  > "$LOG_DIR/env/git-status.txt"
```

保存路径信息：

```bash
{
  echo "PROJECT=$PROJECT"
  echo "OLD_ENV=$OLD_ENV"
  echo "NEW_ENV=$NEW_ENV"
  echo "MODEL_DIR=$MODEL_DIR"
  echo "LOG_DIR=$LOG_DIR"
  echo "IMAGE_PATH=$IMAGE_PATH"
} > "$LOG_DIR/env/paths.txt"
```

---

# 25. 最终目录结果

成功后，你的目录应大致变成：

```text
~/benchmark_logs/
├── nano-vllm/
└── nano-vllm-qwen3.6/
    ├── env/
    ├── text/
    │   ├── qwen35-2b-eager.log
    │   └── qwen35-2b-graph.log
    └── multimodal/
        └── qwen35-2b-image-text.log

~/huggingface/
├── Qwen3-0.6B/
└── Qwen3.5-2B/

~/models/
└── Qwen3-0.6B/

~/venvs/
├── nano-vllm-base/
└── nano-vllm-qwen3.6/

~/workspace/
├── nano-vllm/
└── nano-vllm-qwen3.6/
    ├── nanovllm/
    ├── examples/
    ├── run_text_qwen35_v2.py
    └── run_qwen35_2b_multimodal.py

~/images/
└── image_demo.jpg
```

---

# 26. 最终验收

## 环境

- [ ] `~/venvs/nano-vllm-base` 保持不变；
- [ ] 新建 `~/venvs/nano-vllm-qwen3.6`；
- [ ] PyTorch BF16 CUDA 测试通过；
- [ ] FlashAttention 真实 kernel 测试通过；
- [ ] `nanovllm.__file__` 指向 qwen3.6 fork；
- [ ] Qwen3.5-2B 位于 `~/huggingface/Qwen3.5-2B`。

## 文本实验

- [ ] `TP=1`；
- [ ] `enable_vision=False`；
- [ ] eager 模式成功；
- [ ] 输出文本和 token 非空；
- [ ] 无 OOM、NaN 和 traceback。

## 多模态实验

- [ ] `enable_vision=True`；
- [ ] 图片位于 `~/images/image_demo.jpg`；
- [ ] 图片经过缩小；
- [ ] 图像编码与文本生成都成功；
- [ ] 输出与图片基本相关；
- [ ] 日志保存到 `~/benchmark_logs/nano-vllm-qwen3.6`。

完成以上内容，就可以认定：

```text
nano-vLLM-qwen3.6 已在 RTX 5070 Ti Laptop 上，
使用 Qwen3.5-2B 完成单卡文本 hybrid 推理链路
和图像 + 文本多模态链路验证。
```


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
