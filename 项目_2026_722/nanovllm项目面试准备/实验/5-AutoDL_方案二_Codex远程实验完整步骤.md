# AutoDL 远程服务器中使用 Codex CLI 完成三组实验

> 适用方式：你先手动 SSH 登录 AutoDL，在远程服务器中进入项目目录并启动 Codex CLI，由 Codex 协助创建环境、检查代码、生成脚本、执行冒烟测试和正式实验。  
> 核心原则：**项目源码、Python 环境、模型、结果分目录存放；正式长时间实验和破坏性操作必须由你确认。**

# 一、整体流程

```text
租用 4×RTX 3090
→ 扩容数据盘
→ 本地 SSH 登录
→ 上传项目
→ 在远程服务器安装并登录 Codex CLI
→ 从项目目录启动 Codex
→ Codex 创建独立环境并安装依赖
→ Codex 下载模型
→ 检查并生成三个实验脚本
→ 冒烟测试
→ 你确认
→ 正式实验
→ 打包并下载结果
→ 关机或释放实例
```

你需要亲自完成：

- 租用、启动、关机和释放 AutoDL 实例；
- 记录 SSH 地址、端口和密码；
- 首次验证 SSH 连接；
- 上传项目；
- 完成 Codex 登录；
- 批准安装依赖、下载模型和启动正式实验；
- 下载并检查最终结果。

Codex 可以完成：

- 检查 GPU、CUDA、Python 和仓库；
- 创建独立 Conda 环境；
- 安装项目依赖；
- 下载模型；
- 生成或修正三个实验脚本；
- 增加 Scheduler 统计计数器；
- 运行冒烟测试并排查错误；
- 运行正式实验；
- 整理 JSON、CSV 和日志。

# 二、租用并启动 AutoDL 实例

在 AutoDL 创建一台**单机四卡实例**：

| 项目 | 建议配置 |
|---|---|
| GPU | 4×RTX 3090 24GB |
| 系统 | Ubuntu 22.04 |
| Python | 3.10 |
| PyTorch | 2.4 及以上的兼容镜像 |
| CUDA | 优先 12.1 或 12.4 |
| 数据盘 | 150GB 起，建议尽量扩到 180GB 以上 |
| 计费方式 | 按量计费 |

模型、环境和结果不要放在 30GB 系统盘中，统一放到：

```text
/root/autodl-tmp/
```

实例启动后，从控制台复制 SSH 命令，格式通常为：

```bash
ssh -p <SSH端口> root@<AutoDL远程地址>
```

示例：

```bash
ssh -p 35394 root@region-1.autodl.com
```

# 三、从本地电脑登录服务器

在 Windows PowerShell 中执行：

```powershell
ssh -p <SSH端口> root@<AutoDL远程地址>
```

输入 AutoDL 提供的 SSH 密码。

登录后先执行：

```bash
nvidia-smi
nvidia-smi topo -m
df -h
```

确认：

```text
可见 GPU 数量 = 4
GPU 型号 = RTX 3090
每张显存 = 24GB
/root/autodl-tmp 数据盘容量符合要求
```

再执行：

```bash
python --version
nvcc --version
```

此时不需要急着更改环境。

# 四、建立服务器目录结构

在远程服务器中执行：

```bash
mkdir -p /root/autodl-tmp/workspace
mkdir -p /root/autodl-tmp/envs
mkdir -p /root/autodl-tmp/models
mkdir -p /root/autodl-tmp/hf-cache
mkdir -p /root/autodl-tmp/pip-cache
mkdir -p /root/autodl-tmp/tmp
mkdir -p /root/autodl-tmp/results
mkdir -p /root/autodl-tmp/logs
```

最终目录结构：

```text
/root/autodl-tmp/
├── workspace/
│   └── qwen3.6kvllm/            # 项目和实验脚本
├── envs/
│   └── qwenkv/                  # 独立 Conda 环境
├── models/
│   ├── Qwen3.5-27B/
│   └── Qwen3.6-27B-FP8/
├── hf-cache/
├── pip-cache/
├── tmp/
├── results/
│   ├── exp1/
│   ├── exp2/
│   └── exp3/
└── logs/
```

设置缓存到数据盘：

```bash
cat >> /root/.bashrc <<'EOF'
export HF_HOME=/root/autodl-tmp/hf-cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache/hub
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
export TMPDIR=/root/autodl-tmp/tmp
EOF

source /root/.bashrc
```

# 五、上传项目

在本地 Windows PowerShell 中执行，注意 `-P` 为大写：

```powershell
scp -P <SSH端口> "D:\你的路径\qwen3.6kvllm.zip" `
root@<AutoDL远程地址>:/root/autodl-tmp/workspace/
```

上传完成后，在远程服务器中解压：

```bash
cd /root/autodl-tmp/workspace
unzip qwen3.6kvllm.zip
```

确认项目目录：

```bash
cd /root/autodl-tmp/workspace/qwen3.6kvllm
ls
```

应能看到：

```text
nanovllm/
pyproject.toml
README.md
实验计划 Markdown
已有测试脚本
```

不要把 Conda 环境和模型放进项目目录。

# 六、在服务器安装 Codex CLI

在服务器执行：

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

重新加载 Shell：

```bash
source /root/.bashrc
```

验证：

```bash
codex --version
```

如果提示找不到命令，执行：

```bash
find /root -type f -name codex 2>/dev/null | head
```

将实际安装目录加入 `PATH`，再执行：

```bash
codex --version
```

# 七、在无浏览器服务器中登录 Codex

优先使用设备码登录：

```bash
codex login --device-auth
```

终端会显示登录地址和一次性验证码。你需要：

1. 在本地电脑浏览器打开终端显示的地址；
2. 登录自己的 ChatGPT 账号；
3. 输入一次性验证码；
4. 回到服务器终端等待登录成功。

检查登录状态：

```bash
codex login status
```

## 设备码登录不可用时

可以先在本地电脑安装 Codex 并完成：

```bash
codex login
```

确认本地存在：

```text
~/.codex/auth.json
```

再从本地复制到服务器。Windows 中该文件一般位于用户目录下的 `.codex` 文件夹。复制前先在服务器创建目录：

```bash
mkdir -p /root/.codex
chmod 700 /root/.codex
```

然后从本地执行：

```powershell
scp -P <SSH端口> "$HOME\.codex\auth.json" `
root@<AutoDL远程地址>:/root/.codex/auth.json
```

回到服务器执行：

```bash
chmod 600 /root/.codex/auth.json
codex login status
```

`auth.json` 含登录凭据：

- 不要放入项目目录；
- 不要提交到 Git；
- 不要发给其他人；
- 实例释放前可执行 `codex logout`。

# 八、从项目目录启动 Codex

进入项目：

```bash
cd /root/autodl-tmp/workspace/qwen3.6kvllm
```

建议使用安全的交互模式启动：

```bash
codex \
  --sandbox workspace-write \
  --ask-for-approval on-request \
  --add-dir /root/autodl-tmp/envs \
  --add-dir /root/autodl-tmp/models \
  --add-dir /root/autodl-tmp/results \
  --add-dir /root/autodl-tmp/logs
```

该方式允许 Codex直接修改项目目录；当它需要联网、安装依赖或操作项目外目录时，会请求你批准。

不要使用：

```bash
codex --dangerously-bypass-approvals-and-sandbox
```

也不要让 Codex 无确认执行：

```text
rm -rf
系统 CUDA/驱动升级
大范围删除缓存
关机或释放实例
```

进入 Codex 后，可先运行：

```text
/status
```

检查当前工作目录、权限和模型。

# 九、第一次交给 Codex 的任务

将以下内容直接发送给远程 Codex：

```text
请先完整阅读当前仓库、README、pyproject.toml 和实验计划，不要立即运行正式实验。

依次完成：
1. 检查4张RTX 3090、驱动、CUDA、Python、磁盘和GPU拓扑，并将环境信息保存到 /root/autodl-tmp/results/env；
2. 在 /root/autodl-tmp/envs/qwenkv 创建独立Python 3.10环境，不修改系统CUDA和驱动；
3. 根据仓库实际依赖安装PyTorch、Triton、Transformers、FlashAttention及项目，优先复用镜像兼容版本；
4. 检查已有测试代码并严格按照实验计划生成三个脚本：
   bench_exp1_qwen35_eager.py
   bench_exp2_kv_compression.py
   bench_exp3_mtp1_acceptance.py
5. 实验二只增加Scheduler观测计数器：
   preemption_count、re_prefill_count、re_prefill_tokens；
   不改变调度和压缩算法；
6. 每个脚本支持 --smoke、--model、--output-dir，输出原始JSON、汇总CSV和完整日志；
7. 先只完成环境、代码和静态检查。安装依赖、下载模型、运行GPU任务或执行破坏性操作前向我说明并等待确认。
不得伪造实验数据，不得静默降低模型、上下文长度、输出长度或并发数。
```

# 十、让 Codex 创建独立 Python 环境

Codex提出方案后，检查它准备使用的版本。确认后允许其执行类似：

```bash
source /root/miniconda3/etc/profile.d/conda.sh

conda create \
  -p /root/autodl-tmp/envs/qwenkv \
  python=3.10 \
  -y

conda activate /root/autodl-tmp/envs/qwenkv
```

环境必须位于：

```text
/root/autodl-tmp/envs/qwenkv
```

不能创建到：

```text
项目目录/.venv
/root/miniconda3/envs
```

除非你主动决定使用系统盘。

安装完成后要求 Codex 验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

预期 GPU 数量为4。

# 十一、安装仓库依赖

让 Codex先阅读 `pyproject.toml` 和当前镜像版本，再决定具体安装命令，不要预先强制重装所有组件。

常见流程为：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/qwenkv

python -m pip install -U pip setuptools wheel ninja packaging
python -m pip install -e . --no-deps
```

再根据仓库缺失项安装：

```text
torch
triton
transformers
flash-attn
xxhash
pandas
huggingface_hub
```

要求 Codex遵守：

- 不升级服务器驱动；
- 不修改 `/usr/local/cuda`；
- 不使用 `sudo pip install`；
- 不安装到师兄或公共环境；
- FlashAttention 编译失败时先分析版本兼容关系；
- 安装完成后保存 `pip freeze`。

保存环境信息：

```bash
mkdir -p /root/autodl-tmp/results/env

python -m pip freeze \
  > /root/autodl-tmp/results/env/pip_freeze.txt

python --version \
  > /root/autodl-tmp/results/env/python_version.txt

nvcc --version \
  > /root/autodl-tmp/results/env/nvcc_version.txt

nvidia-smi \
  > /root/autodl-tmp/results/env/nvidia_smi.txt

nvidia-smi topo -m \
  > /root/autodl-tmp/results/env/gpu_topology.txt
```

# 十二、让 Codex检查并生成三个实验脚本

继续发送：

```text
现在检查三个实验脚本是否严格符合实验计划。重点核对：

实验一：
- Qwen3.5-27B；
- TP=4、Eager；
- 8条实际1024 Token输入；
- 并发8；
- 每条输出1024 Token；
- 1次预热、5次正式测试；
- 正确区分单请求TPOT和系统Decode Throughput。

实验二：
- Qwen3.5-27B；
- 16K输入、2K输出；
- 扫描并发1、2、4、8；
- 压缩关闭和开启使用独立Engine或子进程；
- 固定压缩参数；
- 记录压缩事件、活跃KV Block、物理KV Token；
- 记录preemption_count、re_prefill_count、re_prefill_tokens；
- 计算两种模式的零抢占最高并发；
- 不把预分配KV Tensor描述为nvidia-smi显存下降。

实验三：
- Qwen3.6-27B-FP8；
- 5条预热、50条正式中文Prompt；
- 每条实际512 Token输入、128 Token输出；
- MTP-1 Top-1接受率；
- 总接受率使用总Accepted除以总Attempts；
- 检查MTP权重实际加载数量大于0。

只修改完成实验所必需的代码，完成后给出文件清单、改动说明和运行命令，不要开始GPU测试。
```

你需要检查 Codex最终给出的：

```text
修改文件列表
三个脚本路径
Scheduler修改位置
命令行参数
输出文件格式
```

可以执行：

```bash
git diff --stat
git diff
```

确认没有无关框架改动。

# 十三、下载模型

模型统一放在数据盘：

```text
/root/autodl-tmp/models/Qwen3.5-27B
/root/autodl-tmp/models/Qwen3.6-27B-FP8
```

让 Codex在下载前检查：

```bash
df -h /root/autodl-tmp
```

确认空间足够后，再批准下载。

可让 Codex使用：

```bash
hf download Qwen/Qwen3.5-27B \
  --local-dir /root/autodl-tmp/models/Qwen3.5-27B \
  --max-workers 8
```

以及：

```bash
hf download Qwen/Qwen3.6-27B-FP8 \
  --local-dir /root/autodl-tmp/models/Qwen3.6-27B-FP8 \
  --max-workers 8
```

若实际 MTP 权重位于其他 Checkpoint，应以仓库 README 和代码实际要求为准，不要让 Codex凭名称猜测。

下载完成后检查：

```bash
du -sh /root/autodl-tmp/models/*
df -h /root/autodl-tmp
```

# 十四、先运行冒烟测试

不要直接启动正式实验。

先要求 Codex：

```text
请先对三个脚本分别执行 --smoke。冒烟测试统一缩小到：
- TP=4；
- 输入128 Token；
- 输出16 Token；
- 并发1；
- 重复1次。

冒烟测试只验证四卡加载、NCCL、模型前向、结果写入和资源释放。不要自动进入正式实验。每个脚本结束后汇报GPU使用、输出文件、关键计数器和任何警告。
```

在服务器另开一个 SSH 窗口监控：

```bash
watch -n 1 nvidia-smi
```

可能的冒烟命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python bench_exp1_qwen35_eager.py \
  --model /root/autodl-tmp/models/Qwen3.5-27B \
  --output-dir /root/autodl-tmp/results/exp1_smoke \
  --smoke
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python bench_exp2_kv_compression.py \
  --model /root/autodl-tmp/models/Qwen3.5-27B \
  --output-dir /root/autodl-tmp/results/exp2_smoke \
  --smoke
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python bench_exp3_mtp1_acceptance.py \
  --model /root/autodl-tmp/models/Qwen3.6-27B-FP8 \
  --output-dir /root/autodl-tmp/results/exp3_smoke \
  --smoke
```

冒烟测试必须确认：

```text
4张GPU均参与
TP=4初始化成功
无NCCL错误
无OOM
输出文件生成
程序结束后GPU显存释放
MTP权重加载数量大于0
```

# 十五、正式实验前的人工确认

你需要亲自确认以下内容：

```text
[ ] 三个冒烟测试全部成功
[ ] nvidia-smi能看到4张3090
[ ] 模型路径正确
[ ] 实验参数没有被缩小
[ ] Scheduler只增加统计字段
[ ] JSON和CSV能够正常写入
[ ] 数据盘剩余空间充足
[ ] 当前余额足够
[ ] tmux已安装
```

在正式实验前，可让 Codex输出最终参数摘要，但不要让它修改实验计划。

# 十六、使用 tmux 启动 Codex和正式实验

先退出当前 Codex会话：

```text
/exit
```

创建 tmux：

```bash
tmux new -s qwen_exp
```

进入项目并重新启动 Codex：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/qwenkv
cd /root/autodl-tmp/workspace/qwen3.6kvllm

codex \
  --sandbox workspace-write \
  --ask-for-approval on-request \
  --add-dir /root/autodl-tmp/models \
  --add-dir /root/autodl-tmp/results \
  --add-dir /root/autodl-tmp/logs
```

告诉 Codex：

```text
三个冒烟测试已通过。现在按实验计划依次运行正式实验一、实验三、实验二。

要求：
1. 每个实验开始前打印完整命令和参数，等待我确认；
2. 使用CUDA_VISIBLE_DEVICES=0,1,2,3；
3. 日志分别保存到/root/autodl-tmp/logs；
4. 结果分别保存到/root/autodl-tmp/results/exp1、exp2、exp3；
5. 一个实验失败时停止，不自动继续后续实验；
6. 不静默降低输入长度、输出长度、并发数或重复次数；
7. 不自动关机。
```

推荐顺序：

```text
实验一
→ 实验三
→ 实验二
```

实验二最耗时、显存压力最大，因此放最后。

退出 tmux 但保持任务运行：

```text
Ctrl+B
然后按 D
```

重新进入：

```bash
tmux attach -t qwen_exp
```

# 十七、正式实验运行命令

具体参数以脚本实现为准，典型命令如下。

## 实验一

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python bench_exp1_qwen35_eager.py \
  --model /root/autodl-tmp/models/Qwen3.5-27B \
  --output-dir /root/autodl-tmp/results/exp1 \
  2>&1 | tee /root/autodl-tmp/logs/exp1.log
```

## 实验三

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python bench_exp3_mtp1_acceptance.py \
  --model /root/autodl-tmp/models/Qwen3.6-27B-FP8 \
  --output-dir /root/autodl-tmp/results/exp3 \
  2>&1 | tee /root/autodl-tmp/logs/exp3.log
```

## 实验二

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python bench_exp2_kv_compression.py \
  --model /root/autodl-tmp/models/Qwen3.5-27B \
  --output-dir /root/autodl-tmp/results/exp2 \
  2>&1 | tee /root/autodl-tmp/logs/exp2.log
```

# 十八、实验完成后的检查

让 Codex执行：

```text
请检查三个实验结果，不重跑实验。确认：
1. 每个实验都有原始JSON、汇总CSV和完整日志；
2. 环境信息、模型路径和Git Commit已保存；
3. 实验一完成5轮；
4. 实验二每个并发档位均有压缩关闭和开启结果，开启组compression_event_count>0；
5. 实验二记录了三个Scheduler计数器和零抢占最高并发；
6. 实验三统计总Accepted、Rejected、Attempts和总体Accept Rate；
7. 不填补、推测或修改缺失结果；
8. 输出最终文件清单和异常清单。
```

保存 Git 信息：

```bash
cd /root/autodl-tmp/workspace/qwen3.6kvllm

git rev-parse HEAD \
  > /root/autodl-tmp/results/env/git_commit.txt

git diff \
  > /root/autodl-tmp/results/env/final_git_diff.patch
```

# 十九、打包和下载结果

在服务器中执行：

```bash
cd /root/autodl-tmp

tar -czf qwen_experiment_results.tar.gz \
  results \
  logs \
  workspace/qwen3.6kvllm/bench_exp1_qwen35_eager.py \
  workspace/qwen3.6kvllm/bench_exp2_kv_compression.py \
  workspace/qwen3.6kvllm/bench_exp3_mtp1_acceptance.py
```

在本地 Windows PowerShell 中下载：

```powershell
scp -P <SSH端口> `
root@<AutoDL远程地址>:/root/autodl-tmp/qwen_experiment_results.tar.gz `
"D:\实验结果\"
```

至少应保存：

```text
原始JSON
汇总CSV
完整日志
三个实验脚本
实验计划
pip_freeze.txt
nvidia_smi.txt
gpu_topology.txt
git_commit.txt
final_git_diff.patch
```

下载后先在本地解压检查，再关机。

# 二十、退出Codex并关机

退出 Codex：

```text
/exit
```

清除远程 Codex 登录凭据：

```bash
codex logout
```

该步骤不是必须，但如果实例准备释放，建议执行。

确认没有实验进程：

```bash
ps aux | grep -E "bench_exp|python" | grep -v grep
nvidia-smi
```

然后在 AutoDL 控制台手动关机。

只有在确认结果已下载、模型和环境不再需要时，才释放实例。

# 二十一、常见问题

## 1. Codex无法写入项目外目录

确认启动命令包含：

```bash
--add-dir /root/autodl-tmp/models
--add-dir /root/autodl-tmp/results
--add-dir /root/autodl-tmp/logs
--add-dir /root/autodl-tmp/envs
```

若仍然请求批准，逐条批准可信命令即可，不要切换为完全无沙箱模式。

## 2. Codex无法联网安装依赖

在批准提示中允许对应的 `pip`、`conda` 或 `hf download` 命令访问网络。

先手动检查：

```bash
curl -I https://pypi.org
```

## 3. 设备码登录失败

尝试：

```bash
codex login --device-auth
```

若仍失败，使用“本地登录后复制 `auth.json`”的方法。

## 4. Conda环境占用系统盘

检查环境位置：

```bash
conda env list
du -sh /root/autodl-tmp/envs/qwenkv
```

正确环境必须位于数据盘。

## 5. 模型下载到系统盘

检查：

```bash
echo $HF_HOME
du -sh /root/.cache 2>/dev/null
du -sh /root/autodl-tmp/hf-cache
```

模型应明确使用：

```bash
--local-dir /root/autodl-tmp/models/<模型名>
```

## 6. 正式实验断开SSH后停止

必须在 `tmux` 中运行。重新登录后执行：

```bash
tmux attach -t qwen_exp
```

## 7. Codex擅自缩小实验参数

停止当前执行，检查：

```bash
git diff
```

恢复实验计划要求，明确告诉 Codex：

```text
不能为了运行成功而静默降低模型、输入长度、输出长度、并发数和重复次数；无法运行时必须保留错误并停止。
```

## 8. 实验二并发8发生OOM

这是允许出现的真实结果。脚本应按1、2、4、8扫描并记录：

```text
压缩关闭零抢占最高并发
压缩开启零抢占最高并发
```

不得只修改并发8参数，也不得删除失败结果。

# 二十二、最终检查清单

```text
[ ] 租用单机4×RTX 3090
[ ] 数据盘容量足够
[ ] SSH连接成功
[ ] 项目上传到workspace
[ ] Codex CLI安装成功
[ ] Codex登录成功
[ ] Codex从项目目录启动
[ ] Conda环境位于数据盘
[ ] 模型位于数据盘
[ ] 三个实验脚本完成
[ ] Scheduler只增加统计计数器
[ ] 三个冒烟测试成功
[ ] 正式实验在tmux中运行
[ ] JSON、CSV、日志齐全
[ ] 结果已下载到本地
[ ] Codex已退出或登出
[ ] AutoDL已关机或释放


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nanovllm项目面试准备|模块-nanovllm项目面试准备]]

%% 项目关联导航：结束 %%
