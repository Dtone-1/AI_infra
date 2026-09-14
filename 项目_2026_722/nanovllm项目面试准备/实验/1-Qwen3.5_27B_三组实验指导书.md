# Qwen3.5-27B 与 MTP 三组实验指导书

> 硬件条件：4 × RTX 3090，Tensor Parallel Size = 4  
> 项目范围：基于 nano-vLLM 的 Qwen3.5 Hybrid 适配、动态 KV Cache 压缩和 MTP 原型  
> 文档目的：明确三组实验分别证明什么、怎样测量、哪些结果可以写入简历，以及哪些结论不能由当前实验推出。

---

## 一、总体判断与必要调整

你的三组实验方向基本合理，分别对应项目中的三个主要工作：

1. Qwen3.5 Hybrid 架构适配与 CUDA Graph 执行；
2. 长上下文下的动态 KV Cache 压缩；
3. MTP Draft/Verify 原型。

但实验定义需要做三处关键调整。

### 1. CUDA Graph 不能单独“证明适配正确”

只运行 CUDA Graph 路径，只能说明程序没有报错，不能证明 Graph 路径与正常执行语义一致。因此实验一必须以 **Eager 路径作为正确性基线**，在同一模型、同一输入和贪心解码条件下比较：

- 输出 Token ID 是否一致；
- 首个不一致位置；
- TTFT、TPOT 和 Decode Throughput 的变化。

实验一最终证明的是：

> Qwen3.5-27B 在本项目中能够以 TP=4 正确完成推理，并且 CUDA Graph 相对 Eager 在保持输出一致的前提下改善 Decode 执行效率。

它不能单独证明本项目与官方 Transformers/vLLM 的全部数值完全一致。若需要证明“模型适配与官方实现一致”，还需要另做少量官方实现对照，但该内容不作为本次三组核心实验的主结果。

### 2. KV Cache 压缩的三个主指标合理，但必须增加质量约束

你提出的三项主指标是合理的：

1. 吞吐量变化；
2. 并发收益；
3. KV Cache 收益。

不过压缩会改变模型可见的历史 KV，若只展示性能收益而完全不检查输出变化，实验结论是不完整的。因此本指导书将“输出质量”设为 **约束指标**，不作为第四项主要性能结论：

- 至少记录压缩开关前后的 Token 公共前缀长度和首次不一致位置；
- 若简历继续保留“LongBench/Needle 精度下降控制在 Δ 内”，则必须另做正式质量评测；
- 若不做质量评测，应从简历中删除该项精度结论。

### 3. 当前仓库的 MTP 测试目标是 Qwen3.6-27B-FP8

当前 MTP 脚本默认模型为 `Qwen3.6-27B-FP8`，README 也将其定义为 Qwen3.6 MTP 原型。实验三应采用：

> Qwen3.6-27B-FP8，4 × RTX 3090，TP=4，KV 压缩关闭。

若要将实验三统一为 Qwen3.5-27B，必须先确认 Qwen3.5 checkpoint 中的 MTP 权重命名、加载数量和 Draft 前向结果均正确，不能直接把当前 Qwen3.6 的实验结果写成 Qwen3.5 MTP 结果。

---

## 二、仓库现状与实验边界

本实验设计主要依据以下代码和文档：

- `bench_qwen35_fixed.py`
- `bench_kv_compression.py`
- `bench_mtp_draft_sweep.py`
- `run_mtp_fast_decode.py`
- `nanovllm/config.py`
- `nanovllm/engine/model_runner.py`
- `nanovllm/engine/scheduler.py`
- `docs/kv_compression_known_limitations.md`
- `README.md`

### 1. 现有 Qwen3.5 Benchmark 的限制

`bench_qwen35_fixed.py` 当前：

- 仅使用一个很短的固定 Prompt；
- `max_model_len=1024`、`max_num_seqs=1` 被硬编码；
- 主要输出 Prefill tok/s 和 Decode tok/s；
- 没有完整记录 TTFT、TPOT 分位数和输出 Token 一致性。

因此它只能作为冒烟测试脚本，不能原样作为实验一的最终 Benchmark。

### 2. 现有 KV 压缩 Benchmark 的限制

`bench_kv_compression.py` 已经支持：

- 压缩关闭/开启的成对测试；
- TTFT、TPOT、吞吐量；
- 物理 KV Token、物理 KV Block、峰值已用 Block；
- 压缩事件数、压缩步骤耗时、释放 Block 数；
- 压缩开关前后的输出 Token 对比；
- JSON 结果输出。

但当前脚本把 `max_num_seqs` 固定为 1，因此只能进行单请求对照，不能测量高并发吞吐和最大并发。实验二需要在该脚本基础上增加多请求和并发阶梯测试。

### 3. 压缩机制自身的边界

当前实现具有以下明确边界：

- 压缩只发生在 Decode 阶段，Prefill 仍需先物化完整 Prompt KV；
- 只压缩 Full Attention 层的 K/V，Gated DeltaNet 的 recurrent state 和 conv state 不压缩；
- 压缩步骤走 Eager，普通 Decode 步骤随后可以恢复 CUDA Graph；
- 压缩开启时 Prefix Cache 被关闭；
- KV 压缩和 MTP 互斥，不能在同一次实验中同时开启；
- 底层 KV 大 Tensor 在初始化时预分配，释放 Block 不一定使 `nvidia-smi` 显存占用下降；
- 当前压缩逻辑包含 Python 控制流、`.item()` 同步和临时 Gather，不是融合压缩 Kernel。

因此实验二的 KV 收益必须使用“活跃物理 KV Token/Block”和“可复用 Block”表示，不能直接写成“nvidia-smi 显存下降”。

### 4. 现有 MTP Benchmark 的限制

`bench_mtp_draft_sweep.py` 当前：

- 默认运行 Qwen3.6-27B-FP8；
- 默认扫描 Draft Length 1、2、3、4；
- 输出接受率、Decode tok/s、Target Forward 次数、MTP Forward 次数和拒绝重算次数；
- 只使用一个 Prompt；
- 每种 Draft Length 会重新加载模型，适合小规模原型验证，不适合直接跑大规模 Prompt 集。

实验三至少需要支持从 JSONL 读取多条 Prompt，并在同一模型实例中累计统计接受率。

---

## 三、4 × RTX 3090 与 Qwen3.5-27B 可行性判断

Qwen3.5-27B 官方配置包含：

- 64 个 Decoder Layer；
- 每 4 层一个 Full Attention，共 16 个 Full Attention Layer；
- 48 个 Gated DeltaNet Layer；
- 4 个 KV Head；
- Head Dimension 为 256；
- 16 个线性注意力 Key Head 和 48 个 Value Head。

TP=4 时，上述 Attention Head 和 GDN Head 数均可被 4 整除，当前代码中的 TP 分片约束在结构上成立。

官方完整 checkpoint 约 55.6 GB，并包含视觉模块。理想均匀切分时，每张卡约承担 13.9 GB 权重；本项目在 `enable_vision=False` 时不会实例化视觉编码器，未匹配的视觉权重会被跳过，因此文本实验的实际常驻权重应小于完整 checkpoint 大小。

### 1. 理论运行时状态估算

Qwen3.5-27B 的 Full Attention KV Cache：

```text
全部 GPU 合计每 Token：
2(K/V) × 16 层 × 4 KV Head × 256 Head Dim × 2 Byte
= 64 KiB / Token

TP=4 后每张 GPU：
16 KiB / Token
```

当前 GDN 状态按 TP=4 估算，每个活跃请求、每张 GPU 约占：

```text
conv state + recurrent state ≈ 36.7 MiB
```

因此单请求每张 GPU 的理论状态量约为：

| 逻辑上下文 | Full Attention KV | GDN State | 合计约值 |
|---:|---:|---:|---:|
| 8K | 128 MiB | 36.7 MiB | 164.7 MiB |
| 16K | 256 MiB | 36.7 MiB | 292.7 MiB |
| 32K | 512 MiB | 36.7 MiB | 548.7 MiB |

以上不包含 Block 向上取整、CUDA Graph 池、NCCL、FlashAttention 临时空间和其他 Runtime Buffer，只能用于估计实验阶梯，不能当作实际显存结果。

### 2. 最终判断

四张 24 GB RTX 3090 在 TP=4、文本模式下运行 Qwen3.5-27B **具有较高可行性，但当前仓库没有给出该硬件与该模型的实测保证**。正式实验前必须先做以下冒烟测试：

1. `enable_vision=False`；
2. `max_num_seqs=1`；
3. `max_model_len=4096`；
4. 先 Eager，再 CUDA Graph；
5. 从较保守的 `gpu_memory_utilization` 开始；
6. 确认四个 TP Rank 均完成权重加载、Warmup、KV 分配、GDN State 分配和 Graph Capture。

需要特别注意：`allocate_gdn_state()` 会根据剩余显存和 `max_num_seqs` 分配状态池，而 CUDA Graph 在其后捕获。高并发实验中不能把 `max_num_seqs` 随意设为 512，应针对每个并发阶梯设置为实际测试值，否则可能因预分配大量 GDN State 和 Graph Bucket 导致 OOM。

---

## 四、公共实验规范

三组实验均遵循以下规则。

### 1. 固定环境

每次结果必须记录：

- GPU 型号和数量；
- TP Size；
- 驱动版本、CUDA 版本、PyTorch 版本、FlashAttention 版本；
- 仓库 Git Commit 和是否存在未提交修改；
- 模型目录、模型版本和 dtype；
- `max_model_len`、`max_num_batched_tokens`、`max_num_seqs`；
- `gpu_memory_utilization`；
- Eager/Graph、Vision、MTP、KV Compression 开关；
- Prompt Token 数和固定生成 Token 数。

### 2. 固定推理条件

除非实验另有说明，统一设置：

```text
temperature = 0.0
ignore_eos = True
enable_vision = False
固定输出长度
相同 tokenizer 和 chat template
```

使用贪心解码是为了排除随机采样带来的输出差异。

### 3. Warmup 与计时

- 模型加载时间不得计入 TTFT、TPOT 和吞吐量；
- CUDA Graph 首次捕获和首次 Replay 不得计入正式结果；
- 每种配置至少 Warmup 1 次，正式重复至少 5 次；
- 计时前后执行 CUDA Synchronize；
- 报告中优先使用 Median/P50，同时保留 P95；
- 交替执行对照组，避免温度和频率漂移长期偏向某一组。

### 4. 指标定义

#### TTFT

从请求进入引擎到第一个输出 Token 产生的时间：

```text
TTFT = first_token_timestamp - request_submit_timestamp
```

#### TPOT

第一个输出 Token 之后，后续 Token 的平均生成间隔：

```text
TPOT = (end_timestamp - first_token_timestamp) / (output_tokens - 1)
```

多请求场景报告每个请求 TPOT 的 P50/P95。

#### Decode Throughput

```text
Decode Throughput = 全部生成 Token 数 / Decode 阶段墙钟时间
```

实验一使用单请求 Decode Throughput；实验二使用系统总 Decode Throughput。

#### E2E Output Throughput

```text
E2E Output Throughput = 全部生成 Token 数 / 从首个请求提交到全部请求结束的时间
```

长 Prompt 场景中 Prefill 占比较大，因此实验二同时保留 E2E Throughput 作为辅助指标，但核心比较使用 Decode Throughput。

---

## 五、实验一：Qwen3.5 Hybrid CUDA Graph 正确性与执行效率

### 1. 实验目的

验证 Qwen3.5-27B 在项目中的 TP=4 Hybrid 推理路径能够正确运行，并测量 CUDA Graph 相对 Eager 对 Decode 阶段的影响。

### 2. 实验变量

唯一主变量：

| 组别 | `enforce_eager` | 说明 |
|---|---:|---|
| Eager Baseline | `True` | 正确性参考和性能基线 |
| CUDA Graph | `False` | 正式优化路径 |

固定关闭：

```text
kv_compress_enabled = False
enable_mtp = False
enable_vision = False
max_num_seqs = 1
TP = 4
```

### 3. 推荐输入矩阵

准备 10 条固定文本 Prompt，并通过 tokenizer 截取或填充到以下实际 Token 长度：

```text
1K / 4K / 16K Prompt Token
固定生成 256 Token
```

推荐设置：

```text
max_num_batched_tokens = 4096
```

这样 1K、4K 可一次 Prefill，16K 会经过固定的 Chunked Prefill。两组必须保持该参数完全一致。

若 27B 在 16K 条件下运行成本过高，简历核心结果可只选 4K，16K 作为补充结果，但不能只使用几十 Token 的短 Prompt。

### 4. 正确性判据

对每条 Prompt 比较 Eager 与 CUDA Graph：

- 输出 Token 数必须相同；
- `output_token_ids` 必须完全相同；
- 记录 `exact_token_match`；
- 记录 `common_prefix_tokens`；
- 如不一致，记录首个不一致 Token 位置和两侧 Token ID。

主实验的通过条件：

```text
全部测试 Prompt 的 Eager/Graph 输出 Token 完全一致。
```

如果出现少量不一致，不得只展示平均性能，需要先定位 Graph State、GDN State Slot、Block Table、Position 或采样路径是否存在差异。

### 5. 性能指标

核心指标：

1. TTFT P50/P95；
2. TPOT P50/P95；
3. Decode Throughput；
4. E2E Output Throughput，作为辅助结果。

预计 CUDA Graph 主要改善 TPOT 和 Decode Throughput。由于当前 Graph 主要覆盖 Decode，TTFT 大幅改善不是预期结论；若 TTFT 基本不变属于正常结果。

### 6. 现有脚本需要补充的功能

基于 `bench_qwen35_fixed.py` 或 `bench_kv_compression.py` 的计时逻辑，增加：

- `--devices`；
- 可配置 `max_model_len`、`max_num_batched_tokens`；
- 多 Prompt 输入；
- 每请求 TTFT 和 TPOT；
- Eager/Graph 成对运行；
- 输出 Token ID 对比；
- JSON/CSV 保存；
- P50/P95 汇总。

当前 `bench_qwen35_fixed.py` 的短 Prompt 和平均 tok/s 结果不能直接作为简历最终数字。

### 7. 最终结果表

| Prompt 长度 | 模式 | TTFT P50 | TPOT P50 | TPOT P95 | Decode tok/s | Token 完全一致 |
|---:|---|---:|---:|---:|---:|---:|
| 1K | Eager |  |  |  |  | — |
| 1K | Graph |  |  |  |  |  |
| 4K | Eager |  |  |  |  | — |
| 4K | Graph |  |  |  |  |  |
| 16K | Eager |  |  |  |  | — |
| 16K | Graph |  |  |  |  |  |

### 8. 可写入简历的结论边界

可以写：

> 在 Qwen3.5-27B、4 × RTX 3090、TP=4 下完成 Eager/CUDA Graph 贪心输出一致性验证；CUDA Graph 将 4K 输入、256 Token 输出场景的 TPOT P50 改善 X%，Decode Throughput 提升 Y%。

不能写：

- “Qwen3.5 比 Qwen3 快 X%”；
- “CUDA Graph 优化了 Prefill”；
- “与官方 vLLM 完全等价”，除非另做官方实现对照。

---

## 六、实验二：Qwen3.5-27B 动态 KV Cache 压缩

### 1. 实验目的

在完全相同模型、请求和运行参数下，对比压缩关闭与开启时的：

1. 长上下文高并发 Decode Throughput；
2. 可持续并发容量；
3. 活跃物理 KV Token/Block 使用量。

### 2. 对照组

| 组别 | KV 压缩 | 其他配置 |
|---|---:|---|
| Baseline | 关闭 | 与压缩组完全相同 |
| Compression | 开启 | 仅改变压缩开关和压缩参数 |

两组统一：

```text
Qwen3.5-27B
4 × RTX 3090
TP = 4
CUDA Graph 开启
enable_vision = False
enable_mtp = False
temperature = 0.0
ignore_eos = True
```

当前实现规定 MTP 与 KV 压缩互斥，实验二不能同时启用 MTP。

### 3. 推荐压缩参数

核心实验不做大规模参数消融，固定一套参数：

```text
kv_compress_period = 128
kv_compress_window_blocks = 4
kvcache_block_size = 256
kv_compress_keep_ratio = 0.5
kv_compress_smoothing_window = 5
kv_compress_sink_tokens = 1
kv_compress_min_tokens = 0
kv_compress_topk >= 当前测试并发数
```

该配置中：

```text
窗口大小 = 4 × 256 = 1024 Token
每次保留 = 512 Token
理论上每次压缩窗口丢弃 = 512 Token
```

将 `topk` 设为不小于当前并发数，是为了先测量压缩机制本身的容量收益，避免部分请求因 Top-K 调度上限没有被压缩。Top-K 对压缩开销的影响可作为后续补充消融，不进入本次三个核心实验。

### 4. 推荐负载矩阵

#### Smoke

```text
Prompt = 8K Token
Output = 512 Token
Concurrency = 1
kv_compress_period = 64
```

Smoke 的唯一目的，是确认：

```text
compression_event_count > 0
released_kv_blocks > 0
```

若两项不满足，不得进入正式统计。

#### Main

```text
Prompt = 16K Token
Output = 2048 Token
Concurrency 阶梯 = 1 / 2 / 4 / 8 / 12 / 16，直到容量失败
```

#### Stress

```text
Prompt = 32K Token
Output = 2048 Token
Concurrency 阶梯 = 1 / 2 / 4 / 6 / 8，直到容量失败
```

实际 Prompt 长度必须由 tokenizer 验证，不能用字符数代替 Token 数。

### 5. 为什么并发实验需要分成两部分

压缩是 Decode-only，所有 Prompt 在 Prefill 时仍需占用完整 KV。因此若一次性提交大量 32K Prompt，Baseline 和 Compression 都必须先容纳全部原始 Prompt KV，压缩无法帮助尚未进入 Decode 的请求。

实验二需要包含两个子测试。

#### 子测试 A：固定并发吞吐与 KV 使用

- 同时提交固定数量 N 的请求；
- 比较两组的总 Decode Throughput、E2E Throughput 和峰值活跃 KV Block；
- 用于回答“相同并发下，压缩带来多少容量收益和多少执行开销”。

#### 子测试 B：稳态/分批到达并发容量

- 先加入一批请求；
- 等待它们发生真实压缩并释放 Block；
- 再逐批加入新请求；
- 直到出现抢占、容量失败或状态槽不足；
- 记录最大同时活跃请求数。

该测试才适合证明：

> 压缩释放的物理 Block 被后续请求复用，从而提高稳态并发容量。

若只把所有长 Prompt 一次性提交，不应宣称压缩提高了“最大可接入并发”。

### 6. 三项主要指标

#### 指标一：吞吐量变化

核心指标：

```text
System Decode Throughput
= 所有请求生成 Token 总数 / Decode 墙钟时间
```

辅助指标：

```text
E2E Output Throughput
Request Throughput
每请求 TPOT P50/P95
压缩步骤 P50/P95
```

压缩步骤本身会回退 Eager，因此低并发场景下吞吐可能下降；高并发和容量压力下，释放 Block 后才可能出现整体收益。无论结果正负都应如实报告。

#### 指标二：并发收益

推荐定义为“零抢占最大稳态并发”：

```text
N0 = 所有请求完成，且 preemption_count = 0 时的最大同时活跃请求数
```

并发提升率：

```text
Concurrency Gain
= (N_compression - N_baseline) / N_baseline × 100%
```

若当前代码还没有 `preemption_count`，需要在 `Scheduler.preempt()` 中增加统计，同时记录：

- `preemption_count`；
- `re_prefill_count`；
- `re_prefill_tokens`；
- `peak_used_state_slots`；
- 容量失败原因。

#### 指标三：KV Cache 收益

主指标：

```text
Peak Active KV Block Reduction
= 1 - peak_blocks_compression / peak_blocks_baseline
```

同时记录：

- `physical_kv_tokens_max`；
- `physical_kv_blocks_max`；
- `peak_used_kv_blocks`；
- `released_kv_blocks`；
- `compression_event_count`；
- `physical_kv_tokens_final_observed`；
- `num_kvcache_blocks`，确认两组 Block Pool 容量一致。

不要把 `torch.cuda.memory_reserved()` 或 `nvidia-smi` 的变化作为 KV 压缩主结果，因为当前大 KV Tensor 是预分配的。

### 7. 必须保留的质量约束

对每个成对请求记录：

- Baseline 输出 Token；
- Compression 输出 Token；
- `exact_token_match`；
- `common_prefix_tokens`；
- 首个不一致位置。

压缩算法允许输出发生变化，因此“完全一致”不是通过条件，但必须公开变化程度。若不做 LongBench/Needle，不得在简历中写精度下降 Δ。

### 8. 当前 Benchmark 需要增加的功能

基于 `bench_kv_compression.py` 增加：

- `--num-requests`；
- `--max-num-seqs`；
- JSONL Prompt 输入；
- 固定并发和分批到达两种模式；
- 每请求 TTFT/TPOT；
- 全局 Decode Throughput；
- Scheduler 抢占和重算统计；
- GDN State Slot 使用峰值；
- 实际 `num_kvcache_blocks`；
- 容量失败类型；
- P50/P95 汇总。

必须保证 Baseline 和 Compression：

- 使用相同 Prompt 顺序；
- 使用相同输出长度；
- 使用相同 Block Pool；
- 使用相同 `max_num_seqs`；
- 使用相同 Graph Bucket；
- 除压缩配置外没有其他差异。

### 9. 最终结果表

#### 固定并发

| Context | Concurrency | 模式 | Decode tok/s | TPOT P50 | Peak Active Blocks | Released Blocks | Compression Events |
|---:|---:|---|---:|---:|---:|---:|---:|
| 16K | 4 | Baseline |  |  |  | 0 | 0 |
| 16K | 4 | Compression |  |  |  |  |  |
| 16K | 8 | Baseline |  |  |  | 0 | 0 |
| 16K | 8 | Compression |  |  |  |  |  |
| 32K | 4 | Baseline |  |  |  | 0 | 0 |
| 32K | 4 | Compression |  |  |  |  |  |

#### 最大并发

| Context | Baseline 零抢占并发 | Compression 零抢占并发 | 并发提升 | KV Block 峰值降低 | Throughput 变化 |
|---:|---:|---:|---:|---:|---:|
| 16K |  |  |  |  |  |
| 32K |  |  |  |  |  |

### 10. 可写入简历的结论边界

可以写：

> 在 Qwen3.5-27B、4 × RTX 3090、TP=4、16K Prompt 和固定 2048 Token 输出下，动态压缩使活跃 KV Block 峰值降低 X%，零抢占稳态并发由 N 提升至 M，系统 Decode Throughput 变化 Y%。

不能写：

- “GPU 物理显存降低 X%”，除非真实测量并解释 allocator；
- “Prefill 显存降低”；
- “所有 32K 请求可因压缩直接同时接入”；
- “精度下降控制在 Δ”，除非完成正式质量评测。

---

## 七、实验三：MTP Draft 接受率

### 1. 实验目的

评估当前 Qwen3.6 MTP 原型中，Draft Token 被目标模型 Verify 接受的比例，并验证 MTP 输出最终与普通贪心解码保持一致。

本实验只证明：

- MTP 权重能够加载；
- Draft/Verify、Accept/Reject 和状态回滚逻辑可运行；
- 在给定 Prompt 分布下的 Draft 接受率。

本实验不直接证明推理已经获得加速。接受率高不等于最终吞吐一定更高，仍需考虑 Verify、回滚和额外 MTP Forward 开销。

### 2. 模型与开关

```text
Model = Qwen3.6-27B-FP8
4 × RTX 3090
TP = 4
enable_mtp = True
kv_compress_enabled = False
enable_vision = False
verify_mode = graph
temperature = 0.0
```

当前项目会将 FP8 checkpoint 分片后反量化为 BF16 常驻权重，不是原生 FP8 GEMM。因此 FP8 文件较小不代表运行时权重显存也按 FP8 大小计算。

### 3. Prompt 集

单条 Prompt 接受率没有代表性。至少准备 100 条 Prompt，建议分成四类：

| 类型 | 数量 | 示例范围 |
|---|---:|---|
| 中文通用问答 | 25 | 解释、总结、常识问答 |
| 英文通用问答 | 25 | explanation、summary |
| 代码 | 25 | Python/C++ 小函数、代码补全 |
| 数学与逻辑 | 25 | 短推理、计算、逻辑判断 |

每条 Prompt 固定生成 128 Token，并使用同一 chat template。若时间有限，最低使用 40 条 Prompt，但不能只展示一条 Prompt 的接受率。

### 4. 核心实验设置

主结果使用：

```text
Draft Length = 1
```

这样接受率定义最清楚，直接对应 MTP-1：

```text
MTP-1 Accept Rate
= accepted_draft_tokens / compared_draft_tokens
```

补充结果可扫描：

```text
Draft Length = 1 / 2 / 3 / 4
```

当 Draft Length 大于 1 时，仅报告 Token 接受率不够，还应报告：

- 平均接受前缀长度；
- 整段 Draft 全接受率；
- Reject Round 数；
- Reject 后重算次数。

### 5. 正确性通过条件

每条 Prompt 均运行：

- 普通 Greedy Graph Baseline；
- MTP Speculative Decode。

通过条件：

```text
greedy_output_token_ids == mtp_output_token_ids
```

若最终输出不一致，该 Prompt 的接受率结果不得计入正式统计，必须先定位状态回滚、KV Slot、GDN State 或 Commit 边界问题。

### 6. 主要指标

核心指标：

```text
Micro Accept Rate
= 全部 Prompt 接受的 Draft Token 总数
  / 全部 Prompt 实际比较的 Draft Token 总数
```

同时报告：

```text
Macro Accept Rate
= 各 Prompt Accept Rate 的算术平均
```

Micro 结果作为简历主数字，Macro 用于避免少数长样本完全主导结果。

补充指标：

- 各任务类别接受率；
- P50/P95 Prompt 接受率；
- `target_forwards_per_token`；
- `mtp_forwards_per_token`；
- `reject_reruns`；
- `greedy_match_rate`，正式结果要求 100%。

### 7. 当前脚本需要增加的功能

基于 `run_mtp_fast_decode.py` 和 `bench_mtp_draft_sweep.py`：

- 支持 JSONL Prompt 集；
- 模型只加载一次，在同一实例中依次处理多条 Prompt；
- 每条请求结束后彻底释放 KV Block 和 GDN State Slot；
- 累计 accepted、compared、rejected；
- 输出 Micro/Macro 接受率；
- 保存各 Prompt 和各类别结果；
- 保留 Greedy 对齐检查；
- 将模型初始化和 Graph Capture 排除在计时外。

### 8. 最终结果表

| Prompt 类别 | Prompt 数 | Compared Draft Tokens | Accepted Draft Tokens | Micro Accept Rate | Greedy Match Rate |
|---|---:|---:|---:|---:|---:|
| 中文问答 | 25 |  |  |  | 100% |
| 英文问答 | 25 |  |  |  | 100% |
| 代码 | 25 |  |  |  | 100% |
| 数学逻辑 | 25 |  |  |  | 100% |
| 总计 | 100 |  |  |  | 100% |

可选 Draft Length Sweep：

| Draft Length | Token Accept Rate | 平均接受前缀 | 整段全接受率 | Reject Reruns | Greedy Match |
|---:|---:|---:|---:|---:|---:|
| 1 |  |  |  |  | 100% |
| 2 |  |  |  |  | 100% |
| 3 |  |  |  |  | 100% |
| 4 |  |  |  |  | 100% |

### 9. 可写入简历的结论边界

可以写：

> 在 Qwen3.6-27B-FP8、4 × RTX 3090、TP=4 下完成 MTP-1 Draft/Verify、Accept/Reject 和状态回滚验证；在 100 条多类型 Prompt、固定 128 Token 贪心生成中，Draft Token Micro 接受率为 X%，最终输出与 Greedy Baseline 的 Token 一致率为 100%。

不能仅凭接受率写：

- “MTP 将吞吐提升 X%”；
- “实现了生产级投机解码”；
- “Qwen3.5-27B 的 MTP 接受率为 X%”，除非确实使用该模型完成验证。

---

## 八、建议执行顺序

### 阶段 0：硬件和模型 Smoke

1. Qwen3.5-27B，TP=4，Eager，短 Prompt；
2. Qwen3.5-27B，TP=4，CUDA Graph；
3. 记录各卡权重后显存、KV Block 数、State Slot 数；
4. 失败则先处理 OOM、TP 分片和 Graph Capture，暂不开始正式 Benchmark。

### 阶段 1：实验一

1. Eager/Graph 输出一致性；
2. 1K/4K/16K 长度曲线；
3. 完成 TTFT、TPOT、Decode Throughput 结果。

### 阶段 2：实验二

1. 单请求 Smoke，确认真实发生压缩；
2. 固定并发对照；
3. 分批到达并发阶梯；
4. 检查输出变化和压缩事件；
5. 生成吞吐、并发和 KV Block 三项最终结果。

### 阶段 3：实验三

1. `test_mtp_forward.py` 验证 MTP 权重；
2. `test_state_rollback.py` 验证状态回滚；
3. `run_mtp_fast_decode.py --compare-greedy` 验证单 Prompt；
4. 多 Prompt MTP-1 接受率；
5. 时间允许再做 Draft Length Sweep。

---

## 九、最终三条实验结论模板

### 实验一

> 在 Qwen3.5-27B、4 × RTX 3090、TP=4 下，Eager 与 CUDA Graph 在固定贪心解码测试集上输出 Token 完全一致；CUDA Graph 将 4K 输入、256 Token 输出场景的 TPOT P50 由 A ms 降至 B ms，Decode Throughput 由 C tok/s 提升至 D tok/s。

### 实验二

> 在 Qwen3.5-27B、4 × RTX 3090、TP=4、16K Prompt 和 2048 Token 输出下，动态 KV 压缩使峰值活跃 KV Block 由 A 降至 B，降低 X%；零抢占稳态并发由 N 提升至 M，系统 Decode Throughput 变化 Y%。

### 实验三

> 在 Qwen3.6-27B-FP8、4 × RTX 3090、TP=4 下，完成 MTP-1 Draft/Verify、Accept/Reject 和状态回滚验证；在 100 条多类型 Prompt 上 Draft Token Micro 接受率为 X%，最终输出与 Greedy Baseline 的 Token 一致率为 100%。

---

## 十、最终边界总结

这三组实验能够分别证明：

| 实验 | 能证明 | 不能证明 |
|---|---|---|
| Qwen3.5 CUDA Graph | Graph 路径正确、Decode 效率变化 | 与官方实现绝对数值等价、Graph 优化 Prefill |
| KV 压缩 | 活跃 KV Block 减少、稳态并发和吞吐变化 | 预分配显存一定下降、精度无损、Prefill 容量改善 |
| MTP 接受率 | Draft 预测命中率和状态控制正确 | 必然获得推理加速、生产级投机解码 |

按上述边界执行后，三组实验与简历中的三个主要技术点能够一一对应，结论可归因、可复现，也更容易经受 AI Infra 面试中的追问。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nanovllm项目面试准备|模块-nanovllm项目面试准备]]

%% 项目关联导航：结束 %%
