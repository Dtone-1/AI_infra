# nano-vLLM-qwen3.6 根目录文件作用与源码改造分析

> 分析对象：截图中的 `nano-vllm-qwen3.6` 项目根目录文件：  
> `bench.py`、`bench_mtp_draft_sweep.py`、`bench_qwen35_fixed.py`、`LICENSE`、`pyproject.toml`、`README.md`、`run_mtp_fast_decode.py`、`run_text_qwen35_v2.py`、`run_text_qwen36_fp8.py`、`test_mtp_forward.py`、`test_mtp_spec_decode.py`、`test_mtp1_spec_decode.py`、`test_mtp1_verify.py`、`test_state_rollback.py`。
>
> 分析目标：理解这些根目录脚本和配置文件分别做什么；相比原版 `GeeeekExplorer/nano-vllm` 做了哪些新增或改造；这些改造在 Qwen3.5 / Qwen3.6 / hybrid / FP8 / MTP / speculative decoding / state rollback 推理系统中发挥什么作用。

---

## 1. 总体结论

截图中的文件大部分不是 `nanovllm/` 包内部的核心模型层源码，而是**项目根目录的运行脚本、测试脚本、benchmark 脚本和项目配置文件**。

原版 `nano-vLLM` 的定位是一个轻量 vLLM 实现，README 中强调的是“fast offline inference”“readable codebase”和 prefix caching、tensor parallel、torch compilation、CUDA graph 等基础优化，并且主要示例围绕 Qwen3-0.6B、`example.py` 和 `bench.py` 展开。fileciteturn54file0L11-L20 fileciteturn54file0L36-L50

`nano-vllm-qwen3.6` 的 README 则明确说明：它是在 `nano-vllm` 基础上扩展，用于 Qwen3.5 hybrid 模型和 Qwen3.6 FP8 text-only inference 实验；学习重点包括 tensor parallel、KV cache allocation、CUDA Graph decode、hybrid linear-attention state 和 quantized checkpoint loading。fileciteturn53file0L3-L10

所以这个项目相比原版的根本变化是：

```text
原版 nano-vLLM:
    轻量文本推理框架
    主要支持 Qwen3 dense
    主要演示普通 generate / benchmark

nano-vllm-qwen3.6:
    在原版基础上增加 Qwen3.5/Qwen3.6 实验能力
    支持 hybrid 模型、GatedDeltaNet state、FP8 checkpoint 加载
    增加 MTP / speculative decoding / rollback correctness 测试
    增加多组 run、bench、test 脚本用于验证新功能
```

---

## 2. 根目录文件分类

这些文件可以分成 5 类：

| 类别 | 文件 | 作用 |
|---|---|---|
| 原版保留 baseline | `bench.py` | 原版 benchmark，作为普通 Qwen3 dense 推理吞吐基线 |
| 项目配置与说明 | `LICENSE`、`pyproject.toml`、`README.md` | 项目元信息、依赖、文档、运行说明 |
| Qwen3.5 / Qwen3.6 运行脚本 | `run_text_qwen35_v2.py`、`run_text_qwen36_fp8.py` | 分别验证 Qwen3.5 BF16 text-only 和 Qwen3.6 FP8 text-only 路径 |
| benchmark 脚本 | `bench_qwen35_fixed.py`、`bench_mtp_draft_sweep.py` | 测 Qwen3.5 固定场景性能，以及 MTP draft_len sweep 性能 |
| MTP / speculative decoding / rollback 测试脚本 | `run_mtp_fast_decode.py`、`test_mtp_forward.py`、`test_mtp1_verify.py`、`test_mtp1_spec_decode.py`、`test_mtp_spec_decode.py`、`test_state_rollback.py` | 验证 MTP forward、draft/verify、accept/reject、状态回滚和 fast decode 路径 |

---

## 3. 和原版 nano-vLLM 的根目录差异

### 3.1 `bench.py` 基本保持原版

`bench.py` 在原版和 qwen3.6 仓库中的文件 SHA 相同，说明这个文件基本没有改动：原版 `bench.py` 的 SHA 是 `8e61d654...`，qwen3.6 仓库中的 `bench.py` 也是同一个 SHA。fileciteturn50file0L1-L3 fileciteturn34file0L1-L3

它的功能是随机生成 256 个请求，每个请求输入长度随机在 100 到 1024 token 之间，输出长度随机在 100 到 1024 token 之间，然后统计总输出 token 数、总耗时和 tok/s 吞吐。fileciteturn34file0L10-L30

所以 `bench.py` 在 qwen3.6 项目中的意义不是新增能力，而是：

```text
保留原版 nano-vLLM 的普通文本推理 benchmark，
作为后续 Qwen3.5/Qwen3.6、MTP、FP8 实验的 baseline。
```

---

### 3.2 `pyproject.toml` 从普通 nano-vLLM 描述改成 Qwen3.6 实验描述

原版 `pyproject.toml` 中，项目描述是：

```text
a lightweight vLLM implementation built from scratch
```

主页指向 `GeeeekExplorer/nano-vllm`。fileciteturn51file0L7-L25

qwen3.6 版本中，项目描述改成：

```text
A compact nano-vLLM fork for Qwen3.5 and Qwen3.6 FP8 inference experiments
```

并且额外添加了 `Original="https://github.com/GeeeekExplorer/nano-vllm"`，说明这是原版 nano-vLLM 的 fork/扩展版本。fileciteturn52file0L7-L27

这说明该项目定位从“通用轻量 vLLM 实现”变成：

```text
针对 Qwen3.5/Qwen3.6、FP8、hybrid 推理实验的 nano-vLLM fork。
```

---

### 3.3 `README.md` 从原版轻量说明扩展成 Qwen3.5/Qwen3.6 实验手册

原版 README 只简要说明安装、下载 Qwen3-0.6B、Quick Start 和 `bench.py` benchmark。fileciteturn54file0L21-L50

qwen3.6 README 明确列出了新增能力：

```text
Qwen3.5-9B BF16 text inference
Qwen3.5-9B multimodal smoke test
Qwen3.6-27B-FP8 text-only inference
Tensor parallelism for GatedDeltaNet
CUDA Graph decode
Rank-local FP8 checkpoint loading
Qwen3.6 MTP weight loading / single-step forward / MTP-1 draft-verify prototype
```

这些内容在 README 的 What Works 中明确列出。fileciteturn53file0L12-L23

同时 README 也明确说明当前限制：Qwen3.6 FP8 是加载后反量化成 BF16 权重，并没有实现 native FP8 matmul；Qwen3.6 当前是 text-only；MTP 当前只是权重加载、one-step draft-token probing 和 MTP-1 accept-rate measurement 的 prototype，不提供 decode speedup。fileciteturn53file0L25-L36

这点非常重要，因为它告诉我们：

```text
项目并不是完整生产级 serving stack；
更像是为了学习和验证 Qwen3.6 推理机制的实验型工程。
```

---

## 4. 逐文件作用与改造意义

---

## 4.1 `bench.py`

### 作用

`bench.py` 是原版保留的基础 benchmark 脚本。它随机生成大量 token ids 作为输入，随机设置每个请求的 max_tokens，然后调用 `llm.generate()` 统计吞吐。核心逻辑包括：

```text
num_seqs = 256
max_input_len = 1024
max_output_len = 1024
随机 prompt_token_ids
随机 SamplingParams
统计 throughput = total_tokens / time
```

源码中可以看到它默认加载 `~/huggingface/Qwen3-0.6B/`，创建 `LLM(path, enforce_eager=False, max_model_len=4096)`，并计算总输出 token 数和 tok/s。fileciteturn34file0L10-L30

### 相比原版的变化

基本无变化。它和原版 `bench.py` SHA 相同。fileciteturn50file0L1-L3 fileciteturn34file0L1-L3

### 意义

在 qwen3.6 项目中，它作为普通 Qwen3 dense 路径的 baseline：

```text
验证原版功能是否仍然可用
提供基础吞吐参考
方便和 Qwen3.5/Qwen3.6/MTP benchmark 对比
```

---

## 4.2 `bench_qwen35_fixed.py`

### 作用

这是 qwen3.6 新增的 Qwen3.5 固定场景 benchmark。

它默认加载：

```text
~/huggingface/Qwen3.5-9B
```

并支持：

```text
--tp
--max-tokens
--repeats
--temperature
--eager
```

源码中可以看到它使用 `AutoTokenizer.apply_chat_template()` 构造中文 prompt，并创建 `LLM`，配置 `tensor_parallel_size`、`max_model_len`、`max_num_batched_tokens`、`max_num_seqs` 和 `gpu_memory_utilization`。fileciteturn36file0L17-L44

更关键的是，它不用一次性 `llm.generate()`，而是手动：

```text
llm.add_request()
while not llm.is_finished():
    llm.step()
```

并根据 `num_tokens` 的正负区分 prefill 和 decode，分别统计 prefill tok/s、decode tok/s 和 total out_tok/s。fileciteturn36file0L50-L77

### 相比原版的变化

原版 `bench.py` 只统计总体 throughput，不区分 prefill/decode。`bench_qwen35_fixed.py` 则专门拆分：

```text
prefill_time / prefill_tokens
decode_time / decode_tokens
```

### 意义

Qwen3.5 hybrid 推理中，prefill 和 decode 的瓶颈不同：

```text
prefill:
    长 prompt 并行计算多，attention/GDN 状态初始化重要

decode:
    逐 token 推理，KV Cache / GDN state / CUDA Graph 影响更明显
```

因此这个脚本的意义是：

```text
用固定 prompt、固定参数重复跑 Qwen3.5，
分别观察 prefill 和 decode 性能。
```

这比原版总吞吐 benchmark 更适合分析 hybrid 模型推理性能。

---

## 4.3 `bench_mtp_draft_sweep.py`

### 作用

这是 MTP draft length sweep benchmark。

它从 `run_mtp_fast_decode.py` 中导入：

```text
build_prompt
run_greedy_decode
run_mtp_fast_decode
summarize_stats
```

说明它不是单独实现 speculative decoding，而是复用 fast decode 脚本的函数，在不同 `draft_len` 下重复跑实验。fileciteturn35file0L3-L13

它的参数包括：

```text
--model Qwen3.6-27B-FP8
--devices
--tp
--max-tokens
--draft-lens
--verify-mode
--skip-greedy-compare
```

默认 `draft-lens="1,2,3,4"`，默认 verify mode 是 `graph`。fileciteturn35file0L16-L29

输出字段包括：

```text
draft_len
match
tokens
tok_s
seconds
accept_rate
target_fw_per_tok
mtp_fw_per_tok
verify_graph_replays
verify_eager_calls
verify_chunk_calls
reject_reruns
```

源码中有明确的表头和逐 draft_len 打印逻辑。fileciteturn35file0L65-L88

### 相比原版的变化

原版没有 MTP，也没有 speculative decoding，因此不会有：

```text
draft_len
accept_rate
verify_graph_replays
reject_reruns
target_forwards_per_token
mtp_forwards_per_token
```

这些指标。

### 意义

它的作用是回答一个工程问题：

```text
MTP draft_len 取多少比较合适？
```

draft_len 太小：

```text
可接受 token 数少，潜在加速有限
```

draft_len 太大：

```text
draft 错误概率上升
verify 成本上升
reject/rerun 成本上升
```

所以这个脚本用于评估：

```text
不同 draft_len 下的接受率、验证开销、重跑开销和有效 token/s。
```

---

## 4.4 `run_text_qwen35_v2.py`

### 作用

这是 Qwen3.5 文本-only smoke test 脚本。

它默认模型路径是：

```text
~/huggingface/Qwen3.5-9B
```

支持设置：

```text
--devices
--tp
--prompt
--max-model-len
--max-batched-tokens
--max-tokens
--temperature
--gpu-memory-utilization
--eager
```

源码中可以看到它使用 tokenizer 的 `apply_chat_template()`，并设置 `enable_thinking=False`，然后创建 `LLM` 进行普通 `llm.generate()`。fileciteturn38file0L7-L19 fileciteturn38file0L34-L69

### 相比原版的变化

原版 Quick Start 主要是：

```text
LLM("/YOUR/MODEL/PATH")
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(...)
```

更偏普通 Qwen3 dense 测试。fileciteturn54file0L36-L46

`run_text_qwen35_v2.py` 则专门验证：

```text
Qwen3.5-9B
Chat template
tensor parallel
hybrid 模型路径
CUDA Graph/eager 切换
```

### 意义

它是 Qwen3.5 接入后的最小文本推理验证脚本：

```text
先不测 MTP
先不测多模态
先验证 Qwen3.5 hybrid text-only 能否跑通
```

---

## 4.5 `run_text_qwen36_fp8.py`

### 作用

这是 Qwen3.6 FP8 text-only smoke test。

它默认模型路径是：

```text
~/huggingface/Qwen3.6-27B-FP8
```

默认设备是：

```text
0,1,2,3
```

默认 `tp=4`。fileciteturn39file0L7-L18

创建 LLM 时显式设置：

```python
enable_vision=False
tensor_parallel_size=args.tp
max_num_batched_tokens=args.max_batched_tokens
gpu_memory_utilization=args.gpu_memory_utilization
```

这说明该脚本验证的是 Qwen3.6-27B-FP8 的 text-only 路径，而不是视觉多模态路径。fileciteturn39file0L53-L67

### 相比原版的变化

原版没有 FP8 checkpoint 加载脚本，也没有 Qwen3.6-27B-FP8 默认配置。

qwen3.6 README 也明确说明：Qwen3.6-27B-FP8 当前 target 是 text-only，需要 `enable_vision=False`；并且 FP8 checkpoint 会在启动时反量化为 BF16 常驻权重，而不是 native FP8 matmul。fileciteturn53file0L25-L33

### 意义

这个脚本验证的是 FP8 加载链路：

```text
Qwen3.6 FP8 safetensors
  ↓
loader.py 读取 weight_scale_inv
  ↓
quant.py 做 block-wise 反量化
  ↓
linear/embed/lm_head 加载 BF16 参数
  ↓
text-only generate
```

它是整个 FP8 权重加载改造是否正确的 smoke test。

---

## 4.6 `test_mtp_forward.py`

### 作用

这是 Qwen3.6 MTP 单步 forward smoke test。

它默认启用：

```python
enable_mtp=True
enforce_eager=True
```

并且 `enable_vision=False`，说明它专门测 text-only MTP。fileciteturn40file0L53-L63

核心流程是：

```text
构造 prompt
llm.add_request()
scheduler.schedule() 得到 prefill batch
调用 model_runner.run_mtp_probe
打印 main token、draft token、top-k、hidden shape、logits shape、MTP 权重加载情况
```

源码中调用了：

```python
result = llm.model_runner.call("run_mtp_probe", seqs, args.top_k)
```

并打印 `main_token_ids`、`draft_token_ids`、`draft_topk`、`main_hidden_shape`、`mtp_hidden_shape`、`draft_logits_shape`、`mtp_loaded_count`、`mtp_skipped_count`。fileciteturn40file0L73-L103

### 相比原版的变化

原版没有 MTP 模块，也没有 `run_mtp_probe`。

qwen3.6 中新增的 `Qwen3MTP` 明确是 multi-token prediction head prototype，并且源码注释说明它只提供 weight loading 和 single-step forward，还没有完整接入 speculative decoding。fileciteturn58file0L37-L42

### 意义

这个脚本验证的是：

```text
MTP 权重能否加载
MTP 单步 forward 能否跑通
draft logits/token 是否能生成
MTP hidden shape 是否正确
```

它是后续 speculative decoding 之前的最小正确性验证。

---

## 4.7 `test_mtp1_verify.py`

### 作用

这是 MTP-1 draft/verify prototype。

它每轮先调用：

```python
run_mtp_draft_step
```

得到主模型 token 和 MTP draft token；然后再正常 schedule 一次 decode，调用普通 `model_runner.run` 得到 target verify token；最后比较：

```text
draft_token_ids == verify_token_ids
```

统计 accepted、rejected、accept_rate。fileciteturn46file0L65-L111

### 相比原版的变化

原版只有普通自回归 decode：

```text
模型 forward -> sampler -> append token
```

这个脚本增加了：

```text
MTP draft token
主模型 verify token
accept/reject 统计
```

### 意义

它是最简单的 MTP 验证：

```text
draft_len = 1
不做复杂 batch verify
不做完整 rollback spec decode
只测 MTP 的下一个 draft token 和主模型下一个 token 是否一致
```

适合用来测 MTP 的基础 accept rate。

---

## 4.8 `test_mtp1_spec_decode.py`

### 作用

这是 MTP-1 speculative decode prototype。

相比 `test_mtp1_verify.py`，它不只是统计 accept rate，还加入了状态保存和回滚：

```text
save_decode_state
snapshot_scheduler
如果 accept:
    直接 postprocess draft token
如果 reject:
    restore_scheduler
    restore_decode_state
    rerun trusted step
    commit target token
drop_decode_state
```

源码中可以看到它在 verify 前保存 scheduler snapshot 和 decode state，reject 时恢复 scheduler 和 decode state，再 rerun。fileciteturn45file0L95-L140

同时它支持：

```text
--force-reject-attempt
```

用来强制走 reject 分支，验证回滚是否正确。fileciteturn45file0L9-L22

### 相比原版的变化

原版 decode 没有 “接受/拒绝” 和 “状态回滚” 的概念。

MTP/spec decode 必须解决：

```text
draft token 接受时如何提交
draft token 拒绝时如何回滚 KV Cache / GDN state / scheduler 状态
```

### 意义

这个脚本是从 MTP probe 走向 speculative decoding 的关键中间态：

```text
只验证 1 个 draft token
但完整测试 accept/reject/state rollback
```

这对 hybrid 模型尤其重要，因为状态不只是 KV Cache，还包括 GatedDeltaNet recurrent/conv state。

---

## 4.9 `test_mtp_spec_decode.py`

### 作用

这是多 token MTP speculative decode prototype。

它支持：

```text
--draft-len
--verify-mode eager/graph/chunk
--force-reject-step
--debug-mismatch
--skip-greedy-compare
```

源码中默认 `draft_len=4`，并支持 graph/chunk/eager 三种 verify 模式。fileciteturn41file0L13-L31

它的核心流程是：

```text
1. run_greedy 得到 baseline
2. run_speculative 执行 MTP draft
3. 保存 scheduler snapshot 和 decode state
4. batch verify draft tokens
5. 比较 draft 和 target
6. accept 时批量 commit
7. reject 时 restore + rerun trusted path
8. 统计 accept_rate、verify_batch_tokens、reject_reruns、target_forwards_per_token 等
9. 最后和 greedy 输出对齐检查
```

源码中可以看到它在 speculative decode 中保存 snapshot 和 decode state，并调用 `run_verify_auto_probe` 执行 verify。fileciteturn41file0L306-L329

在 reject 分支中，它会 `restore_scheduler`、`restore_decode_state`，然后 rerun trusted verify，并 assert rerun token 与 target prefix 一致。fileciteturn42file0L57-L108

最后它会打印大量 spec stats，包括 accept rate、verify graph replay 次数、chunk mismatch、target forwards、mtp forwards、reject reruns、model call seconds 等。fileciteturn43file0L14-L79

### 相比原版的变化

原版没有 speculative decoding，更没有：

```text
draft batch verify
accept length
reject rerun
greedy alignment
verify graph replay
chunk verify semantic comparison
```

这些都是 qwen3.6 项目为了 MTP/spec decode 新增的 correctness + performance 观测能力。

### 意义

这个文件是整个 MTP speculative decoding 实验链路中最完整的测试脚本：

```text
既测试正确性：
    greedy_match
    mismatch debug
    rerun token assert
    chunk logits compare

又测试性能相关指标：
    accept_rate
    target_forwards_per_token
    mtp_forwards_per_token
    verify_call_seconds
    reject_reruns
```

它还没有等价于成熟生产级 speculative decoding，但已经把工程上最难的几个点显式暴露出来：

```text
状态保存
状态回滚
批量 verify
accept/reject
greedy 对齐
统计开销
```

---

## 4.10 `run_mtp_fast_decode.py`

### 作用

这是 MTP speculative decode 的 fast path benchmark/run 脚本。

它从 `test_state_rollback.py` 引入 scheduler snapshot/restore，从 `test_mtp_spec_decode.py` 引入 prompt、decode、block capacity、manual commit、trusted_verify_mode 等工具。fileciteturn37file0L8-L18

相比 `test_mtp_spec_decode.py`，它更偏性能路径，去掉了很多 top-k/logit-diff probe，直接调用：

```python
run_mtp_draft_fast_step
run_verify_batch_fast
```

源码中可以看到它在 MTP fast decode 中：

```text
主模型 + MTP 生成 main token 和 draft token
保存 scheduler snapshot
保存 decode state range
run_verify_batch_fast
根据 draft/target 对比决定 accept/reject
reject 时 restore + rerun
最后 drop_decode_state
```

核心逻辑集中在 `run_mtp_fast_decode()`。fileciteturn37file0L102-L167 fileciteturn37file0L169-L257

它还定义了 `summarize_stats()`，统计：

```text
decode_tok_s
accept_rate
target_forwards_per_token
mtp_forwards_per_token
verify_graph_replays
verify_eager_calls
verify_chunk_calls
reject_reruns
```

fileciteturn37file0L270-L287

### 相比原版的变化

原版没有 fast decode 的概念；只有普通 generate。

这个脚本的变化是：

```text
从 correctness probe 走向性能路径
减少调试信息
保留接受率和 forward/token 等关键指标
```

### 意义

它是 benchmark MTP/spec decode 是否有实际加速潜力的关键脚本。

如果你想在简历项目中展示 MTP 优化，最重要的指标可能来自这里：

```text
baseline greedy decode tok/s
MTP fast decode tok/s
accept_rate
target_forwards_per_token
mtp_forwards_per_token
reject_reruns
```

---

## 4.11 `test_state_rollback.py`

### 作用

这是 decode-state rollback smoke test。

它定义了两个非常关键的工具函数：

```text
snapshot_scheduler
restore_scheduler
```

`snapshot_scheduler()` 保存：

```text
Sequence 状态
waiting/running 队列
free_block_ids
used_block_ids
hash_to_block_id
每个 block 的 ref_count/hash/token_ids
free_state_slots
```

源码中可以看到它明确保存了 `state_slot_id` 和 `free_state_slots`，这说明它不仅关心 KV Cache blocks，也关心 hybrid GDN state slots。fileciteturn47file0L24-L56

`restore_scheduler()` 则把这些状态恢复回去，包括 sequence、queue、block manager 和 state slot manager。fileciteturn47file0L59-L87

测试流程是：

```text
prefill 一步
进入 decode
保存 scheduler snapshot
save_decode_state
run_step_probe 得到 first token/logits
restore_scheduler
restore_decode_state
再 run_step_probe 得到 second token/logits
比较 token 是否一致、logit diff 是否在 tolerance 内
```

源码中可以看到 save/restore 和 token/logit 对比过程。fileciteturn47file0L127-L165

最后打印：

```text
kv_slots
state_slot_ids
gdn_layers
first/second token
max_logit_diff
rollback_ok
```

fileciteturn47file0L167-L184

### 相比原版的变化

原版 nano-vLLM 的普通 decode 不需要 rollback。

但是 MTP/spec decode 必须支持：

```text
先尝试写入或模拟若干 token
如果 draft 被拒绝，就恢复到 verify 前状态
```

对 hybrid 模型来说，回滚对象包括：

```text
KV Cache
Sequence token_ids
block_table
prefix hash
scheduler queue
GDN state slot
GDN recurrent/conv state
```

### 意义

这是 speculative decoding 正确性的基础设施测试。

没有它，MTP reject 分支很容易把模型状态污染，导致后续 token 全部错误。

---

## 4.12 `LICENSE`

### 作用

`LICENSE` 是开源协议文件，不参与推理流程。

从 `pyproject.toml` 可以看到项目 license 是 MIT，并且使用 `license-files = ["LICENSE"]`。原版和 qwen3.6 版本都保留了这个设置。fileciteturn51file0L7-L13 fileciteturn52file0L7-L13

### 相比原版的变化

功能上没有推理意义变化。

### 意义

表示项目仍然继承原版开源协议体系，方便学习、fork、实验和二次开发。

---

## 4.13 `README.md`

### 作用

`README.md` 是项目说明文件。qwen3.6 版本的 README 不再只是简单 quick start，而是系统性说明：

```text
支持什么
限制是什么
仓库结构
模型下载
Qwen3.5/Qwen3.6 运行方式
MTP forward / verify / spec decode / fast decode
benchmark
development checks
```

README 中的 Repository Layout 明确列出了 `nanovllm/` 内部模块和根目录脚本。fileciteturn53file0L38-L55

Quick Start 中也给出了 Qwen3.5、Qwen3.6 FP8、MTP forward、MTP-1 verify、MTP spec decode、fast decode、draft sweep 和 rollback test 的运行命令。fileciteturn53file0L88-L208

### 相比原版的变化

原版 README 主要说明 Qwen3-0.6B 下载、example.py 和 bench.py。fileciteturn54file0L27-L50

qwen3.6 README 变成了：

```text
Qwen3.5/Qwen3.6 实验说明书
MTP/spec decode 操作手册
FP8 限制说明
benchmark 结果记录
```

### 意义

README 是理解该 fork 的入口，它明确告诉你：

```text
这个项目不是只改了模型名；
它围绕 hybrid state、FP8 loading、MTP、CUDA Graph、rollback 做了一整套实验工程。
```

---

## 5. 这些根目录脚本背后的核心源码改造

截图里的文件大多是 run/test/bench 入口。真正支撑它们的是 `nanovllm/` 内部核心源码改造。

---

## 5.1 Qwen3.5 / Qwen3.6 hybrid 模型结构

`nanovllm/models/qwen3_5.py` 是新增核心模型文件。它引入了：

```text
GatedDeltaNet
GemmaRMSNorm
InterleavedMRoPE
VocabParallelEmbedding
ParallelLMHead
```

源码导入部分就可以看到这些组件。fileciteturn57file0L7-L13

`Qwen3_5Attention` 的 q projection 输出 2 倍维度，然后拆成 query 和 gate；attention 输出后再乘 `sigmoid(gate)` 做 output gating。fileciteturn57file0L31-L43 fileciteturn57file0L65-L90

`Qwen3_5DecoderLayer` 根据 `config.layer_types[layer_idx]` 选择 full attention 或 GatedDeltaNet。fileciteturn57file0L109-L120

这正是 hybrid 架构的核心：

```text
full_attention 层:
    使用 Attention + KV Cache

GatedDeltaNet 层:
    使用 recurrent/conv state
    不按普通 attention 层分配 KV Cache
```

---

## 5.2 多模态入口

`qwen3_5.py` 中 `Qwen3_5Model.forward()` 支持将 `image_embeds` scatter 到 image token 位置：

```python
hidden_states[image_token_mask] = image_embeds.to(hidden_states.dtype)
```

fileciteturn57file0L157-L172

`Qwen3_5ForCausalLM` 还可以在 `vision_config` 存在时创建 `Qwen3VLVisionEncoder`，forward 时用 pixel_values 和 image_grid_thw 生成 image_embeds。fileciteturn57file0L175-L214

所以多模态链路是：

```text
image_processing.py:
    图片 -> pixel_values / image_grid_thw

vision_encoder.py:
    pixel_values / image_grid_thw -> image_embeds

qwen3_5.py:
    image_embeds 替换 image token hidden_states

model_runner.py:
    prefill 阶段组织 pixel_values、image_grid_thw、image_token_mask、MRoPE positions
```

---

## 5.3 FP8 checkpoint 加载

`nanovllm/utils/loader.py` 新增 `LoadResult`、`maybe_dequant_fp8_weight`，并在加载 safetensors 时跳过 `.weight_scale_inv`，只在对应 FP8 weight 存在 scale 时把 scale 传给 weight_loader。fileciteturn59file0L13-L25 fileciteturn59file0L40-L51

同时它支持：

```text
visual_prefix
weight_prefix
packed_modules_mapping
skipped_names / loaded_names
```

具体包括视觉权重 `model.visual.* -> visual.*` 映射、语言模型权重 `model.language_model.* -> model.*` 映射，以及 packed module 加载时传入 `loaded_scale`。fileciteturn59file0L53-L71 fileciteturn59file0L73-L102

这支撑了 `run_text_qwen36_fp8.py` 和 Qwen3.6 FP8 checkpoint 的加载。

---

## 5.4 MTP 模型结构

`qwen3_mtp.py` 中 `Qwen3MTP` 被明确标注为 Qwen3.6 multi-token prediction head prototype，并且源码注释说明它只提供 weight loading 和 single-step forward，还没有完整接入 speculative decoding。fileciteturn58file0L37-L42

它的 forward 输入是：

```text
positions
hidden_states
inputs_embeds
```

先分别 norm embedding 和 hidden state，然后 concat，再用 `ReplicatedLinear(2H -> H)` 融合，经过若干 MTP decoder layer 后输出 hidden states。fileciteturn58file0L44-L70

这就是 `test_mtp_forward.py`、`test_mtp1_verify.py`、`test_mtp_spec_decode.py` 等脚本背后的模型能力。

---

## 5.5 speculative decoding 和 state rollback

MTP/spec decode 的核心难点不是 draft token 生成，而是：

```text
draft token 如果被拒绝，如何恢复模型状态？
```

`test_state_rollback.py` 显式保存并恢复：

```text
Sequence 状态
scheduler waiting/running 队列
KV block manager 状态
prefix cache hash
GDN state slot manager 空闲表
```

fileciteturn47file0L24-L87

`run_mtp_fast_decode.py` 在 verify 前保存 scheduler snapshot 和 decode state range，reject 分支中 restore scheduler 和 restore_decode_state，然后 rerun trusted verify path。fileciteturn37file0L161-L181 fileciteturn37file0L228-L257

这说明项目已经把 speculative decoding 中最容易出错的状态一致性问题显式测试起来。

---

## 6. 文件级对比总表

| 文件 | 原版 nano-vLLM 中的状态 | qwen3.6 中的作用 | 改造意义 |
|---|---|---|---|
| `bench.py` | 原版已有 | 保留基础吞吐 benchmark | 保留 baseline，验证原版 Qwen3 dense 路径仍可用 |
| `bench_qwen35_fixed.py` | 新增 | Qwen3.5 固定 prompt prefill/decode timing | 单独观察 hybrid Qwen3.5 prefill/decode 性能 |
| `bench_mtp_draft_sweep.py` | 新增 | 扫描 MTP draft_len，输出 accept_rate、tok/s、verify/reject 指标 | 评估 MTP draft 长度与性能收益/成本 |
| `run_text_qwen35_v2.py` | 新增 | Qwen3.5-9B text-only smoke test | 验证 Qwen3.5 hybrid 文本推理路径 |
| `run_text_qwen36_fp8.py` | 新增 | Qwen3.6-27B-FP8 text-only smoke test | 验证 FP8 checkpoint 加载 + TP=4 text inference |
| `test_mtp_forward.py` | 新增 | MTP 单步 forward probe | 验证 MTP 权重加载、hidden/logits shape、draft top-k |
| `test_mtp1_verify.py` | 新增 | MTP-1 draft/verify accept-rate 测试 | 最小化验证 MTP draft token 与主模型 token 是否一致 |
| `test_mtp1_spec_decode.py` | 新增 | MTP-1 accept/reject + rollback prototype | 验证 reject 后 KV/GDN/scheduler 状态能恢复 |
| `test_mtp_spec_decode.py` | 新增 | 多 token MTP speculative decode prototype | 验证 batch verify、greedy alignment、reject rerun、性能统计 |
| `run_mtp_fast_decode.py` | 新增 | MTP speculative decode fast path | 去掉大量 probe，接近性能 benchmark 路径 |
| `test_state_rollback.py` | 新增 | decode state rollback smoke test | 验证 KV Cache、GDN state slot、scheduler 状态回滚一致性 |
| `pyproject.toml` | 原版已有 | 描述改为 Qwen3.5/Qwen3.6 FP8 inference experiments | 标明项目定位从原版轻量 vLLM 变成 Qwen3.6 实验 fork |
| `README.md` | 原版已有 | 大幅扩展为 Qwen3.5/Qwen3.6/MTP/FP8 使用手册 | 给出项目能力、限制、脚本运行方法 |
| `LICENSE` | 原版已有 | 保留开源协议 | 不影响推理逻辑 |

---

## 7. 从推理流程角度串起来

这些文件和内部源码可以串成一条完整链路。

### 7.1 普通 Qwen3 dense 路径

```text
bench.py
  ↓
LLM.generate
  ↓
原版 Qwen3 dense model
  ↓
KV Cache + CUDA Graph decode
  ↓
吞吐统计
```

这是原版能力保留。

---

### 7.2 Qwen3.5 hybrid text-only 路径

```text
run_text_qwen35_v2.py / bench_qwen35_fixed.py
  ↓
LLM(model=Qwen3.5-9B)
  ↓
qwen3_5.py
  ↓
layer_types 选择 full_attention 或 GatedDeltaNet
  ↓
Attention 层使用 KV Cache
GDN 层使用 recurrent/conv state
  ↓
generate / step timing
```

---

### 7.3 Qwen3.6 FP8 text-only 路径

```text
run_text_qwen36_fp8.py
  ↓
LLM(model=Qwen3.6-27B-FP8, tp=4, enable_vision=False)
  ↓
loader.py 读取 safetensors 和 weight_scale_inv
  ↓
quant.py block-wise dequant
  ↓
linear/embed/lm_head 加载 BF16 权重
  ↓
text-only generate
```

README 明确说明 Qwen3.6 FP8 当前是加载后 BF16-resident，并不是 native FP8 matmul。fileciteturn53file0L25-L36

---

### 7.4 MTP single-step 路径

```text
test_mtp_forward.py
  ↓
enable_mtp=True
  ↓
model_runner.run_mtp_probe
  ↓
Qwen3MTP
  ↓
main token + draft token + top-k + shape 检查
```

这验证 MTP 模块本身能跑。

---

### 7.5 MTP speculative decode 路径

```text
run_mtp_fast_decode.py / test_mtp_spec_decode.py
  ↓
主模型生成 main token
  ↓
MTP 生成 draft tokens
  ↓
保存 scheduler snapshot + decode state
  ↓
batch verify
  ↓
accept:
      commit draft tokens
reject:
      restore scheduler + restore decode state
      rerun trusted verify
      commit target token
  ↓
统计 accept_rate / tok_s / target_fw_per_tok / reject_reruns
```

---

## 8. 对 AI Infra 简历项目的意义

如果你要把这个项目写到简历里，不能只写“支持 Qwen3.6”。更准确的表达应该是：

```text
基于 nano-vLLM 改造 Qwen3.5/Qwen3.6 推理链路：
新增 Qwen3.5 hybrid 模型结构，支持 full attention 与 GatedDeltaNet 混合；
改造 loader/linear/embed 权重加载链路，支持 rank-local FP8 checkpoint dequant；
接入 Qwen3.6 MTP prototype，提供 single-step draft probe、MTP-1 verify、multi-token speculative decode 原型；
实现 KV Cache、GDN state、scheduler/block manager 的 snapshot/rollback 测试；
编写 run/bench/test 脚本验证 text inference、FP8 loading、MTP accept-rate、draft_len sweep 和 rollback correctness。
```

最适合展示的指标包括：

```text
Qwen3.5 prefill tok/s
Qwen3.5 decode tok/s
Qwen3.6 FP8 text-only decode tok/s
MTP accept_rate
target_forwards_per_token
mtp_forwards_per_token
reject_reruns
greedy_match
rollback_ok
```

README 中已经给出一个本地 smoke-test 结果示例：Qwen3.6-27B-FP8 TP=4 CUDA Graph decode 约 41 tok/s，Qwen3.5-9B TP=4 CUDA Graph decode 约 98 tok/s，但 README 也说明这只是 single-request smoke tests，不是完整 serving benchmark。fileciteturn53file0L225-L244

---

## 9. 最终总结

截图中的这些根目录文件，整体上体现了 `nano-vllm-qwen3.6` 相比原版 `nano-vLLM` 的工程重心变化：

```text
原版:
    小而清晰的 Qwen3 dense 文本推理框架

qwen3.6:
    在小框架上验证更复杂的大模型推理机制
```

具体新增能力是：

1. **Qwen3.5 / Qwen3.6 hybrid 模型支持**  
   `qwen3_5.py` 支持 full attention 与 GatedDeltaNet 混合，每层由 `layer_types` 决定。

2. **FP8 checkpoint 加载支持**  
   `loader.py` 和 `quant.py` 支持 FP8 weight + scale 加载时反量化，`run_text_qwen36_fp8.py` 用于验证。

3. **MTP / speculative decoding 实验**  
   `qwen3_mtp.py` 提供 MTP prototype，`test_mtp_*` 和 `run_mtp_fast_decode.py` 负责 draft、verify、accept/reject 和性能统计。

4. **状态回滚验证**  
   `test_state_rollback.py` 验证 scheduler、KV block、GDN state slot、decode state 的保存与恢复。

5. **benchmark 和 smoke test 完整化**  
   `bench_qwen35_fixed.py`、`bench_mtp_draft_sweep.py`、`run_text_qwen35_v2.py`、`run_text_qwen36_fp8.py` 分别服务于 Qwen3.5、Qwen3.6 FP8 和 MTP 性能验证。

因此，这些根目录文件不是孤立脚本，而是围绕一条主线组织的：

```text
支持新模型结构
  ↓
支持新权重格式
  ↓
支持新 decode 策略
  ↓
验证状态正确性
  ↓
输出可用于项目报告/简历的性能指标
```


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
