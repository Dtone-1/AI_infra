# EvalScope 与模型服务启动脚本常用参数汇总

> 本文用于快速查阅：
>
> 1. `evalscope eval`：模型能力/精度评测常用参数
> 2. `evalscope perf`：模型服务性能压测常用参数
> 3. 模型服务启动脚本常用参数：以 **SGLang / OpenAI API 兼容服务** 为主要示例
>
> 注意：不同 EvalScope / SGLang 版本可能会增加、删除或调整参数，最终以当前环境中的以下命令为准：
>
> ```bash
> evalscope eval --help
> evalscope perf --help
> python -m sglang.launch_server --help
> ```

---

# 一、`evalscope eval` 常用参数

`evalscope eval` 主要用于回答：

> **模型答得对不对？能力是否发生退化？**

常用于 GSM8K、MMLU-Pro、GPQA、HumanEval 等精度/能力测试。

## 1. 基础模型与 API 参数

| 参数命令 | 参数解释 | 常见示例 |
|---|---|---|
| `--model` | 指定被评测的模型。可以是模型名称、本地模型路径，也可以是在 API 服务中使用的模型 ID。它回答的是“我要测哪个模型”。 | `--model /data/models/WeLMV4.5_YARN` |
| `--model-id` | 给本次被测模型设置一个展示名称，主要用于评测报告和结果目录中区分不同模型。 | `--model-id WeLMV4.5_YARN` |
| `--eval-type` | 指定 EvalScope 用什么方式调用模型。常见的 `openai_api` 表示不由 EvalScope 自己加载模型，而是请求一个已经启动的 OpenAI 兼容服务。 | `--eval-type openai_api` |
| `--api-url` | 指定模型 API 服务地址。EvalScope 会把测试题通过 HTTP 请求发到这个地址。 | `--api-url http://127.0.0.1:30000/v1` |
| `--api-key` | API 服务需要鉴权时填写密钥；本地内部服务通常可以为空或使用默认值。 | `--api-key EMPTY` |
| `--model-args` | 模型直接由 EvalScope 加载时，用于设置模型加载参数，例如精度、revision、device map 等。使用远程 API 模式时通常不重点使用。 | `--model-args precision=torch.float16,device_map=auto` |

## 2. 数据集相关参数

| 参数命令 | 参数解释 | 常见示例 |
|---|---|---|
| `--datasets` | 指定使用哪些评测数据集，是能力评测最核心的参数之一。 | `--datasets gsm8k` |
| `--dataset-dir` | 指定评测数据集在本地保存或下载的位置。 | `--dataset-dir /data/eval_datasets` |
| `--dataset-hub` | 指定数据集从哪里获取，例如 ModelScope 或 Hugging Face。 | `--dataset-hub modelscope` |
| `--limit` | 限制实际评测的数据量。调试脚本时非常有用，例如先只跑 10 或 100 条，确认流程正确后再跑全量。 | `--limit 100` |
| `--repeats` | 同一条样本重复推理多少次。代码题、pass@k 或需要观察随机性的测试中比较常见。 | `--repeats 5` |
| `--dataset-args` | 给某个数据集传递额外设置，例如 few-shot 数量、子集、prompt 模板等。 | `--dataset-args '{"gsm8k":{"few_shot_num":0}}'` |

## 3. 推理与生成参数

`--generation-config` 一般用一个 JSON 字符串传入多个生成参数，例如：

```bash
--generation-config '{
  "max_tokens": 4096,
  "temperature": 0.0,
  "top_p": 1.0,
  "stream": true,
  "timeout": 7200
}'
```

其中常见子参数如下。

| 参数命令 | 参数解释 | 常见示例 |
|---|---|---|
| `max_tokens` | 模型单个请求最多允许生成多少 token。设置过小可能导致答案被截断。 | `"max_tokens": 4096` |
| `temperature` | 控制生成随机性。性能回归和精度回归时通常设为 `0.0`，减少随机波动。 | `"temperature": 0.0` |
| `top_p` | Nucleus Sampling 的概率范围。通常与 temperature 一起控制采样行为。 | `"top_p": 1.0` |
| `stream` | 是否使用流式返回。若希望收集真正的 TTFT，一般需要开启。 | `"stream": true` |
| `timeout` | 单个请求最大允许等待多久，防止长推理任务被过早判定为超时。 | `"timeout": 7200` |
| `retries` | API 请求失败时允许重试的次数。 | `"retries": 5` |
| `seed` | 控制随机种子，便于复现实验。 | `"seed": 42` |
| `extra_body` | 向 OpenAI 兼容服务额外传递服务端自定义参数，例如是否启用 thinking。 | `"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}` |

## 4. 评测执行相关参数

| 参数命令 | 参数解释 | 常见示例 |
|---|---|---|
| `--eval-batch-size` | 评测批大小。在远程 API 模式下，可以理解为评测请求并发规模之一。数值越大，并发越高，但也更容易把服务压满。 | `--eval-batch-size 32` |
| `--seed` | EvalScope 整体随机种子，用于保证数据抽样和部分随机行为尽量可复现。 | `--seed 42` |
| `--collect-perf` | 在做精度评测的同时，额外收集请求延迟、TTFT、token 数等性能信息。 | `--collect-perf` |
| `--use-cache` | 复用之前的推理结果，避免重新跑完整模型推理。适合只重新做 review/打分。 | `--use-cache outputs/xxx` |

## 5. 输出相关参数

| 参数命令 | 参数解释 | 常见示例 |
|---|---|---|
| `--work-dir` | 指定评测结果保存目录。预测结果、review、report、日志等一般都存放在这里。 | `--work-dir /projs/solutionsdk/mahao/outputs/gsm8k_after_mtp1` |
| `--no-timestamp` | 不在输出目录名称后自动追加时间戳，使结果路径固定。 | `--no-timestamp` |
| `--enable-progress-tracker` | 将评测过程写入进度文件，方便外部工具或服务实时查看进度。 | `--enable-progress-tracker` |

## 6. 一个典型的 `evalscope eval` 示例

```bash
evalscope eval \
  --model /data/models/WeLMV4.5_YARN \
  --model-id WeLMV4.5_YARN \
  --datasets gsm8k \
  --eval-type openai_api \
  --api-url http://127.0.0.1:30000/v1 \
  --eval-batch-size 32 \
  --generation-config '{
      "max_tokens":4096,
      "temperature":0.0,
      "top_p":1.0,
      "stream":true
  }' \
  --seed 42 \
  --collect-perf \
  --work-dir ./outputs/gsm8k
```

---

# 二、`evalscope perf` 常用参数

`evalscope perf` 主要用于回答：

> **模型服务跑得快不快？能扛多大的并发和请求速率？**

主要观察 TTFT、TPOT、Latency、Throughput、P50/P90/P99 和请求成功率。

## 1. 服务与模型参数

| 参数命令 | 参数解释 | 常见示例 |
|---|---|---|
| `--url` | 被压测模型服务的 API 地址。EvalScope 会把性能请求发送到这里。 | `--url http://127.0.0.1:30000/v1/chat/completions` |
| `--model` | 指定被测模型名称或模型 ID。 | `--model WeLMV4.5_YARN-W8A8` |
| `--api` | 指定被测服务的 API 类型，例如 OpenAI 兼容 Chat Completions。 | `--api openai` |
| `--api-key` | 被测服务需要鉴权时设置 API key；本地服务一般可以为空。 | `--api-key ""` |
| `--tokenizer-path` | 指定 tokenizer 路径。EvalScope 需要 tokenizer 来计算输入、输出 token 数，并生成指定长度的 random workload。 | `--tokenizer-path /data/models/WeLMV4.5_YARN-W8A8` |
| `--name` | 给本轮压测结果指定一个名称，方便后续区分多轮实验。 | `--name welm_mtp_on` |

## 2. 压力与请求数量参数

| 参数命令 | 参数解释 | 常见示例 |
|---|---|---|
| `--parallel` | 最大同时在途的请求数量，也就是常说的并发数。并发越高，服务器同时处理的请求越多。 | `--parallel 40` |
| `--number` | 本轮正式压测一共发送多少个请求。 | `--number 200` |
| `--rate` | 请求调度速率，单位为请求/秒。`4` 可以理解为平均每秒调度约 4 个新请求。默认 closed-loop 下仍会受到 `parallel` 的并发上限约束。 | `--rate 4` |
| `--open-loop` | 开启开放环压测。开启后，请求按指定 rate 发出，不等待前面的请求完成，更接近真实流量到达场景。 | `--open-loop` |
| `--duration` | 按时间限制本轮压测，例如压测 300 秒。与 number 同时设置时，先达到的停止条件会结束继续发新请求。 | `--duration 300` |
| `--warmup-num` | 正式统计之前先发送若干预热请求。预热请求不计入最终性能结果，可减少 JIT、Graph、内存池初始化等冷启动影响。 | `--warmup-num 10` |
| `--sleep-interval` | 多轮性能测试之间休息多少秒，避免连续压测导致服务端一直处在高负载状态。 | `--sleep-interval 5` |

## 3. 输入数据集参数

| 参数命令 | 参数解释 | 常见示例 |
|---|---|---|
| `--dataset` | 指定压测数据来源。`random` 表示 EvalScope 自动生成随机输入，适合固定长度性能对比。 | `--dataset random` |
| `--dataset-path` | 使用自定义或本地真实数据集时指定数据文件路径。 | `--dataset-path ./data/test.jsonl` |
| `--min-prompt-length` | 输入 prompt 最小 token 长度。对于 `random` 数据集，可以和 max 设置成相同值来构造固定长度输入。 | `--min-prompt-length 11000` |
| `--max-prompt-length` | 输入 prompt 最大 token 长度。 | `--max-prompt-length 11000` |
| `--min-tokens` | 要求生成的最少 token 数；并非所有后端都完整支持。 | `--min-tokens 100` |
| `--max-tokens` | 单个请求最多生成多少 token。 | `--max-tokens 100` |
| `--prefix-length` | 为输入增加固定长度前缀，适合测试 prefix cache 等场景。 | `--prefix-length 0` |

## 4. 生成参数

| 参数命令 | 参数解释 | 常见示例 |
|---|---|---|
| `--temperature` | 控制随机性。性能 A/B 测试通常设成 `0.0`，保证不同版本生成行为尽量一致。 | `--temperature 0.0` |
| `--stream` | 是否使用 SSE 流式输出。要准确统计 TTFT，一般需要开启。 | `--stream` |
| `--frequency-penalty` | 调整对已经生成 token 的重复惩罚，一般纯性能压测不需要特别修改。 | `--frequency-penalty 0` |
| `--logprobs` | 是否要求服务返回 token 的 log probability。开启后可能增加额外开销，一般性能基准不建议无理由开启。 | `--logprobs` |

## 5. 网络与超时参数

| 参数命令 | 参数解释 | 常见示例 |
|---|---|---|
| `--total-timeout` | 单个请求从开始到结束允许的最大总等待时间。长上下文/长输出测试通常需要设置较大值。 | `--total-timeout 21600` |
| `--connect-timeout` | 建立 HTTP 连接允许等待的最大时间。 | `--connect-timeout 60` |
| `--read-timeout` | 已连接后等待服务端返回数据的超时时间。 | `--read-timeout 3600` |
| `--headers` | 给每个 HTTP 请求增加额外 header。 | `--headers "Authorization=Bearer xxx"` |
| `--no-test-connection` | 跳过 EvalScope 在正式压测前对服务进行连接检查。一般不需要设置。 | `--no-test-connection` |

## 6. 输出相关参数

| 参数命令 | 参数解释 | 常见示例 |
|---|---|---|
| `--outputs-dir` | 指定性能测试结果保存目录。 | `--outputs-dir WeLM-mlu-outputs` |
| `--no-timestamp` | 输出目录中不自动添加时间戳。 | `--no-timestamp` |
| `--visualizer` | 将测试结果同步到 wandb、swanlab 等可视化工具。 | `--visualizer wandb` |
| `--enable-progress-tracker` | 开启实时进度记录。 | `--enable-progress-tracker` |
| `--debug` | 输出更详细的调试日志，排查压测脚本问题时使用。 | `--debug` |

## 7. 一个典型的 `evalscope perf` 示例

```bash
evalscope perf \
  --url http://127.0.0.1:30000/v1/chat/completions \
  --model WeLMV4.5_YARN-W8A8 \
  --dataset random \
  --parallel 40 \
  --rate 4 \
  --number 200 \
  --min-prompt-length 11000 \
  --max-prompt-length 11000 \
  --min-tokens 100 \
  --max-tokens 100 \
  --temperature 0.0 \
  --tokenizer-path /data/models/WeLMV4.5_YARN-W8A8 \
  --warmup-num 10 \
  --outputs-dir WeLM-mlu-outputs
```

---

# 三、模型服务启动脚本中常见的设置参数

下面以 **SGLang 的 `python -m sglang.launch_server`** 为主要示例。

一个最基础的模型服务启动方式通常类似：

```bash
python -m sglang.launch_server \
  --model-path /data/models/xxx \
  --served-model-name xxx \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 4
```

## 1. 模型与服务地址参数

| 参数命令 | 参数解释 | 常见示例 |
|---|---|---|
| `--model-path` | 指定真正要加载的模型权重目录，是服务启动脚本最核心的参数。 | `--model-path /data/models/WeLMV4.5_YARN` |
| `--served-model-name` | 指定通过 API 对外暴露的模型名称。客户端请求中的 `model` 字段通常需要和这里匹配。 | `--served-model-name WeLMV4.5_YARN` |
| `--host` | 指定服务监听的网络地址。`127.0.0.1` 只能本机访问；`0.0.0.0` 表示监听所有网卡。 | `--host 0.0.0.0` |
| `--port` | 指定服务监听端口。EvalScope 中的 `--url` / `--api-url` 必须使用同一个端口。 | `--port 30000` |
| `--trust-remote-code` | 某些 Hugging Face 模型依赖仓库中的自定义 Python 代码时需要开启。只有信任模型代码来源时才应使用。 | `--trust-remote-code` |

## 2. 多卡与并行参数

| 参数命令 | 参数解释 | 常见示例 |
|---|---|---|
| `--tp-size` | Tensor Parallel 大小。把同一个模型层的张量切到多张卡上协同计算。4 卡 TP 通常设置为 4。 | `--tp-size 4` |
| `--dp-size` | Data Parallel 大小。运行多份模型副本处理不同请求，更偏向提升总吞吐。 | `--dp-size 2` |
| `--ep-size` | Expert Parallel 大小。MoE 模型中把不同 Expert 分布到不同设备。 | `--ep-size 4` |
| `--enable-dp-attention` | 某些 MoE/混合并行场景中，让 Attention 使用 DP 策略，与 FFN/MoE 的 TP/EP 组合使用。 | `--enable-dp-attention` |

## 3. 上下文与 KV Cache 参数

| 参数命令 | 参数解释 | 常见示例 |
|---|---|---|
| `--context-length` | 限制单个请求允许的最大上下文长度。长上下文模型测试时需要特别关注。 | `--context-length 32768` |
| `--mem-fraction-static` | 控制模型权重 + KV Cache 静态内存池占 GPU 显存的比例。调高可增加 KV Cache 容量和并发能力，但留给 activation、Graph 等临时空间会减少，过高可能 OOM。 | `--mem-fraction-static 0.80` |
| `--max-total-tokens` | 限制 KV/内存池能够容纳的总 token 数。多数情况下由框架根据显存自动计算，调试时才常手动指定。 | `--max-total-tokens 200000` |
| `--kv-cache-dtype` | 指定 KV Cache 的数据类型，例如 auto、FP8 等。更低精度通常可以减少 KV 显存，但需要硬件和模型支持。 | `--kv-cache-dtype auto` |
| `--page-size` | Paged KV Cache 中每一页包含多少 token。属于更底层的 KV 管理参数，一般不需要频繁修改。 | `--page-size 1` |

## 4. 并发与调度参数

| 参数命令 | 参数解释 | 常见示例 |
|---|---|---|
| `--max-running-requests` | 服务端最多允许多少请求同时处于运行状态。调高可能提升吞吐，但需要更多 KV Cache/activation 显存；过高容易造成 OOM 或尾延迟恶化。 | `--max-running-requests 256` |
| `--schedule-policy` | 请求调度策略。常见 `fcfs` 表示先到先服务，部分场景可使用 LPM 等提高公共前缀 cache 命中率。 | `--schedule-policy fcfs` |
| `--schedule-conservativeness` | 调度器预留显存的保守程度。更大通常更保守，可减少请求 retract，但可能降低最大并发。 | `--schedule-conservativeness 1.0` |
| `--disable-overlap-schedule` | 关闭调度与模型执行之间的 overlap。通常用于调试、性能定位或保证实验变量更单一。 | `--disable-overlap-schedule` |

## 5. Chunked Prefill 参数

| 参数命令 | 参数解释 | 常见示例 |
|---|---|---|
| `--chunked-prefill-size` | 把超长 prompt 的 Prefill 分成多个 chunk，每个 chunk 最多处理多少 token。可以降低单次 Prefill 峰值显存，但 chunk 太小可能降低 Prefill 性能。 | `--chunked-prefill-size 4096` |
| `--max-prefill-tokens` | 限制一个 Prefill batch 最多包含多少 token，用于控制 Prefill 峰值显存和 batch 大小。 | `--max-prefill-tokens 16384` |

## 6. Graph 相关参数

| 参数命令 | 参数解释 | 常见示例 |
|---|---|---|
| `--disable-cuda-graph` | 关闭 CUDA Graph。开启 Graph 通常能减少 kernel launch 开销、提升 Decode 性能；关闭后更适合调试和逐算子 Profile。 | `--disable-cuda-graph` |
| `--cuda-graph-max-bs-decode` | 控制 Decode CUDA Graph 能覆盖到的最大 batch size。设置更大可能提升大 batch Decode 性能，但会增加显存消耗。 | `--cuda-graph-max-bs-decode 256` |

> 在 MLU 后端中可能有对应的 Graph 实现或不同参数名称，应以当前 `sglang-mlu` 版本的 `--help` 和项目代码为准。

## 7. Prefix Cache / Radix Cache 参数

| 参数命令 | 参数解释 | 常见示例 |
|---|---|---|
| `--disable-radix-cache` | 关闭 SGLang 的 Radix / Prefix Cache。正常在线服务开启它可以复用相同前缀的 KV；做严格性能 A/B 时有时会关闭，避免缓存命中影响测试结果。 | `--disable-radix-cache` |

## 8. 精度、量化与计算后端参数

| 参数命令 | 参数解释 | 常见示例 |
|---|---|---|
| `--dtype` | 指定模型计算数据类型，例如 FP16、BF16。部分情况下可由模型配置自动确定。 | `--dtype bfloat16` |
| `--quantization` | 指定量化方式，例如 FP8 等。只有模型权重和后端都支持时才能使用。 | `--quantization fp8` |
| `--attention-backend` | 选择 Attention 使用的实现后端，例如 FlashAttention 或其他优化 kernel。参数名称和值取决于当前 SGLang/硬件版本。 | `--attention-backend xxx` |
| `--moe-backend` | MoE 模型中选择 Expert 计算和通信实现。不同硬件、EP 数和模型可能适合不同 backend。 | `--moe-backend xxx` |

## 9. MTP / Speculative Decoding 常见参数类别

投机解码相关参数随 SGLang 版本和算法变化较快，因此名称应以当前代码为准。

| 参数命令/类别 | 参数解释 | 常见形式 |
|---|---|---|
| `--speculative-algorithm` | 选择投机解码算法，例如 EAGLE、MTP、DFlash 等。 | `--speculative-algorithm ...` |
| `--speculative-draft-model-path` | 某些 speculative 算法需要额外 draft model 时，用于指定 draft 模型路径。 | `--speculative-draft-model-path /data/models/draft` |
| speculative steps / draft tokens | 控制一次 Draft 预测多少 token。预测越多，理论加速空间越大，但 verify 成本和拒绝概率也会增加。 | 具体名称以版本为准 |
| accept / verify 相关配置 | 控制 speculative decoding 的验证策略。 | 具体名称以版本为准 |

---

# 四、EvalScope 参数与模型启动参数的对应关系

| 模型服务启动端 | EvalScope 客户端 | 对应关系 |
|---|---|---|
| `--port 30000` | `--url http://127.0.0.1:30000/...` | 两边端口必须一致，否则 EvalScope 找不到服务 |
| `--served-model-name WeLM` | `--model WeLM` | 客户端模型名通常要与服务端暴露名称匹配 |
| `--context-length 32768` | `--min/max-prompt-length`、`--max-tokens` | EvalScope 构造的输入+输出不能超过服务端支持的上下文 |
| `--tp-size 4` | 无直接对应参数 | TP 是服务端内部并行方式，EvalScope 不需要知道 |
| `--mem-fraction-static 0.8` | 无直接对应参数 | 决定服务端 KV Cache 容量和并发能力 |
| `--max-running-requests 256` | `--parallel` / `--rate` | 服务端决定最多能跑多少请求，EvalScope 决定给它多大压力 |
| `--disable-radix-cache` | `--prefix-length` / 数据集内容 | 是否存在公共前缀会影响 cache 命中与性能 |
| Graph/MTP/MoE 参数 | EvalScope 无直接对应参数 | 都是服务端内部优化，EvalScope 只负责测最终结果 |

---

# 五、最常用的帮助命令

```bash
evalscope eval --help
```

查看当前安装版本支持的精度评测参数。

```bash
evalscope perf --help
```

查看当前安装版本支持的性能压测参数。

```bash
python -m sglang.launch_server --help
```

查看当前 SGLang / SGLang-MLU 版本真正支持的服务启动参数。

如果公司内部 `sglang-mlu` 对上游 SGLang 做了修改，应优先以：

```text
当前分支代码
+
当前 --help 输出
+
团队启动脚本
```

为准，而不是照搬网上其他版本的参数。

---

# 六、参考资料

- EvalScope 官方文档：能力评测参数说明  
  https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html

- EvalScope 官方文档：性能压测参数说明  
  https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/parameters.html

- SGLang 官方仓库：Server Arguments  
  https://github.com/sgl-project/sglang/blob/main/docs/cookbook/base/reference/server_arguments.mdx

- SGLang 官方仓库：Hyperparameter Tuning  
  https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/hyperparameter_tuning.mdx


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-实习工程能力|模块-实习工程能力]]
- 关联阅读：[[outputs/项目整理/专题-06-服务部署评测与协作|专题-06-服务部署评测与协作]]

%% 项目关联导航：结束 %%
