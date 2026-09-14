# EvalScope 使用、结果分析与内部评测原理 —— 结合当前 WeLM 测试脚本

> 适用场景：使用已经启动的 OpenAI API 兼容模型服务，通过 EvalScope 做 **精度/能力评测** 与 **推理性能压测**。  
> 本文重点结合当前两个脚本：
>
> - `bench_gsm8k(1).sh`：`evalscope eval`，使用 GSM8K 做模型能力/精度评测。
> - `bench_client(1).sh`：`evalscope perf`，使用随机固定长度输入做服务性能压测。

---

## 1. EvalScope 是什么

**EvalScope** 是 ModelScope（魔搭社区）提供的一站式大模型评测框架。

可以把它理解成一个“自动化测试客户端”：

```text
测试数据集
   ↓
EvalScope
   ↓
构造 Prompt / 请求
   ↓
请求模型服务
   ↓
模型生成结果
   ↓
EvalScope 收集结果
   ↓
精度判分 / 性能统计
   ↓
输出报告
```

它主要解决两类问题：

| 目的 | 命令 | 核心问题 |
|---|---|---|
| 模型能力评测 | `evalscope eval` | 模型“答得对不对”？ |
| 模型性能压测 | `evalscope perf` | 模型服务“跑得快不快、能扛多少并发”？ |

因此你现在的两个脚本实际上对应两条完全不同的测试链路：

```text
bench_gsm8k.sh
    ↓
evalscope eval
    ↓
GSM8K
    ↓
Accuracy
    ↓
验证模型能力是否发生退化
```

以及：

```text
bench_client.sh
    ↓
evalscope perf
    ↓
大量并发请求
    ↓
TTFT / TPOT / Latency / Throughput / P99
    ↓
验证性能是否提升
```

在模型适配和性能优化工作中，通常两者都需要：

```text
先验证功能/精度正确
        ↓
再测性能
        ↓
优化代码
        ↓
再次验证精度没有下降
        ↓
比较性能是否提升
```

---

# 2. EvalScope 怎么安装和确认是否可用

如果只做普通能力评测：

```bash
pip install -U evalscope
```

如果主要使用性能压测功能，推荐：

```bash
pip install -U "evalscope[perf]"
```

查看是否安装成功：

```bash
evalscope --help
```

查看版本：

```bash
evalscope --version
```

也可以：

```bash
pip show evalscope
```

查看 `eval` 参数：

```bash
evalscope eval --help
```

查看 `perf` 参数：

```bash
evalscope perf --help
```

在公司环境中，如果测试结果要做正式前后对比，建议同时保存：

```bash
evalscope --version
git rev-parse HEAD
```

原因是：

```text
模型代码版本
EvalScope版本
启动参数
测试数据
并发参数
```

任何一项变化，都可能导致最终结果不可直接横向比较。

---

# 3. 第一类：`evalscope eval` —— 模型能力/精度评测

你的 GSM8K 脚本核心如下：

```bash
evalscope eval \
  --model "${MODEL_PATH}" \
  --model-id "${MODEL_ID}" \
  --datasets gsm8k \
  --eval-type openai_api \
  --api-url "${API_URL}" \
  --generation-config '...' \
  --eval-batch-size 32 \
  --seed 42 \
  --no-timestamp \
  --collect-perf \
  --work-dir /projs/solutionsdk/mahao/outputs/gsm8k_after_mtp1
```

它的目的不是单纯测速度，而是：

> 从 GSM8K 中取数学题 → 发给已经启动的 WeLM 服务 → 获取模型答案 → EvalScope 自动提取答案 → 和标准答案比较 → 统计 Accuracy。

---

## 4. 你的 GSM8K 脚本每个参数是什么意思

### 4.1 `--model`

```bash
--model "${MODEL_PATH}"
```

你的值为：

```bash
/data/models/WeLMV4.5_YARN
```

它用于告诉 EvalScope 当前评测所对应的模型。

在 `openai_api` 模式下，真正推理并不是 EvalScope 自己加载这个模型权重，而是：

```text
EvalScope
   ↓ HTTP
127.0.0.1:30000
   ↓
已经启动好的 SGLang / 其他推理服务
   ↓
真正执行模型推理
```

所以要特别理解：

> `evalscope eval` 在你当前脚本里主要扮演“评测客户端”，模型本体实际上由端口 30000 对应的服务进程负责。

---

### 4.2 `--model-id`

```bash
--model-id "${MODEL_ID}"
```

你的值：

```bash
WeLMV4.5_YARN
```

主要用于给当前模型一个更清晰的标识，方便输出报告、结果目录和多模型比较。

---

### 4.3 `--datasets gsm8k`

```bash
--datasets gsm8k
```

表示使用 **GSM8K** 数据集。

GSM8K 是小学到初中级别的英文数学应用题数据集，重点测试：

- 数学推理；
- 多步推理；
- 从自然语言问题中建立计算过程；
- 最终数字答案是否正确。

这里最终常见核心指标是：

```text
Accuracy
```

例如：

```text
Accuracy = 0.92
```

表示约 92% 的题目最终答案正确。

---

### 4.4 `--eval-type openai_api`

```bash
--eval-type openai_api
```

这一项非常重要。

它表示：

> 不让 EvalScope 在当前 Python 进程里直接加载模型，而是通过 OpenAI API 兼容接口请求一个已经启动的模型服务。

流程是：

```text
EvalScope
   ↓
生成请求
   ↓
HTTP POST
   ↓
127.0.0.1:30000/v1/chat/completions
   ↓
SGLang 服务
   ↓
WeLM
   ↓
返回 response
```

这非常适合你现在的工作，因为你本身就是：

```text
先 launch WeLM 服务
再运行客户端测试脚本
```

---

### 4.5 `--api-url`

```bash
--api-url http://127.0.0.1:30000/v1/chat/completions
```

意思是：

```text
EvalScope 应该把测试请求发送到哪里。
```

这里：

- `127.0.0.1`：当前机器；
- `30000`：模型服务监听端口；
- `/v1/chat/completions`：OpenAI Chat Completion API 路径。

因此测试之前必须确认模型服务已经正常启动。

可以先测试：

```bash
curl http://127.0.0.1:30000/v1/models
```

或者根据你当前服务支持的 API 做一个简单请求。

如果服务根本没有启动，EvalScope 本身不会替你启动 WeLM。

---

## 5. `generation-config` 是干什么的

你的配置中包含：

```json
{
  "timeout": 7200000,
  "batch_size": 32,
  "max_tokens": 20000,
  "top_p": 1.0,
  "temperature": 0.0,
  "seed": 42,
  "do_sample": false,
  "stream": true,
  "extra_body": {
    "chat_template_kwargs": {
      "enable_thinking": false,
      "thinking": false
    }
  }
}
```

可以理解为：

> “EvalScope 每一道题发给模型时，希望模型按照什么生成配置回答。”

关键参数如下。

| 参数 | 含义 |
|---|---|
| `temperature=0.0` | 尽量确定性输出，减少随机性 |
| `do_sample=false` | 不进行随机采样 |
| `seed=42` | 固定随机种子，方便复现 |
| `max_tokens=20000` | 单次回答允许的最大输出 token 数 |
| `stream=true` | 使用流式返回，可统计 TTFT |
| `timeout` | 单个请求允许等待的超时时间 |
| `enable_thinking=false` | 关闭你当前模型对应的 thinking 行为 |

做“前后代码版本精度回归”时，这些参数应该保持不变。

否则：

```text
代码版本变化
+
采样参数变化
```

会导致你无法判断精度变化究竟是谁造成的。

---

# 6. `evalscope eval` 内部到底是怎么评测的

可以分成 6 个阶段。

## 阶段 1：读取数据集

EvalScope 加载 GSM8K，例如一条样本本质上包含：

```text
Question:
某个数学应用题

Reference Answer:
标准答案
```

---

## 阶段 2：数据集 Adapter 构造 Prompt

不同数据集不能完全使用同一套处理逻辑。

因此 EvalScope 会通过对应的数据集 Adapter 做：

```text
原始数据
   ↓
读取 question
   ↓
按照 GSM8K 的模板组织 Prompt
   ↓
得到最终模型输入
```

---

## 阶段 3：发送到模型服务

因为你的配置是：

```bash
--eval-type openai_api
```

所以 EvalScope 会构造 OpenAI API 请求：

```text
EvalScope
   ↓
HTTP Request
   ↓
localhost:30000
```

这里真正运行：

```text
Tokenizer
Prefill
Decode
MTP
MoE
MLU kernel
```

这些工作的是你的 **SGLang/WeLM 服务端**，不是 EvalScope。

---

## 阶段 4：接收模型输出

假设一道 GSM8K 问题的标准答案为：

```text
42
```

模型可能返回：

```text
经过计算，最终答案是 42。
```

EvalScope 保存原始 prediction。

---

## 阶段 5：提取最终答案并判分

对 GSM8K 这种任务，不能简单地拿整段文本做字符串比较。

例如：

```text
标准答案：
42

模型输出：
First ..., then ..., therefore the answer is 42.
```

两段文本显然不一样，但模型是答对的。

因此 GSM8K Adapter 会从模型输出中提取最终数值答案，再与 reference answer 比较。

可以简化理解为：

```text
模型完整输出
   ↓
答案提取
   ↓
42
   ↓
和标准答案 42 比较
   ↓
正确
```

最终单题得到：

```text
1 = 正确
0 = 错误
```

然后：

```text
Accuracy
=
正确题数 / 总题数
```

例如：

```text
总共 1000 道题
答对 930 道

Accuracy = 930 / 1000 = 93%
```

---

## 阶段 6：汇总输出报告

最后 EvalScope 会聚合：

```text
每一道题 prediction
每一道题 reference
每一道题是否正确
        ↓
最终 Accuracy
```

因此 `evalscope eval` 的核心逻辑是：

```text
Dataset
   ↓
Prompt
   ↓
Model API
   ↓
Prediction
   ↓
Answer Extraction
   ↓
Scoring
   ↓
Aggregation
   ↓
Accuracy
```

---

# 7. `--collect-perf` 在精度测试里有什么作用

你的脚本还有：

```bash
--collect-perf
```

它表示在做能力评测的同时，也采集每个请求的一些性能信息，例如：

- Latency；
- TTFT；
- token usage。

你同时设置了：

```json
"stream": true
```

因此 EvalScope 才能比较准确地获取：

```text
请求发送时间
↓
首个流式 chunk 到达时间
↓
TTFT
```

不过要注意：

> `evalscope eval --collect-perf` 中收集到的性能数据主要是辅助信息；正式做吞吐、并发、TTFT/TPOT 压测时，仍然应该使用 `evalscope perf`。

---

# 8. `evalscope eval` 的结果在哪里

你的脚本指定了：

```bash
--work-dir /projs/solutionsdk/mahao/outputs/gsm8k_after_mtp1
--no-timestamp
```

因此结果根目录就是：

```bash
/projs/solutionsdk/mahao/outputs/gsm8k_after_mtp1
```

由于使用了：

```bash
--no-timestamp
```

EvalScope 不再额外给 `work-dir` 添加运行时间戳目录。

当前 EvalScope 的典型目录结构类似：

```text
gsm8k_after_mtp1/
├── configs/
│   └── task_config_xxx.yaml
├── logs/
│   └── eval_log.log
├── predictions/
│   └── WeLMV4.5_YARN/
│       └── gsm8k.jsonl
├── reviews/
│   └── WeLMV4.5_YARN/
│       └── gsm8k.jsonl
└── reports/
    └── WeLMV4.5_YARN/
        └── gsm8k.json
```

不同 EvalScope 版本的具体文件名可能略有差别，但核心概念基本一致。

---

## 9. 这些结果文件分别看什么

### 9.1 `configs/`

查看：

```bash
cat configs/*.yaml
```

这里记录本次评测配置。

用途：

> 将来看到一个结果时，可以确认当时到底用了哪个 dataset、batch size、生成参数等。

---

### 9.2 `logs/eval_log.log`

查看：

```bash
cat logs/eval_log.log
```

或者：

```bash
less logs/eval_log.log
```

主要用于：

- 查看评测过程；
- 查看报错；
- 查看任务是否完整跑完。

---

### 9.3 `predictions/`

这里最值得排查“模型到底生成了什么”。

例如：

```bash
less predictions/WeLMV4.5_YARN/gsm8k.jsonl
```

通常能看到每道题对应的模型 prediction。

当你发现 Accuracy 突然下降时，应该首先抽查这里。

例如：

```text
输入有没有问题？
模型是不是输出空字符串？
是不是输出被截断？
是不是答案格式发生变化？
是不是 MTP 开启后出现异常 token？
```

---

### 9.4 `reviews/`

这里更接近“每一道题经过 EvalScope 判分之后的结果”。

适合检查：

```text
模型答案是什么
标准答案是什么
EvalScope 提取出了什么
这一题最后判对还是判错
```

如果模型看起来明明答对了，但 Accuracy 不正常，就应该重点查看这里。

---

### 9.5 `reports/`

这是最终汇总结果。

重点寻找类似：

```text
Accuracy
Score
Num
```

查看：

```bash
cat reports/WeLMV4.5_YARN/gsm8k.json
```

如果装有 `jq`：

```bash
jq . reports/WeLMV4.5_YARN/gsm8k.json
```

会比直接 `cat` 更容易阅读。

---

# 10. 第二类：`evalscope perf` —— 性能压测

你的另一个脚本核心为：

```bash
evalscope perf \
    --url http://127.0.0.1:30000/v1/chat/completions \
    --model ${MODEL_NAME} \
    --dataset random \
    --parallel ${parallel} \
    --rate "${rate}" \
    --number 200 \
    --temperature 0.0 \
    --min-prompt-length 11000 \
    --max-prompt-length 11000 \
    --min-tokens 100 \
    --max-tokens 100 \
    --tokenizer-path "${tokenizer_path}" \
    --warmup-num 10 \
    --outputs-dir WeLM-mlu-outputs
```

这个脚本不关心模型数学题答得对不对。

它关心的是：

```text
200 个请求打进去后

多久收到第一个 token？
后面的 token 生成有多快？
整体吞吐量多大？
高并发下排队严重不严重？
P99 延迟多大？
有没有请求失败？
```

---

# 11. 你的性能脚本总体在做什么

你定义了：

```bash
declare -A rate_parallel=(
    ["0.3"]=3
    ["0.5"]=5
    ["1.0"]=10
    ["2.0"]=20
    ["4.0"]=40
    ["6.0"]=60
    ["8.0"]=80
    ["20.0"]=200
)
```

然后循环：

```bash
for rate in ...
```

因此它会依次测试：

| Request Rate | Parallel |
|---:|---:|
| 0.3 req/s | 3 |
| 0.5 req/s | 5 |
| 1.0 req/s | 10 |
| 2.0 req/s | 20 |
| 4.0 req/s | 40 |
| 6.0 req/s | 60 |
| 8.0 req/s | 80 |
| 20.0 req/s | 200 |

宏观上相当于：

```text
低压力
 ↓
中压力
 ↓
高压力
 ↓
极高压力
```

通过不断加压，观察服务器性能从稳定到饱和的过程。

---

# 12. `rate` 和 `parallel` 到底有什么区别

这是性能压测中最容易混淆的两个概念。

## `--rate`

例如：

```bash
--rate 4
```

表示目标请求到达速率约为：

```text
每秒发 4 个请求
```

它描述的是：

> 请求“进来得有多快”。

---

## `--parallel`

例如：

```bash
--parallel 40
```

表示最多允许大约 40 个请求同时处于处理中。

它描述的是：

> 同一时刻最多可以挂着多少个还没处理完的请求。

---

可以类比餐厅：

```text
rate
=
每秒有多少新客人进餐厅

parallel
=
餐厅最多允许多少桌客人同时就餐
```

当：

```text
请求进入速度
>
模型服务处理速度
```

就会开始排队。

此时最明显的现象通常是：

```text
TTFT 急剧增加
P99 急剧增加
Latency 增加
吞吐量增长开始变慢甚至不再增长
```

这通常意味着服务已经接近饱和点。

---

# 13. `--number 200`

```bash
--number 200
```

表示这一档配置总共正式发送 200 个测试请求。

例如：

```text
rate=4
parallel=40
number=200
```

就是：

> 以这一档的请求速率和并发上限，完成 200 个正式测试请求，然后统计这一档性能。

---

# 14. `--warmup-num 10`

```bash
--warmup-num 10
```

表示正式统计前先执行 10 个 warmup 请求。

为什么需要 warmup？

因为刚启动模型后可能存在：

- CUDA/MLU kernel 首次初始化；
- 图编译；
- JIT；
- 内存池初始化；
- CUDA Graph / MLU Graph 初始化；
- cache 初始化；
- lazy initialization。

如果把前几个请求直接算进正式数据：

```text
第一次特别慢
↓
把平均 TTFT 拉高
↓
结果不能代表稳定运行状态
```

因此：

```text
10 个 warmup
    ↓
让服务进入稳定状态
    ↓
再正式统计 200 个请求
```

---

# 15. `--dataset random`

你的性能压测使用：

```bash
--dataset random
```

这意味着：

> 不是拿 GSM8K 真实数学题做性能测试，而是让 EvalScope 构造随机文本/token 请求。

原因是性能测试希望尽量控制变量。

如果使用真实数据：

```text
请求 A 输入 100 token
请求 B 输入 3000 token
请求 C 输出 20 token
请求 D 输出 1000 token
```

最终性能很难直接比较。

而你的脚本固定：

```bash
--min-prompt-length 11000
--max-prompt-length 11000

--min-tokens 100
--max-tokens 100
```

因此测试 workload 大体固定为：

```text
输入 ≈ 11,000 token
输出 ≈ 100 token
```

这样特别适合做代码修改前后的 A/B 性能比较。

---

# 16. 你脚本中一个容易忽略的问题

脚本前面定义：

```bash
input_len=11108
output_len=89
```

但是当前真正运行的命令使用的是：

```bash
--min-prompt-length 11000
--max-prompt-length 11000
--min-tokens 100
--max-tokens 100
```

而使用变量的代码：

```bash
--max-prompt-length "${input_len}"
--min-prompt-length "${input_len}"
--max-tokens "${output_len}"
--min-tokens "${output_len}"
```

目前已经被注释掉。

所以：

> 你现在实际测的不是 `11108 / 89`，而是 `11000 / 100`。

这个在汇报测试配置时一定要以真正执行的命令为准。

---

# 17. `tokenizer-path` 为什么必要

```bash
--tokenizer-path "${tokenizer_path}"
```

EvalScope 需要 tokenizer 的原因之一是：

```text
你要求的是 11000 token
而不是 11000 个字符
```

EvalScope 必须知道：

```text
字符串
↓ tokenizer
token IDs
↓
token 数量
```

才能尽量构造指定 token 长度的输入，并统计 input/output token。

因此 tokenizer 最好和被测模型对应。

如果 tokenizer 不一致：

```text
EvalScope 认为输入 11000 token
但服务端实际 tokenizer 计算出来不是 11000
```

测试 workload 就会失真。

---

# 18. `evalscope perf` 内部是怎么工作的

性能压测可以简化成下面的流程：

```text
生成测试请求
     ↓
根据 rate 调度请求发送时间
     ↓
根据 parallel 限制在途请求数
     ↓
HTTP 请求模型服务
     ↓
记录 request_start
     ↓
等待第一个流式 token/chunk
     ↓
记录 first_token_time
     ↓
持续接收后续 token/chunk
     ↓
记录每个响应到达时间
     ↓
最后一个 token 到达
     ↓
记录 request_end
     ↓
计算单请求性能
     ↓
对 200 个请求做平均值 / P50 / P90 / P99 等统计
```

---

# 19. EvalScope 怎么测 TTFT

对于一个请求：

```text
t0 = 请求发送
t1 = 收到第一个生成结果
```

则：

```text
TTFT = t1 - t0
```

即：

```text
TTFT = Time To First Token
```

可以粗略理解为：

> 用户点击“发送”以后，要等多久才第一次看到模型开始回答。

TTFT 里面通常不仅包含模型 Prefill 计算，还可能包含：

```text
请求排队
HTTP 开销
调度器等待
Prefill
首个 Decode
流式传输开销
```

因此高并发时：

```text
请求开始排队
↓
TTFT 会明显增加
```

这正是压测希望观察到的真实服务行为。

---

# 20. EvalScope 怎么测 TPOT

假设：

```text
整个请求耗时 = Latency
首 token 时间 = TTFT
输出 token 数 = N
```

常见 TPOT 可以理解为：

```text
TPOT ≈ (Latency - TTFT) / (N - 1)
```

即：

> 除去首 token 之前的等待，后续平均生成一个 token 需要多少时间。

例如：

```text
TPOT = 20 ms/token
```

可以粗略换算：

```text
单请求稳定 Decode ≈ 1000 / 20 = 50 token/s
```

因此：

```text
TPOT 越小越好
Decode token/s 越大越好
```

---

# 21. Latency 是什么

```text
Latency
=
request_end - request_start
```

也就是：

> 一个完整请求，从发送开始到全部输出完成，总共花了多久。

粗略关系：

```text
Latency
≈
TTFT + 后续 Decode 时间
```

所以：

- TTFT 更偏向“什么时候开始回答”；
- TPOT 更偏向“开始回答后吐字有多快”；
- Latency 是“整道请求最终多久完成”。

---

# 22. Throughput 是什么

性能测试中最重要的是区分：

```text
单请求速度
```

和：

```text
整个服务的总吞吐
```

例如一个模型：

```text
单请求：50 token/s
```

并不代表服务器总吞吐只有 50 token/s。

因为服务器可能 Continuous Batching：

```text
request 1 ┐
request 2 ├─ 同一个 batch
request 3 ├─ 一起 decode
request 4 ┘
```

最终整个服务可能达到：

```text
1000+ output token/s
```

EvalScope 常见的输出吞吐指标：

```text
Output Throughput (tok/s)
=
所有成功请求产生的 output token 总数
/
测试总时间
```

所以：

> 做推理引擎优化时，吞吐量通常越高越好，但不能脱离 TTFT/P99 单独看。

---

# 23. P99 是什么

假设 100 个请求的 TTFT 从小到大排序：

```text
request 1
request 2
...
request 99
request 100
```

P99 可以粗略理解成：

> 99% 的请求都不超过这个值，只有最慢的约 1% 更差。

例如：

```text
Mean TTFT = 300 ms
P99 TTFT = 5000 ms
```

说明：

```text
大部分请求很快
但是少量请求非常慢
```

这对于线上服务是非常重要的问题。

因此：

```text
Mean
```

只能说明“平均用户”。

而：

```text
P99
```

反映“尾部最差用户体验”。

---

# 24. 你最应该关注哪些性能指标

对于你当前 WeLM / MTP / MoE 优化任务，建议重点保留：

| 指标 | 中文 | 趋势 |
|---|---|---|
| TTFT | 首 Token 延迟 | 越低越好 |
| TPOT | 每输出 Token 延迟 | 越低越好 |
| Mean Latency | 平均端到端延迟 | 越低越好 |
| Output Throughput | 输出吞吐量 | 越高越好 |
| P99 TTFT / Latency | 尾延迟 | 越低越好 |
| Success Rate | 成功率 | 越高越好 |

如果你做的是：

```text
MTP 开启后
MoE 从普通路径改为 direct topk 路径
```

那么通常更值得观察：

```text
TPOT
Output Throughput
Latency
P99
```

因为这类优化主要发生在 Decode / MoE 执行路径。

---

# 25. 怎么判断优化是否真的有效

不要只看一项数据。

例如修改前：

```text
TPOT = 20 ms
Throughput = 1000 tok/s
P99 = 4 s
```

修改后：

```text
TPOT = 19 ms
Throughput = 1020 tok/s
P99 = 8 s
```

不能简单说“性能提升”。

因为：

```text
平均性能略微提升
但尾延迟恶化一倍
```

更合理的方法是横向比较：

| 配置 | TTFT | TPOT | Mean | Throughput | P99 |
|---|---:|---:|---:|---:|---:|
| 修复前 + MTP |  |  |  |  |  |
| 修复后 + MTP |  |  |  |  |  |
| 修复后 + No MTP |  |  |  |  |  |

而且必须确保三组：

```text
相同模型
相同输入长度
相同输出长度
相同 batch / rate / parallel
相同服务器
相同卡数
相同启动参数
相同 warmup
相同请求数量
```

这样结果才有可比性。

---

# 26. `evalscope perf` 的结果在哪里

你指定了：

```bash
--outputs-dir WeLM-mlu-outputs
```

因此结果根目录会在你**执行脚本时的当前工作目录**下面：

```text
./WeLM-mlu-outputs/
```

最直接查看：

```bash
ls -lh WeLM-mlu-outputs
```

递归看：

```bash
find WeLM-mlu-outputs -maxdepth 4 -type f
```

当前 EvalScope 版本中，常见性能结果文件包括：

```text
benchmark.log
performance_summary.txt
```

其中单次配置最核心的结果一般写入：

```text
benchmark.log
```

可以查：

```bash
find WeLM-mlu-outputs -name "benchmark.log" -print
```

然后：

```bash
cat <实际路径>/benchmark.log
```

或：

```bash
less <实际路径>/benchmark.log
```

---

# 27. 为什么你的脚本可能有很多结果子目录

你不是一次执行：

```bash
evalscope perf --parallel 3 5 10 20 ...
```

而是 Bash 循环中反复执行多个独立的：

```bash
evalscope perf ...
```

也就是：

```text
rate=0.3 → 启动一次 evalscope perf
结束

rate=0.5 → 再启动一次 evalscope perf
结束

rate=1.0 → 再启动一次 evalscope perf
结束
...
```

所以每一档本质上是独立的测试任务。

这意味着：

> 你应该分别读取每次运行产生的 `benchmark.log`，再把多档 rate/parallel 汇总成自己的表格。

官方所说的 `performance_summary.txt` 更适合一次 `evalscope perf` 内直接传入多个并发配置时生成统一汇总；你的循环方式不一定会得到一份覆盖所有 rate 的统一 summary。

---

# 28. 怎么快速找结果

### 找所有 benchmark log

```bash
find WeLM-mlu-outputs -type f -name "benchmark.log"
```

### 找所有 summary

```bash
find WeLM-mlu-outputs -type f -name "performance_summary.txt"
```

### 查 TTFT

```bash
grep -R "TTFT" WeLM-mlu-outputs
```

### 查 TPOT

```bash
grep -R "TPOT" WeLM-mlu-outputs
```

### 查吞吐

```bash
grep -R "Throughput" WeLM-mlu-outputs
```

### 同时查核心指标

```bash
grep -RE "TTFT|TPOT|Latency|Throughput|P99" WeLM-mlu-outputs
```

---

# 29. 一个典型性能结果应该怎么看

假设看到：

```text
Test Duration                60.0 s
Concurrency                  40
Request Rate                 4.0 req/s
Total Requests               200
Successful Requests          200
Failed Requests              0

Avg Latency                  3.1 s
TTFT                         700 ms
TPOT                         24 ms
Output Throughput            330 tok/s
```

首先看：

```text
200 请求是不是全部成功？
```

如果失败很多，后面的性能数值没有太大意义。

然后看：

```text
TTFT
```

判断排队 + Prefill 是否变慢。

再看：

```text
TPOT
```

判断 Decode 是否变慢。

最后看：

```text
Output Throughput
```

判断整个服务器真正处理了多少输出 token。

再结合：

```text
P50 / P90 / P99
```

判断性能是否稳定。

---

# 30. 最重要的性能分析方法：看“趋势”而不是一个点

你的脚本逐渐增加：

```text
rate
parallel
```

理想情况：

```text
压力增加
↓
吞吐量继续增加
↓
TTFT稍微增加
↓
TPOT相对稳定
```

当到达饱和点后，常见现象：

```text
rate继续增加
↓
吞吐量不再明显增加
↓
请求开始排队
↓
TTFT快速增大
↓
P99快速恶化
```

可以画成：

```text
吞吐
 ^
 |             ─────────── 饱和
 |          /
 |       /
 |    /
 |___/____________________> 请求压力
```

同时 TTFT：

```text
TTFT
 ^
 |                 /
 |               /
 |             /
 |____________/___________> 请求压力
             ↑
           饱和点
```

因此：

> 真正需要找的是“服务达到最大有效吞吐，同时延迟还可以接受”的区间，而不是机械地追求最高并发。

---

# 31. `stream` 对 TTFT 非常重要

TTFT 的本质是：

```text
收到第一个 token 的时间
-
发送请求的时间
```

因此客户端必须能看到“第一个 token 什么时候回来”。

如果非流式：

```text
服务端生成完整答案
↓
一次性返回
```

客户端只能看到：

```text
整个答案什么时候结束
```

这时 TTFT 很可能退化为接近总 Latency，失去真正意义。

因此做 TTFT 性能测试时应该确认当前 EvalScope / API 调用启用了流式模式。

---

# 32. EvalScope 与服务端之间的职责边界

非常重要：

```text
EvalScope 不是推理引擎。
```

你当前系统可以理解为：

```text
┌────────────────────────────┐
│ EvalScope                  │
│                            │
│ 产生请求                    │
│ 控制 rate / parallel       │
│ 记录时间                    │
│ 计算 TTFT / TPOT / P99      │
│ 做 Accuracy 判分            │
└─────────────┬──────────────┘
              │ HTTP
              ↓
┌────────────────────────────┐
│ SGLang / SGLang-MLU        │
│                            │
│ Scheduler                  │
│ Continuous Batching        │
│ Prefill                    │
│ Decode                     │
│ MTP                        │
│ MoE                        │
│ direct topk                │
│ Graph                      │
│ Kernel                     │
└─────────────┬──────────────┘
              ↓
┌────────────────────────────┐
│ MLU / GPU                  │
└────────────────────────────┘
```

所以你修改：

```text
MTP
MoE
direct topk
stream
kernel
```

改的是服务端。

EvalScope 不需要理解这些内部实现。

它只负责：

```text
给服务端施加相同 workload
↓
测出最终表现
```

这也是为什么 EvalScope 很适合做前后性能回归。

---

# 33. 你的性能脚本中还有一个疑似语法错误

当前上传脚本里有这一行：

```bash
--rate "${rate}"  \vim bench_client.sh
```

这非常像你在编辑脚本时误把：

```bash
vim bench_client.sh
```

粘到了命令续行中。

正常应该类似：

```bash
--rate "${rate}" \
--number 200 \
```

建议检查：

```bash
sed -n '32,50p' bench_client.sh
```

如果文件中确实存在：

```bash
\vim bench_client.sh
```

应删除它，否则 EvalScope 很可能会把后面的内容解析成错误参数，甚至直接执行失败。

---

# 34. 运行测试前推荐先做的检查

### 1. 确认模型服务存在

```bash
ps -ef | grep -E "sglang|launch_server"
```

或者：

```bash
ss -lntp | grep 30000
```

如果看到：

```text
LISTEN ... :30000
```

说明端口已经被服务监听。

---

### 2. 确认接口能正常返回

先发一个简单请求。

如果简单请求都失败，就不要直接跑 200 × 多档性能压测。

---

### 3. 确认 EvalScope 版本

```bash
evalscope --version
```

---

### 4. 确认代码版本

```bash
git rev-parse HEAD
```

---

### 5. 确认模型启动参数

尤其是：

```text
MTP 开/关
Graph 开/关
TP/DP/EP
max_running_requests
mem_fraction
MoE backend
direct topk 路径
```

---

# 35. 精度测试和性能测试应该怎么组合

对于你现在这种推理框架优化任务，一个比较标准的流程是：

```text
修改前代码
   ↓
启动模型服务
   ↓
GSM8K 精度测试
   ↓
记录 Accuracy
   ↓
性能测试
   ↓
记录 TTFT / TPOT / Throughput / P99
   ↓
修改代码
   ↓
重新启动服务
   ↓
再次 GSM8K
   ↓
确认 Accuracy 没有明显下降
   ↓
再次性能测试
   ↓
比较性能
```

最终结论应该类似：

```text
功能正确性：
GSM8K Accuracy 与 baseline 基本一致。

性能：
TPOT 下降 x%
Output Throughput 提升 x%
P99 下降 x%

结论：
优化没有导致明显精度退化，同时 Decode 性能得到提升。
```

---

# 36. 三组 MTP / direct-topk 对比时推荐记录的表

如果你的任务是：

1. 修复前 + MTP；
2. 修复后 + MTP；
3. 修复后 + 不开 MTP；

建议最终整理：

| 配置 | GSM8K Acc | TTFT Mean | TTFT P99 | TPOT Mean | TPOT P99 | Mean Latency | Output Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| 修复前 + MTP |  |  |  |  |  |  |  |
| 修复后 + MTP |  |  |  |  |  |  |  |
| 修复后 + No MTP |  |  |  |  |  |  |  |

这样可以同时回答两个问题：

```text
1. 修复有没有影响正确性？
2. 修复有没有真正改善推理性能？
```

---

# 37. 最常用的结果查看命令

## GSM8K

```bash
cd /projs/solutionsdk/mahao/outputs/gsm8k_after_mtp1

find . -maxdepth 4 -type f
```

看 report：

```bash
find . -path "*/reports/*" -type f
```

格式化 JSON：

```bash
jq . reports/*/*.json
```

看 prediction：

```bash
find predictions -type f -name "*.jsonl"
```

看评测日志：

```bash
less logs/eval_log.log
```

---

## Perf

```bash
find WeLM-mlu-outputs -type f
```

查 benchmark：

```bash
find WeLM-mlu-outputs -name "benchmark.log" -print
```

查核心指标：

```bash
grep -RE "TTFT|TPOT|Latency|Throughput|P99" WeLM-mlu-outputs
```

---

# 38. EvalScope 可视化

EvalScope 也提供 Web Dashboard。

安装：

```bash
pip install "evalscope[service]"
```

启动：

```bash
evalscope service
```

一般默认可以通过：

```text
http://127.0.0.1:9000
```

查看。

如果 EvalScope 跑在远程服务器，而浏览器在你的本地电脑，则通常还需要：

- SSH 端口转发；
- 公司内部网络代理；
- 或允许远程访问对应端口。

对于日常开发，你直接看：

```text
JSON
JSONL
benchmark.log
```

通常已经够用；Dashboard 更适合做多轮结果对比和可视化。

---

# 39. 初学者最容易犯的几个错误

## 错误 1：服务没启动就直接跑 EvalScope

EvalScope 在 `openai_api` 模式不会替你启动模型。

正确顺序：

```text
launch 模型
↓
确认端口
↓
evalscope
```

---

## 错误 2：把 `eval` 和 `perf` 当成一回事

```text
eval
=
能力 / Accuracy

perf
=
TTFT / TPOT / Throughput
```

两者用途不同。

---

## 错误 3：只比较 Throughput

吞吐高不代表体验一定好。

必须结合：

```text
Throughput
TTFT
TPOT
P99
Success Rate
```

---

## 错误 4：前后测试 workload 不一致

例如：

```text
baseline：11k / 100
优化后：4k / 100
```

这种数据不能直接比较。

---

## 错误 5：只跑一次就下结论

性能测试会受到：

- 卡上其他进程；
- 温度/频率；
- CPU 调度；
- 网络；
- cache；
- Graph warmup；

等影响。

正式结论最好至少进行多次重复测试，再比较均值和波动。

---

## 错误 6：看到 TTFT 变高就认为 Prefill kernel 变慢

高并发场景下 TTFT 包含排队时间。

因此：

```text
TTFT 高
```

不一定等于：

```text
Prefill kernel 本身变慢
```

也有可能是：

```text
服务已经饱和
↓
请求在 scheduler 中排队
↓
TTFT 变高
```

这时应结合并发、吞吐和 profiler 一起分析。

---

# 40. 一句话理解 EvalScope

你可以把 EvalScope 记成：

> **EvalScope 是站在“客户端”角度测试大模型服务的工具：`evalscope eval` 用标准数据集检查模型答得对不对，`evalscope perf` 用可控请求压力检查模型服务跑得快不快；它负责产生请求、记录响应、判分和统计，而真正的 Prefill、Decode、MTP、MoE、Graph 和算子计算仍然发生在 SGLang/模型服务端。**

---

# 41. 结合你当前两个脚本的最终理解

你的整个系统实际上是：

```text
                     ┌────────────────────┐
                     │    WeLM Model      │
                     │  SGLang-MLU Server │
                     │   Port : 30000     │
                     └─────────┬──────────┘
                               ↑
                               │ OpenAI API
                ┌──────────────┴──────────────┐
                │                             │
        ┌───────┴────────┐           ┌────────┴────────┐
        │ evalscope eval │           │ evalscope perf  │
        └───────┬────────┘           └────────┬────────┘
                │                             │
             GSM8K                         random
                │                         11k / 100
                ↓                             ↓
          Accuracy                     TTFT / TPOT
                                      Latency / P99
                                       Throughput
```

因此：

```text
GSM8K
负责证明：
“代码改完以后模型还是正确的。”

Perf
负责证明：
“代码改完以后模型确实更快了。”
```

这就是 EvalScope 在你当前 WeLM 性能优化工作中的核心作用。

---

# 42. 参考资料

EvalScope 官方仓库：

https://github.com/modelscope/evalscope

EvalScope 官方文档：

https://evalscope.readthedocs.io/

能力评测快速上手：

https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html

性能压测快速上手：

https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/quick_start.html

参数说明：

https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html

常见问题：

https://evalscope.readthedocs.io/zh-cn/latest/get_started/faq.html


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-实习工程能力|模块-实习工程能力]]
- 关联阅读：[[outputs/项目整理/专题-06-服务部署评测与协作|专题-06-服务部署评测与协作]]

%% 项目关联导航：结束 %%
