# Qwen3.5 / Qwen3.6 架构疑问详细解析：结合 nano-vLLM-qwen3.6 代码理解

> 本文面向已经学过原版 `nano-vLLM` / Qwen3 dense 的读者。  
> 目标是回答你在学习 `Qwen3.5 架构与 nano-vLLM-qwen3.6 模型代码对应总览` 后产生的 10 个关键疑问。
>
> 重点不是背代码，而是把这些概念串成一条工程主线：
>
> ```text
> Qwen3 dense
>   ↓
> Qwen3.5/Qwen3.6 hybrid
>   ↓
> full attention + GatedDeltaNet 混合
>   ↓
> KV Cache + GDN state 双状态系统
>   ↓
> FP8 / MTP / Vision 等可选工程能力
> ```

---

# 0. 先建立一个总认知

在原版 `nano-vLLM` 中，Qwen3 dense 的心智模型很简单：

```text
input_ids
  ↓
Embedding
  ↓
DecoderLayer × N
  ↓
Final RMSNorm
  ↓
LM Head
  ↓
logits
  ↓
Sampler
```

每层大致是：

```text
RMSNorm
  ↓
Full Attention
  ↓
RMSNorm
  ↓
MLP
```

每一层都是 full attention，所以每一层都需要 KV Cache。

而在 `nano-vllm-qwen3.6` 中，你要把心智模型改成：

```text
input_ids
  ↓
Embedding
  ↓
Qwen3_5DecoderLayer × N
      ├── full_attention 层：Qwen3_5Attention，使用 KV Cache
      └── GDN 层：GatedDeltaNet，使用 conv/recurrent state
  ↓
Final GemmaRMSNorm
  ↓
LM Head
```

所以它不是“只改了某个 attention 细节”，而是从：

```text
所有层都是 attention
```

变成：

```text
一部分层是 attention，一部分层是 GatedDeltaNet
```

这也是为什么你看到很多文件都改了一点点：因为模型结构一变，调度、KV Cache、state slot、CUDA Graph、MTP、rollback 都要配合变。

---

# 1. Qwen3.6 相对于 Qwen3.5 增加或者改了哪些功能？为什么项目名是 3.6，但代码里很多内容都是 3.5？

## 1.1 先区分两个层面

这个问题最容易混淆，因为这里有两个“版本差异”：

```text
A. 官方模型版本差异：
   Qwen3.5 和 Qwen3.6 官方模型本身可能有训练、权重、配置、能力差异。

B. 这个 GitHub 项目的代码差异：
   nano-vllm-qwen3.6 这个 fork 里，代码怎么组织、怎么命名、支持了哪些推理能力。
```

我们现在主要讨论的是 **这个项目源码层面能看到的差异**。

从这个仓库的代码组织看，它没有单独写一个：

```text
qwen3_6.py
```

而是用：

```text
models/qwen3_5.py
```

来承载 Qwen3.5 / Qwen3.6 这类 hybrid 架构。

也就是说，`qwen3_5.py` 更像是：

```text
Qwen3.5/Qwen3.6 hybrid language model implementation
```

而不只是“只能跑 Qwen3.5”。

---

## 1.2 为什么项目名是 qwen3.6，但模型文件叫 qwen3_5.py？

原因可以这样理解：

```text
Qwen3.5 和 Qwen3.6 在这个项目里共享了一套核心 hybrid 模型代码。
```

这套代码包含：

```text
1. full_attention / GatedDeltaNet 混合层
2. Qwen3_5Attention
3. GemmaRMSNorm
4. InterleavedMRoPE
5. optional vision encoder
6. optional MTP
```

所以作者没有重复写一个几乎相同的 `qwen3_6.py`，而是复用 `qwen3_5.py`。

项目名叫 `nano-vllm-qwen3.6`，更强调的是：

```text
这个 fork 的目标之一是支持 Qwen3.6-27B-FP8 的推理实验。
```

而不是说：

```text
每个文件名都必须叫 qwen3_6。
```

---

## 1.3 从项目源码看，Qwen3.6 相对于 Qwen3.5 主要体现在哪些工程能力上？

从这个项目的代码和脚本看，Qwen3.6 相关能力主要体现在以下几个方向。

### 方向一：Qwen3.6 FP8 checkpoint 加载

Qwen3.6 运行脚本通常指向：

```text
Qwen3.6-27B-FP8
```

所以项目新增了：

```text
utils/quant.py
utils/loader.py 的 FP8 scale 读取
linear.py / embed_head.py 的 loaded_scale 支持
```

其作用是：

```text
磁盘上的 FP8 weight + weight_scale_inv
  ↓
加载时反量化
  ↓
得到 BF16 参数
  ↓
继续普通推理
```

注意，这不是 runtime FP8 GEMM，也不是 FP8 KV Cache，而是：

```text
FP8 checkpoint load-time dequantization
```

---

### 方向二：MTP 相关原型

Qwen3.6 相关脚本中出现了：

```text
test_mtp_forward.py
test_mtp1_verify.py
test_mtp1_spec_decode.py
test_mtp_spec_decode.py
run_mtp_fast_decode.py
bench_mtp_draft_sweep.py
```

这些都是围绕：

```text
MTP / multi-token prediction / draft token / speculative decoding
```

做实验。

核心新增模型文件是：

```text
models/qwen3_mtp.py
```

但是这个模块是可选的，不是普通 forward 必须走的路径。

---

### 方向三：状态回滚和 speculative decoding 验证

因为 MTP / speculative decoding 会产生：

```text
draft token 接受 / 拒绝
```

一旦拒绝，就必须回滚状态。

所以项目新增了：

```text
test_state_rollback.py
ModelRunner 中的 save_decode_state / restore_decode_state
Scheduler snapshot / restore
```

这对 Qwen3.6 hybrid 特别重要，因为它不只有 KV Cache，还有 GDN state。

---

## 1.4 一句话总结

在这个项目里：

```text
Qwen3.5 / Qwen3.6 共用 qwen3_5.py 这套 hybrid 模型代码；
项目名 qwen3.6 主要强调支持 Qwen3.6-27B-FP8 和 MTP 等实验能力；
3.6 相对 3.5 的主要差异更多体现在 checkpoint 格式、FP8 加载、MTP/spec decode 实验和运行脚本上，
而不是单独写了一套完全不同的 qwen3_6.py 模型文件。
```

---

# 2. “视觉 encoder 是可选模块”是什么意思？是被动可选，还是输入有图片就自动开启？

## 2.1 结论先说

“视觉 encoder 是可选模块”不是说：

```text
只要你输入图片，系统就一定自动开启视觉 encoder。
```

更准确地说，它是：

```text
模型构建时可选 + 运行时根据输入是否有图片决定是否调用。
```

它有两个条件：

```text
条件 1：模型初始化时创建了 self.visual
条件 2：forward 时真的传入了 pixel_values / image_grid_thw
```

只有两个条件都满足，视觉 encoder 才会运行。

---

## 2.2 代码层面怎么理解？

在 `Qwen3_5ForCausalLM` 里，大致逻辑是：

```python
self.visual = None

if vision_config is not None:
    self.visual = Qwen3VLVisionEncoder(vision_config)
```

forward 时：

```python
image_embeds = None

if pixel_values is not None and self.visual is not None:
    image_embeds = self.visual(pixel_values, image_grid_thw)
```

所以视觉 encoder 是否运行，取决于：

```text
pixel_values != None
并且
self.visual != None
```

---

## 2.3 什么情况下 self.visual 会存在？

一般取决于：

```text
1. 模型 config 里是否有 vision_config
2. LLM 初始化时是否允许 enable_vision
3. ModelRunner 创建模型时是否把 vision_config 传进去
```

如果你运行的是：

```python
LLM(..., enable_vision=False)
```

即使模型 checkpoint 可能有视觉相关配置，项目也可能不会创建视觉 encoder。

这就是为什么 Qwen3.6 FP8 text-only 脚本里通常会显式写：

```python
enable_vision=False
```

---

## 2.4 输入有图片时会发生什么？

如果入口支持多模态 messages，大致流程是：

```text
messages 中有图片
  ↓
image_processing.process_messages
  ↓
得到 token_ids + pixel_values + image_grid_thw
  ↓
Sequence 保存 pixel_values / image_grid_thw
  ↓
ModelRunner.prepare_prefill 整理多模态 batch
  ↓
Qwen3_5ForCausalLM.forward
  ↓
如果 self.visual 存在，则调用 vision encoder
```

也就是说，输入图片只是触发视觉路径的一个条件。

如果模型没有创建 visual encoder，单靠输入图片不能让视觉模块“凭空开启”。

---

## 2.5 为什么要设计成可选？

因为视觉 encoder 会带来额外成本：

```text
1. 需要加载视觉塔权重
2. 会占用显存
3. prefill 阶段要跑 Conv / Vision Attention / Patch Merger
4. 对 text-only 推理没有必要
```

所以对于 Qwen3.6-27B-FP8 text-only：

```text
禁用 vision 可以省显存、减少加载复杂度、降低运行风险。
```

---

## 2.6 一句话总结

视觉 encoder 是：

```text
构建模型时可选创建；
运行时有图片输入才会调用；
不是只要传图片就自动魔法开启。
```

可以记成：

```text
enable_vision / vision_config 决定“有没有视觉塔”
pixel_values / image_grid_thw 决定“这次 forward 用不用视觉塔”
```

---

# 3. “MTP 是可选模块”是什么意思？详细解释

## 3.1 MTP 是什么？

MTP 是：

```text
Multi-Token Prediction
```

可以理解为：

```text
主模型正常预测下一个 token；
MTP 模块尝试继续预测后面的 draft token。
```

普通自回归 decode 是：

```text
第 1 次 forward -> 生成 token 1
第 2 次 forward -> 生成 token 2
第 3 次 forward -> 生成 token 3
...
```

MTP 想做的是：

```text
主模型 forward -> 生成 token 1
MTP 辅助分支 -> 预测 token 2 / token 3 / ...
然后主模型再验证 draft token 是否正确
```

这和 speculative decoding 有关系。

---

## 3.2 MTP 在模型里长什么样？

在 `Qwen3_5ForCausalLM` 中：

```python
self.mtp = None

if getattr(config, "enable_mtp", False):
    self.mtp = Qwen3MTP(config)
```

也就是说：

```text
enable_mtp=False:
    self.mtp = None

enable_mtp=True:
    self.mtp = Qwen3MTP(config)
```

所以它不是默认必有模块。

---

## 3.3 MTP 为什么要可选？

因为 MTP 需要额外条件：

```text
1. checkpoint 里有 MTP 权重
2. config 里启用了 enable_mtp
3. ModelRunner 有对应调用路径
4. speculative decoding 逻辑能处理 draft / verify / rollback
```

如果某个模型没有 MTP 权重，强行启用会导致：

```text
权重缺失
shape 不匹配
或者虽然模型能创建，但输出没有意义
```

所以它必须是可选的。

---

## 3.4 MTP 普通 generate 会自动用吗？

不会。

普通 generate 路径一般是：

```text
ModelRunner.run
  ↓
model.forward
  ↓
compute_logits
  ↓
sampler
```

MTP 不在普通 forward 的必经路径里。

MTP 通常由专门方法调用，例如：

```text
run_mtp_probe
run_mtp_draft_step
run_mtp_draft_fast_step
```

也就是说：

```text
enable_mtp=True 只是让模型拥有 MTP 模块；
真正用不用 MTP，取决于 ModelRunner 是否调用 MTP 路径。
```

---

## 3.5 MTP 和 speculative decoding 的关系

MTP 只负责：

```text
生成 draft token
```

完整 speculative decoding 还需要：

```text
1. 用主模型 verify draft token
2. 判断 draft token 是否接受
3. 接受就提交多个 token
4. 拒绝就回滚状态
5. 重新走 trusted path
```

所以：

```text
MTP 是 speculative decoding 的候选生成器；
不是完整 speculative decoding 系统本身。
```

---

## 3.6 一句话总结

MTP 是可选模块的意思是：

```text
模型可以挂载一个额外的 draft-token 预测分支；
但是否创建由 enable_mtp / checkpoint 决定；
是否使用由 ModelRunner 的 MTP/spec decode 路径决定；
普通 generate 不一定会自动走 MTP。
```

---

# 4. Qwen3_5DecoderLayer 根据 config.layer_types[layer_idx] 动态选择，这个选择逻辑是什么？

## 4.1 先给结论

这里的“动态选择”不是指：

```text
每个 token 运行时自己选择走 attention 还是 GDN。
```

也不是指：

```text
每次请求根据输入内容选择。
```

它指的是：

```text
模型初始化时，根据 config.layer_types 这个列表，决定每一层固定是什么类型。
```

也就是说，选择是：

```text
per-layer static selection
```

不是：

```text
per-token dynamic routing
```

---

## 4.2 config.layer_types 是什么？

`config.layer_types` 可以理解为一个列表：

```python
config.layer_types = [
    "full_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
    ...
]
```

它的长度通常等于：

```text
num_hidden_layers
```

每个位置对应一层：

```text
layer_types[0] -> 第 0 层
layer_types[1] -> 第 1 层
layer_types[2] -> 第 2 层
...
```

---

## 4.3 DecoderLayer 初始化时怎么用？

在 `Qwen3_5DecoderLayer.__init__()` 中，核心逻辑是：

```python
layer_type = config.layer_types[layer_idx]

if layer_type == "full_attention":
    self.self_attn = Qwen3_5Attention(config, layer_idx)
    self.linear_attn = None
else:
    self.linear_attn = GatedDeltaNet(config, layer_idx)
    self.self_attn = None
```

解释一下：

```text
如果这一层的 layer_type 是 full_attention:
    创建 Qwen3_5Attention
    不创建 GatedDeltaNet

否则:
    创建 GatedDeltaNet
    不创建 Qwen3_5Attention
```

---

## 4.4 forward 时怎么走？

forward 里是：

```python
if self.self_attn is not None:
    hidden_states = self.self_attn(positions, hidden_states)
else:
    hidden_states = self.linear_attn(hidden_states)
```

也就是说：

```text
这一层初始化时是什么类型，forward 时就固定走什么类型。
```

---

## 4.5 这和 MoE routing 有什么区别？

这个机制容易和 MoE 混淆。

MoE 是：

```text
每个 token 运行时由 router 选择 expert
```

而这里是：

```text
每一层在模型结构里固定是 attention 或 GDN
```

所以它不是 token-level routing。

它是 layer-level architecture pattern。

---

## 4.6 为什么要这样设计？

因为 Qwen3.5/Qwen3.6 hybrid 架构不是所有层都用 full attention。

这样做可以：

```text
1. 保留一部分 full attention 层，维持全局上下文建模能力。
2. 用 GatedDeltaNet 层替代一部分 attention，减少对长 KV Cache 的依赖。
3. 让部分历史信息压缩进 recurrent state，而不是每层都保存完整 K/V。
```

---

## 4.7 这个选择会影响哪些系统模块？

一旦某层不是 full attention，就会影响：

```text
1. KV Cache 分配
2. GDN state 分配
3. Scheduler state_slot_id
4. Context.state_indices
5. CUDA Graph 输入
6. speculative decoding rollback
```

原因是：

```text
full_attention 层需要 KV Cache；
GDN 层需要 recurrent/conv state。
```

所以 `layer_types` 是模型结构和推理系统之间的关键桥梁。

---

# 5. 原版 Qwen3 dense 为什么把 q/k/v 打包成一个大线性层 qkv_proj，再 split？

## 5.1 原版写法

概念上，attention 需要：

```text
q = x @ Wq
k = x @ Wk
v = x @ Wv
```

如果直接写，就是三个线性层：

```python
q = q_proj(x)
k = k_proj(x)
v = v_proj(x)
```

原版 nano-vLLM 为了推理效率，把它合并成：

```python
qkv = qkv_proj(x)
q, k, v = qkv.split(...)
```

---

## 5.2 为什么可以合并？

因为三个线性层的输入一样，都是：

```text
hidden_states
```

三个线性层本质都是矩阵乘法：

```text
q = x @ Wq
k = x @ Wk
v = x @ Wv
```

把权重拼起来：

```text
Wqkv = [Wq, Wk, Wv]
```

就可以一次算：

```text
qkv = x @ Wqkv
```

然后再切回 q/k/v。

---

## 5.3 合并的好处一：减少 kernel launch

GPU 上每调用一次矩阵乘法，CPU 都要发起一个 kernel。

三个小 GEMM：

```text
q_proj
k_proj
v_proj
```

会有三次 kernel launch。

合并后：

```text
qkv_proj
```

只需要一次更大的 GEMM。

大 GEMM 通常更容易把 GPU 利用起来。

---

## 5.4 合并的好处二：减少读 hidden_states 的次数

如果分开算：

```text
q_proj 读一次 hidden_states
k_proj 再读一次 hidden_states
v_proj 再读一次 hidden_states
```

合并后：

```text
qkv_proj 只读一次 hidden_states
```

对推理来说，显存带宽经常是瓶颈，少读几次大 tensor 就很重要。

---

## 5.5 合并的好处三：更适合 Tensor Parallel

QKV projection 通常是 Column Parallel：

```text
按输出维切分
```

合并后可以统一管理：

```text
当前 rank 负责 q/k/v 的哪一段输出
```

loader 也可以把 HuggingFace 中分开的：

```text
q_proj.weight
k_proj.weight
v_proj.weight
```

加载到 nano-vLLM 合并后的：

```text
qkv_proj.weight
```

---

## 5.6 为什么 Qwen3.6 又不这么做？

因为 Qwen3.6 的 q 分支变成：

```text
q_proj -> query + gate
```

q 的输出维度不再只是普通 query，而是：

```text
2 * num_q_heads * head_dim
```

如果继续强行合并 q/k/v，就要处理：

```text
q_gate + k + v
```

这当然也能做，但 loader、切片、shape、gate split 都更复杂。

所以 qwen3.6 代码选择：

```text
q_proj / k_proj / v_proj 分开
```

以更清晰地表达模型结构。

---

# 6. Qwen3.6 代码中为什么要给 q 加 gate？不加有影响吗？这是模型规定的吗？

## 6.1 先给结论

这是模型结构规定的，不是 nano-vLLM 作者随便加的优化。

如果不加 gate，会有两个直接后果：

```text
1. q_proj 权重 shape 对不上，checkpoint 无法按原结构加载。
2. attention 数值逻辑不一致，输出会偏离模型训练时的结构。
```

所以它不是可有可无的小技巧，而是模型 forward 的一部分。

---

## 6.2 gate 到底是什么？

Qwen3.6 full attention 中：

```python
q_gate = q_proj(hidden_states)
q, gate = q_gate.chunk(2, dim=-1)
```

也就是说，q_proj 同时产生：

```text
query
gate
```

其中：

```text
query 用于 attention score
gate 用于调制 attention output
```

后面：

```python
o = Attention(q, k, v)
o = o * sigmoid(gate)
```

---

## 6.3 为什么 gate 和 q 放在一起输出？

因为 gate 的 shape 要和 attention 输出对齐。

attention 输出在 o_proj 前的形状是：

```text
[T, num_q_heads, head_dim]
```

gate 也需要是：

```text
[T, num_q_heads, head_dim]
```

而 q_proj 本来就输出：

```text
[T, num_q_heads, head_dim]
```

所以让 q_proj 输出：

```text
[T, num_q_heads, 2 * head_dim]
```

再 split 成：

```text
q
gate
```

是很自然的设计。

---

## 6.4 gate 的作用是什么？

普通 attention 是：

```text
从上下文取回信息 o
```

gated attention 是：

```text
从上下文取回信息 o
再由 gate 决定 o 的每个通道保留多少
```

可以理解为：

```text
Attention:
    负责“找信息”

Gate:
    负责“控制信息通过强弱”
```

公式是：

```text
output = o_proj( Attention(q, k, v) * sigmoid(gate) )
```

---

## 6.5 不加 gate 会怎样？

如果你强行去掉 gate，大概会出现三类问题。

### 问题一：权重 shape 不匹配

原模型的 q_proj 权重是按：

```text
2 * num_q_heads * head_dim
```

训练的。

你如果改回普通 q_proj：

```text
num_q_heads * head_dim
```

权重加载会直接不匹配。

---

### 问题二：即使裁掉 gate，也会破坏数值语义

假设你强行只取前半部分当 q，把 gate 丢掉：

```text
o = Attention(q, k, v)
```

这和模型训练时的：

```text
o = Attention(q, k, v) * sigmoid(gate)
```

不一样。

模型在训练时已经学会了依赖 gate 控制 attention 输出。推理时去掉 gate，会改变每层输出分布。

---

### 问题三：后续层输入分布会层层偏移

Transformer 是深层网络。

一层 attention 输出变了，后面：

```text
post norm
MLP
下一层 attention/GDN
...
```

都会受影响。

最终 logits 会明显偏离。

---

## 6.6 一句话总结

Qwen3.6 给 q 加 gate 是模型结构的一部分。

它的作用是：

```text
让 query 分支同时产生 attention 查询向量和输出门控向量；
attention 负责取信息，gate 负责控制信息通过多少。
```

不加 gate 不是“少一个优化”，而是“不再是同一个模型”。

---

# 7. 原版 q 和 k 为什么要做 norm 之后再做 RoPE？为什么 v 不做？

## 7.1 Attention 中 q/k/v 分别做什么？

先回到 attention 公式：

```text
score = q · k
weight = softmax(score)
output = weight · v
```

三者作用不同：

```text
q:
    当前 token 拿着 query 去问“我要关注谁？”

k:
    历史 token 提供 key，表示“我有什么特征可以被匹配？”

v:
    历史 token 提供 value，表示“如果你关注我，我给你什么内容？”
```

简单说：

```text
q/k 决定注意力权重；
v 提供被加权汇总的内容。
```

---

## 7.2 为什么 q/k 要 norm？

q/k 的点积决定 attention score：

```text
score = q · k
```

如果 q/k 的尺度很大，score 会很大，softmax 可能变得非常尖锐。

如果 q/k 的尺度很小，score 会很小，softmax 可能接近平均分布。

所以 q/k norm 的作用是：

```text
稳定 attention score 的数值尺度。
```

这对深层模型尤其重要。

---

## 7.3 为什么 norm 后再 RoPE？

原版常见顺序是：

```text
q/k projection
  ↓
q_norm / k_norm
  ↓
RoPE
  ↓
attention
```

这样做的直觉是：

```text
先把 q/k 的内容向量尺度稳定下来；
再根据位置对它们做旋转；
最后用旋转后的 q/k 做点积。
```

RoPE 本质上是对 q/k 的部分维度做旋转。它主要改变的是：

```text
q 和 k 之间的相对角度关系
```

从而让点积中带有相对位置信息。

如果先 RoPE 再 norm，在某些情况下也许数学上部分接近，但代码必须和模型训练结构一致。

模型训练时用的是：

```text
norm -> RoPE
```

推理时就必须保持同样顺序。

---

## 7.4 v 为什么不做 RoPE？

因为位置编码主要影响：

```text
谁关注谁
```

而不是：

```text
被取回的内容本身是什么
```

在 attention 里：

```text
q/k 决定 attention 权重；
v 是被权重加权求和的内容。
```

RoPE 加在 q/k 上后，attention score 会包含位置信息，模型就能决定：

```text
当前位置应该关注前面哪个位置。
```

如果给 v 也加 RoPE，就会把位置信息混入 value 内容本身。

这不是标准 RoPE attention 的设计，也会改变模型训练时学到的表示。

---

## 7.5 v 为什么通常不做 q/k norm？

v 不参与点积 score 的计算。

v 的尺度当然也会影响输出，但它通过：

```text
output projection
residual
norm
MLP
```

等后续结构处理。

q/k norm 的主要目标是稳定：

```text
q · k
```

所以只对 q/k 做 norm 是合理的。

---

## 7.6 一句话总结

```text
q/k 决定注意力分布，所以要 norm 稳定点积尺度，并通过 RoPE 注入位置关系；
v 是被加权汇总的内容，不负责计算注意力位置关系，所以通常不做 RoPE。
```

---

# 8. 为什么要做 partial RoPE？全部编码不是更好吗？

## 8.1 partial RoPE 是什么？

标准 RoPE 可以理解为：

```text
head_dim 的所有维度都参与位置旋转。
```

partial RoPE 是：

```text
只有前 rotary_dim 维参与 RoPE；
剩下维度不旋转，直接保留。
```

例如：

```text
head_dim = 128
partial_rotary_factor = 0.25
rotary_dim = 32
```

则：

```text
前 32 维：做 RoPE
后 96 维：不做 RoPE
```

---

## 8.2 为什么不是全部做 RoPE？

直觉上你会觉得：

```text
所有维度都有位置信息，不是更强吗？
```

但实际模型结构不是越多越好。

因为 hidden/head 向量中不仅要表达：

```text
位置信息
```

还要表达：

```text
语义内容
特征匹配
上下文关系
任务信息
```

如果所有维度都被位置旋转约束，模型可能失去一部分纯内容表达空间。

partial RoPE 的设计相当于把 head_dim 分成两部分：

```text
一部分维度负责位置感知；
一部分维度保留更自由的内容表达。
```

---

## 8.3 partial RoPE 对多模态为什么重要？

Qwen3.6 使用的是 InterleavedMRoPE。

多模态 image token 有三维位置：

```text
temporal
height
width
```

这些位置信息要被编码到 q/k 的 rotary 部分。

如果所有维度都参与复杂的多维位置编码，可能会让位置编码过强，也会改变模型结构约束。

partial RoPE 可以让：

```text
部分维度承载 T/H/W 位置结构；
剩余维度保留内容和语义空间。
```

---

## 8.4 更重要的是：必须和训练结构一致

对于推理框架来说，最重要的不是你觉得“全部编码更好”，而是：

```text
模型训练时怎么定义，推理时就必须怎么执行。
```

如果 checkpoint 是按 partial RoPE 训练的，你推理时改成 full RoPE，会导致：

```text
q/k 表示方式和训练时不一致
attention score 改变
输出分布偏移
```

所以 partial RoPE 是模型架构规定，不是推理时可以随便改的开关。

---

## 8.5 一句话总结

partial RoPE 的意义是：

```text
让一部分 head 维度承载位置编码，另一部分维度保留内容表达空间；
同时适配模型训练时的结构设定，尤其适合多模态 MRoPE。
```

全部编码不一定更好，因为模型不是单纯缺位置信息，而是在位置、内容、语义之间做结构平衡。

---

# 9. 为什么要把 RMSNorm 改为 GemmaRMSNorm？

## 9.1 普通 RMSNorm 是什么？

普通 RMSNorm 可以理解为：

```text
y = norm(x) * weight
```

其中：

```text
weight 初始值通常接近 1
```

---

## 9.2 GemmaRMSNorm 是什么？

GemmaRMSNorm 的形式是：

```text
y = norm(x) * (1 + weight)
```

其中：

```text
weight 初始值通常是 0
```

所以两者初始输出都可以接近：

```text
norm(x)
```

但参数语义不同。

普通 RMSNorm：

```text
weight 本身就是缩放系数。
```

GemmaRMSNorm：

```text
weight 是相对于 1 的偏移量。
```

---

## 9.3 为什么不能随便替换？

假设 checkpoint 中的 norm 权重是按照 GemmaRMSNorm 训练的。

也就是说，训练时实际缩放是：

```text
1 + weight
```

如果你推理时错误地用普通 RMSNorm：

```text
weight
```

那缩放因子会直接少 1。

例如某个通道：

```text
checkpoint weight = 0.02
```

GemmaRMSNorm 实际缩放：

```text
1.02
```

普通 RMSNorm 实际缩放：

```text
0.02
```

这差异巨大。

所以这不是小误差，而是完全不同的参数解释。

---

## 9.4 为什么 Qwen3.6 代码使用 GemmaRMSNorm？

因为这个项目中 Qwen3.5/Qwen3.6 hybrid 模型代码假设对应 checkpoint 的 norm 权重语义是：

```text
norm(x) * (1 + weight)
```

所以需要使用 GemmaRMSNorm。

它出现在：

```text
1. attention 中 q_norm / k_norm
2. DecoderLayer 的 input_layernorm
3. DecoderLayer 的 post_attention_layernorm
4. 模型 final norm
5. MTP 模块中若干 norm
```

---

## 9.5 这是不是说明它是 Gemma 模型？

不是。

这里叫 `GemmaRMSNorm`，是因为这种 norm 形式和 Gemma 风格类似。

它不代表这个模型就是 Gemma。

更准确地说：

```text
GemmaRMSNorm 是一种 RMSNorm 参数化方式；
Qwen3.6 代码复用了这种实现来匹配 checkpoint。
```

---

## 9.6 一句话总结

RMSNorm 改为 GemmaRMSNorm 的核心原因是：

```text
checkpoint 的 norm 权重语义变了。
```

如果权重是按：

```text
norm(x) * (1 + weight)
```

训练的，推理时就必须用 GemmaRMSNorm。

否则即使代码能跑，数值也不是同一个模型。

---

# 10. GDN 层要额外分配 conv_states 和 recurrent_states，这和 KV Cache 分配的 block 有什么区别？详细分析 GDN 分配结构

## 10.1 先给结论

KV Cache 和 GDN state 都是“历史信息”，但它们保存历史的方式完全不同。

KV Cache 是：

```text
把历史每个 token 的 K/V 都保存下来。
```

GDN state 是：

```text
把历史信息压缩进固定大小的 recurrent/conv 状态里。
```

所以它们的核心区别是：

```text
KV Cache 随序列长度增长；
GDN state 通常随请求数增长，不随 token 数线性增长。
```

---

## 10.2 KV Cache 保存的是什么？

在 full attention 层中，每来一个 token，模型会计算：

```text
k_t
v_t
```

然后把它们写入 KV Cache。

对一个 attention 层来说，KV Cache 可以理解为：

```text
K history:
    token 0 的 k
    token 1 的 k
    token 2 的 k
    ...

V history:
    token 0 的 v
    token 1 的 v
    token 2 的 v
    ...
```

decode 时，当前 token 的 q 会和历史所有 k 做 attention：

```text
q_t 和 [k_0, k_1, ..., k_t] 做点积
```

再用 attention weight 加权所有 v：

```text
output = sum(weight_i * v_i)
```

所以 KV Cache 是：

```text
per-layer
per-token
per-head
per-head_dim
```

的历史缓存。

---

## 10.3 KV Cache 为什么要用 block？

因为不同请求长度不同，而且会不断增长。

如果为每个请求直接分配连续大数组，会有很多问题：

```text
1. 不同请求长度不同，浪费显存
2. 请求结束后释放不方便
3. continuous batching 中请求不断进出
4. prefix cache 需要复用部分历史 token
```

所以 nano-vLLM 使用 block 管理：

```text
block_size = 例如 16 或 256
每个 block 存一段 token 的 K/V
Sequence.block_table 记录这个请求用了哪些 block
```

可以理解为：

```text
seq A:
    block_table = [3, 7, 10]

seq B:
    block_table = [2, 5]
```

每个请求的历史 token 分散在若干 block 中。

decode attention 通过：

```text
block_tables
context_lens
slot_mapping
```

找到历史 K/V。

---

## 10.4 GDN 层保存的是什么？

GDN 是 GatedDeltaNet。

它不是 full attention，所以它通常不需要保存每个历史 token 的 K/V。

它维护的是更像 RNN / state-space / recurrent 模块的状态：

```text
conv_state:
    保存局部卷积/短窗口相关状态

recurrent_state:
    保存经过递推更新后的长期历史摘要
```

你可以把它理解成：

```text
GDN 不回看所有历史 token；
它把历史不断压缩进一个状态里。
```

每来一个 token：

```text
旧 state + 当前 token
  ↓
GDN 更新
  ↓
新 state
```

下一步 decode 只需要新 state，不需要所有历史 token 的 K/V。

---

## 10.5 conv_state 是什么？

`conv_state` 可以理解为：

```text
短期局部历史缓存
```

GDN 这类结构中通常会有一段短卷积或局部 mixing 逻辑，它需要记住最近若干个 token 的局部特征。

所以 conv_state 负责保存类似：

```text
最近几个 token 的卷积输入缓存
```

它有点像一个滑动窗口。

特征是：

```text
1. 它不是完整序列历史。
2. 它通常只保存短窗口。
3. 它用于局部时序/卷积计算。
```

---

## 10.6 recurrent_state 是什么？

`recurrent_state` 可以理解为：

```text
长期历史摘要
```

每个新 token 到来时，GDN 会用当前 token 更新 recurrent_state。

它的特点是：

```text
1. 不保存每个历史 token。
2. 通过递推公式不断更新。
3. 当前 state 中已经压缩了之前 token 的信息。
```

这和 KV Cache 最大不同：

```text
KV Cache:
    历史 token 都还在

recurrent_state:
    历史 token 被压缩成状态
```

---

## 10.7 GDN state 如何分配？

因为 GDN state 是每个请求都要有一份，所以项目引入了：

```text
state_slot_id
StateSlotManager
state_indices
```

大致结构是：

```text
StateSlotManager:
    维护一批空闲 state slot

Sequence:
    保存自己的 state_slot_id

ModelRunner:
    分配 conv_states / recurrent_states 张量

Context:
    保存 state_indices，告诉 GDN 当前 batch 每个 token 对应哪个 state slot
```

可以理解为：

```text
max_state_slots = 最多同时支持多少个请求的 GDN state

seq A:
    state_slot_id = 0

seq B:
    state_slot_id = 1

seq C:
    state_slot_id = 2
```

GDN 层运行时，根据：

```text
state_indices = [0, 1, 2]
```

知道当前 batch 中每个请求应该读写哪一份 state。

---

## 10.8 GDN state 和 KV block 的对比表

| 对比项 | KV Cache block | GDN state slot |
|---|---|---|
| 服务对象 | full attention 层 | GatedDeltaNet 层 |
| 保存内容 | 每个历史 token 的 K/V | 压缩后的 conv/recurrent 状态 |
| 是否随序列长度增长 | 是 | 通常不是线性增长 |
| 分配单位 | block | state slot |
| Sequence 中记录 | `block_table` | `state_slot_id` |
| Runtime 索引 | `block_tables`、`slot_mapping` | `state_indices` |
| 是否适合 prefix cache | 适合，KV 可按 token/block 复用 | 不天然适合，state 是递推结果 |
| decode 更新方式 | 写入当前 token 的 K/V | 原地更新 recurrent/conv state |
| rollback 难点 | 恢复 KV block 内容和 block table | 恢复 state slot 中的 conv/recurrent 张量 |
| 生命周期 | 请求增长时不断追加 block，请求结束释放 | 请求开始分配 slot，请求结束释放 slot |

---

## 10.9 为什么 GDN state 不能像 KV Cache 一样做 prefix cache？

KV Cache 的 prefix cache 可以这样理解：

```text
两个请求前面 token 一样
  ↓
它们前面若干 block 的 K/V 完全一样
  ↓
可以复用这些 KV block
```

但 GDN state 是递推状态：

```text
state_t = f(state_{t-1}, token_t)
```

它不是简单的 per-token block 列表。

如果你只复用 attention KV prefix，但没有同步复用对应的 GDN recurrent/conv state，就会出现：

```text
Attention 层认为历史已经有了；
GDN 层状态却没有对应历史。
```

这样模型内部状态不一致，生成就会错。

所以 hybrid 模型中 prefix cache 需要非常谨慎。很多简化实现会直接禁用 hybrid 下的 prefix cache，避免 KV 和 GDN state 不一致。

---

## 10.10 GDN state 在 prefill 和 decode 中怎么变化？

### Prefill 阶段

prefill 一次处理 prompt 的多个 token。

对于 GDN 层：

```text
初始 state
  ↓
依次处理 prompt token
  ↓
得到 prompt 结束后的最终 state
```

这个最终 state 会保存在该请求的 state slot 里。

之后 decode 就从这个状态继续。

---

### Decode 阶段

decode 每次处理一个新 token。

对于 GDN 层：

```text
读取 seq.state_slot_id 对应的 state
  ↓
用当前 token 更新 state
  ↓
把新 state 写回同一个 slot
```

所以 decode 中 GDN state 是持续原地更新的。

---

## 10.11 为什么 speculative decoding 必须保存和回滚 GDN state？

MTP/spec decode 中会发生：

```text
先尝试 draft token
再用主模型 verify
如果 draft 被接受，提交
如果 draft 被拒绝，回滚
```

如果你在尝试 draft/verify 时更新了 GDN state，但最后 draft 被拒绝，那么必须恢复到尝试前的 GDN state。

否则后续 decode 会从错误状态继续。

因此 rollback 需要保存：

```text
1. KV Cache 对应位置
2. GDN conv_states
3. GDN recurrent_states
4. Sequence token_ids
5. Scheduler queue
6. block_table
7. state_slot_id / free state slots
```

这就是为什么 `test_state_rollback.py` 在这个项目里非常重要。

---

## 10.12 一个直观类比

可以这样类比：

```text
KV Cache 像一本完整笔记：
    每个历史 token 的 K/V 都写在纸上。
    要查历史时，可以翻到任何一页。

GDN recurrent_state 像一个滚动摘要：
    每读一个 token，就把摘要更新一下。
    后面不保存完整原文，只保存摘要。

GDN conv_state 像最近几句话的短期缓存：
    为局部卷积或短窗口计算服务。
```

所以：

```text
KV Cache 更精确但随长度增长；
GDN state 更紧凑但需要严格维护更新顺序和状态一致性。
```

---

# 11. 把 10 个问题串成一条主线

现在把前面的概念串起来。

原版 Qwen3 dense 是：

```text
每层 full attention
  ↓
每层都有 qkv_proj
  ↓
每层都有 KV Cache
  ↓
RoPE 只处理文本 1D positions
  ↓
普通 RMSNorm
```

qwen3.6 项目变成：

```text
每层先看 layer_types
  ↓
如果是 full_attention:
      q_proj/k_proj/v_proj 分开
      q_proj 输出 q + gate
      q/k 用 GemmaRMSNorm
      q/k 用 InterleavedMRoPE
      attention 输出乘 sigmoid(gate)
      使用 KV Cache

  如果是 GDN:
      使用 GatedDeltaNet
      不使用 KV Cache
      使用 conv_state + recurrent_state

  不管哪种层:
      后面都接 Qwen3_5MLP
```

由于有 GDN：

```text
Scheduler 要分配 state_slot_id
ModelRunner 要分配 GDN states
Context 要传 state_indices
spec decode 要保存/回滚 GDN state
```

由于有 Qwen3.6 FP8：

```text
loader 要读 weight_scale_inv
quant 要做 block-wise dequant
linear/embed_head 要支持 loaded_scale
```

由于有 MTP：

```text
模型可选挂载 Qwen3MTP
ModelRunner 可选调用 draft/probe 路径
spec decode 要 verify 和 rollback
```

由于有 vision：

```text
模型可选创建 visual encoder
有图片输入时才产生 pixel_values/image_grid_thw
forward 时如果 self.visual 存在才运行视觉塔
```

---

# 12. 最终总结

你现在需要形成的核心理解是：

```text
Qwen3.6 这个项目的复杂性，不在某一行代码，而在“模型结构变化牵动整个推理系统”。
```

最关键的变化是：

```text
Qwen3 dense:
    所有层 full attention
    每层 KV Cache
    qkv_proj 合并
    标准 RoPE
    普通 RMSNorm

Qwen3.5/Qwen3.6 hybrid:
    layer_types 决定每层是 full attention 还是 GDN
    full attention 使用 gated attention
    GDN 使用 conv/recurrent state
    q/k 使用 GemmaRMSNorm
    RoPE 使用 InterleavedMRoPE
    vision/MTP 是可选模块
```

最重要的工程后果是：

```text
你不能只看 qwen3_5.py。
```

必须把它和下面这些文件一起理解：

```text
model_runner.py:
    准备 input_ids / positions / KV cache / GDN state / MTP / vision 输入

scheduler.py:
    分配 KV block 和 state slot

sequence.py:
    保存 block_table、state_slot_id、pixel_values、image_grid_thw

context.py:
    传递 block_tables、slot_mapping、state_indices

block_manager.py:
    管理 KV cache block，hybrid 下要小心 prefix cache

gated_delta_net.py:
    使用 conv/recurrent state 处理非 attention 层

loader.py / quant.py / linear.py:
    支持 FP8 checkpoint 和 TP 权重加载
```

一句话记住：

```text
Qwen3.6 不是在原版 Qwen3 dense 上“小修小补”，
而是把模型从“纯 attention decoder”扩展成“attention + GDN 混合状态机”，
同时加入 FP8、MTP、vision 等实验能力。
```


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
