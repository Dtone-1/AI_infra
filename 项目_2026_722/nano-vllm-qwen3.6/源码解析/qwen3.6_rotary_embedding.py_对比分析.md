# qwen3.6_rotary_embedding.py_对比分析

> 对比对象：  
> 1. 原版 `nano-vLLM` 的 `rotary_embedding.py`  
> 2. `nano-vllm-qwen3.6` 中对应的 `rotary_embedding.py`
>
> 本文目标：从整体工程角度分析 qwen3.6 版本相比原版 `rotary_embedding.py` 做了哪些修改、为什么这样改，以及这些修改在 Qwen3.6 / 多模态 / 推理系统中的作用。

---

## 1. 文件整体定位

`rotary_embedding.py` 是 nano-vLLM 中负责 **旋转位置编码 RoPE** 的文件。

在 Qwen / LLaMA / decoder-only Transformer 中，位置编码通常不会直接加到 embedding 上，而是在 Attention 内部对 `q` 和 `k` 做旋转变换：

```text
hidden_states
  ↓
q_proj / k_proj / v_proj
  ↓
q, k 加 RoPE
  ↓
Attention(q, k, v)
```

这个文件的作用就是：

```text
根据 positions 生成 cos/sin
把 cos/sin 应用到 query/key 上
让模型获得 token 的位置信息
```

它位于模型 forward 的 Attention 子层内部，通常在：

```text
QKV projection 之后
FlashAttention / KV Cache Attention 之前
```

---

## 2. qwen3.6 版本整体变化概览

相比原版 `rotary_embedding.py`，qwen3.6 版本主要做了四类改造：

| 改动点 | 原版 nano-vLLM | qwen3.6 版本 | 作用 |
|---|---|---|---|
| RoPE 维度 | 强制 `rotary_dim == head_size` | 支持 `rotary_dim < head_size` | 支持 partial RoPE |
| 标准 RoPE | 对整个 head_dim 做旋转 | 只对 rotary_dim 部分旋转，其余维度直通 | 适配更多模型配置 |
| 多模态 MRoPE | 无 | 新增 `InterleavedMRoPE` | 支持 Qwen VL 类 3D 位置编码 |
| positions 形态 | 只支持 1D positions | 支持 1D 和 3D positions | 文本和图像 token 统一处理 |
| RoPE cache | `@lru_cache(1)` | `@lru_cache(maxsize=4)` | 支持更多 RoPE 实例缓存 |
| get_rope | 只返回标准 RotaryEmbedding | 仍返回 RotaryEmbedding，但文件中新增 MRoPE 类 | MRoPE 可能由模型层直接实例化 |

一句话总结：

**qwen3.6 版本的 `rotary_embedding.py` 从“纯文本标准 RoPE”扩展成了“支持 partial RoPE，并提供多模态 3D MRoPE 能力”的位置编码组件。**

---

## 3. 原版 rotary_embedding.py 的核心设计

原版文件结构非常简单：

```text
apply_rotary_emb
RotaryEmbedding
get_rope
```

原版 `RotaryEmbedding` 有一个重要限制：

```python
assert rotary_dim == head_size
```

这说明原版默认：

```text
每个 attention head 的所有维度都参与 RoPE 旋转
```

也就是：

```text
query: [N, num_heads, head_size]
key:   [N, num_kv_heads, head_size]

整个 head_size 都做 RoPE
```

原版只适合标准文本 decoder-only 模型中常见的 1D position：

```text
positions: [N]
```

它不能处理：

```text
rotary_dim < head_size
positions: [3, N]
多模态时间/高度/宽度位置
```

---

## 4. 保持不变的部分：apply_rotary_emb

原版和 qwen3.6 版本都保留了：

```python
def apply_rotary_emb(x, cos, sin):
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)
```

这个函数是 RoPE 真正应用旋转的位置。

它的含义是：

```text
把 hidden 向量最后一维切成两半
对两半做二维旋转
再拼回原来的维度
```

可以理解为：

```text
RoPE 不是把 position embedding 加到 hidden_states 上，
而是用 position 决定的 cos/sin 去旋转 q/k 向量。
```

这部分 qwen3.6 没有改变，说明底层 RoPE 数学形式仍然沿用原版。

---

## 5. 改动一：RotaryEmbedding 支持 partial RoPE

### 5.1 原版限制

原版初始化中有：

```python
assert rotary_dim == head_size
```

这表示：

```text
rotary_dim 必须等于 head_size
```

也就是说整个 head 维度都参与旋转。

### 5.2 qwen3.6 版本修改

qwen3.6 版本删除了这个 assert，并新增：

```python
self.rotary_dim = rotary_dim
self.is_partial = rotary_dim < head_size
```

这表示它允许：

```text
rotary_dim < head_size
```

也就是只对 head 的前一部分维度做 RoPE。

---

### 5.3 partial RoPE 是什么

假设：

```text
head_size = 128
rotary_dim = 64
```

那么一个 q/k head 向量会被拆成：

```text
前 64 维：参与 RoPE 旋转
后 64 维：不参与旋转，直接保留
```

qwen3.6 代码中体现为：

```python
q_rot = query[..., :self.rotary_dim]
q_pass = query[..., self.rotary_dim:]
...
q_rot = apply_rotary_emb(q_rot, cos, sin)
query = torch.cat((q_rot, q_pass), dim=-1)
```

`q_rot` 是旋转部分，`q_pass` 是直通部分。

---

### 5.4 为什么需要 partial RoPE

不同模型的 RoPE 配置并不完全一样。

有些模型不是对整个 head_dim 都做旋转，而是只对一部分维度做 RoPE。

如果推理框架强制：

```text
rotary_dim == head_size
```

就无法正确支持这类模型。

所以 qwen3.6 版本加入 partial RoPE，使 RoPE 模块更通用。

---

## 6. 改动二：RotaryEmbedding.forward 支持旋转部分和直通部分

### 6.1 原版 forward

原版直接：

```python
query = apply_rotary_emb(query, cos, sin)
key = apply_rotary_emb(key, cos, sin)
```

含义是：

```text
query/key 整个 head_dim 都做旋转
```

### 6.2 qwen3.6 版本 forward

qwen3.6 版本分两种情况：

```text
如果 is_partial:
    只旋转前 rotary_dim
    后面维度直接拼回

否则:
    和原版一样整段旋转
```

这使得同一个类同时兼容：

```text
全维度 RoPE
partial RoPE
```

### 6.3 工程意义

这是一种兼容性增强。

原版实现更简单，但假设更强。

qwen3.6 版本把假设放宽，让模型配置可以决定：

```text
rotary_dim 是否等于 head_size
```

这对于支持不同 Qwen 版本、多模态模型或其他兼容模型都更灵活。

---

## 7. 改动三：新增 InterleavedMRoPE

qwen3.6 版本新增了一个重要类：

```python
class InterleavedMRoPE(nn.Module):
```

注释说明：

```text
Interleaved Multi-dimensional Rotary Position Embedding for Qwen3.5 VL.
```

它的目标是支持：

```text
多维旋转位置编码 MRoPE
```

尤其用于多模态场景。

---

## 8. MRoPE 和普通 RoPE 的区别

### 8.1 普通 RoPE：一维位置

文本 token 只有一维顺序位置：

```text
第 0 个 token
第 1 个 token
第 2 个 token
...
```

所以普通 RoPE 的 positions 是：

```text
positions: [N]
```

每个 token 只有一个位置编号。

### 8.2 MRoPE：三维位置

多模态输入中，图像 token 不只有文本序列位置。

图像 patch 还具有空间结构，例如：

```text
时间 T
高度 H
宽度 W
```

所以多模态位置可以表示成：

```text
positions: [3, N]
```

其中三行分别对应：

```text
temporal position
height position
width position
```

qwen3.6 的 `InterleavedMRoPE` 就是为这种 3D positions 准备的。

---

## 9. InterleavedMRoPE 支持文本和多模态两种路径

### 9.1 1D positions：退化成普通 RoPE

qwen3.6 代码中：

```python
if positions.ndim == 1:
    cos_sin = self.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
```

这说明当输入是普通文本位置：

```text
positions: [N]
```

MRoPE 可以直接退化成标准 RoPE。

这保证了：

```text
文本-only 请求仍然可以正常跑
```

### 9.2 3D positions：走多模态 MRoPE

当：

```text
positions.ndim != 1
```

也就是：

```text
positions: [3, N]
```

代码会计算：

```python
freqs = torch.einsum("dn,h->dnh", positions_3d.float(), inv_freq)
```

得到每个维度的频率：

```text
freqs: [3, N, half_dim]
```

然后通过 `_apply_interleaved_mrope` 把 T/H/W 三个维度的频率交错融合成一个 RoPE 频率。

---

## 10. Interleaved MRoPE 的 interleaving 逻辑

qwen3.6 中有：

```python
self.mrope_section = mrope_section
```

以及：

```python
def _apply_interleaved_mrope(self, freqs):
    freqs_t = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):
        length = self.mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t
```

可以理解为：

```text
先以 temporal 维度作为基础
再把 height / width 维度的频率按间隔插入到部分频率位置
```

也就是不是简单把 T/H/W 拼接，而是按规则交错混合。

这种 interleaving 的目的，是让一个 head 的 rotary 维度中同时包含：

```text
时间位置信息
高度位置信息
宽度位置信息
```

从而让 Attention 在处理视觉 token 时知道图像 patch 的空间/时间结构。

---

## 11. InterleavedMRoPE 也支持 partial RoPE

`InterleavedMRoPE` 中同样有：

```python
self.rotary_dim = int(head_size * partial_rotary_factor)
self.is_partial = self.rotary_dim < head_size
```

这表示它不是简单接收固定 rotary_dim，而是根据：

```text
head_size * partial_rotary_factor
```

计算旋转维度。

和标准 `RotaryEmbedding` 一样，它也支持：

```text
只旋转前 rotary_dim
后面维度直接保留
```

这对多模态模型很重要，因为视觉/文本混合模型的 head 维度布局可能更复杂。

---

## 12. InterleavedMRoPE 和前面 model_runner.py 的关系

前面 `model_runner.py` 中 qwen3.6 版本新增了多模态位置计算：

```text
_compute_mrope_positions
```

它会根据：

```text
token_ids
image_grid_thw
image_token_id
```

生成：

```text
positions_3d: [3, total_tokens]
```

而本文件中的 `InterleavedMRoPE` 正是消费这种 3D positions 的模块。

两者关系是：

```text
model_runner.py:
    根据图像网格生成 3D positions

rotary_embedding.py:
    根据 3D positions 生成 interleaved MRoPE cos/sin
    并应用到 q/k 上
```

所以这两个文件连起来，构成了多模态位置编码路径。

---

## 13. get_rope 的变化

### 13.1 原版

原版：

```python
@lru_cache(1)
def get_rope(...):
    rotary_emb = RotaryEmbedding(...)
    return rotary_emb
```

只缓存 1 个 RoPE 实例。

### 13.2 qwen3.6 版本

qwen3.6 版本：

```python
@lru_cache(maxsize=4)
def get_rope(...):
    rotary_emb = RotaryEmbedding(...)
    return rotary_emb
```

缓存数量从 1 增加到 4。

### 13.3 这个改动的意义

当项目支持更多模型形态后，可能会存在多个不同 RoPE 配置，例如：

```text
不同 head_size
不同 rotary_dim
不同 max_position
不同 base
```

把缓存从 1 扩到 4，可以避免频繁重复构造 RoPE buffer。

不过需要注意：

```text
get_rope 当前仍然返回 RotaryEmbedding，不返回 InterleavedMRoPE。
```

因此 MRoPE 很可能是在模型文件中直接实例化，而不是通过这个通用 `get_rope()` 创建。

---

## 14. qwen3.6 版本没有改变什么

虽然 qwen3.6 增加了很多能力，但它没有改变底层 RoPE 的基本数学核心：

```text
x1, x2 分半
y1 = x1 * cos - x2 * sin
y2 = x2 * cos + x1 * sin
cat(y1, y2)
```

也没有改变：

```text
标准文本 RoPE 的 cos/sin cache 思路
q/k 同时加位置编码
输出 dtype 转回原 dtype
```

所以对于普通 Qwen3 dense 文本模型，标准 RoPE 路径仍然保持和原版一致，只是支持了更多配置情况。

---

## 15. 和 Qwen3.6 / hybrid 架构的关系

这个文件和 Qwen3.6 hybrid 的关系需要分清楚：

```text
它不是 GatedDeltaNet 的 state 管理文件；
它也不负责 recurrent state / conv state；
它负责 Attention 里的位置编码。
```

也就是说，它主要服务于：

```text
Attention 层
多模态 Attention
Q/K 位置编码
```

如果 Qwen3.6 的某些层是 GatedDeltaNet，这些层可能不走普通 Attention RoPE 路径。

但是对于 hybrid 模型中的 full attention 层，以及多模态模型的视觉/文本 token 位置建模，这个文件非常关键。

---

## 16. 和多模态支持的关系

qwen3.6 版本中新增的 `InterleavedMRoPE` 是非常明确的多模态改造。

它使模型可以处理：

```text
文本 token:
    positions 是一维

图像 token:
    positions 包含 temporal / height / width 三维信息
```

这样模型在 attention 中不仅知道：

```text
某个 token 在序列第几个位置
```

还可以知道：

```text
某个图像 patch 在图像的哪一行、哪一列、哪一帧
```

这对视觉语言模型非常重要。

---

## 17. 原版与 qwen3.6 结构对比

### 17.1 原版结构

```text
rotary_embedding.py
  ├── apply_rotary_emb
  ├── RotaryEmbedding
  └── get_rope
```

原版只支持：

```text
标准文本 1D RoPE
全 head_dim 旋转
```

### 17.2 qwen3.6 版本结构

```text
rotary_embedding.py
  ├── apply_rotary_emb
  ├── RotaryEmbedding
  │     ├── full RoPE
  │     └── partial RoPE
  ├── InterleavedMRoPE
  │     ├── 1D text positions
  │     └── 3D multimodal positions
  └── get_rope
```

qwen3.6 版本扩展到：

```text
标准 RoPE
partial RoPE
多模态 3D MRoPE
```

---

## 18. 面试角度应该怎么回答

如果面试官问：

> qwen3.6 版本的 `rotary_embedding.py` 相比原版做了什么改动？

可以这样回答：

`rotary_embedding.py` 负责在 Attention 中给 q/k 加旋转位置编码。原版实现比较简单，要求 `rotary_dim == head_size`，也就是整个 head 维度都做标准 1D RoPE，只支持文本序列位置。qwen3.6 版本主要做了两个扩展：第一，标准 `RotaryEmbedding` 支持 partial RoPE，允许 `rotary_dim < head_size`，只旋转前一部分维度，后面的维度直接保留；第二，新增了 `InterleavedMRoPE`，用于 Qwen VL 类多模态模型，能够同时处理 1D 文本 positions 和 `[3, N]` 形式的 3D positions，把 temporal、height、width 三个维度的位置频率交错融合后应用到 q/k 上。除此之外，qwen3.6 还把 `get_rope` 的缓存从 1 扩到 4，适配更多 RoPE 配置。整体来看，这个文件的改动不是 GDN state 管理，而是让 Attention 的位置编码从纯文本标准 RoPE 扩展到 partial RoPE 和多模态 MRoPE。

---

## 19. 初学者最应该抓住的主线

这个文件可以这样记：

```text
原版:
    文本模型
    1D positions
    整个 head_dim 做 RoPE

qwen3.6:
    文本 + 多模态
    1D positions + 3D positions
    支持 partial RoPE
    支持 Interleaved MRoPE
```

其中最核心的变化是：

```text
positions 不再只能是 [N]，
多模态时可以是 [3, N]，
分别表示 temporal / height / width。
```

---

## 20. 最终结论

qwen3.6 版本 `rotary_embedding.py` 的核心意义是：

```text
扩展 Attention 位置编码能力。
```

具体来说：

1. **支持 partial RoPE**  
   删除原版 `rotary_dim == head_size` 的强假设，允许只旋转 head 的部分维度。

2. **支持多模态 MRoPE**  
   新增 `InterleavedMRoPE`，支持文本 1D positions 和多模态 3D positions。

3. **适配 Qwen VL 类位置编码**  
   通过 `mrope_section` 和 interleaving 逻辑，把 temporal / height / width 三类位置信息融合进 RoPE。

4. **扩大 RoPE 实例缓存**  
   `lru_cache` 从 1 扩到 4，适配更多 RoPE 配置组合。

因此，这个文件在 qwen3.6 项目中的作用可以概括为：

```text
让 nano-vLLM 的 Attention 位置编码从纯文本标准 RoPE，扩展为支持 partial RoPE 和多模态 3D MRoPE，为 Qwen3.6 / Qwen-VL 类模型提供位置编码基础。
```


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-vllm-qwen3.6|模块-nano-vllm-qwen3.6]]

%% 项目关联导航：结束 %%
