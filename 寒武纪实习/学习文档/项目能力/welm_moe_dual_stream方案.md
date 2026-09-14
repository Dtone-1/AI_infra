
## 1. 当前 WeLM MoE 实现

需要更改的类：welmv4.py文件的Qwen2MoeSparseMoeBlock


当前 `forward()` 的主要执行顺序是：

```text
Shared Expert
    ↓
Shared Gate
    ↓
Router
    ↓
TopK
    ↓
Routed Experts
    ↓
Shared + Routed
    ↓
AllReduce
```

因此虽然 Shared Expert 和 Routed Expert 在计算图上互不依赖，但当前都在同一默认 Stream 上提交，实际仍然是串行执行。

当前代码还支持 `router_context`。当 `gather_expert_hidden=True` 时，需要先经过：

```text
Router -> TopK -> prepare_moe_inputs
```

得到新的 `hidden_states`，Shared Expert 才能执行。

另外，WeLM 当前已经创建并向下传递了 `alt_stream`：

```text
WeLMV4MoeForCausalLM
        ↓
Qwen2MoeModel
        ↓
Qwen2MoeDecoderLayer
        ↓
Qwen2MoeSparseMoeBlock
```

但 `Qwen2MoeSparseMoeBlock.forward()` 目前还没有真正使用它。

---
moe-a2a-backend分别指定none和deepep下是什么情况，--tp-size 4--ep-size 4只能是none吗，这个参数是否决定了进入moe时各卡的隐藏状态是完整的还是被切分的
## 2. 参考 Qwen2-MoE 的双 Stream 方案

SGLang 官方仓库已经实现了 Qwen2-MoE 的 Shared/Routed Expert 双 Stream overlap。

官方代码位置：

```text
python/sglang/srt/models/qwen2_moe.py
```

核心函数：

```python
Qwen2MoeSparseMoeBlock.forward_normal_dual_stream()
```

官方对应 PR：

```text
SGLang PR #10252
add dual stream for qwen2_moe
```

该方案已经正式合入 SGLang main，并进行了精度与性能测试，因此可以作为 WeLM-MLU 双 Stream 优化的直接参考。

1. WeLM 当前 MoE 结构与 Qwen2-MoE 高度一致；
2. Shared Expert 和 Routed Expert 都只依赖同一个 `hidden_states`；
3. 两条分支之间没有直接数据依赖，只在最终 Add 时汇合；
4. WeLM 已经存在 `alt_stream` 基础设施，只需接入 MoE forward；

---

## 3. WeLM-MLU 实现方案

首先支持当前实际配置：

```bash
--tp-size 4
--ep-size 4
--moe-runner-backend mlu
--moe-a2a-backend none
```

### 3.1 Stream 分工

保留 Routed 路径在主 Stream：

```text
Main Stream:
Router -> TopK -> Routed Experts
```

把 Shared 路径放到备用 Stream：

```text
alt_stream:
Shared Expert -> Shared Gate
```

这样尽量不改动当前 MLU Router、MoE Runner 和 EP 路径。

### 3.2 修改位置

主要修改：

```python
Qwen2MoeSparseMoeBlock
```

新增：

```python
_forward_shared_expert()
_can_use_dual_stream()
_forward_dual_stream()
```

其中 `_forward_shared_expert()` 统一封装：

```python
shared_output = self.shared_expert(hidden_states)
shared_output = sigmoid(self.shared_expert_gate(hidden_states)) * shared_output
```

### 3.3 普通路径

当：

```text
router_context is None
或
gather_expert_hidden == False
```

可以直接：

```text
alt_stream:
Shared Expert -> Shared Gate

Main Stream:
Router -> TopK -> Routed Experts
```

最后主 Stream：

```python
current_stream.wait_stream(self.alt_stream)
```

然后执行：

```python
final_hidden_states += shared_output
```

### 3.4 gather_expert_hidden 路径

当：

```python
gather_expert_hidden == True
```

不能一开始就执行 Shared Expert。

应先在主 Stream 完成：

```text
Router
  ↓
TopK
  ↓
prepare_moe_inputs
```

得到 gather 后的 `hidden_states`，然后再分叉：

```text
             gathered hidden_states
                     │
        ┌────────────┴────────────┐
        │                         │
 Main Stream                  alt_stream
 Routed Experts             Shared Expert
                                ↓
                           Shared Gate
        │                         │
        └────────────┬────────────┘
                     ↓
                 wait_stream
                     ↓
                    Add
```

### 3.5 同步原则

使用：

```python
alt_stream.wait_stream(current_stream)
current_stream.wait_stream(alt_stream)
```

或 Event。

不要使用：

```python
torch.mlu.synchronize()
```

因为全设备同步会破坏其他 Stream 和通信的并行。


## 4. 总结

当前 WeLM 已经具备 `alt_stream`，但 MoE forward 仍然按照：

```text
Shared -> Router -> TopK -> Routed
```

串行执行。

参考 SGLang 官方 Qwen2-MoE 的 dual-stream 实现，将：

```text
Shared Expert + Shared Gate
```

放到 `alt_stream`，同时保留：

```text
Router -> TopK -> Routed Experts
```

在主 Stream，最终通过 `wait_stream/Event` 在 Add 前汇合。

只改 `Qwen2MoeSparseMoeBlock`，并限定 `moe-a2a-backend=none`，先完成普通路径和 `router_context` 路径的正确性与性能验证，再考虑 DeepEP。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-实习项目能力|模块-实习项目能力]]
- 关联阅读：[[outputs/项目整理/专题-04-并行推理与MoE实习|专题-04-并行推理与MoE实习]]

%% 项目关联导航：结束 %%
