# nano-vLLM：ModelRunner 作用与 Prefill / Decode 数据准备（面试口语版）

> 这份文档只保留 ModelRunner 最核心的内容。  
> 重点是：**ModelRunner 到底干什么，以及 Prefill / Decode 两个阶段分别准备哪些数据。**

---

## 1. ModelRunner 整体是干什么的？

**面试官：你说一下 ModelRunner 主要负责什么？**

我理解 ModelRunner 就是连接 **Scheduler 和真正模型计算** 的一层。Scheduler 前面已经决定好了这一轮哪些 Sequence 要运行，但是 Sequence 里面保存的是 `token_ids`、`num_cached_tokens`、`block_table` 这些请求状态，模型不能直接拿这些 Python 对象去算。所以 ModelRunner 要先把它们整理成 GPU 上真正能用的 Tensor，然后再调用模型 forward，最后拿到 logits 并采样出新 token。除此之外，它还负责模型加载、显存 warmup、KV Cache 和 Qwen3.5 的 GDN State 分配，以及多卡执行和 CUDA Graph 这些工作。

---

## 2. ModelRunner 主要有哪些作用？

**面试官：如果让你概括 ModelRunner 的几个主要作用，你会怎么说？**

我一般把它分成几个部分。第一是初始化 GPU 和 TP 多卡通信；第二是创建模型并加载权重；第三是做 warmup，估算模型本身占多少显存，然后分配 KV Cache，Qwen3.5 还要额外分配 GDN 的 `conv state` 和 `recurrent state`；第四也是最重要的一块，就是根据当前是 Prefill 还是 Decode，把 Sequence 整理成模型需要的数据；第五是执行真正的模型 forward；最后再根据 logits 做采样，得到这一轮的新 token。这里面我认为最核心的就是 **Prefill 和 Decode 的数据准备**。

---

# Prefill 数据准备

## 3. `prepare_prefill()` 主要做什么？

**面试官：Prefill 阶段 ModelRunner 具体准备什么？**

Prefill 的特点是，一个请求这一轮可能一次处理很多个 token，而且不同请求这一轮处理的 token 数还不一样。所以 `prepare_prefill()` 要做的事情，就是从每个 Sequence 里找出这一轮真正需要计算的那一段 token，然后把多个请求拼成一个变长 batch。同时它还要告诉 Attention：每个请求的边界在哪里、前面已经有多少历史 KV、新计算出来的 KV 应该写到哪里；对于 Qwen3.5，还要告诉 GDN 每个请求对应哪个 State Slot。

---

## 4. Prefill 怎么确定这一轮到底计算哪一段 token？

**面试官：ModelRunner 怎么知道一个 Sequence 这一轮该取哪些 token？**

它主要看两个字段：`num_cached_tokens` 和 `num_scheduled_tokens`。`num_cached_tokens` 表示这个请求前面已经有多少 token 算过并且有 KV 了，`num_scheduled_tokens` 表示 Scheduler 这一轮又安排它计算多少个。所以开始位置就是 `num_cached_tokens`，结束位置就是两者相加。比如一个请求已经缓存了 600 个 token，这一轮 Scheduler 又安排 300 个，那么 ModelRunner 实际只会拿 `token_ids[600:900]` 去跑，而不是把整个 Prompt 再算一遍。

---

## 5. Prefill 的 `input_ids` 是什么？

**面试官：Prefill 最后准备的 `input_ids` 是什么？**

`input_ids` 就是这一轮所有请求真正需要计算的 token。比如 A 这一轮算 300 个，B 算 200 个，ModelRunner 会把它们直接拼起来，形成一个总共 500 个 token 的一维 Tensor。这样做是为了支持变长 batch，不要求每个请求的长度都一样。

---

## 6. Prefill 的 `positions` 是什么？

**面试官：`positions` 为什么不能每次都从 0 开始？**

因为它表示 token 在原始 Sequence 里的真实位置。比如一个长 Prompt 前 1024 个 token 已经在上一轮 Chunked Prefill 算完了，这一轮从第 1024 个位置继续，那么 positions 就应该从 1024 开始，而不是重新从 0 开始。这样 RoPE 的位置才是连续的，不会因为 Prompt 被拆成多个 Chunk 就把位置编码弄乱。

---

## 7. `cu_seqlens_q` 和 `cu_seqlens_k` 是干什么的？

**面试官：Prefill 为什么还需要 `cu_seqlens_q` 和 `cu_seqlens_k`？**

因为多个不同长度的请求已经被拼成一条连续 Tensor，所以底层 Attention 需要知道每个请求的边界。`cu_seqlens_q` 记录的是每个请求这一轮新算了多少个 Query token；`cu_seqlens_k` 记录的是这些 Query 实际能够看到多长的 KV 上下文。比如一个请求前面已经缓存 600 个 token，这一轮新算 300 个，那么 Q 长度是 300，但它能看到的 KV 总长度是 900。所以可以简单记：**`cu_seqlens_q` 管这一轮新算多少，`cu_seqlens_k` 管这一轮总共能看多少历史。**

---

## 8. Prefill 的 `slot_mapping` 是什么？

**面试官：`slot_mapping` 在 Prefill 中起什么作用？**

`slot_mapping` 是告诉 Attention，当前这一轮新计算出来的每个 K 和 V 最后应该写到 GPU KV Cache 的哪个具体位置。因为 nano-vLLM 用的是 Paged KV Cache，一个请求的 KV 可能分散在多个物理 Block 里，所以 ModelRunner 要根据 Sequence 的 `block_table` 和 token 在 Block 内的位置，进一步算出每个 token 对应的真实 KV slot。简单说，`block_table` 找到的是哪一个 Block，而 `slot_mapping` 找到的是 **这个 token 在显存里的具体写入位置**。

---

## 9. Prefill 为什么有时还需要 `block_tables`？

**面试官：Prefill 不是正在建立 KV 吗，为什么还需要 `block_tables`？**

如果是第一次完整 Prefill，前面没有任何历史 KV，那确实主要是把新 KV 写进去就行。但如果是 Prefix Cache 命中了，或者 Chunked Prefill 前面已经算过一部分，那么当前 Query 还需要读取之前已经存在的历史 KV。这时候就必须把 `block_tables` 传进去，让 Attention 知道那些历史 KV 分别放在哪些物理 Block 里。

---

## 10. Qwen3.5 的 `state_indices` 是干什么的？

**面试官：适配 Qwen3.5 后 Prefill 又增加了什么？**

Qwen3.5 多了 GDN 层，它的历史状态不是 KV Cache，而是 `conv state` 和 `recurrent state`。所以每个 Sequence 会有一个 `state_slot_id`，表示自己的 GDN State 存在状态池哪个槽位。ModelRunner 会把这一批请求的 `state_slot_id` 整理成 `state_indices`，放进这一轮 Context。后面的 GDN 层根据 `state_indices`，就能找到每个请求自己的状态。所以我一般记成：**`block_tables` 找 Full Attention 的 KV，`state_indices` 找 GDN State。**

---

# Decode 数据准备

## 11. `prepare_decode()` 和 Prefill 最大的区别是什么？

**面试官：Decode 阶段的数据准备为什么简单很多？**

因为 Decode 时每个请求这一轮只计算一个 token，不像 Prefill 一次可能算几百个。所以它不需要再复杂地表示每个请求这一轮有多少个 Query，基本上每个 Sequence 只取一个 `last_token`。但它必须把这个 token 背后的历史信息准备好，比如它现在处于什么位置、之前有多少有效 KV、历史 KV 在哪些 Block，以及当前新 K/V 应该写到哪里。

---

## 12. Decode 的 `input_ids` 是什么？

**面试官：Decode 为什么只取 `last_token`？**

因为之前所有历史 token 的信息已经保存进 KV Cache 和 GDN State 里了，所以这一轮不需要重新输入整个历史。每个请求只需要把上一个刚生成的 `last_token` 送进模型，然后 Full Attention 从 KV Cache 里读取历史，GDN 从自己的 recurrent state 和 conv state 里读取历史。比如这一轮有 8 个请求，那 `input_ids` 基本就是 8 个 `last_token` 组成的 Tensor。

---

## 13. Decode 的 `positions` 是什么？

**面试官：Decode 的 position 怎么确定？**

它就是当前这个新 token 在整个 Sequence 里的真实位置。正常没有 KV 压缩时，基本可以理解成 `len(seq)-1`。如果后面加入了 KV Cache 压缩，就不能再完全依赖物理 KV 长度，而是要用独立维护的 `rope_pos`，因为 KV 虽然可能被删掉一部分，但是 token 在真实时间轴上的位置不能倒退。

---

## 14. `context_lens` 是什么？

**面试官：Decode 里的 `context_lens` 具体有什么用？**

它表示这一轮 batch 中每个请求现在有多少个有效 KV。比如 A 有 1000 个历史 KV，B 有 500 个，C 有 2000 个，那么 `context_lens` 就是 `[1000, 500, 2000]`。Attention 会结合这个长度和 `block_tables`，知道每个请求应该读取多少历史 KV。因为 Decode 虽然每个人只进来一个 Query，但是每个人背后的历史长度完全不同，所以这个参数非常重要。

---

## 15. Decode 的 `block_tables` 是什么？

**面试官：Decode 为什么一定需要 `block_tables`？**

因为 Decode 的核心就是“一个新 Query 去访问整段历史 KV”。这些历史 KV 又是按 Paged KV Cache 分散在不同物理 Block 中的，所以必须通过 `block_tables` 找到这个请求所有历史 KV 的物理位置。可以理解成它就是每个请求自己的 KV 页表。

---

## 16. Decode 的 `slot_mapping` 是什么？

**面试官：Decode 一轮只有一个 token，为什么还要 `slot_mapping`？**

因为这个新 token 算出来以后，也会产生新的 K 和 V，需要继续写进 KV Cache。`slot_mapping` 就是告诉 Attention，这个请求当前新产生的 K/V 应该写到哪一个物理 slot。Prefill 时一个请求可能对应很多个 slot，而 Decode 时一个请求这一轮通常就对应一个 slot。

---

## 17. Decode 中 `state_indices` 怎么用？

**面试官：Qwen3.5 Decode 时 `state_indices` 有什么作用？**

它和 Prefill 时一样，都是告诉 GDN 当前每个请求应该访问哪个 State Slot。区别是 Decode 时 GDN 会直接读取这个请求上一轮留下来的 `conv state` 和 `recurrent state`，结合当前这个新 token 更新一次，再写回同一个 slot。所以如果 `state_indices` 对错了，请求之间的 GDN 状态就会串掉。

---

# 18. Prefill 和 Decode 最后分别准备了什么？

**面试官：你能最后总结一下两条路径最终准备的数据吗？**

可以。Prefill 主要准备 `input_ids`、`positions`、`cu_seqlens_q/k`、`slot_mapping`，有历史 KV 时还需要 `block_tables`，Qwen3.5 再额外准备 `state_indices`。Decode 更简单，每个请求只取一个 `last_token` 作为 `input_ids`，然后准备 `positions`、`context_lens`、`block_tables`、当前新 KV 的 `slot_mapping`，以及 Qwen3.5 的 `state_indices`。这些信息除了 `input_ids` 和 `positions` 直接进入模型，其余大部分会放到这一轮公共 Context 里面，Attention 和 GDN 在计算时再读取。

---

# 19. ModelRunner 的其它作用再简要说一下

**面试官：除了准备 Prefill 和 Decode 数据，ModelRunner 还有什么？**

其它部分我会简单概括。它初始化 TP 多卡通信并绑定 GPU；创建模型、加载权重；先做 warmup 来测模型峰值显存，再根据剩余显存分配 KV Cache，Qwen3.5 还要分配 GDN State；数据准备好以后调用真正的 `model.forward()`；模型得到 logits 后，rank 0 再结合每个 Sequence 的 temperature 做采样；如果开启 CUDA Graph，Decode 还可以直接 replay 已经捕获好的计算图来降低 CPU launch 开销。多模态和 MTP 版本也会继续在 ModelRunner 上扩展，但它的基础职责还是不变的。

---

# 20. 一段完整面试回答

> ModelRunner 我理解就是 Scheduler 和模型执行之间的桥梁。Scheduler 先决定这一轮哪些 Sequence 要运行，ModelRunner 再把这些 Sequence 里的请求状态转换成 GPU 模型真正能用的数据。它还负责模型加载、warmup、KV Cache 和 Qwen3.5 GDN State 的分配、多卡执行、采样和 CUDA Graph，但最核心的是 Prefill 和 Decode 的输入准备。Prefill 时一个请求可能一次算很多 token，所以它会根据 `num_cached_tokens` 和 `num_scheduled_tokens` 截出本轮真正要算的 token，准备 positions，再用 `cu_seqlens_q/k` 描述不同请求的边界和历史长度，同时根据 block table 计算新 KV 的 `slot_mapping`；如果前面已经有缓存 KV，还要准备 `block_tables`。Qwen3.5 还会把每个请求的 `state_slot_id` 整理成 `state_indices`，让 GDN 找到自己的 conv state 和 recurrent state。Decode 时每个请求只取一个 `last_token`，再准备它的 position、`context_lens`、`block_tables`、`slot_mapping` 和 `state_indices`。这些信息准备完成以后，再进入真正的模型 forward。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-面试准备|模块-面试准备]]

%% 项目关联导航：结束 %%
