# `nano-kvllm v0.2.0`

<p align="center">
  <img src="assets/logo.png" alt="nano-kvllm 标志" width="350"/>
</p>

**2.0 版本正式发布！**

**KvLLM** 是一个基于 `nano-vllm` 构建的 **AI/大语言模型推理框架**，专注于大语言模型的高效 KV Cache 显存管理。  
当前版本在一个轻量级研发框架中实现了 **KV Cache 压缩**，旨在缓解 KV Cache 的显存瓶颈，从而提升**高并发**和**长上下文生成**场景下的推理效率。

在接下来的几周中，**KvLLM** 将继续集成前沿的 KV Cache 管理技术，包括：

- **KV Cache 压缩（KV-cache compression）**
- **KV Cache 卸载（KV-cache offloading）**
- **KV Cache 检索（KV-cache retrieval）**

以形成一套更加完整、实用的大语言模型服务显存管理技术栈。

---

# `nano-kvllm v0.2.0` 的新特性

## 设计动机

目前主流的 KV Cache 压缩（稀疏化）方法大多采用**阈值触发式压缩机制**：当某个序列的 KV Cache 长度达到预先设定的阈值时，就会触发压缩。
这种设计可能比较适合单用户部署或 Agent 类场景，但也存在以下几个关键问题：

1. **压缩可能会影响系统提示词的 KV Cache**  
   由于该压缩机制会作用于全部历史 KV Cache，因此可能会剪除与系统提示词对应的缓存条目，从而对生成质量造成不利影响。
2. **在大批量场景下，压缩开销可能会降低吞吐量**  
   在高并发场景中，几乎每个请求都可能在每个 Decode Step 触发一次压缩事件，这反而可能降低输出吞吐量，而不是提升吞吐量。

---

## 方法

KvLLM 2.0 引入了**基于窗口的周期性压缩机制**。
具体来说：

- 从系统启动开始，对全局 Decode Step 进行计数。
- 每经过固定数量的 Decode Step（例如全局每经过 1024 个 Decode Step），触发一次压缩。
- 在每次压缩时，选择 **Top-K** 个序列。
- 对于每个被选中的序列，将其**最后 4 个 Block** 作为压缩窗口。
- 保证这些 Block 是**尚未被压缩的**。
- 使用**最新 Token 的 Query** 计算注意力分数，选择并保留重要的 KV 对，随后执行**显存紧凑化**和**元数据更新**。

---

## 优势

### 基于窗口的压缩

压缩窗口内的 Block 均保证为**未压缩 Block**，这可以避免反复压缩已经压缩过的内容，并有助于维持生成质量。

### 周期性压缩

周期性压缩能够控制压缩事件的发生时机，避免过多请求同时执行压缩。在高并发场景中，这可以有效提高输出吞吐量。

## 实验

我们在 **Math500** 数据集上进行了实验，实验配置如下：

- **批大小（Batch size）**：500（一次性输入全部样本）
- **压缩周期（Compression period）**：1024
- **被压缩的序列数量**：20
- **压缩比例（Compression ratio）**：50%

输出吞吐量提升了 **10%**：

- **2000 token/s → 2200 token/s**

---

## 关于 KvChat

**KvChat** 是一个基于 **KvLLM** 构建的应用层演示程序。  
它提供了一个轻量级的**单用户多轮对话**界面，用于展示如何在长对话过程中应用**在线 KV Cache 压缩**。

在长时间的多轮对话中，KV Cache 会随着已生成上下文的增加而持续增长，这可能导致：

- GPU 显存压力不断增大；
- 吞吐量下降；
- 在长会话中更早发生 OOM（显存不足）。

**KvChat** 通过启用**运行时 KV Cache 压缩**来解决上述问题，并允许用户通过以下方式获得快速、轻量的对话体验：

- 在对话过程中开启或关闭压缩；
- 调整与压缩相关的参数。

# 快速开始

## 1. 配置 `chat_cli.py`

在 `chat_cli.py` 中设置以下参数：

```python
enforce_eager=False     # 尽可能启用适合 CUDA Graph 的 Decode 路径，可显著提升 Decode 速度
tensor_parallel_size=2  # 用于张量并行推理的 GPU 数量
max_tokens = 32000      # 多轮对话中的最大生成长度
```

## 2. 运行

```bash
python chat_cli.py
```

随后即可直接在命令行界面中开始多轮对话。

KvChat 支持**运行时 KV Cache 压缩**，能够显著延缓长对话过程中 KV Cache 的增长。  
可以在 `nanokvllm/config.py` 中调整与压缩相关的参数：

```python
kv_compress_S: int = 511      # 当已生成上下文长度达到 S 时触发压缩
kv_compress_R: int = 257      # 压缩后，在 KV Cache 中保留提示词长度 + R 个 Token
query_window_size: int = 50   # 压缩算法的 Query Window 参数；建议范围：10～100
```

---

# 关于 `nano-kvllm`

`nano-kvllm` 是 **KvLLM** 背后的**开发框架**。  
它在 `nano-vllm` 的基础上提供了一个紧凑、实用的研究框架，用于落地 KV Cache 压缩算法。

开发者可以使用该框架：

1. 进一步提升大批量服务场景下的输出吞吐量；
2. **实现并快速验证 KV Cache 压缩算法**；
3. 进一步探索和扩展**其他 KV Cache 显存管理技术**。

当前版本为 **nano-kvllm v0.2.0**。  
如需了解早期版本，请参阅 **v0.1.0** 和 **v0.1.5**。

---

# 项目结构

- **KvChat**：面向单用户多轮对话的应用/演示层；
- **nano-kvllm**：基于 `nano-vllm` 构建的 KV Cache 显存管理研发框架。

---

## 更新日志

版本历史请参阅 [CHANGE_LOG.md](./CHANGE_LOG.md)。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano-kvllm|模块-nano-kvllm]]

%% 项目关联导航：结束 %%
