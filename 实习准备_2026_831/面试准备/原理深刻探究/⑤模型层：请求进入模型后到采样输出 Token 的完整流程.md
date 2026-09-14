# ⑤ 模型层：请求进入模型后到采样输出 Token 的完整流程

## 模型输入与 Embedding

### 1. 经过model runner后，sequence被整理成可以输入模型的tensor，举一个实际的例子说明刚进入模型的tensor是什么形状，什么内容

在我的 nano\-vLLM 项目里，`ModelRunner` 并不是把多个 Sequence 整理成传统的 `[batch_size, seq_len]` 这种二维 Token 矩阵，而是会把这一轮真正需要计算的 Token **直接拼成一条一维 Tensor**，然后再**通过执行上下文记录不同请求之间的边界**。比如这一轮 Prefill 同时调度两个请求，A 需要计算 5 个 Token，Token ID 是 `[101, 205, 36, 78, 90]`，B 需要计算 3 个 Token，是 `[301, 52, 66]`，那么 `prepare_prefill()` 会把它们拼起来，得到 `input_ids = [101,205,36,78,90,301,52,66]`，它的形状就是 **`[8]`**；同时会生成 `positions=[0,1,2,3,4,0,1,2]`，形状也是 **`[8]`**，表示这些 Token 在各自请求中的逻辑位置。那模型怎么知道前 5 个属于 A、后 3 个属于 B 呢？这就是执行上下文里的 `cu_seqlens` 发挥作用，比如 `cu_seqlens_q=[0,5,8]`，意思就是第 0 到 4 个 Token 属于 A，第 5 到 7 个属于 B；同时还有**形状大约为 ****`[8]`**** 的 ****`slot_mapping`**，告诉 Attention 这 8 个 Token 新计算出来的 K、V 分别应该写到全局 KV Cache 的哪个物理位置，如果需要读取历史 KV，还会准备每个请求的 `block_tables`。所以**真正调用 ****`self.model(input_ids, positions)`**** 时，直接进入模型的最主要就是这两个一维 Tensor**：**`input_ids`**** 负责告诉模型“这一轮算哪些 Token”，****`positions`**** 负责告诉模型“这些 Token 在原序列的什么位置”**，其他 KV 管理信息通过 Context 给 Attention 使用。进入 `VocabParallelEmbedding` 以后，`input_ids` 才会从 `[8]` 个整数 ID 变成比如 `[8, hidden_size]` 的浮点 hidden states，之后 Attention、MLP 真正处理的是这个高维 Tensor。Decode 阶段更直观，比如当前有 A、B、C 三个请求同时生成，那么每个请求这一轮通常只有最新的 1 个 Token，所以 `input_ids` 可能就是 `[90, 152, 763]`，形状为 **`[3]`**，`positions` 可能是 `[512, 87, 1024]`，也就是三个 Token 分别位于各自序列的第 512、87、1024 位置；Embedding 后就变成 `[3, hidden_size]`。所以我会概括成：**ModelRunner 把多个 Sequence 中这一轮要算的 Token 展平成一个连续的一维 Token ID Tensor，同时准备位置和请求边界、KV 地址等元数据；模型先把 ****`[总Token数]`**** 的 ID 经过 Embedding 变成 ****`[总Token数, hidden_size]`****，这才是后面 Transformer 真正开始计算的数据。**

### 2. 进入 `VocabParallelEmbedding` 以后是怎么进行embedding的，隐藏状态的维度是模型训练时就规定好的嘛？比如我的千问3\.5\-27B应该是多大

进入 `VocabParallelEmbedding` 以后，本质上就是**拿每个 Token ID 去 Embedding 权重表里查一行，把一个整数变成一个固定维度的向量**。这个向量的维度就是模型的 `hidden_size`，**它是在模型架构设计时就确定好的**，训练出来的所有权重也都是按照这个维度保存的，所以推理阶段不能随便改。以我用的 **Qwen3\.5\-27B** 为例，官方配置里 `vocab_size=248320`，`hidden_size=5120`，所以如果不考虑多卡，可以把完整 Embedding 表理解成一个 `[248320, 5120]` 的矩阵，一个 Token ID 就是这个矩阵的行号，比如 Token ID 是 100，就取第 100 行，得到一个长度为 5120 的向量。

在我的 nano\-vLLM 项目里用的是 `VocabParallelEmbedding`，因为我是 TP=4，所以这个词表会沿着**词表这一维**分到四张卡上，每张卡只保存 `248320 / 4 = 62080` 行，也就是每张卡的 Embedding 权重大约是 `[62080, 5120]`。比如 Token ID 是 70000，那么它属于第二张卡负责的词表范围，**其他三张卡会把这个 Token 标记掉**，只有负责 70000 的那张卡真正查到一个 `[5120]` 的向量，**其他卡对应结果先置零，最后四张卡做一次 ****`all_reduce`**，因为实际上只有一张卡有非零结果，所以相加之后四张卡都得到同一个完整的 5120 维 Embedding。比如这一轮 ModelRunner 送进来 8 个 Token，`input_ids` 形状原来是 `[8]`，经过 `VocabParallelEmbedding` 后就变成 `[8, 5120]`，这个 `[8,5120]` 才是后面的 Qwen3\.5 Decoder Layer 真正处理的 `hidden_states`。所以这里最关键的是：**Token ID 只是查表的行号，5120 维向量才是模型真正计算的数据；而 5120 这个维度由 Qwen3\.5\-27B 的模型结构提前确定，27B 指的是整个模型大约 270 亿参数，并不是 hidden size。**

### 3. 推理框架是怎么把词表分到四个显卡上的，通过什么方式，推理框架侧是都在cpu上吗，那他是怎么和gpu产生联系的

在我的项目里，**词表分到 4 张 GPU 上主要是靠** **Tensor Parallel 加 ****`VocabParallelEmbedding`** 完成的。程序启动 TP=4 时，会启动 4 个进程，每个进程绑定一张 GPU，并通过 **`torch.distributed.init_process_group("nccl", world_size=4, rank=rank)`**** 建立通信组，同时用 ****`torch.cuda.set_device(rank)`**** 指定这个进程负责哪张卡**。模型创建 `VocabParallelEmbedding` 时，它会读取当前进程的 `rank` 和总卡数 `world_size`，然后把完整词表沿着词表这一维平均切成 4 份。

比如 Qwen3\.5\-27B 的词表大约是 24\.8 万，那么每张卡只创建大约 6\.2 万行的 Embedding 权重。真正加载模型权重时，完整权重文件是从磁盘以 CPU Tensor 的形式读出来的，然后**每个 rank 根据自己的编号，通过 ****`narrow()`**** 只截取属于自己的那一段**，比如 rank0 取第 0～62079 行，rank1 取后面一段，再通过 `param.data.copy_()` 拷到本进程对应 GPU 上的参数里。所以并不是“CPU 先把整个模型切好再一次性发给四张卡”，而是**四个 GPU 进程各自知道自己负责哪一片，在加载权重时各取自己的分片**。另外推理框架也不能简单说“都运行在 CPU 上”：像 Scheduler、Sequence、BlockManager 这些控制逻辑主要运行在 CPU 上，负责决定这一轮算谁、分哪些 KV Block；但 `ModelRunner`、模型权重、Embedding、Attention、MLP、KV Cache 这些真正的大规模数值计算和数据都在 GPU 上。

**CPU 和 GPU 的连接主要通过 PyTorch CUDA Tensor**，比如 ModelRunner 在 CPU 上把 Sequence 整理成 `input_ids` 后，会把这些数据放到 CUDA Tensor，再调用 GPU 上的模型；**而四张 GPU 之间的大张量通信则通过 NCCL，**比如 Embedding 时一个 Token 只属于某张卡负责的词表范围，那张卡查到真正的 5120 维向量，其他卡结果置零，然后四张卡做一次 `all_reduce`，最终每张卡都得到完整的 Embedding 结果。所以我会概括成：**CPU 主要负责调度和控制，GPU 负责模型计算；TP=4 时每个 GPU 进程根据 rank 在加载阶段只拿自己负责的权重分片，多卡之间再通过 NCCL 完成必要的数据同步。**

### 4. 为什么称之为embedding权重表，**把一个整数变成一个固定维度的向量有什么作用，这个向量有什么意义**

之所以叫 **Embedding 权重表**，是因为它本质上就是模型的一组**可训练参数**。假设词表里有 10 万个 Token，模型的隐藏维度是 4096，那么 Embedding 就可以看成一个 `[100000, 4096]` 的大矩阵，每一行对应一个 Token。比如 Token ID 是 520，那么模型就直接取第 520 行，得到一个 4096 维向量。之所以不能直接拿整数 520 去做后面的 Attention，是因为这个数字只是一个编号，本身没有任何语义，520 和 521 数字很接近，也不代表两个 Token 意思接近；

而** Embedding 向量是在模型训练过程中通过反向传播学出来的，它把这个离散的编号变成了一组连续的特征。**比如“猫”这个 Token 对应的向量里，并不是说某一维明确代表“动物”、某一维代表“宠物”，而是**几千个维度共同形成一个模型能够使用的初始表示，经过训练以后，语义、语法或者使用方式比较相近的 Token，在这个向量空间里往往会表现出一定的相关性。**然后这个 `[hidden_size]` 的向量才会进入后面的 Attention 和 MLP，并随着一层层 Transformer 不断结合上下文发生变化。所以我会把它理解成：**Token ID 只是词表中的地址，Embedding 权重表负责把这个地址查成模型能够计算的高维表示，这个向量就是这个 Token 进入 Transformer 时的初始特征表示。**

## CUDA Tensor、NCCL 与 GPU 互联

### 5. 详细介绍什么是 PyTorch CUDA Tensor

PyTorch CUDA Tensor 可以简单理解成：**PyTorch 里存放在 GPU 显存上的张量对象**。普通 `torch.Tensor` 如果默认创建，一般数据是在 CPU 内存里的；当我们执行 **`tensor.to("cuda")`**** 或者直接在 ****`device="cuda"`**** 上创建以后，这块数据就会放到 GPU 显存中，这时它就是 CUDA Tensor**。它本身除了保存具体数据，还会记录这个张量的形状、数据类型以及在哪张 GPU 上，比如 `shape=[8,5120]`、`dtype=bfloat16`、`device=cuda:0`。在推理框架里，Scheduler 和 Sequence 这些控制逻辑主要在 CPU 上运行，而 **ModelRunner 会把这一轮需要计算的 ****`input_ids`****、****`positions`**** 等数据整理成 CUDA Tensor，再送给模型。**之后像 Embedding、矩阵乘法、Attention、MLP 这些操作，只要输入和权重都是 CUDA Tensor，PyTorch 底层就会调用 CUDA kernel，让 GPU 真正完成计算。所以我理解 **CUDA Tensor 就是连接 PyTorch 上层代码和 GPU 计算的核心数据载体：CPU 负责发起操作，数据放在显存里，真正的大规模计算由 GPU 完成。**

### 6. **四张 GPU 之间的大张量通信则通过 NCCL，NCCL是什么，详细说一下**

NCCL 全称是 **NVIDIA Collective Communications Library**，可以理解成 **NVIDIA 专门为多张 GPU 之间做高速通信的一套库。**它解决的问题很直接：像我的项目是 4 张 3090 做 TP=4，同一层的权重和计算被拆到 4 张卡上以后，每张卡只能算出自己那一部分结果，后面经常需要把这 4 份结果合并或者同步，这时候就需要 NCCL。比如行并行线性层里，每张 GPU 会得到一份局部输出，最后要做 `All-Reduce`，也就是 4 张卡把自己的结果相加，然后每张卡都拿到完整结果；再比如词表被切到 4 张卡以后，采样阶段可能需要 `All-Gather`，把各卡自己的候选结果收集起来。**NCCL 就是负责把这些多 GPU 通信操作高效完成的底层库。**它并不是负责模型计算本身，矩阵乘法、Attention 这些还是 CUDA kernel 在做；**NCCL 主要负责 GPU 和 GPU 之间搬数据、做聚合。**工程上，PyTorch 里的 `torch.distributed` 可以把 NCCL 当成通信后端，比如程序启动 4 个进程，每个进程绑定一张 GPU，然后通过 `init_process_group(backend="nccl", world_size=4, rank=...)` 建立一个通信组，之后**上层调用 ****`all_reduce`****、****`all_gather`****、****`broadcast`**** 这些接口，底层真正的数据传输由 NCCL 完成。NCCL 会尽量利用机器上的高速互联，比如 PCIe、NVLink，如果是多机环境还可以走网络，**所以开发者一般不需要自己手写“GPU0 给 GPU1 发哪一块内存”这种复杂逻辑。另外 NCCL 里最常见的 collective 操作包括 `All-Reduce`、`All-Gather`、`Reduce-Scatter` 和 `Broadcast`：`All-Reduce` 是大家贡献一份数据，做求和等操作后每张卡都得到结果；`All-Gather` 是把每张卡不同的数据收集起来，让大家都拿到完整集合；`Reduce-Scatter` 是先聚合再把结果切开分给各卡；`Broadcast` 是某一张卡把数据发给其他卡。所以在我的项目里，我会把 NCCL 理解成 **TP 多卡推理时的“GPU 通信基础设施”**：模型被拆到 4 张卡上以后，各卡独立计算，凡是需要把局部结果重新合并、同步或者交换，就通过 PyTorch Distributed 调 NCCL 来完成。

### 7. **PCIe、NVLink和NCCL是什么关系，什么是NVlink**

PCIe、NVLink 和 NCCL 可以理解成三个不同层次的东西：**PCIe 和 NVLink 是“GPU 之间数据实际走的硬件通道”，NCCL 是“上层负责组织多 GPU 通信的软件库”。**比如我的 4 张 GPU 做 TP 推理时，代码里调用 `all_reduce`，真正负责“这 4 张卡应该**怎么交换数据、怎么把结果相加”的是 NCCL**；但是这些数据最后一定要通过某种**物理链路**在 GPU 之间传输，这个链路可能是 PCIe，也可能是 NVLink。PCIe 是比较通用的高速总线，GPU、网卡、SSD 都可以通过 PCIe 和 CPU 或其他设备通信；它通用性强，但 **GPU 到 GPU 通信时带宽和延迟通常不如专门的 GPU 高速互联。NVLink 是 NVIDIA 专门为 GPU 之间高速互联设计的链路**，可以把它理解成 GPU 之间的“高速专线”，它的带宽通常更高、延迟更低，更适合模型并行这种需要频繁交换大 Tensor 的场景。**比如 TP=4 时，每一层可能都要做 ****`All-Reduce`****，如果 GPU 之间只有 PCIe，那么数据需要通过 PCIe 路径传输；如果 GPU 之间有 NVLink，NCCL 通常会自动利用 NVLink 来走更快的通信路径。**还有一点要注意，**NCCL 本身不是 NVLink，也不是 PCIe，它不会替代这些硬件链路**，它只是知道机器的拓扑，然后尽量选择更高效的路径来完成 `All-Reduce`、`All-Gather` 这些操作。

所以我会把三者关系概括成一句话：**PCIe 和 NVLink 是“路”，NCCL 是“负责规划和组织多卡通信的软件”，其中 NVLink 是 NVIDIA 为 GPU 间高带宽、低延迟通信专门设计的高速互联，特别适合张量并行这种频繁跨卡通信的推理场景。**

## Embedding 的张量并行与集合通信

### 8. 我的说法对吗：一个一维的batchtoken进入模型后，词表会被分到四张卡上，每张卡只会embedding自己拥有词表的那部分token

你的说法大体是对的，但要补一个关键细节：**不是先把这一维 batch token 按 Token 分给四张卡，而是四张卡都会看到完整的 ****`input_ids`****，只是每张卡只负责自己那一段词表。**

比如这一轮 `input_ids=[10, 70000, 130000, 200000]`，完整词表被 TP=4 切成四段，假设 GPU0 负责 `[0,62080)`，GPU1 负责 `[62080,124160)`，GPU2 负责 `[124160,186240)`，GPU3 负责后面那一段。那么 GPU0 会发现只有 Token 10 属于自己的词表范围，就真正查出 Token 10 的 Embedding，其他三个位置先置零；GPU1 只处理 70000，GPU2 只处理 130000，GPU3 只处理 200000。也就是说，**每张卡输入看到的仍然是完整的一维 batch token，只是根据 Token ID 判断“这个 Token 是不是归我负责”**。最后四张卡会做一次 `All-Reduce`，因为**同一个位置只有一张卡产生了非零 Embedding**，所以相加以后，每张卡都会得到完整的 `[总Token数, hidden_size]` 的 hidden states。比如原来 `input_ids` 是 `[4]`，Embedding 后四张卡都会得到完整的 `[4, 5120]`。

所以更准确的说法是：“**一维 batch token 会同时送到四张卡，词表按 TP 切分，每张卡只对落在自己词表分片内的 Token 做有效 Embedding，其他位置置零，最后通过 All\-Reduce 合并成完整的 Embedding 结果。**”

### 9. 讲一下四张卡做完embedding后如何all\-reduce，什么是all\-reduce

四张卡做完 Embedding 以后之所以要做 `All-Reduce`，是因为在 `VocabParallelEmbedding` 里，每张卡只保存完整词表的一部分，所以对于同一个 Token 位置，**真正能查到 Embedding 的通常只有其中一张卡，其他卡这个位置先放 0。**比如这一轮有 4 个 Token，经过各卡局部查表后，GPU0 得到 `[e0,0,0,0]`，GPU1 得到 `[0,e1,0,0]`，GPU2 得到 `[0,0,e2,0]`，GPU3 得到 `[0,0,0,e3]`，这里每个 `e` 实际上都是一个 `hidden_size` 维向量。接下来做 `All-Reduce`，**本质就是让四张卡把同位置的数据做一次规约，比如求和，然后把求和后的完整结果再返回给每一张卡。**因为同一个位置只有一张卡是有效向量，其他卡都是 0，所以相加以后就得到 `[e0,e1,e2,e3]`，并且**四张 GPU 最终都会拿到这一份完整的 Embedding 结果。**`All-Reduce` 里的 `All` 可以理解成“所有卡都参与，而且所有卡最后都拿到结果”，`Reduce` 就是“把各卡的数据按照某种操作合并”，这里通常是求和。工程上一般是 PyTorch Distributed 调用 NCCL 来完成这个通信。所以我会概括成：Embedding 词表虽然被切到了四张卡，但输入 Token 是完整的；**各卡先算自己负责的非零部分，然后通过 All\-Reduce 把四份局部结果相加并同步到所有 GPU，最终每张卡都得到完整的 hidden states，才能继续后面的 Transformer 层计算。**

### 10. 规约是什么意思，All\-Reduce和reduce操作有什么区别，这个操作属于算子吗

“规约”可以理解成：**把多份同类型的数据按照某一种规则合并成一份结果**。比如四张卡上分别有数字 `1、2、3、4`，如果规约操作是求和，那么最后结果就是 `10`；如果是取最大值，那结果就是 `4`。在深度学习多卡通信里最常见的是求和。

`Reduce` 和 `All-Reduce` 的区别主要在于**最后谁能拿到结果**：比如四张 GPU 各有一份 Tensor，**做 ****`Reduce`**** 时，四张卡的数据会先求和，但最终结果只放到指定的一张卡，**比如 GPU0，其他三张卡不会得到完整结果；而 `All-Reduce` 是四张卡先做同样的求和，然后**最终四张卡每一张都拿到相同的完整结果**。所以可以简单理解成** ****`All-Reduce ≈ Reduce + 把结果再同步给所有卡`**，当然实际 NCCL 不一定真的分两步执行，而会用 Ring、Tree 等方式直接高效完成。至于它算不算“算子”，工程上可以把 `All-Reduce` 看成一种**集合通信算子或者通信原语**，和矩阵乘法这种纯计算算子不太一样；矩阵乘法主要是在一张 GPU 内部做计算，而 `All-Reduce` 的核心工作是多张 GPU 之间搬运并规约 Tensor，底层通常由 NCCL 实现，同时也会在 GPU 上执行相应的通信和求和操作。所以面试里我会总结成：**规约就是把多份数据按求和、最大值等规则合成一份；Reduce 是只让指定 GPU 拿到结果，All\-Reduce 是所有 GPU 最后都拿到结果；All\-Reduce 属于多卡并行里的集合通信算子，而不是普通的单卡计算算子。**

## Decoder Layer：RMSNorm 与 Attention 前处理

### 11. 当每张卡都拿到完整的隐藏状态，** 也就是Embedding 结果**后下一步是干什么，以qwen3为例

当四张卡通过 All\-Reduce 都拿到完整的 Embedding 结果以后，这个 Tensor 就正式作为模型的 `hidden_states` 进入 Qwen3 的第一层 Decoder Layer。以 Qwen3 为例，一层里面主要有两大块：Attention 和 MLP。

首先 hidden states **会先经过 RMSNorm 做归一化**，**然后进入 Self\-Attention，在线性层里生成 Q、K、V，**其中 Q、K 会结合当前位置做 RoPE 旋转位置编码，之后计算 Attention，也就是让当前 Token 根据 Q 去和各个 K 计算相关性，再对 V 做加权求和；Attention 的输出再经过输出投影，并和这一层最开始的 hidden states 做一次残差相加。然后这份新的 hidden states 再经过一次 RMSNorm，进入 MLP，Qwen3 的 MLP 主要是 gate、up、down 三个线性层，中间使用 SwiGLU 激活，作用可以理解成进一步对每个 Token 自身的特征做非线性变换；MLP 输出以后再做一次残差相加。这样就完成一个 Decoder Layer，然后这个 hidden states 再继续送进下一层，整个过程重复很多层。比如最开始 Embedding 后的数据形状是 `[总Token数, hidden_size]`，经过每一层以后总体形状基本还是 `[总Token数, hidden_size]`，变化的是里面的特征内容。等所有 Decoder Layer 都执行完，再经过最后一次 RMSNorm，然后通过 `LM Head` 把每个位置的 `hidden_size` 维向量投影成 `vocab_size` 维的 logits，也就是对词表里每个 Token 给一个分数，最后 Sampler 根据这些分数选出下一个 Token ID。所以可以简单记成：**Embedding 得到初始 hidden states → RMSNorm → Attention → 残差 → RMSNorm → MLP → 残差 → 重复多层 → 最终 RMSNorm → LM Head → logits → 采样得到下一个 Token。**

### 12. **讲一下hidden states 是怎么经过 RMSNorm 做归一化的**

在 Qwen3 里，`hidden states` 进入 RMSNorm 以后，会对**每一个 Token 对应的 hidden 向量单独做归一化**。比如某个 Token 的 hidden state 是一个 `hidden_size` 维向量 $(x=[x_1,x_2,\dots,x_d])$，RMSNorm 会先把这个向量**每一维平方，然后求平均，再开根号，得到这个向量整体的均方根，也就是 RMS，接着用原来的每一维除以这个 RMS，让整个向量的数值尺度变得比较稳定，最后再乘上一个训练过程中学出来的缩放权重 \(w\)。**它的公式可以理解成 $(y_i=\frac{x_i}{\sqrt{\frac{1}{d}\sum_j x_j^2+\epsilon}}\cdot w_i)$，其中 \(\\epsilon\) 是一个很小的数，主要是防止除零。比如一个简化的 4 维向量 `[1,2,3,4]`，先算平方平均值 `(1+4+9+16)/4=7.5`，开根号大约是 `2.74`，然后每一维都除以 2\.74，再乘对应的可学习权重。实际 Qwen3 里这个向量是几千维，但原理完全一样。RMSNorm 的作用不是改变特征维度，而是**控制每层输入 hidden state 的数值尺度，防止随着 Transformer 层数增加数值越来越大或者越来越不稳定，让后面的 Attention 和 MLP 更容易稳定计算**。所以像 `[总Token数, hidden_size]` 的 hidden states 经过 RMSNorm 以后形状完全不变，只是里面每个 Token 对应向量的数值被重新缩放了。

### 13. 工程上**RMSNorm这一步怎么做的，每张卡上都要做吗**

在工程实现上，我这个 nano\-vLLM 里的 RMSNorm 是定义在 `layers/layernorm.py` 里的，Qwen3 每个 Decoder Layer 都会创建 `input_layernorm` 和 `post_attention_layernorm`。真正执行时，比如当前 `hidden_states` 形状是 `[总Token数, hidden_size]`，RMSNorm 会沿着最后一个维度，也就是每个 Token 自己的 hidden 向量做归一化：**代码里会先把数据临时转成 ****`float32`**，然后执行 `x.pow(2).mean(dim=-1)` 求每个 Token 向量的平方均值，再加一个很小的 `eps`，通过 `rsqrt` 算倒数平方根，乘回原来的 hidden states，最后再乘上训练好的 RMSNorm 权重，并转换回原来的数据类型。

这里还有一个工程优化，就是项目把**“残差相加 \+ RMSNorm”合并到了 ****`add_rms_forward()`**** 里面，减少一些额外的数据读写。**\*\***在 TP=4 的情况下，这一步四张卡都要各自执行。**\*\*因为在我这个实现里，进入 RMSNorm 时四个 TP rank 上都已经有一份完整的 hidden states，而且 RMSNorm 的权重也很小，只是一个长度为 `hidden_size` 的向量，所以每张卡都保存一份相同的 RMSNorm 权重，然后直接在本卡上对 hidden states 做归一化就可以了，**RMSNorm 本身不需要 All\-Reduce，也不需要 NCCL 通信**。四张卡真正需要通信的一般是在后面的张量并行线性层，比如 Row Parallel 的输出合并；所以可以简单理解成：**RMSNorm 是每张 GPU 对自己当前持有的同一份 hidden states 做完全相同的本地计算，四张卡各算各的，不需要互相通信。**

### 14. 为什么**代码里会先把数据临时转成 float32，原本是什么格式，这个数据格式是由谁决定的**

在我这个项目里，RMSNorm 前会把 `hidden_states` **临时转成 ****`float32`****，主要是为了提高数值计算精度和稳定性**。因为 hidden states 原本通常不是 FP32，而是跟模型推理精度一致，比如 Qwen3/Qwen3\.5 这类模型实际推理时一般会使用 `bfloat16`，有些模型也可能是 `float16`。之所以不用 BF16 直接算 RMS，是因为 RMSNorm 里面要做平方、求平均、开根号或者倒数平方根，这几个操作会把数值误差放大，特别是模型层数很深时，如果每一层都用低精度算归一化，误差容易累计。所以工程上常见做法是先保存原来的 `dtype`，比如 BF16，然后 `hidden_states.float()` 临时转成 FP32，在 FP32 下计算平方均值和归一化，最后再转换回原来的 BF16，这样既保证归一化比较稳定，又不会让后面 Attention、MLP 全部用 FP32，避免显存和计算开销太大。至于**原来的数据类型是谁决定的，本质上是由模型权重的精度和推理框架的加载配置共同决定的。**比如模型 checkpoint 本身通常会声明它推荐使用 BF16，推理框架加载模型时也会根据 `dtype` 配置把权重放成 BF16；Embedding 权重是 BF16，那么 Token ID 查表以后得到的 hidden states 自然也是 BF16，后面的 Attention、MLP 基本也沿用这个数据类型。因此可以简单理解成：**模型主体为了性能和显存通常用 BF16/FP16，而像 RMSNorm 这种对数值精度比较敏感的小计算临时升到 FP32，算完再降回原精度，这是一种“关键位置保精度、主体计算保性能”的混合精度思路。**

### 15. **四张卡rmsnorm为什么要做完全相同的本地计算，四张卡各算各的，这不会资源浪费吗，为什么不一张卡做完再广播给其它卡**

看起来四张卡都做一遍 RMSNorm 好像重复计算了，但实际上**这点计算量非常小，反而比“只让一张卡算完再广播”更划算**。原因是 RMSNorm 主要就是对每个 Token 的 hidden 向量做平方、求均值、乘一个缩放系数，**计算量和后面的 Attention、MLP 矩阵乘法相比非常小**；而如果只让 GPU0 做，然后把完整的 `[总Token数, hidden_size]` hidden states 广播给另外三张卡，就会**额外引入一次跨卡通信**。对 TP 推理来说，**跨卡通信通常比这种简单的本地逐元素计算更贵**，而且还会让 GPU1、2、3 等待 GPU0 算完，形成同步点。现在四张卡本来就都各自持有同样的 hidden states 和同样的 RMSNorm 权重，所以最自然的方式就是每张卡在本地直接算，完全不需要 NCCL。可以简单理解成：**用一点点重复计算，换掉一次大 Tensor 的跨卡传输和同步开销**，这是明显更划算的。也因此在张量并行里，**像 RMSNorm、激活函数、残差相加这类计算量小、又不需要跨 hidden 维度分片的操作，通常都会每张卡本地各算一遍；**真正值得通信的是那些各卡只算出局部结果、必须合并才能继续的地方，比如 Row Parallel 后面的 All\-Reduce。

### 16. 四卡做完各自的RMSnorm，下一步干什么

四张卡各自做完 RMSNorm 以后，下一步就是进入这一层的 **Self\-Attention**。因为四张卡这时都有同样的一份归一化后的 `hidden_states`，但是 Attention 里的 Q、K、V 权重已经按照 TP=4 切到了四张卡上，所以**接下来每张卡都会拿这份完整的 hidden states，和自己保存的那一部分 Q、K、V 权重做线性投影，得到自己负责的那部分 Q、K、V。**比如 Qwen3 一共有很多个 Attention Head，TP=4 后可以理解成每张卡只负责其中四分之一的 Head，因此这一步不是四张卡重复算相同内容，而是**输入相同、权重不同、各算一部分 Attention Head**。接下来每张卡会对自己那部分 Q、K 做 RoPE 位置编码，然后把新产生的 K、V 按照 `slot_mapping` 写到本卡对应的 KV Cache 中，再用 Q 和历史 K、V 做 Attention，得到自己负责的局部 Attention 输出。然后这个局部输出还要经过 Attention 的输出投影；因为输出投影通常采用行并行，每张卡只能算出完整 hidden states 的一部分贡献，所以最后需要做一次 `All-Reduce`，把四张卡的局部结果相加，重新得到每张卡都一致的完整 Attention 输出。之后再和进入 Attention 之前保存的残差做相加，然后进入下一次 RMSNorm，再继续做 MLP。所以这一段可以记成：**四卡 RMSNorm → 每卡用自己的 QKV 权重算局部 Q/K/V → RoPE → Attention \+ KV Cache → 输出投影 → All\-Reduce 合并 → 残差相加 → 下一次 RMSNorm → MLP。**

### 17. 接下来每张卡都会拿这份完整的 hidden states，和自己保存的那一部分 Q、K、V 权重做线性投影，得到自己负责的那部分 Q、K、V。详细说一下这部分

四张卡做完 RMSNorm 以后，每张卡手里都有一份相同的 `hidden_states`，比如这一轮一共处理 \(T\) 个 Token，那么它的形状可以理解成 `[T, hidden_size]`。接下来进入 Attention 的第一步，就是把这份 hidden states 分别通过三组线性层投影成 Q、K、V。数学上可以写成 $(Q=XW_Q,\ K=XW_K,\ V=XW_V)$，这里 \(X\) 就是 RMSNorm 后的 hidden states，`WQ、WK、WV` 都是模型训练好的权重。关键在于 TP=4 以后，**不是把输入 X 切成四份，而是把 Q、K、V 的权重沿“输出维度”，也就是 Attention Head 这一侧切成四份**。比如完整的 `WQ` 如果能够产生 64 个 Query Head，那么可以理解成 GPU0 保存其中 16 个 Head 对应的权重，GPU1 保存另外 16 个，四张卡加起来才是完整的 `WQ`；K、V 也是类似，只不过像 GQA 模型里 KV Head 数量通常比 Query Head 少。于是四张卡都会拿相同的 `X`，但是因为每张卡乘的是不同的权重分片，所以 GPU0 得到自己负责的 `Q0、K0、V0`，GPU1 得到 `Q1、K1、V1`，以此类推。假设为了方便理解，完整 Q 的形状是 `[T, 64, head_dim]`，TP=4 后每张卡可能只得到 `[T, 16, head_dim]`，所以这里实际上是**按 Attention Head 做并行计算**。**这一阶段不需要马上做 All\-Reduce，因为后面的 RoPE 和 Attention 本身也可以继续让每张卡只处理自己负责的这些 Head**：GPU0 对自己的 Q/K 做 RoPE，用自己的 Q 去和自己的 K/V 计算 Attention，GPU1、2、3也是一样。也就是说，从 QKV 投影开始，四张卡已经真正开始“分工”，它们不再重复算相同内容，而是**输入相同、权重不同、各自负责四分之一的 Head**。等每张卡把自己的局部 Attention 算完以后，再进入输出投影；输出投影通常采用行并行，这时候四张卡算出来的是完整 hidden state 的不同贡献，最后才通过一次 All\-Reduce 相加，把四份结果重新合成一份完整的 `[T, hidden_size]`。所以这一段最核心的链路就是：**完整 hidden states 同时在四张卡上 → QKV 权重按 Head 切分 → 每张卡计算自己的 Q/K/V → 每张卡独立做 RoPE 和局部 Attention → 输出投影 → All\-Reduce 合并，重新得到完整 hidden states。**

## 多头注意力、张量并行与 KV Cache

### 18. 简要说一下多头自注意力机制

多头自注意力可以简单理解成：**让同一个 Token 同时从多个不同角度去关注整段序列里的其他 Token。** 具体做法是先把每个 Token 的 hidden state **投影成 Q、K、V，然后把这些向量拆成多个 Head，每个 Head 都独立计算一次注意力**，也就是用当前 Token 的 Q 和所有历史 Token 的 K **算相关性，再根据相关性对 V 做加权求和**。因为每个 Head 的权重参数不同，所以不同 Head 可以学到不同类型的关系，比如有的更关注前后语义，有的更关注句法，有的可能关注远距离依赖。最后再把所有 Head 的结果拼起来，通过一次输出投影得到新的 hidden state。所以多头的核心作用就是：**不是只用一种注意力关系理解上下文，而是并行学习多种不同的关注模式，让模型对上下文的表示更丰富。**

### 19. 举个实际的例子说清楚每个步骤的形状，说明一下多头自注意力计算的过程

我举一个简化但工程上完全一样的例子：假设这一轮有 **4 个 Token**，模型的 `hidden_size=8`，一共有 **2 个注意力头**，那么每个头的维度就是 `head_dim=8/2=4`。首先 RMSNorm 后的 `hidden_states` 形状是 **`[4,8]`**，也就是 4 个 Token，每个 Token 用 8 个数表示；接下来分别乘训练好的 `WQ、WK、WV`，得到 Q、K、V，这时候形状都还是 **`[4,8]`**。然后为了做多头注意力，会把最后的 8 维拆成 `2个头 × 每头4维`，于是 Q、K、V 都变成 **`[4,2,4]`**，实际算 Attention 时通常再调整成 **`[2,4,4]`**，也就是“2 个 Head，每个 Head 有 4 个 Token，每个 Token 是 4 维向量”。接下来每个 Head 独立计算 `Q × K^T`，所以单个 Head 是 `[4,4] × [4,4] → [4,4]`，两个 Head 合起来注意力分数就是 **`[2,4,4]`**；这里每一个 `[4,4]` 矩阵表示 4 个 Token 两两之间的相关程度。然后除以 \(\\sqrt\{head\_dim\}\)，再加因果遮罩，保证第 2 个 Token 不能偷看第 3、4 个未来 Token，再做 softmax，形状仍然是 **`[2,4,4]`**。接着这个注意力权重乘 V，每个 Head 就从 `[4,4] × [4,4]` 得到 **`[4,4]`** 的输出，两个 Head 合起来是 **`[2,4,4]`**。最后把两个 Head重新拼接，恢复成 **`[4,8]`**，再经过输出投影 `Wo`，最终仍然得到 **`[4,8]`** 的 hidden states，继续进入残差连接和后面的 MLP。所以整个形状变化可以记成：**`[4,8] hidden states → [4,8] Q/K/V → [2,4,4] 多头拆分 → [2,4,4] 注意力分数 → [2,4,4] 每个头的输出 → 拼接回 [4,8] → 输出投影后仍是 [4,8]`**。多头注意力的核心就是把原来一个 8 维空间拆成两个 4 维的子空间，让两个 Head 用不同的 QKV 权重独立学习不同的上下文关系，最后再把两部分信息合回来。

### 20. 按照上个例子，这些做投影的权重矩阵形状是怎样的，由谁确定的

按照刚才那个例子，`hidden_size=8`、有 2 个 Attention Head、每个 Head 的 `head_dim=4`，那么输入的 hidden states 是 `[4,8]`，也就是 4 个 Token、每个 Token 8 维。为了得到 Q、K、V，需要分别乘三个训练好的投影权重矩阵。按照数学上 `X @ W` 的写法，**`WQ、WK、WV`**** 都可以看成 ****`[8,8]`****，因为输入维度是 8，而输出需要得到 ****`2个Head × 4维 = 8维`****，所以 ****`[4,8] @ [8,8] → [4,8]`**，然后再把最后的 8 维 reshape 成 `[4,2,4]`，也就是 4 个 Token、2 个 Head、每个 Head 4 维。Attention 算完以后，两个 Head 再拼回 `[4,8]`，然后经过输出投影 `WO`，它在这个例子里同样可以看成 **`[8,8]`**，最终还是得到 `[4,8]`。这些矩阵的形状不是推理时临时决定的，而是**模型设计和训练时就由配置确定好的**，最主要由 `hidden_size`、`num_attention_heads` 和 `head_dim` 决定，一般满足 `head_dim = hidden_size / num_attention_heads`。这里还要注意一个工程细节：我刚才说 `[8,8]` 是按照数学矩阵乘法来描述的，PyTorch 的 `Linear` 层内部权重实际通常按 `[out_features, in_features]` 存储，只是计算时等价于乘它的转置。另外像 Qwen3 实际使用了 GQA 时，Q 的 Head 数和 KV Head 数不一定一样，所以 `WQ` 的输出维度可能还是 `num_attention_heads × head_dim`，但 `WK、WV` 的输出维度是 `num_key_value_heads × head_dim`，因此它们不一定和 `WQ` 一样大。简单来说，**投影矩阵的输入维度由 hidden size 决定，输出维度由要生成多少个 Head、每个 Head 多宽决定，这些结构参数在模型训练前就已经固定了。**

### 21. 按照你的qwen3\.5\-27B有多少个头，隐藏状态多少维

以你项目里的 **Qwen3\.5\-27B** 为例，如果说的是它的 **Full Attention 层**，官方配置里 `hidden_size=5120`，也就是说每个 Token 在 Decoder 层之间传递的 hidden state 是 **5120 维**；它有 **24 个 Query Head**，同时因为用了 GQA，只有 **4 个 KV Head**，而且每个 Attention Head 的 `head_dim=256`。

所以这里有一个很重要的点：**不能再用简单模型里的 ****`head_dim = hidden_size / num_heads`**** 来算 Qwen3\.5\-27B**，因为这里 `5120 / 24` 并不是 256；实际上 Q 投影会把 `[T,5120]` 的 hidden states 投影成 `24×256=6144` 维，然后 reshape 成 `[T,24,256]`，而 K、V 因为只有 4 个 KV Head，各自是 `4×256=1024` 维，也就是 `[T,4,256]`。如果我用 TP=4，那么 Full Attention 可以直观理解成每张卡负责 **6 个 Query Head** 和 **1 个 KV Head**。另外 Qwen3\.5 是 Hybrid 架构，它的 GDN/Linear Attention 层又有自己单独的一套 Head 配置，所以面试时如果问“Qwen3\.5\-27B 有多少个注意力头”，我会明确回答：**Full Attention 层是 24 个 Q Head、4 个 KV Head，每个 Head 256 维，模型主 hidden state 是 5120 维。**

### 22. 在做注意力计算时，每张卡计算的是给自己分配的权重部分矩阵吗，那这是在加载模型参数的时候就分配好了吗，什么时候加载的模型参数

是的，**在做 Attention 时，每张卡计算的就是自己那一部分 Q、K、V 权重对应的结果，而且这些权重分片在模型真正开始接收请求之前、加载模型参数的时候就已经分好了。**

以我这个 nano\-vLLM 为例，TP=4 启动时会创建 4 个 `ModelRunner` 进程，每个进程先通过 `torch.cuda.set_device(rank)` 绑定一张 GPU，然后建立 NCCL 通信组。接着在 `ModelRunner.init()` 里先创建 `Qwen3ForCausalLM` 模型结构，这时候像 QKV 投影用的 `QKVParallelLinear` 已经知道当前 `rank` 和 `world_size=4`，所以**每张卡只创建自己需要的那一部分参数空间**。**随后马上调用 ****`load_model(self.model, config.model)`**** 去读取模型目录下的 ****`safetensors`**** 权重文件。**权重文件读取时原始 Tensor 是**先从磁盘以 CPU Tensor 的形式取出来的，但并不是把完整 QKV 权重全部复制到每张 GPU，而**是**每个 rank 根据自己的编号，把完整权重沿输出维度切成 4 份，只取自己对应的那一份，再拷到自己 GPU 上已经创建好的参数里。**比如完整 Q 有 24 个 Head，TP=4，那么每张卡只保存 6 个 Q Head 对应的投影权重；如果 K、V 各有 4 个 Head，那么每张卡各保存 1 个 K Head 和 1 个 V Head 的权重。所以真正开始推理以后，四张卡虽然都拿到相同的完整 `hidden_states`，但 GPU0 乘的是第 0 份 QKV 权重，GPU1 乘第 1 份，以此类推，因此每张卡得到自己负责的局部 Q、K、V，然后继续独立做 RoPE 和局部 Attention。

也就是说，**模型权重分片不是每次请求进来以后临时切，而是在 ****`LLM`****/****`LLMEngine`**** 初始化、****`ModelRunner`**** 创建模型并调用 ****`load_model()`**** 时一次性完成，之后整个推理生命周期里这些参数就一直驻留在各自 GPU 显存中，请求来了以后只是反复使用这些已经分好的权重做计算。**

### 23. 当每张卡做完各自的注意力计算后会将各个头横着拼接起来再×一个输出矩阵得到隐藏状态的形状，这个过程是不是四个卡需要通信？说一下这个过程

对，但这里有一个很关键的点：从数学上看，多头 Attention 确实是把所有 Head 的结果拼起来，再乘一个输出投影矩阵 \(W\_O\)，**但在 TP=4 的实际工程里，并不会先把四张卡上的 Head 跨卡拼接起来，那样会多一次很大的通信。**以 Qwen3\.5\-27B 的 Full Attention 为例，一共有 24 个 Q Head，每个 Head 256 维，所以完整 Attention 输出如果把所有 Head 拼起来，本来应该是 `[T, 24×256] = [T, 6144]`，然后通过输出投影把它变回 `[T, 5120]` 的 hidden states。**TP=4 以后，每张卡只负责 6 个 Q Head，所以 GPU0 算完自己的 6 个 Head 后，只在本卡内部把这 6 个 Head 拼成 ****`[T, 6×256] = [T,1536]`**，GPU1、2、3 也是一样。然后**输出投影 \(W\_O\) 采用的是行并行**：完整的 \(W\_O\) 本来负责把 6144 维映射到 5120 维，但它**沿着输入的 6144 维切成 4 份**，所以每张卡只保存和自己 1536 维 Attention 输出对应的那部分权重。**于是每张卡分别计算出一个 ****`[T,5120]`**** 的局部贡献**，比如 GPU0 得到 \(Y\_0\)，GPU1 得到 \(Y\_1\)，最后真正完整的输出其实是 $(Y=Y_0+Y_1+Y_2+Y_3)$。**这时候四张卡才需要通过一次 All\-Reduce 做通信，把四份局部结果相加，并让每张卡最终都得到相同的 ****`[T,5120]`**** hidden states**。这样下一步做残差连接和 RMSNorm 时，四张卡又都有完整数据了。

所以整个过程应该理解成：**每卡局部 Head Attention → 本卡拼接自己的 Head → 乘自己那部分输出投影权重 → 得到 ****`[T,5120]`**** 的局部贡献 → 四卡 All\-Reduce 求和 → 每卡得到完整 ****`[T,5120]`**** hidden states。** 最重要的一点就是：**四张卡不是先通信把 24 个 Head 拼起来再做输出投影，而是先各自做输出投影，最后只通信投影后的结果，这样更适合张量并行。**

### 24. 所以是先每张卡得到一个最终隐藏状态的局部贡献，再做一次all\-reduce让每张卡得到完整的隐藏状态吗，然后再继续做后面的残差链接和rmsnorm吗

对，你这个理解是对的。在 TP=4 的 Attention 里，**每张卡先算自己负责的那部分 Head，然后在本卡内部拼接，再乘自己那一片输出投影权重，最终每张卡都会得到一个形状已经是 ****`[T, hidden_size]`**** 的“局部贡献”，**比如分别记成 \(Y\_0、Y\_1、Y\_2、Y\_3\)。这四份结果单独看都不是完整的 Attention 输出，真正的结果应该是 \(Y=Y\_0\+Y\_1\+Y\_2\+Y\_3\)，所以这里会做一次 `All-Reduce`，**把四张卡的局部贡献相加，并把完整结果同步到每一张卡上**。这样 All\-Reduce 结束以后，四张卡手里又都有一份相同的完整 hidden states。接下来才继续做这一层后面的操作，也就是和 Attention 之前保存下来的残差做相加，然后再进入下一次 RMSNorm，接着做 MLP。

### 25. 在注意力计算这一部分有没有工程上的动作，比如什么时候存kv，怎么存的等等

有，而且 Attention 这一部分其实有很多非常典型的工程动作，它不只是做一次 \(QK^T\) 的数学计算，还同时承担 KV Cache 的写入、读取和物理地址管理。在我的项目里，hidden states 经过 RMSNorm 后，每张卡先用自己那部分 Q、K、V 权重得到局部的 Q、K、V，然后给 Q、K 做 RoPE。**K、V 在这里生成以后，就已经具备存进 KV Cache 的条件了**，**Attention 层会读取 ModelRunner 之前放到 Context 里的 ****`slot_mapping`**，这个参数告诉它“当前每一个 Token 的 K、V 应该写到全局 KV Cache 的哪个具体物理位置”，于是把 K、V 写进提前由 BlockManager 分配好的 Paged KV Cache。`block_table` 负责告诉 Attention 一条请求历史上用了哪些物理 Block，而 `slot_mapping` 更细，**负责定位当前 Token 具体写到 Block 里的哪个槽位**。Prefill 阶段，**比如这一轮算 1000 个 Prompt Token，就会批量生成这 1000 个 Token 的 K、V，并写入对应的 KV Cache**，同时用 `flash_attn_varlen_func()` 对这一段 Prompt 做注意力；如果是 Chunked Prefill，第一块算完以后这些 KV 就保留下来了，下一块只计算新的 Token。Decode 阶段更典型，每条请求这一轮通常只有一个新 Token，先算出这个 Token 的 Q、K、V，把新的 K、V 追加写到 KV Cache，然后 `flash_attn_with_kvcache()` 用当前这个 Q 去读取之前所有已经缓存的历史 K、V，这样就不用每生成一个 Token 都重新计算整个 Prompt。TP=4 时，每张卡只保存自己负责的 KV Head，所以 **KV Cache 本身也是按 Head 分片存在四张卡上的，这一步通常不需要先把四张卡的 K、V 合起来**，**每张卡直接读写自己的本地 KV Cache、独立完成局部 Attention**；等局部 Attention 做完、经过输出投影以后，才通过 All\-Reduce 把四张卡的局部结果合成完整 hidden states。所以从工程链路上可以概括成：\*\*QKV 投影 → RoPE → 根据 `slot_mapping` 把新 K/V 写进 Paged KV Cache → 根据 `block_table` 找历史 KV → FlashAttention 做注意力 → 局部 Head 输出 → 输出投影 → All\-Reduce。\*\*这里 KV Cache 的意义就是典型的“用显存换计算”，尤其 Decode 时的历史 K、V 直接读取，不需要重复生成。

## RoPE 与 Q/K/V 设计

### 26. 详细讲一下给QK做RoPE的过程，具体内部是怎么做的，输入输出是什么，举一个具体的例子说明

在 Attention 里，Q、K 做完线性投影以后，会先经过 RoPE，也就是旋转位置编码。它的目的不是改变 Q、K 的内容维度，而是**把 Token 的位置信息直接编码进 Q 和 K 里面**，这样**后面做 \(QK^T\) 时，注意力分数天然就能感知两个 Token 之间的相对位置**。比如这一轮有 \(T\) 个 Token，每张卡负责 6 个 Q Head，每个 Head 是 256 维，那么一张卡上的 Q 形状大概是 `[T, 6, 256]`，K 可能是 `[T, 1, 256]`；另外 **ModelRunner 已经准备好了每个 Token 的 ****`positions`****，比如 ****`[0,1,2,3,...]`**。RoPE 会先根据每个位置和每个维度对应的频率，得到一组 `cos` 和 `sin`。**可以把一个 Head 的 256 维向量理解成很多二维平面，每个二维平面都按照当前位置旋转一个角度**。数学上，如果某一对分量原来是 \(\(x\_1,x\_2\)\)，位置对应的旋转角度是 \(\\theta\)，那么旋转后就是$(x'_1=x_1\cos\theta-x_2\sin\theta)$，$(x'_2=x_1\sin\theta+x_2\cos\theta)$。

Q 和 K 都做同样的旋转，但是 **V 不做 RoPE**。举一个非常简单的例子，假设某个 Token 的一个二维 Q 向量是 `[1,0]`，它位于 position=1，假设这一维对应的旋转角度刚好是 90°，那么 `cos=0，sin=1`，旋转以后 Q 就变成 `[0,1]`；如果同一个 Token 在 position=0，旋转角度是 0°，那它仍然是 `[1,0]`。也就是说，**同样的内容向量放在不同位置，经过 RoPE 后会得到不同的 Q/K，从而让 Attention 能区分“这个词出现在第 10 个位置”和“出现在第 100 个位置”**。实际 Qwen 里当然不会真的用 90° 这种简单角度，而是不同维度使用不同频率，低频维度变化慢、适合表达较长距离关系，高频维度变化快、适合表达局部位置关系。工程实现上通常会提前根据 `position_ids` 算好或缓存对应的 `cos/sin`，然后对 Q、K 做逐元素乘法和旋转组合，所以输入 Q/K 是 `[token数, head数, head_dim]`，输出形状完全不变，仍然是同样的 `[token数, head数, head_dim]`，只是数值已经带上了位置信息。之后新的 K 才会作为带位置编码的 K 写进 KV Cache，Q 则直接拿去和历史已经带 RoPE 的 K 做 Attention。所以整个过程可以记成：**Q/K 投影 → 根据 Token position 取对应 cos/sin → 在 head\_dim 内做二维旋转 → 得到带位置信息的 Q/K → K 写入 KV Cache，Q 用来和历史 K 做注意力计算。**

### 27. 为什么要给QK做RoPE而不用给V做，什么原理和意义，同样的为什么只存kv的cache，而不存q的

之所以 **RoPE 只作用在 Q 和 K 上，而一般不作用在 V 上**，根本原因要从 Attention 的公式来看：**注意力先计算 \(QK^T\) 得到“当前 Token 应该关注哪些历史 Token、关注多少”的权重，然后再用这个权重对 \(V\) 做加权求和**，也就是可以简单写成 **`Attention(Q,K,V)=softmax(QK^T)V`**。所以**真正决定两个 Token 之间位置关系的是 Q 和 K 的点积**。RoPE 把位置 \(m\) 编进 Q，把位置 \(n\) 编进 K以后，**两者做点积时，结果会自然包含 \(m\-n\) 这种相对位置信息，**模型就能知道“这个历史 Token 距离当前 Token 有多远”；**而 V 的主要作用是保存“这个 Token 到底提供什么内容”，它不参与注意力权重的计算，只是在权重算出来以后被取出来做加权汇总，所以通常没有必要再给 V 做同样的位置旋转**。

至于为什么 KV Cache 只缓存 K、V，不缓存 Q，也是同一个公式决定的。比如现在正在生成第 100 个 Token，这一轮只需要计算**当前第 100 个 Token 的 Q**，**然后拿这个 Q 去和前面第 1～99 个 Token 已经保存好的 K 做匹配，再根据注意力权重读取前面保存好的 V**；下一轮生成第 101 个 Token 时，会重新产生一个新的 Q，**而第 100 个 Token 之前算出来的 Q 已经不会再被使用了，所以历史 Q 没有缓存价值**。但历史 K、V 不一样，第 100、101、102……后面的每一个 Token 都还要反复读取前面所有历史 K、V，因此把 K、V 缓存下来可以避免每一轮重新计算。所以可以概括成：**Q 是当前 Token 发出的“查询”，只用当前这一轮；K 是历史 Token 的“索引”，V 是历史 Token 的“内容”，未来每一轮都还会用，因此 RoPE 加在决定位置关系的 Q/K 上，而 KV Cache 只保存会被未来重复使用的 K 和 V。**

## MLP、残差连接与 Decoder Layer 迭代

### 28. 每张卡都做完各自的一个注意力计算得到一个最终隐藏状态的局部贡献，再做一次all\-reduce让每张卡得到完整的隐藏状态后，然后再继续做什么内容，展开说一下

对，四张卡各自完成自己负责的 Attention Head 以后，会先经过各自那部分输出投影，得到一个形状已经是 `[总Token数, hidden_size]` 的局部贡献，然后通过一次 **All\-Reduce 求和**，使四张卡最终都拿到完全相同的、完整的 Attention 输出。

接下来就进入这一层 Transformer 的后半部分。首先会做**残差连接**，也就是**把 Attention 之前保存的那份原始 ****`hidden_states`**** 和刚刚得到的 Attention 输出直接相加**，这样可以保留原来的信息，同时加入 Attention 学到的上下文信息；这一步四张卡都已经有相同的数据，所以**每张卡本地相加即可，不需要通信**。然后对相加后的 hidden states 再做一次 **RMSNorm**，仍然是四张卡各自在本地归一化，得到比较稳定的输入。

接下来进入 **MLP**。以 Qwen3 为例，MLP 里面主要有 `gate_proj`、`up_proj` 和 `down_proj`：**`gate_proj`**** 和 ****`up_proj`**** 通常采用列并行，也就是四张卡拿相同的完整 hidden states，但各自保存四分之一的权重，所以各算自己那部分中间特征**；之后在每张卡本地做 SwiGLU 激活，可以理解成**用 gate 分支控制 up 分支哪些特征应该保留**；然后进入 `down_proj`，把**扩展后的中间维度重新投影回 ****`hidden_size`**，**这一层通常采用行并行，因此四张卡得到的又是完整 hidden state 的四份局部贡献，最后还需要再做一次 All\-Reduce**，把它们相加，让每张卡重新得到完整的 `[总Token数, hidden_size]`。**然后再和进入 MLP 之前保存的 residual 做一次残差相加，**这样一个完整 Decoder Layer 就结束了。之后这份 hidden states 直接进入下一层，又重复 **RMSNorm → Attention → All\-Reduce → 残差 → RMSNorm → MLP → All\-Reduce → 残差**。等所有 Decoder Layer 都执行完以后，再经过模型最后一次 RMSNorm，然后进入 LM Head，把 `[总Token数, hidden_size]` 投影到词表维度得到 logits，最后由 Sampler 选出下一个 Token ID。所以从 Attention 的第一次 All\-Reduce 往后，可以直接记成：**完整 Attention 输出 → 残差相加 → RMSNorm → MLP 的 gate/up 投影 → SwiGLU → down 投影 → All\-Reduce → 残差相加 → 下一层 Decoder，全部层结束后 → 最终 RMSNorm → LM Head → logits → 采样 Token。**

### 29. 举个具体的例子说明从输入MLP到输出的整个过程，包括中间各投影形状以及隐藏状态形状以及各卡在算什么。

可以用我这个 **Qwen3\.5\-27B、TP=4** 的实际配置来理解。假设这一轮一共处理 8 个 Token，那么 Attention 做完、残差相加、再经过 RMSNorm 后，每张卡手里都有一份完整的 `hidden_states`，形状是 **`[8, 5120]`**，因为 Qwen3\.5\-27B 的 `hidden_size=5120`，而它的 MLP 中间维度 `intermediate_size=17408`。

接下来进入 MLP，Qwen 这里采用的是类似 SwiGLU 的结构，主要有 `gate_proj`、`up_proj` 和 `down_proj` 三个投影。首先 `gate_proj` 和 `up_proj` 都属于**列并行**，完整情况下它们都是把 **5120 维扩展到 17408 维**，也就是数学上 `[8,5120] → [8,17408]`；但是 TP=4 后，17408 这个输出维度被平均切成四份，所以每张卡只负责 **4352 维**。因此四张卡虽然输入都是完整的 `[8,5120]`，但 GPU0 拿 gate/up 权重的第 0 份，GPU1 拿第 1 份，最后每张卡分别得到 `gate_i=[8,4352]` 和 `up_i=[8,4352]`。这一步不需要通信，因为四张卡本来就在计算不同的中间特征。**随后每张卡本地做 SwiGLU**，可以简单理解成**先对 ****`gate_i`**** 做 SiLU 激活，再和 ****`up_i`**** 对应位置相乘，也就是 ****`SiLU(gate_i) × up_i`****，形状仍然是 ****`[8,4352]`**。然后进入 **`down_proj`****，它负责把完整的 17408 维重新压回 5120 维，这里采用行并行：**因为前面的 17408 维已经被分成四份，所以每张卡直接拿自己 `[8,4352]` 的局部中间结果，**乘自己对应的 ****`down_proj`**** 权重分片，都会得到一个形状为 ****`[8,5120]`**** 的局部贡献**，比如分别记为 `Y0、Y1、Y2、Y3`。注意这四份虽然形状已经都是 `[8,5120]`，但任何一份都不是最终 MLP 输出，**真正结果是 ****`Y0+Y1+Y2+Y3`****，所以这里再做一次 All\-Reduce 求和，最后四张卡都重新拿到相同的完整 ****`[8,5120]`** MLP 输出。然后再和进入 MLP 之前保存的 residual 做相加，一个 Decoder Layer 就结束了，再把 `[8,5120]` 的 hidden states 送进下一层。

所以整个 MLP 的形状变化可以记成：**每卡完整输入 ****`[8,5120]`**** → gate/up 各自扩展出局部 ****`[8,4352]`**** → 本卡做 SwiGLU 仍是 ****`[8,4352]`**** → down\_proj 每卡得到 ****`[8,5120]`**** 的局部贡献 → 四卡 All\-Reduce → 每卡得到完整 ****`[8,5120]`**** → 残差相加 → 下一层。** 这里最核心的并行思想就是：**扩维的时候把 17408 个中间特征分给四张卡各算四分之一，缩回 hidden size 的时候每张卡算一部分贡献，最后只在 down\_proj 后通信一次。**

### 30. mlp中做计算的公式是什么

以 Qwen3 这类模型为例，MLP 不是最简单的“两层全连接”，而是用了带门控的 SwiGLU 结构。它的计算可以写成：$MLP(x)=Wdown(SiLU(Wgatex)⊙(Wupx))$
这里 \(x\) 就是进入 MLP 的 hidden state。首先同一个 \(x\) 会走两条支路：一条经过 `gate_proj` 得到门控特征，再做 `SiLU` 激活；另一条经过 `up_proj` 得到内容特征。然后这两部分做逐元素相乘，也就是公式里的 $(\odot)$，可以**理解成门控分支在决定“哪些特征应该放大、哪些应该抑制”。**最后再经过 `down_proj`，把中间维度重新投影回模型的 `hidden_size`。比如输入是 `[T, 5120]`，`gate_proj` 和 `up_proj` 都先扩到 `[T, intermediate_size]`，两者逐元素相乘后形状不变，最后 `down_proj` 再变回 `[T,5120]`。所以可以简单记成：**输入 hidden state → gate 和 up 两路投影 → gate 做 SiLU → 两路逐元素相乘 → down 投影回 hidden size。**

### 31. 这里的`gate_proj`、`up_proj` 和 `down_proj`是权重吗，还是隐藏状态已经和权重矩阵做运算了

`gate_proj`、`up_proj` 和 `down_proj` **本身不是某一个具体的权重数值，而是三个线性投影层，也可以理解成三个带权重矩阵的线性变换模块**。比如 `gate_proj` 内部会保存一个训练好的权重矩阵 \(W\_\{gate\}\)，`up_proj` 保存 \(W\_\{up\}\)，`down_proj` 保存 \(W\_\{down\}\)。当 hidden states 真正进入 MLP 时，才会分别和这些权重矩阵做矩阵乘法，比如 \(g=xW\_\{gate\}\)，\(u=xW\_\{up\}\)，这里的 \(g\) 和 \(u\) 才是“hidden states 经过 gate\_proj、up\_proj 之后得到的结果”。然后对 \(g\) 做 SiLU 激活，再和 \(u\) 逐元素相乘，得到中间特征，最后这个中间特征再经过 `down_proj`，也就是再乘 \(W\_\{down\}\)，得到最终 MLP 输出。

所以更准确地说，**`gate_proj/up_proj/down_proj`**** 是层或者操作，里面真正存的是训练好的权重矩阵；而像 ****`gate_proj(x)`****、****`up_proj(x)`**** 这种调用结果，才是 hidden states 和对应权重做完运算之后得到的新 Tensor。**

### 32. mlp和ffn什么关系，他们发挥了什么作用

在 Transformer 里，**MLP 和 FFN 基本可以理解成同一类模块，只是叫法侧重点不一样**：FFN 更强调它是“前馈神经网络”这个结构，MLP 更强调它内部是由多层线性层组成的实现方式。像 Qwen3 里我们前面讲的 `gate_proj、up_proj、down_proj`，这一整块就是它的 MLP，本质上也是一个带门控的 FFN。它的作用和 Attention 不太一样：**Attention 主要负责 Token 和 Token 之间的信息交互，比如当前词应该关注前面哪些词；MLP/FFN 则主要对每个 Token 自己的 hidden state 做更深的非线性特征变换**。比如一个 Token 现在是 5120 维 hidden state，先通过 `gate_proj` 和 `up_proj` 扩展到更高的中间维度，再通过 SiLU 和门控做非线性处理，最后通过 `down_proj` 压回 5120 维。这个过程中不同 Token 之间不会互相混合，每个 Token 都是独立经过同一套 MLP 权重，所以可以简单理解成：**Attention 负责“不同 Token 之间交流信息”，MLP/FFN 负责“把每个 Token 自己的特征进一步加工和提炼”**。很多研究也认为模型大量的知识和特征变换能力都集中在 FFN 这部分，因此它通常也是 Transformer 里参数量最大的模块之一。

### 33. 简要说一下你的mlp的过程以及他发挥了什么作用（面试）

在我的 Qwen3 里，MLP 主要是对 **每个 Token 自己的 hidden state 做进一步的特征加工。**

输入 hidden states 以后，先同时经过 `gate_proj` 和 `up_proj` 两个线性层，把 hidden size 扩展到更大的中间维度；然后 `gate_proj` 的结果先做 SiLU 激活，再和 `up_proj` 的结果逐元素相乘，**相当于用 gate 分支去控制哪些特征应该被保留或放大**；最后再经过 `down_proj`，**把中间维度重新压回原来的 hidden size**，输出继续进入残差连接和下一层。

它和 Attention 的作用不太一样，**Attention 主要负责不同 Token 之间的信息交互，而 MLP 主要负责对单个 Token 的内部特征做非线性变换和提炼**，所以可以简单理解成：Attention 负责“看别人”，MLP 负责“加工自己”。

### 34. 在mlp输出后是不是就进入下一个docoder layer，一直重复直到最后再做一个rmsnorm，再输入lmhead采样就得到一个token

对，整体流程可以这么理解：一层 Decoder Layer 里的 Attention 和 MLP 都做完，并且 MLP 输出和残差相加之后，这一层就结束了，然后把新的 hidden states 继续送进下一层 Decoder Layer，重复“RMSNorm → Attention → 残差 → RMSNorm → MLP → 残差”这个过程，直到所有 Decoder Layer 都执行完。最后模型不会直接拿最后一层输出去采样，而是会先再经过一次最终的 RMSNorm，把 hidden states 的数值尺度做稳定处理，然后进入 `LM Head`。`LM Head` 本质上是把 `hidden_size` 维的 hidden state 投影到整个词表大小，比如从 `[1,5120]` 变成 `[1,vocab_size]`，得到每个候选 Token 的一个分数，也就是 logits。然后这些 logits 才交给 Sampler，根据 temperature、top\-p 等采样参数选出真正的下一个 Token ID。所以准确的链路是：\*\*最后一个 Decoder Layer → 最终 RMSNorm → LM Head → vocab logits → Sampler → 得到下一个 Token ID。\*\*如果是 Decode 阶段，这个新 Token 会 append 到当前 Sequence，下一轮再把它作为新的输入继续 forward；只有遇到 EOS 或达到最大生成长度时，请求才真正结束。

## LM Head、采样与 Token 回传

### 35. 详细说一下隐藏状态进入LMhead后发生了什么，包括发生了什么计算，形状怎么变化，最后输出了什么，多卡怎么并行

隐藏状态进入 LM Head 以后，核心作用就是**把模型内部的 ****`hidden_size`**** 维表示转换成“词表里每一个 Token 的预测分数”**。以我这个 Qwen3\.5\-27B、TP=4 的项目为例，`hidden_size=5120`，词表大约是 `248320`。假设现在 Decode 阶段同时有 4 条请求，那么最后一层 Decoder 和最终 RMSNorm 之后，每张卡都有一份完整的 hidden states，形状是 **`[4,5120]`**。接下来进入 `Qwen3_5ForCausalLM.compute_logits()`，实际调用 `ParallelLMHead`。LM Head 本质上**也是一个线性投影**，单卡情况下完整权重可以理解成 `[248320,5120]`，计算就是 **`[4,5120] × [5120,248320]`**，**理论上会得到 ****`[4,248320]`**** 的 logits，也就是每条请求对整个词表 24 万多个 Token 分别打一个分**。但是我的项目是 TP=4，所以 LM Head 的词表权重也**沿着词表维度切成四份，每张卡只保存 ****`248320/4=62080`**** 个 Token 对应的权重，形状是 ****`[62080,5120]`**。因此**四张卡都拿相同的 ****`[4,5120]`**** hidden states，但 GPU0 只计算自己负责的前 62080 个 Token 的分数，GPU1 算下一段，**以此类推，所以每张卡只得到一个 **`[4,62080]`**** 的局部 logits**。这里我的项目并不会为了得到完整 `[4,248320]` logits 就马上做一次很大的 All\-Gather，因为那样通信量比较大，而是**每张卡直接在自己的 62080 个候选 Token 里先做采样**。比如贪心解码时，每张卡先找出自己分片里分数最高的 Token 和对应分数，再把**本地 Token ID 加上自己的 ****`vocab_start_idx`**，转换成全局 Token ID。假设四张卡分别得到 `(100,8.2)、(70000,9.1)、(140000,7.5)、(210000,8.7)`，这时候**只需要通过一次很小的 ****`All-Gather`**** 把这 4 个候选 Token 和分数收集起来，最后 rank0 再比较一次，发现 ****`70000`**** 的分数 9\.1 最高，于是它就是这条请求真正的下一个 Token ID**。温度采样时原理类似，只不过**每张卡先根据 temperature 对自己的局部 logits 做随机采样打分，再把局部候选汇总起来选全局结果**。另外 Prefill 阶段还有一个优化：假设两条 Prompt 一共产生了 `[1500,5120]` hidden states，我们其实不需要对 1500 个位置全部做 LM Head，因为只需要预测每条 Prompt 后面的第一个 Token，所以代码会先根据 `cu_seqlens` 取出每条请求最后一个位置，比如变成 `[2,5120]`，再做 LM Head。最终 LM Head 这条链可以概括成：**最终 hidden states → 取需要预测的位置 → 四张卡分别用自己的词表权重分片做线性投影 → 每卡得到局部 vocab logits → 每卡找自己的候选 Token → All\-Gather 少量候选和分数 → rank0 选出全局下一个 Token ID → 写回 Sequence，进入下一轮 Decode。**

### 36. 每张卡上先选出来自己卡上的得分最高，再做一次all\-gather。什么是all\-gather？展开说一下all\-gather前后数据在多卡下是怎么变化的？

`All-Gather` 可以理解成：**四张卡各自手里有一部分不同的数据，然后互相交换，最后让每张卡都拿到四张卡数据的完整集合；它和 All\-Reduce 最大的区别是，All\-Gather 只负责“收集和拼起来”，不会把这些数据做求和。**比如按照一个简化的贪心采样过程，Qwen3\.5\-27B 的词表被 TP=4 切成四份，每张卡只对自己负责的词表范围计算分数。假设同一条请求经过 LM Head 后，GPU0 在自己的词表里找到最高分候选是 `(Token 100, 分数 8.2)`，GPU1 得到 `(Token 70000, 9.1)`，GPU2 得到 `(Token 140000, 7.5)`，GPU3 得到 `(Token 210000, 8.7)`。**All\-Gather 之前**，GPU0 只知道自己的 `(100,8.2)`，并不知道其他三张卡谁分数更高；GPU1、2、3也是一样。然后通过 NCCL 做一次 All\-Gather，相当于四张卡把自己的候选都发给其他卡，**All\-Gather 之后**，每张卡都拥有 `[(100,8.2),(70000,9.1),(140000,7.5),(210000,8.7)]` 这四组结果，这时候再比较就能知道全局最高的是 `Token 70000`，分数是 9\.1。假如这一轮同时有 8 个请求，那么每张卡在 All\-Gather 前可能有形状为 `[8]` 的本地候选 Token ID 和 `[8]` 的分数，四卡收集后可以理解成变成 `[4,8]`，也就是“**每个请求都有四张卡各自给出的候选**”，然后再沿着卡这一维选出真正的全局结果。所以可以把它和 All\-Reduce 对比记忆：\*\*All\-Reduce 是四卡的数据做求和等规约，最后每张卡拿到同一个计算结果；All\-Gather 是四卡的数据不做运算，直接收集到一起，让每张卡都看到完整集合。\*\*不过工程实现上要看具体采样器，有些实现会 All\-Gather 局部 logits 或多个候选，而不一定只 Gather 每卡一个最高分；但 All\-Gather 本身的原理都是一样的：**各卡不同分片 → 跨卡收集 → 每卡得到完整数据集合。**

### 37. 进行完all\-gather每张卡都有最后本轮选出的token吗，选出来后怎么传给推理框架

对，但要结合我这个 nano\-vLLM 的实际实现区分一下：\*\*做完 All\-Gather 以后，四张卡确实都收集到了四张卡各自产生的候选 Token ID 和对应分数，**但是最终“哪一个才是全局选中的 Token”只由 rank0，也就是 0 号卡对应的主进程来确定。**\*\*比如四张卡分别得到 `(100,8.2)、(70000,9.1)、(140000,7.5)、(210000,8.7)`，`dist.all_gather()` 以后四张卡都能看到这四组候选；**但是代码里 rank1、2、3 到这里就直接返回 ****`None`****，只有 rank0 会再比较四张卡的分数，找到最高分对应的全局 Token，**比如选出 `70000`。**然后 rank0 会通过 ****`.tolist()`**** 把这个 GPU 上的 Token ID 转成 CPU 侧的 Python 数据，****`ModelRunner.run()`**** 把它返回给 ****`LLMEngine.step()`****。**接下来并不是直接把 Token 发回用户，而是先进入 `Scheduler.postprocess()`，把这个新 Token 通过 `Sequence.append_token()` 加到对应请求的 `Sequence` 里，同时更新 KV 已计算长度、判断是不是 EOS、有没有达到 `max_tokens` 等状态。如果请求还没结束，下一轮 `LLMEngine.step()` 又会把更新后的 Sequence 交给 `ModelRunner`；TP=4 时，rank0 会通过共享内存把这批新的 Sequence 控制信息发送给另外三个 worker 进程，所以另外三张卡在下一轮 Decode 前也会知道刚才生成的最新 Token，然后四张卡再一起计算下一轮。

\*\*所以正常 Decode 主链路可以理解成：四卡各自产生局部候选 → All\-Gather 候选 → rank0 选出全局 Token → Token 从 GPU 回到 rank0 的 CPU 推理框架 → Scheduler 写回 Sequence → 下一轮再把更新后的 Sequence 下发给四个 GPU。\*\*也就是说，最终 Token 不需要在采样结束后专门再做一次 GPU 间广播，主进程维护的是唯一的请求状态，下一轮调度时再把最新状态同步给各个 worker。

### 38. 在经过多层注意力计算后输出的隐藏状态相比于刚进入模型的隐藏状态有什么变化，lmhead的权重矩阵是模型训练时决定的然后加载时就加载在四张卡上面的吗，为什么隐藏状态和这个lmhead的矩阵做一次简单的乘法就能得到每个词表的一个打分

对，这里可以分成三个层次理解。第一，经过多层 Attention 和 MLP 以后，hidden state 的形状通常没有变，比如进入模型时一个 Token 是 5120 维，最后还是 5120 维，但**里面每一个数所表达的信息已经发生了很大变化。**刚进入模型时的 hidden state 主要来自 Embedding，可以理解成这个 Token 本身的初始表示，比如“苹果”刚进来主要表示“苹果这个词”；但经过一层层 Attention 以后，它不断吸收前后文的信息，再经过 MLP 做特征加工，所以最后一个位置的 hidden state 已经**不只是表示当前 Token，而是包含了模型对整个已知上下文的综合理解**。比如输入“法国的首都是”，最后一个位置经过几十层以后形成的 hidden state 就会比较强地包含“接下来应该回答巴黎”这种信息。

第二，LM Head 的权重确实也是模型训练时学出来的参数，不是推理时临时生成的。模型启动时加载 checkpoint，**LM Head 的权重就和 Attention、MLP 的权重一起加载到 GPU**；TP=4 时，会按照词表维度把 LM Head 分到四张卡上，例如完整词表有 24 万多个 Token，每张卡大概保存四分之一的词表权重，这些分片在模型开始接收请求之前就已经加载好了。

第三，为什么 hidden state 只乘一次 LM Head 权重就可以得到每个 Token 的分数？本质上是因为**训练阶段已经把这个线性层训练成了一个“词表分类器”。**假设最后 hidden state 是一个 5120 维向量 \(h\)，LM Head 里每一个词表 Token 都有一个对应的 5120 维权重向量，比如“巴黎”对应 $(w_{\text{巴黎}})$，“伦敦”对应 $(w_{\text{伦敦}})$。计算 $(h\cdot w_{\text{巴黎}})$就得到“巴黎”的一个分数，计算 $(h\cdot w_{\text{伦敦}})$ 就得到“伦敦”的分数；把 h 同整个 LM Head 权重矩阵做一次矩阵乘法，就等于**同时和词表里所有 Token 的权重向量做点积，因此一次就能得到整个词表的 logits**。关键并不是“矩阵乘法天然知道哪个词正确”，而是模型训练时通过交叉熵不断调整前面的 Transformer 参数和 LM Head 权重，使得在某个上下文下，最终 hidden state 会和正确下一个 Token 的权重方向更加匹配，从而让正确 Token 的分数更高。\*\*所以可以简单理解成：Embedding 把 Token 变成初始特征，多层 Transformer 把它变成包含整个上下文信息的最终特征，**LM Head 再把这个特征投影到词表空间**，得到每个 Token 作为下一个词的分**数。**

### 39. 简单说一下为什么经过LMhead就可以得到词表每个词对应的分数

因为 **LM Head 本质上就是一个“把 hidden state 映射到整个词表”的线性层**。假设最后一个 Token 的 hidden state 是 5120 维，而词表有 24 万个 Token，那么 LM Head 的权重可以理解成**给词表里的每个 Token 都准备了一个 5120 维的权重向量。最终 hidden state 和这些权重向量分别做点积，就能得到每个 Token 对应的一个分数，也就是 logits。**比如和“巴黎”对应的权重向量点积结果是 8\.5，和“伦敦”的是 5\.2，那模型就更倾向于预测“巴黎”。之所以这样一个简单的线性层就能工作，是因为在训练阶段，模型已经通过大量数据把前面的 Transformer 和 LM Head 一起训练好了，让正确下一个 Token 对应的分数尽量更高。所以可以简单理解成：**Transformer 先把上下文压成一个有语义的 hidden state，LM Head 再把这个 hidden state 投影到词表空间，一次矩阵乘法就得到整个词表所有 Token 的预测分数。**


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-面试准备|模块-面试准备]]

%% 项目关联导航：结束 %%
