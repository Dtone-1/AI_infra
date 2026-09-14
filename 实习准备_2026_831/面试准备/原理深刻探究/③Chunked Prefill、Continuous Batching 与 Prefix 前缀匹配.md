# ③ Chunked Prefill、Continuous Batching 与 Prefix 前缀匹配

## Prefix 前缀匹配

### 1. 前缀匹配发生在哪一个阶段，是什么时候判断匹不匹配，是按照每个block来判断的吗，如果匹配了是不是就是第一次出现的前缀的引用加1，先详细讲清楚前缀匹配的细节，再举例说明，在实际的vllm中也是这样的吗

在我的项目里，**前缀匹配发生在一条新请求第一次真正被 Scheduler 选中做 Prefill、准备给它分配 KV Block 的时候，而不是 Tokenizer 阶段，也不是模型已经 forward 以后才判断**。具体来说，请求刚创建成 `Sequence` 时，`block_table` 还是空的，先进入 waiting 队列；当 `Scheduler.schedule()` 处理 waiting 请求时，会**先调用 ****`BlockManager.can_allocate()`**** 判断能不能分配，随后调用 ****`BlockManager.allocate()`**，**前缀匹配真正就是在这个 ****`allocate()`**** 过程中完成的**。它不是拿整条 Prompt 一次性比较，也不是一个 Token 一个 Token 比，而是**按照一个个完整的逻辑 Block 从前往后连续匹配。**

比如为了方便说明，假设 `block_size=4`，请求 A 的 Token ID 是 `[10,20,30,40 | 50,60,70,80 | 90,100]`。A 第一次执行时，第一个完整 Block `[10,20,30,40]` 计算完成以后会得到一个哈希值，比如记作 `H1`；第二个 Block 不是简单只对 `[50,60,70,80]` 做哈希，而是把**前一个 Block 的哈希 H1 和当前 Block 的 Token一起参与计算，得到 ****`H2`**，所以实际上形成的是一条链：`H1 = hash(Block1)`，`H2 = hash(H1, Block2)`。这样做非常重要，因为它**保证第二块只有在“第一块也完全相同”的情况下才算同一个前缀，而不是说某个中间 Block 内容碰巧一样就能复用。**项目里会维护类似 `hash_to_block_id` 的表，比如 `H1 → 物理 Block 7`，`H2 → 物理 Block 12`。现在请求 B 到来，它的 Token 是 `[10,20,30,40 | 50,60,70,80 | 200,300]`，第一次调度 B 时，BlockManager 先计算第一块的 `H1`，查表发现 Block 7 已经有相同哈希，而且代码还会再比较实际 `token_ids`，确认确实一样，于是第一块命中；然后带着 `H1` 再计算第二块得到 `H2`，又命中 Block 12，那么 B 前 8 个 Token 就根本不用重新做 Prefill，可以直接复用 A 已经算好的 KV，`num_cached_tokens` 就增加 8。第三块因为内容不同，或者因为它还不是完整可缓存 Block，就重新申请物理 Block并正常计算。这里还有一个很容易说错的地方：**匹配以后并不一定简单理解成“第一次请求那个 Block 的引用永远加 1”。如果请求 A 现在还在运行，Block 7 和 Block 12 的 ****`ref_count`**** 原来都是 1，那么 B 又使用它们时确实会变成 2，说明两个请求同时引用同一份 KV；但如果 A 已经结束，这两个 Block 的引用可能已经降到 0，只是它们的哈希和 KV 内容还保留在 Prefix Cache 中、处于可复用状态，那么 B 命中以后是把这个空闲的缓存 Block重新激活，此时引用计数从 0 变成 1，而不是变成 2。所以 ****`ref_count`**** 表示的是现在有多少活跃请求正在使用这个物理 Block**，并不是“历史上一共命中过多少次”。还有一个非常重要的规则是**前缀必须从第一个 Block 连续命中**。比如 B 的第一块 `[10,20,30,41]` 就和 A 不一样，那么第一块哈希已经不同，即使 B 的第二块恰好也是 `[50,60,70,80]`，也不能复用 A 的第二块，因为第二块的哈希包含前一块的哈希，所以整个链已经不同了；这才叫真正的“前缀缓存”，而不是在 Prompt 中随便寻找相同片段。我的项目当前实现也是只把适合缓存的完整前缀 Block 建立哈希，尾部还没有完整、或者仍可能继续写入的 Block不会直接作为普通前缀命中对象。正式版 vLLM V1 的核心思想也是一样的：新请求进入调度以后，KV Cache Manager 会先寻找已经计算过的最长前缀，再给剩余 Token 分配新的 KV 空间；它的 Block 哈希同样由**前一个 Block 的哈希、当前 Block 的 Token，以及必要的额外信息**共同组成，因此只能从头连续命中。官方设计文档也明确说明，命中的缓存 Block 会被 `touch`，也就是附加给新请求并增加当前引用计数，避免它在使用期间被淘汰。\([vLLM](https://docs.vllm.ai/en/stable/design/prefix_caching/)\) 不过如果面试官问得非常严格，我会补充一点：\*\*传统的、单一 Full Attention KV Cache 场景可以直接理解成“按完整 Block 匹配”，这是最常见也最好理解的情况；但最新 vLLM 对 Hybrid KV Cache 又做得更通用，支持用 `prefix_match_unit` 把哈希匹配粒度设得比某些物理 KV Block 更细，所以不能绝对地说最新版所有模型永远只能按物理 Block 大小匹配。\*\*对于普通单一 KV Cache Group，当前实现的哈希粒度仍与调度 Block 粒度一致；多种 KV Cache Group 的 Hybrid 模型才可能配置更细的匹配边界。\([vLLM](https://docs.vllm.ai/en/stable/api/vllm/v1/core/kv_cache_utils/)\) 所以面试里我最终会概括成一句话：**前缀匹配发生在新请求第一次 Prefill 分配 KV Cache 之前，从 Prompt 开头按照连续的 Block 哈希链往后找最长命中前缀；命中的 KV 直接挂到新请求的 Block Table 上并更新当前引用计数，第一次不匹配以后后续就不能再作为前缀复用，剩余部分再正常分配 Block 和计算 Prefill。**

### 2. 我可不可以这样理解，在第一次某个请求被调度时，会计算这个请求的每个kvblock对应的哈希值，计算好后会在一个前缀匹配的哈希表中去查，这个哈希表的键是每个正在执行的请求block的哈希值，值是该block编号，如果查到了，引用次数加1，没查到就把自己的键值对加入这个哈希表，是这样吗

你的理解**大方向是对的，但有三个关键地方需要修正**。在我的项目里，一条新请求第一次被 Scheduler 调度做 Prefill、调用 `BlockManager.allocate()` 时，**会从 Prompt 开头按照 Block 顺序计算前缀哈希，**比如 `block_size=4`，请求 A 的 Token 是 `[10,20,30,40 | 50,60,70,80 | ...]`，第一块得到 `H1=hash([10,20,30,40])`，第二块得到 `H2=hash(H1,[50,60,70,80])`，然后**拿这个哈希去全局的 ****`hash_to_block_id`**** 中查询**。这个表确实可以简单理解成 **`前缀Block哈希 → 物理Block编号`**，例如 `H1→7，H2→12`，但是第一点要注意，**这个表里不是只保存“当前正在执行请求”的 Block**，还可能保存已经没有活跃请求引用、但 KV 内容暂时还留在缓存里等待以后复用的 Block。第二点，如果查到 `H1→7`，代码还会比较这个 Block 保存的 `token_ids`，防止单纯哈希碰撞；如果确认匹配，并且 Block 7 此时正在被另一个请求使用，比如 `ref_count=1`，那么新请求也使用它以后就变成 `ref_count=2`；但如果之前那个请求已经结束，Block 7 当前 `ref_count=0`、只是作为缓存留着，那么新请求命中后实际上是把它重新激活，引用数从 0 变成 1。

第三点也是最容易说错的地方：\*\*如果没有查到，并不是马上把“新哈希→新Block编号”写进哈希表。\*\*因为这时候只是给请求分配了一个空闲物理 Block，比如 Block 20，它里面真正的 K/V 还没有经过模型计算出来，所以现在还不能算“已计算前缀缓存”。**请求要先进入 ModelRunner 做 Prefill，等这个完整 Block 的 K/V 真正写进 KV Cache以后**，在 `Scheduler.postprocess()` 里调用 `BlockManager.hash_computed_blocks()`，这时才正式把类似 `H1→20` 写入 `hash_to_block_id`，以后其他请求才能复用它。举个完整例子：系统一开始哈希表为空，请求 A 的前两个完整 Block 是 `A1=[10,20,30,40]`、`A2=[50,60,70,80]`，第一次查询都没命中，于是给 A 分配物理 Block 0、1；Prefill 真正算完以后建立 `H1→0，H2→1`。之后请求 B 的开头也是 `[10,20,30,40 | 50,60,70,80 | 90,91,...]`，第一次调度 B 时计算 H1，查到 Block 0，引用数加一；再计算 H2，查到 Block 1，引用数也加一，于是 B 的前 8 个 Token 不需要重新计算 KV；如果第三个 Block 不同，从这个位置开始就算前缀匹配结束，后面的 Block重新分配和计算，不会越过这个不匹配位置继续找。**所以最准确的理解是：第一次 Prefill 分配 KV 之前，从头按 Block 计算链式哈希并查询全局前缀缓存表；命中就复用已有物理 Block并更新引用，第一次 miss 后剩余部分正常分配；而新 Block 必须等 KV 真正计算完成以后，才有资格把它的哈希和物理 Block 编号登记进前缀缓存表。**

### 3. 完整说一下一个请求从被调度器选中开始做prefill，它的blocktable怎么变化以及前缀匹配的过程的生命周期

如果让我完整讲这一段，我会先说明一个我项目里的实际细节：**我当前 Qwen3\.5 Hybrid 版本里，Prefix Cache 实际上是关闭的**，因为在 `Config` 初始化时，只要检测到是 Hybrid 模型，就会把 `prefix_cache_enabled` 设为 false，所以真正跑 Qwen3\.5 时，请求做 Prefill 不会走前缀哈希复用；但是 `BlockManager` 本身仍然保留了完整的 Prefix Cache 机制，所以我会把“当前实际执行”和“开启 Prefix Cache 时的完整生命周期”分开讲。

先说实际执行：一条请求刚创建成 `Sequence` 时，`block_table=[]`，比如有一个比较长的 Prompt，经过 Tokenizer 后是 600 个 Token，我项目默认 `block_size=256`，那这条请求逻辑上一共需要 `ceil(600/256)=3` 个 Block。它进入 waiting 队列以后，第一次被 `Scheduler.schedule()` 选中做 Prefill，会先调用 `BlockManager.can_allocate()` 判断全局空闲 Block 是否够用；因为当前 Qwen3\.5 没开 Prefix Cache，所以这 3 个 Block 都要新分配。随后 `BlockManager.allocate()` 从全局 `free_block_ids` 里依次取物理 Block，比如当前队首依次是 `5、9、12`，那么这一次 allocate 完成以后，Sequence 的 `block_table` 就会直接从 `[]` 变成 `[5,9,12]`，这里的含义是逻辑第 0 块 KV 放物理 Block 5，逻辑第 1 块放 Block 9，第 2 块放 Block 12。要注意，**这时候只是把显存位置预留好了，里面的 KV 还没有真正计算出来**。接着 Scheduler 根据本轮 token budget 设置 `num_scheduled_tokens`，如果 600 个 Token 能一次 Prefill 完，就一次算完；如果开启 Chunked Prefill，比如本轮只允许算 300 个 Token，那么第一次只计算前 300 个，但 `block_table` 仍然是 `[5,9,12]`，不会下一轮再重新分，因为这条请求需要的物理 Block 在第一次 allocate 时已经准备好了。之后进入 `ModelRunner`，根据 `block_table` 和 `slot_mapping` 把实际算出来的 K/V 写入这些物理 Block，forward 成功以后 `Scheduler.postprocess()` 再更新 `kv_num_tokens`，表示这些 KV 现在真的存在了；下一次 Chunked Prefill继续使用同一个 `block_table`，直到整个 Prompt 算完。Prompt 全部 Prefill 完成以后，模型会采样出第一个生成 Token，这个 Token 会 append 到 Sequence，此时逻辑 Token 数变成 601，但这个新 Token 还没有 KV；下一轮进入 Decode 时，`BlockManager.may_append()` 会判断现有最后一个 Block 是否还有空间，如果有就继续使用 `[5,9,12]`，只有以后 Token 数跨过下一个 256 的边界，才会再申请一个新 Block，比如 20，于是 `block_table` 变成 `[5,9,12,20]`。最后请求结束或者被抢占时，`BlockManager.deallocate()` 会把这些 Block 的引用数减掉，没有其他请求使用的 Block重新放回空闲队列，并把这个 Sequence 的 `block_table` 清空。

**如果 Prefix Cache 是开启的**，比如原始 nano\-vLLM 或普通 Full Attention 模型，生命周期在第一次 `allocate()` 时会多一段前缀匹配。还是假设 600 个 Token、256 一个 Block，**系统会从 Prompt 开头按完整 Block连续计算链式哈希，**比如第一块 `H1=hash(Token[0:256])`，第二块 `H2=hash(H1, Token[256:512])`，**然后去全局 ****`hash_to_block_id`**** 查**。假设以前一个请求已经计算过相同的前 512 个 Token，表里有 `H1→7、H2→18`，那么**新请求第一块命中 Block 7，第二块命中 Block 18，就不重新计算这两块 KV，而是直接复用，活跃 Block 的引用数会加一**，同时 `num_cached_tokens` 增加到 512；**第三块因为是新的尾部，就从空闲队列申请，**比如 Block 25，这时新请求的 `block_table` 会直接变成 `[7,18,25]`，也就是说一个 Block Table 里面可以同时存在复用的物理 Block和新分配的物理 Block。而且前缀匹配必须从第一个 Block连续往后，一旦某一块 miss，后面就不再继续寻找“中间碰巧一样”的 Block，因为那已经不叫相同前缀了。对于新分配的 Block，还不能一分配就加入前缀缓存表，因为 KV 还没算出来；必须等 ModelRunner 真正 Prefill 完成，`Scheduler.postprocess()` 调用 `BlockManager.hash_computed_blocks()`，**确认完整 Block 的 KV 已经物化之后，才会计算并登记它的哈希，**比如把新的 `H3→25` 放进哈希表，供未来请求继续复用。所以我会把这整个生命周期概括成：**Sequence 创建时 ****`block_table`**** 为空 → 第一次 Prefill 调度时进行前缀查询和物理 Block 分配 → 命中的 Block直接复用，未命中的 Block从空闲池申请 → ****`block_table`**** 建立逻辑块到物理块的映射 → 模型真正计算并写入 KV → 完整新 Block计算完成后才登记进 Prefix Cache → 后续 Chunked Prefill沿用同一个 Block Table → Decode 只有跨 Block 边界时才继续追加 Block → 请求结束或抢占时释放引用并清空 Block Table。**

### 4. 讲一下nanovllm的前缀匹配机制：

在 nano\-vLLM 里，一条新请求被调度器选中准备做 Prefill 时，首先会由 BlockManager 判断**当前 KV Cache 空间是否能够满足这条请求的执行需求。如果开启了 Prefix Cache，它不会一上来就给整个 Prompt 全部分配新的 KV Block，而是先根据这条请求的 Token ID，从第一个完整 Block 开始依次计算前缀哈希，并到全局的前缀哈希表中查询。** 如果某个 Block 的哈希能够命中已经计算过的缓存 Block，并且 Token 内容也确认一致，那么这部分 KV 就可以直接复用，**不需要重新分配新的物理 Block，也不需要重新做这部分 Prefill**，只需要把对应的物理 Block 加到当前请求的 `block_table` 中，并增加它的引用。前缀匹配是从 Prompt 开头连续向后进行的，**一旦某一个 Block 没有命中，后面的部分就不再作为前缀继续匹配**，而是从空闲 Block 池中为这些未命中的 Token 分配新的物理 KV Block。这样最终一条请求的 `block_table` 里面，前面可能是复用其他请求已经计算好的 Block，后面是这次新申请的 Block。**之后请求真正进入 Prefill，模型只需要计算没有命中的那部分 Token**；在 Attention 层计算出新的 K、V 后，会根据前面分配好的物理位置把 KV Cache 写入这些新 Block。这里要注意，**新 Block 并不是刚分配下来就马上写进前缀哈希表，而是等这个完整 Block 的 KV 真正计算完成以后，才把它对应的前缀哈希和物理 Block 编号登记到前缀缓存表中。** 这样后面再有新的请求拥有相同前缀时，就可以直接命中并复用这些已经计算好的 KV。

所以整个过程可以概括成：**先从 Prompt 开头做前缀哈希匹配，命中的 Block直接复用，第一次不命中以后给剩余部分分配新 Block；然后只计算未命中的 Prefill 部分，把新的 KV 写进这些 Block；等完整 Block 计算完成后再登记到 Prefix Cache，供后续请求继续复用。**

### 5. 讲一下nanovllm的前缀匹配机制（面试版）

当一条新请求被调度器选中准备做 Prefill 时，首先会由 BlockManager 判断当前 KV Cache 空间是否能够满足这条请求的执行需求，如果kvcache空间足够，它不会一上来就给整个请求的提示词的blocktable全部分配新的 KV Block，而是先根据这条请求的 Token ID，从第一个完整 Block 开始依次计算前缀哈希，并到全局的前缀哈希表中查询。 如果某个 Block 的哈希能够命中已经计算过的缓存 Block，并且 Token 内容也确认一致，那么这部分 KV 就可以直接复用，不需要重新分配新的物理 Block，也不需要重新做这部分 Prefill，只需要把对应的物理 Block 加到当前请求的 `block_table` 中，并增加它的引用。这些未命中的 Token 就会给它分配新的物理 KV Block。这样最终一条请求的 `block_table` 里面，前面可能是复用其他请求已经计算好的 Block，后面是这次新申请的 Block。之后请求真正进入 Prefill，模型只需要计算没有命中的那部分 Token，等后续不断decode新增加的这个完整 Block 的 KV 真正计算完成以后，调度器才会把它对应的前缀哈希和物理 Block 编号登记到前缀缓存表中。 这样后面再有新的请求拥有相同前缀时，就可以直接命中并复用这些已经计算好的 KV。

## Chunked Prefill

### 6. 简要讲一下chunckedprefill，作用是什么

Chunked Prefill 可以理解成：**一个很长的 Prompt 不再一次性全部做 Prefill，而是切成几个小块分多轮计算。**比如一个请求有 8K 个输入 Token，如果一次全算，GPU 会被这个长请求占很久，**其他正在 Decode 的请求就可能一直等，导致生成延迟变差**；Chunked Prefill 可以把它切成比如 2K、2K、2K、2K 四段，每轮只算一部分，**同时让 Scheduler 在这些 Prefill 小块之间穿插其他请求的 Decode。**它的主要作用就是避免长 Prompt 一次占满本轮计算预算，**改善 Decode 延迟和请求之间的公平性**，同时还能更灵活地利用每轮的 token budget。所以它本质上不是改变模型计算结果，而是改变 Prefill 的调度粒度，用更细的分块来平衡吞吐和延迟。

### 7. nanovllm中chuncked prefill是怎么运用的，在哪一个环节运用，源代码在哪个文件

在我的 nano\-vLLM 项目里，**Chunked Prefill 主要是在调度阶段实现的，核心代码就在 ****`nanovllm/engine/scheduler.py`**** 的 ****`Scheduler.schedule()`**** 里**。它不是提前把 Prompt 固定切成 2K、4K 这种大小，而是**根据当前这一轮剩余的 ****`max_num_batched_tokens`**** 动态决定这次算多少 Token。**

比如一条 Prompt 有 20K Token，而这一轮最多允许执行 16K Token，那么 Scheduler 不会一次把 20K 全部送进去，而**是设置这条 Sequence 的 ****`num_scheduled_tokens=16K`****，只调度前 16K 做 Prefill**；因为 Prompt 还没算完，这条 Sequence 暂时不会进入正常 Decode，而是保留着当前进度。接下来 `LLMEngine.step()` 把这个 Sequence 和 `is_prefill=True` 交给 `nanovllm/engine/model_runner.py`，在 `ModelRunner.prepare_prefill()` 里会根据 `num_cached_tokens` 和 `num_scheduled_tokens` 算出这一轮真正处理的区间，比如第一次是 `[0, 16384)`，然后只把这部分 `input_ids`、`positions`、`slot_mapping` 等整理出来送进模型，Attention 也只计算这一块的 KV。模型执行完成以后，再回到 `scheduler.py` 的 `Scheduler.postprocess()`，把这 16K Token 标记为已经计算完成，更新 `num_cached_tokens` 和 KV 状态；因为请求还剩 4K 没做完，所以不会采样生成 Token，而是下一轮再次进入 Prefill。**下一次 Scheduler 再看到它时，就从之前结束的位置继续**，比如处理 `[16384, 20000)`，**等最后 4K 也完成以后，Sequence 才真正转入 ****`RUNNING`****，开始后面的 Decode**。所以我会把源码链路概括成：`scheduler.py::schedule()` 决定这一轮 Prefill 切多少 Token → `model_runner.py::prepare_prefill()` 根据这个数量只准备对应的一段 Prompt → 模型完成这一段计算 → `scheduler.py::postprocess()` 更新已经完成的 Prefill 进度 → 没算完继续下一轮，全部算完才进入 Decode。\*\*另外我这个实现还有一个细节，就是**只有当前 Prefill 批次里的第一个请求允许因为 token budget 不够而被切块，后面的请求如果剩余预算放不下，就留到下一轮**，这也是源码里 `only allow chunked prefill for the first seq` 那段逻辑。

### 8. only allow chunked prefill for the first，这是不是意味着当一个请求的prompt足够长时才会触发，一个prompt被切成两段，第一轮只跑前面一段，第二轮如果token预算够就可以和waiting队列后面的请求一起prefill？解释一下这个过程

对，你这个理解基本是对的。`only allow chunked prefill for the first seq` 的准确意思是：**在同一轮 Scheduler 组成的 Prefill batch 里，只允许第一个被选中的 waiting 请求因为 token budget 不够而被“切一段执行”，后面的请求如果剩余预算装不下，就直接留到下一轮，而不会再切第二个请求。**

比如系统一轮最多允许 8K 个 Token，现在 waiting 队列里依次有请求 A 和 B，A 的 Prompt 有 12K，B 有 2K。第一轮开始时 `scheduled_seqs` 还是空的，A 需要 12K，但当前只有 8K 预算，因为 A 是这一轮第一个请求，所以允许 Chunked Prefill，给它设置 `num_scheduled_tokens=8K`，只算前 8K；这时 8K 预算已经全部用完，A 还有 4K 没 Prefill 完，所以它不会进入 running，而是继续留在 waiting 队首，B 这一轮也不会执行。第二轮 Scheduler 的 token budget 重新恢复成 8K，还是先看到 A，这次 A 只剩 4K，所以可以把剩余 4K 全部调度掉，A 的 Prefill 至此完成并进入 running；此时这一轮还剩 4K token budget，Scheduler 就会继续往 waiting 后面看 B，B 只有 2K，所以 B 也可以一起加入这一轮的 Prefill batch，最终这一轮就是 **A 的后 4K \+ B 的 2K 一起做 Prefill**。如果 B 此时不是 2K，而是 6K，因为剩余预算只有 4K，而且前面已经有 A 被放进 `scheduled_seqs` 了，就会触发 `remaining < num_tokens and scheduled_seqs`，因此 B 不会再被切成 4K，而是整个留到下一轮。

还有一点要注意：\*\***Chunked Prefill 并不一定要求 Prompt 本身“特别长”，准确说是它当前还需要 Prefill 的 Token 数超过这一轮可用的 token budget 时才触发。**\*\*另外在 nano\-vLLM 这个调度器里，只要这一轮成功调度了 Prefill，就会直接返回 Prefill batch，所以即使 A 第二轮做完后已经进入 running，也不会在同一轮再给它做 Decode；这一轮可以继续批其他 waiting 请求的 Prefill，但 Decode 还是要等没有 Prefill 被调度的轮次才能执行。

### 9. 在nanovllm这个系统当中，chuncked prefill是不是没发挥什么作用，因为prompt被切成几段之后，一段跑完发现waiting中还有下一段会继续跑prefill，decode仍然被阻塞

对，如果严格按照我这个 nano\-vLLM 项目当前 `scheduler.py` 的实现来看，你这个理解基本是对的：**它虽然实现了 Chunked Prefill，但并没有实现正式 vLLM 里那种“把长 Prefill 切开以后，在中间穿插 Decode”的主要效果。**

具体来说，一条很长的 Prompt 在 `Scheduler.schedule()` 里如果超过这一轮的 `max_num_batched_tokens`，Scheduler 会通过 `num_scheduled_tokens` 只让它先算一部分，比如 20K Prompt、本轮预算 8K，就先 Prefill 前 8K；这一轮结束后 `postprocess()` 会把已经完成的进度记录下来，但是因为 Prompt 还没有 Prefill 完，这条 Sequence 仍然留在 `waiting` 队列里，而且还是队首。下一轮再次调用 `schedule()` 时，代码首先还是执行 `while self.waiting` 这一段，所以它又会优先拿同一条请求继续做下一段 8K Prefill，而不会先去执行 `running` 队列里的 Decode；并且代码后面还有一句 `if scheduled_seqs: return scheduled_seqs, True`，只要这一轮调度到了任何 Prefill 请求，就直接返回了，根本不会继续走下面的 Decode 调度。

因此假设有一个 20K Prompt，被切成 `8K + 8K + 4K`，实际执行很可能就是 `Prefill 8K → Prefill 8K → Prefill 4K → 然后才恢复 Decode`，所以从“降低已有请求 Decode 被长 Prompt 阻塞的时间”这个角度来说，它确实没有发挥完整版 Chunked Prefill 最重要的作用。

不过也不能说它完全没作用，它至少还有三个作用：第一，**允许 Prompt 长度超过单轮 ****`max_num_batched_tokens`****，否则一个 20K Prompt 在 8K token budget 下根本没法被调度**；第二，把一次很大的 Prefill forward 拆成多个较小 forward，**控制单轮计算量和临时显存峰值**；第三，保证每轮不会突破 Scheduler 的 token budget。所以更准确地说，我这个 nano\-vLLM 实现的是“Prefill 分块”，但没有实现“Prefill 和 Decode 混合调度”；**它解决了超长 Prompt 单轮放不下的问题，但没有真正解决长 Prefill 阻塞 Decode 的延迟问题**。正式 vLLM 的 Chunked Prefill 会做得更完整，一般会优先保证 Decode，再把这一轮剩余的 token budget 拿来执行一部分 Prefill，这样长 Prompt 才真正能够和正在生成的请求交错执行。这也是 nano\-vLLM 为了代码简单而相对正式 vLLM 做的一处明显简化。

### 10. 那如果我不做token预算，尽管prompt很长也只让它进行一次forward，会出现什么问题，这样为什么不可以

可以一次性把一个很长的 Prompt 全部做完 Prefill，不是数学上做不了，而是在推理系统里风险和代价都比较大，所以 Scheduler 才要设置 token budget。

比如一个请求突然来了 32K Token，如果完全不限制，一次 forward 就把 32K 全送进模型，首先**这一轮需要准备的 ****`input_ids`****、中间激活值、Attention 的临时计算空间以及新增的 KV Cache 都会明显变大**，很容易**超过当前 GPU 剩余显存导致 OOM；**其次，这个长 Prefill 会长时间独占 GPU，假设系统里还有其他请求正在 Decode，本来用户每隔几十毫秒能收到一个 Token，现在可能要等这个 32K Prompt 整体 Prefill 完以后才能继续生成，延迟会突然变得很差；另外如果同时来了多个长 Prompt，没有 token budget 的话，Scheduler 也很难控制这一轮到底组成多大的 batch，显存和计算量就会变得不可预测。**所以 nano\-vLLM 里设置 ****`max_num_batched_tokens`****，本质上就是给每一轮 forward 设置一个计算规模上限。**比如一轮预算是 8K，而 Prompt 有 20K，就拆成 `8K + 8K + 4K` 分三轮完成，这样可以**控制每轮显存和计算量，也让 Scheduler 有机会在不同请求之间重新调度**。需要说明的是，如果 GPU 显存非常充足，而且系统里只有这一条请求，那一次性做完整 Prefill当然可以，而且可能还会因为减少了多次 forward 的启动开销而更快；token budget 主要解决的是线上多请求场景下的显存安全、调度可控和延迟公平问题。

### 11. 详细讲一下vllm中chuncked prefill的原理，他为什么可以在prefill中夹杂decode

vLLM 里的 **Chunked Prefill**，核心不是单纯“把一个长 Prompt 切成几段”，而是**把长 Prefill 切小以后，让 Scheduler 可以把 Prefill 和正在运行的 Decode 请求放到同一轮执行里**。

当前 vLLM V1 在能够使用 Chunked Prefill 时默认开启，而且调度策略会**优先安排 Decode**，**然后再把这一轮剩余的 ****`max_num_batched_tokens`**** 预算拿给 Prefill；如果剩余预算装不下完整 Prompt，就只取其中一段。 **

比如一轮 token budget 是 8192，现在已经有 3 个请求 A、B、C 正在 Decode，正常情况下它们这一轮各需要计算 1 个新 Token，那么 Scheduler 先把这 3 个 Decode Token 放进 batch，占掉 3 个预算；这时又来了一个 20K 的长 Prompt D，剩余预算是 8189，那么 D 这一轮就只 Prefill 8189 个 Token，于是这一轮实际执行的数据可以理解成 **`A的1个Decode + B的1个Decode + C的1个Decode + D的8189个Prefill Token`**。下一轮 A、B、C 又各自继续 Decode 一个 Token，然后 D 再继续算下一段 Prompt，所以整体效果就是 `Decode + Prefill块 → Decode + Prefill块 → Decode + Prefill块……`，而不是像简单版 nano\-vLLM 那样把 D 的所有 Prefill 块连续算完以后再恢复 Decode。

那为什么 Prefill 和 Decode 明明计算方式不同，却可以放进同一次 forward？因为从 Transformer 前面的角度看，它们本质上都是一批“这一轮需要计算的新 Token”，都要经过 Embedding、Attention、MLP；真正不同的是 **Attention 怎么读取历史信息**。Prefill 的某个 chunk，比如 D 的 `[0:8189]`，这一轮需要对这一大段 Token 做因果注意力并把产生的 K/V 写入 KV Cache；A、B、C 的 Decode 则各自只有一个新的 Query，它们直接通过自己的 `block_table` 和 `context_len` 去读取之前已经保存好的历史 KV。ModelRunner 会把这些 Token 整理成一批输入，同时额外准备每条请求的序列边界、位置、KV Block 映射、上下文长度等元数据，**Attention 根据这些元数据就知道“哪些 Token 属于 Prefill D，哪些属于 Decode A/B/C，各自应该看哪些历史 KV”**，所以它们虽然物理上被打包进同一轮执行，但逻辑上仍然互不干扰。vLLM 的注意力后端**也明确支持把 Prefill Token 和 Decode Token组织到同一个扁平的一维 Query batch 中**。

而 D 的第一段 Prefill 做完以后，它产生的 KV 已经保存在 KV Cache 里，所以第二轮只需要继续送 D 的下一段 Token，**新的这一段可以读取前面第一段已经缓存好的 KV**，不需要从 Prompt 开头重新计算，这就是它能够真正“切开”的基础。这样设计有两个主要目的：第一，**Decode 优先，所以一个新来的超长 Prompt 不会长时间卡住已经在给用户输出 Token 的请求，ITL/TPOT 会更稳定**；第二，Prefill 通常偏计算密集，而 Decode 更偏显存带宽瓶颈，**把两类工作放在一个 batch 中还有机会提高 GPU 的整体利用率。**官方文档也明确把这两点作为 Chunked Prefill 的主要收益。

所以我会把它总结成一句话：**正式 vLLM 的 Chunked Prefill 本质是“Decode 优先 \+ 剩余 token budget 填 Prefill”，长 Prompt 因此被拆成多个 chunk，每轮都可以和已有请求的 Decode 一起执行，而上一块 Prefill 的 KV 又会保留下来供下一块继续使用，所以既不会重复计算，又避免了一次超长 Prefill 长时间阻塞 Decode。**

### 12. 我对vllm的chuncked prefill的理解对吗，如果并发请求数足够多直至占满max\_num\_batched\_tokens，那么这一轮就全是decode；如果这一轮running队列为空，那么就全是prefill，那这一轮prefill怎么约束呢，如果有一条prompt足够长，他会占满max\_num\_batched\_tokens吗

你的理解**基本是对的**。在 vLLM 开启 Chunked Prefill 后，可以把每一轮的 `max_num_batched_tokens` 理解成这一轮 GPU 最多允许处理多少个“新 Token”的总预算，Scheduler 会优先把正在生成的 Decode 请求放进去。比如 `max_num_batched_tokens=8192`，如果当前有足够多的 Decode 请求，它们这一轮需要处理的 Token 总数已经达到 8192，那么这一轮预算就全部被 Decode 占满，不会再塞 Prefill；如果 Decode 只占了 1000 个 Token，那剩下的 7192 个预算就可以继续拿来做 Prefill。**官方当前的调度策略也是先安排 Decode，再用剩余 token budget 安排 Prefill。**

如果当前根本没有正在 Decode 的请求，那这一轮基本就会由 Prefill 来使用整个预算。假设还是 8192 的预算，现在 waiting 里第一条请求的 Prompt 有 20K Token，那么它不会一次 forward 把 20K 全算完，而是会触发 Chunked Prefill，**这一轮最多取其中 8192 个 Token 做 Prefill，也就是说这一条长 Prompt 完全可以把这一轮的 ****`max_num_batched_tokens`**** 吃满**；第一轮算 `[0,8192)`，第二轮如果仍然没有 Decode，就继续算 `[8192,16384)`，第三轮再算剩下的部分。vLLM 的配置说明也明确指出，**Prefill 会根据剩余的 ****`max_num_batched_tokens`**** 被切块。**

如果第一条 Prompt 只有 3000 Token，那么它只占 3000，Scheduler 还可以继续拿后面的 Prefill 请求来填剩余的 5192，**直到预算、最大请求数或者 KV Cache 等其他资源约束达到上限。**

所以可以把 vLLM 的 Chunked Prefill 调度记成一个很简单的原则：**先拿 token budget 保证 Decode，再把剩余预算尽量填 Prefill；没有 Decode 时，Prefill 可以使用整个预算，而单条超长 Prompt 最多只吃掉这一轮的预算大小，剩余部分留到下一轮继续。** 这也是为什么 `max_num_batched_tokens` 实际上控制了每一轮 forward 的最大 Token 规模，而 Chunked Prefill 则保证再长的 Prompt 也不会突破这个上限。

### 13. Chuncked prefill最核心的参数是max\_num\_batched\_tokens吗，如果改变这个参数的大小，会怎么影响整个推理系统

可以说 `max_num_batched_tokens` 是 Chunked Prefill 最核心的调度参数之一，但严格说它不是“固定的 chunk size”，而是每一轮 forward 里 Prefill 和 Decode 共同使用的 Token 总预算。官方 vLLM 对它的定义就是**“一次迭代最多处理多少个 Token”**，当开启 Chunked Prefill 时，Prefill 会根据这一轮剩余的 `max_num_batched_tokens` 被切块。 比如设置成 8192，如果这一轮 Decode 请求一共占了 1000 个 Token，那 Prefill 最多还能使用 7192；如果没有 Decode，一个 20K 的 Prompt 就可以先算 8192，剩余部分下一轮继续。

**所以把这个参数调大，单轮可以塞进更多 Prefill Token，通常 Prefill 吞吐会更高、长 Prompt 的 TTFT 可能更好，而且 forward 次数更少，但代价是一次 Prefill 占 GPU 的时间更长，更容易干扰 Decode，使 TPOT、尤其尾延迟变差，同时单轮显存和计算压力也会更大**；反过来把它调小，长 Prompt 会被切得更细，Decode 更容易及时插进来，所以 **TPOT 通常更稳定，但 Prefill 会被拆成更多轮，kernel 启动和调度开销增加，TTFT 和总吞吐可能下降**。vLLM 官方也明确指出，**较小的 ****`max_num_batched_tokens`**** 更偏向改善 ITL/TPOT，较大的值更偏向改善 TTFT 和吞吐。**所以我会把它理解成一个控制“单轮计算规模”的旋钮，本质上是**在 Prefill 吞吐、TTFT 和 Decode 延迟之间做权衡**。另外实际系统不会只看这一个参数，还会受到 `max_num_seqs`、KV Cache 空间以及 `max_num_partial_prefills` 等限制，所以不能简单理解成把这个值无限调大性能就一定更好。

### 14. prefill阶段的吞吐怎么定义，要是同一个长prompt我切成几段forward和直接一次forward，哪个吞吐高，时间主要浪费在哪里，从实际工程的角度分析

Prefill 阶段的吞吐一般可以理解成：**单位时间内 GPU 能处理多少个输入 Token**，比如一个 8K Prompt 的 Prefill 花了 0\.2 秒，那这次 Prefill 吞吐大约就是 \(8192/0\.2\\approx 4\.1\) 万 token/s；如果是**多个请求**一起 Prefill，就用**这一轮所有实际计算的 Prompt Token 总数除以耗时**。

对于**同一个长 Prompt**，如果显存完全放得下，而且没有其他请求需要照顾，通常**一次性做完整 Prefill 的原始吞吐会更高**，因为比如 16K Prompt 一次 forward，只需要做一次输入整理、一次模型调用，每一层的**大矩阵乘法规模也比较大，GPU 更容易跑满**；如果切成 `4K+4K+4K+4K` 四次，虽然前面已经算过的 Token 不会重新计算，KV Cache 会直接保留下来，但你要额外付出**四次 forward 的调度和启动开销**，包括 ModelRunner 多次准备 `input_ids`、position、slot mapping，CPU 到 GPU 的控制开销，大量 CUDA kernel 重复启动，而且每个 chunk 变小以后 GEMM 和 FlashAttention 的计算规模也变小，GPU 利用率可能下降。另外**后面的 chunk 做 Attention 时还要读取前面 chunk 已经写进 KV Cache 的历史 K/V，所以会增加一些显存访问压力**。

因此如果只比较“这一条 Prompt 尽快 Prefill 完”，**一次性大 Prefill 通常更快**。但是 Chunked Prefill 的目的并不是让这一个 Prompt 本身跑得更快，而是提高整个在线推理系统的调度质量：把长 Prefill 切开之后，Decode 请求可以插进来，TPOT 不会因为一个 32K Prompt 被卡很久，同时还能限制单轮显存和计算规模。所以工程上要区分两个概念：**单请求 Prefill 吞吐通常偏向大 chunk，而整个在线服务的延迟和有效吞吐往往需要适当切 chunk，这实际上是在 Prefill 效率和 Decode 延迟之间做权衡。**

## Continuous Batching

### 15. 简要说一下continues batching，它的作用是什么

Continuous Batching 可以理解成一种**动态批处理机制**。传统静态 batching 往往要等一整个 batch 里的请求都执行完，才能把新的请求加进来，这样只要其中有一个请求生成得特别长，其他已经结束的请求占出来的位置就会一直浪费。

Continuous Batching 的做法是**每一轮生成结束后，Scheduler 都重新看一遍当前请求状态**：已经结束的请求马上移出去，新来的请求可以马上补进来，其他没结束的请求继续 Decode。这样 batch 里的请求是不断动态进入、退出的，而不是从头到尾固定不变。

它最大的作用就是**减少 GPU 空槽和等待时间，提高 GPU 利用率和整体吞吐，同时让新请求不需要等整个旧 batch 全部结束以后才能执行**。所以我会简单概括成：**Continuous Batching 就是按每个推理迭代动态重组 batch，让请求随时进、随时出，从而提高在线推理系统的资源利用率和吞吐。**

### 16. 这个batch指的是一次性送入模型的一批token吗，那理所当然每一轮都会用调度器调度呀，详细说一下静态批处理是什么情况，vllm原生支持continues batching吗？

这里的 **batch 更准确地说是“这一轮模型 forward 被一起执行的一批请求对应的 Token”**，而不是简单理解成“一堆 Token”。比如 Decode 阶段有 A、B、C 三个请求都在运行，那么这一轮可能就是 A、B、C 各拿 1 个最新 Token，一共 3 个 Token 组成一个 batch 送进模型；如果开启 Chunked Prefill，这个 batch 里面甚至还可以同时有某个请求的几千个 Prefill Token。你**说“那理所当然每一轮都应该重新调度”其实是站在现在 vLLM 这种系统的视角看才觉得理所当然**，

**传统静态 batching 并不是这样做的**：它通常先凑齐比如 A、B、C、D 四个请求，形成一个固定 batch，然后这四个请求一起开始生成，在整个生成过程中 batch 成员基本不变。假设 A 生成 20 个 Token 就结束了，但 D 要生成 500 个 Token，那么 A 的位置虽然已经空出来，新来的请求 E 也不能马上补进来，E 要等这个 batch 整体完成以后才能进入下一批；这样越到生成后期，真正还在工作的请求越少，GPU 利用率就越差。

Continuous Batching 改变的核心就是**把调度粒度从“整个请求批次”降到了“模型迭代”**：每生成一轮以后，Scheduler 都重新检查请求，A 结束就立即移出去，新来的 E 可以下一轮马上补进来，B、C、D 则继续 Decode，所以 batch 的成员会在运行过程中不断变化。当前 vLLM 原生就支持 Continuous Batching，官方把“continuous batching of incoming requests”列为 vLLM 的核心能力，并说明它用于保持 GPU 充分利用。

所以我会概括成：**静态 batching 是“先固定一批请求，一起跑到这一批结束再换下一批”；Continuous Batching 是“每轮 forward 后都允许请求退出和新请求加入”，这也是为什么 vLLM 的 Scheduler 要以迭代为粒度持续重新组织 batch。**

### 17. 那只要是vllm类似框架都支持动态批处理吗，比如nanovllm，感觉这个更像是原生框架的优势而非技术的优势

对，可以这么理解一部分：**Continuous Batching 现在已经算是大模型推理框架的基础能力**了，不只是 vLLM 才有，像 nano\-vLLM 这种按照“每轮调度、每轮重新组织请求”来设计的框架，本质上也支持动态批处理。所以如果只是说“我的项目支持 Continuous Batching”，确实不能算一个很强的创新点，更像是**推理框架本身应该具备的能力**。

但是它背后仍然是一个重要的技术设计，因为**普通 PyTorch 直接跑模型并不会自动帮你做到这一点**，你需要有 Scheduler、请求状态管理、Paged KV Cache、请求动态加入退出以及每轮重新准备 batch 这些机制配合起来。比如 nano\-vLLM 里每次 `LLMEngine.step()` 都会重新调用 `Scheduler.schedule()`，已经结束的 Sequence 会被移除，新请求进入 waiting，正在运行的请求继续 Decode，然后重新组成这一轮送给 ModelRunner 的 batch，这其实就是 Continuous Batching。**所以面试时我不会把它说成“我实现了一个很有创新性的优化”，而会说它是 vLLM 类在线推理框架的基础能力；真正有技术含量的是调度策略怎么设计，比如 Prefill 和 Decode 怎么混合、token budget 怎么分、显存不足怎么抢占、长请求怎么避免阻塞短请求，这些才是 Continuous Batching 之上的优化空间。**

### 18. 现在真实的工程企业里面是怎么看待动态批处理的，对这个的研究上心吗，有没有新的对此的研究

现在真实工程里，**Continuous Batching 基本已经不是一个“新功能”，而是大模型在线推理框架的基础能力**。像 vLLM 现在就把 continuous batching、chunked prefill、prefix caching 都作为核心 serving 能力，所以企业一般不会单独研究“要不要动态批处理”，而是默认要有；真正还在重点研究的是**动态 batch 到底怎么组、什么时候让谁进来、一个 batch 放多大，以及在吞吐和用户延迟之间怎么取舍**。

比如线上同时有聊天、代码补全、长文档这些请求，它们 Prompt 长度、输出长度和延迟要求都不一样，如果只是简单按照先来先服务动态拼 batch，很可能出现长请求拖慢短请求或者某些用户长时间抢不到 GPU 的问题，所以现在的研究越来越偏向 **SLO 感知、优先级、公平性和自适应 batch**。2025 年已经有工作专门根据请求的输入长度、预测输出长度和不同 SLO 动态决定请求优先级，而不是固定 FCFS；FairBatching 则专门研究 Prefill 和 Decode 之间怎样公平分配资源。 到 2026 年又进一步出现了更动态的做法，比如 Kairos 会根据当前 Decode 距离 TPOT 上限还有多少余量，动态决定这一轮 batch 还能塞多少请求；ProServe 里的 SlideBatching 会随着系统负载变化动态改变 batch 边界，同时考虑不同请求的优先级。

所以如果面试官问我企业现在还重不重视 Continuous Batching，我会说：机制本身已经比较成熟，更多属于推理框架的基础设施；但它上面的调度问题仍然非常核心，现在研究的重点已经从**“动态组 batch”升级成了“根据实时负载、TTFT、TPOT、请求长度、优先级和 KV Cache 状态，自适应决定这一轮到底组成什么样的 batch”**。这部分其实仍然是目前 LLM Serving 很重要的工程优化方向。

## Chunked Prefill 与 Continuous Batching 的关系

### 19. **动态 batch 到底怎么组、什么时候让谁进来、一个 batch 放多大，以及在吞吐和用户延迟之间怎么取舍这类工作不是和chunked prefill更有关吗**

对，**这两类问题是高度相关的，但不能完全等同**。更准确地说，**Continuous Batching 决定“这一轮有哪些请求可以进 batch、哪些请求退出”，Chunked Prefill 决定“一个还在 Prefill 的长请求，这一轮到底允许放多少 Token 进 batch”**。

所以“**batch 怎么组、谁先进入、放多大”是更上层的调度问题**，而 Chunked Prefill 是其中专门处理 **Prefill 请求粒度** 的一个重要机制。

比如这一轮有 20 个正在 Decode 的请求，还有两个新来的长 Prompt。Scheduler 先要决定哪些 Decode 继续跑、两个新请求要不要进来，这属于 Continuous Batching 和整体调度；如果决定让其中一个长 Prompt 进入，这时它有 16K Token，但这一轮剩余预算只有 4K，那么就靠 Chunked Prefill 只取 4K 放进这一轮 batch。也就是说：**Continuous Batching 解决“请求级动态进出”，Chunked Prefill 解决“Prefill Token 级切分”。**它们最后都会影响同一个结果——这一轮送进模型的 batch 长什么样，所以工程研究里经常会把两者放在一起讨论。

如果再往实际工程里说，现在很多优化其实已经不太会严格区分“这是 Continuous Batching 的优化”还是“这是 Chunked Prefill 的优化”，而是统一看成 **Scheduler 调度优化**。比如根据 TPOT 判断这一轮还能不能塞 Prefill、根据长短请求调整 chunk 大小、避免 64K Prompt 阻塞短请求、控制 `max_num_batched_tokens`，这些表面上是在调 Chunked Prefill，但本质上是在决定整个动态 batch 怎么组成。

所以面试里我会总结成一句话：

**Continuous Batching 是大的框架，决定请求动态进入和退出；Chunked Prefill 是这个框架里针对长 Prefill 的细粒度调度手段，决定一个 Prefill 请求每轮贡献多少 Token。现在很多真实工程研究其实是在同时优化这两者背后的统一 Scheduler。**


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-面试准备|模块-面试准备]]
- 关联阅读：[[outputs/项目整理/专题-02-KV缓存与调度|专题-02-KV缓存与调度]]

%% 项目关联导航：结束 %%
