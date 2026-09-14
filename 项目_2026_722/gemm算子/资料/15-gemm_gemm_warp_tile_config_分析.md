# `gemm_warp_tile_config.h` 源码整体分析

## 1. 文件定位：它是 Warp Tile GEMM 的“编译期配置中心”

`gemm_warp_tile_config.h` 本身不执行矩阵乘法，也不负责启动 CUDA Kernel。它的核心职责是：

> **把不同 GEMM Kernel 的分块大小、Warp 划分方式、线程微块大小和线程数，集中定义为若干组编译期配置。**

在整个项目中，可以把几个关键文件理解为：

```text
test_perf_gemm.cpp
        │
        ▼
launcher.cu
        │
        ▼
gemm_warp_tile.cu
        │
        ├── 读取不同矩阵尺寸
        ├── 选择 Default / Small / Irregular 配置
        │
        ▼
gemm_warp_tile_config.h
        │
        └── 提供 BM、BN、BK、WM、WN、TM、TN 等编译期参数
```

因此，这个文件虽然代码量很少，却直接决定了 Kernel 的很多重要行为：

- 一个线程块计算多大的输出矩阵区域；
- 一个 Warp 计算多大的输出区域；
- 一个线程累计多少个输出元素；
- 一个线程块包含多少个线程和 Warp；
- 共享内存需要多大；
- 一个线程块的计算量和数据复用率；
- 寄存器压力、线程块并发度和 GPU 占用率；
- 小矩阵、规则矩阵和非规则矩阵分别使用哪套参数。

从工程角度看，它相当于 Warp Tile Kernel 的**性能参数表**。

---

## 2. 为什么要单独建立配置文件

在 GEMM 优化中，分块参数不是普通的“业务参数”，而是会改变 Kernel 结构的核心参数。

例如，下面两组 Block Tile：

```text
128 × 256
64 × 64
```

不仅表示线程块负责的输出范围不同，还会进一步改变：

- Grid 中线程块的数量；
- 一个 Block 需要加载的 A、B 数据量；
- 共享内存数组大小；
- Block 内 Warp 的数量；
- 每个 Warp 的位置；
- 每个线程持有的累加器数量；
- 边界区域产生的无效计算量；
- Kernel 的吞吐率和并行度。

如果把这些数字直接散落在 `gemm_warp_tile.cu` 中，代码会出现大量“魔法数字”，难以阅读和调参。

因此项目将参数集中到配置文件中，使 Kernel 实现保持统一：

```cpp
template <typename Cfg>
__global__ void gemm_warp_tile_kernel(...)
```

Kernel 不需要知道当前使用的是哪一套配置，只需要通过：

```cpp
Cfg::BM
Cfg::BN
Cfg::BK
Cfg::NUM_THREADS
```

取得当前配置即可。

这种设计实现了：

```text
同一份 Kernel 算法实现
        +
不同的编译期配置类型
        =
多套针对不同矩阵形状的专用 Kernel
```

---

## 3. `WarpTileConfig` 不是一个运行时对象

文件中定义了一个类模板：

```cpp
template<
    int BM_,
    int BN_,
    int BK_,
    int WM_,
    int WN_,
    int WNITER_,
    int TM_,
    int TN_,
    int NUM_THREADS_
>
struct WarpTileConfig;
```

它不是用来创建普通对象的，而是用来把一组整数参数封装成一个**类型**。

例如：

```cpp
using WarpTileDefaultConfig =
    WarpTileConfig<128, 256, 8, 64, 64, 4, 8, 4, 256>;
```

`WarpTileDefaultConfig` 是一个类型别名，不是一个变量，也不会在 Host 内存或 Device 内存中分配一个配置对象。

Kernel 可以通过：

```cpp
WarpTileDefaultConfig::BM
WarpTileDefaultConfig::BN
```

在编译阶段取得这些参数。

### 为什么参数使用 `static constexpr`

配置结构体内部使用：

```cpp
static constexpr int BM = BM_;
```

这里有两个重点：

- `static`：参数属于配置类型本身，不需要先创建对象；
- `constexpr`：参数在编译期就已经确定。

这些参数必须是编译期常量，因为 Kernel 中存在：

```cpp
__shared__ float AS[2][A_STRIDE * BK];
float res[WMITER * TM * WNITER * TN];
#pragma unroll
__launch_bounds__(Cfg::NUM_THREADS)
```

共享内存数组大小、本地数组大小、循环展开和 `launch_bounds` 都依赖编译期已知的值。

因此，这个文件不是简单地“保存几个数字”，而是在帮助编译器生成高度特化的 CUDA Kernel。

---

## 4. 各配置参数的含义

GEMM 计算为：

```text
C[M, N] = A[M, K] × B[K, N]
```

该 Kernel 采用多级分块：

```text
整个 C 矩阵
  └── Block Tile
        └── Warp Tile
              └── Thread Tile
```

### 4.1 `BM`、`BN`、`BK`

它们表示一个线程块处理的数据块尺寸。

```text
BM：Block 在 M 方向负责的输出行数
BN：Block 在 N 方向负责的输出列数
BK：每次沿 K 方向加载和计算的深度
```

一个线程块最终负责：

```text
C 的 BM × BN 区域
```

但不会一次加载完整的 K 维，而是将 K 切成若干段：

```text
K = BK + BK + BK + ...
```

每轮加载：

```text
A Tile：BM × BK
B Tile：BK × BN
```

然后累加到该 Block 的输出结果中。

因此：

- `BM`、`BN` 决定输出分块大小；
- `BK` 决定一次流水计算处理多少个 K 元素。

---

### 4.2 `WM`、`WN`

它们表示一个 Warp 负责的输出区域：

```text
一个 Warp 计算 WM × WN 个 C 元素
```

线程块中的 Warp 会共同覆盖整个 `BM × BN` Block Tile。

Warp 在 Block 中的排列数量为：

```text
M 方向 Warp 数 = BM / WM
N 方向 Warp 数 = BN / WN
总 Warp 数 = (BM / WM) × (BN / WN)
```

这个总 Warp 数必须和：

```text
NUM_THREADS / 32
```

保持一致，因为一个 Warp 固定包含 32 个线程。

---

### 4.3 `TM`、`TN`

`TM` 和 `TN` 描述线程在一次子块计算中负责的微块尺寸。

但在当前 Kernel 中，一个线程最终负责的输出元素数量并不一定只是：

```text
TM × TN
```

还需要考虑：

```text
WNITER
WMITER
```

因此，每个线程实际维护的累加器数量为：

```text
WMITER × TM × WNITER × TN
```

这些累加结果保存在寄存器数组 `res` 中。

`TM`、`TN` 越大：

- 单线程完成的计算越多；
- A、B 数据复用率通常越高；
- 需要的累加寄存器越多；
- 寄存器压力也越大；
- 每个 SM 能同时驻留的线程块数量可能下降。

所以它们是计算复用率与并发度之间的权衡。

---

### 4.4 `WNITER`

`WNITER` 表示一个 Warp 沿 N 方向分成多少次子块迭代。

Kernel 中会计算：

```text
WSUBN = WN / WNITER
```

也就是先将 Warp Tile 的 N 方向拆成若干个宽度为 `WSUBN` 的子块。

一个线程在每个 N 子块中计算 `TN` 个元素，因此一个线程在 N 方向最终累计：

```text
WNITER × TN
```

个输出元素。

---

### 4.5 `NUM_THREADS`

`NUM_THREADS` 表示每个线程块中的线程数。

它直接用于：

```cpp
dim3 blockDim(Cfg::NUM_THREADS);
```

同时也用于：

```cpp
__launch_bounds__(Cfg::NUM_THREADS)
```

当前三套配置分别使用：

```text
256 个线程 = 8 个 Warp
128 个线程 = 4 个 Warp
```

线程数会影响：

- Block 内的 Warp 数；
- 全局内存到共享内存的协作加载方式；
- 每个线程负责的加载量；
- SM 上可同时驻留的 Block 数量；
- 总并行度与占用率。

---

## 5. 参数之间不是相互独立的

这些参数不能随意修改。它们之间存在一组必须满足的结构关系。

### 5.1 Warp 数量必须匹配

应满足：

```text
(BM / WM) × (BN / WN) = NUM_THREADS / 32
```

否则配置声明的 Warp Tile 无法正好覆盖 Block Tile。

---

### 5.2 分块尺寸需要整除

当前实现隐含要求包括：

```text
BM 能被 WM 整除
BN 能被 WN 整除
WN 能被 WNITER 整除
WSUBN 能被 TN 整除
WM 能被推导出的 WSUBM 整除
```

因为 Kernel 中大量使用整数除法计算 Warp、线程和子块位置。

---

### 5.3 `BM`、`BN` 与 `float4` 加载有关

当前 Kernel 每次以 4 个 `float` 为一组搬运数据，即一次 16 字节：

```text
4 × sizeof(float) = 16 字节
```

加载映射中会使用：

```text
BM / 4
BN / 4
```

因此当前配置中的 `BM`、`BN` 都是 4 的倍数。

Fast Path 写回 C 时也使用 `float4`，所以要求数据和列方向满足相应的对齐条件。

---

### 5.4 `BK` 与协作加载步长需要匹配

Kernel 会根据线程数推导：

```text
row_stride_a = NUM_THREADS / (BM / 4)
row_stride_b = NUM_THREADS / (BN / 4)
```

然后由所有线程协作加载 `BK` 层数据。

当前三套配置中，`BK = 8`，并且加载步长能够完整覆盖这 8 层数据，不会产生越界的异步加载。

因此，修改 `BK` 时不能只看 K 方向计算量，还必须重新检查 A、B 的线程加载映射。

---

## 6. 三套配置分别解决什么问题

文件中定义了三套配置：

```text
WarpTileDefaultConfig
WarpTileSmallConfig
WarpTileIrregularConfig
```

它们对应不同矩阵规模和形状，而不是三种不同的数学算法。

---

## 7. `WarpTileDefaultConfig`：大规模规则矩阵配置

参数为：

```text
BM = 128
BN = 256
BK = 8

WM = 64
WN = 64

WNITER = 4
TM = 8
TN = 4

NUM_THREADS = 256
```

### 7.1 Block 和 Warp 的划分

一个 Block 计算：

```text
128 × 256
```

一个 Warp 计算：

```text
64 × 64
```

所以 Warp 排列为：

```text
M 方向：128 / 64 = 2
N 方向：256 / 64 = 4

总 Warp 数：2 × 4 = 8
```

线程数为：

```text
8 × 32 = 256
```

完全匹配。

可以把一个 Block 想象为：

```text
┌────────┬────────┬────────┬────────┐
│ Warp 0 │ Warp 1 │ Warp 2 │ Warp 3 │  64 行
├────────┼────────┼────────┼────────┤
│ Warp 4 │ Warp 5 │ Warp 6 │ Warp 7 │  64 行
└────────┴────────┴────────┴────────┘
   64列     64列     64列     64列
```

---

### 7.2 每个线程负责多少输出

该配置中：

```text
WSUBN = WN / WNITER
       = 64 / 4
       = 16

WSUBM = (32 / (WSUBN / TN)) × TM
       = (32 / (16 / 4)) × 8
       = 64

WMITER = WM / WSUBM
        = 64 / 64
        = 1
```

每个线程的输出数量为：

```text
WMITER × TM × WNITER × TN
= 1 × 8 × 4 × 4
= 128
```

从 Block 总输出量验证：

```text
128 × 256 / 256
= 128 个输出/线程
```

两种计算方式一致。

这意味着每个线程需要维护 128 个 `float` 累加器，数据复用率高，但寄存器压力也很大。

---

### 7.3 共享内存规模

双缓冲共享内存包括：

```text
A：2 × BM × BK
B：2 × BK × BN
```

代入参数：

```text
2 × (128 × 8 + 8 × 256) × 4 字节
= 24 KB
```

该配置适合较大且规则的矩阵，因为：

- Block Tile 大；
- 每次加载的数据被大量复用；
- 线程块数量仍足以填满 GPU；
- 边界损失相对较少；
- 计算密度较高。

但对很小的矩阵而言，一个 Block 负责的区域过大，会造成并行 Block 数不足。

---

## 8. `WarpTileSmallConfig`：小规模规则矩阵配置

参数为：

```text
BM = 64
BN = 64
BK = 8

WM = 32
WN = 32

WNITER = 2
TM = 4
TN = 4

NUM_THREADS = 128
```

### 8.1 Block 和 Warp 的划分

一个 Block 计算：

```text
64 × 64
```

一个 Warp 计算：

```text
32 × 32
```

Warp 排列：

```text
M 方向：64 / 32 = 2
N 方向：64 / 32 = 2

总 Warp 数：2 × 2 = 4
线程数：4 × 32 = 128
```

---

### 8.2 每线程输出数量

```text
WSUBN = 32 / 2 = 16
WSUBM = (32 / (16 / 4)) × 4 = 32
WMITER = 32 / 32 = 1
```

每线程输出：

```text
1 × 4 × 2 × 4 = 32
```

验证：

```text
64 × 64 / 128 = 32
```

相比 Default 配置的 128 个累加器，这套配置每线程只负责 32 个输出，寄存器压力更低。

---

### 8.3 共享内存规模

```text
2 × (64 × 8 + 8 × 64) × 4 字节
= 8 KB
```

它适合小矩阵的主要原因不是“计算方法更简单”，而是：

- Block Tile 更小；
- 同样大小的矩阵可以启动更多 Block；
- 每个 Block 只使用 128 个线程；
- 共享内存和寄存器需求更低；
- 更容易在 SM 上同时驻留多个 Block；
- 小规模问题下 GPU 并行度更充分。

---

## 9. `WarpTileIrregularConfig`：中小型非规则矩阵配置

参数为：

```text
BM = 64
BN = 128
BK = 8

WM = 32
WN = 64

WNITER = 2
TM = 4
TN = 4

NUM_THREADS = 128
```

所谓非规则矩阵，主要指 M、N、K 不能整齐匹配 Kernel 分块和向量化条件，例如：

```text
M = 100
N = 257
K = 511
```

这类形状会产生边界 Block，并进入带边界判断的 Edge Path。

---

### 9.1 Block 和 Warp 的划分

一个 Block 计算：

```text
64 × 128
```

一个 Warp 计算：

```text
32 × 64
```

Warp 排列：

```text
M 方向：64 / 32 = 2
N 方向：128 / 64 = 2

总 Warp 数：2 × 2 = 4
线程数：4 × 32 = 128
```

---

### 9.2 每线程输出数量

```text
WSUBN = 64 / 2 = 32
WSUBM = (32 / (32 / 4)) × 4 = 16
WMITER = 32 / 16 = 2
```

每线程输出：

```text
2 × 4 × 2 × 4 = 64
```

验证：

```text
64 × 128 / 128 = 64
```

这里 `WMITER = 2`，说明 Warp 在 M 方向需要执行两组线程微块计算。

---

### 9.3 为什么它适合非规则矩阵

如果非规则矩阵仍然使用 Default 配置的 `128 × 256` Block Tile，边界 Block 中可能有很大一部分区域超出真实矩阵范围。

例如，对于较窄或不对齐的矩阵：

```text
真实有效区域较小
但线程仍按 128 × 256 的逻辑组织
```

这会增加：

- 无效线程工作；
- 边界判断；
- 无效共享内存清零；
- 无效计算；
- 最后一个 Block 的浪费。

Irregular 配置将 Block Tile 缩小为：

```text
64 × 128
```

可以降低边界区域的浪费，同时保持一定的数据复用率。

因此它是在以下两者之间折中：

```text
Small：并行度高，但 Tile 较小
Default：复用率高，但边界浪费可能较大
Irregular：针对非规则中小形状的折中配置
```

---

## 10. 三套配置对比

| 配置 | Block Tile | Warp Tile | 线程数 | Warp 数 | 每线程输出数 | 双缓冲共享内存 |
|---|---:|---:|---:|---:|---:|---:|
| Default | `128×256×8` | `64×64` | 256 | 8 | 128 | 24 KB |
| Small | `64×64×8` | `32×32` | 128 | 4 | 32 | 8 KB |
| Irregular | `64×128×8` | `32×64` | 128 | 4 | 64 | 12 KB |

从表中可以看到：

- Default 追求更大的数据复用和吞吐；
- Small 追求小问题下的并行度和较低资源占用；
- Irregular 追求边界效率与计算复用之间的平衡。

---

## 11. 配置是如何真正作用到 Kernel 中的

`gemm_warp_tile.cu` 通过模板接收配置类型：

```cpp
template <typename Cfg>
__global__ void gemm_warp_tile_kernel(...)
```

然后把配置展开为编译期常量：

```cpp
constexpr int BM = Cfg::BM;
constexpr int BN = Cfg::BN;
constexpr int BK = Cfg::BK;
```

这些参数进一步控制以下内容。

### 11.1 Grid 大小

```text
grid.x = ceil(N / BN)
grid.y = ceil(M / BM)
```

因此 `BM`、`BN` 决定整个矩阵要拆成多少个线程块。

---

### 11.2 Block 大小

```text
block.x = NUM_THREADS
```

因此不同配置会启动 128 或 256 个线程。

---

### 11.3 共享内存数组

```text
AS[2][BM × BK]
BS[2][BK × BN]
```

因此配置会在编译时确定共享内存大小。

---

### 11.4 Warp 和线程映射

`BM`、`BN`、`WM`、`WN` 决定：

- Warp 位于 Block Tile 的哪一行、哪一列；
- Warp 负责哪一块输出；
- 每个线程在 Warp Tile 中负责哪些元素。

---

### 11.5 寄存器累加器

`TM`、`TN`、`WNITER` 和推导出的 `WMITER` 决定：

```text
res 数组包含多少个 float 累加器
```

这会直接影响寄存器用量。

---

### 11.6 编译器生成不同 Kernel 实例

当代码分别调用：

```cpp
launch_gemm_warp_tile_cfg<WarpTileDefaultConfig>(...)
launch_gemm_warp_tile_cfg<WarpTileSmallConfig>(...)
launch_gemm_warp_tile_cfg<WarpTileIrregularConfig>(...)
```

编译器会分别实例化三套专用 Kernel。

它们虽然来自同一份模板源码，但编译后的线程映射、数组大小和循环展开结果并不相同。

所以这不是在运行时反复读取一张配置表，而是：

> **在编译阶段生成三份经过不同参数特化的 Kernel，再在运行时选择其中一份启动。**

---

## 12. 当前项目中的配置选择逻辑

`launch_gemm_warp_tile()` 中的选择逻辑可以概括为：

```text
如果是中小规模非规则形状
    → IrregularConfig
否则，如果是小规模规则问题
    → SmallConfig
否则
    → DefaultConfig
```

判定条件为：

```text
small_problem：
M、N、K 都不超过 1024

irregular_shape：
M 不是 64 的倍数
或 N 不是 128 的倍数
或 K 不是 8 的倍数

irregular_medium：
形状不规则
并且 M、N、K 都不超过 3072
```

需要特别注意：

> **Irregular 的判断优先于 Small。**

因此一个矩阵即使尺寸很小，只要形状不规则，也会优先使用 Irregular 配置。

例如：

| 矩阵尺寸 | 配置 |
|---|---|
| `512×512×512` | Small |
| `1024×1024×1024` | Small |
| `100×100×100` | Irregular |
| `257×257×257` | Irregular |
| `1234×1234×1234` | Irregular |
| `2047×2047×2047` | Irregular |
| `4096×4096×4096` | Default |
| 大于 3072 的非规则矩阵 | 当前仍使用 Default |

这说明当前项目采用的是**人工经验规则**，并不是运行时 Auto-Tuning。

---

## 13. 这个文件体现出的 GEMM 优化思想

### 13.1 一个 Kernel 参数无法适合所有尺寸

大矩阵需要：

- 大 Tile；
- 高数据复用；
- 较高计算密度。

小矩阵更需要：

- 更多线程块；
- 更低单 Block 资源占用；
- 更好的启动并行度。

非规则矩阵更需要：

- 减少边界浪费；
- 缩小尾部 Block；
- 控制 Edge Path 开销。

因此高性能 GEMM 通常不会只准备一套固定参数。

---

### 13.2 Tile 越大并不一定越快

更大的 Tile 通常可以提高共享内存数据复用率，但也会增加：

- 每线程累加器数量；
- 寄存器压力；
- 共享内存占用；
- 单 Block 线程数量；
- 边界区域浪费；
- 小问题下的并行度不足。

所以优化目标不是单纯让 `BM`、`BN` 越大越好，而是寻找适合特定矩阵规模和 GPU 架构的平衡点。

---

### 13.3 配置参数本质上是硬件资源分配方案

每一套配置实际上都在回答：

```text
一个 Block 要用多少线程？
一个 Block 要占多少共享内存？
每个线程要用多少寄存器？
一个 SM 能同时运行几个这样的 Block？
A、B 数据能被复用多少次？
边界处会浪费多少计算？
```

因此调节这些参数不是普通的软件参数调整，而是在重新设计 Kernel 与 GPU 硬件资源之间的映射方式。

---

## 14. 当前配置设计的优点

### 14.1 Kernel 实现与调优参数分离

算法代码放在 `.cu` 文件中，参数放在配置头文件中，结构清晰。

### 14.2 没有运行时参数读取开销

所有参数都是编译期常量，编译器可以：

- 展开循环；
- 固定数组大小；
- 传播常量；
- 删除无用分支；
- 生成专用指令序列。

### 14.3 可以方便扩展更多配置

例如可以继续定义：

```text
WarpTileTallConfig
WarpTileWideConfig
WarpTileLargeKConfig
```

然后在启动函数中增加对应选择策略。

### 14.4 同一份 Kernel 避免重复实现

不需要为每个参数组合复制一份 Kernel 源码，降低维护成本。

---

## 15. 当前设计的限制

### 15.1 缺少编译期合法性检查

目前配置类中没有使用 `static_assert` 检查参数关系。

如果未来错误地定义一套配置，例如 Warp 数不匹配，可能出现：

- 输出区域没有完全覆盖；
- 多个 Warp 重复计算同一区域；
- 数组越界；
- 异步加载越界；
- 编译成功但运行结果错误。

更稳健的设计可以增加类似检查：

```cpp
static_assert(NUM_THREADS % 32 == 0);
static_assert(BM % WM == 0);
static_assert(BN % WN == 0);
static_assert((BM / WM) * (BN / WN) == NUM_THREADS / 32);
```

这属于可以进一步完善的工程点。

---

### 15.2 当前只有三套人工配置

真实高性能 GEMM 库通常会根据更多条件选择 Kernel，例如：

- GPU 架构；
- SM 数量；
- 共享内存容量；
- M、N、K 的长宽比例；
- 数据类型；
- 对齐情况；
- Batch 数量；
- `alpha`、`beta` 是否为特殊值；
- Tensor Core 是否可用。

当前项目只根据尺寸和整除关系进行简单判断，属于教学型或初步优化框架。

---

### 15.3 参数与具体 GPU 架构相关

在一张 GPU 上表现良好的配置，在另一张 GPU 上未必最优，因为不同 GPU 的：

- 寄存器文件大小；
- 共享内存容量；
- SM 数量；
- 内存带宽；
- 指令吞吐；
- 最大驻留 Warp 数；

可能不同。

因此这些参数需要通过 Benchmark 和 Nsight Compute 实际验证，而不能仅靠理论计算决定。

---

### 15.4 配置数量增加会提高编译成本

每新增一套配置，模板会实例化一份新的 Kernel。

配置过多会带来：

- 编译时间增加；
- 二进制体积增加；
- 调度逻辑复杂；
- 测试组合增多。

所以工程中通常只保留有明确性能收益的参数组合。

---

## 16. 阅读这个文件时应建立的核心认识

阅读完该文件后，应形成以下理解：

1. `WarpTileConfig` 是编译期参数容器，不是运行时对象。
2. `using` 定义的是配置类型别名，不会分配内存。
3. `BM × BN` 是一个 Block 负责的输出区域。
4. `BK` 是一次 K 方向流水计算的深度。
5. `WM × WN` 是一个 Warp 负责的输出区域。
6. `TM`、`TN` 和迭代次数共同决定每线程的累加器数量。
7. `NUM_THREADS / 32` 必须与 Block 中 Warp Tile 的数量一致。
8. 配置会影响 Grid、Block、共享内存、寄存器、线程映射和性能。
9. 三套配置使用同一份 Kernel 模板，但会生成三份不同的编译实例。
10. Small、Irregular、Default 分别针对小矩阵、非规则中小矩阵和大规模规则矩阵。
11. 配置调优本质上是在计算复用率、资源占用、并行度和边界浪费之间进行权衡。
12. 这个文件是整个 Warp Tile GEMM 的主要调优入口之一。

---

## 17. 在整个项目中的最终意义

`gemm_warp_tile_config.h` 的价值可以概括为：

> **它把 GEMM Kernel 的线程组织和分块策略抽象成编译期配置，使项目能够用同一份 Warp Tile 实现生成多套专用 Kernel，并针对不同矩阵规模选择更合适的资源分配方案。**

它连接了三个层面：

```text
矩阵形状
    ↓
分块与线程配置
    ↓
GPU 上的实际资源使用和性能
```

所以这个文件虽然不包含矩阵乘法循环，却决定了矩阵乘法循环以什么规模、什么线程结构和什么资源占用方式在 GPU 上执行。

在后续学习中，建议把它和 `gemm_warp_tile.cu` 对照阅读：

```text
先在配置文件中确认 BM、BN、BK、WM、WN、TM、TN
再到 Kernel 中追踪这些参数如何决定：
Grid → Block → Warp → Thread → 寄存器累加器
```

这样才能真正理解 GEMM 优化中的多级 Tile，而不是只记住几组数字。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-gemm算子|模块-gemm算子]]

%% 项目关联导航：结束 %%
