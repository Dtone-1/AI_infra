# `gemm_warp_tile.cu` 源码整体分析

## 1. 文件整体概览

`gemm_warp_tile.cu` 是整个 GEMM 项目中最核心的计算实现文件。

前面的 `test_perf_gemm.cpp` 负责准备数据、调用算子、验证正确性和测试性能；`launcher.cu` 负责从统一接口进入具体算法；而真正让 GPU 线程执行矩阵乘法的工作，主要就在这个文件中完成。

它实现的数学目标是：

\[
C = \alpha AB + \beta C
\]

其中：

- `A` 的形状是 `M × K`；
- `B` 的形状是 `K × N`；
- `C` 的形状是 `M × N`；
- 所有矩阵在本项目中都按 Row-major，也就是行优先方式存储；
- 数据类型为 FP32，即 `float`。

这个文件并不是“每个线程算一个 `C[m,n]`”的朴素 GEMM，而是一个经过多级分块、共享内存复用、寄存器累加、向量化读写、异步拷贝和双缓冲优化的 CUDA Kernel。

可以把它概括成：

> 将大矩阵逐级划分为 Block Tile、Warp Tile 和 Thread Tile，使全局内存中的 A、B 数据先进入共享内存，再进入寄存器，并在寄存器中反复进行乘加，最后写回 C。

因此，这个文件是理解本项目“为什么比朴素 GEMM 快”的关键。

---

## 2. 它在整个项目中的位置

整个主要调用链可以理解为：

```text
test_perf_gemm.cpp
    ↓
launch_gemm(...)
    ↓
launcher.cu
    ↓
launch_gemm_warp_tile(...)
    ↓
launch_gemm_warp_tile_cfg<Cfg>(...)
    ↓
gemm_warp_tile_kernel<Cfg><<<gridDim, blockDim>>>(...)
    ↓
GPU 上执行分块 GEMM
```

各文件职责如下：

| 文件 | 主要职责 |
|---|---|
| `tests/test_perf_gemm.cpp` | 分配数据、调用 cuBLAS 和自研 GEMM、计时、校验结果 |
| `src/launcher.cu` | 对外提供统一入口，并选择具体 GEMM 实现 |
| `gemm_warp_tile_config.h` | 定义不同矩阵规模对应的分块参数 |
| `gemm_warp_tile.h` | 声明 `launch_gemm_warp_tile()` |
| `gemm_warp_tile.cu` | 实现 Host 侧配置调度、Kernel 启动和 GPU 端计算 |

所以，从项目架构看，这个文件同时承担两层工作：

1. **Host 侧 Kernel 调度**：根据矩阵规模选择配置，并设置 `gridDim`、`blockDim`；
2. **Device 侧 GEMM 实现**：完成数据搬运、分块计算、边界处理和结果写回。

---

## 3. 这个文件解决的核心性能问题

朴素 GEMM 中，若每个线程独立计算一个输出元素，就需要不断从全局显存读取 A 的一行和 B 的一列。

例如：

\[
C[m,n] = \sum_{k=0}^{K-1} A[m,k]B[k,n]
\]

对于相邻的多个输出元素，A 和 B 中的大量数据会被重复读取。如果每次都从显存重新加载，性能会严重受到显存带宽和访问延迟限制。

该文件通过以下思路解决这个问题：

1. 一个 Thread Block 计算 C 的一大块区域；
2. A、B 对应的数据块只从全局内存加载一次到 Shared Memory；
3. 同一个 Block 内的多个 Warp、Thread 共同复用这些数据；
4. 每个线程在寄存器中同时累加多个 C 元素；
5. 沿 K 方向分批加载和计算，避免一次存下完整矩阵；
6. 使用 `cp.async` 和双缓冲，让下一块数据的搬运与当前块的计算尽量重叠。

这正是高性能 GEMM 的基本思想：

> 用分块提高数据复用率，用共享内存减少全局内存访问，用寄存器完成高密度乘加。

---

## 4. 文件的整体结构

从功能上看，该文件可以划分为六个部分。

### 4.1 基础宏和共享内存地址转换

文件首先定义：

- `CEIL_DIV`：向上取整除法，用于计算需要多少个 Thread Block；
- `WARP_SIZE = 32`：一个 Warp 固定包含 32 个线程；
- `smem_u32addr()`：把普通指针转换为 PTX 指令可使用的 Shared Memory 地址。

`smem_u32addr()` 并不执行矩阵计算，它是后面使用 `cp.async` 的基础设施。

因为内联 PTX 中的 `cp.async` 需要 Shared Memory 地址空间中的 32 位地址，普通 C++ 指针不能直接使用，所以要进行地址空间转换。

### 4.2 `cp.async` 辅助函数

文件封装了三类底层操作：

- `cp_async16()`：一次从 Global Memory 异步搬运 16 Byte 到 Shared Memory，即 4 个 `float`；
- `cp_async_commit()`：把之前发出的异步拷贝提交为一个 group；
- `cp_async_wait_group<N>()`：等待异步拷贝完成到指定程度。

这部分直接使用内联 PTX，说明该 Kernel 已经不只是普通 CUDA C++ 级别优化，而是开始显式控制 GPU 的异步数据搬运流水线。

### 4.3 三类计算函数

文件包含：

- `compute()`；
- `compute_k()`；
- `compute_edge()`。

它们完成的核心工作相同：

1. 从 Shared Memory 读取当前 K 位置的 A、B 数据；
2. 放入线程私有寄存器 `reg_m`、`reg_n`；
3. 做外积式乘加；
4. 把结果累加到寄存器数组 `res`。

三者的区别在于处理场景不同：

| 函数 | 使用场景 |
|---|---|
| `compute()` | 完整的 `BK`，且 M、N 块均完整的快速路径 |
| `compute_k()` | M、N 块完整，但 K 方向最后不足一个 `BK` |
| `compute_edge()` | M 或 N 落在矩阵边缘，需要边界保护 |

将规则区域和边界区域分开处理，是典型的高性能 Kernel 设计方式。规则区域走更少判断、更易展开的 Fast Path；边缘区域通过额外判断保证正确性。

### 4.4 异步 Tile 加载函数

`load_tile_async()` 负责将当前 K Tile 对应的 A、B 数据从 Global Memory 搬入 Shared Memory。

它的核心特点是：

- 每次搬运 4 个连续 `float`；
- 多个线程合作加载一个完整 Tile；
- A 在写入 Shared Memory 时被重新排布；
- B 保持适合后续计算的 `[BK][BN]` 布局；
- A、B 的加载被放入同一个 `cp.async` group。

这部分是“数据搬运阶段”，而 `compute()` 是“计算阶段”。二者组合后形成软件流水线。

### 4.5 主 Kernel

`gemm_warp_tile_kernel<Cfg>()` 是 GPU 上真正运行的 CUDA Kernel。

它负责：

- 从配置 `Cfg` 中取出所有分块参数；
- 计算当前 Block、Warp、Thread 对应的矩阵区域；
- 创建双缓冲 Shared Memory；
- 为每个线程创建寄存器累加数组；
- 判断走 Fast Path 还是 Edge Path；
- 沿 K 轴循环加载并计算；
- 最后执行 `alpha * AB + beta * C` 并写回。

### 4.6 Host 侧配置与启动

文件结尾包含两个 Host 函数：

- `launch_gemm_warp_tile_cfg<Cfg>()`；
- `launch_gemm_warp_tile()`。

前者负责根据模板配置启动 Kernel；后者负责根据 M、N、K 选择合适的配置。

因此，本文件并不是单一 Kernel，而是：

> 一个模板化 Kernel + 三套配置 + 一层运行时 Dispatch。

---

## 5. 最重要的概念：多级分块

理解这个文件的关键不是某一行代码，而是理解以下四级映射：

```text
完整矩阵
  ↓
Thread Block Tile
  ↓
Warp Tile
  ↓
Thread Tile
  ↓
线程寄存器中的多个输出元素
```

---

## 6. 第一级：Matrix → Thread Block Tile

每个 Thread Block 负责计算 C 中一个 `BM × BN` 的区域。

例如默认配置：

```text
BM = 128
BN = 256
BK = 8
```

意味着一个 Block 最终负责：

```text
C 的 128 行 × 256 列
```

但它不会一次沿整个 K 维完成计算，而是每次只处理：

```text
A Tile: BM × BK = 128 × 8
B Tile: BK × BN = 8 × 256
```

然后沿 K 维移动：

```text
K = 0  ~ 7
K = 8  ~ 15
K = 16 ~ 23
...
```

每处理一个 K Tile，就把部分乘加结果累积到寄存器中。

Grid 大小为：

```cpp
gridDim.x = ceil(N / BN)
gridDim.y = ceil(M / BM)
```

因此：

- `blockIdx.x` 决定 C 的列块；
- `blockIdx.y` 决定 C 的行块。

---

## 7. 第二级：Thread Block Tile → Warp Tile

一个 Thread Block 中包含多个 Warp，每个 Warp 负责 C Block 中的一个 `WM × WN` 区域。

默认配置为：

```text
BM = 128
BN = 256
WM = 64
WN = 64
NUM_THREADS = 256
```

256 个线程等于 8 个 Warp。

C Block 可以分成：

```text
M 方向：BM / WM = 128 / 64 = 2
N 方向：BN / WN = 256 / 64 = 4
```

所以总共有：

```text
2 × 4 = 8 个 Warp Tile
```

刚好对应 8 个 Warp。

可以把一个 Block 想成：

```text
+---------+---------+---------+---------+
| Warp 0  | Warp 1  | Warp 2  | Warp 3  |
| 64×64   | 64×64   | 64×64   | 64×64   |
+---------+---------+---------+---------+
| Warp 4  | Warp 5  | Warp 6  | Warp 7  |
| 64×64   | 64×64   | 64×64   | 64×64   |
+---------+---------+---------+---------+
```

这就是 `warp_row` 和 `warp_col` 的意义：确定当前 Warp 在 Block Tile 中负责哪一块。

---

## 8. 第三级：Warp Tile → Thread Tile

一个 Warp 有 32 个线程，而默认情况下一个 Warp 要负责 `64 × 64 = 4096` 个输出元素。

因此，不可能一个线程只计算一个输出元素。每个线程必须负责多个 C 元素。

默认配置中的相关参数为：

```text
WNITER = 4
TM = 8
TN = 4
```

由此计算：

```text
WSUBN = WN / WNITER = 64 / 4 = 16
WSUBM = 64
WMITER = WM / WSUBM = 1
```

一个 Warp 会在 N 方向分 4 轮计算，每轮计算一个 `64 × 16` 的区域。

每一轮中，一个线程负责：

```text
TM × TN = 8 × 4 = 32 个输出元素
```

一共 4 轮，所以一个线程最终负责：

```text
8 × 4 × 4 = 128 个输出元素
```

整个 Warp 共 32 个线程：

```text
32 × 128 = 4096
```

正好覆盖一个 `64 × 64` 的 Warp Tile。

这也解释了为什么 `res` 是一个很大的线程私有数组：它保存当前线程负责的全部输出元素的中间结果。

---

## 9. 数据在不同存储层级中的流动

该文件最核心的数据流为：

```text
Global Memory
    ↓ cp.async / scalar load
Shared Memory
    ↓ 普通读取
Thread Registers
    ↓ 乘加累积
Result Registers
    ↓ float4 或标量写回
Global Memory 中的 C
```

### 9.1 Global Memory 中的数据

输入矩阵：

```text
A: [M][K]
B: [K][N]
C: [M][N]
```

均采用 Row-major。

### 9.2 Shared Memory 中的数据

Kernel 中定义了：

```text
AS[2][BK × BM]
BS[2][BK × BN]
```

最前面的 `2` 表示双缓冲。

虽然 A 在全局内存中的逻辑 Tile 是 `[BM][BK]`，但进入 Shared Memory 后被保存成更接近 `[BK][BM]` 的形式：

```text
AS[d][m]
```

也就是：

- 第一维 `d` 对应 K 方向；
- 第二维 `m` 对应 M 方向。

这样，在计算某个 K 位置时，同一 Warp 中线程读取 M 方向数据更方便。

B 则保存为：

```text
BS[d][n]
```

即 `[BK][BN]`。

### 9.3 操作数寄存器

每个线程维护：

```text
reg_m[WMITER × TM]
reg_n[WNITER × TN]
```

它们分别缓存当前 K 位置下，线程需要使用的 A 数据和 B 数据。

### 9.4 结果寄存器

每个线程还维护：

```text
res[WMITER × TM × WNITER × TN]
```

所有 K Tile 的部分和都累加在这里。

直到完整 K 维计算结束后，才统一写回 C。

这避免了每一次乘加都访问全局内存。

---

## 10. 核心计算为什么是“外积式”

对于某个固定的 K 位置 `d`：

- 从 A 中取出线程负责的若干 M 方向值；
- 从 B 中取出线程负责的若干 N 方向值；
- 两组值两两相乘并累加到输出元素。

可以抽象成：

```text
A 寄存器向量：a0, a1, a2, ...
B 寄存器向量：b0, b1, b2, ...

产生：
a0*b0, a0*b1, ...
a1*b0, a1*b1, ...
...
```

这相当于一次小型外积。

相比每次只计算一个 C 元素，这种方式可以让一份 A 寄存器数据与多份 B 数据复用，也可以让一份 B 数据与多份 A 数据复用，提升每次数据加载对应的浮点运算数量。

---

## 11. Fast Path 与 Edge Path

该 Kernel 并不使用同一套逻辑处理所有矩阵，而是分成两条路径。

### 11.1 Fast Path

满足以下条件时进入快速路径：

- 当前 Block 在 M 方向是完整的；
- 当前 Block 在 N 方向是完整的；
- `K` 可以满足 4 个 `float` 的向量化访问要求；
- `N` 可以满足 4 个 `float` 的向量化访问要求。

Fast Path 的特点是：

- 使用 `cp.async`；
- 每次加载 16 Byte；
- 使用双缓冲；
- 主计算循环中的边界判断很少；
- 写回时使用 `float4`。

这条路径主要服务于规则、对齐良好的矩阵区域。

### 11.2 Edge Path

当 M、N 边缘不完整，或者访问不满足快速路径条件时，进入 Edge Path。

它会：

- 先把 Shared Memory 清零；
- 对 M、N、K 的有效范围进行判断；
- 只加载合法元素；
- 对越界位置使用 0；
- 写回时逐元素检查边界。

它的主要目标是保证任意矩阵尺寸下结果正确，而不是追求最高性能。

因此，这个 Kernel 体现了一个重要工程思想：

> 高性能实现需要同时拥有“规则尺寸快速路径”和“任意尺寸正确路径”。

如果只实现 Fast Path，Kernel 只能处理特定倍数尺寸；如果所有区域都使用 Edge Path，又会因大量判断损失性能。

---

## 12. `cp.async` 和双缓冲在做什么

### 12.1 没有双缓冲时

传统流程是：

```text
加载 Tile 0
等待加载完成
计算 Tile 0
加载 Tile 1
等待加载完成
计算 Tile 1
...
```

加载和计算基本串行进行。

### 12.2 当前实现的思路

Kernel 创建两份 Shared Memory Buffer：

```text
Buffer 0
Buffer 1
```

执行时大致为：

```text
先加载 Tile 0 到 Buffer 0
等待 Tile 0 完成

计算 Tile 0 时，加载 Tile 1 到 Buffer 1
计算 Tile 1 时，加载 Tile 2 到 Buffer 0
计算 Tile 2 时，加载 Tile 3 到 Buffer 1
...
```

通过 `cur` 和 `nxt` 在两个 Buffer 之间交替。

理想情况下：

```text
当前 Tile 的计算时间
        与
下一个 Tile 的显存搬运时间
```

能够部分重叠，从而隐藏一部分 Global Memory 延迟。

需要注意，`cp.async` 不是让数据“自动立刻可用”。程序仍然必须：

1. 提交异步拷贝 group；
2. 等待 group 到达可用状态；
3. 使用 `__syncthreads()` 确保整个 Block 都可以安全读取 Shared Memory。

---

## 13. 为什么同时需要 `cp.async_wait_group` 和 `__syncthreads()`

它们解决的问题不同。

### `cp.async_wait_group`

保证当前线程发起的异步拷贝已经完成到指定阶段，数据已经写入 Shared Memory。

### `__syncthreads()`

保证整个 Thread Block 中所有线程都到达同步点。

因为一个 Tile 是由整个 Block 中多个线程共同加载的，所以某个线程不仅依赖自己加载的数据，也可能依赖其他线程加载的数据。

因此，只有等待异步拷贝和 Block 同步都完成后，才能安全开始计算。

---

## 14. K 方向主循环与尾部处理

GEMM 的 K 维被按 `BK` 切成多个 Tile。

假设：

```text
K = 100
BK = 8
```

则：

```text
完整 Tile 数量 = 100 / 8 = 12
剩余 K = 100 % 8 = 4
```

前 12 个完整 Tile 使用 `compute()`。

最后剩余 4 个 K 元素：

- 先清零 Shared Memory；
- 只加载合法的 4 个 K 位置；
- 使用 `compute_k(..., k_rem=4)`；
- 不计算后面补零的位置。

这就是 K remainder 处理。

它与 M、N 边界不同：

- M、N 边界决定当前 Block 是否超出输出矩阵；
- K remainder 决定最后一轮归约是否不足 `BK`。

---

## 15. 结果写回与 `alpha`、`beta`

结果最终不是简单写成：

```text
C = res
```

而是遵循标准 GEMM 接口：

\[
C = \alpha \cdot res + \beta \cdot C
\]

其中 `res` 就是计算得到的 `AB`。

### Fast Path 写回

使用 `float4`：

- 一次读取 4 个连续 C 元素；
- 完成 `alpha*res + beta*C`；
- 一次写回 4 个连续元素。

这样可以减少指令数量，并形成 16 Byte 向量化访问。

### Edge Path 写回

逐元素检查：

- 行坐标是否小于 M；
- 列坐标是否小于 N；
- 只有合法位置才写回。

这再次体现：快速路径追求吞吐，边界路径保证通用性。

---

## 16. 三套 Kernel 配置

该文件通过 `gemm_warp_tile_config.h` 中的三套配置适应不同问题规模。

### 16.1 Default Config

```text
Block Tile: 128 × 256 × 8
Warp Tile:   64 × 64
线程数:      256
```

适合较大的规则矩阵。

特点：

- 8 个 Warp；
- 每个 Block 计算大量输出；
- 数据复用率较高；
- 每线程结果寄存器较多；
- 更适合大问题摊薄 Kernel 启动和调度开销。

### 16.2 Small Config

```text
Block Tile: 64 × 64 × 8
Warp Tile:  32 × 32
线程数:     128
```

适合 M、N、K 都不超过 1024 的小问题。

较小 Tile 可以减少：

- 不必要的线程工作；
- Shared Memory 使用量；
- 小矩阵中的资源浪费。

### 16.3 Irregular Config

```text
Block Tile: 64 × 128 × 8
Warp Tile:  32 × 64
线程数:     128
```

用于中小规模的非规则尺寸。

它在规则大配置和小配置之间折中，目的是降低非对齐矩阵在边界处的浪费。

---

## 17. 配置选择逻辑

`launch_gemm_warp_tile()` 根据矩阵规模进行运行时选择。

大体规则是：

1. 如果矩阵形状不规则，且 M、N、K 都不超过 3072，选择 `IrregularConfig`；
2. 否则，如果 M、N、K 都不超过 1024，选择 `SmallConfig`；
3. 其他情况选择 `DefaultConfig`。

这里需要区分两个概念：

### 运行时 Dispatch

根据实际输入 M、N、K，决定使用哪一套配置。

### 编译期模板实例化

每一套 `Cfg` 中的参数都是 `static constexpr`，编译器可以：

- 展开循环；
- 固定数组大小；
- 预先计算索引关系；
- 针对不同配置生成独立 Kernel 代码。

所以这个设计同时利用了：

```text
运行时的输入适配能力
+
编译期的常量优化能力
```

---

## 18. `__launch_bounds__` 的意义

主 Kernel 使用：

```text
__launch_bounds__(Cfg::NUM_THREADS)
```

它告诉编译器：

> 这个 Kernel 每个 Block 预计最多使用 `Cfg::NUM_THREADS` 个线程。

编译器可以据此做寄存器分配和占用率相关优化。

这是有必要的，因为该 Kernel 每个线程的 `res` 数组可能很大，寄存器压力很高。

例如默认配置下，一个线程需要维护大量 FP32 累加结果。寄存器越多，单个 Block 的计算能力可能越强，但同一 SM 上能同时驻留的 Block/Warp 数量可能下降。

因此，GEMM 优化通常不是“寄存器越多越好”，而是在以下因素之间平衡：

- 数据复用；
- 指令级并行；
- 寄存器数量；
- Shared Memory 数量；
- Occupancy；
- Warp 调度能力。

---

## 19. 这个实现的主要优化点

从工程角度，该文件包含以下关键优化。

### 19.1 Block Tiling

一个 Block 计算 C 的大块区域，A、B Tile 可被整个 Block 复用。

### 19.2 Warp Tiling

将 Block Tile 进一步划分给不同 Warp，减少线程间复杂协作。

### 19.3 Thread Tiling

每个线程计算多个输出元素，提高寄存器复用和计算密度。

### 19.4 Shared Memory Reuse

A、B 数据进入 Shared Memory 后，被同一个 Block 中多个线程多次读取。

### 19.5 Register Blocking

每个线程在寄存器中保存 A、B 操作数和多个 C 累加结果。

### 19.6 Vectorized Access

使用 16 Byte 搬运和 `float4` 写回，每次处理 4 个 `float`。

### 19.7 `cp.async`

使用异步 Global-to-Shared 拷贝降低数据搬运等待。

### 19.8 Double Buffering

当前 Tile 计算时预取下一个 Tile，形成加载与计算流水线。

### 19.9 Compile-time Configuration

模板参数均为编译期常量，便于循环展开和地址计算优化。

### 19.10 Fast/Edge Path Separation

规则区域使用高性能路径，非规则区域使用带保护的通用路径。

### 19.11 Shape Dispatch

根据输入规模选择不同 Tile 配置，避免一套配置处理所有矩阵。

---

## 20. 这个文件的工程意义

这个文件的价值不仅在于“实现矩阵乘法”，而在于展示了 CUDA 算子优化中的典型方法论。

### 20.1 从数学公式到并行映射

它把：

\[
C[m,n] = \sum_k A[m,k]B[k,n]
\]

映射成：

- Grid 覆盖整个 C；
- Block 负责一个 C Tile；
- Warp 负责一个 Warp Tile；
- Thread 负责多个输出元素；
- K 维通过循环归约。

### 20.2 从显存访问到存储层次设计

它明确规划了：

```text
Global Memory → Shared Memory → Registers
```

这正是 CUDA 性能优化的核心。

### 20.3 从“能运行”到“能处理真实输入”

它不仅处理规则矩阵，还处理：

- M 边界；
- N 边界；
- K remainder；
- 非 4 对齐访问；
- 小矩阵；
- 中等不规则矩阵；
- 大矩阵。

### 20.4 从单一 Kernel 到可调优框架

Tile 参数被抽离到配置结构体中，可以继续：

- 增加更多配置；
- 基于矩阵形状做更细粒度 Dispatch；
- 根据不同 GPU 架构调整参数；
- 通过 Benchmark 寻找更优配置。

因此，这个文件已经具备“小型 GEMM Kernel 框架”的雏形，而不是一次性的演示代码。

---

## 21. 当前实现的局限与需要注意的地方

### 21.1 与 GPU 架构强相关

文件直接使用 `cp.async` PTX，说明实现依赖支持该指令的较新 GPU 架构。

项目 CMake 当前设置为特定 CUDA Architecture，因此它不是完全通用的跨架构实现。

### 21.2 寄存器压力较高

线程一次累加多个输出元素，有利于数据复用，但会消耗大量寄存器。

如果寄存器使用过高，可能导致：

- Occupancy 下降；
- 寄存器 Spill 到 Local Memory；
- 性能反而下降。

因此需要结合 Nsight Compute 检查：

- Registers Per Thread；
- Achieved Occupancy；
- Local Load/Store；
- Eligible Warps；
- FMA 利用率。

### 21.3 Edge Path 性能明显弱于 Fast Path

不规则尺寸需要：

- 清零 Shared Memory；
- 标量加载；
- 多次边界判断；
- 标量写回。

所以项目测试中非规则矩阵的性能比例明显低于规则尺寸，是符合代码结构预期的。

### 21.4 Dispatch 规则仍较粗

当前只按几个阈值和对齐条件选择三套配置，没有考虑：

- M、N、K 极端长宽比；
- GPU 型号；
- SM 数量；
- L2 Cache 容量；
- 不同 K 大小下的计算/访存比例；
- 实际测得的最优配置表。

相比 cuBLAS，这仍然是较简化的启发式 Dispatch。

### 21.5 Kernel 启动接口没有传入 Stream

当前 Kernel 使用默认 Stream 启动。

在更完整的推理框架或算子库中，通常应支持传入 `cudaStream_t`，以便和其他算子进行异步编排。

### 21.6 极小 K 的边界假设需要额外验证

快速路径会在进入主循环前预加载一个完整 `BK` Tile，而 K 尾部又单独处理。

对于项目当前主要测试尺寸通常没有问题，但从完全通用的库实现角度，仍应专门测试：

```text
K < BK
K = 1、2、3、4、7
```

以确认预加载逻辑不会访问超出有效范围的数据。

这一点属于工程健壮性测试，而不是该文件的主要优化思想。

---

## 22. 阅读这个文件时最应该掌握的主线

第一次阅读时，不建议一开始钻入内联 PTX 和复杂下标。应按以下顺序理解。

### 第一阶段：明确计算目标

先明确：

```text
每个 Block 计算 C 的 BM × BN 区域
每次沿 K 处理 BK
```

### 第二阶段：理解线程层级

掌握：

```text
Block → Warp Tile → Thread Tile
```

重点看：

- `blockIdx.x/y`；
- `warp_idx`；
- `warp_row/warp_col`；
- `trow/tcol`。

### 第三阶段：理解数据流

只追踪三类数据：

```text
A/B：Global → Shared → reg_m/reg_n
C：res → Global
```

### 第四阶段：理解 K 循环

明确每个 K Tile 都会：

```text
加载 A/B Tile
→ 读取到寄存器
→ 做乘加
→ 累积到 res
```

### 第五阶段：理解 Fast Path

再看：

- `cp.async`；
- 双缓冲；
- `cur/nxt`；
- `commit/wait`；
- `float4` 写回。

### 第六阶段：理解 Edge Path

最后理解 M、N、K 不整除时如何保证正确性。

---

## 23. 建议你带着这些问题阅读源码

1. 一个 Thread Block 最终负责 C 中多大的区域？
2. 为什么 A、B 只沿 K 方向分批加载？
3. 一个 Warp 在 C Block 中负责哪一块？
4. 一个线程最终需要计算多少个 C 元素？
5. 为什么 A 写入 Shared Memory 后布局发生变化？
6. `reg_m`、`reg_n` 和 `res` 分别保存什么？
7. 为什么 `res` 要一直留在寄存器中，最后才写回？
8. 当前 Tile 计算时，下一个 Tile 被加载到哪里？
9. `cp_async_wait_group()` 和 `__syncthreads()` 分别保证什么？
10. 什么情况下进入 Fast Path？
11. 为什么 Edge Path 要先把 Shared Memory 清零？
12. M/N 边缘和 K remainder 有什么区别？
13. 为什么 Fast Path 可以使用 `float4`，Edge Path 通常不能？
14. 三套配置的 Block Tile、Warp Tile、线程数分别是多少？
15. 为什么大 Tile 不一定适合小矩阵？

能够回答这些问题，就已经掌握了该文件的大部分核心思想。

---

## 24. 一句话总结

`gemm_warp_tile.cu` 是项目的核心 FP32 GEMM Kernel：它通过 Block/Warp/Thread 多级分块，把 A、B 从全局内存分批搬入双缓冲共享内存，再由每个线程在寄存器中计算和累加多个输出元素，并通过 Fast Path、Edge Path 和多配置 Dispatch 同时兼顾规则矩阵性能与任意尺寸正确性。

它在项目中的意义是：

> 把 GEMM 的数学公式真正转化为一个面向 GPU 存储层次、线程层次和流水线执行方式设计的高性能 CUDA 算子。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-gemm算子|模块-gemm算子]]

%% 项目关联导航：结束 %%
