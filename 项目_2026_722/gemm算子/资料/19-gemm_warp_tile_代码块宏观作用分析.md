# `gemm_warp_tile.cu` 代码块宏观作用分析

> 本文不逐行解释 C++/CUDA 语法，也不要求你记住复杂索引公式。
>
> 分析目标是：**以源码中的功能代码块为单位，理解每个代码块为什么存在、接收什么数据、完成什么任务，以及它处在整个 GEMM 数据流的什么位置。**
>
> 当前文件实现的总体计算为：
>
> ```text
> C = alpha × A × B + beta × C
> ```
>
> 它是一个以 Warp Tile 为核心、使用 FP32 CUDA Core 乘加的 GEMM Kernel，主要采用：
>
> - Block Tile、Warp Tile、Thread Tile；
> - Global Memory → Shared Memory → Registers 的分层数据搬运；
> - Shared Memory 双缓冲；
> - `cp.async` 异步搬运；
> - 每线程寄存器累加；
> - Fast Path 与 Edge Path 分离；
> - `float4` 向量化写回；
> - 根据矩阵形状选择不同 Tile 配置。

---

# 1. 先建立整个文件的功能地图

这个 `.cu` 文件从上到下可以分成以下功能代码块：

```text
代码块 1：头文件、常量和基础宏
    ↓
代码块 2：cp.async 异步搬运辅助函数
    ↓
代码块 3：正常 Tile 的寄存器计算函数 compute()
    ↓
代码块 4：K 余数计算函数 compute_k()
    ↓
代码块 5：边界 Tile 计算函数 compute_edge()
    ↓
代码块 6：Global → Shared 异步 Tile 搬运函数 load_tile_async()
    ↓
代码块 7：主 Kernel 的配置解析与线程映射
    ↓
代码块 8：Shared Memory 和线程寄存器资源准备
    ↓
代码块 9：Fast Path 主流水线
    ├── 首块预取
    ├── 双缓冲 K 循环
    ├── K 余数处理
    └── float4 写回
    ↓
代码块 10：Edge / Unaligned Path
    ├── Shared 清零
    ├── 带边界的数据加载
    ├── 边界计算
    └── 带边界写回
    ↓
代码块 11：模板 Kernel Launch 封装
    ↓
代码块 12：按矩阵规模选择配置
```

从数据流角度看，这些代码块共同完成：

```text
完整 A/B/C 位于 Global Memory
    ↓
当前 Block 定位自己的 C Tile
    ↓
沿 K 方向分批选择 A/B Tile
    ↓
A/B Tile 从 Global Memory 搬到 Shared Memory
    ↓
线程从 Shared Memory 取得自己需要的数据
    ↓
数据进入 reg_m / reg_n
    ↓
乘加结果累积在 res
    ↓
K 维处理完成
    ↓
res 写回 Global Memory 中的 C
```

---

# 2. 代码块一：头文件、基础常量与宏

对应源码开头：

```cpp
#include <cstdint>
#include <cuda_runtime.h>
#include "gemm/kernels/gemm_warp_tile.h"
#include "gemm/kernels/gemm_warp_tile_config.h"

#define CEIL_DIV(x, y) ...
#define WARP_SIZE 32
```

## 2.1 这个代码块的作用

这个代码块负责为整份文件准备三类基础信息。

### 第一类：CUDA 运行环境

```cpp
#include <cuda_runtime.h>
```

让本文件能够使用：

- Kernel；
- Block、Thread、Warp；
- Shared Memory；
- CUDA 数据类型；
- Kernel Launch；
- CUDA Runtime 相关能力。

### 第二类：项目对外接口

```cpp
#include "gemm/kernels/gemm_warp_tile.h"
```

它把本文件实现的：

```text
launch_gemm_warp_tile()
```

和项目其他文件连接起来。

从工程角度看：

```text
launcher.cu
    ↓
调用 launch_gemm_warp_tile()
    ↓
进入当前文件
```

### 第三类：Tile 配置

```cpp
#include "gemm/kernels/gemm_warp_tile_config.h"
```

这里提供：

- `WarpTileDefaultConfig`
- `WarpTileSmallConfig`
- `WarpTileIrregularConfig`

以及：

```text
BM、BN、BK、WM、WN、TM、TN、WNITER、NUM_THREADS
```

这些参数决定：

- 一个 Block 负责多大的 C Tile；
- 一个 Warp 负责多大的 Warp Tile；
- 每个线程负责多少输出；
- 一个 Block 使用多少线程；
- Shared Memory 需要多大。

### 第四类：公共基础参数

```cpp
WARP_SIZE = 32
```

整个 Warp Tile 映射都建立在：

```text
1 Warp = 32 Threads
```

之上。

`CEIL_DIV` 用于计算 Grid，保证边界不足完整 Tile 时也会启动对应 Block。

---

# 3. 代码块二：`cp.async` 异步搬运辅助函数

对应函数：

```cpp
smem_u32addr()
cp_async16()
cp_async_commit()
cp_async_wait_group()
```

这几个函数应作为**一个整体代码块**理解，而不是分开钻研 PTX 语法。

---

## 3.1 这个代码块解决什么问题

普通的数据搬运思路是：

```text
Global Memory
    ↓ 普通加载
线程寄存器
    ↓ 写入
Shared Memory
```

当前项目希望采用更高效的路径：

```text
Global Memory
    ↓ cp.async
Shared Memory
```

这样做的主要目的不是改变数学结果，而是优化数据搬运阶段：

- 减少显式寄存器中转；
- 异步提交数据复制；
- 为“计算当前 Tile，同时预取下一 Tile”创造条件；
- 构建 Shared Memory 双缓冲流水线。

---

## 3.2 `smem_u32addr()` 的宏观作用

```cpp
smem_u32addr()
```

负责把普通 Shared Memory 指针转换成底层异步复制指令能够识别的 Shared Memory 地址形式。

宏观上可以理解为：

> 在调用 `cp.async` 前，先把“目标 Shared Memory 位置”转换成硬件指令所需要的地址表示。

它不负责搬数据，只负责准备目标地址。

---

## 3.3 `cp_async16()` 的宏观作用

```cpp
cp_async16(...)
```

每次提交一次：

```text
16 Bytes
=
4 个 float
```

的：

```text
Global Memory → Shared Memory
```

异步复制。

它还带有一个 `guard` 条件，用于控制这次复制是否真正执行。

在当前 Fast Path 中，数据已经满足完整 Tile 和对齐条件，所以大部分复制可以直接执行。

---

## 3.4 `cp_async_commit()` 的宏观作用

一个 Tile 需要很多线程执行很多次 16B 搬运。

```cpp
cp_async_commit()
```

负责把前面提交的一批异步复制归为一个复制组。

可以理解为：

```text
当前 A Tile 的复制任务
+
当前 B Tile 的复制任务
    ↓
组成一批需要统一管理的异步搬运
```

---

## 3.5 `cp_async_wait_group()` 的宏观作用

```cpp
cp_async_wait_group<0>()
```

用于等待此前提交的异步复制完成。

它解决的问题是：

> 计算阶段不能在 Shared Memory 数据尚未完成搬运时就开始读取。

所以典型执行关系是：

```text
提交 Global → Shared 搬运
    ↓
等待这批搬运完成
    ↓
Block 内线程同步
    ↓
开始消费 Shared Memory
```

---

## 3.6 这组辅助函数在文件中的位置

它们只服务于 Fast Path 的数据搬运流水线：

```text
load_tile_async()
    ↓
cp_async16()
    ↓
cp_async_commit()
    ↓
cp_async_wait_group()
```

因此可以把这整个代码块记成：

> **为后面的 Global → Shared 异步搬运和双缓冲提供底层能力。**

---

# 4. 代码块三：`compute()`——完整 BK Tile 的主计算块

对应函数：

```cpp
compute(...)
```

这是整个文件最核心的**数学计算代码块**。

它不负责：

- 选择矩阵配置；
- 从 Global Memory 搬数据；
- 处理整个 Grid；
- 最终写回 C。

它只负责：

> 已知当前 A/B Tile 已经位于 Shared Memory，把当前线程所需的数据读入寄存器，并对 `res[]` 做一次完整 BK Tile 的乘加累积。

---

## 4.1 输入数据处于什么状态

进入 `compute()` 前：

```text
AS 中已经有当前 A Tile
BS 中已经有当前 B Tile
```

逻辑布局：

```text
AS：[BK][BM]
BS：[BK][BN]
```

当前线程还知道：

- 自己属于哪个 Warp；
- 自己在 Warp 内的行列位置；
- 自己负责哪一块 Thread Tile。

---

## 4.2 `compute()` 的三阶段职责

整个函数可以宏观分成三步。

### 第一步：Shared A → `reg_m`

```text
AS 中当前 d 对应的 A 数据
    ↓
当前线程选择自己负责的 M 方向元素
    ↓
存入 reg_m[]
```

`reg_m` 可以理解为：

> 当前线程在当前 K 位置，需要参与乘法的 A 小向量。

### 第二步：Shared B → `reg_n`

```text
BS 中当前 d 对应的 B 数据
    ↓
当前线程选择自己负责的 N 方向元素
    ↓
存入 reg_n[]
```

`reg_n` 可以理解为：

> 当前线程在当前 K 位置，需要参与乘法的 B 小向量。

### 第三步：寄存器外积 → `res`

```text
reg_m
    ×
reg_n
    ↓
对线程负责的多个 C 元素做乘加
    ↓
累积到 res[]
```

---

## 4.3 为什么是外积计算

以 Default 配置为例：

```text
reg_m：8 个 A 元素
reg_n：16 个 B 元素
```

于是当前线程执行：

```text
8 × 16 = 128
```

个乘加，正好更新它负责的 `8 × 16` Thread Tile。

所以 `compute()` 的本质不是“计算一个输出元素”，而是：

> 使用一小组 A 寄存器和一小组 B 寄存器，一次更新整个线程私有输出微块。

---

## 4.4 为什么 `res[]` 不在函数结束后写回

`compute()` 只处理当前一个 `BK` Tile。

矩阵乘法还需要继续处理后续 K Tile，因此：

```text
第 0 个 BK Tile → 累加 res
第 1 个 BK Tile → 继续累加 res
第 2 个 BK Tile → 继续累加 res
...
```

`res[]` 会一直保留在线程寄存器中，直到整个 K 维处理完成。

这个设计的意义是：

```text
避免每个 BK Tile 后都读写 Global C
```

---

# 5. 代码块四：`compute_k()`——K 维余数计算块

对应函数：

```cpp
compute_k(...)
```

它与 `compute()` 的计算思路完全相同。

区别是：

```text
compute()：默认处理完整 BK 个 K 元素
compute_k()：只处理实际剩余的 k_lim 个元素
```

---

## 5.1 为什么需要它

项目的：

```text
BK = 8
```

但 K 不一定能被 8 整除。

例如：

```text
K = 100
```

前面可以处理：

```text
12 个完整 BK Tile = 96
```

还剩：

```text
4 个 K 元素
```

这 4 个元素不能丢弃，所以需要余数计算块。

---

## 5.2 它在整体流程中的位置

```text
完整 BK Tile 循环全部结束
    ↓
检查 K % BK
    ↓
若存在余数
    ↓
准备剩余 A/B 数据
    ↓
调用 compute_k()
    ↓
余数贡献继续累加到同一个 res[]
```

所以它不是另一种 GEMM，而是主计算的收尾阶段。

---

# 6. 代码块五：`compute_edge()`——边界 Tile 的安全计算块

对应函数：

```cpp
compute_edge(...)
```

它仍然执行：

```text
Shared → reg_m/reg_n → res
```

但比 `compute()` 多做一件事：

> 对 M/N 边界进行保护，无效行列按 0 参与计算。

---

## 6.1 为什么需要边界版本

假设：

```text
BM = 64
M 最后只剩 40 行
```

最后一个 Block 的 Shared Memory 和线程布局仍然按完整 64 行组织，但只有前 40 行有效。

同样：

```text
BN = 128
N 最后只剩 27 列
```

则部分线程负责的输出列不存在。

---

## 6.2 它如何保证正确性

宏观逻辑是：

```text
若当前 A 行有效：
    从 AS 读取
否则：
    reg_m 中写 0

若当前 B 列有效：
    从 BS 读取
否则：
    reg_n 中写 0
```

然后仍然执行相同的外积累加。

由于无效数据为 0：

```text
0 × 有效值 = 0
有效值 × 0 = 0
```

不会污染合法结果。

---

## 6.3 它与 Fast Path 的关系

```text
完整且对齐的内部 Block
    → compute()

边界或非对齐 Block
    → compute_edge()
```

数学目标一致，主要差别是：

- Fast Path 追求效率；
- Edge Path 优先保证任意尺寸正确。

---

# 7. 代码块六：`load_tile_async()`——Global → Shared 的协作搬运块

对应函数：

```cpp
load_tile_async(...)
```

这是 Fast Path 的主要数据入口。

它的职责是：

> 让整个 Thread Block 的线程协作，把当前 A Tile 和 B Tile 从 Global Memory 搬入指定的 Shared Memory Buffer。

---

## 7.1 输入和输出是什么

输入：

```text
A_tile：Global Memory 中当前 A Tile 起点
B_tile：Global Memory 中当前 B Tile 起点
```

输出：

```text
AS：当前 Shared A Buffer
BS：当前 Shared B Buffer
```

整体路线：

```text
Global A Tile → AS
Global B Tile → BS
```

---

## 7.2 为什么由整个 Block 协作搬运

Default 配置下，一个 Block 当前需要：

```text
A Tile：128 × 8
B Tile：8 × 256
```

数据量较大，不应由一个线程加载。

所以代码预先为每个线程计算：

```text
它负责 A Tile 的哪一段
它负责 B Tile 的哪一段
```

每个线程搬若干个连续 4-float 片段。

最后：

```text
所有线程搬运结果拼成完整 AS/BS
```

---

## 7.3 为什么 A 与 B 使用不同映射

A Tile 和 B Tile 的形状不同：

```text
A：BM × BK
B：BK × BN
```

而计算阶段希望 Shared Memory 中是：

```text
AS：[BK][BM]
BS：[BK][BN]
```

因此：

- A 搬入时伴随布局重组；
- B 搬入时基本保持 `[BK][BN]` 布局。

宏观目的都是：

> 让后面的 `compute()` 能够在固定 d 时，高效取得 M 方向的 A 数据和 N 方向的 B 数据。

---

## 7.4 为什么函数结尾提交 group

当当前线程已经提交完自己负责的 A/B 搬运后：

```cpp
cp_async_commit()
```

把这批复制任务提交给异步流水线。

随后主 Kernel 决定：

- 何时等待；
- 何时同步；
- 何时开始计算。

---

# 8. 代码块七：主 Kernel 的配置展开

对应：

```cpp
template <typename Cfg>
__global__ void gemm_warp_tile_kernel(...)
```

进入 Kernel 后，首先出现大量：

```text
NUM_THREADS
BM / BN / BK
WM / WN
TM / TN
WNITER
WSUBM / WSUBN
WMITER
```

这一块不应理解为“真正开始计算”，它的作用是：

> 把配置文件中的抽象 Tile 参数，转化为当前 Kernel 可以直接使用的固定布局规则。

---

## 8.1 这个代码块决定了什么

它确定：

- 一个 Block 负责多大 C Tile；
- 一个 Warp 负责多大 Warp Tile；
- Warp Tile 需要分几轮；
- 每个线程负责多少行和列；
- Shared Memory Tile 多大；
- 一个 Block 使用多少线程；
- 每个线程需要多少寄存器累加器。

---

## 8.2 为什么使用模板配置

不同规模矩阵适合不同 Tile：

```text
大矩阵：
需要较大的 Block Tile，提高数据复用

小矩阵：
需要较小 Block，避免并行度不足和资源浪费

不规则矩阵：
需要更适合边界比例的 Tile
```

模板让这些尺寸在编译期固定，有利于：

- 循环展开；
- 固定大小寄存器数组；
- 固定 Shared Memory；
- 编译器优化；
- 避免运行时重复判断。

---

# 9. 代码块八：Block、Warp、Thread 与矩阵区域映射

主 Kernel 接下来计算：

```text
c_row / c_col
warp_idx / warp_row / warp_col
trow / tcol
inner_row / inner_col
```

这一整块的宏观职责是：

> 同时建立“计算责任映射”和“数据搬运责任映射”。

这两种映射需要区分。

---

## 9.1 计算责任映射

相关变量：

```cpp
c_row
c_col
warp_idx
warp_row
warp_col
trow
tcol
```

它们回答：

```text
当前 Block 负责 C 的哪个 Block Tile？
当前 Warp 负责 Block Tile 中的哪个 Warp Tile？
当前线程负责 Warp Tile 中的哪些输出元素？
```

这是**输出矩阵 C 的划分路线**。

---

## 9.2 数据搬运责任映射

相关变量：

```cpp
inner_row_a
inner_col_a
inner_row_b
inner_col_b
```

它们回答：

```text
当前线程负责把 A Tile 的哪些元素搬到 AS？
当前线程负责把 B Tile 的哪些元素搬到 BS？
```

这是**输入矩阵 A/B 的供数路线**。

---

## 9.3 为什么两套映射不同

线程的计算任务和搬运任务不是一一对应的。

一个线程最终可能负责 C 中一个 `8×16` Thread Tile，但它在搬运阶段不一定只搬这块输出所需要的数据。

原因是：

> Shared Memory Tile 是整个 Block 共享的，最重要的是让所有线程共同形成连续、均匀、高效的搬运，而不是让每个线程只搬“自己的数据”。

---

# 10. 代码块九：边界检测与 Fast Path 选择

相关变量：

```cpp
m_edge
n_edge
fast
```

这个代码块先判断：

```text
当前 Block 是否是完整 BM×BN Tile
K、N 是否满足 4-float 对齐条件
```

然后决定执行：

```text
Fast Path
或
Edge / Unaligned Path
```

---

## 10.1 `m_edge` 和 `n_edge`

它们表示：

```text
当前 Block 在 M 方向实际有多少有效行
当前 Block 在 N 方向实际有多少有效列
```

内部 Block 通常：

```text
m_edge = BM
n_edge = BN
```

边界 Block 可能小于完整 Tile。

---

## 10.2 为什么要分两条路径

如果所有 Block 都使用大量边界判断：

- 主计算循环分支更多；
- 无法放心使用统一向量化；
- `cp.async` 地址处理更复杂；
- 内部大多数完整 Tile 的性能被拖慢。

所以项目选择：

```text
大部分内部 Tile：
走高度优化的 Fast Path

少量边界 Tile：
走安全通用的 Edge Path
```

---

# 11. 代码块十：Shared Memory 与寄存器资源准备

相关变量：

```cpp
AS[2]
BS[2]
reg_m[]
reg_n[]
res[]
C_out
```

这块代码是在正式计算前，为数据流准备不同层级的存储。

---

## 11.1 `AS[2]`、`BS[2]`

```text
Block 共享的双缓冲 Shared Memory
```

作用：

```text
Buffer cur：当前正在被 compute() 消费
Buffer nxt：正在接收下一 K Tile
```

---

## 11.2 `reg_m[]`

```text
当前线程本次计算需要的 A 小向量
```

生命周期很短：

```text
每个 d 都会重新从 AS 读取
```

---

## 11.3 `reg_n[]`

```text
当前线程本次计算需要的 B 小向量
```

同样会随着 d 更新。

---

## 11.4 `res[]`

```text
当前线程负责的全部 C 输出累加器
```

生命周期贯穿整个 K 循环。

它是当前 Kernel 最重要的线程私有状态：

```text
开始时清零
每个 K Tile 继续累加
所有 K 完成后写回 C
```

---

## 11.5 `C_out`

它不是新分配的数据，而是指向：

```text
当前 Warp 对应的 Global C 区域起点
```

后续线程根据自己的 Thread Tile 偏移，把 `res[]` 写到正确位置。

---

# 12. 代码块十一：Fast Path 的首块预取

对应逻辑：

```cpp
load_tile_async(... AS[0], BS[0] ...);
cp_async_wait_group<0>();
__syncthreads();
```

这块代码的作用是：

> 在进入主流水线前，先把第一个 K Tile 准备好。

---

## 12.1 为什么需要单独预取第一块

双缓冲循环希望形成：

```text
计算当前 Tile
同时加载下一 Tile
```

但是一开始还没有“当前 Tile”，所以必须先：

```text
Global → Shared Buffer 0
```

等 Buffer 0 准备好后，主循环才能开始。

---

## 12.2 它相当于流水线的启动阶段

完整流水线分三部分：

```text
启动：
预取第一个 Tile

稳定阶段：
计算当前 Tile + 预取下一 Tile

收尾：
处理最后一个 Tile 和 K 余数
```

首块预取就是 Pipeline Prologue。

---

# 13. 代码块十二：Fast Path 的双缓冲主循环

这是文件里最重要的执行代码块。

宏观流程：

```text
确定 cur / nxt
    ↓
若存在下一 Tile：
    异步加载到 nxt
    ↓
使用 cur 调用 compute()
    ↓
等待 nxt 完成
    ↓
Block 同步
    ↓
下一轮交换 cur / nxt
```

---

## 13.1 `cur` 与 `nxt`

```text
cur：当前计算缓冲区
nxt：下一批数据缓冲区
```

两者在 0 和 1 之间来回切换：

```text
第 0 轮：计算 0，加载 1
第 1 轮：计算 1，加载 0
第 2 轮：计算 0，加载 1
```

---

## 13.2 这一代码块试图重叠什么

```text
Shared → Registers → FMA
```

与：

```text
Global → Shared
```

尽量同时进行。

如果没有双缓冲：

```text
加载
等待
计算
加载
等待
计算
```

有双缓冲后：

```text
计算当前
同时加载下一批
```

理想情况下，部分内存等待时间可以被计算覆盖。

---

## 13.3 `compute()` 在这里扮演什么角色

主循环只管理：

- Tile 顺序；
- Buffer 切换；
- 异步搬运；
- 同步。

真正的数学计算交给：

```cpp
compute(...)
```

因此两者的分工是：

```text
主循环：
管理流水线

compute：
消费当前 Shared Tile 并更新 res
```

---

# 14. 代码块十三：Fast Path 的 K 余数处理

对应：

```text
清零 Shared
加载剩余数据
同步
compute_k()
```

这个代码块属于主流水线的收尾阶段。

---

## 14.1 为什么不能直接调用完整 `compute()`

`compute()` 默认处理：

```text
BK 个 K 元素
```

但余数可能只有：

```text
1~BK-1 个
```

所以必须告诉计算函数实际有效长度：

```text
k_rem
```

---

## 14.2 为什么先清零 Shared

Shared Memory 缓冲区的物理大小仍是完整 BK。

若只加载有效余数，没有覆盖的槽位可能保留旧数据。

先清零能保证：

```text
无效位置为 0
```

即使误被访问，也不会对结果产生额外贡献。

---

# 15. 代码块十四：Fast Path 的 `float4` 写回

K 全部处理完成后：

```text
res[] 中保存当前线程的最终输出
```

Fast Path 使用：

```text
一次 4 个 float
```

进行 C 的读取、融合和写回。

---

## 15.1 这个代码块完成什么数学操作

不是简单：

```text
C = res
```

而是：

```text
C = alpha × res + beta × C_old
```

所以需要：

1. 读取原来的 C；
2. 与 `res` 融合；
3. 写回新的 C。

---

## 15.2 为什么用 `float4`

线程的 N 方向基础宽度：

```text
TN = 4
```

正好可以一次处理 4 个连续输出。

这样能：

- 减少读写指令数量；
- 利用连续地址；
- 与前面的 4-float 对齐设计一致。

---

## 15.3 这个代码块在数据流中的位置

```text
res Registers
    ↓
alpha × res + beta × old C
    ↓
float4
    ↓
Global Memory C
```

至此 Fast Path 完成。

---

# 16. 代码块十五：Edge Path 的 K 循环

Edge Path 的总体结构：

```text
沿 K 每次前进 BK
    ↓
计算当前有效 k_rem
    ↓
Shared 清零
    ↓
带边界加载 A/B
    ↓
同步
    ↓
compute_edge()
    ↓
同步
```

---

## 16.1 它为什么不用 Fast Path 流水线

边界 Tile 中：

- 有效行数不足 BM；
- 有效列数不足 BN；
- K/N 可能不满足向量化条件；
- 部分 4-float 数据可能跨越边界。

继续强行使用统一 `cp.async 16B` 会增加复杂的保护逻辑。

所以当前实现选择：

> 边界区域使用更容易保证正确性的标量加载与显式判断。

---

## 16.2 Shared 清零代码块

每个 K Tile 开始前：

```text
所有线程协作把 AS/BS 清零
```

作用：

- 无效位置自动为 0；
- 避免上一轮 Shared 数据残留；
- 简化后续计算逻辑。

---

## 16.3 带边界的 A/B 加载代码块

加载阶段检查：

```text
当前 K 是否有效
当前 M 行是否有效
当前 N 列是否有效
```

只把存在的数据放入 Shared Memory。

所以 Edge Path 的 Global → Shared 流程是：

```text
合法 Global 元素 → Shared
非法位置 → 保持 0
```

---

## 16.4 `compute_edge()` 代码块

它再次从计算层面保护边界。

即使 Shared 已经清零，线程仍根据：

```text
m_edge
n_edge
```

确认自己负责的行列是否合法。

这属于“加载阶段保护 + 计算阶段保护”的保守设计。

---

# 17. 代码块十六：Edge Path 的逐元素写回

Edge Path 最后不能统一用 `float4`，因为最后一行或一列可能不足 4 个元素。

所以写回代码块会对每一个结果判断：

```text
全局行是否小于 M
全局列是否小于 N
```

只有合法输出才写入 C。

---

## 17.1 与 Fast Path 写回的对比

### Fast Path

```text
完整 Tile
连续 4 个元素
float4 写回
```

### Edge Path

```text
不完整 Tile
逐元素检查
标量写回
```

宏观原则：

```text
内部区域追求吞吐
边界区域追求正确
```

---

# 18. 代码块十七：`launch_gemm_warp_tile_cfg()`——配置到 Kernel Launch 的桥梁

对应函数：

```cpp
launch_gemm_warp_tile_cfg<Cfg>()
```

它的职责很单纯：

> 把某一套 Cfg 转换成实际的 Grid、Block，并启动对应模板 Kernel。

---

## 18.1 Block 配置

```text
blockDim = Cfg::NUM_THREADS
```

例如：

```text
Default：256 Threads
Small：128 Threads
Irregular：128 Threads
```

---

## 18.2 Grid 配置

```text
grid.x = ceil(N / BN)
grid.y = ceil(M / BM)
```

保证所有 C Tile 都有一个 Block 负责，包括最后的不完整 Tile。

---

## 18.3 这层封装的意义

上层只需要说：

```text
使用某个 Cfg
```

而不需要重复写：

- Grid 计算；
- Block 计算；
- Kernel 模板实例化；
- Kernel Launch。

---

# 19. 代码块十八：`launch_gemm_warp_tile()`——按问题规模选择配置

文件最后的函数：

```cpp
launch_gemm_warp_tile(...)
```

是当前文件对外的入口。

它不做 GPU 数学计算，主要负责：

> 根据 M/N/K 的规模和规则程度，在 Host 端选择一套 Tile 配置。

---

## 19.1 Irregular 配置选择

当形状存在：

```text
M 不能整除 64
或 N 不能整除 128
或 K 不能整除 8
```

并且规模不超过中等范围时，选择：

```text
WarpTileIrregularConfig
```

目的：

- 减小 Tile；
- 降低边界浪费；
- 更适合不规则中小形状。

---

## 19.2 Small 配置选择

当：

```text
M、N、K 都不大于 1024
```

并且没有先匹配 Irregular 分支时，选择：

```text
WarpTileSmallConfig
```

目的：

- 用更小 Block Tile；
- 让小矩阵产生更多可调度 Block；
- 避免大 Tile 对小问题造成资源浪费。

---

## 19.3 Default 配置选择

其余情况使用：

```text
WarpTileDefaultConfig
```

它面向较大、较规则的 GEMM，使用更大的 Block Tile 和更多寄存器复用。

---

# 20. 把所有代码块串成一次 Kernel 执行

现在可以从宏观角度，把整份文件串成完整时间线。

```text
1. launch_gemm_warp_tile()
   根据 M/N/K 选择 Cfg

2. launch_gemm_warp_tile_cfg<Cfg>()
   计算 Grid/Block，启动 Kernel

3. gemm_warp_tile_kernel<Cfg>()
   展开 BM/BN/BK/WM/WN/TM/TN

4. 建立矩阵映射
   Block → C Block Tile
   Warp → Warp Tile
   Thread → Thread Tile

5. 建立搬运映射
   每个线程负责 A/B Tile 的部分数据

6. 准备存储
   AS/BS 双缓冲
   reg_m/reg_n 输入寄存器
   res 输出累加器

7. 判断当前 Block
   完整且对齐 → Fast Path
   边界或非对齐 → Edge Path

8A. Fast Path
   首块 cp.async 预取
   双缓冲 K 循环
   compute() 更新 res
   compute_k() 处理余数
   float4 写回 C

8B. Edge Path
   Shared 清零
   带边界加载
   compute_edge() 更新 res
   逐元素保护写回 C

9. 当前 Block 完成自己的 C Tile
```

---

# 21. 按“代码块作用”重新分类

除了源码顺序，还可以按职责把代码块分成五个系统。

---

## 21.1 配置与调度系统

包括：

```text
launch_gemm_warp_tile()
launch_gemm_warp_tile_cfg()
Cfg 参数展开
Grid / Block 设置
```

作用：

```text
决定使用什么 Kernel 形态，以及启动多少 Block/Thread
```

---

## 21.2 输出责任划分系统

包括：

```text
c_row / c_col
warp_idx / warp_row / warp_col
trow / tcol
C_out
```

作用：

```text
决定每个 Block、Warp、Thread 分别负责 C 的哪一部分
```

---

## 21.3 数据搬运系统

包括：

```text
load_tile_async()
cp_async16()
cp_async_commit()
cp_async_wait_group()
AS / BS
inner_row / inner_col
```

作用：

```text
把 A/B 从 Global Memory 高效搬到 Shared Memory
```

---

## 21.4 寄存器计算系统

包括：

```text
compute()
compute_k()
compute_edge()
reg_m
reg_n
res
```

作用：

```text
从 Shared Memory 取数据，在寄存器中完成外积乘加
```

---

## 21.5 写回与正确性系统

包括：

```text
fast 判断
m_edge / n_edge
K remainder
float4 write-back
edge scalar write-back
```

作用：

```text
让内部完整区域高效执行，同时保证边界和任意尺寸正确
```

---

# 22. 这个文件的性能优化主线

宏观上，这个 Kernel 的复杂度主要来自四条优化主线。

---

## 22.1 数据复用

```text
Global A/B
    ↓ 只搬一次
Shared A/B Tile
    ↓ 被 Block 内大量线程重复使用
```

减少重复 Global Memory 访问。

---

## 22.2 寄存器分块

```text
一个线程负责多个 C 元素
    ↓
res[] 中长期累加
```

让每次从 Shared 取得的 A/B 数据产生更多乘加。

---

## 22.3 搬运与计算重叠

```text
AS/BS 双缓冲
+
cp.async
```

让下一 Tile 的加载尽量与当前 Tile 的计算重叠。

---

## 22.4 快速路径与通用路径分离

```text
Fast：
完整、对齐、异步、向量化

Edge：
边界判断、清零、标量保护
```

让主区域不承担边界处理成本。

---

# 23. 阅读这份源码时不需要纠结的语法细节

为了宏观理解，当前阶段不需要深入：

- 内联 PTX 每条指令的语法；
- 模板参数尖括号的每一层展开；
- `uint32_t` 地址转换细节；
- `reinterpret_cast` 的语法规则；
- 每个 `#pragma unroll` 的编译器行为；
- 所有下标公式的逐项推导。

当前真正需要掌握的是：

```text
这段代码属于哪个功能代码块？
它的数据从哪里来？
它把数据送到哪里？
它解决性能还是正确性问题？
它与前后哪个代码块连接？
```

---

# 24. 最终脑内模型

重新打开 `gemm_warp_tile.cu` 时，可以把它看成一条流水线，而不是一堆复杂循环：

```text
配置选择
    ↓
线程与矩阵区域映射
    ↓
Global A/B Tile 定位
    ↓
异步搬入 Shared Memory
    ↓
Shared 数据进入 reg_m/reg_n
    ↓
寄存器外积累加到 res
    ↓
沿 K 方向重复
    ↓
处理余数与边界
    ↓
res 融合 alpha/beta
    ↓
写回 Global C
```

整份文件最核心的分工可以压缩成四句话：

1. `load_tile_async()` 负责**把数据送进来**；
2. `compute()` 系列负责**把数据算起来**；
3. Fast/Edge 分支负责**平衡性能与正确性**；
4. `launch_gemm_warp_tile()` 系列负责**选择配置并把 Kernel 启动起来**。

理解这四类代码块后，就已经从宏观上掌握了当前 `gemm_warp_tile.cu` 的整体结构。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-gemm算子|模块-gemm算子]]

%% 项目关联导航：结束 %%
