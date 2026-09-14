# gemm_gemm_warp_tile_config_精读

本文按照 `gemm_warp_tile_config.h` 的代码顺序拆分。每一节先给出完整代码块，再解释模板、结构体、编译期常量、类型别名和三套配置参数。

---

## 1. 防止头文件重复包含

```cpp
#pragma once
```

### 详细解释

`#pragma once` 是预处理指令，作用是保证当前头文件在一次编译过程中只被展开一次。

如果多个头文件间接包含了 `gemm_warp_tile_config.h`，没有防重复机制时，编译器可能多次看到同一个结构体定义并报重复定义错误。

它与传统写法作用相同：

```cpp
#ifndef GEMM_WARP_TILE_CONFIG_H
#define GEMM_WARP_TILE_CONFIG_H

// 头文件内容

#endif
```

当前写法更简洁。

---

## 2. 配置模板参数

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
```

### 详细解释

`template<...>` 表示接下来定义的不是一个固定结构体，而是一个可以根据参数生成不同类型的模板。

这里有 9 个编译期整数参数：

```text
BM_
BN_
BK_
WM_
WN_
WNITER_
TM_
TN_
NUM_THREADS_
```

它们与普通函数参数不同：

```cpp
void func(int x);
```

普通参数通常在程序运行时确定；模板参数必须在编译时确定。

例如：

```cpp
WarpTileConfig<128, 256, 8, 64, 64, 4, 8, 4, 256>
```

和：

```cpp
WarpTileConfig<64, 64, 8, 32, 32, 2, 4, 4, 128>
```

是两个不同的具体 C++ 类型。

参数名末尾的 `_` 用来区分模板参数与结构体内部公开的同义成员，例如：

```cpp
BM_  // 模板传入值
BM   // 对外公开成员
```

---

## 3. 配置结构体模板

```cpp
struct WarpTileConfig {
```

### 详细解释

`struct` 用于定义结构体。

这里定义的是结构体模板：

```cpp
WarpTileConfig<...>
```

它本身不是某一套具体参数，而是一条“如何生成配置类型”的规则。

之所以使用 `struct`，是因为结构体成员默认是 `public`，外部代码可以直接写：

```cpp
Cfg::BM
Cfg::BN
Cfg::NUM_THREADS
```

当前结构体通常不需要创建对象：

```cpp
WarpTileDefaultConfig config;
```

而是直接通过类型访问静态成员：

```cpp
WarpTileDefaultConfig::BM
```

所以它实际充当的是一个“编译期参数容器”。

---

## 4. Block Tile 参数

```cpp
    static constexpr int BM = BM_;
    static constexpr int BN = BN_;
    static constexpr int BK = BK_;
```

### 详细解释

这三行都使用：

```cpp
static constexpr int
```

- `int`：成员是整数；
- `constexpr`：值在编译期确定，并且不可修改；
- `static`：成员属于类型本身，不依赖结构体对象。

因此可以直接访问：

```cpp
WarpTileDefaultConfig::BM
```

而不需要创建对象。

### `BM`

```cpp
static constexpr int BM = BM_;
```

BM 通常表示：

```text
Block Tile M
```

即一个 CUDA block 在输出矩阵 C 的 M 方向处理多少行。

### `BN`

```cpp
static constexpr int BN = BN_;
```

BN 表示一个 block 在 C 的 N 方向处理多少列。

所以一个 block 负责的输出区域为：

```text
BM × BN
```

### `BK`

```cpp
static constexpr int BK = BK_;
```

BK 表示沿矩阵乘法 K 维每次处理多少个元素。

矩阵乘法：

```text
C(M×N) = A(M×K) × B(K×N)
```

不会一次把完整 K 维放入共享内存，而是每轮加载：

```text
A tile：BM × BK
B tile：BK × BN
```

计算完后再处理下一个 K tile。

---

## 5. Warp Tile 参数

```cpp
    static constexpr int WM = WM_;
    static constexpr int WN = WN_;
    static constexpr int WNITER = WNITER_;
```

### 详细解释

### `WM`

表示一个 warp 在 C 的 M 方向负责多少行。

### `WN`

表示一个 warp 在 C 的 N 方向负责多少列。

因此每个 warp 负责：

```text
WM × WN
```

个输出元素。

### `WNITER`

表示一个 warp 的 N 方向任务分成多少轮完成。

例如：

```text
WN = 64
WNITER = 4
```

则每轮处理：

```text
64 / 4 = 16 列
```

分轮处理能够增加数据复用，但会让每个线程保存更多中间结果，增加寄存器压力。

---

## 6. Thread Tile 与线程数参数

```cpp
    static constexpr int TM = TM_;
    static constexpr int TN = TN_;
    static constexpr int NUM_THREADS = NUM_THREADS_;
```

### 详细解释

### `TM`

Thread Tile M，表示每个线程在一个子轮次中负责多少行输出。

### `TN`

Thread Tile N，表示每个线程在一个子轮次中负责多少列输出。

每线程一个子轮次负责：

```text
TM × TN
```

个结果。

如果 warp 还需要执行多个 M/N 子轮次，线程最终负责的结果数量为：

```text
WMITER × TM × WNITER × TN
```

### `NUM_THREADS`

表示一个 CUDA block 中启动多少线程。

例如：

```text
NUM_THREADS = 256
```

因为一个 warp 有 32 个线程，所以：

```text
256 / 32 = 8 个 warp
```

这个值还会用于：

```cpp
dim3 blockDim(Cfg::NUM_THREADS);
```

以及：

```cpp
__launch_bounds__(Cfg::NUM_THREADS)
```

---

## 7. 结束结构体定义

```cpp
};
```

### 详细解释

右花括号结束结构体主体，后面的分号是 C++ 类型定义所必需的。

结构体定义必须写成：

```cpp
struct Example {
    int value;
};
```

省略最后的分号会导致编译错误。

---

## 8. 默认配置

```cpp
using WarpTileDefaultConfig = WarpTileConfig<
    128, 256, 8,
    64, 64,
    4,
    8, 4,
    256
>;
```

### 详细解释

`using` 用于创建类型别名。

这里表示：

```cpp
WarpTileDefaultConfig
```

等价于：

```cpp
WarpTileConfig<
    128, 256, 8,
    64, 64,
    4,
    8, 4,
    256
>
```

根据模板参数顺序，这套配置是：

```text
BM          = 128
BN          = 256
BK          = 8
WM          = 64
WN          = 64
WNITER      = 4
TM          = 8
TN          = 4
NUM_THREADS = 256
```

### Block 层级

一个 block 负责：

```text
128 × 256 = 32768
```

个 C 元素。

### Warp 数量

```text
256 / 32 = 8 个 warp
```

### Warp Tile

每个 warp 负责：

```text
64 × 64 = 4096
```

个元素。

8 个 warp：

```text
8 × 4096 = 32768
```

正好覆盖整个 Block Tile。

### Warp 排列

M 方向 warp 数：

```text
BM / WM = 128 / 64 = 2
```

N 方向 warp 数：

```text
BN / WN = 256 / 64 = 4
```

所以 8 个 warp 排列为：

```text
2 行 × 4 列
```

### 每线程结果数

对应 Kernel 中可推导：

```text
WSUBN = WN / WNITER = 64 / 4 = 16
```

每线程处理 `TN=4` 列，因此 warp 中 N 方向有：

```text
16 / 4 = 4 个线程列
```

warp 有 32 个线程，因此 M 方向有：

```text
32 / 4 = 8 个线程行
```

每线程处理 `TM=8` 行，所以一轮覆盖：

```text
8 × 8 = 64 行
```

刚好等于 `WM=64`，即 `WMITER=1`。

每线程最终负责：

```text
1 × 8 × 4 × 4 = 128 个结果
```

256 个线程合计：

```text
256 × 128 = 32768
```

---

## 9. 小尺寸配置

```cpp
using WarpTileSmallConfig = WarpTileConfig<
    64, 64, 8,
    32, 32,
    2,
    4, 4,
    128
>;
```

### 详细解释

对应参数：

```text
BM          = 64
BN          = 64
BK          = 8
WM          = 32
WN          = 32
WNITER      = 2
TM          = 4
TN          = 4
NUM_THREADS = 128
```

一个 block 负责：

```text
64 × 64 = 4096
```

个结果。

线程数：

```text
128 / 32 = 4 个 warp
```

每个 warp 负责：

```text
32 × 32 = 1024
```

个结果。

4 个 warp：

```text
4 × 1024 = 4096
```

正好覆盖整个 Block Tile。

warp 在 M/N 两个方向都是：

```text
64 / 32 = 2
```

所以 4 个 warp 排列为：

```text
2 行 × 2 列
```

每线程结果数：

```text
WNITER = 2
WSUBN = 32 / 2 = 16
线程列数 = 16 / TN = 16 / 4 = 4
线程行数 = 32 / 4 = 8
WSUBM = 8 × TM = 8 × 4 = 32
WMITER = 32 / 32 = 1
```

因此：

```text
每线程结果数
= 1 × 4 × 2 × 4
= 32
```

128 个线程：

```text
128 × 32 = 4096
```

小配置的 tile 更小，通常更适合小矩阵，可以减少边缘浪费和过大的寄存器、共享内存占用。

---

## 10. 非规则配置注释

```cpp
// Dedicated configuration for irregular tiles on small/medium shapes.
```

### 详细解释

`//` 是单行注释，后面的内容不会参与编译。

这句话说明下一套配置专门用于：

```text
小型或中型、并且矩阵尺寸不规则的情况。
```

“不规则”通常指：

- M 不是 tile 高度的整数倍；
- N 不是 tile 宽度的整数倍；
- K 不是 BK 的整数倍；
- 边界 block 较多。

---

## 11. 非规则形状配置

```cpp
using WarpTileIrregularConfig = WarpTileConfig<
    64, 128, 8,
    32, 64,
    2,
    4, 4,
    128
>;
```

### 详细解释

对应参数：

```text
BM          = 64
BN          = 128
BK          = 8
WM          = 32
WN          = 64
WNITER      = 2
TM          = 4
TN          = 4
NUM_THREADS = 128
```

一个 block 负责：

```text
64 × 128 = 8192
```

个输出元素。

有：

```text
128 / 32 = 4 个 warp
```

每个 warp 负责：

```text
32 × 64 = 2048
```

个元素。

4 个 warp：

```text
4 × 2048 = 8192
```

正好覆盖整个 Block Tile。

warp 排列：

```text
M 方向：64 / 32 = 2
N 方向：128 / 64 = 2
```

所以仍是：

```text
2 行 × 2 列
```

每线程结果推导：

```text
WSUBN = 64 / 2 = 32
线程列数 = 32 / TN = 32 / 4 = 8
线程行数 = 32 / 8 = 4
WSUBM = 4 × TM = 4 × 4 = 16
WMITER = WM / WSUBM = 32 / 16 = 2
```

最终每线程：

```text
2 × 4 × 2 × 4 = 64 个结果
```

128 个线程：

```text
128 × 64 = 8192
```

该配置的 Block Tile 大小介于 Small 和 Default 之间：

```text
Small：     64 × 64
Irregular： 64 × 128
Default：  128 × 256
```

更小的 tile 可以减少非整倍数尺寸的边缘浪费，但也会增加 block 数量，需要在复用率、并行度和边界损失之间权衡。

---

## 12. 三套配置对照

| 配置 | BM | BN | BK | WM | WN | WNITER | TM | TN | 线程数 | Warp 数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Default | 128 | 256 | 8 | 64 | 64 | 4 | 8 | 4 | 256 | 8 |
| Small | 64 | 64 | 8 | 32 | 32 | 2 | 4 | 4 | 128 | 4 |
| Irregular | 64 | 128 | 8 | 32 | 64 | 2 | 4 | 4 | 128 | 4 |

---

## 13. 各级 Tile 的层次

```text
输出矩阵 C
    ↓
Block Tile：BM × BN
    ↓
Warp Tile：WM × WN
    ↓
Warp 的 M/N 子轮次
    ↓
Thread Tile：TM × TN
```

沿 K 方向则按：

```text
BK
```

分块循环。

一个线程最终负责的结果数是：

```text
WMITER × TM × WNITER × TN
```

整个 block 所有线程负责的总结果数应满足：

```text
NUM_THREADS
× 每线程结果数
= BM × BN
```

这三套配置都满足该关系。

---

## 14. 为什么使用模板配置类型

Kernel 的调用形式类似：

```cpp
launch_gemm_warp_tile_cfg<WarpTileDefaultConfig>(...);
```

而不是运行时传入：

```cpp
BM, BN, BK, WM, WN
```

原因是这些参数会用于：

```cpp
__launch_bounds__(Cfg::NUM_THREADS)
```

共享内存数组大小：

```cpp
__shared__ float AS[2][Cfg::BM * Cfg::BK];
```

以及固定次数循环和寄存器数组尺寸。

这些地方需要参数在编译期已知。

编译期配置可以让编译器：

- 展开固定循环；
- 直接计算数组大小；
- 优化寄存器分配；
- 把常量直接写入指令；
- 针对每套配置生成独立 Kernel。

代价是：

- 每套配置会生成一份 Kernel；
- 配置越多，编译时间和二进制体积越大；
- 运行时不能随意修改 tile 参数。

---

## 15. 完整源码

```cpp
#pragma once

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
struct WarpTileConfig {
    static constexpr int BM = BM_;
    static constexpr int BN = BN_;
    static constexpr int BK = BK_;
    static constexpr int WM = WM_;
    static constexpr int WN = WN_;
    static constexpr int WNITER = WNITER_;
    static constexpr int TM = TM_;
    static constexpr int TN = TN_;
    static constexpr int NUM_THREADS = NUM_THREADS_;
};

using WarpTileDefaultConfig = WarpTileConfig<
    128, 256, 8,
    64, 64,
    4,
    8, 4,
    256
>;

using WarpTileSmallConfig = WarpTileConfig<
    64, 64, 8,
    32, 32,
    2,
    4, 4,
    128
>;

// Dedicated configuration for irregular tiles on small/medium shapes.
using WarpTileIrregularConfig = WarpTileConfig<
    64, 128, 8,
    32, 64,
    2,
    4, 4,
    128
>;
```

---

## 16. 在 Kernel 中的实际使用

当代码写：

```cpp
launch_gemm_warp_tile_cfg<WarpTileSmallConfig>(...);
```

模板实例化后：

```cpp
Cfg::BM
Cfg::BN
Cfg::BK
Cfg::NUM_THREADS
```

分别在编译期等于：

```text
64
64
8
128
```

对应 Kernel 中的代码可以近似理解为：

```cpp
__global__ __launch_bounds__(128)
void gemm_warp_tile_kernel(...)
{
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;

    __shared__ float AS[2][64 * 8];
    __shared__ float BS[2][8 * 64];
}
```

因此该配置头文件不会执行矩阵乘法，它只是为 Kernel 提供编译期参数。

---

## 17. 最核心的一句话

```text
gemm_warp_tile_config.h 使用“模板结构体 + static constexpr + using 类型别名”
封装三套 Warp Tile 参数，使同一份 CUDA Kernel 源码能够在编译期生成
Default、Small 和 Irregular 三个不同的专用版本。
```


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-gemm算子|模块-gemm算子]]

%% 项目关联导航：结束 %%
