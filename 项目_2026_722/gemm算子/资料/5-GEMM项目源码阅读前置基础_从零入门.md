# GEMM 项目源码阅读前置基础：从零建立 C++、CUDA 与矩阵乘法知识

> 适用项目：当前 `gemm-project` 仓库中的 CUDA FP32 GEMM 实现。  
> 目标：不是把 C++ 和 CUDA 的所有知识一次学完，而是掌握**足以读懂该项目的数据结构、控制流程、线程映射、内存访问与性能测试**的最小完整知识体系。

---

## 0. 先建立项目全貌：你最终要看懂什么

这个项目实现的是单精度浮点矩阵乘法：

```text
C = alpha × A × B + beta × C
```

矩阵形状为：

```text
A: M × K
B: K × N
C: M × N
```

项目不是直接让一个线程计算整个矩阵，而是把工作分成多个层级：

```text
整个 C 矩阵
  ↓ 按 Block Tile 切分
每个 CUDA 线程块负责一块 C
  ↓ 按 Warp Tile 切分
每个 warp 负责线程块结果中的一小块
  ↓ 按 Thread Tile 切分
每个线程用寄存器累计多个 C 元素
```

与此同时，A、B 的数据按照 K 维分段搬运：

```text
Global Memory
  ↓ cp.async / 普通加载
Shared Memory 双缓冲
  ↓
Registers
  ↓ 乘加累计
C 的 Global Memory
```

当前仓库的重要文件如下：

```text
include/gemm/types.h
    定义 GemmAlgo 枚举

include/gemm/launcher.h
    声明对外接口 launch_gemm

include/gemm/kernels/gemm_warp_tile_config.h
    定义 Block Tile、Warp Tile、Thread Tile 等编译期参数

src/launcher.cu
    选择算法并调用具体 kernel launcher

src/kernels/gemm_warp_tile.cu
    核心 CUDA GEMM 实现

 tests/test_perf_gemm.cpp
    分配内存、运行 cuBLAS、运行自定义 GEMM、计时并校验结果

microbenchmark/
    测试显存、L2、共享内存等硬件性能
```

你需要分三层学习：

| 学习层级 | 目标 | 对应源码 |
|---|---|---|
| 第一层：必须掌握 | 看懂测试程序、接口、内存分配和函数调用 | `types.h`、`launcher.h`、`launcher.cu`、`test_perf_gemm.cpp` |
| 第二层：核心掌握 | 看懂 kernel 如何划分线程、共享内存和计算任务 | `gemm_warp_tile_config.h`、`gemm_warp_tile.cu` 主体 |
| 第三层：进阶理解 | 看懂 `cp.async`、内联 PTX、双缓冲、向量化访存 | `gemm_warp_tile.cu` 的 fast path |

第一次阅读时，不要求立刻看懂内联汇编。先把“一个线程块算哪块、一个 warp 算哪块、一个线程算哪些元素”弄清楚，再研究加载优化。

---

# 1. C/C++ 基础

## 1.1 变量、类型与函数参数

项目中常见的基本类型：

```cpp
int M;
float alpha;
bool fast;
uint32_t addr;
size_t sizeA;
```

含义：

| 类型 | 典型用途 |
|---|---|
| `int` | 矩阵维度、循环变量、索引 |
| `float` | FP32 矩阵元素、计算结果 |
| `bool` | 路径选择，例如是否进入 fast path |
| `uint` / `uint32_t` | 非负索引或固定 32 位整数 |
| `size_t` | 内存大小、元素数量，适合表达可能很大的无符号数 |

函数声明示例：

```cpp
void launch_gemm(const int M,
                 const int N,
                 const int K,
                 const float alpha,
                 const float* A,
                 const float* B,
                 const float beta,
                 float* C);
```

这里可以按三类理解：

```text
M、N、K、alpha、beta：按值传入
A、B：传入只读数据的地址
C：传入可写数据的地址
```

`const int M` 表示函数内部不能给形参 `M` 重新赋值。不过它只是一个局部副本，调用者本身的变量不会被修改。

---

## 1.2 指针到底是什么

指针可以先理解成“保存内存地址的变量”。

```cpp
float x = 3.0f;
float* p = &x;
```

此时：

```text
x       保存数值 3.0
&x      得到 x 的地址
p       保存 x 的地址
*p      访问该地址中的数值，也就是 x
```

示例：

```cpp
float x = 3.0f;
float* p = &x;

*p = 8.0f;

// 此时 x 也变成 8.0f
```

在 GEMM 项目中：

```cpp
float* A = (float*)malloc(sizeA);
float* d_A;
cudaMalloc(&d_A, sizeA);
```

两者虽然都是 `float*`，但地址属于不同内存空间：

```text
A    → CPU 主机内存
 d_A  → GPU 设备显存
```

因此，不能因为它们类型相同，就在 CPU 代码里直接使用 `d_A[i]`。

---

## 1.3 `const float*`、`float* const` 与普通指针

这是阅读项目必须熟练掌握的内容。

### 1.3.1 普通指针

```cpp
float* p;
```

含义：

```text
p 可以指向其他地址；
*p 指向的数据也可以修改。
```

### 1.3.2 指向只读数据的指针

```cpp
const float* p;
```

也可写成：

```cpp
float const* p;
```

含义：

```text
p 可以改变指向；
不能通过 p 修改它所指向的数据。
```

例如：

```cpp
const float* A;
```

表示 kernel 只读取 A，不应写入 A。

### 1.3.3 自身不可改变的指针

```cpp
float* const p = ...;
```

含义：

```text
p 不能再指向别处；
但可以修改 *p。
```

### 1.3.4 指针和数据都不可修改

```cpp
const float* const p = ...;
```

项目中最常见的是：

```cpp
const float* A;
const float* B;
float* C;
```

它明确表达了 GEMM 的数据流：A、B 是输入，C 是输出。

---

## 1.4 一维数组如何模拟二维矩阵

GPU 内存本质上是一段连续的一维地址。二维矩阵通常按一维数组保存。

假设矩阵：

```text
A =
[ a00 a01 a02
  a10 a11 a12 ]
```

它有 2 行、3 列。在 Row-major（行优先）布局中，内存顺序为：

```text
a00, a01, a02, a10, a11, a12
```

第 `row` 行、第 `col` 列的地址：

```cpp
A[row * leading_dim + col]
```

对于没有额外填充的普通矩阵，`leading_dim` 就是列数。

因此：

```cpp
A[row * K + k]  // A 是 M×K
B[k * N + col]  // B 是 K×N
C[row * N + col] // C 是 M×N
```

### 例子

A 是 3×4 矩阵：

```text
row = 2
col = 1
```

下标为：

```text
2 × 4 + 1 = 9
```

所以访问：

```cpp
A[9]
```

### `leading_dim` 为什么不一定等于列数

有些库允许每一行末尾额外填充，使相邻两行的起始位置间隔大于真实列数。此时：

```text
leading_dim ≥ 实际列数
```

在当前项目的自定义 GEMM 中，没有额外 padding：

```text
A 的 leading dimension = K
B 的 leading dimension = N
C 的 leading dimension = N
```

---

## 1.5 指针算术

对 `float*` 加 1，不是地址增加 1 字节，而是前进一个 `float`：

```cpp
float* p;
p + 1;
```

地址实际增加：

```text
sizeof(float) = 4 字节
```

项目中：

```cpp
const float* A_blk = A + c_row * BM * K;
```

含义是把 `A_blk` 移动到 A 矩阵第 `c_row × BM` 行的开头：

```text
行起点 = c_row × BM
每行 K 个 float
偏移 = c_row × BM × K 个 float
```

再如：

```cpp
const float* B_blk = B + c_col * BN;
```

B 是 K×N 行优先矩阵。这个偏移让指针移动到 B 第 0 行中，本线程块负责的列块起点。

---

## 1.6 `size_t` 与 `sizeof`

在测试代码中：

```cpp
size_t sizeA = static_cast<size_t>(M) * K * sizeof(float);
```

假设：

```text
M = 1024
K = 1024
sizeof(float) = 4 字节
```

则：

```text
sizeA = 1024 × 1024 × 4
      = 4,194,304 字节
      = 4 MiB
```

为什么先写：

```cpp
static_cast<size_t>(M)
```

而不是：

```cpp
M * K * sizeof(float)
```

因为 `M * K` 如果先用 32 位 `int` 计算，尺寸很大时可能先溢出，再转换为 `size_t`。提前把第一个操作数转换成 `size_t`，后续乘法都会使用更适合表示内存大小的类型。

### 必须形成的习惯

内存字节数统一写成：

```cpp
size_t bytes = static_cast<size_t>(num_elements) * sizeof(Type);
```

---

## 1.7 `static_cast`：正常的显式类型转换

项目中：

```cpp
static_cast<size_t>(M)
static_cast<float>(cnt)
```

它表示程序员明确要求进行一种常规类型转换。

例如：

```cpp
int x = 3;
float y = static_cast<float>(x);
```

得到：

```text
y = 3.0f
```

推荐使用 `static_cast`，而不是 C 风格写法：

```cpp
(float)x
```

因为前者意图更明确，更容易被编译器检查。

---

## 1.8 `reinterpret_cast`：不改变比特，只改变观察方式

核心 kernel 中：

```cpp
float4 v = *reinterpret_cast<const float4*>(&C_out[row * N + col]);
```

先理解 `float4`：

```cpp
struct float4 {
    float x;
    float y;
    float z;
    float w;
};
```

一个 `float4` 共包含 4 个连续 `float`，也就是 16 字节。

`reinterpret_cast<const float4*>` 的作用不是把一个 float 数值转换为四个 float，而是告诉编译器：

```text
从这个地址开始的 16 字节，请当成一个 float4 读取。
```

随后：

```cpp
v.x
v.y
v.z
v.w
```

分别对应连续四个 C 元素。

写回时：

```cpp
*reinterpret_cast<float4*>(&C_out[row * N + col]) = v;
```

一次写入 16 字节。

### 使用它需要满足的条件

1. 地址具有足够的对齐条件；
2. 后面至少还有 4 个有效 `float`；
3. 访问不能跨出当前有效矩阵范围。

当前项目只在 fast path 中执行 `float4` 写回，并通过尺寸整除、Tile 边界和线程映射保证访问条件。

第一遍阅读时只需记住：

```text
reinterpret_cast<float4*> = 把连续 4 个 float 作为一个 16 字节向量访问
```

---

## 1.9 `enum class` 与 `switch`

项目中：

```cpp
enum class GemmAlgo {
    Auto,
    WarpTile,
};
```

它定义了一个“算法类型”，合法值只有：

```text
GemmAlgo::Auto
GemmAlgo::WarpTile
```

调用接口时：

```cpp
launch_gemm(..., GemmAlgo::Auto);
```

在 `launcher.cu` 中：

```cpp
switch (final_algo) {
    case GemmAlgo::WarpTile:
        launch_gemm_warp_tile(...);
        break;

    case GemmAlgo::Auto:
    default:
        launch_gemm_warp_tile(...);
        break;
}
```

作用是根据枚举值选择实现。

### 为什么不用普通整数

若用：

```cpp
int algo = 1;
```

无法直观看出 1 代表什么，也可能传入 99 这样的无效值。

`enum class` 更安全、可读性更强。

---

## 1.10 模板参数与 `static constexpr`

项目配置文件中：

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
struct WarpTileConfig {
    static constexpr int BM = BM_;
    static constexpr int BN = BN_;
    static constexpr int BK = BK_;
    // ...
};
```

模板可以先理解为“让编译器根据不同参数生成不同版本的代码”。

例如：

```cpp
using WarpTileDefaultConfig = WarpTileConfig<
    128, 256, 8,
    64, 64,
    4,
    8, 4,
    256
>;
```

等价于生成一组编译期配置：

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

kernel 中：

```cpp
template <typename Cfg>
__global__ void gemm_warp_tile_kernel(...) {
    constexpr int BM = Cfg::BM;
    constexpr int BN = Cfg::BN;
}
```

当启动：

```cpp
gemm_warp_tile_kernel<WarpTileDefaultConfig><<<...>>>(...);
```

编译器会得到所有确定常量，因此可以：

- 展开循环；
- 计算数组大小；
- 删除无用分支；
- 为不同尺寸配置分别优化。

### `static constexpr` 的含义

```cpp
static constexpr int BM = BM_;
```

可拆开理解：

- `static`：属于类型本身，不依赖某个对象实例；
- `constexpr`：编译期常量；
- `int`：整数。

你不需要创建 `WarpTileConfig` 对象，直接写：

```cpp
Cfg::BM
```

即可取得配置值。

---

## 1.11 头文件声明与 `.cu/.cpp` 实现分离

项目中：

```text
include/gemm/launcher.h
src/launcher.cu
```

头文件中只有声明：

```cpp
void launch_gemm(...);
```

源文件中给出实现：

```cpp
void launch_gemm(...) {
    // 函数体
}
```

调用者只需要：

```cpp
#include "gemm/launcher.h"
```

就知道函数名称、参数和返回值，不需要看到内部实现。

### `#pragma once`

头文件开头：

```cpp
#pragma once
```

作用是避免一个头文件被重复包含，引发重复定义问题。

### `.cpp` 与 `.cu`

```text
.cpp：通常由 C++ 编译器处理
.cu：由 nvcc 处理，可以包含 CUDA kernel 和 CUDA 语法
```

当前项目中：

```text
tests/test_perf_gemm.cpp     主机端测试程序
src/launcher.cu              CUDA 编译单元，但主要是 host launcher
src/kernels/*.cu             包含 GPU kernel
```

---

## 1.12 `namespace`、`inline` 与匿名命名空间

`src/launcher.cu` 中：

```cpp
namespace {

inline GemmAlgo select_gemm_algo(...) {
    return GemmAlgo::WarpTile;
}

}
```

不带名字的命名空间称为匿名命名空间。里面的函数只在当前源文件可见，避免和其他文件中的同名函数冲突。

`inline` 对当前学习阶段可理解为：

```text
允许把短函数直接展开到调用位置，并且处理头文件中的重复定义规则。
```

是否真的展开由编译器决定。

CUDA 中还有：

```cpp
__forceinline__
```

它比普通 `inline` 更强烈地要求编译器内联。项目用它减少设备函数调用开销，并帮助编译器跨函数优化。

---

## 1.13 宏与 `CEIL_DIV`

项目中：

```cpp
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define WARP_SIZE 32
```

`CEIL_DIV(x, y)` 表示整数向上取整除法。

例如：

```text
CEIL_DIV(1000, 256)
= (1000 + 256 - 1) / 256
= 1255 / 256
= 4
```

虽然前三块只能覆盖 768 个元素，但第四块负责剩下的 232 个元素。

项目中：

```cpp
dim3 gridDim(CEIL_DIV(N, Cfg::BN),
             CEIL_DIV(M, Cfg::BM));
```

确保即使 M、N 不是 Tile 大小的整数倍，也会启动额外线程块处理边界。

---

## 1.14 循环、条件与三目运算符

项目中：

```cpp
const int m_edge =
    (c_row * BM + BM <= M) ? BM : (M - c_row * BM);
```

三目运算符格式：

```cpp
condition ? value_if_true : value_if_false
```

上面的意思：

```text
如果整个 BM 行 Tile 都在矩阵范围内：m_edge = BM；
否则：m_edge = 剩余有效行数。
```

另一个常见写法：

```cpp
int cur = t & 1;
int nxt = 1 - cur;
```

`t & 1` 用于判断奇偶：

```text
t 为偶数 → cur = 0
t 为奇数 → cur = 1
```

因此两个共享内存缓冲区在 0 和 1 之间交替使用。

---

# 2. CUDA 基础

## 2.1 Host 与 Device 的区别

CUDA 程序同时包含 CPU 代码和 GPU 代码。

```text
Host   = CPU 及其主内存
Device = GPU 及其显存
```

典型执行流程：

```text
1. CPU 准备输入数据
2. CPU 调用 cudaMalloc 分配 GPU 显存
3. CPU 调用 cudaMemcpy，把数据复制到 GPU
4. CPU 启动 kernel
5. GPU 并行执行 kernel
6. CPU 等待 GPU 完成
7. CPU 把结果复制回来
8. 释放 GPU 显存
```

测试代码正是这一流程。

---

## 2.2 `cudaMalloc`：分配设备显存

```cpp
float* d_A;
cudaMalloc(&d_A, sizeA);
```

你可以把它理解为：

```text
请 CUDA Runtime 在 GPU 显存中分配 sizeA 字节，
并把首地址写入 d_A。
```

为什么传 `&d_A`：

```text
d_A  是地址变量；
&d_A 是这个地址变量自身的地址；
cudaMalloc 需要修改 d_A，让它指向新分配的显存。
```

与 CPU `malloc` 对比：

```cpp
float* A = static_cast<float*>(malloc(sizeA));  // CPU 内存
float* d_A;
cudaMalloc(&d_A, sizeA);                       // GPU 显存
```

---

## 2.3 `cudaMemcpy`：Host 与 Device 之间复制

```cpp
cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
```

参数顺序：

```text
目标地址、源地址、字节数、复制方向
```

常见方向：

```cpp
cudaMemcpyHostToDevice
cudaMemcpyDeviceToHost
cudaMemcpyDeviceToDevice
```

当前项目：

```cpp
cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
```

分别表示：

```text
A：CPU → GPU
C：GPU → CPU
```

---

## 2.4 `cudaMemset`：按字节填充设备内存

```cpp
cudaMemset(d_C, 0, sizeC);
```

它把 `sizeC` 个字节全部置为 0。

对 IEEE 浮点数而言，全 0 比特表示 `0.0f`，所以可以用它清零 float 数组。

但不要误以为：

```cpp
cudaMemset(d_C, 1, sizeC);
```

会把每个 float 设置为 `1.0f`。它实际会把每个字节设置为 `0x01`，得到的浮点值不是 1.0。

---

## 2.5 `cudaFree`：释放显存

```cpp
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
```

和 CPU 的：

```cpp
free(A);
```

作用类似，但必须配对使用：

```text
malloc    ↔ free
cudaMalloc ↔ cudaFree
```

---

## 2.6 CUDA 函数限定符

### `__global__`

```cpp
__global__ void kernel(...) {
}
```

表示：

```text
函数由 Host 启动，在 Device 上执行。
```

这种函数称为 kernel。

当前项目：

```cpp
__global__
void gemm_warp_tile_kernel(...)
```

### `__device__`

```cpp
__device__ void helper(...) {
}
```

表示：

```text
函数在 GPU 上执行，只能从 GPU 代码中调用。
```

当前项目的：

```cpp
compute(...)
load_tile_async(...)
cp_async16(...)
```

都是设备辅助函数。

### `__host__`

表示函数在 CPU 上执行。普通 C++ 函数默认就是 Host 函数，因此通常不用显式写。

### `__forceinline__`

```cpp
__device__ __forceinline__ void compute(...)
```

表示这是设备函数，并强烈建议编译器把函数体内联到调用位置。

---

## 2.7 Kernel 启动语法

普通函数调用：

```cpp
foo(x, y);
```

CUDA kernel 调用：

```cpp
kernel<<<gridDim, blockDim>>>(x, y);
```

项目中：

```cpp
gemm_warp_tile_kernel<Cfg><<<gridDim, blockDim>>>(
    M, N, K, alpha, A, B, beta, C);
```

其中：

```text
gridDim  决定启动多少个线程块
blockDim 决定每个线程块有多少个线程
```

---

## 2.8 `dim3`、Grid、Block、Thread

CUDA 的线程组织层次：

```text
Grid
└── Block
    └── Thread
```

每一层都可以是一维、二维或三维。

```cpp
dim3 blockDim(256);
dim3 gridDim(4, 3);
```

表示：

```text
每个 block 有 256 个线程；
grid 在 x 方向有 4 个 block，在 y 方向有 3 个 block；
总 block 数 = 4 × 3 = 12；
总线程数 = 12 × 256。
```

项目中：

```cpp
dim3 gridDim(
    CEIL_DIV(N, Cfg::BN),
    CEIL_DIV(M, Cfg::BM));
```

含义：

```text
Grid.x 沿 C 的列方向切 Tile；
Grid.y 沿 C 的行方向切 Tile。
```

---

## 2.9 `blockIdx`、`threadIdx`、`blockDim`、`gridDim`

在 kernel 内，CUDA 自动提供：

```cpp
blockIdx.x
blockIdx.y
threadIdx.x
blockDim.x
gridDim.x
```

含义：

| 变量 | 含义 |
|---|---|
| `blockIdx` | 当前线程块在 Grid 中的编号 |
| `threadIdx` | 当前线程在线程块中的编号 |
| `blockDim` | 每个线程块的形状 |
| `gridDim` | Grid 的形状 |

最经典的一维索引：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

当前 GEMM kernel 采用二维 Block Tile 映射：

```cpp
const uint c_row = blockIdx.y;
const uint c_col = blockIdx.x;
```

所以：

```text
blockIdx.y 决定 C 的第几个行 Tile；
blockIdx.x 决定 C 的第几个列 Tile。
```

---

## 2.10 Warp：硬件调度的 32 线程小组

CUDA 线程块中的线程通常按连续编号划分为 warp：

```text
线程 0～31    → warp 0
线程 32～63   → warp 1
线程 64～95   → warp 2
线程 96～127  → warp 3
```

一个 warp 固定包含 32 个线程。

项目中：

```cpp
#define WARP_SIZE 32

const uint warp_idx = threadIdx.x / WARP_SIZE;
const uint tiwarp   = threadIdx.x % WARP_SIZE;
```

其中：

```text
warp_idx：当前线程属于线程块中的第几个 warp
tiwarp：当前线程在 warp 内的 lane 编号，范围 0～31
```

### 为什么 warp 很重要

GPU 并不是把每个线程完全独立调度，而是以 warp 为主要执行单位。一个 warp 的线程通常执行同一条指令。

如果 warp 内线程进入不同分支：

```cpp
if (condition) {
    // 一部分线程执行
} else {
    // 另一部分线程执行
}
```

硬件可能需要分批执行两条路径，这称为 warp divergence（线程束分歧）。

当前项目的 `fast` 条件由整个 block 的位置决定，同一个 block 内所有线程得到相同结果，因此不会产生 warp 内分歧。

---

## 2.11 CUDA 内存层级

从当前项目角度，可以先掌握四层：

| 存储位置 | 范围 | 特点 | 项目用途 |
|---|---|---|---|
| Global Memory | 整个 GPU | 容量大、延迟高 | 保存完整 A、B、C |
| Shared Memory | 一个 block | 片上、速度快、block 内共享 | 保存 A、B Tile |
| Registers | 单个线程 | 最快、容量有限 | 保存 `reg_m`、`reg_n`、`res` |
| L1/L2 Cache | 硬件管理 | 缓存 Global Memory 访问 | 降低重复显存访问成本 |

数据流可以概括为：

```text
A、B 的 Global Memory
        ↓
AS、BS 的 Shared Memory
        ↓
reg_m、reg_n 寄存器
        ↓
res 寄存器累计
        ↓
C 的 Global Memory
```

---

## 2.12 `__shared__`：线程块共享内存

项目中：

```cpp
__shared__ float AS[2][A_STRIDE * BK];
__shared__ float BS[2][BK * BN];
```

含义：

```text
同一个 block 的所有线程共享 AS 和 BS；
不同 block 之间互相不可见；
每个 block 都有自己独立的 AS、BS。
```

`AS[2]` 和 `BS[2]` 中的 2 表示双缓冲：

```text
缓冲区 0：当前正在计算的 Tile
缓冲区 1：下一块正在加载的 Tile
```

下一轮交换角色。

---

## 2.13 `__syncthreads()`：线程块同步屏障

当多个线程协作向共享内存写数据时，某些线程可能写得快，某些写得慢。

若立即读取：

```cpp
// 多线程写 AS
// 立即读取 AS
```

可能读到尚未写完的数据。

因此使用：

```cpp
__syncthreads();
```

它表示：

```text
当前 block 内的所有线程都到达这里后，任何线程才能继续。
```

典型模式：

```cpp
// 所有线程合作加载共享内存
AS[...] = ...;
BS[...] = ...;

__syncthreads();

// 所有线程读取共享内存并计算
```

### 重要规则

不能让同一个 block 中只有一部分线程执行 `__syncthreads()`，另一部分永远不执行，否则可能死锁。

当前项目中的同步位置按 block 级统一控制。

---

## 2.14 合并访存的基本概念

Global Memory 访问很昂贵。硬件希望同一 warp 的线程访问连续或相邻地址，从而把多个访问合并为较少的内存事务。

理想情况：

```text
lane 0 访问 A[0]
lane 1 访问 A[1]
lane 2 访问 A[2]
...
```

不理想情况：

```text
lane 0 访问 A[0]
lane 1 访问 A[1024]
lane 2 访问 A[2048]
...
```

项目使用：

```cpp
cp_async16(...)
float4
```

让每次操作搬运 16 字节，即连续 4 个 float，并精心设计线程索引，使一个 warp 的访问尽量连续。

第一遍阅读只需掌握：

```text
相邻线程访问相邻地址，通常更有利于显存带宽利用。
```

---

## 2.15 Kernel 启动是异步的

CPU 执行：

```cpp
kernel<<<grid, block>>>(...);
```

通常只是把任务提交给 GPU，然后 CPU 可以继续往下执行。它不一定等待 kernel 完成。

需要等待时，可使用：

```cpp
cudaDeviceSynchronize();
```

或者同步某个 event：

```cpp
cudaEventSynchronize(stop);
```

这也是为什么不能使用普通 CPU 计时直接包住 kernel 启动，否则可能只测到“提交任务”的时间。

---

## 2.16 CUDA Event 计时

项目中：

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
for (...) {
    launch_gemm(...);
}
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
```

流程：

```text
在 GPU 工作流中记录 start
启动多次 kernel
在 GPU 工作流中记录 stop
等待 stop 完成
计算两个 event 之间的毫秒数
```

最后销毁：

```cpp
cudaEventDestroy(start);
cudaEventDestroy(stop);
```

---

## 2.17 CUDA 错误检查

测试代码中：

```cpp
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg
                  << " CUDA ERROR: "
                  << cudaGetErrorString(err)
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}
```

调用：

```cpp
checkCudaError(cudaMalloc(&d_A, sizeA),
               "cudaMalloc d_A failed");
```

这是一种包装函数：每次调用 CUDA API 后立即检查错误。

kernel 启动错误通常检查：

```cpp
kernel<<<...>>>(...);
cudaGetLastError();
```

运行期错误往往要等同步时才暴露：

```cpp
cudaDeviceSynchronize();
```

---

# 3. GEMM 数学与数据布局基础

## 3.1 什么是 GEMM

GEMM 全称：

```text
General Matrix-Matrix Multiplication
```

标准形式：

```text
C = alpha × op(A) × op(B) + beta × C
```

当前自定义实现使用不转置的 A、B：

```text
C = alpha × A × B + beta × C
```

其中：

```text
A: M × K
B: K × N
C: M × N
```

内维 K 必须相同，才能相乘。

---

## 3.2 一个 C 元素是怎样得到的

公式：

```text
C[m,n] = Σ A[m,k] × B[k,n]
          k=0...K-1
```

即：

```text
A 的第 m 行
点乘
B 的第 n 列
```

### 具体例子

```text
A = [1 2 3
     4 5 6]

B = [ 7  8
      9 10
     11 12]
```

A 是 2×3，B 是 3×2，所以 C 是 2×2。

```text
C[0,0] = 1×7 + 2×9 + 3×11 = 58
C[0,1] = 1×8 + 2×10 + 3×12 = 64
C[1,0] = 4×7 + 5×9 + 6×11 = 139
C[1,1] = 4×8 + 5×10 + 6×12 = 154
```

得到：

```text
C = [ 58  64
     139 154]
```

---

## 3.3 CPU 版本 GEMM

最直接的 C++ 实现：

```cpp
void gemm_cpu(int M, int N, int K,
              const float* A,
              const float* B,
              float* C) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}
```

三重循环对应：

```text
m：遍历 C 的行
n：遍历 C 的列
k：完成一个点积
```

理解这个版本，是阅读任何 GPU GEMM 的起点。

GPU 优化版本虽然复杂，本质上仍然是在大量执行：

```cpp
sum += A[...] * B[...];
```

---

## 3.4 `alpha` 和 `beta`

项目输出不是简单的：

```text
C = A × B
```

而是：

```text
C = alpha × A × B + beta × C_old
```

kernel 写回：

```cpp
C_out[...] = alpha * res[...] + beta * C_out[...];
```

常见设置：

```cpp
alpha = 1.0f;
beta = 0.0f;
```

此时：

```text
C = A × B
```

若：

```text
alpha = 2
beta = 1
```

则：

```text
C_new = 2 × A × B + C_old
```

---

## 3.5 GEMM 为什么约有 `2MNK` 次浮点运算

每个 C 元素需要 K 次乘法和大约 K 次加法：

```text
约 2K 次浮点运算
```

C 一共有：

```text
M × N 个元素
```

所以总运算量约为：

```text
2 × M × N × K FLOPs
```

严格计算点积是 K 次乘法与 K-1 次加法，但性能分析中通常按每次 FMA 等价 2 FLOPs 计算，因此使用：

```text
2MNK
```

---

## 3.6 GFLOPS 计算

GFLOPS 表示每秒十亿次浮点运算。

若执行一次 GEMM 的时间是 `time_ms` 毫秒：

```text
GFLOPS = 2MNK / (time_ms × 10^6)
```

推导：

```text
运算次数 = 2MNK
秒数 = time_ms × 10^-3
FLOPs/s = 2MNK / (time_ms × 10^-3)
GFLOPS = FLOPs/s / 10^9
        = 2MNK / (time_ms × 10^6)
```

项目重复 `repeat_time` 次，因此：

```cpp
repeat_time * 2.0f * M * N * K /
(time_ms * 1e6f)
```

---

## 3.7 为什么要 Tile

假设每个线程独立计算一个 C 元素：

```text
C[m,n] 需要读取：
A 的 K 个元素
B 的 K 个元素
```

相邻的多个 C 元素会反复读取相同数据：

- 同一行的 C 元素重复使用 A 的同一行；
- 同一列的 C 元素重复使用 B 的同一列。

Tile 的核心思想：

```text
先把一小块 A、B 搬进共享内存，
让一个 block 内多个线程反复使用，
避免每次乘法都重新访问 Global Memory。
```

这就是截图中“让一份加载的数据被多个乘加重复利用”的含义。

---

## 3.8 最简单的 CUDA GEMM

下面是一线程计算一个 C 元素的版本：

```cpp
__global__ void gemm_naive(int M, int N, int K,
                           const float* A,
                           const float* B,
                           float* C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

启动：

```cpp
dim3 block(16, 16);
dim3 grid(CEIL_DIV(N, 16), CEIL_DIV(M, 16));

gemm_naive<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
```

这里：

```text
每个线程只计算一个 C[row,col]
```

而当前项目更进一步：

```text
每个线程计算多个 C 元素，
并把中间结果长期保存在寄存器中。
```

---

## 3.9 Block Tile、Warp Tile、Thread Tile、K Tile

这是理解当前项目最重要的四个概念。

### Block Tile

一个 CUDA block 负责 C 的一个矩形区域：

```text
BM × BN
```

Default 配置：

```text
BM = 128
BN = 256
```

即一个 block 负责 C 中 128 行 × 256 列，共 32768 个输出元素。

### K Tile

K 维不是一次全部加载，而是每次处理：

```text
BK = 8
```

若 K=1024：

```text
1024 / 8 = 128 个 K Tile
```

每轮加载：

```text
A Tile: BM × BK = 128 × 8
B Tile: BK × BN = 8 × 256
```

计算后继续下一段 K。

### Warp Tile

一个 block 中有多个 warp，每个 warp 负责：

```text
WM × WN
```

Default 配置：

```text
WM = 64
WN = 64
```

所以每个 warp 负责 C 的 64×64 区域。

### Thread Tile

一个 warp 中有 32 个线程，每个线程又负责多个输出元素。

Default 配置中：

```text
TM = 8
TN = 4
WNITER = 4
```

最终每个线程持有：

```text
8 × (4 × 4) = 128 个 C 累加值
```

对应源码：

```cpp
float res[WMITER * TM * WNITER * TN] = {};
```

Default 配置下 `WMITER=1`，所以：

```text
res 大小 = 1 × 8 × 4 × 4 = 128
```

---

# 4. 把当前项目的线程映射彻底展开

## 4.1 Default 配置

```cpp
using WarpTileDefaultConfig = WarpTileConfig<
    128, 256, 8,
    64, 64,
    4,
    8, 4,
    256
>;
```

对应：

| 参数 | 值 | 含义 |
|---|---:|---|
| `BM` | 128 | 一个 block 负责 C 的 128 行 |
| `BN` | 256 | 一个 block 负责 C 的 256 列 |
| `BK` | 8 | 每次沿 K 处理 8 个元素 |
| `WM` | 64 | 一个 warp 负责 64 行 |
| `WN` | 64 | 一个 warp 负责 64 列 |
| `WNITER` | 4 | warp 的列 Tile 再拆成 4 段 |
| `TM` | 8 | 一个线程每段负责 8 行 |
| `TN` | 4 | 一个线程每段负责 4 列 |
| `NUM_THREADS` | 256 | 一个 block 有 256 线程，即 8 个 warp |

---

## 4.2 8 个 warp 如何覆盖 128×256 的 Block Tile

一个 warp 负责 64×64。

沿行方向：

```text
BM / WM = 128 / 64 = 2
```

沿列方向：

```text
BN / WN = 256 / 64 = 4
```

所以 warp 排列为：

```text
2 行 × 4 列 = 8 个 warp
```

示意：

```text
Block Tile: 128 × 256

+--------+--------+--------+--------+
| warp 0 | warp 1 | warp 2 | warp 3 |  64 行
+--------+--------+--------+--------+
| warp 4 | warp 5 | warp 6 | warp 7 |  64 行
+--------+--------+--------+--------+
 每块均为 64 × 64
```

源码：

```cpp
const uint warp_idx = threadIdx.x / 32;
const uint warp_row = warp_idx / (BN / WN);
const uint warp_col = warp_idx % (BN / WN);
```

因为：

```text
BN / WN = 4
```

所以：

| `warp_idx` | `warp_row` | `warp_col` |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 2 | 0 | 2 |
| 3 | 0 | 3 |
| 4 | 1 | 0 |
| 5 | 1 | 1 |
| 6 | 1 | 2 |
| 7 | 1 | 3 |

---

## 4.3 一个线程负责哪些输出

源码计算：

```cpp
constexpr int WSUBN = WN / WNITER;
constexpr int WSUBM = (WARP_SIZE / (WSUBN / TN)) * TM;
constexpr int WMITER = WM / WSUBM;
```

代入 Default 参数：

```text
WSUBN = 64 / 4 = 16
WSUBN / TN = 16 / 4 = 4
WSUBM = (32 / 4) × 8 = 64
WMITER = 64 / 64 = 1
```

warp 内线程坐标：

```cpp
const uint tiwarp = threadIdx.x % 32;
const uint tcol = tiwarp % 4;
const uint trow = tiwarp / 4;
```

因此：

```text
tcol 范围 0～3
trow 范围 0～7
```

32 个线程形成逻辑上的：

```text
8 行 × 4 列线程网格
```

每个线程：

- 行方向负责 `TM=8` 个元素；
- 每个列子块负责 `TN=4` 个元素；
- 共处理 `WNITER=4` 个列子块。

因此每线程累计：

```text
8 行 × 4 列 × 4 个列子块 = 128 个 C 元素
```

整个 warp：

```text
32 × 128 = 4096 个元素 = 64 × 64
```

这正好覆盖一个 Warp Tile。

---

## 4.4 Grid 如何覆盖整个 C

启动代码：

```cpp
dim3 gridDim(
    CEIL_DIV(N, Cfg::BN),
    CEIL_DIV(M, Cfg::BM));
```

假设：

```text
M = 1000
N = 1000
BM = 128
BN = 256
```

则：

```text
Grid.x = ceil(1000 / 256) = 4
Grid.y = ceil(1000 / 128) = 8
```

总共：

```text
4 × 8 = 32 个 block
```

最右侧和最下方的 block 会超出矩阵完整 Tile 范围，所以进入 edge path。

---

# 5. Shared Memory 中的布局转换

## 5.1 Global Memory 中的 A、B

A 是行优先：

```cpp
A[m * K + k]
```

B 是行优先：

```cpp
B[k * N + n]
```

---

## 5.2 Shared Memory 中的 AS、BS

项目注释：

```cpp
// AS layout [BK][A_STRIDE]
// AS[d * A_STRIDE + m]

// BS layout [BK][BN]
// BS[d * BN + n]
```

也就是说：

```text
AS 逻辑形状：BK × BM
BS 逻辑形状：BK × BN
```

注意，A 的 Global Memory Tile 本来是：

```text
BM × BK
```

加载到共享内存后，按：

```text
BK × BM
```

组织。

这样在计算阶段，对固定的 `d`：

```cpp
AS[d * A_STRIDE + row]
```

可以按线程所需的 M 方向读取。

这不是数学意义上把整个 A 矩阵转置，而是为了当前 kernel 的共享内存读取模式，对一个 Tile 做重新排布。

---

## 5.3 为什么 A、B 数据能被重复利用

在某个 K Tile 中：

```text
AS 保存 BM×BK 个 A 元素
BS 保存 BK×BN 个 B 元素
```

通过它们可计算：

```text
BM×BN 个 C 局部结果
```

以 Default 配置为例：

```text
A Tile 元素数 = 128×8 = 1024
B Tile 元素数 = 8×256 = 2048
总加载 = 3072 个 float
```

这些数据参与：

```text
128×256×8 = 262144 次乘法
```

即少量数据加载被大量计算重复使用。

---

# 6. 寄存器分块与计算循环

项目的核心计算可以抽象为：

```cpp
for (int d = 0; d < BK; ++d) {
    // 从共享内存读取 A 的若干值到 reg_m
    // 从共享内存读取 B 的若干值到 reg_n

    for (每个 reg_m 元素) {
        for (每个 reg_n 元素) {
            res[...] += reg_m[...] * reg_n[...];
        }
    }
}
```

这是一种 outer product（外积）式累计：

```text
取 A 在当前 k 上的一组行值
×
取 B 在当前 k 上的一组列值
=
更新一块 C 局部结果
```

### `reg_m`

```cpp
float reg_m[WMITER * TM];
```

保存当前线程需要的 A 数据。

### `reg_n`

```cpp
float reg_n[WNITER * TN];
```

保存当前线程需要的 B 数据。

### `res`

```cpp
float res[WMITER * TM * WNITER * TN] = {};
```

保存当前线程负责的所有 C 元素的累计结果。

`= {}` 表示把整个数组初始化为 0。

---

# 7. Fast Path、Edge Path 与边界处理

## 7.1 为什么需要两条路径

高性能代码常把情况分成：

```text
Fast Path：尺寸对齐、完整 Tile，可以向量化并减少判断
Edge Path：边界不完整或未对齐，需要条件检查
```

项目条件：

```cpp
const bool fast =
    (m_edge == BM) &&
    (n_edge == BN) &&
    (K % 4 == 0) &&
    (N % 4 == 0);
```

含义：

1. 当前 block 在 M 方向是完整 Tile；
2. 当前 block 在 N 方向是完整 Tile；
3. K 可被 4 整除；
4. N 可被 4 整除。

满足这些条件时，可以使用 16 字节异步搬运和 `float4` 写回。

---

## 7.2 `m_edge` 与 `n_edge`

```cpp
const int m_edge =
    (c_row * BM + BM <= M)
        ? BM
        : (M - c_row * BM);
```

例如：

```text
M = 1000
BM = 128
最后一个 block 的 c_row = 7
```

起始行：

```text
7 × 128 = 896
```

剩余：

```text
1000 - 896 = 104 行
```

所以：

```text
m_edge = 104
```

该 block 只有前 104 行有效，其余位置不能读写全局矩阵。

---

## 7.3 Edge Path 如何保证安全

加载时：

```cpp
if (mb >= m_edge) continue;
int valid = (mb + 4 <= m_edge) ? 4 : m_edge - mb;
```

只加载有效元素，越界位置在共享内存中保持 0。

写回时：

```cpp
if (grow >= M) continue;
if (gcol >= N) continue;
```

确保不会写出 C 的有效范围。

性能上，Edge Path 判断较多、向量化程度较低，所以通常比 Fast Path 慢。项目测试中的 257、511、1000、1234、2047 等非对齐尺寸，就是专门验证此类情况。

---

# 8. 双缓冲与 `cp.async`

这一部分属于进阶知识。第一次阅读只需先理解目的，再研究每条 PTX 指令。

## 8.1 没有双缓冲时

每一轮 K Tile：

```text
加载 Tile 0
等待加载完成
计算 Tile 0
加载 Tile 1
等待加载完成
计算 Tile 1
```

加载期间计算单元可能空闲，计算期间内存系统也未充分重叠。

---

## 8.2 双缓冲思想

准备两套共享内存：

```cpp
AS[2]
BS[2]
```

执行过程：

```text
先加载 Tile 0 到 buffer 0

计算 buffer 0 中的 Tile 0
同时加载 Tile 1 到 buffer 1

计算 buffer 1 中的 Tile 1
同时加载 Tile 2 到 buffer 0

继续交替
```

代码：

```cpp
int cur = t & 1;
int nxt = 1 - cur;
```

表示：

```text
当前轮使用 cur；
下一轮加载到 nxt。
```

---

## 8.3 `cp.async` 的作用

项目使用内联 PTX：

```cpp
cp.async.ca.shared.global ...
```

高层含义：

```text
从 Global Memory 异步复制到 Shared Memory，
尽量让数据搬运和计算重叠。
```

项目每次复制：

```text
16 字节 = 4 个 float
```

相关封装：

```cpp
cp_async16(...)
cp_async_commit()
cp_async_wait_group<0>()
```

可暂时理解为：

```text
cp_async16        提交一次 16 字节异步复制
commit_group      把一批复制提交为一个组
wait_group<0>     等待尚未完成的组归零
```

随后使用：

```cpp
__syncthreads();
```

保证整个 block 的线程都可以安全读取已装好的共享内存。

---

## 8.4 内联 PTX 第一遍可以跳过什么

以下代码第一遍不要求逐字符掌握：

```cpp
asm volatile(...)
cvta.to.shared
cvt.u32.u64
cp.async.ca.shared.global.L2::128B
```

你只需知道两个包装函数的语义：

```text
smem_u32addr(ptr)
    把共享内存指针转换成 PTX 指令需要的地址形式

cp_async16(dst, src, guard)
    异步搬运 16 字节 Global Memory 数据到 Shared Memory
```

等主流程看懂后，再学习 PTX 操作数约束和地址空间转换。

---

# 9. `#pragma unroll`、`__launch_bounds__` 与编译期优化

## 9.1 `#pragma unroll`

项目中大量出现：

```cpp
#pragma unroll
for (int d = 0; d < BK; ++d) {
    ...
}
```

若 BK 是编译期常量 8，编译器可以把循环近似展开成：

```cpp
body_for_d0;
body_for_d1;
...
body_for_d7;
```

潜在好处：

- 减少循环判断与跳转；
- 增加指令级并行；
- 让编译器更容易优化。

代价：

- 代码体积可能变大；
- 寄存器压力可能增加。

---

## 9.2 `__launch_bounds__`

```cpp
__global__ __launch_bounds__(Cfg::NUM_THREADS)
void gemm_warp_tile_kernel(...)
```

它告诉编译器：

```text
该 kernel 的一个 block 最多使用 Cfg::NUM_THREADS 个线程。
```

编译器可据此做寄存器分配和占用率方面的权衡。

第一遍阅读只需把它当成“提供给编译器的线程块规模优化提示”。

---

## 9.3 `-O3`、`--use_fast_math`、`-lineinfo`

`CMakeLists.txt` 中：

```cmake
-O3
--use_fast_math
-lineinfo
```

含义：

```text
-O3             高等级编译优化
--use_fast_math 使用更快但部分运算精度或 IEEE 行为可能不同的数学实现
-lineinfo        保留源码行号信息，便于 Nsight 等工具定位
```

对 GEMM 的核心 FMA 计算而言，`-O3` 和代码结构优化很关键。`-lineinfo` 不等同于完整 Debug 构建，通常仍可用于性能分析。

---

# 10. cuBLAS 对照与 Row-major/Column-major 问题

## 10.1 为什么要和 cuBLAS 比较

cuBLAS 是 NVIDIA 提供的高度优化线性代数库。

项目用它做两件事：

```text
1. 正确性参考：自定义 GEMM 结果应接近 cuBLAS
2. 性能基线：my_gflops / cublas_gflops
```

比例：

```cpp
float ratio = my_gflops / cublas_gflops;
```

例如：

```text
ratio = 0.75
```

表示自定义 kernel 吞吐约为 cuBLAS 的 75%。

---

## 10.2 cuBLAS 默认是列优先

C/C++ 项目中的 A、B、C 是 Row-major，而经典 cuBLAS 接口按 Column-major 解释矩阵。

测试代码调用：

```cpp
cublasSgemm(handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, N,
            d_A, K,
            &beta,
            d_C, N);
```

注意它把 B 放在前、A 放在后，并交换 M、N。

利用恒等式：

```text
(A × B)^T = B^T × A^T
```

把 Row-major 内存视作对应转置矩阵的 Column-major 存储，从而得到相同的内存结果。

第一遍阅读时记住：

```text
不是项目把 A、B 写反了，
而是在适配 cuBLAS 的列优先接口。
```

---

# 11. 性能测试为什么要 Warmup 和 Repeat

## 11.1 Warmup

项目先运行：

```cpp
const int warm_time = 10;
```

原因包括：

- CUDA Context 初始化；
- 首次调用库的内部初始化；
- 缓存与指令路径进入稳定状态；
- GPU 频率和功耗状态逐渐稳定。

Warmup 不计入正式性能时间。

---

## 11.2 Repeat

正式测试重复：

```cpp
const int repeat_time = 5;
```

单次 kernel 很短，计时容易受固定开销和抖动影响。重复多次后再除算吞吐，结果更稳定。

更严谨的性能实验通常还会：

- 外层重复多轮；
- 计算均值、标准差、中位数；
- 控制温度和功耗；
- 固定输入形状与比较条件；
- 分开评估对齐与非对齐尺寸。

当前仓库的 `compare_fair_runs.sh` 与稳定基线文件就在完成这类重复比较。

---

## 11.3 正确性检查

项目把 cuBLAS 和自定义结果复制到 CPU：

```cpp
if (fabsf(C_cublas[i] - C[i]) > 1e-5f) {
    error_count++;
}
```

浮点计算通常不能简单使用：

```cpp
C_cublas[i] == C[i]
```

因为不同计算顺序可能产生微小舍入误差，所以使用容差比较。

不过对更通用的测试，建议使用绝对误差与相对误差结合，例如：

```text
abs(a-b) <= atol + rtol × abs(reference)
```

当前测试输入全部为 1 和 2，结果结构较简单，因此 `1e-5` 在此测试中可工作。

---

# 12. CMake 最低限度知识

项目构建命令：

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

可以理解为：

```text
第一条：读取 CMakeLists.txt，在 build 目录生成构建系统
第二条：真正编译源文件并链接程序
```

关键配置：

```cmake
project(gemm_project LANGUAGES CXX CUDA)
```

表示项目同时使用 C++ 和 CUDA。

```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
```

表示使用 C++17/CUDA C++17。

```cmake
add_library(gemm_lib STATIC
    src/launcher.cu
    src/kernels/gemm_warp_tile.cu
)
```

把核心源码编译成静态库 `gemm_lib`。

```cmake
add_executable(test_perf_gemm
    tests/test_perf_gemm.cpp
)
```

生成测试可执行程序。

```cmake
target_link_libraries(test_perf_gemm
    PRIVATE
        gemm_lib
        CUDA::cudart
        CUDA::cublas
)
```

把自定义 GEMM、CUDA Runtime 和 cuBLAS 链接到测试程序。

```cmake
set(CMAKE_CUDA_ARCHITECTURES 89)
```

表示为指定的 GPU 架构生成 CUDA 代码。它不是一个可随意忽略的普通数字；若本机 GPU 架构不同，可能需要按实际环境修改。

---

# 13. 从一次 `test_perf_gemm` 调用串起完整流程

下面按时间顺序串联项目。

## 步骤 1：生成矩阵尺寸

```cpp
const int M = sz;
const int N = sz;
const int K = sz;
```

当前基准主要测试方阵，也包括 257、511、1000 等非对齐尺寸。

---

## 步骤 2：计算内存字节数

```cpp
size_t sizeA = static_cast<size_t>(M) * K * sizeof(float);
size_t sizeB = static_cast<size_t>(K) * N * sizeof(float);
size_t sizeC = static_cast<size_t>(M) * N * sizeof(float);
```

---

## 步骤 3：分配 Host 内存

```cpp
float* A = (float*)malloc(sizeA);
float* B = (float*)malloc(sizeB);
float* C = (float*)malloc(sizeC);
```

---

## 步骤 4：分配 Device 内存

```cpp
cudaMalloc(&d_A, sizeA);
cudaMalloc(&d_B, sizeB);
cudaMalloc(&d_C, sizeC);
```

---

## 步骤 5：初始化并复制输入

```cpp
A[i] = 1.0f;
B[i] = 2.0f;

cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
```

对方阵尺寸 `K`，理论上每个 C 元素为：

```text
1×2 累加 K 次 = 2K
```

---

## 步骤 6：运行 cuBLAS

完成 warmup、event 计时，并把结果复制到 `C_cublas`。

---

## 步骤 7：清空 d_C

```cpp
cudaMemset(d_C, 0, sizeC);
```

因为接下来要独立测试自定义 GEMM。

---

## 步骤 8：调用统一入口

```cpp
launch_gemm(M, N, K,
            alpha, d_A, d_B,
            beta, d_C);
```

默认参数：

```cpp
GemmAlgo algo = GemmAlgo::Auto
```

---

## 步骤 9：算法选择

`src/launcher.cu`：

```text
Auto
  ↓ select_gemm_algo
WarpTile
  ↓
launch_gemm_warp_tile
```

目前只有 WarpTile 实现，因此 Auto 也会选择它。

---

## 步骤 10：配置分发

`launch_gemm_warp_tile` 根据 M、N、K 选择：

```text
WarpTileIrregularConfig
WarpTileSmallConfig
WarpTileDefaultConfig
```

判断：

```cpp
small_problem
irregular_shape
irregular_medium
```

这是 host 端 dispatch：不同输入形状使用不同编译期 Tile 配置。

---

## 步骤 11：计算 Grid 和 Block

```cpp
dim3 blockDim(Cfg::NUM_THREADS);
dim3 gridDim(ceil(N/BN), ceil(M/BM));
```

---

## 步骤 12：GPU kernel 执行

每个 block：

1. 找到自己负责的 C Tile；
2. 判断完整 Tile 或边界 Tile；
3. 将 A、B 当前 K Tile 搬入共享内存；
4. 每个 warp/线程读取对应数据到寄存器；
5. 更新 `res`；
6. 遍历全部 K Tile；
7. 执行 `alpha*res + beta*C`；
8. 写回 C。

---

## 步骤 13：复制结果并校验

```cpp
cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
```

然后逐元素和 cuBLAS 结果比较。

---

## 步骤 14：计算 GFLOPS 和比例

```text
my_gflops
cublas_gflops
ratio = my_gflops / cublas_gflops
```

---

## 步骤 15：释放资源

销毁 cuBLAS handle、event，释放 Device 与 Host 内存。

---

# 14. 阅读源码的推荐顺序

不要直接从 `gemm_warp_tile.cu` 第一行开始硬啃。建议严格按下面顺序。

## 第一阶段：先看懂接口与测试

### 1. `include/gemm/types.h`

目标：

- 看懂 `enum class GemmAlgo`；
- 知道 Auto 与 WarpTile 是算法选项。

### 2. `include/gemm/launcher.h`

目标：

- 能说出 A、B、C 的角色；
- 能说出 M、N、K 对应的矩阵形状；
- 理解 `const float*` 与 `float*`。

### 3. `src/launcher.cu`

目标：

- 看懂 Auto 如何选择算法；
- 看懂 `switch` 如何分发；
- 理解这里仍是 CPU 端代码，并没有真正执行矩阵乘法。

### 4. `tests/test_perf_gemm.cpp`

目标：

- 看懂 Host/Device 内存分配；
- 看懂 H2D、D2H 复制；
- 看懂 warmup、event 计时；
- 看懂 GFLOPS；
- 看懂 cuBLAS 对照和误差检查。

达到这一阶段后，你应该能完整解释：

```text
输入矩阵从哪里来，如何进入 GPU，如何启动 GEMM，结果怎样返回和测试。
```

---

## 第二阶段：理解配置和线程层级

### 5. `include/gemm/kernels/gemm_warp_tile_config.h`

目标：

- 记住 BM、BN、BK、WM、WN、TM、TN；
- 能画出 Default 配置的 128×256 Block Tile；
- 能说明 8 个 warp 如何排列成 2×4；
- 能算出每线程保存 128 个结果。

### 6. `gemm_warp_tile.cu` 第 373～397 行附近

先看最底部 launcher：

```cpp
launch_gemm_warp_tile_cfg
launch_gemm_warp_tile
```

目标：

- 理解 Grid/Block 计算；
- 理解三个配置何时选择；
- 暂时不要从文件顶部内联 PTX 开始。

### 7. kernel 第 176～233 行附近

目标：

- 逐一代入 Default 配置；
- 看懂 block、warp、thread 坐标；
- 看懂 `AS`、`BS`、`reg_m`、`reg_n`、`res`。

---

## 第三阶段：理解计算，再理解加载

### 8. 先看 `compute` 函数

忽略异步加载，先回答：

- `d` 为什么遍历 BK；
- A 数据怎样进入 `reg_m`；
- B 数据怎样进入 `reg_n`；
- `res` 怎样做乘加。

### 9. 再看 edge path

Edge path 使用普通标量加载，逻辑更直观。先看懂：

- 共享内存清零；
- 条件加载；
- `compute_edge`；
- 条件写回。

### 10. 最后看 fast path

顺序：

```text
load_tile_async
cp_async_commit
cp_async_wait_group
双缓冲 cur/nxt
float4 写回
```

### 11. 最后再看内联 PTX

此时才研究：

```text
smem_u32addr
asm volatile
cp.async 指令格式
L2::128B hint
```

---

# 15. 你现在必须熟练、理解和暂时了解的内容

## 15.1 必须熟练掌握

这些内容不熟，源码会频繁卡住：

- 指针、取地址、解引用；
- `const float*` 与 `float*`；
- 一维数组模拟二维矩阵；
- `A[m*K+k]`、`B[k*N+n]`、`C[m*N+n]`；
- `size_t`、`sizeof`、`static_cast`；
- 函数声明与实现分离；
- Host、Device、Global Memory；
- `cudaMalloc/cudaMemcpy/cudaMemset/cudaFree`；
- kernel 启动语法；
- Grid、Block、Thread；
- `blockIdx/threadIdx`；
- warp 固定 32 线程；
- `__shared__` 与 `__syncthreads()`；
- GEMM 公式和 `2MNK`；
- Tile 的数据复用思想。

## 15.2 必须理解，但不要求立刻手写

- 模板参数和 `static constexpr`；
- Block Tile、Warp Tile、Thread Tile；
- 寄存器分块；
- 合并访存；
- fast path/edge path；
- CUDA Event 计时；
- cuBLAS Row-major 适配；
- `float4` 向量化读写；
- 双缓冲。

## 15.3 第一遍只需要知道作用

- 内联 PTX 语法；
- `cvta.to.shared`；
- `cp.async` 的完整硬件细节；
- `__launch_bounds__` 对 occupancy 的精确影响；
- 寄存器分配、bank conflict 的定量分析；
- L1/L2/DRAM microbenchmark 的汇编实现；
- Nsight Compute 中每个硬件指标的细节。

---

# 16. 从零练习：先写三个最小程序

## 练习 1：CPU 矩阵索引

写一个 2×3 矩阵并打印每个元素：

```cpp
#include <iostream>

int main() {
    const int rows = 2;
    const int cols = 3;

    float A[rows * cols] = {
        1, 2, 3,
        4, 5, 6
    };

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            std::cout << A[row * cols + col] << " ";
        }
        std::cout << "\n";
    }
}
```

必须能解释：

```text
A[1*3+2] 为什么等于 6。
```

---

## 练习 2：CPU GEMM

手写三重循环版本，使用小矩阵和纸面结果比较。

检查点：

- A 是 M×K；
- B 是 K×N；
- C 是 M×N；
- 内层循环遍历 K。

---

## 练习 3：向量加法 CUDA 程序

在开始 GEMM 前，至少要独立看懂如下模式：

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void add_kernel(const float* A,
                           const float* B,
                           float* C,
                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int n = 1024;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float* h_A = new float[n];
    float* h_B = new float[n];
    float* h_C = new float[n];

    for (int i = 0; i < n; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    const int block = 256;
    const int grid = (n + block - 1) / block;

    add_kernel<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    std::cout << h_C[0] << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}
```

能完全解释这个程序后，再进入 GEMM kernel 会顺畅很多。

---

# 17. 自测题与答案

## 题 1

A 是 M×K 矩阵，A 第 m 行第 k 列的一维下标是什么？

答案：

```cpp
m * K + k
```

---

## 题 2

`const float* A` 能否执行 `A[0] = 1.0f`？

答案：不能。A 指向的数据通过该指针只读。

---

## 题 3

`cudaMalloc(&d_A, bytes)` 为什么传 `&d_A`？

答案：`cudaMalloc` 需要把新分配的 GPU 地址写入指针变量 `d_A`，因此需要获得 `d_A` 自身的地址。

---

## 题 4

`cudaMemset(d_C, 1, bytes)` 是否会把每个 float 设置为 1.0？

答案：不会。`cudaMemset` 按字节填充，1 表示每字节写入 `0x01`。

---

## 题 5

`blockIdx.x` 和 `threadIdx.x` 有什么区别？

答案：前者是线程块编号，后者是线程在线程块内的编号。

---

## 题 6

256 个线程包含多少个 warp？

答案：

```text
256 / 32 = 8 个 warp
```

---

## 题 7

Default 配置中一个 block 计算多大的 C Tile？

答案：

```text
128 × 256
```

---

## 题 8

Default 配置中 8 个 warp 如何排列？

答案：

```text
2 行 × 4 列，每个 warp 负责 64×64。
```

---

## 题 9

为什么需要 `__syncthreads()`？

答案：保证同一个 block 的所有线程完成共享内存写入后，再开始读取该共享内存。

---

## 题 10

GEMM 的近似浮点运算量为什么是 `2MNK`？

答案：M×N 个输出，每个输出沿 K 维执行 K 次乘法和 K 次左右的加法，按 FMA 计为 2 FLOPs。

---

## 题 11

为什么非对齐矩阵通常性能较差？

答案：边界 Tile 无法完全使用固定大小向量化加载和写回，需要额外条件判断、清零和标量处理，数据利用率也更低。

---

## 题 12

双缓冲解决什么问题？

答案：尝试让下一块 A/B 数据的加载和当前块的计算重叠，减少等待内存的空闲时间。

---

# 18. 阅读 `gemm_warp_tile.cu` 时每段代码该问什么

## 看到指针偏移时

```cpp
A + offset
```

问：

```text
offset 的单位是字节还是元素？
它把指针移动到矩阵的哪一行、哪一列？
```

对 `float*` 来说，offset 单位是 float 元素。

## 看到线程索引时

```cpp
threadIdx.x / 32
threadIdx.x % 32
```

问：

```text
这是在求 warp 编号，还是 lane 编号？
```

## 看到模板常量时

```cpp
Cfg::BM
Cfg::WN
```

问：

```text
这是 Block Tile、Warp Tile 还是 Thread Tile 的尺寸？
```

## 看到共享内存索引时

```cpp
AS[d * A_STRIDE + row]
```

问：

```text
AS 的逻辑形状是什么？
d 和 row 分别是哪一个维度？
```

## 看到 `res[...] +=` 时

问：

```text
这个 res 元素对应 C 的哪一个全局行列？
当前乘法使用的是哪个 k？
```

## 看到同步时

```cpp
__syncthreads();
```

问：

```text
同步前谁在写共享内存？
同步后谁要读取这些数据？
```

## 看到 fast/edge 分支时

问：

```text
fast path 依赖哪些对齐与边界条件？
edge path 用什么方式避免越界？
```

---

# 19. 常见误区

## 误区 1：一个线程只计算一个 C 元素

这只适用于 naïve GEMM。当前项目每线程计算多个输出，并将它们保存在 `res` 数组中。

## 误区 2：一个 block 对应矩阵的一行

当前项目一个 block 对应的是二维 C Tile：`BM×BN`。

## 误区 3：Shared Memory 是所有 GPU 线程共享

错误。Shared Memory 只在同一个 block 内共享。

## 误区 4：`__syncthreads()` 能同步所有 block

错误。它只能同步当前 block，不能进行 Grid 全局同步。

## 误区 5：kernel 调用返回就表示计算完成

错误。kernel 启动通常异步，需要 event 或同步 API 确认完成。

## 误区 6：Tile 越大一定越快

错误。Tile 增大会增加共享内存和寄存器使用，可能降低 occupancy，甚至超出资源限制。最佳配置依赖 GPU 架构与矩阵形状。

## 误区 7：`float4` 自动让程序更快

错误。只有地址对齐、连续访问、边界合法且线程映射合理时，向量化访问才有价值。

## 误区 8：cuBLAS 调用中的 B、A 顺序说明项目数学公式是 B×A

错误。这是在适配 cuBLAS 的 Column-major 接口，最终内存结果仍对应 Row-major 的 A×B。

---

# 20. 最终验收标准

在正式逐行阅读项目之前，你应该能够不看资料回答以下问题：

1. A、B、C 的形状分别是什么？
2. `A[m*K+k]` 为什么能访问二维矩阵元素？
3. `const float* A` 和 `float* C` 有何区别？
4. `sizeA` 为什么是 `M*K*sizeof(float)`？
5. Host 指针和 Device 指针有什么区别？
6. `cudaMalloc`、`cudaMemcpy`、`cudaMemset`、`cudaFree` 各做什么？
7. kernel 的 `<<<gridDim, blockDim>>>` 表示什么？
8. `blockIdx`、`threadIdx`、warp 分别是什么？
9. 为什么 warp 是 32 个线程？源码如何计算 warp 编号和 lane 编号？
10. Shared Memory 为什么能提高 GEMM 数据复用？
11. `__syncthreads()` 在共享内存加载后为什么必要？
12. BM、BN、BK、WM、WN、TM、TN 分别是哪一级 Tile？
13. Default 配置中的 8 个 warp 如何覆盖 128×256 区域？
14. 每个线程为什么会有一个较大的 `res` 数组？
15. 为什么 GEMM 运算量按 `2MNK` 计算？
16. CUDA Event 为什么比普通 CPU 计时更适合测 kernel？
17. fast path 和 edge path 的区别是什么？
18. 双缓冲和 `cp.async` 想隐藏哪一种开销？
19. `reinterpret_cast<float4*>` 在这里表示什么？
20. `test_perf_gemm.cpp → launch_gemm → launch_gemm_warp_tile → kernel` 的调用链是什么？

当你能回答前 12 个问题时，可以开始阅读核心 kernel；当 20 个问题都能回答时，已经具备完整理解当前项目的基础。

---

# 21. 一页速查表

```text
矩阵形状：
A[M,K] × B[K,N] = C[M,N]

Row-major：
A[m,k] → A[m*K+k]
B[k,n] → B[k*N+n]
C[m,n] → C[m*N+n]

GEMM：
C = alpha*A*B + beta*C
FLOPs ≈ 2*M*N*K

CUDA 生命周期：
Host 初始化
→ cudaMalloc
→ cudaMemcpy H2D
→ kernel<<<grid,block>>>
→ 同步/Event
→ cudaMemcpy D2H
→ cudaFree

线程层级：
Grid → Block → Warp(32 threads) → Thread

内存层级：
Global → Shared → Register → C Global

Default Tile：
Block Tile  = 128×256
K Tile      = 8
Warp Tile   = 64×64
Threads     = 256 = 8 warps
Warp 排列   = 2×4
每线程 res  = 128 floats

核心索引：
warp_idx = threadIdx.x / 32
lane     = threadIdx.x % 32
block row tile = blockIdx.y
block col tile = blockIdx.x

grid.x = ceil(N/BN)
grid.y = ceil(M/BM)

同步：
__syncthreads() 只同步当前 block

性能：
GFLOPS = repeat*2MNK/(time_ms*1e6)
ratio  = my_gflops/cublas_gflops
```



%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-gemm算子|模块-gemm算子]]

%% 项目关联导航：结束 %%
