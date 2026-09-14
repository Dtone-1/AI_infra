# gemm_gemm_warp_tile_精读

本文严格按照 `gemm_warp_tile.cu` 的代码顺序进行拆分。每一节先给出一块完整源码，再解释其中的 C++/CUDA 语法、模板参数、线程映射、共享内存、寄存器和内联 PTX 指令。

---

## 1. 头文件与宏定义

```cpp
#include <cstdint>
#include <cuda_runtime.h>
#include "gemm/kernels/gemm_warp_tile.h"
#include "gemm/kernels/gemm_warp_tile_config.h"

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define WARP_SIZE 32
```

### 详细解释

```cpp
#include <cstdint>
```

引入固定宽度整数类型，例如：

```cpp
uint32_t
```

`uint32_t` 表示恰好占 32 bit 的无符号整数。后面代码用它保存共享内存地址。

```cpp
#include <cuda_runtime.h>
```

引入 CUDA Runtime 中的类型和内置变量，例如：

```cpp
dim3
threadIdx
blockIdx
__syncthreads
float4
```

```cpp
#include "gemm/kernels/gemm_warp_tile.h"
```

引入 `launch_gemm_warp_tile` 的函数声明。

```cpp
#include "gemm/kernels/gemm_warp_tile_config.h"
```

引入三组配置类型：

```cpp
WarpTileDefaultConfig
WarpTileSmallConfig
WarpTileIrregularConfig
```

这些类型保存 `BM、BN、BK、WM、WN、TM、TN、NUM_THREADS` 等编译期常量。

---

### `CEIL_DIV`

```cpp
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
```

这是向上取整除法宏。

普通整数除法会向下取整：

```cpp
100 / 64 == 1
```

但覆盖 100 个元素，每块处理 64 个元素，需要 2 块，因此使用：

```cpp
CEIL_DIV(100, 64)
= (100 + 64 - 1) / 64
= 163 / 64
= 2
```

宏参数都使用括号包围，是为了避免运算优先级问题。

---

### `WARP_SIZE`

```cpp
#define WARP_SIZE 32
```

NVIDIA GPU 中一个 warp 包含 32 个线程。

后面会使用：

```cpp
threadIdx.x / WARP_SIZE
```

计算线程属于第几个 warp，以及：

```cpp
threadIdx.x % WARP_SIZE
```

计算线程在 warp 内的编号。

---

## 2. 将普通指针转换为共享内存地址

```cpp
__device__ __forceinline__ uint32_t smem_u32addr(const void *ptr) {
    uint32_t addr;
    asm("{.reg .u64 u64addr;\n"
        " cvta.to.shared.u64 u64addr, %1;\n"
        " cvt.u32.u64 %0, u64addr;}\n"
        : "=r"(addr) : "l"(ptr));
    return addr;
}
```

### 详细解释

这个函数把一个普通 C++ 指针形式的共享内存地址，转换为 `cp.async` 指令需要的 32 位 shared-memory 地址。

---

### CUDA 函数修饰符

```cpp
__device__
```

表示该函数：

- 在 GPU 上执行；
- 只能由 GPU 代码调用；
- 不能直接由 CPU 调用。

```cpp
__forceinline__
```

要求编译器尽量强制内联该函数。

普通 `inline` 只是建议，而 `__forceinline__` 的要求更强。此函数只有几条指令，内联后可以避免 GPU 函数调用开销。

---

### 参数类型

```cpp
const void *ptr
```

`void*` 是无具体元素类型的通用指针。

`const` 表示函数不会通过该指针修改内存内容。

之所以使用 `void*`，是因为这里只关心地址本身，不关心它指向 `float`、`int` 还是其他类型。

---

### 局部变量

```cpp
uint32_t addr;
```

声明一个 32 位无符号整数，用于接收转换后的共享内存地址。

---

### 内联 PTX

```cpp
asm(...)
```

表示在 C++/CUDA 代码中直接嵌入 PTX 汇编。

PTX 是 NVIDIA GPU 的虚拟指令集，位于 CUDA C++ 和最终机器指令 SASS 之间。

字符串中的：

```cpp
\n
```

表示换行，让多条 PTX 指令依次排列。

---

### 声明 PTX 寄存器

```ptx
.reg .u64 u64addr;
```

含义是：

- `.reg`：声明寄存器；
- `.u64`：寄存器保存 64 位无符号整数；
- `u64addr`：寄存器名字。

---

### 地址空间转换

```ptx
cvta.to.shared.u64 u64addr, %1;
```

`cvta.to.shared` 将通用地址转换为共享内存地址空间中的地址。

这里：

```text
%1
```

对应输入操作数 `ptr`。

结果保存到 64 位寄存器：

```text
u64addr
```

---

### 转为 32 位

```ptx
cvt.u32.u64 %0, u64addr;
```

把 64 位地址转换为 32 位无符号整数，并写入 `%0`。

`cp.async` 的 shared-memory 目标地址操作数通常使用 32 位地址形式。

---

### 汇编操作数约束

```cpp
: "=r"(addr) : "l"(ptr)
```

冒号前后分别表示输出操作数和输入操作数。

```cpp
"=r"(addr)
```

含义：

- `=`：该操作数由汇编代码写入；
- `r`：使用普通 32 位寄存器；
- `(addr)`：最终写回 C++ 变量 `addr`。

```cpp
"l"(ptr)
```

含义：

- `l`：使用 64 位寄存器；
- `(ptr)`：输入值来自 C++ 指针 `ptr`。

因此：

```text
%0 → addr
%1 → ptr
```

最后：

```cpp
return addr;
```

把转换后的 32 位共享内存地址返回给调用者。

---

## 3. 异步复制 16 字节

```cpp
// cp.async.ca 16 bytes with L2::128B hint and predicate
__device__ __forceinline__ void cp_async16(uint32_t dst, const void *src, bool guard) {
    asm volatile(
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p cp.async.ca.shared.global.L2::128B [%0], [%1], 16;}\n"
        :: "r"(dst), "l"(src), "r"((int)guard));
}
```

### 详细解释

该函数使用 PTX 指令：

```ptx
cp.async
```

把 16 字节数据从全局内存异步搬运到共享内存。

16 字节恰好等于：

```text
4 个 float × 4 字节 = 16 字节
```

---

### 参数

```cpp
uint32_t dst
```

共享内存目标地址，已经由 `smem_u32addr` 转成 32 位形式。

```cpp
const void *src
```

全局内存源地址。

```cpp
bool guard
```

谓词条件。为 `true` 时执行复制，为 `false` 时不执行。

---

### `asm volatile`

```cpp
asm volatile(...)
```

`volatile` 告诉编译器：

> 不要因为看不到普通 C++ 输出变量，就删除或随意重排这段汇编。

异步复制会改变共享内存内容，因此必须保留。

---

### PTX 谓词寄存器

```ptx
.reg .pred p;
```

声明一个 predicate 寄存器 `p`。

谓词寄存器只保存真假条件，类似 C++ 中的 `bool`。

```ptx
setp.ne.b32 p, %2, 0;
```

含义是：

```text
如果 %2 不等于 0，则 p=true，否则 p=false。
```

`%2` 对应：

```cpp
(int)guard
```

代码把 `bool` 显式转换成 `int`，再传给 PTX。

---

### 带谓词执行

```ptx
@p cp.async.ca.shared.global.L2::128B [%0], [%1], 16;
```

前面的：

```ptx
@p
```

表示只有当 `p` 为真时才执行该指令。

各部分含义：

- `cp.async`：异步复制；
- `.ca`：cache at all levels 的缓存策略；
- `.shared.global`：从 global memory 复制到 shared memory；
- `L2::128B`：提供 128 字节 L2 预取提示；
- `[%0]`：共享内存目标地址；
- `[%1]`：全局内存源地址；
- `16`：复制 16 字节。

---

### 输入约束

```cpp
:: "r"(dst), "l"(src), "r"((int)guard)
```

这里没有普通 C++ 输出操作数，所以两个冒号前为空。

三个输入依次映射为：

```text
%0 → dst，32 位寄存器
%1 → src，64 位寄存器
%2 → guard，32 位寄存器
```

---

### 注意

当前快速路径调用该函数时传入：

```cpp
true
```

所以实际总会执行复制。

若传入 `false`，这里只是跳过 `cp.async`，并不会自动把目标共享内存清零。因此使用谓词关闭复制时，调用者必须明确处理目标位置的旧数据。

---

## 4. 提交异步复制组

```cpp
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n"::);
}
```

### 详细解释

```ptx
cp.async.commit_group;
```

把此前当前线程发出的、尚未提交的 `cp.async` 操作组成一组并提交。

可以把流程理解为：

```text
发出多个 cp.async
        ↓
cp.async.commit_group
        ↓
这些复制形成一个异步操作组
```

函数返回类型是 `void`，因为它只发出一条 GPU 指令，不返回普通值。

末尾：

```cpp
::
```

表示该内联汇编没有显式输出操作数，也没有显式输入操作数。

---

## 5. 等待异步复制组

```cpp
template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n":: "n"(N));
}
```

### 详细解释

这是一个函数模板。

```cpp
template<int N>
```

表示 `N` 是编译期整数模板参数。

调用方式是：

```cpp
cp_async_wait_group<0>();
```

编译时会生成 `N=0` 的具体版本。

---

### PTX 指令

```ptx
cp.async.wait_group N;
```

等待异步复制组，使尚未完成的组数量不超过 N。

当前代码使用：

```cpp
cp_async_wait_group<0>();
```

表示等待所有更早提交的异步复制组完成。

---

### `"n"(N)`

```cpp
"n"(N)
```

`n` 表示把模板常量作为编译期立即数嵌入 PTX 指令，而不是放入运行时寄存器。

因此生成的 PTX 类似：

```ptx
cp.async.wait_group 0;
```

而不是运行时读取变量。

---

## 6. 完整 BK tile 的计算函数

```cpp
template<int BN, int BM, int BK, int WN, int WM,
         int WNITER, int WMITER, int TN, int TM, int WSUBM, int WSUBN, int A_STRIDE>
__device__ __forceinline__ void compute(
    float *reg_m, float *reg_n,
    const float *AS, const float *BS,
    uint trow, uint tcol, uint wrow, uint wcol, float *res)
{
    #pragma unroll
    for (int d = 0; d < BK; d++) {
        #pragma unroll
        for (uint wr = 0; wr < WMITER; wr++)
            #pragma unroll
            for (uint rm = 0; rm < TM; rm++)
                reg_m[wr*TM+rm] = AS[d*A_STRIDE + (wrow*WM + wr*WSUBM + trow*TM + rm)];
        #pragma unroll
        for (uint wc = 0; wc < WNITER; wc++)
            #pragma unroll
            for (uint rn = 0; rn < TN; rn++)
                reg_n[wc*TN+rn] = BS[d*BN + wcol*WN + wc*WSUBN + tcol*TN + rn];
        #pragma unroll
        for (uint wr = 0; wr < WMITER; wr++)
            #pragma unroll
            for (uint wc = 0; wc < WNITER; wc++)
                #pragma unroll
                for (uint rm = 0; rm < TM; rm++)
                    #pragma unroll
                    for (uint rn = 0; rn < TN; rn++)
                        res[(wr*TM+rm)*(WNITER*TN) + wc*TN+rn] +=
                            reg_m[wr*TM+rm] * reg_n[wc*TN+rn];
    }
}
```

### 详细解释

该函数由 GPU 线程调用，负责使用共享内存中的一个完整 `BK` tile，对当前线程负责的结果寄存器进行累加。

---

### 模板参数

```cpp
template<int BN, int BM, int BK, ...>
```

这些值在编译期确定。

主要含义：

- `BM`：一个线程块处理 C 的行数；
- `BN`：一个线程块处理 C 的列数；
- `BK`：一次沿 K 维处理的深度；
- `WM`、`WN`：一个 warp 负责的 C 子块大小；
- `WMITER`、`WNITER`：warp 在 M/N 方向分几轮；
- `TM`、`TN`：一个线程每个子轮次负责的行列数量；
- `WSUBM`、`WSUBN`：warp 每个子轮次处理的尺寸；
- `A_STRIDE`：共享内存 AS 每个 K 层的跨度。

使用模板常量后，编译器可以展开循环、直接计算数组下标并进行更强优化。

---

### 参数

```cpp
float *reg_m, float *reg_n
```

线程私有的寄存器数组：

- `reg_m`：暂存 A 方向的数据；
- `reg_n`：暂存 B 方向的数据。

虽然语法是指针，调用时传入线程局部数组，编译器通常会把它们放在寄存器中，寄存器压力过大时也可能溢出到 local memory。

```cpp
const float *AS, const float *BS
```

指向共享内存中的 A tile 和 B tile。

```cpp
uint trow, uint tcol
```

当前线程在一个 warp 子块中的行列编号。

```cpp
uint wrow, uint wcol
```

当前 warp 在线程块 tile 中的行列编号。

```cpp
float *res
```

线程私有结果数组，保存多个输出元素的累加值。

---

### 沿 K 维循环

```cpp
for (int d = 0; d < BK; d++)
```

对当前共享内存 tile 的每个 K 位置依次执行外积累加。

对于某个 `d`：

1. 从 AS 读取当前线程需要的 A 元素；
2. 从 BS 读取当前线程需要的 B 元素；
3. 两两相乘，累加到 `res`。

---

### `#pragma unroll`

```cpp
#pragma unroll
```

要求编译器尽量展开紧随其后的循环。

例如：

```cpp
for (int i = 0; i < 4; i++) {
    x[i] += 1;
}
```

可能被展开成：

```cpp
x[0] += 1;
x[1] += 1;
x[2] += 1;
x[3] += 1;
```

好处是减少循环判断和跳转，并让编译器更容易安排指令；代价是生成的指令数量增加。

因为这里的循环上限大多是编译期模板常量，所以适合展开。

---

### 读取 A 数据

```cpp
reg_m[wr*TM+rm] =
    AS[d*A_STRIDE +
       (wrow*WM + wr*WSUBM + trow*TM + rm)];
```

共享内存 AS 的逻辑布局是：

```text
AS[BK][A_STRIDE]
```

第一维是 K 方向 `d`，第二维是 M 方向。

M 坐标由四部分组成：

```text
wrow * WM
```

当前 warp tile 在 M 方向的起点。

```text
wr * WSUBM
```

warp 在 M 方向第 `wr` 个子轮次的偏移。

```text
trow * TM
```

当前线程在子块中的起始行。

```text
rm
```

线程自己负责的第 `rm` 行。

---

### 读取 B 数据

```cpp
reg_n[wc*TN+rn] =
    BS[d*BN +
       wcol*WN + wc*WSUBN + tcol*TN + rn];
```

共享内存 BS 的逻辑布局是：

```text
BS[BK][BN]
```

N 坐标由：

```text
wcol*WN
+ wc*WSUBN
+ tcol*TN
+ rn
```

共同确定。

---

### 外积累加

```cpp
res[...] += reg_m[...] * reg_n[...];
```

对当前 `d`：

- `reg_m` 中每个 A 值；
- 与 `reg_n` 中每个 B 值；

两两相乘，形成一个小型外积，并累加到线程负责的所有 C 元素。

结果下标：

```cpp
(wr*TM+rm)*(WNITER*TN) + wc*TN+rn
```

把二维线程结果坐标压平成一维数组。

行坐标是：

```text
wr*TM + rm
```

列坐标是：

```text
wc*TN + rn
```

每行宽度是：

```text
WNITER*TN
```

---

## 7. 只计算部分 K 的函数

```cpp
template<int BN, int BM, int BK, int WN, int WM,
         int WNITER, int WMITER, int TN, int TM, int WSUBM, int WSUBN, int A_STRIDE>
__device__ __forceinline__ void compute_k(
    float *reg_m, float *reg_n,
    const float *AS, const float *BS,
    uint trow, uint tcol, uint wrow, uint wcol, float *res, int k_lim)
{
    for (int d = 0; d < k_lim; d++) {
        #pragma unroll
        for (uint wr = 0; wr < WMITER; wr++)
            #pragma unroll
            for (uint rm = 0; rm < TM; rm++)
                reg_m[wr*TM+rm] = AS[d*A_STRIDE + (wrow*WM + wr*WSUBM + trow*TM + rm)];
        #pragma unroll
        for (uint wc = 0; wc < WNITER; wc++)
            #pragma unroll
            for (uint rn = 0; rn < TN; rn++)
                reg_n[wc*TN+rn] = BS[d*BN + wcol*WN + wc*WSUBN + tcol*TN + rn];
        #pragma unroll
        for (uint wr = 0; wr < WMITER; wr++)
            #pragma unroll
            for (uint wc = 0; wc < WNITER; wc++)
                #pragma unroll
                for (uint rm = 0; rm < TM; rm++)
                    #pragma unroll
                    for (uint rn = 0; rn < TN; rn++)
                        res[(wr*TM+rm)*(WNITER*TN) + wc*TN+rn] +=
                            reg_m[wr*TM+rm] * reg_n[wc*TN+rn];
    }
}
```

### 详细解释

`compute_k` 与 `compute` 的内部计算基本相同。

主要区别是 K 循环上限：

```cpp
compute:
for (int d = 0; d < BK; d++)
```

```cpp
compute_k:
for (int d = 0; d < k_lim; d++)
```

`BK` 是编译期固定值，而 `k_lim` 是运行时参数。

当：

```text
K 不能被 BK 整除
```

最后一个 K tile 中只有部分位置有效，例如：

```text
BK = 8
K 剩余 4
```

此时调用：

```cpp
compute_k(..., 4)
```

只计算 `d=0,1,2,3`，不会处理后面无效的四层。

因为 `k_lim` 是运行时变量，最外层循环没有标记 `#pragma unroll`；内部固定小循环仍然展开。

---

## 8. 边界 tile 的计算函数

```cpp
template<int BN, int BM, int BK, int WN, int WM,
         int WNITER, int WMITER, int TN, int TM, int WSUBM, int WSUBN, int A_STRIDE>
__device__ __forceinline__ void compute_edge(
    float *reg_m, float *reg_n,
    const float *AS, const float *BS,
    uint trow, uint tcol, uint wrow, uint wcol,
    float *res, int k_lim, int m_edge, int n_edge)
{
    for (int d = 0; d < k_lim; d++) {
        #pragma unroll
        for (uint wr = 0; wr < WMITER; wr++)
            #pragma unroll
            for (uint rm = 0; rm < TM; rm++) {
                int row = wrow*WM + wr*WSUBM + trow*TM + rm;
                reg_m[wr*TM+rm] = (row < m_edge) ? AS[d*A_STRIDE + row] : 0.f;
            }
        #pragma unroll
        for (uint wc = 0; wc < WNITER; wc++)
            #pragma unroll
            for (uint rn = 0; rn < TN; rn++) {
                int col = wcol*WN + wc*WSUBN + tcol*TN + rn;
                reg_n[wc*TN+rn] = (col < n_edge) ? BS[d*BN + col] : 0.f;
            }
        #pragma unroll
        for (uint wr = 0; wr < WMITER; wr++)
            #pragma unroll
            for (uint wc = 0; wc < WNITER; wc++)
                #pragma unroll
                for (uint rm = 0; rm < TM; rm++)
                    #pragma unroll
                    for (uint rn = 0; rn < TN; rn++)
                        res[(wr*TM+rm)*(WNITER*TN) + wc*TN+rn] +=
                            reg_m[wr*TM+rm] * reg_n[wc*TN+rn];
    }
}
```

### 详细解释

该函数用于矩阵 M/N 边界处的不完整 tile。

例如：

```text
BM = 64
M = 100
```

M 方向需要两个线程块：

```text
第 0 块：处理 0～63，共 64 行
第 1 块：处理 64～99，只有 36 行有效
```

第二块中的：

```cpp
m_edge = 36
```

---

### 三元运算符

```cpp
(row < m_edge) ? AS[...] : 0.f
```

语法：

```cpp
条件 ? 条件为真时的值 : 条件为假时的值
```

如果当前线程读取的行有效，就从 AS 读取；否则使用 `0.0f`。

B 方向同理：

```cpp
(col < n_edge) ? BS[...] : 0.f
```

无效输入设为 0 后，它参与乘法也不会改变结果。

---

### 为什么仍需要边界判断

边界路径在装载共享内存时已经先清零，再只加载有效位置。

这里再次检查 `row` 和 `col`，属于额外的防御性保护，确保线程不会把超出有效 M/N 范围的共享内存值用于计算。

---


## 9. 异步加载一个 A/B tile

```cpp
template<int BN, int BM, int BK, int NUM_THREADS, int A_STRIDE>
__device__ __forceinline__ void load_tile_async(
    float *AS, float *BS,
    const float *A_tile, const float *B_tile, int K, int N,
    uint inner_row_a, uint inner_col_a,
    uint inner_row_b, uint inner_col_b)
{
    // A: [BM][BK], each thread loads 4 consecutive M elements for BK/row_stride_a K-rows
    // Mapping: inner_row_a = k-index group, inner_col_a = m-index (4 consecutive)
    // row_stride_a = NUM_THREADS / (BM/4)
    constexpr int row_stride_a = NUM_THREADS / (BM / 4);
    constexpr int row_stride_b = NUM_THREADS / (BN / 4);

    #pragma unroll
    for (uint i = 0; i < BK; i += row_stride_a) {
        uint k = inner_row_a + i;
        uint32_t dst = smem_u32addr(&AS[inner_col_a + k*A_STRIDE]);
        cp_async16(dst, A_tile + k*K + inner_col_a, true);
    }

    #pragma unroll
    for (uint i = 0; i < BK; i += row_stride_b) {
        uint32_t dst = smem_u32addr(&BS[(inner_row_b+i)*BN + inner_col_b]);
        cp_async16(dst, B_tile + (inner_row_b+i)*N + inner_col_b, true);
    }
    cp_async_commit();
}
```

### 详细解释

该函数让整个线程块合作，把一个 A tile 和一个 B tile 从全局内存异步复制到共享内存。

目标布局是：

```text
AS：逻辑形状 [BK][BM]
BS：逻辑形状 [BK][BN]
```

其中：

```cpp
A_STRIDE = BM
```

因此 AS 的下标：

```cpp
AS[k*A_STRIDE + m]
```

表示第 `k` 层、第 `m` 行元素。

---

### 编译期步长

```cpp
constexpr int row_stride_a = NUM_THREADS / (BM / 4);
constexpr int row_stride_b = NUM_THREADS / (BN / 4);
```

`constexpr` 表示编译期常量。

每个线程一次加载 4 个 `float`，即 16 字节。

A 的 M 方向被分成：

```text
BM / 4
```

个向量加载单元。

如果线程总数大于 `BM/4`，多出的线程会被分配到不同 K 层。

例如默认配置：

```text
BM = 128
NUM_THREADS = 256
BM/4 = 32
row_stride_a = 256/32 = 8
```

每个 K tile 的深度 `BK=8`，256 个线程刚好覆盖：

```text
8 个 K 层 × 每层 32 个 float4
```

---

### A 的共享内存目标地址

```cpp
uint k = inner_row_a + i;
```

得到当前线程负责的 K 坐标。

```cpp
uint32_t dst =
    smem_u32addr(&AS[inner_col_a + k*A_STRIDE]);
```

共享内存目标起点是：

```text
AS[k][inner_col_a]
```

一次复制 4 个相邻 `float`，因此会写入：

```text
AS[k][inner_col_a + 0]
AS[k][inner_col_a + 1]
AS[k][inner_col_a + 2]
AS[k][inner_col_a + 3]
```

---

### A 的全局内存源地址

```cpp
A_tile + k*K + inner_col_a
```

这行代码按 C/C++ 行主序下标解释为：

```text
A_tile[k][inner_col_a]
```

并从该位置连续读取 4 个 `float`。

但本文件其他路径对 A 的访问是：

```cpp
A_tile[m*K + k]
```

也就是固定 K 坐标、沿 M 方向取值后转置写入 AS。

因此需要特别注意：

> 当前快速异步路径的 A 源地址表达式，与尾部路径和边界路径的 A 行主序访问方式不一致。

如果 A 的实际布局是普通行主序 `[M][K]`，并且没有提前转置，那么共享内存目标：

```text
AS[k][m]
```

通常应接收：

```text
A[m][k]
```

而不是：

```text
A[k][m]
```

测试文件把 A 的全部元素都初始化为 1，这种输入无法发现转置或索引错误，因为无论读取哪个位置都是 1。

这不是语法问题，而是阅读该行时必须核对的数据布局问题。

---

### B 的异步加载

```cpp
uint32_t dst =
    smem_u32addr(&BS[(inner_row_b+i)*BN + inner_col_b]);
```

目标是：

```text
BS[k][n]
```

源地址：

```cpp
B_tile + (inner_row_b+i)*N + inner_col_b
```

对应普通行主序：

```text
B[k][n]
```

一次复制 4 个连续 N 方向元素。

B 的全局内存和共享内存都沿 N 方向连续，因此这里能自然使用 16 字节连续复制。

---

### 提交同一复制组

A 和 B 的多个 `cp.async` 发出后，统一执行：

```cpp
cp_async_commit();
```

所以这些复制操作被提交为同一个异步组。

稍后通过：

```cpp
cp_async_wait_group<0>();
```

等待该组完成。

---

## 10. CUDA Kernel 模板与启动约束

```cpp
template <typename Cfg>
__global__ __launch_bounds__(Cfg::NUM_THREADS)
void gemm_warp_tile_kernel(int M, int N, int K,
                           float alpha, const float *A, const float *B,
                           float beta, float *C)
{
```

### 详细解释

这是实际由 GPU 执行的 CUDA Kernel。

---

### 类型模板参数

```cpp
template <typename Cfg>
```

`Cfg` 是一个配置类型，而不是整数。

调用时可能实例化为：

```cpp
gemm_warp_tile_kernel<WarpTileDefaultConfig>
gemm_warp_tile_kernel<WarpTileSmallConfig>
gemm_warp_tile_kernel<WarpTileIrregularConfig>
```

每个配置类型内部都定义：

```cpp
static constexpr int BM = ...;
static constexpr int BN = ...;
```

因此同一份 Kernel 源码可以在编译期生成多个不同 tile 参数版本。

---

### `__global__`

```cpp
__global__
```

表示这是 CUDA Kernel：

- 由 CPU 端启动；
- 在 GPU 上执行；
- 使用 `<<<gridDim, blockDim>>>` 语法调用；
- 返回类型必须是 `void`。

---

### `__launch_bounds__`

```cpp
__launch_bounds__(Cfg::NUM_THREADS)
```

告诉编译器：

```text
该 Kernel 每个线程块最多会使用 Cfg::NUM_THREADS 个线程。
```

编译器可以根据这个信息进行寄存器分配和占用率优化。

它不会自动设置线程块大小；真正启动时仍需：

```cpp
dim3 blockDim(Cfg::NUM_THREADS);
```

如果实际启动线程数超过这里声明的上限，行为不符合该编译假设。

---

### Kernel 参数

参数表达标准 GEMM：

```text
C = alpha × A × B + beta × C
```

矩阵形状：

```text
A：M×K
B：K×N
C：M×N
```

`A` 和 `B` 是只读设备指针，`C` 是可写设备指针。

---

## 11. 从配置类型取出编译期常量

```cpp
    constexpr int NUM_THREADS = Cfg::NUM_THREADS;
    constexpr int BN = Cfg::BN;
    constexpr int BM = Cfg::BM;
    constexpr int BK = Cfg::BK;
    constexpr int WN = Cfg::WN;
    constexpr int WM = Cfg::WM;
    constexpr int WNITER = Cfg::WNITER;
    constexpr int TN = Cfg::TN;
    constexpr int TM = Cfg::TM;
    constexpr int A_STRIDE = BM;
    constexpr int WSUBN = WN / WNITER;
    constexpr int WSUBM = (WARP_SIZE / (WSUBN / TN)) * TM;
    constexpr int WMITER = WM / WSUBM;
    // A smem mapping: inner_row_a = k-group, inner_col_a = m-offset
    constexpr int row_stride_a = NUM_THREADS / (BM / 4);
    constexpr int row_stride_b = NUM_THREADS / (BN / 4);
```

### 详细解释

这些变量全部是：

```cpp
constexpr int
```

因此在编译期就能确定，不需要运行时从内存读取。

---

### 基本 tile 参数

```cpp
BM, BN, BK
```

一个线程块处理：

```text
C 的 BM×BN 区域
```

并沿 K 方向每次处理 `BK` 层。

```cpp
WM, WN
```

一个 warp 负责：

```text
C 的 WM×WN 区域
```

```cpp
TM, TN
```

一个线程在每个 warp 子轮次中负责：

```text
TM×TN 个结果
```

---

### A 共享内存跨度

```cpp
constexpr int A_STRIDE = BM;
```

AS 按：

```text
[BK][BM]
```

布局，因此每增加一个 K 索引，需要跨过 `BM` 个 `float`。

---

### N 方向子块

```cpp
constexpr int WSUBN = WN / WNITER;
```

一个 warp 的 N tile 宽度是 `WN`，分 `WNITER` 轮完成。

每轮处理宽度：

```text
WSUBN = WN / WNITER
```

---

### M 方向子块

```cpp
constexpr int WSUBM =
    (WARP_SIZE / (WSUBN / TN)) * TM;
```

先计算 N 方向需要多少个线程列：

```text
WSUBN / TN
```

一个 warp 有 32 个线程，所以 M 方向线程行数是：

```text
32 / (WSUBN/TN)
```

每个线程处理 `TM` 行，所以 warp 每轮的 M 高度是：

```text
WSUBM = 线程行数 × TM
```

---

### M 方向迭代次数

```cpp
constexpr int WMITER = WM / WSUBM;
```

一个 warp 总共负责 `WM` 行，每轮处理 `WSUBM` 行，因此需要：

```text
WMITER
```

轮。

---

### 三组配置的具体推导

#### 默认配置

```text
BM=128, BN=256, BK=8
WM=64, WN=64
WNITER=4
TM=8, TN=4
NUM_THREADS=256
```

推导：

```text
WSUBN = 64/4 = 16
WSUBN/TN = 16/4 = 4 个线程列
warp 线程行数 = 32/4 = 8
WSUBM = 8×8 = 64
WMITER = 64/64 = 1
```

每线程结果数：

```text
WMITER×TM×WNITER×TN
= 1×8×4×4
= 128
```

256 个线程总共：

```text
256×128 = 32768
```

正好等于：

```text
BM×BN = 128×256 = 32768
```

#### 小尺寸配置

```text
BM=64, BN=64, BK=8
WM=32, WN=32
WNITER=2
TM=4, TN=4
NUM_THREADS=128
```

每线程结果：

```text
1×4×2×4 = 32
```

总结果：

```text
128×32 = 4096 = 64×64
```

#### 非规则配置

```text
BM=64, BN=128, BK=8
WM=32, WN=64
WNITER=2
TM=4, TN=4
NUM_THREADS=128
```

推导：

```text
WSUBN = 64/2 = 32
线程列 = 32/4 = 8
线程行 = 32/8 = 4
WSUBM = 4×4 = 16
WMITER = 32/16 = 2
```

每线程结果：

```text
2×4×2×4 = 64
```

总结果：

```text
128×64 = 8192 = 64×128
```

---

## 12. 当前线程块对应的 C tile

```cpp
    const uint c_row = blockIdx.y;
    const uint c_col = blockIdx.x;

    const int m_edge = (c_row * BM + BM <= M) ? BM : (M - c_row * BM);
    const int n_edge = (c_col * BN + BN <= N) ? BN : (N - c_col * BN);
    const bool fast = (m_edge == BM) && (n_edge == BN) && (K % 4 == 0) && (N % 4 == 0);

    const float *A_blk = A + c_row * BM * K;
    const float *B_blk = B + c_col * BN;
```

### 详细解释

CUDA 会启动二维 grid：

```text
x 方向覆盖 N
y 方向覆盖 M
```

所以：

```cpp
blockIdx.y
```

是 C tile 在 M 方向的编号。

```cpp
blockIdx.x
```

是 C tile 在 N 方向的编号。

---

### 有效边界大小

```cpp
m_edge
```

表示当前块在 M 方向实际有多少行有效。

三元表达式：

```cpp
(c_row * BM + BM <= M) ? BM : (M - c_row * BM)
```

如果完整 `BM` 行都在矩阵内，就取 `BM`；否则只取剩余行数。

N 方向同理。

---

### 快速路径条件

```cpp
const bool fast =
    (m_edge == BM) &&
    (n_edge == BN) &&
    (K % 4 == 0) &&
    (N % 4 == 0);
```

只有同时满足以下条件才进入快速路径：

1. 当前 M tile 完整；
2. 当前 N tile 完整；
3. K 是 4 的倍数；
4. N 是 4 的倍数。

后两个条件与 16 字节 `float4`/`cp.async16` 访问有关。

---

### A tile 起点

```cpp
const float *A_blk = A + c_row * BM * K;
```

A 是行主序 `[M][K]`。

当前线程块从全局第：

```text
c_row × BM
```

行开始，因此元素偏移是：

```text
c_row × BM × K
```

---

### B tile 起点

```cpp
const float *B_blk = B + c_col * BN;
```

B 是行主序 `[K][N]`。

当前线程块只在 N 方向移动，所以起始列偏移为：

```text
c_col × BN
```

沿 K 方向的移动后面再加：

```cpp
k * N
```

---

## 13. warp 在线程块中的二维映射

```cpp
    const uint warp_idx = threadIdx.x / WARP_SIZE;
    const uint warp_row = warp_idx / (BN / WN);
    const uint warp_col = warp_idx % (BN / WN);
```

### 详细解释

```cpp
threadIdx.x
```

是当前线程在线程块内的一维编号。

```cpp
warp_idx = threadIdx.x / 32
```

计算线程属于第几个 warp。

例如 256 线程配置有：

```text
256/32 = 8 个 warp
```

---

### warp 列数

```cpp
BN / WN
```

表示一个线程块 tile 在 N 方向能放多少个 warp tile。

例如默认配置：

```text
BN/WN = 256/64 = 4
```

所以 warp 按每行 4 个排列。

```cpp
warp_row = warp_idx / 4
warp_col = warp_idx % 4
```

8 个 warp 形成：

```text
2 行 × 4 列
```

每个 warp 负责 `64×64`，共同覆盖 `128×256`。

---

## 14. 线程负责的异步加载坐标

```cpp
    // A smem load indices: k-row and m-col (4 consecutive M)
    const uint inner_row_a = threadIdx.x / (BM / 4);
    const uint inner_col_a = threadIdx.x % (BM / 4) * 4;
    // B smem load indices
    const uint inner_row_b = threadIdx.x / (BN / 4);
    const uint inner_col_b = threadIdx.x % (BN / 4) * 4;
```

### 详细解释

每个线程一次搬运 4 个 `float`。

---

### A 的坐标

```cpp
BM / 4
```

表示一个 K 层中的 A tile 需要多少个 `float4`。

```cpp
inner_col_a =
    threadIdx.x % (BM/4) * 4
```

得到 M 方向起始坐标，并保证是 4 的倍数。

```cpp
inner_row_a =
    threadIdx.x / (BM/4)
```

得到线程初始负责的 K 层。

默认配置中：

```text
BM/4 = 32
```

线程 0～31 负责 K=0，线程 32～63 负责 K=1，以此类推。

---

### B 的坐标

B 的逻辑完全类似，只是使用 `BN/4`。

对于默认配置：

```text
BN/4 = 64
```

256 个线程会分成 4 组，分别负责 4 个初始 K 层；循环步长再覆盖其余 K 层。

---

## 15. 线程在 warp 子块中的二维坐标

```cpp
    const uint tiwarp = threadIdx.x % WARP_SIZE;
    const uint tcol   = tiwarp % (WSUBN / TN);
    const uint trow   = tiwarp / (WSUBN / TN);
```

### 详细解释

```cpp
tiwarp
```

是线程在 warp 内的编号，范围：

```text
0～31
```

每个线程在当前 warp 子块中映射成二维坐标。

N 方向线程列数是：

```text
WSUBN / TN
```

因此：

```cpp
tcol = tiwarp % 线程列数
trow = tiwarp / 线程列数
```

例如默认配置：

```text
WSUBN=16, TN=4
线程列数=4
```

warp 中 32 个线程排列成：

```text
8 行 × 4 列
```

每个线程负责：

```text
TM×TN = 8×4
```

个输出。

---

## 16. 双缓冲共享内存

```cpp
    // Double-buffered smem: AS[BK][A_STRIDE], BS[BK][BN]
    __shared__ float AS[2][A_STRIDE * BK];
    __shared__ float BS[2][BK * BN];
```

### 详细解释

```cpp
__shared__
```

表示数组存放在线程块共享内存中。

同一个 block 的所有线程都能访问，其他 block 不能访问。

---

### 两套缓冲区

第一维大小是 2：

```cpp
AS[2][...]
BS[2][...]
```

表示双缓冲：

```text
buffer 0
buffer 1
```

计算当前 tile 时，可以把下一个 tile 异步加载到另一套缓冲区，从而尝试重叠：

```text
计算
+
全局内存到共享内存搬运
```

---

### 默认配置共享内存量

A：

```text
2 × BK × BM
= 2×8×128
= 2048 float
= 8192 字节
```

B：

```text
2 × BK × BN
= 2×8×256
= 4096 float
= 16384 字节
```

合计：

```text
24576 字节 = 24 KiB
```

---

## 17. 当前 warp 的输出起点与线程寄存器

```cpp
    const uint c_row_start = c_row * BM + warp_row * WM;
    const uint c_col_start = c_col * BN + warp_col * WN;
    float *C_out = C + c_row_start * N + c_col_start;

    float reg_m[WMITER * TM];
    float reg_n[WNITER * TN];
    float res[WMITER * TM * WNITER * TN] = {};
```

### 详细解释

```cpp
c_row_start
c_col_start
```

是当前 warp 负责的 C tile 在全局矩阵中的起始行和列。

```cpp
C_out
```

直接指向该 warp tile 左上角。

后面所有局部行列偏移都相对于 `C_out` 计算。

---

### `reg_m`

```cpp
float reg_m[WMITER * TM];
```

保存当前 K 层中线程需要的 A 值。

---

### `reg_n`

```cpp
float reg_n[WNITER * TN];
```

保存当前 K 层中线程需要的 B 值。

---

### `res`

```cpp
float res[...] = {};
```

保存线程负责的全部输出累加值。

```cpp
= {}
```

表示值初始化，对基础数值数组而言会把所有元素初始化为 `0.0f`。

如果省略初始化，寄存器中可能是未定义值，后续 `+=` 会得到错误结果。

---

### 默认配置下的寄存器数组大小

```text
reg_m：1×8 = 8 个 float
reg_n：4×4 = 16 个 float
res：1×8×4×4 = 128 个 float
```

`res` 很大，会带来较高寄存器压力。这是高性能 GEMM 中常见的“更多线程级结果复用”和“占用率下降”之间的权衡。

---


## 18. 快速路径：K tile 数量与余数

```cpp
    if (fast) {
        int k_tiles = K / BK;
        int k_rem   = K % BK;
```

### 详细解释

如果前面的：

```cpp
fast == true
```

就进入快速路径。

```cpp
k_tiles = K / BK
```

表示 K 方向有多少个完整的 `BK` tile。

```cpp
k_rem = K % BK
```

表示完整 tile 之后还剩多少个 K 元素。

例如：

```text
K=100, BK=8
k_tiles=12
k_rem=4
```

即：

```text
12 个完整 8 层 tile
+
最后 4 层
```

---

## 19. 预加载第 0 个 tile

```cpp
        // Preload tile 0 → buf 0, wait before loop
        load_tile_async<BN,BM,BK,NUM_THREADS,A_STRIDE>(
            AS[0], BS[0], A_blk, B_blk, K, N,
            inner_row_a, inner_col_a, inner_row_b, inner_col_b);
        cp_async_wait_group<0>();
        __syncthreads();
```

### 详细解释

在正式循环前，先把第 0 个 K tile 异步加载到：

```text
AS[0]
BS[0]
```

模板实参：

```cpp
<BN,BM,BK,NUM_THREADS,A_STRIDE>
```

会在编译期生成对应配置的专用版本。

---

### 等待异步加载完成

```cpp
cp_async_wait_group<0>();
```

等待先前提交的全部 `cp.async` 组完成。

但这只保证每个线程自己发出的异步复制完成。

---

### block 级同步

```cpp
__syncthreads();
```

让整个线程块的线程都到达同步点后才能继续。

这是必要的，因为：

- 每个线程只负责共享内存的一小部分；
- 计算时，一个线程可能读取其他线程写入的 AS/BS 数据；
- 必须确保整个 tile 都已经准备好。

可以理解为：

```text
等待自己的异步复制完成
        ↓
等待全 block 所有线程都完成
        ↓
所有线程安全读取完整共享内存 tile
```

---

## 20. 双缓冲主循环

```cpp
        for (int t = 0; t < k_tiles; t++) {
            int cur = t & 1;
            int nxt = 1 - cur;

            if (t + 1 < k_tiles) {
                load_tile_async<BN,BM,BK,NUM_THREADS,A_STRIDE>(
                    AS[nxt], BS[nxt],
                    A_blk + (t+1)*BK, B_blk + (t+1)*BK*N, K, N,
                    inner_row_a, inner_col_a, inner_row_b, inner_col_b);
            }

            compute<BN,BM,BK,WN,WM,WNITER,WMITER,TN,TM,WSUBM,WSUBN,A_STRIDE>(
                reg_m, reg_n, AS[cur], BS[cur],
                trow, tcol, warp_row, warp_col, res);

            if (t + 1 < k_tiles) {
                cp_async_wait_group<0>();
                __syncthreads();
            }
        }
```

### 详细解释

该循环处理所有完整 `BK` tile。

---

### 当前缓冲区

```cpp
int cur = t & 1;
```

`&` 是按位与运算。

对于整数：

```text
偶数最低位是 0
奇数最低位是 1
```

因此：

```text
t=0 → cur=0
t=1 → cur=1
t=2 → cur=0
t=3 → cur=1
```

等价于：

```cpp
t % 2
```

但位运算能直接表达双缓冲奇偶切换。

---

### 下一个缓冲区

```cpp
int nxt = 1 - cur;
```

如果当前是 0，下一个是 1；当前是 1，下一个是 0。

---

### 提前加载下一个 tile

```cpp
if (t + 1 < k_tiles)
```

只有确实存在下一个完整 tile 时才加载。

源地址：

```cpp
A_blk + (t+1)*BK
```

表示沿 A 的 K 方向移动到下一个 tile 起点。

对于行主序 A，每行长度是 K，因此单纯增加 `BK` 表示每一行中的 K 起点右移 `BK` 个元素；后续加载函数还需要按每个 M 行正确加上 `m*K`。

B 的偏移：

```cpp
B_blk + (t+1)*BK*N
```

B 是 `[K][N]`，沿 K 方向移动 `BK` 行，每行有 N 个元素，所以偏移：

```text
BK×N
```

---

### 搬运与计算重叠

执行顺序是：

```text
向 nxt 缓冲区发出下一个 tile 的 cp.async
        ↓
使用 cur 缓冲区计算当前 tile
        ↓
等待 nxt 加载完成
```

理想情况下，GPU 能让全局内存搬运和当前 tile 的算术运算部分重叠。

这就是双缓冲的核心。

---

### 当前 tile 计算

```cpp
compute<...>(
    reg_m, reg_n, AS[cur], BS[cur],
    trow, tcol, warp_row, warp_col, res);
```

`compute` 会遍历：

```text
d = 0～BK-1
```

把当前 tile 对 C 的贡献累加到 `res`。

`res` 不会在每轮清零，因为 K 方向不同 tile 的结果必须不断累加。

---

### 等待下一缓冲区

```cpp
cp_async_wait_group<0>();
__syncthreads();
```

在下一轮开始读取 `AS[nxt]` 和 `BS[nxt]` 前，确保：

1. 异步复制已经完成；
2. 整个 block 的所有线程都完成当前计算；
3. 没有线程提前切换并覆盖仍被其他线程读取的共享内存。

---

### 时间线示例

若有 3 个完整 tile：

```text
预加载 tile0 到 buf0
等待 tile0
计算 tile0，同时加载 tile1 到 buf1
等待 tile1
计算 tile1，同时加载 tile2 到 buf0
等待 tile2
计算 tile2
```

---

## 21. K 方向余数处理：清零共享内存

```cpp
        // K remainder
        if (k_rem > 0) {
            int cur = k_tiles & 1;
            for (int i = threadIdx.x; i < A_STRIDE*BK; i += NUM_THREADS) AS[cur][i] = 0.f;
            for (int i = threadIdx.x; i < BK*BN; i += NUM_THREADS) BS[cur][i] = 0.f;
            __syncthreads();
```

### 详细解释

如果：

```cpp
k_rem > 0
```

说明 K 不能被 BK 整除，需要处理最后一个不完整 tile。

---

### 选择空闲缓冲区

```cpp
int cur = k_tiles & 1;
```

假设完整 tile 数是：

```text
1
```

主循环最后使用 buffer 0，那么余数使用 buffer 1。

假设完整 tile 数是：

```text
2
```

主循环依次使用 0、1，那么余数使用 buffer 0。

因此 `k_tiles & 1` 正好选择下一套缓冲区。

---

### 线程合作清零 AS

```cpp
for (int i = threadIdx.x;
     i < A_STRIDE*BK;
     i += NUM_THREADS)
{
    AS[cur][i] = 0.f;
}
```

每个线程从自己的 `threadIdx.x` 开始，每次跨越整个线程块线程数。

例如 128 个线程清零 1024 个元素：

```text
线程0：0,128,256,...
线程1：1,129,257,...
...
```

所有线程合起来覆盖全部数组。

BS 清零逻辑相同。

---

### 为什么先清零

最后一个 tile 只有 `k_rem` 层有效，其余 `BK-k_rem` 层没有全局数据。

先把完整共享内存 tile 清零，再只写入有效位置，可以保证无效位置为 0。

---

## 22. K 余数路径：A/B 标量加载

```cpp
            const float *A_tail = A_blk + k_tiles*BK;
            const float *B_tail = B_blk + k_tiles*BK*N;
            // A: scalar load into [BK][BM]
            for (uint i = 0; i < BK; i += row_stride_a) {
                uint k = inner_row_a + i;
                if ((int)k < k_rem)
                    for (int mm = 0; mm < 4; mm++)
                        AS[cur][k*A_STRIDE + inner_col_a+mm] = A_tail[(inner_col_a+mm)*K + k];
            }
            // B: scalar load into [BK][BN]
            for (uint i = 0; i < BK; i += row_stride_b) {
                uint k = inner_row_b + i;
                if ((int)k < k_rem)
                    for (int nn = 0; nn < 4; nn++)
                        BS[cur][k*BN + inner_col_b+nn] = B_tail[k*N + inner_col_b+nn];
            }
            __syncthreads();
```

### 详细解释

A 尾部起点：

```cpp
A_tail = A_blk + k_tiles*BK
```

对于每个 A 行，从当前完整 tile 之后的 K 坐标开始。

B 尾部起点：

```cpp
B_tail = B_blk + k_tiles*BK*N
```

沿 K 方向跳过 `k_tiles×BK` 行。

---

### `(int)k`

```cpp
if ((int)k < k_rem)
```

`k` 是无符号 `uint`，`k_rem` 是有符号 `int`。

显式把 `k` 转为 `int`，避免有符号和无符号比较产生警告或意外转换。

---

### A 的正确行主序索引

```cpp
A_tail[(inner_col_a+mm)*K + k]
```

这里：

```text
inner_col_a+mm → M 坐标
k → 当前尾部 tile 内的 K 坐标
```

普通行主序 A `[M][K]` 的下标正是：

```text
m*K + k
```

加载后写入：

```text
AS[k][m]
```

即转置式共享内存布局。

这也进一步说明，本段与快速异步加载中的 A 源地址表达式需要对照检查。

---

### B 索引

```cpp
B_tail[k*N + inner_col_b+nn]
```

B 是行主序 `[K][N]`，下标是：

```text
k*N+n
```

写入共享内存相同布局：

```text
BS[k][n]
```

---

### 为什么使用标量循环

快速路径使用一次 16 字节复制。

尾部可能不足完整 BK，但 M/N tile 仍完整；这里使用 4 次标量赋值，更容易精确控制只读取有效 K 层。

---

## 23. 计算 K 余数

```cpp
            compute_k<BN,BM,BK,WN,WM,WNITER,WMITER,TN,TM,WSUBM,WSUBN,A_STRIDE>(
                reg_m, reg_n, AS[cur], BS[cur],
                trow, tcol, warp_row, warp_col, res, k_rem);
        }
```

### 详细解释

共享内存尾部 tile 准备好后，调用：

```cpp
compute_k(..., k_rem)
```

只遍历有效的：

```text
d=0～k_rem-1
```

结果继续累加到之前完整 tile 已经使用的同一个：

```cpp
res
```

中。

右花括号结束：

```cpp
if (k_rem > 0)
```

---

## 24. 快速路径的 `float4` 回写

```cpp
        // float4 write-back
        for (uint wr = 0; wr < WMITER; wr++) {
            for (uint wc = 0; wc < WNITER; wc++) {
                for (uint rm = 0; rm < TM; rm++) {
                    int row = wr*WSUBM + trow*TM + rm;
                    int col = wc*WSUBN + tcol*TN;
                    int ir = (wr*TM+rm)*(TN*WNITER) + wc*TN;
                    float4 v = *reinterpret_cast<const float4*>(&C_out[row*N + col]);
                    v.x = alpha*res[ir]   + beta*v.x;
                    v.y = alpha*res[ir+1] + beta*v.y;
                    v.z = alpha*res[ir+2] + beta*v.z;
                    v.w = alpha*res[ir+3] + beta*v.w;
                    *reinterpret_cast<float4*>(&C_out[row*N + col]) = v;
                }
            }
        }
```

### 详细解释

计算结束后，每个线程把自己 `res` 中的结果写回全局矩阵 C。

---

### 局部行列

```cpp
row = wr*WSUBM + trow*TM + rm;
```

是相对当前 warp tile 起点 `C_out` 的行偏移。

```cpp
col = wc*WSUBN + tcol*TN;
```

是相对当前 warp tile 起点的列偏移。

---

### 结果寄存器下标

```cpp
int ir =
    (wr*TM+rm)*(TN*WNITER) + wc*TN;
```

得到当前一组连续 `TN` 个结果在 `res` 中的起点。

当前三组配置都满足：

```text
TN=4
```

所以从 `ir` 开始正好有 4 个连续结果。

---

### `float4`

CUDA 定义的：

```cpp
float4
```

包含四个 `float` 成员：

```cpp
v.x
v.y
v.z
v.w
```

总大小是 16 字节。

---

### `reinterpret_cast`

```cpp
reinterpret_cast<const float4*>(
    &C_out[row*N + col]
)
```

把原本的：

```cpp
float*
```

重新解释为：

```cpp
const float4*
```

然后通过前面的 `*` 解引用，一次读取 4 个连续 `float`。

这种转换不会改变内存内容，只改变编译器看待该地址的类型。

---

### GEMM 融合公式

四个元素分别执行：

```cpp
alpha*res[...] + beta*v....
```

对应：

```text
C_new = alpha × AB_result + beta × C_old
```

---

### 向量化写回

```cpp
*reinterpret_cast<float4*>(&C_out[row*N + col]) = v;
```

把更新后的 `float4` 一次写回 16 字节。

快速路径保证：

```text
N % 4 == 0
TN == 4
col 是 4 的倍数
```

从而满足连续 4 元素访问。

此外快速路径只用于完整 M/N tile，所以这里不逐元素检查：

```cpp
row < M
col < N
```

---

### 关于读取旧 C

即使：

```cpp
beta == 0
```

代码仍会先读取 `C_old` 到 `v`。

在当前测试程序中，自定义 GEMM 前调用了：

```cpp
cudaMemset(d_C, 0, sizeC)
```

所以读取是安全且确定的。

若把该接口用于其他调用场景，即使 `beta=0`，也最好保证 C 指向合法已分配内存；若希望完全避免旧 C 读取，可以为 `beta==0` 单独编写回写分支。

---

## 25. 快速路径结束

```cpp
    } else {
```

### 详细解释

这个：

```cpp
else
```

与前面的：

```cpp
if (fast)
```

对应。

当以下任意条件不满足时进入边界/非对齐路径：

- M tile 不完整；
- N tile 不完整；
- K 不是 4 的倍数；
- N 不是 4 的倍数。

该路径不使用 `cp.async16` 和无边界 `float4` 回写，而是使用更谨慎的标量加载和逐元素边界判断。

---


## 26. 边界路径：沿 K 分块循环

```cpp
        // Edge / unaligned path
        for (int k = 0; k < K; k += BK) {
            int k_rem = (k + BK <= K) ? BK : (K - k);
            const float *A_tile = A_blk + k;
            const float *B_tile = B_blk + k*N;
```

### 详细解释

边界路径不使用前面的双缓冲异步流水，而是逐个 K tile：

```text
清零共享内存
→ 加载有效数据
→ 同步
→ 计算
→ 同步
→ 下一个 tile
```

---

### K 循环

```cpp
for (int k = 0; k < K; k += BK)
```

`k` 是当前 tile 在全局 K 方向的起点。

例如：

```text
K=20, BK=8
```

循环起点依次是：

```text
k=0
k=8
k=16
```

---

### 当前 tile 的有效 K 大小

```cpp
int k_rem =
    (k + BK <= K) ? BK : (K - k);
```

完整 tile 取 `BK`。

最后不足 BK 时取剩余数量。

上例中：

```text
k=0  → k_rem=8
k=8  → k_rem=8
k=16 → k_rem=4
```

---

### 当前 A/B tile 起点

```cpp
A_tile = A_blk + k;
```

A 是行主序 `[M][K]`。在每个 A 行中，当前 tile 从 K 坐标 `k` 开始。

```cpp
B_tile = B_blk + k*N;
```

B 是 `[K][N]`，沿 K 方向移动 k 行，每行 N 个元素。

---

## 27. 边界路径：共享内存清零

```cpp
            for (int i = threadIdx.x; i < A_STRIDE*BK; i += NUM_THREADS) AS[0][i] = 0.f;
            for (int i = threadIdx.x; i < BK*BN; i += NUM_THREADS) BS[0][i] = 0.f;
            __syncthreads();
```

### 详细解释

边界路径只使用：

```text
AS[0]
BS[0]
```

不进行双缓冲。

每一轮都先让整个 block 合作清零共享内存。

---

### 为什么每个 K tile 都清零

当前 tile 可能同时存在三类不完整情况：

- M 边界不足 BM；
- N 边界不足 BN；
- K 边界不足 BK。

只有有效位置会从全局内存加载。提前清零可以保证没有被加载的位置为 0。

---

### 为什么清零后立即同步

```cpp
__syncthreads();
```

确保所有线程都完成清零后，才允许任何线程开始写入有效数据。

否则可能出现：

```text
线程 A 已加载有效值
线程 B 仍在清零同一位置
```

导致有效值被覆盖。

---

## 28. 边界路径：加载 A tile

```cpp
            // A into [BK][A_STRIDE]: AS[k*A_STRIDE + m]
            for (uint i = 0; i < BK; i += row_stride_a) {
                uint kk = inner_row_a + i;
                if ((int)kk >= k_rem) continue;
                int mb = inner_col_a;
                if (mb >= m_edge) continue;
                int valid = (mb+4 <= m_edge) ? 4 : m_edge-mb;
                for (int mm = 0; mm < valid; mm++)
                    AS[0][kk*A_STRIDE + mb+mm] = A_tile[(mb+mm)*K + kk];
            }
```

### 详细解释

目标布局仍是：

```text
AS[kk][m]
```

源矩阵 A 是行主序：

```text
A[m][全局K]
```

---

### 当前 K 坐标

```cpp
uint kk = inner_row_a + i;
```

得到当前线程负责的 tile 内 K 索引。

```cpp
if ((int)kk >= k_rem) continue;
```

如果超过当前 tile 的有效 K 范围，直接跳到下一轮循环。

`continue` 表示：

> 不执行本轮剩余代码，直接进入下一次循环。

---

### 当前 M 向量起点

```cpp
int mb = inner_col_a;
```

线程原计划加载：

```text
mb, mb+1, mb+2, mb+3
```

四个 M 位置。

```cpp
if (mb >= m_edge) continue;
```

如果起点已经超出有效 M 范围，本线程本轮不加载。

---

### 有效元素数量

```cpp
int valid =
    (mb+4 <= m_edge) ? 4 : m_edge-mb;
```

如果完整 4 个元素都在边界内，加载 4 个。

否则只加载剩余的 1～3 个元素。

例如：

```text
m_edge=66
mb=64
```

则：

```text
valid=2
```

只加载 M 坐标 64 和 65。

---

### A 的行主序下标

```cpp
A_tile[(mb+mm)*K + kk]
```

其中：

- `mb+mm`：A 的 M 行；
- `kk`：当前 tile 内 K 偏移；
- `A_tile` 已经指向全局 K 起点 `k`。

所以完整全局位置是：

```text
A[mb+mm][k+kk]
```

写入：

```text
AS[kk][mb+mm]
```

即把 A 的 `[M][K]` tile 以 `[K][M]` 形式放入共享内存。

这样计算时固定 `d=kk`，可以连续读取多个 M 值。

---

## 29. 边界路径：加载 B tile

```cpp
            // B into [BK][BN]
            for (uint i = 0; i < BK; i += row_stride_b) {
                uint kk = inner_row_b + i;
                if ((int)kk >= k_rem) continue;
                int nb = inner_col_b;
                if (nb >= n_edge) continue;
                int valid = (nb+4 <= n_edge) ? 4 : n_edge-nb;
                for (int nn = 0; nn < valid; nn++)
                    BS[0][kk*BN + nb+nn] = B_tile[kk*N + nb+nn];
            }
            __syncthreads();
```

### 详细解释

B 的逻辑与 A 类似，但 B 的全局内存和共享内存都是：

```text
[K][N]
```

因此不需要转置。

---

### N 边界

```cpp
int nb = inner_col_b;
```

当前线程计划从 N 坐标 `nb` 开始加载 4 个元素。

```cpp
if (nb >= n_edge) continue;
```

若起点超出有效 N 范围，则跳过。

```cpp
valid = ...
```

计算最后一个 `float4` 分组中实际有效的 1～4 个元素。

---

### B 下标

```cpp
B_tile[kk*N + nb+nn]
```

表示：

```text
B[k+kk][当前线程块N起点+nb+nn]
```

写入：

```text
BS[kk][nb+nn]
```

---

### 加载结束同步

```cpp
__syncthreads();
```

确保整个共享内存 tile 已经由所有线程共同准备完成，随后才能进入计算。

---

## 30. 边界 tile 计算与轮次同步

```cpp
            compute_edge<BN,BM,BK,WN,WM,WNITER,WMITER,TN,TM,WSUBM,WSUBN,A_STRIDE>(
                reg_m, reg_n, AS[0], BS[0],
                trow, tcol, warp_row, warp_col, res, k_rem, m_edge, n_edge);
            __syncthreads();
        }
```

### 详细解释

调用：

```cpp
compute_edge(...)
```

时额外传入：

```cpp
k_rem
m_edge
n_edge
```

所以计算函数能够只处理有效 K 层，并对超出 M/N 边界的线程数据使用 0。

---

### 计算后的同步

```cpp
__syncthreads();
```

这一同步很重要。

下一轮 K tile 开始时，线程会清零：

```text
AS[0]
BS[0]
```

如果某些线程已经开始清零，而另一些线程还在 `compute_edge` 中读取共享内存，就会产生数据竞争。

因此必须先确保：

```text
所有线程完成当前 tile 计算
```

再进入下一轮清零和加载。

---

## 31. 边界路径逐元素回写

```cpp
        for (uint wr = 0; wr < WMITER; wr++) {
            for (uint wc = 0; wc < WNITER; wc++) {
                uint lrb = wr*WSUBM + trow*TM;
                uint lcb = wc*WSUBN + tcol*TN;
                for (uint rm = 0; rm < TM; rm++) {
                    int grow = c_row_start + lrb + rm;
                    if (grow >= M) continue;
                    for (uint rn = 0; rn < TN; rn++) {
                        int gcol = c_col_start + lcb + rn;
                        if (gcol >= N) continue;
                        int ir = (wr*TM+rm)*(TN*WNITER) + wc*TN+rn;
                        C_out[(lrb+rm)*N + lcb+rn] =
                            alpha*res[ir] + beta*C_out[(lrb+rm)*N + lcb+rn];
                    }
                }
            }
        }
```

### 详细解释

边界路径不能直接用无条件 `float4` 回写，因为最后一组可能只有 1～3 个元素有效。

因此逐个 `float` 检查边界并写回。

---

### 局部起点

```cpp
lrb = wr*WSUBM + trow*TM;
```

当前线程在本 warp tile 中的局部行起点。

```cpp
lcb = wc*WSUBN + tcol*TN;
```

局部列起点。

---

### 全局行

```cpp
grow = c_row_start + lrb + rm;
```

把局部行加到当前 warp 的全局起始行上。

```cpp
if (grow >= M) continue;
```

超出矩阵真实行数时跳过。

---

### 全局列

```cpp
gcol = c_col_start + lcb + rn;
```

如果：

```cpp
gcol >= N
```

则不写该元素。

---

### 结果下标

```cpp
ir =
    (wr*TM+rm)*(TN*WNITER)
    + wc*TN+rn;
```

与 `compute_edge` 中累加的位置完全一致。

---

### C 的地址

```cpp
C_out[(lrb+rm)*N + lcb+rn]
```

`C_out` 已经指向当前 warp tile 左上角，所以这里只需要加局部行列偏移。

更新公式仍然是：

```text
C = alpha×res + beta×C
```

---

## 32. 结束边界路径与 Kernel

```cpp
    }
}
```

### 详细解释

第一个右花括号结束：

```cpp
if (fast) { ... } else { ... }
```

第二个右花括号结束：

```cpp
gemm_warp_tile_kernel
```

每个 GPU 线程执行到这里后退出 Kernel。

---

## 33. 配置化 Kernel 启动辅助函数

```cpp
template <typename Cfg>
static inline void launch_gemm_warp_tile_cfg(int M, int N, int K,
                                             float alpha, const float *A, const float *B,
                                             float beta, float *C)
{
    dim3 blockDim(Cfg::NUM_THREADS);
    dim3 gridDim(CEIL_DIV(N, Cfg::BN), CEIL_DIV(M, Cfg::BM));
    gemm_warp_tile_kernel<Cfg><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
```

### 详细解释

这是 CPU 端普通函数，不是 Kernel。

它根据配置类型 `Cfg` 生成 grid、block，并启动对应版本的 Kernel。

---

### `static inline`

```cpp
static
```

在当前源文件范围内限制函数可见性，其他源文件不能直接链接调用该函数。

```cpp
inline
```

提示编译器这是一个适合内联的短函数。

---

### `dim3`

```cpp
dim3 blockDim(Cfg::NUM_THREADS);
```

`dim3` 是 CUDA 提供的三维维度类型。

只传一个参数时：

```text
x = Cfg::NUM_THREADS
y = 1
z = 1
```

所以线程块是一维的。

---

### grid 维度

```cpp
dim3 gridDim(
    CEIL_DIV(N, Cfg::BN),
    CEIL_DIV(M, Cfg::BM)
);
```

grid 的：

```text
x 方向覆盖 N
y 方向覆盖 M
```

例如：

```text
M=1000
N=1000
BM=64
BN=128
```

则：

```text
grid.x = ceil(1000/128) = 8
grid.y = ceil(1000/64) = 16
```

总共：

```text
8×16 = 128 个 block
```

边缘 block 通过 `m_edge` 和 `n_edge` 处理不足完整 tile 的部分。

---

### Kernel 启动语法

```cpp
gemm_warp_tile_kernel<Cfg>
    <<<gridDim, blockDim>>>
    (M, N, K, alpha, A, B, beta, C);
```

分成三部分：

```cpp
gemm_warp_tile_kernel<Cfg>
```

选择模板配置。

```cpp
<<<gridDim, blockDim>>>
```

CUDA Kernel 启动配置。

```cpp
(...)
```

传给每个 GPU 线程的 Kernel 参数。

Kernel 启动通常是异步的。CPU 函数提交 Kernel 后即可返回；调用者通过 CUDA Event、`cudaDeviceSynchronize` 或数据拷贝进行同步。

---

## 34. 对外公开的 Warp Tile 启动函数

```cpp
void launch_gemm_warp_tile(int M, int N, int K,
                           float alpha, const float *A, const float *B,
                           float beta, float *C)
{
    const bool small_problem = (M <= 1024 && N <= 1024 && K <= 1024);
    const bool irregular_shape = (M % 64 != 0) || (N % 128 != 0) || (K % 8 != 0);
    const bool irregular_medium = irregular_shape && (M <= 3072 && N <= 3072 && K <= 3072);
    if (irregular_medium) {
        launch_gemm_warp_tile_cfg<WarpTileIrregularConfig>(M, N, K, alpha, A, B, beta, C);
    } else if (small_problem) {
        launch_gemm_warp_tile_cfg<WarpTileSmallConfig>(M, N, K, alpha, A, B, beta, C);
    } else {
        launch_gemm_warp_tile_cfg<WarpTileDefaultConfig>(M, N, K, alpha, A, B, beta, C);
    }
}
```

### 详细解释

该函数接收统一 GEMM 参数，并根据矩阵形状选择三种配置之一。

---

### 小问题判断

```cpp
const bool small_problem =
    (M <= 1024 && N <= 1024 && K <= 1024);
```

只有三个维度都不超过 1024 才为 `true`。

---

### 非规则形状判断

```cpp
const bool irregular_shape =
    (M % 64 != 0) ||
    (N % 128 != 0) ||
    (K % 8 != 0);
```

只要存在以下任意情况就认为形状不规则：

- M 不是 64 的倍数；
- N 不是 128 的倍数；
- K 不是 8 的倍数。

`||` 是逻辑或，任一条件成立，整体就为真。

这些倍数对应 `WarpTileIrregularConfig` 的：

```text
BM=64
BN=128
BK=8
```

---

### 中小型非规则问题

```cpp
const bool irregular_medium =
    irregular_shape &&
    (M <= 3072 && N <= 3072 && K <= 3072);
```

要求：

1. 形状不规则；
2. 三个维度都不超过 3072。

---

### 分支优先级

代码首先判断：

```cpp
if (irregular_medium)
```

然后才判断：

```cpp
else if (small_problem)
```

所以一个既是小问题、又是不规则形状的矩阵，会优先使用：

```cpp
WarpTileIrregularConfig
```

而不是 SmallConfig。

例如：

```text
M=N=K=100
```

满足：

```text
small_problem = true
irregular_medium = true
```

最终进入第一分支。

---

### 三种配置

#### `WarpTileIrregularConfig`

```text
BM=64, BN=128, BK=8
WM=32, WN=64
WNITER=2
TM=4, TN=4
NUM_THREADS=128
```

用于不规则的中小尺寸。

#### `WarpTileSmallConfig`

```text
BM=64, BN=64, BK=8
WM=32, WN=32
WNITER=2
TM=4, TN=4
NUM_THREADS=128
```

用于规则的小尺寸。

#### `WarpTileDefaultConfig`

```text
BM=128, BN=256, BK=8
WM=64, WN=64
WNITER=4
TM=8, TN=4
NUM_THREADS=256
```

用于其他较大问题。

---

## 35. 三层调用关系

```text
launch_gemm(...)
        ↓
launch_gemm_warp_tile(...)
        ↓
根据形状选择配置
        ↓
launch_gemm_warp_tile_cfg<Cfg>(...)
        ↓
构造 gridDim 和 blockDim
        ↓
gemm_warp_tile_kernel<Cfg><<<...>>>(...)
```

`launch_gemm_warp_tile` 决定用哪套 tile 参数。

`launch_gemm_warp_tile_cfg` 把 tile 参数转换为 CUDA 启动配置。

`gemm_warp_tile_kernel` 真正在 GPU 上完成矩阵乘法。

---

## 36. 一个 block 内的数据层级

以默认配置为例：

```text
Block tile：BM×BN = 128×256
线程数：256 = 8 个 warp
```

8 个 warp 排列成：

```text
2 行 × 4 列
```

每个 warp 负责：

```text
WM×WN = 64×64
```

一个 warp 的 N 方向分 4 轮：

```text
WNITER=4
WSUBN=16
```

M 方向只需 1 轮：

```text
WMITER=1
WSUBM=64
```

warp 内 32 个线程排成：

```text
8 行 × 4 列
```

每线程每轮负责：

```text
TM×TN = 8×4
```

每线程总共负责：

```text
1×8×4×4 = 128 个 C 元素
```

---

## 37. 一个完整 K tile 的数据流

```text
全局内存 A、B
        ↓ cp.async16
共享内存 AS[BK][BM]、BS[BK][BN]
        ↓
每个线程从 AS 读取 reg_m
每个线程从 BS 读取 reg_n
        ↓
reg_m 与 reg_n 做外积
        ↓
累加到线程私有 res
        ↓
继续处理下一个 BK tile
        ↓
所有 K 处理完
        ↓
alpha×res + beta×C
        ↓
写回全局内存 C
```

---

## 38. 快速路径与边界路径对照

| 项目 | 快速路径 | 边界/非对齐路径 |
|---|---|---|
| M/N tile | 必须完整 | 可以不完整 |
| K、N 对齐 | 需要 4 元素对齐 | 不要求 |
| 全局到共享搬运 | `cp.async16` | 标量循环 |
| 共享内存 | 双缓冲 | 只使用 buffer 0 |
| 搬运计算重叠 | 有 | 无 |
| 写回 | `float4` | 逐个 `float` |
| 边界检查 | 基本不需要 | 每个有效元素检查 |
| 计算函数 | `compute/compute_k` | `compute_edge` |

---

## 39. 阅读本文件时需要特别注意的代码一致性

快速加载 A 的代码是：

```cpp
cp_async16(
    dst,
    A_tile + k*K + inner_col_a,
    true
);
```

而 K 余数和边界路径分别使用：

```cpp
A_tail[(inner_col_a+mm)*K + k]
```

```cpp
A_tile[(mb+mm)*K + kk]
```

后两者明确按普通行主序 A `[M][K]` 读取：

```text
A[m][k]
```

再写入共享内存：

```text
AS[k][m]
```

快速路径却按表达式形式读取：

```text
A[k][m]
```

因此在继续学习或验证该项目时，应使用随机且非对称的 A/B 数据做正确性测试，不能只使用：

```cpp
A[i] = 1.0f;
B[i] = 2.0f;
```

全相同输入会掩盖转置、错行和错列问题。

本节只是对源码实际索引关系进行展开说明；最终是否为有意的数据预布局，需要结合调用前 A 的真实存储形式确认。

---

## 40. 源码整体执行顺序

```text
CPU 调用 launch_gemm_warp_tile
        ↓
判断 small / irregular / default
        ↓
选定 Cfg
        ↓
计算 gridDim、blockDim
        ↓
启动 gemm_warp_tile_kernel<Cfg>
        ↓
每个 block 确定自己负责的 BM×BN C tile
        ↓
每个 warp 确定自己负责的 WM×WN 区域
        ↓
每个线程确定 trow、tcol 和结果寄存器范围
        ↓
判断当前 block 是否满足 fast
        ├── fast
        │     ↓
        │   cp.async 双缓冲加载完整 K tile
        │     ↓
        │   compute 累加完整 tile
        │     ↓
        │   compute_k 处理 K 余数
        │     ↓
        │   float4 向量化回写
        │
        └── edge
              ↓
            每轮清零共享内存
              ↓
            标量加载有效 A/B
              ↓
            compute_edge
              ↓
            逐元素边界检查回写
```

---

## 41. 最核心的线程坐标关系

```text
blockIdx.y
    决定 C 的 M 方向 block tile

blockIdx.x
    决定 C 的 N 方向 block tile

warp_idx = threadIdx.x / 32
    决定线程属于哪个 warp

warp_row、warp_col
    决定 warp 负责哪个 WM×WN tile

tiwarp = threadIdx.x % 32
    决定线程在 warp 内的编号

trow、tcol
    决定线程在 WSUBM×WSUBN 子块中的二维位置

wr、wc
    决定 warp 的第几个 M/N 子轮次

rm、rn
    决定线程自己负责的 TM×TN 中具体哪个结果
```



%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-gemm算子|模块-gemm算子]]

%% 项目关联导航：结束 %%
