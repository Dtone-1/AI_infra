# 面试手撕：最简单的 CUDA GEMM 算子

> 目标：面试时能够不依赖复杂优化，直接写出一个**正确、清晰、容易解释**的 CUDA GEMM。
>
> 这不是高性能 GEMM，而是最基础的 **Naive GEMM**：
>
> ```text
> 一个 CUDA 线程负责计算 C 矩阵中的一个元素。
> ```
>
> 你只要把这一版真正写熟，面试官继续追问 Shared Memory、Tiling、Warp Tile、Tensor Core 时，再从这个版本逐层优化即可。

---

# 1. 先明确 GEMM 在算什么

假设：

```text
A：M × K
B：K × N
C：M × N
```

矩阵乘法：

```text
C = A × B
```

那么 C 中第 `row` 行、第 `col` 列的元素为：

```text
C[row][col]
=
A[row][0] × B[0][col]
+
A[row][1] × B[1][col]
+
...
+
A[row][K-1] × B[K-1][col]
```

也就是：

```text
C[row][col] = Σ A[row][k] × B[k][col]
```

CUDA 最简单的并行化思路就是：

```text
线程 0 → 算一个 C 元素
线程 1 → 算另一个 C 元素
线程 2 → 算另一个 C 元素
...
```

所以：

> **整个二维 C 矩阵有多少个元素，就需要多少个逻辑线程去覆盖。**

---

# 2. 面试最推荐手写的核心 Kernel

```cpp
__global__ void gemm_naive(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;

    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}
```

这就是最简单、最标准的 CUDA GEMM。

面试时最重要的是能把它解释成下面一句话：

> 我让一个线程负责输出矩阵 C 的一个元素。线程先根据 Block 和 Thread 索引得到自己的 `row` 和 `col`，然后沿 K 维遍历 A 的这一行和 B 的这一列做点积，最后把结果写到 `C[row][col]`。

---

# 3. 第一部分：Kernel 参数是什么意思

```cpp
__global__ void gemm_naive(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
)
```

## 3.1 `__global__`

```cpp
__global__
```

表示：

> 这是一个 CUDA Kernel，由 CPU 发起调用，在 GPU 上由大量线程并行执行。

普通 C++ 函数：

```cpp
void func()
```

一般在 CPU 上执行。

CUDA Kernel：

```cpp
__global__ void kernel()
```

在 GPU 上执行。

启动方式也不同：

```cpp
gemm_naive<<<grid, block>>>(...);
```

---

## 3.2 `A`

```cpp
const float* A
```

指向 GPU Global Memory 中的矩阵 A。

A 的逻辑形状：

```text
M × K
```

`const` 表示：

```text
Kernel 只读取 A，不修改 A。
```

---

## 3.3 `B`

```cpp
const float* B
```

指向矩阵 B。

逻辑形状：

```text
K × N
```

同样只读。

---

## 3.4 `C`

```cpp
float* C
```

指向输出矩阵 C。

逻辑形状：

```text
M × N
```

Kernel 最终会把计算结果写到这里。

---

## 3.5 `M、N、K`

```cpp
int M,
int N,
int K
```

表示矩阵维度：

```text
A：M × K
B：K × N
C：M × N
```

一定要把这个关系记熟：

```text
A: M × K
B: K × N
C: M × N
```

---

# 4. 第二部分：一个线程如何找到自己要算的 C 元素

核心代码：

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

这是整个 Naive GEMM 最重要的索引逻辑。

---

## 4.1 为什么使用二维线程布局

因为输出矩阵 C 本身是二维的：

```text
C[M][N]
```

所以最自然的映射就是：

```text
CUDA y 方向 → 矩阵行 row
CUDA x 方向 → 矩阵列 col
```

也就是：

```text
threadIdx.y → row
threadIdx.x → col
```

---

## 4.2 `row` 是怎么来的

```cpp
int row =
    blockIdx.y * blockDim.y
    + threadIdx.y;
```

可以理解为：

```text
前面完整 Block 已经覆盖多少行
+
当前线程在本 Block 内是第几行
```

例如：

```text
blockDim.y = 16
blockIdx.y = 2
threadIdx.y = 3
```

那么：

```text
row = 2 × 16 + 3 = 35
```

这个线程负责 C 的第 35 行。

---

## 4.3 `col` 是怎么来的

同理：

```cpp
int col =
    blockIdx.x * blockDim.x
    + threadIdx.x;
```

例如：

```text
blockDim.x = 16
blockIdx.x = 4
threadIdx.x = 7
```

那么：

```text
col = 4 × 16 + 7 = 71
```

这个线程负责 C 的第 71 列。

---

## 4.4 最终线程负责什么

所以这个线程最终只负责：

```text
C[row][col]
```

一个元素。

这就是最简单 GEMM 的线程映射：

```text
一个 Thread
    ↓
一个 C 元素
```

---

# 5. 第三部分：为什么要做边界检查

```cpp
if (row >= M || col >= N) {
    return;
}
```

这是必须写的。

原因是 CUDA 启动线程时，通常会把线程数向上取整。

假设：

```text
M = 100
N = 100
```

线程块：

```text
16 × 16
```

那么需要：

```text
ceil(100 / 16) = 7
```

个 Block。

7 个 Block 可以覆盖：

```text
7 × 16 = 112
```

个位置。

所以 GPU 实际会创建能够覆盖：

```text
112 × 112
```

范围的线程。

但真正合法的矩阵只有：

```text
100 × 100
```

因此最后一些线程会得到：

```text
row >= 100
或者
col >= 100
```

这些线程必须直接退出，否则会访问越界内存。

所以：

```cpp
if (row >= M || col >= N) {
    return;
}
```

的作用就是：

> **让超出矩阵范围的线程什么都不做。**

---

# 6. 第四部分：为什么需要 `sum`

```cpp
float sum = 0.0f;
```

这个变量用于保存：

```text
C[row][col]
```

的累加结果。

矩阵乘法中的一个 C 元素不是一次乘法算出来的，而是：

```text
A 的一整行
和
B 的一整列
```

做点积。

所以需要不断：

```text
sum += ...
```

最终：

```text
sum = C[row][col]
```

在正常情况下，`sum` 这样的线程局部标量通常会被编译器放入寄存器中。

所以从数据流角度看：

```text
Global Memory 中 A/B
    ↓
线程读取
    ↓
寄存器 sum 中累加
    ↓
Global Memory 中 C
```

---

# 7. 第五部分：最核心的 K 循环

```cpp
for (int k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * N + col];
}
```

这就是 GEMM 真正完成矩阵乘法的地方。

---

## 7.1 一个线程固定什么

当前线程已经确定：

```text
row
col
```

所以它负责的输出是：

```text
C[row][col]
```

在整个 `for` 循环里：

```text
row 不变
col 不变
k 不断变化
```

---

## 7.2 A 读取什么

```cpp
A[row * K + k]
```

表示：

```text
A[row][k]
```

也就是说：

> 当前线程沿着 A 的第 `row` 行不断往右读取。

---

## 7.3 B 读取什么

```cpp
B[k * N + col]
```

表示：

```text
B[k][col]
```

也就是：

> 当前线程沿着 B 的第 `col` 列不断往下读取。

---

## 7.4 为什么乘起来就得到 C

每一轮：

```cpp
A[row][k] * B[k][col]
```

然后把所有 K 位置加起来：

```text
A[row][0] × B[0][col]
+
A[row][1] × B[1][col]
+
...
```

这正是：

```text
C[row][col]
```

的定义。

---

# 8. 第六部分：为什么一维数组要写成这种下标

GPU 显存里的矩阵通常通过一维连续数组保存。

例如一个矩阵：

```text
1 2 3
4 5 6
```

在内存里实际是：

```text
1 2 3 4 5 6
```

如果矩阵有 `cols` 列：

```text
matrix[row][col]
```

对应的一维位置就是：

```text
row * cols + col
```

所以：

### A 有 K 列

```cpp
A[row * K + k]
```

### B 有 N 列

```cpp
B[k * N + col]
```

### C 有 N 列

```cpp
C[row * N + col]
```

面试时必须能解释这个公式：

> 前面有 `row` 个完整行，每行有 `cols` 个元素，所以先跳过 `row × cols` 个元素，再向右移动 `col` 个元素。

---

# 9. 第七部分：把结果写回 Global Memory

```cpp
C[row * N + col] = sum;
```

K 循环完成后：

```text
sum
```

就是：

```text
C[row][col]
```

所以直接写回 GPU Global Memory。

完整数据流：

```text
A[row][k] ──┐
             ├─ 乘法 ─→ sum
B[k][col] ──┘             │
                          │ K 次累加
                          ▼
                    C[row][col]
```

---

# 10. Kernel 怎么启动

推荐面试时顺手写一个简单 wrapper：

```cpp
void launch_gemm(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    dim3 block(16, 16);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    gemm_naive<<<grid, block>>>(
        A,
        B,
        C,
        M,
        N,
        K
    );
}
```

---

# 11. 为什么 `block` 设为 `16 × 16`

```cpp
dim3 block(16, 16);
```

表示一个 Block 有：

```text
16 × 16 = 256
```

个线程。

这是一个非常常见、容易解释的二维线程块大小。

选择它的理由不是：

```text
16×16 永远最快
```

而是：

- 256 线程是常见 Block 规模；
- 二维映射矩阵非常直观；
- 方便面试手写；
- 对基础 GEMM 足够合理。

面试中可以说：

> 这里主要为了代码清晰使用 `16×16`，真实高性能实现会根据寄存器、Shared Memory、Occupancy 和具体 GPU 做调优。

---

# 12. Grid 为什么这样算

```cpp
dim3 grid(
    (N + block.x - 1) / block.x,
    (M + block.y - 1) / block.y
);
```

这是向上取整除法。

因为：

```text
x 方向覆盖 N 列
y 方向覆盖 M 行
```

所以：

```text
grid.x = ceil(N / block.x)
grid.y = ceil(M / block.y)
```

---

# 13. 完整的面试手撕版本

真正面试时，我建议记住下面这整段。

```cpp
#include <cuda_runtime.h>

__global__ void gemm_naive(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;

    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

void launch_gemm(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    dim3 block(16, 16);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    gemm_naive<<<grid, block>>>(
        A,
        B,
        C,
        M,
        N,
        K
    );
}
```

---

# 14. 面试时怎么口头解释这段代码

可以直接这样回答：

> 我先写最基础的 Naive GEMM。假设 A 是 `M×K`、B 是 `K×N`，输出 C 是 `M×N`。我用二维 Grid 和二维 Block，让一个 CUDA 线程负责 C 中的一个元素。线程通过 `blockIdx`、`blockDim` 和 `threadIdx` 得到自己的 `row` 和 `col`，然后做边界判断。对于合法线程，它沿 K 维循环，每次读取 `A[row][k]` 和 `B[k][col]` 做乘加，结果先存在当前线程的局部变量 `sum` 中，最后写到 `C[row][col]`。这里默认矩阵按行优先连续存储，所以二维坐标转换成一维地址就是 `row * 列数 + col`。线程块我简单用 `16×16`，Grid 按 M、N 向上取整覆盖整个输出矩阵。

---

# 15. 为什么这一版性能不好

面试官很可能继续问：

> 这版有什么问题？

你可以回答三个最核心的问题。

## 15.1 Global Memory 重复访问非常严重

很多线程会反复读取同一块 A / B 数据，但当前实现没有使用 Shared Memory 做显式复用。

---

## 15.2 一个线程只算一个输出元素

当前：

```text
Thread → 1 个 C 元素
```

所以每个线程只有一个累加器：

```cpp
float sum;
```

寄存器级数据复用能力很弱。

高性能 GEMM 往往：

```text
Thread → 多个 C 元素
```

也就是 Thread Tile / Register Blocking。

---

## 15.3 没有 Tiling

当前数据流：

```text
A/B：Global Memory
        ↓
线程直接读取
        ↓
sum
```

而高性能 GEMM 会变成：

```text
Global Memory
    ↓
Shared Memory Tile
    ↓
Thread Registers
    ↓
多个结果寄存器
```

---

# 16. 从这个版本怎么一步步优化

```text
第 0 版：Naive GEMM
一个 Thread 算一个 C 元素
A/B 直接从 Global Memory 读取

        ↓

第 1 版：Shared Memory Tiling
一个 Block 负责一个 C Tile
A/B Tile 搬入 Shared Memory 后重复使用

        ↓

第 2 版：Thread Tile / Register Blocking
一个线程算多个 C 元素
提高寄存器级数据复用

        ↓

第 3 版：Vectorized Memory Access
float4 等方式减少访存指令

        ↓

第 4 版：Warp Tile
进一步规划 Warp 的计算区域

        ↓

第 5 版：Double Buffer + cp.async
让下一 Tile 的数据搬运和当前 Tile 的计算重叠

        ↓

第 6 版：Tensor Core / MMA
使用专用矩阵乘硬件
```

这也正好对应高性能 GEMM 常见的演进路线。

---

# 17. 当前项目和手撕版本的对应关系

最简单版本：

```cpp
float sum = 0;

for (int k = 0; k < K; ++k) {
    sum += A[row*K+k] * B[k*N+col];
}

C[row*N+col] = sum;
```

复杂优化版本只是把它逐层改造成：

```text
一个 Thread 不再只有一个 sum
    ↓
而是多个 res[]

A/B 不再每次直接从 Global Memory 读取
    ↓
而是 Global → Shared → reg_m/reg_n

一个 Thread 不再只负责一个 C
    ↓
而是负责一个 Thread Tile

一个 Block 不再只是线程集合
    ↓
而是负责一个 Block Tile

Block 内再划分 Warp Tile

K 方向不再一次一个元素简单读取
    ↓
而是 BK 分块 + 双缓冲 + cp.async
```

因此，高性能 GEMM 的本质仍然是：

```text
for k:
    C += A × B
```

只是围绕：

```text
谁算
数据放哪里
数据怎么复用
数据什么时候搬
```

做了大量优化。

---

# 18. 面试最容易写错的地方

## 18.1 把 M/N 搞反

正确：

```text
A：M×K
B：K×N
C：M×N
```

因此：

```cpp
row < M
col < N
```

---

## 18.2 A 的下标写错

A 有 K 列：

```cpp
A[row * K + k]
```

---

## 18.3 B 的下标写错

B 有 N 列：

```cpp
B[k * N + col]
```

---

## 18.4 C 的下标写错

C 有 N 列：

```cpp
C[row * N + col]
```

---

## 18.5 Grid 的 M/N 搞反

因为：

```text
x → col → N
y → row → M
```

所以：

```cpp
grid.x = ceil(N / block.x);
grid.y = ceil(M / block.y);
```

---

## 18.6 忘记边界判断

必须：

```cpp
if (row >= M || col >= N) {
    return;
}
```

---

# 19. 面试时只需要死记的核心逻辑

真正需要熟练到肌肉记忆的是：

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

if (row >= M || col >= N) return;

float sum = 0.0f;

for (int k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * N + col];
}

C[row * N + col] = sum;
```

只要你能不看答案写出这几行，就已经具备最基础的 CUDA GEMM 手撕能力。

---

# 20. 最终记忆模型

把整个 Naive GEMM 压缩成：

```text
一个线程
    ↓
找到自己的 row / col
    ↓
负责 C[row][col]
    ↓
沿 K 方向循环
    ↓
A 的第 row 行
×
B 的第 col 列
    ↓
寄存器 sum 中累加
    ↓
写回 C[row][col]
```

最核心的一句话：

> **一个线程算 C 的一个元素，A 的一行乘 B 的一列，在 K 维做点积。**


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-gemm算子|模块-gemm算子]]

%% 项目关联导航：结束 %%
