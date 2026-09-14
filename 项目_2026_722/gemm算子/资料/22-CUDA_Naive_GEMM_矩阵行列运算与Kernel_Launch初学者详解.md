# CUDA Naive GEMM：从矩阵行列运算到 Kernel Launch 的完整初学者讲解

> 这份文档只讲下面这份最基础的 CUDA GEMM：
>
> ```cpp
> #include <cuda_runtime.h>
>
> __global__ void gemm_naive(
>     const float* A,
>     const float* B,
>     float* C,
>     int M,
>     int N,
>     int K
> ) {
>     int row = blockIdx.y * blockDim.y + threadIdx.y;
>     int col = blockIdx.x * blockDim.x + threadIdx.x;
>
>     if (row >= M || col >= N) {
>         return;
>     }
>
>     float sum = 0.0f;
>
>     for (int k = 0; k < K; ++k) {
>         sum += A[row * K + k] * B[k * N + col];
>     }
>
>     C[row * N + col] = sum;
> }
>
> void launch_gemm(
>     const float* A,
>     const float* B,
>     float* C,
>     int M,
>     int N,
>     int K
> ) {
>     dim3 block(16, 16);
>
>     dim3 grid(
>         (N + block.x - 1) / block.x,
>         (M + block.y - 1) / block.y
>     );
>
>     gemm_naive<<<grid, block>>>(
>         A,
>         B,
>         C,
>         M,
>         N,
>         K
>     );
> }
> ```
>
> 目标：你看完以后，要能真正回答：
>
> 1. A、B、C 分别有多少行多少列；
> 2. 为什么 A 用 `row*K+k`；
> 3. 为什么 B 用 `k*N+col`；
> 4. 为什么 C 用 `row*N+col`；
> 5. 一个 CUDA 线程到底算 C 的哪个位置；
> 6. `block(16,16)` 是什么意思；
> 7. `grid(...)` 为什么这么算；
> 8. `gemm_naive<<<grid, block>>>()` 到底发生了什么。

---

# 1. 先只看数学：GEMM 到底在算什么

假设：

```text
A：M × K
B：K × N
C：M × N
```

也就是：

```text
A 有 M 行、K 列
B 有 K 行、N 列
C 有 M 行、N 列
```

矩阵乘法：

```text
C = A × B
```

---

# 2. 为什么 C 是 `M × N`

看：

```text
A：M × K
B：K × N
```

中间的 K 必须相同，因为：

```text
A 的每一行长度 = K
B 的每一列长度 = K
```

做点积时，两边元素数量必须一致。

所以：

```text
(M × K) × (K × N)
```

最后得到：

```text
M × N
```

即：

```text
C：M × N
```

---

# 3. 用一个具体矩阵例子理解

假设：

```text
M = 2
K = 3
N = 4
```

那么：

```text
A：2 × 3
B：3 × 4
C：2 × 4
```

例如：

```text
A =
[ 1  2  3 ]
[ 4  5  6 ]
```

```text
B =
[  7   8   9  10 ]
[ 11  12  13  14 ]
[ 15  16  17  18 ]
```

C 最终有 2 行 4 列：

```text
C =
[ ?  ?  ?  ? ]
[ ?  ?  ?  ? ]
```

---

# 4. C 中一个元素是怎么计算出来的

假设我们要算：

```text
C[1][2]
```

也就是：

```text
第 1 行，第 2 列
```

矩阵乘法规则：

```text
C 的第 row 行第 col 列
=
A 的第 row 行
点乘
B 的第 col 列
```

这里：

```text
row = 1
col = 2
```

所以：

A 第 1 行：

```text
[4 5 6]
```

B 第 2 列：

```text
[ 9 ]
[13 ]
[17 ]
```

于是：

```text
C[1][2]
=
4×9 + 5×13 + 6×17
```

即：

```text
C[row][col]
=
Σ A[row][k] × B[k][col]
```

这里的 k：

```text
k = 0, 1, 2, ..., K-1
```

所以：

```cpp
for (int k = 0; k < K; ++k) {
    ...
}
```

本质就是：

> 沿着 K 维，把 A 的这一行和 B 的这一列逐个元素相乘再累加。

---

# 5. 为什么 `row` 和 `col` 是固定的，而 `k` 在变化

一个 CUDA 线程只负责：

```text
C[row][col]
```

所以它的：

```text
row 固定
col 固定
```

但是为了算这个 C 元素，它必须遍历整个 K：

```text
A[row][0] × B[0][col]
A[row][1] × B[1][col]
A[row][2] × B[2][col]
...
```

所以：

```text
row：固定
col：固定
k：从 0 变化到 K-1
```

这就是：

```cpp
for (int k = 0; k < K; ++k)
```

存在的原因。

---

# 6. 为什么代码里不是 `A[row][k]`

Kernel 参数写的是：

```cpp
const float* A
const float* B
float* C
```

这说明 A、B、C 在代码里都是：

```text
一维连续内存
```

不是 C++ 真正的二维数组。

所以虽然我们数学上想象：

```text
A[row][k]
B[k][col]
C[row][col]
```

实际上必须自己把二维坐标转换成一维下标。

---

# 7. 二维矩阵是怎么放在一维内存里的

假设：

```text
A =
[ 1 2 3 ]
[ 4 5 6 ]
```

数学上：

```text
2 行 × 3 列
```

但内存里按行连续放：

```text
下标：
 0   1   2   3   4   5

数据：
 1   2   3   4   5   6
```

也就是：

```text
第 0 行 → 1 2 3
第 1 行 → 4 5 6
```

连续拼起来。

这种方式叫：

```text
Row-major（行优先）
```

---

# 8. 二维转一维的万能公式

如果一个矩阵：

```text
有 cols 列
```

那么：

```text
matrix[row][col]
```

在一维内存中的位置就是：

```text
row * cols + col
```

为什么？

因为：

```text
前面已经有 row 个完整行
```

每行有：

```text
cols 个元素
```

所以先跳过：

```text
row × cols
```

个元素。

然后再向右移动：

```text
col
```

个位置。

所以：

```text
一维位置 = row × cols + col
```

---

# 9. 为什么 A 是 `A[row * K + k]`

A 的形状：

```text
A：M × K
```

也就是说：

```text
A 每一行有 K 个元素
```

数学上我们想访问：

```text
A[row][k]
```

套公式：

```text
row × 每行列数 + 当前列
```

对于 A：

```text
每行列数 = K
当前列 = k
```

所以：

```text
A[row][k]
```

变成：

```cpp
A[row * K + k]
```

---

# 10. 用数字验证 A 的下标

还是：

```text
A =
[1 2 3]
[4 5 6]
```

这里：

```text
K = 3
```

假设：

```text
row = 1
```

那么：

### k = 0

```text
A[row*K+k]
=
A[1×3+0]
=
A[3]
=
4
```

对应：

```text
A[1][0]
```

### k = 1

```text
A[1×3+1]
=
A[4]
=
5
```

对应：

```text
A[1][1]
```

### k = 2

```text
A[1×3+2]
=
A[5]
=
6
```

对应：

```text
A[1][2]
```

所以随着 k 增加：

```text
A[row*K+k]
```

就是：

> 沿 A 的第 row 行，从左往右走。

---

# 11. 为什么 B 是 `B[k * N + col]`

B 的形状：

```text
B：K × N
```

也就是说：

```text
B 每一行有 N 个元素
```

数学上需要：

```text
B[k][col]
```

套一维公式：

```text
row × 每行列数 + col
```

对于 B：

```text
row = k
每行列数 = N
col = col
```

所以：

```text
B[k][col]
```

变成：

```cpp
B[k * N + col]
```

---

# 12. 用数字验证 B 的下标

B：

```text
B =
[ 7  8  9 10]
[11 12 13 14]
[15 16 17 18]
```

这里：

```text
N = 4
```

假设：

```text
col = 2
```

我们要沿 B 的第 2 列往下取：

```text
9
13
17
```

### k = 0

```text
B[0×4+2]
=
B[2]
=
9
```

### k = 1

```text
B[1×4+2]
=
B[6]
=
13
```

### k = 2

```text
B[2×4+2]
=
B[10]
=
17
```

所以：

```cpp
B[k * N + col]
```

随着 k 增加，就是：

> 沿 B 的第 col 列，从上往下走。

---

# 13. 为什么 C 是 `C[row * N + col]`

C 的形状：

```text
C：M × N
```

也就是：

```text
C 每一行有 N 个元素
```

数学上当前线程负责：

```text
C[row][col]
```

套一维公式：

```text
row × 每行列数 + col
```

C 每行有 N 个元素，所以：

```text
C[row][col]
```

变成：

```cpp
C[row * N + col]
```

---

# 14. 把 ABC 三个下标放在一起

记住最重要的一条：

> 一维下标永远是：`行号 × 当前矩阵列数 + 列号`

于是：

| 矩阵 | 形状 | 二维访问 | 一维访问 |
|---|---|---|---|
| A | `M×K` | `A[row][k]` | `A[row*K+k]` |
| B | `K×N` | `B[k][col]` | `B[k*N+col]` |
| C | `M×N` | `C[row][col]` | `C[row*N+col]` |

所以不要死记：

```text
A 乘 K
B 乘 N
C 乘 N
```

真正应该记：

```text
看当前矩阵每行有多少列
```

---

# 15. 现在完整理解这个循环

代码：

```cpp
float sum = 0.0f;

for (int k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * N + col];
}

C[row * N + col] = sum;
```

翻译成人话：

> 当前线程已经确定自己负责 C 的第 `row` 行、第 `col` 列。  
> 它让 k 从 0 遍历到 K-1，每一次从 A 的第 row 行取一个元素，同时从 B 的第 col 列取一个对应元素，两者相乘并累加到 sum。  
> 等整个 K 维处理完成，sum 就是 `C[row][col]`，最后把它写回 C。

---

# 16. 用完整例子模拟一次线程执行

假设：

```text
A =
[1 2 3]
[4 5 6]

B =
[ 7  8]
[ 9 10]
[11 12]
```

那么：

```text
M = 2
K = 3
N = 2
```

假设某个线程通过索引计算得到：

```text
row = 1
col = 0
```

这个线程负责：

```text
C[1][0]
```

---

## 第一次循环：k = 0

A：

```text
A[row*K+k]
=
A[1×3+0]
=
A[3]
=
4
```

B：

```text
B[k*N+col]
=
B[0×2+0]
=
B[0]
=
7
```

所以：

```text
sum = 0 + 4×7 = 28
```

---

## 第二次循环：k = 1

A：

```text
A[1×3+1]
=
A[4]
=
5
```

B：

```text
B[1×2+0]
=
B[2]
=
9
```

所以：

```text
sum = 28 + 5×9 = 73
```

---

## 第三次循环：k = 2

A：

```text
A[1×3+2]
=
A[5]
=
6
```

B：

```text
B[2×2+0]
=
B[4]
=
11
```

得到：

```text
sum = 73 + 6×11
= 139
```

最后：

```cpp
C[row * N + col] = sum;
```

即：

```text
C[1×2+0]
=
C[2]
=
139
```

二维来看就是：

```text
C[1][0] = 139
```

---

# 17. 接下来解释 `launch_gemm()`

代码：

```cpp
void launch_gemm(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    ...
}
```

这个函数本身：

```text
不是 GPU Kernel
```

它是一个普通 CPU 函数。

它的任务不是做矩阵乘法，而是：

> **在 CPU 端准备好 Grid 和 Block，然后启动 GPU 上的 `gemm_naive` Kernel。**

可以把它理解成：

```text
launch_gemm()
=
Kernel 的启动器
```

---

# 18. `dim3 block(16,16)` 是什么意思

代码：

```cpp
dim3 block(16, 16);
```

`dim3` 是 CUDA 用来描述三维尺寸的数据类型。

这里相当于：

```text
block.x = 16
block.y = 16
block.z = 1
```

也就是一个 Thread Block 里面：

```text
x 方向有 16 个线程
y 方向有 16 个线程
```

总线程数：

```text
16 × 16 = 256 个线程
```

---

# 19. 一个 Block 在矩阵中覆盖多少区域

因为我们规定：

```text
x → col
y → row
```

所以：

```text
16 × 16 Thread
```

正好可以覆盖 C 中一个：

```text
16 行 × 16 列
```

的区域。

示意：

```text
一个 Block
+----------------+
| 16×16 Threads  |
|                |
| 每个 Thread    |
| 算一个 C 元素  |
+----------------+
```

所以一个 Block 最多负责：

```text
256 个 C 元素
```

---

# 20. 为什么还需要 Grid

一个 Block 只有：

```text
16×16
```

个线程。

如果 C 很大，例如：

```text
C = 1000 × 1000
```

一个 Block 根本覆盖不完。

于是需要很多 Block：

```text
Grid
    ↓
Block 0
Block 1
Block 2
...
```

所以：

```text
Grid = 很多 Block 的集合
```

---

# 21. Grid 的 x 方向为什么看 N

代码：

```cpp
(N + block.x - 1) / block.x
```

这是：

```text
grid.x
```

Grid x 方向负责：

```text
C 的列
```

C 一共有：

```text
N 列
```

每个 Block x 方向能覆盖：

```text
block.x = 16 列
```

所以需要：

```text
ceil(N / 16)
```

个 Block。

---

# 22. 为什么公式是 `(N + block.x - 1) / block.x`

这是整数里的向上取整除法。

正常数学：

```text
ceil(N / block.x)
```

但 C++ 整数除法会直接向下取整。

例如：

```text
N = 100
block.x = 16
```

普通整数除法：

```text
100 / 16 = 6
```

但 6 个 Block 只能覆盖：

```text
6×16 = 96 列
```

还少 4 列。

所以需要 7 个 Block。

写成：

```text
(100 + 16 - 1) / 16
=
115 / 16
=
7
```

因此：

```cpp
(N + block.x - 1) / block.x
```

就是：

```text
ceil(N / block.x)
```

---

# 23. Grid 的 y 方向为什么看 M

同理：

```cpp
(M + block.y - 1) / block.y
```

计算：

```text
grid.y
```

因为：

```text
y → row
```

而 C 一共有：

```text
M 行
```

每个 Block y 方向覆盖：

```text
16 行
```

所以：

```text
grid.y = ceil(M / 16)
```

---

# 24. 一个完整 Grid 示例

假设：

```text
M = 100
N = 70
```

Block：

```text
16 × 16
```

则：

```text
grid.x = ceil(70/16) = 5
grid.y = ceil(100/16) = 7
```

所以：

```text
Grid = 5 × 7 Blocks
```

总共：

```text
35 个 Block
```

每个 Block：

```text
256 个线程
```

GPU 会创建足够多线程覆盖整个 C。

---

# 25. 为什么线程可能会超出矩阵范围

上面的例子：

```text
N = 70
```

但：

```text
5 个 Block × 16 列
=
80 列
```

所以最后一个 Block 会覆盖：

```text
64 ~ 79 列
```

但真正合法只有：

```text
64 ~ 69
```

于是：

```text
70 ~ 79
```

这些线程是多出来的。

所以 Kernel 里必须：

```cpp
if (row >= M || col >= N) {
    return;
}
```

这些多余线程直接退出。

---

# 26. `gemm_naive<<<grid, block>>>()` 是什么意思

代码：

```cpp
gemm_naive<<<grid, block>>>(
    A,
    B,
    C,
    M,
    N,
    K
);
```

这是 CUDA 特有的 Kernel Launch 语法。

普通 C++ 调用：

```cpp
func(a, b);
```

CUDA Kernel 调用：

```cpp
kernel<<<grid, block>>>(...);
```

---

# 27. `<<<grid, block>>>` 分别控制什么

```text
grid
```

告诉 GPU：

> 我要启动多少个 Thread Block。

```text
block
```

告诉 GPU：

> 每个 Block 里面有多少个 Thread。

例如：

```cpp
dim3 block(16,16);
dim3 grid(5,7);
```

那么 GPU 会启动：

```text
7 行 × 5 列 Blocks
```

每个 Block 里面：

```text
16 × 16 Threads
```

---

# 28. Kernel 启动后每个线程执行同一份代码

非常关键：

```cpp
gemm_naive<<<grid, block>>>(...)
```

不是只执行一次 `gemm_naive()`。

而是：

> GPU 创建大量线程，每一个线程都会执行一遍 `gemm_naive()`。

区别只在于每个线程自己的：

```text
blockIdx
threadIdx
```

不同。

所以每个线程算出来：

```cpp
row
col
```

也不同。

于是不同线程分别负责：

```text
C[0][0]
C[0][1]
C[0][2]
...
C[1][0]
C[1][1]
...
```

最终大家一起把 C 算完。

---

# 29. 从 `launch_gemm()` 到所有线程的完整流程

```text
CPU 调用 launch_gemm()
    ↓
设置 block = 16×16
    ↓
根据 M/N 计算 grid
    ↓
调用 gemm_naive<<<grid,block>>>
    ↓
GPU 启动很多 Block
    ↓
每个 Block 启动很多 Thread
    ↓
每个 Thread 计算自己的 row / col
    ↓
若越界则退出
    ↓
沿 K 维做 A 行 × B 列
    ↓
sum 得到一个 C 元素
    ↓
写回 C[row][col]
```

---

# 30. 最终完整脑内模型

你可以把整个程序想成：

```text
A[M×K]       B[K×N]
   \           /
    \         /
     \       /
      ↓     ↓
    每个 CUDA Thread
         ↓
   确定 row / col
         ↓
固定 C[row][col]
         ↓
k = 0 ... K-1
         ↓
A[row][k]
    ×
B[k][col]
         ↓
sum 中不断累加
         ↓
C[row][col] = sum
```

而 `launch_gemm()` 做的是：

```text
决定：
一共有多少个 Block
每个 Block 有多少 Thread
```

也就是：

```text
launch_gemm()
    ↓
Grid
    ↓
Block
    ↓
Thread
    ↓
C 中一个元素
```

---

# 31. 面试时你可以怎么解释

可以这样说：

> 这个最简单的 CUDA GEMM 是让一个线程负责输出矩阵 C 的一个元素。A 是 `M×K`，B 是 `K×N`，所以 C 是 `M×N`。线程通过二维 Grid 和二维 Block 得到全局的 `row` 和 `col`，也就是它负责的 C 位置。然后沿 K 维遍历，读取 `A[row][k]` 和 `B[k][col]` 做乘加。因为矩阵实际在显存里按行优先存成一维数组，所以 A 的二维下标会转成 `row*K+k`，B 是 `k*N+col`，C 是 `row*N+col`。Host 端的 `launch_gemm` 用 `16×16` 的线程块，再按 M/N 向上取整得到 Grid 大小，从而让足够多的线程覆盖整个输出矩阵。

---

# 32. 最后只记住 4 个核心结论

### 结论 1

```text
A：M×K
B：K×N
C：M×N
```

### 结论 2

```text
C[row][col]
=
Σ A[row][k] × B[k][col]
```

### 结论 3

二维转一维：

```text
matrix[row][col]
=
matrix[row × cols + col]
```

所以：

```text
A[row][k] → A[row*K+k]
B[k][col] → B[k*N+col]
C[row][col] → C[row*N+col]
```

### 结论 4

Kernel Launch：

```text
Grid 决定多少个 Block
Block 决定每个 Block 多少 Thread
每个 Thread 负责一个 C 元素
```

一旦这四点完全理解，这份 Naive CUDA GEMM 就不需要死记了，而是可以自己推导出来。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-gemm算子|模块-gemm算子]]

%% 项目关联导航：结束 %%
