# GEMM 项目数据流动详解：Global Memory → Shared Memory → Registers → Global Memory

> 本文建立在你已经理解“完整矩阵 → Block Tile → Warp Tile → Thread Tile”的基础上，继续解释当前 GEMM 项目中：
>
> - A、B、C 最开始存在哪里；
> - 一个 Block 如何取得自己需要的 A/B 数据；
> - 数据怎样从全局内存进入共享内存；
> - 数据怎样从共享内存进入线程寄存器；
> - 线程如何在寄存器中完成乘加累积；
> - 最终结果怎样从寄存器写回全局内存；
> - `cp.async`、双缓冲、向量化、同步、边界路径等机制分别发挥什么作用。
>
> 重点关联源码：
>
> - `tests/test_perf_gemm.cpp`
> - `src/launcher.cu`
> - `src/kernels/gemm_warp_tile.cu`
> - `include/gemm/kernels/gemm_warp_tile_config.h`

---

# 1. 当前项目的数据流总览

整个数据流可以先压缩成下面这条主线：

```text
CPU Host Memory
A / B / C
    │
    │ cudaMemcpy HostToDevice
    ▼
GPU Global Memory
d_A / d_B / d_C
    │
    │ 当前 Block 只选择自己需要的 A Tile 和 B Tile
    │ fast path：cp.async 16B
    │ edge path：带边界判断的标量加载
    ▼
Shared Memory
AS[2][BK][BM]
BS[2][BK][BN]
    │
    │ compute() 按当前 K 位置读取
    ▼
Thread Registers
reg_m[]
reg_n[]
    │
    │ 外积式 FP32 乘加
    ▼
Thread Registers
res[]
    │
    │ fast path：float4 向量化写回
    │ edge path：逐元素边界检查写回
    ▼
GPU Global Memory
d_C
    │
    │ cudaMemcpy DeviceToHost
    ▼
CPU Host Memory
C
```

对应一句话：

> A、B 先从 CPU 内存复制到 GPU 全局内存；Kernel 中每个 Block 沿 K 方向分批把自己的 A/B Tile 搬到双缓冲共享内存；线程再把当前计算所需的 A/B 数据读入 `reg_m` 和 `reg_n`，在寄存器 `res` 中完成外积累加；K 维全部处理完后，结果按照 `alpha × res + beta × C` 写回全局内存。

---

# 2. 各层存储分别放什么

| 存储层次 | 当前项目中的变量 | 存放内容 | 可见范围 |
|---|---|---|---|
| CPU Host Memory | 测试代码中的 `A/B/C` | 测试输入、参考结果、最终结果 | CPU |
| GPU Global Memory | `A/B/C` Kernel 参数，测试中为 `d_A/d_B/d_C` | 完整矩阵 | 全部 GPU 线程 |
| Shared Memory | `AS[2]`、`BS[2]` | 当前 Block 当前/下一 K Tile | 同一 Block |
| Thread Registers | `reg_m[]` | 当前线程本次乘加使用的 A 数据 | 当前线程 |
| Thread Registers | `reg_n[]` | 当前线程本次乘加使用的 B 数据 | 当前线程 |
| Thread Registers | `res[]` | 当前线程负责的 C 输出累加器 | 当前线程 |

最重要的关系是：

```text
完整 A/B：放 Global Memory
当前 Block 要复用的 A/B Tile：放 Shared Memory
当前线程这一拍要计算的数据：放 reg_m/reg_n
当前线程长期累加的输出：放 res
```

---

# 3. Host 端：数据如何进入 GPU Global Memory

在测试程序中，Host 先申请并初始化：

```text
A[M×K]
B[K×N]
C[M×N]
```

然后在 GPU 上申请：

```cpp
cudaMalloc(&d_A, ...);
cudaMalloc(&d_B, ...);
cudaMalloc(&d_C, ...);
```

接着：

```cpp
cudaMemcpy(d_A, A, ..., cudaMemcpyHostToDevice);
cudaMemcpy(d_B, B, ..., cudaMemcpyHostToDevice);
```

因此 Kernel 启动前：

```text
d_A → GPU Global Memory 中完整 A
d_B → GPU Global Memory 中完整 B
d_C → GPU Global Memory 中完整 C
```

调用：

```cpp
launch_gemm(M, N, K, alpha, d_A, d_B, beta, d_C);
```

在 `src/launcher.cu` 中最终进入：

```cpp
launch_gemm_warp_tile(...);
```

再根据形状选择配置，启动：

```cpp
gemm_warp_tile_kernel<Cfg><<<gridDim, blockDim>>>(...);
```

---

# 4. Kernel 入口：每个 Block 先定位自己的数据区域

Kernel 参数：

```cpp
void gemm_warp_tile_kernel(
    int M, int N, int K,
    float alpha,
    const float *A,
    const float *B,
    float beta,
    float *C
)
```

其中：

```text
A、B、C 都是 Global Memory 指针
```

每个 Block 通过：

```cpp
const uint c_row = blockIdx.y;
const uint c_col = blockIdx.x;
```

确定自己负责 C 的哪个 `BM × BN` Tile。

对应 A/B 的起始指针：

```cpp
const float *A_blk = A + c_row * BM * K;
const float *B_blk = B + c_col * BN;
```

可以理解为：

```text
A_blk：
指向当前 Block 对应的 A 行块起点

B_blk：
指向当前 Block 对应的 B 列块起点
```

当前 Block 最终要计算：

```text
C[
  c_row×BM : c_row×BM+BM,
  c_col×BN : c_col×BN+BN
]
```

但它不能一次得到结果，还需要沿 K 方向不断取 A/B 数据。

---

# 5. 为什么必须沿 K 方向分批流动

矩阵乘法：

```text
C[m,n] = Σ A[m,k] × B[k,n]
```

对一个 C Block Tile 来说，需要完整 K 维的数据。

项目设置：

```text
BK = 8
```

表示每次只处理 K 方向的 8 个位置。

Default 配置下，每次所需数据为：

```text
A Tile：BM × BK = 128 × 8
B Tile：BK × BN = 8 × 256
```

整体过程：

```text
K = 0~7       → 加载第 0 个 A/B Tile → 计算
K = 8~15      → 加载第 1 个 A/B Tile → 计算
K = 16~23     → 加载第 2 个 A/B Tile → 计算
...
直到 K 全部处理完成
```

所以：

> 输出 Tile 在整个 Kernel 期间保持不变；输入 A/B 则沿 K 方向一批一批流过 Shared Memory 和寄存器。

---

# 6. Shared Memory 的实际布局

Kernel 中：

```cpp
__shared__ float AS[2][A_STRIDE * BK];
__shared__ float BS[2][BK * BN];
```

其中：

```cpp
constexpr int A_STRIDE = BM;
```

逻辑上：

```text
AS[2][BK][BM]
BS[2][BK][BN]
```

第一个维度 `2` 用于双缓冲。

---

## 6.1 AS 的逻辑布局

源码读取方式：

```cpp
AS[d * A_STRIDE + m]
```

也就是：

```text
AS[d][m]
```

其中：

- `d`：当前 K Tile 内的 K 坐标，范围 `0~BK-1`
- `m`：当前 Block Tile 内的 M 坐标，范围 `0~BM-1`

因此 AS 的逻辑形状是：

```text
[BK][BM]
```

虽然原始 A Tile 数学形状是：

```text
[BM][BK]
```

放进 Shared Memory 后按 `[BK][BM]` 方式访问，相当于为了计算阶段进行布局重组。

示意图：

```text
Global A Tile（数学视图 BM×BK）

             K
        0 1 2 ... 7
M=0     *
M=1     *
...
M=127   *

进入 Shared Memory 后：

AS[d][m]

             M
        0 1 2 ... 127
d=0     *
d=1     *
...
d=7     *
```

这样在固定 `d` 时，线程可以从 AS 中取得不同 M 位置的数据。

---

## 6.2 BS 的逻辑布局

源码读取方式：

```cpp
BS[d * BN + n]
```

即：

```text
BS[d][n]
```

它的逻辑形状：

```text
[BK][BN]
```

这与 B Tile 的数学形状一致。

---

# 7. Fast Path：Global Memory 如何进入 Shared Memory

Kernel 判断：

```cpp
const bool fast =
    (m_edge == BM) &&
    (n_edge == BN) &&
    (K % 4 == 0) &&
    (N % 4 == 0);
```

满足条件时进入 fast path。

这里要求：

- 当前 Block 的 M Tile 完整；
- 当前 Block 的 N Tile 完整；
- K 可以按 4 个 float 对齐；
- B/C 行宽 N 可以按 4 个 float 对齐。

---

# 8. `load_tile_async()`：Block 内线程协作搬运

Fast Path 使用：

```cpp
load_tile_async(...);
```

内部调用：

```cpp
cp_async16(...)
```

对应 PTX：

```text
cp.async.ca.shared.global.L2::128B
```

它表示：

> 异步地把 16 字节数据从 Global Memory 复制到 Shared Memory。

16 字节就是：

```text
4 个 float
```

因此当前项目的基本搬运粒度是：

```text
float4
```

---

## 8.1 为什么每次搬 4 个 float

相对于逐个 `float`：

```text
一次 16B 搬运
```

可以：

- 减少加载指令数量；
- 利用连续内存；
- 与对齐访问配合；
- 更适合形成高效的全局内存事务。

源码中线程的搬运位置由：

```cpp
inner_row_a
inner_col_a
inner_row_b
inner_col_b
```

确定。

---

## 8.2 B 的搬运方式比较直观

源码：

```cpp
cp_async16(
    dst,
    B_tile + (inner_row_b+i)*N + inner_col_b,
    true
);
```

也就是一个线程搬：

```text
B[k][n : n+3]
```

4 个连续 float。

Shared 目标：

```cpp
BS[(inner_row_b+i)*BN + inner_col_b]
```

即：

```text
BS[k][n : n+3]
```

因此 B 的路线是：

```text
Global B[k][连续4列]
    ↓ cp.async 16B
Shared BS[k][连续4列]
```

---

## 8.3 Block 内所有线程共同完成整个 Tile

每个线程只负责部分 16B 搬运。

所有线程合起来覆盖：

```text
A Tile：BM × BK
B Tile：BK × BN
```

这是一种典型的协作加载：

```text
Thread 0 搬一段
Thread 1 搬一段
Thread 2 搬一段
...
整个 Block 拼出完整 Shared Tile
```

线程不是只搬“自己最终要算的输出所需数据”，而是共同搬整个 Block 会复用的数据。

---

# 9. `cp.async` 相关机制

当前项目使用了三个相关函数：

```cpp
cp_async16()
cp_async_commit()
cp_async_wait_group<N>()
```

---

## 9.1 `cp_async16()`

提交一次：

```text
Global Memory → Shared Memory
```

的 16B 异步复制。

它的关键价值是：

> 数据可以直接进入 Shared Memory，而不需要先由普通 Load 明确落入线程寄存器，再 Store 到 Shared Memory。

---

## 9.2 `cp_async_commit()`

```cpp
cp.async.commit_group;
```

表示：

> 把前面提交的一批异步复制组成一个 group。

项目在一次 A/B Tile 加载结束后调用：

```cpp
cp_async_commit();
```

所以 A Tile 和 B Tile 的复制属于同一批异步搬运任务。

---

## 9.3 `cp_async_wait_group<0>()`

```cpp
cp.async.wait_group 0;
```

表示等待尚未完成的异步复制组完成。

项目中：

```cpp
cp_async_wait_group<0>();
__syncthreads();
```

含义是：

1. 等待 Global → Shared 的复制完成；
2. 再进行 Block 级同步；
3. 保证所有线程都可以安全读取 Shared Memory。

---

# 10. 为什么 `cp.async` 后还需要 `__syncthreads()`

`cp_async_wait_group<0>()` 关注的是：

```text
异步复制是否完成
```

`__syncthreads()` 关注的是：

```text
同一 Block 中所有线程是否都到达这个阶段
```

计算阶段会读取整个 Tile，而 Tile 是由不同线程协作搬入的。

所以需要：

```text
等待复制完成
+
等待整个 Block 对齐到同一执行阶段
```

典型顺序：

```cpp
load_tile_async(...);
cp_async_wait_group<0>();
__syncthreads();
compute(...);
```

---

# 11. 双缓冲：为什么 Shared Memory 有 `[2]`

Shared Memory：

```cpp
AS[2]
BS[2]
```

表示有两组缓冲区：

```text
buffer 0
buffer 1
```

循环中：

```cpp
int cur = t & 1;
int nxt = 1 - cur;
```

所以：

```text
t=0：cur=0，nxt=1
t=1：cur=1，nxt=0
t=2：cur=0，nxt=1
...
```

---

## 11.1 双缓冲的时间线

```text
开始：
加载 Tile 0 → Buffer 0

循环 t=0：
Buffer 0 用于计算
同时加载 Tile 1 → Buffer 1

循环 t=1：
Buffer 1 用于计算
同时加载 Tile 2 → Buffer 0

循环 t=2：
Buffer 0 用于计算
同时加载 Tile 3 → Buffer 1
```

示意图：

```text
时间 ─────────────────────────────────────────────→

Buffer 0: [加载 Tile0] [计算 Tile0] [加载 Tile2] [计算 Tile2]
Buffer 1:              [加载 Tile1] [计算 Tile1] [加载 Tile3]
```

理想目标是让：

```text
当前 Tile 的计算
```

和：

```text
下一 Tile 的 Global → Shared 搬运
```

尽量重叠。

这就是软件流水线的基础思想。

---

# 12. Shared Memory 如何进入线程寄存器

真正计算发生在：

```cpp
compute(...)
```

对于每个：

```cpp
for (int d = 0; d < BK; d++)
```

当前线程从 Shared Memory 读取 A/B 数据。

---

## 12.1 A：Shared → `reg_m[]`

源码关键逻辑：

```cpp
reg_m[...] =
    AS[d*A_STRIDE +
       (wrow*WM +
        wr*WSUBM +
        trow*TM +
        rm)];
```

它的含义不是要你背公式，而是：

```text
当前 Warp 在 M 方向的起点
+
当前 Warp 内子块偏移
+
当前线程的 M 方向偏移
+
当前线程 Thread Tile 内部行偏移
```

最终定位到：

```text
当前线程在 d 这一层 K 上需要的 A 值
```

这些值进入：

```cpp
reg_m[]
```

---

## 12.2 B：Shared → `reg_n[]`

源码：

```cpp
reg_n[...] =
    BS[d*BN +
       wcol*WN +
       wc*WSUBN +
       tcol*TN +
       rn];
```

逻辑是：

```text
当前 Warp 在 N 方向的起点
+
当前 Warp 内列子块偏移
+
当前线程的 N 方向偏移
+
Thread Tile 内部列偏移
```

最终定位当前线程需要的 B 值，放入：

```cpp
reg_n[]
```

---

# 13. `reg_m` 和 `reg_n` 为什么不是最终结果

它们只是当前 K 位置的一小批输入数据。

例如 Default 配置：

```text
reg_m 大小 = WMITER × TM = 1 × 8 = 8
reg_n 大小 = WNITER × TN = 4 × 4 = 16
```

也就是当前线程在某个 `d` 上取得：

```text
8 个 A 值
16 个 B 值
```

然后做外积：

```text
8 × 16 = 128 次乘加
```

累加进 `res[128]`。

---

# 14. 寄存器中的外积计算

核心代码：

```cpp
res[...] +=
    reg_m[...] *
    reg_n[...];
```

从数学上看：

```text
reg_m = 8×1 列向量
reg_n = 1×16 行向量
```

外积：

```text
8×1  ×  1×16
=
8×16
```

结果正好对应当前线程负责的：

```text
8 × 16 Thread Tile
```

示意图：

```text
reg_m                reg_n

[a0]          [b0 b1 b2 ... b15]
[a1]
[a2]                 ↓ 外积
...
[a7]

产生：

[a0b0 a0b1 ... a0b15]
[a1b0 a1b1 ... a1b15]
...
[a7b0 a7b1 ... a7b15]
```

这些结果不会覆盖 `res`，而是：

```text
累加到 res
```

因为矩阵乘法还要遍历整个 K。

---

# 15. 一个 K Tile 内部的数据流

Default 配置下：

```text
BK = 8
```

所以 `compute()` 会执行：

```text
d = 0,1,2,...,7
```

每个 d：

```text
Shared AS[d] → reg_m
Shared BS[d] → reg_n
reg_m × reg_n → 累加 res
```

完整过程：

```text
d=0：
A 的第 0 个 K 值 × B 的第 0 个 K 值 → res

d=1：
A 的第 1 个 K 值 × B 的第 1 个 K 值 → res

...

d=7：
A 的第 7 个 K 值 × B 的第 7 个 K 值 → res
```

处理完一个 BK Tile 后，`res[]` 保留结果，下一批 K Tile 到来后继续累加。

---

# 16. `res[]` 的生命周期

`res[]` 在 Kernel 一开始初始化：

```cpp
float res[...] = {};
```

它的生命周期贯穿整个 K 维循环：

```text
Kernel 开始：res = 0
    ↓
第 0 个 K Tile 累加
    ↓
第 1 个 K Tile 累加
    ↓
第 2 个 K Tile 累加
    ↓
...
    ↓
全部 K 处理完
    ↓
写回 C
```

它不会在每个 BK Tile 后写回 Global Memory。

这是高性能 GEMM 的关键：

> 中间结果始终留在线程寄存器，直到完整 K 维累加结束，只写回一次。

如果每次 BK 后都写回 C，会增加大量 Global Memory 读写。

---

# 17. Fast Path 的完整时间线

把上面的机制串成一条实际执行路线：

```text
1. Kernel 进入
2. 当前 Block 定位 A_blk / B_blk / C_out
3. Tile 0：Global → Shared Buffer 0
4. wait_group + __syncthreads
5. 进入 K Tile 循环
6. 下一 Tile：Global → Shared Buffer nxt（异步）
7. 当前 Tile：Shared Buffer cur → reg_m/reg_n
8. reg_m × reg_n → res
9. 等待下一 Tile 完成
10. __syncthreads
11. 交换 cur / nxt
12. 重复 6~11
13. 处理 K remainder
14. res → C Global Memory
```

示意图：

```text
Global A/B
   │
   │ cp.async Tile 0
   ▼
Shared Buffer 0
   │
   │ compute
   ▼
reg_m / reg_n
   │
   │ FMA
   ▼
res

与此同时：

Global A/B 下一 Tile
   │
   │ cp.async
   ▼
Shared Buffer 1
```

---

# 18. K 余数如何处理

Fast Path 要求：

```text
K % 4 == 0
```

但不要求：

```text
K % BK == 0
```

例如：

```text
K = 12
BK = 8
```

则：

```text
完整 Tile：8
余数：4
```

源码：

```cpp
int k_rem = K % BK;
```

余数处理步骤：

```text
1. 清零当前 Shared Buffer
2. 标量加载剩余 A/B
3. __syncthreads
4. compute_k(..., k_rem)
```

清零是因为 Shared Tile 大小仍为 BK，但实际只有 `k_rem` 个有效 K 位置。

无效位置保持 0，就不会影响累加。

---

# 19. Edge Path：边界或非对齐数据怎样流动

若：

- 当前 Block 超出 M 边界；
- 当前 Block 超出 N 边界；
- K 或 N 不满足 4-float 对齐；

则进入 edge path。

它不使用统一的异步向量化加载，而采用：

```text
清零 Shared
    ↓
带 M/N/K 边界判断地标量加载
    ↓
__syncthreads
    ↓
compute_edge()
```

---

## 19.1 为什么先清零 Shared

边界 Tile 中有些位置不存在。

例如：

```text
BM=64
但剩余有效行只有 40
```

Shared Memory 仍按完整 Tile 分配。

因此先：

```cpp
AS[0][i] = 0.f;
BS[0][i] = 0.f;
```

再只填写有效元素。

无效位置为 0，计算时不会产生错误贡献。

---

## 19.2 `compute_edge()`

它在 Shared → Registers 时再次检查：

```cpp
row < m_edge
col < n_edge
```

越界位置写入：

```text
0
```

因此数据流依然是：

```text
Global → Shared → reg_m/reg_n → res
```

只是每一步增加边界保护。

---

# 20. 结果怎样从寄存器写回 Global Memory

当 K 维全部计算完后：

```text
res[] 中就是当前线程负责的最终局部 C 结果
```

写回公式：

```text
C = alpha × res + beta × C
```

---

# 21. Fast Path：`float4` 向量化写回

源码：

```cpp
float4 v =
    *reinterpret_cast<const float4*>(
        &C_out[row*N + col]
    );
```

然后：

```cpp
v.x = alpha*res[...] + beta*v.x;
v.y = alpha*res[...] + beta*v.y;
v.z = alpha*res[...] + beta*v.z;
v.w = alpha*res[...] + beta*v.w;
```

最后：

```cpp
*reinterpret_cast<float4*>(
    &C_out[row*N + col]
) = v;
```

也就是一次处理连续 4 个 C 元素。

数据路线：

```text
Global C 中原来的 4 个 float
    ↓ 加载到 float4 v
与 res 中 4 个寄存器结果融合
    ↓
新的 float4
    ↓
写回 Global C
```

---

## 21.1 为什么还要读取原来的 C

因为通用 GEMM 是：

```text
C = alpha × A × B + beta × C
```

若：

```text
beta != 0
```

就必须读取原来的 C。

若测试中：

```text
beta = 0
```

数学上旧 C 不参与结果，但代码仍保持通用形式。

---

# 22. Edge Path：逐元素写回

边界路径需要检查：

```cpp
grow < M
gcol < N
```

因此使用逐元素写回：

```cpp
C_out[...] =
    alpha * res[ir] +
    beta * C_out[...];
```

它比 `float4` 路径更通用，但：

- 指令更多；
- 分支更多；
- 难以统一向量化；
- 性能通常更低。

---

# 23. 当前项目使用的关键性能机制

## 23.1 Shared Memory 数据复用

A/B Tile 从 Global Memory 搬一次后，被 Block 内很多线程重复读取。

作用：

```text
减少重复 Global Memory 访问
```

---

## 23.2 Register Blocking

线程把多个 C 输出元素保存在 `res[]`。

作用：

```text
提高每次 A/B 数据读取所产生的计算量
减少中间结果的全局写回
```

---

## 23.3 Vectorized Load/Store

使用 16B 的：

```text
cp.async
float4
```

作用：

```text
减少内存指令
利用连续地址
提高访存效率
```

---

## 23.4 `cp.async`

让 Global → Shared 搬运可以异步提交。

作用：

```text
为内存搬运与计算重叠创造条件
减少显式寄存器中转
```

---

## 23.5 Double Buffering

使用：

```text
AS[2]
BS[2]
```

作用：

```text
当前 Buffer 计算
下一 Buffer 预取
```

---

## 23.6 Loop Unrolling

源码中大量：

```cpp
#pragma unroll
```

作用：

- 减少循环控制指令；
- 暴露更多独立指令；
- 帮助编译器调度；
- 增加 ILP。

代价可能是：

- 代码体积增大；
- 寄存器压力增加。

---

## 23.7 Fast / Edge 双路径

Fast Path：

```text
完整、对齐、向量化、cp.async、float4
```

Edge Path：

```text
边界检查、标量加载、标量写回
```

作用：

> 让大部分内部完整 Tile 走高性能路径，同时保持任意尺寸的正确性。

---

# 24. Default 配置下的数据量举例

Default：

```text
BM=128
BN=256
BK=8
线程数=256
```

---

## 24.1 每个 K Tile 搬多少数据

A Tile：

```text
128 × 8 = 1024 float
= 4096 Bytes
```

B Tile：

```text
8 × 256 = 2048 float
= 8192 Bytes
```

每个 K Tile 合计：

```text
12 KiB
```

---

## 24.2 双缓冲 Shared Memory

两份 A/B：

```text
2 × 12 KiB = 24 KiB
```

对应：

```cpp
AS[2][128×8]
BS[2][8×256]
```

---

## 24.3 每线程寄存器数据

Default：

```text
reg_m：8 float
reg_n：16 float
res：128 float
```

此外还有索引、指针和临时变量。

因此这是一个寄存器压力较高的 Kernel。

---

## 24.4 每个 d 的计算量

每个线程：

```text
8 个 A
×
16 个 B
=
128 次乘加
```

一个 K Tile 有：

```text
BK=8 个 d
```

所以每线程每个 K Tile：

```text
128 × 8 = 1024 次乘加
```

一个 Block 有 256 线程：

```text
256 × 1024 = 262144 次乘加
```

这正好对应：

```text
BM × BN × BK
=
128 × 256 × 8
=
262144
```

---

# 25. 一张完整的数据流时序图

```text
┌───────────────────────────────────────────────────────────────┐
│ CPU Host Memory                                               │
│ A / B / C                                                     │
└───────────────────────────┬───────────────────────────────────┘
                            │ cudaMemcpy H2D
                            ▼
┌───────────────────────────────────────────────────────────────┐
│ GPU Global Memory                                             │
│ 完整 A[M×K] / B[K×N] / C[M×N]                                │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            │ 当前 Block 沿 K 选择：
                            │ A Tile[BM×BK]
                            │ B Tile[BK×BN]
                            ▼
┌───────────────────────────────────────────────────────────────┐
│ Shared Memory                                                 │
│ AS[cur][BK×BM] / BS[cur][BK×BN]                              │
│ AS[nxt][BK×BM] / BS[nxt][BK×BN]                              │
└───────────────────────────┬───────────────────────────────────┘
                            │ compute() 每个 d 读取
                            ▼
┌───────────────────────────────────────────────────────────────┐
│ Thread Registers                                              │
│ reg_m[WMITER×TM]                                              │
│ reg_n[WNITER×TN]                                              │
└───────────────────────────┬───────────────────────────────────┘
                            │ 外积 FMA
                            ▼
┌───────────────────────────────────────────────────────────────┐
│ Thread Registers                                              │
│ res[WMITER×TM×WNITER×TN]                                     │
│ 在整个 K 循环期间持续累加                                    │
└───────────────────────────┬───────────────────────────────────┘
                            │ alpha×res + beta×C
                            │ float4 / scalar write-back
                            ▼
┌───────────────────────────────────────────────────────────────┐
│ GPU Global Memory                                             │
│ 最终 C[M×N]                                                   │
└───────────────────────────┬───────────────────────────────────┘
                            │ cudaMemcpy D2H
                            ▼
┌───────────────────────────────────────────────────────────────┐
│ CPU Host Memory                                               │
│ 自定义 GEMM 结果，与 cuBLAS 结果比较                          │
└───────────────────────────────────────────────────────────────┘
```

---

# 26. 代码中的关键函数与数据流位置对照

| 函数/变量 | 数据流阶段 | 作用 |
|---|---|---|
| `launch_gemm()` | Host 调度 | 进入自定义 GEMM |
| `launch_gemm_warp_tile()` | Host 配置选择 | 选择 Default/Small/Irregular |
| `A_blk`、`B_blk` | Global 定位 | 当前 Block 的 A/B 起点 |
| `load_tile_async()` | Global → Shared | 异步搬运 A/B Tile |
| `cp_async16()` | Global → Shared | 一次搬 16B |
| `cp_async_commit()` | 异步任务组织 | 提交复制 group |
| `cp_async_wait_group<0>()` | 异步等待 | 等待复制完成 |
| `AS[2]`、`BS[2]` | Shared Memory | 双缓冲 Tile |
| `__syncthreads()` | Block 同步 | 保证共享数据可安全使用 |
| `compute()` | Shared → Registers → FMA | 主计算路径 |
| `compute_k()` | Shared → Registers → FMA | K 余数计算 |
| `compute_edge()` | Shared → Registers → FMA | 边界计算 |
| `reg_m[]` | Registers | 当前 A 输入片段 |
| `reg_n[]` | Registers | 当前 B 输入片段 |
| `res[]` | Registers | C 输出累加器 |
| `float4 v` | Registers / Global | 向量化读写 C |
| `C_out` | Global 定位 | 当前 Warp 输出起点 |

---

# 27. 重新看代码时应该按什么顺序

现在重新看 `gemm_warp_tile.cu`，不要按文件从第一行机械看到最后一行，建议按数据流阅读：

## 第一步：看 Global 指针定位

```cpp
A_blk
B_blk
C_out
```

回答：

```text
当前 Block / Warp 对应完整矩阵的哪一部分？
```

---

## 第二步：看 Shared Memory

```cpp
AS[2]
BS[2]
```

回答：

```text
当前 K Tile 放在哪里？
为什么是双缓冲？
```

---

## 第三步：看 `load_tile_async()`

回答：

```text
哪些线程搬哪些 A/B 数据？
每次为什么是 16B？
```

---

## 第四步：看 `compute()`

只抓三步：

```text
AS → reg_m
BS → reg_n
reg_m × reg_n → res
```

不要第一遍就陷入所有下标。

---

## 第五步：看双缓冲循环

抓住：

```text
cur 用来计算
nxt 用来预取
```

---

## 第六步：看写回

抓住：

```text
res → alpha×res + beta×C → Global C
```

---

# 28. 最终脑内模型

你现在应该建立两条彼此配合的路线。

## 28.1 输出责任划分路线

```text
完整 C
    ↓
Block Tile
    ↓
Warp Tile
    ↓
Thread Tile
    ↓
res[]
```

## 28.2 输入数据供应路线

```text
完整 A/B（Global）
    ↓
Block 当前 K Tile
    ↓
AS/BS（Shared）
    ↓
reg_m/reg_n（Registers）
    ↓
外积 FMA
    ↓
res[]
```

最终：

```text
res[]
    ↓
alpha×res + beta×C
    ↓
Global C
```

把两条路线合起来，就是当前 GEMM Kernel 的完整本质：

> Block、Warp、Thread 先确定“谁负责哪些 C 输出”；然后 A/B 数据沿 K 方向从 Global Memory 流入 Shared Memory，再进入各线程寄存器，持续喂给 `res[]` 做乘加；K 处理结束后，每个线程把自己的寄存器结果写回 C。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-gemm算子|模块-gemm算子]]
- 关联阅读：[[outputs/项目整理/专题-05-GEMM与推理性能|专题-05-GEMM与推理性能]]

%% 项目关联导航：结束 %%
