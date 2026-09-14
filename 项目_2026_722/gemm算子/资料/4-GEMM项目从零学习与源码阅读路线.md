# GEMM 项目从零学习与源码阅读路线

> 适用仓库：`gemm-project`  
> 项目定位：CUDA FP32 GEMM（SGEMM）手写算子、cuBLAS 对比测试及 GPU 存储层次 Microbenchmark  
> 学习目标：不是“把 397 行 Kernel 看一遍”，而是最终能够独立解释一次 GEMM 调用如何从测试程序进入 CUDA Kernel，矩阵如何被分配到 Block、Warp 和 Thread，数据如何在 Global Memory、Shared Memory、Register 之间流动，以及当前实现为什么快、哪里可能有问题、应如何继续验证和优化。

---

# 1. 先建立对项目的正确认识

## 1.1 这个仓库不是一个完整 AI 推理框架

它不包含模型加载、请求调度、KV Cache、PagedAttention 或 Continuous Batching。它聚焦的是更底层的一个核心问题：

\[
C = \alpha A B + \beta C
\]

其中：

- `A`：`M × K`
- `B`：`K × N`
- `C`：`M × N`
- 数据类型：FP32
- 存储方式：代码接口按照 Row-major 理解 A、B、C

GEMM 是 Transformer 中 QKV 投影、Attention 输出投影、MLP 和 LM Head 等计算的基础。因此，这个项目属于 AI Infra 中的 **CUDA 算子开发与性能优化层**。

## 1.2 整个项目只有一条核心调用链

```text
测试程序
 tests/test_perf_gemm.cpp
        │
        ▼
公开接口
 launch_gemm(...)
 include/gemm/launcher.h
 src/launcher.cu
        │
        ▼
Warp-Tile GEMM 启动器
 launch_gemm_warp_tile(...)
        │
        ├── WarpTileSmallConfig
        ├── WarpTileIrregularConfig
        └── WarpTileDefaultConfig
        │
        ▼
CUDA Kernel
 gemm_warp_tile_kernel<Cfg><<<grid, block>>>(...)
        │
        ├── Edge Path：标量加载、边界判断，逻辑较直观
        └── Fast Path：cp.async、双缓冲、向量化写回
```

学习这个仓库时，必须始终围绕这条调用链。不要把每个文件当成互不相干的代码片段。

## 1.3 核心数据流

一次 Block Tile 的数据流是：

```text
Global Memory 中的 A、B
        │
        │ 线程协作加载
        ▼
Shared Memory 中的 AS、BS
        │
        │ 每个 Warp/Thread 取自己需要的数据
        ▼
reg_m、reg_n 寄存器
        │
        │ Outer Product / FMA
        ▼
res 累加器寄存器
        │
        │ alpha * res + beta * C
        ▼
Global Memory 中的 C
```

从头到尾看懂这个数据流，才算真正看懂项目。

---

# 2. 阅读源码前必须具备的基础

这个项目并不适合作为“第一次接触 CUDA”的代码。正式阅读前，需要补齐以下知识。

## 2.1 必须熟练掌握

### C/C++ 基础

- 指针与一维数组模拟二维矩阵
- `const float*` 与 `float*`
- Row-major 地址计算：`matrix[row * leading_dim + col]`
- 模板参数和 `static constexpr`
- `enum class`
- 头文件声明与 `.cu/.cpp` 实现分离
- `reinterpret_cast`
- `size_t`

### CUDA 基础

- Host 和 Device 的区别
- `cudaMalloc`、`cudaMemcpy`、`cudaMemset`、`cudaFree`
- `__global__`、`__device__`、`__forceinline__`
- Kernel 启动语法 `<<<gridDim, blockDim>>>`
- `blockIdx`、`threadIdx`、`blockDim`、`gridDim`
- Warp 固定包含 32 个线程
- `__shared__` 和 `__syncthreads()`
- CUDA Event 计时
- 合并访存的基本概念

### GEMM 基础

必须能够手算：

```text
C[m, n] = Σ A[m, k] × B[k, n]
```

并理解：

- 每个 `C[m,n]` 是 A 的一行与 B 的一列的点积；
- `2MNK` 是 GEMM 的近似浮点运算次数；
- Tile 的本质是让一份加载的数据被多个乘加重复利用。

## 2.2 阅读到 Fast Path 前必须理解

- `float4` 一次表示 4 个连续 FP32，即 16 Byte；
- Shared Memory 地址空间与普通全局地址不同；
- Inline PTX 的基本输入/输出约束；
- `cp.async` 用于 Global Memory 到 Shared Memory 的异步复制；
- 双缓冲的目的：计算当前 Tile 时预取下一个 Tile；
- 寄存器数量过多可能降低 Occupancy 或导致 Spill。

## 2.3 当前只需了解，不必一开始钻研

- PTX 每条指令的完整语法；
- L1/L2 Cache 的全部微架构细节；
- cuBLAS 内部实现；
- Tensor Core、WMMA、MMA PTX；
- Roofline Model 的完整推导。

第一次阅读的重点是 **坐标映射和数据流**，而不是背 PTX。

---

# 3. 仓库目录与每个文件的职责

```text
gemm-project/
├── CMakeLists.txt
├── README.md
├── include/gemm/
│   ├── types.h
│   ├── launcher.h
│   └── kernels/
│       ├── gemm_warp_tile.h
│       └── gemm_warp_tile_config.h
├── src/
│   ├── launcher.cu
│   └── kernels/gemm_warp_tile.cu
├── tests/
│   ├── test_perf_gemm.cpp
│   └── compare_fair_runs.sh
├── microbenchmark/
│   ├── fair_compare_sgemm.cu
│   ├── cublas_sgemm_bench.cu
│   ├── dram_bandwidth.cu
│   ├── dram_latency.cu
│   ├── l1cache_latency.cu
│   ├── l2cache_bandwidth.cu
│   ├── l2cache_latency.cu
│   ├── smem_bandwidth.cu
│   └── smem_latency.cu
├── tools/generate_stable_baseline.py
├── docs/
└── results/
```

## 3.1 核心功能文件

| 文件 | 作用 | 阅读优先级 |
|---|---|---:|
| `tests/test_perf_gemm.cpp` | 创建输入、调用 cuBLAS 和自定义 GEMM、校验结果、计时、计算 GFLOPS | 最高 |
| `include/gemm/launcher.h` | 对外暴露 `launch_gemm` 接口 | 高 |
| `include/gemm/types.h` | 定义算法枚举 `GemmAlgo` | 高 |
| `src/launcher.cu` | 选择具体算法并调用 Warp-Tile 实现 | 高 |
| `include/gemm/kernels/gemm_warp_tile_config.h` | 定义三套 Tile 参数 | 最高 |
| `src/kernels/gemm_warp_tile.cu` | 核心 CUDA Kernel、数据搬运和计算 | 最高 |

## 3.2 性能实验文件

| 文件 | 作用 |
|---|---|
| `microbenchmark/fair_compare_sgemm.cu` | 更稳定的 cuBLAS 与自定义 Kernel 对比，重复次数更多 |
| `tests/compare_fair_runs.sh` | 连续跑多轮并统计均值、标准差和关键非对齐 Shape |
| `tools/generate_stable_baseline.py` | 汇总 CSV、生成结果摘要与性能曲线 |
| `results/*` | 保存历史性能结果，不是源码真理 |

## 3.3 GPU 存储层次 Microbenchmark

| 文件 | 测量内容 |
|---|---|
| `dram_bandwidth.cu` | DRAM 读、写、复制带宽 |
| `dram_latency.cu` | DRAM 依赖加载延迟 |
| `l2cache_bandwidth.cu` | L2 Cache 带宽 |
| `l2cache_latency.cu` | L2 Cache 延迟 |
| `l1cache_latency.cu` | L1 Cache 延迟 |
| `smem_bandwidth.cu` | Shared Memory 带宽 |
| `smem_latency.cu` | Shared Memory 延迟 |

这些文件应当在核心 GEMM 看懂之后再读。它们用于解释“为什么需要数据复用和 Shared Memory”，不是主调用链的一部分。

## 3.4 `docs/warp_tiles.md` 只能作为辅助图示

不要把该文档作为第一阅读入口，也不要把其中每一句都当成当前代码的准确说明。它存在明显的版本滞后或文字错误，例如：

- 文档写“`warp_idx` 一共 4 个 warp”，但 DefaultConfig 使用 256 个线程，即 8 个 Warp；
- 结果寄存器 Shape 写成了 `WNITER * TM * WNITER * TN`，按代码应是 `WMITER * TM * WNITER * TN`；
- B 的搬运说明中写成“存入 AS”，按代码应存入 BS；
- 部分“行偏移/列偏移”的文字标注与变量名容易混淆。

正确使用方式是：先从配置与 Kernel 代码推导坐标，再把 `docs/pic/1.png`～`5.png` 当成辅助示意图。发生冲突时，以代码中的实际索引为准，并通过测试验证代码本身是否正确。

---

# 4. 最推荐的源码阅读顺序

## 总顺序

```text
0. README.md + CMakeLists.txt
1. tests/test_perf_gemm.cpp
2. include/gemm/types.h
3. include/gemm/launcher.h
4. src/launcher.cu
5. include/gemm/kernels/gemm_warp_tile.h
6. include/gemm/kernels/gemm_warp_tile_config.h
7. src/kernels/gemm_warp_tile.cu 的启动与分发部分
8. Kernel 主体的索引和资源定义
9. Edge Path
10. compute_edge / compute_k / compute
11. Fast Path 与 cp.async
12. 写回路径
13. fair_compare_sgemm.cu + compare_fair_runs.sh
14. Microbenchmark
15. results、docs 和 Python 画图工具
```

特别注意：

> `src/kernels/gemm_warp_tile.cu` 不要从第 1 行一直顺序读到第 397 行。应当按照“外层启动 → Kernel 外壳 → 简单路径 → 计算函数 → 快速路径”的顺序跳着读。

---

# 5. 第一阶段：先看项目怎样被构建和运行

## 5.1 阅读 `README.md`

只需先回答四个问题：

1. 项目实现的数据类型是什么？——FP32。
2. 比较对象是谁？——cuBLAS SGEMM。
3. 主要可执行程序是什么？——`test_perf_gemm`。
4. 当前三套配置分别是什么？——Default、Small、Irregular。

第一次不要相信 README 中所有性能结论。README 是项目说明，不是正确性证明。

## 5.2 阅读 `CMakeLists.txt`

重点跟踪以下依赖关系：

```text
src/launcher.cu
src/kernels/gemm_warp_tile.cu
        │
        ▼
静态库 gemm_lib
        │
        ▼
tests/test_perf_gemm.cpp
        │
        ▼
可执行文件 test_perf_gemm
```

需要理解：

- `find_package(CUDAToolkit REQUIRED)`：寻找 CUDA Toolkit；
- `add_library(gemm_lib STATIC ...)`：把两个 `.cu` 编译成静态库；
- `target_include_directories`：让编译器能找到 `include/gemm/...`；
- `CUDA::cudart`：CUDA Runtime；
- `CUDA::cublas`：cuBLAS；
- `-O3`：编译优化；
- `--use_fast_math`：使用更激进的浮点优化；
- `-lineinfo`：为性能分析工具保留源码行信息；
- `CMAKE_CUDA_ARCHITECTURES 89`：代码当前硬编码面向 `sm_89`。

### 读完后的自测

能够不用看代码说出：

> 编译后，测试程序链接 `gemm_lib`，而 `gemm_lib` 内含统一 launcher 和 Warp-Tile Kernel。

---

# 6. 第二阶段：从 `tests/test_perf_gemm.cpp` 建立完整闭环

这是最适合作为第一份正式源码阅读的文件。

## 6.1 按以下区段阅读

### 1）错误处理：第 14～26 行

- `checkCudaError`
- `checkCublasError`

理解为什么 CUDA API 和 cuBLAS API 都需要检查返回值。

### 2）测试 Shape：第 28～42 行

代码生成：

- 256～8192 的方阵；
- 100、257、511、1000、1234、2047 等非对齐尺寸。

当前测试始终令：

```cpp
M = N = K = sz;
```

这意味着仓库没有覆盖真实 Transformer 中常见的矩形 GEMM。

### 3）Host/Device 内存：第 66～90 行

画出下面的表：

| 名称 | 所在位置 | Shape | 用途 |
|---|---|---|---|
| `A` | CPU | `M×K` | 输入矩阵 |
| `B` | CPU | `K×N` | 输入矩阵 |
| `C_cublas` | CPU | `M×N` | cuBLAS 结果 |
| `C` | CPU | `M×N` | 自定义 Kernel 结果 |
| `d_A` | GPU | `M×K` | GPU 输入 |
| `d_B` | GPU | `K×N` | GPU 输入 |
| `d_C` | GPU | `M×N` | GPU 输出 |

重点理解 `sizeA/sizeB/sizeC` 为什么要乘 `sizeof(float)`。

### 4）cuBLAS 调用：第 92～145 行

最容易困惑的是：为什么接口参数顺序看起来像 `B × A`？

原因是 cuBLAS 传统接口按 Column-major 解释矩阵。代码通过交换 A/B 及 M/N 参数，利用：

\[
(AB)^T = B^T A^T
\]

使 Row-major 的结果等价得到计算。

第一次阅读只需要知道：

```cpp
cublasSgemm(..., N, M, K, ..., d_B, N, d_A, K, ..., d_C, N)
```

是常见的 Row-major 适配写法，不代表项目真的要算 `B×A`。

### 5）自定义 GEMM：第 146～168 行

核心入口只有一句：

```cpp
launch_gemm(M, N, K, alpha, d_A, d_B, beta, d_C);
```

从这里开始向下追踪调用链。

### 6）正确性与性能：第 170～194 行

理解 GFLOPS 公式：

\[
GFLOPS = \frac{2MNK}{time\_seconds \times 10^9}
\]

代码的时间单位是毫秒，并且统计了多轮，因此写成：

```cpp
repeat_time * 2.0f * M * N * K / (time_ms * 1e6f)
```

## 6.2 这一文件必须发现的测试缺陷

当前输入为：

```cpp
A[i] = 1.0f;
B[i] = 2.0f;
```

这种全常数输入非常容易掩盖索引错误，因为错误读取 A 的其他位置仍然得到 1。当前误差判定也只有固定绝对误差 `1e-5`，没有随机数、相对误差、NaN/Inf、矩形 Shape、非默认 `alpha/beta` 测试。

学习时必须建立一个习惯：

> 测试输出 `error_count=0`，不等于 Kernel 已被严格证明正确。

## 6.3 读完后的自测

必须能完整口述：

```text
CPU 初始化 A、B
→ cudaMalloc 分配 d_A、d_B、d_C
→ cudaMemcpy 上传输入
→ cuBLAS warmup 和计时
→ 保存参考结果
→ 清空 d_C
→ 自定义 Kernel warmup 和计时
→ 下载结果
→ 比较误差
→ 计算 GFLOPS 和性能比例
```

---

# 7. 第三阶段：看公开接口与算法分发

## 7.1 `include/gemm/types.h`

只有一个枚举：

```cpp
enum class GemmAlgo {
    Auto,
    WarpTile,
};
```

这说明作者预留了未来加入其他实现的接口，但当前只有 Warp-Tile 一种算法。

## 7.2 `include/gemm/launcher.h`

公开接口：

```cpp
launch_gemm(M, N, K, alpha, A, B, beta, C, algo)
```

把它对应到公式：

\[
C = \alpha AB + \beta C
\]

注意：接口没有传 CUDA Stream，所以 Kernel 默认发射到默认 Stream；也没有 Leading Dimension、Transpose、Batch、数据类型等通用 GEMM 参数。

## 7.3 `src/launcher.cu`

阅读重点：

```cpp
if (final_algo == GemmAlgo::Auto) {
    final_algo = select_gemm_algo(M, N, K);
}
```

但 `select_gemm_algo()` 当前无论 Shape 如何都返回：

```cpp
GemmAlgo::WarpTile
```

因此这里的 `Auto` 只是一层接口包装，真正的 Shape Dispatch 位于 `gemm_warp_tile.cu` 的末尾。

## 7.4 `include/gemm/kernels/gemm_warp_tile.h`

它只是声明更具体的启动函数：

```cpp
launch_gemm_warp_tile(...)
```

调用关系因此变成：

```text
launch_gemm
→ launch_gemm_warp_tile
→ launch_gemm_warp_tile_cfg<Cfg>
→ gemm_warp_tile_kernel<Cfg>
```

### 读完后的自测

能够解释为什么项目要有两层 launcher：

- 上层 `launch_gemm` 提供稳定统一接口；
- 下层 `launch_gemm_warp_tile` 负责该算法内部的配置选择。

---

# 8. 第四阶段：彻底吃透配置文件

阅读：

```text
include/gemm/kernels/gemm_warp_tile_config.h
```

这是看懂核心 Kernel 的钥匙。不要只记参数名称，要把它们和矩阵区域对应起来。

## 8.1 参数含义

| 参数 | 含义 |
|---|---|
| `BM` | 一个 CUDA Block 负责的 C Tile 行数 |
| `BN` | 一个 CUDA Block 负责的 C Tile 列数 |
| `BK` | 一次从 K 维处理的深度 |
| `WM` | 一个 Warp 负责的 C Tile 行数 |
| `WN` | 一个 Warp 负责的 C Tile 列数 |
| `WNITER` | Warp 在 N 方向被分为多少个子区域循环 |
| `TM` | 每个线程在一个子区域中负责多少行 |
| `TN` | 每个线程在一个子区域中负责多少列 |
| `NUM_THREADS` | 一个 Block 的线程数 |

Kernel 内还推导：

```cpp
WSUBN = WN / WNITER;
WSUBM = (32 / (WSUBN / TN)) * TM;
WMITER = WM / WSUBM;
```

## 8.2 三套配置的完整推导

### Default 配置

```text
BM=128, BN=256, BK=8
WM=64, WN=64
WNITER=4, TM=8, TN=4
NUM_THREADS=256
```

推导：

```text
Warp 数量 = 256 / 32 = 8
Block 中 Warp Tile 数量 = (128/64) × (256/64) = 2 × 4 = 8
```

因此恰好一个 Warp 负责一个 `64×64` 输出 Tile。

```text
WSUBN = 64 / 4 = 16
WSUBN / TN = 16 / 4 = 4
WSUBM = (32 / 4) × 8 = 64
WMITER = 64 / 64 = 1
```

每个线程维护：

```text
M 方向：WMITER × TM = 1 × 8 = 8
N 方向：WNITER × TN = 4 × 4 = 16
输出元素数：8 × 16 = 128
```

验证：

```text
32 个线程 × 每线程 128 个结果 = 4096
64 × 64 Warp Tile = 4096
```

### Small 配置

```text
BM=64, BN=64, BK=8
WM=32, WN=32
WNITER=2, TM=4, TN=4
NUM_THREADS=128
```

```text
Warp 数量 = 4
Block 中 Warp Tile = (64/32) × (64/32) = 4
每线程输出 = 4 × 8 = 32
32 线程 × 32 = 1024 = 32×32
```

### Irregular 配置

```text
BM=64, BN=128, BK=8
WM=32, WN=64
WNITER=2, TM=4, TN=4
NUM_THREADS=128
```

```text
Warp 数量 = 4
Block 中 Warp Tile = (64/32) × (128/64) = 4
WSUBN = 32
WSUBM = 16
WMITER = 2
每线程输出 = (2×4) × (2×4) = 64
32 线程 × 64 = 2048 = 32×64
```

## 8.3 Shared Memory 占用

Kernel 定义双缓冲：

```cpp
__shared__ float AS[2][BM * BK];
__shared__ float BS[2][BK * BN];
```

| 配置 | AS | BS | 总 Shared Memory |
|---|---:|---:|---:|
| Default | `2×128×8×4B = 8KB` | `2×8×256×4B = 16KB` | 24KB |
| Small | 4KB | 4KB | 8KB |
| Irregular | 4KB | 8KB | 12KB |

## 8.4 累加器寄存器压力

`res` 的大小：

```text
WMITER × TM × WNITER × TN
```

| 配置 | 每线程 `res` 元素数 |
|---|---:|
| Default | 128 |
| Small | 32 |
| Irregular | 64 |

Default 配置单是结果累加器就有 128 个 FP32 值，再加 `reg_m`、`reg_n`、指针和索引，寄存器压力可能很高。后续使用 Nsight Compute 或 `-Xptxas=-v` 时，需要重点观察 Register 数量、Occupancy 和 Spill。

## 8.5 读完后的自测

不看源码，独立完成以下推导：

1. Default 一个 Block 有几个 Warp？
2. 每个 Warp 计算 C 中多大的区域？
3. 每个线程最终维护多少个 C 元素？
4. Default 一个 Block 共维护多少个 C 元素？
5. 为什么结果应等于 `BM×BN`？

---

# 9. 第五阶段：从 Kernel 文件底部向上读

核心文件：

```text
src/kernels/gemm_warp_tile.cu
```

推荐的文件内阅读顺序：

```text
第 383～397 行：Shape Dispatch
第 373～381 行：Grid/Block 与 Kernel Launch
第 175～234 行：Kernel 外壳、索引和内存资源
第 314～370 行：Edge Path
第 103～137 行：compute_edge
第 40～101 行：compute / compute_k
第 147～173 行：异步加载
第 9～34 行：cp.async PTX helper
第 235～313 行：Fast Path 与写回
```

---

# 10. 第六阶段：先看启动与 Shape Dispatch

## 10.1 `launch_gemm_warp_tile_cfg<Cfg>`：第 373～381 行

```cpp
dim3 blockDim(Cfg::NUM_THREADS);
dim3 gridDim(CEIL_DIV(N, Cfg::BN), CEIL_DIV(M, Cfg::BM));
```

理解：

- `gridDim.x` 沿 C 的 N 方向切分；
- `gridDim.y` 沿 C 的 M 方向切分；
- 每个 Block 负责 `BM×BN` 的 C Tile；
- `CEIL_DIV` 保证边界不足一个完整 Tile 时仍会启动一个 Block。

例如 Default 配置，`M=N=2048`：

```text
grid.x = 2048 / 256 = 8
grid.y = 2048 / 128 = 16
总 Block 数 = 128
每个 Block 线程数 = 256
```

## 10.2 `launch_gemm_warp_tile`：第 383～397 行

分发规则：

```text
非对齐且 M,N,K <= 3072
→ IrregularConfig

否则 M,N,K <= 1024
→ SmallConfig

其余
→ DefaultConfig
```

注意 `irregular_medium` 判断放在 `small_problem` 前面，因此一个 257 方阵会进入 Irregular，而不是 Small。

### 自测

判断以下 Shape 使用哪套配置：

| Shape | 配置 |
|---|---|
| `512×512×512` | Small |
| `257×257×257` | Irregular |
| `2048×2048×2048` | Default |
| `2047×2047×2047` | Irregular |
| `4096×4096×4095` | Default，即使 K 非对齐也超过 Irregular 的规模阈值 |

---

# 11. 第七阶段：理解 Kernel 外壳和坐标系统

阅读第 175～234 行。

## 11.1 模板配置展开

Kernel 先把 `Cfg::BM` 等模板常量变成本地 `constexpr`。模板使不同配置在编译期生成不同 Kernel 版本，循环也更容易被编译器展开。

## 11.2 Block Tile 坐标

```cpp
c_row = blockIdx.y;
c_col = blockIdx.x;
```

对应 C 中 Block Tile 左上角：

```text
全局行起点 = c_row × BM
全局列起点 = c_col × BN
```

A 和 B 的 Tile 起点：

```cpp
A_blk = A + c_row * BM * K;
B_blk = B + c_col * BN;
```

- A 只随 C 的 Block 行变化；
- B 只随 C 的 Block 列变化；
- 随后再沿 K 方向不断移动。

## 11.3 边界大小与 Fast Path

```cpp
m_edge = ...;
n_edge = ...;
fast = 完整 BM × BN Tile，并且 K、N 满足 4 对齐；
```

这里的 `m_edge/n_edge` 表示当前 Block 真正有效的行列数。

## 11.4 Warp 坐标

```cpp
warp_idx = threadIdx.x / 32;
warp_row = warp_idx / (BN / WN);
warp_col = warp_idx % (BN / WN);
```

对于 Default：

```text
BN/WN = 256/64 = 4
warp_idx 0~3 → warp_row=0, warp_col=0~3
warp_idx 4~7 → warp_row=1, warp_col=0~3
```

形成 2×4 的 Warp Tile 网格。

## 11.5 Lane 到 Thread Tile 的映射

```cpp
tiwarp = threadIdx.x % 32;
tcol = tiwarp % (WSUBN / TN);
trow = tiwarp / (WSUBN / TN);
```

对于 Default：

```text
WSUBN/TN = 16/4 = 4
tcol = lane % 4，范围 0~3
trow = lane / 4，范围 0~7
```

所以一个 Warp 的 32 个 Lane 被看成 `8×4` 的二维线程布局。

### Lane 5 示例

```text
lane = 5
trow = 5 / 4 = 1
tcol = 5 % 4 = 1
```

它在每个 N 子区域中负责：

```text
行：trow × TM = 8 开始，共 8 行
列：tcol × TN = 4 开始，共 4 列
```

因为 `WNITER=4`，它还会在列方向跨 4 个子区域，最终负责 `8×16=128` 个输出。

## 11.6 Shared Memory 和 Register

```cpp
AS[2][BM*BK]
BS[2][BK*BN]
reg_m[WMITER*TM]
reg_n[WNITER*TN]
res[WMITER*TM*WNITER*TN]
```

读到这里必须画图，不要仅看数组长度。

---

# 12. 第八阶段：先看更容易理解的 Edge Path

阅读第 314～370 行。虽然它性能较慢，但逻辑比 `cp.async` Fast Path 更直观，适合用来建立正确数据流。

## 12.1 沿 K 维切 Tile

```cpp
for (int k = 0; k < K; k += BK)
```

每次处理 `BK` 个 K 元素。若最后不足 BK，`k_rem` 表示剩余有效深度。

## 12.2 清空 Shared Memory

```cpp
AS[0][i] = 0.f;
BS[0][i] = 0.f;
__syncthreads();
```

边界外的数据保持 0，这样后续乘法不会污染结果。

## 12.3 A Tile 加载

逻辑布局为：

```text
AS[kk][m] = A[m][k + kk]
```

代码地址：

```cpp
AS[0][kk*A_STRIDE + mb+mm] = A_tile[(mb+mm)*K + kk];
```

这里是看懂整个项目最重要的一条索引。

- `mb+mm`：A 的行，即输出 C 的 M 方向；
- `k+kk`：A 的列，即归约 K 方向；
- Shared Memory 中把 A 按 `[K][M]` 形式存放，便于计算阶段固定 `d` 后读取多行 A。

## 12.4 B Tile 加载

```text
BS[kk][n] = B[k + kk][n]
```

代码：

```cpp
BS[0][kk*BN + nb+nn] = B_tile[kk*N + nb+nn];
```

B 本身的连续方向就是 N，因此不需要像 A 一样改变逻辑布局。

## 12.5 计算与写回

加载完后调用 `compute_edge`，所有 K Tile 累加完后，根据 `grow/gcol` 检查全局边界，再执行：

```cpp
C = alpha * res + beta * C;
```

## 12.6 必须手工跟踪一个小例子

假设为了理解，把概念参数缩小成：

```text
BM=4, BN=4, BK=2
M=N=K=4
```

对 C 的左上 4×4 Block：

```text
第一次 K Tile：k=0，加载 A[:,0:2] 和 B[0:2,:]
第二次 K Tile：k=2，加载 A[:,2:4] 和 B[2:4,:]
res 累加两次后得到完整 C Tile
```

虽然仓库实际参数更大，但数据流完全相同。

### 读完后的自测

回答：

1. 为什么 AS 的逻辑布局是 `[BK][BM]`？
2. 为什么 BS 是 `[BK][BN]`？
3. 为什么每个 K Tile 结束后不能清空 `res`？
4. 为什么边界外数据填 0？
5. `__syncthreads()` 分别保护了什么？

---

# 13. 第九阶段：理解 `compute_edge`、`compute_k` 和 `compute`

这三个函数的核心计算完全一致，差别只在边界处理。

## 13.1 固定一个 K 深度 `d`

对每个 `d`：

1. 从 AS 取当前线程需要的若干 A 值到 `reg_m`；
2. 从 BS 取当前线程需要的若干 B 值到 `reg_n`；
3. 对 `reg_m × reg_n` 做 Outer Product；
4. 累加到 `res`。

## 13.2 为什么是 Outer Product

对于固定的 `d`：

```text
A 的多个行元素：a0, a1, ...
B 的多个列元素：b0, b1, ...
```

需要更新：

```text
res[i,j] += ai × bj
```

因此一个 A 值可以复用给多个 B 值，一个 B 值也可以复用给多个 A 值。

## 13.3 三个函数的区别

| 函数 | 使用场景 | 特点 |
|---|---|---|
| `compute` | 完整 BK 的快速路径 | K 循环编译期固定，完全展开 |
| `compute_k` | Fast Path 的 K 尾部 | 只循环到 `k_lim` |
| `compute_edge` | M/N/K 边界路径 | 读取 AS/BS 时检查有效行列 |

## 13.4 `res` 的一维下标

```cpp
res[(row_index) * (WNITER*TN) + col_index]
```

把当前线程负责的二维结果 Tile 展平成一维数组。

建议自己画出 Default 下：

```text
res[8][16]
```

然后把 `wr、rm、wc、rn` 映射到这张表中。

---

# 14. 第十阶段：最后再看 Fast Path

Fast Path 是项目优化价值最集中的部分，但不适合最先阅读。

## 14.1 先看 PTX helper：第 9～34 行

### `smem_u32addr`

把普通 C++ Shared Memory 指针转换成 PTX `cp.async` 所需的 32 位 Shared 地址。

### `cp_async16`

```text
Global Memory → Shared Memory
每次复制 16 Byte = 4 个 float
```

`guard` 被转换为 PTX predicate，决定是否执行复制。

### `cp_async_commit`

提交当前异步复制组。

### `cp_async_wait_group<N>`

等待未完成的异步复制组减少到指定数量。

当前代码使用 `wait_group<0>`，即在使用相应 Shared Memory 前等待所有未完成组完成。实现了异步预取，但流水线比较保守。

## 14.2 `load_tile_async`：第 147～173 行

线程协作加载：

- 每次 `cp_async16` 搬 4 个 float；
- A 目标布局为 `[BK][BM]`；
- B 目标布局为 `[BK][BN]`；
- A、B 的复制最后一起 `commit_group`。

重点推导：

```cpp
row_stride_a = NUM_THREADS / (BM / 4);
row_stride_b = NUM_THREADS / (BN / 4);
```

Default 下：

```text
row_stride_a = 256 / 32 = 8
row_stride_b = 256 / 64 = 4
```

因此：

- A：256 个线程恰好覆盖 `BK×(BM/4)=8×32=256` 个 `float4`；
- B：每个线程需要循环两次，覆盖 `BK×(BN/4)=8×64=512` 个 `float4`。

## 14.3 双缓冲：第 235～265 行

流程：

```text
预加载 Tile 0 到 Buffer 0
等待 Tile 0 完成

循环 t：
  当前计算 Buffer cur
  若存在下一个 Tile：预取到 Buffer nxt
  使用 cur 执行 compute
  等待 nxt，并同步线程
```

`cur=t&1` 和 `nxt=1-cur` 在两个 Shared Buffer 之间交替。

## 14.4 K 尾部：第 267～295 行

理论上 Fast Path 条件包含 `K%4==0`，而 `BK=8`，所以 K 可能是 4 的倍数但不是 8 的倍数，例如 K=12，此时：

```text
k_tiles = 1
k_rem = 4
```

完整 Tile 使用异步加载，最后 4 个 K 元素使用标量尾部路径。

## 14.5 `float4` 写回：第 297～312 行

一次读取和写回 C 中 4 个连续 FP32：

```cpp
float4 v = ...;
v.x = alpha*res[...] + beta*v.x;
...
```

优点：

- 连续、向量化访问；
- 减少 Load/Store 指令；
- 更容易形成合并访存。

前提：地址和列方向满足 16 Byte 对齐。Fast Path 要求 `N%4==0`，线程列偏移也是 4 的倍数。

---

# 15. 阅读时必须识别的代码疑点

下面这些问题不是为了否定项目，而是训练你从“看懂代码”升级到“审查代码”。

## 15.1 Fast Path 的 A 地址与 Edge Path 不一致

代码注释和计算函数都要求：

```text
AS[k][m] = A[m][k]
```

Edge Path 使用：

```cpp
A_tile[(mb+mm)*K + kk]
```

K 尾部也使用：

```cpp
A_tail[(inner_col_a+mm)*K + k]
```

但 Fast Path 的异步加载使用：

```cpp
A_tile + k*K + inner_col_a
```

它更像读取：

```text
A[k][m:m+4]
```

而不是：

```text
A[m:m+4][k]
```

对于 Row-major A，两者通常不等价。这是需要通过随机矩阵测试和手工索引验证的重大疑点。

当前 A 全部为 1，错误读取其他位置仍然得到 1，因此现有正确性测试可能无法发现它。

## 15.2 `K < BK` 时 Fast Path 可能预加载越界

Fast Path 进入后会无条件预加载 Tile 0，然后才根据：

```cpp
k_tiles = K / BK;
```

进入循环。若 `0<K<BK` 且满足 Fast Path 其他条件，`k_tiles=0`，但完整 BK Tile 的预加载已经发生。当前测试没有覆盖这一场景。

## 15.3 Benchmark 只测试方阵和常数输入

缺少：

- 随机正负数；
- `M、N、K` 独立变化；
- `alpha != 1`；
- `beta != 0`；
- 小 M 的 Decode Shape；
- NaN/Inf；
- 相对误差；
- Compute Sanitizer。

## 15.4 构建脚本不完整

CMake 只构建 `test_perf_gemm`，没有构建 README 中提到的 `microbenchmark/fair_compare_sgemm`。`microbenchmark/build.sh` 只是：

```bash
nvcc ${1} -arch sm_${2} -Xptxas=-v
```

它没有自动链接项目 Kernel、include 路径和 cuBLAS。学习时不要认为 README 中所有命令都一定能在干净环境直接复现。

## 15.5 GPU 架构硬编码为 `sm_89`

在不同 GPU 上运行前必须确认实际 Compute Capability，并调整 CMake。不能机械照抄。

---

# 16. 第十一阶段：学习性能 Benchmark

## 16.1 `microbenchmark/fair_compare_sgemm.cu`

它和主测试文件非常接近，但：

- `repeat=20`，主测试是 5；
- cuBLAS Handle 只创建一次；
- 输出单次平均毫秒数；
- 输出格式便于 Shell 脚本解析。

阅读重点不是重复理解 CUDA API，而是理解一个相对公平的 benchmark 至少需要：

1. 相同输入；
2. 两边分别 warmup；
3. GPU Event 计时；
4. 计时区间不包含 Host 初始化和 Host-Device 拷贝；
5. 多次重复；
6. 校验输出；
7. 保存每个 Shape 的延迟和吞吐。

## 16.2 `tests/compare_fair_runs.sh`

脚本连续运行 benchmark，使用 `awk` 统计：

- 多轮 `mean_ratio` 的均值；
- 总体标准差；
- 257、511、1000、1234、2047 等关键 Shape 的平均性能比。

需要理解：

> 多轮均值和标准差能衡量运行稳定性，但不能修复 benchmark Shape 不真实、输入不充分或 Kernel 错误等方法学问题。

## 16.3 `tools/generate_stable_baseline.py`

作用：

- 从实时程序输出或已有 CSV 读取数据；
- 聚合不同运行；
- 写 `stable_baseline_ratios.csv`；
- 写摘要；
- 绘制性能比例曲线。

该文件最后看即可，它负责结果展示，不参与 GEMM 计算。

---

# 17. 第十二阶段：最后学习 GPU 存储层次 Microbenchmark

这些程序大量使用 Inline PTX，不应在核心 Kernel 之前阅读。

## 17.1 建议顺序

```text
smem_latency.cu
→ smem_bandwidth.cu
→ l1cache_latency.cu
→ l2cache_latency.cu
→ l2cache_bandwidth.cu
→ dram_latency.cu
→ dram_bandwidth.cu
```

从 Shared Memory 开始，因为它与 GEMM Kernel 的关系最直接。

## 17.2 阅读这些代码的统一方法

每个程序只回答五个问题：

1. 被测存储层级是什么？
2. 怎样避免编译器删除无用加载？
3. 怎样避免并行请求把单次访问延迟隐藏？
4. 使用 CUDA Event 还是 `%clock` 计时？
5. 最终结果是 cycle、Byte/cycle 还是 GB/s？

## 17.3 与 GEMM 的关系

GEMM 之所以进行分块，是因为：

- 直接从 DRAM 为每次 FMA 读取 A、B，算术强度太低；
- 将 A、B Tile 放入 Shared Memory，可以被多个线程反复读取；
- 再放入寄存器，可以在单线程内部进一步复用；
- 性能优化的核心不是“计算公式改变”，而是“减少昂贵层级的数据搬运”。

---

# 18. 推荐的三遍阅读法

## 第一遍：只建立调用链

只读：

```text
tests/test_perf_gemm.cpp
include/gemm/launcher.h
src/launcher.cu
include/gemm/kernels/gemm_warp_tile.h
src/kernels/gemm_warp_tile.cu 的 373～397 行
```

目标：知道函数怎么调用，不研究每个线程。

最终能够画出：

```text
main
→ launch_gemm
→ launch_gemm_warp_tile
→ launch_gemm_warp_tile_cfg<Cfg>
→ gemm_warp_tile_kernel<Cfg>
```

## 第二遍：只研究分块与计算

读：

```text
gemm_warp_tile_config.h
Kernel 175～234 行
Edge Path 314～370 行
compute 系列 40～137 行
```

目标：弄懂 Block、Warp、Thread 分别计算 C 的哪一块，以及一份 A/B 数据如何复用。

第二遍暂时把 `cp.async` 当成“把 Tile 搬到 Shared Memory”的黑盒。

## 第三遍：研究优化细节和正确性

读：

```text
PTX helper 9～34 行
load_tile_async 147～173 行
Fast Path 235～313 行
Benchmark、Microbenchmark、results
```

目标：理解异步加载、双缓冲、向量化写回、寄存器压力，并审查索引和 benchmark 是否可靠。

---

# 19. 必须完成的手工推演任务

## 19.1 推演一个 Block

使用：

```text
M=N=K=2048
DefaultConfig
blockIdx = (2, 3)
```

计算：

```text
C Block 行起点 = 3 × 128 = 384
C Block 列起点 = 2 × 256 = 512
A_blk 指向 A 第 384 行
B_blk 指向 B 第 512 列
```

这个 Block 负责：

```text
C[384:512, 512:768]
```

## 19.2 推演一个 Warp

选择 `warp_idx=5`：

```text
warp_row = 5 / 4 = 1
warp_col = 5 % 4 = 1
```

该 Warp 负责当前 Block 内：

```text
行 [64,128)
列 [64,128)
```

对应全局 C：

```text
行 [448,512)
列 [576,640)
```

## 19.3 推演一个 Lane

选择 lane 5：

```text
trow=1
tcol=1
```

该线程在 Warp Tile 中负责 8 行，并在四个 N 子区域中各负责连续 4 列，合计 `8×16` 个结果。

建议把 lane 0、lane 5、lane 31 都推一遍，检查是否覆盖完整 64×64 且没有重复或遗漏。

## 19.4 推演一个 K Tile

对于 `t=0`：

```text
A Tile = A[384:512, 0:8]
B Tile = B[0:8, 512:768]
```

对于 `t=1`：

```text
A Tile = A[384:512, 8:16]
B Tile = B[8:16, 512:768]
```

每轮都会更新同一个 `res`，直到遍历完整 K。

---

# 20. 建议边学边做的代码实验

不要直接修改核心文件。先新建独立实验文件，逐步验证概念。

## 实验 1：CPU GEMM

实现三层循环：

```text
for m
  for n
    for k
```

验证 Row-major 地址。

## 实验 2：Naive CUDA GEMM

- 一个线程计算一个 `C[m,n]`；
- 不使用 Shared Memory；
- 使用随机小矩阵和 CPU 结果校验。

目的：先理解 Grid/Block 到 C 元素的映射。

## 实验 3：Block Tile Shared Memory GEMM

- 例如 `16×16` Tile；
- A、B 各加载一块 Shared Memory；
- 一个线程仍只计算一个 C 元素。

目的：理解 K Tile 和 `__syncthreads()`。

## 实验 4：Thread Tile

让一个线程计算 `TM×TN` 个结果，理解寄存器复用。

## 实验 5：Warp Tile 映射

暂时不使用 `cp.async`，只把当前项目的 Warp/Thread 坐标映射复刻出来。

## 实验 6：验证仓库 Fast Path

将测试输入改为固定随机数，并增加小矩形，例如：

```text
M=128, N=256, K=16
M=64, N=128, K=12
M=65, N=129, K=9
M=4, N=8, K=4
```

与 CPU 或 cuBLAS 比较，并运行 Compute Sanitizer。

## 实验 7：统计编译资源

编译时加入：

```bash
-Xptxas=-v
```

观察每个 Kernel 的：

- Registers per thread；
- Shared Memory；
- Spill stores/loads。

---

# 21. 看懂项目后应能回答的面试式问题

1. 该项目计算的数学公式是什么？
2. 为什么 GEMM 的 FLOP 数约为 `2MNK`？
3. 一个 Block 负责 C 的哪一块？
4. Default 配置为什么需要 8 个 Warp？
5. 一个 Default Warp 为什么正好覆盖 `64×64`？
6. 一个 Default 线程为什么维护 128 个结果？
7. AS 为什么逻辑上存成 `[BK][BM]`？
8. BS 为什么存成 `[BK][BN]`？
9. `reg_m`、`reg_n` 和 `res` 分别是什么？
10. `compute` 为什么是 Outer Product？
11. `cp.async` 搬运的源和目的存储层级是什么？
12. 双缓冲隐藏的是什么延迟？
13. 为什么 Fast Path 使用 `float4`？
14. Edge Path 为什么慢？
15. Shape Dispatch 为什么需要三套配置？
16. Default 配置可能遇到什么寄存器问题？
17. cuBLAS 的 Row-major 适配为什么交换 A/B？
18. 当前测试为什么可能掩盖索引错误？
19. 现有 74.6% 平均性能比为什么不能代表所有 Shape？
20. 如何把项目升级为更贴近 LLM 推理的算子项目？

不能清楚回答这些问题，就说明还没有真正看懂。

---

# 22. 最终推荐的实际执行清单

按下面顺序逐项完成，不要跳到 `cp.async`：

- [ ] 手写 CPU GEMM，并确认 Row-major 地址；
- [ ] 学会基础 CUDA 内存管理和 Kernel Launch；
- [ ] 阅读 `README.md` 与 `CMakeLists.txt`；
- [ ] 精读 `tests/test_perf_gemm.cpp`；
- [ ] 画出 Host/Device 数据和调用链；
- [ ] 阅读 `types.h`、`launcher.h`、`launcher.cu`；
- [ ] 精确推导三套 WarpTileConfig；
- [ ] 从 Kernel 文件底部看 Shape Dispatch；
- [ ] 看 Grid/Block、Block/Warp、Warp/Thread 坐标；
- [ ] 先看 Edge Path 的 A/B 标量加载；
- [ ] 看 `compute_edge` 的 Outer Product；
- [ ] 用 Default 配置手推一个 Block、Warp 和 Lane；
- [ ] 再看 `cp.async` helper 和异步加载；
- [ ] 理解双缓冲和 K remainder；
- [ ] 理解 `float4` 写回；
- [ ] 用随机矩形数据重新验证正确性；
- [ ] 检查 Fast Path 的 A 地址疑点；
- [ ] 阅读公平 Benchmark 和多轮统计脚本；
- [ ] 最后阅读 GPU 存储层次 Microbenchmark；
- [ ] 使用 Nsight Compute 分析 Register、Occupancy、Memory Throughput 和 Stall Reason。

---

# 23. 一句话总结最合理的学习路线

> 先从 `tests/test_perf_gemm.cpp` 看清楚一次完整 GEMM 实验，再沿 `launch_gemm → launch_gemm_warp_tile → gemm_warp_tile_kernel` 追踪调用；进入 Kernel 后先学配置和坐标映射，再读简单的 Edge Path 和 Outer Product，最后才学习 `cp.async`、双缓冲和向量化写回，并用随机矩形输入主动验证代码，而不是只接受仓库已有的性能结果。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-gemm算子|模块-gemm算子]]
- 关联阅读：[[outputs/项目整理/专题-05-GEMM与推理性能|专题-05-GEMM与推理性能]]

%% 项目关联导航：结束 %%
