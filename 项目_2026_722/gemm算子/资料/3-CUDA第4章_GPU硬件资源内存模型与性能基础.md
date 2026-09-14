# 一、本章在 CUDA 学习体系中的位置与学习目标

## 1. 目录识别与编号整理

根据截图，本章包含以下内容：

- 4.1 GPU 硬件资源
- 4.2 CUDA 内存模型概述
- 4.3 寄存器和本地内存
- 4.4 全局内存
- 4.5 共享内存
- 4.6 常量内存
- 4.7 GPU 缓存
- 4.8 计算资源分配
- 4.9 延迟隐藏
- 4.10 避免线程束分化

截图中“4.9”出现了两次，同时没有“4.8”。按照知识顺序，建议把“计算资源分配”整理为 4.8，把“延迟隐藏”整理为 4.9。这只是课程编号问题。

本章的主题不是继续增加 CUDA 语法，而是理解：

> GPU 上有哪些计算和存储资源，线程如何使用这些资源，以及资源使用方式为什么会决定算子性能。

前面的章节主要解决：

- 如何编写和启动核函数；
- Grid、Block、Thread 如何组织；
- 如何计算线程索引；
- 如何申请显存、传输数据；
- 如何进行错误检查和计时。

本章进一步解释：

- 为什么功能相同的两个 Kernel 性能会相差很大；
- 为什么全局内存访问模式会影响带宽；
- 为什么共享内存可以加速 GEMM、卷积和转置；
- 为什么“本地内存”并不是高速片上内存；
- 为什么寄存器使用过多会降低并发；
- 为什么 GPU 需要大量 Warp 隐藏延迟；
- 为什么同一个 Warp 中的不同分支会降低效率。

因此，本章是 CUDA 学习从“程序能正确运行”进入“硬件感知性能优化”的分界点。

---

## 2. 本章在 CUDA 算子学习体系中的位置

可以把后续学习路线简化为：

```text
CUDA 程序结构
  ↓
线程模型、索引和显存管理
  ↓
正确性验证、错误检查、计时
  ↓
本章：GPU 资源与内存层次
  ├── 寄存器和本地内存
  ├── 全局内存
  ├── 共享内存
  ├── 常量内存
  ├── Cache
  ├── Occupancy
  ├── 延迟隐藏
  └── Warp Divergence
  ↓
合并访存、共享内存分块、向量化
  ↓
Reduction、Transpose、Softmax、LayerNorm
  ↓
GEMM、Tensor Core
  ↓
FlashAttention、PagedAttention 等复杂算子
```

以 GEMM 为例，典型数据流是：

```text
A、B 大矩阵：全局内存
        ↓
线程块协作加载 Tile：共享内存
        ↓
线程读取当前子块：寄存器
        ↓
寄存器中执行乘加累积
        ↓
结果写回全局内存
```

这里同时使用了本章几乎全部知识：

- 全局内存容量大但延迟高；
- 共享内存用于线程块内的数据复用；
- 寄存器保存每个线程的中间结果；
- 共享内存和寄存器使用量会影响 Occupancy；
- 足够多的 Warp 用于隐藏数据加载延迟；
- 线程映射要保证合并访存并避免严重分化。

---

## 3. 本章必须建立的三个分析视角

### 3.1 数据位于哪里

分析每一类数据时，应回答：

- 它位于 Host 还是 Device？
- 位于全局内存、共享内存、寄存器还是常量内存？
- 哪些线程可见？
- 生命周期多长？
- 是否被重复使用？

例如：

```text
完整输入张量：全局内存
当前线程块重复使用的数据：共享内存
当前线程的累加结果：寄存器
小型只读卷积核：常量内存
```

### 3.2 线程如何访问数据

重点观察：

- 同一 Warp 的相邻线程是否访问相邻地址；
- 是否存在跨步访问；
- 是否存在大量重复加载；
- 是否可以通过共享内存复用；
- 是否满足对齐条件；
- 同一 Warp 是否读取相同常量地址。

### 3.3 资源使用如何影响并发

线程块进入 SM 时需要占用：

- 线程槽位；
- Warp 槽位；
- 寄存器；
- 共享内存；
- 线程块驻留槽位。

大致可以理解为：

```text
每线程寄存器数 × 每块线程数
→ 每块寄存器需求

每块共享内存用量
→ 一个 SM 可驻留多少块

每块线程数
→ 每块 Warp 数和线程上限
```

如果一个线程块占用资源过多，一个 SM 上可同时运行的线程块和 Warp 就会减少。

---

## 4. 学完本章后应具备的能力

完成本章后，应能够：

1. 画出寄存器、本地内存、共享内存、L1、L2、常量内存和全局内存的层次。
2. 解释每种内存的可见范围、生命周期和典型用途。
3. 明确本地内存是线程私有地址空间，但通常物理位于 Device Memory。
4. 分析 Warp 内存访问是否连续，判断是否有利于合并访存。
5. 使用 `__shared__` 和 `__syncthreads()` 完成块内协作。
6. 使用 `__constant__` 与 `cudaMemcpyToSymbol` 保存小型只读数据。
7. 理解寄存器压力、Register Spilling 和 Occupancy 的关系。
8. 使用 `-Xptxas=-v`、`cudaFuncGetAttributes` 查看资源使用。
9. 解释 GPU 如何通过调度其他 Warp 隐藏延迟。
10. 判断分支是否会造成同一 Warp 内的线程束分化。
11. 理解高 Occupancy 不等于高性能，最终结论必须通过 Benchmark 验证。

# 二、课程内容取舍与算子开发补充知识

## 1. 课程目录完整性分析

截图中的课程内容覆盖了 CUDA 性能基础的主要模块，但为了后续进行算子编写，还需要补充：

- SM、Warp Scheduler 和执行单元的关系；
- 内存作用域和生命周期；
- 合并访存；
- Shared Memory Bank 与 Bank Conflict；
- `__syncthreads()` 的正确使用条件；
- 寄存器溢出和本地内存；
- 常量缓存的广播机制；
- Occupancy 的限制因素；
- TLP 与 ILP；
- 算术强度和内存受限；
- 边界分支与主计算区域；
- 编译器资源报告和 Nsight Compute。

学习优先级如下。

### 必须熟练掌握

- 寄存器与本地内存的区别；
- 全局内存连续访问和合并访存；
- 共享内存的加载、同步和复用；
- 常量内存的基本使用；
- Warp 和线程束分化；
- 寄存器、共享内存、线程块大小与 Occupancy 的关系；
- 延迟隐藏；
- `__syncthreads()` 的使用规则。

### 当前需要理解

- L1、L2、常量缓存；
- Bank Conflict；
- Register Spilling；
- TLP 和 ILP；
- 向量化加载；
- 循环展开；
- 内存受限与计算受限；
- Occupancy API。

### 当前建立概念即可

- `cp.async`；
- 双缓冲和软件流水线；
- Tensor Memory Accelerator；
- Cluster Shared Memory；
- SASS 级依赖和 Scoreboard；
- 架构特定 Cache 策略。

---

## 2. GPU 硬件资源

## 2.1 GPU 结构概览

可以将 NVIDIA GPU 简化为：

```text
GPU
├── 多个 SM
│   ├── Warp Scheduler
│   ├── CUDA Core
│   ├── Tensor Core
│   ├── Load/Store Unit
│   ├── Special Function Unit
│   ├── 寄存器文件
│   └── 共享内存 / L1
├── 全 GPU 共享的 L2 Cache
└── Device Global Memory
```

不同架构的具体数量和组织会变化，但核心关系稳定：

- SM 是线程块实际执行和资源分配的核心单位；
- 一个线程块完整驻留在一个 SM 上；
- 一个线程块会被拆成若干 Warp；
- Warp 是硬件调度的重要粒度；
- 寄存器和共享内存是有限的片上资源；
- 全局内存容量大，但访问延迟较高。

---

## 2.2 Block、Warp 和 SM

假设：

```cpp
dim3 block(256);
```

一个线程块有：

```text
256 / 32 = 8 个 Warp
```

若某个 SM 同时驻留 4 个这样的线程块，则有：

```text
4 × 8 = 32 个活跃 Warp
```

但一个 SM 到底能驻留多少块，还受以下条件限制：

- 最大线程数；
- 最大 Warp 数；
- 最大线程块数；
- 每线程寄存器数；
- 每块共享内存；
- 架构资源分配粒度。

---

## 2.3 GPU 为什么需要大量线程

CPU 主要依靠：

- 强单线程性能；
- 大缓存；
- 分支预测；
- 乱序执行；

降低单个任务的延迟。

GPU 更强调：

- 大量线程；
- 高吞吐；
- SIMT；
- Warp 切换；
- 用其他工作隐藏当前 Warp 的等待。

当 Warp A 等待全局内存时，调度器可以执行 Warp B、C、D。前提是 SM 上存在足够多的可运行 Warp。

---

## 2.4 算子可能受到什么资源限制

### 计算受限

执行单元接近饱和，主要瓶颈是计算吞吐。

例如高复用的矩阵乘法可能趋向计算受限。

### 内存带宽受限

大量时间用于搬运数据，单位字节计算量少。

例如向量加法、矩阵加法通常是带宽型算子。

### 延迟受限

可并行工作不足、依赖链过长，无法有效隐藏等待。

### 资源受限

寄存器或共享内存使用过多，导致驻留 Warp 数减少。

---

## 3. CUDA 内存模型概述

## 3.1 内存层次

| 内存类型 | 典型位置 | 可见范围 | 生命周期 | 典型特点 |
|---|---|---|---|---|
| 寄存器 | SM 片上 | 单线程 | 线程期间 | 快、容量有限 |
| 本地内存 | Device Memory 地址空间 | 单线程 | 线程期间 | 线程私有，但通常较慢 |
| 共享内存 | SM 片上 | 同一 Block | Block 期间 | 低延迟、可协作 |
| 常量内存 | Device Memory + 常量缓存 | 全部线程只读 | 模块期间 | 适合小型广播数据 |
| 全局内存 | GPU DRAM | 所有线程 | 显式分配期间 | 容量大、延迟高 |
| L1 Cache | SM 附近 | 当前 SM 访问 | 硬件管理 | 缓存部分全局/本地访问 |
| L2 Cache | 全 GPU 共享 | 所有 SM | 硬件管理 | 减少 DRAM 访问 |

不能机械地只记“速度排序”，因为性能还取决于：

- Cache 是否命中；
- 访问是否合并；
- 是否发生 Bank Conflict；
- 是否发生寄存器溢出；
- 是否有足够 Warp 隐藏延迟。

---

## 3.2 作用域与生命周期

### 寄存器

```text
作用域：单线程
生命周期：线程执行期间
```

### 共享内存

```text
作用域：同一线程块
生命周期：线程块执行期间
```

### 全局内存

```text
作用域：整个 Device
生命周期：cudaMalloc 到 cudaFree
```

### 常量内存

```text
作用域：所有线程只读
生命周期：通常随 CUDA 模块存在
```

选择内存类型时，应匹配：

```text
数据容量
共享范围
复用程度
访问模式
生命周期
```

---

## 3.3 高性能算子的典型数据流

```text
全局内存
  ↓ 线程协作加载
共享内存
  ↓ 每线程读取
寄存器
  ↓ 计算
寄存器
  ↓ 写回
全局内存
```

这条数据流会在 GEMM、卷积、Softmax 和 Attention 中反复出现。

---

## 4. 寄存器和本地内存

## 4.1 寄存器

核函数中的简单局部标量通常被编译器放入寄存器：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float x_value = input[idx];
float result = x_value * scale + bias;
```

`idx`、`x_value`、`result` 很可能使用寄存器。

特点：

- 线程私有；
- 访问延迟低；
- 编译器自动分配；
- 总量有限；
- 每线程用量会影响可驻留线程数量。

---

## 4.2 寄存器压力

若每线程使用 `R` 个寄存器，每块有 `T` 个线程，则一块大致需要：

```text
R × T
```

个寄存器。

每线程寄存器越多，一个 SM 可同时驻留的线程块可能越少，从而降低活跃 Warp 数。

但不能盲目压低寄存器数，因为可能导致：

- 中间值反复计算；
- Register Spilling；
- 本地内存 Load/Store；
- 性能下降。

---

## 4.3 本地内存

CUDA 中“本地”表示线程私有，不表示高速片上存储。

以下代码可能使用本地内存：

```cpp
__global__ void kernel(float* output) {
    float temp[128];
    ...
}
```

常见原因：

- 大型线程私有数组；
- 动态索引数组；
- 编译器无法寄存器化；
- 寄存器不足；
- Register Spilling。

本地内存通常经过 L1/L2 访问，但物理上属于 Device Memory 地址空间，延迟可能较高。

---

## 4.4 Register Spilling

当编译器无法把所有值保存在寄存器中时，部分值会溢出到本地内存。

可以用：

```bash
nvcc -O3 -Xptxas=-v kernel.cu -o kernel
```

查看：

- registers；
- spill loads；
- spill stores；
- shared memory；
- local memory。

也可以使用：

```bash
nvcc --resource-usage ...
```

---

## 4.5 不要盲目限制寄存器

可以使用：

```bash
-maxrregcount=N
```

或者：

```cpp
__global__ __launch_bounds__(256, 2)
void kernel(...) {}
```

但限制太紧可能增加 Spilling。

正确流程是：

```text
查看资源
  ↓
定位瓶颈
  ↓
调整寄存器或线程块
  ↓
检查 Spilling 和 Occupancy
  ↓
重新 Benchmark
```

---

## 5. 全局内存

## 5.1 特点

全局内存即 GPU Device DRAM，特点是：

- 容量大；
- 所有线程可访问；
- 延迟高；
- 峰值带宽高；
- 可以被 L1/L2 缓存；
- 性能强烈依赖访问模式。

---

## 5.2 合并访存

理想访问：

```text
线程 0 → x[0]
线程 1 → x[1]
线程 2 → x[2]
...
线程 31 → x[31]
```

同一 Warp 的线程访问连续地址，硬件可以用较少内存事务完成。

不理想访问：

```text
线程 0 → x[0]
线程 1 → x[stride]
线程 2 → x[2 × stride]
...
```

当 `stride` 很大时，需要更多内存事务，实际带宽下降。

---

## 5.3 行优先矩阵的映射

矩阵线性索引：

```cpp
int index = row * cols + col;
```

因此通常令：

```text
threadIdx.x → col
threadIdx.y → row
```

让相邻线程沿行访问连续地址。

---

## 5.4 对齐与向量化

在满足对齐和边界条件时，可以使用：

```cpp
float2
float4
half2
```

例如一个线程一次处理 4 个 `float`：

```cpp
float4 value =
    reinterpret_cast<const float4*>(input)[idx];
```

潜在收益：

- 减少 Load/Store 指令；
- 增加每线程工作量；
- 提高内存吞吐利用。

但需要保证：

- 地址对齐；
- 长度可处理；
- 边界正确；
- 没有破坏连续访问；
- 编译器确实生成合适指令。

---

## 5.5 算子融合与全局内存流量

例如：

```text
T = X + Bias
Y = ReLU(T)
```

若分成两个 Kernel，中间张量 `T` 需要：

- 写入全局内存；
- 再从全局内存读取。

融合为一个 Kernel：

```text
读取 X 和 Bias
  ↓
寄存器内执行加法和 ReLU
  ↓
写回 Y
```

可以减少全局内存流量、显存占用和 Kernel Launch。这是 AI Infra 中算子融合的重要价值。

---

## 6. 共享内存

## 6.1 定义与特点

静态共享内存：

```cpp
__shared__ float tile[18][18];
```

动态共享内存：

```cpp
extern __shared__ float shared[];
```

启动时指定：

```cpp
kernel<<<grid, block, shared_bytes>>>(...);
```

特点：

- 同一 Block 可见；
- 位于 SM 片上；
- 低延迟；
- 容量有限；
- 需要程序员显式管理；
- 用量会影响驻留线程块数量。

---

## 6.2 典型用途

### 数据复用

GEMM 中同一 A/B Tile 被多个线程使用。

### 线程交换数据

Reduction 中线程写入局部值，再逐步合并。

### 布局转换

矩阵转置先连续读取到共享内存，再重新排列后连续写出。

### 邻域缓存

卷积或 Stencil 中，多个输出共享相邻输入。

---

## 6.3 `__syncthreads()`

典型模式：

```cpp
shared[tid] = input[idx];

__syncthreads();

float value = shared[other_tid];
```

它保证：

- 同一线程块中的线程到达同步点；
- 同步前的共享内存写入对同步后的线程可见；
- 所有线程到达后才继续。

危险代码：

```cpp
if (idx < n) {
    shared[tid] = input[idx];
    __syncthreads();
}
```

如果只有部分线程进入分支，其他线程没有到达同步点，可能产生未定义行为。

安全模式：

```cpp
if (idx < n) {
    shared[tid] = input[idx];
} else {
    shared[tid] = 0.0f;
}

__syncthreads();
```

---

## 6.4 Shared Memory Bank

共享内存划分为多个 Bank。

同一 Warp 的线程访问不同 Bank 时，可以并行服务；若多个线程访问同一 Bank 中的不同地址，可能发生 Bank Conflict。

矩阵转置中常用 Padding：

```cpp
__shared__ float tile[TILE][TILE + 1];
```

`+1` 改变每行起始 Bank，可减少转置访问冲突。

连续线程访问连续 `float` 通常较友好，但仍需用 Nsight Compute 验证。

---

## 6.5 共享内存不是自动缓存

使用共享内存必须明确：

```text
谁加载
加载什么
何时同步
谁复用
是否还需同步
```

若数据只使用一次，例如普通向量加法，先写共享内存通常没有收益，反而增加指令和同步开销。

---

## 7. 常量内存

## 7.1 基本用法

Device 端：

```cpp
__constant__ float c_filter[9];
```

Host 端：

```cpp
cudaMemcpyToSymbol(
    c_filter,
    h_filter,
    9 * sizeof(float)
);
```

Kernel 中：

```cpp
float coefficient = c_filter[k];
```

---

## 7.2 适用场景

常量内存适合：

- 数据量小；
- 只读；
- 多次使用；
- 同一 Warp 经常读取相同地址；
- 程序运行期间不频繁变化。

典型数据：

- 小卷积核；
- 查找表；
- 固定物理参数；
- 小型变换矩阵。

---

## 7.3 常量缓存广播

当同一 Warp 的线程读取相同常量地址时，常量缓存可高效广播。

例如：

```cpp
for (int k = 0; k < 9; ++k) {
    sum += values[k] * c_filter[k];
}
```

在一次循环迭代中，Warp 内线程都读取 `c_filter[k]`，适合常量内存。

若 Warp 内线程读取许多不同常量地址，访问可能被序列化，优势减弱。

---

## 8. GPU 缓存

## 8.1 L1 与 L2

典型访问路径：

```text
线程
  ↓
寄存器
  ↓
L1 / 共享内存相关片上资源
  ↓
L2
  ↓
全局内存
```

L1 更靠近 SM，L2 通常由整个 GPU 共享。

缓存由硬件管理，程序一般不能保证某个数据一直留在 Cache。

---

## 8.2 Cache 不能替代良好访问模式

即使存在 Cache，也应优先保证：

- 合并访存；
- 数据布局合理；
- 减少重复读取；
- 对齐；
- 数据复用。

Cache 是硬件补充，不是忽略访问模式的理由。

---

## 8.3 小工作集 Benchmark 的误区

如果输入较小并反复执行，数据可能长期驻留在 L2，测得结果不能代表真正的大规模 DRAM 访问性能。

做严谨测试时应考虑：

- 工作集是否大于 Cache；
- 是否重复访问同一数据；
- 是否需要刷新或改变输入；
- Kernel 时间和端到端时间是否分开。

---

## 8.4 `const __restrict__`

核函数参数可以写为：

```cpp
const float* __restrict__ input
```

- `const`：不通过该指针修改数据；
- `__restrict__`：承诺相关指针之间不发生别名。

这可能帮助编译器优化加载，但必须保证承诺真实，否则结果可能错误。

---

## 9. 计算资源分配

## 9.1 Occupancy

Occupancy 可以近似理解为：

```text
当前活跃 Warp 数
÷
硬件最大活跃 Warp 数
```

例如最大支持 64 个活跃 Warp，当前只能驻留 32 个：

```text
Occupancy = 50%
```

它反映延迟隐藏的潜力，但不是最终性能指标。

---

## 9.2 限制 Occupancy 的资源

### 每块线程数

线程块越大，每块 Warp 越多，可能减少每 SM 驻留块数。

### 每线程寄存器数

```text
寄存器/线程 × 线程/块
```

过大时会限制驻留块数。

### 每块共享内存

每块共享内存过大时，一个 SM 只能同时放入少量线程块。

### 架构上限

SM 还存在最大：

- 活跃线程数；
- 活跃 Warp 数；
- 活跃 Block 数。

---

## 9.3 查询核函数资源

```cpp
cudaFuncAttributes attr{};

cudaFuncGetAttributes(&attr, kernel);
```

常用字段：

```cpp
attr.numRegs
attr.sharedSizeBytes
attr.localSizeBytes
attr.maxThreadsPerBlock
```

可查看：

- 每线程寄存器数；
- 静态共享内存；
- 本地内存；
- Kernel 允许的最大线程数。

---

## 9.4 Occupancy API

```cpp
int active_blocks = 0;

cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &active_blocks,
    kernel,
    threads_per_block,
    dynamic_shared_bytes
);
```

再计算：

```cpp
int warps_per_block =
    (threads_per_block + warp_size - 1)
    / warp_size;
```

理论活跃 Warp：

```text
active_blocks × warps_per_block
```

但最终仍需真实 Benchmark。

---

## 9.5 线程块大小选择

常见一维候选：

```text
128、256、512
```

二维候选：

```text
16×16、32×8、32×16
```

选择时考虑：

- Warp Size 的整数倍；
- 数据布局；
- 资源使用；
- Grid 是否足够大；
- 是否形成连续访存；
- Benchmark 结果。

“256 个线程”只是常用起点，不是固定最优答案。

---

## 10. 延迟隐藏

## 10.1 Warp 切换

全局内存访问可能具有较长等待时间。

GPU 会尝试：

```text
Warp A 等待
  ↓
执行 Warp B
  ↓
Warp B 等待
  ↓
执行 Warp C
```

只要有其他就绪 Warp，执行单元就不必完全空闲。

---

## 10.2 TLP

TLP，即线程级并行，依靠大量线程和 Warp 提供可调度工作。

提高 TLP 的方式：

- 启动足够多线程块；
- Grid 规模大于 SM 数；
- 避免单块占用过多资源；
- 合理设置线程块大小。

---

## 10.3 ILP

ILP，即指令级并行，表示同一线程中存在相互独立的指令。

例如：

```cpp
float a0 = x[i0];
float a1 = x[i1];
float b0 = y[i0];
float b1 = y[i1];

float c0 = a0 + b0;
float c1 = a1 + b1;
```

两组计算相互独立，硬件可能交错执行。

增加每线程处理的数据可提高 ILP，但也会增加寄存器使用。

---

## 10.4 Occupancy 低时的问题

若只有少数 Warp：

```text
Warp A 等内存
Warp B 也等内存
```

没有其他就绪 Warp，SM 可能空闲。

更多 Warp 可以提供更多可切换工作。

---

## 10.5 高 Occupancy 不一定更快

若为了提高 Occupancy 强制减少寄存器，可能导致：

- Spilling；
- 本地内存访问；
- 中间值重复计算；
- 性能下降。

需要综合平衡：

```text
TLP
ILP
寄存器复用
共享内存复用
访存效率
```

---

## 11. 避免线程束分化

## 11.1 什么是 Warp Divergence

同一 Warp 中部分线程走 `if`，另一部分走 `else`：

```cpp
if (threadIdx.x % 2 == 0) {
    output[idx] = a[idx] + b[idx];
} else {
    output[idx] = a[idx] - b[idx];
}
```

奇偶线程交错，硬件可能先执行一条路径，再执行另一条路径，并分别屏蔽不参与当前路径的线程。

---

## 11.2 Warp 间分支与 Warp 内分支

如果每个 Warp 整体选择同一条路径，例如：

```cpp
if (threadIdx.x / 32 == 0) {
    ...
} else {
    ...
}
```

则主要是不同 Warp 处理不同任务，不是同一 Warp 内严重分化。

判断关键是：

> 分支条件在同一 Warp 内是否一致。

---

## 11.3 边界判断通常可以接受

```cpp
if (idx < n) {
    output[idx] = input[idx];
}
```

通常只有最后一个线程块中的少数线程分化，影响范围有限。

不应为了消除少量边界分支而增加大量复杂逻辑。

---

## 11.4 减少分化的方法

- 让相邻线程处理相同类型数据；
- 将差异很大的任务分成不同 Kernel；
- 对数据重新排序；
- 将完整 Tile 和边界 Tile 分开处理；
- 使用合适的无分支表达式；
- 避免按奇偶线程分配完全不同的工作。

最终要通过性能工具确认分支是否真的是瓶颈。

---

## 12. 本章常见误区

### 本地内存不是高速片上内存

“本地”表示线程私有。

### 共享内存不是自动 Cache

它需要程序员显式加载、同步和使用。

### 寄存器越多不一定越快

过多会减少并发，过少可能 Spilling。

### Occupancy 越高不一定越快

它只是延迟隐藏能力的参考。

### 全局内存带宽高不等于访问延迟低

GPU 依靠并行和 Warp 调度隐藏高延迟。

### 共享内存快不代表所有算子都应使用

没有数据复用时，共享内存可能增加开销。

### 存在分支不等于一定有严重分化

需要看同一 Warp 内线程是否选择不同路径。

# 三、章节综合实例

## 1. 实例目标：共享内存优化的 3×3 Stencil

实现：

```text
output[row, col]
=
Σ input[row + dy, col + dx] × filter[dy, dx]
```

其中：

```text
dy、dx ∈ {-1, 0, 1}
```

该实例类似小型二维卷积，可以综合使用：

- 全局内存保存输入和输出；
- 常量内存保存 3×3 系数；
- 共享内存保存当前 Tile 和 Halo；
- 寄存器累加结果；
- `__syncthreads()` 实现块内同步；
- 合并的全局内存加载；
- 资源查询和 Occupancy 估算；
- 边界分支与 Warp Divergence；
- 全局版本和共享内存版本 Benchmark。

---

## 2. 数据复用分析

若不使用共享内存，一个输出线程需要读取 9 个输入元素。

相邻输出的邻域高度重叠，因此同一输入会被多个线程重复使用。

对于 `16×16` 输出 Tile：

```text
直接版本：
256 个输出 × 9 次线程级读取

共享内存版本：
协作加载 18×18 = 324 个输入元素
然后计算 16×16 = 256 个输出
```

共享内存把重复的全局访问转换为片上复用。

---

## 3. 完整代码

保存为：

```text
stencil_memory_hierarchy.cu
```

```cpp
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t error = (call);                               \
    if (error != cudaSuccess) {                               \
        std::cerr << "CUDA error: "                           \
                  << cudaGetErrorString(error)                \
                  << "\nFile: " << __FILE__                   \
                  << "\nLine: " << __LINE__                   \
                  << std::endl;                               \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)

constexpr int FILTER_WIDTH = 3;
constexpr int RADIUS = 1;
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;
constexpr int SHARED_WIDTH = BLOCK_X + 2 * RADIUS;
constexpr int SHARED_HEIGHT = BLOCK_Y + 2 * RADIUS;

__constant__ float c_filter[FILTER_WIDTH * FILTER_WIDTH];

inline int ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

__global__ void stencil_global_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int height,
    int width
) {
    const int col =
        blockIdx.x * blockDim.x + threadIdx.x;

    const int row =
        blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || col >= width) {
        return;
    }

    float sum = 0.0f;

    #pragma unroll
    for (int fy = 0; fy < FILTER_WIDTH; ++fy) {
        #pragma unroll
        for (int fx = 0; fx < FILTER_WIDTH; ++fx) {
            const int in_row = row + fy - RADIUS;
            const int in_col = col + fx - RADIUS;

            float value = 0.0f;

            if (
                in_row >= 0 && in_row < height &&
                in_col >= 0 && in_col < width
            ) {
                value = input[in_row * width + in_col];
            }

            sum += value * c_filter[fy * FILTER_WIDTH + fx];
        }
    }

    output[row * width + col] = sum;
}

__global__ void stencil_shared_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int height,
    int width
) {
    __shared__ float tile[SHARED_HEIGHT][SHARED_WIDTH];

    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;

    const int block_col = blockIdx.x * BLOCK_X;
    const int block_row = blockIdx.y * BLOCK_Y;

    const int linear_tid = local_y * BLOCK_X + local_x;
    const int threads_per_block = BLOCK_X * BLOCK_Y;
    const int shared_elements = SHARED_WIDTH * SHARED_HEIGHT;

    // 整个线程块协作加载中心区域和 Halo。
    for (
        int linear_index = linear_tid;
        linear_index < shared_elements;
        linear_index += threads_per_block
    ) {
        const int shared_row =
            linear_index / SHARED_WIDTH;

        const int shared_col =
            linear_index % SHARED_WIDTH;

        const int global_row =
            block_row + shared_row - RADIUS;

        const int global_col =
            block_col + shared_col - RADIUS;

        float value = 0.0f;

        if (
            global_row >= 0 && global_row < height &&
            global_col >= 0 && global_col < width
        ) {
            value = input[global_row * width + global_col];
        }

        tile[shared_row][shared_col] = value;
    }

    // 所有线程都必须到达该同步点。
    __syncthreads();

    const int output_row = block_row + local_y;
    const int output_col = block_col + local_x;

    if (output_row < height && output_col < width) {
        // sum 通常位于寄存器。
        float sum = 0.0f;

        #pragma unroll
        for (int fy = 0; fy < FILTER_WIDTH; ++fy) {
            #pragma unroll
            for (int fx = 0; fx < FILTER_WIDTH; ++fx) {
                sum +=
                    tile[local_y + fy][local_x + fx]
                    * c_filter[fy * FILTER_WIDTH + fx];
            }
        }

        output[output_row * width + output_col] = sum;
    }
}

void stencil_cpu(
    const std::vector<float>& input,
    std::vector<float>& output,
    const std::vector<float>& filter,
    int height,
    int width
) {
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            float sum = 0.0f;

            for (int fy = 0; fy < FILTER_WIDTH; ++fy) {
                for (int fx = 0; fx < FILTER_WIDTH; ++fx) {
                    const int in_row = row + fy - RADIUS;
                    const int in_col = col + fx - RADIUS;

                    float value = 0.0f;

                    if (
                        in_row >= 0 && in_row < height &&
                        in_col >= 0 && in_col < width
                    ) {
                        value = input[in_row * width + in_col];
                    }

                    sum +=
                        value
                        * filter[fy * FILTER_WIDTH + fx];
                }
            }

            output[row * width + col] = sum;
        }
    }
}

bool check_result(
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    float atol,
    float rtol
) {
    if (expected.size() != actual.size()) {
        return false;
    }

    float max_error = 0.0f;
    std::size_t max_index = 0;

    for (std::size_t i = 0; i < expected.size(); ++i) {
        const float error =
            std::fabs(expected[i] - actual[i]);

        const float allowed =
            atol + rtol * std::fabs(expected[i]);

        if (error > max_error) {
            max_error = error;
            max_index = i;
        }

        if (error > allowed) {
            std::cerr
                << "Mismatch at " << i
                << ", expected = " << expected[i]
                << ", actual = " << actual[i]
                << ", error = " << error
                << '\n';
            return false;
        }
    }

    std::cout
        << "Maximum absolute error: "
        << max_error
        << " at index "
        << max_index
        << '\n';

    return true;
}

template <typename Kernel>
float benchmark_kernel(
    Kernel kernel,
    dim3 grid,
    dim3 block,
    const float* input,
    float* output,
    int height,
    int width,
    int warmup,
    int iterations
) {
    for (int i = 0; i < warmup; ++i) {
        kernel<<<grid, block>>>(
            input,
            output,
            height,
            width
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block>>>(
            input,
            output,
            height,
            width
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / iterations;
}

template <typename Kernel>
void print_kernel_resources(
    const char* name,
    Kernel kernel,
    int threads_per_block,
    const cudaDeviceProp& prop
) {
    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));

    int active_blocks = 0;

    CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            kernel,
            threads_per_block,
            0
        )
    );

    const int warps_per_block =
        ceil_div(threads_per_block, prop.warpSize);

    const int active_warps =
        active_blocks * warps_per_block;

    const int max_warps =
        prop.maxThreadsPerMultiProcessor / prop.warpSize;

    const double occupancy =
        max_warps > 0
            ? static_cast<double>(active_warps) / max_warps
            : 0.0;

    std::cout << "\nKernel: " << name << '\n';
    std::cout << "Registers per thread: "
              << attr.numRegs << '\n';
    std::cout << "Static shared memory: "
              << attr.sharedSizeBytes << " bytes\n";
    std::cout << "Local memory per thread: "
              << attr.localSizeBytes << " bytes\n";
    std::cout << "Active blocks per SM: "
              << active_blocks << '\n';
    std::cout << "Estimated occupancy: "
              << occupancy * 100.0 << "%\n";
}

int main() {
    const int height = 4096;
    const int width = 4096;
    const int warmup = 10;
    const int iterations = 100;

    const std::size_t elements =
        static_cast<std::size_t>(height) * width;

    const std::size_t bytes =
        elements * sizeof(float);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        std::cerr << "No CUDA device.\n";
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "GPU: " << prop.name << '\n';
    std::cout << "Compute capability: "
              << prop.major << "." << prop.minor << '\n';
    std::cout << "SM count: "
              << prop.multiProcessorCount << '\n';

    std::vector<float> h_input(elements);
    std::vector<float> h_cpu(elements);
    std::vector<float> h_global(elements);
    std::vector<float> h_shared(elements);

    std::mt19937 generator(42);
    std::uniform_real_distribution<float>
        distribution(-1.0f, 1.0f);

    for (float& value : h_input) {
        value = distribution(generator);
    }

    std::vector<float> h_filter = {
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
        2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f
    };

    stencil_cpu(
        h_input,
        h_cpu,
        h_filter,
        height,
        width
    );

    CUDA_CHECK(cudaMemcpyToSymbol(
        c_filter,
        h_filter.data(),
        h_filter.size() * sizeof(float)
    ));

    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    const dim3 block(BLOCK_X, BLOCK_Y);

    const dim3 grid(
        ceil_div(width, BLOCK_X),
        ceil_div(height, BLOCK_Y)
    );

    const int threads_per_block = BLOCK_X * BLOCK_Y;

    print_kernel_resources(
        "stencil_global_kernel",
        stencil_global_kernel,
        threads_per_block,
        prop
    );

    print_kernel_resources(
        "stencil_shared_kernel",
        stencil_shared_kernel,
        threads_per_block,
        prop
    );

    const float global_ms = benchmark_kernel(
        stencil_global_kernel,
        grid,
        block,
        d_input,
        d_output,
        height,
        width,
        warmup,
        iterations
    );

    CUDA_CHECK(cudaMemcpy(
        h_global.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    const bool global_ok =
        check_result(h_cpu, h_global, 1e-5f, 1e-5f);

    const float shared_ms = benchmark_kernel(
        stencil_shared_kernel,
        grid,
        block,
        d_input,
        d_output,
        height,
        width,
        warmup,
        iterations
    );

    CUDA_CHECK(cudaMemcpy(
        h_shared.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    const bool shared_ok =
        check_result(h_cpu, h_shared, 1e-5f, 1e-5f);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nGlobal kernel: "
              << global_ms << " ms\n";
    std::cout << "Shared kernel: "
              << shared_ms << " ms\n";

    if (shared_ms > 0.0f) {
        std::cout << "Speedup: "
                  << global_ms / shared_ms
                  << "x\n";
    }

    std::cout << "Global correctness: "
              << (global_ok ? "PASSED" : "FAILED")
              << '\n';

    std::cout << "Shared correctness: "
              << (shared_ok ? "PASSED" : "FAILED")
              << '\n';

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return global_ok && shared_ok
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
```

---

## 4. 编译与运行

```bash
nvcc \
    -std=c++17 \
    -O3 \
    -lineinfo \
    -Xptxas=-v \
    stencil_memory_hierarchy.cu \
    -o stencil_memory_hierarchy
```

运行：

```bash
./stencil_memory_hierarchy
```

`-Xptxas=-v` 用于查看：

- 每线程寄存器数；
- 静态共享内存；
- 本地内存；
- Spill Load；
- Spill Store。

---

## 5. 按内存层次理解代码

### 全局内存

```cpp
d_input
d_output
```

保存完整输入和输出。

### 常量内存

```cpp
__constant__ float c_filter[9];
```

保存小型只读滤波系数。同一循环位置中，Warp 内线程读取同一个系数，适合广播。

### 共享内存

```cpp
__shared__ float tile[18][18];
```

保存 `16×16` 输出区域需要的中心数据和一圈 Halo。

### 寄存器

```cpp
float sum = 0.0f;
```

以及索引变量通常位于寄存器。

### 本地内存

代码没有显式声明本地内存，但如果寄存器压力过大，编译器可能产生本地内存或 Spill。应通过资源报告检查。

### Cache

Global Kernel 的重复读取可能命中 L1/L2，因此共享内存版本不保证在所有 GPU 和输入规模上都有巨大加速。

---

## 6. 共享内存加载过程

共享 Tile 大小：

```text
18 × 18 = 324 个元素
```

线程块大小：

```text
16 × 16 = 256 个线程
```

使用：

```cpp
for (
    int linear_index = linear_tid;
    linear_index < shared_elements;
    linear_index += threads_per_block
)
```

让部分线程加载第二个元素，从而覆盖完整 Tile 和 Halo。

加载后必须：

```cpp
__syncthreads();
```

确保所有共享内存数据都已准备好。

同步之后再让有效线程执行输出计算。

---

## 7. 为什么同步不能放入输出边界分支

错误方式：

```cpp
if (output_valid) {
    __syncthreads();
    ...
}
```

边缘线程块中部分线程不满足条件，不能到达同步点。

正确顺序：

```text
所有线程协作加载
  ↓
所有线程执行 __syncthreads()
  ↓
有效输出线程计算
```

这也是共享内存 Kernel 最重要的正确性规则之一。

---

## 8. 资源分配与 Occupancy

共享内存版本每块静态共享内存约为：

```text
18 × 18 × 4 = 1296 字节
```

程序还会查询：

```text
Registers per thread
Static shared memory
Local memory per thread
Active blocks per SM
Estimated occupancy
```

分析时要注意：

- Shared Kernel 可能使用更多寄存器；
- 使用共享内存会占用 SM 片上资源；
- Occupancy 可能下降；
- Occupancy 下降并不自动代表性能下降；
- 最终要看真实 Kernel 时间。

---

## 9. 延迟隐藏分析

每块：

```text
256 个线程 = 8 个 Warp
```

如果一个 SM 能驻留多个线程块，就会有多个 Warp 可供调度。

在全局内存加载期间，一部分 Warp 等待数据时，调度器可执行其他 Warp。

共享内存版本通过减少重复全局内存访问，降低了需要隐藏的高延迟操作数量。

---

## 10. 线程束分化分析

主要分支有：

```cpp
if (global_row ... global_col ...)
```

以及：

```cpp
if (output_row < height && output_col < width)
```

内部完整线程块中的线程基本走相同路径；只有边界线程块出现部分分化。

这类边界分化影响有限，通常可以接受。

若性能分析证明边界逻辑明显影响主路径，可以进一步拆分：

```text
内部区域 Kernel
+
边界 Kernel
```

但不应在没有测量的情况下过早增加复杂度。

---

## 11. 为什么共享内存版本不一定始终更快

共享内存版本虽然减少重复全局加载，但增加了：

- 共享内存写入；
- `__syncthreads()`；
- 地址计算；
- 共享内存读取；
- 可能更高的寄存器压力；
- 可能降低 Occupancy。

同时 Global Kernel 的重复数据可能命中 Cache。

因此结论必须来自：

```text
正确性验证
+
稳定 Benchmark
+
资源报告
+
Nsight Compute
```

而不能只根据“共享内存更快”作出判断。

---

## 12. 推荐实验

### 实验一：改变线程块形状

尝试：

```text
8×8
16×16
32×8
32×16
```

记录：

- 平均时间；
- 寄存器数；
- 共享内存；
- Occupancy；
- 加速比。

### 实验二：取消常量内存

把滤波系数放在普通全局内存中，比较时间。

### 实验三：制造寄存器压力

增加线程私有数组，观察：

```text
local memory
spill loads
spill stores
```

只用于学习，不作为优化方案。

### 实验四：改变输入尺寸

比较：

```text
4096×4096
4097×4097
```

观察边界线程比例变化。

### 实验五：使用 Nsight Compute

重点观察：

- Global Load Efficiency；
- L1/L2 命中；
- Shared Memory 吞吐；
- Warp Stall 原因；
- Branch Efficiency；
- Achieved Occupancy。

---

## 13. 从本实例继续学习

### 矩阵转置

学习：

- 合并读取和写入；
- Shared Memory Padding；
- Bank Conflict。

### Reduction

学习：

- 块内归约；
- Warp Shuffle；
- 同步；
- 分支优化。

### Softmax

学习：

- 行级映射；
- Max/Sum Reduction；
- 寄存器和共享内存；
- 数值稳定性。

### GEMM

学习：

- Block Tile；
- Warp Tile；
- Thread Tile；
- 全局内存到共享内存；
- 共享内存到寄存器；
- 双缓冲；
- Tensor Core。

### FlashAttention

学习：

- Q/K/V 分块；
- Shared Memory；
- 寄存器累积；
- 在线 Softmax；
- IO-Awareness；
- 资源与 Occupancy 平衡。

本章最终要形成的分析框架是：

```text
数据在哪里
  ↓
哪些线程访问
  ↓
访问是否连续
  ↓
是否能够复用
  ↓
消耗多少寄存器和共享内存
  ↓
能驻留多少 Warp
  ↓
能否隐藏延迟
  ↓
是否有线程束分化
  ↓
通过 Benchmark 和性能工具验证
```


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-gemm算子|模块-gemm算子]]

%% 项目关联导航：结束 %%
