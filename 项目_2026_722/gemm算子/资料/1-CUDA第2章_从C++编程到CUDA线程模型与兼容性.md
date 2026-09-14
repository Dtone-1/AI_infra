# 一、本章在 CUDA 学习体系中的位置与学习目标

## 1. 目录识别与本章主题

根据课程截图，本章包含以下内容：

- 2.1 从 C++ 编程到 CUDA 编程
- 2.2 核函数
- 2.3 线程模型
- 2.4 线程全局索引计算方式
- 2.5 `nvcc` 编译流程与 GPU 计算能力
- 2.6 CUDA 程序兼容性问题

这一章处在 CUDA 学习路线的**入口位置**。它解决的不是“怎样把算子优化到很快”，而是更基础的问题：

1. CUDA 程序为什么同时包含 CPU 代码和 GPU 代码；
2. GPU 上运行的函数如何定义、启动和执行；
3. 一个核函数为什么会同时产生大量线程；
4. 每个线程怎样知道自己应该处理哪一份数据；
5. CUDA 源代码怎样经过 `nvcc` 编译并最终在特定 GPU 上运行；
6. 为什么同一份 CUDA 程序在不同 GPU、CUDA Toolkit 和驱动环境中可能表现不同。

可以把整个 CUDA 算子学习体系简化为下面几层：

```text
C++ 基础
  ↓
CUDA 程序基本结构
  ↓
核函数、线程层次、索引计算
  ↓
GPU 内存模型与数据传输
  ↓
线程同步、共享内存、访存合并
  ↓
算子实现
  ↓
性能分析与优化
  ↓
GEMM、Softmax、Reduction、Attention 等复杂算子
```

本章位于第二层和第三层，是后续所有 CUDA 算子的共同基础。

无论以后编写的是向量加法、矩阵乘法、归约、Softmax，还是 FlashAttention，最终都要回答两个最基本的问题：

- 启动多少个线程？
- 每个线程负责计算什么？

而这两个问题正是本章“线程模型”和“全局索引计算”的核心。

---

## 2. 从普通 C++ 到 CUDA 程序的思维变化

普通 C++ 程序通常只在 CPU 上执行：

```cpp
for (int i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
}
```

这段代码的执行逻辑是：一个 CPU 线程按照 `i = 0, 1, 2, ...` 的顺序处理所有元素。

CUDA 的基本想法是：如果不同元素之间互不依赖，就让大量 GPU 线程并行处理。

```text
CPU 串行方式：
线程 0：依次计算 c[0]、c[1]、c[2]、...、c[n-1]

GPU 并行方式：
线程 0：计算 c[0]
线程 1：计算 c[1]
线程 2：计算 c[2]
...
线程 n-1：计算 c[n-1]
```

因此，从 C++ 转向 CUDA，最重要的变化不是语法，而是**并行任务划分方式**。

在普通 C++ 中，经常思考：

> 循环应该怎么写？

在 CUDA 中，则要进一步思考：

> 循环中的不同迭代能不能拆给不同线程？  
> 一个线程负责一个元素、一行数据、一个小块，还是多个元素？

这就是算子开发中的“并行映射”。

---

## 3. CUDA 程序的基本结构

一个最小 CUDA 程序通常包含两个执行端：

- **Host：主机端，通常指 CPU**
- **Device：设备端，通常指 GPU**

程序的大致执行流程如下：

```text
CPU 创建和初始化数据
        ↓
CPU 在 GPU 上申请显存
        ↓
CPU 将输入数据复制到 GPU
        ↓
CPU 启动 GPU 核函数
        ↓
GPU 中大量线程并行计算
        ↓
CPU 等待 GPU 计算完成
        ↓
CPU 将结果复制回内存
        ↓
释放 GPU 资源
```

需要注意，CUDA 程序并不是“全部在 GPU 上执行”。更准确地说，它是一个由 CPU 负责组织、由 GPU 负责大规模并行计算的异构程序。

CPU 负责：

- 初始化数据；
- 申请和释放显存；
- 发起数据传输；
- 配置线程数量；
- 启动核函数；
- 检查错误；
- 继续执行后续控制逻辑。

GPU 负责：

- 执行被启动的核函数；
- 让大量线程并行处理数据；
- 完成适合吞吐型并行的计算任务。

---

## 4. 学完本章后应该具备的能力

完成本章学习后，至少应该能够独立完成以下任务：

### 4.1 阅读基本 CUDA 程序

能够区分：

- 哪些代码在 CPU 上执行；
- 哪些代码在 GPU 上执行；
- 哪一行启动了核函数；
- 网格和线程块分别配置了多少线程；
- 每个线程处理哪个数据元素。

### 4.2 编写简单核函数

能够编写以下类型的基础算子：

- 向量加法；
- 数组逐元素乘法；
- ReLU；
- Sigmoid 的逐元素版本；
- SAXPY：`y = a * x + y`；
- 简单的一维数据变换。

### 4.3 正确计算线程索引

能够理解并使用：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

并能解释：

- `threadIdx.x` 是线程在线程块内的编号；
- `blockIdx.x` 是线程块在网格中的编号；
- `blockDim.x` 是每个线程块中的线程数；
- `idx` 是该线程在整个一维网格中的全局编号。

### 4.4 正确配置核函数启动参数

能够根据数据规模计算：

```cpp
int threads_per_block = 256;
int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
```

理解为什么不能简单写成：

```cpp
int blocks_per_grid = n / threads_per_block;
```

因为当 `n` 不能被线程块大小整除时，普通整数除法会遗漏最后一部分元素。

### 4.5 使用 `nvcc` 编译 CUDA 程序

能够执行：

```bash
nvcc vector_add.cu -o vector_add
./vector_add
```

并初步理解 `nvcc` 不是单纯的“GPU 编译器”，而是负责拆分并协调主机代码和设备代码的编译驱动程序。

### 4.6 初步判断程序兼容性

能够认识以下术语：

- Compute Capability；
- `sm_XX`；
- `compute_XX`；
- PTX；
- SASS；
- CUDA Toolkit；
- NVIDIA Driver；
- Fat Binary。

不要求本章结束时精通底层指令，但需要知道：CUDA 程序是否能在某块 GPU 上运行，不只由源代码决定，还受编译目标、GPU 架构和驱动版本影响。

---

## 5. 本章与后续算子开发的关系

本章知识会直接出现在后续算子中。

以 GEMM 为例：

```text
C = A × B
```

最初可以让一个线程计算矩阵 `C` 的一个元素：

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

这仍然是在使用本章的二维线程模型和全局索引。

后续的共享内存分块、向量化访存、双缓冲、寄存器分块等优化，只是在此基础上改变：

- 一个线程负责多少个输出元素；
- 一个线程块负责多大的矩阵分块；
- 数据如何在全局内存、共享内存和寄存器之间移动；
- 线程如何协作。

因此，本章不是一组可以快速跳过的入门语法，而是算子实现的“坐标系统”和“任务分配模型”。

# 二、课程内容取舍与算子开发补充知识

## 1. 课程目录完整性分析

截图中的目录覆盖了 CUDA 入门最核心的框架：

- 从 C++ 过渡到 CUDA；
- 核函数；
- 线程模型；
- 全局索引；
- 编译流程；
- 计算能力；
- 兼容性。

但如果目标是后续进行 CUDA 算子编写，仅靠这些标题仍然不够。至少还需要补充以下内容，才能真正写出一个可以编译、运行、验证和调试的 CUDA 程序：

1. CUDA 函数限定符；
2. 核函数启动语法；
3. `dim3` 和内置线程索引变量；
4. Host 内存与 Device 内存的区别；
5. `cudaMalloc`、`cudaMemcpy`、`cudaFree`；
6. GPU 异步执行与同步；
7. CUDA 错误检查；
8. 越界保护；
9. Grid-Stride Loop；
10. Warp 和 SIMT 的基础概念；
11. PTX、SASS、`sm_XX` 和 `compute_XX` 的关系；
12. GPU 架构、Toolkit 和驱动之间的兼容关系。

这些知识不一定都需要在本章达到同样熟练度。可以按下面三个层次学习。

### 必须熟练掌握

- Host 与 Device 的分工；
- `__global__` 核函数；
- `<<<grid, block>>>` 启动配置；
- Grid、Block、Thread 的层次关系；
- 一维和二维全局索引；
- 向上取整计算线程块数量；
- 越界判断；
- `cudaMalloc`、`cudaMemcpy`、`cudaFree`；
- `cudaDeviceSynchronize`；
- CUDA 错误检查；
- `nvcc` 的基本使用。

### 当前需要理解，但不必立即精通

- Warp；
- SIMT；
- Streaming Multiprocessor，简称 SM；
- PTX 与 SASS；
- Compute Capability；
- `-arch=sm_XX` 和 `-gencode`；
- Fat Binary；
- 驱动向后兼容与 Toolkit 版本关系；
- Grid-Stride Loop。

### 当前只需要建立概念，后续再深入

- Warp 调度细节；
- Occupancy；
- 指令吞吐；
- 寄存器分配；
- PTX 手写；
- SASS 指令分析；
- 多架构二进制裁剪；
- JIT 编译细节。

---

## 2. 从 C++ 编程到 CUDA 编程

### 2.1 源文件扩展名

普通 C++ 文件通常为：

```text
main.cpp
```

CUDA 源文件通常为：

```text
main.cu
```

`.cu` 文件中可以同时包含：

- 普通 C++ 主机代码；
- CUDA 设备代码；
- CUDA Runtime API 调用；
- 核函数启动语法。

### 2.2 Host 与 Device

在 CUDA 语境中：

```text
Host   → CPU 及其内存
Device → GPU 及其显存
```

Host 内存和 Device 内存通常属于不同的地址空间。CPU 不能像访问普通数组一样直接解引用一个传统的 Device 指针；GPU 核函数也不能默认直接访问普通的 Host 堆内存。

因此，基础 CUDA 程序通常需要显式管理两侧内存。

```cpp
float* h_x = new float[n];   // Host 内存
float* d_x = nullptr;        // Device 指针
cudaMalloc(&d_x, n * sizeof(float));
```

变量名前缀 `h_` 和 `d_` 不是 CUDA 强制语法，而是常见命名习惯：

- `h_x`：host 上的数据；
- `d_x`：device 上的数据。

这种命名在复杂算子代码中非常重要，可以降低把 CPU 指针和 GPU 指针混用的风险。

### 2.3 CUDA 程序的异构执行

考虑下面的伪代码：

```cpp
int main() {
    // 1. CPU 初始化输入
    // 2. CPU 申请 GPU 显存
    // 3. CPU 将输入复制到 GPU
    // 4. CPU 启动 GPU 核函数
    // 5. GPU 并行计算
    // 6. CPU 将结果复制回来
}
```

其中，CPU 是任务发起者，GPU 是并行计算执行者。

核函数启动操作本身由 CPU 发出：

```cpp
vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
```

这行代码的含义不是“CPU 调用普通函数并进入函数体”，而是：

> CPU 向 GPU 提交一个核函数任务，并指定该任务需要创建多少个线程块、每个线程块包含多少个线程。

---

## 3. 核函数

## 3.1 什么是核函数

核函数是由 CPU 发起、在 GPU 上由大量线程并行执行的函数。

最典型的定义方式为：

```cpp
__global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

这里的 `__global__` 表示：

- 该函数从 Host 端启动；
- 该函数在 Device 端执行；
- 调用时需要使用 CUDA 的核函数启动语法。

### 3.2 函数限定符

CUDA 常见函数限定符如下。

#### `__global__`

```cpp
__global__ void kernel() {}
```

含义：

- 在 GPU 上执行；
- 通常由 CPU 发起；
- 使用 `<<<...>>>` 启动；
- 返回类型必须是 `void`。

#### `__device__`

```cpp
__device__ float square(float x) {
    return x * x;
}
```

含义：

- 在 GPU 上执行；
- 通常由其他 GPU 函数调用；
- 不能在普通 Host 代码中像普通函数一样直接调用。

#### `__host__`

```cpp
__host__ float cpu_square(float x) {
    return x * x;
}
```

含义：

- 在 CPU 上执行；
- 普通 C++ 函数默认就是 Host 函数，因此通常不必显式书写。

#### 同时使用 `__host__ __device__`

```cpp
__host__ __device__
float square(float x) {
    return x * x;
}
```

表示编译器会分别生成 Host 版本和 Device 版本。

这种写法常用于模板库或需要两端共用的简单数学函数，但函数内部只能使用两端都支持的操作。

### 3.3 核函数启动语法

核函数使用如下形式启动：

```cpp
kernel<<<grid_dim, block_dim>>>(arguments);
```

例如：

```cpp
vector_add<<<100, 256>>>(d_a, d_b, d_c, n);
```

表示：

- 网格中有 100 个线程块；
- 每个线程块中有 256 个线程；
- 理论上共启动 `100 × 256 = 25600` 个线程。

注意，“启动了多少线程”与“有多少数据”不一定完全相等。实际开发中常常会多启动少量线程，再使用边界判断阻止多余线程访问越界位置。

### 3.4 核函数为什么返回 `void`

核函数会由成千上万个线程并行执行，不存在一个适合作为普通函数返回值的单一结果。

核函数通常通过写入 Device 内存输出结果：

```cpp
c[idx] = a[idx] + b[idx];
```

如果需要得到一个标量，例如数组总和，通常也会先将结果写入显存，再复制回 Host，或者通过后续核函数继续处理。

### 3.5 核函数启动是异步的

通常情况下：

```cpp
kernel<<<grid, block>>>(...);
```

只表示 CPU 向 GPU 提交了任务。CPU 不一定会等待 GPU 立即完成，可能继续执行后续 Host 代码。

为了显式等待，可以使用：

```cpp
cudaDeviceSynchronize();
```

这对于以下场景十分重要：

- 测试核函数是否执行成功；
- 准确测量核函数时间；
- 确保 GPU 结果已经生成；
- 在调试阶段定位异步错误。

不过某些操作，例如将结果从 Device 复制回 Host 的同步版 `cudaMemcpy`，通常也会形成必要的等待。

---

## 4. CUDA 线程模型

## 4.1 三层结构：Grid、Block、Thread

CUDA 使用三级逻辑层次组织线程：

```text
Grid（网格）
  ├── Block 0（线程块）
  │     ├── Thread 0
  │     ├── Thread 1
  │     └── ...
  ├── Block 1
  │     ├── Thread 0
  │     ├── Thread 1
  │     └── ...
  └── ...
```

对应关系：

- 一次核函数启动产生一个 Grid；
- 一个 Grid 包含多个 Block；
- 一个 Block 包含多个 Thread。

### 4.2 为什么需要线程块

如果只给所有线程一个连续编号，从逻辑上也能完成简单运算，但线程块提供了后续优化必须依赖的组织单位。

同一线程块中的线程可以：

- 使用共享内存交换数据；
- 使用 `__syncthreads()` 进行块内同步；
- 协作加载一个数据分块；
- 共同完成一个矩阵 tile、归约分段或卷积区域。

不同线程块通常应尽量独立，因为它们：

- 可能以任意顺序执行；
- 可能同时运行，也可能分批运行；
- 可能被调度到不同 SM 上；
- 不能直接使用普通块内同步机制互相等待。

因此，CUDA 算子设计通常需要保证线程块之间没有必须依赖固定执行顺序的关系。

### 4.3 线程组织可以是一维、二维或三维

CUDA 提供 `dim3` 类型描述维度：

```cpp
dim3 block(16, 16);
dim3 grid(
    (width + block.x - 1) / block.x,
    (height + block.y - 1) / block.y
);
```

虽然 `dim3` 名字中包含 3，但可以只设置一维或二维。没有显式设置的维度默认值为 1。

```cpp
dim3 block1(256);          // (256, 1, 1)
dim3 block2(16, 16);       // (16, 16, 1)
dim3 block3(8, 8, 4);      // (8, 8, 4)
```

不同维度只是逻辑组织方式，不代表 GPU 中真的存在几何形状的线程。它们的作用是让数据索引更自然。

适用情况：

- 一维：向量、序列、扁平数组；
- 二维：图像、矩阵；
- 三维：体数据、三维网格、部分批量问题。

### 4.4 内置变量

在核函数中，CUDA 提供以下内置变量：

```cpp
threadIdx
blockIdx
blockDim
gridDim
```

它们都有 `.x`、`.y`、`.z` 三个分量。

#### `threadIdx`

当前线程在线程块内部的坐标。

```cpp
threadIdx.x
threadIdx.y
threadIdx.z
```

#### `blockIdx`

当前线程块在网格中的坐标。

```cpp
blockIdx.x
blockIdx.y
blockIdx.z
```

#### `blockDim`

线程块各维度的大小。

```cpp
blockDim.x
blockDim.y
blockDim.z
```

#### `gridDim`

网格各维度的线程块数量。

```cpp
gridDim.x
gridDim.y
gridDim.z
```

### 4.5 逻辑线程与物理执行

CUDA 代码中看到的线程、线程块和网格属于逻辑执行模型。

物理 GPU 中还存在：

- SM；
- Warp Scheduler；
- CUDA Core；
- Tensor Core；
- 寄存器文件；
- 共享内存。

一个线程块会被调度到某个 SM 上执行，同一线程块不会跨多个 SM 拆开。一个 SM 可以同时驻留多个线程块，但具体能驻留多少取决于：

- 每个线程块的线程数量；
- 每个线程使用的寄存器数量；
- 每个线程块使用的共享内存；
- GPU 架构限制。

这些资源约束会在性能优化阶段重点学习。本章先建立“逻辑线程块最终由 SM 执行”的概念即可。

---

## 5. Warp 与 SIMT：为算子优化提前建立概念

CUDA 的线程并不是完全独立地逐个调度。硬件通常以 Warp 为基本执行批次。

在现代 NVIDIA CUDA 编程模型中，一个 Warp 通常包含 32 个线程。

例如一个包含 256 个线程的线程块，可以看成：

```text
Warp 0：线程 0～31
Warp 1：线程 32～63
Warp 2：线程 64～95
...
Warp 7：线程 224～255
```

SIMT 是 Single Instruction, Multiple Threads，即单指令、多线程。

可以粗略理解为：

> 同一个 Warp 中的线程通常执行同一条指令，但每个线程处理自己的数据。

例如：

```cpp
c[idx] = a[idx] + b[idx];
```

同一个 Warp 中的 32 个线程执行相同的加法指令，但访问不同的 `idx`。

这也是 GPU 适合规则数据并行，而不擅长大量复杂分支的原因之一。

如果同一 Warp 中一部分线程执行 `if`，另一部分执行 `else`，可能出现分支发散。分支发散会在后续性能优化阶段详细学习。本章只需知道：

- Warp 是硬件执行的重要粒度；
- 线程块大小经常选择 32 的整数倍；
- 常见线程块大小包括 128、256、512；
- 256 是很多基础算子的常用起点，但不是永远最优。

---

## 6. 线程全局索引计算

## 6.1 一维索引

最常见的一维全局索引公式为：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

假设：

```text
blockDim.x = 4
```

线程块编号和线程编号如下：

```text
Block 0:
  threadIdx.x = 0 → idx = 0 × 4 + 0 = 0
  threadIdx.x = 1 → idx = 0 × 4 + 1 = 1
  threadIdx.x = 2 → idx = 0 × 4 + 2 = 2
  threadIdx.x = 3 → idx = 0 × 4 + 3 = 3

Block 1:
  threadIdx.x = 0 → idx = 1 × 4 + 0 = 4
  threadIdx.x = 1 → idx = 1 × 4 + 1 = 5
  threadIdx.x = 2 → idx = 1 × 4 + 2 = 6
  threadIdx.x = 3 → idx = 1 × 4 + 3 = 7
```

因此：

```text
blockIdx.x * blockDim.x
```

得到当前线程块负责区间的起始位置，再加上：

```text
threadIdx.x
```

得到线程在该区间内的偏移。

### 6.2 为什么必须做边界判断

假设数据长度：

```text
n = 10
```

每个线程块有 4 个线程：

```text
threads = 4
```

需要的线程块数：

```text
blocks = ceil(10 / 4) = 3
```

实际启动：

```text
3 × 4 = 12 个线程
```

但只有索引 `0～9` 有效，索引 10 和 11 会越界。

因此核函数中必须写：

```cpp
if (idx < n) {
    c[idx] = a[idx] + b[idx];
}
```

边界检查是 CUDA 程序正确性的基础。越界访问可能导致：

- 结果错误；
- 非法内存访问；
- 核函数异步报错；
- 后续 CUDA API 才显示错误；
- 程序直接终止。

### 6.3 向上取整计算线程块数

通用整数向上取整公式：

```cpp
int blocks = (n + threads - 1) / threads;
```

例如：

```text
n = 1000
threads = 256
blocks = (1000 + 256 - 1) / 256
       = 1255 / 256
       = 4
```

整数除法得到 4，最终启动 1024 个线程，足以覆盖 1000 个元素。

可以把它封装成：

```cpp
inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
```

对于很大的数据长度，索引和数据规模更适合使用 `size_t` 或 64 位整数，避免 32 位整数溢出。

### 6.4 二维索引

处理矩阵或图像时，常使用二维网格和二维线程块。

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

对于行优先存储的二维数组，线性下标为：

```cpp
int idx = row * width + col;
```

完整边界判断：

```cpp
if (row < height && col < width) {
    output[row * width + col] = input[row * width + col];
}
```

例如线程块为：

```cpp
dim3 block(16, 16);
```

一个线程块共有：

```text
16 × 16 = 256 个线程
```

每个线程块覆盖一个 `16 × 16` 的二维数据区域。

### 6.5 三维索引

三维情况可以写为：

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
```

若数据布局为 `[depth][height][width]`，线性下标可以写成：

```cpp
int idx = z * height * width + y * width + x;
```

### 6.6 将多维线程编号展平

有时线程块是二维的，但需要得到线程在线程块内的线性编号：

```cpp
int local_tid =
    threadIdx.y * blockDim.x
    + threadIdx.x;
```

三维线程块的线性编号：

```cpp
int local_tid =
    threadIdx.z * blockDim.y * blockDim.x
    + threadIdx.y * blockDim.x
    + threadIdx.x;
```

这类展平在以下场景中很常见：

- 共享内存索引；
- 归约；
- Warp 编号计算；
- 矩阵分块；
- 将二维线程映射到一维数据。

### 6.7 Grid-Stride Loop

基础写法通常是一个线程处理一个元素：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;

if (idx < n) {
    output[idx] = input[idx];
}
```

更通用的写法是 Grid-Stride Loop：

```cpp
for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
     idx < n;
     idx += blockDim.x * gridDim.x) {
    output[idx] = input[idx];
}
```

这里：

```cpp
blockDim.x * gridDim.x
```

是整个网格中的总线程数。

每个线程处理：

```text
idx
idx + 总线程数
idx + 2 × 总线程数
...
```

优点：

- 不要求线程总数覆盖全部数据；
- 能处理任意大的数组；
- 可以限制网格规模；
- 便于线程复用；
- 是许多高性能 CUDA 核函数中的常用模式。

初学时应先熟练掌握“一线程一元素”，再理解 Grid-Stride Loop。

---

## 7. CUDA 内存管理：课程目录未列出但必须补充

如果不学习基本内存管理，就无法写出完整可运行的核函数程序。

### 7.1 申请 Device 内存

```cpp
float* d_a = nullptr;
cudaMalloc(&d_a, n * sizeof(float));
```

含义是在 GPU 显存中申请 `n` 个 `float` 的空间。

更完整的写法通常需要检查返回值：

```cpp
cudaError_t err = cudaMalloc(&d_a, n * sizeof(float));

if (err != cudaSuccess) {
    std::cerr << "cudaMalloc failed: "
              << cudaGetErrorString(err)
              << std::endl;
    return 1;
}
```

### 7.2 Host 到 Device 的数据复制

```cpp
cudaMemcpy(
    d_a,
    h_a,
    n * sizeof(float),
    cudaMemcpyHostToDevice
);
```

参数含义：

1. 目标地址；
2. 源地址；
3. 字节数；
4. 复制方向。

常见方向：

```cpp
cudaMemcpyHostToDevice
cudaMemcpyDeviceToHost
cudaMemcpyDeviceToDevice
```

### 7.3 Device 到 Host 的数据复制

```cpp
cudaMemcpy(
    h_c,
    d_c,
    n * sizeof(float),
    cudaMemcpyDeviceToHost
);
```

### 7.4 释放显存

```cpp
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
```

与 C++ 中的 `new/delete` 类似，显存申请后也必须释放。

---

## 8. CUDA 错误检查：必须形成习惯

CUDA 很多错误具有异步性。核函数中的越界访问可能不会在启动语句当场直接显示，而是在后面的同步或内存复制操作中暴露。

推荐至少检查两个阶段。

### 8.1 检查核函数启动错误

```cpp
kernel<<<grid, block>>>(...);

cudaError_t launch_error = cudaGetLastError();

if (launch_error != cudaSuccess) {
    std::cerr << "Kernel launch failed: "
              << cudaGetErrorString(launch_error)
              << std::endl;
}
```

这可以捕获：

- 启动参数非法；
- 线程数超过硬件上限；
- 核函数配置错误；
- 某些即时启动问题。

### 8.2 检查核函数执行错误

```cpp
cudaError_t sync_error = cudaDeviceSynchronize();

if (sync_error != cudaSuccess) {
    std::cerr << "Kernel execution failed: "
              << cudaGetErrorString(sync_error)
              << std::endl;
}
```

这可以捕获执行期间的错误，例如非法显存访问。

### 8.3 推荐错误检查宏

```cpp
#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t error = (call);                               \
    if (error != cudaSuccess) {                               \
        std::cerr << "CUDA error at "                         \
                  << __FILE__ << ":" << __LINE__ << ": "      \
                  << cudaGetErrorString(error)                 \
                  << std::endl;                               \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)
```

使用：

```cpp
CUDA_CHECK(cudaMalloc(&d_a, bytes));
CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

kernel<<<grid, block>>>(...);

CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

在算子开发中，不做错误检查会显著增加调试难度。

---

## 9. `nvcc` 编译流程

## 9.1 `nvcc` 的角色

`nvcc` 是 CUDA 编译驱动程序。

CUDA 源文件中同时包含：

- Host C++ 代码；
- Device CUDA 代码；
- CUDA 特有语法，例如 `<<<...>>>`；
- 函数限定符，例如 `__global__`。

普通 `g++` 无法直接完整处理这些内容，因此需要 `nvcc` 协调编译。

可以将流程简化理解为：

```text
.cu 源文件
   ↓
nvcc 识别并拆分 Host 与 Device 代码
   ↓
Host 代码交给主机 C++ 编译器
   ↓
Device 代码编译为 PTX 和/或目标 GPU 机器码
   ↓
设备代码被嵌入最终目标文件
   ↓
主机对象与 CUDA Runtime 等库链接
   ↓
生成可执行程序
```

在 Linux 上，Host 部分通常会交给 `g++` 等主机编译器处理。

### 9.2 基本编译命令

```bash
nvcc vector_add.cu -o vector_add
```

运行：

```bash
./vector_add
```

### 9.3 常用编译选项

#### 指定 C++ 标准

```bash
nvcc -std=c++17 vector_add.cu -o vector_add
```

#### 开启优化

```bash
nvcc -O3 vector_add.cu -o vector_add
```

#### 保留行号信息，便于性能分析

```bash
nvcc -O3 -lineinfo vector_add.cu -o vector_add
```

`-lineinfo` 通常比完整调试模式更适合性能分析，因为它可以帮助 Nsight 工具将 GPU 指令映射回源代码行，同时不会像 `-G` 那样严重改变优化结果。

#### 设备调试

```bash
nvcc -G -g vector_add.cu -o vector_add_debug
```

注意：`-G` 会关闭或显著影响设备端优化，不适合用于最终性能测试。

#### 指定 GPU 架构

```bash
nvcc -arch=sm_XX vector_add.cu -o vector_add
```

其中 `XX` 需要替换为目标 GPU 对应的 Compute Capability 编号。

不要在不了解目标 GPU 的情况下机械照抄某个 `sm_XX`。可以通过设备查询程序或 NVIDIA 官方资料确认。

---

## 10. GPU 计算能力

## 10.1 什么是 Compute Capability

Compute Capability，中文常译为计算能力，是 NVIDIA 用来描述 GPU 架构能力和指令特征的一组版本编号。

它常写作：

```text
major.minor
```

并映射为编译参数中的：

```text
sm_XY
compute_XY
```

例如概念上：

```text
Compute Capability X.Y
  ↔ sm_XY
  ↔ compute_XY
```

它决定或反映：

- 支持哪些 GPU 指令；
- 支持哪些数据类型；
- 每个线程块的资源上限；
- 共享内存和寄存器相关能力；
- 是否支持某些 Tensor Core 特性；
- 原子操作和异步拷贝等功能；
- 编译器可以生成哪些目标机器码。

计算能力不是“GPU 算力大小”的简单评分。更准确地说，它是 GPU 编程架构和功能代际标识。

两块 GPU 即使计算能力相同，性能也可能差别很大，因为它们还可能具有不同的：

- SM 数量；
- 显存带宽；
- 时钟频率；
- 缓存容量；
- Tensor Core 数量；
- 功耗上限。

## 10.2 查询设备属性

可以使用 CUDA Runtime API：

```cpp
int device = 0;
cudaDeviceProp prop;

CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

std::cout << "GPU: " << prop.name << '\n';
std::cout << "Compute Capability: "
          << prop.major << "." << prop.minor << '\n';
std::cout << "SM count: "
          << prop.multiProcessorCount << '\n';
std::cout << "Max threads per block: "
          << prop.maxThreadsPerBlock << '\n';
```

也可以通过系统工具查看 GPU 和驱动信息：

```bash
nvidia-smi
```

但要注意，`nvidia-smi` 展示的“CUDA Version”通常代表当前驱动能够支持的最高 CUDA 运行时能力范围，不等同于你实际安装的 CUDA Toolkit 版本。

实际 Toolkit 版本可通过：

```bash
nvcc --version
```

查看。

---

## 11. PTX、SASS、`compute_XX` 与 `sm_XX`

这部分是理解 CUDA 兼容性的关键。

### 11.1 PTX

PTX 可以粗略理解为 NVIDIA GPU 的虚拟指令集或中间表示。

它不是最终直接在 GPU 执行的具体硬件机器码。驱动可以在运行时将合适版本的 PTX JIT 编译为当前 GPU 可执行的机器码。

### 11.2 SASS

SASS 是面向特定 NVIDIA GPU 架构的机器指令。

可以粗略类比：

```text
CUDA C++ 源代码
    ↓
PTX：相对虚拟、面向计算平台
    ↓
SASS：面向具体 GPU 架构的机器码
```

实际编译流程可能根据选项直接生成对应架构的机器码，也可能同时保留 PTX。

### 11.3 `compute_XX`

`compute_XX` 通常表示虚拟架构目标，常与 PTX 生成相关。

例如：

```text
compute_XY
```

表示以某个计算能力级别的虚拟 ISA 能力为目标。

### 11.4 `sm_XX`

`sm_XX` 表示具体 GPU 架构的机器码目标。

例如：

```text
sm_XY
```

通常意味着为该架构生成可直接执行的设备机器码。

### 11.5 Fat Binary

为了让一个程序支持多种 GPU 架构，可以在二进制中同时嵌入多个架构版本的代码。

例如概念上包含：

```text
面向架构 A 的机器码
面向架构 B 的机器码
面向较新架构 JIT 使用的 PTX
```

这种包含多个设备代码版本的可执行文件常被称为 Fat Binary。

代价是：

- 编译时间增加；
- 二进制体积变大。

收益是：

- 能在更多 GPU 上直接运行；
- 可以减少某些 JIT 开销；
- 可以保留一定的未来兼容性。

---

## 12. CUDA 程序兼容性问题

CUDA 兼容性至少涉及四个层次：

```text
源代码
CUDA Toolkit / nvcc
编译目标架构
NVIDIA Driver
实际 GPU
```

## 12.1 GPU 架构兼容性

如果程序只编译了某个特定 `sm_XX` 的机器码，而目标 GPU 不支持该代码，就可能出现：

```text
no kernel image is available for execution on the device
```

因此，发布给多种 GPU 使用的程序通常需要生成多个架构目标，或者保留合适的 PTX。

## 12.2 驱动与 CUDA Runtime

一般原则是：

- 较新的 NVIDIA 驱动通常可以运行由较旧 CUDA Toolkit 构建的程序；
- 较旧驱动通常不能运行依赖更高版本 CUDA Runtime 或新特性的程序；
- 实际兼容关系需要根据 CUDA 官方兼容性说明确认。

对于初学者，遇到环境问题时应分别检查：

```bash
nvidia-smi
nvcc --version
```

并明确区分：

- GPU 驱动版本；
- 驱动显示的最高 CUDA 支持能力；
- 本机安装的 CUDA Toolkit 版本；
- PyTorch 自带或依赖的 CUDA Runtime 版本；
- 编译扩展时实际使用的 `nvcc` 版本。

## 12.3 源码兼容性

即使代码能够编译，也可能由于使用了新架构特性而无法在旧 GPU 上运行。

例如某些代码依赖：

- 新的数据类型；
- 新的 Tensor Core 指令；
- 异步内存拷贝；
- 新的原子操作；
- 特定架构的内联 PTX；
- 特定 CUDA 版本新增的 API。

因此，算子源码中的“最低 GPU 架构要求”需要明确。

## 12.4 性能兼容性不等于运行兼容性

某个算子能在多种 GPU 上运行，不代表它在所有 GPU 上性能都好。

一个为某架构设计的最优参数，例如：

- 线程块大小；
- tile 大小；
- 每线程寄存器分块；
- 共享内存用量；
- Tensor Core 指令形态；

换到另一架构后可能不再最优。

所以算子开发中要区分：

- **功能兼容性：能否正确运行**
- **性能可移植性：换 GPU 后是否仍然高效**

---

## 13. 本章容易混淆的概念

### 13.1 一个核函数不等于一个线程

核函数是一份程序定义。启动核函数时，GPU 会创建大量线程，每个线程都执行同一份核函数代码，但内置索引值不同。

```text
同一份 kernel 代码
  ↓
线程 0 使用 idx = 0
线程 1 使用 idx = 1
线程 2 使用 idx = 2
...
```

### 13.2 一个线程块不等于一个 SM

线程块是软件层面的逻辑组织单位，SM 是硬件执行资源。

一个线程块会被分配到一个 SM 上，但：

- 一个 SM 可以同时驻留多个线程块；
- 一个线程块不会跨多个 SM；
- 线程块数量通常远大于 SM 数量；
- GPU 会分批调度线程块。

### 13.3 线程总数不一定等于数据量

可以多启动线程，再通过：

```cpp
if (idx < n)
```

保护边界。

也可以使用 Grid-Stride Loop，让线程重复处理多个元素。

### 13.4 `nvidia-smi` 中的 CUDA Version 不等于 `nvcc` 版本

`nvidia-smi` 主要反映驱动能力，`nvcc --version` 反映安装的 CUDA Toolkit 编译器版本。

### 13.5 核函数启动成功不代表执行成功

下面这行没有立即报错：

```cpp
kernel<<<grid, block>>>(...);
```

并不一定说明核函数内部没有越界。

应继续检查：

```cpp
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

---

## 14. 面向算子开发的学习标准

完成本章后，应能在不查资料或只少量查语法的情况下，独立解释下面代码：

```cpp
int threads = 256;
int blocks = (n + threads - 1) / threads;

relu_kernel<<<blocks, threads>>>(x, y, n);
```

并回答：

1. 为什么每个线程块设置 256 个线程？
2. 为什么线程块数量需要向上取整？
3. 一共启动了多少线程？
4. 多余线程如何处理？
5. `relu_kernel` 在 CPU 还是 GPU 上执行？
6. 启动语句由 CPU 还是 GPU 执行？
7. 每个线程如何计算自己的 `idx`？
8. `x` 和 `y` 指向 Host 内存还是 Device 内存？
9. 如何检查核函数是否发生非法访问？
10. 如何编译这段 CUDA 代码？
11. 换一块 GPU 后可能遇到哪些兼容性问题？

如果这些问题仍然无法顺畅回答，就不应该急于进入共享内存或 GEMM 优化。

# 三、章节综合实例

## 1. 实例目标

实现下面的逐元素算子：

```text
y[i] = max(alpha × x[i] + bias, 0)
```

这个算子把三个操作融合在一个核函数中：

1. 标量乘法：`alpha × x[i]`
2. 加偏置：`+ bias`
3. ReLU：`max(..., 0)`

它类似神经网络中常见的逐元素融合操作，可以综合练习：

- 从 C++ 循环到 CUDA 并行；
- 核函数定义；
- Host 与 Device；
- 一维线程模型；
- 全局索引；
- 线程块数量向上取整；
- 越界保护；
- 显存申请与数据传输；
- 核函数启动；
- 异步错误检查；
- CPU 结果校验；
- `nvcc` 编译；
- GPU 计算能力查询。

---

## 2. CPU 串行版本

普通 C++ 可以写成：

```cpp
for (int i = 0; i < n; ++i) {
    float value = alpha * x[i] + bias;
    y[i] = value > 0.0f ? value : 0.0f;
}
```

每次循环只处理一个元素，所有元素由同一个 CPU 执行流依次计算。

CUDA 版本将每个元素交给不同 GPU 线程：

```text
GPU 线程 0 → 计算 y[0]
GPU 线程 1 → 计算 y[1]
GPU 线程 2 → 计算 y[2]
...
```

由于不同元素之间没有依赖，这个任务天然适合数据并行。

---

## 3. 完整 CUDA 代码

将下面内容保存为：

```text
fused_affine_relu.cu
```

```cpp
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t error = (call);                               \
    if (error != cudaSuccess) {                               \
        std::cerr << "CUDA error at "                         \
                  << __FILE__ << ":" << __LINE__ << ": "      \
                  << cudaGetErrorString(error)                 \
                  << std::endl;                               \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)

__global__ void fused_affine_relu_kernel(
    const float* x,
    float* y,
    float alpha,
    float bias,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float value = alpha * x[idx] + bias;
        y[idx] = value > 0.0f ? value : 0.0f;
    }
}

void fused_affine_relu_cpu(
    const std::vector<float>& x,
    std::vector<float>& y,
    float alpha,
    float bias
) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        float value = alpha * x[i] + bias;
        y[i] = value > 0.0f ? value : 0.0f;
    }
}

int main() {
    const int n = 1000;
    const float alpha = 1.5f;
    const float bias = -0.25f;

    const std::size_t bytes =
        static_cast<std::size_t>(n) * sizeof(float);

    // 1. Host 端数据
    std::vector<float> h_x(n);
    std::vector<float> h_y_gpu(n, 0.0f);
    std::vector<float> h_y_cpu(n, 0.0f);

    for (int i = 0; i < n; ++i) {
        h_x[i] = static_cast<float>(i % 11 - 5);
    }

    // 2. Device 指针
    float* d_x = nullptr;
    float* d_y = nullptr;

    // 3. 在 GPU 显存中申请空间
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    // 4. 将输入从 Host 复制到 Device
    CUDA_CHECK(cudaMemcpy(
        d_x,
        h_x.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    // 5. 配置线程组织
    const int threads_per_block = 256;
    const int blocks_per_grid =
        (n + threads_per_block - 1) / threads_per_block;

    std::cout << "n = " << n << '\n';
    std::cout << "threads per block = "
              << threads_per_block << '\n';
    std::cout << "blocks per grid = "
              << blocks_per_grid << '\n';
    std::cout << "total launched threads = "
              << blocks_per_grid * threads_per_block
              << '\n';

    // 6. CPU 发起核函数，GPU 并行执行
    fused_affine_relu_kernel<<<
        blocks_per_grid,
        threads_per_block
    >>>(
        d_x,
        d_y,
        alpha,
        bias,
        n
    );

    // 7. 检查核函数启动错误
    CUDA_CHECK(cudaGetLastError());

    // 8. 等待 GPU 执行完成，并检查执行错误
    CUDA_CHECK(cudaDeviceSynchronize());

    // 9. 将结果从 Device 复制回 Host
    CUDA_CHECK(cudaMemcpy(
        h_y_gpu.data(),
        d_y,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    // 10. CPU 计算参考结果
    fused_affine_relu_cpu(
        h_x,
        h_y_cpu,
        alpha,
        bias
    );

    // 11. 校验 GPU 结果
    bool correct = true;
    const float tolerance = 1e-6f;

    for (int i = 0; i < n; ++i) {
        if (std::fabs(h_y_gpu[i] - h_y_cpu[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i
                      << ": GPU = " << h_y_gpu[i]
                      << ", CPU = " << h_y_cpu[i]
                      << '\n';
            correct = false;
            break;
        }
    }

    std::cout << (
        correct
            ? "Result check passed."
            : "Result check failed."
    ) << '\n';

    // 12. 打印前几个结果
    for (int i = 0; i < 12; ++i) {
        std::cout << "x[" << i << "] = "
                  << h_x[i]
                  << ", y[" << i << "] = "
                  << h_y_gpu[i]
                  << '\n';
    }

    // 13. 释放 GPU 显存
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return correct ? 0 : 1;
}
```

---

## 4. 编译与运行

### 4.1 基础编译

```bash
nvcc -std=c++17 fused_affine_relu.cu -o fused_affine_relu
```

运行：

```bash
./fused_affine_relu
```

### 4.2 开启优化并保留行号信息

```bash
nvcc \
    -std=c++17 \
    -O3 \
    -lineinfo \
    fused_affine_relu.cu \
    -o fused_affine_relu
```

### 4.3 指定目标架构

先确认 GPU 的 Compute Capability，再使用正确的架构编号：

```bash
nvcc \
    -std=c++17 \
    -O3 \
    -lineinfo \
    -arch=sm_XX \
    fused_affine_relu.cu \
    -o fused_affine_relu
```

不要直接照抄 `sm_XX`，应替换为目标 GPU 对应的真实架构参数。

---

## 5. 按执行时间线理解程序

### 阶段一：Host 创建输入

```cpp
std::vector<float> h_x(n);
```

数据此时位于 CPU 内存中。

### 阶段二：Host 申请 Device 内存

```cpp
cudaMalloc(&d_x, bytes);
cudaMalloc(&d_y, bytes);
```

`d_x` 和 `d_y` 指向 GPU 显存。

### 阶段三：输入传输

```cpp
cudaMemcpy(
    d_x,
    h_x.data(),
    bytes,
    cudaMemcpyHostToDevice
);
```

将 CPU 内存中的 `h_x` 复制到 GPU 显存中的 `d_x`。

### 阶段四：计算线程块数量

```cpp
const int threads_per_block = 256;
const int blocks_per_grid =
    (n + threads_per_block - 1)
    / threads_per_block;
```

本例：

```text
n = 1000
threads_per_block = 256
blocks_per_grid = 4
```

实际启动：

```text
4 × 256 = 1024 个线程
```

### 阶段五：CPU 启动核函数

```cpp
fused_affine_relu_kernel<<<4, 256>>>(...);
```

创建：

- 1 个一维网格；
- 网格中 4 个线程块；
- 每个线程块 256 个线程。

### 阶段六：每个线程计算自己的全局索引

线程执行：

```cpp
int idx =
    blockIdx.x * blockDim.x
    + threadIdx.x;
```

例如：

```text
blockIdx.x = 2
blockDim.x = 256
threadIdx.x = 10
```

则：

```text
idx = 2 × 256 + 10 = 522
```

该线程负责：

```cpp
y[522] = max(alpha * x[522] + bias, 0);
```

### 阶段七：多余线程退出

总共启动 1024 个线程，但数据只有 1000 个元素。

索引为：

```text
1000～1023
```

的 24 个线程不满足：

```cpp
if (idx < n)
```

因此不会访问数组。

### 阶段八：同步和错误检查

```cpp
cudaGetLastError();
cudaDeviceSynchronize();
```

前者检查核函数启动问题，后者等待执行完成并暴露执行期错误。

### 阶段九：结果复制回 Host

```cpp
cudaMemcpy(
    h_y_gpu.data(),
    d_y,
    bytes,
    cudaMemcpyDeviceToHost
);
```

### 阶段十：CPU 参考校验

GPU 程序不能只看“程序没有报错”，还需要与 CPU 结果比较。

这体现了算子开发中的基本验证方法：

```text
实现 GPU 算子
   ↓
实现可信 CPU 参考
   ↓
输入相同数据
   ↓
逐元素比较
   ↓
判断误差是否在容许范围内
```

---

## 6. 该实例如何对应本章每个知识点

### 对应“从 C++ 编程到 CUDA 编程”

CPU 版本：

```cpp
for (int i = 0; i < n; ++i)
```

CUDA 版本：

```cpp
int idx =
    blockIdx.x * blockDim.x
    + threadIdx.x;
```

原本由一个循环依次完成的工作，被分配给大量 GPU 线程。

### 对应“核函数”

```cpp
__global__ void fused_affine_relu_kernel(...)
```

该函数由 CPU 发起，在 GPU 上并行执行。

### 对应“线程模型”

```cpp
<<<blocks_per_grid, threads_per_block>>>
```

定义一个 Grid 中有多少 Block，每个 Block 中有多少 Thread。

### 对应“全局索引计算”

```cpp
int idx =
    blockIdx.x * blockDim.x
    + threadIdx.x;
```

每个线程根据自己的线程块编号和块内线程编号，计算唯一的全局数据索引。

### 对应“`nvcc` 编译流程”

```bash
nvcc fused_affine_relu.cu -o fused_affine_relu
```

`nvcc` 处理同一个 `.cu` 文件中的 Host 代码和 Device 代码。

### 对应“GPU 计算能力”

使用：

```bash
-arch=sm_XX
```

可以指定为哪一代 GPU 架构生成机器码。

### 对应“CUDA 程序兼容性”

程序是否能运行取决于：

- 目标 GPU；
- 编译目标；
- 驱动；
- Toolkit；
- 是否使用了目标 GPU 不支持的能力。

---

## 7. 将实例改写为 Grid-Stride Loop

原核函数是一个线程只处理一个元素：

```cpp
__global__ void fused_affine_relu_kernel(
    const float* x,
    float* y,
    float alpha,
    float bias,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float value = alpha * x[idx] + bias;
        y[idx] = value > 0.0f ? value : 0.0f;
    }
}
```

可以改成：

```cpp
__global__ void fused_affine_relu_grid_stride_kernel(
    const float* x,
    float* y,
    float alpha,
    float bias,
    int n
) {
    int idx =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    int stride =
        blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        float value = alpha * x[i] + bias;
        y[i] = value > 0.0f ? value : 0.0f;
    }
}
```

假设只启动：

```text
4 个线程块 × 256 个线程 = 1024 个线程
```

即使：

```text
n = 1,000,000
```

线程也可以反复处理：

```text
线程 0：
0、1024、2048、3072、...

线程 1：
1、1025、2049、3073、...
```

这种写法体现了 CUDA 中“逻辑数据规模”和“实际启动线程规模”不必完全相等。

---

## 8. 从本实例继续过渡到 GEMM 算子

本实例使用一维索引：

```cpp
int idx =
    blockIdx.x * blockDim.x
    + threadIdx.x;
```

一个线程计算一个向量元素。

朴素 GEMM 可以使用二维索引：

```cpp
int row =
    blockIdx.y * blockDim.y
    + threadIdx.y;

int col =
    blockIdx.x * blockDim.x
    + threadIdx.x;
```

一个线程计算输出矩阵 `C` 的一个元素：

```cpp
C[row, col]
```

两者的本质完全一致：

```text
逐元素算子：
线程索引 → 一维元素位置

朴素 GEMM：
二维线程索引 → 输出矩阵的行、列位置
```

后续 GEMM 优化只是逐步把任务映射从：

```text
一个线程 → 一个输出元素
```

改造成：

```text
一个线程块 → 一个矩阵 Tile
一个 Warp → 一个子 Tile
一个线程 → 多个寄存器中的输出元素
```

因此，只有真正掌握本章的核函数、线程模型和索引映射，才能进一步理解共享内存分块 GEMM、Warp Tile 和 Tensor Core 算子。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-gemm算子|模块-gemm算子]]

%% 项目关联导航：结束 %%
