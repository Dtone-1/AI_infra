# 一、本章在 CUDA 学习体系中的位置与学习目标

## 1. 目录识别与本章主题

根据课程截图，本章包含以下内容：

- 3.1 CUDA 矩阵加法运算程序
- 3.2 CUDA 错误检查
- 3.3 CUDA 计时
- 3.4 运行时 GPU 信息查询
- 3.5 组织线程模型

这一章处在 CUDA 入门学习中非常关键的“从会写最小程序，到能够写出可验证、可测试、可适配真实 GPU 的完整程序”阶段。

上一章通常解决的是：

- CUDA 程序由 Host 端和 Device 端共同组成；
- 什么是核函数；
- Grid、Block、Thread 之间是什么关系；
- 如何计算线程全局索引；
- 如何使用 `nvcc` 编译 CUDA 程序。

而本章进一步解决：

1. 如何把这些基础概念组合成一个完整的二维 CUDA 算子；
2. 如何发现核函数启动失败、显存访问越界等错误；
3. 如何正确测量 GPU 核函数执行时间；
4. 如何在程序运行时查询当前 GPU 的硬件属性；
5. 如何根据矩阵形状设计二维线程块和二维网格；
6. 如何从“代码能运行”逐步走向“代码正确、可测量、可移植”。

可以把本章在 CUDA 算子学习体系中的位置表示为：

```text
C++ 基础
  ↓
CUDA 核函数与线程层次
  ↓
一维线程索引
  ↓
本章：完整二维算子工程闭环
  ├── 矩阵加法
  ├── 错误检查
  ├── CUDA Event 计时
  ├── GPU 属性查询
  └── 二维线程组织
  ↓
全局内存访问与访存合并
  ↓
共享内存、线程同步
  ↓
Reduction / Softmax / GEMM 等算子
  ↓
Nsight 性能分析与算子优化
```

本章的价值不只是学会“矩阵加法”这一道题，而是第一次建立 CUDA 算子开发的完整工作流：

```text
设计线程映射
    ↓
编写核函数
    ↓
申请显存并传输数据
    ↓
启动核函数
    ↓
检查错误
    ↓
同步并计时
    ↓
复制结果
    ↓
与 CPU 参考结果比较
    ↓
查询 GPU 属性并分析线程配置
```

这个工作流会重复出现在后续绝大多数 CUDA 算子中。

---

## 2. 为什么矩阵加法适合作为本章的第一个二维算子

矩阵加法定义为：

```text
C = A + B
```

对于一个高度为 `height`、宽度为 `width` 的矩阵：

```text
C[row][col] = A[row][col] + B[row][col]
```

每个输出元素只依赖同一位置的两个输入元素，不依赖其他线程的结果。

因此，每个线程可以独立完成一个输出元素：

```text
线程(row, col)
    ↓
读取 A[row][col]
读取 B[row][col]
    ↓
计算 C[row][col]
```

它具备以下特点：

- 线程之间没有数据依赖；
- 不需要共享内存；
- 不需要线程同步；
- 可以自然使用二维线程块；
- 容易验证结果；
- 适合讲解内存布局和二维索引；
- 适合作为 CUDA 计时和错误检查示例。

不过，也必须认识到：矩阵加法在性能上通常属于**内存带宽受限算子**，因为每个元素只做一次加法，却需要多次访问全局内存。

以 `float` 为例，一个输出元素通常涉及：

- 读取 `A`：4 字节；
- 读取 `B`：4 字节；
- 写入 `C`：4 字节；
- 执行一次浮点加法。

也就是大约传输 12 字节，只做 1 次浮点运算。

这说明矩阵加法并不是计算密集型算子，而是一个非常典型的访存型算子。后续学习 Roofline 模型时，可以用它理解“算术强度低、容易受到显存带宽限制”的含义。

---

## 3. 本章与后续 AI Infra 算子开发的关系

虽然矩阵加法很简单，但本章涉及的基础能力会直接迁移到后续算子中。

### 3.1 对逐元素算子的迁移

以下算子都可以采用类似线程映射：

- ReLU；
- Sigmoid；
- GELU；
- Bias Add；
- Residual Add；
- Mask；
- Scale；
- Clamp；
- 类型转换；
- 多个逐元素操作融合。

例如：

```text
Y = GELU(X + Bias)
```

仍然可以让一个线程处理一个元素。

### 3.2 对二维算子的迁移

以下算子会继续使用二维索引：

- 矩阵转置；
- 图像处理；
- 朴素矩阵乘法；
- 二维卷积；
- Attention Score 矩阵处理；
- Masked Fill；
- Softmax 的行级处理。

### 3.3 对性能测量的迁移

任何算子优化都必须建立在可靠计时上。

后续你会比较：

- 优化前后核函数耗时；
- 不同线程块大小；
- 不同 tile 大小；
- 是否使用共享内存；
- 是否使用向量化访存；
- 自定义算子与 PyTorch/CUDA 库的性能差距；
- 不同输入形状下的延迟和吞吐。

如果计时方法错误，所有优化结论都可能错误。

### 3.4 对工程可靠性的迁移

AI Infra 中的 CUDA 算子并不是“跑一次就结束”的课堂代码，而是可能被：

- PyTorch Extension；
- vLLM；
- TensorRT-LLM；
- Triton；
- 推理服务；
- Benchmark 脚本；

反复调用。

因此必须建立以下习惯：

- 每个 CUDA API 调用都检查返回值；
- 核函数启动后检查启动错误；
- 调试阶段同步检查执行错误；
- CPU 参考结果验证正确性；
- 计时时避免把初始化和数据传输混入核函数时间；
- 查询设备属性，而不是把硬件参数写死。

---

## 4. 学完本章后应具备的能力

完成本章后，应能够独立完成以下任务。

### 4.1 编写二维矩阵加法核函数

能够写出：

```cpp
__global__ void matrix_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int rows,
    int cols
)
```

并正确计算：

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

### 4.2 理解二维矩阵的一维存储

能够解释：

```cpp
int index = row * cols + col;
```

并知道这是行优先存储布局。

### 4.3 正确组织线程块和网格

能够根据矩阵大小设计：

```cpp
dim3 block(16, 16);

dim3 grid(
    (cols + block.x - 1) / block.x,
    (rows + block.y - 1) / block.y
);
```

### 4.4 完成完整内存流程

能够独立完成：

```text
Host 分配并初始化矩阵
  ↓
cudaMalloc
  ↓
cudaMemcpyHostToDevice
  ↓
Kernel Launch
  ↓
cudaMemcpyDeviceToHost
  ↓
结果校验
  ↓
cudaFree
```

### 4.5 对 CUDA 代码进行错误检查

能够区分：

- CUDA Runtime API 错误；
- 核函数启动错误；
- 核函数执行期间的异步错误。

### 4.6 使用 CUDA Event 正确计时

能够使用：

```cpp
cudaEventCreate
cudaEventRecord
cudaEventSynchronize
cudaEventElapsedTime
cudaEventDestroy
```

测量核函数执行时间。

### 4.7 查询当前 GPU 的运行时信息

能够获得：

- GPU 名称；
- Compute Capability；
- SM 数量；
- 最大线程块线程数；
- 每个维度的线程上限；
- 每个维度的网格上限；
- 共享内存大小；
- Warp Size；
- 全局显存容量。

### 4.8 建立性能分析的初步意识

能够判断：

- 当前核函数是否主要受计算能力限制；
- 是否更可能受显存带宽限制；
- 启动线程数是否覆盖全部数据；
- 线程块大小是否合法；
- 线程访问是否连续；
- 计时结果是否包含了不应包含的操作。

# 二、课程内容取舍与算子开发补充知识

## 1. 课程目录完整性分析

截图中的五个主题构成了一个合理的 CUDA 实践章节，但如果目标是后续进行算子编写和性能优化，仅根据目录标题学习仍然不够。

本章还需要补充以下内容：

1. 矩阵的一维线性存储；
2. 行优先布局和二维坐标展平；
3. CPU 参考实现与数值校验；
4. CUDA API 错误、启动错误和执行错误的区别；
5. CUDA 异步执行对计时的影响；
6. 为什么不能使用普通 CPU 计时器直接包围核函数启动；
7. Warm-up；
8. 多次迭代取平均时间；
9. 核函数时间、端到端时间和数据传输时间的区别；
10. `dim3` 的使用方式；
11. 线程块维度限制；
12. Grid、Block、Warp、SM 的关系；
13. 线程块形状对访存连续性的影响；
14. 逻辑线程模型与物理 GPU 执行资源之间的区别；
15. 矩阵加法的带宽受限特性；
16. 有效显存带宽的粗略计算方法。

学习优先级可以分为三层。

### 必须熟练掌握

- 二维线程索引；
- 二维坐标到一维地址的映射；
- `dim3 block` 和 `dim3 grid`；
- 边界检查；
- CUDA 错误检查；
- CUDA Event 计时；
- 核函数异步执行；
- Warm-up；
- CPU 参考结果验证；
- GPU 属性查询；
- 线程块维度和线程总数限制。

### 当前需要理解

- Warp；
- SM；
- 一个线程块只能驻留在一个 SM 上；
- 一个 SM 可以同时驻留多个线程块；
- 线程块形状与连续访存；
- 内存带宽受限；
- 核函数时间与端到端时间；
- Effective Bandwidth；
- Occupancy 的基本概念。

### 当前只需要建立概念

- 多 Stream 并发计时；
- Event 在不同 Stream 中的语义；
- CUPTI；
- Nsight Systems；
- Nsight Compute；
- 硬件计数器；
- 指令级吞吐；
- 理论 Occupancy 与实际性能的差异；
- Cache 命中率；
- Memory Coalescing 的底层事务细节。

---

## 2. CUDA 矩阵加法程序

## 2.1 数学定义

设：

```text
A ∈ R^(rows × cols)
B ∈ R^(rows × cols)
C ∈ R^(rows × cols)
```

矩阵加法为：

```text
C[row, col] = A[row, col] + B[row, col]
```

其中：

```text
0 ≤ row < rows
0 ≤ col < cols
```

因为每个位置独立，所以可以让一个 GPU 线程负责一个元素。

---

## 2.2 矩阵在内存中的存储方式

C/C++ 中动态申请的一维数组通常用于存储二维矩阵。

假设矩阵为：

```text
A =
[ a00 a01 a02 a03
  a10 a11 a12 a13
  a20 a21 a22 a23 ]
```

其行优先线性存储为：

```text
a00 a01 a02 a03 a10 a11 a12 a13 a20 a21 a22 a23
```

二维坐标 `(row, col)` 对应的一维索引为：

```cpp
int index = row * cols + col;
```

例如：

```text
rows = 3
cols = 4
row = 2
col = 1
```

则：

```text
index = 2 × 4 + 1 = 9
```

对应元素 `a21`。

这条公式非常重要，因为 GPU 全局内存本质上是线性地址空间。二维线程坐标最终仍然需要转换为一维地址。

---

## 2.3 CPU 参考实现

在编写 GPU 版本前，先写一个简单可信的 CPU 版本：

```cpp
void matrix_add_cpu(
    const float* a,
    const float* b,
    float* c,
    int rows,
    int cols
) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int index = row * cols + col;
            c[index] = a[index] + b[index];
        }
    }
}
```

CPU 参考实现的作用不是追求高性能，而是提供正确性基准。

后续自定义 CUDA 算子开发中，通常需要：

```text
GPU 输出
    与
CPU / PyTorch / NumPy 参考输出
    比较
```

只有结果正确，才有资格讨论性能。

---

## 2.4 CUDA 核函数实现

```cpp
__global__ void matrix_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int rows,
    int cols
) {
    int col =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    int row =
        blockIdx.y * blockDim.y
        + threadIdx.y;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        c[index] = a[index] + b[index];
    }
}
```

该核函数包含三个关键步骤。

### 第一步：计算列索引

```cpp
int col =
    blockIdx.x * blockDim.x
    + threadIdx.x;
```

### 第二步：计算行索引

```cpp
int row =
    blockIdx.y * blockDim.y
    + threadIdx.y;
```

### 第三步：检查边界并计算

```cpp
if (row < rows && col < cols)
```

边界检查必不可少，因为线程块数量使用向上取整后，通常会启动少量超出矩阵边界的线程。

---

## 2.5 线程块和网格配置

常见配置：

```cpp
dim3 block(16, 16);
```

表示：

```text
block.x = 16
block.y = 16
block.z = 1
```

每个线程块中共有：

```text
16 × 16 = 256 个线程
```

网格大小：

```cpp
dim3 grid(
    (cols + block.x - 1) / block.x,
    (rows + block.y - 1) / block.y
);
```

例如矩阵：

```text
rows = 1000
cols = 1500
```

线程块：

```text
block = (16, 16)
```

则：

```text
grid.x = ceil(1500 / 16) = 94
grid.y = ceil(1000 / 16) = 63
```

总线程块数：

```text
94 × 63 = 5922
```

总线程数：

```text
5922 × 256 = 1,516,032
```

有效元素数：

```text
1000 × 1500 = 1,500,000
```

多出的线程通过边界判断退出。

---

## 2.6 为什么通常让 x 方向对应列

对于行优先矩阵：

```cpp
index = row * cols + col;
```

同一行相邻列在内存中连续。

如果同一个 Warp 中线程的 `threadIdx.x` 连续变化，而 `row` 相同，那么它们访问：

```text
A[row, col]
A[row, col + 1]
A[row, col + 2]
...
```

这些地址连续，更有利于形成合并访存。

因此，常见映射是：

```text
x 方向 → 列
y 方向 → 行
```

这不是语法强制要求，而是符合行优先存储和连续访问的常用设计。

---

## 2.7 矩阵加法为什么通常不需要共享内存

每个输入元素：

```text
A[row, col]
B[row, col]
```

只会被当前输出元素使用一次。

共享内存适合：

- 多个线程重复使用同一批数据；
- 线程之间交换数据；
- 减少重复全局内存访问。

但在普通矩阵加法中，输入元素没有明显复用，所以先搬到共享内存通常不会带来收益，反而增加：

- 共享内存写入；
- 同步开销；
- 代码复杂度。

因此：

```text
矩阵加法：
直接读取全局内存通常更合理

矩阵乘法：
同一 A/B Tile 会被多个线程复用，适合共享内存
```

这个区别是后续理解 GEMM 分块的重要基础。

---

## 3. CUDA 错误检查

## 3.1 为什么 CUDA 错误难以定位

CUDA 程序同时包含：

- CPU 控制代码；
- GPU 异步执行代码；
- Host 内存；
- Device 内存；
- Runtime API；
- 核函数启动配置。

很多错误不会在出现位置立即报告。

例如：

```cpp
kernel<<<grid, block>>>(...);
std::cout << "Kernel launched\n";
```

即使核函数内部发生越界访问，CPU 也可能先打印：

```text
Kernel launched
```

随后在：

```cpp
cudaMemcpy
```

或者：

```cpp
cudaDeviceSynchronize
```

处才报告错误。

因此必须理解 CUDA 错误的三种主要来源。

---

## 3.2 CUDA Runtime API 错误

例如：

```cpp
cudaMalloc
cudaMemcpy
cudaEventCreate
cudaGetDeviceProperties
```

这些函数都会返回：

```cpp
cudaError_t
```

正确做法：

```cpp
cudaError_t error = cudaMalloc(&d_a, bytes);

if (error != cudaSuccess) {
    std::cerr
        << cudaGetErrorString(error)
        << std::endl;
}
```

---

## 3.3 核函数启动错误

核函数启动后，应立即检查：

```cpp
kernel<<<grid, block>>>(...);

cudaError_t error = cudaGetLastError();
```

它可以检测某些启动阶段问题，例如：

- 每个线程块线程数超过上限；
- 线程块维度非法；
- 动态共享内存请求过大；
- Kernel 配置错误；
- 无可用的设备代码；
- 前面遗留的 CUDA 错误。

推荐写法：

```cpp
CUDA_CHECK(cudaGetLastError());
```

---

## 3.4 核函数执行错误

即使核函数能够成功启动，执行期间仍可能出错，例如：

- 数组越界；
- 非法地址；
- 未对齐访问导致某些错误；
- 错误使用共享内存；
- Device 端断言失败。

需要同步：

```cpp
CUDA_CHECK(cudaDeviceSynchronize());
```

同步会等待 GPU 完成此前提交的工作，从而暴露执行阶段的异步错误。

调试阶段推荐：

```cpp
kernel<<<grid, block>>>(...);

CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

性能测试时不应在每个小核函数后无条件同步，因为同步会阻断异步执行和并发。不过在初学和调试阶段，显式同步非常有价值。

---

## 3.5 错误检查宏

```cpp
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
```

使用方式：

```cpp
CUDA_CHECK(cudaMalloc(&d_a, bytes));

CUDA_CHECK(cudaMemcpy(
    d_a,
    h_a,
    bytes,
    cudaMemcpyHostToDevice
));

matrix_add_kernel<<<grid, block>>>(...);

CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

宏中使用：

```cpp
do {
    ...
} while (0)
```

是为了让宏在语法上表现得像一条普通语句，避免在 `if/else` 结构中产生意外行为。

---

## 3.6 常见错误示例

### 错误一：Device 指针未分配

```cpp
float* d_a;
cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
```

`d_a` 没有指向合法显存。

应先：

```cpp
cudaMalloc(&d_a, bytes);
```

### 错误二：复制方向写反

```cpp
cudaMemcpy(
    d_a,
    h_a,
    bytes,
    cudaMemcpyDeviceToHost
);
```

目标是 Device，源是 Host，方向却写成了 Device 到 Host。

正确方向：

```cpp
cudaMemcpyHostToDevice
```

### 错误三：矩阵边界判断不完整

```cpp
if (row < rows) {
    c[row * cols + col] = ...;
}
```

只检查 `row`，没有检查 `col`。

正确：

```cpp
if (row < rows && col < cols)
```

### 错误四：核函数线程数超过上限

```cpp
dim3 block(64, 64);
```

线程总数：

```text
64 × 64 = 4096
```

通常远超每线程块最大线程数。

必须查询：

```cpp
prop.maxThreadsPerBlock
```

并确保：

```text
block.x × block.y × block.z
≤ maxThreadsPerBlock
```

### 错误五：忘记检查异步执行错误

```cpp
kernel<<<grid, block>>>(...);
cudaFree(d_a);
```

核函数可能尚未完成，且执行错误未检查。

初学调试阶段应加入：

```cpp
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

---

## 4. CUDA 计时

## 4.1 为什么普通 CPU 计时可能不准确

错误示例：

```cpp
auto start = std::chrono::high_resolution_clock::now();

kernel<<<grid, block>>>(...);

auto end = std::chrono::high_resolution_clock::now();
```

由于核函数启动通常是异步的，CPU 记录 `end` 时，GPU 可能还没有开始执行或尚未执行完成。

测得的可能只是：

```text
CPU 发起核函数的提交开销
```

而不是：

```text
GPU 完成核函数的执行时间
```

如果一定要用 CPU 计时器，至少需要在结束前同步：

```cpp
cudaDeviceSynchronize();
```

但对于 GPU 核函数计时，更推荐使用 CUDA Event。

---

## 4.2 CUDA Event 基本用法

```cpp
cudaEvent_t start;
cudaEvent_t stop;

CUDA_CHECK(cudaEventCreate(&start));
CUDA_CHECK(cudaEventCreate(&stop));

CUDA_CHECK(cudaEventRecord(start));

kernel<<<grid, block>>>(...);

CUDA_CHECK(cudaEventRecord(stop));
CUDA_CHECK(cudaEventSynchronize(stop));

float milliseconds = 0.0f;

CUDA_CHECK(cudaEventElapsedTime(
    &milliseconds,
    start,
    stop
));

CUDA_CHECK(cudaEventDestroy(start));
CUDA_CHECK(cudaEventDestroy(stop));
```

`cudaEventElapsedTime` 返回的单位是毫秒。

---

## 4.3 Event 计时的执行逻辑

可以把 Event 理解为插入 GPU 工作队列中的时间标记：

```text
记录 start
    ↓
执行 kernel
    ↓
记录 stop
    ↓
等待 stop 完成
    ↓
计算 start 与 stop 的时间差
```

它更接近 GPU 时间线，因此适合测量 Device 端操作。

---

## 4.4 必须进行 Warm-up

第一次执行核函数可能包含额外开销，例如：

- CUDA Context 初始化；
- 模块加载；
- PTX JIT；
- Cache 冷启动；
- 内存页映射；
- GPU 从低功耗状态升频。

如果只执行一次并计时，结果可能明显偏大。

推荐先执行若干次预热：

```cpp
for (int i = 0; i < 10; ++i) {
    matrix_add_kernel<<<grid, block>>>(...);
}

CUDA_CHECK(cudaDeviceSynchronize());
```

然后再正式计时。

---

## 4.5 多次迭代取平均

单次测量容易受到噪声影响。

推荐：

```cpp
const int iterations = 100;

CUDA_CHECK(cudaEventRecord(start));

for (int i = 0; i < iterations; ++i) {
    matrix_add_kernel<<<grid, block>>>(...);
}

CUDA_CHECK(cudaEventRecord(stop));
CUDA_CHECK(cudaEventSynchronize(stop));

float total_ms = 0.0f;

CUDA_CHECK(cudaEventElapsedTime(
    &total_ms,
    start,
    stop
));

float average_ms =
    total_ms / iterations;
```

同时应在计时循环后检查：

```cpp
CUDA_CHECK(cudaGetLastError());
```

---

## 4.6 核函数时间、传输时间和端到端时间

必须区分三类时间。

### 核函数时间

只测：

```text
Kernel 执行
```

适合分析算子本身性能。

### 数据传输时间

单独测：

```text
Host → Device
Device → Host
```

适合分析 PCIe 传输开销。

### 端到端时间

测量：

```text
数据准备
+ H2D
+ Kernel
+ D2H
+ 同步
```

适合分析整个应用流程的真实延迟。

在深度学习框架中，输入和输出通常已经位于 GPU 上，因此自定义算子 Benchmark 通常更关注核函数时间，而不是每次都包含 CPU-GPU 数据传输。

---

## 4.7 矩阵加法的有效带宽

对于 `float` 矩阵加法：

```text
C = A + B
```

每个元素大致涉及：

```text
读取 A：4 字节
读取 B：4 字节
写入 C：4 字节
合计：12 字节
```

若矩阵共有：

```text
N = rows × cols
```

核函数平均耗时为：

```text
t 秒
```

则可以粗略估算有效带宽：

```text
Effective Bandwidth
= 12 × N / t
```

换算为 GB/s：

```text
Bandwidth_GB_s
= 12 × N / t / 10^9
```

如果计时单位是毫秒：

```text
Bandwidth_GB_s
= 12 × N / (time_ms × 10^6)
```

代码示例：

```cpp
double bytes_moved =
    3.0 * rows * cols * sizeof(float);

double bandwidth_gb_s =
    bytes_moved / (average_ms * 1e-3) / 1e9;
```

这个指标比 GFLOPS 更适合矩阵加法，因为该算子主要受内存带宽限制。

---

## 4.8 计时常见错误

### 错误一：没有预热

第一次运行时间不能代表稳定性能。

### 错误二：只测一次

单次时间容易抖动。

### 错误三：把内存申请放入核函数计时

```cpp
cudaMalloc
cudaMemcpy
kernel
cudaFree
```

如果目的是测 Kernel，就不应把其他操作包含进去。

### 错误四：不同实现的计时范围不一致

例如：

```text
版本 A：只测 Kernel
版本 B：测了 Kernel + D2H
```

这样的比较没有意义。

### 错误五：使用 `cudaDeviceSynchronize` 的位置不一致

同步位置会显著影响 CPU 计时结果。

### 错误六：Debug 编译下测性能

使用：

```bash
-G
```

会影响设备代码优化。

性能测试应使用类似：

```bash
-O3 -lineinfo
```

而不是设备 Debug 模式。

---

## 5. 运行时 GPU 信息查询

## 5.1 为什么算子需要查询 GPU 属性

不同 GPU 的硬件限制和性能特征不同。

如果程序把线程配置写死，就可能遇到：

- 线程块线程数超过上限；
- 共享内存超过上限；
- Warp 大小假设不匹配；
- 使用目标 GPU 不支持的功能；
- 在不同 GPU 上性能差异巨大。

因此，CUDA 程序可以在运行时查询硬件属性。

---

## 5.2 查询 GPU 数量

```cpp
int device_count = 0;

CUDA_CHECK(cudaGetDeviceCount(&device_count));

std::cout << "CUDA device count: "
          << device_count
          << std::endl;
```

如果返回 0，可能原因包括：

- 没有 NVIDIA GPU；
- 驱动未正确安装；
- WSL GPU 映射异常；
- 当前环境不可见；
- 容器没有获得 GPU 权限。

---

## 5.3 查询当前设备

```cpp
int device_id = 0;

CUDA_CHECK(cudaGetDevice(&device_id));
```

切换设备：

```cpp
CUDA_CHECK(cudaSetDevice(device_id));
```

多 GPU 程序中，设备选择非常重要，因为：

- 显存属于具体 GPU；
- 核函数在当前设备执行；
- Event 和 Stream 也与设备上下文相关。

---

## 5.4 查询设备属性

```cpp
cudaDeviceProp prop;

CUDA_CHECK(cudaGetDeviceProperties(
    &prop,
    device_id
));
```

常用字段如下。

### GPU 名称

```cpp
prop.name
```

### Compute Capability

```cpp
prop.major
prop.minor
```

### SM 数量

```cpp
prop.multiProcessorCount
```

### 每线程块最大线程数

```cpp
prop.maxThreadsPerBlock
```

### 线程块各维度上限

```cpp
prop.maxThreadsDim[0]
prop.maxThreadsDim[1]
prop.maxThreadsDim[2]
```

### 网格各维度上限

```cpp
prop.maxGridSize[0]
prop.maxGridSize[1]
prop.maxGridSize[2]
```

### Warp Size

```cpp
prop.warpSize
```

### 每个线程块可用共享内存

```cpp
prop.sharedMemPerBlock
```

### 全局显存总量

```cpp
prop.totalGlobalMem
```

### 单个 SM 最大线程数

```cpp
prop.maxThreadsPerMultiProcessor
```

这些参数会影响线程块设计和资源使用。

---

## 5.5 查询当前显存使用情况

```cpp
std::size_t free_bytes = 0;
std::size_t total_bytes = 0;

CUDA_CHECK(cudaMemGetInfo(
    &free_bytes,
    &total_bytes
));
```

转换为 GiB：

```cpp
double free_gib =
    free_bytes / 1024.0 / 1024.0 / 1024.0;

double total_gib =
    total_bytes / 1024.0 / 1024.0 / 1024.0;
```

需要注意：

```text
totalGlobalMem
```

通常代表设备总显存属性，而：

```text
cudaMemGetInfo
```

可以查询当前上下文下的可用显存情况。

---

## 5.6 一个完整的 GPU 信息打印函数

```cpp
void print_gpu_info(int device_id) {
    cudaDeviceProp prop;

    CUDA_CHECK(cudaGetDeviceProperties(
        &prop,
        device_id
    ));

    std::cout
        << "GPU name: "
        << prop.name
        << '\n';

    std::cout
        << "Compute capability: "
        << prop.major
        << "."
        << prop.minor
        << '\n';

    std::cout
        << "SM count: "
        << prop.multiProcessorCount
        << '\n';

    std::cout
        << "Warp size: "
        << prop.warpSize
        << '\n';

    std::cout
        << "Max threads per block: "
        << prop.maxThreadsPerBlock
        << '\n';

    std::cout
        << "Max block dimensions: "
        << prop.maxThreadsDim[0]
        << " x "
        << prop.maxThreadsDim[1]
        << " x "
        << prop.maxThreadsDim[2]
        << '\n';

    std::cout
        << "Max grid dimensions: "
        << prop.maxGridSize[0]
        << " x "
        << prop.maxGridSize[1]
        << " x "
        << prop.maxGridSize[2]
        << '\n';

    std::cout
        << "Shared memory per block: "
        << prop.sharedMemPerBlock / 1024.0
        << " KiB\n";

    std::cout
        << "Total global memory: "
        << prop.totalGlobalMem
               / 1024.0
               / 1024.0
               / 1024.0
        << " GiB\n";
}
```

---

## 6. 组织线程模型

## 6.1 逻辑线程组织

CUDA 中的逻辑层次：

```text
Kernel Launch
    ↓
Grid
    ↓
Blocks
    ↓
Threads
```

一次核函数启动对应一个 Grid。

每个线程都有：

```cpp
threadIdx
blockIdx
blockDim
gridDim
```

这些变量用于建立：

```text
线程坐标
    ↔
数据坐标
```

算子开发的核心问题之一就是：

> 如何把数据划分给 Grid、Block、Warp 和 Thread？

---

## 6.2 物理执行组织

GPU 的物理执行层次可以简化为：

```text
GPU
  ↓
多个 SM
  ↓
每个 SM 调度多个 Warp
  ↓
一个 Warp 通常包含 32 个线程
```

逻辑线程块和物理 SM 的关系：

- 一个线程块会被分配到一个 SM；
- 一个线程块不会拆到多个 SM；
- 一个 SM 可以同时驻留多个线程块；
- 不同线程块可能以任意顺序执行；
- 同一线程块中的线程可以共享共享内存；
- 不同线程块不能依赖普通块内同步进行协调。

---

## 6.3 一维、二维和三维线程组织

### 一维

```cpp
dim3 block(256);
dim3 grid((n + block.x - 1) / block.x);
```

适合：

- 向量；
- 扁平数组；
- Token 序列；
- 逐元素算子。

### 二维

```cpp
dim3 block(16, 16);

dim3 grid(
    (cols + block.x - 1) / block.x,
    (rows + block.y - 1) / block.y
);
```

适合：

- 矩阵；
- 图像；
- 二维输出空间；
- 朴素 GEMM。

### 三维

```cpp
dim3 block(8, 8, 4);

dim3 grid(
    ceil_div(width, block.x),
    ceil_div(height, block.y),
    ceil_div(depth, block.z)
);
```

适合：

- 三维体数据；
- 体素；
- 某些批量空间问题。

维度只是逻辑组织形式，不代表硬件真实存在几何形状的线程。

---

## 6.4 线程块大小限制

线程块必须同时满足：

```text
block.x ≤ maxThreadsDim[0]
block.y ≤ maxThreadsDim[1]
block.z ≤ maxThreadsDim[2]
```

并且：

```text
block.x × block.y × block.z
≤ maxThreadsPerBlock
```

例如：

```cpp
dim3 block(32, 32);
```

线程数为：

```text
32 × 32 = 1024
```

可能合法，但已经达到很多 GPU 的每块线程上限。

而：

```cpp
dim3 block(64, 16);
```

线程数也是：

```text
64 × 16 = 1024
```

但还需要检查 `block.x`、`block.y` 各自是否满足维度上限。

---

## 6.5 为什么常见配置是 16×16 或 32×8

两者都是：

```text
256 个线程
```

### `16 × 16`

优点：

- 形状对称；
- 适合二维矩阵；
- 容易理解；
- 常用于教学；
- 每块 256 个线程。

### `32 × 8`

优点：

- x 方向长度为 32；
- 同一 Warp 更容易覆盖一行中的连续 32 个元素；
- 对行优先连续访问更直观。

但不存在对所有算子和输入都最优的固定配置。

实际性能还取决于：

- 访存布局；
- 寄存器使用；
- 共享内存；
- 分支；
- Warp 数量；
- 输入形状；
- GPU 架构。

本章应掌握“合法且合理”，后续再学习“如何找到最优”。

---

## 6.6 Warp 与线程块

假设：

```cpp
dim3 block(16, 16);
```

线程块共有 256 个线程，也就是 8 个 Warp。

线程在线程块内的线性编号通常为：

```cpp
int local_tid =
    threadIdx.y * blockDim.x
    + threadIdx.x;
```

Warp 编号：

```cpp
int warp_id =
    local_tid / warpSize;
```

Lane 编号：

```cpp
int lane_id =
    local_tid % warpSize;
```

其中：

- `warp_id` 表示线程属于块内第几个 Warp；
- `lane_id` 表示线程在 Warp 内的位置。

这些概念会在：

- Warp Reduction；
- Shuffle 指令；
- Tensor Core；
- GEMM Warp Tile；
- FlashAttention；

中频繁出现。

---

## 6.7 线程块形状与访存模式

对于行优先矩阵：

```cpp
index = row * cols + col;
```

如果 Warp 中相邻线程的 `col` 连续，则相邻线程读取相邻地址。

例如：

```text
线程 0 → A[row, 0]
线程 1 → A[row, 1]
线程 2 → A[row, 2]
...
```

这通常比：

```text
线程 0 → A[0, col]
线程 1 → A[1, col]
线程 2 → A[2, col]
...
```

更利于连续访问，因为后一种访问之间相隔一个完整行宽。

因此设计线程映射时，不能只保证“每个线程找到一个元素”，还需要考虑：

> 相邻线程访问的内存地址是否连续？

这就是从功能正确走向性能优化的第一步。

---

## 6.8 Occupancy 当前应如何理解

Occupancy 通常表示：

```text
一个 SM 上实际驻留的活跃 Warp 数
÷
该 SM 理论允许的最大活跃 Warp 数
```

它受以下资源限制：

- 每个线程寄存器数量；
- 每个线程块共享内存；
- 每个线程块线程数；
- GPU 架构上限。

本章不要求计算 Occupancy，但需要建立两个认识：

1. 线程块不是越大越好；
2. Occupancy 不是越高性能就一定越好。

一个线程块过大，可能导致：

- 单块资源占用过多；
- 每个 SM 能同时驻留的线程块减少；
- 调度灵活性降低。

但线程块过小，也可能无法提供足够 Warp 隐藏访存延迟。

---

## 7. 本章常见理解误区

### 误区一：矩阵是二维的，所以显存也是二维的

GPU 全局内存本质上是线性地址空间。二维矩阵通常通过：

```cpp
row * cols + col
```

展平。

### 误区二：核函数启动结束就代表 GPU 执行结束

核函数启动通常是异步的。

必须通过 Event、同步操作或后续依赖操作确认完成。

### 误区三：CPU 计时器包住 Kernel Launch 就能得到 GPU 时间

如果没有同步，测到的主要是提交开销。

### 误区四：只要结果正确，线程组织就合理

不同映射都可能计算正确，但访存模式可能差异巨大。

### 误区五：线程块越大越快

线程块大小会影响：

- Warp 数；
- 寄存器；
- 共享内存；
- Occupancy；
- 调度。

没有“越大越快”的普遍规律。

### 误区六：错误一定出现在报错的那一行

由于异步执行，错误可能在后续同步或复制时才暴露。

### 误区七：矩阵加法适合用共享内存优化

普通矩阵加法没有明显的数据复用，使用共享内存通常收益有限。

### 误区八：GPU 理论带宽就是算子一定能达到的带宽

理论带宽是硬件上限，实际还受到：

- 访问模式；
- Cache；
- 指令；
- 调度；
- 边界；
- 时钟；
- 功耗状态；

等影响。

# 三、章节综合实例

## 1. 实例目标

编写一个完整的 CUDA 矩阵加法程序，实现：

```text
C = A + B
```

程序需要同时完成：

1. 查询 GPU 数量和当前 GPU 属性；
2. 在 Host 端创建矩阵；
3. 在 Device 端申请显存；
4. 将输入矩阵复制到 GPU；
5. 使用二维线程块执行矩阵加法；
6. 检查所有 CUDA API 错误；
7. 检查核函数启动和执行错误；
8. 使用 CUDA Event 进行 Warm-up 和多次计时；
9. 将结果复制回 CPU；
10. 使用 CPU 参考实现验证结果；
11. 计算平均核函数时间；
12. 估算有效显存带宽；
13. 释放所有 CUDA 资源。

这个实例建立的是后续 CUDA 算子开发的标准模板：

```text
设备查询
  ↓
输入准备
  ↓
显存管理
  ↓
线程组织
  ↓
Kernel Launch
  ↓
错误检查
  ↓
性能计时
  ↓
正确性验证
  ↓
性能指标计算
```

---

## 2. 完整代码

将下面代码保存为：

```text
matrix_add_benchmark.cu
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

inline int ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

__global__ void matrix_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int rows,
    int cols
) {
    int col =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    int row =
        blockIdx.y * blockDim.y
        + threadIdx.y;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        c[index] = a[index] + b[index];
    }
}

void matrix_add_cpu(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c,
    int rows,
    int cols
) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int index = row * cols + col;
            c[index] = a[index] + b[index];
        }
    }
}

void print_gpu_info(int device_id) {
    cudaDeviceProp prop{};

    CUDA_CHECK(cudaGetDeviceProperties(
        &prop,
        device_id
    ));

    std::cout << "========== GPU Information ==========\n";

    std::cout
        << "Device ID: "
        << device_id
        << '\n';

    std::cout
        << "GPU name: "
        << prop.name
        << '\n';

    std::cout
        << "Compute capability: "
        << prop.major
        << "."
        << prop.minor
        << '\n';

    std::cout
        << "SM count: "
        << prop.multiProcessorCount
        << '\n';

    std::cout
        << "Warp size: "
        << prop.warpSize
        << '\n';

    std::cout
        << "Max threads per block: "
        << prop.maxThreadsPerBlock
        << '\n';

    std::cout
        << "Max block dimensions: "
        << prop.maxThreadsDim[0]
        << " x "
        << prop.maxThreadsDim[1]
        << " x "
        << prop.maxThreadsDim[2]
        << '\n';

    std::cout
        << "Max grid dimensions: "
        << prop.maxGridSize[0]
        << " x "
        << prop.maxGridSize[1]
        << " x "
        << prop.maxGridSize[2]
        << '\n';

    std::cout
        << "Shared memory per block: "
        << prop.sharedMemPerBlock / 1024.0
        << " KiB\n";

    std::cout
        << "Total global memory: "
        << std::fixed
        << std::setprecision(2)
        << prop.totalGlobalMem
               / 1024.0
               / 1024.0
               / 1024.0
        << " GiB\n";

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;

    CUDA_CHECK(cudaMemGetInfo(
        &free_bytes,
        &total_bytes
    ));

    std::cout
        << "Current free memory: "
        << free_bytes
               / 1024.0
               / 1024.0
               / 1024.0
        << " GiB\n";

    std::cout << "=====================================\n";
}

bool check_result(
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    float tolerance
) {
    if (expected.size() != actual.size()) {
        std::cerr
            << "Result size mismatch.\n";
        return false;
    }

    float max_error = 0.0f;
    std::size_t max_error_index = 0;

    for (std::size_t i = 0; i < expected.size(); ++i) {
        float error =
            std::fabs(expected[i] - actual[i]);

        if (error > max_error) {
            max_error = error;
            max_error_index = i;
        }

        if (error > tolerance) {
            std::cerr
                << "Mismatch at index "
                << i
                << ": expected = "
                << expected[i]
                << ", actual = "
                << actual[i]
                << ", error = "
                << error
                << '\n';

            return false;
        }
    }

    std::cout
        << "Maximum absolute error: "
        << max_error
        << " at index "
        << max_error_index
        << '\n';

    return true;
}

int main() {
    const int rows = 4096;
    const int cols = 4096;

    const int warmup_iterations = 10;
    const int benchmark_iterations = 100;

    const std::size_t num_elements =
        static_cast<std::size_t>(rows)
        * static_cast<std::size_t>(cols);

    const std::size_t bytes =
        num_elements * sizeof(float);

    // 1. 查询 CUDA 设备数量
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count <= 0) {
        std::cerr
            << "No CUDA-capable GPU found.\n";
        return EXIT_FAILURE;
    }

    // 2. 选择并查询 GPU
    const int device_id = 0;
    CUDA_CHECK(cudaSetDevice(device_id));
    print_gpu_info(device_id);

    // 3. Host 端矩阵
    std::vector<float> h_a(num_elements);
    std::vector<float> h_b(num_elements);
    std::vector<float> h_c_gpu(num_elements, 0.0f);
    std::vector<float> h_c_cpu(num_elements, 0.0f);

    std::mt19937 generator(42);
    std::uniform_real_distribution<float>
        distribution(-1.0f, 1.0f);

    for (std::size_t i = 0; i < num_elements; ++i) {
        h_a[i] = distribution(generator);
        h_b[i] = distribution(generator);
    }

    // 4. CPU 参考结果
    matrix_add_cpu(
        h_a,
        h_b,
        h_c_cpu,
        rows,
        cols
    );

    // 5. Device 指针
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // 6. 输入复制到 GPU
    CUDA_CHECK(cudaMemcpy(
        d_a,
        h_a.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_b,
        h_b.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    // 7. 二维线程组织
    dim3 block(16, 16);

    dim3 grid(
        ceil_div(cols, static_cast<int>(block.x)),
        ceil_div(rows, static_cast<int>(block.y))
    );

    std::cout << "\n========== Launch Configuration ==========\n";

    std::cout
        << "Matrix shape: "
        << rows
        << " x "
        << cols
        << '\n';

    std::cout
        << "Block shape: "
        << block.x
        << " x "
        << block.y
        << " x "
        << block.z
        << '\n';

    std::cout
        << "Threads per block: "
        << block.x * block.y * block.z
        << '\n';

    std::cout
        << "Grid shape: "
        << grid.x
        << " x "
        << grid.y
        << " x "
        << grid.z
        << '\n';

    std::cout << "==========================================\n";

    // 8. Warm-up
    for (int i = 0; i < warmup_iterations; ++i) {
        matrix_add_kernel<<<grid, block>>>(
            d_a,
            d_b,
            d_c,
            rows,
            cols
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 9. 创建 CUDA Event
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 10. 多次执行并计时
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < benchmark_iterations; ++i) {
        matrix_add_kernel<<<grid, block>>>(
            d_a,
            d_b,
            d_c,
            rows,
            cols
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms,
        start,
        stop
    ));

    const float average_ms =
        total_ms / benchmark_iterations;

    // 11. 将结果复制回 Host
    CUDA_CHECK(cudaMemcpy(
        h_c_gpu.data(),
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    // 12. 正确性检查
    const bool correct = check_result(
        h_c_cpu,
        h_c_gpu,
        1e-6f
    );

    // 13. 估算有效显存带宽
    //
    // 每个元素：
    // 读取 A：4 字节
    // 读取 B：4 字节
    // 写入 C：4 字节
    const double transferred_bytes =
        3.0
        * static_cast<double>(num_elements)
        * sizeof(float);

    const double average_seconds =
        static_cast<double>(average_ms) * 1e-3;

    const double bandwidth_gb_s =
        transferred_bytes
        / average_seconds
        / 1e9;

    std::cout << "\n========== Benchmark Result ==========\n";

    std::cout
        << "Warm-up iterations: "
        << warmup_iterations
        << '\n';

    std::cout
        << "Benchmark iterations: "
        << benchmark_iterations
        << '\n';

    std::cout
        << "Total kernel time: "
        << total_ms
        << " ms\n";

    std::cout
        << "Average kernel time: "
        << average_ms
        << " ms\n";

    std::cout
        << "Estimated effective bandwidth: "
        << bandwidth_gb_s
        << " GB/s\n";

    std::cout
        << "Result check: "
        << (correct ? "PASSED" : "FAILED")
        << '\n';

    std::cout << "======================================\n";

    // 14. 释放 Event
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 15. 释放显存
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

---

## 3. 编译与运行

### 3.1 编译

```bash
nvcc \
    -std=c++17 \
    -O3 \
    -lineinfo \
    matrix_add_benchmark.cu \
    -o matrix_add_benchmark
```

### 3.2 运行

```bash
./matrix_add_benchmark
```

### 3.3 Debug 编译

在定位越界或执行错误时，可以临时使用：

```bash
nvcc \
    -std=c++17 \
    -g \
    -G \
    matrix_add_benchmark.cu \
    -o matrix_add_debug
```

但不能使用该版本进行正式性能评估，因为 `-G` 会显著影响设备代码优化。

---

## 4. 按时间线理解程序

### 阶段一：查询 GPU

```cpp
cudaGetDeviceCount
cudaSetDevice
cudaGetDeviceProperties
cudaMemGetInfo
```

程序先确认：

- 是否存在 CUDA GPU；
- 使用哪块 GPU；
- 当前 GPU 支持的线程和资源上限。

### 阶段二：Host 初始化矩阵

```cpp
h_a
h_b
h_c_gpu
h_c_cpu
```

这些数据位于 CPU 内存。

### 阶段三：CPU 计算参考结果

```cpp
matrix_add_cpu(...)
```

该结果用于验证 CUDA 核函数。

### 阶段四：申请显存

```cpp
cudaMalloc(&d_a, bytes)
cudaMalloc(&d_b, bytes)
cudaMalloc(&d_c, bytes)
```

### 阶段五：传输输入

```cpp
cudaMemcpyHostToDevice
```

只将 `A`、`B` 复制到 GPU。

`C` 是输出，因此不需要先将 CPU 结果复制到 `d_c`。

### 阶段六：设计二维线程模型

```cpp
dim3 block(16, 16);
```

每块 256 个线程。

```cpp
dim3 grid(
    ceil_div(cols, 16),
    ceil_div(rows, 16)
);
```

每个线程负责一个 `(row, col)` 元素。

### 阶段七：Warm-up

先执行 10 次，不计入正式结果，降低首次运行额外开销的影响。

### 阶段八：CUDA Event 计时

记录：

```text
start
  ↓
执行 100 次 Kernel
  ↓
stop
```

平均时间：

```text
total_ms / 100
```

### 阶段九：结果复制和校验

```cpp
cudaMemcpyDeviceToHost
```

然后逐元素比较 CPU 和 GPU 输出。

### 阶段十：计算有效带宽

矩阵加法每个元素估算传输 12 字节：

```text
2 次读取 + 1 次写入
```

根据平均耗时计算 GB/s。

---

## 5. 以一个小矩阵理解线程映射

假设矩阵：

```text
rows = 5
cols = 7
```

线程块：

```cpp
dim3 block(4, 3);
```

每个线程块：

```text
4 × 3 = 12 个线程
```

网格：

```text
grid.x = ceil(7 / 4) = 2
grid.y = ceil(5 / 3) = 2
```

共启动：

```text
2 × 2 × 12 = 48 个线程
```

有效元素只有：

```text
5 × 7 = 35
```

例如：

```text
blockIdx = (1, 1)
threadIdx = (2, 1)
blockDim = (4, 3)
```

线程坐标为：

```text
col = 1 × 4 + 2 = 6
row = 1 × 3 + 1 = 4
```

对应矩阵元素：

```text
C[4][6]
```

线性索引：

```text
index = 4 × 7 + 6 = 34
```

该线程执行：

```cpp
c[34] = a[34] + b[34];
```

另一个线程：

```text
blockIdx = (1, 1)
threadIdx = (3, 2)
```

坐标：

```text
col = 1 × 4 + 3 = 7
row = 1 × 3 + 2 = 5
```

由于：

```text
col = 7，不满足 col < 7
row = 5，不满足 row < 5
```

该线程不会执行矩阵访问。

---

## 6. 通过修改线程块形状进行实验

可以分别测试：

```cpp
dim3 block(8, 8);    // 64 个线程
dim3 block(16, 16);  // 256 个线程
dim3 block(32, 8);   // 256 个线程
dim3 block(32, 16);  // 512 个线程
```

每次保持：

- 相同矩阵尺寸；
- 相同 Warm-up 次数；
- 相同 Benchmark 次数；
- 相同编译选项；
- 相同 GPU；
- 相同计时范围。

记录：

```text
Block Shape
Threads per Block
Average Time
Effective Bandwidth
Result Correctness
```

你可能会发现：

- 线程数相同，但形状不同，性能可能不同；
- 更大的线程块不一定更快；
- 32×8 可能比 16×16 更符合连续行访问；
- 对简单矩阵加法，差异可能不大，因为主要受内存带宽限制；
- 不同 GPU 上结论可能不同。

---

## 7. 从本实例进一步过渡到算子优化

本实例完成后，下一阶段可以按以下顺序扩展。

### 第一步：矩阵加法改为融合算子

例如：

```text
C = alpha × A + beta × B
```

或者：

```text
Y = ReLU(A + B)
```

这样可以理解算子融合为什么能够减少全局内存读写和 Kernel Launch。

### 第二步：矩阵转置

矩阵转置仍然使用二维线程模型，但会暴露：

- 读取连续、写入跨步；
- 共享内存；
- Bank Conflict；
- 合并访存。

### 第三步：朴素 GEMM

一个线程计算一个 `C[row, col]`：

```cpp
for (int k = 0; k < K; ++k) {
    sum += A[row * K + k]
         * B[k * N + col];
}
```

这会继续使用本章的二维索引。

### 第四步：共享内存分块 GEMM

从：

```text
一个线程独立读取所有数据
```

改为：

```text
一个线程块协作加载 A/B Tile
```

这时会用到：

- 共享内存；
- `__syncthreads()`；
- 数据复用；
- Tile；
- Warp；
- Occupancy。

因此，本章不是一个孤立的“矩阵加法练习”，而是进入 GEMM、Softmax、Reduction 和 Attention 算子开发前，必须掌握的完整工程基础。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-gemm算子|模块-gemm算子]]

%% 项目关联导航：结束 %%
