# gemm_test_perf_gemm_精读

本文按照 `test_perf_gemm.cpp` 的执行顺序拆分源码。每一节先给出一块完整源代码，再解释其中的 C++ 语法、CUDA/cuBLAS API、变量含义和实际执行过程。

---

## 1. 引入头文件

```cpp
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <filesystem>

#include "gemm/launcher.h"
```

### 详细解释

`#include` 是 C/C++ 的预处理指令。正式编译前，预处理器会把头文件中提供的声明引入当前源文件，使后面的代码能够使用相应函数、类型和类。

```cpp
#include <cublas_v2.h>
```

引入 NVIDIA cuBLAS 的 API 声明。当前程序会使用：

```cpp
cublasCreate
cublasSgemm
cublasDestroy
```

cuBLAS 是 NVIDIA 提供的 GPU 线性代数库。这里把 cuBLAS 的计算结果和性能作为自定义 GEMM 的参考基准。

```cpp
#include <cuda_runtime.h>
```

引入 CUDA Runtime API，例如：

```cpp
cudaMalloc
cudaMemcpy
cudaFree
cudaEventCreate
cudaDeviceSynchronize
```

这些函数负责 GPU 显存管理、CPU/GPU 数据传输、GPU 同步、计时和错误检查。

其余标准库头文件的作用如下：

- `<cmath>`：提供 `fabsf`，用于计算两个 `float` 之差的绝对值。
- `<fstream>`：提供 `std::ofstream`，用于写入 CSV 文件。
- `<iostream>`：提供 `std::cout`、`std::cerr` 和 `std::endl`。
- `<string>`：提供 C++ 字符串类型 `std::string`。
- `<vector>`：提供动态数组 `std::vector`。
- `<filesystem>`：提供目录和文件系统操作。

```cpp
#include "gemm/launcher.h"
```

这是项目自己的头文件，其中应声明了：

```cpp
launch_gemm(...)
```

测试程序通过这个普通 C++ 接口间接启动自定义 CUDA GEMM 核函数。

尖括号 `<...>` 通常用于系统库、标准库或第三方库；双引号 `"..."` 通常用于当前项目自己的头文件。

---

## 2. CUDA Runtime 错误检查函数

```cpp
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
```

### 详细解释

```cpp
void checkCudaError(...)
```

定义了一个没有返回值的函数。`void` 表示该函数只执行检查，不返回计算结果。

函数有两个参数：

```cpp
cudaError_t err
```

`cudaError_t` 是 CUDA Runtime 定义的错误状态类型。许多 CUDA API 都返回该类型。如果调用成功，返回值为：

```cpp
cudaSuccess
```

否则会返回显存不足、参数非法、设备不可用等错误码。

```cpp
const char* msg
```

表示指向只读字符序列的指针，用来接收调用者传入的错误说明，例如：

```cpp
"cudaMalloc d_A failed"
```

```cpp
if (err != cudaSuccess)
```

`!=` 表示“不等于”。只有 CUDA API 返回失败状态时，才执行花括号中的错误处理。

```cpp
std::cerr << msg
          << " CUDA ERROR: "
          << cudaGetErrorString(err)
          << std::endl;
```

`std::cerr` 是标准错误输出流。`<<` 把右侧内容依次写入输出流。

```cpp
cudaGetErrorString(err)
```

把 CUDA 错误码转换为可阅读的字符串。例如显存不足时可能输出：

```text
out of memory
```

```cpp
exit(EXIT_FAILURE);
```

立即终止整个程序，并向操作系统返回失败状态。CUDA 操作一旦失败，后续结果通常已经不可信，所以这里选择直接退出。

---

## 3. cuBLAS 错误检查函数

```cpp
void checkCublasError(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << msg << " CUBLAS ERROR: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}
```

### 详细解释

cuBLAS 和 CUDA Runtime 是两套不同的 API：

- CUDA Runtime API 返回 `cudaError_t`；
- cuBLAS API 返回 `cublasStatus_t`。

因此代码单独编写了 cuBLAS 错误检查函数。

```cpp
CUBLAS_STATUS_SUCCESS
```

代表 cuBLAS 调用成功。

当前代码直接打印：

```cpp
status
```

所以失败时可能看到一个整数状态码，而不是完整错误字符串。需要根据 `cublasStatus_t` 的枚举定义查询具体含义。

该函数使主程序可以写成：

```cpp
checkCublasError(cublasSgemm(...), "cublasSgemm failed");
```

也就是先执行 cuBLAS 函数，再立即统一检查其返回值。

---

## 4. 生成测试尺寸

```cpp
std::vector<int> generateTestSizes() {
    std::vector<int> sizes;
    // 原有多倍数尺寸
    for (int i = 256; i <= 8192; i += 256) {
        sizes.push_back(i);
    }
    // 非对齐尺寸
    sizes.push_back(100);
    sizes.push_back(257);
    sizes.push_back(511);
    sizes.push_back(1000);
    sizes.push_back(1234);
    sizes.push_back(2047);
    return sizes;
}
```

### 详细解释

函数返回类型是：

```cpp
std::vector<int>
```

表示返回一个保存整数的动态数组。

```cpp
std::vector<int> sizes;
```

创建空向量。此时其中没有任何测试尺寸。

```cpp
for (int i = 256; i <= 8192; i += 256)
```

标准 `for` 循环由三部分组成：

```cpp
for (初始化; 继续循环的条件; 每轮结束后的更新)
```

这里表示：

1. `i` 从 256 开始；
2. 只要 `i <= 8192` 就继续；
3. 每轮结束后增加 256。

因此会生成：

```text
256, 512, 768, ..., 8192
```

```cpp
sizes.push_back(i);
```

把当前 `i` 追加到向量末尾。

之后又加入：

```text
100, 257, 511, 1000, 1234, 2047
```

这些尺寸不是常见 tile 大小的整数倍，用于检查自定义 kernel 的边界处理。

例如 kernel 每个线程块计算 `128 × 128` 的 tile，而矩阵边长是 257，那么最后一组线程块会有大量线程落在矩阵范围之外。kernel 必须用类似条件保护：

```cpp
if (row < M && col < N)
```

否则可能越界访问显存。

```cpp
return sizes;
```

把最终向量返回给主函数。

---

## 5. 主函数、目录和 CSV 文件

```cpp
int main() {
    std::filesystem::create_directories("./results");
    std::ofstream csv("./results/gemm_perf_results.csv");
    if (!csv.is_open()) {
        std::cerr << "failed to open gemm_perf_results.csv" << std::endl;
        return EXIT_FAILURE;
    }

    // CSV表头
    csv << "size,error_count,cublas_gflops,my_gflops,ratio\n";

    std::vector<int> sizes = generateTestSizes();
    int cnt = 0;
    float sum = 0.0f;
```

### 详细解释

```cpp
int main()
```

`main` 是 C++ 程序入口。程序启动后从这里开始执行。返回 `0` 表示成功，返回非零值通常表示失败。

```cpp
std::filesystem::create_directories("./results");
```

在当前工作目录下创建 `results` 文件夹。`./` 表示程序运行时的当前目录，不一定等于源码文件所在目录。

如果目录已经存在，`create_directories` 通常不会报错。

```cpp
std::ofstream csv("./results/gemm_perf_results.csv");
```

创建输出文件流并尝试打开 CSV 文件。默认情况下，如果文件已存在，旧内容通常会被清空并重新写入。

```cpp
if (!csv.is_open())
```

`csv.is_open()` 返回文件是否成功打开。前面的 `!` 是逻辑非，所以整个条件表示“文件没有成功打开”。

打开失败时：

```cpp
return EXIT_FAILURE;
```

从 `main` 返回失败状态并结束程序。

```cpp
csv << "size,error_count,cublas_gflops,my_gflops,ratio\n";
```

写入 CSV 表头，五列分别是：

1. 矩阵尺寸；
2. 错误元素数量；
3. cuBLAS 的 GFLOPS；
4. 自定义 GEMM 的 GFLOPS；
5. 自定义性能与 cuBLAS 性能之比。

```cpp
std::vector<int> sizes = generateTestSizes();
```

调用前面的函数获得所有测试尺寸。

```cpp
int cnt = 0;
float sum = 0.0f;
```

`cnt` 统计完成了多少个尺寸，`sum` 累加每个尺寸的性能比例，最后用于计算平均比例。

`0.0f` 后面的 `f` 表示这是 `float` 字面量，而不是默认的 `double`。

---

## 6. 遍历尺寸并设置 GEMM 维度

```cpp
    for (int sz : sizes) {
        std::cout << "Test size: " << sz << std::endl;

        const int M = sz;
        const int N = sz;
        const int K = sz;
```

### 详细解释

```cpp
for (int sz : sizes)
```

是 C++11 的范围 `for` 循环。它会依次取出 `sizes` 中每个元素，保存到当前变量 `sz`。

例如向量开头是：

```text
256, 512, 768
```

那么前三轮分别有：

```text
sz = 256
sz = 512
sz = 768
```

```cpp
const int M = sz;
const int N = sz;
const int K = sz;
```

标准 GEMM 的矩阵形状是：

```text
A(M × K) × B(K × N) = C(M × N)
```

当前让 `M=N=K=sz`，所以测试的是方阵乘法。

`const` 表示变量初始化后不能修改。单独保留 `M`、`N`、`K`，能让代码保持标准 GEMM 语义，也方便将来扩展到非方阵测试。

---

## 7. 计算字节数并分配 CPU 内存

```cpp
        size_t sizeA = static_cast<size_t>(M) * K * sizeof(float);
        size_t sizeB = static_cast<size_t>(K) * N * sizeof(float);
        size_t sizeC = static_cast<size_t>(M) * N * sizeof(float);

        float* A = (float*)malloc(sizeA);
        float* B = (float*)malloc(sizeB);
        float* C_cublas = (float*)malloc(sizeC);
        float* C = (float*)malloc(sizeC);
```

### 详细解释

```cpp
size_t
```

是专门表示对象大小、数组长度和内存字节数的无符号整数类型。在 64 位系统中通常是 64 位，适合表示大块内存。

矩阵 A 有：

```text
M × K
```

个 `float`，所以字节数为：

```cpp
M × K × sizeof(float)
```

在常见平台上：

```text
sizeof(float) = 4 字节
```

```cpp
static_cast<size_t>(M)
```

是 C++ 显式类型转换，先把 `M` 转成 `size_t`，让后续乘法在更适合表示内存大小的类型中进行，降低大尺寸整数溢出的风险。

```cpp
float* A = (float*)malloc(sizeA);
```

`malloc` 在 CPU 堆内存中分配 `sizeA` 个字节，返回 `void*`。代码通过：

```cpp
(float*)
```

把它转换为 `float*`。

四个 CPU 数组的作用是：

- `A`：输入矩阵 A；
- `B`：输入矩阵 B；
- `C_cublas`：保存从 GPU 拷回的 cuBLAS 结果；
- `C`：保存从 GPU 拷回的自定义 GEMM 结果。

这些都是普通主机内存，GPU kernel 不能直接把它们当作普通显存使用。

源码没有检查 `malloc` 是否返回 `nullptr`。更严谨的程序应检查主机内存是否分配成功。

---

## 8. 分配 GPU 显存

```cpp
        float *d_A, *d_B, *d_C;
        checkCudaError(cudaMalloc(&d_A, sizeA), "cudaMalloc d_A failed");
        checkCudaError(cudaMalloc(&d_B, sizeB), "cudaMalloc d_B failed");
        checkCudaError(cudaMalloc(&d_C, sizeC), "cudaMalloc d_C failed");
```

### 详细解释

前缀 `d_` 通常是 `device` 的缩写，表示这些指针保存 GPU 显存地址：

- `d_A`：GPU 上的矩阵 A；
- `d_B`：GPU 上的矩阵 B；
- `d_C`：GPU 上的矩阵 C。

```cpp
cudaMalloc(&d_A, sizeA)
```

其接口可以近似理解为：

```cpp
cudaError_t cudaMalloc(void** device_pointer, size_t bytes);
```

第一个参数传入：

```cpp
&d_A
```

即指针变量 `d_A` 自身的地址。CUDA Runtime 分配显存后，需要把得到的 GPU 地址写回 `d_A`，所以必须传入指针变量的地址。

第二个参数是分配字节数。

调用被包裹在：

```cpp
checkCudaError(...)
```

中。显存不足或参数非法时，程序会输出错误并立即退出。

---

## 9. 初始化输入并复制到 GPU

```cpp
        for (int i = 0; i < M * K; ++i) {
            A[i] = 1.0f;
        }
        for (int i = 0; i < K * N; ++i) {
            B[i] = 2.0f;
        }

        checkCudaError(cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice),
                       "copy A to device failed");
        checkCudaError(cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice),
                       "copy B to device failed");
```

### 详细解释

二维矩阵在内存中实际存成连续的一维元素，因此可以通过：

```cpp
A[i]
B[i]
```

遍历全部元素。

A 的每个元素设为 `1.0f`，B 的每个元素设为 `2.0f`。

对于任意输出元素，理论结果是：

```text
C[row,col] = 1×2 + 1×2 + ... + 1×2
```

一共有 K 项，因此：

```text
C[row,col] = 2K
```

所有输出元素都相同，便于检查正确性。

```cpp
cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice)
```

四个参数分别是：

1. 目标地址 `d_A`，位于 GPU；
2. 源地址 `A`，位于 CPU；
3. 复制字节数 `sizeA`；
4. 方向 `cudaMemcpyHostToDevice`，表示 CPU 到 GPU。

B 的复制过程相同。

在 CUDA 术语中：

- Host：CPU 和主机内存；
- Device：GPU 和设备显存。

---

## 10. 创建 cuBLAS 句柄、系数和 CUDA Event

```cpp
        cublasHandle_t handle;
        checkCublasError(cublasCreate(&handle), "cublasCreate failed");

        float alpha = 1.0f;
        float beta = 0.0f;

        cudaEvent_t start, stop;
        checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
        checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

        const int warm_time = 10;
        const int repeat_time = 5;
```

### 详细解释

```cpp
cublasHandle_t handle;
```

声明 cuBLAS 上下文句柄。句柄内部管理 cuBLAS 的运行状态和资源。

```cpp
cublasCreate(&handle)
```

初始化 cuBLAS，并把新句柄写入 `handle`。后面的 `cublasSgemm` 都需要传入它。

GEMM 标准公式是：

```text
C = alpha × op(A) × op(B) + beta × C
```

当前：

```cpp
alpha = 1.0f
beta = 0.0f
```

所以公式简化为：

```text
C = A × B
```

原 C 中的旧值不会参与结果。

```cpp
cudaEvent_t start, stop;
```

声明两个 CUDA Event。Event 可以被记录到 GPU stream 中，用于测量 GPU 工作的实际执行时间。

```cpp
cudaEventCreate(&start)
cudaEventCreate(&stop)
```

分别创建起点和终点事件。

```cpp
const int warm_time = 10;
```

正式计时前预热 10 次，降低首次 CUDA 上下文初始化、kernel 加载、GPU 升频和缓存冷启动的影响。

```cpp
const int repeat_time = 5;
```

正式计时区域连续执行 5 次 GEMM，用多次执行减少单次测量波动。

---

## 11. cuBLAS 预热

```cpp
        // cuBLAS warmup
        for (int i = 0; i < warm_time; i++) {
            checkCublasError(
                cublasSgemm(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            N, M, K,
                            &alpha,
                            d_B, N,
                            d_A, K,
                            &beta,
                            d_C, N),
                "cublasSgemm failed");
        }
        checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after cublas warmup failed");
```

### 详细解释

`cublasSgemm` 名称中的：

- `S`：single precision，表示单精度 `float`；
- `gemm`：general matrix-matrix multiplication，表示通用矩阵乘法。

它计算的标准形式是：

```text
C = alpha × op(A) × op(B) + beta × C
```

这里最容易困惑的是参数顺序：

```cpp
N, M, K,
d_B, N,
d_A, K,
d_C, N
```

原因是 cuBLAS 默认按**列主序**解释矩阵，而普通 C/C++ 代码通常按**行主序**理解矩阵。

自定义 GEMM 期望计算：

```text
C_row(M×N) = A_row(M×K) × B_row(K×N)
```

同一段连续内存如果按列主序解释，相当于看到了原行主序矩阵的转置。代码利用：

```text
(A × B)^T = B^T × A^T
```

交换 A、B 和 M、N 的位置，使 cuBLAS 计算出的内存结果与行主序的 `A × B` 对应。

两个：

```cpp
CUBLAS_OP_N
```

表示 cuBLAS 视角下不再额外转置输入矩阵。`N` 是 No transpose。

每个矩阵指针后面的整数是 leading dimension：

```cpp
d_B, N
d_A, K
d_C, N
```

leading dimension 可以理解为主存储方向上相邻向量起始地址之间的元素跨度。对列主序矩阵，通常至少等于矩阵行数。

`alpha` 和 `beta` 传入的是地址：

```cpp
&alpha
&beta
```

因为 cuBLAS 默认要求标量参数以指针形式传入。

`cublasSgemm` 通常是异步提交。CPU 调用返回时，GPU 不一定已经完成矩阵乘法。因此预热循环结束后执行：

```cpp
cudaDeviceSynchronize()
```

它会阻塞 CPU，直到 GPU 上此前提交的工作全部完成。这样正式计时不会与预热重叠。

---

## 12. cuBLAS 正式计时

```cpp
        // cuBLAS timing
        checkCudaError(cudaEventRecord(start), "cudaEventRecord(start cublas) failed");
        for (int i = 0; i < repeat_time; i++) {
            checkCublasError(
                cublasSgemm(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            N, M, K,
                            &alpha,
                            d_B, N,
                            d_A, K,
                            &beta,
                            d_C, N),
                "cublasSgemm failed");
        }
        checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop cublas) failed");
        checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize(stop cublas) failed");

        float cublas_time = 0.0f;
        checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                       "cudaEventElapsedTime cublas failed");
```

### 详细解释

```cpp
cudaEventRecord(start)
```

把起点事件记录到当前 CUDA stream 中。

它不是简单读取 CPU 当前时间，而是把一个事件插入 GPU 的执行队列。当 GPU 真正执行到该位置时，事件才完成。

随后循环提交 5 次：

```cpp
cublasSgemm(...)
```

这些操作进入同一个默认 stream，因此会按顺序执行。

```cpp
cudaEventRecord(stop)
```

把终点事件排在 5 次 GEMM 后面。

于是 GPU 队列中的顺序是：

```text
start event
→ GEMM 1
→ GEMM 2
→ GEMM 3
→ GEMM 4
→ GEMM 5
→ stop event
```

```cpp
cudaEventSynchronize(stop)
```

阻塞 CPU，直到 GPU 执行到 `stop` 事件。没有这一步就立即读取时间，终点可能还没完成。

```cpp
cudaEventElapsedTime(&cublas_time, start, stop)
```

计算两个事件之间的时间，并写入：

```cpp
cublas_time
```

单位是毫秒。

这里测得的是 5 次 cuBLAS GEMM 的总时间，不是单次时间。

---

## 13. 保存 cuBLAS 结果并清空输出显存

```cpp
        checkCudaError(cudaMemcpy(C_cublas, d_C, sizeC, cudaMemcpyDeviceToHost),
                       "cudaMemcpy C_cublas failed");

        checkCudaError(cudaMemset(d_C, 0, sizeC), "cudaMemset d_C failed");
```

### 详细解释

```cpp
cudaMemcpy(C_cublas, d_C, sizeC, cudaMemcpyDeviceToHost)
```

把 GPU 上的 cuBLAS 结果复制到 CPU 数组 `C_cublas`。

四个参数分别是：

1. CPU 目标地址 `C_cublas`；
2. GPU 源地址 `d_C`；
3. 字节数 `sizeC`；
4. 方向 `cudaMemcpyDeviceToHost`。

复制后，CPU 才能使用：

```cpp
C_cublas[i]
```

逐元素检查结果。

```cpp
cudaMemset(d_C, 0, sizeC)
```

把 `d_C` 的全部字节设为 0。对于 IEEE 754 单精度浮点数，全零位模式就是 `0.0f`。

当前 `beta = 0`，理论上自定义 GEMM 应完全覆盖 C，因此清零不是数学上的必要条件。但它能让测试初始状态明确，也能暴露错误依赖旧 C 值的 kernel。

---

## 14. 自定义 GEMM 预热

```cpp
        // my gemm warmup
        for (int i = 0; i < warm_time; i++) {
            launch_gemm(M, N, K, alpha, d_A, d_B, beta, d_C);
        }
        checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after my gemm warmup failed");
```

### 详细解释

```cpp
launch_gemm(M, N, K, alpha, d_A, d_B, beta, d_C);
```

是项目封装的 GEMM 启动接口。参数含义为：

- `M`：A 和 C 的行数；
- `N`：B 和 C 的列数；
- `K`：A 的列数、B 的行数；
- `alpha`：矩阵乘积缩放系数；
- `d_A`：GPU 输入矩阵 A；
- `d_B`：GPU 输入矩阵 B；
- `beta`：旧 C 的缩放系数；
- `d_C`：GPU 输出矩阵 C。

`launch_gemm` 通常是运行在 CPU 上的普通函数。它内部可能执行类似：

```cpp
gemm_kernel<<<grid, block, shared_memory>>>(...);
```

调用关系可以理解为：

```text
test_perf_gemm.cpp
    ↓
launch_gemm
    ↓
计算 grid/block
    ↓
启动真正的 CUDA kernel
```

预热 10 次后调用：

```cpp
cudaDeviceSynchronize()
```

确保预热 kernel 全部完成，再开始正式计时。

---

## 15. 自定义 GEMM 正式计时

```cpp
        // my gemm timing
        checkCudaError(cudaEventRecord(start), "cudaEventRecord(start my_gemm) failed");
        for (int i = 0; i < repeat_time; i++) {
            launch_gemm(M, N, K, alpha, d_A, d_B, beta, d_C);
        }
        checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop my_gemm) failed");
        checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize(stop my_gemm) failed");
        checkCudaError(cudaGetLastError(), "cudaGetLastError failed");

        float my_time = 0.0f;
        checkCudaError(cudaEventElapsedTime(&my_time, start, stop),
                       "cudaEventElapsedTime my_gemm failed");
```

### 详细解释

这里复用了前面创建的 `start` 和 `stop`。CUDA Event 可以多次重新记录，不需要为 cuBLAS 和自定义 GEMM 各创建一套。

执行顺序是：

```text
记录 start
→ 启动 5 次自定义 GEMM
→ 记录 stop
→ 等待 stop 完成
→ 读取 elapsed time
```

所以：

```cpp
my_time
```

是 5 次自定义 GEMM 的总 GPU 时间，单位也是毫秒。

```cpp
cudaGetLastError()
```

读取并清除最近一次 CUDA Runtime 错误状态。

CUDA kernel 启动语法本身没有普通函数那样的返回值：

```cpp
kernel<<<grid, block>>>(...)
```

因此常通过 `cudaGetLastError` 检查 launch 配置错误，例如：

- block 中线程数超限；
- 动态共享内存过大；
- 启动参数非法。

当前代码是在 `cudaEventSynchronize(stop)` 之后检查。同步已经能暴露部分异步执行错误。

更常见的调试写法是：

```cpp
launch_gemm(...);
checkCudaError(cudaGetLastError(), "kernel launch failed");
checkCudaError(cudaDeviceSynchronize(), "kernel execution failed");
```

这样更容易区分“启动失败”和“执行期间非法访存”。

---

## 16. 复制自定义 GEMM 结果

```cpp
        checkCudaError(cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost),
                       "cudaMemcpy C failed");
```

### 详细解释

自定义 GEMM 的输出位于 GPU 显存 `d_C` 中。

该语句把它复制到 CPU 数组：

```cpp
C
```

之后 CPU 才能逐元素执行：

```cpp
C[i]
```

并与 `C_cublas[i]` 比较。

---

## 17. 数值正确性检查

```cpp
        int error_count = 0;
        for (int i = 0; i < M * N && error_count < 10; ++i) {
            if (fabsf(C_cublas[i] - C[i]) > 1e-5f) {
                error_count++;
            }
        }
```

### 详细解释

```cpp
int error_count = 0;
```

用于统计发现了多少个错误元素。

循环条件是：

```cpp
i < M * N && error_count < 10
```

`&&` 表示逻辑与。只有以下两个条件都成立时才继续：

1. 还没遍历完 C；
2. 错误数量还不到 10。

所以一旦发现 10 个错误，检查就提前停止。这避免在结果完全错误时仍扫描超大矩阵。

```cpp
fabsf(C_cublas[i] - C[i])
```

先计算两个结果之差，再取 `float` 绝对值。

```cpp
> 1e-5f
```

表示绝对误差大于 `0.00001` 时认为结果错误。

浮点结果通常不直接使用：

```cpp
C_cublas[i] == C[i]
```

因为不同累加顺序、FMA、编译优化和 cuBLAS 算法都可能带来微小数值差异。

当前只使用绝对误差。更通用的判断通常同时使用绝对误差和相对误差：

```text
|actual-reference| <= atol + rtol × |reference|
```

不过本测试输入全是 1 和 2，理论结果 `2K` 很简单，当前容差可能足够。

需要注意：这里的 `error_count` 最大只会到 10。因此它表示“最多找到 10 个错误”，不是完整错误总数。

---

## 18. 计算 GFLOPS 与性能比例

```cpp
        float cublas_gflops =
            repeat_time * 2.0f * M * N * K / (cublas_time * 1e6f);
        float my_gflops =
            repeat_time * 2.0f * M * N * K / (my_time * 1e6f);
        float ratio = my_gflops / cublas_gflops;
```

### 详细解释

一次 GEMM：

```text
C(M×N) = A(M×K) × B(K×N)
```

C 一共有 `M×N` 个元素。

每个输出元素要进行 K 次乘法和近似 K 次加法。性能统计通常把一次乘加计为 2 个浮点操作，因此总操作数近似为：

```text
2 × M × N × K
```

正式计时执行了 `repeat_time` 次，所以总操作数是：

```text
repeat_time × 2 × M × N × K
```

CUDA Event 返回的时间单位是毫秒。设总操作数为 `F`，时间为 `t_ms`：

```text
GFLOPS
= F / (t_ms / 1000) / 10^9
= F / (t_ms × 10^6)
```

所以代码除以：

```cpp
time * 1e6f
```

是正确的。

```cpp
float ratio = my_gflops / cublas_gflops;
```

计算自定义实现达到 cuBLAS 的比例。

例如：

```text
cuBLAS = 10000 GFLOPS
my GEMM = 8000 GFLOPS
ratio = 0.8
```

表示自定义 GEMM 达到 cuBLAS 的 80%。

表达式中先出现：

```cpp
repeat_time * 2.0f
```

因此后续乘法会转换成浮点运算，避免 `M*N*K` 完整地以 32 位 `int` 计算。但性能统计使用 `double` 会更稳妥，例如：

```cpp
double flops = 2.0 * static_cast<double>(M) * N * K;
```

---

## 19. 控制台与 CSV 输出

```cpp
        // 控制台输出
        std::cout << "error_count = " << error_count << std::endl;
        std::cout << "cublas_gflops = " << cublas_gflops
                  << " my_gflops = " << my_gflops << std::endl;
        std::cout << "ratio = " << ratio << std::endl;

        // CSV输出
        csv << sz << ","
            << error_count << ","
            << cublas_gflops << ","
            << my_gflops << ","
            << ratio << "\n";

        cnt++;
        sum += ratio;
```

### 详细解释

控制台输出当前尺寸的三类信息：

- `error_count`：是否检测到数值错误；
- `cublas_gflops` 和 `my_gflops`：两种实现的性能；
- `ratio`：自定义实现相对 cuBLAS 的比例。

C++ 允许把一个很长的流表达式分成多行，只要最终有分号即可：

```cpp
std::cout << ...
          << ...
          << ...;
```

CSV 每一行的格式是：

```text
size,error_count,cublas_gflops,my_gflops,ratio
```

例如：

```text
1024,0,15000.2,12000.5,0.80002
```

```cpp
cnt++;
```

让已测试尺寸数量加 1。

```cpp
sum += ratio;
```

等价于：

```cpp
sum = sum + ratio;
```

把当前性能比例加入总和。

---

## 20. 释放本轮资源

```cpp
        checkCublasError(cublasDestroy(handle), "cublasDestroy failed");
        checkCudaError(cudaEventDestroy(start), "cudaEventDestroy(start) failed");
        checkCudaError(cudaEventDestroy(stop), "cudaEventDestroy(stop) failed");
        checkCudaError(cudaFree(d_A), "cudaFree d_A failed");
        checkCudaError(cudaFree(d_B), "cudaFree d_B failed");
        checkCudaError(cudaFree(d_C), "cudaFree d_C failed");

        free(A);
        free(B);
        free(C_cublas);
        free(C);

        checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize at loop end failed");
    }
```

### 详细解释

```cpp
cublasDestroy(handle)
```

释放 `cublasCreate` 创建的 cuBLAS 内部资源。

当前代码每个尺寸都重新创建和销毁句柄。这样逻辑清晰，但会增加整个测试程序的非计时开销。也可以在尺寸循环外只创建一次句柄。

```cpp
cudaEventDestroy(start)
cudaEventDestroy(stop)
```

释放两个 CUDA Event。

```cpp
cudaFree(d_A)
cudaFree(d_B)
cudaFree(d_C)
```

释放 `cudaMalloc` 分配的 GPU 显存。

CPU 和 GPU 内存必须使用匹配的释放方式：

```text
malloc      → free
new         → delete
new[]       → delete[]
cudaMalloc  → cudaFree
```

所以 CPU 数组使用：

```cpp
free(A);
```

不能用 `cudaFree`；GPU 指针也不能用普通 `free`。

循环末尾再次执行：

```cpp
cudaDeviceSynchronize()
```

确保当前尺寸遗留的 GPU 工作全部结束，并检查是否还有未暴露的异步错误。

最后的 `}` 结束范围 `for` 循环，程序开始处理汇总结果。

---

## 21. 汇总平均比例并关闭文件

```cpp
    float mean_ratio = sum / static_cast<float>(cnt);
    std::cout << "mean_ratio = " << mean_ratio << std::endl;

    // 把汇总结果也写进csv
    csv << "mean_ratio,,,,"
        << mean_ratio << "\n";

    csv.close();
    return 0;
}
```

### 详细解释

```cpp
float mean_ratio = sum / static_cast<float>(cnt);
```

把所有尺寸的 `ratio` 相加后除以测试数量，得到算术平均值。

```cpp
static_cast<float>(cnt)
```

显式把整数 `cnt` 转成 `float`，明确进行浮点除法。

该平均值让每个矩阵尺寸拥有相同权重。因此 `100×100×100` 和 `8192×8192×8192` 对最终均值的贡献相同。

它不是：

- 按总计算量加权的平均；
- 总 FLOP 除以总时间；
- 只针对大尺寸的稳定性能。

```cpp
csv << "mean_ratio,,,," << mean_ratio << "\n";
```

四个逗号让数值落在 CSV 的第五列，也就是 `ratio` 列。输出可能是：

```text
mean_ratio,,,,0.75
```

```cpp
csv.close();
```

关闭文件，并刷新尚在缓冲区中的内容。即使不显式调用，对象析构时也会自动关闭，但显式写出更直观。

```cpp
return 0;
```

从 `main` 返回 0，表示程序正常结束。

---

## 22. 完整执行时间线

```text
创建 results 目录
        ↓
打开 CSV 文件并写入表头
        ↓
生成对齐和非对齐测试尺寸
        ↓
遍历一个矩阵尺寸
        ↓
计算 A、B、C 的元素数和字节数
        ↓
分配 CPU 内存
        ↓
分配 GPU 显存
        ↓
初始化 A=1、B=2
        ↓
将 A、B 从 CPU 复制到 GPU
        ↓
创建 cuBLAS 句柄和 CUDA Event
        ↓
cuBLAS 预热 10 次
        ↓
cuBLAS 执行 5 次并用 Event 计时
        ↓
将 cuBLAS 结果复制回 CPU
        ↓
清零 d_C
        ↓
自定义 GEMM 预热 10 次
        ↓
自定义 GEMM 执行 5 次并用 Event 计时
        ↓
将自定义结果复制回 CPU
        ↓
逐元素比较两个结果
        ↓
计算 cuBLAS GFLOPS、自定义 GFLOPS 和 ratio
        ↓
控制台输出并写入 CSV
        ↓
释放本轮 CPU/GPU/cuBLAS/Event 资源
        ↓
测试下一个尺寸
        ↓
全部尺寸结束后计算 mean_ratio
        ↓
关闭 CSV 文件并正常退出
```


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-gemm算子|模块-gemm算子]]

%% 项目关联导航：结束 %%
