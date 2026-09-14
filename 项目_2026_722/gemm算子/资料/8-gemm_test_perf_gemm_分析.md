# `test_perf_gemm.cpp` 源码整体分析

## 1. 文件定位：它不是 GEMM 核函数，而是项目的“测试与性能评测入口”

`test_perf_gemm.cpp` 位于项目的 `tests/` 目录中，是一个运行在 **CPU（Host）端** 的测试程序。它本身不负责实现矩阵乘法的 CUDA 计算逻辑，而是负责组织一次完整的 GEMM 测试流程：

1. 构造不同规模的输入矩阵；
2. 在 CPU 内存和 GPU 显存中分配空间；
3. 分别调用 NVIDIA cuBLAS 和项目自研的 `launch_gemm`；
4. 对比两者计算结果，检查正确性；
5. 使用 CUDA Event 测量 GPU 执行时间；
6. 计算两种实现的 GFLOPS 和性能比值；
7. 将结果输出到终端和 CSV 文件。

因此，这个文件在项目中的角色可以概括为：

> **它是连接“自研 GEMM 实现”和“可验证实验结果”的桥梁。**

没有这个文件，项目虽然可能已经写出了 CUDA Kernel，但无法系统回答以下问题：

- 自研 Kernel 算得对不对？
- 在不同矩阵规模下性能怎么样？
- 与成熟的 cuBLAS 相比达到多少比例？
- 对规则尺寸和非对齐尺寸是否都能工作？
- 多组结果能否保存下来继续画图和分析？

从工程视角看，`test_perf_gemm.cpp` 是项目的 **benchmark driver（基准测试驱动程序）**，而不是核心算子实现文件。

---

## 2. 它在整个项目调用链中的位置

这个项目的主要调用关系可以理解为：

```text
CMakeLists.txt
    │
    ├── 编译 src/launcher.cu
    ├── 编译 src/kernels/gemm_warp_tile.cu
    ├── 生成静态库 gemm_lib
    │
    └── 编译 tests/test_perf_gemm.cpp
            │
            └── 链接 gemm_lib、CUDA Runtime、cuBLAS

运行 ./build/test_perf_gemm
    │
    ├── 调用 cublasSgemm()        ← NVIDIA 官方参考实现
    │
    └── 调用 launch_gemm()        ← 项目统一对外接口
             │
             └── launcher.cu 选择算法
                      │
                      └── launch_gemm_warp_tile()
                               │
                               └── 启动自研 CUDA GEMM Kernel
```

这个文件只依赖：

```cpp
#include "gemm/launcher.h"
```

它并不直接包含或调用具体的 `__global__` 核函数，而是通过 `launch_gemm(...)` 使用项目暴露出的统一接口。这种结构有重要意义：

- 测试代码不需要知道内部 Kernel 的线程块配置；
- 后续可以在 `launcher.cu` 中增加更多 GEMM 算法；
- 测试程序不必随着 Kernel 内部实现频繁修改；
- “接口层、调度层、Kernel 层、测试层”得到解耦。

因此，阅读项目时不能把这个文件当作 GEMM 算法本身。它回答的是“如何测试 GEMM”，而 `gemm_warp_tile.cu` 才回答“GEMM 在 GPU 上具体如何计算”。

---

## 3. 文件完成的整体工作流程

对于每一个测试尺寸 `sz`，程序都令：

```text
M = N = K = sz
```

即测试方阵乘法：

```text
A[M, K] × B[K, N] = C[M, N]
```

整体流程如下：

```text
生成测试尺寸
    ↓
为 A、B、C 分配 CPU 内存
    ↓
为 d_A、d_B、d_C 分配 GPU 显存
    ↓
初始化 A=1、B=2
    ↓
把 A、B 从 Host 复制到 Device
    ↓
cuBLAS 预热 10 次
    ↓
cuBLAS 正式计时 5 次
    ↓
将 cuBLAS 结果复制回 CPU
    ↓
清空 d_C
    ↓
自研 GEMM 预热 10 次
    ↓
自研 GEMM 正式计时 5 次
    ↓
将自研结果复制回 CPU
    ↓
比较两个结果
    ↓
计算 cuBLAS GFLOPS、自研 GFLOPS 和性能比例
    ↓
写入 CSV
    ↓
释放本轮 CPU/GPU 资源
```

这是一套比较标准的 CUDA 算子测试框架，包含了算子开发最关键的两个评价维度：

- **Correctness：正确性**
- **Performance：性能**

只有性能高但结果错误的 Kernel 没有意义；只有结果正确但性能很低，也不能称为有效优化。

---

## 4. 测试尺寸的设计意义

程序测试了两类尺寸。

### 4.1 规则尺寸

```text
256, 512, 768, ... , 8192
```

这些尺寸都是 256 的倍数，通常更容易满足 GEMM Kernel 的 Tile 对齐要求。例如，如果一个线程块计算 `128×256` 的输出 Tile，那么规则尺寸可以被 Tile 较好地整除，边界判断较少，GPU 利用率通常也更高。

这类尺寸主要用于观察 Kernel 在理想条件下的吞吐性能。

### 4.2 非对齐尺寸

```text
100, 257, 511, 1000, 1234, 2047
```

这些尺寸不能稳定地被常见的 Block Tile、Warp Tile 或向量化宽度整除。它们会触发 Kernel 的边界处理逻辑，也可能进入项目中的 irregular 配置或其他调度分支。

非对齐尺寸主要检查：

- Kernel 是否发生越界访问；
- 边缘 Tile 是否正确处理；
- Fast Path 之外的路径是否可用；
- 非规则尺寸下性能会下降多少；
- `launcher` 的算法选择是否合理。

因此，这组尺寸不是随意添加的。它让测试从“只会跑理想尺寸”升级为“能够检验工程鲁棒性”。

---

## 5. CPU 内存与 GPU 显存的角色

程序中存在两组指针。

### CPU 端指针

```cpp
float* A;
float* B;
float* C_cublas;
float* C;
```

它们通过 `malloc` 分配，存放在 Host 内存中：

- `A`、`B`：构造输入数据；
- `C_cublas`：保存 cuBLAS 的参考结果；
- `C`：保存自研 GEMM 的结果。

### GPU 端指针

```cpp
float *d_A, *d_B, *d_C;
```

它们通过 `cudaMalloc` 分配，存放在 Device 显存中。真正的 cuBLAS 和自研 GEMM 都读取这些设备指针。

数据流向是：

```text
CPU: A、B
   │ cudaMemcpyHostToDevice
   ▼
GPU: d_A、d_B
   │ cuBLAS / 自研 Kernel
   ▼
GPU: d_C
   │ cudaMemcpyDeviceToHost
   ▼
CPU: C_cublas 或 C
```

这部分体现了 CUDA 编程的基本模型：CPU 负责准备任务和管理资源，GPU 负责执行大规模并行计算。

需要注意，大尺寸测试会消耗较多内存。以 `8192×8192` 的 FP32 矩阵为例，单个矩阵约为 256 MiB：

- CPU 同时存在 4 个矩阵，约需 1 GiB；
- GPU 同时存在 3 个矩阵，约需 768 MiB。

因此，该测试程序不仅要求 GPU 能运行 Kernel，也要求系统内存和显存足够。

---

## 6. 为什么要使用 cuBLAS 作为参考实现

`cublasSgemm` 是 NVIDIA cuBLAS 提供的单精度矩阵乘法接口，其中：

- `S` 表示 single precision，即 FP32；
- `gemm` 表示 General Matrix Multiplication。

其数学形式为：

```text
C = α × A × B + β × C
```

本文件设置：

```text
α = 1
β = 0
```

所以实际计算退化为：

```text
C = A × B
```

cuBLAS 是 NVIDIA 长期高度优化的数学库，通常接近对应 GPU 和数据类型下的高性能水平。因此它在这个文件中承担两项职责：

1. **正确性基准**：用其结果作为参考答案；
2. **性能基准**：用其 GFLOPS 作为自研 Kernel 的对照上限。

项目最终关注的不只是自研 Kernel 的绝对 GFLOPS，还关注：

```text
ratio = my_gflops / cublas_gflops
```

例如：

```text
ratio = 0.75
```

表示自研实现的吞吐量约达到同次测试中 cuBLAS 的 75%。这种相对指标可以降低不同 GPU、频率和运行环境对绝对性能数字的影响。

---

## 7. cuBLAS 参数看起来“把 A、B 和 M、N 交换了”的原因

程序调用形式为：

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

项目中的矩阵按照 C/C++ 常见的 **Row-major（行主序）** 存储，而传统 cuBLAS 接口按照 **Column-major（列主序）** 解释矩阵。

为避免真正转置整个矩阵，代码利用：

```text
(A × B)^T = B^T × A^T
```

把输入顺序写成 `d_B, d_A`，同时交换 `M` 和 `N`。这样，cuBLAS 按列主序得到的内存结果，恰好可以被程序按行主序理解为期望的 `C=A×B`。

所以这里不是把矩阵乘法写反了，而是一种常见的 **Row-major 适配技巧**。

理解这个调用是阅读该文件最关键的知识点之一。

---

## 8. `launch_gemm` 在这里代表什么

自研 GEMM 通过下面的接口调用：

```cpp
launch_gemm(M, N, K, alpha, d_A, d_B, beta, d_C);
```

这个函数是项目对外暴露的统一入口，其接口表达的也是：

```text
C = alpha × A × B + beta × C
```

当前调用没有显式传入 `GemmAlgo`，因此使用头文件中的默认参数：

```text
GemmAlgo::Auto
```

随后由 `src/launcher.cu` 决定实际采用哪一种实现。在当前仓库中，最终会调度到 Warp Tile GEMM。

从测试文件角度看，它不关心下面这些 Kernel 细节：

- 一个 Block 有多少线程；
- Block Tile 是多大；
- 每个 Warp 计算哪一块；
- 是否使用 Shared Memory；
- 是否采用 `float4`；
- 是否使用双缓冲或 `cp.async`。

这些都被隐藏在 `launch_gemm` 之后。这体现了良好的算子接口设计：调用者只提供问题规模、数据指针和系数，内部负责选择和启动 Kernel。

---

## 9. 为什么正式计时前要预热

cuBLAS 和自研 GEMM 都先执行 10 次 warmup，再正式计时 5 次。

预热的主要作用包括：

- 让 CUDA Context 完成初始化；
- 让动态库、运行时和相关资源进入稳定状态；
- 减弱首次 Kernel Launch 的额外开销；
- 让 GPU 频率和缓存状态更接近稳定运行；
- 避免把一次性初始化时间错误地算进 Kernel 性能。

如果直接测第一次执行，得到的时间往往不能代表稳定性能。

预热结束后使用：

```cpp
cudaDeviceSynchronize();
```

确保所有预热任务都已经完成，正式计时不会和预热执行重叠。

---

## 10. CUDA Event 为什么适合测 GPU Kernel 时间

CUDA Kernel Launch 通常是异步的。CPU 发出调用后可能立即继续执行，而 GPU 仍在后台计算。因此不能简单使用普通 CPU 计时器包住函数调用。

该文件使用：

```text
cudaEventRecord(start)
执行 5 次 GEMM
cudaEventRecord(stop)
cudaEventSynchronize(stop)
cudaEventElapsedTime(...)
```

CUDA Event 被记录在 GPU 执行流中，能够测量同一 CUDA Stream 上两事件之间的 GPU 时间。

`cudaEventElapsedTime` 返回的是所有重复执行的总毫秒数，因此后面的 GFLOPS 公式把 `repeat_time` 也放入总计算量中，而不是先求单次平均时间。

这里默认 cuBLAS 和 `launch_gemm` 都在默认 Stream 上运行，因此开始、计算、停止事件具有正确的先后顺序。

---

## 11. GFLOPS 的计算逻辑

一个 `M×K` 与 `K×N` 的矩阵乘法大约需要：

```text
M × N × K 次乘法
M × N × K 次加法
```

所以通常将 GEMM 浮点运算量近似记为：

```text
2 × M × N × K FLOPs
```

程序重复执行 `repeat_time` 次，因此总运算量为：

```text
repeat_time × 2 × M × N × K
```

CUDA Event 返回毫秒，GFLOPS 表示每秒十亿次浮点运算。换算后：

```text
GFLOPS = repeat_time × 2MNK / (time_ms × 10^6)
```

代码中的 `1e6` 来源于：

```text
毫秒转秒：× 10^3
FLOP 转 GFLOP：÷ 10^9
合并后分母为 10^6
```

最终计算：

```text
cublas_gflops
my_gflops
ratio = my_gflops / cublas_gflops
```

其中 `ratio` 是这个测试文件最重要的性能指标。

---

## 12. 正确性检查的含义与局限

程序把 cuBLAS 结果和自研 GEMM 结果逐元素比较：

```text
|C_cublas[i] - C[i]| > 1e-5
```

若差值超过阈值，则计为一个错误元素。为了避免错误很多时打印或遍历统计过多，`error_count` 达到 10 后就停止继续统计。

因此这里的 `error_count` 应解释为：

- `0`：在当前比较规则下没有发现错误；
- `1～9`：发现对应数量的前部错误；
- `10`：至少存在 10 个错误，而不是“恰好只有 10 个”。

当前输入设置为：

```text
A 的所有元素 = 1
B 的所有元素 = 2
```

理论上每个输出元素都等于：

```text
C[m,n] = K × 2
```

这让结果非常容易理解，也适合初次调试。但它的验证强度有限，因为所有输出元素都相同，一些索引错误、转置错误或数据布局错误可能不容易暴露。

更严格的工程测试通常会进一步加入：

- 随机输入；
- 正数、负数和零；
- 不同的 `alpha`、`beta`；
- 相对误差和绝对误差组合；
- 最大误差、平均误差统计；
- NaN 和 Inf 检查；
- 多种非方阵 `M、N、K` 组合。

因此，该文件已经具备基础正确性测试能力，但不是完整的数值验证框架。

---

## 13. CSV 输出在项目中的意义

程序把每个尺寸的结果写入：

```text
./results/gemm_perf_results.csv
```

字段包括：

```text
size,error_count,cublas_gflops,my_gflops,ratio
```

最后还写入所有尺寸的平均 `ratio`。

这使一次命令行运行产生的结果可以继续用于：

- 画性能曲线；
- 比较不同 Kernel 版本；
- 分析规则尺寸和非规则尺寸；
- 计算多次运行的平均值和标准差；
- 保存项目优化过程中的 baseline；
- 为 README、实验报告和简历提供量化数据。

项目中的 `tools/generate_stable_baseline.py` 和 `results/` 目录正是围绕这些数据继续处理。因此这个测试程序还是整个性能实验流水线的数据源。

---

## 14. 错误检查函数的工程作用

文件封装了两类错误检查：

```text
checkCudaError(...)
checkCublasError(...)
```

它们分别检查：

- CUDA Runtime API 是否成功；
- cuBLAS API 是否成功。

一旦出现错误，程序打印对应阶段和错误码，然后立即退出。这样能够快速定位以下问题：

- 显存不足；
- Host/Device 数据复制失败；
- CUDA Event 创建或计时失败；
- cuBLAS Handle 创建失败；
- cuBLAS GEMM 调用失败；
- Kernel 启动参数非法；
- Kernel 中发生越界访问等异步错误。

`cudaGetLastError()` 放在自研 Kernel 计时结束后，用于检查最近一次 CUDA 调用或 Kernel Launch 的错误。

不过从更严格的调试角度看，若需要精确定位是哪一次 `launch_gemm` 出错，可以在开发阶段每次调用后检查错误，或者使用 `compute-sanitizer`。当前写法更偏向性能测试，避免在每次 Kernel Launch 后增加额外同步开销。

---

## 15. 资源生命周期

每个尺寸测试结束后，程序依次销毁：

- cuBLAS Handle；
- CUDA Event；
- GPU 显存；
- CPU 内存。

这体现了 CUDA/C++ 中常见的资源生命周期：

```text
创建/分配 → 使用 → 销毁/释放
```

具体对应关系为：

```text
cublasCreate   ↔ cublasDestroy
cudaEventCreate ↔ cudaEventDestroy
cudaMalloc      ↔ cudaFree
malloc          ↔ free
ofstream 打开    ↔ close
```

该文件选择在每一个尺寸循环中重新创建和销毁资源，结构直观、不同尺寸相互隔离；但也会带来额外开销。由于创建、拷贝和销毁阶段没有包含在 GEMM Event 计时区间内，所以不会直接污染所报告的 Kernel GFLOPS。

如果以后追求测试程序本身的运行效率，可以把 cuBLAS Handle 和 Event 移到循环外复用，但这不是理解当前项目的核心。

---

## 16. 这个文件“测了什么”与“没有测什么”

### 已经测量

- 方阵 GEMM 的正确性；
- 多种规则和非规则尺寸；
- FP32 GEMM 吞吐量；
- cuBLAS 与自研实现的相对性能；
- 多次重复后的总执行时间；
- 所有测试尺寸的平均性能比例。

### 尚未覆盖

- 非方阵，如 `M≠N≠K`；
- 不同 batch size 的 Batched GEMM；
- FP16、BF16、TF32、FP8；
- Tensor Core 路径；
- 不同 `alpha`、`beta`；
- 转置矩阵输入；
- 多 Stream 并发；
- Kernel Launch latency 单独分析；
- 显存带宽、Occupancy、指令吞吐等底层指标；
- Nsight Compute 的 Kernel 级性能瓶颈分析；
- 大量随机输入下的数值稳定性。

因此，它是一个清晰有效的主 benchmark，但不是覆盖所有 GEMM 场景的完整测试套件。

---

## 17. 阅读这个文件时真正需要抓住的核心

不必一开始纠结所有 C++ 语法细节。阅读它时应先建立以下整体认识：

1. **这是 Host 端程序，不是 Device Kernel。**
2. **cuBLAS 是参考答案和性能基线。**
3. **`launch_gemm` 是进入项目自研 Kernel 的统一入口。**
4. **预热用于消除首次执行的非稳定开销。**
5. **CUDA Event 用于正确测量异步 GPU 执行时间。**
6. **`2MNK` 是 GEMM 的近似浮点运算量。**
7. **`ratio` 表示自研实现达到 cuBLAS 性能的比例。**
8. **规则尺寸测理想性能，非对齐尺寸测边界处理和鲁棒性。**
9. **CSV 将单次运行连接到后续实验统计和绘图。**
10. **这个文件负责“证明 Kernel”，而不是“实现 Kernel”。**

只要先掌握这十点，就已经能够从全局上看懂该文件。

---

## 18. 建议的后续源码阅读顺序

理解 `test_perf_gemm.cpp` 后，建议沿它的调用链继续阅读：

```text
1. include/gemm/launcher.h
   ↓ 看清自研 GEMM 对外接口

2. include/gemm/types.h
   ↓ 理解 GemmAlgo 枚举

3. src/launcher.cu
   ↓ 理解 Auto 如何选择具体算法

4. include/gemm/kernels/gemm_warp_tile_config.h
   ↓ 理解 Block Tile、Warp Tile、线程数等配置

5. include/gemm/kernels/gemm_warp_tile.h
   ↓ 理解 Kernel 启动函数声明

6. src/kernels/gemm_warp_tile.cu
   ↓ 最后进入真正的 CUDA GEMM 计算实现
```

这个顺序相当于从：

```text
怎么测试
→ 调用什么接口
→ 如何调度
→ 使用什么配置
→ GPU 上到底怎么计算
```

逐层深入，比一开始直接进入复杂 Kernel 更容易建立完整认知。

---

## 19. 一句话总结

`test_perf_gemm.cpp` 是该 GEMM 项目的主测试与性能评测程序：它在 Host 端准备矩阵和 GPU 资源，以 cuBLAS 为正确性与性能基准，通过 `launch_gemm` 调用自研 CUDA 实现，利用 CUDA Event 统计 GFLOPS，并将不同尺寸下的正确性和性能比例保存为 CSV，为后续 Kernel 优化、版本对比和实验报告提供量化依据。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-gemm算子|模块-gemm算子]]

%% 项目关联导航：结束 %%
