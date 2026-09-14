# 1. `CMakeLists.txt` 的整体概览和该文件的意义

## 1.1 这个文件负责什么

`CMakeLists.txt` 是该 GEMM 项目的构建入口。它不负责实现矩阵乘法，而是告诉 CMake：

- 项目使用 C++ 和 CUDA；
- C++/CUDA 采用 C++17；
- CUDA 代码面向哪一种 GPU 架构编译；
- 哪些源码组成 GEMM 静态库；
- 哪个文件编译成性能测试程序；
- 头文件目录在哪里；
- 需要链接 CUDA Runtime 和 cuBLAS；
- CUDA Kernel 与测试程序使用哪些优化参数。

可以把它理解为项目的“编译与链接施工图”。

源码描述“程序做什么”，`CMakeLists.txt` 描述“这些源码怎样组合成程序”。

---

## 1.2 它在整个项目中的位置

从当前文件可以推断出项目的大致结构：

```text
gemm_project/
├── CMakeLists.txt
├── include/
│   └── GEMM 对外接口头文件
├── src/
│   ├── launcher.cu
│   └── kernels/
│       └── gemm_warp_tile.cu
└── tests/
    └── test_perf_gemm.cpp
```

对应的逻辑分层大致是：

```text
tests/test_perf_gemm.cpp
        │
        │ 调用公开 GEMM 接口
        ▼
src/launcher.cu
        │
        │ 配置 Grid、Block 并启动 Kernel
        ▼
src/kernels/gemm_warp_tile.cu
        │
        │ GPU 上执行真正的 GEMM
        ▼
      输出矩阵
```

构建关系为：

```text
launcher.cu
gemm_warp_tile.cu
        │
        ▼
  libgemm_lib.a
        │
        ├──────── CUDA::cudart
        └──────── CUDA::cublas
        │
        ▼
test_perf_gemm 可执行程序
```

最终通常会生成：

```text
libgemm_lib.a
test_perf_gemm
```

其中：

- `libgemm_lib.a`：封装自定义 GEMM 实现的静态库；
- `test_perf_gemm`：用于正确性验证和性能测试的可执行程序。

---

## 1.3 从该文件可以看出的项目设计

该项目没有把 Kernel、Host 启动逻辑和测试代码全部塞入一个 `.cu` 文件，而是进行了分层：

### Kernel 实现层

```text
src/kernels/gemm_warp_tile.cu
```

从命名推测，这里实现以 Warp Tile 为核心的 GEMM Kernel，可能涉及：

- Block Tile；
- Warp Tile；
- Thread Tile；
- Shared Memory；
- 寄存器累加；
- 全局内存读取和结果写回。

### Host 启动与封装层

```text
src/launcher.cu
```

可能负责：

- 对外暴露统一 GEMM 函数；
- 检查矩阵尺寸和参数；
- 计算 Grid 和 Block；
- 启动 `gemm_warp_tile` Kernel；
- 检查 CUDA 错误；
- 根据输入形状选择不同 Kernel。

### 测试与 Benchmark 层

```text
tests/test_perf_gemm.cpp
```

可能负责：

- 创建 A、B、C 矩阵；
- 调用自定义 GEMM；
- 调用 cuBLAS 作为参考；
- 验证误差；
- 使用 CUDA Event 计时；
- 计算 GFLOPS/TFLOPS；
- 输出自定义 Kernel 与 cuBLAS 的性能差距。

具体行为还要结合对应源码确认，但这些文件在构建图中的位置已经由当前 `CMakeLists.txt` 明确给出。

---

## 1.4 该文件为什么重要

没有这个文件时，需要手动完成：

```text
1. 用 nvcc 编译 launcher.cu
2. 用 nvcc 编译 gemm_warp_tile.cu
3. 指定 C++17
4. 指定 sm_89
5. 添加 include 路径
6. 开启 -O3、--use_fast_math 和 -lineinfo
7. 生成静态库
8. 编译 test_perf_gemm.cpp
9. 链接自定义库
10. 链接 cudart 和 cublas
```

CMake 将这些步骤统一描述，使项目具备：

- 可重复构建；
- 增量编译；
- 依赖自动传播；
- 多文件组织；
- 编译参数集中管理；
- 后续添加更多 Kernel 和测试目标的能力。

因此，这个文件在整个项目中的作用不是“辅助文件”，而是连接源码、CUDA Toolkit、cuBLAS 和测试程序的工程入口。

# 2. 代码与构建逻辑详细分析

## 2.1 最低 CMake 版本

```cmake
cmake_minimum_required(VERSION 3.20)
```

表示构建项目至少需要 CMake 3.20。

如果版本过低，配置阶段会直接失败。可以使用：

```bash
cmake --version
```

查看当前版本。

这里指定较新的版本，主要是因为项目使用了：

- CUDA 作为 CMake 一等语言；
- `CMAKE_CUDA_ARCHITECTURES`；
- `CUDA::cudart`、`CUDA::cublas`；
- 按编译语言生效的生成器表达式。

---

## 2.2 定义项目并启用 C++、CUDA

```cmake
project(gemm_project LANGUAGES CXX CUDA)
```

### 项目名

项目名为：

```text
gemm_project
```

CMake 会据此设置：

```cmake
PROJECT_NAME
PROJECT_SOURCE_DIR
PROJECT_BINARY_DIR
```

### 启用两种语言

```cmake
LANGUAGES CXX CUDA
```

表示：

- `.cpp` 文件使用 C++ 编译器；
- `.cu` 文件使用 CUDA 编译链；
- 配置阶段需要找到主机 C++ 编译器；
- 配置阶段需要找到 CUDA 编译器；
- CMake 要理解 CUDA Kernel、设备代码和 CUDA 链接步骤。

这说明该项目属于 CPU 与 GPU 协作的异构工程：

```text
C++：Host 控制、测试和接口
CUDA C++：Kernel 与 Device 计算
```

---

## 2.3 指定 C++17 和 CUDA C++17

```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
```

### 普通 C++ 标准

```cmake
set(CMAKE_CXX_STANDARD 17)
```

作用于 `test_perf_gemm.cpp` 等普通 C++ 文件。

### CUDA C++ 标准

```cmake
set(CMAKE_CUDA_STANDARD 17)
```

作用于：

```text
src/launcher.cu
src/kernels/gemm_warp_tile.cu
```

CUDA 文件既包含 Device 代码，也可能包含大量 Host C++ 和模板，因此也需要明确语言标准。

### `STANDARD_REQUIRED ON`

表示不能在编译器不支持时自动降级为旧标准。

如果源码使用了 C++17 特性，而环境只支持旧标准，配置或编译应明确失败，而不是产生隐蔽兼容问题。

---

## 2.4 指定目标 GPU 架构

```cmake
set(CMAKE_CUDA_ARCHITECTURES 89)
```

这表示 CUDA 设备代码以计算能力 8.9 对应的目标架构进行编译。

底层通常会生成与以下含义接近的 `nvcc` 参数：

```bash
-gencode arch=compute_89,code=sm_89
```

### 这一配置决定什么

它影响：

- 生成哪种 GPU 机器码；
- 编译器可使用哪些架构指令；
- 最终二进制能在哪些 GPU 上直接执行；
- 某些架构相关优化是否可用。

### 固定为 89 的优点

- 目标清晰；
- 编译时间较短；
- 二进制体积较小；
- 适合固定实验设备；
- 便于做针对特定架构的 GEMM 优化。

### 固定为 89 的局限

- 换 GPU 时可能需要重新配置；
- 不适合作为多架构通用发布包；
- 目标 GPU 不兼容时，可能出现：

```text
no kernel image is available for execution on the device
```

### 更灵活的写法

```cmake
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
```

这样 89 是默认值，但允许构建时覆盖：

```bash
cmake -S . -B build   -DCMAKE_CUDA_ARCHITECTURES=<目标架构>
```

如果需要同时支持多个架构，也可以设置多个值：

```cmake
set(CMAKE_CUDA_ARCHITECTURES 80 86 89)
```

不过架构越多，编译时间和二进制体积通常也越大。

---

## 2.5 默认使用 Release 模式

```cmake
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
```

逻辑是：

```text
用户没有指定构建类型
    ↓
默认使用 Release
```

如果用户显式传入：

```bash
-DCMAKE_BUILD_TYPE=Debug
```

则不会被覆盖。

### 为什么 GEMM 项目默认 Release

GEMM 项目需要做性能测试。Debug 模式通常会：

- 关闭或削弱优化；
- 改变指令和寄存器使用；
- 影响 Kernel 性能；
- 使 Benchmark 失去代表性。

所以默认 `Release` 是合理的。

常见构建类型：

```text
Debug
Release
RelWithDebInfo
MinSizeRel
```

对 CUDA 性能分析而言，常见组合是：

```text
Release / -O3
+
-lineinfo
```

既保留优化，又能让 Nsight 将设备指令映射到源码行。

---

## 2.6 查找 CUDA Toolkit

```cmake
find_package(CUDAToolkit REQUIRED)
```

要求 CMake 找到已安装的 CUDA Toolkit。

`REQUIRED` 表示这是强依赖。如果找不到，配置直接失败。

### 它提供什么

这条命令会提供现代 CMake 的 CUDA imported targets，例如：

```cmake
CUDA::cudart
CUDA::cublas
```

因此后面不需要手写：

- CUDA 头文件路径；
- CUDA 库目录；
- `libcudart.so` 的绝对路径；
- `libcublas.so` 的绝对路径。

### 与启用 CUDA 语言的区别

```cmake
project(... LANGUAGES CUDA)
```

主要负责启用 CUDA 编译语言。

```cmake
find_package(CUDAToolkit REQUIRED)
```

主要负责查找 Toolkit 中的 Runtime、cuBLAS 等库目标。

两者职责相关，但不完全相同。

---

## 2.7 创建 `gemm_lib` 静态库

```cmake
add_library(gemm_lib STATIC
    src/launcher.cu
    src/kernels/gemm_warp_tile.cu
)
```

创建名为：

```text
gemm_lib
```

的静态库。

Linux 下通常生成：

```text
libgemm_lib.a
```

### 为什么做成库

这样可以将 GEMM 实现与测试程序分离：

```text
gemm_lib：被测试的算子实现
test_perf_gemm：调用者和 Benchmark
```

后续还可以新增：

```text
test_correctness_gemm
benchmark_shapes
benchmark_cublas
example_gemm
Python/C++ 扩展
```

这些目标都可以复用 `gemm_lib`，而不必重复列出所有 Kernel 源码。

### 两个源文件的编译关系

```text
launcher.cu
gemm_warp_tile.cu
        ↓
分别编译成目标文件
        ↓
归档成 libgemm_lib.a
```

静态库不是独立运行程序，它需要被最终可执行文件链接。

---

## 2.8 设置头文件搜索路径

```cmake
target_include_directories(gemm_lib
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)
```

告诉编译器：

```text
编译 gemm_lib 时，到项目根目录/include 查找头文件。
```

源码中可以直接写：

```cpp
#include "gemm.h"
```

而不必写相对路径：

```cpp
#include "../../include/gemm.h"
```

### `${CMAKE_SOURCE_DIR}`

代表顶层项目源码目录。

例如项目位于：

```text
/home/user/gemm_project
```

则该路径展开为：

```text
/home/user/gemm_project/include
```

### 为什么是 `PUBLIC`

`PUBLIC` 同时表示：

1. `gemm_lib` 自己编译时需要该目录；
2. 链接 `gemm_lib` 的目标也继承该目录。

因此：

```text
test_perf_gemm 链接 gemm_lib
    ↓
理论上自动获得 include/ 搜索路径
```

这说明后面再次给 `test_perf_gemm` 设置同一目录，可能是重复配置。

---

## 2.9 给 GEMM 库链接 CUDA Runtime 和 cuBLAS

```cmake
target_link_libraries(gemm_lib
    PUBLIC
        CUDA::cudart
        CUDA::cublas
)
```

### `CUDA::cudart`

CUDA Runtime Library。

如果代码使用：

```cpp
cudaMalloc
cudaMemcpy
cudaFree
cudaEventRecord
cudaDeviceSynchronize
cudaGetLastError
```

就需要 Runtime 支持。

`launcher.cu` 很可能会使用这些 API，因此链接 `CUDA::cudart` 合理。

### `CUDA::cublas`

NVIDIA 提供的高性能 BLAS 库，包含矩阵乘法接口。

该项目很可能使用 cuBLAS：

- 生成参考结果；
- 验证自定义 GEMM；
- 对比性能；
- 计算相对 cuBLAS 的效率。

### 一个需要结合源码确认的问题

当前配置让 `gemm_lib` 本身依赖 cuBLAS。

如果 `launcher.cu` 或其他库源码直接调用 cuBLAS，这是正确的。

但如果只有 `test_perf_gemm.cpp` 使用 cuBLAS，而自定义 GEMM 库完全不依赖 cuBLAS，那么更清晰的设计是：

```cmake
target_link_libraries(gemm_lib
    PUBLIC CUDA::cudart
)

target_link_libraries(test_perf_gemm
    PRIVATE gemm_lib CUDA::cublas
)
```

这样可以明确区分：

```text
自定义实现：gemm_lib
参考实现：cuBLAS
```

### 为什么这里使用 `PUBLIC`

表示：

- `gemm_lib` 自己需要这些库；
- 链接 `gemm_lib` 的目标也继承这些依赖。

所以后面测试程序再次写 `CUDA::cudart`、`CUDA::cublas`，从依赖传播角度可能重复。

---

## 2.10 CUDA 编译参数

```cmake
target_compile_options(gemm_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
```

这里使用生成器表达式：

```cmake
$<$<COMPILE_LANGUAGE:CUDA>:参数>
```

含义是：

> 仅在当前源文件以 CUDA 语言编译时添加该参数。

这样不会把 CUDA 专用选项错误传给普通 C++ 编译器。

---

### 2.10.1 `-O3`

开启较高级别优化，可能包括：

- 函数内联；
- 常量传播；
- 死代码删除；
- 循环优化；
- 指令调度；
- 地址计算优化。

对于 GEMM Benchmark，`-O3` 是必要的基础配置。

但它不能替代算法和 Kernel 设计。GEMM 性能主要仍取决于：

- 分块方式；
- Shared Memory；
- 寄存器分块；
- Warp 映射；
- 访存合并；
- 数据复用；
- 指令流水；
- Tensor Core 使用。

---

### 2.10.2 `--use_fast_math`

开启快速数学模式。

它允许编译器采用更激进或精度较低的数学实现，可能影响：

- 除法；
- 倒数；
- 特殊函数；
- 部分浮点语义；
- 数值误差。

对主要由 FP32 FMA 组成的 GEMM，影响可能没有 Softmax、激活函数等算子明显，但仍应注意数值一致性。

建议把它做成可选开关：

```cmake
option(GEMM_USE_FAST_MATH
       "Enable CUDA fast math"
       ON)
```

这样可分别验证：

```text
Fast Math 开启：性能
Fast Math 关闭：精度和基线
```

---

### 2.10.3 `-lineinfo`

保留设备代码与源码行号的映射信息，方便：

- Nsight Compute；
- Nsight Systems；
- 查看热点源码行；
- 将 SASS/PTX 指令定位回 CUDA 源码。

它通常不会像：

```bash
-G
```

那样严重关闭设备优化，因此适合 Release 性能分析。

对于 GEMM 优化项目，这个参数很有价值。

---

### 2.10.4 为什么是 `PRIVATE`

这些编译参数只作用于 `gemm_lib` 自己，不会传播到链接它的其他目标。

这是合理的，因为这些参数主要用于 CUDA Kernel 实现。

---

## 2.11 创建性能测试程序

```cmake
add_executable(test_perf_gemm
    tests/test_perf_gemm.cpp
)
```

将：

```text
tests/test_perf_gemm.cpp
```

编译成可执行程序：

```text
test_perf_gemm
```

从名称看，它不是普通示例，而是 GEMM 性能测试入口。

可能的运行流程是：

```text
分配矩阵
    ↓
初始化输入
    ↓
调用自定义 GEMM
    ↓
调用 cuBLAS
    ↓
验证误差
    ↓
Warm-up
    ↓
CUDA Event 计时
    ↓
计算 TFLOPS
    ↓
打印性能对比
```

---

## 2.12 测试程序的头文件目录

```cmake
target_include_directories(test_perf_gemm
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)
```

让测试程序能够包含项目公开头文件。

例如：

```cpp
#include "gemm.h"
```

这里使用 `PRIVATE`，表示该 include 路径只服务于 `test_perf_gemm` 本身。

不过前面 `gemm_lib` 已经把同一路径声明为 `PUBLIC`，测试程序链接 `gemm_lib` 后理论上会继承，因此这段可能是冗余的。

冗余不会导致构建错误，但会让依赖关系略显重复。

---

## 2.13 测试程序链接关系

```cmake
target_link_libraries(test_perf_gemm
    PRIVATE
        gemm_lib
        CUDA::cudart
        CUDA::cublas
)
```

测试程序依赖：

### `gemm_lib`

用于调用自定义 GEMM。

### `CUDA::cudart`

测试程序可能直接调用 CUDA Runtime，例如：

```cpp
cudaMalloc
cudaMemcpy
cudaEventCreate
cudaEventElapsedTime
```

### `CUDA::cublas`

用于参考结果或性能对比。

### `PRIVATE`

这些依赖只属于该可执行程序，不需要继续传播。

### 当前存在的重复关系

因为 `gemm_lib` 已经 `PUBLIC` 链接：

```text
CUDA::cudart
CUDA::cublas
```

测试程序链接 `gemm_lib` 后会继承它们。

因此又显式列出一次，通常不是必须的。

更合理的最终结构取决于源码：

#### 库内部使用 cuBLAS

```cmake
target_link_libraries(gemm_lib
    PUBLIC CUDA::cudart CUDA::cublas
)

target_link_libraries(test_perf_gemm
    PRIVATE gemm_lib
)
```

#### 只有测试使用 cuBLAS

```cmake
target_link_libraries(gemm_lib
    PUBLIC CUDA::cudart
)

target_link_libraries(test_perf_gemm
    PRIVATE gemm_lib CUDA::cublas
)
```

第二种更能体现“自定义 GEMM 与参考库分离”。

---

## 2.14 测试程序的 C++ 优化参数

```cmake
target_compile_options(test_perf_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
```

只对 C++ 源文件添加：

```bash
-O3
```

作用于：

```text
tests/test_perf_gemm.cpp
```

不会直接改变已经编译进 `gemm_lib` 的 CUDA Kernel。

这里区分了两类优化：

```text
gemm_lib CUDA 源码
→ CUDA 编译参数

test_perf_gemm.cpp
→ C++ 编译参数
```

即使核心测量对象是 GPU Kernel，Host 测试程序使用 Release 优化仍然是合理的。

---

## 2.15 完整构建流程

在项目根目录执行：

```bash
cmake -S . -B build
cmake --build build -j
```

### 配置阶段

```bash
cmake -S . -B build
```

会完成：

```text
检查 CMake 版本
查找 C++ 编译器
查找 CUDA 编译器
查找 CUDA Toolkit
确认 C++17/CUDA17
设置目标架构 89
生成底层构建文件
```

### 编译阶段

```bash
cmake --build build -j
```

大致执行：

```text
编译 launcher.cu
编译 gemm_warp_tile.cu
生成 libgemm_lib.a
编译 test_perf_gemm.cpp
链接 gemm_lib、cudart、cublas
生成 test_perf_gemm
```

### 运行

通常为：

```bash
./build/test_perf_gemm
```

实际路径取决于构建生成器和输出目录配置。

---

## 2.16 使用 Ninja 构建

如果安装了 Ninja：

```bash
cmake   -S .   -B build   -G Ninja   -DCMAKE_BUILD_TYPE=Release

cmake --build build
```

Ninja 的优点包括：

- 增量构建速度快；
- 并行构建方便；
- 输出清晰；
- 修改单个 `.cu` 文件后只重新编译受影响目标。

---

## 2.17 当前文件的优点

### 项目分层清晰

自定义 GEMM 被编译为库，Benchmark 单独成为可执行程序。

### 使用现代 CMake 目标

采用：

```cmake
CUDA::cudart
CUDA::cublas
```

而不是手写 CUDA 库路径。

### 区分 C++ 与 CUDA 编译参数

通过生成器表达式分别设置。

### 默认面向性能构建

使用：

```text
Release
-O3
--use_fast_math
```

### 保留性能分析信息

使用：

```text
-lineinfo
```

便于 Nsight 分析。

### 适合继续扩展

后续可加入：

```text
gemm_naive.cu
gemm_shared.cu
gemm_thread_tile.cu
gemm_vectorized.cu
gemm_tensor_core.cu
```

---

## 2.18 可以进一步改进的地方

## 2.18.1 架构参数允许外部覆盖

建议：

```cmake
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
```

避免每换设备都修改源码文件。

---

## 2.18.2 Fast Math 做成选项

```cmake
option(
    GEMM_USE_FAST_MATH
    "Enable CUDA fast math"
    ON
)
```

然后按条件添加 `--use_fast_math`。

这样更便于精度与性能对比。

---

## 2.18.3 精简重复 include 和链接依赖

需要结合源码确认：

- `gemm_lib` 是否真的调用 cuBLAS；
- 测试程序是否可直接继承 include；
- `cudart`、`cublas` 应由谁传播。

依赖关系越准确，项目越容易维护。

---

## 2.18.4 增加编译警告

例如普通 C++：

```cmake
-Wall
-Wextra
-Wpedantic
```

可以帮助发现：

- 未使用变量；
- 隐式类型转换；
- 参数错误；
- 潜在逻辑问题。

---

## 2.18.5 分离正确性测试与性能测试

当前只有：

```text
test_perf_gemm
```

更完整的结构可以是：

```text
test_correctness_gemm
benchmark_gemm
```

原因是：

- 正确性测试应覆盖更多边界形状；
- Benchmark 应使用稳定大矩阵和固定计时流程；
- 二者运行目标不同；
- CI 中未必适合执行长时间性能测试。

---

## 2.18.6 统一输出目录

可以设置：

```cmake
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/bin
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/lib
)
```

最终目录更整齐：

```text
build/bin/test_perf_gemm
build/lib/libgemm_lib.a
```

---

## 2.18.7 CUDA 可分离编译

如果不同 `.cu` 文件之间存在 Device 函数跨文件调用，可能需要：

```cmake
set_target_properties(gemm_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
```

当前没有开启并不一定错误。

如果：

- Kernel 和 Device 辅助函数都在同一 `.cu`；
- `launcher.cu` 只做 Host 侧启动；
- 没有跨翻译单元 Device 调用；

则不需要开启。

---

## 2.19 一个更灵活的参考版本

```cmake
cmake_minimum_required(VERSION 3.20)

project(gemm_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(
        CMAKE_BUILD_TYPE
        Release
        CACHE STRING
        "Build type"
        FORCE
    )
endif()

option(
    GEMM_USE_FAST_MATH
    "Enable CUDA fast math"
    ON
)

find_package(CUDAToolkit REQUIRED)

add_library(gemm_lib STATIC
    src/launcher.cu
    src/kernels/gemm_warp_tile.cu
)

target_include_directories(gemm_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(gemm_lib
    PUBLIC
        CUDA::cudart
)

target_compile_options(gemm_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

if(GEMM_USE_FAST_MATH)
    target_compile_options(gemm_lib PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
endif()

add_executable(test_perf_gemm
    tests/test_perf_gemm.cpp
)

target_link_libraries(test_perf_gemm
    PRIVATE
        gemm_lib
        CUDA::cublas
)

target_compile_options(test_perf_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
```

这个版本没有改变原项目核心结构，主要改进了：

- 架构可覆盖；
- Fast Math 可开关；
- 依赖关系更明确；
- 测试程序通过库继承 include 和 Runtime；
- cuBLAS 作为 Benchmark 侧依赖。

如果库内部实际调用 cuBLAS，则仍应将 `CUDA::cublas` 链接到 `gemm_lib`。

---

## 2.20 阅读整个项目时应怎样串联这个文件

推荐按以下顺序继续阅读：

```text
第一步：include/
查看项目对外暴露哪些 GEMM 接口和参数。

第二步：tests/test_perf_gemm.cpp
查看测试程序怎样调用接口、怎样计时和对比 cuBLAS。

第三步：src/launcher.cu
查看 Host 端怎样配置 Grid、Block 和 Kernel 参数。

第四步：src/kernels/gemm_warp_tile.cu
查看真正的 Warp Tile GEMM 如何映射线程和数据。

第五步：回到 CMakeLists.txt
理解这些文件如何编译成库并链接成测试程序。
```

整个构建过程可以最终概括为：

```text
CMake 配置
    ↓
识别 C++、CUDA 和 Toolkit
    ↓
按 sm_89 编译两个 .cu 文件
    ↓
生成 gemm_lib 静态库
    ↓
编译 test_perf_gemm.cpp
    ↓
链接自定义 GEMM、CUDA Runtime 和 cuBLAS
    ↓
生成性能测试程序
```

因此，这个文件在 GEMM 项目中的核心意义是：

> 将 Kernel 实现、Host 启动封装、公开头文件、CUDA Runtime、cuBLAS 参考实现和性能测试程序组织成一个可重复构建的完整工程。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-gemm算子|模块-gemm算子]]

%% 项目关联导航：结束 %%
