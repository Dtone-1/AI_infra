# `launcher.cu` 源码整体分析

## 1. 文件整体概览

`launcher.cu` 是这个 GEMM 项目的**统一算法入口与调度层**。

它本身并不负责完成矩阵乘法的具体计算，也没有直接编写线程、共享内存、寄存器分块等底层 CUDA 逻辑。它的核心任务是：

1. 接收上层传入的 GEMM 参数；
2. 判断应该使用哪一种 GEMM 实现；
3. 将请求转发给对应的具体算法启动函数。

因此，可以把它理解成整个项目中的“**总调度台**”或“**算法路由器**”。

项目上层只需要调用统一接口：

```cpp
launch_gemm(M, N, K, alpha, A, B, beta, C);
```

而不需要知道底层究竟使用了哪一种 CUDA Kernel、哪一种 Tile 配置或多少个线程。

从工程结构上看，它位于“测试程序”和“具体 CUDA Kernel”之间：

```text
test_perf_gemm.cpp
        │
        │ 调用统一 GEMM 接口
        ▼
include/gemm/launcher.h
        │
        │ 声明 launch_gemm
        ▼
src/launcher.cu
        │
        │ 选择具体算法并转发
        ▼
launch_gemm_warp_tile(...)
        │
        ▼
gemm_warp_tile.cu
        │
        │ 选择具体配置并启动 Kernel
        ▼
gemm_warp_tile_kernel<<<gridDim, blockDim>>>(...)
```

所以，`launcher.cu` 是连接项目“外部调用接口”和“内部高性能实现”的关键桥梁。

---

## 2. 这个文件解决的核心问题

如果没有 `launcher.cu`，测试程序可能需要直接调用某个具体实现，例如：

```cpp
launch_gemm_warp_tile(...);
```

这样会产生几个问题：

- 上层代码必须知道项目内部有哪些 Kernel；
- 更换算法时需要修改测试程序或业务代码；
- 新增其他 GEMM 实现后，上层会出现大量 `if`、`switch`；
- 接口与底层实现高度耦合，项目难以扩展。

`launcher.cu` 通过统一的 `launch_gemm()` 接口，将这些问题隔离在调度层内部。

上层只表达：“我要计算一次 GEMM。”

调度层负责决定：“这次应该使用哪一种算法。”

底层 Kernel 负责完成：“具体怎样切分矩阵、搬运数据并执行乘加。”

这体现了典型的软件分层思想：

```text
调用者关心计算目标
调度层关心算法选择
Kernel 层关心具体实现与性能
```

---

## 3. `launch_gemm()` 表达的数学含义

该文件提供的统一接口对应标准 GEMM 运算：

```text
C = alpha × A × B + beta × C
```

其中：

- `A` 是形状为 `M × K` 的矩阵；
- `B` 是形状为 `K × N` 的矩阵；
- `C` 是形状为 `M × N` 的结果矩阵；
- `alpha` 控制矩阵乘法结果的缩放；
- `beta` 控制旧矩阵 `C` 的保留比例。

接口参数如下：

```cpp
void launch_gemm(const int M,
                 const int N,
                 const int K,
                 const float alpha,
                 const float* A,
                 const float* B,
                 const float beta,
                 float* C,
                 GemmAlgo algo);
```

需要注意：

- `A`、`B`、`C` 在本项目中通常是 GPU 显存地址；
- `A` 和 `B` 使用 `const float*`，表示函数只读取它们；
- `C` 使用 `float*`，表示函数会修改其内容；
- 该函数提交的是 CUDA 计算任务，通常不会主动等待 GPU 完成；
- 是否同步、计时和检查结果由上层测试代码负责。

因此，`launch_gemm()` 更准确地说是“**发起一次 GEMM 计算**”，而不是在 CPU 中直接计算 GEMM。

---

## 4. 文件内部的核心工作流程

`launcher.cu` 的执行逻辑可以概括为四步。

### 第一步：接收算法参数

调用者可以显式指定 `GemmAlgo`，也可以使用默认的 `GemmAlgo::Auto`。

在 `launcher.h` 中，该参数具有默认值：

```cpp
GemmAlgo algo = GemmAlgo::Auto
```

所以测试程序可以直接写：

```cpp
launch_gemm(M, N, K, alpha, d_A, d_B, beta, d_C);
```

此时自动进入算法选择流程。

### 第二步：处理 `Auto` 模式

函数先复制一份算法选择结果：

```cpp
GemmAlgo final_algo = algo;
```

如果调用者指定的是 `Auto`，则调用：

```cpp
select_gemm_algo(M, N, K)
```

从设计意图上看，这个函数未来应该根据矩阵规模和形状自动选择最合适的算法。

例如未来可能出现如下策略：

```text
小矩阵              → Small GEMM Kernel
规则大矩阵          → Warp Tile Kernel
超大矩阵            → Tensor Core Kernel
边界不规则矩阵      → Edge-safe Kernel
特殊尺寸或退化情况  → Naive/Fallback Kernel
```

不过当前版本中：

```cpp
inline GemmAlgo select_gemm_algo(const int M, const int N, const int K) {
    return GemmAlgo::WarpTile;
}
```

无论 `M`、`N`、`K` 是多少，都固定返回 `WarpTile`。

这说明当前项目只有一个正式接入统一调度层的主算法，`Auto` 机制主要是为后续扩展预留的框架。

### 第三步：通过 `switch` 分发算法

算法确定后，代码通过 `switch` 转发：

```cpp
switch (final_algo) {
    case GemmAlgo::WarpTile:
        launch_gemm_warp_tile(...);
        break;

    case GemmAlgo::Auto:
    default:
        launch_gemm_warp_tile(...);
        break;
}
```

当前只有 `WarpTile` 分支，因此所有合法调用最终都会进入：

```cpp
launch_gemm_warp_tile(...)
```

这里并没有直接出现：

```cpp
kernel<<<gridDim, blockDim>>>(...)
```

说明 `launcher.cu` 只进行“算法级分发”，真正的 CUDA Kernel 启动由下一层 `gemm_warp_tile.cu` 负责。

### 第四步：进入具体 Warp Tile 实现

`launch_gemm_warp_tile()` 位于具体 Kernel 文件中。

下一层还会根据矩阵特点继续选择配置，例如：

- 不规则中等尺寸使用 `WarpTileIrregularConfig`；
- 较小问题使用 `WarpTileSmallConfig`；
- 其他情况使用 `WarpTileDefaultConfig`。

之后才会计算：

```cpp
dim3 blockDim(...);
dim3 gridDim(...);
```

并启动真正的 CUDA Kernel。

因此，本项目实际上存在两级调度：

```text
第一级：launcher.cu
选择哪一种 GEMM 算法

第二级：gemm_warp_tile.cu
选择该算法内部的哪一种参数配置
```

这两级调度分别处理不同层面的问题：

- `launcher.cu` 负责“算法类别”；
- `gemm_warp_tile.cu` 负责“算法内部配置”。

---

## 5. `launcher.cu` 在整个项目中的位置

整个 GEMM 项目可以按职责划分为以下几层。

### 第一层：测试与性能评估层

典型文件：

```text
tests/test_perf_gemm.cpp
```

主要负责：

- 创建矩阵；
- 分配 CPU 和 GPU 内存；
- 调用 cuBLAS；
- 调用自研 GEMM；
- CUDA Event 计时；
- 正确性比较；
- 计算 GFLOPS 和性能比例；
- 输出 CSV 结果。

该层通过 `launch_gemm()` 使用项目实现。

### 第二层：公共接口声明层

典型文件：

```text
include/gemm/launcher.h
```

主要负责声明：

```cpp
launch_gemm(...)
```

这样测试程序只需要包含公共头文件，不必包含具体 Kernel 的内部头文件。

### 第三层：算法调度层

典型文件：

```text
src/launcher.cu
```

也就是当前分析的文件。

主要负责：

- 接收统一 GEMM 请求；
- 处理 `GemmAlgo::Auto`；
- 选择具体算法；
- 转发给算法启动函数。

### 第四层：具体算法启动层

典型文件：

```text
src/kernels/gemm_warp_tile.cu
```

主要负责：

- 判断矩阵尺寸和对齐情况；
- 选择具体 Warp Tile 配置；
- 设置 `gridDim`、`blockDim`；
- 发起 CUDA Kernel。

### 第五层：CUDA Kernel 计算层

主要负责：

- Block Tile、Warp Tile 和 Thread Tile 映射；
- Global Memory 到 Shared Memory 的搬运；
- `cp.async` 异步加载；
- Shared Memory 双缓冲；
- 寄存器分块；
- FMA 累加；
- 边界处理；
- 将结果写回矩阵 `C`。

所以，`launcher.cu` 虽然代码量很少，但它位于整个调用链的关键中间位置。

---

## 6. 为什么该文件是 `.cu`，但里面没有 Kernel

`.cu` 文件并不意味着其中必须出现 `__global__` Kernel。

`.cu` 的含义是：该文件由 CUDA 编译器 NVCC 参与编译，可以安全地包含 CUDA 相关声明、调用 CUDA 代码，并与其他 CUDA 翻译单元链接。

当前文件虽然主要是普通 C++ 调度逻辑，但它：

- 属于 CUDA 算法调用链；
- 调用了由 CUDA 源文件实现的启动函数；
- 与 CUDA 静态库一起编译和链接；
- 未来可能直接加入 CUDA Runtime 逻辑。

因此将其命名为 `launcher.cu` 是合理的。

更重要的是，要区分以下三类函数：

```text
launch_gemm(...)
    普通 Host 函数，统一算法入口

launch_gemm_warp_tile(...)
    普通 Host 函数，配置并启动具体 Kernel

gemm_warp_tile_kernel<<<...>>>(...)
    真正在 GPU 上执行的 __global__ Kernel
```

`launcher.cu` 中的代码是在 CPU Host 端运行的，只是它最终会间接触发 GPU 工作。

---

## 7. `GemmAlgo` 的工程意义

`GemmAlgo` 是算法类型枚举，用于描述“这次 GEMM 应该采用哪一种实现”。

当前代码中至少涉及：

```text
GemmAlgo::Auto
GemmAlgo::WarpTile
```

它的意义不是改变矩阵乘法的数学结果，而是改变底层执行策略。

所有算法都应当计算：

```text
C = alpha × A × B + beta × C
```

但不同算法可能有不同特点：

- 适合不同矩阵尺寸；
- 使用不同 Tile 大小；
- 使用或不使用 Tensor Core；
- 对齐要求不同；
- 对边界矩阵的处理方式不同；
- 寄存器、共享内存和线程数量不同；
- 性能表现不同。

因此，`GemmAlgo` 是构建可扩展算子库的重要抽象。

未来新增算法时，可以形成如下结构：

```cpp
switch (final_algo) {
    case GemmAlgo::Naive:
        launch_gemm_naive(...);
        break;

    case GemmAlgo::BlockTile:
        launch_gemm_block_tile(...);
        break;

    case GemmAlgo::WarpTile:
        launch_gemm_warp_tile(...);
        break;

    case GemmAlgo::TensorCore:
        launch_gemm_tensor_core(...);
        break;
}
```

此时，上层测试程序仍然只调用 `launch_gemm()`，不需要随底层实现变化而修改。

---

## 8. 匿名命名空间的作用

`select_gemm_algo()` 被放在匿名命名空间中：

```cpp
namespace {
    inline GemmAlgo select_gemm_algo(...) {
        ...
    }
}
```

这表示该辅助函数只在当前 `launcher.cu` 文件内部可见。

它不是项目对外公开的 API，其他源文件不能直接依赖它。

这样做的工程意义包括：

- 避免污染全局符号空间；
- 避免与其他文件中的同名函数发生链接冲突；
- 明确区分公共接口和内部实现细节；
- 允许未来自由修改自动选择策略，而不影响外部代码。

公共接口是：

```cpp
launch_gemm(...)
```

内部辅助逻辑是：

```cpp
select_gemm_algo(...)
```

这种区分有利于保持模块边界清晰。

---

## 9. 当前实现的实际状态

从架构上看，这个文件已经搭好了一个算法调度框架；但从当前功能看，它还比较简单。

### 已经实现的部分

- 提供统一 GEMM 接口；
- 支持显式传入算法类型；
- 支持 `Auto` 模式；
- 使用 `switch` 进行算法分发；
- 接入 Warp Tile 实现；
- 提供默认回退路径。

### 尚未真正实现的部分

- `select_gemm_algo()` 尚未根据 `M/N/K` 做实际判断；
- 当前只有一种算法类别；
- `M/N/K` 在自动选择函数中暂时未被利用；
- 没有根据 GPU 架构、数据类型、对齐方式或显存条件选择算法；
- 没有独立的错误检查与参数合法性检查；
- 没有对 CUDA Stream 进行抽象；
- 没有自动调优或经验表驱动的算法选择。

因此，当前的 `Auto` 更像是“未来自动选择机制的接口占位”，而不是已经完成的智能调度器。

---

## 10. 默认分支的意义

`switch` 中的 `default` 仍然调用 Warp Tile：

```cpp
case GemmAlgo::Auto:
default:
    launch_gemm_warp_tile(...);
    break;
```

这是一种回退机制，保证在算法值异常或未覆盖时，程序仍然尝试使用一个默认实现。

它的优点是：

- 简化当前项目；
- 避免没有执行任何 Kernel；
- 保证现阶段所有请求都能落到已有实现。

但在更严格的工程中，非法枚举值可能更适合：

- 输出错误信息；
- 返回错误码；
- 抛出异常；
- 触发断言；
- 明确回退到一个经过验证的通用 Kernel。

因为静默回退虽然提高了可用性，但也可能掩盖调用者错误。

---

## 11. 为什么不把所有逻辑都写进 `test_perf_gemm.cpp`

测试程序的职责是验证和测量，而不是管理算子内部实现。

若将算法选择、Grid 配置和 Kernel 启动全部写进测试文件，会导致：

- 测试代码过度依赖底层实现；
- 项目无法作为独立库被其他程序复用；
- 更换 Kernel 时必须修改测试代码；
- 正确性测试、性能测试和算法实现混在一起；
- 后续扩展维护困难。

当前结构则更加合理：

```text
test_perf_gemm.cpp
只负责“测”

launcher.cu
负责“选”

gemm_warp_tile.cu
负责“配”和“发射”

gemm_warp_tile_kernel
负责“算”
```

这是一个小型算子项目逐步演化为算子库时需要具备的基本模块化设计。

---

## 12. 该文件与性能的关系

`launcher.cu` 自身几乎不承担 GEMM 的主要计算，因此它不是性能优化的核心位置。

一次 GEMM 通常包含大量浮点运算，而该文件只进行：

- 一次枚举判断；
- 一次函数调用；
- 一次 `switch` 分发。

这些 CPU 开销相对于大型 GEMM Kernel 通常很小。

但它会**间接决定性能**，因为它选择了哪一种底层实现。

未来如果项目拥有多种 Kernel，那么调度策略是否合理会直接影响：

- 小矩阵延迟；
- 大矩阵吞吐；
- 边界尺寸性能；
- GPU 利用率；
- 算法稳定性；
- 不同 GPU 架构上的性能表现。

因此：

```text
Kernel 决定“某个算法能跑多快”
Launcher 决定“当前问题是否选到了合适的算法”
```

两者共同决定最终性能。

---

## 13. 该文件不负责什么

理解 `launcher.cu` 时，需要明确它没有承担以下职责：

- 不负责申请显存；
- 不负责将矩阵从 CPU 复制到 GPU；
- 不负责矩阵初始化；
- 不负责 CUDA Event 计时；
- 不负责正确性检查；
- 不负责计算 GFLOPS；
- 不负责 cuBLAS 对比；
- 不负责 Shared Memory 数据搬运；
- 不负责 Warp 和线程映射；
- 不负责边界元素的实际计算；
- 不负责 CPU/GPU 同步；
- 不负责释放显存。

这些职责分别由测试层、CUDA Runtime 调用层和具体 Kernel 层完成。

---

## 14. 阅读这个文件时最重要的认识

对于初学者来说，最容易产生的误解是：代码文件较短，所以不重要。

实际上，`launcher.cu` 的价值主要体现在**架构位置**而不是代码量。

应重点理解以下几点：

1. `launch_gemm()` 是项目对外的统一算子入口；
2. `GemmAlgo::Auto` 表示由项目内部决定算法；
3. 当前自动选择逻辑尚未真正区分矩阵形状；
4. 当前所有调用最终都转发给 Warp Tile 实现；
5. 该文件只做算法级调度，不直接执行矩阵计算；
6. 真正的 Kernel 启动位于 `gemm_warp_tile.cu`；
7. 真正的性能优化逻辑位于具体 Kernel 与配置文件中；
8. 该结构为未来增加更多算法和自动选择策略预留了扩展点。

---

## 15. 推荐的关联源码阅读顺序

为了真正理解 `launcher.cu`，建议按照以下顺序继续阅读：

```text
1. include/gemm/types.h
   理解 GemmAlgo 枚举有哪些取值

2. include/gemm/launcher.h
   理解 launch_gemm 对外暴露的函数签名和默认参数

3. src/launcher.cu
   理解统一入口、Auto 选择和算法分发

4. include/gemm/kernels/gemm_warp_tile.h
   理解具体算法启动函数的声明

5. src/kernels/gemm_warp_tile.cu 的 launch_gemm_warp_tile()
   理解如何根据矩阵尺寸选择配置

6. gemm_warp_tile_config.h
   理解 BM、BN、BK、WM、WN、TM、TN 等参数

7. gemm_warp_tile_kernel
   理解线程块、Warp、线程和 Tile 的具体映射

8. tests/test_perf_gemm.cpp
   从调用者角度串联完整测试流程
```

这样可以形成完整认识：

```text
外部调用
→ 公共声明
→ 算法调度
→ 配置选择
→ Kernel 启动
→ GPU 计算
→ 同步计时与结果验证
```

---

## 16. 一句话总结

`launcher.cu` 是 GEMM 项目的统一入口和算法分发中心：它接收上层的 GEMM 请求，在 `Auto` 模式下决定使用哪种实现，并将计算转发到具体的 Warp Tile 启动函数；当前选择逻辑固定使用 Warp Tile，但整体结构已经为未来接入多种 GEMM Kernel、按矩阵形状自动选优以及实现更完整的算子调度系统预留了接口。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-gemm算子|模块-gemm算子]]

%% 项目关联导航：结束 %%
