# CUDA FP32 GEMM 项目从零学习与大厂算子岗位能力路线

> 适用对象：目前几乎没有 CUDA、GPU 算子开发和 GEMM 优化基础，但已经接触 Python、C++ 基础、PyTorch、Transformer/vLLM，希望彻底理解本仓库，并将其作为 AI Infra / 高性能算子方向简历项目。
>
> 建议周期：**标准路线 14 周，每周 20～30 小时**。若每天能够稳定投入 6 小时以上，可压缩到 8～10 周，但不建议跳过实验和性能分析环节。

---

## 1. 先明确：这个项目到底难在哪里

本项目不是一个只调用 `cublasSgemm` 的示例，而是一个手写的 CUDA FP32 GEMM Kernel，核心功能包括：

- 实现矩阵乘法：`C = alpha * A * B + beta * C`；
- 使用 Block Tile、Warp Tile、Thread Tile 进行多级分块；
- 将全局内存中的 A、B 分块搬运到 Shared Memory；
- 使用寄存器保存线程私有输入片段和累加结果；
- 使用 `float4` 完成 16 字节向量化访存；
- 使用内联 PTX `cp.async` 进行 Global Memory → Shared Memory 异步拷贝；
- 使用双缓冲隐藏部分数据搬运延迟；
- 为小尺寸、非对齐尺寸和常规大尺寸设置不同 Tile 配置；
- 处理 M、N、K 不能整除 Tile 的边界情况；
- 使用 CUDA Event 计时，并与 cuBLAS SGEMM 对比 GFLOPS；
- 提供 DRAM、L1、L2、Shared Memory 延迟和带宽 Microbenchmark。

因此，“彻底搞懂项目”至少应达到以下六个标准：

1. **能解释**：不看代码也能画出 Block、Warp、Thread 如何共同计算 C 的一个 Tile。
2. **能追踪**：给定一个线程编号，能计算它加载 A/B 的哪些元素、负责 C 的哪些元素。
3. **能修改**：改变 `BM/BN/BK/WM/WN/TM/TN` 后，能判断映射是否合法、资源开销如何变化。
4. **能验证**：能设计随机输入、边界尺寸、不同 `alpha/beta` 的正确性测试。
5. **能分析**：能用 Nsight Compute 判断 Kernel 是计算受限、带宽受限、延迟受限，还是 Occupancy/寄存器受限。
6. **能优化**：能提出假设、修改 Kernel、进行公平 Benchmark，并用数据说明优化是否有效。

只做到“能把代码逐行翻译成中文”，还不能算真正掌握。

---

## 2. 项目文件与所需知识映射

| 项目文件 | 主要作用 | 必须掌握的知识 |
|---|---|---|
| `CMakeLists.txt` | 编译 CUDA/C++、指定 GPU 架构和优化参数 | CMake、NVCC、编译架构、`-O3`、`--use_fast_math`、`-lineinfo` |
| `include/gemm/types.h` | 定义算子算法类型 | C++ 枚举、接口设计 |
| `include/gemm/launcher.h` | 对外暴露 GEMM 接口 | 头文件、指针、ABI、Host API |
| `src/launcher.cu` | 根据算法类型进入具体 Kernel | Dispatch、运行时选择、算子接口分层 |
| `gemm_warp_tile_config.h` | 编译期保存 Tile 参数 | C++ 模板、`constexpr`、编译期特化 |
| `gemm_warp_tile.cu` | 核心 CUDA Kernel | CUDA 执行模型、访存、分块、寄存器、Shared Memory、Warp 映射、PTX、`cp.async`、双缓冲、边界处理 |
| `tests/test_perf_gemm.cpp` | 正确性和性能测试 | CUDA Runtime API、cuBLAS、CUDA Event、GFLOPS、误差分析、Benchmark 方法学 |
| `microbenchmark/*` | 测量各级存储性能 | GPU 存储层次、指针追逐、带宽/延迟、时钟周期、微基准设计 |
| `docs/warp_tiles.md` | 解释分块映射 | Tile 几何关系、线程映射；同时要具备核对文档与代码一致性的能力 |

### 2.1 阅读该仓库时要特别注意的实际问题

这些问题本身就可以作为后续改造任务：

- `CMakeLists.txt` 将 `CMAKE_CUDA_ARCHITECTURES` 固定为 `89`，不适合作为可移植工程；应改为可配置或按本机架构编译。
- `docs/warp_tiles.md` 中部分说明与当前配置并不完全一致。例如默认配置为 256 个线程，即 8 个 Warp，而文档仍写“一共 4 个 Warp”。学习时应以代码推导为准，并修正文档。
- 当前正确性测试主要使用 A 全为 1、B 全为 2，覆盖面不足，无法充分发现索引错位、部分元素遗漏、`beta` 路径等问题。
- 快速写回路径即使 `beta == 0` 仍会读取原 C。若 C 中存在 NaN，`0 * NaN` 仍可能产生 NaN；应增加真正的 `beta == 0` Epilogue 快速路径。
- 当前汇总的 5 次稳定结果中，平均 `my_gflops / cublas_gflops` 约为 **0.745648**，但 257、511、1000 等不规则小尺寸的比值明显偏低。因此简历不能笼统写成“所有尺寸达到 cuBLAS 75%”，应说明统计口径和尺寸分布。

---

## 3. 需要学习的完整技术栈

## 3.1 第一层：C++ 与工程基础

### 必须掌握

- 变量、作用域、函数、数组、结构体；
- 指针、`const`、指针偏移、二维数组线性化；
- 引用、值传递与指针传递；
- 栈、堆、RAII 和资源释放；
- 头文件、源文件、声明与定义；
- `enum class`、命名空间；
- 模板函数、模板结构体、模板参数；
- `constexpr`、`static constexpr`；
- `inline`、`__forceinline__` 的基本含义；
- `reinterpret_cast` 和内存对齐；
- CMake、静态库、可执行文件、链接过程；
- Linux 命令、Git 分支、Commit、Diff。

### 对本项目最关键的 C++ 能力

你不需要先学完整个现代 C++，但必须能够独立理解下面这种代码：

```cpp
template<int BM, int BN, int BK>
struct Config {
    static constexpr int kBM = BM;
};
```

以及：

```cpp
float4 v = *reinterpret_cast<const float4*>(ptr);
```

前者决定 Kernel 的编译期参数，后者关系到向量化访存、地址对齐和潜在越界。

---

## 3.2 第二层：线性代数与 GEMM 基础

必须掌握：

- 矩阵乘法定义与维度：`A[M,K] * B[K,N] = C[M,N]`；
- `C[m,n] = Σ A[m,k] * B[k,n]`；
- 行主序、列主序、Leading Dimension；
- `alpha`、`beta` 的含义；
- 为什么 GEMM 的计算量约为 `2MNK` FLOPs；
- Square GEMM、Rectangular GEMM、Batched GEMM；
- 算术强度 Arithmetic Intensity；
- 为什么分块能够提高数据复用率；
- GEMM 在 Transformer 中对应 QKV Projection、MLP、LM Head 等线性层。

### 必须能回答

- 为什么 A 的一个元素会被多个 N 方向输出复用？
- 为什么 B 的一个元素会被多个 M 方向输出复用？
- 为什么不分块的 Naive GEMM 会反复访问 Global Memory？
- 为什么 Tile 越大不一定越快？

---

## 3.3 第三层：计算机体系结构与性能基础

必须掌握：

- CPU 与 GPU 的设计目标差异；
- 吞吐量与单线程延迟；
- SIMD、SIMT、Warp；
- GPU 的 SM、CUDA Core、Warp Scheduler；
- 指令级并行 ILP、线程级并行 TLP；
- 延迟隐藏的含义；
- 峰值算力、有效算力、带宽上限；
- Cache、Shared Memory、Register 的区别；
- Roofline Model；
- Compute Bound、Memory Bound、Latency Bound；
- Occupancy 与性能之间并非简单正相关。

---

## 3.4 第四层：CUDA C++ 基础

必须掌握：

- Host、Device、Kernel；
- `__global__`、`__device__`、`__host__`；
- `<<<grid, block>>>` Kernel Launch；
- `threadIdx`、`blockIdx`、`blockDim`、`gridDim`；
- Grid、Block、Warp、Thread 的层次；
- CUDA Runtime API：`cudaMalloc/cudaFree/cudaMemcpy/cudaMemset`；
- 同步与异步执行；
- `cudaDeviceSynchronize`；
- CUDA Stream 和 Event；
- Kernel Launch Error 与异步运行时错误；
- `cudaGetLastError`、Compute Sanitizer；
- Device 属性和 Compute Capability。

### 必做基础 Kernel

1. Vector Add；
2. SAXPY；
3. Matrix Add；
4. Reduce Sum；
5. Transpose；
6. Naive GEMM。

如果不能独立完成这六个 Kernel，不应直接进入本项目核心代码。

---

## 3.5 第五层：CUDA 存储层次与访存优化

必须掌握：

- Global Memory、L2、L1、Shared Memory、Register；
- Coalesced Memory Access；
- 32/64/128 字节内存事务；
- 地址对齐；
- `float2/float4` 向量化加载；
- Shared Memory Bank 和 Bank Conflict；
- Register Spill 与 Local Memory；
- Shared Memory 容量、Register 数量对 Occupancy 的限制；
- Cache 命中率与真实性能之间的关系；
- AoS 与 SoA 基本概念。

### 与本项目直接对应

- `float4`：每个线程一次加载或写回 4 个 FP32，即 16 字节；
- `AS`、`BS`：将 A、B 的当前 K Tile 放入 Shared Memory；
- `reg_m`、`reg_n`：缓存本线程计算所需输入；
- `res`：本线程负责的 C 子块累加器；
- `A_STRIDE`：决定 A Tile 在 Shared Memory 中的布局；
- `inner_row_*`、`inner_col_*`：决定每个线程搬运哪些元素。

---

## 3.6 第六层：GEMM 多级分块

本项目最核心的知识是四级映射：

```text
完整 C[M,N]
  └── Thread Block Tile: BM × BN
        └── Warp Tile: WM × WN
              └── Warp 每轮子区域: WSUBM × WSUBN
                    └── Thread Tile: TM × TN，并进行多轮累加
```

必须掌握：

- Block Tiling；
- K 维循环和 `BK`；
- Warp Tiling；
- Thread/ Register Tiling；
- 每个线程保存多个累加器的原因；
- Warp 内线程到二维输出区域的映射；
- Tile 参数的合法性约束；
- Tile 大小与数据复用、Shared Memory、Register Pressure、Occupancy 的权衡。

### 默认配置必须能够手工推导

默认配置：

```text
BM=128, BN=256, BK=8
WM=64,  WN=64
WNITER=4
TM=8, TN=4
NUM_THREADS=256
```

你至少需要推导出：

- 一个 Block 有 256 个线程，即 8 个 Warp；
- C 的一个 Block Tile 有 `128 × 256 = 32768` 个输出；
- 一共有 `(128/64) × (256/64) = 8` 个 Warp Tile；
- 每个 Warp 负责一个 `64 × 64` 输出区域；
- `WSUBN = WN / WNITER = 16`；
- `tcol`、`trow` 如何将 32 个线程映射到子区域；
- 每个线程最终累加多少个 C 元素；
- Shared Memory 双缓冲占用量；
- `res` 数组会带来多大的寄存器压力。

只有能独立完成这组推导，才算真正进入项目核心。

---

## 3.7 第七层：异步拷贝、流水线与内联 PTX

本项目使用：

- `cp.async.ca.shared.global`；
- `cp.async.commit_group`；
- `cp.async.wait_group`；
- 双缓冲 `AS[2]`、`BS[2]`；
- 计算当前 Tile 时预取下一个 Tile。

必须掌握：

- 为什么同步加载会形成“加载—等待—计算—加载”的串行过程；
- Producer/Consumer Pipeline；
- Double Buffering；
- `commit_group` 和 `wait_group` 的语义；
- `__syncthreads()` 与异步拷贝完成之间的区别；
- 为什么 `cp.async` 需要关注地址对齐和有效范围；
- PTX 是什么，SASS 是什么；
- 为什么项目直接写 PTX，而不是只使用普通 CUDA C++。

这一部分属于进阶内容。在理解 Shared Memory Tiling 前，不要提前死磕 PTX。

---

## 3.8 第八层：正确性、Benchmark 与性能分析

必须掌握：

- Warmup 的目的；
- CUDA Event 的正确计时方法；
- 为什么不能用 CPU 普通计时直接包围异步 Kernel；
- GFLOPS 公式；
- 平均值、中位数、标准差；
- 冷启动与稳态性能；
- Clock 波动、温度、功耗限制；
- 公平比较必须保证输入、数据类型、布局、数学语义一致；
- 绝对误差、相对误差、容差；
- NaN/Inf 检查；
- Correctness First，Performance Second。

必须会使用：

- Nsight Systems：观察整体时间线、Kernel Launch、Memcpy、CPU/GPU 空洞；
- Nsight Compute：分析单个 Kernel；
- Compute Sanitizer：检查越界和竞争；
- `ncu` CLI：固定指标进行可重复采集。

重点 Nsight Compute 指标：

- Kernel Duration；
- SM Throughput；
- DRAM Throughput；
- L1/L2 Hit Rate；
- Achieved Occupancy；
- Active Warps；
- Registers per Thread；
- Shared Memory per Block；
- Eligible Warps / Stall Reasons；
- Global Load/Store Efficiency；
- Shared Memory Bank Conflict；
- Roofline 中的算术强度和性能位置。

---

# 4. 14 周详细学习计划表

> 推荐节奏：每周学习 6 天，每天 3～5 小时。每周必须有代码、数据或文档交付物，不能只看视频。

| 周次 | 学习主题 | 核心内容 | 必做实践 | 本周验收标准 |
|---|---|---|---|---|
| 第 1 周 | C++ 算子开发必备基础 | 指针、数组线性化、`const`、头文件、模板、`constexpr`、`reinterpret_cast`、CMake | 写 CPU Matrix Multiply；写模板化 Tile Config；完成一个静态库+测试程序 | 能解释项目 Config 模板和 `float4` 转换，不依赖逐字搜索 |
| 第 2 周 | GPU 与 CUDA 执行模型 | Host/Device、Grid/Block/Warp/Thread、Kernel Launch、CUDA Runtime API | Vector Add、SAXPY、Matrix Add；打印线程索引验证映射 | 能根据 M、N 手算 Grid/Block，并解释每个线程负责什么 |
| 第 3 周 | CUDA 内存与同步 | Global/Shared/Register、Coalescing、Bank Conflict、同步、Stream/Event | Matrix Transpose：Naive、Coalesced、Shared Memory 三版 | 能通过 Nsight 说明三版性能差异来自哪里 |
| 第 4 周 | Reduction 与性能方法 | Warp、分支发散、归约、计时、GFLOPS、正确性容差 | Reduce Sum 多版本；建立统一 Benchmark Harness | 能区分错误计时和正确计时；能输出均值/中位数/标准差 |
| 第 5 周 | Naive GEMM | GEMM 数学、行列主序、cuBLAS 接口、`2MNK` | CPU Reference、CUDA Naive GEMM、cuBLAS 对照 | 随机矩阵和多种 M/N/K 下结果正确，并能解释 cuBLAS 参数顺序 |
| 第 6 周 | Shared Memory Block Tiling | BM/BN/BK、K Tile 循环、数据复用、边界处理 | 实现 Shared Memory Tiled GEMM；测试非整除尺寸 | 能画出 A/B Tile 如何加载，能计算一次 Tile 的数据复用次数 |
| 第 7 周 | Register/Thread Tiling | Thread Tile、寄存器累加器、ILP、寄存器压力 | 每线程计算 2×2、4×4 等多个输出；比较性能与寄存器数 | 能解释“每线程算更多”为什么可能变快，也可能导致 Occupancy 降低 |
| 第 8 周 | Warp Tiling | Warp Tile、二维线程映射、WM/WN/TM/TN | 对默认配置完成逐参数推导；制作线程映射图 | 给定 `threadIdx.x`，能手算其 C 输出坐标和 A/B 读取坐标 |
| 第 9 周 | 向量化与 Shared Memory 布局 | `float4`、16B 对齐、合并访问、Shared Memory 布局 | 给 GEMM 加向量化加载/写回；检查对齐与 Edge Path | 能说明哪些条件允许 `float4`，哪些尺寸必须退化到标量路径 |
| 第 10 周 | `cp.async` 与双缓冲 | PTX、异步拷贝、Pipeline、Commit/Wait、双缓冲 | 先写同步双缓冲概念版，再阅读项目 `load_tile_async` | 能画出 Tile 0/1 的加载与计算时间线，解释每一次同步的必要性 |
| 第 11 周 | 精读本项目核心 Kernel | `compute`、`compute_k`、`compute_edge`、Fast/Edge 路径、Epilogue | 按函数写调用关系和数据流文档；逐配置计算资源使用 | 不看原文档，能够从 Kernel 入口完整讲到 C 写回 |
| 第 12 周 | Dispatch、测试与 Microbenchmark | Small/Irregular/Default 配置选择；DRAM/L1/L2/SMEM 测试 | 扩展随机测试、矩形尺寸、alpha/beta、NaN；运行 Microbenchmark | 测试可发现索引、越界、beta 路径和数值问题；能解释各级存储性能 |
| 第 13 周 | Nsight 深度性能分析 | Roofline、Occupancy、Stall Reasons、Cache、Register、SMEM | 对三种 Config 各抓一份 NCU 报告，建立对比表 | 每个性能结论都有指标证据，不使用“感觉”“可能”代替数据 |
| 第 14 周 | 项目改造与简历交付 | 修复工程问题、优化不规则尺寸、PyTorch 接入、文档与图表 | 完成至少 2 个有效改造；生成最终报告、性能曲线、PR/Commit | 可以进行 15～20 分钟项目答辩，并回答参数、瓶颈、失败实验和取舍 |

---

## 5. 每个阶段应该产出的文件

建议在仓库中增加：

```text
docs/
├── 01_gemm_math_and_layout.md
├── 02_cuda_execution_mapping.md
├── 03_block_warp_thread_tile.md
├── 04_cp_async_pipeline.md
├── 05_ncu_analysis.md
├── 06_correctness_design.md
└── 07_optimization_report.md

benchmarks/
├── benchmark_shapes.json
├── run_benchmark.py
├── correctness_cases.csv
├── perf_raw.csv
└── perf_summary.csv

scripts/
├── run_compute_sanitizer.sh
├── profile_ncu.sh
└── plot_results.py
```

这些交付物比“我看完了课程”更能证明能力。

---

# 6. 项目达到简历标准前必须补齐的改造

## P0：必须完成，否则项目可信度不足

### 6.1 完善正确性测试

至少覆盖：

- 随机正负数，而不是只使用常数矩阵；
- 方阵和矩形矩阵；
- `M/N/K` 分别不对齐；
- 极小尺寸：1、2、3、7、15、31、33；
- 边界尺寸：63/64/65、127/128/129、255/256/257；
- `alpha = 0、1、随机值`；
- `beta = 0、1、随机值`；
- C 初值包含普通随机数，并增加 NaN 防御测试；
- 绝对误差和相对误差；
- Compute Sanitizer 无越界、无竞争。

### 6.2 修正编译架构配置

不要硬编码：

```cmake
set(CMAKE_CUDA_ARCHITECTURES 89)
```

应允许：

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
```

或者通过参数明确指定目标架构。首先用 `deviceQuery` 或运行时 API 确认本机 GPU 的 Compute Capability。

### 6.3 建立可信 Benchmark

至少输出：

- 每个 Shape 的 cuBLAS GFLOPS；
- 自研 Kernel GFLOPS；
- 比值；
- 单次耗时；
- 5～20 次独立运行的均值、中位数、标准差；
- GPU 型号、CUDA、Driver、编译参数、时钟/功耗状态；
- Aligned、Irregular、Small、Large 分组统计；
- 不只报告一个总体平均数。

### 6.4 完成 Nsight 证据链

每个主要优化必须按照以下格式记录：

```text
问题：某尺寸性能低
假设：寄存器过多导致 Active Warps 不足
修改：减小 Thread Tile 或调整 Warp Tile
指标变化：Registers/Thread、Occupancy、Stall、Duration
最终结果：GFLOPS 提升/下降多少
结论：保留或回退，以及原因
```

---

## P1：建议完成，能显著提高简历含金量

### 6.5 优化 Irregular Shape

当前小型不规则尺寸相对 cuBLAS 较弱，可以尝试：

- 更小的 Tile；
- 分离 M/N/K 尾块处理；
- 主体 Kernel + Tail Kernel；
- Padding 与非 Padding 对照；
- `beta == 0` 专用 Epilogue；
- 更精细的 Shape Bucket；
- 基于离线 Benchmark 的 Auto-tuning Dispatch。

目标不是保证每个 Shape 都超过 cuBLAS，而是解释：

- 哪些 Shape 适合自研 Kernel；
- 哪些 Shape 应回退 cuBLAS；
- Dispatch 如何降低最坏情况回退。

### 6.6 增加 PyTorch Custom Operator

完成：

```python
C = torch.ops.my_gemm.gemm(A, B)
```

并具备：

- C++/CUDA Extension；
- Shape、dtype、device 检查；
- 当前 CUDA Stream；
- Contiguous 和非 Contiguous 处理策略；
- `torch.library`/`TORCH_LIBRARY` 注册；
- `torch.compile` 兼容性基础；
- Python 单元测试和 Benchmark。

这样项目才从“孤立 CUDA Demo”变成“可被 AI 框架调用的真实算子”。

### 6.7 增加自动调优与配置搜索

搜索：

- BM、BN、BK；
- WM、WN；
- TM、TN；
- 线程数；
- Stage 数；
- 不同 Shape Bucket。

记录每个配置：

- 合法性；
- Shared Memory；
- Registers；
- Occupancy；
- GFLOPS；
- 编译时间。

这能体现你不是“抄固定参数”，而是理解性能空间。

---

## P2：面向大厂算子岗位的升级方向

### 6.8 Tensor Core 与混合精度 GEMM

当前项目是 FP32 CUDA Core GEMM。AI 大模型算子中更常见：

- FP16；
- BF16；
- TF32；
- INT8；
- FP8；
- 新架构上的 FP4/Block-scaled 计算。

建议新增一个方向：

1. 使用 WMMA 写一个基础 Tensor Core GEMM；
2. 使用 CUTLASS/CuTe 实现同语义 Kernel；
3. 与 cuBLASLt、PyTorch 对比；
4. 分析 Accumulator 精度和数值误差；
5. 增加 Bias 或 Activation Epilogue Fusion。

这一步能把项目从传统 HPC SGEMM，提升到 AI Infra 常用算子场景。

### 6.9 再完成一个非 GEMM 算子

建议从以下选择一个：

- Fused Softmax；
- RMSNorm；
- LayerNorm；
- RoPE；
- SwiGLU / SiLU + Mul；
- Reduction；
- FlashAttention 简化版；
- INT8/FP8 Quantize-Dequantize；
- MoE Top-K / Grouped GEMM。

推荐顺序：

```text
RMSNorm 或 Fused Softmax
    → Triton 实现
    → CUDA 实现
    → PyTorch 接入
    → Nsight 对比
```

原因是仅有 GEMM 容易让面试官认为你只理解一个固定模板；再做一个 Memory-bound/Fusion 算子，可以证明你理解不同类型的瓶颈。

---

# 7. 面向大厂算子岗位，还需要补哪些技术

截至 2026 年的相关岗位通常不只要求“会写 CUDA”，还强调 GPU 架构、Triton/CUTLASS、低精度计算、推理框架、图编译和完整性能调优能力。因此建议分级学习。

## 7.1 第一优先级：求职前必须具备

| 技术 | 要达到的程度 | 为什么重要 |
|---|---|---|
| Linux C/C++ | 能独立编译、调试、分析内存和并发问题 | 算子和推理框架主体通常是 C++/CUDA |
| CUDA C++ | 能独立写 Kernel、优化访存、处理动态 Shape | 算子岗位核心门槛 |
| GPU 架构 | 理解 SM、Warp、Memory Hierarchy、Tensor Core | 决定是否真正具备性能推理能力 |
| Nsight Systems/Compute | 能从指标定位瓶颈 | “会写”与“会优化”的分界线 |
| GEMM 优化 | Block/Warp/Thread Tile、Pipeline、Epilogue | 计算密集型算子基础 |
| Reduction/Norm/Softmax | 会优化 Memory-bound 算子 | 覆盖另一类常见瓶颈 |
| Triton | 能写 Matmul、Softmax、Norm，并调参 | 当前大量团队用于快速算子开发 |
| PyTorch Custom Op | 能让 Kernel 被框架真实调用 | 体现工程落地能力 |
| 数值精度 | FP32/TF32/FP16/BF16/FP8/INT8 误差与累加 | 大模型推理优化必须考虑性能和精度 |
| Benchmark 方法学 | 正确性、稳定性、公平对比、统计 | 简历数据和面试结论必须可信 |

## 7.2 第二优先级：大厂竞争力关键

### CUTLASS 与 CuTe

需要掌握：

- CUTLASS 的 GEMM 分层；
- Mainloop、Epilogue；
- Threadblock/Warp/Instruction Tile；
- Layout、Stride、Tensor；
- Tiled Copy、Tiled MMA；
- CuTe 的 Layout Algebra 基础；
- 用 Profiler 选择和比较 Kernel。

不要求一开始读完全部模板源码，但要能：

- 修改一个官方 GEMM 示例；
- 改数据类型、Layout、Tile 和 Epilogue；
- 解释 CUTLASS 实现和手写 Kernel 的对应关系。

### TensorRT / TensorRT-LLM 或推理框架算子接入

至少理解：

- Plugin/Custom Layer；
- Dynamic Shape；
- Workspace；
- Tactic/Kernel Selection；
- Layer Fusion；
- Quantization；
- TensorRT-LLM 或 vLLM/SGLang 中算子的位置。

结合你已有的 nano-vLLM/vLLM 学习经历，建议把 GEMM 或 RMSNorm 自定义算子接入一个简化推理流程，展示端到端效果，而不是只展示单 Kernel GFLOPS。

### LLM 常见算子和工作负载

需要能分析：

- QKV Projection；
- Attention/FlashAttention；
- RoPE；
- RMSNorm；
- SwiGLU；
- MLP GEMM；
- LM Head；
- KV Cache 读写；
- Quantization/Dequantization；
- MoE Grouped GEMM。

尤其要理解 Prefill 和 Decode 的 Shape 完全不同：

- Prefill 往往更偏大矩阵、计算密集；
- Decode 常出现 M 很小的 GEMV/Small GEMM，更容易受带宽、Launch 和调度影响。

这也是为什么只测试 Square GEMM 不足以代表真实大模型性能。

## 7.3 第三优先级：冲击更高阶岗位

可以根据岗位方向选择：

- `torch.compile`、TorchInductor；
- MLIR、LLVM、TVM；
- TileLang；
- CUDA Graph；
- Persistent Kernel；
- Multi-CTA/Thread Block Cluster；
- TMA；
- NCCL 与通信算子；
- 多 GPU Tensor Parallel；
- AscendC 或其他异构平台；
- SASS/反汇编、指令吞吐和 Scheduler 分析。

这些不应在 CUDA 基础之前学习。正确顺序是：

```text
手写 CUDA Kernel
→ Nsight 定位瓶颈
→ Triton 快速实现
→ CUTLASS/CuTe 高性能模板
→ PyTorch/推理框架接入
→ 编译器和多硬件方向
```

---

# 8. 最适合你的算子项目组合

为了应聘 AI Infra / 大模型推理算子岗位，建议最终形成三个相互补充的项目点：

## 项目 A：本仓库——FP32 GEMM 深度优化

证明：

- CUDA 基础；
- 多级 Tiling；
- Shared Memory/Register 优化；
- `cp.async` Pipeline；
- Nsight 性能分析；
- Dispatch 和 Edge Handling。

## 项目 B：混合精度 Tensor Core GEMM

证明：

- FP16/BF16/FP8 或 INT8；
- WMMA/CUTLASS/CuTe；
- Tensor Core；
- 精度—性能权衡；
- Fused Epilogue。

## 项目 C：一个真实 LLM 融合算子

推荐 RMSNorm、Fused Softmax 或 RoPE：

- CUDA 与 Triton 双实现；
- PyTorch Custom Op；
- 与 PyTorch 原生实现比较；
- 在 nano-vLLM/vLLM 的真实 Shape 上测试；
- 展示单算子和端到端收益。

这三个项目组合比“只手写一个 SGEMM”更符合大模型推理岗位。

---

# 9. 简历项目应该如何形成证据链

最终简历项目必须能够回答：

1. 原始 Baseline 是什么？
2. 你做了哪些优化？
3. 每个优化解决什么瓶颈？
4. Nsight 指标发生了什么变化？
5. 性能在哪些 Shape 上提升？
6. 哪些 Shape 退化？为什么？
7. 与 cuBLAS 比较是否公平？
8. 如何保证正确性？
9. GPU、CUDA、编译参数是什么？
10. 该算子如何接入 PyTorch/推理框架？

### 推荐最终指标结构

不要只写一个“达到 cuBLAS XX%”，而应使用：

```text
测试平台：GPU / CUDA / Driver / 编译参数
测试集合：Aligned Large、Small、Irregular、LLM Shapes
正确性：N 个随机 Case，最大绝对/相对误差，Compute Sanitizer
性能：Median GFLOPS、P50/P90 cuBLAS Ratio、Peak Ratio
稳定性：多次运行标准差
关键优化：优化前后 Kernel Duration、DRAM/SM Throughput、Occupancy、Registers
端到端：PyTorch 或推理模型中的实际耗时变化
```

### 当前仓库数据的正确表述方式

可以表述为：

- 仓库保存的 5 次稳定实验中，所选 Dispatch 策略在既定尺寸集合上的平均 cuBLAS 吞吐比约为 74.6%；
- 部分中大型对齐尺寸接近或达到 cuBLAS，但小型不规则尺寸存在显著性能差距；
- 后续通过 Shape-aware Dispatch、Tail Path 和专用 Epilogue 优化最坏尺寸。

不能表述为：

- “全面超过 cuBLAS”；
- “所有尺寸达到 cuBLAS 75%”；
- “性能提升 74.6%”——0.7456 是相对比值，不是相对原始版本的提升幅度。

---

# 10. 学习过程中的阶段性自测问题

## CUDA 基础阶段

- 为什么一个 Kernel Launch 后 CPU 可以继续执行？
- `cudaDeviceSynchronize` 在什么情况下必要？
- Warp 分支发散如何发生？
- 为什么连续线程访问连续地址更快？

## GEMM 阶段

- BM/BN/BK 分别影响什么？
- 为什么 K 维分块后仍能得到完整结果？
- 每个 A/B 元素在 Block 中被复用多少次？
- 为什么 Thread Tile 会提高 ILP？

## 项目核心阶段

- 默认配置为什么恰好需要 8 个 Warp？
- `warp_row`、`warp_col` 如何计算？
- `trow`、`tcol` 如何映射到二维区域？
- `AS` 为什么按 `[BK][BM]` 方式索引？
- `cp.async` 的目标地址为什么必须先转换为 Shared Address？
- 双缓冲的 `cur/nxt` 如何交替？
- Fast Path 的成立条件是什么？
- Edge Path 为什么不能直接使用同一套向量化加载？
- `res` 过大会产生什么后果？
- Dispatch 为什么不能只按矩阵总元素数选择？

## 性能阶段

- 高 Occupancy 为什么不一定更快？
- Kernel 达不到峰值算力，是计算单元没吃满还是数据供应不足？
- 如何从 Stall Reasons 判断依赖、内存或同步瓶颈？
- 为什么某个优化在 4096 生效，在 257 反而退化？
- 如何证明性能提升不是 GPU 时钟波动？

如果这些问题中超过三分之一无法回答，应回到对应阶段补学习。

---

# 11. 推荐学习资料顺序

按以下顺序学习，避免一上来直接读 CUTLASS 大量模板：

1. NVIDIA CUDA Programming Guide：执行模型、内存层次、同步、异步编程；
2. NVIDIA CUDA Best Practices Guide：合并访存、Occupancy、性能原则；
3. CUDA Samples：Vector Add、Matrix Multiply、Transpose、Reduction；
4. Nsight Compute 官方 Profiling Guide：Roofline、Occupancy、Stall；
5. 本项目：从 Naive GEMM 对照到 Warp Tile Kernel；
6. Triton 官方 Tutorials：Vector Add、Fused Softmax、Matrix Multiplication；
7. CUTLASS Efficient GEMM 与 GEMM API；
8. CuTe Layout/Tiled Copy/Tiled MMA；
9. PyTorch Custom C++ and CUDA Operators；
10. 将算子接入 nano-vLLM/vLLM 的真实调用路径。

---

# 12. 8 周压缩版本

只有在每天能够稳定投入 6 小时以上时使用：

| 周次 | 内容 |
|---|---|
| 第 1 周 | C++ 必备基础 + CUDA 执行模型 + Vector Add/SAXPY |
| 第 2 周 | 内存层次、Transpose、Reduction、Nsight 基础 |
| 第 3 周 | Naive GEMM + Shared Memory Tiling + cuBLAS 对照 |
| 第 4 周 | Register/Thread Tiling + Warp Tiling + 默认配置推导 |
| 第 5 周 | `float4`、Shared Memory 布局、Edge Path |
| 第 6 周 | `cp.async`、双缓冲、完整精读项目 |
| 第 7 周 | 完善正确性、Benchmark、Nsight 证据链 |
| 第 8 周 | 优化 Irregular Shape + PyTorch Extension + 简历报告 |

压缩路线只能压缩课程时间，不能取消实践和验收。

---

# 13. 最终学习优先级结论

## 现在立刻学习

```text
C++ 指针/模板/CMake
→ CUDA 执行模型
→ CUDA 内存层次
→ Naive GEMM
→ Shared Memory Tiling
→ Register/Thread Tiling
→ Warp Tiling
→ Nsight Compute
→ cp.async 双缓冲
→ 完整精读并改造本项目
```

## 把本项目写进简历前补齐

```text
随机正确性测试
+ 多种矩形/非对齐 Shape
+ Compute Sanitizer
+ 公平且稳定的 Benchmark
+ Nsight 指标证据
+ 修复架构硬编码和 beta==0 路径
+ 至少一个可量化优化
+ PyTorch Custom Op 接入
```

## 面向大厂继续学习

```text
Triton
+ CUTLASS/CuTe
+ Tensor Core 与低精度 GEMM
+ RMSNorm/Softmax/Attention 等真实 LLM 算子
+ PyTorch/torch.compile/推理框架接入
+ TensorRT/TensorRT-LLM
+ 可选的 MLIR/TVM/TileLang/NCCL
```

最关键的判断是：**本项目可以成为很好的 CUDA GEMM 入门到进阶项目，但仅凭当前版本还不足以证明成熟的大模型算子工程能力。** 你需要把它从“能运行的 FP32 GEMM Kernel”升级成“正确性充分、性能分析可信、可以接入框架、覆盖真实 AI Shape，并具有混合精度或融合算子扩展”的完整项目。

---

# 参考依据（截至 2026 年 7 月）

- NVIDIA CUDA Programming Guide：CUDA 编程模型、内存模型和异步编程。
- NVIDIA Nsight Compute Documentation / Profiling Guide：Roofline、Occupancy、Kernel 指标分析。
- NVIDIA CUTLASS 官方仓库与 Efficient GEMM 文档：GEMM 分层、CuTe、Tensor Core Kernel。
- Triton 官方 Tutorials：Fused Softmax、Matrix Multiplication、Persistent Matmul、Block-scaled Matmul。
- PyTorch 官方 Custom C++ and CUDA Operators：自定义 CUDA 算子注册与 `torch.compile` 兼容。
- 2026 年大模型训练/推理优化和 GPU 底层优化岗位信息：普遍强调 CUDA C++/Triton、GPU 架构、CUTLASS、低精度、算子调优、TensorRT、vLLM/SGLang、图编译等能力。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-gemm算子|模块-gemm算子]]

%% 项目关联导航：结束 %%
