# gemm_CMakeLists_精读

```cmake
# 指定当前项目要求的最低 CMake 版本为 3.20。
#
# 语法：
#   cmake_minimum_required(VERSION <最低版本>)
#
# 作用：
#   1. 如果用户安装的 CMake 版本低于 3.20，配置阶段会直接报错；
#   2. 让 CMake 按照 3.20 版本对应的策略规则处理该项目；
#   3. CUDA 架构、生成器表达式等功能对 CMake 版本有要求，
#      因此 CUDA 项目通常需要明确指定一个较新的最低版本。
cmake_minimum_required(VERSION 3.20)


# 定义项目名称，并声明项目所使用的编程语言。
#
# 语法：
#   project(<项目名> LANGUAGES <语言1> <语言2> ...)
#
# 参数含义：
#   gemm_project：
#       当前 CMake 工程的名称。
#
#   LANGUAGES CXX CUDA：
#       表示该项目同时包含 C++ 和 CUDA 源代码。
#
# 声明 CXX 后：
#   CMake 会检测系统中的 C++ 编译器，例如 g++。
#
# 声明 CUDA 后：
#   CMake 会检测 CUDA 编译器 nvcc，并启用 .cu 文件的编译支持。
#
# 注意：
#   CUDA 并不是简单地作为一个普通库被链接进来。
#   在这里声明 CUDA，意味着 CUDA 是项目的一种正式编程语言，
#   CMake 会专门为 CUDA 源文件生成编译规则。
project(gemm_project LANGUAGES CXX CUDA)


# 设置 C++ 源码使用 C++17 标准。
#
# CMAKE_CXX_STANDARD 是 CMake 的内置变量。
#
# 设置后，CMake 通常会在编译 C++ 文件时加入类似参数：
#   -std=c++17
#
# 本项目中的 .cpp 文件，例如：
#   tests/test_perf_gemm.cpp
# 会按照 C++17 标准编译。
set(CMAKE_CXX_STANDARD 17)


# 设置 CUDA 源码中的 C++ 语言标准为 C++17。
#
# CUDA 源文件虽然以 .cu 结尾，但其中通常同时包含：
#   1. 普通 C++ 主机端代码；
#   2. __global__ CUDA 核函数；
#   3. __device__ 设备函数；
#   4. CUDA Runtime API 调用。
#
# 设置该变量后，nvcc 会以 C++17 标准编译 .cu 文件。
set(CMAKE_CUDA_STANDARD 17)


# 要求 C++17 必须得到编译器支持。
#
# ON 是 CMake 中的布尔值，表示开启。
#
# 如果不设置 REQUIRED：
#   当编译器不支持 C++17 时，CMake 可能尝试退回较低标准。
#
# 设置为 ON 后：
#   如果当前 C++ 编译器不支持 C++17，CMake 会直接报错，
#   而不是静默降低语言标准。
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 要求 CUDA 编译器必须支持 CUDA C++17。
#
# 该设置主要作用于 nvcc 编译 .cu 文件的过程。
#
# 如果当前 CUDA Toolkit 或宿主编译器组合无法支持 C++17，
# 配置或编译阶段会失败。
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


# 指定 CUDA 代码所面向的 GPU Compute Capability。
#
# 89 对应 Compute Capability 8.9。
#
# CMake 会根据这个值，为 nvcc 生成相应的架构编译参数，
# 效果类似于：
#   -gencode arch=compute_89,code=sm_89
#
# 其中：
#   compute_89：
#       表示 PTX 虚拟指令集目标。
#
#   sm_89：
#       表示为 Compute Capability 8.9 GPU 生成实际机器码。
#
# 该设置的重要性：
#   CUDA 程序并不是对所有 GPU 自动通用。
#   不同 GPU 架构支持的指令、Tensor Core 能力、调度特征不同。
#
# 如果目标 GPU 架构和这里不匹配：
#   1. 可能无法运行；
#   2. 可能依赖 PTX JIT 编译；
#   3. 可能不能发挥目标 GPU 的最佳性能。
#
# 注意：
#   该值应该根据实际 GPU 型号确定。
set(CMAKE_CUDA_ARCHITECTURES 89)


# 判断用户是否已经指定构建类型。
#
# CMAKE_BUILD_TYPE 是单配置生成器中常用的变量，
# 常见取值包括：
#   Debug
#   Release
#   RelWithDebInfo
#   MinSizeRel
#
# NOT 表示逻辑取反。
#
# 因此：
#   if(NOT CMAKE_BUILD_TYPE)
# 表示：
#   如果用户没有设置 CMAKE_BUILD_TYPE，则执行 if 块内的语句。
if(NOT CMAKE_BUILD_TYPE)

    # 将默认构建类型设置为 Release。
    #
    # 完整语法：
    #   set(<变量名> <值> CACHE <类型> <说明文字> FORCE)
    #
    # 参数解释：
    #
    #   CMAKE_BUILD_TYPE：
    #       要设置的变量。
    #
    #   Release：
    #       默认值。Release 模式通常启用编译优化，并关闭大部分调试信息。
    #
    #   CACHE：
    #       表示将该变量写入 CMake 缓存。
    #       CMake 缓存通常保存在构建目录中的 CMakeCache.txt 文件中。
    #
    #   STRING：
    #       表示该缓存变量的类型是字符串。
    #
    #   "Build type"：
    #       该缓存变量的说明文字，供 CMake GUI 或其他工具显示。
    #
    #   FORCE：
    #       强制覆盖缓存中的同名变量。
    #
    # 由于外层已经判断：
    #   只有在 CMAKE_BUILD_TYPE 没有设置时才会执行，
    # 因此这里的 FORCE 不会覆盖用户已经显式传入的构建类型。
    #
    # 用户也可以在配置时手动指定：
    #   cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

# 结束 if 条件块。
endif()


# 查找本机安装的 CUDA Toolkit。
#
# 语法：
#   find_package(<包名> REQUIRED)
#
# CUDAToolkit：
#   是 CMake 提供的 CUDA Toolkit 查找模块。
#
# 查找成功后，会提供一系列可供链接的导入目标，例如：
#   CUDA::cudart
#   CUDA::cublas
#
# REQUIRED：
#   表示 CUDA Toolkit 是必需依赖。
#   如果没有找到 CUDA Toolkit，CMake 配置过程会直接失败。
#
# 注意：
#   project(... LANGUAGES CUDA) 负责启用 CUDA 编译语言；
#   find_package(CUDAToolkit REQUIRED) 负责查找 CUDA Runtime、
#   cuBLAS 等 CUDA 工具包中的库。
find_package(CUDAToolkit REQUIRED)


# 创建一个名为 gemm_lib 的静态库。
#
# 语法：
#   add_library(<目标名> STATIC <源文件列表>)
#
# 参数解释：
#
#   gemm_lib：
#       CMake 目标名称。
#       后续可以通过 target_link_libraries 将它链接到其他目标。
#
#   STATIC：
#       表示生成静态库。
#
# 在 Linux 下通常生成：
#   libgemm_lib.a
#
# 在 Windows 下通常生成：
#   gemm_lib.lib
#
# 该库由下面两个 CUDA 源文件共同编译生成。
add_library(gemm_lib STATIC

    # launcher.cu 通常负责主机端的 CUDA kernel 启动封装，
    # 例如设置 grid、block、共享内存大小，并调用核函数。
    src/launcher.cu

    # gemm_warp_tile.cu 通常包含基于 warp tile 的 GEMM CUDA 核函数实现。
    src/kernels/gemm_warp_tile.cu
)


# 为 gemm_lib 设置头文件搜索目录。
#
# 现代 CMake 推荐使用：
#   target_include_directories
# 而不是全局的 include_directories。
#
# 这样可以将头文件路径限制在具体目标上，
# 避免污染整个项目。
target_include_directories(gemm_lib

    # PUBLIC 表示该头文件目录具有传播性。
    #
    # 对 gemm_lib 自身：
    #   编译 gemm_lib 时可以搜索该目录。
    #
    # 对链接 gemm_lib 的其他目标：
    #   也会自动继承该头文件搜索目录。
    #
    # 因此，test_perf_gemm 链接 gemm_lib 后，
    # 理论上也可以继承该 include 目录。
    PUBLIC

        # CMAKE_SOURCE_DIR 是 CMake 内置变量，
        # 表示最顶层 CMakeLists.txt 所在的源码目录。
        #
        # ${变量名} 是 CMake 的变量展开语法。
        #
        # 假设项目路径为：
        #   /home/user/gemm-project
        #
        # 那么该表达式会展开为：
        #   /home/user/gemm-project/include
        #
        # 编译器最终会收到类似参数：
        #   -I/home/user/gemm-project/include
        ${CMAKE_SOURCE_DIR}/include
)


# 为 gemm_lib 设置链接依赖。
#
# 语法：
#   target_link_libraries(<目标名> <可见性> <依赖目标...>)
target_link_libraries(gemm_lib

    # PUBLIC 表示这些依赖不仅供 gemm_lib 自身使用，
    # 还会传播给链接 gemm_lib 的下游目标。
    PUBLIC

        # CUDA::cudart 是 CMake 创建的 CUDA Runtime 导入目标。
        #
        # cudart 即 CUDA Runtime Library。
        #
        # 常见 CUDA Runtime API 包括：
        #   cudaMalloc
        #   cudaFree
        #   cudaMemcpy
        #   cudaEventCreate
        #   cudaDeviceSynchronize
        #   kernel<<<grid, block>>>()
        #
        # 链接该目标后，程序才能正常使用这些 CUDA Runtime 功能。
        CUDA::cudart

        # CUDA::cublas 是 cuBLAS 库的 CMake 导入目标。
        #
        # cuBLAS 是 NVIDIA 提供的 GPU 线性代数库，
        # 可用于执行高性能矩阵乘法。
        #
        # GEMM 项目通常会把自定义 CUDA GEMM kernel
        # 与 cuBLAS 的执行结果和性能进行对比。
        CUDA::cublas
)


# 为 gemm_lib 设置编译选项。
#
# PRIVATE 表示这些编译选项只用于编译 gemm_lib，
# 不会传播给链接 gemm_lib 的其他目标。
target_compile_options(gemm_lib PRIVATE

    # $<...> 是 CMake 的生成器表达式。
    #
    # 生成器表达式不会在读取 CMakeLists.txt 时立即求值，
    # 而是在生成具体构建规则时，根据目标、语言、配置等条件求值。
    #
    # 语法：
    #   $<$<条件>:条件成立时使用的内容>
    #
    # $<COMPILE_LANGUAGE:CUDA>：
    #   当当前正在编译的源文件语言是 CUDA 时，条件为真。
    #
    # 因此，-O3 只会传给 CUDA 编译器，不会传给普通 C++ 编译器。
    #
    # -O3：
    #   开启较高级别的编译优化。
    #
    # 常见优化包括：
    #   1. 函数内联；
    #   2. 循环优化；
    #   3. 常量传播；
    #   4. 删除无用代码；
    #   5. 指令重排。
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>

    # 仅在编译 CUDA 文件时加入 --use_fast_math。
    #
    # --use_fast_math 会让 nvcc 使用速度更快、
    # 但精度或 IEEE 一致性可能略低的数学实现。
    #
    # 它通常会影响：
    #   除法
    #   平方根
    #   三角函数
    #   某些浮点运算
    #
    # 对 GEMM 而言，核心乘加通常仍由普通浮点指令执行，
    # 但该选项可能改变某些辅助数学运算和浮点语义。
    #
    # 性能测试项目中常使用该参数，
    # 但进行数值正确性验证时需要注意误差容限。
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>

    # 仅在编译 CUDA 文件时加入 -lineinfo。
    #
    # -lineinfo 会在生成的 GPU 代码中保留源码行号信息，
    # 但不像完整 Debug 模式那样关闭优化。
    #
    # 这样可以在使用以下性能分析工具时，
    # 将 GPU 指令或性能热点对应回源码行：
    #   Nsight Compute
    #   Nsight Systems
    #
    # 该选项非常适合：
    #   Release 优化构建 + 性能分析
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)


# 创建一个可执行程序，目标名称为 test_perf_gemm。
#
# 语法：
#   add_executable(<目标名> <源文件列表>)
#
# 该目标由 tests/test_perf_gemm.cpp 编译生成。
#
# 最终通常得到：
#   Linux：test_perf_gemm
#   Windows：test_perf_gemm.exe
add_executable(test_perf_gemm

    # 该文件通常用于：
    #   1. 构造测试矩阵；
    #   2. 调用自定义 GEMM；
    #   3. 调用 cuBLAS 作为基准；
    #   4. 检查计算正确性；
    #   5. 测量执行时间、吞吐量或 TFLOPS。
    tests/test_perf_gemm.cpp
)


# 为 test_perf_gemm 单独设置头文件搜索目录。
target_include_directories(test_perf_gemm

    # PRIVATE 表示该 include 路径仅用于 test_perf_gemm 自身，
    # 不会继续传播给其他目标。
    PRIVATE

        # 将项目根目录下的 include 文件夹加入头文件搜索路径。
        ${CMAKE_SOURCE_DIR}/include
)


# 设置 test_perf_gemm 的链接依赖。
target_link_libraries(test_perf_gemm

    # PRIVATE 表示这些链接依赖只属于 test_perf_gemm，
    # 不向其他目标传播。
    PRIVATE

        # 链接项目自定义的 GEMM 静态库。
        #
        # 链接该目标后，test_perf_gemm 才能调用
        # launcher.cu 和 gemm_warp_tile.cu 中实现的函数。
        gemm_lib

        # 显式链接 CUDA Runtime。
        #
        # 由于 gemm_lib 已经以 PUBLIC 方式链接 CUDA::cudart，
        # test_perf_gemm 理论上可以通过依赖传播获得 cudart。
        #
        # 这里再次显式写出，可以让依赖关系更直观，
        # 但从现代 CMake 的依赖传播角度看存在一定重复。
        CUDA::cudart

        # 显式链接 cuBLAS。
        #
        # 测试程序通常会直接调用 cuBLAS API，
        # 因此显式链接可以清楚表达测试程序本身依赖 cuBLAS。
        CUDA::cublas
)


# 为 test_perf_gemm 设置编译参数。
target_compile_options(test_perf_gemm PRIVATE

    # 仅当当前编译语言为 CXX 时加入 -O3。
    #
    # CXX 表示普通 C++ 源文件。
    #
    # test_perf_gemm 的源文件是：
    #   tests/test_perf_gemm.cpp
    #
    # 因此该生成器表达式条件成立，
    # 编译器会使用 -O3 编译该测试程序。
    #
    # 这里不会影响 .cu 文件；
    # .cu 文件的优化参数已经在 gemm_lib 中单独设置。
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
```


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-gemm算子|模块-gemm算子]]

%% 项目关联导航：结束 %%
