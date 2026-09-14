# gemm_launcher_精读

本文按照 `launcher.cu` 的代码顺序拆分源码。每一节先给出一块完整源代码，再解释其中涉及的 C++ 语法、枚举、函数调用、参数传递和分发逻辑。

---

## 1. 引入项目头文件

```cpp
#include "gemm/launcher.h"
#include "gemm/kernels/gemm_warp_tile.h"
```

### 详细解释

`#include` 是 C/C++ 的预处理指令。

在真正编译代码之前，预处理器会先处理所有以 `#` 开头的指令。对于：

```cpp
#include "gemm/launcher.h"
```

可以暂时理解为：把 `launcher.h` 中提供的声明引入当前 `.cu` 文件，使这里能够使用其中声明的类型、枚举和函数。

这里使用双引号：

```cpp
#include "..."
```

一般表示这是当前项目自己的头文件，而不是 C++ 标准库或系统头文件。

---

### 1.1 `launcher.h`

```cpp
#include "gemm/launcher.h"
```

根据当前文件后面的代码，这个头文件至少应当声明了两类内容：

```cpp
GemmAlgo
```

即 GEMM 算法类型的枚举。

以及：

```cpp
launch_gemm(...)
```

即项目向外暴露的统一 GEMM 调用接口。

当前 `.cu` 文件负责给 `launch_gemm` 提供具体实现。

头文件和源文件的常见关系是：

```text
launcher.h
    声明函数“长什么样”

launcher.cu
    实现函数“具体怎么做”
```

其他文件只需要包含 `launcher.h`，就能够调用：

```cpp
launch_gemm(...)
```

而不需要知道其内部如何选择具体 Kernel。

---

### 1.2 `gemm_warp_tile.h`

```cpp
#include "gemm/kernels/gemm_warp_tile.h"
```

这个头文件应当声明：

```cpp
launch_gemm_warp_tile(...)
```

当前 `launcher.cu` 并没有直接写：

```cpp
kernel<<<grid, block>>>(...)
```

而是继续调用更下层的启动函数：

```cpp
launch_gemm_warp_tile(...)
```

调用层次可以先理解为：

```text
测试程序
    ↓
launch_gemm
    ↓
launch_gemm_warp_tile
    ↓
真正的 CUDA Kernel
```

---

### 1.3 为什么文件后缀是 `.cu`

当前文件名是：

```text
launcher.cu
```

`.cu` 表示该文件由 NVIDIA CUDA 编译器 `nvcc` 处理。

即使这个文件中没有直接出现：

```cpp
__global__
__device__
kernel<<<...>>>()
```

它仍然可以作为 CUDA 编译单元，因为它调用的下层接口与 CUDA Kernel 启动有关，并且项目构建系统将其作为 CUDA 源文件处理。

---

## 2. 匿名命名空间开始

```cpp
namespace {
```

### 详细解释

`namespace` 的中文一般叫“命名空间”。

普通的命名空间可以有名字，例如：

```cpp
namespace gemm {
    void func();
}
```

调用时需要写：

```cpp
gemm::func();
```

当前代码写的是：

```cpp
namespace {
```

花括号前没有名字，这叫做**匿名命名空间**。

---

### 2.1 匿名命名空间的作用

匿名命名空间中的变量和函数只在当前源文件中可见。

也就是说，后面定义的：

```cpp
select_gemm_algo(...)
```

只能在当前 `launcher.cu` 中使用。

其他 `.cpp` 或 `.cu` 文件不能直接调用它。

可以把它理解为：

```text
这是 launcher.cu 自己内部使用的辅助函数，
不是项目对外提供的公共接口。
```

---

### 2.2 为什么不把它写到头文件中

`select_gemm_algo` 只是 `launch_gemm` 内部使用的算法选择辅助函数。

外部调用者只需要知道：

```cpp
launch_gemm(...)
```

不需要知道内部具体如何选择算法。

因此把它放在匿名命名空间中可以：

- 限制可见范围；
- 避免其他文件误调用；
- 避免和其他文件中的同名函数冲突；
- 表明它是当前实现文件的内部细节。

---

### 2.3 与 `static` 的关系

在源文件中写：

```cpp
static GemmAlgo select_gemm_algo(...);
```

也可以让该函数只在当前翻译单元中可见。

现代 C++ 通常更推荐匿名命名空间，因为它不仅可以限制函数，还可以限制类、变量和其他类型的可见性。

---

## 3. 算法选择辅助函数

```cpp
inline GemmAlgo select_gemm_algo(const int M, const int N, const int K) {
    return GemmAlgo::WarpTile;
}
```

### 详细解释

这是一个普通的 C++ 函数定义。

完整结构是：

```cpp
返回类型 函数名(参数列表) {
    函数体
}
```

对应到当前代码：

```cpp
inline GemmAlgo
```

是函数返回类型和修饰符。

```cpp
select_gemm_algo
```

是函数名。

```cpp
const int M, const int N, const int K
```

是参数列表。

---

### 3.1 `GemmAlgo` 返回类型

```cpp
GemmAlgo
```

表示该函数返回一个 `GemmAlgo` 类型的值。

`GemmAlgo` 大概率是一个枚举类型，例如可能定义成：

```cpp
enum class GemmAlgo {
    Auto,
    WarpTile
};
```

枚举的作用是用有含义的名字表示一组离散选项。

相比直接使用整数：

```cpp
0
1
```

枚举写成：

```cpp
GemmAlgo::Auto
GemmAlgo::WarpTile
```

更容易阅读，也不容易传错。

---

### 3.2 `inline`

```cpp
inline
```

是 C++ 关键字。

初学时可以先把它理解为：

```text
这个函数很短，编译器可以考虑把函数体直接展开到调用位置。
```

例如原本调用：

```cpp
final_algo = select_gemm_algo(M, N, K);
```

编译器可能优化成近似：

```cpp
final_algo = GemmAlgo::WarpTile;
```

但必须注意：

> `inline` 并不保证编译器一定会进行函数展开。

是否真正内联，最终由编译器根据优化策略决定。

在当前代码中，这个函数非常短，只返回一个枚举值，因此很适合作为 `inline` 辅助函数。

---

### 3.3 `const int M`

```cpp
const int M
```

表示参数 `M` 是按值传入的整数，并且函数内部不能重新给这个形参赋值。

例如在函数中不允许写：

```cpp
M = 128;
```

因为 `M` 带有 `const`。

但是调用者传进来的变量本身不会受到影响，因为这里是按值传递。

例如：

```cpp
int x = 1024;
select_gemm_algo(x, x, x);
```

函数中的 `M`、`N`、`K` 都只是 `x` 的副本。

---

### 3.4 当前参数实际上没有被使用

函数接收：

```cpp
M
N
K
```

但函数体只有：

```cpp
return GemmAlgo::WarpTile;
```

因此当前版本没有根据矩阵尺寸真正做选择。

无论传入：

```text
M=100, N=100, K=100
```

还是：

```text
M=8192, N=8192, K=8192
```

返回值都一样：

```cpp
GemmAlgo::WarpTile
```

这些参数保留下来，是为了以后扩展。

例如将来可能写成：

```cpp
if (M <= 128 && N <= 128) {
    return GemmAlgo::SmallTile;
}

if (M % 128 != 0 || N % 128 != 0) {
    return GemmAlgo::Irregular;
}

return GemmAlgo::WarpTile;
```

这样就可以根据形状选择不同算法。

---

### 3.5 `return`

```cpp
return GemmAlgo::WarpTile;
```

`return` 会：

1. 结束当前函数；
2. 把右侧值返回给调用者。

调用处：

```cpp
final_algo = select_gemm_algo(M, N, K);
```

最终会得到：

```cpp
final_algo = GemmAlgo::WarpTile;
```

---

### 3.6 `GemmAlgo::WarpTile` 中的 `::`

```cpp
GemmAlgo::WarpTile
```

`::` 是作用域解析运算符。

它表示：

```text
WarpTile 是 GemmAlgo 作用域中的成员。
```

如果 `GemmAlgo` 是 `enum class`，通常必须使用完整写法：

```cpp
GemmAlgo::WarpTile
```

而不能只写：

```cpp
WarpTile
```

这样能减少不同枚举成员之间的命名冲突。

---

## 4. 匿名命名空间结束

```cpp
}
```

### 详细解释

这个右花括号结束前面的匿名命名空间：

```cpp
namespace {
    ...
}
```

因此：

```cpp
select_gemm_algo
```

位于匿名命名空间内部。

而后面的：

```cpp
launch_gemm
```

位于全局命名空间中。

这体现了两类函数的不同身份：

```text
select_gemm_algo
    launcher.cu 内部辅助函数

launch_gemm
    项目对外提供的公共函数
```

---

## 5. `launch_gemm` 函数定义

```cpp
void launch_gemm(const int M,
                 const int N,
                 const int K,
                 const float alpha,
                 const float* A,
                 const float* B,
                 const float beta,
                 float* C,
                 GemmAlgo algo) {
```

### 详细解释

这是当前文件最核心的函数定义。

函数名：

```cpp
launch_gemm
```

表示启动 GEMM 计算。

返回类型：

```cpp
void
```

表示函数不返回普通计算值。

GEMM 的结果不是通过 `return` 返回，而是写入参数：

```cpp
float* C
```

所指向的 GPU 显存。

---

### 5.1 GEMM 的数学形式

这些参数共同表达：

```text
C = alpha × A × B + beta × C
```

矩阵形状是：

```text
A：M × K
B：K × N
C：M × N
```

---

### 5.2 `M`、`N`、`K`

```cpp
const int M,
const int N,
const int K,
```

三者表示矩阵维度：

- `M`：A 和 C 的行数；
- `N`：B 和 C 的列数；
- `K`：A 的列数，也是 B 的行数。

例如：

```text
M = 128
N = 256
K = 64
```

则：

```text
A：128 × 64
B：64 × 256
C：128 × 256
```

这些参数按值传递，并被声明为 `const`，所以函数内部不能修改形参副本。

---

### 5.3 `alpha`

```cpp
const float alpha
```

`alpha` 是矩阵乘积的缩放系数。

例如：

```text
alpha = 2
```

表示先计算 `A×B`，再把结果整体乘以 2。

当前性能测试中通常传：

```cpp
alpha = 1.0f
```

所以不会改变矩阵乘积。

---

### 5.4 `const float* A`

```cpp
const float* A
```

这是一个指向 `float` 的指针。

它保存矩阵 A 的起始地址。

前面的 `const` 修饰指针指向的数据，表示函数不应通过 A 修改输入矩阵内容。

不允许写：

```cpp
A[0] = 3.0f;
```

但可以让局部指针变量 A 指向别处，因为这里不是：

```cpp
float* const A
```

当前调用中，A 通常是 GPU 显存指针，例如：

```cpp
d_A
```

因此这个函数不会在 CPU 上直接读取 A，而是继续把该设备指针传给下层 Kernel 启动函数。

---

### 5.5 `const float* B`

```cpp
const float* B
```

与 A 相同，是只读输入矩阵 B 的 GPU 地址。

---

### 5.6 `beta`

```cpp
const float beta
```

`beta` 用于控制原矩阵 C 的贡献：

```text
C_new = alpha × A × B + beta × C_old
```

如果：

```cpp
beta = 0
```

则旧 C 被忽略。

如果：

```cpp
beta = 1
```

则矩阵乘积会累加到原 C 上。

---

### 5.7 `float* C`

```cpp
float* C
```

C 是可写指针。

函数最终要把 GEMM 结果写入这块内存，所以不能写成：

```cpp
const float* C
```

当前它通常指向 GPU 显存：

```cpp
d_C
```

---

### 5.8 `GemmAlgo algo`

```cpp
GemmAlgo algo
```

表示调用者希望使用的算法。

可能的取值包括：

```cpp
GemmAlgo::Auto
GemmAlgo::WarpTile
```

如果调用者传入：

```cpp
GemmAlgo::WarpTile
```

表示明确指定 Warp Tile 算法。

如果传入：

```cpp
GemmAlgo::Auto
```

表示让 `launcher.cu` 自己选择。

在头文件中，这个参数可能带有默认值，例如：

```cpp
GemmAlgo algo = GemmAlgo::Auto
```

默认参数通常只写在函数声明中，不需要在函数定义中再次写出。

因此测试代码可以只写：

```cpp
launch_gemm(M, N, K, alpha, d_A, d_B, beta, d_C);
```

虽然没有传第九个参数，但编译器会自动使用：

```cpp
GemmAlgo::Auto
```

---

### 5.9 为什么 `launch_gemm` 不是 `__global__`

这个函数没有：

```cpp
__global__
```

说明它是普通 Host 函数，在 CPU 上执行。

它本身不是 GPU Kernel。

它的任务是：

```text
检查或选择算法
    ↓
调用具体实现的 launcher
    ↓
由下层 launcher 启动 GPU Kernel
```

---

## 6. 保存最终算法类型

```cpp
    GemmAlgo final_algo = algo;
```

### 详细解释

这行定义了一个局部变量：

```cpp
final_algo
```

类型是：

```cpp
GemmAlgo
```

初始值来自函数参数：

```cpp
algo
```

可以理解为：

```text
先把调用者的选择复制一份，
后面再根据是否为 Auto 决定要不要修改。
```

---

### 6.1 为什么不直接修改 `algo`

代码也可以写成：

```cpp
if (algo == GemmAlgo::Auto) {
    algo = select_gemm_algo(M, N, K);
}
```

但是当前实现使用新变量：

```cpp
final_algo
```

有两个好处：

1. `algo` 表示调用者原始请求；
2. `final_algo` 表示经过自动选择后的最终结果。

语义更清晰。

---

### 6.2 初始化语法

```cpp
GemmAlgo final_algo = algo;
```

这是复制初始化。

等价地也可以写：

```cpp
GemmAlgo final_algo(algo);
```

或：

```cpp
GemmAlgo final_algo{algo};
```

对于枚举类型，复制成本非常低。

---

## 7. 处理自动算法选择

```cpp
    if (final_algo == GemmAlgo::Auto) {
        final_algo = select_gemm_algo(M, N, K);
    }
```

### 详细解释

`if` 用于条件分支。

基本形式是：

```cpp
if (条件) {
    条件成立时执行的代码
}
```

当前条件：

```cpp
final_algo == GemmAlgo::Auto
```

`==` 表示“是否相等”。

不要和赋值运算符混淆：

```cpp
=   赋值
==  比较是否相等
```

---

### 7.1 调用者指定具体算法时

如果调用者传入：

```cpp
GemmAlgo::WarpTile
```

那么：

```cpp
final_algo == GemmAlgo::Auto
```

为假，花括号内代码不会执行。

最终保持：

```cpp
final_algo = GemmAlgo::WarpTile
```

---

### 7.2 调用者选择 Auto 时

如果调用者传入：

```cpp
GemmAlgo::Auto
```

条件成立，执行：

```cpp
final_algo = select_gemm_algo(M, N, K);
```

当前 `select_gemm_algo` 无条件返回：

```cpp
GemmAlgo::WarpTile
```

所以执行后：

```cpp
final_algo = GemmAlgo::WarpTile
```

---

### 7.3 当前两种入口的最终结果

当前实现中：

```text
调用者传 WarpTile
    ↓
final_algo = WarpTile

调用者传 Auto
    ↓
select_gemm_algo
    ↓
final_algo = WarpTile
```

因此当前所有正常情况最终都会使用：

```cpp
GemmAlgo::WarpTile
```

---

### 7.4 保留 Auto 机制的价值

虽然现在只有一个真正实现，但这种结构方便以后增加：

```cpp
GemmAlgo::Naive
GemmAlgo::SharedMemory
GemmAlgo::WarpTile
GemmAlgo::TensorCore
```

然后根据：

- M、N、K；
- 数据类型；
- 是否对齐；
- GPU 架构；
- 预估资源开销；

自动选择合适的 Kernel。

---

## 8. 根据最终算法进行分发

```cpp
    switch (final_algo) {
```

### 详细解释

`switch` 用于根据一个离散值选择不同分支。

基本结构是：

```cpp
switch (表达式) {
    case 值1:
        ...
        break;

    case 值2:
        ...
        break;

    default:
        ...
        break;
}
```

当前表达式是：

```cpp
final_algo
```

也就是最终确定的 GEMM 算法枚举值。

`switch` 很适合处理枚举，因为每个 `case` 可以对应一种算法。

---

## 9. Warp Tile 分支

```cpp
        case GemmAlgo::WarpTile:
            launch_gemm_warp_tile(M, N, K, alpha, A, B, beta, C);
            break;
```

### 详细解释

```cpp
case GemmAlgo::WarpTile:
```

表示当：

```cpp
final_algo == GemmAlgo::WarpTile
```

时，执行下面的代码。

---

### 9.1 调用具体 Warp Tile 实现

```cpp
launch_gemm_warp_tile(M, N, K, alpha, A, B, beta, C);
```

该函数由：

```cpp
#include "gemm/kernels/gemm_warp_tile.h"
```

提供声明。

它接收与统一接口基本相同的 GEMM 参数，但不再需要 `GemmAlgo`，因为此时已经明确选择了 Warp Tile 实现。

参数被原样继续向下传递：

```text
launch_gemm 的 M     → launch_gemm_warp_tile 的 M
launch_gemm 的 N     → launch_gemm_warp_tile 的 N
launch_gemm 的 K     → launch_gemm_warp_tile 的 K
launch_gemm 的 alpha → 下层 alpha
A、B、beta、C        → 原样传入
```

---

### 9.2 什么是“分发”

这种根据一个枚举值选择具体实现的过程，通常叫：

```text
dispatch
```

中文可以理解为“分发”或“调度”。

统一接口：

```cpp
launch_gemm(...)
```

负责运行时分发。

具体实现：

```cpp
launch_gemm_warp_tile(...)
```

负责某一种算法。

这种设计将调用者和具体 Kernel 解耦。

---

### 9.3 `break`

```cpp
break;
```

表示立即跳出当前 `switch`。

如果没有 `break`，C++ 会继续执行下一个 `case`，这种行为叫：

```text
fall-through
```

例如：

```cpp
case A:
    func_a();

case B:
    func_b();
```

如果进入 A，又没有 `break`，那么 `func_b()` 也会执行。

当前代码显然只希望启动一次 GEMM，所以必须写 `break`。

---

## 10. `Auto` 和默认兜底分支

```cpp
        case GemmAlgo::Auto:
        default:
            launch_gemm_warp_tile(M, N, K, alpha, A, B, beta, C);
            break;
```

### 详细解释

这里有两个标签连续出现：

```cpp
case GemmAlgo::Auto:
default:
```

二者之间没有任何语句，所以它们共享同一段处理逻辑。

也就是说，以下两种情况都会执行：

```cpp
launch_gemm_warp_tile(...)
```

---

### 10.1 `case GemmAlgo::Auto`

理论上，前面的代码已经处理了：

```cpp
if (final_algo == GemmAlgo::Auto)
```

因此正常情况下进入 `switch` 前，`final_algo` 不应再是 `Auto`。

但代码仍保留该分支作为防御性处理。

如果未来 `select_gemm_algo` 错误地返回 `Auto`，程序仍然可以退回 Warp Tile。

---

### 10.2 `default`

```cpp
default:
```

表示当 `final_algo` 不匹配任何已列出的 `case` 时，执行默认分支。

例如以后枚举新增：

```cpp
GemmAlgo::TensorCore
```

但忘记在当前 `switch` 中添加对应 `case`，那么它会落入 `default`。

当前默认行为是：

```cpp
launch_gemm_warp_tile(...)
```

也就是兜底使用已有实现，而不是完全不执行计算。

---

### 10.3 为什么 `case Auto` 和 `default` 能连在一起

标签本身不会自动结束分支。

下面写法合法：

```cpp
case A:
case B:
default:
    do_something();
    break;
```

表示 A、B 和其他未匹配情况都执行同一段代码。

---

### 10.4 当前兜底策略的特点

好处：

- 即使算法类型异常，也会尝试完成 GEMM；
- 代码不会因为漏写一个 `case` 而完全不输出结果；
- 当前只有 Warp Tile 实现，兜底逻辑简单。

风险：

- 如果未来新增算法但忘记处理，程序会静默退回 Warp Tile；
- 调用者可能以为使用了指定算法，实际上没有；
- 不利于及时发现枚举分发错误。

更严格的工程代码可能选择：

```cpp
throw std::runtime_error("Unsupported GEMM algorithm");
```

或打印错误并终止。

当前项目采用的是“保证能运行”的宽松兜底策略。

---

## 11. 结束 `switch` 和函数

```cpp
    }
}
```

### 详细解释

第一个右花括号：

```cpp
}
```

结束：

```cpp
switch (final_algo)
```

第二个右花括号：

```cpp
}
```

结束：

```cpp
void launch_gemm(...)
```

函数没有显式写：

```cpp
return;
```

因为返回类型是 `void`。

执行到函数末尾时，会自动返回调用者。

---

## 12. 完整源码

```cpp
#include "gemm/launcher.h"
#include "gemm/kernels/gemm_warp_tile.h"

namespace {

inline GemmAlgo select_gemm_algo(const int M, const int N, const int K) {
    return GemmAlgo::WarpTile;
}

}  

void launch_gemm(const int M,
                 const int N,
                 const int K,
                 const float alpha,
                 const float* A,
                 const float* B,
                 const float beta,
                 float* C,
                 GemmAlgo algo) {
    GemmAlgo final_algo = algo;

    if (final_algo == GemmAlgo::Auto) {
        final_algo = select_gemm_algo(M, N, K);
    }

    switch (final_algo) {
        case GemmAlgo::WarpTile:
            launch_gemm_warp_tile(M, N, K, alpha, A, B, beta, C);
            break;

        case GemmAlgo::Auto:
        default:
            launch_gemm_warp_tile(M, N, K, alpha, A, B, beta, C);
            break;
    }
}
```

### 完整执行过程

假设测试程序调用：

```cpp
launch_gemm(
    1024,
    1024,
    1024,
    1.0f,
    d_A,
    d_B,
    0.0f,
    d_C
);
```

如果头文件为 `algo` 提供默认参数：

```cpp
GemmAlgo::Auto
```

那么执行过程是：

```text
进入 launch_gemm
        ↓
algo = GemmAlgo::Auto
        ↓
final_algo = algo
        ↓
final_algo 是 Auto
        ↓
调用 select_gemm_algo(1024,1024,1024)
        ↓
返回 GemmAlgo::WarpTile
        ↓
进入 switch
        ↓
匹配 case GemmAlgo::WarpTile
        ↓
调用 launch_gemm_warp_tile(...)
        ↓
break 跳出 switch
        ↓
launch_gemm 结束并返回测试程序
```

如果调用者显式写：

```cpp
launch_gemm(
    M, N, K,
    alpha,
    d_A,
    d_B,
    beta,
    d_C,
    GemmAlgo::WarpTile
);
```

则执行过程是：

```text
final_algo = WarpTile
        ↓
不进入 Auto 判断
        ↓
switch 匹配 WarpTile
        ↓
调用 launch_gemm_warp_tile
```

---

## 13. 本文件中的函数调用链

```text
test_perf_gemm.cpp
        │
        │ 调用统一公开接口
        ▼
launch_gemm(...)
        │
        ├── 如果 algo == Auto
        │       ↓
        │   select_gemm_algo(...)
        │       ↓
        │   当前固定返回 WarpTile
        │
        ▼
switch(final_algo)
        │
        ▼
launch_gemm_warp_tile(...)
        │
        ▼
下层选择配置、计算 grid/block 并启动 CUDA Kernel
```

---

## 14. 需要重点理解的语法汇总

### 14.1 匿名命名空间

```cpp
namespace {
    ...
}
```

其中内容只在当前源文件内可见。

---

### 14.2 内联函数

```cpp
inline GemmAlgo select_gemm_algo(...)
```

表示这是一个适合内联的小函数，但不保证编译器一定展开。

---

### 14.3 枚举成员访问

```cpp
GemmAlgo::Auto
GemmAlgo::WarpTile
```

`::` 表示从 `GemmAlgo` 的作用域中访问枚举成员。

---

### 14.4 只读输入指针

```cpp
const float* A
```

表示不能通过 A 修改其指向的数据。

---

### 14.5 可写输出指针

```cpp
float* C
```

表示函数可以把结果写入 C 指向的内存。

---

### 14.6 条件分支

```cpp
if (final_algo == GemmAlgo::Auto)
```

检查当前算法是否为自动选择模式。

---

### 14.7 运行时分发

```cpp
switch (final_algo)
```

根据枚举值调用具体 GEMM 实现。

---

### 14.8 防止贯穿

```cpp
break;
```

结束当前 `case`，避免继续执行后面的分支。

---

### 14.9 默认兜底

```cpp
default:
```

当没有任何 `case` 匹配时执行。

---

## 15. 当前代码最核心的一句话

```text
launcher.cu 自己不执行矩阵乘法，也不直接编写 CUDA Kernel；
它接收统一 GEMM 参数，根据 GemmAlgo 确定算法，
再把所有参数转交给 launch_gemm_warp_tile。
```


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-gemm算子|模块-gemm算子]]

%% 项目关联导航：结束 %%
