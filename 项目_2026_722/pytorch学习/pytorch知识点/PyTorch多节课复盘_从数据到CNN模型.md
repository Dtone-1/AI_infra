# PyTorch 多节课复盘：从数据到 CNN 模型

> 适用对象：PyTorch 初学者，目标是先打通“图像数据 → DataLoader → CNN 模型 → 输出 logits”的完整流程，并为后续学习 AI Infra 推理、ONNX、TensorRT、vLLM、CUDA、batching、KV Cache 等内容打基础。

---

## 一、先总体判断这几节课在讲什么

这几节课整体是在讲 **PyTorch 图像分类任务的最基础工作流**：从数据集读取开始，到数据预处理、按 batch 加载，再到用 `nn.Module` 搭建神经网络，最后把图片输入模型得到分类输出。  
课程内容并不是孤立的 API 教程，而是在逐步搭一条完整链路：`torchvision.datasets.CIFAR10` 负责拿到图像数据，`transforms.ToTensor()` 负责把图片转成 Tensor，`DataLoader` 负责把单张样本组织成 batch，`nn.Module` 负责定义模型骨架，`Conv2d / MaxPool2d / ReLU / Flatten / Linear` 负责构成 CNN 的计算流程，`Sequential` 负责简化顺序型模型代码。  
这些课里重复出现最多的核心是 **shape 思维**：图像进入 PyTorch 后通常是 `[N, C, H, W]`，每一层都会改变或保持其中某些维度。  
从学习主线看，前半部分是“数据怎么进入模型”，后半部分是“模型怎么把输入 Tensor 变成输出 logits”。  
从 AI Infra 推理方向看，这些基础对应的是未来推理系统中的输入预处理、batching、计算图、算子执行、shape 推导、显存占用、吞吐/延迟分析等问题。  
因此这几节课真正要掌握的不是背 API，而是形成一个基本认知：**深度学习模型就是一串 Tensor 经过一串算子，shape、dtype、device、layout 在每一步都必须正确。**

---

## 二、把所有课程内容整理成一条学习主线

完整流程如下：

```text
图片数据 / CIFAR-10
        ↓
torchvision.datasets.CIFAR10
        ↓
transforms.ToTensor()
        ↓
Dataset
        ↓
DataLoader
        ↓
batch: [N, C, H, W]
        ↓
nn.Module
        ↓
Conv2d
        ↓
ReLU / Sigmoid 等非线性激活
        ↓
MaxPool2d
        ↓
Flatten
        ↓
Linear
        ↓
Sequential
        ↓
输出 logits: [N, num_classes]
```

### 1. 图片数据 / CIFAR-10

CIFAR-10 是一个图像分类数据集，每张图片大小是 `32×32`，有 3 个颜色通道，即 RGB。它有 10 个类别，所以模型最后通常输出 10 个数。  
在这几节课中，CIFAR-10 的作用是提供一个真实的小图像数据集，让你练习从数据读取到模型输出的完整流程。

### 2. `torchvision.datasets.CIFAR10`

`torchvision.datasets.CIFAR10` 是 PyTorch 官方提供的标准数据集封装。  
它解决的问题是：不用你手动下载、解压、解析图片和标签，而是通过一个 Dataset 对象统一管理样本。

核心代码：

```python
dataset = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
```

输入：数据集路径、是否使用训练集、是否下载、是否做 transform。  
输出：一个 Dataset 对象。你可以通过 `dataset[i]` 拿到第 `i` 个样本，通常是 `(image, label)`。

### 3. `transforms.ToTensor()`

原始图片一般是 PIL Image，不能直接送进神经网络。`ToTensor()` 的作用是把图片转成 PyTorch Tensor。

它通常完成三件事：

```text
PIL Image / ndarray
        ↓
torch.Tensor
        ↓
shape 从 [H, W, C] 变成 [C, H, W]
        ↓
像素值从 0~255 缩放到 0~1
```

对 CIFAR-10 来说，单张图片经过 `ToTensor()` 后：

```text
[3, 32, 32]
```

### 4. Dataset

Dataset 负责定义“数据在哪里”和“单个样本怎么取”。

你可以把它理解为：

```text
Dataset 管单个样本
DataLoader 管一批样本
```

对于 CIFAR-10：

```python
image, label = dataset[0]
```

得到：

```text
image: [3, 32, 32]
label: 一个整数，例如 3
```

### 5. DataLoader

DataLoader 把单个样本组成 batch，并且控制是否打乱、一次取多少张、是否多进程加载等。

核心代码：

```python
dataloader = DataLoader(
    dataset=dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    drop_last=False
)
```

如果单张图片是：

```text
[3, 32, 32]
```

那么 batch size 为 64 后：

```text
images: [64, 3, 32, 32]
labels: [64]
```

### 6. batch: `[N, C, H, W]`

PyTorch 图像模型最常见输入格式是：

```text
[N, C, H, W]
```

含义：

```text
N: batch size，一次送进模型多少张图片
C: channel，通道数，RGB 图像为 3
H: height，高度
W: width，宽度
```

CIFAR-10 的一个 batch 通常是：

```text
[64, 3, 32, 32]
```

### 7. `nn.Module`

`nn.Module` 是所有 PyTorch 神经网络的基础类。  
你自己写模型时，一般都要继承它：

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
```

它解决的问题是：让 PyTorch 知道这是一个模型，并自动管理模型里的层、参数、保存加载、训练/推理状态、GPU 转移等。

### 8. Conv2d

`Conv2d` 是卷积层，用来提取图像局部特征。  
它通常改变 channel，也可能改变 H/W。

示例：

```python
nn.Conv2d(3, 32, kernel_size=5, padding=2)
```

含义：

```text
输入通道：3
输出通道：32
卷积核：5×5
padding=2：保持 H/W 不变
```

输入：

```text
[64, 3, 32, 32]
```

输出：

```text
[64, 32, 32, 32]
```

### 9. ReLU / Sigmoid 等非线性激活

激活函数负责给神经网络引入非线性。  
如果没有非线性，多层线性变换叠起来仍然等价于一层线性变换，模型表达能力会很弱。

ReLU：

```python
nn.ReLU()
```

作用：

```text
负数 → 0
正数 → 保留
```

Sigmoid：

```python
nn.Sigmoid()
```

作用：

```text
任意输入 → 0~1 之间
```

激活函数通常不改变 shape，只改变数值。

### 10. MaxPool2d

最大池化负责下采样，减少 H/W，同时尽量保留局部最显著特征。

```python
nn.MaxPool2d(2)
```

默认等价于：

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

输入：

```text
[64, 32, 32, 32]
```

输出：

```text
[64, 32, 16, 16]
```

注意：池化通常不改变 channel，只改变 H/W。

### 11. Flatten

卷积和池化输出的是四维 Tensor：

```text
[N, C, H, W]
```

Linear 需要的是二维特征：

```text
[N, features]
```

所以要展平：

```python
nn.Flatten()
```

例如：

```text
[64, 64, 4, 4]
        ↓
[64, 1024]
```

因为：

```text
64 × 4 × 4 = 1024
```

### 12. Linear

线性层，也叫全连接层，用来把特征向量映射成输出特征。

```python
nn.Linear(1024, 64)
nn.Linear(64, 10)
```

含义：

```text
1024 维图像特征 → 64 维中间特征 → 10 类输出
```

输出：

```text
[64, 10]
```

### 13. Sequential

`nn.Sequential` 是顺序容器。  
当模型是一条直线结构时，可以把层按顺序放进去，代码更简洁。

```python
self.model = nn.Sequential(
    nn.Conv2d(...),
    nn.MaxPool2d(...),
    nn.Flatten(),
    nn.Linear(...)
)
```

`forward()` 中只需要：

```python
return self.model(x)
```

### 14. 输出 logits: `[N, num_classes]`

最终输出：

```text
[64, 10]
```

含义：

```text
64: batch size，64 张图片
10: 每张图片对应 10 个类别得分
```

这些得分叫 logits，不是概率。  
推理时可以：

```python
pred = outputs.argmax(dim=1)
```

得到预测类别。

---

## 三、知识点总表

| 知识点名称 | 流程位置 | 解决什么问题 | PyTorch API / 代码 | 输入 shape | 输出 shape | 掌握程度 | AI Infra 推理关系 |
|---|---|---|---|---|---|---|---|
| `torchvision.datasets.CIFAR10` | 数据源 | 获取标准图像分类数据集 | `torchvision.datasets.CIFAR10(...)` | 路径/配置 | Dataset 对象 | 会用即可 | 推理 benchmark 常用标准数据集 |
| `transforms.ToTensor` | 预处理 | 图片转 Tensor | `transforms.ToTensor()` | PIL Image `[H,W,C]` | Tensor `[C,H,W]` | 必须掌握 | 预处理必须和部署一致 |
| `Dataset` | 单样本管理 | 定义样本读取方式 | `dataset[i]` | index | `(image,label)` | 必须理解 | 数据管线源头 |
| `DataLoader` | batch 构造 | 批量加载数据 | `DataLoader(dataset, ...)` | Dataset | batch | 必须掌握 | 对应推理 batching |
| `batch_size` | DataLoader 参数 | 控制一次取多少样本 | `batch_size=64` | 单样本 | `[64,C,H,W]` | 必须掌握 | 影响吞吐、延迟、显存 |
| `shuffle` | DataLoader 参数 | 是否打乱顺序 | `shuffle=True` | Dataset 顺序 | 随机顺序 | 训练常用 | 推理通常不需要 |
| `num_workers` | DataLoader 参数 | 多进程加载数据 | `num_workers=0/2/4` | 数据读取任务 | 更快加载 | 初学设 0 | 预处理瓶颈分析 |
| `drop_last` | DataLoader 参数 | 是否丢弃最后不完整 batch | `drop_last=True/False` | 最后小 batch | 保留或丢弃 | 理解即可 | 静态 batch 场景可能需要 |
| `nn.Module` | 模型骨架 | 定义神经网络 | `class Model(nn.Module)` | Tensor | Tensor | 必须掌握 | 模型导出源头 |
| `__init__` | 模型初始化 | 定义层和参数 | `def __init__(self)` | 无 | 模型结构 | 必须掌握 | 参数注册、权重加载 |
| `forward` | 前向传播 | 定义计算过程 | `def forward(self,x)` | 输入 Tensor | 输出 Tensor | 必须掌握 | 推理核心路径 |
| `model(x)` | 模型调用 | 执行 forward | `outputs=model(images)` | input | output | 必须掌握 | 推理服务核心调用 |
| `nn.Conv2d` | 特征提取 | 提取局部空间特征 | `nn.Conv2d(...)` | `[N,C,H,W]` | `[N,C_out,H_out,W_out]` | 必须掌握 | ONNX/TensorRT Conv 算子 |
| `in_channels` | Conv 参数 | 匹配输入通道 | `in_channels=3` | C=3 | - | 必须掌握 | layout/shape 校验 |
| `out_channels` | Conv 参数 | 决定输出通道 | `out_channels=32` | - | C=32 | 必须掌握 | 影响 FLOPs/显存 |
| `kernel_size` | Conv/Pool 参数 | 控制窗口大小 | `kernel_size=5` | H/W | H/W 变化 | 必须掌握 | 算子选择/shape 推导 |
| `stride` | Conv/Pool 参数 | 控制移动步长 | `stride=1/2` | H/W | H/W 变化 | 必须掌握 | 影响计算量和输出 shape |
| `padding` | Conv 参数 | 边缘补零，控制尺寸 | `padding=2` | H/W | H/W 变化 | 必须掌握 | 部署一致性 |
| `nn.MaxPool2d` | 下采样 | 降低 H/W | `nn.MaxPool2d(2)` | `[N,C,H,W]` | `[N,C,H/2,W/2]` | 必须掌握 | 减少激活显存 |
| `ceil_mode` | Pool 参数 | 是否保留边缘不完整窗口 | `ceil_mode=True` | H/W | H/W 变化 | 理解即可 | 影响导出 shape |
| `nn.ReLU` | 激活 | 引入非线性 | `nn.ReLU()` | 任意 Tensor | shape 不变 | 必须掌握 | 常被算子融合 |
| `nn.Sigmoid` | 激活 | 压缩到 0~1 | `nn.Sigmoid()` | 任意 Tensor | shape 不变 | 了解 | 涉及指数计算 |
| `inplace` | 激活参数 | 是否原地修改 | `inplace=False` | Tensor | Tensor | 理解即可 | 内存复用/图转换 |
| `torch.flatten` | 函数式展平 | Tensor 变形 | `torch.flatten(x,start_dim=1)` | `[N,C,H,W]` | `[N,C×H×W]` | 必须掌握 | Reshape/Shuffle 节点 |
| `nn.Flatten` | 模型层展平 | 接入 Sequential | `nn.Flatten()` | `[N,C,H,W]` | `[N,C×H×W]` | 必须掌握 | 计算图 shape 节点 |
| `nn.Linear` | 分类头/MLP | 特征映射 | `nn.Linear(1024,10)` | `[N,in]` | `[N,out]` | 必须掌握 | GEMM/MatMul 核心算子 |
| `in_features` | Linear 参数 | 输入特征数 | `1024` | 最后一维 | - | 必须会算 | shape 错误常见来源 |
| `out_features` | Linear 参数 | 输出特征数 | `10` | - | 最后一维 | 必须理解 | 输出 logits/后处理 |
| `nn.Sequential` | 模型容器 | 简化顺序模型 | `nn.Sequential(...)` | Tensor | Tensor | 必须会用 | 计算图更清晰 |
| TensorBoard | 可视化 | 看图片/图结构 | `tensorboard --logdir=...` | logs | UI | 会用即可 | 调试中间 tensor |
| `add_images` | 图片可视化 | 显示 batch 图像 | `writer.add_images(...)` | `[N,C,H,W]` | 日志 | 会用即可 | 中间输出调试 |
| `add_graph` | 计算图可视化 | 显示模型结构 | `writer.add_graph(model,input)` | model+dummy input | graph | 会用即可 | 计算图分析入门 |

---

## 四、按流程精讲核心知识点

### 1. 数据集和预处理：从图片到 Tensor

#### 知识点是什么

数据集和预处理负责把原始图片变成模型能接收的 Tensor。  
模型不能直接处理文件路径、PIL 图片或普通 Python list，它需要规则化的 Tensor。

#### 为什么需要它

神经网络的计算是数值计算。  
图片必须变成多维数组，并且要统一 dtype、shape、数值范围。

#### 输入输出和 shape

```text
原始图片: PIL Image / ndarray
        ↓ ToTensor
单张图片: [3, 32, 32]
        ↓ DataLoader
一个 batch: [64, 3, 32, 32]
```

#### 最小代码示例

```python
import torchvision
from torchvision import transforms

dataset = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

image, label = dataset[0]
print(image.shape, label)
```

#### 常见错误和坑

- 忘记 `transform=transforms.ToTensor()`，导致 PIL Image 不能直接进模型。
- 不理解 `[3,32,32]`，误以为是 `[32,32,3]`。
- 不知道 label 是整数类别 id。
- 训练和推理预处理不一致，导致部署结果不对。

#### 和 AI Infra 推理的关系

推理服务中的预处理必须和训练时一致。  
例如图像模型部署时，输入可能要经过 resize、normalize、HWC→CHW、CPU→GPU、FP32→FP16 等操作。  
如果预处理错了，模型本身再快也没用，输出会错。

---

### 2. DataLoader：从单样本到 batch

#### 知识点是什么

DataLoader 把 Dataset 中的多个样本打包成一个 batch。

#### 为什么需要它

深度学习通常不是一张图一张图训练或推理，而是一批图一起送入模型，提高并行效率。

#### 输入输出和 shape

单张：

```text
image: [3,32,32]
label: scalar
```

batch：

```text
images: [64,3,32,32]
labels: [64]
```

#### 最小代码示例

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

images, labels = next(iter(dataloader))
print(images.shape)
print(labels.shape)
```

#### 常见错误和坑

- `for data in dataloader` 后忘记解包：应写 `images, labels = data`。
- `num_workers` 在 Windows 上报错，初学先用 `0`。
- `drop_last=False` 时最后一个 batch 可能不足 64。
- batch size 太大可能显存爆掉。

#### 和 AI Infra 推理的关系

推理服务里的 batching 和 DataLoader 的 batch 思想类似。  
batch size 越大，吞吐量通常越高，但单请求延迟可能增加，显存占用也更大。  
vLLM 中的 continuous batching、本质上就是更复杂的动态 batch 调度。

---

### 3. nn.Module：模型的骨架

#### 知识点是什么

`nn.Module` 是 PyTorch 所有模型的基类。  
你自己写的网络必须继承它。

#### 为什么需要它

它负责管理网络层、参数、训练/推理模式、保存加载、GPU 转移等。

#### 输入输出和 shape

`nn.Module` 自身不固定 shape。  
具体输入输出由 `forward()` 中的层决定。

#### 最小代码示例

```python
import torch
from torch import nn

class AddOne(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1

model = AddOne()
x = torch.tensor(1.0)
print(model(x))
```

#### 常见错误和坑

- 忘记 `super().__init__()`。
- 没有写 `forward()`。
- 手动调用 `model.forward(x)`，标准写法应是 `model(x)`。
- 在 `forward()` 里创建有参数层，导致参数无法正常管理。

#### 和 AI Infra 推理的关系

推理部署的入口通常就是一个 `nn.Module`。  
ONNX 导出、TorchScript、TensorRT 转换，都要从模型的 forward 计算图出发。

---

### 4. Conv2d：提取图像局部特征

#### 知识点是什么

`Conv2d` 用卷积核在图片上滑动，对局部区域做计算，提取边缘、纹理、形状等特征。

#### 为什么需要它

图片有空间结构，卷积能利用局部相关性，比直接把图片展平后接 Linear 更适合视觉任务。

#### 输入输出和 shape

```python
nn.Conv2d(3, 32, kernel_size=5, padding=2)
```

输入：

```text
[64,3,32,32]
```

输出：

```text
[64,32,32,32]
```

#### 最小代码示例

```python
import torch
from torch import nn

conv = nn.Conv2d(3, 32, kernel_size=5, padding=2)
x = torch.randn(64, 3, 32, 32)
y = conv(x)
print(y.shape)
```

#### 常见错误和坑

- 把 batch size 当成 `in_channels`。
- CIFAR-10 是 3 通道，却写成 `in_channels=1`。
- 不会算 padding，导致 H/W 和预期不一致。
- 下一层 `in_channels` 没接上上一层 `out_channels`。

#### 和 AI Infra 推理的关系

Conv2d 最后会变成 ONNX/TensorRT/cuDNN 中的卷积算子。  
推理框架会优化 Conv，例如 Conv+BN+ReLU 融合、FP16/INT8 量化、Winograd、Tensor Core 加速等。

---

### 5. 激活函数：加入非线性

#### 知识点是什么

ReLU、Sigmoid 等激活函数对 Tensor 中每个元素做非线性变换。

#### 为什么需要它

没有非线性，多层网络仍然等价于单层线性变换，表达能力很弱。

#### 输入输出和 shape

输入什么 shape，输出通常还是同样 shape：

```text
[64,32,32,32] → [64,32,32,32]
```

#### 最小代码示例

```python
import torch
from torch import nn

relu = nn.ReLU()
x = torch.tensor([-1.0, 0.5, 2.0])
print(relu(x))
```

#### 常见错误和坑

- 以为 ReLU 会改变 shape。
- 随便在输出层加激活。
- `inplace=True` 可能修改原 Tensor，初学默认 False 更稳。
- Sigmoid 后再用 `BCEWithLogitsLoss` 属于重复 sigmoid。

#### 和 AI Infra 推理的关系

激活函数是 element-wise 算子，通常访存占比较高，容易和 Conv/Linear 融合。  
ReLU 常被融合成 ConvReLU、LinearReLU，以减少中间 Tensor 读写和 kernel launch。

---

### 6. MaxPool2d：降低空间尺寸

#### 知识点是什么

最大池化用窗口覆盖局部区域，取最大值作为输出。

#### 为什么需要它

它减少 H/W，降低后续计算量和显存，同时保留局部强响应。

#### 输入输出和 shape

```python
nn.MaxPool2d(2)
```

输入：

```text
[64,32,32,32]
```

输出：

```text
[64,32,16,16]
```

#### 最小代码示例

```python
pool = nn.MaxPool2d(2)
x = torch.randn(64, 32, 32, 32)
y = pool(x)
print(y.shape)
```

#### 常见错误和坑

- 以为池化改变通道数。它通常只改 H/W。
- 不知道 MaxPool2d 默认 stride 等于 kernel_size。
- `ceil_mode=True` 会改变输出尺寸。
- 池化太多导致特征图过小。

#### 和 AI Infra 推理的关系

池化会减少中间激活大小，降低显存占用和后续算子计算量。  
导出到 ONNX/TensorRT 后是 MaxPool/Pooling layer，参数必须一致。

---

### 7. Flatten：从特征图到特征向量

#### 知识点是什么

Flatten 把 `[N,C,H,W]` 展平成 `[N,C×H×W]`，让卷积特征可以接入 Linear。

#### 为什么需要它

Linear 需要二维输入：batch 维 + feature 维。

#### 输入输出和 shape

```text
[64,64,4,4]
        ↓
[64,1024]
```

#### 最小代码示例

```python
flatten = nn.Flatten()
x = torch.randn(64, 64, 4, 4)
y = flatten(x)
print(y.shape)
```

#### 常见错误和坑

- 用 `torch.flatten(x)` 把 batch 维也压没。
- 忘记 Flatten，直接把四维 Tensor 送 Linear。
- 改了前面网络结构，Linear 的 `in_features` 没跟着改。

#### 和 AI Infra 推理的关系

Flatten/Reshape 在部署中经常变成 ONNX 的 Flatten/Reshape 或 TensorRT 的 Shuffle layer。  
这类操作不重计算，但非常容易引起 shape 错误。

---

### 8. Linear：输出分类 logits

#### 知识点是什么

Linear 把输入特征向量映射到输出特征，常用于分类头。

#### 为什么需要它

CNN 提取出图像特征后，需要把特征变成类别得分。

#### 输入输出和 shape

```python
nn.Linear(1024, 64)
nn.Linear(64, 10)
```

shape：

```text
[64,1024] → [64,64] → [64,10]
```

#### 最小代码示例

```python
linear = nn.Linear(1024, 10)
x = torch.randn(64, 1024)
y = linear(x)
print(y.shape)
```

#### 常见错误和坑

- `in_features` 算错。
- 把 Linear 输出当成概率。它通常是 logits。
- 最后一层输出类别数写错。
- 不理解 `[64,10]` 的含义。

#### 和 AI Infra 推理的关系

Linear 底层就是矩阵乘法 GEMM/MatMul。  
LLM 中大量核心计算都是 Linear，例如 Q/K/V projection、MLP、LM Head。  
大模型推理优化很大程度就是 GEMM 优化、权重量化、Tensor Core 使用、权重切分和显存管理。

---

### 9. Sequential：简化顺序模型

#### 知识点是什么

`nn.Sequential` 是一个顺序容器，按写入顺序依次执行层。

#### 为什么需要它

当模型是一条直线结构时，它能让代码更简洁。

#### 输入输出和 shape

输入输出由内部层决定。

#### 最小代码示例

```python
model = nn.Sequential(
    nn.Conv2d(3, 32, 5, padding=2),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 16 * 16, 10)
)
```

#### 常见错误和坑

- 层之间忘记逗号。
- 顺序写错，例如 Linear 前忘记 Flatten。
- 有残差、分支、多输入输出时不适合简单 Sequential。
- 复杂 Transformer/vLLM 源码通常要手写 forward。

#### 和 AI Infra 推理的关系

Sequential 对应直线式计算图。  
推理引擎也是按图执行算子，但真实模型可能有分支、跳连、KV Cache、多输入输出等复杂结构。

---

### 10. TensorBoard：看图片和看计算图

#### 知识点是什么

TensorBoard 可以可视化图片、特征图和模型计算图。

#### 为什么需要它

初学时可以观察数据是否正确、shape 是否正确、模型图是否连通。

#### 最小代码示例

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
writer.add_images("input", images, 0)
writer.add_graph(model, dummy_input)
writer.close()
```

#### 常见错误和坑

- `add_images` 输入通道不是 1 或 3 时可能显示不了。
- logdir 写错。
- 多次实验写同一个目录导致混乱。
- `add_graph` 必须给示例输入。

#### 和 AI Infra 推理的关系

它训练的是计算图思维。  
后续你会看 ONNX graph、TensorRT engine graph、Nsight timeline、CUDA Graph、vLLM forward path，本质都是在分析“输入 Tensor 如何经过算子变成输出 Tensor”。

---

## 五、完整可运行代码

下面代码完成：

1. 导入包  
2. 准备 CIFAR-10  
3. 准备 DataLoader  
4. 定义 CNN 模型  
5. 打印模型结构  
6. 用 dummy input 检查 shape  
7. 用真实 DataLoader 跑一个 batch  
8. 打印关键 shape  
9. 写入 TensorBoard 图片和计算图  

```python
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


# =========================
# 1. 准备数据集
# =========================

transform = transforms.ToTensor()

test_dataset = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=False,
    transform=transform,
    download=True
)


# =========================
# 2. 准备 DataLoader
# =========================

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    drop_last=False
)


# =========================
# 3. 定义模型
# =========================

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


model = SimpleCNN()
print(model)


# =========================
# 4. 用 dummy input 检查模型 shape
# =========================

dummy_input = torch.ones((64, 3, 32, 32))
dummy_output = model(dummy_input)

print("dummy input shape:", dummy_input.shape)
print("dummy output shape:", dummy_output.shape)


# =========================
# 5. 用真实 DataLoader 跑一个 batch
# =========================

images, labels = next(iter(test_loader))
outputs = model(images)

print("real images shape:", images.shape)
print("real labels shape:", labels.shape)
print("real outputs shape:", outputs.shape)

preds = outputs.argmax(dim=1)
print("preds shape:", preds.shape)
print("first 10 preds:", preds[:10])
print("first 10 labels:", labels[:10])


# =========================
# 6. TensorBoard 可视化
# =========================

writer = SummaryWriter("logs_pytorch_review")

writer.add_images("input_images", images, 0)
writer.add_graph(model, dummy_input)

writer.close()

print("TensorBoard log saved to logs_pytorch_review")
print("Run: tensorboard --logdir=logs_pytorch_review")
```

---

## 六、逐行解释重点代码

### `transform=transforms.ToTensor()`

把 PIL 图片转成 PyTorch Tensor。  
CIFAR-10 原始图片不能直接进模型，必须先转成 `[C,H,W]` 的 Tensor，并把像素值从 0~255 变到 0~1。

### `DataLoader(dataset, batch_size=64, shuffle=True)`

把 Dataset 中的单个样本组成 batch。  
`batch_size=64` 表示每次取 64 张图。  
`shuffle=True` 表示打乱顺序，训练时常用。复习和测试时也可以设置 False。

### `class SimpleCNN(nn.Module)`

定义自己的模型类，并继承 `nn.Module`。  
这是 PyTorch 模型的标准写法。

### `super().__init__()`

调用父类 `nn.Module` 的初始化方法。  
不写它，模型内部的参数、子模块注册可能出问题。

### `self.model = nn.Sequential(...)`

把多个层按顺序放进容器中。  
输入进入 `self.model` 后，会自动依次经过 Conv、ReLU、Pool、Flatten、Linear。

### `nn.Conv2d(3, 32, kernel_size=5, padding=2)`

输入通道是 3，因为 CIFAR-10 是 RGB 图片。  
输出通道是 32，表示提取 32 种特征。  
`kernel_size=5` 表示 5×5 卷积核。  
`padding=2` 是为了保持 32×32 的空间尺寸不变。

### `nn.MaxPool2d(2)`

用 2×2 窗口做最大池化，默认 stride=2。  
它会把 H/W 减半，例如 32×32 变 16×16。

### `nn.Flatten()`

把 `[N,C,H,W]` 展平成 `[N,C×H×W]`。  
在这个模型中，最后池化后是 `[64,64,4,4]`，展平后是 `[64,1024]`。

### `nn.Linear(64 * 4 * 4, 64)`

第一层全连接。  
输入特征数是 `64×4×4=1024`，输出是 64 维中间特征。

### `nn.Linear(64, 10)`

最后一层分类头。  
CIFAR-10 有 10 个类别，所以输出 10 个 logits。

### `def forward(self, x)`

定义模型前向传播。  
输入 `x` 是图片 batch。

### `return self.model(x)`

把输入直接交给 Sequential。  
Sequential 会按顺序执行里面所有层，并返回最终输出。

### `dummy_input = torch.ones((64, 3, 32, 32))`

构造一个假输入，用来检查模型 shape 是否正确。  
它模拟一个 CIFAR-10 batch。

### `output = model(dummy_input)`

执行一次 forward。  
如果能得到 `[64,10]`，说明模型结构至少在 shape 上是连通的。

### `for images, labels in dataloader`

从 DataLoader 中取出图片和标签。  
`images` 是 `[64,3,32,32]`，`labels` 是 `[64]`。

### `outputs = model(images)`

把真实图片 batch 输入模型，得到输出 logits。  
输出 shape 是 `[64,10]`。

---

## 七、重点整理 shape 变化

以输入 `[64, 3, 32, 32]` 为例：

| 步骤 | 层 | 输出 shape | 解释 |
|---|---|---|---|
| 输入 | 原始 batch | `[64, 3, 32, 32]` | 64 张 RGB 图片 |
| 1 | `Conv2d(3,32,5,padding=2)` | `[64, 32, 32, 32]` | 通道 3→32，H/W 不变 |
| 2 | `MaxPool2d(2)` | `[64, 32, 16, 16]` | H/W 减半 |
| 3 | `Conv2d(32,32,5,padding=2)` | `[64, 32, 16, 16]` | 通道不变，H/W 不变 |
| 4 | `MaxPool2d(2)` | `[64, 32, 8, 8]` | H/W 再减半 |
| 5 | `Conv2d(32,64,5,padding=2)` | `[64, 64, 8, 8]` | 通道 32→64，H/W 不变 |
| 6 | `MaxPool2d(2)` | `[64, 64, 4, 4]` | H/W 再减半 |
| 7 | `Flatten()` | `[64, 1024]` | 64×4×4=1024 |
| 8 | `Linear(1024,64)` | `[64, 64]` | 每张图变成 64 维特征 |
| 9 | `Linear(64,10)` | `[64, 10]` | 每张图输出 10 个类别得分 |

### 为什么 Conv2d 后通道数变化？

因为 `out_channels` 决定输出通道数。  
例如 `Conv2d(3,32,...)` 就是把 3 通道输入变成 32 通道特征。

### 为什么 MaxPool2d 后 H/W 减半？

`MaxPool2d(2)` 默认 stride=2。  
所以每 2×2 区域取一个最大值，空间尺寸约减半。

### 为什么 Flatten 后是 1024？

最后一次池化后，每张图片的特征图是：

```text
[64, 4, 4]
```

元素数：

```text
64 × 4 × 4 = 1024
```

所以 Flatten 后每张图片是 1024 维。

### 为什么最后输出 `[64,10]`？

因为 batch size 是 64，CIFAR-10 有 10 个类别。  
所以每张图片输出 10 个类别得分。

```text
64: 64 张图片
10: 每张图片的 10 个类别 logits
```

### logits 和概率的区别

`Linear` 输出的是 logits，不是概率。  
logits 可以是负数，也不要求加起来等于 1。

推理时可以：

```python
pred = outputs.argmax(dim=1)
```

如果想看概率，可以：

```python
prob = torch.softmax(outputs, dim=1)
```

训练时如果用 `nn.CrossEntropyLoss()`，通常不要手动先 softmax，因为它内部会处理 logits。

---

## 八、重复内容压缩成复习版：核心规律

1. **PyTorch 图像输入通常是 `[N, C, H, W]`**  
   N 是 batch，C 是通道，H/W 是高宽。

2. **Dataset 负责单样本，DataLoader 负责 batch**  
   Dataset 返回 `[C,H,W]`，DataLoader 返回 `[N,C,H,W]`。

3. **Transform 负责把原始数据变成模型能吃的 Tensor**  
   `ToTensor()` 是图像任务最基础的 transform。

4. **Conv2d 改变 channel，也可能改变 H/W**  
   channel 由 `out_channels` 决定，H/W 由 kernel、stride、padding 决定。

5. **MaxPool2d 通常只改变 H/W，不改变 channel**  
   `MaxPool2d(2)` 常用来让 H/W 减半。

6. **激活函数通常不改变 shape，只改变数值**  
   ReLU 把负数截断为 0，Sigmoid 把数压到 0~1。

7. **Flatten 把 `[N,C,H,W]` 变成 `[N,C×H×W]`**  
   注意保留 batch 维。

8. **Linear 改变最后一维 feature**  
   `[N,1024] → [N,10]`。

9. **Sequential 适合一条直线式网络**  
   复杂分支、残差、KV Cache 等结构通常要手写 forward。

10. **写完模型一定要用 dummy input 检查 shape**  
    `torch.ones((64,3,32,32))` 可以快速验证网络是否跑通。

11. **推理时要写 `model.eval()` 和 `torch.no_grad()`**  
    Dropout、BatchNorm、梯度图都会受到影响。

12. **shape 是贯穿 PyTorch、ONNX、TensorRT、vLLM 的核心线索**  
    你能跟踪 shape，就能看懂大部分模型执行流程。

---

## 九、结合 AI Infra 推理方向总结

### 1. 为什么要理解 Tensor shape / dtype / device？

推理系统处理的不是“图片”或“文本”本身，而是 Tensor。

一个 Tensor 至少有：

```text
shape: 形状
dtype: 数据类型，如 FP32、FP16、BF16、INT8
device: CPU 或 GPU
layout: NCHW、NHWC 等
```

如果 shape 错，模型跑不通；如果 dtype 错，可能性能差或数值不一致；如果 device 错，会出现 CPU/GPU 不一致报错；如果 layout 错，结果可能完全错误。

### 2. 为什么 batch size 会影响吞吐量和延迟？

batch size 小：

```text
单请求延迟低
GPU 利用率可能低
吞吐量低
```

batch size 大：

```text
吞吐量高
GPU 利用率高
显存占用大
单请求可能等待更久
```

推理系统的核心就是在 latency 和 throughput 之间做权衡。  
vLLM 的 continuous batching 就是为了解决大模型推理中请求长度不同、到达时间不同的问题。

### 3. 为什么 Conv2d / Linear 最后会变成推理引擎里的算子？

PyTorch 里的层在导出后会变成计算图节点：

```text
nn.Conv2d  → ONNX Conv → TensorRT Convolution
nn.Linear  → ONNX Gemm/MatMul → TensorRT MatrixMultiply
nn.ReLU    → ONNX Relu → TensorRT Activation
```

推理引擎优化的就是这些节点。

### 4. 为什么 Linear / MatMul 是大模型推理核心？

Transformer 中大部分计算都是矩阵乘法：

```text
Q/K/V projection
Attention output projection
MLP up projection
MLP down projection
LM Head
```

这些在 PyTorch 高层看是 Linear，底层看是 GEMM/MatMul。  
所以大模型推理优化绕不开 Tensor Core、FP16/BF16/INT8、权重量化、多 GPU 切分、kernel fusion。

### 5. 为什么 Flatten / Reshape 是部署中常见 shape 问题来源？

Flatten 和 Reshape 不重计算，但它们改变 Tensor 的解释方式。  
如果 batch 维被压没，或者动态 shape 推导失败，后续 MatMul 维度就会错。

ONNX/TensorRT 中经常看到：

```text
Reshape
Flatten
Transpose
Shuffle
```

这些节点都要求 shape 极其明确。

### 6. 为什么 `model.eval()` 和 `torch.no_grad()` 对推理重要？

`model.eval()` 会切换模型到推理模式，影响 Dropout 和 BatchNorm。  
`torch.no_grad()` 会关闭梯度计算，减少显存占用和计算图构建开销。

标准推理写法：

```python
model.eval()

with torch.no_grad():
    outputs = model(inputs)
```

更进一步可以用：

```python
with torch.inference_mode():
    outputs = model(inputs)
```

### 7. 为什么 ONNX / TensorRT 需要静态或动态 shape？

推理引擎需要提前知道 Tensor shape，才能：

```text
分配显存
选择 kernel
做图优化
构建 engine
处理 dynamic batch
```

TensorRT 动态 shape 通常要设置 min/opt/max shape。  
如果输入 shape 范围不合理，engine 构建或运行都会出问题。

### 8. 为什么 vLLM 也能用“输入 Tensor → 算子 → 输出 Tensor → shape 变化”理解？

虽然 vLLM 不是 CNN，而是 LLM 推理框架，但底层思路一样：

```text
token ids
        ↓
Embedding
        ↓
Transformer blocks
        ↓
Attention / MLP / Norm
        ↓
logits
        ↓
采样 next token
```

对应 shape 可能是：

```text
[batch, seq_len]
        ↓
[batch, seq_len, hidden_size]
        ↓
[batch, seq_len, vocab_size]
```

KV Cache 也可以从 Tensor 角度理解：

```text
过去 token 的 K/V Tensor 被缓存起来
新 token decode 时复用缓存
避免重复计算历史 attention
```

所以你现在学 CNN 的 shape 流程，不是只为图像任务服务，而是在训练你未来阅读推理引擎代码的底层思维。

---

## 十、最终复习版总结

这几节课真正串起来之后，可以总结成一句话：

```text
PyTorch 图像分类的基本流程，就是把图片数据通过 Dataset/Transform/DataLoader 变成 batch Tensor，
再把 batch Tensor 输入 nn.Module 定义的 CNN，
依次经过 Conv2d、ReLU、MaxPool2d、Flatten、Linear，
最终得到 [N, num_classes] 的 logits。
```

你现在最应该掌握的不是“背会每个 API 的所有参数”，而是建立下面这条主线：

```text
数据怎么来？
shape 是什么？
batch 怎么形成？
模型怎么定义？
每一层怎么改变 shape？
输出代表什么？
推理时如何执行 forward？
这些层未来会变成什么推理算子？
```

只要这条线清楚，后面学习训练循环、损失函数、优化器、模型保存加载、GPU/CUDA、ONNX、TensorRT、vLLM，都能接得上。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-pytorch学习|模块-pytorch学习]]

%% 项目关联导航：结束 %%
