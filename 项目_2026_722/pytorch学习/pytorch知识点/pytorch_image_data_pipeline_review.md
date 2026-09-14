# PyTorch 图像数据处理多节课复习总纲

> 这份笔记整合前面关于 `Dataset`、`TensorBoard`、`transforms`、`ToTensor`、`Resize`、`RandomCrop`、`Normalize`、`DataLoader` 预备知识等内容。目标是理清楚：**图片文件如何一步步变成 PyTorch 模型能训练/推理的 Tensor 输入**，以及这些基础和后续 AI Infra 推理方向的关系。

---

## 一、总览：这几节课合起来到底在讲什么

这几节课合起来，不是在孤立地讲 `Dataset`、`TensorBoard`、`transforms`，而是在解决一个核心问题：

> 一堆磁盘里的图片，如何一步步变成 PyTorch 模型能训练/推理的标准输入 Tensor。

整体主线是：

```text
图片数据集
→ 自定义 Dataset
→ 根据 index 读取图片和 label
→ 用 PIL.Image.open 打开图片
→ 用 transforms 做预处理
→ Resize / RandomCrop 改变图片尺寸或区域
→ ToTensor 把 PIL/NumPy 图片转成 Tensor
→ Normalize 标准化 Tensor 数值
→ TensorBoard 检查图片和 loss 曲线
→ DataLoader 组成 batch
→ 输入模型 model(images)
→ 计算 loss
→ backward / optimizer 更新参数
```

这些内容放在 PyTorch 学习路线里，属于 **模型训练之前的数据准备阶段**。你现在还没有真正深入 `model / loss / optimizer` 的完整训练循环，但已经在学习训练和推理最前端、最容易出错的一段：

```text
数据读取 → 预处理 → Tensor 化 → batch 化 → 可视化检查
```

---

## 二、按真实图像任务流程重新组织知识点

### 1. 数据文件如何组织

最简单的图像分类数据集通常这样组织：

```text
项目目录/
├── main.py
└── data/
    └── train/
        ├── ants/
        │   ├── 001.jpg
        │   ├── 002.jpg
        │   └── ...
        └── bees/
            ├── 001.jpg
            ├── 002.jpg
            └── ...
```

这里：

```text
data/train 是根目录
ants 是一个类别文件夹
bees 是另一个类别文件夹
图片文件名是具体样本
文件夹名可以作为 label
```

简单分类任务中，label 可以直接来自文件夹名：

```text
data/train/ants/xxx.jpg → label = ants
data/train/bees/xxx.jpg → label = bees
```

但真正训练时，模型更喜欢数字 label：

```python
class_to_idx = {
    "ants": 0,
    "bees": 1
}
```

原因是神经网络和损失函数不认识 `"ants"`、`"bees"` 这种字符串，它们处理的是数字和 Tensor。

---

### 2. Dataset 负责什么

`Dataset` 的职责是：

> 给 PyTorch 提供一个“可以按 index 取样本”的数据集对象。

自定义 `Dataset` 一般要写三个核心方法：

```python
class MyData(Dataset):
    def __init__(self, ...):
        ...

    def __getitem__(self, index):
        ...

    def __len__(self):
        ...
```

#### `__init__`：准备数据集信息

它在创建对象时自动执行。例如：

```python
ants_dataset = MyData("data/train", "ants")
```

这时 `__init__` 会保存：

```text
root_dir = data/train
label_dir = ants
img_dir = data/train/ants
img_names = ants 文件夹下所有图片名
```

#### `self`：保存对象自己的信息

如果只写：

```python
img_dir = ...
```

这个变量只在当前函数内部有效。

如果写：

```python
self.img_dir = ...
```

它就变成对象属性，后面的 `__getitem__` 也能用。

#### `__getitem__`：根据 index 返回一个样本

例如：

```python
image, label = dataset[0]
```

底层会调用：

```python
dataset.__getitem__(0)
```

在 `__getitem__` 里，一般会做：

```text
1. 根据 index 找到图片文件名
2. 拼出图片完整路径
3. 用 Image.open 读取图片
4. 得到 label
5. 如果有 transform，就处理图片
6. return image, label
```

#### `__len__`：返回数据集长度

例如：

```python
len(dataset)
```

底层会调用：

```python
dataset.__len__()
```

一般返回图片数量：

```python
return len(self.img_names)
```

#### `os.listdir` 和 `os.path.join`

`os.listdir(self.img_dir)` 用来列出文件夹下所有图片名，例如：

```python
["001.jpg", "002.jpg", "003.jpg"]
```

但它只返回文件名，不返回完整路径。所以要用：

```python
os.path.join(self.img_dir, img_name)
```

拼出完整路径：

```text
data/train/ants/001.jpg
```

`os.path.join` 的好处是能自动适配 Windows 和 Linux 的路径分隔符。

---

### 3. 图片读取后是什么类型

一张图片在不同阶段会有不同类型。

#### PIL Image

用：

```python
from PIL import Image
image = Image.open(img_path)
```

得到的是：

```python
PIL.Image.Image
```

它适合打开图片、查看尺寸、显示图片、保存图片、做简单图像处理。

PIL 图片的尺寸：

```python
image.size
```

返回：

```text
(width, height)
```

注意是 **宽在前，高在后**。

#### NumPy ndarray

用：

```python
import numpy as np
image_np = np.array(image)
```

得到的是：

```python
numpy.ndarray
```

它常见 shape 是：

```text
[H, W, C]
```

也就是高度、宽度、通道。例如：

```text
(512, 768, 3)
```

OpenCV 读取图片：

```python
import cv2
image_cv = cv2.imread(img_path)
```

得到的也是 NumPy ndarray。注意：OpenCV 默认是 **BGR**，PIL 通常是 **RGB**。

#### torch.Tensor

用：

```python
from torchvision import transforms
tensor = transforms.ToTensor()(image)
```

得到的是：

```python
torch.Tensor
```

常见 shape 是：

```text
[C, H, W]
```

例如：

```text
[3, 512, 768]
```

Tensor 才是 PyTorch 模型真正能处理的格式。Tensor 有几个关键属性：

```python
tensor.shape
tensor.dtype
tensor.device
tensor.requires_grad
tensor.grad_fn
```

---

### 4. transforms 负责什么

`transforms` 负责把原始图片变成模型需要的标准输入。可以把它理解成图像预处理工具箱，里面有很多工具：

```python
transforms.ToTensor()
transforms.Resize()
transforms.RandomCrop()
transforms.Normalize()
transforms.Compose()
```

#### ToTensor

作用：

```text
PIL Image / NumPy ndarray → torch.Tensor
```

通常同时完成：

```text
HWC → CHW
0~255 → 0~1
uint8 → float32
```

#### Resize

作用：改变图片尺寸。

```python
transforms.Resize((512, 512))
```

强制变成 512×512，可能拉伸变形。

```python
transforms.Resize(512)
```

保持宽高比例，让短边变成 512。

#### RandomCrop

作用：随机裁剪。

```python
transforms.RandomCrop(512)
```

表示随机裁剪一个 512×512 的区域。它不是缩放，而是从原图中切出一块。

#### Normalize

作用：对 Tensor 按通道做标准化。

公式是：

```text
output = (input - mean) / std
```

例如：

```python
transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)
```

如果输入像素已经通过 `ToTensor` 变成 0~1，那么这个 Normalize 会把数据大致变成 -1~1。

#### Compose

作用：把多个 transform 按顺序串起来。

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
```

它表示：

```text
PIL Image
→ Resize
→ PIL Image
→ ToTensor
→ Tensor
→ Normalize
→ Tensor
```

核心原则：

> Compose 里前一个 transform 的输出类型，必须能作为后一个 transform 的输入类型。

---

### 5. Tensor 是什么

Tensor 是 PyTorch 的核心数据结构。

可以理解为：

> Tensor = 多维数组 + shape/dtype/device 等元信息 + 可选的梯度信息。

Tensor 常见属性：

| 属性 | 作用 |
|---|---|
| `shape` | 维度，例如 `[3,224,224]`、`[32,3,224,224]` |
| `dtype` | 数值类型，例如 `torch.float32`、`torch.float16` |
| `device` | 数据在哪，例如 `cpu`、`cuda:0` |
| `requires_grad` | 是否需要计算梯度 |
| `grad_fn` | 记录这个 Tensor 是由哪个计算操作生成的 |
| `grad` | 反向传播后得到的梯度 |

为什么模型必须吃 Tensor？

因为神经网络底层做的是大量矩阵运算、卷积运算、向量运算，这些都要用统一的数据结构。PIL Image 只是图片对象，NumPy ndarray 主要在 CPU 上做数组处理，而 Tensor 可以和 PyTorch 模型、Autograd、GPU 计算连接起来。

---

### 6. TensorBoard 负责什么

TensorBoard 负责把训练过程中的数字和图片可视化。它不是模型，也不是训练算法，而是观察工具。

核心 API：

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
writer.add_scalar("Loss/train", loss_value, step)
writer.add_image("image", image_tensor, step)
writer.close()
```

启动：

```bash
tensorboard --logdir=logs
```

#### SummaryWriter

负责把数据写进日志文件。

#### add_scalar

记录标量曲线。例如：

```python
writer.add_scalar("Loss/train", loss.item(), step)
```

含义：

```text
tag = Loss/train，图表名
scalar_value = loss.item()，y 轴
global_step = step，x 轴
```

#### add_image

记录图片。例如：

```python
writer.add_image("input", image_tensor, step)
```

如果 image 是 Tensor，一般是：

```text
[C,H,W]
```

如果 image 是 NumPy，一般是：

```text
[H,W,C]
```

这时要写：

```python
writer.add_image("input", image_np, step, dataformats="HWC")
```

---

### 7. 这些内容和后续训练循环的关系

完整训练循环大概是：

```python
for images, labels in train_loader:
    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

这些课对应的是：

```text
images, labels 是怎么来的？
```

答案是：

```text
Dataset 负责定义单个样本怎么读取
transforms 负责把图片变成 Tensor
DataLoader 负责把多个样本组成 batch
TensorBoard 负责观察图片和 loss
```

也就是说：

```text
Dataset.__getitem__ 返回单张图片 Tensor [C,H,W] 和 label
DataLoader 把多张图片堆成 batch [B,C,H,W]
model(images) 接收 batch Tensor
criterion(outputs, labels) 计算 loss
```

---

### 8. 这些内容和 AI Infra 推理的关系

你未来学 AI Infra 推理时，虽然会接触 vLLM、TensorRT、ONNX、CUDA、KV Cache，但底层意识是一样的：

```text
数据是什么类型？
shape 是多少？
dtype 是什么？
layout 是什么？
device 在哪里？
batch 怎么组织？
显存占用多少？
```

图像任务中：

```text
PIL / NumPy / Tensor
HWC / CHW / NCHW
float32 / float16 / int8
CPU / GPU
batch size
```

LLM 推理中：

```text
input_ids Tensor
attention mask
batch size
sequence length
dtype
device
KV Cache shape
KV Cache dtype
GPU memory
dynamic batching
```

所以这些基础知识不是只服务于图像分类，而是在训练你理解 AI 系统最基础的几个抽象：

```text
数据读取
预处理
Tensor 表示
batch 组织
可视化/debug
shape/dtype/device/layout 管理
```

---

## 三、知识点去重表

| 知识点 | 在流程中的位置 | 解决什么问题 | 关键 API / 代码 | 最容易混淆的地方 | 和 AI Infra 推理的关系 |
|---|---|---|---|---|---|
| `Dataset` | 数据读取层 | 定义如何按 index 读取一个样本 | `class MyData(Dataset)` | 它不是模型，只是数据接口 | 推理数据集、benchmark dataset 也需要样本读取逻辑 |
| `__init__` | Dataset 初始化 | 保存路径、label、图片列表 | `def __init__(...)` | 不是每次取样本都执行，只在创建对象时执行 | 初始化 tokenizer、engine、数据管线配置也类似 |
| `__getitem__` | 单样本读取 | 根据 index 返回 image,label | `dataset[index]` | 它返回的是一个样本，不是整个 batch | 推理请求中单条样本构造类似 |
| `__len__` | 数据集长度 | 告诉 PyTorch 有多少样本 | `len(dataset)` | 返回图片数量，不是 batch 数 | benchmark 需要知道请求/样本总数 |
| `self` | 类内部共享变量 | 让不同方法共享对象属性 | `self.img_dir` | 局部变量和对象属性混淆 | 读框架源码时必须理解对象状态 |
| `index` | 样本索引 | 指定取第几个样本 | `self.img_names[index]` | Python 从 0 开始索引 | batch/request/token position 都依赖索引 |
| `os.listdir` | 扫描文件夹 | 获取图片文件名列表 | `os.listdir(path)` | 只返回文件名，不返回完整路径 | 数据加载、日志扫描、模型文件读取常用 |
| `os.path.join` | 拼接路径 | 跨系统安全拼路径 | `os.path.join(a,b)` | 不要手写 `/` 或 `\\` | Linux 部署、服务器路径管理基础 |
| `PIL.Image.open` | 图片解码 | 从磁盘读出 PIL 图片 | `Image.open(path)` | 得到的是 PIL Image，不是 Tensor | 推理服务中图片解码第一步 |
| `NumPy ndarray` | CPU 数组 | 表示图片数组，常见 HWC | `np.array(img)` | NumPy 不是 Tensor，不能直接喂 PyTorch 模型 | ONNX Runtime 常用 NumPy 输入 |
| `torch.Tensor` | 模型输入 | PyTorch 计算核心数据结构 | `transforms.ToTensor()` | Tensor 有 shape/dtype/device | 推理引擎核心处理对象 |
| `shape` | 任意 Tensor 阶段 | 表示维度结构 | `x.shape` | PIL 的 size 和 Tensor shape 顺序不同 | 动态 shape、batching、KV Cache 基础 |
| `dtype` | Tensor 数值属性 | 表示数值类型 | `x.dtype` | uint8、float32、float16 混淆 | FP32/FP16/INT8 性能和精度关键 |
| `device` | Tensor 存放位置 | 判断 CPU/GPU | `x.device`, `x.to("cuda")` | 模型和输入 device 不一致会报错 | 显存管理、CUDA 执行基础 |
| HWC | 图像数组阶段 | NumPy/OpenCV 常见格式 | `[H,W,C]` | 通道在最后 | NHWC/NCHW 转换影响推理性能 |
| CHW | 单张 Tensor 阶段 | PyTorch 单张图常见格式 | `[C,H,W]` | 和 HWC 混淆 | TensorRT/ONNX layout 相关 |
| NCHW | batch Tensor 阶段 | PyTorch batch 图像格式 | `[B,C,H,W]` | 忘记 batch 维度 | batching、吞吐、显存基础 |
| `transforms` | 预处理阶段 | 图片变换工具箱 | `from torchvision import transforms` | 它不是一个单独函数，而是工具集合 | 推理 preprocess pipeline 对应它 |
| `ToTensor` | PIL/NumPy → Tensor | 转类型、变维度、缩放数值 | `transforms.ToTensor()` | 会 HWC→CHW，0~255→0~1 | 部署时必须复现 |
| `Resize` | 几何预处理 | 统一图片尺寸 | `transforms.Resize(...)` | `Resize(512)` 和 `Resize((512,512))` 不同 | 输入 shape、计算量、TensorRT profile 相关 |
| `RandomCrop` | 数据增强 | 随机裁剪局部区域 | `transforms.RandomCrop(...)` | 它是裁剪，不是缩放 | 训练可随机，推理通常要确定性 |
| `Normalize` | 数值预处理 | 按通道标准化 | `transforms.Normalize(mean,std)` | 必须在 ToTensor 后面 | 部署漏 Normalize 会导致精度崩 |
| `Compose` | transform 流水线 | 串联多个 transforms | `transforms.Compose([...])` | 前后输入输出类型必须匹配 | 推理 pipeline 本质类似 |
| `SummaryWriter` | 可视化日志 | 写 TensorBoard 事件文件 | `SummaryWriter("logs")` | 它不负责打开网页 | 推理监控/日志系统的入门 |
| `add_scalar` | 训练指标可视化 | 画 loss 等曲线 | `writer.add_scalar(tag,value,step)` | x 轴和 y 轴参数别搞反 | latency、throughput、tokens/s 也都是 scalar |
| `add_image` | 图像可视化 | 显示输入/输出图片 | `writer.add_image(...)` | Tensor 用 CHW，NumPy 用 HWC | 推理 debug 输入输出很重要 |
| `loss` | 训练循环 | 衡量预测和真实标签差距 | `criterion(outputs, labels)` | loss 不是 accuracy | 部署前模型质量评估、量化精度评估 |
| `DataLoader` | batch 组织 | 把多个样本组成 batch | `DataLoader(dataset,batch_size=...)` | Dataset 返回单样本，DataLoader 返回 batch | dynamic batching、吞吐优化基础 |
| `batch` | 模型输入阶段 | 一次处理多个样本 | `[B,C,H,W]` | batch 维度不是通道维度 | 推理吞吐、延迟、显存权衡核心 |

---

## 四、完整代码：把这些课串起来

```python
import os
import math
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter


class MyData(Dataset):
    def __init__(self, root_dir, label_dir, class_to_idx, transform=None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.class_to_idx = class_to_idx
        self.transform = transform

        self.img_dir = os.path.join(root_dir, label_dir)

        self.img_names = sorted([
            name for name in os.listdir(self.img_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[self.label_dir]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.img_names)


mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

root_dir = "data/train"
class_to_idx = {"ants": 0, "bees": 1}

ants_dataset = MyData(root_dir, "ants", class_to_idx, transform=transform)
bees_dataset = MyData(root_dir, "bees", class_to_idx, transform=transform)

train_dataset = ConcatDataset([ants_dataset, bees_dataset])

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

images, labels = next(iter(train_loader))

print("images type:", type(images))
print("images shape:", images.shape)   # [B, C, H, W]
print("images dtype:", images.dtype)
print("images device:", images.device)

print("labels type:", type(labels))
print("labels shape:", labels.shape)   # [B]
print("labels dtype:", labels.dtype)
print("labels device:", labels.device)
print("labels:", labels)

writer = SummaryWriter("logs/pytorch_data_pipeline_demo")


def denormalize(batch_images, mean, std):
    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std).view(1, 3, 1, 1)

    imgs = batch_images * std_tensor + mean_tensor
    imgs = torch.clamp(imgs, 0, 1)
    return imgs


images_for_show = denormalize(images, mean, std)
grid = make_grid(images_for_show, nrow=4)
writer.add_image("batch_images_after_transform", grid, global_step=0)

for step in range(100):
    fake_loss = math.exp(-step / 30) + 0.02 * math.sin(step)
    writer.add_scalar("Loss/train_fake", fake_loss, step)

writer.close()

print("TensorBoard logs saved to: logs/pytorch_data_pipeline_demo")
print("Run: tensorboard --logdir=logs")
```

---

## 五、逐段解释完整代码

### 1. 导入库

| 工具 | 作用 |
|---|---|
| `os` | 处理文件夹和路径 |
| `math` | 生成模拟 loss 曲线 |
| `Image` | 读取图片 |
| `Dataset` | 自定义数据集基类 |
| `DataLoader` | 组成 batch |
| `ConcatDataset` | 合并 ants 和 bees |
| `transforms` | 图片预处理 |
| `make_grid` | 把多张图片拼成网格 |
| `SummaryWriter` | 写 TensorBoard 日志 |

### 2. `class MyData(Dataset)`

这表示你自定义了一个数据集类，并继承 PyTorch 的 `Dataset`。这一步解决的问题是：

```text
让你的图片文件夹变成 PyTorch 能识别的数据集对象
```

### 3. `__init__`

输入：

```text
root_dir: 根目录 data/train
label_dir: 类别文件夹 ants 或 bees
class_to_idx: 类别名到数字的映射
transform: 图片预处理流程
```

关键代码：

```python
self.img_dir = os.path.join(root_dir, label_dir)
```

如果：

```python
root_dir = "data/train"
label_dir = "ants"
```

那么：

```python
self.img_dir = "data/train/ants"
```

初学者容易错在：

```text
以为 os.listdir 返回完整路径
忘记 os.path.join
不保存 self.xxx，导致 __getitem__ 用不了
```

### 4. `__getitem__`

这段负责根据 index 取出一张图片和它的 label。

流程：

```text
img_names[index] 取文件名
os.path.join 拼完整路径
Image.open 读成 PIL Image
class_to_idx 把字符串 label 转数字
transform 把 PIL Image 转成 Tensor
return image, label
```

为什么 label 要转数字？

因为后续损失函数比如 `CrossEntropyLoss` 需要的是类别编号，不是字符串。

### 5. `__len__`

```python
def __len__(self):
    return len(self.img_names)
```

这段告诉 PyTorch 这个数据集有多少张图片。

### 6. 定义 transform

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
```

完整输入输出变化是：

```text
PIL Image
→ Resize((224,224))
→ PIL Image, size=(224,224)
→ ToTensor()
→ Tensor [3,224,224], float32, 0~1
→ Normalize()
→ Tensor [3,224,224], float32, 大致 -1~1
```

为什么顺序不能乱？

```text
Resize 通常处理 PIL Image
ToTensor 把 PIL 变成 Tensor
Normalize 需要 Tensor
```

### 7. DataLoader 组成 batch

`Dataset` 一次返回一个样本：

```text
image [3,224,224]
label int
```

`DataLoader` 一次返回一个 batch：

```text
images [4,3,224,224]
labels [4]
```

其中：

```text
B = batch size = 4
C = channel = 3
H = height = 224
W = width = 224
```

### 8. TensorBoard 显示 batch 图片

因为图片经过 Normalize 后数值不一定在 0~1，直接显示可能颜色异常，所以代码里做了反标准化：

```python
images_for_show = denormalize(images, mean, std)
```

然后用 `make_grid` 把一个 batch 的图片拼成网格，再写入 TensorBoard。

### 9. TensorBoard 显示模拟 loss

```python
for step in range(100):
    fake_loss = math.exp(-step / 30) + 0.02 * math.sin(step)
    writer.add_scalar("Loss/train_fake", fake_loss, step)
```

这模拟了一条逐渐下降的 loss 曲线。真实训练时会写：

```python
writer.add_scalar("Loss/train", loss.item(), global_step)
```

---

## 六、数据流主线图

```text
磁盘图片 jpg/png
    ↓
图片路径 img_path
    ↓
Image.open(img_path).convert("RGB")
    ↓
PIL Image
    作用：完成图片解码，得到 Python 中可处理的图片对象

    ↓
transforms.Resize((224,224))
    ↓
PIL Image, size=(224,224)
    作用：统一图片尺寸，保证后面能组成 batch

    ↓
transforms.ToTensor()
    ↓
Tensor [C,H,W], float32, 数值 0~1
    作用：把图片变成 PyTorch 模型能处理的 Tensor

    ↓
transforms.Normalize(mean,std)
    ↓
Tensor [C,H,W], float32, 标准化后的数值
    作用：调整输入分布，让模型训练/推理更符合预期

    ↓
Dataset.__getitem__(index)
    ↓
(image_tensor, label)
    作用：按 index 返回一个样本

    ↓
DataLoader
    ↓
Batch Tensor images [B,C,H,W], labels [B]
    作用：把多个样本堆成 batch，提高训练/推理效率

    ↓
model(images)
    ↓
outputs
    作用：模型前向计算，得到预测结果

    ↓
criterion(outputs, labels)
    ↓
loss
    作用：衡量预测和真实标签差距

    ↓
loss.backward()
    ↓
optimizer.step()
    作用：训练阶段根据 loss 更新模型参数
```

推理阶段则简化为：

```text
图片
→ 同样的预处理
→ Tensor [B,C,H,W]
→ model.eval()
→ torch.no_grad()
→ outputs
→ 后处理
```

推理不需要：

```text
loss.backward()
optimizer.step()
```

---

## 七、最容易混淆的概念对比表

### 1. PIL Image vs NumPy ndarray vs Tensor

| 对比项 | PIL Image | NumPy ndarray | torch.Tensor |
|---|---|---|---|
| 来源 | `Image.open()` | `np.array(img)` / `cv2.imread()` | `ToTensor()` / `torch.tensor()` |
| 常见用途 | 读图、显示、保存 | CPU 数组处理 | 模型输入、GPU 计算 |
| 形状 | `image.size=(W,H)` | `shape=[H,W,C]` | `shape=[C,H,W]` 或 `[B,C,H,W]` |
| 能否直接喂模型 | 不适合 | 不适合 PyTorch 模型 | 可以 |
| AI Infra 关系 | 解码阶段 | CPU 预处理阶段 | 推理引擎输入核心 |

### 2. HWC vs CHW vs NCHW

| 格式 | 示例 | 常见位置 |
|---|---|---|
| HWC | `[224,224,3]` | NumPy / OpenCV |
| CHW | `[3,224,224]` | 单张 PyTorch 图片 Tensor |
| NCHW | `[32,3,224,224]` | PyTorch batch 输入 |

### 3. `Resize(512)` vs `Resize((512,512))`

| 写法 | 含义 | 是否保持比例 |
|---|---|---|
| `Resize(512)` | 短边变成 512，长边等比例变化 | 保持 |
| `Resize((512,512))` | 强制变成 512×512 | 不一定保持 |

### 4. Resize vs RandomCrop

| 对比项 | Resize | RandomCrop |
|---|---|---|
| 本质 | 缩放 | 裁剪 |
| 是否保留完整图像 | 通常保留完整图像，只是缩放 | 只保留局部区域 |
| 是否随机 | 通常不随机 | 随机 |
| 常见用途 | 统一尺寸 | 数据增强 |
| 推理阶段是否常用 | 常用 | 一般不用随机版本 |

### 5. ToTensor vs Normalize

| 对比项 | ToTensor | Normalize |
|---|---|---|
| 输入 | PIL/NumPy | Tensor |
| 输出 | Tensor | Tensor |
| 作用 | 转类型、变维度、缩放到 0~1 | 按 mean/std 标准化 |
| 顺序 | 在 Normalize 前 | 在 ToTensor 后 |
| 常见错误 | 忘记 HWC→CHW | 直接对 PIL Normalize |

### 6. Dataset vs DataLoader

| 对比项 | Dataset | DataLoader |
|---|---|---|
| 作用 | 定义单个样本怎么读 | 负责 batch、shuffle、多进程加载 |
| 返回 | 一个样本 | 一个 batch |
| 示例 | `dataset[0]` | `for images, labels in loader:` |
| shape | `[C,H,W]` | `[B,C,H,W]` |

### 7. add_scalar vs add_image

| 对比项 | add_scalar | add_image |
|---|---|---|
| 记录内容 | 一个数字 | 一张图片 |
| 常见用途 | loss、accuracy、学习率 | 输入图片、输出图片、transform 效果 |
| 关键参数 | tag、value、step | tag、image、step、dataformats |
| TensorBoard 面板 | Scalars | Images |

### 8. loss vs accuracy

| 对比项 | loss | accuracy |
|---|---|---|
| 含义 | 预测和真实答案差多少 | 预测对了多少 |
| 类型 | 连续数值 | 比例/百分比 |
| 是否参与反向传播 | 是 | 通常不是 |
| 训练目标 | 通常让 loss 下降 | 希望 accuracy 上升 |

### 9. 训练阶段 transform vs 推理阶段 transform

| 对比项 | 训练阶段 | 推理阶段 |
|---|---|---|
| 是否可以随机 | 可以，例如 RandomCrop | 通常不随机 |
| 目标 | 增强数据、提高泛化 | 稳定、可复现 |
| 常见操作 | RandomCrop、RandomHorizontalFlip | Resize、CenterCrop、ToTensor、Normalize |
| 注意点 | 增强不能破坏 label | 必须和训练预处理一致 |

### 10. `model.train()` vs `model.eval()` 预告

| 对比项 | `model.train()` | `model.eval()` |
|---|---|---|
| 阶段 | 训练 | 验证/推理 |
| Dropout | 启用 | 关闭 |
| BatchNorm | 用当前 batch 统计 | 用训练好的统计 |
| 是否需要梯度 | 通常需要 | 通常配合 `torch.no_grad()` |
| AI Infra 关系 | 训练框架 | 推理部署必须 eval |

记住一句话：

> 训练用 `model.train()`，推理用 `model.eval()` + `torch.no_grad()`。

---

## 八、复习版精简笔记

### 1. 总主线

PyTorch 图像任务的前处理流程是：

```text
图片文件 → Dataset 读取 → PIL Image → transforms 预处理 → Tensor → DataLoader 组成 batch → model 输入
```

### 2. Dataset

`Dataset` 定义“一个样本怎么读”，核心是 `__init__` 准备路径和文件列表，`__getitem__` 根据 index 返回 `(image, label)`，`__len__` 返回数据集长度。

### 3. 路径

`os.listdir` 获取文件夹里的图片名，`os.path.join` 拼完整路径，避免 Windows/Linux 路径差异。

### 4. 图片类型

`Image.open` 得到 PIL Image，`np.array` 得到 NumPy ndarray，`ToTensor` 得到 PyTorch Tensor；模型最终需要 Tensor。

### 5. shape

PIL 的 `size` 是 `(W,H)`，NumPy 常见是 `[H,W,C]`，PyTorch 单张图是 `[C,H,W]`，batch 后是 `[B,C,H,W]`。

### 6. transforms

`transforms` 是图像预处理工具箱，每个 transform 都要看输入、输出和作用。

### 7. ToTensor

`ToTensor()` 把 PIL/NumPy 图片转成 Tensor，同时通常完成 `HWC→CHW`、`0~255→0~1`。

### 8. Resize

`Resize((H,W))` 强制缩放到指定高宽，`Resize(size)` 保持比例让短边变成 size。

### 9. RandomCrop

`RandomCrop(size)` 是随机裁剪，不是缩放，训练增强常用，推理阶段一般不用随机裁剪。

### 10. Normalize

`Normalize(mean,std)` 的公式是 `(input-mean)/std`，必须放在 `ToTensor()` 后面。

### 11. Compose

`Compose` 把多个 transforms 串起来，前一个 transform 的输出必须能作为后一个 transform 的输入。

### 12. Tensor

Tensor 有 `shape`、`dtype`、`device`、`requires_grad` 等属性，是 PyTorch 模型计算的核心数据结构。

### 13. DataLoader

`Dataset` 返回单个样本 `[C,H,W]`，`DataLoader` 把多个样本组成 batch `[B,C,H,W]`。

### 14. TensorBoard

`SummaryWriter` 写日志，`add_scalar` 画 loss 曲线，`add_image` 显示图片，启动命令是：

```bash
tensorboard --logdir=logs
```

### 15. loss

loss 表示模型预测和真实标签之间的差距，训练目标是通过反向传播让 loss 下降。

### 16. AI Infra 连接

ONNX、TensorRT、CUDA、vLLM 也都离不开 `shape / dtype / device / layout / batch / 显存`，所以现在学的数据处理基础是后续推理部署的底层语言。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-pytorch学习|模块-pytorch学习]]

%% 项目关联导航：结束 %%
