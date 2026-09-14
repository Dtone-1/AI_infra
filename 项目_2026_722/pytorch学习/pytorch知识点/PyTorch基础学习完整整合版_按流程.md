# PyTorch 基础学习完整整合版：按流程从图片到训练、GPU、推理、开源项目

> 适用对象：PyTorch 初学者；目标是打好后续学习 AI Infra / 推理引擎 / vLLM / TensorRT / ONNX / CUDA 的基础。  
> 整合原则：把阶段性笔记中重复内容合并，只保留一条清晰主线：**图片/数据 → 预处理 → Dataset → DataLoader → 模型 → forward → loss → backward → optimizer → test/eval → 保存加载 → GPU/device → 单张图片推理 → 开源项目阅读 → AI Infra 连接**。

---

## 0. 先给你一个整体判断

你现在学到的 PyTorch 内容，已经不是“单个 API 怎么用”的零散阶段，而是进入了 **PyTorch 图像分类项目的完整闭环阶段**。

这几份文件共同覆盖了三层内容：

1. **数据入口层**：图片文件、PIL、NumPy、Tensor、transforms、Dataset、DataLoader。
2. **模型计算层**：`nn.Module`、`forward()`、`Conv2d`、`MaxPool2d`、`Flatten`、`Linear`、logits、shape 变化。
3. **训练部署层**：loss、backward、optimizer、train/test loop、TensorBoard、保存加载、GPU、单张图片推理、开源项目阅读。

你可以把它理解成一条完整链路：

```text
图片/数据
  ↓
变成 Tensor
  ↓
组成 batch
  ↓
进入 CNN 模型
  ↓
输出 logits
  ↓
训练时：算 loss → 反向传播 → 更新参数
  ↓
测试/推理时：只 forward → argmax/后处理
  ↓
保存/加载模型
  ↓
放到 GPU 或部署环境中运行
```

这条链路学清楚后，你就具备了继续学习 AI Infra 推理方向的基础，因为推理系统本质上也是在处理：

```text
输入 Tensor → 模型 forward → 输出 logits/token → 后处理
```

只是 AI Infra 更关注 **显存、batching、latency、throughput、device、KV Cache、算子执行、开源推理框架源码**。

---

# 第一部分：总流程图

## 1. 完整 PyTorch 图像分类流程

```text
磁盘图片 / CIFAR10 数据集
    ↓
PIL.Image.open / torchvision.datasets.CIFAR10
    ↓
transforms.Resize / ToTensor / Normalize
    ↓
Dataset.__getitem__()
    ↓
DataLoader 组成 batch
    ↓
images, targets
    ↓
images.shape = [N, C, H, W]
targets.shape = [N]
    ↓
model(images)
    ↓
outputs / logits
outputs.shape = [N, num_classes]
    ↓
训练阶段：
loss_fn(outputs, targets)
    ↓
optimizer.zero_grad()
    ↓
loss.backward()
    ↓
optimizer.step()
    ↓
测试阶段：
model.eval()
with torch.no_grad()
accuracy / test_loss
    ↓
保存模型：
torch.save(model.state_dict(), "model.pth")
    ↓
加载模型：
model.load_state_dict(torch.load(...))
    ↓
单张图片推理：
image.unsqueeze(0)
output.argmax(1)
    ↓
类别编号 / 类别名称
```

## 2. 每一步到底在干什么

| 阶段 | 作用 | 核心代码 | 你要抓住的关键词 |
|---|---|---|---|
| 图片读取 | 把图片文件读入 Python | `Image.open(path)` | PIL、RGB、路径 |
| 预处理 | 把图片变成模型要求的格式 | `transforms.Compose([...])` | Resize、ToTensor、Normalize |
| Dataset | 定义单个样本怎么取 | `__getitem__` | image、label、index |
| DataLoader | 把多个样本组成 batch | `DataLoader(dataset,batch_size=64)` | batch、shuffle |
| 模型 | 定义 Tensor 如何计算 | `class Net(nn.Module)` | forward、参数 |
| 前向传播 | 输入变输出 | `outputs = model(images)` | logits、shape |
| 损失函数 | 衡量预测错多少 | `CrossEntropyLoss` | output `[N,C]`、target `[N]` |
| 反向传播 | 计算梯度 | `loss.backward()` | 计算图、梯度 |
| 优化器 | 更新参数 | `optimizer.step()` | lr、weight、bias |
| 测试/推理 | 只用模型输出结果 | `eval()` + `no_grad()` | 不更新参数 |
| 保存加载 | 保存训练结果 | `state_dict()` | 权重、checkpoint |
| GPU | 把计算放到 GPU | `.to(device)` | cuda、显存、device |
| 推理 Demo | 对外部图片预测 | `argmax(1)` | 后处理、类别映射 |
| 开源项目 | 看懂别人项目 | README / argparse / train.py | 工程结构 |

---

# 第二部分：数据处理阶段

## 1. 图片文件是怎么进入 PyTorch 的？

最开始你面对的是磁盘里的图片文件：

```text
data/train/ants/001.jpg
data/train/bees/002.jpg
```

图片文件本身不能直接输入模型。它要经历：

```text
图片路径
  ↓
PIL Image
  ↓
Tensor
  ↓
batch Tensor
  ↓
model(images)
```

### 关键类型对比

| 类型 | 怎么得到 | 常见 shape / 表示 | 用途 |
|---|---|---|---|
| PIL Image | `Image.open(path)` | `image.size = (W,H)` | 读图、显示、简单处理 |
| NumPy ndarray | `np.array(image)` | `[H,W,C]` | CPU 数组处理、OpenCV |
| torch.Tensor | `transforms.ToTensor()` | `[C,H,W]` 或 `[N,C,H,W]` | 模型输入、GPU 计算 |

记住：**PyTorch 模型最终需要 Tensor，不是 PIL，也不是普通图片路径。**

---

## 2. Dataset：定义“一个样本怎么取”

`Dataset` 解决的问题是：**给定一个 index，返回一张图片和它的 label。**

自定义 Dataset 通常有三个方法：

```python
class MyData(Dataset):
    def __init__(self, ...):
        ...

    def __getitem__(self, index):
        ...

    def __len__(self):
        ...
```

### 三个方法分别负责什么

| 方法 | 作用 | 类比理解 |
|---|---|---|
| `__init__` | 初始化路径、标签、图片列表 | 先把书架整理好 |
| `__getitem__` | 根据 index 取一个样本 | 给我第 10 本书 |
| `__len__` | 返回数据集长度 | 书架上一共有多少本书 |

### 最小 Dataset 示例

```python
import os
from PIL import Image
from torch.utils.data import Dataset

class MyData(Dataset):
    def __init__(self, root_dir, label_dir, transform=None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.transform = transform

        self.img_dir = os.path.join(root_dir, label_dir)
        self.img_names = os.listdir(self.img_dir)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        label = self.label_dir

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.img_names)
```

### 常见坑

| 坑 | 原因 | 正确理解 |
|---|---|---|
| `os.listdir` 只返回文件名 | 它不会返回完整路径 | 要用 `os.path.join` 拼完整路径 |
| 不写 `self.xxx` | 变量只在当前函数有效 | 跨方法使用必须写 `self.xxx` |
| label 是字符串 | 模型和 loss 更喜欢数字 label | 需要 `class_to_idx` |
| PIL 图片没转 Tensor | 模型不能直接处理 PIL | 用 `transforms.ToTensor()` |

---

## 3. transforms：图片预处理流水线

`transforms` 解决的问题是：**把原始图片变成模型需要的 Tensor 格式。**

常见 transform：

| API | 作用 | 输入 | 输出 |
|---|---|---|---|
| `Resize` | 改图片尺寸 | PIL / Tensor | PIL / Tensor |
| `RandomCrop` | 随机裁剪 | PIL / Tensor | PIL / Tensor |
| `ToTensor` | 图片转 Tensor | PIL / NumPy | Tensor |
| `Normalize` | 标准化数值 | Tensor | Tensor |
| `Compose` | 串联多个变换 | 取决于第一步 | 取决于最后一步 |

### 推荐写法

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
```

输入输出变化：

```text
PIL Image
  ↓ Resize((32,32))
PIL Image, 32×32
  ↓ ToTensor()
Tensor [3,32,32], float32, 数值 0~1
```

### ToTensor 做了什么？

`ToTensor()` 通常做三件事：

```text
PIL / NumPy → torch.Tensor
HWC → CHW
0~255 → 0~1
```

比如一张 RGB 图片：

```text
原始 NumPy: [H,W,C]
PyTorch Tensor: [C,H,W]
```

### Normalize 做了什么？

公式：

```text
output = (input - mean) / std
```

例如：

```python
transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
```

如果输入在 `[0,1]`，输出大致变到 `[-1,1]`。

### 训练和推理 transform 的区别

| 阶段 | transform 特点 |
|---|---|
| 训练 | 可以有随机增强，如 RandomCrop、RandomHorizontalFlip |
| 推理 | 应该稳定、确定，通常 Resize / CenterCrop / ToTensor / Normalize |

---

## 4. DataLoader：从单张图片到 batch

`Dataset` 返回一个样本：

```text
image: [3,32,32]
label: scalar
```

`DataLoader` 返回一个 batch：

```text
images: [64,3,32,32]
targets: [64]
```

### 代码

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    drop_last=False
)
```

### 参数解释

| 参数 | 作用 |
|---|---|
| `batch_size` | 每次取多少个样本 |
| `shuffle` | 是否打乱顺序 |
| `num_workers` | 用几个进程加载数据 |
| `drop_last` | 最后一个不足 batch_size 的 batch 是否丢掉 |

### 常见坑

| 坑 | 解释 |
|---|---|
| Dataset 和 DataLoader 混淆 | Dataset 管单样本，DataLoader 管 batch |
| batch_size 太大 | 可能显存爆掉 |
| 最后一个 batch 不足 64 | 正常现象 |
| Windows 上 num_workers 报错 | 初学先设 0 |

---

# 第三部分：Tensor、shape、dtype、device 主线

## 1. Tensor 是什么？

Tensor 可以理解为：

```text
Tensor = 多维数组 + shape + dtype + device + 是否需要梯度
```

常看属性：

```python
x.shape
x.dtype
x.device
x.requires_grad
x.grad_fn
```

## 2. 图像任务常见 shape

| 场景 | shape | 解释 |
|---|---|---|
| PIL Image | `(W,H)` | PIL 的 size 宽在前 |
| NumPy 图片 | `[H,W,C]` | OpenCV/NumPy 常见 |
| 单张 Tensor 图片 | `[C,H,W]` | PyTorch 单图 |
| batch 图片 | `[N,C,H,W]` | PyTorch 模型输入 |
| CIFAR10 单图 | `[3,32,32]` | RGB 32×32 |
| CIFAR10 batch | `[64,3,32,32]` | 64 张图片 |
| 分类输出 | `[64,10]` | 每张图 10 类 logits |
| target | `[64]` | 每张图一个类别编号 |

## 3. 为什么模型需要 batch 维度？

CNN 的输入通常是四维：

```text
[N,C,H,W]
```

即使只有一张图片，也要写成：

```text
[1,3,32,32]
```

因为模型的第一维默认表示 batch。

单张图推理时：

```python
image = image.unsqueeze(0)
```

变化：

```text
[3,32,32] → [1,3,32,32]
```

## 4. dtype 是什么？

`dtype` 是 Tensor 的数值类型。

常见：

| dtype | 说明 |
|---|---|
| `torch.float32` | 训练最常用 |
| `torch.float16` | GPU 推理/训练加速常用 |
| `torch.bfloat16` | 大模型训练/推理常用 |
| `torch.int64` / `torch.long` | 分类 target 常用 |
| `torch.int8` | 量化推理常用 |

`CrossEntropyLoss` 的 target 通常应该是 `torch.long`，不是 float。

## 5. device 是什么？

`device` 表示 Tensor 在哪里：

```text
cpu
cuda:0
cuda:1
```

模型参数也是 Tensor，所以模型也有 device。

错误示例：

```text
model 在 cuda:0
images 在 cpu
```

会报 device mismatch。

推荐写法：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
images = images.to(device)
targets = targets.to(device)
```

---

# 第四部分：模型定义阶段

## 1. nn.Module：模型的标准写法

```python
from torch import nn

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(...)

    def forward(self, x):
        return self.model(x)
```

### 重点理解

| 部分 | 作用 |
|---|---|
| `nn.Module` | PyTorch 所有模型的基类 |
| `__init__` | 定义网络层 |
| `super().__init__()` | 初始化父类，注册参数和子模块 |
| `forward` | 定义输入如何变成输出 |
| `model(x)` | 自动调用 `forward(x)` |

---

## 2. CNN 模型中的层

### Conv2d

```python
nn.Conv2d(3, 32, kernel_size=5, padding=2)
```

含义：

| 参数 | 含义 |
|---|---|
| `3` | 输入通道 RGB |
| `32` | 输出通道，提取 32 种特征 |
| `kernel_size=5` | 5×5 卷积核 |
| `padding=2` | 保持 H/W 不变 |

shape：

```text
[64,3,32,32] → [64,32,32,32]
```

### ReLU

```python
nn.ReLU()
```

作用：引入非线性。通常不改变 shape。

### MaxPool2d

```python
nn.MaxPool2d(2)
```

作用：让 H/W 减半。

```text
[64,32,32,32] → [64,32,16,16]
```

### Flatten

```python
nn.Flatten()
```

作用：把 `[N,C,H,W]` 变成 `[N,C*H*W]`。

```text
[64,64,4,4] → [64,1024]
```

### Linear

```python
nn.Linear(1024, 10)
```

作用：把特征映射到类别 logits。

```text
[64,1024] → [64,10]
```

---

## 3. 完整 CNN shape 变化

以 CIFAR10 输入 `[64,3,32,32]` 为例：

| 步骤 | 层 | 输出 shape |
|---|---|---|
| 输入 | images | `[64,3,32,32]` |
| 1 | Conv2d(3→32, padding=2) | `[64,32,32,32]` |
| 2 | MaxPool2d(2) | `[64,32,16,16]` |
| 3 | Conv2d(32→32, padding=2) | `[64,32,16,16]` |
| 4 | MaxPool2d(2) | `[64,32,8,8]` |
| 5 | Conv2d(32→64, padding=2) | `[64,64,8,8]` |
| 6 | MaxPool2d(2) | `[64,64,4,4]` |
| 7 | Flatten | `[64,1024]` |
| 8 | Linear(1024→64) | `[64,64]` |
| 9 | Linear(64→10) | `[64,10]` |

---

# 第五部分：训练阶段

## 1. 训练的本质

训练的目标是：

```text
让模型参数不断调整，使 loss 下降。
```

核心模板：

```python
model.train()

for images, targets in train_loader:
    images = images.to(device)
    targets = targets.to(device)

    outputs = model(images)
    loss = loss_fn(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 2. forward：模型输出 logits

```python
outputs = model(images)
```

输入：

```text
images: [64,3,32,32]
```

输出：

```text
outputs: [64,10]
```

这 10 个数是 logits，不是概率。

## 3. CrossEntropyLoss

```python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(outputs, targets)
```

要求：

```text
outputs: [N,C]
targets: [N]
targets dtype: torch.long
```

注意：不要在 `CrossEntropyLoss` 前手动 softmax。

## 4. zero_grad / backward / step

### `optimizer.zero_grad()`

清空上一轮梯度。PyTorch 默认梯度会累加。

### `loss.backward()`

沿计算图反向传播，计算每个参数的梯度。

### `optimizer.step()`

根据梯度更新模型参数。

## 5. epoch、batch、step

| 概念 | 含义 |
|---|---|
| epoch | 完整训练集过一遍 |
| batch | 一次送进模型的一批样本 |
| step / iteration | 一个 batch 完成一次参数更新 |

如果训练集 50000 张，batch_size=64，一个 epoch 大约有 782 个 step。

---

# 第六部分：测试/验证/推理阶段

## 1. 测试和训练的区别

测试阶段不更新参数，只评估模型表现：

```python
model.eval()

with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, targets)
        preds = outputs.argmax(1)
```

## 2. model.eval()

作用：切换推理模式，影响 Dropout、BatchNorm。

注意：`eval()` 不会关闭梯度。

## 3. torch.no_grad()

作用：关闭梯度计算，不构建反向传播图。

注意：`no_grad()` 不会切换 Dropout/BatchNorm 行为。

标准推理写法：

```python
model.eval()
with torch.no_grad():
    outputs = model(inputs)
```

## 4. accuracy

```python
preds = outputs.argmax(1)
correct = (preds == targets).sum().item()
accuracy = correct / len(test_data)
```

`argmax(1)` 是沿类别维度取最大值。

---

# 第七部分：TensorBoard 记录

TensorBoard 用来记录训练指标：

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
writer.add_scalar("train_loss", loss.item(), total_train_step)
writer.add_scalar("test_loss", avg_test_loss, epoch)
writer.add_scalar("test_accuracy", accuracy, epoch)
writer.close()
```

启动：

```bash
tensorboard --logdir=logs
```

## 常见坑

| 坑 | 说明 |
|---|---|
| step 不变 | 曲线画不出来 |
| tag 混乱 | train/test 分不清 |
| 忘记 close | 日志可能没写完整 |
| 频繁 `.item()` | GPU 上可能触发同步，影响性能测试 |

---

# 第八部分：模型保存与加载

## 1. 推荐保存 state_dict

```python
torch.save(model.state_dict(), "model.pth")
```

加载：

```python
model = Tudui()
state_dict = torch.load("model.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
```

## 2. 保存整个模型

```python
torch.save(model, "model_full.pth")
model = torch.load("model_full.pth")
```

缺点：依赖原来的 Python 类定义和路径，不如 `state_dict` 稳。

## 3. map_location

GPU 保存的模型，在 CPU 加载时：

```python
torch.load("model.pth", map_location="cpu")
```

这是跨设备加载常见写法。

---

# 第九部分：GPU / device 阶段

## 1. `.cuda()` 写法

```python
if torch.cuda.is_available():
    model = model.cuda()
    loss_fn = loss_fn.cuda()

for images, targets in train_loader:
    if torch.cuda.is_available():
        images = images.cuda()
        targets = targets.cuda()
```

## 2. `.to(device)` 推荐写法

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
loss_fn = loss_fn.to(device)

for images, targets in train_loader:
    images = images.to(device)
    targets = targets.to(device)
```

## 3. 为什么推荐 `.to(device)`

| `.cuda()` | `.to(device)` |
|---|---|
| 只能 CUDA | CPU/GPU/多设备更通用 |
| 写死设备 | 可通过变量控制 |
| 简单直观 | 工程更常用 |
| 多卡不灵活 | 可写 `cuda:0`、`cuda:1` |

## 4. GPU 为什么快？

GPU 擅长大规模并行计算：

```text
卷积
矩阵乘法
backward
Linear/GEMM
```

但如果 batch 太小、数据搬运太频繁，GPU 利用率也可能不高。

---

# 第十部分：单张图片推理

## 1. 推理完整流程

```text
外部图片
  ↓
Image.open().convert("RGB")
  ↓
Resize((32,32))
  ↓
ToTensor()
  ↓
unsqueeze(0)
  ↓
to(device)
  ↓
load model
  ↓
model.eval()
  ↓
no_grad forward
  ↓
argmax(1)
  ↓
类别名称
```

## 2. 推理代码

```python
import torch
from PIL import Image
from torchvision import transforms
from model import Tudui

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image = Image.open("./images/dog.png").convert("RGB")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

image = transform(image)       # [3,32,32]
image = image.unsqueeze(0)     # [1,3,32,32]
image = image.to(device)

model = Tudui().to(device)
state_dict = torch.load("model.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    output = model(image)
    pred_idx = output.argmax(1).item()

print(classes[pred_idx])
```

## 3. 常见坑

| 坑 | 解释 |
|---|---|
| 忘记 `convert("RGB")` | PNG 可能是 4 通道，灰度图可能是 1 通道 |
| 忘记 Resize | 输入尺寸和模型不匹配 |
| 忘记 unsqueeze | `[3,32,32]` 少 batch 维 |
| 忘记 eval/no_grad | 推理不规范、浪费显存 |
| 类别表顺序错 | 预测编号解释错 |

---

# 第十一部分：开源项目阅读

## 1. 读项目顺序

```text
README
  ↓
训练命令 / 测试命令
  ↓
argparse / config
  ↓
train.py
  ↓
dataset / dataloader
  ↓
model
  ↓
loss / optimizer / scheduler
  ↓
test.py / demo.py
  ↓
checkpoint 保存加载
```

## 2. argparse

开源项目经常这样运行：

```bash
python train.py --dataroot ./datasets/maps --name exp1 --model pix2pix
```

代码里对应：

```python
parser.add_argument("--dataroot", type=str, required=True)
parser.add_argument("--name", type=str, default="experiment")
parser.add_argument("--model", type=str, default="pix2pix")
```

### required 和 default

| 参数 | 含义 |
|---|---|
| `required=True` | 必须从命令行传 |
| `default=...` | 不传就用默认值 |
| `type=str/int/float` | 把命令行字符串转成对应类型 |

如果 PyCharm 右键运行报错缺参数，可以：

1. 在 Run Configuration 中填参数；
2. 临时把 `required=True` 改成 `default=...` 方便调试。

---

# 第十二部分：完整代码整合版

## 1. `model.py`

```python
import torch
from torch import nn


class Tudui(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = Tudui()
    x = torch.ones((64, 3, 32, 32))
    y = model(x)
    print(y.shape)
```

## 2. `train.py`

```python
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model import Tudui


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

test_data = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

model = Tudui().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

writer = SummaryWriter("logs")

num_epochs = 10
total_train_step = 0

for epoch in range(num_epochs):
    print(f"-------- epoch {epoch + 1} --------")

    model.train()

    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1

        if total_train_step % 100 == 0:
            print(f"step={total_train_step}, loss={loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    model.eval()
    total_test_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, targets)

            total_test_loss += loss.item()
            total_correct += (outputs.argmax(1) == targets).sum().item()

    avg_test_loss = total_test_loss / len(test_loader)
    accuracy = total_correct / len(test_data)

    print(f"test_loss={avg_test_loss:.4f}, accuracy={accuracy:.4f}")

    writer.add_scalar("test_loss", avg_test_loss, epoch + 1)
    writer.add_scalar("test_accuracy", accuracy, epoch + 1)

    torch.save(model.state_dict(), f"tudui_epoch_{epoch + 1}.pth")

writer.close()
```

## 3. `predict.py`

```python
import torch
from PIL import Image
from torchvision import transforms

from model import Tudui


classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image = Image.open("./images/dog.png").convert("RGB")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

image = transform(image)
image = image.unsqueeze(0)
image = image.to(device)

model = Tudui().to(device)
state_dict = torch.load("tudui_epoch_10.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    output = model(image)
    pred_idx = output.argmax(1).item()

print("预测类别:", classes[pred_idx])
```

---

# 第十三部分：常见错误总表

| 错误现象 | 常见原因 | 如何排查 | 正确写法 |
|---|---|---|---|
| 模型输入维度错误 | 单张图片少 batch 维 | 打印 `image.shape` | `image = image.unsqueeze(0)` |
| Linear shape mismatch | Flatten 后维度算错 | 打印每层 shape | `nn.Linear(64*4*4, 64)` |
| target dtype 错 | CrossEntropyLoss 需要 long | 打印 `targets.dtype` | `targets = targets.long()` |
| device mismatch | model/data 不在同设备 | 打印 `.device` | 全部 `.to(device)` |
| GPU 模型 CPU 加载失败 | 没写 map_location | 看报错提示 | `torch.load(path,map_location='cpu')` |
| loss 不下降 | 学习率/模型/数据问题 | 看 loss 曲线 | 调 lr、检查数据 |
| 忘记 zero_grad | 梯度累加 | 观察 loss 异常 | 每步 `optimizer.zero_grad()` |
| backward 后参数没变 | 忘记 step | 检查代码 | `optimizer.step()` |
| 测试显存高 | 忘记 no_grad | 看测试代码 | `with torch.no_grad()` |
| 推理结果不稳定 | 忘记 eval | 检查 Dropout/BN | `model.eval()` |
| 预测类别解释错 | classes 顺序错 | 对照数据集类别 | 使用正确 label map |
| TensorBoard 无曲线 | step 不变或 logdir 错 | 检查日志目录 | `add_scalar(tag,value,step)` |
| 右键运行开源项目报错 | argparse required 缺参数 | 搜 `required=True` | 命令行传参或 default |

---

# 第十四部分：和 AI Infra 推理方向的连接

## 1. 数据处理 → 推理请求预处理

PyTorch 中图片要经过 Resize、ToTensor、Normalize。推理服务中，用户请求也要经过解码、预处理、batch 组织。预处理错误会直接导致推理结果错误。

## 2. batch → continuous batching

CNN 的 batch 是 `[N,3,32,32]`。LLM 的 batch 是多个 prompt/request。vLLM 的 continuous batching 解决的是动态请求如何组成 batch，提高 GPU 利用率。

## 3. forward → 推理引擎核心

训练和推理都要 forward。AI Infra 推理优化的核心就是让 forward 更快，例如算子融合、kernel 优化、Tensor Core、CUDA Graph、图优化。

## 4. no_grad → 推理省显存

训练需要计算图和梯度，推理不需要。`no_grad` / `inference_mode` 是 PyTorch 层面的推理优化基础。

## 5. device / CUDA → GPU 推理

`.to(device)` 背后是 CPU 内存到 GPU 显存的数据搬运。推理系统必须管理权重、activation、KV Cache、输入输出 Tensor 的位置。

## 6. state_dict → 大模型权重加载

PyTorch 小模型保存 `state_dict`。大模型通常保存为 `.bin`、`.safetensors`、分片权重。思想仍然是结构和权重分离。

## 7. argmax → token sampling

图像分类用 argmax 从 logits 得类别。LLM 用 greedy/top-k/top-p/temperature 从 logits 得下一个 token。它们都是后处理。

## 8. TensorBoard → benchmark 监控

训练中记录 loss/accuracy。推理中记录 latency、throughput、QPS、tokens/s、显存占用、KV Cache 使用率。

## 9. 开源项目阅读 → vLLM / TensorRT / SGLang

读 PyTorch 项目时找 README、train.py、test.py、argparse。读 vLLM 时也要找入口、engine、scheduler、worker、KV Cache、model runner、sampling。

---

# 第十五部分：推荐复习顺序

## 第 1 天：数据流

重点复习：

```text
图片 → PIL → Tensor → Dataset → DataLoader
```

必须能解释：

```text
PIL / NumPy / Tensor 的区别
HWC / CHW / NCHW 的区别
```

## 第 2 天：transforms 和 DataLoader

重点复习：

```text
Resize / ToTensor / Normalize / Compose
batch_size / shuffle / num_workers
```

## 第 3 天：CNN 模型结构

重点复习：

```text
nn.Module / forward / Conv2d / MaxPool2d / Flatten / Linear
```

必须手算 CIFAR10 的 shape 变化。

## 第 4 天：训练流程

重点背熟：

```python
outputs = model(images)
loss = loss_fn(outputs, targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## 第 5 天：测试、保存、推理

重点区分：

```text
训练：train + backward + step
推理：eval + no_grad + forward + argmax
```

## 第 6 天：GPU/device

重点掌握：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
images = images.to(device)
```

## 第 7 天：开源项目和 AI Infra 连接

找一个小项目，按：

```text
README → train.py → argparse → data → model → loss → optimizer → test.py
```

读一遍。然后把这些概念对应到 vLLM / TensorRT / ONNX Runtime。

---

# 最后一页：一张总表记住全流程

| 阶段 | 你要问自己的问题 | 代表代码 |
|---|---|---|
| 数据 | 图片怎么读？label 怎么来？ | `Dataset.__getitem__` |
| 预处理 | 输入尺寸、数值范围对吗？ | `transforms.Compose` |
| batch | shape 是 `[N,C,H,W]` 吗？ | `DataLoader` |
| 模型 | 每层 shape 能接上吗？ | `nn.Module.forward` |
| 输出 | logits 是 `[N,num_classes]` 吗？ | `outputs = model(images)` |
| loss | output/target 格式对吗？ | `CrossEntropyLoss` |
| 反传 | 梯度清零了吗？ | `zero_grad/backward/step` |
| 测试 | eval/no_grad 写了吗？ | `model.eval()` |
| 保存 | 保存的是 state_dict 吗？ | `torch.save(model.state_dict())` |
| 加载 | device 映射对吗？ | `map_location` |
| 推理 | 单图加 batch 维了吗？ | `unsqueeze(0)` |
| GPU | model/data 同 device 吗？ | `.to(device)` |
| 开源项目 | 参数从哪来？入口在哪？ | `argparse + train.py` |
| AI Infra | forward 怎么更快？显存怎么管？ | vLLM/TensorRT/ONNX |

---

## 最终总结

这几份 PyTorch 学习文件整合后，真正要记住的是：

```text
PyTorch 基础不是一堆 API，而是一条数据和 Tensor 的流动路线。
```

从图片到模型训练：

```text
图片 → Tensor → batch → model → logits → loss → backward → optimizer
```

从训练到推理：

```text
训练好的权重 → load_state_dict → eval → no_grad → forward → argmax/后处理
```

从 PyTorch 到 AI Infra：

```text
forward → GPU → batch → 显存 → latency → throughput → KV Cache → 推理引擎
```

你现在已经完成了 PyTorch 入门最重要的一步：**知道一个模型从数据进入到训练完成，再到推理使用，中间到底发生了什么。**


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-pytorch学习|模块-pytorch学习]]

%% 项目关联导航：结束 %%
