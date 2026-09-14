# PyTorch 课程总复盘：从训练到推理的完整主线

> 适用对象：PyTorch 初学者，目标是打好后续学习 AI Infra 推理方向的基础。  
> 核心目标：把多节课中分散、重复的知识点合并成一条清晰主线：**数据 → 模型 → 训练 → 测试 → 保存加载 → GPU → 推理 → 阅读开源项目**。

---

## 一、总主题判断

这些课程整体属于 **PyTorch 入门后半段：从“会写单个 API”到“能跑通完整训练与推理流程”** 的阶段。前面的课程可能分别讲 Dataset、DataLoader、nn.Module、卷积层、损失函数、优化器等单点知识；而这一组课程的核心，是把这些单点串成一个真正可运行的深度学习项目。你学到的不只是 `torch.save()`、`loss.backward()`、`.to(device)` 这些 API，而是一个模型从数据进入、训练更新、测试评估、保存权重、加载推理，到最后阅读开源项目的完整闭环。

从学习顺序看，这些课大致经历了：先学损失函数和反向传播，理解“模型为什么能学”；再学优化器，理解“参数如何被更新”；然后用 CIFAR10 串起完整训练循环；接着加入测试、TensorBoard、模型保存；再学习 GPU/CUDA 训练；最后学习单张图片推理和开源项目阅读。整体主线其实只有一句话：**把图片数据变成 Tensor，送入模型得到 logits，用 loss 衡量错误，用 backward 计算梯度，用 optimizer 更新参数，训练完成后保存模型，推理时只加载模型并执行 forward。**

这些内容和 AI Infra 推理方向的关系非常直接。AI Infra 推理不是从零训练模型，而是围绕已经训练好的模型，解决如何高效加载权重、管理显存、组织 batch、执行 forward、优化延迟和吞吐的问题。因此你现在必须先分清训练和推理：训练有 loss、backward、optimizer；推理只有 forward、no_grad/inference_mode、后处理。PyTorch 入门课给你的是“模型计算的基本语法和流程”，后续 vLLM、TensorRT、ONNX Runtime、SGLang 等推理系统，都是在这个基础上做工程化和性能优化。

---

## 二、按学习顺序梳理课程主线

### 1. 数据准备：Dataset、DataLoader、transforms

**在完整流程中的位置：** 这是模型训练的入口。没有数据，模型无法训练；数据格式不对，模型也无法接收。

**解决什么问题：** 把图片文件或 CIFAR10 数据集，变成 PyTorch 模型可以处理的 Tensor，并按 batch 喂给模型。

**依赖哪些知识点：** 需要理解图片、Tensor、shape、batch 的概念。

**容易混淆：** Dataset 是“数据源”，DataLoader 是“按 batch 取数据的工具”；`ToTensor()` 不是简单读图，它会把 PIL 图片变成 `[C,H,W]` 的 float Tensor，并把像素缩放到 `[0,1]`。

---

### 2. 模型定义：nn.Module、Sequential、Conv2d、Linear

**在完整流程中的位置：** 数据准备好以后，需要一个模型来接收输入并输出预测结果。

**解决什么问题：** 定义神经网络的计算结构，也就是图片经过哪些层，最后输出多少个类别。

**依赖哪些知识点：** 需要理解输入 shape、卷积层、池化层、展平、全连接层。

**容易混淆：** `nn.Module` 是所有模型的基类；`forward()` 定义前向传播；`nn.Sequential` 适合顺序堆叠层，但复杂分支结构不能完全依赖它。

---

### 3. 前向传播：`outputs = model(images)`

**在完整流程中的位置：** 这是训练和推理都共有的核心步骤。

**解决什么问题：** 把输入图片经过模型计算，得到模型对各类别的预测分数，也就是 logits。

**依赖哪些知识点：** 模型结构、输入 shape、device 一致性。

**容易混淆：** `outputs` 不是类别名，也不是概率；对于 CIFAR10，它通常是 `[batch_size, 10]` 的 logits。

---

### 4. 损失函数：`CrossEntropyLoss`

**在完整流程中的位置：** 训练阶段前向传播之后，用 loss 衡量预测和真实标签之间的差距。

**解决什么问题：** 把模型输出和真实标签之间的差距变成一个标量，方便反向传播。

**依赖哪些知识点：** logits、target、分类任务、dtype。

**容易混淆：** `CrossEntropyLoss` 的输入应该是原始 logits，不要先手动 softmax；target 应该是类别编号 `[N]`，dtype 通常是 `torch.long`，不是 one-hot。

---

### 5. 反向传播：`loss.backward()`

**在完整流程中的位置：** loss 算出来以后，用 backward 计算每个参数对 loss 的梯度。

**解决什么问题：** 告诉模型每个参数应该往哪个方向调整，loss 才可能下降。

**依赖哪些知识点：** 计算图、梯度、Autograd、参数。

**容易混淆：** `loss.backward()` 只计算梯度，不更新参数；真正更新参数的是 `optimizer.step()`。

---

### 6. 优化器：`optimizer.step()`

**在完整流程中的位置：** backward 之后，根据参数梯度更新模型参数。

**解决什么问题：** 让模型真的“学习”，也就是改变 weight 和 bias。

**依赖哪些知识点：** 参数、梯度、学习率、`model.parameters()`。

**容易混淆：** 每轮训练前要 `optimizer.zero_grad()`，否则 PyTorch 默认梯度会累加；`optimizer` 不需要 `.cuda()`，它管理的是模型参数引用。

---

### 7. 完整训练循环

**在完整流程中的位置：** 把数据、模型、loss、backward、optimizer 串起来，重复训练多轮。

**解决什么问题：** 让模型反复看数据并不断更新参数。

**依赖哪些知识点：** Dataset/DataLoader、model、loss、optimizer、epoch、batch、step。

**容易混淆：** epoch 是完整数据集训练一轮；batch 是一次送入模型的一小批数据；step/iteration 是一个 batch 完成一次更新。

---

### 8. 测试/验证循环

**在完整流程中的位置：** 每轮训练后，用测试集评估模型效果。

**解决什么问题：** 判断模型是否真的学会，而不是只在训练集上 loss 下降。

**依赖哪些知识点：** `model.eval()`、`torch.no_grad()`、accuracy、argmax。

**容易混淆：** 测试阶段不应该 backward，也不应该 optimizer.step；测试阶段只做 forward 和指标统计。

---

### 9. TensorBoard 记录

**在完整流程中的位置：** 训练和测试过程中，把 loss、accuracy 写入日志并画成曲线。

**解决什么问题：** 命令行输出太乱，TensorBoard 能直观看趋势。

**依赖哪些知识点：** step、loss.item()、SummaryWriter。

**容易混淆：** `global_step` 要变化，否则曲线画不出来；训练 loss 和测试 loss 要用不同 tag。

---

### 10. 模型保存与加载

**在完整流程中的位置：** 训练完成后保存权重；推理或继续训练时重新加载。

**解决什么问题：** 避免每次都重新训练，把训练结果保存下来。

**依赖哪些知识点：** 模型结构、参数、`state_dict()`、`torch.save()`、`torch.load()`。

**容易混淆：** 保存整个模型和保存 `state_dict()` 是两种不同方式；官方和工程实践更推荐保存 `state_dict()`。

---

### 11. GPU/CUDA 训练

**在完整流程中的位置：** 在原训练流程基础上，把模型、数据、loss 放到 GPU 上加速。

**解决什么问题：** CPU 训练慢，GPU 更适合大量矩阵运算、卷积运算和反向传播。

**依赖哪些知识点：** device、Tensor、显存、`.cuda()`、`.to(device)`。

**容易混淆：** 不是只把 model 放到 GPU 就行，images、targets、loss_fn 也要在同一个 device；Tensor 的 `.to(device)` 需要重新赋值。

---

### 12. 单张图片推理

**在完整流程中的位置：** 模型训练保存后，加载模型，对外部真实图片进行预测。

**解决什么问题：** 把训练好的模型真正拿来用。

**依赖哪些知识点：** PIL 读图、Resize、ToTensor、unsqueeze、model.eval、no_grad、argmax、类别映射。

**容易混淆：** 单张图片经过 ToTensor 后是 `[3,32,32]`，模型需要 `[1,3,32,32]`，所以要增加 batch 维度。

---

### 13. 开源项目阅读

**在完整流程中的位置：** 学完基础后，从课程代码过渡到真实项目代码。

**解决什么问题：** 看懂 GitHub 项目中的 README、train.py、test.py、options/config、argparse 参数。

**依赖哪些知识点：** 完整训练流程、命令行运行、参数解析、项目结构。

**容易混淆：** 开源项目看起来复杂，但底层仍然是参数配置 → 数据 → 模型 → loss → optimizer → 训练循环 → 测试 → 保存。

---

## 三、核心知识点提炼表

| 知识点 | 所属阶段 | 解决什么问题 | 关键 API / 代码 | 必须掌握程度 | 常见坑 | 和 AI Infra 推理关系 |
|---|---|---|---|---|---|---|
| Dataset | 数据准备 | 表示一个数据集 | `torchvision.datasets.CIFAR10` | 必须会用 | 和 DataLoader 混淆 | 推理请求输入也需要数据组织 |
| transforms | 数据预处理 | 把图片处理成模型输入格式 | `Resize`、`ToTensor`、`Compose` | 必须会用 | 训练/推理预处理不一致 | 部署中预处理错误会导致输出错误 |
| DataLoader | 批量加载 | 按 batch 喂数据 | `DataLoader(dataset, batch_size=64)` | 必须会用 | batch_size 太大显存爆 | batch 是推理吞吐优化基础 |
| batch | shape 主线 | 一次处理多个样本 | `[N,C,H,W]` | 必须理解 | 忘记单张图也要 batch 维 | vLLM continuous batching 的前置概念 |
| nn.Module | 模型定义 | 定义神经网络结构 | `class Net(nn.Module)` | 必须掌握 | 忘记 `super().__init__()` | 推理引擎执行的就是模型 forward |
| forward | 前向传播 | 定义输入如何变成输出 | `def forward(self,x)` | 必须掌握 | 拼错 forward | 推理主要优化 forward |
| Sequential | 模型定义 | 顺序堆叠层 | `nn.Sequential(...)` | 会用即可 | 不适合复杂分支 | 对应推理图中的算子链 |
| Conv2d | 特征提取 | 从图片中提取局部特征 | `nn.Conv2d(3,32,5,padding=2)` | 理解输入输出通道 | channel 写错 | TensorRT/ONNX 会优化 Conv 算子 |
| MaxPool2d | 降采样 | 减小空间尺寸 | `nn.MaxPool2d(2)` | 理解尺寸减半 | pool 次数太多 | 推理图中的常见算子 |
| Flatten | 接全连接层 | 把多维特征拉平 | `nn.Flatten()` | 必须理解 | Linear 输入维度算错 | shape 变换是部署常见问题 |
| Linear | 分类头 | 输出类别 logits | `nn.Linear(64*4*4,10)` | 必须掌握 | `in_features` 写错 | LLM 中大量矩阵乘法也是 Linear 类思想 |
| logits | 前向输出 | 表示每类分数 | `outputs = model(images)` | 必须理解 | 误以为是概率 | 推理后处理从 logits 开始 |
| CrossEntropyLoss | 训练损失 | 分类任务计算误差 | `nn.CrossEntropyLoss()` | 必须掌握 | 手动 softmax、target dtype 错 | LLM 训练也用交叉熵思想 |
| Autograd | 反向传播 | 自动求梯度 | `loss.backward()` | 必须理解 | 以为 backward 更新参数 | 推理不需要 Autograd |
| optimizer | 参数更新 | 根据梯度改参数 | `SGD(model.parameters(), lr)` | 必须掌握 | 忘记 `step()` | 推理阶段没有 optimizer |
| zero_grad | 梯度清零 | 防止梯度累加 | `optimizer.zero_grad()` | 必须掌握 | 忘记导致梯度累加 | 训练专属，推理不用 |
| epoch | 训练循环 | 数据集训练一轮 | `for epoch in range(...)` | 必须理解 | 和 step 混淆 | 推理没有 epoch，但有 request/token step |
| model.train | 训练模式 | 让 Dropout/BatchNorm 用训练行为 | `model.train()` | 规范代码必须写 | 以为它自动训练 | 推理不能用训练模式 |
| model.eval | 推理模式 | 让模型进入测试/推理行为 | `model.eval()` | 推理前必须写 | 以为它关闭梯度 | 部署前必须确保 eval |
| no_grad | 关闭梯度 | 测试/推理不构建计算图 | `with torch.no_grad()` | 推理必备 | 训练时误用 | 减少推理显存和开销 |
| TensorBoard | 记录曲线 | 可视化 loss/accuracy | `SummaryWriter.add_scalar` | 会用即可 | step 不变 | benchmark 监控思想相通 |
| state_dict | 保存权重 | 保存模型参数 | `model.state_dict()` | 必须掌握 | 和完整模型保存混淆 | 大模型权重加载的基础思想 |
| torch.save/load | 保存加载 | 模型持久化 | `torch.save`、`torch.load` | 必须掌握 | 保存/加载方式不对应 | 部署服务启动要加载权重 |
| device | 设备管理 | CPU/GPU 统一管理 | `torch.device(...)` | 必须掌握 | 模型数据 device 不一致 | AI Infra 的显存和硬件基础 |
| `.to(device)` | 迁移设备 | 把模型/Tensor 放到指定设备 | `x = x.to(device)` | 必须掌握 | Tensor 没重新赋值 | 多 GPU/推理部署基础 |
| map_location | 跨设备加载 | GPU 模型在 CPU 加载 | `torch.load(path,map_location='cpu')` | 必须知道 | CUDA 报错不会处理 | 跨设备部署常见 |
| argmax | 后处理 | logits 转类别编号 | `outputs.argmax(1)` | 必须掌握 | dim 写错 | LLM token 选择有类似思想 |
| argparse | 开源项目 | 命令行传参数 | `parser.add_argument` | 必须会看 | required 导致右键运行报错 | vLLM/TensorRT 项目大量用参数 |

---

## 四、完整 PyTorch 训练流程主线

```text
图片文件 / CIFAR10 数据
    ↓
transforms.ToTensor / Resize
    ↓
Dataset
    ↓
DataLoader
    ↓
images, targets
    ↓
model(images)
    ↓
outputs / logits
    ↓
loss_fn(outputs, targets)
    ↓
optimizer.zero_grad()
    ↓
loss.backward()
    ↓
optimizer.step()
    ↓
测试集验证
    ↓
TensorBoard 记录
    ↓
保存模型
    ↓
加载模型
    ↓
单张图片推理
```

逐步解释：

1. **图片文件 / CIFAR10 数据**：原始数据来源，可以是标准数据集，也可以是自己下载的图片。
2. **transforms.ToTensor / Resize**：把图片调整到模型需要的尺寸，并转成 Tensor。
3. **Dataset**：定义“数据从哪里来，以及每个样本是什么”。
4. **DataLoader**：把 Dataset 按 batch 组织起来，每次返回一批 `images, targets`。
5. **images, targets**：`images` 是输入图片 Tensor，`targets` 是真实类别编号。
6. **model(images)**：执行模型前向传播，输出每个类别的分数。
7. **outputs / logits**：模型原始输出，CIFAR10 中 shape 通常是 `[N,10]`。
8. **loss_fn(outputs, targets)**：把预测分数和真实标签比较，得到 loss。
9. **optimizer.zero_grad()**：清空上一轮梯度，防止梯度累加。
10. **loss.backward()**：反向传播，计算每个参数的梯度。
11. **optimizer.step()**：根据梯度更新模型参数。
12. **测试集验证**：训练一轮后，在测试集上只做 forward，计算 loss 和 accuracy。
13. **TensorBoard 记录**：记录 train_loss、test_loss、accuracy，观察训练趋势。
14. **保存模型**：把训练好的模型参数保存成 `.pth` 文件。
15. **加载模型**：推理或继续训练时恢复模型参数。
16. **单张图片推理**：读取外部图片，预处理后输入模型，用 argmax 得到预测类别。

---

## 五、训练和推理的区别

### 训练阶段

训练阶段的目的是**更新模型参数**。它包括：

```python
model.train()
outputs = model(images)
loss = loss_fn(outputs, targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

训练阶段需要计算图，因为 backward 要沿着计算图计算梯度。训练阶段会占用更多显存，因为需要保存中间激活、梯度、优化器状态。

### 测试 / 推理阶段

测试或推理阶段的目的是**使用固定模型参数得到输出**。它包括：

```python
model.eval()
with torch.no_grad():
    outputs = model(images)
    preds = outputs.argmax(1)
```

推理阶段不需要梯度，不需要 loss.backward，不需要 optimizer.step。模型权重固定，只执行 forward，然后做后处理。

| 对比项 | 训练 | 测试/推理 |
|---|---|---|
| 模型模式 | `model.train()` | `model.eval()` |
| 是否 forward | 是 | 是 |
| 是否计算 loss | 是 | 测试可算，实际推理通常不算 |
| 是否 backward | 是 | 否 |
| 是否 optimizer.step | 是 | 否 |
| 是否更新参数 | 是 | 否 |
| 是否需要梯度 | 需要 | 不需要 |
| 是否构建计算图 | 需要 | 不需要 |
| 是否使用 no_grad | 不能包住训练 forward | 推荐使用 |
| 显存占用 | 高 | 低 |
| 主要显存内容 | 权重、激活、梯度、优化器状态 | 权重、输入 batch、激活、KV Cache |
| 输出后处理 | 通常用于 loss | argmax/topk/sampling |
| AI Infra 关注点 | 训练吞吐、收敛 | forward 延迟、吞吐、显存、batching |

AI Infra 推理方向主要优化 forward，而不是 backward，因为线上服务面对的是“用户输入 → 模型输出”的过程，模型参数已经训练完成。推理系统要做的是让 forward 更快、更省显存、更能处理高并发请求。比如 TensorRT 优化算子融合和 kernel 执行，vLLM 优化 batching 和 KV Cache，ONNX Runtime 优化跨平台推理图执行，它们的核心都不是训练更新参数，而是高效执行模型前向图。

---

## 六、最终推荐版完整代码

### 1. `model.py`

```python
import torch
from torch import nn


class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = Tudui()
    x = torch.ones((64, 3, 32, 32))
    y = model(x)
    print(y.shape)  # torch.Size([64, 10])
```

---

### 2. `train.py`

```python
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model import Tudui


# 1. 选择设备：有 GPU 用 GPU，否则用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备：{device}")


# 2. 准备数据集
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


# 3. 创建模型、损失函数、优化器
model = Tudui().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)


# 4. 训练参数
num_epochs = 10
total_train_step = 0
writer = SummaryWriter("logs")


# 5. 训练 + 测试
for epoch in range(num_epochs):
    print(f"--------第 {epoch + 1} 轮训练开始--------")

    # 训练模式
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
            print(f"训练次数：{total_train_step}, Loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试模式
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
    test_accuracy = total_correct / len(test_data)

    print(f"测试集平均 Loss: {avg_test_loss:.4f}")
    print(f"测试集正确率: {test_accuracy:.4f}")

    writer.add_scalar("test_loss", avg_test_loss, epoch + 1)
    writer.add_scalar("test_accuracy", test_accuracy, epoch + 1)

    # 推荐保存 state_dict
    torch.save(model.state_dict(), f"tudui_epoch_{epoch + 1}.pth")
    print("模型已保存")

writer.close()
```

---

### 3. `predict.py`

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


# 1. 读取图片
image_path = "./images/dog.png"
image = Image.open(image_path).convert("RGB")


# 2. 预处理：要和训练时模型输入要求一致
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

image = transform(image)       # [3, 32, 32]
image = image.unsqueeze(0)     # [1, 3, 32, 32]
image = image.to(device)


# 3. 加载模型
model = Tudui().to(device)
state_dict = torch.load("tudui_epoch_10.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()


# 4. 推理
with torch.no_grad():
    output = model(image)          # [1, 10]
    pred_idx = output.argmax(1).item()
    pred_name = classes[pred_idx]

print(f"预测类别编号：{pred_idx}")
print(f"预测类别名称：{pred_name}")
```

---

## 七、重点代码解释

### `transforms.Compose(...)`

它是把多个图像预处理操作串起来的工具。训练 CIFAR10 时，最基本是 `ToTensor()`；单张外部图片推理时，还要加 `Resize((32,32))`。输入是 PIL Image，输出通常是 Tensor。常见错误是训练和推理预处理不一致，比如训练时只用 CIFAR10 的 32×32 图，推理时却直接把 400×300 图片送入模型。

---

### `DataLoader(..., batch_size=64, shuffle=True)`

DataLoader 负责按 batch 取数据。`batch_size=64` 表示每次取 64 张图片，输出 `images.shape=[64,3,32,32]`，`targets.shape=[64]`。训练集通常 `shuffle=True`，测试集通常 `shuffle=False`。常见错误是把 Dataset 当成 DataLoader，或者 batch_size 太大导致显存爆。

---

### `class Tudui(nn.Module)`

这是自定义模型类，继承 `nn.Module` 后，PyTorch 才能管理里面的参数。`__init__` 中定义层，`forward` 中定义数据如何流过这些层。常见错误是忘记 `super().__init__()` 或忘记写 `forward()`。

---

### `forward()`

`forward()` 定义前向传播过程。调用 `model(images)` 时，PyTorch 实际会执行 `forward(images)`。输入 shape 是 `[N,3,32,32]`，输出 shape 是 `[N,10]`。常见错误是以为需要手动调用 `model.forward(images)`，通常不推荐这样写。

---

### `nn.Sequential`

`nn.Sequential` 是顺序容器，适合把 Conv、Pool、Flatten、Linear 按顺序串起来。它内部相当于自动执行 `x=layer1(x); x=layer2(x); ...`。常见错误是复杂网络结构也强行用 Sequential；如果有残差连接、多输入输出，就需要手写 forward。

---

### `Conv2d / MaxPool2d / Flatten / Linear`

`Conv2d` 提取图像局部特征；`MaxPool2d` 降低空间尺寸；`Flatten` 把 `[N,64,4,4]` 拉平成 `[N,1024]`；`Linear` 做分类。CIFAR10 经过三次 2×2 池化后，空间尺寸从 32→16→8→4，所以 Linear 的输入是 `64*4*4`。常见错误是 Linear 的 `in_features` 算错。

---

### `outputs = model(images)`

这是前向传播。输入 `images` 是 `[64,3,32,32]`，输出 `outputs` 是 `[64,10]`。这 10 个数是每张图对 10 个类别的预测分数。常见错误是把 outputs 当成概率或类别名。

---

### `loss = loss_fn(outputs, targets)`

这是计算分类损失。`outputs` shape 是 `[N,C]`，`targets` shape 是 `[N]`。`CrossEntropyLoss` 内部包含 LogSoftmax，所以不要手动 softmax。常见错误是 target 用 float 或 one-hot。

---

### `optimizer.zero_grad()`

PyTorch 默认梯度累加，所以每次反向传播前要清空上一轮梯度。常见错误是忘记写，导致梯度越累越大，训练异常。

---

### `loss.backward()`

反向传播，沿计算图计算每个参数的梯度，结果存在参数的 `.grad` 中。它只算梯度，不更新参数。常见错误是以为 backward 后模型已经学习了。

---

### `optimizer.step()`

根据参数梯度更新模型参数。训练真正改变模型权重的是这一步。常见错误是只 backward，不 step。

---

### `model.train()`

切换到训练模式，主要影响 Dropout、BatchNorm。它不是“开始训练按钮”，真正训练还要 forward、loss、backward、step。常见错误是误以为不写 train 就不能训练。

---

### `model.eval()`

切换到测试/推理模式，主要影响 Dropout、BatchNorm。推理前必须养成写它的习惯。常见错误是以为 eval 会关闭梯度；实际上关闭梯度要用 no_grad。

---

### `with torch.no_grad()`

关闭梯度计算，不构建反向传播计算图。测试和推理时应该使用它来节省显存和计算。常见错误是训练时误用 no_grad，导致 backward 失败。

---

### `images = images.to(device)`

把输入数据移动到 CPU/GPU 设备。模型和输入必须在同一个 device。Tensor 的 `.to(device)` 要重新赋值。常见错误是只写 `images.to(device)`，后续 images 仍可能在原设备。

---

### `torch.save(model.state_dict(), ...)`

保存模型参数字典。推荐这种方式，因为它更轻量、更适合工程。常见错误是保存 state_dict 后，加载时却直接 `model = torch.load(path)`。

---

### `model.load_state_dict(...)`

把参数字典加载到模型结构里。必须先创建同样结构的模型，再加载参数。常见错误是模型结构和权重 shape 不匹配。

---

### `image.unsqueeze(0)`

给单张图片增加 batch 维度。`[3,32,32]` 变成 `[1,3,32,32]`。常见错误是忘记 batch 维度，导致模型报输入维度错误。

---

### `output.argmax(1)`

从 `[N,10]` 的 logits 中，按类别维度取最大值所在位置，得到预测类别编号。常见错误是写成 `argmax(0)`，那是按 batch 维度比较。

---

## 八、shape 主线

以 CIFAR10 为例：

| 场景 | shape | 含义 |
|---|---|---|
| 单张图片 | `[3, 32, 32]` | 3 通道，32×32 |
| 一个 batch | `[64, 3, 32, 32]` | 64 张图片 |
| 模型输出 | `[64, 10]` | 每张图片 10 类 logits |
| target | `[64]` | 每张图片一个类别编号 |
| 单张图片推理前 | `[3, 32, 32]` | 没有 batch 维度 |
| unsqueeze 后 | `[1, 3, 32, 32]` | batch_size=1 |
| 单张图片输出 | `[1, 10]` | 一张图片的 10 类 logits |
| argmax 后 | `[1]` | 一个预测类别编号 |

模型需要 batch 维度，是因为 PyTorch 的 CNN 默认输入是 `[N,C,H,W]`。即使只有一张图片，也要写成 `[1,C,H,W]`，表示一个 batch 里有一张图片。

`CrossEntropyLoss` 要求 output 是 `[N,C]`，target 是 `[N]`。其中 `N` 是 batch size，`C` 是类别数。对于 CIFAR10，`C=10`。target 不需要 one-hot，只需要类别编号，如 `0,1,2,...,9`。

`argmax(dim=1)` 是按类别维度取最大值。因为输出 `[N,10]` 中第 0 维是 batch，第 1 维是类别，所以要沿 dim=1 找每张图片预测分数最高的类别。

---

## 九、device / GPU 主线

CPU 是通用处理器，适合复杂控制逻辑；GPU 有大量并行计算单元，适合矩阵乘法、卷积、反向传播这类大规模并行计算。PyTorch 中每个 Tensor 都有 device，例如 `cpu`、`cuda:0`。模型参数本质也是 Tensor，所以模型也有 device。

模型和输入必须在同一个 device。比如模型在 `cuda:0`，images 在 CPU，就会报 device mismatch。正确写法是：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
images = images.to(device)
targets = targets.to(device)
```

`.cuda()` 是直接移动到 CUDA GPU，写法简单但不够通用；`.to(device)` 更推荐，因为它可以根据 device 变量选择 CPU、CUDA、不同 GPU，甚至其他后端。`torch.cuda.is_available()` 用来判断当前环境是否能使用 CUDA。`map_location` 用来解决 GPU 保存的模型在 CPU 环境加载的问题，例如：

```python
state_dict = torch.load("model.pth", map_location="cpu")
```

GPU 训练和 CPU 训练的核心逻辑不变，只是把模型、数据、loss_fn 移动到 GPU。GPU 快，是因为卷积、矩阵乘法、backward 都能并行执行。

和 AI Infra 推理相关的关键词：

- **显存**：GPU 上的内存，模型权重、输入 batch、activation、KV Cache 都占显存。
- **权重**：训练好的参数，推理时必须加载到设备上。
- **activation**：模型中间结果，训练时要保存用于 backward，推理时只临时使用。
- **batch**：训练中提高稳定性和效率，推理中提高吞吐。
- **KV Cache**：LLM 推理中缓存历史 token 的 key/value，减少重复计算，但占大量显存。
- **CPU-GPU 数据搬运**：`.to(device)` 背后可能发生内存拷贝，频繁拷贝会增加 latency。
- **latency**：单次请求延迟。
- **throughput**：单位时间处理多少请求或 token。

---

## 十、常见错误和排查方法

| 错误现象 | 常见原因 | 如何排查 | 正确写法 |
|---|---|---|---|
| shape 不匹配 | Linear 输入维度写错 | 打印每层输出 shape | 根据 Flatten 后维度设置 `Linear(in_features, ...)` |
| 忘记 batch 维度 | 单张图是 `[C,H,W]` | 打印 `image.shape` | `image = image.unsqueeze(0)` |
| target dtype 错误 | CrossEntropyLoss 需要 long | 打印 `targets.dtype` | `targets = targets.long()` |
| device 不一致 | model 在 GPU，data 在 CPU | 打印 `images.device` 和参数 device | `images = images.to(device)` |
| GPU 保存模型 CPU 加载报错 | checkpoint 记录 cuda | 看报错中的 map_location 提示 | `torch.load(path, map_location='cpu')` |
| 忘记 model.eval() | Dropout/BatchNorm 仍是训练行为 | 检查推理代码 | 推理前 `model.eval()` |
| 忘记 torch.no_grad() | 推理仍构建计算图 | 看显存占用 | `with torch.no_grad():` |
| 忘记 zero_grad() | 梯度累加 | loss 异常震荡 | 每步前 `optimizer.zero_grad()` |
| backward 后忘记 step | 参数没更新 | 比较参数是否变化 | `loss.backward(); optimizer.step()` |
| 保存/加载方式不匹配 | state_dict 和完整模型混用 | 看 `torch.load` 返回类型 | state_dict 用 `load_state_dict` |
| CrossEntropyLoss 前手动 softmax | 重复 softmax | 检查 loss 输入 | 直接传 logits |
| DataLoader 和 Dataset 混淆 | Dataset 不能直接按 batch | 打印类型 | `DataLoader(dataset, batch_size=...)` |
| Tensor `.to(device)` 后没赋值 | Tensor 仍在原设备 | 打印 device | `x = x.to(device)` |
| argmax 维度写错 | `dim=0` 按 batch 比 | 打印 output shape | 分类 `[N,C]` 用 `argmax(1)` |
| 路径错误 | 相对路径基准不清 | `Path(path).exists()` | 用正确相对路径或绝对路径 |
| argparse required 导致无法右键运行 | 命令行必填参数没传 | 搜索 `required=True` | 命令行传参或设置 `default` |

---

## 十一、和 AI Infra 推理方向的关系

### 1. Dataset/DataLoader 和推理请求输入

训练中 DataLoader 把样本组织成 batch；推理服务中，请求输入也要被解码、预处理、组织成 batch。区别是训练数据来自数据集，推理输入来自用户请求。

### 2. batch 和 vLLM continuous batching

普通 CNN 中 batch 是 `[N,3,32,32]`。LLM 中 batch 是多个请求的 token 序列。vLLM 的 continuous batching 会动态把不同用户请求合并执行，提高 GPU 利用率。

### 3. model.forward 和推理引擎优化

PyTorch 中 `outputs = model(inputs)` 就是 forward。推理引擎做的事情，就是让这个 forward 更快，例如算子融合、kernel 优化、图优化、减少内存拷贝。

### 4. no_grad / inference_mode 和推理显存优化

推理不需要梯度，所以不需要计算图和 backward 中间状态。`no_grad` 和 `inference_mode` 能减少显存和运行开销。推理系统本质上只保留 forward 需要的内容。

### 5. device / CUDA 和 GPU 推理

AI Infra 推理离不开 GPU。你要知道模型权重在哪张 GPU、输入在哪张 GPU、显存是否够、是否有不必要的 CPU-GPU 拷贝。

### 6. state_dict / 模型加载 和大模型权重加载

PyTorch 中 `state_dict` 保存参数。大模型中也有类似思想，只是文件更大，可能被分片保存成 `safetensors`。vLLM 加载模型时，本质也是读取权重并放到 GPU 显存。

### 7. argmax / 后处理 和 LLM token sampling

图像分类中 logits 经过 argmax 得到类别。LLM 中 logits 经过 greedy、top-k、top-p、temperature sampling 得到下一个 token。二者都是后处理，只是 LLM 更复杂。

### 8. TensorBoard / 日志 和 benchmark 监控

训练中记录 loss、accuracy；推理中记录 latency、throughput、QPS、tokens/s、显存占用、KV Cache 使用率。核心都是把系统状态指标化、可视化。

### 9. 开源项目阅读能力和 vLLM、TensorRT、SGLang

AI Infra 学习高度依赖读开源项目。你现在学 README、train.py、test.py、argparse，是为了以后能读懂 vLLM 的 scheduler、KV Cache、worker、engine，TensorRT 的构建脚本和 benchmark，SGLang 的 serving 和调度逻辑。

---

## 十二、7 天复习计划

### Day 1：数据与 shape

复习 Dataset、DataLoader、transforms、batch。重点手写打印：`images.shape`、`targets.shape`。

### Day 2：模型结构

复习 `nn.Module`、`forward`、`Sequential`、Conv2d、MaxPool2d、Flatten、Linear。重点手算 CNN 的 shape 变化。

### Day 3：训练核心三件套

复习 `CrossEntropyLoss`、`loss.backward()`、`optimizer.step()`。背熟：`zero_grad → forward → loss → backward → step`。

### Day 4：完整训练与测试

手写 train loop 和 test loop。重点区分 `model.train()`、`model.eval()`、`torch.no_grad()`。

### Day 5：保存加载与推理

复习 `state_dict`、`torch.save`、`torch.load`、`map_location`。写一个 `predict.py` 对单张图片推理。

### Day 6：GPU/device

复习 `.cuda()`、`.to(device)`、`torch.cuda.is_available()`。重点排查 device mismatch。

### Day 7：开源项目阅读 + AI Infra 连接

找一个小型 PyTorch 项目，按 README → train.py → argparse → data → model → test.py 的顺序读。最后把 PyTorch 的 batch、forward、no_grad、device、state_dict 对应到 vLLM/TensorRT 的推理概念。

---

## 最后总结

这几节课不是孤立知识，而是在搭一条完整链路：

```text
数据进入 PyTorch
    ↓
模型完成 forward
    ↓
loss 衡量错误
    ↓
backward 计算梯度
    ↓
optimizer 更新参数
    ↓
测试集评估效果
    ↓
保存模型权重
    ↓
加载模型推理
    ↓
阅读开源项目进入工程实践
```

你现在最该形成的判断是：

- 训练代码的核心是：forward + loss + backward + optimizer。
- 推理代码的核心是：load model + eval + no_grad/inference_mode + forward + 后处理。
- AI Infra 推理的核心是：让这个 forward 在真实硬件和真实请求场景下跑得更快、更稳、更省显存。



%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-pytorch学习|模块-pytorch学习]]

%% 项目关联导航：结束 %%
