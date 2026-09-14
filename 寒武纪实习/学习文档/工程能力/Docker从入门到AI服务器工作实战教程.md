# Docker 从入门到 AI 服务器工作实战教程
## 面向公司服务器、SGLang/vLLM、GPU/MLU 模型测试场景

> 这份文档按你现在每天真实的工作流程来组织：**SSH 登录服务器 → 查看 Docker 环境 → 找镜像 → 起容器 → 挂载代码/模型/结果目录 → 映射设备 → 进入容器 → 安装框架与依赖 → 启动模型服务 → 跑精度/性能测试 → 保存环境 → 换服务器迁移 → 排错**。
>
> 目标不是让你背 Docker 命令，而是让你真正理解：**宿主机、镜像、容器、挂载目录、设备、网络、端口、Python 环境到底分别属于哪一层。**

---

# 1. 先用一句话理解 Docker

Docker 可以理解为：

> **把程序运行所需要的软件环境装进一个相对独立的“容器”中运行。**

假设一台公司服务器上有很多人：

```text
张三：Python 3.10 + Torch A + SGLang A
李四：Python 3.11 + Torch B + vLLM B
你：Python 3.10 + torch_mlu + SGLang-MLU + WeLM
```

如果所有人都直接往宿主机安装：

```text
服务器系统
├── Python 被谁升级了？
├── torch 到底该用哪个版本？
├── transformers 谁又降级了？
├── 自定义算子是谁编译的？
└── 今天能跑，明天可能就坏
```

Docker 把它变成：

```text
物理服务器 Host
│
├── 容器 A：WeLM + SGLang + torch_mlu
├── 容器 B：GLM + vLLM + torch_mlu
└── 容器 C：另一个项目环境
```

不同项目互相隔离，所以 AI Infra 工作里 Docker 非常常见。

---

# 2. 你必须先分清四个核心概念

Docker 最重要的是：

```text
宿主机 Host
镜像 Image
容器 Container
挂载 Mount / Volume
```

## 2.1 宿主机 Host

你先执行：

```bash
ssh mahao@10.x.x.x
```

登录进去的那台真正服务器，就是 **宿主机**。

宿主机上可能有：

```text
/projs
/data
/data1
/data2
/tmp
Docker daemon
16 张 GPU/MLU
几百 GB 内存
```

你执行 `docker run` 之前，本质都还在宿主机环境。

## 2.2 镜像 Image

镜像是一个已经准备好的 **环境模板**。

例如一个公司镜像名字可能包含：

```text
ubuntu22.04
py310
torch2.x
torchmlu1.x
```

可以把镜像理解为：

```text
Ubuntu
+ Python
+ Torch
+ torch_mlu
+ 系统库
+ 一些预装工具
```

它只是模板，本身不是一个正在运行的环境。

## 2.3 容器 Container

容器是：

> **镜像真正运行起来以后生成的实例。**

类比：

```text
类        → 对象
镜像 Image → 容器 Container
```

一个镜像可以创建很多容器：

```text
welm_env:v1
├── mahao_welm
├── test_welm_1
└── zhangsan_welm
```

## 2.4 挂载 Mount

挂载就是把宿主机某个真实目录“映射”到容器里。

例如：

```bash
-v /projs:/projs
```

意思：

```text
宿主机 /projs
      ↓
容器   /projs
```

容器里访问 `/projs/xxx`，实际上是在操作宿主机 `/projs/xxx`。

---

# 3. 一张图理解你现在的工作环境

```text
┌─────────────────────────────────┐
│       公司物理服务器 Host        │
│                                 │
│ CPU / RAM / GPU / MLU           │
│                                 │
│ /projs   ← 代码、结果、共享文件   │
│ /data    ← 模型、数据            │
│ /tmp                            │
│                                 │
│ Docker Daemon                   │
│      │                          │
│      ├─────────────┐            │
│      │             │            │
│ Container A   Container B       │
│ WeLM          GLM                │
│ SGLang        vLLM               │
│ Python A      Python B           │
│      │             │            │
│      └── mount ────┘            │
│          /projs /data           │
└─────────────────────────────────┘
```

所以：

> **容器提供运行环境，宿主机提供真实硬件，挂载目录负责把代码、模型和测试结果带进容器。**

---

# 4. Docker 和虚拟机不是一回事

虚拟机通常是：

```text
物理服务器
↓
虚拟硬件
↓
完整 Guest OS
↓
应用程序
```

Docker 更像：

```text
物理服务器
↓
Linux Kernel
↓
Docker
↓
隔离的用户空间
↓
应用程序
```

所以 Docker：

- 启动快；
- 更轻；
- 更适合频繁创建开发/测试环境；
- 容器依然共享宿主机 Linux Kernel。

因此容器看起来像一台小服务器，但它不是完整虚拟机。

---

# 5. 容器里面的文件到底存在哪里

镜像通常是只读模板，容器启动后 Docker 再增加一个 **容器可写层**：

```text
镜像只读层
+
容器可写层
=
你看到的容器文件系统
```

例如你进入容器执行：

```bash
pip install evalscope
```

如果 Python 位于普通容器内部，那么这些新安装的文件通常写入：

```text
容器可写层
```

这直接解释了一个你工作中很常见的问题。

---

# 6. 为什么删容器以后 pip 安装的包没了

过程是：

```text
pip install evalscope
↓
写入容器可写层
↓
docker rm 容器
↓
容器可写层被删除
↓
evalscope 也没了
```

所以：

> **删除容器不是“退出环境”，而是真的把这个容器自己的可写文件系统删除。**

如果只是：

```bash
docker stop my_container
```

容器还存在，之后：

```bash
docker start my_container
```

原来安装的东西还在。

---

# 7. 为什么 `/projs` 里的代码删容器以后还在

因为你很可能使用了：

```bash
-v /projs:/projs
```

于是 `/projs` 的真实数据在宿主机：

```text
宿主机 /projs
```

容器只是把它显示成：

```text
容器 /projs
```

所以：

```text
docker rm 容器
```

只删除容器层，不删除宿主机真实 `/projs`。

这就是为什么你之前发现：

```text
删容器后 pip 包没了
但是 /projs 里的代码和测试结果还在
```

两者完全符合 Docker 的设计。

---

# 8. `-v` 挂载一定要看懂左右两边

格式：

```bash
-v 宿主机路径:容器路径
```

例如：

```bash
-v /workspace:/workspace/volume
```

表示：

```text
宿主机 /workspace/test.py
         ↓
容器 /workspace/volume/test.py
```

左边永远是宿主机，右边是容器内路径。

## 一个很容易踩的坑

如果镜像里原本：

```text
/workspace
├── a.py
└── b.py
```

你又：

```bash
-v /host/empty:/workspace
```

那么容器启动后 `/workspace` 会显示宿主机 `/host/empty` 的内容。

镜像原来 `/workspace` 中的文件并不是一定被删除了，而是被 mount 视图“盖住”了。

---

# 9. 代码、模型、结果和环境应该分别放哪里

这是 AI Infra 工作里最重要的工程习惯之一。

## 环境

例如：

```text
Ubuntu
Python
Torch
torch_mlu
SGLang/vLLM 的安装依赖
系统库
```

适合由：

```text
Docker Image / Container
```

管理。

## 代码

例如：

```text
sglang-mlu
vllm_mlu
你自己修改的 Python 文件
```

最好：

```text
Git + 挂载目录
```

## 模型和数据

几十 GB、几百 GB 的模型不要塞进个人容器层。

更适合：

```text
/data/models
/data1/models
共享模型盘
```

再通过 `-v` 挂进去。

## 测试结果

例如：

```text
GSM8K 结果
MMLU-Pro 结果
benchmark JSON/CSV
CNPerf 文件
server log
```

建议保存：

```text
/projs/.../results
共享存储目录
```

而不是只放容器内部 `/root/results`。

一句话：

> **Docker 镜像负责环境，Git 负责代码，共享存储负责模型和结果。**

---

# 10. Docker 生命周期

最基本生命周期：

```text
Image
  │ docker run
  ↓
Container running
  │ docker stop
  ↓
Container stopped
  │ docker start
  ↓
Container running
  │ docker rm
  ↓
Container deleted
```

这几个命令一定要分清。

---

# 11. `docker run`：创建一个新容器

作用：

> **根据镜像创建一个新容器，并启动。**

例如：

```bash
docker run -it \
  --name mahao_test \
  ubuntu:22.04 \
  /bin/bash
```

可以拆成：

```text
docker run       创建并启动
-it              交互式终端
--name           容器名字
ubuntu:22.04     使用哪个镜像
/bin/bash        容器启动后运行 Bash
```

注意：

> `docker run` 通常意味着“新建”，不是“进入原来的容器”。

---

# 12. `docker start`：重新启动已有容器

```bash
docker start mahao_welm
```

只是把已经存在、但停止的容器重新启动。

不会重新创建环境。

所以：

```text
run   = 创建 + 启动
start = 启动已有容器
```

---

# 13. `docker exec`：进入正在运行的容器

最常用：

```bash
docker exec -it mahao_welm /bin/bash
```

含义：

```text
在 mahao_welm 这个正在运行的容器里
↓
再启动一个 /bin/bash
↓
把当前终端接进去
```

日常所谓“进入容器”，大部分就是这条命令。

注意：

```bash
docker exec
```

只能用于 **正在运行** 的容器。

如果容器已经 stopped：

```bash
docker start mahao_welm
docker exec -it mahao_welm /bin/bash
```

---

# 14. `docker ps` 和 `docker ps -a`

查看正在运行：

```bash
docker ps
```

查看所有容器，包括 stopped/exited：

```bash
docker ps -a
```

如果：

```bash
docker ps
```

没有看到自己的容器，但：

```bash
docker ps -a
```

能看到：

```text
Exited (...)
```

说明：

> 容器没有丢，只是停止了。

---

# 15. `docker images`

查看当前 Docker daemon 本地有哪些镜像：

```bash
docker images
```

常见列：

```text
REPOSITORY
TAG
IMAGE ID
CREATED
SIZE
```

例如：

```text
glm5_vllm_env    20260907    abc123...    35GB
```

一个镜像通常用：

```text
镜像名:TAG
```

表示版本：

```text
glm5_vllm_env:20260907
```

---

# 16. `docker stop` / `docker rm`

停止容器：

```bash
docker stop mahao_welm
```

删除停止的容器：

```bash
docker rm mahao_welm
```

强制停止并删除：

```bash
docker rm -f mahao_welm
```

工作服务器上不要随便 `rm -f`。

先确认：

```bash
docker ps -a
docker inspect mahao_welm
```

尤其确认是否有重要文件仍只存在于容器内部。

---

# 17. `docker rmi`

删除镜像：

```bash
docker rmi IMAGE:TAG
```

例如：

```bash
docker rmi glm5_vllm_env:20260907
```

容器和镜像一定不要混：

```text
docker rm  → 删除容器
docker rmi → 删除镜像
```

---

# 18. 看懂你公司常见的一条 `docker run`

AI 开发容器常见：

```bash
docker run -it \
  --name="$MY_CONTAINER" \
  --network=host \
  --shm-size 20g \
  --cap-add=sys_ptrace \
  -v /workspace:/workspace/volume \
  -v /tmp:/workspace/tmp \
  -v /data:/data \
  -v /data1:/data1 \
  -v /data2:/data2 \
  -v /projs:/projs \
  --privileged \
  --device=/dev/cambricon_ctl \
  --device=/dev/dri \
  IMAGE_NAME \
  /bin/bash
```

不要整段硬看，应该分模块：

```text
docker run
│
├─ 交互终端
│  └─ -it
│
├─ 容器身份
│  └─ --name
│
├─ 网络
│  └─ --network=host
│
├─ 共享内存
│  └─ --shm-size
│
├─ 调试能力
│  └─ --cap-add=sys_ptrace
│
├─ 文件挂载
│  └─ -v ...
│
├─ 权限
│  └─ --privileged
│
├─ 加速卡设备
│  └─ --device
│
├─ 环境模板
│  └─ IMAGE_NAME
│
└─ 启动命令
   └─ /bin/bash
```

理解这个拆解以后，长 Docker 命令就不会再觉得复杂。

---

# 19. `-it` 是什么

```bash
-it
```

其实是：

```text
-i  interactive，保持标准输入
-t  分配 TTY
```

通俗理解：

> 让你可以像正常 Linux 终端一样和容器交互。

开发容器里非常常见。

---

# 20. `--network=host`

你的模型服务场景很常见：

```bash
--network=host
```

意思是：

> 容器直接使用宿主机的网络命名空间。

如果容器内：

```bash
python -m sglang.launch_server --port 30000
```

那宿主机通常直接就能访问：

```bash
curl http://127.0.0.1:30000
```

这就是为什么你的启动脚本和 `evalscope` 测试脚本经常都直接访问同一个宿主机端口。

---

# 21. `-p` 端口映射

如果不用 host network，常见：

```bash
-p 31000:30000
```

意思：

```text
宿主机 31000
      ↓
容器   30000
```

访问：

```text
HOST:31000
```

最终进入容器的：

```text
30000
```

而使用：

```bash
--network=host
```

时通常不需要再 `-p`。

---

# 22. 为什么模型服务会端口冲突

如果两个用户都用 host network，而且都启动：

```text
port 30000
```

第二个人可能报：

```text
Address already in use
```

查看：

```bash
ss -lntp | grep 30000
```

有 `lsof` 时：

```bash
lsof -i :30000
```

所以多人服务器一定要避免随便占公共端口。

---

# 23. `--shm-size 20g`

这是容器 `/dev/shm` 的共享内存大小。

AI 框架中的：

```text
multiprocessing
DataLoader
通信
共享张量
部分 Profiler
```

可能使用 `/dev/shm`。

Docker 默认共享内存可能较小，所以深度学习容器常设：

```bash
--shm-size 20g
```

进入容器可以看：

```bash
df -h /dev/shm
```

---

# 24. `--device`

例如：

```bash
--device=/dev/cambricon_ctl
--device=/dev/dri
```

Linux 里很多硬件通过 `/dev/...` 设备节点访问。

容器默认不是自动拥有宿主机全部硬件权限，所以需要把相应设备提供给容器。

可以理解：

```text
宿主机真实 MLU 设备
↓ --device
容器可以访问
↓
torch_mlu / SGLang 使用设备
```

注意：

```text
MLU_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES
```

主要是框架层“选哪些卡”，前提仍是容器本身有访问硬件的权限。

---

# 25. `--privileged`

```bash
--privileged
```

给予容器很高的宿主机权限。

在硬件调试、Profiler、底层算子开发中可能会用，但隔离性会显著降低。

所以要知道：

> 它是一个很强的权限开关，不是普通 Docker 容器都应该随便开的参数。

公司内部开发脚本用了，就按照团队规范使用；自己写正式服务时不要无脑加。

---

# 26. `--cap-add=sys_ptrace`

Linux 把 root 权限拆成很多 capability。

`SYS_PTRACE` 和：

```text
调试
进程跟踪
Profiler
```

等工作相关。

所以你抓 CNPerf/Profile 之类底层信息时，经常看到它。

---

# 27. `-w /workspace`

```bash
-w /workspace
```

指定容器默认工作目录。

可以近似理解为：

```bash
docker 启动完成后自动 cd /workspace
```

---

# 28. `docker inspect` 是非常重要的排错工具

查看完整信息：

```bash
docker inspect mahao_welm
```

里面能看到：

```text
使用哪个镜像
Mounts
网络模式
环境变量
工作目录
设备
启动命令
IP
HostConfig
```

尤其当你不知道：

> “这个容器的 `/workspace` 到底从宿主机哪里挂进来的？”

就应该看：

```bash
docker inspect mahao_welm
```

看 Mounts。

---

# 29. 查看容器挂载

如果系统支持：

```bash
docker inspect -f '{{json .Mounts}}' mahao_welm
```

你主要关注：

```text
Source       宿主机路径
Destination  容器路径
RW           是否可写
```

---

# 30. `docker logs`

如果容器主进程就是服务：

```bash
docker logs mahao_welm
```

实时：

```bash
docker logs -f mahao_welm
```

不过你现在常见的模式是：

```text
容器主进程 = bash
↓
docker exec 进入
↓
手工 bash launch_welm.sh
```

这种情况下，模型服务日志未必完整出现在 `docker logs` 中。

你自己的：

```bash
2>&1 | tee server.log
```

通常更加可靠。

---

# 31. `docker stats`

看容器 CPU、内存等：

```bash
docker stats
```

适合快速确认：

```text
容器是不是疯狂占内存
进程是否还活跃
```

但 GPU/MLU 利用率仍通常用硬件专用监控工具看。

---

# 32. `docker cp`

容器复制到宿主机：

```bash
docker cp mahao_welm:/root/a.log ./a.log
```

宿主机复制进容器：

```bash
docker cp test.sh mahao_welm:/root/test.sh
```

但是如果文件本来就在挂载目录：

```text
/projs
/data
/workspace/volume
```

一般根本不需要 `docker cp`。

直接操作宿主机对应文件即可。

---

# 33. `docker commit`：保存你已经配好的环境

这是你工作里很实用的命令。

例如：

```text
官方基础镜像
↓
pip install vllm
↓
安装 vllm_mlu
↓
安装 torch_mlu_ops
↓
patch transformers
↓
pip install -e .
↓
模型终于能跑
```

如果以后还要重复使用，可以：

```bash
docker commit mahao_glm5 glm5_vllm_env:20260907
```

表示：

> 把当前容器的文件系统变化固化成一个新镜像。

---

# 34. `docker commit` 保存什么，不保存什么

主要保存：

```text
容器内部 pip 安装包
apt 安装包
/usr/local 下环境
容器内部修改文件
```

但通过：

```bash
-v /projs:/projs
```

挂进去的 `/projs` 数据并不是靠 commit 保存的。

所以正确分工：

```text
环境 → commit / image
代码 → Git
模型/结果 → 挂载目录
```

---

# 35. `docker save` 和 `docker load`

假设服务器 A 上有：

```text
glm5_vllm_env:20260907
```

但服务器 B 没有。

在 A：

```bash
docker save -o /projs/solutionsdk/mahao/glm5_env.tar \
  glm5_vllm_env:20260907
```

这一步：

```text
Docker Image
↓
打包成 tar 文件
```

如果 `/projs` 是共享存储，服务器 B 也能看到这个 tar。

在 B：

```bash
docker load -i /projs/solutionsdk/mahao/glm5_env.tar
```

然后：

```bash
docker images
```

就能看到镜像。

完整链路：

```text
Server A
Container
↓ docker commit
Image
↓ docker save
/projs/env.tar
↓ shared storage
Server B
↓ docker load
Image
↓ docker run
New Container
```

---

# 36. 为什么共享 `/projs` 不代表镜像也共享

共享存储解决的是：

```text
文件
```

Docker image 通常属于当前 Docker daemon 的本地镜像库。

所以：

```bash
docker images
```

在服务器 A 和 B 上通常可能完全不同。

但：

```text
/projs/xxx.tar
```

因为是共享存储，两边都能看到。

因此才需要：

```text
save → tar → load
```

或者使用公司内部 Docker Registry。

---

# 37. Docker Registry

公司镜像地址经常长这样：

```text
registry.company.com/team/image:tag
```

Registry 可以类比：

```text
GitLab 保存 Git 仓库
Docker Registry 保存 Docker 镜像
```

常见：

```bash
docker pull REGISTRY/IMAGE:TAG
docker push REGISTRY/IMAGE:TAG
```

私有仓库可能需要：

```bash
docker login REGISTRY
```

如果团队已经有正式镜像仓库，通常比手工 save/load 更规范。

---

# 38. Docker 和 Git 到底有什么区别

Git：

```text
管理代码版本
```

Docker：

```text
管理运行环境
```

例如一次 WeLM 性能测试真正需要记录：

```text
Docker image
+
Git branch / commit
+
模型版本
+
启动参数
+
测试参数
```

只知道 Git commit 并不一定能复现实验，因为环境可能已经变了。

---

# 39. Docker 和 Conda/venv 区别

Conda/venv 主要解决：

```text
Python 环境隔离
```

Docker 隔离范围更大：

```text
Linux 用户空间
系统库
Python
工具链
网络
进程
文件系统
设备访问
```

所以完全可以有：

```text
宿主机
↓
Docker
↓
Conda
↓
Python
```

只是很多团队镜像本身已经把 Python 环境准备好了，就不一定需要再套 Conda。

---

# 40. 宿主机 Python 和容器 Python 是两个环境

宿主机：

```bash
which python
```

和进入容器后：

```bash
which python
```

很可能完全不同。

因此：

```bash
宿主机 pip install evalscope
```

不代表：

```text
容器里也有 evalscope
```

反过来也一样。

遇到 Python 环境问题，第一组命令建议永远是：

```bash
which python
python -V
python -m pip -V
python -m pip show sglang
```

---

# 41. `pip install -e .` 和 Docker 的关系

假设代码在挂载目录：

```bash
cd /projs/solutionsdk/mahao/sglang-mlu
python -m pip install -e .
```

`-e` 是 editable install。

一般会让 Python 安装信息指向当前源码目录。

于是：

```text
你改挂载目录里的 Python 源码
↓
Python 下一次 import 通常直接用修改后的代码
```

这很适合开发框架。

但是：

```text
C++/CUDA/MLU 扩展
编译产物
setup 配置
依赖列表
```

变化以后仍可能需要重新编译或重新安装。

---

# 42. 为什么切 Git commit 后环境可能突然坏掉

你经常做：

```bash
git checkout OLD_COMMIT
```

但此时：

```text
代码变成旧版本
容器环境仍然是之前的新版本环境
```

如果不同 commit 对：

```text
torch_mlu_ops
transformers
自定义算子
Python API
编译产物
```

要求不同，就可能报错。

所以：

> **Git 回退只回退代码，不会自动回退 Docker 环境和 pip 包。**

正式 A/B 测试要尽量记录环境版本。

---

# 43. 什么时候可以复用旧容器

通常可以：

```text
只改 Python 逻辑
只改启动参数
只改 benchmark 参数
切的是非常接近的 commit
框架依赖没变化
```

如果环境稳定，继续复用最省时间。

---

# 44. 什么时候应该重新起干净容器

建议考虑重建：

```text
负责人换了基础镜像
框架版本跨度很大
依赖已经装乱
pip 冲突严重
编译扩展不匹配
正式性能基线需要干净环境
```

不要为了“省一次安装”让环境长期处于无法解释的状态。

---

# 45. 什么时候应该 `docker commit`

比较适合：

```text
环境配置花了很久
以后会反复复用
环境已经验证模型能跑
准备换服务器
需要给自己保留一个稳定基线
```

不适合因为：

```text
只改了一行代码
只生成了一次结果
```

就 commit 一个新镜像。

代码交给 Git，结果交给共享目录。

---

# 46. 推荐镜像命名方法

不建议：

```text
test
new
final
final2
final_final
```

建议：

```text
welm_sglang:v0.5.18-20260913
glm5_vllm:20260907
welm_mtp:before_direct_topk
welm_mtp:after_direct_topk
```

让名字能表达：

```text
模型/框架 + 关键版本 + 日期/用途
```

---

# 47. Docker 目录权限为什么经常出问题

挂载目录本质属于宿主机。

Linux 判断权限主要看：

```text
UID
GID
rwx
```

宿主机执行：

```bash
id
```

容器里也执行：

```bash
id
```

可能发现 UID/GID 不同。

再看：

```bash
ls -ld /projs/solutionsdk/mahao/results
```

如果当前容器用户没有 write 权限，就会：

```text
Permission denied
```

常见解决方向：

```text
正确 chown/chmod
指定 --user
让 UID/GID 对齐
使用团队规定的可写目录
```

不要为了图省事就：

```bash
chmod -R 777 /projs
```

多人共享环境里风险很大。

---

# 48. 如何判断自己现在是在宿主机还是容器

可以检查：

```bash
ls /.dockerenv
```

很多 Docker 容器会有：

```text
/.dockerenv
```

还可以：

```bash
cat /proc/1/cgroup
hostname
```

以及最实用的习惯：

```bash
pwd
which python
```

你在 VSCode 多层 Remote SSH + Container 场景下尤其要经常确认。

---

# 49. 为什么一个容器会“自己停掉”

Docker 容器的生命周期和它的主进程有关。

如果主进程结束：

```text
容器就停止
```

例如：

```bash
docker run IMAGE echo hello
```

`echo` 执行完以后容器就结束。

开发容器一般让主进程是：

```bash
/bin/bash
```

或者一个长期运行的服务。

---

# 50. 为什么 `docker exec` 有时提示 container is not running

因为 `exec` 只能对 running container 操作。

解决：

```bash
docker ps -a
docker start CONTAINER
docker exec -it CONTAINER /bin/bash
```

如果刚 `start` 又立刻 exited，就应该：

```bash
docker logs CONTAINER
docker inspect CONTAINER
```

看主进程为什么退出。

---

# 51. VSCode 和 Docker 的常见关系

常见流程：

```text
本地 Windows/Ubuntu
↓ VSCode Remote SSH
公司服务器 Host
↓ Attach to Running Container
Docker Container
```

这样 VSCode 中：

```text
文件浏览器
终端
Python 插件
Git
```

都可以工作在容器环境。

不过一定要分清：

> VSCode 当前打开的是宿主机目录，还是容器目录。

---

# 52. 你的典型每日工作流程

建议形成下面这套固定思路。

## 第 1 步：登录服务器

```bash
ssh mahao@SERVER_IP
```

先确认：

```bash
hostname
whoami
pwd
```

## 第 2 步：看 Docker 状态

```bash
docker ps
docker ps -a
docker images
```

## 第 3 步：判断是复用还是新建

容器存在：

```bash
docker start MY_CONTAINER
docker exec -it MY_CONTAINER /bin/bash
```

不存在：

```bash
docker run ...
```

## 第 4 步：进入后确认环境

```bash
pwd
which python
python -V
python -m pip -V
```

## 第 5 步：确认代码版本

```bash
cd /projs/.../sglang-mlu
git status
git branch --show-current
git rev-parse HEAD
```

## 第 6 步：确认框架 import 到哪里

```bash
python -c "import sglang; print(sglang.__file__)"
```

## 第 7 步：启动模型服务

例如：

```bash
bash launch_welm.sh graph
```

## 第 8 步：另一个终端看服务是否监听

```bash
ss -lntp | grep 30000
```

## 第 9 步：跑测试

```bash
bash bench_gsm8k.sh
bash bench_client.sh
```

## 第 10 步：确认结果在挂载目录

```bash
find /projs/.../results -type f | tail
```

## 第 11 步：环境很有价值时固化

```bash
docker commit ...
```

## 第 12 步：换机器时迁移

```bash
docker save ...
docker load ...
```

---

# 53. 一个更合理的 `start_docker.sh` 思维模板

```bash
#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${USER}_welm"
IMAGE="welm_env:latest"

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "container exists: $CONTAINER"
    docker start "$CONTAINER" >/dev/null || true
else
    echo "creating container: $CONTAINER"

    docker run -dit \
        --name "$CONTAINER" \
        --network host \
        --shm-size 20g \
        -v /projs:/projs \
        -v /data:/data \
        "$IMAGE" \
        /bin/bash
fi

docker exec -it "$CONTAINER" /bin/bash
```

逻辑非常简单：

```text
容器是否存在？
├─ 存在 → start
└─ 不存在 → run 创建
↓
exec 进入
```

---

# 54. Docker 常见排错：目录不对

症状：

```text
为什么容器里的 /workspace 没有我想要的代码？
```

排查：

```bash
docker inspect CONTAINER
```

重点看：

```text
Mounts.Source
Mounts.Destination
```

确认宿主机路径和容器路径到底映射到了哪里。

---

# 55. Docker 常见排错：端口访问不到

按顺序检查：

```text
1. 模型进程还活着吗？
2. 实际监听的是哪个端口？
3. 使用 host network 还是 bridge？
4. 如果不是 host，有没有 -p？
5. 服务监听 127.0.0.1 还是 0.0.0.0？
```

命令：

```bash
ps -ef | grep sglang
ss -lntp | grep 30000
curl http://127.0.0.1:30000
```

---

# 56. Docker 常见排错：设备看不到

先判断宿主机硬件是否正常，再看容器。

容器侧重点：

```text
--device 是否正确
--privileged 是否按要求配置
相关 /dev 节点是否存在
Torch/torch_mlu 是否正确
设备环境变量是否正确
```

不要一看到模型报 MLU/GPU 错误就直接认为是框架 bug。

---

# 57. Docker 常见排错：磁盘突然满了

查看：

```bash
df -h
docker system df
```

AI 环境常见大户：

```text
Docker images
容器可写层
pip cache
编译 cache
大日志
误下载到 /root/.cache 的模型权重
```

几十 GB 模型尽量保存到挂载模型盘，而不是容器层。

---

# 58. 工作服务器不要随便 `docker system prune -a`

这个命令可能大量删除：

```text
停止容器
未使用镜像
build cache
网络
```

在多人共享服务器上可能影响别人。

除非团队明确允许，否则不要把它当普通清理命令使用。

---

# 59. Docker 常见排错：我明明安装了包，为什么还是 No module named

先执行：

```bash
which python
python -V
python -m pip -V
python -m pip show PACKAGE
```

再确认：

```text
你安装包时是在宿主机还是容器？
现在启动服务用的是哪个 Python？
有没有切换到另一个容器？
有没有删旧容器新起？
```

这类问题很多不是“pip 没装成功”，而是：

> **你现在工作的 Python 环境已经不是当时安装那个环境。**

---

# 60. Docker 高频命令速查表

| 命令 | 作用 |
|---|---|
| `docker ps` | 查看运行中容器 |
| `docker ps -a` | 查看所有容器 |
| `docker images` | 查看本机镜像 |
| `docker run ...` | 从镜像新建并启动容器 |
| `docker start C` | 启动已有容器 |
| `docker stop C` | 停止容器 |
| `docker exec -it C /bin/bash` | 进入运行中容器 |
| `docker inspect C` | 查看容器详细配置 |
| `docker logs C` | 看容器主进程日志 |
| `docker stats` | 看容器 CPU/内存等 |
| `docker cp` | 宿主机与容器复制文件 |
| `docker rm C` | 删除容器 |
| `docker rmi IMAGE` | 删除镜像 |
| `docker commit C IMAGE:TAG` | 容器固化为镜像 |
| `docker save -o x.tar IMAGE` | 镜像保存成 tar |
| `docker load -i x.tar` | 从 tar 恢复镜像 |
| `docker pull IMAGE` | 从 Registry 拉镜像 |
| `docker push IMAGE` | 推镜像到 Registry |
| `docker system df` | 查看 Docker 磁盘占用 |

---

# 61. 你现阶段最应该熟练掌握的内容

## 第一优先级：每天都会用

```text
Host / Image / Container / Mount
run
start
exec
ps
ps -a
images
stop
rm
-v
--network=host
```

## 第二优先级：AI 工作很重要

```text
--device
--privileged
--shm-size
端口
UID/GID 权限
inspect
logs
stats
```

## 第三优先级：提高效率

```text
commit
save/load
Registry
镜像版本管理
环境复现
```

## 后续再系统学习

```text
Dockerfile
FROM
RUN
COPY
WORKDIR
ENV
ENTRYPOINT
CMD
Docker build
```

---

# 62. 为什么你之后应该学习 Dockerfile

`docker commit` 的优点是：

```text
快
方便
适合研发中保存手工环境
```

缺点是：

> 别人很难知道你为了得到这个环境到底执行了哪些命令。

Dockerfile 会把环境构建过程写成代码：

```dockerfile
FROM ubuntu:22.04

RUN apt-get update
RUN pip install xxx

WORKDIR /workspace
```

于是环境从：

```text
“我手工装好了，你拿我的镜像用”
```

进阶到：

```text
“环境的构建步骤本身也可以版本管理和重复构建”
```

你目前先把日常 Docker 使用搞熟，再学 Dockerfile 会非常顺。

---

# 63. 一次性能测试真正应该记录什么

以后做修复前后 A/B 测试，建议至少记录：

```text
服务器/卡型号
Docker image:tag
Git branch
Git commit
模型路径/版本
GPU/MLU 卡号
启动参数
graph/eager
MTP on/off
服务端口
输入长度
输出长度
parallel
rate
request number
测试脚本版本
结果目录
时间
```

这样一个月以后你仍然能回答：

> “这组数据到底是在什么环境、什么代码、什么参数下跑出来的？”

---

# 64. 推荐的 Docker 工作习惯

1. **自己的容器名带用户名**，例如 `mahao_welm`。
2. **重要代码放 Git + 挂载目录**，不要只放容器内部。
3. **模型放共享模型盘**，不要下载到容器可写层。
4. **测试结果写 `/projs` 等持久目录**。
5. 删容器前先 `docker inspect` 看 Mounts。
6. 切 Git commit 后如果异常，要想到“依赖环境没跟着回退”。
7. 一个配了很久且稳定的环境及时 `docker commit`。
8. 换服务器优先 Registry，内部临时流程也可 `save/load`。
9. 多人机器不要随便 `docker system prune -a`。
10. 每次进容器先确认 `which python` 和当前 Git commit。

---

# 65. 最后用一套模型彻底记住 Docker

你可以永远按这四层理解：

```text
第 1 层：物理资源
Host Server
CPU / RAM / GPU / MLU / Disk / Network

第 2 层：运行环境
Docker Image / Container
Ubuntu / Python / Torch / torch_mlu / libs

第 3 层：代码
Git Repository
SGLang / vLLM / 你修改的源码

第 4 层：数据
Model / Dataset / Benchmark Result / Profile / Logs
Shared Storage
```

然后再看你每天的工作：

```text
SSH 登录 Host
↓
选择 Image
↓
docker run 创建 Container
↓
-v 把代码/模型/结果挂进去
↓
--device 把加速卡提供给 Container
↓
进入容器
↓
使用容器里的 Python/Torch/SGLang
↓
读取挂载目录里的模型和源码
↓
启动 HTTP 模型服务
↓
EvalScope 请求服务跑测试
↓
结果写回共享挂载目录
↓
环境值得保留 → commit
↓
换服务器 → save/load 或 Registry
```

如果你真正理解了这条链路，你对 Docker 就已经不再是“会照着脚本起容器”，而是开始具备 AI Infra 开发环境的整体工程视角。

---

# 66. 最实用的 Docker 排错 Checklist

遇到问题时按顺序问自己：

```text
[ ] 我现在在哪台服务器？
[ ] 我现在在宿主机还是容器？
[ ] docker ps / ps -a 能看到容器吗？
[ ] 容器是 Running 还是 Exited？
[ ] 容器基于哪个 image？
[ ] Mounts 中 Source / Destination 是什么？
[ ] 我的代码是不是在挂载目录？
[ ] 模型是不是在挂载目录？
[ ] 测试结果是不是写在持久目录？
[ ] 当前 which python 是什么？
[ ] python -m pip 对应哪个环境？
[ ] 当前 Git branch / commit 是什么？
[ ] Git commit 和已安装依赖是否匹配？
[ ] GPU/MLU 设备是否被映射进容器？
[ ] MLU_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES 是否正确？
[ ] /dev/shm 是否足够？
[ ] 服务端口是否冲突？
[ ] 网络模式是 host 还是 bridge？
[ ] 输出目录 UID/GID 是否可写？
[ ] 删除容器前还有没有文件只存在容器层？
```

这份 Checklist 对你目前的服务器模型测试工作非常实用。



%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-实习工程能力|模块-实习工程能力]]
- 关联阅读：[[outputs/项目整理/专题-06-服务部署评测与协作|专题-06-服务部署评测与协作]]

%% 项目关联导航：结束 %%
