# Shell 脚本与 Linux 终端命令实战教程
## ——面向“服务器启动模型服务 / Docker / SGLang / EvalScope 性能测试”场景

> 这份文档不是一份只罗列语法的 Bash 手册，而是围绕你实际在服务器上经常遇到的工作流来讲：
>
> **SSH 登录服务器 → 进入/启动 Docker → 配置环境变量 → 启动模型服务 → 传入大量参数 → 保存日志 → 跑精度/性能测试 → 循环多组测试配置 → 查看结果与排错。**
>
> 目标是让你达到三个层次：
>
> 1. **能看懂**别人写的 `.sh` 脚本；
> 2. **敢修改**模型路径、GPU/MLU 卡号、端口、并发、输入输出长度等参数；
> 3. **能自己写**简单但规范的启动脚本、测试脚本和自动化脚本。

---

# 目录

1. Shell、Bash、终端命令到底是什么
2. 一个 Shell 脚本是怎样执行的
3. Shell 最基础语法
4. 变量与环境变量
5. 参数传递：`$0`、`$1`、`$@`
6. 字符串、引号与变量展开
7. 命令替换：`$(...)`
8. 退出码与 `set -euo pipefail`
9. 条件判断：`if`
10. 多分支选择：`case`
11. 循环：`for`、`while`
12. 数组与关联数组
13. 管道：`|`
14. 重定向：`>`、`>>`、`2>&1`
15. `tee`：终端显示同时写日志
16. `exec`：让服务成为主进程
17. 反斜杠 `\` 与多行命令
18. 常用文本处理命令
19. 常用文件与目录命令
20. 常用进程与服务排查命令
21. Docker 常见命令与脚本写法
22. 你常见的模型服务启动脚本怎么读
23. Speculative/MTP 启动脚本怎么读
24. EvalScope 精度测试脚本怎么读
25. EvalScope 性能测试脚本怎么读
26. 自动创建/进入 Docker 的脚本怎么读
27. 你的脚本中值得注意的真实问题
28. 如何安全修改别人的 Shell 脚本
29. 如何自己写一个模型服务启动脚本
30. 如何自己写一个批量性能测试脚本
31. Shell 排错方法
32. 高频语法速查表
33. 建议的学习路线

---

# 1. Shell、Bash、终端命令到底是什么

初学时最容易混淆的是：

- Linux
- Terminal
- Shell
- Bash
- Shell Script

它们不是同一个东西。

可以把它们理解成：

```text
你
↓
终端 Terminal
↓
Shell（例如 Bash）
↓
Linux 内核 / 各种程序
```

## 1.1 Terminal

Terminal（终端）就是你输入命令、看到输出的那个窗口。

例如你通过 SSH 登录服务器以后看到：

```bash
mahao@server:~$
```

这个界面就是终端。

终端本身主要负责：

- 接收键盘输入；
- 显示输出；
- 把输入交给 Shell。

---

## 1.2 Shell

Shell 是一个“命令解释器”。

你输入：

```bash
cd /workspace
```

Shell 会解析：

- `cd` 是什么；
- `/workspace` 是参数；
- 然后执行切换目录。

你输入：

```bash
python test.py
```

Shell 会寻找 `python` 这个程序，再把 `test.py` 作为参数交给它。

---

## 1.3 Bash

Bash 是 Linux 上最常见的一种 Shell。

Shell 有很多种：

```text
sh
bash
zsh
fish
...
```

你现在公司服务器中的 `.sh` 文件，大部分实际上是在用 **Bash 语法**。

可以执行：

```bash
echo $SHELL
```

查看自己的默认 Shell。

常见结果：

```text
/bin/bash
```

---

## 1.4 Shell Script

你平时手敲：

```bash
export MLU_VISIBLE_DEVICES=4,5,6,7
python -m sglang.launch_server \
  --model-path /data/models/WeLM \
  --tp-size 4 \
  --port 30000
```

如果每天都要重复执行，就可以把这些命令写进：

```text
launch_welm.sh
```

然后：

```bash
bash launch_welm.sh
```

Shell 就会从上到下自动执行。

所以：

> **Shell 脚本本质上就是“把原本需要你一条一条输入的终端命令保存到一个文本文件里，再让 Bash 批量执行”。**

---

# 2. 一个 Shell 脚本是怎样执行的

最简单的脚本：

```bash
#!/usr/bin/env bash

echo "hello"
pwd
ls
```

执行：

```bash
bash test.sh
```

Bash 大致会：

```text
读取第 1 行
↓
读取 echo "hello"
↓
执行 echo
↓
读取 pwd
↓
执行 pwd
↓
读取 ls
↓
执行 ls
↓
脚本结束
```

默认情况下就是：

> **从上到下顺序执行。**

只有遇到：

- `if`
- `case`
- `for`
- `while`
- 函数
- `exit`
- 错误终止

等情况时，执行路径才会改变。

---

# 3. Shell 最基础语法

## 3.1 注释

```bash
# 这是注释
```

Bash 不执行 `#` 后面的内容。

例如：

```bash
# 模型路径
MODEL_PATH=/data/models/WeLM
```

你性能脚本中：

```bash
# input_len=3000
# output_len=300
```

表示这两行暂时被禁用。

---

## 3.2 命令与参数

Linux 命令通常都是：

```text
命令 参数 参数 参数
```

例如：

```bash
docker start mahao_welm
```

这里：

```text
docker        主程序
start         docker 的子命令
mahao_welm    参数
```

再比如：

```bash
python -m sglang.launch_server --port 30000
```

可以拆成：

```text
python
├── -m
├── sglang.launch_server
├── --port
└── 30000
```

因此很多你看到的复杂启动命令，其实只是：

> **一个程序 + 很多参数。**

---

# 4. 变量与环境变量

Shell 中非常核心。

---

## 4.1 普通变量

定义变量：

```bash
MODEL_NAME=WeLM
PORT=30000
```

注意：

```bash
PORT=30000
```

正确。

而：

```bash
PORT = 30000
```

错误。

Shell 中 `=` 两边通常不能有空格。

读取变量：

```bash
echo "$PORT"
```

输出：

```text
30000
```

---

## 4.2 为什么推荐 `${VAR}`

两种写法都可以：

```bash
echo $PORT
echo ${PORT}
```

但是更推荐：

```bash
echo "${PORT}"
```

因为边界更清晰。

例如：

```bash
NAME=welm
echo "${NAME}_server"
```

结果：

```text
welm_server
```

如果写：

```bash
echo "$NAME_server"
```

Shell 可能把它理解成变量：

```text
NAME_server
```

而不是：

```text
NAME + "_server"
```

---

# 5. 环境变量：`export`

普通变量：

```bash
PORT=30000
```

只属于当前 Shell。

环境变量：

```bash
export MLU_VISIBLE_DEVICES=4,5,6,7
```

不仅当前 Shell 能看到，**由这个 Shell 启动的子进程也能看到**。

这对模型服务非常重要。

例如：

```bash
export MLU_VISIBLE_DEVICES=4,5,6,7

python -m sglang.launch_server ...
```

Python 启动以后可以读取：

```text
MLU_VISIBLE_DEVICES=4,5,6,7
```

于是框架知道应该使用哪些 MLU。

可以把它理解成：

```text
Shell
│
├── MLU_VISIBLE_DEVICES=4,5,6,7
│
└── 启动 Python
      ↓
   Python 继承这个环境变量
```

查看环境变量：

```bash
echo "$MLU_VISIBLE_DEVICES"
```

或者：

```bash
env | grep MLU
```

---

# 6. Shell 参数：`$0`、`$1`、`$2`、`$@`

假设脚本：

```bash
bash launch_welm.sh graph
```

Shell 会认为：

```text
$0 = launch_welm.sh
$1 = graph
```

如果：

```bash
bash test.sh hello world
```

那么：

```text
$0 = test.sh
$1 = hello
$2 = world
```

常见特殊变量：

| 写法 | 含义 |
|---|---|
| `$0` | 当前脚本名字 |
| `$1` | 第 1 个参数 |
| `$2` | 第 2 个参数 |
| `$#` | 参数数量 |
| `$@` | 所有参数 |
| `$?` | 上一个命令退出码 |
| `$$` | 当前 Shell PID |

例如：

```bash
echo "script = $0"
echo "arg1 = $1"
```

---

# 7. `${1:-default}`：参数默认值

你的启动脚本中有非常典型的：

```bash
MODE="${1:-eager}"
```

意思是：

> 如果用户提供了 `$1`，使用 `$1`；否则使用 `eager`。

因此：

```bash
bash launch_welm.sh graph
```

得到：

```text
MODE=graph
```

而：

```bash
bash launch_welm.sh
```

得到：

```text
MODE=eager
```

---

你还经常看到：

```bash
PORT="${PORT:-30000}"
```

这和 `$1` 略有不同。

意思是：

> 如果当前环境已经设置 `PORT`，就使用外部的 `PORT`；否则默认 30000。

例如：

```bash
bash launch_welm.sh
```

使用：

```text
30000
```

但你也可以临时这样启动：

```bash
PORT=31000 bash launch_welm.sh
```

此时：

```text
PORT=31000
```

这是一种非常好的脚本设计。

同一个脚本不需要改源码，就能覆盖参数。

---

# 8. 引号：单引号、双引号、不加引号

这是 Shell 初学者非常容易出错的地方。

---

## 8.1 双引号 `"..."`

变量会展开。

```bash
NAME=welm
echo "model=$NAME"
```

输出：

```text
model=welm
```

因此一般变量都推荐：

```bash
"${MODEL_PATH}"
"${PORT}"
"${rate}"
```

---

## 8.2 单引号 `'...'`

内容基本按原样保存，变量不会展开。

```bash
NAME=welm
echo '$NAME'
```

输出：

```text
$NAME
```

而不是：

```text
welm
```

你的 EvalScope 中：

```bash
--generation-config '{"timeout":7200000,...}'
```

使用单引号非常合理，因为里面是 JSON。

Shell 会尽量原封不动把 JSON 作为一个参数交给 EvalScope。

---

## 8.3 不加引号

例如：

```bash
--model ${MODEL_NAME}
```

很多情况下也能工作。

但如果变量值包含：

- 空格
- 通配符
- 特殊字符

就可能出问题。

所以推荐：

```bash
--model "${MODEL_NAME}"
```

而不是：

```bash
--model ${MODEL_NAME}
```

---

# 9. 命令替换：`$(...)`

你的脚本中：

```bash
TIME_STAMP=$(date '+%Y%m%d_%H%M%S')
```

Shell 会先执行：

```bash
date '+%Y%m%d_%H%M%S'
```

假设结果：

```text
20260913_010530
```

然后变量成为：

```bash
TIME_STAMP=20260913_010530
```

这叫：

> **Command Substitution，命令替换。**

基本语法：

```bash
VAR=$(某个命令)
```

例如：

```bash
USER_NAME=$(whoami)
CURRENT_DIR=$(pwd)
FILE_COUNT=$(ls | wc -l)
```

---

## 9.1 旧写法：反引号

你 Docker 脚本中有：

```bash
num=`docker ps -a | grep ... | wc -l`
```

这是旧语法。

建议改成：

```bash
num=$(docker ps -a | grep ... | wc -l)
```

原因：

- 更容易读；
- 可以嵌套；
- 现代 Bash 风格统一。

---

# 10. `set -euo pipefail`

你几个较规范的脚本开头有：

```bash
set -euo pipefail
```

这是非常常见的“严格模式”。

可以拆成三个部分。

---

## 10.1 `set -e`

```bash
set -e
```

意思是：

> 某个重要命令执行失败时，脚本尽量立即退出，而不是继续往下跑。

例如：

```bash
set -e

cd /not/exist
python test.py
```

如果：

```bash
cd /not/exist
```

失败，后面的 Python 通常就不会继续执行。

对于：

- 安装依赖
- 启动环境
- 自动测试
- CI

非常有价值。

---

## 10.2 `set -u`

```bash
set -u
```

意思是：

> 使用没有定义的变量时，直接报错。

例如：

```bash
echo "$MODEL_PTAH"
```

你本来想写：

```text
MODEL_PATH
```

但拼错成：

```text
MODEL_PTAH
```

如果不开 `-u`，可能只是得到空字符串。

开启以后会立即告诉你：

```text
unbound variable
```

这能避免很多隐蔽问题。

---

## 10.3 `set -o pipefail`

默认情况下：

```bash
cmd1 | cmd2
```

Shell 通常更关注最后一个命令 `cmd2` 的退出状态。

例如：

```bash
python server.py | tee server.log
```

假设：

```text
python server.py 失败
tee server.log 成功
```

如果没有 `pipefail`，整个管道可能看起来是成功的。

开启：

```bash
set -o pipefail
```

以后，只要管道中关键命令失败，整个 pipeline 就能返回失败。

因此：

```bash
set -euo pipefail
```

特别适合你的服务器脚本。

---

# 11. 退出码：`$?`

Linux 命令执行结束后都会返回一个整数：

```text
0      成功
非 0   失败
```

执行：

```bash
ls /workspace
```

然后：

```bash
echo $?
```

如果成功：

```text
0
```

如果：

```bash
ls /abc_not_exist
```

再：

```bash
echo $?
```

可能得到：

```text
2
```

因此 Shell 中判断“命令成功失败”，很多时候本质就是判断 **exit code**。

---

# 12. `exit`

脚本中：

```bash
exit 2
```

表示：

> 立即结束脚本，并返回退出码 2。

例如你的启动脚本：

```bash
*)
    echo "Usage: $0 [eager|graph]" >&2
    exit 2
    ;;
```

如果用户输入：

```bash
bash launch_welm.sh abc
```

因为 `abc` 既不是：

```text
eager
```

也不是：

```text
graph
```

于是：

1. 输出正确用法；
2. 返回错误码 2；
3. 脚本停止。

---

# 13. 条件判断：`if`

基础结构：

```bash
if 条件; then
    命令
else
    命令
fi
```

注意：

```text
if
then
else
fi
```

其中 `fi` 就是 `if` 反过来写。

---

## 13.1 你的 Docker 脚本

类似：

```bash
if [ 0 -eq "$num" ]; then
    echo "new docker"
else
    echo "docker start"
fi
```

意思是：

```text
如果 num 等于 0
    创建新容器
否则
    启动已有容器
```

---

# 14. `[ ... ]` 是什么

你经常看到：

```bash
[ 0 -eq "$num" ]
```

它其实是一个条件测试。

数字比较：

| 写法 | 含义 |
|---|---|
| `-eq` | 等于 |
| `-ne` | 不等于 |
| `-gt` | 大于 |
| `-lt` | 小于 |
| `-ge` | 大于等于 |
| `-le` | 小于等于 |

例如：

```bash
if [ "$num" -eq 0 ]; then
    echo "zero"
fi
```

---

## 14.1 字符串判断

```bash
if [ "$MODE" = "graph" ]; then
    echo "graph mode"
fi
```

常见：

| 写法 | 含义 |
|---|---|
| `=` | 相等 |
| `!=` | 不相等 |
| `-z "$x"` | 字符串为空 |
| `-n "$x"` | 字符串非空 |

---

## 14.2 文件判断

服务器工作中非常有用。

```bash
[ -f file ]
```

表示普通文件存在。

```bash
[ -d dir ]
```

表示目录存在。

常见：

| 写法 | 含义 |
|---|---|
| `-f` | 文件存在 |
| `-d` | 目录存在 |
| `-e` | 路径存在 |
| `-r` | 可读 |
| `-w` | 可写 |
| `-x` | 可执行 |

例如：

```bash
if [ ! -d "$MODEL_PATH" ]; then
    echo "model path not found: $MODEL_PATH"
    exit 1
fi
```

非常适合启动模型之前做检查。

---

# 15. `[[ ... ]]`

Bash 中还有：

```bash
[[ ... ]]
```

例如：

```bash
if [[ "$MODE" == "graph" ]]; then
    ...
fi
```

如果明确使用 Bash，一般推荐 `[[ ... ]]`，因为：

- 更安全；
- 字符串处理能力更强；
- 支持模式匹配；
- 少一些奇怪的单词拆分问题。

例如：

```bash
if [[ "$MODEL_NAME" == *"WeLM"* ]]; then
    echo "This is WeLM"
fi
```

---

# 16. `case`：多分支选择

你的模型启动脚本：

```bash
case "${MODE}" in
  eager)
    ARGS+=(--disable-cuda-graph)
    ;;
  graph)
    ;;
  *)
    echo "Usage: $0 [eager|graph]" >&2
    exit 2
    ;;
esac
```

可以理解为 Python：

```python
if mode == "eager":
    ...
elif mode == "graph":
    ...
else:
    ...
```

Shell 结构：

```bash
case "$VAR" in
    pattern1)
        ...
        ;;
    pattern2)
        ...
        ;;
    *)
        ...
        ;;
esac
```

其中：

```text
*)
```

相当于默认分支。

---

## 16.1 为什么 graph 分支什么都不写

例如：

```bash
graph)
    ;;
```

意思是：

> graph 模式下不额外增加参数。

因为框架默认可能就启用 graph。

而 eager：

```bash
ARGS+=(--disable-cuda-graph)
```

明确增加：

```text
--disable-cuda-graph
```

因此：

```text
graph = 默认行为
eager = 显式禁用 graph
```

---

# 17. `for` 循环

你的性能测试脚本会跑很多组 rate。

基础写法：

```bash
for x in 1 2 3; do
    echo "$x"
done
```

输出：

```text
1
2
3
```

---

## 17.1 更实际的例子

```bash
for rate in 0.5 1.0 2.0 4.0; do
    echo "rate=$rate"
done
```

就会依次执行四次。

这就是批量性能测试的本质：

```text
第 1 轮 rate=0.5
第 2 轮 rate=1.0
第 3 轮 rate=2.0
第 4 轮 rate=4.0
```

每一轮调用一次：

```bash
evalscope perf ...
```

---

# 18. `while` 循环

虽然你当前几个脚本中没有明显使用，但服务器上很常见。

例如每 5 秒检查一次服务：

```bash
while true; do
    curl http://127.0.0.1:30000/health
    sleep 5
done
```

或者：

```bash
count=0

while [ "$count" -lt 5 ]; do
    echo "$count"
    count=$((count + 1))
done
```

---

# 19. 算术运算 `$((...))`

Shell 做整数运算：

```bash
x=10
y=20
z=$((x + y))
echo "$z"
```

结果：

```text
30
```

例如：

```bash
count=$((count + 1))
```

---

# 20. 普通数组

你的启动脚本中非常重要：

```bash
ARGS=(
  --model-path "${MODEL_PATH}"
  --device mlu
  --dtype bfloat16
  --tp-size 4
  --ep-size 4
)
```

这是 Bash 数组。

逻辑上相当于：

```text
ARGS[0] = --model-path
ARGS[1] = /data/models/...
ARGS[2] = --device
ARGS[3] = mlu
...
```

---

## 20.1 为什么启动命令推荐数组

你本来可以写：

```bash
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --device mlu \
    --dtype bfloat16
```

但当参数很多，并且某些参数要动态添加时，数组更灵活。

例如：

```bash
if [[ "$MODE" == "eager" ]]; then
    ARGS+=(--disable-cuda-graph)
fi
```

最终：

```bash
python -m sglang.launch_server "${ARGS[@]}"
```

---

## 20.2 `"${ARGS[@]}"`

这一句非常重要：

```bash
"${ARGS[@]}"
```

表示：

> 把数组中的每一个元素作为一个独立参数传进去。

例如：

```bash
ARGS=(
  --port
  30000
  --host
  0.0.0.0
)

python server.py "${ARGS[@]}"
```

实际效果约等于：

```bash
python server.py --port 30000 --host 0.0.0.0
```

---

## 20.3 动态追加数组

```bash
ARGS+=(--disable-cuda-graph)
```

表示：

> 在数组末尾追加一个元素。

也可以：

```bash
ARGS+=(--port "$PORT")
```

追加两个元素。

---

# 21. 关联数组：`declare -A`

你的性能脚本：

```bash
declare -A rate_parallel=(
    ["0.3"]=3
    ["0.5"]=5
    ["1.0"]=10
    ["2.0"]=20
)
```

这是 Bash 的“字典 / 哈希表”。

类似 Python：

```python
rate_parallel = {
    "0.3": 3,
    "0.5": 5,
    "1.0": 10,
    "2.0": 20,
}
```

访问：

```bash
parallel=${rate_parallel[$rate]}
```

如果：

```text
rate=1.0
```

那么：

```text
parallel=10
```

---

# 22. `${!array[@]}` 是什么

例如：

```bash
"${!rate_parallel[@]}"
```

表示：

> 获取关联数组所有 key。

例如：

```text
0.3
0.5
1.0
2.0
...
```

而：

```bash
"${rate_parallel[@]}"
```

得到所有 value。

---

# 23. 管道 `|`

例如：

```bash
docker ps -a | grep welm | wc -l
```

可以理解成流水线：

```text
docker ps -a
    ↓
输出交给 grep welm
    ↓
匹配结果交给 wc -l
    ↓
统计行数
```

即：

```text
前一个命令的标准输出
↓
成为后一个命令的标准输入
```

---

## 23.1 实际拆解

```bash
docker ps -a
```

输出所有容器。

然后：

```bash
docker ps -a | grep mahao
```

只保留包含：

```text
mahao
```

的行。

然后：

```bash
docker ps -a | grep mahao | wc -l
```

统计有多少行。

如果结果：

```text
1
```

说明匹配到 1 个。

---

# 24. `grep`

`grep` 用于搜索文本。

例如：

```bash
ps -ef | grep sglang
```

查找带有 `sglang` 的进程。

```bash
docker ps | grep mahao
```

查找你的容器。

常用：

```bash
grep "error" server.log
```

忽略大小写：

```bash
grep -i "error" server.log
```

显示行号：

```bash
grep -n "error" server.log
```

递归搜索代码：

```bash
grep -R "direct_topk" python/
```

更常用的现代代码搜索工具还有：

```bash
rg "direct_topk"
```

即 ripgrep。

---

# 25. `wc`

统计工具。

```bash
wc -l file.txt
```

统计行数。

```bash
docker ps | wc -l
```

统计输出行数。

---

# 26. `sort`

你的性能脚本：

```bash
sort -n
```

`-n` 表示按数值排序。

例如：

```text
20.0
2.0
0.5
1.0
```

执行：

```bash
sort -n
```

得到：

```text
0.5
1.0
2.0
20.0
```

---

# 27. `tr`

你的性能脚本：

```bash
tr ' ' '\n'
```

表示：

> 把空格替换成换行。

假设：

```text
0.3 0.5 1.0 2.0
```

转换后：

```text
0.3
0.5
1.0
2.0
```

然后就方便交给：

```bash
sort -n
```

排序。

---

# 28. 重定向

Shell 有三个最重要的数据流：

```text
stdin   标准输入    0
stdout  标准输出    1
stderr  标准错误    2
```

---

## 28.1 `>`

覆盖写文件：

```bash
echo hello > test.txt
```

`test.txt` 原内容会被覆盖。

---

## 28.2 `>>`

追加：

```bash
echo hello >> test.txt
```

不会清空原文件，而是在末尾添加。

日志经常使用：

```bash
python server.py >> server.log
```

---

## 28.3 `2>`

只把错误输出写文件：

```bash
python test.py 2> error.log
```

---

## 28.4 `2>&1`

你启动脚本里：

```bash
python ... 2>&1
```

意思是：

> 把 stderr（2）重定向到 stdout（1）当前去向。

也就是把：

```text
普通输出
错误输出
```

合并成一个流。

这对于保存完整日志非常重要。

---

# 29. `tee`

你的脚本：

```bash
python ... 2>&1 | tee "${LOG_FILE}"
```

如果只是：

```bash
python ... > server.log
```

日志进文件了，但终端可能看不到实时输出。

而：

```bash
python ... | tee server.log
```

表示：

```text
Python 输出
   ↓
 tee
 ├── 显示到终端
 └── 写入 server.log
```

因此你能：

- 实时看模型加载进度；
- 同时保存完整日志。

---

## 29.1 `tee -a`

追加而不是覆盖：

```bash
command | tee -a server.log
```

---

# 30. `exec`

你的模型启动脚本：

```bash
exec python -m sglang.launch_server "${ARGS[@]}"
```

普通情况：

```bash
bash
└── python
```

Shell 作为父进程启动 Python。

而 `exec`：

```bash
exec python ...
```

会让当前 Shell 进程直接被 Python 替换。

可以近似理解成：

```text
原 Bash 进程
↓ exec
变成 Python 服务进程
```

这在：

- Docker
- 服务脚本
- 容器 entrypoint

中很常见。

好处是：

- 信号传递更直接；
- PID 关系更简单；
- Ctrl+C / SIGTERM 更容易正确交给模型服务。

---

## 30.1 `exec` + 管道的细节

类似：

```bash
exec python ... 2>&1 | tee server.log
```

因为存在管道，Bash 的进程关系会比单独 `exec python ...` 更复杂。

但从实际使用角度，你重点理解为：

```text
启动模型服务
↓
合并 stdout/stderr
↓
通过 tee 实时显示并保存日志
```

即可。

如果脚本使用：

```bash
set -o pipefail
```

还能避免 Python 失败但 `tee` 成功导致整体被误判为成功。

---

# 31. 多行命令：反斜杠 `\`

你经常看到：

```bash
evalscope perf \
    --url http://127.0.0.1:30000 \
    --parallel 10 \
    --number 200 \
    --temperature 0
```

这里每行结尾：

```text
\
```

表示：

> 当前命令还没结束，下一行继续。

如果没有反斜杠：

```bash
evalscope perf
--url ...
```

Shell 会认为：

```text
第一条命令：evalscope perf
第二条命令：--url ...
```

自然会报错。

---

## 31.1 一个很重要的规则

反斜杠：

```text
\
```

必须是这一行最后真正的字符之一。

不要写成：

```bash
--rate "$rate" \ something
```

因为此时语义已经完全变了。

---

# 32. `echo`

最简单的打印命令：

```bash
echo "Starting server"
```

它在脚本中的作用不只是“显示文字”，还常用于告诉你当前运行到了哪一步。

例如：

```bash
echo "mode: $MODE"
echo "model: $MODEL_PATH"
echo "port: $PORT"
```

启动服务之前把关键配置打印出来，是很好的习惯。

否则你测完以后甚至不知道自己到底用了哪个配置。

---

# 33. `printf`

相比 `echo`，`printf` 更稳定、格式化能力更强。

例如：

```bash
printf "mode=%s port=%s\n" "$MODE" "$PORT"
```

适合正式脚本。

---

# 34. `date`

你脚本中：

```bash
date +%Y%m%d_%H%M%S
```

可能得到：

```text
20260913_012530
```

常用于日志名：

```bash
LOG_FILE="server_$(date +%Y%m%d_%H%M%S).log"
```

每次启动就产生：

```text
server_20260913_012530.log
server_20260913_021100.log
...
```

避免日志相互覆盖。

---

# 35. 常用文件系统命令

## `pwd`

查看当前目录：

```bash
pwd
```

---

## `ls`

查看目录：

```bash
ls
```

详细：

```bash
ls -lh
```

隐藏文件：

```bash
ls -la
```

---

## `cd`

进入目录：

```bash
cd /workspace
```

返回上一级：

```bash
cd ..
```

回 home：

```bash
cd ~
```

回上一次目录：

```bash
cd -
```

---

## `mkdir`

创建目录：

```bash
mkdir results
```

自动创建多层：

```bash
mkdir -p outputs/gsm8k/run1
```

---

## `cp`

复制：

```bash
cp a.sh b.sh
```

复制目录：

```bash
cp -r src backup
```

---

## `mv`

移动或重命名：

```bash
mv test.sh test_old.sh
```

---

## `rm`

删除：

```bash
rm file
```

目录：

```bash
rm -r dir
```

强制递归：

```bash
rm -rf dir
```

服务器上使用：

```bash
rm -rf
```

一定要非常谨慎。

---

# 36. 查看文件内容

## `cat`

```bash
cat launch_welm.sh
```

适合短文件。

---

## `less`

```bash
less server.log
```

适合大日志。

常见操作：

```text
q      退出
/xxx   搜索 xxx
n      下一个结果
G      跳到文件尾
g      跳到文件头
```

---

## `head`

```bash
head -n 20 server.log
```

看前 20 行。

---

## `tail`

```bash
tail -n 50 server.log
```

看最后 50 行。

实时看日志：

```bash
tail -f server.log
```

这是服务器工作极其常用的命令。

---

# 37. 查找文件：`find`

例如：

```bash
find /workspace -name "*.log"
```

查找 log。

你之前类似：

```bash
find /workspace -type f | grep -E "json|csv|log|result|output"
```

可以理解为：

```text
find /workspace -type f
```

先找所有普通文件，然后：

```text
grep -E "json|csv|log|result|output"
```

只保留路径中包含这些关键字的文件。

更直接也可以：

```bash
find /workspace -type f \( -name "*.json" -o -name "*.csv" -o -name "*.log" \)
```

---

# 38. `which` 与 `command -v`

查看命令来自哪里：

```bash
which python
which evalscope
```

更推荐：

```bash
command -v python
command -v evalscope
```

例如：

```text
/usr/local/bin/python
```

用于排查：

> 为什么我明明安装了包，却执行到另一个 Python 环境？

---

# 39. `ps`

查看进程。

```bash
ps -ef
```

查模型服务：

```bash
ps -ef | grep sglang
```

---

# 40. `kill`

结束进程：

```bash
kill PID
```

例如：

```bash
kill 12345
```

先尽量不要上来就：

```bash
kill -9 12345
```

`-9` 是强制杀死，进程没有清理机会。

---

# 41. `nohup`

如果希望退出 SSH 后程序继续运行：

```bash
nohup bash launch_welm.sh > server.log 2>&1 &
```

这里：

```text
nohup      忽略终端退出信号
>          stdout 写日志
2>&1       stderr 合并
&          后台运行
```

不过实际开发中，更推荐使用：

```text
tmux
```

因为你可以重新进入会话继续看终端。

---

# 42. `&`

命令放后台：

```bash
python server.py &
```

Shell 不再等待它结束。

查看当前 Shell 后台任务：

```bash
jobs
```

---

# 43. `tmux`

服务器非常推荐。

新建：

```bash
tmux new -s welm
```

在里面启动：

```bash
bash launch_welm.sh
```

暂时退出但不结束任务：

```text
Ctrl+B
然后 D
```

重新进入：

```bash
tmux attach -t welm
```

查看：

```bash
tmux ls
```

这样 SSH 断开通常也不会导致里面的任务跟着停止。

---

# 44. Docker 基础理解

你公司的工作流通常是：

```text
物理服务器
↓
Docker 容器
↓
Python / torch / torch_mlu / SGLang
↓
模型服务
```

Docker 容器不是虚拟机，但可以先近似理解为：

> 在服务器上隔离出来的一个运行环境。

它有：

- 自己的进程空间；
- 自己的软件环境；
- 自己的 Python 包；
- 自己的文件系统层。

同时又可以通过 `-v` 挂载宿主机目录。

---

# 45. `docker ps`

查看正在运行：

```bash
docker ps
```

查看包括停止的：

```bash
docker ps -a
```

---

# 46. `docker images`

查看本机镜像：

```bash
docker images
```

---

# 47. `docker start`

启动已有容器：

```bash
docker start mahao_dev_wrksp
```

注意：

> `start` 是启动已经存在但停止的容器，不会创建新容器。

---

# 48. `docker exec`

进入正在运行的容器：

```bash
docker exec -it mahao_dev_wrksp /bin/bash
```

参数：

```text
docker exec
-i    保持标准输入
-t    分配终端
容器名
/bin/bash
```

最终就是：

> 在这个容器中启动一个 Bash，然后把你的终端接进去。

---

# 49. `docker run`

创建并启动新容器。

例如：

```bash
docker run -it \
    --name=mahao_dev \
    --network=host \
    -v /projs:/projs \
    image_name \
    /bin/bash
```

---

## 49.1 `--name`

```bash
--name=mahao_dev
```

给容器起名字。

---

## 49.2 `--network=host`

```bash
--network=host
```

容器直接使用宿主机网络。

所以容器中模型服务监听：

```text
30000
```

宿主机通常也能直接通过相同端口访问。

---

## 49.3 `-v`

例如：

```bash
-v /projs:/projs
```

格式：

```text
-v 宿主机目录:容器目录
```

即：

```text
宿主机 /projs
↓ 挂载
容器 /projs
```

容器里修改挂载目录中的文件，实际会写回宿主机对应目录。

这也是为什么：

> 删除容器后，挂载目录中的代码和结果通常仍然存在。

---

## 49.4 `-w`

```bash
-w /workspace
```

表示：

> 容器启动后的默认工作目录是 `/workspace`。

类似进入容器后自动：

```bash
cd /workspace
```

---

## 49.5 `--shm-size`

```bash
--shm-size 20g
```

设置 `/dev/shm` 共享内存大小。

深度学习、数据加载、多进程程序经常需要较大的共享内存。

---

## 49.6 `--device`

例如：

```bash
--device=/dev/cambricon_ctl
```

把宿主机设备节点暴露给容器，使容器能够访问相应硬件。

---

## 49.7 `--privileged`

```bash
--privileged
```

给容器非常高的宿主机权限。

公司内部 GPU/MLU 调试环境中可能会用到，但普通生产环境要谨慎，因为隔离性会明显降低。

---

# 50. 解析你的 `launch_welm.sh`

你的模型启动脚本大致可以分成 6 个阶段：

```text
1. 指定解释器
2. 开启严格模式
3. 读取用户参数 / 默认值
4. 设置设备环境变量
5. 构造 SGLang 参数数组
6. 根据 eager/graph 动态修改参数
7. 打印配置
8. 启动服务
```

---

## 50.1 脚本头

```bash
#!/usr/bin/env bash
set -euo pipefail
```

第一行表示：

> 从当前环境的 PATH 中寻找 `bash` 来执行脚本。

相比：

```bash
#!/bin/bash
```

`/usr/bin/env bash` 在不同系统环境中通常更灵活。

---

## 50.2 MODE

```bash
MODE="${1:-eager}"
```

意味着：

```bash
bash launch_welm.sh
```

默认：

```text
eager
```

而：

```bash
bash launch_welm.sh graph
```

就是：

```text
graph
```

---

## 50.3 可覆盖的模型路径和端口

```bash
MODEL_PATH="${MODEL_PATH:-/data/models/WeLMV4.5_YARN}"
PORT="${PORT:-30000}"
HOST="${HOST:-0.0.0.0}"
```

这是一种很实用的写法。

默认：

```bash
bash launch_welm.sh
```

临时换端口：

```bash
PORT=31000 bash launch_welm.sh
```

临时换模型：

```bash
MODEL_PATH=/data/models/new_model bash launch_welm.sh
```

同时换：

```bash
MODEL_PATH=/data/models/new_model \
PORT=31000 \
MLU_VISIBLE_DEVICES=0,1,2,3 \
bash launch_welm.sh graph
```

无需修改脚本。

---

## 50.4 设备变量

```bash
export MLU_VISIBLE_DEVICES="${MLU_VISIBLE_DEVICES:-4,5,6,7}"
```

默认使用：

```text
4,5,6,7
```

但支持外部覆盖：

```bash
MLU_VISIBLE_DEVICES=0,1,2,3 bash launch_welm.sh
```

---

## 50.5 参数数组

```bash
ARGS=(
  --model-path "${MODEL_PATH}"
  --device mlu
  --dtype bfloat16
  --tp-size 4
  --ep-size 4
  ...
)
```

最终：

```bash
python -m sglang.launch_server "${ARGS[@]}"
```

会拼成一个完整启动命令。

可以理解成最终实际执行：

```bash
python -m sglang.launch_server \
  --model-path /data/models/WeLMV4.5_YARN \
  --device mlu \
  --dtype bfloat16 \
  --tp-size 4 \
  --ep-size 4 \
  ...
```

---

## 50.6 eager 与 graph

```bash
case "${MODE}" in
  eager)
    ARGS+=(--disable-cuda-graph)
    ;;
  graph)
    ;;
esac
```

因此：

### eager

数组额外包含：

```text
--disable-cuda-graph
```

### graph

不增加该参数。

---

## 50.7 打印启动配置

```bash
echo "Starting WeLM service"
echo "  mode:    ${MODE}"
echo "  model:   ${MODEL_PATH}"
echo "  devices: ${MLU_VISIBLE_DEVICES}"
echo "  address: http://${HOST}:${PORT}"
```

这是非常值得保留的做法。

以后抓 profile 或做 A/B 性能测试时，一眼就知道：

- 哪个模式；
- 哪个模型；
- 哪几张卡；
- 哪个端口。

---

## 50.8 最终启动

```bash
exec python -m sglang.launch_server "${ARGS[@]}"
```

表示真正启动 SGLang server。

---

# 51. 解析 `launch_welm_spec.sh`

这个脚本相比普通启动脚本，多了一组 Speculative Decoding / MTP 相关配置。

---

## 51.1 开启环境开关

```bash
export SGLANG_ENABLE_SPEC_V2=1
```

表示给 SGLang 子进程传入一个环境变量开关。

框架内部可以读取该变量并选择对应逻辑。

---

## 51.2 动态追加 speculative 参数

```bash
ARGS+=(
  --speculative-algorithm EAGLE
  --speculative-draft-model-path "${MODEL_PATH}"
  --speculative-num-steps 1
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 2
)
```

这体现了数组写法的优点：

```text
公共参数
+
Speculative 专属参数
```

脚本结构很清晰。

---

## 51.3 日志文件

```bash
LOG_FILE="server_$(date +%Y%m%d_%H%M%S).log"
```

例如生成：

```text
server_20260913_013012.log
```

---

## 51.4 保存日志

```bash
exec python -m sglang.launch_server "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
```

数据流：

```text
SGLang stdout ─┐
               ├─> tee ─> 终端
SGLang stderr ─┘       └─> server_xxx.log
```

---

# 52. 解析 `bench_gsm8k.sh`

它的整体目的很简单：

> 调用已经启动好的 OpenAI API 兼容模型服务，让 EvalScope 使用 GSM8K 数据集做评测，并保存结果。

结构：

```text
配置模型
↓
配置 API 地址
↓
evalscope eval
↓
传数据集和 generation 参数
↓
保存结果
```

---

## 52.1 模型与服务地址

```bash
MODEL_PATH="/data/models/WeLMV4.5_YARN"
MODEL_ID="WeLMV4.5_YARN"
API_URL="http://127.0.0.1:30000/v1/chat/completions"
```

这里最重要的联系是：

```text
服务启动脚本 PORT=30000
↓
测试脚本 API_URL 也必须访问 30000
```

如果启动服务：

```text
31000
```

但测试脚本仍然访问：

```text
30000
```

那自然连接失败。

---

## 52.2 `evalscope eval`

```bash
evalscope eval \
  --model "${MODEL_PATH}" \
  --model-id "${MODEL_ID}" \
  --datasets gsm8k \
  ...
```

本质：

```text
程序：evalscope
子命令：eval
参数：--model ...
      --datasets gsm8k
      ...
```

---

## 52.3 JSON 配置为什么加单引号

```bash
--generation-config '{"timeout":..., "temperature":0.0, ...}'
```

因为希望整段 JSON：

```json
{"timeout":7200000,"batch_size":32,...}
```

作为 **一个完整参数** 传给 EvalScope。

单引号可以避免 Shell 随意解释里面的大部分特殊字符。

---

# 53. 解析 `bench_client.sh`

这是典型的：

> **自动跑多组压力测试配置。**

它做的事情：

```text
准备模型参数
↓
定义 rate → parallel 映射
↓
按 rate 从小到大排序
↓
每组调用一次 evalscope perf
↓
保存结果
```

---

## 53.1 定义测试参数

```bash
MODEL_NAME=WeLMV4.5_YARN-W8A8
tokenizer_path=/data/models/WeLMV4.5_YARN-W8A8
```

---

## 53.2 定义关联数组

```bash
declare -A rate_parallel=(
    ["0.3"]=3
    ["0.5"]=5
    ["1.0"]=10
    ["2.0"]=20
    ["4.0"]=40
)
```

表示：

```text
rate=0.3 → parallel=3
rate=0.5 → parallel=5
rate=1.0 → parallel=10
...
```

---

## 53.3 获取并排序 rate

原脚本：

```bash
for rate in $(echo "${!rate_parallel[@]}" | tr ' ' '\n' | sort -n); do
```

拆开：

### 第一步

```bash
"${!rate_parallel[@]}"
```

获取所有 key。

### 第二步

```bash
echo ...
```

输出。

### 第三步

```bash
tr ' ' '\n'
```

把空格变成换行。

### 第四步

```bash
sort -n
```

按数值排序。

### 第五步

```bash
for rate in ...
```

逐个执行。

---

## 53.4 根据 rate 找 parallel

```bash
parallel=${rate_parallel[$rate]}
```

例如：

```text
rate=4.0
```

则：

```text
parallel=40
```

---

## 53.5 跑压测

```bash
evalscope perf \
    --url http://127.0.0.1:30000/v1/chat/completions \
    --model "${MODEL_NAME}" \
    --dataset random \
    --parallel "${parallel}" \
    --rate "${rate}" \
    --number 200 \
    ...
```

每一轮只是：

```text
rate
parallel
```

变化，其他条件基本固定。

这就是标准的性能扫点。

---

# 54. 解析 `start_docker.sh`

宏观流程：

```text
生成自己的容器名
↓
检查容器是否存在
↓
不存在
    → docker run 新建
存在
    → docker start
    → docker exec 进入
```

---

## 54.1 根据用户名自动命名

```bash
export MY_CONTAINER="$(whoami)_dev_wrksp"
```

如果：

```bash
whoami
```

输出：

```text
mahao
```

那么：

```text
MY_CONTAINER=mahao_dev_wrksp
```

因此同一个脚本给不同员工执行时：

```text
zhangsan_dev_wrksp
lisi_dev_wrksp
mahao_dev_wrksp
```

不会轻易重名。

---

## 54.2 检查容器是否存在

旧写法类似：

```bash
num=`docker ps -a | grep ... | wc -l`
```

更推荐：

```bash
num=$(docker ps -a --format '{{.Names}}' | grep -cx "$MY_CONTAINER" || true)
```

甚至可以不统计数量：

```bash
if docker ps -a --format '{{.Names}}' | grep -qx "$MY_CONTAINER"; then
    ...
else
    ...
fi
```

这样更清楚。

---

# 55. 你的脚本中两个真实问题

---

## 55.1 `start_docker.sh` 的 shebang

当前：

```bash
#/bin/bash
```

标准应为：

```bash
#!/bin/bash
```

或者：

```bash
#!/usr/bin/env bash
```

差别是缺少：

```text
!
```

为什么有时你执行：

```bash
bash start_docker.sh
```

似乎仍然能运行？

因为这时是你明确告诉：

```text
bash
```

去解释文件，所以第一行只是被当成注释。

但如果赋予执行权限：

```bash
chmod +x start_docker.sh
```

然后：

```bash
./start_docker.sh
```

正确 shebang 就很重要。

---

## 55.2 `bench_client.sh` 中疑似误插入的 `vim`

当前存在类似：

```bash
--rate "${rate}"  \vim bench_client.sh
```

这很像你在终端中原本想执行：

```bash
vim bench_client.sh
```

但误进入了脚本正文。

正确大概率应该是：

```bash
--rate "${rate}" \
```

为什么原写法有问题？

Shell 会把：

```text
\vim
```

解析为一个奇怪的转义/文本，而原本的多行续行结构被破坏。

最终 `evalscope perf` 接收到的参数会异常，下一行：

```bash
--number 200
```

甚至可能被当成另一条独立命令。

所以修改 Shell 脚本以后，强烈建议先运行：

```bash
bash -n bench_client.sh
```

做语法检查。

---

# 56. `bash -n`：只检查语法，不执行

非常适合你。

```bash
bash -n launch_welm.sh
```

如果没有输出，一般表示 Bash 语法层面没发现问题。

注意：

> 它只能检查语法，不能判断模型路径是否存在、端口是否占用、参数是否被 SGLang 支持。

---

# 57. `bash -x`：调试神器

执行：

```bash
bash -x launch_welm.sh graph
```

Bash 会把实际展开后的命令打印出来。

例如你写：

```bash
PORT="${PORT:-30000}"
echo "$PORT"
```

调试时可能看到：

```text
+ PORT=30000
+ echo 30000
30000
```

这对于搞懂：

```text
变量到底是多少？
哪个 if 被执行了？
数组最后有什么？
实际执行的命令是什么？
```

特别有用。

---

# 58. 临时开启调试

脚本中：

```bash
set -x
```

开始打印调试信息。

```bash
set +x
```

关闭。

例如：

```bash
echo "before"

set -x
python server.py --port "$PORT"
set +x

echo "after"
```

---

# 59. 如何修改别人的 Shell 脚本：先分层

拿到一个复杂脚本，不要从第一个字符逐字看。

先找这几层：

```text
① 输入是什么？
② 配置变量是什么？
③ 环境变量是什么？
④ 主命令是什么？
⑤ 条件分支是什么？
⑥ 循环变量是什么？
⑦ 输出保存在哪里？
```

以模型启动脚本为例：

### 输入

```bash
$1
MODEL_PATH
PORT
MLU_VISIBLE_DEVICES
```

### 主命令

```bash
python -m sglang.launch_server
```

### 可变参数

```bash
ARGS=(...)
```

### 分支

```bash
case "$MODE"
```

### 输出

```text
终端 / server_xxx.log
```

这样脚本立刻就从几十行变成一张逻辑图。

---

# 60. 修改脚本时优先改“变量”，不要乱改主逻辑

假设只是换端口。

优先改：

```bash
PORT=31000
```

而不要到处搜索：

```text
30000
```

然后全部手工替换。

更好的脚本甚至不需要改文件：

```bash
PORT=31000 bash launch_welm.sh
```

---

# 61. 修改前备份

最简单：

```bash
cp launch_welm.sh launch_welm.sh.bak
```

更推荐 Git：

```bash
git status
git diff
```

修改以后：

```bash
git diff -- launch_welm.sh
```

看自己到底改了什么。

---

# 62. 修改后的推荐检查顺序

```text
1. bash -n
2. git diff
3. echo/打印关键变量
4. 小规模执行
5. 查看 exit code
6. 查看日志
7. 再跑正式测试
```

例如：

```bash
bash -n bench_client.sh
git diff -- bench_client.sh
bash -x bench_client.sh
```

---

# 63. 自己写模型启动脚本：推荐模板

```bash
#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-graph}"

MODEL_PATH="${MODEL_PATH:-/data/models/WeLM}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
DEVICES="${MLU_VISIBLE_DEVICES:-0,1,2,3}"

export MLU_VISIBLE_DEVICES="$DEVICES"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: model path does not exist: $MODEL_PATH" >&2
    exit 1
fi

ARGS=(
    --model-path "$MODEL_PATH"
    --device mlu
    --tp-size 4
    --host "$HOST"
    --port "$PORT"
)

case "$MODE" in
    eager)
        ARGS+=(--disable-cuda-graph)
        ;;
    graph)
        ;;
    *)
        echo "Usage: $0 [eager|graph]" >&2
        exit 2
        ;;
esac

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/server_$(date +%Y%m%d_%H%M%S).log"

echo "========== Server Config =========="
echo "mode    : $MODE"
echo "model   : $MODEL_PATH"
echo "devices : $MLU_VISIBLE_DEVICES"
echo "address : http://$HOST:$PORT"
echo "log     : $LOG_FILE"
echo "==================================="

python -m sglang.launch_server "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
```

这个模板已经具备：

- 默认值；
- 参数覆盖；
- 输入检查；
- 参数数组；
- eager/graph 分支；
- 自动日志目录；
- 时间戳日志；
- 错误输出；
- 严格模式。

---

# 64. 自己写批量性能测试脚本：推荐模板

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-WeLM}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/data/models/WeLM}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:30000/v1/chat/completions}"
OUTPUT_DIR="${OUTPUT_DIR:-bench_outputs}"

NUMBER="${NUMBER:-200}"
WARMUP="${WARMUP:-10}"

mkdir -p "$OUTPUT_DIR"

declare -A RATE_PARALLEL=(
    ["0.3"]=3
    ["0.5"]=5
    ["1.0"]=10
    ["2.0"]=20
    ["4.0"]=40
)

mapfile -t RATES < <(
    printf '%s\n' "${!RATE_PARALLEL[@]}" | sort -n
)

for rate in "${RATES[@]}"; do
    parallel="${RATE_PARALLEL[$rate]}"

    echo
    echo "========================================"
    echo "rate=$rate parallel=$parallel"
    echo "========================================"

    evalscope perf \
        --url "$SERVER_URL" \
        --model "$MODEL_NAME" \
        --dataset random \
        --api-key "" \
        --parallel "$parallel" \
        --rate "$rate" \
        --number "$NUMBER" \
        --temperature 0.0 \
        --min-prompt-length 11000 \
        --max-prompt-length 11000 \
        --min-tokens 100 \
        --max-tokens 100 \
        --prefix-length 0 \
        --tokenizer-path "$TOKENIZER_PATH" \
        --name "$MODEL_NAME" \
        --warmup-num "$WARMUP" \
        --outputs-dir "$OUTPUT_DIR"

    echo "Finished rate=$rate parallel=$parallel"
done

echo "All benchmark runs completed."
```

---

# 65. 为什么这里用了 `mapfile`

原来：

```bash
for rate in $(...)
```

能工作，但存在 word splitting。

更稳健的 Bash 写法：

```bash
mapfile -t RATES < <(
    printf '%s\n' "${!RATE_PARALLEL[@]}" | sort -n
)
```

然后：

```bash
for rate in "${RATES[@]}"; do
```

这属于更规范的 Bash 风格。

你初期可以先掌握原写法，再逐渐学习这种写法。

---

# 66. `< <(...)` 是什么

这是 Bash 的 process substitution。

例如：

```bash
mapfile -t arr < <(printf '%s\n' a b c)
```

可以先理解成：

> 把右边命令的输出，当成左边命令的输入。

属于稍进阶语法，不要求一开始就熟练。

---

# 67. 函数

当脚本变长时，可以封装。

例如：

```bash
log() {
    echo "[$(date '+%F %T')] $*"
}
```

然后：

```bash
log "Starting server"
log "port=$PORT"
```

输出：

```text
[2026-09-13 01:30:00] Starting server
```

---

## 67.1 带参数函数

```bash
check_dir() {
    local path="$1"

    if [[ ! -d "$path" ]]; then
        echo "directory not found: $path" >&2
        return 1
    fi
}
```

调用：

```bash
check_dir "$MODEL_PATH"
```

---

# 68. `local`

函数内部：

```bash
local path="$1"
```

表示这个变量主要只属于当前函数。

避免污染外部变量。

---

# 69. `return` 和 `exit`

函数里一般：

```bash
return 1
```

退出函数。

脚本整体：

```bash
exit 1
```

退出整个脚本。

---

# 70. `&&`

```bash
cmd1 && cmd2
```

表示：

> `cmd1` 成功以后才执行 `cmd2`。

例如：

```bash
cd /workspace && ls
```

如果 `/workspace` 不存在，`ls` 不执行。

---

# 71. `||`

```bash
cmd1 || cmd2
```

表示：

> `cmd1` 失败以后才执行 `cmd2`。

例如：

```bash
mkdir results || echo "mkdir failed"
```

或者：

```bash
grep -q pattern file || true
```

常用于在 `set -e` 环境中允许某个命令失败。

---

# 72. `;`

```bash
cmd1; cmd2
```

无论前面成功失败，一般都会继续执行后面。

例如：

```bash
pwd; ls
```

---

# 73. `&&`、`||`、`;` 的区别

```text
cmd1 && cmd2
```

前面成功才执行后面。

```text
cmd1 || cmd2
```

前面失败才执行后面。

```text
cmd1 ; cmd2
```

通常无条件继续。

---

# 74. `$?`

例如：

```bash
python test.py
echo $?
```

如果：

```text
0
```

表示程序正常结束。

调试脚本时很有用。

---

# 75. `curl`

模型 API 排错非常有用。

例如测试端口：

```bash
curl http://127.0.0.1:30000
```

如果框架有 health endpoint：

```bash
curl http://127.0.0.1:30000/health
```

如果：

```text
Connection refused
```

通常意味着：

- 服务没启动；
- 端口不对；
- 服务已经挂了；
- 监听地址有问题。

---

# 76. 查看端口

常用：

```bash
ss -lntp
```

过滤：

```bash
ss -lntp | grep 30000
```

含义：

```text
-l   listening
-n   数字形式
-t   TCP
-p   进程
```

这样可以确认：

> 30000 到底有没有服务监听。

---

# 77. `lsof`

某些服务器安装了：

```bash
lsof -i :30000
```

可以看谁占用了 30000。

---

# 78. 模型服务排错的标准顺序

如果：

```bash
bash launch_welm.sh
```

失败，不要随机试命令。

建议：

```text
① Shell 脚本语法是否正确？
   bash -n launch_welm.sh

② 变量是否正确？
   bash -x launch_welm.sh

③ Python 是哪个？
   which python

④ SGLang 是否安装？
   python -c "import sglang; print(sglang.__file__)"

⑤ 模型路径是否存在？
   ls -lh "$MODEL_PATH"

⑥ 卡是否可见？
   echo "$MLU_VISIBLE_DEVICES"

⑦ 端口是否被占？
   ss -lntp | grep 30000

⑧ 看日志最后几十行
   tail -n 100 server.log

⑨ 搜 Error
   grep -ni "error" server.log
```

---

# 79. 性能脚本排错顺序

如果 `evalscope perf` 失败：

```text
① 服务是否还活着？
② URL 是否一致？
③ 端口是否一致？
④ endpoint 是 /v1/chat/completions 还是 /v1/completions？
⑤ model/name 参数是否匹配？
⑥ tokenizer 路径是否存在？
⑦ 输入长度是否超过 context length？
⑧ parallel/rate 是否过高？
⑨ outputs-dir 是否可写？
```

---

# 80. `chmod +x`

让脚本可直接执行：

```bash
chmod +x launch_welm.sh
```

之后：

```bash
./launch_welm.sh
```

而不是：

```bash
bash launch_welm.sh
```

前提是第一行 shebang 正确：

```bash
#!/usr/bin/env bash
```

---

# 81. `source`

假设：

```bash
env.sh
```

内容：

```bash
export PORT=30000
export MODEL_PATH=/data/models/WeLM
```

如果：

```bash
bash env.sh
```

变量只存在于子 Shell，脚本结束后通常不会留在当前 Shell。

如果：

```bash
source env.sh
```

或者：

```bash
. env.sh
```

则变量直接进入当前 Shell。

这是 `source` 最重要的含义。

---

# 82. 临时环境变量

这类写法非常实用：

```bash
PORT=31000 MLU_VISIBLE_DEVICES=0,1,2,3 bash launch_welm.sh graph
```

这些变量只对这次命令和它的子进程生效。

不会永久修改你的环境。

非常适合 A/B 测试。

---

# 83. 查看所有环境变量

```bash
env
```

或者：

```bash
printenv
```

只看一个：

```bash
printenv MLU_VISIBLE_DEVICES
```

---

# 84. `export A=B` 与 `A=B command`

### 持续当前 Shell

```bash
export PORT=30000
bash launch.sh
```

以后当前终端仍然有 `PORT`。

### 只对单次命令

```bash
PORT=30000 bash launch.sh
```

命令结束后，不会因此永久保留。

---

# 85. 路径：绝对路径与相对路径

绝对路径：

```bash
/data/models/WeLM
```

从 `/` 开始。

相对路径：

```bash
scripts/launch.sh
```

相对于当前目录。

因此执行脚本前：

```bash
pwd
```

非常重要。

---

# 86. `$PWD`

当前目录：

```bash
echo "$PWD"
```

---

# 87. 脚本自身所在目录

一个非常实用的写法：

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
```

这样无论从哪里运行脚本，都能得到脚本文件所在目录。

例如：

```bash
CONFIG="$SCRIPT_DIR/config.json"
```

比依赖当前 `pwd` 更稳定。

这是你以后写较正式脚本时很值得掌握的技巧。

---

# 88. `dirname`

例如：

```bash
dirname /workspace/scripts/launch.sh
```

结果：

```text
/workspace/scripts
```

---

# 89. `basename`

```bash
basename /workspace/scripts/launch.sh
```

结果：

```text
launch.sh
```

---

# 90. `read`

接收用户输入：

```bash
read -r name
echo "$name"
```

或者：

```bash
read -r -p "Continue? [y/N] " answer
```

不过自动化性能测试脚本一般不要太多交互，否则不方便无人值守执行。

---

# 91. Here Document

服务器脚本里偶尔会见到：

```bash
cat <<EOF
model=$MODEL_PATH
port=$PORT
EOF
```

输出多行文本。

也可以给命令传多行输入。

---

# 92. 常用 Git + Shell 组合

查看状态：

```bash
git status
```

看修改：

```bash
git diff
```

看当前 commit：

```bash
git rev-parse HEAD
```

你做性能对比时，非常建议脚本保存：

```bash
COMMIT=$(git rev-parse HEAD)
echo "commit=$COMMIT"
```

这样以后不会忘记：

> 这一批结果到底对应哪个版本代码。

---

# 93. 在测试脚本中记录 Git Commit

例如：

```bash
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)

echo "git_commit=$GIT_COMMIT"
```

还可以写进日志：

```bash
echo "git_commit=$GIT_COMMIT" | tee -a "$LOG_FILE"
```

对于你现在经常做的：

- 修复前
- 修复后
- MTP on/off
- graph/eager

A/B 测试非常重要。

---

# 94. 推荐的测试元信息

性能脚本启动时建议至少打印：

```text
git commit
model
device
visible devices
service port
graph/eager
MTP on/off
input length
output length
request number
parallel
rate
timestamp
output directory
```

否则一个月后很容易无法复现实验。

---

# 95. 更规范的输出目录

例如：

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="outputs/${RUN_ID}"
mkdir -p "$OUTPUT_DIR"
```

甚至：

```bash
OUTPUT_DIR="outputs/${MODE}_${RUN_ID}"
```

避免不同测试覆盖。

---

# 96. 常见 Shell 错误 1：变量未加引号

危险：

```bash
rm -rf $OUTPUT_DIR/*
```

如果：

```text
OUTPUT_DIR
```

为空，后果可能非常严重。

更安全：

```bash
[[ -n "${OUTPUT_DIR:-}" ]] || exit 1
rm -rf "${OUTPUT_DIR:?}/"*
```

所以 `set -u` 和引号很重要。

---

# 97. 常见错误 2：等号旁边有空格

错误：

```bash
PORT = 30000
```

Shell 会尝试运行命令：

```text
PORT
```

正确：

```bash
PORT=30000
```

---

# 98. 常见错误 3：多行续行符后面有东西

正确：

```bash
command \
  --a 1 \
  --b 2
```

危险：

```bash
command \ 
  --a 1
```

注意反斜杠后如果存在空格，续行可能失效。

---

# 99. 常见错误 4：端口不一致

服务：

```bash
PORT=30000
```

测试：

```bash
--url http://127.0.0.1:31000
```

必然连不到正确服务。

建议统一变量：

```bash
SERVER_PORT="${SERVER_PORT:-30000}"
SERVER_URL="http://127.0.0.1:${SERVER_PORT}/v1/chat/completions"
```

---

# 100. 常见错误 5：修改代码但 Python 仍使用旧安装

当仓库使用：

```bash
pip install .
```

可能安装了一份复制版本。

而：

```bash
pip install -e .
```

是 editable install，通常让 Python 直接引用当前仓库。

排查：

```bash
python -c "import sglang; print(sglang.__file__)"
```

看实际 import 到哪里。

---

# 101. 常见错误 6：切 Git commit 后依赖不匹配

你经常会：

```bash
git checkout <old_commit>
```

然后重新启动服务。

但不同 commit 之间可能：

- Python API 改了；
- 依赖版本改了；
- C++/MLU 扩展改了；
- 编译产物不兼容；
- 安装脚本改了。

因此代码回退后，必要时需要重新：

```bash
pip install -e .
```

甚至重新安装扩展或依赖。

这不是 Git 本身的问题，而是：

> **代码版本和运行环境必须匹配。**

---

# 102. `set -e` 不是万能的

虽然：

```bash
set -e
```

很好用，但不要理解成：

> 任何非 0 都一定立即退出。

Shell 在：

- `if` 条件
- `while` 条件
- `&&`
- `||`
- 某些 pipeline

中的行为有细节。

初学阶段只需要牢记：

> `set -euo pipefail` 能显著减少“前面已经失败但脚本还傻傻继续跑”的情况。

---

# 103. 推荐脚本头

你以后自己写 Bash，建议：

```bash
#!/usr/bin/env bash
set -euo pipefail
```

如果需要调试，再临时加：

```bash
set -x
```

正式提交前可以删除 `set -x`，避免日志太吵。

---

# 104. ShellCheck

如果公司环境允许安装，ShellCheck 是非常好用的静态检查工具。

执行：

```bash
shellcheck script.sh
```

可以发现：

- 没加引号；
- 可疑变量展开；
- 旧式反引号；
- 不可靠的 `for x in $(...)`；
- 重定向问题；
- 很多潜在 bug。

即使你是初学者，它也非常适合辅助学习。

---

# 105. 一个完整的服务器工作流示例

假设你要测试 WeLM。

---

## 第一步：进入服务器

```bash
ssh mahao@SERVER_IP
```

---

## 第二步：启动/进入容器

```bash
bash start_docker.sh
```

脚本内部：

```text
检查容器
↓
没有 → docker run
有 → docker start
↓
docker exec
```

---

## 第三步：进入代码

```bash
cd /workspace/sglang-mlu
```

---

## 第四步：确认代码版本

```bash
git status
git branch --show-current
git rev-parse HEAD
```

---

## 第五步：确认 Python 环境

```bash
which python
python -V
python -c "import sglang; print(sglang.__file__)"
```

---

## 第六步：启动服务

```bash
bash launch_welm.sh graph
```

或者：

```bash
MLU_VISIBLE_DEVICES=4,5,6,7 \
PORT=30000 \
bash launch_welm.sh graph
```

---

## 第七步：确认端口

另一个终端：

```bash
ss -lntp | grep 30000
```

---

## 第八步：跑 GSM8K

```bash
bash bench_gsm8k.sh
```

---

## 第九步：跑性能

```bash
bash bench_client.sh
```

---

## 第十步：查输出

```bash
find . -type f | grep -E "json|csv|log|result|output"
```

---

# 106. 阅读任何 Shell 脚本的“七问法”

以后拿到陌生脚本，先问：

### 1. 它最终执行哪个主程序？

找：

```text
python
docker
evalscope
git
cmake
make
```

---

### 2. 输入在哪里？

找：

```bash
$1
$2
${VAR:-default}
read
```

---

### 3. 哪些变量可以修改？

找：

```bash
MODEL_PATH=
PORT=
DEVICE=
OUTPUT_DIR=
```

---

### 4. 是否有环境变量？

找：

```bash
export
```

---

### 5. 有没有分支？

找：

```bash
if
case
```

---

### 6. 有没有循环？

找：

```bash
for
while
```

---

### 7. 结果写到哪里？

找：

```bash
--output
--outputs-dir
--work-dir
>
>>
tee
```

只要这 7 个问题能回答，通常你已经理解了脚本 70%～80%。

---

# 107. 高频符号速查表

| 符号 | 作用 |
|---|---|
| `#` | 注释 |
| `#!` | shebang 开头 |
| `$VAR` | 读取变量 |
| `${VAR}` | 更明确地读取变量 |
| `${VAR:-x}` | 未设置时使用默认值 x |
| `$1` | 第一个脚本参数 |
| `$0` | 脚本名 |
| `$?` | 上一个命令退出码 |
| `$(cmd)` | 执行命令并获取输出 |
| `$((...))` | 整数运算 |
| `"` | 双引号，允许变量展开 |
| `'` | 单引号，基本按原样 |
| `\` | 转义 / 行末续行 |
| `|` | 管道 |
| `>` | 覆盖重定向 |
| `>>` | 追加重定向 |
| `2>` | stderr 重定向 |
| `2>&1` | stderr 合并到 stdout |
| `&` | 后台运行 |
| `&&` | 前一个成功才继续 |
| `||` | 前一个失败才继续 |
| `;` | 命令分隔 |
| `()` | 数组或子 Shell 等语法环境 |
| `[]` | test 条件 |
| `[[ ]]` | Bash 增强条件判断 |
| `${ARR[@]}` | 数组所有元素 |
| `${!ARR[@]}` | 关联数组所有 key |

---

# 108. 高频命令速查表

| 命令 | 常见用途 |
|---|---|
| `pwd` | 当前目录 |
| `ls -lah` | 查看文件 |
| `cd` | 切目录 |
| `mkdir -p` | 创建目录 |
| `cp` | 复制 |
| `mv` | 移动/改名 |
| `rm` | 删除 |
| `cat` | 查看短文件 |
| `less` | 查看长文件 |
| `head` | 看开头 |
| `tail -f` | 实时看日志 |
| `grep` | 搜文本 |
| `find` | 找文件 |
| `wc -l` | 统计行数 |
| `sort` | 排序 |
| `tr` | 字符替换 |
| `sed` | 文本替换/处理 |
| `awk` | 按列和规则处理文本 |
| `ps -ef` | 查看进程 |
| `kill` | 结束进程 |
| `ss -lntp` | 查看监听端口 |
| `curl` | 请求 HTTP 接口 |
| `which` | 查看命令路径 |
| `env` | 查看环境变量 |
| `source` | 在当前 Shell 执行脚本 |
| `chmod +x` | 增加执行权限 |
| `bash -n` | 检查 Shell 语法 |
| `bash -x` | 跟踪执行 |
| `docker ps` | 查看运行中容器 |
| `docker ps -a` | 查看所有容器 |
| `docker images` | 查看镜像 |
| `docker run` | 新建并启动容器 |
| `docker start` | 启动已有容器 |
| `docker exec -it` | 进入运行中容器 |
| `tmux` | 保持远程会话 |
| `git status` | Git 状态 |
| `git diff` | 查看代码修改 |
| `git rev-parse HEAD` | 当前 commit |

---

# 109. 对你当前阶段最应该掌握的内容

不需要一下子把 Bash 所有高级语法学完。

结合你现在的大模型推理/性能测试工作，优先掌握：

```text
第一优先级
├── 变量
├── 环境变量 export
├── 引号
├── ${VAR:-default}
├── $1 / $0
├── \
├── 管道 |
├── > / >> / 2>&1
├── tee
└── exit code

第二优先级
├── if
├── case
├── for
├── 数组
├── ${ARGS[@]}
├── command substitution
└── set -euo pipefail

第三优先级
├── 函数
├── mapfile
├── process substitution
├── sed
├── awk
└── 更复杂的 Bash 自动化
```

---

# 110. 最重要的认知：Shell 脚本不是“神秘代码”

看到：

```bash
exec python -m sglang.launch_server "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
```

初学时会觉得很复杂。

但拆开后只是：

```text
python -m sglang.launch_server
```

启动 Python 模块；

```text
"${ARGS[@]}"
```

把准备好的参数全部传进去；

```text
2>&1
```

错误输出和正常输出合并；

```text
|
```

把输出交给下一个程序；

```text
tee "$LOG_FILE"
```

一边显示，一边保存；

```text
exec
```

让服务进程更直接地接管当前执行环境。

所以以后遇到复杂 Shell，永远记住一个方法：

> **从主命令开始找，然后逐个剥离变量、参数、重定向、管道和控制结构。**

你会发现绝大多数服务器 Shell 脚本，本质上都只是把几十条你本来要手工执行的命令组织得更自动、更稳定、更容易复现。

---

# 111. 你这几类脚本的关系图

```text
start_docker.sh
│
│  创建 / 启动 / 进入容器
↓
Docker 容器
│
├── launch_welm.sh
│      │
│      └── 启动普通 WeLM 服务
│
├── launch_welm_spec.sh
│      │
│      └── 启动带 Speculative/MTP 的 WeLM 服务
│
├── bench_gsm8k.sh
│      │
│      └── 请求模型 API → GSM8K 精度/性能信息
│
└── bench_client.sh
       │
       └── 多组 rate / parallel → 压测模型服务
```

再从数据流看：

```text
模型权重
   ↓
launch_welm*.sh
   ↓
SGLang Server
   ↓
http://127.0.0.1:30000
   ↓
┌──────────────────────┐
│                      │
bench_gsm8k.sh     bench_client.sh
│                      │
GSM8K Eval          Perf Test
│                      │
└──────────┬───────────┘
           ↓
      outputs / logs
```

这就是你目前公司里大量 Shell 脚本的核心使用场景。

---

# 112. 推荐练习

你可以按下面顺序自己动手改一遍。

## 练习 1：变量

写：

```bash
#!/usr/bin/env bash

MODEL=welm
PORT=30000

echo "model=$MODEL"
echo "port=$PORT"
```

---

## 练习 2：默认参数

```bash
MODE="${1:-graph}"
echo "$MODE"
```

分别执行：

```bash
bash test.sh
bash test.sh eager
```

---

## 练习 3：环境覆盖

```bash
PORT="${PORT:-30000}"
echo "$PORT"
```

执行：

```bash
bash test.sh
PORT=31000 bash test.sh
```

---

## 练习 4：case

```bash
MODE="${1:-graph}"

case "$MODE" in
    graph)
        echo "graph mode"
        ;;
    eager)
        echo "eager mode"
        ;;
    *)
        echo "invalid mode"
        exit 1
        ;;
esac
```

---

## 练习 5：数组

```bash
ARGS=(
    --port 30000
    --host 0.0.0.0
)

printf '<%s>\n' "${ARGS[@]}"
```

---

## 练习 6：循环

```bash
for rate in 0.5 1.0 2.0; do
    echo "rate=$rate"
done
```

---

## 练习 7：日志

```bash
LOG="test_$(date +%Y%m%d_%H%M%S).log"

echo "hello" 2>&1 | tee "$LOG"
```

做到这里以后，再回头看你的模型脚本，会明显容易很多。

---

# 113. 一份建议长期保留的 Shell 排错 Checklist

```text
[ ] 当前在哪台服务器？
[ ] 当前在宿主机还是 Docker 容器？
[ ] pwd 是什么？
[ ] 当前 Git 分支是什么？
[ ] 当前 commit 是什么？
[ ] Python 路径是什么？
[ ] Python import 到哪个仓库？
[ ] 模型路径存在吗？
[ ] GPU/MLU 卡号是否正确？
[ ] 端口是否已经被占用？
[ ] Shell 脚本 bash -n 是否通过？
[ ] bash -x 展开的参数是否正确？
[ ] 服务端口和客户端 URL 是否一致？
[ ] 输出目录是否存在且可写？
[ ] 日志保存在哪里？
[ ] 上一个命令 exit code 是多少？
```

---

# 114. 最后总结

对于你当前的工作，Shell 可以浓缩成四件事：

```text
1. 保存配置
2. 组织命令
3. 控制执行流程
4. 自动化重复工作
```

你的启动模型脚本本质是：

```text
准备变量
→ 组装参数
→ 选择运行模式
→ 启动 Python 服务
→ 保存日志
```

你的性能测试脚本本质是：

```text
准备测试参数
→ 循环多组配置
→ 请求模型 API
→ 自动保存结果
```

你的 Docker 脚本本质是：

```text
检查环境
→ 没有容器就创建
→ 有容器就启动
→ 进入容器
```

因此你真正需要掌握的不是背几百个 Linux 命令，而是建立这种拆解能力：

```text
输入
↓
变量
↓
条件 / 循环
↓
主命令
↓
参数
↓
输出 / 日志
```

一旦这个思维建立起来，你以后看到安装脚本、启动脚本、benchmark 脚本、CI 脚本，理解方式其实都高度类似。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-实习工程能力|模块-实习工程能力]]
- 关联阅读：[[outputs/项目整理/专题-06-服务部署评测与协作|专题-06-服务部署评测与协作]]

%% 项目关联导航：结束 %%
