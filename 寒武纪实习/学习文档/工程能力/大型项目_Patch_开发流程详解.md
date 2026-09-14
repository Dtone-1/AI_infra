# 大型项目中的 Patch 开发流程详解

> 适用场景：基于上游开源项目或主干代码做二次开发，但暂时不方便直接把所有改动长期合入上游主分支，需要通过 Patch 保存、分发、安装和维护改动。

---

## 1. Patch 到底是什么

Patch 本质上是一份“代码差异说明”，记录某个基础版本相比修改后版本发生了什么变化：哪些文件改了、哪些行删除、哪些行新增。

```text
原始代码
   ↓
修改代码
   ↓
形成差异
   ↓
Patch 文件
```

最简单的 Patch 可能长这样：

```diff
diff --git a/model.py b/model.py
--- a/model.py
+++ b/model.py
@@ -10,7 +10,7 @@
-if enable_mtp:
-    use_direct_topk = False
+if enable_mtp:
+    use_direct_topk = True
```

它表达的是：相对于某个基线版本，把 `use_direct_topk = False` 改成了 `True`。

可以把 Patch 理解成：

> **不是直接给你一份最终源码，而是告诉你“相对于某个基础版本，我到底改了什么”。**

---

## 2. 为什么大型项目会使用 Patch

大型项目使用 Patch，通常是因为存在“上游代码”和“下游适配代码”之间的关系，例如：

```text
上游项目
SGLang / vLLM / PyTorch / Linux
        │
        │ 固定某个版本
        ↓
下游团队适配
MLU / 特定模型 / 芯片 / 客户需求
        │
        ↓
Patch 1
Patch 2
Patch 3
...
```

最终运行代码可能是：

```text
上游源码
+
Patch 集合
=
最终可运行源码
```

常见目的包括：

- 尽量保持上游源码干净；
- 清楚地区分“上游原始代码”和“团队自己的修改”；
- 同一组修改可以重复应用到不同机器或环境；
- 适合芯片适配、模型适配、客户定制等下游开发；
- 某些修改还没准备好正式合入上游时，可以先用 Patch 维护；
- 便于把修改作为独立文件交付、审核和发布。

---

## 3. 大型 Patch 项目的整体结构

一个典型 Patch 型项目可以理解成：

```text
                   上游仓库
               SGLang v0.5.15
                      │
                      │ 固定基线
                      ↓
             ┌─────────────────┐
             │   原始上游源码  │
             └─────────────────┘
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
      Patch 1      Patch 2      Patch 3
      模型适配      MLU适配      性能优化
          │           │           │
          └───────────┼───────────┘
                      ↓
                 安装脚本
                      ↓
              按顺序应用 Patch
                      ↓
             得到最终运行代码
                      ↓
           build / install / test
                      ↓
               模型服务启动
```

所以 Patch 项目往往不仅有源码，还会有：

```text
project/
├── patches/
│   ├── 0001-add-welm-model.patch
│   ├── 0002-add-mlu-moe.patch
│   └── 0003-fix-mtp-direct-topk.patch
├── scripts/
│   ├── install.sh
│   └── apply_patches.sh
├── requirements/
└── README.md
```

---

## 4. Patch 项目的核心：固定基线版本

Patch 一定是相对于某个基础版本生成的。

例如团队规定：

```text
Upstream: SGLang
Version: v0.5.15
Commit: abc1234
```

那么所有 Patch 都默认建立在这个版本上：

```text
abc1234
   ↓
Patch 1
   ↓
Patch 2
   ↓
Patch 3
```

如果把基础代码换成 `v0.5.18`，原 Patch 不一定还能正常应用，因为原来想修改的代码可能已经被上游改掉。

例如旧版本：

```python
if enable_mtp:
    use_direct_topk = False
```

Patch：

```diff
- use_direct_topk = False
+ use_direct_topk = True
```

但新版本可能已经变成：

```python
if speculative_config is not None:
    enable_direct_route = False
```

这时旧 Patch 很可能报：

```text
patch failed
patch does not apply
```

因此大型 Patch 项目必须明确记录：

```text
基线版本
Commit ID
Patch 顺序
依赖版本
运行环境
```

---

## 5. 大型 Patch 开发的完整生命周期

### 5.1 确定上游基线

```bash
git clone <upstream-repo>
cd sglang
git checkout abc1234
```

意义：所有开发人员从同一份基础代码开始，避免 Patch 上下文不一致。

### 5.2 创建开发分支

即使最终交付的是 Patch，开发阶段仍然建议使用普通 Git 分支：

```bash
git switch -c welm_mtp_fix
```

### 5.3 正常开发和测试

```bash
git status
git diff

# 修改代码
vim ...

git add ...
git commit -m "[WeLM]: fix MTP direct topk route"
```

然后完成：

```text
功能测试
精度测试
性能测试
单元测试
启动服务验证
```

确认修改有效后，再生成 Patch。

---

## 6. 两种最常见的 Patch 形式

大型项目里常见两类：

```text
A. git diff + git apply
B. git format-patch + git am
```

它们的区别非常重要。

---

## 7. 方式一：git diff + git apply

这种方式主要保存：

> **代码差异**

如果当前工作区有修改但还没 Commit：

```bash
git diff
```

可以直接生成：

```bash
git diff > welm_mtp_fix.patch
```

以后在相同基线代码上：

```bash
git apply welm_mtp_fix.patch
```

即可把修改重新应用回来。

### 7.1 应用前检查

```bash
git apply --check welm_mtp_fix.patch
```

没有报错通常表示可以应用。

然后：

```bash
git apply welm_mtp_fix.patch
```

查看结果：

```bash
git diff
```

注意：

```bash
git apply xxx.patch
```

只会把代码修改放到工作区，不会自动生成 Commit。

如果需要提交：

```bash
git add .
git commit -m "Apply WeLM MTP patch"
```

---

## 8. 方式二：git format-patch + git am

这种方式保存的是：

> **完整 Commit**

包括：

```text
Commit message
作者
时间
修改内容
Commit 顺序
```

假设已经有：

```bash
git log --oneline
```

输出：

```text
a1b2c3d [WeLM]: fix MTP direct topk route
```

生成 Patch：

```bash
git format-patch -1 a1b2c3d
```

会得到类似：

```text
0001-WeLM-fix-MTP-direct-topk-route.patch
```

别人应用：

```bash
git am 0001-WeLM-fix-MTP-direct-topk-route.patch
```

Git 会自动：

```text
读取 Patch
↓
应用代码
↓
恢复 Commit message
↓
恢复作者信息
↓
生成 Commit
```

所以可以记成：

```text
git apply = 把代码修改打进工作区
git am    = 把完整 Commit 通过 Patch 搬进来
```

---

## 9. 多个 Patch：Patch Stack / Patch Queue

大型项目中经常维护一整组 Patch：

```text
0001-add-welm-model.patch
0002-add-mlu-moe.patch
0003-add-mtp-support.patch
0004-optimize-direct-topk.patch
0005-fix-graph-mode.patch
```

它们可能有顺序依赖：

```text
Patch 1
   ↓
Patch 2 依赖 Patch 1
   ↓
Patch 3 依赖前两个
   ↓
Patch 4
```

这就叫 Patch Stack 或 Patch Queue。

生成最近 5 个 Commit：

```bash
git format-patch -5
```

指定范围：

```bash
git format-patch <base_commit>..HEAD
```

应用：

```bash
git am 0001-*.patch
git am 0002-*.patch
git am 0003-*.patch
```

或者：

```bash
git am patches/*.patch
```

前提是编号和 shell 展开顺序正确。

---

## 10. 为什么 Patch 文件常用 0001、0002 编号

编号就是为了明确应用顺序：

```text
0001 → 0002 → 0003 → 0004
```

因为后面的 Patch 可能依赖前面的 Patch 新增的文件、接口或逻辑。

如果顺序错了，后续 Patch 可能直接失败。

---

## 11. 安装脚本在 Patch 项目中的作用

大型项目不会要求每个人手工执行十几次 `git apply`，一般会把流程写进脚本。

例如：

```bash
#!/usr/bin/env bash
set -e

git checkout abc1234

git apply patches/0001-add-welm-model.patch
git apply patches/0002-add-mlu-moe.patch
git apply patches/0003-fix-mtp.patch

pip install -e .
```

使用者只需要：

```bash
bash scripts/install.sh
```

安装脚本负责：

```text
确认版本
↓
应用 Patch
↓
安装依赖
↓
编译扩展
↓
pip install
```

所以安装脚本本质上是在自动完成：

> **基础源码 + Patch + 依赖 → 最终可运行环境。**

---

## 12. Patch 项目中的 Review

Patch 本身就是文本差异，因此天然适合 Review。

查看内容：

```bash
cat xxx.patch
```

查看涉及哪些文件：

```bash
git apply --stat xxx.patch
```

查看每个文件新增、删除多少行：

```bash
git apply --numstat xxx.patch
```

`git format-patch` 生成的 Patch 里还能看到：

```text
From
Date
Subject
Commit message
Diff
```

所以 Patch 同时是：

```text
代码修改载体
+
Review 材料
```

---

## 13. Patch 应用失败是什么意思

最常见原因：

```text
1. 基线版本不一致
2. 上游代码已经修改
3. 文件路径变化
4. 前置 Patch 没有应用
5. Patch 顺序错误
6. 同一区域已经被别的修改改过
```

检查：

```bash
git apply --check xxx.patch
```

如果看到：

```text
error: patch failed: model.py:120
error: model.py: patch does not apply
```

意思是：Patch 想在 `model.py` 某个上下文位置找到原始代码，但现在找不到了。

---

## 14. git apply 失败后的处理

可以尝试：

```bash
git apply --reject xxx.patch
```

能应用的部分会正常应用，失败的部分会生成：

```text
xxx.rej
```

例如：

```text
model.py.rej
```

开发者需要：

```text
打开 .rej
↓
理解原 Patch 想改什么
↓
结合当前新代码人工修改
↓
重新测试
```

---

## 15. git am 冲突处理

如果：

```bash
git am 0003-fix-mtp.patch
```

发生冲突：

```bash
git status
```

手动修改冲突文件后：

```bash
git add <file>
git am --continue
```

放弃整个 `git am`：

```bash
git am --abort
```

跳过当前 Patch：

```bash
git am --skip
```

所以 `git am` 的冲突处理方式很像 `rebase` 和 `cherry-pick`。

---

## 16. 如何撤销 Patch

如果 Patch 是通过 `git apply` 打进去且还没 Commit，可以反向应用：

```bash
git apply -R xxx.patch
```

`-R` 表示 Reverse。

例如原 Patch：

```diff
- False
+ True
```

反向应用后就是：

```text
True → False
```

如果已经形成 Commit，则更适合：

```bash
git revert <commit>
```

私人本地历史里也可以根据情况使用：

```bash
git reset
```

---

## 17. 大型 Patch 项目如何升级上游版本

假设旧版本：

```text
SGLang v0.5.15
+
10 个 Patch
```

现在升级到：

```text
SGLang v0.5.18
```

典型流程：

```text
① 获取 v0.5.18
↓
② 从干净新基线开始
↓
③ 按顺序重新应用全部 Patch
↓
④ 能自动应用的直接通过
↓
⑤ 不能应用的人工解决
↓
⑥ 重新生成 / refresh Patch
↓
⑦ 全量功能、精度、性能测试
↓
⑧ 发布新的 Patch Stack
```

这实际上是在做：

> **把所有下游修改重新移植到新的上游版本。**

常见叫法包括：

```text
Port patches
Refresh patches
Rebase downstream patches
```

---

## 18. 为什么 Patch 项目的维护成本会越来越高

如果长期存在：

```text
上游不断升级
+
下游 Patch 越来越多
```

可能从：

```text
10 个 Patch
↓
30 个 Patch
↓
80 个 Patch
```

每次上游升级都要逐个检查、解决冲突、重新测试，而且 Patch 之间还可能有依赖。

这会形成所谓的：

```text
Downstream Patch Debt
```

可以理解成：

> **下游补丁技术债。**

---

## 19. Patch 和普通分支开发的区别

Patch 模式：

```text
上游仓库
+
Patch Stack
=
最终代码
```

普通 Git 分支模式：

```text
上游基线
↓
项目稳定分支
↓
Feature Branch
↓
Commit
↓
MR / PR
↓
Merge / Cherry-pick
```

Patch 更强调：

```text
“相对于上游，我改了什么”
```

普通分支开发更强调：

```text
“这个项目当前完整历史是什么”
```

---

## 20. Patch 和 Cherry-pick 的区别

Cherry-pick：

```bash
git cherry-pick <commit-id>
```

要求当前 Git 能访问那个 Commit。

而 Patch 可以：

```text
机器 A
生成 patch 文件
↓
scp / 邮件 / 制品库
↓
机器 B
git am
```

所以可以把：

```text
git format-patch + git am
```

理解成：

> **“通过文件传输 Commit”。**

而：

```text
git cherry-pick
```

则是：

> **“直接通过 Git 对象复制某个 Commit 的修改”。**

---

## 21. 大型 Patch 项目的推荐规范

应该明确维护：

```text
1. Patch 基于哪个 Commit
2. Patch 应用顺序
3. 每个 Patch 的功能说明
4. Patch 的负责人
5. Patch 是否已经被上游合入
6. Patch 对应哪些测试
7. Patch 是否仍需要保留
```

最好维护一个：

```text
patches/README.md
```

例如：

```text
Base Commit:
13fdfe0

Patch List:
0001 add WeLM model
0002 add MLU MoE
0003 add MTP
0004 optimize direct topk
```

---

# 22. 实际项目示例：SGLang + WeLM + MLU Patch 开发

下面用一个接近真实工程的例子说明完整流程。

假设：

```text
上游项目：SGLang
基础版本：v0.5.15
目标：增加 WeLM MLU 支持
```

最终希望：

```text
官方 SGLang v0.5.15
+
WeLM 模型 Patch
+
MLU MoE Patch
+
MTP Patch
=
可以在 MLU 上运行的 WeLM
```

---

## 22.1 Clone 并固定基线

```bash
git clone ssh://git@example.com/sglang.git
cd sglang

git fetch --all
git checkout v0.5.15
```

记录具体 Commit：

```bash
git rev-parse HEAD
```

假设输出：

```text
13fdfe0f2a2e79c0d0a78a62811dfxxx
```

以后就可以把：

```text
13fdfe0
```

作为 Patch 基线。

---

## 22.2 创建开发分支

```bash
git switch -c welm_mlu_dev
```

---

## 22.3 修改代码

例如修改：

```text
python/sglang/srt/models/welm.py
python/sglang/srt/layers/moe.py
```

查看：

```bash
git status
git diff
```

---

## 22.4 提交第一个功能

```bash
git add python/sglang/srt/models/welm.py

git commit -m "[WeLM]: add model support"
```

---

## 22.5 提交第二个功能

```bash
git add python/sglang/srt/layers/moe.py

git commit -m "[WeLM][MLU]: add MoE support"
```

---

## 22.6 提交第三个功能

修改 MTP direct topk 后：

```bash
git add python/sglang/srt/models/welm.py

git commit -m "[WeLM][MTP]: enable direct topk MoE route"
```

此时：

```bash
git log --oneline
```

可能看到：

```text
c3c3c3c [WeLM][MTP]: enable direct topk MoE route
b2b2b2b [WeLM][MLU]: add MoE support
a1a1a1a [WeLM]: add model support
13fdfe0 upstream v0.5.15 base
```

---

## 23. 生成整套 Patch

从基线到当前 HEAD：

```bash
git format-patch 13fdfe0..HEAD
```

生成：

```text
0001-WeLM-add-model-support.patch
0002-WeLM-MLU-add-MoE-support.patch
0003-WeLM-MTP-enable-direct-topk-MoE-route.patch
```

然后放进：

```text
patches/
```

---

## 24. 在干净环境应用 Patch

重新获取一份干净源码：

```bash
git clone ssh://git@example.com/sglang.git sglang_clean
cd sglang_clean

git checkout 13fdfe0
```

确认工作区干净：

```bash
git status
```

应用：

```bash
git am /path/to/patches/0001-WeLM-add-model-support.patch
git am /path/to/patches/0002-WeLM-MLU-add-MoE-support.patch
git am /path/to/patches/0003-WeLM-MTP-enable-direct-topk-MoE-route.patch
```

也可以：

```bash
git am /path/to/patches/*.patch
```

应用后：

```bash
git log --oneline
```

会看到三条新的 Commit 接在 `13fdfe0` 后面。

接着：

```bash
pip install -e .
```

启动服务：

```bash
bash launch_welm.sh
```

再进行：

```text
精度测试
性能测试
MTP 测试
CNPerf / Profile
```

---

## 25. 如果只想生成一份“总代码差异 Patch”

开发机：

```bash
git diff 13fdfe0 HEAD > welm_all.patch
```

目标机：

```bash
git checkout 13fdfe0
git apply --check welm_all.patch
git apply welm_all.patch
```

确认：

```bash
git diff
```

然后再：

```bash
git add .
git commit -m "Apply WeLM patches"
```

这种方式更简单，但会丢失原本每个独立 Commit 的结构。

---

## 26. 自动安装脚本示例

```bash
#!/usr/bin/env bash
set -euo pipefail

BASE_COMMIT="13fdfe0"
PATCH_DIR="$(pwd)/patches"

git reset --hard
git clean -fd

git checkout "${BASE_COMMIT}"

git am "${PATCH_DIR}/0001-WeLM-add-model-support.patch"
git am "${PATCH_DIR}/0002-WeLM-MLU-add-MoE-support.patch"
git am "${PATCH_DIR}/0003-WeLM-MTP-enable-direct-topk-MoE-route.patch"

pip install -e .

echo "Patch installation completed."
```

用户以后只需要：

```bash
bash install_welm.sh
```

就能得到：

```text
固定基础代码
+
固定 Patch Stack
+
固定安装方式
=
可复现环境
```

> 注意：`git reset --hard` 和 `git clean -fd` 会清理本地修改和未跟踪文件，真实生产脚本使用前必须确认这是团队设计好的行为。

---

## 27. 上游升级到 v0.5.18 时怎么办

获取新版本：

```bash
git fetch origin
git switch -c welm_v0.5.18 origin/v0.5.18
```

逐个尝试旧 Patch：

```bash
git am patches/0001-*.patch
git am patches/0002-*.patch
git am patches/0003-*.patch
```

如果第三个冲突：

```bash
git status
```

人工修改：

```bash
vim python/sglang/srt/models/welm.py
```

解决后：

```bash
git add python/sglang/srt/models/welm.py
git am --continue
```

全部 Patch 适配完成并测试通过后，重新生成新版 Patch：

```bash
git format-patch <new-base-commit>..HEAD
```

这就叫：

> **把旧 Patch Stack 移植 / refresh 到新的上游基线。**

---

## 28. 为什么团队后来可能会放弃 Patch 模式

如果团队宣布：

```text
不再使用 Patch
与主分支 v0.5.18 对齐
```

通常意味着要从：

```text
上游代码
+
Patch Stack
=
最终代码
```

切换为：

```text
主线新基线
↓
稳定项目分支
↓
Feature Branch
↓
MR
↓
Cherry-pick / Merge
```

好处包括：

```text
Git 历史更清楚
Review 更方便
Cherry-pick 更直接
减少 Patch 冲突
版本升级成本下降
代码来源更容易追踪
```

并且可以直接利用：

```bash
git log
git blame
git diff
git rebase
git cherry-pick
```

来追踪代码演进。

---

## 29. 最后总结

Patch 项目的核心思想可以浓缩成：

```text
固定一个上游基线
        ↓
开发下游修改
        ↓
把修改保存成 Patch
        ↓
Patch 按顺序维护
        ↓
安装时重新应用 Patch
        ↓
生成最终可运行代码
        ↓
完整测试
```

最重要的两套命令：

### 代码差异型 Patch

```bash
git diff > xxx.patch
git apply --check xxx.patch
git apply xxx.patch
```

### Commit 型 Patch

```bash
git format-patch <base>..HEAD
git am xxx.patch
```

可以直接记住：

> **`git apply` 是“把代码差异打进去”，`git am` 是“把完整 Commit 通过 Patch 文件搬进来”。**

大型 Patch 项目真正难的地方不在“怎么执行 `git apply`”，而在于：

> **如何固定基线、维护 Patch 顺序、控制依赖、处理上游升级、解决冲突、重新验证功能/精度/性能，并确保任何人都能通过同一套脚本稳定复现最终代码。**


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-实习工程能力|模块-实习工程能力]]
- 关联阅读：[[outputs/项目整理/专题-06-服务部署评测与协作|专题-06-服务部署评测与协作]]

%% 项目关联导航：结束 %%
