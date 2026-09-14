# Git 常用命令速查手册（公司项目协作版）

> 适用场景：参与公司项目开发、创建个人分支、提交 Commit、Push、提 MR/PR、同步项目分支、处理冲突、回退代码、Cherry-pick、Rebase 等。

---

## 1. Git 协作流程总览

```text
远程仓库 GitLab / GitHub
        │
        │ clone / fetch / pull
        ↓
本地仓库
        │
        ├─ 项目分支：welm_0.5.15
        └─ 个人分支：mahao_welm_xxx
                    │
                    │ 修改代码
                    ↓
                  工作区
                    │ git add
                    ↓
                  暂存区
                    │ git commit
                    ↓
                 本地 Commit
                    │ git push
                    ↓
                 远程个人分支
                    │
                    ↓
                  MR / PR
                    │
                    ↓
                 项目分支
```

最常见开发链路：

```bash
git fetch origin
git switch welm_0.5.15
git pull
git switch -c mahao_feature_xxx

# 修改代码...

git status
git diff
git add <file>
git diff --cached
git commit -m "[Model][WeLM]: fix xxx"
git push -u origin mahao_feature_xxx
```

之后在 GitLab/GitHub 创建 **MR / PR**，目标分支选择项目分支。

---

## 2. 查看当前仓库状态

| 命令 | 作用 |
|---|---|
| `git status` | 查看当前分支、修改文件、暂存状态 |
| `git branch` | 查看本地分支 |
| `git branch -r` | 查看远程分支 |
| `git branch -a` | 查看本地 + 远程全部分支 |
| `git remote -v` | 查看远程仓库名称和地址 |
| `git log` | 查看提交历史 |
| `git log --oneline` | 简洁查看 Commit |
| `git log --oneline --graph --all` | 图形化查看全部分支历史 |
| `git show <commit>` | 查看某个 Commit 具体修改 |
| `git rev-parse HEAD` | 查看当前完整 Commit ID |
| `git rev-parse --abbrev-ref HEAD` | 查看当前分支名 |

常用：

```bash
git status
git log --oneline -10
git branch -a
```

---

## 3. Clone：第一次获取项目

```bash
git clone <repository-url>
```

示例：

```bash
git clone ssh://git@gitlab.example.com/project/sglang-mlu.git
cd sglang-mlu
```

指定分支：

```bash
git clone -b welm_0.5.15 <repository-url>
```

> `git clone` 一般只在第一次获取仓库时使用，之后更新代码主要使用 `fetch / pull`。

---

## 4. Fetch：获取远程最新信息

```bash
git fetch
git fetch origin
git fetch --all
```

作用：

```text
远程 GitLab 最新状态
        ↓ git fetch
更新本地的 origin/xxx
        ↓
不会直接修改当前工作代码
```

例如：

```bash
git fetch origin
git log --oneline HEAD..origin/welm_0.5.15
```

查看远程项目分支比当前代码多出的 Commit。

查看差异：

```bash
git diff HEAD origin/welm_0.5.15
```

---

## 5. Pull：获取并合入远程代码

```bash
git pull
```

可粗略理解为：

```text
git pull ≈ git fetch + merge/rebase
```

指定远程分支：

```bash
git pull origin welm_0.5.15
```

推荐多人协作时先：

```bash
git fetch origin
```

确认远程变化后，再决定 `merge` 或 `rebase`。

---

## 6. 分支操作

### 查看分支

```bash
git branch
git branch -r
git branch -a
```

### 创建并切换新分支

```bash
git switch -c mahao_feature_xxx
```

旧写法：

```bash
git checkout -b mahao_feature_xxx
```

### 切换分支

```bash
git switch welm_0.5.15
```

旧写法：

```bash
git checkout welm_0.5.15
```

### 删除本地分支

```bash
git branch -d mahao_feature_xxx
```

强制删除：

```bash
git branch -D mahao_feature_xxx
```

### 删除远程分支

```bash
git push origin --delete mahao_feature_xxx
```

### 重命名当前分支

```bash
git branch -m new_branch_name
```

---

## 7. Diff：查看代码修改

### 查看未暂存修改

```bash
git diff
```

### 查看指定文件

```bash
git diff path/to/file.py
```

### 查看已经 `git add` 的修改

```bash
git diff --cached
```

### 比较两个 Commit

```bash
git diff <commit1> <commit2>
```

### 比较两个分支

```bash
git diff branch1 branch2
```

### 查看个人分支相对项目分支新增的修改

```bash
git fetch origin
git diff origin/welm_0.5.15...HEAD
```

查看新增 Commit：

```bash
git log --oneline origin/welm_0.5.15..HEAD
```

---

## 8. Add：加入暂存区

加入单个文件：

```bash
git add path/to/file.py
```

加入多个文件：

```bash
git add file1.py file2.py
```

加入当前目录全部修改：

```bash
git add .
```

推荐提交前检查：

```bash
git status
git diff --cached
```

---

## 9. Commit：提交代码

```bash
git commit -m "Fix MTP direct topk MoE path"
```

公司项目常见格式：

```bash
git commit -m "[Model][WeLM]: fix MTP direct topk route"
```

修改最近一次 Commit 信息：

```bash
git commit --amend
```

直接修改 message：

```bash
git commit --amend -m "new commit message"
```

> `--amend` 会重新生成 Commit ID；如果旧 Commit 已经 Push，后续通常需要重新 Push。

---

## 10. Push：上传到远程仓库

第一次 Push 新分支：

```bash
git push -u origin mahao_feature_xxx
```

之后：

```bash
git push
```

指定：

```bash
git push origin mahao_feature_xxx
```

查看当前跟踪关系：

```bash
git branch -vv
```

如果因为 Rebase / Amend 修改了已经 Push 的 Commit：

```bash
git push --force-with-lease origin mahao_feature_xxx
```

> 推荐 `--force-with-lease`，不要轻易使用裸 `--force`。它会先检查远程分支是否被别人更新，安全性更高。

---

## 11. MR / PR

### 名称

```text
GitLab  → Merge Request（MR）
GitHub  → Pull Request（PR）
```

典型关系：

```text
个人分支 mahao_feature_xxx
             │
             │ MR / PR
             ↓
项目分支 welm_0.5.15
```

MR/PR 本身通常在 GitLab/GitHub 页面创建，不属于纯 Git 命令。

提交 MR/PR 前推荐检查：

```bash
git status
git fetch origin
git log --oneline origin/welm_0.5.15..HEAD
git diff origin/welm_0.5.15...HEAD
```

---

## 12. Merge：合并分支

把指定分支合入当前分支：

```bash
git merge <branch>
```

例如：

```bash
git switch welm_0.5.15
git merge mahao_feature_xxx
```

结果大致：

```text
A---B---C  welm
     \
      D---E  feature

merge 后：

A---B---C------M
     \        /
      D------E
```

---

## 13. Rebase：把自己的提交接到最新项目代码之后

常见公司协作场景：

```bash
git fetch origin
git switch mahao_feature_xxx
git rebase origin/welm_0.5.15
```

原来：

```text
A---B---E---F        项目最新
     \
      C---D          你的提交
```

Rebase 后：

```text
A---B---E---F---C'---D'
```

如果发生冲突：

```bash
# 手动解决冲突
git add <conflict-file>
git rebase --continue
```

放弃 Rebase：

```bash
git rebase --abort
```

Rebase 后如果该分支之前已经 Push：

```bash
git push --force-with-lease
```

---

## 14. Cherry-pick：搬运指定 Commit

```bash
git cherry-pick <commit-id>
```

例如：

```bash
git switch welm_0.5.15
git cherry-pick a63bd2c
```

含义：

> 把 `a63bd2c` 这个 Commit 所包含的修改复制到当前分支，并生成新的 Commit。

多个 Commit：

```bash
git cherry-pick commit1 commit2
```

冲突处理：

```bash
git add <file>
git cherry-pick --continue
```

放弃：

```bash
git cherry-pick --abort
```

---

## 15. Restore：恢复文件修改

### 放弃未暂存修改

```bash
git restore path/to/file.py
```

全部恢复：

```bash
git restore .
```

### 取消暂存，但保留代码修改

```bash
git restore --staged path/to/file.py
```

例如：

```text
git add file.py
      ↓
已经进入暂存区

git restore --staged file.py
      ↓
撤出暂存区，但修改仍然保留
```

---

## 16. Reset：移动分支指针 / 撤销本地提交

### 撤销最近一次 Commit，但保留修改在暂存区

```bash
git reset --soft HEAD~1
```

### 撤销 Commit，并保留代码修改但取消暂存

```bash
git reset HEAD~1
```

等价常见模式：

```bash
git reset --mixed HEAD~1
```

### 完全删除最近 Commit 和代码修改

```bash
git reset --hard HEAD~1
```

切到指定 Commit 状态：

```bash
git reset --hard <commit-id>
```

> `reset --hard` 会丢弃修改；公共分支上不要随意使用，更不要轻易配合强制 Push。

---

## 17. Revert：安全撤销已经提交的 Commit

```bash
git revert <commit-id>
```

例如：

```text
A → B → C
```

执行：

```bash
git revert C
```

变成：

```text
A → B → C → D
```

`D` 是一个“反向撤销 C”的新 Commit。

适合：

- Commit 已经 Push；
- 项目公共分支；
- 希望保留完整历史。

---

## 18. Stash：临时保存未提交修改

暂存当前修改：

```bash
git stash
```

带说明：

```bash
git stash push -m "WIP: MTP optimization"
```

查看：

```bash
git stash list
```

恢复最近一次：

```bash
git stash pop
```

恢复但不删除 Stash：

```bash
git stash apply
```

删除：

```bash
git stash drop
```

常见场景：

```bash
# 当前功能做到一半
git stash

# 临时切其他分支
git switch welm_0.5.15

# 完成后切回来
git switch mahao_feature_xxx
git stash pop
```

---

## 19. Checkout / Detached HEAD：切历史版本测试

查看历史：

```bash
git log --oneline
```

临时切到某个历史 Commit：

```bash
git switch --detach <commit-id>
```

旧写法：

```bash
git checkout <commit-id>
```

适合修复前后性能对比。

测试完成返回：

```bash
git switch mahao_feature_xxx
```

> Detached HEAD 表示当前不处于普通分支上。临时测试没问题，但不要把长期开发留在 Detached HEAD。

---

## 20. 冲突处理

冲突通常发生在：

```text
merge
rebase
cherry-pick
pull
```

查看：

```bash
git status
```

文件中可能出现：

```text
<<<<<<< HEAD
当前分支代码
=======
另一边代码
>>>>>>> branch
```

手动保留正确代码并删除冲突标记，然后：

```bash
git add <file>
```

Merge：

```bash
git commit
```

Rebase：

```bash
git rebase --continue
```

Cherry-pick：

```bash
git cherry-pick --continue
```

放弃：

```bash
git merge --abort
git rebase --abort
git cherry-pick --abort
```

---

## 21. 远程仓库 Remote

查看：

```bash
git remote -v
```

典型：

```text
origin  ssh://git@gitlab.example.com/project.git (fetch)
origin  ssh://git@gitlab.example.com/project.git (push)
```

添加远程：

```bash
git remote add origin <url>
```

修改地址：

```bash
git remote set-url origin <new-url>
```

查看远程详细信息：

```bash
git remote show origin
```

---

## 22. Tag：版本标签

查看：

```bash
git tag
```

创建：

```bash
git tag v1.0.0
```

带说明：

```bash
git tag -a v1.0.0 -m "release v1.0.0"
```

Push：

```bash
git push origin v1.0.0
```

全部 Push：

```bash
git push origin --tags
```

---

## 23. 查找代码历史

查看某文件历史：

```bash
git log -- path/to/file.py
```

查看某文件每一行由谁修改：

```bash
git blame path/to/file.py
```

搜索 Commit message：

```bash
git log --grep="MTP"
```

搜索引入/删除某段文本的 Commit：

```bash
git log -S "direct_topk"
```

查看某 Commit：

```bash
git show <commit-id>
```

---

## 24. Clean：删除未跟踪文件

先预览：

```bash
git clean -n
```

删除未跟踪文件：

```bash
git clean -f
```

删除未跟踪目录：

```bash
git clean -fd
```

> `git clean` 删除后通常无法通过 Git 恢复，执行前先 `git clean -n`。

---

## 25. 常用 HEAD 表达

| 表达 | 含义 |
|---|---|
| `HEAD` | 当前所在 Commit |
| `HEAD~1` | 上一个 Commit |
| `HEAD~2` | 上上个 Commit |
| `origin/main` | 本地记录的远程 `main` |
| `origin/welm_0.5.15` | 本地记录的远程项目分支 |

例如：

```bash
git diff HEAD~1 HEAD
git show HEAD
git reset --soft HEAD~1
```

---

## 26. 公司项目推荐标准流程

### 场景 A：从项目最新代码开始开发

```bash
git fetch origin
git switch welm_0.5.15
git pull
git switch -c mahao_feature_xxx
```

修改完成：

```bash
git status
git diff
git add <files>
git diff --cached
git commit -m "[Model][WeLM]: fix xxx"
git push -u origin mahao_feature_xxx
```

然后提 MR：

```text
mahao_feature_xxx → welm_0.5.15
```

---

### 场景 B：开发过程中项目分支更新了

```bash
git fetch origin
git switch mahao_feature_xxx
git rebase origin/welm_0.5.15
```

解决冲突后：

```bash
git add <files>
git rebase --continue
git push --force-with-lease
```

---

### 场景 C：MR Review 后继续修改

```bash
# 修改代码
git add <files>
git commit -m "Fix review comments"
git push
```

MR 会自动更新，不需要重新创建。

如果要求保持一个 Commit：

```bash
git add <files>
git commit --amend
git push --force-with-lease
```

---

### 场景 D：把个人分支某个 Commit 合到项目分支

```bash
git switch welm_0.5.15
git pull
git cherry-pick <commit-id>
git push origin welm_0.5.15
```

仅在团队流程允许直接 Push 项目分支时使用；否则通常通过 MR/PR 合入。

---

### 场景 E：切换修复前 / 修复后版本做性能测试

```bash
git log --oneline

git switch --detach <old-commit>
# 启动服务、跑测试

git switch --detach <new-commit>
# 启动服务、跑测试

git switch mahao_feature_xxx
```

如果依赖、编译产物或 Python Editable Install 会随 Commit 改动，切 Commit 后可能需要重新安装或重新编译。

---

## 27. 高风险命令速记

执行前重点确认：

```bash
git reset --hard
git clean -fd
git push --force
git branch -D
```

更推荐：

```bash
git push --force-with-lease
```

公共项目分支上优先：

```bash
git revert <commit>
```

而不是直接改写历史。

---

## 28. 最值得熟练掌握的 25 条命令

```bash
git status
git branch
git branch -a
git switch <branch>
git switch -c <new-branch>

git fetch origin
git pull
git remote -v

git diff
git diff --cached
git log --oneline
git show <commit>

git add <file>
git add .
git commit -m "message"
git commit --amend

git push
git push -u origin <branch>
git push --force-with-lease

git rebase origin/<branch>
git merge <branch>
git cherry-pick <commit>

git restore <file>
git revert <commit>
git stash
```

---

## 29. 一句话区分最容易混淆的命令

| 命令 | 一句话理解 |
|---|---|
| `fetch` | 看远程最新发生了什么，不直接改当前代码 |
| `pull` | 把远程最新代码拿下来并合入当前分支 |
| `add` | 选择哪些修改准备提交 |
| `commit` | 在本地生成一次版本快照 |
| `push` | 把本地 Commit 上传远程 |
| `merge` | 把两个分支历史合起来 |
| `rebase` | 把自己的 Commit 重新接到最新基线后 |
| `cherry-pick` | 单独复制某个 Commit 到当前分支 |
| `restore` | 恢复文件 / 取消暂存 |
| `reset` | 移动分支历史位置 |
| `revert` | 新建 Commit 来撤销旧 Commit |
| `stash` | 临时保存没提交的修改 |
| `MR / PR` | 请求把个人分支代码合入目标分支 |

---

## 30. 推荐工作习惯

```bash
# 开始工作
git status
git fetch origin

# 修改完成
git status
git diff

# 提交前
git add <files>
git diff --cached

# 提交后
git log --oneline -5

# Push 前
git status
```

核心原则：

> **先看状态，再操作；先 Fetch，再判断；Commit 前看 Diff；公共历史尽量不要强制改写。**


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-实习工程能力|模块-实习工程能力]]
- 关联阅读：[[outputs/项目整理/专题-06-服务部署评测与协作|专题-06-服务部署评测与协作]]

%% 项目关联导航：结束 %%
