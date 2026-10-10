# Git 与 GitHub 操作指南

这份文档说明：本项目的版本控制是怎么做的、还需要你做什么才能推到自己的 GitHub。

---

## 一、本地已经做好的部分

项目根目录 `ai-lab/` 已经是一个独立的 Git 仓库：

- 分支：`main`
- 忽略规则：`.gitignore`（排除 `runs/`、`datasets/`、`*.pt`、`*.onnx`、缓存目录等）
- 换行策略：`.gitattributes`（统一 LF，避免跨系统出现整文件差异）

当前提交历史（按开发顺序）：

```
chore: 初始化项目结构与说明文档
feat(llm): 实现大模型接入层与命令行客户端
feat(agent): 实现轻量智能体与工具调用
chore: 统一换行符为 LF
feat(server): 实现零依赖 HTTP 服务与检测后端
feat(web): 实现对话与目标检测界面
feat(yolo): 提供训练、推理、实时与导出脚本
feat(desktop): 提供 Edge 应用模式与 Electron 桌面外壳
test: 增加端到端冒烟测试与示例图生成脚本
docs: 补充工程日志、实现思路与部署记录
```

每次提交只做一件事，看提交信息就能知道改了什么，不用去翻 diff。

---

## 二、需要你做的：改成本人的提交身份

环境里没有你的 Git 身份信息，我暂时用的是占位值。**推之前先改成你自己的**：

```powershell
cd D:\tools\codex\ai-lab
git config user.name "你的名字"
git config user.email "你的GitHub邮箱"
```

> 如果希望前面这些提交也换成你的身份，可以执行：
> `git rebase -i --root` 把每条提交标成 `edit` 后逐个
> `git commit --amend --author="你的名字 <你的邮箱>" --no-edit`，
> 或直接用一个新仓库重来一次提交。

---

## 三、推送到自己的 GitHub

### 1. 在 GitHub 上建仓库

打开 <https://github.com/new>：

- Repository name 填 `ai-lab`；
- 可见性按课程要求选择（Private 也能邀请老师查看，Public 更方便检查）；
- **不要**勾选 "Add a README file"，本地已经有了，勾了反而要处理冲突。

### 2. 关联远程仓库并推送

把下面的 `你的用户名` 换成自己的：

```powershell
cd D:\tools\codex\ai-lab
git remote add origin https://github.com/你的用户名/ai-lab.git
git push -u origin main
```

第一次推送会弹出浏览器要求登录 GitHub，按提示授权即可。

### 3. 确认推送成功

```powershell
git remote -v
git log --oneline --decorate -3
git status
```

`git status` 显示 `working tree clean`，且 `git log` 里能看到 `origin/main`，
就说明云端已经同步。

---

## 四、后续每次改动的流程

```powershell
git status                      # 先看改了哪些文件
git diff                        # 再确认改了什么
git add <相关文件>               # 只加这次要提交的部分，别用 git add -A 一锅端
git commit -m "fix(web): 修正检测框缩放偏移"
git push
```

约定：

1. **一次提交只做一件事**，功能、修 bug、改文档不要混在一起；
2. 提交信息写"做了什么"，不写"改了文件"；
3. 推之前先跑 `python tools\smoke_test.py`，避免把坏掉的版本推上去；
4. 训练产物、数据集、模型权重不要提交（`.gitignore` 已经挡住了）。

---

## 五、大文件怎么办

如果以后数据集或权重确实必须进仓库，用 Git LFS，不要硬塞：

```powershell
git lfs install
git lfs track "*.pt"
git lfs track "datasets/**"
git add .gitattributes
git commit -m "chore: 用 Git LFS 管理模型权重与数据集"
```

GitHub 单个文件超过 100 MB 会被直接拒绝，这一点提前注意。
