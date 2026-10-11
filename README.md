# AI Lab：本地大模型 + YOLO 一体化应用

在一台 8G 显存的 RTX 4060 上，**本地部署 Qwen2.5 做对话、自己训练 YOLO 做目标检测**，
用一个 Web 应用把两者串起来；同一套后端还提供 CLI 和桌面形态。

主应用**零第三方依赖**（只用 Python 标准库），克隆下来就能跑起来。

## 文档

| 文档 | 内容 |
| --- | --- |
| [作业一：大模型](作业一-大模型.md) | 本地部署、显存实测、智能体、通过 API 接入应用 |
| [作业二：YOLO](作业二-YOLO.md) | 训练、实时推理、接入应用、局限说明 |
| [工作日志](工作日志.md) | 每个阶段卡在哪、当时怎么判断的 |

---

## 快速开始

前置条件：装好 [Ollama](https://ollama.com/download) 并拉取模型

```powershell
ollama pull qwen2.5:1.5b
```

然后双击项目根目录的 **`start-server.cmd`**，浏览器打开 <http://127.0.0.1:8000/>。

等价的手工命令（想看清楚参数就用这条）：

```powershell
python run_server.py --provider ollama --model qwen2.5:1.5b `
  --vision-backend ultralytics `
  --vision-weights runs\train\coco8-ft\weights\best.pt --vision-conf 0.45
```

> `runs\train\coco8-ft\weights\best.pt` 是训练产物，**不在仓库里**（权重文件被忽略）。
> 没有它也能启动——检测面板会提示"未就绪"，按 [作业二](作业二-YOLO.md) 里的步骤跑一次训练即可。

命令行版和桌面版：

```powershell
python run_cli.py                 # 命令行对话
python desktop\run_desktop.py     # 桌面窗口（用 Edge/Chrome 应用模式）
```

---

## 对应作业要求

### 一、大模型

| 要求 | 实现 |
| --- | --- |
| 1. 完成本地大模型部署 | Ollama + `qwen2.5:1.5b`（Q4_K_M），实测占显存 1166 MB、全部在 GPU 上；详见 [作业一](作业一-大模型.md) |
| 2. 搭建自己的智能体 | `app/agent/`：意图识别 → 执行工具 → 结果交回模型；同时支持把 Dify / 扣子等平台的 OpenAI 兼容接口当模型后端 |
| 3. 作为 model provider 通过 API 接入应用 | `POST /api/chat`（SSE 流式） |
| 4. 应用形式 | CLI（`run_cli.py`）、Web（`web/`）、桌面（`desktop/`，Edge 应用模式 + Electron 两种） |

### 二、YOLO

| 要求 | 实现 |
| --- | --- |
| 1. 完成一次 ultralytics 训练 | 两轮：合成数据从零训练（mAP50 0.995）+ 真实 COCO8 微调（mAP50 0.872），均在 RTX 4060 上完成 |
| 2. 实时推理部署 | `yolo/realtime.py` 实测 **49.5 FPS**（单帧 18.4 ms）；网页端也支持摄像头实时检测；另有独立推理服务 `yolo/serve.py` |
| 3. 接入应用 | 应用加载**自己微调的模型**，真实街景照片上识别出车辆、电动车、行人 |

---

## 设计思路

### 1. 先想清楚"哪些东西会变"

会变的只有四件事：用哪个大模型、模型跑在哪、YOLO 用什么权重、应用长什么样。
所以按"会变的东西"切层，而不是按技术栈切：

```
界面层   CLI | Web | 桌面
协议层   HTTP + JSON + SSE        ← 唯一固定的接口
能力层   对话服务 | 检测服务 | 智能体
后端层   模型后端可插拔 | 检测后端可插拔
```

换模型、换推理位置都只发生在最底层，上面三层不用动。

### 2. 为什么主应用零依赖

这台机器装不上第三方包（访问不了软件源）。如果主应用依赖 FastAPI，
"打开就能看"就变成"先想办法装上 FastAPI"。改成标准库之后，
代价是自己实现 HTTP 服务、SSE 分块输出和 multipart 解析，
换来的是任何装了 Python 的机器都能跑起来。

**边界要说清楚**：零依赖只约束主应用。YOLO 那边该装 ultralytics 就装，
不为了"零依赖"去手写检测器。

### 3. 智能体为什么不用原生 function calling

本机是 2B 级小模型，实测原生 tool calling 有两个现实问题：很多小模型不支持；
支持的也常把 JSON 格式写错，一处出错整条链路就断。

所以改成**两段式**：本地规则识别意图 → 执行工具 → 把结果作为系统消息交回模型，
让模型只负责"把结果说清楚"。好处是**数字不会算错**，这是小模型最容易翻车的地方。

代价是意图识别靠关键词，覆盖面有限。这是明确的取舍：
宁可少支持几个工具，也不要让 2B 模型去猜 JSON。

### 4. 检测链路的三级降级

```
ultralytics（本机，最快）
    ↓ 不可用
远程推理服务（yolo/serve.py，可放到别的机器）
    ↓ 不可用
明确返回"未就绪"，前端降级到浏览器端算法并标注"非 YOLO 结果"
```

**"没有模型"也是一个合法结果**，要如实告诉用户，而不是编一个看起来像检测结果的东西。

### 5. 桌面端为什么不直接上 Electron 打包

Electron 需要 `npm install`（几百 MB）和构建步骤，在装不了依赖的环境里走不通。
所以给了两条路：`desktop/run_desktop.py` 复用系统已有的 Edge/Chrome（零依赖，用于演示），
`desktop/electron/` 负责后续产品化打包。

---

## 目录结构

```
ai-lab/
├─ README.md              项目总说明
├─ 作业一-大模型.md        本地部署、智能体、API 接入
├─ 作业二-YOLO.md          训练、实时推理、接入应用
├─ 工作日志.md             开发过程记录
├─ images/                文档用的截图
├─ app/                   后端：配置、模型接入、智能体、检测、HTTP 服务、CLI
├─ web/                   前端（原生三件套，无构建步骤）
├─ yolo/                  训练、推理、实时、导出、独立推理服务
├─ desktop/               桌面端（Edge 应用模式 + Electron）
├─ tools/                 数据集下载、自检、压测等工具
├─ run_server.py          服务入口
├─ run_cli.py             命令行入口
└─ start-server.cmd       双击启动（Windows）
```

---

## 接口说明

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 服务状态、当前模型、检测后端 |
| GET | `/api/config` | 前端配置（密钥已脱敏） |
| GET | `/api/samples` | 内置示例图列表 |
| POST | `/api/chat` | SSE 流式对话，事件：`meta` / `step` / `tool` / `delta` / `done` / `stats` |
| POST | `/api/detect` | 上传图片（multipart，字段名 `file`），返回统一 JSON |

`/api/chat` 请求体：

```json
{
  "messages": [{ "role": "user", "content": "你好" }],
  "provider": "ollama",
  "model": "qwen2.5:1.5b",
  "use_tools": true
}
```

`/api/detect` 响应体：

```json
{
  "ok": true,
  "engine": "ultralytics",
  "model": "runs/train/coco8-ft/weights/best.pt",
  "width": 720, "height": 1280, "elapsed_ms": 21.3,
  "boxes": [
    { "x1": 167.4, "y1": 483.7, "x2": 276.5, "y2": 560.1,
      "score": 0.7855, "label": "car", "cls": 2 }
  ]
}
```

---

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `AILAB_LLM_PROVIDER` | `demo` | `demo` / `ollama` / `openai` |
| `AILAB_LLM_MODEL` | `qwen2.5:1.5b` | 模型名 |
| `AILAB_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama 地址 |
| `AILAB_OPENAI_BASE_URL` | `http://127.0.0.1:1234/v1` | 任意 OpenAI 兼容地址（LM Studio / Dify / 扣子 / 云端） |
| `AILAB_OPENAI_API_KEY` | `not-needed` | 本地服务随便填 |
| `AILAB_YOLO_BACKEND` | `auto` | `auto` / `ultralytics` / `remote` / `none` |
| `AILAB_YOLO_WEIGHTS` | `yolo11n.pt` | 权重路径 |
| `AILAB_YOLO_CONF` | `0.25` | 置信度阈值 |
| `AILAB_YOLO_DEVICE` | 空 | `0` / `cpu` |
| `AILAB_YOLO_REMOTE_URL` | 空 | 远程推理服务地址 |
| `AILAB_PORT` | `8000` | 服务端口 |

---

## 自测

```powershell
python tools\smoke_test.py
```

会自己拉起服务、依次访问所有接口，覆盖静态资源、SSE 流式对话、工具调用、
图片上传检测、错误分支，以及**加载真实权重跑一次推理**。当前 27 项全部通过。

另有三个专项检查：

```powershell
python tools\check_llm.py --provider ollama --model qwen2.5:1.5b   # 大模型连通性与速度
python tools\check_gpu.py                                          # 显卡与 torch 环境
python tools\bench_detect.py --frames 60                           # 检测接口往返耗时
```

---

## 常见问题

**网页打不开？**
先看 `start-server.cmd` 窗口里的日志。Windows 上如果 8000 被占用，脚本会自动往后找端口，
注意它打印的实际地址。另外确认 Ollama 在运行（`ollama list` 能看到 `qwen2.5:1.5b`）。

**检测面板显示"未就绪"？**
说明没有可用的权重。按 [作业二](作业二-YOLO.md) 跑一次训练，
或者用环境变量 `AILAB_YOLO_WEIGHTS` 指向已有的 `.pt` 文件。

**Ollama 连不上？**
确认服务在跑（托盘图标），再用 `curl http://127.0.0.1:11434/api/tags` 看能否列出模型。

**训练报 CUDA out of memory？**
按顺序试：`--batch 4` → `--imgsz 512` → 换 `yolo11n.pt`。

---

## 版本控制

仓库用 Conventional Commits 规范提交（`feat` / `fix` / `docs` / `test` / `chore`），
一次提交只做一件事。`.gitignore` 排除了 `runs/`、`datasets/`、`*.pt`、`*.mp4` 等训练产物，
`.gitattributes` 统一按 LF 存储。

> 训练权重、数据集和测试视频都**不在仓库里**，需要按文档里的命令重新生成。
