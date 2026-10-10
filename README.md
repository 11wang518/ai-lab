# AI Lab：本地大模型 + YOLO 一体化应用

一台 8G 显存的 RTX 4060 上，把「本地小模型对话」和「YOLO 目标检测」接进同一个应用里。
主应用**零第三方依赖**，只用 Python 标准库，克隆下来就能启动；大模型和 YOLO 通过可插拔后端接入，
装不装、用哪种，都不影响应用先跑起来。

先看这两份文档，能省不少时间：

- [实现思路与架构](docs/实现思路与架构.md)：为什么这么设计，有哪些取舍
- [工程日志](docs/开发日志.md)：每天做了什么、卡在哪、下一步
- [模型部署与显存规划](docs/模型部署与显存规划.md)：8G 显存怎么选模型、怎么调参

---

## 一、对应作业要求

| 要求 | 本项目实现 |
| --- | --- |
| 1. 本地部署大模型（Llama.cpp / Ollama / LM Studio） | `app/llm/providers.py` 提供 Ollama 与 OpenAI 兼容两类客户端，覆盖 Ollama、LM Studio、llama.cpp server、vLLM；模型选了 2B 级的 `qwen2.5:1.5b` |
| 2. 搭建智能体（Dify / n8n / 扣子等） | `app/agent/` 自带轻量智能体（意图识别 → 工具执行 → 交回模型），同时支持把 Dify / 扣子的 OpenAI 兼容地址当作模型后端接入 |
| 3. 以 API 形式接入应用 | `app/server/http_app.py`：`POST /api/chat`（SSE 流式）、`POST /api/detect`，全部 HTTP + JSON |
| 4. 应用形式 | CLI（`run_cli.py`）、Web（`web/`）、桌面（`desktop/`，Edge 应用模式 + Electron 两种） |
| 二 1. 完成一次 Ultralytics 训练 | `yolo/train.py`，默认参数按 8G 显存给（yolo11n + 640 + batch 8 + AMP） |
| 二 2. 实时推理部署 | `yolo/infer.py`（图片/目录/视频）、`yolo/realtime.py`（摄像头 + FPS）、`yolo/export.py`（导出 ONNX/TensorRT）、`yolo/serve.py`（独立推理服务） |
| 二 3. 接入应用 | Web 端上传图片 → 后端推理 → 前端 canvas 画框；桌面端复用同一个页面 |

---

## 二、快速开始（不装任何依赖）

```powershell
cd D:\tools\codex\ai-lab
python run_server.py
```

浏览器打开 <http://127.0.0.1:8000>。

此时模型后端是内置的示例模型（规则匹配 + 流式输出），检测后端显示"未就绪"，
这两条提示都是有意设计的：**先确认链路通，再逐个换成真模型**。

想一键看到满血效果，打开 <http://127.0.0.1:8000/?demo=1>：
自动载入示例图、跑一次检测、发一条会触发工具调用的消息。

命令行版：

```powershell
python run_cli.py
python run_cli.py --once "帮我算一下 12*(3+4)"
```

桌面版（用 Edge/Chrome 的应用模式，无需打包）：

```powershell
python desktop\run_desktop.py
```

---

## 三、接入真实大模型

### 方案 A：Ollama（推荐，命令最少）

```powershell
# 1. 安装：https://ollama.com/download
# 2. 拉一个 2B 级的千问（8G 显存非常宽裕）
ollama pull qwen2.5:1.5b

# 3. 让应用指向 Ollama
python run_server.py --provider ollama --model qwen2.5:1.5b
```

Ollama 自带 OpenAI 兼容接口，所以下面的方案 B 其实也能指向它
（`http://127.0.0.1:11434/v1`）。

### 方案 B：LM Studio

1. 打开 LM Studio，搜索并下载 `Qwen2.5-1.5B-Instruct`（或 3B）。
2. 左侧 `Developer` → 启动本地服务器（默认 `http://127.0.0.1:1234`）。
3. 启动应用：

```powershell
$env:AILAB_LLM_PROVIDER="openai"
$env:AILAB_OPENAI_BASE_URL="http://127.0.0.1:1234/v1"
$env:AILAB_LLM_MODEL="qwen2.5-1.5b-instruct"
python run_server.py
```

### 方案 C：llama.cpp

```powershell
# 下载 GGUF 权重（魔搭 / HuggingFace 都有 Qwen2.5-1.5B-Instruct-GGUF）
.\llama-server.exe -m qwen2.5-1.5b-instruct-q4_k_m.gguf -c 4096 --port 8080

$env:AILAB_LLM_PROVIDER="openai"
$env:AILAB_OPENAI_BASE_URL="http://127.0.0.1:8080/v1"
python run_server.py
```

### 方案 D：Dify / 扣子等平台

在平台上把应用发布成 API，拿到 OpenAI 兼容的 `Base URL` 和 `API Key`，然后：

```powershell
$env:AILAB_LLM_PROVIDER="openai"
$env:AILAB_OPENAI_BASE_URL="https://你的平台地址/v1"
$env:AILAB_LLM_MODEL="你的应用名"
$env:AILAB_OPENAI_API_KEY="sk-xxxx"
python run_server.py
```

模型下载源：

- 魔搭社区 <https://www.modelscope.cn/models>
- Hugging Face <https://huggingface.co/models>

---

## 四、跑通 YOLO

### 1. 装依赖

```powershell
pip install ultralytics
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()` 必须是 `True`。如果显卡版 PyTorch 没装上，
按 <https://pytorch.org/get-started/locally/> 选择 CUDA 版本重装。

### 2. 训练

先用 `yolo/data.example.yaml` 的格式准备数据集（`images/` + `labels/`），
然后：

```powershell
python yolo/train.py --data datasets\mydata\data.yaml --epochs 50 --batch 8 --imgsz 640
```

8G 显存的建议：

| 模型 | imgsz | batch | 说明 |
| --- | --- | --- | --- |
| `yolo11n.pt` | 640 | 8 | 默认，最稳 |
| `yolo11s.pt` | 640 | 4 | 精度更高，显存吃紧 |
| `yolo11n.pt` | 512 | 8 | 报显存不足时先降分辨率 |

训练完输出在 `runs/train/exp/weights/best.pt`，脚本会自动跑一次验证集并打印 mAP。

### 3. 推理与实时部署

```powershell
# 单张图片
python yolo\infer.py --weights runs\train\exp\weights\best.pt --source test.jpg --json out.json

# 整个文件夹
python yolo\infer.py --weights yolo11n.pt --source images\

# 摄像头实时（弹窗里显示 FPS，按 q 退出）
python yolo\realtime.py --weights yolo11n.pt --source 0 --imgsz 480 --half
```

### 4. 接进应用（两种方式）

**方式一：本机进程内推理**

```powershell
$env:AILAB_YOLO_WEIGHTS="runs\train\exp\weights\best.pt"
python run_server.py --vision-backend ultralytics
```

不设 `AILAB_YOLO_WEIGHTS` 时会用默认的 `yolo11n.pt`。

启动后打开网页的"目标检测"面板，上传图片即可看到画框结果。

**方式二：独立推理服务**（把推理放到另一台机器，或把依赖和主应用隔离）

```powershell
# 终端 1
python yolo\serve.py --weights yolo11n.pt --port 8600

# 终端 2
$env:AILAB_YOLO_REMOTE_URL="http://127.0.0.1:8600/predict"
python run_server.py
```

### 5. 导出成部署格式

```powershell
python yolo\export.py --weights yolo11n.pt --format onnx --simplify
python yolo\export.py --format engine --half      # 需要装 TensorRT
```

---

## 五、目录结构

```
ai-lab/
├─ app/                    主应用（Python 标准库）
│  ├─ config.py            全部配置，环境变量覆盖
│  ├─ llm/                 模型后端：demo / ollama / openai 兼容
│  ├─ agent/               轻量智能体：tools + 意图路由
│  ├─ vision/              检测后端：ultralytics / remote / none
│  ├─ server/              HTTP 服务：静态资源 + SSE + 上传
│  └─ cli.py               命令行客户端
├─ web/                    前端（原生三件套，无构建步骤）
├─ yolo/                   训练、推理、实时、导出、推理服务
├─ desktop/                桌面壳：Edge 应用模式 + Electron
├─ tools/                  示例图生成、端到端冒烟测试
└─ docs/                   部署笔记与实验报告
```

---

## 六、接口说明

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
  "model": "yolo11n.pt",
  "width": 640,
  "height": 400,
  "elapsed_ms": 42.7,
  "boxes": [
    { "x1": 72, "y1": 262, "x2": 214, "y2": 322, "score": 0.87, "label": "car", "cls": 2 }
  ]
}
```

`engine` 为 `none` 时表示没有可用模型，前端会降级到浏览器端的显著性算法演示，
并在界面上明确标注"非 YOLO 结果"，不会拿假结果冒充模型输出。

---

## 七、环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `AILAB_LLM_PROVIDER` | `demo` | `demo` / `ollama` / `openai` |
| `AILAB_LLM_MODEL` | `qwen2.5:1.5b` | 模型名 |
| `AILAB_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama 地址 |
| `AILAB_OPENAI_BASE_URL` | `http://127.0.0.1:1234/v1` | OpenAI 兼容地址 |
| `AILAB_OPENAI_API_KEY` | `not-needed` | 本地服务随便填 |
| `AILAB_LLM_TEMPERATURE` | `0.7` | 采样温度 |
| `AILAB_YOLO_BACKEND` | `auto` | `auto` / `ultralytics` / `remote` / `none` |
| `AILAB_YOLO_WEIGHTS` | `yolo11n.pt` | 权重路径 |
| `AILAB_YOLO_CONF` | `0.25` | 置信度阈值 |
| `AILAB_YOLO_IMGSZ` | `640` | 推理分辨率 |
| `AILAB_YOLO_DEVICE` | 空 | `0` / `cpu` |
| `AILAB_YOLO_REMOTE_URL` | 空 | 远程推理服务地址 |
| `AILAB_PORT` | `8000` | 服务端口 |

---

## 八、版本控制与提交规范

仓库已经初始化好（`main` 分支），关键文件：

- `.gitignore`：排除 `runs/`、`datasets/`、`*.pt`、`*.onnx` 等训练产物和大文件；
- `.gitattributes`：统一按 LF 存储，避免跨系统换行差异。

提交信息用 [Conventional Commits](https://www.conventionalcommits.org/) 前缀 + 中文描述，
例如 `feat(llm): 实现大模型接入层与命令行客户端`。前缀含义：

| 前缀 | 用在什么地方 |
| --- | --- |
| `feat` | 新增功能 |
| `fix` | 修 bug |
| `docs` | 只改文档 |
| `test` | 只改测试 |
| `chore` | 构建、配置、依赖等杂项 |

推送到 GitHub 的完整步骤见 [Git 与 GitHub 操作指南](docs/Git与GitHub操作指南.md)。

---

## 九、自测

```powershell
python tools\smoke_test.py
```

会自己拉起服务、依次访问所有接口，覆盖静态资源、SSE 流式对话、工具调用、
图片上传检测和错误分支，最后打印通过/失败统计。当前版本 23 项全部通过。

---

## 十、常见问题

**端口被占用？**
`run_server.py` 会自动往后找可用端口，启动日志里有实际地址。想固定端口就用 `--port`。

**Ollama 连不上？**
先确认 `ollama serve` 在跑，再 `curl http://127.0.0.1:11434/api/tags` 看能不能列出模型。
界面顶栏的状态点变绿才算接通。

**训练报 CUDA out of memory？**
按顺序试：`--batch 4` → `--imgsz 512` → `--cache` 去掉 → 换 `yolo11n.pt`。

**检测面板一直显示"未就绪"？**
说明 `import ultralytics` 失败。先 `pip install ultralytics`；
不想在主环境装，就用 `yolo/serve.py` 起独立服务，再配 `AILAB_YOLO_REMOTE_URL`。
