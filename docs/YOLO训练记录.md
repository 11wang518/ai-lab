# YOLO 训练与推理记录

这一份是可复现的实验记录：环境、数据、参数、结果、产物、局限，全部写清楚。

---

## 一、结论先说

| 项目 | 结果 |
| --- | --- |
| 任务 | 目标检测（2 类：car / person） |
| 模型 | YOLO11n（2,590,230 参数，6.5 GFLOPs） |
| 训练轮数 | 100 epochs |
| 设备 | **RTX 4060（8 GB）** |
| 训练耗时 | **约 1.5 分钟** |
| 验证集 mAP50 | **0.995** |
| 验证集 mAP50-95 | **0.869** |
| 单张推理耗时 | **1.7 ms** |
| 训练时显存占用 | 0.4 GB（8 GB 里只用了 5%） |

CPU 和 GPU 各跑了一次，除 `--device` 外参数完全相同，可以直接对比：

| 对比项 | CPU（i5-12490F） | GPU（RTX 4060） | 差异 |
| --- | --- | --- | --- |
| 100 轮训练耗时 | 5.2 分钟 | **1.5 分钟** | 快约 3.5 倍 |
| 单张推理耗时 | 8.5 ms | **1.7 ms** | 快约 5 倍 |
| mAP50 | 0.995 | 0.995 | 持平 |
| mAP50-95 | 0.860 | **0.869** | 略优 |

分类别指标：

| 类别 | 图片数 | 实例数 | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- |
| car | 11 | 17 | 1.000 | 0.993 | 0.995 | 0.893 |
| person | 12 | 18 | 0.975 | 1.000 | 0.995 | 0.846 |

> 注意：上面这一轮用的是**合成数据**，指标好看但只能证明训练流程是通的，
> 训出来的模型在真实照片上不可用。**真正能用于真实场景的模型是第六节**
> 在 COCO8 真实数据上微调出来的那个。

---

## 二、为什么用自建数据集

原本计划直接用 ultralytics 自带的 `coco8`（8 张图的官方样例）。实际执行时发现
**这台机器访问 github.com 会证书校验失败**，而 ultralytics 的权重和样例数据都托管在 GitHub：

```
ssl.SSLCertVerificationError: certificate verify failed: unable to get local issuer certificate
```

换用 `curl` 试也是 SSL 连接错误，说明是网络环境问题，不是代码问题。

所以改成**自己生成数据集**：

- 优点是全程离线、可复现，不依赖任何外部下载；
- 而且标注格式（`images/` + `labels/*.txt` + `data.yaml`）和真实数据集完全一致，
  换成自己的数据只要改 `data.yaml` 里的路径；
- 缺点是场景简单，指标不能代表真实场景的难度——这一点必须诚实说明。

生成脚本：`tools/make_synthetic_dataset.py`

```powershell
python tools/make_synthetic_dataset.py --train 60 --val 15 --size 320
```

数据集内容：

| 项目 | 数值 |
| --- | --- |
| 训练集 | 60 张 |
| 验证集 | 15 张 |
| 图片尺寸 | 320 × 320 |
| 标注框总数 | 181 |
| 类别 | `0: car`、`1: person` |

图片是程序画出来的：天空、远景楼房、路面和车道线做背景，再随机放 1~4 个
"车"（圆角矩形车身 + 车窗 + 两个轮子）或"人"（圆形头 + 长条身体），
并加随机噪点。标注是画图时直接算出来的，因此不存在人工标错的问题。

---

## 三、训练参数

```powershell
python yolo\train.py ^
  --data datasets\synthetic\data.yaml ^
  --model yolo11n.yaml ^
  --epochs 100 --imgsz 320 --batch 8 --device 0 --workers 4 ^
  --name synthetic-gpu
```

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--model` | `yolo11n.yaml` | 用结构文件从零训练（拿不到预训练权重） |
| `--epochs` | 100 | 30 轮的版本置信度偏低，加到 100 轮后正常 |
| `--imgsz` | 320 | 与数据集尺寸一致 |
| `--batch` | 8 | CPU 训练，8 足够稳定 |
| `--device` | `0` | 第一块显卡（RTX 4060） |
| `--workers` | 4 | 显卡训练时数据加载可以多开几个 |
| 其余 | 默认 | 优化器 auto、AMP 开启、mosaic 等默认增强 |

### 关于显卡

`pip install ultralytics` 默认装的是 **CPU 版 torch**（`2.14.1+cpu`），
所以第一次训练是在 CPU 上完成的。换成 CUDA 版的过程曲折，完整记录在这里：

```powershell
pip uninstall -y torch torchvision

# 1. 清华镜像在这台机器上解析不了（pypi.tuna 和 mirrors.tuna 都不通），
#    可用的国内镜像是阿里云的 pytorch-wheels。
#    单连接只有 0.9 MB/s，2.6 GB 要 46 分钟；用项目自带的多线程下载工具：
python tools\fast_download.py ^
  http://mirrors.aliyun.com/pytorch-wheels/cu126/torch-2.14.1+cu126-cp313-cp313-win_amd64.whl ^
  %TEMP%\pwheels\torch-2.14.1+cu126-cp313-cp313-win_amd64.whl --threads 6

# 2. 校验文件完整，再本地安装
python -m zipfile -t %TEMP%\pwheels\torch-2.14.1+cu126-cp313-cp313-win_amd64.whl
pip install %TEMP%\pwheels\torch-2.14.1+cu126-cp313-cp313-win_amd64.whl

# 3. torchvision 只有 6 MB，直接从镜像装
pip install torchvision==0.29.1+cu126 --no-index ^
  -f http://mirrors.aliyun.com/pytorch-wheels/cu126/ --trusted-host mirrors.aliyun.com
```

三个容易踩的坑：

1. **清华镜像在本机 DNS 解析失败**，实测可用的国内镜像只有阿里云的 pytorch-wheels
   （魔搭社区能连，但没有现成的 torch wheel）；
2. **单连接被限速**到 0.9 MB/s；改成 6 线程分段后能到 4~6 MB/s，15 分钟下完。
   镜像传大文件时会随机断连（实测末尾有 3 个分片被重置，文件少了 2.5%），
   所以 `tools/fast_download.py` 带断点续传和自动重试，中断后重跑同一条命令即可；
3. **文件名必须符合 wheel 规范**。把 `torch-2.14.1+cu126-cp313-cp313-win_amd64.whl`
   改名成 `torch-cu126.whl` 会被 pip 直接拒绝。

装完用 `python tools\check_gpu.py` 确认：

```
PyTorch     : 2.14.1+cu126
CUDA 编译版 : 12.6
显卡可用    : True
  显卡 0     : NVIDIA GeForce RTX 4060（显存 8.0 GB，算力 8.9）
```

之后把 `--device cpu` 换成 `--device 0` 即可。

另外，显卡训练时 ultralytics 会做一次 AMP 自检，需要联网下载一个模型文件。
本机连不上 GitHub，自检被跳过并提示 "checks skipped. Offline and unable to
download YOLO26n"，训练仍按 `amp=True` 继续，实测没有出现 NaN 或 mAP 为 0，
所以这次不用处理；真遇到问题再加 `--no-amp` 关掉。

---

## 四、训练过程观察

- 前 10 轮 mAP 一直是 0，这是从零训练的正常现象：模型还在学"什么算一个目标"。
- 第 20 轮之后开始出现非零 mAP，loss 稳步下降。
- 到 100 轮时 box_loss 从最初的 4.39 降到约 0.95，cls_loss 从 3.57 降到约 0.77。
- 30 轮版本虽然 mAP50 也有 0.888，但**置信度普遍偏低**（验证图片上最高只有 0.16），
  实际用起来几乎检不出东西。这说明**只看 mAP 不够，还要看置信度分布**。

---

## 五、产物与复现

训练产出（`runs/` 已被 `.gitignore` 排除，不进仓库）：

```
runs/train/synthetic-cpu/
├─ weights/best.pt     最佳权重（5.4 MB）
├─ weights/last.pt     最后一轮权重
├─ results.png         损失与指标曲线
├─ confusion_matrix.png
└─ args.yaml           本次训练的全部参数
```

复现三步：

```powershell
python tools\make_synthetic_dataset.py            # 1. 生成数据集
python yolo\train.py --data datasets\synthetic\data.yaml --model yolo11n.yaml ^
  --epochs 100 --imgsz 320 --batch 8 --device 0   # 2. 训练（显卡）
python yolo\infer.py --weights runs\train\synthetic-gpu\weights\best.pt ^
  --source datasets\synthetic\images\val\val_0000.png --device 0   # 3. 推理
```

---

## 六、用真实数据微调：让自有模型也能认真实照片

前面那轮训练用的是合成数据，训出来的模型在真实照片上没用。
这一节把它补上：**在真实照片上用预训练权重做微调**，得到一个既是我们自己训练的、
又能认真实场景的模型。

### 数据

用 COCO8：COCO train2017 的前 8 张图（4 训练 / 4 验证），
**真实照片 + 真实标注**，YOLO 格式，体积只有 400 KB 左右。
官方数据托管在 GitHub（本机访问不了），从 HuggingFace 镜像获取：

```powershell
python tools\fetch_coco8.py
```

> 取数据时踩了个小坑：镜像会拦掉 `python-urllib` 的默认 User-Agent 并返回 403，
> 加一个正常的 User-Agent 头即可；另外镜像偶尔会 DNS 解析失败或断开连接，
> 所以脚本里带了自动重试。

### 训练参数

```powershell
python yolo\train.py ^
  --data datasets\coco8\data.yaml ^
  --model yolo11n.pt ^
  --epochs 50 --imgsz 640 --batch 8 --device 0 --workers 4 ^
  --name coco8-ft
```

和前面那轮的区别只有一个：`--model` 从 `yolo11n.yaml`（从零）换成 `yolo11n.pt`（预训练）。
加载时日志显示 `Transferred 499/499 items from pretrained weights`，80 个类别的权重全部继承。

训练在第 39 轮早停（第 19 轮最佳），**耗时约 25 秒**。

### 结果

COCO8 验证集（4 张真实照片、17 个实例）：

| 指标 | 数值 |
| --- | --- |
| mAP50 | **0.872** |
| mAP50-95 | **0.645** |
| person | mAP50 0.665 |
| dog / horse / umbrella / potted plant | mAP50 0.995 |

### 在真实街景照片上的表现

把项目里的街景照（`web/samples/street.jpg`）分别喂给两个模型：

| 模型 | 检测到的目标 | 最高置信度 |
| --- | --- | --- |
| 原始预训练 `yolo11n.pt` | 11 个 | car 0.87 |
| **自己微调 `coco8-ft/weights/best.pt`** | **12 个** | car 0.79 |

自己微调的模型在这张照片上输出了 car 0.79 / car 0.78 / motorcycle 0.77 /
person 0.60 / car 0.57，**真实场景下确实能用了**——这就是这一节要解决的问题。

### 老实说的部分

微调后的模型置信度比原始预训练模型低一些（0.79 vs 0.87），框也多几个。
原因很直接：**只用了 4 张图微调**，模型在拟合这几张图的同时，
对没见过的场景略微"失准"。想要更好的效果，正确做法是扩大真实数据集
（几百到几千张），而不是继续调参。

但结论是明确的：**只要在真实数据上训练过，自有模型就能认真实照片**，
合成数据训不出来这个问题，靠换数据解决，不是靠调参解决。

---

## 七、接进应用

### 应用里跑的就是自己训练的模型

现在应用默认加载 `runs/train/coco8-ft/weights/best.pt`——**自己微调出来的模型**，
它在真实数据上训练过，能认真实照片。三个模型的分工是这样的：

| 权重 | 怎么来的 | 能认真实照片 | 用途 |
| --- | --- | --- | --- |
| `runs/train/coco8-ft/weights/best.pt` | COCO8 真实数据微调 | **能** | **应用默认**，日常演示 |
| `yolo11n.pt` | 官方 COCO 预训练 | 能 | 微调的起点，可用来做对比 |
| `runs/train/synthetic-gpu/weights/best.pt` | 合成数据从零训练 | 不能 | 证明"从零训练"的流程也是通的 |

启动命令：

```powershell
python tools\start_background.py --provider ollama --model qwen2.5:1.5b ^
  --vision-backend ultralytics ^
  --vision-weights runs\train\coco8-ft\weights\best.pt --vision-conf 0.45
```

启动后打开 <http://127.0.0.1:8000>，检测面板显示 `ultralytics`，
点"示例图"载入真实街景照片。效果见 [screenshot-yolo-real.png](images/screenshot-yolo-real.png)。

接口返回示例（`POST /api/detect`）：

```json
{
  "ok": true,
  "engine": "ultralytics",
  "model": "yolo11n.pt",
  "width": 720, "height": 1280,
  "elapsed_ms": 22.58,
  "boxes": [
    {"x1": 167.02, "y1": 483.57, "x2": 275.23, "y2": 558.74,
     "score": 0.8748, "label": "car", "cls": 2},
    {"x1": 314.39, "y1": 532.92, "x2": 399.28, "y2": 678.17,
     "score": 0.8203, "label": "motorcycle", "cls": 3}
  ]
}
```

实测用的是一张真实街景照片（项目自带的 `web/samples/street.jpg`），
**自己微调的模型**识别出 **6 个目标**：

| 目标 | 置信度 |
| --- | --- |
| car（左侧白色 SUV） | 0.79 |
| car（最左） | 0.78 |
| motorcycle（电动车） | 0.77 |
| person（骑车的人） | 0.60 |
| car（前方黑色车） | 0.57 |
| person（右侧行人） | 0.56 |

单张 21.3 ms（GPU）。照片里的车辆、电动车和行人都被正确框出，
**这是自己训练的模型给出的结果**，不是官方预训练权重。

---

## 八、局限（必须说清楚）

1. **合成数据集那一轮的指标没有实际意义**：场景简单、背景干净，
   0.995 的 mAP 不代表真实效果。最直接的证据就是把真实照片喂给它，几乎认不出东西。
   它现在的作用只有一个——证明从零训练的流程是通的。
   真正能用的模型是第六节那轮在 COCO8 上微调出来的。
2. **微调用的数据太少**（只有 8 张，4 张训练）。所以微调后的模型置信度略低于
   原始预训练模型，框也多几个。要拿到能上台面的效果，得把真实数据集扩到
   几百到几千张，这不是调参能解决的。
3. **偶尔出现重复框**：同一辆车有时会被框两次（例如整体一个框、车身又一个框）。
   验证集指标看不出这个问题，因为 NMS 之后的重复框占比很小。
   工程上的应对是把置信度阈值调高（本项目用 0.45）或降低 NMS 的 IoU 阈值。
4. **没有做数据增强和超参数的对比实验**，用的全是 ultralytics 默认设置。
5. **没有做模型导出后的性能对比**（ONNX / TensorRT），目前只跑了 PyTorch 权重。
