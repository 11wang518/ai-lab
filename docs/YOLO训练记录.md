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

## 六、接进应用

### 应用演示用 COCO 预训练权重

自己训的模型只认识"合成风格的车和人"——用真实照片去测，效果很差，
这是拿合成数据训练的必然结果。

所以分成两条线，各自说清楚用途：

| 权重 | 用途 | 能不能测真实照片 |
| --- | --- | --- |
| `yolo11n.pt`（COCO 预训练） | 应用里的真实图片演示 | 能，80 个真实类别 |
| `runs/train/synthetic-gpu/weights/best.pt`（自己训练） | 证明训练流程跑通 | 不能，只认合成风格 |

预训练权重从 HuggingFace 镜像下载（GitHub 在这台机器上不通）：

```powershell
curl -L -o yolo11n.pt https://hf-mirror.com/Ultralytics/YOLO11/resolve/main/yolo11n.pt

python tools\start_background.py --provider ollama --model qwen2.5:1.5b ^
  --vision-backend ultralytics --vision-weights yolo11n.pt --vision-conf 0.45
```

启动后打开 <http://127.0.0.1:8000>，检测面板右上角显示 `ultralytics`，
点"示例图"载入的就是真实照片。效果见 [screenshot-yolo-real.png](images/screenshot-yolo-real.png)。

### 自己训练的权重也能接

```powershell
python tools\start_background.py --provider ollama --model qwen2.5:1.5b ^
  --vision-backend ultralytics ^
  --vision-weights runs\train\synthetic-gpu\weights\best.pt --vision-conf 0.45
```

这条链路只在合成风格的图片上有效，用来验证"自己训练的权重能跑通应用"。

接口返回示例（`POST /api/detect`）：

```json
{
  "ok": true,
  "engine": "ultralytics",
  "model": "yolo11n.pt",
  "width": 810, "height": 1080,
  "elapsed_ms": 25.84,
  "boxes": [
    {"x1": 3.8, "y1": 229.36, "x2": 796.22, "y2": 728.41,
     "score": 0.9403, "label": "bus", "cls": 5},
    {"x1": 671.02, "y1": 394.81, "x2": 809.81, "y2": 878.72,
     "score": 0.8882, "label": "person", "cls": 0}
  ]
}
```

实测：真实照片（一辆公交车 + 四名行人）上，模型给出 bus 0.94、
person 0.89 / 0.88 / 0.86 / 0.62，全部正确，单张 25.8 ms。

---

## 七、局限（必须说清楚）

1. **数据集是合成的**，场景简单、背景干净，0.995 的 mAP 不代表真实道路场景的效果。
   最直接的证据：把真实照片喂给这个自己训练的模型，它几乎认不出东西——
   它学的是"合成风格的车和人"。要得到有意义的指标，必须换成 COCO 子集、
   VOC 或自己标注的真实数据。
2. **偶尔出现重复框**：同一辆车有时会被框两次（例如整体一个框、车身又一个框）。
   验证集指标看不出这个问题，因为 NMS 之后的重复框占比很小。
   工程上的应对是把置信度阈值调高（本项目用 0.45）或降低 NMS 的 IoU 阈值。
3. **没有做数据增强的调参实验**，用的全是 ultralytics 默认增强。
4. **这次训练是从零开始的**（`--model yolo11n.yaml`），因为一开始拿不到预训练权重。
   后来从 HuggingFace 镜像下到了 `yolo11n.pt`，所以下一步应该改成
   **用预训练权重在真实数据上微调**：把 `--model yolo11n.yaml` 换成
   `--model yolo11n.pt`，再加一个真实数据集，用更少的轮数就能得到能用于真实场景的模型。
