# YOLO 训练与推理记录

这一份是可复现的实验记录：环境、数据、参数、结果、产物、局限，全部写清楚。

---

## 一、结论先说

| 项目 | 结果 |
| --- | --- |
| 任务 | 目标检测（2 类：car / person） |
| 模型 | YOLO11n（2,590,230 参数，6.5 GFLOPs） |
| 训练轮数 | 100 epochs |
| 训练耗时 | 0.087 小时（约 5.2 分钟） |
| 设备 | CPU（12th Gen Intel Core i5-12490F） |
| 验证集 mAP50 | **0.995** |
| 验证集 mAP50-95 | **0.860** |
| 单张推理耗时 | 8.5 ms（CPU） |

分类别指标：

| 类别 | 图片数 | 实例数 | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- |
| car | 11 | 17 | 0.979 | 1.000 | 0.995 | 0.860 |
| person | 12 | 18 | 0.988 | 1.000 | 0.995 | 0.859 |

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
  --epochs 100 --imgsz 320 --batch 8 --device cpu --workers 2 ^
  --name synthetic-cpu
```

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--model` | `yolo11n.yaml` | 用结构文件从零训练（拿不到预训练权重） |
| `--epochs` | 100 | 30 轮的版本置信度偏低，加到 100 轮后正常 |
| `--imgsz` | 320 | 与数据集尺寸一致 |
| `--batch` | 8 | CPU 训练，8 足够稳定 |
| `--device` | `cpu` | 见下文"关于显卡" |
| 其余 | 默认 | 优化器 auto、AMP 开启、mosaic 等默认增强 |

### 关于显卡

本机是 RTX 4060（8 GB），本该用 GPU 训练。但 `pip install ultralytics` 默认装的是
**CPU 版 torch**（`2.14.1+cpu`），要换成 CUDA 版需要再从
`download.pytorch.org` 下载 2.6 GB 的轮子，实测下载速度只有 0.65 MB/s，
预计要一小时以上，中途还多次连接超时。

权衡之后先走 CPU 路线把流程跑通——对这个数据集规模来说完全够用（5 分钟训完）。
要把训练切到显卡的完整步骤写在 `tools/check_gpu.py` 的提示里，也贴在这里：

```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# 官方源慢的话，国内镜像（实测可用）：
pip install torch==2.14.1+cu126 torchvision==0.29.1+cu126 --no-index ^
  -f https://mirrors.aliyun.com/pytorch-wheels/cu126/
```

装完用 `python tools\check_gpu.py` 确认 `显卡可用: True`，再把 `--device cpu`
换成 `--device 0` 即可。

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
  --epochs 100 --imgsz 320 --batch 8 --device cpu   # 2. 训练
python yolo\infer.py --weights runs\train\synthetic-cpu\weights\best.pt ^
  --source datasets\synthetic\images\val\val_0000.png --device cpu   # 3. 推理
```

---

## 六、接进应用

把训练出的权重接进 Web 应用：

```powershell
python tools\start_background.py --provider ollama --model qwen2.5:1.5b ^
  --vision-backend ultralytics ^
  --vision-weights runs\train\synthetic-cpu\weights\best.pt ^
  --vision-conf 0.45
```

启动后打开 <http://127.0.0.1:8000>，检测面板右上角会从"未加载"变成 `ultralytics`，
上传图片即可看到真实的检测框。效果截图见
[docs/images/screenshot-yolo.png](images/screenshot-yolo.png)。

接口返回示例（`POST /api/detect`）：

```json
{
  "ok": true,
  "engine": "ultralytics",
  "model": "runs/train/synthetic-cpu/weights/best.pt",
  "width": 640, "height": 400,
  "elapsed_ms": 48.42,
  "boxes": [
    {"x1": 88.51, "y1": 265.37, "x2": 191.74, "y2": 322.97,
     "score": 0.8821, "label": "car", "cls": 0}
  ]
}
```

实测：把项目自带的示例图（3 辆车 + 1 个行人）喂进去，4 个目标全部正确识别，
置信度分别是 0.88 / 0.86 / 0.79 / 0.64。

---

## 七、局限（必须说清楚）

1. **数据集是合成的**，场景简单、背景干净，0.995 的 mAP 不代表真实道路场景的效果。
   要得到有意义的指标，应该换成 COCO 子集、VOC 或自己标注的数据。
2. **偶尔出现重复框**：同一辆车有时会被框两次（例如整体一个框、车身又一个框）。
   验证集指标看不出这个问题，因为 NMS 之后的重复框占比很小。
   工程上的应对是把置信度阈值调高（本项目用 0.45）或降低 NMS 的 IoU 阈值。
3. **没有做数据增强的调参实验**，用的全是 ultralytics 默认增强。
4. **模型是从零训练的**，没用预训练权重做迁移学习——因为拿不到权重文件。
   能联网时把 `--model yolo11n.yaml` 换成 `--model yolo11n.pt`，
   用更少轮数就能取得更好的效果。
