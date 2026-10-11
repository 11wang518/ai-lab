#!/usr/bin/env python
"""检查 YOLO 训练/推理所需的显卡环境是否就绪。

跑训练之前先跑这个，能提前发现"装成了 CPU 版 torch"这类问题。

用法：
    python tools/check_gpu.py
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.env_setup import prepare  # noqa: E402

prepare()  # 必须在 import ultralytics 之前


def main() -> int:
    print("=" * 60)
    print("YOLO 运行环境检查")
    print("=" * 60)
    print(f"Python      : {sys.version.split()[0]} ({platform.machine()})")
    print(f"操作系统    : {platform.system()} {platform.release()}")

    try:
        import torch
    except ImportError:
        print("\n[失败] 没有安装 torch。先执行：pip install ultralytics")
        return 1

    print(f"PyTorch     : {torch.__version__}")
    print(f"CUDA 编译版 : {torch.version.cuda or '无（CPU 版）'}")
    available = torch.cuda.is_available()
    print(f"显卡可用    : {available}")

    if available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            print(
                f"  显卡 {index}     : {props.name}"
                f"（显存 {props.total_memory / 1024**3:.1f} GB，"
                f"算力 {props.major}.{props.minor}）"
            )

    try:
        import ultralytics

        print(f"ultralytics : {ultralytics.__version__}")
    except ImportError:
        print("ultralytics : 未安装（pip install ultralytics）")
        return 1

    print()
    if available:
        print("[通过] 可以用显卡训练。训练时加 --device 0。")
        return 0

    print("[警告] torch 是 CPU 版，训练会慢很多。")
    print("       要用显卡，按下面步骤换成 CUDA 版（本机实测可行的路径）：")
    print("       1. 先用多线程工具把 wheel 拉到本地（单连接会被限速到 0.9 MB/s）：")
    print("          python tools\\fast_download.py \\")
    print("            http://mirrors.aliyun.com/pytorch-wheels/cu126/"
          "torch-2.14.1+cu126-cp313-cp313-win_amd64.whl \\")
    print("            %TEMP%\\pwheels\\torch-2.14.1+cu126-cp313-cp313-win_amd64.whl --threads 6")
    print("       2. 校验并安装（文件名必须符合 wheel 规范，不能随便改名）：")
    print("          python -m zipfile -t %TEMP%\\pwheels\\torch-2.14.1+cu126-cp313-cp313-win_amd64.whl")
    print("          pip uninstall -y torch torchvision")
    print("          pip install %TEMP%\\pwheels\\torch-2.14.1+cu126-cp313-cp313-win_amd64.whl")
    print("          pip install torchvision==0.29.1+cu126 --no-index \\")
    print("            -f http://mirrors.aliyun.com/pytorch-wheels/cu126/ --trusted-host mirrors.aliyun.com")
    print("       说明：清华镜像在本机解析不了，官方源速度只有 0.65 MB/s；")
    print("             阿里云镜像可用，魔搭社区能连但没有现成的 torch wheel。")
    print("\n       小数据集（比如 coco8）用 CPU 也能跑通，只是慢一点。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
