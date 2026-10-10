#!/usr/bin/env python
"""用 ultralytics 训练 YOLO 模型。

本机是 8G 显存的 RTX 4060，默认参数就是按这个显存档位给的：
    yolo11n + imgsz=640 + batch=8 + AMP
如果出现显存不足，先把 --batch 降到 4，再考虑 --imgsz 512。

示例：
    python yolo/train.py --data yolo/data.example.yaml --epochs 50
    python yolo/train.py --model yolo11s.pt --epochs 100 --batch 4 --name exp-s
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 YOLO 模型")
    parser.add_argument("--data", default="yolo/data.example.yaml", help="数据集配置")
    parser.add_argument("--model", default="yolo11n.pt", help="预训练权重或模型结构")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8, help="8G 显存建议 4~8")
    parser.add_argument("--device", default="0", help="0 表示第一块 GPU，cpu 表示用 CPU")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20, help="早停轮数")
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--resume", action="store_true", help="从上次中断处继续")
    parser.add_argument("--cache", action="store_true", help="缓存图片加速训练（占内存）")
    parser.add_argument("--no-amp", action="store_true", help="关闭混合精度")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "没有找到 ultralytics。请先安装：\n"
            "  pip install ultralytics\n"
            "（会自动带上 torch。若要用显卡，请按 PyTorch 官网选择 CUDA 版本安装命令。）",
            file=sys.stderr,
        )
        return 2

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"数据集配置不存在：{data_path}", file=sys.stderr)
        return 2

    model = YOLO(args.model)
    print(
        f"开始训练：model={args.model} data={args.data} epochs={args.epochs} "
        f"imgsz={args.imgsz} batch={args.batch} device={args.device}"
    )
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        seed=args.seed,
        project=args.project,
        name=args.name,
        resume=args.resume,
        cache=args.cache,
        amp=not args.no_amp,
        plots=True,
        exist_ok=True,
    )

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    print("\n训练完成")
    print(f"  输出目录 : {save_dir}")
    print(f"  最佳权重 : {save_dir / 'weights' / 'best.pt'}")
    print(f"  最后权重 : {save_dir / 'weights' / 'last.pt'}")
    print(f"  结果曲线 : {save_dir / 'results.png'}")

    best = save_dir / "weights" / "best.pt"
    if best.exists():
        metrics = YOLO(str(best)).val(data=str(data_path), imgsz=args.imgsz, device=args.device)
        print("\n验证集指标")
        print(f"  mAP50    : {metrics.box.map50:.4f}")
        print(f"  mAP50-95 : {metrics.box.map:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
