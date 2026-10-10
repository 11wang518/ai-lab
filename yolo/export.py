#!/usr/bin/env python
"""把 PyTorch 权重导出成部署格式（ONNX / TensorRT / OpenVINO 等）。

示例：
    python yolo/export.py --weights runs/train/exp/weights/best.pt --format onnx
    python yolo/export.py --format engine --half --imgsz 640    # 需要本机有 TensorRT
"""

from __future__ import annotations

import argparse
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 YOLO 模型")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument(
        "--format",
        default="onnx",
        choices=["onnx", "torchscript", "engine", "openvino", "ncnn", "coreml", "tflite"],
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="FP16")
    parser.add_argument("--int8", action="store_true", help="INT8 量化（需校准数据）")
    parser.add_argument("--dynamic", action="store_true", help="动态输入尺寸")
    parser.add_argument("--simplify", action="store_true", help="简化 ONNX 计算图")
    parser.add_argument("--device", default="0")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("没有找到 ultralytics，请先 pip install ultralytics", file=sys.stderr)
        return 2

    model = YOLO(args.weights)
    path = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        dynamic=args.dynamic,
        simplify=args.simplify,
        device=args.device,
    )
    print(f"导出完成：{path}")
    if args.format == "onnx":
        print(
            "下一步可以用 ONNX Runtime 推理：\n"
            "  pip install onnxruntime-gpu\n"
            "  python yolo/serve.py --weights 导出文件.onnx"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
