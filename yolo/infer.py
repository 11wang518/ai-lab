#!/usr/bin/env python
"""用训练好的权重做推理：单图 / 文件夹 / 视频 / 摄像头。

示例：
    python yolo/infer.py --weights runs/train/exp/weights/best.pt --source test.jpg
    python yolo/infer.py --weights yolo11n.pt --source images/
    python yolo/infer.py --weights yolo11n.pt --source 0          # 摄像头
    python yolo/infer.py --weights yolo11n.pt --source demo.mp4 --show
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 推理")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--source", required=True, help="图片/目录/视频/摄像头编号")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/infer")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--json", dest="json_path", help="把检测结果写到指定 JSON 文件")
    parser.add_argument("--show", action="store_true", help="弹窗实时显示（需要图形界面）")
    parser.add_argument("--no-save", action="store_true", help="不保存标注后的图片/视频")
    return parser.parse_args(argv)


def boxes_of(result) -> dict:
    names = getattr(result, "names", {}) or {}
    items = []
    for raw in result.boxes:
        cls_id = int(raw.cls[0].item())
        xyxy = raw.xyxy[0].tolist()
        items.append(
            {
                "label": str(names.get(cls_id, cls_id)),
                "cls": cls_id,
                "score": round(float(raw.conf[0].item()), 4),
                "x1": round(xyxy[0], 2),
                "y1": round(xyxy[1], 2),
                "x2": round(xyxy[2], 2),
                "y2": round(xyxy[3], 2),
            }
        )
    height, width = result.orig_shape[:2]
    return {"width": width, "height": height, "boxes": items}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("没有找到 ultralytics，请先 pip install ultralytics", file=sys.stderr)
        return 2

    model = YOLO(args.weights)
    collected: list[dict] = []
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=not args.no_save,
        save_txt=args.save_txt,
        show=args.show,
        project=args.project,
        name=args.name,
        stream=True,
        verbose=False,
    )

    total = 0
    for index, result in enumerate(results):
        payload = boxes_of(result)
        total += len(payload["boxes"])
        collected.append({"source": str(getattr(result, "path", index)), **payload})
        summary = "、".join(f"{b['label']} {b['score']:.2f}" for b in payload["boxes"][:5])
        print(f"[{index + 1}] {len(payload['boxes'])} 个目标  {summary}")

    print(f"\n共处理 {len(collected)} 帧/张，检测到 {total} 个目标")
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(collected, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"结果已写入 {args.json_path}")
    print(f"标注结果保存在 {Path(args.project) / args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
