#!/usr/bin/env python
"""把几张真实照片拼成一段测试视频，用来在没有摄像头时验证实时推理。

每张图片会做轻微的平移，避免所有帧完全一样。

用法：
    python tools/make_test_video.py
    python tools/make_test_video.py --out test.mp4 --frames-per-image 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IMAGES = [
    ROOT / "web" / "samples" / "street.jpg",
    ROOT / "datasets" / "coco8" / "images" / "val" / "000000000036.jpg",
    ROOT / "datasets" / "coco8" / "images" / "val" / "000000000049.jpg",
    ROOT / "datasets" / "coco8" / "images" / "val" / "000000000061.jpg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成实时推理测试视频")
    parser.add_argument("--out", default=str(ROOT / "runs" / "test-realtime.mp4"))
    parser.add_argument("--size", type=int, default=640, help="输出边长")
    parser.add_argument("--frames-per-image", type=int, default=40)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import cv2
    except ImportError:
        print("需要 opencv-python：pip install opencv-python", file=sys.stderr)
        return 2

    images = [path for path in DEFAULT_IMAGES if path.exists()]
    if not images:
        print(
            "没有找到可用图片（先运行 tools/fetch_coco8.py 或放入 web/samples/）",
            file=sys.stderr,
        )
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (args.size, args.size),
    )
    if not writer.isOpened():
        print(f"无法创建视频：{out}", file=sys.stderr)
        return 1

    total = 0
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        scale = max(args.size / width, args.size / height)
        resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
        max_x = max(0, resized.shape[1] - args.size)
        max_y = max(0, resized.shape[0] - args.size)
        for index in range(args.frames_per_image):
            # 轻微平移，制造一点运动感
            offset = index % 12
            x = min(max_x, offset)
            y = min(max_y, offset // 2)
            crop = resized[y : y + args.size, x : x + args.size]
            if crop.shape[0] != args.size or crop.shape[1] != args.size:
                crop = cv2.resize(crop, (args.size, args.size))
            writer.write(crop)
            total += 1
        print(f"  已写入 {path.name}（{args.frames_per_image} 帧）")

    writer.release()
    print(f"\n完成：{out}（{total} 帧，{args.fps} fps，约 {total / args.fps:.1f} 秒）")
    print("\n接下来可以这样跑实时推理基准：")
    print(
        "  python yolo\\realtime.py "
        "--weights runs\\train\\coco8-ft\\weights\\best.pt "
        f"--source {out.as_posix()} --no-show --max-frames {total}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
