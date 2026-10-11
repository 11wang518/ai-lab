#!/usr/bin/env python
"""生成一个小型合成数据集（图片 + YOLO 格式标注），用于离线跑通训练流程。

为什么需要它：这台机器连不上 GitHub，拿不到 coco8 这类官方样例数据。
与其卡住，不如自己造一份——流程（images/labels/data.yaml）和真实数据集完全一致，
换成自己的数据集只需要改 data.yaml 里的路径。

标注格式：每个 txt 一行一个目标，`类别 cx cy w h`，
其中 cx/cy/w/h 都是相对图片宽高归一化到 0~1 的值。

用法：
    python tools/make_synthetic_dataset.py
    python tools/make_synthetic_dataset.py --train 60 --val 15 --size 320
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from pngcanvas import Canvas

ROOT = Path(__file__).resolve().parents[1]

CLASSES = ["car", "person"]
CAR_COLORS = [
    (176, 62, 54),
    (44, 96, 168),
    (222, 178, 62),
    (58, 132, 96),
    (140, 92, 168),
    (206, 118, 62),
]
PERSON_COLORS = [
    (36, 78, 92),
    (92, 62, 48),
    (48, 52, 96),
    (122, 44, 60),
]


def _overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / min(area_a, area_b)


def draw_background(canvas: Canvas, rng: random.Random) -> tuple[int, int]:
    width, height = canvas.width, canvas.height
    horizon = int(height * 0.45)
    for y in range(horizon):
        ratio = y / max(1, horizon)
        canvas.rect(
            0,
            y,
            width,
            y + 1,
            (int(170 + 45 * ratio), int(200 + 30 * ratio), int(222 + 20 * ratio)),
        )
    # 远处楼房
    x = 0
    while x < width:
        bw = rng.randint(int(width * 0.08), int(width * 0.18))
        bh = rng.randint(int(height * 0.08), int(height * 0.28))
        tone = rng.randint(140, 178)
        canvas.rect(x, horizon - bh, x + bw, horizon, (tone, tone + 6, tone + 12))
        x += bw + rng.randint(2, 10)
    # 路面
    canvas.rect(0, horizon, width, height, (88, 92, 98))
    canvas.rect(0, horizon, width, horizon + int(height * 0.04), (176, 178, 182))
    dash_y = int(height * 0.78)
    dash_w = max(8, width // 12)
    for dx in range(0, width, dash_w * 2):
        canvas.rect(dx, dash_y, dx + dash_w, dash_y + max(2, height // 90), (232, 230, 224))
    return horizon, height


def draw_car(canvas: Canvas, box: tuple[int, int, int, int], rng: random.Random) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    body_color = rng.choice(CAR_COLORS)
    body_top = y0 + int(h * 0.24)
    canvas.round_rect(x0, body_top, x1, y1, max(4, h // 8), body_color)
    cabin = (x0 + int(w * 0.18), y0, x1 - int(w * 0.20), body_top + int(h * 0.06))
    canvas.round_rect(cabin[0], cabin[1], cabin[2], cabin[3], max(3, h // 10), (228, 236, 242))
    canvas.rect(
        cabin[0] + 4, cabin[1] + 4, cabin[2] - 4, cabin[3] - 2, (92, 122, 148)
    )
    radius = max(3, int(h * 0.15))
    wheel_y = y1 - radius
    for cx in (x0 + int(w * 0.24), x1 - int(w * 0.24)):
        canvas.circle(cx, wheel_y, radius, (36, 38, 42))
        canvas.circle(cx, wheel_y, max(1, radius // 3), (152, 156, 162))


def draw_person(canvas: Canvas, box: tuple[int, int, int, int], rng: random.Random) -> None:
    x0, y0, x1, y1 = box
    w = x1 - x0
    head_radius = max(3, int(w * 0.85))
    center_x = (x0 + x1) // 2
    canvas.circle(center_x, y0 + head_radius, head_radius, (214, 176, 148))
    body_top = y0 + head_radius * 2
    canvas.round_rect(
        x0, body_top, x1, y1, max(2, w // 3), rng.choice(PERSON_COLORS)
    )


def sample_box(
    rng: random.Random, kind: str, width: int, height: int, horizon: int
) -> tuple[int, int, int, int]:
    if kind == "car":
        box_w = rng.randint(int(width * 0.18), int(width * 0.34))
        box_h = rng.randint(int(height * 0.13), int(height * 0.22))
        x0 = rng.randint(4, max(5, width - box_w - 4))
        y0 = rng.randint(horizon + int(height * 0.06), max(horizon + int(height * 0.07), height - box_h - 4))
    else:
        box_w = rng.randint(int(width * 0.06), int(width * 0.11))
        box_h = rng.randint(int(height * 0.22), int(height * 0.34))
        x0 = rng.randint(4, max(5, width - box_w - 4))
        y0 = rng.randint(int(horizon * 0.55), max(int(horizon * 0.56), height - box_h - 4))
    return (x0, y0, x0 + box_w, y0 + box_h)


def make_image(size: int, rng: random.Random) -> tuple[bytes, list[tuple[int, int, int, int, int]]]:
    canvas = Canvas(size, size)
    horizon, _ = draw_background(canvas, rng)
    boxes: list[tuple[int, int, int, int, int]] = []
    target = rng.randint(1, 4)
    for _ in range(target):
        for _attempt in range(60):
            kind = "car" if rng.random() < 0.62 else "person"
            candidate = sample_box(rng, kind, size, size, horizon)
            if all(_overlap_ratio(candidate, b[:4]) < 0.15 for b in boxes):
                class_id = CLASSES.index(kind)
                draw_car(canvas, candidate, rng) if kind == "car" else draw_person(
                    canvas, candidate, rng
                )
                boxes.append((*candidate, class_id))
                break
    canvas.noise(rng, size * size // 22, amount=6)
    return canvas.to_png(), boxes


def write_label(path: Path, boxes: list[tuple[int, int, int, int, int]], size: int) -> None:
    lines = []
    for x0, y0, x1, y1, class_id in boxes:
        cx = (x0 + x1) / 2 / size
        cy = (y0 + y1) / 2 / size
        w = (x1 - x0) / size
        h = (y1 - y0) / size
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成合成数据集")
    parser.add_argument("--out", default=str(ROOT / "datasets" / "synthetic"))
    parser.add_argument("--train", type=int, default=60)
    parser.add_argument("--val", type=int, default=15)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    total_boxes = 0
    for split, count in (("train", args.train), ("val", args.val)):
        for index in range(count):
            data, boxes = make_image(args.size, rng)
            name = f"{split}_{index:04d}"
            (out / "images" / split / f"{name}.png").write_bytes(data)
            write_label(out / "labels" / split / f"{name}.txt", boxes, args.size)
            total_boxes += len(boxes)
        print(f"{split}: {count} 张图片")

    yaml_text = (
        "# 自动生成的合成数据集配置\n"
        f"path: {out.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        f"nc: {len(CLASSES)}\n"
        "names:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASSES))
    )
    (out / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print(f"\n输出目录：{out}")
    print(f"共 {total_boxes} 个标注框，类别：{', '.join(CLASSES)}")
    print(f"数据集配置：{out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
