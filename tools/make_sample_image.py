"""生成一张用于演示的合成图片（纯标准库写 PNG，不依赖 Pillow）。

用途：在没有数据集、也没下载 YOLO 权重时，仍然有一张固定图片可以验证
「上传 → 推理接口 → 前端画框」的完整链路。
"""

from __future__ import annotations

import random
from pathlib import Path

from pngcanvas import Canvas

WIDTH, HEIGHT = 640, 400
OUT = Path(__file__).resolve().parents[1] / "web" / "samples" / "demo-street.png"


def build() -> Canvas:
    canvas = Canvas(WIDTH, HEIGHT)
    rng = random.Random(20261010)

    # 天空
    for y in range(0, 230):
        t = y / 230
        canvas.rect(0, y, WIDTH, y + 1, (int(176 + 40 * t), int(206 + 26 * t), int(226 + 18 * t)))

    # 远处楼房
    buildings = [
        (20, 90, 130, 232, (150, 158, 168)),
        (140, 60, 250, 232, (132, 141, 152)),
        (262, 108, 360, 232, (162, 169, 178)),
        (372, 74, 470, 232, (138, 147, 158)),
        (482, 120, 620, 232, (168, 175, 184)),
    ]
    for x0, y0, x1, y1, color in buildings:
        canvas.rect(x0, y0, x1, y1, color)
        for wy in range(y0 + 14, y1 - 12, 26):
            for wx in range(x0 + 12, x1 - 12, 24):
                canvas.rect(wx, wy, wx + 11, wy + 14, (226, 214, 168))

    # 路面与人行道
    canvas.rect(0, 232, WIDTH, HEIGHT, (86, 90, 96))
    canvas.rect(0, 232, WIDTH, 250, (172, 174, 178))
    for x in range(0, WIDTH, 64):
        canvas.rect(x + 10, 318, x + 46, 326, (232, 230, 224))

    # 车辆：三个明显的矩形色块
    cars = [
        (72, 268, 214, 322, (176, 62, 54)),
        (268, 282, 400, 330, (44, 96, 168)),
        (452, 262, 596, 316, (222, 178, 62)),
    ]
    for x0, y0, x1, y1, body in cars:
        canvas.round_rect(x0, y0, x1, y1, 12, body)
        canvas.round_rect(x0 + 26, y0 - 20, x1 - 22, y0 + 12, 10, (224, 232, 238))
        canvas.rect(x0 + 30, y0 - 14, x1 - 26, y0 + 6, (86, 116, 140))
        canvas.circle(x0 + 32, y1 - 4, 15, (38, 40, 44))
        canvas.circle(x1 - 34, y1 - 4, 15, (38, 40, 44))
        canvas.circle(x0 + 32, y1 - 4, 6, (150, 154, 160))
        canvas.circle(x1 - 34, y1 - 4, 6, (150, 154, 160))

    # 行人：一个高窄色块
    canvas.round_rect(224, 196, 250, 292, 8, (36, 78, 92))
    canvas.circle(237, 184, 13, (214, 176, 148))

    # 轻微噪点，避免大块纯色影响对比度算法
    for _ in range(26000):
        x = rng.randrange(WIDTH)
        y = rng.randrange(HEIGHT)
        index = (y * WIDTH + x) * 3
        delta = rng.randint(-9, 9)
        for channel in range(3):
            value = canvas.pixels[index + channel] + delta
            canvas.pixels[index + channel] = 0 if value < 0 else 255 if value > 255 else value

    return canvas


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas = build()
    OUT.write_bytes(canvas.to_png())
    print(f"已生成 {OUT} ({OUT.stat().st_size} bytes, {WIDTH}x{HEIGHT})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
