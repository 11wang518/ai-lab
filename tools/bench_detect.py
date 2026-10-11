#!/usr/bin/env python
"""压测检测接口的往返耗时，用来估算网页端"摄像头实时"能达到多少帧率。

网页端的实时模式是"取一帧 → 上传 → 推理 → 画框 → 再取下一帧"的串行循环，
所以能跑多少帧，取决于一次往返要多久。这个脚本就是测这个数。

用法（需要先启动应用服务）：
    python tools/bench_detect.py --frames 60
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / "web" / "samples" / "street.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检测接口性能压测")
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/detect")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--max-side", type=int, default=640, help="模拟网页端缩放后的尺寸")
    return parser.parse_args()


def crop_to_jpeg(source: Path, max_side: int) -> bytes:
    """按网页端的方式把图片缩到最长边 max_side，再编码成 JPEG。"""
    try:
        from PIL import Image
    except ImportError:
        return source.read_bytes()
    with Image.open(source) as image:
        image = image.convert("RGB")
        scale = min(1.0, max_side / max(image.width, image.height))
        if scale < 1.0:
            image = image.resize((int(image.width * scale), int(image.height * scale)))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70)
        return buffer.getvalue()


def post_frame(url: str, payload: bytes) -> tuple[float, dict]:
    boundary = "----bench" + uuid.uuid4().hex
    crlf = b"\r\n"
    body = b"".join(
        [
            b"--" + boundary.encode() + crlf,
            b'Content-Disposition: form-data; name="file"; filename="frame.jpg"' + crlf,
            b"Content-Type: image/jpeg" + crlf + crlf,
            payload,
            crlf,
            b"--" + boundary.encode() + b"--" + crlf,
        ]
    )
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))
    return (time.perf_counter() - started) * 1000, data


def main() -> int:
    args = parse_args()
    source = Path(args.image)
    if not source.exists():
        print(f"图片不存在：{source}", file=sys.stderr)
        return 2

    payload = crop_to_jpeg(source, args.max_side)
    print(f"测试图片：{source.name}（上传 {len(payload) / 1024:.1f} KB）")
    print(f"接口：{args.url}")
    print(f"帧数：{args.frames}\n")

    # 第一帧通常包含模型加载，单独计时，不计入统计
    try:
        warmup_ms, warmup = post_frame(args.url, payload)
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"[失败] 连不上服务：{exc}", file=sys.stderr)
        print("       先运行 python tools\\start_background.py")
        return 1
    boxes = len(warmup.get("boxes", []))
    print(f"预热帧：{warmup_ms:.0f} ms（含模型加载），检出 {boxes} 个目标\n")

    samples: list[float] = []
    for index in range(args.frames):
        elapsed_ms, data = post_frame(args.url, payload)
        samples.append(elapsed_ms)
        if (index + 1) % 10 == 0:
            print(f"  已完成 {index + 1}/{args.frames}")

    samples.sort()
    mean = statistics.mean(samples)
    print("\n往返耗时统计")
    print(f"  平均   : {mean:.1f} ms  → 约 {1000 / mean:.1f} FPS")
    print(f"  最快   : {samples[0]:.1f} ms")
    print(f"  P50    : {samples[len(samples) // 2]:.1f} ms")
    print(f"  P95    : {samples[int(len(samples) * 0.95)]:.1f} ms")
    print(f"  最慢   : {samples[-1]:.1f} ms")
    print(f"\n结论：网页端串行取帧的方式，理论上限约 {1000 / mean:.0f} FPS（实际会低一些，"
          f"还要算上浏览器截帧和编码的时间）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
