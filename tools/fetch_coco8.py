#!/usr/bin/env python
"""下载 COCO8 数据集（8 张真实图片，YOLO 标注格式）。

为什么用它：COCO8 是 COCO train2017 的前 8 张图（4 训练 / 4 验证），
体积只有几百 KB，但**是真实照片、带真实标注**，适合用来做微调的冒烟验证。
官方数据托管在 GitHub（这台机器访问不了），所以从 HuggingFace 镜像取。

数据来源：https://hf-mirror.com/datasets/cc92yy3344/coco8 （AGPL-3.0）

用法：
    python tools/fetch_coco8.py
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "datasets" / "coco8"

REPO = "cc92yy3344/coco8"
MIRROR = "https://hf-mirror.com"

TRAIN_IDS = ["000000000009", "000000000025", "000000000030", "000000000034"]
VAL_IDS = ["000000000036", "000000000042", "000000000049", "000000000061"]


def files() -> list[str]:
    items = ["data.yaml", "LICENSE"]
    for split, ids in (("train", TRAIN_IDS), ("val", VAL_IDS)):
        items += [f"images/{split}/{name}.jpg" for name in ids]
        items += [f"labels/{split}/{name}.txt" for name in ids]
    return items


def download(relative: str) -> None:
    target = OUT_DIR / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"  已存在，跳过 {relative}")
        return
    url = f"{MIRROR}/datasets/{REPO}/resolve/main/{relative}"
    # 镜像会拦掉 python-urllib 的默认 User-Agent（返回 403），带一个正常的即可
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (compatible; ai-lab-fetch/1.0)"}
    )
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(request, timeout=60) as response, target.open(
                "wb"
            ) as handle:
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    handle.write(chunk)
            last_error = None
            break
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            target.unlink(missing_ok=True)
            print(f"  第 {attempt} 次失败（{exc}），重试…")
            time.sleep(2 * attempt)
    if last_error is not None:
        raise RuntimeError(f"下载失败 {relative}：{last_error}") from last_error
    print(f"  下载 {relative}（{target.stat().st_size} 字节）")


def fix_data_yaml() -> None:
    """把 data.yaml 里的 path 改成数据集所在目录，避免依赖原始仓库结构。"""
    path = OUT_DIR / "data.yaml"
    text = path.read_text(encoding="utf-8")
    lines = [
        line
        for line in text.splitlines()
        if not line.startswith(("path:", "train:", "val:", "test:"))
    ]
    header = [
        "# COCO8：真实图片 + 真实标注，仅用于验证训练/微调流程",
        f"path: {OUT_DIR.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
    ]
    path.write_text("\n".join(header + lines) + "\n", encoding="utf-8")
    print(f"  已修正数据集路径：{path}")


def main() -> int:
    print(f"下载 COCO8 到 {OUT_DIR}")
    try:
        for relative in files():
            download(relative)
    except RuntimeError as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        print("       检查网络，或手动从 https://hf-mirror.com/datasets/cc92yy3344/coco8 下载")
        return 1
    fix_data_yaml()

    images = list((OUT_DIR / "images").rglob("*.jpg"))
    labels = list((OUT_DIR / "labels").rglob("*.txt"))
    print(f"\n完成：{len(images)} 张图片，{len(labels)} 个标注文件")
    print(f"数据集配置：{OUT_DIR / 'data.yaml'}")
    print("\n下一步微调（用预训练权重，在真实数据上训练）：")
    print("  python yolo\\train.py --data datasets\\coco8\\data.yaml "
          "--model yolo11n.pt --epochs 50 --imgsz 640 --batch 8 --device 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
