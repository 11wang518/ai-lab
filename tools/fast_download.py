"""多线程分段下载大文件，专治"单连接被限速"。

背景：装 CUDA 版 PyTorch 要下 2.6 GB 的 wheel。单连接从国内镜像下载只有
0.9 MB/s，要 46 分钟；改成 6~8 个分片并发后能跑到 4~6 MB/s，十几分钟就够。
镜像在传大文件时还会随机断开连接，所以每个分片都带断点续传和自动重试。

用法：
    python tools/fast_download.py <URL> <输出文件> --threads 6

中断后重跑同一条命令即可续传（分片文件会保留在输出文件旁边）。
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import urllib.request
from pathlib import Path


def get_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def fetch_range(
    url: str, start: int, end: int, path: Path, done: list[int], lock, index: int
) -> bool:
    """下载一个分片，断线后从已下载的位置继续（最多重试 8 次）。"""
    expected = end - start + 1
    for attempt in range(1, 9):
        have = path.stat().st_size if path.exists() else 0
        if have >= expected:
            with lock:
                done[index] = expected
            return True
        with lock:
            done[index] = have
        try:
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={start + have}-{end}"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                with path.open("ab" if have else "wb") as handle:
                    while True:
                        chunk = response.read(262144)
                        if not chunk:
                            break
                        handle.write(chunk)
                        have += len(chunk)
                        with lock:
                            done[index] = have
        except Exception as exc:  # noqa: BLE001 - 网络中断是常态，需要继续重试
            print(
                f"  分片 {index} 第 {attempt} 次中断（{type(exc).__name__}），已收 "
                f"{have / 1024**2:.0f} MB，稍后重试",
                flush=True,
            )
            time.sleep(min(3 * attempt, 15))
    have = path.stat().st_size if path.exists() else 0
    return have >= expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    total = get_size(args.url)
    print(f"文件大小 {total / 1024**3:.2f} GB，使用 {args.threads} 个线程")

    part = total // args.threads
    ranges = []
    for index in range(args.threads):
        start = index * part
        end = total - 1 if index == args.threads - 1 else start + part - 1
        ranges.append((start, end))

    done = [0] * args.threads
    lock = threading.Lock()
    parts = [output.with_suffix(output.suffix + f".part{index}") for index in range(args.threads)]

    threads = [
        threading.Thread(
            target=fetch_range,
            args=(args.url, start, end, parts[index], done, lock, index),
            daemon=True,
        )
        for index, (start, end) in enumerate(ranges)
    ]

    started = time.time()
    for thread in threads:
        thread.start()

    last = 0
    while any(thread.is_alive() for thread in threads):
        time.sleep(5)
        current = sum(done)
        speed = (current - last) / 5 / 1024**2
        last = current
        percent = current / total * 100
        print(f"  {percent:5.1f}%  {current / 1024**2:7.0f} MB  {speed:5.2f} MB/s", flush=True)

    results = []
    for thread in threads:
        thread.join()

    for index, (start, end) in enumerate(ranges):
        expected = end - start + 1
        actual = parts[index].stat().st_size if parts[index].exists() else 0
        results.append(actual >= expected)
        if actual < expected:
            print(f"  分片 {index} 不完整：{actual}/{expected} 字节")

    elapsed = time.time() - started
    print(f"分片完成，耗时 {elapsed / 60:.1f} 分钟，平均 {total / elapsed / 1024**2:.2f} MB/s")
    if not all(results):
        print("有分片未下完，保留 .part 文件，下次运行会自动续传。")
        return 1

    print("正在合并…")
    with output.open("wb") as target:
        for path in parts:
            with path.open("rb") as source:
                while True:
                    chunk = source.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
            path.unlink()
    print(f"完成：{output}（{output.stat().st_size} 字节）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
