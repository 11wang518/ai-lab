#!/usr/bin/env python
"""停止由 start_background.py 启动的服务（按端口找进程，避免误杀）。"""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="停止 AI Lab 后台服务")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def find_pid(port: int) -> int | None:
    try:
        output = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    needle = f":{port}"
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].upper() == "TCP" and parts[1].endswith(needle):
            if parts[3].upper() == "LISTENING":
                try:
                    return int(parts[4])
                except ValueError:
                    continue
    return None


def main() -> int:
    args = parse_args()
    pid = find_pid(args.port)
    if pid is None:
        print(f"[提示] 端口 {args.port} 上没有正在监听的服务")
        return 0
    result = subprocess.run(
        ["taskkill", "/F", "/PID", str(pid)], capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"[成功] 已停止进程 {pid}（端口 {args.port}）")
        return 0
    print(f"[失败] 无法停止进程 {pid}：{result.stdout.strip() or result.stderr.strip()}")
    print("       如果提示拒绝访问，请用管理员身份运行本脚本。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
