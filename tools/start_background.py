#!/usr/bin/env python
"""把服务放到后台启动，不占用当前命令行窗口。

和直接 `python run_server.py` 的区别：
- 服务在独立进程里跑，关掉当前终端不会把它带走；
- 输出写到临时目录的日志文件里，方便出问题时回看；
- 启动后会等健康检查通过，打印可访问的地址再退出。

用法：
    python tools/start_background.py
    python tools/start_background.py --provider ollama --model qwen2.5:1.5b --port 8000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="后台启动 AI Lab 服务")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--vision-backend", default="auto", help="auto | ultralytics | remote | none")
    parser.add_argument("--vision-weights", default="", help="YOLO 权重路径")
    parser.add_argument("--vision-conf", type=float, default=None, help="置信度阈值")
    parser.add_argument("--timeout", type=float, default=25.0, help="等待启动的秒数")
    return parser.parse_args()


def wait_for_health(url: str, timeout: float) -> dict | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=3) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(0.5)
    return None


def main() -> int:
    args = parse_args()
    log_path = Path(tempfile.gettempdir()) / "ailab-server.log"

    command = [
        sys.executable,
        str(ROOT / "run_server.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--no-fallback-port",
    ]
    if args.vision_backend:
        command += ["--vision-backend", args.vision_backend]
    if args.vision_weights:
        command += ["--vision-weights", args.vision_weights]
    if args.vision_conf is not None:
        command += ["--vision-conf", str(args.vision_conf)]

    creation_flags = 0
    if sys.platform == "win32":
        # DETACHED_PROCESS：脱离当前控制台；CREATE_NEW_PROCESS_GROUP：不受 Ctrl+C 影响
        creation_flags = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
            close_fds=True,
        )

    url = f"http://{args.host}:{args.port}"
    health = wait_for_health(url, args.timeout)
    if health is None:
        print(f"[失败] {args.timeout:.0f} 秒内没能启动成功，进程号 {process.pid}")
        print(f"       日志：{log_path}")
        tail = log_path.read_text(encoding="utf-8", errors="replace").strip()
        if tail:
            print("       日志最后几行：")
            for line in tail.splitlines()[-8:]:
                print(f"       {line}")
        return 1

    llm = health.get("llm", {})
    vision = health.get("vision", {})
    print("[成功] 服务已在后台启动")
    print(f"       进程号   ：{process.pid}")
    print(f"       访问地址 ：{url}/")
    print(f"       模型后端 ：{llm.get('label')} / {llm.get('model')}")
    print(f"       检测后端 ：{vision.get('active_backend')}")
    print(f"       日志文件 ：{log_path}")
    print("\n停止服务：python tools\\stop_background.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
