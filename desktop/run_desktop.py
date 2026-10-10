#!/usr/bin/env python
"""零依赖桌面壳：启动本地服务，并用 Edge/Chrome 的应用模式打开。

应用模式（--app=）没有地址栏和标签页，效果接近原生桌面应用，
不用装 Electron、也不用打包，适合课堂上快速演示。

用法：
    python desktop/run_desktop.py
    python desktop/run_desktop.py --provider ollama --model qwen2.5:1.5b
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import webbrowser
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import AppConfig  # noqa: E402
from app.server.http_app import create_server, find_free_port, serve_in_background  # noqa: E402

WINDOWS_BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]
UNIX_BROWSERS = [
    "microsoft-edge",
    "google-chrome",
    "chromium",
    "chromium-browser",
]


def find_browser() -> str | None:
    for path in WINDOWS_BROWSERS:
        if Path(path).exists():
            return path
    for name in UNIX_BROWSERS:
        found = shutil.which(name)
        if found:
            return found
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="桌面版 AI Lab")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--provider", help="demo | ollama | openai")
    parser.add_argument("--model", help="模型名，例如 qwen2.5:1.5b")
    parser.add_argument("--width", type=int, default=1320)
    parser.add_argument("--height", type=int, default=880)
    parser.add_argument("--browser-only", action="store_true", help="只用默认浏览器打开")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.from_env()

    llm = config.llm
    if args.provider:
        llm = replace(llm, provider=args.provider)
    if args.model:
        llm = replace(llm, model=args.model)
    config = replace(config, llm=llm)

    port = find_free_port(args.host, args.port)
    config = replace(config, server=replace(config.server, host=args.host, port=port))
    server = create_server(config)
    serve_in_background(server)
    url = f"http://{args.host}:{port}/"
    print(f"[desktop] 应用已启动：{url}")

    browser = None if args.browser_only else find_browser()
    if browser:
        import subprocess

        print(f"[desktop] 使用应用模式打开：{browser}")
        subprocess.Popen(
            [
                browser,
                f"--app={url}",
                f"--window-size={args.width},{args.height}",
                "--no-first-run",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        print("[desktop] 未找到 Edge/Chrome，改用默认浏览器")
        webbrowser.open(url)

    print("[desktop] 关闭窗口后按 Ctrl+C 结束服务")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[desktop] 已停止")
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    raise SystemExit(main())
