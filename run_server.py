#!/usr/bin/env python
"""Web 服务入口：python run_server.py [--port 8000] [--provider ollama]"""

from __future__ import annotations

import argparse
from dataclasses import replace

from app.config import AppConfig
from app.server.http_app import create_server, find_free_port, serve_forever


def main() -> int:
    parser = argparse.ArgumentParser(description="启动 AI Lab Web 服务")
    parser.add_argument("--host", help="监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, help="监听端口，默认 8000")
    parser.add_argument("--provider", help="demo | ollama | openai")
    parser.add_argument("--model", help="模型名，例如 qwen2.5:1.5b")
    parser.add_argument(
        "--vision-backend", help="auto | ultralytics | remote | none"
    )
    parser.add_argument("--vision-url", help="远程推理服务地址")
    parser.add_argument("--vision-weights", help="YOLO 权重路径，例如 yolo11n.pt")
    parser.add_argument("--vision-conf", type=float, help="置信度阈值，默认 0.25")
    parser.add_argument("--verbose", action="store_true", help="打印访问日志")
    parser.add_argument(
        "--no-fallback-port", action="store_true", help="端口被占用时直接报错"
    )
    args = parser.parse_args()

    config = AppConfig.from_env()
    server_config = config.server
    if args.host:
        server_config = replace(server_config, host=args.host)
    if args.port:
        server_config = replace(server_config, port=args.port)

    llm = config.llm
    if args.provider:
        llm = replace(llm, provider=args.provider)
    if args.model:
        llm = replace(llm, model=args.model)

    vision = config.vision
    if args.vision_backend:
        vision = replace(vision, backend=args.vision_backend)
    if args.vision_url:
        vision = replace(vision, remote_url=args.vision_url)
    if args.vision_weights:
        vision = replace(vision, weights=args.vision_weights)
    if args.vision_conf is not None:
        vision = replace(vision, conf=args.vision_conf)

    config = AppConfig(llm=llm, vision=vision, server=server_config)

    if not args.no_fallback_port:
        config = replace(
            config,
            server=replace(
                config.server,
                port=find_free_port(config.server.host, config.server.port),
            ),
        )

    try:
        server = create_server(config, verbose=args.verbose)
    except OSError as exc:
        # 最常见的是端口被占用（Windows 上是 WinError 10048）
        print(f"[启动失败] 端口 {config.server.port} 无法绑定：{exc}")
        print("          可能已经有一个服务在跑，或者端口被别的程序占用。")
        print("          换一个端口试试：python run_server.py --port 8010")
        return 1
    serve_forever(server)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
