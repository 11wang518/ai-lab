#!/usr/bin/env python
"""大模型连通性自检：测真实模型的延迟、速度和流式输出。

先直接连模型后端测一遍，再通过运行中的 Web 服务测一遍，两边都通才算接通。

用法：
    python tools/check_llm.py
    python tools/check_llm.py --provider ollama --model qwen2.5:1.5b
    python tools/check_llm.py --url http://127.0.0.1:8000 --prompt "用一句话说明什么是目标检测"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import AppConfig  # noqa: E402
from app.llm import ChatMessage, ProviderError, build_provider  # noqa: E402

DEFAULT_PROMPT = "用三句话说明什么是目标检测，并说明它和图像分类的区别。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="大模型连通性自检")
    parser.add_argument("--provider", help="demo | ollama | openai")
    parser.add_argument("--model", help="模型名，例如 qwen2.5:1.5b")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="应用服务地址")
    parser.add_argument("--skip-http", action="store_true", help="只测模型后端，不测应用服务")
    return parser.parse_args()


def check_direct(config, prompt: str) -> bool:
    """直接调用模型后端。"""
    print("=" * 62)
    print(f"1. 直连模型后端：{config.provider} / {config.model}")
    print("=" * 62)
    try:
        provider = build_provider(config)
    except ProviderError as exc:
        print(f"  [失败] {exc}")
        return False

    messages = [
        ChatMessage(role="system", content=config.system_prompt),
        ChatMessage(role="user", content=prompt),
    ]
    started = time.perf_counter()
    first_token_at = None
    chunks: list[str] = []
    try:
        for piece in provider.stream(messages):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            chunks.append(piece)
            print(piece, end="", flush=True)
    except ProviderError as exc:
        print(f"\n  [失败] {exc}")
        return False
    except KeyboardInterrupt:
        print("\n  [中断]")
        return False

    total = time.perf_counter() - started
    text = "".join(chunks)
    print()
    if not text.strip():
        print("  [失败] 模型没有返回任何内容")
        return False

    first_latency = (first_token_at - started) if first_token_at else 0.0
    # 只有装了对应分词器才能精确数 token，这里如实报"字/秒"，
    # token 数按千问分词器的经验系数 0.65 折算，并标明是估算。
    generation_time = max(total - first_latency, 1e-6)
    chars_per_second = len(text) / generation_time
    print(f"\n  首 token 延迟 : {first_latency * 1000:.0f} ms")
    print(f"  总耗时        : {total:.2f} s")
    print(f"  输出长度      : {len(text)} 字")
    print(f"  生成速度      : {chars_per_second:.0f} 字/秒")
    print(f"  折算 token    : 约 {chars_per_second * 0.65:.0f} token/s（估算）")
    print("  [通过] 模型后端可用")
    return True


def check_http(url: str, prompt: str, provider: str, model: str) -> bool:
    """通过应用的 /api/chat 接口测一遍，验证前后端链路。"""
    print("\n" + "=" * 62)
    print(f"2. 通过应用服务：{url}/api/chat")
    print("=" * 62)
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "provider": provider,
        "model": model,
        "use_tools": False,
    }
    request = urllib.request.Request(
        f"{url.rstrip('/')}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
    )
    request.add_header("Content-Type", "application/json")

    started = time.perf_counter()
    deltas: list[str] = []
    event_name = ""
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                # 只取 delta 事件的增量文本；done 事件里的 text 是完整回答，
                # 如果也累加会把结果算成两遍。
                if event_name == "delta" and "text" in event:
                    deltas.append(event["text"])
                elif event_name == "error":
                    print(f"  [失败] {event}")
                    return False
    except urllib.error.URLError as exc:
        print(f"  [跳过] 连不上应用服务（{exc.reason}）。先用 python run_server.py 启动它。")
        return False
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        print(f"  [失败] HTTP {exc.code}：{detail}")
        return False

    elapsed = time.perf_counter() - started
    text = "".join(deltas)
    if not text.strip():
        print("  [失败] 接口没有返回文本")
        return False
    print(f"  收到 {len(deltas)} 个流式分片，共 {len(text)} 字，耗时 {elapsed:.2f} s")
    print(f"  回答：{text[:80]}{'…' if len(text) > 80 else ''}")
    print("  [通过] 应用服务链路可用")
    return True


def main() -> int:
    args = parse_args()
    config = AppConfig.from_env().llm
    if args.provider:
        config = replace(config, provider=args.provider)
    if args.model:
        config = replace(config, model=args.model)

    print(f"提示词：{args.prompt}\n")
    direct_ok = check_direct(config, args.prompt)

    http_ok = False
    if not args.skip_http:
        http_ok = check_http(args.url, args.prompt, config.provider, config.model)

    print("\n" + "=" * 62)
    print(f"模型后端：{'通过' if direct_ok else '未通过'}")
    if not args.skip_http:
        print(f"应用服务：{'通过' if http_ok else '未通过'}")
    print("=" * 62)
    return 0 if direct_ok and (args.skip_http or http_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
