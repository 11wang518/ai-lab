"""命令行对话客户端（应用形式之一：CLI）。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from .agent import AgentRunner
from .config import AppConfig
from .llm import ChatMessage, ProviderError, build_provider

BANNER = """
========================================
 AI Lab 命令行助手
 模型后端 : {provider} / {model}
 工具调用 : {tools}
 输入 /help 查看命令，/exit 退出
========================================
"""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地大模型命令行对话客户端")
    parser.add_argument("--provider", help="demo | ollama | openai")
    parser.add_argument("--model", help="模型名，例如 qwen2.5:1.5b")
    parser.add_argument("--once", help="只问一个问题就退出")
    parser.add_argument("--no-tools", action="store_true", help="关闭工具调用")
    parser.add_argument("--no-stream", action="store_true", help="等全部生成完再输出")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = AppConfig.from_env()
    llm_config = config.llm
    if args.provider:
        llm_config = replace(llm_config, provider=args.provider)
    if args.model:
        llm_config = replace(llm_config, model=args.model)

    try:
        provider = build_provider(llm_config)
    except ProviderError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    runner = AgentRunner(provider, enable_tools=not args.no_tools)
    labels = provider_catalog_labels(provider.name)
    if not args.once:
        print(
            BANNER.format(
                provider=labels,
                model=provider.model,
                tools="开启" if not args.no_tools else "关闭",
            )
        )

    history: list[ChatMessage] = [
        ChatMessage(role="system", content=llm_config.system_prompt)
    ]

    def ask(text: str) -> None:
        history.append(ChatMessage(role="user", content=text))
        answer: list[str] = []
        try:
            for event, data in runner.run(history):
                if event == "step":
                    print(f"  · {data['message']}", flush=True)
                elif event == "tool":
                    print(f"  · 工具结果: {data['result']}", flush=True)
                elif event == "delta":
                    answer.append(data["text"])
                    if not args.no_stream:
                        print(data["text"], end="", flush=True)
                elif event == "error":
                    print(f"\n[错误] {data['message']}", file=sys.stderr)
                    return
        except KeyboardInterrupt:
            print("\n[已中断本轮生成]")
        if args.no_stream:
            print("".join(answer))
        else:
            print()
        history.append(ChatMessage(role="assistant", content="".join(answer)))

    if args.once:
        ask(args.once)
        return 0

    while True:
        try:
            text = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/exit", "/quit", ":q"):
            break
        if text == "/help":
            print("命令：/exit 退出 | /clear 清空上下文 | /tools 查看可用工具")
            continue
        if text == "/clear":
            history = [ChatMessage(role="system", content=llm_config.system_prompt)]
            print("上下文已清空")
            continue
        if text == "/tools":
            from .agent import describe_tools

            for tool in describe_tools():
                print(f"  {tool['name']}: {tool['description']}")
            continue
        print("助手 > ", end="", flush=True)
        ask(text)
    return 0


def provider_catalog_labels(name: str) -> str:
    return {"demo": "内置示例", "ollama": "Ollama", "openai": "OpenAI 兼容"}.get(name, name)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
