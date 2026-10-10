"""一个足够小但结构完整的智能体。

设计取舍：本地小模型（2B 级）对 function calling 的支持参差不齐，硬上官方
tool-calling 协议很容易失败。这里采用「本地意图识别 + 工具执行 + 把结果交回模型」
的两段式方案，任何模型都能用。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from ..llm.base import ChatMessage, ChatProvider, ProviderError
from . import tools as tool_module

_ARITH_RE = re.compile(r"\d[\d\s.,+\-*/×÷^()]*")


@dataclass
class AgentTurn:
    """一次智能体调用的完整结果，便于日志与前端展示。"""

    text: str = ""
    tool_name: str = ""
    tool_arguments: dict = field(default_factory=dict)
    tool_result: dict | None = None
    steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "tool_name": self.tool_name,
            "tool_arguments": self.tool_arguments,
            "tool_result": self.tool_result,
            "steps": self.steps,
        }


def _plan(user_text: str) -> tuple[str, dict] | None:
    """从用户输入里判断要不要调用工具。"""
    text = user_text.strip()
    if not text:
        return None

    if any(word in text for word in ("几点", "现在时间", "当前时间", "今天几号", "今天是")):
        return "current_time", {"timezone_label": "Asia/Shanghai"}

    if any(word in text for word in ("字数", "多少个字", "统计一下", "几行")):
        return "text_stats", {"text": text}

    if any(word in text for word in ("算一下", "计算", "等于多少", "帮我算")):
        match = _ARITH_RE.search(text)
        if match:
            return "calculator", {"expression": match.group(0).strip()}

    # 没有任何提示词，但整句就是一个算式
    if re.fullmatch(r"[\d\s.,+\-*/×÷^()]+", text) and any(
        symbol in text for symbol in "+-*/×÷^"
    ):
        return "calculator", {"expression": text}

    return None


class AgentRunner:
    def __init__(self, provider: ChatProvider, *, enable_tools: bool = True) -> None:
        self.provider = provider
        self.enable_tools = enable_tools

    def run(
        self,
        messages: Sequence[ChatMessage],
        *,
        use_tools: bool | None = None,
        **overrides,
    ) -> Iterator[tuple[str, dict | None]]:
        """流式执行一轮对话。

        产出 (事件名, 数据) 二元组，事件名取值：step / tool / delta / done。
        """
        enabled = self.enable_tools if use_tools is None else use_tools
        turn = AgentTurn()
        chat_messages = list(messages)

        if enabled:
            last_user = next(
                (m.content for m in reversed(chat_messages) if m.role == "user"), ""
            )
            plan = _plan(last_user)
            if plan is not None:
                name, arguments = plan
                turn.tool_name = name
                turn.tool_arguments = arguments
                yield "step", {"message": f"准备调用工具 {name}"}
                try:
                    result = tool_module.call_tool(name, arguments)
                    turn.tool_result = result
                    yield "tool", {"name": name, "arguments": arguments, "result": result}
                    chat_messages.append(
                        ChatMessage(
                            role="system",
                            content=(
                                "工具调用结果如下，请用自然语言回答用户，"
                                f"不要输出 JSON：\n{json.dumps(result, ensure_ascii=False)}"
                            ),
                        )
                    )
                except Exception as exc:  # 工具失败不应该中断整轮对话
                    turn.tool_result = {"error": str(exc)}
                    yield "tool", {
                        "name": name,
                        "arguments": arguments,
                        "result": turn.tool_result,
                    }
                    yield "step", {"message": f"工具执行失败：{exc}"}

        try:
            collected: list[str] = []
            for piece in self.provider.stream(chat_messages, **overrides):
                collected.append(piece)
                yield "delta", {"text": piece}
            turn.text = "".join(collected)
        except ProviderError as exc:
            yield "error", {"message": str(exc)}
            return

        yield "done", turn.to_dict()
