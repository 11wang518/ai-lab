"""模型接入的统一接口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence


class ProviderError(RuntimeError):
    """模型服务不可用或返回异常时抛出，消息会直接展示给用户。"""


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_any(cls, item: Any) -> "ChatMessage":
        if isinstance(item, ChatMessage):
            return item
        if isinstance(item, dict):
            role = str(item.get("role", "user"))
            content = item.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            return cls(role=role, content=content)
        raise TypeError(f"无法识别的消息格式: {type(item)!r}")


def normalize_messages(messages: Iterable[Any]) -> list[ChatMessage]:
    return [ChatMessage.from_any(item) for item in messages]


class ChatProvider:
    """所有模型后端的基类。"""

    name = "base"
    label = "Base"
    model = ""
    supports_tools = False

    def stream(self, messages: Sequence[ChatMessage], **overrides: Any) -> Iterator[str]:
        raise NotImplementedError

    def complete(self, messages: Sequence[ChatMessage], **overrides: Any) -> str:
        return "".join(self.stream(messages, **overrides))

    def describe(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "model": self.model,
            "supports_tools": self.supports_tools,
        }
