"""模型后端实现：内置示例 / Ollama / OpenAI 兼容接口。"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Iterator, Sequence

from ..config import LLMConfig
from .base import ChatMessage, ChatProvider, ProviderError


def _post_json(url: str, payload: dict, headers: dict, timeout: float):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json; charset=utf-8")
    request.add_header("Accept", "application/json, text/event-stream")
    for key, value in headers.items():
        request.add_header(key, value)
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:  # 服务返回了 4xx/5xx
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:500]
        except Exception:  # pragma: no cover - 读取失败不影响主流程
            detail = ""
        raise ProviderError(
            f"模型服务返回 HTTP {exc.code}：{detail or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"无法连接模型服务：{exc.reason}") from exc
    except TimeoutError as exc:
        raise ProviderError("模型服务响应超时，请检查服务状态或加大超时时间。") from exc


class DemoProvider(ChatProvider):
    """内置示例模型。

    不依赖任何外部服务，用于在没有部署大模型时验证「前端 → 后端 → 流式返回」的链路。
    它只做规则匹配，不具备真实语言能力，界面上会明确标注。
    """

    name = "demo"
    label = "内置示例模型"
    supports_tools = False

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.model = "demo-rule-based"
        self._delay = max(0.0, config.demo_chunk_delay)

    def _compose(self, messages: Sequence[ChatMessage]) -> str:
        tool_payload = None
        for message in messages:
            if message.role == "system" and message.content.startswith("工具调用结果如下"):
                _, _, raw = message.content.partition("\n")
                try:
                    tool_payload = json.loads(raw.strip())
                except json.JSONDecodeError:
                    tool_payload = None
                break
        if isinstance(tool_payload, dict):
            if "value" in tool_payload:
                return (
                    f"计算结果是 {tool_payload['value']}。\n\n"
                    "这个数由本机 `calculator` 工具算出，模型只负责把结果说清楚，不参与计算。"
                )
            if "readable" in tool_payload:
                return f"当前时间是 {tool_payload['readable']}，星期{tool_payload.get('weekday', '')}。"
            if "characters" in tool_payload:
                return (
                    f"这段文本共 {tool_payload['characters']} 个字符、"
                    f"{tool_payload['lines']} 行，其中中文 {tool_payload['chinese_characters']} 个字。"
                )

        user_text = ""
        for message in reversed(list(messages)):
            if message.role == "user":
                user_text = message.content.strip()
                break
        topic = user_text[:40] if user_text else "你的问题"
        lowered = user_text.lower()

        if any(word in user_text for word in ("你好", "您好", "hi", "hello")):
            return (
                "你好，我是内置示例模型，用来验证应用链路是否通。\n\n"
                "要接真实模型，可以任选一种：\n"
                "1. 本机装好 Ollama 后运行 `ollama pull qwen2.5:7b`，"
                "再用 `AILAB_LLM_PROVIDER=ollama` 启动服务。\n"
                "2. 打开 LM Studio 加载模型并开启本地服务器，"
                "把 `AILAB_OPENAI_BASE_URL` 指向它的 `/v1`。\n"
                "3. 用 Dify、扣子等平台发布应用，把 OpenAI 兼容地址填进同一个变量。"
            )
        if "你是谁" in user_text or "什么模型" in user_text:
            return (
                "我是项目自带的规则型示例模型，不是真正的大模型。\n"
                "它的作用只有一个：在模型还没部署好之前，"
                "让聊天界面、流式输出、接口协议都能先跑起来。"
            )
        if any(word in user_text for word in ("yolo", "检测", "识别")) and any(
            word in user_text for word in ("怎么", "如何", "使用", "跑")
        ):
            return (
                "检测面板的处理流程是：\n\n"
                "1. 图片上传到 `POST /api/detect`。\n"
                "2. 后端按配置选择推理后端：本机 `ultralytics` → 远程推理服务 → 不可用提示。\n"
                "3. 返回统一 JSON（`boxes` 数组，含坐标、类别、置信度），前端画框。\n\n"
                "要跑真实模型，先执行 `pip install ultralytics`，"
                "或者用 `python yolo/serve.py` 单独起一个推理服务再配 `AILAB_YOLO_REMOTE_URL`。"
            )
        return (
            f"关于「{topic}」，当前回复由内置示例模型生成，因此内容只用于演示链路。\n\n"
            "这条消息是流式返回的，说明前后端接口是通的。"
            "接上真实模型后，这里就会变成模型自己的回答。"
        )

    def stream(self, messages: Sequence[ChatMessage], **overrides: Any) -> Iterator[str]:
        text = self._compose(messages)
        chunk_size = 6
        for index in range(0, len(text), chunk_size):
            if self._delay:
                time.sleep(self._delay)
            yield text[index : index + chunk_size]


class OllamaProvider(ChatProvider):
    """通过 Ollama 的 /api/chat 接口流式对话。"""

    name = "ollama"
    label = "Ollama 本地模型"
    supports_tools = False

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.model = config.model or "qwen2.5:7b"
        self._host = config.ollama_host.rstrip("/")

    def stream(self, messages: Sequence[ChatMessage], **overrides: Any) -> Iterator[str]:
        payload = {
            "model": overrides.get("model", self.model),
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "options": {
                "temperature": float(overrides.get("temperature", self.config.temperature))
            },
        }
        response = _post_json(
            f"{self._host}/api/chat", payload, {}, self.config.request_timeout
        )
        with response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("error"):
                    raise ProviderError(f"Ollama 返回错误：{event['error']}")
                piece = (event.get("message") or {}).get("content", "")
                if piece:
                    yield piece
                if event.get("done"):
                    break


class OpenAICompatProvider(ChatProvider):
    """任何 OpenAI 兼容的 /chat/completions 接口。"""

    name = "openai"
    label = "OpenAI 兼容接口"
    supports_tools = True

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.model = config.model or "local-model"
        self._base = config.openai_base_url.rstrip("/")

    def _headers(self) -> dict:
        headers = {"Authorization": f"Bearer {self.config.openai_api_key}"}
        return headers

    def stream(self, messages: Sequence[ChatMessage], **overrides: Any) -> Iterator[str]:
        payload = {
            "model": overrides.get("model", self.model),
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "temperature": float(overrides.get("temperature", self.config.temperature)),
            "max_tokens": int(overrides.get("max_tokens", self.config.max_tokens)),
        }
        response = _post_json(
            f"{self._base}/chat/completions", payload, self._headers(), self.config.request_timeout
        )
        with response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    yield piece
