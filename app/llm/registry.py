"""根据配置构造模型后端。"""

from __future__ import annotations

from ..config import LLMConfig
from .base import ChatProvider, ProviderError
from .providers import DemoProvider, OllamaProvider, OpenAICompatProvider

PROVIDER_LABELS = {
    "demo": "内置示例模型",
    "ollama": "Ollama 本地模型",
    "openai": "OpenAI 兼容接口",
}


def provider_catalog() -> list[dict]:
    return [
        {
            "name": "demo",
            "label": PROVIDER_LABELS["demo"],
            "hint": "无需外部服务，用于验证前端到后端的完整链路。",
        },
        {
            "name": "ollama",
            "label": PROVIDER_LABELS["ollama"],
            "hint": "先运行 ollama serve，再 ollama pull qwen2.5:7b。",
        },
        {
            "name": "openai",
            "label": PROVIDER_LABELS["openai"],
            "hint": "LM Studio / vLLM / Dify / 扣子 等 OpenAI 兼容端点。",
        },
    ]


def build_provider(config: LLMConfig) -> ChatProvider:
    name = (config.provider or "demo").lower()
    if name == "demo":
        return DemoProvider(config)
    if name == "ollama":
        return OllamaProvider(config)
    if name in ("openai", "openai-compatible", "lmstudio", "vllm", "dify"):
        return OpenAICompatProvider(config)
    raise ProviderError(
        f"未知的模型后端 {name!r}，可用：{', '.join(PROVIDER_LABELS)}"
    )
