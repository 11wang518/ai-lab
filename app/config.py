"""集中配置。

所有配置都可以用环境变量覆盖，默认值保证"零依赖 + 无模型"时应用依然能启动。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

DEFAULT_SYSTEM_PROMPT = (
    "你是一个部署在本地设备上的中文助手。回答要准确、简洁，"
    "不确定的地方要直说，不要编造数据。"
)


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class LLMConfig:
    """大模型接入配置。

    provider:
        demo     内置示例模型，无需任何外部服务，用于验证应用链路。
        ollama   本地 Ollama 服务 (http://127.0.0.1:11434)。
        openai   任何 OpenAI 兼容接口：LM Studio / vLLM / Dify / 扣子 / 硅基流动等。
    """

    provider: str = "demo"
    model: str = "qwen2.5:1.5b"
    ollama_host: str = "http://127.0.0.1:11434"
    openai_base_url: str = "http://127.0.0.1:1234/v1"
    openai_api_key: str = "not-needed"
    temperature: float = 0.7
    max_tokens: int = 1024
    request_timeout: float = 300.0
    demo_chunk_delay: float = 0.012
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            provider=_env("AILAB_LLM_PROVIDER", "demo").lower(),
            model=_env("AILAB_LLM_MODEL", "qwen2.5:1.5b"),
            ollama_host=_env("AILAB_OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/"),
            openai_base_url=_env("AILAB_OPENAI_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/"),
            openai_api_key=_env("AILAB_OPENAI_API_KEY", "not-needed"),
            temperature=_env_float("AILAB_LLM_TEMPERATURE", 0.7),
            max_tokens=_env_int("AILAB_LLM_MAX_TOKENS", 1024),
            request_timeout=_env_float("AILAB_LLM_TIMEOUT", 300.0),
            demo_chunk_delay=_env_float("AILAB_DEMO_DELAY", 0.012),
            system_prompt=_env("AILAB_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        )


@dataclass
class VisionConfig:
    """YOLO 推理配置。

    backend:
        auto         优先本地 ultralytics，其次远程推理服务，都不可用时返回提示。
        ultralytics  强制使用本机安装的 ultralytics（需要 torch）。
        remote       强制调用 yolo/serve.py 之类的远程推理服务。
        none         关闭检测，仅保留前端演示能力。
    """

    backend: str = "auto"
    weights: str = "yolo11n.pt"
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    device: str = ""
    max_det: int = 100
    remote_url: str = ""
    remote_timeout: float = 60.0
    max_upload_bytes: int = 20 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "VisionConfig":
        return cls(
            backend=_env("AILAB_YOLO_BACKEND", "auto").lower(),
            weights=_env("AILAB_YOLO_WEIGHTS", "yolo11n.pt"),
            conf=_env_float("AILAB_YOLO_CONF", 0.25),
            iou=_env_float("AILAB_YOLO_IOU", 0.45),
            imgsz=_env_int("AILAB_YOLO_IMGSZ", 640),
            device=_env("AILAB_YOLO_DEVICE", ""),
            max_det=_env_int("AILAB_YOLO_MAX_DET", 100),
            remote_url=_env("AILAB_YOLO_REMOTE_URL", ""),
            remote_timeout=_env_float("AILAB_YOLO_REMOTE_TIMEOUT", 60.0),
            max_upload_bytes=_env_int("AILAB_MAX_UPLOAD_BYTES", 20 * 1024 * 1024),
        )


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def from_env(cls) -> "ServerConfig":
        return cls(
            host=_env("AILAB_HOST", "127.0.0.1"),
            port=_env_int("AILAB_PORT", 8000),
        )


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            llm=LLMConfig.from_env(),
            vision=VisionConfig.from_env(),
            server=ServerConfig.from_env(),
        )

    def public(self) -> dict:
        """可以安全发给前端的配置，不含密钥。"""
        data = asdict(self)
        data["llm"]["openai_api_key"] = "***" if self.llm.openai_api_key else ""
        return data
