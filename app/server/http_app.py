"""基于标准库的 HTTP 服务：静态页面 + 聊天 SSE + 检测接口。"""

from __future__ import annotations

import json
import mimetypes
import socket
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .. import __version__
from ..agent import AgentRunner, describe_tools
from ..config import AppConfig
from ..llm import ChatMessage, ProviderError, build_provider, provider_catalog
from ..vision import Detector, DetectorError
from .multipart import parse_multipart

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


class Application:
    """把配置、模型后端、检测器组装到一起。"""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig.from_env()
        self.detector = Detector(self.config.vision)
        self._provider_cache: dict[str, object] = {}
        self.started_at = time.time()

    # 模型后端按 (provider, model, base_url) 缓存，避免每次请求重建
    def provider_for(self, provider: str | None = None, model: str | None = None):
        llm_config = self.config.llm
        if provider:
            llm_config = replace(llm_config, provider=provider)
        if model:
            llm_config = replace(llm_config, model=model)
        key = f"{llm_config.provider}|{llm_config.model}|{llm_config.openai_base_url}"
        cached = self._provider_cache.get(key)
        if cached is None:
            cached = build_provider(llm_config)
            self._provider_cache[key] = cached
        return cached

    def health(self) -> dict:
        provider = self.provider_for()
        return {
            "ok": True,
            "version": __version__,
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "llm": {
                "provider": provider.name,
                "model": provider.model,
                "label": provider.label,
                "live": provider.name in ("ollama", "openai"),
            },
            "vision": self.detector.capabilities(),
        }

    def client_config(self) -> dict:
        data = self.config.public()
        data["version"] = __version__
        data["providers"] = provider_catalog()
        data["tools"] = describe_tools()
        data["vision"] = self.detector.capabilities()
        return data


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"ai-lab/{__version__}"
    app: Application  # 由 create_server 注入

    # ---------- 基础工具 ----------

    def log_message(self, fmt: str, *args) -> None:  # 控制台保持安静
        if self.server.verbose:  # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message: str, status: int = 400) -> None:
        self._send_json({"ok": False, "error": message}, status=status)

    def _read_body(self, limit: int) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return b""
        if length > limit:
            raise ValueError(f"请求体过大（{length} 字节），上限 {limit} 字节")
        return self.rfile.read(length)

    # ---------- SSE ----------

    def _sse_start(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self._cors()
        self.end_headers()

    def _sse_send(self, event: str, data: dict) -> bool:
        payload = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        raw = payload.encode("utf-8")
        try:
            self.wfile.write(f"{len(raw):X}\r\n".encode("ascii"))
            self.wfile.write(raw)
            self.wfile.write(b"\r\n")
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return False

    def _sse_end(self) -> None:
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass

    # ---------- 路由 ----------

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json(self.app.health())
            return
        if path == "/api/config":
            self._send_json(self.app.client_config())
            return
        if path == "/api/samples":
            self._send_json({"samples": self._list_samples()})
            return
        self._serve_static(path)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/chat":
                self._handle_chat()
                return
            if path == "/api/detect":
                self._handle_detect()
                return
        except ValueError as exc:
            self._send_error_json(str(exc), status=413)
            return
        self._send_error_json(f"未知接口：{path}", status=404)

    # ---------- 具体处理 ----------

    def _handle_chat(self) -> None:
        try:
            raw = self._read_body(2 * 1024 * 1024)
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_error_json(f"请求体不是合法 JSON：{exc}")
            return

        messages = [
            ChatMessage.from_any(item) for item in payload.get("messages", [])
        ]
        if not messages:
            self._send_error_json("messages 不能为空")
            return

        system_prompt = self.app.config.llm.system_prompt
        if system_prompt and not any(m.role == "system" for m in messages):
            messages.insert(0, ChatMessage(role="system", content=system_prompt))

        try:
            provider = self.app.provider_for(
                payload.get("provider"), payload.get("model")
            )
        except ProviderError as exc:
            self._send_error_json(str(exc))
            return

        overrides = {}
        if "temperature" in payload:
            overrides["temperature"] = payload["temperature"]
        if "max_tokens" in payload:
            overrides["max_tokens"] = payload["max_tokens"]

        runner = AgentRunner(provider)
        self._sse_start()
        started = time.perf_counter()
        self._sse_send("meta", {"provider": provider.name, "model": provider.model})
        for event, data in runner.run(
            messages, use_tools=payload.get("use_tools"), **overrides
        ):
            if not self._sse_send(event, data):
                break
            if event == "done":
                self._sse_send(
                    "stats",
                    {
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                        "provider": provider.name,
                        "model": provider.model,
                    },
                )
        self._sse_end()

    def _handle_detect(self) -> None:
        limit = self.app.config.vision.max_upload_bytes
        raw = self._read_body(limit)
        content_type = self.headers.get("Content-Type", "")
        filename = "upload.jpg"
        data = raw

        if content_type.startswith("multipart/form-data"):
            boundary = ""
            for part in content_type.split(";"):
                part = part.strip()
                if part.startswith("boundary="):
                    boundary = part[len("boundary=") :].strip('"')
            parsed = parse_multipart(raw, boundary)
            files = parsed["files"]
            if not files:
                self._send_error_json("没有在请求里找到文件字段")
                return
            first = next(iter(files.values()))
            data = first["data"]
            filename = first["filename"]

        if not data:
            self._send_error_json("图片内容为空")
            return

        try:
            result = self.app.detector.detect(data, filename)
        except DetectorError as exc:
            self._send_error_json(str(exc), status=503)
            return
        self._send_json({"ok": True, "filename": filename, **result.to_dict()})

    # ---------- 静态资源 ----------

    def _list_samples(self) -> list[dict]:
        sample_dir = WEB_DIR / "samples"
        if not sample_dir.is_dir():
            return []
        items = []
        for path in sorted(sample_dir.iterdir()):
            if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                items.append({"name": path.name, "url": f"/samples/{path.name}"})
        return items

    def _serve_static(self, path: str) -> None:
        relative = unquote(path).lstrip("/") or "index.html"
        target = (WEB_DIR / relative).resolve()
        try:
            target.relative_to(WEB_DIR.resolve())
        except ValueError:
            self._send_error_json("非法路径", status=403)
            return
        if target.is_dir():
            target = target / "index.html"
        if not target.is_file():
            self._send_error_json(f"找不到资源：{relative}", status=404)
            return
        data = target.read_bytes()
        mime, _ = mimetypes.guess_type(target.name)
        self.send_response(200)
        self.send_header("Content-Type", f"{mime or 'application/octet-stream'}")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(data)


class LabServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, app: Application, verbose: bool = False):
        self.app = app
        self.verbose = verbose
        handler.app = app
        super().__init__(address, handler)

    def handle_error(self, request, client_address) -> None:
        # 浏览器/命令行客户端提前断开是正常现象，不必打印整页堆栈。
        import sys

        error = sys.exc_info()[1]
        if isinstance(error, (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def create_server(
    config: AppConfig | None = None, *, verbose: bool = False
) -> LabServer:
    app = Application(config)
    server = LabServer((app.config.server.host, app.config.server.port), Handler, app, verbose)
    return server


def serve_forever(server: LabServer) -> None:
    host, port = server.server_address[0], server.server_address[1]
    print(f"[ai-lab] 服务已启动: http://{host}:{port}")
    print(f"[ai-lab] 模型后端: {server.app.health()['llm']['provider']}"
          f" ({server.app.health()['llm']['model']})")
    print(f"[ai-lab] 检测后端: {server.app.health()['vision']['active_backend']}")
    print("[ai-lab] 按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ai-lab] 已停止")
    finally:
        server.server_close()


def find_free_port(host: str, start: int, attempts: int = 20) -> int:
    for offset in range(attempts):
        port = start + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"从 {start} 开始的 {attempts} 个端口都被占用")


def serve_in_background(server: LabServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread
