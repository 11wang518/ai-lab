"""目标检测后端。

三种运行方式，按顺序自动选择：

1. ultralytics   —— 本机安装，进程内推理（速度最快，推荐）。
2. remote        —— 调用 yolo/serve.py 或任何返回统一 JSON 的推理服务。
3. none          —— 两者都不可用，返回明确提示，前端会自动降级到浏览器端演示。
"""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field

from ..config import VisionConfig
from ..env_setup import prepare as prepare_ultralytics_env


class DetectorError(RuntimeError):
    """推理后端不可用或推理失败。"""


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str
    cls: int

    def to_dict(self) -> dict:
        data = asdict(self)
        data["score"] = round(self.score, 4)
        for key in ("x1", "y1", "x2", "y2"):
            data[key] = round(data[key], 2)
        return data


@dataclass
class DetectionResult:
    engine: str
    boxes: list[Box] = field(default_factory=list)
    width: int = 0
    height: int = 0
    elapsed_ms: float = 0.0
    model: str = ""
    note: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "model": self.model,
            "boxes": [box.to_dict() for box in self.boxes],
            "width": self.width,
            "height": self.height,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "note": self.note,
            "meta": self.meta,
        }


def _http_post_multipart(url: str, data: bytes, filename: str, timeout: float) -> dict:
    boundary = "----ailab" + uuid.uuid4().hex
    crlf = b"\r\n"
    body = io.BytesIO()
    body.write(b"--" + boundary.encode() + crlf)
    body.write(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode("utf-8")
        + crlf
    )
    body.write(b"Content-Type: application/octet-stream" + crlf + crlf)
    body.write(data)
    body.write(crlf + b"--" + boundary.encode() + b"--" + crlf)
    payload = body.getvalue()

    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    request.add_header("Content-Length", str(len(payload)))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise DetectorError(f"远程推理服务返回 HTTP {exc.code}：{detail}") from exc
    except urllib.error.URLError as exc:
        raise DetectorError(f"无法连接远程推理服务：{exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise DetectorError("远程推理服务返回的不是合法 JSON") from exc


class Detector:
    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self._model = None
        self._model_key = ""
        self._ultralytics_error = ""

    # ---------- 能力探测 ----------

    def _has_ultralytics(self) -> bool:
        prepare_ultralytics_env()
        try:
            import ultralytics  # noqa: F401
        except Exception as exc:  # 记录原因，便于界面上解释
            self._ultralytics_error = f"{type(exc).__name__}: {exc}"
            return False
        return True

    def resolve_backend(self) -> str:
        backend = (self.config.backend or "auto").lower()
        if backend == "auto":
            if self._has_ultralytics():
                return "ultralytics"
            if self.config.remote_url:
                return "remote"
            return "none"
        if backend == "ultralytics":
            return "ultralytics" if self._has_ultralytics() else "none"
        if backend == "remote":
            return "remote" if self.config.remote_url else "none"
        return "none"

    def capabilities(self) -> dict:
        backend = self.resolve_backend()
        note = ""
        if backend == "none":
            if self.config.backend == "remote" and not self.config.remote_url:
                note = "已指定远程后端，但没有配置 AILAB_YOLO_REMOTE_URL。"
            elif self.config.backend == "ultralytics" and self._ultralytics_error:
                note = f"ultralytics 不可用（{self._ultralytics_error}）。"
            else:
                note = (
                    "未检测到可用的 YOLO 推理后端。安装 ultralytics 或配置远程服务后即可启用。"
                )
        return {
            "configured_backend": self.config.backend,
            "active_backend": backend,
            "weights": self.config.weights,
            "conf": self.config.conf,
            "imgsz": self.config.imgsz,
            "remote_url": self.config.remote_url,
            "note": note,
        }

    # ---------- 推理 ----------

    def _load_model(self):
        key = f"{self.config.weights}|{self.config.device}"
        if self._model is not None and self._model_key == key:
            return self._model
        prepare_ultralytics_env()
        from ultralytics import YOLO  # 延迟导入，避免无依赖时启动失败

        self._model = YOLO(self.config.weights)
        self._model_key = key
        return self._model

    def _detect_ultralytics(self, data: bytes) -> DetectionResult:
        model = self._load_model()
        from PIL import Image  # ultralytics 依赖链里一定会有 Pillow

        image = Image.open(io.BytesIO(data))
        image.load()
        results = model.predict(
            source=image,
            conf=self.config.conf,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            max_det=self.config.max_det,
            device=self.config.device or None,
            verbose=False,
        )
        boxes: list[Box] = []
        for result in results:
            names = getattr(result, "names", {}) or {}
            for raw in result.boxes:
                cls_id = int(raw.cls[0].item())
                boxes.append(
                    Box(
                        x1=float(raw.xyxy[0][0]),
                        y1=float(raw.xyxy[0][1]),
                        x2=float(raw.xyxy[0][2]),
                        y2=float(raw.xyxy[0][3]),
                        score=float(raw.conf[0].item()),
                        label=str(names.get(cls_id, cls_id)),
                        cls=cls_id,
                    )
                )
        return DetectionResult(
            engine="ultralytics",
            boxes=boxes,
            width=image.width,
            height=image.height,
            model=self.config.weights,
            note="本机 ultralytics 进程内推理。",
            meta={"device": self.config.device or "auto"},
        )

    def _detect_remote(self, data: bytes, filename: str) -> DetectionResult:
        payload = _http_post_multipart(
            self.config.remote_url, data, filename, self.config.remote_timeout
        )
        boxes = [
            Box(
                x1=float(item["x1"]),
                y1=float(item["y1"]),
                x2=float(item["x2"]),
                y2=float(item["y2"]),
                score=float(item.get("score", 0.0)),
                label=str(item.get("label", item.get("cls", "object"))),
                cls=int(item.get("cls", 0)),
            )
            for item in payload.get("boxes", [])
        ]
        return DetectionResult(
            engine="remote",
            boxes=boxes,
            width=int(payload.get("width", 0)),
            height=int(payload.get("height", 0)),
            model=str(payload.get("model", "remote")),
            note=f"由远程推理服务处理：{self.config.remote_url}",
            meta={"remote": self.config.remote_url},
        )

    def detect(self, data: bytes, filename: str = "upload.jpg") -> DetectionResult:
        started = time.perf_counter()
        backend = self.resolve_backend()
        if backend == "none":
            result = DetectionResult(
                engine="none",
                model="",
                note=self.capabilities()["note"],
            )
            result.elapsed_ms = (time.perf_counter() - started) * 1000
            return result
        try:
            if backend == "ultralytics":
                result = self._detect_ultralytics(data)
            else:
                result = self._detect_remote(data, filename)
        except DetectorError:
            raise
        except Exception as exc:
            raise DetectorError(f"推理失败：{type(exc).__name__}: {exc}") from exc
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result
