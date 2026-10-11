#!/usr/bin/env python
"""独立的 YOLO 推理服务（返回统一 JSON），供主应用远程调用。

为什么单独起一个服务：主应用是零依赖的，而 ultralytics 依赖 torch。
两者用 HTTP 解耦后，主应用可以放在任何机器上，推理放到有显卡的机器上。

启动：
    python yolo/serve.py --weights yolo11n.pt --port 8600

然后让主应用指向它：
    set AILAB_YOLO_REMOTE_URL=http://127.0.0.1:8600/predict
    python run_server.py

接口：
    GET  /health          -> 服务状态
    POST /predict         -> multipart 上传 file 字段，返回 boxes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.env_setup import prepare  # noqa: E402
from app.server.multipart import parse_multipart  # noqa: E402

prepare()  # 必须在 import ultralytics 之前

MODEL = None
CONF = 0.25
IOU = 0.45
IMGSZ = 640
DEVICE = "0"
MAX_BYTES = 20 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ailab-yolo/1.0"

    def log_message(self, fmt, *args):
        print(f"[yolo] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/health"):
            names = getattr(MODEL, "names", {}) or {}
            self._json(
                {
                    "ok": True,
                    "model": str(getattr(MODEL, "ckpt_path", "")),
                    "classes": len(names),
                    "imgsz": IMGSZ,
                    "conf": CONF,
                }
            )
            return
        self._json({"ok": False, "error": "未知路径"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.startswith("/predict"):
            self._json({"ok": False, "error": "未知路径"}, status=404)
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BYTES:
            self._json({"ok": False, "error": "请求体为空或过大"}, status=400)
            return
        body = self.rfile.read(length)
        content_type = self.headers.get("Content-Type", "")
        boundary = ""
        for part in content_type.split(";"):
            part = part.strip()
            if part.startswith("boundary="):
                boundary = part[len("boundary=") :].strip('"')
        parsed = parse_multipart(body, boundary)
        if not parsed["files"]:
            self._json({"ok": False, "error": "缺少 file 字段"}, status=400)
            return
        first = next(iter(parsed["files"].values()))

        try:
            payload = run_inference(first["data"])
        except Exception as exc:  # noqa: BLE001 - 服务端需要把错误原样返回
            self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)
            return
        self._json({"ok": True, **payload})


def run_inference(data: bytes) -> dict:
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(data))
    image.load()
    started = time.perf_counter()
    results = MODEL.predict(
        source=image,
        conf=CONF,
        iou=IOU,
        imgsz=IMGSZ,
        device=DEVICE or None,
        verbose=False,
    )
    boxes = []
    for result in results:
        names = getattr(result, "names", {}) or {}
        for raw in result.boxes:
            cls_id = int(raw.cls[0].item())
            xyxy = raw.xyxy[0].tolist()
            boxes.append(
                {
                    "label": str(names.get(cls_id, cls_id)),
                    "cls": cls_id,
                    "score": round(float(raw.conf[0].item()), 4),
                    "x1": round(xyxy[0], 2),
                    "y1": round(xyxy[1], 2),
                    "x2": round(xyxy[2], 2),
                    "y2": round(xyxy[3], 2),
                }
            )
    return {
        "model": str(getattr(MODEL, "ckpt_path", "yolo")),
        "engine": "ultralytics-service",
        "width": image.width,
        "height": image.height,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "boxes": boxes,
    }


def main(argv: list[str] | None = None) -> int:
    global MODEL, CONF, IOU, IMGSZ, DEVICE

    parser = argparse.ArgumentParser(description="YOLO 推理服务")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("没有找到 ultralytics，请先 pip install ultralytics", file=sys.stderr)
        return 2

    CONF, IOU, IMGSZ, DEVICE = args.conf, args.iou, args.imgsz, args.device
    MODEL = YOLO(args.weights)
    print(f"[yolo] 模型已加载：{args.weights}")
    print(f"[yolo] 推理服务：http://{args.host}:{args.port}/predict")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[yolo] 已停止")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
