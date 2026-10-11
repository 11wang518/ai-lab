"""端到端冒烟测试：不依赖任何第三方库，直接拉起服务并访问所有接口。

用法：python tools/smoke_test.py
退出码 0 表示全部通过。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import AppConfig  # noqa: E402
from app.server.http_app import create_server, find_free_port, serve_in_background  # noqa: E402

# 用真实照片做示例，和界面上"示例图"按钮保持一致
SAMPLE = ROOT / "web" / "samples" / "street.jpg"

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    if condition:
        PASSED.append(name)
        print(f"  [通过] {name}{(' — ' + detail) if detail else ''}")
    else:
        FAILED.append(name)
        print(f"  [失败] {name}{(' — ' + detail) if detail else ''}")
    return condition


def get(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, response.read()


def post_json(url: str, payload: dict, timeout: float = 60.0):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def post_image(url: str, data: bytes, filename: str, timeout: float = 120.0):
    boundary = "----smoke" + uuid.uuid4().hex
    crlf = b"\r\n"
    body = b"".join(
        [
            b"--" + boundary.encode() + crlf,
            f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode(),
            crlf,
            b"Content-Type: image/png" + crlf + crlf,
            data,
            crlf,
            b"--" + boundary.encode() + b"--" + crlf,
        ]
    )
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def parse_sse(raw: bytes) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in raw.decode("utf-8", "replace").split("\n\n"):
        if not block.strip():
            continue
        name = "message"
        data_lines = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if data_lines:
            try:
                events.append((name, json.loads("\n".join(data_lines))))
            except json.JSONDecodeError:
                events.append((name, {"raw": "\n".join(data_lines)}))
    return events


def main() -> int:
    print("=" * 60)
    print("AI Lab 端到端冒烟测试")
    print("=" * 60)

    config = AppConfig.from_env()
    # 第一段测试固定用 none 后端：这条分支在任何机器上都必须返回同样的结果，
    # 不受"有没有装 ultralytics、有没有下载权重"影响。
    config.vision.backend = "none"
    port = find_free_port(config.server.host, 8321)
    config.server.port = port
    server = create_server(config)
    serve_in_background(server)
    base = f"http://{config.server.host}:{port}"
    time.sleep(0.3)
    print(f"服务地址: {base}\n")

    try:
        print("1. 基础接口")
        status, body = get(f"{base}/api/health")
        health = json.loads(body)
        check("GET /api/health 返回 200", status == 200, f"HTTP {status}")
        check("健康检查 ok=True", health.get("ok") is True)
        check(
            "健康检查包含检测后端信息",
            "active_backend" in health.get("vision", {}),
            str(health.get("vision", {}).get("active_backend")),
        )

        status, body = get(f"{base}/api/config")
        payload = json.loads(body)
        check("GET /api/config 返回 200", status == 200)
        check("配置里带模型后端列表", len(payload.get("providers", [])) == 3)
        check(
            "配置里的 API Key 已脱敏",
            payload["llm"]["openai_api_key"] in ("", "***"),
        )

        status, body = get(f"{base}/")
        check("首页返回 HTML", status == 200 and b"AI Lab" in body)
        status, body = get(f"{base}/app.js")
        check("静态资源 app.js 可访问", status == 200 and b"fallbackDetect" in body)

        status, body = get(f"{base}/api/samples")
        samples = json.loads(body).get("samples", [])
        check("示例图列表非空", len(samples) >= 1, f"{len(samples)} 张")

        print("\n2. 对话接口（SSE 流式）")
        status, body = post_json(
            f"{base}/api/chat",
            {
                "messages": [{"role": "user", "content": "你好"}],
                "provider": "demo",
            },
        )
        events = parse_sse(body)
        kinds = [name for name, _ in events]
        check("POST /api/chat 返回 200", status == 200, f"HTTP {status}")
        check("返回 meta 事件", "meta" in kinds)
        check("返回 delta 流式分片", kinds.count("delta") > 1, f"{kinds.count('delta')} 片")
        check("返回 done 事件", "done" in kinds)
        text = "".join(data.get("text", "") for name, data in events if name == "delta")
        check("拼接后的回答非空", len(text) > 20, f"{len(text)} 字")

        print("\n3. 工具调用")
        status, body = post_json(
            f"{base}/api/chat",
            {
                "messages": [{"role": "user", "content": "帮我算一下 12*(3+4)"}],
                "provider": "demo",
                "use_tools": True,
            },
        )
        events = parse_sse(body)
        tool_events = [data for name, data in events if name == "tool"]
        check("触发了工具调用事件", len(tool_events) == 1)
        if tool_events:
            result = tool_events[0]["result"]
            check("计算器结果正确 (12*(3+4)=84)", result.get("value") == 84, str(result))

        print("\n4. 目标检测接口")
        if not SAMPLE.exists():
            check("示例图片存在", False, str(SAMPLE))
        else:
            status, body = post_image(
                f"{base}/api/detect", SAMPLE.read_bytes(), SAMPLE.name
            )
            payload = json.loads(body)
            check("POST /api/detect 返回 200", status == 200, f"HTTP {status}")
            check("响应 ok=True", payload.get("ok") is True)
            check(
                "引擎字段存在（none 表示需装 ultralytics）",
                "engine" in payload,
                f"engine={payload.get('engine')}",
            )
            if payload.get("engine") == "none":
                check("未配置模型时给出解释", bool(payload.get("note")))
                print(f"       提示：{payload.get('note')}")

        print("\n5. 错误处理")
        status, body = post_json(f"{base}/api/chat", {"messages": []})
        check("空 messages 返回 400", status == 400, f"HTTP {status}")
        status, body = post_json(
            f"{base}/api/chat",
            {"messages": [{"role": "user", "content": "hi"}], "provider": "不存在"},
        )
        check("未知后端返回 400", status == 400, f"HTTP {status}")
        status, body = post_image(f"{base}/api/detect", b"", "empty.png")
        check("空图片返回 400", status == 400, f"HTTP {status}")

        print("\n6. 真实 YOLO 推理（有训练权重时才跑）")
        # 优先用真实数据微调出来的权重（应用默认加载的就是它）
        weights = ROOT / "runs" / "train" / "coco8-ft" / "weights" / "best.pt"
        if not weights.exists():
            weights = ROOT / "runs" / "train" / "synthetic-cpu" / "weights" / "best.pt"
        sample_image = ROOT / "web" / "samples" / "street.jpg"
        if not weights.exists():
            print(f"  [跳过] 没有找到训练权重：{weights}")
        elif not sample_image.exists():
            print(f"  [跳过] 没有找到验证图片：{sample_image}")
        else:
            real_config = AppConfig.from_env()
            real_config.vision.backend = "ultralytics"
            real_config.vision.weights = str(weights)
            real_config.vision.conf = 0.25
            real_config.server.port = find_free_port(config.server.host, 8361)
            real_server = create_server(real_config)
            serve_in_background(real_server)
            real_base = f"http://{real_config.server.host}:{real_config.server.port}"
            time.sleep(0.3)
            try:
                status, body = post_image(
                    f"{real_base}/api/detect", sample_image.read_bytes(), sample_image.name
                )
                payload = json.loads(body)
                check(
                    "真实推理返回 200",
                    status == 200,
                    f"HTTP {status}",
                )
                check(
                    "使用 ultralytics 引擎",
                    payload.get("engine") == "ultralytics",
                    str(payload.get("engine")),
                )
                boxes = payload.get("boxes", [])
                check("至少检出一个目标", len(boxes) >= 1, f"{len(boxes)} 个")
                if boxes:
                    labels = {box["label"] for box in boxes}
                    check(
                        "识别出 car / person",
                        labels & {"car", "person"} != set(),
                        "、".join(sorted(labels)),
                    )
            finally:
                real_server.shutdown()
                real_server.server_close()

    finally:
        server.shutdown()
        server.server_close()

    print("\n" + "=" * 60)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        print("失败项：" + "、".join(FAILED))
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
