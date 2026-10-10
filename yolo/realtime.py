#!/usr/bin/env python
"""摄像头实时推理，并在画面左上角显示 FPS。

示例：
    python yolo/realtime.py --weights yolo11n.pt --source 0
    python yolo/realtime.py --weights yolo11n.pt --source demo.mp4 --imgsz 480 --half

按 q 退出，按 s 保存当前帧。
"""

from __future__ import annotations

import argparse
import sys
import time


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 实时推理")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--source", default="0", help="摄像头编号或视频文件")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true", help="FP16 推理，显卡上更快")
    parser.add_argument("--width", type=int, default=1280, help="采集宽度")
    parser.add_argument("--height", type=int, default=720, help="采集高度")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"缺少依赖：{exc}\n请先 pip install ultralytics opencv-python", file=sys.stderr)
        return 2

    source = int(args.source) if str(args.source).isdigit() else args.source
    model = YOLO(args.weights)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"无法打开视频源：{args.source}", file=sys.stderr)
        return 2
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("实时推理已启动，按 q 退出，按 s 保存当前帧")
    frames = 0
    started = time.perf_counter()
    fps = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                half=args.half,
                verbose=False,
            )
            annotated = results[0].plot()

            frames += 1
            elapsed = time.perf_counter() - started
            if elapsed >= 0.5:
                fps = frames / elapsed
                frames = 0
                started = time.perf_counter()
            cv2.putText(
                annotated,
                f"FPS {fps:5.1f}",
                (14, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("YOLO realtime (q=quit, s=save)", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                name = f"capture-{int(time.time())}.jpg"
                cv2.imwrite(name, annotated)
                print(f"已保存 {name}")
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
