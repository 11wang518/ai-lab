#!/usr/bin/env python
"""摄像头 / 视频实时推理，并在画面左上角显示 FPS。

示例：
    python yolo/realtime.py --weights yolo11n.pt --source 0
    python yolo/realtime.py --weights yolo11n.pt --source demo.mp4 --imgsz 480 --half

没有摄像头、或者要在服务器上跑基准测试时，用无窗口模式：
    python yolo/realtime.py --weights yolo11n.pt --source test.mp4 --no-show --max-frames 150
    python yolo/realtime.py --weights yolo11n.pt --source test.mp4 --no-show --save out.mp4

按 q 退出，按 s 保存当前帧。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.env_setup import prepare  # noqa: E402

prepare()  # 必须在 import ultralytics 之前


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
    parser.add_argument("--no-show", action="store_true", help="不开窗口，只推理（用于服务器/基准测试）")
    parser.add_argument("--save", help="把标注后的画面写成视频文件")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧，0 表示不限")
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
    if args.no_show:
        print("（无窗口模式，处理完自动结束）")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            args.save,
            fourcc,
            30.0,
            (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )
        if not writer.isOpened():
            print(f"无法创建视频文件：{args.save}", file=sys.stderr)
            writer = None
        else:
            print(f"标注后的画面将写入 {args.save}")

    frames = 0
    total_frames = 0
    inference_ms = 0.0
    run_started = time.perf_counter()
    window_started = run_started
    fps = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            infer_started = time.perf_counter()
            predict_kwargs = {
                "source": frame,
                "imgsz": args.imgsz,
                "conf": args.conf,
                "device": args.device,
                "verbose": False,
            }
            if args.half:
                predict_kwargs["half"] = True
            results = model.predict(
                **predict_kwargs
            )
            inference_ms += (time.perf_counter() - infer_started) * 1000
            annotated = results[0].plot()

            frames += 1
            total_frames += 1
            elapsed = time.perf_counter() - window_started
            if elapsed >= 0.5:
                fps = frames / elapsed
                frames = 0
                window_started = time.perf_counter()
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
            if writer is not None:
                writer.write(annotated)

            if not args.no_show:
                cv2.imshow("YOLO realtime (q=quit, s=save)", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    name = f"capture-{int(time.time())}.jpg"
                    cv2.imwrite(name, annotated)
                    print(f"已保存 {name}")

            if args.max_frames and total_frames >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    total_elapsed = time.perf_counter() - run_started
    print("\n处理完成")
    print(f"  总帧数     : {total_frames}")
    print(f"  总耗时     : {total_elapsed:.2f} s")
    if total_elapsed > 0 and total_frames:
        print(f"  平均帧率   : {total_frames / total_elapsed:.1f} FPS")
        print(f"  单帧推理   : {inference_ms / total_frames:.2f} ms")
        print(f"  实时性     : {'达到实时（>=25 FPS）' if total_frames / total_elapsed >= 25 else '未达 25 FPS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
