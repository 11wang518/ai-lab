"""YOLO 推理接入层。"""

from .detector import Detector, DetectorError, DetectionResult

__all__ = ["Detector", "DetectorError", "DetectionResult"]
