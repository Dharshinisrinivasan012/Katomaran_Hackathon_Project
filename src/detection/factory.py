"""Create a FaceDetector from config. Callers do not import a specific YOLO class."""

from __future__ import annotations

from src.config.loader import AppConfig, DetectionConfig
from src.detection.exceptions import DetectorBackendError
from src.detection.yolo import YoloFaceDetector


def create_face_detector(config: AppConfig | DetectionConfig) -> YoloFaceDetector:
    """Return the configured detector. Only Ultralytics YOLO is implemented."""
    det_cfg = config.detection if isinstance(config, AppConfig) else config
    if det_cfg.backend == "ultralytics":
        return YoloFaceDetector(det_cfg)
    raise DetectorBackendError(
        f"Unsupported detection.backend {det_cfg.backend!r}; expected 'ultralytics'"
    )
