"""Face detection (YOLO). Tracking and recognition are out of scope for this module."""

from src.detection.exceptions import (
    DetectionError,
    DetectorBackendError,
    ModelLoadError,
    ModelNotFoundError,
)
from src.detection.results import BoundingBox, FaceDetection
from src.detection.skip import DetectionSkipPolicy
from src.detection.factory import create_face_detector
from src.detection.yolo import YoloFaceDetector, resolve_model_path

__all__ = [
    "BoundingBox",
    "DetectionError",
    "DetectionSkipPolicy",
    "DetectorBackendError",
    "FaceDetection",
    "ModelLoadError",
    "ModelNotFoundError",
    "YoloFaceDetector",
    "create_face_detector",
    "resolve_model_path",
]
