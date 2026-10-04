"""YOLO face detector (Ultralytics backend). Does not download weights."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.config.loader import DetectionConfig, PROJECT_ROOT
from src.detection.exceptions import DetectorBackendError, ModelLoadError, ModelNotFoundError
from src.detection.results import BoundingBox, FaceDetection
from src.input.frame import VideoFrame

logger = logging.getLogger(__name__)


def resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


class YoloFaceDetector:
    """Ultralytics YOLO wrapper. Requires a local weights file at ``model_path``."""

    def __init__(self, config: DetectionConfig) -> None:
        if config.backend != "ultralytics":
            raise DetectorBackendError(
                f"Unsupported detection.backend {config.backend!r}; expected 'ultralytics'"
            )
        self._config = config
        self._model_path = resolve_model_path(config.model_path)
        if not self._model_path.is_file():
            raise ModelNotFoundError(
                "YOLO face weights were not found at "
                f"{self._model_path}. Place a YOLO face detection model at "
                "detection.model_path (see README). Weights are not committed to git "
                "and are not downloaded automatically."
            )
        self._model = self._load_model()

    def _load_model(self) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorBackendError(
                "The 'ultralytics' package is required for YOLO detection. "
                "Install dependencies from requirements.txt."
            ) from exc
        try:
            model = YOLO(str(self._model_path))
        except Exception as exc:  # noqa: BLE001 — surface load failures clearly
            raise ModelLoadError(
                f"Failed to load YOLO weights from {self._model_path}: {exc}"
            ) from exc
        logger.info(
            "Loaded YOLO model %s from %s",
            self._config.model_name or self._model_path.name,
            self._model_path,
        )
        return model

    def detect(
        self,
        frame: VideoFrame | np.ndarray,
        *,
        frame_index: int | None = None,
    ) -> Sequence[FaceDetection]:
        image, inferred_index = _image_and_index(frame, frame_index)
        device = None if self._config.device == "auto" else self._config.device
        try:
            results = self._model.predict(
                source=image,
                conf=self._config.confidence_threshold,
                iou=self._config.iou_threshold,
                imgsz=self._config.input_size,
                device=device,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"YOLO inference failed: {exc}") from exc

        detections: list[FaceDetection] = []
        if not results:
            return detections
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections
        xyxy = boxes.xyxy
        confs = boxes.conf
        cls_ids = getattr(boxes, "cls", None)
        names = getattr(result, "names", None) or {}
        count = len(xyxy)
        for i in range(count):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i].tolist())
            confidence = float(confs[i].item()) if confs is not None else 0.0
            class_id = int(cls_ids[i].item()) if cls_ids is not None else None
            class_name = None
            if class_id is not None and isinstance(names, dict):
                class_name = names.get(class_id)
            detections.append(
                FaceDetection(
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=confidence,
                    frame_index=inferred_index,
                    class_id=class_id,
                    class_name=class_name,
                )
            )
        return detections


def _image_and_index(
    frame: VideoFrame | np.ndarray,
    frame_index: int | None,
) -> tuple[np.ndarray, int | None]:
    if isinstance(frame, VideoFrame):
        return frame.image, frame.index if frame_index is None else frame_index
    return frame, frame_index
