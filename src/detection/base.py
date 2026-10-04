"""Detector interface — swap YOLO backends without changing callers."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from src.detection.results import FaceDetection
from src.input.frame import VideoFrame

FrameLike = VideoFrame | np.ndarray


class FaceDetector(Protocol):
    def detect(self, frame: FrameLike, *, frame_index: int | None = None) -> Sequence[FaceDetection]:
        """Return face boxes for one image. No tracking or recognition."""
