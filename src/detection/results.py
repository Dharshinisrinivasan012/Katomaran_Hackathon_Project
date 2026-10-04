"""Structured face-detection results (no tracking or identity)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in pixel coordinates (inclusive-exclusive style xyxy)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


@dataclass(frozen=True)
class FaceDetection:
    """One face candidate from the detector."""

    bbox: BoundingBox
    confidence: float
    frame_index: int | None = None
    class_id: int | None = None
    class_name: str | None = None
