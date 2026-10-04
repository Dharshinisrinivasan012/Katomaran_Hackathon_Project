"""Tracker interface — swap ByteTrack backends without changing callers."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from src.detection.results import FaceDetection
from src.tracking.results import TrackedFace

DetectionsOrSkip = Sequence[FaceDetection] | None


class FaceTracker(Protocol):
    def update(
        self,
        detections: DetectionsOrSkip,
        *,
        frame_index: int,
        image: np.ndarray | None = None,
    ) -> Sequence[TrackedFace]:
        """Associate detections for this frame.

        Pass a sequence (possibly empty) when YOLO ran. Pass ``None`` on skipped
        detection frames so the tracker can predict existing IDs without starting
        new tracks from a missing measurement.
        """

    def reset(self) -> None:
        """Clear all tracks and reset ID allocation."""
