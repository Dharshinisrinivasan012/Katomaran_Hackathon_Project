"""Reusable detection-skip policy for later tracking integration."""

from __future__ import annotations


class DetectionSkipPolicy:
    """Run the detector every ``skip_frames + 1`` frames.

    ``detection_skip_frames = 0`` means detect every frame.
    ``detection_skip_frames = 2`` means detect on frames 0, 3, 6, ...
    Intermediate frames are for the tracker: pass ``detections=None`` so ByteTrack
    Kalman-predicts existing IDs instead of treating the skip as a new measurement.
    """

    def __init__(self, skip_frames: int) -> None:
        if skip_frames < 0:
            raise ValueError("skip_frames must be >= 0")
        self.skip_frames = skip_frames
        self.interval = skip_frames + 1

    def should_detect(self, frame_index: int) -> bool:
        if frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        return frame_index % self.interval == 0
