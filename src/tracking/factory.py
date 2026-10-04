"""Create a FaceTracker from config."""

from __future__ import annotations

from src.config.loader import AppConfig, TrackingConfig
from src.tracking.bytetrack import ByteTrackFaceTracker
from src.tracking.exceptions import TrackerBackendError


def create_face_tracker(config: AppConfig | TrackingConfig) -> ByteTrackFaceTracker:
    tracking = config.tracking if isinstance(config, AppConfig) else config
    if tracking.tracker_type == "bytetrack":
        return ByteTrackFaceTracker(tracking)
    raise TrackerBackendError(
        f"Unsupported tracking.tracker_type {tracking.tracker_type!r}; expected 'bytetrack'"
    )
