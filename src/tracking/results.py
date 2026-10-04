"""Tracked-face results. Track IDs are temporary motion identities, not face IDs."""

from __future__ import annotations

from dataclasses import dataclass

from src.detection.results import BoundingBox


@dataclass(frozen=True)
class TrackedFace:
    """One ByteTrack output for a frame."""

    track_id: int
    bbox: BoundingBox
    confidence: float
    frame_index: int | None = None
    state: str = "tracked"
