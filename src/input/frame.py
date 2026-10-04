"""Shared frame payload for file and RTSP sources."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VideoFrame:
    """One decoded BGR frame plus source metadata."""

    image: np.ndarray
    index: int
    source_type: str
    source_label: str
    width: int
    height: int
    timestamp_ms: float
