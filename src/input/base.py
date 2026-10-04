"""Frame source protocol used by the rest of the application."""

from __future__ import annotations

from typing import Iterator, Protocol

from src.input.frame import VideoFrame


class FrameSource(Protocol):
    """Produces frames; callers must not care whether the origin is a file or RTSP."""

    source_type: str

    def open(self) -> None:
        """Open the underlying capture device."""

    def read(self) -> VideoFrame | None:
        """Return the next frame, or None at end-of-file for local videos."""

    def close(self) -> None:
        """Release capture resources."""

    def __iter__(self) -> Iterator[VideoFrame]:
        """Open, yield frames, then close."""
