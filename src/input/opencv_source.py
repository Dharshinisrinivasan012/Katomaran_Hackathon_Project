"""OpenCV-backed local file and RTSP frame sources."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config.loader import InputConfig, PROJECT_ROOT
from src.input.exceptions import (
    RtspConnectionError,
    RtspNotConfiguredError,
    VideoFileNotFoundError,
    VideoOpenError,
    VideoPathNotConfiguredError,
    VideoReadError,
    VideoSourceError,
)
from src.input.frame import VideoFrame

logger = logging.getLogger(__name__)

_SAMPLE_DRIVE = (
    "https://drive.google.com/drive/folders/15YCN3CYb97GyIoNUV6NJxGNIf-rUBFUJ?usp=sharing"
)


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise VideoSourceError(
            "opencv-python is required for video input. Install from requirements.txt."
        ) from exc
    return cv2


def is_rtsp_placeholder(url: str) -> bool:
    stripped = url.strip()
    if not stripped:
        return True
    return "<" in stripped and ">" in stripped


def _resolve_video_path(video_path: str) -> Path:
    path = Path(video_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


class OpenCvFrameSource:
    """Shared OpenCV VideoCapture wrapper for file and RTSP."""

    def __init__(
        self,
        *,
        source_type: str,
        uri: str,
        source_label: str,
        reconnect_delay_sec: float = 0,
        max_reconnect_attempts: int = 0,
        frame_width: int = 0,
        frame_height: int = 0,
        fps_limit: int = 0,
    ) -> None:
        self.source_type = source_type
        self._uri = uri
        self._source_label = source_label
        self._reconnect_delay_sec = reconnect_delay_sec
        self._max_reconnect_attempts = max_reconnect_attempts
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._fps_limit = fps_limit
        self._cap: Any = None
        self._index = 0
        self._last_read_monotonic = 0.0

    def open(self) -> None:
        self.close()
        self._index = 0
        self._cap = self._open_capture()
        self._apply_optional_size()
        logger.info("Opened %s source: %s", self.source_type, self._source_label)

    def _open_capture(self) -> Any:
        cv2 = _cv2()
        cap = cv2.VideoCapture(self._uri)
        if not cap.isOpened():
            cap.release()
            if self.source_type == "rtsp":
                raise RtspConnectionError(
                    f"Unable to open RTSP stream: {self._source_label}"
                )
            raise VideoOpenError(
                f"Unable to open video file: {self._source_label}"
            )
        return cap

    def _apply_optional_size(self) -> None:
        if self._cap is None:
            return
        cv2 = _cv2()
        if self._frame_width > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
        if self._frame_height > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)

    def read(self) -> VideoFrame | None:
        if self._cap is None:
            raise VideoSourceError("Frame source is not open")

        ok, image = self._cap.read()
        if not ok or image is None:
            if self.source_type == "file":
                if self._index == 0:
                    raise VideoReadError(
                        f"Failed to read the first frame from {self._source_label}"
                    )
                logger.info("Reached end of local video: %s", self._source_label)
                return None
            self._reconnect_or_raise()
            ok, image = self._cap.read() if self._cap is not None else (False, None)
            if not ok or image is None:
                raise VideoReadError(
                    f"Unexpected read failure from RTSP source {self._source_label}"
                )

        self._respect_fps_limit()
        frame = self._to_frame(image)
        self._index += 1
        return frame

    def _reconnect_or_raise(self) -> None:
        attempts = self._max_reconnect_attempts
        if attempts <= 0:
            raise RtspConnectionError(
                "RTSP read failed and reconnect is disabled "
                f"(input.max_reconnect_attempts=0) for {self._source_label}"
            )
        last_error: Exception | None = None
        for tried in range(1, attempts + 1):
            logger.warning(
                "RTSP read failed; reconnect attempt %s/%s",
                tried,
                attempts,
            )
            if self._reconnect_delay_sec > 0:
                time.sleep(self._reconnect_delay_sec)
            try:
                if self._cap is not None:
                    self._cap.release()
                self._cap = self._open_capture()
                self._apply_optional_size()
                return
            except RtspConnectionError as exc:
                last_error = exc
        raise RtspConnectionError(
            f"RTSP reconnect failed after {attempts} attempt(s) for {self._source_label}"
        ) from last_error

    def _respect_fps_limit(self) -> None:
        if self._fps_limit <= 0:
            return
        min_interval = 1.0 / float(self._fps_limit)
        now = time.monotonic()
        if self._last_read_monotonic > 0:
            elapsed = now - self._last_read_monotonic
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self._last_read_monotonic = time.monotonic()

    def _to_frame(self, image: np.ndarray) -> VideoFrame:
        height, width = image.shape[:2]
        timestamp_ms = 0.0
        if self._cap is not None:
            cv2 = _cv2()
            timestamp_ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        return VideoFrame(
            image=image,
            index=self._index,
            source_type=self.source_type,
            source_label=self._source_label,
            width=int(width),
            height=int(height),
            timestamp_ms=timestamp_ms,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.debug("Closed %s source: %s", self.source_type, self._source_label)

    def __iter__(self):
        self.open()
        try:
            while True:
                frame = self.read()
                if frame is None:
                    break
                yield frame
        finally:
            self.close()

    def __enter__(self) -> OpenCvFrameSource:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def create_file_source(config: InputConfig) -> OpenCvFrameSource:
    if not config.video_path.strip():
        raise VideoPathNotConfiguredError(
            "input.video_path is empty. Download the official sample video from "
            f"{_SAMPLE_DRIVE} and set input.video_path in config.json to the local file."
        )
    path = _resolve_video_path(config.video_path)
    if not path.exists():
        raise VideoFileNotFoundError(
            f"Configured video file does not exist: {path}. "
            "Set input.video_path after downloading the official sample from "
            f"{_SAMPLE_DRIVE}"
        )
    if not path.is_file():
        raise VideoOpenError(f"input.video_path is not a file: {path}")
    return OpenCvFrameSource(
        source_type="file",
        uri=str(path),
        source_label=str(path),
        reconnect_delay_sec=0,
        max_reconnect_attempts=0,
        frame_width=config.frame_width,
        frame_height=config.frame_height,
        fps_limit=config.fps_limit,
    )


def create_rtsp_source(config: InputConfig) -> OpenCvFrameSource:
    url = config.rtsp_url.strip()
    if is_rtsp_placeholder(url):
        raise RtspNotConfiguredError(
            "input.rtsp_url is empty or still a placeholder. "
            "Set a real RTSP URL for interview / live-camera testing."
        )
    if not url.lower().startswith("rtsp://"):
        raise RtspNotConfiguredError(
            f"input.rtsp_url must start with rtsp:// (got {url!r})"
        )
    return OpenCvFrameSource(
        source_type="rtsp",
        uri=url,
        source_label=_redact_rtsp(url),
        reconnect_delay_sec=config.reconnect_delay_sec,
        max_reconnect_attempts=config.max_reconnect_attempts,
        frame_width=config.frame_width,
        frame_height=config.frame_height,
        fps_limit=config.fps_limit,
    )


def _redact_rtsp(url: str) -> str:
    if "@" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host = rest.split("@", 1)
    if ":" in creds:
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host}"
    return f"{scheme}://***@{host}"
