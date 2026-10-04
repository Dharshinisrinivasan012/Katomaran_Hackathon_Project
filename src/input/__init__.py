"""Configurable video input: local file or RTSP, OpenCV frames."""

from src.input.exceptions import (
    RtspConnectionError,
    RtspNotConfiguredError,
    VideoFileNotFoundError,
    VideoOpenError,
    VideoPathNotConfiguredError,
    VideoReadError,
    VideoSourceError,
)
from src.input.factory import create_frame_source
from src.input.frame import VideoFrame
from src.input.opencv_source import OpenCvFrameSource, is_rtsp_placeholder

__all__ = [
    "OpenCvFrameSource",
    "RtspConnectionError",
    "RtspNotConfiguredError",
    "VideoFileNotFoundError",
    "VideoFrame",
    "VideoOpenError",
    "VideoPathNotConfiguredError",
    "VideoReadError",
    "VideoSourceError",
    "create_frame_source",
    "is_rtsp_placeholder",
]
