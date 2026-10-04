"""Create a FrameSource from config.json input settings."""

from __future__ import annotations

from src.config.loader import AppConfig, InputConfig
from src.input.exceptions import VideoSourceError
from src.input.opencv_source import OpenCvFrameSource, create_file_source, create_rtsp_source


def create_frame_source(
    config: AppConfig | InputConfig,
) -> OpenCvFrameSource:
    """Return a file or RTSP source. Callers consume frames the same way either way."""
    input_cfg = config.input if isinstance(config, AppConfig) else config
    if input_cfg.source_type == "file":
        return create_file_source(input_cfg)
    if input_cfg.source_type == "rtsp":
        return create_rtsp_source(input_cfg)
    raise VideoSourceError(
        f"Unsupported input.source_type {input_cfg.source_type!r}; expected 'file' or 'rtsp'"
    )
