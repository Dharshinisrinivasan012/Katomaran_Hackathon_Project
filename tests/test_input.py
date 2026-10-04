"""Tests for configurable file / RTSP video input (no official sample filename)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.config.loader import load_config
from src.input.exceptions import (
    RtspNotConfiguredError,
    VideoFileNotFoundError,
    VideoOpenError,
    VideoPathNotConfiguredError,
    VideoReadError,
    VideoSourceError,
)
from src.input.factory import create_frame_source
from src.input.opencv_source import is_rtsp_placeholder


def _write_synthetic_video(path: Path, frame_count: int = 4, size: int = 32) -> None:
    cv2 = pytest.importorskip("cv2")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 5.0, (size, size))
    assert writer.isOpened(), "OpenCV could not create a synthetic test video"
    for i in range(frame_count):
        frame = np.full((size, size, 3), i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_is_rtsp_placeholder() -> None:
    assert is_rtsp_placeholder("")
    assert is_rtsp_placeholder("rtsp://<username>:<password>@<host>:<port>/<stream>")
    assert not is_rtsp_placeholder("rtsp://127.0.0.1:554/live")


def test_create_file_source_requires_configured_path(
    config_with_empty_video_path: Path,
) -> None:
    # Uses a config whose video_path is empty — real project config now has a
    # working sample so we must not use valid_config_path for this assertion.
    config = load_config(config_with_empty_video_path)
    assert config.input.source_type == "file"
    with pytest.raises(VideoPathNotConfiguredError) as exc:
        create_frame_source(config)
    assert "input.video_path" in str(exc.value)
    assert "drive.google.com" in str(exc.value)


def test_create_file_source_invalid_path(valid_config_path: Path, tmp_path: Path) -> None:
    config = load_config(valid_config_path)
    missing = tmp_path / "does-not-exist.mp4"
    config = replace(config, input=replace(config.input, video_path=str(missing)))
    with pytest.raises(VideoFileNotFoundError) as exc:
        create_frame_source(config)
    assert str(missing) in str(exc.value)


def test_create_file_source_rejects_directory(valid_config_path: Path, tmp_path: Path) -> None:
    config = load_config(valid_config_path)
    config = replace(config, input=replace(config.input, video_path=str(tmp_path)))
    with pytest.raises(VideoOpenError) as exc:
        create_frame_source(config)
    assert "not a file" in str(exc.value)


def test_file_source_unopenable_non_video(valid_config_path: Path, tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    junk = tmp_path / "not-a-video.mp4"
    junk.write_text("this is not video data", encoding="utf-8")
    config = load_config(valid_config_path)
    file_cfg = replace(config.input, source_type="file", video_path=str(junk))
    source = create_frame_source(file_cfg)
    with pytest.raises((VideoOpenError, VideoReadError, VideoSourceError)):
        source.open()
        source.read()
        source.close()


def test_create_frame_source_rejects_unknown_type(valid_config_path: Path) -> None:
    config = load_config(valid_config_path)
    with pytest.raises(VideoSourceError):
        create_frame_source(replace(config.input, source_type="webcam"))


def test_rtsp_placeholder_is_rejected(valid_config_path: Path) -> None:
    config = load_config(valid_config_path)
    rtsp_cfg = replace(config.input, source_type="rtsp")
    with pytest.raises(RtspNotConfiguredError):
        create_frame_source(rtsp_cfg)


def test_rtsp_source_is_created_when_url_looks_real(valid_config_path: Path) -> None:
    config = load_config(valid_config_path)
    rtsp_cfg = replace(
        config.input,
        source_type="rtsp",
        rtsp_url="rtsp://127.0.0.1:554/stream",
    )
    source = create_frame_source(rtsp_cfg)
    assert source.source_type == "rtsp"


def test_file_source_reads_synthetic_video_until_eof(
    valid_config_path: Path, tmp_path: Path
) -> None:
    pytest.importorskip("cv2")
    video_path = tmp_path / "synthetic_input.avi"
    _write_synthetic_video(video_path, frame_count=3)
    config = load_config(valid_config_path)
    file_cfg = replace(config.input, source_type="file", video_path=str(video_path))
    source = create_frame_source(file_cfg)
    frames = list(source)
    assert len(frames) == 3
    assert frames[0].source_type == "file"
    assert frames[0].index == 0
    assert frames[-1].index == 2
    assert frames[0].image.ndim == 3
