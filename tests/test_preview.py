"""Preview CLI: configuration errors without claiming detections."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.preview_detections import run_preview


def test_preview_reports_empty_video_path(
    config_with_empty_video_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Uses a config whose video_path is empty — real project config now has a
    # working sample so we must not use valid_config_path for this assertion.
    pytest.importorskip("cv2")
    with caplog.at_level("ERROR"):
        code = run_preview(["--config", str(config_with_empty_video_path), "--no-display"])
    assert code == 2
    assert "input.video_path" in caplog.text
    assert "drive.google.com" in caplog.text
