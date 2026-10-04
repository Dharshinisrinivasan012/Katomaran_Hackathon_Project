"""Pytest fixtures and path setup."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def valid_config_path(project_root: Path) -> Path:
    return project_root / "config.json"


@pytest.fixture
def config_with_empty_video_path(valid_config_path: Path, tmp_path: Path) -> Path:
    """A config.json copy whose input.video_path is intentionally empty.

    Used by tests that verify VideoPathNotConfiguredError is raised when no
    local sample video is configured. The real project config now has a real
    video path, so those tests must not use valid_config_path directly.
    """
    raw = json.loads(valid_config_path.read_text(encoding="utf-8"))
    raw["input"]["video_path"] = ""
    dest = tmp_path / "config_empty_video.json"
    dest.write_text(json.dumps(raw), encoding="utf-8")
    return dest


@pytest.fixture
def invalid_config_missing_detection() -> dict:
    path = FIXTURES_DIR / "invalid_missing_detection.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def invalid_config_bad_skip_frames() -> dict:
    path = FIXTURES_DIR / "invalid_bad_skip_frames.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_face_tracker.db"
