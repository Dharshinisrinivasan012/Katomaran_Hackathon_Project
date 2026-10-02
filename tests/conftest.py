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
