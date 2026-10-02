"""Unit tests for config loader and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config.exceptions import ConfigValidationError
from src.config.loader import load_config, validate_config_data


def test_load_default_project_config(valid_config_path: Path) -> None:
    config = load_config(valid_config_path)
    assert config.project_name == "Intelligent Face Tracker"
    assert config.detection.detection_skip_frames == 2
    assert config.database.sqlite_path == "data/sqlite/face_tracker.db"
    assert config.config_path == valid_config_path.resolve()


def test_validate_missing_detection_section(invalid_config_missing_detection: dict) -> None:
    with pytest.raises(ConfigValidationError) as exc:
        validate_config_data(invalid_config_missing_detection)
    assert "detection" in str(exc.value)


def test_validate_bad_detection_skip_frames(invalid_config_bad_skip_frames: dict) -> None:
    with pytest.raises(ConfigValidationError) as exc:
        validate_config_data(invalid_config_bad_skip_frames)
    assert "detection_skip_frames" in str(exc.value)


def test_load_config_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.json")


def test_load_config_from_temp_valid_json(tmp_path: Path, valid_config_path: Path) -> None:
    dest = tmp_path / "config.json"
    dest.write_text(valid_config_path.read_text(encoding="utf-8"), encoding="utf-8")
    config = load_config(dest)
    assert config.input.source_type == "video"


def test_resolve_path_relative(project_root: Path, valid_config_path: Path) -> None:
    config = load_config(valid_config_path)
    resolved = config.resolve_path(config.database.sqlite_path)
    assert resolved == project_root / "data/sqlite/face_tracker.db"
