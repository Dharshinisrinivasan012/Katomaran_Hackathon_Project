"""Tests for ByteTrack wrapper structure and ID handling (not MOT accuracy)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.config.exceptions import ConfigValidationError
from src.config.loader import load_config, validate_config_data
from src.detection.results import BoundingBox, FaceDetection
from src.tracking.exceptions import TrackerBackendError
from src.tracking.factory import create_face_tracker
from src.tracking.results import TrackedFace

pytest.importorskip("ultralytics")


def _box(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> FaceDetection:
    return FaceDetection(bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2), confidence=conf)


def _tracker(valid_config_path: Path):
    config = load_config(valid_config_path)
    tracker = create_face_tracker(config.tracking)
    tracker.reset()
    return tracker


def test_tracked_face_structure() -> None:
    item = TrackedFace(
        track_id=3,
        bbox=BoundingBox(x1=1, y1=2, x2=10, y2=20),
        confidence=0.8,
        frame_index=4,
        state="tracked",
    )
    assert item.track_id == 3
    assert item.bbox.as_int_tuple() == (1, 2, 10, 20)
    assert item.state == "tracked"


def test_tracker_config_requires_bytetrack(valid_config_path: Path) -> None:
    config = load_config(valid_config_path)
    assert config.tracking.tracker_type == "bytetrack"
    raw = json.loads(valid_config_path.read_text(encoding="utf-8"))
    raw["tracking"]["tracker_type"] = "deepsort"
    with pytest.raises(ConfigValidationError) as exc:
        validate_config_data(raw)
    assert "tracker_type" in str(exc.value)


def test_factory_rejects_non_bytetrack(valid_config_path: Path) -> None:
    config = load_config(valid_config_path)
    with pytest.raises(TrackerBackendError):
        create_face_tracker(replace(config.tracking, tracker_type="sort"))


def test_empty_detections_return_no_tracks(valid_config_path: Path) -> None:
    tracker = _tracker(valid_config_path)
    tracks = tracker.update([], frame_index=0)
    assert tracks == []


def test_multiple_detections_get_track_ids(valid_config_path: Path) -> None:
    tracker = _tracker(valid_config_path)
    dets = [_box(10, 10, 40, 40), _box(80, 10, 120, 50)]
    tracks = tracker.update(dets, frame_index=0)
    assert len(tracks) >= 1
    for tr in tracks:
        assert isinstance(tr, TrackedFace)
        assert tr.track_id >= 1
        assert tr.bbox.x2 >= tr.bbox.x1
        assert 0.0 <= tr.confidence <= 1.0


def test_repeated_boxes_reuse_track_ids(valid_config_path: Path) -> None:
    tracker = _tracker(valid_config_path)
    dets = [_box(20, 20, 60, 70)]
    first = tracker.update(dets, frame_index=0)
    second = tracker.update(dets, frame_index=1)
    assert first and second
    assert {t.track_id for t in first} == {t.track_id for t in second}


def test_skip_frame_keeps_ids_without_new_tracks(valid_config_path: Path) -> None:
    tracker = _tracker(valid_config_path)
    dets = [_box(15, 15, 55, 65)]
    first = tracker.update(dets, frame_index=0)
    skipped = tracker.update(None, frame_index=1)
    assert first
    assert {t.track_id for t in skipped} == {t.track_id for t in first}
    assert all(t.state == "predicted" for t in skipped)


def test_reset_clears_tracks(valid_config_path: Path) -> None:
    tracker = _tracker(valid_config_path)
    tracker.update([_box(10, 10, 40, 40)], frame_index=0)
    tracker.reset()
    after = tracker.update([], frame_index=1)
    assert after == []
