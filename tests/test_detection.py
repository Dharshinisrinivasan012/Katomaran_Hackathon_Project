"""Tests for detector config, result structure, and skip policy.

These tests do not claim that YOLO found faces. Inference is only attempted when
local weights exist, and even then the assertion is on result types, not counts.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.config.exceptions import ConfigValidationError
from src.config.loader import load_config, validate_config_data
from src.detection.exceptions import DetectorBackendError, ModelNotFoundError
from src.detection.results import BoundingBox, FaceDetection
from src.detection.skip import DetectionSkipPolicy
from src.detection.factory import create_face_detector
from src.detection.yolo import YoloFaceDetector, resolve_model_path


def test_detection_skip_every_frame() -> None:
    policy = DetectionSkipPolicy(0)
    assert [policy.should_detect(i) for i in range(4)] == [True, True, True, True]


def test_detection_skip_frames_two() -> None:
    policy = DetectionSkipPolicy(2)
    assert [policy.should_detect(i) for i in range(6)] == [
        True,
        False,
        False,
        True,
        False,
        False,
    ]


def test_detection_result_structure() -> None:
    det = FaceDetection(
        bbox=BoundingBox(x1=1, y1=2, x2=10, y2=20),
        confidence=0.91,
        frame_index=3,
        class_id=0,
        class_name="face",
    )
    assert det.bbox.as_int_tuple() == (1, 2, 10, 20)
    assert det.confidence == 0.91
    assert det.frame_index == 3


def test_detector_config_from_project_file(valid_config_path: Path) -> None:
    config = load_config(valid_config_path)
    assert config.detection.backend == "ultralytics"
    assert config.detection.model_name
    assert config.detection.model_path
    assert 0 <= config.detection.confidence_threshold <= 1
    assert config.detection.detection_skip_frames >= 0


def test_detector_rejects_unknown_backend(valid_config_path: Path) -> None:
    raw = json.loads(valid_config_path.read_text(encoding="utf-8"))
    raw["detection"]["backend"] = "opencv-haar"
    with pytest.raises(ConfigValidationError) as exc:
        validate_config_data(raw)
    assert "backend" in str(exc.value)


def test_create_face_detector_requires_local_weights(
    valid_config_path: Path, tmp_path: Path
) -> None:
    config = load_config(valid_config_path)
    missing = tmp_path / "missing-yolo-face.pt"
    det_cfg = replace(config.detection, model_path=str(missing))
    with pytest.raises(ModelNotFoundError):
        create_face_detector(det_cfg)


def test_yolo_detector_requires_local_weights(valid_config_path: Path, tmp_path: Path) -> None:
    config = load_config(valid_config_path)
    missing = tmp_path / "missing-yolo-face.pt"
    det_cfg = replace(config.detection, model_path=str(missing))
    with pytest.raises(ModelNotFoundError) as exc:
        YoloFaceDetector(det_cfg)
    assert "not found" in str(exc.value).lower()


def test_yolo_detector_rejects_non_ultralytics_backend(
    valid_config_path: Path, tmp_path: Path
) -> None:
    config = load_config(valid_config_path)
    weights = tmp_path / "dummy.pt"
    weights.write_bytes(b"not-a-real-model")
    det_cfg = replace(config.detection, backend="other", model_path=str(weights))
    with pytest.raises(DetectorBackendError):
        YoloFaceDetector(det_cfg)


def _model_file_exists(valid_config_path: Path) -> bool:
    config = load_config(valid_config_path)
    return resolve_model_path(config.detection.model_path).is_file()


def test_yolo_inference_requires_model(valid_config_path: Path) -> None:
    """Do not fabricate detections. Skip unless configured weights are on disk."""
    if not _model_file_exists(valid_config_path):
        pytest.skip(
            "Requires a local YOLO face weights file at detection.model_path; "
            "weights are not committed and are not downloaded in tests."
        )
    pytest.importorskip("ultralytics")
    config = load_config(valid_config_path)
    detector = YoloFaceDetector(config.detection)
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    results = detector.detect(blank, frame_index=0)
    assert isinstance(results, list)
    for item in results:
        assert isinstance(item, FaceDetection)
        assert 0.0 <= item.confidence <= 1.0
        assert item.bbox.x2 >= item.bbox.x1
        assert item.bbox.y2 >= item.bbox.y1
