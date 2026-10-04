"""Load and validate config.json."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.config.exceptions import ConfigValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"

_ENV_CONFIG_PATH = "CONFIG_PATH"


def _require_mapping(data: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(data, dict):
        raise ConfigValidationError(f"Expected object", path=key)
    return data


def _require_str(data: Mapping[str, Any], key: str, *, parent: str) -> str:
    path = f"{parent}.{key}" if parent else key
    if key not in data:
        raise ConfigValidationError("Missing required key", path=path)
    value = data[key]
    if not isinstance(value, str):
        raise ConfigValidationError("Expected string", path=path)
    return value


def _require_bool(data: Mapping[str, Any], key: str, *, parent: str) -> bool:
    path = f"{parent}.{key}" if parent else key
    if key not in data:
        raise ConfigValidationError("Missing required key", path=path)
    value = data[key]
    if not isinstance(value, bool):
        raise ConfigValidationError("Expected boolean", path=path)
    return value


def _require_number(
    data: Mapping[str, Any],
    key: str,
    *,
    parent: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    path = f"{parent}.{key}" if parent else key
    if key not in data:
        raise ConfigValidationError("Missing required key", path=path)
    value = data[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigValidationError("Expected number", path=path)
    if min_value is not None and value < min_value:
        raise ConfigValidationError(f"Must be >= {min_value}", path=path)
    if max_value is not None and value > max_value:
        raise ConfigValidationError(f"Must be <= {max_value}", path=path)
    return float(value)


def _require_int(
    data: Mapping[str, Any],
    key: str,
    *,
    parent: str,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    path = f"{parent}.{key}" if parent else key
    if key not in data:
        raise ConfigValidationError("Missing required key", path=path)
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError("Expected integer", path=path)
    if min_value is not None and value < min_value:
        raise ConfigValidationError(f"Must be >= {min_value}", path=path)
    if max_value is not None and value > max_value:
        raise ConfigValidationError(f"Must be <= {max_value}", path=path)
    return value


def _require_str_list(data: Mapping[str, Any], key: str, *, parent: str) -> list[str]:
    path = f"{parent}.{key}" if parent else key
    if key not in data:
        raise ConfigValidationError("Missing required key", path=path)
    value = data[key]
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ConfigValidationError("Expected list of strings", path=path)
    return list(value)


def _require_line(
    data: Mapping[str, Any], key: str, *, parent: str
) -> list[tuple[float, float]]:
    path = f"{parent}.{key}" if parent else key
    if key not in data:
        raise ConfigValidationError("Missing required key", path=path)
    raw = data[key]
    if not isinstance(raw, list) or len(raw) != 2:
        raise ConfigValidationError("Expected two-point line `[[x,y],[x,y]]`", path=path)
    points: list[tuple[float, float]] = []
    for i, pt in enumerate(raw):
        if not isinstance(pt, list) or len(pt) != 2:
            raise ConfigValidationError("Each line point must be [x, y]", path=f"{path}[{i}]")
        x, y = pt[0], pt[1]
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ConfigValidationError("Line coordinates must be numbers", path=f"{path}[{i}]")
        if not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
            raise ConfigValidationError(
                "Normalized line coordinates must be in [0, 1]", path=f"{path}[{i}]"
            )
        points.append((float(x), float(y)))
    return points


@dataclass(frozen=True)
class InputConfig:
    source_type: str
    video_path: str
    rtsp_url: str
    reconnect_delay_sec: float
    max_reconnect_attempts: int
    frame_width: int
    frame_height: int
    fps_limit: int


@dataclass(frozen=True)
class DetectionConfig:
    backend: str
    model_name: str
    model_path: str
    confidence_threshold: float
    iou_threshold: float
    device: str
    detection_skip_frames: int
    input_size: int


@dataclass(frozen=True)
class TrackingConfig:
    tracker_type: str
    track_thresh: float
    track_buffer: int
    match_thresh: float
    frame_rate: int


@dataclass(frozen=True)
class RecognitionConfig:
    model_name: str
    embedding_dim: int
    similarity_threshold: float
    min_face_size_px: int
    quality_min_score: float
    providers: list[str]


@dataclass(frozen=True)
class RegistrationConfig:
    auto_register_unknown: bool
    min_detections_before_register: int
    register_cooldown_sec: float
    crop_save_enabled: bool
    crop_dir: str


@dataclass(frozen=True)
class EntryExitConfig:
    mode: str
    entry_line: list[tuple[float, float]]
    exit_line: list[tuple[float, float]]
    direction_entry: str
    direction_exit: str
    debounce_sec: float
    min_track_age_frames: int


@dataclass(frozen=True)
class DatabaseConfig:
    sqlite_path: str
    wal_mode: bool


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    console_enabled: bool
    file_path: str
    rotate_max_bytes: int
    rotate_backup_count: int
    log_detections: bool
    log_embeddings: bool


@dataclass(frozen=True)
class PathsConfig:
    registered_faces_dir: str
    events_snapshot_dir: str
    samples_dir: str


@dataclass(frozen=True)
class PerformanceConfig:
    use_half_precision: bool
    num_threads: int
    max_pipeline_queue_size: int


@dataclass(frozen=True)
class AppConfig:
    project_name: str
    version: str
    input: InputConfig
    detection: DetectionConfig
    tracking: TrackingConfig
    recognition: RecognitionConfig
    registration: RegistrationConfig
    entry_exit: EntryExitConfig
    database: DatabaseConfig
    logging: LoggingConfig
    paths: PathsConfig
    performance: PerformanceConfig
    config_path: Path

    def resolve_path(self, relative: str) -> Path:
        """Resolve a config path relative to the project root."""
        p = Path(relative)
        if p.is_absolute():
            return p
        return PROJECT_ROOT / p


def _parse_input(section: Mapping[str, Any]) -> InputConfig:
    parent = "input"
    source_type = _require_str(section, "source_type", parent=parent)
    if source_type not in ("file", "rtsp"):
        raise ConfigValidationError(
            "source_type must be 'file' or 'rtsp'", path="input.source_type"
        )
    return InputConfig(
        source_type=source_type,
        video_path=_require_str(section, "video_path", parent=parent),
        rtsp_url=_require_str(section, "rtsp_url", parent=parent),
        reconnect_delay_sec=_require_number(
            section, "reconnect_delay_sec", parent=parent, min_value=0
        ),
        max_reconnect_attempts=_require_int(
            section, "max_reconnect_attempts", parent=parent, min_value=0
        ),
        frame_width=_require_int(section, "frame_width", parent=parent, min_value=0),
        frame_height=_require_int(section, "frame_height", parent=parent, min_value=0),
        fps_limit=_require_int(section, "fps_limit", parent=parent, min_value=0),
    )


def _parse_detection(section: Mapping[str, Any]) -> DetectionConfig:
    parent = "detection"
    backend = _require_str(section, "backend", parent=parent)
    if backend not in ("ultralytics",):
        raise ConfigValidationError(
            "backend must be 'ultralytics'", path="detection.backend"
        )
    return DetectionConfig(
        backend=backend,
        model_name=_require_str(section, "model_name", parent=parent),
        model_path=_require_str(section, "model_path", parent=parent),
        confidence_threshold=_require_number(
            section, "confidence_threshold", parent=parent, min_value=0, max_value=1
        ),
        iou_threshold=_require_number(
            section, "iou_threshold", parent=parent, min_value=0, max_value=1
        ),
        device=_require_str(section, "device", parent=parent),
        detection_skip_frames=_require_int(
            section, "detection_skip_frames", parent=parent, min_value=0, max_value=120
        ),
        input_size=_require_int(section, "input_size", parent=parent, min_value=32, max_value=4096),
    )


def _parse_tracking(section: Mapping[str, Any]) -> TrackingConfig:
    parent = "tracking"
    tracker_type = _require_str(section, "tracker_type", parent=parent)
    if tracker_type not in ("bytetrack",):
        raise ConfigValidationError(
            "tracker_type must be 'bytetrack'", path="tracking.tracker_type"
        )
    return TrackingConfig(
        tracker_type=tracker_type,
        track_thresh=_require_number(
            section, "track_thresh", parent=parent, min_value=0, max_value=1
        ),
        track_buffer=_require_int(section, "track_buffer", parent=parent, min_value=1),
        match_thresh=_require_number(
            section, "match_thresh", parent=parent, min_value=0, max_value=1
        ),
        frame_rate=_require_int(section, "frame_rate", parent=parent, min_value=1, max_value=240),
    )


def _parse_recognition(section: Mapping[str, Any]) -> RecognitionConfig:
    parent = "recognition"
    return RecognitionConfig(
        model_name=_require_str(section, "model_name", parent=parent),
        embedding_dim=_require_int(section, "embedding_dim", parent=parent, min_value=1),
        similarity_threshold=_require_number(
            section, "similarity_threshold", parent=parent, min_value=0, max_value=1
        ),
        min_face_size_px=_require_int(section, "min_face_size_px", parent=parent, min_value=1),
        quality_min_score=_require_number(
            section, "quality_min_score", parent=parent, min_value=0, max_value=1
        ),
        providers=_require_str_list(section, "providers", parent=parent),
    )


def _parse_registration(section: Mapping[str, Any]) -> RegistrationConfig:
    parent = "registration"
    return RegistrationConfig(
        auto_register_unknown=_require_bool(section, "auto_register_unknown", parent=parent),
        min_detections_before_register=_require_int(
            section, "min_detections_before_register", parent=parent, min_value=1
        ),
        register_cooldown_sec=_require_number(
            section, "register_cooldown_sec", parent=parent, min_value=0
        ),
        crop_save_enabled=_require_bool(section, "crop_save_enabled", parent=parent),
        crop_dir=_require_str(section, "crop_dir", parent=parent),
    )


def _parse_entry_exit(section: Mapping[str, Any]) -> EntryExitConfig:
    parent = "entry_exit"
    mode = _require_str(section, "mode", parent=parent)
    if mode not in ("line_crossing",):
        raise ConfigValidationError(
            "Unsupported entry_exit.mode (expected 'line_crossing')", path="entry_exit.mode"
        )
    return EntryExitConfig(
        mode=mode,
        entry_line=_require_line(section, "entry_line", parent=parent),
        exit_line=_require_line(section, "exit_line", parent=parent),
        direction_entry=_require_str(section, "direction_entry", parent=parent),
        direction_exit=_require_str(section, "direction_exit", parent=parent),
        debounce_sec=_require_number(section, "debounce_sec", parent=parent, min_value=0),
        min_track_age_frames=_require_int(
            section, "min_track_age_frames", parent=parent, min_value=0
        ),
    )


def _parse_database(section: Mapping[str, Any]) -> DatabaseConfig:
    parent = "database"
    return DatabaseConfig(
        sqlite_path=_require_str(section, "sqlite_path", parent=parent),
        wal_mode=_require_bool(section, "wal_mode", parent=parent),
    )


def _parse_logging(section: Mapping[str, Any]) -> LoggingConfig:
    parent = "logging"
    level = _require_str(section, "level", parent=parent)
    allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level not in allowed:
        raise ConfigValidationError(
            f"level must be one of {sorted(allowed)}", path="logging.level"
        )
    return LoggingConfig(
        level=level,
        console_enabled=_require_bool(section, "console_enabled", parent=parent),
        file_path=_require_str(section, "file_path", parent=parent),
        rotate_max_bytes=_require_int(section, "rotate_max_bytes", parent=parent, min_value=1024),
        rotate_backup_count=_require_int(
            section, "rotate_backup_count", parent=parent, min_value=0
        ),
        log_detections=_require_bool(section, "log_detections", parent=parent),
        log_embeddings=_require_bool(section, "log_embeddings", parent=parent),
    )


def _parse_paths(section: Mapping[str, Any]) -> PathsConfig:
    parent = "paths"
    return PathsConfig(
        registered_faces_dir=_require_str(section, "registered_faces_dir", parent=parent),
        events_snapshot_dir=_require_str(section, "events_snapshot_dir", parent=parent),
        samples_dir=_require_str(section, "samples_dir", parent=parent),
    )


def _parse_performance(section: Mapping[str, Any]) -> PerformanceConfig:
    parent = "performance"
    return PerformanceConfig(
        use_half_precision=_require_bool(section, "use_half_precision", parent=parent),
        num_threads=_require_int(section, "num_threads", parent=parent, min_value=0),
        max_pipeline_queue_size=_require_int(
            section, "max_pipeline_queue_size", parent=parent, min_value=1
        ),
    )


def validate_config_data(raw: Mapping[str, Any]) -> AppConfig:
    """Validate parsed JSON and return a typed AppConfig (config_path unset)."""
    project_name = _require_str(raw, "project_name", parent="")
    version = _require_str(raw, "version", parent="")

    return AppConfig(
        project_name=project_name,
        version=version,
        input=_parse_input(_require_mapping(raw.get("input"), "input")),
        detection=_parse_detection(_require_mapping(raw.get("detection"), "detection")),
        tracking=_parse_tracking(_require_mapping(raw.get("tracking"), "tracking")),
        recognition=_parse_recognition(_require_mapping(raw.get("recognition"), "recognition")),
        registration=_parse_registration(_require_mapping(raw.get("registration"), "registration")),
        entry_exit=_parse_entry_exit(_require_mapping(raw.get("entry_exit"), "entry_exit")),
        database=_parse_database(_require_mapping(raw.get("database"), "database")),
        logging=_parse_logging(_require_mapping(raw.get("logging"), "logging")),
        paths=_parse_paths(_require_mapping(raw.get("paths"), "paths")),
        performance=_parse_performance(_require_mapping(raw.get("performance"), "performance")),
        config_path=DEFAULT_CONFIG_PATH,
    )


def load_config(path: str | Path | None = None) -> AppConfig:
    """
    Load config.json from ``path``, ``CONFIG_PATH`` env var, or project default.

    Raises:
        ConfigValidationError: Invalid or missing configuration.
        FileNotFoundError: Config file does not exist.
        json.JSONDecodeError: Malformed JSON.
    """
    if path is None:
        env_path = os.environ.get(_ENV_CONFIG_PATH)
        config_path = Path(env_path) if env_path else DEFAULT_CONFIG_PATH
    else:
        config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, dict):
        raise ConfigValidationError("Root of config.json must be a JSON object")

    config = validate_config_data(raw)
    return AppConfig(
        project_name=config.project_name,
        version=config.version,
        input=config.input,
        detection=config.detection,
        tracking=config.tracking,
        recognition=config.recognition,
        registration=config.registration,
        entry_exit=config.entry_exit,
        database=config.database,
        logging=config.logging,
        paths=config.paths,
        performance=config.performance,
        config_path=config_path.resolve(),
    )
