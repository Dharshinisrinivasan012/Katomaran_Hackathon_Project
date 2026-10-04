"""Structured application logging setup.

Sets up two sinks:
  1. Console handler  — human-readable, level from config.
  2. RotatingFileHandler — logs/events.log, JSON-like lines for audit.

Call ``setup_logging(config)`` once at application startup.

Events written to the log file include:
  - face registration (REGISTER)
  - face recognition (RECOGNIZE)
  - tracking (TRACK, TRACK_LOST)
  - entry / exit (ENTRY, EXIT)
  - embedding generation (EMBED) when enabled
  - errors / system events (ERROR, SYSTEM)
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from src.config.loader import LoggingConfig, PROJECT_ROOT

_EVENT_LOGGER_NAME = "events"
_ROOT_LOGGER_NAME = "src"

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_EVENT_FORMAT = "%(asctime)s %(message)s"


def _resolve_path(file_path: str) -> Path:
    p = Path(file_path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def setup_logging(config: LoggingConfig) -> logging.Logger:
    """Configure root src logger and events.log file logger.

    Returns the events logger so callers can write structured audit lines.
    """
    level = getattr(logging, config.level.upper(), logging.INFO)

    # ── root src logger ────────────────────────────────────────────────
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(level)
    root.handlers.clear()

    if config.console_enabled:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(ch)

    # ── events.log rotating file ────────────────────────────────────────
    log_path = _resolve_path(config.file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.handlers.RotatingFileHandler(
        str(log_path),
        maxBytes=config.rotate_max_bytes,
        backupCount=config.rotate_backup_count,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)  # capture everything — filter at call site
    fh.setFormatter(logging.Formatter(_EVENT_FORMAT))

    event_logger = logging.getLogger(_EVENT_LOGGER_NAME)
    event_logger.setLevel(logging.DEBUG)
    event_logger.handlers.clear()
    event_logger.addHandler(fh)
    event_logger.propagate = False  # don't echo to console twice

    return event_logger


def get_event_logger() -> logging.Logger:
    """Return the events audit logger (no-op until setup_logging called)."""
    return logging.getLogger(_EVENT_LOGGER_NAME)
