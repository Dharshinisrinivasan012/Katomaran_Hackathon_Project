"""Visit lifecycle manager — bridges EntryExitFSM events to the database.

Responsibilities:
  - Persist ENTRY events: create_event → open_visit (increments total_visits once).
  - Persist EXIT events: create_event → close_visit.
  - Save a face crop snapshot to the filesystem for every ENTRY and EXIT.
  - Write a structured line to the events audit logger.
  - Never create duplicate visits; never double-increment total_visits.
  - Return the summary counts for final reporting.

CRITICAL identity rule:
  All operations are keyed on persistent face_id, NOT track_id.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from src.config.loader import PROJECT_ROOT, LoggingConfig, PathsConfig
from src.database.connection import utc_now_iso
from src.database.exceptions import VisitNotFoundError
from src.database.repositories import (
    close_visit,
    create_event,
    get_open_visit,
    get_person_by_face_id,
    open_visit,
)
from src.logging.setup import get_event_logger
from src.pipeline.entry_exit import EntryExitEvent, EntryExitFSM

logger = logging.getLogger(__name__)


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _save_event_snapshot(
    crop: np.ndarray | None,
    face_id: str,
    event_type: str,
    events_dir: str,
) -> str | None:
    """Write crop to data/events/YYYY-MM-DD/{entry|exit}_{face_id_suffix}.jpg.

    Returns relative path (from project root) or None on failure.
    """
    if crop is None or crop.size == 0:
        return None
    try:
        today = date.today().isoformat()
        short_id = face_id.replace("FT-", "")[:16]
        filename = f"{event_type.lower()}_{short_id}.jpg"
        dest_dir = _resolve(events_dir) / today
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename
        # Avoid overwriting if multiple exits happen same day — append numeric suffix
        counter = 0
        while dest.exists():
            counter += 1
            dest = dest_dir / f"{event_type.lower()}_{short_id}_{counter}.jpg"
        ok = cv2.imwrite(str(dest), crop)
        if not ok:
            logger.warning("imwrite failed for %s", dest)
            return None
        rel = str(dest.relative_to(PROJECT_ROOT))
        logger.debug("Saved %s snapshot → %s", event_type, rel)
        return rel
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save %s snapshot for %s: %s", event_type, face_id, exc)
        return None


@dataclass
class VisitSummary:
    """Accumulated counts for end-of-run reporting."""
    total_entries: int = 0
    total_exits:   int = 0
    total_new_persons: int = 0   # maintained externally by pipeline


class VisitManager:
    """Wires EntryExitFSM events into the SQLite visit lifecycle.

    Usage::

        vm = VisitManager(conn, fsm, paths_config, logging_config)
        for ev in fsm_events:
            vm.handle(ev)
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        fsm: EntryExitFSM,
        paths_config: PathsConfig,
        *,
        source_label: str = "video",
    ) -> None:
        self._conn = conn
        self._fsm = fsm
        self._paths = paths_config
        self._source_label = source_label
        self._event_log = get_event_logger()
        self.summary = VisitSummary()

    def handle(self, event: EntryExitEvent) -> None:
        """Persist one ENTRY or EXIT event to the database and filesystem."""
        now = utc_now_iso()
        face_id = event.face_id
        track_id = event.track_id

        # Safety: must be a registered person
        if not face_id or get_person_by_face_id(self._conn, face_id) is None:
            logger.warning(
                "%s event ignored — face_id=%r not in database",
                event.event_type,
                face_id,
            )
            return

        if event.event_type == "ENTRY":
            self._handle_entry(event, face_id, track_id, now)
        elif event.event_type == "EXIT":
            self._handle_exit(event, face_id, track_id, now)
        else:
            logger.warning("Unknown event_type=%r in VisitManager.handle", event.event_type)

    # ------------------------------------------------------------------
    # ENTRY
    # ------------------------------------------------------------------

    def _handle_entry(
        self,
        event: EntryExitEvent,
        face_id: str,
        track_id: int | None,
        now: str,
    ) -> None:
        # Guard: don't open a second visit if one is already open
        existing = get_open_visit(self._conn, face_id)
        if existing is not None:
            logger.debug(
                "ENTRY suppressed for face_id=%s — open visit %d already exists",
                face_id,
                existing.visit_id,
            )
            return

        # Save snapshot before creating DB rows
        snap_path = _save_event_snapshot(
            event.crop, face_id, "ENTRY", self._paths.events_snapshot_dir
        )

        # Create ENTRY event
        entry_event_id = create_event(
            self._conn,
            "ENTRY",
            face_id=face_id,
            track_id=track_id,
            image_path=snap_path,
            metadata={
                "frame_index": event.frame_index,
                "source": self._source_label,
            },
            created_at=now,
        )

        # Open visit — increments persons.total_visits
        visit_id = open_visit(
            self._conn,
            face_id,
            entry_event_id=entry_event_id,
            entry_at=now,
            increment_total_visits=True,
        )

        # Store visit_id back in FSM so EXIT can reference it
        self._fsm.set_visit_id(face_id, visit_id)

        self._conn.commit()
        self.summary.total_entries += 1

        self._event_log.info(
            '{"event":"ENTRY","face_id":"%s","track_id":%s,"visit_id":%d,'
            '"frame":%d,"snapshot":"%s","ts":"%s"}',
            face_id,
            track_id if track_id is not None else "null",
            visit_id,
            event.frame_index,
            snap_path or "",
            now,
        )
        logger.info(
            "ENTRY  face_id=%s  visit_id=%d  frame=%d",
            face_id, visit_id, event.frame_index,
        )

    # ------------------------------------------------------------------
    # EXIT
    # ------------------------------------------------------------------

    def _handle_exit(
        self,
        event: EntryExitEvent,
        face_id: str,
        track_id: int | None,
        now: str,
    ) -> None:
        visit_id = self._fsm.get_visit_id(face_id)
        if visit_id is None:
            # Fallback: look up from DB
            open_row = get_open_visit(self._conn, face_id)
            if open_row is None:
                logger.debug(
                    "EXIT ignored for face_id=%s — no open visit found", face_id
                )
                return
            visit_id = open_row.visit_id

        snap_path = _save_event_snapshot(
            event.crop, face_id, "EXIT", self._paths.events_snapshot_dir
        )

        exit_event_id = create_event(
            self._conn,
            "EXIT",
            face_id=face_id,
            track_id=track_id,
            visit_id=visit_id,
            image_path=snap_path,
            metadata={
                "frame_index": event.frame_index,
                "source": self._source_label,
            },
            created_at=now,
        )

        try:
            closed = close_visit(
                self._conn,
                visit_id,
                exit_event_id=exit_event_id,
                exit_at=now,
            )
        except VisitNotFoundError as exc:
            logger.warning("Could not close visit %d: %s", visit_id, exc)
            self._conn.rollback()
            return

        self._conn.commit()
        self.summary.total_exits += 1

        self._event_log.info(
            '{"event":"EXIT","face_id":"%s","track_id":%s,"visit_id":%d,'
            '"frame":%d,"snapshot":"%s","ts":"%s"}',
            face_id,
            track_id if track_id is not None else "null",
            visit_id,
            event.frame_index,
            snap_path or "",
            now,
        )
        logger.info(
            "EXIT   face_id=%s  visit_id=%d  frame=%d  duration=%s",
            face_id,
            visit_id,
            event.frame_index,
            _duration(closed.entry_at, closed.exit_at),
        )

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def handle_all(self, events: list[EntryExitEvent]) -> None:
        for ev in events:
            self.handle(ev)


def _duration(entry_at: str | None, exit_at: str | None) -> str:
    if not entry_at or not exit_at:
        return "unknown"
    try:
        from datetime import datetime, timezone
        fmt = "%Y-%m-%dT%H:%M:%S%z"
        t1 = datetime.fromisoformat(entry_at)
        t2 = datetime.fromisoformat(exit_at)
        secs = int((t2 - t1).total_seconds())
        return f"{secs}s"
    except Exception:
        return "unknown"
