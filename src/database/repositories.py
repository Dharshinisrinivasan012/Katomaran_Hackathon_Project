"""CRUD repositories for persons, embeddings, events, and visits."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from src.database.connection import utc_now_iso
from src.database.exceptions import (
    DuplicateEmbeddingError,
    PersonNotFoundError,
    VisitNotFoundError,
)

EVENT_TYPES = frozenset(
    {"REGISTER", "RECOGNIZE", "ENTRY", "EXIT", "TRACK_LOST", "ERROR", "SYSTEM"}
)


@dataclass(frozen=True)
class PersonRow:
    face_id: str
    display_label: str | None
    first_seen_at: str
    last_seen_at: str
    total_visits: int
    is_active: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class EmbeddingRow:
    embedding_id: int
    face_id: str
    model_name: str
    model_version: str | None
    dimension: int
    embedding: bytes
    embedding_hash: str
    quality_score: float | None
    source_image_path: str | None
    created_at: str


@dataclass(frozen=True)
class EventRow:
    event_id: int
    event_type: str
    face_id: str | None
    track_id: int | None
    visit_id: int | None
    confidence: float | None
    image_path: str | None
    metadata_json: str | None
    created_at: str


@dataclass(frozen=True)
class VisitRow:
    visit_id: int
    face_id: str
    entry_event_id: int | None
    exit_event_id: int | None
    entry_at: str | None
    exit_at: str | None
    status: str
    created_at: str
    updated_at: str


def _row_to_person(row: sqlite3.Row) -> PersonRow:
    return PersonRow(
        face_id=row["face_id"],
        display_label=row["display_label"],
        first_seen_at=row["first_seen_at"],
        last_seen_at=row["last_seen_at"],
        total_visits=row["total_visits"],
        is_active=bool(row["is_active"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def register_person(
    conn: sqlite3.Connection,
    face_id: str,
    *,
    display_label: str | None = None,
    seen_at: str | None = None,
) -> PersonRow:
    """Insert a new person row. Raises sqlite3.IntegrityError if face_id exists."""
    now = seen_at or utc_now_iso()
    conn.execute(
        """
        INSERT INTO persons (
            face_id, display_label, first_seen_at, last_seen_at,
            total_visits, is_active, created_at, updated_at
        ) VALUES (?, ?, ?, ?, 0, 1, ?, ?)
        """,
        (face_id, display_label, now, now, now, now),
    )
    person = get_person_by_face_id(conn, face_id)
    assert person is not None
    return person


def get_person_by_face_id(conn: sqlite3.Connection, face_id: str) -> PersonRow | None:
    row = conn.execute("SELECT * FROM persons WHERE face_id = ?", (face_id,)).fetchone()
    return _row_to_person(row) if row else None


def require_person(conn: sqlite3.Connection, face_id: str) -> PersonRow:
    person = get_person_by_face_id(conn, face_id)
    if person is None:
        raise PersonNotFoundError(f"Unknown face_id: {face_id}")
    return person


def touch_person_seen(conn: sqlite3.Connection, face_id: str, *, seen_at: str | None = None) -> None:
    """Update last_seen_at for an existing person."""
    require_person(conn, face_id)
    now = seen_at or utc_now_iso()
    conn.execute(
        "UPDATE persons SET last_seen_at = ?, updated_at = ? WHERE face_id = ?",
        (now, now, face_id),
    )


def increment_person_visits(conn: sqlite3.Connection, face_id: str) -> None:
    """Increment total_visits (typically on new visit ENTRY)."""
    require_person(conn, face_id)
    now = utc_now_iso()
    conn.execute(
        """
        UPDATE persons
        SET total_visits = total_visits + 1, updated_at = ?
        WHERE face_id = ?
        """,
        (now, face_id),
    )


def add_embedding(
    conn: sqlite3.Connection,
    face_id: str,
    *,
    model_name: str,
    dimension: int,
    embedding: bytes,
    embedding_hash: str,
    model_version: str | None = None,
    quality_score: float | None = None,
    source_image_path: str | None = None,
    created_at: str | None = None,
) -> int:
    """Insert embedding; returns embedding_id. Raises DuplicateEmbeddingError on hash clash."""
    require_person(conn, face_id)
    now = created_at or utc_now_iso()
    try:
        cur = conn.execute(
            """
            INSERT INTO face_embeddings (
                face_id, model_name, model_version, dimension, embedding,
                embedding_hash, quality_score, source_image_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                face_id,
                model_name,
                model_version,
                dimension,
                embedding,
                embedding_hash,
                quality_score,
                source_image_path,
                now,
            ),
        )
    except sqlite3.IntegrityError as exc:
        if "embedding_hash" in str(exc).lower() or "UNIQUE" in str(exc):
            raise DuplicateEmbeddingError(
                f"Duplicate embedding_hash: {embedding_hash}"
            ) from exc
        raise
    return int(cur.lastrowid)


def list_embeddings_for_gallery(conn: sqlite3.Connection) -> list[EmbeddingRow]:
    """Return all embeddings for gallery search (recognition phase)."""
    rows = conn.execute(
        "SELECT * FROM face_embeddings ORDER BY face_id, embedding_id"
    ).fetchall()
    return [_row_to_embedding(r) for r in rows]


def list_embeddings_for_face(conn: sqlite3.Connection, face_id: str) -> list[EmbeddingRow]:
    rows = conn.execute(
        "SELECT * FROM face_embeddings WHERE face_id = ? ORDER BY embedding_id",
        (face_id,),
    ).fetchall()
    return [_row_to_embedding(r) for r in rows]


def _row_to_embedding(row: sqlite3.Row) -> EmbeddingRow:
    return EmbeddingRow(
        embedding_id=row["embedding_id"],
        face_id=row["face_id"],
        model_name=row["model_name"],
        model_version=row["model_version"],
        dimension=row["dimension"],
        embedding=row["embedding"],
        embedding_hash=row["embedding_hash"],
        quality_score=row["quality_score"],
        source_image_path=row["source_image_path"],
        created_at=row["created_at"],
    )


def create_event(
    conn: sqlite3.Connection,
    event_type: str,
    *,
    face_id: str | None = None,
    track_id: int | None = None,
    visit_id: int | None = None,
    confidence: float | None = None,
    image_path: str | None = None,
    metadata: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> int:
    """Append a business event; returns event_id."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Invalid event_type: {event_type}")
    if face_id is not None:
        require_person(conn, face_id)
    now = created_at or utc_now_iso()
    meta_json = json.dumps(metadata) if metadata is not None else None
    cur = conn.execute(
        """
        INSERT INTO events (
            event_type, face_id, track_id, visit_id, confidence,
            image_path, metadata_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_type,
            face_id,
            track_id,
            visit_id,
            confidence,
            image_path,
            meta_json,
            now,
        ),
    )
    return int(cur.lastrowid)


def open_visit(
    conn: sqlite3.Connection,
    face_id: str,
    *,
    entry_event_id: int | None = None,
    entry_at: str | None = None,
    increment_total_visits: bool = True,
) -> int:
    """
    Create an open visit for face_id.

    Raises sqlite3.IntegrityError if an open visit already exists (app should check first).
    """
    require_person(conn, face_id)
    existing = get_open_visit(conn, face_id)
    if existing is not None:
        raise ValueError(
            f"Open visit already exists for {face_id} (visit_id={existing.visit_id})"
        )
    now = entry_at or utc_now_iso()
    cur = conn.execute(
        """
        INSERT INTO visits (
            face_id, entry_event_id, entry_at, status, created_at, updated_at
        ) VALUES (?, ?, ?, 'open', ?, ?)
        """,
        (face_id, entry_event_id, now, now, now),
    )
    visit_id = int(cur.lastrowid)
    if increment_total_visits:
        increment_person_visits(conn, face_id)
    return visit_id


def close_visit(
    conn: sqlite3.Connection,
    visit_id: int,
    *,
    exit_event_id: int | None = None,
    exit_at: str | None = None,
) -> VisitRow:
    """Close an open visit."""
    row = conn.execute("SELECT * FROM visits WHERE visit_id = ?", (visit_id,)).fetchone()
    if row is None:
        raise VisitNotFoundError(f"Visit not found: {visit_id}")
    if row["status"] != "open":
        raise VisitNotFoundError(f"Visit {visit_id} is not open")
    now = exit_at or utc_now_iso()
    conn.execute(
        """
        UPDATE visits
        SET exit_event_id = ?, exit_at = ?, status = 'closed', updated_at = ?
        WHERE visit_id = ?
        """,
        (exit_event_id, now, now, visit_id),
    )
    updated = get_visit(conn, visit_id)
    assert updated is not None
    return updated


def get_open_visit(conn: sqlite3.Connection, face_id: str) -> VisitRow | None:
    row = conn.execute(
        "SELECT * FROM visits WHERE face_id = ? AND status = 'open' LIMIT 1",
        (face_id,),
    ).fetchone()
    return _row_to_visit(row) if row else None


def get_visit(conn: sqlite3.Connection, visit_id: int) -> VisitRow | None:
    row = conn.execute("SELECT * FROM visits WHERE visit_id = ?", (visit_id,)).fetchone()
    return _row_to_visit(row) if row else None


def _row_to_visit(row: sqlite3.Row) -> VisitRow:
    return VisitRow(
        visit_id=row["visit_id"],
        face_id=row["face_id"],
        entry_event_id=row["entry_event_id"],
        exit_event_id=row["exit_event_id"],
        entry_at=row["entry_at"],
        exit_at=row["exit_at"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def create_track_session(
    conn: sqlite3.Connection,
    track_id: int,
    source_type: str,
    *,
    face_id: str | None = None,
    started_at: str | None = None,
) -> int:
    """Optional debug table for track → face mapping."""
    if face_id is not None:
        require_person(conn, face_id)
    now = started_at or utc_now_iso()
    cur = conn.execute(
        """
        INSERT INTO track_sessions (track_id, face_id, started_at, source_type)
        VALUES (?, ?, ?, ?)
        """,
        (track_id, face_id, now, source_type),
    )
    return int(cur.lastrowid)
