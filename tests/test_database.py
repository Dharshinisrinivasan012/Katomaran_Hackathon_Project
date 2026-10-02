"""Unit tests for SQLite schema and repositories."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.database import (
    DuplicateEmbeddingError,
    PersonNotFoundError,
    add_embedding,
    close_visit,
    connection_scope,
    create_event,
    get_open_visit,
    get_person_by_face_id,
    initialize_schema,
    list_embeddings_for_gallery,
    open_initialized,
    open_visit,
    register_person,
    touch_person_seen,
)
from src.database.connection import connect


@pytest.fixture
def db_conn(temp_db_path: Path):
    conn = open_initialized(temp_db_path, wal_mode=False)
    yield conn
    conn.close()


def test_schema_creates_all_tables(db_conn: sqlite3.Connection) -> None:
    rows = db_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    names = {r[0] for r in rows}
    assert "persons" in names
    assert "face_embeddings" in names
    assert "events" in names
    assert "visits" in names
    assert "track_sessions" in names
    assert "daily_stats" in names
    assert "schema_migrations" in names


def test_register_person_and_lookup(db_conn: sqlite3.Connection) -> None:
    person = register_person(db_conn, "FT-001", display_label="Visitor A")
    assert person.face_id == "FT-001"
    assert person.total_visits == 0
    fetched = get_person_by_face_id(db_conn, "FT-001")
    assert fetched is not None
    assert fetched.display_label == "Visitor A"


def test_add_embedding_and_gallery(db_conn: sqlite3.Connection) -> None:
    register_person(db_conn, "FT-002")
    blob = b"\x00" * (512 * 4)
    eid = add_embedding(
        db_conn,
        "FT-002",
        model_name="buffalo_l",
        dimension=512,
        embedding=blob,
        embedding_hash="abc123",
    )
    assert eid >= 1
    gallery = list_embeddings_for_gallery(db_conn)
    assert len(gallery) == 1
    assert gallery[0].embedding_hash == "abc123"


def test_duplicate_embedding_hash(db_conn: sqlite3.Connection) -> None:
    register_person(db_conn, "FT-003")
    blob = b"\x01" * 8
    add_embedding(
        db_conn,
        "FT-003",
        model_name="buffalo_l",
        dimension=2,
        embedding=blob,
        embedding_hash="dup-hash",
    )
    with pytest.raises(DuplicateEmbeddingError):
        add_embedding(
            db_conn,
            "FT-003",
            model_name="buffalo_l",
            dimension=2,
            embedding=blob,
            embedding_hash="dup-hash",
        )


def test_create_event_requires_person(db_conn: sqlite3.Connection) -> None:
    with pytest.raises(PersonNotFoundError):
        create_event(db_conn, "REGISTER", face_id="missing")


def test_visit_open_close_and_total_visits(db_conn: sqlite3.Connection) -> None:
    register_person(db_conn, "FT-004")
    entry_id = create_event(db_conn, "ENTRY", face_id="FT-004", track_id=1)
    visit_id = open_visit(db_conn, "FT-004", entry_event_id=entry_id)
    person = get_person_by_face_id(db_conn, "FT-004")
    assert person is not None
    assert person.total_visits == 1
    open_row = get_open_visit(db_conn, "FT-004")
    assert open_row is not None
    assert open_row.visit_id == visit_id
    exit_id = create_event(db_conn, "EXIT", face_id="FT-004", visit_id=visit_id)
    closed = close_visit(db_conn, visit_id, exit_event_id=exit_id)
    assert closed.status == "closed"
    assert closed.exit_event_id == exit_id
    assert get_open_visit(db_conn, "FT-004") is None


def test_touch_person_seen(db_conn: sqlite3.Connection) -> None:
    register_person(db_conn, "FT-005", seen_at="2026-10-02T10:00:00+00:00")
    touch_person_seen(db_conn, "FT-005", seen_at="2026-10-02T11:00:00+00:00")
    person = get_person_by_face_id(db_conn, "FT-005")
    assert person is not None
    assert person.last_seen_at == "2026-10-02T11:00:00+00:00"


def test_connection_scope_commits(temp_db_path: Path) -> None:
    with connection_scope(temp_db_path, wal_mode=False) as conn:
        initialize_schema(conn)
        register_person(conn, "FT-scope")
    conn2 = connect(temp_db_path, wal_mode=False)
    try:
        assert get_person_by_face_id(conn2, "FT-scope") is not None
    finally:
        conn2.close()
