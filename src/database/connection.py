"""SQLite connection helpers and schema initialization."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from src.database.schema import SCHEMA_SQL, SCHEMA_VERSION


def utc_now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def connect(
    db_path: str | Path,
    *,
    wal_mode: bool = True,
) -> sqlite3.Connection:
    """
    Open a SQLite connection with foreign keys enabled.

    Caller is responsible for closing the connection.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if wal_mode:
        conn.execute("PRAGMA journal_mode = WAL")
    return conn


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Create tables and record schema version on first run."""
    conn.executescript(SCHEMA_SQL)
    row = conn.execute(
        "SELECT MAX(version) AS v FROM schema_migrations"
    ).fetchone()
    current = row["v"] if row and row["v"] is not None else None
    if current is None:
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, utc_now_iso()),
        )
    elif current < SCHEMA_VERSION:
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, utc_now_iso()),
        )


@contextmanager
def connection_scope(
    db_path: str | Path,
    *,
    wal_mode: bool = True,
) -> Iterator[sqlite3.Connection]:
    """Context manager: connect, commit on success, rollback on error."""
    conn = connect(db_path, wal_mode=wal_mode)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def open_initialized(
    db_path: str | Path,
    *,
    wal_mode: bool = True,
) -> sqlite3.Connection:
    """Open connection and ensure schema exists."""
    conn = connect(db_path, wal_mode=wal_mode)
    initialize_schema(conn)
    conn.commit()
    return conn
