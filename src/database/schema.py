"""SQLite DDL for the face tracker (see ARCHITECTURE.md)."""

SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- Persons: one row per unique visitor identity (Face ID)
CREATE TABLE IF NOT EXISTS persons (
    face_id           TEXT PRIMARY KEY,
    display_label     TEXT,
    first_seen_at     TEXT NOT NULL,
    last_seen_at      TEXT NOT NULL,
    total_visits      INTEGER NOT NULL DEFAULT 0,
    is_active         INTEGER NOT NULL DEFAULT 1,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    embedding_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id           TEXT NOT NULL REFERENCES persons(face_id) ON DELETE CASCADE,
    model_name        TEXT NOT NULL,
    model_version     TEXT,
    dimension         INTEGER NOT NULL,
    embedding         BLOB NOT NULL,
    embedding_hash    TEXT NOT NULL UNIQUE,
    quality_score     REAL,
    source_image_path TEXT,
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_face_embeddings_face_id ON face_embeddings(face_id);

CREATE TABLE IF NOT EXISTS track_sessions (
    track_session_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id          INTEGER NOT NULL,
    face_id           TEXT REFERENCES persons(face_id),
    started_at        TEXT NOT NULL,
    ended_at          TEXT,
    source_type       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type        TEXT NOT NULL CHECK (event_type IN (
                          'REGISTER', 'RECOGNIZE', 'ENTRY', 'EXIT',
                          'TRACK_LOST', 'ERROR', 'SYSTEM'
                      )),
    face_id           TEXT REFERENCES persons(face_id),
    track_id          INTEGER,
    visit_id          INTEGER,
    confidence        REAL,
    image_path        TEXT,
    metadata_json     TEXT,
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_face_id ON events(face_id);
CREATE INDEX IF NOT EXISTS idx_events_type_time ON events(event_type, created_at);

CREATE TABLE IF NOT EXISTS visits (
    visit_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id           TEXT NOT NULL REFERENCES persons(face_id),
    entry_event_id    INTEGER,
    exit_event_id     INTEGER,
    entry_at          TEXT,
    exit_at           TEXT,
    status            TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    UNIQUE (face_id, entry_event_id)
);

CREATE INDEX IF NOT EXISTS idx_visits_face_open ON visits(face_id, status);

CREATE TABLE IF NOT EXISTS daily_stats (
    stat_date         TEXT NOT NULL,
    unique_visitors   INTEGER NOT NULL DEFAULT 0,
    total_entries     INTEGER NOT NULL DEFAULT 0,
    total_exits       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (stat_date)
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version           INTEGER PRIMARY KEY,
    applied_at        TEXT NOT NULL
);
"""
