"""AMV Tools schema (v4 onward, on top of legacy Scene Scout v3 schema).

Layout:
    videos               -- replaces processed_videos with richer metadata
    scenes               -- replaces scene_embeddings, adds parent_scene_id + proxy_path
    scene_tags           -- new: per-scene wd-tagger v3 tags with confidence
    settings             -- new: KV store for app settings
    index_queue          -- inherited
    image_embeddings     -- kept for backward compat, unused going forward
    search_history       -- new: persisted recent queries
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA_VERSION = 4

V4_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    duration_ms INTEGER,
    fps REAL,
    resolution TEXT,
    modified_at REAL NOT NULL,
    model_version TEXT,
    status TEXT DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS scenes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    scene_index INTEGER NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    parent_scene_id INTEGER REFERENCES scenes(id) ON DELETE SET NULL,
    embedding BLOB,
    thumbnail BLOB,
    proxy_path TEXT
);

CREATE TABLE IF NOT EXISTS scene_tags (
    scene_id INTEGER NOT NULL REFERENCES scenes(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    category TEXT,
    confidence REAL NOT NULL,
    PRIMARY KEY (scene_id, tag)
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS index_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    is_directory INTEGER NOT NULL DEFAULT 0,
    recursive INTEGER NOT NULL DEFAULT 1,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS search_history (
    query TEXT PRIMARY KEY,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    use_count INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_scenes_video ON scenes(video_id);
CREATE INDEX IF NOT EXISTS idx_scenes_parent ON scenes(parent_scene_id);
CREATE INDEX IF NOT EXISTS idx_scene_tags_tag ON scene_tags(tag, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_scene_tags_scene ON scene_tags(scene_id);
CREATE INDEX IF NOT EXISTS idx_search_history_used ON search_history(last_used DESC);
"""


@contextmanager
def get_conn(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def current_version(db_path: str | Path) -> int:
    with get_conn(db_path) as conn:
        return conn.execute("PRAGMA user_version").fetchone()[0]


def init_db(db_path: str | Path) -> None:
    """Apply schema and migrations. Idempotent."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with get_conn(db_path) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version >= SCHEMA_VERSION:
            return
        if version == 0:
            conn.executescript(V4_SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            return
        if version < 4:
            _migrate_v3_to_v4(conn)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def _migrate_v3_to_v4(conn: sqlite3.Connection) -> None:
    """Migrate from Scene Scout v3 (processed_videos + scene_embeddings) to v4."""
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.executescript(V4_SCHEMA)

    has_legacy = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='processed_videos'"
    ).fetchone() is not None
    if not has_legacy:
        return

    conn.execute(
        """
        INSERT INTO videos (id, filepath, display_name, duration_ms, fps, resolution,
                            modified_at, model_version, status)
        SELECT id,
               filepath,
               REPLACE(SUBSTR(filepath, RTRIM(filepath, REPLACE(filepath, '/', '')) - LENGTH(filepath)),
                       '/', '') AS display_name,
               NULL, NULL, NULL,
               modified_at, model_version,
               COALESCE(status, 'completed')
          FROM processed_videos
        """
    )

    conn.execute(
        """
        INSERT INTO scenes (id, video_id, scene_index, start_ms, end_ms, embedding, thumbnail)
        SELECT id, video_id, scene_index, start_time_ms, end_time_ms, embedding, thumbnail
          FROM scene_embeddings
        """
    )

    conn.execute("DROP TABLE scene_embeddings")
    conn.execute("DROP TABLE processed_videos")
    conn.execute("PRAGMA foreign_keys = ON")
