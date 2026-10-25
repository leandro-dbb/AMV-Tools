"""Schema creation + upgrade paths (fresh install and older databases)."""
from __future__ import annotations

import sqlite3

from backend.db import schema


def _tables(db_path) -> set[str]:
    with sqlite3.connect(str(db_path)) as conn:
        return {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }


def _columns(db_path, table: str) -> set[str]:
    with sqlite3.connect(str(db_path)) as conn:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def test_fresh_database_reaches_current_version(tmp_path):
    db = tmp_path / "fresh.db"
    schema.init_db(db)

    assert schema.current_version(db) == schema.SCHEMA_VERSION
    tables = _tables(db)
    assert {
        "videos", "scenes", "scene_tags", "settings", "index_queue",
        "search_history", "derush_folders", "derush_items",
        "derush_item_folders",
    } <= tables
    assert "group_name" in _columns(db, "videos")
    assert "derushed" in _columns(db, "videos")
    assert "favorite" in _columns(db, "derush_items")


def test_init_is_idempotent(tmp_path):
    db = tmp_path / "idempotent.db"
    schema.init_db(db)
    schema.init_db(db)
    schema.init_db(db)
    assert schema.current_version(db) == schema.SCHEMA_VERSION
    with schema.get_conn(db) as conn:
        # No duplicated tables/indexes on repeated runs.
        videos = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    assert videos == 0


def test_migrates_legacy_v3_database(tmp_path):
    db = tmp_path / "v3.db"
    with sqlite3.connect(str(db)) as conn:
        conn.executescript(
            """
            CREATE TABLE processed_videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT UNIQUE NOT NULL,
                modified_at REAL NOT NULL,
                model_version TEXT,
                status TEXT
            );
            CREATE TABLE scene_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                scene_index INTEGER NOT NULL,
                start_time_ms INTEGER NOT NULL,
                end_time_ms INTEGER NOT NULL,
                embedding BLOB,
                thumbnail BLOB
            );
            INSERT INTO processed_videos (filepath, modified_at, model_version, status)
                VALUES ('/library/Episode 01.mkv', 123.0, 'legacy', 'completed');
            INSERT INTO scene_embeddings
                (video_id, scene_index, start_time_ms, end_time_ms, embedding, thumbnail)
                VALUES (1, 0, 0, 1000, X'00', X'01'),
                       (1, 1, 1000, 2500, X'00', X'02');
            PRAGMA user_version = 3;
            """
        )

    schema.init_db(db)

    assert schema.current_version(db) == schema.SCHEMA_VERSION
    tables = _tables(db)
    assert "processed_videos" not in tables
    assert "scene_embeddings" not in tables
    with schema.get_conn(db) as conn:
        video = conn.execute(
            "SELECT filepath, status, display_name FROM videos"
        ).fetchone()
        scenes = conn.execute(
            "SELECT scene_index, start_ms, end_ms FROM scenes ORDER BY scene_index"
        ).fetchall()
    assert video[0] == "/library/Episode 01.mkv"
    assert video[1] == "completed"
    assert video[2] and video[2].endswith("Episode 01.mkv")
    assert [tuple(s) for s in scenes] == [(0, 0, 1000), (1, 1000, 2500)]
    # New v6+ columns exist after the legacy import.
    assert "group_name" in _columns(db, "videos")
    assert "favorite" in _columns(db, "derush_items")


def test_migrates_v4_database_and_backfills_derush_folder(tmp_path):
    db = tmp_path / "v4.db"
    with sqlite3.connect(str(db)) as conn:
        conn.executescript(schema.V4_SCHEMA)
        conn.executescript(schema.V5_SCHEMA)
        conn.execute(
            "INSERT INTO videos (filepath, display_name, modified_at, status) "
            "VALUES ('/x/ep.mkv', 'ep', 1.0, 'completed')"
        )
        conn.execute(
            "INSERT INTO scenes (video_id, scene_index, start_ms, end_ms) "
            "VALUES (1, 0, 0, 1000)"
        )
        conn.execute("INSERT INTO derush_folders (name) VALUES ('S23')")
        conn.execute(
            "INSERT INTO derush_items (scene_id, folder_id) VALUES (1, 1)"
        )
        conn.execute("PRAGMA user_version = 4")

    schema.init_db(db)

    assert schema.current_version(db) == schema.SCHEMA_VERSION
    with schema.get_conn(db) as conn:
        linked = conn.execute(
            "SELECT item_id, folder_id FROM derush_item_folders"
        ).fetchall()
        fav = conn.execute("SELECT favorite FROM derush_items").fetchone()[0]
    assert [tuple(r) for r in linked] == [(1, 1)]
    assert fav == 0
    assert "group_name" in _columns(db, "videos")
    assert "group_name" in _columns(db, "index_queue")


def test_migrates_v7_database(tmp_path):
    db = tmp_path / "v7.db"
    with sqlite3.connect(str(db)) as conn:
        conn.executescript(schema.V4_SCHEMA)
        conn.executescript(schema.V5_SCHEMA)
        conn.executescript(schema.V7_SCHEMA)
        conn.execute(
            "INSERT INTO videos (filepath, display_name, modified_at, status) "
            "VALUES ('/x/ep.mkv', 'ep', 1.0, 'completed')"
        )
        conn.execute("PRAGMA user_version = 7")

    schema.init_db(db)

    assert schema.current_version(db) == schema.SCHEMA_VERSION
    assert "derushed" in _columns(db, "videos")


def test_add_column_tolerates_a_concurrent_migration(tmp_path, monkeypatch):
    import sqlite3

    from backend.db import schema as sch

    conn = sqlite3.connect(tmp_path / "race.db")
    conn.execute("CREATE TABLE t (a INTEGER)")
    real_execute = conn.execute

    class Racing:
        """PRAGMA sees the old columns, but another process added it since."""
        def execute(self, sql, *args):
            if sql.startswith("PRAGMA"):
                return real_execute(sql, *args)
            real_execute("ALTER TABLE t ADD COLUMN b TEXT") if "b" not in {
                r[1] for r in real_execute("PRAGMA table_info(t)")} else None
            return real_execute(sql, *args)

    sch._add_column_if_missing(Racing(), "t", "b", "TEXT")
    assert {r[1] for r in conn.execute("PRAGMA table_info(t)")} == {"a", "b"}
