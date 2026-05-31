"""Query helpers — keep raw SQL out of API handlers."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np

from .schema import get_conn


def upsert_video(conn: sqlite3.Connection, filepath: str, display_name: str,
                 duration_ms: int, fps: float, resolution: str,
                 model_version: str, status: str = "indexing") -> int:
    mtime = os.path.getmtime(filepath) if os.path.exists(filepath) else 0
    conn.execute(
        """
        INSERT INTO videos (filepath, display_name, duration_ms, fps, resolution,
                            modified_at, model_version, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(filepath) DO UPDATE SET
            display_name=excluded.display_name,
            duration_ms=excluded.duration_ms,
            fps=excluded.fps,
            resolution=excluded.resolution,
            modified_at=excluded.modified_at,
            model_version=excluded.model_version,
            status=excluded.status
        """,
        (filepath, display_name, duration_ms, fps, resolution, mtime, model_version, status),
    )
    return conn.execute("SELECT id FROM videos WHERE filepath = ?", (filepath,)).fetchone()[0]


def set_video_status(conn: sqlite3.Connection, video_id: int, status: str) -> None:
    conn.execute("UPDATE videos SET status = ? WHERE id = ?", (status, video_id))


def insert_scene(conn: sqlite3.Connection, video_id: int, scene_index: int,
                 start_ms: int, end_ms: int, embedding: np.ndarray | None,
                 thumbnail: bytes | None, parent_scene_id: int | None = None) -> int:
    cur = conn.execute(
        """INSERT INTO scenes (video_id, scene_index, start_ms, end_ms,
                                parent_scene_id, embedding, thumbnail)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (video_id, scene_index, start_ms, end_ms, parent_scene_id,
         embedding.tobytes() if embedding is not None else None, thumbnail),
    )
    return cur.lastrowid


def insert_tags(conn: sqlite3.Connection, scene_id: int,
                tags: Iterable[tuple[str, str, float]]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO scene_tags (scene_id, tag, category, confidence) VALUES (?, ?, ?, ?)",
        ((scene_id, tag, category, conf) for tag, category, conf in tags),
    )


def update_scene_embedding(conn: sqlite3.Connection, scene_id: int, embedding: np.ndarray) -> None:
    conn.execute(
        "UPDATE scenes SET embedding = ? WHERE id = ?",
        (embedding.tobytes(), scene_id),
    )


def list_scenes_for_video(conn: sqlite3.Connection, video_id: int) -> list[tuple[int, int, int, int]]:
    """Return (scene_id, scene_index, start_ms, end_ms) for all scenes of a video."""
    rows = conn.execute(
        """SELECT id, scene_index, start_ms, end_ms
             FROM scenes
            WHERE video_id = ?
            ORDER BY scene_index""",
        (video_id,),
    ).fetchall()
    return [tuple(r) for r in rows]


def clear_scene_tags(conn: sqlite3.Connection, scene_id: int) -> None:
    conn.execute("DELETE FROM scene_tags WHERE scene_id = ?", (scene_id,))


def delete_scenes_for_video(conn: sqlite3.Connection, video_id: int) -> None:
    conn.execute("DELETE FROM scenes WHERE video_id = ?", (video_id,))


def get_all_scenes_with_embeddings(conn: sqlite3.Connection) -> list[tuple]:
    rows = conn.execute(
        """SELECT s.id, s.video_id, v.display_name, s.scene_index, s.start_ms, s.end_ms, s.embedding
             FROM scenes s
             JOIN videos v ON v.id = s.video_id
            WHERE v.status = 'completed' AND s.embedding IS NOT NULL"""
    ).fetchall()
    return rows


def get_scene_tags(conn: sqlite3.Connection, scene_ids: list[int]) -> dict[int, list[str]]:
    if not scene_ids:
        return {}
    placeholders = ",".join("?" * len(scene_ids))
    rows = conn.execute(
        f"SELECT scene_id, tag FROM scene_tags WHERE scene_id IN ({placeholders}) ORDER BY confidence DESC",
        scene_ids,
    ).fetchall()
    result: dict[int, list[str]] = {}
    for sid, tag in rows:
        result.setdefault(sid, []).append(tag)
    return result


def list_videos(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """SELECT v.id, v.filepath, v.display_name, v.duration_ms, v.fps, v.resolution, v.status,
                  (SELECT COUNT(*) FROM scenes s WHERE s.video_id = v.id) AS scene_count
             FROM videos v
            ORDER BY v.display_name"""
    ).fetchall()
    return [
        dict(id=r[0], filepath=r[1], display_name=r[2], duration_ms=r[3] or 0,
             fps=r[4] or 0.0, resolution=r[5] or "", status=r[6], scene_count=r[7])
        for r in rows
    ]


def video_tags_with_counts(conn: sqlite3.Connection, video_id: int) -> list[dict]:
    rows = conn.execute(
        """SELECT t.tag, COALESCE(MIN(t.category), '') AS cat, COUNT(*) AS c
             FROM scene_tags t
             JOIN scenes s ON s.id = t.scene_id
            WHERE s.video_id = ?
            GROUP BY t.tag
            ORDER BY c DESC, t.tag""",
        (video_id,),
    ).fetchall()
    return [dict(tag=r[0], category=r[1], count=r[2]) for r in rows]


def scenes_by_video_and_tag(conn: sqlite3.Connection, video_id: int, tag: str | None,
                            threshold: float, sort: str) -> list[dict]:
    order_clause = "s.start_ms" if sort != "confidence" else "COALESCE(t.confidence, 0) DESC, s.start_ms"
    if tag:
        rows = conn.execute(
            f"""SELECT s.id, s.scene_index, s.start_ms, s.end_ms, v.display_name, t.confidence
                  FROM scenes s
                  JOIN videos v ON v.id = s.video_id
                  JOIN scene_tags t ON t.scene_id = s.id AND t.tag = ?
                 WHERE s.video_id = ? AND t.confidence >= ?
                 ORDER BY {order_clause}""",
            (tag, video_id, threshold),
        ).fetchall()
        return [
            dict(id=r[0], scene_index=r[1], start_ms=r[2], end_ms=r[3],
                 video_display=r[4], video_id=video_id, confidence=r[5], score=r[5])
            for r in rows
        ]
    rows = conn.execute(
        """SELECT s.id, s.scene_index, s.start_ms, s.end_ms, v.display_name
             FROM scenes s
             JOIN videos v ON v.id = s.video_id
            WHERE s.video_id = ?
            ORDER BY s.start_ms""",
        (video_id,),
    ).fetchall()
    return [
        dict(id=r[0], scene_index=r[1], start_ms=r[2], end_ms=r[3],
             video_display=r[4], video_id=video_id, confidence=None, score=0.0)
        for r in rows
    ]


def get_scene(conn: sqlite3.Connection, scene_id: int) -> dict | None:
    r = conn.execute(
        """SELECT s.id, s.video_id, v.filepath, v.display_name, s.scene_index,
                  s.start_ms, s.end_ms, s.thumbnail, s.proxy_path
             FROM scenes s JOIN videos v ON v.id = s.video_id
            WHERE s.id = ?""",
        (scene_id,),
    ).fetchone()
    if not r:
        return None
    return dict(id=r[0], video_id=r[1], filepath=r[2], display_name=r[3],
                scene_index=r[4], start_ms=r[5], end_ms=r[6],
                thumbnail=r[7], proxy_path=r[8])


def set_scene_proxy(conn: sqlite3.Connection, scene_id: int, proxy_path: str) -> None:
    conn.execute("UPDATE scenes SET proxy_path = ? WHERE id = ?", (proxy_path, scene_id))


def add_to_queue(conn: sqlite3.Connection, path: str, is_directory: bool, recursive: bool) -> bool:
    try:
        conn.execute(
            "INSERT OR IGNORE INTO index_queue (path, is_directory, recursive) VALUES (?, ?, ?)",
            (str(path), int(is_directory), int(recursive)),
        )
        return True
    except sqlite3.Error:
        return False


def get_queue(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, path, is_directory, recursive FROM index_queue ORDER BY id"
    ).fetchall()
    return [dict(id=r[0], path=r[1], is_directory=bool(r[2]), recursive=bool(r[3])) for r in rows]


def clear_queue(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM index_queue")


def remove_from_queue(conn: sqlite3.Connection, item_id: int) -> None:
    conn.execute("DELETE FROM index_queue WHERE id = ?", (item_id,))


def record_search(conn: sqlite3.Connection, query: str) -> None:
    conn.execute(
        """INSERT INTO search_history (query, last_used, use_count) VALUES (?, CURRENT_TIMESTAMP, 1)
           ON CONFLICT(query) DO UPDATE SET last_used = CURRENT_TIMESTAMP, use_count = use_count + 1""",
        (query,),
    )


def recent_searches(conn: sqlite3.Connection, limit: int = 5) -> list[str]:
    rows = conn.execute(
        "SELECT query FROM search_history ORDER BY last_used DESC LIMIT ?", (limit,)
    ).fetchall()
    return [r[0] for r in rows]


def all_video_paths(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    return conn.execute("SELECT id, filepath FROM videos").fetchall()


def update_video_filepath(conn: sqlite3.Connection, video_id: int, new_filepath: str) -> bool:
    try:
        conn.execute("UPDATE videos SET filepath = ? WHERE id = ?", (new_filepath, video_id))
        return True
    except sqlite3.IntegrityError:
        return False


def delete_video(conn: sqlite3.Connection, video_id: int) -> None:
    conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))


def cleanup_orphans(db_path: str) -> int:
    """Delete video rows whose filepath no longer exists. Returns count deleted."""
    removed = 0
    with get_conn(db_path) as conn:
        rows = conn.execute("SELECT id, filepath FROM videos").fetchall()
        missing_ids = [vid for vid, fp in rows if not os.path.exists(fp)]
        if missing_ids:
            placeholders = ",".join("?" * len(missing_ids))
            conn.execute(f"DELETE FROM videos WHERE id IN ({placeholders})", missing_ids)
            removed = len(missing_ids)
    return removed


def db_stats(db_path: str) -> dict:
    if not Path(db_path).exists():
        return dict(scenes=0, videos=0, size_kb=0)
    with get_conn(db_path) as conn:
        scenes = conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]
        videos = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    return dict(scenes=scenes, videos=videos, size_kb=int(Path(db_path).stat().st_size / 1024))
