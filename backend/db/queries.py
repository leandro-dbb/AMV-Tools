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
                 model_version: str, status: str = "indexing",
                 group_name: str | None = None) -> int:
    mtime = os.path.getmtime(filepath) if os.path.exists(filepath) else 0
    conn.execute(
        """
        INSERT INTO videos (filepath, display_name, duration_ms, fps, resolution,
                            modified_at, model_version, status, group_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(filepath) DO UPDATE SET
            display_name=excluded.display_name,
            duration_ms=excluded.duration_ms,
            fps=excluded.fps,
            resolution=excluded.resolution,
            modified_at=excluded.modified_at,
            model_version=excluded.model_version,
            status=excluded.status,
            group_name=COALESCE(excluded.group_name, videos.group_name)
        """,
        (filepath, display_name, duration_ms, fps, resolution, mtime, model_version,
         status, group_name),
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
                  (SELECT COUNT(*) FROM scenes s WHERE s.video_id = v.id) AS scene_count,
                  v.group_name, v.derushed
             FROM videos v
            ORDER BY v.group_name IS NULL, v.group_name, v.display_name"""
    ).fetchall()
    return [
        dict(id=r[0], filepath=r[1], display_name=r[2], duration_ms=r[3] or 0,
             fps=r[4] or 0.0, resolution=r[5] or "", status=r[6], scene_count=r[7],
             group_name=r[8], derushed=bool(r[9]))
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
                            threshold: float, sort: str, top_only: bool = False) -> list[dict]:
    """``top_only`` drops sub-segments (parent_scene_id set) — the Derush
    player wants the non-overlapping top-level cut segments."""
    top_clause = " AND s.parent_scene_id IS NULL" if top_only else ""
    order_clause = "s.start_ms" if sort != "confidence" else "COALESCE(t.confidence, 0) DESC, s.start_ms"
    if tag:
        rows = conn.execute(
            f"""SELECT s.id, s.scene_index, s.start_ms, s.end_ms, v.display_name, t.confidence
                  FROM scenes s
                  JOIN videos v ON v.id = s.video_id
                  JOIN scene_tags t ON t.scene_id = s.id AND t.tag = ?
                 WHERE s.video_id = ? AND t.confidence >= ?{top_clause}
                 ORDER BY {order_clause}""",
            (tag, video_id, threshold),
        ).fetchall()
        return [
            dict(id=r[0], scene_index=r[1], start_ms=r[2], end_ms=r[3],
                 video_display=r[4], video_id=video_id, confidence=r[5], score=r[5])
            for r in rows
        ]
    rows = conn.execute(
        f"""SELECT s.id, s.scene_index, s.start_ms, s.end_ms, v.display_name
             FROM scenes s
             JOIN videos v ON v.id = s.video_id
            WHERE s.video_id = ?{top_clause}
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


def add_to_queue(conn: sqlite3.Connection, path: str, is_directory: bool, recursive: bool,
                 group_name: str | None = None) -> bool:
    try:
        conn.execute(
            """INSERT INTO index_queue (path, is_directory, recursive, group_name)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(path) DO UPDATE SET group_name = excluded.group_name""",
            (str(path), int(is_directory), int(recursive), group_name),
        )
        return True
    except sqlite3.Error:
        return False


def get_queue(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, path, is_directory, recursive, group_name FROM index_queue ORDER BY id"
    ).fetchall()
    return [dict(id=r[0], path=r[1], is_directory=bool(r[2]), recursive=bool(r[3]),
                 group_name=r[4]) for r in rows]


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


def merge_scenes(conn: sqlite3.Connection, scene_ids: list[int],
                 gap_tolerance_ms: int = 300) -> dict:
    """Merge 2+ time-adjacent top-level scenes of the same video into one.

    The earliest scene survives and absorbs the full time range; the others
    are deleted after their tags are copied over, their sub-segments are
    reparented, and any derush keep is transferred. Raises ValueError with a
    user-facing message when the selection isn't mergeable.
    """
    if len(scene_ids) < 2:
        raise ValueError("select at least two scenes to merge")
    placeholders = ",".join("?" * len(scene_ids))
    rows = conn.execute(
        f"""SELECT id, video_id, start_ms, end_ms, parent_scene_id
              FROM scenes WHERE id IN ({placeholders})""",
        scene_ids,
    ).fetchall()
    if len(rows) != len(set(scene_ids)):
        raise ValueError("some scenes no longer exist")
    if len({r[1] for r in rows}) != 1:
        raise ValueError("scenes belong to different videos")
    if any(r[4] is not None for r in rows):
        raise ValueError("sub-segments can't be merged — select top-level scenes")

    ordered = sorted(rows, key=lambda r: r[2])
    for prev, nxt in zip(ordered, ordered[1:]):
        if nxt[2] - prev[3] > gap_tolerance_ms:
            raise ValueError("scenes are not adjacent on the timeline")

    keep_id = ordered[0][0]
    new_start = ordered[0][2]
    new_end = max(r[3] for r in ordered)
    losers = [r[0] for r in ordered[1:]]

    conn.execute("UPDATE scenes SET end_ms = ? WHERE id = ?", (new_end, keep_id))
    for loser in losers:
        conn.execute(
            """INSERT OR IGNORE INTO scene_tags (scene_id, tag, category, confidence)
               SELECT ?, tag, category, confidence FROM scene_tags WHERE scene_id = ?""",
            (keep_id, loser),
        )
        conn.execute("UPDATE scenes SET parent_scene_id = ? WHERE parent_scene_id = ?",
                     (keep_id, loser))
        keep_item = conn.execute(
            "SELECT id FROM derush_items WHERE scene_id = ?", (keep_id,)
        ).fetchone()
        loser_item = conn.execute(
            "SELECT id, favorite FROM derush_items WHERE scene_id = ?", (loser,)
        ).fetchone()
        if loser_item is not None:
            if keep_item is None:
                conn.execute("UPDATE derush_items SET scene_id = ? WHERE id = ?",
                             (keep_id, loser_item[0]))
            else:
                # Both kept: merge folder memberships + favorite flag into the
                # survivor, then let the loser's item cascade away.
                conn.execute(
                    """INSERT OR IGNORE INTO derush_item_folders (item_id, folder_id)
                       SELECT ?, folder_id FROM derush_item_folders WHERE item_id = ?""",
                    (keep_item[0], loser_item[0]),
                )
                if loser_item[1]:
                    conn.execute("UPDATE derush_items SET favorite = 1 WHERE id = ?",
                                 (keep_item[0],))
        conn.execute("DELETE FROM scenes WHERE id = ?", (loser,))
    return {"kept_scene_id": keep_id, "start_ms": new_start, "end_ms": new_end,
            "removed": losers}


# ── derush (selects) ────────────────────────────────────────────────────────
def derush_list(conn: sqlite3.Connection) -> tuple[list[dict], list[dict]]:
    """Return (folders, items). Items carry the scene/video info the library
    grid needs, plus ``folder_ids`` — an item can live in several folders."""
    folders = [
        dict(id=r[0], name=r[1], item_count=r[2])
        for r in conn.execute(
            """SELECT f.id, f.name,
                      (SELECT COUNT(*) FROM derush_item_folders l WHERE l.folder_id = f.id)
                 FROM derush_folders f ORDER BY f.name"""
        ).fetchall()
    ]
    memberships: dict[int, list[int]] = {}
    for item_id, folder_id in conn.execute(
        "SELECT item_id, folder_id FROM derush_item_folders ORDER BY folder_id"
    ).fetchall():
        memberships.setdefault(item_id, []).append(folder_id)
    items = [
        dict(id=r[0], scene_id=r[1], folder_ids=memberships.get(r[0], []),
             custom_name=r[2], added_at=r[3],
             video_id=r[4], video_display=r[5], scene_index=r[6],
             start_ms=r[7], end_ms=r[8], has_proxy=bool(r[9]), favorite=bool(r[10]))
        for r in conn.execute(
            """SELECT i.id, i.scene_id, i.custom_name, i.added_at,
                      s.video_id, v.display_name, s.scene_index, s.start_ms, s.end_ms,
                      s.proxy_path, i.favorite
                 FROM derush_items i
                 JOIN scenes s ON s.id = i.scene_id
                 JOIN videos v ON v.id = s.video_id
                ORDER BY i.added_at DESC, i.id DESC"""
        ).fetchall()
    ]
    return folders, items


def derush_set_item_folders(conn: sqlite3.Connection, item_id: int,
                            folder_ids: list[int]) -> bool:
    """Replace an item's folder memberships (an item can be in several)."""
    row = conn.execute("SELECT 1 FROM derush_items WHERE id = ?", (item_id,)).fetchone()
    if row is None:
        return False
    conn.execute("DELETE FROM derush_item_folders WHERE item_id = ?", (item_id,))
    if folder_ids:
        placeholders = ",".join("?" * len(folder_ids))
        conn.execute(
            f"""INSERT OR IGNORE INTO derush_item_folders (item_id, folder_id)
                SELECT ?, id FROM derush_folders WHERE id IN ({placeholders})""",
            (item_id, *folder_ids),
        )
    return True


def _derush_link_folder(conn: sqlite3.Connection, item_id: int,
                        folder_id: int | None) -> None:
    if folder_id is not None:
        conn.execute(
            """INSERT OR IGNORE INTO derush_item_folders (item_id, folder_id)
               SELECT ?, id FROM derush_folders WHERE id = ?""",
            (item_id, folder_id),
        )


def derush_toggle(conn: sqlite3.Connection, scene_id: int,
                  folder_id: int | None = None,
                  favorite: bool = False) -> tuple[bool, int | None, bool]:
    """Keep/unkeep a scene. Returns (kept_now, item_id, is_favorite).

    Plain toggle (favorite=False): add if absent, remove if present.
    Favorite toggle (favorite=True): absent → add AS favorite; present but
    normal → upgrade to favorite; already favorite → remove entirely (the
    same rhythm as the plain toggle, one level up)."""
    row = conn.execute(
        "SELECT id, favorite FROM derush_items WHERE scene_id = ?", (scene_id,)
    ).fetchone()
    if row is None:
        cur = conn.execute(
            "INSERT INTO derush_items (scene_id, favorite) VALUES (?, ?)",
            (scene_id, int(favorite)),
        )
        _derush_link_folder(conn, cur.lastrowid, folder_id)
        return True, cur.lastrowid, favorite
    item_id, is_fav = row[0], bool(row[1])
    if favorite and not is_fav:
        conn.execute("UPDATE derush_items SET favorite = 1 WHERE id = ?", (item_id,))
        return True, item_id, True
    conn.execute("DELETE FROM derush_items WHERE id = ?", (item_id,))
    return False, None, False


def derush_set_level(conn: sqlite3.Connection, scene_id: int, delta: int,
                     folder_id: int | None = None) -> tuple[int, int | None]:
    """Walk the keep ladder: 0 = not kept, 1 = kept, 2 = favorite.

    ``delta`` of +1/-1 moves one rung (clamped). Returns (level, item_id)."""
    row = conn.execute(
        "SELECT id, favorite FROM derush_items WHERE scene_id = ?", (scene_id,)
    ).fetchone()
    current = 0 if row is None else (2 if row[1] else 1)
    new = max(0, min(2, current + (1 if delta > 0 else -1)))
    if new == current:
        return current, (row[0] if row else None)
    if new == 0:
        conn.execute("DELETE FROM derush_items WHERE id = ?", (row[0],))
        return 0, None
    if row is None:
        cur = conn.execute(
            "INSERT INTO derush_items (scene_id, favorite) VALUES (?, ?)",
            (scene_id, int(new == 2)),
        )
        _derush_link_folder(conn, cur.lastrowid, folder_id)
        return new, cur.lastrowid
    conn.execute("UPDATE derush_items SET favorite = ? WHERE id = ?", (int(new == 2), row[0]))
    return new, row[0]


def derush_kept_scene_ids(conn: sqlite3.Connection, video_id: int | None = None) -> set[int]:
    if video_id is None:
        rows = conn.execute("SELECT scene_id FROM derush_items").fetchall()
    else:
        rows = conn.execute(
            """SELECT i.scene_id FROM derush_items i
                 JOIN scenes s ON s.id = i.scene_id WHERE s.video_id = ?""",
            (video_id,),
        ).fetchall()
    return {r[0] for r in rows}


def derush_update_item(conn: sqlite3.Connection, item_id: int,
                       custom_name: str | None = ...,
                       favorite: bool = ...) -> bool:
    """Partial update; pass Ellipsis (default) to leave a field untouched.
    Folder memberships are handled by :func:`derush_set_item_folders`."""
    sets, params = [], []
    if custom_name is not ...:
        sets.append("custom_name = ?")
        params.append(custom_name)
    if favorite is not ...:
        sets.append("favorite = ?")
        params.append(int(favorite))
    if not sets:
        return True
    params.append(item_id)
    cur = conn.execute(f"UPDATE derush_items SET {', '.join(sets)} WHERE id = ?", params)
    return cur.rowcount > 0


def derush_items_for_export(conn: sqlite3.Connection,
                            folder_id: int | None | str = "all") -> list[dict]:
    """Items + everything export needs.
    folder_id: 'all', 'fav' (favorites only), None (root) or a folder id.

    ``folder_name`` drives the export subdirectory. Items can belong to
    several folders: exporting a specific folder files them under THAT
    folder's name; broader scopes use the first folder alphabetically."""
    first_folder = """(SELECT f.name FROM derush_item_folders l
                         JOIN derush_folders f ON f.id = l.folder_id
                        WHERE l.item_id = i.id ORDER BY f.name LIMIT 1)"""
    if folder_id == "all":
        folder_expr, where, params = first_folder, "", ()
    elif folder_id == "fav":
        folder_expr, where, params = first_folder, "WHERE i.favorite = 1", ()
    elif folder_id is None:
        folder_expr = "NULL"
        where = "WHERE NOT EXISTS (SELECT 1 FROM derush_item_folders l WHERE l.item_id = i.id)"
        params = ()
    else:
        folder_expr = "(SELECT name FROM derush_folders WHERE id = ?)"
        where = "WHERE EXISTS (SELECT 1 FROM derush_item_folders l WHERE l.item_id = i.id AND l.folder_id = ?)"
        params = (folder_id, folder_id)
    rows = conn.execute(
        f"""SELECT i.id, i.scene_id, i.custom_name, s.start_ms, s.end_ms,
                   v.filepath, v.display_name, s.scene_index,
                   {folder_expr}
              FROM derush_items i
              JOIN scenes s ON s.id = i.scene_id
              JOIN videos v ON v.id = s.video_id
              {where}
             ORDER BY v.display_name, s.start_ms""",
        params,
    ).fetchall()
    return [
        dict(item_id=r[0], scene_id=r[1], custom_name=r[2], start_ms=r[3], end_ms=r[4],
             filepath=r[5], display_name=r[6], scene_index=r[7], folder_name=r[8])
        for r in rows
    ]


def db_stats(db_path: str) -> dict:
    if not Path(db_path).exists():
        return dict(scenes=0, videos=0, size_kb=0)
    with get_conn(db_path) as conn:
        scenes = conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]
        videos = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    return dict(scenes=scenes, videos=videos, size_kb=int(Path(db_path).stat().st_size / 1024))
