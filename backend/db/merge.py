"""Merge multiple AMV Tools (v4+) databases into a target one.

Copies videos + scenes + scene_tags. Deduplicates on filepath (skips videos that
already exist in the target). Useful when consolidating libraries built on
different machines.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, List, Optional

from .schema import get_conn, init_db


def merge_databases(target_db: str, sources: List[str],
                    progress_cb: Optional[Callable[[str], None]] = None) -> dict:
    init_db(target_db)
    target_abs = str(Path(target_db).resolve())
    safe = [s for s in sources if str(Path(s).resolve()) != target_abs]

    stats = {"videos_added": 0, "scenes_added": 0, "tags_added": 0, "skipped_videos": 0}

    with get_conn(target_abs) as tgt:
        existing = {fp for (fp,) in tgt.execute("SELECT filepath FROM videos").fetchall()}

        for i, src in enumerate(safe):
            if progress_cb:
                progress_cb(f"Merging {i+1}/{len(safe)}: {Path(src).name}")
            try:
                init_db(src)
                src_uri = f"file:{Path(src).as_posix()}?mode=ro"
                with sqlite3.connect(src_uri, uri=True, timeout=10.0) as src_conn:
                    sc = src_conn.cursor()
                    sc.execute("""SELECT id, filepath, display_name, duration_ms, fps, resolution,
                                          modified_at, model_version, status FROM videos""")
                    for vrow in sc.fetchall():
                        old_id, filepath = vrow[0], vrow[1]
                        if filepath in existing:
                            stats["skipped_videos"] += 1
                            continue
                        tgt.execute(
                            """INSERT INTO videos (filepath, display_name, duration_ms, fps, resolution,
                                                    modified_at, model_version, status)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                            vrow[1:],
                        )
                        new_video_id = tgt.execute("SELECT last_insert_rowid()").fetchone()[0]
                        existing.add(filepath)
                        stats["videos_added"] += 1

                        # Scenes: re-create with new parent_scene_id mapping
                        scene_map: dict[int, int] = {}
                        scenes_rows = src_conn.execute(
                            """SELECT id, scene_index, start_ms, end_ms, parent_scene_id,
                                      embedding, thumbnail, proxy_path
                                 FROM scenes WHERE video_id = ?
                                ORDER BY scene_index""",
                            (old_id,),
                        ).fetchall()
                        for srow in scenes_rows:
                            old_sid, scene_idx, sm, em, parent_old, emb, thumb, prox = srow
                            parent_new = scene_map.get(parent_old) if parent_old else None
                            tgt.execute(
                                """INSERT INTO scenes (video_id, scene_index, start_ms, end_ms,
                                                        parent_scene_id, embedding, thumbnail, proxy_path)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                                (new_video_id, scene_idx, sm, em, parent_new, emb, thumb, prox),
                            )
                            new_sid = tgt.execute("SELECT last_insert_rowid()").fetchone()[0]
                            scene_map[old_sid] = new_sid
                            stats["scenes_added"] += 1

                            for tag, cat, conf in src_conn.execute(
                                "SELECT tag, category, confidence FROM scene_tags WHERE scene_id = ?",
                                (old_sid,),
                            ):
                                tgt.execute(
                                    """INSERT OR IGNORE INTO scene_tags
                                       (scene_id, tag, category, confidence) VALUES (?, ?, ?, ?)""",
                                    (new_sid, tag, cat, conf),
                                )
                                stats["tags_added"] += 1
            except Exception as e:
                if progress_cb:
                    progress_cb(f"Error merging {src}: {e}")
                raise
    return stats
