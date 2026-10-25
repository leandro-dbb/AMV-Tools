"""SQL helpers for the v9 search+ tables (scene_metrics)."""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Iterable, Optional

from .analysis import ANALYSIS_VERSION, SceneMetrics

_METRIC_COLS = (
    "motion_type", "motion_magnitude", "motion_dx", "motion_dy", "motion_zoom",
    "activity", "intensity", "cut_rate", "flash_count", "brightness",
    "dominant_color", "accent_color", "palette",
)


def upsert_metrics(conn: sqlite3.Connection,
                   rows: Iterable[tuple[int, int, int, SceneMetrics]]) -> int:
    """rows: (scene_id, start_ms, end_ms, metrics). Returns rows written."""
    now = time.time()
    payload = []
    for sid, s_ms, e_ms, m in rows:
        d = m.as_row()
        payload.append((
            sid, ANALYSIS_VERSION, s_ms, e_ms,
            *(json.dumps(d[c]) if c == "palette" else d[c] for c in _METRIC_COLS),
            now,
        ))
    if not payload:
        return 0
    cols = ("scene_id", "version", "start_ms", "end_ms", *_METRIC_COLS, "computed_at")
    conn.executemany(
        f"INSERT OR REPLACE INTO scene_metrics ({', '.join(cols)}) "
        f"VALUES ({', '.join('?' * len(cols))})",
        payload,
    )
    return len(payload)


# A scene needs (re)analysis when it has no row, an older estimator version,
# or its range changed since (scene merge extends end_ms).
_PENDING_WHERE = """
    (m.scene_id IS NULL OR m.version < ? OR m.start_ms != s.start_ms OR m.end_ms != s.end_ms)
"""


def pending_videos(conn: sqlite3.Connection) -> list[tuple[int, str, str, int]]:
    """(video_id, filepath, display_name, pending_scene_count) for videos with
    scenes lacking up-to-date metrics, most pending first."""
    rows = conn.execute(
        f"""SELECT v.id, v.filepath, v.display_name, COUNT(*)
              FROM scenes s
              JOIN videos v ON v.id = s.video_id
              LEFT JOIN scene_metrics m ON m.scene_id = s.id
             WHERE v.status NOT IN ('indexing', 'pending') AND {_PENDING_WHERE}
             GROUP BY v.id
             ORDER BY v.id""",
        (ANALYSIS_VERSION,),
    ).fetchall()
    return [(r[0], r[1], r[2], r[3]) for r in rows]


def pending_scenes_for_video(conn: sqlite3.Connection, video_id: int) -> list[tuple[int, int, int, Optional[bytes]]]:
    rows = conn.execute(
        f"""SELECT s.id, s.start_ms, s.end_ms, s.thumbnail
              FROM scenes s
              LEFT JOIN scene_metrics m ON m.scene_id = s.id
             WHERE s.video_id = ? AND {_PENDING_WHERE}
             ORDER BY s.start_ms""",
        (video_id, ANALYSIS_VERSION),
    ).fetchall()
    return [(r[0], r[1], r[2], r[3]) for r in rows]


def coverage(conn: sqlite3.Connection) -> dict:
    total = conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]
    done = conn.execute(
        """SELECT COUNT(*) FROM scene_metrics m JOIN scenes s ON s.id = m.scene_id
            WHERE m.version >= ? AND m.start_ms = s.start_ms AND m.end_ms = s.end_ms""",
        (ANALYSIS_VERSION,),
    ).fetchone()[0]
    return {"total": int(total), "analyzed": int(done), "pending": int(total - done)}


def metrics_for(conn: sqlite3.Connection, scene_ids: list[int]) -> dict[int, dict]:
    """Compact metrics dict per scene id (only for scenes that have a row)."""
    if not scene_ids:
        return {}
    out: dict[int, dict] = {}
    # Chunk to stay under SQLite's host-parameter limit on old builds.
    for i in range(0, len(scene_ids), 900):
        chunk = scene_ids[i:i + 900]
        rows = conn.execute(
            f"""SELECT scene_id, motion_type, motion_magnitude, intensity, activity,
                       brightness, dominant_color, accent_color, palette
                  FROM scene_metrics WHERE scene_id IN ({','.join('?' * len(chunk))})""",
            chunk,
        ).fetchall()
        for r in rows:
            try:
                pal = json.loads(r[8]) if r[8] else []
            except (TypeError, ValueError):
                pal = []
            out[int(r[0])] = {
                "motion_type": r[1],
                "motion_magnitude": r[2],
                "intensity": r[3],
                "activity": r[4],
                "brightness": r[5],
                "dominant_color": r[6],
                "accent_color": r[7],
                "palette": [p[0] for p in pal][:3],
            }
    return out
