"""Load the user's indexed shots and query scores for the music planner."""
from __future__ import annotations

import re
import sqlite3
from typing import Callable, Optional

import numpy as np

from ..searchplus import filters as sp_filters

_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)
_STOP = {"vs", "and", "or", "the", "a", "an", "of", "in", "on", "et", "le", "la", "les",
         "un", "une", "de", "des", "du", "avec", "with"}


def load_shots(conn: sqlite3.Connection) -> list[dict]:
    """Every usable shot: leaf scenes (a sub-segmented scene is represented
    by its sub-shots) of completed videos, with their Search+ metrics."""
    rows = conn.execute(
        """SELECT s.id, s.video_id, v.display_name, v.group_name, s.start_ms, s.end_ms,
                  m.intensity, m.activity, m.motion_type, m.motion_magnitude,
                  m.flash_count, m.cut_rate, v.filepath
             FROM scenes s
             JOIN videos v ON v.id = s.video_id
             LEFT JOIN scene_metrics m ON m.scene_id = s.id
            WHERE v.status = 'completed'
              AND s.end_ms > s.start_ms
              AND NOT EXISTS (SELECT 1 FROM scenes c WHERE c.parent_scene_id = s.id)
            ORDER BY s.video_id, s.start_ms"""
    ).fetchall()
    return [
        {
            "scene_id": int(r[0]), "video_id": int(r[1]), "video_name": r[2] or "",
            "group": r[3], "start_ms": int(r[4]), "end_ms": int(r[5]),
            "intensity": r[6], "activity": r[7], "motion_type": r[8],
            "motion_magnitude": r[9], "flash_count": r[10], "cut_rate": r[11],
            "filepath": r[12],
        }
        for r in rows
    ]


def split_shots(shots: list[dict], cuts_by_path: dict[str, list[int]], min_ms: int = 250) -> list[dict]:
    """Split shots at hard cuts the indexer missed (found by a preview
    render), so a proposed window never straddles a change of shot."""
    if not cuts_by_path:
        return shots
    out = []
    for s in shots:
        cuts = [c for c in cuts_by_path.get(s.get("filepath") or "", [])
                if s["start_ms"] + min_ms < c < s["end_ms"] - min_ms]
        if not cuts:
            out.append(s)
            continue
        edges = [s["start_ms"], *cuts, s["end_ms"]]
        for a, b in zip(edges, edges[1:]):
            if b - a >= min_ms:
                out.append({**s, "start_ms": a, "end_ms": b})
    return out


def character_scene_ids(conn: sqlite3.Connection, character_ids: list[int]) -> set[int]:
    if not character_ids:
        return set()
    f = sp_filters.SearchFilters(characters=[int(c) for c in character_ids])
    return set(int(x) for x in sp_filters.matching_scene_ids(conn, f))


def _tag_scores(conn: sqlite3.Connection, query: str) -> dict[int, float]:
    tokens = {t.lower() for t in _TOKEN_RE.findall(query)} - _STOP
    if not tokens:
        return {}
    ph = ",".join("?" * len(tokens))
    rows = conn.execute(
        f"SELECT scene_id, COUNT(DISTINCT LOWER(tag)) FROM scene_tags "
        f"WHERE LOWER(tag) IN ({ph}) GROUP BY scene_id", list(tokens)).fetchall()
    out = {int(sid): min(1.0, n / len(tokens)) for sid, n in rows}
    # Characters double as search tags ("Goku" on the drop).
    hits = [cid for cid, name in conn.execute("SELECT id, name FROM characters").fetchall()
            if name and name.lower().strip() in tokens | {query.lower().strip()}]
    if hits:
        for (sid,) in conn.execute(
                f"""SELECT cm.scene_id FROM character_matches cm
                      JOIN characters c ON c.id = cm.character_id
                     WHERE cm.character_id IN ({','.join('?' * len(hits))})
                       AND cm.score >= c.threshold""", hits).fetchall():
            out[int(sid)] = 1.0
    return out


def query_scores(conn: sqlite3.Connection, index: Optional[dict], query: str,
                 embed_text: Optional[Callable[[str], np.ndarray]]) -> dict[int, float]:
    """0-1 relevance of every shot for a text wish: SigLIP similarity
    (normalised over the library) blended with tag / character matches."""
    query = (query or "").strip()
    if not query:
        return {}
    scores = _tag_scores(conn, query)
    if embed_text is not None and index is not None and len(index.get("sids", [])):
        vec = np.asarray(embed_text(query), dtype=np.float32)
        sims = index["embeddings"] @ vec
        lo, hi = np.percentile(sims, [50, 99.5])
        rel = np.clip((sims - lo) / max(1e-6, hi - lo), 0.0, 1.0)
        for sid, r in zip(index["sids"], rel):
            sid = int(sid)
            scores[sid] = float(max(scores.get(sid, 0.0) * 0.6 + 0.6 * r, r))
    return scores
