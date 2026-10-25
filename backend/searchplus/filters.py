"""Combined search filters (characters, motion, intensity, duration, colour,
group, favorites) + result enrichment for every search surface.

Filtering is done in SQL against indexed columns and returns a set of scene
ids; the semantic search then masks its in-memory similarity vector with it,
so the SigLIP matmul path stays exactly as fast as before when no filter is
set (one cheap ``is_empty`` check).
"""
from __future__ import annotations

import json
import random
import sqlite3
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator

from ..db import queries
from .analysis import COLOR_BUCKETS, MOTION_TYPES
from .store import metrics_for

# Sentinel group value meaning "videos without an import group".
NO_GROUP = "__none__"
# Score added to a character's scenes when the query names the character.
CHARACTER_QUERY_BOOST = 0.2


class SearchFilters(BaseModel):
    characters: list[int] = Field(default_factory=list)
    character_mode: Literal["any", "all"] = "any"
    motion: list[str] = Field(default_factory=list)
    motion_min: Optional[float] = None          # camera magnitude 0-1
    intensity_min: Optional[float] = None
    intensity_max: Optional[float] = None
    duration_min_ms: Optional[int] = None
    duration_max_ms: Optional[int] = None
    colors: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)
    video_ids: list[int] = Field(default_factory=list)
    exclude_video_ids: list[int] = Field(default_factory=list)
    favorites: bool = False

    @field_validator("motion")
    @classmethod
    def _known_motion(cls, v: list[str]) -> list[str]:
        return [m for m in v if m in MOTION_TYPES]

    @field_validator("colors")
    @classmethod
    def _known_colors(cls, v: list[str]) -> list[str]:
        return [c for c in v if c in COLOR_BUCKETS]

    def is_empty(self) -> bool:
        return not (
            self.characters or self.motion or self.colors or self.groups
            or self.video_ids or self.exclude_video_ids or self.favorites
            or self.motion_min is not None
            or self.intensity_min is not None or self.intensity_max is not None
            or self.duration_min_ms is not None or self.duration_max_ms is not None
        )

    def needs_metrics(self) -> bool:
        return bool(
            self.motion or self.colors or self.motion_min is not None
            or self.intensity_min is not None or self.intensity_max is not None
        )


def parse_filters(raw: Optional[str]) -> Optional[SearchFilters]:
    """Filters arrive JSON-encoded on GET/multipart endpoints. Raises
    ``ValueError`` (the API answers 422) instead of silently not filtering."""
    if not raw:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("filters must be a JSON object")
    return SearchFilters(**data)


def _where(f: SearchFilters) -> tuple[list[str], list]:
    clauses: list[str] = []
    params: list = []
    motion = [m for m in f.motion if m in MOTION_TYPES]
    if motion:
        clauses.append(f"m.motion_type IN ({','.join('?' * len(motion))})")
        params += motion
    if f.motion_min is not None:
        clauses.append("m.motion_magnitude >= ?")
        params.append(float(f.motion_min))
    if f.intensity_min is not None:
        clauses.append("m.intensity >= ?")
        params.append(float(f.intensity_min))
    if f.intensity_max is not None:
        clauses.append("m.intensity <= ?")
        params.append(float(f.intensity_max))
    if f.duration_min_ms is not None:
        clauses.append("(s.end_ms - s.start_ms) >= ?")
        params.append(int(f.duration_min_ms))
    if f.duration_max_ms is not None:
        clauses.append("(s.end_ms - s.start_ms) <= ?")
        params.append(int(f.duration_max_ms))
    colors = [c for c in f.colors if c in COLOR_BUCKETS]
    if colors:
        ph = ",".join("?" * len(colors))
        clauses.append(f"(m.dominant_color IN ({ph}) OR m.accent_color IN ({ph}))")
        params += colors + colors
    if f.groups:
        named = [g for g in f.groups if g != NO_GROUP]
        parts = []
        if named:
            parts.append(f"v.group_name IN ({','.join('?' * len(named))})")
            params += named
        if NO_GROUP in f.groups:
            parts.append("v.group_name IS NULL")
        clauses.append("(" + " OR ".join(parts) + ")")
    if f.video_ids:
        clauses.append(f"s.video_id IN ({','.join('?' * len(f.video_ids))})")
        params += [int(v) for v in f.video_ids]
    if f.exclude_video_ids:
        clauses.append(f"s.video_id NOT IN ({','.join('?' * len(f.exclude_video_ids))})")
        params += [int(v) for v in f.exclude_video_ids]
    if f.favorites:
        clauses.append("s.id IN (SELECT scene_id FROM derush_items WHERE favorite = 1)")
    if f.characters:
        sub = """s.id IN (SELECT cm.scene_id FROM character_matches cm
                            JOIN characters c ON c.id = cm.character_id
                           WHERE cm.character_id {op} AND cm.score >= c.threshold)"""
        ids = [int(c) for c in f.characters]
        if f.character_mode == "all":
            for cid in ids:
                clauses.append(sub.format(op="= ?"))
                params.append(cid)
        else:
            clauses.append(sub.format(op=f"IN ({','.join('?' * len(ids))})"))
            params += ids
    return clauses, params


def _from(f: SearchFilters) -> str:
    join_m = "JOIN scene_metrics m ON m.scene_id = s.id" if f.needs_metrics() else ""
    return f"FROM scenes s JOIN videos v ON v.id = s.video_id {join_m}"


def matching_scene_ids(conn: sqlite3.Connection, f: SearchFilters) -> np.ndarray:
    clauses, params = _where(f)
    where = " AND ".join(clauses) if clauses else "1"
    rows = conn.execute(f"SELECT s.id {_from(f)} WHERE {where}", params).fetchall()
    return np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))


def filter_mask(conn: sqlite3.Connection, f: Optional[SearchFilters], sids: np.ndarray) -> Optional[np.ndarray]:
    """Boolean mask over the search index rows, or None when unfiltered."""
    if f is None or f.is_empty():
        return None
    ids = matching_scene_ids(conn, f)
    return np.isin(sids, ids, assume_unique=True)


def browse(conn: sqlite3.Connection, f: SearchFilters, *, top_k: int = 200,
           sort: str = "relevance") -> list[dict]:
    """Filter-only search (no text/image query): "every zoom_in of Gojo with
    intensity > 0.7". Score = best character score when filtering on
    characters, else the scene's intensity."""
    clauses, params = _where(f)
    clauses.insert(0, "v.status = 'completed'")
    where = " AND ".join(clauses)
    if f.characters:
        ids = [int(c) for c in f.characters]
        score_expr = (
            "(SELECT MAX(cm.score) FROM character_matches cm WHERE cm.scene_id = s.id "
            f"AND cm.character_id IN ({','.join('?' * len(ids))}))"
        )
        score_params = ids
    else:
        score_expr = "(SELECT intensity FROM scene_metrics mm WHERE mm.scene_id = s.id)"
        score_params = []
    order = {
        "duration": "(s.end_ms - s.start_ms) DESC",
        "video": "v.display_name, s.start_ms",
        "random": "RANDOM()",
    }.get(sort, "score DESC, s.id")
    rows = conn.execute(
        f"""SELECT s.id, s.video_id, v.display_name, s.scene_index, s.start_ms, s.end_ms,
                   COALESCE({score_expr}, 0) AS score
              {_from(f)}
             WHERE {where}
             ORDER BY {order}
             LIMIT ?""",
        [*score_params, *params, int(top_k)],
    ).fetchall()
    results = [
        {"id": r[0], "video_id": r[1], "video_display": r[2], "scene_index": r[3],
         "start_ms": r[4], "end_ms": r[5], "score": float(r[6] or 0.0)}
        for r in rows
    ]
    return finish_results(conn, results)


def rows_to_results(conn: sqlite3.Connection, index: dict, positions, scores) -> list[dict]:
    """Same result shape as /api/search for a list of index positions."""
    rows = index["rows"]
    results = []
    for i in positions:
        sid, vid_id, vid_display, scene_idx, s_ms, e_ms, _ = rows[int(i)]
        results.append({
            "id": sid, "video_id": vid_id, "video_display": vid_display,
            "scene_index": scene_idx, "start_ms": s_ms, "end_ms": e_ms,
            "score": float(scores[int(i)]),
        })
    return finish_results(conn, results)


def finish_results(conn: sqlite3.Connection, results: list[dict]) -> list[dict]:
    """Attach tags + proxy flag (like /api/search) and the search+ fields."""
    sids = [r["id"] for r in results]
    if not sids:
        return results
    tags_for = queries.get_scene_tags(conn, sids)
    proxy_rows = conn.execute(
        f"SELECT id, proxy_path FROM scenes WHERE id IN ({','.join('?' * len(sids))})", sids,
    ).fetchall()
    has_proxy = {sid: bool(p) for sid, p in proxy_rows}
    for r in results:
        r.setdefault("tags", tags_for.get(r["id"], [])[:8])
        r.setdefault("proxy_path", True if has_proxy.get(r["id"]) else None)
    return enrich(conn, results)


def enrich(conn: sqlite3.Connection, results: list[dict]) -> list[dict]:
    """Add ``metrics`` and ``characters`` to result dicts (in place)."""
    sids = [int(r["id"]) for r in results]
    if not sids:
        return results
    metrics = metrics_for(conn, sids)
    chars: dict[int, list[dict]] = {}
    ph = ",".join("?" * len(sids))
    for sid, cid, name, score in conn.execute(
        f"""SELECT cm.scene_id, c.id, c.name, cm.score
              FROM character_matches cm JOIN characters c ON c.id = cm.character_id
             WHERE cm.scene_id IN ({ph}) AND cm.score >= c.threshold
             ORDER BY cm.score DESC""",
        sids,
    ).fetchall():
        lst = chars.setdefault(int(sid), [])
        if len(lst) < 4:
            lst.append({"id": cid, "name": name, "score": round(float(score), 3)})
    for r in results:
        r["metrics"] = metrics.get(int(r["id"]))
        r["characters"] = chars.get(int(r["id"]), [])
    return results


def character_query_boost(conn: sqlite3.Connection, query: str, sids: np.ndarray,
                          boost: float = CHARACTER_QUERY_BOOST) -> Optional[np.ndarray]:
    """Characters double as search tags: a query mentioning a character's
    name ("gojo running") lifts that character's scenes."""
    q = f" {query.lower()} "
    hits = [cid for cid, name in conn.execute("SELECT id, name FROM characters").fetchall()
            if name and f" {name.lower().strip()} " in q]
    if not hits:
        return None
    rows = conn.execute(
        f"""SELECT cm.scene_id, MAX(cm.score) FROM character_matches cm
              JOIN characters c ON c.id = cm.character_id
             WHERE cm.character_id IN ({','.join('?' * len(hits))}) AND cm.score >= c.threshold
             GROUP BY cm.scene_id""",
        hits,
    ).fetchall()
    if not rows:
        return None
    ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
    out = np.zeros(len(sids), dtype=np.float32)
    out[np.isin(sids, ids, assume_unique=True)] = boost
    return out


def rank(scores: np.ndarray, threshold: float, top_k: int, sort: str) -> np.ndarray:
    """Threshold + order like /api/search (relevance or random)."""
    candidate_idx = np.where(scores >= threshold)[0]
    if sort == "random":
        ordered = candidate_idx.tolist()
        random.shuffle(ordered)
        return np.array(ordered[:top_k], dtype=np.int64)
    if len(candidate_idx) > top_k:
        part = np.argpartition(-scores[candidate_idx], top_k - 1)[:top_k]
        candidate_idx = candidate_idx[part]
    return candidate_idx[np.argsort(-scores[candidate_idx], kind="stable")]


def post_sort(results: list[dict], sort: str) -> list[dict]:
    if sort == "duration":
        results.sort(key=lambda r: r["end_ms"] - r["start_ms"], reverse=True)
    elif sort == "video":
        results.sort(key=lambda r: (r["video_display"], r["start_ms"]))
    return results
