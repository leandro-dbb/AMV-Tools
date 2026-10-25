"""Search+ endpoints: similar shots, filter facets, background scene analysis."""
from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from ..db import schema
from ..searchplus import characters as chars
from ..searchplus import filters as sf
from ..searchplus import store
from ..searchplus.analysis import COLOR_BUCKETS, MOTION_TYPES
from ..searchplus.jobs import get_job
from ..state import get_state

router = APIRouter()
log = logging.getLogger(__name__)

_AUTO_DELAY_S = 8.0
_auto_started = False


@router.get("/api/search/similar/{scene_id}")
def similar(scene_id: int,
            top_k: int = Query(default=100, ge=1, le=1000),
            threshold: float = Query(default=0.0),
            exclude_same_video: bool = False,
            same_group: bool = False,
            sort: str = "relevance",
            filters: Optional[str] = None):
    """"More shots like this one": nearest neighbours of the scene's own
    SigLIP 2 embedding (no new model, no query encode)."""
    state = get_state()
    db_path = state.primary_db
    schema.init_db(db_path)
    try:
        f = sf.parse_filters(filters)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"bad filters: {exc}")
    index = state.get_search_index()
    with schema.get_conn(db_path) as conn:
        row = conn.execute(
            """SELECT s.embedding, s.video_id, v.group_name, v.display_name, s.start_ms, s.end_ms
                 FROM scenes s JOIN videos v ON v.id = s.video_id WHERE s.id = ?""",
            (scene_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="scene not found")
        emb_blob, video_id, group_name, display, s_ms, e_ms = row
        seed = {"id": scene_id, "video_id": video_id, "video_display": display,
                "group_name": group_name, "start_ms": s_ms, "end_ms": e_ms}
        if not emb_blob:
            raise HTTPException(status_code=409, detail="scene has no embedding yet — run the embed phase")
        E = index["embeddings"]
        q = np.frombuffer(emb_blob, dtype=np.float32)
        if E.size == 0 or E.shape[1] != q.shape[0]:
            return {"results": [], "total": 0, "seed": seed}

        sids = index["sids"]
        scores = (E @ q).astype(np.float32)
        excluded = sids == scene_id
        row_vids = chars._row_video_ids(index)
        if exclude_same_video:
            excluded |= row_vids == video_id
        else:
            # Overlapping sub-segments of the seed are near-duplicates, not
            # "other shots like it".
            ov = [r[0] for r in conn.execute(
                "SELECT id FROM scenes WHERE video_id = ? AND start_ms < ? AND end_ms > ?",
                (video_id, e_ms, s_ms),
            )]
            excluded |= np.isin(sids, np.asarray(ov, dtype=np.int64))
        if same_group:
            if group_name is None:
                vids = [r[0] for r in conn.execute("SELECT id FROM videos WHERE group_name IS NULL")]
            else:
                vids = [r[0] for r in conn.execute("SELECT id FROM videos WHERE group_name = ?", (group_name,))]
            excluded |= ~np.isin(row_vids, np.asarray(vids, dtype=np.int64))
        if f is not None and f.characters:
            chars.ensure_fresh(conn, index, f.characters)
        mask = sf.filter_mask(conn, f, sids)
        if mask is not None:
            excluded |= ~mask
        scores[excluded] = -np.inf
        keep = sf.rank(scores, threshold, top_k, sort)
        results = sf.rows_to_results(conn, index, keep, scores)
        seed_extra = sf.enrich(conn, [dict(seed)])[0]
    return {"results": sf.post_sort(results, sort), "total": len(results), "seed": seed_extra}


@router.get("/api/searchplus/facets")
def facets():
    """Everything the filter panel needs in one call."""
    state = get_state()
    db_path = state.primary_db
    schema.init_db(db_path)
    with schema.get_conn(db_path) as conn:
        groups = [
            {"name": g if g is not None else sf.NO_GROUP, "label": g, "videos": n}
            for g, n in conn.execute(
                "SELECT group_name, COUNT(*) FROM videos GROUP BY group_name ORDER BY group_name IS NULL, group_name"
            ).fetchall()
        ]
        motion_counts = dict(conn.execute(
            "SELECT motion_type, COUNT(*) FROM scene_metrics GROUP BY motion_type"
        ).fetchall())
        color_counts: dict[str, int] = {}
        for c, n in conn.execute("SELECT dominant_color, COUNT(*) FROM scene_metrics GROUP BY dominant_color"):
            if c:
                color_counts[c] = color_counts.get(c, 0) + n
        characters = chars.list_characters(conn)
        coverage = store.coverage(conn)
    return {
        "motion_types": [{"id": m, "count": int(motion_counts.get(m, 0))} for m in MOTION_TYPES],
        "colors": [{"id": c, "count": int(color_counts.get(c, 0))} for c in COLOR_BUCKETS],
        "groups": groups,
        "characters": characters,
        "coverage": coverage,
        "analysis": dict(get_job().status),
    }


@router.get("/api/searchplus/analysis")
def analysis_status():
    state = get_state()
    schema.init_db(state.primary_db)
    with schema.get_conn(state.primary_db) as conn:
        cov = store.coverage(conn)
    return {**get_job().status, "coverage": cov}


@router.post("/api/searchplus/analyze")
def analysis_start():
    started = get_job().start(get_state())
    return {"started": started, **get_job().status}


@router.post("/api/searchplus/analyze/stop")
def analysis_stop():
    get_job().stop()
    return {"stopped": True}


def start_auto_analyze() -> None:
    """Old libraries get their metrics computed in the background shortly
    after launch (opt-out: settings ``searchplus.auto_analyze = false``).
    Called once from the sidecar lifespan (``backend/server.py``)."""
    global _auto_started
    if _auto_started:
        return
    _auto_started = True

    def later():
        try:
            state = get_state()
            if not state.settings.get("searchplus", {}).get("auto_analyze", True):
                return
            from ..paths import user_data_dir
            if not (user_data_dir() / ".setup_complete").exists():
                return
            schema.init_db(state.primary_db)
            with schema.get_conn(state.primary_db) as conn:
                pending = store.coverage(conn)["pending"]
            if pending:
                log.info("search+: %d scenes without metrics, starting background analysis", pending)
                get_job().start(state)
        except Exception as exc:
            log.warning("search+ auto analysis not started: %s", exc)

    t = threading.Timer(_AUTO_DELAY_S, later)
    t.daemon = True
    t.start()
