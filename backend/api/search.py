"""Hybrid SigLIP 2 + wd-tagger boosted search (Roadmap §5 Feature 2)."""
from __future__ import annotations

import random
import re
from typing import Optional

import io

import numpy as np
from fastapi import APIRouter, File, Form, UploadFile
from PIL import Image
from pydantic import BaseModel

from ..db import queries, schema
from ..state import get_state

router = APIRouter()

_TOKENIZE_RE = re.compile(r"[A-Za-z0-9_]+")


class SearchRequest(BaseModel):
    query: Optional[str] = None
    image_path: Optional[str] = None
    top_k: int = 200
    threshold: float = 0.25
    sort: str = "relevance"
    tag_boost: Optional[float] = None


@router.post("/api/search")
def search(req: SearchRequest):
    state = get_state()
    db_path = state.primary_db
    schema.init_db(db_path)

    if not req.query and not req.image_path:
        return {"results": [], "total": 0}

    # Cheap pre-check: don't bother loading 400 MB of model weights for an empty library.
    index = state.get_search_index()
    scenes = index["rows"]
    if not scenes:
        return {"results": [], "total": 0}

    siglip = state.get_siglip()
    if req.image_path:
        from PIL import Image
        img = Image.open(req.image_path).convert("RGB")
        query_vec = siglip.embed_images([img])[0]
    else:
        query_vec = siglip.embed_text(req.query or "")

    sids = index["sids"]
    embeddings = index["embeddings"]
    sims = embeddings @ query_vec.astype(np.float32)

    boost = req.tag_boost if req.tag_boost is not None else state.settings["search"]["tag_boost"]
    boosts = np.zeros_like(sims)
    if req.query and boost > 0:
        tokens = {t.lower() for t in _TOKENIZE_RE.findall(req.query)}
        # Drop common stop-tokens that never carry tag meaning.
        tokens -= {"vs", "and", "or", "the", "a", "an", "of", "in", "on", "et", "le", "la", "les", "un", "une"}
        if tokens:
            placeholders = ",".join("?" for _ in tokens)
            # First find which of the query tokens actually exist as tags in
            # the DB. Tokens that aren't tags ("explosion", "sunset", proper
            # nouns wd-tagger doesn't know) shouldn't penalize coverage.
            with schema.get_conn(db_path) as conn:
                actual_tags = {
                    r[0] for r in conn.execute(
                        f"SELECT DISTINCT LOWER(tag) FROM scene_tags WHERE LOWER(tag) IN ({placeholders})",
                        list(tokens),
                    ).fetchall()
                }
            n_target = len(actual_tags)
            if n_target > 0:
                tag_placeholders = ",".join("?" for _ in actual_tags)
                with schema.get_conn(db_path) as conn:
                    tag_rows = conn.execute(
                        f"SELECT scene_id, COUNT(DISTINCT LOWER(tag)) FROM scene_tags "
                        f"WHERE LOWER(tag) IN ({tag_placeholders}) GROUP BY scene_id",
                        list(actual_tags),
                    ).fetchall()
                # Coverage-based boost: a scene that matches all the query's
                # tag-tokens gets the full boost; partial matches get
                # proportionally less. This is what makes
                # "lucario vs dracaufeu" rank co-occurring scenes above scenes
                # with only one of the two.
                for sid, n_matched in tag_rows:
                    pos = index["id_to_pos"].get(int(sid))
                    if pos is not None:
                        coverage = n_matched / n_target
                        boosts[pos] = coverage * boost

    scores = np.clip(sims + boosts, -1.0, 1.5)
    candidate_idx = np.where(scores >= req.threshold)[0]

    if req.sort == "random":
        ordered = candidate_idx.tolist()
        random.shuffle(ordered)
        keep_idx = np.array(ordered[: req.top_k], dtype=np.int64)
    else:
        order = np.argsort(-scores[candidate_idx])
        keep_idx = candidate_idx[order][: req.top_k]

    tag_sids = [int(sids[i]) for i in keep_idx]
    with schema.get_conn(db_path) as conn:
        tags_for = queries.get_scene_tags(conn, tag_sids)
        proxy_rows = conn.execute(
            f"SELECT id, proxy_path FROM scenes WHERE id IN ({','.join('?' * len(tag_sids))})",
            tag_sids,
        ).fetchall() if tag_sids else []
    has_proxy = {sid: bool(p) for sid, p in proxy_rows}

    results = []
    for i in keep_idx:
        sid, vid_id, vid_display, scene_idx, s_ms, e_ms, _ = scenes[int(i)]
        results.append({
            "id": sid,
            "video_id": vid_id,
            "video_display": vid_display,
            "scene_index": scene_idx,
            "start_ms": s_ms,
            "end_ms": e_ms,
            "score": float(scores[int(i)]),
            "tags": tags_for.get(sid, [])[:8],
            "proxy_path": True if has_proxy.get(sid) else None,
        })

    if req.query:
        with schema.get_conn(db_path) as conn:
            queries.record_search(conn, req.query)

    if req.sort == "duration":
        results.sort(key=lambda r: r["end_ms"] - r["start_ms"], reverse=True)
    elif req.sort == "video":
        results.sort(key=lambda r: (r["video_display"], r["start_ms"]))

    return {"results": results, "total": len(results)}


@router.get("/api/search/history")
def history():
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        return {"queries": queries.recent_searches(conn, limit=10)}


@router.post("/api/search/image")
async def search_image(image: UploadFile = File(...),
                       threshold: float = Form(default=0.25),
                       top_k: int = Form(default=200)):
    state = get_state()
    db_path = state.primary_db
    schema.init_db(db_path)
    raw = await image.read()
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    siglip = state.get_siglip()
    query_vec = siglip.embed_images([pil])[0]

    index = state.get_search_index()
    rows = index["rows"]
    if not rows:
        return {"results": [], "total": 0}

    sids = index["sids"]
    embeddings = index["embeddings"]
    sims = embeddings @ query_vec.astype(np.float32)
    keep_idx = np.where(sims >= threshold)[0]
    keep_idx = keep_idx[np.argsort(-sims[keep_idx])][:top_k]

    tag_sids = [int(sids[i]) for i in keep_idx]
    with schema.get_conn(db_path) as conn:
        tags_for = queries.get_scene_tags(conn, tag_sids)
        proxy_rows = conn.execute(
            f"SELECT id, proxy_path FROM scenes WHERE id IN ({','.join('?' * len(tag_sids))})",
            tag_sids,
        ).fetchall() if tag_sids else []
    has_proxy = {sid: bool(p) for sid, p in proxy_rows}

    results = []
    for i in keep_idx:
        sid, vid_id, vid_display, scene_idx, s_ms, e_ms, _ = rows[int(i)]
        results.append({
            "id": sid, "video_id": vid_id, "video_display": vid_display,
            "scene_index": scene_idx, "start_ms": s_ms, "end_ms": e_ms,
            "score": float(sims[int(i)]), "tags": tags_for.get(sid, [])[:8],
            "proxy_path": True if has_proxy.get(sid) else None,
        })
    return {"results": results, "total": len(results)}
