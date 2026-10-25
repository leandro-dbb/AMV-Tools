"""Model tiers + embeddings-only re-index.

  GET  /api/models/tiers            platform, detected memory, tiers, embedding state
  GET  /api/models/reembed          last/current re-embed job (or null)
  POST /api/models/reembed          {"model"?: repo id, "force"?: bool} -> start
  POST /api/models/reembed/cancel

Progress streams on ``/ws/progress`` as ``reembed`` events:
``{"type": "reembed", "percent", "message", "done", "ok", "videos_missing"}``.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..db import schema
from ..indexing import reembed
from ..models import tiers
from ..state import get_state

router = APIRouter()


class ReembedBody(BaseModel):
    model: Optional[str] = None
    force: bool = False


def _indexing_running(state) -> bool:
    from ..db.merge import merge_running
    t = getattr(state, "_index_thread", None)
    return bool(t is not None and t.is_alive()) or merge_running()


def _job_summary() -> Optional[dict]:
    st = reembed.status()
    if st is None:
        return None
    return {k: st[k] for k in ("running", "percent", "message", "error", "done", "ok",
                                "videos_missing", "target_model", "from_model", "switched")}


@router.get("/api/models/tiers")
def model_tiers():
    state = get_state()
    sel = state.model_selection()
    db = state.primary_db
    schema.init_db(db)
    target = sel["models"]["embedding"]["model"]
    emb = reembed.embedding_status(db, target)
    emb["job"] = _job_summary()
    return {
        "platform": sel["platform"],
        "detected_gb": sel["detected_gb"],
        "auto_tier": sel["auto_tier"],
        "selected": sel["selected"],
        "effective_tier": sel["effective_tier"],
        "tiers": tiers.describe_tiers(sel["platform"]),
        "embedding": emb,
    }


@router.get("/api/models/reembed")
def reembed_status():
    return {"job": reembed.status()}


@router.post("/api/models/reembed")
def reembed_start(body: Optional[ReembedBody] = None):
    body = body or ReembedBody()
    state = get_state()
    if _indexing_running(state):
        raise HTTPException(409, "indexing is running — re-embed after it finishes")
    target = (body.model or "").strip() or state.embedding_target()[0]
    info = tiers.MODEL_INFO.get(target)
    if info is not None and info.get("kind") not in (None, "embedding"):
        raise HTTPException(400, f"{target} is not an embedding model")
    schema.init_db(state.primary_db)
    try:
        job = reembed.start(state, target, force=body.force,
                            indexing_running=lambda: _indexing_running(state))
    except reembed.ReembedBusy as exc:
        raise HTTPException(409, str(exc))
    return {"ok": True, "started": True, "job": job.to_dict()}


@router.post("/api/models/reembed/cancel")
def reembed_cancel():
    if not reembed.cancel():
        raise HTTPException(409, "no re-embed job is running")
    return {"ok": True}
