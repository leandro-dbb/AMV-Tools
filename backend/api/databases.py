from pathlib import Path
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from ..db import queries, schema
from ..db.merge import merge_databases
from ..state import get_state

router = APIRouter()


@router.get("/api/databases")
def list_databases():
    state = get_state()
    out = []
    for db in state.active_dbs:
        stats = queries.db_stats(db)
        out.append({
            "path": db,
            "scenes": stats["scenes"],
            "videos": stats["videos"],
            "size_kb": stats["size_kb"],
            "primary": db == state.primary_db,
        })
    return {"databases": out}


class PrimaryBody(BaseModel):
    path: str


@router.post("/api/databases/primary")
def set_primary(body: PrimaryBody):
    state = get_state()
    schema.init_db(body.path)
    new_active = state.settings["databases"]["active"]
    if body.path not in new_active:
        new_active = [*new_active, body.path]
    state.update_settings({"databases": {"active": new_active, "primary": body.path}})
    return {"ok": True}


class AddBody(BaseModel):
    path: str


@router.post("/api/databases")
def add_database(body: AddBody):
    state = get_state()
    schema.init_db(body.path)
    actives = state.settings["databases"]["active"]
    if body.path not in actives:
        state.update_settings({"databases": {"active": [*actives, body.path]}})
    return {"ok": True}


@router.delete("/api/databases")
def remove_database(path: str):
    state = get_state()
    actives = [p for p in state.settings["databases"]["active"] if p != path]
    primary = state.settings["databases"]["primary"]
    if primary == path:
        primary = actives[0] if actives else ""
    state.update_settings({"databases": {"active": actives, "primary": primary}})
    return {"ok": True}


@router.post("/api/databases/cleanup")
def cleanup(body: AddBody):
    """Remove video rows whose filepath no longer exists on disk."""
    removed = queries.cleanup_orphans(body.path)
    return {"removed": removed}


class VerifyResponse(BaseModel):
    missing: List[dict]


@router.get("/api/databases/verify")
def verify(path: str):
    schema.init_db(path)
    with schema.get_conn(path) as conn:
        rows = queries.all_video_paths(conn)
    missing = [{"id": vid, "filepath": fp} for vid, fp in rows if not Path(fp).exists()]
    return {"missing": missing}


class RelinkBody(BaseModel):
    db_path: str
    video_id: int
    new_filepath: str


@router.post("/api/databases/relink")
def relink(body: RelinkBody):
    if not Path(body.new_filepath).exists():
        return {"ok": False, "error": "new file does not exist"}
    with schema.get_conn(body.db_path) as conn:
        ok = queries.update_video_filepath(conn, body.video_id, body.new_filepath)
    return {"ok": ok, "error": None if ok else "filepath collision"}


class MergeBody(BaseModel):
    target: str
    sources: List[str]


@router.post("/api/databases/merge")
def merge(body: MergeBody):
    state = get_state()
    stats = merge_databases(body.target, body.sources, progress_cb=lambda m: state.publish({"type": "merge", "message": m}))
    return stats
