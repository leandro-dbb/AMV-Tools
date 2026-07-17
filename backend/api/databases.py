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


class RelinkFolderBody(BaseModel):
    db_path: str
    search_dirs: List[str]
    # When set, only these videos are candidates for relinking — used by the
    # per-subfolder flow (e.g. relink just "S23" against its new folder).
    video_ids: List[int] | None = None


@router.post("/api/databases/relink_folder")
def relink_folder(body: RelinkFolderBody):
    """Bulk relink after moving a library to another machine.

    Scans the given folders (recursively) for video files and rewrites the
    filepath of every missing video whose basename matches exactly one
    candidate. When several candidates share a basename, the one whose parent
    folder name matches the old path's parent wins; otherwise the video is
    reported as ambiguous and left untouched.
    """
    import os

    from ..config import VIDEO_EXTENSIONS

    # Index every video file found under the search dirs by lowercased basename.
    candidates: dict[str, list[str]] = {}
    for root_dir in body.search_dirs:
        if not Path(root_dir).is_dir():
            continue
        for dirpath, _dirs, files in os.walk(root_dir):
            for fname in files:
                if fname.lower().endswith(VIDEO_EXTENSIONS):
                    candidates.setdefault(fname.lower(), []).append(os.path.join(dirpath, fname))

    schema.init_db(body.db_path)
    only_ids = set(body.video_ids) if body.video_ids is not None else None
    relinked = 0
    ambiguous: List[str] = []
    still_missing: List[str] = []
    with schema.get_conn(body.db_path) as conn:
        for vid, fp in queries.all_video_paths(conn):
            if only_ids is not None and vid not in only_ids:
                continue
            if Path(fp).exists():
                continue
            matches = candidates.get(Path(fp).name.lower(), [])
            if len(matches) > 1:
                # Disambiguate by parent folder name (e.g. "Season 2/ep01.mkv").
                old_parent = Path(fp).parent.name.lower()
                narrowed = [m for m in matches if Path(m).parent.name.lower() == old_parent]
                matches = narrowed if len(narrowed) == 1 else matches
            if len(matches) == 1:
                if queries.update_video_filepath(conn, vid, matches[0]):
                    relinked += 1
                else:
                    still_missing.append(fp)
            elif len(matches) > 1:
                ambiguous.append(fp)
            else:
                still_missing.append(fp)

    return {
        "relinked": relinked,
        "ambiguous": ambiguous,
        "still_missing": still_missing,
    }


class MergeBody(BaseModel):
    target: str
    sources: List[str]


@router.post("/api/databases/merge")
def merge(body: MergeBody):
    state = get_state()
    stats = merge_databases(body.target, body.sources, progress_cb=lambda m: state.publish({"type": "merge", "message": m}))
    return stats
