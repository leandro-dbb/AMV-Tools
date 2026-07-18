"""Derush (selects) endpoints — the AMV-maker dailies workflow.

The Derush tab plays whole episodes with JKL shuttle; one keystroke keeps the
scene under the playhead. Kept scenes land in ``derush_items`` (optionally
grouped into ``derush_folders``), can be renamed, and are batch-exported with
the regular export settings.

Batch export runs in a worker thread and streams per-item progress through
``/ws/progress`` as events of type ``derush_export`` so the UI can show a bar
without blocking the HTTP call.
"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..db import queries, schema
from ..export import export_scene
from ..paths import user_data_dir
from ..state import get_state

log = logging.getLogger(__name__)
router = APIRouter()

_export_lock = threading.Lock()
_export_running = False


# ── bodies ──────────────────────────────────────────────────────────────────
class ToggleBody(BaseModel):
    scene_id: int
    folder_id: Optional[int] = None
    # True = the "Favoris" level (ù key): same toggle rhythm as a plain keep,
    # but flags the item for pre-treatment. On an existing plain keep it
    # upgrades instead of removing.
    favorite: bool = False


class ItemPatch(BaseModel):
    # Sentinel-free PATCH: fields are only applied when present in the JSON.
    custom_name: Optional[str] = None
    folder_ids: Optional[list[int]] = None  # replace ALL folder memberships
    clear_name: bool = False                # explicit "drop the custom name"
    favorite: Optional[bool] = None         # flip the Favoris flag


class FolderBody(BaseModel):
    name: str


class ExportBody(BaseModel):
    # "all" (default), a folder id, root (unfiled), or favorites only.
    folder_id: Optional[int] = None
    scope: str = "all"               # "all" | "folder" | "root" | "favorites"


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip("_ ") or "scene"


# ── listing ─────────────────────────────────────────────────────────────────
@router.get("/api/derush")
def derush_list():
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        folders, items = queries.derush_list(conn)
    return {"folders": folders, "items": items, "exporting": _export_running}


@router.get("/api/derush/kept")
def derush_kept(video_id: Optional[int] = None):
    """Scene ids currently kept — the player uses this to paint the segment
    strip of the loaded episode."""
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        ids = queries.derush_kept_scene_ids(conn, video_id)
    return {"scene_ids": sorted(ids)}


class LevelBody(BaseModel):
    scene_id: int
    delta: int                       # +1 = up the ladder, -1 = down
    folder_id: Optional[int] = None


# ── keep / unkeep ───────────────────────────────────────────────────────────
@router.post("/api/derush/level")
def derush_level(body: LevelBody):
    """Arrow-key keep ladder: none → kept → favorite (+1) and back down (-1)."""
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        scene = queries.get_scene(conn, body.scene_id)
        if not scene:
            raise HTTPException(404, "scene not found")
        level, item_id = queries.derush_set_level(
            conn, body.scene_id, body.delta, body.folder_id,
        )
    return {"ok": True, "level": level, "kept": level > 0, "favorite": level == 2,
            "item_id": item_id, "scene_id": body.scene_id}


@router.post("/api/derush/toggle")
def derush_toggle(body: ToggleBody):
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        scene = queries.get_scene(conn, body.scene_id)
        if not scene:
            raise HTTPException(404, "scene not found")
        kept, item_id, favorite = queries.derush_toggle(
            conn, body.scene_id, body.folder_id, favorite=body.favorite,
        )
    return {"ok": True, "kept": kept, "item_id": item_id, "scene_id": body.scene_id,
            "favorite": favorite}


@router.patch("/api/derush/items/{item_id}")
def derush_patch_item(item_id: int, body: ItemPatch):
    state = get_state()
    custom_name = ...
    if body.clear_name:
        custom_name = None
    elif body.custom_name is not None:
        custom_name = body.custom_name.strip() or None
    favorite = ... if body.favorite is None else bool(body.favorite)
    with schema.get_conn(state.primary_db) as conn:
        ok = queries.derush_update_item(conn, item_id, custom_name=custom_name,
                                        favorite=favorite)
        if ok and body.folder_ids is not None:
            ok = queries.derush_set_item_folders(conn, item_id, body.folder_ids)
    if not ok:
        raise HTTPException(404, "item not found")
    return {"ok": True}


@router.delete("/api/derush/items/{item_id}")
def derush_delete_item(item_id: int):
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        cur = conn.execute("DELETE FROM derush_items WHERE id = ?", (item_id,))
    return {"ok": True, "deleted": cur.rowcount > 0}


# ── folders ─────────────────────────────────────────────────────────────────
@router.post("/api/derush/folders")
def derush_create_folder(body: FolderBody):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "folder name is empty")
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        cur = conn.execute("INSERT INTO derush_folders (name) VALUES (?)", (name,))
    return {"ok": True, "id": cur.lastrowid, "name": name}


@router.patch("/api/derush/folders/{folder_id}")
def derush_rename_folder(folder_id: int, body: FolderBody):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "folder name is empty")
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        cur = conn.execute("UPDATE derush_folders SET name = ? WHERE id = ?", (name, folder_id))
    if cur.rowcount == 0:
        raise HTTPException(404, "folder not found")
    return {"ok": True}


@router.delete("/api/derush/folders/{folder_id}")
def derush_delete_folder(folder_id: int):
    """Items fall back to the root (ON DELETE SET NULL), they are NOT removed."""
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        cur = conn.execute("DELETE FROM derush_folders WHERE id = ?", (folder_id,))
    return {"ok": True, "deleted": cur.rowcount > 0}


# ── batch export ────────────────────────────────────────────────────────────
def _suffix_for_codec(codec: str) -> str:
    if codec in ("libx264", "libx265", "libsvtav1", "h264_nvenc", "h264_videotoolbox"):
        return ".mp4"
    if codec in ("prores_ks", "dnxhr"):
        return ".mov"
    return ".webm"


def _export_worker(items: list[dict], export_settings: dict) -> None:
    global _export_running
    state = get_state()
    base_dir = Path(export_settings["output_folder"]) if export_settings["output_folder"] \
        else (user_data_dir() / "exports")
    suffix = _suffix_for_codec(export_settings["codec"])
    total = len(items)
    done, failed = 0, 0
    try:
        for i, it in enumerate(items):
            # exports/derush/<folder>/<custom name | episode_scene>.<ext>
            out_dir = base_dir / "derush" / (_sanitize(it["folder_name"]) if it["folder_name"] else "")
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = it["custom_name"] or f"{it['display_name']}_scene{it['scene_index']:04d}"
            target = out_dir / f"{_sanitize(stem)}{suffix}"
            n = 1
            while target.exists():
                n += 1
                target = out_dir / f"{_sanitize(stem)}_{n}{suffix}"
            state.publish({
                "type": "derush_export", "current": i + 1, "total": total,
                "percent": int(i / max(1, total) * 100),
                "message": target.name, "done": False,
            })
            try:
                if not Path(it["filepath"]).exists():
                    raise RuntimeError("source file missing")
                export_scene(
                    it["filepath"], it["start_ms"], it["end_ms"], str(target),
                    codec=export_settings["codec"], crf=export_settings["crf"],
                    audio=export_settings["audio"], resolution=export_settings["resolution"],
                    audio_bitrate_kbps=int(export_settings.get("audio_bitrate_kbps", 320)),
                )
                done += 1
            except Exception as exc:
                failed += 1
                log.warning("derush export failed for item %s: %s", it["item_id"], exc)
        state.publish({
            "type": "derush_export", "current": total, "total": total, "percent": 100,
            "message": f"{done} exported" + (f", {failed} failed" if failed else ""),
            "done": True, "ok": failed == 0,
            "output": str(base_dir / "derush"),
        })
    finally:
        with _export_lock:
            _export_running = False


@router.post("/api/derush/export")
def derush_export(body: ExportBody):
    global _export_running
    state = get_state()
    if body.scope == "folder":
        if body.folder_id is None:
            raise HTTPException(400, "scope=folder requires folder_id")
        selector: Union[int, None, str] = body.folder_id
    elif body.scope == "root":
        selector = None
    elif body.scope == "favorites":
        selector = "fav"
    else:
        selector = "all"

    with schema.get_conn(state.primary_db) as conn:
        items = queries.derush_items_for_export(conn, selector)
    if not items:
        raise HTTPException(400, "nothing to export in this scope")

    with _export_lock:
        if _export_running:
            raise HTTPException(409, "a derush export is already running")
        _export_running = True

    export_settings = dict(state.settings["export"])
    t = threading.Thread(
        target=_export_worker, args=(items, export_settings), daemon=True,
    )
    t.start()
    return {"ok": True, "started": True, "total": len(items)}
