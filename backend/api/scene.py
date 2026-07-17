"""Scene-level endpoints: thumbnail + proxy + source streaming + merge."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from ..db import queries, schema
from ..state import get_state

router = APIRouter()


class MergeRequest(BaseModel):
    scene_ids: list[int]


@router.post("/api/scenes/merge")
def merge_scenes(body: MergeRequest):
    """Merge adjacent top-level scenes into one (Derush strip selection)."""
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        try:
            result = queries.merge_scenes(conn, body.scene_ids)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    # Merged/deleted scene rows change the embedding set the search index maps to.
    state.invalidate_search_cache()
    return {"ok": True, **result}


_VIDEO_MIME = {
    "mp4": "video/mp4",
    "m4v": "video/mp4",
    "mkv": "video/x-matroska",
    "webm": "video/webm",
    "mov": "video/quicktime",
    "avi": "video/x-msvideo",
    "ts": "video/mp2t",
    "m2ts": "video/mp2t",
    "wmv": "video/x-ms-wmv",
}


@router.get("/api/scene/{scene_id}/thumbnail")
def thumbnail(scene_id: int):
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        s = queries.get_scene(conn, scene_id)
    if not s or not s["thumbnail"]:
        raise HTTPException(status_code=404, detail="no thumbnail")
    return Response(content=s["thumbnail"], media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"})


@router.get("/api/scene/{scene_id}/proxy")
def proxy(scene_id: int):
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        s = queries.get_scene(conn, scene_id)
    if not s or not s["proxy_path"] or not Path(s["proxy_path"]).exists():
        raise HTTPException(status_code=404, detail="no proxy")
    # Dispatch the right MIME type so the <video> tag accepts both legacy
    # VP9 .webm files and the new NVENC H.264 .mp4 files.
    media_type = "video/mp4" if s["proxy_path"].lower().endswith(".mp4") else "video/webm"
    return FileResponse(s["proxy_path"], media_type=media_type)


@router.get("/api/scene/{scene_id}/source")
def source(scene_id: int):
    """Stream the original video file. FastAPI's FileResponse advertises
    Accept-Ranges, so Chromium's <video> tag can seek directly to a scene's
    start_ms without loading the whole file."""
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        s = queries.get_scene(conn, scene_id)
    if not s:
        raise HTTPException(status_code=404, detail="scene not found")
    filepath = s["filepath"]
    if not filepath or not Path(filepath).exists():
        raise HTTPException(status_code=404, detail="source missing")
    ext = Path(filepath).suffix.lower().lstrip(".")
    return FileResponse(filepath, media_type=_VIDEO_MIME.get(ext, "video/mp4"))


@router.get("/api/scene/{scene_id}/preview")
def preview(scene_id: int):
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        s = queries.get_scene(conn, scene_id)
    if not s:
        raise HTTPException(status_code=404)
    return {
        "frames": [f"/api/scene/{scene_id}/thumbnail"],
        "proxy_url": f"/api/scene/{scene_id}/proxy" if s["proxy_path"] else None,
    }


@router.get("/api/scene/{scene_id}/info")
def scene_info(scene_id: int):
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        s = queries.get_scene(conn, scene_id)
        if not s:
            raise HTTPException(404)
        tags = queries.get_scene_tags(conn, [scene_id]).get(scene_id, [])
    return {
        "id": s["id"],
        "video_id": s["video_id"],
        "video_display": s["display_name"],
        "filepath": s["filepath"],
        "scene_index": s["scene_index"],
        "start_ms": s["start_ms"],
        "end_ms": s["end_ms"],
        "has_proxy": bool(s["proxy_path"]),
        "tags": tags,
    }
