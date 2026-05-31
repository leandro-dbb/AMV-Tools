"""Videos + tags endpoints (Roadmap §3 + Feature 3)."""
from fastapi import APIRouter, Query

from ..db import queries, schema
from ..state import get_state

router = APIRouter()


@router.get("/api/videos")
def list_videos():
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        return {"videos": queries.list_videos(conn)}


@router.get("/api/videos/{video_id}/tags")
def video_tags(video_id: int):
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        return {"tags": queries.video_tags_with_counts(conn, video_id)}


@router.get("/api/videos/{video_id}/scenes")
def video_scenes(video_id: int,
                 tag: str | None = Query(default=None),
                 threshold: float = Query(default=0.0),
                 sort: str = Query(default="timecode")):
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        scenes = queries.scenes_by_video_and_tag(conn, video_id, tag, threshold, sort)
        sids = [s["id"] for s in scenes]
        tags_for = queries.get_scene_tags(conn, sids)
        proxy_rows = conn.execute(
            f"SELECT id, proxy_path FROM scenes WHERE id IN ({','.join('?' * len(sids))})", sids
        ).fetchall() if sids else []
    has_proxy = {sid: bool(p) for sid, p in proxy_rows}
    for s in scenes:
        s["tags"] = tags_for.get(s["id"], [])[:8]
        s["proxy_path"] = True if has_proxy.get(s["id"]) else None
    return {"scenes": scenes}


@router.get("/api/scenes")
def scenes_multi(video_id: list[int] = Query(default=[]),
                 tag: str | None = Query(default=None),
                 threshold: float = Query(default=0.0),
                 sort: str = Query(default="timecode")):
    """Multi-video aggregation for the multi-select video picker."""
    state = get_state()
    out: list[dict] = []
    with schema.get_conn(state.primary_db) as conn:
        for vid in video_id:
            out.extend(queries.scenes_by_video_and_tag(conn, vid, tag, threshold, sort))
        sids = [s["id"] for s in out]
        tags_for = queries.get_scene_tags(conn, sids)
        proxy_rows = conn.execute(
            f"SELECT id, proxy_path FROM scenes WHERE id IN ({','.join('?' * len(sids))})", sids
        ).fetchall() if sids else []
    has_proxy = {sid: bool(p) for sid, p in proxy_rows}
    for s in out:
        s["tags"] = tags_for.get(s["id"], [])[:8]
        s["proxy_path"] = True if has_proxy.get(s["id"]) else None
    out.sort(key=lambda r: (r["video_display"], r["start_ms"]))
    return {"scenes": out}
