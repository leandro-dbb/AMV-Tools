"""Videos + tags endpoints (Roadmap §3 + Feature 3) and Derush episode playback."""
import logging
import subprocess
import threading
from pathlib import Path

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..db import queries, schema
from ..paths import user_data_dir
from ..state import get_state
from .scene import _VIDEO_MIME

log = logging.getLogger(__name__)
router = APIRouter()

# Containers Chromium's <video> demuxes natively. Everything else (mkv, ts…)
# goes through the remux cache: stream-copy into .mp4 — a few seconds per
# episode, no quality loss. Codecs Chromium can't DECODE (hevc, 10-bit h264)
# still fail after remux; the client then requests the transcode fallback.
_DIRECT_PLAYABLE = {".mp4", ".m4v", ".webm"}
_prepare_lock = threading.Lock()
_preparing: set[int] = set()


def _video_path_or_404(video_id: int) -> str:
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        row = conn.execute("SELECT filepath FROM videos WHERE id = ?", (video_id,)).fetchone()
    if not row:
        raise HTTPException(404, "video not found")
    if not row[0] or not Path(row[0]).exists():
        raise HTTPException(410, "source file missing")
    return row[0]


def _cache_dir() -> Path:
    p = user_data_dir() / "derush_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


@router.get("/api/videos")
def list_videos():
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        return {"videos": queries.list_videos(conn)}


class VideoPatch(BaseModel):
    group_name: Optional[str] = None
    clear_group: bool = False
    derushed: Optional[bool] = None   # "I'm done derushing this episode"


class GroupRename(BaseModel):
    old: str
    new: Optional[str] = None        # None/empty = dissolve the folder


@router.post("/api/videos/groups/rename")
def rename_group(body: GroupRename):
    """Rename an import group across all its episodes, or dissolve it
    (new=None → the episodes fall back to ungrouped)."""
    old = body.old.strip()
    if not old:
        raise HTTPException(400, "old group name is empty")
    new = (body.new or "").strip() or None
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        cur = conn.execute("UPDATE videos SET group_name = ? WHERE group_name = ?", (new, old))
    return {"ok": True, "updated": cur.rowcount, "group_name": new}


@router.patch("/api/videos/{video_id}")
def patch_video(video_id: int, body: VideoPatch):
    """Per-episode updates: import group (the collapsible playlist folders)
    and/or the "derushed" done-flag. Only the fields present are touched."""
    state = get_state()
    sets, params = [], []
    if body.clear_group or body.group_name is not None:
        group = None if body.clear_group else ((body.group_name or "").strip() or None)
        sets.append("group_name = ?")
        params.append(group)
    if body.derushed is not None:
        sets.append("derushed = ?")
        params.append(int(body.derushed))
    if not sets:
        raise HTTPException(400, "nothing to update")
    params.append(video_id)
    with schema.get_conn(state.primary_db) as conn:
        cur = conn.execute(f"UPDATE videos SET {', '.join(sets)} WHERE id = ?", params)
    if cur.rowcount == 0:
        raise HTTPException(404, "video not found")
    return {"ok": True}


@router.get("/api/videos/{video_id}/source")
def video_source(video_id: int):
    """Stream the original episode file. FileResponse advertises Accept-Ranges
    so Chromium's <video> can seek freely."""
    filepath = _video_path_or_404(video_id)
    ext = Path(filepath).suffix.lower().lstrip(".")
    return FileResponse(filepath, media_type=_VIDEO_MIME.get(ext, "video/mp4"))


@router.get("/api/videos/{video_id}/playable")
def video_playable(video_id: int):
    """Resolve a URL the Derush player can actually play.

    - .mp4/.webm sources stream directly.
    - Other containers (mkv…) are stream-copied into a cached .mp4 once
      (synchronous — a few seconds, the UI shows a loader).
    - A previously transcoded decode-proof variant wins if present.
    """
    filepath = Path(_video_path_or_404(video_id))
    enc = _cache_dir() / f"{video_id}.enc.mp4"
    if enc.exists():
        return {"url": f"/api/videos/{video_id}/stream?variant=enc", "kind": "transcode"}

    if filepath.suffix.lower() in _DIRECT_PLAYABLE:
        return {"url": f"/api/videos/{video_id}/source", "kind": "direct"}

    from ..indexing.cuts import _ffmpeg_path
    from ..export.ffmpeg import _creation_flags

    remux = _cache_dir() / f"{video_id}.remux.mp4"
    if not remux.exists() or remux.stat().st_mtime < filepath.stat().st_mtime:
        tmp = remux.with_suffix(".tmp.mp4")
        cmd = [
            _ffmpeg_path(), "-hide_banner", "-loglevel", "error",
            "-i", str(filepath),
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ac", "2",
            "-sn", "-dn",
            "-movflags", "+faststart",
            "-y", str(tmp),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              creationflags=_creation_flags())
        if proc.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise HTTPException(500, f"remux failed: {proc.stderr[-400:]}")
        tmp.replace(remux)
    return {"url": f"/api/videos/{video_id}/stream?variant=remux", "kind": "remux"}


@router.get("/api/videos/{video_id}/stream")
def video_stream(video_id: int, variant: str = Query(default="remux")):
    _video_path_or_404(video_id)  # 404/410 if the library entry vanished
    if variant not in ("remux", "enc"):
        raise HTTPException(400, "variant must be remux or enc")
    f = _cache_dir() / f"{video_id}.{variant}.mp4"
    if not f.exists():
        raise HTTPException(404, "cached variant not found")
    return FileResponse(f, media_type="video/mp4")


@router.post("/api/videos/{video_id}/transcode")
def video_transcode(video_id: int):
    """Decode-proof fallback for sources Chromium can't decode even after the
    remux (hevc, 10-bit h264): background H.264 8-bit transcode (NVENC when
    available), progress streamed as ``derush_prepare`` events."""
    filepath = _video_path_or_404(video_id)
    enc = _cache_dir() / f"{video_id}.enc.mp4"
    if enc.exists():
        return {"ok": True, "ready": True, "url": f"/api/videos/{video_id}/stream?variant=enc"}

    with _prepare_lock:
        if video_id in _preparing:
            return {"ok": True, "ready": False, "already_running": True}
        _preparing.add(video_id)

    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        row = conn.execute("SELECT duration_ms FROM videos WHERE id = ?", (video_id,)).fetchone()
    duration_ms = (row[0] or 0) if row else 0

    def worker():
        from ..indexing.cuts import _ffmpeg_path
        from ..export.ffmpeg import _creation_flags, _has_nvenc_h264
        tmp = enc.with_suffix(".tmp.mp4")
        vcodec = (["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
                  if _has_nvenc_h264() else
                  ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"])
        cmd = [
            _ffmpeg_path(), "-hide_banner", "-loglevel", "error",
            "-i", filepath,
            "-map", "0:v:0", "-map", "0:a:0?",
            *vcodec, "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ac", "2",
            "-sn", "-dn", "-movflags", "+faststart",
            "-progress", "pipe:1", "-nostats",
            "-y", str(tmp),
        ]
        ok = False
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    text=True, creationflags=_creation_flags())
            assert proc.stdout is not None
            for line in proc.stdout:
                if line.startswith("out_time_ms=") and duration_ms > 0:
                    try:
                        done_ms = int(line.split("=", 1)[1]) / 1000.0
                        state.publish({
                            "type": "derush_prepare", "video_id": video_id,
                            "percent": min(99, int(done_ms / duration_ms * 100)),
                            "done": False,
                        })
                    except ValueError:
                        pass
            ok = proc.wait() == 0
            if ok:
                tmp.replace(enc)
            else:
                tmp.unlink(missing_ok=True)
        except Exception as exc:
            log.warning("derush transcode failed for video %s: %s", video_id, exc)
            tmp.unlink(missing_ok=True)
        finally:
            with _prepare_lock:
                _preparing.discard(video_id)
            state.publish({
                "type": "derush_prepare", "video_id": video_id,
                "percent": 100, "done": True, "ok": ok,
            })

    threading.Thread(target=worker, daemon=True).start()
    return {"ok": True, "ready": False, "started": True}


@router.get("/api/videos/{video_id}/tags")
def video_tags(video_id: int):
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        return {"tags": queries.video_tags_with_counts(conn, video_id)}


@router.get("/api/videos/{video_id}/scenes")
def video_scenes(video_id: int,
                 tag: str | None = Query(default=None),
                 threshold: float = Query(default=0.0),
                 sort: str = Query(default="timecode"),
                 top_only: bool = Query(default=False)):
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        scenes = queries.scenes_by_video_and_tag(conn, video_id, tag, threshold, sort, top_only)
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
