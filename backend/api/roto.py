"""Roto editor endpoints (v06): the AI cuts the character out, the user
finishes it with a brush and clicks, propagation carries the fixes along
the shot. Contract: docs/roto-editor-api.md. Logic: backend/roto/."""
from __future__ import annotations

import io
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..db import queries, schema
from ..roto import engine
from ..roto.edges import EdgeSettings, apply_edges
from ..roto.project import RotoProject, find_projects, get_project, register
from ..state import get_state

log = logging.getLogger(__name__)
router = APIRouter()


# ── bodies ──────────────────────────────────────────────────────────────────
class OpenBody(BaseModel):
    scene_id: int
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None


class DetectBody(BaseModel):
    layer: Optional[str] = None
    target: Literal["all", "main", "click"] = "all"
    frame: int = 0
    positive: List[Tuple[float, float]] = Field(default_factory=list)
    negative: List[Tuple[float, float]] = Field(default_factory=list)


class Stroke(BaseModel):
    points: List[Tuple[float, float]] = Field(max_length=20_000)
    radius: float = Field(default=12.0, ge=1, le=512)
    mode: Literal["add", "remove", "erase"] = "add"


class BrushBody(BaseModel):
    frame: int
    layer: Optional[str] = None
    strokes: List[Stroke] = Field(default_factory=list, max_length=64)


class ClickBody(BaseModel):
    frame: int
    layer: Optional[str] = None
    x: float
    y: float
    positive: bool = True


class LockBody(BaseModel):
    frame: int
    layer: Optional[str] = None
    locked: bool = True


class PropagateBody(BaseModel):
    frame: int
    layer: Optional[str] = None
    direction: Literal["forward", "backward", "both"] = "both"


class SettingsBody(BaseModel):
    keep_ink: bool = True
    softness: float = 0.0
    grow_px: float = 0.0
    temporal: float = 0.0


class LayerBody(BaseModel):
    name: Optional[str] = None


class ExportBody(BaseModel):
    formats: List[Literal["prores", "png", "matte"]] = Field(default_factory=lambda: ["prores", "png", "matte"])
    add_to_selection: bool = True


# ── jobs (one at a time per project) ────────────────────────────────────────
class Job:
    def __init__(self, kind: str):
        self.kind = kind
        self.percent = 0
        self.message = ""
        self.done = False
        self.cancelled = False
        self.error: Optional[str] = None
        self.result = None
        self.cancel_event = threading.Event()

    def check(self) -> None:
        if self.cancel_event.is_set():
            raise engine.Cancelled()

    def to_dict(self) -> dict:
        return {"kind": self.kind, "percent": int(self.percent), "message": self.message, "done": self.done,
                "cancelled": self.cancelled, "error": self.error, "result": self.result}


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def _running(pid: str) -> Optional[Job]:
    j = _jobs.get(pid)
    return j if j is not None and not j.done else None


def _start_job(p: RotoProject, kind: str, fn) -> dict:
    with _jobs_lock:
        if _running(p.id) is not None:
            raise HTTPException(409, "Another task is running on this shot; wait for it or cancel it.")
        job = Job(kind)
        _jobs[p.id] = job

    def run():
        try:
            fn(job)
            job.percent = 100
        except engine.Cancelled:
            job.cancelled = True
            job.message = "Cancelled"
        except Exception as exc:   # never kill the sidecar
            log.exception("roto %s job failed", kind)
            job.error = str(exc) or type(exc).__name__
        finally:
            job.done = True

    threading.Thread(target=run, name=f"roto-{kind}-{p.id}", daemon=True).start()
    return job.to_dict()


def _no_job(p: RotoProject) -> None:
    j = _running(p.id)
    if j is not None:
        raise HTTPException(409, f"Wait for the {j.kind} to finish (or cancel it) before correcting.")


# ── helpers ─────────────────────────────────────────────────────────────────
def _project_or_404(pid: str) -> RotoProject:
    p = get_project(pid)
    if p is None:
        raise HTTPException(404, "roto project not found")
    return p


def _layer(p: RotoProject, layer: Optional[str]) -> str:
    try:
        return p.check_layer(layer)
    except KeyError:
        raise HTTPException(404, "layer not found")


def _frame(p: RotoProject, t: int) -> int:
    if not 0 <= int(t) < p.n:
        raise HTTPException(404, "frame out of range")
    return int(t)


def _keyframes(p: RotoProject) -> dict:
    return {lid: sorted(p.locked.get(lid, ())) for lid in p.layer_ids()}


def project_dict(p: RotoProject) -> dict:
    state = get_state()
    job = _jobs.get(p.id)
    return {
        "id": p.id, "scene_id": p.meta.get("scene_id"), "name": p.meta.get("name", ""),
        "n_frames": p.n, "fps": float(p.meta["fps"]), "width": p.size[0], "height": p.size[1],
        "start_ms": int(p.meta["start_ms"]), "end_ms": int(p.meta["end_ms"]),
        "has_ai_pass": p.has_ai_pass(), "layers": p.layers, "keyframes": _keyframes(p),
        "suspects": p.suspects(), "settings": p.settings.to_dict(), "models": engine.models_info(state),
        "job": job.to_dict() if job is not None else None, "version": p.version,
        "frame_versions": {str(k): v for k, v in p.frame_versions.items()},
        "exports": p.meta.get("exports"), "can_undo": bool(p.undo_stack), "can_redo": bool(p.redo_stack),
    }


def _edit_response(p: RotoProject, t: int, ms: int, **extra) -> dict:
    return {"ms": ms, "version": p.version, "frame": t, "keyframes": _keyframes(p),
            "suspects": p.suspects(), "can_undo": bool(p.undo_stack), "can_redo": bool(p.redo_stack), **extra}


def _scene_name(scene: dict) -> str:
    stem = Path(scene.get("filepath") or "shot").stem
    idx = scene.get("scene_index")
    return f"{stem}_scene{int(idx):04d}" if idx is not None else stem


# ── project ─────────────────────────────────────────────────────────────────
@router.post("/api/roto/open")
def open_project(body: OpenBody):
    from .segment import _scene_or_404
    from ..indexing.cuts import probe_video
    from ..models.sam2 import cleanup_frames_dir, extract_clip_frames
    scene = _scene_or_404(body.scene_id)
    start = max(0, int(scene["start_ms"] if body.start_ms is None else body.start_ms))
    end = max(start + 50, int(scene["end_ms"] if body.end_ms is None else body.end_ms))
    for meta in find_projects(body.scene_id):
        if int(meta.get("start_ms", -1)) == start and int(meta.get("end_ms", -1)) == end:
            p = get_project(meta["id"])
            if p is not None:
                if p.has_ai_pass():
                    engine.warm_in_background(get_state())
                return project_dict(p)
    try:
        fps = float(probe_video(scene["filepath"]).get("fps") or 24.0)
    except Exception:
        fps = 24.0
    fps = max(1.0, min(120.0, fps))
    max_dim = int(get_state().settings.get("models", {}).get("mask_max_dim", 1080) or 1080)
    frames_dir, frame_paths, _wh = extract_clip_frames(scene["filepath"], start, end, fps=fps,
                                                       max_dim=max(360, min(2160, max_dim)))
    try:
        p = RotoProject.create(frame_paths, fps=fps, name=_scene_name(scene), scene_id=body.scene_id,
                               video_path=scene["filepath"], start_ms=start, end_ms=end, move_frames=True)
    finally:
        cleanup_frames_dir(frames_dir)
    register(p)
    return project_dict(p)


@router.get("/api/roto/scene/{scene_id}")
def scene_projects(scene_id: int):
    return {"projects": [{"id": m["id"], "start_ms": m["start_ms"], "end_ms": m["end_ms"],
                          "has_ai_pass": any(m.get("ai_done", {}).values()), "exported": bool(m.get("exports"))}
                         for m in find_projects(scene_id)]}


@router.get("/api/roto/{pid}")
def get_one(pid: str):
    return project_dict(_project_or_404(pid))


@router.get("/api/roto/{pid}/frame/{idx}")
def frame_jpg(pid: str, idx: int):
    p = _project_or_404(pid)
    t = _frame(p, idx)
    return Response(content=p.jpg_paths[t].read_bytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=3600"})


_matte_cache: "OrderedDict[tuple, bytes]" = OrderedDict()
_matte_lock = threading.Lock()


def preview_matte(p: RotoProject, t: int, layer: Optional[str]) -> np.ndarray:
    with p.lock:
        if layer:
            a = p.masks[layer][t].astype(np.float32) / 255.0
            nb = lambda u: p.masks[layer][u].astype(np.float32) / 255.0   # noqa: E731
        else:
            a = p.combined_u8(t).astype(np.float32) / 255.0
            nb = lambda u: p.combined_u8(u).astype(np.float32) / 255.0    # noqa: E731
        s = p.settings
        prev = nb(t - 1) if s.temporal > 0 and t > 0 else None
        nxt = nb(t + 1) if s.temporal > 0 and t + 1 < p.n else None
    return apply_edges(a, p.rgb(t), s, prev=prev, nxt=nxt,
                       rgb_prev=p.rgb(t - 1) if prev is not None else None,
                       rgb_next=p.rgb(t + 1) if nxt is not None else None)


@router.get("/api/roto/{pid}/matte/{idx}")
def matte_png(pid: str, idx: int, layer: Optional[str] = None, v: Optional[str] = None):
    p = _project_or_404(pid)
    t = _frame(p, idx)
    if layer:
        layer = _layer(p, layer)
    if not p.has_ai_pass() and not any(t in s for s in p.locked.values()):
        raise HTTPException(404, "no matte yet: run the AI pass first")
    nbv = tuple(p.frame_versions.get(u, 0) for u in (t - 1, t, t + 1)) if p.settings.temporal > 0 \
        else (p.frame_versions.get(t, 0),)
    key = (p.id, t, layer, nbv, p.settings.key(), len(p.layers))
    with _matte_lock:
        hit = _matte_cache.get(key)
        if hit is not None:
            _matte_cache.move_to_end(key)
    if hit is None:
        a = preview_matte(p, t, layer)
        buf = io.BytesIO()
        Image.fromarray((np.clip(a, 0, 1) * 255.0).round().astype(np.uint8), "L").save(buf, format="PNG",
                                                                                       compress_level=1)
        hit = buf.getvalue()
        with _matte_lock:
            _matte_cache[key] = hit
            while len(_matte_cache) > 192:
                _matte_cache.popitem(last=False)
    return Response(content=hit, media_type="image/png", headers={"Cache-Control": "no-cache"})


# ── AI pass, propagation, export (background jobs) ─────────────────────────
@router.post("/api/roto/{pid}/detect")
def detect(pid: str, body: DetectBody):
    p = _project_or_404(pid)
    layer = _layer(p, body.layer)
    if body.target == "click" and not body.positive:
        raise HTTPException(400, "click on the character to cut out first")
    state = get_state()

    def run(job: Job):
        job.message = "Cutting out the characters"

        def progress(done, total):
            job.percent = min(99, int(100 * done / max(1, total)))
        masks = engine.ai_pass(p, state, target=body.target, frame=_frame(p, body.frame),
                               positive=body.positive, negative=body.negative,
                               progress=progress, cancel=job.check)
        job.check()
        with p.lock:
            locked = p.locked[layer]
            p.apply(layer, {t: masks[t] for t in range(p.n) if t not in locked}, label="ai", ai=True)
            if p.ai.get(layer) is not None:
                for t in locked:   # the AI copy of locked frames too (eraser)
                    p.ai[layer][t] = np.clip(np.rint(masks[t] * 255), 0, 255).astype(np.uint8)
                    p._dirty.add(("ai:" + layer, t))
            p.meta["shots"] = None
        job.message = "Looking for frames to check"
        p.suspects()   # computed here, not on the next GET (the UI reloads right away)
        engine.warm_in_background(state)
        job.message = "Done"
    return _start_job(p, "detect", run)


@router.post("/api/roto/{pid}/propagate")
def propagate(pid: str, body: PropagateBody):
    p = _project_or_404(pid)
    layer = _layer(p, body.layer)
    t = _frame(p, body.frame)
    state = get_state()

    def run(job: Job):
        job.message = "Following the corrections along the shot"

        def progress(done, total):
            job.percent = min(99, int(100 * done / max(1, total)))
        with p.lock:
            if t not in p.locked[layer]:
                p.set_locked(layer, t, True)   # the frame it starts from is taken as right
        new, method = engine.propagate(p, state, layer, t, body.direction, progress=progress, cancel=job.check)
        job.check()
        with p.lock:
            new = {u: m for u, m in new.items() if u not in p.locked[layer]}
            if new:
                p.apply(layer, new, label="propagate")
        p.suspects()
        job.result = {"frames": len(new), "method": method}
        job.message = f"{len(new)} frames updated ({method})"
    return _start_job(p, "propagate", run)


@router.post("/api/roto/{pid}/export")
def export(pid: str, body: ExportBody):
    p = _project_or_404(pid)
    if not p.has_ai_pass() and not any(p.locked.values()):
        raise HTTPException(400, "nothing to export yet: run the AI pass first")
    state = get_state()

    def run(job: Job):
        from ..paths import user_data_dir
        from ..roto.export import export_project
        job.message = "Writing the files"
        out_cfg = state.settings.get("export", {}).get("output_folder")
        out_dir = Path(out_cfg) if out_cfg else user_data_dir() / "exports"

        def progress(done, total):
            job.percent = min(99, int(100 * done / max(1, total)))
        files = export_project(p, out_dir, body.formats, progress=progress, cancel=job.check)
        added = False
        if body.add_to_selection and p.meta.get("scene_id") is not None:
            added = add_to_selection(int(p.meta["scene_id"]))
        with p.lock:
            p.meta["exports"] = {k: v for k, v in files.items()}
            p.schedule_save(0.0)
        job.result = {"files": files, "added_to_selection": added}
        job.message = "Exported"
    return _start_job(p, "export", run)


def add_to_selection(scene_id: int) -> bool:
    """The shot goes into the user's selection (the dérush list the
    timeline export reads); True when it is there afterwards."""
    state = get_state()
    try:
        with schema.get_conn(state.primary_db) as conn:
            row = conn.execute("SELECT id FROM derush_items WHERE scene_id = ?", (scene_id,)).fetchone()
            if row is None:
                conn.execute("INSERT INTO derush_items (scene_id, favorite) VALUES (?, 0)", (scene_id,))
                conn.commit()
        return True
    except Exception:
        log.exception("roto: could not add scene %s to the selection", scene_id)
        return False


@router.get("/api/roto/{pid}/job")
def job_status(pid: str):
    _project_or_404(pid)
    j = _jobs.get(pid)
    return j.to_dict() if j is not None else None


@router.post("/api/roto/{pid}/cancel")
def cancel(pid: str):
    j = _jobs.get(pid)
    if j is not None and not j.done:
        j.cancel_event.set()
    return {"ok": True}


# ── live corrections ───────────────────────────────────────────────────────
@router.post("/api/roto/{pid}/brush")
def brush(pid: str, body: BrushBody):
    p = _project_or_404(pid)
    _no_job(p)
    layer = _layer(p, body.layer)
    t = _frame(p, body.frame)
    tm = engine.Timer()
    with p.lock:
        ai = p.ai.get(layer)
        new = engine.paint(p.matte(layer, t), None if ai is None else ai[t].astype(np.float32) / 255.0,
                           [s.model_dump() for s in body.strokes])
        p.apply(layer, {t: new}, label="brush", lock=[t])
    ms = tm.ms()
    return _edit_response(p, t, ms)


@router.post("/api/roto/{pid}/click")
def click(pid: str, body: ClickBody):
    p = _project_or_404(pid)
    _no_job(p)
    layer = _layer(p, body.layer)
    t = _frame(p, body.frame)
    w, h = p.size
    if not (0 <= body.x < w and 0 <= body.y < h):
        raise HTTPException(400, "click outside the frame")
    tm = engine.Timer()
    new, model = engine.click(p, get_state(), layer, t, body.x, body.y, body.positive)
    with p.lock:
        p.apply(layer, {t: new}, label="click", lock=[t])
    return _edit_response(p, t, tm.ms(), model=model)


@router.post("/api/roto/{pid}/lock")
def lock(pid: str, body: LockBody):
    p = _project_or_404(pid)
    _no_job(p)
    layer = _layer(p, body.layer)
    p.set_locked(layer, _frame(p, body.frame), body.locked)
    return {"keyframes": _keyframes(p), "suspects": p.suspects(),
            "can_undo": bool(p.undo_stack), "can_redo": bool(p.redo_stack)}


@router.post("/api/roto/{pid}/undo")
def undo(pid: str):
    p = _project_or_404(pid)
    _no_job(p)
    f = p.undo()
    return _edit_response(p, -1 if f is None else f, 0)


@router.post("/api/roto/{pid}/redo")
def redo(pid: str):
    p = _project_or_404(pid)
    _no_job(p)
    f = p.redo()
    return _edit_response(p, -1 if f is None else f, 0)


@router.put("/api/roto/{pid}/settings")
def put_settings(pid: str, body: SettingsBody):
    p = _project_or_404(pid)
    with p.lock:
        p.settings = EdgeSettings.from_dict(body.model_dump())
        p.schedule_save()
    return project_dict(p)


@router.post("/api/roto/{pid}/layers")
def add_layer(pid: str, body: LayerBody):
    p = _project_or_404(pid)
    _no_job(p)
    p.add_layer(body.name)
    return project_dict(p)


@router.delete("/api/roto/{pid}/layers/{layer}")
def delete_layer(pid: str, layer: str):
    p = _project_or_404(pid)
    _no_job(p)
    _layer(p, layer)
    try:
        p.remove_layer(layer)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return project_dict(p)
