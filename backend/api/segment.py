"""SAM 2 segmentation + alpha-export endpoints.

Workflow exposed to the MiniEditor MaskMode:

  1. ``POST /api/scene/{id}/segment/preview`` — extract the clip frames into a
     tempdir, run the SAM 2 image predictor on the reference frame with the
     user's prompts, return a session id + PNG urls.
  2. ``GET  /api/scene/segment/{session}/frame/{idx}`` — JPG of frame N.
  3. ``GET  /api/scene/segment/{session}/mask/{idx}`` — RGBA PNG of mask N
     (semi-transparent green) for the review slider.
  4. ``POST /api/scene/{id}/segment/track`` — propagate the mask across all
     extracted frames using the SAM 2 video predictor.
  5. ``POST /api/scene/{id}/segment/export`` — re-decode the source on the
     session's frame timeline, merge each frame with its matte, encode
     ProRes 4444 alpha .mov (or VP9 alpha .webm).
  6. ``DELETE /api/scene/segment/{session}`` — clean the tempdir.

Sessions live in process memory (cleared by the Danger Zone reinstall). For a
single-user desktop app this is fine; if you ever multi-tenant, swap to a real
KV store with TTL.
"""
from __future__ import annotations

import io
import importlib.util
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..db import queries, schema
from ..indexing.cuts import probe_video
from ..models.sam2 import cleanup_frames_dir, extract_clip_frames
from ..paths import user_data_dir
from ..state import get_state

log = logging.getLogger(__name__)
router = APIRouter()


# ── pydantic bodies ─────────────────────────────────────────────────────────
class Prompt(BaseModel):
    positive: List[Tuple[int, int]] = Field(default_factory=list)
    negative: List[Tuple[int, int]] = Field(default_factory=list)
    box: Optional[Tuple[int, int, int, int]] = None
    reference_frame_offset: Optional[int] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    # Optional per-request override. None means "use the engine set in
    # Settings → Models → Mask engine". Valid values: "birefnet" | "sam2".
    engine: Optional[str] = None


class TrackRequest(Prompt):
    session_id: str


class ExportAlphaRequest(BaseModel):
    session_id: str
    codec: str = "prores_4444_alpha"  # or "vp9_alpha"
    output_path: Optional[str] = None


# ── server-side session (in-memory) ─────────────────────────────────────────
@dataclass
class SegmentSession:
    """One MaskMode session: bound to a single scene, holds the extracted frames
    on disk and the latest masks in memory.

    Multiple preview calls reuse the same tempdir so we don't re-extract on
    every click; the track call reuses it again. Cleanup happens on DELETE or
    when the user reinstalls everything from the Danger Zone."""
    session_id: str
    scene_id: int
    video_path: str
    start_ms: int
    end_ms: int
    fps: float
    frames_dir: Path
    frame_paths: List[Path]
    frame_size: Tuple[int, int]                        # (w, h)
    masks: Optional[np.ndarray] = None                 # (T, H, W) bool or float32 alpha, set by /track
    reference_frame_offset: int = 0
    reference_mask: Optional[np.ndarray] = None        # (H, W) bool or float32 alpha, preview only
    last_prompt: Optional[Prompt] = None
    created_at: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _resolve_engine(body_engine: Optional[str]) -> str:
    """Per-request override > saved setting > hard fallback to BiRefNet."""
    if body_engine in ("birefnet", "sam2", "matanyone"):
        engine = body_engine
    else:
        engine = get_state().settings.get("models", {}).get("mask_engine", "birefnet")
    if engine == "sam2" and importlib.util.find_spec("sam2") is None:
        if body_engine == "sam2":
            raise HTTPException(
                400,
                "Manual SAM 2 mode is not installed for this backend. Use Auto.",
            )
        return "birefnet"
    if engine == "matanyone" and importlib.util.find_spec("matanyone") is None:
        if body_engine == "matanyone":
            raise HTTPException(
                400,
                "Temporal mode is not available in this build. Use Auto or Manual.",
            )
        return "birefnet"
    return engine if engine in ("birefnet", "sam2", "matanyone") else "birefnet"


def _scene_or_404(scene_id: int) -> dict:
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        s = queries.get_scene(conn, scene_id)
    if not s:
        raise HTTPException(404, "scene not found")
    if not Path(s["filepath"]).exists():
        raise HTTPException(410, "source file missing")
    return s


def _get_session_or_404(session_id: str) -> SegmentSession:
    sess = get_state().get_sam2_session(session_id)
    if sess is None:
        raise HTTPException(404, "segment session not found")
    return sess


def _mask_to_png_overlay(mask: np.ndarray) -> bytes:
    """Render a mask as a "spotlight" matte PNG: the selected area is fully
    transparent (the underlying video shines through), the rest is darkened.

    Accepts either a bool mask (SAM 2 — hard boundary) or a float32 0-1 soft
    matte (BiRefNet — continuous alpha for hair / motion blur / partial
    occlusion). For soft mattes the darkening alpha follows the inverse of
    the mask, so the eye reads the falloff exactly the way it'll appear in
    the exported ProRes 4444 alpha — no thresholding lies."""
    h, w = mask.shape
    if mask.dtype == bool:
        m = mask.astype(np.float32)
        is_binary = True
    else:
        m = np.clip(mask.astype(np.float32), 0.0, 1.0)
        # Soft mask becomes "effectively binary" if the values are bimodal
        # near 0/1 — in that case the magenta outline still helps readability.
        is_binary = bool(((m < 0.05) | (m > 0.95)).mean() > 0.97)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # Outside-mask alpha = (1 - m) * 178 ≈ 70% opaque black at m=0, fully
    # transparent at m=1, smooth gradient in between for soft mattes.
    rgba[..., 3] = ((1.0 - m) * 178.0).astype(np.uint8)

    # Magenta 2-px outline only for hard-boundary masks (binary, or soft mattes
    # that thresholded effectively binary). For genuinely soft mattes the
    # gradient itself reads as the boundary.
    if is_binary:
        bin_m = m > 0.5
        if bin_m.any():
            edge = np.zeros_like(bin_m)
            edge[1:, :]  |= bin_m[1:, :]  & ~bin_m[:-1, :]
            edge[:-1, :] |= bin_m[:-1, :] & ~bin_m[1:, :]
            edge[:, 1:]  |= bin_m[:, 1:]  & ~bin_m[:, :-1]
            edge[:, :-1] |= bin_m[:, :-1] & ~bin_m[:, 1:]
            rgba[edge] = (236, 72, 153, 255)  # tailwind pink-500

    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _clean_soft_alpha_for_frame(mask: np.ndarray, frame_path: Path, models_cfg: dict) -> np.ndarray:
    """Apply colour-key cleanup to soft alpha mattes near their boundary.

    BiRefNet/MatAnyone can include bits of the original background in tight
    hair gaps. This cleanup removes only boundary pixels whose source colour
    still looks like the frame background.
    """
    if not np.issubdtype(mask.dtype, np.floating):
        return mask
    alpha = np.clip(mask.astype(np.float32, copy=False), 0.0, 1.0)
    suppress_enabled = bool(models_cfg.get("mask_soft_bg_suppress_enabled", False))
    needs_refine = (
        int(models_cfg.get("mask_soft_shrink_px", 0)) > 0
        or float(models_cfg.get("mask_soft_alpha_black", 0.0)) > 0.0
        or float(models_cfg.get("mask_soft_alpha_white", 1.0)) < 1.0
    )
    if not suppress_enabled and not needs_refine:
        return alpha

    from ..models.mask_postprocess import refine_soft_alpha, suppress_bg_in_alpha_band

    with Image.open(frame_path) as im:
        rgb = np.asarray(im.convert("RGB"))
    if suppress_enabled:
        alpha = suppress_bg_in_alpha_band(
            rgb,
            alpha,
            color_dist_threshold=float(models_cfg.get("mask_soft_bg_suppress_threshold", 25.0)),
            edge_band_px=int(models_cfg.get("mask_soft_bg_suppress_edge_px", 16)),
        )
    return refine_soft_alpha(
        alpha,
        shrink_px=int(models_cfg.get("mask_soft_shrink_px", 0)),
        black_point=float(models_cfg.get("mask_soft_alpha_black", 0.0)),
        white_point=float(models_cfg.get("mask_soft_alpha_white", 1.0)),
    )


# ── endpoints: preview ─────────────────────────────────────────────────────
@router.post("/api/scene/{scene_id}/segment/preview")
def preview(scene_id: int, body: Prompt):
    """Click → mask on the reference frame. Lazy-loads SAM 2 and lazy-extracts
    the clip's frames; subsequent preview calls on the same scene reuse the
    same session (no re-extraction).

    If no prompts are passed, the call only initialises the session and
    extracts the reference frame — the client uses this on mount of MaskMode
    so the user sees the real frame (with the right resolution) BEFORE
    placing any click. SAM 2 itself is not invoked in that case."""
    state = get_state()
    scene = _scene_or_404(scene_id)
    clip_start_ms = scene["start_ms"] if body.start_ms is None else int(body.start_ms)
    clip_end_ms = scene["end_ms"] if body.end_ms is None else int(body.end_ms)
    clip_start_ms = max(0, clip_start_ms)
    clip_end_ms = max(clip_start_ms + 50, clip_end_ms)

    # Reuse the most recent session for this scene if there is one, so the
    # frames_dir survives across clicks. Sessions are scoped to the MiniEditor
    # trim window too, so a 4-second user cut does not track/export the full
    # detected scene.
    existing = None
    with state._sam2_sessions_lock:  # safe: we don't mutate, just iterate
        for sess in state._sam2_sessions.values():
            if (
                sess.scene_id == scene_id
                and sess.start_ms == clip_start_ms
                and sess.end_ms == clip_end_ms
            ):
                existing = sess
                break

    if existing is not None:
        session = existing
    else:
        # Fresh extract. Probe the source's real frame rate — the mask session
        # and the alpha export share this value, so mask N and export frame N
        # stay locked to the same source frame. (A hard-coded 24 here used to
        # make the matte drift progressively on 23.976/29.97/60 fps sources.)
        try:
            fps = float(probe_video(scene["filepath"]).get("fps") or 24.0)
        except Exception:
            log.warning("fps probe failed for %s — assuming 24", scene["filepath"])
            fps = 24.0
        fps = max(1.0, min(120.0, fps))
        mask_max_dim = int(state.settings.get("models", {}).get("mask_max_dim", 1080) or 1080)
        mask_max_dim = max(360, min(2160, mask_max_dim))
        frames_dir, frame_paths, (w, h) = extract_clip_frames(
            scene["filepath"], clip_start_ms, clip_end_ms,
            fps=fps, max_dim=mask_max_dim,
        )
        session = SegmentSession(
            session_id=str(uuid.uuid4()),
            scene_id=scene_id,
            video_path=scene["filepath"],
            start_ms=clip_start_ms,
            end_ms=clip_end_ms,
            fps=fps,
            frames_dir=frames_dir,
            frame_paths=frame_paths,
            frame_size=(w, h),
            reference_frame_offset=0,
        )
        state.register_sam2_session(session.session_id, session)

    # Reference frame offset: caller can override (e.g. when re-tracking from a
    # frame the user picked in the review slider).
    ref_idx = body.reference_frame_offset
    if ref_idx is None:
        ref_idx = session.reference_frame_offset
    ref_idx = max(0, min(len(session.frame_paths) - 1, int(ref_idx)))
    session.reference_frame_offset = ref_idx

    engine = _resolve_engine(body.engine)
    has_prompts = bool(body.positive) or body.box is not None

    # BiRefNet runs without prompts — we kick it off as soon as the session is
    # ready (the client doesn't need to wait for a click). SAM 2 only runs
    # when the user has actually placed at least one prompt.
    # MatAnyone's preview = BiRefNet on the reference frame. The MatAnyone
    # value-add (temporal propagation) shows up only at /track time, so the
    # preview UI is identical for "birefnet" and "matanyone" engines.
    auto_engines = ("birefnet", "matanyone")
    run_mask = (engine in auto_engines) or has_prompts

    if run_mask:
        with Image.open(session.frame_paths[ref_idx]) as im:
            ref_rgb = im.convert("RGB").copy()
        with session.lock:
            if engine in auto_engines:
                birefnet = state.get_birefnet()
                mask = birefnet.mask_image(ref_rgb)
            else:
                sam2 = state.get_sam2()
                mask = sam2.preview_mask(
                    ref_rgb,
                    positive_points=body.positive,
                    negative_points=body.negative,
                    box=body.box,
                )
            session.reference_mask = mask
            session.last_prompt = body

    w, h = session.frame_size
    return {
        "ok": True,
        "session_id": session.session_id,
        "scene_id": scene_id,
        "n_frames": len(session.frame_paths),
        "frame_w": w,
        "frame_h": h,
        "reference_frame_offset": ref_idx,
        "engine": engine,
        "has_mask": run_mask,
        "frame_url": f"/api/scene/segment/{session.session_id}/frame/{ref_idx}",
        "mask_url": f"/api/scene/segment/{session.session_id}/mask_preview" if run_mask else None,
    }


@router.get("/api/scene/segment/{session_id}/frame/{idx}")
def session_frame(session_id: str, idx: int):
    """PNG of the extracted frame N. Used by the MiniEditor's review slider so
    the canvas can show the original frame under the mask overlay."""
    sess = _get_session_or_404(session_id)
    if idx < 0 or idx >= len(sess.frame_paths):
        raise HTTPException(404, "frame out of range")
    data = sess.frame_paths[idx].read_bytes()
    return Response(content=data, media_type="image/png",
                    headers={"Cache-Control": "no-cache"})


@router.get("/api/scene/segment/{session_id}/mask_preview")
def session_mask_preview(session_id: str):
    """Semi-transparent green overlay PNG of the reference-frame mask. Drawn on
    top of the video / frame canvas in MaskMode."""
    sess = _get_session_or_404(session_id)
    if sess.reference_mask is None:
        raise HTTPException(404, "no preview mask yet — POST /segment/preview first")
    models_cfg = get_state().settings.get("models", {})
    mask = _clean_soft_alpha_for_frame(
        sess.reference_mask,
        sess.frame_paths[sess.reference_frame_offset],
        models_cfg,
    )
    return Response(content=_mask_to_png_overlay(mask),
                    media_type="image/png",
                    headers={"Cache-Control": "no-cache"})


@router.get("/api/scene/segment/{session_id}/mask/{idx}")
def session_mask_at(session_id: str, idx: int):
    """RGBA overlay PNG of mask N. Used by the review slider after /track has
    populated the full mask stack."""
    sess = _get_session_or_404(session_id)
    if sess.masks is None:
        raise HTTPException(404, "tracking not run yet — POST /segment/track first")
    if idx < 0 or idx >= sess.masks.shape[0]:
        raise HTTPException(404, "frame out of range")
    models_cfg = get_state().settings.get("models", {})
    mask = _clean_soft_alpha_for_frame(sess.masks[idx], sess.frame_paths[idx], models_cfg)
    return Response(content=_mask_to_png_overlay(mask),
                    media_type="image/png",
                    headers={"Cache-Control": "no-cache"})


# ── endpoints: track ───────────────────────────────────────────────────────
@router.post("/api/scene/{scene_id}/segment/track")
def track(scene_id: int, body: TrackRequest):
    """Build per-frame masks for the whole clip. Engine-dependent:

    - **BiRefNet** (Auto): runs the model independently on every extracted
      frame. ~50 ms/frame on a modern GPU, no temporal model — the per-frame
      output is stable enough on stylized content.
    - **SAM 2** (Manual): single propagation pass seeded by the prompts at
      the reference frame.

    Synchronous; progress events go through ``/ws/progress`` so the UI can
    show a bar in either path."""
    sess = _get_session_or_404(body.session_id)
    if sess.scene_id != scene_id:
        raise HTTPException(400, "session is for a different scene")

    state = get_state()
    engine = _resolve_engine(body.engine)
    n_frames = len(sess.frame_paths)
    progress_state = {"last_pct": -1}

    def progress(seen: int, total: int):
        pct = int((seen / max(1, total)) * 100)
        if pct != progress_state["last_pct"]:
            progress_state["last_pct"] = pct
            state.publish({
                "type": "sam2_track",
                "session_id": sess.session_id,
                "percent": pct,
                "frame_idx": seen,
                "total": total,
                "engine": engine,
            })

    with sess.lock:
        if engine == "birefnet":
            birefnet = state.get_birefnet()
            w, h = sess.frame_size
            masks = np.zeros((n_frames, h, w), dtype=np.float32)
            for i, fp in enumerate(sess.frame_paths):
                with Image.open(fp) as im:
                    masks[i] = birefnet.mask_image(im.convert("RGB"))
                progress(i + 1, n_frames)
            # Per-frame inference re-decides the silhouette on every frame —
            # smooth the stack so static regions stop flickering. Hard cuts /
            # teleport motion reset the EMA history inside the helper.
            models_cfg = state.settings.get("models", {})
            if bool(models_cfg.get("mask_temporal_smooth_enabled", True)) and n_frames >= 3:
                from ..models.mask_postprocess import temporal_smooth_alpha
                masks = temporal_smooth_alpha(
                    masks,
                    strength=float(models_cfg.get("mask_temporal_smooth_strength", 0.5)),
                )
            sess.masks = masks
        elif engine == "matanyone":
            # Two-stage: BiRefNet for the chosen reference-frame seed (good
            # single-frame silhouette on anime), MatAnyone for temporal
            # propagation. MatAnyone expects the seed mask to belong to the
            # first frame it receives, so run one chronological pass forward
            # and, when needed, one reversed pass backward instead of shuffling
            # the timeline into a non-temporal order.
            ref_idx = max(0, min(n_frames - 1, int(sess.reference_frame_offset)))
            birefnet = state.get_birefnet()
            with Image.open(sess.frame_paths[ref_idx]) as im:
                seed_mask = birefnet.mask_image(im.convert("RGB"))
            # Publish ~5% progress for the seed step so the bar moves visibly
            # before MatAnyone loads (loading takes ~5-10 s the first time).
            state.publish({
                "type": "sam2_track", "session_id": sess.session_id,
                "percent": 5, "frame_idx": 0, "total": n_frames,
                "engine": engine,
            })
            matanyone = state.get_matanyone()

            filled: set[int] = set()

            def progress_for(indices: list[int]):
                def _cb(seen: int, _total: int):
                    filled.update(indices[:seen])
                    progress(len(filled), n_frames)
                return _cb

            forward_indices = list(range(ref_idx, n_frames))
            forward = matanyone.propagate(
                [sess.frame_paths[i] for i in forward_indices],
                seed_mask=seed_mask,
                progress_cb=progress_for(forward_indices),
            )
            w, h = sess.frame_size
            masks = np.zeros((n_frames, h, w), dtype=forward.dtype)
            for pos, orig_idx in enumerate(forward_indices):
                masks[orig_idx] = forward[pos]

            if ref_idx > 0:
                backward_indices = list(range(ref_idx, -1, -1))
                backward = matanyone.propagate(
                    [sess.frame_paths[i] for i in backward_indices],
                    seed_mask=seed_mask,
                    progress_cb=progress_for(backward_indices),
                )
                for pos, orig_idx in enumerate(backward_indices):
                    masks[orig_idx] = backward[pos]

            sess.masks = masks
        else:
            if not body.positive and body.box is None:
                raise HTTPException(400, "SAM 2 needs at least one positive point or a box")
            sam2 = state.get_sam2()
            masks = sam2.predict_video_masks(
                positive_points=body.positive,
                negative_points=body.negative,
                box=body.box,
                reference_frame_offset=(
                    body.reference_frame_offset
                    if body.reference_frame_offset is not None
                    else sess.reference_frame_offset
                ),
                progress_cb=progress,
                frames_dir=sess.frames_dir,
                frame_paths=sess.frame_paths,
                frame_size=sess.frame_size,
            )
            sess.masks = masks

        sess.last_prompt = Prompt(
            positive=body.positive, negative=body.negative,
            box=body.box, reference_frame_offset=body.reference_frame_offset,
            start_ms=sess.start_ms, end_ms=sess.end_ms,
            engine=engine,
        )

    w, h = sess.frame_size
    state.publish({"type": "sam2_track", "session_id": sess.session_id,
                   "percent": 100, "frame_idx": n_frames, "total": n_frames,
                   "engine": engine})
    return {
        "ok": True,
        "session_id": sess.session_id,
        "n_frames": n_frames,
        "frame_w": w,
        "frame_h": h,
        "engine": engine,
    }


# ── endpoints: export with alpha ───────────────────────────────────────────
def _sanitize(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "scene"


@router.post("/api/scene/{scene_id}/segment/export")
def export_alpha(scene_id: int, body: ExportAlphaRequest):
    """Export the clip with the propagated alpha channel as ProRes 4444 .mov
    (or VP9 alpha .webm). Re-decodes the source at full resolution on the same
    frame timeline as the mask session, pairs frame i with mask i, and encodes
    the RGBA sequence — see ``export_scene_with_alpha``."""
    sess = _get_session_or_404(body.session_id)
    if sess.scene_id != scene_id:
        raise HTTPException(400, "session is for a different scene")
    if sess.masks is None:
        raise HTTPException(400, "no masks yet — POST /segment/track first")

    state = get_state()
    settings = state.settings["export"]
    output_dir = Path(settings["output_folder"]) if settings["output_folder"] else (user_data_dir() / "exports")
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".mov" if body.codec == "prores_4444_alpha" else ".webm"
    name = settings["naming_template"].format(
        anime=_sanitize(Path(sess.video_path).stem),
        episode=_sanitize(Path(sess.video_path).stem),
        scene_id=sess.scene_id, tags="alpha",
    )
    target = body.output_path or str(output_dir / f"{_sanitize(name)}_alpha{suffix}")

    # Per-frame hard-mask cleanup: shrink the silhouette, then optionally drop
    # boundary-band FG pixels whose source RGB matches the BG reference.
    #
    # Soft mattes use colour-key cleanup instead of erosion. That removes bits
    # of background caught between hair spikes while keeping fractional alpha
    # on the actual strand edges.
    models_cfg = state.settings.get("models", {})
    shrink_px = int(models_cfg.get("mask_shrink_px", 0))
    suppress_enabled = bool(models_cfg.get("mask_bg_suppress_enabled", False))
    suppress_threshold = float(models_cfg.get("mask_bg_suppress_threshold", 25.0))

    if np.issubdtype(sess.masks.dtype, np.floating):
        export_masks = np.empty_like(sess.masks, dtype=np.float32)
        for i, m in enumerate(sess.masks):
            export_masks[i] = _clean_soft_alpha_for_frame(m, sess.frame_paths[i], models_cfg)
    elif shrink_px > 0 or suppress_enabled:
        from ..models.mask_postprocess import shrink_mask, suppress_bg_in_edge_band
        processed = np.empty_like(sess.masks)
        for i, m in enumerate(sess.masks):
            if shrink_px > 0:
                m = shrink_mask(m, shrink_px)
            if suppress_enabled:
                with Image.open(sess.frame_paths[i]) as im:
                    rgb = np.asarray(im.convert("RGB"))
                m = suppress_bg_in_edge_band(
                    rgb, m, color_dist_threshold=suppress_threshold,
                )
            processed[i] = m
        export_masks = processed
    else:
        export_masks = sess.masks

    from ..export.ffmpeg import export_scene_with_alpha
    out = export_scene_with_alpha(
        src_path=sess.video_path,
        start_ms=sess.start_ms,
        end_ms=sess.end_ms,
        masks=export_masks,
        output_path=target,
        codec=body.codec,
        fps=sess.fps,
        decontaminate_rgb=bool(models_cfg.get("mask_rgb_decontaminate_enabled", False)),
        edge_refine=bool(models_cfg.get("mask_edge_refine_enabled", True)),
        bg_aware_cleanup=bool(models_cfg.get("mask_bg_aware_cleanup_enabled", True)),
    )
    return {"ok": True, "output": out}


# ── endpoints: cleanup ─────────────────────────────────────────────────────
@router.delete("/api/scene/segment/{session_id}")
def drop(session_id: str):
    sess = get_state().drop_sam2_session(session_id)
    if sess is None:
        return {"ok": True, "dropped": False}
    cleanup_frames_dir(sess.frames_dir)
    return {"ok": True, "dropped": True}
