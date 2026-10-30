"""Models behind the roto editor.

- :func:`ai_pass`   the first pass, the app's best Auto pipeline (Stable v5:
  BiRefNet's soft matte, SAM 3 picks the characters, SAM 2.1 tracks, ink
  line, temporal guard, held objects) — ``backend/api/segment._track_auto``.
- :func:`paint`     a brush stroke: add / remove / erase (= put the AI pass back).
- :func:`click`     a click: the part under it (SAM 2.1 image predictor,
  image embedding cached per frame) added (positive) or removed (negative).
- :func:`propagate` video tracking from the corrected frame(s): SAM 2.1
  video predictor, its memory initialised with the user's locked frames;
  without it (CPU), the corrected matte carried by optical flow. Either
  way the result is merged with the AI pass of each frame
  (:func:`merge_track`), and locked frames are never written.

Long work takes ``progress(done, total)`` and ``cancel()`` callbacks;
``cancel`` raises :class:`Cancelled` (a BaseException, so the models'
``except Exception`` guards do not swallow it).
"""
from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

NEAR_PX = 5            # the AI matte is kept within this distance of the tracked object (px)
TRACK_ERODE_PX = 1     # the tracked object fills where the AI matte has nothing, shrunk by this
THIN_PX = 3            # AI matte parts thinner than ~2x this (strands, tips) survive a propagation
TRACK_LO, TRACK_HI = 0.35, 0.65   # SAM 2 probabilities ramped to 0..1 between these (no faint veil)
CLICK_MAX_FRAC = 0.45  # a click region bigger than this share of the frame is refused
WINDOW = 48            # SAM 2 propagation window (bounded memory on long shots)

ProgressFn = Callable[[int, int], None]
CancelFn = Callable[[], None]


class Cancelled(BaseException):
    """The user pressed Cancel."""


def _noop(*_a, **_k):
    return None


def sam2_installed() -> bool:
    return importlib.util.find_spec("sam2") is not None


# ── machine / models ───────────────────────────────────────────────────────
def editor_tracker(state) -> Optional[str]:
    """SAM 2.1 variant the editor uses for clicks and propagation: large on
    NVIDIA (the best tracker; the editor works on one shot at a time), the
    tier's pick on a Mac, tiny on CPU for clicks (propagation there uses
    optical flow). None when SAM 2 is not installed."""
    if not sam2_installed():
        return None
    backend = getattr(state.device, "backend", "cpu")
    if backend == "cuda":
        from ..models.sam3_concept import total_vram_gb
        return "large" if total_vram_gb() >= 10 or total_vram_gb() == 0 else "base_plus"
    from ..models.auto_roto import mask_models
    return mask_models(state).get("tracker") or "tiny"


def models_info(state) -> dict:
    from ..models.auto_roto import MASK_MODEL_INFO, mask_models
    mm = mask_models(state)
    backend = getattr(state.device, "backend", "cpu")
    sam3 = False
    wants_sam3 = mm.get("selector") == "sam3"
    if wants_sam3:
        try:
            from ..models.sam3_concept import weights_accessible
            sam3 = weights_accessible(state.settings.get("models", {}).get("hf_token") or None)
        except Exception:
            sam3 = False
    trk = editor_tracker(state)
    trk_label = MASK_MODEL_INFO.get(trk or "", {}).get("label") if trk else None
    det = MASK_MODEL_INFO.get(mm.get("detector") or "general", {}).get("label", "BiRefNet")
    return {
        "device": backend,
        "detector": det,
        "characters": "SAM 3" if sam3 else None,
        "tracker": trk_label if (trk and backend != "cpu") else None,
        "clicks": trk_label or f"{det} (character under the click)",
        # only where this machine's tier uses SAM 3 and its gated weights are
        # not reachable (no Hugging Face access): the clicks then use SAM 2.1
        "sam3_missing": wants_sam3 and not sam3,
        "slow_machine": backend == "cpu",
    }


# ── AI pass ────────────────────────────────────────────────────────────────
def ai_pass(project, state, *, target: str = "all", frame: int = 0,
            positive: Sequence = (), negative: Sequence = (),
            progress: ProgressFn = _noop, cancel: CancelFn = _noop) -> np.ndarray:
    """``(T, H, W)`` float32 mattes of the shot from the app's Auto pipeline
    (Stable v5 where SAM 3 runs). ``target``: "all", "main" or "click"
    (the character(s) under ``positive`` on ``frame``)."""
    from ..api.segment import SegmentSession, TrackRequest, _track_auto, _sam2_available
    from ..models.auto_roto import mask_models

    def prog(seen: int, total: int) -> None:
        cancel()
        progress(seen, total)

    w, h = project.size
    sess = SegmentSession(
        session_id=f"roto-{project.id}", scene_id=int(project.meta.get("scene_id") or -1),
        video_path=str(project.meta.get("video_path") or ""), start_ms=int(project.meta["start_ms"]),
        end_ms=int(project.meta["end_ms"]), fps=float(project.meta["fps"]),
        frames_dir=project.dir / "frames", frame_paths=list(project.frame_paths), frame_size=(w, h),
    )
    clicks = None
    tracked_click = target == "click" and bool(mask_models(state).get("tracker")) and _sam2_available()
    if target == "click" and positive and tracked_click:
        clicks = {int(frame): ([tuple(p) for p in positive], [tuple(p) for p in negative])}
    body = TrackRequest(session_id=sess.session_id,
                        auto_target=target if target in ("main", "all") else "all")
    masks = _track_auto(sess, body, prog, clicks=clicks)
    masks = np.clip(np.asarray(masks, np.float32), 0.0, 1.0)
    if target == "click" and positive and clicks is None:
        masks = _follow_click(masks, frame, positive, negative, project)
    return masks


def _follow_click(masks: np.ndarray, frame: int, positive, negative, project) -> np.ndarray:
    """No tracker on this machine: keep the detected pieces under the click
    on ``frame``, then on each neighbour the pieces that overlap the
    previous frame's choice carried by optical flow."""
    from ..models.auto_roto import _flow_warp, clicked_targets, match_targets
    T = len(masks)
    out = np.zeros_like(masks)
    sel = clicked_targets(masks[frame], positive, negative)
    out[frame] = masks[frame] * sel
    for order in (range(frame + 1, T), range(frame - 1, -1, -1)):
        prev = frame
        for t in order:
            ref = _flow_warp(project.rgb(prev), project.rgb(t), out[prev]) > 0.5
            out[t] = masks[t] * match_targets(masks[t], ref)
            prev = t
    return out


# ── brush ──────────────────────────────────────────────────────────────────
def stroke_cover(shape, strokes: Sequence[dict]) -> List[tuple]:
    """``[(mode, coverage float32)]`` of brush strokes (frame pixels)."""
    import cv2
    h, w = shape
    out = []
    for st in strokes:
        pts = [(int(round(float(x))), int(round(float(y)))) for x, y in (st.get("points") or [])]
        if not pts:
            continue
        r = max(1, int(round(float(st.get("radius", 12)))))
        layer = np.zeros((h, w), np.uint8)
        for a, b in zip(pts, pts[1:] or pts):
            cv2.line(layer, a, b, 255, thickness=2 * r, lineType=cv2.LINE_AA)
        for p in (pts[0], pts[-1]):
            cv2.circle(layer, p, r, 255, -1, lineType=cv2.LINE_AA)
        out.append((str(st.get("mode", "add")), layer.astype(np.float32) / 255.0))
    return out


SNAP_RADIUS = 3        # the edge of a stroke snaps to a drawn edge this close (px)
SNAP_EPS = 2e-3       # guided filter regularisation (higher: smoother, less snapping)


def snap_to_edges(new: np.ndarray, before: np.ndarray, rgb: np.ndarray, radius: int = SNAP_RADIUS) -> np.ndarray:
    """The outline a brush stroke drew, snapped to the drawing: in a thin
    band around the part of the outline the stroke changed, the alpha is
    replaced by its guided filter on the frame (He et al.), so a stroke that
    ends 1-3 px from a line ends on the line, anti-aliased like the
    drawing, instead of leaving the brush's round, jagged edge."""
    import cv2
    changed = np.abs(new - before) > 0.05
    if not changed.any():
        return new
    edge = (new > 0.02) & (new < 0.98)
    hard = new > 0.5
    k = np.ones((3, 3), np.uint8)
    outline = (cv2.dilate(hard.astype(np.uint8), k) - cv2.erode(hard.astype(np.uint8), k)) > 0
    band = cv2.dilate(((edge | outline) & cv2.dilate(changed.astype(np.uint8), _disk(radius)).astype(bool))
                      .astype(np.uint8), _disk(radius)) > 0
    if not band.any():
        return new
    guide = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    ks = (2 * radius + 1, 2 * radius + 1)
    box = lambda x: cv2.boxFilter(x, -1, ks)   # noqa: E731
    mi, mp = box(guide), box(new)
    a = (box(guide * new) - mi * mp) / (box(guide * guide) - mi * mi + SNAP_EPS)
    b = mp - a * mi
    q = np.clip(box(a) * guide + box(b), 0.0, 1.0)
    return np.where(band, q, new).astype(np.float32)


def paint(cur: np.ndarray, ai: Optional[np.ndarray], strokes: Sequence[dict],
          rgb: Optional[np.ndarray] = None) -> np.ndarray:
    """``cur`` (float matte) with the strokes applied: ``add`` paints the
    character, ``remove`` the background, ``erase`` puts the AI pass back.
    With the frame (``rgb``), the new outline snaps to the drawing
    (:func:`snap_to_edges`)."""
    before = np.clip(np.asarray(cur, np.float32), 0.0, 1.0)
    out = before.copy()
    snap = False
    for mode, cov in stroke_cover(out.shape, strokes):
        if mode == "remove":
            out = out * (1.0 - cov)
            snap = True
        elif mode == "erase":
            base = ai if ai is not None else np.zeros_like(out)
            out = out * (1.0 - cov) + base * cov
        else:
            out = np.maximum(out, cov)
            snap = True
    if snap and rgb is not None and rgb.shape[:2] == out.shape:
        out = snap_to_edges(out, before, rgb)
    return out.astype(np.float32)


# ── click ──────────────────────────────────────────────────────────────────
_pred_lock = threading.Lock()
_pred_key: dict = {"key": None}
_det_cache: dict = {}


def click_region(project, state, t: int, x: float, y: float, positive: bool = True) -> tuple[np.ndarray, str]:
    """Soft region under a click on frame ``t`` and the model that found it.
    SAM proposes 3 nested regions (part, object, whole): a positive click
    takes the most confident one (not over :data:`CLICK_MAX_FRAC` of the
    frame); a negative click takes the smallest confident one — an Alt-click
    on an aura flame must not take the character the aura surrounds."""
    rgb = project.rgb(t)
    h, w = rgb.shape[:2]
    variant = editor_tracker(state)
    if variant is not None:
        import torch
        from ..models.sam2 import _autocast
        sam2 = state.get_sam2(variant=variant)
        sam2._ensure_image_loaded()
        sam2._touch()
        pred = sam2._image_predictor
        with _pred_lock, torch.inference_mode(), _autocast(state.device):
            key = (project.id, int(t), variant, id(pred))
            if _pred_key["key"] != key:
                pred.set_image(rgb)
                _pred_key["key"] = key
            masks, scores, _ = pred.predict(
                point_coords=np.array([[float(x), float(y)]], np.float32),
                point_labels=np.array([1], np.int32), multimask_output=True, return_logits=True)
        scores = np.asarray(scores, np.float32)
        areas = np.array([(m > 0).mean() for m in masks])
        ok = [i for i in range(len(masks)) if 0 < areas[i] <= CLICK_MAX_FRAC]
        best = None
        if ok and positive:
            best = masks[max(ok, key=lambda i: scores[i])]
        elif ok:
            good = [i for i in ok if scores[i] >= 0.6 * float(scores[ok].max())]
            best = masks[min(good or ok, key=lambda i: areas[i])]
        if best is None:
            return np.zeros((h, w), np.float32), f"SAM 2.1 {variant}"
        # SAM's logits come from a 256-px grid: their sigmoid is blocky over
        # wide low-confidence areas (a checkered half-alpha patch where a
        # negative click cut). The region is its mask, feathered over ~1 px.
        import cv2
        from ..models.auto_roto import MASK_MODEL_INFO
        hard = _piece_under(best > 0, x, y)
        region = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 0.8)
        return region.astype(np.float32), MASK_MODEL_INFO.get(variant, {}).get("label", variant)
    # no SAM 2 here: the detected piece under the click
    from PIL import Image
    from ..models.auto_roto import clicked_targets
    key = (project.id, int(t))
    det = _det_cache.get(key)
    if det is None:
        det = state.get_birefnet().mask_image(Image.fromarray(rgb))
        _det_cache.clear()
        _det_cache[key] = det
    sel = clicked_targets(det, [(x, y)])
    return (det * sel).astype(np.float32), "BiRefNet"


def _piece_under(m: np.ndarray, x: float, y: float) -> np.ndarray:
    """The connected piece of ``m`` under the click (the nearest one when
    the click itself fell just outside): SAM's region for a translucent aura
    also has scattered islands inside the character it surrounds."""
    import cv2
    n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    if n <= 2:
        return m
    h, w = m.shape
    xi, yi = int(np.clip(round(x), 0, w - 1)), int(np.clip(round(y), 0, h - 1))
    k = int(lab[yi, xi])
    if k == 0:
        d, idx = cv2.distanceTransformWithLabels((lab == 0).astype(np.uint8), cv2.DIST_L2, 3,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(lab > 0)
        pix = idx[ys, xs]
        hit = np.nonzero(pix == idx[yi, xi])[0]
        k = int(lab[ys[hit[0]], xs[hit[0]]]) if len(hit) else int(np.bincount(lab[lab > 0]).argmax())
    return lab == k


def warm_click_model(state) -> None:
    """Load the click model ahead (in the background, when the editor opens
    and after the AI pass): the first click otherwise waits for the model to
    load (5-50 s) instead of ~0.1 s."""
    try:
        variant = editor_tracker(state)
        if variant is not None and getattr(state.device, "backend", "cpu") != "cpu":
            import torch
            from ..models.sam2 import _autocast
            sam2 = state.get_sam2(variant=variant)
            sam2._ensure_image_loaded()
            sam2._touch()
            # one dummy inference: the first real one otherwise pays the GPU
            # kernels' warm-up (seconds) on the user's first click
            pred = sam2._image_predictor
            with _pred_lock, torch.inference_mode(), _autocast(state.device):
                pred.set_image(np.zeros((576, 1024, 3), np.uint8))
                pred.predict(point_coords=np.array([[512.0, 288.0]], np.float32),
                             point_labels=np.array([1], np.int32), multimask_output=True)
                _pred_key["key"] = None
    except Exception:
        log.exception("roto: could not preload the click model")


def warm_in_background(state) -> None:
    threading.Thread(target=warm_click_model, args=(state,), name="roto-warm", daemon=True).start()


def apply_click(cur: np.ndarray, region: np.ndarray, positive: bool) -> np.ndarray:
    cur = np.clip(np.asarray(cur, np.float32), 0.0, 1.0)
    if positive:
        return np.maximum(cur, region).astype(np.float32)
    return (cur * (1.0 - region)).astype(np.float32)


def click(project, state, layer: str, t: int, x: float, y: float, positive: bool) -> tuple[np.ndarray, str]:
    region, model = click_region(project, state, t, x, y, positive)
    return apply_click(project.matte(layer, t), region, positive), model


# ── propagation ────────────────────────────────────────────────────────────
def _disk(r: int) -> np.ndarray:
    import cv2
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def crisp_track(track: np.ndarray) -> np.ndarray:
    """SAM 2's mask probabilities, sharpened to a ramp of ~1 px around 0.5:
    its low-confidence areas (sigmoid of slightly negative logits, 0.05 to
    0.4 over a whole aura) otherwise became a faint veil of alpha over large
    parts of the frame."""
    t = np.asarray(track, np.float32)
    return np.clip((t - TRACK_LO) / (TRACK_HI - TRACK_LO), 0.0, 1.0).astype(np.float32)


def merge_track(ai: Optional[np.ndarray], track: np.ndarray, near_px: int = NEAR_PX) -> np.ndarray:
    """Propagated matte of a frame: the AI pass's soft edge where it agrees
    with the tracked object (kept within ``near_px`` of it: what the user
    removed on the corrected frame stays removed), the tracked object where
    the AI pass has nothing (what the user added, a character the AI lost)."""
    import cv2
    tr = crisp_track(track)
    trm = (tr > 0.5).astype(np.uint8)
    if ai is None or float(ai.max()) < 0.3 or not trm.any():
        return tr if trm.any() else np.zeros_like(tr)
    ai = np.asarray(ai, np.float32)
    near = cv2.dilate(trm, _disk(near_px)).astype(np.float32)
    near = cv2.GaussianBlur(near, (0, 0), 1.2)
    kept = ai * np.clip(near, 0.0, 1.0)
    fill = cv2.erode(tr, _disk(TRACK_ERODE_PX)) if TRACK_ERODE_PX > 0 else tr
    out = np.maximum(kept, fill)
    # thin parts of the AI matte (a hair strand, the tip of a cigarette)
    # that SAM 2's coarser mask does not follow, still attached to the
    # character: kept whole. Thick pieces (a chunk of scenery the user cut
    # off on the corrected frame) stay out.
    solid = (ai > 0.5).astype(np.uint8)
    if solid.any():
        thick = cv2.morphologyEx(solid, cv2.MORPH_OPEN, _disk(THIN_PX))
        thin = (solid > 0) & (thick == 0) & (near < 0.99)
        if thin.any():
            body = out > 0.5
            n, lab = cv2.connectedComponents((thin | body).astype(np.uint8), connectivity=8)
            hit = np.zeros(n, bool)
            hit[np.unique(lab[body])] = True
            hit[0] = False
            keep = thin & hit[lab]
            out = np.where(keep, np.maximum(out, ai), out)
    return out.astype(np.float32)


def _ranges(n: int, t: int, direction: str, locked: set) -> List[List[int]]:
    """Frame orders to propagate through, each starting at ``t`` and ending
    at the next locked frame (included, as an anchor) or the shot's end."""
    out = []
    if direction in ("forward", "both"):
        seq = [t]
        for u in range(t + 1, n):
            seq.append(u)
            if u in locked:
                break
        if len(seq) > 1:
            out.append(seq)
    if direction in ("backward", "both"):
        seq = [t]
        for u in range(t - 1, -1, -1):
            seq.append(u)
            if u in locked:
                break
        if len(seq) > 1:
            out.append(seq)
    return out


def _sam2_track(sam2, device, jpgs: List[Path], seq: List[int], prompts: Dict[int, np.ndarray],
                tick: Callable[[], None]) -> Dict[int, np.ndarray]:
    """SAM 2.1 video predictor along ``seq`` (any order: a reversed list
    is a backward pass), mask prompts on the frames in ``prompts``; windows
    of :data:`WINDOW` frames, each re-anchored on ``seq[0]`` and the last
    frame of the previous window."""
    import torch
    from ..models.sam2 import SOFT_LOGIT_SCALE, _autocast
    sam2._ensure_video_loaded()
    sam2._touch()
    vp = sam2._video_predictor
    out: Dict[int, np.ndarray] = {}
    carry = None
    for c0 in range(0, len(seq), WINDOW):
        chunk = seq[c0:c0 + WINDOW]
        pre: List[tuple] = []
        if c0 > 0:
            pre.append((seq[0], prompts[seq[0]]))
            if carry is not None and carry[1].any():
                pre.append(carry)
        vseq = [f for f, _m in pre] + chunk
        tmp = Path(tempfile.mkdtemp(prefix="amv_roto_trk_"))
        try:
            for j, f in enumerate(vseq):
                dst = tmp / f"{j:05d}.jpg"
                try:
                    os.link(jpgs[f], dst)
                except OSError:
                    shutil.copyfile(jpgs[f], dst)
            with torch.inference_mode(), _autocast(device):
                st = vp.init_state(video_path=str(tmp), offload_video_to_cpu=True)
                for j, (_f, m) in enumerate(pre):
                    vp.add_new_mask(inference_state=st, frame_idx=j, obj_id=1, mask=m > 0.5)
                for j, f in enumerate(chunk):
                    if f in prompts:
                        vp.add_new_mask(inference_state=st, frame_idx=len(pre) + j, obj_id=1, mask=prompts[f] > 0.5)
                for local, _ids, logits in vp.propagate_in_video(st, start_frame_idx=0):
                    if local < len(pre):
                        continue
                    f = vseq[local]
                    lg = logits[0, 0].float()
                    out[f] = torch.sigmoid(lg * SOFT_LOGIT_SCALE).cpu().numpy().astype(np.float32)
                    sam2._touch()   # not idle: the offload watcher must wait
                    tick()
                vp.reset_state(st)
                del st
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        carry = (chunk[-1], out.get(chunk[-1], np.zeros(1, np.float32)))
    return out


def propagate(project, state, layer: str, t: int, direction: str = "both", *,
              progress: ProgressFn = _noop, cancel: CancelFn = _noop,
              use_sam2: Optional[bool] = None) -> tuple[Dict[int, np.ndarray], str]:
    """New mattes ``{frame: matte}`` of ``layer`` from the corrected frame
    ``t`` (and the other locked frames), never for a locked frame; and the
    method used ("SAM 2.1 large", "optical flow")."""
    with project.lock:
        locked = set(project.locked[layer])
        ai = project.ai.get(layer)
        start = project.matte(layer, t)
        prompts = {u: project.matte(layer, u) for u in locked}
    prompts[t] = start
    orders = _ranges(project.n, t, direction, locked - {t})
    total = sum(len(s) - 1 for s in orders) or 1
    done = [0]

    def tick():
        cancel()
        done[0] += 1
        progress(min(done[0], total), total)

    variant = editor_tracker(state)
    if use_sam2 is None:
        use_sam2 = variant is not None and getattr(state.device, "backend", "cpu") != "cpu"
    out: Dict[int, np.ndarray] = {}
    if use_sam2 and variant is not None:
        sam2 = state.get_sam2(variant=variant)
        from ..models.auto_roto import MASK_MODEL_INFO
        method = MASK_MODEL_INFO.get(variant, {}).get("label", f"SAM 2.1 {variant}")
        for seq in orders:
            got = _sam2_track(sam2, state.device, project.jpg_paths, seq,
                              {u: prompts[u] for u in seq if u in prompts}, tick)
            for u in seq[1:]:
                if u in locked or u == t or u not in got:
                    continue
                out[u] = merge_track(None if ai is None else ai[u].astype(np.float32) / 255.0, got[u])
        return out, method
    from ..models.auto_roto import _flow_warp
    for seq in orders:
        prev_m = start
        for a, b in zip(seq, seq[1:]):
            tick()
            if b in locked:
                break
            carried = _flow_warp(project.rgb(a), project.rgb(b), prev_m)
            m = merge_track(None if ai is None else ai[b].astype(np.float32) / 255.0, carried)
            out[b] = m
            prev_m = m
    return out, "optical flow"


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def ms(self) -> int:
        return int(round((time.perf_counter() - self.t0) * 1000))
