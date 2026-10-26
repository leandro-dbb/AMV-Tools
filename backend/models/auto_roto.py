"""Automatic roto: no click, a tracked character matte for a whole clip.

The 0.4 Auto engine ran an anime matting model on each frame on its own. On
real episodes (see ``docs/reports/chantier-roto.md``) that fails in three
ways: it keeps whatever looks salient on a given frame (a tractor, a house, a
second character that walks in), it flickers when the silhouette is
ambiguous, and it drops characters smeared by motion blur.

This pipeline keeps the anime matte for what it is good at — the outline of a
cel-shaded character — and lets a video tracker with memory (SAM 2.1) decide
*which* pixels belong to the character over time:

1. ``detector`` (ToonOut / ISNet anime / BiRefNet) runs on every frame:
   per-frame alpha ``D_t``.
2. Targets are picked automatically on the clearest frame of each shot:
   ``"main"`` = the biggest, most central character; ``"all"`` = every
   character-sized component.
3. SAM 2 is seeded with that mask (a mask prompt, no click) and propagates
   both ways with its memory. Shots are split at hard cuts and re-seeded.
4. Re-seeding: where the track drifts away from what the detector sees on
   the target (IoU below :data:`RESEED_IOU` for a few frames), the detector
   mask of the target becomes a new keyframe and the shot is re-propagated.
5. Fusion: inside the tracked region (dilated) the detector's soft alpha
   gives the edges; the tracker's core fills what the detector misses on
   motion-blurred frames; everything outside the track is dropped.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

MIN_COMPONENT_FRAC = 0.004   # components smaller than this share of the frame are noise
ALL_MIN_REL = 0.15           # "all": keep components >= 15 % of the biggest one
RESEED_IOU = 0.5             # track vs detector agreement under which we re-seed
RESEED_MIN_RUN = 3           # ... for at least this many consecutive frames
MAX_RESEEDS = 3              # re-propagations per shot
GATE_FRAC = 0.025            # dilation of the track gate, share of the frame diagonal
CORE_FRAC = 0.012            # erosion of the track core, share of the frame diagonal
CORE_MISS = 0.7              # adaptive core: fill from the track only when the detector
                             # sees less than this share of the tracked area (motion blur)
CORE_MIN_SEEN = 0.15         # below this share of the track seen by the detector: hidden
TRACK_SPILL = 1.4            # track area above this x the usual target area = spill
CORE_MODE = "adaptive"       # "always": fill from the track core on every frame


# Models of the Auto engine per memory tier (backend/models/tiers.py ids).
# detector: BiRefNetModel variant; tracker: SAM2Model variant, None = no
# tracking (per-frame detector only — SAM 2 takes 20-40 s per frame on a CPU);
# selector: model that says what a character is before the detector's matte
# is used ("sam3": SAM 3 concept "anime character", gated weights — skipped
# when not accessible), None = none.
# Measured in docs/reports/chantier-roto.md.
MASK_TIERS: dict[str, dict[str, dict[str, Optional[str]]]] = {
    # BiRefNet "general" is the best per-frame detector measured on real
    # episodes (J 0.85 vs 0.75 ISNet anime, 0.72 ToonOut); the tracker size
    # barely changes the Auto result (it only follows the detector's seed),
    # so small cards take the lighter SAM 2.1.
    "nvidia": {
        # "fast": SAM 3 adds ~0.3 s per frame on an RTX 3090
        "fast": {"detector": "general", "tracker": "tiny", "selector": None, "quality": "standard"},
        "8gb": {"detector": "general", "tracker": "tiny", "selector": "sam3", "quality": "max"},
        "12gb": {"detector": "general", "tracker": "base_plus", "selector": "sam3", "quality": "max"},
        "16gb": {"detector": "general", "tracker": "base_plus", "selector": "sam3", "quality": "max"},
        "24gb": {"detector": "general", "tracker": "large", "selector": "sam3", "quality": "max"},
        "max": {"detector": "general", "tracker": "large", "selector": "sam3", "quality": "max"},
    },
    # Not measured on a Mac (unreachable during the 0.5 work): same detector,
    # lighter trackers.
    "mac": {
        "fast": {"detector": "general", "tracker": "tiny", "selector": None},
        "mac8": {"detector": "general", "tracker": "tiny", "selector": None},
        "mac16": {"detector": "general", "tracker": "tiny", "selector": None},
        "mac24": {"detector": "general", "tracker": "base_plus", "selector": None},
        "mac32": {"detector": "general", "tracker": "base_plus", "selector": None},
        "max": {"detector": "general", "tracker": "large", "selector": None},
    },
    # CPU: BiRefNet takes seconds per frame there, ISNet anime ~0.5 s.
    "cpu": {"fast": {"detector": "anime", "tracker": None, "selector": None}},
}
FALLBACK = {"detector": "anime", "tracker": "base_plus", "selector": None}
SELECTOR_PROMPT = "anime character"
TIGHT_FRAC = 0.008           # final matte hugged to SAM 3's outline at this reach (share of the diagonal)
# "Max quality" (0.6, roto 3): hugged at 0.4 % with the opaque tips SAM 3
# under-segments given back, loose fragments dropped; export edges re-solved
# from the local colours with the ink line protected. See
# docs/reports/chantier-roto.md.
MAX_TIGHT_FRAC = 0.004
MAX_TIP_TOUCH = 0.30


def mask_quality(state, value: Optional[str] = None) -> str:
    """"max" or "standard": the setting ("auto" = the tier's pick). "max"
    needs the SAM 3 selector; without it, standard."""
    if value in ("max", "standard"):
        return value
    try:
        return (mask_models(state) or {}).get("quality") or "standard"
    except Exception:
        return "standard"


# Catalogue for Settings > Models (label, download size, license).
MASK_MODEL_INFO: dict[str, dict] = {
    "toonout": {"id": "joelseytre/toonout", "label": "ToonOut (BiRefNet anime)", "size_mb": 885, "license": "mit"},
    "anime": {"id": "skytnt/anime-seg", "label": "ISNet anime-seg", "size_mb": 176, "license": "apache-2.0"},
    "general": {"id": "ZhengPeng7/BiRefNet", "label": "BiRefNet", "size_mb": 445, "license": "mit"},
    "tiny": {"id": "facebook/sam2.1-hiera-tiny", "label": "SAM 2.1 tiny", "size_mb": 156, "license": "apache-2.0"},
    "base_plus": {"id": "facebook/sam2.1-hiera-base-plus", "label": "SAM 2.1 base+", "size_mb": 324, "license": "apache-2.0"},
    "large": {"id": "facebook/sam2.1-hiera-large", "label": "SAM 2.1 large", "size_mb": 898, "license": "apache-2.0"},
    # gated on Hugging Face: the user's own account must accept the license
    "sam3": {"id": "facebook/sam3", "label": "SAM 3 (« anime character »)", "size_mb": 3440,
             "license": "SAM License (Meta)", "gated": True},
}


def describe_mask_models(platform: str, tier: str) -> dict[str, dict]:
    """``{"mask_detector": ..., "mask_tracker": ...}`` rows of a tier for the
    Settings table (no tracker row on CPU)."""
    choice = MASK_TIERS.get(platform, {}).get(tier) or {}
    out = {}
    for role, key in (("mask_selector", "selector"), ("mask_detector", "detector"), ("mask_tracker", "tracker")):
        v = choice.get(key)
        if v in MASK_MODEL_INFO:
            out[role] = dict(MASK_MODEL_INFO[v])
    return out


def mask_models(state) -> dict[str, Optional[str]]:
    """Auto engine models for this machine's effective tier."""
    try:
        sel = state.model_selection()
        table = MASK_TIERS.get(sel["platform"], {})
        return dict(table.get(sel["effective_tier"]) or next(iter(table.values())))
    except Exception:
        return dict(FALLBACK)


def resolve_mask_variant(state, role: str, value: Optional[str]) -> str:
    """Setting value -> model variant. "auto" (the default) and unknown
    values take the tier's pick; a tier without a tracker still gets a SAM 2
    variant for the Manual engine."""
    from .birefnet import VARIANTS as DET
    from .sam2 import VARIANTS as TRK
    known = DET if role == "detector" else TRK
    if value and value != "auto" and value in known:
        return value
    pick = mask_models(state).get(role)
    if pick in known:
        return pick
    return "anime" if role == "detector" else "base_plus"


def _components(m: np.ndarray):
    import cv2
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    return n, lab, stats, cent


def select_targets(alpha: np.ndarray, mode: str = "main") -> np.ndarray:
    """Boolean mask of the target character(s) on one frame's detector alpha.

    ``main``: the component with the best area x centrality score (a big
    character in the middle beats a slightly bigger one cut by the frame
    edge). ``all``: every component at least :data:`ALL_MIN_REL` of the
    biggest. Empty when nothing character-sized is detected."""
    m = alpha > 0.5
    h, w = m.shape
    n, lab, stats, cent = _components(m)
    if n <= 1:
        return np.zeros_like(m)
    areas = stats[1:, 4].astype(np.float64)
    ok = areas >= MIN_COMPONENT_FRAC * h * w
    if not ok.any():
        return np.zeros_like(m)
    ids = np.flatnonzero(ok) + 1
    if mode == "all":
        keep = [i for i in ids if stats[i, 4] >= ALL_MIN_REL * areas.max()]
    else:
        cx, cy = cent[ids, 0] / w - 0.5, cent[ids, 1] / h - 0.5
        central = 1.0 - 0.8 * np.clip(np.hypot(cx, cy) / 0.7071, 0, 1)
        score = stats[ids, 4] * central
        keep = [int(ids[int(np.argmax(score))])]
    return np.isin(lab, keep)


def clicked_targets(alpha: np.ndarray, positive, negative=()) -> np.ndarray:
    """Detector components under the user's positive clicks (minus those
    under negative clicks): a click picks a whole character, not the flat
    colour patch it landed on (what SAM alone does on cel-shading)."""
    m = alpha > 0.5
    if not m.any():
        return np.zeros_like(m)
    h, w = m.shape
    _n, lab, _stats, _ = _components(m)

    def ids(points):
        out = set()
        for x, y in points or ():
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < w and 0 <= yi < h):
                continue
            if lab[yi, xi] > 0:
                out.add(int(lab[yi, xi]))
                continue
            # a click in a hole of the matte (an eye, a gap): nearest component
            r = max(4, int(0.02 * max(h, w)))
            win = lab[max(0, yi - r):yi + r + 1, max(0, xi - r):xi + r + 1]
            if (win > 0).any():
                vals, counts = np.unique(win[win > 0], return_counts=True)
                out.add(int(vals[np.argmax(counts)]))
        return out

    keep = ids(positive) - ids(negative)
    return np.isin(lab, sorted(keep)) if keep else np.zeros_like(m)


def match_targets(alpha: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Detector components of ``alpha`` that overlap the reference mask (the
    tracked character on this frame) — the detector's view of *our* target,
    without the other salient things it found."""
    m = alpha > 0.5
    if not ref.any() or not m.any():
        return np.zeros_like(m)
    n, lab, stats, _ = _components(m)
    hit = np.bincount(lab[ref], minlength=n)
    keep = [i for i in range(1, n) if hit[i] >= 0.2 * stats[i, 4] or hit[i] >= 0.5 * ref.sum()]
    return np.isin(lab, keep) if keep else np.zeros_like(m)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def fuse(detector: np.ndarray, track: np.ndarray, ref_area: float = 0.0) -> np.ndarray:
    """Final alpha of one frame: detector edges inside the tracked region,
    tracker core where the detector missed (motion blur), nothing outside.

    ``ref_area``: usual pixel area of the target in this shot (median of the
    detector's picks). A track much bigger than that is the tracker spilling
    onto smoke / ground / an empty frame — then only the detector counts."""
    import cv2
    h, w = track.shape
    diag = float(np.hypot(h, w))
    t = (track > 0.5).astype(np.uint8)
    if not t.any():
        return np.zeros_like(detector, dtype=np.float32)
    rg = max(1.0, GATE_FRAC * diag)
    rc = max(1.0, CORE_FRAC * diag)
    # distances to the track (outside) / to its border (inside): one pass each
    d_out = cv2.distanceTransform(1 - t, cv2.DIST_L2, 3)
    d_in = cv2.distanceTransform(t, cv2.DIST_L2, 3)
    gate = np.clip((rg - d_out) / (0.5 * rg), 0.0, 1.0)      # 1 near the track, soft fall-off
    core = (d_in > rc).astype(np.float32)
    # detector components touching the track only: another character inside
    # the gate must not leak in
    det = detector.astype(np.float32) * match_targets(detector, t > 0)
    seen = float((det > 0.5).sum())
    if (ref_area > 0 and t.sum() > TRACK_SPILL * ref_area) or seen < CORE_MIN_SEEN * t.sum():
        # spill, or the detector sees (almost) nothing of the character on
        # this frame — hidden by smoke, not blurred: the detector decides
        return np.clip(det * gate, 0, 1).astype(np.float32)
    if CORE_MODE == "adaptive" and (det > 0.5).sum() >= CORE_MISS * t.sum():
        # the detector saw the character on this frame: trust its outline,
        # the track only selects which components to keep
        return np.clip(det * gate, 0, 1).astype(np.float32)
    return np.clip(np.maximum(det * gate, core * np.clip(track, 0, 1)), 0, 1).astype(np.float32)


def split_shots(frames: Sequence[np.ndarray], n: int) -> List[tuple[int, int]]:
    """[start, end) ranges between hard cuts (cuts detected on the frames)."""
    from ..finish.matte import cuts_from_frames
    try:
        cuts = sorted(c for c in cuts_from_frames(frames) if 0 < c < n)
    except Exception:
        cuts = []
    bounds = [0] + cuts + [n]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


def auto_track(
    frame_paths: List[Path],
    detector_alpha: np.ndarray,
    sam2,
    *,
    frames_dir: Path,
    frame_size: tuple[int, int],
    mode: str = "main",
    frames_small: Optional[Sequence[np.ndarray]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    fuse_edges: bool = True,
    extra_keyframes: Optional[List[dict]] = None,
    clicks: Optional[dict] = None,
) -> np.ndarray:
    """Tracked alpha ``(T, H, W)`` float32 for the whole clip, no prompt.

    ``detector_alpha``: per-frame detector output (T, H, W), float or uint8.
    ``frames_small``: downscaled RGB frames for cut detection (optional; no
    cut detection without them). ``sam2``: a :class:`SAM2Model`.
    ``extra_keyframes``: ``{"frame_idx", "mask"}`` user corrections (brush);
    they condition the tracker of their shot like the automatic seed.
    ``mode="click"``: ``clicks`` = ``{frame_idx: (positive, negative)}`` pick
    the detector component(s) to track instead of the automatic choice."""
    T = len(frame_paths)
    det = detector_alpha
    if det.dtype == np.uint8:
        det = det.astype(np.float32) / 255.0
    h, w = det.shape[1:]
    out = np.zeros((T, h, w), dtype=np.float32)
    shots = split_shots(frames_small, T) if frames_small is not None else [(0, T)]
    done = 0
    for a, b in shots:
        t_shot = time.perf_counter()
        sub_paths = frame_paths[a:b]
        if mode == "click":
            # Manual engine: the clicked character(s) on the clicked frame(s)
            # seed the tracker; re-seeding follows the detector's view of
            # the tracked character (no automatic pick on other frames).
            targets = [np.zeros(det.shape[1:], bool) for _ in range(a, b)]
            for fi, (pos, neg) in (clicks or {}).items():
                if a <= int(fi) < b:
                    targets[int(fi) - a] = clicked_targets(det[int(fi)], pos, neg)
            if not any(t.any() for t in targets) and a > 0 and (out[a - 1] > 0.5).any():
                # no click in this shot (a flash taken for a cut, or a real
                # cut): carry the character tracked at the end of the
                # previous shot over
                targets[0] = match_targets(det[a], out[a - 1] > 0.5)
        else:
            targets = [select_targets(det[t], mode) for t in range(a, b)]
        areas = np.array([m.sum() for m in targets], dtype=np.float64)
        user = [{"frame_idx": int(k["frame_idx"]) - a, "mask": np.asarray(k["mask"]) > 0.5}
                for k in (extra_keyframes or []) if a <= int(k["frame_idx"]) < b]
        keyframes = list(user)
        if mode == "click" and areas.max() <= 0:
            # the detector found nothing under the clicks: plain SAM 2 clicks
            for fi, (pos, neg) in (clicks or {}).items():
                if a <= int(fi) < b and pos:
                    keyframes.append({"frame_idx": int(fi) - a, "positive": list(pos), "negative": list(neg or [])})
        if areas.max() <= 0 and not keyframes:
            done += b - a
            continue
        if mode == "click":
            for t in np.flatnonzero(areas > 0):  # every clicked frame
                if all(k["frame_idx"] != int(t) for k in user):
                    keyframes.append({"frame_idx": int(t), "mask": targets[int(t)]})
        elif areas.max() > 0:
            # seed on the frame where the detector sees the most of the target
            seed = int(np.argmax(areas))
            if all(k["frame_idx"] != seed for k in user):
                keyframes.append({"frame_idx": seed, "mask": targets[seed]})
        track = None
        # SAM 2 must never see across a cut: a sub-shot is tracked from a
        # folder of hard links to its own frames
        shot_dir, shot_paths, tmp = (frames_dir, sub_paths, None) if (a == 0 and b == T) else _link_shot(sub_paths)
        rounds = MAX_RESEEDS if (b - a) <= getattr(sam2, "window_frames", 96) else 1
        for _round in range(rounds + 1):
            track = sam2.track_keyframes(keyframes, frames_dir=shot_dir, frame_paths=shot_paths,
                                         frame_size=frame_size, soft=True)
            # where does the track disagree with the detector's view of the target?
            bad_run, new_kf = 0, None
            have = {k["frame_idx"] for k in keyframes}
            for t in range(b - a):
                tm = track[t] > 0.5
                dm = match_targets(det[a + t], tm) if tm.any() else targets[t]
                if not dm.any():
                    bad_run = 0
                    continue
                ok = iou(tm, dm) >= RESEED_IOU
                bad_run = 0 if ok else bad_run + 1
                # (a drift that runs to the end of the shot counts from 2 frames)
                if bad_run >= RESEED_MIN_RUN or (bad_run >= 2 and t == b - a - 1):
                    cand = t - bad_run + 1
                    # re-seed with the detector's selection on that frame
                    m = targets[cand] if targets[cand].any() else dm
                    if cand not in have and m.any():
                        new_kf = {"frame_idx": cand, "mask": m}
                        break
            if new_kf is None:
                break
            keyframes.append(new_kf)
        assert track is not None
        if hasattr(sam2, "release_state"):
            sam2.release_state()
        if tmp is not None:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        log.info("auto roto: shot %d-%d, %d keyframe(s), track %.1fs", a, b, len(keyframes),
                 time.perf_counter() - t_shot)
        nz = areas[areas > 0]
        ref_area = float(np.median(nz)) if nz.size and mode != "click" else 0.0
        if mode == "click" and nz.size:
            ref_area = float(nz.max())
        for t in range(b - a):
            out[a + t] = fuse(det[a + t], track[t], ref_area) if fuse_edges else track[t]
        done += b - a
        if progress_cb is not None:
            try:
                progress_cb(done, T)
            except Exception:
                pass
    return out


def _link_shot(sub_paths: List[Path]):
    """(folder, frame paths, folder to delete) of hard links to one shot's
    frames, numbered from 0 (copies where hard links are not possible)."""
    import os
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="amv_autoshot_"))
    paths = []
    for j, p in enumerate(sub_paths):
        dst = tmp / f"{j:05d}{p.suffix}"
        try:
            os.link(p, dst)
        except OSError:
            shutil.copyfile(p, dst)
        paths.append(dst)
    return tmp, paths, tmp


def gate_alpha(base: np.ndarray, selector: np.ndarray, *, box: bool = False,
               reach_frac: float = GATE_FRAC, keep_if_empty: bool = False,
               tight_frac: Optional[float] = None, cover_min: float = 0.85) -> np.ndarray:
    """``base`` matte (fine outline) kept only near what ``selector`` calls a
    character (SAM 3 concept mask, or filled person boxes when ``box``),
    with a soft fall-off ``reach_frac`` of the frame diagonal wide: an aura
    or a prop the selector leaves out is cut, the hair it under-segments is
    kept. ``keep_if_empty``: a frame where the selector sees nobody keeps
    the base matte (the selector missed a smeared character) instead of
    coming out empty.

    ``tight_frac``: when the selector covers at least ``cover_min`` of the
    base matte (it sees the whole character on this frame), the gate hugs
    the selector's outline at that reach instead: the background the base
    matte kept between the arms, under the chin or between hair spikes,
    which the selector leaves out, is cut. Where the selector misses part of
    the character (motion blur, darkness) the wide gate is kept."""
    import cv2
    s = (selector > 0.5).astype(np.uint8)
    if not s.any():
        if keep_if_empty:
            return np.clip(base.astype(np.float32), 0, 1)
        return np.zeros_like(base, dtype=np.float32)
    h, w = s.shape
    if tight_frac is not None and not box:
        b = base > 0.5
        nb = int(b.sum())
        if nb and float((b & (s > 0)).sum()) >= cover_min * nb:
            reach_frac = tight_frac
    r = max(1.0, (0.5 if box else 1.0) * reach_frac * float(np.hypot(h, w)))
    d_out = cv2.distanceTransform(1 - s, cv2.DIST_L2, 3)
    gate = np.clip((r - d_out) / (0.5 * r), 0.0, 1.0)
    return np.clip(base.astype(np.float32) * gate, 0, 1).astype(np.float32)


def apply_selector(det: np.ndarray, frames: Sequence, selector, *, prompt: str = SELECTOR_PROMPT,
                   progress_cb: Optional[Callable[[int, int], None]] = None,
                   keep: Optional[list] = None, zoom: bool = False) -> np.ndarray:
    """Detector mattes ``(T, H, W)`` (uint8 or float) kept only near what the
    selector (a :class:`~backend.models.sam3_concept.Sam3Concept`) calls a
    character on each frame (:func:`gate_alpha`; a frame where it sees nobody
    keeps the detector's matte). ``frames``: RGB arrays or image paths. Any
    selector failure returns ``det`` unchanged — the 0.5 pipeline.
    ``keep``: a list that receives the selector's masks (uint8 0/1, one per
    frame) for :func:`tighten` after tracking. ``zoom``: the selector's
    zoomed second pass per character (max quality)."""
    from PIL import Image
    out = np.empty_like(det)
    T = len(det)
    try:
        for t in range(T):
            f = frames[t]
            if not isinstance(f, np.ndarray):
                with Image.open(f) as im:
                    f = np.asarray(im.convert("RGB"))
            if zoom and hasattr(selector, "segment_zoomed"):
                sel = selector.segment_zoomed(f, prompt)
            else:
                sel = selector.segment(f, prompt)
            if keep is not None:
                keep.append((sel > 0.5).astype(np.uint8))
            d = det[t].astype(np.float32) / 255.0 if det.dtype == np.uint8 else det[t]
            g = gate_alpha(d, sel, keep_if_empty=True)
            out[t] = np.clip(np.rint(g * 255.0), 0, 255).astype(np.uint8) if det.dtype == np.uint8 else g
            if progress_cb is not None:
                progress_cb(t + 1, T)
    except Exception as exc:  # OOM, gated weights, ...: keep the 0.5 behaviour
        log.warning("auto roto: character selector failed (%s), detector matte kept", exc)
        if keep is not None:
            keep.clear()
        return det
    return out


def _fill_holes(m: np.ndarray) -> np.ndarray:
    """Background regions not connected to the frame border, filled."""
    import cv2
    inv = (~m).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    holes = [i for i in range(1, n) if i not in border]
    out = m.copy()
    if holes:
        out[np.isin(lab, holes)] = True
    return out


def _background_palette(rgb: np.ndarray, base: np.ndarray, selector: np.ndarray, k: int = 8):
    """Main colours (k-means, 0..1 RGB) of the background around the
    character: pixels the base matte calls background, in a ring 10-40 px
    outside the selector."""
    import cv2
    s = (selector > 0.5).astype(np.uint8)
    d = cv2.distanceTransform(1 - s, cv2.DIST_L2, 3)
    ring = (d > 10) & (d < 40) & (base < 0.1)
    px = rgb[ring].astype(np.float32) / 255.0
    if len(px) < 50:
        return None
    if len(px) > 20000:
        px = px[np.random.default_rng(0).choice(len(px), 20000, replace=False)]
    kk = min(k, len(px))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-3)
    _c, _l, centers = cv2.kmeans(px, kk, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    return centers


def hug_with_tips(base: np.ndarray, selector: np.ndarray, *, tight_frac: float = 0.004,
                  reach_frac: float = GATE_FRAC, opaque: float = 0.9, max_touch: float = 0.45,
                  fringe_px: int = 2, rgb: Optional[np.ndarray] = None,
                  bg_share: float = 0.35, bg_dist: float = 0.12) -> np.ndarray:
    """Matte hugged to the selector's outline at ``tight_frac``, giving back
    the *tips* the selector under-segments.

    Outside the tight gate, a piece of the base matte comes back only if it
    is opaque (> ``opaque``: a hair spike or a finger is solid and inked; a
    glow, an aura smear or a halo is semi-transparent), stays within
    ``reach_frac`` of the selector, and sticks *out* of the character: less
    than ``max_touch`` of its outline touches the hugged matte (a pocket of
    background enclosed between the arms touches it on most sides and stays
    cut). Its antialiased fringe (``fringe_px``) comes back with it.

    With ``rgb``: an opaque piece enclosed by the character (or in a hole of
    the selector's mask) comes back too when it does not have the colours of
    the background around (less than ``bg_share`` of its pixels within
    ``bg_dist`` of the background palette): an object held in front of the
    body (a bowl) is not a hole of sky between the arms. Not used by
    :func:`tighten`: on the benchmark it also gave back background pockets
    whose colours differ from the ring around (enclosed background 5.7 ->
    7.2 %)."""
    import cv2
    s = (selector > 0.5).astype(np.uint8)
    b = np.clip(base.astype(np.float32), 0, 1)
    if not s.any():
        return b
    h, w = s.shape
    diag = float(np.hypot(h, w))
    tight = gate_alpha(b, s, keep_if_empty=True, reach_frac=tight_frac)
    d_out = cv2.distanceTransform(1 - s, cv2.DIST_L2, 3)
    body = tight > 0.5
    cand = (b > opaque) & ~body & (d_out <= reach_frac * diag)
    if rgb is not None:   # holes of the selector's mask, however far from its outline
        cand |= (b > opaque) & ~body & _fill_holes(s > 0) & (s == 0)
    if not cand.any():
        return tight
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    k3 = np.ones((3, 3), np.uint8)
    near_body = cv2.dilate(body.astype(np.uint8), k3) > 0
    keep = np.zeros(n, bool)
    palette = _background_palette(rgb, b, s) if rgb is not None else None
    src = rgb.astype(np.float32) / 255.0 if palette is not None else None
    for i in range(1, n):
        x, y, bw, bh, _a = stats[i]
        x0, y0, x1, y1 = max(0, x - 1), max(0, y - 1), min(w, x + bw + 1), min(h, y + bh + 1)
        comp = lab[y0:y1, x0:x1] == i
        edge = comp & ~(cv2.erode(comp.astype(np.uint8), k3) > 0)
        ring = (cv2.dilate(comp.astype(np.uint8), k3) > 0) & ~comp
        nr = int(ring.sum())
        if nr == 0:
            continue
        touch = float((ring & near_body[y0:y1, x0:x1]).sum()) / nr
        keep[i] = 0 < touch < max_touch and edge.any()
        if not keep[i] and palette is not None and touch >= max_touch:
            cols = src[y0:y1, x0:x1][comp]
            dmin = np.sqrt(((cols[:, None, :] - palette[None, :, :]) ** 2).sum(-1)).min(1)
            keep[i] = float((dmin < bg_dist).mean()) < bg_share
    if not keep.any():
        return tight
    tips = keep[lab]
    if fringe_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * fringe_px + 1, 2 * fringe_px + 1))
        tips = cv2.dilate(tips.astype(np.uint8), k) > 0
    return np.maximum(tight, np.where(tips, b, 0.0)).astype(np.float32)


def drop_fragments(alpha: np.ndarray, selector: np.ndarray, *, max_rel: float = 0.02,
                   min_inside: float = 0.5, fringe_px: int = 2) -> np.ndarray:
    """Remove the small pieces of matte cut loose by the selector gate (bits
    of a halo ring, of a staff, of an aura): components of the matte smaller
    than ``max_rel`` of the biggest one and mostly (``min_inside``) outside
    what the selector calls a character, with their antialiased fringe."""
    import cv2
    a = np.clip(alpha.astype(np.float32), 0, 1)
    m = (a > 0.5).astype(np.uint8)
    if not m.any():
        return a
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2:
        return a
    big = stats[1:, 4].max()
    s = selector > 0.5
    inside = np.bincount(lab[s], minlength=n)
    drop = [i for i in range(1, n) if stats[i, 4] < max_rel * big and inside[i] < min_inside * stats[i, 4]]
    if not drop:
        return a
    gone = np.isin(lab, drop).astype(np.uint8)
    keep = (m > 0) & (gone == 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * fringe_px + 1, 2 * fringe_px + 1))
    zone = (cv2.dilate(gone, k) > 0) & ~(cv2.dilate(keep.astype(np.uint8), k) > 0)
    out = a.copy()
    out[gone > 0] = 0.0
    out[zone] = 0.0
    return out


def tighten(masks: np.ndarray, selector_masks: Sequence[np.ndarray], tight_frac: float = TIGHT_FRAC,
            quality: str = "standard") -> np.ndarray:
    """Final mattes ``(T, H, W)`` float hugged to the selector's outline
    (:func:`gate_alpha` at ``tight_frac`` of the diagonal) on every frame
    where the selector sees someone: background left between the arms,
    under the chin or between hair spikes is cut. 0.8 % keeps the hair tips
    SAM 3 under-segments (0.4 % clipped them, measured in
    docs/reports/chantier-roto.md); waiting for the selector to cover the
    whole tracked character first measured worse. Frames where the selector
    sees nobody are unchanged. ``quality="max"``: :func:`hug_with_tips` at
    0.4 % + :func:`drop_fragments`."""
    if len(selector_masks) != len(masks):
        return masks
    out = np.empty_like(masks)
    for t in range(len(masks)):
        sel = selector_masks[t]
        if sel is None or not np.any(sel):
            out[t] = masks[t]
            continue
        if quality == "max":
            o = hug_with_tips(masks[t], sel, tight_frac=MAX_TIGHT_FRAC, max_touch=MAX_TIP_TOUCH)
            out[t] = drop_fragments(o, sel)
        else:
            out[t] = gate_alpha(masks[t], sel, keep_if_empty=True, reach_frac=tight_frac)
    return out
