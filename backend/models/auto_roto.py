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
CORE_MODE = "adaptive"       # "always": fill from the track core on every frame


# Models of the Auto engine per memory tier (backend/models/tiers.py ids).
# detector: BiRefNetModel variant; tracker: SAM2Model variant, None = no
# tracking (per-frame detector only — SAM 2 takes 20-40 s per frame on a CPU).
# Measured in docs/reports/chantier-roto.md.
MASK_TIERS: dict[str, dict[str, dict[str, Optional[str]]]] = {
    # BiRefNet "general" is the best per-frame detector measured on real
    # episodes (J 0.85 vs 0.75 ISNet anime, 0.72 ToonOut); the tracker size
    # barely changes the Auto result (it only follows the detector's seed),
    # so small cards take the lighter SAM 2.1.
    "nvidia": {
        "fast": {"detector": "general", "tracker": "tiny"},
        "8gb": {"detector": "general", "tracker": "tiny"},
        "12gb": {"detector": "general", "tracker": "base_plus"},
        "16gb": {"detector": "general", "tracker": "base_plus"},
        "24gb": {"detector": "general", "tracker": "large"},
        "max": {"detector": "general", "tracker": "large"},
    },
    # Not measured on a Mac (unreachable during the 0.5 work): same detector,
    # lighter trackers.
    "mac": {
        "fast": {"detector": "general", "tracker": "tiny"},
        "mac8": {"detector": "general", "tracker": "tiny"},
        "mac16": {"detector": "general", "tracker": "tiny"},
        "mac24": {"detector": "general", "tracker": "base_plus"},
        "mac32": {"detector": "general", "tracker": "base_plus"},
        "max": {"detector": "general", "tracker": "large"},
    },
    # CPU: BiRefNet takes seconds per frame there, ISNet anime ~0.5 s.
    "cpu": {"fast": {"detector": "anime", "tracker": None}},
}
FALLBACK = {"detector": "anime", "tracker": "base_plus"}


# Catalogue for Settings > Models (label, download size, license).
MASK_MODEL_INFO: dict[str, dict] = {
    "toonout": {"id": "joelseytre/toonout", "label": "ToonOut (BiRefNet anime)", "size_mb": 885, "license": "mit"},
    "anime": {"id": "skytnt/anime-seg", "label": "ISNet anime-seg", "size_mb": 176, "license": "apache-2.0"},
    "general": {"id": "ZhengPeng7/BiRefNet", "label": "BiRefNet", "size_mb": 445, "license": "mit"},
    "tiny": {"id": "facebook/sam2.1-hiera-tiny", "label": "SAM 2.1 tiny", "size_mb": 156, "license": "apache-2.0"},
    "base_plus": {"id": "facebook/sam2.1-hiera-base-plus", "label": "SAM 2.1 base+", "size_mb": 324, "license": "apache-2.0"},
    "large": {"id": "facebook/sam2.1-hiera-large", "label": "SAM 2.1 large", "size_mb": 898, "license": "apache-2.0"},
}


def describe_mask_models(platform: str, tier: str) -> dict[str, dict]:
    """``{"mask_detector": ..., "mask_tracker": ...}`` rows of a tier for the
    Settings table (no tracker row on CPU)."""
    choice = MASK_TIERS.get(platform, {}).get(tier) or {}
    out = {}
    for role, key in (("mask_detector", "detector"), ("mask_tracker", "tracker")):
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


def fuse(detector: np.ndarray, track: np.ndarray) -> np.ndarray:
    """Final alpha of one frame: detector edges inside the tracked region,
    tracker core where the detector missed (motion blur), nothing outside."""
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
        for t in range(b - a):
            out[a + t] = fuse(det[a + t], track[t]) if fuse_edges else track[t]
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
