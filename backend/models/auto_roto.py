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
        "8gb": {"detector": "general", "tracker": "tiny", "selector": "sam3", "quality": "stable"},
        "12gb": {"detector": "general", "tracker": "base_plus", "selector": "sam3", "quality": "stable"},
        "16gb": {"detector": "general", "tracker": "base_plus", "selector": "sam3", "quality": "stable"},
        "24gb": {"detector": "general", "tracker": "large", "selector": "sam3", "quality": "stable"},
        "max": {"detector": "general", "tracker": "large", "selector": "sam3", "quality": "stable"},
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
SELECTOR_CHUNK = 16          # frames handed to the batched SAM 3 at once
MAX_TIP_TOUCH = 0.30
# Held / worn objects (0.6 roto held): SAM 3 object prompts, kept when they
# touch a character and the detector sees them as foreground. See
# docs/reports/chantier-roto-held.md.
HELD_PROMPTS = ("weapon", "sword", "staff", "halo", "bowl", "hammer", "cup")
HELD_TOUCH_FRAC = 0.006
HELD_MIN_COVER = 0.6
HELD_MAX_REL = 2.0
HELD_MAX_GLOW = 0.3
HELD_DEFAULT = True          # "auto" value of the mask_keep_held setting (measured win)


# "Stable" (0.6 roto stable, the default where SAM 3 runs): BiRefNet's matte
# is the floor — every piece of it that touches a character (SAM 3, the SAM 2
# track or a held object) is kept whole, up to STABLE_REACH of the diagonal
# from the character (farther: set pieces and aura flares); SAM 3 only drops
# what is detached from the characters and never hugs the outline. A frame
# whose matte collapses next to its neighbours (motion blur, camera shake,
# flash) is refilled from the neighbours carried by optical flow. See
# docs/reports/chantier-roto.md, part 5.
STABLE_REACH = 0.05          # full keep up to half of it, soft fall-off to 0 at it
STABLE_DET_LOW = 0.3         # BiRefNet pieces counted from this alpha (inclusive)
STABLE_SEL_ERODE = 2         # SAM 3's character mask added to the matte, eroded by this (px)
LOST_FRAC = 0.6              # matte area under this x the rolling median = frame lost
LOST_WINDOW = 12             # rolling median over +-12 frames (~1 s), same shot only
JUMP_FRAC = 0.25             # area change between neighbours counted as a jump
STABLE_OPAQUE = 0.8          # BiRefNet pixels above: solid (ink, hair, props), always kept in a kept piece
STABLE_SOLID_REACH = 2.0     # solid pixels reach this x farther (props, long strands)
STABLE_CORE_PX = 3.0         # matte beyond this distance from SAM 3's character: "extras", sorted by size
SEL_FILL_SEEN = 0.1          # SAM 3 adds a character BiRefNet sees less than this share of
SEL_SPIKE = 1.6              # SAM 3 mask over this x its rolling median: swollen (aura taken for the character)
TRACK_MIN_REL = 0.3          # a track piece away from SAM 3's characters counts if >= this x their area
GUARD_SEEN_MIN = 0.3         # temporal guard: refill only if BiRefNet / SAM 3 still see this share of it
GUARD_NEAR = 0.02            # ... and only near what they see (share of the diagonal)
EXTRA_LO = 0.02              # extras under this x the characters' area stay (strands, props)
EXTRA_HI = 0.10              # ... over it go (aura flares, fused scenery)
EXTRA_BG_SHARE = 0.25        # ... in between, they go when this share of their pixels has the background's colours
EXTRA_BG_DIST = 0.12         # colour distance (0..1 RGB) to the background palette counted as "background colour"
STABLE_SOFT_PX = 2.0         # semi-transparent pixels kept within this of a solid one / SAM 3 (px)
SPECK_LO = 0.0002            # detached islands under this share of the frame: background specks
SPECK_HI = 0.001             # ... above: kept; in between: as on the previous frame
ORPHAN_PX = 3                # faint pixels farther than this from the solid matte: specks


def mask_quality(state, value: Optional[str] = None) -> str:
    """"stable", "max" or "standard": the setting ("auto" = the tier's
    pick). "stable" and "max" need the SAM 3 selector; without it,
    standard."""
    if value in ("max", "standard", "stable"):
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
                   keep: Optional[list] = None, zoom: bool = False,
                   held: Optional[list] = None, held_prompts: Sequence[str] = HELD_PROMPTS,
                   reach_frac: float = GATE_FRAC) -> np.ndarray:
    """Detector mattes ``(T, H, W)`` (uint8 or float) kept only near what the
    selector (a :class:`~backend.models.sam3_concept.Sam3Concept`) calls a
    character on each frame (:func:`gate_alpha`; a frame where it sees nobody
    keeps the detector's matte). ``frames``: RGB arrays or image paths. Any
    selector failure returns ``det`` unchanged — the 0.5 pipeline.
    ``keep``: a list that receives the selector's masks (uint8 0/1, one per
    frame) for :func:`tighten` after tracking. ``zoom``: the selector's
    zoomed second pass per character (max quality). ``held``: a list that
    receives, per frame, the detector's matte (uint8) on the objects the
    characters hold or wear (:func:`held_regions` on the selector's
    ``held_prompts`` instances, :func:`persist_held`), or None — for
    :func:`tighten` to give them back. ``reach_frac``: width of the gate
    around the selector's characters (share of the diagonal)."""
    from PIL import Image
    out = np.empty_like(det)
    T = len(det)
    held_masks: list = []
    objs_many = getattr(selector, "segment_objects_many", None) if held is not None else None
    many = getattr(selector, "segment_zoomed_many" if zoom else "segment_many", None)
    if held is not None and objs_many is None and hasattr(selector, "segment_with_objects"):
        many = None     # one frame at a time: character + objects share one encoder pass
    chunk = SELECTOR_CHUNK if (many is not None or objs_many is not None) else 1
    try:
        for c0 in range(0, T, chunk):
            imgs = []
            for t in range(c0, min(T, c0 + chunk)):
                f = frames[t]
                if not isinstance(f, np.ndarray):
                    with Image.open(f) as im:
                        f = np.asarray(im.convert("RGB"))
                imgs.append(f)
            objs = None
            if objs_many is not None:
                # batched SAM 3, character + held-object concepts on one encoder pass
                sels, objs = objs_many(imgs, prompt, held_prompts, zoom=zoom)
            elif held is not None and hasattr(selector, "segment_with_objects"):
                sel, o = selector.segment_with_objects(imgs[0], prompt, held_prompts, zoom=zoom)
                sels, objs = [sel], [o]
            elif many is not None:
                # batched SAM 3: text encoded once, images in batches
                sels = many(imgs, prompt)
            elif zoom and hasattr(selector, "segment_zoomed"):
                sels = [selector.segment_zoomed(imgs[0], prompt)]
            else:
                sels = [selector.segment(imgs[0], prompt)]
            for j, sel in enumerate(sels):
                if objs is not None:
                    d = det[c0 + j].astype(np.float32) / 255.0 if det.dtype == np.uint8 else det[c0 + j]
                    held_masks.append(held_regions(d, sel, objs[j], rgb=imgs[j]))
                _gate_one(det, out, c0 + j, sel, keep, reach_frac)
            if progress_cb is not None:
                progress_cb(min(T, c0 + chunk), T)
    except Exception as exc:  # OOM, gated weights, ...: keep the 0.5 behaviour
        log.warning("auto roto: character selector failed (%s), detector matte kept", exc)
        if keep is not None:
            keep.clear()
        if held is not None:
            held.clear()
        return det
    if held is not None and len(held_masks) == T:
        for t, m in enumerate(persist_held(held_masks)):
            if m is None or not np.any(m):
                held.append(None)
                continue
            d = det[t].astype(np.float32) / 255.0 if det.dtype == np.uint8 else det[t]
            a = keep_held(np.zeros(m.shape, np.float32), d, m)
            held.append(np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8))
    return out


def _gate_one(det: np.ndarray, out: np.ndarray, t: int, sel: np.ndarray, keep: Optional[list],
              reach_frac: float = GATE_FRAC) -> None:
    if keep is not None:
        keep.append((sel > 0.5).astype(np.uint8))
    d = det[t].astype(np.float32) / 255.0 if det.dtype == np.uint8 else det[t]
    g = gate_alpha(d, sel, keep_if_empty=True, reach_frac=reach_frac)
    out[t] = np.clip(np.rint(g * 255.0), 0, 255).astype(np.uint8) if det.dtype == np.uint8 else g


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


def _background_palette(rgb: np.ndarray, base: np.ndarray, selector: np.ndarray, k: int = 8,
                        max_px: int = 20000, attempts: int = 2):
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
    if len(px) > max_px:
        px = px[np.random.default_rng(0).choice(len(px), max_px, replace=False)]
    kk = min(k, len(px))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-3)
    _c, _l, centers = cv2.kmeans(px, kk, None, crit, attempts, cv2.KMEANS_PP_CENTERS)
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


def held_regions(base: np.ndarray, selector: np.ndarray, objects: Sequence[np.ndarray], *,
                 touch_frac: float = HELD_TOUCH_FRAC, min_cover: float = HELD_MIN_COVER,
                 max_rel: float = HELD_MAX_REL, min_frac: float = 0.0003,
                 rgb: Optional[np.ndarray] = None, max_glow: float = HELD_MAX_GLOW) -> np.ndarray:
    """Union (bool) of the objects a character holds or wears, from the
    selector's object instances (SAM 3 "staff", "weapon", "halo", "bowl"...):
    an instance counts when it touches what the selector calls a character
    (within ``touch_frac`` of the diagonal), the detector's own matte
    ``base`` covers at least ``min_cover`` of it (the detector sees it as
    foreground: a pillar or a tree the concept prompt mislabels is
    background for it) and it is not bigger than ``max_rel`` x the
    character. With the frame ``rgb``: an instance more than ``max_glow``
    of which glows — near white (luma > 0.85) or vivid (brightest channel >
    0.85, saturation > 0.5) — is a light effect (an energy beam or a ki
    flame SAM 3 calls a "weapon"), not a prop: measured on the benchmark,
    effects 49-97 % of such pixels, props at most 18 %."""
    import cv2
    s = selector > 0.5
    out = np.zeros(s.shape, bool)
    if not s.any() or not len(objects):
        return out
    h, w = s.shape
    r = max(1, int(round(touch_frac * float(np.hypot(h, w)))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    near = cv2.dilate(s.astype(np.uint8), k) > 0
    b = np.clip(base.astype(np.float32), 0, 1)
    ns = int(s.sum())
    bright = None
    if rgb is not None and max_glow is not None:
        c = np.asarray(rgb, np.float32) / 255.0
        mx, mn = c.max(-1), c.min(-1)
        white = c @ np.array([0.299, 0.587, 0.114], np.float32) > 0.85
        vivid = (mx > 0.85) & ((mx - mn) > 0.5 * mx)
        bright = white | vivid
    for o in objects:
        o = np.asarray(o) > 0.5
        n = int(o.sum())
        if n < min_frac * h * w or n > max_rel * ns or not (o & near).any():
            continue
        if float(b[o].mean()) < min_cover:
            continue
        if bright is not None and float(bright[o].mean()) > max_glow:
            continue
        out |= o
    return out


def keep_held(alpha: np.ndarray, base: np.ndarray, held: np.ndarray, fringe_px: int = 2,
              fill_px: Optional[int] = None) -> np.ndarray:
    """``alpha`` with the detector's matte given back on the ``held`` objects
    (:func:`held_regions`) and their antialiased fringe. ``fill_px``: the
    object's own mask, eroded by that many pixels, is made opaque too (the
    detector leaves holes in a bowl's contents)."""
    import cv2
    if held is None or not np.any(held):
        return alpha
    h8 = np.asarray(held).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * fringe_px + 1, 2 * fringe_px + 1))
    zone = cv2.dilate(h8, k) > 0
    b = np.clip(base.astype(np.float32), 0, 1)
    out = np.maximum(alpha, np.where(zone, b, 0.0))
    if fill_px:
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * fill_px + 1, 2 * fill_px + 1))
        out = np.maximum(out, (cv2.erode(h8, ke) > 0).astype(np.float32))
    return out.astype(np.float32)


def persist_held(held: Sequence[Optional[np.ndarray]], min_overlap: float = 0.3) -> list:
    """Temporal filter of per-frame held masks: a held piece (connected
    component) is kept on frame t only if the held mask of t-1 or t+1
    overlaps at least ``min_overlap`` of it — a one-frame false detection
    would flicker in and out."""
    import cv2
    out = []
    T = len(held)
    for t in range(T):
        m = held[t]
        if m is None or not np.any(m):
            out.append(m)
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(np.asarray(m).astype(np.uint8), connectivity=8)
        keep = np.zeros(n, bool)
        for u in (t - 1, t + 1):
            if 0 <= u < T and held[u] is not None and np.any(held[u]):
                cov = np.bincount(lab[np.asarray(held[u]) > 0], minlength=n)
                keep |= cov >= min_overlap * stats[:, 4]
        keep[0] = False
        out.append(keep[lab])
    return out


def _touching(held: np.ndarray, alpha: np.ndarray, touch_px: int) -> np.ndarray:
    """``held`` (uint8 matte) restricted to its pieces that touch ``alpha``
    (the tracked target: in "main character" mode a prop of another
    character stays out)."""
    import cv2
    m = (held > 127).astype(np.uint8)
    n, lab = cv2.connectedComponents(m, connectivity=8)
    if n <= 1:
        return held
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * touch_px + 1, 2 * touch_px + 1))
    near = cv2.dilate((alpha > 0.5).astype(np.uint8), k) > 0
    ok = np.zeros(n, bool)
    ok[np.unique(lab[near & (m > 0)])] = True
    ok[0] = False
    zone = cv2.dilate(ok[lab].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return np.where(zone, held, 0).astype(held.dtype)


def tighten(masks: np.ndarray, selector_masks: Sequence[np.ndarray], tight_frac: float = TIGHT_FRAC,
            quality: str = "standard", held: Optional[Sequence[Optional[np.ndarray]]] = None) -> np.ndarray:
    """Final mattes ``(T, H, W)`` float hugged to the selector's outline
    (:func:`gate_alpha` at ``tight_frac`` of the diagonal) on every frame
    where the selector sees someone: background left between the arms,
    under the chin or between hair spikes is cut. 0.8 % keeps the hair tips
    SAM 3 under-segments (0.4 % clipped them, measured in
    docs/reports/chantier-roto.md); waiting for the selector to cover the
    whole tracked character first measured worse. Frames where the selector
    sees nobody are unchanged. ``quality="max"``: :func:`hug_with_tips` at
    0.4 % + :func:`drop_fragments`. ``held``: per-frame mattes of the held
    objects (:func:`apply_selector`), given back where they touch the
    result."""
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
        if held is not None and t < len(held) and held[t] is not None:
            h, w = held[t].shape
            r = max(1, int(round(HELD_TOUCH_FRAC * float(np.hypot(h, w)))))
            hk = _touching(held[t], out[t], r)
            out[t] = np.maximum(out[t], hk.astype(np.float32) / 255.0)
    return out


def _rolling_median(a: np.ndarray, t: int, lo: int, hi: int, window: int = LOST_WINDOW) -> float:
    """Median of ``a`` over [t - window, t + window] ∩ [lo, hi), t excluded."""
    idx = [u for u in range(max(lo, t - window), min(hi, t + window + 1)) if u != t]
    return float(np.median(a[idx])) if idx else float(a[t])


def lost_frames(masks: np.ndarray, shots: Optional[List[tuple[int, int]]] = None,
                frac: float = LOST_FRAC, window: int = LOST_WINDOW) -> list[int]:
    """Frames whose matte area (alpha > 0.5) falls under ``frac`` x the
    rolling median of the area in the same shot: the character vanished or
    was eaten on that frame while its neighbours have it."""
    T = len(masks)
    area = np.array([float((m > 0.5).sum()) for m in masks])
    out = []
    for a, b in (shots or [(0, T)]):
        for t in range(a, b):
            med = _rolling_median(area, t, a, b, window)
            if med > 0 and area[t] < frac * med:
                out.append(t)
    return out


def area_jumps(masks: np.ndarray, shots: Optional[List[tuple[int, int]]] = None,
               frac: float = JUMP_FRAC) -> list[int]:
    """Frames t whose matte area differs from frame t-1's by more than
    ``frac`` of the larger of the two (same shot only)."""
    T = len(masks)
    area = np.array([float((m > 0.5).sum()) for m in masks])
    out = []
    for a, b in (shots or [(0, T)]):
        for t in range(a + 1, b):
            big = max(area[t], area[t - 1])
            if big > 0 and abs(area[t] - area[t - 1]) > frac * big:
                out.append(t)
    return out


def _flow_warp(src_rgb: np.ndarray, dst_rgb: np.ndarray, alpha: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """``alpha`` (on the frame ``src_rgb``) carried onto the frame
    ``dst_rgb`` by dense optical flow (Farneback, half resolution)."""
    import cv2
    h, w = alpha.shape

    def gray(f):
        g = cv2.cvtColor(np.ascontiguousarray(f[..., :3]), cv2.COLOR_RGB2GRAY)
        if g.shape != (h, w):
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
        return cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gs, gd = gray(src_rgb), gray(dst_rgb)
    flow = cv2.calcOpticalFlowFarneback(gd, gs, None, 0.5, 4, 21, 3, 5, 1.1, 0)   # dst -> src
    flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / scale
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(alpha.astype(np.float32), gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)


def stable_frame(det: np.ndarray, selector: Optional[np.ndarray], track: Optional[np.ndarray] = None,
                 held: Optional[np.ndarray] = None, *, reach_frac: float = STABLE_REACH,
                 low: float = STABLE_DET_LOW, sel_erode: int = STABLE_SEL_ERODE,
                 opaque: float = STABLE_OPAQUE, soft_px: float = STABLE_SOFT_PX,
                 solid_reach: float = STABLE_SOLID_REACH, core_px: float = STABLE_CORE_PX,
                 extra_lo: float = EXTRA_LO, extra_hi: float = EXTRA_HI,
                 prev: Optional[np.ndarray] = None, track_min_rel: float = TRACK_MIN_REL,
                 rgb: Optional[np.ndarray] = None, det_ok: bool = True) -> np.ndarray:
    """One frame of the "stable" matte.

    ``det``: BiRefNet's alpha (float 0..1), the floor. ``selector``: SAM 3's
    character mask, ``track``: the tracked matte of the characters (SAM 2 +
    fusion), ``held``: the matte of the objects they hold (uint8 or float).
    Their union says where the characters are. Every BiRefNet piece (alpha >
    ``low``) that touches it is kept — its outline, hair strands, a held
    cigarette — up to ``reach_frac`` of the diagonal from the characters,
    ``solid_reach`` x farther for its solid pixels (beyond: soft fall-off,
    then cut: set pieces fused with the character, aura flares); pieces
    that do not touch a character are dropped. Inside a kept piece the
    solid pixels (alpha > ``opaque``: ink, hair, props) all
    stay; the semi-transparent ones stay within ``soft_px`` px of a solid
    pixel or of SAM 3's character (the antialiased edge, a motion blur SAM 3
    sees) — a wide soft halo of background around the hair does not.
    What sticks out of SAM 3's character by more than ``core_px`` px is
    sorted (:func:`_drop_big_extras`): a piece under ``extra_lo`` x the
    characters' area (a strand, a cigarette, a finger SAM 3 missed) stays;
    a bigger one goes if it is over ``extra_hi`` x (an aura flare) or has
    the colours of the background around (``rgb``: scenery fused with the
    outline, a pocket between the arms), the previous frame's decision
    (``prev``) weighing in so it does not blink. Held objects and
    characters SAM 3 missed but the track has are never sorted. SAM
    3's mask (eroded by ``sel_erode`` px) is added on top when BiRefNet
    lost the character on this frame (``det_ok`` False: its matte collapsed
    next to the neighbouring frames' — motion blur), otherwise only for a
    character BiRefNet does not see at all; the track is added where SAM 3
    sees nobody. No character evidence at all: BiRefNet as is."""
    import cv2
    d = np.clip(det.astype(np.float32), 0, 1)
    h, w = d.shape
    ev = np.zeros((h, w), bool)
    s = None
    if selector is not None and np.any(selector):
        s = np.asarray(selector) > 0.5
        ev |= s
    tr = None
    missed = None
    if track is not None and np.any(track):
        tr = np.clip(np.asarray(track, np.float32), 0, 1)
        tm = tr > 0.5
        if s is not None:
            # where SAM 3 sees characters, a piece of the track away from
            # them counts only if it is character-sized (one SAM 3 missed on
            # this frame), not a rock the tracker picked up
            n_t, lab_t, st_t, _ = cv2.connectedComponentsWithStats(tm.astype(np.uint8), connectivity=8)
            if n_t > 1:
                ok = np.zeros(n_t, bool)
                ok[np.unique(lab_t[tm & s])] = True
                touch = ok.copy()
                ok |= st_t[:, 4] >= track_min_rel * float(s.sum())
                ok[0] = False
                tm = ok[lab_t]
                missed = (ok & ~touch)[lab_t]   # characters SAM 3 missed: never sorted as extras
        ev |= tm
    hm = None
    if held is not None and np.any(held):
        hm = np.asarray(held, np.float32)
        hm = hm / 255.0 if hm.max() > 1.0 else hm
        ev |= hm > 0.5
    if not ev.any():
        return d
    m = (d > low).astype(np.uint8)
    n, lab = cv2.connectedComponents(m, connectivity=8)
    keep = np.zeros(n, bool)
    if n > 1:
        keep[np.unique(lab[ev & (m > 0)])] = True
        keep[0] = False
    r = max(2.0, reach_frac * float(np.hypot(h, w)))
    d_out = cv2.distanceTransform((~ev).astype(np.uint8), cv2.DIST_L2, 3)
    reach = np.clip((r - d_out) / (0.5 * r), 0.0, 1.0)
    if solid_reach and solid_reach > 1.0:
        # solid pixels (a prop, a long strand) reach farther than a soft glow
        rs = solid_reach * r
        reach = np.where(d > opaque, np.clip((rs - d_out) / (0.5 * rs), 0.0, 1.0), reach)
    # the antialiased fringe of a kept piece (alpha under ``low``) comes along
    zone = cv2.dilate(keep[lab].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    out = np.where(zone, d, 0.0) * reach
    if soft_px is not None and soft_px > 0:
        solid = (out > opaque) | (s if s is not None else False)
        if solid.any():
            d_solid = cv2.distanceTransform((~solid).astype(np.uint8), cv2.DIST_L2, 3)
            out = np.where((out > opaque) | (d_solid <= soft_px), out, 0.0)
    if s is not None and core_px is not None and extra_hi is not None:
        protect = hm
        if missed is not None and missed.any():
            protect = np.maximum(hm if hm is not None else 0.0, missed.astype(np.float32))
        out = _drop_big_extras(out, s, protect, core_px, extra_lo, extra_hi, prev, rgb)
    if s is not None:
        se = s.astype(np.uint8)
        if not det_ok:
            pass   # BiRefNet lost the character on this frame (blur): SAM 3 fills it all
        else:
            # BiRefNet sees the characters: SAM 3 only adds one it does not
            # see at all (SAM 3 sometimes glues an aura to the character)
            n_s, lab_s = cv2.connectedComponents(se, connectivity=8)
            if n_s > 1:
                tot = np.bincount(lab_s.ravel(), minlength=n_s)
                seen = np.bincount(lab_s[d > low], minlength=n_s)
                add = seen < SEL_FILL_SEEN * np.maximum(tot, 1)
                add[0] = False
                se = add[lab_s].astype(np.uint8)
        if sel_erode > 0 and se.any():
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * sel_erode + 1, 2 * sel_erode + 1))
            se = cv2.erode(se, k)
        out = np.maximum(out, se.astype(np.float32))
    elif tr is not None:
        out = np.maximum(out, tr)
    if hm is not None:
        out = np.maximum(out, hm)
    return np.clip(out, 0, 1).astype(np.float32)


def _drop_big_extras(out: np.ndarray, s: np.ndarray, hm: Optional[np.ndarray], core_px: float,
                     lo: float, hi: float, prev: Optional[np.ndarray],
                     rgb: Optional[np.ndarray] = None, bg_share: Optional[float] = None) -> np.ndarray:
    """:func:`stable_frame`: sort the pieces of the matte sticking out of
    SAM 3's character (beyond ``core_px``). Up to ``lo`` x its area (a
    strand, a cigarette, a finger): kept. Bigger: removed when over ``hi``
    x, or — with the frame ``rgb`` — when at least ``bg_share`` of its
    pixels have the colours of the background around the character (a
    chunk of sky, grass or crystals BiRefNet fused with the outline, a
    pocket of background between the arms). A piece the previous frame
    (``prev``) kept needs 1.5 x those thresholds to go, one it did not keep
    stays out unless clearly character-coloured: the decision does not
    blink."""
    import cv2
    if bg_share is None:
        bg_share = EXTRA_BG_SHARE
    ns = float(s.sum())
    if ns <= 0:
        return out
    d_s = cv2.distanceTransform((~s).astype(np.uint8), cv2.DIST_L2, 3)
    extra = (out > 0.5) & (d_s > core_px)
    if hm is not None:
        extra &= ~(cv2.dilate((np.asarray(hm) > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    if not extra.any():
        return out
    n, lab, stats, _ = cv2.connectedComponentsWithStats(extra.astype(np.uint8), connectivity=8)
    rel = stats[:, 4] / ns
    big = rel > lo
    big[0] = False
    if not big.any():
        return out
    had = np.zeros(n, bool)
    if prev is not None:
        pv = cv2.dilate((np.asarray(prev) > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        had = np.bincount(lab[pv], minlength=n) >= 0.5 * stats[:, 4]
    share = np.zeros(n)
    palette = _background_palette(rgb, out, s, max_px=4000, attempts=1) if rgb is not None else None
    if palette is not None:
        src = rgb.astype(np.float32) / 255.0
        for i in np.flatnonzero(big):
            x, y, bw, bh, _a = stats[i]
            comp = lab[y:y + bh, x:x + bw] == i
            cols = src[y:y + bh, x:x + bw][comp]
            if len(cols) > 4000:
                cols = cols[np.random.default_rng(0).choice(len(cols), 4000, replace=False)]
            dmin = np.sqrt(((cols[:, None, :] - palette[None, :, :]) ** 2).sum(-1)).min(1)
            share[i] = float((dmin < EXTRA_BG_DIST).mean())
    k = np.where(had, 1.5, 1.0)
    drop = big & ((rel > hi * k) | (share >= np.minimum(0.95, bg_share * k)))
    if not drop.any():
        return out
    gone = cv2.dilate(drop[lab].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    gone &= d_s > core_px
    o = out.copy()
    o[gone] = 0.0
    return o


def clean_masks(masks: np.ndarray, shots: Optional[List[tuple[int, int]]] = None,
                held: Optional[Sequence[Optional[np.ndarray]]] = None,
                frames: Optional[Sequence] = None) -> np.ndarray:
    """:func:`clean_specks` on every frame, each decided with the previous
    frame's cleaned matte (not across a cut) — run last, after the temporal
    stabilisation (its median leaves faint pixels of its own). With the
    ``frames``: the character's outer ink line is then absorbed, opaque
    (:func:`~backend.models.edge_refine.absorb_ink`)."""
    T = len(masks)
    out = np.asarray(masks, np.float32).copy()
    starts = {a for a, _b in (shots or [(0, T)])}
    hl = list(held) if held is not None and len(held) == T else [None] * T
    for t in range(T):
        out[t] = clean_specks(out[t], None if t in starts else out[t - 1], hl[t])
    if frames is not None and len(frames) == T:
        import cv2
        from .edge_refine import absorb_ink
        for t in range(T):
            rgb = _rgb_of(frames[t])
            if rgb.shape[:2] != out[t].shape:
                rgb = cv2.resize(rgb, (out[t].shape[1], out[t].shape[0]), interpolation=cv2.INTER_AREA)
            out[t] = absorb_ink(rgb, out[t])[0]
    return out


def clean_specks(alpha: np.ndarray, prev: Optional[np.ndarray] = None, protect: Optional[np.ndarray] = None,
                 *, lo_frac: float = SPECK_LO, hi_frac: float = SPECK_HI, orphan_px: int = ORPHAN_PX) -> np.ndarray:
    """Background specks removed from a matte, the same way on every frame.

    Detached islands (alpha > 0.5, not the biggest piece) smaller than
    ``lo_frac`` of the frame go; islands bigger than ``hi_frac`` stay; in
    between, an island stays only if it overlaps what was kept on the
    previous frame (``prev``): a piece around the threshold does not blink
    in and out. Islands touching ``protect`` (held objects) always stay.
    Semi-transparent pixels farther than ``orphan_px`` from any solid pixel
    (alpha > 0.5) — the faint confetti around the hair — go too. Nothing
    connected to a kept piece is ever removed."""
    import cv2
    a = np.clip(alpha.astype(np.float32), 0, 1)
    m = (a > 0.5).astype(np.uint8)
    if not m.any():
        return a
    h, w = m.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = a.copy()
    if n > 2:
        area = stats[:, 4].astype(np.float64)
        big = int(np.argmax(area[1:])) + 1
        lo, hi = lo_frac * h * w, hi_frac * h * w
        keep = np.ones(n, bool)
        keep[0] = False
        pv = None if prev is None else np.asarray(prev) > 0.5
        pr = None if protect is None or not np.any(protect) else np.asarray(protect, np.float32) > 0.5
        hit_prev = np.bincount(lab[pv], minlength=n) if pv is not None else np.zeros(n)
        hit_prot = np.bincount(lab[pr], minlength=n) if pr is not None else np.zeros(n)
        for i in range(1, n):
            if i == big or area[i] >= hi or hit_prot[i] > 0:
                continue
            if area[i] < lo or hit_prev[i] < 0.5 * area[i]:
                keep[i] = False
        gone = (~keep[lab]) & (m > 0)
        if gone.any():
            out[cv2.dilate(gone.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 0.0
            out[keep[lab] & (m > 0)] = a[keep[lab] & (m > 0)]
    solid = out > 0.5
    if orphan_px and solid.any():
        d_solid = cv2.distanceTransform((~solid).astype(np.uint8), cv2.DIST_L2, 3)
        out[(out > 0) & ~solid & (d_solid > orphan_px)] = 0.0
    return out


def specks(alpha: np.ndarray, *, island_frac: float = SPECK_HI, orphan_px: int = ORPHAN_PX) -> tuple[int, int]:
    """(detached islands smaller than ``island_frac`` of the frame, faint
    pixels > 0.02 farther than ``orphan_px`` from any solid pixel) of a
    matte: the background specks a user would have to brush away."""
    import cv2
    a = np.asarray(alpha, np.float32)
    m = (a > 0.5).astype(np.uint8)
    if not m.any():
        return 0, int((a > 0.02).sum())
    h, w = m.shape
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    islands = 0
    if n > 2:
        area = stats[1:, 4]
        islands = int(((area < island_frac * h * w) & (area < area.max())).sum())
    d_solid = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 3)
    return islands, int(((a > 0.02) & (m == 0) & (d_solid > orphan_px)).sum())


def temporal_guard(masks: np.ndarray, frames: Sequence[np.ndarray], det: Optional[np.ndarray] = None,
                   shots: Optional[List[tuple[int, int]]] = None, *, frac: float = LOST_FRAC,
                   window: int = LOST_WINDOW, low: float = STABLE_DET_LOW,
                   selector: Optional[Sequence[Optional[np.ndarray]]] = None,
                   seen_min: float = GUARD_SEEN_MIN, near_frac: float = GUARD_NEAR) -> tuple[np.ndarray, list[int]]:
    """No frame may lose the character its neighbours have. A frame whose
    matte area falls under ``frac`` x the rolling median of its shot
    (:func:`lost_frames`) gets the previous frame's matte carried over by
    optical flow (the next frame's at the start of a shot), plus the
    BiRefNet pieces (``det``) that touch it; frames are repaired in order so
    a run of lost frames is carried along. The carried matte is only laid
    where the frame still shows something — BiRefNet (``det`` > ``low``) or
    SAM 3 (``selector``) within ``near_frac`` of the diagonal — and only if
    they see at least ``seen_min`` of it: a character that really left the
    frame or went behind a wall is not drawn back. Returns (masks, repaired
    frames). ``frames``: the RGB frames (any size, resized to the masks)."""
    import cv2
    T = len(masks)
    out = masks.astype(np.float32, copy=True)
    shots = shots or [(0, T)]
    lost = set(lost_frames(out, shots, frac, window))
    if not lost:
        return out, []
    fixed = []

    def rgb(t):
        f = frames[t]
        if not isinstance(f, np.ndarray):
            from PIL import Image
            with Image.open(f) as im:
                f = np.asarray(im.convert("RGB"))
        return f

    for a, b in shots:
        ok = [t for t in range(a, b) if t not in lost]
        if not ok:
            continue
        order = [t for t in range(a, b) if t in lost and t > ok[0]]
        order += [t for t in range(b - 1, a - 1, -1) if t in lost and t < ok[0]]
        for t in order:
            u = t - 1 if t > ok[0] else t + 1
            carried = _flow_warp(rgb(u), rgb(t), out[u])
            cm = carried > 0.5
            if not cm.any():
                continue
            seen = np.zeros(cm.shape, bool)
            d = None
            if det is not None:
                d = det[t].astype(np.float32) / (255.0 if det.dtype == np.uint8 else 1.0)
                seen |= d > low
            if selector is not None and t < len(selector) and selector[t] is not None:
                seen |= np.asarray(selector[t]) > 0.5
            if det is None and selector is None:
                seen[:] = True
            if float((seen & cm).sum()) < seen_min * float(cm.sum()):
                continue   # nothing there any more: gone for real (exit, occlusion)
            r = max(1, int(round(near_frac * float(np.hypot(*cm.shape)))))
            near = cv2.dilate(seen.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
            new = np.maximum(out[t], np.where(near, carried, 0.0))
            if d is not None:
                m = (d > low).astype(np.uint8)
                n, lab = cv2.connectedComponents(m, connectivity=8)
                if n > 1:
                    hit = np.zeros(n, bool)
                    hit[np.unique(lab[cm & (m > 0)])] = True
                    hit[0] = False
                    new = np.maximum(new, np.where(hit[lab], d, 0.0))
            out[t] = new
            fixed.append(t)
    return out, sorted(fixed)


def _touching_mask(m: Optional[np.ndarray], ref: np.ndarray) -> Optional[np.ndarray]:
    """Connected pieces of ``m`` that overlap ``ref`` (None stays None)."""
    import cv2
    if m is None or not np.any(m):
        return m
    b = (np.asarray(m, np.float32) > (127 if np.asarray(m).dtype == np.uint8 else 0.5)).astype(np.uint8)
    n, lab = cv2.connectedComponents(b, connectivity=8)
    ok = np.zeros(n, bool)
    ok[np.unique(lab[ref & (b > 0)])] = True
    ok[0] = False
    return np.where(ok[lab], m, 0).astype(np.asarray(m).dtype)


def _rgb_of(f) -> np.ndarray:
    if isinstance(f, np.ndarray):
        return f
    from PIL import Image
    with Image.open(f) as im:
        return np.asarray(im.convert("RGB"))


def persist_selector(sel: Sequence[Optional[np.ndarray]], shots: Optional[List[tuple[int, int]]] = None,
                     min_overlap: float = 0.3, slack_frac: float = 0.012, spike: float = SEL_SPIKE,
                     frames: Optional[Sequence] = None) -> list:
    """SAM 3's character masks without its one-frame false detections: a
    piece (connected component) of frame t is kept if the mask of t-1 or
    t+1 (same shot, dilated by ``slack_frac`` of the diagonal for motion)
    covers at least ``min_overlap`` of it, or if it is the frame's biggest
    piece (a rock SAM 3 calls a character on one blurred frame goes, the
    character stays). A frame whose mask swells over ``spike`` x the
    rolling median of the shot keeps it only near the mask of the nearest
    normal frame (SAM 3 taking an aura for the character on blurred
    frames) — carried by optical flow when the ``frames`` are given."""
    import cv2
    T = len(sel)
    out = list(sel)
    for a, b in (shots or [(0, T)]):
        if b - a < 2:
            continue
        near = {}
        for t in range(a, b):
            m = sel[t]
            if m is None or not np.any(m):
                continue
            m = np.asarray(m) > 0.5
            r = max(1, int(round(slack_frac * float(np.hypot(*m.shape)))))
            near[t] = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
        for t in range(a, b):
            m = sel[t]
            if m is None or not np.any(m):
                continue
            mb = (np.asarray(m) > 0.5).astype(np.uint8)
            n, lab, st, _ = cv2.connectedComponentsWithStats(mb, connectivity=8)
            if n <= 2:
                continue
            keep = np.zeros(n, bool)
            keep[int(np.argmax(st[1:, 4])) + 1] = True
            for u in (t - 1, t + 1):
                if u in near:
                    keep |= np.bincount(lab[near[u]], minlength=n) >= min_overlap * st[:, 4]
            keep[0] = False
            if not keep[1:].all():
                out[t] = np.where(keep[lab], np.asarray(m), 0).astype(np.asarray(m).dtype)
        # a sudden swell (SAM 3 taking the aura of a blurred frame for the
        # character) is brought back to the shape of the nearest normal frame
        area = np.array([0.0 if out[t] is None else float((np.asarray(out[t]) > 0.5).sum()) for t in range(T)])
        for t in range(a, b):
            med = _rolling_median(area, t, a, b)
            if med <= 0 or area[t] <= spike * med:
                continue
            normal = [u for u in range(a, b) if u != t and 0.6 * med <= area[u] <= spike * med]
            if not normal:
                continue
            u = min(normal, key=lambda v: abs(v - t))
            mu = np.asarray(out[u]) > 0.5
            r = max(1, int(round(2 * slack_frac * float(np.hypot(*mu.shape)))))
            if frames is not None and len(frames) == T:
                # the neighbour's mask carried onto this frame (camera shake),
                # with a tighter margin
                mu = _flow_warp(_rgb_of(frames[u]), _rgb_of(frames[t]), mu.astype(np.float32)) > 0.5
                r = max(1, int(round(0.5 * slack_frac * float(np.hypot(*mu.shape)))))
            zone = cv2.dilate(mu.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
            out[t] = np.where(zone, np.asarray(out[t]), 0).astype(np.asarray(out[t]).dtype)
    return out


def det_seen(det: np.ndarray, shots: Optional[List[tuple[int, int]]] = None) -> list:
    """Per frame: does BiRefNet still see the characters (its matte area at
    least LOST_FRAC x the rolling median of the shot)? False on the frames
    where a motion blur or a flash makes it collapse."""
    T = len(det)
    thr = 127 if det.dtype == np.uint8 else 0.5
    area = np.array([float((d > thr).sum()) for d in det])
    ok = [True] * T
    for a, b in (shots or [(0, T)]):
        for t in range(a, b):
            med = _rolling_median(area, t, a, b)
            ok[t] = not (med > 0 and area[t] < LOST_FRAC * med)
    return ok


def stable_masks(det: np.ndarray, selector_masks: Sequence[Optional[np.ndarray]],
                 track: Optional[np.ndarray] = None, held: Optional[Sequence[Optional[np.ndarray]]] = None,
                 frames: Optional[Sequence] = None, shots: Optional[List[tuple[int, int]]] = None,
                 reach_frac: float = STABLE_REACH, only_tracked: bool = False, clean: bool = True) -> np.ndarray:
    """The "stable" mattes ``(T, H, W)`` float: :func:`stable_frame` on every
    frame, then :func:`temporal_guard` (with ``frames``). ``det``: BiRefNet's
    alphas (uint8 or float), ``selector_masks``: SAM 3's character masks
    (empty list = none), ``track``: the tracked mattes, ``held``: the held
    objects' mattes (:func:`apply_selector`). ``only_tracked``: the
    characters SAM 3 sees are used only where they touch the track (one
    clicked / main character: the others stay out). ``clean``: background
    specks removed (:func:`clean_specks`, decided with the previous frame)."""
    T = len(det)
    out = np.empty(det.shape, np.float32)
    sel = list(selector_masks) if selector_masks is not None and len(selector_masks) == T else [None] * T
    sel = persist_selector(sel, shots, frames=frames if frames is not None and len(frames) == T else None)
    hl = list(held) if held is not None and len(held) == T else [None] * T
    starts = {a for a, _b in (shots or [(0, T)])}
    det_ok = det_seen(det, shots)
    for t in range(T):
        d = det[t].astype(np.float32) / (255.0 if det.dtype == np.uint8 else 1.0)
        if only_tracked and track is not None:
            tm = track[t] > 0.5
            sel[t] = _touching_mask(sel[t], tm)
            hl[t] = _touching_mask(hl[t], tm)
        rgb = None
        if frames is not None and len(frames) == T:
            rgb = frames[t]
            if not isinstance(rgb, np.ndarray):
                from PIL import Image
                with Image.open(rgb) as im:
                    rgb = np.asarray(im.convert("RGB"))
            if rgb.shape[:2] != d.shape:
                import cv2
                rgb = cv2.resize(rgb, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_AREA)
        out[t] = stable_frame(d, sel[t], None if track is None else track[t], hl[t], reach_frac=reach_frac,
                              prev=None if t in starts else out[t - 1], rgb=rgb, det_ok=det_ok[t])
    if frames is not None and len(frames) == T:
        out, fixed = temporal_guard(out, frames, det, shots, selector=sel)
        if fixed:
            log.info("auto roto (stable): %d frame(s) refilled from their neighbours: %s", len(fixed), fixed[:20])
    if clean:
        out = clean_masks(out, shots, hl)
    return out
