"""Frames of a roto project that probably need a look, computed with the
roto benchmark's metrics (tests/eval/roto_eval.py, backend/models/auto_roto.py)
— no reference needed:

- ``lost``        matte area under 60 % of the rolling median (character gone);
- ``jump``        area change over 25 % with the previous frame;
- ``specks``      detached islands / faint pixels away from the matte;
- ``object_lost`` a sizeable piece the previous frame had is gone here
                  (a held object, an arm), while the frame did not cut;
- ``hole``        small transparent holes inside the character;
- ``rough``       outline much more jagged than the rest of the shot.

Per-frame features are cached by the project (only the frames that changed
are recomputed after a correction); the shot-level decisions (rolling
median, roughness relative to the shot) are cheap and redone every time.
Frames the user locked (corrected or validated) are never suspect.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

SCALE = 0.5                 # metrics on a half-size matte (fast, same verdicts)
ROUGH_ABS = 9.0             # roughness % above this ...
ROUGH_REL = 1.8             # ... and this x the shot median: rough edge
OBJ_MIN_FRAC = 0.0015       # a piece of the previous frame at least this share of the frame ...
OBJ_KEPT = 0.15             # ... with less than this share still covered here: object lost
OBJ_NEAR_FRAC = 0.03        # coverage looked for within this share of the diagonal (motion)
SPECK_ORPHANS = 40          # faint isolated pixels (half size) counted as specks from this many

WEIGHTS = {"lost": 5.0, "object_lost": 3.0, "jump": 2.0, "hole": 1.5, "specks": 1.0, "rough": 1.0}


def _small(a: np.ndarray) -> np.ndarray:
    import cv2
    a = np.asarray(a)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    h, w = a.shape
    return cv2.resize(a.astype(np.float32), (max(8, int(w * SCALE)), max(8, int(h * SCALE))),
                      interpolation=cv2.INTER_AREA)


def frame_features(a: np.ndarray, prev: Optional[np.ndarray] = None) -> dict:
    """Features of one matte (any size; uint8 or float). ``prev``: the
    previous frame's matte, for ``object_lost``."""
    import cv2
    from ..models.auto_roto import specks
    s = _small(a)
    m = s > 0.5
    area = float(m.mean())
    isl, orph = specks(s, orphan_px=2) if m.any() else (0, 0)
    # holes: small transparent pockets enclosed in the matte
    hole_n = 0
    if m.any():
        inv = (~m).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
        if n > 2:
            border = np.zeros(n, bool)
            border[np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))] = True
            hh, ww = m.shape
            small = ~border & (st[:, 4] < max(16.0, 0.001 * hh * ww)) & (st[:, 4] >= 2)
            small[0] = False
            hole_n = int(small.sum())
    rough = float("nan")
    if m.sum() >= 100:
        def length(mm):
            cs, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            return float(sum(cv2.arcLength(c, True) for c in cs if len(c) >= 12))
        ls = length(cv2.GaussianBlur(s, (0, 0), 1.0) > 0.5)
        rough = (length(m) / ls - 1.0) * 100.0 if ls > 0 else float("nan")
    obj = 0.0
    if prev is not None:
        p = _small(prev) > 0.5
        if p.any():
            hh, ww = p.shape
            n, lab, st, _ = cv2.connectedComponentsWithStats(p.astype(np.uint8), connectivity=8)
            r = max(1, int(OBJ_NEAR_FRAC * float(np.hypot(hh, ww))))
            near = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
            for i in range(1, n):
                if st[i, 4] < OBJ_MIN_FRAC * hh * ww:
                    continue
                comp = lab == i
                kept = float((comp & near).sum()) / float(st[i, 4])
                if kept < OBJ_KEPT:
                    obj = max(obj, float(st[i, 4]) / (hh * ww))
    return {"area": area, "islands": int(isl), "orphans": int(orph), "holes": hole_n,
            "rough": rough, "obj": obj}


def decide(features: List[dict], locked: Iterable[int] = (), shots=None) -> List[dict]:
    """``[{frame, score, reasons}]`` of the suspect frames, sorted by frame."""
    from ..models.auto_roto import LOST_FRAC, JUMP_FRAC, _rolling_median
    T = len(features)
    if T == 0:
        return []
    locked = set(int(x) for x in locked)
    area = np.array([f["area"] for f in features], float)
    rough = np.array([f["rough"] for f in features], float)
    med_rough = float(np.nanmedian(rough)) if np.isfinite(rough).any() else float("nan")
    shots = shots or [(0, T)]
    out = []
    for a, b in shots:
        for t in range(a, b):
            if t in locked:
                continue
            f = features[t]
            reasons: Dict[str, float] = {}
            med = _rolling_median(area, t, a, b)
            if med > 0 and area[t] < LOST_FRAC * med:
                reasons["lost"] = 1.0 - area[t] / med
            if t > a:
                big = max(area[t], area[t - 1])
                if big > 0 and abs(area[t] - area[t - 1]) > JUMP_FRAC * big and "lost" not in reasons:
                    reasons["jump"] = abs(area[t] - area[t - 1]) / big
            if f.get("obj", 0) > 0 and "lost" not in reasons:
                reasons["object_lost"] = min(1.0, f["obj"] * 100)
            if f.get("islands", 0) > 0 or f.get("orphans", 0) >= SPECK_ORPHANS:
                reasons["specks"] = min(1.0, 0.3 * f.get("islands", 0) + f.get("orphans", 0) / 400.0)
            if f.get("holes", 0) > 0:
                reasons["hole"] = min(1.0, 0.25 * f["holes"])
            r = f.get("rough", float("nan"))
            if np.isfinite(r) and np.isfinite(med_rough) and r > ROUGH_ABS and r > ROUGH_REL * med_rough:
                reasons["rough"] = min(1.0, r / (3 * ROUGH_ABS))
            if reasons:
                score = sum(WEIGHTS[k] * (0.5 + 0.5 * v) for k, v in reasons.items())
                out.append({"frame": t, "score": round(float(score), 3),
                            "reasons": sorted(reasons, key=lambda k: -WEIGHTS[k])})
    out.sort(key=lambda s: s["frame"])
    return out
