"""Edge settings of the roto editor, applied the same way to the preview
and to the export (working-size matte in, matte out).

None of them is mandatory: the defaults (ink kept, everything else at 0)
give the Stable v5 export look."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

GROW_MAX_PX = 6
SOFT_SIGMA_PX = 2.5      # softness +1: blur of this sigma
HARD_GAIN = 5.0          # softness -1: alpha contrast x(1 + this)


@dataclass
class EdgeSettings:
    keep_ink: bool = True     # "Keep the black line"
    softness: float = 0.0     # -1 harder .. +1 softer
    grow_px: float = 0.0      # -6 shrink .. +6 grow (working-size pixels)
    temporal: float = 0.0     # 0 .. 1 smoothing over time

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "EdgeSettings":
        d = d or {}
        return cls(
            keep_ink=bool(d.get("keep_ink", True)),
            softness=float(np.clip(float(d.get("softness", 0.0) or 0.0), -1.0, 1.0)),
            grow_px=float(np.clip(float(d.get("grow_px", 0.0) or 0.0), -GROW_MAX_PX, GROW_MAX_PX)),
            temporal=float(np.clip(float(d.get("temporal", 0.0) or 0.0), 0.0, 1.0)),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def key(self) -> tuple:
        return (self.keep_ink, round(self.softness, 3), round(self.grow_px, 2), round(self.temporal, 3))


def temporal_blend(alpha: np.ndarray, prev: Optional[np.ndarray], nxt: Optional[np.ndarray],
                   rgb: Optional[np.ndarray], rgb_prev: Optional[np.ndarray], rgb_next: Optional[np.ndarray],
                   strength: float) -> np.ndarray:
    """Average with the neighbouring frames' mattes carried onto this frame
    by optical flow (no ghost on moving edges), weight ``strength`` / 2 each."""
    if strength <= 0 or (prev is None and nxt is None):
        return alpha
    from ..models.auto_roto import _flow_warp
    acc, wsum = alpha * 1.0, 1.0
    for nb, nb_rgb in ((prev, rgb_prev), (nxt, rgb_next)):
        if nb is None:
            continue
        w = 0.5 * strength
        carried = _flow_warp(nb_rgb, rgb, nb) if (rgb is not None and nb_rgb is not None) else nb
        acc = acc + w * carried
        wsum += w
    return (acc / wsum).astype(np.float32)


def grow(alpha: np.ndarray, px: float) -> np.ndarray:
    """Grey dilation (px > 0) or erosion (px < 0) of the matte; fractional
    pixels blend between the two nearest integer radii."""
    import cv2
    if abs(px) < 0.05:
        return alpha
    op = cv2.dilate if px > 0 else cv2.erode
    r = abs(px)

    def morph(rad: int) -> np.ndarray:
        if rad <= 0:
            return alpha
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
        return op(alpha, k)

    lo, hi = int(np.floor(r)), int(np.ceil(r))
    if lo == hi:
        return morph(lo)
    t = r - lo
    return ((1 - t) * morph(lo) + t * morph(hi)).astype(np.float32)


def soften(alpha: np.ndarray, s: float) -> np.ndarray:
    """s > 0: blur the edge (keeps the solid inside solid); s < 0: steeper
    alpha ramp (crisper edge)."""
    import cv2
    if abs(s) < 0.01:
        return alpha
    if s > 0:
        b = cv2.GaussianBlur(alpha, (0, 0), SOFT_SIGMA_PX * s)
        # only the edge softens: inside the solid core stays at 1
        core = cv2.erode((alpha > 0.99).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        return np.where(core, np.maximum(alpha, b), b).astype(np.float32)
    g = 1.0 + HARD_GAIN * (-s)
    return np.clip((alpha - 0.5) * g + 0.5, 0.0, 1.0).astype(np.float32)


def apply_edges(alpha: np.ndarray, rgb: Optional[np.ndarray], settings: EdgeSettings, *,
                prev: Optional[np.ndarray] = None, nxt: Optional[np.ndarray] = None,
                rgb_prev: Optional[np.ndarray] = None, rgb_next: Optional[np.ndarray] = None,
                ink: bool = True) -> np.ndarray:
    """Working-size matte with the settings applied: smoothing over time,
    grow / shrink, soften / harden, then (``keep_ink`` and ``ink``) the
    outer ink line added the way the Stable v5 export does (alpha only
    raised, tiny holes closed). ``ink=False``: leave the ink to a later
    full-resolution pass (the export)."""
    a = np.clip(np.asarray(alpha, np.float32), 0.0, 1.0)
    if settings.temporal > 0:
        a = temporal_blend(a, prev, nxt, rgb, rgb_prev, rgb_next, settings.temporal)
    a = grow(a, settings.grow_px)
    a = soften(a, settings.softness)
    if ink and settings.keep_ink and rgb is not None and a.any():
        from ..models.edge_refine import absorb_ink, fill_small_holes
        a, _ = absorb_ink(rgb, a, cut_fringe=False)
        a = fill_small_holes(a)
    return np.clip(a, 0.0, 1.0).astype(np.float32)
