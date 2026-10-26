"""Outline refinement for Mask Mode: get the background out of the hair.

The character models (BiRefNet, SAM) decide *which* pixels are the character
well, but along the outline they leave background in: the sky between two
hair spikes, a ring of the old background around the silhouette, colour of
the old background in the semi-transparent edge pixels. On anime the
background next to the outline is usually a flat or slowly varying colour,
and it is right there in the frame. This module uses that:

* :func:`estimate_foreground` — colour decontamination of the export
  (blur-fusion foreground estimation, Germer et al. 2021): a semi-transparent
  pixel keeps the character's colour, never the old background's. Used by
  every alpha export (ProRes 4444, VP9, PNG) since 0.6;
* :func:`finish_edges` — the export step; ``mode="max"`` (max-quality
  pipeline) first re-solves the alpha of a narrow outline band from the
  local colours with ink lines protected (:func:`colour_solve_safe`);
* :func:`kill_background`, :func:`colour_solve`, :class:`EdgeRefiner`
  (ViTMatte on a trimap around the outline) — candidates measured on the
  benchmark and not used by default: on real episodes they did not lower the
  background kept along the outline, or ate the ink line / hair (numbers in
  docs/reports/chantier-roto.md).
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def _local_mean(img: np.ndarray, weight: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalised box mean of ``img`` over ``weight`` (and the weight cover)."""
    import cv2
    wsum = cv2.blur(weight, (k, k))
    mean = cv2.blur(img * weight[..., None], (k, k)) / (wsum[..., None] + 1e-6)
    return mean, wsum


def kill_background(
    rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    band_frac: float = 0.008,
    bg_dist: float = 0.10,
    ratio: float = 0.5,
) -> np.ndarray:
    """Remove the background caught along the outline of a matte.

    For each pixel within ``band_frac`` of the frame diagonal of the outline,
    and in background pockets enclosed by the silhouette, the local
    background colour B (mean of the confident background around, two scales)
    and the local character colours (mean of the confident character, and the
    darkest nearby line colour) are estimated. A pixel whose colour C is
    within ``bg_dist`` (RGB, 0..1) of B and clearly closer to B than to the
    character colours (``|C-B| < ratio * |C-F|``) is background: its alpha
    drops to 0, with a soft ramp in between. Never adds alpha."""
    import cv2
    a = np.clip(alpha.astype(np.float32, copy=False), 0.0, 1.0)
    m = a > 0.5
    if not m.any() or m.all():
        return a
    h, w = m.shape
    diag = float(np.hypot(h, w))
    r = max(2, int(round(band_frac * diag)))
    src = rgb.astype(np.float32) / 255.0
    m8 = m.astype(np.uint8)
    d_in = cv2.distanceTransform(m8, cv2.DIST_L2, 3)
    d_out = cv2.distanceTransform(1 - m8, cv2.DIST_L2, 3)
    fg = (d_in > r).astype(np.float32)
    bg = (d_out > r).astype(np.float32)
    # local background colour: near scale first, far scale where the near
    # window holds no confident background (enclosed pockets)
    k1 = 4 * r + 1
    k2 = 16 * r + 1
    b1, w1 = _local_mean(src, bg, k1)
    b2, w2 = _local_mean(src, bg, k2)
    B = np.where((w1 > 0.05)[..., None], b1, b2)
    wb = np.maximum(w1, w2)
    f1, wf = _local_mean(src, fg, k1)
    zone = (a > 0.02) & (d_in <= r) & (wb > 0.01)
    # pockets: background enclosed by the silhouette (closed envelope)
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k1 + 1, 2 * k1 + 1))
    env = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, kk) > 0
    zone |= env & (a > 0.02) & (wb > 0.01) & (d_in <= 3 * r)
    if not zone.any():
        return a
    C = src[zone]
    db = np.sqrt(((C - B[zone]) ** 2).sum(1))
    df = np.sqrt(((C - f1[zone]) ** 2).sum(1))
    df = np.where(wf[zone] > 0.01, df, 1.0)
    # also compare with the darkest colour around (the ink line): a line pixel
    # is never background unless the background itself is that dark
    dark = cv2.erode(src.min(axis=2), np.ones((k1 // 2 * 2 + 1,) * 2, np.uint8))[zone]
    dl = np.abs(C.max(axis=1) - dark)
    dfg = np.minimum(df, np.where(dark < 0.25, dl + 0.5 * bg_dist, 1.0))
    s = np.clip((bg_dist - db) / (0.5 * bg_dist), 0.0, 1.0) * (db < ratio * dfg)
    out = a.copy()
    out[zone] = a[zone] * (1.0 - s)
    return out


def estimate_foreground(rgb: np.ndarray, alpha: np.ndarray, r1: int = 91, r2: int = 7,
                        unmix: bool = False, nearest_below: float = 0.0) -> np.ndarray:
    """Foreground colour of every pixel (uint8 RGB) so that semi-transparent
    edge pixels carry no background colour: blur-fusion estimator (Germer,
    Uelwer, Conrad, Harmeling 2021, MIT), two passes, a few ms per frame."""
    import cv2
    img = rgb.astype(np.float32) / 255.0
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)[..., None]

    def blur_fusion(F, B, r):
        ba = cv2.blur(a[..., 0], (r, r))[..., None]
        bF = cv2.blur(F * a, (r, r)) / (ba + 1e-5)
        bB = cv2.blur(B * (1 - a), (r, r)) / ((1 - ba) + 1e-5)
        F = bF + a * (img - a * bF - (1 - a) * bB)
        return np.clip(F, 0, 1), bB

    F, bB = blur_fusion(img, img, r1)
    F, bB2 = blur_fusion(F, bB, r2)
    if unmix:
        # where the alpha is reliable enough, invert the compositing equation
        # against the local background: F = (I - (1 - a) B) / a
        u = (img - (1 - a) * bB2) / np.maximum(a, 1e-3)
        w = np.clip((a - 0.3) / 0.4, 0, 1)
        F = np.clip(w * u + (1 - w) * F, 0, 1)
    if nearest_below > 0:
        # faint pixels (mostly background): colour of the nearest opaque one
        core = a[..., 0] >= 0.98
        if core.any():
            _, lab = cv2.distanceTransformWithLabels((~core).astype(np.uint8), cv2.DIST_L2, 3,
                                                     labelType=cv2.DIST_LABEL_PIXEL)
            cy, cx = np.nonzero(core)
            idx = np.clip(lab.astype(np.int64) - 1, 0, len(cy) - 1)
            near = img[cy[idx], cx[idx]]
            w = np.clip((nearest_below - a) / max(nearest_below, 1e-3), 0, 1)
            if nearest_below > 1.0:   # every semi-transparent pixel: nearest opaque colour
                w = (a < 0.999).astype(np.float32)
            F = w * near + (1 - w) * F
    # opaque pixels keep their exact colour
    F = np.where(a >= 0.999, img, F)
    return np.clip(np.rint(F * 255.0), 0, 255).astype(np.uint8)


def colour_solve(rgb: np.ndarray, alpha: np.ndarray, band_frac: float = 0.008,
                 in_frac: Optional[float] = None, diag: Optional[float] = None) -> np.ndarray:
    """Alpha re-solved from colour in a band around the outline (nearest
    character / background colours, compositing equation). Benchmark only:
    it eats ink lines (a black line is neither the hair nor the sky)."""
    import cv2
    from .mask_postprocess import clean_edges_with_background
    a = np.clip(alpha.astype(np.float32), 0, 1)
    m = (a > 0.5).astype(np.uint8)
    h, w = m.shape
    dg = float(np.hypot(h, w)) if diag is None else diag
    r = max(2, int(round(band_frac * dg)))
    ri = r if in_frac is None else max(0, int(round(in_frac * dg)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    ki = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ri + 1, 2 * ri + 1))
    core = (cv2.erode(m, ki) > 0) if ri > 0 else (m > 0)
    out_bg = cv2.dilate(m, k) == 0
    t = np.where(core, 1.0, np.where(out_bg, 0.0, 0.5)).astype(np.float32)
    _, a2 = clean_edges_with_background(rgb, t, resolve_alpha=True)
    band = ~core & ~out_bg
    res = a.copy()
    res[band] = np.minimum(a[band], a2[band])
    return res


def colour_solve_safe(rgb: np.ndarray, alpha: np.ndarray, band_frac: float = 0.008,
                      ink_margin: float = 0.12, in_frac: Optional[float] = None) -> np.ndarray:
    """:func:`colour_solve` that never lowers the alpha of an ink pixel: a
    pixel clearly darker than both the nearby character and background
    colours is the outline, not a mix of the two. Only lowers alpha.
    Computed on the bounding box of the outline band, padded by the reach of
    every neighbourhood used (same result as on the whole frame)."""
    import cv2
    a = np.clip(alpha.astype(np.float32), 0, 1)
    h, w = a.shape
    r = max(2, int(round(band_frac * np.hypot(h, w))))
    edge = cv2.morphologyEx((a > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    edge |= (a > 0.02) & (a < 0.98)
    if not edge.any():
        return a
    ys, xs = np.nonzero(edge)
    pad = 3 * r + 56
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    if (y1 - y0) * (x1 - x0) < 0.8 * h * w:
        out = a.copy()
        out[y0:y1, x0:x1] = _colour_solve_safe_full(np.ascontiguousarray(rgb[y0:y1, x0:x1]),
                                                    np.ascontiguousarray(a[y0:y1, x0:x1]),
                                                    band_frac, ink_margin, in_frac, diag=float(np.hypot(h, w)))
        return out
    return _colour_solve_safe_full(rgb, a, band_frac, ink_margin, in_frac, diag=float(np.hypot(h, w)))


def _colour_solve_safe_full(rgb: np.ndarray, a: np.ndarray, band_frac: float, ink_margin: float,
                            in_frac: Optional[float], diag: float) -> np.ndarray:
    import cv2
    a2 = colour_solve(rgb, a, band_frac, in_frac=in_frac, diag=diag)
    src = rgb.astype(np.float32) / 255.0
    lum = src @ np.array([0.299, 0.587, 0.114], np.float32)
    m = (a > 0.5).astype(np.uint8)
    r = max(2, int(round(band_frac * diag)))
    k = (4 * r + 1, 4 * r + 1)
    fg = (cv2.erode(m, np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0).astype(np.float32)
    bg = (cv2.dilate(m, np.ones((2 * r + 1, 2 * r + 1), np.uint8)) == 0).astype(np.float32)
    lf = cv2.blur(lum * fg, k) / (cv2.blur(fg, k) + 1e-6)
    lb = cv2.blur(lum * bg, k) / (cv2.blur(bg, k) + 1e-6)
    ink = lum < np.minimum(lf, lb) - ink_margin
    return np.where(ink, a, np.minimum(a, a2)).astype(np.float32)


def alpha_levels(alpha: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Alpha levels: below ``lo`` -> 0 (faint glow, halo), above ``hi`` -> 1."""
    a = np.clip((alpha.astype(np.float32) - lo) / max(hi - lo, 1e-3), 0, 1)
    return a


class EdgeRefiner:
    """ViTMatte (hustvl, Apache-2.0 weights trained on Composition-1k) on a
    trimap: the outline band of the matte is unknown, the rest is kept."""

    def __init__(self, device, repo: str = "hustvl/vitmatte-small-composition-1k",
                 band_frac: float = 0.008, hf_token: Optional[str] = None):
        self.device = device
        self.repo = repo
        self.band_frac = band_frac
        self.hf_token = hf_token
        self._model = None
        self._proc = None
        self._lock = threading.Lock()

    def _ensure(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from transformers import VitMatteForImageMatting, VitMatteImageProcessor
            self._proc = VitMatteImageProcessor.from_pretrained(self.repo, token=self.hf_token)
            self._model = VitMatteForImageMatting.from_pretrained(self.repo, token=self.hf_token)
            self._model = self._model.to(self.device.torch_device).eval()

    def refine(self, rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        import cv2
        import torch
        from PIL import Image
        a = np.clip(alpha.astype(np.float32), 0, 1)
        m = (a > 0.5).astype(np.uint8)
        if not m.any():
            return a
        self._ensure()
        h, w = m.shape
        r = max(2, int(round(self.band_frac * np.hypot(h, w))))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        core = cv2.erode(m, k) > 0
        out_bg = cv2.dilate(m, k) == 0
        unknown = ~core & ~out_bg
        tri = np.where(core, 255, np.where(out_bg, 0, 128)).astype(np.uint8)
        inputs = self._proc(images=Image.fromarray(rgb), trimaps=Image.fromarray(tri), return_tensors="pt")
        inputs = {k_: v.to(self.device.torch_device) for k_, v in inputs.items()}
        with torch.inference_mode():
            pred = self._model(**inputs).alphas[0, 0, :h, :w].float().cpu().numpy()
        out = a.copy()
        out[unknown] = pred[unknown]
        out[core] = np.maximum(out[core], 1.0)
        return np.clip(out, 0, 1)


MAX_EDGE_IN = 0.002    # outline band re-solved by colour, inside / outside (share of the diagonal)
MAX_EDGE_OUT = 0.012
MAX_UNMIX_FAINT = 0.05   # unmix down to this alpha, nearest opaque colour below
MAX_UNMIX_SOLID = 0.30   # full unmix from this alpha up


def finish_edges(rgb: np.ndarray, alpha: np.ndarray, mode: str = "fg") -> tuple[np.ndarray, np.ndarray]:
    """Export-time edge finishing, ``(rgb_uint8, alpha_float32)``.

    ``fg``: alpha as is + foreground estimation of the edge colours.
    ``max``: the alpha of a band 2 px inside / 12 px outside the outline is
    first re-solved from the local character / background colours (glows,
    halos and background-coloured fringes drop out), never on ink pixels
    (a dark outline is neither colour); the edge colours are then unmixed
    against the local visible background (:func:`unmix_with_background`,
    down to alpha 0.05, nearest opaque colour below). Ground-truth colour
    contamination on the benchmark composites: 13.4 % (``fg``) -> 2.6 %."""
    a = np.clip(alpha.astype(np.float32, copy=False), 0.0, 1.0)
    if mode == "max":
        a = colour_solve_safe(rgb, a, band_frac=MAX_EDGE_OUT, in_frac=MAX_EDGE_IN)
        return unmix_with_background(rgb, a, faint=MAX_UNMIX_FAINT, solid=MAX_UNMIX_SOLID), a
    return estimate_foreground(rgb, a), a


def unmix_with_background(rgb: np.ndarray, alpha: np.ndarray, *, bg: Optional[np.ndarray] = None,
                          inpaint_radius: int = 5, solid: float = 0.5, faint: float = 0.15) -> np.ndarray:
    """Edge colours unmixed against the background behind them.

    The background behind each semi-transparent pixel is estimated from the
    visible background around it (``bg``: e.g. a background plate of the
    shot; otherwise the mean of the clear background pixels within
    ``inpaint_radius``, the nearest clear pixel where there is none), then
    the compositing equation is inverted: F = (C - (1 - a) B) / a. Where the
    alpha is too faint for the division to be stable (< ``faint``) the colour
    of the nearest opaque pixel is used, with a linear blend up to
    ``solid``. Opaque and clear pixels keep their colour. Only the
    semi-transparent pixels are computed. Returns uint8 RGB."""
    import cv2
    a = np.clip(alpha.astype(np.float32, copy=False), 0.0, 1.0)
    soft = (a > 0.004) & (a < 0.996)
    if not soft.any():
        return rgb
    ys, xs = np.nonzero(soft)
    av = a[ys, xs][:, None]
    C = rgb[ys, xs].astype(np.float32)
    if bg is not None:
        Bv = bg[ys, xs].astype(np.float32)
    else:
        clear = (a <= 0.004).astype(np.float32)
        k = 2 * inpaint_radius + 1
        wsum = cv2.blur(clear, (k, k))
        msum = cv2.blur(rgb.astype(np.float32) * clear[..., None], (k, k))
        wv = wsum[ys, xs]
        Bv = msum[ys, xs] / (wv[:, None] + 1e-6)
        lack = wv <= 0.05
        if lack.any() and clear.any():
            _, labb = cv2.distanceTransformWithLabels((clear == 0).astype(np.uint8), cv2.DIST_L2, 3,
                                                      labelType=cv2.DIST_LABEL_PIXEL)
            by, bx = np.nonzero(clear > 0)
            kb = np.clip(labb[ys[lack], xs[lack]].astype(np.int64) - 1, 0, len(by) - 1)
            Bv[lack] = rgb[by[kb], bx[kb]].astype(np.float32)
        elif lack.any():
            Bv[lack] = C[lack]
    F = np.clip((C - (1.0 - av) * Bv) / np.maximum(av, 1e-3), 0, 255)
    core = a >= 0.996
    blend = av[:, 0] < solid
    if core.any() and blend.any():
        # colour of the opaque pixels just inside: their mean within 3 px
        # (the nearest one, by distance transform, where none is that close)
        cf = core.astype(np.float32)
        csum = cv2.blur(cf, (7, 7))
        cmean = cv2.blur(rgb.astype(np.float32) * cf[..., None], (7, 7))
        by_, bx_ = ys[blend], xs[blend]
        cw = csum[by_, bx_]
        near = cmean[by_, bx_] / (cw[:, None] + 1e-6)
        far = cw <= 1e-3
        if far.any():
            _, lab = cv2.distanceTransformWithLabels((~core).astype(np.uint8), cv2.DIST_L2, 3,
                                                     labelType=cv2.DIST_LABEL_PIXEL)
            cy, cx = np.nonzero(core)
            kk = np.clip(lab[by_[far], bx_[far]].astype(np.int64) - 1, 0, len(cy) - 1)
            near[far] = rgb[cy[kk], cx[kk]].astype(np.float32)
        w = np.clip((av[blend] - faint) / max(solid - faint, 1e-3), 0, 1)
        F[blend] = w * F[blend] + (1 - w) * near
    out = rgb.copy()
    out[ys, xs] = np.clip(np.rint(F), 0, 255).astype(np.uint8)
    return out
