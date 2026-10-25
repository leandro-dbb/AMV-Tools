"""Temporal stabilisation shared by depth maps and alpha mattes.

Per-frame models (Depth Anything, BiRefNet/ISNet, SAM 2 decoding) re-decide
their output on every frame. The result "boils": edges hop by a pixel or two
from frame to frame and, for relative depth, the whole map pumps. A plain EMA
(``mask_postprocess.temporal_smooth_alpha``) fixes static regions but drags
a ghost behind anything that moves, because it blends the *previous position*
of an edge into the current frame — on the Mask Mode benchmark it doubled the
edge error.

:func:`stabilize_stack` instead takes, for every pixel, the **median of the
current estimate and its motion-compensated neighbours** (±``radius``
frames):

1. **Motion compensation** — neighbour estimates are warped onto the current
   frame with dense optical flow (OpenCV DIS on a downscaled copy of the RGB
   frames), so a moving edge is compared with *itself*.
2. **Photometric validity** — a warped neighbour only votes where the warped
   neighbour *image* matches the current image (occlusions, flow failures and
   newly revealed areas are excluded).
3. **Median** — an isolated one-frame flip (the typical flicker) is outvoted,
   while a real change persists across neighbours and survives.
4. **Cut aware** — neighbours never cross a hard cut.

``strength`` blends between the raw estimate (0) and the median (1).
"""
from __future__ import annotations

import os
import threading
from typing import Callable, Optional, Sequence

import numpy as np

# Frames stabilised side by side: every frame's median is independent, and
# OpenCV's remap/DIS release the GIL. 120 depth frames (518x924) on an M1 Pro:
# 7.5 s -> see stabilize_stack; same output.
_THREADS = max(1, min(6, (os.cpu_count() or 2) - 2))


def nanmedian0(votes: np.ndarray) -> np.ndarray:
    """``np.nanmedian(votes, axis=0)`` for (k, H, W) votes where at least one
    vote per pixel is valid, as a sort (NaN sorts last) and the mean of the
    two middle valid votes — numpy's even-count convention, same float32
    arithmetic. numpy's version masks small axes (k = 5 here) and was most of
    the stabilisation time."""
    srt = np.sort(votes, axis=0)
    n = np.count_nonzero(~np.isnan(votes), axis=0)
    lo = np.take_along_axis(srt, ((n - 1) // 2)[None], axis=0)[0]
    hi = np.take_along_axis(srt, (n // 2)[None], axis=0)[0]
    return np.true_divide(lo + hi, np.float32(2.0) if votes.dtype == np.float32 else 2.0)


def _flow_size(h: int, w: int, max_side: int = 480) -> tuple[int, int]:
    s = min(1.0, max_side / float(max(h, w)))
    return max(16, int(round(w * s))), max(16, int(round(h * s)))


class FlowWarper:
    """Dense flow between consecutive frames + warping of per-frame maps."""

    def __init__(self, frames: Sequence[np.ndarray], max_side: int = 480):
        import cv2

        self._cv2 = cv2
        h, w = frames[0].shape[:2]
        self.size = (w, h)
        self.small = _flow_size(h, w, max_side)
        self._gray = [
            cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), self.small, interpolation=cv2.INTER_AREA)
            for f in frames
        ]
        self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self._cache: dict[tuple[int, int], np.ndarray] = {}
        # one DIS instance per thread (they are stateless between calls, but
        # not thread-safe); the main thread keeps ``_dis``
        self._local = threading.local()
        self._main = threading.get_ident()

    def _dis_here(self):
        if threading.get_ident() == self._main:
            return self._dis
        dis = getattr(self._local, "dis", None)
        if dis is None:
            dis = self._local.dis = self._cv2.DISOpticalFlow_create(self._cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        return dis

    def _flow(self, dst: int, src: int) -> np.ndarray:
        """Flow such that ``dst(x) ≈ src(x + flow(x))`` at full resolution."""
        key = (dst, src)
        if key not in self._cache:
            cv2 = self._cv2
            f = self._dis_here().calc(self._gray[dst], self._gray[src], None)
            sw, sh = self.small
            w, h = self.size
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
            f[..., 0] *= w / float(sw)
            f[..., 1] *= h / float(sh)
            self._cache[key] = f
        return self._cache[key]

    def forget(self, dst: int) -> None:
        """Drop the cached flows onto frame ``dst`` (each is used by one
        ``warp`` + ``valid`` pair only; kept, they grew by ~4 full-resolution
        float32 flow fields per frame for the whole clip)."""
        for key in [k for k in list(self._cache) if k[0] == dst]:
            self._cache.pop(key, None)

    def valid(self, dst: int, src: int, size: tuple[int, int], tau: float = 10.0) -> np.ndarray:
        """Bool mask (at ``size`` = (w, h)) where frame ``src`` warped onto
        ``dst`` matches ``dst`` photometrically (|Δgray| < ``tau``)."""
        cv2 = self._cv2
        sw, sh = self.small
        flow = self._flow(dst, src)
        small_flow = cv2.resize(flow, (sw, sh), interpolation=cv2.INTER_LINEAR)
        small_flow[..., 0] *= sw / float(self.size[0])
        small_flow[..., 1] *= sh / float(self.size[1])
        gx, gy = np.meshgrid(np.arange(sw, dtype=np.float32), np.arange(sh, dtype=np.float32))
        warped = cv2.remap(self._gray[src].astype(np.float32), gx + small_flow[..., 0], gy + small_flow[..., 1],
                           interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        err = cv2.GaussianBlur(np.abs(warped - self._gray[dst].astype(np.float32)), (3, 3), 0)
        ok = (err < tau).astype(np.uint8)
        return cv2.resize(ok, size, interpolation=cv2.INTER_NEAREST).astype(bool)

    def warp(self, value: np.ndarray, dst: int, src: int) -> np.ndarray:
        """Resample ``value`` (a map aligned with frame ``src``) onto frame
        ``dst``'s geometry."""
        cv2 = self._cv2
        flow = self._flow(dst, src)
        h, w = value.shape[:2]
        if (w, h) != self.size:
            # value at a different resolution than the frames: scale flow.
            fx, fy = w / float(self.size[0]), h / float(self.size[1])
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
            flow[..., 0] *= fx
            flow[..., 1] *= fy
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        return cv2.remap(value.astype(np.float32), gx + flow[..., 0], gy + flow[..., 1],
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def stabilize_stack(
    stack: np.ndarray,
    *,
    cuts: Optional[set[int]] = None,
    strength: float = 0.6,
    radius: int = 2,
    warper: Optional[FlowWarper] = None,
    clip_range: Optional[tuple[float, float]] = (0.0, 1.0),
    check_cancel: Optional[Callable[[], None]] = None,
) -> np.ndarray:
    """Flow-aligned, photometrically validated temporal median of ``(T, H, W)``.

    ``cuts`` holds indices i with a hard cut between i and i+1. ``warper``
    is a :class:`FlowWarper` over the clip's RGB frames; ``None`` = static
    camera assumption (identity warp, no validity test — tests only).
    """
    if stack.ndim != 3 or stack.shape[0] < 2 or strength <= 0 or radius <= 0:
        return stack.astype(np.float32, copy=True)
    if warper is not None:
        from . import temporal_cuda
        if temporal_cuda.available():
            # Same algorithm on the GPU (the CPU warps + nanmedian were most
            # of a cutout/depth job on NVIDIA).
            try:
                return temporal_cuda.stabilize_stack_cuda(
                    stack, cuts=cuts, strength=strength, radius=radius, warper=warper,
                    clip_range=clip_range, check_cancel=check_cancel)
            except Exception as exc:
                if type(exc).__name__ == "JobCancelled":
                    raise
                import logging
                logging.getLogger(__name__).warning("GPU temporal stabilisation failed (%s); using the CPU", exc)
    cuts = cuts or set()
    t_total, h, w = stack.shape
    src = stack.astype(np.float32, copy=False)
    # shot id per frame, so neighbours never cross a cut
    shot = np.cumsum([0] + [1 if i in cuts else 0 for i in range(t_total - 1)])
    out = np.empty_like(src)
    k = min(1.0, max(0.0, float(strength)))

    def one(t: int) -> None:
        if check_cancel:
            check_cancel()
        votes = [src[t]]
        for s in range(max(0, t - radius), min(t_total, t + radius + 1)):
            if s == t or shot[s] != shot[t]:
                continue
            if warper is None:
                votes.append(src[s])
                continue
            warped = warper.warp(src[s], t, s)
            ok = warper.valid(t, s, (w, h))
            votes.append(np.where(ok, warped, np.nan).astype(np.float32))
        if len(votes) == 1:
            out[t] = src[t]
            return
        if warper is not None and hasattr(warper, "forget"):
            warper.forget(t)
        med = nanmedian0(np.stack(votes)) if src.dtype == np.float32 else np.nanmedian(np.stack(votes), axis=0)
        out[t] = src[t] + k * (med - src[t])

    if _THREADS > 1 and t_total > 2:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=_THREADS, thread_name_prefix="stabilize") as ex:
            for _ in ex.map(one, range(t_total)):
                pass
    else:
        for t in range(t_total):
            one(t)
    if clip_range is not None:
        np.clip(out, clip_range[0], clip_range[1], out=out)
    return out


def align_scale_shift(cur: np.ndarray, ref: np.ndarray, *, sample: int = 4) -> np.ndarray:
    """Least-squares ``a*cur + b ≈ ref`` (relative depth has an arbitrary
    per-frame scale/shift; aligning consecutive frames removes the global
    brightness pumping before the per-pixel smoothing)."""
    c = cur[::sample, ::sample].reshape(-1).astype(np.float64)
    r = ref[::sample, ::sample].reshape(-1).astype(np.float64)
    var = c.var()
    if var < 1e-12:
        return cur
    a = ((c - c.mean()) * (r - r.mean())).mean() / var
    if not (0.2 < a < 5.0):          # degenerate fit (e.g. occlusion): leave as is
        return cur
    b = r.mean() - a * c.mean()
    return (a * cur + b).astype(np.float32)
