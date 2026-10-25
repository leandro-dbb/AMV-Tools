"""GPU (CUDA) version of :func:`backend.finish.temporal.stabilize_stack`.

The CPU implementation spends its time in full-resolution ``cv2.remap`` /
resize of the flow fields and in ``np.nanmedian`` over the votes — 0.5-1 s
per 1080p frame, the bulk of a cutout or depth job. Here the dense flow is
still computed by OpenCV DIS on the small gray frames (cheap, and the very
same flow as the CPU path), then upsampled, warped, validated and
median-voted on the GPU with the same operations:

- ``cv2.resize(INTER_LINEAR)``  → ``F.interpolate(bilinear, align_corners=False)``
- ``cv2.remap(BORDER_REPLICATE)`` → ``F.grid_sample(border, align_corners=True)``
- 3×3 Gaussian (σ = 0.8, OpenCV's default for that size) → conv2d
- ``np.nanmedian`` → sort + mean of the two middle valid votes (numpy's
  even-count convention, unlike ``torch.nanmedian``).

Results match the CPU path to within interpolation rounding (see
``tests/test_cuda_perf.py`` and the quality guard rail of the NVIDIA bench).
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np


def available() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        from ..state import get_state
        return getattr(get_state().device, "backend", None) == "cuda"
    except Exception:
        return False


def _gauss3(device):
    import torch
    s = 0.3 * ((3 - 1) * 0.5 - 1) + 0.8          # OpenCV's sigma for ksize 3
    g = torch.tensor([math.exp(-(x * x) / (2 * s * s)) for x in (-1, 0, 1)], device=device)
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, 3, 3)


class _GpuWarper:
    def __init__(self, warper, device):
        import torch

        self.w = warper
        self.dev = device
        self.size = warper.size            # (w, h) of the RGB frames
        self.small = warper.small          # (sw, sh)
        self.gray = {}
        self.kernel = _gauss3(device)
        self._flows: dict = {}
        self._small: dict = {}
        self._torch = torch

    def precompute(self, pairs: list, ahead: int = 64) -> None:
        """DIS flows (CPU, OpenCV releases the GIL) for the pairs the
        smoothing will ask for, in order, computed by a few threads up to
        ``ahead`` pairs in advance (bounded memory) — one DIS instance per
        thread, same preset: the same flows as computing them one by one."""
        import os
        import threading
        from concurrent.futures import ThreadPoolExecutor

        cv2 = self.w._cv2
        local = threading.local()

        def one(key):
            dis = getattr(local, "dis", None)
            if dis is None:
                dis = local.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            return dis.calc(self.w._gray[key[0]], self.w._gray[key[1]], None)

        self._pool = ThreadPoolExecutor(max_workers=max(1, min(4, (os.cpu_count() or 2) - 1)))
        self._todo = list(pairs)
        self._ahead = ahead
        self._one = one
        self._fill()

    def _fill(self) -> None:
        while self._todo and len(self._small) < self._ahead:
            key = self._todo.pop(0)
            self._small[key] = self._pool.submit(self._one, key)

    def close(self) -> None:
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    def _gray(self, i):
        if i not in self.gray:
            self.gray[i] = self._torch.from_numpy(self.w._gray[i].astype(np.float32)).to(self.dev)
        return self.gray[i]

    def _resize(self, x, size_wh):
        import torch.nn.functional as F
        w, h = size_wh
        return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

    def flow_full(self, dst, src):
        """Full-frame flow (1, 2, h, w), like FlowWarper._flow."""
        key = (dst, src)
        if key not in self._flows:
            fut = self._small.pop(key, None)
            if fut is not None:
                f = fut.result()
                self._fill()
            else:
                f = self.w._dis.calc(self.w._gray[dst], self.w._gray[src], None)   # small, CPU
            t = self._torch.from_numpy(f).to(self.dev).permute(2, 0, 1)[None]
            sw, sh = self.small
            w, h = self.size
            t = self._resize(t, (w, h))
            t[:, 0] *= w / float(sw)
            t[:, 1] *= h / float(sh)
            self._flows[key] = t
        return self._flows[key]

    def drop_before(self, t):
        for k in [k for k in self._flows if k[0] < t]:
            del self._flows[k]
        for k in [k for k in self.gray if k < t - 8]:
            del self.gray[k]

    @staticmethod
    def _remap(img, flow):
        """cv2.remap(img, gx + fx, gy + fy, LINEAR, BORDER_REPLICATE) for
        img (1, C, h, w) and flow (1, 2, h, w) in pixels."""
        import torch
        import torch.nn.functional as F
        _, _, h, w = img.shape
        gy, gx = torch.meshgrid(torch.arange(h, device=img.device, dtype=torch.float32),
                                torch.arange(w, device=img.device, dtype=torch.float32), indexing="ij")
        x = (gx + flow[0, 0]) * (2.0 / max(w - 1, 1)) - 1.0
        y = (gy + flow[0, 1]) * (2.0 / max(h - 1, 1)) - 1.0
        grid = torch.stack([x, y], dim=-1)[None]
        return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def warp(self, value, dst, src):
        """value: (1, 1, hv, wv) aligned with frame src → onto frame dst."""
        flow = self.flow_full(dst, src)
        hv, wv = value.shape[-2:]
        if (wv, hv) != tuple(self.size):
            fx, fy = wv / float(self.size[0]), hv / float(self.size[1])
            flow = self._resize(flow, (wv, hv)).clone()
            flow[:, 0] *= fx
            flow[:, 1] *= fy
        return self._remap(value, flow)

    def valid(self, dst, src, size_wh, tau: float = 10.0):
        import torch.nn.functional as F
        sw, sh = self.small
        flow = self._resize(self.flow_full(dst, src), (sw, sh)).clone()
        flow[:, 0] *= sw / float(self.size[0])
        flow[:, 1] *= sh / float(self.size[1])
        warped = self._remap(self._gray(src)[None, None], flow)
        err = (warped - self._gray(dst)[None, None]).abs()
        err = F.conv2d(F.pad(err, (1, 1, 1, 1), mode="reflect"), self.kernel)
        ok = (err < tau).float()
        w, h = size_wh
        return F.interpolate(ok, size=(h, w), mode="nearest") > 0.5


def _nanmedian0(votes):
    """numpy.nanmedian(axis=0) for (k, h, w) with at least one valid vote."""
    import torch
    srt, _ = torch.sort(votes, dim=0)                       # NaN sorts last
    n = (~torch.isnan(votes)).sum(dim=0, keepdim=True)
    lo = torch.gather(srt, 0, ((n - 1) // 2).clamp(min=0))
    hi = torch.gather(srt, 0, (n // 2).clamp(max=votes.shape[0] - 1))
    return ((lo + hi) * 0.5)[0]


def stabilize_stack_cuda(
    stack: np.ndarray,
    *,
    cuts: Optional[set[int]],
    strength: float,
    radius: int,
    warper,
    clip_range: Optional[tuple[float, float]],
    check_cancel: Optional[Callable[[], None]] = None,
) -> np.ndarray:
    import torch

    dev = torch.device("cuda")
    cuts = cuts or set()
    t_total, h, w = stack.shape
    src = stack.astype(np.float32, copy=False)
    shot = np.cumsum([0] + [1 if i in cuts else 0 for i in range(t_total - 1)])
    out = np.empty_like(src)
    k = min(1.0, max(0.0, float(strength)))
    gw = _GpuWarper(warper, dev)
    gw.precompute([(t, s) for t in range(t_total)
                   for s in range(max(0, t - radius), min(t_total, t + radius + 1))
                   if s != t and shot[s] == shot[t]])
    up: dict = {}
    try:
        _smooth(stack, src, out, shot, k, radius, clip_range, check_cancel, gw, up, torch, dev, w, h, t_total)
    finally:
        gw.close()
    return out


def _smooth(stack, src, out, shot, k, radius, clip_range, check_cancel, gw, up, torch, dev, w, h, t_total):
    with torch.inference_mode():
        for t in range(t_total):
            if check_cancel:
                check_cancel()
            for s in range(max(0, t - radius), min(t_total, t + radius + 1)):
                if s not in up:
                    up[s] = torch.from_numpy(src[s]).to(dev)[None, None]
            for s in [s for s in up if s < t - radius]:
                del up[s]
            gw.drop_before(t - radius)
            cur = up[t][0, 0]
            votes = [cur]
            for s in range(max(0, t - radius), min(t_total, t + radius + 1)):
                if s == t or shot[s] != shot[t]:
                    continue
                warped = gw.warp(up[s], t, s)[0, 0]
                ok = gw.valid(t, s, (w, h))[0, 0]
                votes.append(torch.where(ok, warped, torch.full_like(warped, float("nan"))))
            if len(votes) == 1:
                out[t] = src[t]
                continue
            med = _nanmedian0(torch.stack(votes))
            res = cur + k * (med - cur)
            if clip_range is not None:
                res = res.clamp(clip_range[0], clip_range[1])
            out[t] = res.cpu().numpy()
