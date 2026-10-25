"""Search+ camera-motion estimates on the GPU, many sample pairs at a time.

Same estimator as :func:`analysis.estimate_pair` — Shi-Tomasi corners,
pyramidal Lucas-Kanade (forward-backward checked) and a seeded 2-point
RANSAC similarity fit — written as batched PyTorch so a whole block of pairs
runs in a few kernels instead of one OpenCV call per pair on the CPU (two
worker processes on NVIDIA used to compete with ffmpeg and the main thread
for 6 vCPUs).

Faithful to OpenCV where it matters for the result:

* corners: Sobel 3×3 structure tensor, 5×5 un-normalised box, min
  eigenvalue, ``qualityLevel`` threshold relative to the image max, 3×3
  local maxima, 1-pixel border excluded, ``minDistance=3`` as a greedy
  strongest-first suppression (iterated to its fixed point), top 120;
* LK: 4-level pyrDown pyramid (5×5 Gaussian, reflect-101), Scharr
  derivatives, 11×11 window, 20 iterations / eps 0.03, the oscillation
  half-step, ``minEigThreshold`` 1e-4 at level 0, forward-backward < 0.75 px;
* RANSAC: the very hypotheses ``np.random.default_rng(1234)`` draws for the
  number of tracked points, first best count wins, then the two refits.

Bilinear sampling is float (OpenCV uses 14-bit fixed point) and greedy ties
may order differently, so the numbers differ slightly from the CPU path; the
per-scene motion class is what the guard rail compares.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from . import analysis

log = logging.getLogger(__name__)

_WIN = 11
_HALF = (_WIN - 1) // 2
_LEVELS = 4                  # maxLevel=3
_ITERS = 20
_EPS = 0.03
_FB_MAX = 0.75
# OpenCV's minEigThreshold (1e-4) is on its fixed-point gradients: Scharr
# ×32, squared, ÷2^20, ÷window area → λmin(Σ g gᵀ) in grey levels².
_MIN_EIG = 1e-4 * 1024.0 * _WIN * _WIN
# ... and its D < FLT_EPSILON test, in the same units
_MIN_DET = 1.1920929e-07 * 1024.0 * 1024.0
_RANSAC_ITERS = 32
# pairs per block: one block is ~2400 small kernels whatever its size, and
# their Python dispatch shares the GIL with the decode loop
_BATCH = 1024

_rng_table: dict[int, np.ndarray] = {}


def _hypotheses(n: int) -> np.ndarray:
    """The (i, j) draws of ``_ransac_similarity`` for n points, i != j kept."""
    t = _rng_table.get(n)
    if t is None:
        idx = np.random.default_rng(1234).integers(0, n, size=(_RANSAC_ITERS, 2)) if n > 0 \
            else np.zeros((_RANSAC_ITERS, 2), dtype=np.int64)
        t = idx
        _rng_table[n] = t
    return t


class GpuMotionEstimator:
    """Batched ``estimate_pair`` on a torch device."""

    def __init__(self, device: str = "cuda"):
        import torch
        import torch.nn.functional as F
        self.torch, self.F = torch, F
        self.device = torch.device(device)
        f32 = dict(dtype=torch.float32, device=self.device)
        g = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], **f32) / 16.0
        self._gauss = (g[:, None] * g[None, :])[None, None]
        self._sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], **f32)[None, None]
        self._sobel_y = self._sobel_x.transpose(2, 3).contiguous()
        sch = torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]], **f32) / 32.0
        self._scharr = torch.cat([sch[None, None], sch.t().contiguous()[None, None]], 0)
        r = torch.arange(-_HALF, _HALF + 1, **f32)
        wy, wx = torch.meshgrid(r, r, indexing="ij")
        self._win = torch.stack([wx.reshape(-1), wy.reshape(-1)], -1)          # (121, 2)
        hyp = np.stack([_hypotheses(n) for n in range(analysis._MAX_CORNERS + 1)])
        self._hyp = torch.from_numpy(hyp).to(self.device)                     # (121, 32, 2)

    # ── corners ──────────────────────────────────────────────────────────
    def _corners(self, a):
        """(B,1,H,W) float → points (B,K,2) xy, valid (B,K) — goodFeaturesToTrack."""
        torch, F = self.torch, self.F
        B, _, H, W = a.shape
        p = F.pad(a, (1, 1, 1, 1), mode="reflect")
        dx = F.conv2d(p, self._sobel_x)
        dy = F.conv2d(p, self._sobel_y)
        cov = torch.cat([dx * dx, dx * dy, dy * dy], 1)
        cov = F.avg_pool2d(F.pad(cov, (2, 2, 2, 2), mode="reflect"), 5, stride=1) * 25.0
        sa, sb, sc = cov[:, 0], cov[:, 1], cov[:, 2]
        eig = (sa + sc) * 0.5 - torch.sqrt(((sa - sc) * 0.5) ** 2 + sb * sb)
        eig = eig.clamp_min(0.0)
        mx = eig.flatten(1).amax(1).view(B, 1, 1)
        eig = torch.where(eig > mx * 0.01, eig, torch.zeros_like(eig))
        dil = F.max_pool2d(eig[:, None], 3, stride=1, padding=1)[:, 0]
        cand = (eig > 0) & (eig == dil)
        cand[:, 0, :] = False
        cand[:, -1, :] = False
        cand[:, :, 0] = False
        cand[:, :, -1] = False
        # minDistance=3 → any other kept corner within Chebyshev 2 (all such
        # integer offsets are < 3 px). Greedy strongest-first == the fixed
        # point of "keep unless a stronger kept neighbour exists".
        # (fixed iteration count, no host sync: past the fixed point an
        # iteration changes nothing)
        keep = cand
        for _ in range(6):
            kv = torch.where(keep, eig, torch.zeros_like(eig))
            nb = F.max_pool2d(kv[:, None], 5, stride=1, padding=2)[:, 0]
            keep = cand & ~(nb > eig)
        score = torch.where(keep, eig, torch.full_like(eig, -1.0)).flatten(1)
        k = analysis._MAX_CORNERS
        val, idx = score.topk(k, dim=1)
        valid = val > 0
        pts = torch.stack([(idx % W).float(), (idx // W).float()], -1)
        return pts, valid

    # ── LK ───────────────────────────────────────────────────────────────
    def _pyramid(self, x):
        F = self.F
        levels = [x]
        for _ in range(_LEVELS - 1):
            p = F.pad(levels[-1], (2, 2, 2, 2), mode="reflect")
            levels.append(F.conv2d(p, self._gauss)[:, :, ::2, ::2])
        return levels

    def _deriv(self, x):
        return self.F.conv2d(self.F.pad(x, (1, 1, 1, 1), mode="reflect"), self._scharr)

    def _sample(self, img, pts):
        """Bilinear 11×11 windows around pts (B,K,2) → (B,C,K,121)."""
        H, W = img.shape[-2:]
        g = pts[:, :, None, :] + self._win[None, None]
        gx = g[..., 0] * (2.0 / max(W - 1, 1)) - 1.0
        gy = g[..., 1] * (2.0 / max(H - 1, 1)) - 1.0
        # (edge replication: reflect-101 like OpenCV's pyramids measured
        # 93.7 % same-class scenes vs the CPU path on the 4-min extract, 96.2 % here)
        return self.F.grid_sample(img, self.torch.stack([gx, gy], -1), mode="bilinear",
                                  padding_mode="border", align_corners=True)

    def _lk(self, pa, pb, pts):
        """Track pts from pyramid pa to pb. Returns (next pts, status)."""
        torch = self.torch
        B, K, _ = pts.shape
        status = torch.ones(B, K, dtype=torch.bool, device=self.device)
        nxt = None
        for lvl in range(_LEVELS - 1, -1, -1):
            I, J = pa[lvl], pb[lvl]
            H, W = I.shape[-2:]
            prev = pts / (1 << lvl)
            nxt = prev.clone() if nxt is None else nxt * 2.0
            d = self._deriv(I)
            s = self._sample(torch.cat([I, d], 1), prev)                 # (B,3,K,121)
            iv, ix, iy = s[:, 0], s[:, 1], s[:, 2]
            a11, a12, a22 = (ix * ix).sum(-1), (ix * iy).sum(-1), (iy * iy).sum(-1)
            det = a11 * a22 - a12 * a12
            min_eig = (a11 + a22 - torch.sqrt((a11 - a22) ** 2 + 4.0 * a12 * a12)) * 0.5
            ok = (min_eig >= _MIN_EIG) & (det >= _MIN_DET)
            ip = torch.floor(prev - _HALF)
            inb = (ip[..., 0] >= -_WIN) & (ip[..., 0] < W) & (ip[..., 1] >= -_WIN) & (ip[..., 1] < H)
            ok = ok & inb
            if lvl == 0:
                status &= ok
            inv = torch.where(ok, 1.0 / torch.where(ok, det, torch.ones_like(det)), torch.zeros_like(det))
            active = ok.clone()
            prev_delta = torch.zeros_like(nxt)
            for j in range(_ITERS):
                inn = torch.floor(nxt - _HALF)
                inb = (inn[..., 0] >= -_WIN) & (inn[..., 0] < W) & (inn[..., 1] >= -_WIN) & (inn[..., 1] < H)
                if lvl == 0:
                    status &= inb | ~active
                active = active & inb
                # CUDA: no early exit (a host sync per iteration) — finished
                # points are masked, so the remaining iterations leave them
                # as is; on the CPU the check is free and saves the work
                if self.device.type != "cuda" and not bool(active.any()):
                    break
                jv = self._sample(J, nxt)[:, 0]
                diff = jv - iv
                b1, b2 = (diff * ix).sum(-1), (diff * iy).sum(-1)
                delta = torch.stack([(a12 * b2 - a22 * b1) * inv, (a12 * b1 - a11 * b2) * inv], -1)
                am = active[..., None]
                nxt = torch.where(am, nxt + delta, nxt)
                dd = (delta * delta).sum(-1)
                conv = dd <= _EPS * _EPS
                if j > 0:
                    osc = ((delta + prev_delta).abs() < 0.01).all(-1) & ~conv
                    nxt = torch.where((active & osc)[..., None], nxt - delta * 0.5, nxt)
                    conv = conv | osc
                prev_delta = delta
                active = active & ~conv
        return nxt, status

    # ── RANSAC similarity ────────────────────────────────────────────────
    def _fit(self, x0, x1, w):
        """Weighted closed-form similarity LSQ: (sa, sb, tx, ty), each (...)."""
        torch = self.torch
        ws = w.sum(-1).clamp_min(1e-12)
        m0 = (x0 * w[..., None]).sum(-2) / ws[..., None]
        m1 = (x1 * w[..., None]).sum(-2) / ws[..., None]
        y0, y1 = x0 - m0[..., None, :], x1 - m1[..., None, :]
        den = ((y0 * y0).sum(-1) * w).sum(-1)
        den_safe = torch.where(den > 1e-18, den, torch.ones_like(den))
        sa = ((y0 * y1).sum(-1) * w).sum(-1) / den_safe
        sb = ((y0[..., 0] * y1[..., 1] - y0[..., 1] * y1[..., 0]) * w).sum(-1) / den_safe
        sa = torch.where(den > 1e-18, sa, torch.ones_like(sa))
        sb = torch.where(den > 1e-18, sb, torch.zeros_like(sb))
        tx = m1[..., 0] - (sa * m0[..., 0] - sb * m0[..., 1])
        ty = m1[..., 1] - (sb * m0[..., 0] + sa * m0[..., 1])
        return sa, sb, tx, ty

    def _apply(self, params, x):
        sa, sb, tx, ty = (p[..., None] for p in params)
        return self.torch.stack([sa * x[..., 0] - sb * x[..., 1] + tx,
                                 sb * x[..., 0] + sa * x[..., 1] + ty], -1)

    def _ransac(self, x0, x1, ok, w_px: float):
        """x0/x1 (B,K,2) float64 centred / frame width, ok (B,K) → params, valid."""
        torch = self.torch
        B, K, _ = x0.shape
        tol = 1.0 / w_px
        n = ok.sum(-1)                                                        # (B,)
        # compact the tracked points, keeping their (strength) order
        order = torch.argsort((~ok).to(torch.int8), dim=1, stable=True)
        x0c = torch.gather(x0, 1, order[..., None].expand(-1, -1, 2))
        x1c = torch.gather(x1, 1, order[..., None].expand(-1, -1, 2))
        live = torch.arange(K, device=self.device)[None] < n[:, None]
        w_live = live.to(x0.dtype)
        hyp = self._hyp[n]                                                    # (B,32,2)
        i, j = hyp[..., 0], hyp[..., 1]
        distinct = i != j
        g = lambda x, ix: torch.gather(x, 1, ix[..., None].expand(-1, -1, 2))  # noqa: E731
        a0i, a0j, a1i, a1j = g(x0c, i), g(x0c, j), g(x1c, i), g(x1c, j)
        d0, d1 = a0j - a0i, a1j - a1i
        den = (d0 * d0).sum(-1)
        dok = (den > 1e-10) & distinct
        den = torch.where(dok, den, torch.ones_like(den))
        sa = (d0[..., 0] * d1[..., 0] + d0[..., 1] * d1[..., 1]) / den
        sb = (d0[..., 0] * d1[..., 1] - d0[..., 1] * d1[..., 0]) / den
        tx = a1i[..., 0] - (sa * a0i[..., 0] - sb * a0i[..., 1])
        ty = a1i[..., 1] - (sb * a0i[..., 0] + sa * a0i[..., 1])
        X0, X1 = x0c[:, None], x1c[:, None]                                   # (B,1,K,2)
        px = sa[..., None] * X0[..., 0] - sb[..., None] * X0[..., 1] + tx[..., None]
        py = sb[..., None] * X0[..., 0] + sa[..., None] * X0[..., 1] + ty[..., None]
        inl_all = (torch.hypot(px - X1[..., 0], py - X1[..., 1]) < tol) & live[:, None]
        counts = torch.where(dok, inl_all.sum(-1), torch.full_like(den, -1, dtype=torch.long))
        # numpy argmax: first of the best → unique key
        key = counts * 64 + (_RANSAC_ITERS - 1 - torch.arange(_RANSAC_ITERS, device=self.device))[None]
        best = key.argmax(-1)
        best_count = counts.gather(1, best[:, None])[:, 0]
        best_inl = inl_all.gather(1, best[:, None, None].expand(-1, 1, K))[:, 0]
        degenerate = best_count < max(analysis._MIN_TRACKS // 2, 2)
        w1 = torch.where(degenerate[:, None], w_live, best_inl.to(x0.dtype))
        p1 = self._fit(x0c, x1c, w1)
        err = torch.linalg.norm(self._apply(p1, x0c) - x1c, dim=-1)
        inl = (err < tol) & live
        refit = (inl.sum(-1) >= 2) & ~degenerate
        p2 = self._fit(x0c, x1c, inl.to(x0.dtype))
        params = tuple(torch.where(refit, b, a) for a, b in zip(p1, p2))
        return params, x0c, x1c, w_live, n >= 2

    # ── public ───────────────────────────────────────────────────────────
    def estimate(self, a: np.ndarray, b: np.ndarray) -> list:
        """a, b: (P,H,W) uint8 → [estimate_pair-compatible tuple or None] × P."""
        out: list = []
        for s in range(0, len(a), _BATCH):
            out.extend(self._estimate_block(a[s:s + _BATCH], b[s:s + _BATCH]))
        return out

    def _estimate_block(self, a: np.ndarray, b: np.ndarray) -> list:
        torch = self.torch
        with torch.inference_mode():
            ta = torch.from_numpy(np.ascontiguousarray(a)).to(self.device, non_blocking=True)
            tb = torch.from_numpy(np.ascontiguousarray(b)).to(self.device, non_blocking=True)
            ta = ta[:, None].float()
            tb = tb[:, None].float()
            P, _, H, W = ta.shape
            p0, cvalid = self._corners(ta)
            pa, pb = self._pyramid(ta), self._pyramid(tb)
            p1, st = self._lk(pa, pb, p0)
            back, st2 = self._lk(pb, pa, p1)
            fb = torch.linalg.norm(back - p0, dim=-1)
            ok = cvalid & st & st2 & (fb < _FB_MAX)
            n_corners = cvalid.sum(-1)
            c = torch.tensor([W / 2.0, H / 2.0], dtype=torch.float64, device=self.device)
            x0 = (p0.double() - c) / W
            x1 = (p1.double() - c) / W
            params, x0c, x1c, w_live, fit_ok = self._ransac(x0, x1, ok, float(W))
            pred = self._apply(params, x0c)
            nl = w_live.sum(-1).clamp_min(1.0)
            resid = (torch.linalg.norm(pred - x1c, dim=-1) * w_live).sum(-1) / nl
            flow = (torch.linalg.norm(x1c - x0c, dim=-1) * w_live).sum(-1) / nl
            sa, sb, tx, ty = params
            zoom = torch.hypot(sa, sb) - 1.0
            good = (n_corners >= analysis._MIN_TRACKS) & (ok.sum(-1) >= analysis._MIN_TRACKS) & fit_ok
            res = torch.stack([tx, ty, zoom, resid, flow], -1).cpu().numpy()
            good = good.cpu().numpy()
        return [tuple(float(v) for v in r) if g else None for r, g in zip(res, good)]


class GpuFrameStatsAccumulator(analysis.FrameStatsAccumulator):
    """``FrameStatsAccumulator`` whose pair estimates are queued and run in
    blocks on the GPU (from the accumulator's worker thread) — same lists,
    same aggregation. Falls back to the CPU estimator if the GPU fails."""

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self._gpu: Optional[GpuMotionEstimator] = GpuMotionEstimator(device)
        self._gpu_pending: list = []    # [(pair index, frame a, frame b)]

    def reset(self) -> None:
        super().reset()
        self._gpu_pending = []

    def start_async(self, maxsize: int = 1024, processes: int = 0):
        return super().start_async(maxsize=maxsize, processes=0)

    def add(self, ts_ms: int, gray: np.ndarray) -> None:
        if self._gpu is None:
            return super().add(ts_ms, gray)
        cv2_was = self._cv2
        self._cv2 = None        # base class: no in-line estimate, just the pair row
        try:
            n_before = len(self.p_t0)
            prev = self._prev
            super().add(ts_ms, gray)
        finally:
            self._cv2 = cv2_was
        if len(self.p_t0) > n_before and self.p_diff[-1] < analysis._PAIR_CUT_DIFF:
            self._gpu_pending.append((len(self.p_t0) - 1, prev, self._prev))
            if len(self._gpu_pending) >= _BATCH:
                self._flush_gpu()

    def _flush_gpu(self) -> None:
        items, self._gpu_pending = self._gpu_pending, []
        if not items:
            return
        try:
            ests = self._gpu.estimate(np.stack([a for _, a, _ in items]), np.stack([b for _, _, b in items]))
        except Exception as exc:
            log.warning("GPU motion estimates failed (%s); back to the CPU estimator", exc)
            self._gpu = None
            ests = [analysis.estimate_pair(a, b) if self._cv2 is not None else None for _, a, b in items]
        self._arrays_cache = None
        for (i, _a, _b), est in zip(items, ests):
            if est is not None and i < len(self.p_valid):
                (self.p_tx[i], self.p_ty[i], self.p_zoom[i], self.p_resid[i], self.p_flow[i]) = est
                self.p_valid[i] = True

    def finish(self) -> None:
        super().finish()
        self._flush_gpu()


def accumulator_for(device: str) -> analysis.FrameStatsAccumulator:
    """GPU accumulator when the device can run it, else the CPU one."""
    try:
        return GpuFrameStatsAccumulator(device)
    except Exception as exc:
        log.warning("GPU motion analysis unavailable on %s (%s); using the CPU", device, exc)
        return analysis.FrameStatsAccumulator()
