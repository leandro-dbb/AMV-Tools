"""Hard-cut detector shared by every decode path (CPU, NVIDIA, Apple Silicon).

The paths used to compare a gray histogram of frames 4 apart against one
fixed threshold. On real anime that misses most cuts (two shots of the same
palette have the same histogram: 38-69 % of the annotated cuts found) and
fires on flashes, explosions and aura bursts (precision ~67 %).

This detector sees every frame (160×90 gray, what each path already
decodes) and decides on frame-pair evidence *in context*:

- the difference is measured on the picture area only (letterbox bars are
  left out: they diluted every change of a 2.39:1 episode), after the best
  small shift (±3 px at 80×45, so pans and shakes do not count) and as a
  histogram distance;
- it must stand out from the local activity (the 2nd largest change in the
  ±4 surrounding frames: fast action changes a lot on every drawing);
- the new picture must not be the old one coming back within 6 frames
  (one-frame flashes, impact frames, white blasts), nor a picture seen just
  before (the end of such a flash).

The thresholds were chosen on frame-accurate annotations of two real
episodes (DBS, H.264 and Black Clover, HEVC, 109 cuts) by a grid search
that keeps the worst of the two scores, then checked the other way round
(chosen on one episode, measured on the other); see docs/reports/v06-fix.md.
A logistic model and gradient boosting on the same measures were tried:
better on the episodes they were fitted on, clearly worse on the other.

Decisions come ``LOOKAHEAD`` frames late (the flash test looks ahead): feed
frames with :meth:`ShotDetector.push`, which returns the frame indices
(0-based, first frame of the new shot) confirmed so far; call
:meth:`ShotDetector.finish` at the end of the stream.

With the TransNetV2 network (:mod:`cutnet`, weights downloaded on demand)
the decision is a logistic combination of the network's probabilities, the
measures above and, when the decode path hands over the 160×90 colour
picture, a colour-histogram distance. Its weights were fitted on four
annotated extracts and measured on three others (other episodes / other
passages; docs/reports/coupes.md). Decisions then come
``ShotDetector.lookahead`` frames late (the network's 100-frame windows).
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from . import cutnet

W, H = 160, 90            # input planes (``cuts._DIFF_W/_DIFF_H``)
_SW, _SH = 80, 45         # working size
_R = 3                    # motion compensation search radius (working px)
_CTX = 4                  # local activity window (frames each side)
_FLASH = 6                # how far ahead the old picture may come back
_BACK = 7                 # how far back the new picture may have been seen
LOOKAHEAD = max(_CTX, _FLASH) + 1
_BLACK = 24               # a row darker than this in both frames is a bar
_MIN_CANDIDATE = 0.035    # below this no pair is a cut (saves the context work)

# rule thresholds (see module docstring)
MIN_CHANGE = 0.20         # d + HIST_WEIGHT * hist
HIST_WEIGHT = 0.5
MIN_RATIO = 1.8           # d / (local activity + 0.02)
MIN_NEW = 0.7             # the old picture must not come back, nor the new one have been seen
# Settings > Indexing "scene detection threshold" (default 0.40) scales the
# change needed: lower = more cuts, higher = fewer.
DEFAULT_THRESHOLD = 0.40


def _small(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray, dtype=np.float32).reshape(H, W)
    return g.reshape(_SH, 2, _SW, 2).mean(axis=(1, 3))


def _rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rows that carry picture in either frame (letterbox bars excluded)."""
    rows = np.maximum(a.max(axis=1), b.max(axis=1)) > _BLACK
    if rows.sum() < 6:
        rows[:] = True
    return rows


def _span(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    rows = _rows(a, b)
    return int(np.argmax(rows)), len(rows) - int(np.argmax(rows[::-1]))


def motion_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean abs difference in [0, 1] between two working-size frames, on
    their picture rows, after the best shift of up to ±``_R`` px."""
    lo, hi = _span(a, b)
    if hi - lo > 2 * _R + 2:
        core = b[lo + _R:hi - _R, _R:_SW - _R]
        # every shift of ``a`` at once: (2R+1, 2R+1, h, w) views
        shifts = np.lib.stride_tricks.sliding_window_view(a[lo:hi], core.shape)
        best = float(np.abs(shifts - core).mean(axis=(2, 3)).min())
    else:
        best = float(np.abs(a[lo:hi] - b[lo:hi]).mean())
    return best / 255.0


def hist_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Gray histogram distance in [0, 1] on the picture rows."""
    lo, hi = _span(a, b)
    ha = np.bincount((a[lo:hi] // 8).astype(np.int32).ravel(), minlength=32).astype(np.float64)
    hb = np.bincount((b[lo:hi] // 8).astype(np.int32).ravel(), minlength=32).astype(np.float64)
    return 0.5 * float(np.abs(ha / max(ha.sum(), 1) - hb / max(hb.sum(), 1)).sum())


def distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(motion_diff, hist_diff)."""
    return motion_diff(a, b), hist_diff(a, b)


def strength_at(frames: list, d: list, t: int, min_change: float = MIN_CHANGE) -> float:
    """> 0 when frame t starts a new shot (larger = more certain).
    ``frames`` are working-size frames, ``d[i]`` = motion-compensated
    difference between i-1 and i (``d[0] = 0``); needs ``_CTX``/``_FLASH``
    frames after t (fewer at the end of the stream). The costly look-ahead
    and look-back comparisons only run once the cheap tests pass."""
    n = len(frames)
    dt = d[t]
    if dt < _MIN_CANDIDATE:
        return 0.0
    nb = sorted(d[j] for j in range(max(1, t - _CTX), min(n, t + _CTX + 1)) if j != t)
    ratio = dt / ((nb[-2] if len(nb) > 1 else 0.0) + 0.02)
    if ratio <= MIN_RATIO:
        return 0.0
    hist = hist_diff(frames[t - 1], frames[t])
    change = dt + HIST_WEIGHT * hist
    if change <= min_change:
        return 0.0
    # a flash / impact frame: the old picture comes back
    for j in range(t + 1, min(n, t + _FLASH + 1)):
        if motion_diff(frames[t - 1], frames[j]) <= MIN_NEW * dt:
            return 0.0
    # the end of such a flash: the new picture was there just before
    for i in range(max(0, t - _BACK), t - 1):
        if motion_diff(frames[i], frames[t]) <= MIN_NEW * dt:
            return 0.0
    return float(change * min(ratio, 3.0))


class ShotDetector:
    """Streaming hard-cut detector. ``push`` every decoded frame in order."""

    def __init__(self, threshold: Optional[float] = None, net="auto"):
        scale = (threshold / DEFAULT_THRESHOLD) if threshold and threshold > 0 else 1.0
        self.min_change = MIN_CHANGE * scale
        md = cutnet.load() if net == "auto" else net
        self._net = _NetDecision(md, scale) if md else None
        self.lookahead = self._net.lookahead if self._net else LOOKAHEAD
        # the decode paths hand over the 160×90 colour picture only when it
        # is used (it costs a conversion and 3× the pipe bytes)
        self.wants_color = self._net is not None
        self._frames: deque = deque()     # working-size frames, oldest first
        self._d: deque = deque()          # d[i] aligned with _frames
        self._base = 0                    # global index of _frames[0]
        self._next = 1                    # next global index to decide on
        self._n = 0                       # frames pushed
        self._probs: dict[int, float] = {}

    def push(self, gray, color=None) -> list[int]:
        """``gray``: 160×90 uint8; ``color``: the same picture as 90×160×3
        RGB uint8 when the decode path has it (better decisions with the
        network)."""
        if self._net is not None:
            return self._net.push(gray, color)
        f = _small(gray)
        if self._frames:
            self._d.append(motion_diff(self._frames[-1], f))
        else:
            self._d.append(0.0)
        self._frames.append(f)
        self._n += 1
        return self._decide(final=False)

    def finish(self) -> list[int]:
        if self._net is not None:
            return self._net.finish()
        return self._decide(final=True)

    def _decide(self, final: bool) -> list[int]:
        out: list[int] = []
        frames, d = list(self._frames), list(self._d)
        # a frame is decided once its look-ahead is in (or the stream ended);
        # its neighbours' probabilities are needed for the local-max rule
        while self._next < self._n and (final or self._next + LOOKAHEAD < self._n):
            t = self._next
            lt = t - self._base
            p = self._prob(frames, d, lt, t)
            pn = self._prob(frames, d, lt + 1, t + 1) if t + 1 < self._n else 0.0
            pp = self._probs.get(t - 1, 0.0)
            if p > 0 and p >= pp and p > pn:
                out.append(t)
            self._next += 1
        # keep what the next decisions need (look-back + context)
        keep = _BACK + _CTX + 2
        while self._frames and self._base < self._next - keep:
            self._frames.popleft()
            self._d.popleft()
            self._probs.pop(self._base - 1, None)
            self._base += 1
        return out

    def _prob(self, frames, d, lt: int, t: int) -> float:
        if t in self._probs:
            return self._probs[t]
        if lt < 1 or lt >= len(frames):
            return 0.0
        p = strength_at(frames, d, lt, self.min_change)
        self._probs[t] = p
        return p


def detect(grays, threshold: Optional[float] = None, colors=None, net="auto") -> list[int]:
    """All cut frame indices of a sequence of 160×90 gray frames (and their
    90×160×3 colour pictures, optional)."""
    det = ShotDetector(threshold, net=net)
    out: list[int] = []
    for i, g in enumerate(grays):
        out += det.push(g, None if colors is None else colors[i])
    return out + det.finish()


# ── combined decision (TransNetV2 + the measures above) ────────────────────

# Logistic weights over ``net_features`` — "color" when the decode path
# hands over the colour picture, "gray" otherwise (the network then sees
# the gray picture on its three channels). Fitted on the tuning extracts
# (DBS Beerus 01 x2, Black Clover S02E04 x2), see docs/reports/coupes.md.
NET_MODELS = {
    "color": {"w": np.array([2.7373, 2.6882, 0.4245, 0.3045, 2.072, 1.3218, 0.5524, 2.0204, -0.2918, 0.5052, 0.5684, 3.6413]), "b": -7.7736, "thr": 0.2},
    "gray": {"w": np.array([2.7881, 2.5212, 0.4396, 0.3714, 2.2534, 2.2336, 0.5617, 4.0514, -0.6872, 0.3698, 0.8062]), "b": -7.0442, "thr": 0.5},
}


def color_hist(rgb: np.ndarray) -> np.ndarray:
    """64-bin (4×4×4) RGB histogram, L1-normalised."""
    q = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3) >> 6
    h = np.bincount((q[:, 0] << 4) | (q[:, 1] << 2) | q[:, 2], minlength=64).astype(np.float32)
    return h / max(float(h.sum()), 1.0)


def area_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """float32 area (box) downscale of an (H, W) or (H, W, C) array to
    (h, w[, C])."""
    import cv2
    return cv2.resize(np.asarray(img, dtype=np.float32), (w, h), interpolation=cv2.INTER_AREA)


def net_input(gray, color) -> np.ndarray:
    """The network's 27×48×3 frame: the colour picture (or the gray one on
    three channels) area-downscaled; a 27×48×3 colour frame is taken as is."""
    src = color if color is not None else np.asarray(gray).reshape(H, W)
    src = np.asarray(src)
    if src.shape[:2] == (cutnet.IN_H, cutnet.IN_W) and src.ndim == 3:
        return src.astype(np.uint8, copy=False)
    import cv2
    small = cv2.resize(src.astype(np.uint8, copy=False), (cutnet.IN_W, cutnet.IN_H), interpolation=cv2.INTER_AREA)
    return small if small.ndim == 3 else np.repeat(small[..., None], 3, axis=2)


# feature columns 8-10 (flash look-ahead / look-back, hand-made rule) need
# the costly shifted comparisons: ``net_features(cheap=True)`` fills them
# with the values that maximise the score instead, so a frame that cannot
# reach the threshold even then is settled without them
_COSTLY = slice(8, 11)
_BEST = {"pos": (3.0, 3.0, 1.0), "neg": (0.0, 0.0, 0.0)}


def net_features(frames, d, single, many, chd, t: int, with_color: bool,
                 best: Optional[np.ndarray] = None) -> np.ndarray:
    """Feature row of frame t (first frame of a would-be new shot).
    ``frames``/``d`` as in :func:`strength_at` (frames indexable by global
    index), ``single``/``many`` the network's per-frame probabilities,
    ``chd[t]`` the colour-histogram distance between t-1 and t. With
    ``best`` (3 values) the costly columns are not computed but set to it;
    a 4th value stands for the gray-histogram distance."""
    n = len(d)
    get = lambda a, i: float(a[i]) if 0 <= i < len(a) else 0.0
    s1, s2, s0 = get(single, t - 1), get(single, t - 2), get(single, t)
    dt = d[t]
    nb = sorted(d[j] for j in range(max(1, t - _CTX), min(n, t + _CTX + 1)) if j != t)
    ratio = dt / ((nb[-2] if len(nb) > 1 else 0.0) + 0.02)
    hist = hist_diff(frames[t - 1], frames[t]) if best is None or len(best) < 4 else best[3]
    if best is not None:
        back, seen, strength = best[:3]
    elif dt > 0.03:
        back = min([motion_diff(frames[t - 1], frames[j]) for j in range(t + 1, min(n, t + _FLASH + 1))] or [1.0])
        seen = min([motion_diff(frames[i], frames[t]) for i in range(max(0, t - _BACK), t - 1)] or [1.0])
        back, seen = back / max(dt, 1e-3), seen / max(dt, 1e-3)
        strength = strength_at(frames, d, t) > 0
    else:
        back = seen = 1.0
        strength = strength_at(frames, d, t) > 0
    row = [s1, get(many, t - 1), s2, s0, max(s1 - max(s2, s0), 0.0),
           dt, min(ratio, 10.0), hist, min(back, 3.0), min(seen, 3.0), float(strength)]
    if with_color:
        row.append(float(chd[t]))
    return np.array(row, dtype=np.float32)


class _Window(dict):
    """Working frames by global index (``strength_at`` indexes them)."""

    def __len__(self):
        return self.n

    n = 0


class _NetDecision:
    """Streaming combined decision (see ``NET_MODELS``)."""

    def __init__(self, model_dev, scale: float):
        self.stream = cutnet.Stream(model_dev)
        self.scale = scale
        self.frames = _Window()
        self.d: list[float] = []
        self.chd: list[float] = []
        self.prev_hist = None
        self.color: Optional[bool] = None
        self.p: dict[int, float] = {}
        self.next = 1
        self.n = 0
        self.lookahead = cutnet.BATCH.get(self.stream.dev, 1) * cutnet.STEP + cutnet.PAD + _FLASH + 3

    def push(self, gray, color) -> list[int]:
        if self.color is None:
            self.color = color is not None
        f = _small(gray)
        self.d.append(motion_diff(self.frames[self.n - 1], f) if self.n else 0.0)
        self.frames[self.n] = f
        x = net_input(gray, color if self.color else None)
        if self.color:
            # colour-histogram distance, on the network's 27×48 frame
            h = color_hist(x)
            self.chd.append(0.5 * float(np.abs(h - self.prev_hist).sum()) if self.prev_hist is not None else 0.0)
            self.prev_hist = h
        self.stream.push(x)
        self.n += 1
        self.frames.n = self.n
        return self._decide(final=False)

    def finish(self) -> list[int]:
        self.stream.finish()
        return self._decide(final=True)

    def _ready(self, t: int, final: bool) -> bool:
        return final or (len(self.stream.single) > t + 2 and t + 1 + _FLASH + 1 < self.n)

    def _prob(self, t: int) -> float:
        if t in self.p:
            return self.p[t]
        if t < 1 or t >= self.n:
            return 0.0
        m = NET_MODELS["color" if self.color else "gray"]
        args = (self.frames, self.d, self.stream.single, self.stream.many, self.chd, t, bool(self.color))
        # A frame whose best possible score stays under the threshold cannot
        # be a cut, and as a neighbour it only has to compare lower than one
        # that is above it: 0 then gives the same decisions.
        best = np.where(m["w"][_COSTLY] > 0, _BEST["pos"], _BEST["neg"])
        for bound in (np.r_[best, 1.0 if m["w"][7] > 0 else 0.0], best):
            z = float(np.dot(m["w"], net_features(*args, best=bound)) + m["b"])
            if 1.0 / (1.0 + np.exp(-z)) <= self.thr:
                self.p[t] = 0.0
                return 0.0
        z = float(np.dot(m["w"], net_features(*args)) + m["b"])
        p = 1.0 / (1.0 + np.exp(-z))
        self.p[t] = p
        return p

    def _decide(self, final: bool) -> list[int]:
        out: list[int] = []
        m = NET_MODELS["color" if self.color else "gray"]
        self.thr = thr = min(0.95, m["thr"] * self.scale)
        while self.next < self.n and self._ready(self.next + 1, final):
            t = self.next
            p, pn, pp = self._prob(t), self._prob(t + 1), self.p.get(t - 1, 0.0)
            if p > thr and p >= pp and p > pn:
                out.append(t)
            self.next += 1
        # forget what no later decision needs
        for i in [i for i in self.frames if i < self.next - _BACK - _CTX - 2]:
            del self.frames[i]
        for i in [i for i in self.p if i < self.next - 2]:
            del self.p[i]
        return out
