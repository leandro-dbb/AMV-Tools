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
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

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

    def __init__(self, threshold: Optional[float] = None):
        scale = (threshold / DEFAULT_THRESHOLD) if threshold and threshold > 0 else 1.0
        self.min_change = MIN_CHANGE * scale
        self._frames: deque = deque()     # working-size frames, oldest first
        self._d: deque = deque()          # d[i] aligned with _frames
        self._base = 0                    # global index of _frames[0]
        self._next = 1                    # next global index to decide on
        self._n = 0                       # frames pushed
        self._probs: dict[int, float] = {}

    def push(self, gray) -> list[int]:
        f = _small(gray)
        if self._frames:
            self._d.append(motion_diff(self._frames[-1], f))
        else:
            self._d.append(0.0)
        self._frames.append(f)
        self._n += 1
        return self._decide(final=False)

    def finish(self) -> list[int]:
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


def detect(grays, threshold: Optional[float] = None) -> list[int]:
    """All cut frame indices of a sequence of 160×90 gray frames."""
    det = ShotDetector(threshold)
    out: list[int] = []
    for g in grays:
        out += det.push(g)
    return out + det.finish()
