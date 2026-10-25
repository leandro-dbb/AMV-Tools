"""Choice rules of the automatic tuning (pure functions, no hardware).

- a *count* (parallel decodes, VideoToolbox sessions): the smallest value
  whose time is within ``min_gain`` of the fastest one — a bigger value has
  to save at least 5 % to be worth its memory and its CPU;
- an *engine* or a *batch size* with a reference (the current default): the
  fastest candidate, but only if it beats the reference by ``min_gain`` (and,
  for engines, gives the same output — see ``equivalent``).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

MIN_GAIN = 0.05


def _valid(measures: Mapping[Any, Optional[float]]) -> dict:
    return {k: float(v) for k, v in measures.items()
            if v is not None and np.isfinite(v) and v > 0}


def smallest_within(measures: Mapping[Any, Optional[float]], min_gain: float = MIN_GAIN):
    """``measures``: candidate -> seconds (None/inf = failed). The smallest
    candidate the fastest one does not beat by ``min_gain``; None when
    nothing was measured."""
    ok = _valid(measures)
    if not ok:
        return None
    best = min(ok.values())
    return min(k for k, t in ok.items() if best >= t * (1.0 - min_gain))


def against_reference(measures: Mapping[Any, Optional[float]], reference, min_gain: float = MIN_GAIN):
    """The fastest candidate if it saves ``min_gain`` over ``reference``,
    else ``reference`` (also when the reference itself failed and nothing
    else ran, or when nothing ran at all)."""
    ok = _valid(measures)
    if not ok:
        return reference
    fastest = min(ok, key=ok.get)
    if reference not in ok:
        return fastest
    return fastest if ok[fastest] <= ok[reference] * (1.0 - min_gain) else reference


def equivalent(ref: np.ndarray, alt: np.ndarray, *, max_abs: float = 0.02, top_k: int = 5,
               min_top_agree: float = 0.95) -> bool:
    """Same scores within ``max_abs`` and the same top-k labels on at least
    ``min_top_agree`` of the rows (tagger probabilities, one row per image)."""
    ref = np.asarray(ref, dtype=np.float32)
    alt = np.asarray(alt, dtype=np.float32)
    if ref.shape != alt.shape or ref.size == 0 or not np.isfinite(alt).all():
        return False
    if float(np.abs(ref - alt).max()) > max_abs:
        return False
    if ref.ndim < 2:
        return True
    k = min(top_k, ref.shape[1])
    a = np.sort(np.argsort(-ref, axis=1)[:, :k], axis=1)
    b = np.sort(np.argsort(-alt, axis=1)[:, :k], axis=1)
    return float((a == b).all(axis=1).mean()) >= min_top_agree
