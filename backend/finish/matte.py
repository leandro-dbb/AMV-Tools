"""Alpha-matte post-processing shared by Mask Mode and the "Cutout" finishing job.

Measured on synthetic anime clips with exact ground truth (see
``backend/finish/bench_mask.py`` and the CHANGELOG), the 0.3 pipeline lost
most of the model's quality *after* inference:

- the bidirectional EMA (``temporal_smooth_alpha``) dragged a ghost behind
  moving edges — edge error x2, flicker x1.5;
- the export-time "BG-aware" alpha re-solve eroded cel-shaded interiors next
  to ink lines — IoU -2 to -4 points (-6 to -8 combined with the EMA).

This module keeps the model's alpha and only removes flicker with a
flow-aligned, photometrically validated temporal median
(:func:`backend.finish.temporal.stabilize_stack`). Halo removal at export now
fixes edge *colour* only (``clean_edges_with_background(resolve_alpha=False)``).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


def cuts_from_frames(frames: Sequence[np.ndarray], threshold: float = 0.30) -> set[int]:
    from .interpolate import detect_cuts
    return detect_cuts(list(frames), threshold)


def stabilize_alpha(masks: np.ndarray, frames: Sequence[np.ndarray], *, strength: float = 0.5,
                    cuts: Optional[set[int]] = None,
                    check_cancel: Optional[Callable[[], None]] = None) -> np.ndarray:
    """Flicker removal for a (T, H, W) float alpha stack aligned with
    ``frames`` (RGB uint8, any resolution)."""
    from .temporal import FlowWarper, stabilize_stack

    if masks.ndim != 3 or masks.shape[0] < 2 or strength <= 0:
        return masks.astype(np.float32, copy=False)
    if cuts is None:
        cuts = cuts_from_frames(frames)
    return stabilize_stack(masks.astype(np.float32), cuts=cuts, strength=strength,
                           warper=FlowWarper(frames), check_cancel=check_cancel)


def postprocess_stack(masks: np.ndarray, frames: Sequence[np.ndarray], *, stabilize: bool = True,
                      strength: float = 0.5,
                      check_cancel: Optional[Callable[[], None]] = None) -> np.ndarray:
    out = masks.astype(np.float32, copy=True)
    if stabilize:
        out = stabilize_alpha(out, frames, strength=strength, check_cancel=check_cancel)
    return out
