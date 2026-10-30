"""Beat This! (CPJKU, MIT) — beats and first beats of the bars.

Vendored (model code, see LICENSE-beat-this / LICENSE-rotary-embedding-torch)
and run only when its small checkpoint (8.5 MB) was downloaded from
Settings > Models (registry id ``beat-this-small0``) and torch imports.
Measured on 6 real songs: 10 of 14 hand-annotated drops land on one of its
first beats, against 5 with the accent heuristic (docs/reports/v06-fix.md).
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

MODEL_ID = "beat-this-small0"
_lock = threading.Lock()
_model = None
_model_path: Optional[str] = None


def checkpoint() -> Optional[str]:
    try:
        from ...finish import registry
        if not registry.is_installed(MODEL_ID):
            return None
        return str(registry.model_dir(MODEL_ID) / registry.MODELS[MODEL_ID].files[0].name)
    except Exception:
        return None


def available() -> bool:
    return checkpoint() is not None


def beats_and_downbeats(y: np.ndarray, sr: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """(beats, downbeats) in seconds, or None (model absent, torch missing,
    any failure: the caller keeps its own estimate)."""
    global _model, _model_path
    ck = checkpoint()
    if ck is None:
        return None
    try:
        with _lock:
            if _model is None or _model_path != ck:
                from .inference import Audio2Beats
                import torch
                # 8.5 MB model: a few seconds on a recent CPU, instant on a GPU
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _model, _model_path = Audio2Beats(ck, device=device), ck
            beats, downbeats = _model(np.asarray(y, dtype=np.float32), sr)
        return np.asarray(beats, dtype=np.float64), np.asarray(downbeats, dtype=np.float64)
    except Exception as exc:
        log.warning("Beat This! unavailable (%s): bars from the heuristic", exc)
        return None
