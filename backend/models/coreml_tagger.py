"""wd-tagger through Core ML on the GPU (Apple Silicon), in a worker process.

Same timm checkpoint as :class:`backend.models.wd_tagger._TorchTagger`,
compiled by Core ML instead of dispatched op by op by PyTorch MPS: M1 Pro,
batches of 8 real episode frames, 17.0 -> 24.3 img/s; scores within 0.0045
of MPS fp16, top-5 tags identical on 64/64 frames, thresholded tag sets on
98.4 %. The worker is a separate process (Core ML's predict holds the GIL);
the MPS session serves until it is ready. Conversion (~10 s) is cached in
the models folder.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

BATCH = 8


def start(config_path: str, weights_path: str):
    """A Worker (see ``backend.finish.coreml``), or None when Core ML is off."""
    from ..finish import coreml, registry
    if not coreml.available():
        return None
    try:
        cache = registry.models_root() / "coreml"
        return coreml.Worker("backend.finish.coreml:tagger_handler", str(config_path), str(weights_path),
                             str(cache), BATCH, name="coreml-tagger")
    except Exception as exc:
        log.warning("Core ML wd-tagger unavailable (%s)", exc)
        return None
