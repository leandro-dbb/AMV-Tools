"""Pass 2: sub-segmentation. For scenes > min_duration, sample every step seconds
and split when wd-tagger embedding cosine drift exceeds threshold."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def sub_segment(start_ms: int, end_ms: int, embeddings: list[tuple[int, np.ndarray]],
                drift_threshold: float = 0.3,
                min_sub_duration_ms: int = 1500) -> List[Tuple[int, int]]:
    """Given timed wd-tagger embeddings inside [start_ms, end_ms], return sub-boundaries.

    Returns list of (sub_start_ms, sub_end_ms). If no significant drift, returns the
    parent range as a single segment.
    """
    if len(embeddings) < 2:
        return [(start_ms, end_ms)]

    embeddings = sorted(embeddings, key=lambda e: e[0])

    boundaries: List[int] = [start_ms]
    last_emb = embeddings[0][1]
    last_boundary_ms = start_ms

    for ts, emb in embeddings[1:]:
        sim = float(np.dot(last_emb, emb))
        if (1.0 - sim) > drift_threshold and (ts - last_boundary_ms) >= min_sub_duration_ms:
            boundaries.append(ts)
            last_boundary_ms = ts
            last_emb = emb

    boundaries.append(end_ms)
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)
            if boundaries[i + 1] - boundaries[i] >= min_sub_duration_ms]
