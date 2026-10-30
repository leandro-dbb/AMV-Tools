"""The cut-out clip of a shot, for the timeline export: a shot exported
from the roto editor is placed in the timeline as its ProRes 4444 with
alpha instead of the episode (same frames, transparent background)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .project import projects_root


def cutout_for(scene_id: Optional[int], start_ms: Optional[int], end_ms: Optional[int],
               root: Optional[Path] = None) -> Optional[dict]:
    """``{"path", "offset_ms", "duration_ms"}`` of the newest exported
    cut-out of ``scene_id`` whose range covers [start_ms, end_ms], or None."""
    if scene_id is None:
        return None
    best = None
    for f in (root or projects_root()).glob("*/project.json"):
        try:
            meta = json.loads(f.read_text())
        except Exception:
            continue
        if meta.get("scene_id") != scene_id:
            continue
        mov = (meta.get("exports") or {}).get("prores")
        if not mov or not Path(mov).exists():
            continue
        p0, p1 = int(meta.get("start_ms", 0)), int(meta.get("end_ms", 0))
        s0 = p0 if start_ms is None else int(start_ms)
        s1 = p1 if end_ms is None else int(end_ms)
        if s0 < p0 - 1 or s1 > p1 + 1:
            continue
        cand = {"path": mov, "offset_ms": max(0, s0 - p0), "duration_ms": max(1, s1 - s0),
                "updated": float(meta.get("updated", 0))}
        if best is None or cand["updated"] > best["updated"]:
            best = cand
    return best


def link_cutouts(specs: list) -> int:
    """Point the timeline clips that have an exported cut-out at it
    (in place); returns how many were linked."""
    n = 0
    for spec in specs:
        c = cutout_for(getattr(spec, "scene_id", None), spec.start_ms, spec.end_ms)
        if c is None:
            continue
        spec.source_path = c["path"]
        spec.start_ms = c["offset_ms"]
        spec.end_ms = c["offset_ms"] + c["duration_ms"]
        spec.in_frame = spec.out_frame = None
        spec.name = f"{spec.name} (cut-out)" if spec.name else "cut-out"
        n += 1
    return n
