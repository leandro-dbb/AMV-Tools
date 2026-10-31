"""Cut detection measured on the annotated extracts (tests/data/cuts_gt).

Runs the real indexing decode paths on the extracts and scores them at the
exact frame against the hand annotations, per extract and per split (tune /
test, see each ground-truth file's ``split``; the two original extracts are
tuning material). Usage::

    python -m tests.eval.cuts_eval --media DIR [--paths sw,single_pass,...] [--net 0|1]

``--media``: folder(s) holding the extracts (``video`` of each JSON). Frame
numbers in the annotations are decode order; the app numbers frames on the
``-fps_mode cfr`` grid, which duplicates a frame that follows a timestamp
gap (Black Clover's stream has one) — :func:`gt` maps them through the
packet timestamps.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

GT_DIR = Path(__file__).resolve().parents[1] / "data" / "cuts_gt"
ORIGINAL = {"dbs_beerus_01_extract.json", "black_clover_s02e04_extract.json"}
ORIGINAL_STEMS = {n[:-5] for n in ORIGINAL}


def ground_truths() -> dict[str, dict]:
    out = {}
    for p in sorted(GT_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        d.setdefault("split", "tune" if p.name in ORIGINAL else "test")
        out[p.stem] = d
    return out


def find_media(gtd: dict, media: list[Path]) -> Path | None:
    for m in media:
        if (m / gtd["video"]).exists():
            return m / gtd["video"]
    return None


def _pts(path: Path) -> np.ndarray:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.split()
    return np.sort(np.array([int(x) for x in out if x.strip().lstrip("-").isdigit()]))


def gt(gtd: dict, path: Path) -> tuple[list[int], set[int]]:
    """(cut frames, don't-care frames) on the app's cfr frame grid."""
    p = _pts(path)
    step = 1 / (Fraction(gtd["fps"]) * Fraction(gtd.get("time_base", "1/1000")))   # pts ticks per frame
    extra = np.maximum(0, np.round(np.diff(p) / float(step)).astype(int) - 1)
    m = lambda n: int(n + extra[:max(0, n - 1)].sum())
    cuts = sorted(m(c["frame"]) for c in gtd["hard_cuts"])
    dc: set[int] = set()
    for a in gtd.get("ambiguous", []):
        dc.update(range(m(a["frame"]), m(a.get("end", a["frame"])) + 1))
    for x in gtd.get("dissolves", []):
        dc.update(range(m(x["start"]) - 1, m(x["end"]) + 2))
    return cuts, dc


def score(cuts: list[int], dc: set[int], det: list[int]) -> dict:
    """Exact-frame recall / precision (one frame off counts as a miss and a
    false cut); detections inside dissolves / ambiguous zones are ignored."""
    ds, cs = set(det), set(cuts)
    tp = [c for c in cuts if c in ds]
    fp = sorted(d for d in ds if d not in cs and d not in dc)
    return {"n": len(cuts), "tp": len(tp), "recall": len(tp) / max(1, len(cuts)),
            "precision": len(tp) / max(1, len(tp) + len(fp)),
            "missed": [c for c in cuts if c not in ds], "false": fp}


def run_path(name: str, path: Path, threshold: float = 0.40) -> list[int]:
    """Cut frames found by one indexing decode path."""
    from backend.indexing import cuts
    fps = cuts.probe_video(str(path))["fps"]
    os.environ.pop("AMV_DECODE", None)
    if name in ("sw", "pyav", "hw"):
        os.environ["AMV_DECODE"] = name
        scenes = cuts.detect_cuts(str(path), threshold=threshold)
        starts = [s for s, _ in scenes[1:]]
    elif name == "single_pass":
        from backend.indexing import single_pass
        os.environ["AMV_DECODE"] = "sw"
        scenes, _ = single_pass.detect_cuts_with_medians(str(path), threshold=threshold)
        starts = [s for s, _ in scenes[1:]]
    elif name == "onepass":
        from backend.indexing import onepass
        scenes, _ = onepass.detect_cuts_and_medians(str(path), threshold)
        starts = [s for s, _ in scenes[1:]]
    elif name == "nvc":
        from backend.indexing import nvc_pass
        res = nvc_pass.detect_cuts_and_medians(str(path), threshold)
        if res is None:
            raise RuntimeError("nvc_pass does not handle this file")
        starts = [s for s, _ in res[0][1:]]
    else:
        raise ValueError(name)
    os.environ.pop("AMV_DECODE", None)
    return [round(s * fps / 1000) for s in starts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", action="append", required=True)
    ap.add_argument("--paths", default="sw")
    ap.add_argument("--net", default=None, help="force the cut network on (1) or off (0)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", default=None, help="comma-separated ground-truth stems")
    a = ap.parse_args()
    if a.net is not None:
        os.environ["AMV_CUTNET"] = a.net
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    media = [Path(m) for m in a.media]
    rows = []
    for stem, gtd in ground_truths().items():
        if a.only and stem not in a.only.split(","):
            continue
        path = find_media(gtd, media)
        if path is None:
            print(f"{stem}: media not found", flush=True)
            continue
        cuts_, dc = gt(gtd, path)
        for p in a.paths.split(","):
            t0 = time.time()
            det = run_path(p, path)
            dt = time.time() - t0
            r = score(cuts_, dc, det)
            rows.append({"gt": stem, "split": gtd["split"], "path": p, "seconds": round(dt, 2), **r})
            print(f"{gtd['split']:4} {stem:34} {p:11} recall {r['recall']:.3f} precision {r['precision']:.3f} "
                  f"({r['tp']}/{r['n']}, {len(r['false'])} false) {dt:6.1f} s  missed {r['missed']} false {r['false']}",
                  flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
