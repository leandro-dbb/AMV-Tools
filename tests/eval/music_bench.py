"""Music analysis benchmark: beat / downbeat / drop precision-recall + speed.

    python tests/eval/music_bench.py ~/dev/amv-testdata/music [--json out.json]

For every ``<name>.wav|mp3`` that has a ``<name>.json`` ground truth next to
it ({"beats": [...], "downbeats": [...], "drops": [...]}, seconds), prints
precision / recall / F1 with the usual 70 ms beat tolerance; tracks without
ground truth only get their tempo, sections and timing.  Synthetic tracks
come from ``tests/eval/music_synth.py``.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.music.analysis import analyze_audio  # noqa: E402

BEAT_TOL = 0.070
DROP_TOL = 0.5


def match(est: list[float], ref: list[float], tol: float) -> dict:
    est_a = np.asarray(sorted(est), dtype=float)
    ref_a = np.asarray(sorted(ref), dtype=float)
    used = np.zeros(len(est_a), dtype=bool)
    hits = 0
    errors = []
    for r in ref_a:
        if not len(est_a):
            break
        d = np.abs(est_a - r)
        d[used] = np.inf
        i = int(np.argmin(d))
        if d[i] <= tol:
            used[i] = True
            hits += 1
            errors.append(float(est_a[i] - r))
    p = hits / len(est_a) if len(est_a) else (1.0 if not len(ref_a) else 0.0)
    r = hits / len(ref_a) if len(ref_a) else (1.0 if not len(est_a) else 0.0)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
            "n_est": int(len(est_a)), "n_ref": int(len(ref_a)),
            "mean_abs_err_ms": round(1000 * float(np.mean(np.abs(errors))), 1) if errors else None}


def run(folder: Path) -> list[dict]:
    rows = []
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a"))
    for path in files:
        t0 = time.perf_counter()
        c0 = time.process_time()
        k0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        data = analyze_audio(str(path))
        wall = time.perf_counter() - t0
        k1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        # CPU seconds (ours + ffmpeg's): stable even when the box is loaded.
        cpu = time.process_time() - c0 + (k1.ru_utime - k0.ru_utime) + (k1.ru_stime - k0.ru_stime)
        row = {
            "track": path.stem,
            "duration_s": round(data["duration_ms"] / 1000, 1),
            "seconds": round(wall, 2),
            "cpu_seconds": round(cpu, 2),
            "bpm": data["bpm"],
            "variable_tempo": data["variable_tempo"],
            "drops_s": [round(x / 1000, 2) for x in data["drops_ms"]],
            "sections": [(s["type"], round(s["start_ms"] / 1000, 1)) for s in data["sections"]],
        }
        truth_path = path.with_suffix(".json")
        if truth_path.exists():
            truth = json.loads(truth_path.read_text())
            beats = [b / 1000 for b in data["beats_ms"]]
            downs = [b / 1000 for b in data["downbeats_ms"]]
            drops = [b / 1000 for b in data["drops_ms"]]
            row["truth_bpm"] = truth.get("bpm")
            if truth.get("beats"):
                row["beats"] = match(beats, truth["beats"], BEAT_TOL)
            if truth.get("downbeats"):
                row["downbeats"] = match(downs, truth["downbeats"], BEAT_TOL)
            if "drops" in truth:
                row["drops"] = match(drops, truth["drops"], DROP_TOL)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--json")
    args = ap.parse_args()
    # Warm librosa's numba kernels once so the timings are per-song costs.
    from backend.music.analysis import analyze_signal
    analyze_signal(np.random.default_rng(0).standard_normal(22050 * 8).astype(np.float32) * 0.1)
    rows = run(Path(args.folder).expanduser())
    for r in rows:
        line = (f"{r['track']:<26} {r['duration_s']:>6}s  wall={r['seconds']:>5}s"
                f" cpu={r['cpu_seconds']:>5}s  bpm={r['bpm']:<7}")
        if r.get("truth_bpm"):
            line += f"(ref {r['truth_bpm']})"
        for k in ("beats", "downbeats", "drops"):
            if k in r:
                m = r[k]
                line += f"  {k} P={m['precision']} R={m['recall']}"
                if m["mean_abs_err_ms"] is not None and k != "drops":
                    line += f" err={m['mean_abs_err_ms']}ms"
                elif k == "drops" and m["mean_abs_err_ms"] is not None:
                    line += f" err={m['mean_abs_err_ms']}ms"
        print(line)
        print("    drops:", r["drops_s"], "| sections:", r["sections"])
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
