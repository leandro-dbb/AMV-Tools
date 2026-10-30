"""Real end-to-end check of the music sync against a running sidecar.

    python tests/eval/music_real.py --port 8763 --song song.mp3 --out DIR

Analyses the song, proposes an edit from the indexed library, exports it as
FCP 7 XML + markers-only, renders the preview MP4, then measures how far
the cuts land from the beats:

* in the exported timeline (clip record starts, in sequence frames);
* in the rendered MP4, from the cuts ffmpeg's scene detector actually sees.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

import numpy as np


def call(port: int, path: str, body: dict | None = None, timeout: float = 600) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def nearest(values: np.ndarray, t: float) -> float:
    return float(values[np.argmin(np.abs(values - t))] - t) if values.size else float("nan")


def stats(errors_ms: list[float]) -> dict:
    a = np.abs(np.asarray(errors_ms, dtype=float))
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "mean_ms": round(float(a.mean()), 1), "median_ms": round(float(np.median(a)), 1),
            "p95_ms": round(float(np.percentile(a, 95)), 1), "max_ms": round(float(a.max()), 1),
            "within_70ms": round(float(np.mean(a <= 70)), 3)}


def detected_cuts(mp4: Path, threshold: float) -> list[float]:
    ff = shutil.which("ffmpeg") or "ffmpeg"
    proc = subprocess.run([ff, "-hide_banner", "-nostdin", "-i", str(mp4), "-map", "0:v:0",
                           "-vf", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
                          capture_output=True, text=True)
    return [float(m) for m in re.findall(r"pts_time:([0-9.]+)", proc.stderr)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8763)
    ap.add_argument("--song", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--density", type=float, default=0.5)
    ap.add_argument("--query", default="")
    ap.add_argument("--drop-query", default="")
    ap.add_argument("--scene-threshold", type=float, default=0.3)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"song": args.song}

    t0 = time.perf_counter()
    track = call(args.port, "/api/music/analyze", {"path": args.song})
    report["analysis"] = {"seconds_wall": round(time.perf_counter() - t0, 2), "bpm": track["bpm"],
                          "beats": len(track["beats_ms"]), "drops_s": [d / 1000 for d in track["drops_ms"]],
                          "sections": [(s["type"], s["start_ms"] / 1000) for s in track["sections"]]}
    beats = np.asarray(track["beats_ms"], dtype=float) / 1000
    marks = np.asarray(sorted(set(track["beats_ms"]) | set(track["strong_onsets_ms"])
                              | {s["start_ms"] for s in track["sections"]}), dtype=float) / 1000

    section_queries = {}
    if args.drop_query:
        section_queries = {str(s["index"]): args.drop_query for s in track["sections"] if s["type"] == "drop"}
    t0 = time.perf_counter()
    prop = call(args.port, "/api/music/propose", {
        "hash": track["hash"], "seed": 1,
        "constraints": {"density": args.density, "query": args.query, "section_queries": section_queries}})
    clips = prop["clips"]
    report["proposal"] = {"seconds_wall": round(time.perf_counter() - t0, 2), "clips": len(clips),
                          "library": prop["library"], "warnings": prop["warnings"],
                          "unique_shots": len({c["scene_id"] for c in clips}),
                          "series": sorted({c["group"] or "-" for c in clips})}
    (out / "proposal.json").write_text(json.dumps(prop, indent=1))
    plan_cuts = [c["start_ms"] / 1000 for c in clips[1:]]
    report["planned_cut_vs_beat"] = stats([1000 * nearest(marks, t) for t in plan_cuts])

    xml_path = out / "edit.xml"
    res = call(args.port, "/api/music/export/timeline", {
        "hash": track["hash"], "clips": clips, "format": "fcp7xml", "output_path": str(xml_path),
        "beat_markers": True, "lang": "fr"})
    report["export"] = res
    root = ET.fromstring(xml_path.read_text().split("\n", 2)[2])
    tb = int(root.findtext("./sequence/rate/timebase"))
    ntsc = root.findtext("./sequence/rate/ntsc") == "TRUE"
    fps = Fraction(tb * 1000, 1001) if ntsc else Fraction(tb)
    starts = [int(ci.findtext("start")) for ci in root.findall("./sequence/media/video/track/clipitem")]
    report["timeline_cut_vs_beat"] = stats([1000 * nearest(marks, float(f / fps)) for f in starts[1:]])
    report["timeline_cut_vs_plan"] = stats([1000 * (float(f / fps) - c["start_ms"] / 1000)
                                            for f, c in zip(starts[1:], clips[1:])])
    for fmt in ("fcp7xml", "edl", "csv"):
        call(args.port, "/api/music/export/markers", {
            "hash": track["hash"], "format": fmt, "output_path": str(out / f"markers-{fmt}"), "lang": "fr"})

    t0 = time.perf_counter()
    job = call(args.port, "/api/music/preview", {"hash": track["hash"], "clips": clips})
    while True:
        st = call(args.port, f"/api/music/preview/{job['id']}")
        if st["status"] in ("done", "failed"):
            break
        time.sleep(2)
    report["preview"] = {"status": st["status"], "error": st["error"],
                         "seconds_wall": round(time.perf_counter() - t0, 1)}
    if st["status"] == "done":
        mp4 = out / "apercu.mp4"
        shutil.copyfile(st["output"], mp4)
        cuts = detected_cuts(mp4, args.scene_threshold)
        plan = np.asarray(plan_cuts)
        report["preview_detected_cuts"] = len(cuts)
        det = np.asarray(cuts)
        frame_ms = 1000 / 23.976
        # Each planned cut: is there a visible cut within one frame in the
        # render, and how far?  (A cut between two moments of the same shot
        # can be invisible to the detector.)
        seen = [1000 * nearest(det, t) for t in plan]
        hit = [e for e in seen if abs(e) <= frame_ms + 1]
        report["preview_planned_cuts_seen"] = round(len(hit) / max(1, len(plan)), 3)
        report["preview_seen_cut_vs_plan"] = stats(hit)
        report["preview_seen_cut_vs_beat"] = stats([1000 * nearest(marks, t) for t in det
                                                    if abs(nearest(plan, t)) * 1000 <= frame_ms + 1])
        # Detected cuts far from every planned cut = cuts inside the source
        # "shots" (missed by the indexer's shot detection).
        extra = [t for t in det if abs(nearest(plan, t)) * 1000 > 2 * frame_ms]
        report["preview_cuts_inside_shots"] = len(extra)
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
