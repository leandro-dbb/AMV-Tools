"""Time the hover-proxy pass of one video, and variants of it (NVIDIA).

    python -m tests.eval.proxy_probe VIDEO [--variants default,c6,async] [--keep DIR]

Runs the indexing decode once (``nvc_pass``, 240p proxy copy written as in
the app), then the app's batched proxy pass (``AppState._proxies_batched``)
on that copy for each variant, with the scene edges the decode found:

* ``default``    : what the app does (3 chunks, NVENC ``-tune ll``)
* ``c1`` / ``c6``: 1 / 6 chunks side by side
* ``async``      : NVENC allowed to keep frames in flight (``-delay 8``)
* ``hwaccel``    : ffmpeg's NVDEC hwaccel instead of ``h264_cuvid``
* ``sw_cN``      : software decode of the copy, N chunks side by side

Prints wall time and segment count per variant (every variant must make one
proxy per scene). Written for docs/reports/windows-rtx3090.md.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

VARIANTS = {
    "default": {},
    "c1": {"AMV_PROXY_CHUNKS": "1"},
    "c6": {"AMV_PROXY_CHUNKS": "6"},
    "c8": {"AMV_PROXY_CHUNKS": "8"},
    "async": {"AMV_PROXY_ENC_ARGS": "-delay 8"},
    "async_c6": {"AMV_PROXY_ENC_ARGS": "-delay 8", "AMV_PROXY_CHUNKS": "6"},
    "hwaccel": {"AMV_PROXY_DEC_ARGS": "-hwaccel cuda -hwaccel_output_format cuda"},
    # software decode of the 240p copy, chunks side by side (no NVDEC latency)
    "sw_c3": {"AMV_PROXY_DEC_ARGS": "-threads 2", "AMV_PROXY_CHUNKS": "3"},
    "sw_c6": {"AMV_PROXY_DEC_ARGS": "-threads 1", "AMV_PROXY_CHUNKS": "6"},
    "sw_c8": {"AMV_PROXY_DEC_ARGS": "-threads 1", "AMV_PROXY_CHUNKS": "8"},
    "sw_c12": {"AMV_PROXY_DEC_ARGS": "-threads 1", "AMV_PROXY_CHUNKS": "12"},
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--variants", default="default,c1,c6,async")
    p.add_argument("--keep", help="folder for the 240p copy (reused when present)")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--direct", action="store_true",
                   help="the decode encodes the proxies themselves (onepass.set_proxy_direct)")
    p.add_argument("--psnr", type=int, default=0, help="PSNR of N proxies against the source (240p, 15 fps)")
    a = p.parse_args()
    os.environ.setdefault("AMV_AUTOTUNE", "0")
    from backend.db import schema
    from backend.indexing import nvc_pass, onepass
    from backend.state import AppState

    cache = Path(a.keep or tempfile.mkdtemp(prefix="amv-pxprobe-"))
    cache.mkdir(parents=True, exist_ok=True)
    onepass.set_proxy_cache(cache)
    if a.direct:
        onepass.set_proxy_direct({"bitrate": 500_000})
    scenes_file = cache / "scenes.txt"
    have = onepass.proxy_direct_for(a.video, cache) if a.direct else onepass.proxy_source_for(a.video, cache)
    if have is None or not scenes_file.exists():
        t0 = time.perf_counter()
        scenes, _ = nvc_pass.detect_cuts_and_medians(a.video, 0.40)
        scenes_file.write_text("\n".join(f"{s} {e}" for s, e in scenes))
        print(f"decode + 240p copy: {time.perf_counter() - t0:.1f} s, {len(scenes)} scenes", flush=True)
    scenes = [tuple(map(int, ln.split())) for ln in scenes_file.read_text().split("\n") if ln]
    rows = [(i + 1, s, e) for i, (s, e) in enumerate(scenes)]
    db = str(cache / "probe.db")
    schema.init_db(db)
    fake = SimpleNamespace(_proxy_children={}, primary_db=db)
    for _ in range(a.repeat):
        for name in a.variants.split(","):
            env = VARIANTS[name]
            old = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                with tempfile.TemporaryDirectory(prefix="px-", dir=str(cache)) as out:
                    done: set = set()
                    t0 = time.perf_counter()
                    try:
                        n = AppState._proxies_batched(fake, a.video, rows, Path(out), "medium", done,
                                                      source_cache=cache, chunks=3)
                        err = ""
                    except Exception as exc:
                        n, err = 0, repr(exc)[:200]
                    dt = time.perf_counter() - t0
                    size = sum(f.stat().st_size for f in Path(out).glob("scene_*.mp4"))
                    q = _psnr(a.video, rows, Path(out), a.psnr) if a.psnr else ""
                print(f"  {name:9s} {dt:6.2f} s  {n}/{len(rows)} proxies  {size / 1e6:.1f} MB {q} {err}", flush=True)
            finally:
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v


def _psnr(video: str, rows, out: Path, k: int) -> str:
    """Mean PSNR (dB) and frame counts of ``k`` proxies spread over the
    video against the source scaled the same way (240p, 15 fps)."""
    import re
    import subprocess
    from backend.indexing import cuts
    vals, frames = [], []
    for sid, s, e in rows[:: max(1, len(rows) // k)][:k]:
        f = out / f"scene_{sid}.mp4"
        if not f.exists():
            continue
        r = subprocess.run([cuts._ffmpeg_path(), "-hide_banner", "-nostats", "-i", str(f),
                            "-ss", f"{s / 1000:.3f}", "-t", f"{(e - s) / 1000:.3f}", "-i", video,
                            "-filter_complex", "[1:v]fps=15,scale=-2:240[r];[0:v][r]psnr", "-f", "null", "-"],
                           capture_output=True, text=True)
        m = re.search(r"average:([0-9.]+|inf)", r.stderr)
        n = re.findall(r"frame=\s*(\d+)", r.stderr)
        if m and m.group(1) != "inf":
            vals.append(float(m.group(1)))
        if n:
            frames.append(int(n[-1]))
    return f"PSNR {sum(vals) / max(1, len(vals)):.2f} dB over {len(vals)} ({sum(frames)} frames)"


if __name__ == "__main__":
    main()
