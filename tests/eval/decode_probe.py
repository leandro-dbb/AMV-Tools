"""Where the indexing decode pass spends its time on this machine (NVIDIA).

    python -m tests.eval.decode_probe VIDEO [--steps ffmpeg,nvc_raw,nvc,onepass,single] [--proxies 0|1]

Each step decodes the whole video once and prints its wall time and
frames/s, from the raw decoder up to the app's full pass:

* ``ffmpeg``  : ``ffmpeg -hwaccel cuda -hwaccel_output_format cuda -f null``
                (NVDEC floor through ffmpeg, frames never leave the GPU)
* ``nvc_raw`` : PyNvVideoCodec demux + decode only (NVDEC floor of the
                tensor pass, per-frame Python loop)
* ``nvc``     : ``nvc_pass.detect_cuts_and_medians`` (downsizing, scan, cut
                network if its weights are there, 240p proxy copy with
                ``--proxies 1``)
* ``onepass`` : the ffmpeg single pass (``onepass``), same outputs
* ``single``  : ``pipeline._single_pass_decode`` (nvc + motion), what the
                app runs

Written for the Windows / Linux comparison of docs/reports/windows-rtx3090.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _frames(video: str) -> int:
    from backend.indexing import cuts
    info = cuts.probe_video(video)
    return int(round(info["duration_ms"] / 1000.0 * info["fps"]))


def step_ffmpeg(video: str) -> dict:
    from backend.indexing import cuts
    t0 = time.perf_counter()
    subprocess.run([cuts._ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-hwaccel", "cuda",
                    "-hwaccel_output_format", "cuda", "-i", video, "-an", "-sn", "-f", "null", "-"],
                   check=True, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    return {"wall_s": time.perf_counter() - t0}


def step_nvc_raw(video: str) -> dict:
    import torch
    from backend.indexing import nvc_pass
    nvc = nvc_pass._module()
    torch.cuda.init()
    ctx, _ = nvc_pass._torch_cuda_handles(torch)
    stream = torch.cuda.Stream()
    t0 = time.perf_counter()
    dmx = nvc.CreateDemuxer(filename=video)
    dec = nvc.CreateDecoder(gpuid=0, codec=dmx.GetNvCodecId(), cudacontext=ctx,
                            cudastream=int(stream.cuda_stream), usedevicememory=True)
    n = 0
    t_dec = 0.0
    for pkt in dmx:
        a = time.perf_counter()
        frames = dec.Decode(pkt)
        t_dec += time.perf_counter() - a
        n += len(frames)
    stream.synchronize()
    return {"wall_s": time.perf_counter() - t0, "frames": n, "in_decode_call_s": round(t_dec, 2)}


def step_nvc(video: str) -> dict:
    from backend.indexing import nvc_pass
    t0 = time.perf_counter()
    res = nvc_pass.detect_cuts_and_medians(video, 0.40)
    if res is None:
        return {"error": "not handled"}
    return {"wall_s": time.perf_counter() - t0, "scenes": len(res[0])}


def step_onepass(video: str) -> dict:
    from backend.indexing import onepass
    t0 = time.perf_counter()
    scenes, _ = onepass.detect_cuts_and_medians(video, 0.40)
    return {"wall_s": time.perf_counter() - t0, "scenes": len(scenes)}


def step_single(video: str) -> dict:
    from backend.indexing import pipeline
    t0 = time.perf_counter()
    scenes, _m, _acc = pipeline._single_pass_decode(video, 0.40, lambda: False)
    return {"wall_s": time.perf_counter() - t0, "scenes": len(scenes)}


STEPS = {"ffmpeg": step_ffmpeg, "nvc_raw": step_nvc_raw, "nvc": step_nvc, "onepass": step_onepass,
         "single": step_single}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--steps", default="ffmpeg,nvc_raw,nvc,onepass,single")
    p.add_argument("--proxies", type=int, default=0, help="1: the passes also write the 240p proxy copy")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--direct", action="store_true", help="with --proxies 1: encode the proxies themselves")
    p.add_argument("--json")
    a = p.parse_args()
    from backend.indexing import cutnet, onepass
    tmp = None
    if a.proxies:
        tmp = tempfile.mkdtemp(prefix="amv-probe-")
        onepass.set_proxy_cache(Path(tmp))
        if a.direct:
            onepass.set_proxy_direct({"bitrate": 500_000})
    n = _frames(a.video)
    out = {"video": a.video, "frames": n, "platform": sys.platform, "cutnet": cutnet.enabled(),
           "proxies": bool(a.proxies), "steps": {}}
    print(f"{a.video}: {n} frames, cut network {'on' if out['cutnet'] else 'off'}, "
          f"proxy copy {'on' if a.proxies else 'off'}", flush=True)
    for _ in range(a.repeat):
        for name in a.steps.split(","):
            try:
                r = STEPS[name](a.video)
            except Exception as exc:
                r = {"error": repr(exc)[:300]}
            if "wall_s" in r:
                r["fps"] = round(n / r["wall_s"], 1)
                r["wall_s"] = round(r["wall_s"], 2)
            out["steps"].setdefault(name, []).append(r)
            print(f"  {name:8s} {r}", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    if tmp:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    os.environ.setdefault("AMV_AUTOTUNE", "0")
    main()
