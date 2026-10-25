"""The measurements of the automatic tuning (real hardware, ~30-60 s).

Each ``measure_*`` returns a setting record::

    {"value": <chosen>, "default": <constant it replaces>, "unit": "...",
     "measures": {<candidate>: <time>}, "note": "..."}

or raises ``Skip`` (nothing to choose on this machine: the default stays).
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import choose

log = logging.getLogger(__name__)

CLIP_S = 10.0
ROUNDS = 2            # macOS: every session decodes this many clips per candidate
NVIDIA_JOBS = 4       # NVIDIA: the same 4 decodes (a small queue) for every candidate
NVIDIA_DECODES = (1, 2, 4)
NVIDIA_LOOPS = 3      # the 10 s extract looped to 30 s: amortises ffmpeg/CUDA start-up
MAC_SESSIONS = (2, 4, 8, 12)   # 1 session is the known-slow case (~6x real time)


class Skip(Exception):
    """Nothing to measure for this setting here (reason in the message)."""


def _flags() -> int:
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _ffmpeg(args: list[str]) -> None:
    from ..indexing import cuts
    subprocess.run([cuts._ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", *args],
                   check=True, capture_output=True, timeout=120, creationflags=_flags())


def library_video() -> Optional[str]:
    """A completed video of the primary library that still exists, the
    longest first (an episode, not a 5 s test clip)."""
    try:
        from ..db import schema
        from ..state import get_state
        with schema.get_conn(get_state().primary_db) as conn:
            rows = conn.execute("SELECT filepath FROM videos WHERE status = 'completed' "
                                "ORDER BY duration_ms DESC LIMIT 20").fetchall()
    except Exception as exc:
        log.debug("autotune: no library video (%s)", exc)
        return None
    return next((r[0] for r in rows if r[0] and os.path.isfile(r[0])), None)


def make_clip(workdir: Path, source: Optional[str], loops: int = 1) -> tuple[Path, str]:
    """(clip, origin): ``CLIP_S`` of ``source`` (stream copy from its middle)
    or a synthetic 1080p H.264 clip, repeated ``loops`` times."""
    base = workdir / "extract.mkv"
    origin = "synthetic"
    if source:
        try:
            from ..indexing import cuts
            dur = cuts.probe_video(source)["duration_ms"] / 1000.0
            start = max(0.0, dur / 2 - CLIP_S / 2)
            _ffmpeg(["-ss", f"{start:.3f}", "-i", source, "-t", f"{CLIP_S}", "-map", "0:v:0",
                     "-c", "copy", str(base)])
            if cuts.probe_video(str(base))["duration_ms"] >= CLIP_S * 800:
                origin = "library"
        except Exception as exc:
            log.info("autotune: extract from %s failed (%s); synthetic clip", source, exc)
    if origin == "synthetic":
        # cuts every 2 s (colour changes) so the detector has scenes to close
        _ffmpeg(["-f", "lavfi", "-i", f"testsrc2=size=1920x1080:rate=24000/1001:duration={CLIP_S}",
                 "-vf", "hue=h=t*36", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                 "-g", "48", str(base)])
    if loops <= 1:
        return base, origin
    looped = workdir / "looped.mkv"
    _ffmpeg(["-stream_loop", str(loops - 1), "-i", str(base), "-c", "copy", str(looped)])
    return looped, origin


def time_parallel(run_one: Callable[[], None], candidates, cancelled: Callable[[], bool],
                  rounds: int = ROUNDS, jobs: Optional[int] = None) -> dict:
    """Seconds per job with ``n`` jobs in flight, for each ``n`` (ascending):
    ``jobs`` decodes in total (the same work for every candidate), or
    ``rounds`` per worker. A candidate that errors gets None and ends the
    sweep; so does one 5 % slower than the best so far (past the peak)."""
    fixed = jobs
    out: dict = {}
    best = float("inf")
    for n in sorted(candidates):
        if cancelled():
            raise Skip("cancelled")
        jobs = fixed or n * rounds
        t0 = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=n, thread_name_prefix="autotune-decode") as pool:
                for f in [pool.submit(run_one) for _ in range(jobs)]:
                    f.result()
        except Exception as exc:
            log.warning("autotune: %d parallel decodes failed (%s)", n, exc)
            out[n] = None
            break
        per_job = (time.perf_counter() - t0) / jobs
        out[n] = round(per_job, 4)
        if per_job > best * (1 + choose.MIN_GAIN):
            break
        best = min(best, per_job)
    return out


# ── decode parallelism ──────────────────────────────────────────────────────
def _clip_note(origin: str, seconds: float) -> str:
    if origin == "synthetic":
        return f"synthetic clip, {seconds:.0f} s: default kept until a library video exists"
    return f"library clip, {seconds:.0f} s"


def pick_count(measures: dict, default: int, origin: str):
    """Library clip: the smallest count within 5 % of the fastest. Synthetic
    clip (empty library): it decodes far cheaper than an episode (the RTX
    3090 measured 1 decode 6-8 % faster there, 2 on a real one), so the
    default stays — the next launch measures again once the library has a
    video."""
    if origin == "synthetic" and measures.get(default) is not None:
        return default
    return choose.smallest_within(measures)


def measure_nvdec(workdir: Path, source: Optional[str], cancelled: Callable[[], bool]) -> dict:
    """Single-pass decodes kept in flight by the NVIDIA queue scheduler
    (``pipeline._DecodeScheduler``, was 2 on the RTX 3090)."""
    from ..indexing import onepass, pipeline
    if not onepass.available():
        raise Skip("no NVDEC single pass")
    clip, origin = make_clip(workdir, source, NVIDIA_LOOPS)

    def one():
        res = pipeline._single_pass_decode(str(clip), 0.30, cancelled)
        if res is None or not res[0]:
            raise RuntimeError("empty decode")

    one()   # warm-up: CUDA context, kernels, page cache
    measures = time_parallel(one, NVIDIA_DECODES, cancelled, jobs=NVIDIA_JOBS)
    return {"value": pick_count(measures, 2, origin), "default": 2, "unit": "s/clip",
            "measures": measures, "note": _clip_note(origin, CLIP_S * NVIDIA_LOOPS)}


def measure_videotoolbox(workdir: Path, source: Optional[str], cancelled: Callable[[], bool]) -> dict:
    """VideoToolbox sessions of the chunked decode (``chunked.WORKERS``, 8
    on an M1 Pro): one chunk = the clip."""
    from ..indexing import chunked, cuts
    if sys.platform != "darwin":
        raise Skip("not macOS")
    prof = cuts._ffmpeg_hw_profile()
    if prof is None or prof.get("name") != "videotoolbox":
        raise Skip("no VideoToolbox in ffmpeg")
    clip, origin = make_clip(workdir, source, 1)
    fps = cuts.probe_video(str(clip))["fps"] or 30.0

    def one():
        ch = chunked._Chunk(0, 0, None)
        chunked._run_chunk(str(clip), fps, ch, 0.30, None, None, cancelled, set(), threading.Lock(), True)
        if ch.error is not None:
            raise ch.error

    one()
    measures = time_parallel(one, MAC_SESSIONS, cancelled)
    return {"value": pick_count(measures, chunked.WORKERS, origin), "default": chunked.WORKERS, "unit": "s/clip",
            "measures": measures, "note": _clip_note(origin, CLIP_S)}


# ── frames for the model measurements ───────────────────────────────────────
def sample_frames(workdir: Path, source: Optional[str], n: int) -> list:
    """``n`` RGB frames (PIL) spread over the clip, at the indexing size."""
    from PIL import Image
    clip, _ = make_clip(workdir, source, 1)
    out_dir = workdir / "frames"
    out_dir.mkdir(exist_ok=True)
    k = min(n, 24)
    _ffmpeg(["-i", str(clip), "-vf", f"fps={k / CLIP_S:.4f},scale=640:-2", "-frames:v", str(k),
             str(out_dir / "f%03d.png")])
    imgs = [Image.open(p).convert("RGB") for p in sorted(out_dir.glob("f*.png"))]
    if not imgs:
        raise RuntimeError("no frames decoded")
    # repeat with small shifts so a batch is never n copies of the same image
    return [imgs[i % len(imgs)].rotate(0.5 * (i // len(imgs))) for i in range(n)]


# ── tagger engine (Apple Silicon: MPS vs Core ML) ───────────────────────────
def measure_tagger_engine(frames: list, cancelled: Callable[[], bool]) -> dict:
    from ..models import coreml_tagger
    from ..models.wd_tagger import _TorchTagger
    from ..state import get_state
    tagger = get_state().get_tagger()
    tagger._ensure_loaded()
    sess = tagger._session
    if not isinstance(sess, _TorchTagger):
        raise Skip("wd-tagger not on MPS")
    worker = sess._cml or coreml_tagger.start(*sess.paths)
    if worker is None:
        raise Skip("Core ML unavailable")
    deadline = time.monotonic() + 120      # first conversion ~10 s, then cached
    while not worker.usable() and not worker.failed and time.monotonic() < deadline:
        if cancelled():
            raise Skip("cancelled")
        time.sleep(0.5)
    x = np.stack([tagger._preprocess(im) for im in frames[:32]]).astype(np.float32)
    if not worker.usable():
        raise Skip(f"Core ML worker not ready ({worker.failed or 'timeout'})")
    own, sess._cml = sess._cml, None
    try:
        sess.run(None, {"input": x[:8]})                      # warm-up
        t0 = time.perf_counter()
        ref = sess.run(None, {"input": x})[0]
        t_mps = (time.perf_counter() - t0) / len(x)
        sess._cml = worker
        sess._run_coreml(x[:8])
        t0 = time.perf_counter()
        alt = sess._run_coreml(x)
        t_cml = (time.perf_counter() - t0) / len(x)
    finally:
        sess._cml = own
    same = choose.equivalent(ref, alt)
    measures = {"mps": round(t_mps, 5), "coreml": round(t_cml, 5) if same else None}
    value = choose.against_reference(measures, "mps")
    if value == "coreml" and own is None:
        sess._cml = worker                 # keep the worker that just proved itself
    elif value == "mps" and worker is not None:
        if own is worker:
            sess._cml = None
        worker.close()
    note = "" if same else f"Core ML output differs (max Δ {float(np.abs(ref - alt).max()):.4f})"
    return {"value": value, "default": "coreml", "unit": "s/image", "measures": measures, "note": note}


# ── tagger + SigLIP batch size ──────────────────────────────────────────────
def batch_candidates(backend: str, vram_gb: Optional[float], mem_gb: Optional[float]) -> tuple[list, int]:
    """(candidates, reference): the memory tier's batch is the ceiling and
    the reference (``pipeline._effective_batch``)."""
    if backend == "cuda":
        v = vram_gb or 0
        cap = 64 if v >= 20 else 32 if v >= 11 else 16 if v >= 7 else 8
        return [b for b in (8, 16, 32, 64) if b <= cap], cap
    if backend == "mps":
        cap = 16 if (mem_gb or 0) >= 16 else 8
        return [b for b in (4, 8, 16) if b <= cap], 8
    raise Skip(f"no batch tuning on {backend}")


def measure_batch(frames: list, cancelled: Callable[[], bool]) -> dict:
    from ..state import get_state
    st = get_state()
    dev = st.device
    mem_gb = None
    if dev.backend == "mps":
        try:
            mem_gb = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                        timeout=5).stdout.strip()) / 2**30
        except Exception:
            mem_gb = None
    candidates, reference = batch_candidates(dev.backend, getattr(dev, "vram_gb", None), mem_gb)
    tagger, siglip = st.get_tagger(), st.get_siglip()
    imgs = frames[:4 * max(candidates)]
    measures: dict = {}
    for b in candidates:
        if cancelled():
            raise Skip("cancelled")
        try:
            tagger.embed_images(imgs[:b])
            siglip.embed_images(imgs[:b])                    # warm-up at this shape
            t0 = time.perf_counter()
            for k in range(0, len(imgs), b):
                tagger.embed_images(imgs[k:k + b])
                siglip.embed_images(imgs[k:k + b])
            measures[b] = round((time.perf_counter() - t0) / len(imgs), 5)
        except Exception as exc:          # out of memory: larger ones won't fit either
            log.warning("autotune: batch %d failed (%s)", b, exc)
            measures[b] = None
            break
    return {"value": choose.against_reference(measures, reference), "default": reference,
            "unit": "s/image", "measures": measures,
            "note": f"wd-tagger + SigLIP, {len(imgs)} frames"}


def tensorrt_note() -> dict:
    """NVIDIA: PyTorch vs TensorRT. The app has no TensorRT path for the
    default models (SigLIP NaFlex refuses it, see ``state``), so this only
    records whether it is installed."""
    import importlib.util
    installed = importlib.util.find_spec("torch_tensorrt") is not None
    raise Skip("TensorRT installed, no compatible model path: PyTorch kept" if installed
               else "TensorRT not installed: PyTorch kept")


def workdir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="amv-autotune-")
