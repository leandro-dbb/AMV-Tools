"""Job bodies for the Finishing toolbox + 5-frame speed estimate.

Each job takes a *source* — a library scene (optionally trimmed in the
MiniEditor) or any video file on disk — and writes one output next to the
regular exports (``<export folder>/finish/``).
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .jobs import JobContext
from .video_io import ClipInfo, FrameReader, FrameWriter, probe_clip, read_frames, suffix_for_codec

log = logging.getLogger(__name__)

KINDS = ("upscale", "interpolate", "depth", "matte")

_cache_lock = threading.Lock()
_cache: dict[tuple, object] = {}


# ── model cache ─────────────────────────────────────────────────────────────
def _device():
    from ..state import get_state
    return get_state().device


def _free_other_models() -> None:
    """Evict the indexing + mask models before a heavy finishing pass — they
    lazy-reload on next use (same policy as Mask Mode)."""
    try:
        from ..state import get_state
        st = get_state()
        with st._models_lock:
            st._offload_others_for_mask()
            st._offload_other_mask_engines(keep="")
    except Exception as exc:
        log.debug("model eviction skipped: %s", exc)


def _get_model(key: tuple, factory: Callable[[], object]):
    with _cache_lock:
        if key not in _cache:
            # One finishing model resident at a time (and its Core ML worker
            # process goes with it now, not whenever it is collected).
            for old in _cache.values():
                close = getattr(old, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            _cache.clear()
            _free_other_models()
            _cache[key] = factory()
        return _cache[key]


def release_models() -> None:
    import gc

    with _cache_lock:
        had = bool(_cache)
        for old in _cache.values():
            close = getattr(old, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        _cache.clear()
    if had:
        gc.collect()
        try:
            from ..models.device import empty_device_cache
            empty_device_cache(_device().backend)
        except Exception:
            pass


# ── tier defaults (backend/models/tiers.py) ─────────────────────────────────
def _tier_models() -> dict:
    """The selected tier's model choice (defaults when no state exists yet)."""
    from ..models import tiers
    try:
        from .. import state as state_mod
        st = state_mod._state
        cfg = st.settings.get("models", {}) if st is not None else {}
        return tiers.resolve(cfg, _device())["models"]
    except Exception as exc:
        log.debug("tier defaults unavailable: %s", exc)
        return dict(tiers.DEFAULT_CHOICE)


def model_for(kind: str, p: dict) -> str:
    """``params["model"]``, else the tier's model for ``kind``."""
    from ..models import tiers
    return p.get("model") or _tier_models().get(kind) or tiers.DEFAULT_CHOICE[kind]


def depth_input_size(p: dict) -> Optional[int]:
    v = p.get("depth_input_size") or _tier_models().get("depth_input_size")
    return int(v) if v else None


def _rife(dev, model_id: str, scale: float):
    from .interpolate import RIFE_MODEL, RifeInterpolator
    # model_id only when it is not the historical default: works with both
    # the current class and the per-model one
    kw = {"model_id": model_id} if model_id != RIFE_MODEL else {}
    return _get_model(("rife", model_id, dev.backend, scale), lambda: RifeInterpolator(dev, scale=scale, **kw))


def _depth(dev, model_id: str, input_size: Optional[int]):
    from .depth import DEPTH_MODEL, DepthEstimator
    from .vda import VDA_MODEL, VideoDepthEstimator
    kw: dict = {"model_id": model_id} if model_id not in (DEPTH_MODEL, VDA_MODEL) else {}
    if input_size:
        kw["input_size"] = int(input_size)
    cls = VideoDepthEstimator if model_id == VDA_MODEL else DepthEstimator
    return _get_model(("depth", model_id, dev.backend, input_size), lambda: cls(dev, **kw))


# ── memory guard (depth / cutout hold the whole clip in RAM) ────────────────
# Peak bytes per working-resolution pixel, per frame, from the copies alive
# at the same time:
# - depth (model resolution): float16 map (2) + RGB small (3) + the float32
#   copy given to ``stabilize_depth`` (4) + aligned list (4) + ``np.stack``
#   (4) + normalised stack (4) + normalisation temporary (4) + stabilised
#   output (4) = 29;
# - cutout (frame scaled to ``mask_max_dim``): RGB (3) + float32 masks (4) +
#   ``np.stack`` (4) + ``postprocess_stack`` copy (4) + the stabiliser's
#   float32 copy (4) + stabilised output (4) = 23.
_BYTES_PER_PX = {"depth": 29, "matte": 23}
RAM_BUDGET_FRACTION = 0.5       # of the RAM available when the job is queued
MAX_SECONDS_NO_RAM_INFO = 180.0  # fallback cap when free RAM is unknown


class ClipTooLong(ValueError):
    """The clip would not fit in RAM for a depth / cutout job."""


def available_ram() -> Optional[int]:
    """Bytes of RAM available now, None when it cannot be read."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def mask_max_dim(models_cfg: Optional[dict]) -> int:
    return max(360, min(2160, int((models_cfg or {}).get("mask_max_dim", 1080) or 1080)))


def work_size(kind: str, clip: ClipInfo, p: dict, models_cfg: Optional[dict] = None) -> tuple[int, int]:
    """(w, h) the depth / cutout job keeps in memory for each frame."""
    w, h = max(1, clip.width), max(1, clip.height)
    if kind == "depth":
        # Depth Anything resizes the short side to the input size
        s = float(depth_input_size(p) or 518) / min(w, h)
    else:
        s = min(1.0, mask_max_dim(models_cfg) / float(max(w, h)))
    return max(1, int(round(w * s))), max(1, int(round(h * s)))


def memory_needed(kind: str, clip: ClipInfo, p: dict, models_cfg: Optional[dict] = None) -> int:
    ww, wh = work_size(kind, clip, p, models_cfg)
    return clip.n_frames * ww * wh * _BYTES_PER_PX[kind]


def check_memory(kind: str, clip: ClipInfo, p: dict, models_cfg: Optional[dict] = None) -> None:
    """:class:`ClipTooLong` when a depth / cutout job on ``clip`` would need
    more than :data:`RAM_BUDGET_FRACTION` of the available RAM (the OS would
    kill the whole sidecar, not just the job)."""
    if kind not in _BYTES_PER_PX:
        return
    what = "depth" if kind == "depth" else "cutout"
    dur = clip.duration_s
    avail = available_ram()
    if avail is None:
        if dur > MAX_SECONDS_NO_RAM_INFO:
            raise ClipTooLong(
                f"This scene is too long for {what} ({dur:.0f} s, the limit is "
                f"{MAX_SECONDS_NO_RAM_INFO:.0f} s). Split it or shorten the range.")
        return
    need = memory_needed(kind, clip, p, models_cfg)
    if need > avail * RAM_BUDGET_FRACTION:
        gb = 1024.0 ** 3
        raise ClipTooLong(
            f"This scene is too long for {what} ({dur:.0f} s: needs ~{need / gb:.1f} GB of RAM, "
            f"~{avail / gb:.1f} GB available, at most half of it is used). "
            f"Split it or shorten the range.")


# ── source / output resolution ──────────────────────────────────────────────
def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "clip"


def resolve_source(src: dict) -> tuple[ClipInfo, str]:
    """``{"scene_id", "start_ms"?, "end_ms"?}`` or ``{"path", ...}`` ->
    (ClipInfo, output name stem)."""
    if src.get("scene_id") is not None:
        from ..db import queries, schema
        from ..state import get_state

        with schema.get_conn(get_state().primary_db) as conn:
            scene = queries.get_scene(conn, int(src["scene_id"]))
        if not scene:
            raise FileNotFoundError(f"scene {src['scene_id']} not found")
        path = scene["filepath"]
        start = src.get("start_ms", scene["start_ms"])
        end = src.get("end_ms", scene["end_ms"])
        stem = f"{scene.get('display_name') or Path(path).stem}_scene{scene['id']}"
    elif src.get("path"):
        path = str(src["path"])
        start, end = src.get("start_ms"), src.get("end_ms")
        stem = Path(path).stem
    else:
        raise ValueError("source needs scene_id or path")
    if not Path(path).exists():
        raise FileNotFoundError(f"source file missing: {path}")
    return probe_clip(path, start, end), _sanitize(stem)


def output_dir() -> Path:
    from ..paths import user_data_dir
    from ..state import get_state

    folder = get_state().settings.get("export", {}).get("output_folder") or ""
    base = Path(folder) if folder else user_data_dir() / "exports"
    d = base / "finish"
    d.mkdir(parents=True, exist_ok=True)
    return d


def unique_path(d: Path, stem: str, suffix: str) -> Path:
    p = d / f"{stem}{suffix}"
    n = 1
    while p.exists():
        n += 1
        p = d / f"{stem}_{n}{suffix}"
    return p


def _export_settings() -> dict:
    from ..state import get_state
    return dict(get_state().settings.get("export", {}))


# ── job dispatch ────────────────────────────────────────────────────────────
def run_job(ctx: JobContext) -> Optional[str]:
    job = ctx.job
    clip, stem = resolve_source(job.params.get("source") or {})
    out_dir = Path(job.params["output_dir"]) if job.params.get("output_dir") else output_dir()
    body = {"upscale": _run_upscale, "interpolate": _run_interpolate,
            "depth": _run_depth, "matte": _run_matte}.get(job.kind)
    if body is None:
        raise ValueError(f"unknown finishing job kind {job.kind!r}")
    ctx.progress(0.0, "loading model", force=True)
    return body(ctx, clip, stem, out_dir, job.params)


def _overlap() -> bool:
    """Decode-ahead / encode-behind threads (see ``overlap``): CUDA, and
    Apple Silicon, where the models run in Core ML / Neural Engine worker
    processes and the job thread would otherwise wait on them between
    decoding and encoding. CPU / ROCm / DirectML keep the sequential loop
    they were tuned with."""
    try:
        return _device().backend in ("cuda", "mps")
    except Exception:
        return False


def _i420_in_writer() -> bool:
    """RGB -> I420 by OpenCV in the writer thread (CUDA: ffmpeg's single-
    threaded swscale capped 4K NVENC encodes). Elsewhere ffmpeg converts,
    as before (bit-identical outputs)."""
    try:
        return _device().backend == "cuda"
    except Exception:
        return False


def _queue_depth(frame_bytes: int, budget: int = 256 * 2**20) -> int:
    """Frames in flight per overlap queue: up to 4, within ~256 MB (an 8K
    RGB frame is 100 MB — the x4 upscale must not queue a gigabyte)."""
    return max(1, min(4, budget // max(1, frame_bytes)))


def _writer_for(ctx: JobContext, path: Path, **kw) -> FrameWriter:
    w = FrameWriter(str(path), yuv420_in=_i420_in_writer(), **kw)
    if _overlap():
        from .overlap import AsyncWriter
        w = AsyncWriter(w, depth=_queue_depth(w.width * w.height * w.channels * np.dtype(w.dtype).itemsize))
    ctx.on_cancel_cleanup(w.abort)
    return w


def _frames(reader):
    if _overlap():
        from .overlap import prefetch
        return prefetch(reader, depth=_queue_depth(reader.width * reader.height * 3))
    return reader


def _reader_for(ctx: JobContext, clip: ClipInfo, **kw) -> FrameReader:
    r = FrameReader(clip, **kw)
    ctx.on_cancel_cleanup(r.close)
    return r


# ── upscale ─────────────────────────────────────────────────────────────────
def _run_upscale(ctx, clip: ClipInfo, stem: str, out_dir: Path, p: dict) -> str:
    from .upscale import Upscaler, auto_tile, output_size

    model_id = model_for("upscale", p)
    scale = int(p.get("scale") or 2)
    if scale not in (2, 4):
        raise ValueError("scale must be 2 or 4")
    dev = _device()
    from . import ane
    early_ane = None
    from .upscale import NATIVE_X2
    ane_weights = {"realesr-animevideov3": "realesr-animevideov3.pth",
                   **{k: v[0] for k, v in NATIVE_X2.items()}}.get(model_id)
    if dev.backend == "mps" and int(p.get("scale") or 2) == 2 and ane_weights and ane.available():
        # start the Neural Engine worker first: it loads while MPS does
        from . import registry
        try:
            early_ane = ane.AneUpscaler(str(registry.require(model_id) / ane_weights),
                                        clip.height, clip.width)
        except Exception as exc:
            log.warning("Neural Engine upscale unavailable: %s", exc)
    try:
        up = _get_model(("upscale", model_id, dev.backend), lambda: Upscaler(model_id, dev))
    except BaseException:
        if early_ane is not None:
            early_ane.close()
        raise
    tile = p.get("tile")
    tile = auto_tile(dev.backend, dev.vram_gb, clip.width, clip.height) if tile in (None, "auto") else int(tile)
    ow, oh = output_size(clip.width, clip.height, scale)
    ex = _export_settings()
    codec = p.get("codec") or ex.get("codec") or "libx264"
    target = unique_path(out_dir, f"{stem}_x{scale}", suffix_for_codec(codec))
    writer = _writer_for(ctx, target, width=ow, height=oh, fps=clip.fps, codec=codec,
                         crf=int(ex.get("crf", 18)), audio=clip,
                         audio_mode="mute" if ex.get("audio") == "mute" else "encode",
                         audio_bitrate_kbps=int(ex.get("audio_bitrate_kbps", 320)))
    total = clip.n_frames
    # CUDA: frames leave the GPU already in the encoder's I420
    gpu_i420 = dev.backend == "cuda" and bool(getattr(writer, "accepts_i420", False))
    ctx.progress(0.0, f"upscaling x{scale}", force=True)
    if early_ane is not None and tile != 0:
        early_ane.close()
        early_ane = None
    if early_ane is not None:
        # Apple Silicon: the Neural Engine takes every other frame while the
        # GPU does the rest (see ``ane``).
        from . import registry
        weights = str(registry.require(model_id) / ane_weights)
        with _reader_for(ctx, clip) as reader:
            n_ane = ane.run_split(
                _frames(reader), lambda f: up.upscale(f, scale=scale, tile=tile, check_cancel=ctx.check_cancel),
                writer.write, weights, clip.height, clip.width, check_cancel=ctx.check_cancel,
                progress=lambda k: ctx.progress(min(0.99, k / total), f"frame {k}/{total}"), ane=early_ane)
        log.info("upscale: %d of the frames on the Neural Engine", n_ane)
        return writer.close()
    with _reader_for(ctx, clip) as reader:
        for i, frame in enumerate(_frames(reader)):
            ctx.check_cancel()
            kw = {"i420": True} if gpu_i420 else {}
            writer.write(up.upscale(frame, scale=scale, tile=tile, check_cancel=ctx.check_cancel, **kw))
            ctx.progress(min(0.99, (i + 1) / total), f"frame {i + 1}/{total}")
    return writer.close()


# ── interpolation ───────────────────────────────────────────────────────────
def _run_interpolate(ctx, clip: ClipInfo, stem: str, out_dir: Path, p: dict) -> str:
    from .interpolate import CUT_THRESHOLD, plan_timeline, scene_change_score, stream_interpolated

    mode = p.get("mode") or "slowmo"
    factor = int(p.get("factor") or 2)
    target_fps = float(p.get("target_fps") or 60.0)
    skip_cuts = bool(p.get("skip_cuts", True))
    threshold = float(p.get("cut_threshold") or CUT_THRESHOLD)
    dev = _device()
    rife_scale = 0.5 if max(clip.width, clip.height) > 2000 else 1.0
    rife = _rife(dev, model_for("interpolate", p), rife_scale)

    # Validates the parameters and gives the output fps / audio speed; the
    # positions themselves are streamed (the true frame count is only known
    # once the decoder is done).
    plan = plan_timeline(clip.n_frames, clip.fps, mode=mode, factor=factor, target_fps=target_fps)
    step = 1.0 / factor if mode == "slowmo" else clip.fps / plan.out_fps

    ex = _export_settings()
    codec = p.get("codec") or ex.get("codec") or "libx264"
    tag = f"slowmo_x{factor}" if mode == "slowmo" else f"{int(round(plan.out_fps))}fps"
    target = unique_path(out_dir, f"{stem}_{tag}", suffix_for_codec(codec))
    audio_mode = "mute" if ex.get("audio") == "mute" else "encode"
    if mode == "slowmo" and p.get("audio", "mute") != "stretch":
        audio_mode = "mute"
    writer = _writer_for(ctx, target, width=clip.width, height=clip.height, fps=plan.out_fps, codec=codec,
                         crf=int(ex.get("crf", 18)), audio=clip, audio_mode=audio_mode,
                         audio_bitrate_kbps=int(ex.get("audio_bitrate_kbps", 320)),
                         audio_speed=plan.audio_speed)
    cuts: list[int] = []
    total = max(1, len(plan.positions))
    with _reader_for(ctx, clip) as reader:
        # CUDA: every output frame is converted to I420 on the GPU (the
        # writer thread's OpenCV conversion was the job's bottleneck: ~45 ms
        # per 1080p frame on this VM, 180 ms at 4K)
        at = rife.at
        gpu_i420 = dev.backend == "cuda" and getattr(writer, "accepts_i420", False) \
            and clip.width % 2 == 0 and clip.height % 2 == 0
        if gpu_i420:
            import torch
            from .yuv_cuda import rgb255_to_i420
            at = lambda pair, t: rife.at(pair, t, i420=True)  # noqa: E731
        frames = stream_interpolated(
            _frames(reader), step, make_pair=rife.make_pair, at=at,
            is_cut=(lambda a, b, _i: scene_change_score(a, b) > threshold) if skip_cuts else None,
            on_cut=cuts.append,
        )
        for k, out in enumerate(frames):
            ctx.check_cancel()
            if gpu_i420 and isinstance(out, np.ndarray):
                with torch.inference_mode():
                    out = rgb255_to_i420(torch.from_numpy(out).to("cuda").float())
            writer.write(out)
            msg = f"{len(cuts)} cut(s) kept sharp" if cuts else "interpolating"
            ctx.progress(min(0.99, (k + 1) / total), f"{msg} · frame {k + 1}")
    ctx.job.message = f"{len(cuts)} cut(s) kept sharp" if cuts else "done"
    return writer.close()


# ── depth ───────────────────────────────────────────────────────────────────
def _depth_credit(p: dict) -> str:
    """Model label + license for the PNG sequence's README.txt."""
    from .registry import MODELS
    spec = MODELS.get(model_for("depth", p))
    return f"{spec.label} ({spec.license})" if spec else "depth model"


def _run_depth(ctx, clip: ClipInfo, stem: str, out_dir: Path, p: dict) -> str:
    import cv2

    from .depth import DEPTH_FORMATS, stabilize_depth, to_output_res
    from .interpolate import detect_cuts

    fmt = p.get("format") or "png16"
    if fmt not in DEPTH_FORMATS:
        raise ValueError(f"depth format must be one of {DEPTH_FORMATS}")
    smoothing = float(p.get("smoothing", 0.5))
    invert = bool(p.get("invert", False))
    check_memory("depth", clip, p)
    dev = _device()
    est = _depth(dev, model_for("depth", p), depth_input_size(p))

    import time
    raw: list[np.ndarray] = []
    smalls: list[np.ndarray] = []
    total = clip.n_frames
    t0 = time.monotonic()
    if dev.backend == "cuda" and not getattr(est, "temporal", False):
        # CUDA: frames go through the model 8 at a time (GPU preprocessing)
        batch: list[np.ndarray] = []

        def flush():
            for frame, d in zip(batch, est.predict_batch(batch)):
                raw.append(d.astype(np.float16))
                smalls.append(cv2.resize(frame, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_AREA))
            batch.clear()
            ctx.progress(min(0.85, 0.85 * len(raw) / total), f"depth {len(raw)}/{total}")

        with _reader_for(ctx, clip) as reader:
            for frame in _frames(reader):
                ctx.check_cancel()
                batch.append(frame)
                if len(batch) >= 8:
                    flush()
            if batch:
                flush()
    else:
        with _reader_for(ctx, clip) as reader:
            for i, (frame, d) in enumerate(est.predict_stream(_frames(reader))):
                ctx.check_cancel()
                raw.append(d.astype(np.float16))
                smalls.append(cv2.resize(frame, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_AREA))
                ctx.progress(min(0.85, 0.85 * (i + 1) / total), f"depth {i + 1}/{total}")
    if not raw:
        raise RuntimeError("no frames decoded")
    t1 = time.monotonic()
    cuts = detect_cuts(smalls)
    ctx.progress(0.86, "temporal smoothing", force=True)
    depth = stabilize_depth([r.astype(np.float32) for r in raw], smalls, cuts=cuts,
                            smoothing=smoothing, check_cancel=ctx.check_cancel)
    log.info("depth: %d frames inferred in %.1f s, stabilised in %.1f s", len(raw), t1 - t0, time.monotonic() - t1)
    if invert:
        depth = 1.0 - depth

    n = depth.shape[0]
    if fmt == "png16":
        seq_dir = unique_path(out_dir, f"{stem}_depth", "")
        tmp_dir = seq_dir.with_name(seq_dir.name + ".partial")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)
        ctx.on_cancel_cleanup(lambda: shutil.rmtree(tmp_dir, ignore_errors=True))
        def write_png(i: int) -> None:
            d16 = (to_output_res(depth[i], clip.width, clip.height) * 65535.0).round().astype(np.uint16)
            if not cv2.imwrite(str(tmp_dir / f"{stem}_depth_{i:05d}.png"), d16):
                raise OSError(f"could not write depth frame {i}")

        # PNG deflate and the cubic upsize release the GIL: 4 writers
        # (M1 Pro, 120 1080p frames: 4.5 s -> 1.2 s; 5 s clip on NVIDIA:
        # 14 s -> ~4 s), same files.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, min(4, (os.cpu_count() or 2) - 1)),
                                thread_name_prefix="depth-png") as ex:
            for k, _ in enumerate(ex.map(write_png, range(n))):
                ctx.check_cancel()
                ctx.progress(0.88 + 0.12 * (k + 1) / n, f"writing {k + 1}/{n}")
        tmp_dir.rename(seq_dir)
        (seq_dir / "README.txt").write_text(
            f"{_depth_credit(p)} - 16-bit grayscale PNG sequence\n"
            f"source: {clip.path}\nwindow: {clip.start_ms}-{clip.end_ms} ms\nfps: {clip.fps}\n"
            f"frames: {n} (frame-synchronous with the clip)\n"
            f"convention: {'black' if invert else 'white'} = near\n"
            f"After Effects: File > Import, tick 'PNG Sequence', then Interpret Footage > {clip.fps} fps.\n"
        )
        return str(seq_dir)

    if fmt == "prores":
        target = unique_path(out_dir, f"{stem}_depth", ".mov")
        writer = _writer_for(ctx, target, width=clip.width, height=clip.height, fps=clip.fps,
                             pix_fmt_in="gray16le", audio=None,
                             extra_video_args=["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"])
        scale, dtype = 65535.0, np.uint16
    else:
        target = unique_path(out_dir, f"{stem}_depth", ".mp4")
        writer = _writer_for(ctx, target, width=clip.width, height=clip.height, fps=clip.fps,
                             pix_fmt_in="gray", audio=None,
                             extra_video_args=["-c:v", "libx264", "-preset", "medium", "-crf", "12",
                                               "-pix_fmt", "yuv420p"])
        scale, dtype = 255.0, np.uint8
    for i in range(n):
        ctx.check_cancel()
        writer.write((to_output_res(depth[i], clip.width, clip.height) * scale).round().astype(dtype))
        ctx.progress(0.88 + 0.11 * (i + 1) / n, f"encoding {i + 1}/{n}")
    return writer.close()


# ── cutout (alpha matte) ────────────────────────────────────────────────────
def _run_matte(ctx, clip: ClipInfo, stem: str, out_dir: Path, p: dict) -> str:
    import cv2
    from PIL import Image

    from ..api.segment import _edge_mode
    from ..export.ffmpeg import alpha_partial_path, export_scene_with_alpha
    from ..state import get_state
    from .matte import postprocess_stack

    st = get_state()
    models_cfg = st.settings.get("models", {})
    codec = p.get("codec") or "prores_4444_alpha"
    if codec not in ("prores_4444_alpha", "vp9_alpha"):
        raise ValueError("codec must be prores_4444_alpha or vp9_alpha")
    max_dim = mask_max_dim(models_cfg)
    check_memory("matte", clip, p, models_cfg)
    model = _get_model(("matte", models_cfg.get("birefnet_variant", "general"), _device().backend),
                       lambda: st.get_birefnet())

    frames: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    total = clip.n_frames
    # 0.5: the Auto engine's tracker (auto_roto) when the tier has one — the
    # frames are also written as JPEGs for SAM 2.
    from ..models.auto_roto import mask_models
    tracker = mask_models(st).get("tracker") if models_cfg.get("mask_auto_track", True) else None
    if tracker:
        import importlib.util
        tracker = tracker if importlib.util.find_spec("sam2") is not None else None
    jpg_dir = Path(tempfile.mkdtemp(prefix="amv_cutout_")) if tracker else None
    jpgs: list[Path] = []
    try:
        span = 0.4 if tracker else 0.8
        with _reader_for(ctx, clip) as reader:
            for i, frame in enumerate(_frames(reader)):
                ctx.check_cancel()
                h, w = frame.shape[:2]
                s = min(1.0, max_dim / float(max(h, w)))
                small = cv2.resize(frame, (int(round(w * s)) // 2 * 2, int(round(h * s)) // 2 * 2),
                                   interpolation=cv2.INTER_AREA) if s < 1.0 else frame.copy()
                frames.append(small)
                a = model.mask_image(Image.fromarray(small))
                masks.append(np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8) if tracker else a)
                if jpg_dir is not None:
                    jpgs.append(jpg_dir / f"{i:05d}.jpg")
                    Image.fromarray(small).save(jpgs[-1], quality=95)
                ctx.progress(min(span, span * (i + 1) / total), f"matting {i + 1}/{total}")
        if not frames:
            raise RuntimeError("no frames decoded")
        from ..api.segment import _select_characters
        sel_masks = np.stack(masks)
        if sel_masks.dtype != np.uint8:
            sel_masks = np.clip(np.rint(sel_masks * 255.0), 0, 255).astype(np.uint8)
        selector_masks: list = []
        gated = _select_characters(sel_masks, frames, models_cfg, keep=selector_masks)
        if gated is not sel_masks:
            masks = list(gated) if tracker else [m.astype(np.float32) / 255.0 for m in gated]
        del sel_masks, gated
        if tracker:
            from ..models.auto_roto import auto_track
            ctx.check_cancel()
            sam2 = st.get_sam2(variant=tracker)
            hh, ww = frames[0].shape[:2]
            det = np.stack(masks)
            masks = list(auto_track(
                jpgs, det, sam2, frames_dir=jpg_dir, frame_size=(ww, hh),
                mode=p.get("target") or models_cfg.get("mask_auto_target", "all"),
                frames_small=[cv2.resize(f, (256, max(2, round(256 * hh / ww))), interpolation=cv2.INTER_AREA)
                              for f in frames],
                progress_cb=lambda seen, n: ctx.progress(0.4 + 0.4 * seen / max(1, n), f"tracking {seen}/{n}"),
            ))
            del det
        if selector_masks:
            from ..models.auto_roto import tighten
            from ..api.segment import _quality
            masks = list(tighten(np.stack([np.asarray(m, np.float32) / (255.0 if np.asarray(m).dtype == np.uint8 else 1.0)
                                           for m in masks]), selector_masks, quality=_quality(models_cfg)))
        del selector_masks
    finally:
        if jpg_dir is not None:
            shutil.rmtree(jpg_dir, ignore_errors=True)
    stack = postprocess_stack(
        np.stack(masks), frames,
        stabilize=bool(p.get("stabilize", models_cfg.get("mask_temporal_smooth_enabled", True))),
        strength=float(p.get("strength", models_cfg.get("mask_temporal_smooth_strength", 0.5))),
        check_cancel=ctx.check_cancel,
    )
    ctx.progress(0.85, "encoding alpha", force=True)
    target = unique_path(out_dir, f"{stem}_cutout", ".mov" if codec == "prores_4444_alpha" else ".webm")
    ctx.on_cancel_cleanup(lambda: (Path(target).unlink(missing_ok=True),
                                   alpha_partial_path(str(target)).unlink(missing_ok=True)))
    out = export_scene_with_alpha(
        clip.path, clip.start_ms, clip.end_ms, stack, str(target), codec=codec, fps=clip.fps,
        decontaminate_rgb=bool(models_cfg.get("mask_rgb_decontaminate_enabled", False)),
        edge_refine=bool(models_cfg.get("mask_edge_refine_enabled", True)),
        bg_aware_cleanup=bool(models_cfg.get("mask_bg_aware_cleanup_enabled", True)),
        edge_mode=_edge_mode(models_cfg),
        raw_pipes=_i420_in_writer(),   # CUDA path, unchanged elsewhere
        check_cancel=ctx.check_cancel,
    )
    return out


# ── speed estimate ──────────────────────────────────────────────────────────
def estimate(kind: str, params: dict, *, bench_frames: int = 5) -> dict:
    """Load the model, time it on up to ``bench_frames`` frames of the real
    clip, extrapolate to the whole job."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    clip, _ = resolve_source(params.get("source") or {})
    dev = _device()
    frames = read_frames(clip, max_frames=bench_frames + 1)
    if not frames:
        raise RuntimeError("no frames decoded")
    t_load = time.perf_counter()
    n_out = clip.n_frames
    if kind == "upscale":
        from .upscale import Upscaler, auto_tile
        model_id = model_for("upscale", params)
        up = _get_model(("upscale", model_id, dev.backend), lambda: Upscaler(model_id, dev))
        tile = auto_tile(dev.backend, dev.vram_gb, clip.width, clip.height)
        load_s = time.perf_counter() - t_load
        fn = lambda f, _n: up.upscale(f, scale=int(params.get("scale") or 2), tile=tile)  # noqa: E731
    elif kind == "interpolate":
        from .interpolate import plan_timeline
        rife_scale = 0.5 if max(clip.width, clip.height) > 2000 else 1.0
        rife = _rife(dev, model_for("interpolate", params), rife_scale)
        load_s = time.perf_counter() - t_load
        plan = plan_timeline(clip.n_frames, clip.fps, mode=params.get("mode") or "slowmo",
                             factor=int(params.get("factor") or 2),
                             target_fps=float(params.get("target_fps") or 60.0))
        n_out = sum(1 for pos in plan.positions if abs(pos - round(pos)) > 1e-4)
        fn = lambda f, n: rife.at(rife.make_pair(f, frames[min(n + 1, len(frames) - 1)]), 0.5)  # noqa: E731
    elif kind == "depth":
        est = _depth(dev, model_for("depth", params), depth_input_size(params))
        load_s = time.perf_counter() - t_load
        fn = lambda f, _n: est.predict(f)  # noqa: E731
    else:
        from PIL import Image
        from ..state import get_state
        st = get_state()
        model = _get_model(("matte", st.settings.get("models", {}).get("birefnet_variant", "general"),
                            dev.backend), lambda: st.get_birefnet())
        load_s = time.perf_counter() - t_load
        fn = lambda f, _n: model.mask_image(Image.fromarray(f))  # noqa: E731

    fn(frames[0], 0)  # warm-up (lazy init, kernel selection)
    sample = frames[1:bench_frames + 1] or frames[:1]
    t0 = time.perf_counter()
    for n, f in enumerate(sample):
        fn(f, n)
    per_frame = (time.perf_counter() - t0) / len(sample)
    if kind == "depth":
        # a temporal model's predict() runs a whole window for one frame
        per_frame /= getattr(est, "frames_per_predict", 1)
    # encode/decode overhead: ~10% on top of model time is a fair average
    est_s = per_frame * max(1, n_out) * 1.1
    return {
        "kind": kind,
        "device": dev.backend,
        "device_name": dev.name,
        "cpu_only": dev.backend == "cpu",
        "frames": clip.n_frames,
        "work_frames": n_out,
        "seconds_per_frame": round(per_frame, 4),
        "model_load_seconds": round(load_s, 2),
        "estimated_seconds": round(est_s, 1),
        "width": clip.width,
        "height": clip.height,
        "fps": clip.fps,
    }
