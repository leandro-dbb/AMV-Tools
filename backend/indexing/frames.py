"""Median-frame extraction: GPU pipeline via ffmpeg subprocess when available,
PyAV CPU fallback otherwise.

For each (start_ms, end_ms) scene we need the median frame to feed into
wd-tagger and SigLIP. With PyAV+CPU we decoded every frame full-res 1920×1080
and downloaded it back from GPU to CPU before any downscale — ~30 s per
22-min episode just for this phase.

The new GPU path runs the downscale inside ffmpeg via `scale_cuda`, so each
frame leaves the GPU already at 512×288 RGB (442 KB) instead of full 1080p
(6 MB). That's ~14× less bandwidth on the GPU→host bus and brings extraction
down to ~10-12 s on a typical 5070-class GPU.

Both consumer models we run downstream are fine with 512×288 input:
  - wd-tagger preprocesses to 448² (pads aspect ratio) — 512×288 → 448×252 → pad
  - SigLIP NaFlex with max_num_patches=128 covers ~256×128 effective patches

Falls back to PyAV CPU decode whenever ffmpeg+scale_cuda isn't usable.
"""
from __future__ import annotations

import io
import logging
import subprocess
import sys
from typing import Iterator, Optional, Tuple

import numpy as np
from PIL import Image

import av

from . import cuts  # for _ffmpeg_path and _ffmpeg_supports_scale_cuda

log = logging.getLogger(__name__)
av.logging.set_level(av.logging.PANIC)


# Safety margin: seek lands on the previous keyframe, so we ask for a position
# slightly before the first target to avoid landing too late.
_SEEK_SAFETY_MS = 500

# GPU pipeline output size. 512×288 keeps 16:9 and satisfies both downstream
# models (see module docstring).
_GPU_FRAME_W, _GPU_FRAME_H = 512, 288


def _extract_frames_via_ffmpeg_cuda(
    path: str, targets_ms: list[int], fps: float
) -> Iterator[Tuple[int, Image.Image]]:
    """Single ffmpeg subprocess that decodes from `targets_ms[0] - safety` to
    the end, with `scale_cuda` doing the downscale on-device. We read raw
    512×288×3 RGB frames from stdout, count them to recover their timestamp
    (relative to the seek point), and yield only those that match a requested
    target."""
    if not targets_ms:
        return
    targets = sorted(set(targets_ms))
    fps_safe = fps if fps > 0 else 30.0

    seek_ms = max(0, targets[0] - _SEEK_SAFETY_MS)
    seek_seconds = seek_ms / 1000.0

    w, h = _GPU_FRAME_W, _GPU_FRAME_H
    frame_bytes = w * h * 3  # RGB

    # `-ss` BEFORE `-i` = fast seek (jumps to nearest keyframe, decodes silently
    # forward to the requested timestamp). `-fps_mode cfr` normalizes timestamps
    # so frame_idx / fps is exact regardless of VFR.
    cmd = [
        cuts._ffmpeg_path(),
        "-hide_banner", "-nostats", "-loglevel", "error",
        "-fflags", "+discardcorrupt",
        "-hwaccel", "cuda",
        "-hwaccel_output_format", "cuda",
    ]
    if seek_seconds > 0:
        cmd += ["-ss", f"{seek_seconds:.3f}"]
    cmd += [
        "-i", path,
        "-an", "-sn",
        # Force NV12 between scale_cuda and hwdownload, else the filter graph
        # fails with EINVAL on this ffmpeg build (see cuts.py for the same fix).
        "-vf", f"scale_cuda={w}:{h}:format=nv12,hwdownload,format=nv12,format=rgb24",
        "-fps_mode", "cfr",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1",
    ]
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=flags,
        bufsize=frame_bytes * 4,
    )

    target_idx = 0
    frame_idx = -1
    yielded_count = 0
    try:
        assert proc.stdout is not None
        read = proc.stdout.read
        while target_idx < len(targets):
            buf = read(frame_bytes)
            if not buf:
                break
            if len(buf) < frame_bytes:
                # Short read = end of stream; pad to avoid reshape crash, then exit.
                buf = buf + bytes(frame_bytes - len(buf))
            frame_idx += 1
            now_ms = seek_ms + int((frame_idx / fps_safe) * 1000)

            # Decode the frame to PIL ONLY when we're about to yield it — saves
            # a lot of np.frombuffer / Image.fromarray cycles on frames we skip.
            while target_idx < len(targets) and now_ms >= targets[target_idx]:
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
                # arr is a view on the read buffer; .copy() so the next read
                # doesn't stomp on the bytes still referenced by the PIL image.
                yield targets[target_idx], Image.fromarray(arr.copy())
                target_idx += 1
                yielded_count += 1
    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)

    # Only raise on hard failure. Non-zero return *after* we already yielded
    # every requested frame is the normal early-termination case: ffmpeg got
    # SIGPIPE when we stopped reading because we had what we needed.
    if yielded_count == 0 and proc.returncode not in (0, None):
        stderr_tail = ""
        try:
            stderr_tail = (proc.stderr.read() or b"").decode("utf-8", errors="replace")[-500:]
        except Exception:
            pass
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}: {stderr_tail!r}")
    if yielded_count > 0 and yielded_count < len(targets):
        log.debug(
            "ffmpeg yielded %d/%d targets before exiting (rc=%s) — pipeline will skip the missing scenes",
            yielded_count, len(targets), proc.returncode,
        )


def _extract_frames_via_pyav(
    path: str, targets_ms: list[int]
) -> Iterator[Tuple[int, Image.Image]]:
    """CPU PyAV decode with keyframe seek. Used when ffmpeg+scale_cuda is
    unavailable or the subprocess path fails."""
    if not targets_ms:
        return
    targets = sorted(set(targets_ms))
    container = av.open(path, options={"err_detect": "ignore_err"})
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        seek_us = max(0, (targets[0] - _SEEK_SAFETY_MS) * 1000)
        if seek_us > 0:
            try:
                container.seek(seek_us)
            except (av.AVError, OSError):
                pass

        target_idx = 0
        for frame in container.decode(stream):
            if target_idx >= len(targets):
                break
            now_ms = int(frame.time * 1000) if frame.time is not None else 0
            while target_idx < len(targets) and now_ms >= targets[target_idx]:
                img = Image.fromarray(frame.to_ndarray(format="rgb24"))
                yield targets[target_idx], img
                target_idx += 1
    finally:
        container.close()


def extract_frames_at(path: str, timestamps_ms: list[int]) -> Iterator[Tuple[int, Image.Image]]:
    """Yield (requested_ms, PIL.Image) for each ms in `timestamps_ms`, in order.

    Prefers ffmpeg+scale_cuda (GPU pipeline). Falls back to PyAV CPU if the
    subprocess path is unavailable or fails mid-stream.
    """
    if not timestamps_ms:
        return

    if cuts._ffmpeg_supports_scale_cuda():
        # We can't easily resume a generator if the subprocess fails mid-way
        # (we've already yielded some frames). Strategy: collect targets, run
        # the GPU path eagerly, fall back to PyAV only on outright failure
        # before yielding anything.
        try:
            info = cuts.probe_video(path)
            fps = info["fps"]
            yield from _extract_frames_via_ffmpeg_cuda(path, timestamps_ms, fps)
            return
        except Exception as exc:
            log.warning(
                "ffmpeg scale_cuda extract failed for %s (%s); falling back to PyAV",
                path, exc,
            )

    yield from _extract_frames_via_pyav(path, timestamps_ms)


def thumbnail_bytes(img: Image.Image, size: tuple[int, int] = (320, 180), quality: int = 60) -> bytes:
    # NEAREST is acceptable here: thumbs are 320×180 hover previews, not display-quality.
    # optimize=False skips the slow Huffman/quantization optimization pass — the JPEG is
    # ~5% larger but produced 3-5× faster, which matters when we churn through hundreds
    # per video.
    thumb = img.copy()
    thumb.thumbnail(size, Image.Resampling.NEAREST)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=quality, optimize=False)
    return buf.getvalue()
