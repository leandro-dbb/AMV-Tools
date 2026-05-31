"""Hard cut detection — full-GPU path via ffmpeg subprocess + scale_cuda.

We tried two earlier approaches:

  1) Subprocess `ffmpeg -vf select=gt(scene,T),showinfo` and parse stderr.
     Worked but decoded on CPU and rebuilt a process per video.
  2) PyAV with NVDEC + per-frame `reformat()` on CPU.
     Decodes on GPU but every frame is downloaded full-res to CPU before the
     downscale, which costs ~60 s for a 22-min episode on consumer NVMe.

This module now uses approach 3: spawn ffmpeg with NVDEC + the `scale_cuda`
filter so the 160×90 downscale happens on the GPU, and only the tiny
grayscale plane is piped to stdout. Net effect: detect_cuts becomes
~3× faster, dominated by the NVDEC decode alone (which is itself near
the limit of what consumer GPUs can do on h264).

Fallbacks, in order of preference:
  - ffmpeg + cuda + scale_cuda  (this is what we want)
  - PyAV + NVDEC + CPU reformat  (works without scale_cuda)
  - PyAV + CPU decode + CPU reformat  (works without any NVIDIA stack)
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple

import av
import av.error
import numpy as np

try:
    from av.video.reformatter import Interpolation as _Interp
    _FAST_INTERP = _Interp.FAST_BILINEAR
except (ImportError, AttributeError):
    _FAST_INTERP = None

log = logging.getLogger(__name__)


# ── Tuning constants ────────────────────────────────────────────────────────
_HIST_BINS = 32           # intensity histogram bins per frame
_DIFF_W, _DIFF_H = 160, 90  # downscale target
_FRAME_STRIDE = 4         # sample 1 frame in N (≈133 ms on 30 fps content)
_MIN_CUT_GAP_MS = 200     # minimum spacing between two detected cuts
_MIN_SCENE_MS = 120       # drop scenes shorter than this (flashes)


def _ffmpeg_path() -> str:
    """Path to the ffmpeg binary. Also used by `backend.export.ffmpeg` for
    encoding clips and proxies."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        which = shutil.which("ffmpeg")
        if which:
            return which
        raise RuntimeError("ffmpeg not found")


# ── scale_cuda probe ────────────────────────────────────────────────────────
_FFMPEG_CUDA_PROBED = False
_FFMPEG_CUDA_AVAILABLE = False


def _ffmpeg_supports_scale_cuda() -> bool:
    """Return True if the ffmpeg binary has both the `cuda` hwaccel and the
    `scale_cuda` filter compiled in. Cached for the process lifetime."""
    global _FFMPEG_CUDA_PROBED, _FFMPEG_CUDA_AVAILABLE
    if _FFMPEG_CUDA_PROBED:
        return _FFMPEG_CUDA_AVAILABLE
    _FFMPEG_CUDA_PROBED = True
    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        filters = subprocess.run(
            [_ffmpeg_path(), "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=10, creationflags=flags,
        )
        accels = subprocess.run(
            [_ffmpeg_path(), "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=10, creationflags=flags,
        )
        has_filter = "scale_cuda" in (filters.stdout or "")
        has_cuda = "cuda" in (accels.stdout or "")
        _FFMPEG_CUDA_AVAILABLE = bool(has_filter and has_cuda)
        if _FFMPEG_CUDA_AVAILABLE:
            log.info("ffmpeg+scale_cuda available — using full-GPU detect_cuts")
        else:
            log.info(
                "ffmpeg+scale_cuda not available (filter=%s, hwaccel=%s); "
                "detect_cuts will use PyAV fallback",
                has_filter, has_cuda,
            )
    except Exception as exc:
        log.warning("ffmpeg capability probe failed: %s", exc)
        _FFMPEG_CUDA_AVAILABLE = False
    return _FFMPEG_CUDA_AVAILABLE


# ── NVDEC handle (for the PyAV fallback path only) ──────────────────────────
_NVDEC_PROBED = False
_NVDEC_HW: Optional[av.codec.hwaccel.HWAccel] = None


def _nvdec_handle() -> Optional[av.codec.hwaccel.HWAccel]:
    global _NVDEC_PROBED, _NVDEC_HW
    if not _NVDEC_PROBED:
        _NVDEC_PROBED = True
        try:
            _NVDEC_HW = av.codec.hwaccel.HWAccel(device_type="cuda", allow_software_fallback=True)
            log.info("NVDEC enabled for PyAV fallback path")
        except Exception as exc:
            log.debug("NVDEC unavailable: %s", exc)
            _NVDEC_HW = None
    return _NVDEC_HW


# ── Probe ───────────────────────────────────────────────────────────────────
def probe_video(path: str) -> dict:
    """Return duration_ms, fps, resolution via PyAV."""
    fps = 30.0
    width = 0
    height = 0
    duration_ms = 0
    container = av.open(path, options={"err_detect": "ignore_err"})
    try:
        if container.streams.video:
            stream = container.streams.video[0]
            if stream.average_rate:
                fps = float(stream.average_rate)
            width = int(stream.codec_context.width or 0)
            height = int(stream.codec_context.height or 0)
        if container.duration is not None:
            duration_ms = int(container.duration / av.time_base * 1000)
        elif container.streams.video:
            stream = container.streams.video[0]
            if stream.duration is not None and stream.time_base is not None:
                duration_ms = int(float(stream.duration * stream.time_base) * 1000)
    finally:
        container.close()
    return dict(fps=fps, duration_ms=duration_ms, resolution=f"{width}x{height}")


# ── Histogram helpers ───────────────────────────────────────────────────────
def _hist_from_plane_bytes(buf: bytes, w: int, h: int) -> np.ndarray:
    """L1-normalized intensity histogram from a packed gray plane."""
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
    hist, _ = np.histogram(arr, bins=_HIST_BINS, range=(0, 256))
    hist_norm = hist.astype(np.float32)
    total = hist_norm.sum()
    return hist_norm / total if total > 0 else hist_norm


def _hist_from_av_plane(small_frame) -> np.ndarray:
    """Same as `_hist_from_plane_bytes` but reads from a PyAV frame plane,
    honoring its line_size (which may be padded)."""
    plane = small_frame.planes[0]
    stride = plane.line_size or _DIFF_W
    buf = np.frombuffer(plane, dtype=np.uint8)
    arr = buf.reshape(_DIFF_H, stride)[:, :_DIFF_W]
    hist, _ = np.histogram(arr, bins=_HIST_BINS, range=(0, 256))
    hist_norm = hist.astype(np.float32)
    total = hist_norm.sum()
    return hist_norm / total if total > 0 else hist_norm


# ── Primary path: subprocess ffmpeg + scale_cuda ────────────────────────────
def _detect_cuts_via_ffmpeg_cuda(
    path: str, threshold: float, fps: float, duration_ms: int
) -> List[int]:
    """Decode + downscale + grayscale entirely on the GPU via ffmpeg, then
    read 160×90 gray planes from stdout. Returns the list of cut timestamps
    (start of each scene, in ms)."""
    w, h = _DIFF_W, _DIFF_H
    frame_bytes = w * h

    # `scale_cuda` resizes on-device, `hwdownload` copies to host memory,
    # `format=gray` discards chroma. -an / -sn skip audio + subs (anime MKVs
    # are often loaded with both).
    #
    # `-fps_mode cfr` is what saves us on VFR sources: ffmpeg duplicates or
    # drops frames so the output stream is at a constant rate. Without it,
    # `frame_idx / fps` would drift by tens of ms per minute on screen
    # recordings, phone clips, etc. With it, the math is exact for any source.
    cmd = [
        _ffmpeg_path(),
        "-hide_banner", "-nostats", "-loglevel", "error",
        "-fflags", "+discardcorrupt",
        "-hwaccel", "cuda",
        "-hwaccel_output_format", "cuda",
        "-i", path,
        "-an", "-sn",
        # `scale_cuda` must declare its GPU output pixel format explicitly,
        # otherwise hwdownload chokes with EINVAL on some ffmpeg builds (ours
        # included). Force NV12 → CPU → gray.
        "-vf", f"scale_cuda={w}:{h}:format=nv12,hwdownload,format=nv12,format=gray",
        "-fps_mode", "cfr",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "pipe:1",
    ]
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=flags,
        bufsize=frame_bytes * 16,
    )

    cuts_ms: List[int] = [0]
    prev_hist: Optional[np.ndarray] = None
    frame_idx = -1
    fps_safe = fps if fps > 0 else 30.0

    try:
        assert proc.stdout is not None
        read = proc.stdout.read
        while True:
            buf = read(frame_bytes)
            if not buf:
                break
            if len(buf) < frame_bytes:
                # Pipe is closing or we got a partial trailing frame — fill the
                # remainder so the reshape doesn't blow up.
                buf = buf + bytes(frame_bytes - len(buf))
            frame_idx += 1
            if frame_idx % _FRAME_STRIDE != 0:
                continue
            hist_norm = _hist_from_plane_bytes(buf, w, h)
            if prev_hist is not None:
                diff = float(np.abs(hist_norm - prev_hist).sum()) * 0.5
                if diff > threshold:
                    ts_ms = int((frame_idx / fps_safe) * 1000)
                    if ts_ms > cuts_ms[-1] + _MIN_CUT_GAP_MS:
                        cuts_ms.append(ts_ms)
            prev_hist = hist_norm
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

    if proc.returncode not in (0, None):
        stderr_tail = ""
        try:
            stderr_tail = (proc.stderr.read() or b"").decode("utf-8", errors="replace")[-500:]
        except Exception:
            pass
        # Non-zero with an empty pipe = a real failure. Caller will retry the
        # PyAV path.
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}: {stderr_tail!r}")

    return cuts_ms


# ── Fallback path: PyAV (NVDEC or CPU) + CPU reformat ───────────────────────
def _open_for_decode(path: str) -> tuple[av.container.InputContainer, bool]:
    hw = _nvdec_handle()
    if hw is not None:
        try:
            return av.open(path, hwaccel=hw, options={"err_detect": "ignore_err"}), True
        except av.AVError as exc:
            log.warning("NVDEC open failed for %s, falling back to CPU: %s", path, exc)
    return av.open(path, options={"err_detect": "ignore_err"}), False


def _cpu_downscale(frame):
    if _FAST_INTERP is not None:
        return frame.reformat(
            width=_DIFF_W, height=_DIFF_H, format="gray",
            interpolation=_FAST_INTERP,
        )
    return frame.reformat(width=_DIFF_W, height=_DIFF_H, format="gray")


def _detect_cuts_via_pyav(path: str, threshold: float) -> List[int]:
    container, _used_nvdec = _open_for_decode(path)
    cuts_ms: List[int] = [0]
    prev_hist: Optional[np.ndarray] = None
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        frame_idx = -1
        for frame in container.decode(stream):
            frame_idx += 1
            if frame_idx % _FRAME_STRIDE != 0:
                continue
            ts_ms = int(frame.time * 1000) if frame.time is not None else 0
            try:
                small = _cpu_downscale(frame)
            except av.AVError:
                continue
            hist_norm = _hist_from_av_plane(small)
            if prev_hist is not None:
                diff = float(np.abs(hist_norm - prev_hist).sum()) * 0.5
                if diff > threshold and ts_ms > cuts_ms[-1] + _MIN_CUT_GAP_MS:
                    cuts_ms.append(ts_ms)
            prev_hist = hist_norm
    finally:
        container.close()
    return cuts_ms


# ── Public entry point ──────────────────────────────────────────────────────
def detect_cuts(path: str, threshold: float = 0.30) -> List[Tuple[int, int]]:
    """Return list of (start_ms, end_ms) tuples for the detected scenes.

    `threshold` is the L1 distance between consecutive normalized intensity
    histograms; range is [0, 1]. Defaults to 0.30 — empirically close to the
    legacy `select=gt(scene,0.30)` ffmpeg filter on anime content.
    """
    info = probe_video(path)
    duration_ms = info["duration_ms"]
    fps = info["fps"]
    if duration_ms <= 0:
        return []

    cuts_ms: Optional[List[int]] = None
    if _ffmpeg_supports_scale_cuda():
        try:
            cuts_ms = _detect_cuts_via_ffmpeg_cuda(path, threshold, fps, duration_ms)
        except Exception as exc:
            log.warning(
                "ffmpeg scale_cuda path failed for %s (%s); falling back to PyAV",
                path, exc,
            )
            cuts_ms = None

    if cuts_ms is None:
        cuts_ms = _detect_cuts_via_pyav(path, threshold)

    cuts_ms.append(duration_ms)
    scenes: List[Tuple[int, int]] = []
    for i in range(len(cuts_ms) - 1):
        start = cuts_ms[i]
        end = cuts_ms[i + 1]
        if end - start >= _MIN_SCENE_MS:
            scenes.append((start, end))
    return scenes
