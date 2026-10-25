"""Hard cut detection — full-GPU path via ffmpeg subprocess + scale_cuda.

We tried two earlier approaches:

  1) Subprocess `ffmpeg -vf select=gt(scene,T),showinfo` and parse stderr.
     Worked but decoded on CPU and rebuilt a process per video.
  2) PyAV with NVDEC + per-frame `reformat()` on CPU.
     Decodes on GPU but every frame is downloaded full-res to CPU before the
     downscale, which costs ~60 s for a 22-min episode on consumer NVMe.

This module now uses approach 3: spawn ffmpeg with hardware decode (NVDEC on
NVIDIA, VideoToolbox on macOS) + an on-device downscale filter (`scale_cuda`
or `scale_vt`) so the 160×90 downscale happens on the GPU, and only the tiny
grayscale plane is piped to stdout. Net effect: detect_cuts becomes
~3× faster, dominated by the hardware decode alone.

Fallbacks, in order of preference:
  - ffmpeg + hw decode + on-device scale  (scale_cuda / scale_vt)
  - ffmpeg + VideoToolbox decode + CPU scale  (macOS builds without scale_vt)
  - PyAV + NVDEC/VideoToolbox + CPU reformat
  - PyAV + CPU decode + CPU reformat  (works without any GPU stack)
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from typing import Callable, List, Optional, Tuple

import av
import av.error
import numpy as np

# PyAV 15+ exposes FFmpegError; older releases called it AVError. Resolve once
# so a CPU-only machine can still fall back cleanly when NVDEC init fails.
_AV_ERROR = getattr(av, "AVError", av.error.FFmpegError)

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


_FFMPEG_PATH: Optional[str] = None


def _ffmpeg_path() -> str:
    """Path to the ffmpeg binary. Also used by `backend.export.ffmpeg` for
    encoding clips and proxies. Resolved once per process."""
    global _FFMPEG_PATH
    if _FFMPEG_PATH is None:
        _FFMPEG_PATH = _resolve_ffmpeg()
    return _FFMPEG_PATH


def _nvidia_gpu_present() -> bool:
    if sys.platform.startswith("linux"):
        return os.path.exists("/dev/nvidia0") or os.path.exists("/dev/nvidiactl")
    if sys.platform == "win32":
        return shutil.which("nvidia-smi") is not None
    return False


def _ffmpeg_has_cuda(path: str) -> bool:
    """NVDEC decode + on-device scaling + NVENC compiled into this build."""
    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        run = lambda *a: subprocess.run(  # noqa: E731
            [path, "-hide_banner", *a], capture_output=True, text=True, timeout=10, creationflags=flags,
        ).stdout or ""
        return ("cuda" in run("-hwaccels") and "scale_cuda" in run("-filters")
                and "h264_nvenc" in run("-encoders"))
    except Exception:
        return False


def _resolve_ffmpeg() -> str:
    """``AMV_FFMPEG`` override, else the bundled imageio-ffmpeg binary.

    On NVIDIA machines the bundled build must also carry CUDA: the Linux
    imageio-ffmpeg binary (static johnvansickle build) ships without
    NVDEC/NVENC, which silently pushed cut detection, frame extraction,
    proxies and exports onto the CPU. There we prefer a system ffmpeg that
    has them (Debian/Ubuntu/Arch packages do) — ~3-5x faster indexing.
    """
    forced = os.environ.get("AMV_FFMPEG")
    if forced and os.path.isfile(forced):
        return forced
    bundled: Optional[str] = None
    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    if sys.platform != "darwin" and _nvidia_gpu_present():
        if bundled and _ffmpeg_has_cuda(bundled):
            return bundled
        system = shutil.which("ffmpeg")
        if system and system != bundled and _ffmpeg_has_cuda(system):
            log.info("bundled ffmpeg lacks NVDEC/NVENC; using %s", system)
            return system
    if bundled:
        return bundled
    which = shutil.which("ffmpeg")
    if which:
        return which
    raise RuntimeError("ffmpeg not found")


# ── hardware decode profile probe ───────────────────────────────────────────
_HW_PROFILE_PROBED = False
_HW_PROFILE: Optional[dict] = None


def _ffmpeg_hw_profile() -> Optional[dict]:
    """Best hardware decode+downscale profile for this ffmpeg build.

    Returns a dict (cached for the process lifetime) with:
      name        — 'cuda' | 'videotoolbox'
      input_args  — hwaccel flags to place before `-i`
      scale_vf    — vf prefix with `{w}`/`{h}` placeholders; after it the
                    frames are in CPU memory, ready for a trailing `format=`
    or None when no hardware decode path exists (callers fall back to PyAV).

    On macOS we prefer `scale_vt` (VTPixelTransferSession, ffmpeg 6.1+) so
    the downscale happens on-device like `scale_cuda`; older builds still get
    VideoToolbox decode with a cheap CPU downscale, which already removes the
    dominant cost (full-res software decode).
    """
    global _HW_PROFILE_PROBED, _HW_PROFILE
    if _HW_PROFILE_PROBED:
        return _HW_PROFILE
    _HW_PROFILE_PROBED = True
    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        filters = subprocess.run(
            [_ffmpeg_path(), "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=10, creationflags=flags,
        ).stdout or ""
        accels = subprocess.run(
            [_ffmpeg_path(), "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=10, creationflags=flags,
        ).stdout or ""

        if "cuda" in accels and "scale_cuda" in filters:
            # `scale_cuda` must declare its GPU output pixel format explicitly,
            # otherwise hwdownload chokes with EINVAL on some ffmpeg builds
            # (ours included). Force NV12 → CPU.
            _HW_PROFILE = {
                "name": "cuda",
                "input_args": ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"],
                "scale_vf": "scale_cuda={w}:{h}:format=nv12,hwdownload,format=nv12",
            }
        elif sys.platform == "darwin" and "videotoolbox" in accels:
            if "scale_vt" in filters:
                _HW_PROFILE = {
                    "name": "videotoolbox",
                    "input_args": ["-hwaccel", "videotoolbox",
                                   "-hwaccel_output_format", "videotoolbox_vld"],
                    "scale_vf": "scale_vt={w}:{h},hwdownload,format=nv12",
                }
            else:
                _HW_PROFILE = {
                    "name": "videotoolbox",
                    "input_args": ["-hwaccel", "videotoolbox"],
                    "scale_vf": "scale={w}:{h}:flags=fast_bilinear",
                }
        if _HW_PROFILE is not None:
            log.info("ffmpeg hardware decode available (%s) — using full-HW detect_cuts",
                     _HW_PROFILE["name"])
        else:
            log.info("no ffmpeg hardware decode path; detect_cuts will use PyAV fallback")
    except Exception as exc:
        log.warning("ffmpeg capability probe failed: %s", exc)
        _HW_PROFILE = None
    return _HW_PROFILE


# On Apple Silicon, ffmpeg's VideoToolbox hwaccel decodes one frame at a time
# (submit → wait), so its throughput is latency-bound, while libavcodec's
# frame-threaded software decoder scales over every core. Measured on an
# M1 Pro (8P+2E), whole-file decode of a 4-min clip: H.264 1080p 37.6 s VT
# vs 5.5 s software, HEVC 10-bit 1080p 15.0 vs 9.3 s, H.264 4K 15.3 vs 3.7 s;
# only HEVC 4K is a tie (6.0 vs 6.5 s) — there VT wins by leaving the CPU to
# the motion analysis that runs alongside.
_SOFTWARE_PROFILE = {
    "name": "software",
    "input_args": [],
    "scale_vf": "scale={w}:{h}:flags=fast_bilinear",
    # Model-input frames (512×288 RGB): SigLIP 2 is sensitive to resampling.
    # Against a Lanczos-from-1080p reference, bicubic kept cos 0.9965 (VT's
    # scale_vt 0.9868, fast_bilinear 0.9919) on 80 real anime frames.
    "scale_vf_rgb": "scale={w}:{h}:flags=bicubic",
}
_DECODE_CHOICE: dict[str, Optional[dict]] = {}


def _software_profile() -> dict:
    """``_SOFTWARE_PROFILE``, with ``AMV_DECODE_THREADS`` decoder threads if
    set (benchmarks)."""
    n = os.environ.get("AMV_DECODE_THREADS", "").strip()
    if not n.isdigit():
        return _SOFTWARE_PROFILE
    return {**_SOFTWARE_PROFILE, "input_args": ["-threads", n]}


def _decode_profile(path: str) -> Optional[dict]:
    """ffmpeg decode profile for ``path``: the hardware profile, except on
    macOS where the software decoder is used unless VideoToolbox measured
    faster for this codec/size (HEVC above 1440p). ``AMV_DECODE=hw|sw``
    forces either side (benchmarks)."""
    forced = os.environ.get("AMV_DECODE", "").lower()
    if forced == "sw":
        return _software_profile()
    profile = _ffmpeg_hw_profile()
    if forced == "hw" or profile is None or profile["name"] != "videotoolbox":
        return profile
    if path in _DECODE_CHOICE:
        return _DECODE_CHOICE[path]
    choice = _software_profile()
    try:
        with av.open(path, options={"err_detect": "ignore_err"}) as c:
            cc = c.streams.video[0].codec_context
            if cc.name in ("hevc", "h265") and int(cc.height or 0) > 1440:
                choice = profile
    except Exception:
        pass
    _DECODE_CHOICE[path] = choice
    log.info("decode path for %s: %s", path, choice["name"])
    return choice


# ── hwaccel handle (for the PyAV fallback path only) ────────────────────────
_PYAV_HW_PROBED = False
_PYAV_HW: Optional[av.codec.hwaccel.HWAccel] = None


def _pyav_hwaccel_handle() -> Optional[av.codec.hwaccel.HWAccel]:
    global _PYAV_HW_PROBED, _PYAV_HW
    if not _PYAV_HW_PROBED:
        _PYAV_HW_PROBED = True
        device_type = "videotoolbox" if sys.platform == "darwin" else "cuda"
        try:
            _PYAV_HW = av.codec.hwaccel.HWAccel(device_type=device_type, allow_software_fallback=True)
            log.info("%s enabled for PyAV fallback path", device_type)
        except Exception as exc:
            log.debug("PyAV hwaccel (%s) unavailable: %s", device_type, exc)
            _PYAV_HW = None
    return _PYAV_HW


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
    if 256 % _HIST_BINS == 0:
        # same counts as np.histogram(range=(0, 256)) on 8-bit data, ~6x faster
        hist = np.bincount((arr // (256 // _HIST_BINS)).ravel(), minlength=_HIST_BINS)
    else:
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


def _exact_cut_index(sampled_idx: int, between: list, hist_of: Callable[[object], np.ndarray]) -> int:
    """Frame index where the shot actually changes.

    The detector compares frames ``_FRAME_STRIDE`` apart, so a change seen
    at sampled frame ``sampled_idx`` happened on one of the frames since the
    previous sample; reporting ``sampled_idx`` put up to stride-1 frames of
    the new shot at the end of the previous scene — a flash at the end of
    every exported clip. ``between`` holds the planes of frames
    ``sampled_idx - stride .. sampled_idx`` (oldest first, missing ones
    allowed at the start); the cut is the largest consecutive jump.
    """
    if len(between) < 2:
        return sampled_idx
    hists = [hist_of(p) for p in between]
    jumps = [float(np.abs(hists[i + 1] - hists[i]).sum()) for i in range(len(hists) - 1)]
    return sampled_idx - (len(between) - 1) + 1 + int(np.argmax(jumps))


# ── Primary path: subprocess ffmpeg + hardware decode ───────────────────────
def _detect_cuts_via_ffmpeg_hw(
    path: str, threshold: float, fps: float, duration_ms: int, profile: dict,
    frame_sink: Optional[Callable[[int, np.ndarray], None]] = None,
) -> List[int]:
    """Decode + downscale on the GPU via ffmpeg (`profile` from
    `_ffmpeg_hw_profile`), then read 160×90 gray planes from stdout. Returns
    the list of cut timestamps (start of each scene, in ms)."""
    w, h = _DIFF_W, _DIFF_H
    frame_bytes = w * h

    # The profile's scale_vf resizes (on-device when the build allows it),
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
        *profile["input_args"],
        "-i", path,
        "-an", "-sn",
        "-vf", profile["scale_vf"].format(w=w, h=h) + ",format=gray",
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
    recent: list = []   # planes since the previous sample, for the exact cut
    last_detect_ms = 0

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
            recent.append(buf)
            if frame_idx % _FRAME_STRIDE != 0:
                continue
            between, recent = recent, [buf]
            hist_norm = _hist_from_plane_bytes(buf, w, h)
            if frame_sink is not None:
                frame_sink(int((frame_idx / fps_safe) * 1000),
                           np.frombuffer(buf, dtype=np.uint8).reshape(h, w))
            if prev_hist is not None:
                diff = float(np.abs(hist_norm - prev_hist).sum()) * 0.5
                if diff > threshold:
                    ts_ms = int((frame_idx / fps_safe) * 1000)
                    # gap rule on the sampled instants: same decisions as
                    # before, only the reported position is refined
                    if ts_ms > last_detect_ms + _MIN_CUT_GAP_MS:
                        last_detect_ms = ts_ms
                        exact = _exact_cut_index(frame_idx, between,
                                                 lambda b: _hist_from_plane_bytes(b, w, h))
                        cuts_ms.append(max(cuts_ms[-1] + 1, int((exact / fps_safe) * 1000)))
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


# ── Fallback path: PyAV (NVDEC/VideoToolbox or CPU) + CPU reformat ──────────
def _open_for_decode(path: str) -> tuple[av.container.InputContainer, bool]:
    hw = _pyav_hwaccel_handle()
    if hw is not None:
        try:
            return av.open(path, hwaccel=hw, options={"err_detect": "ignore_err"}), True
        except _AV_ERROR as exc:
            log.warning("hwaccel open failed for %s, falling back to CPU: %s", path, exc)
    return av.open(path, options={"err_detect": "ignore_err"}), False


def _cpu_downscale(frame):
    if _FAST_INTERP is not None:
        return frame.reformat(
            width=_DIFF_W, height=_DIFF_H, format="gray",
            interpolation=_FAST_INTERP,
        )
    return frame.reformat(width=_DIFF_W, height=_DIFF_H, format="gray")


def _detect_cuts_via_pyav(path: str, threshold: float,
                          frame_sink: Optional[Callable[[int, np.ndarray], None]] = None) -> List[int]:
    container, _used_nvdec = _open_for_decode(path)
    cuts_ms: List[int] = [0]
    prev_hist: Optional[np.ndarray] = None
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        frame_idx = -1
        last_detect_ms = 0
        recent: list = []   # frames since the previous sample, for the exact cut
        for frame in container.decode(stream):
            frame_idx += 1
            recent.append(frame)
            if frame_idx % _FRAME_STRIDE != 0:
                continue
            between, recent = recent, [frame]
            ts_ms = int(frame.time * 1000) if frame.time is not None else 0
            try:
                small = _cpu_downscale(frame)
            except _AV_ERROR:
                continue
            hist_norm = _hist_from_av_plane(small)
            if frame_sink is not None:
                plane = small.planes[0]
                stride = plane.line_size or _DIFF_W
                frame_sink(ts_ms, np.frombuffer(plane, dtype=np.uint8)
                           .reshape(_DIFF_H, stride)[:, :_DIFF_W].copy())
            if prev_hist is not None:
                diff = float(np.abs(hist_norm - prev_hist).sum()) * 0.5
                if diff > threshold and ts_ms > last_detect_ms + _MIN_CUT_GAP_MS:
                    last_detect_ms = ts_ms
                    exact = _exact_cut_index(len(between) - 1, between,
                                             lambda f: _hist_from_av_plane(_cpu_downscale(f)))
                    first = between[exact] if 0 <= exact < len(between) else frame
                    cut_ts = int(first.time * 1000) if first.time is not None else ts_ms
                    cuts_ms.append(max(cuts_ms[-1] + 1, cut_ts))
            prev_hist = hist_norm
    finally:
        container.close()
    return cuts_ms


# ── Public entry point ──────────────────────────────────────────────────────
def detect_cuts(path: str, threshold: float = 0.30,
                frame_sink: Optional[Callable[[int, np.ndarray], None]] = None) -> List[Tuple[int, int]]:
    """Return list of (start_ms, end_ms) tuples for the detected scenes.

    ``frame_sink(ts_ms, gray_160x90)`` — optional callback receiving every
    sampled low-res frame, so motion analysis reuses this decode pass.

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
    profile = _decode_profile(path)
    if profile is not None:
        try:
            cuts_ms = _detect_cuts_via_ffmpeg_hw(path, threshold, fps, duration_ms, profile,
                                                 frame_sink=frame_sink)
        except Exception as exc:
            log.warning(
                "ffmpeg %s path failed for %s (%s); falling back to PyAV",
                profile["name"], path, exc,
            )
            cuts_ms = None

    if cuts_ms is None:
        cuts_ms = _detect_cuts_via_pyav(path, threshold, frame_sink=frame_sink)

    cuts_ms.append(duration_ms)
    scenes: List[Tuple[int, int]] = []
    for i in range(len(cuts_ms) - 1):
        start = cuts_ms[i]
        end = cuts_ms[i + 1]
        if end - start >= _MIN_SCENE_MS:
            scenes.append((start, end))
    return scenes
