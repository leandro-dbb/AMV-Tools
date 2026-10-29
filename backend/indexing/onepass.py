"""Single-decode indexing pass for NVIDIA GPUs: cut detection, Search+
motion samples AND the median frame of every scene from one NVDEC decode.

The generic pipeline decodes each video twice — once at 160×90 gray for the
cut detector, then again from the first scene to the end to pull the median
frames at 512×288. On a 22-min episode each pass is a full H.264/HEVC decode,
so the second one roughly doubles the decode time.

Here ffmpeg decodes once (NVDEC), downsizes on the GPU with ``scale_cuda`` and
pipes 512×288 NV12 frames with the detector's 160×90 gray plane stacked
underneath (290 KB each, ~230 MB/s at NVDEC speed — cheap for a pipe). Every
4th frame feeds the cut detector / motion sink, and a sliding buffer keeps the
frames that can still become the median of the scene being decoded: a scene
that started at ``s`` and is still open at ``t`` will end at ``e >= t``, so
its median ``s + (e - s) // 2`` is at least ``s + (t - s) // 2`` and every
older frame can be dropped. When a cut closes the scene, its median frame is
converted to RGB and kept. Memory is bounded by half of the longest scene
(capped — very long shots fall back to the regular extractor).

Only used with the ffmpeg ``cuda`` profile; every other backend keeps the
two-pass path unchanged.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import threading
from collections import deque
from typing import Callable, Iterable, Optional

import numpy as np
from PIL import Image

from . import cuts, shotdetect

log = logging.getLogger(__name__)

FRAME_W, FRAME_H = 512, 288   # same size as frames._GPU_FRAME_W/H
# Max frames held for one open scene (~330 MB of NV12). Past that the scene's
# median is left to the regular extractor.
_MAX_BUFFER = 1500


# ffmpeg stacks the cut detector's 160×90 gray plane — produced by exactly the
# same scale_cuda as the two-pass detector, so cuts and motion samples are
# bit-identical — under the 512×288 frame: one NV12 frame of 512×(288+90).
# Rows 288.. of the luma plane hold the gray image (left 160 columns).
GRAY_W, GRAY_H = cuts._DIFF_W, cuts._DIFF_H
STACK_H = FRAME_H + GRAY_H
FRAME_BYTES = FRAME_W * STACK_H * 3 // 2
_FILTER_CPU_STACK = (
    "[0:v]split=2[a][b];"
    f"[a]scale_cuda={FRAME_W}:{FRAME_H}:format=nv12,hwdownload,format=nv12[big];"
    f"[b]scale_cuda={GRAY_W}:{GRAY_H}:format=nv12,hwdownload,format=nv12,format=gray,"
    f"scale=out_range=full:flags=neighbor,format=nv12,pad={FRAME_W}:{GRAY_H}[small];"
    "[big][small]vstack"
)
# Preferred: the stacking happens on the GPU (overlay_cuda onto a scratch
# canvas), so each frame costs one device→host copy and no CPU filtering —
# ~44 s instead of ~50 s for a 22-min episode, against a 42 s NVDEC floor.
# The gray plane then holds scale_cuda's limited-range luma; ``_gray_of``
# applies swscale's limited→full expansion, which is exactly this per-value
# rounding (checked over all 256 levels), so cuts stay identical.
_FILTER_GPU_STACK = (
    "[0:v]split=3[a][b][c];"
    f"[a]scale_cuda={FRAME_W}:{STACK_H}:format=nv12[canvas];"
    f"[b]scale_cuda={FRAME_W}:{FRAME_H}:format=nv12[big];"
    f"[c]scale_cuda={GRAY_W}:{GRAY_H}:format=nv12[small];"
    f"[canvas][big]overlay_cuda=0:0[t];[t][small]overlay_cuda=0:{FRAME_H},hwdownload,format=nv12"
)
_TV_TO_PC = np.clip(np.round((np.arange(256) - 16) * 255.0 / 219.0), 0, 255).astype(np.uint8)
_HAS_OVERLAY_CUDA: Optional[bool] = None


def _gpu_stacking() -> bool:
    global _HAS_OVERLAY_CUDA
    if _HAS_OVERLAY_CUDA is None:
        try:
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            out = subprocess.run([cuts._ffmpeg_path(), "-hide_banner", "-filters"], capture_output=True,
                                 text=True, timeout=10, creationflags=flags).stdout or ""
            _HAS_OVERLAY_CUDA = "overlay_cuda" in out
        except Exception:
            _HAS_OVERLAY_CUDA = False
    return _HAS_OVERLAY_CUDA


# Hover proxies (NVENC): the indexing decode also encodes a 240p
# copy of the video (scale_cuda + NVENC, no extra NVDEC pass), and the proxy
# pass re-cuts that small file instead of decoding the 1080p source again.
# Set by the app while a queue runs with proxies on; None = off.
_proxy_cache_dir = None
PROXY_SOURCE_H = 240


_DECODERS: Optional[str] = None


def has_decoder(name: str) -> bool:
    global _DECODERS
    if _DECODERS is None:
        try:
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            _DECODERS = subprocess.run([cuts._ffmpeg_path(), "-hide_banner", "-decoders"], capture_output=True,
                                       text=True, timeout=10, creationflags=flags).stdout or ""
        except Exception:
            _DECODERS = ""
    return f" {name} " in _DECODERS


def set_proxy_cache(directory) -> None:
    global _proxy_cache_dir
    _proxy_cache_dir = directory


def _proxy_source_name(video_path: str) -> str:
    import hashlib
    import os
    try:
        st = os.stat(video_path)
        key = f"{os.path.abspath(video_path)}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        key = os.path.abspath(video_path)
    return hashlib.sha1(key.encode("utf-8", "surrogatepass")).hexdigest()[:20] + ".mkv"


def proxy_source_for(video_path: str, directory=None):
    """The finished 240p proxy source of ``video_path`` (Path) or None."""
    from pathlib import Path
    d = directory if directory is not None else _proxy_cache_dir
    if d is None:
        return None
    f = Path(d) / _proxy_source_name(video_path)
    return f if f.is_file() and f.stat().st_size > 0 else None


class Cancelled(Exception):
    """The caller's cancel check fired mid-decode."""


def available() -> bool:
    prof = cuts._ffmpeg_hw_profile()
    return prof is not None and prof.get("name") == "cuda"


def _gray_of(buf: bytes, lut: Optional[np.ndarray] = None) -> np.ndarray:
    y = np.frombuffer(buf, dtype=np.uint8, count=FRAME_W * STACK_H).reshape(STACK_H, FRAME_W)
    g = y[FRAME_H:, :GRAY_W]
    return lut[g] if lut is not None else np.ascontiguousarray(g)


# (Kr, Kb) of the YUV→RGB matrices swscale picks from the stream's colour
# metadata — BT.601 when unspecified, like ffmpeg.
_MATRICES = {"bt709": (0.2126, 0.0722), "bt601": (0.299, 0.114)}


def _nv12_to_rgb(buf: bytes, matrix: str = "bt601", full_range: bool = False) -> Image.Image:
    """RGB of the 512×288 part of a stacked frame — the same conversion the
    two-pass extractor gets from ffmpeg (stream matrix, nearest chroma), so
    the models see the same pixels (≥ 38 dB vs that path on anime)."""
    plane = np.frombuffer(buf, dtype=np.uint8).reshape(STACK_H * 3 // 2, FRAME_W)
    y = plane[:FRAME_H].astype(np.float32)
    uv = plane[STACK_H:STACK_H + FRAME_H // 2].reshape(FRAME_H // 2, FRAME_W // 2, 2).astype(np.float32)
    uv = uv.repeat(2, axis=0).repeat(2, axis=1) - 128.0
    kr, kb = _MATRICES.get(matrix, _MATRICES["bt601"])
    kg = 1.0 - kr - kb
    if full_range:
        cy, cc = 1.0, 1.0
    else:
        y = y - 16.0
        cy, cc = 255.0 / 219.0, 255.0 / 224.0
    yy = y * cy
    u, v = uv[..., 0] * cc, uv[..., 1] * cc
    r = yy + 2.0 * (1.0 - kr) * v
    b = yy + 2.0 * (1.0 - kb) * u
    g = yy - (2.0 * kb * (1.0 - kb) * u + 2.0 * kr * (1.0 - kr) * v) / kg
    return Image.fromarray(np.clip(np.dstack([r, g, b]).round(), 0, 255).astype(np.uint8))


_GPU_RGB = None   # cached (torch, device) or False


def _nv12_to_rgb_gpu(buf: bytes, matrix: str = "bt601", full_range: bool = False) -> Image.Image:
    """:func:`_nv12_to_rgb` on the CUDA device (same float maths), the
    numpy version if torch/CUDA is unavailable."""
    global _GPU_RGB
    if _GPU_RGB is None:
        try:
            import torch
            _GPU_RGB = (torch, torch.device("cuda")) if torch.cuda.is_available() else False
        except Exception:
            _GPU_RGB = False
    if not _GPU_RGB:
        return _nv12_to_rgb(buf, matrix, full_range)
    torch, dev = _GPU_RGB
    try:
        with torch.inference_mode():
            plane = torch.frombuffer(bytearray(buf), dtype=torch.uint8).to(dev).view(STACK_H * 3 // 2, FRAME_W)
            y = plane[:FRAME_H].float()
            uv = plane[STACK_H:STACK_H + FRAME_H // 2].reshape(FRAME_H // 2, FRAME_W // 2, 2).float()
            uv = uv.repeat_interleave(2, 0).repeat_interleave(2, 1) - 128.0
            kr, kb = _MATRICES.get(matrix, _MATRICES["bt601"])
            kg = 1.0 - kr - kb
            if full_range:
                cy, cc = 1.0, 1.0
            else:
                y = y - 16.0
                cy, cc = 255.0 / 219.0, 255.0 / 224.0
            yy = y * cy
            u, v = uv[..., 0] * cc, uv[..., 1] * cc
            r = yy + 2.0 * (1.0 - kr) * v
            b = yy + 2.0 * (1.0 - kb) * u
            g = yy - (2.0 * kb * (1.0 - kb) * u + 2.0 * kr * (1.0 - kr) * v) / kg
            rgb = torch.stack([r, g, b], -1).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()
        return Image.fromarray(rgb)
    except Exception as exc:
        log.debug("GPU NV12->RGB failed (%s); numpy from now on", exc)
        _GPU_RGB = False
        return _nv12_to_rgb(buf, matrix, full_range)


def _color_info(path: str) -> tuple[str, bool]:
    """(matrix, full_range) from the stream metadata."""
    try:
        import av
        with av.open(path, options={"err_detect": "ignore_err"}) as c:
            cc = c.streams.video[0].codec_context
            # AVColorSpace 1 = BT.709; AVColorRange 2 = JPEG/full.
            return ("bt709" if int(cc.colorspace or 0) == 1 else "bt601"), int(cc.color_range or 0) == 2
    except Exception:
        return "bt601", False


def detect_cuts_and_medians(
    path: str, threshold: float,
    frame_sink: Optional[Callable[[int, np.ndarray], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> tuple[list[tuple[int, int]], dict[int, Image.Image]]:
    """Same scenes as :func:`cuts.detect_cuts` plus ``{median_ms: RGB image}``
    for the scenes whose median frame was captured (median_ms computed exactly
    like the pipeline: ``start + (end - start) // 2``). Raises on failure —
    callers fall back to the two-pass path."""
    info = cuts.probe_video(path)
    duration_ms, fps = info["duration_ms"], info["fps"]
    if duration_ms <= 0:
        return [], {}
    profile = cuts._ffmpeg_hw_profile()
    if profile is None or profile.get("name") != "cuda":
        raise RuntimeError("single-pass indexing needs the ffmpeg cuda profile")

    fb = FRAME_BYTES
    gpu_stack = _gpu_stacking()
    graph = (_FILTER_GPU_STACK if gpu_stack else _FILTER_CPU_STACK) + "[raw]"
    proxy_args: list[str] = []
    proxy_tmp = proxy_dst = None
    if _proxy_cache_dir is not None:
        from pathlib import Path
        Path(_proxy_cache_dir).mkdir(parents=True, exist_ok=True)
        proxy_dst = Path(_proxy_cache_dir) / _proxy_source_name(path)
        proxy_tmp = proxy_dst.with_suffix(".part.mkv")
        proxy_dst.unlink(missing_ok=True)
        # one more branch off the decoded frames: "[0:v]split=N[a]…" → N+1
        n = 3 if gpu_stack else 2
        graph = graph.replace(f"[0:v]split={n}", f"[0:v]split={n + 1}[pxin]", 1)
        # already at the proxies' 15 fps: a third fewer frames to encode
        # here and to decode again in the proxy pass (the frame picked for a
        # 15 fps slot may be the neighbouring source frame, 41 ms away)
        graph += f";[pxin]fps=15,scale_cuda=-2:{PROXY_SOURCE_H}:format=nv12[px]"
        # quality well above the proxies' 300k-1000k: the re-cut is the
        # only lossy step that shows (QP 20: half the bitstream of QP 16 to
        # write and to parse again, 4-min extract: 21 MB)
        proxy_args = ["-map", "[px]", "-c:v", "h264_nvenc", "-preset", "p1", "-rc", "constqp",
                      "-qp", "20", "-g", "30",
                      "-f", "matroska", "-y", str(proxy_tmp)]
    matrix, full_range = _color_info(path)
    # GPU stacking leaves the luma in the source's range: expand limited
    # range like swscale's gray conversion did (full-range sources: as is).
    gray_lut = _TV_TO_PC if gpu_stack and not full_range else None
    cmd = [
        cuts._ffmpeg_path(),
        "-hide_banner", "-nostats", "-loglevel", "error",
        "-fflags", "+discardcorrupt",
        *profile["input_args"],
        "-i", path,
        "-filter_complex", graph,
        "-map", "[raw]", "-an", "-sn",
        "-fps_mode", "cfr",
        "-f", "rawvideo", "-pix_fmt", "nv12",
        "pipe:1",
        *proxy_args,
    ]
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    # buffered pipes (stderr is read line by line: unbuffered, that was one
    # syscall per byte); the frames are read from stdout's raw file
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=flags)
    # Drain stderr concurrently: a damaged stream can log more than a pipe
    # buffer holds, which would otherwise stall ffmpeg (and us) for good.
    err_tail: deque[bytes] = deque(maxlen=20)
    drain = threading.Thread(target=lambda: err_tail.extend(proc.stderr), daemon=True)
    drain.start()

    try:
        assert proc.stdout is not None
        cuts_ms, medians, n_frames = scan_stream(getattr(proc.stdout, "raw", proc.stdout), fps, duration_ms, threshold, frame_sink,
                                                 to_rgb=lambda b: _nv12_to_rgb_gpu(b, matrix, full_range),
                                                 cancelled=cancelled, gray_lut=gray_lut)
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
    if proxy_tmp is not None:
        if proc.returncode == 0 and n_frames > 0 and proxy_tmp.is_file():
            proxy_tmp.replace(proxy_dst)
        else:
            proxy_tmp.unlink(missing_ok=True)
    if proc.returncode not in (0, None) or n_frames == 0:
        drain.join(timeout=2)
        tail = b"".join(err_tail).decode("utf-8", errors="replace")[-500:]
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}: {tail!r}")

    scenes = [(cuts_ms[i], cuts_ms[i + 1]) for i in range(len(cuts_ms) - 1)
              if cuts_ms[i + 1] - cuts_ms[i] >= cuts._MIN_SCENE_MS]
    return scenes, medians


class _Scanner:
    """Core of the single pass over a preallocated circular buffer of
    stacked frames: the pipe is read straight into it (``readinto``, blocks
    of frames, no 290 KB allocation per frame) and only the sampled frames
    get Python work. Same decisions and medians as the frame-by-frame loop
    it replaces: the ring of median candidates is the index range
    ``[keep, n)``."""

    def __init__(self, fps: float, threshold: float, frame_sink, to_rgb, cancelled, gray_lut):
        self.fps = fps if fps > 0 else 30.0
        self.threshold = threshold
        self.frame_sink = frame_sink
        self.to_rgb = to_rgb or _nv12_to_rgb
        self.cancelled = cancelled
        self.gray_lut = gray_lut
        self.max_ring = _MAX_BUFFER
        self.cap = _MAX_BUFFER + 2 * _BLOCK + cuts._FRAME_STRIDE + 2
        self.buf = np.empty((self.cap, FRAME_BYTES), dtype=np.uint8)
        self.n = 0                 # frames received
        self.keep = 0              # oldest frame still a median candidate
        self.overflow = False      # open scene outgrew the buffer
        self.cuts_ms: list[int] = [0]
        self.medians: dict[int, Image.Image] = {}
        self.detector = shotdetect.ShotDetector(threshold)
        # a cut is confirmed this late (the detector's flash look-ahead)
        self.stride_ms = int((shotdetect.LOOKAHEAD + 1) * 1000 / self.fps) + 1

    # frame index <-> timestamp, exactly as the pipeline computes it
    def ts(self, idx: int) -> int:
        return int((idx / self.fps) * 1000)

    def first_at(self, t_ms: int) -> int:
        """Smallest frame index whose timestamp is >= t_ms."""
        i = max(0, int(t_ms * self.fps / 1000.0))
        while i > 0 and self.ts(i - 1) >= t_ms:
            i -= 1
        while self.ts(i) < t_ms:
            i += 1
        return i

    def frame(self, idx: int) -> np.ndarray:
        return self.buf[idx % self.cap]

    def hist_of(self, b) -> np.ndarray:
        return cuts._hist_from_plane_bytes(_gray_of(b, self.gray_lut).tobytes(), cuts._DIFF_W, cuts._DIFF_H)

    def close_scene(self, start: int, end: int) -> None:
        m = start + (end - start) // 2
        if self.overflow:
            return
        i = max(self.keep, self.first_at(m))
        if i < self.n:
            self.medians[m] = self.to_rgb(self.frame(i).tobytes())

    def write_view(self, want: int) -> np.ndarray:
        """Up to ``want`` free slots after the last frame (contiguous)."""
        start = self.n % self.cap
        return self.buf[start:start + min(want, self.cap - start)]

    def commit(self, k: int) -> None:
        """Process the ``k`` frames just written after frame ``n``: every
        frame goes to the cut detector, the sampled ones to motion analysis."""
        stride = cuts._FRAME_STRIDE
        first, end = self.n, self.n + k
        self.n = end
        if self.cancelled is not None and first // 256 != end // 256 and self.cancelled():
            raise Cancelled()
        for idx in range(first, end):
            gray = _gray_of(self.frame(idx), self.gray_lut)
            if self.frame_sink is not None and idx % stride == 0:
                self.frame_sink(self.ts(idx), gray)
            self._cuts(self.detector.push(gray))
            if idx % stride == 0:
                self._advance_floor(idx)
        self._advance_floor(end - 1)

    def _cuts(self, found: list[int]) -> None:
        for c in found:
            cut = max(self.cuts_ms[-1] + 1, self.ts(c))
            self.close_scene(self.cuts_ms[-1], cut)
            self.cuts_ms.append(cut)
            self.overflow = False
            self.keep = max(self.keep, self.first_at(cut))

    def _advance_floor(self, idx: int) -> None:
        # The open scene's median can't be earlier than half-way to now —
        # minus a stride: the cut that closes it may land up to a stride
        # before the sample that detects it.
        start = self.cuts_ms[-1]
        ts = self.ts(idx)
        self.keep = max(self.keep, self.first_at(start + max(0, ts - start - self.stride_ms) // 2))
        if idx + 1 - self.keep > self.max_ring:
            self.overflow = True
            self.keep = idx + 1

    def finish(self, duration_ms: int) -> tuple[list[int], dict[int, Image.Image], int]:
        self._cuts(self.detector.finish())
        if self.n:
            self.close_scene(self.cuts_ms[-1], duration_ms)
        self.cuts_ms.append(duration_ms)
        return self.cuts_ms, self.medians, self.n


_BLOCK = 32   # frames per pipe read


def scan_stream(raw, fps: float, duration_ms: int, threshold: float, frame_sink=None, to_rgb=None,
                cancelled=None, gray_lut=None) -> tuple[list[int], dict[int, Image.Image], int]:
    """:func:`scan` reading an unbuffered pipe straight into the buffer."""
    sc = _Scanner(fps, threshold, frame_sink, to_rgb, cancelled, gray_lut)
    fb = FRAME_BYTES
    eof = False
    while not eof:
        view = sc.write_view(_BLOCK)
        flat = memoryview(view.reshape(-1))
        want, got = len(flat), 0
        while got < want:
            r = raw.readinto(flat[got:])
            if not r:
                eof = True
                break
            got += r
        if got // fb:
            sc.commit(got // fb)
    return sc.finish(duration_ms)


def scan(chunks: Iterable[bytes], fps: float, duration_ms: int, threshold: float,
         frame_sink: Optional[Callable[[int, np.ndarray], None]] = None,
         to_rgb: Optional[Callable[[bytes], Image.Image]] = None,
         cancelled: Optional[Callable[[], bool]] = None,
         gray_lut: Optional[np.ndarray] = None,
         ) -> tuple[list[int], dict[int, Image.Image], int]:
    """Core of the single pass, fed with stacked NV12 frames (see
    ``_FILTER``) in decode order. Returns (cut list ending with ``duration_ms``, medians, frame count).
    Cut decisions match ``cuts._detect_cuts_via_ffmpeg_hw`` (same stride,
    histogram distance, minimum gap)."""
    sc = _Scanner(fps, threshold, frame_sink, to_rgb, cancelled, gray_lut)
    for buf in chunks:
        sc.write_view(1)[0] = np.frombuffer(buf, dtype=np.uint8, count=FRAME_BYTES)
        sc.commit(1)
    return sc.finish(duration_ms)
