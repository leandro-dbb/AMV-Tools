"""Streaming frame I/O for the Finishing toolbox.

Frames travel through ffmpeg pipes as raw RGB24 — no PNG round-trip on disk,
so an upscale of a 10 s clip does not need gigabytes of temp space. The
decoder uses the same ``-ss``/``-t``/``fps=`` arguments as the roto frame
sampler (``export.ffmpeg.extract_png_frames``), so frame N here is frame N
there; the encoder reuses the user's export codec settings and muxes the
source audio window back in when the output keeps the clip's timing.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from ..indexing.cuts import _ffmpeg_path

log = logging.getLogger(__name__)


def _flags() -> int:
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


@dataclass
class ClipInfo:
    path: str
    start_ms: int
    end_ms: int
    fps: float
    width: int
    height: int
    has_audio: bool

    @property
    def duration_s(self) -> float:
        return max(0.05, (self.end_ms - self.start_ms) / 1000.0)

    @property
    def n_frames(self) -> int:
        """Expected frame count of the decoded window (CFR timeline)."""
        return max(1, int(round(self.duration_s * self.fps)))


def probe_clip(path: str, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> ClipInfo:
    import av

    container = av.open(path, options={"err_detect": "ignore_err"})
    try:
        if not container.streams.video:
            raise RuntimeError(f"no video stream in {path}")
        vs = container.streams.video[0]
        fps = float(vs.average_rate) if vs.average_rate else 24.0
        width = int(vs.codec_context.width or 0)
        height = int(vs.codec_context.height or 0)
        has_audio = bool(container.streams.audio)
        if container.duration is not None:
            duration_ms = int(container.duration / av.time_base * 1000)
        elif vs.duration is not None and vs.time_base is not None:
            duration_ms = int(float(vs.duration * vs.time_base) * 1000)
        else:
            duration_ms = 0
    finally:
        container.close()
    fps = max(1.0, min(240.0, fps))
    s = 0 if start_ms is None else max(0, int(start_ms))
    e = duration_ms if end_ms is None else int(end_ms)
    if duration_ms:
        e = min(e, duration_ms)
    e = max(s + 50, e)
    return ClipInfo(path=path, start_ms=s, end_ms=e, fps=fps, width=width, height=height, has_audio=has_audio)


def _close_pipes(proc: subprocess.Popen, drain: Optional[threading.Thread]) -> None:
    """Close the pipe file objects of an exited ffmpeg (``wait()`` alone
    leaves the fds open until garbage collection — a leak per pass in a long
    finishing queue). stderr is left to its drain thread if still reading."""
    if drain is not None:
        drain.join(timeout=1)
    for f in (proc.stdin, proc.stdout):
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
    if proc.stderr is not None and (drain is None or not drain.is_alive()):
        try:
            proc.stderr.close()
        except Exception:
            pass


class FrameReader:
    """Iterate RGB uint8 frames ``(H, W, 3)`` of a clip window.

    ``max_frames`` caps decoding (used by the 5-frame speed bench). Call
    :meth:`close` (or use as a context manager) to kill ffmpeg early, e.g. on
    job cancellation.
    """

    def __init__(self, clip: ClipInfo, *, max_frames: Optional[int] = None):
        self.clip = clip
        self.width = clip.width
        self.height = clip.height
        cmd = [
            _ffmpeg_path(), "-hide_banner", "-loglevel", "error",
            "-ss", f"{clip.start_ms / 1000.0}",
            "-i", clip.path,
            "-t", f"{clip.duration_s}",
            "-vf", f"fps={clip.fps}",
            "-an", "-sn",
        ]
        if max_frames is not None:
            cmd += ["-frames:v", str(int(max_frames))]
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_flags(),
        )
        self._stderr: list[bytes] = []
        self._err_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._err_thread.start()

    def _drain_stderr(self):
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            self._stderr.append(line)

    def __iter__(self) -> Iterator[np.ndarray]:
        frame_bytes = self.width * self.height * 3
        assert self._proc.stdout is not None
        while True:
            buf = self._proc.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 3)
        self._proc.wait()
        if self._proc.returncode not in (0, None) and not self._killed:
            err = b"".join(self._stderr).decode(errors="replace")[-800:]
            raise RuntimeError(f"ffmpeg decode failed: {err}")

    _killed = False

    def close(self):
        if self._proc.poll() is None:
            self._killed = True
            self._proc.kill()
        try:
            self._proc.wait(timeout=5)
        except Exception:
            pass
        _close_pipes(self._proc, self._err_thread)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_frames(clip: ClipInfo, max_frames: Optional[int] = None) -> list[np.ndarray]:
    with FrameReader(clip, max_frames=max_frames) as r:
        return [f.copy() for f in r]


# ── encoding ────────────────────────────────────────────────────────────────
def suffix_for_codec(codec: str) -> str:
    if codec in ("auto", "libx264", "libx265", "libsvtav1", "h264_nvenc", "h264_videotoolbox"):
        return ".mp4"
    if codec in ("prores_ks", "dnxhr"):
        return ".mov"
    return ".webm"


def video_encoder_args(codec: str, crf: int, width: int, height: int, fps: float) -> list[str]:
    """ffmpeg video encoder flags for a raw-RGB input, mirroring the export
    settings (same codec names as Settings > Export)."""
    from ..export.ffmpeg import _ADAPTIVE_BPP, _ADAPTIVE_MAX_BPS, _ADAPTIVE_MIN_BPS, _hw_h264_encoder

    if codec == "auto":
        codec = _hw_h264_encoder() or "libx264"
    if codec in ("h264_nvenc", "h264_videotoolbox"):
        enc = _hw_h264_encoder() or "libx264"
        target = int(_ADAPTIVE_BPP * width * height * min(120.0, max(1.0, fps)))
        target = max(_ADAPTIVE_MIN_BPS, min(_ADAPTIVE_MAX_BPS * 2, target))
        rate = ["-b:v", str(target), "-maxrate", str(int(target * 1.5)), "-bufsize", str(int(target * 2))]
        if enc == "h264_nvenc":
            return ["-c:v", enc, "-preset", "p5", "-profile:v", "high", "-rc", "vbr", *rate, "-pix_fmt", "yuv420p"]
        if enc == "h264_videotoolbox":
            return ["-c:v", enc, "-profile:v", "high", "-allow_sw", "1", *rate, "-pix_fmt", "yuv420p"]
        return ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if codec in ("libx264", "libx265"):
        return ["-c:v", codec, "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if codec == "libsvtav1":
        return ["-c:v", codec, "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if codec == "libvpx-vp9":
        return ["-c:v", codec, "-crf", str(crf), "-b:v", "0", "-row-mt", "1", "-pix_fmt", "yuv420p"]
    if codec == "prores_ks":
        return ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]
    if codec == "dnxhr":
        return ["-c:v", "dnxhd", "-profile:v", "dnxhr_hq", "-pix_fmt", "yuv422p"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p"]


def audio_encoder_args(suffix: str, audio_mode: str, bitrate_kbps: int = 320) -> list[str]:
    if audio_mode == "mute":
        return []
    if suffix == ".mov":
        return ["-c:a", "pcm_s16le"]
    if suffix == ".webm":
        return ["-c:a", "libopus", "-b:a", "192k"]
    kbps = max(96, min(512, int(bitrate_kbps or 320)))
    return ["-c:a", "aac", "-b:a", f"{kbps}k", "-ar", "48000"]


def _atempo_chain(speed: float) -> str:
    """ffmpeg's atempo accepts 0.5..2 per instance; chain for 1/4, 1/8."""
    parts = []
    while speed < 0.5 - 1e-9:
        parts.append("atempo=0.5")
        speed /= 0.5
    parts.append(f"atempo={speed:.6f}")
    return ",".join(parts)


def _pix_fmt_out(args: list[str]) -> Optional[str]:
    for i, a in enumerate(args[:-1]):
        if a == "-pix_fmt":
            return args[i + 1]
    return None


class FrameWriter:
    """Pipe RGB24 (or GRAY8 / GRAY16LE) frames into ffmpeg and mux the clip's audio.

    ``audio`` is a ClipInfo whose window is muxed (``None`` = silent).
    ``audio_speed`` < 1 time-stretches it (slow motion keeps pitch).
    Output goes to ``<path>.partial<suffix>`` and is renamed on success, so a
    cancelled job never leaves a half-written file under the final name.
    """

    def __init__(self, output_path: str, *, width: int, height: int, fps: float,
                 codec: str = "libx264", crf: int = 18, pix_fmt_in: str = "rgb24",
                 audio: Optional[ClipInfo] = None, audio_mode: str = "encode",
                 audio_bitrate_kbps: int = 320, audio_speed: float = 1.0,
                 extra_video_args: Optional[list[str]] = None, yuv420_in: bool = False):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp_path = self.output_path.with_name(
            f"{self.output_path.stem}.partial{self.output_path.suffix}"
        )
        self.width, self.height = width, height
        self.channels = 1 if pix_fmt_in in ("gray", "gray16le") else 3
        self.dtype = np.uint16 if pix_fmt_in == "gray16le" else np.uint8
        self.frames_written = 0
        suffix = self.output_path.suffix.lower()

        vargs = extra_video_args if extra_video_args is not None else video_encoder_args(codec, crf, width, height, fps)
        # yuv420_in: RGB frames are converted to I420 here (OpenCV, SIMD and
        # multi-threaded, ~5 ms per 4K frame) instead of by ffmpeg's single-
        # threaded swscale, which capped a 4K NVENC encode at ~6 frames/s.
        # Same BT.601 limited-range conversion (luma within 1/255).
        self._to_i420 = (yuv420_in and pix_fmt_in == "rgb24" and width % 2 == 0 and height % 2 == 0
                         and _pix_fmt_out(vargs) == "yuv420p")
        cmd = [
            _ffmpeg_path(), "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "yuv420p" if self._to_i420 else pix_fmt_in,
            "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
        ]
        use_audio = audio is not None and audio.has_audio and audio_mode != "mute"
        if use_audio:
            cmd += ["-ss", f"{audio.start_ms / 1000.0}", "-t", f"{audio.duration_s}", "-i", audio.path]
        cmd += ["-map", "0:v:0"]
        if use_audio:
            cmd += ["-map", "1:a:0?"]
            if abs(audio_speed - 1.0) > 1e-6:
                cmd += ["-af", _atempo_chain(audio_speed)]
            cmd += audio_encoder_args(suffix, audio_mode, audio_bitrate_kbps)
        cmd += vargs
        if suffix == ".mp4":
            cmd += ["-movflags", "+faststart"]
        cmd += ["-y", str(self.tmp_path)]
        log.debug("finish encode: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_flags(),
        )
        self._stderr: list[bytes] = []
        self._err_thread = threading.Thread(target=self._drain, daemon=True)
        self._err_thread.start()

    def _drain(self):
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            self._stderr.append(line)

    @property
    def accepts_i420(self) -> bool:
        """True when this writer feeds ffmpeg I420: callers may then hand it
        :class:`yuv_cuda.I420Frame` frames converted on the GPU."""
        return self._to_i420

    def write(self, frame: np.ndarray) -> None:
        from .yuv_cuda import I420Frame
        if isinstance(frame, I420Frame):
            if not self._to_i420 or (frame.width, frame.height) != (self.width, self.height):
                raise ValueError("I420 frame given to a writer that does not take it")
            assert self._proc.stdin is not None
            try:
                self._proc.stdin.write(frame.data.tobytes())
            except BrokenPipeError:
                err = b"".join(self._stderr).decode(errors="replace")[-800:]
                raise RuntimeError(f"ffmpeg encode failed: {err}")
            self.frames_written += 1
            return
        if frame.dtype != self.dtype:
            raise ValueError(f"FrameWriter expects {np.dtype(self.dtype).name} frames, got {frame.dtype}")
        expected = (self.height, self.width) if self.channels == 1 else (self.height, self.width, 3)
        if frame.shape != expected:
            raise ValueError(f"frame shape {frame.shape} != {expected}")
        assert self._proc.stdin is not None
        if self._to_i420:
            import cv2
            frame = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2YUV_I420)
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.dtype(self.dtype).newbyteorder("<")).tobytes())
        except BrokenPipeError:
            err = b"".join(self._stderr).decode(errors="replace")[-800:]
            raise RuntimeError(f"ffmpeg encode failed: {err}")
        self.frames_written += 1

    def close(self) -> str:
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        self._proc.wait()
        _close_pipes(self._proc, self._err_thread)
        if self._proc.returncode != 0:
            err = b"".join(self._stderr).decode(errors="replace")[-800:]
            self.tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg encode failed: {err}")
        self.tmp_path.replace(self.output_path)
        return str(self.output_path)

    def abort(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except Exception:
            pass
        if self._proc.poll() is None:
            self._proc.kill()
        try:
            self._proc.wait(timeout=5)
        except Exception:
            pass
        _close_pipes(self._proc, self._err_thread)
        self.tmp_path.unlink(missing_ok=True)
