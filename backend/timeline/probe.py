"""Media probing for timeline clips.

PyAV is a base dependency of AMV Tools, so it is the primary probe.  When it
cannot open a file we fall back to ``ffprobe`` (system or the sibling of the
imageio-ffmpeg binary).  Results are cached on ``(path, mtime, size)``.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Optional

from .timecode import is_ntsc, parse_rate, rate_string

log = logging.getLogger(__name__)


@dataclass
class SourceInfo:
    path: str
    fps: Fraction = Fraction(25, 1)
    width: int = 0
    height: int = 0
    duration_frames: Optional[int] = None
    has_audio: bool = False
    audio_channels: int = 0
    is_vfr: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def fps_decimal(self) -> float:
        return float(self.fps)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.duration_frames is None:
            return None
        return float(self.duration_frames) / float(self.fps)

    def with_path(self, path: str) -> "SourceInfo":
        return SourceInfo(
            path=path,
            fps=self.fps,
            width=self.width,
            height=self.height,
            duration_frames=self.duration_frames,
            has_audio=self.has_audio,
            audio_channels=self.audio_channels,
            is_vfr=self.is_vfr,
            warnings=list(self.warnings),
        )


_CACHE: dict[tuple[str, float, int], SourceInfo] = {}
_CACHE_LOCK = threading.Lock()


def _creation_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _ffprobe_path() -> Optional[str]:
    found = shutil.which("ffprobe")
    if found:
        return found
    try:
        import imageio_ffmpeg

        ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
        for name in ("ffprobe", "ffprobe.exe"):
            candidate = ffmpeg.with_name(name)
            if candidate.exists():
                return str(candidate)
    except Exception:
        pass
    return None


def probe_source(path: str | os.PathLike, *, use_cache: bool = True) -> SourceInfo:
    """Probe one source file.  Raises ``FileNotFoundError`` if it is absent."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    try:
        stat = p.stat()
        key = (str(p.resolve()), float(stat.st_mtime), int(stat.st_size))
    except OSError:
        key = (str(p), 0.0, 0)
    if use_cache:
        with _CACHE_LOCK:
            cached = _CACHE.get(key)
            if cached is not None:
                return cached

    info: Optional[SourceInfo] = None
    try:
        info = _probe_with_av(str(p))
    except Exception as exc:  # PyAV missing/broken codec → try ffprobe
        log.debug("PyAV probe failed for %s: %s", p, exc)
    if info is None:
        info = _probe_with_ffprobe(str(p))
    if info is None:
        raise RuntimeError(f"could not probe media: {path}")
    if use_cache:
        with _CACHE_LOCK:
            _CACHE[key] = info
    return info


def _fraction_or_none(value) -> Optional[Fraction]:
    if value is None:
        return None
    try:
        frac = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if frac <= 0:
        return None
    return frac


def _probe_with_av(path: str) -> Optional[SourceInfo]:
    import av

    with av.open(path) as container:
        video = next((s for s in container.streams if s.type == "video"), None)
        if video is None:
            return None
        avg = _fraction_or_none(getattr(video, "average_rate", None))
        guessed = _fraction_or_none(getattr(video, "guessed_rate", None))
        fps = avg or guessed or Fraction(25, 1)
        warnings: list[str] = []
        is_vfr = False
        if avg and guessed and float(guessed) > 0:
            if abs(float(avg) - float(guessed)) / float(guessed) > 0.005:
                is_vfr = True
                warnings.append(
                    f"VFR source detected (average {float(avg):.3f} fps, nominal "
                    f"{float(guessed):.3f} fps); using the average rate."
                )
        duration_frames: Optional[int] = None
        if video.frames and video.frames > 0:
            duration_frames = int(video.frames)
        else:
            seconds = None
            if container.duration:
                seconds = float(container.duration) / float(av.time_base)
            elif video.duration and video.time_base:
                seconds = float(video.duration * video.time_base)
            if seconds:
                duration_frames = int(round(seconds * float(fps)))

        audio = [s for s in container.streams if s.type == "audio"]
        channels = 0
        for stream in audio:
            layout = getattr(stream, "layout", None)
            try:
                channels = max(channels, int(layout.channels))
            except Exception:
                codec = getattr(stream, "codec_context", None)
                channels = max(channels, int(getattr(codec, "channels", 0) or 0))
        return SourceInfo(
            path=path,
            fps=fps,
            width=int(video.codec_context.width or 0),
            height=int(video.codec_context.height or 0),
            duration_frames=duration_frames,
            has_audio=bool(audio),
            audio_channels=channels,
            is_vfr=is_vfr,
            warnings=warnings,
        )


def _probe_with_ffprobe(path: str) -> Optional[SourceInfo]:
    binary = _ffprobe_path()
    if not binary:
        return None
    cmd = [
        binary, "-v", "error",
        "-show_entries",
        "stream=index,codec_type,width,height,r_frame_rate,avg_frame_rate,"
        "nb_frames,duration,channels:format=duration",
        "-of", "json", path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                          creationflags=_creation_flags())
    if proc.returncode != 0:
        log.debug("ffprobe failed for %s: %s", path, proc.stderr[-400:])
        return None
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return None
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        return None
    avg = parse_rate(video.get("avg_frame_rate"))
    nominal = parse_rate(video.get("r_frame_rate"))
    fps = avg or nominal or Fraction(25, 1)
    warnings: list[str] = []
    is_vfr = False
    if avg and nominal and abs(float(avg) - float(nominal)) / float(nominal) > 0.005:
        is_vfr = True
        warnings.append(
            f"VFR source detected (average {float(avg):.3f} fps, nominal "
            f"{float(nominal):.3f} fps); using the average rate."
        )
    duration_frames: Optional[int] = None
    nb_frames = video.get("nb_frames")
    if nb_frames not in (None, "", "N/A"):
        try:
            duration_frames = int(nb_frames)
        except ValueError:
            duration_frames = None
    if duration_frames is None:
        seconds = None
        for candidate in (video.get("duration"), (data.get("format") or {}).get("duration")):
            if candidate not in (None, "", "N/A"):
                try:
                    seconds = float(candidate)
                    break
                except ValueError:
                    pass
        if seconds:
            duration_frames = int(round(seconds * float(fps)))
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    channels = max((int(s.get("channels") or 0) for s in audio), default=0)
    return SourceInfo(
        path=path,
        fps=fps,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        duration_frames=duration_frames,
        has_audio=bool(audio),
        audio_channels=channels,
        is_vfr=is_vfr,
        warnings=warnings,
    )


def source_summary(info: SourceInfo) -> dict:
    return {
        "path": info.path,
        "fps": rate_string(info.fps),
        "fps_decimal": info.fps_decimal,
        "ntsc": is_ntsc(info.fps),
        "width": info.width,
        "height": info.height,
        "duration_frames": info.duration_frames,
        "duration_seconds": info.duration_seconds,
        "has_audio": info.has_audio,
        "audio_channels": info.audio_channels,
        "is_vfr": info.is_vfr,
        "warnings": list(info.warnings),
    }
