"""Decoding + content fingerprint.

Everything is decoded through ffmpeg to mono float32 PCM, so mp3, wav, flac,
m4a, ogg *and the audio track of a video file* all go through one path.  The
file is never re-encoded on disk: ffmpeg streams raw PCM on stdout.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

DEFAULT_SR = 22_050
# An AMV song is 2-6 min; past 20 min it's a whole episode or a mistake, and
# 20 min of 22 kHz float32 is already ~100 MB of RAM.
MAX_SECONDS = 20 * 60
_READ_CHUNK = 1 << 20
# Files above this are fingerprinted from their size + head + tail only
# (a full sha256 of a 4 GB episode takes ~10 s for no benefit).
_FULL_HASH_LIMIT = 256 << 20
_PARTIAL_BLOCK = 8 << 20

AUDIO_EXTENSIONS = (
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wma",
    ".aif", ".aiff",
    # audio track of a video
    ".mp4", ".m4v", ".mkv", ".mov", ".webm", ".avi",
)


class AudioDecodeError(RuntimeError):
    pass


def ffmpeg_exe() -> str:
    # Same resolution as the indexer (AMV_FFMPEG override, bundled
    # imageio-ffmpeg, then PATH), so a user fix applies everywhere.
    from ..indexing.cuts import _ffmpeg_path

    try:
        return _ffmpeg_path()
    except Exception as exc:  # pragma: no cover - depends on the install
        raise AudioDecodeError(f"ffmpeg is required to analyse audio: {exc}") from exc


def _creation_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0  # type: ignore[attr-defined]


def fingerprint_file(path: str | Path) -> str:
    """Content hash — the cache key.

    Stable across renames and copies, so the analysis cache follows the
    audio content and not the path.  Big files (videos) hash their size +
    first and last 8 MB.
    """
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        if size <= _FULL_HASH_LIMIT:
            while True:
                block = fh.read(_READ_CHUNK)
                if not block:
                    break
                h.update(block)
        else:
            h.update(f"partial:{size}:".encode())
            h.update(fh.read(_PARTIAL_BLOCK))
            fh.seek(size - _PARTIAL_BLOCK)
            h.update(fh.read(_PARTIAL_BLOCK))
    return h.hexdigest()


def check_source(path: str | Path) -> Path:
    """Resolve a user-supplied path to an existing regular file."""
    src = Path(path).expanduser()
    if not src.is_file():
        raise FileNotFoundError(str(src))
    return src.resolve()


def decode_audio(path: str | Path, *, sr: int = DEFAULT_SR) -> np.ndarray:
    """Decode any supported audio (or video's audio track) to mono float32."""
    src = check_source(path)
    cmd = [
        ffmpeg_exe(), "-v", "error", "-nostdin",
        # "file:" pins the protocol: a name like "concat:…" or "http://…"
        # can never make ffmpeg open something else than this local file.
        "-i", f"file:{src}",
        "-t", str(MAX_SECONDS),
        "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(int(sr)),
        "-f", "f32le", "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          creationflags=_creation_flags())
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        tail = detail[-1] if detail else f"ffmpeg exited with {proc.returncode}"
        raise AudioDecodeError(f"could not decode audio from {src.name}: {tail}")
    if not proc.stdout:
        raise AudioDecodeError(f"no audio stream in {src.name}")
    y = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32, copy=True)
    if y.size < sr * 2:
        raise AudioDecodeError(f"audio stream too short in {src.name} (under 2 s)")
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y
