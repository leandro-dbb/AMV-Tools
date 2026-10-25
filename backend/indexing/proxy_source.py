"""Hover-proxy source recorded during the indexing decode (Apple Silicon).

The hover proxies are made after the index queue, from the source video:
one more full software decode of every 1080p episode (~37 s per 24-min
episode on an M1 Pro) just to produce a 240p, 15 fps preview. The
single-pass indexing decode (``single_pass``) already decodes every frame,
so it also feeds them, downscaled, to the VideoToolbox encoder: a 240p
copy of the whole video at a high bitrate (visually lossless at that size).
The proxy pass then cuts and re-encodes that small file instead of decoding
the episode again; it is deleted once the proxies exist.

Only used where the proxy pass would encode with VideoToolbox — the
batched path that reads this file — and when proxies are enabled.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

HEIGHT = 240
FPS = 15
# 3x the default proxy bitrate (1.5x "high"). M1 Pro, 4-min extract: the
# final proxies are 41.2 dB from a lossless 240p reference (41.6 dB when made
# from the source); a higher rate doesn't re-encode any faster (encoder-bound).
BITRATE = "1.5M"


def _dir() -> Path:
    from ..paths import user_data_dir
    return user_data_dir() / "proxies" / "sources"


def path_for(video_path: str) -> Path:
    """Keyed by path, size and mtime: a file replaced in place never reuses
    the copy of the previous one."""
    p = Path(video_path).resolve()
    try:
        st = p.stat()
        ident = f"{p}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        ident = str(p)
    key = hashlib.sha1(ident.encode("utf-8")).hexdigest()[:16]
    return _dir() / f"{key}.mp4"


def existing(video_path: str) -> Optional[Path]:
    try:
        p = path_for(video_path)
    except Exception:
        return None
    return p if p.is_file() and p.stat().st_size > 0 else None


def wanted() -> bool:
    """True when the proxy pass encodes with VideoToolbox (the only path that
    reads the recorded source)."""
    try:
        from ..export.ffmpeg import _hw_h264_encoder
        return _hw_h264_encoder() == "h264_videotoolbox"
    except Exception:
        return False


def filter_chain() -> str:
    return f"fps={FPS},scale=-2:{HEIGHT}"


def output_args(tmp_path: Path) -> list[str]:
    return [
        "-c:v", "h264_videotoolbox", "-allow_sw", "1",
        "-b:v", BITRATE, "-an", "-sn", "-f", "mp4", "-y", str(tmp_path),
    ]


def temp_path(video_path: str) -> Path:
    p = path_for(video_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.with_suffix(".part")


def commit(video_path: str) -> None:
    tmp = path_for(video_path).with_suffix(".part")
    try:
        if tmp.is_file() and tmp.stat().st_size > 0:
            tmp.replace(path_for(video_path))
    except OSError as exc:
        log.debug("proxy source not kept for %s: %s", video_path, exc)


def discard(video_path: str) -> None:
    for p in (path_for(video_path), path_for(video_path).with_suffix(".part")):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def discard_all() -> None:
    """Drop every recorded copy (end of a proxy pass: whatever is left
    belongs to videos whose proxies failed or were cancelled)."""
    try:
        d = _dir()
        if d.is_dir():
            for p in list(d.glob("*.mp4")) + list(d.glob("*.part")):
                p.unlink(missing_ok=True)
    except OSError:
        pass
