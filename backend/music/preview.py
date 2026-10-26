"""Low-resolution preview render: the proposed edit + the song, as an MP4.

Each shot is cut frame-exactly into a small H.264 segment whose frame count
is fixed by its music times (same rounding as the timeline export), the
segments are concatenated without re-encoding and the song is muxed in.  On
CPU a 3-minute edit renders in a minute or two; the user watches it right
away instead of round-tripping through their editor.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Optional

from ..timeline.timecode import ms_to_frames
from .audio import ffmpeg_exe

log = logging.getLogger(__name__)

PREVIEW_FPS = Fraction(24000, 1001)
PREVIEW_HEIGHT = 360
MAX_JOBS_KEPT = 6


def _flags() -> int:
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0  # type: ignore[attr-defined]


@dataclass
class PreviewJob:
    id: str
    status: str = "queued"          # queued | running | done | failed
    progress: float = 0.0
    message: str = ""
    output: Optional[str] = None
    error: Optional[str] = None
    created: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {"id": self.id, "status": self.status, "progress": round(self.progress, 3),
                "message": self.message, "output": self.output, "error": self.error}


def plan_segments(clips: list[dict], fps: Fraction = PREVIEW_FPS) -> list[dict]:
    """Frame counts per clip so cut k lands on frame round(start_k * fps)."""
    out = []
    for c in sorted(clips, key=lambda c: int(c["start_ms"])):
        a = ms_to_frames(int(c["start_ms"]), fps)
        b = ms_to_frames(int(c["end_ms"]), fps)
        out.append({**c, "rec_start": a, "frames": max(1, b - a)})
    # Contiguity: each segment runs until the next one starts.
    for cur, nxt in zip(out, out[1:]):
        cur["frames"] = max(1, nxt["rec_start"] - cur["rec_start"])
    return out


def _render_segment(seg: dict, src: str, dst: Path, fps: Fraction) -> None:
    start = max(0.0, int(seg["src_in_ms"]) / 1000.0)
    vf = (f"scale=-2:{PREVIEW_HEIGHT}:flags=bilinear,fps={fps.numerator}/{fps.denominator},"
          "format=yuv420p,setsar=1")
    cmd = [
        ffmpeg_exe(), "-v", "error", "-nostdin", "-y",
        "-ss", f"{start:.3f}", "-i", f"file:{src}",
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", vf, "-frames:v", str(int(seg["frames"])),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        # No B-frames: the raw streams are glued byte-wise and re-timed by
        # frame count, which needs decode order == display order.
        "-bf", "0", "-g", "48", "-threads", "2",
        "-f", "h264", str(dst),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                          creationflags=_flags())
    if proc.returncode != 0 or not dst.exists():
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["ffmpeg failed"]
        raise RuntimeError(f"segment {seg.get('slot')}: {tail[0]}")


def render_preview(clips: list[dict], scene_paths: dict[int, str], music_path: str,
                   out_path: Path, *, fps: Fraction = PREVIEW_FPS,
                   progress=None, workers: int = 3) -> Path:
    if not clips:
        raise ValueError("the edit is empty")
    segs = plan_segments(clips, fps)
    missing = sorted({str(s["scene_id"]) for s in segs if not scene_paths.get(int(s["scene_id"]))})
    if missing:
        raise FileNotFoundError("unknown shot(s): " + ", ".join(missing))
    tmp = Path(tempfile.mkdtemp(prefix="amv-music-preview-"))
    try:
        done = 0
        lock = threading.Lock()

        def one(i_seg):
            nonlocal done
            i, seg = i_seg
            dst = tmp / f"seg{i:05d}.h264"
            _render_segment(seg, scene_paths[int(seg["scene_id"])], dst, fps)
            with lock:
                done += 1
                if progress:
                    progress(0.95 * done / len(segs), f"{done}/{len(segs)}")
            return dst

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            parts = list(pool.map(one, enumerate(segs)))
        # Annex-B H.264 concatenates byte-wise into one valid stream; the
        # concat demuxer crashes (SIGSEGV) in the bundled ffmpeg 7.0.2 build.
        stream = tmp / "all.h264"
        with open(stream, "wb") as fh:
            for p in parts:
                with open(p, "rb") as src:
                    shutil.copyfileobj(src, fh)
        total_frames = sum(int(sg["frames"]) for sg in segs)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        part = out_path.with_name(out_path.stem + ".part.mp4")
        cmd = [
            ffmpeg_exe(), "-v", "error", "-nostdin", "-y",
            "-r", f"{fps.numerator}/{fps.denominator}", "-f", "h264", "-i", str(stream),
            "-i", f"file:{music_path}",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
            # Explicit length, not -shortest: with a timestamp-less raw video
            # input, -shortest drops the whole audio stream.
            "-t", f"{total_frames / float(fps):.6f}",
            "-movflags", "+faststart", str(part),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              creationflags=_flags())
        if proc.returncode != 0:
            tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["ffmpeg failed"]
            raise RuntimeError(f"preview mux failed: {tail[0]}")
        part.replace(out_path)
        if progress:
            progress(1.0, "done")
        return out_path
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class PreviewManager:
    """One render at a time (CPU-bound); finished files are kept for a few jobs."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.jobs: dict[str, PreviewJob] = {}
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()

    def start(self, clips: list[dict], scene_paths: dict[int, str], music_path: str) -> PreviewJob:
        job = PreviewJob(id=uuid.uuid4().hex[:12])
        with self._lock:
            self.jobs[job.id] = job
            self._prune()

        def run():
            with self._run_lock:
                job.status = "running"
                job.message = "rendering"

                def progress(p, msg):
                    job.progress = p
                    job.message = msg

                try:
                    out = render_preview(clips, scene_paths, music_path,
                                         self.folder / f"preview-{job.id}.mp4", progress=progress)
                    job.output = str(out)
                    job.status = "done"
                except Exception as exc:  # reported to the UI, never crashes the sidecar
                    log.exception("music preview failed")
                    job.status = "failed"
                    job.error = str(exc)

        threading.Thread(target=run, name=f"music-preview-{job.id}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Optional[PreviewJob]:
        return self.jobs.get(job_id)

    def _prune(self) -> None:
        finished = sorted((j for j in self.jobs.values() if j.status in ("done", "failed")),
                          key=lambda j: j.created)
        while len(finished) > MAX_JOBS_KEPT:
            old = finished.pop(0)
            self.jobs.pop(old.id, None)
            if old.output:
                Path(old.output).unlink(missing_ok=True)
