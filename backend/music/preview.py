"""Low-resolution preview render: the proposed edit + the song, as an MP4.

Each shot is cut frame-exactly into a small H.264 segment whose frame count
is fixed by its music times (same rounding as the timeline export), the
segments are concatenated without re-encoding and the song is muxed in.  On
CPU a 3-minute edit renders in a minute or two; the user watches it right
away instead of round-tripping through their editor.
"""
from __future__ import annotations

import logging
import re
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
PREVIEW_WIDTH = 640
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
    # Slots whose shot turned out to contain a hard cut (off-beat cut).
    cut_slots: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"id": self.id, "status": self.status, "progress": round(self.progress, 3),
                "message": self.message, "output": self.output, "error": self.error,
                "cut_slots": list(self.cut_slots)}


def plan_segments(clips: list[dict], fps: Fraction = PREVIEW_FPS) -> list[dict]:
    """Frame counts per clip so cut k lands on frame round(start_k * fps).

    Holes in the edit (a slot no shot could fill, the start of the song
    before the first clip) become black ``gap`` segments, so the picture
    never drifts from the music and no shot is stretched past its out point.
    """
    out: list[dict] = []
    cursor = 0
    for c in sorted(clips, key=lambda c: int(c["start_ms"])):
        a = max(cursor, ms_to_frames(int(c["start_ms"]), fps))
        b = ms_to_frames(int(c["end_ms"]), fps)
        if b <= a:
            continue
        if a > cursor:
            out.append({"gap": True, "rec_start": cursor, "frames": a - cursor})
        out.append({**c, "rec_start": a, "frames": b - a})
        cursor = b
    return out


# scdet score (0-100) above which a frame starts a new shot; 14 finds the
# same cuts as select='gt(scene,0.3)' on the test episodes.  scdet (unlike
# select) passes every frame, so a shot without any cut is not an error.
SCENE_CUT_THRESHOLD = 14
_PTS_RE = re.compile(r"lavfi\.scd\.time:\s*([0-9.]+)")


def _timeout_for(frames: int) -> float:
    # Generous (a seek in a long HEVC episode on a slow CPU), but a hung
    # ffmpeg (unreachable network share) must not block every preview.
    return 120.0 + frames * 0.5


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              creationflags=_flags(), timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg took more than {int(timeout)} s") from exc


def _render_gap(frames: int, dst: Path, fps: Fraction) -> None:
    cmd = [
        ffmpeg_exe(), "-v", "error", "-nostdin", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s={PREVIEW_WIDTH}x{PREVIEW_HEIGHT}:r={fps.numerator}/{fps.denominator}",
        "-frames:v", str(int(frames)), "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-bf", "0", "-g", "48",
        "-f", "h264", str(dst),
    ]
    proc = _run(cmd, timeout=_timeout_for(frames))
    if proc.returncode != 0 or not dst.exists():
        raise RuntimeError("could not render a gap: " + proc.stderr.decode("utf-8", "replace")[-200:])


def _render_segment(seg: dict, src: str, dst: Path, fps: Fraction) -> list[float]:
    """Encode one shot; returns the hard cuts seen *inside* it (seconds from
    its start) — the indexer missed those, and they would cut off-beat."""
    start = max(0.0, int(seg["src_in_ms"]) / 1000.0)
    frames = int(seg["frames"])
    # Fixed 640x360 (letterboxed): segments are glued byte-wise, so every
    # one must have the same picture size whatever the episode's aspect.
    vf = (f"scale={PREVIEW_WIDTH}:{PREVIEW_HEIGHT}:force_original_aspect_ratio=decrease:flags=bilinear,"
          f"pad={PREVIEW_WIDTH}:{PREVIEW_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
          f"fps={fps.numerator}/{fps.denominator},format=yuv420p,setsar=1")
    graph = (f"[0:v:0]{vf},split=2[enc][det];"
             f"[det]scale=160:-2,scdet=threshold={SCENE_CUT_THRESHOLD}[cuts]")
    cmd = [
        ffmpeg_exe(), "-hide_banner", "-v", "info", "-nostats", "-nostdin", "-y",
        "-ss", f"{start:.3f}", "-i", f"file:{src}",
        "-filter_complex", graph,
        "-map", "[enc]", "-an", "-sn", "-dn", "-frames:v", str(frames),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        # No B-frames: the raw streams are glued byte-wise and re-timed by
        # frame count, which needs decode order == display order.
        "-bf", "0", "-g", "48", "-threads", "2",
        "-f", "h264", str(dst),
        "-map", "[cuts]", "-frames:v", str(frames), "-f", "null", "-",
    ]
    proc = _run(cmd, timeout=_timeout_for(frames))
    err = proc.stderr.decode("utf-8", "replace")
    if proc.returncode != 0 or not dst.exists():
        lines = [ln for ln in err.strip().splitlines() if "lavfi.scd" not in ln]
        raise RuntimeError(f"segment {seg.get('slot')}: {(lines[-1:] or ['ffmpeg failed'])[0]}")
    frame_s = 1.0 / float(fps)
    length = frames * frame_s
    return [float(t) for t in _PTS_RE.findall(err)
            if 1.5 * frame_s < float(t) < length - 0.5 * frame_s]


def render_preview(clips: list[dict], scene_paths: dict[int, str], music_path: str,
                   out_path: Path, *, fps: Fraction = PREVIEW_FPS,
                   progress=None, workers: int = 3, inner_cuts: Optional[list] = None) -> Path:
    """Render the preview; ``inner_cuts`` (if a list) receives the shots that
    turned out to contain a hard cut: {slot, scene_id, path, cuts_ms}."""
    if not clips:
        raise ValueError("the edit is empty")
    segs = plan_segments(clips, fps)
    missing = sorted({str(s["scene_id"]) for s in segs
                      if not s.get("gap") and not scene_paths.get(int(s["scene_id"]))})
    if missing:
        raise FileNotFoundError("unknown shot(s): " + ", ".join(missing))
    tmp = Path(tempfile.mkdtemp(prefix="amv-music-preview-"))
    try:
        done = 0
        lock = threading.Lock()
        found: list[dict] = []

        def one(i_seg):
            nonlocal done
            i, seg = i_seg
            dst = tmp / f"seg{i:05d}.h264"
            if seg.get("gap"):
                _render_gap(int(seg["frames"]), dst, fps)
                inner = []
            else:
                inner = _render_segment(seg, scene_paths[int(seg["scene_id"])], dst, fps)
            if inner:
                with lock:
                    found.append({"slot": int(seg.get("slot", i)), "scene_id": int(seg["scene_id"]),
                                  "path": scene_paths[int(seg["scene_id"])],
                                  "cuts_ms": [int(seg["src_in_ms"]) + int(round(t * 1000)) for t in inner]})
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
        total_frames = int(segs[-1]["rec_start"]) + int(segs[-1]["frames"])
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
        proc = _run(cmd, timeout=_timeout_for(total_frames))
        if proc.returncode != 0:
            tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["ffmpeg failed"]
            raise RuntimeError(f"preview mux failed: {tail[0]}")
        part.replace(out_path)
        if inner_cuts is not None:
            inner_cuts.extend(sorted(found, key=lambda x: x["slot"]))
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
        # Previews of earlier sessions are not reachable any more.
        for old in Path(folder).glob("preview-*.mp4"):
            try:
                if time.time() - old.stat().st_mtime > 3600:
                    old.unlink()
            except OSError:
                pass
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()

    def start(self, clips: list[dict], scene_paths: dict[int, str], music_path: str,
              on_inner_cuts=None) -> PreviewJob:
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
                    inner: list[dict] = []
                    out = render_preview(clips, scene_paths, music_path,
                                         self.folder / f"preview-{job.id}.mp4", progress=progress,
                                         inner_cuts=inner)
                    job.cut_slots = [int(x["slot"]) for x in inner]
                    if inner and on_inner_cuts is not None:
                        try:
                            on_inner_cuts(inner)
                        except Exception:
                            log.exception("could not store the cuts found in shots")
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
                try:
                    Path(old.output).unlink(missing_ok=True)
                except OSError:  # still open by the player (Windows)
                    pass
