"""Single-pass indexing decode split into chunks decoded in parallel (Apple
Silicon, VideoToolbox).

ffmpeg's VideoToolbox hwaccel decodes one frame at a time per session, which
is why the software decoder won on Apple Silicon (4 min of 1080p H.264:
37.6 s vs 5.5 s). But sessions run side by side on the media engine: with
the single-pass graph of ``single_pass`` (same filters, frames downloaded to
the CPU), 8 sessions decode the 4-min extract in 5.6 s with **11 s of CPU
instead of 36 s**, and 4 min of HEVC 10-bit in 5.7 s instead of 9.6 s
(11.5 s of CPU instead of 56). Decoding was most of the CPU an indexing run
uses, and indexing is CPU-bound on a Mac (the GPU models, motion analysis
and the database work all wait for cores). The decoded frames are bit-
identical to the software decoder's (checked on the H.264 and HEVC
episodes), so the cut detector sees the same pixels.

The video is cut into chunks of ``CHUNK_S`` seconds aligned on the
detector's sampling stride; each chunk runs the single-pass graph on its own
VideoToolbox session and its own consumer thread, and the results are
stitched into exactly what the sequential pass (``single_pass``) returns:

- a sample whose histogram jumps is a *candidate*; whether it becomes a cut
  depends on the previous accepted one (minimum gap). Inside a chunk, a
  candidate more than the gap after the previous candidate (or after the
  chunk start) is accepted whatever came before, and from there on every
  decision is local. Each chunk runs the sequential algorithm from that
  first *certain* cut and keeps every sample before it (its *head*);
- the stitcher replays the sequential algorithm over the heads and the
  chunk boundaries (the first sample of a chunk against the last one of the
  previous chunk), with the open scene's buffer carried over;
- Search+ motion statistics run per chunk and are concatenated with the
  pair that joins two chunks (see ``FrameStatsAccumulator.extend_from``);
- the hover-proxy source (``proxy_source``) is recorded per chunk with
  absolute timestamps and remuxed into one file.

Any failure (a chunk that ends early, an exotic stream) raises and the
caller falls back to the sequential pass. Sub-segmentation (which needs
every sample of a scene) keeps the sequential pass.
"""
from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from . import cuts, single_pass

log = logging.getLogger(__name__)

CHUNK_S = 60.0
WORKERS = 8          # VideoToolbox sessions (M1 Pro: 6 -> 6.35 s, 8 -> 5.6 s, 12 -> 5.4 s per 4 min)
SW_WORKERS = int(os.environ.get("AMV_CHUNK_SW", "0") or 0)   # software decoders next to them
_MAX_HEAD = 1200     # samples kept before a chunk's first certain cut (~3.5 min)
# One decoder thread per session: VideoToolbox does the decoding, and the
# default thread pool only cost memory (8 sessions: 1.86 -> 1.32 GB, same
# speed).
_HWACCEL = ["-threads", "1", "-hwaccel", "videotoolbox"]


def workers() -> int:
    """VideoToolbox sessions: ``AMV_CHUNK_WORKERS``, else the automatic
    tuning's measure for this Mac (``backend.autotune``), else ``WORKERS``."""
    from .. import autotune
    return max(1, int(os.environ.get("AMV_CHUNK_WORKERS") or autotune.value("chunk_workers", WORKERS)))


def supported(path: str) -> bool:
    """macOS with ffmpeg's VideoToolbox, a video the single pass would
    decode in software, long enough to split."""
    # Off by default since v06: the cut detector now decides on every frame
    # with look-back/look-ahead context (``shotdetect``), which this
    # chunk-stitching path does not carry across chunk boundaries yet; the
    # sequential single pass gives the right cuts. ``AMV_CHUNKED=1`` (benchmarks).
    if sys.platform != "darwin" or os.environ.get("AMV_CHUNKED", "0") != "1":
        return False
    prof = cuts._ffmpeg_hw_profile()
    if prof is None or prof.get("name") != "videotoolbox" or not single_pass.supported(path):
        return False
    try:
        return cuts.probe_video(path)["duration_ms"] >= 2 * CHUNK_S * 1000
    except Exception:
        return False


@dataclass
class _Chunk:
    k: int
    first_idx: int            # global index of the chunk's first sample
    n_samples: Optional[int]  # None for the last chunk (until the end)
    start_ts: int = 0
    done: threading.Event = field(default_factory=threading.Event)
    error: Optional[BaseException] = None
    n_read: int = 0
    first_hist: Optional[np.ndarray] = None
    first_gray: Optional[np.ndarray] = None
    last_hist: Optional[np.ndarray] = None
    last_group: Optional[np.ndarray] = None
    last_gray: Optional[np.ndarray] = None
    last_ts: int = 0
    head: list = field(default_factory=list)          # [(idx, ts, rgb)] before the first certain cut
    head_cands: dict = field(default_factory=dict)    # idx -> exact cut ms (provisional candidates)
    certain: Optional[tuple] = None                   # (idx, ts, exact cut ms)
    local_cuts: list = field(default_factory=list)    # cuts from the certain one on
    local_last_detect: int = 0
    medians: dict = field(default_factory=dict)
    tail: list = field(default_factory=list)          # open scene's buffer at the chunk end
    acc: object = None
    proxy_part: Optional[Path] = None


def _exact_cut(group_before: np.ndarray, first_after: np.ndarray, sample: int, fps: float) -> int:
    """``single_pass``'s exact_cut, on the global sample index."""
    gw, gh, stride = cuts._DIFF_W, cuts._DIFF_H, cuts._FRAME_STRIDE
    frames_ = [group_before[:, i * gw:(i + 1) * gw] for i in range(stride)] + [first_after]
    hists = [cuts._hist_from_plane_bytes(np.ascontiguousarray(f).tobytes(), gw, gh) for f in frames_]
    jumps = [float(np.abs(hists[i + 1] - hists[i]).sum()) for i in range(stride)]
    first = (sample - 1) * stride + 1 + int(np.argmax(jumps))
    return int(first / fps * 1000)


def _prune(buf: list, start: int, ts: int) -> None:
    floor = start + (ts - start) // 2
    while len(buf) > 2 and buf[1][0] <= floor:
        buf.pop(0)


def _pick(buf, start: int, end: int):
    """Median sample of [start, end) from ``buf`` (sequential rule), or None."""
    if end - start < cuts._MIN_SCENE_MS or not buf:
        return None
    target = start + (end - start) // 2
    return target, min(buf, key=lambda item: abs(item[0] - target))[1]


def _command(path: str, fps: float, ch: _Chunk, proxy_part: Optional[Path], hw: bool = True) -> list[str]:
    stride = cuts._FRAME_STRIDE
    start_frame = ch.first_idx * stride
    # 2 ms before the chunk's first frame: the accurate seek keeps that frame
    # (container timestamps are rounded to 1 ms) and the fps filter sees it
    # on its first slot — half a frame earlier, it duplicated it.
    seek = ["-ss", f"{max(0.0, start_frame / fps - 0.002):.6f}"] if start_frame > 0 else []
    if ch.n_samples is not None:
        seek += ["-t", f"{ch.n_samples * stride / fps:.6f}"]
    prof = {**cuts._SOFTWARE_PROFILE, "input_args": [*(_HWACCEL if hw else []), *seek]}
    cmd = single_pass._command(path, fps, prof, proxy_part)
    if ch.n_samples is not None:
        j = cmd.index("pipe:1")
        cmd[j:j] = ["-frames:v", str(ch.n_samples)]
    if proxy_part is not None:
        # absolute timestamps, so the parts remux into one timeline
        cmd[-1:-1] = ["-output_ts_offset", f"{start_frame / fps:.6f}"]
    return cmd


def _run_chunk(path: str, fps: float, ch: _Chunk, threshold: float, on_median, make_acc,
               cancelled: Callable[[], bool], procs: set, lock: threading.Lock, hw: bool = True) -> None:
    gw, gh, stride = cuts._DIFF_W, cuts._DIFF_H, cuts._FRAME_STRIDE
    W, RH = single_pass._W, single_pass._RGB_H
    frame_bytes = W * (RH + gh) * 3
    gap = cuts._MIN_CUT_GAP_MS
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    acc = make_acc() if make_acc is not None else None
    ch.acc = acc
    proc = subprocess.Popen(_command(path, fps, ch, ch.proxy_part, hw), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, creationflags=flags, bufsize=frame_bytes * 4)
    with lock:
        procs.add(proc)
    err_tail: deque[bytes] = deque(maxlen=20)
    threading.Thread(target=lambda: err_tail.extend(proc.stderr), daemon=True).start()
    try:
        ch.start_ts = int(ch.first_idx * stride / fps * 1000)
        prev_hist = prev_group = None
        prev_cand_ts: Optional[int] = None
        buf: list = []
        read = proc.stdout.read
        i = -1
        while True:
            raw = read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            i += 1
            if i % 64 == 0 and cancelled():
                raise Cancelled()
            g = ch.first_idx + i
            ts = int(g * stride / fps * 1000)
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(RH + gh, W, 3)
            group = arr[RH:, :gw * stride, 0]
            gray = np.ascontiguousarray(group[:, :gw])
            rgb = np.ascontiguousarray(arr[:RH, :single_pass._RGB_W])
            hist = cuts._hist_from_plane_bytes(gray.tobytes(), gw, gh)
            if acc is not None:
                acc.sink(ts, gray)
            if i == 0:
                ch.first_hist, ch.first_gray = hist, gray
            else:
                diff = float(np.abs(hist - prev_hist).sum()) * 0.5
                if diff > threshold:
                    exact = _exact_cut(prev_group, gray, g, fps)
                    if ch.certain is None:
                        ref = prev_cand_ts if prev_cand_ts is not None else ch.start_ts
                        if ts > ref + gap:
                            ch.certain = (g, ts, exact)
                            ch.local_cuts = [exact]
                            ch.local_last_detect = ts
                        else:
                            ch.head_cands[g] = exact
                    elif ts > ch.local_last_detect + gap:
                        cut = max(ch.local_cuts[-1] + 1, exact)
                        got = _pick(buf, ch.local_cuts[-1], cut)
                        if got is not None:
                            img = Image.fromarray(got[1])
                            ch.medians[got[0]] = img
                            if on_median is not None:
                                on_median(got[0], img)
                        ch.local_cuts.append(cut)
                        ch.local_last_detect = ts
                        buf = []
                    prev_cand_ts = ts
            prev_hist, prev_group = hist, group.copy()
            ch.last_gray, ch.last_ts = gray, ts
            if ch.certain is None:
                ch.head.append((g, ts, rgb))
                if len(ch.head) > _MAX_HEAD:
                    raise RuntimeError("chunk without a certain cut for too long")
            else:
                buf.append((ts, rgb))
                _prune(buf, ch.local_cuts[-1], ts)
        ch.n_read = i + 1
        ch.last_hist, ch.last_group = prev_hist, prev_group
        ch.tail = buf
        proc.stdout.close()
        proc.wait(timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg exited {proc.returncode}: "
                               f"{b''.join(err_tail).decode('utf-8', 'replace')[-300:]!r}")
        if ch.n_read == 0 or (ch.n_samples is not None and ch.n_read != ch.n_samples):
            raise RuntimeError(f"chunk {ch.k}: {ch.n_read} samples, expected {ch.n_samples}")
        if acc is not None:
            acc.finish()
    except BaseException as exc:
        ch.error = exc
        if proc.poll() is None:
            proc.kill()
        if acc is not None:
            try:
                acc.finish()
            except Exception:
                pass
    finally:
        with lock:
            procs.discard(proc)
        ch.done.set()


class Cancelled(Exception):
    """The caller's cancel check fired mid-decode."""


def _boundary_pairs(prev: _Chunk, cur: _Chunk, make_acc):
    """Pair-only accumulator for the pair joining two chunks."""
    b = make_acc()
    b.add(prev.last_ts, prev.last_gray)
    b.add(int(cur.first_idx * cuts._FRAME_STRIDE / cur._fps * 1000), cur.first_gray)
    b.times.clear()
    b.luma.clear()
    return b


def _join_proxy_parts(parts: list[Path], out: Path) -> None:
    import av

    with av.open(str(out), "w", format="mp4") as dst:
        ost = None
        for p in parts:
            with av.open(str(p)) as src:
                ist = src.streams.video[0]
                if ost is None:
                    ost = dst.add_stream_from_template(ist)
                for pkt in src.demux(ist):
                    if pkt.dts is None:
                        continue
                    pkt.stream = ost
                    dst.mux(pkt)


def detect_cuts_with_medians(
    path: str,
    threshold: float = 0.30,
    motion=None,
    on_median: Optional[Callable[[int, Image.Image], None]] = None,
    proxy_source: bool = False,
    cancelled: Optional[Callable[[], bool]] = None,
) -> tuple[list[tuple[int, int]], dict[int, Image.Image]]:
    """Same result as ``single_pass.detect_cuts_with_medians`` (no
    sub-segmentation). ``motion``: a ``FrameStatsAccumulator`` filled as if
    every sample had been fed to it in order."""
    cancelled = cancelled or (lambda: False)
    info = cuts.probe_video(path)
    duration_ms, fps = info["duration_ms"], info["fps"] or 30.0
    stride = cuts._FRAME_STRIDE
    total_samples = int(duration_ms / 1000.0 * fps / stride) + 1
    per = max(1, int(CHUNK_S * fps / stride))
    starts = list(range(0, total_samples, per))
    if len(starts) > 1 and total_samples - starts[-1] < per // 4:
        starts.pop()                       # no tiny last chunk
    chunks = [_Chunk(k, s, (starts[k + 1] - s) if k + 1 < len(starts) else None)
              for k, s in enumerate(starts)]
    for ch in chunks:
        ch._fps = fps
    make_acc = None
    if motion is not None:
        from ..searchplus.analysis import FrameStatsAccumulator

        def make_acc():
            return FrameStatsAccumulator().start_async()

    proxy_parts: list[Path] = []
    if proxy_source:
        from . import proxy_source as ps
        tmp = ps.temp_path(path)
        for ch in chunks:
            ch.proxy_part = tmp.with_name(f"{tmp.stem}.c{ch.k:03d}.mp4")
            proxy_parts.append(ch.proxy_part)

    procs: set = set()
    lock = threading.Lock()
    todo: queue.Queue = queue.Queue()
    for ch in chunks:
        todo.put(ch)
    stop = threading.Event()
    failures: list = []

    def worker(hw: bool):
        while not stop.is_set():
            try:
                ch = todo.get_nowait()
            except queue.Empty:
                return
            _run_chunk(path, fps, ch, threshold, on_median, make_acc,
                       lambda: stop.is_set() or cancelled(), procs, lock, hw)
            if ch.error is not None:
                if not isinstance(ch.error, Cancelled):
                    failures.append(ch.error)
                stop.set()
        while True:   # stopped: mark what is left so the stitcher never waits on it
            try:
                ch = todo.get_nowait()
            except queue.Empty:
                return
            ch.error = Cancelled()
            ch.done.set()

    kinds = ([True] * workers() + [False] * SW_WORKERS)[:len(chunks)]
    threads = [threading.Thread(target=worker, args=(hw,), daemon=True, name=f"{'vt' if hw else 'sw'}-chunk-{i}")
               for i, hw in enumerate(kinds)]
    for t in threads:
        t.start()

    gap = cuts._MIN_CUT_GAP_MS
    cuts_ms: list[int] = [0]
    medians: dict[int, Image.Image] = {}
    last_detect = 0
    buf: list = []
    prev: Optional[_Chunk] = None

    def close(start: int, end: int) -> None:
        got = _pick(buf, start, end)
        if got is not None:
            img = Image.fromarray(got[1])
            medians[got[0]] = img
            if on_median is not None:
                on_median(got[0], img)

    try:
        for ch in chunks:
            while not ch.done.wait(0.5):
                if cancelled():
                    stop.set()
            if ch.error is not None:
                if cancelled() and not failures:
                    raise Cancelled()
                raise RuntimeError(f"chunked decode failed: {(failures or [ch.error])[0]!r}")
            cands = dict(ch.head_cands)
            if prev is not None:
                diff = float(np.abs(ch.first_hist - prev.last_hist).sum()) * 0.5
                if diff > threshold:
                    cands[ch.first_idx] = _exact_cut(prev.last_group, ch.first_gray, ch.first_idx, fps)
            for g, ts, rgb in ch.head:
                if g in cands and ts > last_detect + gap:
                    last_detect = ts
                    cut = max(cuts_ms[-1] + 1, cands[g])
                    close(cuts_ms[-1], cut)
                    cuts_ms.append(cut)
                    buf = []
                buf.append((ts, rgb))
                _prune(buf, cuts_ms[-1], ts)
            if ch.certain is not None:
                _g, ts, exact = ch.certain
                cut = max(cuts_ms[-1] + 1, exact)
                if not (ts > last_detect + gap) or cut != ch.local_cuts[0]:
                    raise RuntimeError(f"chunk {ch.k}: certain cut not certain")
                close(cuts_ms[-1], cut)
                cuts_ms.extend(ch.local_cuts)
                medians.update(ch.medians)
                last_detect = ch.local_last_detect
                buf = list(ch.tail)
            ch.head = []
            ch.tail = []
            prev = ch
        close(cuts_ms[-1], duration_ms)
        cuts_ms.append(duration_ms)
    finally:
        stop.set()
        with lock:
            for p in list(procs):
                if p.poll() is None:
                    p.kill()
        for t in threads:
            t.join(timeout=10)

    if motion is not None:
        parts = []
        for k, ch in enumerate(chunks):
            if k > 0:
                parts.append(_boundary_pairs(chunks[k - 1], ch, make_acc_plain))
            parts.append(ch.acc)
        motion.extend_from(parts)

    if proxy_source:
        from . import proxy_source as ps
        try:
            _join_proxy_parts(proxy_parts, ps.temp_path(path))
            ps.commit(path)
        except Exception as exc:
            log.warning("hover-proxy source not recorded for %s: %s", path, exc)
            ps.discard(path)
        finally:
            for p in proxy_parts:
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass

    scenes = [(cuts_ms[i], cuts_ms[i + 1]) for i in range(len(cuts_ms) - 1)
              if cuts_ms[i + 1] - cuts_ms[i] >= cuts._MIN_SCENE_MS]
    return scenes, medians


def make_acc_plain():
    from ..searchplus.analysis import FrameStatsAccumulator
    return FrameStatsAccumulator()
