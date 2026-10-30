"""End-to-end indexing pipeline (Roadmap §5 Feature 1).

Two-phase architecture for 8 GB VRAM friendliness:

  Pre-work (shared, always done):
      - ffprobe duration / fps / resolution
      - ffmpeg hard-cut detection (`select=gt(scene,T)`)
      - optional sub-segmentation: for scenes > 4 s, sample every 0.5 s, run
        wd-tagger drift detection, split where embedding distance spikes
      - extract median frame per final segment, encode JPEG thumbnail
      - upsert video row, insert scenes with thumbnails (embedding = NULL, no tags)

  Phase A — "tag" (if 'tag' in phases):
      - load wd-tagger, batch through all median frames, insert tags
      - offload tagger before phase B so it doesn't compete with SigLIP for VRAM

  Phase B — "embed" (if 'embed' in phases):
      - load SigLIP, batch through all median frames, update scene embeddings

  Final:
      - set video status: "completed" (both), "tags_only", or "embeddings_only"

Holding the median frames in RAM between the two phases avoids re-decoding the
video. ~621 segments × ~1.4 MB ≈ 870 MB peak — acceptable.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

from ..db import queries, schema
from ..models import SigLIP2Model, WDTaggerModel
from . import chunked, cuts, frames, nvc_pass, onepass, proxy_source, single_pass, streaming, sub_segment

log = logging.getLogger(__name__)

LONG_SCENE_THRESHOLD_MS = 4_000
SUB_SAMPLE_STEP_MS = 500

ALL_PHASES: tuple[str, ...] = ("tag", "embed")


def _display_name(filepath: str) -> str:
    stem = Path(filepath).stem
    return stem.replace("_", " ").replace(".", " ").strip()


def _median_ms(start_ms: int, end_ms: int) -> int:
    return start_ms + (end_ms - start_ms) // 2


def _status_for_phases(phases: Sequence[str]) -> str:
    has_tag = "tag" in phases
    has_embed = "embed" in phases
    if has_tag and has_embed:
        return "completed"
    if has_tag:
        return "tags_only"
    if has_embed:
        return "embeddings_only"
    return "completed"


def _is_video_current_for_phases(
    db_path: str, video_path: str, model_version: str, phases: Sequence[str]
) -> bool:
    """True if the DB already covers every requested phase for this video.

    Skips only when *all* requested phases are present:
      - status='completed' covers both tag and embed
      - status='tags_only' covers tag
      - status='embeddings_only' covers embed
    """
    if not os.path.exists(video_path):
        return False
    with schema.get_conn(db_path) as conn:
        row = conn.execute(
            """SELECT modified_at, model_version, status
                 FROM videos
                WHERE filepath = ?""",
            (video_path,),
        ).fetchone()
    if not row:
        return False
    modified_at, stored_model, status = row
    if stored_model != model_version:
        return False
    if abs(float(modified_at or 0) - os.path.getmtime(video_path)) > 1e-6:
        return False

    if "tag" in phases and status not in ("tags_only", "completed"):
        return False
    if "embed" in phases and status not in ("embeddings_only", "completed"):
        return False
    return True


def _batched_tagger_embeddings(
    video_path: str,
    timestamps_ms: list[int],
    tagger: WDTaggerModel,
    batch_size: int,
    cancelled: Callable[[], bool],
    on_batch: Optional[Callable[[int], None]] = None,
) -> list[tuple[int, object]]:
    """Run wd-tagger over (deduped, sorted) timestamps with proper batching.
    ``on_batch(done)``: called after each batch (progress for slow CPUs)."""
    embeddings: list[tuple[int, object]] = []
    batch_ts: list[int] = []
    batch_imgs: list = []

    def flush():
        nonlocal batch_ts, batch_imgs
        if not batch_imgs:
            return
        batch_embs = tagger.embed_images(batch_imgs)
        embeddings.extend((ts, emb) for ts, emb in zip(batch_ts, batch_embs))
        batch_ts = []
        batch_imgs = []
        if on_batch is not None:
            on_batch(len(embeddings))

    for ts, img in frames.extract_frames_at(video_path, timestamps_ms):
        if cancelled():
            return embeddings
        batch_ts.append(ts)
        batch_imgs.append(img)
        if len(batch_imgs) >= batch_size:
            flush()
    flush()
    return embeddings


def _offload_tagger_between_phases(tagger: WDTaggerModel) -> bool:
    """Free the tagger before SigLIP loads? Needed on discrete GPUs (8 GB
    VRAM). On Apple Silicon the two share unified memory (~1 GB together in
    fp16): keeping both resident for the whole queue is what lets the decode
    stream frames to them (``streaming``) and avoids a reload per video;
    ``index_queue`` frees the tagger at the end."""
    return getattr(tagger, "device_backend", None) != "mps"


def _prewarm(models: Sequence[object]) -> None:
    """Load models in the background while the video is being decoded.

    Apple Silicon only (same condition as tagger residency): SigLIP 2
    (~5 s) and the tagger (~1.5 s) load while the CPU decodes, so the first
    streamed scenes do not wait for them. On discrete GPUs the tagger and
    SigLIP must not be resident together, so nothing is preloaded there.
    Failures are swallowed — the phase that needs the model calls
    ``_ensure_loaded`` again and reports.
    """
    def run():
        for m in models:
            try:
                m._ensure_loaded()
            except Exception as exc:  # pragma: no cover - surfaced by the real call
                log.debug("prewarm of %s failed: %s", type(m).__name__, exc)

    threading.Thread(target=run, daemon=True, name="model-prewarm").start()


def _new_motion_accumulator(processes: int = 0, device: Optional[str] = None):
    try:
        from ..searchplus.analysis import FrameStatsAccumulator
        if device is not None and os.environ.get("AMV_MOTION_CPU") != "1":
            from ..searchplus.motion_gpu import accumulator_for
            acc = accumulator_for(device)
        else:
            acc = FrameStatsAccumulator()
        return acc.start_async(processes=processes) if acc.available else None
    except Exception as exc:
        log.warning("search+ motion analysis unavailable: %s", exc)
        return None


def _preload_models(siglip, tagger, phases, cancelled: Callable[[], bool]) -> Optional[threading.Thread]:
    """Load the tag/embed models while NVDEC decodes (first video of a queue:
    ~4-8 s of reading weights that no longer sits on the critical path).
    Only when both fit side by side — on small cards the pipeline keeps
    loading them one after the other."""
    dev = getattr(siglip, "device", None)
    backend = getattr(dev, "backend", None)
    if backend not in ("cuda", "mps") or (getattr(dev, "vram_gb", None) or 0) < 10:
        return None

    done = threading.Event()

    def run():
        try:
            tag_thread = None
            if "tag" in phases:
                if backend == "mps":
                    # Apple Silicon: both load at once (and the tagger's
                    # Core ML worker starts) while the first chunks decode.
                    tag_thread = threading.Thread(target=tagger._ensure_loaded, daemon=True)
                    tag_thread.start()
                else:
                    tagger._ensure_loaded()
            if "embed" in phases:
                siglip._ensure_loaded()
            if tag_thread is not None:
                tag_thread.join()
            if "embed" in phases:
                # Keep SigLIP's idle-offload watcher away until the decode
                # is over (a long episode outlasts the idle timeout).
                deadline = time.monotonic() + 3600  # never pin it forever (error paths)
                while not done.wait(2.0) and not cancelled() and time.monotonic() < deadline:
                    siglip._touch()
                siglip._touch()
        except Exception as exc:  # the phases retry (and report) on their own
            log.debug("model preload failed: %s", exc)

    t = threading.Thread(target=run, daemon=True, name="index-preload")
    t.start()
    t.release = done.set  # type: ignore[attr-defined]
    return t


def _single_pass_decode(video_path: str, scene_threshold: float, cancelled: Callable[[], bool]):
    """(hard cuts, median frames, motion accumulator) from one NVDEC decode.
    The motion tracker's pair estimates run in blocks on the GPU
    (``AMV_MOTION_CPU=1``: two CPU worker processes, the previous path)."""
    acc = _new_motion_accumulator(processes=2 if (os.cpu_count() or 1) >= 4 else 0, device="cuda")
    try:
        sink = acc.sink if acc is not None else None
        res = None
        if nvc_pass.available():
            # NVDEC straight into CUDA tensors (no ffmpeg filter graph):
            # ~5x less CPU per frame; None = not handled (codec, bit depth)
            try:
                res = nvc_pass.detect_cuts_and_medians(video_path, scene_threshold, frame_sink=sink,
                                                       cancelled=cancelled)
            except onepass.Cancelled:
                raise
            except Exception as exc:
                log.warning("NVDEC tensor pass failed for %s (%s); using ffmpeg", video_path, exc)
                res = None
        hard_cuts, medians = res if res is not None else onepass.detect_cuts_and_medians(
            video_path, scene_threshold, frame_sink=sink, cancelled=cancelled)
    finally:
        if acc is not None:
            acc.finish()
    return hard_cuts, medians, acc


class _DecodeScheduler:
    """NVIDIA queues: keeps ``depth`` single-pass decodes in flight, the next
    one starting as soon as one finishes. One ffmpeg pipeline leaves NVDEC
    partly idle (decode -> CUDA filters -> download -> pipe run in one
    thread: ~50 s per 23-min H.264 episode for a ~42 s decode floor); two of
    them keep it busy. The next file the index loop needs always decodes;
    the other slot takes the cheapest of the next few, so a short or HEVC
    file decodes beside a long H.264 one instead of last and alone."""

    LOOKAHEAD = 3

    def __init__(self, files: list, scene_threshold: float, cancelled: Callable[[], bool],
                 should_decode: Callable[[str], bool], depth: int = 2):
        self._order = list(files)
        self._pending = list(files)
        self._threshold = scene_threshold
        self._cancelled = cancelled
        self._should = should_decode
        self._jobs: dict = {}
        self._taken: set = set()
        self._costs: dict = {}
        self._lock = threading.Lock()
        with self._lock:
            self._pending = [p for p in self._pending if self._should(p)]
        for _ in range(depth):
            self._launch_next()

    def _cost(self, path: str) -> float:
        """Rough NVDEC time: duration, HEVC/AV1 counted ~2x cheaper per frame
        than H.264 (their pipeline, not the decoder, is the limit)."""
        if path not in self._costs:
            try:
                import av
                with av.open(path) as c:
                    name = c.streams.video[0].codec_context.name or ""
                    dur = float(c.duration or 0) / 1e6
                self._costs[path] = dur * (0.55 if name in ("hevc", "av1") else 1.0)
            except Exception:
                self._costs[path] = float("inf")
        return self._costs[path]

    def _launch_next(self) -> None:
        with self._lock:
            if not self._pending or self._cancelled():
                return
            needed = next((p for p in self._order if p not in self._taken
                           and (p in self._pending or p in self._jobs)), None)
            if needed in self._pending:
                path = needed
            else:
                path = min(self._pending[:self.LOOKAHEAD], key=self._cost)
            self._pending.remove(path)
            self._jobs[path] = _Prefetch(path, self._threshold, self._cancelled, on_done=self._launch_next)

    def get(self, path: str):
        with self._lock:
            self._taken.add(path)
            return self._jobs.pop(path, None)


_ACTIVE = [0]
_ACTIVE_LOCK = threading.Lock()


def decodes_running() -> int:
    """Single passes decoding right now (other work can size itself on it)."""
    with _ACTIVE_LOCK:
        return _ACTIVE[0]


class _Prefetch:
    """Single-pass decode of the queue's next video, in a thread."""

    def __init__(self, video_path: str, scene_threshold: float, cancelled: Callable[[], bool],
                 on_done: Optional[Callable[[], None]] = None):
        self.video_path = video_path
        self.scene_threshold = scene_threshold
        self._result = None
        self._on_done = on_done
        self._thread = threading.Thread(target=self._run, args=(cancelled,), daemon=True,
                                        name="index-prefetch")
        self._thread.start()

    def _run(self, cancelled):
        with _ACTIVE_LOCK:
            _ACTIVE[0] += 1
        try:
            self._result = _single_pass_decode(self.video_path, self.scene_threshold, cancelled)
        except onepass.Cancelled:
            pass
        except Exception as exc:  # the regular path retries and reports
            log.warning("prefetch decode failed for %s: %s", self.video_path, exc)
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE[0] -= 1
            if self._on_done is not None:
                self._on_done()

    def matches(self, video_path: str, scene_threshold: float) -> bool:
        return self.video_path == video_path and self.scene_threshold == scene_threshold

    def take(self):
        self._thread.join()
        return self._result


_DEFAULT_BATCH = 8


def _effective_batch(siglip, batch_size: int) -> int:
    """NVIDIA: a batch size left at the default follows the Settings
    recommendation for the card (24 GB -> 64; the default models peak at
    ~1.1 GB at 64). Fewer, bigger batches: less Python per image in the tag
    and embed passes, which share the process with the decode."""
    dev = getattr(siglip, "device", None)
    backend = getattr(dev, "backend", None)
    if batch_size != _DEFAULT_BATCH or backend not in ("cuda", "mps"):
        return batch_size
    from .. import autotune
    tuned = autotune.value("batch_size", None)   # measured within the tier's limit
    if backend == "mps":
        return int(tuned) if tuned else batch_size
    vram = getattr(dev, "vram_gb", None) or 0
    cap = 64 if vram >= 20 else 32 if vram >= 11 else 16 if vram >= 7 else batch_size
    return min(int(tuned), cap) if tuned else cap


def _cache_key_for(siglip, tagger, phases: Sequence[str]) -> str:
    siglip_patches = getattr(siglip, "max_num_patches", 256)
    return (
        f"{siglip.checkpoint}|patches:{siglip_patches}|"
        f"tagger:{tagger.repo_id}|phases:{','.join(sorted(phases))}"
    )


def _store_scene_metrics(db_path: str, acc, work: list, scene_db_ids: list[int], gpu: bool = False) -> None:
    """Aggregate motion/intensity/palette per scene. Never fails indexing —
    scenes left without metrics are picked up by the background analyzer."""
    try:
        import numpy as np
        from ..searchplus.analysis import aggregate_scene
        from ..searchplus.store import upsert_metrics
        rgbs = [np.asarray(img) for (_i, _s, _e, _p, img) in work]
        pals = None
        if gpu:
            # CUDA: the palettes (the costly part) in one GPU batch; None when
            # the GPU batch failed → the CPU path below
            from ..searchplus.palette_gpu import palettes
            pals = palettes(rgbs)
        if pals is not None:
            rows = [
                (sid, s_ms, e_ms, aggregate_scene(acc, s_ms, e_ms, rgb, pal=pals[k]))
                for k, ((_idx, s_ms, e_ms, _parent, _img), sid, rgb) in enumerate(zip(work, scene_db_ids, rgbs))
            ]
        else:
            def one(item):
                (_idx, s_ms, e_ms, _parent, _img), sid, rgb = item
                return sid, s_ms, e_ms, aggregate_scene(acc, s_ms, e_ms, rgb)

            # Scenes are independent (palette, motion aggregation): a few
            # threads (OpenCV / numpy release the GIL), same rows in the same
            # order.
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4, thread_name_prefix="scene-metrics") as ex:
                rows = list(ex.map(one, zip(work, scene_db_ids, rgbs)))
        with schema.get_conn(db_path) as conn:
            upsert_metrics(conn, rows)
    except Exception as exc:
        log.warning("search+ metrics failed for %d scenes: %s", len(scene_db_ids), exc)


_PROGRESS_KEY = "index_progress:{}"
# A run left one of these behind (app closed, crash, Stop, failure): the
# episode is not indexed and its next run picks up where this one stopped.
UNFINISHED_STATUSES = ("indexing", "cancelled", "failed")


def _finish_video(db_path: str, video_id: int, status: str) -> None:
    with schema.get_conn(db_path) as conn:
        queries.set_video_status(conn, video_id, status)
        conn.execute("DELETE FROM settings WHERE key = ?", (_PROGRESS_KEY.format(video_id),))


def mark_failed(db_path: str, video_path: str) -> None:
    """An indexing error: never leave the episode looking indexed."""
    try:
        with schema.get_conn(db_path) as conn:
            conn.execute("UPDATE videos SET status = 'failed' WHERE filepath = ? AND status != 'completed'",
                         (video_path,))
    except Exception as exc:
        log.warning("could not mark %s as failed: %s", video_path, exc)


def _resume_point(db_path: str, video_path: str, cache_key: str) -> Optional[dict]:
    """Where an unfinished run of this video stopped, if it can be resumed:
    same file (mtime), same models and phases (``cache_key``), scene rows
    written. Returns ``{video_id, scenes: [(id, idx, start, end, has_emb)],
    tagged_through}`` — scenes with ``id <= tagged_through`` have their tags
    committed (atomically with that marker), embeddings are per row."""
    if not os.path.exists(video_path):
        return None
    with schema.get_conn(db_path) as conn:
        row = conn.execute("SELECT id, modified_at, model_version, status FROM videos WHERE filepath = ?",
                           (video_path,)).fetchone()
        if not row:
            return None
        video_id, modified_at, stored_model, status = row
        if status not in UNFINISHED_STATUSES or stored_model != cache_key:
            return None
        if abs(float(modified_at or 0) - os.path.getmtime(video_path)) > 1e-6:
            return None
        scenes = conn.execute(
            "SELECT id, scene_index, start_ms, end_ms, embedding IS NOT NULL FROM scenes "
            "WHERE video_id = ? ORDER BY id", (video_id,)).fetchall()
        mark = conn.execute("SELECT value FROM settings WHERE key = ?",
                            (_PROGRESS_KEY.format(video_id),)).fetchone()
    if not scenes:
        return None
    try:
        tagged_through = int(mark[0]) if mark else 0
    except ValueError:
        tagged_through = 0
    return {"video_id": video_id, "scenes": scenes, "tagged_through": tagged_through}


def _tag_and_embed(db_path: str, video_path: str, video_id: int, tag_items: list, embed_items: list, *,
                   phases: Sequence[str], tagger, siglip, batch_size: int, tag_threshold: float,
                   cancelled: Callable[[], bool], report: Callable[..., None],
                   pre_tags: Optional[dict] = None, pre_embs: Optional[dict] = None,
                   total: Optional[int] = None) -> bool:
    """Tag then embed ``[(scene_id, median_ms, image)]``. False when cancelled
    (status ``cancelled``, resumable). A failing batch raises: the episode
    must not end up "indexed" with its tags or embeddings silently missing
    (e.g. onnxruntime absent)."""
    pre_tags = pre_tags or {}
    pre_embs = pre_embs or {}
    total = total or max(len(tag_items), len(embed_items), 1)
    done_before_tag = total - len(tag_items)
    done_before_emb = total - len(embed_items)
    progress_key = _PROGRESS_KEY.format(video_id)

    # ── Phase A — Tag pass ───────────────────────────────────────────────────
    if "tag" in phases and tag_items:
        report(30, f"Tagging {total} scenes")
        try:
            # One connection for the whole pass (a connection + commit per
            # batch cost ~3 s per episode on a hard disk); periodic commits keep
            # the write lock short for the Search+ metrics writer.
            with schema.get_conn(db_path) as conn:
                committed = time.monotonic()
                for batch_start in range(0, len(tag_items), batch_size):
                    if cancelled():
                        queries.set_video_status(conn, video_id, "cancelled")
                        return False
                    batch = tag_items[batch_start:batch_start + batch_size]
                    batch_end = done_before_tag + batch_start + len(batch)
                    try:
                        if all(t in pre_tags for _sid, t, _img in batch):
                            batch_tags = [pre_tags[t] for _sid, t, _img in batch]
                        else:
                            batch_tags = tagger.tag_images([img for _sid, _t, img in batch])
                    except Exception as e:
                        log.error("tagging batch failed for %s: %s", video_path, e)
                        conn.commit()   # keep the batches done: the next run resumes after them
                        raise RuntimeError(f"tagging failed: {e}") from e
                    for (sid, _t, _img), tags in zip(batch, batch_tags):
                        keep = [(t, c, cf) for t, c, cf in (tags or []) if cf >= tag_threshold]
                        if keep:
                            queries.insert_tags(conn, sid, keep)
                    # same transaction as the tags: a resumed run re-tags none twice
                    conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                                 (progress_key, str(batch[-1][0])))
                    if time.monotonic() - committed > 0.5:
                        conn.commit()
                        committed = time.monotonic()
                    pct = 30 + int((batch_end / total) * (30 if "embed" in phases else 65))
                    report(pct, f"Tagged {batch_end}/{total}")
        finally:
            # Free the tagger before SigLIP loads on the same GPU.
            if "embed" in phases and _offload_tagger_between_phases(tagger):
                tagger.offload()

    # ── Phase B — Embed pass ─────────────────────────────────────────────────
    if "embed" in phases and embed_items:
        report(60 if "tag" in phases else 30, f"Embedding {total} scenes")
        with schema.get_conn(db_path) as conn:
            committed = time.monotonic()
            for batch_start in range(0, len(embed_items), batch_size):
                if cancelled():
                    queries.set_video_status(conn, video_id, "cancelled")
                    return False
                batch = embed_items[batch_start:batch_start + batch_size]
                batch_end = done_before_emb + batch_start + len(batch)
                try:
                    if all(t in pre_embs for _sid, t, _img in batch):
                        batch_embs = [pre_embs[t] for _sid, t, _img in batch]
                    else:
                        batch_embs = siglip.embed_images([img for _sid, _t, img in batch])
                except Exception as e:
                    log.error("embedding batch failed for %s: %s", video_path, e)
                    conn.commit()
                    raise
                for (sid, _t, _img), emb in zip(batch, batch_embs):
                    queries.update_scene_embedding(conn, sid, emb)
                if time.monotonic() - committed > 0.5:
                    conn.commit()
                    committed = time.monotonic()
                pct = (60 if "tag" in phases else 30) + int((batch_end / total) * (35 if "tag" in phases else 65))
                report(pct, f"Embedded {batch_end}/{total}")
    return True


def _resume_video(db_path: str, video_path: str, point: dict, *, phases: Sequence[str], tagger, siglip,
                  batch_size: int, tag_threshold: float, cancelled: Callable[[], bool],
                  report: Callable[..., None], final_status: str) -> None:
    """Finish an interrupted run: no cut detection, no new scene rows; only
    the median frames of the scenes still missing tags or an embedding are
    decoded."""
    video_id, scenes = point["video_id"], point["scenes"]
    through = point["tagged_through"]
    need_tag = {sid for sid, *_ in scenes if sid > through} if "tag" in phases else set()
    need_emb = {sid for sid, _i, _s, _e, has in scenes if not has} if "embed" in phases else set()
    log.info("resuming %s: %d/%d scenes left to tag, %d to embed", video_path, len(need_tag),
             len(scenes), len(need_emb))
    report(3, f"Resuming: {len(scenes) - len(need_tag)} scenes tagged, "
              f"{len(scenes) - len(need_emb)} embedded")
    todo = [(sid, _median_ms(s, e)) for sid, _i, s, e, _h in scenes if sid in need_tag or sid in need_emb]
    images: dict = {}
    if todo:
        report(22, f"Extracting {len(todo)} median frames")
        images = dict(frames.extract_frames_at(video_path, sorted({t for _sid, t in todo})))
        if cancelled():
            return
    items = [(sid, t, images[t]) for sid, t in todo if t in images]
    if len(items) < len(todo):
        raise RuntimeError(f"could not decode {len(todo) - len(items)} median frames")
    if not _tag_and_embed(db_path, video_path, video_id,
                          [it for it in items if it[0] in need_tag],
                          [it for it in items if it[0] in need_emb],
                          phases=phases, tagger=tagger, siglip=siglip, batch_size=batch_size,
                          tag_threshold=tag_threshold, cancelled=cancelled, report=report,
                          total=len(scenes)):
        return
    _finish_video(db_path, video_id, final_status)
    report(100, "Done")


def index_video(
    db_path: str,
    video_path: str,
    siglip: SigLIP2Model,
    tagger: WDTaggerModel,
    *,
    phases: Sequence[str] = ALL_PHASES,
    mode: str = "fast",
    scene_threshold: float = 0.40,
    enable_sub_segmentation: bool = False,
    sub_seg_threshold: float = 0.30,
    tag_threshold: float = 0.50,
    batch_size: int = 8,
    auto_skip_indexed: bool = True,
    cancel_event: Optional[threading.Event] = None,
    progress_cb: Optional[Callable[[dict], None]] = None,
    group_name: Optional[str] = None,
    next_video: Optional[str] = None,
    prefetched: Optional["_Prefetch"] = None,
    proxy_sources: bool = False,
) -> Optional["_Prefetch"]:
    """Index one video. On the NVIDIA single-pass path, ``next_video`` (the
    queue's next file) starts decoding as soon as this one's decode is done,
    and the returned prefetch is handed back through ``prefetched``.
    ``proxy_sources``: hover proxies will follow — the software single pass
    records their 240p source on the same decode (``proxy_source``)."""
    phases = tuple(p for p in phases if p in ALL_PHASES) or ALL_PHASES
    batch_size = _effective_batch(siglip, max(1, batch_size))

    def report(percent: int, message: str = ""):
        if progress_cb:
            progress_cb({"type": "indexing", "video": _display_name(video_path),
                         "percent": percent, "message": message})

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    needs_tagger_for_subseg = enable_sub_segmentation and mode != "fast"
    final_status = _status_for_phases(phases)
    # The DB tracks which phases were last run for this video; we encode it in
    # the model_version so that bumping any model invalidates the cache cleanly.
    # max_num_patches is part of the cache key: a video re-encoded with a
    # different patch budget produces embeddings in a slightly different
    # space, so we must invalidate to keep search consistent.
    cache_key = _cache_key_for(siglip, tagger, phases)

    schema.init_db(db_path)
    if auto_skip_indexed and _is_video_current_for_phases(db_path, video_path, cache_key, phases):
        report(100, "Already indexed")
        return
    point = _resume_point(db_path, video_path, cache_key) if auto_skip_indexed else None
    if point is not None:
        _resume_video(db_path, video_path, point, phases=phases, tagger=tagger, siglip=siglip,
                      batch_size=batch_size, tag_threshold=tag_threshold, cancelled=cancelled,
                      report=report, final_status=final_status)
        return None

    if not _offload_tagger_between_phases(tagger):
        _prewarm([m for m, phase in ((tagger, "tag"), (siglip, "embed"))
                  if phase in phases or (m is tagger and needs_tagger_for_subseg)])

    report(1, "Probing video")
    info = cuts.probe_video(video_path)

    report(3, "Detecting cuts (pass 1)")
    # Search+ motion analysis rides on the cut detector's low-res decode.
    # NVIDIA: one NVDEC decode also captures every scene's median frame, so
    # the extraction step below doesn't decode the video a second time. The
    # decode then outruns the motion tracker, whose pair estimates go to two
    # worker processes (GIL-bound: threads don't help).
    nvdec_pass = onepass.available()
    preload = (_preload_models(siglip, tagger, phases, cancelled)
               if nvdec_pass or getattr(tagger, "device_backend", None) == "mps" else None)
    motion_acc = None
    early_medians: dict = {}
    hard_cuts = None
    if nvdec_pass:
        res = None
        if prefetched is not None and prefetched.matches(video_path, scene_threshold):
            res = prefetched.take()
        if res is None:
            try:
                res = _single_pass_decode(video_path, scene_threshold, cancelled)
            except onepass.Cancelled:
                return None
            except Exception as exc:
                log.warning("single-pass decode failed for %s (%s); using two passes", video_path, exc)
        if res is not None:
            hard_cuts, early_medians, motion_acc = res
    # Software decode (Apple Silicon): the median frames ride on the same
    # decode as the cut detector instead of a second full decode.
    decoded_medians: Optional[dict] = None
    # ... and with both models resident (Apple Silicon), each median is
    # tagged/embedded on the GPU as soon as its scene closes.
    # Sub-segmentation rides along too: every sample of a closed scene goes
    # to the GPU worker, which splits it and tags/embeds the pieces.
    stream: Optional[streaming.StreamingInference] = None
    resident = not _offload_tagger_between_phases(tagger)
    subseg_step_ms = SUB_SAMPLE_STEP_MS if mode == "accurate" else SUB_SAMPLE_STEP_MS * 2
    streamed_segments: Optional[dict] = None
    t_decode0 = time.monotonic()
    if hard_cuts is None:
        motion_acc = _new_motion_accumulator()
        sink = motion_acc.sink if motion_acc is not None else None
        try:
            if (resident or not needs_tagger_for_subseg) and single_pass.supported(video_path):
                if resident:
                    stream = streaming.StreamingInference(
                        tagger, siglip, phases, batch_size, cancel_event,
                        subseg={"min_ms": LONG_SCENE_THRESHOLD_MS, "step_ms": subseg_step_ms,
                                "threshold": sub_seg_threshold} if needs_tagger_for_subseg else None)
                if not needs_tagger_for_subseg and chunked.supported(video_path):
                    # Apple Silicon: the same pass on parallel VideoToolbox
                    # sessions (see ``chunked``); identical results.
                    try:
                        hard_cuts, decoded_medians = chunked.detect_cuts_with_medians(
                            video_path, threshold=scene_threshold, motion=motion_acc,
                            on_median=stream.submit if stream is not None else None,
                            proxy_source=proxy_sources and proxy_source.wanted(), cancelled=cancelled)
                    except chunked.Cancelled:
                        return None
                    except Exception as exc:
                        log.warning("chunked decode failed for %s (%s); one sequential pass", video_path, exc)
                        hard_cuts = None
                if hard_cuts is None:
                    try:
                        hard_cuts, decoded_medians = single_pass.detect_cuts_with_medians(
                            video_path, threshold=scene_threshold, frame_sink=sink,
                            on_median=stream.submit if stream is not None and not needs_tagger_for_subseg else None,
                            on_scene=stream.submit_scene if stream is not None and needs_tagger_for_subseg else None,
                            **({"proxy_source": True} if proxy_sources and proxy_source.wanted() else {}))
                        if stream is not None and needs_tagger_for_subseg:
                            streamed_segments = stream.segments
                    except Exception as exc:
                        log.warning("single-pass decode failed for %s (%s); using two passes", video_path, exc)
                        hard_cuts = cuts.detect_cuts(video_path, threshold=scene_threshold, frame_sink=sink)
            else:
                hard_cuts = cuts.detect_cuts(video_path, threshold=scene_threshold, frame_sink=sink)
        finally:
            t_dec = time.monotonic()
            if motion_acc is not None:
                motion_acc.finish()
            t_mot = time.monotonic()
            if stream is not None:
                stream.finish()
            log.info("%s: decode %.1f s, then motion %.1f s, inference tail %.1f s", _display_name(video_path),
                     t_dec - t_decode0, t_mot - t_dec, time.monotonic() - t_mot)
            if stream is not None:
                if stream.error is not None:
                    # Later scenes were left unsplit: redo sub-segmentation the
                    # regular way (tags/embeddings already made are kept).
                    streamed_segments = None
    pre_tags = stream.tags if stream is not None else {}
    pre_embs = stream.embs if stream is not None else {}
    if streamed_segments is not None and decoded_medians is not None:
        decoded_medians.update(stream.medians)
    if decoded_medians:
        early_medians = decoded_medians
    next_prefetch = None
    if nvdec_pass and next_video and not cancelled():
        # NVDEC is free again: decode the next file while this one runs its
        # tag / embed / database phases on the SMs and the CPU.
        if not (auto_skip_indexed and _is_video_current_for_phases(db_path, next_video, cache_key, phases)):
            next_prefetch = _Prefetch(next_video, scene_threshold, cancelled)
    if not hard_cuts:
        hard_cuts = [(0, max(info["duration_ms"], 1000))]
    log.info("video %s -> %d hard cuts", video_path, len(hard_cuts))
    if cancelled():
        return

    # ── Sub-segmentation (optional) ─────────────────────────────────────────
    # final_segments: list of (start_ms, end_ms, parent_seq_id_or_None)
    final_segments: list[tuple[int, int, int | None]] = []
    parents_seq: dict[int, tuple[int, int]] = {}

    if (needs_tagger_for_subseg and streamed_segments is not None
            and all(s_ms in streamed_segments for s_ms, _ in hard_cuts)):
        next_seq = 0
        for s_ms, e_ms in hard_cuts:
            segs = streamed_segments[s_ms]
            if len(segs) <= 1:
                final_segments.append((s_ms, e_ms, None))
                continue
            next_seq += 1
            parents_seq[next_seq] = (s_ms, e_ms)
            final_segments.append((s_ms, e_ms, -next_seq))
            final_segments.extend((a, b, next_seq) for a, b, _parent in segs[1:])
    elif needs_tagger_for_subseg:
        step_ms = subseg_step_ms
        samples_by_cut: dict[int, list[int]] = {}
        all_sample_ts: set[int] = set()
        for i, (s_ms, e_ms) in enumerate(hard_cuts):
            if e_ms - s_ms <= LONG_SCENE_THRESHOLD_MS:
                continue
            samples = list(range(s_ms, e_ms, step_ms))
            samples_by_cut[i] = samples
            all_sample_ts.update(samples)

        if all_sample_ts:
            n_samples = len(all_sample_ts)
            report(5, f"Sub-segmenting (decoding {n_samples} samples in one pass)")
            # On CPU this pass alone takes minutes: keep the bar moving (5 → 15).
            global_embs_list = _batched_tagger_embeddings(
                video_path, sorted(all_sample_ts), tagger, batch_size, cancelled,
                on_batch=lambda done: report(5 + int(done / n_samples * 10),
                                             f"Sub-segmenting (decoding {n_samples} samples in one pass)"),
            )
            if cancelled():
                return
            global_embs: dict[int, object] = dict(global_embs_list)
        else:
            global_embs = {}

        next_seq = 0
        for i, (s_ms, e_ms) in enumerate(hard_cuts):
            if cancelled():
                return
            if i not in samples_by_cut:
                final_segments.append((s_ms, e_ms, None))
                continue
            embeddings = [(ts, global_embs[ts]) for ts in samples_by_cut[i] if ts in global_embs]
            subs = sub_segment.sub_segment(s_ms, e_ms, embeddings, drift_threshold=sub_seg_threshold)
            if len(subs) <= 1:
                final_segments.append((s_ms, e_ms, None))
            else:
                next_seq += 1
                this_seq = next_seq
                parents_seq[this_seq] = (s_ms, e_ms)
                final_segments.append((s_ms, e_ms, -this_seq))  # negative = "I am parent #this_seq"
                for sub_s, sub_e in subs:
                    final_segments.append((sub_s, sub_e, this_seq))  # positive = child of parent #this_seq
            pct = 15 + int(((i + 1) / len(hard_cuts)) * 5)
            report(pct, f"Sub-segmenting {i + 1}/{len(hard_cuts)}")

        # If embed is requested, free the tagger before SigLIP loads. If tag
        # phase is also requested we'll reload it just below — cheap (<2 s).
        if "embed" in phases and _offload_tagger_between_phases(tagger):
            tagger.offload()
    else:
        final_segments = [(s, e, None) for s, e in hard_cuts]

    if cancelled():
        return
    if preload is not None:
        preload.release()
        preload.join()

    # ── Extract median frames once, keep them in RAM for both phases ────────
    report(22, f"Extracting {len(final_segments)} median frames")
    median_ts = [_median_ms(s, e) for s, e, _ in final_segments]
    pulled_map = {ts: early_medians[ts] for ts in median_ts if ts in early_medians}
    missing = [ts for ts in median_ts if ts not in pulled_map]
    if missing:
        pulled_map.update(frames.extract_frames_at(video_path, missing))

    # Build a flat work list in scene order: (scene_idx, s_ms, e_ms, parent_marker, img)
    work: list[tuple[int, int, int, int | None, object]] = []
    for idx, (s_ms, e_ms, parent_marker) in enumerate(final_segments):
        img = pulled_map.get(median_ts[idx])
        if img is None:
            continue
        work.append((idx, s_ms, e_ms, parent_marker, img))

    if not work:
        log.warning("video %s: no median frames pulled, skipping", video_path)
        return next_prefetch

    # ── Insert scenes with thumbnails (embedding = NULL, no tags yet) ───────
    # Done in scene order so that parent rows exist before their children.
    report(25, f"Inserting {len(work)} scenes (thumbnails)")
    t_ins0 = time.monotonic()
    thumbs = [frames.thumbnail_bytes(img) for _, _, _, _, img in work]
    t_ins1 = time.monotonic()

    scene_db_ids: list[int] = []  # parallel to work
    parent_seq_to_db_id: dict[int, int] = {}
    with schema.get_conn(db_path) as conn:
        video_id = queries.upsert_video(
            conn, video_path, _display_name(video_path),
            info["duration_ms"], info["fps"], info["resolution"],
            model_version=cache_key, status="indexing",
            group_name=group_name,
        )
        queries.delete_scenes_for_video(conn, video_id)
        conn.execute("DELETE FROM settings WHERE key = ?", (_PROGRESS_KEY.format(video_id),))
        for (scene_idx, s_ms, e_ms, parent_marker, _img), thumb in zip(work, thumbs):
            if parent_marker is None:
                parent_db_id = None
            elif parent_marker < 0:
                parent_db_id = None  # I am a parent
            else:
                parent_db_id = parent_seq_to_db_id.get(parent_marker)
            sid = queries.insert_scene(
                conn, video_id, scene_idx, s_ms, e_ms,
                embedding=None, thumbnail=thumb, parent_scene_id=parent_db_id,
            )
            scene_db_ids.append(sid)
            if parent_marker is not None and parent_marker < 0:
                parent_seq_to_db_id[-parent_marker] = sid

    metrics_thread: Optional[threading.Thread] = None
    if motion_acc is not None:
        if nvdec_pass:
            # CPU-only (palette, aggregation): runs beside the GPU phases.
            metrics_thread = threading.Thread(
                target=_store_scene_metrics, args=(db_path, motion_acc, work, scene_db_ids, True),
                daemon=True, name="scene-metrics")
            metrics_thread.start()
        else:
            t_ins2 = time.monotonic()
            _store_scene_metrics(db_path, motion_acc, work, scene_db_ids)
            log.info("%s: thumbnails %.2f s, scene rows %.2f s, Search+ metrics %.2f s", _display_name(video_path),
                     t_ins1 - t_ins0, t_ins2 - t_ins1, time.monotonic() - t_ins2)

    if cancelled():
        with schema.get_conn(db_path) as conn:
            queries.set_video_status(conn, video_id, "cancelled")
        return

    tag_items = embed_items = [(sid, _median_ms(w[1], w[2]), w[4]) for w, sid in zip(work, scene_db_ids)]
    if not _tag_and_embed(db_path, video_path, video_id, tag_items, embed_items, phases=phases,
                          tagger=tagger, siglip=siglip, batch_size=batch_size, tag_threshold=tag_threshold,
                          cancelled=cancelled, report=report, pre_tags=pre_tags, pre_embs=pre_embs):
        return

    # ── Done ────────────────────────────────────────────────────────────────
    if metrics_thread is not None:
        metrics_thread.join()
    _finish_video(db_path, video_id, final_status)
    report(100, "Done")
    return next_prefetch


def index_queue(
    db_path: str,
    siglip: SigLIP2Model,
    tagger: WDTaggerModel,
    cancel_event: Optional[threading.Event] = None,
    progress_cb: Optional[Callable[[dict], None]] = None,
    **kwargs,
) -> str:
    from ..config import VIDEO_EXTENSIONS

    schema.init_db(db_path)
    with schema.get_conn(db_path) as conn:
        items = queries.get_queue(conn)

    # (filepath, group) pairs — the queue item's user-chosen import group is
    # stamped on every video found under it.
    all_files: list[tuple[str, str | None]] = []
    # queued items that give nothing to index: said, not silently dropped
    skipped: list[tuple[str, str]] = []
    for item in items:
        p = Path(item["path"])
        group = item.get("group_name") or None
        if not p.exists():
            skipped.append((str(p), "not found (moved, renamed or drive unplugged?)"))
            continue
        if p.is_file():
            if p.suffix.lower() in VIDEO_EXTENSIONS:
                all_files.append((str(p), group))
            continue
        before = len(all_files)
        pattern_iter = p.rglob("*") if item["recursive"] else p.iterdir()
        for sub in pattern_iter:
            if sub.is_file() and sub.suffix.lower() in VIDEO_EXTENSIONS:
                all_files.append((str(sub), group))
        if len(all_files) == before:
            skipped.append((str(p), "no video file in this folder"))

    # Episodes whose last run did not finish (app closed, Stop, failure) are
    # picked up again even when their folder already left the queue; their
    # run resumes where it stopped (``_resume_point``).
    seen = {f for f, _g in all_files}
    with schema.get_conn(db_path) as conn:
        unfinished = conn.execute(
            f"SELECT filepath, group_name FROM videos WHERE status IN ({','.join('?' * len(UNFINISHED_STATUSES))}) "
            "ORDER BY id", UNFINISHED_STATUSES).fetchall()
    for fp, group in unfinished:
        if fp in seen:
            continue
        if os.path.exists(fp):
            all_files.append((fp, group))
            seen.add(fp)
        else:
            skipped.append((fp, "not found: cannot finish indexing it (moved, renamed or drive unplugged?)"))
    if progress_cb:
        for path_, why in skipped:
            progress_cb({"type": "indexing", "video": Path(path_).name, "percent": 0,
                         "message": f"Failed: {why}", "failed": True, "path": path_})

    video_done_cb = kwargs.pop("video_done_cb", None)
    pending = None
    scheduler = None
    from .. import autotune
    depth = int(os.environ.get("AMV_PARALLEL_DECODES") or autotune.value("parallel_decodes", 2))
    if len(all_files) > 1 and depth > 1 and onepass.available():
        phases = tuple(p for p in kwargs.get("phases", ALL_PHASES) if p in ALL_PHASES) or ALL_PHASES
        key = _cache_key_for(siglip, tagger, phases)
        skip = kwargs.get("auto_skip_indexed", True)
        scheduler = _DecodeScheduler(
            [f for f, _g in all_files], kwargs.get("scene_threshold", 0.40),
            lambda: bool(cancel_event and cancel_event.is_set()),
            lambda path: not (skip and _is_video_current_for_phases(db_path, path, key, phases)),
            depth=depth)

    def file_progress(i: int):
        # Tags every event with the file's rank so the UI can say "file 1/2".
        if not progress_cb:
            return None
        return lambda evt: progress_cb({**evt, "file_index": i + 1, "file_count": len(all_files)})

    try:
        for i, (fp, group) in enumerate(all_files):
            if cancel_event and cancel_event.is_set():
                return "cancelled"
            nxt = all_files[i + 1][0] if i + 1 < len(all_files) else None
            if scheduler is not None:
                pending, nxt = scheduler.get(fp), None
            try:
                pending = index_video(db_path, fp, siglip, tagger, cancel_event=cancel_event,
                                      progress_cb=file_progress(i), group_name=group,
                                      next_video=nxt, prefetched=pending, **kwargs)
                if video_done_cb is not None and not (cancel_event and cancel_event.is_set()):
                    try:
                        video_done_cb(fp, i == len(all_files) - 1)
                    except Exception as exc:  # proxies are retried at the end of the queue
                        log.warning("post-index hook failed for %s: %s", fp, exc)
            except Exception as e:
                pending = None
                log.error("indexing failed for %s: %s", fp, e, exc_info=True)
                mark_failed(db_path, fp)
                if progress_cb:
                    progress_cb({"type": "indexing", "video": Path(fp).name, "percent": 0,
                                 "message": f"Failed: {e}", "failed": True, "path": fp,
                                 "file_index": i + 1, "file_count": len(all_files)})
    finally:
        # The tagger may have stayed resident across videos (Apple Silicon).
        tagger.offload()

    with schema.get_conn(db_path) as conn:
        queries.clear_queue(conn)

    if progress_cb:
        progress_cb({"type": "idle"})
    return "completed"
