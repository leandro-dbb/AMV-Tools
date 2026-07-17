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
from pathlib import Path
from typing import Callable, Optional, Sequence

from ..db import queries, schema
from ..models import SigLIP2Model, WDTaggerModel
from . import cuts, frames, sub_segment

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
) -> list[tuple[int, object]]:
    """Run wd-tagger over (deduped, sorted) timestamps with proper batching."""
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

    for ts, img in frames.extract_frames_at(video_path, timestamps_ms):
        if cancelled():
            return embeddings
        batch_ts.append(ts)
        batch_imgs.append(img)
        if len(batch_imgs) >= batch_size:
            flush()
    flush()
    return embeddings


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
) -> None:
    phases = tuple(p for p in phases if p in ALL_PHASES) or ALL_PHASES
    batch_size = max(1, batch_size)

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
    siglip_patches = getattr(siglip, "max_num_patches", 256)
    cache_key = (
        f"{siglip.checkpoint}|patches:{siglip_patches}|"
        f"tagger:{tagger.repo_id}|phases:{','.join(sorted(phases))}"
    )

    schema.init_db(db_path)
    if auto_skip_indexed and _is_video_current_for_phases(db_path, video_path, cache_key, phases):
        report(100, "Already indexed")
        return

    report(1, "Probing video")
    info = cuts.probe_video(video_path)

    report(3, "Detecting cuts (pass 1)")
    hard_cuts = cuts.detect_cuts(video_path, threshold=scene_threshold)
    if not hard_cuts:
        hard_cuts = [(0, max(info["duration_ms"], 1000))]
    log.info("video %s -> %d hard cuts", video_path, len(hard_cuts))
    if cancelled():
        return

    # ── Sub-segmentation (optional) ─────────────────────────────────────────
    # final_segments: list of (start_ms, end_ms, parent_seq_id_or_None)
    final_segments: list[tuple[int, int, int | None]] = []
    parents_seq: dict[int, tuple[int, int]] = {}

    if needs_tagger_for_subseg:
        step_ms = SUB_SAMPLE_STEP_MS if mode == "accurate" else SUB_SAMPLE_STEP_MS * 2
        samples_by_cut: dict[int, list[int]] = {}
        all_sample_ts: set[int] = set()
        for i, (s_ms, e_ms) in enumerate(hard_cuts):
            if e_ms - s_ms <= LONG_SCENE_THRESHOLD_MS:
                continue
            samples = list(range(s_ms, e_ms, step_ms))
            samples_by_cut[i] = samples
            all_sample_ts.update(samples)

        if all_sample_ts:
            report(5, f"Sub-segmenting (decoding {len(all_sample_ts)} samples in one pass)")
            global_embs_list = _batched_tagger_embeddings(
                video_path, sorted(all_sample_ts), tagger, batch_size, cancelled
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
            pct = 5 + int(((i + 1) / len(hard_cuts)) * 15)
            report(pct, f"Sub-segmenting {i + 1}/{len(hard_cuts)}")

        # If embed is requested, free the tagger before SigLIP loads. If tag
        # phase is also requested we'll reload it just below — cheap (<2 s).
        if "embed" in phases:
            tagger.offload()
    else:
        final_segments = [(s, e, None) for s, e in hard_cuts]

    if cancelled():
        return

    # ── Extract median frames once, keep them in RAM for both phases ────────
    report(22, f"Extracting {len(final_segments)} median frames")
    median_ts = [_median_ms(s, e) for s, e, _ in final_segments]
    pulled = list(frames.extract_frames_at(video_path, median_ts))
    pulled_map = {ts: img for ts, img in pulled}

    # Build a flat work list in scene order: (scene_idx, s_ms, e_ms, parent_marker, img)
    work: list[tuple[int, int, int, int | None, object]] = []
    for idx, (s_ms, e_ms, parent_marker) in enumerate(final_segments):
        img = pulled_map.get(median_ts[idx])
        if img is None:
            continue
        work.append((idx, s_ms, e_ms, parent_marker, img))

    if not work:
        log.warning("video %s: no median frames pulled, skipping", video_path)
        return

    # ── Insert scenes with thumbnails (embedding = NULL, no tags yet) ───────
    # Done in scene order so that parent rows exist before their children.
    report(25, f"Inserting {len(work)} scenes (thumbnails)")
    thumbs = [frames.thumbnail_bytes(img) for _, _, _, _, img in work]

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

    if cancelled():
        with schema.get_conn(db_path) as conn:
            queries.set_video_status(conn, video_id, "cancelled")
        return

    # ── Phase A — Tag pass ───────────────────────────────────────────────────
    if "tag" in phases:
        report(30, f"Tagging {len(work)} scenes")
        try:
            for batch_start in range(0, len(work), batch_size):
                if cancelled():
                    with schema.get_conn(db_path) as conn:
                        queries.set_video_status(conn, video_id, "cancelled")
                    return
                batch_end = min(batch_start + batch_size, len(work))
                batch_imgs = [w[4] for w in work[batch_start:batch_end]]
                batch_sids = scene_db_ids[batch_start:batch_end]
                try:
                    batch_tags = tagger.tag_images(batch_imgs)
                except Exception as e:
                    log.warning("tagging batch failed for %s: %s", video_path, e)
                    batch_tags = [[] for _ in batch_imgs]
                with schema.get_conn(db_path) as conn:
                    for sid, tags in zip(batch_sids, batch_tags):
                        if not tags:
                            continue
                        keep = [(t, c, cf) for t, c, cf in tags if cf >= tag_threshold]
                        if keep:
                            queries.insert_tags(conn, sid, keep)
                pct = 30 + int((batch_end / len(work)) * (30 if "embed" in phases else 65))
                report(pct, f"Tagged {batch_end}/{len(work)}")
        finally:
            # Always free the tagger before SigLIP loads on the same GPU.
            if "embed" in phases:
                tagger.offload()

    # ── Phase B — Embed pass ─────────────────────────────────────────────────
    if "embed" in phases:
        report(60 if "tag" in phases else 30, f"Embedding {len(work)} scenes")
        for batch_start in range(0, len(work), batch_size):
            if cancelled():
                with schema.get_conn(db_path) as conn:
                    queries.set_video_status(conn, video_id, "cancelled")
                return
            batch_end = min(batch_start + batch_size, len(work))
            batch_imgs = [w[4] for w in work[batch_start:batch_end]]
            batch_sids = scene_db_ids[batch_start:batch_end]
            try:
                batch_embs = siglip.embed_images(batch_imgs)
            except Exception as e:
                log.error("embedding batch failed for %s: %s", video_path, e)
                raise
            with schema.get_conn(db_path) as conn:
                for sid, emb in zip(batch_sids, batch_embs):
                    queries.update_scene_embedding(conn, sid, emb)
            pct = (60 if "tag" in phases else 30) + int((batch_end / len(work)) * (35 if "tag" in phases else 65))
            report(pct, f"Embedded {batch_end}/{len(work)}")

    # ── Done ────────────────────────────────────────────────────────────────
    with schema.get_conn(db_path) as conn:
        queries.set_video_status(conn, video_id, final_status)
    report(100, "Done")


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
    for item in items:
        p = Path(item["path"])
        group = item.get("group_name") or None
        if not p.exists():
            continue
        if p.is_file():
            if p.suffix.lower() in VIDEO_EXTENSIONS:
                all_files.append((str(p), group))
            continue
        pattern_iter = p.rglob("*") if item["recursive"] else p.iterdir()
        for sub in pattern_iter:
            if sub.is_file() and sub.suffix.lower() in VIDEO_EXTENSIONS:
                all_files.append((str(sub), group))

    for i, (fp, group) in enumerate(all_files):
        if cancel_event and cancel_event.is_set():
            return "cancelled"
        try:
            index_video(db_path, fp, siglip, tagger, cancel_event=cancel_event,
                        progress_cb=progress_cb, group_name=group, **kwargs)
        except Exception as e:
            log.error("indexing failed for %s: %s", fp, e, exc_info=True)
            if progress_cb:
                progress_cb({"type": "indexing", "video": Path(fp).name, "percent": 0,
                             "message": f"Failed: {e}"})

    with schema.get_conn(db_path) as conn:
        queries.clear_queue(conn)

    if progress_cb:
        progress_cb({"type": "idle"})
    return "completed"
