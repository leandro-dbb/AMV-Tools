"""Singleton state shared across API handlers (settings cache, models, jobs)."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from .config import DEFAULT_SETTINGS, _LEGACY_SIGLIP_REMAP
from .db import schema, queries
from .paths import default_db_path, settings_path

log = logging.getLogger(__name__)


def _deep_merge(base: dict, override: dict) -> dict:
    out = {**base}
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class AppState:
    def __init__(self):
        self._settings: dict[str, Any] = {}
        self._siglip = None     # SigLIP2Model
        self._tagger = None     # WDTaggerModel
        self._sam2 = None       # SAM2Model — Manual click engine
        self._birefnet = None   # BiRefNetModel — Auto engine (default for anime)
        self._matanyone = None  # MatAnyoneModel — temporal propagation after BiRefNet seed
        self._device = None
        self._sam2_sessions: dict[str, Any] = {}     # session_id → SegmentSession
        self._sam2_sessions_lock = threading.Lock()
        self._search_cache = None
        self._search_cache_lock = threading.Lock()
        self._models_lock = threading.Lock()
        self._index_thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        self._listeners: list[asyncio.Queue] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.load_settings()

    # ---------- settings ----------
    def load_settings(self) -> None:
        p = settings_path()
        if p.exists():
            try:
                self._settings = _deep_merge(DEFAULT_SETTINGS, json.loads(p.read_text()))
            except (json.JSONDecodeError, OSError):
                self._settings = json.loads(json.dumps(DEFAULT_SETTINGS))
        else:
            self._settings = json.loads(json.dumps(DEFAULT_SETTINGS))
            self.save_settings()

        # Auto-migrate stale SigLIP checkpoint names that the current
        # transformers refuses to load against Siglip2Model.
        current_siglip = self._settings.get("models", {}).get("siglip_variant")
        replacement = _LEGACY_SIGLIP_REMAP.get(current_siglip)
        if replacement:
            log.info("migrating SigLIP checkpoint %s -> %s", current_siglip, replacement)
            self._settings.setdefault("models", {})["siglip_variant"] = replacement
            self.save_settings()

        # Auto-disable TensorRT for SigLIP NaFlex: the dynamic input shapes
        # trigger a fresh torch_tensorrt compile per batch, which OOMs on
        # consumer GPUs (16 GB VRAM) once SigLIP + tagger + buffers are loaded.
        # Net effect was a slowdown, not a speedup. Users can re-enable manually
        # in Settings → Models if their setup supports it.
        #
        # When we hit this, we also reset the pipeline defaults to the new
        # speed-tuned profile (fast mode, no sub-segmentation, no proxies). The
        # `_speedup_applied` marker prevents re-overriding the user's later
        # choices on subsequent launches.
        models = self._settings.setdefault("models", {})
        speedup_marker = self._settings.get("_speedup_applied")
        siglip_is_naflex = "naflex" in (models.get("siglip_variant") or "")
        if not speedup_marker and siglip_is_naflex:
            changed = False
            if models.get("use_tensorrt"):
                log.info("disabling use_tensorrt for SigLIP NaFlex (incompatible with 16 GB VRAM)")
                models["use_tensorrt"] = False
                changed = True
            indexing = self._settings.setdefault("indexing", {})
            if indexing.get("mode") == "accurate":
                log.info("switching indexing.mode accurate -> fast for speed")
                indexing["mode"] = "fast"
                changed = True
            if indexing.get("sub_segmentation") is True:
                log.info("disabling sub_segmentation for speed")
                indexing["sub_segmentation"] = False
                changed = True
            if indexing.get("generate_proxies") is True:
                log.info("disabling generate_proxies for speed")
                indexing["generate_proxies"] = False
                changed = True
            self._settings["_speedup_applied"] = True
            if changed:
                self.save_settings()

        # v2 (deprecated): an earlier revision flipped generate_proxies back ON
        # for previously speed-disabled installs because NVENC made the proxy
        # pass cheap. The project default is now "proxies off" again — the
        # MiniEditor streams the source via FastAPI Range requests fine, and
        # forcing proxies surprised users who never asked for them. We keep
        # the marker so the v2 block never re-runs on existing settings, but
        # we no longer touch generate_proxies. Users opt in explicitly via
        # Settings → Indexing.
        if self._settings.get("_speedup_applied") and not self._settings.get("_nvenc_proxies_applied"):
            self._settings["_nvenc_proxies_applied"] = True
            self.save_settings()

        # v3: search behaviour tweaks. We keep `base-naflex` as the speed-first
        # default so first-time users get the ~35 s/episode experience. Users
        # who want legacy-quality proper-noun precision can switch to
        # `so400m-naflex` in Settings → Models — the dropdown explains the
        # trade-off. What we DO normalize here:
        #   - tag_boost reset to 0: with the fixed `embed_text` (lowercase +
        #     max_length=64), SigLIP's own scores already discriminate well,
        #     and the previous boost was over-amplifying single-tag matches.
        if not self._settings.get("_legacy_quality_migrated"):
            search = self._settings.setdefault("search", {})
            changed = False
            if search.get("tag_boost", 0) > 0:
                log.info(
                    "disabling tag_boost (default). Re-enable in Settings → Search "
                    "if you specifically want hybrid SigLIP+tag scoring."
                )
                search["tag_boost"] = 0.0
                changed = True
            self._settings["_legacy_quality_migrated"] = True
            if changed:
                self.save_settings()

        # v4: mask export now preserves BiRefNet/MatAnyone soft alpha. The
        # previous defaults were tuned for hard binary masks and visibly ate
        # anime hair tips, so migrate only untouched default-looking values.
        if not self._settings.get("_soft_alpha_roto_defaults_applied"):
            models = self._settings.setdefault("models", {})
            changed = False
            if models.get("mask_shrink_px", 2) == 2:
                models["mask_shrink_px"] = 0
                changed = True
            if models.get("mask_bg_suppress_enabled", True) is True:
                models["mask_bg_suppress_enabled"] = False
                changed = True
            self._settings["_soft_alpha_roto_defaults_applied"] = True
            if changed:
                self.save_settings()

        # v5: the first soft-alpha cleanup defaults were too aggressive on
        # anime and could remove real hair/body pixels. Preserve user-tuned
        # values, but reset the exact bad default set to the conservative
        # preserve-detail profile.
        if not self._settings.get("_soft_alpha_safe_defaults_applied"):
            models = self._settings.setdefault("models", {})
            changed = False
            if models.get("mask_soft_bg_suppress_enabled") is True:
                models["mask_soft_bg_suppress_enabled"] = False
                changed = True
            if int(models.get("mask_soft_shrink_px", 0) or 0) == 1:
                models["mask_soft_shrink_px"] = 0
                changed = True
            if abs(float(models.get("mask_soft_alpha_black", 0.0) or 0.0) - 0.08) < 1e-6:
                models["mask_soft_alpha_black"] = 0.0
                changed = True
            if abs(float(models.get("mask_soft_alpha_white", 1.0) or 1.0) - 0.85) < 1e-6:
                models["mask_soft_alpha_white"] = 1.0
                changed = True
            self._settings["_soft_alpha_safe_defaults_applied"] = True
            if changed:
                self.save_settings()

        if not self._settings["databases"]["active"]:
            default_db = str(default_db_path())
            schema.init_db(default_db)
            self._settings["databases"]["active"] = [default_db]
            self._settings["databases"]["primary"] = default_db
            self.save_settings()
        else:
            for db in self._settings["databases"]["active"]:
                schema.init_db(db)

    def save_settings(self) -> None:
        settings_path().write_text(json.dumps(self._settings, indent=2))

    @property
    def settings(self) -> dict[str, Any]:
        return self._settings

    def update_settings(self, partial: dict[str, Any]) -> dict[str, Any]:
        before = self._settings
        self._settings = _deep_merge(before, partial)
        self.save_settings()
        before_indexing = before.get("indexing", {})
        new_indexing = self._settings.get("indexing", {})
        if before_indexing.get("device") != new_indexing.get("device"):
            with self._models_lock:
                if self._siglip is not None:
                    try:
                        self._siglip.offload()
                    except Exception:
                        pass
                self._siglip = None
                self._tagger = None
                self._device = None
        if before.get("databases") != self._settings.get("databases"):
            self.invalidate_search_cache()
        before_models = before.get("models", {})
        new_models = self._settings.get("models", {})
        siglip_keys = ("siglip_variant", "hf_token", "vram_idle_offload", "use_tensorrt", "siglip_max_num_patches")
        if any(before_models.get(k) != new_models.get(k) for k in siglip_keys):
            with self._models_lock:
                if self._siglip is not None:
                    try:
                        self._siglip.offload()
                    except Exception:
                        pass
                self._siglip = None
        tagger_keys = ("wd_tagger_variant", "hf_token", "default_tag_threshold")
        if any(before_models.get(k) != new_models.get(k) for k in tagger_keys):
            with self._models_lock:
                self._tagger = None
        sam2_keys = ("sam2_variant", "hf_token", "vram_idle_offload")
        if any(before_models.get(k) != new_models.get(k) for k in sam2_keys):
            with self._models_lock:
                if self._sam2 is not None:
                    try: self._sam2.offload()
                    except Exception: pass
                self._sam2 = None
        birefnet_keys = ("birefnet_variant", "hf_token", "vram_idle_offload")
        if any(before_models.get(k) != new_models.get(k) for k in birefnet_keys):
            with self._models_lock:
                if self._birefnet is not None:
                    try: self._birefnet.offload()
                    except Exception: pass
                self._birefnet = None
        return self._settings

    def replace_settings(self, full: dict[str, Any]) -> dict[str, Any]:
        """Import — replace entirely (still merged into defaults so missing keys keep defaults)."""
        base = json.loads(json.dumps(DEFAULT_SETTINGS))
        self._settings = _deep_merge(base, full)
        self.save_settings()
        with self._models_lock:
            self._siglip = None
            self._tagger = None
            self._device = None
        self.invalidate_search_cache()
        return self._settings

    @property
    def primary_db(self) -> str:
        return self._settings["databases"]["primary"] or str(default_db_path())

    @property
    def active_dbs(self) -> list[str]:
        return list(self._settings["databases"]["active"])

    # ---------- models ----------
    @property
    def device(self):
        if self._device is None:
            from .models import detect_device
            self._device = detect_device(self._settings["indexing"]["device"])
        return self._device

    def get_siglip(self):
        from .models import SigLIP2Model
        with self._models_lock:
            if self._siglip is None:
                self._siglip = SigLIP2Model(
                    self.device,
                    checkpoint=self._settings["models"]["siglip_variant"],
                    idle_offload_seconds=self._settings["models"]["vram_idle_offload"],
                    hf_token=self._settings["models"].get("hf_token") or None,
                    use_tensorrt=bool(self._settings["models"].get("use_tensorrt")),
                    max_num_patches=int(self._settings["models"].get("siglip_max_num_patches") or 128),
                )
            return self._siglip

    def get_tagger(self):
        from .models import WDTaggerModel
        with self._models_lock:
            if self._tagger is None:
                self._tagger = WDTaggerModel(
                    repo_id=self._settings["models"]["wd_tagger_variant"],
                    general_threshold=self._settings["models"]["default_tag_threshold"],
                    hf_token=self._settings["models"].get("hf_token") or None,
                    device_backend=self.device.backend,
                )
            return self._tagger

    def _offload_others_for_mask(self) -> None:
        """Free VRAM for whichever mask engine is about to load. Called before
        get_sam2 / get_birefnet — the indexing models (SigLIP/tagger) lazy-
        reload on next use so it's fine to evict them."""
        if self._siglip is not None:
            try: self._siglip.offload()
            except Exception: pass
        if self._tagger is not None:
            try: self._tagger.offload()
            except Exception: pass

    def _offload_other_mask_engines(self, keep: str) -> None:
        """Drop every loaded mask engine except ``keep``. Used by ``get_<X>()``
        so only one engine sits in VRAM at a time — fits 8 GB cards once
        SigLIP/tagger are evicted too."""
        for attr in ("_sam2", "_birefnet", "_matanyone"):
            if attr == keep:
                continue
            m = getattr(self, attr, None)
            if m is not None:
                try: m.offload()
                except Exception: pass
                setattr(self, attr, None)

    def get_sam2(self):
        """Lazy-load SAM 2 (Manual click engine). Reloading SigLIP/tagger
        after a mask session is ~2 s — cheaper than running the mask model
        on a fragmented VRAM heap."""
        from .models import SAM2Model
        with self._models_lock:
            if self._sam2 is None:
                self._offload_others_for_mask()
                self._offload_other_mask_engines(keep="_sam2")
                self._sam2 = SAM2Model(
                    self.device,
                    variant=self._settings["models"].get("sam2_variant", "base_plus"),
                    idle_offload_seconds=self._settings["models"].get("vram_idle_offload", 30),
                    hf_token=self._settings["models"].get("hf_token") or None,
                )
            return self._sam2

    def get_birefnet(self):
        """Lazy-load BiRefNet (Auto engine, default for anime)."""
        from .models import BiRefNetModel
        with self._models_lock:
            if self._birefnet is None:
                self._offload_others_for_mask()
                self._offload_other_mask_engines(keep="_birefnet")
                self._birefnet = BiRefNetModel(
                    self.device,
                    variant=self._settings["models"].get("birefnet_variant", "general"),
                    idle_offload_seconds=self._settings["models"].get("vram_idle_offload", 30),
                    hf_token=self._settings["models"].get("hf_token") or None,
                )
            return self._birefnet

    def get_matanyone(self):
        """Lazy-load MatAnyone (temporal propagation, seeded by BiRefNet/SAM 2).
        The seed has already been produced (and stored in the session) by the
        time this is called — so we can safely offload the seed-producing
        engine to free VRAM for the propagation pass."""
        from .models import MatAnyoneModel
        with self._models_lock:
            if self._matanyone is None:
                self._offload_others_for_mask()
                self._offload_other_mask_engines(keep="_matanyone")
                self._matanyone = MatAnyoneModel(
                    self.device,
                    idle_offload_seconds=self._settings["models"].get("vram_idle_offload", 30),
                    hf_token=self._settings["models"].get("hf_token") or None,
                )
            return self._matanyone

    def models_loaded(self) -> bool:
        return self._siglip is not None and self._tagger is not None

    # ---------- SAM 2 segmentation sessions ----------
    def register_sam2_session(self, session_id: str, session: Any) -> None:
        with self._sam2_sessions_lock:
            self._sam2_sessions[session_id] = session

    def get_sam2_session(self, session_id: str) -> Optional[Any]:
        with self._sam2_sessions_lock:
            return self._sam2_sessions.get(session_id)

    def drop_sam2_session(self, session_id: str) -> Optional[Any]:
        with self._sam2_sessions_lock:
            return self._sam2_sessions.pop(session_id, None)

    # ---------- search index cache ----------
    def invalidate_search_cache(self) -> None:
        with self._search_cache_lock:
            self._search_cache = None

    def _search_signature(self, conn) -> tuple[int, int, int]:
        row = conn.execute(
            """SELECT COUNT(*), COALESCE(MAX(s.id), 0), COALESCE(SUM(LENGTH(s.embedding)), 0)
                 FROM scenes s
                 JOIN videos v ON v.id = s.video_id
                WHERE v.status = 'completed' AND s.embedding IS NOT NULL"""
        ).fetchone()
        return int(row[0] or 0), int(row[1] or 0), int(row[2] or 0)

    def get_search_index(self) -> dict[str, Any]:
        import numpy as np

        db_path = self.primary_db
        schema.init_db(db_path)
        with schema.get_conn(db_path) as conn:
            signature = (db_path, *self._search_signature(conn))
            with self._search_cache_lock:
                if self._search_cache and self._search_cache["signature"] == signature:
                    return self._search_cache

            rows = queries.get_all_scenes_with_embeddings(conn)

        if rows:
            embeddings = np.vstack([np.frombuffer(r[6], dtype=np.float32) for r in rows]).astype(np.float32, copy=False)
            sids = np.array([r[0] for r in rows], dtype=np.int64)
            id_to_pos = {int(sid): i for i, sid in enumerate(sids)}
        else:
            embeddings = np.empty((0, 0), dtype=np.float32)
            sids = np.empty((0,), dtype=np.int64)
            id_to_pos = {}

        cache = {
            "signature": signature,
            "rows": rows,
            "sids": sids,
            "embeddings": embeddings,
            "id_to_pos": id_to_pos,
        }
        with self._search_cache_lock:
            self._search_cache = cache
        return cache

    def runtime_status(self) -> dict[str, Any]:
        warnings: list[str] = []
        tagger_provider = None
        device_backend = getattr(self._device, "backend", None)

        if self._tagger is not None:
            providers = self._tagger.active_providers()
            tagger_provider = providers[0] if providers else None
            warnings.extend(self._tagger.runtime_warnings())
        elif device_backend == "cuda":
            try:
                from .models.wd_tagger import cuda12_runtime_warning
                warning = cuda12_runtime_warning()
                if warning:
                    warnings.append(warning)
            except Exception as exc:
                warnings.append(f"Could not inspect wd-tagger CUDA runtime: {exc}")

        return {
            "tagger_provider": tagger_provider,
            "warnings": list(dict.fromkeys(warnings)),
        }

    # ---------- progress fan-out ----------
    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._listeners.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._listeners:
            self._listeners.remove(q)

    def publish(self, event: dict) -> None:
        msg = dict(event)
        print(f"progress: {json.dumps(msg)}", flush=True)
        if not self._loop:
            return
        for q in list(self._listeners):
            self._loop.call_soon_threadsafe(q.put_nowait, msg)

    # ---------- index worker ----------
    def start_indexing(self, phases: tuple[str, ...] = ("tag", "embed")) -> bool:
        if self._index_thread and self._index_thread.is_alive():
            return False
        self._cancel_event.clear()
        from .indexing import index_queue

        # Offload mask engines before indexing — they're the heaviest of the
        # bunch and not needed for tag/embed phases.
        with self._models_lock:
            for attr in ("_sam2", "_birefnet", "_matanyone"):
                m = getattr(self, attr, None)
                if m is not None:
                    try: m.offload()
                    except Exception: pass

        # Always instantiate both wrappers (cheap), but only force-load the
        # ones the requested phases actually need. Lazy load happens at
        # _ensure_loaded() inside the pipeline.
        siglip = self.get_siglip()
        tagger = self.get_tagger()

        kwargs = dict(
            phases=tuple(phases),
            mode=self._settings["indexing"]["mode"],
            scene_threshold=self._settings["indexing"]["scene_detect_threshold"],
            enable_sub_segmentation=self._settings["indexing"]["sub_segmentation"],
            sub_seg_threshold=self._settings["indexing"]["sub_segmentation_threshold"],
            tag_threshold=self._settings["models"]["default_tag_threshold"],
            batch_size=self._settings["indexing"]["batch_size"],
            auto_skip_indexed=self._settings["indexing"].get("auto_skip_indexed", True),
        )

        def run():
            try:
                self.publish({"type": "indexing", "message": "Starting", "percent": 0})
                index_queue(
                    self.primary_db, siglip, tagger,
                    cancel_event=self._cancel_event,
                    progress_cb=self.publish,
                    **kwargs,
                )
                self.invalidate_search_cache()
            except Exception as e:
                log.exception("indexing crashed")
                self.publish({"type": "indexing", "message": f"Failed: {e}", "percent": 0})
            finally:
                self.invalidate_search_cache()
                if self._settings["indexing"]["generate_proxies"]:
                    self._generate_proxies_for_new_scenes()
                self.publish({"type": "idle"})

        self._index_thread = threading.Thread(target=run, daemon=True)
        self._index_thread.start()
        return True

    def stop_indexing(self) -> None:
        self._cancel_event.set()

    def _generate_proxies_for_new_scenes(self) -> None:
        import concurrent.futures
        from .export.ffmpeg import _hw_h264_encoder
        from .paths import user_data_dir

        proxies_dir = user_data_dir() / "proxies"
        proxies_dir.mkdir(parents=True, exist_ok=True)
        quality = self._settings["indexing"]["proxy_quality"]

        # Fetch every orphan scene grouped by video so we can run one ffmpeg
        # per video instead of one per scene. The order matters: scenes must
        # come out sorted by start_ms within each video for the `segment`
        # muxer to map cleanly onto them.
        with schema.get_conn(self.primary_db) as conn:
            rows = conn.execute(
                """SELECT s.id, v.id, v.filepath, s.start_ms, s.end_ms
                     FROM scenes s JOIN videos v ON v.id = s.video_id
                    WHERE s.proxy_path IS NULL AND v.status = 'completed'
                    ORDER BY v.id, s.start_ms"""
            ).fetchall()
        total = len(rows)
        if total == 0:
            return

        by_video: dict[int, tuple[str, list[tuple[int, int, int]]]] = {}
        for sid, vid, fp, s_ms, e_ms in rows:
            slot = by_video.setdefault(vid, (fp, []))
            slot[1].append((sid, s_ms, e_ms))

        use_batched = _hw_h264_encoder() is not None
        completed = [0]
        progress_lock = threading.Lock()

        def _publish(delta: int):
            with progress_lock:
                completed[0] += delta
                pct = int((completed[0] / total) * 100) if total else 100
            self.publish({"type": "proxy", "percent": pct})

        def _do_video(item):
            if self._cancel_event.is_set():
                return
            video_id, (video_path, scenes) = item
            produced = self._proxies_for_one_video(
                video_id, video_path, scenes, proxies_dir, quality, use_batched=use_batched,
            )
            _publish(produced)

        # 2 ffmpeg pipelines in parallel: each owns one hardware decode + one
        # hardware encode session (NVDEC+NVENC or VideoToolbox), well under
        # what consumer GPUs support. Going higher would contend on memory
        # bandwidth and gain little (hw decode is faster than the h264 source).
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            futures = [ex.submit(_do_video, item) for item in by_video.items()]
            for _ in concurrent.futures.as_completed(futures):
                if self._cancel_event.is_set():
                    for f in futures:
                        f.cancel()
                    return

    def _proxies_for_one_video(
        self,
        video_id: int,
        video_path: str,
        scenes: list[tuple[int, int, int]],
        proxies_dir,
        quality: str,
        *,
        use_batched: bool,
    ) -> int:
        """Generate proxies for every scene of one video. Returns count produced.

        Strategy: when a hardware H.264 encoder is available (NVENC or
        VideoToolbox) AND the scenes are contiguous (every end_ms[i] ==
        start_ms[i+1] — the normal case for `fast` indexing without
        sub-segmentation), we spawn ONE ffmpeg with the segment muxer that
        streams a single hw-decode+scale+hw-encode pass and writes N output
        files. This eliminates ~200 ms of spawn overhead per scene (~1 min
        saved per episode of ~300 scenes).

        We fall back to per-scene generation when scenes overlap (sub-seg
        children inside parents) or when no hardware encoder is available.
        """
        from .export.ffmpeg import generate_proxy

        scenes_sorted = sorted(scenes, key=lambda x: x[1])

        if use_batched and len(scenes_sorted) >= 2:
            contiguous = all(
                scenes_sorted[i][2] == scenes_sorted[i + 1][1]
                for i in range(len(scenes_sorted) - 1)
            )
            if contiguous:
                try:
                    return self._proxies_batched(video_path, scenes_sorted, proxies_dir, quality)
                except Exception as e:
                    log.warning(
                        "batched proxy gen failed for video %s, falling back to per-scene: %s",
                        video_id, e,
                    )

        count = 0
        for sid, s_ms, e_ms in scenes_sorted:
            if self._cancel_event.is_set():
                return count
            out = proxies_dir / f"scene_{sid}.mp4"
            try:
                generate_proxy(video_path, s_ms, e_ms, str(out), quality=quality)
                with schema.get_conn(self.primary_db) as conn:
                    queries.set_scene_proxy(conn, sid, str(out))
                count += 1
            except Exception as e:
                log.warning("proxy gen failed for scene %s: %s", sid, e)
        return count

    def _proxies_batched(
        self,
        video_path: str,
        scenes: list[tuple[int, int, int]],
        proxies_dir,
        quality: str,
    ) -> int:
        """Single ffmpeg invocation that emits all proxies of one video via
        the `segment` muxer. Caller has already verified contiguity."""
        import shutil
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        from .indexing.cuts import _ffmpeg_path
        from .export.ffmpeg import _PROXY_BITRATES, _hw_h264_encoder

        bitrate = _PROXY_BITRATES.get(quality, _PROXY_BITRATES["medium"])
        hw_encoder = _hw_h264_encoder()
        if hw_encoder is None:
            raise RuntimeError("no hardware encoder, skipping batched path")
        # Break points = end_ms of every scene except the last. ffmpeg emits
        # N+1 segments for N break points; with N = len(scenes)-1 we get
        # exactly len(scenes) segments.
        break_times = [s[2] / 1000.0 for s in scenes[:-1]]
        if not break_times:
            # Single-scene video: batching gains nothing, let the per-scene
            # path handle it.
            raise RuntimeError("only one scene, skipping batched path")
        segment_times_str = ",".join(f"{t:.3f}" for t in break_times)

        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        with tempfile.TemporaryDirectory(prefix="amv_proxies_", dir=str(proxies_dir)) as tmp:
            tmp_path = Path(tmp)
            seg_pattern = tmp_path / "seg_%05d.mp4"

            # `-force_key_frames` at the same instants the muxer cuts on is
            # what guarantees clean splits — otherwise segments would start
            # with junk P/B frames and be slow to seek into.
            if hw_encoder == "h264_nvenc":
                decode_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
                scale_args = ["-vf", "scale_cuda=240:-2:format=nv12,hwdownload,format=nv12"]
                encode_args = ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll"]
            else:  # h264_videotoolbox
                # VT decode auto-downloads to CPU frames; the 240px-wide CPU
                # downscale is negligible next to the hw decode + encode.
                decode_args = ["-hwaccel", "videotoolbox"]
                scale_args = ["-vf", "scale=240:-2:flags=fast_bilinear"]
                encode_args = ["-c:v", "h264_videotoolbox", "-realtime", "1", "-allow_sw", "1"]
            cmd = [
                _ffmpeg_path(),
                "-hide_banner", "-nostats", "-loglevel", "error",
                "-fflags", "+discardcorrupt",
                *decode_args,
                "-i", video_path,
                "-an", "-sn",
                *scale_args,
                "-r", "15",
                *encode_args,
                "-b:v", bitrate,
                "-force_key_frames", segment_times_str,
                "-f", "segment",
                "-segment_times", segment_times_str,
                "-reset_timestamps", "1",
                str(seg_pattern),
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, creationflags=flags,
            )
            if proc.returncode != 0:
                stderr_tail = (proc.stderr or "")[-500:]
                raise RuntimeError(f"ffmpeg segment exited {proc.returncode}: {stderr_tail!r}")

            count = 0
            for i, (sid, _, _) in enumerate(scenes):
                src = tmp_path / f"seg_{i:05d}.mp4"
                if not src.exists():
                    log.warning("segment %05d missing for scene %s — skipping", i, sid)
                    continue
                dst = proxies_dir / f"scene_{sid}.mp4"
                shutil.move(str(src), str(dst))
                with schema.get_conn(self.primary_db) as conn:
                    queries.set_scene_proxy(conn, sid, str(dst))
                count += 1
            return count


def python_version_string() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


_state: Optional[AppState] = None


def get_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState()
    return _state
