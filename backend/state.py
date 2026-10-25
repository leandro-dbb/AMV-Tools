"""Singleton state shared across API handlers (settings cache, models, jobs)."""
from __future__ import annotations

import asyncio
import json
import logging
import os
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
        self._siglip_owned = None   # the _siglip this state built (vs. injected)
        self._siglip_ckpt = None    # its checkpoint
        self._siglip_aux = None     # a second checkpoint asked explicitly (re-embed job)
        self._db_model_cache = None  # (db_path, search generation, model)
        self._tagger = None     # WDTaggerModel
        self._sam2 = None       # SAM2Model — Manual click engine
        self._birefnet = None   # BiRefNetModel — Auto engine (default for anime)
        self._matanyone = None  # MatAnyoneModel — temporal propagation after BiRefNet seed
        self._sam3 = None       # Sam3Concept — Auto engine character selector (gated weights)
        self._device = None
        self._sam2_sessions: dict[str, Any] = {}     # session_id → SegmentSession
        self._sam2_sessions_lock = threading.Lock()
        self._search_cache = None
        self._search_generation = 0
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

        self._migrate_model_tier()

        if not self._settings["databases"]["active"]:
            default_db = str(default_db_path())
            schema.init_db(default_db)
            self._settings["databases"]["active"] = [default_db]
            self._settings["databases"]["primary"] = default_db
            self.save_settings()
        else:
            for db in self._settings["databases"]["active"]:
                schema.init_db(db)

    def _migrate_model_tier(self) -> None:
        """Model tiers (0.4): installs that picked their own SigLIP / tagger /
        patches keep them ("custom") instead of moving to the auto tier."""
        if self._settings.get("_model_tier_migrated"):
            return
        from .config import DEFAULT_SIGLIP, DEFAULT_WD_TAGGER
        models = self._settings.setdefault("models", {})
        if models.get("tier", "auto") == "auto" and (
                (models.get("siglip_variant") or DEFAULT_SIGLIP) != DEFAULT_SIGLIP
                or (models.get("wd_tagger_variant") or DEFAULT_WD_TAGGER) != DEFAULT_WD_TAGGER
                or int(models.get("siglip_max_num_patches") or 128) != 128):
            log.info("model tier: keeping the hand-picked models (tier = custom)")
            models["tier"] = "custom"
        self._settings["_model_tier_migrated"] = True
        self.save_settings()

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
        siglip_keys = ("siglip_variant", "hf_token", "vram_idle_offload", "use_tensorrt", "siglip_max_num_patches",
                       "tier")
        if any(before_models.get(k) != new_models.get(k) for k in siglip_keys):
            with self._models_lock:
                if self._siglip is not None:
                    try:
                        self._siglip.offload()
                    except Exception:
                        pass
                self._siglip = None
        tagger_keys = ("wd_tagger_variant", "hf_token", "default_tag_threshold", "tier")
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
        if before_models.get("hf_token") != new_models.get("hf_token"):
            from .models import sam3_concept
            sam3_concept._ACCESS.clear()   # a new token may open the gated weights
            with self._models_lock:
                if self._sam3 is not None:
                    try: self._sam3.offload()
                    except Exception: pass
                self._sam3 = None
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

    # ---------- model tiers (backend/models/tiers.py) ----------
    def model_selection(self) -> dict[str, Any]:
        """Effective models of the selected tier on this machine."""
        from .models import tiers
        return tiers.resolve(self._settings.get("models", {}), self.device)

    def embedding_target(self) -> tuple[str, int]:
        """(checkpoint, patches) the tier wants for semantic search."""
        emb = self.model_selection()["models"]["embedding"]
        return emb["model"], int(emb.get("patches") or 128)

    def db_embedding_model(self, db_path: Optional[str] = None) -> str:
        """Embedding model of a database (the primary by default): queries
        and newly indexed videos must use it. Unbound (no vectors yet): the
        tier's model."""
        from .indexing import reembed
        db = db_path or self.primary_db
        with self._search_cache_lock:
            generation = self._search_generation
        cached = self._db_model_cache
        if cached is not None and cached[0] == db and cached[1] == generation:
            return cached[2]
        model = reembed.db_embedding_model(db, default=self.embedding_target()[0])
        self._db_model_cache = (db, generation, model)
        return model

    def _make_siglip(self, checkpoint: str):
        from .models import SigLIP2Model
        _target, patches = self.embedding_target()
        return SigLIP2Model(
            self.device,
            checkpoint=checkpoint,
            idle_offload_seconds=self._settings["models"]["vram_idle_offload"],
            hf_token=self._settings["models"].get("hf_token") or None,
            use_tensorrt=bool(self._settings["models"].get("use_tensorrt")),
            max_num_patches=patches,
        )

    def get_siglip(self, checkpoint: Optional[str] = None):
        """SigLIP for ``checkpoint``; by default the primary database's
        embedding model (never mix embedding spaces in one database). An
        explicit checkpoint other than the resident one (the re-embed job's
        target) gets its own slot so searches on the old model don't evict
        it."""
        explicit = checkpoint is not None
        cur = self._siglip
        if cur is not None and cur is not self._siglip_owned:
            return cur   # injected (tests, benches)
        if checkpoint is None:
            checkpoint = self.db_embedding_model()
        with self._models_lock:
            cur = self._siglip
            if cur is not None and (cur is not self._siglip_owned or self._siglip_ckpt == checkpoint):
                return cur
            aux = self._siglip_aux
            if aux is not None and getattr(aux, "checkpoint", None) == checkpoint:
                return aux
            model = self._make_siglip(checkpoint)
            if explicit:
                if aux is not None:
                    try: aux.offload()
                    except Exception: pass
                self._siglip_aux = model
            else:
                if cur is not None:
                    try: cur.offload()
                    except Exception: pass
                self._siglip = self._siglip_owned = model
                self._siglip_ckpt = checkpoint
            return model

    def promote_siglip(self, checkpoint: str) -> None:
        """The re-embed job switched the library: its model becomes the one
        search uses."""
        with self._models_lock:
            aux = self._siglip_aux
            if aux is None or getattr(aux, "checkpoint", None) != checkpoint:
                return
            cur = self._siglip
            if cur is not None and cur is self._siglip_owned:
                try: cur.offload()
                except Exception: pass
            if cur is None or cur is self._siglip_owned:
                self._siglip = self._siglip_owned = aux
                self._siglip_ckpt = checkpoint
            self._siglip_aux = None

    def drop_aux_siglip(self) -> None:
        with self._models_lock:
            aux, self._siglip_aux = self._siglip_aux, None
        if aux is not None:
            try: aux.offload()
            except Exception: pass

    def get_tagger(self):
        from .models import WDTaggerModel
        with self._models_lock:
            if self._tagger is None:
                self._tagger = WDTaggerModel(
                    repo_id=self.model_selection()["models"]["tagger"],
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
        for attr in ("_sam2", "_birefnet", "_matanyone", "_sam3"):
            if attr == keep:
                continue
            m = getattr(self, attr, None)
            if m is not None:
                try: m.offload()
                except Exception: pass
                setattr(self, attr, None)

    def get_sam2(self, variant: Optional[str] = None):
        """Lazy-load SAM 2 (Manual click engine, and the tracker of the Auto
        engine). Reloading SigLIP/tagger after a mask session is ~2 s —
        cheaper than running the mask model on a fragmented VRAM heap.
        ``variant`` overrides the setting ("auto" = the memory tier's pick)."""
        from .models import SAM2Model
        from .models.auto_roto import resolve_mask_variant
        variant = resolve_mask_variant(self, "tracker", variant or self._settings["models"].get("sam2_variant", "auto"))
        with self._models_lock:
            if self._sam2 is not None and getattr(self._sam2, "variant", variant) != variant:
                try: self._sam2.offload()
                except Exception: pass
                self._sam2 = None
            if self._sam2 is None:
                self._offload_others_for_mask()
                self._offload_other_mask_engines(keep="_sam2")
                self._sam2 = SAM2Model(
                    self.device,
                    variant=variant,
                    idle_offload_seconds=self._settings["models"].get("vram_idle_offload", 30),
                    hf_token=self._settings["models"].get("hf_token") or None,
                )
            return self._sam2

    def get_birefnet(self, variant: Optional[str] = None):
        """Lazy-load the Auto engine's per-frame detector (ToonOut / ISNet
        anime / BiRefNet). ``variant`` overrides the setting ("auto" = the
        memory tier's pick)."""
        from .models import BiRefNetModel
        from .models.auto_roto import resolve_mask_variant
        variant = resolve_mask_variant(self, "detector", variant or self._settings["models"].get("birefnet_variant", "auto"))
        with self._models_lock:
            if self._birefnet is not None and getattr(self._birefnet, "variant", variant) != variant:
                try: self._birefnet.offload()
                except Exception: pass
                self._birefnet = None
            if self._birefnet is None:
                self._offload_others_for_mask()
                self._offload_other_mask_engines(keep="_birefnet")
                self._birefnet = BiRefNetModel(
                    self.device,
                    variant=variant,
                    idle_offload_seconds=self._settings["models"].get("vram_idle_offload", 30),
                    hf_token=self._settings["models"].get("hf_token") or None,
                )
            return self._birefnet

    def get_sam3(self):
        """Lazy SAM 3 concept segmenter (Auto engine character selector), or
        None when its gated weights are not accessible with the user's token
        (then the Auto engine runs as in 0.5)."""
        from .models.sam3_concept import Sam3Concept, weights_accessible
        token = self._settings["models"].get("hf_token") or None
        if not weights_accessible(token):
            return None
        with self._models_lock:
            if self._sam3 is None:
                self._offload_others_for_mask()
                self._offload_other_mask_engines(keep="_sam3")
                self._sam3 = Sam3Concept(self.device, hf_token=token)
            return self._sam3

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
        """Drop the in-memory search index. The index is NOT re-validated
        against the DB on every search (that full scan cost ~70-350 ms on 50k
        scenes), so every in-process writer that changes the embedded scene
        set must call this: indexing (per finished video + end of run), scene
        merge, database merge/cleanup, database switch/import."""
        with self._search_cache_lock:
            self._search_cache = None
            self._search_generation += 1

    @staticmethod
    def _rows_signature(rows) -> tuple[int, int, int]:
        """Content signature of the embedded scene set: (count, max id,
        total embedding bytes). Persisted by Search+ characters
        (``scored_signature``) to detect stale matches across restarts, so it
        must stay content-based — computed once per index build from the rows
        already loaded, never per search."""
        if not rows:
            return 0, 0, 0
        return len(rows), max(int(r[0]) for r in rows), sum(len(r[6]) for r in rows)

    def start_search_warmup(self) -> None:
        """NVIDIA and Apple Silicon: load SigLIP 2 (and the torch/transformers
        imports behind it) and the search index in the background as the
        sidecar starts, so the first search does not pay the cold start
        (M1 Pro: 3.6 s). Only with a library to search and a GPU with room to
        spare; the idle-offload watcher frees the memory again if nobody
        searches."""
        def run():
            try:
                dev = self.device
                if getattr(dev, "backend", None) not in ("cuda", "mps") or (getattr(dev, "vram_gb", None) or 0) < 8:
                    return
                if not self.get_search_index()["rows"]:
                    return
                siglip = self.get_siglip()
                siglip.embed_text("warm-up")
            except Exception as exc:  # the first search loads it as before
                log.debug("search warm-up skipped: %s", exc)

        threading.Thread(target=run, daemon=True, name="search-warmup").start()

    def get_search_index(self) -> dict[str, Any]:
        import numpy as np

        db_path = self.primary_db
        with self._search_cache_lock:
            cache = self._search_cache
            generation = self._search_generation
        if cache is not None and cache["db_path"] == db_path:
            return cache

        schema.init_db(db_path)
        with schema.get_conn(db_path) as conn:
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
            "db_path": db_path,
            "signature": (db_path, *self._rows_signature(rows)),
            "rows": rows,
            "sids": sids,
            "embeddings": embeddings,
            "id_to_pos": id_to_pos,
        }
        with self._search_cache_lock:
            # An invalidation that landed while we were reading may have
            # missed rows: serve this build once but don't keep it.
            if self._search_generation == generation:
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

    def close_listeners(self) -> None:
        """Shutdown: ``None`` tells each progress websocket to close."""
        for q in list(self._listeners):
            if self._loop:
                self._loop.call_soon_threadsafe(q.put_nowait, None)
            else:
                q.put_nowait(None)

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
        from .indexing import reembed
        if reembed.is_running():
            return False   # never two embedding writers on one library
        from .db.merge import merge_running
        if merge_running():
            return False   # the merge holds the library's write lock
        self._cancel_event.clear()
        from .indexing import index_queue

        # Offload mask engines before indexing — they're the heaviest of the
        # bunch and not needed for tag/embed phases.
        with self._models_lock:
            for attr in ("_sam2", "_birefnet", "_matanyone", "_sam3"):
                m = getattr(self, attr, None)
                if m is not None:
                    try: m.offload()
                    except Exception: pass

        # Always instantiate both wrappers (cheap), but only force-load the
        # ones the requested phases actually need. Lazy load happens at
        # _ensure_loaded() inside the pipeline.
        siglip = self.get_siglip()
        tagger = self.get_tagger()
        if "embed" in phases:
            try:
                reembed.bind_model(self.primary_db, getattr(siglip, "checkpoint", None) or self.embedding_target()[0],
                                   getattr(siglip, "max_num_patches", None))
            except Exception as exc:
                log.warning("could not record the library's embedding model: %s", exc)

        kwargs = dict(
            phases=tuple(phases),
            mode=self._settings["indexing"]["mode"],
            scene_threshold=self._settings["indexing"]["scene_detect_threshold"],
            enable_sub_segmentation=self._settings["indexing"]["sub_segmentation"],
            sub_seg_threshold=self._settings["indexing"]["sub_segmentation_threshold"],
            tag_threshold=self._settings["models"]["default_tag_threshold"],
            batch_size=self._settings["indexing"]["batch_size"],
            auto_skip_indexed=self._settings["indexing"].get("auto_skip_indexed", True),
            proxy_sources=bool(self._settings["indexing"]["generate_proxies"]),
        )

        def progress(event: dict) -> None:
            # A finished episode is now searchable: drop the cached index so
            # searches made while the rest of the queue runs include it.
            if event.get("type") == "indexing" and event.get("percent") == 100 and event.get("message") == "Done":
                self.invalidate_search_cache()
                if early_proxies:
                    self._kick_early_proxies()
            self.publish(event)

        # Apple Silicon: the indexing decode leaves CPU free (VideoToolbox
        # sessions), so a finished episode's hover proxies are made while the
        # next one indexes instead of all of them after the queue.
        early_proxies = bool(self._settings["indexing"]["generate_proxies"]) and \
            getattr(self.device, "backend", None) == "mps"

        def run():
            overlap = None
            try:
                self.publish({"type": "indexing", "message": "Starting", "percent": 0})
                self._set_proxy_source_cache()
                from .indexing import onepass as _op
                video_done = None
                if (self._settings["indexing"]["generate_proxies"] and _op._proxy_cache_dir is not None
                        and os.environ.get("AMV_PROXY_OVERLAP", "1") != "0"):
                    # NVIDIA: each video's hover proxies are cut from its 240p
                    # copy as soon as it is indexed, beside the next decodes
                    # (they used to wait for the whole queue: ~28 s at the end
                    # of a 3-episode season).
                    import concurrent.futures
                    overlap = concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                                                    thread_name_prefix="proxy-overlap")
                    cache = _op._proxy_cache_dir

                    def video_done(path: str, last: bool = False) -> None:
                        # the last one is the tail of the queue: 3 chunks side
                        # by side; the others decide when they start
                        overlap.submit(self._generate_proxies_for_new_scenes, cache, path, False,
                                       3 if last else None)
                self._indexing_active = True
                try:
                    index_queue(
                        self.primary_db, siglip, tagger,
                        cancel_event=self._cancel_event,
                        progress_cb=progress,
                        video_done_cb=video_done,
                        **kwargs,
                    )
                finally:
                    self._indexing_active = False
                self.invalidate_search_cache()
            except Exception as e:
                log.exception("indexing crashed")
                self.publish({"type": "indexing", "message": f"Failed: {e}", "percent": 0})
            finally:
                self.invalidate_search_cache()
                from .indexing import onepass
                cache_dir = onepass._proxy_cache_dir
                onepass.set_proxy_cache(None)
                if overlap is not None:
                    overlap.shutdown(wait=True)
                t = getattr(self, "_early_proxy_thread", None)
                if t is not None:
                    t.join()
                try:
                    if self._settings["indexing"]["generate_proxies"]:
                        self._generate_proxies_for_new_scenes(cache_dir)
                finally:
                    if cache_dir is not None:
                        import shutil
                        shutil.rmtree(cache_dir, ignore_errors=True)
                self.publish({"type": "idle"})

        self._index_thread = threading.Thread(target=run, daemon=True)
        self._index_thread.start()
        return True

    def stop_indexing(self) -> None:
        self._cancel_event.set()

    def _set_proxy_source_cache(self) -> None:
        """NVIDIA + NVENC + proxies on: the indexing decode also writes each
        video's 240p proxy source (see ``onepass.set_proxy_cache``)."""
        from .indexing import onepass
        onepass.set_proxy_cache(None)
        try:
            from .export.ffmpeg import _hw_h264_encoder
            from .paths import user_data_dir
            if (self._settings["indexing"]["generate_proxies"] and onepass.available()
                    and _hw_h264_encoder() == "h264_nvenc"):
                onepass.set_proxy_cache(user_data_dir() / "proxies" / "_source")
        except Exception as exc:  # proxies then decode the source, as before
            log.debug("proxy source cache unavailable: %s", exc)

    def _kick_early_proxies(self) -> None:
        lock = self.__dict__.setdefault("_early_proxy_lock", threading.Lock())
        with lock:
            self._early_proxy_dirty = True
            t = getattr(self, "_early_proxy_thread", None)
            if t is not None and t.is_alive():
                return

            def loop():
                while not self._cancel_event.is_set():
                    with lock:
                        if not self._early_proxy_dirty:
                            return
                        self._early_proxy_dirty = False
                    try:
                        self._generate_proxies_for_new_scenes(final=False)
                    except Exception:
                        log.exception("early hover proxies failed (the final pass retries)")

            self._early_proxy_thread = threading.Thread(target=loop, daemon=True, name="early-proxies")
            self._early_proxy_thread.start()

    def _generate_proxies_for_new_scenes(self, source_cache=None, only_video: Optional[str] = None,
                                         final: bool = True, chunks: Optional[int] = None) -> None:
        """Proxies of every completed video that lacks them (or of
        ``only_video``). ``final=False`` (while the queue still runs): no
        progress events, and the 240p sources of videos still being indexed
        are left alone. ``source_cache``: NVIDIA's 240p copies
        (``onepass.set_proxy_cache``)."""
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
            if only_video is not None:
                rows = [r for r in rows if r[2] == only_video]
            # sub-segmented scenes: parent id -> its children's ids
            children: dict[int, list[int]] = {}
            for cid, pid in conn.execute(
                    "SELECT id, parent_scene_id FROM scenes WHERE parent_scene_id IS NOT NULL"):
                children.setdefault(pid, []).append(cid)
        self._proxy_children = children
        total = len(rows)
        if total == 0:
            from .indexing import proxy_source
            if final:
                proxy_source.discard_all()
            return

        by_video: dict[int, tuple[str, list[tuple[int, int, int]]]] = {}
        for sid, vid, fp, s_ms, e_ms in rows:
            slot = by_video.setdefault(vid, (fp, []))
            slot[1].append((sid, s_ms, e_ms))

        use_batched = _hw_h264_encoder() is not None
        # nothing else running (end of the queue): split each pass in 3
        if chunks is None:
            # beside running decodes: one niced chunk; once the queue has
            # nothing left to decode: 3 chunks at normal priority
            from .indexing.pipeline import decodes_running
            busy = getattr(self, "_indexing_active", False) and decodes_running() > 0
            chunks = 1 if busy else 3
        completed = [0]
        progress_lock = threading.Lock()

        def _publish(delta: int):
            with progress_lock:
                completed[0] += delta
                pct = int((completed[0] / total) * 100) if total else 100
            if final:
                self.publish({"type": "proxy", "percent": pct})

        def _do_video(item):
            if self._cancel_event.is_set():
                return
            video_id, (video_path, scenes) = item
            produced = self._proxies_for_one_video(
                video_id, video_path, scenes, proxies_dir, quality, use_batched=use_batched,
                source_cache=source_cache, chunks=chunks,
            )
            _publish(produced)

        # 2 ffmpeg pipelines in parallel: each owns one hardware decode + one
        # hardware encode session (NVDEC+NVENC or VideoToolbox), well under
        # what consumer GPUs support. Going higher would contend on memory
        # bandwidth and gain little (hw decode is faster than the h264 source).
        from .indexing import proxy_source
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                futures = [ex.submit(_do_video, item) for item in by_video.items()]
                for _ in concurrent.futures.as_completed(futures):
                    if self._cancel_event.is_set():
                        for f in futures:
                            f.cancel()
                        return
        finally:
            if final:
                proxy_source.discard_all()

    def _proxies_for_one_video(
        self,
        video_id: int,
        video_path: str,
        scenes: list[tuple[int, int, int]],
        proxies_dir,
        quality: str,
        *,
        use_batched: bool,
        source_cache=None,
        chunks: int = 1,
    ) -> int:
        """Generate proxies for every scene of one video. Returns count produced.

        With a hardware H.264 encoder (NVENC or VideoToolbox) ONE ffmpeg pass
        encodes the whole video and the segment muxer cuts it at every scene
        boundary (no ~200 ms process spawn and seek per scene). The leaf
        scenes — hard cuts that were not split, and sub-segments — tile the
        timeline (gaps where a flash was dropped are skipped); a split
        scene's proxy is its children's segments joined without re-encoding.
        Per-scene generation remains the fallback (no hardware encoder, or a
        failed pass).
        """
        from .export.ffmpeg import generate_proxy

        scenes_sorted = sorted(scenes, key=lambda x: x[1])
        done: set[int] = set()
        count = 0
        if use_batched and len(scenes_sorted) >= 2:
            try:
                count = self._proxies_batched(video_path, scenes_sorted, proxies_dir, quality, done,
                                              source_cache=source_cache, chunks=chunks)
                if source_cache is not None:
                    from .indexing.onepass import proxy_source_for
                    small = proxy_source_for(video_path, source_cache)
                    if small is not None:
                        small.unlink(missing_ok=True)   # one 240p copy at a time on disk
            except Exception as e:
                log.warning(
                    "batched proxy gen failed for video %s, falling back to per-scene: %s",
                    video_id, e,
                )
        # the 240p source recorded by the indexing decode has served
        from .indexing import proxy_source
        proxy_source.discard(video_path)

        for sid, s_ms, e_ms in scenes_sorted:
            if sid in done:
                continue
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
        done: set[int],
        source_cache=None,
        chunks: int = 1,
    ) -> int:
        """Single ffmpeg pass that emits the proxies of one video through
        the `segment` muxer; adds the ids it produced to ``done``."""
        import shutil
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        from .indexing.cuts import _ffmpeg_hw_profile, _ffmpeg_path
        from .export.ffmpeg import _PROXY_BITRATES, _hw_h264_encoder

        bitrate = _PROXY_BITRATES.get(quality, _PROXY_BITRATES["medium"])
        hw_encoder = _hw_h264_encoder()
        if hw_encoder is None:
            raise RuntimeError("no hardware encoder, skipping batched path")
        children = getattr(self, "_proxy_children", {})
        by_id = {sid: (s, e) for sid, s, e in scenes}
        parents = [sid for sid, _s, _e in scenes if children.get(sid)]
        leaves = [(sid, s, e) for sid, s, e in scenes if not children.get(sid)]
        # Leaves must not overlap (they tile the timeline); anything else is
        # left to the per-scene path.
        leaves = [x for i, x in enumerate(leaves) if i == 0 or x[1] >= leaves[i - 1][2]]
        bounds = sorted({t for _sid, s, e in leaves for t in (s, e)})
        breaks = [t for t in bounds if t > 0]
        if len(leaves) < 2 or not breaks:
            raise RuntimeError("fewer than two scenes, skipping batched path")
        # segment k covers [edges[k], edges[k + 1]) with edges = [0] + breaks
        edges = [0] + breaks

        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        with tempfile.TemporaryDirectory(prefix="amv_proxies_", dir=str(proxies_dir)) as tmp:
            tmp_path = Path(tmp)

            # `-force_key_frames` at the instants the muxer cuts on is what
            # guarantees clean splits. NVENC needs `-forced-idr 1`: its forced
            # frames are plain I-frames otherwise, which the segment muxer
            # does not cut on — most scenes of a video ended up without a
            # proxy (measured: 18 % produced on a 4-min extract).
            prof = _ffmpeg_hw_profile()
            from .indexing.onepass import proxy_source_for
            small = proxy_source_for(video_path, source_cache) if source_cache is not None else None
            src_path = video_path
            chunkable = False
            if small is not None and hw_encoder == "h264_nvenc":
                # 240p copy written by the indexing decode: NVDEC ->
                # NVENC with the cut key frames, no 1080p decode again.
                src_path = str(small)
                chunkable = True
                decode_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
                from .indexing.onepass import has_decoder
                if has_decoder("h264_cuvid"):
                    # NVIDIA's own parser, asynchronous: 1.6x faster than the
                    # hwaccel decode on 240p frames (per-frame latency bound)
                    decode_args += ["-c:v", "h264_cuvid"]
                scale_args = []
                encode_args = ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll", "-forced-idr", "1"]
            elif hw_encoder == "h264_nvenc" and prof is not None and prof.get("name") == "cuda":
                # NVDEC -> scale_cuda -> NVENC: the frames never leave the GPU.
                decode_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
                scale_args = ["-vf", "scale_cuda=-2:240:format=nv12"]
                encode_args = ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll", "-forced-idr", "1"]
            elif hw_encoder == "h264_nvenc":
                decode_args = []
                scale_args = ["-vf", "scale=-2:240"]
                encode_args = ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll", "-forced-idr", "1"]
            else:  # h264_videotoolbox
                # Software decode unless VideoToolbox measured faster for
                # this codec (M1 Pro, 4-min 1080p H.264: 35.6 s -> 8.0 s).
                from .export.ffmpeg import _videotoolbox_decode_args
                decode_args = _videotoolbox_decode_args(video_path)
                scale_args = ["-vf", "scale=-2:240"]
                # No `-realtime 1` here: it paces VideoToolbox for live
                # capture — 3.6 s instead of 8.3 s for the proxies of 4 min.
                encode_args = ["-c:v", "h264_videotoolbox", "-allow_sw", "1"]
                # The 240p copy recorded by the indexing decode, if any: no
                # second decode of the episode (indexing.proxy_source).
                from .indexing import proxy_source
                recorded = proxy_source.existing(video_path)
                if recorded is not None:
                    src_path, decode_args, scale_args = str(recorded), [], []
                    # 240p frames are too small for VideoToolbox, whose cost
                    # is per frame: two passes over the copies of a 4-min
                    # extract take 7.2 s there, 1.8 s with libx264 superfast
                    # on the (otherwise idle) CPU, at 43.1 dB instead of 41.2
                    # from a lossless reference, same size.
                    encode_args = ["-c:v", "libx264", "-preset", "superfast", "-pix_fmt", "yuv420p"]
            # Time chunks cut on scene edges, encoded side by side (the 240p
            # copy only: accurate seeks are cheap there). Used when nothing
            # else needs the CPU — the last video of a queue (NVIDIA).
            bounds_at = [0]
            n_chunks = chunks if chunkable else 1
            for j in range(1, n_chunks):
                target = edges[-1] * j / n_chunks
                cand = [e for e in edges if e > bounds_at[-1]]
                if len(cand) < 3:
                    break
                bounds_at.append(min(cand[:-1], key=lambda e: abs(e - target)))
            seg_at: dict[int, Path] = {}
            cmds = []
            for j, a in enumerate(bounds_at):
                b = bounds_at[j + 1] if j + 1 < len(bounds_at) else None
                local = [e for e in edges if e >= a and (b is None or e < b)]
                local_times = ",".join(f"{(e - a) / 1000.0:.3f}" for e in local if e > a)
                if not local_times and b is not None:
                    raise RuntimeError("empty proxy chunk")
                pattern = tmp_path / f"c{j}_seg_%05d.mp4"
                for k, e in enumerate(local):
                    seg_at[e] = tmp_path / f"c{j}_seg_{k:05d}.mp4"
                seek = ["-ss", f"{a / 1000.0:.3f}"] if a > 0 else []
                stop = ["-to", f"{b / 1000.0:.3f}"] if b is not None else []
                # force_key_frames puts each key frame on the 15 fps frame
                # *nearest* the cut, up to half a frame early, while the
                # segment muxer only splits at key frames at or after the
                # requested time: without the delta it skipped such cuts
                # and every later proxy belonged to the previous scene.
                split = ["-force_key_frames", local_times, "-f", "segment", "-segment_times", local_times,
                         "-segment_time_delta", "0.034"] \
                    if local_times else ["-f", "segment", "-segment_time", "86400"]
                cmds.append([
                    _ffmpeg_path(),
                    "-hide_banner", "-nostats", "-loglevel", "error",
                    "-fflags", "+discardcorrupt",
                    *decode_args,
                    *seek, *stop,
                    "-i", src_path,
                    "-an", "-sn",
                    *scale_args,
                    "-r", "15",
                    *encode_args,
                    "-b:v", bitrate,
                    *split,
                    "-reset_timestamps", "1",
                    str(pattern),
                ])
            # Beside the queue's decodes (one chunk), the pass runs at a lower
            # CPU priority: on a machine with few cores the decode wins and
            # the proxies use what is left. The queue's tail runs normally.
            lower = chunks == 1 and chunkable and hasattr(os, "nice")
            procs = [subprocess.Popen(c, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=flags,
                                      preexec_fn=(lambda: os.nice(10)) if lower else None)
                     for c in cmds]
            errs = [p.communicate()[1] for p in procs]
            for p, err in zip(procs, errs):
                if p.returncode != 0:
                    stderr_tail = (err or b"").decode("utf-8", "replace")[-500:]
                    raise RuntimeError(f"ffmpeg segment exited {p.returncode}: {stderr_tail!r}")

            def seg_file(start_ms: int) -> Optional[Path]:
                f = seg_at.get(start_ms)
                return f if f is not None and f.exists() and f.stat().st_size > 0 else None

            produced: list[tuple[int, str]] = []
            for sid in parents:
                kids = sorted((by_id[c] + (c,) for c in children[sid] if c in by_id), key=lambda x: x[0])
                s, e = by_id[sid]
                files = [seg_file(ks) for ks, _ke, _c in kids]
                if (not kids or kids[0][0] != s or kids[-1][1] != e or None in files
                        or any(kids[i][1] != kids[i + 1][0] for i in range(len(kids) - 1))):
                    continue
                lst = tmp_path / f"parent_{sid}.txt"
                lst.write_text("".join(f"file '{f.as_posix()}'\n" for f in files))
                dst = proxies_dir / f"scene_{sid}.mp4"
                r = subprocess.run([_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-f", "concat",
                                    "-safe", "0", "-i", str(lst), "-c", "copy", "-y", str(dst)],
                                   capture_output=True, text=True, creationflags=flags)
                if r.returncode == 0:
                    produced.append((sid, str(dst)))
            for sid, s_ms, _e in leaves:
                src = seg_file(s_ms)
                if src is None:
                    log.warning("proxy segment missing for scene %s — will be made on its own", sid)
                    continue
                dst = proxies_dir / f"scene_{sid}.mp4"
                shutil.move(str(src), str(dst))
                produced.append((sid, str(dst)))
            # one transaction for the whole video (was a connection per scene)
            with schema.get_conn(self.primary_db) as conn:
                for sid, path in produced:
                    queries.set_scene_proxy(conn, sid, path)
            done.update(sid for sid, _ in produced)
            return len(produced)


def python_version_string() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


_state: Optional[AppState] = None


def get_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState()
    return _state
