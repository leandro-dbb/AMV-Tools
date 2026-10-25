"""Model tiers: which model each function uses, per platform and memory size.

A *tier* maps every model-backed function to one choice:

    {"embedding":   {"model": <HF repo>, "patches": <SigLIP NaFlex max patches>},
     "tagger":      <HF repo>,
     "upscale":     <finish registry id>,
     "interpolate": <finish registry id>,
     "depth":       <finish registry id>,
     "depth_input_size": <int, optional>}

Editing the tables
------------------
``NVIDIA_TIERS`` / ``MAC_TIERS`` / ``CPU_TIERS`` below are plain dicts: change
a value, restart the sidecar. Every model id used there must also have an
entry in ``MODEL_INFO`` (label, download size, license, kind) — the Settings
page shows it and ``tests/test_model_tiers.py`` checks that every license is
on the download allow-list. Finishing ids must exist in
``backend/finish/registry.py``.

Auto selection (``auto_tier``): NVIDIA by VRAM, Apple Silicon by TOTAL unified
memory (not Metal's recommended working set), CPU always "fast". The
thresholds sit between the nominal sizes so cards that report a little less
than their label (7.6 / 11.6 / 15.6 / 23.6 GB) land on their own tier.
``AMV_VRAM_LIMIT_GB`` (see ``device.simulated_vram_gb``) overrides the
detected size on both GPU platforms for the tier measurements.
"""
from __future__ import annotations

import copy
import functools
import logging
import os
import subprocess
import sys
from typing import Any, Optional

from ..config import DEFAULT_SIGLIP, DEFAULT_WD_TAGGER

log = logging.getLogger(__name__)

PLATFORMS = ("nvidia", "mac", "cpu")
FUNCTIONS = ("embedding", "tagger", "upscale", "interpolate", "depth")

# ── per-model catalogue ─────────────────────────────────────────────────────
# size_mb: download size (approximate for the HF repos). license: SPDX-ish id
# as written on the model card, lowercase. kind: the function it serves.
MODEL_INFO: dict[str, dict[str, Any]] = {
    # semantic-search embeddings (SigLIP 2 NaFlex)
    "google/siglip2-base-patch16-naflex": {
        "label": "SigLIP 2 base (NaFlex)", "size_mb": 1500, "license": "apache-2.0", "kind": "embedding"},
    "google/siglip2-so400m-patch16-naflex": {
        "label": "SigLIP 2 so400m (NaFlex)", "size_mb": 4550, "license": "apache-2.0", "kind": "embedding"},
    # anime tagger (wd-tagger v3)
    "SmilingWolf/wd-vit-tagger-v3": {
        "label": "wd-tagger v3 ViT", "size_mb": 380, "license": "apache-2.0", "kind": "tagger"},
    "SmilingWolf/wd-vit-large-tagger-v3": {
        "label": "wd-tagger v3 ViT-Large", "size_mb": 1260, "license": "apache-2.0", "kind": "tagger"},
    "SmilingWolf/wd-eva02-large-tagger-v3": {
        "label": "wd-tagger v3 EVA02-Large", "size_mb": 1260, "license": "apache-2.0", "kind": "tagger"},
    # finishing (backend/finish/registry.py ids)
    "realesr-animevideov3": {
        "label": "Real-ESRGAN AnimeVideo v3", "size_mb": 2.5, "license": "bsd-3-clause", "kind": "upscale"},
    "realesrgan-x4plus-anime-6b": {
        "label": "Real-ESRGAN x4plus anime 6B", "size_mb": 18, "license": "bsd-3-clause", "kind": "upscale"},
    "hfa2k-avc-compact": {
        "label": "2xHFA2kAVCCompact (Phhofm)", "size_mb": 4.8, "license": "cc-by-4.0", "kind": "upscale"},
    "rife-4.25": {
        "label": "RIFE 4.25", "size_mb": 25, "license": "mit", "kind": "interpolate"},
    "rife-4.26": {
        "label": "RIFE 4.26", "size_mb": 25, "license": "mit", "kind": "interpolate"},
    "rife-4.26-heavy": {
        "label": "RIFE 4.26 heavy", "size_mb": 25, "license": "mit", "kind": "interpolate"},
    "video-depth-anything-small": {
        "label": "Video Depth Anything Small", "size_mb": 116, "license": "apache-2.0", "kind": "depth"},
    "depth-anything-v2-small": {
        "label": "Depth Anything V2 Small", "size_mb": 99, "license": "apache-2.0", "kind": "depth"},
}

# ── tier tables ─────────────────────────────────────────────────────────────
# Filled from the chantier 4 measurements (docs/reports/chantier-4-modeles.md):
# SigLIP 2 so400m NaFlex (128 patches) is the search model from 8 GB up
# (P@10 0.80 vs 0.71, same embedding-stage time in the pipeline at the
# tier's batch size); wd-vit stays the tagger (no tagger knows more of the
# cast, the big ones are 4-14x slower); 2xHFA2kAVCCompact upscales closest to
# the real 1080p; RIFE 4.26 is 4.25 with better weights. "fast" keeps the
# lightest set (+ depth at 364 px, 2x faster); "max" takes the slower
# options that measured a little better.
NVIDIA_TIERS: dict[str, dict[str, Any]] = {
    "fast": {
        "embedding": {"model": "google/siglip2-base-patch16-naflex", "patches": 128},
        "tagger": "SmilingWolf/wd-vit-tagger-v3",
        "upscale": "realesr-animevideov3",
        "interpolate": "rife-4.26",
        "depth": "video-depth-anything-small",
        "depth_input_size": 364,
    },
    "8gb": {
        "embedding": {"model": "google/siglip2-so400m-patch16-naflex", "patches": 128},
        "tagger": "SmilingWolf/wd-vit-tagger-v3",
        "upscale": "hfa2k-avc-compact",
        "interpolate": "rife-4.26",
        "depth": "video-depth-anything-small",
        "depth_input_size": 364,
    },
    "12gb": {
        "embedding": {"model": "google/siglip2-so400m-patch16-naflex", "patches": 128},
        "tagger": "SmilingWolf/wd-vit-tagger-v3",
        "upscale": "hfa2k-avc-compact",
        "interpolate": "rife-4.26",
        "depth": "video-depth-anything-small",
        "depth_input_size": 364,
    },
    "16gb": {
        "embedding": {"model": "google/siglip2-so400m-patch16-naflex", "patches": 128},
        "tagger": "SmilingWolf/wd-vit-tagger-v3",
        "upscale": "hfa2k-avc-compact",
        "interpolate": "rife-4.26",
        "depth": "video-depth-anything-small",
    },
    "24gb": {
        "embedding": {"model": "google/siglip2-so400m-patch16-naflex", "patches": 128},
        "tagger": "SmilingWolf/wd-vit-tagger-v3",
        "upscale": "hfa2k-avc-compact",
        "interpolate": "rife-4.26",
        "depth": "video-depth-anything-small",
    },
    "max": {
        "embedding": {"model": "google/siglip2-so400m-patch16-naflex", "patches": 256},
        "tagger": "SmilingWolf/wd-eva02-large-tagger-v3",
        "upscale": "hfa2k-avc-compact",
        "interpolate": "rife-4.26-heavy",
        "depth": "video-depth-anything-small",
    },
}

# Apple Silicon: so400m runs 3.4x slower than base on an M1 Pro and made a
# 16 GB Mac swap during a full season (11-70k swap-outs, none with base):
# SigLIP base up to 16 GB, so400m from 24 GB (extrapolated). Depth: Video
# Depth Anything at 364 px from 16 GB (13.6 im/s vs 7.7 for DA V2 518 on Core
# ML, flicker 0.35 % vs 1.10 %, but an 8 GB footprint; 518 px thrashed the
# 16 GB Mac); 8 GB Macs and Fast keep Depth Anything V2 on Core ML. Max
# keeps wd-vit: eva02-large runs at 2.9 im/s (MPS) / 5.0 (Core ML) there.
MAC_TIERS: dict[str, dict[str, Any]] = {
    "fast": {
        "embedding": {"model": "google/siglip2-base-patch16-naflex", "patches": 128},
        "tagger": "SmilingWolf/wd-vit-tagger-v3",
        "upscale": "realesr-animevideov3",
        "interpolate": "rife-4.26",
        "depth": "depth-anything-v2-small",
        "depth_input_size": 364,
    },
    "mac8": {
        "embedding": {"model": "google/siglip2-base-patch16-naflex", "patches": 128},
        "tagger": "SmilingWolf/wd-vit-tagger-v3",
        "upscale": "hfa2k-avc-compact",
        "interpolate": "rife-4.26",
        "depth": "depth-anything-v2-small",
    },
    "mac16": {
        "embedding": {"model": "google/siglip2-base-patch16-naflex", "patches": 128},
        "tagger": "SmilingWolf/wd-vit-tagger-v3",
        "upscale": "hfa2k-avc-compact",
        "interpolate": "rife-4.26",
        "depth": "video-depth-anything-small",
        "depth_input_size": 364,
    },
    "mac24": {
        "embedding": {"model": "google/siglip2-so400m-patch16-naflex", "patches": 128},
        "tagger": "SmilingWolf/wd-vit-tagger-v3",
        "upscale": "hfa2k-avc-compact",
        "interpolate": "rife-4.26",
        "depth": "video-depth-anything-small",
        "depth_input_size": 364,
    },
    "mac32": {
        "embedding": {"model": "google/siglip2-so400m-patch16-naflex", "patches": 128},
        "tagger": "SmilingWolf/wd-vit-tagger-v3",
        "upscale": "hfa2k-avc-compact",
        "interpolate": "rife-4.26",
        "depth": "video-depth-anything-small",
        "depth_input_size": 364,
    },
    "max": {
        "embedding": {"model": "google/siglip2-so400m-patch16-naflex", "patches": 256},
        "tagger": "SmilingWolf/wd-vit-tagger-v3",
        "upscale": "hfa2k-avc-compact",
        "interpolate": "rife-4.26-heavy",
        "depth": "video-depth-anything-small",
        "depth_input_size": 364,
    },
}

# CPU / DirectML / Intel XPU / unknown: the previous defaults, unchanged.
CPU_TIERS: dict[str, dict[str, Any]] = {
    "fast": {
        "embedding": {"model": "google/siglip2-base-patch16-naflex", "patches": 128},
        "tagger": "SmilingWolf/wd-vit-tagger-v3",
        "upscale": "realesr-animevideov3",
        "interpolate": "rife-4.25",
        "depth": "depth-anything-v2-small",
    },
}

# Nominal memory of each tier (GB) shown in Settings; None = never picked
# automatically ("fast" = defaults, "max" = best quality whatever the size).
TIER_MIN_GB: dict[str, dict[str, Optional[float]]] = {
    "nvidia": {"fast": None, "8gb": 8.0, "12gb": 12.0, "16gb": 16.0, "24gb": 24.0, "max": None},
    "mac": {"fast": None, "mac8": 8.0, "mac16": 16.0, "mac24": 24.0, "mac32": 32.0, "max": None},
    "cpu": {"fast": None},
}

# Auto selection: first (upper bound GB, tier) whose bound is above the
# detected size, else the last tier id.
AUTO_THRESHOLDS: dict[str, list[tuple[float, str]]] = {
    "nvidia": [(10.0, "8gb"), (14.0, "12gb"), (20.0, "16gb")],
    "mac": [(12.0, "mac8"), (20.0, "mac16"), (28.0, "mac24")],
}
AUTO_TOP: dict[str, str] = {"nvidia": "24gb", "mac": "mac32"}

_TABLES = {"nvidia": NVIDIA_TIERS, "mac": MAC_TIERS, "cpu": CPU_TIERS}

# What a choice falls back to when a tier omits a function.
DEFAULT_CHOICE: dict[str, Any] = {
    "embedding": {"model": DEFAULT_SIGLIP, "patches": 128},
    "tagger": DEFAULT_WD_TAGGER,
    "upscale": "realesr-animevideov3",
    "interpolate": "rife-4.25",
    "depth": "depth-anything-v2-small",
}


# ── platform + memory ───────────────────────────────────────────────────────
def detect_platform(device) -> str:
    """"nvidia" for torch CUDA (ROCm reports as CUDA too), "mac" for MPS,
    "cpu" for everything else (CPU, DirectML, XPU, unknown)."""
    backend = getattr(device, "backend", None)
    if backend == "cuda":
        return "nvidia"
    if backend == "mps":
        return "mac"
    return "cpu"


def _simulated_gb() -> Optional[float]:
    from .device import simulated_vram_gb
    return simulated_vram_gb()


@functools.lru_cache(maxsize=1)
def mac_total_memory_gb() -> Optional[float]:
    """Total unified memory of this Mac in GB (``sysctl hw.memsize``, then
    ``os.sysconf``). None when neither works."""
    total = None
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            total = int(out.stdout.strip())
        except Exception:
            total = None
    if not total:
        try:
            total = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
        except (ValueError, OSError, AttributeError):
            total = None
    return round(total / 2**30, 1) if total else None


def detected_memory_gb(platform: str, device) -> Optional[float]:
    """Memory size the auto tier is chosen from: VRAM on NVIDIA, total
    unified memory on a Mac, None on CPU. ``AMV_VRAM_LIMIT_GB`` wins on both
    GPU platforms."""
    if platform == "cpu":
        return None
    sim = _simulated_gb()
    if sim:
        return float(sim)
    if platform == "nvidia":
        v = getattr(device, "vram_gb", None)
        return float(v) if v else None
    return mac_total_memory_gb()


def auto_tier(platform: str, memory_gb: Optional[float]) -> str:
    if platform not in AUTO_THRESHOLDS:
        return "fast"
    if not memory_gb or memory_gb <= 0:
        # unknown size: the defaults, as before tiers existed
        return "fast"
    for bound, tier in AUTO_THRESHOLDS[platform]:
        if memory_gb < bound:
            return tier
    return AUTO_TOP[platform]


def tiers_for(platform: str) -> dict[str, dict[str, Any]]:
    return _TABLES.get(platform, CPU_TIERS)


# ── resolution ──────────────────────────────────────────────────────────────
def _normalize(choice: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(DEFAULT_CHOICE)
    for k, v in (choice or {}).items():
        if k == "embedding" and isinstance(v, dict):
            out["embedding"] = {**out["embedding"], **v}
        else:
            out[k] = copy.deepcopy(v)
    out["embedding"]["patches"] = int(out["embedding"].get("patches") or 128)
    return out


def custom_models(settings_models: dict) -> dict[str, Any]:
    """"custom": the legacy per-model settings (Settings > Models), finishing
    on the defaults."""
    m = settings_models or {}
    return _normalize({
        "embedding": {"model": m.get("siglip_variant") or DEFAULT_SIGLIP,
                      "patches": int(m.get("siglip_max_num_patches") or 128)},
        "tagger": m.get("wd_tagger_variant") or DEFAULT_WD_TAGGER,
    })


def resolve(settings_models: dict, device) -> dict[str, Any]:
    """Effective model choice for this machine.

    ``settings_models`` is ``settings["models"]``; its ``tier`` is "auto"
    (default), "custom" or a tier id. A tier id of another platform (a
    settings file moved from an NVIDIA PC to a Mac) falls back to auto."""
    platform = detect_platform(device)
    gb = detected_memory_gb(platform, device)
    auto = auto_tier(platform, gb)
    selected = str((settings_models or {}).get("tier") or "auto")
    table = tiers_for(platform)
    if selected == "custom":
        effective = "custom"
        models = custom_models(settings_models)
    else:
        effective = selected if selected in table else auto
        if effective not in table:
            effective = next(iter(table))
        models = _normalize(table[effective])
    return {
        "platform": platform,
        "detected_gb": gb,
        "auto_tier": auto,
        "selected": selected,
        "effective_tier": effective,
        "models": models,
    }


def finish_default(kind: str, settings_models: dict, device) -> Optional[str]:
    """Tier default finishing model id for ``kind`` (upscale / interpolate /
    depth), None for other kinds."""
    if kind not in ("upscale", "interpolate", "depth"):
        return None
    return resolve(settings_models, device)["models"].get(kind)


def model_info(model_id: str) -> dict[str, Any]:
    """Catalogue entry for ``model_id`` (finishing ids missing from
    ``MODEL_INFO`` are described from the registry)."""
    info = MODEL_INFO.get(model_id)
    if info is None:
        try:
            from ..finish import registry
            spec = registry.MODELS.get(model_id)
        except Exception:
            spec = None
        if spec is not None:
            info = {"label": spec.label, "size_mb": round(spec.size / 1e6, 1),
                    "license": spec.license.lower(), "kind": spec.kind}
        else:
            info = {"label": model_id, "size_mb": None, "license": None, "kind": None}
    return {"id": model_id, **info}


def describe_tiers(platform: str) -> list[dict[str, Any]]:
    """The platform's tiers for the API, in table order (fast .. max)."""
    out = []
    mins = TIER_MIN_GB.get(platform, {})
    for tid, choice in tiers_for(platform).items():
        c = _normalize(choice)
        models: dict[str, Any] = {}
        for fn in FUNCTIONS:
            if fn == "embedding":
                models[fn] = {**model_info(c["embedding"]["model"]), "patches": c["embedding"]["patches"]}
            else:
                models[fn] = model_info(c[fn])
        if c.get("depth_input_size"):
            models["depth"]["input_size"] = int(c["depth_input_size"])
        out.append({"id": tid, "min_gb": mins.get(tid), "models": models})
    return out
