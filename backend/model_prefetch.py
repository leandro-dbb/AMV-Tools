"""First indexing: download the models up front, with visible progress.

SigLIP 2 and wd-tagger (~1.9 GB together) used to be fetched lazily by
``from_pretrained`` in the middle of the first indexing run. Nothing moved on
screen meanwhile: on the Windows CI VM, a slow Hugging Face day kept the
status bar on "Indexing… 0 %" for 16 minutes.

:func:`prefetch_indexing_models` runs at the start of an indexing job. For
each model missing from the Hugging Face cache it downloads the same files
the loader needs (``model_download.download_for_key``: retries, resume, size
and hash checks, clear "check your internet" hint) while a watcher measures
the repo's cache folder (``*.incomplete`` files included) every second and
publishes a regular ``indexing`` progress event — the status bar shows
"SigLIP 2 base (NaFlex) ↓ 512 / 1500 MB" and the percentage, no UI change.

It never raises: a failed prefetch only logs, the loader then retries and
reports the error exactly as before.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

_KEYS = ("wd_tagger", "siglip")


def _repo_cache_dir(repo_id: str) -> Path:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        root = Path(HF_HUB_CACHE)
    except Exception:
        root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return root / ("models--" + repo_id.replace("/", "--"))


def _dir_bytes(path: Path) -> int:
    total = 0
    blobs = path / "blobs"
    if not blobs.is_dir():
        return 0
    for entry in blobs.iterdir():
        try:
            total += entry.stat().st_size
        except OSError:
            pass
    return total


def _expected_mb(repo_id: str) -> Optional[float]:
    try:
        from .models.tiers import MODEL_INFO
        return MODEL_INFO.get(repo_id, {}).get("size_mb")
    except Exception:
        return None


def _label(repo_id: str, fallback: str) -> str:
    try:
        from .models.tiers import MODEL_INFO
        return MODEL_INFO.get(repo_id, {}).get("label") or fallback
    except Exception:
        return fallback


def progress_event(label: str, done_bytes: int, total_mb: Optional[float]) -> dict[str, Any]:
    done_mb = done_bytes / 2**20
    if total_mb:
        pct = max(0, min(99, int(done_mb * 100 / total_mb)))
        text = f"{label} ↓ {done_mb:.0f} / {total_mb:.0f} MB"
    else:
        pct = 0
        text = f"{label} ↓ {done_mb:.0f} MB"
    return {"type": "indexing", "message": "Downloading model", "video": text, "percent": pct}


def _real_wrappers(state) -> set[str]:
    """Keys whose model is the real downloadable one (tests and benches
    inject fakes: never reach the network for those)."""
    from .models import SigLIP2Model, WDTaggerModel
    keys = set()
    if isinstance(getattr(state, "_siglip", None), SigLIP2Model):
        keys.add("siglip")
    if isinstance(getattr(state, "_tagger", None), WDTaggerModel):
        keys.add("wd_tagger")
    return keys


def _missing_specs(state) -> list[dict[str, Any]]:
    from .diagnostics import _hf_cache_index, _model_specs
    real = _real_wrappers(state)
    if not real:
        return []
    cached = _hf_cache_index()
    out = []
    for spec in _model_specs(state):
        if spec["key"] not in _KEYS or spec["key"] not in real:
            continue
        entry = cached.get(spec["repo_id"])
        expected = _expected_mb(spec["repo_id"])
        have_mb = (entry or {}).get("size_bytes", 0) / 2**20
        # Present and (when the size is known) complete enough: skip.
        if entry and (not expected or have_mb >= 0.9 * expected):
            continue
        out.append(spec)
    return out


def prefetch_indexing_models(state, publish: Callable[[dict], None],
                             should_stop: Callable[[], bool] = lambda: False,
                             interval_s: float = 1.0) -> list[str]:
    """Download missing indexing models with progress. Returns the repo ids it
    fetched successfully. Never raises."""
    from .model_download import download_for_key

    try:
        missing = _missing_specs(state)
    except Exception as exc:
        log.warning("model prefetch: cache scan failed (%s); models load lazily", exc)
        return []
    fetched = []
    for spec in missing:
        if should_stop():
            break
        repo_id = spec["repo_id"]
        label = _label(repo_id, spec.get("label") or repo_id)
        total_mb = _expected_mb(repo_id)
        cache_dir = _repo_cache_dir(repo_id)
        done = threading.Event()

        def watch() -> None:
            last = -1
            while not done.wait(interval_s):
                size = _dir_bytes(cache_dir)
                if size != last:
                    last = size
                    publish(progress_event(label, size, total_mb))

        log.info("model prefetch: downloading %s (~%s MB)", repo_id, total_mb or "?")
        publish(progress_event(label, _dir_bytes(cache_dir), total_mb))
        watcher = threading.Thread(target=watch, name=f"prefetch-{spec['key']}", daemon=True)
        watcher.start()
        try:
            result = download_for_key(state, spec["key"])
        except Exception as exc:   # download_for_key already maps hub errors
            result = {"ok": False, "error": str(exc)}
        finally:
            done.set()
            watcher.join(timeout=5)
        if result.get("ok"):
            fetched.append(repo_id)
            publish(progress_event(label, _dir_bytes(cache_dir), total_mb) | {"percent": 100})
        else:
            log.warning("model prefetch failed for %s: %s %s", repo_id,
                        result.get("error"), result.get("hint", ""))
    return fetched
