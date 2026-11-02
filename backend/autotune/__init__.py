"""Automatic tuning of the performance settings for this machine.

Several settings were fixed on the two test machines: 8 VideoToolbox sessions
for the chunked decode (M1 Pro), 2 single-pass NVDEC decodes in flight (RTX
3090, one NVDEC unit), the Core ML wd-tagger on Apple Silicon, the batch
size of the tagger/SigLIP passes per memory tier. On first launch — and
whenever the machine fingerprint changes (GPU/chip, driver, torch, ffmpeg,
app version, see ``fingerprint``) — a background micro-bench (< 1 min, only
while nothing is indexing) measures them here and stores the choice in
``autotune.json`` in the data folder.

Readers call ``value(name, default)``: the measured value, or ``default``
(the former constant) when the bench has not run, failed or skipped that
setting. Environment variables keep priority at each call site.

Settings: ``parallel_decodes`` (NVIDIA), ``chunk_workers`` (macOS),
``tagger_engine`` (macOS: "coreml" | "mps"), ``batch_size`` (CUDA / MPS),
``cutnet_device`` (macOS: "mps" | "cpu") and ``cutnet`` ("auto" | "manual":
whether the cut network runs by itself, see ``bench.measure_cutnet``).
``AMV_AUTOTUNE=0`` disables the background run.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from . import cache, fingerprint, yielding

log = logging.getLogger(__name__)

START_DELAY_S = 20.0        # let the sidecar finish its own warm-up first
IDLE_POLL_S = 30.0
USER_POLL_S = 1.0           # waiting for the user to be idle (yielding.QUIET_S)

_entry_lock = threading.Lock()
_entry: dict[str, Any] = {}     # {"key": ..., "entry": ...} memo of the cache read
_status: dict[str, Any] = {"state": "idle", "step": None, "error": None}
_thread: Optional[threading.Thread] = None
_cancel = threading.Event()


def _current_entry() -> Optional[dict]:
    key, _ = fingerprint.current()
    with _entry_lock:
        if _entry.get("key") != key:
            _entry["key"] = key
            _entry["entry"] = cache.load(key)
        return _entry["entry"]


def _forget() -> None:
    with _entry_lock:
        _entry.clear()


def value(name: str, default):
    """The tuned ``name`` for this machine, else ``default``."""
    try:
        rec = ((_current_entry() or {}).get("settings") or {}).get(name)
    except Exception as exc:          # never let tuning break a caller
        log.debug("autotune read failed: %s", exc)
        return default
    if isinstance(rec, dict) and rec.get("value") is not None:
        return rec["value"]
    return default


def needs_run(entry: Optional[dict]) -> bool:
    """No result for this fingerprint, one measured on the synthetic clip
    while the library now has a video, or one taken before the cut network
    was downloaded."""
    if entry is None:
        return True
    if "TransNetV2" in str((entry.get("skipped") or {}).get("cutnet", "")):
        # the cut network was downloaded after the last run: measure what it costs
        from ..indexing import cutnet
        if cutnet.weights_path() is not None:
            return True
    if entry.get("source") == "synthetic":
        from . import bench
        return bench.library_video() is not None
    return False


def _busy() -> bool:
    """Indexing or a re-embed job is running (they would skew each other)."""
    try:
        from ..indexing import reembed
        from ..state import get_state
        return bool(getattr(get_state(), "_indexing_active", False)) or reembed.is_running()
    except Exception:
        return False


def _platform() -> str:
    from ..indexing import cuts, onepass
    if onepass.available():
        return "nvidia"
    prof = cuts._ffmpeg_hw_profile()
    if prof is not None and prof.get("name") == "videotoolbox":
        return "mac"
    return "cpu"


def plan(platform: str) -> list[str]:
    """Settings measured on ``platform`` (in this order)."""
    return {
        "nvidia": ["parallel_decodes", "tensorrt", "batch_size", "cutnet"],
        "mac": ["chunk_workers", "tagger_engine", "batch_size", "cutnet_device", "cutnet"],
    }.get(platform, ["cutnet"])


def run(measure: Optional[dict[str, Callable]] = None, cancelled: Optional[Callable[[], bool]] = None,
        platform: Optional[str] = None, source: Optional[str] = None,
        progress: Optional[dict] = None) -> dict:
    """Measure, choose, store; returns the cache entry. ``measure``: setting
    -> callable(ctx) override (tests). A failed measurement keeps that
    setting's default; the others still run. ``progress``: the results of
    an interrupted run of the same fingerprint, filled in as settings are
    measured — a new run measures only what is missing."""
    from . import bench
    cancelled = cancelled or (lambda: False)
    key, fp = fingerprint.current()
    platform = platform or _platform()
    t0 = time.perf_counter()
    progress = {} if progress is None else progress
    if progress.get("key") != key:
        progress.clear()
        progress["key"] = key
    settings: dict[str, Any] = progress.setdefault("settings", {})
    skipped: dict[str, str] = progress.setdefault("skipped", {})
    errors: dict[str, str] = progress.setdefault("errors", {})
    with bench.workdir() as tmp:
        from pathlib import Path
        ctx: dict[str, Any] = {"workdir": Path(tmp), "source": source, "cancelled": cancelled,
                               "frames": None}
        if source is None and measure is None:
            ctx["source"] = bench.library_video()

        def frames(n: int = 256):
            if ctx["frames"] is None:
                ctx["frames"] = bench.sample_frames(ctx["workdir"], ctx["source"], n)
            return ctx["frames"]

        ctx["get_frames"] = frames
        default_measure = {
            "parallel_decodes": lambda c: bench.measure_nvdec(c["workdir"], c["source"], c["cancelled"]),
            "chunk_workers": lambda c: bench.measure_videotoolbox(c["workdir"], c["source"], c["cancelled"]),
            "tagger_engine": lambda c: bench.measure_tagger_engine(c["get_frames"](), c["cancelled"]),
            "batch_size": lambda c: bench.measure_batch(c["get_frames"](), c["cancelled"]),
            "tensorrt": lambda c: bench.tensorrt_note(),
            "cutnet_device": lambda c: bench.measure_cutnet_device(c["cancelled"]),
            "cutnet": lambda c: bench.measure_cutnet(c["workdir"], c["source"], c["cancelled"]),
        }
        for name in plan(platform) if measure is None else list(measure):
            if name in settings or name in skipped or name in errors:
                continue                  # measured before the interruption
            if cancelled():
                raise bench.Skip("cancelled")
            _status["step"] = name
            fn = (measure or default_measure)[name]
            try:
                rec = fn(ctx)
                if rec.get("value") is None:
                    raise RuntimeError("no candidate could be measured")
                settings[name] = rec
                log.info("autotune %s = %s (%s)", name, rec["value"], rec.get("measures"))
            except bench.Skip as exc:
                if str(exc) == "cancelled":
                    raise
                skipped[name] = str(exc)
            except Exception as exc:
                if cancelled():
                    # stopped mid-measure (a decode cut short): not a failure,
                    # measured again when the run resumes
                    raise bench.Skip("cancelled") from exc
                log.warning("autotune %s failed (%s); keeping the default", name, exc)
                errors[name] = repr(exc)[:300]
    if cancelled():
        raise bench.Skip("cancelled")
    entry = {
        "fingerprint": fp,
        "platform": platform,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(time.perf_counter() - t0, 1),
        "source": "library" if ctx["source"] else "synthetic",
        "settings": dict(settings),
        "skipped": dict(skipped),
        "errors": dict(errors),
    }
    cache.store(key, entry)
    progress.clear()
    _forget()
    return entry


def _stale_torch() -> Optional[str]:
    """Why this process cannot measure for the installed torch, if it can't:
    the engine install replaces torch (CPU -> CUDA) under a sidecar that
    already imported the old one; measured there (Windows VM, after a real
    install), the cut network and the batches ran on the CPU and the stored
    choices — cut network "manual" — were filed under the new torch's
    fingerprint, for good."""
    import sys
    mod = sys.modules.get("torch")
    if mod is None:
        return None
    try:
        import importlib.metadata
        installed = importlib.metadata.version("torch")
    except Exception:
        return None
    loaded = str(getattr(mod, "__version__", ""))
    if loaded and installed and loaded != installed:
        return f"torch {loaded} loaded, {installed} installed: measured at the next start"
    return None


def _must_yield() -> bool:
    """Indexing / a re-embed runs, or the user is using the app."""
    return _busy() or yielding.active()


def _background(force: bool, delay: float) -> None:
    from . import bench
    yielding.lower_this_thread()
    try:
        if delay:
            time.sleep(delay)
        from ..paths import user_data_dir
        if not force and not (user_data_dir() / ".setup_complete").exists():
            # First run: the benches import torch, and the onboarding is about
            # to replace it (CPU -> CUDA build). On Windows a loaded torch keeps
            # its DLLs locked against that reinstall, and a torch that failed
            # to load stays half-imported in this process. The setup starts
            # the autotune once the chosen backend is installed.
            _status.update(state="waiting_setup", step=None)
            return
        if not force and not needs_run(_current_entry()):
            _status.update(state="done", step=None)
            return
        stale = _stale_torch()
        if stale:
            _status.update(state="postponed", step=None, error=stale)
            log.info("autotune postponed: %s", stale)
            return
        progress: dict[str, Any] = {}
        while True:
            # Indexing, a re-embed or the user (a search, a preview, an
            # export…) comes first: wait, and stop a running bench as soon as
            # one of them starts; it resumes where it stopped (``progress``).
            while _must_yield():
                _status.update(state="waiting", step=None)
                if _cancel.wait(IDLE_POLL_S if _busy() else USER_POLL_S):
                    return
            _status.update(state="running", step=None, error=None, started_at=time.time())
            try:
                run(cancelled=lambda: _cancel.is_set() or _must_yield(), progress=progress)
                _status.update(state="done", step=None)
                return
            except bench.Skip:
                if _cancel.is_set():
                    raise
                log.info("autotune paused (%s); resumes when the app is idle",
                         "indexing" if _busy() else "user request")
    except bench.Skip:
        _status.update(state="postponed", step=None)
    except Exception as exc:
        log.warning("autotune failed (%s); the default settings stay", exc)
        _status.update(state="failed", step=None, error=repr(exc)[:300])


def start_background(force: bool = False, delay: float = START_DELAY_S) -> bool:
    """Start the bench thread unless one is running. False when disabled
    (``AMV_AUTOTUNE=0``) or already running."""
    global _thread
    if not force and (os.environ.get("AMV_AUTOTUNE", "1") == "0" or "PYTEST_CURRENT_TEST" in os.environ):
        _status.update(state="disabled")
        return False
    if _thread is not None and _thread.is_alive():
        return False
    _cancel.clear()
    _thread = threading.Thread(target=_background, args=(force, delay), daemon=True, name="autotune")
    _thread.start()
    return True


def status() -> dict[str, Any]:
    key, fp = fingerprint.current()
    entry = _current_entry()
    return {
        **{k: v for k, v in _status.items() if k != "started_at"},
        "running": _thread is not None and _thread.is_alive() and _status.get("state") == "running",
        "key": key,
        "fingerprint": fp,
        "entry": entry,
        "cache_path": str(cache.path()),
    }
