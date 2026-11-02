"""Prune the app's private uv cache once the venv is installed.

The installed app keeps uv's cache in its data folder (``UV_CACHE_DIR``,
set by Electron). After the first launch and the GPU engine install it held
7.7 GB on Windows (PyTorch CUDA, cuDNN, cuBLAS…, plus the CPU PyTorch that
the CUDA install replaced), never cleaned. uv links the venv's files to the
cache (hard links on Linux and Windows), so only what the venv no longer
uses really takes room — but that is the old PyTorch, and nothing in the
cache is needed again: a later dependency update downloads only the
packages whose version changed, whatever the cache holds.

``uv cache prune --ci`` drops the pre-built wheels and keeps the wheels uv
built from source (slow to rebuild). ``--force``: the ``uv run`` that
started this sidecar holds the cache's lock for as long as the app runs, so
without it the prune waits forever; nothing else uses this private cache
while we prune (installs run the prune once they are done).

Runs after a successful engine install (``/api/setup/install``) and, at
start-up, once per ``uv.lock`` after an app update (Electron re-syncs the
venv then). Off unless Electron asks for it (``AMV_UV_CACHE_PRUNE=1``,
packaged app only): a developer's own cache is left alone.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

STAMP = ".uv-cache-pruned"
_lock = threading.Lock()


def enabled() -> bool:
    return os.environ.get("AMV_UV_CACHE_PRUNE") == "1" and bool(os.environ.get("UV_CACHE_DIR"))


def _lock_sha() -> Optional[str]:
    from .paths import PROJECT_ROOT
    try:
        return hashlib.sha1((PROJECT_ROOT / "uv.lock").read_bytes()).hexdigest()
    except OSError:
        return None


def _stamp_path() -> Path:
    from .paths import user_data_dir
    return user_data_dir() / STAMP


def _dir_bytes(root: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
    return total


def prune(reason: str) -> dict:
    """Run the prune now (blocking). Never raises; returns what happened:
    ``ok``, ``cache_before`` / ``cache_after`` (bytes in the cache folder,
    hard links counted), ``disk_freed`` (the volume's free space gained)."""
    cache_dir = os.environ.get("UV_CACHE_DIR")
    if not cache_dir:
        return {"ok": False, "reason": "no UV_CACHE_DIR"}
    from .api.setup import _uv_exe
    try:
        uv = _uv_exe()
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}
    with _lock:
        root = Path(cache_dir)
        before = _dir_bytes(root) if root.is_dir() else 0
        try:
            free_before = shutil.disk_usage(root if root.is_dir() else root.parent).free
        except OSError:
            free_before = None
        kwargs: dict = {"capture_output": True, "text": True, "encoding": "utf-8", "errors": "replace",
                        "timeout": 1800}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
        elif hasattr(os, "nice"):
            kwargs["preexec_fn"] = lambda: os.nice(10)
        try:
            proc = subprocess.run([uv, "cache", "prune", "--ci", "--force"], **kwargs)
            ok, out = proc.returncode == 0, (proc.stdout + proc.stderr).strip()
        except Exception as exc:
            ok, out = False, str(exc)
        after = _dir_bytes(root) if root.is_dir() else 0
        try:
            freed = shutil.disk_usage(root if root.is_dir() else root.parent).free - free_before \
                if free_before is not None else None
        except OSError:
            freed = None
    res = {"ok": ok, "reason": reason, "cache_before": before, "cache_after": after,
           "disk_freed": freed, "output": out[-2000:]}
    log.info("uv cache prune (%s): ok=%s, cache %.2f -> %.2f GB, disk freed %s GB: %s", reason, ok,
             before / 1e9, after / 1e9, f"{freed / 1e9:.2f}" if freed is not None else "?",
             out.splitlines()[-1] if out else "")
    if ok:
        sha = _lock_sha()
        if sha:
            try:
                _stamp_path().write_text(sha)
            except OSError:
                pass
    return res


def wait_idle() -> None:
    """Block until a running prune is over (an install must not sync while
    the cache is being emptied)."""
    with _lock:
        pass


def prune_in_background(reason: str) -> Optional[threading.Thread]:
    if not enabled():
        return None
    t = threading.Thread(target=prune, args=(reason,), daemon=True, name="uv-cache-prune")
    t.start()
    return t


def prune_after_update() -> Optional[threading.Thread]:
    """Start-up: prune once per ``uv.lock`` the venv was synced to, if the
    engine is installed (the first launch's base sync is pruned with the
    engine install)."""
    if not enabled():
        return None
    from .paths import user_data_dir
    if not (user_data_dir() / ".setup_complete").exists():
        return None
    sha = _lock_sha()
    try:
        done = _stamp_path().read_text().strip()
    except OSError:
        done = ""
    if sha is None or done == sha:
        return None
    return prune_in_background("app update")
