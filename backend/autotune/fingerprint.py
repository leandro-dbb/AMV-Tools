"""Machine fingerprint: what makes a tuning result stale.

GPU / chip model, driver (or macOS version), torch, ffmpeg and app versions.
Any change gives another key, so the next launch measures again.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from typing import Any, Optional

SCHEMA = 1   # bump when the measurements change meaning


def _run(cmd: list[str]) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                             creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _nvidia() -> tuple[Optional[str], Optional[str]]:
    """(GPU names, driver) from nvidia-smi — no CUDA context in this process."""
    out = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if not out:
        return None, None
    rows = [[c.strip() for c in line.split(",")] for line in out.splitlines() if line.strip()]
    return " + ".join(r[0] for r in rows), rows[0][1] if len(rows[0]) > 1 else None


def _torch_version() -> Optional[str]:
    try:
        return importlib.metadata.version("torch")
    except Exception:
        return None


def _ffmpeg_version() -> Optional[str]:
    try:
        from ..indexing import cuts
        out = _run([cuts._ffmpeg_path(), "-hide_banner", "-version"])
        return out.splitlines()[0] if out else None
    except Exception:
        return None


def _app_version() -> str:
    try:
        from ..diagnostics import _pyproject_version
        return _pyproject_version()
    except Exception:
        return "unknown"


def collect() -> dict[str, Any]:
    if sys.platform == "darwin":
        chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.machine()
        mem = _run(["sysctl", "-n", "hw.memsize"])
        gpu = f"{chip} {round(int(mem) / 2**30)} GB" if mem and mem.isdigit() else chip
        driver = f"macOS {platform.mac_ver()[0]}"
    else:
        gpu, driver = _nvidia()
    return {
        "schema": SCHEMA,
        "os": sys.platform,
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "gpu": gpu,
        "driver": driver,
        "torch": _torch_version(),
        "ffmpeg": _ffmpeg_version(),
        "app": _app_version(),
    }


def key_of(fp: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]


@functools.lru_cache(maxsize=1)
def current() -> tuple[str, dict[str, Any]]:
    """(key, fingerprint) of this machine, computed once per process."""
    fp = collect()
    return key_of(fp), fp
