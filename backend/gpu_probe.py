"""Which GPU backend should the first-run setup suggest?

Runs BEFORE any GPU wheel is installed (the base venv has CPU torch only),
so it cannot ask torch. It asks the OS instead:

* NVIDIA: ``nvidia-smi`` (ships with every driver, on PATH on Windows and
  Linux). CUDA 13 wheels need driver >= 580 and a Turing or newer card
  (compute capability >= 7.5); anything older gets the CUDA 12.6 wheels.
* Windows without NVIDIA: the display adapters from WMI. A real AMD/Intel
  GPU gets DirectML; only the "Microsoft Basic Display Adapter" (VMs,
  missing driver) means CPU.
* macOS on Apple Silicon: MPS.

Every probe is bounded (timeout) and never raises: a machine where nothing
can be detected simply gets the CPU suggestion, the user can still pick any
backend by hand.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from typing import Any

_TIMEOUT_S = 8
# Software / virtual adapters that cannot run DirectML compute usefully.
_VIRTUAL_ADAPTERS = ("microsoft basic", "microsoft remote display", "hyper-v", "vmware", "virtualbox",
                     "parsec", "citrix", "remote desktop", "idd", "dummy")


def _run(cmd: list[str]) -> str | None:
    kwargs: dict[str, Any] = {"capture_output": True, "text": True, "timeout": _TIMEOUT_S,
                              "encoding": "utf-8", "errors": "replace"}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    try:
        proc = subprocess.run(cmd, **kwargs)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _nvidia_gpus() -> list[dict[str, Any]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    out = _run([exe, "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader,nounits"])
    if out is None:
        # Drivers older than ~510 don't know compute_cap.
        out = _run([exe, "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader,nounits"])
    gpus = []
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        gpus.append({
            "vendor": "nvidia",
            "name": parts[0],
            "driver": parts[1],
            "vram_gb": _float(parts[2], scale=1 / 1024),
            "compute_cap": _float(parts[3]) if len(parts) > 3 else None,
        })
    return gpus


def _float(text: str, scale: float = 1.0) -> float | None:
    try:
        return round(float(text) * scale, 1)
    except (TypeError, ValueError):
        return None


def _windows_adapters() -> list[str]:
    out = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
                "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"])
    return [line.strip() for line in (out or "").splitlines() if line.strip()]


def _vendor(name: str) -> str:
    low = name.lower()
    if "nvidia" in low or "geforce" in low or "quadro" in low or "rtx" in low:
        return "nvidia"
    if "amd" in low or "radeon" in low or "ati " in low:
        return "amd"
    if "intel" in low or " arc" in low:
        return "intel"
    if any(v in low for v in _VIRTUAL_ADAPTERS):
        return "virtual"
    return "other"


def _driver_major(driver: str | None) -> int:
    try:
        return int(str(driver).split(".")[0])
    except (TypeError, ValueError):
        return 0


def recommend(nvidia: list[dict[str, Any]], adapters: list[str], system: str, machine: str) -> tuple[str, str]:
    """(backend id, short English reason). Pure function, unit-tested."""
    if system == "Darwin":
        if machine.lower() in ("arm64", "aarch64"):
            return "mps", "Apple Silicon"
        return "cpu", "Intel Mac (no supported GPU backend)"
    if nvidia:
        gpu = nvidia[0]
        cap = gpu.get("compute_cap")
        major = _driver_major(gpu.get("driver"))
        if cap is not None and cap < 7.5:
            return "cu126", f"{gpu['name']} predates RTX 20xx (CUDA 13 needs compute 7.5+)"
        if major and major < 580:
            return "cu126", f"NVIDIA driver {gpu.get('driver')} is older than 580 (CUDA 13)"
        return "cu130", gpu["name"]
    if system == "Windows":
        real = [a for a in adapters if _vendor(a) in ("amd", "intel", "nvidia", "other")]
        if real:
            return "dml", real[0]
        return "cpu", "no GPU detected"
    if system == "Linux":
        amd = [a for a in adapters if _vendor(a) == "amd"]
        if amd:
            return "rocm", amd[0]
    return "cpu", "no GPU detected"


def probe() -> dict[str, Any]:
    system = platform.system()
    machine = platform.machine()
    nvidia = _nvidia_gpus() if system in ("Windows", "Linux") else []
    adapters = _windows_adapters() if system == "Windows" else []
    backend, reason = recommend(nvidia, adapters, system, machine)
    return {
        "system": system,
        "machine": machine,
        "nvidia": nvidia,
        "adapters": [{"name": a, "vendor": _vendor(a)} for a in adapters],
        "recommended": backend,
        "reason": reason,
    }
