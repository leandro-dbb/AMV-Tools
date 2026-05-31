"""GPU backend installer — replaces the install.bat / install.sh menu.

Streams `uv sync --extra X` stdout line-by-line through /ws/progress as
events of type `install`, and returns the final log + exit code synchronously.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from ..paths import PROJECT_ROOT, user_data_dir
from ..state import get_state

router = APIRouter()


class SetupRequest(BaseModel):
    backend: str


_VALID = {"cu130", "cu130-trt", "cu126", "dml", "xpu", "rocm", "cpu"}
_TORCH_BACKENDS = {"cu130", "cu130-trt", "cu126", "xpu", "rocm"}


def _failure_hint(log: str, backend: str) -> str:
    lowered = log.lower()
    if "git executable not found" in lowered or "git operation failed" in lowered:
        return (
            "This installer should not require Git. Rebuild the app with the "
            "current pyproject.toml/uv.lock, then retry the setup."
        )
    if "failed to download" in lowered or "connection" in lowered or "timed out" in lowered:
        return "Download failed. Check your connection, then retry the same backend."
    if "no space left" in lowered or "not enough space" in lowered:
        return "Not enough disk space for the Python/GPU wheels. Free space and retry."
    if "no solution found" in lowered or "resolution failed" in lowered:
        return "Dependency resolution failed. The packaged lockfile may be stale."
    if backend == "dml":
        return "DirectML can be picky. Retry once, then use CPU if your AMD/Intel setup still fails."
    return "Setup failed while installing the selected backend. Retry, or choose CPU to get the app open."


def _settings_for_backend(backend: str) -> dict:
    device_by_backend = {
        "cu130": "cuda",
        "cu130-trt": "cuda",
        "cu126": "cuda",
        "dml": "dml",
        "xpu": "xpu",
        "cpu": "cpu",
        "rocm": "auto",
    }
    return {
        "indexing": {"device": device_by_backend.get(backend, "auto")},
        "models": {"use_tensorrt": backend == "cu130-trt"},
    }


def _uv_exe() -> str:
    candidate = PROJECT_ROOT / ".uv" / ("uv.exe" if sys.platform == "win32" else "uv")
    if candidate.exists():
        return str(candidate)
    import shutil
    found = shutil.which("uv")
    if found:
        return found
    raise RuntimeError("uv executable not found")


def _install_flags(backend: str) -> list[str]:
    flags = ["--locked", "--extra", backend]
    if backend == "dml":
        flags += ["--python", "3.12"]
    if backend in _TORCH_BACKENDS:
        flags += ["--reinstall-package", "torch", "--reinstall-package", "torchvision"]
    return flags


def _validate_backend(backend: str) -> tuple[bool, str]:
    if backend == "cpu":
        return True, "CPU backend selected."

    checks = {
        "cu130": "cuda",
        "cu130-trt": "cuda",
        "cu126": "cuda",
        "dml": "dml",
        "xpu": "xpu",
        "rocm": "cuda",
    }
    expected = checks.get(backend)
    if not expected:
        return True, "No backend validation required."

    code = r"""
import sys

backend = sys.argv[1]
expected = sys.argv[2]

if expected == "cuda":
    import torch
    if not torch.version.cuda:
        raise SystemExit(f"torch {torch.__version__} was installed without CUDA support")
    if not torch.cuda.is_available():
        raise SystemExit(f"torch {torch.__version__} has CUDA {torch.version.cuda}, but no CUDA device is available")
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, device {torch.cuda.get_device_name(0)}")
elif expected == "xpu":
    import torch
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise SystemExit(f"torch {torch.__version__} was installed, but Intel XPU is not available")
    print(f"torch {torch.__version__}, Intel XPU available")
elif expected == "dml":
    import torch_directml
    if not torch_directml.is_available():
        raise SystemExit("torch-directml is installed, but DirectML is not available")
    print("DirectML available")
else:
    raise SystemExit(f"Unknown validation target for {backend}: {expected}")
"""
    kwargs = {
        "cwd": str(PROJECT_ROOT),
        "capture_output": True,
        "text": True,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[assignment]
    proc = subprocess.run(
        [sys.executable, "-c", code, backend, expected],
        **kwargs,  # type: ignore[arg-type]
    )
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, output or f"Backend validation exited with code {proc.returncode}"


@router.post("/api/setup/install")
async def install_backend(req: SetupRequest):
    if req.backend not in _VALID:
        return {"ok": False, "log": f"Unknown backend: {req.backend}"}

    cmd = [_uv_exe(), "sync", *_install_flags(req.backend)]
    state = get_state()
    state.publish({"type": "install", "line": f"$ {' '.join(cmd)}\n"})

    kwargs = dict(
        cwd=str(PROJECT_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[arg-type]

    proc = await asyncio.create_subprocess_exec(*cmd, **kwargs)
    out_chunks: list[str] = []
    assert proc.stdout
    async for raw in proc.stdout:
        line = raw.decode(errors="replace")
        out_chunks.append(line)
        state.publish({"type": "install", "line": line})

    rc = await proc.wait()
    log = "".join(out_chunks)
    if rc == 0:
        valid, validation_log = _validate_backend(req.backend)
        log = f"{log}\n\n[validation]\n{validation_log}\n"
        if not valid:
            hint = (
                f"{req.backend} installed, but the runtime validation failed. "
                "Retry the backend setup; if it keeps failing, use Clean reinstall from Settings."
            )
            state.publish({
                "type": "install",
                "line": f"\n[!] Backend validation failed\n{validation_log}\n{hint}\n",
                "done": True,
                "ok": False,
                "hint": hint,
            })
            return {"ok": False, "log": log[-12_000:], "hint": hint}
        state.update_settings(_settings_for_backend(req.backend))
        (user_data_dir() / ".setup_complete").write_text(req.backend)
        state.publish({
            "type": "install",
            "line": f"\n[OK] Installation complete.\n{validation_log}\n",
            "done": True,
            "ok": True,
        })
    else:
        hint = _failure_hint(log, req.backend)
        state.publish({
            "type": "install",
            "line": f"\n[!] uv sync exited with code {rc}\n{hint}\n",
            "done": True,
            "ok": False,
            "hint": hint,
        })
        return {"ok": False, "log": log[-12_000:], "hint": hint}
    return {"ok": True, "log": log[-12_000:], "hint": ""}


@router.get("/api/setup/status")
def setup_status():
    marker = user_data_dir() / ".setup_complete"
    return {"complete": marker.exists(), "backend": marker.read_text() if marker.exists() else None}
