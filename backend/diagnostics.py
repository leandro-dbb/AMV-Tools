"""Runtime diagnostics + rotating backend logs.

The Settings → Diagnostic panel and the bug-report issue template both consume
``build_diagnostics()``. Everything here is best-effort: a missing optional
dependency must never prevent the report from being produced.
"""
from __future__ import annotations

import importlib.metadata
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

from .config import DEFAULT_SIGLIP, DEFAULT_WD_TAGGER
from .paths import PROJECT_ROOT, user_data_dir

_LOG_MAX_BYTES = 2_000_000
_LOG_BACKUPS = 3
_RING_MAX = 200


# ── error ring buffer ───────────────────────────────────────────────────────
class _ErrorRing(logging.Handler):
    def __init__(self, maxlen: int = _RING_MAX):
        super().__init__(level=logging.WARNING)
        self._records: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        entry = {
            "time": self._format_time(record),
            "level": record.levelname,
            "logger": record.name,
            "message": message[:4000],
        }
        if record.exc_info:
            try:
                entry["traceback"] = logging.Formatter().formatException(record.exc_info)[:8000]
            except Exception:
                pass
        with self._lock:
            self._records.append(entry)

    @staticmethod
    def _format_time(record: logging.LogRecord) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))

    def entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records)


_ring = _ErrorRing()
_ring.setFormatter(logging.Formatter("%(message)s"))
_file_handler: Optional[RotatingFileHandler] = None


def logs_dir() -> Path:
    path = user_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_file_path() -> Path:
    return logs_dir() / "backend.log"


def setup_logging() -> Path:
    """Attach a rotating file handler (+ the in-memory error ring) to root.

    Safe to call more than once; handlers are reused rather than duplicated.
    """
    global _file_handler
    root = logging.getLogger()
    if _ring not in root.handlers:
        root.addHandler(_ring)

    target = log_file_path()
    if _file_handler is not None and Path(getattr(_file_handler, "baseFilename", "")) == target:
        return target
    if _file_handler is not None:
        try:
            root.removeHandler(_file_handler)
            _file_handler.close()
        except Exception:
            pass
        _file_handler = None

    try:
        handler = RotatingFileHandler(
            target, maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUPS, encoding="utf-8",
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        ))
        root.addHandler(handler)
        root.setLevel(min(root.level or logging.INFO, logging.INFO))
        _file_handler = handler
    except Exception as exc:  # pragma: no cover - permission/disk failures
        logging.getLogger(__name__).warning("could not open log file %s: %s", target, exc)
    return target


def recent_errors(limit: int = 20) -> list[dict[str, Any]]:
    return list(reversed(_ring.entries()))[:limit]


# ── version helpers ─────────────────────────────────────────────────────────
def _pyproject_version() -> str:
    try:
        import tomllib

        data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
        return str(data.get("project", {}).get("version", "0.0.0"))
    except Exception:
        return "0.0.0"


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "version": _package_version("torch"),
        "cuda_available": None,
        "mps_available": None,
        "cuda_version": None,
    }
    if not info["version"]:
        return info
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        mps = getattr(torch.backends, "mps", None)
        info["mps_available"] = bool(mps and mps.is_available())
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _ffmpeg_info() -> dict[str, Any]:
    try:
        from .indexing.cuts import _ffmpeg_path

        path = _ffmpeg_path()
    except Exception as exc:
        return {"path": None, "version": None, "error": str(exc)}
    try:
        proc = subprocess.run(
            [path, "-version"], capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        first = (proc.stdout or proc.stderr or "").splitlines()
        version = first[0] if first else None
        return {"path": path, "version": version}
    except Exception as exc:
        return {"path": path, "version": None, "error": str(exc)}


# ── device / fallback ───────────────────────────────────────────────────────
def _fallback_reason(requested: str, active: str, torch_info: dict[str, Any]) -> Optional[str]:
    if active == requested:
        return None
    if requested == "cpu":
        return None
    if requested == "auto":
        if active == "cpu":
            if torch_info.get("version") is None:
                return "No GPU runtime detected (torch is not installed); running on CPU."
            if torch_info.get("cuda_available") is False and torch_info.get("mps_available") is False:
                return "No supported GPU detected (CUDA/MPS/DirectML/XPU unavailable); running on CPU."
            return "No supported GPU detected; running on CPU."
        return None
    labels = {
        "cuda": "CUDA",
        "mps": "Apple MPS",
        "dml": "DirectML",
        "xpu": "Intel XPU",
    }
    label = labels.get(requested, requested)
    if requested == "cuda" and torch_info.get("cuda_available") is False:
        if torch_info.get("cuda_version") is None:
            return "CUDA was requested, but this PyTorch build has no CUDA support; fell back to CPU."
        return "CUDA was requested, but no CUDA device is available; fell back to CPU."
    return f"{label} was requested, but it is unavailable; fell back to CPU."


# Public alias for the lightweight status endpoint (no torch probe).
fallback_reason = _fallback_reason


def _device_info(state) -> dict[str, Any]:
    requested = str(state.settings.get("indexing", {}).get("device", "auto"))
    torch_info = _torch_info()
    try:
        device = state.device
        active = getattr(device, "backend", "unknown")
        name = getattr(device, "name", None)
        vram = getattr(device, "vram_gb", None)
        error = None
    except Exception as exc:
        active, name, vram = "unknown", None, None
        error = f"{type(exc).__name__}: {exc}"
    return {
        "requested": requested,
        "active": active,
        "name": name,
        "vram_gb": vram,
        "gpu_detected": active not in ("cpu", "unknown"),
        "fallback_reason": _fallback_reason(requested, active, torch_info),
        "error": error,
    }


# ── models ──────────────────────────────────────────────────────────────────
def _hf_cache_index() -> dict[str, dict[str, Any]]:
    try:
        from huggingface_hub import scan_cache_dir

        out: dict[str, dict[str, Any]] = {}
        for repo in scan_cache_dir().repos:
            out[repo.repo_id] = {
                "size_bytes": int(getattr(repo, "size_on_disk", 0) or 0),
                "files": int(getattr(repo, "nb_files", 0) or 0),
            }
        return out
    except Exception:
        return {}


def _model_specs(state) -> list[dict[str, Any]]:
    settings = state.settings.get("models", {})
    siglip_repo = settings.get("siglip_variant") or DEFAULT_SIGLIP
    tagger_repo = settings.get("wd_tagger_variant") or DEFAULT_WD_TAGGER
    try:  # model tiers: the library's embedding model + the tier's tagger
        siglip_repo = state.db_embedding_model()
        tagger_repo = state.model_selection()["models"]["tagger"]
    except Exception:
        pass

    from .models.birefnet import VARIANTS as BIREfNET_VARIANTS
    from .models.wd_tagger import weight_files as tagger_weight_files
    from .models.sam2 import VARIANTS as SAM2_VARIANTS

    biref_variant = settings.get("birefnet_variant", "general")
    biref_repo = BIREfNET_VARIANTS.get(biref_variant, BIREfNET_VARIANTS["general"])
    sam2_variant = settings.get("sam2_variant", "base_plus")
    sam2_repo = SAM2_VARIANTS.get(sam2_variant, SAM2_VARIANTS["base_plus"])
    mask_enabled = bool(settings.get("enable_mask"))
    mask_engine = settings.get("mask_engine", "birefnet")

    return [
        {
            "key": "siglip",
            "label": "SigLIP 2",
            "repo_id": siglip_repo,
            "files": None,
            "enabled": True,
            "loaded": state._siglip is not None,
        },
        {
            "key": "wd_tagger",
            "label": "wd-tagger",
            "repo_id": tagger_repo,
            "files": list(tagger_weight_files(getattr(state._device, "backend", None))),
            "enabled": True,
            "loaded": state._tagger is not None,
        },
        {
            "key": "birefnet",
            "label": f"BiRefNet / Auto ({biref_variant})",
            "repo_id": biref_repo,
            "files": ["isnetis.onnx"] if biref_variant == "anime" else None,
            "enabled": mask_enabled and mask_engine == "birefnet",
            "loaded": state._birefnet is not None,
        },
        {
            "key": "sam2",
            "label": f"SAM 2 ({sam2_variant})",
            "repo_id": sam2_repo,
            "files": None,
            "enabled": mask_enabled and mask_engine == "sam2",
            "loaded": state._sam2 is not None,
        },
    ]


def _last_error_for(repo_id: str, errors: list[dict[str, Any]]) -> Optional[str]:
    needle = repo_id.lower()
    for entry in errors:
        text = f"{entry.get('message', '')}\n{entry.get('traceback', '')}".lower()
        if needle in text:
            return entry.get("message")
    return None


def build_model_states(state) -> list[dict[str, Any]]:
    cache = _hf_cache_index()
    errors = recent_errors(50)
    out = []
    for spec in _model_specs(state):
        cached = cache.get(spec["repo_id"])
        present = bool(cached) or bool(spec["loaded"])
        error = _last_error_for(spec["repo_id"], errors)
        out.append({
            "key": spec["key"],
            "label": spec["label"],
            "repo_id": spec["repo_id"],
            "enabled": spec["enabled"],
            "present": present,
            "loaded": spec["loaded"],
            "size_bytes": cached["size_bytes"] if cached else None,
            "file_count": cached["files"] if cached else 0,
            "error": error,
        })
    return out


def find_model_spec(state, key: str) -> Optional[dict[str, Any]]:
    for spec in _model_specs(state):
        if spec["key"] == key:
            return spec
    return None


# ── report ──────────────────────────────────────────────────────────────────
def _disk_info(path: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(str(path))
        return {
            "path": str(path),
            "total_gb": round(usage.total / 2**30, 2),
            "free_gb": round(usage.free / 2**30, 2),
            "used_percent": round((usage.used / usage.total) * 100, 1) if usage.total else None,
        }
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}


def _fmt_bytes(value: Optional[int]) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# Video/media file names often carry episode titles: the copied report masks
# them (keeping only the extension) so a public bug ticket doesn't list them.
_MEDIA_EXT = r"\.(mkv|mp4|m4v|mov|avi|webm|wmv|flv|ts|m2ts|mpg|mpeg|ogv|mxf)(?!\.?\w)"
# A path may contain spaces, but not the start of another path nor a ": "/", ".
_PATH_CHAR = r"(?:(?!\s(?:/|~|[A-Za-z]:\\)|[:;,]\s)[^\n`'\"<>|])"
_MEDIA_RE = re.compile(
    # an absolute path up to a media extension …
    r"(?:[A-Za-z]:[\\/]|\\\\|~[\\/]|/)" + _PATH_CHAR + "*?" + _MEDIA_EXT
    # … a quoted file name …
    + r"|(?<=['\"`])[^\n`'\"<>|]+?" + _MEDIA_EXT + r"(?=['\"`])"
    # … or a bare file name.
    + r"|[^\s`'\"<>|\\/]+" + _MEDIA_EXT,
    re.IGNORECASE,
)


def anonymize(text: str) -> str:
    """Strip personal details from text meant to be pasted in a public ticket.

    The home folder becomes ``~``, any remaining occurrence of the user name
    becomes ``<user>``, and media file paths become ``<video>.<ext>``.
    """
    home = str(Path.home())
    variants = {home, home.replace("\\", "/"), Path.home().as_posix()}
    for h in sorted((v for v in variants if len(v) > 1), key=len, reverse=True):
        text = re.sub(re.escape(h), "~", text, flags=re.IGNORECASE)
    text = _MEDIA_RE.sub(lambda m: f"<video>.{next(g for g in m.groups() if g).lower()}", text)
    user = Path.home().name
    for name in {user, os.environ.get("USER", ""), os.environ.get("USERNAME", "")}:
        if len(name) >= 3:
            text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", "<user>", text,
                          flags=re.IGNORECASE)
    return text


def render_markdown(report: dict[str, Any]) -> str:
    return anonymize(_render_markdown(report))


def _render_markdown(report: dict[str, Any]) -> str:
    dev = report.get("device", {})
    lines = [
        "## AMV Tools diagnostic report",
        "",
        f"- **App version:** {report.get('app', {}).get('version')}",
        f"- **Python:** {report.get('python')}",
        f"- **Platform:** {report.get('platform')}",
        f"- **Torch:** {report.get('torch', {}).get('version') or 'not installed'}",
        f"- **FFmpeg:** {report.get('ffmpeg', {}).get('version') or 'not found'}",
        f"- **Device requested:** {dev.get('requested')}",
        f"- **Device active:** {dev.get('active')} ({dev.get('name') or 'unknown'})",
    ]
    if dev.get("fallback_reason"):
        lines.append(f"- **GPU fallback:** {dev['fallback_reason']}")
    disk = report.get("data_dir", {})
    if disk.get("free_gb") is not None:
        lines.append(
            f"- **Data folder:** `{disk['path']}` — {disk['free_gb']} GB free "
            f"of {disk['total_gb']} GB ({disk['used_percent']}% used)"
        )
    lines += ["", "### Models", ""]
    for model in report.get("models", []):
        state = "present" if model.get("present") else "absent"
        if model.get("loaded"):
            state += ", loaded"
        size = _fmt_bytes(model.get("size_bytes"))
        suffix = f" — {size}" if model.get("size_bytes") is not None else ""
        error = f" — error: {model['error']}" if model.get("error") else ""
        lines.append(f"- **{model.get('label')}** (`{model.get('repo_id')}`): {state}{suffix}{error}")
    lines += ["", "### Logs", "", f"- Backend log: `{report.get('logs', {}).get('path')}`"]
    errs = report.get("logs", {}).get("recent_errors") or []
    if errs:
        lines.append("")
        lines.append("### Recent errors")
        lines.append("")
        lines.append("```text")
        for entry in errs[:8]:
            lines.append(f"[{entry.get('time')}] {entry.get('level')} {entry.get('logger')}: {entry.get('message')}")
        lines.append("```")
    else:
        lines.append("- No recent warnings/errors.")
    lines.append("")
    lines.append("<!-- Generated by AMV Tools → Settings → Diagnostic -->")
    return "\n".join(lines)


def build_diagnostics() -> dict[str, Any]:
    from .state import get_state

    state = get_state()
    data_dir = user_data_dir()
    setup_logging()
    report = {
        "app": {
            "name": "AMV Tools",
            "version": _pyproject_version(),
            "frozen": bool(getattr(sys, "frozen", False)),
        },
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "torch": _torch_info(),
        "ffmpeg": _ffmpeg_info(),
        "device": _device_info(state),
        "data_dir": _disk_info(data_dir),
        "models": build_model_states(state),
        "logs": {
            "path": str(log_file_path()),
            "dir": str(logs_dir()),
            "recent_errors": recent_errors(10),
        },
        "settings": {
            "indexing_device": state.settings.get("indexing", {}).get("device"),
            "mask_enabled": bool(state.settings.get("models", {}).get("enable_mask")),
        },
    }
    report["markdown"] = render_markdown(report)
    return report


def ensure_logging_on_import() -> None:
    """Convenience for headless launches; server.py calls setup_logging()."""
    setup_logging()


__all__ = [
    "build_diagnostics",
    "build_model_states",
    "find_model_spec",
    "logs_dir",
    "log_file_path",
    "recent_errors",
    "setup_logging",
]
