"""On-demand model registry for the Finishing toolbox.

Weights are NOT bundled and NOT pulled at startup: each model is downloaded
explicitly from Settings > Models (or from the Finish dialog) into
``<user data>/models/finish/<model id>/``. Deleting a model removes that
folder. Nothing here goes through the HF cache, so the size shown in the UI
is exactly what sits on disk and "Delete" really frees it.

Downloads are throttled (10 MB/s by default, ``AMV_DOWNLOAD_LIMIT_MBPS``)
so a model fetch never saturates the user's line, and written to ``.part``
files that are only renamed once complete — an interrupted download never
leaves a truncated checkpoint that would later fail to load.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    size: int                      # bytes, used for progress + UI before download
    sha256: Optional[str] = None


@dataclass(frozen=True)
class ModelSpec:
    id: str
    kind: str                      # "upscale" | "interpolate" | "depth"
    label: str
    license: str
    source: str                    # human-facing project URL
    files: tuple[ModelFile, ...] = field(default_factory=tuple)

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)


_ESRGAN_RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"
# Pinned to a commit so the checksums below stay valid.
_DEPTH_REPO = ("https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/"
               "5426e4f0f36572d16453bbda7a8389317b1bef99")

MODELS: dict[str, ModelSpec] = {
    m.id: m for m in (
        ModelSpec(
            id="realesr-animevideov3",
            kind="upscale",
            label="Real-ESRGAN AnimeVideo v3 (fast)",
            license="BSD-3-Clause",
            source="https://github.com/xinntao/Real-ESRGAN",
            files=(ModelFile(
                "realesr-animevideov3.pth",
                f"{_ESRGAN_RELEASES}/v0.2.5.0/realesr-animevideov3.pth",
                2_504_012,
                "b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d",
            ),),
        ),
        ModelSpec(
            id="realesrgan-x4plus-anime-6b",
            kind="upscale",
            label="Real-ESRGAN x4plus anime 6B (quality)",
            license="BSD-3-Clause",
            source="https://github.com/xinntao/Real-ESRGAN",
            files=(ModelFile(
                "RealESRGAN_x4plus_anime_6B.pth",
                f"{_ESRGAN_RELEASES}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                17_938_799,
                "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
            ),),
        ),
        ModelSpec(
            id="hfa2k-avc-compact",
            kind="upscale",
            # Attribution required by CC-BY-4.0: Philip Hofmann (Phhofm).
            label="2xHFA2kAVCCompact by Phhofm (faithful x2)",
            license="CC-BY-4.0",
            source="https://github.com/Phhofm/models",
            files=(ModelFile(
                "2xHFA2kAVCCompact.pth",
                "https://github.com/Phhofm/models/raw/main/2xHFA2kAVCCompact/2xHFA2kAVCCompact.pth",
                4_838_301,
                "ceaf8d6b5db4bf1ddbc9bc757eb08e67b3409472d9373c5342ea3711ed28365e",
            ),),
        ),
        ModelSpec(
            id="rife-4.25",
            kind="interpolate",
            label="RIFE 4.25 (Practical-RIFE)",
            license="MIT",
            source="https://github.com/hzwer/Practical-RIFE",
            # Practical-RIFE itself only publishes Google Drive/Baidu links;
            # vs-rife (MIT) mirrors the identical flownet on GitHub Releases,
            # which is a stable, scriptable URL.
            files=(ModelFile(
                "flownet_v4.25.pkl",
                "https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.25.pkl",
                24_636_301,
                "6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2",
            ),),
        ),
        ModelSpec(
            id="rife-4.26",
            kind="interpolate",
            label="RIFE 4.26 (Practical-RIFE)",
            license="MIT",
            source="https://github.com/hzwer/Practical-RIFE",
            files=(ModelFile(
                "flownet_v4.26.pkl",
                "https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.26.pkl",
                24_636_301,
                "45c7f74156704769dc9f85cfcaf8552e1e926f9399dcfa3a553dee88fac6f53f",
            ),),
        ),
        ModelSpec(
            id="rife-4.25-lite",
            kind="interpolate",
            label="RIFE 4.25 lite (Practical-RIFE)",
            license="MIT",
            source="https://github.com/hzwer/Practical-RIFE",
            files=(ModelFile(
                "flownet_v4.25.lite.pkl",
                "https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.25.lite.pkl",
                24_467_405,
                "81cdba223fe72a120130cc8552e5d2ecac824259d406f0c15323b3decf96b8b1",
            ),),
        ),
        ModelSpec(
            id="rife-4.26-heavy",
            kind="interpolate",
            label="RIFE 4.26 heavy (Practical-RIFE)",
            license="MIT",
            source="https://github.com/hzwer/Practical-RIFE",
            files=(ModelFile(
                "flownet_v4.26.heavy.pkl",
                "https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.26.heavy.pkl",
                24_925_133,
                "4cc518e172156ad6207b9c7a43364f518832d83a4325d484240493a9e2980537",
            ),),
        ),
        ModelSpec(
            id="video-depth-anything-small",
            kind="depth",
            # Only Small is Apache-2.0 (Base/Large: CC-BY-NC-4.0, not offered).
            label="Video Depth Anything Small (temporally stable)",
            license="Apache-2.0",
            source="https://github.com/DepthAnything/Video-Depth-Anything",
            files=(ModelFile(
                "video_depth_anything_vits.pth",
                "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/"
                "256875362cff76724b920335dfb4b29dd611f66e/video_depth_anything_vits.pth",
                116_440_756,
                "13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609",
            ),),
        ),
        ModelSpec(
            id="depth-anything-v2-small",
            kind="depth",
            # Only the Small checkpoint is Apache-2.0; Base/Large/Giant are
            # CC-BY-NC-4.0 and must never be offered here.
            label="Depth Anything V2 Small",
            license="Apache-2.0",
            source="https://github.com/DepthAnything/Depth-Anything-V2",
            files=(
                ModelFile("config.json", f"{_DEPTH_REPO}/config.json", 950,
                          "c56698d3643dde1f83ea2212759e6b31a22b8f827246a36dd007ee8a22b3ff75"),
                ModelFile("preprocessor_config.json", f"{_DEPTH_REPO}/preprocessor_config.json", 775,
                          "d41175c0d889477ca8fc67191e540faef14baf6275157b3fdecf78469e6bbf84"),
                ModelFile("model.safetensors", f"{_DEPTH_REPO}/model.safetensors", 99_173_660,
                          "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"),
            ),
        ),
    )
}


class ModelMissing(RuntimeError):
    """Raised when a job needs weights that have not been downloaded yet."""

    def __init__(self, model_id: str):
        spec = MODELS.get(model_id)
        size = f" (~{spec.size / 1e6:.0f} MB)" if spec else ""
        super().__init__(
            f"Model '{model_id}' is not downloaded{size}. "
            "Download it from Settings > Models > Finishing models."
        )
        self.model_id = model_id


def models_root() -> Path:
    forced = os.environ.get("AMV_MODELS_DIR")
    if forced:
        root = Path(forced)
    else:
        from ..paths import user_data_dir
        root = user_data_dir() / "models" / "finish"
    root.mkdir(parents=True, exist_ok=True)
    return root


def model_dir(model_id: str) -> Path:
    return models_root() / model_id


def is_installed(model_id: str) -> bool:
    spec = MODELS[model_id]
    d = model_dir(model_id)
    return all((d / f.name).is_file() and (d / f.name).stat().st_size > 0 for f in spec.files)


def require(model_id: str) -> Path:
    if model_id not in MODELS:
        raise KeyError(f"unknown model {model_id!r}")
    if not is_installed(model_id):
        raise ModelMissing(model_id)
    return model_dir(model_id)


def disk_size(model_id: str) -> int:
    d = model_dir(model_id)
    if not d.exists():
        return 0
    return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())


def describe(model_id: str) -> dict:
    spec = MODELS[model_id]
    with _dl_lock:
        dl = dict(_downloads.get(model_id) or {})
    return {
        "id": spec.id,
        "kind": spec.kind,
        "label": spec.label,
        "license": spec.license,
        "source": spec.source,
        "size": spec.size,
        "installed": is_installed(model_id),
        "disk_size": disk_size(model_id),
        "downloading": bool(dl.get("running")),
        "download_percent": dl.get("percent"),
        "download_error": dl.get("error"),
    }


def list_models() -> list[dict]:
    return [describe(mid) for mid in MODELS]


def delete(model_id: str) -> bool:
    if model_id not in MODELS:
        raise KeyError(model_id)
    d = model_dir(model_id)
    if not d.exists():
        return False
    shutil.rmtree(d, ignore_errors=True)
    return True


# ── download ────────────────────────────────────────────────────────────────
_dl_lock = threading.Lock()
_downloads: dict[str, dict] = {}


def _limit_bytes_per_s() -> float:
    try:
        mbps = float(os.environ.get("AMV_DOWNLOAD_LIMIT_MBPS", "10"))
    except ValueError:
        mbps = 10.0
    return max(0.1, mbps) * 1024 * 1024


def _fetch(url: str, dest: Path, *, on_bytes: Callable[[int], None],
           cancel: Optional[threading.Event], expected_sha: Optional[str]) -> None:
    import requests

    tmp = dest.with_suffix(dest.suffix + ".part")
    limit = _limit_bytes_per_s()
    sha = hashlib.sha256() if expected_sha else None
    with requests.get(url, stream=True, timeout=30, allow_redirects=True) as r:
        r.raise_for_status()
        started = time.monotonic()
        received = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=256 * 1024):
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("download cancelled")
                if not chunk:
                    continue
                fh.write(chunk)
                if sha:
                    sha.update(chunk)
                received += len(chunk)
                on_bytes(len(chunk))
                # Token-bucket style throttle: never get ahead of the budget.
                ahead = received / limit - (time.monotonic() - started)
                if ahead > 0:
                    time.sleep(ahead)
    if sha and sha.hexdigest() != expected_sha:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {dest.name}")
    os.replace(tmp, dest)


def download(model_id: str, *, progress_cb: Optional[Callable[[int, int], None]] = None,
             cancel: Optional[threading.Event] = None) -> Path:
    """Blocking download of every file of ``model_id``. Safe to call on an
    installed model (no-op). Raises on network errors; partial ``.part``
    files are removed."""
    spec = MODELS[model_id]
    from ..model_download import check_finish_license
    check_finish_license(spec)
    d = model_dir(model_id)
    d.mkdir(parents=True, exist_ok=True)
    total = spec.size
    done = {"n": 0}

    def on_bytes(n: int):
        done["n"] += n
        if progress_cb:
            progress_cb(done["n"], total)

    try:
        for f in spec.files:
            dest = d / f.name
            if dest.is_file() and dest.stat().st_size > 0:
                on_bytes(f.size)
                continue
            log.info("downloading %s -> %s", f.url, dest)
            _fetch(f.url, dest, on_bytes=on_bytes, cancel=cancel, expected_sha=f.sha256)
    finally:
        for p in d.glob("*.part"):
            p.unlink(missing_ok=True)
    return d


def start_download(model_id: str, publish: Callable[[dict], None]) -> bool:
    """Run :func:`download` in a background thread, streaming
    ``finish_model`` progress events. Returns False if already running."""
    if model_id not in MODELS:
        raise KeyError(model_id)
    with _dl_lock:
        cur = _downloads.get(model_id)
        if cur and cur.get("running"):
            return False
        _downloads[model_id] = {"running": True, "percent": 0, "error": None}

    def run():
        last = {"pct": -1}

        def cb(n: int, total: int):
            pct = int(n * 100 / max(1, total))
            if pct != last["pct"]:
                last["pct"] = pct
                with _dl_lock:
                    _downloads[model_id]["percent"] = pct
                publish({"type": "finish_model", "model_id": model_id, "percent": pct, "done": False})

        try:
            download(model_id, progress_cb=cb)
            with _dl_lock:
                _downloads[model_id] = {"running": False, "percent": 100, "error": None}
            publish({"type": "finish_model", "model_id": model_id, "percent": 100, "done": True, "ok": True})
        except Exception as exc:
            log.warning("model download failed for %s: %s", model_id, exc)
            with _dl_lock:
                _downloads[model_id] = {"running": False, "percent": None, "error": str(exc)}
            publish({"type": "finish_model", "model_id": model_id, "done": True, "ok": False,
                     "message": str(exc)})

    threading.Thread(target=run, daemon=True, name=f"dl-{model_id}").start()
    return True
