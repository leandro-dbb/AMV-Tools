"""Finishing toolbox endpoints (upscale, slow-mo / 60 fps, depth, cutout).

  GET    /api/finish/capabilities          device + which models are ready
  GET    /api/finish/models                registry (size, license, installed)
  POST   /api/finish/models/{id}/download  throttled background download
  DELETE /api/finish/models/{id}           remove weights from disk
  POST   /api/finish/estimate              5-frame speed bench -> ETA
  GET    /api/finish/jobs                  queue + history
  POST   /api/finish/jobs                  enqueue one job per source
  POST   /api/finish/jobs/{id}/cancel
  DELETE /api/finish/jobs/{id}             forget a finished job
  DELETE /api/finish/jobs                  forget all finished jobs

Progress streams on ``/ws/progress`` as ``finish_job`` / ``finish_model``
events. Missing weights or a missing torch runtime surface as 409/503 with a
readable message — never a sidecar crash.
"""
from __future__ import annotations

import importlib.util
import logging
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..finish import registry
from ..state import get_state

log = logging.getLogger(__name__)
router = APIRouter()

Kind = Literal["upscale", "interpolate", "depth", "matte"]

# Which registry model each job kind needs ("matte" uses the Mask Mode engine
# managed by the existing Settings > Models > Mask section).
_KIND_MODEL = {
    "upscale": None,          # depends on params.model
    "interpolate": "rife-4.25",
    "depth": "depth-anything-v2-small",
    "matte": None,
}


class Source(BaseModel):
    scene_id: Optional[int] = None
    path: Optional[str] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    label: Optional[str] = None


class JobsBody(BaseModel):
    kind: Kind
    sources: list[Source] = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class EstimateBody(BaseModel):
    kind: Kind
    source: Source
    params: dict[str, Any] = Field(default_factory=dict)


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _required_model(kind: str, params: dict) -> Optional[str]:
    if kind in ("upscale", "interpolate", "depth"):
        # no "model" param: the selected tier's model for this kind
        from ..finish.runner import model_for
        model = model_for(kind, params)
        spec = registry.MODELS.get(model)
        if spec is None or spec.kind != kind:
            raise HTTPException(400, f"unknown {kind} model {model!r}")
        if kind == "upscale":
            from ..finish.upscale import UPSCALE_MODELS
            if model not in UPSCALE_MODELS:
                raise HTTPException(400, f"unknown upscale model {model!r}")
        return model
    return _KIND_MODEL.get(kind)


def _with_model(kind: str, params: dict) -> dict:
    """Params with the model actually used recorded (tier default filled in)."""
    model = _required_model(kind, params)
    return {**params, "model": model} if model and kind != "matte" else params


def _check_ready(kind: str, params: dict) -> None:
    if not _torch_available():
        raise HTTPException(503, "PyTorch is not installed in this runtime. "
                                 "Install a GPU/CPU backend from Settings > Models first.")
    model = _required_model(kind, params)
    if model and not registry.is_installed(model):
        raise HTTPException(409, str(registry.ModelMissing(model)))
    if kind == "matte" and not get_state().settings.get("models", {}).get("enable_mask", False):
        raise HTTPException(409, "Mask Mode is disabled. Enable it in Settings > Models.")


def _validate_params(kind: str, p: dict) -> None:
    if kind == "upscale" and int(p.get("scale") or 2) not in (2, 4):
        raise HTTPException(400, "scale must be 2 or 4")
    if kind == "interpolate":
        mode = p.get("mode") or "slowmo"
        if mode not in ("slowmo", "fps"):
            raise HTTPException(400, "mode must be slowmo or fps")
        if mode == "slowmo" and int(p.get("factor") or 2) not in (2, 4, 8):
            raise HTTPException(400, "factor must be 2, 4 or 8")
    if kind == "depth" and (p.get("format") or "png16") not in ("png16", "prores", "h264"):
        raise HTTPException(400, "format must be png16, prores or h264")
    if kind == "depth" and p.get("depth_input_size") is not None:
        try:
            size = int(p["depth_input_size"])
        except (TypeError, ValueError):
            raise HTTPException(400, "depth_input_size must be an integer")
        if not 140 <= size <= 2058:
            raise HTTPException(400, "depth_input_size must be between 140 and 2058")


def _check_memory(kind: str, sources: list[Source], params: dict) -> None:
    """Depth and cutout keep the whole clip in RAM: refuse (400) a source
    that would not fit before queueing anything, instead of letting the OS
    kill the sidecar mid-job."""
    if kind not in ("depth", "matte"):
        return
    from ..finish.runner import ClipTooLong, check_memory, resolve_source

    models_cfg = get_state().settings.get("models", {})
    for src in sources:
        if src.scene_id is None and not src.path:
            continue
        try:
            clip, _ = resolve_source(src.model_dump(exclude_none=True, exclude={"label"}))
        except Exception as exc:  # missing / unreadable: the job reports it
            log.debug("memory check skipped for %s: %s", src, exc)
            continue
        try:
            check_memory(kind, clip, params, models_cfg)
        except ClipTooLong as exc:
            label = src.label or (f"scene {src.scene_id}" if src.scene_id is not None else src.path)
            raise HTTPException(400, f"{label}: {exc}" if len(sources) > 1 else str(exc))


@router.get("/api/finish/capabilities")
def capabilities():
    torch_ok = _torch_available()
    device = {"backend": "unavailable", "name": "", "vram_gb": None}
    if torch_ok:
        try:
            d = get_state().device
            device = {"backend": d.backend, "name": d.name, "vram_gb": d.vram_gb}
        except Exception as exc:
            device = {"backend": "unavailable", "name": str(exc), "vram_gb": None}
    return {
        "torch": torch_ok,
        "device": device,
        "cpu_only": device["backend"] == "cpu",
        "mask_enabled": bool(get_state().settings.get("models", {}).get("enable_mask", False)),
        "models": registry.list_models(),
    }


@router.get("/api/finish/models")
def list_models():
    return {"models": registry.list_models(), "root": str(registry.models_root())}


@router.post("/api/finish/models/{model_id}/download")
def download_model(model_id: str):
    if model_id not in registry.MODELS:
        raise HTTPException(404, "unknown model")
    from ..model_download import ModelDownloadError, check_finish_license
    try:
        check_finish_license(registry.MODELS[model_id])
    except ModelDownloadError as exc:
        raise HTTPException(403, str(exc))
    started = registry.start_download(model_id, get_state().publish)
    return {"ok": True, "started": started, "model": registry.describe(model_id)}


@router.delete("/api/finish/models/{model_id}")
def delete_model(model_id: str):
    if model_id not in registry.MODELS:
        raise HTTPException(404, "unknown model")
    with registry._dl_lock:
        downloading = bool(registry._downloads.get(model_id, {}).get("running"))
    if downloading:
        raise HTTPException(409, "this model is downloading: wait for it to finish")
    from ..finish.jobs import get_manager
    if any(j["status"] in ("queued", "running") for j in get_manager().list()):
        raise HTTPException(409, "finishing jobs are queued or running: delete the model after them")
    from ..finish.runner import release_models
    release_models()
    removed = registry.delete(model_id)
    return {"ok": True, "removed": removed, "model": registry.describe(model_id)}


@router.post("/api/finish/estimate")
def estimate(body: EstimateBody):
    _check_ready(body.kind, body.params)
    _validate_params(body.kind, body.params)
    from ..finish.jobs import get_manager
    from ..finish.runner import estimate as run_estimate

    if any(j["status"] == "running" for j in get_manager().list()):
        raise HTTPException(409, "a finishing job is running — estimate after it finishes")
    params = _with_model(body.kind, body.params)
    try:
        return run_estimate(body.kind, {**params, "source": body.source.model_dump(exclude_none=True)})
    except FileNotFoundError as exc:
        raise HTTPException(410, str(exc))
    except registry.ModelMissing as exc:
        raise HTTPException(409, str(exc))


@router.get("/api/finish/jobs")
def list_jobs():
    from ..finish.jobs import get_manager
    return {"jobs": get_manager().list()}


@router.post("/api/finish/jobs")
def create_jobs(body: JobsBody):
    _check_ready(body.kind, body.params)
    _validate_params(body.kind, body.params)
    from ..finish.jobs import get_manager

    mgr = get_manager()
    job_params = _with_model(body.kind, body.params)
    _check_memory(body.kind, body.sources, job_params)
    created = []
    for src in body.sources:
        if src.scene_id is None and not src.path:
            raise HTTPException(400, "each source needs scene_id or path")
        label = src.label or (f"scene {src.scene_id}" if src.scene_id is not None else src.path)
        params = {**job_params, "source": src.model_dump(exclude_none=True, exclude={"label"})}
        created.append(mgr.submit(body.kind, params, label=label or "").to_dict())
    return {"ok": True, "jobs": created}


@router.post("/api/finish/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    from ..finish.jobs import get_manager
    if not get_manager().cancel(job_id):
        raise HTTPException(409, "job is not queued or running")
    return {"ok": True}


@router.delete("/api/finish/jobs/{job_id}")
def remove_job(job_id: str):
    from ..finish.jobs import get_manager
    if not get_manager().remove(job_id):
        raise HTTPException(409, "job not found or still active")
    return {"ok": True}


@router.delete("/api/finish/jobs")
def clear_jobs():
    from ..finish.jobs import get_manager
    return {"ok": True, "removed": get_manager().clear_finished()}
