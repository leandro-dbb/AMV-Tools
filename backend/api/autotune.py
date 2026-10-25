"""Automatic tuning: Settings > Diagnostic shows the choices and re-runs it."""
from __future__ import annotations

from fastapi import APIRouter

from .. import autotune

router = APIRouter()


@router.get("/api/autotune")
def autotune_status():
    return autotune.status()


@router.post("/api/autotune/run")
def autotune_run():
    """Measure again now (in the background; poll GET /api/autotune)."""
    started = autotune.start_background(force=True, delay=0.0)
    return {"started": started, **autotune.status()}
