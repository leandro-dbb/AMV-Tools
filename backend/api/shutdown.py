"""``POST /api/shutdown``: Electron's graceful stop request (works on every
platform, Windows has no SIGTERM). The process exits once the bounded
shutdown in ``backend/shutdown.py`` is done."""
from __future__ import annotations

from fastapi import APIRouter

from .. import shutdown

router = APIRouter()


@router.post("/api/shutdown")
def shutdown_sidecar():
    return {"stopping": shutdown.request_exit()}
