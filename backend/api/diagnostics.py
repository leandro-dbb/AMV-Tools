"""Diagnostics endpoint consumed by Settings → Diagnostic and bug reports."""
from __future__ import annotations

from fastapi import APIRouter

from ..diagnostics import build_diagnostics

router = APIRouter()


@router.get("/api/diagnostics")
def diagnostics():
    return build_diagnostics()
