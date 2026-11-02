"""Model-weight status + explicit retry endpoint (Settings → Diagnostic)."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ..diagnostics import build_model_states
from ..model_download import download_for_key
from ..state import get_state

router = APIRouter()


class DownloadBody(BaseModel):
    key: str


@router.get("/api/models/status")
def models_status():
    state = get_state()
    return {"models": build_model_states(state)}


@router.post("/api/models/download")
def models_download(body: DownloadBody):
    state = get_state()
    result = download_for_key(state, body.key)
    if body.key == "transnetv2" and result.get("ok"):
        # The autotune skipped the cut network while it was missing: measure
        # what it costs now (Settings > Indexing "Precise cuts" shows it)
        # instead of at the next launch.
        from .. import autotune
        autotune.start_background()
    # Real surfaces are refreshed by the next diagnostics/status poll.
    return result
