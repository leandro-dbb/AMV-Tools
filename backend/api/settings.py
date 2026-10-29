import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from typing import Any

from ..state import get_state

router = APIRouter()


@router.get("/api/settings")
def get_settings():
    return get_state().settings


def _check_output_folder(payload: dict[str, Any]) -> None:
    """Refuse an export folder the app cannot write to now, rather than
    letting every later export fail with an ffmpeg error."""
    folder = (payload.get("export") or {}).get("output_folder")
    current = (get_state().settings.get("export") or {}).get("output_folder")
    if not folder or folder == current:
        return
    try:
        Path(folder).mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder, prefix=".amv-write-test-"):
            pass
    except OSError as exc:
        raise HTTPException(status_code=400, detail={
            "error": f"The export folder is not writable: {folder} ({exc.strerror or exc})",
            "code": "output_folder_not_writable",
        })


@router.put("/api/settings")
def put_settings(payload: dict[str, Any]):
    _check_output_folder(payload)
    return get_state().update_settings(payload)


@router.post("/api/settings/import")
def import_settings(payload: dict[str, Any]):
    return get_state().replace_settings(payload)
