from fastapi import APIRouter
from typing import Any

from ..state import get_state

router = APIRouter()


@router.get("/api/settings")
def get_settings():
    return get_state().settings


@router.put("/api/settings")
def put_settings(payload: dict[str, Any]):
    return get_state().update_settings(payload)


@router.post("/api/settings/import")
def import_settings(payload: dict[str, Any]):
    return get_state().replace_settings(payload)
