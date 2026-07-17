from fastapi import APIRouter

from ..paths import user_data_dir
from ..state import get_state, python_version_string

router = APIRouter()


@router.get("/api/status")
def status():
    state = get_state()
    setup_marker = user_data_dir() / ".setup_complete"
    setup_required = not setup_marker.exists()
    if setup_required:
        return {
            "ready": False,
            "setup_required": True,
            "device": "not_configured",
            "gpu_name": "Choose a backend",
            "vram_gb": None,
            "models_loaded": False,
            "python_version": python_version_string(),
            "tagger_provider": None,
            "warnings": [],
        }

    try:
        device = state.device
        device_name = device.name
        device_backend = device.backend
        device_vram = device.vram_gb
    except Exception as e:
        device_name = f"unavailable: {e}"
        device_backend = "unknown"
        device_vram = None

    runtime = state.runtime_status()
    return {
        "ready": not setup_required,
        "setup_required": setup_required,
        "device": device_backend,
        "gpu_name": device_name,
        "vram_gb": device_vram,
        "models_loaded": state.models_loaded(),
        "python_version": python_version_string(),
        "tagger_provider": runtime["tagger_provider"],
        "warnings": runtime["warnings"],
    }
