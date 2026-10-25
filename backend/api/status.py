from fastapi import APIRouter

from ..diagnostics import fallback_reason
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
            "requested_device": "not_configured",
            "device_fallback": None,
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
    requested = str(state.settings.get("indexing", {}).get("device", "auto"))
    fallback = fallback_reason(requested, device_backend, {})
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
        "requested_device": requested,
        "device_fallback": fallback,
    }
