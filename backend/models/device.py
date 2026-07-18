"""Device auto-detection — adapted from src/model_loader.py but cleaned up."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DeviceInfo:
    backend: str           # 'cuda' | 'xpu' | 'dml' | 'mps' | 'cpu'
    name: str              # human-readable
    torch_device: object   # torch.device or torch_directml.device()
    dtype: object          # torch.dtype
    vram_gb: Optional[float] = None  # total VRAM; None when the backend can't report it


def empty_device_cache(backend: str) -> None:
    """Release cached allocator blocks back to the OS/driver.

    Matters twice as much on Apple Silicon: MPS allocations live in unified
    memory, so a bloated cache pressures the entire system, not just "VRAM".
    """
    try:
        import torch
        if backend == "cuda":
            torch.cuda.empty_cache()
        elif backend == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except Exception:
        pass


def detect_device(forced: Optional[str] = None) -> DeviceInfo:
    import torch

    try:
        import torch_directml
    except ImportError:
        torch_directml = None

    def cuda_info() -> DeviceInfo:
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.float16 if major >= 7 else torch.float32
        try:
            vram = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        except Exception:
            vram = None
        return DeviceInfo("cuda", torch.cuda.get_device_name(), torch.device("cuda"), dtype, vram_gb=vram)

    def xpu_vram() -> Optional[float]:
        try:
            return round(torch.xpu.get_device_properties(0).total_memory / 2**30, 1)
        except Exception:
            return None

    def mps_info() -> DeviceInfo:
        # fp16 halves inference time and unified-memory footprint on Apple
        # Silicon — same trade-off we already make on CUDA (sm70+). The
        # BiRefNet checkpoint even ships fp16 natively.
        try:
            # Metal's working-set ceiling (~70-75% of unified memory) — the
            # honest equivalent of VRAM for the batch-size recommendation.
            mem = round(torch.mps.recommended_max_memory() / 2**30, 1)
        except Exception:
            mem = None
        return DeviceInfo("mps", "Apple Silicon GPU (MPS)", torch.device("mps"),
                          torch.float16, vram_gb=mem)

    if forced and forced != "auto":
        if forced == "cuda" and torch.cuda.is_available():
            return cuda_info()
        if forced == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            return DeviceInfo("xpu", "Intel XPU", torch.device("xpu"), torch.float16, vram_gb=xpu_vram())
        if forced == "dml" and torch_directml and torch_directml.is_available():
            return DeviceInfo("dml", "DirectML GPU", torch_directml.device(), torch.float32)
        if forced == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return mps_info()
        return DeviceInfo("cpu", "CPU (forced fallback)", torch.device("cpu"), torch.float32)

    if torch.cuda.is_available():
        return cuda_info()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return DeviceInfo("xpu", "Intel XPU", torch.device("xpu"), torch.float16, vram_gb=xpu_vram())
    if torch_directml and torch_directml.is_available():
        return DeviceInfo("dml", "DirectML GPU", torch_directml.device(), torch.float32)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return mps_info()
    return DeviceInfo("cpu", "CPU", torch.device("cpu"), torch.float32)
