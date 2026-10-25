"""Search+ scene palettes for a whole video at once on the GPU (CUDA).

:func:`analysis.palette` runs a tiny 3-colour k-means per scene with numpy:
~13 ms of CPU per scene on the 6-vCPU NVIDIA test machine, ~13 s per
3-episode season, all of it holding the GIL the decode loop needs. Here the
48x27 downsize and the deterministic initialisation stay exactly the CPU
ones (PIL, numpy) and the 10 k-means iterations of every scene of a video
run as one batch on the GPU.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import analysis

log = logging.getLogger(__name__)


def _prepare(rgb: np.ndarray):
    from PIL import Image
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    img = img.resize((48, 27), Image.Resampling.BILINEAR)
    px = np.asarray(img, dtype=np.float32).reshape(-1, 3)
    lum = px @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    order = np.argsort(lum)
    centers = px[order[[int(len(order) * q) for q in np.linspace(0.15, 0.85, 3)]]].copy()
    return px, centers


def palettes(rgbs: list, device: str = "cuda", k: int = 3) -> Optional[list]:
    """``[analysis.palette(rgb) for rgb in rgbs]``, batched; None if the GPU
    path is unavailable (callers fall back to the CPU version)."""
    if not rgbs:
        return []
    try:
        import torch

        prepared = [_prepare(r) for r in rgbs]
        px = torch.from_numpy(np.stack([p for p, _c in prepared])).to(device)            # (N, P, 3)
        centers = torch.from_numpy(np.stack([c for _p, c in prepared])).to(device)       # (N, k, 3)
        with torch.inference_mode():
            for _ in range(10):
                d = ((px[:, :, None, :] - centers[:, None, :, :]) ** 2).sum(-1)       # (N, P, k)
                labels = d.argmin(-1)
                onehot = torch.nn.functional.one_hot(labels, k).to(px.dtype)        # (N, P, k)
                counts = onehot.sum(1)                                              # (N, k)
                # element-wise (an einsum/matmul may run in TF32 once SigLIP
                # has set the float32 matmul precision to "high")
                sums = (onehot[..., None] * px[:, :, None, :]).sum(1)                # (N, k, 3)
                new = sums / counts.clamp_min(1)[..., None]
                centers = torch.where((counts > 0)[..., None], new, centers)
            shares = (counts / float(px.shape[1])).cpu().numpy()
            cent = centers.cpu().numpy()
        out = []
        for sh, ce in zip(shares, cent):
            pal = []
            for c in np.argsort(-sh):
                if sh[c] <= 0:
                    continue
                r, g, b = (int(round(float(x))) for x in ce[c])
                pal.append([f"#{r:02x}{g:02x}{b:02x}", round(float(sh[c]), 3), analysis.color_bucket(r, g, b)])
            out.append(pal)
        return out
    except Exception as exc:
        log.debug("GPU palettes unavailable (%s); CPU per scene", exc)
        return None
