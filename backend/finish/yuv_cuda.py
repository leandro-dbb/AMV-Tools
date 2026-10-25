"""RGB -> I420 on the GPU for the finishing writers (CUDA).

The writer thread used to convert every output frame with OpenCV on the CPU
(a 4K upscale frame: 25 MB of RGB in, 12 MB of I420 out). When the frame
was produced on the GPU anyway, converting it there ships half the bytes
back and leaves the CPU alone. Same conversion as OpenCV's
``COLOR_RGB2YUV_I420``: BT.601 limited range, chroma taken from the
top-left pixel of each 2x2 block, rounded to nearest (within 1/255 of it).
"""
from __future__ import annotations

import numpy as np


class I420Frame:
    """An already converted frame for :class:`video_io.FrameWriter`:
    ``data`` is the packed Y, U, V planes (uint8, H*W*3/2 bytes)."""

    __slots__ = ("data", "width", "height")

    def __init__(self, data: np.ndarray, width: int, height: int):
        self.data, self.width, self.height = data, width, height


def rgb255_to_i420(rgb) -> I420Frame:
    """``rgb``: (H, W, 3) float tensor of 0..255 values (already rounded, as
    the uint8 frame would be) on the GPU; H and W even."""
    import torch

    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    rs, gs, bs = r[0::2, 0::2], g[0::2, 0::2], b[0::2, 0::2]
    u = 128.0 + (-37.797 * rs - 74.203 * gs + 112.0 * bs) / 255.0
    v = 128.0 + (112.0 * rs - 93.786 * gs - 18.214 * bs) / 255.0
    planes = torch.cat([y.reshape(-1), u.reshape(-1), v.reshape(-1)])
    data = planes.round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()
    return I420Frame(data, w, h)
