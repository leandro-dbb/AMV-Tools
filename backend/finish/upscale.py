"""Anime upscaling with Real-ESRGAN (PyTorch, CUDA/MPS/CPU).

Two checkpoints, both x4 natively:

- ``realesr-animevideov3`` — tiny SRVGG net (2.4 MB) trained for anime
  *video*: temporally steadier, ~10x faster than the 6B model. Default.
- ``realesrgan-x4plus-anime-6b`` — RRDB net (18 MB), crisper line work on
  stills, slower and slightly more prone to frame-to-frame shimmer.

x2 output = x4 inference + area downscale — the same "outscale" trick the
official Real-ESRGAN CLI uses, and visibly better than a dedicated x2 pass of
the compact net on anime line art.

Frames larger than ``tile`` are processed in overlapping tiles (each tile is
padded by ``tile_pad`` px of real context, then the padding is cropped from
the output), which bounds VRAM regardless of the input resolution. With a
pad larger than the network's effective receptive field the tiled result is
numerically identical to a whole-frame pass (covered by tests).
"""
from __future__ import annotations

import logging
import math
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

UPSCALE_MODELS = ("realesr-animevideov3", "realesrgan-x4plus-anime-6b", "hfa2k-avc-compact",
                  "animejanai-v3-uc")
# Native x2 Compact checkpoints: (weights file, SRVGGNetCompact num_conv).
NATIVE_X2 = {"hfa2k-avc-compact": ("2xHFA2kAVCCompact.pth", 16),
             "animejanai-v3-uc": ("2x_AnimeJaNai_HD_V3_UltraCompact.pth", 8)}
DEFAULT_UPSCALE_MODEL = "realesr-animevideov3"


def auto_tile(backend: str, vram_gb: Optional[float], width: int, height: int) -> int:
    """Input-pixel tile size. 0 = whole frame."""
    if backend == "cpu":
        # Tiling barely costs anything on CPU and keeps RAM bounded for 1080p
        # inputs (a x4 RRDB activation map at 1080p is several GB).
        return 256
    if backend == "cuda":
        if vram_gb is None or vram_gb <= 4:
            return 192
        if vram_gb <= 8:
            return 384
        return 0 if max(width, height) <= 1280 else 512
    if backend == "mps":
        # Unified memory: a whole 1080p frame (x4 activations, fp16) peaks at
        # ~3 GB of Metal memory and runs 11 % faster than 384 px tiles on an
        # M1 Pro. Bigger inputs or small Macs keep tiling.
        if (vram_gb or 0) >= 10 and max(width, height) <= 1920:
            return 0
        return 768 if (vram_gb or 0) >= 10 else 384
    return 384  # xpu / dml: conservative


def tiled_apply(fn: Callable, img, *, scale: int, tile: int, tile_pad: int = 10,
                check_cancel: Optional[Callable[[], None]] = None):
    """Run ``fn`` (NCHW tensor -> NCHW tensor upscaled by ``scale``) over
    ``img`` in tiles of ``tile`` px with ``tile_pad`` px of context."""
    import torch

    n, c, h, w = img.shape
    if tile <= 0 or (h <= tile and w <= tile):
        return fn(img)
    out = torch.empty((n, c, h * scale, w * scale), dtype=img.dtype, device=img.device)
    tiles_y = math.ceil(h / tile)
    tiles_x = math.ceil(w / tile)
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            if check_cancel:
                check_cancel()
            y0, x0 = ty * tile, tx * tile
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            py0, px0 = max(y0 - tile_pad, 0), max(x0 - tile_pad, 0)
            py1, px1 = min(y1 + tile_pad, h), min(x1 + tile_pad, w)
            res = fn(img[:, :, py0:py1, px0:px1])
            oy0, ox0 = (y0 - py0) * scale, (x0 - px0) * scale
            out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = \
                res[:, :, oy0:oy0 + (y1 - y0) * scale, ox0:ox0 + (x1 - x0) * scale]
    return out


def output_size(width: int, height: int, scale: int) -> tuple[int, int]:
    """Final (w, h) for an x``scale`` request, rounded to even numbers so
    4:2:0 encoders accept it."""
    w, h = width * scale, height * scale
    return w - (w % 2), h - (h % 2)


class Upscaler:
    def __init__(self, model_id: str, device, *, dtype=None):
        import torch

        from . import registry
        from .archs.realesrgan import RRDBNet, SRVGGNetCompact

        if model_id not in UPSCALE_MODELS:
            raise ValueError(f"unknown upscale model {model_id!r}")
        d = registry.require(model_id)
        self.model_id = model_id
        self.device = device
        self.native_scale = 4
        if model_id == "realesr-animevideov3":
            net = SRVGGNetCompact(num_feat=64, num_conv=16, upscale=4)
            ckpt = d / "realesr-animevideov3.pth"
        elif model_id in NATIVE_X2:
            fname, num_conv = NATIVE_X2[model_id]
            net = SRVGGNetCompact(num_feat=64, num_conv=num_conv, upscale=2)
            ckpt = d / fname
            self.native_scale = 2
        else:
            net = RRDBNet(num_block=6)
            ckpt = d / "RealESRGAN_x4plus_anime_6B.pth"
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        sd = sd.get("params_ema") or sd.get("params") or sd
        net.load_state_dict(sd, strict=True)
        # fp16 on MPS too: -12 % time, PSNR 62.7 dB vs the fp32 output on
        # real 1080p anime frames (max 7/255 on a handful of pixels).
        self.dtype = dtype or (torch.float16 if device.backend in ("cuda", "mps") else torch.float32)
        self.net = net.eval().to(device.torch_device, self.dtype)
        if device.backend == "cuda":
            # NHWC convolutions (tensor cores, fp16): -20-25 % per frame on an
            # RTX 3090, output within 1/255 of NCHW.
            self.net = self.net.to(memory_format=torch.channels_last)

    def upscale(self, frame: np.ndarray, *, scale: int = 4, tile: int = 0, tile_pad: int = 10,
                check_cancel: Optional[Callable[[], None]] = None, i420: bool = False):
        """``frame``: RGB uint8 (H, W, 3) -> RGB uint8 (H*scale, W*scale, 3)
        (dimensions rounded down to even). ``i420`` (CUDA): return an
        :class:`yuv_cuda.I420Frame` converted on the GPU when the output size
        needs no CPU resize."""
        import torch

        h, w = frame.shape[:2]
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(frame)).to(self.device.torch_device)
            x = x.permute(2, 0, 1).unsqueeze(0).to(self.dtype) / 255.0
            y = tiled_apply(self.net, x, scale=self.native_scale, tile=tile, tile_pad=tile_pad,
                            check_cancel=check_cancel)
            if scale > self.native_scale:
                # x4 from a native x2 net: a second x2 pass (tiles twice as
                # big in input pixels would quadruple the activations).
                y = tiled_apply(self.net, y.clamp(0.0, 1.0), scale=self.native_scale,
                                tile=tile // 2 if tile else 0, tile_pad=tile_pad, check_cancel=check_cancel)
            if self.device.backend == "cuda" and scale < self.native_scale and self.native_scale % scale == 0:
                # x2 from the x4 net: the area downscale (exact box average
                # for an integer ratio, like cv2.INTER_AREA) runs on the GPU,
                # so a 1080p frame ships 25 MB back to the host instead of an
                # 8K 100 MB one plus a CPU resize — the job's former bottleneck.
                import torch.nn.functional as F
                k = self.native_scale // scale
                y = F.avg_pool2d(y.float(), k)
            y = y.float().clamp_(0.0, 1.0)
            if self.device.backend == "mps" and self.native_scale == 2 * scale:
                # x4 -> x2 on the GPU: a 2x2 mean is exactly INTER_AREA for an
                # integer factor, and the host copy shrinks 4x (100 MB per
                # 1080p frame -> 25 MB).
                import torch.nn.functional as F
                y = F.avg_pool2d(y, 2)
            tw, th = output_size(w, h, scale)
            if i420 and self.device.backend == "cuda" and (y.shape[-1], y.shape[-2]) == (tw, th):
                from .yuv_cuda import rgb255_to_i420
                return rgb255_to_i420((y[0].permute(1, 2, 0) * 255.0).round())
            out = (y[0].permute(1, 2, 0) * 255.0).round().byte().cpu().numpy()
        if (out.shape[1], out.shape[0]) != (tw, th):
            import cv2
            out = cv2.resize(out, (tw, th), interpolation=cv2.INTER_AREA)
        return out
