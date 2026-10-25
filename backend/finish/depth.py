"""Depth maps with Depth Anything V2 Small (Apache-2.0) for parallax/3D work.

Only the *Small* checkpoint is used: Base/Large/Giant are CC-BY-NC-4.0 and
cannot ship in a redistributable tool.

The model predicts relative inverse depth (disparity): bigger = nearer. We
export it as **white = near**, the convention After Effects' Displacement
Map / Depth-based effects and most 2.5D parallax plugins expect (``invert``
flips it).

Output is frame-synchronous with the clip: the same decoder timeline as
every other finishing job, one depth frame per source frame, same fps,
no audio. Formats:

- ``png16`` — 16-bit grayscale PNG sequence (max precision for AE),
- ``prores`` — ProRes 422 HQ fed with 16-bit gray (keeps 10 bits),
- ``h264``  — 8-bit H.264 preview/proxy.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

DEPTH_MODEL = "depth-anything-v2-small"
DEPTH_FORMATS = ("png16", "prores", "h264")


class DepthEstimator:
    def __init__(self, device, *, input_size: int = 518):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        from . import registry

        d = registry.require(DEPTH_MODEL)
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(str(d))
        model = AutoModelForDepthEstimation.from_pretrained(str(d))
        self.dtype = torch.float16 if device.backend == "cuda" else torch.float32
        self.model = model.eval().to(device.torch_device, self.dtype)
        self.input_size = input_size
        self._model_dir = str(d)
        # Apple Silicon: Core ML runs this net 1.75x faster than MPS (see
        # ``coreml``); the worker starts on the first frame's input shape.
        from . import coreml
        self._use_coreml = device.backend == "mps" and coreml.available()
        self._worker = None

    def predict_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """``predict`` for several frames: one fp16 forward pass for the
        batch (CUDA). The resize/normalise stays the processor's (PIL, CPU,
        ~15 ms/frame): the GPU version measured 39.4 dB on the final depth
        against it, under the 40 dB bar; the batched model alone, 79 dB."""
        import torch

        if self.device.backend != "cuda" or len(frames) < 2:
            return [self.predict(f) for f in frames]
        if getattr(self, "_pre_pool", None) is None:
            import os
            from concurrent.futures import ThreadPoolExecutor
            self._pre_pool = ThreadPoolExecutor(max_workers=max(1, min(4, (os.cpu_count() or 2) - 1)),
                                                thread_name_prefix="depth-pre")
        size = {"height": self.input_size, "width": self.input_size}
        # one call per frame, in parallel: faster than one batched call
        pix = torch.cat(list(self._pre_pool.map(
            lambda f: self.processor(images=f, return_tensors="pt", size=size)["pixel_values"], frames)))
        pix = pix.to(self.device.torch_device, self.dtype)
        with torch.inference_mode():
            out = self.model(pixel_values=pix).predicted_depth
        return list(out.float().cpu().numpy())

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """RGB uint8 (H, W, 3) -> relative disparity float32 at model
        resolution (aspect preserved, ~518 px on the short side)."""
        import torch

        inputs = self.processor(images=frame, return_tensors="pt",
                                size={"height": self.input_size, "width": self.input_size})
        if self._use_coreml:
            out = self._coreml(inputs["pixel_values"])
            if out is not None:
                return out
        pix = inputs["pixel_values"].to(self.device.torch_device, self.dtype)
        with torch.inference_mode():
            out = self.model(pixel_values=pix).predicted_depth
        return out[0].float().cpu().numpy()


    def predict_stream(self, frames):
        """``(frame, depth)`` for every frame, in order. On Core ML the next
        frame is already queued to the worker while this one is post-
        processed (one ahead), so the GPU never waits on the job thread."""
        pending: list = []
        for frame in frames:
            inputs = self.processor(images=frame, return_tensors="pt",
                                    size={"height": self.input_size, "width": self.input_size})
            rid = None
            if self._use_coreml:
                self._ensure_worker(inputs["pixel_values"])
                if self._worker.usable():
                    try:
                        rid = self._worker.submit([inputs["pixel_values"].numpy().astype(np.float32)])
                    except Exception as exc:
                        log.warning("Core ML depth stopped (%s); back to MPS", exc)
                        self._use_coreml = False
            pending.append((frame, rid, inputs))
            while len(pending) > (1 if rid is not None else 0):
                yield self._finish_one(*pending.pop(0))
        while pending:
            yield self._finish_one(*pending.pop(0))

    def _finish_one(self, frame, rid, inputs):
        import torch

        if rid is not None:
            try:
                return frame, self._worker.result(rid)[0]
            except Exception as exc:
                log.warning("Core ML depth stopped (%s); back to MPS", exc)
                self._use_coreml = False
        pix = inputs["pixel_values"].to(self.device.torch_device, self.dtype)
        with torch.inference_mode():
            out = self.model(pixel_values=pix).predicted_depth
        return frame, out[0].float().cpu().numpy()

    def _ensure_worker(self, pix) -> None:
        from . import coreml
        shape = tuple(pix.shape[-2:])
        if self._worker is None or getattr(self, "_worker_shape", None) != shape:
            if self._worker is not None:
                self._worker.close()
            self._worker = coreml.Worker("backend.finish.coreml:depth_handler", self._model_dir, *shape,
                                         name="coreml-depth")
            self._worker_shape = shape

    def _coreml(self, pix) -> Optional[np.ndarray]:
        from . import coreml
        shape = tuple(pix.shape[-2:])
        if self._worker is None or getattr(self, "_worker_shape", None) != shape:
            if self._worker is not None:
                self._worker.close()
            self._worker = coreml.Worker("backend.finish.coreml:depth_handler", self._model_dir, *shape,
                                         name="coreml-depth")
            self._worker_shape = shape
        if not self._worker.usable():
            return None
        try:
            return self._worker.call([pix.numpy().astype(np.float32)])[0]
        except Exception as exc:
            log.warning("Core ML depth stopped (%s); back to MPS", exc)
            self._use_coreml = False
            return None

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None


def normalize_stack(stack: np.ndarray, *, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    """Clip-wide robust normalisation to 0..1. One range for the whole clip
    (not per frame) — per-frame min/max is the #1 cause of depth flicker."""
    sample = stack[:, ::4, ::4]
    lo, hi = np.percentile(sample, [lo_pct, hi_pct])
    if hi - lo < 1e-6:
        return np.zeros_like(stack, dtype=np.float32)
    return np.clip((stack - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def stabilize_depth(raw: list[np.ndarray], frames_small: list[np.ndarray], *, cuts: set[int],
                    smoothing: float, check_cancel: Optional[Callable[[], None]] = None) -> np.ndarray:
    """Align per-frame scale/shift, normalise clip-wide, then (optionally)
    motion-compensated temporal smoothing. Returns (T, h, w) in 0..1."""
    from .temporal import FlowWarper, align_scale_shift, stabilize_stack

    aligned: list[np.ndarray] = []
    for i, d in enumerate(raw):
        if check_cancel:
            check_cancel()
        if i > 0 and (i - 1) not in cuts:
            d = align_scale_shift(d, aligned[-1])
        aligned.append(d.astype(np.float32))
    stack = normalize_stack(np.stack(aligned))
    if smoothing > 0 and len(raw) >= 2:
        stack = stabilize_stack(stack, cuts=cuts, strength=smoothing,
                                warper=FlowWarper(frames_small), check_cancel=check_cancel)
    return stack


def to_output_res(depth01: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    up = cv2.resize(depth01, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(up, 0.0, 1.0)
