"""MatAnyone wrapper — temporal mask propagation seeded by another model.

Why this exists alongside BiRefNet:

BiRefNet runs independently on every frame, so the per-frame mask shape jitters
slightly even when the character is barely moving. On a clip of 100+ frames
this reads as silhouette flicker. MatAnyone (CVPR 2025, Yang et al.) is a
*matting* model with a memory-propagation architecture: you give it the mask
for frame 0 (the "seed"), and it tracks the same subject through every
subsequent frame using temporal memory. Result: temporally stable mask edges,
which is exactly what AMV compositing needs.

The seed comes from BiRefNet (which is solid on a single anime frame). The
propagation uses MatAnyone (trained on real-world video, but its memory module
operates on silhouette + motion features that mostly transfer to anime).

The model produces a soft alpha (matting), which we now preserve through the
export pipeline instead of thresholding. That keeps anime hair tips and ink
anti-aliasing from turning into a jagged hard cutout.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, List, Optional

import numpy as np
from PIL import Image

from .device import DeviceInfo

log = logging.getLogger(__name__)


# Official MatAnyone checkpoint on HuggingFace. The author publishes one
# canonical weight set — no tiny/base/large variants like SAM 2.
DEFAULT_HF_REPO = "PeiqingYang/MatAnyone"


class MatAnyoneModel:
    """Lazy-loaded MatAnyone wrapper. Mirrors the lifecycle of
    :class:`SigLIP2Model` / :class:`BiRefNetModel` / :class:`SAM2Model` so it
    fits the same VRAM rotation."""

    def __init__(
        self,
        device: DeviceInfo,
        idle_offload_seconds: int = 60,
        hf_token: Optional[str] = None,
        repo_id: str = DEFAULT_HF_REPO,
    ):
        self.device = device
        self.repo_id = repo_id
        self.idle_offload_seconds = idle_offload_seconds
        self.hf_token = hf_token
        self._processor = None
        self._lock = threading.Lock()
        self._last_used = 0.0
        self._offload_thread: Optional[threading.Thread] = None
        self._stop_offload = threading.Event()

    # ── load / offload ──────────────────────────────────────────────────────
    def _ensure_loaded(self):
        if self._processor is not None:
            return
        with self._lock:
            if self._processor is not None:
                return
            from matanyone import InferenceCore

            log.info("loading MatAnyone (%s)", self.repo_id)
            # The repo_id string form pulls the checkpoint from HuggingFace
            # and instantiates the network + config under the hood.
            self._processor = InferenceCore(
                self.repo_id, device=self.device.torch_device,
            )
            self._start_offload_watcher()

    def _touch(self):
        self._last_used = time.monotonic()

    def _start_offload_watcher(self):
        if self.idle_offload_seconds <= 0 or self._offload_thread:
            return

        def watch():
            while not self._stop_offload.is_set():
                self._stop_offload.wait(5)
                if self._processor is None:
                    continue
                if time.monotonic() - self._last_used > self.idle_offload_seconds:
                    self.offload()

        self._offload_thread = threading.Thread(target=watch, daemon=True)
        self._offload_thread.start()

    def offload(self):
        import gc
        with self._lock:
            self._processor = None
        gc.collect()
        if self.device.backend == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    # ── inference ───────────────────────────────────────────────────────────
    def propagate(
        self,
        frame_paths: List,
        seed_mask: np.ndarray,
        *,
        threshold: float = 0.5,
        return_alpha: bool = True,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """Propagate ``seed_mask`` (from frame 0) across every frame in
        ``frame_paths`` using MatAnyone's memory propagation.

        Parameters
        ----------
        frame_paths : list[Path]
            Extracted clip frames in temporal order. Must all share the same
            dimensions as ``seed_mask``.
        seed_mask : (H, W) bool
            Subject mask for ``frame_paths[0]``. Provided by BiRefNet (Auto) or
            SAM 2 (Manual click).
        threshold : float
            Cutoff used for the binary seed mask and for legacy hard-mask
            output when ``return_alpha=False``.
        return_alpha : bool
            Return MatAnyone's soft matte as float32 [0, 1]. This is the
            default because it preserves hair tips and antialiased anime line
            work for alpha export.
        progress_cb : callable(int, int) | None
            Called after each frame as ``(processed, total)`` so the API can
            push WebSocket progress events.

        Returns
        -------
        (T, H, W) float32 or bool
            Soft alpha for every frame in ``frame_paths`` by default.
        """
        import torch

        if not frame_paths:
            raise ValueError("propagate(): frame_paths is empty")
        if seed_mask.dtype != bool:
            seed_mask = seed_mask >= threshold

        self._ensure_loaded()
        self._touch()
        assert self._processor is not None

        H, W = seed_mask.shape
        n = len(frame_paths)
        # idx_mask=True path: pass a (H, W) long tensor of class indices
        # (0 = background, 1 = foreground object). The alternative
        # (idx_mask=False with a (N, H, W) float tensor) hits a bug in
        # MatAnyone's inference_core at line 330 where `mask.max(0) > 0.5`
        # is compared to a float — but `.max(0)` returns a NamedTuple on
        # modern PyTorch, not a tensor. The idx_mask=True branch uses
        # `mask > 0` instead, which works fine.
        seed_t = (
            torch.from_numpy(seed_mask.astype(np.uint8))
            .long()
            .to(self.device.torch_device)
        )

        out_dtype = np.float32 if return_alpha else bool
        out = np.zeros((n, H, W), dtype=out_dtype)

        with torch.inference_mode():
            for i, fp in enumerate(frame_paths):
                with Image.open(fp) as im:
                    rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
                # (H, W, 3) → (3, H, W) float in [0, 1] on device.
                # MatAnyone's encode_image normalises via
                # ``(image - mean) / std`` with ImageNet stats that expect
                # [0, 1] inputs — feeding raw [0, 255] pixels produces a
                # garbage internal representation and an empty (all-bg) mask
                # downstream (the "black screen on export" symptom).
                img_t = (
                    torch.from_numpy(rgb).permute(2, 0, 1)
                    .to(self.device.torch_device)
                )

                if i == 0:
                    # matting=False forces step() through aggregate+softmax
                    # for ``pred_prob_with_bg`` instead of the cat([1-m, m])
                    # path. The latter triggers a chain that ends with
                    # mask_encoder receiving a 6D tensor and crashing on the
                    # cat with the 5D image features. The softmax path keeps
                    # ``last_mask`` at the expected 4D shape (1, N, H, W).
                    # Net effect on quality: identical for binary single-
                    # object input — both paths produce the same downstream
                    # probability map for our mask shape.
                    matte = self._processor.step(
                        img_t,
                        mask=seed_t,
                        objects=[1],
                        first_frame_pred=True,
                        idx_mask=True,
                        matting=False,
                    )
                else:
                    matte = self._processor.step(img_t)

                # step() returns (N_classes+1, H, W) — channel 0 is the
                # background probability, channels 1.. are the foreground
                # objects' soft mattes. For single-object we take channel 1.
                # output_prob_to_mask(matting=True) does the same thing
                # canonically (calls matte[1:].squeeze(0)).
                fg = self._processor.output_prob_to_mask(matte, matting=True)
                m = fg.detach().float().cpu().numpy()
                if m.ndim != 2:
                    # Safety net if MatAnyone ever returns extra dims for some
                    # configuration we hit later — collapse to (H, W).
                    m = m.squeeze()
                if return_alpha:
                    out[i] = np.clip(m, 0.0, 1.0).astype(np.float32, copy=False)
                else:
                    out[i] = m >= threshold

                if progress_cb is not None:
                    try:
                        progress_cb(i + 1, n)
                    except Exception:
                        pass

        return out
