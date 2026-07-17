"""SigLIP 2 loader — image + text embedding for semantic search."""
from __future__ import annotations

import threading
import time
import logging
from typing import List, Optional

import numpy as np
from PIL import Image

from .device import DeviceInfo

log = logging.getLogger(__name__)


class SigLIP2Model:
    """Lazy-loaded SigLIP 2 wrapper.

    - Default checkpoint is ``google/siglip2-base-patch16-256`` (per ROADMAP §2).
    - Embeddings are L2-normalized so cosine similarity == dot product.
    - Provides ``offload_after_idle`` to free VRAM when not in use.
    """

    def __init__(self, device: DeviceInfo, checkpoint: str = "google/siglip2-base-patch16-naflex",
                 idle_offload_seconds: int = 60, hf_token: Optional[str] = None,
                 use_tensorrt: bool = False, max_num_patches: int = 128):
        self.device = device
        self.checkpoint = checkpoint
        self.hf_token = hf_token
        self.idle_offload_seconds = idle_offload_seconds
        self.use_tensorrt = use_tensorrt
        self.max_num_patches = int(max_num_patches) if max_num_patches else 128
        self._model = None
        self._processor = None
        self._lock = threading.Lock()
        self._last_used = 0.0
        self._offload_thread: Optional[threading.Thread] = None
        self._stop_offload = threading.Event()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoProcessor, Siglip2Model

            if self.device.backend == "cuda" and hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            self._processor = AutoProcessor.from_pretrained(self.checkpoint, token=self.hf_token)
            kwargs = dict(torch_dtype=self.device.dtype)
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                kwargs["attn_implementation"] = "sdpa"
            self._model = Siglip2Model.from_pretrained(
                self.checkpoint, token=self.hf_token, **kwargs
            ).to(self.device.torch_device)
            self._model.eval()
            if self.use_tensorrt and self.device.backend == "cuda":
                self._enable_tensorrt(torch)
            self._start_offload_watcher()

    def _enable_tensorrt(self, torch):
        try:
            import torch_tensorrt  # noqa: F401
        except Exception as exc:
            log.warning("TensorRT requested but torch_tensorrt is unavailable: %s", exc)
            return
        if not hasattr(torch, "compile") or not hasattr(self._model, "vision_model"):
            return
        try:
            if hasattr(torch, "_dynamo"):
                torch._dynamo.config.suppress_errors = True
            self._model.vision_model = torch.compile(
                self._model.vision_model,
                backend="tensorrt",
                dynamic=True,
            )
            log.info("TensorRT optimization enabled for SigLIP vision model")
        except Exception as exc:
            log.warning("TensorRT optimization failed; using standard CUDA: %s", exc)

    def _touch(self):
        self._last_used = time.monotonic()

    def _start_offload_watcher(self):
        if self.idle_offload_seconds <= 0 or self._offload_thread:
            return

        def watch():
            while not self._stop_offload.is_set():
                self._stop_offload.wait(5)
                if self._model is None:
                    continue
                if time.monotonic() - self._last_used > self.idle_offload_seconds:
                    self.offload()

        self._offload_thread = threading.Thread(target=watch, daemon=True)
        self._offload_thread.start()

    def offload(self):
        """Free VRAM. Will reload on next use."""
        import torch
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
            if self.device.backend == "cuda":
                torch.cuda.empty_cache()

    def embed_images(self, images: List[Image.Image], max_num_patches: Optional[int] = None) -> np.ndarray:
        if max_num_patches is None:
            max_num_patches = self.max_num_patches
        import torch
        self._ensure_loaded()
        self._touch()
        kwargs = {"return_tensors": "pt"}
        try:
            kwargs["max_num_patches"] = max_num_patches
            inputs = self._processor(images=images, **kwargs)
        except TypeError:
            inputs = self._processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device.torch_device, non_blocking=True) for k, v in inputs.items()}
        try:
            with torch.inference_mode():
                out = self._model.get_image_features(**inputs)
                feats = out.pooler_output if hasattr(out, "pooler_output") else out
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            return feats.float().cpu().numpy().astype(np.float32)
        except RuntimeError as exc:
            if len(images) <= 1 or "out of memory" not in str(exc).lower():
                raise
            if self.device.backend == "cuda":
                torch.cuda.empty_cache()
            mid = len(images) // 2
            log.warning("SigLIP batch of %d ran out of memory; retrying as smaller batches", len(images))
            return np.concatenate([
                self.embed_images(images[:mid], max_num_patches=max_num_patches),
                self.embed_images(images[mid:], max_num_patches=max_num_patches),
            ], axis=0)

    def embed_text(self, text: str) -> np.ndarray:
        """Encode a text query. Matches what scene-scout's legacy code did so
        searches behave the same:
          - lowercase: SigLIP's tokenizer was trained on lowercased captions;
            mixing case (e.g. "Lucario") tokenizes differently and produces
            a noisier embedding for proper nouns
          - max_length=64: explicit cap on token budget — same as legacy
        """
        import torch
        self._ensure_loaded()
        self._touch()
        inputs = self._processor(
            text=[text.lower()],
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device.torch_device, non_blocking=True) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self._model.get_text_features(**inputs)
            feats = out.pooler_output if hasattr(out, "pooler_output") else out
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return feats.float().cpu().numpy().astype(np.float32)[0]
