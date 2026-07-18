"""BiRefNet wrapper — automatic foreground segmentation for anime / illustration.

Why this exists alongside SAM 2:

SAM 2 was trained on real-world video (SA-V) and underperforms on stylized art
where cel-shading, flat colors, and strong outlines violate its assumptions.
BiRefNet (Zheng et al., 2024, MIT license) is a dichotomous image segmentation
network that was trained explicitly on DIS5K + portraits + HD-art — exactly the
distribution AMV creators care about.

Key differences vs. SAM 2:

- No prompts: BiRefNet outputs the salient foreground directly, no click needed.
- Per-frame independent: we run it on each frame separately. The model is
  consistent enough on stylized content that the per-frame masks line up
  visually — we don't need a temporal propagation module to get a stable
  matte across the clip.
- Limitation: can't disambiguate between multiple subjects. If the user wants
  ONE specific character among several, they have to fall back to SAM 2.

The model is loaded via transformers' AutoModelForImageSegmentation with
``trust_remote_code=True`` because BiRefNet ships its own modeling code on the
Hub (``ZhengPeng7/BiRefNet``).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np
from PIL import Image

from .device import DeviceInfo

log = logging.getLogger(__name__)


VARIANTS: dict[str, str] = {
    # General-purpose checkpoint. Trained on DIS5K + HRSOD + PASCAL-S + more.
    # Best all-rounder; what we use by default for AMV / anime.
    "general":  "ZhengPeng7/BiRefNet",
    # Higher-resolution variant — handles 2K+ inputs more cleanly at the cost
    # of ~2× VRAM. Worth it if the user's library is 4K or has fine hair/fur.
    "hr":       "ZhengPeng7/BiRefNet_HR",
    # Portrait-tuned — better hair/face edges, mediocre on full-body.
    "portrait": "ZhengPeng7/BiRefNet-portrait",
    # NOT a BiRefNet checkpoint: SkyTNT's ISNet trained specifically on anime
    # character segmentation. Runs through onnxruntime (already a base dep for
    # wd-tagger), ~170 MB. Usually beats the photo-trained checkpoints on
    # cel-shading, flat colors, and low-contrast painted backgrounds.
    "anime":    "skytnt/anime-seg",
}
DEFAULT_VARIANT = "general"
# Standard BiRefNet inference size. The model itself is fully convolutional
# but the official recipes (and the HF model card examples) use 1024×1024 —
# bigger costs quadratic VRAM and adds little quality on typical anime frames.
INPUT_SIZE = 1024
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


class BiRefNetModel:
    """Lazy-loaded BiRefNet wrapper. Mirrors the SigLIP/SAM 2 lifecycle so it
    fits the model rotation (idle offload, eviction before indexing, etc.)."""

    def __init__(
        self,
        device: DeviceInfo,
        variant: str = DEFAULT_VARIANT,
        idle_offload_seconds: int = 60,
        hf_token: Optional[str] = None,
    ):
        self.device = device
        if variant not in VARIANTS:
            log.warning("unknown BiRefNet variant %r, falling back to %r", variant, DEFAULT_VARIANT)
            variant = DEFAULT_VARIANT
        self.variant = variant
        self.repo_id = VARIANTS[variant]
        self.idle_offload_seconds = idle_offload_seconds
        self.hf_token = hf_token
        self._model = None
        self._transform = None
        self._model_dtype = None  # set on _ensure_loaded
        self._onnx_session = None   # anime variant only (ISNet via onnxruntime)
        self._onnx_input: Optional[str] = None
        self._lock = threading.Lock()
        self._last_used = 0.0
        self._offload_thread: Optional[threading.Thread] = None
        self._stop_offload = threading.Event()

    def _ensure_loaded(self):
        if self.variant == "anime":
            self._ensure_loaded_anime()
            return
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from torchvision import transforms
            from transformers import AutoModelForImageSegmentation

            log.info("loading BiRefNet (%s)", self.repo_id)
            model = AutoModelForImageSegmentation.from_pretrained(
                self.repo_id, trust_remote_code=True, token=self.hf_token,
            )
            # The HF checkpoint config sets torch_dtype=fp16, so from_pretrained
            # returns the model in half precision on CUDA-capable cards. We let
            # it stay that way (saves ~440 MB VRAM and matches the upstream
            # demo recipe) — we just cast the input to match in mask_image.
            model = model.to(self.device.torch_device)
            model.eval()
            self._model = model
            self._model_dtype = next(model.parameters()).dtype
            self._transform = transforms.Compose([
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ])
            self._start_offload_watcher()

    def _ensure_loaded_anime(self):
        """Load SkyTNT's ISNet anime-seg checkpoint through onnxruntime.

        Same provider selection as the wd-tagger: CUDA/DML/CoreML when the
        app runs on that backend, CPU fallback otherwise. Preprocessing lives
        in :meth:`_mask_image_anime` and mirrors the upstream demo exactly.
        """
        if self._onnx_session is not None:
            return
        with self._lock:
            if self._onnx_session is not None:
                return
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download

            from .wd_tagger import _register_cuda12_dll_dirs, build_onnx_providers, make_onnx_session

            _register_cuda12_dll_dirs(required=(self.device.backend == "cuda"), ort=ort)

            log.info("loading anime-seg ISNet (%s)", self.repo_id)
            model_path = hf_hub_download(self.repo_id, "isnetis.onnx", token=self.hf_token)

            providers = build_onnx_providers(self.device.backend, ort)
            self._onnx_session = make_onnx_session(ort, model_path, providers)
            self._onnx_input = self._onnx_session.get_inputs()[0].name
            self._start_offload_watcher()

    def _touch(self):
        self._last_used = time.monotonic()

    def _start_offload_watcher(self):
        if self.idle_offload_seconds <= 0 or self._offload_thread:
            return

        def watch():
            while not self._stop_offload.is_set():
                self._stop_offload.wait(5)
                if self._model is None and self._onnx_session is None:
                    continue
                if time.monotonic() - self._last_used > self.idle_offload_seconds:
                    self.offload()

        self._offload_thread = threading.Thread(target=watch, daemon=True)
        self._offload_thread.start()

    def offload(self):
        import gc
        from .device import empty_device_cache
        with self._lock:
            self._model = None
            self._transform = None
            self._model_dtype = None
            self._onnx_session = None
            self._onnx_input = None
        gc.collect()
        empty_device_cache(self.device.backend)

    # ── inference ───────────────────────────────────────────────────────────
    def mask_image(
        self,
        image: Image.Image,
        threshold: float = 0.5,
        *,
        return_alpha: bool = True,
    ) -> np.ndarray:
        """Return a foreground alpha matte ``(H, W)`` matching ``image``'s size.

        BiRefNet outputs a 1024x1024 sigmoid prediction; we resize it back to
        the input's actual HxW with bilinear interpolation. By default the
        output is float32 alpha, not a hard threshold, because that preserves
        hair tips, motion blur, and antialiased anime line work for export.
        Set ``return_alpha=False`` to recover the legacy binary silhouette."""
        if self.variant == "anime":
            alpha = self._mask_image_anime(image)
            if return_alpha:
                return alpha
            return alpha >= threshold

        import torch
        import torch.nn.functional as F

        self._ensure_loaded()
        self._touch()
        assert self._model is not None and self._transform is not None

        rgb = image.convert("RGB")
        w, h = rgb.size

        with torch.inference_mode():
            tensor = (
                self._transform(rgb).unsqueeze(0)
                .to(self.device.torch_device)
                .to(self._model_dtype)
            )
            preds = self._model(tensor)[-1].sigmoid()
            pred = F.interpolate(
                preds.float(),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )[0, 0].clamp(0.0, 1.0).cpu().numpy()

        alpha = pred.astype(np.float32, copy=False)
        if return_alpha:
            return alpha
        return alpha >= threshold

    def _mask_image_anime(self, image: Image.Image) -> np.ndarray:
        """ISNet anime-seg inference. Mirrors the upstream skytnt demo:
        aspect-preserving resize to 1024, zero-pad to square, [0, 1] floats
        with NO ImageNet normalisation, NCHW input named ``img``."""
        self._ensure_loaded()  # routes to _ensure_loaded_anime for this variant
        self._touch()
        assert self._onnx_session is not None and self._onnx_input is not None

        rgb = image.convert("RGB")
        w, h = rgb.size
        side = 1024
        ratio = side / max(w, h)
        nw = max(1, round(w * ratio))
        nh = max(1, round(h * ratio))
        resized = np.asarray(rgb.resize((nw, nh), Image.LANCZOS), dtype=np.float32) / 255.0

        canvas = np.zeros((side, side, 3), dtype=np.float32)
        px, py = (side - nw) // 2, (side - nh) // 2
        canvas[py:py + nh, px:px + nw] = resized

        inp = canvas.transpose(2, 0, 1)[np.newaxis]
        pred = self._onnx_session.run(None, {self._onnx_input: inp})[0]
        pred = np.squeeze(pred).astype(np.float32)   # (1024, 1024) in [0, 1]

        crop = np.ascontiguousarray(pred[py:py + nh, px:px + nw])
        out = Image.fromarray(crop, mode="F").resize((w, h), Image.BILINEAR)
        return np.clip(np.asarray(out, dtype=np.float32), 0.0, 1.0)

    def mask_images(self, images: list[Image.Image], threshold: float = 0.5) -> np.ndarray:
        """Run BiRefNet on a list of frames sequentially. Returns ``(T, H, W)``
        bool where H/W are the first frame's size — all frames must share the
        same size (which is the case when they come from
        :func:`extract_clip_frames`)."""
        if not images:
            return np.empty((0, 0, 0), dtype=np.float32)
        first = self.mask_image(images[0], threshold=threshold)
        out = np.zeros((len(images), *first.shape), dtype=first.dtype)
        out[0] = first
        for i, img in enumerate(images[1:], start=1):
            out[i] = self.mask_image(img, threshold=threshold)
        return out
