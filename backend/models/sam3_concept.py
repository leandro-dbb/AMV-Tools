"""SAM 3 concept segmentation ("anime character", "person") for Mask Mode.

SAM 3 (Meta, SAM License) finds every instance of a text concept in an image
and, in its video form, tracks them. Mask Mode uses it as a *character
detector*: it knows what a character is (an aura, a tractor or a pillar is
not one), while BiRefNet draws the finer outline. Weights are gated on
Hugging Face (``facebook/sam3``): they need the user's own approved account
token; without it the model is reported missing and Mask Mode keeps its
previous pipeline.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional, Sequence

import numpy as np

from .device import DeviceInfo

log = logging.getLogger(__name__)

REPO_ID = "facebook/sam3"
DEFAULT_PROMPTS = ("anime character",)
SCORE_THRESHOLD = 0.4


_ACCESS: dict = {}


def weights_accessible(hf_token: Optional[str] = None) -> bool:
    """Can the gated SAM 3 weights be fetched with this token (or the token
    huggingface_hub finds on its own)? Checked once per process."""
    if os.environ.get("AMV_SAM3_PATH"):
        return True
    key = hf_token or ""
    if key not in _ACCESS:
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(REPO_ID, "config.json", token=hf_token or None)
            _ACCESS[key] = True
        except Exception as exc:
            log.warning("SAM 3 weights not accessible (%s): accept the license on "
                        "huggingface.co/facebook/sam3 and set your Hugging Face token in Settings; "
                        "Mask Mode Auto runs without it", type(exc).__name__)
            _ACCESS[key] = False
    return _ACCESS[key]


def _source() -> str:
    """Local folder override (benchmarks, offline installs) or the Hub id."""
    return os.environ.get("AMV_SAM3_PATH") or REPO_ID


class Sam3Concept:
    """Lazy SAM 3 wrapper: image concept segmentation and video tracking."""

    def __init__(self, device: DeviceInfo, hf_token: Optional[str] = None):
        self.device = device
        self.hf_token = hf_token
        self._img = None
        self._img_proc = None
        self._vid = None
        self._vid_proc = None
        self._lock = threading.Lock()

    @property
    def _dtype(self):
        import torch
        return torch.bfloat16 if self.device.backend == "cuda" else torch.float32

    def _ensure_image(self):
        if self._img is not None:
            return
        with self._lock:
            if self._img is not None:
                return
            from transformers import Sam3Model, Sam3Processor
            log.info("loading SAM 3 (image)")
            self._img_proc = Sam3Processor.from_pretrained(_source(), token=self.hf_token)
            self._img = Sam3Model.from_pretrained(_source(), token=self.hf_token, dtype=self._dtype)
            self._img = self._img.to(self.device.torch_device).eval()

    def _ensure_video(self):
        if self._vid is not None:
            return
        with self._lock:
            if self._vid is not None:
                return
            from transformers import Sam3VideoModel, Sam3VideoProcessor
            log.info("loading SAM 3 (video)")
            self._vid_proc = Sam3VideoProcessor.from_pretrained(_source(), token=self.hf_token)
            self._vid = Sam3VideoModel.from_pretrained(_source(), token=self.hf_token, dtype=self._dtype)
            self._vid = self._vid.to(self.device.torch_device).eval()

    def offload(self):
        with self._lock:
            self._img = self._img_proc = self._vid = self._vid_proc = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def segment(self, rgb: np.ndarray, prompts: str | Sequence[str] = DEFAULT_PROMPTS,
                threshold: float = SCORE_THRESHOLD) -> np.ndarray:
        """Union of the instances of the concept(s) on one frame, float32 (H, W)."""
        import torch
        from PIL import Image
        self._ensure_image()
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        prompts = [q for p in prompts for q in p.split("|")]
        h, w = rgb.shape[:2]
        out = np.zeros((h, w), np.float32)
        img = Image.fromarray(rgb)
        for p in prompts:
            inputs = self._img_proc(images=img, text=p, return_tensors="pt").to(self.device.torch_device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)
            with torch.inference_mode():
                res = self._img(**inputs)
            r = self._img_proc.post_process_instance_segmentation(
                res, threshold=threshold, mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist())[0]
            m = r["masks"]
            if len(m):
                out = np.maximum(out, m.any(0).float().cpu().numpy())
        return out

    def segment_video(self, frames: Sequence[np.ndarray], prompts: str | Sequence[str] = DEFAULT_PROMPTS
                      ) -> np.ndarray:
        """Concept detection + SAM 3 tracking over a shot, (T, H, W) float32."""
        import torch
        self._ensure_video()
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        prompts = [q for p in prompts for q in p.split("|")]
        T = len(frames)
        h, w = frames[0].shape[:2]
        out = np.zeros((T, h, w), np.float32)
        dev = self.device.torch_device
        session = self._vid_proc.init_video_session(
            video=list(frames), inference_device=dev, processing_device="cpu",
            video_storage_device="cpu", dtype=self._dtype)
        session = self._vid_proc.add_text_prompt(inference_session=session, text=prompts)
        with torch.inference_mode():
            for mo in self._vid.propagate_in_video_iterator(inference_session=session, max_frame_num_to_track=T):
                pp = self._vid_proc.postprocess_outputs(session, mo)
                m = pp["masks"]
                if len(m):
                    out[mo.frame_idx] = m.any(0).float().cpu().numpy()
        del session
        return out
