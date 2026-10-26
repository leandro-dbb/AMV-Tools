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

    def segment_with_objects(self, rgb: np.ndarray, prompts: str | Sequence[str] = DEFAULT_PROMPTS,
                             objects: Sequence[str] = (), threshold: float = SCORE_THRESHOLD,
                             zoom: bool = False) -> tuple[np.ndarray, list]:
        """(character mask as :meth:`segment` / :meth:`segment_zoomed`, list
        of bool instance masks of the ``objects`` concepts — "staff",
        "halo"...). One image-encoder pass is shared by every prompt; only
        the light decoder runs once per prompt."""
        import torch
        from PIL import Image
        self._ensure_image()
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        prompts = [q for p in prompts for q in p.split("|")]
        h, w = rgb.shape[:2]
        dev = self.device.torch_device
        inputs = self._img_proc(images=Image.fromarray(rgb), return_tensors="pt").to(dev)
        sizes = inputs.get("original_sizes").tolist()
        full = np.zeros((h, w), np.float32)
        objs: list = []
        with torch.inference_mode():
            ve = self._img.get_vision_features(pixel_values=inputs["pixel_values"].to(self._dtype))
            for p in prompts + list(objects):
                ti = self._img_proc(text=p, return_tensors="pt").to(dev)
                res = self._img(vision_embeds=ve, input_ids=ti["input_ids"],
                                attention_mask=ti.get("attention_mask"))
                m = self._img_proc.post_process_instance_segmentation(
                    res, threshold=threshold, mask_threshold=0.5, target_sizes=sizes)[0]["masks"]
                if not len(m):
                    continue
                if p in prompts:
                    full = np.maximum(full, m.any(0).float().cpu().numpy())
                else:
                    objs.extend(x.cpu().numpy().astype(bool) for x in m)
        sel = self.segment_zoomed(rgb, prompts, threshold, full=full) if zoom else full
        return sel, objs

    def segment_zoomed(self, rgb: np.ndarray, prompts: str | Sequence[str] = DEFAULT_PROMPTS,
                       threshold: float = SCORE_THRESHOLD, margin: float = 0.12,
                       min_frac: float = 0.004, full: Optional[np.ndarray] = None) -> np.ndarray:
        """:meth:`segment`, then every character found is cropped (with a
        margin) and segmented again: SAM 3 works at a fixed 1008 px input, so
        a crop of one character gets 1.5-3x the resolution — finer outline,
        hair tips that the full-frame pass under-segments. Inside each crop
        the zoomed mask replaces the full-frame one; outside, the full-frame
        mask stays. ``full``: the full-frame mask when already computed."""
        import cv2
        if full is None:
            full = self.segment(rgb, prompts, threshold)
        m = (full > 0.5).astype(np.uint8)
        if not m.any():
            return full
        h, w = m.shape
        n, _lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        boxes = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < min_frac * h * w:
                continue
            mx, my = int(margin * bw) + 8, int(margin * bh) + 8
            boxes.append([max(0, x - mx), max(0, y - my), min(w, x + bw + mx), min(h, y + bh + my)])
        # merge overlapping crops
        merged = []
        for b in sorted(boxes):
            for mb in merged:
                if b[0] < mb[2] and b[2] > mb[0] and b[1] < mb[3] and b[3] > mb[1]:
                    mb[:] = [min(mb[0], b[0]), min(mb[1], b[1]), max(mb[2], b[2]), max(mb[3], b[3])]
                    break
            else:
                merged.append(list(b))
        out = full.copy()
        for x0, y0, x1, y1 in merged:
            if (x1 - x0) > 0.85 * w and (y1 - y0) > 0.85 * h:
                continue   # the crop is the frame: nothing to gain
            sub = self.segment(np.ascontiguousarray(rgb[y0:y1, x0:x1]), prompts, threshold)
            if (sub > 0.5).sum() < 0.3 * (m[y0:y1, x0:x1] > 0).sum():
                continue   # the zoomed pass lost the character: keep the full-frame mask
            out[y0:y1, x0:x1] = sub
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
