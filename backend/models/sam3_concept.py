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
# zoomed pass skipped only when the crop is (nearly) the whole frame; a
# stricter rule (crops <= 66 % of the frame) saved time but lost the dark
# close-up s19 (enclosed background 2.9 -> 14.5 %), see the report part 4
ZOOM_MAX_W = 0.85
ZOOM_MAX_H = 0.85


KEEP_LOADED_GB = 16.0  # cards with this much VRAM keep SAM 3 loaded (and compiled) between clips


def total_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 2**30
    except Exception:
        pass
    return 0.0


def default_batch() -> int:
    """Images per SAM 3 forward pass: 8 on cards with 12 GB or more, 2 below
    (the vision encoder at 1008 px is what fills the memory)."""
    try:
        import torch
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 2**30
            return 8 if gb >= 11.5 else 2
    except Exception:
        pass
    return 1
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
            self._eager_encoder = self._img.vision_encoder
            if (self.device.backend == "cuda" and os.environ.get("AMV_SAM3_COMPILE", "1") != "0"
                    and total_vram_gb() >= KEEP_LOADED_GB):
                # torch.compile of the 1008 px vision encoder: ~1.6x faster on
                # an RTX 3090, same masks; batches are always full (one shape,
                # one compilation, ~1 min once per load — on these cards the
                # model stays loaded between clips); eager on any error
                try:
                    import torch
                    self._img.vision_encoder = torch.compile(self._eager_encoder, dynamic=False)
                    self._compiled = True
                except Exception as exc:  # pragma: no cover - depends on the install
                    log.warning("SAM 3: torch.compile unavailable (%s), eager mode", exc)

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
            self._compiled = False
            self._eager_encoder = None
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

    def segment_zoomed(self, rgb: np.ndarray, prompts: str | Sequence[str] = DEFAULT_PROMPTS,
                       threshold: float = SCORE_THRESHOLD, margin: float = 0.12,
                       min_frac: float = 0.004) -> np.ndarray:
        """:meth:`segment`, then every character found is cropped (with a
        margin) and segmented again: SAM 3 works at a fixed 1008 px input, so
        a crop of one character gets 1.5-3x the resolution — finer outline,
        hair tips that the full-frame pass under-segments. Inside each crop
        the zoomed mask replaces the full-frame one; outside, the full-frame
        mask stays."""
        import cv2
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
            if (x1 - x0) > ZOOM_MAX_W * w and (y1 - y0) > ZOOM_MAX_H * h:
                continue   # the crop is the frame: nothing to gain
            sub = self.segment(np.ascontiguousarray(rgb[y0:y1, x0:x1]), prompts, threshold)
            if (sub > 0.5).sum() < 0.3 * (m[y0:y1, x0:x1] > 0).sum():
                continue   # the zoomed pass lost the character: keep the full-frame mask
            out[y0:y1, x0:x1] = sub
        return out

    def _text_embeds(self, prompt: str):
        """Text features of a prompt, computed once and reused for every frame."""
        cache = getattr(self, "_text_cache", None)
        if cache is None:
            cache = self._text_cache = {}
        if prompt not in cache:
            import torch
            ti = self._img_proc(text=prompt, return_tensors="pt").to(self.device.torch_device)
            with torch.inference_mode():
                te = self._img.get_text_features(input_ids=ti["input_ids"],
                                                 attention_mask=ti.get("attention_mask")).pooler_output
            cache[prompt] = (te, ti.get("attention_mask"))
        return cache[prompt]

    def segment_many(self, images: Sequence[np.ndarray], prompt: str = DEFAULT_PROMPTS[0],
                     threshold: float = SCORE_THRESHOLD, batch: Optional[int] = None) -> list:
        """:meth:`segment` for several images at once: the text is encoded
        once, the images go through the vision encoder in batches. Same
        masks as one call per image (float32, each image's own size)."""
        import torch
        from PIL import Image
        self._ensure_image()
        batch = batch or default_batch()
        te, am = self._text_embeds(prompt)
        out = []
        for i in range(0, len(images), batch):
            chunk = images[i:i + batch]
            real = len(chunk)
            if getattr(self, "_compiled", False) and real < batch:
                chunk = list(chunk) + [chunk[-1]] * (batch - real)   # one shape for the compiled encoder
            inputs = self._img_proc(images=[Image.fromarray(np.ascontiguousarray(c)) for c in chunk],
                                    return_tensors="pt").to(self.device.torch_device)
            pv = inputs["pixel_values"].to(self._dtype)
            n = pv.shape[0]
            kw = dict(text_embeds=te.expand(n, -1, -1), attention_mask=None if am is None else am.expand(n, -1))
            with torch.inference_mode():
                try:
                    res = self._img(pixel_values=pv, **kw)
                except Exception as exc:
                    if not getattr(self, "_compiled", False):
                        raise
                    log.warning("SAM 3: compiled encoder failed (%s), back to eager mode", exc)
                    self._img.vision_encoder = self._eager_encoder
                    self._compiled = False
                    res = self._img(pixel_values=pv, **kw)
            sizes = [c.shape[:2] for c in chunk]
            pp = self._img_proc.post_process_instance_segmentation(
                res, threshold=threshold, mask_threshold=0.5, target_sizes=[list(sz) for sz in sizes])
            for r, (hh, ww) in list(zip(pp, sizes))[:real]:
                m = r["masks"]
                out.append(m.any(0).float().cpu().numpy() if len(m) else np.zeros((hh, ww), np.float32))
        return out

    def segment_zoomed_many(self, frames: Sequence[np.ndarray], prompt: str = DEFAULT_PROMPTS[0],
                            threshold: float = SCORE_THRESHOLD, batch: Optional[int] = None, margin: float = 0.12,
                            min_frac: float = 0.004) -> list:
        """:meth:`segment_zoomed` for a list of frames, both passes batched
        (all the character crops of the frames go through SAM 3 together)."""
        import cv2
        fulls = self.segment_many(frames, prompt, threshold, batch)
        jobs = []   # (frame index, box)
        for t, (rgb, full) in enumerate(zip(frames, fulls)):
            m = (full > 0.5).astype(np.uint8)
            if not m.any():
                continue
            h, w = m.shape
            n, _lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            boxes = []
            for i in range(1, n):
                x, y, bw, bh, area = stats[i]
                if area < min_frac * h * w:
                    continue
                mx, my = int(margin * bw) + 8, int(margin * bh) + 8
                boxes.append([max(0, x - mx), max(0, y - my), min(w, x + bw + mx), min(h, y + bh + my)])
            merged = []
            for b in sorted(boxes):
                for mb in merged:
                    if b[0] < mb[2] and b[2] > mb[0] and b[1] < mb[3] and b[3] > mb[1]:
                        mb[:] = [min(mb[0], b[0]), min(mb[1], b[1]), max(mb[2], b[2]), max(mb[3], b[3])]
                        break
                else:
                    merged.append(list(b))
            for x0, y0, x1, y1 in merged:
                if (x1 - x0) > ZOOM_MAX_W * w and (y1 - y0) > ZOOM_MAX_H * h:
                    continue
                jobs.append((t, (x0, y0, x1, y1)))
        crops = [np.ascontiguousarray(frames[t][y0:y1, x0:x1]) for t, (x0, y0, x1, y1) in jobs]
        subs = self.segment_many(crops, prompt, threshold, batch) if crops else []
        outs = [f.copy() for f in fulls]
        for (t, (x0, y0, x1, y1)), sub in zip(jobs, subs):
            if (sub > 0.5).sum() < 0.3 * (fulls[t][y0:y1, x0:x1] > 0.5).sum():
                continue
            outs[t][y0:y1, x0:x1] = sub
        return outs

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
