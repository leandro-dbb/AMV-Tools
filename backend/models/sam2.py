"""SAM 2 loader — click → mask + temporal propagation for the roto/alpha feature.

Mirrors the lazy-load / offload pattern of `SigLIP2Model` and `WDTaggerModel` so
SAM 2 plays nice in the 2-phase pipeline. Defaults to the SAM 2.1 base+ variant
(~350 MB, ~1.5 GB VRAM peak) — best quality/VRAM trade-off for 8 GB GPUs once
SigLIP has been offloaded.

Two entry points:

- ``preview_mask`` — one-shot single-frame inference for the live UI preview
  when the user clicks on the median frame. Fast (~150-300 ms on a modern
  consumer GPU).
- ``predict_video_masks`` — full clip tracking over the session's extracted
  PNG frames (mirrored to JPEG for SAM 2's folder loader), runs propagation,
  returns a stack of binary masks aligned 1:1 with the extracted frames.

Checkpoints are pulled from HuggingFace (``facebook/sam2.1-hiera-*``) and cached
under the standard HF cache, same as SigLIP and the tagger.
"""
from __future__ import annotations

import logging
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image

from .device import DeviceInfo

log = logging.getLogger(__name__)


VARIANTS: dict[str, str] = {
    # Maps the user-facing variant name to the HuggingFace repo id. SAM 2.1 is
    # the refreshed checkpoint set Meta released in October 2024 — same arch as
    # the original SAM 2, retrained on more data, strictly better on every
    # public benchmark.
    "tiny":      "facebook/sam2.1-hiera-tiny",
    "small":     "facebook/sam2.1-hiera-small",
    "base_plus": "facebook/sam2.1-hiera-base-plus",
    "large":     "facebook/sam2.1-hiera-large",
}
DEFAULT_VARIANT = "base_plus"

# Prompts arrive as (x, y) in the coordinate frame of the rendered preview the
# user clicked on. We carry them straight through to SAM 2, which expects pixel
# coordinates in the source frame's resolution — the API expects callers to
# scale them to whatever frame size was loaded into the predictor.


class SAM2Model:
    """Lazy-loaded SAM 2 wrapper with separate image / video predictors.

    The image predictor and video predictor share weights internally but each
    holds its own `SAM2Base` instance in memory after `from_pretrained`. We
    lazy-init them independently so the cheap preview path doesn't pay for the
    video predictor's heavier state, and the expensive video path doesn't get
    spawned just for a click preview.
    """

    def __init__(
        self,
        device: DeviceInfo,
        variant: str = DEFAULT_VARIANT,
        idle_offload_seconds: int = 60,
        hf_token: Optional[str] = None,
    ):
        self.device = device
        if variant not in VARIANTS:
            log.warning("unknown SAM 2 variant %r, falling back to %r", variant, DEFAULT_VARIANT)
            variant = DEFAULT_VARIANT
        self.variant = variant
        self.repo_id = VARIANTS[variant]
        self.idle_offload_seconds = idle_offload_seconds
        self.hf_token = hf_token
        self._image_predictor = None
        self._video_predictor = None
        self._lock = threading.Lock()
        self._last_used = 0.0
        self._offload_thread: Optional[threading.Thread] = None
        self._stop_offload = threading.Event()

    # ── load / offload ──────────────────────────────────────────────────────
    def _ensure_image_loaded(self):
        if self._image_predictor is not None:
            return
        with self._lock:
            if self._image_predictor is not None:
                return
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            log.info("loading SAM 2 image predictor (%s)", self.repo_id)
            self._image_predictor = SAM2ImagePredictor.from_pretrained(
                self.repo_id, device=self.device.torch_device, token=self.hf_token,
            )
            self._start_offload_watcher()

    def _ensure_video_loaded(self):
        if self._video_predictor is not None:
            return
        with self._lock:
            if self._video_predictor is not None:
                return
            from sam2.sam2_video_predictor import SAM2VideoPredictor

            log.info("loading SAM 2 video predictor (%s)", self.repo_id)
            self._video_predictor = SAM2VideoPredictor.from_pretrained(
                self.repo_id, device=self.device.torch_device, token=self.hf_token,
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
                if self._image_predictor is None and self._video_predictor is None:
                    continue
                if time.monotonic() - self._last_used > self.idle_offload_seconds:
                    self.offload()

        self._offload_thread = threading.Thread(target=watch, daemon=True)
        self._offload_thread.start()

    def offload(self):
        """Free both predictors. Will lazy-reload on next use."""
        import gc
        from .device import empty_device_cache
        with self._lock:
            self._image_predictor = None
            self._video_predictor = None
        gc.collect()
        empty_device_cache(self.device.backend)

    # ── single-frame preview ────────────────────────────────────────────────
    def preview_mask(
        self,
        frame: Image.Image,
        positive_points: List[Tuple[int, int]],
        negative_points: Optional[List[Tuple[int, int]]] = None,
        box: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """Run SAM 2 on a single frame and return a boolean mask (H, W).

        ``positive_points`` and ``negative_points`` are pixel coordinates in the
        same frame's resolution. ``box`` is (x0, y0, x1, y1). Returns the
        highest-scoring mask among the three SAM 2 always proposes.
        """
        import torch

        self._ensure_image_loaded()
        self._touch()
        assert self._image_predictor is not None

        rgb = np.array(frame.convert("RGB"))
        coords, labels = _stack_prompts(positive_points, negative_points)
        box_arr = np.asarray(box, dtype=np.float32) if box else None

        with torch.inference_mode(), _autocast(self.device):
            self._image_predictor.set_image(rgb)
            masks, scores, _ = self._image_predictor.predict(
                point_coords=coords if coords is not None else None,
                point_labels=labels if labels is not None else None,
                box=box_arr,
                multimask_output=True,
            )
        # Pick the highest-score proposal among the multimask output.
        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(bool)

    # ── full clip tracking ──────────────────────────────────────────────────
    def predict_video_masks(
        self,
        positive_points: List[Tuple[int, int]],
        negative_points: Optional[List[Tuple[int, int]]] = None,
        box: Optional[Tuple[int, int, int, int]] = None,
        reference_frame_offset: Optional[int] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        *,
        frames_dir: Path,
        frame_paths: List[Path],
        frame_size: Tuple[int, int],
    ) -> np.ndarray:
        """Track a single object across an already-extracted JPG frame sequence.

        The caller (``/segment/track``) is the owner of ``frames_dir``: it was
        produced by ``/segment/preview`` via :func:`extract_clip_frames` and
        gets cleaned up by the DELETE endpoint. We do NOT re-extract here —
        that used to be a ~3 s + tempdir-leak per track.
        """
        import torch

        self._ensure_video_loaded()
        self._touch()
        assert self._video_predictor is not None

        w, h = frame_size
        n_frames = len(frame_paths)
        if n_frames == 0:
            raise RuntimeError("SAM 2: no frames extracted for this clip")

        # SAM 2's init_state only accepts MP4s or folders of JPEG frames.
        # Session frames are PNG (lossless for the matting models), so mirror
        # them once into a JPEG subdir for the propagation pass. quality=95
        # matches the old direct-JPEG extraction quality.
        video_dir = frames_dir
        if frame_paths[0].suffix.lower() == ".png":
            jpeg_dir = frames_dir / "sam2_jpeg"
            already_mirrored = jpeg_dir.exists() and len(list(jpeg_dir.glob("*.jpg"))) == n_frames
            if not already_mirrored:
                jpeg_dir.mkdir(exist_ok=True)
                for i, fp in enumerate(frame_paths):
                    with Image.open(fp) as im:
                        im.convert("RGB").save(jpeg_dir / f"{i:05d}.jpg", quality=95)
            video_dir = jpeg_dir

        if reference_frame_offset is None:
            reference_frame_offset = n_frames // 2
        reference_frame_offset = max(0, min(n_frames - 1, reference_frame_offset))

        coords, labels = _stack_prompts(positive_points, negative_points)
        box_arr = np.asarray(box, dtype=np.float32) if box else None

        with torch.inference_mode(), _autocast(self.device):
            state = self._video_predictor.init_state(video_path=str(video_dir))
            kwargs = dict(
                inference_state=state,
                frame_idx=reference_frame_offset,
                obj_id=1,
            )
            if coords is not None:
                kwargs["points"] = coords
                kwargs["labels"] = labels
            if box_arr is not None:
                kwargs["box"] = box_arr
            self._video_predictor.add_new_points_or_box(**kwargs)

            masks_out = np.zeros((n_frames, h, w), dtype=bool)
            seen = 0
            for frame_idx, _obj_ids, mask_logits in self._video_predictor.propagate_in_video(state):
                # mask_logits is (N_obj, 1, H, W) on device — threshold at 0,
                # take the first (and only) object, drop to CPU as bool.
                m = (mask_logits[0, 0] > 0.0).cpu().numpy()
                masks_out[frame_idx] = m
                seen += 1
                if progress_cb is not None:
                    try:
                        progress_cb(seen, n_frames)
                    except Exception:
                        pass

        return masks_out


# ── helpers ────────────────────────────────────────────────────────────────
def _stack_prompts(
    positive: List[Tuple[int, int]],
    negative: Optional[List[Tuple[int, int]]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Pack positive (label=1) and negative (label=0) points the way SAM 2
    wants them: ``(N, 2)`` float coords and ``(N,)`` int labels."""
    pts: list[Tuple[int, int]] = []
    lbls: list[int] = []
    for x, y in positive or []:
        pts.append((x, y))
        lbls.append(1)
    for x, y in negative or []:
        pts.append((x, y))
        lbls.append(0)
    if not pts:
        return None, None
    return (
        np.asarray(pts, dtype=np.float32),
        np.asarray(lbls, dtype=np.int32),
    )


@contextmanager
def _autocast(device: DeviceInfo):
    """SAM 2 wants bfloat16 autocast on CUDA — same recipe as the upstream
    notebooks. On CPU/MPS we just no-op."""
    import torch

    if device.backend == "cuda":
        try:
            major, _ = torch.cuda.get_device_capability()
        except Exception:
            major = 0
        # bfloat16 is preferred on Ampere+ (>=8), but half is fine on Volta/Turing.
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            yield
    else:
        yield


def extract_clip_frames(
    video_path: str,
    start_ms: int,
    end_ms: int,
    *,
    fps: float = 24.0,
    max_dim: int = 720,
) -> Tuple[Path, List[Path], Tuple[int, int]]:
    """Extract the clip's frames to a fresh tempdir as PNGs named ``00000.png``.

    ``fps`` must be the source's probed average rate — the caller owns the
    probe. The extraction goes through :func:`export.ffmpeg.extract_png_frames`,
    the same sampler the alpha export re-runs at full resolution, so mask
    index N and export frame N always come from the same source frame.

    PNG rather than JPG because the matting models and the colour-distance
    edge cleanups are both sensitive to compression artefacts around line
    work. (SAM 2's video predictor still needs JPEGs — see the shim in
    :meth:`SAM2Model.predict_video_masks`.)

    Returns ``(frames_dir, frame_paths, (w, h))`` where ``frames_dir`` is a
    freshly-created tempdir the caller must clean up.
    """
    from ..export.ffmpeg import extract_png_frames

    frames_dir = Path(tempfile.mkdtemp(prefix="amv_sam2_"))
    try:
        frame_paths = extract_png_frames(
            video_path, start_ms, end_ms, frames_dir / "%05d.png",
            fps=fps, max_dim=max_dim,
        )
    except Exception:
        cleanup_frames_dir(frames_dir)
        raise
    if not frame_paths:
        cleanup_frames_dir(frames_dir)
        raise RuntimeError("ffmpeg produced no frames for the mask session (check start/end ms)")

    # Read the first frame to get the post-resize resolution.
    with Image.open(frame_paths[0]) as im:
        w, h = im.size
    return frames_dir, frame_paths, (w, h)


def cleanup_frames_dir(frames_dir: Path) -> None:
    """Best-effort cleanup of an ``extract_clip_frames`` tempdir."""
    import shutil
    try:
        shutil.rmtree(frames_dir, ignore_errors=True)
    except Exception as exc:
        log.debug("cleanup_frames_dir(%s) failed: %s", frames_dir, exc)
