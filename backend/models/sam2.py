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
# Soft alpha = sigmoid(SOFT_LOGIT_SCALE * logits). SAM 2's logits come from a
# 256² decoder upsampled to the frame, so a plain sigmoid gives a ~3 px blur;
# x2 keeps an anti-aliased edge while matching the true boundary better
# (Mask Mode benchmark: edge error -13%, flicker -22% vs. hard masks).
SOFT_LOGIT_SCALE = 2.0
# Frames per tracking window. SAM 2 holds each frame of its state at 1024²
# float32 (~12.6 MB) plus memory features: up to 96 frames (4 s at 24 fps,
# ~1.2 GB) the whole clip is one state, as in 0.4; longer clips are tracked
# window by window (frames offloaded to the CPU) so memory stays bounded
# whatever the clip length — this was the old ~4 s ceiling.
WINDOW_FRAMES = 96
# User keyframes outside the current window that are carried into it as
# conditioning frames (the nearest ones first).
MAX_ANCHORS = 3

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
        self._state_cache = None  # ((video_dir, n_frames), SAM 2 inference state)
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
            self._state_cache = None
        gc.collect()
        empty_device_cache(self.device.backend)

    # ── single-frame preview ────────────────────────────────────────────────
    def preview_mask(
        self,
        frame: Image.Image,
        positive_points: List[Tuple[int, int]],
        negative_points: Optional[List[Tuple[int, int]]] = None,
        box: Optional[Tuple[int, int, int, int]] = None,
        *,
        soft: bool = False,
    ) -> np.ndarray:
        """Run SAM 2 on a single frame and return a boolean mask (H, W)
        (``soft=True``: float32 sigmoid alpha).

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
                return_logits=soft,
            )
        # Pick the highest-score proposal among the multimask output.
        best_idx = int(np.argmax(scores))
        if soft:
            logits = masks[best_idx].astype(np.float32) * SOFT_LOGIT_SCALE
            return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
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
        soft: bool = True,
    ) -> np.ndarray:
        """Single-keyframe convenience wrapper around :meth:`track_keyframes`."""
        n = len(frame_paths)
        ref = n // 2 if reference_frame_offset is None else reference_frame_offset
        return self.track_keyframes(
            [{"frame_idx": ref, "positive": positive_points, "negative": negative_points or [], "box": box}],
            progress_cb=progress_cb, frames_dir=frames_dir, frame_paths=frame_paths,
            frame_size=frame_size, soft=soft,
        )

    def track_keyframes(
        self,
        keyframes: List[dict],
        progress_cb: Optional[Callable[[int, int], None]] = None,
        *,
        frames_dir: Path,
        frame_paths: List[Path],
        frame_size: Tuple[int, int],
        soft: bool = True,
        window: int = 0,
    ) -> np.ndarray:
        """Track one object through the clip from prompts on several frames.

        ``window`` (frames, default :data:`WINDOW_FRAMES`): clips longer than
        this are tracked in windows so memory stays bounded whatever the clip
        length (SAM 2 otherwise keeps every frame at 1024² plus its memory
        features — the old ~4 s ceiling).

        Each keyframe is ``{"frame_idx", "positive", "negative", "box",
        "mask"}``. ``mask`` (H, W) — typically a brush-corrected matte — is
        fed as a SAM 2 mask prompt and wins over points on that frame. All
        keyframes are conditioning memory for every frame, so a correction
        on frame 40 also repairs frames 30-50 on re-propagation.

        Propagation runs forward from the earliest keyframe to the end, then
        backward from the earliest keyframe to frame 0 (the old code only
        went forward, so every frame before the reference stayed empty).

        ``soft=True`` returns ``sigmoid(2 * logits)`` float32 alpha (1-2 px of
        real falloff on edges instead of a staircase); ``False`` returns the
        legacy boolean masks.
        """
        import torch

        self._ensure_video_loaded()
        self._touch()
        assert self._video_predictor is not None

        w, h = frame_size
        n_frames = len(frame_paths)
        window = int(window or WINDOW_FRAMES)
        if n_frames == 0:
            raise RuntimeError("SAM 2: no frames extracted for this clip")
        kfs = [k for k in keyframes if k.get("mask") is not None or k.get("positive") or k.get("box") is not None]
        if not kfs:
            raise ValueError("SAM 2 needs at least one keyframe with a click, a box or a mask")

        # SAM 2's init_state only accepts MP4s or folders of JPEG frames.
        # Session frames are PNG (lossless for the matting models), so mirror
        # them once into a JPEG subdir for the propagation pass.
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

        out_dtype = np.float32 if soft else bool
        masks_out = np.zeros((n_frames, h, w), dtype=out_dtype)
        if n_frames > window:
            # Long clip: bounded memory (see _track_windowed). The short-clip
            # path below is kept as is — it is what every clip under ~2 s used.
            return self._track_windowed(kfs, video_dir, n_frames, masks_out, soft=soft,
                                        window=window, progress_cb=progress_cb)
        seen = 0

        def emit(frame_idx: int, mask_logits) -> None:
            nonlocal seen
            logits = mask_logits[0, 0].float()
            if soft:
                masks_out[frame_idx] = torch.sigmoid(logits * SOFT_LOGIT_SCALE).cpu().numpy()
            else:
                masks_out[frame_idx] = (logits > 0.0).cpu().numpy()
            seen += 1
            if progress_cb is not None:
                try:
                    progress_cb(min(seen, n_frames), n_frames)
                except Exception:
                    pass

        with torch.inference_mode(), _autocast(self.device):
            # Re-tracking the same clip (Auto re-seeding, Manual
            # re-propagation) reuses the decoded frames of the last state.
            cached = self._state_cache
            if cached is not None and cached[0] == (str(video_dir), n_frames):
                state = cached[1]
                self._video_predictor.reset_state(state)
            else:
                self._state_cache = None
                state = self._video_predictor.init_state(video_path=str(video_dir))
                self._state_cache = ((str(video_dir), n_frames), state)
            for kf in sorted(kfs, key=lambda k: int(k["frame_idx"])):
                idx = max(0, min(n_frames - 1, int(kf["frame_idx"])))
                if kf.get("mask") is not None:
                    m = np.asarray(kf["mask"])
                    m = (m > 0.5) if m.dtype != bool else m
                    self._video_predictor.add_new_mask(
                        inference_state=state, frame_idx=idx, obj_id=1, mask=m,
                    )
                    continue
                coords, labels = _stack_prompts(kf.get("positive") or [], kf.get("negative") or [])
                kwargs = dict(inference_state=state, frame_idx=idx, obj_id=1)
                if coords is not None:
                    kwargs["points"] = coords
                    kwargs["labels"] = labels
                if kf.get("box") is not None:
                    kwargs["box"] = np.asarray(kf["box"], dtype=np.float32)
                self._video_predictor.add_new_points_or_box(**kwargs)

            first = min(max(0, min(n_frames - 1, int(k["frame_idx"]))) for k in kfs)
            for frame_idx, _obj_ids, mask_logits in self._video_predictor.propagate_in_video(
                state, start_frame_idx=first,
            ):
                emit(frame_idx, mask_logits)
            if first > 0:
                for frame_idx, _obj_ids, mask_logits in self._video_predictor.propagate_in_video(
                    state, start_frame_idx=first, reverse=True,
                ):
                    if frame_idx == first:
                        continue  # already emitted by the forward pass
                    emit(frame_idx, mask_logits)
            self._video_predictor.reset_state(state)

        return masks_out

    def release_state(self) -> None:
        """Drop the cached tracking state (decoded frames) of the last clip."""
        self._state_cache = None

    def _add_prompt(self, state, local_idx: int, kf: dict) -> None:
        if kf.get("mask") is not None:
            m = np.asarray(kf["mask"])
            m = (m > 0.5) if m.dtype != bool else m
            self._video_predictor.add_new_mask(inference_state=state, frame_idx=local_idx, obj_id=1, mask=m)
            return
        coords, labels = _stack_prompts(kf.get("positive") or [], kf.get("negative") or [])
        kwargs = dict(inference_state=state, frame_idx=local_idx, obj_id=1)
        if coords is not None:
            kwargs["points"] = coords
            kwargs["labels"] = labels
        if kf.get("box") is not None:
            kwargs["box"] = np.asarray(kf["box"], dtype=np.float32)
        self._video_predictor.add_new_points_or_box(**kwargs)

    def _track_windowed(self, kfs: List[dict], video_dir: Path, n_frames: int, masks_out: np.ndarray, *,
                        soft: bool, window: int,
                        progress_cb: Optional[Callable[[int, int], None]]) -> np.ndarray:
        """Streaming propagation for long clips, memory bounded by ``window``.

        The clip is walked forward from the earliest keyframe, then backward
        from it, ``window`` frames at a time. Each window is a small SAM 2
        video made of (a) up to :data:`MAX_ANCHORS` user keyframes lying
        outside the window, with their prompts — SAM 2 attends to
        conditioning frames regardless of their distance, so a correction
        still steers the whole clip —, (b) the last frame of the previous
        window with its mask as a mask prompt (the hand-over), (c) the
        window's own frames. Only (c) is emitted."""
        import os
        import shutil
        import tempfile

        import torch

        by_idx: dict[int, dict] = {}
        for kf in kfs:
            idx = max(0, min(n_frames - 1, int(kf["frame_idx"])))
            if idx in by_idx and by_idx[idx].get("mask") is not None:
                continue  # a brush correction wins over clicks on the same frame
            by_idx[idx] = kf
        first = min(by_idx)
        orders = [list(range(first, n_frames))]
        if first > 0:
            orders.append(list(range(first, -1, -1)))
        done = 0

        def jpg(i: int) -> Path:
            return video_dir / f"{i:05d}.jpg"

        for order in orders:
            carry: Optional[Tuple[int, np.ndarray]] = None
            for start in range(0, len(order), window):
                chunk = order[start:start + window]
                in_chunk = set(chunk)
                anchors = sorted((k for k in by_idx if k not in in_chunk),
                                 key=lambda k: abs(k - chunk[0]))[:MAX_ANCHORS]
                seq = list(anchors) + ([carry[0]] if carry is not None else []) + chunk
                tmp = Path(tempfile.mkdtemp(prefix="amv_sam2w_"))
                try:
                    for j, i in enumerate(seq):
                        dst = tmp / f"{j:05d}.jpg"
                        try:
                            os.link(jpg(i), dst)
                        except OSError:
                            shutil.copyfile(jpg(i), dst)
                    with torch.inference_mode(), _autocast(self.device):
                        state = self._video_predictor.init_state(video_path=str(tmp), offload_video_to_cpu=True)
                        for j, k in enumerate(anchors):
                            self._add_prompt(state, j, by_idx[k])
                        off = len(anchors)
                        if carry is not None:
                            if carry[1].any():
                                self._add_prompt(state, off, {"mask": carry[1]})
                            off += 1
                        for j, i in enumerate(chunk):
                            if i in by_idx:
                                self._add_prompt(state, off + j, by_idx[i])
                        if not anchors and carry is None and chunk[0] not in by_idx:
                            raise RuntimeError("SAM 2 window without any prompt")
                        has_prompt = bool(anchors) or (carry is not None and carry[1].any()) or any(
                            i in by_idx for i in chunk)
                        last_logits = None
                        if has_prompt:
                            for local, _ids, logits in self._video_predictor.propagate_in_video(state, start_frame_idx=0):
                                if local < off:
                                    continue
                                i = chunk[local - off]
                                lg = logits[0, 0].float()
                                if soft:
                                    masks_out[i] = torch.sigmoid(lg * SOFT_LOGIT_SCALE).cpu().numpy()
                                else:
                                    masks_out[i] = (lg > 0.0).cpu().numpy()
                                if i == chunk[-1]:
                                    last_logits = (lg > 0.0).cpu().numpy()
                        self._video_predictor.reset_state(state)
                        del state
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
                carry = (chunk[-1], last_logits if last_logits is not None else np.zeros(masks_out.shape[1:], bool))
                done += len(chunk) - (1 if (order is not orders[0] and start == 0) else 0)
                if progress_cb is not None:
                    try:
                        progress_cb(min(done, n_frames), n_frames)
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
