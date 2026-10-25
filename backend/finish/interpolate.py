"""Frame interpolation with RIFE 4.25 (PyTorch, CUDA/MPS/CPU).

Integration choice: native PyTorch, not ONNX. The 4.25 IFNet is ~250 lines,
has no custom CUDA ops (``grid_sample`` only) and loads the official
``flownet_v4.25.pkl`` directly, so the same weights run on CUDA, MPS and CPU
with no conversion step. The ONNX exports floating around are per-resolution
or need opset-16 ``GridSample`` and an extra 100+ MB runtime path for no
speed gain on CPU — not worth it here.

Two modes, both expressed as a list of *source positions* (fractional frame
indices) to render:

- ``slowmo``  x2/x4/x8 — N output frames per source frame, played back at the
  source fps (the clip lasts N times longer; audio is muted or stretched).
- ``fps``     e.g. 60 — output frames on the target fps grid over the same
  duration (audio kept as-is).

``skip_cuts``: RIFE morphs two unrelated shots into a ghostly blend at a hard
cut. When a scene change is detected between frames i and i+1, every
in-between position is snapped to the nearest real frame instead
(duplication), which is what an editor wants at a cut.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

RIFE_MODEL = "rife-4.25"
CUT_THRESHOLD = 0.30


def scene_change_score(a: np.ndarray, b: np.ndarray) -> float:
    """0..1 dissimilarity of two RGB frames, robust to motion.

    Combines the luma histogram distance (what changes at a cut: the whole
    palette) with a coarse mean absolute difference (catches cuts between
    shots of similar palette). Fast action inside one shot moves pixels but
    keeps the histogram, so it stays well under the threshold.
    """
    import cv2

    def small_gray(x):
        g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        return cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA)

    ga, gb = small_gray(a), small_gray(b)
    ha = np.histogram(ga, bins=32, range=(0, 256))[0].astype(np.float32)
    hb = np.histogram(gb, bins=32, range=(0, 256))[0].astype(np.float32)
    ha /= max(1.0, ha.sum())
    hb /= max(1.0, hb.sum())
    hist = 0.5 * float(np.abs(ha - hb).sum())           # total variation, 0..1
    mad = float(np.abs(ga.astype(np.float32) - gb.astype(np.float32)).mean()) / 255.0
    return min(1.0, hist + mad)


def detect_cuts(frames: list[np.ndarray], threshold: float = CUT_THRESHOLD) -> set[int]:
    """Indices i such that there is a cut between frame i and i+1."""
    return {i for i in range(len(frames) - 1) if scene_change_score(frames[i], frames[i + 1]) > threshold}


@dataclass
class TimelinePlan:
    positions: list[float]     # fractional source indices, one per output frame
    out_fps: float
    audio_speed: float         # 1.0 = keep, <1 = stretched (slowmo)


def plan_timeline(n_frames: int, src_fps: float, *, mode: str, factor: int = 2,
                  target_fps: float = 60.0) -> TimelinePlan:
    if n_frames <= 0:
        return TimelinePlan([], src_fps, 1.0)
    if mode == "slowmo":
        if factor not in (2, 4, 8):
            raise ValueError("slow-motion factor must be 2, 4 or 8")
        # Positions k/N for every output slot of every source frame; the last
        # N-1 slots past the final frame collapse onto it (duplicates).
        positions = [min(k / factor, n_frames - 1) for k in range(n_frames * factor)]
        return TimelinePlan(positions, src_fps, 1.0 / factor)
    if mode == "fps":
        if target_fps <= src_fps + 1e-6:
            raise ValueError("target fps must be higher than the source fps")
        duration = n_frames / src_fps
        n_out = max(1, int(math.floor(duration * target_fps + 1e-6)))
        step = src_fps / target_fps
        positions = [min(k * step, n_frames - 1) for k in range(n_out)]
        return TimelinePlan(positions, target_fps, 1.0)
    raise ValueError(f"unknown interpolation mode {mode!r}")


def resolve_position(pos: float, cuts: set[int], *, eps: float = 1e-4) -> tuple[int, int, float]:
    """Map a fractional position to ``(i, j, t)``: blend frame i and j at t.

    ``j == i`` (and t == 0) means "copy frame i" — exact source frames and
    in-betweens that straddle a cut (snapped to the nearest side).
    """
    i = int(math.floor(pos + eps))
    t = pos - i
    if t < eps:
        return i, i, 0.0
    if i in cuts:
        return (i, i, 0.0) if t < 0.5 else (i + 1, i + 1, 0.0)
    return i, i + 1, t


# Registry id -> IFNet variant (see ``archs.rife.VARIANTS``).
RIFE_VARIANTS = {
    "rife-4.25": "4.25",
    "rife-4.26": "4.26",
    "rife-4.25-lite": "4.25.lite",
    "rife-4.26-heavy": "4.26.heavy",
}


class RifeInterpolator:
    def __init__(self, device, *, scale: float = 1.0, model_id: str = RIFE_MODEL):
        import torch

        from . import registry
        from .archs.rife import VARIANTS, IFNet

        if model_id not in RIFE_VARIANTS:
            raise ValueError(f"unknown interpolation model {model_id!r}")
        d = registry.require(model_id)
        variant = RIFE_VARIANTS[model_id]
        sd = torch.load(d / registry.MODELS[model_id].files[0].name, map_location="cpu", weights_only=True)
        # Drop the training-only distillation/time heads; keep strict
        # loading for everything the inference graph actually uses.
        sd = {k.replace("module.", ""): v for k, v in sd.items()
              if not k.startswith(("module.teacher.", "module.caltime."))}
        net = IFNet(scale=scale, variant=variant)
        net.load_state_dict(sd, strict=True)
        self.model_id = model_id
        self.device = device
        # fp16 on CUDA only: grid_sample/flow accumulation in half precision
        # on MPS produces visible warping errors.
        self.dtype = torch.float16 if device.backend == "cuda" else torch.float32
        self.net = net.eval().to(device.torch_device, self.dtype)
        self.scale = scale
        modulo = VARIANTS[variant]["modulo"]
        self.modulo = max(modulo, int(modulo / scale))
        self._model_dir = str(d)
        # Apple Silicon: the same net through Core ML, 2x faster than MPS and
        # identical output (see ``coreml``); pairs go there once its worker
        # is up.
        from . import coreml
        self._use_coreml = device.backend == "mps" and coreml.available()
        self._worker = None
        self._worker_shape = None

    def _coreml_ready(self, h: int, w: int) -> bool:
        if not self._use_coreml:
            return False
        from . import coreml
        ph = math.ceil(h / self.modulo) * self.modulo
        pw = math.ceil(w / self.modulo) * self.modulo
        if self._worker is None or self._worker_shape != (ph, pw):
            if self._worker is not None:
                self._worker.close()
            self._worker = coreml.Worker("backend.finish.coreml:rife_handler", self._model_dir, ph, pw,
                                         float(self.scale), name="coreml-rife")
            self._worker_shape = (ph, pw)
        if self._worker.failed:
            log.warning("Core ML RIFE stopped (%s); back to MPS", self._worker.failed)
            self._use_coreml = False
            return False
        return self._worker.usable()

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def _to_tensor(self, frame: np.ndarray):
        import torch
        import torch.nn.functional as F

        h, w = frame.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(frame)).to(self.device.torch_device)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.dtype) / 255.0
        ph = math.ceil(h / self.modulo) * self.modulo
        pw = math.ceil(w / self.modulo) * self.modulo
        if (ph, pw) != (h, w):
            x = F.pad(x, (0, pw - w, 0, ph - h), mode="replicate")
        return x

    def make_pair(self, a: np.ndarray, b: np.ndarray):
        """Pre-encode a frame pair once; :meth:`at` then renders any t."""
        import torch

        if self._coreml_ready(*a.shape[:2]):
            return ("coreml", a, b)
        with torch.inference_mode():
            ta, tb = self._to_tensor(a), self._to_tensor(b)
            return ta, tb, self.net.encode(ta), self.net.encode(tb), a.shape[:2]

    def at(self, pair, t: float, i420: bool = False):
        """Frame at ``t`` (RGB uint8; ``i420`` on CUDA with even sizes: a
        :class:`yuv_cuda.I420Frame` converted on the GPU)."""
        import torch

        if isinstance(pair[0], str) and pair[0] == "coreml":
            try:
                return self._worker.call([pair[1], pair[2], np.array([t], dtype=np.float32)])[0]
            except Exception as exc:
                log.warning("Core ML RIFE stopped (%s); back to MPS", exc)
                self._use_coreml = False
                pair = self.make_pair(pair[1], pair[2])
        ta, tb, fa, fb, (h, w) = pair
        with torch.inference_mode():
            y = self.net(ta, tb, t, fa, fb)[:, :, :h, :w].float().clamp_(0.0, 1.0)
            rgb = (y[0].permute(1, 2, 0) * 255.0).round()
            if i420 and self.device.backend == "cuda" and h % 2 == 0 and w % 2 == 0:
                from .yuv_cuda import rgb255_to_i420
                return rgb255_to_i420(rgb)
            return rgb.byte().cpu().numpy()


def stream_interpolated(frames, step: float, *, make_pair, at,
                        is_cut: Optional[Callable[[np.ndarray, np.ndarray, int], bool]] = None,
                        on_cut: Optional[Callable[[int], None]] = None):
    """Yield output frames for source positions ``0, step, 2*step, …`` while
    reading ``frames`` (any iterable) once — only the current pair is held in
    memory, so long or 4K clips do not need the whole clip in RAM.

    ``step`` is ``1/factor`` (slow motion) or ``src_fps/target_fps``. The
    segment between frames i and i+1 is a cut when ``is_cut(a, b, i)``; its
    in-betweens are snapped to the nearest real frame (see
    :func:`resolve_position`). Trailing positions past the last frame
    duplicate it, which matches :func:`plan_timeline`.
    """
    eps = 1e-6
    k = 0
    prev = None
    i = -1
    cuts: set[int] = set()
    for frame in frames:
        i += 1
        if prev is not None:
            seg = i - 1
            if is_cut is not None and is_cut(prev, frame, seg):
                cuts.add(seg)
                if on_cut:
                    on_cut(seg)
            pair = None
            while k * step < i - eps:
                a, b, t = resolve_position(k * step, cuts)
                if a == b:
                    yield prev if a == seg else frame
                else:
                    if pair is None:
                        pair = make_pair(prev, frame)
                    yield at(pair, t)
                k += 1
        prev = frame
    n = i + 1
    while prev is not None and k * step < n - eps:
        yield prev
        k += 1
