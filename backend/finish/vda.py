"""Temporally consistent depth with Video Depth Anything (Small: Apache-2.0;
Large: CC-BY-NC-4.0 weights, same network code, "Max quality" tier).

Same windowing as the upstream ``infer_video_depth`` (32-frame windows that
advance by 22 frames; the first 10 inputs of a window are key frames of the
previous one; consecutive windows are aligned by a least-squares
scale/shift on two key frames and cross-faded over 8 frames), but streamed
and kept on the GPU at model resolution: frames are resized/normalised on
the device, and only model-resolution maps leave it. Upstream upsamples
every map to the source size and aligns on the CPU, which made it ~5x
slower at 1080p for the same result.

Measured (chantier 4, 18 calm/still shots): frame-to-frame flicker 0.24 %
of the depth range before any smoothing, vs 1.15 % for Depth Anything V2
Small (0.51 % after the app's flow smoothing).
"""
from __future__ import annotations

import logging
import math
from typing import Iterable, Iterator

import numpy as np

log = logging.getLogger(__name__)

VDA_MODEL = "video-depth-anything-small"
# registry id -> (encoder, features, out_channels, checkpoint file); upstream
# ``run.py`` model_configs
VDA_VARIANTS = {
    VDA_MODEL: ("vits", 64, (48, 96, 192, 384), "video_depth_anything_vits.pth"),
    "video-depth-anything-base": ("vitb", 128, (96, 192, 384, 768), "video_depth_anything_vitb.pth"),
    "video-depth-anything-large": ("vitl", 256, (256, 512, 1024, 1024), "video_depth_anything_vitl.pth"),
}
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def model_size(h: int, w: int, input_size: int = 518) -> tuple[int, int]:
    """Upstream ``Resize(lower_bound, ensure_multiple_of=14)`` (16:9 and
    narrower; wider frames get a smaller input like upstream)."""
    if max(h, w) / min(h, w) > 1.78:
        input_size = round(int(input_size * 1.777 / (max(h, w) / min(h, w))) / 14) * 14
    scale = max(input_size / h, input_size / w)

    def fit(x: float) -> int:
        y = round(x / 14) * 14
        return y if y >= input_size else math.ceil(x / 14) * 14
    return fit(h * scale), fit(w * scale)


def _scale_shift(pred, target):
    """Least-squares a*pred + b = target (torch, all pixels)."""
    a00 = (pred * pred).sum()
    a01 = pred.sum()
    a11 = float(pred.numel())
    b0 = (pred * target).sum()
    b1 = target.sum()
    det = a00 * a11 - a01 * a01
    if float(det) == 0.0:
        return 1.0, 0.0
    return float((a11 * b0 - a01 * b1) / det), float((-a01 * b0 + a00 * b1) / det)


class VideoDepthEstimator:
    """Streaming depth for a clip; yields ``(frame, depth)`` like
    :meth:`DepthEstimator.predict_stream` (depth at model resolution)."""

    temporal = True

    def __init__(self, device, *, input_size: int = 518, model_id: str = VDA_MODEL, weights=None):
        import torch

        from . import registry
        from .archs.vda import VideoDepthAnything

        encoder, features, out_channels, fname = VDA_VARIANTS[model_id]
        path = weights or registry.require(model_id) / fname
        net = VideoDepthAnything(encoder=encoder, features=features, out_channels=out_channels)
        sd = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(sd, strict=True)
        self.device = device
        self.input_size = int(input_size)
        self.net = net.eval().to(device.torch_device)
        # fp16 autocast on CUDA like upstream; fp32 elsewhere
        self.autocast = device.backend == "cuda"
        dev = device.torch_device
        self._mean = torch.tensor(_MEAN, device=dev).view(1, 3, 1, 1)
        self._std = torch.tensor(_STD, device=dev).view(1, 3, 1, 1)

    def _prep(self, frame: np.ndarray):
        import torch
        import torch.nn.functional as F

        h, w = frame.shape[:2]
        mh, mw = model_size(h, w, self.input_size)
        x = torch.from_numpy(np.ascontiguousarray(frame)).to(self.device.torch_device)
        x = x.permute(2, 0, 1)[None].float() / 255.0
        x = F.interpolate(x, size=(mh, mw), mode="bicubic", align_corners=False).clamp_(0, 1)
        return ((x - self._mean) / self._std)[0]

    def _window(self, inputs: list):
        import torch

        x = torch.stack(inputs)[None]
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=self.autocast):
            d = self.net(x)
        return d[0].float()

    def predict_stream(self, frames: Iterable[np.ndarray]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        from .archs.vda import INFER_LEN, INTERP_LEN, KEYFRAMES, OVERLAP

        step = INFER_LEN - OVERLAP                   # 22 new frames per window
        it = iter(frames)

        def take(n: int) -> list:
            got = []
            for f in it:
                got.append(f)
                if len(got) == n:
                    break
            return got

        def pad(xs: list, n: int) -> list:
            return xs + [xs[-1]] * (n - len(xs))

        first = take(INFER_LEN)
        if not first:
            return
        cur = pad([self._prep(f) for f in first], INFER_LEN)
        d = self._window(cur)
        if len(first) < INFER_LEN:
            for i, f in enumerate(first):
                yield f, d[i].cpu().numpy()
            return
        keep = INFER_LEN - INTERP_LEN                # maps final once the next window is known
        for i in range(keep):
            yield first[i], d[i].cpu().numpy()
        tail_frames, tail = first[keep:], [d[i] for i in range(keep, INFER_LEN)]
        ref = [d[0], d[KEYFRAMES[1]]]                # long-term anchor (frame 0) + previous key frame
        prev = cur
        weights = [i / (INTERP_LEN - 1) for i in range(INTERP_LEN)]
        while True:
            new = take(step)
            if not new:
                for f, m in zip(tail_frames, tail):
                    yield f, m.cpu().numpy()
                return
            x = [prev[k] for k in KEYFRAMES] + pad([self._prep(f) for f in new], step)
            d = self._window(x)
            a, b = _scale_shift(torch_cat([d[0], d[1]]), torch_cat(ref))
            # inputs 2..9 are the previous window's last 8 frames: cross-fade
            for i, f in enumerate(tail_frames):
                post = (d[OVERLAP - INTERP_LEN + i] * a + b).clamp_(min=0)
                yield f, (tail[i] * (1 - weights[i]) + post * weights[i]).cpu().numpy()
            maps = [(d[i] * a + b).clamp_(min=0) for i in range(OVERLAP, INFER_LEN)]
            ref = [ref[0], maps[KEYFRAMES[1] - OVERLAP]]
            if len(new) < step:
                for f, m in zip(new, maps):
                    yield f, m.cpu().numpy()
                return
            for i in range(step - INTERP_LEN):
                yield new[i], maps[i].cpu().numpy()
            tail_frames, tail = new[step - INTERP_LEN:], maps[step - INTERP_LEN:]
            prev = x

    # predict() (estimates only) runs one window, which the stream amortises
    # over this many new frames
    frames_per_predict = 22

    def predict(self, frame: np.ndarray) -> np.ndarray:
        x = self._prep(frame)
        return self._window([x] * 32)[0].cpu().numpy()

    def close(self) -> None:
        pass


def torch_cat(maps):
    import torch
    return torch.cat([m.reshape(-1) for m in maps])
