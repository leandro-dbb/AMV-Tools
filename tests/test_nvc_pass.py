"""CPU tests for the NVDEC -> tensor single pass (``backend/indexing/nvc_pass``).
The NVDEC/NVENC parts need PyNvVideoCodec and a GPU (covered by tests/hw and
tests/bench_pipeline.py); here: the ffmpeg 5.1 scale_cuda emulation, the
-fps_mode cfr numbering and the fallback."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from backend.indexing import nvc_pass  # noqa: E402


def test_scale_cuda_emulation_is_the_2x2_mean_truncated_for_12x():
    """1920x1080 -> 160x90 with scale_cuda's default bicubic (A = 0): each
    output is the mean of the 2x2 texels at (12x+5..12x+6, 12y+5..12y+6),
    normalised k/255 to float32 and truncated — what ffmpeg 5.1 produces."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (2, 1080, 1920)).astype(np.uint8)
    sc = nvc_pass._Scaler(torch, (1920, 1080), (160, 90), torch.device("cpu"))
    got = sc(torch.from_numpy(img)).numpy()
    n = (np.arange(256, dtype=np.float64) / 255.0).astype(np.float32)
    f = n[img]
    a, b = f[:, 5::12, 5::12], f[:, 5::12, 6::12]
    c, d = f[:, 6::12, 5::12], f[:, 6::12, 6::12]
    half = np.float32(0.5)
    r1 = (a * half + b * half).astype(np.float32)
    r2 = (c * half + d * half).astype(np.float32)
    ref = ((r1 * half + r2 * half).astype(np.float32) * np.float32(255.0)).astype(np.uint8)
    assert got.shape == (2, 90, 160) and np.array_equal(got, ref)


def test_scale_handles_interleaved_chroma():
    uv = torch.full((1, 540, 960, 2), 128, dtype=torch.uint8)
    out = nvc_pass._Scaler(torch, (960, 540), (256, 144), torch.device("cpu"))(uv)
    assert out.shape == (1, 144, 256, 2) and int(out.min()) >= 127 and int(out.max()) <= 128


@pytest.mark.parametrize("times,expected", [
    ([0, 1, 2, 3], [1, 1, 1, 1]),            # regular
    ([0, 1, 1, 1, 1], [1, 1, 1, 1, 0]),      # drifting early: dropped past 1.1 frames
    ([0, 1, 4, 5], [1, 1, 3, 1]),            # a gap: duplicated
])
def test_cfr_numbering_follows_ffmpeg(times, expected):
    fps, sync, got = 24.0, 0, []
    for k in times:
        nb = nvc_pass.cfr_copies(k / fps, fps, sync)
        got.append(nb)
        sync += nb
    assert got == expected


def test_unavailable_without_pynvvideocodec(monkeypatch):
    monkeypatch.setattr(nvc_pass, "_NVC", False)
    assert not nvc_pass.available()
    assert nvc_pass.detect_cuts_and_medians("x.mkv", 0.4) is None
