"""Shared pytest fixtures for the AMV Tools backend test-suite.

Everything here is CPU-only: synthetic clips are generated with ffmpeg and
the heavy ML models are replaced by deterministic fakes. The real model
wrappers are only exercised by their own integration paths, never by these
unit/flow tests.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

# `backend.paths` resolves the data dir at call time, so pointing it at a tmp
# directory must happen before any AppState is constructed.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
# The sidecar only serves Host 127.0.0.1/localhost (backend/security.py);
# TestClient sends Host: testserver.
os.environ.setdefault("AMV_ALLOWED_HOSTS", "testserver")


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        found = shutil.which("ffmpeg")
        if not found:
            raise RuntimeError("ffmpeg is required for the test-suite")
        return found


def make_solid_clip(
    path: Path,
    colors: Iterable[str] = ("red", "blue", "green"),
    *,
    seconds_per_color: float = 1.0,
    size: str = "320x240",
    fps: int = 25,
) -> Path:
    """Build an H.264 MP4 made of back-to-back solid-colour segments.

    Hard cuts between pure colours produce a large histogram distance, which is
    exactly what ``backend.indexing.cuts`` keys on. N colours therefore yield N
    scenes (within the 200 ms minimum-gap guard).
    """
    colors = list(colors)
    inputs: list[str] = []
    for color in colors:
        inputs += [
            "-f", "lavfi",
            "-i", f"color=c={color}:s={size}:d={seconds_per_color}:r={fps}",
        ]
    labels = "".join(f"[{i}:v]" for i in range(len(colors)))
    graph = f"{labels}concat=n={len(colors)}:v=1:a=0,format=yuv420p[v]"
    cmd = [
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        *inputs,
        "-filter_complex", graph,
        "-map", "[v]",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    assert path.exists() and path.stat().st_size > 0
    return path


@pytest.fixture
def data_dir(tmp_path, monkeypatch) -> Path:
    """Isolated AMV data dir + fresh AppState singleton for each test."""
    root = tmp_path / "amv-data"
    root.mkdir()
    monkeypatch.setenv("AMV_DATA_DIR", str(root))

    import backend.state as state_module

    state_module._state = None
    # A marker keeps the status endpoint out of the onboarding flow.
    (root / ".setup_complete").write_text("cpu")
    yield root
    state_module._state = None


class FakeTorchDevice:
    type = "cpu"
    index = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "cpu"


class FakeDevice:
    """Stand-in for ``backend.models.device.DeviceInfo`` (no torch import)."""

    backend = "cpu"
    name = "CPU (test)"
    torch_device = FakeTorchDevice()
    dtype = "float32"
    vram_gb = None


@pytest.fixture
def state(data_dir):
    import backend.state as state_module

    st = state_module.get_state()
    st._device = FakeDevice()
    return st


@pytest.fixture
def db_path(state) -> str:
    return state.primary_db


@pytest.fixture
def app(state):
    from backend.server import make_app

    return make_app()


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


@pytest.fixture
def clip(tmp_path) -> Path:
    return make_solid_clip(tmp_path / "three-cuts.mp4")


class FakeSigLIP:
    """Deterministic SigLIP stand-in: an embedding is a normalized RGB mean."""

    checkpoint = "fake/siglip"
    max_num_patches = 8

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.offloaded = False

    def embed_images(self, images) -> np.ndarray:
        out = np.zeros((len(images), self.dim), dtype=np.float32)
        for i, img in enumerate(images):
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
            flat = arr.reshape(-1, 3).mean(axis=0)
            out[i, : min(3, self.dim)] = flat[: min(3, self.dim)]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.clip(norms, 1e-12, None)).astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for i, b in enumerate(text.encode("utf-8")):
            vec[i % self.dim] += (b % 17) / 17.0
        n = np.linalg.norm(vec)
        return (vec / (n if n else 1.0)).astype(np.float32)

    def offload(self) -> None:
        self.offloaded = True


class FakeTagger:
    """Deterministic wd-tagger stand-in."""

    repo_id = "fake/wd-tagger"

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.offloaded = False

    def tag_images(self, images):
        result = []
        for img in images:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32)
            bright = float(arr.mean())
            result.append([
                ("bright" if bright >= 100 else "dark", "general", 0.9),
                ("test-tag", "general", 0.8),
            ])
        return result

    def embed_images(self, images) -> np.ndarray:
        out = np.zeros((len(images), self.dim), dtype=np.float32)
        for i, img in enumerate(images):
            arr = np.asarray(img.convert("RGB"), dtype=np.float32)
            out[i, 0] = arr[:, :, 0].mean() / 255.0
            out[i, 1] = arr[:, :, 1].mean() / 255.0
            out[i, 2] = arr[:, :, 2].mean() / 255.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.clip(norms, 1e-12, None)).astype(np.float32)

    def offload(self) -> None:
        self.offloaded = True


@pytest.fixture
def fake_models(state):
    siglip = FakeSigLIP()
    tagger = FakeTagger()
    state._siglip = siglip
    state._tagger = tagger
    return siglip, tagger
