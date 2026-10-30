"""Chantier 4b: non-commercial models (AnimeJaNai V3 upscale, Video Depth
Anything Base/Large wiring), zip-member downloads, non-commercial flag."""
from __future__ import annotations

import hashlib
import io
import zipfile

import numpy as np
import pytest


class _Resp:
    def __init__(self, payload: bytes):
        self.payload = payload

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for i in range(0, len(self.payload), chunk_size):
            yield self.payload[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_zip_member_download_keeps_only_the_member(tmp_path, monkeypatch):
    import requests

    from backend.finish import registry

    wanted, other = b"w" * 50_000, b"o" * 80_000
    payload = _zip({"pack/wanted.pth": wanted, "pack/other.pth": other})
    spec = registry.ModelSpec(
        id="zip-model", kind="upscale", label="t", license="CC-BY-NC-SA-4.0", source="x",
        files=(registry.ModelFile("wanted.pth", "http://x/pack.zip", len(payload),
                                  hashlib.sha256(payload).hexdigest(), member="pack/wanted.pth"),),
    )
    monkeypatch.setitem(registry.MODELS, spec.id, spec)
    monkeypatch.setenv("AMV_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(payload))
    seen = []
    registry.download(spec.id, progress_cb=lambda n, t: seen.append((n, t)))
    d = tmp_path / spec.id
    assert sorted(p.name for p in d.iterdir()) == ["wanted.pth"]
    assert (d / "wanted.pth").read_bytes() == wanted
    assert seen[-1] == (len(payload), len(payload))
    info = registry.describe(spec.id)
    assert info["installed"] and info["non_commercial"] and info["disk_size"] == len(wanted)

    # a tampered archive is refused and leaves nothing behind
    registry.delete(spec.id)
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(payload[:-1] + b"!"))
    with pytest.raises(RuntimeError, match="checksum"):
        registry.download(spec.id)
    assert not any(d.glob("*"))


def test_animejanai_is_registered_as_native_x2_non_commercial():
    from backend.finish import registry, upscale
    from backend.models import tiers

    spec = registry.MODELS["animejanai-v3-uc"]
    assert spec.kind == "upscale" and spec.license == "CC-BY-NC-SA-4.0"
    assert spec.files[0].member.endswith("2x_AnimeJaNai_HD_V3_UltraCompact.pth")
    assert spec.files[0].url.startswith("https://github.com/the-database/")
    assert "animejanai-v3-uc" in upscale.UPSCALE_MODELS
    assert upscale.NATIVE_X2["animejanai-v3-uc"] == ("2x_AnimeJaNai_HD_V3_UltraCompact.pth", 8)
    info = tiers.model_info("animejanai-v3-uc")
    assert info["non_commercial"] and info["license"] == "cc-by-nc-sa-4.0"
    assert not tiers.model_info("hfa2k-avc-compact")["non_commercial"]


def test_ultracompact_upscaler_runs_x2_and_x4(tmp_path, monkeypatch):
    import torch

    from backend.finish import upscale
    from backend.finish.archs.realesrgan import SRVGGNetCompact
    from backend.models.device import DeviceInfo

    d = tmp_path / "animejanai-v3-uc"
    d.mkdir()
    net = SRVGGNetCompact(num_feat=64, num_conv=8, upscale=2)
    torch.save({"params": net.state_dict()}, d / "2x_AnimeJaNai_HD_V3_UltraCompact.pth")
    monkeypatch.setenv("AMV_MODELS_DIR", str(tmp_path))
    up = upscale.Upscaler("animejanai-v3-uc", DeviceInfo("cpu", "cpu", torch.device("cpu"), torch.float32))
    assert up.native_scale == 2
    frame = (np.random.default_rng(0).random((24, 40, 3)) * 255).astype(np.uint8)
    assert up.upscale(frame, scale=2).shape == (48, 80, 3)
    assert up.upscale(frame, scale=4).shape == (96, 160, 3)


def test_vda_variants_match_upstream_configs():
    from backend.finish.vda import VDA_MODEL, VDA_VARIANTS

    assert VDA_VARIANTS[VDA_MODEL][0] == "vits"
    assert VDA_VARIANTS["video-depth-anything-base"][:3] == ("vitb", 128, (96, 192, 384, 768))
    assert VDA_VARIANTS["video-depth-anything-large"][:3] == ("vitl", 256, (256, 512, 1024, 1024))
