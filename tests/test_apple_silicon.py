"""Apple Silicon performance/stability paths, exercised on any CPU box.

Nothing here needs a Mac: the Core ML session, the VideoToolbox probe and the
MPS device are replaced by fakes, so the tests pin the *decisions* (decode
path per codec, fixed-shape tagger batches, model residency) that the
benchmarks in docs/PERFORMANCE.md were measured with.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.indexing import cuts, pipeline
from backend.models import wd_tagger
from tests.conftest import FakeSigLIP, FakeTagger, ffmpeg_exe

_VT = {"name": "videotoolbox", "input_args": ["-hwaccel", "videotoolbox"],
       "scale_vf": "scale={w}:{h}:flags=fast_bilinear"}


@pytest.fixture
def vt_mac(monkeypatch):
    """Pretend this ffmpeg is a macOS build with VideoToolbox."""
    monkeypatch.setattr(cuts, "_ffmpeg_hw_profile", lambda: _VT)
    monkeypatch.setattr(cuts, "_DECODE_CHOICE", {})
    monkeypatch.delenv("AMV_DECODE", raising=False)


def _hevc_clip(path: Path, w: int, h: int) -> Path:
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", f"color=c=red:s={w}x{h}:d=0.2:r=10",
                    "-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "log-level=none",
                    str(path)], check=True)
    return path


def test_h264_decodes_in_software_on_videotoolbox_machines(vt_mac, clip):
    prof = cuts._decode_profile(str(clip))
    assert prof["name"] == "software"
    assert prof["input_args"] == []


def test_large_hevc_keeps_videotoolbox(vt_mac, tmp_path):
    try:
        big = _hevc_clip(tmp_path / "big.mkv", 256, 1600)
    except subprocess.CalledProcessError:
        pytest.skip("ffmpeg build without libx265")
    small = _hevc_clip(tmp_path / "small.mkv", 256, 720)
    assert cuts._decode_profile(str(big))["name"] == "videotoolbox"
    assert cuts._decode_profile(str(small))["name"] == "software"


def test_decode_override_env(vt_mac, clip, monkeypatch):
    monkeypatch.setenv("AMV_DECODE", "hw")
    assert cuts._decode_profile(str(clip))["name"] == "videotoolbox"
    monkeypatch.setenv("AMV_DECODE", "sw")
    assert cuts._decode_profile(str(clip))["name"] == "software"


def test_other_hardware_profiles_are_untouched(monkeypatch, clip):
    cuda = {"name": "cuda", "input_args": [], "scale_vf": "x"}
    monkeypatch.delenv("AMV_DECODE", raising=False)
    monkeypatch.setattr(cuts, "_ffmpeg_hw_profile", lambda: cuda)
    assert cuts._decode_profile(str(clip)) is cuda
    # no hardware path: the software ffmpeg decoder (subprocess, outside
    # the GIL), PyAV only on request
    monkeypatch.setattr(cuts, "_ffmpeg_hw_profile", lambda: None)
    assert cuts._decode_profile(str(clip))["name"] == "software"
    monkeypatch.setenv("AMV_DECODE", "pyav")
    assert cuts._decode_profile(str(clip)) is None


def test_software_decode_path_finds_the_same_cuts(vt_mac, clip):
    scenes = cuts.detect_cuts(str(clip), threshold=0.30)
    assert [s for s, _ in scenes] == [0, 1000, 2000]


def test_cuts_land_on_the_first_frame_of_the_new_shot(clip, monkeypatch):
    """CPU (PyAV) path too: the boundary is the exact frame, not the next
    sampled one (which left up to 3 frames of the next shot in the clip)."""
    monkeypatch.setattr(cuts, "_ffmpeg_hw_profile", lambda: None)
    for decode in ("sw", "pyav"):
        monkeypatch.setenv("AMV_DECODE", decode)
        assert [s for s, _ in cuts.detect_cuts(str(clip), threshold=0.30)] == [0, 1000, 2000]


# ── wd-tagger on Core ML ────────────────────────────────────────────────────
class _FakeOrt:
    def get_available_providers(self):
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def test_coreml_provider_options(data_dir):
    providers = wd_tagger.build_onnx_providers("mps", _FakeOrt(), coreml_cache_key="wd-tagger-b8")
    name, opts = providers[0]
    assert name == "CoreMLExecutionProvider"
    assert opts["ModelFormat"] == "MLProgram"
    assert opts["MLComputeUnits"] == "CPUAndGPU"
    assert opts["ModelCacheDirectory"].endswith("wd-tagger-b8")
    assert providers[-1] == "CPUExecutionProvider"
    # CUDA / CPU lists unchanged
    assert wd_tagger.build_onnx_providers("cpu", _FakeOrt()) == ["CPUExecutionProvider"]


class _RecordingSession:
    def __init__(self):
        self.batches: list[int] = []

    def run(self, _outputs, feed):
        x = next(iter(feed.values()))
        self.batches.append(len(x))
        # one "score" per image: its mean pixel value
        return [x.reshape(len(x), -1).mean(axis=1, keepdims=True).repeat(3, axis=1)]


def test_static_batch_pads_and_trims():
    tagger = wd_tagger.WDTaggerModel(device_backend="mps")
    sess = _RecordingSession()
    tagger._session = sess
    tagger._input_name = "input"
    tagger._input_size = 16
    tagger._static_batch = 8
    imgs = [Image.new("RGB", (32, 18), (i * 20, i * 20, i * 20)) for i in range(11)]
    out = tagger._run_batch(imgs)
    assert sess.batches == [8, 8]          # last chunk padded to the fixed shape
    assert out.shape == (11, 3)            # padding trimmed
    means = out[:, 0]
    assert np.all(np.diff(means[:11]) != 0)  # order preserved, no duplicate rows


# ── tagger residency across a queue ─────────────────────────────────────────
class _MacTagger(FakeTagger):
    device_backend = "mps"

    def __init__(self):
        super().__init__()
        self.offload_calls = 0

    def offload(self) -> None:
        super().offload()
        self.offload_calls += 1


def test_tagger_stays_resident_between_phases_on_mps(db_path, clip):
    tagger = _MacTagger()
    pipeline.index_video(db_path, str(clip), FakeSigLIP(), tagger, auto_skip_indexed=False)
    assert tagger.offload_calls == 0


def test_tagger_is_offloaded_between_phases_elsewhere(db_path, clip):
    tagger = FakeTagger()
    pipeline.index_video(db_path, str(clip), FakeSigLIP(), tagger, auto_skip_indexed=False)
    assert tagger.offloaded


def test_queue_frees_the_resident_tagger_at_the_end(db_path, clip):
    from backend.db import queries, schema

    with schema.get_conn(db_path) as conn:
        queries.add_to_queue(conn, str(clip), is_directory=False, recursive=False)
    tagger = _MacTagger()
    assert pipeline.index_queue(db_path, FakeSigLIP(), tagger) == "completed"
    assert tagger.offload_calls == 1


# ── SigLIP host→device copies ───────────────────────────────────────────────
class _Recorder:
    def __init__(self):
        self.kwargs = None

    def to(self, _device, **kw):
        self.kwargs = kw
        return self


@pytest.mark.parametrize("backend,expected", [("mps", False), ("cpu", False), ("cuda", True)])
def test_siglip_copies_are_blocking_except_on_cuda(backend, expected):
    from types import SimpleNamespace

    from backend.models.siglip2 import SigLIP2Model

    m = SigLIP2Model.__new__(SigLIP2Model)
    m.device = SimpleNamespace(backend=backend, torch_device="dev")
    rec = _Recorder()
    m._to_device({"pixel_values": rec})
    assert rec.kwargs == {"non_blocking": expected}


# ── single decode for cuts + medians ────────────────────────────────────────
def test_single_pass_matches_the_two_pass_decode(vt_mac, clip):
    from backend.indexing import frames, single_pass
    from backend.searchplus.analysis import FrameStatsAccumulator

    assert single_pass.supported(str(clip))
    a1, a2 = FrameStatsAccumulator(), FrameStatsAccumulator()
    two = cuts.detect_cuts(str(clip), threshold=0.30, frame_sink=a1.sink)
    one, medians = single_pass.detect_cuts_with_medians(str(clip), threshold=0.30, frame_sink=a2.sink)
    # same cut decisions, both on the exact first frame of the new shot
    # (the clip changes colour every 25 frames; sampled every 4th frame the
    # detector used to report 1120 / 2080 ms)
    assert one == two
    assert [s for s, _ in one] == [0, 1000, 2000]
    assert a1.times == a2.times            # identical motion samples
    assert np.allclose(a1.p_diff, a2.p_diff)
    targets = [s + (e - s) // 2 for s, e in one]
    assert sorted(medians) == targets
    ref = dict(frames.extract_frames_at(str(clip), targets))
    for ts in targets:
        a = np.asarray(medians[ts], dtype=np.int16)
        b = np.asarray(ref[ts], dtype=np.int16)
        assert a.shape == b.shape == (288, 512, 3)
        assert np.abs(a - b).mean() < 2.0


def test_pipeline_uses_one_decode_on_the_software_path(vt_mac, db_path, clip, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("second decode")
    monkeypatch.setattr(pipeline.frames, "extract_frames_at", boom)
    pipeline.index_video(db_path, str(clip), FakeSigLIP(), FakeTagger(), auto_skip_indexed=False)
    from backend.db import schema
    with schema.get_conn(db_path) as conn:
        n, with_emb = conn.execute(
            "SELECT COUNT(*), COUNT(embedding) FROM scenes").fetchone()
    assert n == 3 and with_emb == 3


def test_single_pass_failure_falls_back_to_two_passes(vt_mac, db_path, clip, monkeypatch):
    from backend.indexing import single_pass

    def broken(*_a, **_k):
        raise RuntimeError("ffmpeg died")
    monkeypatch.setattr(single_pass, "detect_cuts_with_medians", broken)
    pipeline.index_video(db_path, str(clip), FakeSigLIP(), FakeTagger(), auto_skip_indexed=False)
    from backend.db import schema
    with schema.get_conn(db_path) as conn:
        assert conn.execute("SELECT COUNT(embedding) FROM scenes").fetchone()[0] == 3


def test_tagger_weight_files_per_backend():
    assert wd_tagger.weight_files("mps") == ("config.json", "model.safetensors", "selected_tags.csv")
    assert wd_tagger.weight_files("cuda") == ("model.onnx", "selected_tags.csv")
    assert wd_tagger.weight_files(None) == ("model.onnx", "selected_tags.csv")


def test_mps_tagger_falls_back_to_onnx_when_torch_path_fails():
    tagger = wd_tagger.WDTaggerModel(device_backend="mps")

    def no_weights(*_a, **_k):
        raise RuntimeError("offline")
    assert tagger._load_torch_mps(no_weights) is False
    assert tagger._session is None


def test_torch_tagger_matches_the_onnx_input_contract(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    import json

    from safetensors.torch import save_file

    cfg = {"architecture": "vit_tiny_patch16_224", "num_classes": 5,
           "model_args": {"img_size": 32, "class_token": False, "global_pool": "avg"},
           "pretrained_cfg": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}}
    import timm
    torch.manual_seed(0)
    ref = timm.create_model(cfg["architecture"], num_classes=5, pretrained=False, **cfg["model_args"]).eval()
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    save_file(ref.state_dict(), str(tmp_path / "model.safetensors"))
    tt = wd_tagger._TorchTagger(str(tmp_path / "config.json"), str(tmp_path / "model.safetensors"), device="cpu")
    tt.dtype = torch.float32
    tt.model = tt.model.float()
    bgr = np.random.default_rng(0).uniform(0, 255, (2, 32, 32, 3)).astype(np.float32)
    out = tt.run(None, {"input": bgr})[0]
    expected = torch.sigmoid(ref(torch.from_numpy(bgr).permute(0, 3, 1, 2) / 127.5 - 1.0)).detach().numpy()
    assert out.shape == (2, 5)
    assert np.abs(out - expected).max() < 1e-4, np.abs(out - expected).max()


class _CountingSigLIP(FakeSigLIP):
    def __init__(self):
        super().__init__()
        self.images = 0

    def embed_images(self, images):
        self.images += len(images)
        return super().embed_images(images)


class _CountingMacTagger(_MacTagger):
    def __init__(self):
        super().__init__()
        self.images = 0

    def tag_images(self, images):
        self.images += len(images)
        return super().tag_images(images)


def test_medians_are_tagged_and_embedded_during_the_decode(vt_mac, db_path, clip):
    from backend.db import schema

    tagger, siglip = _CountingMacTagger(), _CountingSigLIP()
    pipeline.index_video(db_path, str(clip), siglip, tagger, auto_skip_indexed=False)
    # each median went through each model exactly once (no second pass)
    assert tagger.images == 3 and siglip.images == 3
    with schema.get_conn(db_path) as conn:
        assert conn.execute("SELECT COUNT(embedding) FROM scenes").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(DISTINCT scene_id) FROM scene_tags").fetchone()[0] == 3


def test_streaming_failure_falls_back_to_the_regular_phases(vt_mac, db_path, clip):
    from backend.db import schema

    class Flaky(_CountingMacTagger):
        calls = 0

        def tag_images(self, images):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise RuntimeError("GPU hiccup")
            return super().tag_images(images)

    tagger = Flaky()
    pipeline.index_video(db_path, str(clip), FakeSigLIP(), tagger, auto_skip_indexed=False)
    with schema.get_conn(db_path) as conn:
        assert conn.execute("SELECT COUNT(DISTINCT scene_id) FROM scene_tags").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(embedding) FROM scenes").fetchone()[0] == 3


def test_proxy_decode_follows_the_indexing_rule(vt_mac, clip, monkeypatch):
    from backend.export.ffmpeg import _videotoolbox_decode_args

    assert _videotoolbox_decode_args(str(clip)) == []          # H.264: software
    monkeypatch.setenv("AMV_DECODE", "hw")
    assert _videotoolbox_decode_args(str(clip)) == ["-hwaccel", "videotoolbox"]


# ── characters: automatic threshold on a same-series library ────────────────
def test_character_auto_threshold_respects_the_library(tmp_path):
    """Within one series every shot looks alike to SigLIP (shared style):
    the example-only threshold matched 262/418 scenes of a real episode.
    The library floor keeps the automatic threshold above ~90 % of it."""
    from backend.db import queries, schema
    from backend.searchplus import characters as chars
    from tests.test_search_plus import add_scene, add_video

    rng = np.random.default_rng(3)

    def unit(v):
        return v / np.linalg.norm(v)

    style, hero = unit(rng.normal(size=64)), unit(rng.normal(size=64))
    db = str(tmp_path / "lib.db")
    schema.init_db(db)
    with schema.get_conn(db) as conn:
        vid = add_video(conn, "Ep 01")
        hero_ids = [add_scene(conn, vid, i, unit(style + 0.5 * hero + 0.5 * unit(rng.normal(size=64))))
                    for i in range(20)]
        other_ids = [add_scene(conn, vid, 100 + i, unit(style + 0.6 * unit(rng.normal(size=64))))
                     for i in range(280)]
        rows = queries.get_all_scenes_with_embeddings(conn)
        index = {"rows": rows, "sids": np.array([r[0] for r in rows]),
                 "embeddings": np.vstack([np.frombuffer(r[6], np.float32) for r in rows]),
                 "signature": ("lib", len(rows))}
        cid = chars.create_character(conn, "Hero", None)
        conn.execute("UPDATE characters SET threshold_auto = 1 WHERE id = ?", (cid,))
        for sid in hero_ids[:3]:
            chars.add_example(conn, cid, chars.scene_embedding(conn, sid), scene_id=sid)
        res = chars.rescore(conn, cid, index)
        matched = {m["id"] for m in chars.matches(conn, cid, limit=5000)
                   if m["score"] >= res["threshold"]}
    # example-only threshold here: 189 of the 280 other scenes matched
    assert len(matched & set(other_ids)) <= 30
    assert len(matched & set(hero_ids)) >= 15            # still finds the character


def test_sub_segmentation_streams_from_the_single_decode(db_path, clip, monkeypatch):
    """Accurate profile on the single-pass path: the worker splits a long
    scene from its decoded samples and tags/embeds every piece — no extra
    decode for the samples or the sub-segment medians."""
    from backend.db import schema
    from backend.indexing import single_pass

    red = np.zeros((288, 512, 3), np.uint8)
    red[..., 0] = 220
    blue = np.zeros((288, 512, 3), np.uint8)
    blue[..., 2] = 220

    def fake_decode(path, threshold=0.3, frame_sink=None, on_median=None, on_scene=None):
        samples = [(t, red if t < 4000 else blue) for t in range(0, 8000, 167)]
        assert on_scene is not None and on_median is None
        on_scene(0, 8000, samples)
        return [(0, 8000)], {4000: Image.fromarray(red)}

    def boom(*_a, **_k):
        raise AssertionError("extra decode")

    monkeypatch.setattr(single_pass, "supported", lambda _p: True)
    monkeypatch.setattr(single_pass, "detect_cuts_with_medians", fake_decode)
    monkeypatch.setattr(pipeline.frames, "extract_frames_at", boom)
    tagger, siglip = _CountingMacTagger(), _CountingSigLIP()
    pipeline.index_video(db_path, str(clip), siglip, tagger, auto_skip_indexed=False,
                         mode="balanced", enable_sub_segmentation=True)
    with schema.get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT id, start_ms, end_ms, parent_scene_id, embedding IS NOT NULL FROM scenes ORDER BY id"
        ).fetchall()
        tagged = conn.execute("SELECT COUNT(DISTINCT scene_id) FROM scene_tags").fetchone()[0]
    parent = rows[0]
    assert (parent[1], parent[2], parent[3]) == (0, 8000, None)
    assert [(r[1], r[2], r[3]) for r in rows[1:]] == [(0, 4000, parent[0]), (4000, 8000, parent[0])]
    assert all(r[4] for r in rows) and tagged == 3
    assert tagger.images == 3 and siglip.images == 3
