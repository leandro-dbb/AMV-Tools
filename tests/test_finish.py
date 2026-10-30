"""Finishing toolbox tests (chantier D).

Fast tests mock the networks and check the plumbing: tiling, frame assembly,
audio preservation, cut-aware interpolation, job cancellation, registry
downloads, temporal stabilisation and Mask Mode corrections.

``@pytest.mark.slow`` tests run the *real* models on CPU over a tiny
synthetic clip. They need the weights in ``AMV_MODELS_DIR`` (default
``~/.cache/amv-tools/finish-models``); set ``AMV_TEST_DOWNLOAD=1`` to let the
test fetch them (throttled, ~145 MB total), otherwise they are skipped.
Run only the fast ones with ``pytest -m "not slow"``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Isolated data dir before anything instantiates AppState.
_DATA = Path(tempfile.mkdtemp(prefix="amv-finish-test-"))
os.environ.setdefault("AMV_DATA_DIR", str(_DATA))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

MODELS_CACHE = Path(os.environ.get("AMV_TEST_MODELS_DIR",
                                   str(Path.home() / ".cache" / "amv-tools" / "finish-models")))


# ── helpers / fixtures (local: conftest.py belongs to chantier A) ───────────
def _ffmpeg() -> str:
    from backend.indexing.cuts import _ffmpeg_path
    return _ffmpeg_path()


def _probe(path: Path) -> dict:
    import av

    c = av.open(str(path))
    try:
        v = c.streams.video[0]
        n = sum(1 for _ in c.decode(video=0))
        out = {"w": v.codec_context.width, "h": v.codec_context.height,
               "fps": float(v.average_rate), "frames": n,
               "audio": bool(c.streams.audio)}
        if c.streams.audio:
            a = c.streams.audio[0]
            out["audio_s"] = float(a.duration * a.time_base) if a.duration else None
        return out
    finally:
        c.close()


def make_clip(path: Path, *, w=128, h=72, fps=24, frames=8, cut_at: int | None = None,
              audio=True) -> Path:
    """testsrc2 (motion) with an optional hard cut to a solid colour."""
    dur_a = (cut_at if cut_at is not None else frames) / fps
    cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate={fps}:duration={dur_a}"]
    graph = "[0:v]format=yuv420p[v]"
    if cut_at is not None:
        cmd += ["-f", "lavfi", "-i", f"color=c=orange:size={w}x{h}:rate={fps}:duration={(frames - cut_at) / fps}"]
        graph = "[0:v][1:v]concat=n=2:v=1,format=yuv420p[v]"
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={frames / fps}"]
    cmd += ["-filter_complex", graph, "-map", "[v]"]
    if audio:
        cmd += ["-map", f"{2 if cut_at is not None else 1}:a", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-frames:v", str(frames), str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


@pytest.fixture
def finish_clip(tmp_path) -> Path:
    return make_clip(tmp_path / "clip.mp4")


@pytest.fixture
def cut_clip(tmp_path) -> Path:
    return make_clip(tmp_path / "cut.mp4", cut_at=4)


class _Ctx:
    """Minimal JobContext stand-in for calling job bodies directly."""

    def __init__(self):
        from backend.finish.jobs import Job
        self.job = Job(kind="test", params={})
        self.cleanups = []
        self.cancel_after: int | None = None
        self.calls = 0

    def check_cancel(self):
        from backend.finish.jobs import JobCancelled
        if self.job.cancel_event.is_set():
            raise JobCancelled()

    def on_cancel_cleanup(self, fn):
        self.cleanups.append(fn)

    def progress(self, fraction, message=None, force=False):
        self.calls += 1
        if self.cancel_after is not None and self.calls >= self.cancel_after:
            self.job.cancel_event.set()
        self.check_cancel()


class _CpuDevice:
    backend = "cpu"
    name = "CPU (test)"
    vram_gb = None

    @property
    def torch_device(self):
        import torch
        return torch.device("cpu")


# ── tiling ──────────────────────────────────────────────────────────────────
def test_tiled_upscale_matches_whole_frame():
    torch = pytest.importorskip("torch")
    from backend.finish.upscale import tiled_apply

    torch.manual_seed(0)
    # Receptive radius 2 (two 3x3 convs) + nearest x4 — smaller than the pad,
    # so tiles must reproduce the whole-frame result exactly.
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, 1, 1), torch.nn.ReLU(), torch.nn.Conv2d(8, 3 * 16, 3, 1, 1),
        torch.nn.PixelShuffle(4),
    ).eval()
    x = torch.rand(1, 3, 37, 53)
    with torch.inference_mode():
        full = net(x)
        tiled = tiled_apply(net, x, scale=4, tile=16, tile_pad=4)
    assert tiled.shape == full.shape == (1, 3, 148, 212)
    assert torch.allclose(tiled, full, atol=1e-5)


def test_tiled_apply_no_tiling_when_frame_fits():
    torch = pytest.importorskip("torch")
    from backend.finish.upscale import tiled_apply

    calls = []

    def fn(t):
        calls.append(t.shape)
        return torch.nn.functional.interpolate(t, scale_factor=2, mode="nearest")

    tiled_apply(fn, torch.rand(1, 3, 20, 20), scale=2, tile=64)
    assert len(calls) == 1
    tiled_apply(fn, torch.rand(1, 3, 20, 40), scale=2, tile=16)
    assert len(calls) == 1 + 2 * 3


def test_upscale_output_size_is_even():
    from backend.finish.upscale import auto_tile, output_size
    assert output_size(853, 480, 2) == (1706, 960)
    assert output_size(427, 241, 4) == (1708, 964)
    assert auto_tile("cpu", None, 1920, 1080) == 256
    assert auto_tile("cuda", 24.0, 1280, 720) == 0


# ── frame assembly + audio ──────────────────────────────────────────────────
def test_reader_writer_roundtrip_keeps_frames_and_audio(finish_clip, tmp_path):
    from backend.finish.video_io import FrameReader, FrameWriter, probe_clip

    info = probe_clip(str(finish_clip))
    assert (info.width, info.height, info.has_audio) == (128, 72, True)
    with FrameReader(info) as r:
        frames = list(r)
    assert len(frames) == 8 and frames[0].shape == (72, 128, 3)
    out = tmp_path / "rt.mp4"
    w = FrameWriter(str(out), width=128, height=72, fps=info.fps, audio=info)
    for f in frames:
        w.write(f)
    assert w.close() == str(out)
    p = _probe(out)
    assert p["frames"] == 8 and p["audio"]
    assert not list(tmp_path.glob("*.partial*"))


def test_reader_honours_trim_window(tmp_path):
    from backend.finish.video_io import probe_clip, read_frames

    src = make_clip(tmp_path / "long.mp4", frames=24)
    info = probe_clip(str(src), 250, 750)       # 0.5 s at 24 fps
    assert info.n_frames == 12
    assert len(read_frames(info)) == 12


def _patch_runner(monkeypatch, tmp_path):
    from backend.finish import runner
    monkeypatch.setattr(runner, "_device", lambda: _CpuDevice())
    monkeypatch.setattr(runner, "_free_other_models", lambda: None)
    monkeypatch.setattr(runner, "_export_settings",
                        lambda: {"codec": "libx264", "crf": 18, "audio": "copy", "audio_bitrate_kbps": 192})
    runner.release_models()
    return runner


def test_upscale_job_mocked_model_keeps_audio(finish_clip, tmp_path, monkeypatch):
    runner = _patch_runner(monkeypatch, tmp_path)
    import backend.finish.upscale as up

    class FakeUpscaler:
        def __init__(self, model_id, device):
            self.model_id = model_id

        def upscale(self, frame, *, scale=4, tile=0, tile_pad=10, check_cancel=None):
            return np.repeat(np.repeat(frame, scale, 0), scale, 1)

    monkeypatch.setattr(up, "Upscaler", FakeUpscaler)
    from backend.finish.video_io import probe_clip
    out = runner._run_upscale(_Ctx(), probe_clip(str(finish_clip)), "clip", tmp_path / "out", {"scale": 2})
    p = _probe(Path(out))
    assert (p["w"], p["h"], p["frames"]) == (256, 144, 8)
    assert p["audio"], "source audio must be muxed back"
    assert p["audio_s"] == pytest.approx(8 / 24, abs=0.06)


# ── interpolation logic ─────────────────────────────────────────────────────
def test_plan_timeline_slowmo_and_fps():
    from backend.finish.interpolate import plan_timeline

    p = plan_timeline(4, 24.0, mode="slowmo", factor=4)
    assert len(p.positions) == 16 and p.out_fps == 24.0 and p.audio_speed == 0.25
    assert p.positions[:5] == [0, 0.25, 0.5, 0.75, 1.0]
    assert p.positions[-1] == 3  # trailing slots duplicate the last frame
    q = plan_timeline(24, 24.0, mode="fps", target_fps=60.0)
    assert len(q.positions) == 60 and q.out_fps == 60.0 and q.audio_speed == 1.0
    assert q.positions[5] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        plan_timeline(10, 60.0, mode="fps", target_fps=30.0)


def test_cut_detection_on_hard_cut_not_on_motion(cut_clip):
    from backend.finish.interpolate import detect_cuts
    from backend.finish.video_io import probe_clip, read_frames

    frames = read_frames(probe_clip(str(cut_clip)))
    assert detect_cuts(frames) == {3}


def test_interpolation_never_blends_across_a_cut():
    from backend.finish.interpolate import plan_timeline, stream_interpolated

    frames = [np.full((4, 4, 3), i * 30, np.uint8) for i in range(6)]
    blended: list[tuple[int, int, float]] = []

    def make_pair(a, b):
        return (int(a[0, 0, 0]) // 30, int(b[0, 0, 0]) // 30)

    def at(pair, t):
        i, j = pair
        blended.append((i, j, t))
        return ((1 - t) * frames[i] + t * frames[j]).astype(np.uint8)

    out = list(stream_interpolated(iter(frames), 0.25, make_pair=make_pair, at=at,
                                   is_cut=lambda a, b, i: i == 2))   # cut between 2 and 3
    assert len(out) == len(plan_timeline(6, 24.0, mode="slowmo", factor=4).positions) == 24
    assert all(not (i <= 2 < j) for i, j, _ in blended)
    # 2.25 snaps back to frame 2; 2.5 and 2.75 snap forward to frame 3
    assert np.array_equal(out[9], frames[2]) and np.array_equal(out[10], frames[3])
    assert np.array_equal(out[11], frames[3])
    assert (0, 1, 0.5) in blended
    assert all(np.array_equal(o, frames[5]) for o in out[20:])   # tail duplicates


def test_stream_fps_positions_match_plan():
    from backend.finish.interpolate import plan_timeline, stream_interpolated

    frames = [np.full((2, 2, 3), i, np.uint8) for i in range(24)]
    ts: list[float] = []
    out = list(stream_interpolated(iter(frames), 24 / 60, make_pair=lambda a, b: (a, b),
                                   at=lambda pair, t: ts.append(t) or pair[0]))
    assert len(out) == len(plan_timeline(24, 24.0, mode="fps", target_fps=60.0).positions) == 60
    assert {round(t, 3) for t in ts} == {0.2, 0.4, 0.6, 0.8}


def test_interpolate_job_mocked_rife_duplicates_at_cut(cut_clip, tmp_path, monkeypatch):
    runner = _patch_runner(monkeypatch, tmp_path)
    import backend.finish.interpolate as itp

    class FakeRife:
        def __init__(self, device, scale=1.0):
            pass

        def make_pair(self, a, b):
            return (a, b)

        def at(self, pair, t):
            a, b = pair
            return ((1 - t) * a.astype(np.float32) + t * b.astype(np.float32)).astype(np.uint8)

    monkeypatch.setattr(itp, "RifeInterpolator", FakeRife)
    from backend.finish.video_io import probe_clip, read_frames
    out = runner._run_interpolate(_Ctx(), probe_clip(str(cut_clip)), "cut", tmp_path / "o",
                                  {"mode": "slowmo", "factor": 2, "skip_cuts": True})
    p = _probe(Path(out))
    assert p["frames"] == 16 and not p["audio"]  # slow-mo mutes audio by default
    got = read_frames(probe_clip(out))
    # frame 3 -> 4 is the cut: output slot 7 (position 3.5) must be a copy of
    # frame 4 (orange), not a testsrc/orange blend.
    orange = got[8].astype(int)
    assert np.abs(got[7].astype(int) - orange).mean() < 3


def test_fps_mode_keeps_audio(finish_clip, tmp_path, monkeypatch):
    runner = _patch_runner(monkeypatch, tmp_path)
    import backend.finish.interpolate as itp

    class FakeRife:
        def __init__(self, *a, **k):
            pass

        def make_pair(self, a, b):
            return (a, b)

        def at(self, pair, t):
            return pair[0]

    monkeypatch.setattr(itp, "RifeInterpolator", FakeRife)
    from backend.finish.video_io import probe_clip
    out = runner._run_interpolate(_Ctx(), probe_clip(str(finish_clip)), "c", tmp_path / "o",
                                  {"mode": "fps", "target_fps": 60})
    p = _probe(Path(out))
    assert p["fps"] == pytest.approx(60.0) and p["frames"] == 20 and p["audio"]


def test_slowmo_audio_stretch(finish_clip, tmp_path, monkeypatch):
    runner = _patch_runner(monkeypatch, tmp_path)
    import backend.finish.interpolate as itp

    class FakeRife:
        def __init__(self, *a, **k):
            pass

        def make_pair(self, a, b):
            return (a, b)

        def at(self, pair, t):
            return pair[0]

    monkeypatch.setattr(itp, "RifeInterpolator", FakeRife)
    from backend.finish.video_io import probe_clip
    out = runner._run_interpolate(_Ctx(), probe_clip(str(finish_clip)), "c", tmp_path / "o",
                                  {"mode": "slowmo", "factor": 2, "audio": "stretch"})
    p = _probe(Path(out))
    assert p["audio"] and p["audio_s"] == pytest.approx(16 / 24, abs=0.08)


# ── job queue / cancellation ────────────────────────────────────────────────
def _manager(runner_fn):
    from backend.finish.jobs import JobManager
    events: list[dict] = []
    return JobManager(runner_fn, events.append), events


def test_job_cancel_running_and_queued():
    started = threading.Event()
    cleaned = []

    def body(ctx):
        ctx.on_cancel_cleanup(lambda: cleaned.append(ctx.job.id))
        started.set()
        for i in range(500):
            time.sleep(0.01)
            ctx.progress(i / 500)
        return "never"

    mgr, events = _manager(body)
    a = mgr.submit("upscale", {})
    b = mgr.submit("upscale", {})
    assert started.wait(5)
    assert mgr.cancel(b.id)               # queued -> cancelled, never runs
    assert mgr.cancel(a.id)               # running -> cancelled at next progress()
    assert mgr.wait_idle(5)
    jobs = {j["id"]: j for j in mgr.list()}
    assert jobs[a.id]["status"] == "cancelled" and jobs[a.id]["output"] is None
    assert jobs[b.id]["status"] == "cancelled" and jobs[b.id]["started_at"] is None
    assert cleaned == [a.id]
    assert any(e["job"]["status"] == "cancelled" for e in events)
    assert not mgr.cancel(a.id)


def test_job_failure_is_reported_not_raised():
    def body(ctx):
        raise RuntimeError("boom")

    mgr, _ = _manager(body)
    j = mgr.submit("depth", {})
    assert mgr.wait_idle(5)
    got = mgr.list()[0]
    assert got["status"] == "failed" and "boom" in got["error"]
    assert mgr.remove(j.id) and mgr.list() == []


def test_cancelled_upscale_leaves_no_partial_file(finish_clip, tmp_path, monkeypatch):
    runner = _patch_runner(monkeypatch, tmp_path)
    import backend.finish.upscale as up
    from backend.finish.jobs import JobCancelled
    from backend.finish.video_io import probe_clip

    class SlowUpscaler:
        def __init__(self, *a):
            pass

        def upscale(self, frame, *, scale=4, tile=0, tile_pad=10, check_cancel=None):
            return np.repeat(np.repeat(frame, scale, 0), scale, 1)

    monkeypatch.setattr(up, "Upscaler", SlowUpscaler)
    ctx = _Ctx()
    ctx.cancel_after = 3
    out_dir = tmp_path / "o"
    with pytest.raises(JobCancelled):
        runner._run_upscale(ctx, probe_clip(str(finish_clip)), "clip", out_dir, {"scale": 2})
    for fn in ctx.cleanups:
        fn()
    assert not [p for p in out_dir.glob("*") if p.is_file()]


# ── registry ────────────────────────────────────────────────────────────────
class _FakeResp:
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


def test_registry_download_checksum_and_delete(tmp_path, monkeypatch):
    import hashlib

    import requests

    from backend.finish import registry

    payload = os.urandom(300_000)
    spec = registry.ModelSpec(
        id="unit-model", kind="upscale", label="t", license="MIT", source="x",
        files=(registry.ModelFile("w.bin", "http://x/w.bin", len(payload),
                                  hashlib.sha256(payload).hexdigest()),),
    )
    monkeypatch.setitem(registry.MODELS, spec.id, spec)
    monkeypatch.setenv("AMV_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResp(payload))
    seen = []
    assert not registry.is_installed(spec.id)
    with pytest.raises(registry.ModelMissing):
        registry.require(spec.id)
    registry.download(spec.id, progress_cb=lambda n, t: seen.append(n))
    assert registry.is_installed(spec.id) and seen[-1] == len(payload)
    assert registry.describe(spec.id)["disk_size"] == len(payload)
    assert registry.delete(spec.id) and not registry.is_installed(spec.id)

    # corrupted download -> no file left behind
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResp(payload[:-1] + b"!"))
    with pytest.raises(RuntimeError, match="checksum"):
        registry.download(spec.id)
    assert not any((tmp_path / spec.id).glob("*"))


def test_registry_download_is_throttled(tmp_path, monkeypatch):
    import requests

    from backend.finish import registry

    payload = b"x" * 400_000
    spec = registry.ModelSpec(id="slow-model", kind="depth", label="t", license="MIT", source="x",
                              files=(registry.ModelFile("w.bin", "http://x", len(payload)),))
    monkeypatch.setitem(registry.MODELS, spec.id, spec)
    monkeypatch.setenv("AMV_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("AMV_DOWNLOAD_LIMIT_MBPS", "1")    # 1 MiB/s -> ~0.38 s
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResp(payload))
    t0 = time.monotonic()
    registry.download(spec.id)
    assert time.monotonic() - t0 >= 0.3


def test_registry_lists_only_permissive_models():
    from backend.finish import registry
    # CC-BY-4.0: redistributable with attribution (credited in the label/README);
    # CC-BY-NC-SA-4.0 (chantier 4b): non-commercial, flagged for the UI
    models = registry.list_models()
    assert {m["license"] for m in models} <= {"BSD-3-Clause", "MIT", "Apache-2.0", "CC-BY-4.0",
                                              "CC-BY-NC-SA-4.0", "CC-BY-NC-4.0"}
    assert all(m["non_commercial"] == ("-NC" in m["license"]) for m in models)
    depth = registry.MODELS["depth-anything-v2-small"]
    assert all("Small" in f.url for f in depth.files)


# ── temporal stabilisation (mask + depth) ───────────────────────────────────
def _moving_disc(n=12, h=64, w=96, speed=3.0):
    import cv2
    frames, alpha = [], []
    for t in range(n):
        a = np.zeros((h * 4, w * 4), np.float32)
        cv2.circle(a, (int((20 + speed * t) * 4), 32 * 4), 14 * 4, 1.0, -1)
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
        bg = np.full((h, w, 3), (40, 90, 160), np.float32)
        fg = np.array([250, 200, 40], np.float32)
        frames.append((bg * (1 - a[..., None]) + fg * a[..., None]).astype(np.uint8))
        alpha.append(a)
    return frames, np.stack(alpha)


def test_stabilizer_removes_flicker_without_ghosting():
    from backend.finish.matte import stabilize_alpha
    from backend.finish.synthetic import matte_metrics
    from backend.models.mask_postprocess import temporal_smooth_alpha

    frames, gt = _moving_disc()
    rng = np.random.default_rng(0)
    import cv2
    noisy = []
    for g in gt:  # per-frame boundary jitter, like a frame-independent model
        k = int(rng.integers(0, 3))
        m = cv2.dilate(g, np.ones((3, 3), np.uint8), iterations=k) if rng.random() < 0.5 \
            else cv2.erode(g, np.ones((3, 3), np.uint8), iterations=k)
        noisy.append(np.clip(m + rng.normal(0, 0.04, g.shape).astype(np.float32) * (m > 0), 0, 1))
    noisy = np.stack(noisy)
    raw = matte_metrics(noisy, gt)
    old = matte_metrics(temporal_smooth_alpha(noisy, strength=0.5), gt)
    new = matte_metrics(stabilize_alpha(noisy, frames, strength=0.6), gt)
    assert new["flicker"] < raw["flicker"] * 0.85
    assert new["edge_mae"] <= old["edge_mae"]     # no ghost trail on the moving edge
    assert new["iou"] >= raw["iou"] - 0.01


def test_stabilizer_resets_at_cuts():
    from backend.finish.temporal import stabilize_stack

    stack = np.zeros((6, 8, 8), np.float32)
    stack[3:] = 1.0
    out = stabilize_stack(stack, cuts={2}, strength=1.0)
    assert np.allclose(out[:3], 0.0) and np.allclose(out[3:], 1.0)


def test_depth_alignment_removes_scale_pumping():
    from backend.finish.depth import stabilize_depth

    rng = np.random.default_rng(1)
    base = np.linspace(0, 1, 32 * 48, dtype=np.float32).reshape(32, 48)
    raw = [base * s + o for s, o in zip(rng.uniform(0.6, 1.6, 8), rng.uniform(-2, 2, 8))]
    frames = [np.full((32, 48, 3), 128, np.uint8)] * 8
    out = stabilize_depth(raw, frames, cuts=set(), smoothing=0.0)
    assert out.shape == (8, 32, 48) and out.min() >= 0 and out.max() <= 1
    assert np.abs(np.diff(out, axis=0)).max() < 0.02


# ── Mask Mode corrections ───────────────────────────────────────────────────
def test_brush_paint_add_and_remove():
    from backend.api.segment import BrushStroke, _paint

    base = np.zeros((40, 60), np.float32)
    added = _paint(base, [BrushStroke(points=[(10, 20), (40, 20)], radius=5, mode="add")])
    assert added[20, 25] == pytest.approx(1.0) and added[5, 5] == 0
    removed = _paint(added, [BrushStroke(points=[(25, 20)], radius=4, mode="remove")])
    assert removed[20, 25] < 0.05 and removed[20, 12] == pytest.approx(1.0)


def test_synthetic_metrics_are_sane():
    from backend.finish.synthetic import matte_metrics, render_clip

    frames, gt = render_clip("walk", n_frames=4, w=96, h=54)
    m = matte_metrics(gt, gt)
    assert m["iou"] == 1.0 and m["edge_mae"] == 0 and m["flicker"] == 0
    worse = matte_metrics(np.roll(gt, 2, axis=2), gt)
    assert worse["iou"] < 1.0 and worse["edge_mae"] > 0


# ── API surface (needs httpx for TestClient) ────────────────────────────────
def test_api_models_and_missing_model_error(tmp_path, monkeypatch, finish_clip):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("AMV_MODELS_DIR", str(tmp_path / "empty-models"))
    sys.path.insert(0, str(ROOT / "backend"))
    from backend.server import make_app

    c = TestClient(make_app())
    r = c.get("/api/finish/models")
    from backend.finish import registry
    assert r.status_code == 200 and len(r.json()["models"]) == len(registry.MODELS)
    assert all(not m["installed"] for m in r.json()["models"])
    r = c.post("/api/finish/jobs", json={"kind": "depth", "sources": [{"path": str(finish_clip)}], "params": {}})
    assert r.status_code == 409 and "not downloaded" in r.json()["detail"]
    r = c.post("/api/finish/jobs", json={"kind": "upscale", "sources": [{"path": str(finish_clip)}],
                                         "params": {"scale": 3}})
    assert r.status_code in (400, 409)
    assert c.get("/api/finish/jobs").json() == {"jobs": []}


# ── real models on CPU (slow) ───────────────────────────────────────────────
def _models_dir_or_skip(*ids: str) -> Path:
    from backend.finish import registry

    os.environ["AMV_MODELS_DIR"] = str(MODELS_CACHE)
    missing = [i for i in ids if not registry.is_installed(i)]
    if missing and os.environ.get("AMV_TEST_DOWNLOAD") == "1":
        for i in missing:
            registry.download(i)
        missing = [i for i in ids if not registry.is_installed(i)]
    if missing:
        pytest.skip(f"weights not available: {missing} (set AMV_TEST_DOWNLOAD=1)")
    return MODELS_CACHE


@pytest.fixture
def real_runner(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    import torch
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    runner = _patch_runner(monkeypatch, tmp_path)
    yield runner
    runner.release_models()


@pytest.mark.slow
@pytest.mark.parametrize("model_id,scale", [("realesr-animevideov3", 2), ("realesrgan-x4plus-anime-6b", 4)])
def test_real_upscale_cpu(real_runner, finish_clip, tmp_path, model_id, scale):
    _models_dir_or_skip(model_id)
    from backend.finish.video_io import probe_clip, read_frames

    out = real_runner._run_upscale(_Ctx(), probe_clip(str(finish_clip)), "clip", tmp_path / "o",
                                   {"model": model_id, "scale": scale})
    p = _probe(Path(out))
    assert (p["w"], p["h"], p["frames"], p["audio"]) == (128 * scale, 72 * scale, 8, True)
    src = read_frames(probe_clip(str(finish_clip)))[3].astype(np.float32)
    got = read_frames(probe_clip(out))[3]
    import cv2
    back = cv2.resize(got, (128, 72), interpolation=cv2.INTER_AREA).astype(np.float32)
    assert np.abs(back - src).mean() < 12, "upscaled frame must still look like the source"


@pytest.mark.slow
def test_real_rife_cpu(real_runner, cut_clip, tmp_path):
    _models_dir_or_skip("rife-4.25")
    from backend.finish.interpolate import RifeInterpolator
    from backend.finish.video_io import probe_clip, read_frames

    out = real_runner._run_interpolate(_Ctx(), probe_clip(str(cut_clip)), "cut", tmp_path / "o",
                                       {"mode": "slowmo", "factor": 2, "skip_cuts": True})
    assert _probe(Path(out))["frames"] == 16

    # quality: the in-between of a moving disc beats a naive cross-fade
    rife = RifeInterpolator(_CpuDevice())
    frames, _ = _moving_disc(n=3, speed=10.0)
    mid = rife.at(rife.make_pair(frames[0], frames[2]), 0.5).astype(np.float32)
    psnr = lambda a, b: 10 * np.log10(255 ** 2 / max(1e-6, ((a - b) ** 2).mean()))  # noqa: E731
    gt = frames[1].astype(np.float32)
    blend = (frames[0].astype(np.float32) + frames[2]) / 2
    assert psnr(mid, gt) > psnr(blend, gt) + 6


@pytest.mark.slow
@pytest.mark.parametrize("fmt", ["png16", "prores"])
def test_real_depth_cpu(real_runner, finish_clip, tmp_path, fmt):
    _models_dir_or_skip("depth-anything-v2-small")
    import cv2

    from backend.finish.video_io import probe_clip

    out = Path(real_runner._run_depth(_Ctx(), probe_clip(str(finish_clip)), "clip", tmp_path / "o",
                                      {"format": fmt, "smoothing": 0.5}))
    if fmt == "png16":
        pngs = sorted(out.glob("*.png"))
        assert len(pngs) == 8
        d = cv2.imread(str(pngs[0]), cv2.IMREAD_UNCHANGED)
        assert d.dtype == np.uint16 and d.shape == (72, 128) and d.max() > d.min()
    else:
        p = _probe(out)
        assert (p["w"], p["h"], p["frames"], p["audio"]) == (128, 72, 8, False)


# ── memory guard for depth / cutout (I4) ────────────────────────────────────
def _clip_info(seconds: float, w=1920, h=1080, fps=24.0):
    from backend.finish.video_io import ClipInfo
    return ClipInfo(path="/x.mkv", start_ms=0, end_ms=int(seconds * 1000), fps=fps,
                    width=w, height=h, has_audio=False)


def test_memory_guard_refuses_long_depth_and_matte(monkeypatch):
    from backend.finish import runner

    monkeypatch.setattr(runner, "available_ram", lambda: 8 * 2**30)
    # 1080p cutout: 1920x1080 x 23 B per frame -> 60 s at 24 fps is ~64 GB
    with pytest.raises(runner.ClipTooLong) as exc:
        runner.check_memory("matte", _clip_info(60), {}, {"mask_max_dim": 1080})
    msg = str(exc.value)
    assert "too long for cutout" in msg and "60 s" in msg and "GB" in msg and "Split it" in msg
    runner.check_memory("matte", _clip_info(2), {}, {"mask_max_dim": 1080})  # ~2.7 GB: fine
    # depth works at the model resolution (518 short side), not the source's
    assert runner.work_size("depth", _clip_info(1), {"depth_input_size": 518}) == (921, 518)
    with pytest.raises(runner.ClipTooLong, match="too long for depth"):
        runner.check_memory("depth", _clip_info(300), {"depth_input_size": 518})
    runner.check_memory("depth", _clip_info(10), {"depth_input_size": 518})
    runner.check_memory("upscale", _clip_info(3600), {})  # streamed: never refused


def test_memory_guard_duration_cap_without_ram_info(monkeypatch):
    from backend.finish import runner

    monkeypatch.setattr(runner, "available_ram", lambda: None)
    runner.check_memory("matte", _clip_info(60, 64, 36), {}, {})
    with pytest.raises(runner.ClipTooLong, match="limit is 180 s"):
        runner.check_memory("matte", _clip_info(181, 64, 36), {}, {})


def test_depth_job_refuses_long_clip_before_loading_the_model(monkeypatch, tmp_path):
    runner = _patch_runner(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "available_ram", lambda: 2**30)
    monkeypatch.setattr(runner, "_depth", lambda *a: pytest.fail("model loaded"))
    with pytest.raises(runner.ClipTooLong):
        runner._run_depth(_Ctx(), _clip_info(120), "clip", tmp_path / "o", {})


def test_api_refuses_too_long_cutout_with_400(tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    sys.path.insert(0, str(ROOT / "backend"))
    from backend.api import finish as api_finish
    from backend.finish import runner
    from backend.server import make_app

    monkeypatch.setattr(api_finish, "_check_ready", lambda kind, params: None)
    monkeypatch.setattr(runner, "available_ram", lambda: 4 * 2**30)
    monkeypatch.setattr(runner, "resolve_source", lambda src: (_clip_info(90), "clip"))
    c = TestClient(make_app())
    before = c.get("/api/finish/jobs").json()["jobs"]
    r = c.post("/api/finish/jobs", json={"kind": "matte", "sources": [{"path": "/x.mkv"}], "params": {}})
    assert r.status_code == 400
    assert "too long for cutout" in r.json()["detail"] and "Split it" in r.json()["detail"]
    r = c.post("/api/finish/jobs", json={"kind": "depth", "sources": [{"scene_id": 1, "label": "A"},
                                                                      {"scene_id": 2, "label": "B"}],
                                         "params": {}})
    assert r.status_code == 400 and r.json()["detail"].startswith("A: ")
    assert c.get("/api/finish/jobs").json()["jobs"] == before  # nothing queued


def test_stabilizer_drops_flow_cache_as_it_goes():
    from backend.finish.temporal import FlowWarper, stabilize_stack

    frames, alpha = _moving_disc(n=6)
    warper = FlowWarper(frames)
    stabilize_stack(alpha, cuts=set(), strength=0.5, warper=warper)
    assert warper._cache == {}


# ── atomic alpha export (I5) ────────────────────────────────────────────────
@pytest.mark.parametrize("raw_pipes", [False, True])
def test_alpha_export_failure_leaves_no_file(finish_clip, tmp_path, monkeypatch, raw_pipes):
    from backend.export import ffmpeg as ff
    import backend.models.mask_postprocess as mp

    calls = {"n": 0}

    def boom(rgb, alpha, **kw):
        calls["n"] += 1
        if calls["n"] >= 3:
            raise RuntimeError("edge cleanup failed")
        return rgb, alpha

    monkeypatch.setattr(mp, "clean_edges_with_background", boom)
    # 0.6 default export path (edge colour decontamination)
    import backend.models.edge_refine as er
    monkeypatch.setattr(er, "estimate_foreground", lambda rgb, alpha, **kw: boom(rgb, alpha)[0])
    target = tmp_path / "out" / "clip_cutout.mov"
    masks = np.full((8, 72, 128), 0.5, np.float32)
    with pytest.raises(RuntimeError, match="edge cleanup failed"):
        ff.export_scene_with_alpha(str(finish_clip), 0, 8000 // 24, masks, str(target),
                                   fps=24.0, edge_refine=False, raw_pipes=raw_pipes)
    assert not target.exists()
    assert not ff.alpha_partial_path(str(target)).exists()
    assert not [p for p in target.parent.glob("*") if p.is_file()]


def test_alpha_export_success_renames_partial(finish_clip, tmp_path):
    from backend.export import ffmpeg as ff

    target = tmp_path / "clip_cutout.webm"
    masks = np.full((8, 72, 128), 0.5, np.float32)
    out = ff.export_scene_with_alpha(str(finish_clip), 0, 8000 // 24, masks, str(target), codec="vp9_alpha",
                                     fps=24.0, edge_refine=False, bg_aware_cleanup=False, raw_pipes=True)
    assert out == str(target) and target.stat().st_size > 0
    assert not ff.alpha_partial_path(str(target)).exists()


def test_cancelled_cutout_stops_encoding_and_leaves_no_file(finish_clip, tmp_path, monkeypatch):
    runner = _patch_runner(monkeypatch, tmp_path)
    import backend.models.mask_postprocess as mp
    import backend.state as state_mod
    from backend.finish.jobs import JobCancelled
    from backend.finish.video_io import probe_clip

    ctx = _Ctx()
    seen = {"n": 0}

    class FakeMatte:
        def mask_image(self, img):
            return np.full((img.height, img.width), 0.5, np.float32)

    class FakeState:
        settings = {"models": {"mask_bg_aware_cleanup_enabled": True, "mask_edge_refine_enabled": False}}

        def get_birefnet(self):
            return FakeMatte()

    def cancel_mid_export(rgb, alpha, **kw):
        seen["n"] += 1
        ctx.job.cancel_event.set()   # the user presses Cancel during encoding
        return rgb, alpha

    monkeypatch.setattr(state_mod, "get_state", lambda: FakeState())
    monkeypatch.setattr(mp, "clean_edges_with_background", cancel_mid_export)
    import backend.models.edge_refine as er   # 0.6 default export path
    monkeypatch.setattr(er, "estimate_foreground", lambda rgb, alpha, **kw: cancel_mid_export(rgb, alpha)[0])
    out_dir = tmp_path / "o"
    with pytest.raises(JobCancelled):
        runner._run_matte(ctx, probe_clip(str(finish_clip)), "clip", out_dir, {"stabilize": False})
    assert seen["n"] == 1, "encoding must stop at the next frame"
    for fn in ctx.cleanups:
        fn()
    assert not [p for p in out_dir.glob("*") if p.is_file()]


def teardown_module(_m):
    shutil.rmtree(_DATA, ignore_errors=True)


def test_model_delete_refused_while_downloading(client, monkeypatch):
    from backend.finish import registry

    mid = next(iter(registry.MODELS))
    monkeypatch.setitem(registry._downloads, mid, {"running": True, "percent": 5, "error": None})
    res = client.delete(f"/api/finish/models/{mid}")
    assert res.status_code == 409
