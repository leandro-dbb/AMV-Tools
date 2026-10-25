"""CPU-runnable tests for the NVIDIA performance paths (single-pass decode,
ffmpeg selection, overlapped finishing I/O, parallel motion estimates).
The CUDA-only code is exercised with synthetic inputs; real-GPU coverage
lives in ``tests/bench_pipeline.py``."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from backend.indexing import cuts, onepass

W, H = onepass.FRAME_W, onepass.FRAME_H


def _nv12(luma: int, idx: int) -> bytes:
    """Stacked frame as ffmpeg emits it: 512×288 luma, the 160×90 gray plane
    underneath, then the interleaved chroma."""
    y = np.full((onepass.STACK_H, W), luma, dtype=np.uint8)
    y[H:, onepass.GRAY_W:] = 0
    y[0, :4] = np.frombuffer(int(idx).to_bytes(4, "little"), dtype=np.uint8)  # frame id
    uv = np.full((onepass.STACK_H // 2, W), 128, dtype=np.uint8)
    out = y.tobytes() + uv.tobytes()
    assert len(out) == onepass.FRAME_BYTES
    return out


def _frame_id(buf: bytes) -> int:
    return int.from_bytes(buf[:4], "little")


def _expected_median_idx(m: int, fps: float) -> int:
    i = 0
    while int(i / fps * 1000) < m:
        i += 1
    return i


@pytest.mark.parametrize("lengths", [[40, 9, 120, 33, 64], [8, 8, 8], [300]])
def test_scan_picks_the_exact_median_frame(monkeypatch, lengths):
    monkeypatch.setattr(onepass, "_nv12_to_rgb", _frame_id)
    fps = 23.976
    lumas = [20, 220, 60, 180, 100]
    frames, idx = [], 0
    for k, n in enumerate(lengths):
        for _ in range(n):
            frames.append(_nv12(lumas[k % len(lumas)], idx))
            idx += 1
    duration_ms = int(len(frames) / fps * 1000)
    cut_list, medians, n = onepass.scan(iter(frames), fps, duration_ms, threshold=0.4)
    assert n == len(frames)
    assert cut_list[-1] == duration_ms
    # Scene changes land on the first frame of each new shot (the sampled
    # detection is refined among the frames since the previous sample).
    starts = np.cumsum([0] + lengths[:-1])
    expect_cuts = [0] + [int(s / fps * 1000) for s in starts[1:]]
    assert cut_list[:-1] == expect_cuts
    for s, e in zip(cut_list, cut_list[1:]):
        m = s + (e - s) // 2
        assert medians[m] == _expected_median_idx(m, fps)


def test_scan_overflow_leaves_long_scene_to_the_fallback(monkeypatch):
    monkeypatch.setattr(onepass, "_nv12_to_rgb", _frame_id)
    monkeypatch.setattr(onepass, "_MAX_BUFFER", 20)
    fps = 24.0
    frames = [_nv12(20, i) for i in range(16)] + [_nv12(220, 16 + i) for i in range(200)] \
        + [_nv12(60, 216 + i) for i in range(16)]
    cut_list, medians, _ = onepass.scan(iter(frames), fps, int(len(frames) / fps * 1000), 0.4)
    scenes = list(zip(cut_list, cut_list[1:]))
    assert len(scenes) == 3
    got = {s for s, e in scenes if s + (e - s) // 2 in medians}
    assert got == {scenes[0][0], scenes[2][0]}   # the 200-frame shot overflowed


def test_stacked_frame_unpacking():
    buf = bytearray(_nv12(40, 7))
    plane = np.frombuffer(buf, dtype=np.uint8)
    y = plane[:W * onepass.STACK_H].reshape(onepass.STACK_H, W)
    y[H:, :onepass.GRAY_W] = 200          # the detector's gray plane
    g = onepass._gray_of(bytes(buf))
    assert g.shape == (cuts._DIFF_H, cuts._DIFF_W) and (g == 200).all()
    rgb = np.asarray(onepass._nv12_to_rgb(bytes(buf)))
    assert rgb.shape == (H, W, 3)
    assert abs(int(rgb[100, 100, 0]) - int(round((40 - 16) * 255 / 219))) <= 2   # gray 40 → RGB ≈ 28


def test_scan_feeds_motion_sink_with_strided_gray():
    seen = []
    frames = [_nv12(50 + i, i) for i in range(17)]
    onepass.scan(iter(frames), 24.0, 700, 0.4, frame_sink=lambda ts, g: seen.append((ts, g.shape)))
    assert [t for t, _ in seen] == [int(i / 24.0 * 1000) for i in range(0, 17, cuts._FRAME_STRIDE)]
    assert all(shape == (cuts._DIFF_H, cuts._DIFF_W) for _, shape in seen)


def test_onepass_only_for_the_cuda_profile(monkeypatch):
    monkeypatch.setattr(cuts, "_ffmpeg_hw_profile", lambda: {"name": "videotoolbox"})
    assert not onepass.available()
    monkeypatch.setattr(cuts, "_ffmpeg_hw_profile", lambda: None)
    assert not onepass.available()
    monkeypatch.setattr(cuts, "_ffmpeg_hw_profile", lambda: {"name": "cuda"})
    assert onepass.available()


# ── ffmpeg selection ────────────────────────────────────────────────────────
def test_resolve_ffmpeg_prefers_a_cuda_build_on_nvidia(monkeypatch, tmp_path):
    import imageio_ffmpeg
    monkeypatch.delenv("AMV_FFMPEG", raising=False)
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: "/bundled/ffmpeg")
    monkeypatch.setattr(cuts.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(cuts.sys, "platform", "linux")
    caps = {"/bundled/ffmpeg": False, "/usr/bin/ffmpeg": True}
    monkeypatch.setattr(cuts, "_ffmpeg_has_cuda", lambda p: caps[p])

    monkeypatch.setattr(cuts, "_nvidia_gpu_present", lambda: True)
    assert cuts._resolve_ffmpeg() == "/usr/bin/ffmpeg"
    caps["/bundled/ffmpeg"] = True
    assert cuts._resolve_ffmpeg() == "/bundled/ffmpeg"
    # No NVIDIA GPU (CPU, ROCm, Apple...): the bundled build, as before.
    caps["/bundled/ffmpeg"] = False
    monkeypatch.setattr(cuts, "_nvidia_gpu_present", lambda: False)
    assert cuts._resolve_ffmpeg() == "/bundled/ffmpeg"
    forced = tmp_path / "ffmpeg"
    forced.write_text("")
    monkeypatch.setenv("AMV_FFMPEG", str(forced))
    assert cuts._resolve_ffmpeg() == str(forced)


def test_resolve_ffmpeg_never_swaps_on_macos(monkeypatch):
    import imageio_ffmpeg
    monkeypatch.delenv("AMV_FFMPEG", raising=False)
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: "/bundled/ffmpeg")
    monkeypatch.setattr(cuts.sys, "platform", "darwin")
    monkeypatch.setattr(cuts, "_nvidia_gpu_present", lambda: True)
    monkeypatch.setattr(cuts, "_ffmpeg_has_cuda", lambda p: pytest.fail("probed on macOS"))
    assert cuts._resolve_ffmpeg() == "/bundled/ffmpeg"


# ── overlapped finishing I/O ────────────────────────────────────────────────
def test_prefetch_preserves_order_and_propagates_errors():
    from backend.finish.overlap import prefetch

    assert list(prefetch(iter(range(50)), depth=3)) == list(range(50))

    def boom():
        yield 1
        raise RuntimeError("decode failed")

    it = prefetch(boom())
    assert next(it) == 1
    with pytest.raises(RuntimeError, match="decode failed"):
        next(it)


def test_async_writer_keeps_order_and_reports_failures():
    from backend.finish.overlap import AsyncWriter

    class Sink:
        def __init__(self, fail_at=None):
            self.got, self.fail_at, self.aborted = [], fail_at, False
            self.frames_written = 0

        def write(self, f):
            if f == self.fail_at:
                raise RuntimeError("encoder died")
            self.got.append(f)
            self.frames_written += 1

        def close(self):
            return "out.mp4"

        def abort(self):
            self.aborted = True

    sink = Sink()
    w = AsyncWriter(sink, depth=2)
    for i in range(30):
        w.write(i)
    assert w.close() == "out.mp4"
    assert sink.got == list(range(30))

    bad = Sink(fail_at=3)
    w = AsyncWriter(bad, depth=2)
    with pytest.raises(RuntimeError, match="encoder died"):
        for i in range(30):
            w.write(i)
        w.close()
    assert bad.aborted or bad.got == [0, 1, 2]


def test_async_writer_abort_with_a_full_queue_ends_the_thread():
    import threading

    from backend.finish.overlap import AsyncWriter

    gate = threading.Event()

    class Slow:
        frames_written = 0

        def write(self, f):
            gate.wait()

        def abort(self):
            gate.set()   # killing ffmpeg unblocks the pending write

    w = AsyncWriter(Slow(), depth=2)
    for i in range(3):   # one being written + a full queue
        w.write(i)
    w.abort()
    assert not w._t.is_alive()


def test_finish_overlap_is_cuda_and_mps_only(monkeypatch):
    from backend.finish import runner

    class Dev:
        backend = "cpu"

    class Reader(list):
        width, height = 1920, 1080

    monkeypatch.setattr(runner, "_device", lambda: Dev())
    assert runner._overlap() is False
    src = Reader([1, 2])
    assert runner._frames(src) is src
    Dev.backend = "mps"
    assert runner._overlap() is True and runner._i420_in_writer() is False
    Dev.backend = "cuda"
    assert runner._overlap() is True and runner._i420_in_writer() is True
    assert list(runner._frames(Reader([1, 2]))) == [1, 2]
    # Queues stay within ~256 MB: 4 frames of 1080p, 2 of 8K.
    assert runner._queue_depth(1920 * 1080 * 3) == 4
    assert runner._queue_depth(7680 * 4320 * 3) == 2


# ── motion estimates on a process pool ──────────────────────────────────────────────
def test_motion_pool_gives_identical_results():
    cv2 = pytest.importorskip("cv2")
    from backend.searchplus.analysis import FrameStatsAccumulator

    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur((rng.random((140, 240)) * 255).astype(np.uint8), (5, 5), 0)
    frames = [np.ascontiguousarray(base[i % 30:i % 30 + 90, (2 * i) % 60:(2 * i) % 60 + 160])
              for i in range(120)]
    runs = []
    for processes in (0, 2):
        acc = FrameStatsAccumulator().start_async(processes=processes)
        for i, f in enumerate(frames):
            acc.sink(i * 166, f)
        acc.finish()
        runs.append((acc.p_tx, acc.p_ty, acc.p_zoom, acc.p_valid, acc.p_diff))
    assert runs[0] == runs[1]
    assert sum(runs[0][3]) > 50
    assert threading.active_count() < 50


# ── queue: next video decoded during the current one's GPU phases ─────────
def test_index_queue_prefetches_next_video_decode(monkeypatch, tmp_path, fake_models, db_path):
    from PIL import Image

    from backend.db import queries, schema
    from backend.indexing import pipeline

    siglip, tagger = fake_models
    files = []
    for name in ("ep01.mkv", "ep02.mkv", "ep03.mkv"):
        p = tmp_path / name
        p.write_bytes(b"x")
        files.append(str(p))
    calls: list[tuple[str, str]] = []

    def fake_decode(path, threshold, frame_sink=None, cancelled=None):
        calls.append((path, threading.current_thread().name))
        img = Image.new("RGB", (64, 36), (200, 40, 40))
        return [(0, 1000), (1000, 2000)], {500: img, 1500: img}

    monkeypatch.setattr(onepass, "available", lambda: True)
    monkeypatch.setattr(onepass, "detect_cuts_and_medians", fake_decode)
    monkeypatch.setattr(cuts, "probe_video", lambda p: {"duration_ms": 2000, "fps": 24.0,
                                                        "resolution": "64x36"})
    with schema.get_conn(db_path) as conn:
        for f in files:
            queries.add_to_queue(conn, f, False, False)

    assert pipeline.index_queue(db_path, siglip, tagger, mode="fast", batch_size=2) == "completed"
    # One decode per file, all ahead of the index loop (two at a time).
    assert sorted(c[0] for c in calls) == files
    assert all(c[1] == "index-prefetch" for c in calls)
    with schema.get_conn(db_path) as conn:
        rows = conn.execute("SELECT status, (SELECT COUNT(*) FROM scenes s WHERE s.video_id = v.id) "
                            "FROM videos v ORDER BY filepath").fetchall()
    assert rows == [("completed", 2)] * 3


def test_frame_writer_i420_input_only_for_yuv420_encoders(tmp_path, monkeypatch):
    from backend.finish import video_io

    started = []

    class FakeProc:
        stdin = stderr = None

        def __init__(self, cmd, **kw):
            started.append(cmd)
            import io
            self.stderr = io.BytesIO()
            self.stdin = io.BytesIO()

    monkeypatch.setattr(video_io.subprocess, "Popen", FakeProc)
    w = video_io.FrameWriter(str(tmp_path / "a.mp4"), width=64, height=36, fps=24, codec="libx264", yuv420_in=True)
    assert w._to_i420 and started[-1][started[-1].index("-pix_fmt") + 1] == "yuv420p"
    w.write(np.zeros((36, 64, 3), dtype=np.uint8))
    assert len(w._proc.stdin.getvalue()) == 64 * 36 * 3 // 2
    w = video_io.FrameWriter(str(tmp_path / "a.mov"), width=64, height=36, fps=24, codec="prores_ks", yuv420_in=True)
    assert not w._to_i420                       # 4:2:2 10-bit: ffmpeg converts
    w = video_io.FrameWriter(str(tmp_path / "b.mp4"), width=64, height=36, fps=24, codec="libx264")
    assert not w._to_i420                       # off unless asked (CUDA runner)


def test_motion_pool_failure_falls_back_in_thread():
    pytest.importorskip("cv2")
    from backend.searchplus.analysis import FrameStatsAccumulator

    class BrokenPool:
        def submit(self, *a, **k):
            raise RuntimeError("BrokenProcessPool")

        def shutdown(self, wait=True):
            pass

    rng = np.random.default_rng(3)
    base = (rng.random((120, 200)) * 255).astype(np.uint8)
    acc = FrameStatsAccumulator()
    acc._pool = BrokenPool()
    for i in range(6):
        acc.add(i * 166, np.ascontiguousarray(base[i:i + 90, i:i + 160]))
    acc.finish()
    assert acc._pool is None and not acc._broken
    assert len(acc.p_valid) == 5


# ── chantier 2: GPU motion estimates, proxy source, GPU colour conversion ──
def _textured_pair(dx: float, dy: float, seed: int = 0):
    import cv2
    rng = np.random.default_rng(seed)
    big = cv2.GaussianBlur((rng.random((400, 600)) * 255).astype(np.uint8), (0, 0), 2.0)
    big = cv2.GaussianBlur(((big > 128) * 200 + 30).astype(np.uint8), (0, 0), 0.8)
    crop = lambda img: cv2.resize(img[150:330, 200:520], (160, 90), interpolation=cv2.INTER_AREA)  # noqa: E731
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return crop(big), crop(cv2.warpAffine(big, m, (600, 400)))


def test_gpu_motion_estimator_matches_opencv():
    """Same estimator as analysis.estimate_pair (run here on the CPU device)."""
    from backend.searchplus import analysis, motion_gpu
    pairs = [_textured_pair(3.0, -2.0, 0), _textured_pair(-1.5, 0.5, 1)]
    flat = np.full((90, 160), 128, np.uint8)
    a = np.stack([p[0] for p in pairs] + [flat])
    b = np.stack([p[1] for p in pairs] + [flat])
    got = motion_gpu.GpuMotionEstimator("cpu").estimate(a, b)
    assert got[2] is None and analysis.estimate_pair(flat, flat) is None
    for (pa, pb), g in zip(pairs, got[:2]):
        ref = analysis.estimate_pair(pa, pb)
        assert g is not None and ref is not None
        assert np.allclose(g, ref, atol=2e-4), (g, ref)


def test_gpu_accumulator_fills_the_same_rows(monkeypatch):
    from backend.searchplus import analysis, motion_gpu
    pa, pb = _textured_pair(2.0, 1.0)
    frames = [pa, pb, pa, pb]
    cpu = analysis.FrameStatsAccumulator()
    gpu = motion_gpu.GpuFrameStatsAccumulator("cpu").start_async()
    for i, f in enumerate(frames):
        cpu.add(i * 167, f)
        gpu.sink(i * 167, f)
    gpu.finish()
    assert gpu.p_valid == cpu.p_valid and gpu.p_t0 == cpu.p_t0 and gpu.p_diff == cpu.p_diff
    assert np.allclose(gpu.p_tx, cpu.p_tx, atol=2e-4)


def test_onepass_writes_the_proxy_source_as_a_second_output(monkeypatch, tmp_path):
    """With a proxy cache the single pass adds a scale_cuda → NVENC branch
    (source frame rate and pts) and only keeps the file on success."""
    seen = {}

    class FakeProc:
        returncode = 0
        stdout = None
        stderr = iter(())

        def __init__(self, cmd, **kw):
            seen["cmd"] = cmd
            import io
            self.stdout = io.BytesIO(_nv12(100, 0) * 3)
            if "-y" in cmd:
                open(cmd[cmd.index("-y") + 1], "wb").write(b"x")

        def poll(self):
            return 0

    monkeypatch.setattr(cuts, "probe_video", lambda p: {"duration_ms": 125, "fps": 24.0})
    monkeypatch.setattr(cuts, "_ffmpeg_hw_profile", lambda: {"name": "cuda", "input_args": []})
    monkeypatch.setattr(onepass, "_gpu_stacking", lambda: True)
    monkeypatch.setattr(onepass, "_color_info", lambda p: ("bt709", False))
    monkeypatch.setattr(onepass.subprocess, "Popen", FakeProc)
    video = tmp_path / "v.mkv"
    video.write_bytes(b"v")
    monkeypatch.setattr(onepass, "_proxy_cache_dir", tmp_path / "src")
    onepass.detect_cuts_and_medians(str(video), 0.4)
    cmd = seen["cmd"]
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert graph.startswith("[0:v]split=4[pxin]")
    assert "[pxin]fps=15,scale_cuda" in graph and cmd[cmd.index("-qp") + 1] == "20"
    assert cmd.count("-map") == 2 and "h264_nvenc" in cmd
    assert onepass.proxy_source_for(str(video)) is not None
    monkeypatch.setattr(onepass, "_proxy_cache_dir", None)
    onepass.detect_cuts_and_medians(str(video), 0.4)
    cmd = seen["cmd"]
    assert cmd[cmd.index("-filter_complex") + 1].startswith("[0:v]split=3[a]") and "h264_nvenc" not in cmd


def test_gpu_nv12_conversion_falls_back_to_numpy(monkeypatch):
    buf = _nv12(120, 0)
    monkeypatch.setattr(onepass, "_GPU_RGB", False)
    a = np.asarray(onepass._nv12_to_rgb_gpu(buf, "bt709", False))
    assert np.array_equal(a, np.asarray(onepass._nv12_to_rgb(buf, "bt709", False)))


def test_cut_histogram_is_unchanged_by_the_bincount_path():
    rng = np.random.default_rng(3)
    for i in range(50):
        a = rng.integers(0, 256, (cuts._DIFF_H, cuts._DIFF_W)).astype(np.uint8)
        if i % 5 == 0:
            a[:] = rng.integers(0, 256)
        ref, _ = np.histogram(a, bins=cuts._HIST_BINS, range=(0, 256))
        ref = ref.astype(np.float32)
        ref = ref / ref.sum()
        got = cuts._hist_from_plane_bytes(a.tobytes(), cuts._DIFF_W, cuts._DIFF_H)
        assert got.dtype == ref.dtype and np.array_equal(got, ref)


def test_gpu_i420_matches_opencv():
    import cv2
    import torch
    from backend.finish.yuv_cuda import rgb255_to_i420
    rng = np.random.default_rng(5)
    rgb = rng.integers(0, 256, (36, 64, 3)).astype(np.uint8)
    ref = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420).reshape(-1).astype(int)
    got = rgb255_to_i420(torch.from_numpy(rgb).float())
    assert (got.width, got.height) == (64, 36)
    assert np.abs(got.data.astype(int) - ref).max() <= 1


def test_frame_writer_takes_gpu_i420_frames(tmp_path, monkeypatch):
    import torch
    from backend.finish import video_io
    from backend.finish.yuv_cuda import rgb255_to_i420
    w = video_io.FrameWriter(str(tmp_path / "o.mp4"), width=64, height=36, fps=24.0, codec="libx264",
                             yuv420_in=True)
    assert w.accepts_i420
    frame = np.full((36, 64, 3), 90, np.uint8)
    w.write(rgb255_to_i420(torch.from_numpy(frame).float()))
    w.write(frame)
    out = w.close()
    assert w.frames_written == 2 and (tmp_path / "o.mp4").exists() and out
    plain = video_io.FrameWriter(str(tmp_path / "p.mp4"), width=64, height=36, fps=24.0, codec="libx264")
    with pytest.raises(ValueError):
        plain.write(rgb255_to_i420(torch.from_numpy(frame).float()))
    plain.abort()



def test_index_queue_keeps_two_decodes_in_flight(monkeypatch, tmp_path, fake_models, db_path):
    """NVIDIA queues: two single passes run at once (one ffmpeg pipeline
    leaves NVDEC partly idle), never more, each video decoded once."""
    import time
    from PIL import Image

    from backend.db import queries, schema
    from backend.indexing import pipeline

    siglip, tagger = fake_models
    files = []
    for name in ("ep01.mkv", "ep02.mkv", "ep03.mkv", "ep04.mkv"):
        p = tmp_path / name
        p.write_bytes(b"x")
        files.append(str(p))
    lock = threading.Lock()
    state = {"now": 0, "max": 0, "calls": []}

    def fake_decode(path, threshold, frame_sink=None, cancelled=None):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
            state["calls"].append(path)
        time.sleep(0.15)
        with lock:
            state["now"] -= 1
        img = Image.new("RGB", (64, 36), (200, 40, 40))
        return [(0, 1000), (1000, 2000)], {500: img, 1500: img}

    monkeypatch.setattr(onepass, "available", lambda: True)
    monkeypatch.setattr(onepass, "detect_cuts_and_medians", fake_decode)
    monkeypatch.setattr(cuts, "probe_video", lambda p: {"duration_ms": 2000, "fps": 24.0,
                                                        "resolution": "64x36"})
    with schema.get_conn(db_path) as conn:
        for f in files:
            queries.add_to_queue(conn, f, False, False)
    assert pipeline.index_queue(db_path, siglip, tagger, mode="fast", batch_size=2) == "completed"
    assert sorted(state["calls"]) == files and state["max"] == 2
    with schema.get_conn(db_path) as conn:
        rows = conn.execute("SELECT status FROM videos").fetchall()
    assert rows == [("completed",)] * 4


def test_batched_proxies_split_on_nearest_key_frames_and_chunks(state, tmp_path, monkeypatch):
    """The proxy pass cuts the 240p copy in time chunks on scene edges, and
    tolerates key frames placed on the frame nearest a cut (half a 15 fps
    frame early): without -segment_time_delta the muxer skipped such cuts and
    every later proxy belonged to the previous scene."""
    import re
    import subprocess as sp

    from backend.export import ffmpeg as exp
    from backend.indexing import cuts as cuts_mod

    video = tmp_path / "ep.mkv"
    video.write_bytes(b"v")
    cache = tmp_path / "src"
    cache.mkdir()
    (cache / onepass._proxy_source_name(str(video))).write_bytes(b"copy")
    scenes = [(i + 1, i * 2000, (i + 1) * 2000) for i in range(9)]   # 9 scenes of 2 s
    cmds = []

    class FakePopen:
        def __init__(self, cmd, **kw):
            self.returncode = 0
            if "segment" not in cmd:          # ffmpeg capability probes
                return
            cmds.append(cmd)
            pattern = cmd[-1]
            times = cmd[cmd.index("-segment_times") + 1].split(",") if "-segment_times" in cmd else []
            for k in range(len(times) + 1):
                Path(pattern % k).write_bytes(b"seg")

        def communicate(self):
            return b"", b""

    from pathlib import Path
    monkeypatch.setattr(exp, "_hw_h264_encoder", lambda: "h264_nvenc")
    monkeypatch.setattr(cuts_mod, "_ffmpeg_hw_profile", lambda: {"name": "cuda", "input_args": []})
    monkeypatch.setattr(cuts_mod, "_ffmpeg_path", lambda: "ffmpeg")
    monkeypatch.setattr(onepass, "has_decoder", lambda name: True)
    monkeypatch.setattr(sp, "Popen", FakePopen)
    proxies = tmp_path / "proxies"
    proxies.mkdir()
    state._proxy_children = {}
    done: set = set()
    n = state._proxies_batched(str(video), scenes, proxies, "medium", done, source_cache=cache, chunks=3)
    assert n == 9 and done == set(range(1, 10))
    assert len(cmds) == 3
    for c in cmds:
        assert c[c.index("-segment_time_delta") + 1] == "0.034"
        assert "h264_cuvid" in c
    # chunk starts are scene edges, segment times relative to them
    starts = [float(c[c.index("-ss") + 1]) if "-ss" in c else 0.0 for c in cmds]
    assert starts[0] == 0.0 and all(re.fullmatch(r"\d+\.000", f"{s:.3f}") for s in starts)
    assert all((proxies / f"scene_{i}.mp4").exists() for i in range(1, 10))
