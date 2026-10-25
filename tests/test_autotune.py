"""Automatic tuning: choice rules, fingerprint, cache invalidation, fallback
to the former constants, environment priority (measurements mocked)."""
from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from backend import autotune
from backend.autotune import bench, cache, choose, fingerprint

FP_A = {"schema": 1, "os": "linux", "gpu": "NVIDIA GeForce RTX 3090", "driver": "580.1",
        "torch": "2.8.0", "ffmpeg": "ffmpeg version 7.1", "app": "0.4.0a0"}


@pytest.fixture
def tuned(tmp_path, monkeypatch):
    """Isolated data folder + a fixed fingerprint (switch with ``use``)."""
    monkeypatch.setenv("AMV_DATA_DIR", str(tmp_path))
    state = {"fp": dict(FP_A)}

    def use(**changes):
        state["fp"] = {**FP_A, **changes}
        autotune._forget()

    monkeypatch.setattr(fingerprint, "current", lambda: (fingerprint.key_of(state["fp"]), state["fp"]))
    autotune._forget()
    yield use
    autotune._forget()


def _rec(value, default, measures=None):
    return {"value": value, "default": default, "unit": "s", "measures": measures or {}}


# ── choice rules ────────────────────────────────────────────────────────────
def test_smallest_within_keeps_the_smaller_count_under_5_percent():
    # the RTX 3090 shape: 2 decodes keep the single NVDEC busy, 4 add nothing
    assert choose.smallest_within({1: 3.1, 2: 2.0, 4: 1.95}) == 2
    assert choose.smallest_within({1: 3.1, 2: 2.0, 4: 1.85}) == 4
    assert choose.smallest_within({2: 1.0, 4: 0.6, 8: 0.45, 12: 0.44}) == 8
    assert choose.smallest_within({1: 1.0, 2: 1.0}) == 1


def test_smallest_within_ignores_failures():
    assert choose.smallest_within({1: 2.0, 2: 1.0, 4: None}) == 2
    assert choose.smallest_within({1: float("inf"), 2: 1.0}) == 2
    assert choose.smallest_within({}) is None
    assert choose.smallest_within({1: None}) is None


def test_against_reference_needs_a_5_percent_gain():
    assert choose.against_reference({"mps": 1.0, "coreml": 0.97}, "mps") == "mps"
    assert choose.against_reference({"mps": 1.0, "coreml": 0.7}, "mps") == "coreml"
    assert choose.against_reference({"mps": 1.0, "coreml": None}, "mps") == "mps"
    assert choose.against_reference({}, 64) == 64
    assert choose.against_reference({8: 0.010, 16: 0.0098, 32: 0.0090}, 32) == 32
    assert choose.against_reference({8: 0.010, 16: 0.0080, 32: None}, 32) == 16   # reference failed (OOM)


def test_equivalent_outputs():
    rng = np.random.default_rng(0)
    ref = rng.random((16, 50)).astype(np.float32)
    assert choose.equivalent(ref, ref + 0.004)
    assert not choose.equivalent(ref, ref + 0.05)
    shuffled = ref.copy()
    shuffled[:, :] = ref[:, ::-1]
    assert not choose.equivalent(ref, shuffled)
    assert not choose.equivalent(ref, ref[:8])
    bad = ref.copy()
    bad[0, 0] = np.nan
    assert not choose.equivalent(ref, bad)


def test_time_parallel_finds_the_engine_width():
    """A decoder with two units: 2 jobs in flight halve the time per job, 4
    don't add anything -> 2 is kept."""
    units = threading.Semaphore(2)

    def job():
        with units:
            time.sleep(0.04)

    measures = bench.time_parallel(job, (1, 2, 4), lambda: False)
    assert set(measures) <= {1, 2, 4} and 1 in measures and 2 in measures
    assert measures[2] < measures[1] * 0.7
    assert choose.smallest_within(measures) == 2


def test_time_parallel_stops_on_error():
    calls = {"n": 0}

    def job():
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("decoder gone")

    measures = bench.time_parallel(job, (1, 2, 4), lambda: False)
    assert measures[1] is not None and measures[2] is None and 4 not in measures


def test_batch_candidates_respect_the_memory_tier():
    assert bench.batch_candidates("cuda", 24.0, None) == ([8, 16, 32, 64], 64)
    assert bench.batch_candidates("cuda", 12.0, None) == ([8, 16, 32], 32)
    assert bench.batch_candidates("cuda", 8.0, None) == ([8, 16], 16)
    assert bench.batch_candidates("mps", None, 16.0) == ([4, 8, 16], 8)
    assert bench.batch_candidates("mps", None, 8.0) == ([4, 8], 8)
    with pytest.raises(bench.Skip):
        bench.batch_candidates("cpu", None, None)


def test_plan_per_platform():
    assert autotune.plan("nvidia") == ["parallel_decodes", "tensorrt", "batch_size"]
    assert autotune.plan("mac") == ["chunk_workers", "tagger_engine", "batch_size"]
    assert autotune.plan("cpu") == []


# ── fingerprint ─────────────────────────────────────────────────────────────
def test_fingerprint_key_changes_with_the_machine():
    base = fingerprint.key_of(FP_A)
    assert fingerprint.key_of(dict(FP_A)) == base
    for field, val in (("gpu", "NVIDIA GeForce RTX 5090"), ("driver", "590.2"), ("torch", "2.9.0"),
                       ("ffmpeg", "ffmpeg version 8.0"), ("app", "0.4.1")):
        assert fingerprint.key_of({**FP_A, field: val}) != base, field


def test_fingerprint_collect_on_linux(monkeypatch):
    monkeypatch.setattr(fingerprint.sys, "platform", "linux")

    def fake_run(cmd):
        if cmd[0] == "nvidia-smi":
            return "NVIDIA GeForce RTX 3090, 580.95.05"
        return "ffmpeg version 7.1 Copyright"

    monkeypatch.setattr(fingerprint, "_run", fake_run)
    fp = fingerprint.collect()
    assert fp["gpu"] == "NVIDIA GeForce RTX 3090" and fp["driver"] == "580.95.05"
    assert fp["schema"] == fingerprint.SCHEMA and "app" in fp and "torch" in fp


def test_fingerprint_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(fingerprint.sys, "platform", "linux")
    monkeypatch.setattr(fingerprint, "_run", lambda cmd: None)
    fp = fingerprint.collect()
    assert fp["gpu"] is None and fp["driver"] is None


# ── cache ───────────────────────────────────────────────────────────────────
def test_cache_roundtrip_and_pruning(tmp_path):
    p = tmp_path / "autotune.json"
    for i in range(cache.KEEP + 2):
        cache.store(f"k{i}", {"settings": {"batch_size": _rec(i, 8)}}, p)
    data = json.loads(p.read_text())
    assert list(data["results"]) == [f"k{i}" for i in range(2, cache.KEEP + 2)]
    assert cache.load("k3", p)["settings"]["batch_size"]["value"] == 3
    assert cache.load("k0", p) is None
    cache.drop("k3", p)
    assert cache.load("k3", p) is None


def test_corrupt_cache_means_not_measured(tmp_path):
    p = tmp_path / "autotune.json"
    p.write_text("{not json")
    assert cache.load("k", p) is None
    cache.store("k", {"settings": {}}, p)          # rewritten cleanly
    assert cache.load("k", p) == {"settings": {}}


# ── value(): cache, invalidation, defaults ─────────────────────────────────
def test_value_defaults_without_a_bench(tuned):
    assert autotune.value("parallel_decodes", 2) == 2
    assert autotune.value("chunk_workers", 8) == 8


def test_value_reads_this_machine_and_invalidates_on_change(tuned):
    autotune.run(measure={"parallel_decodes": lambda c: _rec(4, 2, {1: 3.0, 2: 2.0, 4: 1.5})},
                 platform="nvidia", source="none")
    assert autotune.value("parallel_decodes", 2) == 4
    tuned(driver="590.2")                     # driver update -> stale
    assert autotune.value("parallel_decodes", 2) == 2
    assert autotune.status()["entry"] is None
    tuned()                                   # back to the measured machine
    assert autotune.value("parallel_decodes", 2) == 4


def test_failed_and_skipped_measures_keep_the_defaults(tuned):
    def boom(_ctx):
        raise RuntimeError("NVDEC busy")

    def skip(_ctx):
        raise bench.Skip("TensorRT not installed")

    entry = autotune.run(measure={"parallel_decodes": boom, "tensorrt": skip,
                                  "batch_size": lambda c: _rec(32, 64, {32: 0.01, 64: 0.02}),
                                  "chunk_workers": lambda c: _rec(None, 8)},
                         platform="nvidia", source="none")
    assert "parallel_decodes" in entry["errors"] and "chunk_workers" in entry["errors"]
    assert entry["skipped"] == {"tensorrt": "TensorRT not installed"}
    assert autotune.value("parallel_decodes", 2) == 2
    assert autotune.value("chunk_workers", 8) == 8
    assert autotune.value("batch_size", None) == 32


def test_cancelled_run_stores_nothing(tuned):
    flag = {"on": False}

    def measure(_ctx):
        flag["on"] = True
        return _rec(4, 2)

    with pytest.raises(bench.Skip):
        autotune.run(measure={"parallel_decodes": measure}, cancelled=lambda: flag["on"],
                     platform="nvidia", source="none")
    assert autotune.value("parallel_decodes", 2) == 2


def test_value_survives_a_broken_reader(monkeypatch):
    def broken():
        raise OSError("disk gone")

    monkeypatch.setattr(fingerprint, "current", broken)
    autotune._forget()
    assert autotune.value("batch_size", 8) == 8
    autotune._forget()


# ── call sites: env first, then the tuning, then the constant ──────────────
def test_chunk_workers_priority(tuned, monkeypatch):
    from backend.indexing import chunked
    monkeypatch.delenv("AMV_CHUNK_WORKERS", raising=False)
    assert chunked.workers() == chunked.WORKERS
    autotune.run(measure={"chunk_workers": lambda c: _rec(12, 8)}, platform="mac", source="none")
    assert chunked.workers() == 12
    monkeypatch.setenv("AMV_CHUNK_WORKERS", "4")
    assert chunked.workers() == 4


def test_effective_batch_uses_the_tuning_within_the_tier(tuned):
    from backend.indexing import pipeline
    cuda24 = SimpleNamespace(device=SimpleNamespace(backend="cuda", vram_gb=24.0))
    cuda12 = SimpleNamespace(device=SimpleNamespace(backend="cuda", vram_gb=12.0))
    mps = SimpleNamespace(device=SimpleNamespace(backend="mps", vram_gb=None))
    cpu = SimpleNamespace(device=SimpleNamespace(backend="cpu", vram_gb=None))
    assert pipeline._effective_batch(cuda24, 8) == 64           # no bench: the former rule
    assert pipeline._effective_batch(mps, 8) == 8
    autotune.run(measure={"batch_size": lambda c: _rec(48, 64)}, platform="nvidia", source="none")
    assert pipeline._effective_batch(cuda24, 8) == 48
    assert pipeline._effective_batch(cuda12, 8) == 32           # never above the tier
    assert pipeline._effective_batch(mps, 8) == 48
    assert pipeline._effective_batch(cuda24, 16) == 16          # the user's setting wins
    assert pipeline._effective_batch(cpu, 8) == 8


# ── background thread + API ─────────────────────────────────────────────────
def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not pred():
        time.sleep(0.02)
    return pred()


def test_background_skips_a_fresh_cache_and_reruns_when_forced(tuned, monkeypatch):
    calls = []
    monkeypatch.setattr(autotune, "run", lambda **kw: calls.append(kw) or {})
    monkeypatch.setattr(autotune, "_busy", lambda: False)
    cache.store(fingerprint.current()[0], {"settings": {}})
    autotune._forget()
    assert autotune.start_background(force=True, delay=0.0)   # pytest: only forced runs start
    assert _wait(lambda: not autotune._thread.is_alive())
    assert len(calls) == 1 and autotune.status()["state"] == "done"
    assert autotune.start_background() is False                # not forced: off under pytest


def test_background_waits_while_indexing(tuned, monkeypatch):
    busy = {"on": True}
    calls = []
    monkeypatch.setattr(autotune, "IDLE_POLL_S", 0.05)
    monkeypatch.setattr(autotune, "_busy", lambda: busy["on"])
    monkeypatch.setattr(autotune, "run", lambda **kw: calls.append(kw) or {})
    autotune.start_background(force=True, delay=0.0)
    assert _wait(lambda: autotune.status()["state"] == "waiting")
    assert not calls
    busy["on"] = False
    assert _wait(lambda: autotune.status()["state"] == "done")
    assert len(calls) == 1


def test_background_failure_is_reported(tuned, monkeypatch):
    def boom(**_kw):
        raise RuntimeError("no GPU")

    monkeypatch.setattr(autotune, "_busy", lambda: False)
    monkeypatch.setattr(autotune, "run", boom)
    autotune.start_background(force=True, delay=0.0)
    assert _wait(lambda: autotune.status()["state"] == "failed")
    assert "no GPU" in autotune.status()["error"]
    assert autotune.value("parallel_decodes", 2) == 2


def test_api_status_and_rerun(tuned, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.api import autotune as api
    monkeypatch.setattr(autotune, "_busy", lambda: False)
    monkeypatch.setattr(autotune, "run", lambda **kw: {})
    app = FastAPI()
    app.include_router(api.router)
    client = TestClient(app)
    body = client.get("/api/autotune").json()
    assert body["entry"] is None and body["fingerprint"]["gpu"] == FP_A["gpu"]
    assert client.post("/api/autotune/run").json()["started"] is True
    assert _wait(lambda: not autotune._thread.is_alive())


def test_synthetic_clip_when_the_library_is_empty(tmp_path):
    """Real ffmpeg (CPU): the fallback clip decodes and has the clip length."""
    from backend.indexing import cuts
    clip, origin = bench.make_clip(tmp_path, None, loops=2)
    assert origin == "synthetic"
    assert abs(cuts.probe_video(str(clip))["duration_ms"] - 2 * bench.CLIP_S * 1000) < 1500


def test_synthetic_clip_keeps_the_default_count():
    # RTX 3090 on the synthetic clip: 1 decode looked 6-8 % faster; the default stays
    assert bench.pick_count({1: 1.306, 2: 1.395}, 2, "synthetic") == 2
    assert bench.pick_count({1: 1.292, 2: 1.397}, 2, "library") == 1
    assert bench.pick_count({2: 1.0, 4: 0.5, 8: 0.6}, 8, "synthetic") == 8
    assert bench.pick_count({2: 1.0, 4: 0.5}, 8, "synthetic") == 4    # default not measured


def test_synthetic_result_is_measured_again_once_the_library_has_a_video(monkeypatch):
    assert autotune.needs_run(None)
    assert not autotune.needs_run({"source": "library", "settings": {}})
    monkeypatch.setattr(bench, "library_video", lambda: None)
    assert not autotune.needs_run({"source": "synthetic", "settings": {}})
    monkeypatch.setattr(bench, "library_video", lambda: "/videos/ep01.mkv")
    assert autotune.needs_run({"source": "synthetic", "settings": {}})
