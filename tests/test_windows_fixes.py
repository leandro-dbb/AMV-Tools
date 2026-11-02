"""Fixes for what the Windows 11 + RTX 3090 test left open
(docs/reports/windows-rtx3090.md, « Corrections »)."""
from __future__ import annotations

import pytest


# ── 7. exported clip names ──────────────────────────────────────────────────
@pytest.mark.parametrize("display, anime, episode", [
    ("[Groupe] Dragon Ball Super - Beerus - 01 (1080p) [ABCD1234]", "Dragon Ball Super - Beerus", "01"),
    ("Black Clover S02E04 1080p", "Black Clover", "S02E04"),
    ("[Erai-raws] Shingeki no Kyojin - The Final Season - 28v2 [1080p]",
     "Shingeki no Kyojin - The Final Season", "28"),
    ("Naruto Episode 12", "Naruto", "12"),
    ("Kimetsu no Yaiba 第3話", "Kimetsu no Yaiba", "第3話"),
    ("Mon film été", "Mon film été", ""),
])
def test_split_title(display, anime, episode):
    from backend.export.naming import split_title
    assert split_title(display) == (anime, episode)


def test_exported_clip_name_has_the_episode_once(tmp_path, monkeypatch):
    """The default template ``{anime}_{episode}_scene_{id}`` got the whole
    file name twice: ``…ABCD1234_Groupe_…ABCD1234_scene_468.mp4``."""
    from backend.api import export as api_export
    seen = {}
    monkeypatch.setattr(api_export, "export_scene", lambda src, s, e, target, **kw: seen.setdefault("t", target))
    scene = {"id": 468, "display_name": "[Groupe] Dragon Ball Super - Beerus - 01 (1080p) [ABCD1234]",
             "filepath": "x.mkv", "start_ms": 0, "end_ms": 1000}
    settings = {"naming_template": "{anime}_{episode}_scene_{scene_id}", "codec": "auto", "crf": 18,
                "audio": "copy", "resolution": "source"}
    out = api_export._export_one(scene, settings, tmp_path)
    from pathlib import Path
    assert Path(out).name == "Dragon_Ball_Super_-_Beerus_01_scene_468.mp4"
    assert Path(out).name.count("Beerus") == 1


def test_clip_name_without_episode_number_drops_the_empty_field():
    from backend.export.naming import clip_stem
    assert clip_stem("{anime}_{episode}_scene_{scene_id}", "Mon film", 3) == "Mon_film_scene_3"
    assert clip_stem("{episode}-{anime}-{scene_id}", "Mon film", 3) == "Mon_film-3"
    assert clip_stem("{anime}_{scene_id}_{tags}", "Mon film", 3, tags="alpha") == "Mon_film_3_alpha"


# ── 6. slow motion: download button in the dialog ───────────────────────────
def test_capabilities_name_the_models_jobs_will_use(tmp_path, monkeypatch):
    """The dialog checked a fixed ``rife-4.25`` while the job ran the tier's
    RIFE 4.26: missing, the job failed with "Download it from Settings >
    Models" and the dialog had no download button. Capabilities now name the
    model each kind runs without a ``model`` param."""
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from backend.finish import runner
    from backend.server import make_app

    monkeypatch.setenv("AMV_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(runner, "_tier_models", lambda: {"interpolate": "rife-4.26", "depth": "depth-anything-v2-small"})
    caps = TestClient(make_app()).get("/api/finish/capabilities").json()
    assert caps["defaults"]["interpolate"] == "rife-4.26"
    assert caps["defaults"]["depth"] == "depth-anything-v2-small"
    rife = next(m for m in caps["models"] if m["id"] == "rife-4.26")
    assert rife["installed"] is False          # -> the dialog shows "Download"


# ── 4. first launch: the uv cache is pruned once the venv is installed ─────
@pytest.fixture
def fake_uv(tmp_path, monkeypatch):
    """A ``uv`` that logs its arguments and empties the cache like a prune."""
    import stat
    import sys
    log = tmp_path / "uv-calls.txt"
    exe = tmp_path / "uv"
    exe.write_text(f"#!{sys.executable}\nimport os, shutil, sys\n"
                   f"open({str(log)!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
                   "shutil.rmtree(os.path.join(os.environ['UV_CACHE_DIR'], 'archive-v0'), ignore_errors=True)\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    from backend.api import setup
    monkeypatch.setattr(setup, "_uv_exe", lambda: str(exe))
    cache = tmp_path / "uv-cache"
    (cache / "archive-v0" / "abc").mkdir(parents=True)
    (cache / "archive-v0" / "abc" / "torch.dll").write_bytes(b"x" * 4096)
    data = tmp_path / "data"
    monkeypatch.setenv("AMV_DATA_DIR", str(data))
    monkeypatch.setenv("UV_CACHE_DIR", str(cache))
    monkeypatch.setenv("AMV_UV_CACHE_PRUNE", "1")
    return log


@pytest.mark.skipif(__import__("sys").platform == "win32", reason="shebang fake uv")
def test_prune_runs_forced_ci_prune_and_reports_sizes(fake_uv):
    from backend import uvcache
    res = uvcache.prune("engine install")
    # --force: the `uv run` that started the sidecar holds the cache lock
    assert fake_uv.read_text().split() == ["cache", "prune", "--ci", "--force"]
    assert res["ok"] and res["cache_before"] == 4096 and res["cache_after"] == 0


@pytest.mark.skipif(__import__("sys").platform == "win32", reason="shebang fake uv")
def test_prune_after_update_runs_once_per_lock(fake_uv, monkeypatch):
    from backend import uvcache
    from backend.paths import user_data_dir
    assert uvcache.prune_after_update() is None            # engine not installed yet
    (user_data_dir() / ".setup_complete").write_text("cu130")
    t = uvcache.prune_after_update()
    assert t is not None
    t.join(10)
    assert uvcache.prune_after_update() is None            # same uv.lock: done
    assert len(fake_uv.read_text().splitlines()) == 1
    monkeypatch.setattr(uvcache, "_lock_sha", lambda: "an-updated-lock")
    t = uvcache.prune_after_update()
    t.join(10)
    assert len(fake_uv.read_text().splitlines()) == 2


def test_prune_is_off_outside_the_packaged_app(fake_uv, monkeypatch):
    from backend import uvcache
    monkeypatch.delenv("AMV_UV_CACHE_PRUNE")
    assert not uvcache.enabled() and uvcache.prune_in_background("x") is None


def test_real_uv_prune_keeps_a_hard_linked_venv_working(tmp_path, monkeypatch):
    """With the real uv: the venv it linked from the cache still imports
    after the prune, and a later sync (an app update) needs nothing more."""
    import shutil
    import subprocess
    import sys
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not on PATH")
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "pyproject.toml").write_text('[project]\nname = "p"\nversion = "0"\nrequires-python = ">=3.10"\n'
                                         'dependencies = ["six"]\n')
    env = {**__import__("os").environ, "UV_CACHE_DIR": str(tmp_path / "c"), "UV_PYTHON": sys.executable}
    r = subprocess.run([uv, "sync", "-q"], cwd=proj, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"uv sync needs the network: {r.stderr[-200:]}")
    monkeypatch.setenv("UV_CACHE_DIR", env["UV_CACHE_DIR"])
    from backend import uvcache
    from backend.api import setup
    monkeypatch.setattr(setup, "_uv_exe", lambda: uv)
    monkeypatch.setenv("AMV_DATA_DIR", str(tmp_path / "data"))
    res = uvcache.prune("test")
    assert res["ok"] and res["cache_after"] < res["cache_before"]
    py = proj / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    assert subprocess.run([str(py), "-c", "import six"]).returncode == 0
    r = subprocess.run([uv, "sync", "--offline"], cwd=proj, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ── 5. the autotune never makes the user wait ───────────────────────────────
@pytest.fixture
def quiet_autotune(tmp_path, monkeypatch):
    from backend import autotune
    from backend.autotune import fingerprint, yielding
    monkeypatch.setenv("AMV_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(fingerprint, "current", lambda: ("k1", {"gpu": "test"}))
    monkeypatch.setattr(autotune, "_busy", lambda: False)
    monkeypatch.setattr(autotune, "USER_POLL_S", 0.01)
    monkeypatch.setattr(yielding, "QUIET_S", 0.2)
    yielding.reset()
    autotune._forget()
    yield autotune
    autotune._cancel.set()
    if autotune._thread is not None:
        autotune._thread.join(5)
    autotune._cancel.clear()
    yielding.reset()


def _poll(pred, timeout=20.0):
    import time
    end = time.monotonic() + timeout
    while time.monotonic() < end and not pred():
        time.sleep(0.01)
    return pred()


def test_a_user_request_stops_the_bench_and_it_resumes_where_it_stopped(quiet_autotune, monkeypatch):
    """A search arriving mid-bench stops the measure in progress (its
    ``cancelled`` turns true); once the app is quiet the run resumes and
    measures only what is left."""
    from backend.autotune import yielding
    autotune = quiet_autotune
    calls = []

    def nvdec(ctx):
        calls.append("parallel_decodes")
        return {"value": 2, "default": 2}

    real_run = autotune.run
    monkeypatch.setattr(autotune, "run", lambda **kw: real_run(
        measure={"parallel_decodes": nvdec, "batch_size": lambda c: _batch_until_cancelled(c, calls)},
        platform="nvidia", source="none", **kw))
    autotune.start_background(force=True, delay=0.0)
    assert _poll(lambda: calls.count("batch_size") == 1)
    yielding.begin()                           # the search starts…
    assert _poll(lambda: autotune.status()["state"] == "waiting")
    yielding.end()                             # …and ends; quiet for QUIET_S
    assert _poll(lambda: autotune.status()["state"] == "done")
    assert calls == ["parallel_decodes", "batch_size", "batch_size"]   # nvdec not measured twice
    assert autotune.value("batch_size", None) == 32


def _batch_until_cancelled(ctx, calls):
    import time
    calls.append("batch_size")
    if calls.count("batch_size") == 1:
        while not ctx["cancelled"]():
            time.sleep(0.005)
        raise RuntimeError("decode cut short")   # an interrupted measure is not a failure
    return {"value": 32, "default": 64}


def test_the_bench_does_not_start_while_the_user_is_active(quiet_autotune, monkeypatch):
    from backend.autotune import yielding
    autotune = quiet_autotune
    calls = []
    monkeypatch.setattr(autotune, "run", lambda **kw: calls.append(1) or {})
    yielding.begin()
    autotune.start_background(force=True, delay=0.0)
    assert _poll(lambda: autotune.status()["state"] == "waiting")
    import time
    time.sleep(0.3)
    assert not calls
    yielding.end()
    assert _poll(lambda: autotune.status()["state"] == "done") and calls == [1]


@pytest.mark.parametrize("method, path, user", [
    ("POST", "/api/search", True),
    ("POST", "/api/search/image", True),
    ("GET", "/api/search/similar/12", True),
    ("GET", "/api/scene/12/proxy", True),
    ("GET", "/api/scene/12/thumbnail", True),
    ("GET", "/api/videos/3/stream", True),
    ("POST", "/api/finish/jobs", True),
    ("POST", "/api/export", True),
    ("GET", "/api/status", False),
    ("GET", "/api/autotune", False),
    ("POST", "/api/autotune/run", False),
    ("GET", "/api/index/queue", False),
    ("GET", "/api/finish/jobs", False),
    ("GET", "/api/diagnostics", False),
    ("GET", "/health", False),
])
def test_which_requests_count_as_the_user(method, path, user):
    from backend.autotune.yielding import is_user_request
    assert is_user_request(method, path) is user


def test_middleware_marks_search_requests(monkeypatch, tmp_path):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from backend.autotune import yielding
    from backend.server import make_app
    monkeypatch.setenv("AMV_DATA_DIR", str(tmp_path))
    seen = []
    monkeypatch.setattr(yielding, "begin", lambda: seen.append("begin"))
    monkeypatch.setattr(yielding, "end", lambda: seen.append("end"))
    c = TestClient(make_app())
    c.get("/api/status")
    assert seen == []
    c.get("/api/scene/987654/thumbnail")         # a user request (404 here: cheap)
    assert seen == ["begin", "end"]


def test_interrupted_measures_raise_skip_instead_of_storing_a_partial_result():
    """A decode or a batch cut short for the user must not become the
    stored measurement."""
    from backend.autotune import bench
    calls = {"n": 0}
    stop = {"on": False}

    def one():
        calls["n"] += 1
        stop["on"] = True
        raise RuntimeError("decode cancelled")

    with pytest.raises(bench.Skip):
        bench.time_parallel(one, [1, 2], lambda: stop["on"], jobs=2)

    class Fake:
        def embed_images(self, imgs):
            stop2["n"] += 1
    stop2 = {"n": 0}
    from types import SimpleNamespace
    import backend.state as st_mod
    fake_state = SimpleNamespace(device=SimpleNamespace(backend="cuda", vram_gb=24.0),
                                 get_tagger=lambda: Fake(), get_siglip=lambda: Fake())
    orig = st_mod.get_state
    st_mod.get_state = lambda: fake_state
    try:
        with pytest.raises(bench.Skip):
            bench.measure_batch([object()] * 256, lambda: stop2["n"] >= 4)
    finally:
        st_mod.get_state = orig
    assert stop2["n"] < 10            # stopped within a batch, not after 256 images


# ── 1. hover proxies encoded by the indexing decode (one NVENC session) ─────
def _slots_for(fps, n_frames, pfps=15):
    """Source frame of each 15 fps slot, as nvc_pass picks them."""
    out, j = [], 0
    while True:
        src = int(round(j * fps / pfps))
        if src >= n_frames:
            return out
        out.append(src)
        j += 1


class _FakeScanner:
    def __init__(self, fps, lookahead=40):
        self.fps, self.n = fps, 0
        self.cuts_ms = [0]
        self.detector = type("D", (), {"lookahead": lookahead})()

    def ts(self, idx):
        return int((idx / self.fps) * 1000)

    def first_at(self, t_ms):
        i = max(0, int(t_ms * self.fps / 1000.0))
        while i > 0 and self.ts(i - 1) >= t_ms:
            i -= 1
        while self.ts(i) < t_ms:
            i += 1
        return i


@pytest.fixture
def direct_writer(tmp_path, monkeypatch):
    from backend.indexing import nvc_pass, onepass
    monkeypatch.setattr(onepass, "_proxy_cache_dir", tmp_path)
    monkeypatch.setattr(onepass, "_proxy_direct", {"bitrate": 500_000})
    encoded = []

    class FakeNvc:
        class NV_ENC_PIC_PARAMS:
            encodePicFlags = 0
            inputTimeStamp = 0

        class NV_ENC_PIC_FLAGS:
            FORCEIDR = 4

    def fake_encode(self, frame, idr=False):
        encoded.append((self.n, idr, int(frame)))
        if idr:
            self.idr.append(self.n)
        self.n += 1

    monkeypatch.setattr(nvc_pass, "_module", lambda: FakeNvc)
    monkeypatch.setattr(nvc_pass._ProxyWriter, "_encode", fake_encode)
    monkeypatch.setattr(nvc_pass._ProxyWriter, "_open", lambda self: setattr(self, "enc", object()))
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: type("S", (), {"synchronize": lambda s: None})())
    return nvc_pass, encoded


@pytest.mark.parametrize("threaded", ["0", "1"])
def test_direct_proxies_put_an_idr_on_each_shots_first_slot_once_decided(direct_writer, monkeypatch, threaded):
    """Frames are held until the detector cannot cut before them any more;
    the first 15 fps frame of every shot is an IDR; the stream end drains.
    Same frames and IDRs with the encoder on its own thread (Windows)."""
    import torch
    nvc_pass, encoded = direct_writer
    monkeypatch.setenv("AMV_PROXY_ENC_THREAD", threaded)
    monkeypatch.setattr(torch.cuda, "Event", lambda **kw: type("E", (), {"record": lambda s, x=None: None,
                                                                         "synchronize": lambda s: None})())
    fps = 24000 / 1001
    n_frames = 2400
    cuts = [37, 38 + 24, 500, 1201, 2390]               # source frames that start a shot
    sc = _FakeScanner(fps)
    w = nvc_pass._ProxyWriter("ep.mkv", 426, 240, fps)
    assert w.direct
    slots = _slots_for(fps, n_frames)
    batch = nvc_pass._BATCH
    j = 0
    for first in range(0, n_frames, batch):
        sc.n = min(n_frames, first + batch)
        # the detector reports a cut ``lookahead`` frames late
        for c in cuts:
            if c + sc.detector.lookahead < sc.n and sc.ts(c) not in sc.cuts_ms:
                sc.cuts_ms.append(sc.ts(c))
        picks = []
        while j < len(slots) and slots[j] < sc.n:
            picks.append(slots[j])
            j += 1
        if picks:
            w.write(torch.tensor(picks), picks)
        w.release(sc)
        if threaded == "1":
            _poll(lambda: w._q is None or w._q.empty())
        # never encoded a frame the detector could still cut before
        if encoded:
            assert slots[encoded[-1][0]] < sc.n - sc.detector.lookahead
    for c in cuts:
        if sc.ts(c) not in sc.cuts_ms:
            sc.cuts_ms.append(sc.ts(c))
    sc.cuts_ms.append(int(n_frames / fps * 1000))        # finish() appends the duration
    w.release(sc, final=True)
    w.finish_encode()
    assert [e[0] for e in encoded] == list(range(len(slots)))
    expect = {0} | {next(k for k, s in enumerate(slots) if s >= c) for c in cuts}
    assert {e[0] for e in encoded if e[1]} == expect
    meta = {"fps": fps, "proxy_fps": 15, "idr": w.idr, "slots": w.n}
    from backend.indexing import onepass
    assert onepass.direct_split_ok([sc.ts(c) for c in cuts], meta)


def test_direct_split_check_refuses_a_missing_or_early_key():
    from backend.indexing import onepass
    fps = 24000 / 1001
    slots = _slots_for(fps, 2400)
    first_slot = lambda c: next(k for k, s in enumerate(slots) if s >= c)   # noqa: E731
    cuts = [100, 700, 1500]
    edges = [int(c / fps * 1000) for c in cuts]
    meta = {"fps": fps, "proxy_fps": 15, "slots": len(slots), "idr": [first_slot(c) for c in cuts]}
    assert onepass.direct_split_ok(edges, meta)
    assert not onepass.direct_split_ok(edges + [int(1900 / fps * 1000)], meta)        # shot without IDR
    late = dict(meta, idr=[first_slot(100), first_slot(700) + 1, first_slot(1500)])     # IDR one slot late
    assert not onepass.direct_split_ok(edges, late)
    assert onepass.direct_split_ok(edges + [int(2400 / fps * 1000)], meta)              # the stream's end


def test_proxy_pass_cuts_the_direct_stream_without_reencoding(state, tmp_path, monkeypatch):
    """With a direct stream that fits the shots: one ffmpeg, stream copy,
    no forced key frames, no decoder; one that doesn't fit: the old pass."""
    import json
    import subprocess as sp
    from pathlib import Path

    from backend.export import ffmpeg as exp
    from backend.indexing import cuts as cuts_mod
    from backend.indexing import onepass
    fps = 24000 / 1001
    video = tmp_path / "ep.mkv"
    video.write_bytes(b"v")
    cache = tmp_path / "src"
    cache.mkdir()
    base = cache / onepass._proxy_source_name(str(video))
    base.with_suffix(".direct.h264").write_bytes(b"h264")
    frames = [i * 48 for i in range(1, 9)]                  # 9 shots of 48 frames
    scenes = [(k + 1, int(a / fps * 1000), int(b / fps * 1000))
              for k, (a, b) in enumerate(zip([0] + frames, frames + [432]))]
    slots = _slots_for(fps, 432)
    idr = [next(k for k, s in enumerate(slots) if s >= c) for c in frames]
    meta = {"fps": fps, "proxy_fps": 15, "slots": len(slots), "idr": idr, "bitrate": 500_000}
    base.with_suffix(".direct.json").write_text(json.dumps(meta))
    cmds = []

    class FakePopen:
        def __init__(self, cmd, **kw):
            self.returncode = 0
            if "segment" not in cmd:
                return
            cmds.append(cmd)
            times = cmd[cmd.index("-segment_times") + 1].split(",")
            for k in range(len(times) + 1):
                Path(cmd[-1] % k).write_bytes(b"seg")

        def communicate(self):
            return b"", b""

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
    assert n == 9 and len(cmds) == 1
    c = cmds[0]
    assert c[c.index("-c:v") + 1] == "copy" and "-force_key_frames" not in c and "-hwaccel" not in c
    # a quality setting other than the one encoded: the old pass
    cmds.clear()
    done = set()
    state._proxies_batched(str(video), scenes, proxies, "high", done, source_cache=cache, chunks=3)
    assert cmds and "copy" not in cmds[0]


def test_proxy_writer_without_proxies_is_a_no_op(monkeypatch):
    """Proxies off (the default "precise" profile): the writer must still
    answer release() — it raised AttributeError and failed the NVDEC pass."""
    from backend.indexing import nvc_pass, onepass
    monkeypatch.setattr(onepass, "_proxy_cache_dir", None)
    monkeypatch.setattr(onepass, "_proxy_direct", {"bitrate": 500_000})
    w = nvc_pass._ProxyWriter("ep.mkv", 426, 240, 23.976)
    assert not w.ok and not w.direct
    w.release(_FakeScanner(23.976))
    w.release(_FakeScanner(23.976), final=True)
    w.finish_encode()
    w.close(ok=True)


def test_windows_leaves_the_prune_to_electron(fake_uv, monkeypatch):
    """Windows refused to delete a cache hard link of a DLL the sidecar had
    loaded (c10.dll, "Accès refusé"): there the sidecar only leaves a marker
    that Electron acts on before the next start."""
    from backend import uvcache
    from backend.paths import user_data_dir
    monkeypatch.setattr(uvcache.sys, "platform", "win32")
    assert uvcache.prune_in_background("engine install") is None
    assert (user_data_dir() / uvcache.PENDING).exists()
    assert not fake_uv.exists()                     # uv was not run in-process
    (user_data_dir() / ".setup_complete").write_text("cu130")
    assert uvcache.prune_after_update() is None     # already pending


def test_autotune_does_not_measure_with_the_torch_the_install_replaced(quiet_autotune, monkeypatch):
    """After the engine install the sidecar still has the old (CPU) torch
    loaded while the installed one is the CUDA build: measured there, the
    choices were CPU ones stored under the CUDA fingerprint, for good."""
    import importlib.metadata
    import sys
    import types
    autotune = quiet_autotune
    calls = []
    monkeypatch.setattr(autotune, "run", lambda **kw: calls.append(1) or {})
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(__version__="2.11.0+cpu"))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.11.0+cu130")
    from backend.paths import user_data_dir
    (user_data_dir() / ".setup_complete").write_text("cu130")
    autotune._background(False, 0.0)
    assert not calls and autotune.status()["state"] == "postponed"
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.11.0+cpu")
    autotune._background(False, 0.0)
    assert calls == [1]
