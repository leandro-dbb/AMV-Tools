"""Fixes for what the Windows 11 + RTX 3090 test left open
(docs/reports/windows-rtx3090.md, « Corrections »)."""
from __future__ import annotations

import pytest


# ── 7. exported clip names ──────────────────────────────────────────────────
@pytest.mark.parametrize("display, anime, episode", [
    ("[SubsPlease] Dragon Ball Super - Beerus - 01 (1080p) [B3FEF9B7]", "Dragon Ball Super - Beerus", "01"),
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
    file name twice: ``…B3FEF9B7_SubsPlease_…B3FEF9B7_scene_468.mp4``."""
    from backend.api import export as api_export
    seen = {}
    monkeypatch.setattr(api_export, "export_scene", lambda src, s, e, target, **kw: seen.setdefault("t", target))
    scene = {"id": 468, "display_name": "[SubsPlease] Dragon Ball Super - Beerus - 01 (1080p) [B3FEF9B7]",
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
    c.post("/api/search", json={"query": "beerus"})
    assert seen == ["begin", "end"]
