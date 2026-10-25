"""Model tiers, embedding versioning, embeddings-only re-index, license
allow-list. CPU only: SigLIP and the frame decoder are fakes."""
from __future__ import annotations

import io
import json
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.db import schema
from backend.indexing import reembed
from backend.models import tiers
from backend.models.device import DeviceInfo

DIM = 8


def _dev(backend: str, vram=None) -> DeviceInfo:
    return DeviceInfo(backend, backend, None, None, vram_gb=vram)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("AMV_VRAM_LIMIT_GB", raising=False)
    reembed.reset()
    yield
    reembed.reset()


# ── tier selection ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("vram,tier", [
    (6, "8gb"), (7.6, "8gb"), (8, "8gb"), (11.6, "12gb"), (12, "12gb"), (15.6, "16gb"),
    (16, "16gb"), (23.6, "24gb"), (24, "24gb"), (48, "24gb"), (None, "fast"),
])
def test_nvidia_auto_tier(vram, tier):
    r = tiers.resolve({"tier": "auto"}, _dev("cuda", vram))
    assert r["platform"] == "nvidia"
    assert r["auto_tier"] == tier and r["effective_tier"] == tier
    assert r["detected_gb"] == (float(vram) if vram else None)


@pytest.mark.parametrize("mem,tier", [
    (8, "mac8"), (16, "mac16"), (18, "mac16"), (24, "mac24"), (32, "mac32"), (64, "mac32"), (128, "mac32"),
])
def test_mac_auto_tier_uses_total_memory(monkeypatch, mem, tier):
    monkeypatch.setattr(tiers, "mac_total_memory_gb", lambda: float(mem))
    # vram_gb on MPS is Metal's working set (~70 %): must be ignored
    r = tiers.resolve({}, _dev("mps", mem * 0.7))
    assert r["platform"] == "mac" and r["detected_gb"] == float(mem)
    assert r["auto_tier"] == tier and r["effective_tier"] == tier and r["selected"] == "auto"


@pytest.mark.parametrize("backend", ["cpu", "dml", "xpu", "weird"])
def test_cpu_like_platforms_only_have_fast(backend):
    r = tiers.resolve({"tier": "auto"}, _dev(backend, 16))
    assert r["platform"] == "cpu" and r["detected_gb"] is None
    assert r["effective_tier"] == "fast" and list(tiers.tiers_for("cpu")) == ["fast"]
    m = r["models"]
    assert m["embedding"] == {"model": "google/siglip2-base-patch16-naflex", "patches": 128}
    assert m["tagger"] == "SmilingWolf/wd-vit-tagger-v3"
    assert (m["upscale"], m["interpolate"], m["depth"]) == (
        "realesr-animevideov3", "rife-4.25", "depth-anything-v2-small")


def test_rocm_reports_as_cuda_and_is_nvidia_platform():
    assert tiers.detect_platform(_dev("cuda", 16)) == "nvidia"
    assert tiers.detect_platform(None) == "cpu"


def test_vram_limit_env_overrides(monkeypatch):
    monkeypatch.setenv("AMV_VRAM_LIMIT_GB", "8")
    assert tiers.resolve({}, _dev("cuda", 24))["effective_tier"] == "8gb"
    monkeypatch.setattr(tiers, "mac_total_memory_gb", lambda: 64.0)
    monkeypatch.setenv("AMV_VRAM_LIMIT_GB", "16")
    r = tiers.resolve({}, _dev("mps", 48))
    assert r["detected_gb"] == 16.0 and r["effective_tier"] == "mac16"
    monkeypatch.setenv("AMV_VRAM_LIMIT_GB", "16")
    assert tiers.resolve({}, _dev("cpu"))["effective_tier"] == "fast"


def test_explicit_and_foreign_tier_ids():
    assert tiers.resolve({"tier": "12gb"}, _dev("cuda", 24))["effective_tier"] == "12gb"
    assert tiers.resolve({"tier": "max"}, _dev("cuda", 8))["effective_tier"] == "max"
    # a Mac tier on an NVIDIA machine (settings file moved over): auto
    r = tiers.resolve({"tier": "mac16"}, _dev("cuda", 24))
    assert r["selected"] == "mac16" and r["effective_tier"] == "24gb"
    assert tiers.resolve({"tier": "24gb"}, _dev("cpu"))["effective_tier"] == "fast"


def test_custom_uses_legacy_per_model_settings():
    r = tiers.resolve({"tier": "custom", "siglip_variant": "google/siglip2-so400m-patch16-naflex",
                       "siglip_max_num_patches": 256,
                       "wd_tagger_variant": "SmilingWolf/wd-eva02-large-tagger-v3"}, _dev("cuda", 24))
    assert r["effective_tier"] == "custom"
    assert r["models"]["embedding"] == {"model": "google/siglip2-so400m-patch16-naflex", "patches": 256}
    assert r["models"]["tagger"] == "SmilingWolf/wd-eva02-large-tagger-v3"
    assert r["models"]["interpolate"] == "rife-4.25"


def test_tables_are_consistent():
    from backend.finish import registry
    from backend.model_download import license_allowed

    for model_id, info in tiers.MODEL_INFO.items():
        assert license_allowed(info["license"]), model_id
        assert info["kind"] in tiers.FUNCTIONS and info["label"]
    for platform in tiers.PLATFORMS:
        table = tiers.tiers_for(platform)
        assert list(table)[0] == "fast"
        assert set(table) == set(tiers.TIER_MIN_GB[platform])
        for tid, choice in table.items():
            assert tiers.MODEL_INFO[choice["embedding"]["model"]]["kind"] == "embedding"
            assert tiers.MODEL_INFO[choice["tagger"]]["kind"] == "tagger"
            for kind in ("upscale", "interpolate", "depth"):
                assert registry.MODELS[choice[kind]].kind == kind, (platform, tid, kind)
    assert list(tiers.NVIDIA_TIERS) == ["fast", "8gb", "12gb", "16gb", "24gb", "max"]
    assert list(tiers.MAC_TIERS) == ["fast", "mac8", "mac16", "mac24", "mac32", "max"]


def test_mac_total_memory_reads_something():
    gb = tiers.mac_total_memory_gb()
    assert gb is None or gb > 0.5


def test_describe_tiers_shape():
    out = tiers.describe_tiers("nvidia")
    assert [t["id"] for t in out] == list(tiers.NVIDIA_TIERS)
    t8 = out[1]
    assert t8["min_gb"] == 8.0 and out[0]["min_gb"] is None
    assert set(t8["models"]) == {"embedding", "tagger", "upscale", "interpolate", "depth",
                                 "mask_detector", "mask_tracker"}
    emb = t8["models"]["embedding"]
    assert set(emb) >= {"id", "label", "size_mb", "license", "patches"}
    assert t8["models"]["depth"]["license"] == "apache-2.0"


def test_state_migrates_hand_picked_models_to_custom(data_dir):
    from backend.config import DEFAULT_SETTINGS

    def write(models: dict) -> None:
        # a full file: _deep_merge shares the default's nested dicts otherwise
        full = json.loads(json.dumps(DEFAULT_SETTINGS))
        full["models"].pop("tier")
        full["models"].update(models)
        (data_dir / "settings.json").write_text(json.dumps(full))

    write({"siglip_variant": "google/siglip2-so400m-patch16-naflex"})
    import backend.state as state_module
    st = state_module.get_state()
    assert st.settings["models"]["tier"] == "custom"
    state_module._state = None
    write({})
    assert state_module.get_state().settings["models"]["tier"] == "auto"


# ── fakes ───────────────────────────────────────────────────────────────────
class FakeEmbedder:
    """A deterministic 'embedding space' per checkpoint: the image's mean
    colour, permuted by a model-specific permutation."""

    max_num_patches = 128

    def __init__(self, checkpoint: str, seed: int):
        self.checkpoint = checkpoint
        self.perm = np.random.default_rng(seed).permutation(DIM)
        self.calls = 0
        self.on_call = None

        class _D:
            backend = "cpu"
            vram_gb = None
        self.device = _D()

    def _vec(self, img) -> np.ndarray:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(axis=0) / 255.0
        v = np.zeros(DIM, dtype=np.float32)
        v[:3] = arr
        v[3:] = 0.05
        v = v[self.perm]
        return v / max(float(np.linalg.norm(v)), 1e-12)

    def embed_images(self, images):
        self.calls += 1
        if self.on_call:
            self.on_call(self)
        return np.stack([self._vec(i) for i in images]).astype(np.float32)

    def embed_text(self, text):
        return self._vec(Image.new("RGB", (4, 4), (200, 30, 30)))

    def offload(self):
        pass


def _frame(path: str, ts: int) -> Image.Image:
    h = sum(path.encode()) % 200
    return Image.new("RGB", (64, 36), ((ts // 7) % 256, h, (ts // 13) % 256))


def _jpeg(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def lib(state, tmp_path, monkeypatch):
    """Primary library with 2 videos x 3 scenes embedded by 'old/a', one
    character (scene + crop + uploaded-image examples)."""
    fakes = {"old/a": FakeEmbedder("old/a", 1), "new/b": FakeEmbedder("new/b", 2),
             "third/c": FakeEmbedder("third/c", 3)}
    decoded: list[tuple[str, list[int]]] = []

    def fake_extract(path, timestamps):
        decoded.append((path, list(timestamps)))
        for ts in timestamps:
            yield ts, _frame(path, ts)

    from backend.indexing import frames
    monkeypatch.setattr(frames, "extract_frames_at", fake_extract)
    monkeypatch.setattr(state, "get_siglip", lambda checkpoint=None: fakes[checkpoint or state.db_embedding_model()])
    licensed: list[str] = []
    import backend.model_download as md
    monkeypatch.setattr(md, "ensure_license_allowed", lambda repo, token=None: licensed.append(repo))
    events: list[dict] = []
    monkeypatch.setattr(state, "publish", lambda ev: events.append(dict(ev)))

    db = state.primary_db
    schema.init_db(db)
    a = fakes["old/a"]
    paths = []
    with schema.get_conn(db) as conn:
        for v in range(2):
            fp = tmp_path / f"ep{v}.mkv"
            fp.write_bytes(b"x")
            paths.append(fp)
            vid = conn.execute(
                """INSERT INTO videos (filepath, display_name, duration_ms, fps, resolution, modified_at,
                                       model_version, status)
                   VALUES (?, ?, 6000, 24, '640x360', 0, ?, 'completed')""",
                (str(fp), f"ep{v}", "old/a|patches:128|tagger:SmilingWolf/wd-vit-tagger-v3|phases:embed,tag"),
            ).lastrowid
            for i in range(3):
                s, e = i * 2000, i * 2000 + 2000
                img = _frame(str(fp), s + (e - s) // 2)
                conn.execute(
                    """INSERT INTO scenes (video_id, scene_index, start_ms, end_ms, embedding, thumbnail)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (vid, i, s, e, a._vec(img).tobytes(), _jpeg(img.resize((32, 18)))))
        sids = [r[0] for r in conn.execute("SELECT id FROM scenes ORDER BY id")]
        cid = conn.execute("INSERT INTO characters (name) VALUES ('Gojo')").lastrowid
        from backend.searchplus import characters as chars
        chars.add_example(conn, cid, np.frombuffer(conn.execute(
            "SELECT embedding FROM scenes WHERE id = ?", (sids[0],)).fetchone()[0], dtype=np.float32),
            scene_id=sids[0])
        chars.add_example(conn, cid, a._vec(_frame(str(paths[0]), 3000)), scene_id=sids[1],
                          crop=[0.0, 0.0, 0.5, 0.5], thumbnail=_jpeg(_frame("x", 1)), source="crop")
        upload = Image.new("RGB", (40, 40), (10, 200, 10))
        chars.add_example(conn, cid, a._vec(upload), thumbnail=_jpeg(upload), source="image")
    state.invalidate_search_cache()

    class Lib:
        pass
    L = Lib()
    L.state, L.db, L.fakes, L.events, L.decoded, L.paths, L.sids, L.cid = (
        state, db, fakes, events, decoded, paths, sids, cid)
    L.licensed = licensed
    return L


def _active(db) -> dict[int, np.ndarray]:
    with schema.get_conn(db) as conn:
        return {sid: (np.frombuffer(b, dtype=np.float32) if b is not None else None)
                for sid, b in conn.execute("SELECT id, embedding FROM scenes ORDER BY id")}


def _setting(db, key):
    with schema.get_conn(db) as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _expected(lib, model: str) -> dict[int, np.ndarray]:
    f = lib.fakes[model]
    with schema.get_conn(lib.db) as conn:
        rows = conn.execute("""SELECT s.id, v.filepath, s.start_ms, s.end_ms FROM scenes s
                                 JOIN videos v ON v.id = s.video_id ORDER BY s.id""").fetchall()
    return {sid: f._vec(_frame(fp, s + (e - s) // 2)) for sid, fp, s, e in rows}


# ── embedding-model resolution ──────────────────────────────────────────────
def test_db_model_empty_legacy_recorded(state, tmp_path):
    from backend.config import DEFAULT_SIGLIP

    empty = str(tmp_path / "empty.db")
    schema.init_db(empty)
    with schema.get_conn(empty) as conn:
        assert reembed.bound_model(conn) is None
    assert reembed.db_embedding_model(empty, default="tier/model") == "tier/model"
    assert reembed.bind_model(empty, "tier/model", 64) == "tier/model"
    assert _setting(empty, "embedding_model") == "tier/model"
    assert _setting(empty, "embedding_patches") == "64"

    legacy = str(tmp_path / "legacy.db")
    schema.init_db(legacy)
    with schema.get_conn(legacy) as conn:
        for i, mv in enumerate(["google/siglip2-so400m-patch16-naflex|patches:128|tagger:t|phases:embed,tag",
                                "google/siglip2-so400m-patch16-naflex|patches:256|tagger:t|phases:embed",
                                None]):
            vid = conn.execute("INSERT INTO videos (filepath, display_name, modified_at, model_version, status) "
                               "VALUES (?, 'v', 0, ?, 'completed')", (f"/v{i}", mv)).lastrowid
            conn.execute("INSERT INTO scenes (video_id, scene_index, start_ms, end_ms, embedding) "
                         "VALUES (?, 0, 0, 1000, ?)", (vid, np.ones(4, np.float32).tobytes()))
    assert reembed.db_embedding_model(legacy, default="tier/model") == "google/siglip2-so400m-patch16-naflex"
    assert reembed.bind_model(legacy, "tier/model") == "google/siglip2-so400m-patch16-naflex"
    assert _setting(legacy, "embedding_model") == "google/siglip2-so400m-patch16-naflex"

    nover = str(tmp_path / "nover.db")
    schema.init_db(nover)
    with schema.get_conn(nover) as conn:
        vid = conn.execute("INSERT INTO videos (filepath, display_name, modified_at, status) "
                           "VALUES ('/x', 'x', 0, 'completed')").lastrowid
        conn.execute("INSERT INTO scenes (video_id, scene_index, start_ms, end_ms, embedding) "
                     "VALUES (?, 0, 0, 1000, ?)", (vid, np.ones(4, np.float32).tobytes()))
    assert reembed.db_embedding_model(nover, default="tier/model") == DEFAULT_SIGLIP

    with schema.get_conn(legacy) as conn:
        reembed.record_model(conn, "recorded/model")
    assert reembed.db_embedding_model(legacy, default="tier/model") == "recorded/model"


def test_swap_model_version():
    mv = "old/a|patches:128|tagger:SmilingWolf/wd-vit-tagger-v3|phases:embed,tag"
    assert reembed.swap_model_version(mv, "new/b", 256) == \
        "new/b|patches:256|tagger:SmilingWolf/wd-vit-tagger-v3|phases:embed,tag"
    assert reembed.swap_model_version(None, "new/b", 1) is None
    assert reembed.swap_model_version("x", "new/b", 1) == "x"


def test_get_siglip_follows_the_database_model(state, monkeypatch):
    import backend.models as models_pkg

    made = []

    class Recorder:
        def __init__(self, device, checkpoint, max_num_patches=128, **kw):
            self.checkpoint, self.max_num_patches = checkpoint, max_num_patches
            self.offloaded = False
            made.append(self)

        def offload(self):
            self.offloaded = True

    monkeypatch.setattr(models_pkg, "SigLIP2Model", Recorder)
    # empty library: the tier's model
    assert state.get_siglip().checkpoint == "google/siglip2-base-patch16-naflex"
    with schema.get_conn(state.primary_db) as conn:
        vid = conn.execute("INSERT INTO videos (filepath, display_name, modified_at, status) "
                           "VALUES ('/x', 'x', 0, 'completed')").lastrowid
        conn.execute("INSERT INTO scenes (video_id, scene_index, start_ms, end_ms, embedding) "
                     "VALUES (?, 0, 0, 1000, ?)", (vid, np.ones(4, np.float32).tobytes()))
        reembed.record_model(conn, "google/siglip2-so400m-patch16-naflex")
    state.invalidate_search_cache()
    main = state.get_siglip()
    assert main.checkpoint == "google/siglip2-so400m-patch16-naflex" and made[0].offloaded
    # explicit target (re-embed): a second slot, the search model stays
    aux = state.get_siglip(checkpoint="google/siglip2-base-patch16-naflex")
    assert aux is not main and not main.offloaded
    assert state.get_siglip() is main
    state.promote_siglip("google/siglip2-base-patch16-naflex")
    assert state._siglip is aux and main.offloaded and state._siglip_aux is None


# ── re-embed job ────────────────────────────────────────────────────────────
def test_reembed_end_to_end_and_instant_switch_back(lib):
    st, db = lib.state, lib.db
    old_vectors = _active(db)
    job = reembed.start(st, "new/b", background=False)
    assert job.ok, job.message
    ev = [e for e in lib.events if e["type"] == "reembed"]
    assert ev[0]["percent"] == 0 and ev[-1]["done"] and ev[-1]["ok"] and ev[-1]["percent"] == 100
    assert all(set(e) >= {"percent", "message", "done", "ok", "videos_missing"} for e in ev)
    pcts = [e["percent"] for e in ev]
    assert pcts == sorted(pcts) and len(ev) >= 3
    assert lib.licensed == ["new/b"]
    # median frames of every scene, decoded once per video (+ the crop
    # example's median frame)
    passes = [(p, ts) for p, ts in lib.decoded if len(ts) > 1]
    assert sorted(p for p, _ts in passes) == sorted(str(p) for p in lib.paths)
    assert all(ts == [1000, 3000, 5000] for _p, ts in passes)
    assert (str(lib.paths[0]), [3000]) in lib.decoded

    assert _setting(db, "embedding_model") == "new/b"
    exp_b = _expected(lib, "new/b")
    for sid, v in _active(db).items():
        np.testing.assert_allclose(v, exp_b[sid], atol=1e-6)
    with schema.get_conn(db) as conn:
        archived = {(sid, m): np.frombuffer(b, dtype=np.float32)
                    for sid, m, b in conn.execute("SELECT scene_id, model, embedding FROM scene_embeddings")}
        assert {m for _s, m in archived} == {"old/a", "new/b"}
        for sid, v in old_vectors.items():
            np.testing.assert_array_equal(archived[(sid, "old/a")], v)
        mvs = {r[0] for r in conn.execute("SELECT model_version FROM videos")}
        assert mvs == {"new/b|patches:128|tagger:SmilingWolf/wd-vit-tagger-v3|phases:embed,tag"}
        ex = conn.execute("SELECT id, scene_id, crop, embedding FROM character_examples ORDER BY id").fetchall()
        assert conn.execute("SELECT COUNT(*) FROM character_example_embeddings").fetchone()[0] == 6
        sig, proto = conn.execute("SELECT scored_signature, prototype FROM characters").fetchone()
    # scene example = its scene's new vector; crop/image re-embedded with new/b
    np.testing.assert_allclose(np.frombuffer(ex[0][3], dtype=np.float32), exp_b[lib.sids[0]], atol=1e-6)
    crop_img = _frame(str(lib.paths[0]), 3000).crop((0, 0, 32, 18))
    np.testing.assert_allclose(np.frombuffer(ex[1][3], dtype=np.float32), lib.fakes["new/b"]._vec(crop_img),
                               atol=1e-6)
    assert sig is not None and proto is not None

    # search now embeds queries with new/b
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from backend.api import search as search_api
    app = FastAPI()
    app.include_router(search_api.router)
    used = []
    real = st.get_siglip
    st.get_siglip = lambda checkpoint=None: used.append(st.db_embedding_model()) or real(checkpoint)
    r = TestClient(app).post("/api/search", json={"query": "red", "threshold": -1})
    assert r.status_code == 200 and r.json()["total"] == 6 and used == ["new/b"]
    st.get_siglip = real

    # switching back: vectors already archived -> no decode, no model
    lib.decoded.clear()
    calls = {m: f.calls for m, f in lib.fakes.items()}
    job = reembed.start(st, "old/a", background=False)
    assert job.ok and lib.decoded == []
    assert {m: f.calls for m, f in lib.fakes.items()} == calls
    assert _setting(db, "embedding_model") == "old/a"
    for sid, v in _active(db).items():
        np.testing.assert_array_equal(v, old_vectors[sid])

    # already on target: nothing to do
    assert reembed.start(st, "old/a", background=False).ok


def test_reembed_refuses_missing_files_then_force(lib):
    st, db = lib.state, lib.db
    before = _active(db)
    lib.paths[1].unlink()
    job = reembed.start(st, "third/c", background=False)
    assert not job.ok and job.videos_missing == 1 and job.error == "videos_missing"
    assert "relink" in job.message and lib.decoded == []
    last = lib.events[-1]
    assert last["type"] == "reembed" and last["done"] and not last["ok"] and last["videos_missing"] == 1
    assert _setting(db, "embedding_model") is None      # still the inferred old/a
    assert st.db_embedding_model() == "old/a"
    assert {k: v.tobytes() for k, v in _active(db).items()} == {k: v.tobytes() for k, v in before.items()}

    job = reembed.start(st, "third/c", force=True, background=False)
    assert job.ok and job.videos_missing == 1
    act = _active(db)
    with schema.get_conn(db) as conn:
        gone = {r[0] for r in conn.execute(
            "SELECT s.id FROM scenes s JOIN videos v ON v.id = s.video_id WHERE v.filepath = ?",
            (str(lib.paths[1]),))}
    assert all(act[s] is None for s in gone) and all(act[s] is not None for s in act if s not in gone)
    assert st.db_embedding_model() == "third/c"
    # back to old/a: the dropped scenes come back from the archive
    job = reembed.start(st, "old/a", background=False)
    assert job.ok
    assert {k: v.tobytes() for k, v in _active(db).items()} == {k: v.tobytes() for k, v in before.items()}


def test_reembed_cancel_keeps_old_model_and_partial_vectors(lib):
    st, db = lib.state, lib.db
    st.settings["indexing"]["batch_size"] = 1
    lib.fakes["new/b"].on_call = lambda f: reembed.cancel() if f.calls == 2 else None
    job = reembed.start(st, "new/b", background=False)
    assert not job.ok and job.error == "cancelled" and not job.switched
    assert st.db_embedding_model() == "old/a" and _setting(db, "embedding_model") is None
    with schema.get_conn(db) as conn:
        n = conn.execute("SELECT COUNT(*) FROM scene_embeddings WHERE model = 'new/b'").fetchone()[0]
    assert 1 <= n < 6
    # resume: only the missing scenes are decoded
    lib.fakes["new/b"].on_call = None
    lib.decoded.clear()
    job = reembed.start(st, "new/b", background=False)
    assert job.ok and sum(len(ts) for _p, ts in lib.decoded) == (6 - n) + 1   # + the crop example


def test_unbound_library_just_records_target(state, monkeypatch):
    events = []
    monkeypatch.setattr(state, "publish", events.append)
    job = reembed.start(state, "new/b", background=False)
    assert job.ok and job.switched and _setting(state.primary_db, "embedding_model") == "new/b"
    # still unbound (no vectors): the tier's model is used for queries
    assert state.db_embedding_model() == "google/siglip2-base-patch16-naflex"


def test_reembed_api_and_indexing_exclusion(lib, client):
    st = lib.state
    r = client.get("/api/models/tiers")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"platform", "detected_gb", "auto_tier", "selected", "effective_tier", "tiers", "embedding"}
    assert body["platform"] == "cpu" and body["effective_tier"] == "fast" and body["selected"] == "auto"
    assert [t["id"] for t in body["tiers"]] == ["fast"]
    emb = body["embedding"]
    assert emb["db_model"] == "old/a" and emb["target_model"] == "google/siglip2-base-patch16-naflex"
    assert emb["reindex_needed"] is True and emb["archived_models"] == [] and emb["job"] is None

    assert client.get("/api/models/reembed").json() == {"job": None}
    assert client.post("/api/models/reembed/cancel").status_code == 409
    assert client.post("/api/models/reembed", json={"model": "rife-4.25"}).status_code == 400

    # indexing running -> 409
    stop = threading.Event()
    st._index_thread = threading.Thread(target=stop.wait, daemon=True)
    st._index_thread.start()
    try:
        assert client.post("/api/models/reembed", json={"model": "new/b"}).status_code == 409
    finally:
        stop.set()
        st._index_thread.join()

    # re-embed running -> indexing refused, second job 409
    gate = threading.Event()
    lib.fakes["new/b"].on_call = lambda f: gate.wait(10)
    r = client.post("/api/models/reembed", json={"model": "new/b"})
    assert r.status_code == 200 and r.json()["job"]["running"]
    try:
        assert st.start_indexing() is False
        assert client.post("/api/models/reembed", json={"model": "new/b"}).status_code == 409
        assert client.get("/api/models/tiers").json()["embedding"]["job"]["running"] is True
    finally:
        gate.set()
    reembed.current().thread.join(10)
    job = client.get("/api/models/reembed").json()["job"]
    assert job["done"] and job["ok"] and job["target_model"] == "new/b"
    emb = client.get("/api/models/tiers").json()["embedding"]
    assert emb["db_model"] == "new/b" and emb["archived_models"] == ["old/a"]


def test_merge_refuses_mixed_embedding_spaces(lib, tmp_path):
    other = str(tmp_path / "other.db")
    schema.init_db(other)
    with schema.get_conn(other) as conn:
        vid = conn.execute("INSERT INTO videos (filepath, display_name, modified_at, status) "
                           "VALUES ('/o', 'o', 0, 'completed')").lastrowid
        conn.execute("INSERT INTO scenes (video_id, scene_index, start_ms, end_ms, embedding) "
                     "VALUES (?, 0, 0, 1000, ?)", (vid, np.ones(4, np.float32).tobytes()))
        reembed.record_model(conn, "new/b")
    assert "re-embed" in reembed.merge_conflict(lib.db, [other])
    empty = str(tmp_path / "empty.db")
    schema.init_db(empty)
    assert reembed.merge_conflict(lib.db, [empty]) is None


# ── finishing: tier defaults ────────────────────────────────────────────────
def test_finish_models_default_to_the_tier(state):
    from fastapi import HTTPException
    from backend.api import finish as finish_api
    from backend.finish import runner

    assert runner.model_for("interpolate", {}) == "rife-4.25"
    assert runner.model_for("depth", {}) == "depth-anything-v2-small"
    assert runner.model_for("upscale", {"model": "realesrgan-x4plus-anime-6b"}) == "realesrgan-x4plus-anime-6b"
    assert runner.depth_input_size({}) is None and runner.depth_input_size({"depth_input_size": 700}) == 700
    assert finish_api._with_model("interpolate", {"mode": "fps"}) == {"mode": "fps", "model": "rife-4.25"}
    assert finish_api._required_model("depth", {}) == "depth-anything-v2-small"
    for kind, bad in (("interpolate", "realesr-animevideov3"), ("depth", "nope"), ("upscale", "rife-4.25")):
        with pytest.raises(HTTPException) as exc:
            finish_api._required_model(kind, {"model": bad})
        assert exc.value.status_code == 400
    with pytest.raises(HTTPException):
        finish_api._validate_params("depth", {"depth_input_size": 10})
    finish_api._validate_params("depth", {"depth_input_size": 518})


def test_runner_passes_model_and_input_size(monkeypatch):
    from backend.finish import depth, interpolate, runner

    seen = {}

    class FakeRife:
        def __init__(self, device, scale=1.0, **kw):
            seen["rife"] = kw

    class FakeDepth:
        def __init__(self, device, **kw):
            seen["depth"] = kw

    class _D:
        backend = "cpu"
    monkeypatch.setattr(interpolate, "RifeInterpolator", FakeRife)
    monkeypatch.setattr(depth, "DepthEstimator", FakeDepth)
    monkeypatch.setattr(runner, "_free_other_models", lambda: None)
    runner.release_models()
    runner._rife(_D(), "rife-4.25", 1.0)
    assert seen["rife"] == {}
    runner._rife(_D(), "rife-future", 1.0)
    assert seen["rife"] == {"model_id": "rife-future"}
    runner._depth(_D(), "depth-anything-v2-small", 700)
    assert seen["depth"] == {"input_size": 700}
    runner.release_models()


# ── license allow-list ──────────────────────────────────────────────────────
def test_license_allow_list(monkeypatch, tmp_path):
    import backend.model_download as md

    for ok in ("apache-2.0", "MIT", "bsd-2-clause", "BSD-3-Clause", "cc-by-4.0", "cc0-1.0", ["mit"]):
        assert md.license_allowed(ok), ok
    for bad in ("openrail", "openrail++", "creativeml-openrail-m", "cc-by-nc-4.0", "cc-by-nc-sa-4.0",
                "gpl-3.0", "agpl-3.0", "other", "", None, ["mit", "gpl-3.0"]):
        assert not md.license_allowed(bad), bad

    monkeypatch.setattr(md, "hub_license", lambda repo, token=None: "cc-by-nc-4.0")
    with pytest.raises(md.ModelDownloadError) as exc:
        md.check_license("someone/model")
    assert "allow-list" in str(exc.value) and "Apache-2.0" in exc.value.hint
    # the model card wins over the catalogue
    with pytest.raises(md.ModelDownloadError):
        md.check_license("google/siglip2-base-patch16-naflex")
    # offline: the catalogue's license, unknown repos refused
    monkeypatch.setattr(md, "hub_license", lambda repo, token=None: None)
    assert md.check_license("google/siglip2-base-patch16-naflex") == "apache-2.0"
    with pytest.raises(md.ModelDownloadError) as exc:
        md.check_license("someone/unknown")
    assert "could not be determined" in str(exc.value)

    # cached repos: no network, no refusal
    def boom(*a, **k):
        raise AssertionError("network call for a cached model")
    monkeypatch.setattr(md, "hub_license", boom)
    from huggingface_hub import constants
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    snap = tmp_path / "models--someone--unknown" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    assert md.is_repo_cached("someone/unknown")
    md.ensure_license_allowed("someone/unknown")

    # first download of a refused repo never reaches the Hub client
    monkeypatch.setattr(md, "hub_license", lambda repo, token=None: "openrail")
    monkeypatch.setattr(md, "snapshot_download", boom)
    monkeypatch.setattr(md, "hf_hub_download", boom)
    with pytest.raises(md.ModelDownloadError):
        md.download_snapshot("someone/other", max_attempts=1)
    with pytest.raises(md.ModelDownloadError):
        md.download_model_files("someone/other", ["a.onnx"], max_attempts=1)


def test_finish_registry_license_check(monkeypatch, client):
    from backend.finish import registry
    import backend.model_download as md

    for spec in registry.MODELS.values():
        md.check_finish_license(spec)
    nc = registry.ModelSpec(id="nc-model", kind="depth", label="nc", license="CC-BY-NC-4.0", source="x",
                            files=(registry.ModelFile("w.bin", "http://x/w.bin", 3),))
    monkeypatch.setitem(registry.MODELS, nc.id, nc)
    started = []
    monkeypatch.setattr(registry, "start_download", lambda *a, **k: started.append(a) or True)
    r = client.post("/api/finish/models/nc-model/download")
    assert r.status_code == 403 and "allow-list" in r.json()["detail"] and not started
    r = client.post("/api/finish/models/rife-4.25/download")
    assert r.status_code == 200 and started


# ── the measured tables (chantier 4 report) ─────────────────────────────────
def test_measured_choices_per_tier():
    from backend.models import tiers

    so400m = "google/siglip2-so400m-patch16-naflex"
    base = "google/siglip2-base-patch16-naflex"
    for tid in ("8gb", "12gb", "16gb", "24gb"):
        t = tiers.NVIDIA_TIERS[tid]
        assert t["embedding"] == {"model": so400m, "patches": 128}
        assert t["tagger"] == "SmilingWolf/wd-vit-tagger-v3"
        assert t["upscale"] == "hfa2k-avc-compact" and t["interpolate"] == "rife-4.26"
    for tid in ("mac24", "mac32"):
        assert tiers.MAC_TIERS[tid]["embedding"]["model"] == so400m
    for tid in ("mac8", "mac16"):
        assert tiers.MAC_TIERS[tid]["embedding"]["model"] == base
    for table in (tiers.NVIDIA_TIERS, tiers.MAC_TIERS):
        assert table["fast"]["embedding"]["model"] == base
        assert table["fast"]["depth_input_size"] == 364
        assert table["max"]["embedding"]["patches"] == 256
        assert table["max"]["interpolate"] == "rife-4.26-heavy"
    assert tiers.NVIDIA_TIERS["24gb"]["depth"] == "video-depth-anything-small"
    assert tiers.NVIDIA_TIERS["8gb"]["depth_input_size"] == 364
    # CPU keeps the previous defaults: no regression there
    cpu = tiers.CPU_TIERS["fast"]
    assert cpu["embedding"]["model"] == base and cpu["upscale"] == "realesr-animevideov3"
    assert cpu["interpolate"] == "rife-4.25" and "depth_input_size" not in cpu


def test_every_tier_model_is_registered_and_licensed():
    from backend.finish import registry
    from backend.models import tiers

    for table in (tiers.NVIDIA_TIERS, tiers.MAC_TIERS, tiers.CPU_TIERS):
        for t in table.values():
            for kind in ("upscale", "interpolate", "depth"):
                assert registry.MODELS[t[kind]].kind == kind
                assert t[kind] in tiers.MODEL_INFO
            assert t["embedding"]["model"] in tiers.MODEL_INFO
            assert t["tagger"] in tiers.MODEL_INFO


def test_rife_variants_build_and_native_x2_upscaler(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from backend.finish.archs.rife import VARIANTS, IFNet

    for name in ("4.25", "4.26", "4.25.lite", "4.26.heavy"):
        net = IFNet(variant=name).eval()
        m = VARIANTS[name]["modulo"]
        a = torch.rand(1, 3, m, m)
        with torch.inference_mode():
            assert net(a, a, 0.5).shape == a.shape

    from backend.finish.archs.realesrgan import SRVGGNetCompact
    from backend.finish import registry
    from backend.finish.upscale import Upscaler
    from backend.models.device import DeviceInfo

    monkeypatch.setenv("AMV_MODELS_DIR", str(tmp_path))
    d = registry.model_dir("hfa2k-avc-compact")
    d.mkdir(parents=True, exist_ok=True)
    net = SRVGGNetCompact(num_feat=64, num_conv=16, upscale=2)
    torch.save({"params": net.state_dict()}, d / "2xHFA2kAVCCompact.pth")
    up = Upscaler("hfa2k-avc-compact", DeviceInfo("cpu", "cpu", torch.device("cpu"), torch.float32))
    import numpy as np
    frame = (np.random.default_rng(0).random((20, 24, 3)) * 255).astype(np.uint8)
    assert up.upscale(frame, scale=2).shape == (40, 48, 3)
    assert up.upscale(frame, scale=4).shape == (80, 96, 3)


@pytest.mark.parametrize("n", [1, 5, 31, 32, 33, 53, 54, 55, 76, 100])
def test_video_depth_stream_keeps_every_frame_in_order(n):
    """The 32-frame windows / key frames / cross-fade bookkeeping of the
    Video Depth Anything stream: one map per frame, in order, and with an
    identity network every map is its own frame's (alignment = identity)."""
    torch = pytest.importorskip("torch")
    from backend.finish.vda import VideoDepthEstimator, model_size

    est = VideoDepthEstimator.__new__(VideoDepthEstimator)
    est._prep = lambda f: torch.full((1, 2, 2), float(f[0, 0, 0]))
    est._window = lambda xs: torch.stack(xs)[:, 0]
    frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(n)]
    out = list(est.predict_stream(iter(frames)))
    assert [int(f[0, 0, 0]) for f, _ in out] == list(range(n))
    assert all(np.allclose(d, i) for i, (_, d) in enumerate(out))
    assert model_size(1080, 1920) == (518, 924)
