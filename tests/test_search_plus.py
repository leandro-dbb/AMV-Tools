"""Search+ (v0.4 chantier C): motion analysis, metrics, v9 migration,
similar shots, characters, combined filters.

Run: uv run pytest tests/test_search_plus.py
Everything is CPU-only; SigLIP / wd-tagger are mocked, video clips are
generated on the fly with ffmpeg (nothing is committed).
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Isolate settings/DBs from the developer's real data dir BEFORE importing
# anything that resolves user paths.
_DATA_DIR = tempfile.mkdtemp(prefix="amv_searchplus_")
os.environ["AMV_DATA_DIR"] = _DATA_DIR
_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend.db import schema  # noqa: E402
from backend.indexing import cuts  # noqa: E402
from backend.searchplus import analysis, characters as chars, filters as sf, store  # noqa: E402

DIM = 32


# ── helpers / fixtures ──────────────────────────────────────────────────────
def _ffmpeg(*args: str) -> None:
    cmd = [cuts._ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", *args]
    subprocess.run(cmd, check=True, capture_output=True)


_ENC = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]


@pytest.fixture(scope="session")
def clips(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("clips")
    src = "testsrc2=size=1280x720:rate=24"
    still = d / "still.png"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=1", "-frames:v", "1", str(still))
    out = {}

    def crop_clip(name: str, x: str, y: str) -> None:
        p = d / f"{name}.mp4"
        _ffmpeg("-f", "lavfi", "-i", src, "-t", "3", "-vf", f"crop=640:360:x='{x}':y='{y}'", *_ENC, str(p))
        out[name] = p

    crop_clip("pan_right", "min(t*160,640)", "180")      # window slides right → camera pans right
    crop_clip("pan_left", "640-min(t*160,640)", "180")
    crop_clip("tilt_down", "320", "min(t*110,360)")
    crop_clip("tilt_up", "320", "360-min(t*110,360)")

    zin = d / "zoom_in.mp4"
    _ffmpeg("-i", str(still), "-vf",
            "zoompan=z='1+0.008*on':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=72:s=640x360:fps=24",
            *_ENC, str(zin))
    out["zoom_in"] = zin
    zout = d / "zoom_out.mp4"
    _ffmpeg("-i", str(zin), "-vf", "reverse", *_ENC, str(zout))
    out["zoom_out"] = zout

    static = d / "static.mp4"
    _ffmpeg("-loop", "1", "-i", str(still), "-t", "3", "-r", "24", "-vf", "scale=640:360", *_ENC, str(static))
    out["static"] = static
    shake = d / "shake.mp4"
    _ffmpeg("-loop", "1", "-i", str(still), "-t", "3", "-r", "24", "-vf",
            "crop=640:360:x='320+40*sin(9*t)*cos(23*t)':y='180+30*sin(17*t)'", *_ENC, str(shake))
    out["shake"] = shake

    # Three shots separated by hard cuts (negate → very different histogram).
    combo = d / "combo.mp4"
    _ffmpeg("-i", str(static), "-i", str(out["pan_right"]), "-i", str(zin), "-filter_complex",
            "[0:v]negate,setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];"
            "[2:v]negate,setpts=PTS-STARTPTS[c];[a][b][c]concat=n=3:v=1:a=0,fps=24[v]",
            "-map", "[v]", *_ENC, str(combo))
    out["combo"] = combo
    return out


class FakeSigLIP:
    """Deterministic stand-in: text/images map to fixed unit vectors."""

    checkpoint = "fake/siglip"
    max_num_patches = 128

    def __init__(self, text_vecs: dict[str, np.ndarray] | None = None):
        self.text_vecs = text_vecs or {}
        self.calls = 0

    def embed_text(self, text: str) -> np.ndarray:
        v = self.text_vecs.get(text.lower())
        if v is None:
            v = _unit(abs(hash(text)) % 1000)
        return v

    def embed_images(self, images) -> np.ndarray:
        self.calls += len(images)
        out = []
        for img in images:
            arr = np.asarray(img.convert("RGB").resize((8, 4)), dtype=np.float32).ravel()[:DIM]
            out.append(_norm(arr + 1.0))
        return np.vstack(out)

    def offload(self):
        pass


class FakeTagger:
    repo_id = "fake/tagger"

    def tag_images(self, images):
        return [[("1girl", "general", 0.9)] for _ in images]

    def embed_images(self, images):
        return [np.ones(4, dtype=np.float32) for _ in images]

    def offload(self):
        pass


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _unit(seed: int) -> np.ndarray:
    return _norm(np.random.default_rng(seed).normal(size=DIM))


@pytest.fixture()
def sp_state(tmp_path, monkeypatch):
    """Real AppState pointed at a fresh DB, with SigLIP mocked."""
    import backend.state as st
    st._state = None
    s = st.get_state()
    db = str(tmp_path / "lib.db")
    schema.init_db(db)
    s._settings["databases"] = {"active": [db], "primary": db}
    s.invalidate_search_cache()
    fake = FakeSigLIP()
    monkeypatch.setattr(s, "get_siglip", lambda: fake)
    s.fake_siglip = fake
    yield s
    st._state = None


@pytest.fixture()
def sp_client(sp_state):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api import characters as characters_api, search, searchplus
    app = FastAPI()
    for r in (search.router, searchplus.router, characters_api.router):
        app.include_router(r)
    return TestClient(app)  # no `with`: startup hooks (auto analysis) stay off


def add_video(conn, name: str, group: str | None = None, status: str = "completed",
              filepath: str | None = None) -> int:
    cur = conn.execute(
        """INSERT INTO videos (filepath, display_name, duration_ms, fps, resolution, modified_at,
                               model_version, status, group_name)
           VALUES (?, ?, 60000, 24, '640x360', 0, 'x', ?, ?)""",
        (filepath or f"/nowhere/{name}.mkv", name, status, group),
    )
    return int(cur.lastrowid)


def add_scene(conn, video_id: int, idx: int, emb: np.ndarray | None, start_ms: int | None = None,
              dur_ms: int = 2000, metrics: dict | None = None, thumb: bytes | None = None) -> int:
    s = start_ms if start_ms is not None else idx * dur_ms
    cur = conn.execute(
        """INSERT INTO scenes (video_id, scene_index, start_ms, end_ms, embedding, thumbnail)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (video_id, idx, s, s + dur_ms, None if emb is None else _norm(emb).tobytes(), thumb),
    )
    sid = int(cur.lastrowid)
    if metrics is not None:
        m = analysis.SceneMetrics(
            motion_type=metrics.get("motion_type", "static"),
            motion_magnitude=metrics.get("motion_magnitude", 0.0),
            motion_dx=0.0, motion_dy=0.0, motion_zoom=0.0,
            activity=metrics.get("activity", 0.0),
            intensity=metrics.get("intensity", 0.0),
            cut_rate=0.0, flash_count=0, brightness=metrics.get("brightness", 0.5),
            dominant_color=metrics.get("dominant_color"),
            accent_color=metrics.get("accent_color", metrics.get("dominant_color")),
            palette=[],
        )
        store.upsert_metrics(conn, [(sid, s, s + dur_ms, m)])
    return sid


# ── 1. motion analysis on synthetic clips ───────────────────────────────────
@pytest.mark.parametrize("name", [
    "static", "pan_left", "pan_right", "tilt_up", "tilt_down", "zoom_in", "zoom_out", "shake",
])
def test_motion_classification(clips, name):
    acc = analysis.collect_frame_stats(str(clips[name]))
    dur = cuts.probe_video(str(clips[name]))["duration_ms"]
    m = analysis.aggregate_scene(acc, 0, dur)
    assert m.motion_type == name, m
    if name == "static":
        assert m.motion_magnitude < 0.05
    else:
        assert m.motion_magnitude > 0.08


def test_pan_speed_estimate(clips):
    """160 px/s on a 640 px window = 0.25 frame widths per second."""
    acc = analysis.collect_frame_stats(str(clips["pan_right"]))
    m = analysis.aggregate_scene(acc, 0, 3000)
    assert m.motion_dx == pytest.approx(-0.25, abs=0.05)
    assert abs(m.motion_dy) < 0.05


def test_intensity_ordering(clips):
    def intensity(name):
        acc = analysis.collect_frame_stats(str(clips[name]))
        return analysis.aggregate_scene(acc, 0, 3000).intensity
    static, pan, shake = intensity("static"), intensity("pan_right"), intensity("shake")
    assert 0.0 <= static < 0.05
    assert static < pan < shake <= 1.0


def test_flashes_raise_intensity():
    acc = analysis.FrameStatsAccumulator()
    rng = np.random.default_rng(0)
    base = (rng.random((90, 160)) * 120).astype(np.uint8)
    for i in range(20):
        frame = base if i % 3 else np.clip(base.astype(int) + 110, 0, 255).astype(np.uint8)
        acc.add(i * 167, frame)
    m = analysis.aggregate_scene(acc, 0, 20 * 167)
    assert m.flash_count >= 5
    calm = analysis.FrameStatsAccumulator()
    for i in range(20):
        calm.add(i * 167, base)
    assert m.intensity > analysis.aggregate_scene(calm, 0, 20 * 167).intensity + 0.1


def test_palette_and_color_buckets():
    img = np.zeros((90, 160, 3), dtype=np.uint8)
    img[:, :100] = (220, 30, 30)    # red, dominant
    img[:, 100:] = (30, 60, 230)    # blue
    pal = analysis.palette(img)
    assert pal[0][2] == "red" and pal[0][1] > 0.55
    assert {p[2] for p in pal} >= {"red", "blue"}
    dom, acc = analysis.dominant_and_accent(pal)
    assert dom == "red" and acc in ("red", "blue")
    assert analysis.color_bucket(250, 250, 250) == "white"
    assert analysis.color_bucket(5, 5, 5) == "black"
    assert analysis.color_bucket(240, 140, 20) == "orange"


def test_accumulator_resets_when_decoder_restarts():
    acc = analysis.FrameStatsAccumulator()
    f = np.zeros((90, 160), dtype=np.uint8)
    for t in (0, 100, 200):
        acc.add(t, f)
    acc.add(0, f)  # hw path failed → PyAV restarts from 0
    assert acc.times == [0] and acc.p_t0 == []


# ── 2. indexing integration ────────────────────────────────────────────────
def test_index_video_stores_metrics(clips, tmp_path):
    from backend.indexing import pipeline
    db = str(tmp_path / "idx.db")
    pipeline.index_video(db, str(clips["combo"]), FakeSigLIP(), FakeTagger(),
                         phases=("tag", "embed"), scene_threshold=0.40, auto_skip_indexed=False)
    with schema.get_conn(db) as conn:
        rows = conn.execute(
            """SELECT s.start_ms, s.end_ms, m.motion_type, m.intensity, m.dominant_color, m.version
                 FROM scenes s JOIN scene_metrics m ON m.scene_id = s.id ORDER BY s.start_ms"""
        ).fetchall()
        n_scenes = conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]
    assert n_scenes == 3 and len(rows) == 3, rows
    assert [r[2] for r in rows] == ["static", "pan_right", "zoom_in"]
    assert all(r[5] == analysis.ANALYSIS_VERSION and r[4] for r in rows)


def test_background_analyzer_fills_old_library(clips, sp_state):
    """A pre-v0.4 library (scenes, no metrics) gets analysed without reindexing."""
    from backend.searchplus.jobs import AnalysisJob
    db = sp_state.primary_db
    with schema.get_conn(db) as conn:
        vid = add_video(conn, "pan", filepath=str(clips["pan_right"]))
        sid = add_scene(conn, vid, 0, _unit(1), start_ms=0, dur_ms=3000)
        missing = add_video(conn, "gone", filepath="/nowhere/gone.mkv")
        add_scene(conn, missing, 0, _unit(2))
        assert store.coverage(conn)["pending"] == 2
    job = AnalysisJob()
    events = []
    sp_state.publish = events.append
    job._run(sp_state)
    with schema.get_conn(db) as conn:
        row = conn.execute("SELECT motion_type FROM scene_metrics WHERE scene_id = ?", (sid,)).fetchone()
        cov = store.coverage(conn)
    assert row[0] == "pan_right"
    assert cov["analyzed"] == 1 and job.status["skipped_videos"] == 1
    assert any(e.get("type") == "analysis" and e.get("done") for e in events)
    # Merging/extending a scene makes its metrics stale again.
    with schema.get_conn(db) as conn:
        conn.execute("UPDATE scenes SET end_ms = 2500 WHERE id = ?", (sid,))
        assert [v[0] for v in store.pending_videos(conn)] == [vid, missing]


# ── 3. migration ───────────────────────────────────────────────────────────
def _make_v8(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(schema.V4_SCHEMA)
    conn.executescript(schema.V5_SCHEMA)
    schema._apply_v6(conn)
    schema._apply_v7(conn)
    schema._apply_v8(conn)
    conn.execute("PRAGMA user_version = 8")
    conn.execute("""INSERT INTO videos (filepath, display_name, modified_at, status, group_name)
                    VALUES ('/a.mkv', 'A', 0, 'completed', 'JJK')""")
    conn.execute("""INSERT INTO scenes (video_id, scene_index, start_ms, end_ms, embedding)
                    VALUES (1, 0, 0, 1000, ?)""", (_unit(3).tobytes(),))
    conn.execute("INSERT INTO derush_items (scene_id, favorite) VALUES (1, 1)")
    conn.commit()
    conn.close()


def _tables(path: str) -> set[str]:
    with schema.get_conn(path) as conn:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','index')")}


V9_OBJECTS = {"scene_metrics", "characters", "character_examples", "character_matches",
              "idx_scene_metrics_motion", "idx_scene_metrics_intensity", "idx_scenes_duration",
              "idx_character_matches_score"}


def test_migration_v8_to_v9(tmp_path):
    db = str(tmp_path / "v8.db")
    _make_v8(db)
    assert schema.current_version(db) == 8
    schema.init_db(db)
    assert schema.current_version(db) == 9 == schema.SCHEMA_VERSION
    assert V9_OBJECTS <= _tables(db)
    with schema.get_conn(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0] == 1
        assert conn.execute("SELECT favorite FROM derush_items").fetchone()[0] == 1
        assert store.coverage(conn) == {"total": 1, "analyzed": 0, "pending": 1}
        # idempotent: applying v9 again (or init twice) is a no-op
        schema._apply_v9(conn)
    schema.init_db(db)
    assert schema.current_version(db) == 9


def test_migration_fresh_db(tmp_path):
    db = str(tmp_path / "fresh.db")
    schema.init_db(db)
    assert schema.current_version(db) == 9
    assert V9_OBJECTS <= _tables(db)
    with schema.get_conn(db) as conn:
        schema._apply_v9(conn)
        vid = add_video(conn, "x")
        add_scene(conn, vid, 0, _unit(1), metrics={"motion_type": "pan_left", "intensity": 0.5})
        # Deleting scenes (re-index) cascades metrics + matches but keeps
        # character examples (their embedding is stored).
        cid = chars.create_character(conn, "Gojo", None)
        chars.add_example(conn, cid, _unit(1), scene_id=1)
        conn.execute("INSERT INTO character_matches VALUES (?, 1, 0.9)", (cid,))
        conn.execute("DELETE FROM scenes")
        assert conn.execute("SELECT COUNT(*) FROM scene_metrics").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM character_matches").fetchone()[0] == 0
        assert conn.execute("SELECT scene_id FROM character_examples").fetchone()[0] is None


# ── 4. similar shots ───────────────────────────────────────────────────────
def _cluster(seed: int, n: int, spread: float = 0.15) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    center = _unit(seed)
    return [_norm(center + spread * rng.normal(size=DIM)) for _ in range(n)]


def test_similar_endpoint(sp_client, sp_state):
    with schema.get_conn(sp_state.primary_db) as conn:
        v1 = add_video(conn, "JJK 01", group="JJK")
        v2 = add_video(conn, "JJK 02", group="JJK")
        v3 = add_video(conn, "Naruto 01", group="Naruto")
        a = _cluster(10, 6)
        seed = add_scene(conn, v1, 0, a[0])
        same_ep = add_scene(conn, v1, 1, a[1])
        other_ep = [add_scene(conn, v2, i, a[2 + i], metrics={"motion_type": "zoom_in" if i == 0 else "static"})
                    for i in range(2)]
        other_series = add_scene(conn, v3, 0, a[4])
        noise = [add_scene(conn, v3, 1 + i, _unit(500 + i)) for i in range(5)]

    r = sp_client.get(f"/api/search/similar/{seed}", params={"top_k": 5}).json()
    ids = [x["id"] for x in r["results"]]
    assert seed not in ids
    assert set(ids[:4]) == {same_ep, *other_ep, other_series}
    assert r["seed"]["id"] == seed

    r = sp_client.get(f"/api/search/similar/{seed}", params={"exclude_same_video": True, "top_k": 50}).json()
    assert same_ep not in [x["id"] for x in r["results"]]

    r = sp_client.get(f"/api/search/similar/{seed}", params={"same_group": True, "top_k": 50}).json()
    assert {x["id"] for x in r["results"]} == {same_ep, *other_ep}

    import json
    r = sp_client.get(f"/api/search/similar/{seed}",
                   params={"filters": json.dumps({"motion": ["zoom_in"]})}).json()
    assert [x["id"] for x in r["results"]] == [other_ep[0]]
    assert r["results"][0]["metrics"]["motion_type"] == "zoom_in"

    assert sp_client.get("/api/search/similar/99999").status_code == 404
    with schema.get_conn(sp_state.primary_db) as conn:
        no_emb = add_scene(conn, v1, 9, None)
    assert sp_client.get(f"/api/search/similar/{no_emb}").status_code == 409
    assert noise  # noise scenes exist but rank below the cluster


# ── 5. characters ──────────────────────────────────────────────────────────
def test_score_embeddings_and_threshold():
    gojo = _cluster(1, 8, spread=0.2)
    other = [_unit(900 + i) for i in range(20)]
    E = np.vstack(gojo + other)
    s = chars.score_embeddings(E, np.vstack(gojo[:2]))
    assert s[:8].min() > s[8:].max()
    thr = chars.suggest_threshold(np.vstack(gojo[:3]))
    assert chars.MIN_THRESHOLD <= thr <= chars.MAX_THRESHOLD
    # A negative right next to some scene pushes that scene down.
    s_neg = chars.score_embeddings(E, np.vstack(gojo[:2]), np.vstack([other[0]]))
    assert s_neg[8] < s[8] and np.allclose(s_neg[:8], s[:8], atol=0.05)


def test_character_lifecycle(sp_client, sp_state):
    with schema.get_conn(sp_state.primary_db) as conn:
        v1 = add_video(conn, "JJK 01", group="JJK")
        v2 = add_video(conn, "Naruto", group="Naruto")
        gojo_vecs = _cluster(42, 10, spread=0.08)
        gojo = [add_scene(conn, v1, i, e, metrics={"motion_type": "pan_left" if i < 3 else "static",
                                                       "intensity": 0.2 + 0.08 * i})
                for i, e in enumerate(gojo_vecs)]
        others = [add_scene(conn, v1, 20 + i, _unit(700 + i)) for i in range(15)]
        # Same look in another series — excluded by the series scope.
        lookalike = add_scene(conn, v2, 0, gojo_vecs[0] + 0.01)
        # A decoy close to Gojo the user will reject.
        decoy = add_scene(conn, v1, 40, _norm(gojo_vecs[0] + 0.9 * _unit(9999)))

    r = sp_client.post("/api/characters", json={
        "name": "Gojo", "group_name": "JJK",
        "examples": [{"scene_id": gojo[0]}, {"scene_id": gojo[1]}, {"scene_id": gojo[2]}],
    })
    assert r.status_code == 200, r.text
    cid = r.json()["id"]
    assert r.json()["positives"] == 3
    assert sp_client.post("/api/characters", json={"name": "gojo", "group_name": "JJK"}).status_code == 409

    m = sp_client.get(f"/api/characters/{cid}/matches").json()
    thr = m["threshold"]
    above = {x["id"] for x in m["results"] if x["score"] >= thr}
    assert set(gojo) <= above
    assert lookalike not in {x["id"] for x in m["results"]}
    assert not (set(others) & above)
    labels = {x["id"]: x["label"] for x in m["results"]}
    assert labels[gojo[0]] == "positive"

    # Feedback: reject the decoy, accept another scene, then re-score.
    r = sp_client.post(f"/api/characters/{cid}/feedback", json={"reject": [decoy], "accept": [gojo[5]]}).json()
    assert r["negatives"] == 1 and r["positives"] == 4
    m = sp_client.get(f"/api/characters/{cid}/matches").json()
    ids = {x["id"] for x in m["results"]}
    assert decoy not in ids
    assert {x["id"]: x["label"] for x in m["results"]}[gojo[5]] == "positive"

    # Usable as a filter, in browse mode and combined with motion.
    r = sp_client.post("/api/search", json={"filters": {"characters": [cid]}, "top_k": 100}).json()
    assert {x["id"] for x in r["results"]} == set(gojo)
    assert r["results"][0]["characters"][0]["name"] == "Gojo"
    r = sp_client.post("/api/search", json={"filters": {"characters": [cid], "motion": ["pan_left"]}}).json()
    assert {x["id"] for x in r["results"]} == set(gojo[:3])

    # …and as a tag in text queries: "gojo" lifts Gojo scenes.
    v = _unit(31337)
    sp_state.fake_siglip.text_vecs["gojo"] = sp_state.fake_siglip.text_vecs["someone"] = v
    named = sp_client.post("/api/search", json={"query": "gojo", "threshold": -2, "top_k": 100}).json()
    plain = sp_client.post("/api/search", json={"query": "someone", "threshold": -2, "top_k": 100}).json()
    named = {x["id"]: x["score"] for x in named["results"]}
    plain = {x["id"]: x["score"] for x in plain["results"]}
    for sid in gojo:
        assert named[sid] - plain[sid] == pytest.approx(sf.CHARACTER_QUERY_BOOST, abs=1e-5)
    assert named[others[0]] == pytest.approx(plain[others[0]])

    # Manual threshold disables auto-threshold; delete cascades.
    r = sp_client.patch(f"/api/characters/{cid}", json={"threshold": 0.99}).json()
    assert r["threshold_auto"] is False and r["threshold"] == pytest.approx(chars.MAX_THRESHOLD)
    assert sp_client.get("/api/characters").json()["characters"][0]["name"] == "Gojo"
    sp_client.delete(f"/api/characters/{cid}")
    with schema.get_conn(sp_state.primary_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM character_matches").fetchone()[0] == 0


def test_character_from_crop_and_image(sp_client, sp_state):
    from PIL import Image
    import io
    img = Image.new("RGB", (320, 180), (200, 40, 40))
    for x in range(160, 320):
        for y in range(0, 180, 4):
            img.putpixel((x, y), (20, 20, 220))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    with schema.get_conn(sp_state.primary_db) as conn:
        vid = add_video(conn, "ep")  # filepath missing → crop falls back to thumbnail
        sid = add_scene(conn, vid, 0, _unit(5), thumb=buf.getvalue())
    cid = sp_client.post("/api/characters", json={"name": "Itadori"}).json()["id"]
    r = sp_client.post(f"/api/characters/{cid}/examples",
                    json={"scene_id": sid, "crop": [0.5, 0.0, 1.0, 1.0]})
    assert r.status_code == 200, r.text
    ex = r.json()["examples"][0]
    assert ex["crop"] == [0.5, 0.0, 1.0, 1.0] and ex["has_thumbnail"]
    assert sp_state.fake_siglip.calls == 1
    assert sp_client.get(f"/api/characters/{cid}/examples/{ex['id']}/thumbnail").status_code == 200
    r = sp_client.post(f"/api/characters/{cid}/examples/image",
                    files={"image": ("ref.jpg", buf.getvalue(), "image/jpeg")}, data={"polarity": "1"})
    assert r.status_code == 200 and r.json()["positives"] == 2
    assert sp_client.post(f"/api/characters/{cid}/examples",
                       json={"scene_id": sid, "crop": [0.5, 0.5, 0.505, 0.505]}).status_code == 422


# ── 6. combined filters ────────────────────────────────────────────────────
def test_combined_filters(sp_client, sp_state):
    with schema.get_conn(sp_state.primary_db) as conn:
        v1 = add_video(conn, "JJK 01", group="JJK")
        v2 = add_video(conn, "Misc")  # no group
        q = _unit(77)
        sp_state.fake_siglip.text_vecs["fight"] = q
        near = lambda k: _norm(q + 0.3 * _unit(1000 + k))  # noqa: E731
        s_pan = add_scene(conn, v1, 0, near(0), dur_ms=1500,
                          metrics={"motion_type": "pan_left", "intensity": 0.9, "dominant_color": "red",
                                   "motion_magnitude": 0.6})
        s_zoom = add_scene(conn, v1, 1, near(1), dur_ms=4000,
                           metrics={"motion_type": "zoom_in", "intensity": 0.4, "dominant_color": "blue",
                                    "accent_color": "red", "motion_magnitude": 0.3})
        s_static = add_scene(conn, v2, 0, near(2), dur_ms=8000,
                             metrics={"motion_type": "static", "intensity": 0.05, "dominant_color": "white"})
        s_nometrics = add_scene(conn, v2, 1, near(3), dur_ms=2500)
        conn.execute("INSERT INTO derush_items (scene_id, favorite) VALUES (?, 1)", (s_zoom,))
        conn.execute("INSERT INTO derush_items (scene_id, favorite) VALUES (?, 0)", (s_static,))

    def ids(filters, query="fight"):
        body = {"query": query, "threshold": -1.0, "top_k": 50, "filters": filters}
        res = sp_client.post("/api/search", json=body).json()["results"]
        return {r["id"] for r in res}

    every = {s_pan, s_zoom, s_static, s_nometrics}
    assert ids(None) == every
    assert ids({}) == every
    assert ids({"motion": ["pan_left", "zoom_in"]}) == {s_pan, s_zoom}
    assert ids({"intensity_min": 0.3}) == {s_pan, s_zoom}
    assert ids({"intensity_min": 0.3, "intensity_max": 0.5}) == {s_zoom}
    assert ids({"duration_min_ms": 2000}) == {s_zoom, s_static, s_nometrics}
    assert ids({"duration_max_ms": 3000}) == {s_pan, s_nometrics}
    assert ids({"colors": ["red"]}) == {s_pan, s_zoom}  # dominant OR accent
    assert ids({"groups": ["JJK"]}) == {s_pan, s_zoom}
    assert ids({"groups": [sf.NO_GROUP]}) == {s_static, s_nometrics}
    assert ids({"favorites": True}) == {s_zoom}
    assert ids({"motion_min": 0.5}) == {s_pan}
    assert ids({"groups": ["JJK"], "duration_min_ms": 2000, "colors": ["red"]}) == {s_zoom}
    assert ids({"motion": ["bogus"]}) == every  # unknown values are ignored

    # Browse (no query): ordered by intensity.
    res = sp_client.post("/api/search", json={"filters": {"groups": ["JJK"]}}).json()["results"]
    assert [r["id"] for r in res] == [s_pan, s_zoom]
    assert res[0]["metrics"]["motion_type"] == "pan_left"
    # Plain search results carry search+ fields too.
    res = sp_client.post("/api/search", json={"query": "fight", "threshold": -1}).json()["results"]
    by_id = {r["id"]: r for r in res}
    assert by_id[s_pan]["metrics"]["intensity"] == pytest.approx(0.9)
    assert by_id[s_nometrics]["metrics"] is None and by_id[s_nometrics]["characters"] == []

    # Image search honours the same filters (multipart JSON field).
    from PIL import Image
    import io
    import json
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 200, 10)).save(buf, format="PNG")
    r = sp_client.post("/api/search/image", files={"image": ("q.png", buf.getvalue(), "image/png")},
                    data={"threshold": "-1", "filters": json.dumps({"favorites": True})}).json()
    assert [x["id"] for x in r["results"]] == [s_zoom]

    facets = sp_client.get("/api/searchplus/facets").json()
    assert {g["name"] for g in facets["groups"]} == {"JJK", sf.NO_GROUP}
    assert facets["coverage"] == {"total": 4, "analyzed": 3, "pending": 1}
    assert dict((m["id"], m["count"]) for m in facets["motion_types"])["pan_left"] == 1


# ── 6b. search index cache ─────────────────────────────────────────────────
_OLD_SIGNATURE_SQL = """SELECT COUNT(*), COALESCE(MAX(s.id), 0), COALESCE(SUM(LENGTH(s.embedding)), 0)
                          FROM scenes s JOIN videos v ON v.id = s.video_id
                         WHERE v.status = 'completed' AND s.embedding IS NOT NULL"""


def test_search_index_cached_until_invalidated(sp_state, tmp_path, monkeypatch):
    db = sp_state.primary_db
    with schema.get_conn(db) as conn:
        v = add_video(conn, "Ep 01")
        a = add_scene(conn, v, 0, _unit(1))
        b = add_scene(conn, v, 1, _unit(2))
        add_scene(conn, add_video(conn, "Ep 02", status="pending"), 0, _unit(3))

    idx = sp_state.get_search_index()
    assert sorted(idx["sids"].tolist()) == [a, b]
    # Same value as the former per-search SQL signature, so the
    # ``characters.scored_signature`` already stored in user DBs stays valid.
    with schema.get_conn(db) as conn:
        assert idx["signature"] == (db, *conn.execute(_OLD_SIGNATURE_SQL).fetchone())

    # Warm lookups don't touch the DB any more...
    with schema.get_conn(db) as conn:
        c = add_scene(conn, v, 2, _unit(4))
    assert sp_state.get_search_index() is idx
    # ...until a writer invalidates.
    sp_state.invalidate_search_cache()
    idx = sp_state.get_search_index()
    assert sorted(idx["sids"].tolist()) == [a, b, c]

    # Scene merge (Derush) invalidates.
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api import scene as scene_api
    app = FastAPI()
    app.include_router(scene_api.router)
    assert TestClient(app).post("/api/scenes/merge", json={"scene_ids": [b, c]}).status_code == 200
    idx = sp_state.get_search_index()
    assert sorted(idx["sids"].tolist()) == [a, b]

    # A finished episode during an indexing run invalidates (per video, not
    # only at the end of the queue).
    import backend.indexing as indexing
    seen = {}

    def fake_queue(db_path, siglip, tagger, *, progress_cb, **kw):
        sp_state.get_search_index()
        progress_cb({"type": "indexing", "video": "x", "percent": 50, "message": "Embedded 1/2"})
        seen["mid"] = sp_state._search_cache is not None
        progress_cb({"type": "indexing", "video": "x", "percent": 100, "message": "Done"})
        seen["done"] = sp_state._search_cache is not None

    monkeypatch.setattr(indexing, "index_queue", fake_queue)
    monkeypatch.setattr(sp_state, "get_tagger", lambda: FakeTagger())
    sp_state._settings["indexing"]["generate_proxies"] = False
    assert sp_state.start_indexing()
    sp_state._index_thread.join(10)
    assert seen == {"mid": True, "done": False}

    # Switching the primary database invalidates.
    other = str(tmp_path / "other.db")
    schema.init_db(other)
    sp_state.get_search_index()
    sp_state.update_settings({"databases": {"active": [db, other], "primary": other}})
    assert sp_state.get_search_index()["rows"] == []


def test_sidecar_lifespan_runs_startup_hooks(sp_state, monkeypatch):
    """server.py uses a lifespan (no deprecated on_event): entering it starts
    the search+ auto analysis and hands the event loop to the state."""
    import warnings
    from fastapi.testclient import TestClient
    from backend.api import searchplus
    from backend import server

    calls = []
    monkeypatch.setattr(searchplus, "start_auto_analyze", lambda: calls.append(1))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        app = server.make_app()
    assert sp_state._loop is None
    with TestClient(app) as c:
        assert c.get("/api/status").status_code == 200
        assert calls == [1]
        assert sp_state._loop is not None


@pytest.mark.parametrize("enabled", [True, False])
def test_auto_analyze_setting(sp_state, monkeypatch, enabled):
    """Settings > Search toggle ``searchplus.auto_analyze`` gates the
    background analysis started at launch."""
    import time
    from backend.api import searchplus
    from backend.paths import user_data_dir

    assert sp_state.settings["searchplus"]["auto_analyze"] is True  # default
    sp_state.update_settings({"searchplus": {"auto_analyze": enabled}})
    with schema.get_conn(sp_state.primary_db) as conn:
        add_scene(conn, add_video(conn, "Old ep"), 0, _unit(1))  # no metrics yet
    marker = user_data_dir() / ".setup_complete"
    had_marker = marker.exists()
    marker.touch()

    started = []

    class _Job:
        def start(self, state):
            started.append(state)
            return True

    monkeypatch.setattr(searchplus, "get_job", lambda: _Job())
    monkeypatch.setattr(searchplus, "_AUTO_DELAY_S", 0.01)
    monkeypatch.setattr(searchplus, "_auto_started", False)
    try:
        searchplus.start_auto_analyze()
        time.sleep(0.5)
    finally:
        if not had_marker:
            marker.unlink()
    assert started == ([sp_state] if enabled else [])


# ── 7. speed on a 50k-scene library ────────────────────────────────────────
def test_filtered_search_speed_50k(tmp_path):
    """Sanity bound only — see tests/bench_search_plus.py for real numbers."""
    from bench_search_plus import build_library, time_queries
    db = str(tmp_path / "bench.db")
    build_library(db, n_scenes=50_000, dim=64)
    timings = time_queries(db, repeats=3)
    assert timings["filters_sql_ms"] < 400
    assert timings["browse_ms"] < 400


def test_scene_metrics_fall_back_to_cpu_when_gpu_palettes_fail(db_path, monkeypatch):
    import numpy as np

    from backend.indexing import pipeline
    from backend.searchplus import analysis, palette_gpu, store

    monkeypatch.setattr(palette_gpu, "palettes", lambda rgbs: None)
    monkeypatch.setattr(analysis, "aggregate_scene",
                        lambda acc, s, e, rgb, pal=None: {"pal": pal, "s": s})
    stored = []
    monkeypatch.setattr(store, "upsert_metrics", lambda conn, rows: stored.extend(rows))
    img = np.zeros((4, 4, 3), np.uint8)
    work = [(0, 0, 1000, None, img), (1, 1000, 2000, None, img)]
    pipeline._store_scene_metrics(db_path, None, work, [11, 12], gpu=True)
    assert [r[0] for r in stored] == [11, 12]


def test_invalid_filters_are_rejected_not_ignored(client):
    import io

    png = io.BytesIO()
    from PIL import Image
    Image.new("RGB", (8, 8)).save(png, format="PNG")
    for bad in ("{not json", "[1, 2]", '{"intensity_min": "high"}'):
        res = client.post("/api/search/image", data={"filters": bad},
                          files={"image": ("q.png", png.getvalue(), "image/png")})
        assert res.status_code == 422, (bad, res.status_code)
