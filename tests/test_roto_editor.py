"""Roto editor (v06): project, corrections, undo, suspects, propagation,
edge settings, export, timeline link, API flow — CPU, models faked."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


# ── synthetic shot: a dark-outlined disc moving right on a light background ─
def _frames(tmp: Path, n: int = 8, w: int = 160, h: int = 96) -> list[Path]:
    import cv2
    d = tmp / "src"
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        img = np.full((h, w, 3), 220, np.uint8)
        cx = 50 + 4 * i
        cv2.circle(img, (cx, 48), 24, (30, 30, 30), -1)
        cv2.circle(img, (cx, 48), 21, (200, 80, 60), -1)
        p = d / f"{i:05d}.png"
        Image.fromarray(img).save(p)
        out.append(p)
    return out


def _disc(i: int, w: int = 160, h: int = 96, r: int = 24) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return (((xx - (50 + 4 * i)) ** 2 + (yy - 48) ** 2) <= r * r).astype(np.float32)


@pytest.fixture
def project(tmp_path):
    from backend.roto.project import RotoProject
    return RotoProject.create(_frames(tmp_path), fps=24.0, name="test", root=tmp_path / "projects")


def _ai(project, masks):
    project.apply("main", {t: masks[t] for t in range(project.n)}, label="ai", ai=True)


def test_project_create_save_load_roundtrip(project):
    from backend.roto.project import RotoProject
    _ai(project, [_disc(i) for i in range(project.n)])
    project.apply("main", {3: np.zeros((96, 160), np.float32)}, label="brush", lock=[3])
    project.save()
    p2 = RotoProject.load(project.dir)
    assert p2.n == 8 and p2.size == (160, 96)
    assert p2.has_ai_pass()
    assert p2.locked["main"] == {3}
    assert p2.masks["main"][3].max() == 0
    assert p2.masks["main"][2].max() == 255
    assert p2.ai["main"][3].max() == 255      # the AI copy survives the correction


def test_brush_add_remove_erase_and_undo_redo(project):
    from backend.roto import engine
    _ai(project, [_disc(i) for i in range(project.n)])
    cur = project.matte("main", 0)
    new = engine.paint(cur, project.ai["main"][0] / 255.0,
                       [{"points": [(140, 10), (150, 10)], "radius": 5, "mode": "add"}])
    assert new[10, 145] > 0.9 and cur[10, 145] == 0
    new2 = engine.paint(new, project.ai["main"][0] / 255.0,
                        [{"points": [(50, 48)], "radius": 6, "mode": "remove"}])
    assert new2[48, 50] < 0.1
    new3 = engine.paint(new2, project.ai["main"][0] / 255.0,
                        [{"points": [(50, 48)], "radius": 8, "mode": "erase"}])
    assert new3[48, 50] > 0.9                 # the AI pass is back under the eraser
    project.apply("main", {0: new2}, label="brush", lock=[0])
    assert 0 in project.locked["main"]
    assert project.undo() == 0
    assert project.masks["main"][0][48, 50] == 255 and 0 not in project.locked["main"]
    assert project.redo() == 0
    assert project.masks["main"][0][48, 50] < 30 and 0 in project.locked["main"]


def test_suspects_flag_lost_frame_and_skip_locked(project):
    masks = [_disc(i) for i in range(project.n)]
    masks[5] = np.zeros_like(masks[5])
    _ai(project, masks)
    s = project.suspects()
    lost = [x for x in s if "lost" in x["reasons"]]
    assert [x["frame"] for x in lost] == [5]
    project.set_locked("main", 5, True)
    assert all(x["frame"] != 5 for x in project.suspects())


def test_suspects_specks_and_holes():
    from backend.roto.suspects import decide, frame_features
    a = _disc(0, r=30)
    b = a.copy()
    b[46:50, 48:52] = 0          # a hole in the character
    b[5:8, 140:143] = 1          # a speck far away
    feats = [frame_features(a), frame_features(b, a), frame_features(a, b)]
    s = {x["frame"]: x["reasons"] for x in decide(feats)}
    assert "hole" in s.get(1, []) and "specks" in s.get(1, [])
    assert 0 not in s


def test_object_lost_reason():
    from backend.roto.suspects import decide, frame_features
    a = _disc(0)
    a[10:30, 120:150] = 1        # a held object away from the body
    b = _disc(0)
    feats = [frame_features(a), frame_features(b, a)]
    s = {x["frame"]: x["reasons"] for x in decide(feats)}
    assert "object_lost" in s.get(1, [])


def test_merge_track_keeps_ai_edge_and_user_changes():
    from backend.roto.engine import merge_track
    ai = _disc(0)
    ai[40:56, 74:100] = 1.0      # background chunk glued to the character by the AI
    track = _disc(0)             # what the user kept on the corrected frame, tracked
    track[0:10, 0:10] = 1.0      # ... plus something the AI missed
    m = merge_track(ai, track)
    assert m[48, 95] < 0.1       # chunk removed
    assert m[48, 50] > 0.99      # body kept
    assert m[3, 3] > 0.9         # addition kept
    assert np.allclose(merge_track(None, track), track)


def test_propagate_flow_respects_locked_frames(project, state):
    from backend.roto import engine
    masks = [_disc(i) for i in range(project.n)]
    masks[4] = np.zeros_like(masks[4])          # the AI lost frame 4
    _ai(project, masks)
    fixed = _disc(2)
    project.apply("main", {2: fixed}, label="brush", lock=[2])
    project.apply("main", {6: np.zeros_like(fixed)}, label="brush", lock=[6])   # user says: empty
    new, method = engine.propagate(project, state, "main", 2, "forward", use_sam2=False)
    assert method == "optical flow"
    assert set(new) == {3, 4, 5}                # stops at locked frame 6, never writes it
    assert (new[4] > 0.5).sum() > 0.6 * _disc(4).sum()   # the lost frame is refilled
    new_b, _ = engine.propagate(project, state, "main", 2, "backward", use_sam2=False)
    assert set(new_b) == {0, 1}


def test_edges_settings():
    from backend.roto.edges import EdgeSettings, apply_edges
    a = _disc(0)
    base = apply_edges(a, None, EdgeSettings(keep_ink=False))
    grown = apply_edges(a, None, EdgeSettings(keep_ink=False, grow_px=3))
    shrunk = apply_edges(a, None, EdgeSettings(keep_ink=False, grow_px=-3))
    assert grown.sum() > base.sum() > shrunk.sum()
    soft = apply_edges(a, None, EdgeSettings(keep_ink=False, softness=1))
    hard = apply_edges(soft, None, EdgeSettings(keep_ink=False, softness=-1))
    n_mid = lambda x: int(((x > 0.05) & (x < 0.95)).sum())   # noqa: E731
    assert n_mid(soft) > n_mid(base) and n_mid(hard) < n_mid(soft)
    assert soft[48, 50] == 1.0                  # the inside stays solid
    s = EdgeSettings.from_dict({"grow_px": 99, "softness": -5, "temporal": 2})
    assert s.grow_px == 6 and s.softness == -1 and s.temporal == 1


def test_keep_ink_adds_the_outline(project):
    from backend.roto.edges import EdgeSettings, apply_edges
    a = _disc(0, r=21)                          # matte stops inside the dark outline
    rgb = project.rgb(0)
    with_ink = apply_edges(a, rgb, EdgeSettings(keep_ink=True))
    without = apply_edges(a, rgb, EdgeSettings(keep_ink=False))
    ring = (_disc(0, r=24) > 0) & ~(_disc(0, r=21) > 0)
    assert with_ink[ring].mean() > without[ring].mean() + 0.3


def test_export_writes_prores_png_and_matte(project, tmp_path):
    import subprocess
    from backend.roto.export import export_project
    from tests.conftest import ffmpeg_exe
    _ai(project, [_disc(i) for i in range(project.n)])
    files = export_project(project, tmp_path / "out", ["prores", "png", "matte"])
    assert Path(files["prores"]).exists() and Path(files["matte"]).exists()
    pngs = sorted(Path(files["png_dir"]).glob("*.png"))
    assert len(pngs) == project.n
    with Image.open(pngs[0]) as im:
        assert im.mode == "RGBA"
        a = np.asarray(im)[..., 3]
    assert a[48, 50] == 255 and a[5, 5] == 0
    probe = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", files["prores"]], capture_output=True, text=True)
    assert "prores" in probe.stderr and "yuva444p" in probe.stderr


def test_cutout_link_for_timeline(project, tmp_path, monkeypatch):
    from backend.roto import links
    from backend.timeline.model import ClipSpec
    mov = tmp_path / "x_alpha.mov"
    mov.write_bytes(b"0")
    project.meta.update({"scene_id": 7, "start_ms": 1000, "end_ms": 4000, "exports": {"prores": str(mov)}})
    project.save()
    monkeypatch.setattr(links, "projects_root", lambda: project.dir.parent)
    spec = ClipSpec(source_path="/ep.mkv", name="ep_scene", start_ms=1500, end_ms=3000, scene_id=7)
    other = ClipSpec(source_path="/ep.mkv", name="b", start_ms=500, end_ms=3000, scene_id=7)   # not covered
    assert links.link_cutouts([spec, other]) == 1
    assert spec.source_path == str(mov) and (spec.start_ms, spec.end_ms) == (500, 2000)
    assert other.source_path == "/ep.mkv"


# ── API flow with faked models ─────────────────────────────────────────────
def _fake_models(monkeypatch):
    from backend.roto import engine

    def fake_ai_pass(project, state, *, target="all", frame=0, positive=(), negative=(), progress=None,
                     cancel=None):
        out = np.stack([_disc(i) for i in range(project.n)])
        out[4] = 0
        for i in range(project.n):
            cancel()
            progress(i + 1, project.n)
        return out

    def fake_region(project, state, t, x, y, positive=True):
        yy, xx = np.mgrid[0:96, 0:160]
        return (((xx - x) ** 2 + (yy - y) ** 2) <= 64).astype(np.float32), "fake SAM"

    monkeypatch.setattr(engine, "ai_pass", fake_ai_pass)
    monkeypatch.setattr(engine, "click_region", fake_region)


def _wait(client, pid):
    for _ in range(200):
        j = client.get(f"/api/roto/{pid}/job").json()
        if j and j["done"]:
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_api_editor_flow(client, db_path, tmp_path, monkeypatch):
    from backend.db import queries, schema
    from tests.conftest import make_solid_clip
    _fake_models(monkeypatch)
    clip = make_solid_clip(tmp_path / "c.mp4", ("red", "blue"), size="160x96", fps=24)
    with schema.get_conn(db_path) as conn:
        vid = queries.upsert_video(conn, str(clip), "Episode", 2000, 24.0, "160x96", "test", status="completed")
        sid = queries.insert_scene(conn, vid, 0, 0, 1000, None, None)
    r = client.post("/api/roto/open", json={"scene_id": sid})
    assert r.status_code == 200, r.text
    p = r.json()
    pid, n = p["id"], p["n_frames"]
    assert n >= 20 and not p["has_ai_pass"] and p["models"]["device"] == "cpu"
    assert client.get(f"/api/roto/{pid}/matte/0").status_code == 404
    assert client.get(f"/api/roto/{pid}/frame/0").headers["content-type"] == "image/jpeg"

    assert client.post(f"/api/roto/{pid}/detect", json={"target": "all"}).status_code == 200
    j = _wait(client, pid)
    assert j["error"] is None and j["percent"] == 100
    p = client.get(f"/api/roto/{pid}").json()
    assert p["has_ai_pass"] and any(s["frame"] == 4 and "lost" in s["reasons"] for s in p["suspects"])
    m = client.get(f"/api/roto/{pid}/matte/0")
    assert m.status_code == 200 and m.headers["content-type"] == "image/png"

    r = client.post(f"/api/roto/{pid}/brush", json={"frame": 4, "strokes": [
        {"points": [[70, 48], [80, 48]], "radius": 20, "mode": "add"}]})
    assert r.status_code == 200 and 4 in r.json()["keyframes"]["main"]
    assert all(s["frame"] != 4 for s in r.json()["suspects"])
    r = client.post(f"/api/roto/{pid}/click", json={"frame": 2, "x": 10, "y": 10, "positive": True})
    assert r.status_code == 200 and r.json()["model"] == "fake SAM" and r.json()["ms"] >= 0
    assert client.post(f"/api/roto/{pid}/click", json={"frame": 2, "x": 999, "y": 10}).status_code == 400

    assert client.post(f"/api/roto/{pid}/propagate", json={"frame": 2, "direction": "forward"}).status_code == 200
    j = _wait(client, pid)
    assert j["error"] is None and j["result"]["method"] == "optical flow"
    assert j["result"]["frames"] == 1          # frame 3 only: frame 4 is locked

    u = client.post(f"/api/roto/{pid}/undo").json()
    assert u["frame"] == 3 and u["can_redo"]
    assert client.post(f"/api/roto/{pid}/redo").json()["frame"] == 3

    r = client.put(f"/api/roto/{pid}/settings", json={"keep_ink": False, "softness": 0.5, "grow_px": 2,
                                                      "temporal": 0.3})
    assert r.json()["settings"]["grow_px"] == 2
    assert client.get(f"/api/roto/{pid}/matte/5").status_code == 200

    r = client.post(f"/api/roto/{pid}/layers", json={"name": "Asta"}).json()
    assert len(r["layers"]) == 2
    lid = r["layers"][1]["id"]
    assert client.delete(f"/api/roto/{pid}/layers/{lid}").json()["layers"][0]["id"] == "main"

    assert client.post(f"/api/roto/{pid}/export", json={"formats": ["prores", "matte"],
                                                        "add_to_selection": True}).status_code == 200
    j = _wait(client, pid)
    assert j["error"] is None, j
    assert Path(j["result"]["files"]["prores"]).exists() and j["result"]["added_to_selection"]
    with schema.get_conn(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM derush_items WHERE scene_id = ?", (sid,)).fetchone()[0] == 1
    # the timeline export now places the cut-out clip
    tl = client.post("/api/timeline/preview", json={"scene_ids": [sid]})
    assert tl.status_code == 200, tl.text
    assert "_alpha.mov" in tl.text
    # every editor format writes a timeline with the cut-out clip (no audio in it)
    for fmt in ("fcp7xml", "fcpxml", "edl", "otio", "vegas", "aftereffects"):
        out = tmp_path / f"tl_{fmt}"
        r = client.post("/api/timeline/export", json={"scene_ids": [sid], "format": fmt, "output_path": str(out)})
        assert r.status_code == 200, (fmt, r.text)
    off = client.post("/api/timeline/preview", json={"scene_ids": [sid], "options": {"use_cutouts": False}})
    assert "_alpha.mov" not in off.text

    # reopening the same shot resumes the saved project
    from backend.roto import project as proj
    proj._open.clear()
    again = client.post("/api/roto/open", json={"scene_id": sid}).json()
    assert again["id"] == pid and again["has_ai_pass"] and 4 in again["keyframes"]["main"]
    assert any(s["id"] == pid for s in client.get(f"/api/roto/scene/{sid}").json()["projects"])


def test_api_cancel_and_busy(client, db_path, tmp_path, monkeypatch):
    from backend.db import queries, schema
    from backend.roto import engine
    from tests.conftest import make_solid_clip

    def slow(project, state, *, progress=None, cancel=None, **kw):
        for _ in range(200):
            cancel()
            time.sleep(0.02)
        return np.zeros((project.n, 96, 160), np.float32)
    monkeypatch.setattr(engine, "ai_pass", slow)
    clip = make_solid_clip(tmp_path / "c.mp4", ("red",), size="160x96", fps=24)
    with schema.get_conn(db_path) as conn:
        vid = queries.upsert_video(conn, str(clip), "Episode", 1000, 24.0, "160x96", "test", status="completed")
        sid = queries.insert_scene(conn, vid, 0, 0, 1000, None, None)
    pid = client.post("/api/roto/open", json={"scene_id": sid}).json()["id"]
    assert client.post(f"/api/roto/{pid}/detect", json={"target": "main"}).status_code == 200
    assert client.post(f"/api/roto/{pid}/detect", json={"target": "main"}).status_code == 409
    assert client.post(f"/api/roto/{pid}/brush", json={"frame": 0, "strokes": []}).status_code == 409
    client.post(f"/api/roto/{pid}/cancel")
    j = _wait(client, pid)
    assert j["cancelled"] and not client.get(f"/api/roto/{pid}").json()["has_ai_pass"]


def test_steady_background_pocket_is_not_a_suspect_hole():
    from backend.roto.suspects import decide, frame_features
    a = _disc(0, r=34)
    a[30:44, 40:56] = 0          # a pocket that stays on every frame (between arm and body)
    feats = [frame_features(a), frame_features(a, a), frame_features(a, a)]
    assert not any("hole" in x["reasons"] for x in decide(feats))
