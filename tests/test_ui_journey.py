"""Backend fixes found while driving the real UI end to end (chantier v06 B)."""
from __future__ import annotations

import backend.api.status as status_api


def test_cpu_fallback_does_not_claim_torch_missing(client, state, monkeypatch):
    monkeypatch.setattr(status_api, "_package_version", lambda name: "2.4.1")
    body = client.get("/api/status").json()
    assert body["requested_device"] == "auto"
    assert "not installed" not in body["device_fallback"]
    assert "CPU" in body["device_fallback"]


def test_cpu_fallback_reports_missing_torch(client, state, monkeypatch):
    monkeypatch.setattr(status_api, "_package_version", lambda name: None)
    body = client.get("/api/status").json()
    assert "torch is not installed" in body["device_fallback"]


def test_e2e_install_skips_uv_sync_but_finishes_setup(client, state, data_dir, monkeypatch):
    monkeypatch.setenv("AMV_E2E_SKIP_SYNC", "1")
    (data_dir / ".setup_complete").unlink()
    assert client.get("/api/status").json()["setup_required"] is True
    r = client.post("/api/setup/install", json={"backend": "cpu"}).json()
    assert r["ok"] is True
    assert "uv sync skipped" in r["log"]
    assert (data_dir / ".setup_complete").read_text() == "cpu"
    assert state.settings["indexing"]["device"] == "cpu"


def test_subsegment_tagger_pass_reports_progress(monkeypatch):
    """The sub-segmentation tagger pass used to stay silent for minutes on
    CPU (bar frozen at 5 %): it now reports after every batch."""
    from backend.indexing import pipeline

    monkeypatch.setattr(pipeline.frames, "extract_frames_at",
                        lambda path, ts: ((t, object()) for t in ts))

    class Tagger:
        def embed_images(self, imgs):
            return [0.0] * len(imgs)

    seen: list[int] = []
    out = pipeline._batched_tagger_embeddings("x.mkv", list(range(10)), Tagger(), 4,
                                              lambda: False, on_batch=seen.append)
    assert len(out) == 10
    assert seen == [4, 8, 10]


def test_index_queue_tags_events_with_file_rank_and_failures(tmp_path, monkeypatch):
    from backend.db import schema
    from backend.indexing import pipeline

    db = str(tmp_path / "lib.db")
    schema.init_db(db)
    media = tmp_path / "Séries test"
    media.mkdir()
    for name in ("a.mkv", "b.mkv"):
        (media / name).write_bytes(b"not a video")
    with schema.get_conn(db) as conn:
        from backend.db import queries
        queries.add_to_queue(conn, str(media), True, True)

    def fake_index_video(db_path, fp, siglip, tagger, progress_cb=None, **kw):
        if fp.endswith("b.mkv"):
            raise ValueError("Invalid data found when processing input")
        progress_cb({"type": "indexing", "video": "a", "percent": 100, "message": "Done"})

    monkeypatch.setattr(pipeline, "index_video", fake_index_video)

    class Tagger:
        def offload(self):
            pass

    events: list[dict] = []
    pipeline.index_queue(db, None, Tagger(), progress_cb=events.append)
    done = next(e for e in events if e.get("message") == "Done")
    failed = next(e for e in events if e.get("failed"))
    assert done["file_count"] == failed["file_count"] == 2
    assert {done["file_index"], failed["file_index"]} == {1, 2}
    assert failed["path"].endswith("b.mkv")
    assert "Invalid data" in failed["message"]
    assert events[-1] == {"type": "idle"}


def test_settings_refuse_a_read_only_export_folder(client, state, tmp_path):
    import os

    locked = tmp_path / "Exports lecture seule"
    locked.mkdir()
    os.chmod(locked, 0o500)
    try:
        if os.access(locked, os.W_OK):  # running as root: permissions are moot
            return
        r = client.put("/api/settings", json={"export": {"output_folder": str(locked)}})
        assert r.status_code == 400
        assert r.json()["detail"]["code"] == "output_folder_not_writable"
        assert state.settings["export"]["output_folder"] != str(locked)
        ok = tmp_path / "Exports été"
        r = client.put("/api/settings", json={"export": {"output_folder": str(ok)}})
        assert r.status_code == 200 and ok.is_dir()
        # Other export settings stay editable once the folder is saved.
        assert client.put("/api/settings", json={"export": {"output_folder": str(ok), "crf": 20}}).status_code == 200
    finally:
        os.chmod(locked, 0o700)


def test_cpu_chosen_on_purpose_is_not_labelled_a_fallback():
    from backend.models.device import detect_device

    assert detect_device("cpu").name == "CPU"


def test_export_names_keep_accented_and_japanese_letters():
    from backend.api import derush, export

    assert export._sanitize("Clip été 01") == "Clip_été_01"
    assert export._sanitize("Pokémon: 進撃/の巨人?") == "Pokémon_進撃_の巨人"
    assert derush._sanitize("Black Clover · scène 3") == "Black Clover _ scène 3"
    assert export._sanitize('a<b>c:"d|e*') == "a_b_c_d_e"
