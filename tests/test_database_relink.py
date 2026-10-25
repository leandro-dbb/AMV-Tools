"""Moving a database to a new machine: verify + bulk relink."""
from __future__ import annotations

import shutil

from backend.db import queries, schema


def _seed_missing_video(db_path, clip, tmp_path):
    old_path = tmp_path / "old-library" / clip.name
    with schema.get_conn(db_path) as conn:
        vid = queries.upsert_video(
            conn, str(old_path), "Episode", 3000, 25.0, "320x240", "test",
            status="completed",
        )
    return vid, old_path


def test_verify_lists_missing_files(client, db_path, clip, tmp_path):
    vid, old_path = _seed_missing_video(db_path, clip, tmp_path)

    missing = client.get(f"/api/databases/verify?path={db_path}").json()["missing"]
    assert missing == [{"id": vid, "filepath": str(old_path)}]


def test_relink_folder_after_library_move(client, db_path, clip, tmp_path):
    vid, _old_path = _seed_missing_video(db_path, clip, tmp_path)

    new_root = tmp_path / "new-library"
    new_root.mkdir()
    moved = new_root / clip.name
    shutil.copy2(clip, moved)

    res = client.post(
        "/api/databases/relink_folder",
        json={"db_path": db_path, "search_dirs": [str(new_root)]},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["relinked"] == 1
    assert body["still_missing"] == []
    assert body["ambiguous"] == []

    with schema.get_conn(db_path) as conn:
        filepath = conn.execute(
            "SELECT filepath FROM videos WHERE id = ?", (vid,)
        ).fetchone()[0]
    assert filepath == str(moved)
    assert client.get(f"/api/databases/verify?path={db_path}").json()["missing"] == []


def test_relink_ambiguous_duplicate_basenames_are_skipped(client, db_path, clip, tmp_path):
    vid, _old_path = _seed_missing_video(db_path, clip, tmp_path)

    root = tmp_path / "dup"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    shutil.copy2(clip, root / "a" / clip.name)
    shutil.copy2(clip, root / "b" / clip.name)

    body = client.post(
        "/api/databases/relink_folder",
        json={"db_path": db_path, "search_dirs": [str(root)]},
    ).json()
    assert body["relinked"] == 0
    assert len(body["ambiguous"]) == 1

    with schema.get_conn(db_path) as conn:
        filepath = conn.execute(
            "SELECT filepath FROM videos WHERE id = ?", (vid,)
        ).fetchone()[0]
    assert filepath.endswith("old-library/" + clip.name)


def test_single_video_relink(client, db_path, clip, tmp_path):
    vid, _old_path = _seed_missing_video(db_path, clip, tmp_path)
    body = client.post(
        "/api/databases/relink",
        json={"db_path": db_path, "video_id": vid, "new_filepath": str(clip)},
    ).json()
    assert body["ok"] is True
    assert client.get(f"/api/databases/verify?path={db_path}").json()["missing"] == []


def test_library_changes_refused_while_indexing(client, state, db_path, tmp_path):
    import threading

    other = str(tmp_path / "other.db")
    schema.init_db(other)
    release = threading.Event()
    state._index_thread = threading.Thread(target=release.wait, daemon=True)
    state._index_thread.start()
    try:
        res = client.post("/api/databases/primary", json={"path": other})
        assert res.status_code == 409
        assert "Indexing is running" in res.json()["detail"]
        assert state.primary_db == db_path
        assert client.post("/api/databases/merge",
                           json={"target": db_path, "sources": [other]}).status_code == 409
        assert client.post("/api/databases/cleanup", json={"path": db_path}).status_code == 409
        # re-selecting the current primary is harmless
        assert client.post("/api/databases/primary", json={"path": db_path}).status_code == 200
    finally:
        release.set()
        state._index_thread.join()
    assert client.post("/api/databases/primary", json={"path": other}).status_code == 200


def test_indexing_and_reembed_wait_for_a_merge(state):
    from backend.api.reembed import _indexing_running
    from backend.db.merge import MERGE_LOCK

    with MERGE_LOCK:
        assert state.start_indexing() is False
        assert _indexing_running(state) is True
    assert _indexing_running(state) is False


def test_merge_from_a_path_with_uri_characters(tmp_path, db_path):
    from backend.db.merge import merge_databases

    src = tmp_path / "lib #1 ?50%" / "src.db"
    schema.init_db(str(src))
    with schema.get_conn(str(src)) as conn:
        queries.upsert_video(conn, str(tmp_path / "ep.mkv"), "Episode", 3000, 25.0, "320x240",
                             "test", status="completed")
    stats = merge_databases(db_path, [str(src)])
    assert stats["videos_added"] == 1
    src.rename(src.with_name("moved.db"))   # not left open / locked
