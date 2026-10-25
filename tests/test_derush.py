"""Derush keep/favorite/folder flows."""
from __future__ import annotations

from backend.db import queries, schema


def _seed(db_path, clip):
    with schema.get_conn(db_path) as conn:
        vid = queries.upsert_video(
            conn, str(clip), "Episode", 3000, 25.0, "320x240", "test", status="completed"
        )
        sid = queries.insert_scene(conn, vid, 0, 0, 1000, None, b"")
    return vid, sid


def test_toggle_keep_and_unkeep(client, db_path, clip):
    _vid, sid = _seed(db_path, clip)

    first = client.post("/api/derush/toggle", json={"scene_id": sid}).json()
    assert first["kept"] is True
    assert first["favorite"] is False
    assert first["item_id"] is not None

    listing = client.get("/api/derush").json()
    assert [i["scene_id"] for i in listing["items"]] == [sid]

    second = client.post("/api/derush/toggle", json={"scene_id": sid}).json()
    assert second["kept"] is False
    assert client.get("/api/derush").json()["items"] == []


def test_level_ladder_none_kept_favorite(client, db_path, clip):
    _vid, sid = _seed(db_path, clip)

    levels = []
    for _ in range(3):
        levels.append(
            client.post("/api/derush/level", json={"scene_id": sid, "delta": 1}).json()["level"]
        )
    assert levels == [1, 2, 2]

    down = [
        client.post("/api/derush/level", json={"scene_id": sid, "delta": -1}).json()["level"]
        for _ in range(3)
    ]
    assert down == [1, 0, 0]


def test_favorite_toggle_rhythm(client, db_path, clip):
    _vid, sid = _seed(db_path, clip)

    added = client.post(
        "/api/derush/toggle", json={"scene_id": sid, "favorite": True}
    ).json()
    assert added["kept"] is True and added["favorite"] is True

    removed = client.post(
        "/api/derush/toggle", json={"scene_id": sid, "favorite": True}
    ).json()
    assert removed["kept"] is False


def test_folders_membership_and_item_patch(client, db_path, clip):
    _vid, sid = _seed(db_path, clip)
    folder = client.post("/api/derush/folders", json={"name": "S23"}).json()
    assert folder["ok"] is True

    added = client.post(
        "/api/derush/toggle", json={"scene_id": sid, "folder_id": folder["id"]}
    ).json()
    item_id = added["item_id"]

    listing = client.get("/api/derush").json()
    assert listing["folders"][0]["name"] == "S23"
    assert listing["items"][0]["folder_ids"] == [folder["id"]]

    patched = client.patch(
        f"/api/derush/items/{item_id}",
        json={"custom_name": "Hero shot", "favorite": True},
    ).json()
    assert patched["ok"] is True

    item = client.get("/api/derush").json()["items"][0]
    assert item["custom_name"] == "Hero shot"
    assert item["favorite"] is True

    # Root-only listing excludes the filed item; the folder listing includes it.
    assert client.get("/api/derush").json()["folders"][0]["item_count"] == 1


def test_kept_scene_ids_can_be_filtered_by_video(client, db_path, clip):
    vid, sid = _seed(db_path, clip)
    client.post("/api/derush/toggle", json={"scene_id": sid})

    kept = client.get(f"/api/derush/kept?video_id={vid}").json()["scene_ids"]
    assert kept == [sid]
    assert client.get("/api/derush/kept?video_id=999").json()["scene_ids"] == []


def test_toggle_missing_scene_is_404(client, db_path):
    res = client.post("/api/derush/toggle", json={"scene_id": 12345})
    assert res.status_code == 404
