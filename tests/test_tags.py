"""Tag aggregation endpoints."""
from __future__ import annotations

from backend.db import queries, schema


def _seed(db_path, clip):
    with schema.get_conn(db_path) as conn:
        vid = queries.upsert_video(
            conn, str(clip), "Episode", 3000, 25.0, "320x240", "test", status="completed"
        )
        s1 = queries.insert_scene(conn, vid, 0, 0, 1000, None, b"")
        s2 = queries.insert_scene(conn, vid, 1, 1000, 2000, None, b"")
        queries.insert_tags(conn, s1, [("sunset", "general", 0.9), ("sky", "general", 0.8)])
        queries.insert_tags(conn, s2, [("sunset", "general", 0.7)])
    return vid, s1, s2


def test_all_tags_are_aggregated(client, db_path, clip):
    _seed(db_path, clip)

    tags = client.get("/api/tags/all").json()["tags"]
    by_name = {t["tag"]: t["count"] for t in tags}
    assert by_name == {"sunset": 2, "sky": 1}
    assert tags[0]["tag"] == "sunset"


def test_per_video_tags_with_counts(client, db_path, clip):
    vid, _s1, _s2 = _seed(db_path, clip)

    tags = client.get(f"/api/videos/{vid}/tags").json()["tags"]
    by_name = {t["tag"]: t["count"] for t in tags}
    assert by_name == {"sunset": 2, "sky": 1}
    assert all(t["category"] == "general" for t in tags)


def test_all_tags_limit(client, db_path, clip):
    _seed(db_path, clip)
    tags = client.get("/api/tags/all?limit=1").json()["tags"]
    assert len(tags) == 1
