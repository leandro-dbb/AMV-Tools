"""Index queue + two-phase pipeline with fully mocked ML models."""
from __future__ import annotations

from backend.db import schema
from backend.indexing import pipeline


def _enqueue(client, path) -> None:
    res = client.post(
        "/api/index/queue",
        json={"paths": [{"path": str(path), "recursive": False}]},
    )
    assert res.status_code == 200
    assert res.json()["added"] == 1


def test_queue_endpoint_add_and_list(client, clip):
    _enqueue(client, clip)

    items = client.get("/api/index/queue").json()["items"]
    assert len(items) == 1
    assert items[0]["path"] == str(clip)
    assert items[0]["is_directory"] is False


def test_index_queue_runs_both_phases(client, clip, fake_models, db_path):
    siglip, tagger = fake_models
    _enqueue(client, clip)

    result = pipeline.index_queue(
        db_path, siglip, tagger,
        mode="fast",
        enable_sub_segmentation=False,
        batch_size=2,
        tag_threshold=0.5,
    )

    assert result == "completed"
    with schema.get_conn(db_path) as conn:
        video = conn.execute(
            "SELECT id, status, duration_ms FROM videos WHERE filepath = ?",
            (str(clip),),
        ).fetchone()
        assert video is not None
        video_id, status, duration_ms = video
        assert status == "completed"
        assert duration_ms >= 2800

        scenes = conn.execute(
            "SELECT id, embedding FROM scenes WHERE video_id = ? ORDER BY scene_index",
            (video_id,),
        ).fetchall()
        assert len(scenes) == 3
        assert all(row[1] is not None for row in scenes)

        tag_count = conn.execute(
            "SELECT COUNT(*) FROM scene_tags t JOIN scenes s ON s.id = t.scene_id "
            "WHERE s.video_id = ?",
            (video_id,),
        ).fetchone()[0]
        assert tag_count > 0

        # The worker drains the queue once finished.
        assert conn.execute("SELECT COUNT(*) FROM index_queue").fetchone()[0] == 0


def test_auto_skip_does_not_duplicate_scenes(client, clip, fake_models, db_path):
    siglip, tagger = fake_models
    _enqueue(client, clip)

    kwargs = dict(mode="fast", enable_sub_segmentation=False, batch_size=2)
    pipeline.index_queue(db_path, siglip, tagger, **kwargs)
    pipeline.index_queue(db_path, siglip, tagger, **kwargs)

    with schema.get_conn(db_path) as conn:
        video_id = conn.execute(
            "SELECT id FROM videos WHERE filepath = ?", (str(clip),)
        ).fetchone()[0]
        count = conn.execute(
            "SELECT COUNT(*) FROM scenes WHERE video_id = ?", (video_id,)
        ).fetchone()[0]
    assert count == 3


def test_index_video_records_failure_status_for_corrupt_file(
    tmp_path, fake_models, db_path
):
    siglip, tagger = fake_models
    bad = tmp_path / "not-a-video.mkv"
    bad.write_bytes(b"definitely not a media file")

    try:
        pipeline.index_video(db_path, str(bad), siglip, tagger, mode="fast")
    except Exception:
        pass

    # Whatever the failure mode, the queue worker must not leave a completed
    # row with scenes for a file it could not decode.
    with schema.get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT status FROM videos WHERE filepath = ?", (str(bad),)
        ).fetchone()
    assert row is None or row[0] != "completed"
