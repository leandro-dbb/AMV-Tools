"""Real CPU libx264 clip exports, validated with ffprobe."""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from backend.db import queries, schema
from backend.export.ffmpeg import export_scene


def _ffprobe(path):
    exe = shutil.which("ffprobe")
    if not exe:
        pytest.skip("ffprobe not installed")
    proc = subprocess.run(
        [
            exe, "-v", "error",
            "-show_entries", "format=duration:stream=codec_name,codec_type",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout)


def test_export_scene_libx264_duration(tmp_path, clip):
    out = tmp_path / "clip.mp4"
    export_scene(
        str(clip), 500, 1600, str(out),
        codec="libx264", crf=23, audio="mute", resolution="source",
    )
    assert out.exists() and out.stat().st_size > 0

    info = _ffprobe(out)
    duration = float(info["format"]["duration"])
    assert 0.8 <= duration <= 1.4
    video_streams = [s for s in info["streams"] if s["codec_type"] == "video"]
    assert video_streams and video_streams[0]["codec_name"] == "h264"


def test_export_endpoint_writes_clip(tmp_path, client, db_path, clip, state):
    with schema.get_conn(db_path) as conn:
        vid = queries.upsert_video(
            conn, str(clip), "Episode", 3000, 25.0, "320x240", "test", status="completed"
        )
        sid = queries.insert_scene(conn, vid, 0, 200, 1200, None, b"")

    output_dir = tmp_path / "exports"
    state.settings["export"]["output_folder"] = str(output_dir)

    res = client.post("/api/export", json={"scene_id": sid})
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True

    out = body["output"]
    assert out.endswith(".mp4")
    assert (output_dir / out.split("/")[-1]).exists()

    info = _ffprobe(out)
    assert 0.7 <= float(info["format"]["duration"]) <= 1.5


def test_batch_export_reports_failures_per_scene(tmp_path, client, db_path, clip, state):
    output_dir = tmp_path / "batch"
    state.settings["export"]["output_folder"] = str(output_dir)
    with schema.get_conn(db_path) as conn:
        vid = queries.upsert_video(
            conn, str(clip), "Episode", 3000, 25.0, "320x240", "test", status="completed"
        )
        good = queries.insert_scene(conn, vid, 0, 0, 800, None, b"")
        missing = queries.upsert_video(
            conn, str(tmp_path / "missing.mp4"), "Missing", 3000, 25.0,
            "320x240", "test", status="completed",
        )
        bad = queries.insert_scene(conn, missing, 0, 0, 800, None, b"")

    res = client.post("/api/export/batch", json={"scene_ids": [good, bad]})
    body = res.json()
    assert [e["scene_id"] for e in body["exported"]] == [good]
    assert [f["scene_id"] for f in body["failed"]] == [bad]
