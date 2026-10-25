"""Hard-cut detection on synthetic footage with known cut points."""
from __future__ import annotations

from backend.indexing import cuts


def test_detects_expected_number_of_cuts(clip):
    scenes = cuts.detect_cuts(str(clip), threshold=0.30)

    # Three solid colours -> three scenes. Allow tiny boundary jitter but not
    # an extra/missing scene.
    assert len(scenes) == 3
    assert scenes[0][0] == 0
    assert scenes[-1][1] >= 2900
    for start, end in scenes:
        assert end > start
    # Boundaries are roughly one second apart.
    assert abs(scenes[0][1] - 1000) < 250
    assert abs(scenes[1][1] - 2000) < 250


def test_probe_video_reports_duration_and_resolution(clip):
    info = cuts.probe_video(str(clip))
    assert 2800 <= info["duration_ms"] <= 3400
    assert info["resolution"] == "320x240"
    assert info["fps"] > 0
