"""Drops on real songs (v06 fix), against hand annotations made on
spectrograms / bass and loudness curves (tests/data/music_drops_gt.json,
±0.25 s). Runs when the songs are present (``AMV_MUSIC_MEDIA`` or
~/dev/amv-testdata/music; Kevin MacLeod, CC-BY)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from backend.music.analysis import analyze_audio

GT = json.loads((Path(__file__).parent / "data" / "music_drops_gt.json").read_text())
MEDIA = Path(os.environ.get("AMV_MUSIC_MEDIA", Path.home() / "dev/amv-testdata/music"))
# measured at the fix: 8 of 14 annotated drops (2 before), no drop outside
# the annotations (before: Voltaic 160 s; the bar phase now follows the
# drops, which moved that section boundary)
MIN_FOUND, MAX_EXTRA = 8, 0


def test_real_songs_drops():
    tracks = {name: MEDIA / name for name in GT["tracks"]}
    missing = [n for n, p in tracks.items() if not p.exists()]
    if missing:
        pytest.skip(f"real songs not present: {missing}")
    tol = GT["tolerance_s"]
    found = extra = 0
    report = {}
    for name, path in tracks.items():
        drops = [d / 1000 for d in analyze_audio(str(path))["drops_ms"]]
        want = [d["t"] for d in GT["tracks"][name]["drops"]]
        found += sum(any(abs(w - d) <= tol for d in drops) for w in want)
        extra += sum(not any(abs(w - d) <= tol for w in want) for d in drops)
        report[name] = (drops, want)
    assert found >= MIN_FOUND and extra <= MAX_EXTRA, report
    # the song without any drop (pauses where the same groove resumes) stays clean
    assert report["Monkeys_Spinning_Monkeys.mp3"][0] == []
