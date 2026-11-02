"""Fixes for what the Windows 11 + RTX 3090 test left open
(docs/reports/windows-rtx3090.md, « Corrections »)."""
from __future__ import annotations

import pytest


# ── 7. exported clip names ──────────────────────────────────────────────────
@pytest.mark.parametrize("display, anime, episode", [
    ("[SubsPlease] Dragon Ball Super - Beerus - 01 (1080p) [B3FEF9B7]", "Dragon Ball Super - Beerus", "01"),
    ("Black Clover S02E04 1080p", "Black Clover", "S02E04"),
    ("[Erai-raws] Shingeki no Kyojin - The Final Season - 28v2 [1080p]",
     "Shingeki no Kyojin - The Final Season", "28"),
    ("Naruto Episode 12", "Naruto", "12"),
    ("Kimetsu no Yaiba 第3話", "Kimetsu no Yaiba", "第3話"),
    ("Mon film été", "Mon film été", ""),
])
def test_split_title(display, anime, episode):
    from backend.export.naming import split_title
    assert split_title(display) == (anime, episode)


def test_exported_clip_name_has_the_episode_once(tmp_path, monkeypatch):
    """The default template ``{anime}_{episode}_scene_{id}`` got the whole
    file name twice: ``…B3FEF9B7_SubsPlease_…B3FEF9B7_scene_468.mp4``."""
    from backend.api import export as api_export
    seen = {}
    monkeypatch.setattr(api_export, "export_scene", lambda src, s, e, target, **kw: seen.setdefault("t", target))
    scene = {"id": 468, "display_name": "[SubsPlease] Dragon Ball Super - Beerus - 01 (1080p) [B3FEF9B7]",
             "filepath": "x.mkv", "start_ms": 0, "end_ms": 1000}
    settings = {"naming_template": "{anime}_{episode}_scene_{scene_id}", "codec": "auto", "crf": 18,
                "audio": "copy", "resolution": "source"}
    out = api_export._export_one(scene, settings, tmp_path)
    from pathlib import Path
    assert Path(out).name == "Dragon_Ball_Super_-_Beerus_01_scene_468.mp4"
    assert Path(out).name.count("Beerus") == 1


def test_clip_name_without_episode_number_drops_the_empty_field():
    from backend.export.naming import clip_stem
    assert clip_stem("{anime}_{episode}_scene_{scene_id}", "Mon film", 3) == "Mon_film_scene_3"
    assert clip_stem("{episode}-{anime}-{scene_id}", "Mon film", 3) == "Mon_film-3"
    assert clip_stem("{anime}_{scene_id}_{tags}", "Mon film", 3, tags="alpha") == "Mon_film_3_alpha"
