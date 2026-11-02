"""Fixes from the Linux installer test (AppImage / .deb, Debian 12 + RTX 3090)."""
from backend.api.derush import _clip_stem, _sanitize


def test_derush_clip_names_drop_release_tags():
    item = {"custom_name": None, "scene_index": 5,
            "display_name": "[Groupe] Dragon Ball Super - Beerus - 01 (1080p) [ABCD1234]"}
    assert _sanitize(_clip_stem(item)) == "Dragon Ball Super - Beerus 01_scene0005"


def test_derush_clip_names_keep_custom_and_unsplittable_names():
    assert _clip_stem({"custom_name": "Mon plan", "scene_index": 1, "display_name": "x"}) == "Mon plan"
    item = {"custom_name": None, "scene_index": 12, "display_name": "Mon film d'été"}
    assert _sanitize(_clip_stem(item)) == "Mon film d_été_scene0012"
