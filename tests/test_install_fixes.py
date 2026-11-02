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


def test_finish_output_names_drop_release_tags_and_keep_accents():
    from backend.finish.runner import _sanitize as finish_sanitize, _scene_stem
    stem = _scene_stem("[Groupe] Dragon Ball Super - Beerus - 01 (1080p) [ABCD1234]", "/x.mkv", 10)
    assert finish_sanitize(stem) == "Dragon_Ball_Super_-_Beerus_01_scene10"
    assert finish_sanitize(_scene_stem("", "/v/Pokémon été.mkv", 3)) == "Pokémon_été_scene3"


def test_roto_export_names_drop_release_tags():
    from backend.api.roto import _scene_name
    from backend.roto.export import _sanitize as roto_sanitize
    scene = {"filepath": "/v/[Groupe] Dragon Ball Super - Beerus - 01 (1080p) [ABCD1234].mkv", "scene_index": 9}
    assert roto_sanitize(_scene_name(scene)) == "Dragon_Ball_Super_-_Beerus_01_scene0009"
    assert roto_sanitize(_scene_name({"filepath": "/v/Pokémon été.mkv", "scene_index": 2})) == "Pokémon_été_scene0002"
