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


_FIRST_LAUNCH = r'''
import sys, time
from fastapi.testclient import TestClient
from backend.server import make_app
with TestClient(make_app(), base_url="http://127.0.0.1:8731") as c:   # lifespan on
    for route in ("/api/status", "/api/setup/hardware", "/api/setup/status", "/api/settings"):
        assert c.get(route).status_code == 200, route
    time.sleep(2.0)                                                     # background threads
    print("TORCH-LOADED" if "torch" in sys.modules else "TORCH-NOT-LOADED")
'''


def test_first_launch_sidecar_does_not_import_torch(tmp_path):
    """Before the onboarding installs the GPU engine, nothing may import
    torch in the sidecar: the CPU build loaded then stays in the process and
    the freshly installed CUDA build is ignored until the next start (Linux,
    0.5.0-beta.3: "CPU (forced fallback)" right after installing CUDA 13)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = {**os.environ, "AMV_DATA_DIR": str(tmp_path), "HOME": str(tmp_path), "HF_HUB_OFFLINE": "1"}
    env.pop("AMV_TOKEN", None)
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, "-c", _FIRST_LAUNCH], cwd=root, env=env,
                         capture_output=True, text=True, timeout=120)
    assert "TORCH-NOT-LOADED" in out.stdout, out.stdout + out.stderr[-3000:]
