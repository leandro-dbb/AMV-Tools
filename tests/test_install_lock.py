"""uv.lock must give every first launch an installable torch.

The installed app bootstraps its venv with `uv run --no-dev` (+ `--extra cpu`
on Linux, app/electron/syncExtra.ts) and uv picks the newest Python allowed by
requires-python when the system has none (3.13 today). 0.5.0-beta.3 died on
the first launch on Debian 12: the Linux base deps resolve torch 2.4.1, which
has no cp313 wheel.
"""
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# (platform marker, uv extra used by the bootstrap, wheel platform tag)
BOOTSTRAPS = [
    ("linux", "cpu", "manylinux_2_28_x86_64"),
    ("win32", None, "win_amd64"),
    ("darwin", None, "macosx_11_0_arm64"),
]


def _exported_torch(extra, platform):
    cmd = ["uv", "export", "--frozen", "--no-dev", "--no-hashes", "--no-emit-project"]
    if extra:
        cmd += ["--extra", extra]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        m = re.match(r"torch==(\S+)\s*;\s*(.*)$", line)
        if m and f"sys_platform == '{platform}'" in m.group(2):
            return m.group(1)
    return None


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
@pytest.mark.parametrize("platform,extra,tag", BOOTSTRAPS)
def test_bootstrap_torch_has_wheels_for_every_allowed_python(platform, extra, tag):
    version = _exported_torch(extra, platform)
    assert version, f"no torch for {platform}"
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    wheels = [
        w["url"].rsplit("/", 1)[1]
        for p in lock["package"]
        if p["name"] == "torch" and p["version"] == version
        for w in p.get("wheels", [])
    ]
    for py in ("cp312", "cp313"):
        assert any(f"-{py}-{py}-" in w and tag in w for w in wheels), (
            f"torch {version} ({platform}, extra={extra}) has no {py} {tag} wheel"
        )
