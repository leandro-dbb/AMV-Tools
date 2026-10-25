"""Apple Silicon full-pipeline optimisations (chantier 2), exercised on any
CPU box: VideoToolbox is replaced by libx264, the software decode profile is
forced, export_scene is faked. See docs/reports/chantier-2-pipeline-apple-silicon.md.
"""
from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

import pytest

from backend.indexing import cuts, proxy_source, single_pass
from tests.conftest import ffmpeg_exe, make_solid_clip


@pytest.fixture
def software_decode(monkeypatch):
    monkeypatch.setenv("AMV_DECODE", "sw")
    monkeypatch.setattr(cuts, "_DECODE_CHOICE", {})


@pytest.fixture
def x264_proxy_source(monkeypatch):
    """The recording branch with libx264 instead of VideoToolbox."""
    def output_args(tmp_path):
        return ["-c:v", "libx264", "-preset", "ultrafast", "-b:v", proxy_source.BITRATE,
                "-an", "-sn", "-f", "mp4", "-y", str(tmp_path)]
    monkeypatch.setattr(proxy_source, "output_args", output_args)


def _probe(path: Path) -> tuple[int, int, str]:
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        n = sum(1 for _ in c.decode(s))
        return s.codec_context.height, n, str(s.average_rate)


def test_single_pass_records_the_proxy_source(tmp_path, data_dir, software_decode, x264_proxy_source):
    clip = make_solid_clip(tmp_path / "cuts.mp4", seconds_per_color=1.0, size="640x360")
    plain, _ = single_pass.detect_cuts_with_medians(str(clip), threshold=0.30)
    assert proxy_source.existing(str(clip)) is None
    scenes, medians = single_pass.detect_cuts_with_medians(str(clip), threshold=0.30, proxy_source=True)
    # the extra branch changes nothing for the cut detector
    assert scenes == plain and len(medians) == len(scenes) == 3
    rec = proxy_source.existing(str(clip))
    assert rec is not None and rec.parent == data_dir / "proxies" / "sources"
    assert not rec.with_suffix(".part").exists()
    height, frames, rate = _probe(rec)
    assert height == proxy_source.HEIGHT and rate == str(proxy_source.FPS)
    assert abs(frames - 3 * proxy_source.FPS) <= 1


def test_failed_pass_keeps_no_partial_source(tmp_path, data_dir, software_decode, monkeypatch):
    clip = make_solid_clip(tmp_path / "cuts.mp4")
    monkeypatch.setattr(proxy_source, "output_args",
                        lambda tmp: ["-c:v", "no_such_encoder", "-f", "mp4", "-y", str(tmp)])
    with pytest.raises(Exception):
        single_pass.detect_cuts_with_medians(str(clip), threshold=0.30, proxy_source=True)
    assert proxy_source.existing(str(clip)) is None
    assert not proxy_source.path_for(str(clip)).with_suffix(".part").exists()


def test_proxy_source_key_follows_the_file(tmp_path, data_dir):
    f = tmp_path / "ep.mkv"
    f.write_bytes(b"a" * 10)
    first = proxy_source.path_for(str(f))
    os.utime(f, ns=(1, 1))
    assert proxy_source.path_for(str(f)) != first   # replaced in place
    p = proxy_source.path_for(str(f))
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    p.with_suffix(".part").write_bytes(b"x")
    assert proxy_source.existing(str(f)) == p
    proxy_source.discard_all()
    assert not p.exists() and not p.with_suffix(".part").exists()


def test_proxy_source_only_for_videotoolbox(monkeypatch):
    from backend.export import ffmpeg as ff
    monkeypatch.setattr(ff, "_hw_h264_encoder", lambda: "h264_nvenc")
    assert not proxy_source.wanted()
    monkeypatch.setattr(ff, "_hw_h264_encoder", lambda: "h264_videotoolbox")
    assert proxy_source.wanted()
    assert "-realtime" not in proxy_source.output_args(Path("x.part"))


def test_parallel_derush_export_reserves_distinct_names(tmp_path, monkeypatch):
    from backend.api import derush

    events: list[dict] = []

    class _State:
        def publish(self, ev):
            events.append(ev)

    made: list[str] = []
    lock = threading.Lock()

    def fake_export(src, s, e, out, **kw):
        with lock:
            made.append(out)
        Path(out).write_bytes(b"clip")

    src = tmp_path / "ep.mkv"
    src.write_bytes(b"x")
    monkeypatch.setattr(derush, "get_state", lambda: _State())
    monkeypatch.setattr(derush, "export_scene", fake_export)
    monkeypatch.setattr(derush, "_EXPORT_WORKERS", 3)
    items = [{"item_id": i, "folder_name": None, "custom_name": "same", "display_name": "ep",
              "scene_index": i, "filepath": str(src), "start_ms": 0, "end_ms": 1000} for i in range(5)]
    items.append({**items[0], "item_id": 9, "filepath": str(tmp_path / "missing.mkv")})
    derush._export_running = True
    derush._export_worker(items, {"output_folder": str(tmp_path / "out"), "codec": "libx264", "crf": 18,
                                  "audio": "mute", "resolution": "source"})
    assert len(made) == 5 and len(set(made)) == 5
    final = events[-1]
    assert final["done"] and final["message"] == "5 exported, 1 failed" and not final["ok"]
    assert derush._export_running is False
    assert sorted(e["current"] for e in events[:-1]) == list(range(1, 7))


def test_export_seek_without_preroll_is_frame_exact(tmp_path):
    """The macOS export drops the 10-s pre-roll: input seeking alone lands on
    the same frames (checked here on a synthetic GOP-heavy clip)."""
    src = tmp_path / "count.mp4"
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc2=s=320x180:r=25:d=30", "-c:v", "libx264", "-g", "250",
                    "-preset", "ultrafast", str(src)], check=True)

    def md5s(fast, exact):
        out = subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-ss", f"{fast}", "-i", str(src),
                              "-ss", f"{exact}", "-t", "1", "-map", "0:v:0", "-f", "framemd5", "-"],
                             capture_output=True, text=True, check=True).stdout
        return [ln.split(",")[-1].strip() for ln in out.splitlines() if ln and not ln.startswith("#")]

    for start in (3.0, 12.36, 21.0):
        assert md5s(max(0.0, start - 10.0), start - max(0.0, start - 10.0)) == md5s(start, 0.0)


def test_parallel_exports_only_for_videotoolbox(monkeypatch):
    from backend.api import derush

    monkeypatch.setattr(derush, "_EXPORT_WORKERS", 3)
    assert derush._parallel_exports("auto") == 3
    assert derush._parallel_exports("h264_videotoolbox") == 3
    for codec in ("libx265", "prores_ks", "libsvtav1", "libx264"):
        assert derush._parallel_exports(codec) == 1
