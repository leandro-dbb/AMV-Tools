"""Chunked single-pass decode (Apple Silicon, tour 3): the stitched result
must be exactly the sequential pass's — cuts, median frames and Search+
motion statistics. VideoToolbox is replaced by the software decoder (the
frames are bit-identical, checked on real H.264/HEVC episodes)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from backend.indexing import chunked, cuts, single_pass
from tests.conftest import ffmpeg_exe


def _clip(path: Path, seed: int) -> Path:
    """40 s at 25 fps: moving test pattern cut by solid-colour shots at
    random instants (some a few frames apart, around chunk boundaries) and a
    6-s static shot longer than a chunk."""
    rng = np.random.default_rng(seed)
    t, parts = 0.0, []
    colors = ["red", "blue", "green", "yellow", "white", "black", "purple", "orange"]
    while t < 40.0:
        d = float(rng.choice([0.12, 0.2, 0.36, 0.8, 1.5, 2.6]))
        if 20.0 <= t < 21.0:
            d = 6.0
        parts.append((t, d, colors[len(parts) % len(colors)] if len(parts) % 3 else None))
        t += d
    inputs, labels = [], []
    for i, (_t, d, c) in enumerate(parts):
        src = f"color=c={c}:s=320x180:d={d}:r=25" if c else f"testsrc2=s=320x180:d={d}:r=25"
        inputs += ["-f", "lavfi", "-i", src]
        labels.append(f"[{i}:v]")
    graph = "".join(labels) + f"concat=n={len(parts)}:v=1:a=0,format=yuv420p[v]"
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *inputs,
                    "-filter_complex", graph, "-map", "[v]", "-t", "40", "-c:v", "libx264", "-preset", "ultrafast",
                    "-g", "50", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture
def software_chunks(monkeypatch):
    monkeypatch.setenv("AMV_DECODE", "sw")
    monkeypatch.setattr(cuts, "_DECODE_CHOICE", {})
    monkeypatch.setattr(chunked, "_HWACCEL", [])
    monkeypatch.setattr(chunked, "CHUNK_S", 3.0)
    monkeypatch.setattr(chunked, "WORKERS", 3)


@pytest.mark.parametrize("seed", [0, 1])
def test_chunked_equals_sequential(tmp_path, software_chunks, seed):
    pytest.importorskip("cv2")
    from backend.searchplus.analysis import FrameStatsAccumulator

    clip = _clip(tmp_path / f"cuts{seed}.mp4", seed)
    a1 = FrameStatsAccumulator()
    s1, m1 = single_pass.detect_cuts_with_medians(str(clip), threshold=0.30, frame_sink=a1.sink)
    a2 = FrameStatsAccumulator().start_async()
    seen = {}
    s2, m2 = chunked.detect_cuts_with_medians(str(clip), threshold=0.30, motion=a2,
                                              on_median=lambda t, im: seen.__setitem__(t, im))
    a2.finish()
    assert len(s1) > 20
    assert s2 == s1
    assert sorted(m2) == sorted(m1) == sorted(seen)
    assert all(np.array_equal(np.asarray(m1[k]), np.asarray(m2[k])) for k in m1)
    for name in FrameStatsAccumulator._PER_SAMPLE + FrameStatsAccumulator._PER_PAIR:
        assert getattr(a2, name) == getattr(a1, name), name


def test_failed_chunk_raises_instead_of_hanging(tmp_path, software_chunks, monkeypatch):
    clip = _clip(tmp_path / "c.mp4", 3)
    real = chunked._command

    def broken(path, fps, ch, proxy_part, hw=True):
        cmd = real(path, fps, ch, proxy_part, hw)
        if ch.k == 2:
            cmd[cmd.index("-i") + 1] = str(tmp_path / "missing.mp4")
        return cmd

    monkeypatch.setattr(chunked, "_command", broken)
    with pytest.raises(RuntimeError, match="chunked decode failed"):
        chunked.detect_cuts_with_medians(str(clip), threshold=0.30)


def test_chunked_only_on_macos(monkeypatch, tmp_path):
    monkeypatch.setattr(chunked.sys, "platform", "linux")
    assert not chunked.supported(str(tmp_path / "x.mkv"))
