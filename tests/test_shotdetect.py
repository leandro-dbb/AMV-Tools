"""Cut detection (v06 fix): every frame, context, flash rejection.

The real-footage test scores the indexing paths against frame-accurate
annotations (tests/data/cuts_gt, made frame by frame and checked on contact
sheets); it runs when the extracts are present (``AMV_CUTS_MEDIA`` or
~/dev/amv-testdata/Séries test)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from backend.indexing import cuts, shotdetect, single_pass

GT_DIR = Path(__file__).parent / "data" / "cuts_gt"
MEDIA = Path(os.environ.get("AMV_CUTS_MEDIA", Path.home() / "dev/amv-testdata/Séries test"))
EXTRACTS = {
    "dbs_beerus_01_extract.json": "DBS Beerus 01 - extrait.mkv",
    "black_clover_s02e04_extract.json": "Black Clover S02E04 - extrait.mkv",
}
# measured at the fix (docs/reports/v06-fix.md); the old detector: recall
# 0.69 / 0.38, precision 0.69 / 0.67
FLOOR = {"dbs_beerus_01_extract.json": (0.80, 0.91), "black_clover_s02e04_extract.json": (0.85, 0.88)}


def _score(gt: dict, det: list[int]) -> tuple[float, float, list, list]:
    cuts_ = [c["frame"] for c in gt["hard_cuts"]]
    dont_care: set[int] = set()
    for a in gt["ambiguous"]:
        dont_care.update(range(a["frame"], a.get("end", a["frame"]) + 1))
    for x in gt["dissolves"]:
        dont_care.update(range(x["start"] - 1, x["end"] + 2))
    det_s, cs = set(det), set(cuts_)
    exact = [c for c in cuts_ if c in det_s]
    wrong = [d for d in det_s if d not in cs and d not in dont_care]
    return len(exact) / len(cuts_), len(exact) / max(1, len(exact) + len(wrong)), \
        [c for c in cuts_ if c not in det_s], wrong


def _frames(gt_name: str, path: Path, scenes) -> list[int]:
    fps = cuts.probe_video(str(path))["fps"]
    return [round(s * fps / 1000) for s, _e in scenes[1:]]


@pytest.mark.parametrize("gt_name", sorted(EXTRACTS))
@pytest.mark.parametrize("decode", ["sw", "pyav", "single_pass"])
def test_real_extracts_meet_the_measured_floor(gt_name, decode, monkeypatch):
    path = MEDIA / EXTRACTS[gt_name]
    if not path.exists():
        pytest.skip(f"real extract not present: {path}")
    gt = json.loads((GT_DIR / gt_name).read_text())
    monkeypatch.setattr(cuts, "_ffmpeg_hw_profile", lambda: None)
    if decode == "single_pass":
        monkeypatch.setenv("AMV_DECODE", "sw")
        scenes, medians = single_pass.detect_cuts_with_medians(str(path), threshold=0.40)
        assert len(medians) == len(scenes)
    else:
        monkeypatch.setenv("AMV_DECODE", decode)
        scenes = cuts.detect_cuts(str(path), threshold=0.40)
    recall, precision, missed, wrong = _score(gt, _frames(gt_name, path, scenes))
    lo_r, lo_p = FLOOR[gt_name]
    assert recall >= lo_r and precision >= lo_p, (recall, precision, missed, wrong)


# ── synthetic: what the rules are for ───────────────────────────────────────
def _shot(seed: int, n: int, *, dx: int = 0, letterbox: int = 0) -> list[np.ndarray]:
    """``n`` frames of a textured picture panning ``dx`` px per frame."""
    rng = np.random.default_rng(seed)
    base = np.full((90, 400), rng.integers(30, 220), np.float32)   # flat colour areas, like cel animation
    for _ in range(25):
        y, x = rng.integers(0, 80), rng.integers(0, 380)
        base[y:y + rng.integers(8, 40), x:x + rng.integers(10, 60)] = rng.integers(0, 256)
    out = []
    for i in range(n):
        f = base[:, i * dx:i * dx + 160].astype(np.uint8).copy()
        if letterbox:
            f[:letterbox] = 0
            f[-letterbox:] = 0
        out.append(f)
    return out


def test_cut_lands_on_the_first_frame_of_the_new_shot():
    frames = _shot(1, 30, dx=1) + _shot(2, 30, dx=1) + _shot(3, 30)
    assert shotdetect.detect(frames) == [30, 60]


def test_a_flash_or_impact_frame_is_not_a_cut():
    a = _shot(1, 40, dx=1)
    a[20] = np.full((90, 160), 250, np.uint8)            # one white frame
    a[30] = np.asarray(_shot(9, 1)[0])                   # one unrelated impact frame
    assert shotdetect.detect(a) == []


def test_pans_are_not_cuts_but_letterboxed_cuts_are_found():
    frames = _shot(4, 40, dx=2, letterbox=22) + _shot(5, 40, dx=2, letterbox=22)
    assert shotdetect.detect(frames) == [40]


def test_streaming_equals_batch():
    frames = _shot(1, 25) + _shot(2, 3) + _shot(3, 25)
    det = shotdetect.ShotDetector()
    got = []
    for f in frames:
        got += det.push(f)
    assert got + det.finish() == shotdetect.detect(frames) == [25, 28]
