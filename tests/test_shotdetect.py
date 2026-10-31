"""Cut detection: every frame, context, flash rejection; optional TransNetV2.

The real-footage tests score the indexing paths against frame-accurate
annotations (tests/data/cuts_gt: 7 extracts, made frame by frame and checked
on contact sheets, scored at the exact frame on the app's cfr frame grid by
tests/eval/cuts_eval.py); they run when the extracts are present
(``AMV_CUTS_MEDIA``, ``os.pathsep``-separated folders, default
~/dev/amv-testdata/Séries test and ~/dev/amv-testdata/cuts-new)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from backend.indexing import cuts, cutnet, shotdetect

from tests.eval import cuts_eval

_HOME = Path.home() / "dev/amv-testdata"
MEDIA = [Path(p) for p in os.environ.get(
    "AMV_CUTS_MEDIA", os.pathsep.join([str(_HOME / "Séries test"), str(_HOME / "cuts-new")])).split(os.pathsep)]
GTS = cuts_eval.ground_truths()
# hand-made rule alone (no network), CPU ffmpeg path, measured on
# 2026-10-31 (docs/reports/coupes.md): recall, precision, minus a margin
FLOOR_RULE = {
    "dbs_beerus_01_extract": (0.80, 0.92), "black_clover_s02e04_extract": (0.85, 0.88),
    "dbs_beerus_01_b": (0.95, 0.99), "black_clover_s02e04_c": (0.82, 0.99),
    "dbs_beerus_02_a": (0.84, 0.94), "dbs_beerus_02_b": (0.57, 0.90), "black_clover_s02e04_b": (0.91, 0.99),
}
# with the TransNetV2 network (weights downloaded), same path
FLOOR_NET = {
    "dbs_beerus_01_extract": (0.90, 0.90), "black_clover_s02e04_extract": (0.90, 0.87),
    "dbs_beerus_01_b": (0.97, 0.99), "black_clover_s02e04_c": (0.91, 0.98),
    "dbs_beerus_02_a": (0.97, 0.93), "dbs_beerus_02_b": (0.79, 0.90), "black_clover_s02e04_b": (0.95, 0.99),
}


def _measure(stem: str, decode: str, monkeypatch) -> dict:
    gtd = GTS[stem]
    path = cuts_eval.find_media(gtd, MEDIA)
    if path is None:
        pytest.skip(f"real extract not present: {gtd['video']}")
    monkeypatch.setattr(cuts, "_ffmpeg_hw_profile", lambda: None)
    cuts_, dc = cuts_eval.gt(gtd, path)
    return cuts_eval.score(cuts_, dc, cuts_eval.run_path(decode, path))


@pytest.mark.parametrize("stem", sorted(FLOOR_RULE))
@pytest.mark.parametrize("decode", ["sw", "pyav", "single_pass"])
def test_real_extracts_rule_meets_the_measured_floor(stem, decode, monkeypatch):
    if decode != "sw" and stem not in cuts_eval.ORIGINAL_STEMS:
        pytest.skip("other paths: measured on the two original extracts")
    monkeypatch.setenv("AMV_CUTNET", "0")
    r = _measure(stem, decode, monkeypatch)
    lo_r, lo_p = FLOOR_RULE[stem]
    assert r["recall"] >= lo_r and r["precision"] >= lo_p, r


@pytest.mark.slow
@pytest.mark.parametrize("stem", sorted(FLOOR_NET))
def test_real_extracts_with_the_network_meet_the_measured_floor(stem, monkeypatch):
    from backend.indexing import cutnet
    if cutnet.weights_path() is None:
        pytest.skip("TransNetV2 weights not downloaded")
    monkeypatch.setenv("AMV_CUTNET", "1")
    r = _measure(stem, "sw", monkeypatch)
    lo_r, lo_p = FLOOR_NET[stem]
    assert r["recall"] >= lo_r and r["precision"] >= lo_p, r


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
    assert shotdetect.detect(frames, net=None) == [30, 60]


def test_a_flash_or_impact_frame_is_not_a_cut():
    a = _shot(1, 40, dx=1)
    a[20] = np.full((90, 160), 250, np.uint8)            # one white frame
    a[30] = np.asarray(_shot(9, 1)[0])                   # one unrelated impact frame
    assert shotdetect.detect(a, net=None) == []


def test_pans_are_not_cuts_but_letterboxed_cuts_are_found():
    frames = _shot(4, 40, dx=2, letterbox=22) + _shot(5, 40, dx=2, letterbox=22)
    assert shotdetect.detect(frames, net=None) == [40]


def test_streaming_equals_batch():
    frames = _shot(1, 25) + _shot(2, 3) + _shot(3, 25)
    det = shotdetect.ShotDetector(net=None)
    got = []
    for f in frames:
        got += det.push(f)
    assert got + det.finish() == shotdetect.detect(frames, net=None) == [25, 28]


# ── the optional network ────────────────────────────────────────────────────
def test_without_weights_the_detector_keeps_the_hand_made_rule(monkeypatch):
    monkeypatch.setattr(cutnet, "weights_path", lambda: None)
    det = shotdetect.ShotDetector()
    assert det.wants_color is False and det.lookahead == shotdetect.LOOKAHEAD
    frames = _shot(1, 30, dx=1) + _shot(2, 30, dx=1)
    assert shotdetect.detect(frames) == [30]


def _random_net():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    return cutnet._build(torch).eval(), "cpu"


def test_streamed_network_equals_one_window_per_50_frames():
    torch = pytest.importorskip("torch")
    model, dev = _random_net()
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (87, cutnet.IN_H, cutnet.IN_W, 3), dtype=np.uint8)
    single, many = cutnet.predict(frames, (model, dev))
    assert len(single) == len(many) == 87
    # reference: pad 25 / 25 + up to a multiple of 50, windows of 100 every 50
    pad_end = 25 + 50 - (87 % 50)
    padded = np.concatenate([np.repeat(frames[:1], 25, 0), frames, np.repeat(frames[-1:], pad_end, 0)])
    ref = []
    with torch.inference_mode():
        for s in range(0, len(padded) - 99, 50):
            out, _ = model(torch.from_numpy(padded[None, s:s + 100]))
            ref += torch.sigmoid(out[0, 25:75, 0]).tolist()
    np.testing.assert_allclose(single, ref[:87], atol=1e-5)


def test_combined_detector_streaming_equals_batch_and_finds_a_plain_cut():
    model = _random_net()
    frames = _shot(1, 40, dx=1) + _shot(2, 40, dx=1)
    colors = [np.repeat(f[..., None], 3, axis=2) for f in frames]
    det = shotdetect.ShotDetector(net=model)
    assert det.wants_color and det.lookahead > shotdetect.LOOKAHEAD
    got = []
    for f, c in zip(frames, colors):
        got += det.push(f, c)
    got += det.finish()
    assert got == shotdetect.detect(frames, colors=colors, net=model)
    assert all(0 < t < len(frames) for t in got)
