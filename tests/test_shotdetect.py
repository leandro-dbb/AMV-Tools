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
# with the TransNetV2 network (weights downloaded), same path, measured the
# same day on the CPU (the NVIDIA paths give the same, dbs_beerus_02_a: one
# more false cut)
FLOOR_NET = {
    "dbs_beerus_01_extract": (0.89, 0.93), "black_clover_s02e04_extract": (0.88, 0.87),
    "dbs_beerus_01_b": (0.97, 0.99), "black_clover_s02e04_c": (0.91, 0.99),
    "dbs_beerus_02_a": (0.95, 0.93), "dbs_beerus_02_b": (0.77, 0.90), "black_clover_s02e04_b": (0.95, 0.99),
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
    monkeypatch.setenv("AMV_CUTNET", "1")
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


# ── performance paths: same decisions ───────────────────────────────────────
def _colour_clip():
    rng = np.random.default_rng(3)
    frames, colors = [], []
    for k, n in enumerate((37, 5, 60, 2, 44, 70)):
        for f in _shot(10 + k, n, dx=2):
            frames.append(f)
            tint = rng.integers(0, 40, 3)
            colors.append(np.clip(f[..., None].astype(int) + tint, 0, 255).astype(np.uint8))
    return frames, colors


def test_batched_decision_equals_every_frame_scored_in_full():
    """The decision scores frames in blocks and settles hopeless ones by a
    bound: same cuts as scoring every frame with all its measures."""
    model = _random_net()
    frames, colors = _colour_clip()
    got = shotdetect.detect(frames, colors=colors, net=model)
    x = np.stack([shotdetect.net_input(f, c) for f, c in zip(frames, colors)])
    single, many = cutnet.predict(x, model)
    small = {i: shotdetect._small(f) for i, f in enumerate(frames)}
    win = shotdetect._Window(small)
    win.n = len(frames)
    d = [0.0] + [shotdetect.motion_diff(small[i - 1], small[i]) for i in range(1, len(frames))]
    hists = [shotdetect.color_hist(f) for f in x]
    chd = [0.0] + [0.5 * float(np.abs(hists[i] - hists[i - 1]).sum()) for i in range(1, len(frames))]
    m = shotdetect.NET_MODELS["color"]
    p = [0.0] + [1.0 / (1.0 + np.exp(-(float(np.dot(m["w"], shotdetect.net_features(
        win, d, single, many, chd, t, True))) + m["b"]))) for t in range(1, len(frames))] + [0.0]
    want = [t for t in range(1, len(frames)) if p[t] > m["thr"] and p[t] >= p[t - 1] and p[t] > p[t + 1]]
    assert got == want


def test_frames_handed_over_in_blocks_give_the_same_cuts():
    """NVIDIA: the network's frames and colour histograms come per batch
    (``push_net``), the gray planes per frame."""
    model = _random_net()
    frames, colors = _colour_clip()
    want = shotdetect.detect(frames, colors=colors, net=model)
    x = np.stack([shotdetect.net_input(f, c) for f, c in zip(frames, colors)])
    counts = np.stack([np.bincount(((q[..., 0] << 4) | (q[..., 1] << 2) | q[..., 2]).ravel(), minlength=64)
                       for q in (x.astype(np.int64) >> 6)]).astype(np.int32)
    det = shotdetect.ShotDetector(net=model)
    got = []
    for i, f in enumerate(frames):
        if i % 32 == 0:
            det.push_net(x[i:i + 32], counts[i:i + 32])
        got += det.push(f)
    assert got + det.finish() == want


def test_asynchronous_network_gives_the_same_probabilities():
    model = _random_net()
    rng = np.random.default_rng(1)
    frames = np.repeat(rng.integers(0, 256, (30, cutnet.IN_H, cutnet.IN_W, 3), dtype=np.uint8), 9, axis=0)
    a = cutnet.predict(frames, model)
    st = cutnet.Stream(model, asynchronous=True)
    for i in range(0, len(frames), 7):
        st.push_many(frames[i:i + 7])
    st.finish()
    assert np.array_equal(a[0], st.single) and np.array_equal(a[1], st.many)


def test_2d_formulation_matches_the_network():
    torch = pytest.importorskip("torch")
    model, _ = _random_net()
    rng = np.random.default_rng(2)
    x = torch.from_numpy(rng.integers(0, 256, (2, 100, cutnet.IN_H, cutnet.IN_W, 3), dtype=np.uint8))
    with torch.inference_mode():
        a, b = model(x)
        a2, b2 = cutnet.forward_2d(model, x)
    np.testing.assert_allclose(torch.sigmoid(a2).numpy(), torch.sigmoid(a).numpy(), atol=1e-5)
    np.testing.assert_allclose(torch.sigmoid(b2).numpy(), torch.sigmoid(b).numpy(), atol=1e-5)


def test_batched_colour_frames_match_the_per_frame_ones():
    """``onepass.SmallRgbGpu`` (run here on the CPU) against ``_small_rgb``
    and ``color_hist``: area weights as matrices, same conversion."""
    torch = pytest.importorskip("torch")
    from backend.indexing import onepass
    rng = np.random.default_rng(4)
    n = 6
    st = np.zeros((n, onepass.STACK_H * 3 // 2, onepass.FRAME_W), dtype=np.uint8)
    st[:, :onepass.STACK_H] = rng.integers(16, 236, (n, onepass.STACK_H, onepass.FRAME_W))
    st[:, onepass.STACK_H:] = rng.integers(16, 240, (n, onepass.STACK_H // 2, onepass.FRAME_W))
    st[:, onepass.FRAME_H:onepass.STACK_H, :onepass.GRAY_W] = np.repeat(rng.integers(16, 236, (n, 18, 32)), 5, 1).repeat(5, 2)
    make = onepass.SmallRgbGpu(torch, "cpu", "bt709", False, onepass._TV_TO_PC)
    rgb, counts = make(torch.from_numpy(st))
    for i in range(n):
        gray = onepass._gray_of(st[i].tobytes(), onepass._TV_TO_PC)
        ref = onepass._small_rgb(st[i].reshape(-1), gray, "bt709", False)
        diff = np.abs(rgb[i].numpy().astype(int) - ref.astype(int))
        assert diff.max() <= 1 and (diff > 0).mean() < 0.01
        np.testing.assert_allclose(counts[i].numpy() / counts[i].sum().item(), shotdetect.color_hist(rgb[i].numpy()))


def test_area_matrix_is_cv2_inter_area():
    cv2 = pytest.importorskip("cv2")
    from backend.indexing import onepass
    img = np.random.default_rng(5).random((90, 160)).astype(np.float32) * 255
    m = onepass.area_matrix(90, 27).astype(np.float32) @ img @ onepass.area_matrix(160, 48).astype(np.float32).T
    np.testing.assert_allclose(m, cv2.resize(img, (48, 27), interpolation=cv2.INTER_AREA), atol=1e-3)


# ── when the network runs by itself ─────────────────────────────────────────
@pytest.mark.parametrize("measured, setting, cuda, want", [
    ("auto", False, False, True), ("manual", False, True, False), ("manual", True, False, True),
    (None, False, True, True), (None, False, False, False), (None, True, False, True),
])
def test_network_runs_by_itself_only_where_it_is_cheap(monkeypatch, measured, setting, cuda, want):
    from types import SimpleNamespace

    from backend import autotune, state
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("AMV_CUTNET", raising=False)
    monkeypatch.setattr(cutnet, "weights_path", lambda: "/w.pth")
    monkeypatch.setattr(autotune, "value", lambda name, default: measured if name == "cutnet" else default)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(state, "get_state", lambda: SimpleNamespace(settings={"indexing": {"precise_cuts": setting}}))
    assert cutnet.enabled() is want
    monkeypatch.setenv("AMV_CUTNET", "0")
    assert cutnet.enabled() is False
