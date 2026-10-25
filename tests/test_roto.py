"""Mask Mode 0.5: automatic roto pipeline and windowed SAM 2 tracking.

Models are faked: these tests pin the logic (target selection, fusion,
re-seeding, window hand-over), not the networks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.models import auto_roto
from backend.models.auto_roto import fuse, match_targets, select_targets


def _blob(h, w, x0, y0, x1, y1, val=1.0):
    a = np.zeros((h, w), np.float32)
    a[y0:y1, x0:x1] = val
    return a


def test_select_main_prefers_big_central_character():
    a = _blob(100, 160, 60, 20, 100, 90)          # central, 40x70
    a += _blob(100, 160, 0, 10, 45, 95)           # bigger but on the frame edge, 45x85
    a += _blob(100, 160, 140, 5, 145, 10)         # speck
    main = select_targets(a, "main")
    assert main[50, 80] and not main[50, 20]
    both = select_targets(a, "all")
    assert both[50, 80] and both[50, 20] and not both[7, 142]


def test_select_targets_empty_frame():
    assert not select_targets(np.zeros((40, 40), np.float32), "all").any()


def test_match_targets_keeps_only_overlapping_components():
    det = _blob(60, 100, 10, 10, 30, 50) + _blob(60, 100, 60, 10, 90, 50)
    ref = np.zeros((60, 100), bool)
    ref[20:30, 15:25] = True
    m = match_targets(det, ref)
    assert m[30, 20] and not m[30, 70]


def test_fuse_drops_detections_outside_the_track_and_fills_blur():
    h, w = 120, 200
    track = _blob(h, w, 40, 20, 100, 110)
    det = _blob(h, w, 42, 20, 98, 60, 0.9)           # detector misses the blurred lower half
    det += _blob(h, w, 150, 20, 190, 100)            # a house the detector also kept
    out = fuse(det, track)
    assert out[100:, 150:].max() == 0                # house gone
    assert out[90, 70] > 0.9                          # blurred part filled by the tracker core
    assert abs(out[21, 70] - 0.9) < 1e-5              # detector alpha kept on the edge band


class FakeSAM2:
    """track_keyframes returns, on every frame, the mask of the keyframe
    closest in time — enough to exercise seeding and re-seeding."""

    window_frames = 96

    def __init__(self):
        self.calls = []

    def track_keyframes(self, keyframes, *, frames_dir, frame_paths, frame_size, soft=True):
        self.calls.append(sorted(k["frame_idx"] for k in keyframes))
        n = len(frame_paths)
        w, h = frame_size
        out = np.zeros((n, h, w), np.float32)
        for t in range(n):
            k = min(keyframes, key=lambda kf: abs(kf["frame_idx"] - t))
            out[t] = np.asarray(k["mask"], np.float32)
        return out

    def release_state(self):
        pass


def _frames(tmp_path: Path, n: int, w=64, h=40):
    paths = []
    for i in range(n):
        p = tmp_path / f"{i:05d}.png"
        Image.new("RGB", (w, h), (10 * i % 255, 0, 0)).save(p)
        paths.append(p)
    return paths


def test_auto_track_seeds_and_reseeds_where_the_track_drifts(tmp_path):
    n, h, w = 8, 40, 64
    paths = _frames(tmp_path, n, w, h)
    det = np.zeros((n, h, w), np.float32)
    for t in range(n):  # the character walks right; the fake tracker cannot follow
        x = 4 + 3 * t
        det[t, 10:35, x:x + 16] = 1.0
    sam = FakeSAM2()
    out = auto_track(paths, det, sam, tmp_path, (w, h))
    assert len(sam.calls) >= 2                       # at least one re-seed
    assert out[-1, 20, 4 + 3 * (n - 1) + 8] > 0.5    # last frame follows the character


def auto_track(paths, det, sam, frames_dir, size, **kw):
    return auto_roto.auto_track(paths, det, sam, frames_dir=frames_dir, frame_size=size, **kw)


def test_auto_track_user_correction_is_a_keyframe(tmp_path):
    n, h, w = 6, 30, 40
    paths = _frames(tmp_path, n, w, h)
    det = np.zeros((n, h, w), np.float32)
    det[:, 5:25, 5:20] = 1.0
    fix = np.zeros((h, w), bool)
    fix[5:25, 5:30] = True
    sam = FakeSAM2()
    auto_track(paths, det, sam, tmp_path, (w, h), extra_keyframes=[{"frame_idx": 4, "mask": fix}])
    assert 4 in sam.calls[0]


def test_auto_track_nothing_detected_returns_empty(tmp_path):
    paths = _frames(tmp_path, 4)
    det = np.zeros((4, 40, 64), np.float32)
    out = auto_track(paths, det, FakeSAM2(), tmp_path, (64, 40))
    assert out.shape == (4, 40, 64) and out.max() == 0


def test_resolve_mask_variant_by_tier():
    class St:
        def __init__(self, platform, tier):
            self.sel = {"platform": platform, "effective_tier": tier}

        def model_selection(self):
            return self.sel

    assert auto_roto.resolve_mask_variant(St("nvidia", "24gb"), "detector", "auto") == "general"
    assert auto_roto.resolve_mask_variant(St("nvidia", "24gb"), "detector", "anime") == "anime"
    assert auto_roto.resolve_mask_variant(St("cpu", "fast"), "detector", "auto") == "anime"
    assert auto_roto.mask_models(St("cpu", "fast"))["tracker"] is None
    # CPU has no Auto tracker, but the Manual engine still needs a SAM 2 variant
    assert auto_roto.resolve_mask_variant(St("cpu", "fast"), "tracker", "auto") == "base_plus"
    assert auto_roto.resolve_mask_variant(St("nvidia", "8gb"), "tracker", "bogus") in ("tiny", "base_plus", "large")


def test_every_tier_has_mask_models_with_allowed_licenses():
    from backend.model_download import ALLOWED_LICENSES
    from backend.models import tiers
    for platform in tiers.PLATFORMS:
        for tid in tiers.tiers_for(platform):
            rows = auto_roto.describe_mask_models(platform, tid)
            assert "mask_detector" in rows, (platform, tid)
            for row in rows.values():
                assert row["license"] in ALLOWED_LICENSES


# ── windowed SAM 2 propagation ──────────────────────────────────────────────
class FakePredictor:
    """SAM 2 video predictor stand-in: records each window and outputs, for
    every frame of a window, logits > 0 where the window's prompts say so."""

    def __init__(self):
        self.windows = []

    def init_state(self, video_path, offload_video_to_cpu=False, **_):
        files = sorted(Path(video_path).glob("*.jpg"))
        st = {"n": len(files), "prompts": {}, "files": [f.name for f in files]}
        self.windows.append(st)
        return st

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        inference_state["prompts"][frame_idx] = ("mask", np.asarray(mask, bool))

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, points=None, labels=None, box=None):
        inference_state["prompts"][frame_idx] = ("points", None)

    def propagate_in_video(self, state, start_frame_idx=0, reverse=False):
        import torch
        for i in range(start_frame_idx, state["n"]):
            yield i, [1], torch.full((1, 1, 8, 10), 5.0)

    def reset_state(self, state):
        state["prompts"].clear()


def test_windowed_tracking_covers_every_frame_with_bounded_windows(tmp_path):
    from backend.models.sam2 import SAM2Model

    n = 30
    for i in range(n):
        Image.new("RGB", (10, 8)).save(tmp_path / f"{i:05d}.jpg")
    m = SAM2Model.__new__(SAM2Model)
    m.device = type("D", (), {"backend": "cpu"})()
    m._video_predictor = FakePredictor()
    m._state_cache = None
    out = np.zeros((n, 8, 10), np.float32)
    seen = []
    kfs = [{"frame_idx": 12, "positive": [(1, 1)], "negative": [], "box": None},
           {"frame_idx": 25, "mask": np.ones((8, 10), bool)}]
    m._track_windowed(kfs, tmp_path, n, out, soft=True, window=8,
                      progress_cb=lambda s, t: seen.append(s))
    assert (out > 0.5).all()                                   # every frame emitted
    sizes = [w["n"] for w in m._video_predictor.windows]
    assert max(sizes) <= 8 + 3 + 1                             # window + anchors + hand-over frame
    assert seen[-1] == n
    # a window without its own prompt still conditions on the user keyframes
    # (frames 28-29: 2 anchors + the hand-over frame + 2 frames)
    later = m._video_predictor.windows[2]
    assert later["n"] == 5


@pytest.mark.parametrize("value", [None, "main", "all", "bogus"])
def test_auto_target_setting(value, monkeypatch):
    from backend.api import segment

    class St:
        settings = {"models": {"mask_auto_target": "main"}}

    monkeypatch.setattr(segment, "get_state", lambda: St())
    got = segment._auto_target(value)
    assert got == (value if value in ("main", "all") else "main")


def test_click_mode_carries_the_character_across_a_cut(tmp_path, monkeypatch):
    n, h, w = 6, 30, 40
    paths = _frames(tmp_path, n, w, h)
    det = np.zeros((n, h, w), np.float32)
    det[:, 5:25, 5:20] = 1.0
    monkeypatch.setattr(auto_roto, "split_shots", lambda frames, n: [(0, 3), (3, 6)])
    out = auto_track(paths, det, FakeSAM2(), tmp_path, (w, h), mode="click",
                     frames_small=[np.zeros((4, 4, 3), np.uint8)] * n, clicks={1: ([(10, 15)], [])})
    assert (out[5] > 0.5).any()                      # second shot not left empty
