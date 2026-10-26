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
                # SAM 3: gated, fetched with the user's own approved token,
                # license shown in Settings (AMV Tools is not commercial)
                assert row["license"] in ALLOWED_LICENSES or (row.get("gated") and row["id"] == "facebook/sam3")


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



# ── 0.6: character selector, export edges ─────────────────────────────────
def test_gate_alpha_cuts_what_the_selector_does_not_call_a_character():
    from backend.models.auto_roto import gate_alpha
    base = _blob(100, 200, 20, 20, 60, 90) + _blob(100, 200, 140, 20, 190, 90)   # character + aura blob
    sel = _blob(100, 200, 22, 22, 58, 88)                                      # selector: only the character
    g = gate_alpha(base, sel)
    assert g[50, 40] == 1.0 and g[21, 21] > 0.9        # outline kept (selector under-segments)
    assert g[50, 170] == 0.0                            # far blob cut
    assert not gate_alpha(base, np.zeros_like(sel)).any()
    assert np.array_equal(gate_alpha(base, np.zeros_like(sel), keep_if_empty=True), base)


def test_apply_selector_gates_uint8_and_falls_back_on_failure():
    from backend.models.auto_roto import apply_selector
    det = np.stack([(_blob(60, 100, 5, 5, 30, 50) + _blob(60, 100, 70, 5, 95, 50)) * 255] * 3).astype(np.uint8)
    frames = [np.zeros((60, 100, 3), np.uint8)] * 3

    class Sel:
        def segment(self, rgb, prompt):
            return _blob(60, 100, 5, 5, 30, 50)

    out = apply_selector(det, frames, Sel())
    assert out.dtype == np.uint8 and out[1, 20, 15] == 255 and out[1, 20, 85] == 0

    class Broken:
        def segment(self, rgb, prompt):
            raise RuntimeError("CUDA out of memory")

    assert apply_selector(det, frames, Broken()) is det


def test_selector_in_gpu_tiers_only_and_listed_in_settings():
    assert all(c["selector"] is None for c in auto_roto.MASK_TIERS["mac"].values())
    assert auto_roto.MASK_TIERS["cpu"]["fast"]["selector"] is None
    assert auto_roto.MASK_TIERS["nvidia"]["8gb"]["selector"] == "sam3"
    rows = auto_roto.describe_mask_models("nvidia", "24gb")
    assert rows["mask_selector"]["id"] == "facebook/sam3"
    assert "mask_selector" not in auto_roto.describe_mask_models("mac", "mac16")


def test_select_characters_without_sam3_weights_keeps_the_detector(monkeypatch):
    from backend.api import segment
    det = np.zeros((2, 8, 8), np.uint8)

    class St:
        def model_selection(self):
            return {"platform": "nvidia", "effective_tier": "24gb"}

        def get_sam3(self):
            return None   # gated weights not accessible

    monkeypatch.setattr(segment, "get_state", lambda: St())
    assert segment._select_characters(det, [None, None], {}) is det
    assert segment._select_characters(det, [None, None], {"mask_selector_enabled": False}) is det


def test_sam3_access_is_checked_once(monkeypatch):
    from backend.models import sam3_concept
    calls = []

    def fake_download(repo, name, token=None):
        calls.append(repo)
        raise OSError("401 gated")

    import huggingface_hub
    monkeypatch.delenv("AMV_SAM3_PATH", raising=False)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    sam3_concept._ACCESS.clear()
    assert not sam3_concept.weights_accessible("tok")
    assert not sam3_concept.weights_accessible("tok")
    assert calls == ["facebook/sam3"]
    sam3_concept._ACCESS.clear()


def test_estimate_foreground_removes_background_colour_from_soft_edges():
    from backend.models.edge_refine import estimate_foreground
    h, w = 40, 120
    alpha = np.tile(np.clip((np.arange(w) - 50) / 20.0, 0, 1), (h, 1)).astype(np.float32)[:, ::-1].copy()
    F = np.array([230, 40, 40], np.float32)     # red character on the left
    B = np.array([40, 60, 230], np.float32)     # blue sky on the right
    img = (alpha[..., None] * F + (1 - alpha[..., None]) * B).round().astype(np.uint8)
    out = estimate_foreground(img, alpha).astype(np.float32)
    soft = (alpha > 0.1) & (alpha < 0.9)
    before = np.abs(img[soft].astype(np.float32) - F).mean()
    after = np.abs(out[soft] - F).mean()
    assert after < 0.25 * before
    opaque = alpha >= 0.999
    assert np.array_equal(out[opaque].astype(np.uint8), img[opaque])


def test_export_finish_frame_fg_mode_keeps_the_matte_and_fixes_colour(tmp_path, monkeypatch):
    """The 0.6 export path: no guided-filter halo (alpha only resized),
    edge colour decontaminated."""
    from backend.export import ffmpeg as ff
    seen = {}

    def fake_raw(src, s, e, masks, out, *, codec, fps, finish_frame, check_cancel=None):
        h, w = 20, 60
        alpha_small = np.zeros((10, 30), np.float32)
        alpha_small[:, :12] = 1.0
        alpha_small[:, 12] = 0.5
        rgb = np.zeros((h, w, 3), np.uint8)
        rgb[:, :25] = (230, 40, 40)
        rgb[:, 25:] = (40, 60, 230)
        seen["rgba"] = finish_frame(rgb, alpha_small)
        return out

    monkeypatch.setattr(ff, "_export_alpha_raw", fake_raw)
    ff._encode_scene_with_alpha("x.mkv", 0, 1000, np.zeros((1, 10, 30), np.float32), str(tmp_path / "o.mov"),
                                raw_pipes=True)
    rgba = seen["rgba"]
    assert rgba.shape == (20, 60, 4)
    assert rgba[:, 40:, 3].max() == 0                       # no halo far out in the sky
    soft = (rgba[..., 3] > 25) & (rgba[..., 3] < 230)
    assert soft.any() and rgba[soft][:, 2].mean() < 120    # soft edge not blue


def test_tighten_cuts_pockets_where_the_selector_sees_someone():
    from backend.models.auto_roto import tighten
    h, w = 200, 300
    body = _blob(h, w, 100, 40, 200, 180)
    pocket = _blob(h, w, 200, 100, 240, 140)            # background kept next to the body
    m = np.stack([body + pocket, body + pocket])
    sel = (body > 0).astype(np.uint8)                   # SAM 3: the body, not the pocket
    out = tighten(m, [sel, np.zeros_like(sel)], tight_frac=0.01)
    assert out[0, 120, 150] == 1.0 and out[0, 120, 235] == 0.0   # pocket cut
    assert out[0, 120, 201] > 0.9                                 # outline kept
    assert np.array_equal(out[1], m[1])                           # selector saw nobody: unchanged
    assert tighten(m, []) is m


# ── 0.6 max quality (roto 3) ─────────────────────────────────────────────────
def test_hug_with_tips_gives_back_opaque_tips_not_glow_or_pockets():
    from backend.models.auto_roto import hug_with_tips
    h, w = 608, 1080                              # working size (the reach scales with it)
    body = _blob(h, w, 400, 100, 700, 560)
    base = body.copy()
    base[200:204, 375:400] = 1.0                  # opaque spike sticking out (a hair tip)
    base[450:540, 700:715] = 0.4                  # semi-transparent glow next to the body
    base[300:340, 700:740] = 1.0                  # opaque pocket between body and ...
    base[290:350, 740:760] = 1.0                  # ... an arm (the pocket touches both)
    sel = body.copy()
    sel[290:350, 740:760] = 1.0                   # SAM 3: body + arm, not the pocket
    out = hug_with_tips(base, sel, tight_frac=0.004, max_touch=0.3)
    assert out[202, 380] == 1.0                   # tip given back
    assert out[500, 712] == 0.0                   # glow cut
    assert out[320, 725] == 0.0                   # pocket cut
    assert out[320, 550] == 1.0


def test_drop_fragments_removes_loose_bits_outside_the_selector():
    from backend.models.auto_roto import drop_fragments
    a = _blob(100, 200, 50, 10, 120, 90) + _blob(100, 200, 160, 40, 164, 44)   # body + a bit of halo
    sel = _blob(100, 200, 50, 10, 120, 90)
    out = drop_fragments(a, sel)
    assert out[50, 80] == 1.0 and out[42, 162] == 0.0
    keep = drop_fragments(a, sel + _blob(100, 200, 158, 38, 166, 46))   # the selector sees it: kept
    assert keep[42, 162] == 1.0


def test_finish_edges_max_keeps_the_ink_line_and_drops_the_glow():
    from backend.models.edge_refine import finish_edges
    h, w = 608, 1080
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = (60, 120, 230)                       # blue sky
    rgb[:, :500] = (240, 220, 120)                # blonde hair
    rgb[:, 497:502] = (20, 15, 10)                # ink outline
    rgb[:, 502:510] = (150, 170, 175)             # pale glow (half sky, half hair colour)
    alpha = np.zeros((h, w), np.float32)
    alpha[:, :502] = 1.0
    alpha[:, 502:510] = 0.5                       # the model kept the glow at half alpha
    col, a = finish_edges(rgb, alpha, "max")
    assert (a[300, 497:502] == 1.0).all()         # ink untouched
    assert a[300, 506] < 0.3                      # glow fades
    col_fg, a_fg = finish_edges(rgb, alpha, "fg")
    assert np.array_equal(a_fg, alpha)


def test_quality_by_tier_and_setting():
    class St:
        def __init__(self, platform, tier):
            self.sel = {"platform": platform, "effective_tier": tier}

        def model_selection(self):
            return self.sel

    assert auto_roto.mask_quality(St("nvidia", "24gb")) == "stable"
    assert auto_roto.mask_quality(St("nvidia", "24gb"), "max") == "max"
    assert auto_roto.mask_quality(St("nvidia", "fast")) == "standard"
    assert auto_roto.mask_quality(St("mac", "mac16")) is not None
    assert auto_roto.mask_quality(St("cpu", "fast"), "auto") == "standard"
    assert auto_roto.mask_quality(St("nvidia", "24gb"), "standard") == "standard"


def test_tighten_max_quality_uses_the_tips_aware_hug():
    from backend.models.auto_roto import tighten
    h, w = 608, 1080
    body = _blob(h, w, 400, 100, 700, 560)
    base = body.copy()
    base[200:204, 375:400] = 1.0
    m = np.stack([base])
    std = tighten(m, [body.astype(np.uint8)], quality="standard", tight_frac=0.004)
    mx = tighten(m, [body.astype(np.uint8)], quality="max")
    assert std[0, 202, 380] == 0.0 and mx[0, 202, 380] == 1.0


def test_hug_gives_back_a_held_object_but_not_a_pocket_of_sky():
    from backend.models.auto_roto import hug_with_tips
    h, w = 608, 1080
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = (90, 150, 230)                        # sky
    body = _blob(h, w, 400, 100, 700, 560)
    rgb[100:560, 400:700] = (150, 90, 160)         # purple character
    rgb[300:360, 500:580] = (240, 240, 240)        # white bowl held in front of the chest
    rgb[200:240, 620:660] = (90, 150, 230)         # sky seen through a gap in the body
    base = body.copy()                             # the detector keeps both, opaque
    sel = body.copy()
    sel[300:360, 500:580] = 0.0                    # SAM 3 leaves out the bowl ...
    sel[200:240, 620:660] = 0.0                    # ... and the gap
    out = hug_with_tips(base, sel, rgb=rgb)
    assert out[330, 540] == 1.0                    # bowl given back
    assert out[220, 640] == 0.0                    # sky stays out
    no_rgb = hug_with_tips(base, sel)
    assert no_rgb[330, 540] == 0.0


def test_unmix_recovers_the_true_edge_colour_of_a_composite():
    from backend.models.edge_refine import finish_edges, unmix_with_background
    h, w = 120, 400
    F = np.zeros((h, w, 3), np.float32); F[:] = (240, 200, 80)       # character
    B = np.zeros((h, w, 3), np.float32); B[:] = (40, 90, 200)        # background
    alpha = np.tile(np.clip((210 - np.arange(w)) / 20.0, 0, 1), (h, 1)).astype(np.float32)
    I = (alpha[..., None] * F + (1 - alpha[..., None]) * B).round().astype(np.uint8)
    out = unmix_with_background(I, alpha).astype(np.float32)
    soft = (alpha > 0.3) & (alpha < 0.95)
    assert np.abs(out[soft] - F[soft]).max() < 6                   # true colour back
    col, a = finish_edges(I, alpha, "max")
    assert np.abs(col[soft].astype(np.float32) - F[soft]).mean() < 8


def test_apply_selector_batched_matches_one_by_one():
    from backend.models.auto_roto import apply_selector
    det = np.stack([(_blob(60, 100, 5 + t, 5, 30 + t, 50) + _blob(60, 100, 70, 5, 95, 50)) * 255
                    for t in range(7)]).astype(np.uint8)
    frames = [np.full((60, 100, 3), t, np.uint8) for t in range(7)]

    class One:
        def segment(self, rgb, prompt):
            t = int(rgb[0, 0, 0])
            return _blob(60, 100, 5 + t, 5, 30 + t, 50)

    class Many(One):
        calls = 0

        def segment_many(self, images, prompt):
            Many.calls += 1
            return [self.segment(f, prompt) for f in images]

    k1, k2 = [], []
    a = apply_selector(det, frames, One(), keep=k1)
    b = apply_selector(det, frames, Many(), keep=k2)
    assert np.array_equal(a, b) and len(k1) == len(k2) == 7
    assert Many.calls == 1                       # one batched call for the whole chunk


# ── 0.6 held objects (roto held) ─────────────────────────────────────────────
def _staff_scene(h=608, w=1080):
    body = _blob(h, w, 400, 100, 600, 560)
    staff = _blob(h, w, 380, 40, 396, 580)             # held against the body's side
    pillar = _blob(h, w, 900, 0, 940, 608)              # far from anyone
    base = np.maximum(body, staff)
    base = np.maximum(base, pillar)
    return body, staff > 0.5, pillar > 0.5, base


def test_held_regions_keeps_objects_touching_a_character_that_the_detector_sees():
    from backend.models.auto_roto import held_regions
    body, staff, pillar, base = _staff_scene()
    ghost = np.zeros_like(staff)
    ghost[200:300, 600:640] = True                      # touches, but the detector sees background there
    held = held_regions(base, body, [staff, pillar, ghost])
    assert held[300, 386] and not held[300, 920] and not held[250, 620]
    assert not held_regions(base, np.zeros_like(body), [staff]).any()   # nobody: nothing held
    huge = np.ones_like(staff)
    assert not held_regions(np.ones_like(base), body, [huge]).any()    # bigger than 2x the character


def test_keep_held_gives_back_the_detector_matte_with_its_fringe():
    from backend.models.auto_roto import keep_held
    body, staff, _p, base = _staff_scene()
    base = base.copy()
    base[300, 397] = 0.5                                # antialiased fringe of the staff
    out = keep_held(body, base, staff)
    assert out[300, 386] == 1.0 and out[300, 397] == 0.5 and out[300, 920] == 0.0
    assert keep_held(body, base, None) is body


def test_persist_held_drops_one_frame_detections():
    from backend.models.auto_roto import persist_held
    a = np.zeros((40, 40), bool)
    a[10:20, 10:20] = True
    b = np.zeros((40, 40), bool)
    b[25:30, 25:30] = True                              # only on frame 1
    out = persist_held([a, a | b, a])
    assert out[1][15, 15] and not out[1][27, 27] and out[0][15, 15]
    assert persist_held([None, a])[0] is None


def test_apply_selector_and_tighten_give_back_held_objects():
    from backend.models.auto_roto import apply_selector, tighten
    body, staff, pillar, base = _staff_scene()
    det = np.stack([np.clip(base * 255, 0, 255).astype(np.uint8)] * 3)
    frames = [np.zeros(body.shape + (3,), np.uint8)] * 3

    class Sel:
        def segment(self, rgb, prompt):
            return body

        def segment_with_objects(self, rgb, prompt, objects, zoom=False):
            assert "staff" in objects
            return body, [staff, pillar]

    keep, held = [], []
    gated = apply_selector(det, frames, Sel(), keep=keep, held=held)
    assert gated[1, 50, 386] == 0 and held[1] is not None and held[1][50, 386] == 255   # beyond the gate
    assert held[1][300, 920] == 0
    m = gated.astype(np.float32) / 255.0
    out = tighten(m, keep, quality="max", held=held)
    assert out[1, 50, 386] == 1.0 and out[1, 300, 920] == 0.0
    assert tighten(m, keep, quality="max")[1, 50, 386] == 0.0
    # main-character mode: a prop that does not touch the tracked target stays out
    other = np.zeros_like(m)
    other[:, 100:300, 700:850] = 1.0
    assert tighten(other, keep, quality="max", held=held)[1, 50, 386] == 0.0


def test_keep_held_setting_is_max_quality_only(monkeypatch):
    from backend.api import segment
    monkeypatch.setattr(segment, "_quality", lambda cfg: cfg.get("q", "max"))
    assert segment._keep_held({"mask_keep_held": True}) is True
    assert segment._keep_held({"mask_keep_held": True, "q": "standard"}) is False
    assert segment._keep_held({"mask_keep_held": False}) is False
    assert segment._keep_held({}) is auto_roto.HELD_DEFAULT


def test_held_regions_rejects_light_effects():
    from backend.models.auto_roto import held_regions
    body, staff, _p, base = _staff_scene()
    rgb = np.zeros(body.shape + (3,), np.uint8)
    rgb[:] = (200, 60, 60)
    beam = staff.copy()
    rgb[beam] = (255, 250, 245)                         # a white-hot energy beam SAM 3 calls a "sword"
    assert held_regions(base, body, [beam]).any()       # without the frame: kept
    assert not held_regions(base, body, [beam], rgb=rgb).any()
    rgb[beam] = (60, 170, 250)                          # a blue ki flame: vivid
    assert not held_regions(base, body, [beam], rgb=rgb).any()
    rgb[beam] = (150, 210, 235)                         # a pale blue halo: bright but not vivid
    assert held_regions(base, body, [staff], rgb=rgb).any()
    rgb[beam] = (40, 40, 120)                           # a dark blue staff
    assert held_regions(base, body, [staff], rgb=rgb).any()


# ── 0.6 stable Auto (roto stable) ────────────────────────────────────────────
def _smoker(h=400, w=600):
    """BiRefNet's matte of a character holding a cigarette that sticks out
    of SAM 3's mask, a detached piece of scenery and a faint halo."""
    det = _blob(h, w, 100, 40, 180, 190)                # body
    det[100:104, 180:214] = 1.0                         # cigarette, touching the hand
    det[20:60, 250:290] = 1.0                           # scenery, detached
    det[40:190, 180:200] = np.maximum(det[40:190, 180:200], 0.4)   # soft halo of sky beside the body
    sel = _blob(h, w, 100, 40, 180, 190)                # SAM 3: the body only
    return det, sel


def test_stable_frame_keeps_what_touches_the_character_and_drops_detached_scenery():
    from backend.models.auto_roto import stable_frame
    det, sel = _smoker()
    out = stable_frame(det, sel)
    assert out[102, 212] == 1.0                         # the cigarette is never cut
    assert out[40, 270] == 0.0                          # detached scenery dropped
    assert out[100, 140] == 1.0                         # body
    assert out[150, 181] > 0.3                          # the antialiased edge stays
    assert out[150, 195] == 0.0                         # the wide soft halo does not
    assert stable_frame(det, sel, soft_px=0)[150, 195] > 0.3   # (without the soft-edge limit it would)


def test_stable_frame_adds_the_selector_where_birefnet_loses_a_blurred_character():
    from backend.models.auto_roto import stable_frame
    det, sel = _smoker()
    out = stable_frame(np.zeros_like(det), sel)
    assert out[100, 140] == 1.0                         # BiRefNet empty (motion blur): SAM 3 keeps him
    assert stable_frame(det, None)[40, 270] == 1.0      # no character evidence: BiRefNet as is


def test_temporal_guard_refills_a_lost_frame_from_its_neighbours():
    from backend.models.auto_roto import lost_frames, temporal_guard
    T, h, w = 9, 120, 160
    masks = np.stack([_blob(h, w, 40 + t, 20, 100 + t, 100) for t in range(T)])
    masks[4] = 0.0                                      # the character vanishes on one frame
    frames = [np.full((h, w, 3), 30, np.uint8) for _ in range(T)]
    for t in range(T):
        frames[t][20:100, 40 + t:100 + t] = 200
    assert lost_frames(masks) == [4]
    out, fixed = temporal_guard(masks, frames)
    assert fixed == [4]
    assert (out[4] > 0.5).sum() > 0.8 * (masks[3] > 0.5).sum()
    assert lost_frames(out) == []


def test_area_jumps_counts_big_changes_between_neighbours_only_inside_a_shot():
    from backend.models.auto_roto import area_jumps
    h, w = 50, 50
    m = np.stack([_blob(h, w, 0, 0, 20, 20), _blob(h, w, 0, 0, 20, 20), _blob(h, w, 0, 0, 40, 40)])
    assert area_jumps(m) == [2]
    assert area_jumps(m, [(0, 2), (2, 3)]) == []         # across a cut: not a jump


def test_clean_specks_is_stable_in_time_and_keeps_held_objects():
    from backend.models.auto_roto import clean_specks, specks
    h, w = 200, 300
    a = _blob(h, w, 50, 50, 150, 150)
    a[10:12, 250:252] = 1.0                              # 4 px confetti: always gone
    a[20:27, 200:207] = 1.0                              # 49 px island, between the two thresholds
    a[180, 20] = 0.3                                     # faint orphan pixel
    lo, hi = 10 / (h * w), 100 / (h * w)
    out = clean_specks(a, None, lo_frac=lo, hi_frac=hi)
    assert out[11, 251] == 0.0 and out[180, 20] == 0.0
    assert out[23, 203] == 0.0                           # no previous frame keeping it: gone
    assert clean_specks(a, a, lo_frac=lo, hi_frac=hi)[23, 203] == 1.0   # kept before: kept now
    held = np.zeros((h, w), np.float32); held[10:12, 250:252] = 1.0
    assert clean_specks(a, None, held, lo_frac=lo, hi_frac=hi)[11, 251] == 1.0
    assert out[100, 100] == 1.0                          # the character itself never
    assert specks(a, island_frac=hi)[0] == 2 and specks(out, island_frac=hi) == (0, 0)


def test_stable_masks_main_mode_ignores_characters_away_from_the_track():
    from backend.models.auto_roto import stable_masks
    h, w = 100, 200
    det = np.stack([_blob(h, w, 10, 10, 60, 90) + _blob(h, w, 120, 10, 170, 90)] * 3)
    sel = [((_blob(h, w, 10, 10, 60, 90) + _blob(h, w, 120, 10, 170, 90)) > 0).astype(np.uint8)] * 3
    track = np.stack([_blob(h, w, 10, 10, 60, 90)] * 3)
    both = stable_masks(det, sel, track)
    one = stable_masks(det, sel, track, only_tracked=True)
    assert both[1, 50, 140] == 1.0 and one[1, 50, 140] == 0.0 and one[1, 50, 30] == 1.0


def test_apply_selector_batched_objects_pass():
    from backend.models.auto_roto import apply_selector
    h, w = 120, 200
    body = _blob(h, w, 60, 10, 120, 110)
    stick = _blob(h, w, 120, 50, 190, 56)
    det = np.stack([np.maximum(body, stick) * 255] * 3).astype(np.uint8)
    frames = [np.full((h, w, 3), 90, np.uint8) for _ in range(3)]

    class Sel:
        calls = 0

        def segment_objects_many(self, images, prompt, objects, zoom=False):
            Sel.calls += 1
            return [body.copy() for _ in images], [[stick > 0.5] for _ in images]

    keep, held = [], []
    apply_selector(det, frames, Sel(), keep=keep, held=held)
    assert Sel.calls == 1 and len(keep) == 3 and len(held) == 3
    assert held[1] is not None and held[1][53, 170] > 200


def test_stable_frame_drops_an_aura_flare_but_keeps_a_strand_and_does_not_blink():
    from backend.models.auto_roto import stable_frame
    h, w = 400, 600
    body = _blob(h, w, 200, 50, 300, 350)               # SAM 3's character, 30 000 px
    det = body.copy()
    det[100:104, 300:340] = 1.0                         # a strand / cigarette: 160 px
    det[50:350, 300:330] = 1.0                          # an aura flare along the side: 9 000 px (30 %)
    out = stable_frame(det, body)
    assert out[200, 320] == 0.0                         # the flare goes
    assert out[200, 302] == 1.0                         # ... but not the character's own edge zone
    det2 = body.copy()
    det2[100:104, 300:340] = 1.0
    assert stable_frame(det2, body)[102, 335] == 1.0    # the strand stays
    det3 = body.copy()
    det3[50:150, 300:330] = 1.0                         # 3 000 px: 10 %, between the thresholds
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = (70, 140, 230)                             # sky all around
    rgb[50:350, 200:300] = (230, 120, 40)               # the character
    assert stable_frame(det3, body, rgb=rgb)[100, 320] == 0.0      # sky-coloured chunk: out
    assert stable_frame(det3, body)[100, 320] == 1.0               # (no colours: kept, inclusive)
    rgb[50:150, 300:330] = (230, 120, 40)                          # character-coloured: a sleeve
    assert stable_frame(det3, body, rgb=rgb)[100, 320] == 1.0
    rgb[50:150, 300:330] = (70, 140, 230)
    rgb[50:75, 300:330] = (230, 120, 40)                           # 25 % sleeve, 75 % sky
    assert stable_frame(det3, body, rgb=rgb)[100, 320] == 0.0
    rgb[50:110, 300:330] = (230, 120, 40)                          # 60 % sleeve, 40 % sky
    assert stable_frame(det3, body, rgb=rgb)[100, 320] == 0.0      # over 35 %: out ...
    assert stable_frame(det3, body, rgb=rgb, prev=det3)[100, 320] == 1.0   # ... unless kept before (no blink)


def test_temporal_guard_does_not_draw_back_a_character_that_went_behind_a_wall():
    from backend.models.auto_roto import temporal_guard
    T, h, w = 9, 120, 160
    masks = np.stack([_blob(h, w, 40, 20, 100, 100) for _ in range(T)])
    masks[4] = 0.0
    frames = [np.full((h, w, 3), 30, np.uint8) for _ in range(T)]
    det = masks.copy()                                  # BiRefNet sees nothing there on frame 4 either
    out, fixed = temporal_guard(masks, frames, det)
    assert fixed == [] and out[4].max() == 0.0
    det[4] = _blob(h, w, 42, 22, 98, 60) * 0.9          # it still sees half of him (blurred)
    out, fixed = temporal_guard(masks, frames, det)
    assert fixed == [4] and out[4, 50, 70] > 0.5        # refilled where it still shows


def test_track_auto_stable_path_runs_with_fake_models(monkeypatch, tmp_path):
    """The app's Auto route in "stable" mode, models faked: BiRefNet sees the
    character on every frame but one (motion blur), SAM 3 always sees it,
    SAM 2 follows its seed; no frame comes out empty, the cigarette stays."""
    from backend.api import segment
    from backend.models import auto_roto as ar
    T, h, w = 6, 120, 200
    paths = []
    for t in range(T):
        f = np.full((h, w, 3), 40, np.uint8)
        f[20:110, 60:120] = 200
        p = tmp_path / f"{t:05d}.jpg"
        Image.fromarray(f).save(p)
        paths.append(p)
    body = _blob(h, w, 60, 20, 120, 110)
    with_cig = body.copy(); with_cig[60:63, 120:135] = 1.0

    class Bi:
        calls = 0

        def alpha_batch(self, rgbs, batch=4):
            out = []
            for _ in rgbs:
                out.append(np.zeros((h, w), np.float32) if Bi.calls == 3 else with_cig.copy())
                Bi.calls += 1
            return out

    class S3:
        def segment_objects_many(self, images, prompt, objects, zoom=False):
            return [body.copy() for _ in images], [[] for _ in images]

        def offload(self):
            pass

    class S2:
        window_frames = 96

        def track_keyframes(self, keyframes, frames_dir, frame_paths, frame_size, soft=True):
            return np.stack([body] * len(frame_paths))

    class St:
        settings = {"models": {"mask_temporal_smooth_enabled": True}}

        def get_birefnet(self):
            return Bi()

        def get_sam3(self):
            return S3()

        def get_sam2(self, variant=None):
            return S2()

        def model_selection(self):
            return {"platform": "nvidia", "effective_tier": "24gb"}

    monkeypatch.setattr(segment, "get_state", lambda: St())
    monkeypatch.setattr(ar, "mask_models", lambda state: dict(ar.MASK_TIERS["nvidia"]["24gb"]))
    monkeypatch.setattr(segment, "_sam2_available", lambda: True)
    sess = segment.SegmentSession(session_id="s", scene_id=1, video_path="v", start_ms=0, end_ms=250, fps=24.0,
                                  frames_dir=tmp_path, frame_paths=paths, frame_size=(w, h))
    body_req = segment.TrackRequest(session_id="s")
    masks = segment._track_auto(sess, body_req, lambda seen, total: None)
    assert masks.shape == (T, h, w)
    assert all((m > 0.5).sum() > 0.8 * body.sum() for m in masks)      # no frame lost
    assert all(m[61, 130] > 0.5 for i, m in enumerate(masks) if i != 3)  # the cigarette stays


def test_stable_frame_ignores_a_rock_the_tracker_picked_up():
    from backend.models.auto_roto import stable_frame
    h, w = 400, 600
    body = _blob(h, w, 200, 50, 300, 350)
    rock = _blob(h, w, 20, 20, 60, 60)                  # 1 600 px, detached
    det = np.maximum(body, rock)
    track = det.copy()                                  # SAM 2 followed the rock too
    assert stable_frame(det, body, track)[40, 40] == 0.0
    other = _blob(h, w, 400, 50, 500, 350)              # a whole character SAM 3 missed on this frame
    det2, track2 = np.maximum(body, other), np.maximum(body, other)
    assert stable_frame(det2, body, track2)[200, 450] == 1.0


def test_persist_selector_drops_a_one_frame_false_character():
    from backend.models.auto_roto import persist_selector
    h, w = 100, 200
    goku = _blob(h, w, 80, 10, 120, 90)
    rock = _blob(h, w, 10, 10, 30, 30)
    sel = [goku, goku, np.maximum(goku, rock), goku, goku]
    out = persist_selector(sel)
    assert out[2][20, 20] == 0.0 and out[2][50, 100] == 1.0
    sel = [np.maximum(goku, rock)] * 3                  # a second character seen on every frame
    assert persist_selector(sel)[1][20, 20] == 1.0
