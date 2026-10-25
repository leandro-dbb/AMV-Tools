"""Synthetic anime-like clips with exact ground-truth alpha.

Used by the Mask Mode benchmark (``python -m backend.finish.bench_mask``)
and by ``tests/test_finish.py``. A flat-shaded "character" (head, hair
spikes, body, swinging arms, black ink outline) moves over a painted
background. Everything is rendered at 4x and area-downsampled, so the ground
truth alpha carries real anti-aliased fractional edges; the ``fast`` scenario
also averages several sub-frame positions, producing genuine motion-blurred
alpha — the hard case for matting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SS = 4  # supersampling factor


@dataclass
class Scenario:
    name: str
    speed: float           # px/frame (output resolution)
    blur_samples: int      # >1 = motion blur
    low_contrast: bool
    cut_at: int | None     # frame index where the shot changes


SCENARIOS = {
    "walk": Scenario("walk", 3.0, 1, False, None),
    "fast": Scenario("fast", 14.0, 5, False, None),
    "lowcontrast": Scenario("lowcontrast", 3.0, 1, True, None),
    "cut": Scenario("cut", 4.0, 1, False, 12),
}


def _background(w: int, h: int, variant: int, low_contrast: bool) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(100 + variant)
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    if low_contrast:
        top, bottom = np.array([214, 170, 150]), np.array([190, 140, 128])
    elif variant == 0:
        top, bottom = np.array([120, 170, 235]), np.array([235, 225, 200])
    else:
        top, bottom = np.array([60, 40, 90]), np.array([200, 90, 60])
    bg = (top * (1 - y) + bottom * y) * np.ones((1, w, 1), np.float32)
    bg = bg.astype(np.uint8)
    # A few flat "buildings"/props with ink lines, like a painted anime BG.
    for _ in range(7):
        x0 = int(rng.integers(0, w - 40)); bw = int(rng.integers(30, 120))
        bh = int(rng.integers(h // 5, h // 2))
        col = np.clip(bottom + rng.integers(-50, 50, 3), 0, 255) if not low_contrast \
            else np.clip(bottom + rng.integers(-15, 15, 3), 0, 255)
        cv2.rectangle(bg, (x0, h - bh), (x0 + bw, h), tuple(int(c) for c in col), -1)
        cv2.rectangle(bg, (x0, h - bh), (x0 + bw, h), (40, 40, 50), 1)
    return cv2.GaussianBlur(bg, (3, 3), 0)


def _draw_character(w: int, h: int, cx: float, cy: float, phase: float, low_contrast: bool):
    """Render at SS x resolution. Returns (rgb_ss uint8, alpha_ss float32)."""
    import cv2

    W, H = w * SS, h * SS
    mask = np.zeros((H, W), np.uint8)
    rgb = np.zeros((H, W, 3), np.uint8)
    s = SS * h / 360.0
    X, Y = cx * SS, cy * SS

    skin = (238, 200, 176) if not low_contrast else (222, 178, 156)
    hair = (60, 60, 140) if not low_contrast else (200, 156, 140)
    cloth = (220, 60, 80) if not low_contrast else (205, 150, 135)

    def P(dx, dy):
        return int(round(X + dx * s)), int(round(Y + dy * s))

    parts = []
    # body
    parts.append(("poly", cloth, np.array([P(-30, 10), P(30, 10), P(42, 120), P(-42, 120)])))
    # arms (swinging)
    for side in (-1, 1):
        ang = np.sin(phase + (0 if side < 0 else np.pi)) * 0.6
        ex, ey = 34 * side, 18
        hx, hy = ex + side * 20 + np.sin(ang) * 40, ey + np.cos(ang) * 70
        parts.append(("line", skin, (P(ex, ey), P(hx, hy), int(12 * s))))
    # head + hair spikes
    parts.append(("ellipse", skin, (P(0, -28), (int(30 * s), int(36 * s)))))
    spikes = np.array([P(-36, -40), P(-48, -70), P(-20, -62), P(-8, -92), P(8, -64),
                       P(30, -84), P(28, -58), P(52, -60), P(34, -30), P(0, -52)])
    parts.append(("poly", hair, spikes))

    for kind, col, geom in parts:
        m = np.zeros_like(mask)
        if kind == "poly":
            cv2.fillPoly(m, [geom], 255, lineType=cv2.LINE_8)
        elif kind == "line":
            a, b, th = geom
            cv2.line(m, a, b, 255, th)
        else:
            c, axes = geom
            cv2.ellipse(m, c, axes, 0, 0, 360, 255, -1)
        rgb[m > 0] = col
        mask |= m
    # ink outline (inside the silhouette, like cel line art)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(3 * SS), int(3 * SS)))
    edge = mask & ~cv2.erode(mask, k)
    rgb[edge > 0] = (25, 20, 30)
    return rgb, (mask > 0).astype(np.float32)


def load_sprite(path: str, max_side: int = 1400) -> np.ndarray:
    """RGBA uint8 character cutout (e.g. a transparent anime PNG)."""
    from PIL import Image

    im = Image.open(path).convert("RGBA")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    arr = np.asarray(im)
    ys, xs = np.nonzero(arr[..., 3] > 0)
    return np.ascontiguousarray(arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


def _draw_sprite(w: int, h: int, cx: float, cy: float, phase: float, sprite: np.ndarray,
                 low_contrast: bool):
    """Affine-transformed sprite at SS x resolution: (rgb_ss, alpha_ss)."""
    import cv2

    W, H = w * SS, h * SS
    sh, sw = sprite.shape[:2]
    scale = (0.82 * H) / sh
    ang = np.sin(phase) * 5.0  # sway, degrees
    m = cv2.getRotationMatrix2D((sw / 2.0, sh / 2.0), ang, scale)
    m[0, 2] += cx * SS - sw / 2.0
    m[1, 2] += cy * SS - sh / 2.0
    rgb = sprite[..., :3].astype(np.float32)
    if low_contrast:
        rgb = rgb * 0.55 + np.array([196, 180, 176], np.float32) * 0.45
    a = sprite[..., 3].astype(np.float32) / 255.0
    rgb_ss = cv2.warpAffine(rgb * a[..., None], m, (W, H), flags=cv2.INTER_LINEAR)
    a_ss = cv2.warpAffine(a, m, (W, H), flags=cv2.INTER_LINEAR)
    safe = np.maximum(a_ss, 1e-6)[..., None]
    return np.clip(rgb_ss / safe, 0, 255).astype(np.uint8), np.clip(a_ss, 0, 1)


def render_clip(scenario: str = "walk", *, n_frames: int = 24, w: int = 320, h: int = 180,
                sprite: np.ndarray | None = None, return_fg: bool = False):
    """Return ``(frames uint8 (T,H,W,3), alpha float32 (T,H,W))`` (plus the
    premultiplied foreground layer ``(T,H,W,3)`` float32 with ``return_fg``).

    ``sprite`` (RGBA, see :func:`load_sprite`) replaces the built-in
    geometric figure with a real character cutout."""
    import cv2

    sc = SCENARIOS[scenario]
    bgs = [_background(w, h, 0, sc.low_contrast).astype(np.float32),
           _background(w, h, 1, sc.low_contrast).astype(np.float32)]
    frames = np.zeros((n_frames, h, w, 3), np.uint8)
    alphas = np.zeros((n_frames, h, w), np.float32)
    fgs = np.zeros((n_frames, h, w, 3), np.float32) if return_fg else None
    cut_at = None if sc.cut_at is None else (sc.cut_at if sc.cut_at < n_frames else n_frames // 2)
    for t in range(n_frames):
        shot = 1 if (cut_at is not None and t >= cut_at) else 0
        acc_rgb = np.zeros((h, w, 3), np.float32)
        acc_a = np.zeros((h, w), np.float32)
        n_sub = sc.blur_samples
        for k in range(n_sub):
            tt = t + (k / n_sub - 0.5 * (n_sub - 1) / n_sub if n_sub > 1 else 0.0)
            if shot == 0:
                cx = w * 0.25 + sc.speed * tt
                cy = h * 0.45 + 4 * np.sin(tt * 0.5)
            else:
                cx = w * 0.7 - sc.speed * (tt - cut_at)
                cy = h * 0.40
            if sprite is not None:
                rgb_ss, a_ss = _draw_sprite(w, h, cx, h * 0.5 + (cy - h * 0.45), tt * 0.35, sprite,
                                            sc.low_contrast)
            else:
                rgb_ss, a_ss = _draw_character(w, h, cx, cy, tt * 0.35, sc.low_contrast)
            a = cv2.resize(a_ss, (w, h), interpolation=cv2.INTER_AREA)
            # premultiplied colour of the character layer
            col = cv2.resize(rgb_ss.astype(np.float32) * a_ss[..., None], (w, h), interpolation=cv2.INTER_AREA)
            acc_rgb += col
            acc_a += a
        a = acc_a / n_sub
        fg_premul = acc_rgb / n_sub
        comp = fg_premul + bgs[shot] * (1.0 - a[..., None])
        frames[t] = np.clip(comp, 0, 255).round().astype(np.uint8)
        alphas[t] = np.clip(a, 0.0, 1.0)
        if fgs is not None:
            fgs[t] = fg_premul
    if return_fg:
        return frames, alphas, fgs
    return frames, alphas


def codec_roundtrip(frames: np.ndarray, crf: int = 23, fps: float = 24.0) -> np.ndarray:
    """H.264 4:2:0 encode/decode, like real episode footage."""
    import subprocess

    from ..indexing.cuts import _ffmpeg_path

    t, h, w = frames.shape[:3]
    enc = subprocess.run(
        [_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", str(crf),
         "-pix_fmt", "yuv420p", "-f", "h264", "-"],
        input=frames.tobytes(), check=True, capture_output=True).stdout
    dec = subprocess.run(
        [_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-f", "h264", "-i", "-",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        input=enc, check=True, capture_output=True).stdout
    return np.frombuffer(dec, np.uint8).reshape(-1, h, w, 3)[:t].copy()


def write_clip(path: str, frames: np.ndarray, fps: float = 24.0, *, with_audio: bool = True) -> str:
    """Encode frames to a near-lossless H.264 MP4 (+ a sine audio track)."""
    import subprocess

    from ..indexing.cuts import _ffmpeg_path

    t, h, w = frames.shape[:3]
    cmd = [_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-"]
    if with_audio:
        cmd += ["-f", "lavfi", "-t", f"{t / fps}", "-i", "sine=frequency=440:sample_rate=48000"]
    cmd += ["-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", "-preset", "ultrafast"]
    if with_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += ["-y", path]
    subprocess.run(cmd, input=frames.tobytes(), check=True, capture_output=True)
    return path


# ── metrics ─────────────────────────────────────────────────────────────────
def boundary_band(gt: np.ndarray, px: int = 3) -> np.ndarray:
    import cv2

    b = (gt > 0.5).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    band = (cv2.dilate(b, k) - cv2.erode(b, k)) > 0
    return band | ((gt > 0.02) & (gt < 0.98))


def composite_error(rgb: np.ndarray, alpha: np.ndarray, gt_alpha: np.ndarray, gt_fg_premul: np.ndarray,
                    bg=(255.0, 0.0, 255.0)) -> float:
    """Mean |composite − true composite| over a magenta background, in the
    boundary band, as % of full scale. Captures what the user actually sees
    (halo of old background, eaten edges), including RGB decontamination."""
    bgc = np.asarray(bg, np.float32)
    errs = []
    for f, a, ga, gf in zip(rgb, alpha, gt_alpha, gt_fg_premul):
        a = np.clip(a.astype(np.float32), 0, 1)[..., None]
        pred = f.astype(np.float32) * a + bgc * (1 - a)
        true = gf + bgc * (1 - ga[..., None])
        band = boundary_band(ga)
        errs.append(np.abs(pred - true)[band].mean() / 255.0 * 100 if band.any() else 0.0)
    return float(np.mean(errs))


def matte_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """``pred``/``gt``: (T, H, W) float in 0..1.

    - ``iou``: mean per-frame IoU of the >0.5 silhouettes (higher = better)
    - ``edge_mae``: mean |alpha error| inside a ±3 px band around the true
      boundary, x100 (lower = better)
    - ``flicker``: dtSSD — RMS of (pred temporal change − true temporal
      change), x100 (lower = steadier; 0 = moves exactly like the truth)
    - ``mae``: whole-frame mean |alpha error| x100
    """
    pred = np.clip(pred.astype(np.float32), 0, 1)
    gt = gt.astype(np.float32)
    ious, edges = [], []
    for p, g in zip(pred, gt):
        pb, gb = p > 0.5, g > 0.5
        union = (pb | gb).sum()
        ious.append(1.0 if union == 0 else (pb & gb).sum() / union)
        band = boundary_band(g)
        edges.append(np.abs(p - g)[band].mean() if band.any() else 0.0)
    dp = np.diff(pred, axis=0)
    dg = np.diff(gt, axis=0)
    flick = np.sqrt(((dp - dg) ** 2).mean(axis=(1, 2))).mean() if len(pred) > 1 else 0.0
    return {
        "iou": float(np.mean(ious)),
        "iou_min": float(np.min(ious)),
        "edge_mae": float(np.mean(edges) * 100),
        "flicker": float(flick * 100),
        "mae": float(np.abs(pred - gt).mean() * 100),
    }
