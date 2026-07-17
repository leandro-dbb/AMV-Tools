"""Mask post-processing utilities used by alpha export.

The functions in this module are conservative by design: they operate near the
mask boundary, preserve confident foreground interiors, and are exposed through
settings so users can choose between edge cleanup and detail preservation.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_MIN_BG_SAMPLES = 100


def shrink_mask(mask: np.ndarray, shrink_px: int = 2) -> np.ndarray:
    """Erode the foreground support by ``shrink_px`` pixels."""
    if shrink_px <= 0:
        return mask

    import cv2

    if mask.dtype == bool:
        binary = mask.astype(np.uint8)
    else:
        binary = (mask > 0.5).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(binary, kernel, iterations=int(shrink_px))

    if mask.dtype == bool:
        return eroded.astype(bool)
    return mask * eroded.astype(mask.dtype)


def suppress_bg_in_edge_band(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    color_dist_threshold: float = 25.0,
    edge_band_px: int = 4,
) -> np.ndarray:
    """Drop boundary foreground pixels whose RGB matches the background."""
    import cv2

    assert image.ndim == 3 and image.shape[2] == 3, f"image must be (H,W,3), got {image.shape}"
    assert mask.shape == image.shape[:2], f"mask {mask.shape} != image {image.shape[:2]}"

    bin_mask = mask if mask.dtype == bool else mask > 0.5
    if not bin_mask.any():
        return mask

    iterations = max(1, int(edge_band_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded_core = cv2.erode(bin_mask.astype(np.uint8), kernel, iterations=iterations).astype(bool)
    edge_band = bin_mask & ~eroded_core
    if not edge_band.any():
        return mask

    dilated_mask = cv2.dilate(bin_mask.astype(np.uint8), kernel, iterations=iterations).astype(bool)
    confident_bg = ~dilated_mask
    bg_count = int(confident_bg.sum())
    if bg_count < _MIN_BG_SAMPLES:
        log.warning(
            "suppress_bg_in_edge_band: only %d background samples; skipping",
            bg_count,
        )
        return mask

    img_rgb = image.astype(np.float32)
    if img_rgb.size and img_rgb.max() <= 1.0:
        img_rgb *= 255.0

    bg_color = np.median(img_rgb[confident_bg], axis=0)
    edge_rgb = img_rgb[edge_band]
    dist = np.sqrt(((edge_rgb - bg_color) ** 2).sum(axis=1))
    matches_bg = dist < float(color_dist_threshold)
    if not matches_bg.any():
        return mask

    edge_idx = np.argwhere(edge_band)
    suppress_idx = edge_idx[matches_bg]
    result = mask.copy()
    if mask.dtype == bool:
        result[suppress_idx[:, 0], suppress_idx[:, 1]] = False
    else:
        result[suppress_idx[:, 0], suppress_idx[:, 1]] = 0.0

    if mask.dtype == bool:
        assert np.array_equal(result[eroded_core], mask[eroded_core])
    else:
        diff = np.abs(result[eroded_core] - mask[eroded_core])
        assert diff.max() < 1e-6 if diff.size else True

    return result


def suppress_bg_in_alpha_band(
    image: np.ndarray,
    alpha: np.ndarray,
    *,
    color_dist_threshold: float = 20.0,
    edge_band_px: int = 10,
    alpha_min: float = 0.02,
) -> np.ndarray:
    """Remove background-colored pixels from a soft-alpha boundary band."""
    import cv2

    assert image.ndim == 3 and image.shape[2] == 3, f"image must be (H,W,3), got {image.shape}"
    assert alpha.shape == image.shape[:2], f"alpha {alpha.shape} != image {image.shape[:2]}"

    alpha_f = np.clip(alpha.astype(np.float32, copy=False), 0.0, 1.0)
    support = alpha_f > float(alpha_min)
    if not support.any():
        return alpha_f

    iterations = max(1, int(edge_band_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded_support = cv2.erode(support.astype(np.uint8), kernel, iterations=iterations).astype(bool)
    edge_band = support & ~eroded_support
    if not edge_band.any():
        return alpha_f

    dilated_support = cv2.dilate(support.astype(np.uint8), kernel, iterations=iterations).astype(bool)
    confident_bg = ~dilated_support
    bg_count = int(confident_bg.sum())
    if bg_count < _MIN_BG_SAMPLES:
        log.warning(
            "suppress_bg_in_alpha_band: only %d background samples; skipping",
            bg_count,
        )
        return alpha_f

    img_rgb = image.astype(np.float32)
    if img_rgb.size and img_rgb.max() <= 1.0:
        img_rgb *= 255.0

    bg_color = np.median(img_rgb[confident_bg], axis=0)
    edge_rgb = img_rgb[edge_band]
    dist = np.sqrt(((edge_rgb - bg_color) ** 2).sum(axis=1))
    matches_bg = dist < float(color_dist_threshold)
    if not matches_bg.any():
        return alpha_f

    edge_idx = np.argwhere(edge_band)
    suppress_idx = edge_idx[matches_bg]
    result = alpha_f.copy()
    result[suppress_idx[:, 0], suppress_idx[:, 1]] = 0.0
    return result


def refine_soft_alpha(
    alpha: np.ndarray,
    *,
    shrink_px: int = 1,
    black_point: float = 0.08,
    white_point: float = 0.85,
) -> np.ndarray:
    """Apply optional support erosion and levels to a soft alpha matte."""
    alpha_f = np.clip(alpha.astype(np.float32, copy=False), 0.0, 1.0)

    if shrink_px > 0:
        import cv2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha_f = cv2.erode(alpha_f, kernel, iterations=int(shrink_px))

    if white_point > black_point:
        alpha_f = np.clip((alpha_f - black_point) / (white_point - black_point), 0.0, 1.0)
        alpha_f = alpha_f * alpha_f * (3.0 - 2.0 * alpha_f)

    return alpha_f.astype(np.float32, copy=False)


def upsample_alpha_guided(
    alpha: np.ndarray,
    image: np.ndarray,
    *,
    radius: int | None = None,
    eps: float = 1e-4,
) -> np.ndarray:
    """Upscale a low-res alpha matte to ``image``'s resolution, snapping the
    edge falloff back onto the full-res art with a guided filter.

    The mask session runs the models on frames capped at ``mask_max_dim``; a
    plain bilinear upscale of that matte reads as a soft halo around anime
    line work at export resolution. Guided filtering (He et al. 2010,
    implemented here with plain box filters so core OpenCV is enough)
    re-anchors the alpha gradient to the luminance edges of the full-res
    frame at millisecond cost. Falls back to bilinear when OpenCV is
    unavailable.
    """
    assert image.ndim == 3 and image.shape[2] == 3, f"image must be (H,W,3), got {image.shape}"
    h, w = image.shape[:2]
    alpha_f = np.clip(alpha.astype(np.float32, copy=False), 0.0, 1.0)

    try:
        import cv2
    except ImportError:
        log.warning("upsample_alpha_guided: OpenCV missing — falling back to bilinear")
        from PIL import Image as PILImage
        img = PILImage.fromarray(np.ascontiguousarray(alpha_f), mode="F").resize((w, h), PILImage.BILINEAR)
        return np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)

    if alpha_f.shape != (h, w):
        alpha_f = cv2.resize(alpha_f, (w, h), interpolation=cv2.INTER_LINEAR)

    guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    if radius is None:
        # Scale the window with resolution: ≈6 px at 1080p. Too small and the
        # filter can't reach past the bilinear blur; too large and thin hair
        # strands get averaged away.
        radius = max(4, int(round(min(h, w) / 180.0)))
    ksize = (2 * int(radius) + 1, 2 * int(radius) + 1)

    def box(x: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(x, ddepth=-1, ksize=ksize)

    mean_i = box(guide)
    mean_p = box(alpha_f)
    var_i = box(guide * guide) - mean_i * mean_i
    cov_ip = box(guide * alpha_f) - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    q = box(a) * guide + box(b)
    return np.clip(q, 0.0, 1.0).astype(np.float32, copy=False)


def temporal_smooth_alpha(
    masks: np.ndarray,
    *,
    strength: float = 0.5,
    cut_threshold: float = 0.12,
) -> np.ndarray:
    """Bidirectional EMA over a ``(T, H, W)`` float alpha stack.

    Per-frame models (BiRefNet/ISNet) re-decide the silhouette on every
    frame, so static regions of the matte jitter frame to frame — reads as
    "alpha flicker" once composited. A symmetric forward+backward EMA keeps
    edges steady without the lag a one-directional filter would add. Frames
    whose raw matte differs from their neighbour by more than
    ``cut_threshold`` (mean |Δalpha|; in-scene motion is typically < 0.05)
    reset the history so hard cuts and teleport-fast motion don't ghost.
    """
    if masks.ndim != 3 or masks.shape[0] < 3:
        return masks
    if not np.issubdtype(masks.dtype, np.floating) or strength <= 0.0:
        return masks

    # History weight per direction. Capped so that even at strength=1 the
    # current frame keeps the majority share after the two passes are merged.
    k = min(1.0, max(0.0, float(strength))) * 0.5
    t_total = masks.shape[0]

    diff = np.zeros(t_total, dtype=np.float32)
    for i in range(1, t_total):
        diff[i] = float(np.mean(np.abs(masks[i] - masks[i - 1])))

    fwd = masks.astype(np.float32, copy=True)
    for i in range(1, t_total):
        if diff[i] > cut_threshold:
            continue
        fwd[i] = (1.0 - k) * masks[i] + k * fwd[i - 1]

    bwd = masks.astype(np.float32, copy=True)
    for i in range(t_total - 2, -1, -1):
        if diff[i + 1] > cut_threshold:
            continue
        bwd[i] = (1.0 - k) * masks[i] + k * bwd[i + 1]

    return ((fwd + bwd) * 0.5).astype(np.float32, copy=False)


def clean_edges_with_background(
    image: np.ndarray,
    alpha: np.ndarray,
    *,
    band_lo: float = 0.02,
    band_hi: float = 0.98,
    min_fb_dist: float = 12.0,
    max_distance_px: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Background-aware edge cleanup: re-solve the matte's soft band against
    the *actual* local background instead of trusting the model's falloff.

    The classic "beige aura" halo happens because boundary pixels keep the
    source RGB (old background) under semi-transparent alpha. But we know
    more than a generic matting model does: the background is right there in
    the frame. For every pixel in the soft band we look up

      F = colour of the nearest confident-foreground pixel (alpha ≥ band_hi)
      B = colour of the nearest confident-background pixel (alpha ≤ band_lo)

    via distance transforms, then invert the compositing equation
    ``C = a·F_true + (1-a)·B``:

      a  = clamp( (C-B)·(F-B) / |F-B|² )      — colour-solved alpha
      F_true = (C - (1-a)·B) / a              — spill-free edge colour

    Where F and B are too similar (|F-B| < min_fb_dist, e.g. dark hair on a
    dark doorway) the colour solve is meaningless, so the model's alpha and
    a nearest-F fill are kept instead. Pixels farther than max_distance_px
    from both regions are also left untouched.

    Returns ``(rgb_uint8, alpha_float32)``. No-op (returns inputs) when
    OpenCV is missing or the matte has no soft band.
    """
    assert image.ndim == 3 and image.shape[2] == 3, f"image must be (H,W,3), got {image.shape}"
    assert alpha.shape == image.shape[:2], f"alpha {alpha.shape} != image {image.shape[:2]}"

    a = np.clip(alpha.astype(np.float32, copy=False), 0.0, 1.0)
    try:
        import cv2
    except ImportError:
        log.warning("clean_edges_with_background: OpenCV missing — skipping")
        return image, a

    core = a >= float(band_hi)
    bg = a <= float(band_lo)
    band = ~core & ~bg
    if not core.any() or not bg.any() or not band.any():
        return image, a

    img = image.astype(np.float32)
    if img.size and img.max() <= 1.0:
        img *= 255.0

    def _nearest_colors(region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(dist, region_coords, labels) for every pixel, via the same
        DIST_LABEL_PIXEL trick as decontaminate_rgb_with_alpha: labels
        enumerate the zero pixels of the input in row-major order, so
        ``region_coords[labels-1]`` is the nearest region pixel."""
        non_region = (~region).astype(np.uint8)
        dist, labels = cv2.distanceTransformWithLabels(
            non_region, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL,
        )
        coords = np.argwhere(region)
        return dist, coords, labels

    dist_f, core_coords, labels_f = _nearest_colors(core)
    dist_b, bg_coords, labels_b = _nearest_colors(bg)

    reach = float(max(1, int(max_distance_px)))
    solvable = band & (dist_f <= reach) & (dist_b <= reach) & (labels_f > 0) & (labels_b > 0)
    if not solvable.any():
        return image, a

    idx_f = labels_f[solvable].astype(np.int64) - 1
    idx_b = labels_b[solvable].astype(np.int64) - 1
    if idx_f.max(initial=-1) >= len(core_coords) or idx_b.max(initial=-1) >= len(bg_coords):
        return image, a

    fc = core_coords[idx_f]
    bc = bg_coords[idx_b]
    f_col = img[fc[:, 0], fc[:, 1]]              # (N, 3) nearest-foreground colour
    b_col = img[bc[:, 0], bc[:, 1]]              # (N, 3) nearest-background colour
    c_col = img[solvable]                        # (N, 3) observed colour
    a_model = a[solvable]                        # (N,)

    fb = f_col - b_col
    denom = (fb * fb).sum(axis=1)                # |F-B|²
    reliable = denom >= float(min_fb_dist) ** 2

    a_color = np.clip(((c_col - b_col) * fb).sum(axis=1) / np.maximum(denom, 1e-6), 0.0, 1.0)
    a_new = np.where(reliable, a_color, a_model).astype(np.float32)

    # Spill-free edge colour: unpremultiply against the known background.
    # Below ~0.15 alpha the division amplifies noise, so fall back to the
    # nearest confident-foreground colour there (the pixel is nearly
    # transparent anyway — what matters is that it is NOT background-coloured).
    safe_a = np.maximum(a_new, 0.15)[:, None]
    f_true = (c_col - (1.0 - safe_a) * b_col) / safe_a
    low = (a_new < 0.15)[:, None] | ~reliable[:, None]
    f_true = np.where(low, f_col, f_true)
    f_true = np.clip(f_true, 0.0, 255.0)

    out_a = a.copy()
    out_a[solvable] = a_new
    out_rgb = image.copy()
    ys, xs = np.nonzero(solvable)
    out_rgb[ys, xs] = f_true.round().astype(np.uint8)
    return out_rgb, out_a


def decontaminate_rgb_with_alpha(
    image: np.ndarray,
    alpha: np.ndarray,
    *,
    core_threshold: float = 0.98,
    alpha_min: float = 0.02,
    max_distance_px: int = 32,
) -> np.ndarray:
    """Replace soft-edge RGB with nearby confident foreground color."""
    import cv2

    assert image.ndim == 3 and image.shape[2] == 3, f"image must be (H,W,3), got {image.shape}"
    assert alpha.shape == image.shape[:2], f"alpha {alpha.shape} != image {image.shape[:2]}"

    alpha_f = np.clip(alpha.astype(np.float32, copy=False), 0.0, 1.0)
    core = alpha_f >= float(core_threshold)
    target = (alpha_f > float(alpha_min)) & ~core
    if not core.any() or not target.any():
        return image

    non_core = (~core).astype(np.uint8)
    dist, labels = cv2.distanceTransformWithLabels(
        non_core,
        cv2.DIST_L2,
        3,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    core_coords = np.argwhere(core)
    valid = target & (dist <= max(1, int(max_distance_px))) & (labels > 0)
    if not valid.any():
        return image

    label_idx = labels[valid].astype(np.int64) - 1
    if label_idx.max(initial=-1) >= len(core_coords):
        return image

    nearest = core_coords[label_idx]
    result = image.copy()
    valid_coords = np.argwhere(valid)
    result[valid_coords[:, 0], valid_coords[:, 1]] = image[nearest[:, 0], nearest[:, 1]]
    return result
