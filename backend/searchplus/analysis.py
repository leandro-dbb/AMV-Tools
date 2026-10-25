"""Per-scene camera motion, intensity and colour metrics.

The cut detector already decodes every video once at 160×90 gray (every 4th
frame, ≈6 samples/s on 24 fps anime). ``FrameStatsAccumulator`` plugs into
that pass as a ``frame_sink`` and, for each pair of consecutive samples,
tracks Shi-Tomasi corners with pyramidal Lucas-Kanade (forward-backward
checked) and fits a global similarity transform with RANSAC:

    x' = s·R·x + t            (x centred, in frame widths)

``t`` is translation (pan/tilt), ``s − 1`` the scale change (zoom). Corners
sit on line art, which is exactly where anime carries motion information
(flat fills carry none), and RANSAC keeps foreground action from dragging
the camera estimate. What the model can't explain is *local* motion —
characters moving — which feeds the intensity score.

Only a handful of floats per sample pair are kept, so a 24-minute episode
costs ~100 KB of RAM regardless of resolution. Scenes are aggregated after the
cuts are known: every pair whose both ends lie inside the scene contributes.

Sign conventions (image coordinates, y down): content sliding left means the
camera pans right; content sliding up means the camera tilts down; content
expanding means zoom in.
"""
from __future__ import annotations

import colorsys
import io
import logging
import math
import queue
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Bump when the estimator/aggregation changes so the background analyzer
# recomputes stale rows.
ANALYSIS_VERSION = 1

MOTION_TYPES = (
    "static", "pan_left", "pan_right", "tilt_up", "tilt_down",
    "zoom_in", "zoom_out", "shake",
)

COLOR_BUCKETS = (
    "red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink",
    "brown", "white", "gray", "black",
)

# ── tuning ──────────────────────────────────────────────────────────────────
_MAX_CORNERS = 120           # Shi-Tomasi corners tracked per sample
_MIN_TRACKS = 8              # fewer → no reliable camera estimate for the pair
_PAIR_CUT_DIFF = 0.35        # histogram L1/2 diff: the pair straddles a cut
_SUBCUT_DIFF = 0.12          # big intra-scene change: flash, impact frame, smear
_FLASH_JUMP = 0.12           # luminance jump (0-1) counted as a flash
_STATIC_SPEED = 0.03         # frame widths / s below which the camera is still
_ZOOM_RADIUS = 0.35          # converts zoom rate into an equivalent edge speed
_SHAKE_MIN_SPEED = 0.10      # mean instantaneous speed needed to call "shake"
_SHAKE_COHERENCE = 0.35      # |net velocity| / mean speed below this → shake


def _cv2():
    try:
        import cv2  # noqa: F401
        return cv2
    except Exception as exc:  # pragma: no cover - only on broken installs
        log.warning("OpenCV unavailable, motion analysis disabled: %s", exc)
        return None


def _hist(gray: np.ndarray) -> np.ndarray:
    # == np.histogram(bins=32, range=(0, 256)), ~10× faster
    h = np.bincount((gray >> 3).ravel(), minlength=32).astype(np.float32)
    s = h.sum()
    return h / s if s > 0 else h


@dataclass
class FrameStatsAccumulator:
    """Streaming per-sample statistics. Feed it with ``add(ts_ms, gray)``."""

    times: list[int] = field(default_factory=list)
    luma: list[float] = field(default_factory=list)
    # per pair (sample i-1 → i), parallel lists
    p_t0: list[int] = field(default_factory=list)
    p_t1: list[int] = field(default_factory=list)
    p_tx: list[float] = field(default_factory=list)
    p_ty: list[float] = field(default_factory=list)
    p_zoom: list[float] = field(default_factory=list)
    p_resid: list[float] = field(default_factory=list)
    p_flow: list[float] = field(default_factory=list)
    p_diff: list[float] = field(default_factory=list)
    p_valid: list[bool] = field(default_factory=list)

    def __post_init__(self):
        self._cv2 = _cv2()
        self._prev: Optional[np.ndarray] = None
        self._prev_hist: Optional[np.ndarray] = None
        self._broken = False
        self._queue: Optional[queue.Queue] = None
        self._worker: Optional[threading.Thread] = None
        # Optional pool: pair estimates are independent, so they can run on
        # several cores while ``add`` keeps the per-sample order.
        self._pool = None
        self._pending: list = []   # [(pair index, future, frame a, frame b)]
        self._arrays_cache = None

    def reset(self) -> None:
        for _i, fut, _a, _b in self._pending:
            fut.cancel()
        self._pending = []
        self._arrays_cache = None
        for name in ("times", "luma", "p_t0", "p_t1", "p_tx", "p_ty", "p_zoom",
                     "p_resid", "p_flow", "p_diff", "p_valid"):
            getattr(self, name).clear()
        self._prev = None
        self._prev_hist = None

    @property
    def available(self) -> bool:
        return self._cv2 is not None

    def sink(self, ts_ms: int, gray: np.ndarray) -> None:
        """``add`` that never raises — safe to hand to the cut detector, which
        must not fail because of an analysis bug. After ``start_async()``
        frames are queued to a worker thread so tracking overlaps decoding
        (OpenCV releases the GIL); call ``finish()`` before aggregating."""
        if self._broken:
            return
        if self._queue is not None:
            # The decoder may reuse its buffer: hand the worker a copy.
            self._queue.put((ts_ms, np.array(gray, dtype=np.uint8, copy=True)))
            return
        self._safe_add(ts_ms, gray)

    def _safe_add(self, ts_ms: int, gray: np.ndarray) -> None:
        if self._broken:
            return
        try:
            self.add(ts_ms, gray)
        except Exception as exc:  # pragma: no cover - defensive
            self._broken = True
            log.warning("motion analysis disabled for this video: %s", exc)

    def start_async(self, maxsize: int = 256, processes: int = 0) -> "FrameStatsAccumulator":
        """``processes > 0`` also spreads the pair estimates over a process
        pool. The estimator is GIL-bound (small numpy ops), so threads don't
        scale; processes do. Results are identical — each estimate is
        deterministic. Falls back to in-thread estimates if the pool can't
        start."""
        if processes > 0 and self._pool is None and self._cv2 is not None:
            try:
                import multiprocessing as mp
                from concurrent.futures import ProcessPoolExecutor
                self._pool = ProcessPoolExecutor(max_workers=processes, mp_context=mp.get_context("spawn"),
                                                 initializer=_pool_worker_init)
            except Exception as exc:  # pragma: no cover - platform specific
                log.warning("motion process pool unavailable (%s); estimating in-thread", exc)
                self._pool = None
        if self._queue is None:
            self._queue = queue.Queue(maxsize=maxsize)

            def work():
                while True:
                    item = self._queue.get()
                    if item is None:
                        return
                    self._safe_add(*item)

            self._worker = threading.Thread(target=work, daemon=True, name="motion-analysis")
            self._worker.start()
        return self

    def finish(self) -> None:
        if self._queue is not None:
            self._queue.put(None)
            self._worker.join()
            self._queue = None
            self._worker = None
        self._drain_pending(0)
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def _drain_pending(self, keep: int) -> None:
        """Write finished pool estimates back, oldest first, until at most
        ``keep`` are still in flight (bounds memory to ``keep`` frame pairs)."""
        while len(self._pending) > keep:
            i, fut, a, b = self._pending.pop(0)
            try:
                est = fut.result()
            except Exception as exc:
                # A dead worker (OOM-killed, …) must not lose the pair.
                log.debug("motion pair estimate failed in pool (%s); retrying in-thread", exc)
                est = self._estimate(a, b)
            self._arrays_cache = None
            if est is not None and i < len(self.p_valid):
                (self.p_tx[i], self.p_ty[i], self.p_zoom[i],
                 self.p_resid[i], self.p_flow[i]) = est
                self.p_valid[i] = True

    def add(self, ts_ms: int, gray: np.ndarray) -> None:
        if gray.ndim == 3:
            gray = gray.mean(axis=2).astype(np.uint8)
        gray = np.ascontiguousarray(gray, dtype=np.uint8)
        if self.times and ts_ms <= self.times[-1]:
            # The decoder restarted (hardware path failed → PyAV fallback):
            # drop the partial run instead of mixing two timelines.
            self.reset()
        hist = _hist(gray)
        self.times.append(int(ts_ms))
        self.luma.append(float(gray.mean()) / 255.0)
        if self._prev is not None and self._prev.shape == gray.shape and ts_ms > self.times[-2]:
            diff = float(np.abs(hist - self._prev_hist).sum()) * 0.5
            tx = ty = zoom = resid = flow = 0.0
            valid = False
            if diff < _PAIR_CUT_DIFF and self._cv2 is not None:
                fut = None
                if self._pool is not None:
                    try:
                        fut = self._pool.submit(estimate_pair, self._prev, gray)
                    except Exception as exc:  # broken pool: carry on in-thread
                        log.warning("motion process pool failed (%s); estimating in-thread", exc)
                        self._pool = None
                if fut is not None:
                    self._pending.append((len(self.p_t0), fut, self._prev, gray))
                    self._drain_pending(64)
                else:
                    est = self._estimate(self._prev, gray)
                    if est is not None:
                        tx, ty, zoom, resid, flow = est
                        valid = True
            self.p_t0.append(self.times[-2])
            self.p_t1.append(int(ts_ms))
            self.p_tx.append(tx)
            self.p_ty.append(ty)
            self.p_zoom.append(zoom)
            self.p_resid.append(resid)
            self.p_flow.append(flow)
            self.p_diff.append(diff)
            self.p_valid.append(valid)
        self._prev = gray
        self._prev_hist = hist

    _PER_SAMPLE = ("times", "luma")
    _PER_PAIR = ("p_t0", "p_t1", "p_tx", "p_ty", "p_zoom", "p_resid", "p_flow", "p_diff", "p_valid")

    def extend_from(self, parts: list["FrameStatsAccumulator"]) -> None:
        """Concatenate finished accumulators of consecutive stretches of one
        video, in order, with pair-only parts (no samples) for the pairs that
        join two stretches (see ``backend.indexing.chunked``). The result is
        what feeding every sample to this accumulator in order gives."""
        for part in parts:
            for name in self._PER_SAMPLE + self._PER_PAIR:
                getattr(self, name).extend(getattr(part, name))
        self._arrays_cache = None

    # ── sparse tracking + global model ─────────────────────────────────────
    def _estimate(self, a: np.ndarray, b: np.ndarray):
        """Track corners a→b (pyramidal LK, forward-backward checked) and fit
        a similarity transform with a small deterministic RANSAC.

        Returns (tx, ty, zoom, residual, flow) in frame widths per pair, or
        None when the frame has too little texture to say anything."""
        return estimate_pair(a, b)


def _pool_worker_init() -> None:
    """OpenCV's own thread pool only adds spin-wait overhead on 160×90
    frames — and every pool process would start one. One thread each."""
    cv2 = _cv2()
    if cv2 is not None:
        cv2.setNumThreads(1)


def estimate_pair(a: np.ndarray, b: np.ndarray):
    """Module-level body of ``FrameStatsAccumulator._estimate`` (picklable,
    so it can run in a process pool)."""
    cv2 = _cv2()
    if cv2 is None:
        return None
    h, w = a.shape
    p0 = cv2.goodFeaturesToTrack(a, maxCorners=_MAX_CORNERS, qualityLevel=0.01,
                                 minDistance=3, blockSize=5)
    if p0 is None or len(p0) < _MIN_TRACKS:
        return None
    lk = dict(winSize=(11, 11), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, **lk)
    if p1 is None:
        return None
    back, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, p1, None, **lk)
    fb = np.linalg.norm((back - p0).reshape(-1, 2), axis=1)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.75)
    if int(ok.sum()) < _MIN_TRACKS:
        return None
    c = np.array([w / 2.0, h / 2.0], dtype=np.float64)
    x0 = (p0.reshape(-1, 2)[ok].astype(np.float64) - c) / w
    x1 = (p1.reshape(-1, 2)[ok].astype(np.float64) - c) / w
    params, inliers = _ransac_similarity(x0, x1, tol=1.0 / w)
    if params is None:
        return None
    sa, sb, tx, ty = params
    pred = _apply_similarity(params, x0)
    resid = float(np.linalg.norm(pred - x1, axis=1).mean())
    flow = float(np.linalg.norm(x1 - x0, axis=1).mean())
    zoom = math.hypot(sa, sb) - 1.0
    return float(tx), float(ty), float(zoom), resid, flow


def _apply_similarity(params, x: np.ndarray) -> np.ndarray:
    sa, sb, tx, ty = params
    return np.stack([sa * x[:, 0] - sb * x[:, 1] + tx,
                     sb * x[:, 0] + sa * x[:, 1] + ty], axis=1)


def _fit_similarity(x0: np.ndarray, x1: np.ndarray):
    n = len(x0)
    A = np.zeros((2 * n, 4))
    A[:n, 0] = x0[:, 0]
    A[:n, 1] = -x0[:, 1]
    A[:n, 2] = 1.0
    A[n:, 0] = x0[:, 1]
    A[n:, 1] = x0[:, 0]
    A[n:, 3] = 1.0
    rhs = np.concatenate([x1[:, 0], x1[:, 1]])
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return sol


def _ransac_similarity(x0: np.ndarray, x1: np.ndarray, tol: float, iters: int = 32):
    """Numpy RANSAC (cv2.estimateAffinePartial2D hits SIGILL on some OpenCV
    5 CPU builds, so we don't rely on it). All 2-point hypotheses are scored
    in one vectorised pass; seeded → deterministic."""
    n = len(x0)
    if n < 2:
        return None, None
    rng = np.random.default_rng(1234)
    idx = rng.integers(0, n, size=(iters, 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    i, j = idx[:, 0], idx[:, 1]
    d0 = x0[j] - x0[i]
    d1 = x1[j] - x1[i]
    den = (d0 * d0).sum(axis=1)
    ok = den > 1e-10
    den = np.where(ok, den, 1.0)
    sa = (d0[:, 0] * d1[:, 0] + d0[:, 1] * d1[:, 1]) / den
    sb = (d0[:, 0] * d1[:, 1] - d0[:, 1] * d1[:, 0]) / den
    tx = x1[i, 0] - (sa * x0[i, 0] - sb * x0[i, 1])
    ty = x1[i, 1] - (sb * x0[i, 0] + sa * x0[i, 1])
    px = sa[:, None] * x0[None, :, 0] - sb[:, None] * x0[None, :, 1] + tx[:, None]
    py = sb[:, None] * x0[None, :, 0] + sa[:, None] * x0[None, :, 1] + ty[:, None]
    inl_all = np.hypot(px - x1[None, :, 0], py - x1[None, :, 1]) < tol
    counts = np.where(ok, inl_all.sum(axis=1), -1)
    best_count = int(counts.max()) if len(counts) else -1
    if best_count < max(_MIN_TRACKS // 2, 2):
        # Degenerate set: plain least squares is still informative.
        return _fit_similarity(x0, x1), np.ones(n, dtype=bool)
    best_inl = inl_all[int(np.argmax(counts))]
    params = _fit_similarity(x0[best_inl], x1[best_inl])
    err = np.linalg.norm(_apply_similarity(params, x0) - x1, axis=1)
    inl = err < tol
    if inl.sum() >= 2:
        params = _fit_similarity(x0[inl], x1[inl])
    return params, inl


@dataclass
class SceneMetrics:
    motion_type: str
    motion_magnitude: float
    motion_dx: float
    motion_dy: float
    motion_zoom: float
    activity: float
    intensity: float
    cut_rate: float
    flash_count: int
    brightness: float
    dominant_color: Optional[str] = None
    accent_color: Optional[str] = None
    palette: list = field(default_factory=list)  # [[hex, share, bucket], ...]

    def as_row(self) -> dict:
        return {
            "motion_type": self.motion_type,
            "motion_magnitude": round(self.motion_magnitude, 4),
            "motion_dx": round(self.motion_dx, 5),
            "motion_dy": round(self.motion_dy, 5),
            "motion_zoom": round(self.motion_zoom, 5),
            "activity": round(self.activity, 4),
            "intensity": round(self.intensity, 4),
            "cut_rate": round(self.cut_rate, 4),
            "flash_count": int(self.flash_count),
            "brightness": round(self.brightness, 4),
            "dominant_color": self.dominant_color,
            "accent_color": self.accent_color,
            "palette": self.palette,
        }


def _sat(x: float, scale: float) -> float:
    """Soft saturation into [0, 1)."""
    return 1.0 - math.exp(-max(0.0, x) / scale)


def classify_motion(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> tuple[str, float, float, float, float]:
    """Classify per-pair camera velocities (frame widths/s, zoom rate/s).

    Returns (motion_type, magnitude 0-1, net_dx, net_dy, net_zoom)."""
    if vx.size == 0:
        return "static", 0.0, 0.0, 0.0, 0.0
    mx, my, mz = float(np.median(vx)), float(np.median(vy)), float(np.median(vz))
    speed = math.hypot(mx, my)
    zoom_speed = abs(mz) * _ZOOM_RADIUS
    # Medians everywhere: a few mistracked pairs (sprites crossing the frame,
    # repetitive patterns) must not flip a steady pan into "shake".
    inst = np.hypot(vx, vy) + np.abs(vz) * _ZOOM_RADIUS
    med_inst = float(np.median(inst))
    net = speed + zoom_speed
    coherent = net / med_inst if med_inst > 1e-9 else 1.0

    if vx.size >= 3 and med_inst > _SHAKE_MIN_SPEED and coherent < _SHAKE_COHERENCE:
        return "shake", _sat(med_inst, 0.35), mx, my, mz
    dominant = max(speed, zoom_speed)
    if dominant < _STATIC_SPEED:
        return "static", _sat(dominant, 0.35), mx, my, mz
    mag = _sat(dominant, 0.35)
    if zoom_speed >= 0.8 * speed:
        return ("zoom_in" if mz > 0 else "zoom_out"), mag, mx, my, mz
    if abs(mx) >= abs(my):
        return ("pan_right" if mx < 0 else "pan_left"), mag, mx, my, mz
    return ("tilt_down" if my < 0 else "tilt_up"), mag, mx, my, mz


_ARRAY_DTYPES = {"times": np.int64, "luma": np.float64, "p_t0": np.int64, "p_t1": np.int64,
                 "p_tx": np.float64, "p_ty": np.float64, "p_zoom": np.float64, "p_resid": np.float64,
                 "p_flow": np.float64, "p_diff": np.float64, "p_valid": bool}


def _acc_arrays(acc: FrameStatsAccumulator) -> dict:
    """numpy views of the accumulator lists, built once per accumulator state
    (aggregating an episode used to re-convert every list for each of its
    ~400 scenes: O(samples × scenes))."""
    key = (len(acc.times), len(acc.p_t0))
    cache = getattr(acc, "_arrays_cache", None)
    if cache is None or cache[0] != key:
        cache = (key, {n: np.asarray(getattr(acc, n), dtype=d) for n, d in _ARRAY_DTYPES.items()})
        acc._arrays_cache = cache
    return cache[1]


def aggregate_scene(acc: FrameStatsAccumulator, start_ms: int, end_ms: int,
                    rgb: Optional[np.ndarray] = None, pal: Optional[list] = None) -> SceneMetrics:
    """Summarise the samples inside [start_ms, end_ms] into SceneMetrics.

    ``rgb`` (H×W×3 uint8, any size) drives the palette; without it only the
    luminance-based fields are filled."""
    dur_s = max((end_ms - start_ms) / 1000.0, 1e-3)
    A = _acc_arrays(acc)
    t0, t1 = A["p_t0"], A["p_t1"]
    inside = (t0 >= start_ms - 1) & (t1 <= end_ms + 1)
    valid = inside & A["p_valid"]

    dt = np.maximum((t1 - t0) / 1000.0, 1e-3)
    vx = (A["p_tx"] / dt)[valid] if valid.any() else np.empty(0)
    vy = (A["p_ty"] / dt)[valid] if valid.any() else np.empty(0)
    vz = (A["p_zoom"] / dt)[valid] if valid.any() else np.empty(0)
    motion_type, magnitude, mdx, mdy, mz = classify_motion(vx, vy, vz)

    if valid.any():
        flow_speed = float(np.median((A["p_flow"] / dt)[valid]))
        resid_speed = float(np.median((A["p_resid"] / dt)[valid]))
    else:
        flow_speed = resid_speed = 0.0
    activity = _sat(resid_speed, 0.15)

    diffs = A["p_diff"][inside] if inside.any() else np.empty(0)
    subcuts = int((diffs > _SUBCUT_DIFF).sum())
    cut_rate = subcuts / dur_s

    times, luma = A["times"], A["luma"]
    in_s = (times >= start_ms) & (times <= end_ms)
    lum = luma[in_s]
    if lum.size:
        brightness = float(lum.mean())
        lum_std = float(lum.std())
        flashes = int((np.diff(lum) > _FLASH_JUMP).sum()) if lum.size > 1 else 0
    else:
        brightness = float(np.asarray(rgb).mean() / 255.0) if rgb is not None else 0.0
        lum_std = 0.0
        flashes = 0

    dur_term = min(max((3.0 - dur_s) / 2.5, 0.0), 1.0)
    intensity = (
        0.35 * _sat(flow_speed, 0.35)
        + 0.15 * magnitude
        + 0.20 * _sat(cut_rate, 1.5)
        + 0.10 * min(lum_std / 0.15, 1.0)
        + 0.10 * _sat(flashes, 2.0)
        + 0.10 * dur_term
    )
    m = SceneMetrics(
        motion_type=motion_type,
        motion_magnitude=magnitude,
        motion_dx=mdx, motion_dy=mdy, motion_zoom=mz,
        activity=activity,
        intensity=float(min(max(intensity, 0.0), 1.0)),
        cut_rate=cut_rate,
        flash_count=flashes,
        brightness=brightness,
    )
    if rgb is not None:
        try:
            pal = palette(rgb) if pal is None else pal   # pal: precomputed (palette_gpu)
            m.palette = pal
            m.dominant_color, m.accent_color = dominant_and_accent(pal)
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("palette failed: %s", exc)
    return m


# ── colour ─────────────────────────────────────────────────────────────────
def color_bucket(r: int, g: int, b: int) -> str:
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    if v < 0.18:
        return "black"
    if s < 0.18:
        return "white" if v > 0.82 else "gray"
    deg = h * 360.0
    if deg < 15 or deg >= 345:
        return "red"
    if deg < 42:
        return "brown" if v < 0.6 else "orange"
    if deg < 70:
        return "yellow"
    if deg < 165:
        return "green"
    if deg < 195:
        return "cyan"
    if deg < 255:
        return "blue"
    if deg < 290:
        return "purple"
    return "pink"


def palette(rgb: np.ndarray, k: int = 3) -> list:
    """3-colour palette via a tiny deterministic k-means.

    Returns [[hex, share, bucket], ...] sorted by share (largest first)."""
    from PIL import Image
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    img = img.resize((48, 27), Image.Resampling.BILINEAR)
    px = np.asarray(img, dtype=np.float32).reshape(-1, 3)
    lum = px @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    order = np.argsort(lum)
    # Deterministic init: pixels at luminance quantiles.
    centers = px[order[[int(len(order) * q) for q in np.linspace(0.15, 0.85, k)]]].copy()
    labels = np.zeros(len(px), dtype=np.int64)
    for _ in range(10):
        d = ((px[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        for c in range(k):
            sel = labels == c
            if sel.any():
                centers[c] = px[sel].mean(axis=0)
    shares = np.bincount(labels, minlength=k) / float(len(px))
    out = []
    for c in np.argsort(-shares):
        if shares[c] <= 0:
            continue
        r, g, b = (int(round(x)) for x in centers[c])
        out.append([f"#{r:02x}{g:02x}{b:02x}", round(float(shares[c]), 3), color_bucket(r, g, b)])
    return out


def dominant_and_accent(pal: list) -> tuple[Optional[str], Optional[str]]:
    if not pal:
        return None, None
    dominant = pal[0][2]
    best, best_score = dominant, -1.0
    for hexc, share, bucket in pal:
        r, g, b = int(hexc[1:3], 16), int(hexc[3:5], 16), int(hexc[5:7], 16)
        _, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        score = s * v * math.sqrt(share)
        if s > 0.25 and score > best_score:
            best, best_score = bucket, score
    return dominant, best


def rgb_from_jpeg(data: Optional[bytes]) -> Optional[np.ndarray]:
    if not data:
        return None
    try:
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception:
        return None


# ── whole-video helper (background analyzer, tests) ────────────────────────
def collect_frame_stats(video_path: str, threshold: float = 0.40) -> FrameStatsAccumulator:
    """Decode the video once through the cut detector and return the
    accumulated per-sample stats (cuts themselves are discarded)."""
    from ..indexing import cuts
    acc = FrameStatsAccumulator().start_async()
    try:
        cuts.detect_cuts(video_path, threshold=threshold, frame_sink=acc.sink)
    finally:
        acc.finish()
    return acc
