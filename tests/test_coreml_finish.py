"""Apple Silicon finishing on Core ML / the Neural Engine (tour 3), exercised
on any box: the Core ML models are replaced by numpy stand-ins, the worker
process protocol runs for real (spawn)."""
from __future__ import annotations

import numpy as np
import pytest

from backend.finish import ane, coreml, temporal


# ── worker process protocol ─────────────────────────────────────────────────
def echo_handler(scale: float):
    def handle(arrays):
        a = arrays[0]
        if a.size == 1 and float(a.ravel()[0]) < 0:
            raise ValueError("negative")
        return [a.astype(np.float32) * scale, np.array([a.shape[0]], dtype=np.int64)]
    return handle


def broken_loader():
    raise RuntimeError("no Core ML here")


def test_worker_round_trip_large_arrays_in_order():
    w = coreml.Worker("tests.test_coreml_finish:echo_handler", 2.0, name="test-worker")
    try:
        assert w.wait_ready(60)
        big = np.arange(3 * 1088 * 1920, dtype=np.uint8).reshape(1088, 1920, 3)  # 6 MB in, 25 MB out
        ids = [w.submit([big]) for _ in range(3)]     # several in flight: no pipe deadlock
        for rid in ids:
            out, n = w.result(rid, timeout=60)
            assert out.dtype == np.float32 and np.array_equal(out, big.astype(np.float32) * 2.0)
            assert int(n[0]) == 1088
    finally:
        w.close()


def test_worker_errors_are_reported_not_hung():
    w = coreml.Worker("tests.test_coreml_finish:echo_handler", 1.0)
    try:
        assert w.wait_ready(60)
        with pytest.raises(RuntimeError, match="negative"):
            w.call([np.array([-1.0])], timeout=60)
        assert not w.usable()
    finally:
        w.close()
    bad = coreml.Worker("tests.test_coreml_finish:broken_loader")
    assert bad.wait_ready(60) is False and "no Core ML" in bad.failed
    bad.close()


def test_coreml_off_outside_macos(monkeypatch):
    monkeypatch.setattr(coreml.sys, "platform", "linux")
    monkeypatch.setattr(ane.sys, "platform", "linux")
    assert not coreml.available() and not ane.available()


# ── ANE upscale tiling ──────────────────────────────────────────────────────
def _fake_net_x2(x: np.ndarray) -> np.ndarray:
    """A local operator with a 2-px radius (like a conv stack), x2 output:
    any tiling with enough context must reproduce the whole-frame result."""
    c = x[0].astype(np.float32)
    p = np.pad(c, ((0, 0), (2, 2), (2, 2)))          # zero padding, like convs
    h, w = c.shape[1:]
    acc = sum(p[:, dy:dy + h, dx:dx + w] * (1 + dy + 2 * dx) for dy in range(5) for dx in range(5)) / 75.0
    return np.repeat(np.repeat(acc, 2, axis=1), 2, axis=2)[None]


@pytest.mark.parametrize("h,w", [(1080, 1920), (720, 1280), (577, 1031), (300, 400)])
def test_tiles_cover_the_frame_with_context(h, w):
    in_h, in_w, tiles = ane.tile_grid(h, w)
    assert in_h <= ane.MAX_TILE_H and in_w <= ane.MAX_TILE_W
    covered = np.zeros((h, w), dtype=np.int32)
    for wy, wx, y0, y1, x0, x1 in tiles:
        assert 0 <= wy and wy + in_h <= h and 0 <= wx and wx + in_w <= w
        assert y0 - wy >= ane.TILE_PAD or wy == 0
        assert x0 - wx >= ane.TILE_PAD or wx == 0
        assert wy + in_h - y1 >= ane.TILE_PAD or wy + in_h == h
        assert wx + in_w - x1 >= ane.TILE_PAD or wx + in_w == w
        covered[y0:y1, x0:x1] += 1
    assert (covered == 1).all()
    if (h, w) == (1080, 1920):
        assert (in_h, in_w, len(tiles)) == (576, 516, 8)   # the measured shape


def test_tiled_x2_equals_whole_frame():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (300, 520, 3), dtype=np.uint8)
    whole = np.clip(np.rint(_fake_net_x2(frame.transpose(2, 0, 1)[None].astype(np.float16))[0]), 0, 255)
    whole = whole.astype(np.uint8).transpose(1, 2, 0)
    in_h, in_w, tiles = ane.tile_grid(*frame.shape[:2], pad=4)
    out = ane.upscale_tiles(_fake_net_x2, frame, in_h, in_w, tiles)
    assert np.array_equal(out, whole)


class _FakeAne:
    """In-process stand-in for AneUpscaler: answers one frame per poll."""
    fail_after = None

    def __init__(self, weights, h, w):
        self.h, self.w = h, w
        self.ready = True
        self.failed = False
        self.in_flight = []
        self.frames_done = 0
        self.closed = False

    def can_take(self):
        return not self.failed and len(self.in_flight) < 2

    def submit(self, idx, frame):
        self.in_flight.append((idx, frame))

    def poll(self, block=False):
        if self.failed or not self.in_flight:
            return []
        if _FakeAne.fail_after is not None and self.frames_done >= _FakeAne.fail_after:
            self.failed = True
            return []
        idx, f = self.in_flight.pop(0)
        self.frames_done += 1
        return [(idx, np.full((2 * self.h, 2 * self.w, 3), 200, np.uint8))]

    def take_back(self):
        back, self.in_flight = self.in_flight, []
        return back

    def close(self):
        self.closed = True


@pytest.mark.parametrize("fail_after", [None, 3])
@pytest.mark.parametrize("procs", [1, 2])
def test_split_writes_every_frame_in_order(monkeypatch, fail_after, procs):
    monkeypatch.setattr(ane, "AneUpscaler", _FakeAne)
    monkeypatch.setattr(ane, "ANE_PROCS", procs)
    _FakeAne.fail_after = fail_after
    frames = [np.full((4, 6, 3), i, np.uint8) for i in range(20)]
    written = []
    n = ane.run_split(iter(frames), lambda f: np.full((8, 12, 3), int(f[0, 0, 0]), np.uint8),
                      written.append, "w.pth", 4, 6)
    assert len(written) == 20
    gpu = [i for i, f in enumerate(written) if int(f[0, 0, 0]) != 200]
    assert all(int(written[i][0, 0, 0]) == i for i in gpu)
    assert n == 20 - len(gpu) and n > 0
    if fail_after is not None:
        assert n == fail_after * procs


# ── temporal stabilisation ──────────────────────────────────────────────────
def test_sort_nanmedian_is_numpys():
    rng = np.random.default_rng(1)
    for k in (2, 3, 4, 5):
        v = (rng.random((k, 40, 60), dtype=np.float32) * 3 - 1)
        mask = rng.random(v.shape) < 0.35
        mask[0] = False
        v[mask] = np.nan
        assert np.array_equal(temporal.nanmedian0(v), np.nanmedian(v, axis=0))


def test_threaded_stabilisation_matches_sequential(monkeypatch):
    pytest.importorskip("cv2")
    rng = np.random.default_rng(2)
    frames = [rng.integers(0, 256, (48, 64, 3), dtype=np.uint8) for _ in range(8)]
    stack = rng.random((8, 48, 64), dtype=np.float32)
    monkeypatch.setattr(temporal, "_THREADS", 1)
    seq = temporal.stabilize_stack(stack, cuts={3}, warper=temporal.FlowWarper(frames))
    monkeypatch.setattr(temporal, "_THREADS", 4)
    par = temporal.stabilize_stack(stack, cuts={3}, warper=temporal.FlowWarper(frames))
    assert np.array_equal(seq, par)


# ── RIFE: a tensor timestep (traced graphs) gives the float one's output ────
def test_rife_tensor_timestep_matches_float():
    torch = pytest.importorskip("torch")
    from backend.finish.archs.rife import IFNet

    torch.manual_seed(0)
    net = IFNet(scale=1.0).eval()
    a, b = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        ref = net(a, b, 0.25)
        out = net(a, b, torch.tensor([0.25]))
    assert torch.equal(ref, out)
