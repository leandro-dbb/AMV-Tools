"""Real-ESRGAN (realesr-animevideov3) on the Apple Neural Engine, next to MPS.

On Apple Silicon the x2 upscale is GPU-bound (98 % busy) while the Neural
Engine sits idle. The compact SRVGG net converts to Core ML as-is, and on an
M1 Pro the ANE runs it as fast as the GPU (1080p frame in 8 tiles: 514 ms,
whole frame on MPS fp16: 540 ms; PSNR 65.9 dB vs fp32). Driven from the same
process, Core ML's ``predict`` holds the GIL and the GPU loop collapses
(62 -> 10 tiles/s); from a worker process the two add up (512-px tiles:
GPU 15.0/s alone, 14.4 + 18.0 ANE together).

So an upscale job hands every other frame to a worker process that owns the
Core ML model (converted once per input tile shape, cached on disk) while
the GPU keeps processing the rest; frames go back in order. The worker
computes exactly what :meth:`Upscaler.upscale` does for x2 — x4 net, clamp,
2x2 mean, x255, round — on overlapping tiles with 18 px of real context
(the net's receptive field), so tiles join without seams.

Optional: needs ``coremltools`` (macOS ``mps`` extra); without it, or with
``AMV_ANE=0``, the GPU does every frame as before.
"""
from __future__ import annotations

import importlib.util
import logging
import math
import os
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

TILE_PAD = 18           # receptive field of SRVGGNetCompact (18 3x3 convs)
MAX_TILE_H, MAX_TILE_W = 576, 516   # 1080p -> 2 x 4 tiles (measured best)
IN_FLIGHT = 2           # frames queued to the worker (one runs, one waits)
ANE_PROCS = int(os.environ.get("AMV_ANE_PROCS", "2") or 2)


def available() -> bool:
    if sys.platform != "darwin" or os.environ.get("AMV_ANE", "1") == "0":
        return False
    return importlib.util.find_spec("coremltools") is not None


def tile_grid(h: int, w: int, pad: int = TILE_PAD) -> tuple[int, int, list[tuple[int, int, int, int, int, int]]]:
    """(in_h, in_w, tiles): every tile reads a window of the same
    ``in_h x in_w`` size at (wy, wx) and contributes the core rows/cols
    [y0, y1) x [x0, x1) of the frame. Windows are shifted inside the frame
    at the borders, so each core keeps >= ``pad`` px of real context or
    touches the frame edge (where the whole-frame pass zero-pads too)."""
    ny = max(1, math.ceil(h / (MAX_TILE_H - 2 * pad)))
    nx = max(1, math.ceil(w / (MAX_TILE_W - 2 * pad)))
    ch, cw = math.ceil(h / ny), math.ceil(w / nx)
    in_h, in_w = min(h, ch + 2 * pad), min(w, cw + 2 * pad)
    tiles = []
    for iy in range(ny):
        for ix in range(nx):
            y0, x0 = iy * ch, ix * cw
            y1, x1 = min(h, y0 + ch), min(w, x0 + cw)
            if y0 >= y1 or x0 >= x1:
                continue
            wy = min(max(0, y0 - pad), h - in_h)
            wx = min(max(0, x0 - pad), w - in_w)
            tiles.append((wy, wx, y0, y1, x0, x1))
    return in_h, in_w, tiles


def _cache_dir() -> Path:
    """Next to the weights: the converted program goes away with the model
    (Settings > Models) and survives everything else."""
    from . import registry
    d = registry.model_dir("realesr-animevideov3") / "coreml"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build_model(weights: str, in_h: int, in_w: int, cache: str):
    """Core ML program of the x2 path for ``in_h x in_w`` tiles (cached)."""
    import coremltools as ct

    # Native x2 Compact checkpoints (e.g. 2xHFA2kAVCCompact) run as they
    # are; the x4 animevideov3 net is followed by the 2x2 mean, as on MPS.
    native2 = Path(weights).name.startswith("2x")
    name = Path(weights).stem if native2 else "srvgg-animevideov3-x2"
    path = Path(cache) / f"{name}-{in_h}x{in_w}-v1.mlpackage"
    if not path.exists():
        import torch
        import torch.nn.functional as F

        from .archs.realesrgan import SRVGGNetCompact

        sd = torch.load(weights, map_location="cpu", weights_only=True)
        sd = sd.get("params_ema") or sd.get("params") or sd
        # body = conv, prelu, (conv, prelu) * num_conv, conv: Compact = 16,
        # UltraCompact = 8
        n_conv = sum(1 for k, v in sd.items() if k.startswith("body.") and k.endswith(".weight") and v.dim() == 4)
        net = SRVGGNetCompact(num_feat=64, num_conv=n_conv - 2, upscale=2 if native2 else 4)
        net.load_state_dict(sd, strict=True)

        class _X2(torch.nn.Module):
            def __init__(self, n):
                super().__init__()
                self.net = n

            def forward(self, x):          # 0..255 RGB, NCHW
                y = self.net(x / 255.0).clamp(0.0, 1.0)
                return (y if native2 else F.avg_pool2d(y, 2)) * 255.0

        m = _X2(net).eval()
        x = torch.rand(1, 3, in_h, in_w) * 255.0
        ml = ct.convert(torch.jit.trace(m, x),
                        inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16)],
                        outputs=[ct.TensorType(name="y", dtype=np.float16)],
                        convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                        compute_units=ct.ComputeUnit.CPU_AND_NE,
                        minimum_deployment_target=ct.target.macOS13)
        tmp = path.with_name(path.stem + ".partial.mlpackage")
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        ml.save(str(tmp))
        os.replace(tmp, path)
    return ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)


def upscale_tiles(predict, frame: np.ndarray, in_h: int, in_w: int, tiles) -> np.ndarray:
    """x2 output of ``frame`` (H, W, 3 uint8) from per-tile ``predict``
    (float16 NCHW 0..255 -> float NCHW 0..255 at x2)."""
    h, w = frame.shape[:2]
    out = np.empty((2 * h, 2 * w, 3), dtype=np.uint8)
    chw = np.ascontiguousarray(frame.transpose(2, 0, 1)).astype(np.float16)
    for wy, wx, y0, y1, x0, x1 in tiles:
        y = predict(chw[None, :, wy:wy + in_h, wx:wx + in_w])[0]
        core = y[:, 2 * (y0 - wy):2 * (y1 - wy), 2 * (x0 - wx):2 * (x1 - wx)]
        out[2 * y0:2 * y1, 2 * x0:2 * x1] = np.clip(np.rint(core), 0, 255).astype(np.uint8).transpose(1, 2, 0)
    return out


def _worker(conn, weights: str, h: int, w: int, cache: str) -> None:
    try:
        in_h, in_w, tiles = tile_grid(h, w)
        model = _build_model(weights, in_h, in_w, cache)

        def predict(x):
            return model.predict({"x": x})["y"]

        conn.send(("ready", None))
    except Exception as exc:  # pragma: no cover - platform specific
        conn.send(("error", repr(exc)))
        return
    import queue

    inbox: queue.Queue = queue.Queue()

    def receive():
        # Read frames as soon as they come, so the sender never waits for
        # the tile loop.
        try:
            while True:
                msg = conn.recv()
                if msg is None:
                    break
                inbox.put((msg, conn.recv_bytes()))
        except (EOFError, OSError):
            pass
        inbox.put(None)

    threading.Thread(target=receive, daemon=True).start()
    while True:
        item = inbox.get()
        if item is None:
            return
        idx, raw = item
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        try:
            res = upscale_tiles(predict, frame, in_h, in_w, tiles)
        except Exception as exc:
            conn.send(("error", repr(exc)))
            return
        conn.send(("done", idx))
        conn.send_bytes(res.tobytes())


class AneUpscaler:
    """Worker process upscaling frames x2 on the Neural Engine. ``submit``
    and ``poll`` are called from the job thread; a receiver thread drains
    the pipe so a 25-MB result never blocks the next 6-MB frame going out."""

    def __init__(self, weights: str, h: int, w: int):
        import multiprocessing as mp
        import queue

        ctx = mp.get_context("spawn")
        self.h, self.w = h, w
        self._conn, child = ctx.Pipe(duplex=True)
        self._proc = ctx.Process(target=_worker, args=(child, weights, h, w, str(_cache_dir())),
                                 daemon=True, name="ane-upscale")
        self._proc.start()
        child.close()
        self.ready = False
        self.failed = False
        self.in_flight: deque[tuple[int, np.ndarray]] = deque()
        self.frames_done = 0
        self._events: queue.Queue = queue.Queue()
        self._rx = threading.Thread(target=self._receive, daemon=True, name="ane-rx")
        self._rx.start()

    def _receive(self) -> None:
        try:
            while True:
                kind, val = self._conn.recv()
                if kind == "done":
                    data = self._conn.recv_bytes()
                    self._events.put(("done", val, np.frombuffer(data, dtype=np.uint8).reshape(2 * self.h, 2 * self.w, 3)))
                    continue
                self._events.put((kind, val, None))
                if kind == "error":
                    return
        except (EOFError, OSError) as exc:
            self._events.put(("error", repr(exc), None))

    def _fail(self, why: str) -> None:
        if not self.failed:
            log.warning("Neural Engine upscale stopped (%s); the GPU does the remaining frames", why)
        self.failed = True

    def can_take(self) -> bool:
        if self.failed:
            return False
        if not self.ready:
            self.poll()
        return self.ready and len(self.in_flight) < IN_FLIGHT

    def submit(self, idx: int, frame: np.ndarray) -> None:
        try:
            self._conn.send(idx)
            self._conn.send_bytes(np.ascontiguousarray(frame).tobytes())
            self.in_flight.append((idx, frame))
        except Exception as exc:
            self._fail(repr(exc))
            raise

    def poll(self, block: bool = False, timeout: float = 120.0) -> list[tuple[int, np.ndarray]]:
        """Finished frames [(idx, x2 frame)]; ``block`` waits for the next
        message first (a worker silent for ``timeout`` s counts as failed)."""
        import queue

        done = []
        wait = block
        while not self.failed:
            try:
                kind, val, arr = self._events.get(timeout=timeout) if wait else self._events.get_nowait()
            except queue.Empty:
                if wait:
                    self._fail(f"no answer for {timeout:.0f} s")
                break
            wait = False
            if kind == "ready":
                self.ready = True
            elif kind == "error":
                self._fail(val)
            else:
                idx, _frame = self.in_flight.popleft()
                if idx != val:
                    self._fail(f"out of order ({val} != {idx})")
                    self.in_flight.appendleft((idx, _frame))
                    break
                done.append((idx, arr))
                self.frames_done += 1
        return done

    def take_back(self) -> list[tuple[int, np.ndarray]]:
        """Frames the (failed) worker still held, to be redone on the GPU."""
        back = list(self.in_flight)
        self.in_flight.clear()
        return back

    def close(self) -> None:
        try:
            if not self.failed:
                self._conn.send(None)
        except Exception:
            pass
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2)
        try:
            self._conn.close()
        except Exception:
            pass


def run_split(frames, gpu_upscale, write, weights: str, h: int, w: int,
              check_cancel=None, progress=None, ane: Optional["AneUpscaler"] = None) -> int:
    """Upscale ``frames`` x2, the GPU (``gpu_upscale``) and the ANE worker
    side by side, calling ``write`` in frame order. Returns the number of
    frames the ANE did. ``ane``: a worker started earlier (it loads while
    the GPU model does)."""
    import time

    ane = ane or AneUpscaler(weights, h, w)
    # A second Neural Engine process: while one post-processes its tiles on
    # the CPU, the other's predict keeps the engine busy.
    extra = [AneUpscaler(weights, h, w) for _ in range(max(0, ANE_PROCS - 1))]
    workers = [ane] + extra
    t0 = time.monotonic()
    t_ready = None
    ready: dict[int, np.ndarray] = {}
    nxt = 0
    count = 0

    def flush():
        nonlocal nxt
        while nxt in ready:
            write(ready.pop(nxt))
            nxt += 1
            if progress:
                progress(nxt)

    try:
        for i, frame in enumerate(frames):
            if check_cancel:
                check_cancel()
            free = [wk for wk in workers if wk.can_take()]
            if free:
                wk = min(free, key=lambda x: len(x.in_flight))
                if t_ready is None:
                    t_ready = time.monotonic() - t0
                try:
                    wk.submit(i, frame)
                except Exception:
                    ready[i] = gpu_upscale(frame)
            else:
                ready[i] = gpu_upscale(frame)
            for wk in workers:
                for idx, out in wk.poll():
                    ready[idx] = out
                if wk.failed:
                    for idx, f in wk.take_back():
                        ready[idx] = gpu_upscale(f)
            flush()
            count = i + 1
        for wk in workers:
            while wk.in_flight and not wk.failed:
                for idx, out in wk.poll(block=True):
                    ready[idx] = out
            for idx, f in wk.take_back():
                ready[idx] = gpu_upscale(f)
        flush()
        if nxt != count:
            raise RuntimeError(f"upscale lost frames ({nxt} written of {count})")
        n_ane = sum(wk.frames_done for wk in workers)
        log.info("upscale split: %d frames, %d on the Neural Engine (%d processes, ready after %.1f s), %.1f s",
                 count, n_ane, len(workers), t_ready if t_ready is not None else -1.0, time.monotonic() - t0)
        return n_ane
    finally:
        for wk in workers:
            wk.close()
