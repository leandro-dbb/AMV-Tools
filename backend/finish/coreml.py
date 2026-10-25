"""Finishing models through Core ML on Apple Silicon, in worker processes.

PyTorch MPS dispatches these nets op by op; Core ML compiles the whole
graph for the GPU. Measured on an M1 Pro, same weights:

- RIFE 4.25, 1080p pair: 187 ms (MPS fp32) -> 91 ms (Core ML GPU fp32),
  output identical (PSNR 124.7 dB);
- Depth Anything V2 Small, 518x924: 179 ms (MPS fp32) -> 123 ms (Core ML GPU
  fp32, min PSNR 121 dB vs MPS). Core ML fp16 is faster (102 ms) but only
  44.9 dB in the worst frame (41.7 dB on the finished clip): not kept. The
  Neural Engine is slower and below 40 dB (365 ms, 35.5 dB): GPU only.

Each model runs in its own spawned process: Core ML's ``predict`` holds the
GIL, which would stall the sidecar (and any PyTorch work) for the duration
of every call. Converted programs are cached next to the model weights,
keyed by the input shape; a job never waits for a conversion — it runs on
MPS until its worker reports ready (a conversion takes ~3-8 s per input
shape).

Optional: needs ``coremltools`` (macOS ``mps`` extra). ``AMV_COREML=0``
turns it off.
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)


def available() -> bool:
    if sys.platform != "darwin" or os.environ.get("AMV_COREML", "1") == "0":
        return False
    return importlib.util.find_spec("coremltools") is not None


def _cached(path: Path, build: Callable[[Path], None]) -> Path:
    """``path`` (an .mlpackage), building it through a temporary name so an
    interrupted conversion never leaves a half-written program behind."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + f".partial{os.getpid()}.mlpackage")
        shutil.rmtree(tmp, ignore_errors=True)
        build(tmp)
        try:
            os.replace(tmp, path)
        except OSError:          # another process got there first
            shutil.rmtree(tmp, ignore_errors=True)
    return path


# ── worker process ──────────────────────────────────────────────────────────
def _serve(conn, loader: str, args: tuple) -> None:
    import queue

    try:
        mod, fn = loader.split(":")
        handle = getattr(importlib.import_module(mod), fn)(*args)
        conn.send(("ready", None, None))
    except Exception as exc:  # pragma: no cover - platform specific
        conn.send(("error", None, repr(exc)))
        return
    inbox: queue.Queue = queue.Queue()

    def receive():
        try:
            while True:
                msg = conn.recv()
                if msg is None:
                    break
                rid, metas = msg
                arrays = [np.frombuffer(conn.recv_bytes(), dtype=dt).reshape(sh) for dt, sh in metas]
                inbox.put((rid, arrays))
        except (EOFError, OSError):
            pass
        inbox.put(None)

    threading.Thread(target=receive, daemon=True).start()
    while True:
        item = inbox.get()
        if item is None:
            return
        rid, arrays = item
        try:
            outs = [np.ascontiguousarray(o) for o in handle(arrays)]
        except Exception as exc:
            conn.send(("error", rid, repr(exc)))
            return
        conn.send(("done", rid, [(o.dtype.str, o.shape) for o in outs]))
        for o in outs:
            conn.send_bytes(o.tobytes())


class Worker:
    """A spawned process serving ``loader(*args)``'s handler: a function
    ``list[ndarray] -> list[ndarray]``. Calls are answered in order."""

    def __init__(self, loader: str, *args, name: str = "coreml"):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe(duplex=True)
        self._proc = ctx.Process(target=_serve, args=(child, loader, args), daemon=True, name=name)
        self._proc.start()
        child.close()
        self.name = name
        self._t0 = time.monotonic()
        self.ready = False
        self.failed: Optional[str] = None
        self._cv = threading.Condition()
        self._results: dict[int, list[np.ndarray]] = {}
        self._next = 0
        self._send_lock = threading.Lock()
        threading.Thread(target=self._receive, daemon=True, name=f"{name}-rx").start()

    def _receive(self) -> None:
        try:
            while True:
                kind, rid, val = self._conn.recv()
                if kind == "done":
                    outs = [np.frombuffer(self._conn.recv_bytes(), dtype=dt).reshape(sh) for dt, sh in val]
                    with self._cv:
                        self._results[rid] = outs
                        self._cv.notify_all()
                    continue
                with self._cv:
                    if kind == "ready":
                        self.ready = True
                        log.info("%s worker ready after %.1f s", self.name, time.monotonic() - self._t0)
                    else:
                        self.failed = val or "worker error"
                    self._cv.notify_all()
                if kind == "error":
                    return
        except (EOFError, OSError) as exc:
            with self._cv:
                self.failed = self.failed or repr(exc)
                self._cv.notify_all()

    def usable(self) -> bool:
        return self.ready and not self.failed

    def wait_ready(self, timeout: float) -> bool:
        with self._cv:
            self._cv.wait_for(lambda: self.ready or self.failed, timeout)
        return self.usable()

    def submit(self, arrays: list[np.ndarray]) -> int:
        with self._send_lock:
            rid = self._next
            self._next += 1
            arrays = [np.ascontiguousarray(a) for a in arrays]
            self._conn.send((rid, [(a.dtype.str, a.shape) for a in arrays]))
            for a in arrays:
                self._conn.send_bytes(a.tobytes())
        return rid

    def result(self, rid: int, timeout: float = 300.0) -> list[np.ndarray]:
        with self._cv:
            if not self._cv.wait_for(lambda: rid in self._results or self.failed, timeout):
                self.failed = f"no answer for {timeout:.0f} s"
            if rid in self._results:
                return self._results.pop(rid)
        raise RuntimeError(f"{self.name}: {self.failed}")

    def call(self, arrays: list[np.ndarray], timeout: float = 300.0) -> list[np.ndarray]:
        return self.result(self.submit(arrays), timeout)

    def close(self) -> None:
        try:
            with self._send_lock:
                self._conn.send(None)
        except Exception:
            pass
        if not self.ready and not self.failed:
            # Still converting: let it finish and cache the program (it exits
            # on the queued None), the next job will start from the cache.
            return
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2)
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self):  # pragma: no cover - best effort
        try:
            if self._proc.is_alive():
                self.close()
        except Exception:
            pass


# ── RIFE 4.25 ───────────────────────────────────────────────────────────────
def rife_handler(model_dir: str, ph: int, pw: int, scale: float):
    """Interpolated frame for (a, b uint8 (H, W, 3) padded to ph x pw by
    replication, t float32 (1,)) -> uint8 (H, W, 3); fp32 like the MPS path."""
    import coremltools as ct

    def build(tmp: Path) -> None:
        import torch

        from .archs.rife import IFNet
        from .interpolate import RIFE_VARIANTS

        # The model folder is named after the registry id (one flownet each).
        variant = RIFE_VARIANTS.get(Path(model_dir).name, "4.25")
        pkl = next(iter(sorted(Path(model_dir).glob("*.pkl"))))
        sd = torch.load(pkl, map_location="cpu", weights_only=True)
        sd = {k.replace("module.", ""): v for k, v in sd.items()
              if not k.startswith(("module.teacher.", "module.caltime."))}
        net = IFNet(scale=scale, variant=variant)
        net.load_state_dict(sd, strict=True)
        net.eval()

        class _Pair(torch.nn.Module):
            def __init__(self, n):
                super().__init__()
                self.n = n

            def forward(self, a, b, t):
                return self.n(a, b, t, self.n.encode(a), self.n.encode(b))

        a = torch.rand(1, 3, ph, pw)
        b = torch.rand(1, 3, ph, pw)
        t = torch.tensor([0.5])
        ml = ct.convert(torch.jit.trace(_Pair(net).eval(), (a, b, t)),
                        inputs=[ct.TensorType(name="a", shape=a.shape), ct.TensorType(name="b", shape=b.shape),
                                ct.TensorType(name="t", shape=t.shape)],
                        outputs=[ct.TensorType(name="y")], convert_to="mlprogram",
                        compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.CPU_AND_GPU,
                        minimum_deployment_target=ct.target.macOS13)
        ml.save(str(tmp))

    path = _cached(Path(model_dir) / "coreml" / f"rife425-s{scale:g}-{ph}x{pw}-v1.mlpackage", build)
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_GPU)

    def prep(f: np.ndarray) -> np.ndarray:
        h, w = f.shape[:2]
        x = f.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        if (h, w) != (ph, pw):
            x = np.pad(x, ((0, 0), (0, 0), (0, ph - h), (0, pw - w)), mode="edge")
        return x

    def handle(arrays):
        a, b, t = arrays
        h, w = a.shape[:2]
        y = model.predict({"a": prep(a), "b": prep(b), "t": t.astype(np.float32)})["y"]
        y = np.clip(y[0, :, :h, :w], 0.0, 1.0)
        return [np.rint(y.transpose(1, 2, 0) * 255.0).astype(np.uint8)]

    return handle


# ── Depth Anything V2 Small ─────────────────────────────────────────────────
def depth_handler(model_dir: str, h: int, w: int):
    """Relative disparity for pixel_values float32 (1, 3, h, w) -> float32
    (h, w), as ``DepthEstimator.predict`` returns it."""
    import coremltools as ct

    def build(tmp: Path) -> None:
        import torch
        from transformers import AutoModelForDepthEstimation

        model = AutoModelForDepthEstimation.from_pretrained(model_dir).eval()
        emb = model.backbone.embeddings
        with torch.no_grad():
            dummy = torch.zeros(1, (h // 14) * (w // 14) + 1, emb.position_embeddings.shape[-1])
            fixed = emb.interpolate_pos_encoding(dummy, h, w).detach().clone()
        # the input size is fixed: a constant instead of the traced resize
        emb.interpolate_pos_encoding = lambda e, _h, _w: fixed.to(e.device)

        class _Depth(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(pixel_values=x).predicted_depth

        x = torch.rand(1, 3, h, w)
        ep = torch.export.export(_Depth(model).eval(), (x,)).run_decompositions({})
        ml = ct.convert(ep, inputs=[ct.TensorType(name="x", shape=x.shape)],
                        outputs=[ct.TensorType(name="y")], convert_to="mlprogram",
                        compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.CPU_AND_GPU,
                        minimum_deployment_target=ct.target.macOS13)
        ml.save(str(tmp))

    path = _cached(Path(model_dir) / "coreml" / f"depth-anything-v2-small-{h}x{w}-fp32-v1.mlpackage", build)
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_GPU)

    def handle(arrays):
        y = model.predict({"x": arrays[0].astype(np.float32)})["y"]
        return [np.asarray(y, dtype=np.float32).reshape(y.shape[-2:])]

    return handle


# ── wd-tagger (indexing; lives here so its worker imports nothing heavy) ────
def tagger_handler(config_path: str, weights_path: str, cache_dir: str, batch: int = 8):
    import coremltools as ct

    def build(tmp: Path) -> None:
        import json

        import timm
        import torch
        from safetensors.torch import load_file

        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        net = timm.create_model(cfg["architecture"], num_classes=cfg["num_classes"],
                                pretrained=False, **cfg.get("model_args", {}))
        net.load_state_dict(load_file(weights_path))
        net.eval()
        mean = (cfg.get("pretrained_cfg") or {}).get("mean") or [0.5, 0.5, 0.5]
        std = (cfg.get("pretrained_cfg") or {}).get("std") or [0.5, 0.5, 0.5]

        class _Tagger(torch.nn.Module):
            def __init__(self, n):
                super().__init__()
                self.n = n
                self.register_buffer("mu", torch.tensor(mean).view(1, 3, 1, 1) * 255.0)
                self.register_buffer("sd", torch.tensor(std).view(1, 3, 1, 1) * 255.0)

            def forward(self, x):            # (N, 448, 448, 3) BGR 0..255, like the ONNX input
                x = x.permute(0, 3, 1, 2)
                return torch.sigmoid(self.n((x - self.mu) / self.sd))

        x = torch.rand(batch, 448, 448, 3) * 255.0
        ml = ct.convert(torch.jit.trace(_Tagger(net).eval(), x),
                        inputs=[ct.TensorType(name="x", shape=x.shape)], outputs=[ct.TensorType(name="y")],
                        convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                        compute_units=ct.ComputeUnit.CPU_AND_GPU, minimum_deployment_target=ct.target.macOS13)
        ml.save(str(tmp))

    # One program per checkpoint: the HF cache folder names the repo
    # (".../models--SmilingWolf--wd-eva02-large-tagger-v3/snapshots/<sha>/").
    repo = Path(config_path).parent.parent.parent.name.replace("models--", "").split("--")[-1]
    name = "wd-vit-tagger-v3" if repo in ("", "wd-vit-tagger-v3") else repo
    path = _cached(Path(cache_dir) / f"{name}-b{batch}-fp16-v1.mlpackage", build)
    # AMV_COREML_UNITS (measurements): ALL / CPU_AND_GPU / CPU_AND_NE / CPU_ONLY
    units = getattr(ct.ComputeUnit, os.environ.get("AMV_COREML_UNITS", "CPU_AND_GPU"), ct.ComputeUnit.CPU_AND_GPU)
    model = ct.models.MLModel(str(path), compute_units=units)

    def handle(arrays):
        x = arrays[0].astype(np.float32)
        n = len(x)
        if n < batch:
            x = np.concatenate([x, np.repeat(x[-1:], batch - n, axis=0)])
        return [np.asarray(model.predict({"x": x})["y"], dtype=np.float32)[:n]]

    return handle
