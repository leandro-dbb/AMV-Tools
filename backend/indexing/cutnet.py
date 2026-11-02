"""TransNetV2 shot-transition network, streamed frame by frame (optional).

TransNetV2 (Souček & Lokoč 2020, MIT, github.com/soCzech/TransNetV2) looks
at 100 consecutive 48×27 RGB frames and gives every frame the probability
that a transition happens there. It is the reference shot detector; on our
annotated anime extracts it alone finds 91-96 % of the cuts at the exact
frame where the hand-made detector finds 84-91 % (docs/reports/coupes.md).
``shotdetect`` combines both.

The PyTorch weights (``Sn4kehead/TransNetV2``, Apache-2.0, 30 MB) are
downloaded on demand from Settings > Models; without them, or without torch,
:func:`load` returns None and the cut detector keeps its hand-made rule.

Inference follows the reference: windows of 100 frames every 50 frames
(the first/last frame repeated 25 times at the ends), the middle 50
predictions of each window kept. Every frame is seen; a probability is
known ~75 frames after its frame was pushed.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

REPO_ID = "Sn4kehead/TransNetV2"
WEIGHTS = "transnetv2-pytorch-weights.pth"
IN_W, IN_H = 48, 27
WINDOW, STEP, PAD = 100, 50, 25
# windows per forward pass (GPU); a probability then comes up to
# BATCH * STEP + PAD frames late
BATCH = {"cuda": 4, "mps": 1, "cpu": 1, "coreml": 1}
if os.environ.get("AMV_CUTNET_BATCH", "").isdigit():      # measurements only
    BATCH["cuda"] = max(1, int(os.environ["AMV_CUTNET_BATCH"]))

_lock = threading.Lock()
_cache: dict = {}


def weights_path() -> Optional[str]:
    """Local path of the weights when they are in the Hugging Face cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(REPO_ID, WEIGHTS)
        return p if isinstance(p, str) and os.path.exists(p) else None
    except Exception:
        return None


def enabled() -> bool:
    """The network is used when its weights were downloaded (Settings >
    Models) and :func:`wanted`; ``AMV_CUTNET=0`` / ``1`` forces it off / on."""
    env = os.environ.get("AMV_CUTNET", "").strip()
    if env == "0" or weights_path() is None:
        return False
    return env == "1" or wanted()


# The network is turned on by itself only where the autotune measured that
# it costs at most this much on the cut pass (decode + cuts); elsewhere it is
# Settings > Indexing "Precise cuts (slower)", off by default.
AUTO_MAX_OVERHEAD = 0.10


def wanted() -> bool:
    """Autotune "cutnet" = "auto" (overhead ≤ ``AUTO_MAX_OVERHEAD``), or the
    "precise_cuts" setting. Not measured yet: on with CUDA (+3 % measured on
    an RTX 3090), off otherwise."""
    try:
        from ..state import get_state
        if get_state().settings.get("indexing", {}).get("precise_cuts"):
            return True
    except Exception:
        pass
    try:
        from ..autotune import value
        rec = value("cutnet", None)
        if rec is not None:
            return rec == "auto"
    except Exception:
        pass
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _coreml_wanted() -> bool:
    """Apple Silicon: the autotune's "cutnet_device" is "coreml" (default)."""
    try:
        from ..autotune import value
        return value("cutnet_device", "coreml") == "coreml"
    except Exception:
        return False


def _device() -> str:
    forced = os.environ.get("AMV_CUTNET_DEVICE", "").strip().lower()
    if forced:
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        import torch
        if torch.backends.mps.is_available():
            # Apple Silicon: Core ML on the Neural Engine (``Stream``), MPS
            # meanwhile, or the CPU (autotune "cutnet_device"); MPS gives
            # the CPU's probabilities to 1e-7, Core ML (fp16) to 1e-3
            from ..autotune import value
            return "cpu" if value("cutnet_device", "coreml") == "cpu" else "mps"
    except Exception:
        pass
    return "cpu"


def load():
    """(model, device) or None. One model per process, shared."""
    if not enabled():
        return None
    with _lock:
        if "model" in _cache:
            return _cache["model"]
        try:
            import torch
            dev = _device()
            m = _build(torch)
            sd = torch.load(weights_path(), map_location="cpu", weights_only=True)
            m.load_state_dict(sd)
            m.eval().to(dev)
            _cache["model"] = (m, dev)
            log.info("cut network TransNetV2 loaded on %s", dev)
        except Exception as exc:  # never break indexing over the optional net
            log.warning("cut network unavailable (%s); hand-made cut rule only", exc)
            _cache["model"] = None
        return _cache["model"]


class Stream:
    """Feed 27×48×3 uint8 frames in order (one numpy frame with
    :meth:`push`, or a block — numpy or a tensor already on the network's
    device — with :meth:`push_many`); ``single``/``many`` grow with the
    probabilities of the frames whose window has run.

    Forward passes run on a worker thread (``asynchronous``, the default
    off the CPU) so decoding and the per-frame measures go on meanwhile; one
    pass at most is in flight, so the probabilities come at most one more
    batch of windows late (``lag``) and the results do not depend on timing.
    """

    def __init__(self, model_dev, asynchronous: Optional[bool] = None, coreml: Optional[bool] = None):
        self.model, self.dev = model_dev
        self.worker = None
        if self.dev == "mps" and (_coreml_wanted() if coreml is None else coreml):
            # the whole file on one engine (results never depend on timing)
            self.worker = coreml_worker(wait=30.0)
            if self.worker is not None:
                self.dev = "coreml"
        self.batch = BATCH.get(self.dev, 1)
        self.chunks: list = []        # frames not yet dropped, global index self.base on
        self.base = 0
        self.n = 0
        self.next_win = 0             # padded-stream start of the next window
        self.single: list[float] = []
        self.many: list[float] = []
        if asynchronous is None and os.environ.get("AMV_CUTNET_ASYNC") in ("0", "1"):   # measurements
            asynchronous = os.environ["AMV_CUTNET_ASYNC"] == "1"
        self.asynchronous = (self.dev != "cpu") if asynchronous is None else asynchronous
        self.lag = self.batch * STEP if self.asynchronous else 0
        self._pending = None          # (thread, box) of the pass in flight

    def push(self, f: np.ndarray) -> None:
        self.push_many(np.asarray(f, dtype=np.uint8)[None])

    def push_many(self, frames) -> None:
        if len(frames) == 0:
            return
        self.chunks.append(frames)
        self.n += len(frames)
        if self._pending is not None and not self._pending[0].is_alive():
            self._collect()
        self._run(final=False)

    def finish(self) -> None:
        if self.n:
            self._run(final=True)
        self._collect()

    def _frames(self):
        """All kept frames as one array / tensor (concatenated once)."""
        if len(self.chunks) > 1:
            if isinstance(self.chunks[0], np.ndarray):
                self.chunks = [np.concatenate(self.chunks)]
            else:
                import torch
                self.chunks = [torch.cat([c if not isinstance(c, np.ndarray) else torch.from_numpy(c).to(self.dev)
                                          for c in self.chunks])]
        return self.chunks[0]

    def _run(self, final: bool) -> None:
        # the padded stream ends so that its length minus PAD is a multiple of STEP
        total = PAD + self.n + PAD + (STEP - self.n % STEP if self.n % STEP else 0) if final else None
        wins = []
        while True:
            s = self.next_win
            if final:
                if s + WINDOW > total:
                    break
            elif s + WINDOW > PAD + self.n:
                break
            wins.append(s)
            self.next_win += STEP
            if not final and len(wins) == self.batch:
                self._forward(wins, final)
                wins = []
        if wins:
            if final:
                self._forward(wins, final)
            else:
                # wait for a full batch (one launch, and the CUDA graph)
                self.next_win -= STEP * len(wins)
        # drop frames no future window needs
        keep_from = min(max(0, self.next_win - PAD - 1), self.n - 1)
        if keep_from > self.base:
            arr = self._frames()
            self.chunks = [arr[keep_from - self.base:]]
            self.base = keep_from

    def _forward(self, wins: list[int], final: bool) -> None:
        """Gather the windows (25 copies of the first frame before the
        stream, copies of the last one after it) and run them."""
        arr = self._frames()
        idx = np.clip(np.array(wins)[:, None] + np.arange(WINDOW)[None] - PAD, 0, self.n - 1) - self.base
        # (the windows that start before the stream come while frame 0 is
        # still kept: base is 0 then)
        ready = None
        if isinstance(arr, np.ndarray):
            x = arr[idx]
        else:
            import torch
            x = arr[torch.from_numpy(idx).to(arr.device)]
            if x.is_cuda:
                # the network's stream waits for the frames' own stream
                ready = torch.cuda.Event()
                ready.record()
        room = self.n if final else None
        if self.asynchronous:
            self._collect()
            box: dict = {"ready": ready}
            th = threading.Thread(target=self._infer, args=(x, box), daemon=True, name="cutnet")
            th.start()
            self._pending = (th, box, room)
        else:
            box = {"ready": ready}
            self._infer(x, box)
            self._store(box, room)

    def _infer(self, x, box: dict) -> None:
        import torch
        try:
            with torch.inference_mode():
                if self.dev == "cuda":
                    stream = _cuda_stream(torch)
                    with torch.cuda.stream(stream):
                        if box.get("ready") is not None:
                            stream.wait_event(box["ready"])
                        if not isinstance(x, torch.Tensor):
                            x = torch.from_numpy(np.ascontiguousarray(x)).to("cuda")
                        graph = _graph(torch, self, tuple(x.shape))
                        if graph is not None:
                            # one launch instead of ~300 kernels launched
                            # from Python (they hold the GIL the decode
                            # thread needs); same kernels, same numbers
                            graph["in"].copy_(x)
                            graph["g"].replay()
                            single, many = graph["out"]
                        else:
                            single, many = self._model(x)
                        ev = torch.cuda.Event(blocking=True)
                        ev.record(stream)
                    ev.synchronize()     # the thread sleeps (no spinning core)
                elif self.dev == "coreml":
                    box["single"], box["many"] = self.worker.call([np.ascontiguousarray(x)])
                    return
                else:
                    single, many = self._model(x)
                box["single"] = single.float().cpu().numpy()
                box["many"] = many.float().cpu().numpy()
        except BaseException as exc:      # surfaced by _collect
            box["error"] = exc

    def _model(self, x):
        import torch
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.ascontiguousarray(x))
        x = x.to(self.dev, non_blocking=True)
        if self.dev == "cuda" and _HALF:
            # half-precision convolutions: 1.75x faster on an RTX 3090, as
            # close to the float32 reference as the TF32 default (≤ 7e-4
            # on the probabilities; docs/reports/coupes.md, Performances)
            with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
                single, many = self.model(x)
        elif self.dev == "mps":
            single, many = forward_2d(self.model, x)
        else:
            single, many = self.model(x)
        return (torch.sigmoid(single[:, PAD:PAD + STEP, 0].float()),
                torch.sigmoid(many[:, PAD:PAD + STEP, 0].float()))

    def _collect(self) -> None:
        if self._pending is None:
            return
        th, box, room = self._pending
        self._pending = None
        th.join()
        self._store(box, room)

    def _store(self, box: dict, room) -> None:
        if "error" in box:
            raise box["error"]
        for a, b in zip(box["single"], box["many"]):
            r = (room - len(self.single)) if room is not None else len(a)
            self.single += a[:max(r, 0)].tolist()
            self.many += b[:max(r, 0)].tolist()


_HALF = os.environ.get("AMV_CUTNET_HALF", "1") != "0"
_streams: dict = {}


def _graph(torch, st: "Stream", shape: tuple):
    """The network's forward pass for full batches of windows captured as a
    CUDA graph (``AMV_CUTNET_GRAPH=0``: eager), built once per process on
    the network's stream; None for other shapes (last windows) or when
    capture fails."""
    if os.environ.get("AMV_CUTNET_GRAPH", "1") == "0" or shape != (st.batch, WINDOW, IN_H, IN_W, 3):
        return None
    key = ("graph", shape)
    if key in _streams:
        return _streams[key]
    g = None
    try:
        static = torch.zeros(shape, dtype=torch.uint8, device="cuda")
        for _ in range(2):                       # warm-up (cuDNN plans, autocast)
            st._model(static)
        torch.cuda.current_stream().synchronize()
        graph = torch.cuda.CUDAGraph()
        # thread_local: the decoder's thread keeps using CUDA meanwhile
        with torch.cuda.graph(graph, stream=torch.cuda.current_stream(), capture_error_mode="thread_local"):
            out = st._model(static)
        g = {"g": graph, "in": static, "out": out}
    except Exception as exc:
        log.info("cut network: no CUDA graph (%s); eager passes", exc)
    _streams[key] = g
    return g


def _cuda_stream(torch):
    """The network's own CUDA stream, at the lowest priority: decoding and
    downsizing (high-priority streams) go first, the network fills the gaps."""
    if "s" not in _streams:
        _streams["s"] = torch.cuda.Stream(priority=0)
    return _streams["s"]


def predict(frames: np.ndarray, model_dev, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Whole-array convenience (tests, benchmarks)."""
    st = Stream(model_dev, **kw)
    for f in frames:
        st.push(f)
    st.finish()
    return np.array(st.single[:len(frames)]), np.array(st.many[:len(frames)])


def forward_2d(m, inputs):
    """The network's forward pass with 2D operations only: each (2+1)D
    convolution as a 3×3 convolution over the B·T frames and its dilated
    temporal kernel as three shifted 1×1 convolutions; batch norm and the
    (1, 2, 2) pooling per frame. Same maths as ``m(inputs)`` (1e-7 apart on
    the probabilities) but 3.5× faster on MPS, whose 3D convolutions are
    slow (docs/reports/coupes.md, Performances); slower on the CPU."""
    import torch
    F = torch.nn.functional
    B, T, H, W = inputs.shape[:4]
    x = inputs.permute(0, 1, 4, 2, 3).reshape(B * T, 3, H, W).float().div_(255.0)

    def ddcnn(blk, x):
        outs = []
        for conv in (blk.Conv3D_1, blk.Conv3D_2, blk.Conv3D_4, blk.Conv3D_8):
            a, b = conv.layers
            y = F.conv2d(x, a.weight[:, :, 0], None, padding=1)
            _, c2, h, w = y.shape
            d = b.dilation[0]
            yp = F.pad(y.view(B, T, c2, h, w), (0, 0, 0, 0, 0, 0, d, d))
            k3 = b.weight[:, :, :, 0, 0]
            acc = None
            for k in range(3):
                z = F.conv2d(yp[:, k * d:k * d + T].reshape(B * T, c2, h, w), k3[:, :, k, None, None],
                             b.bias if k == 0 else None)
                acc = z if acc is None else acc + z
            outs.append(acc)
        x = torch.cat(outs, 1)
        bn = blk.bn
        x = F.batch_norm(x, bn.running_mean, bn.running_var, bn.weight, bn.bias, False, 0.0, bn.eps)
        return blk.activation(x) if blk.activation is not None else x

    feats = []
    for sd in m.SDDCNN:
        shortcut = None
        for blk in sd.DDCNN:
            x = ddcnn(blk, x)
            if shortcut is None:
                shortcut = x
        x = F.avg_pool2d(F.relu(x) + shortcut, 2)
        feats.append(x)
    c, h, w = x.shape[1:]
    fs = m.frame_sim_layer
    sim = torch.cat([f.mean(dim=[2, 3]).view(B, T, -1) for f in feats], dim=2)
    sim = F.normalize(fs.projection(sim), p=2, dim=2)
    sim = F.relu(fs.fc(m.lookup(torch.bmm(sim, sim.transpose(1, 2)), fs.lookup_window)))
    x = x.view(B, T, c, h, w).permute(0, 1, 3, 4, 2).reshape(B, T, -1)
    x = torch.cat([m.color_hist_layer(inputs), sim, x], 2)
    x = F.relu(m.fc1(x))
    return m.cls_layer1(x), m.cls_layer2(x)


# ── Core ML (Apple Neural Engine) ──────────────────────────────────────────
#
# On a Mac the GPU is busy during indexing (motion analysis, VideoToolbox
# scaling) while the Neural Engine idles: the convolution trunk of the 2D
# formulation runs there through Core ML (fp16, compiled once and cached),
# in a worker process (Core ML's predict holds the GIL, see
# ``backend.finish.coreml``); the small head (frame similarities, colour
# histograms, classifier) stays in PyTorch in that process.

COREML_VERSION = 2
_coreml: dict = {}


def coreml_worker(wait: float = 0.0):
    """The Core ML worker when it is ready, else None. Started on the first
    call, which waits up to ``wait`` seconds for it (the program is compiled
    once, ~6 s, then cached; starting the process takes a few seconds); if
    it is not ready by then, that file uses MPS."""
    from ..finish import coreml
    if not coreml.available() or os.environ.get("AMV_CUTNET_COREML", "1") == "0":
        return None
    with _lock:
        w = _coreml.get("worker")
        if w is None:
            try:
                from ..finish import registry
                cache = registry.models_root() / "coreml"
                w = coreml.Worker("backend.indexing.cutnet:coreml_handler", weights_path(), str(cache),
                                  name="coreml-cutnet")
            except Exception as exc:
                log.warning("Core ML cut network unavailable (%s)", exc)
                w = False
            _coreml["worker"] = w
    if w and not w.usable() and wait > 0:
        w.wait_ready(wait)
    return w if w and w.usable() else None


def prewarm() -> None:
    """Sidecar start (Apple Silicon): start the Core ML worker in the
    background when the network will be used, so the first file does not
    wait for it (~10 s: imports, Neural Engine load)."""
    import sys
    if sys.platform != "darwin":
        return

    def run():
        try:
            if enabled() and _coreml_wanted():
                coreml_worker()
        except Exception as exc:
            log.debug("cut network prewarm: %s", exc)

    threading.Thread(target=run, daemon=True, name="cutnet-prewarm").start()


def coreml_handler(weights: str, cache_dir: str):
    import coremltools as ct
    import torch
    F = torch.nn.functional
    m = _build(torch)
    m.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    m.eval()

    class Trunk(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.w_last = torch.nn.Parameter(net.fc1.weight[:, 256:].detach().clone(), requires_grad=False)

        def forward(self, x):                   # (WINDOW, 3, 27, 48) float 0..1
            means = []
            for sd in self.net.SDDCNN:
                shortcut = None
                for blk in sd.DDCNN:
                    outs = []
                    for conv in (blk.Conv3D_1, blk.Conv3D_2, blk.Conv3D_4, blk.Conv3D_8):
                        a, b = conv.layers
                        y = F.conv2d(x, a.weight[:, :, 0], None, padding=1)
                        d = b.dilation[0]
                        yp = F.pad(y.unsqueeze(0), (0, 0, 0, 0, 0, 0, d, d))[0]
                        k3 = b.weight[:, :, :, 0, 0]
                        acc = None
                        for k in range(3):
                            z = F.conv2d(yp[k * d:k * d + WINDOW], k3[:, :, k, None, None], b.bias if k == 0 else None)
                            acc = z if acc is None else acc + z
                        outs.append(acc)
                    x = torch.cat(outs, 1)
                    bn = blk.bn
                    x = F.batch_norm(x, bn.running_mean, bn.running_var, bn.weight, bn.bias, False, 0.0, bn.eps)
                    x = blk.activation(x) if blk.activation is not None else x
                    if shortcut is None:
                        shortcut = x
                x = F.avg_pool2d(F.relu(x) + shortcut, 2)
                means.append(x.mean(dim=[2, 3]))
            # the classifier's big product (its trunk-feature columns, 90 %
            # of the head) on the Neural Engine too
            flat = x.permute(0, 2, 3, 1).reshape(WINDOW, -1)
            return means[0], means[1], means[2], F.linear(flat, self.w_last)

    def build(tmp) -> None:
        ex = torch.rand(WINDOW, 3, IN_H, IN_W)
        with torch.no_grad():
            traced = torch.jit.trace(Trunk(m).eval(), ex)
        ml = ct.convert(traced, inputs=[ct.TensorType(name="frames", shape=ex.shape)],
                        outputs=[ct.TensorType(name=n) for n in ("m0", "m1", "m2", "last")],
                        convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                        compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.macOS13)
        ml.save(str(tmp))

    from pathlib import Path
    from ..finish.coreml import _cached
    path = _cached(Path(cache_dir) / f"transnetv2-trunk-fp16-v{COREML_VERSION}.mlpackage", build)
    units = getattr(ct.ComputeUnit, os.environ.get("AMV_COREML_UNITS", "CPU_AND_NE"), ct.ComputeUnit.CPU_AND_NE)
    trunk = ct.models.MLModel(str(path), compute_units=units)
    torch.set_num_threads(2)

    def handle(arrays):
        """uint8 (B, WINDOW, 27, 48, 3) windows -> sigmoid (B, STEP) single, many."""
        wins = arrays[0]
        singles, manys = [], []
        with torch.inference_mode():
            for w in wins:
                x = np.ascontiguousarray(w.transpose(0, 3, 1, 2), dtype=np.float32) / np.float32(255.0)
                o = trunk.predict({"frames": x})
                proj = torch.from_numpy(np.asarray(o["last"], dtype=np.float32))[None]
                fs = m.frame_sim_layer
                sim = torch.cat([torch.from_numpy(np.asarray(o[k], dtype=np.float32)) for k in ("m0", "m1", "m2")], 1)[None]
                sim = F.normalize(fs.projection(sim), p=2, dim=2)
                sim = F.relu(fs.fc(m.lookup(torch.bmm(sim, sim.transpose(1, 2)), fs.lookup_window)))
                y = torch.cat([m.color_hist_layer(torch.from_numpy(w[None].copy())), sim], 2)
                y = F.relu(F.linear(y, m.fc1.weight[:, :256], m.fc1.bias) + proj)
                singles.append(torch.sigmoid(m.cls_layer1(y)[0, PAD:PAD + STEP, 0]).numpy())
                manys.append(torch.sigmoid(m.cls_layer2(y)[0, PAD:PAD + STEP, 0]).numpy())
        return [np.stack(singles), np.stack(manys)]

    return handle


# ── TransNetV2 (PyTorch port of the reference implementation, MIT) ─────────

def _build(torch):
    nn = torch.nn
    F = torch.nn.functional

    class Conv3DConfigurable(nn.Module):
        def __init__(self, in_filters, filters, dilation_rate, use_bias=True):
            super().__init__()
            # (2+1)D convolution
            self.layers = nn.ModuleList([
                nn.Conv3d(in_filters, 2 * filters, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
                nn.Conv3d(2 * filters, filters, kernel_size=(3, 1, 1), dilation=(dilation_rate, 1, 1),
                          padding=(dilation_rate, 0, 0), bias=use_bias),
            ])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class DilatedDCNNV2(nn.Module):
        def __init__(self, in_filters, filters, activation=None):
            super().__init__()
            self.Conv3D_1 = Conv3DConfigurable(in_filters, filters, 1, use_bias=False)
            self.Conv3D_2 = Conv3DConfigurable(in_filters, filters, 2, use_bias=False)
            self.Conv3D_4 = Conv3DConfigurable(in_filters, filters, 4, use_bias=False)
            self.Conv3D_8 = Conv3DConfigurable(in_filters, filters, 8, use_bias=False)
            self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3)
            self.activation = activation

        def forward(self, x):
            x = torch.cat([self.Conv3D_1(x), self.Conv3D_2(x), self.Conv3D_4(x), self.Conv3D_8(x)], dim=1)
            x = self.bn(x)
            return self.activation(x) if self.activation is not None else x

    class StackedDDCNNV2(nn.Module):
        def __init__(self, in_filters, n_blocks, filters):
            super().__init__()
            self.DDCNN = nn.ModuleList([
                DilatedDCNNV2(in_filters if i == 1 else filters * 4, filters,
                              activation=F.relu if i != n_blocks else None) for i in range(1, n_blocks + 1)])
            self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

        def forward(self, x):
            shortcut = None
            for block in self.DDCNN:
                x = block(x)
                if shortcut is None:
                    shortcut = x
            x = F.relu(x)
            x = x + shortcut
            return self.pool(x)

    def lookup(sim, window):
        b, t = sim.shape[0], sim.shape[1]
        sim = F.pad(sim, [(window - 1) // 2, (window - 1) // 2])
        bi = torch.arange(b, device=sim.device).view(b, 1, 1).repeat(1, t, window)
        ti = torch.arange(t, device=sim.device).view(1, t, 1).repeat(b, 1, window)
        li = torch.arange(window, device=sim.device).view(1, 1, window).repeat(b, t, 1) + ti
        return sim[bi, ti, li]

    class FrameSimilarity(nn.Module):
        def __init__(self, in_filters, similarity_dim=128, lookup_window=101, output_dim=128):
            super().__init__()
            self.projection = nn.Linear(in_filters, similarity_dim, bias=True)
            self.fc = nn.Linear(lookup_window, output_dim)
            self.lookup_window = lookup_window

        def forward(self, inputs):
            x = torch.cat([torch.mean(x, dim=[3, 4]) for x in inputs], dim=1).transpose(1, 2)
            x = F.normalize(self.projection(x), p=2, dim=2)
            return F.relu(self.fc(lookup(torch.bmm(x, x.transpose(1, 2)), self.lookup_window)))

    class ColorHistograms(nn.Module):
        def __init__(self, lookup_window=101, output_dim=128):
            super().__init__()
            self.fc = nn.Linear(lookup_window, output_dim)
            self.lookup_window = lookup_window

        def forward(self, frames):
            b, t, h, w, _ = frames.shape
            f = frames.int().view(b * t, h * w, 3)
            bins = ((f[:, :, 0] >> 5) << 6) + ((f[:, :, 1] >> 5) << 3) + (f[:, :, 2] >> 5)
            bins = (bins + (torch.arange(b * t, device=frames.device) << 9).view(-1, 1)).view(-1)
            hist = torch.zeros(b * t * 512, dtype=torch.int32, device=frames.device)
            hist.scatter_add_(0, bins, torch.ones(len(bins), dtype=torch.int32, device=frames.device))
            x = F.normalize(hist.view(b, t, 512).float(), p=2, dim=2)
            return F.relu(self.fc(lookup(torch.bmm(x, x.transpose(1, 2)), self.lookup_window)))

    class TransNetV2(nn.Module):
        def __init__(self, Fi=16, L=3, S=2, D=1024):
            super().__init__()
            self.SDDCNN = nn.ModuleList(
                [StackedDDCNNV2(3, S, Fi)] +
                [StackedDDCNNV2((Fi * 2 ** (i - 1)) * 4, S, Fi * 2 ** i) for i in range(1, L)])
            self.frame_sim_layer = FrameSimilarity(sum((Fi * 2 ** i) * 4 for i in range(L)))
            self.color_hist_layer = ColorHistograms()
            self.dropout = nn.Dropout(0.5)
            self.fc1 = nn.Linear(((Fi * 2 ** (L - 1)) * 4) * 3 * 6 + 256, D)
            self.cls_layer1 = nn.Linear(D, 1)
            self.cls_layer2 = nn.Linear(D, 1)

        def forward(self, inputs):
            # uint8 [B, T, 27, 48, 3] -> float [B, 3, T, 27, 48]
            x = inputs.permute([0, 4, 1, 2, 3]).float().div_(255.0)
            feats = []
            for block in self.SDDCNN:
                x = block(x)
                feats.append(x)
            x = x.permute(0, 2, 3, 4, 1)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = torch.cat([self.frame_sim_layer(feats), x], 2)
            x = torch.cat([self.color_hist_layer(inputs), x], 2)
            x = self.dropout(F.relu(self.fc1(x)))
            return self.cls_layer1(x), self.cls_layer2(x)

    net = TransNetV2()
    net.lookup = lookup
    return net
