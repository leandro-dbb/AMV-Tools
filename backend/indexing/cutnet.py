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
BATCH = {"cuda": 4, "mps": 1, "cpu": 1}

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
    Models) and ``AMV_CUTNET`` is not ``0``."""
    return os.environ.get("AMV_CUTNET", "1") != "0" and weights_path() is not None


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
    # MPS: TransNetV2's ops fall back to the CPU inconsistently there (the
    # reference port warns about different results), so Macs use the CPU
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
    """Feed 27×48×3 uint8 frames in order; get (single, many) probability
    arrays for the frames whose window has run."""

    def __init__(self, model_dev):
        self.model, self.dev = model_dev
        self.batch = BATCH.get(self.dev, 1)
        self.frames: list = []        # frames not yet dropped (global index self.base)
        self.base = 0
        self.n = 0
        self.next_win = 0             # padded-stream start of the next window
        self.single: list[float] = []
        self.many: list[float] = []
        self.first = None

    def push(self, f: np.ndarray) -> None:
        if self.first is None:
            self.first = f
        self.frames.append(f)
        self.n += 1
        self._run(final=False)

    def finish(self) -> None:
        if self.n:
            self._run(final=True)

    def _padded(self, i: int, final: bool):
        """Frame ``i`` of the padded stream (25 copies of the first frame,
        the frames, copies of the last one)."""
        j = i - PAD
        if j < 0:
            return self.first
        if j >= self.n:
            return self.frames[-1] if final else None
        return self.frames[j - self.base]

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
            self._forward(wins, final)
        # drop frames no future window needs
        keep_from = max(0, self.next_win - PAD - 1)
        while self.base < keep_from and len(self.frames) > 1:
            self.frames.pop(0)
            self.base += 1

    def _forward(self, wins: list[int], final: bool) -> None:
        import torch
        x = np.stack([np.stack([self._padded(s + k, final) for k in range(WINDOW)]) for s in wins])
        with torch.inference_mode():
            single, many = self.model(torch.from_numpy(x).to(self.dev))
            single = torch.sigmoid(single[:, PAD:PAD + STEP, 0]).float().cpu().numpy()
            many = torch.sigmoid(many[:, PAD:PAD + STEP, 0]).float().cpu().numpy()
        for a, b in zip(single, many):
            room = self.n - len(self.single) if final else len(a)
            self.single += a[:room].tolist()
            self.many += b[:room].tolist()


def predict(frames: np.ndarray, model_dev) -> tuple[np.ndarray, np.ndarray]:
    """Whole-array convenience (tests, benchmarks)."""
    st = Stream(model_dev)
    for f in frames:
        st.push(f)
    st.finish()
    return np.array(st.single[:len(frames)]), np.array(st.many[:len(frames)])


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

    return TransNetV2()
