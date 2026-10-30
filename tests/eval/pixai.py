"""PixAI tagger v0.9 (deepghs ONNX export, Apache-2.0) for the evaluation.

Preprocessing per ``preprocess.json`` of deepghs/pixai-tagger-v0.9-onnx:
RGB on white, stretched to 448x448 (bilinear), (x - 0.5) / 0.5, NCHW.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO = "deepghs/pixai-tagger-v0.9-onnx"

# deepghs category ids -> the wd-tagger convention used by the app
_CATS = {"general": 0, "character": 4, "rating": 9, "copyright": 3, "artist": 1, "meta": 5}


class PixAITagger:
    def __init__(self, model: str, backend: str):
        import onnxruntime as ort
        from huggingface_hub import snapshot_download

        d = Path(snapshot_download(REPO))
        self.dir = d
        providers = []
        if backend == "cuda":
            providers.append("CUDAExecutionProvider")
        elif backend == "mps":
            providers.append(("CoreMLExecutionProvider", {"ModelFormat": "MLProgram",
                                                          "MLComputeUnits": "CPUAndGPU"}))
        providers.append("CPUExecutionProvider")
        so = None
        self.static = None
        if backend == "mps":
            # Core ML needs a pinned batch (a dynamic one aborts, see wd_tagger)
            so = ort.SessionOptions()
            so.add_free_dimension_override_by_name("batch_size", 8)
            self.static = 8
        self.sess = ort.InferenceSession(str(d / "model.onnx"), sess_options=so, providers=providers)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.nchw = inp.shape[1] == 3
        self.size = 448
        self.out_name = self.sess.get_outputs()[0].name
        for o in self.sess.get_outputs():
            if o.name == "prediction":
                self.out_name = o.name
        with open(d / "selected_tags.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        def cat(r):
            c = r.get("category")
            try:
                return int(c)
            except (TypeError, ValueError):
                return _CATS.get(str(c), 0)
        self._labels = [(r["name"], cat(r)) for r in rows]
        self.meta = {p.name: json.loads(p.read_text()) for p in d.glob("*.json")}

    def active_providers(self):
        return self.sess.get_providers()

    def _pre(self, img: Image.Image) -> np.ndarray:
        img = img.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        x = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
        return x.transpose(2, 0, 1) if self.nchw else x

    def _run_batch(self, images):
        x = np.stack([self._pre(im) for im in images]).astype(np.float32)
        if self.static:
            outs = []
            for i in range(0, len(x), self.static):
                c = x[i:i + self.static]
                n = len(c)
                if n < self.static:
                    c = np.concatenate([c, np.repeat(c[-1:], self.static - n, axis=0)])
                outs.append(self.sess.run([self.out_name], {self.input_name: c})[0][:n])
            y = np.concatenate(outs)
        else:
            y = self.sess.run([self.out_name], {self.input_name: x})[0]
        if y.min() < 0 or y.max() > 1:
            y = 1 / (1 + np.exp(-y))
        return y


# ── PixAI tagger v1.0 (chantier 4b) ─────────────────────────────────────────
# pixai-labs/pixai-tagger-v1.0: Apache-2.0 on the card, but a finetuned SAM 3
# backbone (Meta "SAM License", non-commercial use allowed). 486 M params,
# 1008x1008 input (aspect kept, zero padding), 30 877 tags. The repo's own
# model code (``tagger_pipeline.py``) is loaded from the download folder
# (``PIXAI_V1_DIR``) without going through transformers' remote-code path.
_V1_CATS = {"general": 0, "character": 4, "copyright": 3, "style": 1, "meta": 5, "rating": 9}


class PixAIV1Tagger:
    def __init__(self, model: str, backend: str):
        import ast
        import importlib.util
        import os

        import torch

        d = Path(os.environ["PIXAI_V1_DIR"])
        spec = importlib.util.spec_from_file_location("pixai_v1_pipeline", d / "tagger_pipeline.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        raw = json.loads((d / "config.json").read_text())
        # transformers' own loader (a hand-built model + load_state_dict
        # gave wrong scores: Goku 0/12)
        net = mod.ViTDetCls.from_pretrained(str(d))
        tags = net.config.tags
        tags = ast.literal_eval(tags) if isinstance(tags, str) else tags
        self.dev = torch.device(backend)
        self.dtype = torch.bfloat16 if backend == "cuda" else (torch.float16 if backend == "mps" else torch.float32)
        # device only: ``.to(dtype)`` would also cast the complex RoPE
        # buffers (freqs_cis) to real and wreck the model; bf16 via autocast
        self.net = net.eval().to(self.dev)
        self.size = int(raw.get("img_size", 1008))
        self.mod = mod
        self._labels = []
        for cat, count in net.config.tags_split:
            self._labels += [(None, _V1_CATS.get(cat, 0))] * count
        self._labels = [(t, c) for t, (_, c) in zip(tags, self._labels)]
        self.thresholds = raw.get("category_best_threshold")

    def active_providers(self):
        return [f"torch-{self.dev.type}-{str(self.dtype).split('.')[-1]}"]

    def _run_batch(self, images):
        import torch
        import torchvision.transforms.functional as vF
        xs = [vF.to_tensor(im.convert("RGB")).to(self.dev) for im in images]
        xs = [vF.normalize(self.mod.rescale_pad(x, self.size), [0.5] * 3, [0.5] * 3) for x in xs]
        with torch.inference_mode(), torch.autocast(self.dev.type, dtype=self.dtype,
                                                    enabled=self.dtype != torch.float32):
            y = self.net(torch.stack(xs)).float().sigmoid()
        return y.cpu().numpy()
