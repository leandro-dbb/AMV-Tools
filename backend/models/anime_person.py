"""Anime person detector (deepghs/anime_person_detection, YOLOv8, MIT).

A small ONNX model trained on anime illustrations: boxes around each
character. Used by Mask Mode to tell characters from other salient things
(an aura, a vehicle, a piece of set) before the outline is drawn."""
from __future__ import annotations

import logging
import threading
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

REPO_ID = "deepghs/anime_person_detection"
MODEL = "person_detect_v1.3_s"
THRESHOLD = 0.324   # the model's F1-optimal threshold (threshold.json)
INFER_SIZE = 640


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float = 0.7) -> List[int]:
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        o = inter / (area(boxes[i]) + area(boxes[rest]) - inter + 1e-6)
        order = rest[o < iou]
    return keep


class AnimePersonDetector:
    def __init__(self, model_path: Optional[str] = None, hf_token: Optional[str] = None,
                 providers: Optional[list] = None):
        self.model_path = model_path
        self.hf_token = hf_token
        self.providers = providers
        self._sess = None
        self._lock = threading.Lock()

    def _ensure(self):
        if self._sess is not None:
            return
        with self._lock:
            if self._sess is not None:
                return
            import onnxruntime as ort
            path = self.model_path
            if path is None:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(REPO_ID, f"{MODEL}/model.onnx", token=self.hf_token)
            providers = self.providers or [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider",
                                                       "CPUExecutionProvider")
                                           if p in ort.get_available_providers()]
            self._sess = ort.InferenceSession(path, providers=providers)

    def detect(self, rgb: np.ndarray, threshold: float = THRESHOLD) -> List[Tuple[float, float, float, float, float]]:
        """``[(x0, y0, x1, y1, score), ...]`` in the frame's pixels."""
        import cv2
        self._ensure()
        h, w = rgb.shape[:2]
        s = INFER_SIZE / max(h, w)
        nw, nh = int(round(w * s)), int(round(h * s))
        pw, ph = (nw + 31) // 32 * 32, (nh + 31) // 32 * 32
        img = np.full((ph, pw, 3), 114, np.uint8)
        img[:nh, :nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        x = img.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out = self._sess.run(None, {self._sess.get_inputs()[0].name: x})[0][0]
        if out.shape[0] == 5:
            out = out.T                       # (N, 5): cx, cy, w, h, score
        sc = out[:, 4]
        ok = sc >= threshold
        if not ok.any():
            return []
        b, sc = out[ok, :4], sc[ok]
        boxes = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                          b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1) / s
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return [(*map(float, boxes[i]), float(sc[i])) for i in _nms(boxes, sc)]
