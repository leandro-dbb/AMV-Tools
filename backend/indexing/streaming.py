"""Tag + embed scene medians while the video is still being decoded.

On the single-pass software decode (Apple Silicon, see ``single_pass``) a
scene's median frame is known the moment the cut detector closes the scene.
Decoding is CPU-bound and both models run on the GPU, so feeding them from a
worker thread during the decode hides most of the inference time instead of
adding it afterwards (M1 Pro, 24-min episode: 37.6 s decode then 31 s of
tagging/embedding before this, overlapped now).

Results are keyed by median timestamp; the pipeline's tag/embed phases use
them when every scene of a batch is present and fall back to running the
models themselves otherwise (worker error, cancellation, scene list changed).
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)


class StreamingInference:
    QUEUE_SCENES = 8
    QUEUE_MEDIANS = 256

    def __init__(self, tagger, siglip, phases: Sequence[str], batch_size: int,
                 cancel_event: Optional[threading.Event] = None,
                 subseg: Optional[dict] = None):
        self.tagger_model = tagger
        self.tagger = tagger if "tag" in phases else None
        self.siglip = siglip if "embed" in phases else None
        self.batch_size = max(1, int(batch_size))
        self.cancel_event = cancel_event
        # Sub-segmentation, same parameters as the pipeline's two-pass path:
        # {"min_ms", "step_ms", "threshold"}.
        self.subseg = subseg
        self.tags: dict[int, list] = {}
        self.embs: dict[int, object] = {}
        # hard-cut start -> [(start, end, is_parent)] and the frames picked
        # for their medians (sub-segmentation only)
        self.segments: dict[int, list] = {}
        self.medians: dict[int, object] = {}
        self.error: Optional[BaseException] = None
        self._side = None   # tagger thread when the tagger runs out of process
        # Bounded: when the GPU falls behind (sub-segmentation sends every
        # sample of each scene, ~2.6 MB per second of footage) the decoder
        # waits instead of piling frames up in memory. Single medians are
        # small (~0.4 MB): a deep queue absorbs bursts of short shots —
        # 8 items there made the decoder wait (24-min episode, Fast:
        # 43 s -> 53 s).
        self._q: queue.Queue = queue.Queue(
            maxsize=self.QUEUE_SCENES if subseg else self.QUEUE_MEDIANS)
        self._thread = threading.Thread(target=self._run, daemon=True, name="index-streaming")
        self._thread.start()

    def submit(self, ts_ms: int, image) -> None:
        self._q.put((ts_ms, image))

    def submit_scene(self, start_ms: int, end_ms: int, samples: list) -> None:
        """A closed hard-cut scene with its decoded samples [(ts, rgb)]."""
        self._q.put(("scene", start_ms, end_ms, samples))

    def _split(self, start: int, end: int, samples: list) -> list:
        """Sub-segment a scene from its samples (tagger drift, exactly the
        rule of ``sub_segment.sub_segment``) -> [(s, e, is_parent)]."""
        from PIL import Image

        from .sub_segment import sub_segment

        cfg = self.subseg
        if end - start <= cfg["min_ms"] or self.error is not None or self._cancelled():
            return [(start, end, False)]
        times = np.array([t for t, _ in samples])
        grid, imgs = [], []
        for g in range(start, end, cfg["step_ms"]):
            i = int(np.argmin(np.abs(times - g)))
            grid.append(g)
            imgs.append(Image.fromarray(samples[i][1]))
        try:
            embs = []
            for k in range(0, len(imgs), self.batch_size):
                embs.extend(self.tagger_model.embed_images(imgs[k:k + self.batch_size]))
        except Exception as exc:
            log.warning("streaming sub-segmentation stopped: %s", exc)
            self.error = exc
            return [(start, end, False)]
        subs = sub_segment(start, end, list(zip(grid, embs)), drift_threshold=cfg["threshold"])
        if len(subs) <= 1:
            return [(start, end, False)]
        return [(start, end, True)] + [(a, b, False) for a, b in subs]

    def _scene(self, start: int, end: int, samples: list, pending: list) -> None:
        from PIL import Image

        segs = self._split(start, end, samples)
        self.segments[start] = segs
        times = np.array([t for t, _ in samples])
        for s, e, _parent in segs:
            target = s + (e - s) // 2
            img = Image.fromarray(samples[int(np.argmin(np.abs(times - target)))][1])
            self.medians[target] = img
            pending.append((target, img))
            if len(pending) >= self.batch_size:
                self._flush(pending)
                pending.clear()

    def finish(self) -> None:
        self._q.put(None)
        self._thread.join()
        if self._side is not None:
            self._side.shutdown(wait=True)

    def _cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    def _flush(self, pending: list) -> None:
        if not pending or self.error is not None or self._cancelled():
            return
        ts = [t for t, _ in pending]
        imgs = [im for _, im in pending]
        try:
            if (self.tagger is not None and self.siglip is not None
                    and getattr(self.tagger, "runs_out_of_process", lambda: False)()):
                # The tagger runs in its Core ML worker (Apple Silicon): SigLIP
                # works on MPS meanwhile instead of after it.
                if self._side is None:
                    from concurrent.futures import ThreadPoolExecutor
                    self._side = ThreadPoolExecutor(max_workers=1, thread_name_prefix="index-tagger")
                fut = self._side.submit(self.tagger.tag_images, imgs)
                for t, emb in zip(ts, self.siglip.embed_images(imgs)):
                    self.embs[t] = emb
                for t, tags in zip(ts, fut.result()):
                    self.tags[t] = tags
                return
            if self.tagger is not None:
                for t, tags in zip(ts, self.tagger.tag_images(imgs)):
                    self.tags[t] = tags
            if self.siglip is not None:
                for t, emb in zip(ts, self.siglip.embed_images(imgs)):
                    self.embs[t] = emb
        except Exception as exc:
            # The regular phases redo whatever is missing and report errors.
            log.warning("streaming inference stopped: %s", exc)
            self.error = exc

    def _run(self) -> None:
        pending: list = []
        while True:
            item = self._q.get()
            if item is None:
                self._flush(pending)
                return
            if item[0] == "scene":
                self._scene(item[1], item[2], item[3], pending)
                continue
            pending.append(item)
            if len(pending) >= self.batch_size:
                self._flush(pending)
                pending = []
