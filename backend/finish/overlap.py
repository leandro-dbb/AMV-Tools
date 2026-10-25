"""Overlap decode / inference / encode for the finishing jobs (CUDA).

The job bodies are written as a simple loop — read a frame, run the model,
write the result — so the GPU idles while ffmpeg decodes the next frame and
again while the encoder (libx264 on the CPU, or NVENC) swallows the output.
On a 1080p → 4K upscale the x264 encode alone is longer than the model.

``prefetch`` decodes ahead in a thread; ``AsyncWriter`` encodes behind in
another. Both are bounded queues so host RAM stays at a few frames, and
errors from either thread surface in the caller. Pipe I/O and the numpy
copies release the GIL, so the three stages genuinely run in parallel.
"""
from __future__ import annotations

import queue
import threading
from typing import Iterable, Iterator, Optional

_END = object()


class _Failure:
    def __init__(self, exc: BaseException):
        self.exc = exc


def prefetch(frames: Iterable, depth: int = 4) -> Iterator:
    """Iterate ``frames`` from a background thread, ``depth`` items ahead."""
    q: queue.Queue = queue.Queue(maxsize=max(1, depth))
    stop = threading.Event()

    def run():
        try:
            for f in frames:
                while not stop.is_set():
                    try:
                        q.put(f, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
            q.put(_END)
        except BaseException as exc:  # surfaced in the consumer
            q.put(_Failure(exc))

    t = threading.Thread(target=run, daemon=True, name="finish-prefetch")
    t.start()
    try:
        while True:
            item = q.get()
            if item is _END:
                return
            if isinstance(item, _Failure):
                raise item.exc
            yield item
    finally:
        stop.set()


class AsyncWriter:
    """Wrap a ``FrameWriter``: ``write`` enqueues, a thread encodes."""

    def __init__(self, writer, depth: int = 4):
        self.writer = writer
        self._q: queue.Queue = queue.Queue(maxsize=max(1, depth))
        self._error: Optional[BaseException] = None
        self._t = threading.Thread(target=self._run, daemon=True, name="finish-writer")
        self._t.start()

    def _run(self):
        while True:
            item = self._q.get()
            if item is _END:
                return
            if self._error is not None:
                continue  # drain so the producer never blocks forever
            try:
                self.writer.write(item)
            except BaseException as exc:
                self._error = exc

    def write(self, frame) -> None:
        if self._error is not None:
            raise self._error
        self._q.put(frame)

    @property
    def frames_written(self) -> int:
        return self.writer.frames_written

    @property
    def accepts_i420(self) -> bool:
        return bool(getattr(self.writer, "accepts_i420", False))

    def close(self) -> str:
        self._q.put(_END)
        self._t.join()
        if self._error is not None:
            self.writer.abort()
            raise self._error
        return self.writer.close()

    def abort(self) -> None:
        self._error = self._error or RuntimeError("aborted")
        # a full queue must not keep _END out (the thread would wait forever
        # on get()): drop the pending frames first, they won't be written
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        try:
            self._q.put_nowait(_END)
        except queue.Full:
            pass
        self.writer.abort()
        self._t.join(timeout=5)
