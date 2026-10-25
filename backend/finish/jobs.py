"""Finishing job queue: FIFO, one worker thread, cooperative cancellation.

Jobs are executed one at a time on purpose — every finishing model wants the
whole GPU (or every CPU core), so running two at once only makes both slower
and risks OOM. Progress is fanned out through the existing ``/ws/progress``
socket as ``{"type": "finish_job", "job": {...}}`` events, so the frontend
needs no extra connection.

The queue lives in process memory (like Mask Mode sessions): finished jobs
point at files on disk, which outlive the sidecar anyway.
"""
from __future__ import annotations

import itertools
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

log = logging.getLogger(__name__)


class JobCancelled(Exception):
    """Raised inside a job body when the user cancelled it."""


@dataclass
class Job:
    kind: str
    params: dict
    label: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "queued"          # queued | running | done | failed | cancelled
    progress: float = 0.0            # 0..1
    message: str = ""
    output: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    seq: int = 0
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> dict:
        eta = None
        if self.status == "running" and self.started_at and self.progress > 0.02:
            elapsed = time.time() - self.started_at
            eta = max(0.0, elapsed / self.progress - elapsed)
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "params": self.params,
            "status": self.status,
            "progress": round(self.progress, 4),
            "percent": int(self.progress * 100),
            "message": self.message,
            "output": self.output,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "eta_seconds": None if eta is None else round(eta, 1),
        }


class JobContext:
    """Handle passed to a job body: progress reporting + cancellation."""

    def __init__(self, job: Job, publish: Callable[[Job], None]):
        self.job = job
        self._publish = publish
        self._last_pub = 0.0
        self._cleanups: list[Callable[[], None]] = []

    @property
    def cancelled(self) -> bool:
        return self.job.cancel_event.is_set()

    def check_cancel(self) -> None:
        if self.job.cancel_event.is_set():
            raise JobCancelled()

    def on_cancel_cleanup(self, fn: Callable[[], None]) -> None:
        """Register a best-effort cleanup (kill ffmpeg, delete partial file)
        that runs if the job is cancelled or fails."""
        self._cleanups.append(fn)

    def run_cleanups(self) -> None:
        for fn in reversed(self._cleanups):
            try:
                fn()
            except Exception as exc:
                log.debug("job cleanup failed: %s", exc)
        self._cleanups.clear()

    def progress(self, fraction: float, message: Optional[str] = None, *, force: bool = False) -> None:
        self.check_cancel()
        self.job.progress = max(0.0, min(1.0, float(fraction)))
        if message is not None:
            self.job.message = message
        now = time.monotonic()
        # Throttle: one event every 0.4 s is plenty for a progress bar and
        # keeps stdout (state.publish also prints) from flooding.
        if force or now - self._last_pub >= 0.4:
            self._last_pub = now
            self._publish(self.job)


Runner = Callable[[JobContext], Optional[str]]


class JobManager:
    def __init__(self, runner: Runner, publish: Callable[[dict], None], *, max_history: int = 200):
        self._runner = runner
        self._publish_raw = publish
        self._jobs: dict[str, Job] = {}
        self._queue: deque[str] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._seq = itertools.count()
        self._max_history = max_history
        self._idle_hooks: list[Callable[[], None]] = []
        self._worker = threading.Thread(target=self._loop, daemon=True, name="finish-jobs")
        self._worker.start()

    # ── public API ──────────────────────────────────────────────────────────
    def submit(self, kind: str, params: dict, label: str = "") -> Job:
        job = Job(kind=kind, params=dict(params), label=label)
        with self._wake:
            job.seq = next(self._seq)
            self._jobs[job.id] = job
            self._queue.append(job.id)
            self._prune_locked()
            self._wake.notify()
        self._publish(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.seq)
        return [j.to_dict() for j in jobs]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status not in ("queued", "running"):
                return False
            job.cancel_event.set()
            if job.status == "queued":
                try:
                    self._queue.remove(job_id)
                except ValueError:
                    pass
                job.status = "cancelled"
                job.finished_at = time.time()
        self._publish(job)
        return True

    def remove(self, job_id: str) -> bool:
        """Forget a finished job (the output file is left untouched)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in ("queued", "running"):
                return False
            del self._jobs[job_id]
        return True

    def clear_finished(self) -> int:
        with self._lock:
            ids = [j.id for j in self._jobs.values() if j.status not in ("queued", "running")]
            for i in ids:
                del self._jobs[i]
        return len(ids)

    def add_idle_hook(self, fn: Callable[[], None]) -> None:
        """Called after the queue drains (used to free model memory)."""
        self._idle_hooks.append(fn)

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Test helper: block until no job is queued/running."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                busy = bool(self._queue) or any(j.status == "running" for j in self._jobs.values())
            if not busy:
                return True
            time.sleep(0.02)
        return False

    # ── internals ───────────────────────────────────────────────────────────
    def _publish(self, job: Job) -> None:
        try:
            self._publish_raw({"type": "finish_job", "job": job.to_dict()})
        except Exception as exc:  # never let a dead socket kill a job
            log.debug("finish publish failed: %s", exc)

    def _prune_locked(self) -> None:
        done = [j for j in self._jobs.values() if j.status in ("done", "failed", "cancelled")]
        excess = len(self._jobs) - self._max_history
        for j in sorted(done, key=lambda j: j.seq)[:max(0, excess)]:
            del self._jobs[j.id]

    def _loop(self) -> None:
        while True:
            with self._wake:
                while not self._queue:
                    self._wake.wait()
                job_id = self._queue.popleft()
                job = self._jobs.get(job_id)
                if job is None or job.status != "queued":
                    continue
                job.status = "running"
                job.started_at = time.time()
            self._publish(job)
            ctx = JobContext(job, self._publish)
            try:
                out = self._runner(ctx)
                if job.cancel_event.is_set():
                    raise JobCancelled()
                job.output = out
                job.progress = 1.0
                job.status = "done"
                job.message = job.message or "done"
            except JobCancelled:
                ctx.run_cleanups()
                job.status = "cancelled"
                job.message = "cancelled"
            except Exception as exc:
                ctx.run_cleanups()
                log.exception("finish job %s (%s) failed", job.id, job.kind)
                job.status = "failed"
                job.error = str(exc) or exc.__class__.__name__
            finally:
                job.finished_at = time.time()
            self._publish(job)
            with self._lock:
                idle = not self._queue
            if idle:
                for hook in self._idle_hooks:
                    try:
                        hook()
                    except Exception as exc:
                        log.debug("idle hook failed: %s", exc)


_manager: Optional[JobManager] = None
_manager_lock = threading.Lock()


def get_manager() -> JobManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            from ..state import get_state
            from .runner import release_models, run_job

            _manager = JobManager(run_job, get_state().publish)
            _manager.add_idle_hook(release_models)
        return _manager

