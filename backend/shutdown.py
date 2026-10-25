"""Bounded, orphan-free sidecar shutdown.

Stopping used to hang as long as a progress websocket was open: uvicorn waits
for every running ASGI task before exiting, with no time limit by default,
and the ``/ws/progress`` handler sat in ``queue.get()`` forever. Ending the
process now goes through :class:`SidecarServer`:

1. ``request_stop`` (start of uvicorn's shutdown): cancel the background
   work (indexing, re-embed, Search+ analysis, autotune, finish queue) and
   close the progress websockets.
2. uvicorn waits at most ``GRACE_S`` for the remaining requests.
3. ``finish``: a short wait for the worker threads, then every child process
   still alive (ffmpeg, Core ML / ANE workers, the motion process pool) is
   terminated, killed if it ignores that.

SIGTERM, Ctrl+C and ``POST /api/shutdown`` (what Electron uses, the only
option on Windows) all take that path; ``exit_process`` then ends the
process even if a worker thread is still inside a long native call.
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import subprocess
import sys
import threading
import time
import weakref
from typing import Optional

import uvicorn

log = logging.getLogger("amv.shutdown")

GRACE_S = 2.0           # uvicorn: open requests / websockets after the stop request
THREADS_WAIT_S = 1.0    # worker threads, once their cancel event is set
CHILD_WAIT_S = 1.0      # children between terminate() and kill()

_stopping = threading.Event()
_children: "weakref.WeakSet[subprocess.Popen]" = weakref.WeakSet()
_server: Optional["SidecarServer"] = None


class _TrackedPopen(subprocess.Popen):
    """``subprocess.Popen`` that remembers its instances, so the shutdown can
    reach the ffmpeg processes started anywhere in the backend (and through
    ``subprocess.run``, which builds its Popen from this module attribute)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _children.add(self)


def install_child_tracking() -> None:
    """Called by the sidecar entrypoint only, never on import (tests)."""
    if subprocess.Popen is not _TrackedPopen:
        subprocess.Popen = _TrackedPopen  # type: ignore[misc]


def stopping() -> bool:
    return _stopping.is_set()


def request_exit() -> bool:
    """Ask the running server to stop (``POST /api/shutdown``)."""
    if _server is None:
        return False
    _server.should_exit = True
    return True


def request_stop() -> None:
    """Non-blocking: flag the stop and cancel every background job."""
    if _stopping.is_set():
        return
    _stopping.set()
    log.info("shutdown requested: cancelling background work")
    from .state import get_state

    state = get_state()
    steps = [state.stop_indexing, _cancel_reembed, _stop_analysis, _cancel_autotune,
             _cancel_finish_jobs, state.close_listeners]
    for step in steps:
        try:
            step()
        except Exception as exc:
            log.warning("shutdown step %s failed: %s", getattr(step, "__name__", step), exc)


def _cancel_reembed() -> None:
    from .indexing import reembed
    reembed.cancel()


def _stop_analysis() -> None:
    from .searchplus.jobs import get_job
    get_job().stop()


def _cancel_autotune() -> None:
    from . import autotune
    autotune._cancel.set()


def _cancel_finish_jobs() -> None:
    from .finish import jobs
    manager = jobs._manager   # never build the manager (and its models) just to stop it
    if manager is None:
        return
    for job in manager.list():
        if job.get("status") in ("queued", "running"):
            manager.cancel(job["id"])


def _worker_threads() -> list[threading.Thread]:
    from . import autotune
    from .indexing import reembed
    from .searchplus.jobs import get_job
    from .state import get_state

    candidates = [getattr(get_state(), "_index_thread", None), autotune._thread,
                  get_job()._thread, getattr(reembed._job, "thread", None)]
    return [t for t in candidates if isinstance(t, threading.Thread) and t.is_alive()]


def _live_children() -> list:
    procs: list = [p for p in list(_children) if p.poll() is None]
    procs += multiprocessing.active_children()   # Core ML / ANE workers, motion pool
    return procs


def finish(threads_wait: float = THREADS_WAIT_S, child_wait: float = CHILD_WAIT_S) -> None:
    """Blocking: give the cancelled threads a moment, then end every child."""
    request_stop()
    deadline = time.monotonic() + threads_wait
    for t in _worker_threads():
        t.join(max(0.0, deadline - time.monotonic()))
    children = _live_children()
    if not children:
        return
    log.info("terminating %d child process(es)", len(children))
    for p in children:
        try:
            p.terminate()
        except Exception:
            pass
    deadline = time.monotonic() + child_wait
    for p in children:
        left = max(0.0, deadline - time.monotonic())
        try:
            if isinstance(p, subprocess.Popen):
                p.wait(left)
            else:
                p.join(left)
        except Exception:
            pass
    for p in _live_children():
        try:
            p.kill()
        except Exception:
            pass


class SidecarServer(uvicorn.Server):
    """uvicorn server whose shutdown is bounded and takes the children along."""

    def __init__(self, config: uvicorn.Config):
        global _server
        super().__init__(config)
        _server = self

    async def shutdown(self, sockets=None) -> None:
        request_stop()
        try:
            await super().shutdown(sockets)
        finally:
            finish()


def exit_process(code: int = 0) -> None:
    """End the process now: a worker thread still inside a long native call
    (or a ThreadPoolExecutor, joined at interpreter exit) must not keep it
    alive. Everything worth keeping was stopped by ``finish``."""
    logging.shutdown()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    os._exit(code)
