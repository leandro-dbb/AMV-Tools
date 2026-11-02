"""The autotune gives way to the user.

Measured on Windows + RTX 3090: after a relaunch, the first search waited
more than 90 s while the background bench decoded clips and ran SigLIP /
the tagger on the GPU (14.8 s cold without it). The bench now stops as soon
as a user request comes in — a search, a hover preview, an export, a
finishing job… — and resumes, where it stopped, once the app has been quiet
for ``QUIET_S`` (``autotune._background``). Its thread also runs at a lower
OS priority.

User requests are marked by :class:`UserActivityMiddleware`: every request
that changes something (POST / PUT / DELETE) and the GETs that serve what
the user is looking at; the UI's status polling does not count.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time

log = logging.getLogger(__name__)

QUIET_S = 15.0

_lock = threading.Lock()
_inflight = 0
_last_end = float("-inf")

# GETs the user waits for (results, thumbnails, previews, playback, roto…)
_USER_GET = re.compile(
    r"^/api/(search|scene/|scenes|videos/\d+/(stream|playable|source|scenes|tags)|characters/\d+/matches"
    r"|roto/|music/|derush|timeline/)")
# POSTs that are not the user waiting on something
_NOT_USER = re.compile(r"^/api/(autotune|shutdown|setup/|status)")


def is_user_request(method: str, path: str) -> bool:
    if not path.startswith("/api/") or _NOT_USER.match(path):
        return False
    if method in ("POST", "PUT", "DELETE", "PATCH"):
        return True
    return method == "GET" and bool(_USER_GET.match(path))


def begin() -> None:
    global _inflight
    with _lock:
        _inflight += 1


def end() -> None:
    global _inflight, _last_end
    with _lock:
        _inflight = max(0, _inflight - 1)
        _last_end = time.monotonic()


def active(quiet_s: float | None = None) -> bool:
    """A user request is running or ended less than ``quiet_s`` ago."""
    q = QUIET_S if quiet_s is None else quiet_s
    with _lock:
        return _inflight > 0 or time.monotonic() - _last_end < q


def reset() -> None:
    """Tests: forget past requests."""
    global _inflight, _last_end
    with _lock:
        _inflight, _last_end = 0, float("-inf")


class UserActivityMiddleware:
    """Pure ASGI (streamed responses — video playback — count until their
    last byte)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or not is_user_request(scope.get("method", ""), scope.get("path", "")):
            await self.app(scope, receive, send)
            return
        begin()
        try:
            await self.app(scope, receive, send)
        finally:
            end()


def lower_this_thread() -> None:
    """Below-normal OS priority for the calling thread (and, on Linux, the
    threads and processes it starts: they inherit its nice value)."""
    try:
        if sys.platform == "win32":
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetThreadPriority(k32.GetCurrentThread(), -1)     # THREAD_PRIORITY_BELOW_NORMAL
        elif sys.platform.startswith("linux"):
            tid = threading.get_native_id()
            os.setpriority(os.PRIO_PROCESS, tid, min(19, os.getpriority(os.PRIO_PROCESS, tid) + 10))
        # macOS: nice is per process there — the sidecar's own work would suffer
    except Exception as exc:
        log.debug("autotune priority unchanged: %s", exc)
