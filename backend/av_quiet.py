"""Keep PyAV's ffmpeg log callback off.

``av.logging.set_level(<int>)`` installs a Python log callback. A
frame-threaded decoder (HEVC above all) then logs from its own threads
while the thread that closes the codec holds the GIL: a deadlock that froze
the whole sidecar, API included. PyAV's default (no callback, ffmpeg logs
dropped) is safe — but ``torchvision.io`` sets ERROR when it is imported,
which happens as soon as the models load. ``install()`` wraps ``av.open``
so the default is restored before any container is opened.
"""
from __future__ import annotations

import functools


def quiet() -> None:
    import av
    if av.logging.get_level() is not None:
        av.logging.set_level(None)


def install() -> None:
    import av
    if getattr(av.open, "_amv_quiet", False):
        return
    original = av.open

    @functools.wraps(original)
    def open_quiet(*args, **kwargs):
        quiet()
        return original(*args, **kwargs)

    open_quiet._amv_quiet = True
    av.open = open_quiet
    quiet()
