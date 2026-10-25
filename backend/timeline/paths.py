"""Cross-platform path → URL conversion for timeline files.

NLEs expect a properly percent-encoded ``file://`` URL.  ``Path.as_uri()``
does the right thing on the host OS, but the exporters must also handle a
Windows path written from Linux (tests, shared projects) and UNC shares, so
the conversion is explicit here.
"""
from __future__ import annotations

import ntpath
import os
import re
from pathlib import Path, PurePosixPath
from urllib.parse import quote

_WIN_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")
_UNC = re.compile(r"^\\\\[^\\]+\\[^\\]+")


def is_windows_path(path: str) -> bool:
    return bool(_WIN_DRIVE.match(path) or _UNC.match(path))


def to_file_url(path: str | os.PathLike, *, windows: bool | None = None) -> str:
    """Return a percent-encoded ``file://`` URL for a local path."""
    raw = os.fspath(path)
    if windows is None:
        windows = is_windows_path(raw)
    if windows:
        # Normalise separators first: file:///C:/... or file://server/share/...
        normalised = raw.replace("\\", "/")
        if normalised.startswith("//"):
            # UNC: //server/share/... -> file://server/share/...
            return "file:" + quote(normalised, safe="/:._-~")
        drive = normalised.lstrip("/")
        return "file:///" + quote(drive, safe="/:._-~")
    # POSIX: always absolute in the URL, even if the caller handed us a
    # relative path (NLEs cannot resolve relative file:// URLs).
    absolute = os.path.abspath(raw)
    return "file://" + quote(absolute, safe="/:._-~")


def relative_url(path: str | os.PathLike, base_dir: str | os.PathLike) -> str:
    """Percent-encoded path relative to ``base_dir`` (FCPXML/OTIO friendly).

    Windows paths are resolved with ``ntpath`` on every host, so a project
    written from Linux (tests, shared libraries) gives the same result.
    Raises ``ValueError`` when no relative path exists (other drive)."""
    raw, base = os.fspath(path), os.fspath(base_dir)
    if is_windows_path(raw) and is_windows_path(base):
        rel = ntpath.relpath(raw, base).replace("\\", "/")
    else:
        rel = os.path.relpath(raw, base)
        rel = rel.replace(os.sep, "/")
        if os.sep != "/":
            rel = rel.replace("\\", "/")
    return quote(rel, safe="/:._-~")


def media_url(path: str | os.PathLike, *, base_dir: str | os.PathLike | None,
              relative: bool = False, windows: bool | None = None) -> str:
    if relative and base_dir is not None:
        try:
            return relative_url(path, base_dir)
        except ValueError:
            # e.g. a source that could not be collected and lives on another
            # drive: an absolute URL still opens on this machine.
            pass
    return to_file_url(path, windows=windows)


def safe_stem(name: str, fallback: str = "clip") -> str:
    """File-name safe yet Unicode preserving (accents/Japanese survive)."""
    cleaned = re.sub(r'[\x00-\x1f<>:"/\\|?*]+', "_", name).strip(" .")
    cleaned = cleaned.replace(os.sep, "_")
    return cleaned or fallback


def unique_filename(name: str, used: set[str]) -> str:
    """``foo.mp4`` → ``foo_2.mp4`` when the name is already taken."""
    stem = Path(name).stem
    suffix = Path(name).suffix
    candidate = name
    n = 1
    while candidate.casefold() in used:
        n += 1
        candidate = f"{stem}_{n}{suffix}"
    used.add(candidate.casefold())
    return candidate


def posix_name(path: str | os.PathLike) -> str:
    """Basename that works for Windows paths even on a POSIX host."""
    raw = os.fspath(path).replace("\\", "/")
    return PurePosixPath(raw).name
