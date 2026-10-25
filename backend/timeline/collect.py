"""Optional "collect media" step: copy sources beside the timeline.

No transcoding: ``shutil.copy2`` keeps the original bytes and mtime.  The
timeline's clips are rewritten to the copied paths so collected projects are
self-contained when handed to another editor.

Never overwrites: a file already in the media folder is reused only when it
is the same copy (same size and mtime, i.e. a previous collect of the same
source); any other name clash gets a ``_2``/``_3`` suffix.  Copies go to a
temporary file in the same folder and are renamed into place, so an
interrupted copy (disk full) never leaves a truncated media file behind.
"""
from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Optional

from .model import Timeline
from .paths import posix_name, safe_stem, unique_filename


def _same_copy(source: Path, dest: Path) -> bool:
    try:
        if os.path.samefile(source, dest):
            return True
        a, b = source.stat(), dest.stat()
    except OSError:
        return False
    # copy2 preserves mtime; allow for coarse filesystem timestamps (FAT: 2 s).
    return a.st_size == b.st_size and abs(a.st_mtime - b.st_mtime) < 2.0


def _copy_atomic(source: Path, dest: Path) -> None:
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex[:8]}.partial")
    try:
        shutil.copy2(source, tmp)
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def collect_media(timeline: Timeline, media_dir: str | Path,
                  created: Optional[list[str]] = None) -> tuple[Timeline, list[str], str]:
    """Copy every unique source into ``media_dir`` and rewrite clip paths.

    Paths of the files this call actually copied are appended to ``created``
    (when given) so the caller can remove them if the export fails later."""
    target = Path(media_dir)
    target.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    copied: dict[str, str] = {}
    warnings: list[str] = []

    for info in timeline.unique_sources():
        source = Path(info.path)
        if not source.exists():
            warnings.append(f"Source missing, not collected: {info.path}")
            copied[info.path] = info.path
            continue
        basename = posix_name(info.path)
        stem = safe_stem(Path(basename).stem, "clip")
        suffix = Path(basename).suffix
        filename = unique_filename(f"{stem}{suffix}", used)
        dest = target / filename
        while dest.exists() and not _same_copy(source, dest):
            filename = unique_filename(f"{stem}{suffix}", used)
            dest = target / filename
        try:
            if not dest.exists():
                _copy_atomic(source, dest)
                if created is not None:
                    created.append(str(dest))
            copied[info.path] = str(dest)
        except OSError as exc:
            warnings.append(f"Could not collect {info.path}: {exc}")
            copied[info.path] = info.path

    clips = []
    for clip in timeline.clips:
        new_path = copied.get(clip.source.path, clip.source.path)
        clips.append(replace(clip, source=clip.source.with_path(new_path)))

    updated = replace(timeline, clips=clips, base_dir=str(target.parent),
                      warnings=list(dict.fromkeys(timeline.warnings + warnings)),
                      path_mode="relative")
    return updated, warnings, str(target)
