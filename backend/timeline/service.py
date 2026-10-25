"""High-level timeline build/export service shared by the API and tests."""
from __future__ import annotations

import os
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

from . import edl, fcp7, fcpxml, otio
from .collect import collect_media
from .model import (
    ClipSpec,
    Timeline,
    TimelineOptions,
    build_timeline,
    timeline_summary,
)
from .probe import SourceInfo, probe_source

FORMATS: dict[str, dict] = {
    "fcp7xml": {
        "id": "fcp7xml",
        "extension": ".xml",
        "label": "FCP 7 XML (xmeml)",
        "label_fr": "FCP 7 XML (xmeml)",
        "apps": ["Premiere Pro", "After Effects (Dynamic Link)", "DaVinci Resolve", "Vegas Pro", "Final Cut Pro 7"],
        "description": "Premiere, Resolve, Vegas and AE via Premiere/Dynamic Link.",
    },
    "fcpxml": {
        "id": "fcpxml",
        "extension": ".fcpxml",
        "label": "FCPXML 1.9",
        "label_fr": "FCPXML 1.9",
        "apps": ["Final Cut Pro", "DaVinci Resolve"],
        "description": "Final Cut Pro 10.4+ and Resolve.",
    },
    "edl": {
        "id": "edl",
        "extension": ".edl",
        "label": "CMX3600 EDL",
        "label_fr": "EDL CMX3600",
        "apps": ["Premiere Pro", "After Effects", "DaVinci Resolve", "Final Cut Pro", "Vegas Pro"],
        "description": "Universal cut list; reel names + FROM CLIP NAME comments.",
    },
    "otio": {
        "id": "otio",
        "extension": ".otio",
        "label": "OpenTimelineIO",
        "label_fr": "OpenTimelineIO",
        "apps": ["DaVinci Resolve", "Kdenlive", "OpenTimelineIO tools"],
        "description": "Interchange format for Resolve and Kdenlive.",
    },
}

_WRITERS = {
    "fcp7xml": fcp7.dumps,
    "fcpxml": fcpxml.dumps,
    "edl": edl.dumps,
    "otio": otio.dumps,
}


class SourceMissingError(RuntimeError):
    def __init__(self, paths: list[str]):
        self.paths = list(paths)
        super().__init__("source file(s) missing: " + ", ".join(self.paths))


def _write_atomic(path: Path, content: str) -> None:
    """Write via a temporary file in the same folder, then rename."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def formats() -> list[dict]:
    return [dict(FORMATS[key]) for key in ("fcp7xml", "fcpxml", "edl", "otio")]


def probe_specs(specs: Iterable[ClipSpec]) -> tuple[dict[str, SourceInfo], list[str]]:
    sources: dict[str, SourceInfo] = {}
    missing: list[str] = []
    for spec in specs:
        if spec.source_path in sources or spec.source_path in missing:
            continue
        try:
            sources[spec.source_path] = probe_source(spec.source_path)
        except FileNotFoundError:
            missing.append(spec.source_path)
        except Exception as exc:
            raise RuntimeError(f"could not probe {spec.source_path}: {exc}") from exc
    if missing:
        raise SourceMissingError(missing)
    return sources, []


def prepare_timeline(specs: Iterable[ClipSpec], options: TimelineOptions) -> Timeline:
    spec_list = list(specs)
    if not spec_list:
        raise ValueError("no clips selected")
    sources, _ = probe_specs(spec_list)
    return build_timeline(spec_list, sources, options)


def preview_timeline(specs: Iterable[ClipSpec], options: TimelineOptions) -> dict:
    timeline = prepare_timeline(specs, options)
    return timeline_summary(timeline)


def export_timeline(
    specs: Iterable[ClipSpec],
    fmt: str,
    output_path: str,
    options: Optional[TimelineOptions] = None,
) -> dict:
    if fmt not in _WRITERS:
        raise ValueError(f"unknown timeline format: {fmt!r}")
    opts = options or TimelineOptions()
    output = Path(output_path)
    if output.suffix == "":
        output = output.with_suffix(FORMATS[fmt]["extension"])
    output.parent.mkdir(parents=True, exist_ok=True)

    timeline = prepare_timeline(specs, opts)
    timeline = replace(timeline, base_dir=str(output.parent), path_mode=opts.path_mode)

    collected_dir: Optional[str] = None
    collect_warnings: list[str] = []
    media_dir = output.parent / f"{output.stem}_media"
    media_dir_existed = media_dir.exists()
    created: list[str] = []
    try:
        if opts.collect_media:
            timeline, collect_warnings, collected_dir = collect_media(
                timeline, media_dir, created=created)
            timeline = replace(
                timeline,
                warnings=list(dict.fromkeys(timeline.warnings + collect_warnings)),
                base_dir=str(output.parent),
                path_mode="relative",
            )

        content = _WRITERS[fmt](timeline, opts)
        _write_atomic(output, content)
    except BaseException:
        # Don't leave this run's media copies behind a timeline that was
        # never written (files that were already there are untouched).
        for path in created:
            Path(path).unlink(missing_ok=True)
        if opts.collect_media and not media_dir_existed:
            try:
                media_dir.rmdir()
            except OSError:
                pass
        raise
    summary = timeline_summary(timeline)
    summary.update({
        "ok": True,
        "format": fmt,
        "output": str(output),
        "collected_dir": collected_dir,
    })
    return summary
