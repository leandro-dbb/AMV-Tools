"""Timeline export ("send to your editor") for AMV Tools.

Plain files, no re-encode: each writer only references the original media
with frame-accurate in/out points.
"""
from .model import (
    ClipSpec,
    Marker,
    PlacedClip,
    Timeline,
    TimelineOptions,
    build_timeline,
    timeline_summary,
)
from .probe import SourceInfo, probe_source, source_summary
from .service import (
    FORMATS,
    SourceMissingError,
    export_timeline,
    formats,
    preview_timeline,
)

__all__ = [
    "ClipSpec",
    "Marker",
    "PlacedClip",
    "Timeline",
    "TimelineOptions",
    "build_timeline",
    "timeline_summary",
    "SourceInfo",
    "probe_source",
    "source_summary",
    "FORMATS",
    "SourceMissingError",
    "export_timeline",
    "formats",
    "preview_timeline",
]
