"""Internal timeline model shared by every writer.

The model is deliberately small:

* :class:`ClipSpec` — what the API knows (source path + ms or frame range).
* :class:`PlacedClip` — the same clip after frame conversion, handles and
  placement on the sequence.
* :class:`Timeline` — sequence format, placed clips, markers, warnings.

Nothing here imports FastAPI or touches SQLite; writers can be unit-tested
with hand-built models (the golden tests do exactly that).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Iterable, Optional

from .probe import SourceInfo
from .timecode import (
    convert_frames,
    fps_label,
    is_drop_frame_rate,
    ms_to_frames,
    parse_rate,
    rate_string,
)


@dataclass
class ClipSpec:
    """A requested clip before probing/placement."""

    source_path: str
    name: str = ""
    in_frame: Optional[int] = None
    out_frame: Optional[int] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    label: str = ""
    folder: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    scene_id: Optional[int] = None
    item_id: Optional[int] = None

    def frames_for(self, info: SourceInfo) -> tuple[int, int]:
        if self.in_frame is not None and self.out_frame is not None:
            start, end = int(self.in_frame), int(self.out_frame)
        else:
            start_ms = self.start_ms if self.start_ms is not None else 0
            end_ms = self.end_ms if self.end_ms is not None else start_ms
            start = ms_to_frames(start_ms, info.fps)
            end = ms_to_frames(end_ms, info.fps)
        if end <= start:
            end = start + 1
        if info.duration_frames is not None and info.duration_frames > 0:
            duration = info.duration_frames
            start = max(0, min(start, duration - 1))
            end = max(start + 1, min(end, duration))
        else:
            start = max(0, start)
        return start, end


@dataclass
class TimelineOptions:
    sequence_fps: Optional[str] = "first"      # "first" | "24000/1001" | "25"…
    resolution: Optional[str] = "first"        # "first" | "1920x1080"
    layout: str = "sequential"                 # "sequential" | "gaps"
    gap_frames: int = 0
    handles_frames: int = 0
    drop_frame: bool = False
    markers: bool = True
    include_audio: bool = True
    collect_media: bool = False
    path_mode: str = "absolute"                # "absolute" | "relative"
    title: str = "AMV Tools timeline"
    otio_use_library: bool = True             # ignored: one hand-written OTIO path


@dataclass
class PlacedClip:
    source: SourceInfo
    in_frame: int
    out_frame: int
    record_start: int
    record_end: int
    name: str
    label: str = ""
    folder: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    scene_id: Optional[int] = None
    item_id: Optional[int] = None

    @property
    def duration_frames(self) -> int:
        return self.out_frame - self.in_frame

    @property
    def record_duration(self) -> int:
        return self.record_end - self.record_start

    @property
    def source_fps(self) -> Fraction:
        return self.source.fps


@dataclass
class Marker:
    frame: int
    name: str
    comment: str = ""
    color: str = ""


@dataclass
class Timeline:
    clips: list[PlacedClip]
    sequence_fps: Fraction
    width: int
    height: int
    title: str
    include_audio: bool = True
    drop_frame: bool = False
    markers: list[Marker] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    base_dir: Optional[str] = None
    path_mode: str = "absolute"

    @property
    def duration_frames(self) -> int:
        return max((c.record_end for c in self.clips), default=0)

    @property
    def duration_seconds(self) -> float:
        return self.duration_frames / float(self.sequence_fps)

    @property
    def drop_frame_active(self) -> bool:
        return bool(self.drop_frame and is_drop_frame_rate(self.sequence_fps))

    def unique_sources(self) -> list[SourceInfo]:
        seen: dict[str, SourceInfo] = {}
        for clip in self.clips:
            seen.setdefault(clip.source.path, clip.source)
        return list(seen.values())


def resolve_sequence_fps(value: Optional[str], first: SourceInfo) -> Fraction:
    if not value or str(value).strip().lower() in ("first", "auto", "source"):
        return first.fps
    parsed = parse_rate(value)
    if parsed is None or parsed <= 0:
        raise ValueError(f"invalid sequence frame rate: {value!r}")
    return parsed


def resolve_resolution(value: Optional[str], first: SourceInfo) -> tuple[int, int]:
    if not value or str(value).strip().lower() in ("first", "auto", "source"):
        return (first.width or 1920, first.height or 1080)
    text = str(value).strip().lower().replace("×", "x")
    if "x" not in text:
        raise ValueError(f"invalid resolution: {value!r}")
    w, h = text.split("x", 1)
    return (int(w), int(h))


def _auto_label(spec: ClipSpec) -> str:
    return spec.label or ""


def build_timeline(specs: Iterable[ClipSpec], sources: dict[str, SourceInfo],
                   options: TimelineOptions) -> Timeline:
    spec_list = [s for s in specs]
    if not spec_list:
        raise ValueError("the timeline is empty")

    warnings: list[str] = []
    first_source = sources[spec_list[0].source_path]
    seq_fps = resolve_sequence_fps(options.sequence_fps, first_source)
    width, height = resolve_resolution(options.resolution, first_source)
    if options.drop_frame and not is_drop_frame_rate(seq_fps):
        warnings.append(
            f"Drop-frame requested at {fps_label(seq_fps)} fps; "
            "timecode stays non-drop (drop-frame only exists for 29.97/59.94)."
        )
    source_rates = {sources[spec.source_path].fps for spec in spec_list}
    if len(source_rates) > 1:
        listed = ", ".join(sorted(fps_label(rate) for rate in source_rates))
        warnings.append(
            f"Mixed source frame rates ({listed} fps); each clip is converted "
            f"to the {fps_label(seq_fps)} fps sequence without retiming."
        )

    cursor = 0
    placed: list[PlacedClip] = []
    markers: list[Marker] = []
    handle_seq_frames = max(0, int(options.handles_frames))
    gap = max(0, int(options.gap_frames)) if options.layout == "gaps" else 0

    for index, spec in enumerate(spec_list):
        info = sources[spec.source_path]
        if info.duration_frames is None:
            warnings.append(f"Unknown source duration for {info.path}; handles were not clamped.")
        if info.is_vfr:
            warnings.extend(info.warnings or [
                f"VFR source {info.path}; using average {fps_label(info.fps)} fps."
            ])
        start, end = spec.frames_for(info)
        if handle_seq_frames:
            src_handles = convert_frames(handle_seq_frames, seq_fps, info.fps)
            start = max(0, start - src_handles)
            if info.duration_frames is not None:
                end = min(info.duration_frames, end + src_handles)
            else:
                end = end + src_handles

        duration_seq = convert_frames(end - start, info.fps, seq_fps)
        if duration_seq <= 0:
            duration_seq = 1
        if index > 0 and gap:
            cursor += gap
        record_start = cursor
        record_end = record_start + duration_seq
        cursor = record_end

        name = spec.name or f"{_source_stem(info.path)} #{index + 1:04d}"
        placed.append(PlacedClip(
            source=info,
            in_frame=start,
            out_frame=end,
            record_start=record_start,
            record_end=record_end,
            name=name,
            label=_auto_label(spec),
            folder=spec.folder,
            tags=list(spec.tags),
            scene_id=spec.scene_id,
            item_id=spec.item_id,
        ))
        if options.markers:
            comment = " · ".join(spec.tags) if spec.tags else ""
            if spec.folder:
                comment = f"[{spec.folder}] " + comment
            markers.append(Marker(frame=record_start, name=name, comment=comment.strip(),
                                  color=spec.label or ""))

    return Timeline(
        clips=placed,
        sequence_fps=seq_fps,
        width=width,
        height=height,
        title=options.title or "AMV Tools timeline",
        include_audio=bool(options.include_audio),
        drop_frame=bool(options.drop_frame),
        markers=markers,
        warnings=list(dict.fromkeys(warnings)),
        path_mode=options.path_mode,
    )


def timeline_summary(timeline: Timeline) -> dict:
    return {
        "clip_count": len(timeline.clips),
        "duration_frames": timeline.duration_frames,
        "duration_seconds": round(timeline.duration_seconds, 6),
        "sequence_fps": rate_string(timeline.sequence_fps),
        "sequence_fps_decimal": float(timeline.sequence_fps),
        "drop_frame": timeline.drop_frame_active,
        "width": timeline.width,
        "height": timeline.height,
        "title": timeline.title,
        "sources": [s.path for s in timeline.unique_sources()],
        "warnings": list(timeline.warnings),
    }


def with_source_path(clip: PlacedClip, source: SourceInfo) -> PlacedClip:
    return replace(clip, source=source)


def _source_stem(path: str) -> str:
    from .paths import posix_name
    name = posix_name(path)
    return name.rsplit(".", 1)[0] if "." in name else name
