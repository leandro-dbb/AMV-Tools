"""OpenTimelineIO writer.

We write the OTIO JSON schema ourselves (one code path, with markers and
warning metadata), which Resolve/Kdenlive/OTIO readers parse identically.
The reference ``opentimelineio`` package (Apache-2.0) is only a dev/test
dependency: the tests load every export with it to validate the schema.

``timeline`` metadata also carries the AMV-specific bits (label, folder,
tags, source fps as an exact rational) that OTIO's core schema has no slot
for.
"""
from __future__ import annotations

import json
from typing import Optional

from .model import Timeline, TimelineOptions
from .paths import media_url, posix_name
from .timecode import convert_frames, otio_rate, rate_string


def _rational(value: int, rate: float) -> dict:
    return {
        "OTIO_SCHEMA": "RationalTime.1",
        "rate": rate,
        "value": float(value),
    }


def _time_range(start: int, duration: int, rate: float) -> dict:
    return {
        "OTIO_SCHEMA": "TimeRange.1",
        "duration": _rational(duration, rate),
        "start_time": _rational(start, rate),
    }


def _metadata(clip, timeline: Timeline) -> dict:
    return {
        "amv": {
            "label": clip.label,
            "folder": clip.folder,
            "tags": list(clip.tags),
            "scene_id": clip.scene_id,
            "item_id": clip.item_id,
            "source_fps": rate_string(clip.source.fps),
            "source_in_frame": clip.in_frame,
            "source_out_frame": clip.out_frame,
            "record_start": clip.record_start,
            "record_end": clip.record_end,
        }
    }


def _marker_dict(marker, clip, timeline: Timeline) -> dict:
    # marked_range lives in the clip's media time, like source_range: it
    # starts at the source in point, not at 0.
    src_rate = otio_rate(clip.source.fps)
    local = convert_frames(max(0, marker.frame - clip.record_start),
                           timeline.sequence_fps, clip.source.fps)
    local = min(local, max(0, clip.out_frame - clip.in_frame - 1))
    return {
        "OTIO_SCHEMA": "Marker.2",
        "metadata": {"amv": {"color": marker.color}},
        "name": marker.name,
        "marked_range": _time_range(clip.in_frame + local, 1, src_rate),
        "color": "RED" if marker.color else "GREEN",
        "comment": marker.comment,
    }


def _clip_dict(clip, timeline: Timeline) -> dict:
    src_rate = otio_rate(clip.source.fps)
    seq_rate = otio_rate(timeline.sequence_fps)
    url = media_url(clip.source.path, base_dir=timeline.base_dir,
                    relative=timeline.path_mode == "relative")
    duration = clip.source.duration_frames or clip.out_frame
    # start_time stays on the SOURCE rate (the exact source frame: rewriting
    # it on the sequence rate landed a 23.976 clip one frame off in a 25 fps
    # sequence once rendered by MLT). The duration is the PLACED duration on
    # the sequence rate, so the track (sum of durations) lines up exactly
    # with the record positions. OTIO RationalTimes carry their own rate;
    # TimeRange arithmetic rescales.
    return {
        "OTIO_SCHEMA": "Clip.2",
        "metadata": _metadata(clip, timeline),
        "name": clip.name,
        "source_range": {
            "OTIO_SCHEMA": "TimeRange.1",
            "duration": _rational(clip.record_duration, seq_rate),
            "start_time": _rational(clip.in_frame, src_rate),
        },
        "effects": [],
        "markers": [
            _marker_dict(m, clip, timeline)
            for m in timeline.markers
            if clip.record_start <= m.frame < clip.record_end
        ],
        "enabled": True,
        "color": None,
        # OTIO 0.18 moved from a single media_reference to a keyed map.  Both
        # Resolve and the reference library read this layout.
        "media_references": {
            "DEFAULT_MEDIA": {
                "OTIO_SCHEMA": "ExternalReference.1",
                "metadata": {"amv": {"source_fps": rate_string(clip.source.fps)}},
                "name": posix_name(clip.source.path),
                "available_range": _time_range(0, duration, src_rate),
                "available_image_bounds": None,
                "target_url": url,
            }
        },
        "active_media_reference_key": "DEFAULT_MEDIA",
    }


def _gap_dict(frames: int, rate: float) -> dict:
    return {
        "OTIO_SCHEMA": "Gap.1",
        "metadata": {},
        "source_range": _time_range(0, frames, rate),
        "effects": [],
        "markers": [],
        "enabled": True,
    }


def _track_dict(timeline: Timeline, kind: str) -> dict:
    seq_rate = otio_rate(timeline.sequence_fps)
    children: list[dict] = []
    cursor = 0
    for clip in timeline.clips:
        if kind == "Audio" and not clip.source.has_audio:
            continue
        if clip.record_start > cursor:
            children.append(_gap_dict(clip.record_start - cursor, seq_rate))
        children.append(_clip_dict(clip, timeline))
        cursor = clip.record_end
    if kind == "Audio" and cursor < timeline.duration_frames:
        # Audio is optional per source.  Without a trailing gap the audio
        # track is shorter than the video track, which some readers (OTIO's
        # own otiostat, Resolve) flag as an inconsistent timeline.
        children.append(_gap_dict(timeline.duration_frames - cursor, seq_rate))
    return {
        "OTIO_SCHEMA": "Track.1",
        "metadata": {"amv": {"kind": kind}},
        "name": "V1" if kind == "Video" else "A1",
        "source_range": None,
        "effects": [],
        "markers": [],
        "enabled": True,
        "children": children,
        "kind": kind,
    }


def _timeline_dict(timeline: Timeline) -> dict:
    tracks = [_track_dict(timeline, "Video")]
    if timeline.include_audio and any(c.source.has_audio for c in timeline.clips):
        tracks.append(_track_dict(timeline, "Audio"))
    return {
        "OTIO_SCHEMA": "Timeline.1",
        "metadata": {
            "amv": {
                "sequence_fps": rate_string(timeline.sequence_fps),
                "drop_frame": timeline.drop_frame_active,
                "width": timeline.width,
                "height": timeline.height,
                "warnings": list(timeline.warnings),
            }
        },
        "name": timeline.title,
        "global_start_time": _rational(0, otio_rate(timeline.sequence_fps)),
        "tracks": {
            "OTIO_SCHEMA": "Stack.1",
            "metadata": {},
            "name": "tracks",
            "source_range": None,
            "effects": [],
            "markers": [],
            "enabled": True,
            "children": tracks,
        },
    }


def dumps(timeline: Timeline, options: Optional[TimelineOptions] = None) -> str:
    if not timeline.clips:
        raise ValueError("the timeline is empty")
    # ``options.otio_use_library`` is accepted for API compatibility but
    # ignored: the library path dropped markers and warnings (single path).
    del options
    return json.dumps(_timeline_dict(timeline), ensure_ascii=False, indent=2) + "\n"
