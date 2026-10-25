"""FCPXML 1.9 writer (Final Cut Pro 10.4+, DaVinci Resolve).

Times are exact rationals (``1001/24000s``); source ranges use the asset's
own frame rate, timeline offsets use the sequence frame rate, so mixed-rate
timelines survive the round-trip.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional
from .model import Timeline, TimelineOptions
from .paths import media_url, posix_name
from .timecode import (
    fcpxml_frame_duration,
    fcpxml_seconds,
    fcpxml_time,
    is_ntsc,
    nominal_rate,
)
from .xmlsafe import xml_attr, xml_text

_XML_HEADER = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    "<!DOCTYPE fcpxml>\n"
)


def _format_name(rate: Fraction, width: int, height: int) -> str:
    nominal = nominal_rate(rate)
    scan = "" if is_ntsc(rate) else ""
    return f"AMV {width}x{height} {nominal}p{scan}"


def _format_xml(fmt_id: str, rate: Fraction, width: int, height: int) -> str:
    return (
        f'<format id="{fmt_id}" name="{xml_attr(_format_name(rate, width, height))}" '
        f'frameDuration="{fcpxml_frame_duration(rate)}" '
        f'width="{int(width)}" height="{int(height)}"/>'
    )


def _tc_format(timeline: Timeline) -> str:
    return "DF" if timeline.drop_frame_active else "NDF"


def _marker_xml(marker, clip, seq_rate: Fraction) -> str:
    # A marker's start is in the asset-clip's local time, which begins at the
    # source in point (the clip's own ``start``), not at 0.
    local = max(0, marker.frame - clip.record_start)
    start = Fraction(int(clip.in_frame)) / clip.source.fps + Fraction(local) / seq_rate
    return (
        "<marker "
        f'start="{fcpxml_seconds(start)}" '
        f'duration="{fcpxml_time(1, seq_rate)}" '
        f'value="{xml_attr(marker.name)}">'
        f"<note>{xml_text(marker.comment)}</note>"
        "</marker>"
    )


def dumps(timeline: Timeline, options: Optional[TimelineOptions] = None) -> str:
    if not timeline.clips:
        raise ValueError("the timeline is empty")

    seq_rate = timeline.sequence_fps
    seq_tc = _tc_format(timeline)

    # Resources: one format per distinct (rate, resolution) source + sequence.
    formats: dict[tuple[Fraction, int, int], str] = {}
    source_format_ids: dict[str, str] = {}
    asset_ids: dict[str, str] = {}
    resources: list[str] = []

    seq_fmt_id = "r1"
    resources.append(_format_xml(seq_fmt_id, seq_rate, timeline.width, timeline.height))
    next_id = 2

    for clip in timeline.clips:
        info = clip.source
        key = (info.fps, int(info.width or timeline.width), int(info.height or timeline.height))
        if key not in formats:
            fmt_id = f"r{next_id}"
            next_id += 1
            formats[key] = fmt_id
            resources.append(_format_xml(fmt_id, key[0], key[1], key[2]))
        source_format_ids[info.path] = formats[key]
        if info.path not in asset_ids:
            asset_id = f"a{len(asset_ids) + 1}"
            asset_ids[info.path] = asset_id
            url = media_url(info.path, base_dir=timeline.base_dir,
                            relative=timeline.path_mode == "relative")
            asset_duration = info.duration_frames or clip.out_frame
            has_audio = "1" if info.has_audio else "0"
            audio_attrs = ""
            if info.has_audio:
                channels = max(1, info.audio_channels or 2)
                audio_attrs = (
                    f' hasAudio="1" audioSources="1" audioChannels="{channels}"'
                )
            else:
                audio_attrs = ' hasAudio="0"'
            resources.append(
                f'<asset id="{asset_id}" name="{xml_attr(posix_name(info.path))}" '
                f'start="0s" duration="{fcpxml_time(asset_duration, info.fps)}" '
                f'hasVideo="1"{audio_attrs} format="{source_format_ids[info.path]}">'
                f'<media-rep kind="original-media" src="{xml_attr(url)}"/>'
                "</asset>"
            )

    spine: list[str] = []
    cursor = 0
    for clip in timeline.clips:
        if clip.record_start > cursor:
            spine.append(
                f'<gap offset="{fcpxml_time(cursor, seq_rate)}" '
                f'duration="{fcpxml_time(clip.record_start - cursor, seq_rate)}" '
                'name="Gap"/>'
            )
        markers = "".join(
            _marker_xml(m, clip, seq_rate)
            for m in timeline.markers
            if clip.record_start <= m.frame < clip.record_end
        )
        spine.append(
            f'<asset-clip ref="{asset_ids[clip.source.path]}" '
            f'offset="{fcpxml_time(clip.record_start, seq_rate)}" '
            f'name="{xml_attr(clip.name)}" '
            f'duration="{fcpxml_time(clip.record_duration, seq_rate)}" '
            f'start="{fcpxml_time(clip.in_frame, clip.source.fps)}" '
            f'format="{source_format_ids[clip.source.path]}" '
            f'tcFormat="{seq_tc}">'
            f"{markers}"
            "</asset-clip>"
        )
        cursor = clip.record_end

    sequence = (
        f'<sequence format="{seq_fmt_id}" '
        f'duration="{fcpxml_time(timeline.duration_frames, seq_rate)}" '
        f'tcStart="0s" tcFormat="{seq_tc}">'
        "<spine>"
        + "".join(spine)
        + "</spine>"
        "</sequence>"
    )
    project = (
        f'<project name="{xml_attr(timeline.title)}">'
        f"{sequence}"
        "</project>"
    )
    library = (
        '<library>'
        f'<event name="{xml_attr(timeline.title)}">'
        f"{project}"
        "</event>"
        "</library>"
    )
    return (
        _XML_HEADER
        + '<fcpxml version="1.9">'
        + "<resources>"
        + "".join(resources)
        + "</resources>"
        + library
        + "</fcpxml>\n"
    )
