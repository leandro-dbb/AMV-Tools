"""CMX3600 EDL writer.

CMX3600 is the lowest common denominator: every NLE can import it, but it
has no concept of media paths, mixed frame rates or per-clip frame rates.
Source timecodes are therefore expressed on the sequence timebase, and the
real path / source fps / tags are carried as ``*`` comments (the convention
Premiere and Resolve already understand).

Lines are kept at or under 80 columns, as the CMX3600 spec requires; long
paths are chunked over continuation comment lines.
"""
from __future__ import annotations

import re
from typing import Optional

from .model import Timeline, TimelineOptions
from .paths import posix_name, relative_url
from .timecode import convert_frames, frames_to_timecode

_REEL_SAFE = re.compile(r"[^A-Z0-9_]+")
_MAX_LINE = 80


def _reel_name(source_path: str, index: int, used: set[str]) -> str:
    stem = posix_name(source_path)
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    base = _REEL_SAFE.sub("_", stem.upper()).strip("_") or f"R{index:03d}"
    base = base[:8]
    candidate = base
    n = 1
    while candidate in used:
        n += 1
        suffix = str(n)
        candidate = (base[: 8 - len(suffix)] + suffix)[:8]
    used.add(candidate)
    return candidate


def _source_path_comment(clip, timeline: Timeline) -> str:
    if timeline.path_mode == "relative" and timeline.base_dir:
        try:
            return relative_url(clip.source.path, timeline.base_dir)
        except ValueError:
            pass
    return clip.source.path


def _comment_lines(prefix: str, value: str) -> list[str]:
    """Wrap a ``*`` comment so no line exceeds the 80-column EDL limit."""
    text = f"{prefix} {value}".rstrip()
    if len(text) <= _MAX_LINE:
        return [text]
    lines: list[str] = []
    current = prefix
    for token in value.split(" "):
        candidate = f"{current} {token}" if current else token
        if len(candidate) <= _MAX_LINE:
            current = candidate
            continue
        if current.strip():
            lines.append(current.rstrip())
        while len(token) > _MAX_LINE - 2:
            lines.append("* " + token[: _MAX_LINE - 2])
            token = token[_MAX_LINE - 2:]
        current = "* " + token
    if current.strip():
        lines.append(current.rstrip())
    return lines


def _title_line(title: str) -> str:
    line = f"TITLE: {title}"
    return line if len(line) <= _MAX_LINE else line[: _MAX_LINE - 3] + "..."


def dumps(timeline: Timeline, options: Optional[TimelineOptions] = None) -> str:
    del options
    if not timeline.clips:
        raise ValueError("the timeline is empty")

    seq_rate = timeline.sequence_fps
    drop = timeline.drop_frame_active
    fcm = "DROP FRAME" if drop else "NON-DROP FRAME"

    lines = [
        _title_line(timeline.title),
        f"FCM: {fcm}",
        "",
    ]

    reels: dict[str, str] = {}
    used_reels: set[str] = set()
    event = 0
    for clip in timeline.clips:
        if clip.source.path not in reels:
            reels[clip.source.path] = _reel_name(
                clip.source.path, len(reels) + 1, used_reels)
        reel = reels[clip.source.path]

        # EDL timecodes are on the sequence timebase.  Source in/out are
        # converted from the clip's own rate, never guessed from ms.
        src_in = convert_frames(clip.in_frame, clip.source.fps, seq_rate)
        src_out = convert_frames(clip.out_frame, clip.source.fps, seq_rate)
        rec_in = clip.record_start
        rec_out = clip.record_end
        tc_in = frames_to_timecode(src_in, seq_rate, drop)
        tc_out = frames_to_timecode(src_out, seq_rate, drop)
        rec_tc_in = frames_to_timecode(rec_in, seq_rate, drop)
        rec_tc_out = frames_to_timecode(rec_out, seq_rate, drop)

        event += 1
        lines.append(
            f"{event:03d}  {reel:<8} V     C        "
            f"{tc_in} {tc_out} {rec_tc_in} {rec_tc_out}"
        )
        lines.extend(_comment_lines("* FROM CLIP NAME:", clip.name))
        lines.extend(_comment_lines("* FROM FILE:", _source_path_comment(clip, timeline)))
        lines.extend(_comment_lines("* SOURCE FPS:", f"{float(clip.source.fps):.6f}"))
        if clip.label:
            lines.extend(_comment_lines("* LABEL:", clip.label))
        if clip.folder:
            lines.extend(_comment_lines("* FOLDER:", clip.folder))
        if clip.tags:
            lines.extend(_comment_lines("* TAGS:", ", ".join(clip.tags)))

        if timeline.include_audio and clip.source.has_audio:
            # A single A line is the CMX way of saying "the clip's audio".
            event += 1
            lines.append(
                f"{event:03d}  {reel:<8} A     C        "
                f"{tc_in} {tc_out} {rec_tc_in} {rec_tc_out}"
            )
            lines.extend(_comment_lines("* FROM CLIP NAME:", clip.name))
    return "\n".join(lines) + "\n"
