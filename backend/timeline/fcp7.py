"""Final Cut Pro 7 XML (xmeml v4/v5) writer.

Premiere Pro imports this natively (File → Import), Final Cut Pro 7 too, and
Resolve/Vegas/AE-via-Dynamic-Link accept it.  Time values are integer frames
in the element's own timebase; NTSC is expressed as nominal timebase + the
``ntsc`` flag, exactly like FCP 7 wrote it.
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional
from .model import Timeline, TimelineOptions
from .paths import media_url, posix_name
from .timecode import is_ntsc, nominal_rate
from .xmlsafe import xml_text

_LABELS = {
    "iris", "lavender", "rose", "cerulean", "forest", "chocolate", "violet",
    "slate", "cobalt", "ocean", "mint", "olive", "tangerine", "cayenne",
    "mango", "sand", "cocoa", "aluminum",
}


def _rate_xml(rate: Fraction) -> str:
    return (
        "<rate>"
        f"<timebase>{nominal_rate(rate)}</timebase>"
        f"<ntsc>{'TRUE' if is_ntsc(rate) else 'FALSE'}</ntsc>"
        "</rate>"
    )


def _safe_label(label: str) -> str:
    text = (label or "").strip()
    return text if text.lower() in _LABELS else "Mango"


def _labels_xml(label: str) -> str:
    # No label: no <labels>, so unlabelled clips stay distinguishable from
    # the ones the user colour-tagged.
    if not (label or "").strip():
        return ""
    return (
        "<labels>"
        f"<label>{_safe_label(label)}</label>"
        f"<label2>{_safe_label(label)}</label2>"
        "</labels>"
    )


def _sample_characteristics(rate: Fraction, width: int, height: int) -> str:
    return (
        "<samplecharacteristics>"
        f"{_rate_xml(rate)}"
        f"<width>{int(width)}</width>"
        f"<height>{int(height)}</height>"
        "<anamorphic>FALSE</anamorphic>"
        "<pixelaspectratio>square</pixelaspectratio>"
        "<fielddominance>none</fielddominance>"
        "</samplecharacteristics>"
    )


_DRIVE = re.compile(r"^([A-Za-z]):/")


def xmeml_pathurl(url: str) -> str:
    """``file:///C:/a b.mp4`` → ``file://localhost/C%3a/a%20b.mp4``.

    The exact spelling Premiere Pro writes in its own xmeml exports (and
    Resolve on import/export): ``localhost`` host, drive colon encoded.
    macOS paths become ``file://localhost/Users/...``; UNC shares
    (``file://server/share/...``) are left alone."""
    if not url.startswith("file:///"):
        return url
    path = url[len("file:///"):]
    path = _DRIVE.sub(lambda m: f"{m.group(1)}%3a/", path)
    return "file://localhost/" + path


def _file_xml(file_id: str, info, timeline: Timeline) -> str:
    # Always absolute, even with "collect media" (path_mode="relative"):
    # xmeml's <pathurl> is a file:// URL and Premiere, FCP 7, Resolve and
    # Vegas don't resolve relative ones (they report the media offline).
    # After collection the URL points at the copy in <timeline>_media/, and
    # <name> carries the file name so a moved project relinks by folder.
    url = xmeml_pathurl(media_url(info.path, base_dir=timeline.base_dir, relative=False))
    duration = info.duration_frames or 0
    # Canonical FCP 7 order inside file/media/video is samplecharacteristics
    # (the file-level <duration> already carries the length).  Premiere is
    # lenient, but Resolve and FCP 7's own importer expect this layout.
    video = (
        "<video>"
        f"{_sample_characteristics(info.fps, info.width or 0, info.height or 0)}"
        "</video>"
    )
    audio = ""
    if info.has_audio:
        audio = (
            "<audio>"
            "<samplecharacteristics>"
            "<depth>16</depth><samplerate>48000</samplerate>"
            "</samplecharacteristics>"
            f"<channelcount>{max(1, info.audio_channels or 2)}</channelcount>"
            "</audio>"
        )
    return (
        f'<file id="{file_id}">'
        f"<name>{xml_text(posix_name(info.path))}</name>"
        f"<pathurl>{xml_text(url)}</pathurl>"
        f"{_rate_xml(info.fps)}"
        f"<duration>{duration}</duration>"
        f"<media>{video}{audio}</media>"
        "</file>"
    )


def _clipitem_xml(item_id: str, clip, file_ref: str, *, audio: bool) -> str:
    return (
        f'<clipitem id="{item_id}">'
        f"<name>{xml_text(clip.name)}</name>"
        f"<duration>{clip.source.duration_frames or clip.out_frame}</duration>"
        f"{_rate_xml(clip.source.fps)}"
        f"<start>{clip.record_start}</start>"
        f"<end>{clip.record_end}</end>"
        f"<in>{clip.in_frame}</in>"
        f"<out>{clip.out_frame}</out>"
        f"{file_ref}"
        f"{_labels_xml(clip.label)}"
        "<sourcetrack>"
        f"<mediatype>{'audio' if audio else 'video'}</mediatype>"
        "<trackindex>1</trackindex>"
        "</sourcetrack>"
        "</clipitem>"
    )


def dumps(timeline: Timeline, options: Optional[TimelineOptions] = None) -> str:
    del options
    if not timeline.clips:
        raise ValueError("the timeline is empty")

    # One full <file> element per unique source, referenced by every later
    # clipitem.  Reusing the id (instead of duplicating the element) is what
    # FCP 7 itself wrote and what Premiere's importer expects.
    file_ids: dict[str, str] = {}
    full_file_xml: dict[str, str] = {}
    for clip in timeline.clips:
        if clip.source.path not in file_ids:
            file_id = f"file-{len(file_ids) + 1}"
            file_ids[clip.source.path] = file_id
            full_file_xml[clip.source.path] = _file_xml(file_id, clip.source, timeline)

    video_items: list[str] = []
    audio_items: list[str] = []
    seen_file: set[str] = set()
    for i, clip in enumerate(timeline.clips, start=1):
        fid = file_ids[clip.source.path]
        if clip.source.path in seen_file:
            file_ref = f'<file id="{fid}"/>'
        else:
            file_ref = full_file_xml[clip.source.path]
            seen_file.add(clip.source.path)
        video_items.append(_clipitem_xml(f"clipitem-{i}", clip, file_ref, audio=False))
        if timeline.include_audio and clip.source.has_audio:
            # The video item above already declared the file element, so audio
            # must reference it (duplicating the <file> with the same id is
            # invalid XML for Premiere's importer).
            audio_items.append(
                _clipitem_xml(f"clipitem-{i}-a", clip, f'<file id="{fid}"/>', audio=True)
            )

    markers_xml = [
        "<marker>"
        f"<name>{xml_text(m.name)}</name>"
        f"<comment>{xml_text(m.comment)}</comment>"
        f"<in>{m.frame}</in>"
        f"<out>{m.frame}</out>"
        "</marker>"
        for m in timeline.markers
    ]

    audio_section = ""
    if audio_items:
        audio_section = (
            "<audio>"
            "<format><samplecharacteristics>"
            "<depth>16</depth><samplerate>48000</samplerate>"
            "</samplecharacteristics></format>"
            "<track>"
            + "".join(audio_items)
            + "</track>"
            "</audio>"
        )

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!DOCTYPE xmeml>\n"
        '<xmeml version="5">\n'
        '  <sequence id="sequence-1">\n'
        f"    <name>{xml_text(timeline.title)}</name>\n"
        f"    <duration>{timeline.duration_frames}</duration>\n"
        f"    {_rate_xml(timeline.sequence_fps)}\n"
        "    <timecode>\n"
        f"      {_rate_xml(timeline.sequence_fps)}\n"
        f"      <string>{'00:00:00;00' if timeline.drop_frame_active else '00:00:00:00'}</string>\n"
        "      <frame>0</frame>\n"
        f"      <displayformat>{'DF' if timeline.drop_frame_active else 'NDF'}</displayformat>\n"
        "    </timecode>\n"
        "    <media>\n"
        "      <video>\n"
        "        <format>"
        + _sample_characteristics(timeline.sequence_fps, timeline.width, timeline.height)
        + "</format>\n"
        "        <track>"
        + "".join(video_items)
        + "</track>\n"
        "      </video>\n"
        + audio_section
        + "\n    </media>\n"
        + "".join(markers_xml)
        + "\n  </sequence>\n"
        "</xmeml>\n"
    )
