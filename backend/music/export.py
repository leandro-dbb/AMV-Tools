"""Music edit → editor timeline, markers-only files.

Full timeline
    The shots are placed with :func:`backend.timeline.build_timeline` (the
    regular "send to your editor" export, untouched), then every record
    position is pinned to the frame nearest to its music time — so cuts stay
    on the beats however the source frame rates round — and the music is
    added as its own audio track by a small post-processing step per format.
    Section / drop / beat markers ride along.

Markers only
    The song alone on a sequence with its beats and sections as markers, to
    cut by hand in Premiere / Resolve / Vegas / After Effects.
"""
from __future__ import annotations

import csv
import io
import json
import os
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Optional

from ..timeline import edl, fcp7, fcpxml, otio
from ..timeline.model import ClipSpec, Marker, Timeline, TimelineOptions, build_timeline
from ..timeline.paths import media_url, posix_name
from ..timeline.probe import probe_source
from ..timeline.timecode import (
    fcpxml_time,
    frames_to_timecode,
    is_ntsc,
    ms_to_frames,
    nominal_rate,
    otio_rate,
    parse_rate,
    rate_string,
)
from ..timeline.xmlsafe import xml_attr, xml_text

TIMELINE_FORMATS = ("fcp7xml", "fcpxml", "edl", "otio")
MARKER_FORMATS = TIMELINE_FORMATS + ("csv",)
EXTENSIONS = {"fcp7xml": ".xml", "fcpxml": ".fcpxml", "edl": ".edl", "otio": ".otio", "csv": ".csv"}
DEFAULT_MARKER_FPS = "24000/1001"

SECTION_NAMES = {
    "en": {"intro": "Intro", "verse": "Verse", "build": "Build-up", "drop": "DROP",
           "chorus": "Chorus", "bridge": "Calm part", "outro": "Outro"},
    "fr": {"intro": "Intro", "verse": "Couplet", "build": "Montée", "drop": "DROP",
           "chorus": "Refrain", "bridge": "Passage calme", "outro": "Outro"},
}
# Resolve marker colours for the marker EDL.
_RESOLVE_COLORS = {"intro": "ResolveColorBlue", "verse": "ResolveColorGreen",
                   "build": "ResolveColorYellow", "drop": "ResolveColorRed",
                   "chorus": "ResolveColorPink", "bridge": "ResolveColorCyan",
                   "outro": "ResolveColorBlue", "beat": "ResolveColorSand",
                   "bar": "ResolveColorCream"}


@dataclass
class MusicMarker:
    frame: int
    duration: int
    kind: str      # section type, "beat" or "bar"
    name: str
    comment: str = ""


def _write_atomic(path: Path, content: str) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(content, encoding="utf-8", newline="")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def output_path(raw: str, fmt: str) -> Path:
    out = Path(raw).expanduser()
    if out.suffix == "":
        out = out.with_suffix(EXTENSIONS[fmt])
    if out.exists() and out.is_dir():
        raise ValueError(f"output is a folder: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


# ── markers ─────────────────────────────────────────────────────────────────
def music_markers(analysis: dict, fps: Fraction, *, beat_markers: bool, lang: str = "en") -> list[MusicMarker]:
    names = SECTION_NAMES.get(lang, SECTION_NAMES["en"])
    bpm = analysis.get("bpm")
    out: list[MusicMarker] = []
    for s in analysis.get("sections") or []:
        a = ms_to_frames(int(s["start_ms"]), fps)
        b = ms_to_frames(int(s["end_ms"]), fps)
        energy = int(round(100 * float(s.get("energy") or 0.0)))
        comment = (f"{names.get(s['type'], s['type'])} · {energy}% · {bpm} BPM"
                   if lang != "fr" else f"{names.get(s['type'], s['type'])} · énergie {energy} % · {bpm} BPM")
        out.append(MusicMarker(a, max(1, b - a), s["type"], names.get(s["type"], s["type"]), comment))
    if beat_markers:
        section_frames = {m.frame for m in out}
        downbeats = {int(d) for d in analysis.get("downbeats_ms") or []}
        meter = int(analysis.get("meter") or 4)
        beats = [int(t) for t in analysis.get("beats_ms") or []]
        # Beat number in the bar, counted from the first downbeat (a pickup
        # before it ends the previous bar: … 3, 4 | 1).
        first = next((i for i, t in enumerate(beats) if t in downbeats), 0)
        for i, t in enumerate(beats):
            is_bar = t in downbeats
            f = ms_to_frames(t, fps)
            if f in section_frames:
                continue
            label = str((i - first) % meter + 1)
            out.append(MusicMarker(f, 1, "bar" if is_bar else "beat", label))
    out.sort(key=lambda m: (m.frame, m.kind not in SECTION_NAMES["en"]))
    return out


def _as_timeline_markers(markers: list[MusicMarker]) -> list[Marker]:
    return [Marker(frame=m.frame, name=m.name, comment=m.comment) for m in markers]


# ── full timeline ───────────────────────────────────────────────────────────
def build_edit_timeline(analysis: dict, clips: list[dict], scene_paths: dict[int, str],
                        *, include_source_audio: bool = False, title: str = "AMV Tools music edit",
                        sequence_fps: Optional[str] = None) -> tuple[Timeline, list[str]]:
    """Timeline with each shot pinned to its music time (frame-exact)."""
    if not clips:
        raise ValueError("the edit is empty")
    ordered = sorted(clips, key=lambda c: int(c["start_ms"]))
    sources = {}
    missing = []
    for c in ordered:
        path = scene_paths.get(int(c["scene_id"]))
        if not path:
            missing.append(str(c["scene_id"]))
            continue
        if path not in sources:
            if not Path(path).is_file():
                missing.append(path)
                continue
            sources[path] = probe_source(path)
    if missing:
        raise FileNotFoundError("source file(s) missing: " + ", ".join(sorted(set(missing))))

    first = sources[scene_paths[int(ordered[0]["scene_id"])]]
    seq_fps = parse_rate(sequence_fps) if sequence_fps else first.fps
    specs: list[ClipSpec] = []
    records: list[tuple[int, int]] = []
    for c in ordered:
        path = scene_paths[int(c["scene_id"])]
        info = sources[path]
        rec_a = ms_to_frames(int(c["start_ms"]), seq_fps)
        rec_b = max(rec_a + 1, ms_to_frames(int(c["end_ms"]), seq_fps))
        src_in = ms_to_frames(int(c["src_in_ms"]), info.fps)
        # Same duration in the source's own rate (no retiming).
        src_len = max(1, int(round((rec_b - rec_a) * info.fps / seq_fps)))
        if info.duration_frames:
            src_in = max(0, min(src_in, info.duration_frames - src_len))
        specs.append(ClipSpec(source_path=path, in_frame=src_in, out_frame=src_in + src_len,
                              name=f"{c.get('video_name') or posix_name(path)} #{int(c['slot']) + 1:03d}",
                              scene_id=int(c["scene_id"]),
                              tags=[str(c.get("motion_type") or "")] if c.get("motion_type") else []))
        records.append((rec_a, rec_b))
    options = TimelineOptions(sequence_fps=rate_string(seq_fps), markers=False,
                              include_audio=include_source_audio, title=title)
    timeline = build_timeline(specs, sources, options)
    # Pin every clip on its beat: build_timeline places them back to back
    # with per-clip rounding; the music decides where each one starts.
    for placed, (a, b) in zip(timeline.clips, records):
        placed.record_start = a
        placed.record_end = b
    return timeline, list(timeline.warnings)


def _music_frames(analysis: dict, fps: Fraction) -> int:
    return max(1, ms_to_frames(int(analysis["duration_ms"]), fps))


def _fcp7_music_file(music_path: str, frames: int, fps: Fraction) -> str:
    rate = (f"<rate><timebase>{nominal_rate(fps)}</timebase>"
            f"<ntsc>{'TRUE' if is_ntsc(fps) else 'FALSE'}</ntsc></rate>")
    return (
        '<file id="file-music">'
        f"<name>{xml_text(posix_name(music_path))}</name>"
        f"<pathurl>{xml_text(media_url(music_path, base_dir=None))}</pathurl>"
        f"{rate}<duration>{frames}</duration>"
        "<media><audio><channelcount>2</channelcount>"
        "<samplecharacteristics><depth>16</depth><samplerate>48000</samplerate></samplecharacteristics>"
        "</audio></media></file>"
    ), rate


def _fcp7_music_track(music_path: str, frames: int, fps: Fraction) -> str:
    file_xml, rate = _fcp7_music_file(music_path, frames, fps)
    return (
        "<track>"
        '<clipitem id="clipitem-music">'
        f"<name>{xml_text(posix_name(music_path))}</name>"
        f"<duration>{frames}</duration>{rate}"
        f"<start>0</start><end>{frames}</end><in>0</in><out>{frames}</out>"
        f"{file_xml}"
        "<sourcetrack><mediatype>audio</mediatype><trackindex>1</trackindex></sourcetrack>"
        "</clipitem>"
        "</track>"
    )


def _add_music_fcp7(xml: str, music_path: str, frames: int, fps: Fraction) -> str:
    root = ET.fromstring(xml.split("\n", 2)[2] if xml.startswith("<?xml") else xml)
    media = root.find("./sequence/media")
    audio = media.find("audio")
    if audio is None:
        audio = ET.SubElement(media, "audio")
        audio.append(ET.fromstring(
            "<format><samplecharacteristics><depth>16</depth><samplerate>48000</samplerate>"
            "</samplecharacteristics></format>"))
    track = ET.fromstring(_fcp7_music_track(music_path, frames, fps))
    # The song goes on A1, above any episode audio.
    fmt_index = 1 if audio.find("format") is not None else 0
    audio.insert(fmt_index, track)
    seq = root.find("./sequence")
    dur = seq.find("duration")
    if dur is not None and int(dur.text or 0) < frames:
        dur.text = str(frames)
    body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n' + body + "\n"


def _fcpxml_markers(markers: list[MusicMarker], fps: Fraction) -> str:
    return "".join(
        f'<marker start="{fcpxml_time(m.frame, fps)}" duration="{fcpxml_time(m.duration, fps)}" '
        f'value="{xml_attr(m.name)}">'
        f"<note>{xml_text(m.comment)}</note></marker>"
        for m in markers
    )


def _fcpxml_music_asset(music_path: str, frames: int, fps: Fraction) -> str:
    return (
        f'<asset id="a-music" name="{xml_attr(posix_name(music_path))}" start="0s" '
        f'duration="{fcpxml_time(frames, fps)}" hasVideo="0" hasAudio="1" audioSources="1" '
        'audioChannels="2" audioRate="48000">'
        f'<media-rep kind="original-media" src="{xml_attr(media_url(music_path, base_dir=None))}"/>'
        "</asset>"
    )


def _add_music_fcpxml(xml: str, music_path: str, frames: int, fps: Fraction,
                      markers: list[MusicMarker]) -> str:
    header, _, rest = xml.partition("<fcpxml")
    root = ET.fromstring("<fcpxml" + rest)
    root.find("resources").append(ET.fromstring(_fcpxml_music_asset(music_path, frames, fps)))
    spine = root.find(".//sequence/spine")
    first = next(iter(spine), None)
    if first is None:
        raise ValueError("the timeline is empty")
    # Connected to the first spine item at its own start: plays from 0 on
    # the sequence, below the picture (lane -1), and carries the markers.
    parent_start = first.get("start", "0s")
    music = ET.fromstring(
        f'<asset-clip ref="a-music" lane="-1" offset="{parent_start}" name="{xml_attr(posix_name(music_path))}" '
        f'duration="{fcpxml_time(frames, fps)}" start="0s" audioRole="music">'
        f"{_fcpxml_markers(markers, fps)}</asset-clip>")
    first.insert(0, music)
    seq = root.find(".//sequence")
    seq.set("duration", fcpxml_time(max(frames, _fcpxml_frames(seq.get("duration"), fps)), fps))
    return header + ET.tostring(root, encoding="unicode") + "\n"


def _fcpxml_frames(value: Optional[str], fps: Fraction) -> int:
    if not value:
        return 0
    v = value.rstrip("s")
    seconds = Fraction(v) if "/" in v else Fraction(v)
    return int(round(seconds * fps))


def _otio_music_track(music_path: str, frames: int, fps: Fraction, markers: list[MusicMarker]) -> dict:
    rate = otio_rate(fps)

    def rt(v: int) -> dict:
        return {"OTIO_SCHEMA": "RationalTime.1", "rate": rate, "value": float(v)}

    def tr(a: int, d: int) -> dict:
        return {"OTIO_SCHEMA": "TimeRange.1", "start_time": rt(a), "duration": rt(d)}

    color = {"drop": "RED", "build": "YELLOW", "chorus": "PINK", "bar": "ORANGE", "beat": "BLUE"}
    return {
        "OTIO_SCHEMA": "Track.1",
        "metadata": {"amv": {"kind": "Audio", "role": "music"}},
        "name": "Music",
        "source_range": None,
        "effects": [],
        "markers": [],
        "enabled": True,
        "kind": "Audio",
        "children": [{
            "OTIO_SCHEMA": "Clip.2",
            "metadata": {},
            "name": posix_name(music_path),
            "source_range": tr(0, frames),
            "effects": [],
            "markers": [{
                "OTIO_SCHEMA": "Marker.2",
                "metadata": {"amv": {"kind": m.kind}},
                "name": m.name,
                "marked_range": tr(m.frame, m.duration),
                "color": color.get(m.kind, "GREEN"),
                "comment": m.comment,
            } for m in markers],
            "enabled": True,
            "color": None,
            "media_references": {"DEFAULT_MEDIA": {
                "OTIO_SCHEMA": "ExternalReference.1",
                "metadata": {},
                "name": posix_name(music_path),
                "available_range": tr(0, frames),
                "available_image_bounds": None,
                "target_url": media_url(music_path, base_dir=None),
            }},
            "active_media_reference_key": "DEFAULT_MEDIA",
        }],
    }


def _add_music_otio(text: str, music_path: str, frames: int, fps: Fraction,
                    markers: list[MusicMarker], timeline: Optional[Timeline] = None) -> str:
    data = json.loads(text)
    tracks = data["tracks"]["children"]
    if timeline is not None:
        # OTIO places clips back to back by their source_range, in each
        # source's own rate: with mixed frame rates the per-clip rounding
        # adds up and cuts drift off the beats.  Give every clip the exact
        # (fractional) source length of its slot on the sequence.
        seq = timeline.sequence_fps
        for track in tracks:
            items = [c for c in track.get("children", []) if c.get("OTIO_SCHEMA", "").startswith("Clip")]
            placed = [pc for pc in timeline.clips
                      if track.get("kind") != "Audio" or pc.source.has_audio]
            for item, pc in zip(items, placed):
                if pc.source.fps == seq:
                    continue
                exact = (pc.record_end - pc.record_start) * pc.source.fps / seq
                item["source_range"]["duration"]["value"] = float(exact)
    # Audio tracks after the picture; the song first among them.
    tracks.insert(1, _otio_music_track(music_path, frames, fps, markers))
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def _edl_music_event(event: int, music_path: str, frames: int, fps: Fraction, channel: str,
                     reel: str = "MUSIC") -> list[str]:
    tc_in = frames_to_timecode(0, fps)
    tc_out = frames_to_timecode(frames, fps)
    return [
        f"{event:03d}  {reel:<8} {channel:<5} C        {tc_in} {tc_out} {tc_in} {tc_out}",
        f"* FROM CLIP NAME: {posix_name(music_path)}"[:80],
        f"* FROM FILE: {music_path}"[:80] if len(music_path) < 66 else "* FROM FILE: " + posix_name(music_path)[:66],
    ]


def _add_music_edl(text: str, music_path: str, frames: int, fps: Fraction, has_source_audio: bool) -> str:
    lines = text.rstrip("\n").split("\n")
    events = [ln for ln in lines if ln[:3].isdigit()]
    last = int(events[-1][:3]) if events else 0
    # The song's reel must not reuse a source's reel name (a "music.mkv"
    # episode would get "MUSIC"): editors link events by reel.
    used = {ln.split()[1] for ln in events if len(ln.split()) > 1}
    reel = next(r for r in ["MUSIC"] + [f"MUSIC{i}" for i in range(2, 1000)] if r not in used)
    lines.extend(_edl_music_event(last + 1, music_path, frames, fps, "A2" if has_source_audio else "A", reel))
    return "\n".join(lines) + "\n"


def export_edit(fmt: str, output: str, analysis: dict, music_path: str, clips: list[dict],
                scene_paths: dict[int, str], *, beat_markers: bool = True,
                include_source_audio: bool = False, lang: str = "en",
                title: Optional[str] = None) -> dict:
    if fmt not in TIMELINE_FORMATS:
        raise ValueError(f"unknown timeline format: {fmt!r}")
    out = output_path(output, fmt)
    title = title or f"AMV — {Path(music_path).stem}"
    timeline, warnings = build_edit_timeline(analysis, clips, scene_paths,
                                             include_source_audio=include_source_audio, title=title)
    fps = timeline.sequence_fps
    frames = _music_frames(analysis, fps)
    markers = music_markers(analysis, fps, beat_markers=beat_markers, lang=lang)
    if fmt == "fcp7xml":
        # Sequence-level markers: written by the regular FCP 7 writer.
        content = _add_music_fcp7(fcp7.dumps(replace(timeline, markers=_as_timeline_markers(markers))),
                                  music_path, frames, fps)
    elif fmt == "fcpxml":
        content = _add_music_fcpxml(fcpxml.dumps(timeline), music_path, frames, fps, markers)
    elif fmt == "otio":
        content = _add_music_otio(otio.dumps(timeline), music_path, frames, fps, markers, timeline)
    else:
        content = _add_music_edl(edl.dumps(timeline), music_path, frames, fps,
                                 include_source_audio and any(c.source.has_audio for c in timeline.clips))
        warnings = warnings + ["edl_no_markers"]
    _write_atomic(out, content)
    return {"ok": True, "output": str(out), "format": fmt, "clip_count": len(timeline.clips),
            "marker_count": 0 if fmt == "edl" else len(markers),
            "sequence_fps": rate_string(fps), "warnings": warnings,
            "record_frames": [(c.record_start, c.record_end) for c in timeline.clips]}


# ── markers only ────────────────────────────────────────────────────────────
def _markers_fcp7(analysis: dict, music_path: str, fps: Fraction, markers: list[MusicMarker], title: str) -> str:
    frames = _music_frames(analysis, fps)
    rate = (f"<rate><timebase>{nominal_rate(fps)}</timebase>"
            f"<ntsc>{'TRUE' if is_ntsc(fps) else 'FALSE'}</ntsc></rate>")
    marker_xml = "".join(
        "<marker>"
        f"<name>{xml_text(m.name)}</name><comment>{xml_text(m.comment)}</comment>"
        f"<in>{m.frame}</in><out>{m.frame + m.duration if m.duration > 1 else -1}</out>"
        "</marker>"
        for m in markers
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n<xmeml version="5">\n'
        '  <sequence id="sequence-1">\n'
        f"    <name>{xml_text(title)}</name>\n    <duration>{frames}</duration>\n    {rate}\n"
        f"    <timecode>{rate}<string>00:00:00:00</string><frame>0</frame>"
        "<displayformat>NDF</displayformat></timecode>\n"
        "    <media>\n"
        "      <video><format><samplecharacteristics>"
        f"{rate}<width>1920</width><height>1080</height><anamorphic>FALSE</anamorphic>"
        "<pixelaspectratio>square</pixelaspectratio><fielddominance>none</fielddominance>"
        "</samplecharacteristics></format><track/></video>\n"
        "      <audio><format><samplecharacteristics><depth>16</depth><samplerate>48000</samplerate>"
        "</samplecharacteristics></format>"
        f"{_fcp7_music_track(music_path, frames, fps)}</audio>\n"
        "    </media>\n"
        f"    {marker_xml}\n"
        "  </sequence>\n</xmeml>\n"
    )


def _markers_fcpxml(analysis: dict, music_path: str, fps: Fraction, markers: list[MusicMarker], title: str) -> str:
    frames = _music_frames(analysis, fps)
    fmt = (f'<format id="r1" name="FFVideoFormat1080p{nominal_rate(fps)}" '
           f'frameDuration="{fcpxml_time(1, fps)}" width="1920" height="1080"/>')
    spine = (
        f'<asset-clip ref="a-music" offset="0s" name="{xml_attr(posix_name(music_path))}" '
        f'duration="{fcpxml_time(frames, fps)}" start="0s" audioRole="music">'
        f"{_fcpxml_markers(markers, fps)}</asset-clip>"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n<fcpxml version="1.9">'
        f"<resources>{fmt}{_fcpxml_music_asset(music_path, frames, fps)}</resources>"
        f'<library><event name="{xml_attr(title)}"><project name="{xml_attr(title)}">'
        f'<sequence format="r1" duration="{fcpxml_time(frames, fps)}" tcStart="0s" tcFormat="NDF">'
        f"<spine>{spine}</spine></sequence></project></event></library></fcpxml>\n"
    )


def _markers_edl(fps: Fraction, markers: list[MusicMarker], title: str) -> str:
    """DaVinci Resolve marker EDL (Timeline > Import > Timeline Markers from
    EDL).  Timecodes start at 01:00:00:00, Resolve's default timeline start."""
    # Non-drop timecode counts nominal frames: 01:00:00:00 is 3600 × 24
    # frames at 23.976, not 3600 s.
    offset = 3600 * nominal_rate(fps)
    lines = [f"TITLE: {title}"[:80], "FCM: NON-DROP FRAME", ""]
    for i, m in enumerate(markers, start=1):
        a = frames_to_timecode(offset + m.frame, fps)
        b = frames_to_timecode(offset + m.frame + max(1, m.duration), fps)
        lines.append(f"{i:03d}  001      V     C        {a} {b} {a} {b}")
        name = m.name.replace("|", "/")
        comment = m.comment.replace("|", "/")
        lines.append(f" |C:{_RESOLVE_COLORS.get(m.kind, 'ResolveColorBlue')} |M:{name} |D:{max(1, m.duration)}"
                     + (f" |N:{comment}" if comment else ""))
        lines.append("")
    return "\n".join(lines) + "\n"


def _markers_otio(analysis: dict, music_path: str, fps: Fraction, markers: list[MusicMarker], title: str) -> str:
    rate = otio_rate(fps)
    data = {
        "OTIO_SCHEMA": "Timeline.1",
        "metadata": {"amv": {"sequence_fps": rate_string(fps), "bpm": analysis.get("bpm")}},
        "name": title,
        "global_start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": rate, "value": 0.0},
        "tracks": {
            "OTIO_SCHEMA": "Stack.1", "metadata": {}, "name": "tracks", "source_range": None,
            "effects": [], "markers": [], "enabled": True,
            "children": [_otio_music_track(music_path, _music_frames(analysis, fps), fps, markers)],
        },
    }
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def _markers_csv(fps: Fraction, markers: list[MusicMarker]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["index", "seconds", "timecode", "frame", "duration_frames", "kind", "name", "comment"])
    for i, m in enumerate(markers, start=1):
        w.writerow([i, f"{float(m.frame / fps):.3f}", frames_to_timecode(m.frame, fps), m.frame,
                    m.duration, m.kind, m.name, m.comment])
    return buf.getvalue()


def export_markers(fmt: str, output: str, analysis: dict, music_path: str, *,
                   fps: Optional[str] = None, beat_markers: bool = True, lang: str = "en",
                   title: Optional[str] = None) -> dict:
    if fmt not in MARKER_FORMATS:
        raise ValueError(f"unknown marker format: {fmt!r}")
    try:
        rate = parse_rate(fps or DEFAULT_MARKER_FPS)
    except (ValueError, ZeroDivisionError):
        rate = None
    if rate is None or rate <= 0:
        raise ValueError(f"invalid frame rate: {fps!r}")
    out = output_path(output, fmt)
    title = title or f"AMV — {Path(music_path).stem}"
    markers = music_markers(analysis, rate, beat_markers=beat_markers, lang=lang)
    if fmt == "fcp7xml":
        content = _markers_fcp7(analysis, music_path, rate, markers, title)
    elif fmt == "fcpxml":
        content = _markers_fcpxml(analysis, music_path, rate, markers, title)
    elif fmt == "edl":
        content = _markers_edl(rate, markers, title)
    elif fmt == "otio":
        content = _markers_otio(analysis, music_path, rate, markers, title)
    else:
        content = _markers_csv(rate, markers)
    _write_atomic(out, content)
    return {"ok": True, "output": str(out), "format": fmt, "marker_count": len(markers),
            "sequence_fps": rate_string(rate), "warnings": []}
