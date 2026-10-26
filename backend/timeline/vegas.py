"""Vegas Pro "EDL Text File" (.txt) writer.

Vegas Pro does not read CMX3600, FCP 7 XML or FCPXML out of the box; its
built-in importer is *File > Import > EDL Text File (*.txt)*, which reads
Vegas' own semicolon-separated event list (the one *File > Export > EDL Text
File* writes): one header row of quoted field names, then one row per event,
times in milliseconds.

Layout (28 fields, as exported by Vegas):

    "ID";"Track";"StartTime";"Length";"PlayRate";"Locked";"Normalized";
    "StretchMethod";"Looped";"OnRuler";"MediaType";"FileName";"Stream";
    "StreamStart";"StreamLength";"FadeTimeIn";"FadeTimeOut";"SustainGain";
    "CurveIn";"GainIn";"CurveOut";"GainOut";"Layer";"Color";"CurveInR";
    "CurveOutR";"PlayPitch";"LockPitch"

``StartTime``/``Length`` place the event on the timeline (sequence time),
``StreamStart``/``StreamLength`` are the in point and length inside the
source file. Video events go on track 0, audio events on track 1. Times are
derived from exact frame numbers (Fractions), so 23.976 material does not
drift. The file uses CRLF line endings like every Windows-native text file.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional

from .model import Timeline, TimelineOptions

FIELDS = (
    "ID", "Track", "StartTime", "Length", "PlayRate", "Locked", "Normalized",
    "StretchMethod", "Looped", "OnRuler", "MediaType", "FileName", "Stream",
    "StreamStart", "StreamLength", "FadeTimeIn", "FadeTimeOut", "SustainGain",
    "CurveIn", "GainIn", "CurveOut", "GainOut", "Layer", "Color", "CurveInR",
    "CurveOutR", "PlayPitch", "LockPitch",
)

VIDEO_TRACK = 0
AUDIO_TRACK = 1


def _ms(frames: int, rate: Fraction) -> str:
    return f"{float(Fraction(int(frames)) / rate * 1000):.6f}"


def _file_field(path: str) -> str:
    # Quoted like Vegas does; a double quote cannot appear in a Windows path
    # and is dropped defensively on other systems.
    return '"' + path.replace('"', "") + '"'


def _row(event_id: int, track: int, media_type: str, path: str,
         start: str, length: str, stream_start: str) -> str:
    values = [
        str(event_id), str(track), start, length, "1.000000", "FALSE", "FALSE",
        "0", "FALSE", "FALSE", media_type, _file_field(path), "0",
        stream_start, length, "0.000000", "0.000000", "1.000000",
        "4", "0.000000", "4", "0.000000", "0", "-1", "4", "4", "0.000000", "FALSE",
    ]
    return ";".join(values)


def dumps(timeline: Timeline, options: Optional[TimelineOptions] = None) -> str:
    del options
    if not timeline.clips:
        raise ValueError("the timeline is empty")
    seq_rate = timeline.sequence_fps
    rows = [";".join(f'"{name}"' for name in FIELDS)]
    event_id = 0
    for clip in timeline.clips:
        src_rate = clip.source.fps or seq_rate
        start = _ms(clip.record_start, seq_rate)
        length = _ms(clip.record_end - clip.record_start, seq_rate)
        stream_start = _ms(clip.in_frame, src_rate)
        event_id += 1
        rows.append(_row(event_id, VIDEO_TRACK, "VIDEO", clip.source.path, start, length, stream_start))
        if timeline.include_audio and clip.source.has_audio:
            event_id += 1
            rows.append(_row(event_id, AUDIO_TRACK, "AUDIO", clip.source.path, start, length, stream_start))
    return "\r\n".join(rows) + "\r\n"
