"""After Effects timeline script (.jsx) writer.

After Effects has no XML timeline importer.  The reliable path is a
ExtendScript file the editor runs from *File > Scripts > Run Script File*:
it imports every source once, creates a composition at the timeline's frame
rate/resolution and places each clip with its exact in/out point.

The script is deliberately ES3 (no ``let``/``const``/arrow functions) because
that is the dialect After Effects' ExtendScript engine understands.  All
non-ASCII characters are ``\\uXXXX``-escaped so the file is pure ASCII and
survives any Windows code page.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional

from .model import Timeline, TimelineOptions
from .timecode import nominal_rate

_HEADER = "#target aftereffects\n"


def _js_str(value: object) -> str:
    """A single-quoted ExtendScript string, pure ASCII."""
    text = "" if value is None else str(value)
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == "'":
            out.append("\\'")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif 0x20 <= code <= 0x7E:
            out.append(ch)
        elif code <= 0xFFFF:
            out.append(f"\\u{code:04x}")
        else:
            code -= 0x10000
            out.append(f"\\u{0xD800 + (code >> 10):04x}\\u{0xDC00 + (code & 0x3FF):04x}")
    return "'" + "".join(out) + "'"


def _seconds_expr(frames: int, rate: Fraction) -> str:
    """Exact seconds as a JS expression (keeps 23.976 drift-free)."""
    return f"({int(frames)} * {rate.denominator} / {rate.numerator})"


def _path_for_extendscript(path: str) -> str:
    # ExtendScript's File() accepts forward slashes on Windows and macOS.
    return path.replace("\\", "/")


def _marker_lines(clip, timeline: Timeline, layer_var: str) -> list[str]:
    lines: list[str] = []
    for marker in timeline.markers:
        if not (clip.record_start <= marker.frame < clip.record_end):
            continue
        comment = marker.comment or ""
        lines += [
            f"    var mv = new MarkerValue({_js_str(marker.name)});",
            f"    mv.comment = {_js_str(comment)};",
            f"    {layer_var}.property('Marker').setValueAtTime("
            f"{_seconds_expr(marker.frame, timeline.sequence_fps)}, mv);",
        ]
    return lines


def dumps(timeline: Timeline, options: Optional[TimelineOptions] = None) -> str:
    del options
    if not timeline.clips:
        raise ValueError("the timeline is empty")

    seq_rate = timeline.sequence_fps
    seq_fps = float(seq_rate)
    duration = Fraction(timeline.duration_frames, 1) / seq_rate

    # Unique sources in first-seen order.
    indexes: dict[str, int] = {}
    sources: list[object] = []
    for clip in timeline.clips:
        if clip.source.path not in indexes:
            indexes[clip.source.path] = len(sources)
            sources.append(clip.source)

    lines = [
        _HEADER.rstrip("\n"),
        "// AMV Tools - timeline for After Effects",
        "// Open After Effects, then: File > Scripts > Run Script File...",
        "// The script imports the sources, creates a composition and places",
        "// every clip with its exact in/out point; press Ctrl/Cmd+Z to undo.",
        "(function () {",
        "    app.beginUndoGroup('AMV Tools timeline');",
        "    var missing = [];",
        "    function importMedia(path, label) {",
        "        var f = new File(path);",
        "        if (!f.exists) { missing.push(path); return null; }",
        "        try {",
        "            var io = new ImportOptions(f);",
        "            return app.project.importFile(io);",
        "        } catch (err) {",
        "            missing.push(path + ' (' + err.toString() + ')');",
        "            return null;",
        "        }",
        "    }",
    ]
    for position, info in enumerate(sources):
        lines.append(
            f"    var media{position} = importMedia("
            f"{_js_str(_path_for_extendscript(info.path))}, "
            f"{_js_str(_path_for_extendscript(info.path))});"
        )

    lines += [
        f"    var comp = app.project.items.addComp(",
        f"        {_js_str(timeline.title)}, {int(timeline.width)}, {int(timeline.height)}, "
        f"1, {float(duration):.6f}, {seq_fps:.6f});",
        "    comp.displayStartTime = 0;",
    ]

    for number, clip in enumerate(timeline.clips, start=1):
        src_rate = clip.source.fps or seq_rate
        layer = f"layer{number}"
        lines.append(f"    // clip {number}")
        lines.append(f"    var {layer} = null;")
        lines.append(f"    if (media{indexes[clip.source.path]} !== null) {{")
        lines.append(f"        {layer} = comp.layers.add(media{indexes[clip.source.path]});")
        lines.append(f"        {layer}.name = {_js_str(clip.name or f'Clip {number}')};")
        lines.append(
            f"        {layer}.startTime = {_seconds_expr(clip.record_start, seq_rate)}"
            f" - {_seconds_expr(clip.in_frame, src_rate)};"
        )
        lines.append(f"        {layer}.inPoint = {_seconds_expr(clip.record_start, seq_rate)};")
        lines.append(f"        {layer}.outPoint = {_seconds_expr(clip.record_end, seq_rate)};")
        if clip.folder or clip.tags:
            note = " · ".join(
                ([f"[{clip.folder}]"] if clip.folder else []) + list(clip.tags)
            )
            lines.append(f"        {layer}.comment = {_js_str(note)};")
        lines.extend(_marker_lines(clip, timeline, layer))
        lines.append("    }")
    lines += [
        "    if (missing.length > 0) {",
        "        alert('AMV Tools: ' + missing.length + ' source file(s) could not be imported:\\n\\n' "
        "+ missing.join('\\n'));",
        "    }",
        "    app.endUndoGroup();",
        "    comp.openInViewer();",
        "})();",
    ]
    return "\n".join(lines) + "\n"


def nominal_fps(timeline: Timeline) -> int:  # pragma: no cover - convenience for docs
    return nominal_rate(timeline.sequence_fps)
