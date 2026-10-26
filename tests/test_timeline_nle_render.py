"""Open every timeline export with third-party readers and RENDER it with MLT.

No Premiere/Resolve/AE/Vegas on Linux, so this is the closest real-world
check we can run headless:

1. each export is read back by code that is not ours, or by a small reader
   written against the format spec (never by our writer's helpers):
   - FCP 7 XML, CMX3600 EDL and OTIO → OpenTimelineIO's adapters;
   - FCPXML → spec reader below (spine offsets must be contiguous, as FCP
     and Resolve require);
   - Vegas EDL text → its semicolon layout;
   - After Effects .jsx → executed in Node against a mock of the AE object
     model (addComp / importFile / layers.add, inPoint, outPoint,
     startTime), so the script really runs, not just parses;
2. the edit list each reader recovered is handed to MLT (``melt``, the
   engine inside Kdenlive and Shotcut), which decodes the real media files
   (names with spaces, accents, apostrophes) and renders the timeline;
3. the first and last rendered frame of every clip is compared with the
   decoded source: it must be the source frame AMV Tools asked for (exactly
   when the rates match, ±1 frame across rates).

Sources are synthetic (ffmpeg testsrc2: every frame differs), 23.976, 25 and
29.97 fps, with a 3-frame clip. Skipped when melt/ffmpeg/node are missing.
"""
from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pytest

from backend.timeline import aftereffects, edl, fcp7, fcpxml, otio, vegas
from backend.timeline.model import ClipSpec, TimelineOptions, build_timeline
from backend.timeline.probe import probe_source

MELT = shutil.which("melt")
FFMPEG = shutil.which("ffmpeg")
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(not (MELT and FFMPEG), reason="needs melt (MLT) and ffmpeg")

W, H = 160, 90
SOURCES = {
    # name: (rate, seconds, audio)
    "Épisode 01 — l'été.mp4": ("24000/1001", 4, True),
    "Black Clover S02E04 25 fps.mkv": ("25", 4, False),
    "DBS Beerus 02 (29,97).mp4": ("30000/1001", 4, False),
}


@dataclass
class Event:
    path: str
    src_in_s: float      # source in point, seconds
    rec_start: int       # sequence frames
    dur: int             # sequence frames


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("nle") / "médias d'essai"
    root.mkdir()
    out = {}
    for name, (rate, secs, audio) in SOURCES.items():
        path = root / name
        cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "lavfi", "-i", f"testsrc2=size=320x180:rate={rate}", "-t", str(secs)]
        if audio:
            cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", str(secs),
                    "-c:a", "aac"]
        cmd += ["-c:v", "libx264", "-g", "1", "-pix_fmt", "yuv420p", str(path)]
        subprocess.run(cmd, check=True)
        out[name] = path
    return out


def _frames(path: Path) -> np.ndarray:
    raw = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-vf", f"scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
        check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, H, W).astype(np.float32)


def _timeline(media: dict[str, Path], seq_fps: str):
    a, b, c = (str(media[n]) for n in SOURCES)
    specs = [
        ClipSpec(source_path=a, in_frame=12, out_frame=36, name="A — l'été"),
        ClipSpec(source_path=b, in_frame=30, out_frame=55, name="B 25 fps"),
        ClipSpec(source_path=c, in_frame=40, out_frame=43, name="C court (3 images)"),
        ClipSpec(source_path=a, in_frame=60, out_frame=84, name="A encore"),
    ]
    sources = {str(p): probe_source(str(p)) for p in media.values()}
    return build_timeline(specs, sources, TimelineOptions(
        sequence_fps=seq_fps, layout="gaps", gap_frames=3, markers=True,
        include_audio=True, title="Rendu MLT — l'été"))


# ── readers (not our writer code) ───────────────────────────────────────────
def _url_to_path(url: str) -> str:
    if not url.startswith("file:"):
        return url
    p = urlparse(url)
    path = unquote(p.path)
    if p.netloc not in ("", "localhost"):
        return f"//{p.netloc}{path}"
    return path


def _read_otio(text: str, adapter: str, seq_rate: float) -> list[Event]:
    otio_lib = pytest.importorskip("opentimelineio")
    # A CMX3600 EDL has no rate of its own: like an NLE, the reader is told
    # the project rate (the adapter would assume 24 otherwise).
    extra = {"rate": seq_rate} if adapter == "cmx_3600" else {}
    tl = otio_lib.adapters.read_from_string(text, adapter, **extra)
    track = [t for t in tl.tracks if t.kind == "Video"][0]
    events = []
    for item in track:
        if not isinstance(item, otio_lib.schema.Clip):
            continue
        rng = track.range_of_child(item)
        ref = item.media_reference
        url = getattr(ref, "target_url", None) or ""
        events.append(Event(
            path=_url_to_path(url),
            src_in_s=item.source_range.start_time.to_seconds(),
            rec_start=round(rng.start_time.to_seconds() * seq_rate),
            dur=round(rng.duration.to_seconds() * seq_rate)))
    return events


def _rat(value: str) -> Fraction:
    value = value.rstrip("s")
    return Fraction(value) if value else Fraction(0)


def _read_fcpxml(text: str, seq_rate: float) -> list[Event]:
    root = ET.fromstring(text)
    assets = {a.get("id"): a.find("media-rep").get("src") for a in root.iter("asset")}
    spine = root.find(".//sequence/spine")
    events, cursor = [], Fraction(0)
    for child in spine:
        offset, duration = _rat(child.get("offset", "0s")), _rat(child.get("duration", "0s"))
        # FCP/Resolve: spine items are back to back, each offset = running sum.
        assert offset == cursor, f"spine not contiguous at {child.tag} ({offset} != {cursor})"
        cursor += duration
        if child.tag == "asset-clip":
            events.append(Event(
                path=_url_to_path(assets[child.get("ref")]),
                src_in_s=float(_rat(child.get("start", "0s"))),
                rec_start=round(float(offset) * seq_rate),
                dur=round(float(duration) * seq_rate)))
        else:
            assert child.tag == "gap", child.tag
    return events


def _read_vegas(text: str, seq_rate: float) -> list[Event]:
    rows = list(csv.reader(io.StringIO(text), delimiter=";", quotechar='"'))
    header = rows[0]
    events = []
    for row in rows[1:]:
        e = dict(zip(header, row))
        if e["MediaType"] != "VIDEO":
            continue
        events.append(Event(
            path=e["FileName"],
            src_in_s=float(e["StreamStart"]) / 1000,
            rec_start=round(float(e["StartTime"]) / 1000 * seq_rate),
            dur=round(float(e["Length"]) / 1000 * seq_rate)))
    return events


_AE_MOCK = r"""
var fs = require('fs');
var path = require('path');
var layers = [];
function File(p) { this.fsName = p; this.exists = fs.existsSync(p);
                   this.parent = { fsName: path.dirname(p) }; }
var $ = { fileName: process.argv[2] };
function ImportOptions(f) { this.file = f; }
function MarkerValue(name) { this.name = name; }
function alert(msg) { console.error('ALERT ' + msg); process.exitCode = 3; }
var app = {
  beginUndoGroup: function () {}, endUndoGroup: function () {},
  project: {
    importFile: function (io) { return { footage: io.file.fsName }; },
    items: { addComp: function (name, w, h, par, dur, fps) {
      var comp = { name: name, width: w, height: h, duration: dur, frameRate: fps,
        openInViewer: function () {},
        layers: { add: function (item) {
          var layer = { source: item.footage, startTime: 0,
            property: function () { return { setValueAtTime: function () {} }; } };
          layers.push(layer); return layer; } } };
      app.comp = comp; return comp; } } }
};
"""


def _read_ae(text: str, seq_rate: float, tmp_path: Path) -> list[Event]:
    if not NODE:
        pytest.skip("node is not installed")
    body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    script = tmp_path / "run_ae.js"
    script.write_text(_AE_MOCK + body + "\nconsole.log(JSON.stringify({comp: app.comp, layers: layers}));\n",
                      encoding="utf-8")
    proc = subprocess.run([NODE, str(script), str(tmp_path / "timeline.jsx")],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert abs(data["comp"]["frameRate"] - seq_rate) < 1e-3
    events = []
    for layer in data["layers"]:
        events.append(Event(
            path=layer["source"],
            src_in_s=layer["inPoint"] - layer["startTime"],
            rec_start=round(layer["inPoint"] * seq_rate),
            dur=round((layer["outPoint"] - layer["inPoint"]) * seq_rate)))
    return sorted(events, key=lambda e: e.rec_start)


# ── MLT render ──────────────────────────────────────────────────────────────
def _render_mlt(events: list[Event], rate: Fraction, out_dir: Path) -> np.ndarray:
    """Builds a one-track MLT playlist from the recovered edit list, renders it."""
    seq = float(rate)
    out_dir.mkdir(parents=True, exist_ok=True)
    producers, entries, cursor = [], [], 0
    for i, ev in enumerate(events):
        producers.append(f'<producer id="p{i}"><property name="resource">{_xml(ev.path)}</property></producer>')
        if ev.rec_start > cursor:
            entries.append(f'<blank length="{ev.rec_start - cursor}"/>')
        # MLT in/out are in profile frames; the source is resampled by time.
        src_in = round(ev.src_in_s * seq)
        entries.append(f'<entry producer="p{i}" in="{src_in}" out="{src_in + ev.dur - 1}"/>')
        cursor = ev.rec_start + ev.dur
    doc = (
        '<?xml version="1.0" encoding="utf-8"?><mlt>'
        f'<profile width="{W}" height="{H}" frame_rate_num="{rate.numerator}" '
        f'frame_rate_den="{rate.denominator}" progressive="1" sample_aspect_num="1" '
        'sample_aspect_den="1" display_aspect_num="16" display_aspect_den="9" colorspace="709"/>'
        + "".join(producers)
        + '<playlist id="v">' + "".join(entries) + "</playlist>"
        '<tractor><multitrack><track producer="v"/></multitrack></tractor></mlt>'
    )
    project = out_dir / "timeline.mlt"
    project.write_text(doc, encoding="utf-8")
    raw = out_dir / "render.gray"
    proc = subprocess.run([MELT, "-quiet", str(project), "-consumer", f"avformat:{raw}",
                           "f=rawvideo", "pix_fmt=gray", "an=1"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    data = np.frombuffer(raw.read_bytes(), dtype=np.uint8)
    return data.reshape(-1, H, W).astype(np.float32)


def _xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")


def _best(frame: np.ndarray, source: np.ndarray, around: int, radius: int = 4) -> int:
    lo, hi = max(0, around - radius), min(len(source), around + radius + 1)
    errs = [float(np.mean((source[i] - frame) ** 2)) for i in range(lo, hi)]
    return lo + int(np.argmin(errs))


READERS = ["fcp7xml", "fcpxml", "edl", "otio", "vegas", "aftereffects"]
WRITERS = {"fcp7xml": fcp7.dumps, "fcpxml": fcpxml.dumps, "edl": edl.dumps,
           "otio": otio.dumps, "vegas": vegas.dumps, "aftereffects": aftereffects.dumps}


@pytest.mark.parametrize("seq_fps", ["24000/1001", "25"])
@pytest.mark.parametrize("fmt", READERS)
def test_export_reopens_and_renders_the_right_frames(fmt, seq_fps, media, tmp_path):
    timeline = _timeline(media, seq_fps)
    rate = timeline.sequence_fps
    seq = float(rate)
    text = WRITERS[fmt](timeline, TimelineOptions(otio_use_library=False))
    (tmp_path / "timeline.jsx").write_text(text, encoding="utf-8")

    if fmt == "fcp7xml":
        events = _read_otio(text, "fcp_xml", seq)
    elif fmt == "edl":
        events = _read_otio(text, "cmx_3600", seq)
        # CMX3600 carries no paths: the reader takes the FROM FILE comment.
    elif fmt == "otio":
        events = _read_otio(text, "otio_json", seq)
    elif fmt == "fcpxml":
        events = _read_fcpxml(text, seq)
    elif fmt == "vegas":
        events = _read_vegas(text, seq)
    else:
        events = _read_ae(text, seq, tmp_path)

    # 1. structure: count, order, placement, durations, paths.
    assert len(events) == len(timeline.clips)
    for ev, clip in zip(events, timeline.clips):
        assert Path(ev.path) == Path(clip.source.path), (fmt, ev.path)
        assert ev.rec_start == clip.record_start, (fmt, clip.name)
        assert ev.dur == clip.record_duration, (fmt, clip.name)
        exact_in = float(Fraction(clip.in_frame) / clip.source.fps)
        tolerance = 0.5 / seq + 1e-6 if fmt == "edl" else 1e-6   # EDL: one rate for all
        assert abs(ev.src_in_s - exact_in) <= tolerance, (fmt, clip.name, ev.src_in_s, exact_in)

    # 2. MLT decodes the real files at those points.
    rendered = _render_mlt(events, rate, tmp_path / "mlt")
    assert len(rendered) >= timeline.duration_frames - 1
    decoded = {str(p): _frames(p) for p in media.values()}
    for clip in timeline.clips:
        src = decoded[clip.source.path]
        same_rate = clip.source.fps == rate
        for offset in (0, clip.record_duration - 1):
            want = clip.in_frame + round(offset * float(clip.source.fps) / seq)
            got = _best(rendered[clip.record_start + offset], src, want)
            slack = 0 if same_rate and fmt != "edl" else 1
            assert abs(got - want) <= slack, (fmt, clip.name, offset, got, want)


# ── same check on real episodes (opt-in) ────────────────────────────────────
# AMV_NLE_MEDIA="/path/a.mkv:/path/b.mkv" runs it on real extracts (H.264 /
# HEVC, long GOPs, 23.976): the files are linked under names with spaces,
# accents and an apostrophe. Not in CI: the media is not in the repo.
def _window(path: Path, first: int, count: int, fps: Fraction) -> np.ndarray:
    raw = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-ss", f"{float(first / fps):.6f}", "-i", str(path),
         "-frames:v", str(count), "-vf", f"scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
        check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, H, W).astype(np.float32)


@pytest.mark.skipif(not __import__("os").environ.get("AMV_NLE_MEDIA"), reason="set AMV_NLE_MEDIA to real extracts")
@pytest.mark.parametrize("fmt", READERS)
def test_real_episodes_render_the_right_frames(fmt, tmp_path):
    import os
    real = [Path(p) for p in os.environ["AMV_NLE_MEDIA"].split(":") if p]
    folder = tmp_path / "Épisodes d'essai"
    folder.mkdir()
    names = ["DBS Beerus 01 — l'été.mkv", "Black Clover S02E04 (日本).mkv"]
    links = []
    for src, name in zip(real, names):
        (folder / name).symlink_to(src)
        links.append(str(folder / name))
    sources = {p: probe_source(p) for p in links}
    specs = []
    for i, p in enumerate(links):
        fps = sources[p].fps
        # Mid-GOP in points, a 2-frame clip, a long clip.
        specs.append(ClipSpec(source_path=p, in_frame=int(37.3 * fps), out_frame=int(37.3 * fps) + 50, name=f"{i} a"))
        specs.append(ClipSpec(source_path=p, in_frame=int(81.7 * fps), out_frame=int(81.7 * fps) + 2, name=f"{i} b"))
    timeline = build_timeline(specs, sources, TimelineOptions(sequence_fps="first", layout="gaps", gap_frames=2))
    rate = timeline.sequence_fps
    seq = float(rate)
    text = WRITERS[fmt](timeline, TimelineOptions())
    (tmp_path / "timeline.jsx").write_text(text, encoding="utf-8")
    reader = {"fcp7xml": lambda: _read_otio(text, "fcp_xml", seq),
              "edl": lambda: _read_otio(text, "cmx_3600", seq),
              "otio": lambda: _read_otio(text, "otio_json", seq),
              "fcpxml": lambda: _read_fcpxml(text, seq),
              "vegas": lambda: _read_vegas(text, seq),
              "aftereffects": lambda: _read_ae(text, seq, tmp_path)}[fmt]
    events = reader()
    assert [(e.rec_start, e.dur) for e in events] == [(c.record_start, c.record_duration) for c in timeline.clips]
    rendered = _render_mlt(events, rate, tmp_path / "mlt")
    checked: list[bool] = []
    for clip in timeline.clips:
        for offset in (0, clip.record_duration - 1):
            want = clip.in_frame + offset
            win = _window(Path(clip.source.path), want - 4, 9, clip.source.fps)
            frame = rendered[clip.record_start + offset]
            errs = [float(np.mean((w - frame) ** 2)) for w in win]
            # Anime holds drawings for 2-3 frames (and stills for longer):
            # identical neighbours tie, so the wanted frame must be AS GOOD
            # as the best one, and strictly better than any differing one.
            assert errs[4] <= min(errs) + 1.0, (fmt, clip.name, offset, [round(e) for e in errs])
            checked.append(max(errs) - errs[4] > 5.0)
    # At least half of the probes landed where neighbours differ (not all stills).
    assert sum(checked) >= len(checked) // 2, checked
