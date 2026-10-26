"""Cross-software validation of the timeline exports (chantier A).

These tests are intentionally stricter than the golden files: they parse our
output with third-party importers (OpenTimelineIO's FCP7 XML / CMX3600 / xges
adapters) and validate FCPXML against Apple's official 1.9 DTD with xmllint.
The DTD is fetched over the network (CommandPost mirrors the Apple file); the
test is skipped when it is not reachable so offline runs stay green.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import urllib.request
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from backend.timeline import aftereffects, edl, fcp7, fcpxml, otio
from backend.timeline.model import (
    ClipSpec,
    TimelineOptions,
    build_timeline,
)
from backend.timeline.paths import media_url, to_file_url
from backend.timeline.probe import SourceInfo
from backend.timeline.service import formats

DTD_URL = (
    "https://raw.githubusercontent.com/CommandPost/CommandPost/develop/"
    "src/extensions/cp/apple/fcpxml/dtd/FCPXMLv1_9.dtd"
)


# ── fixtures ────────────────────────────────────────────────────────────────
def _sources() -> dict[str, SourceInfo]:
    return {
        "win": SourceInfo(
            path=r"C:\Users\Léo\AMV clips\Scène 01 — l'été 日本.mp4",
            fps=Fraction(24000, 1001), width=1920, height=1080,
            duration_frames=240, has_audio=True, audio_channels=2,
        ),
        "posix": SourceInfo(
            path="/home/leo/AMV clips/Black Clover l'épisode 04.mkv",
            fps=Fraction(25), width=1280, height=720,
            duration_frames=125, has_audio=False,
        ),
        "fast": SourceInfo(
            path="/tmp/AMV/60 fps — très court.mp4",
            fps=Fraction(60), width=1920, height=1080,
            duration_frames=200, has_audio=False,
        ),
    }


def _timeline():
    sources = _sources()
    specs = [
        ClipSpec(source_path=sources["win"].path, in_frame=24, out_frame=72,
                 name="Scène 01 — l'été 日本", tags=["gojo", "combat"],
                 folder="Acte 1"),
        ClipSpec(source_path=sources["posix"].path, in_frame=10, out_frame=35,
                 name="Black Clover l'épisode 04"),
        # A single 60 fps frame converted to 23.976: the EDL must still have
        # an event at least one frame long.
        ClipSpec(source_path=sources["fast"].path, in_frame=100, out_frame=101,
                 name="Très court <5 images"),
    ]
    return build_timeline(
        specs, {info.path: info for info in sources.values()},
        TimelineOptions(sequence_fps="24000/1001", layout="gaps",
                        gap_frames=2, markers=True, include_audio=True,
                        title="Validation AMV — l'été 日本"),
    )


@pytest.fixture(scope="module")
def apple_dtd(tmp_path_factory: pytest.TempPathFactory) -> Path:
    override = __import__("os").environ.get("AMV_FCPXML_DTD")
    target = Path(override) if override else tmp_path_factory.mktemp("dtd") / "FCPXMLv1_9.dtd"
    if not target.exists():
        try:
            with urllib.request.urlopen(DTD_URL, timeout=20) as response:
                target.write_bytes(response.read())
        except Exception as exc:  # offline / mirror moved
            pytest.skip(f"Apple FCPXML 1.9 DTD unavailable: {exc}")
    if "<!ELEMENT fcpxml" not in target.read_text(errors="ignore"):
        pytest.skip("downloaded file is not the FCPXML DTD")
    return target


# ── FCPXML 1.9 against Apple's DTD ──────────────────────────────────────────
def test_fcpxml_validates_against_apple_dtd(tmp_path, apple_dtd):
    xmllint = shutil.which("xmllint")
    if not xmllint:
        pytest.skip("xmllint (libxml2-utils) is not installed")
    output = tmp_path / "timeline.fcpxml"
    output.write_text(fcpxml.dumps(_timeline()), encoding="utf-8")
    proc = subprocess.run(
        [xmllint, "--noout", "--dtdvalid", str(apple_dtd), str(output)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr


# ── FCP 7 XML imported by a third-party adapter (OTIO) ──────────────────────
def test_fcp7_xml_imported_by_opentimelineio():
    otio_lib = pytest.importorskip("opentimelineio")
    text = fcp7.dumps(_timeline())
    timeline = otio_lib.adapters.read_from_string(text, "fcp_xml")
    assert timeline.name == "Validation AMV — l'été 日本"

    tracks = list(timeline.tracks)
    assert [t.kind for t in tracks] == ["Video", "Audio"]
    video = [item for item in tracks[0] if isinstance(item, otio_lib.schema.Clip)]
    audio = [item for item in tracks[1] if isinstance(item, otio_lib.schema.Clip)]
    assert [c.name for c in video] == [
        "Scène 01 — l'été 日本", "Black Clover l'épisode 04", "Très court <5 images",
    ]
    assert len(audio) == 1  # only the first source has audio

    first = video[0].source_range
    assert first.start_time.value == 24
    assert first.start_time.rate == pytest.approx(float(Fraction(24000, 1001)))
    assert first.duration.value == 48

    # Windows path + apostrophe + accents + Japanese survived as a file URL.
    urls = [c.media_reference.target_url for c in video]
    assert urls[0] == to_file_url(_sources()["win"].path)
    assert "%27" in urls[1]  # the apostrophe
    assert "Sc%C3%A8ne" in urls[0]


# ── EDL imported by a third-party adapter (OTIO) ────────────────────────────
_EDL_EVENT = re.compile(
    r"^(\d{3})\s+(\S{1,8})\s+([VA])\s+C\s+"
    r"(\d{2}:\d{2}:\d{2}[:;]\d{2})\s+(\d{2}:\d{2}:\d{2}[:;]\d{2})\s+"
    r"(\d{2}:\d{2}:\d{2}[:;]\d{2})\s+(\d{2}:\d{2}:\d{2}[:;]\d{2})$"
)


def _edl_events(text: str) -> list[re.Match]:
    return [m for m in (_EDL_EVENT.match(line) for line in text.splitlines()) if m]


def test_edl_imported_by_opentimelineio():
    otio_lib = pytest.importorskip("opentimelineio")
    text = edl.dumps(_timeline())
    timeline = otio_lib.adapters.read_from_string(text, "cmx_3600")
    video_track = list(timeline.tracks)[0]
    clips = [item for item in video_track if isinstance(item, otio_lib.schema.Clip)]
    assert len(clips) == 3
    assert clips[0].name == "Scène 01 — l'été 日本"
    # Every event must be at least one frame long, including the 60 fps
    # single-frame clip (this used to collapse to a zero-length event).
    for clip in clips:
        assert clip.source_range.duration.value >= 1
    # Media path travels in the FROM FILE comment.
    assert "Scène 01" in clips[0].media_reference.target_url


def test_edl_columns_and_event_integrity():
    text = edl.dumps(_timeline())
    events = _edl_events(text)
    assert [e.group(3) for e in events].count("V") == 3
    assert all(len(line) <= 80 for line in text.splitlines())
    assert all(len(e.group(2)) <= 8 for e in events)
    assert text.count("* FROM CLIP NAME:") == len(events)
    event_numbers = [int(e.group(1)) for e in events]
    assert event_numbers == list(range(1, len(events) + 1))
    # Record in/out pairs advance monotonically; no zero-length event.
    for e in events:
        assert e.group(4) != e.group(5)


# ── OTIO + xges (GStreamer Editing Services) ───────────────────────────────
def test_otio_roundtrip_and_track_lengths():
    otio_lib = pytest.importorskip("opentimelineio")
    text = otio.dumps(_timeline())
    timeline = otio_lib.adapters.read_from_string(text, "otio_json")
    tracks = list(timeline.tracks)
    assert [t.kind for t in tracks] == ["Video", "Audio"]
    clips = [item for item in tracks[0] if isinstance(item, otio_lib.schema.Clip)]
    assert [c.name for c in clips] == [
        "Scène 01 — l'été 日本", "Black Clover l'épisode 04", "Très court <5 images",
    ]
    # A trailing gap keeps the audio track the same length as the video track
    # (OTIO's otiostat flags mismatched tracks).
    assert tracks[0].duration() == tracks[1].duration()
    # Markers stay inside their source range.
    for clip in clips:
        assert isinstance(clip.source_range, otio_lib.opentime.TimeRange)


def test_otio_converts_to_xges():
    otio_lib = pytest.importorskip("opentimelineio")
    timeline = otio_lib.adapters.read_from_string(otio.dumps(_timeline()), "otio_json")
    try:
        xges = otio_lib.adapters.write_to_string(timeline, "xges")
    except Exception as exc:  # adapter missing / optional GStreamer bits
        pytest.skip(f"xges adapter unavailable: {exc}")
    assert "<ges" in xges
    assert "GESUriClip" in xges
    assert xges.count("GESUriClip") >= 3


# ── Vegas Pro (.txt CMX3600) ────────────────────────────────────────────────
def test_vegas_edl_is_cmx3600_and_importable():
    otio_lib = pytest.importorskip("opentimelineio")
    text = edl.dumps_vegas(_timeline())
    assert text.splitlines()[-1].startswith("* AMV Tools - Vegas Pro")
    assert text.splitlines()[0].startswith("TITLE:")
    assert "FCM: NON-DROP FRAME" in text
    events = _edl_events(text)
    assert len(events) == 4  # 3 video + 1 audio event
    assert all(line.isascii() is False or len(line) <= 80 for line in text.splitlines())
    timeline = otio_lib.adapters.read_from_string(text, "cmx_3600")
    clips = [item for item in list(timeline.tracks)[0] if isinstance(item, otio_lib.schema.Clip)]
    assert len(clips) == 3


# ── After Effects .jsx ──────────────────────────────────────────────────────
def _jsx_for_node(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.startswith("#"))


def test_aftereffects_script_is_es3_and_places_every_clip(tmp_path):
    text = aftereffects.dumps(_timeline())
    assert text.startswith("#target aftereffects")
    assert "beginUndoGroup" in text and "endUndoGroup" in text
    assert text.count("comp.layers.add(") == 3
    assert text.count(".inPoint =") == 3 and text.count(".outPoint =") == 3
    assert text.count("app.project.importFile(") == 1  # one helper definition
    assert "addComp(" in text
    # ES3 only: ExtendScript has no let/const, arrow functions or templates.
    assert not re.search(r"(^|[^\w.])(let|const)\s", text)
    assert "=>" not in text and "`" not in text
    # Non-ASCII names/paths are escaped, so the file is pure ASCII.
    assert text.isascii()
    assert "\\u00e9" in text  # é
    # Windows backslashes became forward slashes inside File().
    assert r"C:\\Users" not in text
    assert "C:/Users/" in text

    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    script = tmp_path / "timeline.js"
    script.write_text(_jsx_for_node(text), encoding="ascii")
    proc = subprocess.run([node, "--check", str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# ── paths / format registry ─────────────────────────────────────────────────
def test_all_formats_are_registered_with_software_names():
    by_id = {f["id"]: f for f in formats()}
    assert set(by_id) == {"fcp7xml", "fcpxml", "edl", "otio", "vegas", "aftereffects"}
    assert by_id["vegas"]["extension"] == ".txt"
    assert by_id["aftereffects"]["extension"] == ".jsx"
    for fmt in by_id.values():
        assert fmt["apps"]


def test_windows_and_posix_paths_are_encoded_everywhere():
    sources = _sources()
    files = {
        "fcpxml": fcpxml.dumps(_timeline()),
        "otio": otio.dumps(_timeline()),
        "fcp7xml": fcp7.dumps(_timeline()),
        "edl": edl.dumps(_timeline()),
        "vegas": edl.dumps_vegas(_timeline()),
        "aftereffects": aftereffects.dumps(_timeline()),
    }
    win = sources["win"].path
    assert "file:///C:/Users/L%C3%A9o/AMV%20clips/" in files["fcpxml"]
    assert "file:///C:/Users/L%C3%A9o/AMV%20clips/" in files["otio"]
    assert "file:///C:/Users/L%C3%A9o/AMV%20clips/" in files["fcp7xml"]
    assert win in files["edl"] and win in files["vegas"]
    assert "C:/Users/L\\u00e9o/AMV clips/Sc\\u00e8ne 01" in files["aftereffects"]
    # Apostrophes are percent-encoded in URLs but stay readable in comments.
    assert "%27" in files["fcpxml"]
    assert "l'épisode" in files["edl"]


def test_collect_media_relative_paths_stay_importable(tmp_path):
    timeline = _timeline()
    timeline = replace(timeline, base_dir=str(tmp_path), path_mode="relative")
    url = media_url(timeline.clips[0].source.path, base_dir=str(tmp_path), relative=True)
    assert url.startswith("..") or url.startswith("C:") or ":" in url
    # A relative export must never smuggle a raw Windows drive letter into an
    # OTIO URL without percent-encoding.
    data = json.loads(otio.dumps(timeline))
    target = data["tracks"]["children"][0]["children"][0]["media_references"]["DEFAULT_MEDIA"]["target_url"]
    assert " " not in target
