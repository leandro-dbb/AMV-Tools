"""Timeline exporter tests (chantier B).

Synthetic media is generated with ffmpeg (small files, deleted with the
temporary directory).  The golden files pin the four writers byte-for-byte so
a refactor cannot silently change an NLE-visible value.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.timeline import edl, fcp7, fcpxml, otio  # noqa: E402
from backend.timeline.collect import collect_media  # noqa: E402
from backend.timeline.model import (  # noqa: E402
    ClipSpec,
    Marker,
    PlacedClip,
    Timeline,
    TimelineOptions,
    build_timeline,
    timeline_summary,
)
from backend.timeline.paths import to_file_url  # noqa: E402
from backend.timeline.probe import SourceInfo, probe_source  # noqa: E402
from backend.timeline.service import export_timeline, preview_timeline  # noqa: E402
from backend.timeline.timecode import (  # noqa: E402
    frames_to_seconds,
    frames_to_timecode,
    ms_to_frames,
    timecode_to_frames,
)

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
FFMPEG = shutil.which("ffmpeg")


# ── fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def synthetic_media(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    if FFMPEG is None:
        pytest.skip("ffmpeg is required for the timeline media fixtures")
    root = tmp_path_factory.mktemp("timeline-media")
    clips: dict[str, Path] = {}

    def make(name: str, rate: str, seconds: int, audio: bool) -> Path:
        target = root / name
        cmd = [
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc2=size=320x180:rate={rate}",
        ]
        if audio:
            cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
        cmd += ["-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p"]
        if audio:
            cmd += ["-c:a", "aac", "-shortest"]
        cmd.append(str(target))
        subprocess.run(cmd, check=True, capture_output=True)
        return target

    clips["ntsc23"] = make("a_23976.mp4", "24000/1001", 3, True)
    clips["pal25"] = make("b_25.mp4", "25", 3, False)
    clips["ntsc29"] = make("c_2997.mp4", "30000/1001", 3, False)
    return clips


def _golden_timeline() -> Timeline:
    first = SourceInfo(
        path=r"C:\AMV Golden\clip_é 日本.mp4",
        fps=Fraction(24000, 1001),
        width=1920,
        height=1080,
        duration_frames=240,
        has_audio=True,
        audio_channels=2,
    )
    second = SourceInfo(
        path=r"C:\AMV Golden\clip_b.mp4",
        fps=Fraction(25, 1),
        width=1280,
        height=720,
        duration_frames=125,
        has_audio=False,
    )
    clips = [
        PlacedClip(
            source=first, in_frame=24, out_frame=72,
            record_start=0, record_end=48,
            name="Scène 01 — 東京", label="Mango", folder="Acte 1",
            tags=["gojo", "combat"], scene_id=10, item_id=5,
        ),
        PlacedClip(
            source=second, in_frame=10, out_frame=35,
            record_start=48, record_end=72,
            name="Clip B", tags=[], scene_id=11,
        ),
    ]
    return Timeline(
        clips=clips,
        sequence_fps=Fraction(24000, 1001),
        width=1920,
        height=1080,
        title="Golden timeline",
        include_audio=True,
        markers=[
            Marker(frame=0, name="Scène 01 — 東京",
                   comment="[Acte 1] gojo · combat", color="Mango"),
            Marker(frame=48, name="Clip B", comment=""),
        ],
    )


def _collected_golden_timeline() -> Timeline:
    """The golden timeline as ``collect_media`` leaves it for an export saved
    as ``C:\\AMV Golden\\export\\timeline.<ext>``: sources rewritten to the
    copies in ``timeline_media\\`` and ``path_mode="relative"``."""
    import ntpath
    timeline = _golden_timeline()
    media = r"C:\AMV Golden\export\timeline_media"
    clips = [replace(c, source=c.source.with_path(ntpath.join(media, ntpath.basename(c.source.path))))
             for c in timeline.clips]
    return replace(timeline, clips=clips, base_dir=r"C:\AMV Golden\export",
                   path_mode="relative")


# ── timecode / fps maths ────────────────────────────────────────────────────
def test_ntsc_fractions_are_exact():
    assert Fraction(24000, 1001) == Fraction(24000, 1001)
    assert float(Fraction(24000, 1001)) == pytest.approx(23.976023976)
    assert frames_to_seconds(24, Fraction(24000, 1001)) == Fraction(1001, 1000)


def test_non_drop_timecode_roundtrip():
    for fps in (Fraction(24), Fraction(25), Fraction(30), Fraction(60)):
        for frame in (0, 1, 23, 24, 1000, 12345):
            tc = frames_to_timecode(frame, fps, drop_frame=False)
            assert timecode_to_frames(tc, fps, drop_frame=False) == frame


def test_drop_frame_2997_known_values():
    fps = Fraction(30000, 1001)
    assert frames_to_timecode(0, fps, True) == "00:00:00;00"
    assert frames_to_timecode(1799, fps, True) == "00:00:59;29"
    assert frames_to_timecode(1800, fps, True) == "00:01:00;02"
    assert frames_to_timecode(17982, fps, True) == "00:10:00;00"
    assert timecode_to_frames("00:01:00;02", fps, True) == 1800
    assert timecode_to_frames("00:10:00;00", fps, True) == 17982
    for frame in (0, 1799, 1800, 17982, 36000):
        tc = frames_to_timecode(frame, fps, True)
        assert timecode_to_frames(tc, fps, True) == frame


def test_drop_frame_5994():
    fps = Fraction(60000, 1001)
    assert frames_to_timecode(3599, fps, True) == "00:00:59;59"
    assert frames_to_timecode(3600, fps, True) == "00:01:00;04"
    assert timecode_to_frames("00:01:00;04", fps, True) == 3600


def test_ms_to_frames_uses_exact_rate():
    assert ms_to_frames(1000, Fraction(24000, 1001)) == 24
    assert ms_to_frames(1000, Fraction(25)) == 25
    assert ms_to_frames(2000, Fraction(30000, 1001)) == 60


# ── probe ───────────────────────────────────────────────────────────────────
def test_probe_reports_exact_rates(synthetic_media):
    expected = {
        "ntsc23": Fraction(24000, 1001),
        "pal25": Fraction(25, 1),
        "ntsc29": Fraction(30000, 1001),
    }
    for key, rate in expected.items():
        info = probe_source(synthetic_media[key])
        assert info.fps == rate
        assert info.width == 320 and info.height == 180
        assert info.duration_frames and info.duration_frames > 0
        assert not info.is_vfr


def test_probe_audio(synthetic_media):
    assert probe_source(synthetic_media["ntsc23"]).has_audio
    assert not probe_source(synthetic_media["pal25"]).has_audio


# ── path encoding ───────────────────────────────────────────────────────────
def test_file_urls_encode_accents_japanese_and_windows():
    assert to_file_url("/tmp/a é 日本.mp4") == (
        "file:///tmp/a%20%C3%A9%20%E6%97%A5%E6%9C%AC.mp4"
    )
    assert to_file_url(r"C:\Users\é\clip 01.mp4") == (
        "file:///C:/Users/%C3%A9/clip%2001.mp4"
    )
    assert to_file_url(r"\\server\share\clip 01.mp4") == (
        "file://server/share/clip%2001.mp4"
    )


def test_relative_media_urls_windows_paths_on_any_host():
    from backend.timeline.paths import media_url
    base = r"C:\AMV\export"
    assert media_url(r"C:\AMV\export\timeline_media\é 1.mp4", base_dir=base, relative=True) == (
        "timeline_media/%C3%A9%201.mp4"
    )
    # No relative path to another drive (uncollected source): absolute URL.
    assert media_url(r"D:\anime\ep 01.mkv", base_dir=base, relative=True) == (
        "file:///D:/anime/ep%2001.mkv"
    )


# ── FCP7 XML ────────────────────────────────────────────────────────────────
def test_fcp7_structure_and_exact_frames():
    xml = fcp7.dumps(_golden_timeline())
    root = ET.fromstring(xml)
    assert root.tag == "xmeml"
    seq = root.find("sequence")
    assert seq.findtext("duration") == "72"
    assert seq.findtext("rate/timebase") == "24"
    assert seq.findtext("rate/ntsc") == "TRUE"
    assert seq.findtext("timecode/displayformat") == "NDF"

    items = seq.findall("media/video/track/clipitem")
    assert len(items) == 2
    assert items[0].findtext("in") == "24"
    assert items[0].findtext("out") == "72"
    assert items[0].findtext("start") == "0"
    assert items[0].findtext("end") == "48"
    assert items[1].findtext("in") == "10"
    assert items[1].findtext("out") == "35"
    assert items[1].findtext("rate/timebase") == "25"
    assert items[1].findtext("rate/ntsc") == "FALSE"

    url = items[0].findtext("file/pathurl")
    # Premiere's own spelling: localhost host, drive colon percent-encoded.
    assert url == "file://localhost/C%3a/AMV%20Golden/clip_%C3%A9%20%E6%97%A5%E6%9C%AC.mp4"

    # Audio included: one audio clipitem for the first source only.
    audio_items = seq.findall("media/audio/track/clipitem")
    assert len(audio_items) == 1
    assert audio_items[0].findtext("in") == "24"
    # The full <file> definition appears once per source; later clipitems
    # reference it with a self-closing <file id="…"/> (FCP 7 convention).
    declared = [e for e in root.iter() if e.tag == "file" and len(e) > 0]
    assert len(declared) == 2

    markers = seq.findall("marker")
    assert [m.findtext("in") for m in markers] == ["0", "48"]


def test_fcp7_include_audio_off():
    timeline = _golden_timeline()
    timeline.include_audio = False
    root = ET.fromstring(fcp7.dumps(timeline))
    assert root.find("sequence/media/audio") is None


# ── FCPXML ──────────────────────────────────────────────────────────────────
def test_fcpxml_structure_and_rational_times():
    xml = fcpxml.dumps(_golden_timeline())
    root = ET.fromstring(xml)
    assert root.tag == "fcpxml"
    assert root.get("version") == "1.9"
    seq = root.find("library/event/project/sequence")
    assert seq.get("duration") == "3003/1000s"  # 72 frames @ 24000/1001
    assert seq.get("tcFormat") == "NDF"
    fmt = root.find("resources/format")
    assert fmt.get("frameDuration") == "1001/24000s"

    clips = seq.findall("spine/asset-clip")
    assert len(clips) == 2
    assert clips[0].get("offset") == "0s"
    assert clips[0].get("duration") == "1001/500s"  # 48 frames
    assert clips[0].get("start") == "1001/1000s"    # 24 frames
    assert clips[1].get("offset") == "1001/500s"
    assert clips[1].get("duration") == "1001/1000s"  # 24 frames

    assets = root.findall("resources/asset")
    assert len(assets) == 2
    src = assets[0].find("media-rep").get("src")
    assert src == "file:///C:/AMV%20Golden/clip_%C3%A9%20%E6%97%A5%E6%9C%AC.mp4"

    markers = clips[0].findall("marker")
    assert markers[0].get("value") == "Scène 01 — 東京"


# ── EDL ─────────────────────────────────────────────────────────────────────
_EDL_EVENT = re.compile(
    r"^\d{3}\s+\S{1,8}\s+[VA]\s+C\s+"
    r"\d{2}:\d{2}:\d{2}[:;]\d{2}\s+\d{2}:\d{2}:\d{2}[:;]\d{2}\s+"
    r"\d{2}:\d{2}:\d{2}[:;]\d{2}\s+\d{2}:\d{2}:\d{2}[:;]\d{2}$"
)


def test_edl_columns_and_comments():
    text = edl.dumps(_golden_timeline())
    lines = text.splitlines()
    assert lines[0].startswith("TITLE:")
    assert lines[1] == "FCM: NON-DROP FRAME"
    events = [line for line in lines if _EDL_EVENT.match(line)]
    assert len(events) == 3  # V, A for clip 1; V for clip 2
    assert all(len(line) <= 80 for line in lines)
    assert lines[3] == (
        "001  CLIP     V     C        "
        "00:00:01:00 00:00:03:00 00:00:00:00 00:00:02:00"
    )
    assert "* FROM CLIP NAME: Scène 01 — 東京" in lines
    assert "* FROM FILE: C:\\AMV Golden\\clip_é 日本.mp4" in lines
    assert text.count("* FROM CLIP NAME:") == 3


def test_edl_drop_frame_uses_semicolon():
    timeline = _golden_timeline()
    timeline.sequence_fps = Fraction(30000, 1001)
    timeline.drop_frame = True
    # Rebuild record positions at 30 fps semantics (same integers, new rate).
    text = edl.dumps(timeline)
    assert "FCM: DROP FRAME" in text
    assert re.search(r"\d{2}:\d{2}:\d{2};\d{2}", text)
    assert "00:00:00;00" in text


# ── OTIO ────────────────────────────────────────────────────────────────────
def test_otio_native_json_is_valid():
    text = otio.dumps(_golden_timeline(), TimelineOptions(otio_use_library=False))
    data = json.loads(text)
    assert data["OTIO_SCHEMA"] == "Timeline.1"
    assert data["name"] == "Golden timeline"
    stack = data["tracks"]
    assert stack["OTIO_SCHEMA"] == "Stack.1"
    video = stack["children"][0]
    assert video["kind"] == "Video"
    clips = [c for c in video["children"] if c["OTIO_SCHEMA"] == "Clip.2"]
    assert len(clips) == 2
    assert clips[0]["source_range"]["start_time"]["value"] == 24
    assert clips[0]["source_range"]["duration"]["value"] == 48
    assert clips[0]["media_references"]["DEFAULT_MEDIA"]["target_url"] == (
        "file:///C:/AMV%20Golden/clip_%C3%A9%20%E6%97%A5%E6%9C%AC.mp4"
    )
    assert clips[0]["metadata"]["amv"]["tags"] == ["gojo", "combat"]
    assert stack["children"][1]["kind"] == "Audio"


def test_otio_if_library_installed():
    otio_lib = pytest.importorskip("opentimelineio")
    text = otio.dumps(_golden_timeline(), TimelineOptions(otio_use_library=False))
    timeline = otio_lib.adapters.read_from_string(text, "otio_json")
    assert timeline.name == "Golden timeline"
    tracks = list(timeline.tracks)
    assert len(tracks) == 2
    clips = [item for item in tracks[0] if isinstance(item, otio_lib.schema.Clip)]
    assert len(clips) == 2
    assert clips[0].name == "Scène 01 — 東京"


# ── model / options ─────────────────────────────────────────────────────────
def test_handles_and_gaps():
    source = SourceInfo(path="/tmp/a.mp4", fps=Fraction(25), width=640, height=360,
                        duration_frames=100, has_audio=False)
    specs = [
        ClipSpec(source_path="/tmp/a.mp4", in_frame=10, out_frame=20, name="a"),
        ClipSpec(source_path="/tmp/a.mp4", in_frame=30, out_frame=40, name="b"),
    ]
    timeline = build_timeline(
        specs, {"/tmp/a.mp4": source},
        TimelineOptions(sequence_fps="25", layout="gaps", gap_frames=5,
                        handles_frames=2, markers=False),
    )
    first, second = timeline.clips
    assert (first.in_frame, first.out_frame) == (8, 22)
    assert (first.record_start, first.record_end) == (0, 14)
    assert (second.record_start, second.record_end) == (19, 33)


def test_mixed_rates_are_converted_to_sequence_rate():
    a = SourceInfo(path="/tmp/a.mp4", fps=Fraction(24000, 1001), width=1920,
                   height=1080, duration_frames=240, has_audio=False)
    b = SourceInfo(path="/tmp/b.mp4", fps=Fraction(25), width=1920,
                   height=1080, duration_frames=125, has_audio=False)
    specs = [
        ClipSpec(source_path="/tmp/a.mp4", in_frame=0, out_frame=24, name="a"),
        ClipSpec(source_path="/tmp/b.mp4", in_frame=0, out_frame=25, name="b"),
    ]
    timeline = build_timeline(
        specs, {"/tmp/a.mp4": a, "/tmp/b.mp4": b},
        TimelineOptions(sequence_fps="24", markers=False),
    )
    assert timeline.clips[0].record_duration == 24
    assert timeline.clips[1].record_duration == 24
    assert timeline.duration_frames == 48


def test_vfr_source_warns_and_uses_average():
    source = SourceInfo(path="/tmp/vfr.mp4", fps=Fraction(30000, 1001),
                        width=640, height=360, duration_frames=100,
                        has_audio=False, is_vfr=True,
                        warnings=["VFR source detected"])
    timeline = build_timeline(
        [ClipSpec(source_path="/tmp/vfr.mp4", in_frame=0, out_frame=10, name="v")],
        {"/tmp/vfr.mp4": source}, TimelineOptions(sequence_fps="first"),
    )
    assert any("VFR" in w for w in timeline.warnings)


def test_preview_summary(synthetic_media):
    specs = [
        ClipSpec(source_path=str(synthetic_media["ntsc23"]), start_ms=0,
                 end_ms=1000, name="a"),
        ClipSpec(source_path=str(synthetic_media["pal25"]), start_ms=500,
                 end_ms=1500, name="b"),
    ]
    preview = preview_timeline(specs, TimelineOptions(sequence_fps="first"))
    assert preview["clip_count"] == 2
    assert preview["sequence_fps"] == "24000/1001"
    assert preview["duration_frames"] > 0
    assert any("Mixed source frame rates" in w for w in preview["warnings"])


# ── collect media ───────────────────────────────────────────────────────────
def test_collect_media_copies_without_transcoding(synthetic_media, tmp_path):
    unicode_clip = tmp_path / "クリップ é.mp4"
    shutil.copy2(synthetic_media["ntsc23"], unicode_clip)
    specs = [
        ClipSpec(source_path=str(synthetic_media["ntsc23"]), start_ms=0,
                 end_ms=1000, name="a"),
        ClipSpec(source_path=str(synthetic_media["pal25"]), start_ms=0,
                 end_ms=1000, name="b"),
        ClipSpec(source_path=str(unicode_clip), start_ms=0,
                 end_ms=1000, name="c"),
    ]
    out = tmp_path / "session" / "timeline.xml"
    result = export_timeline(specs, "fcp7xml", str(out), TimelineOptions(
        sequence_fps="first", collect_media=True, path_mode="relative",
    ))
    media_dir = Path(result["collected_dir"])
    assert media_dir.is_dir()
    assert len(list(media_dir.iterdir())) == 3
    for src in (synthetic_media["ntsc23"], synthetic_media["pal25"], unicode_clip):
        copied = media_dir / src.name
        assert copied.exists()
        assert copied.stat().st_size == src.stat().st_size
    text = out.read_text(encoding="utf-8")
    # FCP 7 XML can't hold relative media paths: absolute URLs of the copies.
    urls = re.findall(r"<pathurl>([^<]+)</pathurl>", text)
    assert len(urls) == 3
    for url in urls:
        assert url.startswith("file://")
        assert url.startswith(fcp7.xmeml_pathurl(to_file_url(media_dir)) + "/")
    assert "%E3%82%AF%E3%83%AA%E3%83%83%E3%83%97" in text

    # FCPXML / OTIO / EDL point at the same copies, relative to the timeline.
    for fmt, ext in (("fcpxml", ".fcpxml"), ("otio", ".otio"), ("edl", ".edl")):
        other = tmp_path / "session" / f"timeline{ext}"
        export_timeline(specs, fmt, str(other), TimelineOptions(
            sequence_fps="first", collect_media=True))
        body = other.read_text(encoding="utf-8")
        assert "timeline_media/" in body, fmt
        assert str(tmp_path) not in body and "file://" not in body, fmt


def test_export_all_formats_with_unicode_paths(synthetic_media, tmp_path):
    unicode_dir = tmp_path / "アクセント é"
    unicode_dir.mkdir()
    unicode_clip = unicode_dir / "クリップ é 01.mp4"
    shutil.copy2(synthetic_media["ntsc23"], unicode_clip)
    specs = [
        ClipSpec(source_path=str(unicode_clip), start_ms=250, end_ms=1750,
                 name="Scène é 日本"),
        ClipSpec(source_path=str(synthetic_media["ntsc29"]), start_ms=0,
                 end_ms=1000, name="Palette 29.97"),
    ]
    extensions = {"fcp7xml": ".xml", "fcpxml": ".fcpxml", "edl": ".edl", "otio": ".otio"}
    for fmt, ext in extensions.items():
        out = tmp_path / f"export_{fmt}{ext}"
        result = export_timeline(specs, fmt, str(out))
        assert result["ok"] is True
        assert out.exists() and out.stat().st_size > 0
        if fmt in ("fcp7xml", "fcpxml"):
            ET.parse(out)
        if fmt == "otio":
            data = json.loads(out.read_text(encoding="utf-8"))
            assert data["OTIO_SCHEMA"] == "Timeline.1"
        if fmt == "edl":
            assert "FROM CLIP NAME" in out.read_text(encoding="utf-8")


# ── golden files ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("fmt,filename", [
    ("fcp7xml", "timeline.fcp7.xml"),
    ("fcpxml", "timeline.fcpxml"),
    ("edl", "timeline.edl"),
    ("otio", "timeline.otio"),
    ("fcp7xml", "timeline.collected.fcp7.xml"),
    ("fcpxml", "timeline.collected.fcpxml"),
    ("edl", "timeline.collected.edl"),
    ("otio", "timeline.collected.otio"),
])
def test_golden_outputs(fmt, filename):
    timeline = _collected_golden_timeline() if ".collected." in filename else _golden_timeline()
    options = TimelineOptions(otio_use_library=False)
    content = {
        "fcp7xml": fcp7.dumps,
        "fcpxml": fcpxml.dumps,
        "edl": edl.dumps,
        "otio": otio.dumps,
    }[fmt](timeline, options)
    golden = GOLDEN_DIR / filename
    if not golden.exists():
        pytest.fail(f"missing golden file {golden}; regenerate with tests/golden/generate.py")
    assert content == golden.read_text(encoding="utf-8")


# ── review fixes: escaping, marker placement, OTIO path, collect, DF ────────
_NASTY_TITLE = 'Mon AMV "v2" & <final>\x07 \'ok\''
_NASTY_CLIP = 'Combat "final" & <b>\x01\x1f'


def _nasty_timeline() -> Timeline:
    timeline = _golden_timeline()
    timeline.title = _NASTY_TITLE
    timeline.clips[0].name = _NASTY_CLIP
    timeline.markers[0].name = _NASTY_CLIP
    timeline.markers[0].comment = 'note "x" & <y>\x0b'
    return timeline


def _clean(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)


def test_fcpxml_escapes_quotes_and_strips_control_chars():
    text = fcpxml.dumps(_nasty_timeline())
    root = ET.fromstring(text)  # well-formed despite " & < and control chars
    assert root.find("library/event").get("name") == _clean(_NASTY_TITLE)
    assert root.find("library/event/project").get("name") == _clean(_NASTY_TITLE)
    clip = root.find("library/event/project/sequence/spine/asset-clip")
    assert clip.get("name") == _clean(_NASTY_CLIP)
    marker = clip.find("marker")
    assert marker.get("value") == _clean(_NASTY_CLIP)
    # FCPXML 1.9 declares <marker> EMPTY: the comment is the ``note``
    # attribute, not a child element (validated against Apple's DTD).
    assert marker.get("note") == 'note "x" & <y>'
    assert "\x07" not in text and "\x01" not in text


def test_fcp7_escapes_and_strips_control_chars():
    text = fcp7.dumps(_nasty_timeline())
    root = ET.fromstring(text)
    seq = root.find("sequence")
    assert seq.findtext("name") == _clean(_NASTY_TITLE)
    assert seq.find("media/video/track/clipitem").findtext("name") == _clean(_NASTY_CLIP)
    assert seq.find("marker").findtext("name") == _clean(_NASTY_CLIP)
    assert seq.find("marker").findtext("comment") == 'note "x" & <y>'


def test_fcp7_omits_labels_when_empty():
    root = ET.fromstring(fcp7.dumps(_golden_timeline()))
    items = root.findall("sequence/media/video/track/clipitem")
    assert items[0].findtext("labels/label") == "Mango"
    assert items[1].find("labels") is None  # "Clip B" has no label


def _rational_seconds(value: str) -> Fraction:
    assert value.endswith("s")
    return Fraction(value[:-1])


def _marker_timeline() -> Timeline:
    """Clips with in points > 0 and markers mid-clip, mixed source rates."""
    timeline = _golden_timeline()
    timeline.markers = [
        Marker(frame=0, name="first"),
        Marker(frame=10, name="mid"),
        Marker(frame=50, name="second-mid"),
    ]
    return timeline


def test_fcpxml_markers_fall_inside_their_clip():
    root = ET.fromstring(fcpxml.dumps(_marker_timeline()))
    clips = root.findall("library/event/project/sequence/spine/asset-clip")
    seen = 0
    for clip in clips:
        start = _rational_seconds(clip.get("start"))
        duration = _rational_seconds(clip.get("duration"))
        assert start > 0  # in point > 0 for both golden clips
        for marker in clip.findall("marker"):
            seen += 1
            at = _rational_seconds(marker.get("start"))
            assert start <= at < start + duration, (clip.get("name"), at)
    assert seen == 3
    first = clips[0].findall("marker")
    # in 24 @ 24000/1001 + 10 sequence frames @ 24000/1001
    assert _rational_seconds(first[1].get("start")) == Fraction(34 * 1001, 24000)


def _otio_markers_in_range(data: dict) -> int:
    count = 0
    for track in data["tracks"]["children"]:
        for item in track["children"]:
            if item["OTIO_SCHEMA"] != "Clip.2":
                continue
            src = item["source_range"]
            lo = src["start_time"]["value"]
            hi = lo + src["duration"]["value"]
            for marker in item["markers"]:
                rng = marker["marked_range"]
                assert rng["start_time"]["rate"] == src["start_time"]["rate"]
                assert lo <= rng["start_time"]["value"] < hi, (item["name"], marker)
                count += 1
    return count


def test_otio_default_path_keeps_markers_inside_source_range():
    # Default options (otio_use_library left at its default) take the single
    # hand-written path: markers and warnings are always written.
    timeline = _marker_timeline()
    timeline.warnings = ["VFR source detected"]
    data = json.loads(otio.dumps(timeline))
    assert data["metadata"]["amv"]["warnings"] == ["VFR source detected"]
    assert _otio_markers_in_range(data) == 5  # 3 on V1, 2 on A1 (clip 1 only)
    assert otio.dumps(timeline, TimelineOptions(otio_use_library=True)) == \
        otio.dumps(timeline, TimelineOptions(otio_use_library=False))


def test_otio_markers_validated_by_library():
    otio_lib = pytest.importorskip("opentimelineio")
    text = otio.dumps(_marker_timeline())
    result = otio_lib.adapters.read_from_string(text, "otio_json")
    video = list(result.tracks)[0]
    clips = [item for item in video if isinstance(item, otio_lib.schema.Clip)]
    names = []
    for clip in clips:
        rng = clip.source_range
        for marker in clip.markers:
            names.append(marker.name)
            start = marker.marked_range.start_time
            assert rng.start_time <= start < rng.end_time_exclusive(), marker.name
    assert names == ["first", "mid", "second-mid"]


def test_drop_frame_inverse_known_value():
    fps = Fraction(30000, 1001)
    assert frames_to_timecode(3596, fps, True) == "00:01:59;28"
    assert timecode_to_frames("00:01:59;28", fps, True) == 3596


@pytest.mark.parametrize("fps", [Fraction(30000, 1001), Fraction(60000, 1001)])
def test_drop_frame_roundtrip_many_frames(fps):
    frames = list(range(0, 40000, 7)) + [107892, 107891, 215784, 1_000_000]
    for frame in frames:
        tc = frames_to_timecode(frame, fps, True)
        assert timecode_to_frames(tc, fps, True) == frame, (frame, tc)


def _fake_source(path: Path, payload: bytes) -> SourceInfo:
    path.write_bytes(payload)
    return SourceInfo(path=str(path), fps=Fraction(25), width=320, height=180,
                      duration_frames=100, has_audio=False)


def _fake_timeline(*sources: SourceInfo) -> Timeline:
    clips = [
        PlacedClip(source=s, in_frame=0, out_frame=10,
                   record_start=i * 10, record_end=i * 10 + 10, name=f"c{i}")
        for i, s in enumerate(sources)
    ]
    return Timeline(clips=clips, sequence_fps=Fraction(25), width=320,
                    height=180, title="t", include_audio=False)


def test_collect_never_overwrites_and_reuses_identical_copy(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    a = _fake_source(src_dir / "ep.mkv", b"A" * 100)
    media = tmp_path / "out" / "timeline_media"
    media.mkdir(parents=True)
    foreign = media / "ep.mkv"
    foreign.write_bytes(b"other project")

    created: list[str] = []
    updated, warnings, _ = collect_media(_fake_timeline(a), media, created=created)
    assert not warnings
    assert foreign.read_bytes() == b"other project"  # untouched
    dest = Path(updated.clips[0].source.path)
    assert dest.name == "ep_2.mkv" and dest.read_bytes() == b"A" * 100
    assert created == [str(dest)]

    # Second collect of the same source: the identical copy is reused.
    created2: list[str] = []
    again, _, _ = collect_media(_fake_timeline(a), media, created=created2)
    assert Path(again.clips[0].source.path) == dest
    assert created2 == []
    assert sorted(p.name for p in media.iterdir()) == ["ep.mkv", "ep_2.mkv"]


def test_collect_failed_copy_leaves_no_partial(tmp_path, monkeypatch):
    from backend.timeline import collect as collect_mod
    a = _fake_source(tmp_path / "ep.mkv", b"A" * 100)
    media = tmp_path / "media"

    def broken_copy(src, dst, *args, **kwargs):
        Path(dst).write_bytes(b"trunc")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(collect_mod.shutil, "copy2", broken_copy)
    updated, warnings, _ = collect_media(_fake_timeline(a), media)
    assert warnings and "Could not collect" in warnings[0]
    assert updated.clips[0].source.path == a.path
    assert list(media.iterdir()) == []


def test_export_failure_removes_this_runs_copies(tmp_path, monkeypatch):
    from backend.timeline import service
    a = _fake_source(tmp_path / "ep.mkv", b"A" * 100)
    monkeypatch.setattr(service, "prepare_timeline",
                        lambda specs, opts: _fake_timeline(a))

    def broken_writer(timeline, options):
        raise RuntimeError("writer failed")

    monkeypatch.setitem(service._WRITERS, "fcpxml", broken_writer)
    out = tmp_path / "out" / "timeline.fcpxml"
    with pytest.raises(RuntimeError):
        service.export_timeline([ClipSpec(source_path=a.path)], "fcpxml", str(out),
                                TimelineOptions(collect_media=True))
    assert not (tmp_path / "out" / "timeline_media").exists()
    assert not out.exists()
    assert list((tmp_path / "out").iterdir()) == []


def test_export_writes_timeline_atomically(tmp_path, monkeypatch):
    from backend.timeline import service
    a = _fake_source(tmp_path / "ep.mkv", b"A" * 100)
    monkeypatch.setattr(service, "prepare_timeline",
                        lambda specs, opts: _fake_timeline(a))
    out = tmp_path / "timeline.edl"
    out.write_text("previous export", encoding="utf-8")

    real_replace = service.os.replace

    def failing_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(service.os, "replace", failing_replace)
    with pytest.raises(OSError):
        service.export_timeline([ClipSpec(source_path=a.path)], "edl", str(out))
    assert out.read_text(encoding="utf-8") == "previous export"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ep.mkv", "timeline.edl"]

    monkeypatch.setattr(service.os, "replace", real_replace)
    result = service.export_timeline([ClipSpec(source_path=a.path)], "edl", str(out))
    assert result["ok"] and "FROM CLIP NAME" in out.read_text(encoding="utf-8")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ep.mkv", "timeline.edl"]


def test_api_bad_sequence_fps_is_a_400_not_a_500(client, db_path, clip):
    from backend.db import queries, schema

    with schema.get_conn(db_path) as conn:
        vid = queries.upsert_video(conn, str(clip), "Episode", 3000, 25.0, "320x240", "test",
                                   status="completed")
        sid = queries.insert_scene(conn, vid, 0, 0, 1000, None, None)
    res = client.post("/api/timeline/preview",
                      json={"scene_ids": [sid], "options": {"sequence_fps": "24/0"}})
    assert res.status_code == 400, res.text
