"""Music sync: analysis on synthetic signals with known beats/drops, the
shot planner, exports (timeline + markers), preview render and the API."""
from __future__ import annotations

import json
import shutil
import subprocess
import wave
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from backend.music import analysis as music_analysis
from backend.music import export as music_export
from backend.music import matcher
from backend.music.audio import AudioDecodeError, decode_audio, fingerprint_file
from backend.music.cache import AnalysisCache
from backend.music.preview import plan_segments, render_preview
from tests.conftest import ffmpeg_exe, make_solid_clip
from tests.eval import music_synth
from tests.eval.music_bench import match

SR = music_synth.SR


def _wav(path: Path, y: np.ndarray, sr: int = SR) -> Path:
    music_synth._write_wav(path, y, sr)
    return path


@pytest.fixture(scope="module")
def edm():
    y, truth = music_synth.edm()
    return music_analysis.analyze_signal(y, SR), truth


# ── analysis ────────────────────────────────────────────────────────────────
def test_metronome_tempo_beats_and_downbeats():
    y, truth = music_synth.metronome(120.0, 30.0)
    data = music_analysis.analyze_signal(y, SR)
    assert abs(data["bpm"] - 120.0) < 1.0
    beats = match([b / 1000 for b in data["beats_ms"]], truth["beats"], 0.07)
    assert beats["f1"] >= 0.97
    assert beats["mean_abs_err_ms"] < 25
    downs = match([b / 1000 for b in data["downbeats_ms"]], truth["downbeats"], 0.07)
    assert downs["f1"] >= 0.95
    assert data["drops_ms"] == []


def test_variable_tempo_is_followed():
    y, truth = music_synth.tempo_ramp(90.0, 130.0, 40.0)
    data = music_analysis.analyze_signal(y, SR)
    beats = match([b / 1000 for b in data["beats_ms"]], truth["beats"], 0.07)
    assert beats["f1"] >= 0.9


def test_edm_drops_build_ups_and_sections(edm):
    data, truth = edm
    assert abs(data["bpm"] - 128.0) < 1.5
    drops = [d / 1000 for d in data["drops_ms"]]
    assert len(drops) == len(truth["drops"]) == 2
    for est, ref in zip(drops, truth["drops"]):
        assert abs(est - ref) < 0.1
    types = [s["type"] for s in data["sections"]]
    for i, s in enumerate(data["sections"]):
        if s["type"] == "drop":
            assert data["sections"][i - 1]["type"] == "build"
    assert "bridge" in types and types[0] == "intro"
    # Sections tile the song.
    assert data["sections"][0]["start_ms"] == 0
    assert data["sections"][-1]["end_ms"] == data["duration_ms"]
    for a, b in zip(data["sections"], data["sections"][1:]):
        assert a["end_ms"] == b["start_ms"]
    # Drops are louder than build-ups.
    energy = {s["type"]: s["energy"] for s in data["sections"]}
    assert energy["drop"] > energy["build"] > 0
    assert len(data["waveform"]) == 1500
    assert all(0.0 <= v <= 1.0 for v in data["energy"]["values"])


def test_analysis_is_fast_enough():
    y, _ = music_synth.edm()
    # 4 minutes of material: the spec is < 15 s on CPU.
    long = np.concatenate([y, y])
    data = music_analysis.analyze_signal(long, SR)
    assert data["analysis_seconds"] < 15


@pytest.mark.parametrize("ext,args", [
    (".mp3", ["-c:a", "libmp3lame", "-b:a", "128k"]),
    (".flac", ["-c:a", "flac"]),
    (".ogg", ["-c:a", "libvorbis"]),
    (".m4a", ["-c:a", "aac", "-b:a", "128k"]),
])
def test_decode_common_formats(tmp_path, ext, args):
    y, _ = music_synth.metronome(120.0, 6.0)
    src = _wav(tmp_path / "click.wav", y)
    dst = tmp_path / f"click{ext}"
    proc = subprocess.run([ffmpeg_exe(), "-v", "error", "-y", "-i", str(src), *args, str(dst)],
                          capture_output=True)
    if proc.returncode != 0:
        pytest.skip(f"encoder unavailable for {ext}")
    out = decode_audio(dst)
    assert abs(len(out) / 22050 - 6.0) < 0.15


def test_decode_audio_track_of_a_video(tmp_path):
    y, _ = music_synth.metronome(120.0, 4.0)
    wav = _wav(tmp_path / "a.wav", y)
    video = tmp_path / "clip.mp4"
    subprocess.run([ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=4:r=25",
                    "-i", str(wav), "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", str(video)], check=True, capture_output=True)
    out = decode_audio(video)
    assert abs(len(out) / 22050 - 4.0) < 0.2


def test_decode_rejects_missing_and_silent_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        decode_audio(tmp_path / "nope.mp3")
    # A name that looks like an ffmpeg protocol is just a missing local file.
    with pytest.raises(FileNotFoundError):
        decode_audio("concat:/etc/passwd|/etc/hosts")
    video = tmp_path / "mute.mp4"
    make_solid_clip(video, ("red",), seconds_per_color=1.0)
    with pytest.raises(AudioDecodeError):
        decode_audio(video)


def test_cache_follows_content_not_path(tmp_path):
    y, _ = music_synth.metronome(128.0, 8.0)
    a = _wav(tmp_path / "song.wav", y)
    cache = AnalysisCache(tmp_path / "cache")
    first = cache.analyze(str(a))
    assert first["cached"] is False and first["hash"] == fingerprint_file(a)
    assert cache.analyze(str(a))["cached"] is True
    b = tmp_path / "renamed.wav"
    shutil.move(a, b)
    again = cache.analyze(str(b))
    assert again["cached"] is True and again["path"] == str(b.resolve())
    assert [t["hash"] for t in cache.tracks()] == [first["hash"]]
    with pytest.raises(ValueError):
        cache.get("../../etc/passwd")


# ── planner ─────────────────────────────────────────────────────────────────
def _shots(n: int = 60, seconds: float = 6.0) -> list[dict]:
    rng = np.random.default_rng(1)
    out = []
    for i in range(n):
        inten = i / (n - 1)
        out.append({
            "scene_id": i + 1, "video_id": 1 + i % 3, "video_name": f"ep{1 + i % 3}",
            "group": "A" if i % 2 else "B",
            "start_ms": int(i * seconds * 1000), "end_ms": int((i + 1) * seconds * 1000),
            "intensity": inten, "activity": inten, "motion_type": "shake" if inten > 0.8 else "static",
            "motion_magnitude": inten, "flash_count": int(inten > 0.9), "cut_rate": 0.0,
        })
    rng.shuffle(out)
    return out


def test_slots_tile_the_song_on_beats(edm):
    data, _ = edm
    slots = matcher.plan_slots(data, 0.5)
    assert slots[0].start_ms == 0 and slots[-1].end_ms == data["duration_ms"]
    for a, b in zip(slots, slots[1:]):
        assert a.end_ms == b.start_ms
    marks = np.array(sorted(set(data["beats_ms"]) | set(data["strong_onsets_ms"])
                            | {s["start_ms"] for s in data["sections"]}))
    for s in slots[1:]:
        assert np.min(np.abs(marks - s.start_ms)) == 0
    by_type: dict[str, list[int]] = {}
    for s in slots:
        by_type.setdefault(s.section_type, []).append(s.beats)
    assert max(by_type["drop"]) <= 2
    assert min(by_type["intro"]) >= 4
    # The build-up accelerates.
    build = [s for s in slots if s.section_type == "build" and s.section_index == 1]
    assert build[0].beats > build[-1].beats


def test_density_changes_cut_rate(edm):
    data, _ = edm
    calm = len(matcher.plan_slots(data, 0.0))
    mid = len(matcher.plan_slots(data, 0.5))
    nervous = len(matcher.plan_slots(data, 1.0))
    assert calm < mid < nervous


def test_proposal_matches_intensity_to_the_music(edm):
    data, _ = edm
    res = matcher.propose(data, _shots(), matcher.Constraints(), seed=3)
    clips = res["clips"]
    assert clips and clips[0]["start_ms"] == 0 and clips[-1]["end_ms"] == data["duration_ms"]
    types = {s["index"]: s["type"] for s in data["sections"]}
    loud = [c["intensity"] for c in clips if types[c["section_index"]] == "drop"]
    calm = [c["intensity"] for c in clips if types[c["section_index"]] in ("intro", "bridge", "outro")]
    assert np.mean(loud) > np.mean(calm) + 0.3
    for c in clips:
        assert c["src_out_ms"] - c["src_in_ms"] == c["end_ms"] - c["start_ms"]
        assert c["alternatives"]
        for alt in c["alternatives"]:
            assert alt["src_out_ms"] - alt["src_in_ms"] == c["end_ms"] - c["start_ms"]


def test_no_reuse_when_the_library_is_big_enough(edm):
    data, _ = edm
    res = matcher.propose(data, _shots(200, 3.0), matcher.Constraints(avoid_reuse=True), seed=1)
    windows: dict[int, list[tuple[int, int]]] = {}
    for c in res["clips"]:
        for a, b in windows.get(c["scene_id"], []):
            assert c["src_out_ms"] <= a or b <= c["src_in_ms"]
        windows.setdefault(c["scene_id"], []).append((c["src_in_ms"], c["src_out_ms"]))
    assert not any(w.startswith("reused") for w in res["warnings"])


def test_groups_locks_regenerate_and_keywords(edm):
    data, _ = edm
    shots = _shots()
    only_a = matcher.propose(data, shots, matcher.Constraints(groups=["A"]), seed=2)
    assert {c["group"] for c in only_a["clips"]} == {"A"}

    first = matcher.propose(data, shots, matcher.Constraints(), seed=2)["clips"]
    first[5]["locked"] = True
    locked = dict(first[5])
    again = matcher.propose(data, shots, matcher.Constraints(), previous=first, seed=99)["clips"]
    kept = [c for c in again if c["start_ms"] == locked["start_ms"]]
    assert kept and kept[0]["scene_id"] == locked["scene_id"] and kept[0]["locked"]

    drop_idx = next(s["index"] for s in data["sections"] if s["type"] == "drop")
    regen = matcher.propose(data, shots, matcher.Constraints(), previous=first,
                            regenerate_sections=[drop_idx], seed=7)["clips"]
    before = {c["start_ms"]: c["scene_id"] for c in first if c["section_index"] != drop_idx}
    after = {c["start_ms"]: c["scene_id"] for c in regen if c["section_index"] != drop_idx}
    assert before == after
    assert [c["start_ms"] for c in regen] == [c["start_ms"] for c in first]

    # "Goku" on the drop: the scenes matching the wish win that section.
    goku = {s["scene_id"]: 1.0 for s in shots if s["scene_id"] % 7 == 0}
    wished = matcher.propose(data, shots, matcher.Constraints(section_queries={drop_idx: "goku"}),
                             query_scores={drop_idx: goku}, seed=4)["clips"]
    on_drop = [c for c in wished if c["section_index"] == drop_idx]
    plain = [c for c in first if c["section_index"] == drop_idx]
    share = sum(c["scene_id"] in goku for c in on_drop) / len(on_drop)
    baseline = sum(c["scene_id"] in goku for c in plain) / len(plain)
    # 8 matching shots for 32 drop slots: they come back as often as the
    # anti-repeat rule allows.
    assert share > baseline + 0.3


def test_cut_offsets_measure_beat_alignment():
    clips = [{"start_ms": 0}, {"start_ms": 1010}, {"start_ms": 1990}]
    assert matcher.cut_offsets_ms(clips, [0, 1000, 2000], []) == [10, -10]


# ── exports ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def media(tmp_path_factory):
    root = tmp_path_factory.mktemp("music-media")
    video = make_solid_clip(root / "ep.mp4", ("red", "blue", "green", "yellow", "white", "black"),
                            seconds_per_color=2.0, fps=24)
    y, truth = music_synth.metronome(120.0, 12.0)
    song = _wav(root / "song.wav", y)
    data = music_analysis.analyze_audio(str(song))
    shots = [{"scene_id": i + 1, "video_id": 1, "video_name": "ep", "group": None,
              "start_ms": i * 2000, "end_ms": (i + 1) * 2000, "intensity": i / 5,
              "activity": 0.2, "motion_type": "static", "motion_magnitude": 0.1,
              "flash_count": 0} for i in range(6)]
    clips = matcher.propose(data, shots, matcher.Constraints(density=1.0), seed=1)["clips"]
    return {"video": video, "song": song, "analysis": data, "clips": clips,
            "paths": {i + 1: str(video) for i in range(6)}}


def test_export_fcp7_pins_cuts_on_beats_and_adds_music(media, tmp_path):
    res = music_export.export_edit("fcp7xml", str(tmp_path / "edit"), media["analysis"],
                                   str(media["song"]), media["clips"], media["paths"])
    out = Path(res["output"])
    assert out.suffix == ".xml" and out.read_text().startswith("<?xml")
    root = ET.fromstring(out.read_text().split("\n", 2)[2])
    fps = Fraction(24)
    starts = [int(ci.findtext("start")) for ci in root.findall("./sequence/media/video/track/clipitem")]
    assert starts == [round(c["start_ms"] * fps / 1000) for c in media["clips"]]
    tracks = root.findall("./sequence/media/audio/track")
    assert tracks and tracks[0].find("clipitem/file/name").text == "song.wav"
    assert int(tracks[0].findtext("clipitem/end")) == round(media["analysis"]["duration_ms"] * 24 / 1000)
    markers = root.findall("./sequence/marker")
    assert len(markers) == res["marker_count"] > len(media["analysis"]["sections"])
    # Cut error on the 24 fps grid: at most half a frame from the beat.
    for c in media["clips"][1:]:
        frame = round(c["start_ms"] * 24 / 1000)
        assert abs(frame / 24 * 1000 - c["start_ms"]) <= 1000 / 48 + 1


def test_export_other_formats(media, tmp_path):
    for fmt in ("fcpxml", "otio", "edl"):
        res = music_export.export_edit(fmt, str(tmp_path / f"edit-{fmt}"), media["analysis"],
                                       str(media["song"]), media["clips"], media["paths"],
                                       beat_markers=False, lang="fr")
        text = Path(res["output"]).read_text()
        if fmt == "fcpxml":
            root = ET.fromstring(text.partition("<fcpxml")[1] + text.partition("<fcpxml")[2])
            music = root.find(".//asset-clip[@ref='a-music']")
            assert music is not None and music.get("lane") == "-1"
            assert len(music.findall("marker")) == len(media["analysis"]["sections"])
        elif fmt == "otio":
            data = json.loads(text)
            kinds = [t["kind"] for t in data["tracks"]["children"]]
            assert kinds[:2] == ["Video", "Audio"]
            try:
                import opentimelineio as otio_lib
            except ImportError:
                otio_lib = None
            if otio_lib is not None:
                tl = otio_lib.adapters.read_from_string(text, "otio_json")
                assert len(tl.tracks) == 2
        else:
            assert "MUSIC" in text and "edl_no_markers" in res["warnings"]


@pytest.mark.parametrize("fmt", ["fcp7xml", "fcpxml", "edl", "otio", "csv"])
def test_markers_only_export(media, tmp_path, fmt):
    res = music_export.export_markers(fmt, str(tmp_path / "markers"), media["analysis"],
                                      str(media["song"]), beat_markers=True, lang="fr")
    text = Path(res["output"]).read_text(encoding="utf-8")
    n_beats = len(media["analysis"]["beats_ms"])
    assert res["marker_count"] >= n_beats
    if fmt == "fcp7xml":
        root = ET.fromstring(text.split("\n", 2)[2])
        assert len(root.findall("./sequence/marker")) == res["marker_count"]
    elif fmt == "edl":
        assert text.count("|M:") == res["marker_count"] and "01:00:00:00" in text
    elif fmt == "csv":
        assert text.splitlines()[0].startswith("index,seconds,timecode")
        assert len(text.splitlines()) == res["marker_count"] + 1
    elif fmt == "otio":
        assert len(json.loads(text)["tracks"]["children"][0]["children"][0]["markers"]) == res["marker_count"]


def test_preview_segments_are_frame_exact(media):
    segs = plan_segments(media["clips"])
    fps = Fraction(24000, 1001)
    total = sum(s["frames"] for s in segs)
    assert total == round(media["clips"][-1]["end_ms"] * fps / 1000) - round(media["clips"][0]["start_ms"] * fps / 1000)


def test_preview_render(media, tmp_path):
    av = pytest.importorskip("av")
    clips = media["clips"][:4]
    out = render_preview(clips, media["paths"], str(media["song"]), tmp_path / "p.mp4", workers=2)
    expected = sum(s["frames"] for s in plan_segments(clips))
    with av.open(str(out)) as container:
        frames = sum(1 for _ in container.decode(video=0))
    with av.open(str(out)) as container:
        assert len(container.streams.audio) == 1
    # Frame-exact: every cut sits on the frame its beat asked for.
    assert frames == expected


# ── API ─────────────────────────────────────────────────────────────────────
def _seed_library(db_path: str, video: Path, n: int = 6) -> None:
    from backend.db import schema

    schema.init_db(db_path)
    with schema.get_conn(db_path) as conn:
        conn.execute("INSERT INTO videos (filepath, display_name, duration_ms, fps, modified_at, status, group_name)"
                     " VALUES (?, 'ep', ?, 24, 0, 'completed', 'Show')", (str(video), n * 2000))
        for i in range(n):
            cur = conn.execute("INSERT INTO scenes (video_id, scene_index, start_ms, end_ms) VALUES (1, ?, ?, ?)",
                               (i, i * 2000, (i + 1) * 2000))
            conn.execute("INSERT INTO scene_metrics (scene_id, version, start_ms, end_ms, intensity, activity,"
                         " motion_type, motion_magnitude, flash_count) VALUES (?, 1, ?, ?, ?, 0.3, 'static', 0.1, 0)",
                         (cur.lastrowid, i * 2000, (i + 1) * 2000, i / n))


def test_api_flow(client, db_path, media, tmp_path):
    _seed_library(db_path, media["video"])
    r = client.post("/api/music/analyze", json={"path": str(media["song"])})
    assert r.status_code == 200, r.text
    track = r.json()
    assert track["bpm"] and track["beats_ms"] and track["cached"] is False
    h = track["hash"]
    assert client.post("/api/music/analyze", json={"path": str(media["song"])}).json()["cached"] is True
    assert client.get("/api/music/tracks").json()["tracks"][0]["hash"] == h
    assert client.get(f"/api/music/tracks/{h}/audio").status_code == 200
    assert client.get("/api/music/tracks/nothex/audio").status_code == 400
    assert client.get(f"/api/music/tracks/{'0' * 64}").status_code == 404
    assert client.post("/api/music/analyze", json={"path": str(tmp_path / "x.mp3")}).status_code == 404

    prop = client.post("/api/music/propose", json={"hash": h, "constraints": {"density": 0.8}, "seed": 1})
    assert prop.status_code == 200, prop.text
    body = prop.json()
    assert body["library"]["scenes"] == 6 and body["clips"]
    assert {c["group"] for c in body["clips"]} == {"Show"}

    out = tmp_path / "out" / "edit.xml"
    r = client.post("/api/music/export/timeline",
                    json={"hash": h, "clips": body["clips"], "format": "fcp7xml", "output_path": str(out)})
    assert r.status_code == 200, r.text
    assert out.exists() and r.json()["clip_count"] == len(body["clips"])
    r = client.post("/api/music/export/markers",
                    json={"hash": h, "format": "csv", "output_path": str(tmp_path / "m.csv"), "lang": "fr"})
    assert r.status_code == 200 and (tmp_path / "m.csv").exists()

    r = client.post("/api/music/preview", json={"hash": h, "clips": body["clips"][:3]})
    assert r.status_code == 200
    job = r.json()["id"]
    import time
    for _ in range(120):
        st = client.get(f"/api/music/preview/{job}").json()
        if st["status"] in ("done", "failed"):
            break
        time.sleep(0.25)
    assert st["status"] == "done", st
    assert client.get(f"/api/music/preview/{job}/video").status_code == 200
    assert client.get("/api/music/preview/unknown").status_code == 404


def test_beat_markers_are_numbered_in_the_bar():
    analysis = {"bpm": 120, "meter": 4, "sections": [], "beats_ms": [0, 500, 1000, 1500, 2000, 2500, 3000],
                "downbeats_ms": [1000, 3000]}
    names = [m.name for m in music_export.music_markers(analysis, Fraction(24), beat_markers=True)]
    assert names == ["3", "4", "1", "2", "3", "4", "1"]


def test_preview_finds_cuts_inside_shots_and_proposals_split_them(media, tmp_path):
    from backend.music.cache import ShotCuts
    from backend.music.library import split_shots

    # A clear change (black → white at 2.0 s); slot 1 straddles it.
    video = str(make_solid_clip(tmp_path / "bw.mp4", ("black", "white"), seconds_per_color=2.0,
                                size="640x360", fps=24))
    clips = [
        {"slot": 0, "scene_id": 1, "start_ms": 0, "end_ms": 1000, "src_in_ms": 200, "src_out_ms": 1200},
        {"slot": 1, "scene_id": 2, "start_ms": 1000, "end_ms": 2000, "src_in_ms": 1500, "src_out_ms": 2500},
    ]
    inner: list = []
    render_preview(clips, {1: video, 2: video}, str(media["song"]), tmp_path / "p.mp4", inner_cuts=inner)
    assert [x["slot"] for x in inner] == [1]
    assert all(abs(c - 2000) <= 60 for c in inner[0]["cuts_ms"])

    store = ShotCuts(tmp_path / "cuts.json")
    store.add(inner)
    shot = {"scene_id": 9, "filepath": video, "start_ms": 0, "end_ms": 4000}
    pieces = split_shots([shot], store.load())
    assert [(p["start_ms"], p["end_ms"]) for p in pieces] == [(0, inner[0]["cuts_ms"][0]),
                                                              (inner[0]["cuts_ms"][0], 4000)]


def test_preview_fills_gaps_with_black_and_letterboxes(media, tmp_path):
    av = pytest.importorskip("av")
    narrow = make_solid_clip(tmp_path / "ep43.mp4", ("red", "green"), seconds_per_color=2.0,
                             size="320x240", fps=25)
    clips = [  # starts at 0.5 s, hole between 1.5 and 2.0 s, 4:3 source
        {"slot": 0, "scene_id": 1, "start_ms": 500, "end_ms": 1500, "src_in_ms": 100, "src_out_ms": 1100},
        {"slot": 1, "scene_id": 2, "start_ms": 2000, "end_ms": 3000, "src_in_ms": 200, "src_out_ms": 1200},
    ]
    segs = plan_segments(clips)
    assert [bool(s.get("gap")) for s in segs] == [True, False, True, False]
    out = render_preview(clips, {1: str(media["video"]), 2: str(narrow)}, str(media["song"]), tmp_path / "g.mp4")
    with av.open(str(out)) as c:
        st = c.streams.video[0]
        frames = sum(1 for _ in c.decode(video=0))
        assert (st.codec_context.width, st.codec_context.height) == (640, 360)
    assert frames == round(3.0 * 24000 / 1001)


def test_otio_mixed_rates_keep_cuts_on_beats(media, tmp_path):
    other = make_solid_clip(tmp_path / "ep25.mp4", ("red", "blue", "green", "white", "black", "yellow"),
                            seconds_per_color=2.0, fps=25)
    paths = {i + 1: (str(media["video"]) if i % 2 == 0 else str(other)) for i in range(6)}
    res = music_export.export_edit("otio", str(tmp_path / "mixed"), media["analysis"], str(media["song"]),
                                   media["clips"], paths)
    data = json.loads(Path(res["output"]).read_text())
    video = data["tracks"]["children"][0]["children"]
    clips = iter(sorted(media["clips"], key=lambda c: c["start_ms"]))
    t = 0.0
    for item in video:
        if item["OTIO_SCHEMA"].startswith("Clip"):
            # each clip starts within half a sequence frame of its beat
            assert abs(t * 1000 - next(clips)["start_ms"]) <= 1000 / 48 + 1
        d = item["source_range"]["duration"]
        t += d["value"] / d["rate"]


def test_edl_music_reel_does_not_collide(media, tmp_path):
    music_named = tmp_path / "music.mp4"
    shutil.copy(media["video"], music_named)
    paths = {i + 1: str(music_named) for i in range(6)}
    res = music_export.export_edit("edl", str(tmp_path / "e"), media["analysis"], str(media["song"]),
                                   media["clips"], paths)
    lines = [ln for ln in Path(res["output"]).read_text().splitlines() if ln[:3].isdigit()]
    reels = {ln.split()[1] for ln in lines[:-1]}
    assert "MUSIC" in reels and lines[-1].split()[1] == "MUSIC2"


def test_markers_reject_bad_frame_rate(media, tmp_path):
    with pytest.raises(ValueError):
        music_export.export_markers("csv", str(tmp_path / "m"), media["analysis"], str(media["song"]), fps="1/0")


def test_regenerating_a_drop_keeps_intense_targets(edm):
    data, _ = edm
    shots = _shots()
    first = matcher.propose(data, shots, matcher.Constraints(), seed=2)["clips"]
    drop_idx = next(s["index"] for s in data["sections"] if s["type"] == "drop")
    regen = matcher.propose(data, shots, matcher.Constraints(), previous=first,
                            regenerate_sections=[drop_idx], seed=11)["clips"]
    on_drop = [c["intensity"] for c in regen if c["section_index"] == drop_idx]
    calm = [c["intensity"] for c in regen
            if next(s["type"] for s in data["sections"] if s["index"] == c["section_index"]) in ("intro", "bridge")]
    assert min(on_drop) > np.mean(calm)
