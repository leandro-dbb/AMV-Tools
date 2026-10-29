"""v06 bug-fix pass: interrupted indexing, swallowed model errors, HEVC."""
from __future__ import annotations

import threading

import pytest

from backend.db import queries, schema
from backend.indexing import frames, pipeline

from .conftest import FakeSigLIP, FakeTagger, make_solid_clip


class _Interrupt(Exception):
    """Stands for the app being closed in the middle of a phase."""


class _CountingTagger(FakeTagger):
    def __init__(self, fail_after: int | None = None):
        super().__init__()
        self.seen = 0
        self.fail_after = fail_after

    def tag_images(self, images):
        if self.fail_after is not None and self.seen + len(images) > self.fail_after:
            raise _Interrupt()
        self.seen += len(images)
        return super().tag_images(images)


class _CountingSigLIP(FakeSigLIP):
    def __init__(self, fail_after: int | None = None):
        super().__init__()
        self.seen = 0
        self.fail_after = fail_after

    def embed_images(self, images):
        if self.fail_after is not None and self.seen + len(images) > self.fail_after:
            raise _Interrupt()
        self.seen += len(images)
        return super().embed_images(images)


@pytest.fixture
def six_shots(tmp_path):
    return make_solid_clip(tmp_path / "six.mp4", ("red", "blue", "green", "white", "black", "yellow"))


def _run(db_path, clip, siglip, tagger, **kw):
    return pipeline.index_video(db_path, str(clip), siglip, tagger, mode="fast",
                                enable_sub_segmentation=False, batch_size=2, **kw)


def _video(db_path):
    with schema.get_conn(db_path) as conn:
        vid, status = conn.execute("SELECT id, status FROM videos").fetchone()
        stats = queries.db_stats(db_path)
        n_emb = conn.execute("SELECT COUNT(*) FROM scenes WHERE embedding IS NOT NULL").fetchone()[0]
        n_tagged = conn.execute("SELECT COUNT(DISTINCT scene_id) FROM scene_tags").fetchone()[0]
        n_scenes = conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]
    return dict(id=vid, status=status, stats=stats, emb=n_emb, tagged=n_tagged, scenes=n_scenes)


@pytest.mark.parametrize("stop_in", ["tag", "embed"])
def test_interrupted_video_is_incomplete_then_resumes_where_it_stopped(db_path, six_shots, monkeypatch, stop_in):
    schema.init_db(db_path)
    tagger = _CountingTagger(fail_after=2 if stop_in == "tag" else None)
    siglip = _CountingSigLIP(fail_after=4 if stop_in == "embed" else None)
    with pytest.raises(Exception):
        _run(db_path, six_shots, siglip, tagger)

    v = _video(db_path)
    assert v["scenes"] == 6
    # never shown as indexed: not counted, flagged as incomplete
    assert v["status"] != "completed"
    assert v["stats"]["videos"] == 0 and v["stats"]["scenes"] == 0 and v["stats"]["incomplete"] == 1
    tagged_before = tagger.seen
    embedded_before = siglip.seen

    # the resumed run neither detects cuts again nor rewrites the scene rows
    def no_decode(*a, **k):
        raise AssertionError("cut detection must not run again on resume")
    monkeypatch.setattr(pipeline.cuts, "detect_cuts", no_decode)
    monkeypatch.setattr(pipeline.single_pass, "detect_cuts_with_medians", no_decode)
    asked: list[int] = []
    real_extract = frames.extract_frames_at

    def spy(path, ts):
        asked.extend(ts)
        return real_extract(path, ts)
    monkeypatch.setattr(pipeline.frames, "extract_frames_at", spy)
    with schema.get_conn(db_path) as conn:
        ids_before = [r[0] for r in conn.execute("SELECT id FROM scenes ORDER BY id")]

    tagger2, siglip2 = _CountingTagger(), _CountingSigLIP()
    _run(db_path, six_shots, siglip2, tagger2)

    v = _video(db_path)
    assert v["status"] == "completed"
    assert v["stats"] == {**v["stats"], "videos": 1, "scenes": 6, "incomplete": 0}
    assert v["emb"] == 6 and v["tagged"] == 6
    with schema.get_conn(db_path) as conn:
        assert [r[0] for r in conn.execute("SELECT id FROM scenes ORDER BY id")] == ids_before
        # no scene tagged twice
        dup = conn.execute("SELECT COUNT(*) FROM (SELECT scene_id, tag, COUNT(*) c FROM scene_tags "
                           "GROUP BY scene_id, tag HAVING c > 1)").fetchone()[0]
        assert dup == 0
    # only the remaining work was done
    assert tagger2.seen == 6 - tagged_before
    assert siglip2.seen == 6 - embedded_before
    assert len(asked) == max(6 - tagged_before, 6 - embedded_before)


def test_unfinished_episode_is_picked_up_without_being_queued(client, db_path, six_shots, state):
    schema.init_db(db_path)
    with pytest.raises(Exception):
        _run(db_path, six_shots, _CountingSigLIP(fail_after=2), _CountingTagger())
    assert client.get("/api/databases").json()["databases"][0]["incomplete"] == 1
    # the queue is empty (folder already consumed): "Resume indexing" still finishes it
    pipeline.index_queue(db_path, _CountingSigLIP(), _CountingTagger(), mode="fast",
                         enable_sub_segmentation=False, batch_size=2)
    assert _video(db_path)["status"] == "completed"
    db = client.get("/api/databases").json()["databases"][0]
    assert (db["videos"], db["scenes"], db["incomplete"]) == (1, 6, 0)


def test_tagger_failure_is_not_swallowed(db_path, six_shots):
    """onnxruntime missing (or any tagger error) used to log a warning and mark
    the episode indexed without a single tag."""
    class Broken(FakeTagger):
        def tag_images(self, images):
            raise ModuleNotFoundError("No module named 'onnxruntime'")
    schema.init_db(db_path)
    events = []
    with schema.get_conn(db_path) as conn:
        queries.add_to_queue(conn, str(six_shots), False, False)
    pipeline.index_queue(db_path, FakeSigLIP(), Broken(), mode="fast", enable_sub_segmentation=False,
                         batch_size=2, progress_cb=events.append)
    assert _video(db_path)["status"] == "failed"
    assert any(e.get("failed") and "onnxruntime" in e.get("message", "") for e in events)


def test_pyav_log_callback_is_not_installed():
    """``av.logging.set_level(<int>)`` installs a Python log callback: the
    frame-threaded HEVC decoder then logs from its own threads while the
    indexing thread holds the GIL in ``avcodec_close`` — a deadlock that froze
    the whole sidecar (API included) on HEVC episodes."""
    import av
    import backend.indexing  # noqa: F401  (imports every indexing module)
    import backend.server  # noqa: F401
    assert av.logging.get_level() is None


def _hevc_clip(tmp_path, bits: int):
    import subprocess
    from backend.indexing import cuts
    out = tmp_path / f"hevc{bits}.mkv"
    pix = "yuv420p10le" if bits == 10 else "yuv420p"
    colors = ("red", "blue", "green", "white")
    cmd = [cuts._ffmpeg_path(), "-y", "-loglevel", "error"]
    for c in colors:
        cmd += ["-f", "lavfi", "-i", f"color=c={c}:s=320x240:d=1:r=24"]
    cmd += ["-filter_complex", "".join(f"[{i}:v]" for i in range(4)) + "concat=n=4:v=1:a=0[v]",
            "-map", "[v]", "-c:v", "libx265", "-pix_fmt", pix, "-x265-params", "log-level=error:frame-threads=2",
            str(out)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        pytest.skip("this ffmpeg has no libx265")
    return out


@pytest.mark.parametrize("bits", [8, 10])
@pytest.mark.parametrize("decode", ["sw", "pyav"])
def test_hevc_indexes_on_cpu_without_freezing(tmp_path, db_path, monkeypatch, bits, decode):
    monkeypatch.setattr(pipeline.cuts, "_ffmpeg_hw_profile", lambda: None)
    monkeypatch.setattr(pipeline.onepass, "available", lambda: False)
    monkeypatch.setenv("AMV_DECODE", decode)
    clip = _hevc_clip(tmp_path, bits)
    schema.init_db(db_path)
    done = threading.Event()
    err: list = []

    def run():
        try:
            _run(db_path, clip, FakeSigLIP(), FakeTagger())
        except Exception as e:  # pragma: no cover
            err.append(e)
        finally:
            done.set()
    t = threading.Thread(target=run, daemon=True)
    t.start()
    # the main thread keeps running while the indexing thread decodes HEVC
    ticks = 0
    while not done.wait(0.05):
        ticks += 1
    assert done.wait(60), "HEVC indexing hung"
    assert not err
    v = _video(db_path)
    assert v["status"] == "completed" and v["scenes"] == 4
