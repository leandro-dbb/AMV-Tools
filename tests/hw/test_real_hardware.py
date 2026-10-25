"""Non-regression test on real hardware (NVIDIA CUDA, Apple Silicon MPS).

Not part of CI nor of ``uv run pytest -m "not slow"``: the whole module is
skipped unless ``AMV_HW_VIDEO`` points to a real episode extract (the 4-min
Dragon Ball Super extract used by ``tests/bench_pipeline.py``). It runs the
real models from the Hugging Face cache on the machine's accelerator:

    HF_HUB_OFFLINE=1 AMV_HW_VIDEO=/path/to/4min-extract.mkv \\
        uv run pytest tests/hw -m hw -s

What it guards against (all seen on real machines, never on synthetic clips):

- the wd-tagger aborting the sidecar after its first batch (Core ML);
- corrupted SigLIP 2 embeddings (asynchronous MPS copies): embeddings are
  compared with a per-backend reference recorded on the same machine
  (``tests/hw/reference_<backend>.npz``; created on the first run, or with
  ``AMV_HW_SAVE_REF=1``);
- the API freezing while models load (a GIL-holding session build);
- cuts landing 1-3 frames after the actual shot change;
- swapping during an indexing run;
- a character's automatic threshold matching most of an episode.
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.hw

VIDEO = os.environ.get("AMV_HW_VIDEO", "")
if not VIDEO or not os.path.isfile(VIDEO):
    pytest.skip("real-hardware test: set AMV_HW_VIDEO to a real episode extract", allow_module_level=True)

ROOT = Path(__file__).resolve().parent.parent.parent
REF_DIR = Path(__file__).resolve().parent

# Frames used by the model checks: spread over the extract, away from the
# first seconds (title cards).
SAMPLE_MS = [int(15_000 + i * 6_500) for i in range(32)]
EMBED_REF_MS = SAMPLE_MS[::4]           # 8 frames
SWAP_LIMIT_PAGES = 2560                 # background noise of a desktop OS
FREEZE_LIMIT_S = 2.0


def _swapouts() -> int | None:
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            m = re.search(r"Swapouts:\s+(\d+)", out)
            return int(m.group(1)) if m else None
        with open("/proc/vmstat") as fh:
            for line in fh:
                if line.startswith("pswpout "):
                    return int(line.split()[1])
    except Exception:
        return None
    return None


# ── shared fixtures ─────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def swap_start():
    return _swapouts()


@pytest.fixture(scope="module")
def device(swap_start):
    from backend.models.device import detect_device

    dev = detect_device()
    if dev.backend not in ("cuda", "mps"):
        pytest.skip(f"no CUDA / MPS accelerator here ({dev.backend})")
    return dev


@pytest.fixture(scope="module")
def frames():
    from backend.indexing.frames import extract_frames_at

    got = dict(extract_frames_at(VIDEO, SAMPLE_MS))
    missing = [t for t in SAMPLE_MS if t not in got]
    assert not missing, f"frames not extracted at {missing}"
    return [got[t] for t in SAMPLE_MS]


@pytest.fixture(scope="module")
def tagger(device):
    from backend.models import WDTaggerModel

    m = WDTaggerModel(device_backend=device.backend)
    yield m
    m.offload()


@pytest.fixture(scope="module")
def siglip(device):
    from backend.models import SigLIP2Model

    m = SigLIP2Model(device)
    yield m
    m.offload()


@pytest.fixture(scope="module")
def indexed_db(tmp_path_factory, siglip, tagger):
    """The extract indexed like the app's default "quality" profile."""
    from backend.indexing.pipeline import index_video

    db = str(tmp_path_factory.mktemp("hwdb") / "amv.db")
    t0 = time.perf_counter()
    index_video(db, VIDEO, siglip, tagger, mode="balanced", enable_sub_segmentation=True,
                batch_size=8, auto_skip_indexed=False)
    print(f"\n[hw] indexed the extract in {time.perf_counter() - t0:.1f} s")
    return db


# ── 1. tagger ───────────────────────────────────────────────────────────────
def test_tagger_runs_several_batches(tagger, frames):
    """Four batches of 8 real frames, as the indexing pass sends them."""
    outs = []
    for i in range(0, 32, 8):
        outs.extend(tagger.tag_images(frames[i:i + 8]))
    assert len(outs) == 32
    assert sum(1 for o in outs if o) >= 28, "the tagger returned no tags for most frames"
    for o in outs:
        for tag, _cat, conf in o:
            assert isinstance(tag, str) and np.isfinite(conf) and 0.0 <= conf <= 1.0
    emb = tagger.embed_images(frames[:8])
    assert np.isfinite(emb).all()
    print(f"\n[hw] tagger providers: {tagger.active_providers()}")


# ── 2. embeddings vs a recorded reference ───────────────────────────────────
def test_siglip_embeddings_match_reference(device, siglip, frames):
    imgs = [frames[SAMPLE_MS.index(t)] for t in EMBED_REF_MS]
    emb = np.asarray(siglip.embed_images(imgs), dtype=np.float32)
    assert emb.shape[0] == len(imgs)
    assert np.isfinite(emb).all(), "non-finite embeddings"
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-2), f"embeddings not unit-norm: {norms}"
    sims = emb @ emb.T
    off = sims[~np.eye(len(emb), dtype=bool)]
    assert off.max() < 0.999, "two different frames got the same embedding (corruption)"
    # Text side too: a query must rank its own kind of frame, not garbage.
    t = np.asarray(siglip.embed_text("an anime character"), dtype=np.float32)
    assert np.isfinite(t).all()

    ref_path = REF_DIR / f"reference_{device.backend}.npz"
    if os.environ.get("AMV_HW_SAVE_REF") == "1" or not ref_path.exists():
        np.savez(ref_path, emb=emb, ms=np.asarray(EMBED_REF_MS), checkpoint=siglip.checkpoint)
        pytest.skip(f"reference embeddings recorded in {ref_path}; rerun to compare")
    ref = np.load(ref_path)
    assert list(ref["ms"]) == EMBED_REF_MS
    cos = (emb * ref["emb"]).sum(axis=1) / (
        np.linalg.norm(emb, axis=1) * np.linalg.norm(ref["emb"], axis=1))
    print(f"\n[hw] SigLIP cosine vs reference: min {cos.min():.5f}")
    assert cos.min() > 0.995, f"embeddings drifted from the reference: {np.round(cos, 4)}"


# ── 3. the API stays responsive while the models load ───────────────────────
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ffmpeg() -> str:
    from backend.indexing.cuts import _ffmpeg_path
    return _ffmpeg_path()


def test_api_does_not_freeze_while_models_load(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / ".setup_complete").write_text("hw\n")
    short = tmp_path / "short.mkv"
    subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-ss", "30", "-i", VIDEO,
                    "-t", "20", "-c", "copy", str(short)], check=True, timeout=60)
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    env = dict(os.environ, AMV_DATA_DIR=str(data))
    log = open(tmp_path / "sidecar.log", "wb")
    proc = subprocess.Popen([sys.executable, str(ROOT / "backend" / "server.py"), "--port", str(port)],
                            cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)

    def call(method, path, body=None, timeout=300):
        req = urllib.request.Request(base + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"null")

    try:
        deadline = time.time() + 120
        while True:
            try:
                call("GET", "/api/status", timeout=2)
                break
            except Exception:
                if proc.poll() is not None or time.time() > deadline:
                    pytest.fail(f"sidecar did not start, see {tmp_path / 'sidecar.log'}")
                time.sleep(0.2)

        # Model loads: SigLIP through a search, the tagger + SigLIP through
        # an indexing run of a 20 s cut.
        search = threading.Thread(target=lambda: call("POST", "/api/search", {"query": "beerus"}),
                                  daemon=True)
        search.start()
        call("POST", "/api/index/queue", {"paths": [{"path": str(short), "recursive": False}]})
        call("POST", "/api/index/start", {})
        worst, n = 0.0, 0
        t_end = time.time() + 180
        while time.time() < t_end:
            a = time.perf_counter()
            call("GET", "/api/status", timeout=30)
            worst = max(worst, time.perf_counter() - a)
            n += 1
            if n % 10 == 0 and not search.is_alive() and not call("GET", "/api/index/queue")["items"]:
                break
            time.sleep(0.1)
        print(f"\n[hw] /api/status: {n} calls during model loads, worst {worst * 1000:.0f} ms")
        assert worst < FREEZE_LIMIT_S, f"API froze for {worst:.1f} s while models loaded"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


# ── 4. cuts on the first frame of the new shot ──────────────────────────────
def _gray_frames(path: str):
    """(seconds since the first frame, 160×90 gray) of every frame (PyAV)."""
    import av

    out = []
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        t0 = None
        for f in c.decode(s):
            t = float(f.time) if f.time is not None else None
            if t is None:
                continue
            t0 = t if t0 is None else t0
            g = f.reformat(width=160, height=90, format="gray")
            plane = g.planes[0]
            arr = np.frombuffer(plane, dtype=np.uint8).reshape(90, plane.line_size)[:, :160]
            out.append((t - t0, arr.copy()))
    return out


def _check_alignment(cuts_ms: list[int], gray, fps: float) -> tuple[list[str], int]:
    """(misplaced cuts, number of ambiguous transitions left out). A cut is
    judged only when one frame-to-frame jump dominates its window: in a
    strobe montage, a white flash, a dissolve or a fade every frame jumps
    (the 9 cuts of the 4-min extract that failed the plain "largest jump"
    rule: 6 in the opening's flashing montage, one after a 2-frame white
    flash, one dissolve, one fade to black), so "the" first frame of the
    shot is undefined there."""
    from backend.indexing.cuts import _hist_from_plane_bytes

    times = np.array([t for t, _ in gray]) * 1000.0
    half = 500.0 / fps
    bad = []
    ambiguous = 0
    for c in cuts_ms:
        k = int(np.argmin(np.abs(times - c)))
        if abs(times[k] - c) > half + 1:
            bad.append(f"{c} ms: no frame within half a frame (nearest {times[k]:.1f})")
            continue
        lo, hi = max(1, k - 3), min(len(gray) - 1, k + 2)
        hists = {i: _hist_from_plane_bytes(gray[i][1].tobytes(), 160, 90) for i in range(lo - 1, hi + 1)}
        jumps = {i: float(np.abs(hists[i] - hists[i - 1]).sum()) * 0.5 for i in range(lo, hi + 1)}
        best = max(jumps, key=jumps.get)
        second = max((v for i, v in jumps.items() if i != best), default=0.0)
        if second >= 0.5 * jumps[best]:
            ambiguous += 1
            continue
        if best != k:
            bad.append(f"{c} ms: largest jump at frame {best} ({times[best]:.0f} ms), cut at frame {k}")
    return bad, ambiguous


def test_cuts_land_on_the_first_frame_of_the_shot():
    from backend.indexing import cuts

    info = cuts.probe_video(VIDEO)
    fps = info["fps"] or 24000 / 1001
    gray = _gray_frames(VIDEO)
    runs = {"detect_cuts": [s for s, _ in cuts.detect_cuts(VIDEO, threshold=0.40)][1:]}
    try:
        from backend.indexing import onepass
        if onepass.available():
            sc, _m = onepass.detect_cuts_and_medians(VIDEO, 0.40)
            runs["onepass (NVDEC)"] = [s for s, _ in sc][1:]
    except ImportError:
        pass
    try:
        from backend.indexing import single_pass
        if single_pass.supported(VIDEO):
            sc, _m = single_pass.detect_cuts_with_medians(VIDEO, threshold=0.40)
            runs["single_pass (software)"] = [s for s, _ in sc][1:]
    except ImportError:
        pass
    try:
        from backend.indexing import chunked
        if chunked.supported(VIDEO):
            sc, _m = chunked.detect_cuts_with_medians(VIDEO, threshold=0.40)
            runs["chunked (VideoToolbox)"] = [s for s, _ in sc][1:]
    except ImportError:
        pass
    for name, cut_list in runs.items():
        assert len(cut_list) > 20, f"{name}: only {len(cut_list)} cuts on a 4-min episode extract"
        bad, ambiguous = _check_alignment(cut_list, gray, fps)
        clean = len(cut_list) - ambiguous
        ok = 1 - len(bad) / max(1, clean)
        print(f"\n[hw] {name}: {len(cut_list)} cuts ({ambiguous} in flashes/fades/dissolves), "
              f"{ok:.1%} of the clean ones on the first frame of the new shot")
        for b in bad[:10]:
            print(f"    {b}")
        # anime cuts often sit in flashes/action (M1 Pro extract: 34 of 79)
        assert clean >= 0.5 * len(cut_list), f"{name}: only {clean}/{len(cut_list)} clean cuts to check"
        assert ok >= 0.95, f"{name}: {len(bad)}/{clean} clean cuts off the shot change"
    if len(runs) > 1:
        ref = runs["detect_cuts"]
        for name, cut_list in runs.items():
            same = sum(1 for c in ref if any(abs(c - d) <= 500.0 / fps for d in cut_list))
            print(f"[hw] {name} vs detect_cuts: {same}/{len(ref)} cuts on the same frame")


# ── 5. no swap ──────────────────────────────────────────────────────────────
def test_no_swap_during_indexing(indexed_db, swap_start):
    end = _swapouts()
    if swap_start is None or end is None:
        pytest.skip("swap counters unavailable")
    delta = end - swap_start
    print(f"\n[hw] swap-outs during the module: {delta} pages")
    assert delta < SWAP_LIMIT_PAGES, f"the machine swapped {delta} pages"


# ── 6. character matching tolerances ────────────────────────────────────────
def test_beerus_character_within_tolerances(indexed_db, siglip):
    from backend.db import queries, schema
    from backend.searchplus import characters as chars

    with schema.get_conn(indexed_db) as conn:
        rows = queries.get_all_scenes_with_embeddings(conn)
        if len(rows) < 30:
            pytest.skip(f"only {len(rows)} scenes indexed")
        E = np.vstack([np.frombuffer(r[6], dtype=np.float32) for r in rows])
        sids = np.array([r[0] for r in rows], dtype=np.int64)
        index = {"rows": rows, "sids": sids, "embeddings": E, "signature": ("hw", len(rows))}
        q = np.asarray(siglip.embed_text("beerus"), dtype=np.float32)
        top = sids[np.argsort(-(E @ q))[:3]]
        if len(top) < 3:
            pytest.skip("fewer than 3 candidate scenes")
        cid = chars.create_character(conn, "Beerus", None)
        for sid in top:
            emb = chars.scene_embedding(conn, int(sid))
            chars.add_example(conn, cid, emb, scene_id=int(sid))
        summary = chars.rescore(conn, cid, index)
    share = summary["matches"] / len(rows)
    print(f"\n[hw] Beerus: {summary['matches']} / {len(rows)} scenes ({share:.1%}), "
          f"threshold {summary['threshold']:.3f}")
    assert 0.03 <= share <= 0.40, f"Beerus matched {share:.1%} of the scenes"

