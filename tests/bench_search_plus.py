"""Synthetic 50k-scene benchmark for search+ (filters, browse, similar,
characters). SigLIP is mocked — this measures the DB + numpy side only.

    uv run python tests/bench_search_plus.py            # 50 000 scenes, 768-d
    uv run python tests/bench_search_plus.py 100000 1152
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.db import schema  # noqa: E402
from backend.searchplus import analysis  # noqa: E402
from backend.searchplus import filters as sf  # noqa: E402

N_VIDEOS = 500
N_GROUPS = 20
N_CHARACTERS = 10


def build_library(db: str, n_scenes: int = 50_000, dim: int = 768, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    schema.init_db(db)
    per_video = max(1, n_scenes // N_VIDEOS)
    motions = np.array(analysis.MOTION_TYPES)
    colors = np.array(analysis.COLOR_BUCKETS)
    with schema.get_conn(db) as conn:
        conn.executemany(
            """INSERT INTO videos (id, filepath, display_name, duration_ms, fps, resolution,
                                   modified_at, model_version, status, group_name)
               VALUES (?, ?, ?, 1320000, 24, '1920x1080', 0, 'bench', 'completed', ?)""",
            [(v + 1, f"/bench/ep{v:04d}.mkv", f"Episode {v:04d}", f"Series {v % N_GROUPS:02d}")
             for v in range(N_VIDEOS)],
        )
        sid = 0
        scene_rows, metric_rows = [], []
        for v in range(N_VIDEOS):
            t = 0
            for i in range(per_video):
                if sid >= n_scenes:
                    break
                sid += 1
                dur = int(rng.integers(300, 9000))
                emb = rng.normal(size=dim).astype(np.float32)
                emb /= np.linalg.norm(emb)
                scene_rows.append((sid, v + 1, i, t, t + dur, emb.tobytes()))
                pal = json.dumps([["#aa3322", 0.5, "red"], ["#2233aa", 0.3, "blue"], ["#ffffff", 0.2, "white"]])
                metric_rows.append((
                    sid, analysis.ANALYSIS_VERSION, t, t + dur, str(rng.choice(motions)),
                    float(rng.random()), 0.0, 0.0, 0.0, float(rng.random()), float(rng.random()),
                    0.0, 0, float(rng.random()), str(rng.choice(colors)), str(rng.choice(colors)), pal, 0.0,
                ))
                t += dur
        conn.executemany(
            "INSERT INTO scenes (id, video_id, scene_index, start_ms, end_ms, embedding) VALUES (?, ?, ?, ?, ?, ?)",
            scene_rows,
        )
        conn.executemany(
            f"INSERT INTO scene_metrics VALUES ({','.join('?' * 18)})", metric_rows,
        )
        conn.executemany(
            "INSERT INTO derush_items (scene_id, favorite) VALUES (?, ?)",
            [(int(s), int(s % 3 == 0)) for s in rng.choice(np.arange(1, sid + 1), size=min(3000, sid), replace=False)],
        )
        for c in range(N_CHARACTERS):
            cur = conn.execute(
                "INSERT INTO characters (name, group_name, threshold, scored_signature) VALUES (?, ?, 0.75, 'bench')",
                (f"Char{c}", f"Series {c % N_GROUPS:02d}"),
            )
            cid = cur.lastrowid
            picks = rng.choice(np.arange(1, sid + 1), size=min(3000, sid), replace=False)
            conn.executemany(
                "INSERT INTO character_matches VALUES (?, ?, ?)",
                [(cid, int(s), float(0.6 + 0.4 * rng.random())) for s in picks],
            )
        conn.execute("ANALYZE")


COMBINED = sf.SearchFilters(
    motion=["pan_left", "pan_right", "zoom_in"], intensity_min=0.5, duration_max_ms=4000,
    colors=["red", "blue"], groups=["Series 01", "Series 02", "Series 03"],
)
WITH_CHAR = sf.SearchFilters(characters=[1, 2], intensity_min=0.3)
FAVS = sf.SearchFilters(favorites=True, motion=["static"])


def _ms(fn, repeats: int) -> float:
    out = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000)
    return statistics.median(out)


def time_queries(db: str, repeats: int = 5) -> dict[str, float]:
    """Function-level timings (no HTTP)."""
    with schema.get_conn(db) as conn:
        rows = conn.execute("SELECT id, embedding FROM scenes ORDER BY id").fetchall()
        sids = np.array([r[0] for r in rows], dtype=np.int64)
        E = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        q = E[123]
        res = {
            "vector_topk_ms": _ms(lambda: sf.rank(E @ q, 0.0, 200, "relevance"), repeats),
            "filters_sql_ms": _ms(lambda: sf.filter_mask(conn, COMBINED, sids), repeats),
            "filters_char_ms": _ms(lambda: sf.filter_mask(conn, WITH_CHAR, sids), repeats),
            "filters_fav_ms": _ms(lambda: sf.filter_mask(conn, FAVS, sids), repeats),
            "browse_ms": _ms(lambda: sf.browse(conn, COMBINED, top_k=200), repeats),
            "browse_char_ms": _ms(lambda: sf.browse(conn, WITH_CHAR, top_k=200), repeats),
        }
        sample = [{"id": int(s)} for s in sids[:200]]
        res["enrich_200_ms"] = _ms(lambda: sf.enrich(conn, [dict(x) for x in sample]), repeats)
        res["n_scenes"] = float(len(sids))
    return res


def _baseline_app():
    """The v0.3.0 /api/search handler loaded straight from git, for an A/B
    against the filter-aware one on the same DB and warm index cache."""
    import importlib.util
    import subprocess
    try:
        src = subprocess.run(["git", "show", "6efca1f:backend/api/search.py"], cwd=_ROOT,
                             capture_output=True, text=True, check=True).stdout
    except Exception:
        return None
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    tmp = Path(tempfile.mkdtemp(prefix="amv_v03_")) / "search_v03.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location("backend.api._search_v03", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # pydantic resolves postponed annotations via sys.modules
    spec.loader.exec_module(mod)
    app = FastAPI()
    app.include_router(mod.router)
    return TestClient(app)


def time_endpoints(db: str, dim: int, repeats: int = 5) -> dict[str, float]:
    """End-to-end through FastAPI (search index cache warm, SigLIP mocked)."""
    os.environ.setdefault("AMV_DATA_DIR", tempfile.mkdtemp(prefix="amv_bench_"))
    import backend.state as st
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api import characters as characters_api, search, searchplus

    st._state = None
    state = st.get_state()
    state._settings["databases"] = {"active": [db], "primary": db}
    state.invalidate_search_cache()
    rng = np.random.default_rng(1)

    class _Fake:
        checkpoint = "bench"

        def embed_text(self, text):
            v = rng.normal(size=dim).astype(np.float32)
            return v / np.linalg.norm(v)

    state.get_siglip = lambda: _Fake()
    app = FastAPI()
    for r in (search.router, searchplus.router, characters_api.router):
        app.include_router(r)
    c = TestClient(app)
    t = time.perf_counter()
    state.get_search_index()
    out = {"index_build_ms": (time.perf_counter() - t) * 1000}
    combined = COMBINED.model_dump()
    body = {"query": "x", "threshold": -1}
    base = _baseline_app()
    if base is not None:
        # Interleaved A/B so machine load hits both handlers equally.
        new_t, old_t = [], []
        for _ in range(max(repeats, 7)):
            new_t.append(_ms(lambda: c.post("/api/search", json=body), 1))
            old_t.append(_ms(lambda: base.post("/api/search", json=body), 1))
        out["search_plain_ms"] = statistics.median(new_t)
        out["search_plain_v03_ms"] = statistics.median(old_t)
    else:
        out["search_plain_ms"] = _ms(lambda: c.post("/api/search", json=body), repeats)
    # Per-search freshness check on a warm index (v0.4 search+ ran a full
    # COUNT/MAX/SUM scan here; it is now an in-memory lookup).
    out["index_lookup_ms"] = _ms(state.get_search_index, repeats)
    out["search_filtered_ms"] = _ms(
        lambda: c.post("/api/search", json={"query": "x", "threshold": -1, "filters": combined}), repeats)
    out["browse_ms"] = _ms(lambda: c.post("/api/search", json={"filters": combined}), repeats)
    out["similar_ms"] = _ms(lambda: c.get("/api/search/similar/4242", params={"exclude_same_video": True}), repeats)
    out["similar_filtered_ms"] = _ms(
        lambda: c.get("/api/search/similar/4242", params={"filters": json.dumps(combined)}), repeats)
    # Character rescore: give Char1 three real examples first.
    from backend.searchplus import characters as chars
    with schema.get_conn(db) as conn:
        for s in (10, 20, 30):
            chars.add_example(conn, 1, chars.scene_embedding(conn, s), scene_id=s)
    out["character_rescore_ms"] = _ms(lambda: c.post("/api/characters/1/rescore"), repeats)
    st._state = None
    return out


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    dim = int(sys.argv[2]) if len(sys.argv) > 2 else 768
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "bench.db")
        t = time.perf_counter()
        build_library(db, n, dim)
        print(f"built {n} scenes × {dim}-d in {time.perf_counter() - t:.1f}s "
              f"({os.path.getsize(db) / 1e6:.0f} MB)")
        for k, v in time_queries(db).items():
            print(f"  {k:24s} {v:9.2f}")
        for k, v in time_endpoints(db, dim).items():
            print(f"  {k:24s} {v:9.2f}")


if __name__ == "__main__":
    main()
