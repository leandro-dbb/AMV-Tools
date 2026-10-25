"""Semantic search endpoint with mocked embeddings."""
from __future__ import annotations

import numpy as np

from backend.db import queries, schema


def _seed(db_path, clip, vectors):
    with schema.get_conn(db_path) as conn:
        vid = queries.upsert_video(
            conn, str(clip), "Episode", 3000, 25.0, "320x240", "test", status="completed"
        )
        sids = []
        for i, vec in enumerate(vectors):
            sids.append(
                queries.insert_scene(
                    conn, vid, i, i * 1000, (i + 1) * 1000,
                    embedding=np.asarray(vec, dtype=np.float32),
                    thumbnail=b"",
                )
            )
    return vid, sids


def test_search_ranks_closest_embedding_first(client, db_path, clip, fake_models):
    siglip, _ = fake_models
    vec_a = siglip.embed_text("sunset")
    vec_b = siglip.embed_text("night city")
    _vid, sids = _seed(db_path, clip, [vec_b, vec_a])

    res = client.post(
        "/api/search",
        json={"query": "sunset", "threshold": -1.0, "top_k": 10},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["total"] == 2
    assert body["results"][0]["id"] == sids[1]
    assert body["results"][0]["score"] > body["results"][1]["score"]


def test_search_honours_threshold(client, db_path, clip, fake_models):
    siglip, _ = fake_models
    _seed(db_path, clip, [siglip.embed_text("sunset"), siglip.embed_text("night city")])

    res = client.post(
        "/api/search",
        json={"query": "sunset", "threshold": 0.99, "top_k": 10},
    )
    body = res.json()
    assert body["total"] == 1
    assert body["results"][0]["score"] > 0.99


def test_empty_library_returns_no_results(client, fake_models):
    res = client.post("/api/search", json={"query": "anything"})
    assert res.status_code == 200
    assert res.json() == {"results": [], "total": 0}


def test_search_history_is_recorded(client, db_path, clip, fake_models):
    siglip, _ = fake_models
    _seed(db_path, clip, [siglip.embed_text("sunset")])
    client.post("/api/search", json={"query": "sunset", "threshold": -1.0})

    history = client.get("/api/search/history").json()["queries"]
    assert history[0] == "sunset"
