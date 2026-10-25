"""User-defined characters: a few example scenes (or dropped images, or
crops of a scene) → SigLIP 2 prototype → every scene of the series scored.

No new model: examples reuse the scene embeddings already in the DB; only a
crop or an uploaded image needs one SigLIP image forward pass.

Scoring (all vectors L2-normalised, so dot = cosine):

    s_proto = E · normalise(mean(positives))
    s_knn   = max_i E · positive_i         (copes with outfits/angles)
    s       = ½ s_proto + ½ s_knn − max(0, max_j E · negative_j − s_knn)

The negative term only bites when a scene is closer to a rejected example
than to any accepted one, which is what the validate/reject loop needs.
Explicit examples override the score: positives → 1.0, negatives → dropped.
"""
from __future__ import annotations

import io
import json
import logging
import sqlite3
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.75
MIN_THRESHOLD, MAX_THRESHOLD = 0.40, 0.97
# Matches are stored down to (threshold - margin) so the UI can show the
# borderline candidates worth validating/rejecting.
CANDIDATE_MARGIN = 0.12
MAX_STORED_MATCHES = 5000


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-12 else v
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def score_embeddings(E: np.ndarray, pos: np.ndarray, neg: Optional[np.ndarray] = None) -> np.ndarray:
    """Score every row of ``E`` (N×D) against positives (P×D) / negatives (Q×D)."""
    if E.size == 0 or pos is None or len(pos) == 0:
        return np.zeros(len(E), dtype=np.float32)
    P = _norm(pos)
    proto = _norm(P.mean(axis=0))
    s_proto = E @ proto
    s_knn = (E @ P.T).max(axis=1)
    s = 0.5 * s_proto + 0.5 * s_knn
    if neg is not None and len(neg):
        s_neg = (E @ _norm(neg).T).max(axis=1)
        s = s - np.maximum(0.0, s_neg - s_knn)
    return s.astype(np.float32)


# The example-only threshold assumes unrelated scenes score low. Within one
# series SigLIP 2 rates every shot as fairly similar (same drawing style,
# palette, lighting): on a real 24-min episode the median scene scored 0.74
# against "Beerus", the 3-example threshold was 0.70 and 262 of 418 scenes
# matched (precision ~0.11). The automatic threshold therefore never goes
# below the score of the library's 90th percentile — a character is assumed
# to appear in at most ~10 % of the scenes in scope until the user lowers the
# threshold or validates more examples (that episode: precision 0.56-0.67,
# recall 0.80-0.83). Needs enough scenes for the percentile to mean anything.
LIBRARY_FLOOR_QUANTILE = 0.90
LIBRARY_FLOOR_MIN_SCENES = 100


def _library_floor(scores: np.ndarray, eligible: np.ndarray, pos_sids: set, sids: np.ndarray) -> float:
    pool = eligible.copy()
    if pos_sids:
        pool &= ~np.isin(sids, np.fromiter(pos_sids, dtype=np.int64))
    if int(pool.sum()) < LIBRARY_FLOOR_MIN_SCENES:
        return MIN_THRESHOLD
    floor = float(np.quantile(scores[pool], LIBRARY_FLOOR_QUANTILE))
    return float(min(max(floor, MIN_THRESHOLD), MAX_THRESHOLD))


def suggest_threshold(pos: np.ndarray, neg: Optional[np.ndarray] = None) -> float:
    """Pick a starting threshold from the examples themselves: a bit below
    the leave-one-out score of the least typical positive, and above the
    best-scoring negative when that is still compatible."""
    pos = _norm(pos) if len(pos) else pos
    thr = DEFAULT_THRESHOLD
    if len(pos) >= 2:
        loo = []
        for i in range(len(pos)):
            others = np.delete(pos, i, axis=0)
            loo.append(float(score_embeddings(pos[i:i + 1], others)[0]))
        # Few examples underestimate the character's spread (outfits,
        # angles, lighting): be more permissive and let feedback tighten.
        thr = min(loo) - (0.04 + 0.12 / np.sqrt(len(pos)))
    if neg is not None and len(neg) and len(pos):
        best_neg = float(score_embeddings(_norm(neg), pos, None).max())
        if best_neg + 0.02 < MAX_THRESHOLD:
            thr = max(thr, best_neg + 0.02)
    return float(min(max(thr, MIN_THRESHOLD), MAX_THRESHOLD))


# ── DB layer ────────────────────────────────────────────────────────────────
def _emb(blob: Optional[bytes]) -> Optional[np.ndarray]:
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def list_characters(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """SELECT c.id, c.name, c.group_name, c.threshold, c.threshold_auto,
                  (SELECT COUNT(*) FROM character_examples e WHERE e.character_id = c.id AND e.polarity > 0),
                  (SELECT COUNT(*) FROM character_examples e WHERE e.character_id = c.id AND e.polarity < 0),
                  (SELECT COUNT(*) FROM character_matches m WHERE m.character_id = c.id AND m.score >= c.threshold),
                  (SELECT e.scene_id FROM character_examples e
                    WHERE e.character_id = c.id AND e.polarity > 0 AND e.scene_id IS NOT NULL
                    ORDER BY e.id LIMIT 1),
                  (SELECT e.id FROM character_examples e
                    WHERE e.character_id = c.id AND e.polarity > 0 ORDER BY e.id LIMIT 1),
                  c.scored_signature
             FROM characters c ORDER BY c.name COLLATE NOCASE"""
    ).fetchall()
    return [
        {"id": r[0], "name": r[1], "group_name": r[2], "threshold": r[3],
         "threshold_auto": bool(r[4]), "positives": r[5], "negatives": r[6],
         "matches": r[7], "cover_scene_id": r[8], "cover_example_id": r[9],
         "scored": r[10] is not None}
        for r in rows
    ]


def get_character(conn: sqlite3.Connection, cid: int) -> Optional[dict]:
    for c in list_characters(conn):
        if c["id"] == cid:
            return c
    return None


def create_character(conn: sqlite3.Connection, name: str, group_name: Optional[str]) -> int:
    name = name.strip()
    if not name:
        raise ValueError("name is required")
    dup = conn.execute(
        "SELECT id FROM characters WHERE name = ? COLLATE NOCASE AND COALESCE(group_name, '') = ?",
        (name, group_name or ""),
    ).fetchone()
    if dup:
        raise ValueError("a character with this name already exists for this series")
    cur = conn.execute(
        "INSERT INTO characters (name, group_name, threshold) VALUES (?, ?, ?)",
        (name, group_name, DEFAULT_THRESHOLD),
    )
    return int(cur.lastrowid)


def examples(conn: sqlite3.Connection, cid: int) -> list[dict]:
    rows = conn.execute(
        """SELECT id, scene_id, polarity, crop, embedding, source, thumbnail IS NOT NULL
             FROM character_examples WHERE character_id = ? ORDER BY id""",
        (cid,),
    ).fetchall()
    return [
        {"id": r[0], "scene_id": r[1], "polarity": r[2],
         "crop": json.loads(r[3]) if r[3] else None,
         "embedding": _emb(r[4]), "source": r[5], "has_thumbnail": bool(r[6])}
        for r in rows
    ]


def add_example(conn: sqlite3.Connection, cid: int, embedding: np.ndarray, *,
                scene_id: Optional[int] = None, polarity: int = 1,
                crop: Optional[list[float]] = None, thumbnail: Optional[bytes] = None,
                source: str = "scene") -> int:
    """Insert an example. A full-frame scene example replaces any earlier
    full-frame example of the same scene (so flipping accept↔reject works)."""
    polarity = 1 if polarity >= 0 else -1
    if scene_id is not None and crop is None:
        conn.execute(
            "DELETE FROM character_examples WHERE character_id = ? AND scene_id = ? AND crop IS NULL",
            (cid, scene_id),
        )
    cur = conn.execute(
        """INSERT INTO character_examples (character_id, scene_id, polarity, crop, embedding, thumbnail, source)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (cid, scene_id, polarity, json.dumps(crop) if crop else None,
         _norm(embedding).astype(np.float32).tobytes(), thumbnail, source),
    )
    return int(cur.lastrowid)


def scene_embedding(conn: sqlite3.Connection, scene_id: int) -> Optional[np.ndarray]:
    row = conn.execute("SELECT embedding FROM scenes WHERE id = ?", (scene_id,)).fetchone()
    return _emb(row[0]) if row else None


def _row_video_ids(index: dict) -> np.ndarray:
    cached = index.get("_row_video_ids")
    if cached is None or len(cached) != len(index["rows"]):
        cached = np.fromiter((r[1] for r in index["rows"]), dtype=np.int64, count=len(index["rows"]))
        index["_row_video_ids"] = cached
    return cached


def signature_key(index: dict) -> str:
    return json.dumps(list(index.get("signature") or []))


def rescore(conn: sqlite3.Connection, cid: int, index: dict) -> dict:
    """Recompute prototype + matches for one character against the search
    index (completed videos). Returns a small summary."""
    row = conn.execute(
        "SELECT group_name, threshold, threshold_auto FROM characters WHERE id = ?", (cid,)
    ).fetchone()
    if row is None:
        raise KeyError(cid)
    group_name, threshold, thr_auto = row
    E = index["embeddings"]
    dim = E.shape[1] if E.ndim == 2 and E.size else None

    exs = examples(conn, cid)
    pos, neg, pos_sids, neg_sids, skipped = [], [], set(), set(), 0
    for ex in exs:
        emb = ex["embedding"]
        # Model switched since the example was added (e.g. base → so400m):
        # a plain scene example can be refreshed from the scene's current
        # embedding; crops/images can't and are skipped until re-added.
        if dim is not None and emb is not None and emb.shape[0] != dim:
            fresh = scene_embedding(conn, ex["scene_id"]) if ex["scene_id"] and not ex["crop"] else None
            if fresh is not None and fresh.shape[0] == dim:
                emb = fresh
                conn.execute("UPDATE character_examples SET embedding = ? WHERE id = ?",
                             (fresh.astype(np.float32).tobytes(), ex["id"]))
            else:
                skipped += 1
                continue
        if emb is None:
            continue
        if ex["polarity"] > 0:
            pos.append(emb)
            if ex["scene_id"] is not None:
                pos_sids.add(int(ex["scene_id"]))
        else:
            neg.append(emb)
            if ex["scene_id"] is not None:
                neg_sids.add(int(ex["scene_id"]))

    conn.execute("DELETE FROM character_matches WHERE character_id = ?", (cid,))
    sig = signature_key(index)
    if not pos or dim is None:
        conn.execute(
            "UPDATE characters SET prototype = NULL, scored_signature = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (sig, cid),
        )
        return {"matches": 0, "threshold": threshold, "skipped_examples": skipped}

    P = np.vstack(pos).astype(np.float32)
    N = np.vstack(neg).astype(np.float32) if neg else None
    if thr_auto:
        threshold = suggest_threshold(P, N)
    scores = score_embeddings(E, P, N)

    sids = index["sids"]
    eligible = np.ones(len(sids), dtype=bool)
    if group_name:
        vids = [r[0] for r in conn.execute("SELECT id FROM videos WHERE group_name = ?", (group_name,))]
        eligible = np.isin(_row_video_ids(index), np.asarray(vids, dtype=np.int64))
    if pos_sids:
        scores[np.isin(sids, np.fromiter(pos_sids, dtype=np.int64))] = 1.0
    if neg_sids:
        eligible &= ~np.isin(sids, np.fromiter(neg_sids, dtype=np.int64))

    if thr_auto:
        threshold = max(threshold, _library_floor(scores, eligible, pos_sids, sids))

    floor = threshold - CANDIDATE_MARGIN
    keep = np.where(eligible & (scores >= floor))[0]
    if len(keep) > MAX_STORED_MATCHES:
        keep = keep[np.argsort(-scores[keep])[:MAX_STORED_MATCHES]]
    conn.executemany(
        "INSERT INTO character_matches (character_id, scene_id, score) VALUES (?, ?, ?)",
        ((cid, int(sids[i]), float(scores[i])) for i in keep),
    )
    proto = _norm(P.mean(axis=0)).astype(np.float32)
    conn.execute(
        """UPDATE characters SET prototype = ?, threshold = ?, scored_signature = ?,
                  updated_at = CURRENT_TIMESTAMP WHERE id = ?""",
        (proto.tobytes(), float(threshold), sig, cid),
    )
    above = int((scores[keep] >= threshold).sum())
    return {"matches": above, "candidates": int(len(keep)), "threshold": float(threshold),
            "skipped_examples": skipped}


def ensure_fresh(conn: sqlite3.Connection, index: dict, ids: Optional[list[int]] = None) -> int:
    """Rescore characters whose matches predate the current search index
    (new episodes indexed, scenes merged…). Cheap: one matmul each."""
    sig = signature_key(index)
    rows = conn.execute("SELECT id, scored_signature FROM characters").fetchall()
    n = 0
    for cid, scored in rows:
        if ids is not None and cid not in ids:
            continue
        if scored != sig:
            try:
                rescore(conn, cid, index)
                n += 1
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("character %s rescore failed: %s", cid, exc)
    return n


def matches(conn: sqlite3.Connection, cid: int, *, limit: int = 300, offset: int = 0) -> list[dict]:
    labels = {int(r[0]): int(r[1]) for r in conn.execute(
        "SELECT scene_id, polarity FROM character_examples WHERE character_id = ? AND scene_id IS NOT NULL AND crop IS NULL",
        (cid,),
    )}
    rows = conn.execute(
        """SELECT s.id, s.video_id, v.display_name, s.scene_index, s.start_ms, s.end_ms, m.score
             FROM character_matches m
             JOIN scenes s ON s.id = m.scene_id
             JOIN videos v ON v.id = s.video_id
            WHERE m.character_id = ?
            ORDER BY m.score DESC, s.id
            LIMIT ? OFFSET ?""",
        (cid, int(limit), int(offset)),
    ).fetchall()
    out = []
    for r in rows:
        lab = labels.get(int(r[0]))
        out.append({
            "id": r[0], "video_id": r[1], "video_display": r[2], "scene_index": r[3],
            "start_ms": r[4], "end_ms": r[5], "score": float(r[6]),
            "label": "positive" if lab == 1 else ("negative" if lab == -1 else None),
        })
    return out


# ── images ──────────────────────────────────────────────────────────────────
def crop_image(img, crop: Optional[list[float]]):
    """Crop a PIL image with a normalised [x0, y0, x1, y1] box."""
    if not crop:
        return img
    x0, y0, x1, y1 = (min(max(float(v), 0.0), 1.0) for v in crop)
    if x1 - x0 < 0.02 or y1 - y0 < 0.02:
        raise ValueError("crop box is too small")
    w, h = img.size
    return img.crop((int(x0 * w), int(y0 * h), max(int(x1 * w), int(x0 * w) + 1),
                     max(int(y1 * h), int(y0 * h) + 1)))


def thumb_jpeg(img, size=(240, 240)) -> bytes:
    t = img.copy().convert("RGB")
    t.thumbnail(size)
    buf = io.BytesIO()
    t.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


def scene_frame(conn: sqlite3.Connection, scene_id: int):
    """Median frame of a scene as a PIL image: decoded from the source when it
    is reachable (full quality crops), else the stored 320×180 thumbnail."""
    from PIL import Image
    row = conn.execute(
        """SELECT v.filepath, s.start_ms, s.end_ms, s.thumbnail
             FROM scenes s JOIN videos v ON v.id = s.video_id WHERE s.id = ?""",
        (scene_id,),
    ).fetchone()
    if row is None:
        raise KeyError(scene_id)
    fp, s_ms, e_ms, thumb = row
    try:
        import os
        if fp and os.path.exists(fp):
            from ..indexing import frames
            mid = s_ms + (e_ms - s_ms) // 2
            for _ts, img in frames.extract_frames_at(fp, [mid]):
                return img.convert("RGB")
    except Exception as exc:
        log.debug("scene %s frame decode failed, using thumbnail: %s", scene_id, exc)
    if thumb:
        return Image.open(io.BytesIO(thumb)).convert("RGB")
    raise ValueError("scene has no thumbnail and its video is unreachable")
