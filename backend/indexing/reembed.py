"""Embedding versioning + embeddings-only re-index.

Every database is bound to ONE embedding model: search embeds its queries
with that model and new videos indexed into it use it too, so two embedding
spaces never mix in a database. Changing the model (a new tier, a new
default) re-embeds the existing scenes — no new cuts, no re-tagging — from
the same 512x288 median frames the indexing pipeline feeds SigLIP, in the
background while search keeps running on the old vectors; the switch is one
transaction at the end.

Storage (additive, created lazily here — no schema version bump):

* ``settings['embedding_model']``: the database's model (HF repo id). Legacy
  databases without the key: inferred from ``videos.model_version``
  (``"<checkpoint>|patches:N|tagger:...|phases:..."``), else ``DEFAULT_SIGLIP``.
  A database without any embedding is *unbound*: it takes the tier's model.
* ``scene_embeddings(scene_id, model, embedding)``: vectors per model.
  ``scenes.embedding`` keeps the ACTIVE model's vectors, so every reader
  works unchanged; the previous model's vectors are archived there, which
  makes switching back instant (no decode).
* ``character_example_embeddings(example_id, model, embedding)``: the same
  for Search+ character examples.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from collections import Counter
from typing import Any, Callable, Optional

import numpy as np

from ..config import DEFAULT_SIGLIP
from ..db import schema

log = logging.getLogger(__name__)

SETTINGS_KEY = "embedding_model"
PATCHES_KEY = "embedding_patches"

_TABLES = (
    """CREATE TABLE IF NOT EXISTS scene_embeddings (
        scene_id INTEGER NOT NULL REFERENCES scenes(id) ON DELETE CASCADE,
        model TEXT NOT NULL,
        embedding BLOB NOT NULL,
        PRIMARY KEY (scene_id, model)
    ) WITHOUT ROWID""",
    "CREATE INDEX IF NOT EXISTS idx_scene_embeddings_model ON scene_embeddings(model)",
    """CREATE TABLE IF NOT EXISTS character_example_embeddings (
        example_id INTEGER NOT NULL REFERENCES character_examples(id) ON DELETE CASCADE,
        model TEXT NOT NULL,
        embedding BLOB NOT NULL,
        PRIMARY KEY (example_id, model)
    ) WITHOUT ROWID""",
)


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Idempotent, additive. Needs the v9 tables (``character_examples``)."""
    for stmt in _TABLES:
        conn.execute(stmt)


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


# ── which model a database is bound to ──────────────────────────────────────
def recorded_model(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (SETTINGS_KEY,)).fetchone()
    return row[0] if row and row[0] else None


def infer_model(conn: sqlite3.Connection) -> str:
    """Legacy database: the checkpoint most of its embedded videos were
    indexed with (``videos.model_version``), else ``DEFAULT_SIGLIP``."""
    rows = conn.execute(
        """SELECT v.model_version FROM videos v
            WHERE v.model_version IS NOT NULL
              AND EXISTS (SELECT 1 FROM scenes s WHERE s.video_id = v.id AND s.embedding IS NOT NULL)"""
    ).fetchall()
    counts = Counter()
    for (mv,) in rows:
        ckpt = parse_model_version(mv)
        if ckpt:
            counts[ckpt] += 1
    if counts:
        return counts.most_common(1)[0][0]
    return DEFAULT_SIGLIP


def parse_model_version(model_version: Optional[str]) -> Optional[str]:
    if not model_version or "|patches:" not in model_version:
        return None
    ckpt = model_version.split("|", 1)[0].strip()
    return ckpt or None


def has_embeddings(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM scenes WHERE embedding IS NOT NULL LIMIT 1").fetchone() is not None


def bound_model(conn: sqlite3.Connection) -> Optional[str]:
    """The model the database's vectors belong to, None when it has none
    (empty / tags-only database: free to take the tier's model)."""
    if not has_embeddings(conn):
        return None
    return recorded_model(conn) or infer_model(conn)


def db_embedding_model(db_path: str, default: str) -> str:
    """Model to embed queries / new videos with for ``db_path``."""
    try:
        with schema.get_conn(db_path) as conn:
            return bound_model(conn) or default
    except sqlite3.Error as exc:
        log.debug("embedding model of %s unreadable: %s", db_path, exc)
        return default


def record_model(conn: sqlite3.Connection, model: str, patches: Optional[int] = None) -> None:
    conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (SETTINGS_KEY, model))
    if patches:
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (PATCHES_KEY, str(int(patches))))


def bind_model(db_path: str, model: str, patches: Optional[int] = None) -> str:
    """Called when indexing starts: an unbound database records the model
    about to fill it; a legacy one records its inferred model. Returns the
    database's model."""
    with schema.get_conn(db_path) as conn:
        bound = bound_model(conn)
        if bound is None:
            record_model(conn, model, patches)
            return model
        if recorded_model(conn) is None:
            record_model(conn, bound)
        return bound


def archived_models(conn: sqlite3.Connection, exclude: Optional[str] = None) -> list[str]:
    if not _has_table(conn, "scene_embeddings"):
        return []
    rows = conn.execute("SELECT DISTINCT model FROM scene_embeddings ORDER BY model").fetchall()
    return [r[0] for r in rows if r[0] != exclude]


def embedding_status(db_path: str, target: str) -> dict[str, Any]:
    with schema.get_conn(db_path) as conn:
        bound = bound_model(conn)
        db_model = bound or target
        return {
            "db_model": db_model,
            "target_model": target,
            "reindex_needed": bound is not None and bound != target,
            "archived_models": archived_models(conn, exclude=db_model),
        }


def swap_model_version(model_version: Optional[str], model: str, patches: Optional[int]) -> Optional[str]:
    """``"<old>|patches:N|rest"`` -> ``"<model>|patches:<patches>|rest"`` so
    the indexing auto-skip still recognises the re-embedded videos."""
    if not model_version or "|patches:" not in model_version:
        return model_version
    _old, rest = model_version.split("|patches:", 1)
    n, sep, tail = rest.partition("|")
    if patches:
        n = str(int(patches))
    return f"{model}|patches:{n}{sep}{tail}"


# ── the job ─────────────────────────────────────────────────────────────────
class ReembedBusy(RuntimeError):
    """Another re-embed job (or the indexing) is running."""


class ReembedCancelled(Exception):
    pass


_lock = threading.Lock()
_job: Optional["ReembedJob"] = None


class ReembedJob:
    def __init__(self, state, db_path: str, target: str, force: bool):
        self.state = state
        self.db_path = db_path
        self.target = target
        self.force = force
        self.from_model: Optional[str] = None
        self.running = True
        self.done = False
        self.ok = False
        self.percent = 0
        self.message = "Starting"
        self.error: Optional[str] = None
        self.videos_missing = 0
        self.missing_paths: list[str] = []
        self.switched = False
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.cancel_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self._last_pub = 0.0
        self._siglip = None

    # -- reporting --
    def to_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "done": self.done,
            "ok": self.ok,
            "percent": int(self.percent),
            "message": self.message,
            "error": self.error,
            "target_model": self.target,
            "from_model": self.from_model,
            "videos_missing": self.videos_missing,
            "missing_paths": self.missing_paths[:50],
            "switched": self.switched,
            "force": self.force,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    def _publish(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_pub < 0.25:
            return
        self._last_pub = now
        try:
            self.state.publish({
                "type": "reembed", "percent": int(self.percent), "message": self.message,
                "done": self.done, "ok": self.ok if self.done else self.error is None,
                "videos_missing": self.videos_missing,
                "target_model": self.target,
            })
        except Exception as exc:  # pragma: no cover - a dead socket never stops the job
            log.debug("reembed publish failed: %s", exc)

    def progress(self, percent: float, message: str, force: bool = False) -> None:
        self.percent = max(0, min(100, int(percent)))
        self.message = message
        self._publish(force)

    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise ReembedCancelled()

    def finish(self, ok: bool, message: str, error: Optional[str] = None) -> None:
        self.ok = ok
        self.error = error
        self.message = message
        self.done = True
        self.running = False
        self.finished_at = time.time()
        if ok:
            self.percent = 100
        self._publish(force=True)

    # -- model --
    def model(self):
        if self._siglip is None:
            self.progress(self.percent, f"Loading {self.target}", force=True)
            from .. import model_download
            token = (self.state.settings.get("models", {}).get("hf_token") or None)
            model_download.ensure_license_allowed(self.target, token=token)
            self._siglip = self.state.get_siglip(checkpoint=self.target)
        return self._siglip

    def batch_size(self, siglip) -> int:
        from .pipeline import _effective_batch
        configured = int(self.state.settings.get("indexing", {}).get("batch_size") or 8)
        return _effective_batch(siglip, max(1, configured))

    # -- body --
    def run(self) -> None:
        try:
            self._run()
        except ReembedCancelled:
            self.finish(False, "Cancelled — the previous model stays active", error="cancelled")
        except Exception as exc:
            log.exception("re-embed failed")
            self.finish(False, f"Failed: {exc}", error=str(exc))
        finally:
            try:
                if not self.switched and hasattr(self.state, "drop_aux_siglip"):
                    self.state.drop_aux_siglip()
            except Exception:
                pass

    def _run(self) -> None:
        from . import frames
        from .pipeline import _median_ms

        schema.init_db(self.db_path)
        with schema.get_conn(self.db_path) as conn:
            ensure_tables(conn)
            old = bound_model(conn)
            self.from_model = old
            if old is None:
                # Nothing embedded yet: the database simply takes the target.
                record_model(conn, self.target)
            elif old == self.target and recorded_model(conn) is None:
                record_model(conn, old)
            have = {r[0] for r in conn.execute(
                "SELECT scene_id FROM scene_embeddings WHERE model = ?", (self.target,))}
            rows = conn.execute(
                """SELECT s.id, s.video_id, s.start_ms, s.end_ms, v.filepath
                     FROM scenes s JOIN videos v ON v.id = s.video_id
                    WHERE s.embedding IS NOT NULL
                    ORDER BY s.video_id, s.start_ms"""
            ).fetchall()
        if old is None:
            self.switched = True
            self.state.invalidate_search_cache()
            self.finish(True, f"Library bound to {self.target} (nothing to re-embed)")
            return
        if old == self.target:
            self.finish(True, f"{self.target} is already this library's model")
            return

        by_video: dict[int, tuple[str, list[tuple[int, int, int]]]] = {}
        for sid, vid, s_ms, e_ms, fp in rows:
            if sid in have:
                continue
            by_video.setdefault(vid, (fp, []))[1].append((sid, s_ms, e_ms))
        missing = [vid for vid, (fp, _sc) in by_video.items() if not fp or not os.path.exists(fp)]
        self.videos_missing = len(missing)
        self.missing_paths = [by_video[v][0] for v in missing]
        if missing and not self.force:
            self.finish(False, f"{len(missing)} video(s) missing — relink them or remove them, "
                               f"or force the switch (their scenes leave search). "
                               f"{old} stays active.", error="videos_missing")
            return

        missing_set = set(missing)
        todo = [(vid, fp, sc) for vid, (fp, sc) in by_video.items() if vid not in missing_set]
        total = sum(len(sc) for _v, _fp, sc in todo)
        done = 0
        failed = 0
        if total:
            self.progress(0, f"Re-embedding {total} scenes with {self.target}", force=True)
        for vid, fp, scenes in todo:
            self.check_cancel()
            siglip = self.model()
            bs = self.batch_size(siglip)
            ts_to_sids: dict[int, list[int]] = {}
            for sid, s_ms, e_ms in scenes:
                ts_to_sids.setdefault(_median_ms(s_ms, e_ms), []).append(sid)
            written = 0
            with schema.get_conn(self.db_path) as conn:
                batch_ts: list[int] = []
                batch_imgs: list = []

                def flush() -> int:
                    if not batch_imgs:
                        return 0
                    embs = siglip.embed_images(batch_imgs)
                    out = []
                    for ts, emb in zip(batch_ts, embs):
                        blob = np.asarray(emb, dtype=np.float32).tobytes()
                        out.extend((sid, self.target, blob) for sid in ts_to_sids[ts])
                    conn.executemany(
                        "INSERT OR REPLACE INTO scene_embeddings (scene_id, model, embedding) VALUES (?, ?, ?)", out)
                    conn.commit()
                    n = len(out)
                    batch_ts.clear()
                    batch_imgs.clear()
                    return n

                try:
                    for ts, img in frames.extract_frames_at(fp, sorted(ts_to_sids)):
                        self.check_cancel()
                        batch_ts.append(ts)
                        batch_imgs.append(img)
                        if len(batch_imgs) >= bs:
                            written += flush()
                            self.progress(90 * (done + written) / total,
                                          f"Re-embedded {done + written}/{total} scenes")
                    written += flush()
                except ReembedCancelled:
                    raise
                except Exception as exc:
                    log.warning("re-embed of %s failed: %s", fp, exc)
            failed += len(scenes) - written
            done += len(scenes)
            self.progress(90 * done / total, f"Re-embedded {done}/{total} scenes")

        if failed and not self.force:
            self.finish(False, f"{failed} scene(s) could not be re-embedded — {old} stays active",
                        error="scenes_failed")
            return

        self.check_cancel()
        self.progress(92, "Re-embedding character examples", force=True)
        n_examples = self._embed_examples()
        self.check_cancel()
        self.progress(96, "Switching the library", force=True)
        self._switch(old)
        self.switched = True
        if hasattr(self.state, "promote_siglip"):
            self.state.promote_siglip(self.target)
        self.state.invalidate_search_cache()
        self._rescore_characters()
        dropped = f", {self.videos_missing} missing video(s) left out of search" if self.videos_missing else ""
        self.finish(True, f"Switched to {self.target} ({total} scenes re-embedded, "
                          f"{n_examples} character example(s){dropped})")

    def _embed_examples(self) -> int:
        """Target-model vectors for every character example that lacks one:
        plain scene examples take their scene's new vector (or its stored
        thumbnail when the scene could not be re-embedded), crops are
        re-cut from the scene's median frame, uploaded images re-embedded
        from their stored thumbnail."""
        import io

        from PIL import Image

        from ..searchplus import characters as chars

        with schema.get_conn(self.db_path) as conn:
            if not _has_table(conn, "character_examples"):
                return 0
            rows = conn.execute(
                """SELECT e.id, e.scene_id, e.crop, e.thumbnail,
                          (SELECT x.embedding FROM scene_embeddings x
                            WHERE x.scene_id = e.scene_id AND x.model = ?),
                          (SELECT s.thumbnail FROM scenes s WHERE s.id = e.scene_id)
                     FROM character_examples e
                    WHERE NOT EXISTS (SELECT 1 FROM character_example_embeddings y
                                       WHERE y.example_id = e.id AND y.model = ?)""",
                (self.target, self.target),
            ).fetchall()
            n = 0
            for ex_id, scene_id, crop, thumb, scene_vec, scene_thumb in rows:
                self.check_cancel()
                vec = None
                img = None
                try:
                    if scene_id is not None and not crop:
                        if scene_vec is not None:
                            vec = np.frombuffer(scene_vec, dtype=np.float32)
                        elif scene_thumb:
                            img = Image.open(io.BytesIO(scene_thumb)).convert("RGB")
                    elif scene_id is not None and crop:
                        import json
                        try:
                            img = chars.crop_image(chars.scene_frame(conn, scene_id), json.loads(crop))
                        except Exception:
                            img = Image.open(io.BytesIO(thumb)).convert("RGB") if thumb else None
                    elif thumb:
                        img = Image.open(io.BytesIO(thumb)).convert("RGB")
                    if vec is None and img is not None:
                        vec = self.model().embed_images([img])[0]
                except ReembedCancelled:
                    raise
                except Exception as exc:
                    log.warning("character example %s not re-embedded: %s", ex_id, exc)
                if vec is None:
                    continue
                conn.execute(
                    "INSERT OR REPLACE INTO character_example_embeddings (example_id, model, embedding) "
                    "VALUES (?, ?, ?)",
                    (ex_id, self.target, chars._norm(vec).astype(np.float32).tobytes()))
                n += 1
            return n

    def _switch(self, old: str) -> None:
        """One transaction: archive the active vectors under ``old``, make
        the target's vectors active, bind the database to the target."""
        patches = getattr(self._siglip, "max_num_patches", None) if self._siglip is not None else None
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT OR REPLACE INTO scene_embeddings (scene_id, model, embedding)
                   SELECT id, ?, embedding FROM scenes WHERE embedding IS NOT NULL""", (old,))
            has_examples = _has_table(conn, "character_examples")
            if has_examples:
                conn.execute(
                    """INSERT OR REPLACE INTO character_example_embeddings (example_id, model, embedding)
                       SELECT id, ?, embedding FROM character_examples""", (old,))
            conn.execute(
                """UPDATE scenes SET embedding =
                        (SELECT x.embedding FROM scene_embeddings x WHERE x.scene_id = scenes.id AND x.model = ?)
                    WHERE embedding IS NOT NULL
                       OR id IN (SELECT scene_id FROM scene_embeddings WHERE model = ?)""",
                (self.target, self.target))
            if has_examples:
                conn.execute(
                    """UPDATE character_examples SET embedding =
                            (SELECT y.embedding FROM character_example_embeddings y
                              WHERE y.example_id = character_examples.id AND y.model = ?)
                        WHERE id IN (SELECT example_id FROM character_example_embeddings WHERE model = ?)""",
                    (self.target, self.target))
            record_model(conn, self.target, patches)
            for vid, mv in conn.execute("SELECT id, model_version FROM videos").fetchall():
                new = swap_model_version(mv, self.target, patches)
                if new != mv:
                    conn.execute("UPDATE videos SET model_version = ? WHERE id = ?", (new, vid))
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()

    def _rescore_characters(self) -> None:
        from ..searchplus import characters as chars
        try:
            index = self.state.get_search_index()
            with schema.get_conn(self.db_path) as conn:
                if not _has_table(conn, "characters"):
                    return
                for (cid,) in conn.execute("SELECT id FROM characters").fetchall():
                    try:
                        chars.rescore(conn, cid, index)
                    except Exception as exc:
                        log.warning("character %s rescore failed after re-embed: %s", cid, exc)
        except Exception as exc:
            log.warning("character rescore after re-embed failed: %s", exc)


# ── manager ─────────────────────────────────────────────────────────────────
def current() -> Optional[ReembedJob]:
    return _job


def is_running() -> bool:
    j = _job
    return bool(j is not None and j.running)


def status() -> Optional[dict[str, Any]]:
    j = _job
    return j.to_dict() if j is not None else None


def start(state, target: Optional[str] = None, *, force: bool = False,
          indexing_running: Optional[Callable[[], bool]] = None, background: bool = True) -> ReembedJob:
    """Start re-embedding the primary database with ``target`` (default:
    the tier's embedding model). Raises :class:`ReembedBusy`."""
    global _job
    if target is None:
        target = state.embedding_target()[0]
    with _lock:
        if _job is not None and _job.running:
            raise ReembedBusy("a re-embed job is already running")
        if indexing_running is not None and indexing_running():
            raise ReembedBusy("indexing is running — re-embed after it finishes")
        job = ReembedJob(state, state.primary_db, target, force)
        _job = job
    job.progress(0, "Starting", force=True)
    if background:
        job.thread = threading.Thread(target=job.run, daemon=True, name="reembed")
        job.thread.start()
    else:
        job.run()
    return job


def cancel() -> bool:
    j = _job
    if j is None or not j.running:
        return False
    j.cancel_event.set()
    return True


def reset() -> None:
    """Forget the last job (tests)."""
    global _job
    with _lock:
        _job = None


def merge_conflict(target_db: str, sources: list[str]) -> Optional[str]:
    """A merge would mix embedding spaces: message, else None."""
    def model_of(path: str) -> Optional[str]:
        if not os.path.exists(path):
            return None
        try:
            with schema.get_conn(path) as conn:
                return bound_model(conn)
        except sqlite3.Error:
            return None

    target = model_of(target_db)
    for src in sources:
        m = model_of(src)
        if target is not None and m is not None and m != target:
            return (f"{os.path.basename(src)} is embedded with {m} but the target library uses {target}: "
                    f"re-embed one of them (Settings > Models) before merging")
        target = target or m
    return None
