"""Background analyzer: fills scene_metrics for scenes indexed before v0.4
(or whose range changed), one video at a time, without re-indexing.

Progress goes through the regular progress websocket as
``{"type": "analysis", ...}`` events. The job yields to indexing (it waits
between videos while an index run is active) and can be cancelled.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Optional

from ..db import schema
from . import analysis, store

log = logging.getLogger(__name__)


class AnalysisJob:
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self.status: dict[str, Any] = {
            "running": False, "done": 0, "total": 0, "percent": 0,
            "video": None, "skipped_videos": 0, "error": None, "finished_at": None,
        }

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, state) -> bool:
        with self._lock:
            if self.running:
                return False
            self._cancel.clear()
            self._thread = threading.Thread(target=self._run, args=(state,), daemon=True,
                                            name="searchplus-analysis")
            self.status.update(running=True, done=0, total=0, percent=0, video=None,
                               skipped_videos=0, error=None)
            self._thread.start()
            return True

    def stop(self) -> None:
        self._cancel.set()

    def _indexing(self, state) -> bool:
        t = getattr(state, "_index_thread", None)
        return t is not None and t.is_alive()

    def _publish(self, state, **extra) -> None:
        evt = {"type": "analysis", "percent": self.status["percent"],
               "current": self.status["done"], "total": self.status["total"],
               "video": self.status["video"]}
        evt.update(extra)
        try:
            state.publish(evt)
        except Exception:  # pragma: no cover - no loop in tests
            pass

    def _run(self, state) -> None:
        db_path = state.primary_db
        threshold = float(state.settings.get("indexing", {}).get("scene_detect_threshold", 0.40))
        try:
            schema.init_db(db_path)
            with schema.get_conn(db_path) as conn:
                todo = store.pending_videos(conn)
            self.status["total"] = sum(n for *_, n in todo)
            last_pub = 0.0
            for video_id, filepath, display, n_pending in todo:
                while self._indexing(state) and not self._cancel.is_set():
                    self._cancel.wait(2.0)
                if self._cancel.is_set():
                    break
                self.status["video"] = display
                if not filepath or not os.path.exists(filepath):
                    self.status["skipped_videos"] += 1
                    self.status["done"] += n_pending
                    continue
                with schema.get_conn(db_path) as conn:
                    scenes = store.pending_scenes_for_video(conn, video_id)
                if not scenes:
                    continue
                self._publish(state, message="decoding")
                try:
                    acc = analysis.collect_frame_stats(filepath, threshold=threshold)
                except Exception as exc:
                    log.warning("analysis decode failed for %s: %s", filepath, exc)
                    self.status["skipped_videos"] += 1
                    self.status["done"] += len(scenes)
                    continue
                try:
                    rows = [
                        (sid, s_ms, e_ms,
                         analysis.aggregate_scene(acc, s_ms, e_ms, analysis.rgb_from_jpeg(thumb)))
                        for sid, s_ms, e_ms, thumb in scenes
                    ]
                    with schema.get_conn(db_path) as conn:
                        store.upsert_metrics(conn, rows)
                except Exception as exc:
                    # e.g. the video was re-indexed meanwhile (its scenes are
                    # gone: FK error): skip it, keep analysing the others
                    log.warning("analysis failed for %s: %s", filepath, exc)
                    self.status["skipped_videos"] += 1
                self.status["done"] += len(scenes)
                total = max(self.status["total"], 1)
                self.status["percent"] = min(99, int(self.status["done"] * 100 / total))
                now = time.monotonic()
                if now - last_pub > 0.5:
                    self._publish(state)
                    last_pub = now
        except Exception as exc:
            log.exception("search+ analysis crashed")
            self.status["error"] = str(exc)
        finally:
            self.status.update(running=False, percent=100, video=None, finished_at=time.time())
            self._publish(state, done=True, ok=self.status["error"] is None)
            if not self._indexing(state):
                try:
                    state.publish({"type": "idle"})
                except Exception:  # pragma: no cover
                    pass


_job: Optional[AnalysisJob] = None


def get_job() -> AnalysisJob:
    global _job
    if _job is None:
        _job = AnalysisJob()
    return _job
