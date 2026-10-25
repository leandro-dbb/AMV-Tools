"""``autotune.json`` in the user data folder: results per machine key."""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

KEEP = 4          # results kept for other fingerprints (driver rollback, dual boot)
_lock = threading.Lock()


def path() -> Path:
    from ..paths import user_data_dir
    return user_data_dir() / "autotune.json"


def _read(p: Path) -> dict:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and isinstance(data.get("results"), dict) else {"results": {}}
    except FileNotFoundError:
        return {"results": {}}
    except Exception as exc:
        log.warning("autotune cache unreadable (%s); measuring again", exc)
        return {"results": {}}


def load(key: str, p: Optional[Path] = None) -> Optional[dict[str, Any]]:
    with _lock:
        entry = _read(p or path())["results"].get(key)
    return entry if isinstance(entry, dict) and isinstance(entry.get("settings"), dict) else None


def store(key: str, entry: dict[str, Any], p: Optional[Path] = None) -> None:
    p = p or path()
    with _lock:
        data = _read(p)
        results = data["results"]
        results.pop(key, None)
        results[key] = entry
        for old in list(results)[:-KEEP]:
            results.pop(old, None)
        tmp = p.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=False), encoding="utf-8")
        os.replace(tmp, p)


def drop(key: str, p: Optional[Path] = None) -> None:
    p = p or path()
    with _lock:
        data = _read(p)
        if data["results"].pop(key, None) is not None:
            p.write_text(json.dumps(data, indent=1), encoding="utf-8")
