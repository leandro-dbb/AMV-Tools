"""Analysis cache: one JSON per audio content hash in the user data folder.

Files, not SQLite: an analysis belongs to a song, not to a library, so it
survives switching databases and needs no schema migration.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional

from .analysis import ANALYSIS_VERSION, analyze_audio
from .audio import check_source, fingerprint_file

HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def valid_hash(value: str) -> bool:
    return bool(HASH_RE.match(value or ""))


class AnalysisCache:
    def __init__(self, folder: Path):
        self.folder = Path(folder)

    def _file(self, file_hash: str) -> Path:
        if not valid_hash(file_hash):
            raise ValueError("invalid track id")
        return self.folder / f"{file_hash}.json"

    def get(self, file_hash: str) -> Optional[dict]:
        p = self._file(file_hash)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if data.get("analysis_version") != ANALYSIS_VERSION:
            return None
        return data

    def put(self, data: dict) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        p = self._file(data["hash"])
        tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
            os.replace(tmp, p)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def analyze(self, path: str) -> dict:
        """Cached analysis of a file; re-analysed when the content changed."""
        src = check_source(path)
        file_hash = fingerprint_file(src)
        cached = self.get(file_hash)
        if cached is not None:
            if cached.get("path") != str(src) and not Path(cached.get("path") or "").is_file():
                # Same song moved/renamed: remember where it lives now.
                cached["path"] = str(src)
                cached["name"] = src.name
                self.put(cached)
            return {**cached, "cached": True}
        data = analyze_audio(str(src), file_hash=file_hash)
        data["path"] = str(src)
        data["name"] = src.name
        data["analyzed_at"] = time.time()
        self.put(data)
        return {**data, "cached": False}

    def tracks(self) -> list[dict]:
        out = []
        if not self.folder.exists():
            return out
        for p in self.folder.glob("*.json"):
            if not valid_hash(p.stem):
                continue
            data = self.get(p.stem)
            if not data:
                continue
            out.append({
                "hash": data["hash"], "name": data.get("name") or Path(data.get("path", "")).name,
                "path": data.get("path", ""), "duration_ms": data.get("duration_ms", 0),
                "bpm": data.get("bpm", 0), "analyzed_at": data.get("analyzed_at", 0),
            })
        out.sort(key=lambda t: -float(t["analyzed_at"] or 0))
        return out
