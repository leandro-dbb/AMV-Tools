"""Export endpoint."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..db import queries, schema
from ..export import export_scene
from ..paths import user_data_dir
from ..state import get_state

router = APIRouter()


class ExportRequest(BaseModel):
    scene_id: int
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    output_path: Optional[str] = None


class BatchExportRequest(BaseModel):
    scene_ids: list[int]


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "scene"


def _suffix_for_codec(codec: str) -> str:
    if codec in ("libx264", "libx265", "libsvtav1", "h264_nvenc", "h264_videotoolbox"):
        return ".mp4"
    if codec in ("prores_ks", "dnxhr"):
        return ".mov"
    return ".webm"


def _export_one(scene: dict, settings: dict, output_dir: Path,
                start_ms: int | None = None, end_ms: int | None = None,
                output_path: str | None = None) -> str:
    name = settings["naming_template"].format(
        anime=_sanitize(scene["display_name"]),
        episode=_sanitize(scene["display_name"]),
        scene_id=scene["id"],
        tags="",
    )
    suffix = _suffix_for_codec(settings["codec"])
    target = output_path or str(output_dir / f"{_sanitize(name)}{suffix}")
    return export_scene(
        scene["filepath"],
        start_ms if start_ms is not None else scene["start_ms"],
        end_ms if end_ms is not None else scene["end_ms"],
        target,
        codec=settings["codec"], crf=settings["crf"],
        audio=settings["audio"], resolution=settings["resolution"],
        audio_bitrate_kbps=int(settings.get("audio_bitrate_kbps", 320)),
    )


@router.post("/api/export")
def export(req: ExportRequest):
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        scene = queries.get_scene(conn, req.scene_id)
    if not scene:
        raise HTTPException(404, "scene not found")
    if not Path(scene["filepath"]).exists():
        raise HTTPException(410, "source file missing")

    start_ms = req.start_ms if req.start_ms is not None else scene["start_ms"]
    end_ms = req.end_ms if req.end_ms is not None else scene["end_ms"]

    settings = state.settings["export"]
    output_dir = Path(settings["output_folder"]) if settings["output_folder"] else (user_data_dir() / "exports")
    output_dir.mkdir(parents=True, exist_ok=True)

    out = _export_one(scene, settings, output_dir, req.start_ms, req.end_ms, req.output_path)
    return {"ok": True, "output": out}


@router.post("/api/export/batch")
def export_batch(req: BatchExportRequest):
    state = get_state()
    settings = state.settings["export"]
    output_dir = Path(settings["output_folder"]) if settings["output_folder"] else (user_data_dir() / "exports")
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: list[dict] = []
    failed: list[dict] = []
    with schema.get_conn(state.primary_db) as conn:
        scenes = []
        for sid in req.scene_ids:
            s = queries.get_scene(conn, sid)
            if s:
                scenes.append(s)
    for s in scenes:
        try:
            out = _export_one(s, settings, output_dir)
            exported.append({"scene_id": s["id"], "output": out})
        except Exception as e:
            failed.append({"scene_id": s["id"], "error": str(e)})
    return {"exported": exported, "failed": failed}
