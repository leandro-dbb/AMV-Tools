"""Timeline export endpoints ("send to your editor").

The heavy lifting lives in :mod:`backend.timeline`; this module only resolves
scene/derush selectors to clip specs, applies trim overrides and validates the
request.  Files are written, never transcoded.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..db import queries, schema
from ..paths import user_data_dir
from ..state import get_state
from ..timeline import (
    ClipSpec,
    TimelineOptions,
    export_timeline,
    formats,
    preview_timeline,
)
from ..timeline.service import FORMATS, SourceMissingError

router = APIRouter()


# ── request models ──────────────────────────────────────────────────────────
class TrimRange(BaseModel):
    start_ms: int
    end_ms: int


class TimelineOptionsBody(BaseModel):
    sequence_fps: Optional[str] = "first"
    resolution: Optional[str] = "first"
    layout: Literal["sequential", "gaps"] = "sequential"
    gap_frames: int = Field(default=0, ge=0)
    handles_frames: int = Field(default=0, ge=0)
    drop_frame: bool = False
    markers: bool = True
    include_audio: bool = True
    collect_media: bool = False
    path_mode: Literal["absolute", "relative"] = "absolute"
    title: Optional[str] = None
    otio_use_library: bool = True
    # Optional: some API clients put the destination here instead of at the
    # top level; both are accepted.
    output_path: Optional[str] = None
    # Optional mini-editor / manual trim overrides, keyed by scene id or
    # derush item id (both as strings in JSON).
    trims: Optional[dict[str, TrimRange]] = None

    def to_model(self) -> TimelineOptions:
        return TimelineOptions(
            sequence_fps=self.sequence_fps or "first",
            resolution=self.resolution or "first",
            layout=self.layout,
            gap_frames=self.gap_frames,
            handles_frames=self.handles_frames,
            drop_frame=self.drop_frame,
            markers=self.markers,
            include_audio=self.include_audio,
            collect_media=self.collect_media,
            path_mode=self.path_mode,
            title=self.title or "AMV Tools timeline",
            otio_use_library=self.otio_use_library,
        )


class TimelineSelection(BaseModel):
    scene_ids: Optional[list[int]] = None
    derush_item_ids: Optional[list[int]] = None
    derush_folder_id: Optional[int] = None
    derush_scope: Optional[Literal["all", "folder", "root", "favorites"]] = None
    options: TimelineOptionsBody = Field(default_factory=TimelineOptionsBody)


class TimelineExportBody(TimelineSelection):
    format: Literal["fcp7xml", "fcpxml", "edl", "otio", "vegas", "aftereffects"]
    output_path: Optional[str] = None


# ── selector resolution ─────────────────────────────────────────────────────
_TAGS_SQL = "SELECT scene_id, tag FROM scene_tags WHERE scene_id IN ({}) ORDER BY confidence DESC"


def _tags_by_scene(conn: sqlite3.Connection, scene_ids: list[int]) -> dict[int, list[str]]:
    if not scene_ids:
        return {}
    placeholders = ",".join("?" * len(scene_ids))
    rows = conn.execute(_TAGS_SQL.format(placeholders), scene_ids).fetchall()
    out: dict[int, list[str]] = {}
    for sid, tag in rows:
        out.setdefault(sid, []).append(tag)
    return out


def _trim_for(options: TimelineOptionsBody, spec: ClipSpec) -> tuple[int, int]:
    trims = options.trims or {}
    for key in (str(spec.item_id) if spec.item_id is not None else None,
                str(spec.scene_id) if spec.scene_id is not None else None):
        if key is not None and key in trims:
            return trims[key].start_ms, trims[key].end_ms
    return spec.start_ms or 0, spec.end_ms or 0


def _scene_specs(conn: sqlite3.Connection, scene_ids: list[int],
                 options: TimelineOptionsBody) -> list[ClipSpec]:
    if not scene_ids:
        return []
    placeholders = ",".join("?" * len(scene_ids))
    rows = conn.execute(
        f"""SELECT s.id, s.video_id, v.filepath, v.display_name, s.scene_index,
                   s.start_ms, s.end_ms, i.id AS item_id, i.custom_name,
                   COALESCE(i.favorite, 0),
                   (SELECT f.name FROM derush_item_folders l
                      JOIN derush_folders f ON f.id = l.folder_id
                     WHERE l.item_id = i.id ORDER BY f.name LIMIT 1)
              FROM scenes s
              JOIN videos v ON v.id = s.video_id
              LEFT JOIN derush_items i ON i.scene_id = s.id
             WHERE s.id IN ({placeholders})""",
        scene_ids,
    ).fetchall()
    by_id = {r[0]: r for r in rows}
    tags = _tags_by_scene(conn, scene_ids)
    specs: list[ClipSpec] = []
    for sid in scene_ids:
        row = by_id.get(sid)
        if row is None:
            continue
        _, _, filepath, display_name, scene_index, start_ms, end_ms, item_id, custom_name, favorite, folder = row
        name = custom_name or f"{display_name}_scene{scene_index:04d}"
        spec = ClipSpec(
            source_path=filepath,
            name=name,
            start_ms=start_ms,
            end_ms=end_ms,
            label="Mango" if favorite else "",
            folder=folder,
            tags=tags.get(sid, []),
            scene_id=sid,
            item_id=item_id,
        )
        spec.start_ms, spec.end_ms = _trim_for(options, spec)
        specs.append(spec)
    return specs


def _derush_rows(conn: sqlite3.Connection, item_ids: list[int],
                 selector) -> list[tuple]:
    if item_ids:
        placeholders = ",".join("?" * len(item_ids))
        rows = conn.execute(
            f"""SELECT i.id, i.scene_id, i.custom_name, s.start_ms, s.end_ms,
                       v.filepath, v.display_name, s.scene_index, i.favorite,
                       (SELECT f.name FROM derush_item_folders l
                          JOIN derush_folders f ON f.id = l.folder_id
                         WHERE l.item_id = i.id ORDER BY f.name LIMIT 1)
                  FROM derush_items i
                  JOIN scenes s ON s.id = i.scene_id
                  JOIN videos v ON v.id = s.video_id
                 WHERE i.id IN ({placeholders})""",
            item_ids,
        ).fetchall()
        by_id = {r[0]: r for r in rows}
        return [by_id[i] for i in item_ids if i in by_id]

    rows = conn.execute(
        """SELECT i.id, i.scene_id, i.custom_name, s.start_ms, s.end_ms,
                  v.filepath, v.display_name, s.scene_index, i.favorite,
                  (SELECT f.name FROM derush_item_folders l
                     JOIN derush_folders f ON f.id = l.folder_id
                    WHERE l.item_id = i.id ORDER BY f.name LIMIT 1)
             FROM derush_items i
             JOIN scenes s ON s.id = i.scene_id
             JOIN videos v ON v.id = s.video_id
            ORDER BY v.display_name, s.start_ms, i.id"""
    ).fetchall()
    if selector == "all":
        return list(rows)
    if selector == "fav":
        return [r for r in rows if r[8]]
    if selector is None:
        # root = items that belong to no folder
        connected = {
            r[0] for r in conn.execute("SELECT DISTINCT item_id FROM derush_item_folders").fetchall()
        }
        return [r for r in rows if r[0] not in connected]
    folder_ids = {
        r[0] for r in conn.execute(
            "SELECT item_id FROM derush_item_folders WHERE folder_id = ?", (selector,)
        ).fetchall()
    }
    return [r for r in rows if r[0] in folder_ids]


def _derush_specs(conn: sqlite3.Connection, item_ids: list[int], selector,
                  options: TimelineOptionsBody) -> list[ClipSpec]:
    rows = _derush_rows(conn, item_ids, selector)
    tags = _tags_by_scene(conn, [r[1] for r in rows])
    specs: list[ClipSpec] = []
    for r in rows:
        (item_id, scene_id, custom_name, start_ms, end_ms, filepath,
         display_name, scene_index, favorite, folder) = r
        name = custom_name or f"{display_name}_scene{scene_index:04d}"
        spec = ClipSpec(
            source_path=filepath,
            name=name,
            start_ms=start_ms,
            end_ms=end_ms,
            label="Mango" if favorite else "",
            folder=folder,
            tags=tags.get(scene_id, []),
            scene_id=scene_id,
            item_id=item_id,
        )
        spec.start_ms, spec.end_ms = _trim_for(options, spec)
        specs.append(spec)
    return specs


def _resolve_specs(conn: sqlite3.Connection, body: TimelineSelection) -> list[ClipSpec]:
    provided = [
        bool(body.scene_ids),
        bool(body.derush_item_ids),
        body.derush_folder_id is not None,
        body.derush_scope is not None,
    ]
    if sum(provided) > 1:
        raise HTTPException(400, "provide only one selection: scene_ids, derush_item_ids, derush_folder_id or derush_scope")
    if not any(provided):
        raise HTTPException(400, "no selection: provide scene_ids, derush_item_ids, derush_folder_id or derush_scope")

    if body.scene_ids:
        specs = _scene_specs(conn, body.scene_ids, body.options)
    elif body.derush_item_ids:
        specs = _derush_specs(conn, body.derush_item_ids, "all", body.options)
    elif body.derush_folder_id is not None:
        specs = _derush_specs(conn, [], body.derush_folder_id, body.options)
    else:
        scope = body.derush_scope or "all"
        if scope == "folder":
            raise HTTPException(400, "scope=folder requires derush_folder_id")
        selector = "fav" if scope == "favorites" else (None if scope == "root" else "all")
        specs = _derush_specs(conn, [], selector, body.options)
    if not specs:
        raise HTTPException(400, "nothing to export in this selection")
    return specs


def _run(body: TimelineSelection, *, fmt: str | None = None,
         output_path: str | None = None) -> dict:
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        specs = _resolve_specs(conn, body)
    options = body.options.to_model()
    try:
        if fmt is None:
            return preview_timeline(specs, options)
        return export_timeline(specs, fmt, output_path or "", options)
    except SourceMissingError as exc:
        raise HTTPException(410, {"error": "source_missing", "paths": exc.paths})
    except (ValueError, ZeroDivisionError) as exc:   # e.g. sequence_fps "24/0"
        raise HTTPException(400, str(exc) or "invalid frame rate")
    except PermissionError as exc:
        raise HTTPException(403, f"cannot write the timeline: {exc}")
    except OSError as exc:
        raise HTTPException(500, f"cannot write the timeline: {exc}")
    except RuntimeError as exc:   # ffprobe failed / timed out on a source
        raise HTTPException(422, f"cannot read a source video: {exc}")


# ── routes ──────────────────────────────────────────────────────────────────
@router.get("/api/timeline/formats")
def timeline_formats():
    return {"formats": formats()}


@router.post("/api/timeline/preview")
def timeline_preview(body: TimelineSelection):
    return _run(body)


@router.post("/api/timeline/export")
def timeline_export(body: TimelineExportBody):
    requested = body.output_path or body.options.output_path
    if requested:
        output = Path(requested)
        if not output.is_absolute():
            raise HTTPException(400, "output_path must be absolute")
    else:
        # Browser/API fallback when no Electron save dialog was used.
        folder = user_data_dir() / "timelines"
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / ("amv-timeline" + FORMATS[body.format]["extension"])
    if not output.parent.exists():
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(400, f"cannot create output folder: {exc}")
    return _run(body, fmt=body.format, output_path=str(output))
