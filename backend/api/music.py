"""Music sync endpoints: analyse a song, propose a beat-synced edit from the
indexed shots, export it (timeline / markers only) and render a preview."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..db import schema
from ..music import export as music_export
from ..music import library, matcher
from ..music.audio import AudioDecodeError
from ..music.cache import AnalysisCache, valid_hash
from ..music.preview import PreviewManager
from ..paths import user_data_dir
from ..state import get_state

log = logging.getLogger(__name__)
router = APIRouter()

_AUDIO_MIME = {
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac", ".m4a": "audio/mp4",
    ".aac": "audio/aac", ".ogg": "audio/ogg", ".oga": "audio/ogg", ".opus": "audio/ogg",
    ".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
}

_cache: Optional[AnalysisCache] = None
_warm_lock = threading.Lock()
_warm_started = False
_previews: Optional[PreviewManager] = None


def _music_dir() -> Path:
    return user_data_dir() / "music"


def get_cache() -> AnalysisCache:
    global _cache
    folder = _music_dir() / "analyses"
    if _cache is None or _cache.folder != folder:
        _cache = AnalysisCache(folder)
    return _cache


def get_previews() -> PreviewManager:
    global _previews
    folder = _music_dir() / "previews"
    if _previews is None or _previews.folder != folder:
        _previews = PreviewManager(folder)
    return _previews


def _analysis_or_404(file_hash: str) -> dict:
    if not valid_hash(file_hash):
        raise HTTPException(400, "invalid track id")
    data = get_cache().get(file_hash)
    if data is None:
        raise HTTPException(404, "track not analysed (import it again)")
    return data


def _public(data: dict) -> dict:
    keys = ("hash", "name", "path", "duration_ms", "bpm", "variable_tempo", "meter", "beats_ms",
            "downbeats_ms", "strong_onsets_ms", "drops_ms", "sections", "energy", "waveform",
            "analysis_seconds", "cached")
    out = {k: data.get(k) for k in keys}
    out["cached"] = bool(data.get("cached"))
    return out


# ── analysis ────────────────────────────────────────────────────────────────
class AnalyzeBody(BaseModel):
    path: str


@router.post("/api/music/analyze")
def analyze(body: AnalyzeBody):
    try:
        data = get_cache().analyze(body.path)
    except FileNotFoundError:
        raise HTTPException(404, "file not found")
    except AudioDecodeError as exc:
        raise HTTPException(422, str(exc))
    return _public(data)


def _warm_up() -> None:
    """librosa's numba kernels compile on first use (10-20 s on a slow CPU):
    do it while the user is still picking a file."""
    global _warm_started
    with _warm_lock:
        if _warm_started:
            return
        _warm_started = True

    def run():
        try:
            import numpy as np

            from ..music.analysis import analyze_signal

            analyze_signal(np.random.default_rng(0).standard_normal(22050 * 6).astype("float32") * 0.1)
        except Exception:  # best effort only
            log.debug("music warm-up failed", exc_info=True)

    threading.Thread(target=run, name="music-warmup", daemon=True).start()


@router.get("/api/music/tracks")
def tracks():
    # The Music tab lists tracks when it opens: warm the analyser up then.
    _warm_up()
    return {"tracks": get_cache().tracks()}


@router.get("/api/music/tracks/{file_hash}")
def track(file_hash: str):
    return _public({**_analysis_or_404(file_hash), "cached": True})


@router.get("/api/music/tracks/{file_hash}/audio")
def track_audio(file_hash: str):
    # Only ever serves the file recorded for an analysed hash: no path input.
    data = _analysis_or_404(file_hash)
    path = Path(data.get("path") or "")
    if not path.is_file():
        raise HTTPException(404, "audio file moved or deleted")
    return FileResponse(str(path), media_type=_AUDIO_MIME.get(path.suffix.lower(), "application/octet-stream"))


# ── proposal ────────────────────────────────────────────────────────────────
class ConstraintsBody(BaseModel):
    groups: list[str] = Field(default_factory=list)
    characters: list[int] = Field(default_factory=list)
    density: float = Field(default=0.5, ge=0.0, le=1.0)
    avoid_reuse: bool = True
    query: str = ""
    section_queries: dict[str, str] = Field(default_factory=dict)


class ClipBody(BaseModel):
    model_config = {"extra": "allow"}

    slot: int = 0
    section_index: int
    start_ms: int
    end_ms: int
    beats: int = 1
    scene_id: int
    video_id: int = 0
    video_name: str = ""
    group: Optional[str] = None
    src_in_ms: int
    src_out_ms: int
    intensity: Optional[float] = None
    intensity_rank: Optional[float] = None
    motion_type: Optional[str] = None
    target: float = 0.5
    score: float = 0.0
    locked: bool = False
    alternatives: list[dict] = Field(default_factory=list)


class ProposeBody(BaseModel):
    hash: str
    constraints: ConstraintsBody = Field(default_factory=ConstraintsBody)
    clips: Optional[list[ClipBody]] = None
    regenerate_sections: Optional[list[int]] = None
    seed: Optional[int] = None


def _query_scores(conn, c: ConstraintsBody, warnings: list[str]) -> dict:
    wishes: dict = {}
    if c.query.strip():
        wishes["global"] = c.query
    for key, text in (c.section_queries or {}).items():
        if text and text.strip():
            try:
                wishes[int(key)] = text
            except ValueError:
                continue
    if not wishes:
        return {}
    state = get_state()
    embed = None
    index = None
    try:
        index = state.get_search_index()
        if len(index.get("sids", [])):
            embed = state.get_siglip().embed_text
    except Exception as exc:  # model missing / not installed: tags still work
        log.warning("music: text wishes without SigLIP (%s)", exc)
        warnings.append("query_tags_only")
    return {k: library.query_scores(conn, index, text, embed) for k, text in wishes.items()}


@router.post("/api/music/propose")
def propose(body: ProposeBody):
    analysis = _analysis_or_404(body.hash)
    state = get_state()
    schema.init_db(state.primary_db)
    warnings: list[str] = []
    with schema.get_conn(state.primary_db) as conn:
        shots = library.load_shots(conn)
        chars = library.character_scene_ids(conn, body.constraints.characters)
        scores = _query_scores(conn, body.constraints, warnings)
    c = body.constraints
    result = matcher.propose(
        analysis, shots,
        matcher.Constraints(groups=c.groups, characters=c.characters, density=c.density,
                            avoid_reuse=c.avoid_reuse, query=c.query,
                            section_queries={int(k): v for k, v in c.section_queries.items()
                                             if str(k).lstrip("-").isdigit()}),
        query_scores=scores, character_scene_ids=chars,
        previous=[cl.model_dump() for cl in body.clips] if body.clips else None,
        regenerate_sections=body.regenerate_sections, seed=body.seed,
    )
    with_metrics = sum(1 for s in shots if s.get("intensity") is not None)
    if shots and with_metrics < len(shots):
        warnings.append(f"metrics_missing:{len(shots) - with_metrics}")
    return {
        "hash": body.hash,
        "bpm": analysis.get("bpm"),
        "duration_ms": analysis.get("duration_ms"),
        "clips": result["clips"],
        "warnings": warnings + result["warnings"],
        "library": {"scenes": len(shots), "with_metrics": with_metrics},
    }


# ── exports ─────────────────────────────────────────────────────────────────
def _scene_paths(scene_ids: list[int]) -> dict[int, str]:
    if not scene_ids:
        return {}
    state = get_state()
    ids = sorted(set(int(s) for s in scene_ids))
    out: dict[int, str] = {}
    with schema.get_conn(state.primary_db) as conn:
        for i in range(0, len(ids), 900):
            chunk = ids[i:i + 900]
            rows = conn.execute(
                f"SELECT s.id, v.filepath FROM scenes s JOIN videos v ON v.id = s.video_id "
                f"WHERE s.id IN ({','.join('?' * len(chunk))})", chunk).fetchall()
            out.update({int(r[0]): r[1] for r in rows})
    return out


class ExportTimelineBody(BaseModel):
    hash: str
    clips: list[ClipBody]
    format: Literal["fcp7xml", "fcpxml", "edl", "otio"] = "fcp7xml"
    output_path: str
    beat_markers: bool = True
    include_source_audio: bool = False
    lang: Literal["en", "fr"] = "en"


@router.post("/api/music/export/timeline")
def export_timeline(body: ExportTimelineBody):
    analysis = _analysis_or_404(body.hash)
    if not body.clips:
        raise HTTPException(400, "the edit is empty")
    clips = [c.model_dump() for c in body.clips]
    paths = _scene_paths([c["scene_id"] for c in clips])
    try:
        result = music_export.export_edit(
            body.format, body.output_path, analysis, analysis["path"], clips, paths,
            beat_markers=body.beat_markers, include_source_audio=body.include_source_audio,
            lang=body.lang)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except OSError as exc:
        raise HTTPException(500, f"could not write the timeline: {exc}")
    result.pop("record_frames", None)
    return result


class ExportMarkersBody(BaseModel):
    hash: str
    format: Literal["fcp7xml", "fcpxml", "edl", "otio", "csv"] = "fcp7xml"
    output_path: str
    beat_markers: bool = True
    fps: Optional[str] = None
    lang: Literal["en", "fr"] = "en"


@router.post("/api/music/export/markers")
def export_markers(body: ExportMarkersBody):
    analysis = _analysis_or_404(body.hash)
    try:
        return music_export.export_markers(body.format, body.output_path, analysis, analysis["path"],
                                           fps=body.fps, beat_markers=body.beat_markers, lang=body.lang)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except OSError as exc:
        raise HTTPException(500, f"could not write the markers: {exc}")


# ── preview ─────────────────────────────────────────────────────────────────
class PreviewBody(BaseModel):
    hash: str
    clips: list[ClipBody]


@router.post("/api/music/preview")
def start_preview(body: PreviewBody):
    analysis = _analysis_or_404(body.hash)
    if not body.clips:
        raise HTTPException(400, "the edit is empty")
    music_path = analysis.get("path") or ""
    if not Path(music_path).is_file():
        raise HTTPException(404, "audio file moved or deleted")
    clips = [c.model_dump() for c in body.clips]
    paths = _scene_paths([c["scene_id"] for c in clips])
    missing = sorted({p for p in paths.values() if not Path(p).is_file()})
    if missing:
        raise HTTPException(404, "source file(s) missing: " + ", ".join(missing))
    return get_previews().start(clips, paths, music_path).as_dict()


def _job_or_404(job_id: str):
    job = get_previews().get(job_id)
    if job is None:
        raise HTTPException(404, "preview not found")
    return job


@router.get("/api/music/preview/{job_id}")
def preview_status(job_id: str):
    return _job_or_404(job_id).as_dict()


@router.get("/api/music/preview/{job_id}/video")
def preview_video(job_id: str):
    job = _job_or_404(job_id)
    if job.status != "done" or not job.output or not Path(job.output).is_file():
        raise HTTPException(409, "preview not ready")
    return FileResponse(job.output, media_type="video/mp4")
