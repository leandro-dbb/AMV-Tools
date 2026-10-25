"""Characters: user-defined, per-series, built from example scenes/images."""
from __future__ import annotations

import io
import json
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field

from ..db import schema
from ..searchplus import characters as chars
from ..searchplus import filters as sf
from ..state import get_state

router = APIRouter()
log = logging.getLogger(__name__)


class ExampleIn(BaseModel):
    scene_id: int
    polarity: int = 1
    crop: Optional[list[float]] = None  # normalised [x0, y0, x1, y1]


class CharacterCreate(BaseModel):
    name: str
    group_name: Optional[str] = None
    examples: list[ExampleIn] = Field(default_factory=list)


class CharacterPatch(BaseModel):
    name: Optional[str] = None
    group_name: Optional[str] = None
    clear_group: bool = False
    threshold: Optional[float] = None
    threshold_auto: Optional[bool] = None


class FeedbackIn(BaseModel):
    accept: list[int] = Field(default_factory=list)
    reject: list[int] = Field(default_factory=list)
    clear: list[int] = Field(default_factory=list)


def _db() -> str:
    db = get_state().primary_db
    schema.init_db(db)
    return db


def _embed_image(img) -> "object":
    try:
        return get_state().get_siglip().embed_images([img])[0]
    except ImportError:
        raise
    except Exception as exc:
        log.warning("SigLIP unavailable for character example: %s", exc)
        raise HTTPException(status_code=503, detail=f"SigLIP 2 model unavailable: {exc}")


def _add_scene_example(conn, cid: int, ex: ExampleIn) -> int:
    if ex.crop:
        try:
            img = chars.crop_image(chars.scene_frame(conn, ex.scene_id), ex.crop)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"scene {ex.scene_id} not found")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        emb = _embed_image(img)
        return chars.add_example(conn, cid, emb, scene_id=ex.scene_id, polarity=ex.polarity,
                                 crop=ex.crop, thumbnail=chars.thumb_jpeg(img), source="crop")
    emb = chars.scene_embedding(conn, ex.scene_id)
    if emb is None:
        raise HTTPException(status_code=409, detail=f"scene {ex.scene_id} has no embedding yet")
    return chars.add_example(conn, cid, emb, scene_id=ex.scene_id, polarity=ex.polarity,
                             source="scene" if ex.polarity > 0 else "feedback")


def _rescore(conn, cid: int) -> dict:
    return chars.rescore(conn, cid, get_state().get_search_index())


def _detail(conn, cid: int) -> dict:
    c = chars.get_character(conn, cid)
    if c is None:
        raise HTTPException(status_code=404, detail="character not found")
    c["examples"] = [
        {k: v for k, v in ex.items() if k != "embedding"} for ex in chars.examples(conn, cid)
    ]
    return c


@router.get("/api/characters")
def list_characters():
    db = _db()
    index = get_state().get_search_index()
    with schema.get_conn(db) as conn:
        chars.ensure_fresh(conn, index)
        return {"characters": chars.list_characters(conn)}


@router.post("/api/characters")
def create_character(body: CharacterCreate):
    db = _db()
    group = (body.group_name or "").strip() or None
    with schema.get_conn(db) as conn:
        try:
            cid = chars.create_character(conn, body.name, group)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        for ex in body.examples:
            _add_scene_example(conn, cid, ex)
        summary = _rescore(conn, cid)
        return {**_detail(conn, cid), "rescore": summary}


@router.get("/api/characters/{cid}")
def character_detail(cid: int):
    with schema.get_conn(_db()) as conn:
        return _detail(conn, cid)


@router.patch("/api/characters/{cid}")
def patch_character(cid: int, body: CharacterPatch):
    with schema.get_conn(_db()) as conn:
        if chars.get_character(conn, cid) is None:
            raise HTTPException(status_code=404, detail="character not found")
        if body.name is not None and body.name.strip():
            conn.execute("UPDATE characters SET name = ? WHERE id = ?", (body.name.strip(), cid))
        if body.clear_group:
            conn.execute("UPDATE characters SET group_name = NULL WHERE id = ?", (cid,))
        elif body.group_name is not None:
            conn.execute("UPDATE characters SET group_name = ? WHERE id = ?",
                         (body.group_name.strip() or None, cid))
        if body.threshold is not None:
            thr = min(max(float(body.threshold), chars.MIN_THRESHOLD), chars.MAX_THRESHOLD)
            conn.execute("UPDATE characters SET threshold = ?, threshold_auto = 0 WHERE id = ?", (thr, cid))
        if body.threshold_auto is not None:
            conn.execute("UPDATE characters SET threshold_auto = ? WHERE id = ?",
                         (int(body.threshold_auto), cid))
        summary = _rescore(conn, cid)
        return {**_detail(conn, cid), "rescore": summary}


@router.delete("/api/characters/{cid}")
def delete_character(cid: int):
    with schema.get_conn(_db()) as conn:
        conn.execute("DELETE FROM characters WHERE id = ?", (cid,))
    return {"ok": True}


@router.post("/api/characters/{cid}/examples")
def add_example(cid: int, body: ExampleIn):
    with schema.get_conn(_db()) as conn:
        if chars.get_character(conn, cid) is None:
            raise HTTPException(status_code=404, detail="character not found")
        ex_id = _add_scene_example(conn, cid, body)
        summary = _rescore(conn, cid)
        return {**_detail(conn, cid), "example_id": ex_id, "rescore": summary}


@router.post("/api/characters/{cid}/examples/image")
async def add_image_example(cid: int, image: UploadFile = File(...),
                            polarity: int = Form(default=1),
                            crop: Optional[str] = Form(default=None)):
    raw = await image.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        box = json.loads(crop) if crop else None
        img = chars.crop_image(img, box)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"unreadable image: {exc}")
    with schema.get_conn(_db()) as conn:
        if chars.get_character(conn, cid) is None:
            raise HTTPException(status_code=404, detail="character not found")
        emb = _embed_image(img)
        ex_id = chars.add_example(conn, cid, emb, polarity=polarity, crop=box,
                                  thumbnail=chars.thumb_jpeg(img), source="image")
        summary = _rescore(conn, cid)
        return {**_detail(conn, cid), "example_id": ex_id, "rescore": summary}


@router.delete("/api/characters/{cid}/examples/{ex_id}")
def delete_example(cid: int, ex_id: int):
    with schema.get_conn(_db()) as conn:
        conn.execute("DELETE FROM character_examples WHERE id = ? AND character_id = ?", (ex_id, cid))
        summary = _rescore(conn, cid)
        return {**_detail(conn, cid), "rescore": summary}


@router.get("/api/characters/{cid}/examples/{ex_id}/thumbnail")
def example_thumbnail(cid: int, ex_id: int):
    with schema.get_conn(_db()) as conn:
        row = conn.execute(
            """SELECT e.thumbnail, s.thumbnail FROM character_examples e
                 LEFT JOIN scenes s ON s.id = e.scene_id
                WHERE e.id = ? AND e.character_id = ?""",
            (ex_id, cid),
        ).fetchone()
    data = (row[0] or row[1]) if row else None
    if not data:
        raise HTTPException(status_code=404, detail="no thumbnail")
    return Response(content=data, media_type="image/jpeg")


@router.post("/api/characters/{cid}/feedback")
def feedback(cid: int, body: FeedbackIn):
    """Validate / reject scored scenes, then re-score: accepted scenes become
    positive examples, rejected ones negative examples."""
    with schema.get_conn(_db()) as conn:
        if chars.get_character(conn, cid) is None:
            raise HTTPException(status_code=404, detail="character not found")
        for sid in body.clear:
            conn.execute(
                "DELETE FROM character_examples WHERE character_id = ? AND scene_id = ? AND crop IS NULL",
                (cid, sid),
            )
        for sid in body.accept:
            _add_scene_example(conn, cid, ExampleIn(scene_id=sid, polarity=1))
        for sid in body.reject:
            _add_scene_example(conn, cid, ExampleIn(scene_id=sid, polarity=-1))
        summary = _rescore(conn, cid)
        return {**_detail(conn, cid), "rescore": summary}


@router.post("/api/characters/{cid}/rescore")
def rescore(cid: int):
    with schema.get_conn(_db()) as conn:
        if chars.get_character(conn, cid) is None:
            raise HTTPException(status_code=404, detail="character not found")
        return _rescore(conn, cid)


@router.get("/api/characters/{cid}/matches")
def character_matches(cid: int, limit: int = Query(default=300, ge=1, le=5000),
                      offset: int = Query(default=0, ge=0)):
    index = get_state().get_search_index()
    with schema.get_conn(_db()) as conn:
        c = chars.get_character(conn, cid)
        if c is None:
            raise HTTPException(status_code=404, detail="character not found")
        chars.ensure_fresh(conn, index, [cid])
        c = chars.get_character(conn, cid)
        results = sf.finish_results(conn, chars.matches(conn, cid, limit=limit, offset=offset))
    return {"character": c, "results": results, "threshold": c["threshold"]}
