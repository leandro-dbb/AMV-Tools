"""Index queue + WebSocket progress."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..config import VIDEO_EXTENSIONS
from ..db import queries, schema
from ..state import get_state

router = APIRouter()


class QueueEntry(BaseModel):
    path: str
    recursive: bool = True


class EnqueueBody(BaseModel):
    paths: List[QueueEntry]


@router.get("/api/index/queue")
def get_queue():
    state = get_state()
    schema.init_db(state.primary_db)
    with schema.get_conn(state.primary_db) as conn:
        return {"items": queries.get_queue(conn)}


@router.post("/api/index/queue")
def add_queue(body: EnqueueBody):
    state = get_state()
    schema.init_db(state.primary_db)
    added = 0
    with schema.get_conn(state.primary_db) as conn:
        for e in body.paths:
            p = Path(e.path)
            if not p.exists():
                continue
            is_dir = p.is_dir()
            if not is_dir and p.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            if queries.add_to_queue(conn, str(p), is_dir, e.recursive):
                added += 1
    return {"added": added}


@router.delete("/api/index/queue/{item_id}")
def remove_queue(item_id: int):
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        queries.remove_from_queue(conn, item_id)
    return {"ok": True}


class StartIndexBody(BaseModel):
    phases: List[str] | None = None  # subset of ["tag", "embed"]; default = both


@router.post("/api/index/start")
def start_indexing(body: StartIndexBody | None = None):
    requested = (body.phases if body and body.phases else ["tag", "embed"])
    phases = tuple(p for p in requested if p in ("tag", "embed")) or ("tag", "embed")
    started = get_state().start_indexing(phases=phases)
    return {"started": started, "phases": list(phases)}


@router.post("/api/index/stop")
def stop_indexing():
    get_state().stop_indexing()
    return {"stopped": True}


@router.websocket("/ws/progress")
async def progress(ws: WebSocket):
    await ws.accept()
    state = get_state()
    queue = state.subscribe()
    try:
        while True:
            event = await queue.get()
            await ws.send_text(json.dumps(event))
    except WebSocketDisconnect:
        pass
    finally:
        state.unsubscribe(queue)
