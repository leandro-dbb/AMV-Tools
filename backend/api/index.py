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
from ..security import origin_allowed
from ..state import get_state

router = APIRouter()


class QueueEntry(BaseModel):
    path: str
    recursive: bool = True
    # User-chosen import group ("Pokemon S23"…) stamped on every video found
    # under this path; shows up as a collapsible folder in the Derush playlist.
    group: str | None = None


class EnqueueBody(BaseModel):
    paths: List[QueueEntry]


@router.get("/api/index/queue")
def get_queue():
    state = get_state()
    schema.init_db(state.primary_db)
    with schema.get_conn(state.primary_db) as conn:
        items = queries.get_queue(conn)
    # the indexing thread also runs the post-queue proxy pass: the UI must
    # not offer "Start" while it is alive (the start would be refused)
    return {"items": items, "running": state.indexing_running()}


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
            group = (e.group or "").strip() or None
            if queries.add_to_queue(conn, str(p), is_dir, e.recursive, group_name=group):
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
    state = get_state()
    started = state.start_indexing(phases=phases)
    reason = None if started else state.indexing_busy_reason()
    return {"started": started, "phases": list(phases), "reason": reason}


@router.post("/api/index/stop")
def stop_indexing():
    get_state().stop_indexing()
    return {"stopped": True}


@router.websocket("/ws/progress")
async def progress(ws: WebSocket):
    # Browsers always send Origin on websockets: refuse other web pages (the
    # token itself is checked by LocalAccessMiddleware, ?token= here).
    if not origin_allowed(ws.headers.get("origin")):
        await ws.close(code=1008)
        return
    await ws.accept()
    state = get_state()
    queue = state.subscribe()
    # Watch receive() too: it is how the handler sees the socket close. Parked
    # in queue.get() alone it never did, and uvicorn's shutdown waited for it
    # forever (the sidecar stayed up while the renderer was connected).
    closed = asyncio.ensure_future(_until_disconnect(ws))
    try:
        while True:
            getter = asyncio.ensure_future(queue.get())
            await asyncio.wait({getter, closed}, return_when=asyncio.FIRST_COMPLETED)
            if not getter.done():
                getter.cancel()
                break
            event = getter.result()
            if event is None:   # sidecar shutting down (state.close_listeners)
                try:
                    await ws.close(code=1001)
                except RuntimeError:
                    pass        # uvicorn's own shutdown closed it first
                break
            await ws.send_text(json.dumps(event))
    except WebSocketDisconnect:
        pass
    finally:
        closed.cancel()
        state.unsubscribe(queue)


async def _until_disconnect(ws: WebSocket) -> None:
    try:
        while (await ws.receive())["type"] != "websocket.disconnect":
            pass
    except (WebSocketDisconnect, RuntimeError):
        pass
