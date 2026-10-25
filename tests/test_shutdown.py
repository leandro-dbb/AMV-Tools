"""Sidecar shutdown: a real ``backend/server.py`` process, with the progress
websocket open and an ffmpeg job running, must exit together with all its
children within a few seconds of SIGTERM (it used to wait on the websocket
forever)."""
from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from backend.db import queries, schema

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals, FIFO and ps")

ROOT = Path(__file__).resolve().parent.parent
TOKEN = "shutdown-test-token"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _descendants(pid: int) -> set[int]:
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True).stdout
    children: dict[int, list[int]] = {}
    for line in out.splitlines():
        p, pp = (int(x) for x in line.split())
        children.setdefault(pp, []).append(p)
    found, todo = set(), [pid]
    while todo:
        for c in children.get(todo.pop(), []):
            if c not in found:
                found.add(c)
                todo.append(c)
    return found


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # a zombie is gone for our purpose (its parent just hasn't reaped it)
    st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout
    return bool(st.strip()) and not st.strip().startswith("Z")


def _post(port: int, path: str) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="POST", data=b"",
                                 headers={"X-AMV-Token": TOKEN})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


@pytest.fixture
def sidecar(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    # A FIFO nobody writes to: the transcode's ffmpeg blocks on it, i.e. a
    # job that is still running whenever the stop request comes.
    fifo = tmp_path / "never-ending.mp4"
    os.mkfifo(fifo)
    db = data / "amv_tools.db"
    schema.init_db(db)
    with schema.get_conn(db) as conn:
        video_id = queries.upsert_video(conn, str(fifo), "fifo", 60_000, 25.0, "320x240", "test",
                                        status="done")
        conn.commit()
    port = _free_port()
    env = dict(os.environ, AMV_DATA_DIR=str(data), AMV_TOKEN=TOKEN, AMV_AUTOTUNE="0",
               PYTHONUNBUFFERED="1")
    env.pop("PYTEST_CURRENT_TEST", None)
    log = open(tmp_path / "sidecar.log", "w")
    proc = subprocess.Popen([sys.executable, str(ROOT / "backend" / "server.py"), "--port", str(port)],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        deadline = time.monotonic() + 60
        while True:
            assert proc.poll() is None, (tmp_path / "sidecar.log").read_text()
            try:
                socket.create_connection(("127.0.0.1", port), 0.2).close()
                break
            except OSError:
                assert time.monotonic() < deadline, "sidecar did not start"
                time.sleep(0.1)
        yield proc, port, video_id
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        log.close()


async def _stop_with_ws_open(proc, port: int, video_id: int, stop) -> tuple[float, set[int]]:
    import websockets

    async with websockets.connect(f"ws://127.0.0.1:{port}/ws/progress?token={TOKEN}") as ws:
        # the socket still streams progress (empty Search+ analysis: one "done" event)
        await asyncio.to_thread(_post, port, "/api/searchplus/analyze")
        while json.loads(await asyncio.wait_for(ws.recv(), 5)).get("type") != "analysis":
            pass
        assert (await asyncio.to_thread(_post, port, f"/api/videos/{video_id}/transcode"))["ok"]
        deadline = time.monotonic() + 10
        while not (kids := _descendants(proc.pid)):
            assert time.monotonic() < deadline, "the transcode never started ffmpeg"
            await asyncio.sleep(0.05)
        t0 = time.monotonic()
        await asyncio.to_thread(stop)
        while proc.poll() is None and time.monotonic() - t0 < 10:
            await asyncio.sleep(0.05)
        elapsed = time.monotonic() - t0
        with pytest.raises(websockets.ConnectionClosed):   # closed, not left hanging
            while True:
                await asyncio.wait_for(ws.recv(), 2)
    return elapsed, kids


def test_sigterm_with_websocket_and_job_exits_fast(sidecar):
    proc, port, video_id = sidecar
    elapsed, kids = asyncio.run(_stop_with_ws_open(
        proc, port, video_id, lambda: os.kill(proc.pid, signal.SIGTERM)))
    assert proc.poll() is not None and elapsed < 5, f"still running after {elapsed:.1f}s"
    assert not [k for k in kids if _alive(k)], "child process left behind"


def test_shutdown_endpoint_with_websocket_and_job_exits_fast(sidecar):
    proc, port, video_id = sidecar
    elapsed, kids = asyncio.run(_stop_with_ws_open(
        proc, port, video_id, lambda: _post(port, "/api/shutdown")))
    assert proc.poll() == 0 and elapsed < 5, f"exit {proc.poll()} after {elapsed:.1f}s"
    assert not [k for k in kids if _alive(k)], "child process left behind"
