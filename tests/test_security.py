"""Sidecar access control: Host check, per-launch token, websocket Origin."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

TOKEN = "abc123"
BASE = "http://127.0.0.1:8731"


@pytest.fixture
def secured(state, monkeypatch):
    monkeypatch.setenv("AMV_TOKEN", TOKEN)
    from backend.server import make_app

    # No `with`: the lifespan (warm-up, autotune) stays off.
    return TestClient(make_app(), base_url=BASE)


@pytest.fixture
def open_client(state, monkeypatch):
    monkeypatch.delenv("AMV_TOKEN", raising=False)
    from backend.server import make_app

    return TestClient(make_app(), base_url=BASE)


def test_foreign_host_rejected(secured, open_client):
    for c in (secured, open_client):
        r = c.get("/api/status", headers={"Host": "evil.com", "X-AMV-Token": TOKEN})
        assert r.status_code == 400
        # DNS rebinding keeps the port: still refused.
        assert c.get("/api/status", headers={"Host": "evil.com:8731"}).status_code == 400


def test_local_hosts_accepted(open_client):
    for host in ("127.0.0.1:8731", "localhost:8731", "localhost", "127.0.0.1"):
        assert open_client.get("/api/status", headers={"Host": host}).status_code == 200


def test_token_required_when_set(secured):
    assert secured.get("/api/status").status_code == 403
    assert secured.get("/api/status", headers={"X-AMV-Token": "wrong"}).status_code == 403
    assert secured.post("/api/index/start").status_code == 403
    assert secured.get("/api/status", headers={"X-AMV-Token": TOKEN}).status_code == 200


def test_query_token_for_media_get(secured):
    # <img>/<video> cannot send headers: ?token= reaches the route (404 = no
    # such scene, i.e. past the middleware), a wrong one does not.
    assert secured.get("/api/scene/999999/thumbnail").status_code == 403
    assert secured.get("/api/scene/999999/thumbnail?token=nope").status_code == 403
    assert secured.get(f"/api/scene/999999/thumbnail?token={TOKEN}").status_code == 404
    assert secured.get(f"/api/status?token={TOKEN}").status_code == 200


def test_query_token_not_accepted_for_writes(secured):
    assert secured.post(f"/api/index/stop?token={TOKEN}").status_code == 403
    assert secured.post("/api/index/stop", headers={"X-AMV-Token": TOKEN}).status_code == 200


def test_no_token_needed_without_env(open_client):
    assert open_client.get("/api/status").status_code == 200


def test_cors_preflight_allows_token_header(secured):
    r = secured.options(
        "/api/status",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-amv-token,content-type",
        },
    )
    assert r.status_code == 200
    assert "x-amv-token" in r.headers["access-control-allow-headers"].lower()


def test_foreign_origin_rejected_on_http(open_client):
    assert open_client.post("/api/index/stop", headers={"Origin": "https://evil.com"}).status_code == 403
    assert open_client.post("/api/index/stop", headers={"Origin": "file://"}).status_code == 200
    # Sandboxed iframes on any site send Origin: null.
    assert open_client.post("/api/index/stop", headers={"Origin": "null"}).status_code == 403


def test_ws_progress_token_and_origin(secured):
    with secured.websocket_connect(f"/ws/progress?token={TOKEN}", headers={"Origin": "file://"}):
        pass
    with secured.websocket_connect(f"/ws/progress?token={TOKEN}", headers={"Origin": "http://localhost:5173"}):
        pass
    with pytest.raises(WebSocketDisconnect):
        with secured.websocket_connect("/ws/progress", headers={"Origin": "file://"}):
            pass
    with pytest.raises(WebSocketDisconnect):
        with secured.websocket_connect(f"/ws/progress?token={TOKEN}", headers={"Origin": "https://evil.com"}):
            pass


def test_ws_origin_checked_in_route(state):
    """/ws/progress refuses foreign origins even without the middleware."""
    from fastapi import FastAPI
    from backend.api import index

    app = FastAPI()
    app.include_router(index.router)
    c = TestClient(app)
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/ws/progress", headers={"Origin": "https://evil.com"}):
            pass
    with c.websocket_connect("/ws/progress"):
        pass
