"""Local-only access control for the sidecar.

The sidecar listens on 127.0.0.1, but any web page open in a browser can
still reach it (DNS rebinding, body-less cross-site POSTs, websockets). Two
layers close that:

* ``Host`` check, always on: only ``127.0.0.1`` / ``localhost`` (any port)
  are served, which defeats DNS rebinding. ``AMV_ALLOWED_HOSTS`` (comma
  list) adds names, for the test-suite's ``testserver`` only.
* Per-launch token, on when ``AMV_TOKEN`` is set (Electron generates one at
  every launch): each request must carry it in the ``X-AMV-Token`` header,
  or, for what cannot set headers (``<img>``/``<video>`` src, WebSocket),
  in a ``?token=`` query parameter on GET/HEAD and websocket requests.
  A standalone ``python backend/server.py`` without ``AMV_TOKEN`` keeps
  working without a token (development).

Requests carrying a browser ``Origin`` outside the app's own origins are
refused too, so a web page cannot drive a token-less dev sidecar.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
from urllib.parse import parse_qsl

log = logging.getLogger("amv.security")

TOKEN_ENV = "AMV_TOKEN"
TOKEN_HEADER = "x-amv-token"
TOKEN_QUERY = "token"

DEFAULT_HOSTS = frozenset({"127.0.0.1", "localhost"})

# Origins the renderer can run from: the packaged app (Electron sends
# "file://" from loadFile pages), a custom app:// scheme, and the Vite dev
# server. "null" is deliberately absent: any site can send it from a
# sandboxed iframe.
ALLOWED_ORIGINS = frozenset({
    "file://",
    "app://",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
})


def origin_allowed(origin: str | None) -> bool:
    """True for no Origin (non-browser client) or one of the app's origins."""
    if origin is None or origin == "":
        return True
    origin = origin.strip().lower()
    return origin in ALLOWED_ORIGINS or origin.startswith("app://")


def _host_name(host: str) -> str:
    host = host.strip().lower()
    if host.startswith("["):  # [::1]:port
        return host[: host.find("]") + 1] if "]" in host else host
    return host.rsplit(":", 1)[0] if ":" in host else host


def _allowed_hosts() -> frozenset[str]:
    extra = {h.strip().lower() for h in os.environ.get("AMV_ALLOWED_HOSTS", "").split(",") if h.strip()}
    return DEFAULT_HOSTS | extra


def token_matches(expected: str, given: str | None) -> bool:
    if not given:
        return False
    return hmac.compare_digest(expected.encode("utf-8"), given.encode("utf-8"))


class LocalAccessMiddleware:
    """Pure ASGI middleware (covers both HTTP and websocket scopes)."""

    def __init__(self, app, token: str | None = None, allowed_hosts: frozenset[str] | None = None):
        self.app = app
        self.token = token if token is not None else (os.environ.get(TOKEN_ENV) or None)
        self.allowed_hosts = allowed_hosts if allowed_hosts is not None else _allowed_hosts()
        if self.token:
            log.info("sidecar API token required (X-AMV-Token)")

    async def __call__(self, scope, receive, send):
        kind = scope["type"]
        if kind not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}

        host = headers.get("host", "")
        if _host_name(host) not in self.allowed_hosts:
            await self._deny(kind, send, 400, "invalid_host")
            return

        method = scope.get("method", "GET")
        # A CORS preflight never carries custom headers; CORSMiddleware
        # (inside this one) answers it without reaching any route.
        preflight = method == "OPTIONS" and "access-control-request-method" in headers
        if not preflight and not origin_allowed(headers.get("origin")):
            await self._deny(kind, send, 403, "forbidden_origin")
            return

        if self.token and not preflight:
            given = headers.get(TOKEN_HEADER)
            if not given and (kind == "websocket" or method in ("GET", "HEAD")):
                query = scope.get("query_string", b"").decode("latin-1")
                given = dict(parse_qsl(query)).get(TOKEN_QUERY)
            if not token_matches(self.token, given):
                await self._deny(kind, send, 403, "invalid_token")
                return

        await self.app(scope, receive, send)

    _denied_logged = 0

    async def _deny(self, kind, send, status: int, error: str):
        # A hostile page could hammer the port: log the first few only.
        if LocalAccessMiddleware._denied_logged < 20:
            LocalAccessMiddleware._denied_logged += 1
            log.warning("refused %s request: %s (%d)", kind, error, status)
        if kind == "websocket":
            # Closing before accept makes the server answer the handshake 403.
            await send({"type": "websocket.close", "code": 1008})
            return
        body = json.dumps({"error": error}).encode()
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        })
        await send({"type": "http.response.body", "body": body})
