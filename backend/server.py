"""AMV Tools sidecar entrypoint.

Usage:
    python backend/server.py --port 8731
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Allow `python backend/server.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from backend.api import ROUTERS  # noqa: E402
from backend.state import get_state  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("amv.server")


def _silence_proactor_reset(loop, context):
    """asyncio handler that swallows the harmless ConnectionResetError thrown
    by Windows' Proactor event loop when the renderer closes a websocket
    abruptly (Ctrl+R, window close, navigation). It's a known Windows-only
    bug-but-noisy quirk; on Unix the equivalent EPIPE is silent. We drop the
    matching context here so the logs stay clean."""
    exc = context.get("exception")
    if isinstance(exc, ConnectionResetError):
        return
    # Default behaviour for anything else.
    loop.default_exception_handler(context)


def make_app() -> FastAPI:
    app = FastAPI(title="AMV Tools sidecar", version="0.2.0a")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "file://",
            "app://",
        ],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )
    for r in ROUTERS:
        app.include_router(r)

    @app.exception_handler(ImportError)
    async def _import_error_handler(request: Request, exc: ImportError):
        # BiRefNet (and other HF models loaded with trust_remote_code=True)
        # can raise ImportError at first use if a transitive dep is missing
        # from the sidecar venv — `timm`, `einops`, `kornia`, etc. We turn
        # that into a 503 with a structured payload so the frontend can show
        # a friendly "runtime incomplete" banner instead of a bare 500.
        log.exception("ImportError reaching the FastAPI surface: %s", exc)
        return JSONResponse(
            status_code=503,
            content={
                "error": "runtime_incomplete",
                "detail": str(exc),
                "hint": (
                    "A model runtime dependency is missing from the sidecar venv. "
                    "Reinstall the GPU backend from Settings → Models, or run "
                    "`uv sync --extra <your-backend>` from a checkout."
                ),
            },
        )

    @app.on_event("startup")
    async def _on_start():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(_silence_proactor_reset)
        get_state().set_loop(loop)
        log.info("AMV Tools sidecar ready")

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8731)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    app = make_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
