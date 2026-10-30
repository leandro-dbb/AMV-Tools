"""AMV Tools sidecar entrypoint.

Usage:
    python backend/server.py --port 8731
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# PyTorch's MPS backend still lacks a few ops; opt into the CPU fallback
# before anything can import torch, so a manual `python backend/server.py`
# behaves like an Electron launch (main.ts sets the same variable).
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # The MPS allocator only starts recycling its cached blocks at 1.4x
    # Metal's working set by default (~16.5 GB on a 16 GB Mac), so SAM 2's
    # varying per-frame transients grew a 12 GB cache over a ~2 GB live set
    # and made the machine swap (M1 Pro: 13.1 GB footprint, ~4 GB swapped
    # for a 2 s clip). At 0.25 the footprint is 7.5 GB, no swap, and no pass
    # got slower (indexing, upscale, RIFE, depth, SAM 2 measured).
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.25")

# Allow `python backend/server.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from backend.api import ROUTERS, searchplus  # noqa: E402
from backend.diagnostics import setup_logging  # noqa: E402
from backend import shutdown  # noqa: E402
from backend.security import LocalAccessMiddleware  # noqa: E402
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


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Same order as the former on_event("startup") hooks: the search+ router
    # hook was registered (via include_router) before the app-level one.
    searchplus.start_auto_analyze()
    setup_logging()
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(_silence_proactor_reset)
    get_state().set_loop(loop)
    get_state().start_search_warmup()
    from backend import autotune
    autotune.start_background()
    log.info("AMV Tools sidecar ready")
    yield


def make_app() -> FastAPI:
    app = FastAPI(title="AMV Tools sidecar", version="0.4.0a0", lifespan=_lifespan)
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
    # Added last so it wraps CORS: Host / Origin / per-launch token check.
    app.add_middleware(LocalAccessMiddleware)
    for r in ROUTERS:
        app.include_router(r)

    @app.get("/health")
    def health():
        # Cheap liveness probe used by the Windows smoke test and by anyone
        # wiring the sidecar into a launcher/healthcheck.  It must not touch
        # the DB or the models so it answers even while indexing runs.
        return {"ok": True, "service": "amv-tools"}

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
                # the hint in "detail" too: most panels only show that field
                "detail": f"{exc} — the app's Python environment is missing a module: "
                          "restart AMV Tools (it installs new dependencies after an update) "
                          "or reinstall the backend from Settings → Models.",
                "hint": (
                    "A model runtime dependency is missing from the sidecar venv. "
                    "Reinstall the GPU backend from Settings → Models, or run "
                    "`uv sync --extra <your-backend>` from a checkout."
                ),
            },
        )

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8731)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    setup_logging()
    shutdown.install_child_tracking()
    app = make_app()
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info", access_log=False,
                            timeout_graceful_shutdown=shutdown.GRACE_S)
    server = shutdown.SidecarServer(config)
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    # SIGTERM never gets here (uvicorn re-raises it once stopped); Ctrl+C and
    # POST /api/shutdown do, and must not wait on leftover worker threads.
    shutdown.exit_process(0 if server.started else 3)


if __name__ == "__main__":
    main()
