# Contributing to AMV Tools

Thanks for helping! This project is local-first: the Electron/React app talks to
a FastAPI sidecar over `127.0.0.1`, and all data stays on the user's machine.

## Setup

```bash
uv sync --extra cpu      # CPU-only dev environment
cd app
npm ci
```

Node 20+ and Python 3.12/3.13 (managed by `uv`) are required. See the README
for GPU extras and packaging.

## Checks before opening a PR

```bash
uv run pytest -m "not slow"
cd app && npx tsc --noEmit && npx vitest run && npm run build
```

Backend tests use synthetic ffmpeg clips and mocked ML models, so they run on a
CPU-only machine. Do not commit media files larger than 1 MB; generate fixtures
with ffmpeg instead.

## Guidelines

- Keep user-visible text in both French and English (`app/src/i18n/`).
- Never silently swallow an error: surface an actionable message in the UI and
  log it on the backend.
- Prefer new files for new features; keep edits to shared files small and
  localized.
- SQLite schema changes are centralised — coordinate before adding migrations.
- Do not commit generated artifacts (`app/release/`, `.venv/`, `dist/`).

## Reporting bugs

Use the bug-report issue template. It asks for the report from
**Settings → Diagnostic → Copy report**; that includes versions, the requested
vs. active acceleration backend, model state and recent backend errors. Never
attach copyrighted video files.
