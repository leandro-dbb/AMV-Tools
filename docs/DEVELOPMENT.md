# Development: build from source, tests, architecture

[← Back to the README](../README.md)

## Build From Source

Requirements:

- Python 3.12 or 3.13, managed by `uv`.
- Node.js 20+.
- A CUDA-capable NVIDIA GPU for the recommended path, or CPU otherwise.

Install dependencies:

```bash
uv sync --extra cpu
cd app
npm install
```

Run the Electron app in development:

```bash
cd app
npm run dev:electron
```

Build and test checks:

```bash
uv run pytest -m "not slow"
cd app
npx tsc --noEmit
npx vitest run
npm run build
```

Backend tests use synthetic clips generated with ffmpeg and mocked ML models,
so a CPU-only machine is enough. `uv sync --extra cpu` installs the test group
automatically. Tests marked `slow` run the real finishing models on CPU and are
skipped unless their weights are present (`AMV_TEST_DOWNLOAD=1` fetches them).

End-to-end UI journey (the built Electron app + the real sidecar and models on
CPU, driven by Playwright under its own Xvfb, throwaway profile):

```bash
cd app
npm run build
npm run e2e                                   # synthetic clips, ~5 min
AMV_E2E_MEDIA="/path/to/Séries test" AMV_E2E_CAPTURES=/tmp/caps npm run e2e
```

The first run needs the SigLIP 2 and wd-tagger weights (~2 GB, Hugging Face cache).

Create a local unsigned Windows installer:

```bash
cd app
npm run build
npm run build:electron:unsigned
```

The installer is written to `app/release/`. Release artifacts are ignored by
Git; publish them through GitHub Releases instead of committing them.

### Building For macOS

macOS builds must be produced on macOS (electron-builder limitation). On a Mac:

```bash
# Get the uv binary the app bundles
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p .uv && cp "$(command -v uv)" .uv/uv

cd app
npm install
npm run build
npx cross-env CSC_IDENTITY_AUTO_DISCOVERY=false electron-builder --mac
```

Alternatively, the repository ships `.github/workflows/build.yml`, which builds
Windows, macOS (arm64) and Linux (`AppImage` + `.deb`) artifacts on GitHub-hosted
runners (trigger it from the Actions tab, or push a `v*` tag). Pushing a tag also
creates a draft GitHub Release with all installers attached. `.github/workflows/ci.yml`
runs the backend/frontend test suites on every push and pull request.

## What Is Inside

- Electron + React desktop shell, bilingual EN/FR.
- FastAPI sidecar for indexing, search, tagging, and exports.
- SQLite scene databases that stay on your machine.
- SigLIP 2 semantic embeddings for visual scene search, similar shots and
  user-defined characters (base or so400m, picked by the model tier).
- OpenCV (Apache-2.0) optical-flow camera-motion / intensity analysis, CPU only.
- wd-tagger for anime-oriented tag discovery.
- ffmpeg-based proxies, trim previews, and clip exports.
- librosa (ISC) beat / section / drop analysis for the Music tab (CPU, no
  model weights; pulls numba, scikit-learn, soundfile and soxr — the latter
  LGPL-2.1+, used as an unmodified dynamic library).
- BiRefNet / ISNet anime + SAM 2 alpha workflow for short character cutouts (beta).
- Finishing toolbox: Real-ESRGAN anime upscaling, RIFE interpolation, Depth
  Anything V2 depth maps, as queued background jobs.

## Repository Layout

```text
app/        Electron main process and React renderer (i18n in app/src/i18n)
backend/    FastAPI sidecar, indexing pipeline, model wrappers, export tools
assets/     Icons and screenshots used by the app and README
```

Runtime data, model caches, generated proxies, exports, local databases, and
virtual environments are intentionally ignored by Git.
