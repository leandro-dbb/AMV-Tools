#!/usr/bin/env bash
# AMV Tools bootstrap. The GPU backend choice happens inside the app on first run.
set -euo pipefail

cd "$(dirname "$0")"

UV_DIR="$PWD/.uv"
export UV_PYTHON_INSTALL_DIR="$UV_DIR/python"
export UV_CACHE_DIR="$UV_DIR/uv_cache"
export UV_VENV_CLEAR=1
UV_EXE="$UV_DIR/uv"

if [ ! -x "$UV_EXE" ]; then
  echo "Downloading uv into $UV_DIR..."
  mkdir -p "$UV_DIR"
  UV_INSTALL_DIR="$UV_DIR" UV_UNMANAGED_INSTALL=1 \
    bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

export PATH="$UV_DIR:$PATH"

echo "Syncing CPU-only baseline (GPU backend installs in-app)..."
"$UV_EXE" sync --extra cpu

if ! command -v npm >/dev/null 2>&1; then
  echo "[!] npm not found. Install Node.js 20+ from https://nodejs.org"
  exit 1
fi

pushd app >/dev/null
echo "Building Electron frontend..."
npm install --silent
npm run build
popd >/dev/null

echo "Bootstrap complete. Launch AMV Tools to pick your GPU backend."
