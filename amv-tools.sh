#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

UV_DIR="$PWD/.uv"
export PATH="$UV_DIR:$PATH"

if ! command -v npx >/dev/null 2>&1; then
  echo "Node.js / npx not found. Run install.sh first."
  exit 1
fi

cd app
npx electron .
