#!/usr/bin/env bash
# Webcam 3D Mapper - macOS / Linux launcher.
# Creates .venv if needed, installs dependencies, starts the local server.

set -euo pipefail
cd "$(dirname "$0")/.."

printf '\n  Webcam 3D Mapper\n  ----------------\n\n'

# --- Python -----------------------------------------------------------------
PY=""
for candidate in python3.13 python3.12 python3.11 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
done

if [ -z "$PY" ]; then
  echo "  [X] Python was not found. Install Python 3.11 or newer and re-run this script."
  exit 1
fi

if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "  [X] Python 3.11 or newer is required (found $("$PY" --version))."
  exit 1
fi

# --- Virtual environment ------------------------------------------------------
if [ ! -x ".venv/bin/python" ]; then
  echo "  Creating virtual environment..."
  "$PY" -m venv .venv
fi
VPY=".venv/bin/python"

# Install only when the dependency set changed, so normal starts stay fast.
STAMP=".venv/.deps-installed"
if ! { [ -f "$STAMP" ] && cmp -s "$STAMP" requirements.txt; }; then
  echo "  Installing dependencies..."
  "$VPY" -m pip install --quiet --upgrade pip
  "$VPY" -m pip install --quiet -r requirements.txt
  cp requirements.txt "$STAMP"
fi

# --- COLMAP notice ------------------------------------------------------------
if ! command -v colmap >/dev/null 2>&1; then
  cat <<'NOTE'
  [!] COLMAP was not found on PATH.
      Scanning will work, but reconstruction needs COLMAP:
        macOS  brew install colmap
        Linux  sudo apt install colmap   (or build from source)
      You can also set the path in the app's Settings panel, or set COLMAP_PATH.

NOTE
fi

echo "  Starting server on http://127.0.0.1:8765"
echo "  Press Ctrl+C to stop."
echo
exec "$VPY" app.py --open "$@"
