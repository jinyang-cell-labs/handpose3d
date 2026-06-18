#!/usr/bin/env bash
#
# Bootstrap + launch the IP stream recorder GUI.
#   - finds (or creates) a local .venv next to this script
#   - installs requirements.txt on first run (or with --reinstall)
#   - launches the PyQt recorder
#
# Usage:
#   ./start_recorder.bash              # normal launch
#   ./start_recorder.bash --reinstall  # rebuild deps, then launch
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQS="$SCRIPT_DIR/requirements.txt"
STAMP="$VENV_DIR/.deps_installed"

REINSTALL=0
for arg in "$@"; do
    case "$arg" in
        --reinstall) REINSTALL=1 ;;
    esac
done

# Pick a python3.
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "error: '$PYTHON_BIN' not found on PATH. Set PYTHON_BIN=/path/to/python3" >&2
    exit 1
fi

# Create the venv if missing.
if [ ! -d "$VENV_DIR" ]; then
    echo ">> creating virtualenv at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    REINSTALL=1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Install deps on first run, forced reinstall, or if the stamp is missing/stale.
if [ "$REINSTALL" -eq 1 ] || [ ! -f "$STAMP" ] || [ "$REQS" -nt "$STAMP" ]; then
    echo ">> installing requirements"
    python -m pip install --upgrade pip
    python -m pip install -r "$REQS"
    touch "$STAMP"
fi

echo ">> launching recorder"
exec python "$SCRIPT_DIR/recorder.py" "$@"
