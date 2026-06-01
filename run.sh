#!/usr/bin/env bash
#
# Auto-detect a Python virtual environment. If it does not exist, create it
# and install dependencies from requirements.txt. Then activate it.
#
# Usage:
#   source run.sh           # set up + activate the venv in your current shell
#   ./run.sh <args...>       # set up the venv, then run handpose3d.py with args

set -euo pipefail

# Resolve the directory this script lives in, so it works from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"

# Pick a python interpreter.
PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "Error: no python3/python interpreter found on PATH." >&2
    exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "No virtual environment found. Creating one at ${VENV_DIR} ..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"

    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    echo "Upgrading pip ..."
    python -m pip install --upgrade pip

    if [[ -f "${REQUIREMENTS}" ]]; then
        echo "Installing dependencies from requirements.txt ..."
        python -m pip install -r "${REQUIREMENTS}"
    fi
else
    echo "Found existing virtual environment at ${VENV_DIR}. Activating ..."
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

echo "Virtual environment is active: $(python --version) @ $(command -v python)"

# If the script was executed (not sourced) and given arguments, run the app.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]] && [[ "$#" -gt 0 ]]; then
    exec python "${SCRIPT_DIR}/handpose3d.py" "$@"
fi
