#!/usr/bin/env bash
#
# start_evaluation.bash — set up and enter the log-analysis Python venv.
#
# Auto-detects ./.venv (creating it if missing), installs requirements.txt into
# it (only when that file changed), activates it, and keeps you inside the venv
# for analysing/visualising the handpose JSONL logs from
# ros2_ws/src/mediapie_landmarks_extraction (2D/3D plots, triangulation,
# non-linear optimisation).
#
# Usage:
#   ./start_evaluation.bash        # build (if needed) + drop into a venv shell
#   source start_evaluation.bash   # build (if needed) + activate in THIS shell
#
# Override the base interpreter with PYTHON_BIN=python3.x ./start_evaluation.bash

# --- locate self (works whether sourced or executed) -----------------------
if [ -n "${BASH_SOURCE[0]:-}" ]; then
  _self="${BASH_SOURCE[0]}"
else
  _self="$0"
fi
SCRIPT_DIR="$(cd "$(dirname "$_self")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQ_FILE="$SCRIPT_DIR/requirements.txt"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# Make the log loader (load_handpose_log.py) importable from analysis scripts.
LOADER_DIR="$REPO_ROOT/ros2_ws/src/mediapie_landmarks_extraction/scripts"

# Sourced or executed?
_sa_sourced=0
if [ -n "${BASH_SOURCE[0]:-}" ] && [ "${BASH_SOURCE[0]}" != "$0" ]; then
  _sa_sourced=1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

# --- build / refresh the venv ----------------------------------------------
_se_build() {
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[start_evaluation] '$PYTHON_BIN' not found on PATH" >&2
    return 1
  fi
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "[start_evaluation] creating venv: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR" || return 1
  else
    echo "[start_evaluation] reusing venv: $VENV_DIR"
  fi

  "$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip || return 1

  if [ -f "$REQ_FILE" ]; then
    local stamp="$VENV_DIR/.requirements.sha"
    local cur
    cur="$(sha1sum "$REQ_FILE" | awk '{print $1}')"
    if [ "$(cat "$stamp" 2>/dev/null)" != "$cur" ]; then
      echo "[start_evaluation] installing requirements (this may take a moment)"
      "$VENV_DIR/bin/python" -m pip install -r "$REQ_FILE" || return 1
      echo "$cur" >"$stamp"
    else
      echo "[start_evaluation] requirements unchanged; skipping install"
    fi
  fi
  return 0
}

if ! _se_build; then
  echo "[start_evaluation] FAILED to set up venv" >&2
  if [ "$_sa_sourced" = "1" ]; then return 1; else exit 1; fi
fi

export PYTHONPATH="$LOADER_DIR:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# --- activate / stay inside -------------------------------------------------
if [ "$_sa_sourced" = "1" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  echo "[start_evaluation] activated in current shell: $(python --version 2>&1)"
  echo "[start_evaluation] loader importable: 'from load_handpose_log import load_log'"
else
  # Launch an interactive shell that sources the user's rc, then activates the
  # venv (so PATH/prompt are correct), then self-deletes its temp rcfile. This
  # keeps you "inside" the venv until you 'exit'.
  _rc="$(mktemp)"
  {
    echo "[ -f ~/.bashrc ] && source ~/.bashrc"
    echo "source '$VENV_DIR/bin/activate'"
    echo "export PYTHONPATH='$PYTHONPATH'"
    echo "rm -f '$_rc'"
    echo "echo '[start_evaluation] venv shell ready ('\"\$(python --version 2>&1)\"'). Type exit to leave.'"
    echo "echo '[start_evaluation] log loader importable: from load_handpose_log import load_log'"
  } >"$_rc"
  exec "${SHELL:-/bin/bash}" --rcfile "$_rc" -i
fi
