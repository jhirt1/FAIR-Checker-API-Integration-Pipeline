#!/usr/bin/env bash
set -euo pipefail

# Run the FAIR-Checker API Integration Pipeline and log output.
# Uses the project's virtual environment created by setup.sh.
# On macOS, uses caffeinate to prevent idle sleep while running.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

VENV_DIR="$REPO_ROOT/.venv"
PY_SCRIPT="$REPO_ROOT/data_collection.py"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_pipeline.log"

# Ensure venv exists
if [ ! -d "$VENV_DIR" ]; then
  echo "Virtual environment not found at: $VENV_DIR" >&2
  echo "Run ./setup.sh first." >&2
  exit 1
fi

# Activate venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Ensure script exists
if [ ! -f "$PY_SCRIPT" ]; then
  echo "Python script not found: $PY_SCRIPT" >&2
  exit 1
fi

echo "Starting pipeline at $(date)" | tee -a "$LOG_FILE"
echo "Python: $(python --version 2>&1)" | tee -a "$LOG_FILE"
echo "Script: $PY_SCRIPT" | tee -a "$LOG_FILE"
echo "Log:    $LOG_FILE" | tee -a "$LOG_FILE"
echo "Args:   $*" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# Prefer caffeinate on macOS if available; otherwise run normally.
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i -- python "$PY_SCRIPT" "$@" 2>&1 | tee -a "$LOG_FILE"
  PY_EXIT=${PIPESTATUS[0]}
else
  echo "caffeinate not found; running without sleep prevention." | tee -a "$LOG_FILE"
  python "$PY_SCRIPT" "$@" 2>&1 | tee -a "$LOG_FILE"
  PY_EXIT=${PIPESTATUS[0]}
fi

echo "----------------------------------------" | tee -a "$LOG_FILE"
echo "Finished pipeline at $(date) (exit code: $PY_EXIT)" | tee -a "$LOG_FILE"
exit "$PY_EXIT"