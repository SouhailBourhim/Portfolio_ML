#!/usr/bin/env bash
# Run DVC with this repository's virtual environment on PATH.
#
# Calling `.venv/bin/dvc` directly is not sufficient: DVC executes the commands
# declared in dvc.yaml through the shell, where `python` would otherwise resolve
# to the system interpreter. That produced a misleading `ModuleNotFoundError:
# numpy` on an otherwise installed project. Keep stage commands portable (`python
# src/...`) and use this wrapper as the supported entry point.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="$PROJECT_ROOT/.venv/bin"

if [[ ! -x "$VENV_BIN/python" ]]; then
    echo "error: $VENV_BIN/python not found — create the project virtual environment first." >&2
    exit 1
fi

export PATH="$VENV_BIN:$PATH"
exec "$VENV_BIN/python" -m dvc "$@"
