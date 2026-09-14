#!/usr/bin/env bash
# Run DVC with this repository's virtual environment on PATH.
#
# Calling `.venv/bin/dvc` directly is not sufficient: DVC executes the commands
# declared in dvc.yaml through the shell, where `python` would otherwise resolve
# to the system interpreter. That produced a misleading `ModuleNotFoundError:
# numpy` on an otherwise installed project. Keep stage commands portable (`python
# src/...`) and use this wrapper as the supported entry point.
#
# CROSS-PLATFORM. CPython lays out a virtualenv differently per OS:
# `.venv/bin/python` on macOS and Linux, `.venv/Scripts/python.exe` on Windows.
# Hardcoding the POSIX layout made this wrapper — and therefore every documented
# `dvc` invocation — unusable from Git Bash on Windows. Both layouts are probed
# here. Windows users can also run `scripts\dvc.ps1`, which needs no Bash at all.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    VENV_BIN="$PROJECT_ROOT/.venv/bin"
    VENV_PYTHON="$VENV_BIN/python"
elif [[ -x "$PROJECT_ROOT/.venv/Scripts/python.exe" ]]; then
    VENV_BIN="$PROJECT_ROOT/.venv/Scripts"
    VENV_PYTHON="$VENV_BIN/python.exe"
else
    echo "error: no interpreter at $PROJECT_ROOT/.venv/bin/python or" >&2
    echo "       $PROJECT_ROOT/.venv/Scripts/python.exe — create the project" >&2
    echo "       virtual environment first (see README.md, 'Local install')." >&2
    exit 1
fi

export PATH="$VENV_BIN:$PATH"
exec "$VENV_PYTHON" -m dvc "$@"
