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
#
# It also loads the project's .env, so the R2 credentials live in exactly one
# place. See the block below for why DVC cannot pick that file up on its own.
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

# Load .env into the environment, without overriding what is already there.
#
# DVC is a separate process that never imports `src/`, so the `load_dotenv`
# calls in src/ingest.py and src/pipeline.py do nothing for it — boto3 reads the
# real process environment. Without this block the R2 keys sit in a filled-in
# .env and `dvc pull` still fails on credentials, which is confusing precisely
# because every other entry point honours the file.
#
# Precedence is deliberate: an already-exported variable wins. CI injects the
# keys as real environment variables and asserts no credential file is written
# (.github/workflows/ci.yml), and has no .env at all, so this block is inert
# there. Values are never echoed.
ENV_FILE="$PROJECT_ROOT/.env"
if [[ -f "$ENV_FILE" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Tolerate CRLF: .env is gitignored, so a Windows editor may write it.
        line="${line%$'\r'}"
        # Skips comments, blank lines, and placeholder keys with no value —
        # the trailing \S in the value group requires at least one non-space.
        if [[ ! "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*[^[:space:]])[[:space:]]*$ ]]; then
            continue
        fi
        key="${BASH_REMATCH[1]}"
        value="${BASH_REMATCH[2]}"
        if [[ -n "${!key:-}" ]]; then
            continue
        fi
        value="${value#[\"\']}"
        value="${value%[\"\']}"
        export "$key=$value"
    done < "$ENV_FILE"
fi

export PATH="$VENV_BIN:$PATH"
exec "$VENV_PYTHON" -m dvc "$@"
