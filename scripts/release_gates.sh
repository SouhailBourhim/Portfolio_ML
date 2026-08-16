#!/usr/bin/env bash
# Every check that must pass before a release tag, as one command.
#
# Addresses: P4 — until now the tag procedure lived in commit messages and in
# somebody's head. A procedure you have to remember is one you eventually skip,
# and the gates most worth running are exactly the ones that only fail when
# something has quietly drifted.
#
#   ./scripts/release_gates.sh                # everything (~6 min, runs the suite)
#   ./scripts/release_gates.sh --skip-tests   # data-dependent gates only (~30 s)
#
# Exits non-zero on the FIRST failure, naming the gate. CI calls this with
# --skip-tests because the suite already runs in its own job.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python"          # CI installs into the job's env
DVC="$PYTHON -m dvc"

SKIP_TESTS=0
[ "${1:-}" = "--skip-tests" ] && SKIP_TESTS=1

FAILED=0
run_gate() {
    local name="$1"; shift
    printf '\n── %s\n' "$name"
    if "$@"; then
        printf '   PASS\n'
    else
        printf '   FAIL — %s\n' "$name"
        FAILED=1
        return 1
    fi
}

# 1. Data matches dvc.lock. Everything downstream assumes this, so it goes
#    first: a stale artifact makes every later gate meaningless rather than
#    failing honestly.
#    FROZEN-STAGE WARNINGS. `dvc status` prints one
#    "WARNING: stage: '<name>' is frozen." line per frozen stage, on stderr,
#    even when everything is up to date. The global_2004 stages are frozen on
#    purpose — they are run-once evidence that must never be rebuilt — so those
#    lines are EXPECTED output, not a defect. They are echoed for the reader
#    but stripped before the equality check, which otherwise can never pass
#    once any stage is frozen. (That is exactly how this gate broke on the
#    2026-08-16 merge to main.)
gate_dvc_status() {
    local out checked
    out="$($DVC status 2>&1)" || { echo "$out"; return 1; }
    echo "$out"
    checked="$(printf '%s\n' "$out" \
        | grep -v "^WARNING: stage: '.*' is frozen\." || true)"
    [ "$checked" = "Data and pipelines are up to date." ]
}

# 2. The manifest identifies the code that produced these artifacts, was
#    written from a clean tree, and every checksum still matches.
gate_snapshot() { $PYTHON src/snapshot.py verify; }

# 3. The serving bundle is complete AND verified.
gate_bundle() { $PYTHON scripts/check_artifacts.py --verify; }

# 4. Regenerating the model cards is a no-op.
#
#    ORDERING NOTE, learned the hard way: the cards embed the manifest's
#    git_commit, so a manifest regenerated after the cards were built leaves
#    them stale. The sequence is manifest -> cards -> commit both. This gate
#    only reports; fixing it by regenerating here would hide the drift it
#    exists to surface.
gate_model_cards() {
    $PYTHON scripts/build_model_cards.py >/dev/null || return 1
    if ! git diff --quiet -- docs/MODEL_CARD_*.md; then
        echo "Model cards changed when regenerated — they are stale."
        git --no-pager diff --stat -- docs/MODEL_CARD_*.md
        echo
        echo "Fix: regenerate the snapshot manifest FIRST, then the cards, then"
        echo "commit both:"
        echo "  ./scripts/dvc.sh repro --single-item --force snapshot_manifest"
        echo "  ./.venv/bin/python scripts/build_model_cards.py"
        return 1
    fi
    echo "regeneration is a no-op"
}

# 5. The working tree is clean. A tag on a dirty tree names a revision that
#    does not contain what was tested.
gate_clean_tree() {
    local dirty
    dirty="$(git status --porcelain)"
    if [ -n "$dirty" ]; then
        echo "Uncommitted changes:"; echo "$dirty"
        return 1
    fi
    echo "working tree clean at $(git rev-parse --short HEAD)"
}

gate_tests() { $PYTHON -m pytest tests/ -q; }

echo "Release gates — $(git rev-parse --short HEAD)"

run_gate "1/6  dvc status"                  gate_dvc_status
run_gate "2/6  snapshot verification"       gate_snapshot
run_gate "3/6  artifact bundle (--verify)"  gate_bundle
run_gate "4/6  model cards are current"     gate_model_cards
run_gate "5/6  working tree is clean"       gate_clean_tree

if [ "$SKIP_TESTS" -eq 1 ]; then
    printf '\n── 6/6  test suite\n   SKIPPED (--skip-tests)\n'
else
    run_gate "6/6  test suite"              gate_tests
fi

printf '\n'
if [ "$FAILED" -eq 0 ]; then
    echo "All release gates passed. Safe to tag."
    exit 0
fi
echo "RELEASE GATES FAILED — do not tag."
exit 1
