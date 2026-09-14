<#
.SYNOPSIS
    Every check that must pass before a release tag, as one command (Windows).

.DESCRIPTION
    The PowerShell twin of scripts/release_gates.sh. Same six gates, same order,
    same exit semantics -- it exists so the release procedure is executable on
    Windows without Git Bash.

    Addresses: P4 -- until now the tag procedure lived in commit messages and in
    somebody's head. A procedure you have to remember is one you eventually
    skip, and the gates most worth running are exactly the ones that only fail
    when something has quietly drifted.

.PARAMETER SkipTests
    Run the data-dependent gates only (~30 s) and skip the suite (~6 min). CI
    passes this because the suite already runs in its own job.

.EXAMPLE
    .\scripts\release_gates.ps1
    .\scripts\release_gates.ps1 -SkipTests
#>
[CmdletBinding()]
param([switch] $SkipTests)

# NOT $ErrorActionPreference = 'Stop': gates are expected to fail, and each one
# must be reported by name rather than aborting the script at the first native
# non-zero exit.
$ErrorActionPreference = 'Continue'

$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { $Python = 'python' }   # CI / ambient env

$script:Failed = $false

function Invoke-Gate {
    param([string] $Name, [scriptblock] $Body)
    Write-Host ""
    Write-Host ("-- {0}" -f $Name)

    # A script block emits EVERYTHING its body writes to the success stream, not
    # just the value it returns. A gate that shells out to a native command
    # therefore yields [pytest output..., $false], and `if ($ok)` on a non-empty
    # array is truthy -- so a FAILING gate reported PASS. Verified: a body that
    # prints one line and returns $false emits [String, Boolean] and passed.
    #
    # Two defences, because either alone is fragile:
    #   - every gate below routes native output to the host with `| Out-Host`,
    #     so only its Boolean reaches the success stream;
    #   - this takes the LAST emitted object and coerces it explicitly, so a
    #     gate that forgets that still cannot pass on stray output.
    # The same failure this guards against is the one gate 5 exists to catch:
    # a check that reports success because it never really ran.
    $ok = [bool](@(& $Body) | Select-Object -Last 1)

    if ($ok) {
        Write-Host "   PASS"
    } else {
        Write-Host ("   FAIL -- {0}" -f $Name)
        $script:Failed = $true
    }
}

# 1. Data matches dvc.lock. Everything downstream assumes this, so it goes
#    first: a stale artifact makes every later gate meaningless rather than
#    failing honestly.
#
#    FROZEN-STAGE WARNINGS. `dvc status` prints one
#    "WARNING: stage: '<name>' is frozen." line per frozen stage, on stderr,
#    even when everything is up to date. The global_2004 stages are frozen on
#    purpose -- they are run-once evidence that must never be rebuilt -- so
#    those lines are EXPECTED output, not a defect. They are echoed for the
#    reader but stripped before the equality check, which otherwise can never
#    pass once any stage is frozen.
function Test-DvcStatus {
    $out = & $Python -m dvc status 2>&1 | ForEach-Object { $_.ToString() }
    if ($LASTEXITCODE -ne 0) { $out | Write-Host; return $false }
    $out | Write-Host
    $checked = $out | Where-Object { $_ -notmatch "^WARNING: stage: '.*' is frozen\." }
    $joined = ($checked -join "`n").Trim()
    return $joined -eq 'Data and pipelines are up to date.'
}

# 2. The manifest identifies the code that produced these artifacts, was
#    written from a clean tree, and every checksum still matches.
function Test-Snapshot {
    & $Python src/snapshot.py verify | Out-Host
    return $LASTEXITCODE -eq 0
}

# 3. The serving bundle is complete AND verified.
function Test-Bundle {
    & $Python scripts/check_artifacts.py --verify | Out-Host
    return $LASTEXITCODE -eq 0
}

# 4. Regenerating the model cards is a no-op.
#
#    ORDERING NOTE, learned the hard way: the cards embed the manifest's
#    git_commit, so a manifest regenerated after the cards were built leaves
#    them stale. The sequence is manifest -> cards -> commit both. This gate
#    only reports; fixing it by regenerating here would hide the drift it
#    exists to surface.
function Test-ModelCards {
    & $Python scripts/build_model_cards.py | Out-Null
    if ($LASTEXITCODE -ne 0) { return $false }
    git diff --quiet -- docs/MODEL_CARD_*.md
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Model cards changed when regenerated -- they are stale."
        git --no-pager diff --stat -- docs/MODEL_CARD_*.md | Out-Host
        Write-Host ""
        Write-Host "Fix: regenerate the snapshot manifest FIRST, then the cards, then"
        Write-Host "commit both:"
        Write-Host "  .\scripts\dvc.ps1 repro --single-item --force snapshot_manifest"
        Write-Host "  $Python scripts\build_model_cards.py"
        return $false
    }
    Write-Host "regeneration is a no-op"
    return $true
}

# 5. The working tree is clean. A tag on a dirty tree names a revision that
#    does not contain what was tested.
function Test-CleanTree {
    # Check git's EXIT CODE, not just whether output was empty. Outside a
    # repository `git status --porcelain` writes to stderr and prints nothing to
    # stdout, so an emptiness test alone reports "working tree clean" for a
    # checkout with no history at all -- a gate passing because it could not run,
    # which is the precise failure these gates exist to catch.
    $dirty = git status --porcelain 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Not a git repository -- this gate cannot answer whether the"
        Write-Host "tree matches a commit, so it fails rather than passing blindly."
        Write-Host "Restore the history (git clone, or git init + remote) before tagging."
        return $false
    }
    # A repository created by `git init` with no commits also returns success
    # and empty output here, so the exit-code check alone still reports a clean
    # tree against a revision that does not exist. There must be a HEAD to tag.
    git rev-parse --verify HEAD 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Repository has no commits -- there is no revision to tag, so"
        Write-Host "this gate cannot answer whether the tree matches one."
        return $false
    }
    if ($dirty) {
        Write-Host "Uncommitted changes:"
        $dirty | Write-Host
        return $false
    }
    $sha = git rev-parse --short HEAD 2>$null
    Write-Host "working tree clean at $sha"
    return $true
}

function Test-Suite {
    & $Python -m pytest tests/ -q | Out-Host
    return $LASTEXITCODE -eq 0
}

$headSha = git rev-parse --short HEAD 2>$null
if ($LASTEXITCODE -ne 0) { $headSha = '(not a git repository)' }
Write-Host ("Release gates -- {0}" -f $headSha)

Invoke-Gate '1/6  dvc status'                 { Test-DvcStatus }
Invoke-Gate '2/6  snapshot verification'      { Test-Snapshot }
Invoke-Gate '3/6  artifact bundle (--verify)' { Test-Bundle }
Invoke-Gate '4/6  model cards are current'    { Test-ModelCards }
Invoke-Gate '5/6  working tree is clean'      { Test-CleanTree }

if ($SkipTests) {
    Write-Host ""
    Write-Host "-- 6/6  test suite"
    Write-Host "   SKIPPED (-SkipTests)"
} else {
    Invoke-Gate '6/6  test suite'             { Test-Suite }
}

Write-Host ""
if (-not $script:Failed) {
    Write-Host "All release gates passed. Safe to tag."
    exit 0
}
Write-Host "RELEASE GATES FAILED -- do not tag."
exit 1
