<#
.SYNOPSIS
    Run DVC with this repository's virtual environment on PATH (Windows).

.DESCRIPTION
    The PowerShell twin of scripts/dvc.sh, and the supported DVC entry point on
    Windows where Git Bash may not be installed.

    Calling .venv\Scripts\dvc.exe directly is not sufficient: DVC executes the
    commands declared in dvc.yaml through a shell, where `python` would
    otherwise resolve to the system interpreter -- producing a misleading
    `ModuleNotFoundError: numpy` on an otherwise fully installed project. Stage
    commands stay portable (`python src/...`) and this wrapper puts the right
    interpreter first on PATH.

    It also loads the project's .env, so the R2 credentials live in exactly one
    place. DVC cannot pick that file up on its own -- see the block below.

.EXAMPLE
    .\scripts\dvc.ps1 status
    .\scripts\dvc.ps1 pull
    .\scripts\dvc.ps1 repro --single-item snapshot_manifest
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $DvcArgs
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvBin = Join-Path $ProjectRoot '.venv\Scripts'
$VenvPython = Join-Path $VenvBin 'python.exe'

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Error @"
No interpreter at $VenvPython -- create the project virtual environment first:

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.lock.txt

See docs\WINDOWS_SETUP.md for the full procedure.
"@
}

# Load .env into the environment, without overriding what is already there.
#
# DVC is a separate process that never imports src\, so the load_dotenv calls in
# src\ingest.py and src\pipeline.py do nothing for it -- boto3 reads the real
# process environment. Without this block the R2 keys sit in a filled-in .env
# and `dvc pull` still fails on credentials, which is confusing precisely
# because every other entry point honours the file.
#
# Precedence is deliberate: an already-set variable wins. CI injects the keys as
# real environment variables and asserts no credential file is written
# (.github\workflows\ci.yml), and has no .env at all, so this block is inert
# there. Values are never echoed.
$EnvFile = Join-Path $ProjectRoot '.env'
if (Test-Path -LiteralPath $EnvFile) {
    foreach ($line in Get-Content -LiteralPath $EnvFile) {
        # Skips comments, blank lines, and placeholder keys with no value --
        # the trailing \S in the value group requires at least one non-space.
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*\S)\s*$') {
            $key = $matches[1]
            if ([Environment]::GetEnvironmentVariable($key)) { continue }
            Set-Item -Path "env:$key" -Value $matches[2].Trim('"').Trim("'")
        }
    }
}

$env:PATH = "$VenvBin;$env:PATH"

& $VenvPython -m dvc @DvcArgs
exit $LASTEXITCODE
