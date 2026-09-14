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

$env:PATH = "$VenvBin;$env:PATH"

& $VenvPython -m dvc @DvcArgs
exit $LASTEXITCODE
