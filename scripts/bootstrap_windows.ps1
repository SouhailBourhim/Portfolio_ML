<#
.SYNOPSIS
    Create and populate the project virtual environment on Windows.

.DESCRIPTION
    The Windows form of README.md's "Local install". It exists because two steps
    of the macOS procedure have no Windows equivalent and one of them fails
    outright:

    1. LAYOUT. A virtualenv is `.venv/bin/python` on macOS and Linux but
       `.venv\Scripts\python.exe` on Windows, so `source .venv/bin/activate`
       does not exist here.

    2. uvloop. `requirements.lock.txt` was frozen on macOS and contains
       distributions that DO NOT BUILD OR INSTALL ON WINDOWS -- uvloop most
       importantly, which has no Windows support at all. A plain
       `pip install -r requirements.lock.txt` therefore dies partway through,
       leaving a half-populated environment.

       The lock is NOT edited to fix this. It is SHA-256'd by src/snapshot.py
       and is an input to the release manifest, so changing one byte of it
       invalidates every published snapshot. Instead this script filters the
       POSIX-only distributions out into a temporary file and installs that.
       The set actually installed differs from the lock only by packages that
       cannot run on this platform in the first place.

    3. UTF-8. Windows still defaults `open()` to cp1252, while every artifact
       and document in this repository is UTF-8 with French text in it. The
       source now passes `encoding="utf-8"` explicitly everywhere, so this is
       belt and braces -- but Jupyter notebooks and any ad-hoc snippet benefit,
       so PYTHONUTF8=1 is written into the generated activation scripts.

.PARAMETER PythonVersion
    Interpreter to build the environment with. Defaults to 3.11, which is what
    the Dockerfile, the CI workflow and requirements.lock.txt all pin. Pass
    another (e.g. 3.12) only knowingly: the pinned scientific stack is what the
    committed results were produced under.

.PARAMETER Recreate
    Delete an existing .venv first.

.PARAMETER SkipTests
    Do not run the test suite at the end.

.EXAMPLE
    .\scripts\bootstrap_windows.ps1
    .\scripts\bootstrap_windows.ps1 -PythonVersion 3.12 -Recreate
#>
[CmdletBinding()]
param(
    [string] $PythonVersion = '3.11',
    [switch] $Recreate,
    [switch] $SkipTests
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

# -- 1. Interpreter ----------------------------------------------------------
$available = (& py -0p) -join "`n"
if ($available -notmatch [regex]::Escape("-V:$PythonVersion")) {
    Write-Host "Python $PythonVersion is not installed. Available interpreters:"
    & py -0p
    Write-Error @"
Install it first:

    winget install --id Python.Python.$($PythonVersion -replace '\.','.')

or pass -PythonVersion with one of the versions listed above. 3.11 is what the
Dockerfile, .github/workflows/ci.yml and requirements.lock.txt all pin.
"@
}

$VenvDir = Join-Path $Root '.venv'
if ($Recreate -and (Test-Path -LiteralPath $VenvDir)) {
    Write-Host "Removing existing $VenvDir ..."
    Remove-Item -LiteralPath $VenvDir -Recurse -Force
}

if (-not (Test-Path -LiteralPath $VenvDir)) {
    Write-Host "Creating virtual environment with Python $PythonVersion ..."
    & py "-$PythonVersion" -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Write-Error "venv creation failed." }
}

$Python = Join-Path $VenvDir 'Scripts\python.exe'
Write-Host ("Interpreter: {0}" -f (& $Python --version))

# -- 2. UTF-8 by default inside this environment -----------------------------
# .venv/ is gitignored, so these generated files are the right place for a
# machine-local default. Idempotent: re-running does not duplicate the line.
foreach ($pair in @(
    @{ File = 'Activate.ps1'; Line = '$env:PYTHONUTF8 = "1"' },
    @{ File = 'activate.bat'; Line = 'set "PYTHONUTF8=1"' }
)) {
    $target = Join-Path $VenvDir "Scripts\$($pair.File)"
    if ((Test-Path -LiteralPath $target) -and
        -not (Select-String -LiteralPath $target -SimpleMatch 'PYTHONUTF8' -Quiet)) {
        Add-Content -LiteralPath $target -Value ''
        Add-Content -LiteralPath $target -Value '# Portfolio ML: UTF-8 I/O regardless of the Windows ANSI codepage.'
        Add-Content -LiteralPath $target -Value $pair.Line
    }
}

# -- 3. Filter the lock, then install it -------------------------------------
# Distributions in requirements.lock.txt that cannot be installed or have no
# purpose on Windows. Each is listed with the reason, because a silent exclusion
# list rots into superstition.
$PosixOnly = @{
    'uvloop'   = 'no Windows support whatsoever; the install aborts on the sdist build'
    'appnope'  = 'macOS App Nap shim; a no-op import elsewhere'
    'pexpect'  = 'POSIX pty control, pulled in by IPython; unused on Windows'
    'ptyprocess' = 'pexpect dependency, same reason'
}

$lockPath = Join-Path $Root 'requirements.lock.txt'
$lock = Get-Content -LiteralPath $lockPath
$filtered = @()
$dropped = @()
foreach ($line in $lock) {
    $name = ($line -split '==')[0].Trim()
    $key = $PosixOnly.Keys | Where-Object { $_ -eq $name }
    if ($key -and $line -match '==') {
        $dropped += "  {0,-12} {1}" -f $name, $PosixOnly[$key]
    } else {
        $filtered += $line
    }
}

$tempReq = Join-Path ([System.IO.Path]::GetTempPath()) 'portfolio_ml_requirements.win.txt'
Set-Content -LiteralPath $tempReq -Value $filtered -Encoding utf8

if ($dropped) {
    Write-Host ""
    Write-Host "Excluded from the install (POSIX-only; requirements.lock.txt itself is untouched):"
    $dropped | ForEach-Object { Write-Host $_ }
    Write-Host ""
}

Write-Host "Installing the pinned runtime (this takes a few minutes) ..."
& $Python -m pip install --upgrade pip --quiet
& $Python -m pip install -r $tempReq
if ($LASTEXITCODE -ne 0) {
    Write-Error "pip install failed -- see the output above."
}

Remove-Item -LiteralPath $tempReq -Force -ErrorAction SilentlyContinue

# -- 4. Prove it -------------------------------------------------------------
Write-Host ""
Write-Host "Import check ..."
& $Python -c @"
# `import importlib.util`, not `import importlib`: the parent package does not
# bind the `util` submodule as an attribute, so the short form raises
# AttributeError on the first find_spec call.
import importlib.util, sys
mods = ['pandas','numpy','pyarrow','yaml','yfinance','fredapi','pandera','duckdb',
        'statsmodels','scipy','sklearn','hmmlearn','arch','xgboost','mlflow','dvc',
        'dagster','matplotlib','seaborn','plotly','streamlit','fastapi','uvicorn',
        'httpx','pptx','docx','cryptography','BVCscrap','lxml','dotenv','requests']
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print('python', sys.version.split()[0])
if missing:
    print('MISSING:', ', '.join(missing)); sys.exit(1)
print(f'all {len(mods)} top-level imports resolve')
"@
if ($LASTEXITCODE -ne 0) { Write-Error "Import check failed." }

if (-not $SkipTests) {
    Write-Host ""
    Write-Host "Running the offline test suite ..."
    & $Python -m pytest tests/ -q
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "The test suite reported failures -- see the output above."
    }
}

Write-Host ""
Write-Host "Done. Activate the environment with:"
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Next steps are in docs\WINDOWS_SETUP.md."
