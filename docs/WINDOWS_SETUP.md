# Running this project on Windows

This repository was developed on macOS. Everything in it runs on Windows, but
five things had to change for that to be true, and most of them fail in ways
that do not look like a portability problem. This page is the Windows form of
README.md's *Reproduce and verify* section, plus the reasoning behind each
difference.

> **Nothing here weakens the reproducibility contract.** `requirements.lock.txt`,
> `params.yaml`, `dvc.yaml` and every Gold artifact are byte-identical to what
> macOS and CI produce. The porting work was to keep them that way, not to
> relax them.

**Verified on Windows 11, Python 3.11.9, 2026-09-14:**

| Surface | Result |
|---|---|
| Test suite | **760 passed, 131 skipped, 0 failed** (2m28s) — the skips are the data-dependent tests |
| Smoke pipeline (CI gate) | passes — 7 strategies on synthetic data in 20s |
| Dependency install | all 331 packages of the filtered lock |
| Import check | all 31 top-level imports resolve |
| Artifact preflight | correctly *refuses* an incomplete bundle |
| FastAPI app | imports, 15 routes registered |
| Streamlit data layer | imports |
| MLflow SQLite backend | logs and closes a run against a backslash Windows path |
| `dvc.ps1` wrapper | resolves DVC 3.67.1 through the venv |
| Git | 236 commits + release tags adopted in place; **0 files lost** in the transfer |
| DVC pipeline | resolves all 28 stages; reports changes only for the files actually edited |
| `dvc pull` | reaches the R2 remote and stops at credentials alone (52 files pending) |
| Release gates | all six execute; gate 5 (clean tree) **passes**, 1–4 fail only for want of the data bundle |

---

## 1. What actually differed, and why

### 1.1 Line endings — the one that silently breaks the release gates

Git on Windows defaults to `core.autocrlf=true`, which rewrites every text file
to CRLF on checkout. For most projects that is cosmetic. Here it is not:

| Breaks | How |
|---|---|
| `src/snapshot.py verify` | The manifest SHA-256s `params.yaml`, `dvc.yaml`, `requirements.txt` and `requirements.lock.txt`. CRLF changes those bytes, so verification fails against any manifest produced elsewhere — **release gate 2**. |
| `dvc status` | DVC hashes `params.yaml` as a stage dependency. CRLF reports every stage as changed, inviting a pointless full `dvc repro` — **release gate 1**. |
| `scripts/*.sh` | A shell script with CRLF fails under Git Bash and in the Linux container with `$'\r': command not found`. |

`.gitattributes` now pins `eol=lf` for all text and marks the binary types, so a
Windows working tree is byte-identical to the macOS and CI ones. The working
copy in this checkout was converted back to LF in place.

If you clone fresh and see the whole tree show up as modified, the cause is a
stale `core.autocrlf`. Fix it once:

```powershell
git config --global core.autocrlf false
git rm --cached -r .
git reset --hard
```

### 1.2 Text encoding — cp1252 versus UTF-8

Windows still defaults `open()`, `Path.read_text()` and `Path.write_text()` to
the ANSI codepage (cp1252 here), while every artifact, document and docstring in
this repository is UTF-8 with French text in it. That produces two distinct
failures, and the quiet one is worse:

- **Loud:** writing `→`, `─` or `≥` raises `UnicodeEncodeError`.
- **Quiet:** reading UTF-8 as cp1252 usually *succeeds* and yields mojibake —
  `numéraire` becomes `numÃ©raire` — which then flows into a JSON artifact, a
  model card or a report table with nothing raising at all.

Every text-mode file call in `src/`, `scripts/`, `tests/`, `experiments/` and
`dashboard/` now passes `encoding="utf-8"` explicitly (114 call sites). This is
the real fix: it is platform-independent and needs no environment variable.

`scripts/bootstrap_windows.ps1` additionally writes `PYTHONUTF8=1` into the
generated venv activation scripts, which covers Jupyter and any ad-hoc snippet
typed at a prompt. Belt and braces, not the primary defence.

### 1.3 `requirements.lock.txt` contains packages that cannot install here

The lock was frozen on macOS and pins **uvloop**, which has no Windows support
at all — `pip install -r requirements.lock.txt` aborts on its sdist build and
leaves a half-populated environment. `appnope`, `pexpect` and `ptyprocess` are
POSIX-only in purpose if not in packaging.

**The lock is not edited to fix this.** It is SHA-256'd by `src/snapshot.py` and
is an input to the release manifest; changing one byte invalidates every
published snapshot. Instead `scripts/bootstrap_windows.ps1` filters those four
distributions into a temporary requirements file and installs that. The
environment differs from the lock only by packages that cannot run on this
platform in the first place.

### 1.4 Virtualenv layout and shell scripts

`.venv/bin/python` on macOS is `.venv\Scripts\python.exe` on Windows, and
`source .venv/bin/activate` has no equivalent. Every entry point that hardcoded
the POSIX layout now probes both, and the ones you are most likely to run have
native PowerShell twins:

| macOS / Linux | Windows |
|---|---|
| `./scripts/dvc.sh …` | `.\scripts\dvc.ps1 …` |
| `./scripts/release_gates.sh` | `.\scripts\release_gates.ps1` |
| `./scripts/setup_launchd.sh` | `.\scripts\setup_dagster_tasks.ps1` |
| `source .venv/bin/activate` | `.\.venv\Scripts\Activate.ps1` |

`scripts/setup_launchd.sh` now refuses to run off macOS rather than failing
halfway through after writing files into `$HOME`.

### 1.5 The `.ps1` files are deliberately pure ASCII

Windows PowerShell 5.1 — the `powershell.exe` shipped with Windows, and what you
get unless you install PowerShell 7 — decodes a `.ps1` file using the ANSI
codepage unless the file carries a BOM. A UTF-8 em dash (`E2 80 94`) therefore
arrives as three cp1252 characters, one of which is a **smart double quote** that
PowerShell treats as a real string delimiter.

Inside a comment that is survivable. Inside a double-quoted string it truncates
the string and the entire file stops parsing, with an error pointing at a line
40 lines below the actual cause:

```text
The string is missing the terminator: ".
Missing closing '}' in statement block or type definition.
```

This is not hypothetical — it happened while writing `release_gates.ps1`. So
these four scripts contain no character above U+007F, which parses identically
under PowerShell 5.1 and 7, with or without a BOM. **If you edit them, keep them
ASCII**: write `--`, not `—`. The `.gitattributes` rule `*.ps1 text eol=crlf`
handles line endings but cannot help with encoding.

---

## 2. Git — and the setting that caused the CRLF damage

The copy on this machine arrived without its `.git` directory, which blocked
more than it looked like it did: **DVC is built on Git and will not run at all
without it**, exiting with `ERROR: <path> is not a git repository`. No `status`,
no `pull`, no `repro` — therefore no data bundle.

This has been set up. The history was adopted **in place**, which keeps the
ported working tree rather than forcing a clone-and-move:

```powershell
git init -b main
git config core.autocrlf false      # see below — this is the important one
git config core.longpaths true      # docs/rapport_final/assets/ nests deeply
git remote add origin https://github.com/SouhailBourhim/Portfolio_ML.git
git fetch origin --tags
git reset --mixed origin/main       # adopts history; does NOT touch the working tree
```

`git reset --mixed` is what makes this safe: it moves `HEAD` and the index to
`origin/main` and leaves every working-tree file exactly as it was. The result
is 236 commits of real history — including the release tags that
`src/snapshot.py`'s ancestry check needs — under an unchanged working tree.

### The root cause of the line-ending corruption

`core.autocrlf` was **`true` at the system level** on this machine, which is the
Git for Windows installer default. That is what rewrote all 245 text files to
CRLF during the transfer, and it would do it again on every future checkout.

It is now `false` in this repository's local config, which overrides the system
value. `.gitattributes` (§1.1) is the durable fix and applies even before it is
committed — Git reads it from the working tree — but the local config setting is
deliberate belt-and-braces, because a future `git clone` elsewhere gets the
system default again until `.gitattributes` lands in the index.

**Verification that the LF restoration was byte-perfect:** against `origin/main`
the working tree shows *348 insertions and 144 deletions across 76 files*, and
no file is rewritten wholesale. Had any CRLF survived, every line of every
affected file would appear as changed — tens of thousands of lines. Nothing was
lost in the transfer either: `git status` reports **zero deleted files**.

### Git identity is not configured

`user.name` and `user.email` are unset at every scope, so `git commit` will
refuse. Nothing above needed them, and nothing has been committed. Set them when
you are ready to commit the port:

```powershell
git config user.name  "Your Name"
git config user.email "you@example.com"
```

---

## 3. Install

Requires **Python 3.11** — what the Dockerfile, `.github/workflows/ci.yml` and
`requirements.lock.txt` all pin. `py -0p` lists what you have.

```powershell
winget install --id Python.Python.3.11
```

Then, from the project root:

```powershell
.\scripts\bootstrap_windows.ps1
```

That creates `.venv`, installs the filtered lock, checks that all 31 top-level
imports resolve, and runs the offline test suite. Roughly five minutes.

If PowerShell refuses to run the script, allow signed-and-local scripts for your
user once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Doing it by hand instead:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.lock.txt   # will FAIL on uvloop — see 1.3
pytest -q
```

Use the bootstrap script; the manual path is shown only so the difference from
README.md is explicit.

---

## 4. Daily commands

With `.\.venv\Scripts\Activate.ps1` sourced, everything in README.md works
unchanged — `pytest -q`, `python src/run_phase5.py`, `streamlit run …`.

The other pages under `docs/` spell their commands `./.venv/bin/python <script>`
because they were written on macOS. With the environment activated, every one of
those is just `python <script>` here — for example
`./.venv/bin/python experiments/dividend_bias.py` becomes
`python experiments\dividend_bias.py`. Only the commands below need a different
file, not merely a different prefix:

```powershell
# Test suite (offline, needs no data)
pytest -q

# Research API — http://127.0.0.1:8000/docs
uvicorn api.main:app --app-dir src

# Streamlit dashboard — needs the DVC bundle present
streamlit run dashboard/streamlit_app.py

# DVC (use the wrapper, not dvc.exe directly — see scripts/dvc.ps1 for why)
.\scripts\dvc.ps1 status
.\scripts\dvc.ps1 pull

# Snapshot verification
python src/snapshot.py verify

# Release gates
.\scripts\release_gates.ps1
.\scripts\release_gates.ps1 -SkipTests

# Dagster scheduling across logons (the launchd equivalent)
.\scripts\setup_dagster_tasks.ps1
```

---

## 5. Data

`data/` is DVC-managed, licence-restricted and not in Git, so a fresh Windows
checkout has none of it. The offline test suite, the smoke pipeline and the API
contract tests all pass without it; the artifact-consistency, release-facts and
model-integrity tests **skip**, and the dashboard and API refuse to start.

**The S3 backend is a separate install.** `requirements.lock.txt` pins `dvc` but
not `dvc-s3`, and the R2 remote is an `s3://` URL — so a lock-only environment
reports `s3 is supported, but requires 'dvc-s3' to be installed`. This is not a
Windows quirk: `.github/workflows/ci.yml` installs it the same way, as
`pip install -r requirements.lock.txt 'dvc[s3]'`, because only the release-gates
job ever fetches the bundle. It is already installed in this `.venv`; on a fresh
environment run:

```powershell
pip install "dvc[s3]"
```

To fetch the data you then need the R2 credentials. DVC reads them through
boto3's default credential chain, so set them as environment variables rather
than writing `.dvc/config.local` — that keeps them off disk, which is the same
property CI relies on:

```powershell
$env:AWS_ACCESS_KEY_ID     = "<read-only R2 key id>"
$env:AWS_SECRET_ACCESS_KEY = "<read-only R2 secret>"
.\scripts\dvc.ps1 pull
.\scripts\dvc.ps1 status
python src/snapshot.py verify
```

The bucket URL, region and endpoint are already in the committed `.dvc/config`
and are not secrets. See `docs/DATA_GOVERNANCE.md` §2 for what may and may not
be redistributed.

The live pipeline additionally needs a free FRED API key in a `.env` file at the
project root:

```ini
FRED_API_KEY=your_key_here
```

---

## 6. What the port did to DVC and snapshot state

The porting changes touched 25 of the 49 code/config files that `dvc.lock`
records as stage dependencies, so **`dvc status` will report those stages as
changed** the first time you run it with the bundle present. That is expected,
and it does *not* mean any result is stale.

The evidence that it does not: the 24 files the port did not touch are
byte-identical to the hashes recorded on macOS, confirmed by comparing each
against `dvc.lock`. The line-ending restoration was exact. The 25 that differ are
exactly the files edited here, and every edit is behaviour-preserving on
macOS and Linux:

- `encoding="utf-8"` made explicit — the platform default there was already
  UTF-8, so the bytes read and written are unchanged.
- `Path.as_posix()` instead of `str()` for snapshot manifest keys — identical
  output on any POSIX filesystem.
- the `openssl` shell-out in `src/dividends.py` replaced by `cryptography` —
  a Bronze-cache network path that no DVC stage exercises.

So when you have the bundle, **`dvc commit` is the correct move, not
`dvc repro`**: accept the new dependency hashes without recomputing. Reproducing
would churn frozen research evidence to arrive at the same numbers.

```powershell
.\scripts\dvc.ps1 status        # expect: the stages above reported as changed
.\scripts\dvc.ps1 commit        # accept the new code hashes, keep the outputs
```

One further consequence: `requirements.txt` is a **snapshot input**, hashed by
`src/snapshot.py`, and it gained two declarations it should always have had —
`python-docx` (imported by the Livrable builders but never declared, so a fresh
`pip install -r requirements.txt` produced a tree where those scripts died on
`ModuleNotFoundError`) and `cryptography`. `python src/snapshot.py verify` will
therefore report `Snapshot mismatch: requirements.txt` until the manifest is
regenerated, after committing:

```powershell
.\scripts\dvc.ps1 repro --single-item --force snapshot_manifest
python scripts\build_model_cards.py      # the cards embed the manifest commit
```

That ordering — manifest first, then cards — is the one release gate 4 exists to
enforce. Commit both together.

## 7. Optional external tools

Neither is needed for the test suite or any pipeline stage.

| Tool | Needed by | Install |
|---|---|---|
| **Poppler** (`pdftoppm`, `pdftotext`) | `scripts/build_slides_global_2004.py` rasterises report figures for the deck; three PDF-content tests skip without `pdftotext` | `winget install --id oschwartz10612.Poppler`, then add the package's `Library\bin` to PATH |
| **Tectonic** or **XeLaTeX** | rebuilding `docs/rapport_final/main.pdf` | `winget install --id TectonicTypesetting.Tectonic` |

`scripts/build_slides_global_2004.py` now names Poppler and the install command
when `pdftoppm` is missing, instead of raising a bare `FileNotFoundError`.

---

## 8. Docker

Unchanged and fully portable — the image is Linux and pins Python 3.11, so none
of the above applies inside it. Requires Docker Desktop with the WSL 2 backend.

```powershell
docker compose run --rm test
docker compose up api
docker compose up dashboard
docker compose up notebook
```

The one PowerShell difference is the shell-expansion syntax in README's mounted
variant — `$PWD` works, but quote it the PowerShell way:

```powershell
docker compose run --rm -v "${PWD}/data:/app/data:ro" test
```

---

## 9. Known platform caveats

- **`n_jobs=1` is a standing policy**, in `src/ml_signals.py`,
  `src/model_selection.py`, `src/run_explainability.py` and
  `src/run_monitoring_baseline.py`. It was adopted after a native worker crash
  under XGBoost on macOS, and it is kept on Windows: single-threaded fits are
  what the committed results were produced under, so changing it would change
  the numbers before it changed the runtime.
- **torch is deliberately absent** from `requirements.txt`. The LSTM signal
  strategy was dropped after torch and xgboost segfaulted when loaded in the
  same process. That was a macOS diagnosis; nothing here re-opens it.
- **Long paths.** `docs/rapport_final/assets/figures/` nests deeply. If a
  checkout under a long parent directory produces `filename too long`, enable
  long paths once: `git config --global core.longpaths true`.
