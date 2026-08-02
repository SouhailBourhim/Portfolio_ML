# Phase 1 — known gaps and the scheduled follow-up run

Phase 1 delivered a verifiable release snapshot: `src/snapshot.py verify` checksums the
23 inputs and artifacts behind every published number, refuses a manifest built from an
uncommitted tree, and refuses a revision that is not an ancestor of the checked-out one.

Two gaps remain. **Neither affects the numbers in the current release** — both are
integrity-of-process issues — but both require re-running the research stages to close,
which is why they are scheduled rather than fixed in place.

---

## Gap 1 — `src/memo.py` is not a declared DVC dependency (integrity hole)

`src/memo.py` implements the content-addressed cache used by
`ml_signals.fit_predict_expected_returns` and `dcc_garch.dcc_covariance`. It is imported by
both, and it materially affects what those functions compute (or, more precisely, whether
they recompute).

It is **not listed in `deps:` for any stage in `dvc.yaml`.**

Consequence: editing the cache logic — changing the key, the eviction bound, or what is
cached — would **not** mark `phase4_compare`, `phase4b_compare`, `phase4c_compare`,
`phase5_compare` or `dashboard_data` as stale. DVC would report the pipeline up to date
while the code that produces the numbers had changed. That is precisely the class of
silent staleness the snapshot manifest exists to prevent, so it should not be left open.

Why it is not fixed here: adding a dep changes the stage hash, which invalidates all five
stages and requires the full re-run below.

**Fix, in the scheduled run:** add `- src/memo.py` to the `deps:` of `phase4_compare`,
`phase4b_compare`, `phase4c_compare`, `phase5_compare` and `dashboard_data`.

While there, audit the same question for every other cross-module import — the check is
"does this file influence the stage's output?", not "is it named in the stage's `cmd`?".

## Gap 2 — a knowingly-stale docstring in `src/regime.py`

`_fit_hmm_uncached`'s docstring says the split exists "so the multi-restart EM can be
memoized on its inputs". That was true when written; HMM memoization was then **removed**
after the estimator was shown not to be bit-for-bit reproducible on this runtime, so the
docstring now describes something that no longer happens.

The correction was written and then reverted deliberately: `src/regime.py` is a dep of five
stages, so a comment-only edit invalidates roughly four hours of computation. Reverting kept
the release verifiable today.

The accurate explanation is not lost — it lives in `src/memo.py`, under
**"WHAT IS DELIBERATELY *NOT* MEMOIZED, AND WHY"**, together with the rule it produced:

> memoize an estimator only after its bit-reproducibility has been demonstrated on real
> data, not inferred from a seed argument.

**Fix, in the scheduled run:** restore the corrected docstring at the same time as Gap 1,
so one re-run pays for both.

---

## The scheduled run

Both fixes touch stage dependencies, so they need one clean end-to-end rebuild. Run it from
a **clean working tree**, otherwise the regenerated manifest will be rejected by its own
verifier.

```bash
# 1. Apply both fixes, run the suite, commit.
./.venv/bin/python -m pytest -q
git commit -am "Close Phase 1 dep gap: declare src/memo.py, correct the HMM docstring"

# 2. Rebuild everything the fixes invalidate (~3-5 h, dominated by phase5_compare).
./scripts/dvc.sh repro snapshot_manifest

# 3. Confirm, then tag.
./scripts/dvc.sh status                       # expect: up to date
./.venv/bin/python src/snapshot.py verify     # expect: Snapshot verification passed.
```

Measured stage costs on this machine, for planning:

| stage | wall time |
|---|---:|
| `phase4_compare` | 19 min |
| `phase4b_compare` | ~30 min (not separately timed) |
| `phase4c_compare` | 44 min |
| `phase5_compare` | ~3 h (the dominant cost) |
| `dashboard_data`, `crisis_windows`, `snapshot_manifest` | minutes |

Use `./scripts/dvc.sh repro --single-item <stage>` when you genuinely mean one stage.
Plain `--force` re-runs the **entire upstream chain**, including a network re-ingest that
regenerates the Gold layer — which is not usually what is wanted, and cost this project a
recovery from the DVC cache once already.

## Not a gap, but a standing limitation

No DVC remote is configured. The snapshot is verifiable locally, but the data cannot be
reconstructed from Git alone; a reviewer needs an approved remote or a separately supplied
release archive. Market data is not republished here for licensing reasons. See the
"Comment vérifier qu'un résultat est à jour" section of the README.
