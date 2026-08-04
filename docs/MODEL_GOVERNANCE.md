# Model governance — Portfolio ML

> **Scope.** This document describes how models in this repository are owned,
> versioned, reviewed, and retired. It is written to the shape of an internal
> model-review policy so that the project can be assessed by someone who
> reviews models for a living. **It is not a claim of certification, of
> regulatory approval, or of production readiness.** No independent validation
> function has reviewed this work, and none of it is approved for use with
> client capital.

Companion documents:
[`MODEL_CARD_REGIME_CONDITIONAL.md`](MODEL_CARD_REGIME_CONDITIONAL.md) (primary
system), [`MODEL_CARD_RF_CHALLENGER.md`](MODEL_CARD_RF_CHALLENGER.md) and
[`MODEL_CARD_XGB_CHALLENGER.md`](MODEL_CARD_XGB_CHALLENGER.md) (challengers),
[`MONITORING_RUNBOOK.md`](MONITORING_RUNBOOK.md).

---

## 1. Roles

| Role | Holder | Responsibility |
|---|---|---|
| Model owner | Souhail Bourhim | Design, implementation, documentation, reproducibility of the reference system |
| Contributors | EL WALI Zakarya, BOUAJINE Yasmine | Feature engineering, evaluation, deliverables |
| Academic supervision | M. Abdelmouttalib MAQIL (EURAFRIC Information) | Scope, review, acceptance of deliverables |
| Independent validation | **Not appointed** | — |

The empty row is deliberate and is the single most important line in this
document. Every result here was produced and checked by the same team.
Effective challenge from outside the build team is what would move this from
*documented* to *validated*, and it has not happened.

---

## 2. Model inventory and status

| Model | Role | Status | Card |
|---|---|---|---|
| `regime_conditional` | Reference allocation system | Research prototype — documented, reproducible, **not approved** | primary card |
| `rf_signal_tuned` | Challenger | Exploratory — did not displace the reference | RF card |
| `xgb_signal_tuned` | Challenger | Exploratory — did not displace the reference | XGB card |
| `equal_weight`, `min_variance_lw`, `max_sharpe`, `min_variance`, `min_variance_ewma`, `dcc_garch` | Baselines and ablation rungs | Retained for comparison | covered in the primary card |
| `LSTMSignalStrategy` | Challenger | **Withdrawn** — a `torch` + `xgboost` native-library conflict crashed the shared test process. Code removed rather than shipped unstable | — |

**Approval status: none.** There is no environment in which any of these models
is authorised to allocate real capital.

---

## 3. Versioning convention

Three identifiers must travel together. Any one alone is insufficient to
reproduce a number.

| Identifier | What it pins | Where |
|---|---|---|
| Git revision | Source code, pipeline definition, parameters | `git rev-parse HEAD` |
| Snapshot manifest | SHA-256 of every input and published artifact | `data/gold/snapshot_manifest.json` |
| DVC lock | The data version each stage consumed and produced | `dvc.lock`, remote `origin` |

**Release tags** are immutable and named `phaseN-<theme>`:

| Tag | Meaning |
|---|---|
| `phase1-reproducible`, `phase1-reproducible-v2` | Pipeline reproducible from a clean tree |
| `phase2-validation-protocol` | Forward-only selection and paired comparison of differences |
| `phase3-auditable` | Governance, explainability, inference contract, packaging |

A tag is never moved. If a tagged state proves wrong, a new tag supersedes it
and the reason is recorded here — a moved tag silently invalidates every
citation of it. `phase1-reproducible-v2` is that rule in action: the first
Phase 1 tag was left in place and a second one cut beside it.

**Recorded exception — `phase3-industrial-ready`, 2026-08-03.** Cut at
`e354aed`, then re-cut at `afc61b7` a few minutes later to include two commits
that landed immediately after: the release-gates CI job and the snapshot-
verification fix it exposed. The tag's own message asserts that the release
gates pass, so a tag predating the automation of those gates was the weaker
artifact. Re-cutting was acceptable **only** because nothing had cited the tag
yet — no report, no deliverable, no external reference. Once a tag has been
cited, the rule above applies without exception and a `-v2` is the answer.

The snapshot manifest records whether the working tree was clean when it was
written, and verification **rejects** a manifest produced from a dirty tree:
`git_commit` alone cannot distinguish "built from that commit" from "built from
that commit plus uncommitted edits". That distinction was not hypothetical; it
is why `schema_version` is 2.

---

## 4. Challenger policy

A challenger may replace the reference system only when **all** of the
following hold. These are the conditions this project's own challengers failed,
stated in advance rather than after seeing a result.

1. Hyperparameters selected on **strictly forward-only** folds
   (`PurgedWalkForwardSplit`), never on the frozen test segment.
2. Evaluated on the frozen test window, with the reference system re-evaluated
   on **identical dates**.
3. A **paired** moving-block bootstrap on the return differences whose
   interval excludes zero, with a null-centred p-value below the stated
   threshold. Non-overlapping marginal confidence intervals do **not** satisfy
   this and never will — marginal intervals are not a test of a difference.
4. The result survives on **both** universes, or the restriction to one is
   argued explicitly from a property of that universe.
5. The Deflated Sharpe Ratio is reported against the **full** trial count of
   the search that produced it, including the ML grid.
6. The multiple-testing position is stated. If a Reality Check or SPA test is
   not run, that is declared rather than implied.

A challenger that is better on point estimate and fails (3) is recorded as a
negative result and its card says so. That has happened, and the cards say so.

---

## 5. Retraining policy

**There is no automatic retraining.** The Dagster schedule is stopped and must
stay stopped until §8's preconditions are met.

Retraining is a deliberate, human-initiated action:

1. Refresh Bronze, rebuild through Gold, `dvc commit`, `dvc push`.
2. Re-run the affected comparison stages **end-to-end on one snapshot**. A long
   modelling run and a live scheduler must never share a data directory — that
   produced a torn artifact in which one universe read a newer snapshot than
   the other, and every affected baseline was wrong by exactly the amount the
   extra days explain.
3. Regenerate the snapshot manifest and the model cards; run the full suite.
4. Record what moved and why, in the same commit.

Selection must be re-run whenever retraining occurs. Carrying hyperparameters
chosen on an older window forward to a new one is selection leakage in slow
motion.

---

## 6. Rollback

Every published state is recoverable, and rollback is a checkout rather than a
rebuild:

```bash
git checkout <tag>
./scripts/dvc.sh checkout          # restore data for that revision
./.venv/bin/python src/snapshot.py verify
```

`dvc checkout` restores from the local cache; `dvc pull` from the R2 remote.

**Constraint on the remote:** it holds the current state, not the full history.
`dvc push --all-commits` cannot collect commits from before 2026-07 — their
`dvc.yaml` uses a stage structure DVC no longer loads — so those versions exist
only in the model owner's local cache. **`dvc gc` must never be run on this
repository**; it would reclaim about 5 MB and destroy the only copy.

---

## 7. Human review requirements

| Change | Required review |
|---|---|
| Any change to selection protocol, fold geometry, or test-set boundaries | Model owner **and** supervisor, before the run |
| A new challenger model | Pre-registered success criteria (§4) written down before the result is seen |
| A change to a published number | Regenerate every derived surface; the cards, dashboard, API and report are generated from artifacts precisely so this is mechanical |
| A wording change to a claim | Treated as a change to a result. §5.2 of the project record exists because a retracted claim survived in six surfaces at once |
| Restarting the scheduler | §8 preconditions, all of them |

---

## 8. Monitoring status, and what a deployment phase would require

Monitoring is implemented as an **offline, on-demand validation tool**
(`src/monitoring.py`). It compares a supplied evaluation window against a
versioned reference distribution taken from the Phase 2 training data
(`data/gold/monitoring_baseline.json`).

**No live schedule is active. Drift alerts, retraining triggers, and production
incident response are not operational.** Monitoring emits warnings; it never
alters model behaviour.

A later deployment phase may restart scheduling only after the torn-artifact
failure has a real fix:

1. **Atomic artifact publication** — a reader must never observe a half-written
   result set.
2. **Manifest verification before reads** — consumers validate the snapshot
   they are about to use rather than trusting an mtime.
3. **Versioned release directories with locking** — a modelling run and a
   scheduled run must not be able to share a mutable data directory.
4. **A rollback path exercised at least once**, not merely documented.

Until then the honest description is *monitoring-ready*, not *monitored*.

---

## 9. Data governance

> Full detail — per-source provenance and licensing, the Law 09-08 and GDPR
> assumptions with the condition under which they hold, the audit-log format
> and its prohibited fields — is in
> [`DATA_GOVERNANCE.md`](DATA_GOVERNANCE.md). Summary below.

| Question | Answer |
|---|---|
| Personal data processed | **None.** Public market prices, published macro series, and public issuer dividend disclosures only |
| Client data | None. The system has no client, no account, and no order path |
| Special-category data | None |
| Data subjects | None identified |

Because no personal data is processed, the obligations of Moroccan **Law
09-08** and of the **GDPR** relating to data subjects are not engaged by the
model itself. This is a statement of the processing performed, **not a legal
opinion**, and it would stop being true the moment a client portfolio, an
account identifier, or a user profile entered the system. Any such extension
requires a fresh assessment before it is built.

**Source licensing.** Market data is obtained from Yahoo Finance, medias24 via
BVCscrap, FRED, casablanca-bourse.com, and (for one research experiment)
investing.com. Terms differ per source and several prohibit redistribution.
The DVC remote is therefore a **private** bucket, and no market data is
published in this repository or in any release archive.

**Logging.** API logs record the endpoint, status, and artifact version. They
must not record request bodies or any caller-supplied content verbatim.

---

## 10. Known limitations carried at governance level

- **No independent validation** (§1).
- **No paired test of the difference existed until Phase 2**, so every
  comparative claim made before it — in either direction — over-claimed. Those
  statements are retracted in the project record rather than quietly edited.
- **Multiple-testing correction is `established`.** The experiment the earlier
  `not_established` status named has been run: every one of the 240 reachable
  configurations was re-evaluated on the frozen test dates, each net-return
  series stored, and White's Reality Check and Hansen's SPA bootstrapped over
  the whole set. Against the pre-specified primary benchmark
  (`regime_conditional`) no candidate establishes outperformance on either
  universe or statistic. Comparisons against `equal_weight` are exploratory by
  pre-specification and are not the basis of that status. See
  [`MULTIPLE_TESTING.md`](MULTIPLE_TESTING.md) and
  `data/gold/reality_check_results.json`.
- **`full_2021` has a short test window** (~1.75 years), which is why its
  intervals are wide. That is a sample-size property, not a model defect, and
  no model choice fixes it.
- **On `etf_2017` the weight cap very nearly determines the allocation.** Any
  conclusion drawn there about model choice must account for the constraint
  rather than attributing the outcome to the estimator.
