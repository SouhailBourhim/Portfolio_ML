# Monitoring runbook

> ## Status: monitoring-**ready**, not monitored
>
> This is an **offline, on-demand validation tool**. A human runs a command and
> reads a report.
>
> **There is no live schedule.** Drift alerts, retraining triggers, and
> production incident response are **not operational**. The Dagster schedule is
> stopped and stays stopped until the four preconditions in
> [`MODEL_GOVERNANCE.md`](MODEL_GOVERNANCE.md) §8 are met.
>
> **Monitoring emits warnings. It never alters model behaviour.** No function
> in `src/monitoring.py` writes an artifact, fits an estimator, or changes a
> weight — and `test_the_module_exposes_no_write_or_fit_path` fails the build
> if one ever does.

Code: [`src/monitoring.py`](../src/monitoring.py),
[`src/run_monitoring_baseline.py`](../src/run_monitoring_baseline.py).
Artifact: `data/gold/monitoring_baseline.json` (DVC stage `monitoring_baseline`).
Tests: `tests/test_monitoring.py`.

---

## 1. Commands

Rebuild the reference (only after a deliberate retraining — see §6):

```bash
./scripts/dvc.sh repro monitoring_baseline
```

Run a drift check against the stored reference:

```bash
./.venv/bin/python src/run_monitoring_baseline.py --evaluate
```

Reading the output: `alert` = significant shift, investigate before citing any
result computed on that window. `warning` = moderate shift, note it. Neither is
an instruction to retrain, and neither changes anything on its own.

## 2. The reference window

| Universe | Reference (Phase 5 train + validation) | Rows |
|---|---|---:|
| `etf_2017` | 2004-11-19 → 2018-12-21 | 3,614 |
| `full_2021` | 2021-07-02 → 2024-10-16 | 797 |

**Fixed, not trailing.** A moving baseline makes slow drift undetectable: the
reference travels with the data and reports "no change" while the world
changes underneath it. Pinning it to the training period asks the question that
matters — *has the world moved away from what the model learned?*

**The frozen test segment is deliberately excluded.** A reference containing it
would compare that window against itself and report stability by construction.

The artifact stores quantile **edges**, not observations: it must stay
reviewable and must not become a second copy of licence-restricted market data.
That also makes it self-sufficient — a reviewer re-runs the comparison from the
committed JSON without holding the training window.

## 3. What is tracked

| Metric | Question | Function |
|---|---|---|
| Feature missingness | Did an input stop arriving? | `feature_missingness` |
| PSI per feature | Did an input's distribution move? | `psi_from_reference_summary` |
| Regime distribution | Is the detector spending more time in the defensive state? | `categorical_shift` |
| Prediction distribution | Are the challengers' predicted returns shaped differently? | `compare_distributions` |
| Allocation concentration | Is the book concentrating? (HHI, effective N, max weight) | `allocation_concentration` |
| Turnover | Is trading intensity drifting up? | `turnover_summary` |
| Cap-binding rate | Does the constraint still leave the optimizer room to act? | `cap_binding_rate` |
| Fallback rate — regime path | Is the regime model defaulting rather than deciding? | `fallback_rate` |
| Fallback rate — ALL estimator paths | Was a published result produced by a substitute model? | `model_fallback_rates` |

Three of these are unusual choices and are here on purpose:

- **Cap-binding rate.** On a small universe the weight cap can determine the
  allocation outright, in which case no change in the covariance model can
  express itself. A *rising* rate means the system has progressively less room
  to act on any view at all. That is a health signal about the system, not the
  market, and it appears in no returns-based metric.
- **Fallback rate, across every path.** `fallback_rate` reads the regime
  timeline's `converged` column and is blind to DCC-GARCH degrading to
  Ledoit-Wolf or an ML signal degrading to the naive sample mean;
  `model_fallback_rates` consumes `fit_reports.parquet` and covers all three.
  A baseline built on the regime column alone would understate how often a
  labelled model was not the model that produced the number. An absent
  artifact reports `not_measured`, never an implied zero.
- **Fallback rate (regime).** A non-converged regime fit is a documented degradation,
  not an error — the neutral posterior resolves to the defensive sub-strategy.
  But a rising rate means the system increasingly defaults instead of deciding,
  and returns will not show it.
- **Turnover.** Phase 4B's entire finding was a strategy with the best gross
  Sharpe of the comparison losing it to trading costs. A turnover distribution
  drifting upward matters even when returns look unchanged.

### PSI thresholds

`0.10` moderate, `0.25` significant — the **conventional** credit-risk bands.
They are not derived from this project's data and carry no statistical
guarantee here. They are stated in the artifact so a reader can disagree with a
specific number rather than an unstated judgement.

Bin edges come from the **reference** quantiles, never the pooled sample.
Binning on the union would let the evaluation window redefine the bins it is
being judged against, which suppresses exactly the shift being looked for.

## 4. Current reading (frozen test segment vs reference)

Run on the committed snapshot. **17 warnings.** This is a *diagnostic*, not a
result, and nothing below changes any published number.

### `etf_2017` — test 2018-12-24 → 2026-07-24

| Feature | PSI | |
|---|---:|---|
| `AVG_PAIRWISE_CORR` | 0.662 | alert |
| `CREDIT_SPREAD_DIFF_L1` | 0.312 | alert |
| `MARKET_DRAWDOWN` | 0.284 | alert |
| `USDMAD_DIFF_L1` | 0.232 | warning |

### `full_2021` — test 2024-10-17 → 2026-07-24

| Feature | PSI | |
|---|---:|---|
| `MARKET_DRAWDOWN` | 4.335 | alert |
| `MARKET_VOL_LONG` | 3.333 | alert |
| `TAUX_DIR_DIFF_L1` | 2.696 | alert |
| `AVG_PAIRWISE_CORR` | 0.758 | alert |

### Estimator fallback — every path, both windows

| Universe | Reference window | Evaluation window | Shift |
|---|---:|---:|:---:|
| `etf_2017` | 0 / 157 rebalances | 0 / 91 | stable |
| `full_2021` | 0 / 28 rebalances | 0 / 21 | stable |

Covers all four fallback-capable strategies per universe. No estimator was
substituted in either window, so no published figure is a hybrid on this
snapshot. See [`MODEL_INTEGRITY.md`](MODEL_INTEGRITY.md) for the release
statement and `data/gold/fit_reports.parquet` for the per-rebalance source.

### Prediction distributions (one fixed model, two input windows)

| Universe | `rf_signal_tuned` | `xgb_signal_tuned` |
|---|---:|---:|
| `etf_2017` | **0.317** alert | 0.030 stable |
| `full_2021` | **0.676** alert | 0.061 stable |

The model is held **fixed** — fitted once on reference rows, then scored over
both windows — so this isolates input drift from refitting. The asymmetry is
striking and consistent across both universes: the RandomForest's output
distribution moves substantially while XGBoost's barely does. A plausible
reading is that the shallower, more regularized XGB configuration selected in
Phase 5 produces a narrower and more stable output range; this is an
observation to investigate, not an established mechanism.

### How to read this

**`AVG_PAIRWISE_CORR` shifts significantly on both universes.** That is the
direct P3 signal — cross-asset correlation, the quantity whose breakdown in
crises motivated the regime layer. It moved between the training period and the
evaluation period on both universes independently.

**The very large `full_2021` values are partly a window-length artefact.** That
universe's test segment is 462 rows against a 797-row reference, and drawdown
and long-window volatility are strongly autocorrelated, so a single sustained
episode can dominate the histogram. Do not read PSI = 4.3 as "4.3 times worse
than PSI = 1.0"; the statistic is unbounded and not linear in severity.

**Health metrics are steadier than the inputs.** On `full_2021` turnover moved
0.239 → 0.293 and the cap-binding position rate 0.194 → 0.138; on `etf_2017`
the fallback rate fell to zero and cap binding held at ~0.80. The system's
*behaviour* changed far less than its *inputs* did — which is what a
constraint-dominated allocator should look like, and is worth knowing.

**Regime share moved toward the defensive state on both** — `etf_2017` bear
0.33 → 0.47, `full_2021` bear 0.50 → 0.62 — but the categorical PSI stays in the
stable band (0.084 and 0.058).

None of this establishes that the model has degraded. It establishes that the
inputs are not drawn from the same distribution as the training period, which is
P2 (non-stationarity) showing up in a measurement instead of an argument.

## 5. Investigating an alert

1. **Is it a schema change or drift?** A feature missing from one side is
   reported separately as `__schema_mismatch__` — that is a pipeline break, not
   a moving market, and it is fixed rather than interpreted.
2. **Is the window long enough?** PSI on a few hundred rows of autocorrelated
   data is noisy. Check `evaluation_rows` before reacting.
3. **Does a health metric agree?** An input shift with no movement in turnover,
   concentration, cap binding or fallback rate is a shift the system absorbed.
4. **Only then** consider whether a retraining is warranted — a human decision
   under §6, never an automatic consequence.

## 6. Retraining is not a monitoring output

Monitoring cannot trigger a retrain. Retraining is the deliberate procedure in
[`MODEL_GOVERNANCE.md`](MODEL_GOVERNANCE.md) §5: refresh Bronze, rebuild
end-to-end **on one snapshot**, re-run selection (carrying old hyperparameters
into a new window is selection leakage in slow motion), regenerate the manifest
and the model cards, and record what moved.

Rebuild this baseline **after** such a retraining, never before — a reference
regenerated on the new data would erase the evidence that anything changed.

## 7. What would have to be true before this became live monitoring

From `MODEL_GOVERNANCE.md` §8, all four:

1. Atomic artifact publication — no reader ever observes a half-written result set.
2. Manifest verification before reads — consumers validate the snapshot rather
   than trusting an mtime.
3. Versioned release directories with locking — a modelling run and a scheduled
   run cannot share a mutable data directory.
4. A rollback path exercised at least once, not merely documented.

Until all four hold, the honest description is *monitoring-ready*.
