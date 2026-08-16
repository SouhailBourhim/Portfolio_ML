# Pre-Registration — Global Multi-Asset Universe (`global_2004`)

**Status:** **FROZEN PROTOCOL — no performance evaluated.**
**Commit:** standalone, containing this file only, **before** any ingestion, feature build, or
performance calculation for the new universe. Pushed immediately, so the protocol timestamp is
externally verifiable rather than merely local.
**Checkout:** `/Users/apple/Projects/portfolio_ml_release` exclusively. Do not run
ingestion, DVC, or reporting commands for this experiment from any other working copy.

**Self-description, stated up front:** this is a *prospectively frozen protocol following
exploratory universe-design diagnostics*. It is **not** a pristine pre-registration. Allocation
diagnostics over 2004–2026 were computed and inspected before this document was written, and
that inspection informed the exclusion of two instruments (§7). What this document does
guarantee is that **no strategy performance from the new universe has been observed by anyone**,
and that the ten tickers, the evaluation protocol, the decision rules, and the reporting
commitments below are fixed as of its commit.

The commit timestamp of this file is the evidence that the protocol preceded the results. That
is the entire point of committing it alone.

---

## 1. Why a new universe

Two defects in the existing universes are established, measured, and documented. Neither is a
modelling failure; both are properties of the *opportunity set*, which is why no modelling change
can address them.

**`etf_2017` — the constraint empirically dominates the allocation.** With 5 assets and
`max_weight = 0.25`, `5 × 0.25 = 1.25`, so every feasible long-only portfolio must hold at least
four assets with **positive weight**, and no asset may exceed equal weight by more than five
percentage points. The arithmetic alone does **not** force a corner — equal weight (20% each) is
feasible with nothing at the cap. What is *measured* is that the constraint nonetheless dominates
the objective: `min_variance_lw` emits **one distinct allocation across 248 rebalances** (versus
**171** at a 0.30 cap), and `min_variance` / `min_variance_lw` / `regime_conditional` returned
byte-identical weights post-2018 (`max|diff| 4.4e-16`). The covariance ablation ladder cannot be
tested on a universe where the optimizer demonstrably does not express a view.

Source: `data/gold/etf_cap_verdict.json` (DVC stage `etf_cap_sweep`) — the **canonical** cap
sweep, and the only one with a surviving artifact. A historical sweep in `etf_deep_history.py`
reported 169 on a different cap grid; it is non-canonical and is not quoted here.
See also AGENTS.md §10.1.

**`full_2021` — the covariance input is known-biased.** Casablanca and NYSE sessions do not
overlap and BVC prices are frequently stale. Measured on the target universe with `etf_2017` as
a clean control (`docs/NONSYNC_COVARIANCE.md`, `data/gold/nonsync_covariance.json`):

| | BVC (4 non-US) | US-listed (4) |
|---|---:|---:|
| mean same-day correlation vs `SPY` | 0.0041 | 0.4902 |
| mean lag-1 correlation vs `SPY` | 0.0779 | −0.0287 |
| lag-1 ÷ same-day | **19.1×** | −0.06 |
| share of zero-return days | **17.1%** | 0.6% |

The signature is present on the target and absent on the control. The daily covariance matrix
`full_2021` supplies to every optimizer systematically understates cross-market dependence, and
switching estimator moves the allocation at **100% of rebalances** (mean turnover to switch
0.166 weekly / 0.181 Dimson).

What that experiment did **not** establish is which estimator is correct: paired tests against
the production daily estimator returned `p = 0.3038` (weekly) and `p = 0.4423` (Dimson). We
therefore do not retroactively adopt whichever alternative scored highest, and `full_2021`
remains as-is with the bias documented as a limitation.

**Consequently:** the covariance-model ladder (sample → Ledoit-Wolf → EWMA → DCC-GARCH) and the
HMM regime layer have never had a test that is simultaneously *synchronous* and *empirically
allocation-expressive under the same management constraint*. `global_2004` is built to be exactly
that test, and nothing more.

⚠️ **"Unconstrained" would be the wrong word and is deliberately not used.** The 25% cap stays on
(§4) — removing it would be a second treatment and would abandon the *contraintes réalistes de
gestion* the brief requires. The claim is only that on ten assets the optimizer demonstrably
varies its allocation under that same cap, where on five it did not.

---

## 2. The frozen universe

**Ten instruments, frozen as of this commit:**

```
SPY  QQQ  IWM  EFA  EEM  IEF  TLT  LQD  IYR  GLD
```

**Identifier:** `global_2004`. **Start:** `2004-11-18` (`GLD` inception is binding).
**End:** the prevailing Gold snapshot date at run time.

### 2.1 Eligibility rules

All rules are stated in terms of **ex-ante instrument attributes** — inception date, listing,
denomination, prospectus category, stated holdings. **None require return data.** The ticker set
is therefore fully determined without observing a single return.

| # | Rule | Basis |
|---|---|---|
| E1 | US-listed, USD-denominated, exchange-traded fund | listing venue |
| E2 | Continuous adjusted-price coverage from `2004-11-18` | inception date |
| E3 | Covers a distinct economic sleeve: US large-cap, US growth/tech, US small-cap, developed international, emerging markets, intermediate Treasury, long Treasury, investment-grade credit, real estate, gold | stated mandate |
| E4 | **Exclude cash-equivalent and ultrashort instruments** — the mandate is risky-asset allocation | prospectus category / stated effective duration |
| E5 | **Exclude composite funds substantially replicating dedicated sleeves already present** in the set | published holdings composition |
| E6 | Liquid: continuously exchange-traded over the full window, no suspension or reconstitution gaps | trading record |

**E4 excludes `SHY`** (ultrashort Treasury; a cash proxy is not a risky-asset allocation
decision). **E5 excludes `AGG`** (an aggregate blend whose Treasury / corporate / MBS components
are already represented by `IEF`, `TLT` and `LQD`).

Both exclusions are justified on instrument attributes, not on realized volatility or realized
correlation. An earlier draft justified them with realized statistics (a 3% volatility floor, a
0.77 correlation); those justifications are **withdrawn** because a threshold chosen after
observing the data is vulnerable to exactly the tuning accusation this document exists to
foreclose. See §7 for the disclosure of what was in fact observed.

**Broad commodities are excluded** because the earliest broad-commodity ETFs (`DBC` 2006-02,
`GSG` 2006-07) would move the common start from 2004-11-18 to 2006-02, and preserving the full
window — including the pre-crisis regime the HMM needs in order to enter 2008 already trained —
is worth more than a tenth sleeve.

### 2.2 No changes after performance is calculated

The ticker list, the start date, and rules E1–E6 are frozen. If a data-quality defect is
discovered during ingestion that requires a change, the change and its reason must be recorded as
a dated amendment in §10 of this file **before** any performance is calculated, and the amendment
commit must precede the results commit.

---

## 3. Macro-feature policy

`ml_features_global.parquet` inherits the Phase 3 feature definitions unchanged: six causal
return features and the lagged macro block. One deviation is registered here.

### 3.1 Excluded: the Bank Al-Maghrib block

`TAUX_DIR_DIFF_L1`, `EURMAD_DIFF_L1` and `USDMAD_DIFF_L1` are **excluded** from
`ml_features_global.parquet`. The universe is US-listed and USD-denominated; a Moroccan policy
rate and MAD cross-rates have no economic role in it.

### 3.2 Retained: the globally relevant block

`VIX_DIFF_L1`, `US10Y_DIFF_L1`, `CREDIT_SPREAD_DIFF_L1`, `DXY_DIFF_L1` are retained — all are
global risk/rate/credit/dollar signals that bear on a USD multi-asset portfolio.

### 3.3 Leading incomplete rows — recorded, never backfilled

Backfill and interpolation use future information and are banned project-wide (§15.4). The
retained block still begins later than the price history:

| column | expected leading NaN |
|---|---:|
| `DXY_DIFF_L1` | ~231 |
| `VIX_DIFF_L1`, `US10Y_DIFF_L1`, `CREDIT_SPREAD_DIFF_L1` | 0 |

Expected `max_leading_nan ≈ 231` (~11 months), against **3,101** had the BAM block been retained —
`TAUX_DIR_DIFF_L1` alone would have pushed the first fully-dense row to 2017-01-04, discarding
twelve of the twenty-one years and destroying the reason for the universe. The manifest must
record `leading_nan_by_column`, `max_leading_nan` and `fully_complete_rows` per §15.13, and the
warm-up check required by §11 must be re-run and its result recorded before the first fit.

### 3.4 Attribution consequences — registered before the run

This is a second treatment alongside the asset cross-section, and it does not affect the two
primary questions equally:

- **Q1 remains a clean universe-design comparison.** `regime_conditional` and `max_sharpe` both
  consume only return-derived inputs — `regime.REGIME_FEATURES` is `MARKET_RETURN`,
  `MARKET_VOL_SHORT`, `AVG_PAIRWISE_CORR`, all computed from the universe's own returns. Neither
  strategy reads a macro column, so the macro-feature policy cannot influence Q1.
- **Q2 confounds two changes.** The RF/XGBoost challengers consume the macro block. Their results
  on `global_2004` therefore reflect *both* the wider asset cross-section *and* the changed
  macro-feature policy, and **challenger outcomes may not be attributed exclusively to the wider
  universe.** This limitation must be restated wherever Q2 results are reported.

---

## 4. Frozen evaluation protocol

Everything below is **inherited unchanged** from the existing universes. The sole intended
treatment is universe design; any protocol deviation would confound the attribution.

| Parameter | Value | Source |
|---|---|---|
| Weight cap | `max_weight = 0.25`, long-only | `params.yaml: backtest.max_weight` |
| Transaction costs | 10 bps (all instruments are ETFs; the 30 bps BVC rate does not apply) | `params.yaml: backtest.costs_bps` |
| Rebalancing | month-end (`ME`) | `params.yaml: backtest.rebalance_freq` |
| Minimum training window | 252 trading days, expanding | `params.yaml: backtest.min_train_days` |
| Frozen test segment | `test_frac = 0.35` (final 35% of the OOS window) | `params.yaml: phase5.test_frac` |
| ML selection CV | **forward-only** `PurgedWalkForwardSplit` — expanding window, label-horizon purge, separate embargo, no training date after the validation fold (`train_end < embargo_start <= val_start <= val_end < test_start`), IC-scored | `params.yaml: walk_forward_cv` |
| Risk-free rate | `0.0` annual | `params.yaml: backtest.risk_free_annual` |
| Regime model | 2-state HMM, `diag`, 5 restarts, `min_regime_train_days = 252` | `params.yaml: regime` |
| Covariance ladder | sample, Ledoit-Wolf, EWMA (halflife 63), DCC-GARCH | `params.yaml: covariance_*` |
| Bootstrap | moving-block, monthly blocks, seeded, `n_boot = 2000` | `metrics.block_bootstrap_sharpe_ci` |

**No-lookahead handling is inherited too:** the engine slices `returns` and every `extras` frame
to `:τ` before each fit; strategies never read feature Parquet directly (§15.15). The Phase 3 and
Phase 4 future-corruption integration gates must pass against `global_2004` before any result is
reported.

---

## 5. Primary questions

Exactly two, fixed in advance. Both are evaluated on the frozen test segment.

### Q1 — Does the regime layer beat the classical comparator?

> **`regime_conditional` versus `max_sharpe`, net of transaction costs, on `global_2004`.**

`max_sharpe` is the pre-specified classical comparator, named now, before any result.

**One pre-specified paired comparison.** Paired moving-block bootstrap on the net return
differences over identical test dates, same machinery as `data/gold/paired_comparison_results.json`.
Reported: observed net-Sharpe difference, its bootstrap confidence interval, and the one-sided
p-value for H₀ = no outperformance.

**No RC/SPA correction is applied to Q1.** It is a single hypothesis fixed before the run and was
never part of the challenger search; correcting it over a candidate grid it does not belong to
would be over-conservative and could bury a real effect.

**Two requirements, reported separately, neither implying the other:**

1. **Economic materiality** — an **absolute** improvement of at least **0.05 Sharpe points**:

   ```
   MATERIAL_MARGIN = 0.05          # Sharpe points, NOT a percentage
   S_regime − S_maxsharpe ≥ 0.05
   ```

   This is the project's existing constant, used with its existing meaning
   (`experiments/regime_conditional_cap.py:107`: `entry["sharpe_net"] - baseline >=
   MATERIAL_MARGIN`). An earlier draft of this document specified a *relative* 5% improvement;
   that was **overruled** in review, for four reasons worth recording so it is not re-proposed:
   it would have preserved the name while changing the quantity (the drift class §17.11 warns
   about); the absolute form is well-defined when the comparator is zero or negative; percentage
   improvement in a Sharpe ratio is unstable and misleading near zero; and the absolute form
   needs no second constant and no fallback edge case.

   Describe it as *an absolute improvement of 0.05 in net Sharpe* — never as "5%".

2. **Statistical evidence** — the paired test above.

Economic materiality and statistical evidence are **separate requirements**. Clearing one does
not license a claim about the other, and both must be reported whichever way each resolves.

### Q2 — Do the ML challengers beat the regime layer?

> **RF / XGBoost signal challengers versus `regime_conditional` on `global_2004`.**

`regime_conditional` is the pre-specified benchmark. **Full multiple-testing correction applies:**
White (2000) Reality Check and Hansen (2005) SPA over the complete reachable candidate ledger —
the entire space the hierarchical search could have selected, not the subset of trials the DSR
ledger happens to record (`docs/MULTIPLE_TESTING.md`).

Subject to the attribution limitation registered in §3.4.

---

## 6. What does not count as success

Registered in advance, because each of these has been mistaken for a result at least once in this
project's history:

- **A higher information coefficient is not success.** Three independent experiments have raised
  IC without improving allocation: deep-Morocco (IC ×2–4, no portfolio edge), fundamentals (IC ×2,
  Sharpe −0.331), Phase 5 (honest tuning, no edge). IC is a diagnostic.
- **More distinct allocations is not success.** It establishes that the universe *can* express a
  view, which is a precondition for the experiment being informative, not evidence about any
  model.
- **A higher point Sharpe is not success.** Every point ranking in this project has proven
  unstable to evaluation design — `full_2021` produced three different orderings across three
  evaluations.
- **A balanced-looking allocation is not success**, and is not a design target. If long-only
  minimum variance concentrates in duration, that is the finding. The universe will **not** be
  modified to make weights look diversified. A maximum sleeve weight, if ever wanted, is a
  separate pre-registered management-constraint experiment, not an invisible universe-selection
  rule.
- **Overlapping marginal confidence intervals are not a test of a difference**, in either
  direction (§5.2 rule 2). Only the paired test speaks to the difference.

---

## 7. Exploratory information observed before protocol freeze

**This section is disclosure, not evidence.** Nothing below is pre-registered, and none of it may
be cited as a result of this experiment.

Before this document was written, allocation diagnostics were computed over the full
2004-11-19 → 2026-08-14 history on a 12-ticker candidate set
(`SPY QQQ IWM EFA EEM SHY IEF TLT LQD AGG IYR GLD`) and on the pruned 10-ticker set, using
`MinVarianceLW` and `MaxSharpe` at `max_weight = 0.25`, expanding window, 252-day minimum, 249
month-end rebalances.

**Observed:**

| set | strategy | distinct allocations | assets at cap | effective N | fixed-income share |
|---|---|---:|---:|---:|---:|
| 12-ticker | `min_variance_lw` | 249 / 249 | 2.79 | 4.8 | 87% |
| 12-ticker | `max_sharpe` | 249 / 249 | 2.77 | 4.7 | 78% |
| 10-ticker | `min_variance_lw` | 249 / 249 | 2.12 | 4.9 | 69% |
| 10-ticker | `max_sharpe` | 249 / 249 | 1.85 | 4.7 | 60% |

Also observed: full-sample annualised volatilities of the twelve candidates, and the pairwise
correlation matrix of the five fixed-income candidates.

**What was *not* observed: no portfolio return, net or gross Sharpe, drawdown, turnover-cost, or
any other performance quantity was computed for any strategy on any candidate universe.** Only
weights and their descriptive statistics were produced.

**How this influenced the design:** the 87% / 78% fixed-income concentration and the fact that
`SHY` and `AGG` sat at the cap in nearly every rebalance prompted their exclusion. The exclusions
are justified in §2.1 on instrument attributes that stand independently — but they were
*identified* with this diagnostic in hand, and presenting them as arrived at a priori would be
false.

**Standing:** the 249/249 figure is design evidence that the universe can express a view. It is
not an acceptance filter, and no further ticker modification will be made on the basis of any
diagnostic of this kind.

**Residual outer selection.** RC/SPA (Q2) corrects the *strategy* search. It does not correct the
outer selection involved in *designing a new universe* after observing that the existing two were
each defective. Any positive result must be reported with that residual multiplicity stated.

---

## 8. Registered hypothesis — regime feature conditioning

Not a primary question; recorded so it is falsifiable rather than asserted after the fact.

`AVG_PAIRWISE_CORR` is computed from the universe's own returns: 45 pairs on `global_2004` versus
10 on `etf_2017`. **Hypothesis:** the wider cross-section may reduce the influence of
individual-asset noise on the regime correlation feature. This is *not* a claim of better
conditioning — the pairs are dependent and several instruments share common factors, so 45 pairs
carry far less than 4.5× the independent information.

**Evaluated through:** feature stability across rebalances, HMM state occupancy and transition
rates, and out-of-sample allocation behaviour — not through Sharpe.

---

## 9. Intended outputs and lineage

**None of these exist yet. None may be created before this document is committed.** Every Gold
output must land with its Dagster asset and DVC stage in the same change (§15.10, §17.7 — an
unwired universe silently going stale against Bronze is a documented repeat failure here).

**Bronze**
- `data/bronze/raw_global_prices.parquet` — Dagster asset `raw_global_prices` (group `bronze`)

**Silver**
- `data/silver/log_returns_global.parquet` — Dagster asset `log_returns_global` (group `silver`)
- `data/silver/validation_report_log_returns_global.json`
- Pandera contract in `src/schemas.py`

**Gold**
- `data/gold/log_returns_global.parquet`
- `data/gold/ml_features_global.parquet` — via the existing `ml_features_layer` asset, extended
- `data/gold/ml_features_manifest.json` — new `global_2004` entry carrying
  `leading_nan_by_column`, `max_leading_nan`, `fully_complete_rows`

**Results**
- `data/gold/global_universe_results.json` — strategy comparison
- `data/gold/global_universe_paired.json` — Q1 paired bootstrap
- `data/gold/global_universe_reality_check.json` + `_series.parquet` — Q2 RC/SPA + candidate ledger

**DVC stages** (`dvc.yaml`): `ingest_global`, `clean_global`, `ml_features_global`,
`global_universe_compare`, `global_universe_paired`, `global_universe_reality_check`

**`params.yaml`**
- `ingest.global_tickers` — the ten, frozen
- `backtest.universes.global_2004: "data/gold/log_returns_global.parquet"`
- `ml_features.outputs.global_2004`
- `ml_features.macro_block.global_2004` — the §3 exclusion policy, declared in config, not code

**Tests**
- `tests/test_artifact_consistency.py` extended to cover the new universe
- Phase 3 / Phase 4 future-corruption gates parametrized over `global_2004`
- `tests/test_orchestration.py` asserting the new assets resolve

---

## 10. Reporting commitments and amendments

### 10.1 The null is a complete result

> **"No difference established" on either primary question is a complete, reportable result, and
> will not trigger a change of universe, benchmark, threshold, or evaluation window.**

This is the commitment the rest of the document exists to make credible. This project's strongest
asset is its honest negative findings; the failure mode being foreclosed is the slow drift toward
searching until something clears.

### 10.2 Sequencing

1. Commit this document alone. ← the protocol timestamp
2. Implement the data universe and its lineage (§9). No performance calculation.
3. Verify: no-lookahead gates green, manifest warm-up check recorded, `dagster definitions
   validate` clean, artifact-consistency tests passing.
4. Only then run Q1 and Q2, and report both regardless of outcome.

### 10.3 Amendments

Any change to §2–§5 after this file is committed must be appended below with its date and reason,
in its own commit, before any affected result is calculated. An empty log is the expected state.

| Date | Section | Change | Reason |
|---|---|---|---|
| 2026-08-15 | Readiness gates (checkpoint 1) | `synchronous_trading` ratio gate **replaced** by `no_lag_dominance` | The original statistic was mis-specified. Full record below. |
| 2026-08-15 | §9 lineage | `global_2004` config lives in `params_global_2004.yaml`, not in `params.yaml` | Approved in review. Isolating the experiment from the released runners is the correct design. |
| 2026-08-15 | §5 Q2 | Endpoint hierarchy and RC/SPA disagreement rule frozen | §5 named the tests but not how to read them. Full record below. |

#### Amendment 1 — `synchronous_trading` → `no_lag_dominance` (2026-08-15)

**Legitimacy.** No performance quantity had been computed for `global_2004` when this amendment
was made, and this document was committed before it. The repair is therefore a protocol fix made
in the open, not a result-driven one. It is committed **before** the code that implements it.

**The original gate, preserved verbatim rather than deleted:**

```yaml
# ORIGINAL — mis-specified, superseded 2026-08-15
max_lag1_over_same_day: 0.25   # full_2021's BVC block measured 19.1x
```
scored as `max_i |ρ_i(1) / ρ_i(0)|` against the reference asset `SPY`.

**Why it was wrong.** The statistic divides by the contemporaneous correlation, which is near zero
for any asset genuinely uncorrelated with the reference. On `global_2004` it observed **0.6228**
and FAILED, driven entirely by `GLD` (ρ(0) = 0.0657, ρ(1) = 0.0396 — both tiny, the ratio
meaningless). Gold is simply uncorrelated with equities; nothing about that is a stale-price
problem.

**Why the obvious repair was ALSO rejected.** An absolute bound `max_i |ρ_i(1)| ≤ 0.20` was
proposed and rejected in review, correctly: the documented stale-price block in `full_2021` has a
largest lag correlation of only **0.0897**, so that gate would have passed the known-bad control.
**A gate that passes both the clean universe and the defective control tests nothing.** Recorded
because the rejected repair is as informative as the accepted one.

**The replacement — lag dominance.** For each non-reference asset `i`:

```
D_i = max(|ρ_i(−1)|, |ρ_i(+1)|) − |ρ_i(0)|        gate: max_i D_i ≤ 0
```

Contemporaneous dependence must dominate **both** lead and lag dependence. It has no denominator,
so nothing explodes when ρ(0) → 0, and no numerical threshold was tuned against a result — the
bound is zero, fixed by the meaning of the statistic rather than chosen to admit a universe.

**Measured on the committed artifacts before adoption:**

| Universe | Worst asset | `max D` | Verdict |
|---|---|---:|---|
| `global_2004` | `GLD` | **−0.0248** | pass |
| `etf_2017` (clean control) | `GLD` | **−0.0264** | pass |
| `full_2021` — `IAM.CS` | | **+0.0807** | **fail** |
| `full_2021` — `CIH.CS` | | **+0.0682** | **fail** |
| `full_2021` — `ATW.CS` | | **+0.0557** | **fail** |
| `full_2021` — `BCP.CS` | | **+0.0479** | **fail** |

It separates both clean USD universes from the documented stale-price block, and does so
**per asset**: the four US-listed instruments inside `full_2021` pass (`TLT` −0.0931, `GLD`
−0.1405, `EEM` −0.6208, `QQQ` −0.8929) while all four BVC assets fail. That is the discrimination
the gate exists to provide, and it is locked in by regression tests over all three universes.

**What the gate licenses you to say.** *"No stale-price lead/lag signature detected."* **Not**
*"synchrony proven"* — the statistic can only fail to find a signature it is built to detect, and
absence of evidence at daily frequency is not proof of simultaneous price formation.

#### Amendment 2 — Q2 endpoint hierarchy and RC/SPA disagreement rule (2026-08-15)

**Legitimacy.** Committed before Q2's implementation and before any Q2 return has been computed.
Q1 is already frozen and is not touched by this amendment.

**What was under-specified.** §5 Q2 named White (2000) Reality Check and Hansen (2005) SPA, and
required the full reachable candidate ledger. It did **not** say which statistic is the endpoint,
nor how to read the two tests when they disagree. Left open, that is a researcher degree of
freedom exercised *after* seeing results — the precise thing this document exists to remove. It is
closed now, in advance.

**1. Endpoint hierarchy.**

| | Endpoint | Status |
|---|---|---|
| Primary | **net-Sharpe differential** vs the benchmark | the registered question |
| Secondary | **annualized mean-return differential** | reported, labelled SECONDARY, never promoted |

The secondary endpoint is reported because it is informative — in Q1 its interval was entirely
negative while the Sharpe interval spanned zero — but it can never substitute for the primary. A
result significant only on the mean-return endpoint is **not** family outperformance.

**2. Significance level.** `α = 0.10`, inherited unchanged from the existing evaluation
(`params.yaml phase5.bootstrap.alpha`). Not re-chosen for Q2.

**3. Both tests reported separately.** White RC and Hansen SPA are reported as distinct p-values
per endpoint. Neither is presented as "the" answer, and the artifact never reports a single
p-value for the family.

**4. The concordance rule.**

```
concordant_evidence_of_family_outperformance = (RC rejects at α) AND (SPA rejects at α)
                                                on the PRIMARY Sharpe endpoint
```

- **Both reject** → concordant evidence of family outperformance.
- **Exactly one rejects** → **"test-dependent evidence"**. Reported as such, explicitly NOT as
  established outperformance. SPA is the more powerful test by construction (it discards
  irrelevant poor candidates), so RC-rejects-while-SPA-does-not is the more surprising direction
  and must be reported, not resolved.
- **Neither rejects** → no evidence of family outperformance.

**5. No statistic shopping.** The reported conclusion may **not** be taken from whichever test or
endpoint yields the smaller p-value. All four cells (2 tests × 2 endpoints) are computed and
persisted; the verdict reads only the primary-endpoint pair, under the rule above. This is stated
because the failure mode is silent: with four p-values in an artifact, quoting the smallest is a
single sentence away and looks like reporting.

---

*Drafted 2026-08-14; revised the same day after review resolved four points: the materiality
margin is **absolute** (0.05 Sharpe points, §5 Q1); the `etf_2017` cap is **empirically
cap-dominated**, not mathematically cap-determined (§1); Phase 5 selects with **forward-only**
`PurgedWalkForwardSplit`, not purged K-Fold (§4); and the goal is a universe that is
**allocation-expressive under the same 25% cap**, never an "unconstrained" one (§1). The canonical
cap-sweep count is **171** from `etf_cap_verdict.json`. No open decisions remain in this document.*

*Frozen on commit. Any subsequent change requires a dated amendment in §10.3, in its own commit,
before any affected result is calculated.*
