# `global_2004` — Results

> **GENERATED FILE — do not edit by hand.** Every number below is read from a
> committed artifact by `scripts/build_global_2004_results.py`. Regenerate it
> rather than correcting it in place.

Protocol: [`GLOBAL_UNIVERSE_PREREGISTRATION.md`](GLOBAL_UNIVERSE_PREREGISTRATION.md), frozen before any data was ingested.

---

## 1. Why this universe exists

Two **measured** defects in the released universes made them unable to test the
ML stack cleanly. Neither is a modelling failure; both are properties of the
opportunity set, which is why no modelling change could address them.

**`etf_2017` — the constraint dominated the objective.** With 5 assets at a 25%
cap, `min_variance_lw` emitted **1 distinct allocation across
248 rebalances**. The arithmetic does not force that corner — equal weight
is feasible with nothing at the cap — but empirically the constraint picked the
portfolio, so a better covariance estimate had nowhere to show up.

**`full_2021` — the covariance input was known-biased.** Casablanca and NYSE
sessions barely overlap and BVC prices are frequently stale: same-day correlation
with SPY of 0.0041 against lag-1 of 0.0779, with 17.1% zero-return days.

`global_2004` was built to have neither defect, under the **same 25% cap**.

## 2. Allocation expressiveness — the defect is removed

| | `etf_2017` | **`global_2004`** |
|---|---:|---:|
| `min_variance_lw` distinct allocations | **1 / 248** | **249 / 249** |
| mean assets at the cap | ~4 | 2.1205 |
| effective positions | — | 4.9173 of 10 |
| window | 20.7 yr | 21.73 yr, 5,467 rows |

Data readiness: **READY**, 10/10 pre-registered gates, including no stale-price lead/lag signature (max lag dominance -0.0248, requirement ≤ 0) and both no-lookahead gates.

Same cap, same costs, same engine — and the optimizer now varies its allocation
at **every** rebalance instead of emitting one portfolio for two decades.

## 3. Q1 — the regime layer versus the classical comparator

One pre-specified comparison on the frozen test segment (2019-01-04 → 2026-08-14, 1,913 days). No multiple-testing
correction, because a single hypothesis fixed in advance needs none.

| | net Sharpe | geometric ann. return | max DD | turnover |
|---|---:|---:|---:|---:|
| `regime_conditional` | 0.8923 | 8.10% | -24.22% | 0.122383 |
| `max_sharpe` | **0.9785** | 9.52% | -25.27% | 0.028491 |

- observed ΔSharpe **-0.0862**, 90% CI [-0.2133, 0.0414]
- one-sided null-centred p = **0.85957**
- P(ΔSharpe > 0) = 0.1435
- `candidate_improvement_at_least_0_05` = **False**
- `evidence_of_candidate_outperformance` = **False**
- `observed_absolute_sharpe_gap_at_least_0_05` = **True** (candidate_below_comparator)

**No Sharpe outperformance is established. The observed net Sharpe difference is NEGATIVE and the paired interval spans zero, so the candidate is not shown to beat the comparator and the comparator is not shown to beat the candidate.**

*Secondary.* As a SECONDARY result, the paired bootstrap interval for the ANNUALIZED MEAN return difference is entirely negative. Do not therefore say 'no difference established' without qualification: that is true of the Sharpe comparison, which is the registered question, and false of this interval.

Transaction costs worsened the gap but did not cause it. The candidate's GROSS Sharpe was already lower — 0.908 versus 0.982 — so this is not a story about an informative signal priced out by trading friction.

The candidate had a SMALLER maximum drawdown — -24.22% versus -25.27%. It did not lose on every risk dimension, and reporting only the Sharpe comparison would omit that.

## 4. The honestly selected challengers

The configurations a disciplined practitioner would actually have deployed,
chosen forward-only on train+validation with the frozen test segment never
shown to a selector.

| strategy | net Sharpe | Δ vs regime | geometric ann. return | fallbacks |
|---|---:|---:|---:|---:|
| `regime_conditional` | 0.8923 | — | 8.10% | — |
| `rf_signal_tuned` | 0.8266 | -0.0657 | 7.78% | 0 |
| `xgb_signal_tuned` | 0.8025 | -0.0898 | 8.05% | 0 |

Both honestly selected challengers UNDERPERFORM the benchmark on the frozen test segment. That is a different and stronger statement than the family verdict: the family test says the BEST of 240 cannot be believed, while this says the ones a disciplined practitioner would actually have chosen, using only training data, did worse.

## 5. Q2 — the 240-candidate family test

Every reachable RF/XGBoost × portfolio-lever configuration — **240 expected from frozen configuration, 240 executed** — against `regime_conditional`. No candidate was deduplicated on observed performance.

| endpoint | White RC | Hansen SPA | beating benchmark | best raw differential |
|---|---:|---:|---:|---:|
| **PRIMARY** (Sharpe) | 0.904548 | 0.865567 | 17/240 | **+0.093015** |
| SECONDARY (mean return) | 0.456772 | 0.357821 | 92/240 | +0.000101 |

**`concordant_evidence_of_family_outperformance` = False** — `no_evidence_of_family_outperformance`.

### ⭐ Why this is the project's clearest methodological result

The best of the 240 candidates beat the benchmark by **+0.0930 Sharpe** — *larger in magnitude than Q1's -0.0862 gap, and in the opposite direction*. Quoted alone it would read as a headline result.

It is not one. It is the maximum of a 240-wide search, and **RC p = 0.904548** is exactly the correction pricing that fact. Only 17 of 240 candidates beat the benchmark on Sharpe at all.

The auditable evidence showing why the attractive raw winner should not be
believed is the contribution here — more so than the negative result itself.

## 6. Telemetry — the results are not a fallback artefact

- **Benchmark:** 3 fallbacks in 249 fits, **all before the frozen test segment**. Q1 reports zero because it counts only the rebalances inside that segment; the two counts cover different windows and are consistent.
- **Candidate family:** 1 fallback in 59,760 fits.
- **Both honestly selected challengers: 0, 0 fallbacks.**

Fallback behaviour therefore does **not** explain the negative results — every
number came from the model its label names.

## 7. Limitations

### These evaluations are not independent

etf_2017 and global_2004 are TWO DISTINCT BUT STATISTICALLY OVERLAPPING evaluations, not independent ones. They share five instruments (SPY, QQQ, EEM, GLD, TLT) and largely overlapping evaluation periods, so they share the same market shocks. Agreement between them is weaker evidence than agreement between independent samples would be.

Describe them as **two distinct but statistically overlapping evaluations**,
never as independent confirmations.

### Attribution

- ATTRIBUTION. Q2 changes TWO things at once relative to the released universes: the asset cross-section AND the macro-feature policy (the Bank Al-Maghrib block is excluded, §3.4). The challengers consume macro features, so their results here may NOT be attributed to the wider universe alone. Q1's comparison is unaffected, because neither of its strategies reads a macro column.
- RESIDUAL OUTER SELECTION. White RC and Hansen SPA correct the STRATEGY search. Nothing corrects the outer decision to CONSTRUCT global_2004 after measuring that the two released universes were each defective. Any positive result must be reported with that residual multiplicity stated.
- USD numéraire, single-currency: not directly comparable to full_2021, which is expressed in MAD.
- One frozen split. No nested walk-forward is run here; §12G showed point orderings on a single split can be unstable to evaluation design.

## 8. What is licensed

> The 25% cap made etf_2017 weak for model attribution. After restoring allocation expressiveness in global_2004, no regime or challenger advantage was established either. Cap dominance was therefore not the sole explanation for the earlier negative result.

Three separate things, deliberately not merged:

| | |
|---|---|
| **Engineering success** | A pre-registered universe was built, gated on ten data-readiness checks, wired into DVC and Dagster, and frozen as run-once evidence. |
| **Methodological success** | The correction did its job: a selected +0.093 Sharpe difference was prevented from becoming a false headline. |
| **Absence of established ML outperformance** | Neither the regime layer (Q1) nor the challenger family (Q2) established an advantage over simpler portfolio rules on this universe. |

Once the two identification defects were removed, the complex models were
finally given a fair test — and still did not establish an advantage.

---

*`global_2004` is a RESEARCH EXPERIMENT. It is not wired into the API, the
dashboard, or any production-facing allocation, and no released result depends
on it.*
