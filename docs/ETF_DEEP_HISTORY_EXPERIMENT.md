# ETF Deep History — Does a Longer Window Fix the Statistical Power Problem?

*Research note, 2026-07-25. Companion to `experiments/etf_deep_history.py`.*

## Motivation

Across Phase 5, the deep-Morocco experiment and the fundamentals experiment, every conclusion died
the same death: **the confidence intervals were too wide to resolve anything.** Test windows of
1.7–1.8 years produce block-bootstrap CIs spanning roughly two Sharpe points, so no comparison
between strategies can be called significant.

That is a sample-size problem, not a model problem. And `etf_2017` starts in 2017 only because of
a Phase 1 project decision — yfinance actually serves all five ETFs back to **2004-11-18** (GLD's
inception is the binding constraint; SPY reaches 1993). Twelve extra years, dividend-adjusted, free,
and containing the 2008 crisis the regime detector has never been tested on.

This experiment measures whether taking them changes anything. It is deliberately a **controlled
comparison**: same five assets, same four strategies, same engine, same construction functions
(`clean.align_calendars`, `clean.compute_log_returns`, `ml_features.build_ml_feature_set`), built
under a separate filename so the committed `log_returns_etf.parquet` and every number already in
the deliverables stay untouched. **Only the window differs.**

Three effects, separated because they are easy to conflate:

| Effect | Question | Design |
|---|---|---|
| **A** | Does more *training* history make the strategies better? | Both universes scored on the **same** OOS window (2018+) |
| **B** | Does more *test* history tighten the CIs? | Each universe scored on its **own** full OOS window |
| **C** | Is the weight cap, not the model, choosing the portfolio? | Cap swept 0.25 → 1.00, everything else fixed |

Effect C was **not planned** — it was forced by an anomaly in Effect A, described below.

## Effect B — the intended result, and it worked

| Strategy | CI width, `etf_2017` (8.9 yr OOS) | CI width, `etf_2005` (21.4 yr OOS) | Change |
|---|---:|---:|---:|
| `equal_weight` | 1.13 | 0.69 | **−39.3%** |
| `min_variance_lw` | 1.15 | 0.72 | **−37.1%** |
| `max_sharpe` | 1.12 | 0.71 | **−37.0%** |
| `regime_conditional` | 1.15 | 0.70 | **−38.7%** |
| **mean** | **1.137** | **0.705** | **−38.0%** |

**Extending the window cut confidence-interval width by 38%.** This is the cleanest positive result
the project has produced on the statistical-power question, and it confirms the diagnosis: the
recurring "everything is statistically indistinguishable" verdict is driven by sample size, and
sample size is partly a choice we made rather than a constraint we were handed.

The ML-vs-classical verdict also moves in the right direction on this universe, though it does not
flip: `regime_conditional` goes from **−4.0%** vs. the best classical strategy on the short window to
**−1.6%** on the long one. Still a loss on `etf_2017` — reported as such.

## Effect C — the unplanned discovery, and the more important one

Effect A produced an impossible-looking result: on the deep universe, `min_variance_lw`,
`max_sharpe` and `regime_conditional` returned **byte-identical weights** after 2018
(`max|diff| = 4.4e-16`). Three genuinely different objectives cannot agree to machine precision by
chance, so this was investigated rather than reported.

It is not a coding error. **It is arithmetic.**

> With 5 assets and a 25% cap, `5 × 0.25 = 1.25`. Any feasible long-only portfolio summing to 1
> must therefore hold **at least four assets at the cap**. The optimizer's only remaining freedom
> is *which one to drop*. Every objective lands on the same corner.

The cap sweep is the causal test — hold everything fixed, vary only the cap:

| Cap | `min_variance_lw` distinct allocations | `max_sharpe` distinct allocations | min-var Sharpe |
|---:|---:|---:|---:|
| **0.25** (project default) | **1** | 12 | 0.953 |
| 0.30 | 169 | 127 | 0.939 |
| 0.40 | 248 | 224 | 0.911 |
| 0.60 | 248 | 248 | 0.847 |
| 1.00 | 248 | 248 | 0.846 |

At the project's own cap, **`min_variance_lw` produces exactly one allocation across all 248
rebalances.** The Ledoit-Wolf covariance estimate is computed, and then discarded by the constraint.
Loosening the cap by five percentage points restores 169 distinct allocations.

The same degeneracy is present, less completely, on the committed `etf_2017` universe:
`max_sharpe` sits at the degenerate corner in **91.3%** of its 103 rebalances, with only 10 distinct
allocations ever produced.

### What this means for the project's standing conclusions

Phase 4 concluded that regime/dynamic-covariance ML "adds no value on `etf_2017`", and explained it
as the ETF universe having less mispricing left to correct. **That explanation is at best incomplete.**
The more mundane cause is that on a 5-asset universe with a 25% cap, the constraint set leaves almost
no room for *any* optimizer to express a view — so of course a better covariance estimate cannot show
up in the results. It has nowhere to go.

This confounds every `etf_2017` comparison in the project since Phase 2. It does **not** affect
`full_2021`: with 9 assets, `9 × 0.25 = 2.25`, so a feasible portfolio needs only 4 of 9 at the cap
and the optimizer retains real freedom — which is consistent with `regime_conditional` genuinely
differentiating there (+14.3%).

### A second-order finding worth noting

The constrained portfolio has the **highest** Sharpe in the sweep (0.953 at cap 0.25 versus 0.846
uncapped). The cap is acting as a powerful regularizer — it is doing the estimation-error control
that the ML was introduced to do. This is the Jagannathan & Ma (2003) result, reproduced
incidentally on our own data: imposing a binding weight constraint on a mean-variance optimizer is
mathematically equivalent to shrinking the covariance matrix.

## Effect A — reported, but not to be trusted

| Strategy | `etf_2017` training | `etf_2005` training | Δ |
|---|---:|---:|---:|
| `equal_weight` | 0.830 | 0.831 | +0.001 |
| `min_variance_lw` | 0.823 | 0.958 | +0.135 |
| `max_sharpe` | 0.930 | 0.958 | +0.028 |
| `regime_conditional` | 0.893 | 0.958 | +0.066 |

These look like clean improvements from extra training history. **They are not interpretable**, for
the reason Effect C establishes: on the deep universe all three optimizers collapsed onto the same
cap corner, so "0.823 → 0.958" is not "a better covariance estimate", it is "noisy corner-hopping
replaced by one stable corner". The three identical `0.958` values are the tell.

Effect A must be re-run at a non-binding cap before any claim is made about training history.
Recorded here rather than quietly dropped.

## Conclusions

1. **The ETF extension delivers on its purpose.** −38% CI width, at the cost of one config change.
   Worth adopting.
2. **A confound has been sitting under `etf_2017` since Phase 2.** The 25% cap on a 5-asset universe
   is close to fully determining the allocation, which makes "ML doesn't help here" unfalsifiable as
   stated. This needs to be recorded in the project's conclusions, not buried.
3. **`full_2021` is unaffected** — 9 assets give the optimizer genuine freedom, so the headline
   +14.3% result stands.

## ADOPTED — 2026-07-25, same day

`ingest.start_date` is now **`2004-11-18`** in production. Re-ingested, re-cleaned, re-baselined.

| | Before | After |
|---|---:|---:|
| `etf_2017` rows (Silver) | 2,493 | **5,656** |
| OOS rebalances | 103 | **248** |
| OOS window | 8.5 yr | **20.7 yr** |
| `equal_weight` max drawdown | COVID-era | **−36.2%** (2008 now in sample) |
| `etf_2017` hurdle | `max_sharpe` 0.928 | **`min_variance_lw` 0.953** |

The hurdle numbers reproduce the experiment's predictions exactly (min-var 0.953), which is the
check that the production path and the experiment agree.

`full_2021` is deliberately unchanged (1,321 rows): it is truncated to 2021-07 by BVC availability
regardless of the requested start, and the pipeline says so loudly rather than silently — *"Calendar
alignment is dropping 6069 days … because ['IAM.CS', 'ATW.CS', 'CIH.CS', 'BCP.CS'] have no data
before that point."*

**Two things checked rather than assumed:**

- **The universe key stays `etf_2017`.** Renaming it would break the params / DVC / dashboard
  contracts for no analytical gain. The true window lives in the validation report and is displayed
  in the dashboard.
- **Macro coverage does not silently degrade the models.** Extending to 2005 leaves three macro
  columns partly NaN (`DXY` starts 2006, `EURMAD` similar, and `TAUX_DIR` only reaches 2017 because
  it is a hand-maintained BAM decision list — §17.2). Verified that the HMM's three inputs
  (`MARKET_RETURN`, `MARKET_VOL_SHORT`, `AVG_PAIRWISE_CORR`) are **NaN-free across the whole
  2005–2026 span**, and that neither the regime model nor F7 reads the affected macro columns. The
  gap is real, pre-existing, and now merely visible; it affects macro EDA only.

Also fixed in passing: `src/ingest.py` never called `load_dotenv`, so running it standalone (a
documented entry point) failed on the macro step with a confusing `EnvironmentError` while
`FRED_API_KEY` sat in `.env`. `pipeline.py` had been masking it.

## SETTLED — the `etf_2017` question, answered at a non-binding cap

*2026-07-25. `experiments/etf_cap_verdict.py` → `data/gold/etf_cap_verdict.json`.*

Effect C established that "regime/covariance ML adds no value on `etf_2017`" was **not falsifiable
as stated** — the 25% cap left the optimizer nowhere to express a view. That is an objection to the
*test*, not evidence for either answer. This run removes the objection by sweeping the cap and
re-asking the question where the optimizer is genuinely free.

**Three outcomes were pre-registered in the script's docstring before it ran**, so the result could
not be rationalised after the fact: (A) ML wins once the cap loosens → the Phase 4 conclusion was a
constraint artefact; (B) ML loses at every free cap → the conclusion survives a stronger test;
(C) everything degrades as the cap loosens → the cap is the dominant driver.

| Cap | Best classical | Classical | `regime_conditional` | Lift | Optimizer free? |
|---:|---|---:|---:|---:|---|
| **0.25** | `min_variance_lw` | **0.953** | 0.937 | −1.6% | **NO — capped** |
| 0.30 | `min_variance_lw` | 0.939 | 0.931 | −0.9% | yes |
| 0.35 | `min_variance_lw` | 0.932 | 0.897 | −3.8% | yes |
| 0.40 | `max_sharpe` | 0.912 | 0.883 | −3.2% | yes |
| 1.00 | `max_sharpe` | 0.865 | 0.832 | −3.8% | yes |

*"Optimizer free" = `min_variance_lw` produced more than a handful of distinct allocations, i.e. the
cap was not dictating the answer. The verdict logic ignores caps where it was.*

### Outcome B — and the result is now stronger than before

**Regime ML loses at every cap where the optimizer is free**, on 248 rebalances over 20.7 years
including 2008. The Phase 4 `etf_2017` conclusion was correct; it was simply resting on an argument
that could not have been wrong. It is now **defensible rather than merely unfalsifiable** — which is
a better position than it was in this morning, even though the number did not move.

Note the lift is *most* negative at the loosest caps (−3.8% at both 0.35 and 1.00) and least
negative at the binding 0.25 (−1.6%). The cap was partly *masking* the gap by forcing every strategy
toward the same corner. Giving the model freedom did not help it; it exposed more of the shortfall.

### Outcome C also holds, and is the more interesting finding

**The tightest cap is the best-performing configuration for every strategy.** Best classical Sharpe
runs 0.953 at cap 0.25 monotonically down to 0.865 unconstrained — a **10% Sharpe swing driven
purely by the constraint**, larger than any modelling difference measured anywhere in this project
on this universe.

This is Jagannathan & Ma (2003) reproduced on our own data: imposing a binding weight constraint on
a mean-variance optimizer is mathematically equivalent to shrinking the covariance matrix, and on a
5-asset universe with noisy estimates that shrinkage is worth more than any of the covariance
models we built.

**The uncomfortable implication, stated plainly:** on `etf_2017`, the 25% cap — chosen in Phase 2 as
a *realistic management constraint*, not as a modelling device — is doing more estimation-error
control than Ledoit-Wolf, EWMA, DCC-GARCH or the HMM regime switch. That is a real result about
where the value lies, and it belongs in the deliverable rather than in a footnote.

`full_2021` is unaffected throughout: 9 assets and a 25% cap leave the optimizer real freedom, which
is consistent with `regime_conditional` genuinely differentiating there (+6.2%).

## Recommended follow-ups

- ~~Adopt the deep ETF window~~ — **done, above.**
- ~~Re-run the `etf_2017` conclusions at a non-binding cap~~ — **done, this section. Outcome B.**
- **Consider reporting the cap as a modelling choice.** Its effect on this universe (10% Sharpe)
  exceeds every model's. Worth stating in the Phase 2 constraint discussion rather than treating
  25% as a fixed given.
- **Re-run the `etf_2017` conclusions at a non-binding cap** (0.35–0.40), or state the confound
  explicitly wherever the "no ML benefit on `etf_2017`" claim appears.
- **Reconsider the cap as a modelling choice, not just a constraint.** At 25% on 5 assets it is the
  dominant driver of performance. It deserves to be reported as such — and its regularizing effect
  is a legitimate finding in its own right.
- **The BVC deep history remains blocked on dividends** — see the dividend-adjustment analysis;
  unlike this ETF extension, it is *not* methodologically free.

## Reproducing

```bash
.venv/bin/python experiments/etf_deep_history.py    # ~9 min, artifact: data/gold/etf_deep_history.json
```

Prices are cached at `data/bronze/etf_deep_prices.parquet` after the first run, so repeat runs are
offline and deterministic.
