# Fundamentals Experiment — Does Data QUALITY Break the Signal Ceiling?

*Research note, 2026-07-23. Companion to `experiments/fundamentals_ic_lift.py`,
`experiments/fundamentals_portfolio.py`, `src/fundamentals.py`, `tests/test_fundamentals.py`.*

## Motivation

The deep-Morocco experiment (see [`DEEP_MOROCCO_EXPERIMENT.md`](DEEP_MOROCCO_EXPERIMENT.md)) closed
off "more price history" as the missing ingredient for F7: purged-CV information coefficient rose
2–4× with 5× the data, but portfolio Sharpe did not follow. Its explicit conclusion was that the
ceiling is data **QUALITY**, not quantity — which in practice means a new class of features. The
obvious candidate is **fundamentals**: point-in-time valuation ratios (P/E, P/B, P/S, D/E) that
encode information price history cannot.

This experiment tests exactly that:

> Take the existing F7 pipeline (Phase 4B, regime-conditioned RandomForest on the `full_2021`
> universe) and add point-in-time fundamentals as new per-asset features. Does the purged-CV IC
> rise? If yes, does portfolio Sharpe follow, or does the "prediction accuracy ≠ portfolio
> performance" ceiling hold a **third** time?

## Data — source, coverage, causal discipline

**Source.** [stockanalysis.com](https://stockanalysis.com), free tier, S&P Global-sourced. Its
`robots.txt` explicitly permits `User-agent: *`. The financial data blob is embedded in the
initial HTML as a JavaScript object literal (SvelteKit hydration payload), so no browser is
needed — a small tokenizer in [`src/fundamentals.py`](../src/fundamentals.py) reshapes it into a
tidy `(period_end, ticker, metric, value)` frame.

**Coverage that actually came through** (verified live, 2026-07-23):

| Metric | IAM | ATW | CIH | BCP | Notes |
|--------|:---:|:---:|:---:|:---:|-------|
| `pe` | ✓ | ✓ | ✓ | ✓ | valuation |
| `pb` | ✓ | ✓ | ✓ | ✓ | valuation |
| `ps` | ✓ | ✓ | ✓ | ✓ | valuation |
| `debtequity` | ✓ | ✓ | ✓ | ✓ | leverage |
| `roe`, `grossMargin`, `operatingMargin`, `profitMargin` | ✗ | ✗ | ✗ | ✗ | **annual-only for Moroccan issuers** |

**Frequency by ticker** — IAM reports semi-annually (10 periods 2021–2025); ATW, CIH, BCP report
quarterly (~20 periods each). Point-in-time forward-fill handles the mixed frequency uniformly.

**Causal discipline (§15.8-style, non-negotiable).** stockanalysis.com exposes only period-end
dates, not filing dates. Feeding a period-end value into a model at that period-end would be a
lookahead leak — Moroccan issuers file **60–90 days** after period-end under AMMC rules. The
implementation applies **90 business days** as a conservative fixed lag
(`fundamentals.apply_publication_lag`, `publication_lag_days=90`) to compute an `available_from`
date, and every downstream consumer sees only that. The
`test_future_value_corruption_cannot_change_any_past_panel_row` unit test locks this in end-to-end,
mirroring the `test_phase3_integration.py` guarantee that Phase 3 already established for market
features.

## Wiring

`ml_signals.attach_fundamentals_features` (new) adds fundamentals as **per-asset feature columns**
onto the F7 pooled `(Date, ASSET)` panel:

- **BVC assets** (IAM, ATW, CIH, BCP) — each row `(d, asset)` gets that asset's own PE/PB/PS/D/E
  at date `d`. Genuine per-asset signal.
- **ETF assets** (SPY, QQQ, EEM, GLD, TLT) — each row gets the **cross-sectional median** of the
  BVC assets' fundamentals at date `d`, plus a `HAS_FUND=0` indicator. This preserves the training
  row (dropping 5/9 of a pooled model's assets would be catastrophic) while letting the tree
  discover the asset-type distinction itself rather than us hard-coding it. Feature-importance
  analysis (below) confirms the tree does not lean on `HAS_FUND` — the ETFs' median-filled
  fundamentals are effectively ignored, which is exactly the intended behaviour.

The integration goes through the existing `extras={"fundamentals": ...}` channel — no engine
changes, no new strategy classes, no new tests broken (`pytest tests/ -q` → 319 passed).

## Stage A — IC lift (does the model learn more?)

`experiments/fundamentals_ic_lift.py`, purged-CV grid search on `full_2021`, apples-to-apples
(baseline restricted to the exact same `(date, asset)` rows as treatment — see the script
docstring for why the naive comparison was a confound):

| Algorithm | Baseline mean IC (Phase 5 setup) | Treatment mean IC (+ fundamentals) | Lift |
|-----------|:---:|:---:|:---:|
| RandomForest | **0.0281** ± 0.0196 | **0.0581** ± 0.0379 | **+0.0301** |
| XGBoost | **0.0291** ± 0.0370 | **0.0398** ± 0.0511 | **+0.0107** |

**RF's IC nearly doubled.** Both fold-stdevs straddle zero, so neither lift is significant at the
fold level, but the RF direction is consistent and its treatment IC of 0.058 is the highest ML IC
this project has measured on `full_2021`. Feature importance on the best RF confirms the tree
genuinely uses fundamentals:

```
PRICE_REL_MA_63D                   0.4170
FUND_pb                            0.1575   ← 2nd most important feature
RET_63D                            0.1134
RET_5D                             0.0912
RET_21D                            0.0767
VOL_21D                            0.0370
VOL_63D                            0.0287
FUND_pe                            0.0255
FUND_debtequity                    0.0249
REGIME_BULL_PROB                   0.0185
FUND_ps                            0.0095
HAS_FUND                           0.0000
                        ── fundamentals block: 21.7% of total importance ──
```

At this point in the deep-Morocco experiment, the same "IC rose but doesn't mean anything for
portfolio performance" narrative held. The point of Stage B is to test whether it holds again.

## Stage B — Portfolio verdict (does the alpha survive?)

`experiments/fundamentals_portfolio.py`, walk-forward backtest on the **frozen Phase 5 test window**
(2024-10-14 → 2026-07-20, 461 rows / 1.8 years), 4 strategies × 1 config (no lever grid — that is
what makes Phase 5 runs take hours; this experiment is scoped to a verdict, not a tuning pass).
Sharpes are annualized net-of-cost, with 90% block-bootstrap confidence intervals.

| Strategy | Net Sharpe | 90% CI | Avg turnover |
|---|---:|:---:|---:|
| `equal_weight` | 0.866 | [−0.20, 2.00] | 0.035 |
| `regime_conditional` | **1.199** | [0.19, 2.28] | 0.321 |
| `rf_signal_baseline` (F7, prices only) | **1.214** | [0.27, 2.20] | 0.774 |
| `rf_signal_fundamentals` (F7 + fundamentals) | 0.882 | [−0.09, 1.89] | 0.392 |

**`rf_signal` LIFT from adding fundamentals: −0.331 Sharpe.** All CIs overlap heavily; the lift is
**not statistically significant** at 90% confidence. But every point estimate — of a feature that
doubled the CV IC — is *worse* than the price-only baseline. Fundamentals cut the turnover roughly
in half (0.774 → 0.392, as expected from a slower-moving signal), and cut the Sharpe by nearly the
same fraction.

## Interpretation — the third confirmation of the ceiling

This is the third independent test of the "prediction accuracy ≠ portfolio performance" pattern
this project has now run:

| Experiment | What was scaled | CV IC change | Portfolio Sharpe change |
|---|---|---|---|
| Phase 5 (`CLAUDE.md` §12D) | Honest tuning of F7 on prices | baseline ~0.03 | statistically = regime |
| Deep-Morocco (§12E) | 5× more price history, 20 years | 2–4× higher (0.07) | no significant edge |
| Fundamentals (this doc) | Add a genuinely new data class | nearly 2× higher (0.058) | **point Sharpe DROPPED** |

The pattern is consistent enough now to be treated as an empirical finding of this project, not a
one-off:

- Better inputs (more data, more feature types) do make the F7 pooled-cross-sectional model measurably
  better at predicting next-period returns in IC terms — the ML machinery works.
- That improvement does **not** flow through to net Sharpe on a real walk-forward backtest with
  monthly rebalancing and 10/30bps costs. Something between the prediction and the portfolio —
  optimizer noise amplification (Chopra & Ziemba 1993), regime-boundary turnover, the SLSQP
  Sharpe objective's magnitude sensitivity — reliably absorbs the alpha.

**A single-figure specific note on this run.** `rf_signal_baseline` (my F7 without shrinkage or
turnover penalty) posted the highest point Sharpe of every strategy in the comparison, 1.214,
edging even the Phase 4 hurdle `regime_conditional` at 1.199. Phase 5's *tuned*
`rf_signal_shrunk` on the same test window scored only 0.785 — this run's own baseline shows that
Phase 5's selected `shrinkage_weight=0.5, turnover_penalty=1.0` were arguably over-regularized for
this specific test window. All three (baseline F7, tuned F7, regime) have overlapping CIs on this
1.8-year window, so the ranking is not to be trusted; the observation is that Phase 5's
"statistically indistinguishable" finding continues to hold, and one signal ordering can flip to
another purely on regularization-strength noise. This is a well-known small-sample fragility, not
a discovery, but it's worth noting in a research write-up.

## Consequences and where this leaves the project

**What is now empirically closed off, at this universe scale:**

1. More price history alone. (deep-Morocco)
2. Point-in-time valuation-ratio fundamentals alone. (this experiment)

**What is not tested and remains theoretically open** (but not currently prioritized — see the
"stopping" reasoning below):

- Non-valuation fundamentals (earnings surprise, revenue growth, quality metrics, margin
  compression). All of these need annual data or SEC-equivalent structured filings — for BVC,
  that's the AMMC PDF route, materially more expensive than this experiment's Bronze cache.
- Text/alt-data (news sentiment on the BVC, macro nowcasts specific to Morocco).
- Non-return targets (predicting volatility instead of return, then trading with a vol overlay).

**Recommendation on stopping.** Three consecutive negative or null portfolio results, on genuinely
different lever choices (data volume, data richness), is empirical evidence that the ML return-
prediction approach at this universe scale — 9 assets, monthly rebalancing, 10/30bps costs — has a
real ceiling near or below the regime-conditional baseline. Continuing to hunt for the next
feature class **before** the underlying diagnostic question is answered (which specific piece of
the ML → portfolio path is absorbing the alpha? optimizer? cost model? rebalance frequency?
constraints?) would be a P4 anti-pattern: keep testing until you get lucky with a positive result.

For the EURAFRIC deliverable, the **honest, defensible story** is now more complete than before:
- The **regime + dynamic-covariance system** genuinely and reproducibly beats classical Markowitz
  by ~15% net Sharpe on the target universe (Phase 4 hurdle, `full_2021`, verified in Phase 5).
- The **F7 return-prediction layer** does not add statistically significant value on top of that.
  Three independent tests support this reading. Reporting this honestly, with the CIs, IS the
  quality of the work — the alternative would be publishing whichever variant happened to peak on
  the test window (`rf_signal_baseline` here) and calling it the winner, which the confidence
  intervals do not support.

## Reproducing

```bash
# 1. Rebuild the fundamentals Gold panel (uses the committed Bronze HTML cache — no network needed
#    once the cache is warm; add --force-refetch for a fresh scrape).
.venv/bin/python src/fundamentals.py

# 2. Stage A: IC lift (~3 min)
.venv/bin/python experiments/fundamentals_ic_lift.py

# 3. Stage B: portfolio verdict (~4 min)
.venv/bin/python experiments/fundamentals_portfolio.py

# Artifacts (all gitignored, under data/gold/, DVC-eligible):
#   fundamentals_features.parquet
#   fundamentals_manifest.json
#   fundamentals_ic_lift.json
#   fundamentals_portfolio.json
```

All seeds fixed. All tests: `pytest tests/ -q` → 319 passed (20 new for this experiment).
