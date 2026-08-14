# What this evaluation does not measure well

**Status:** examiner-facing. Every claim below is derived from a committed artifact.
**Date:** 2026-08-11

This document exists because an independent review asked five questions the repository
could not answer. Four are recorded here; the fifth was a correctness defect and was fixed
(see `docs/MODEL_INTEGRITY.md` and the tag `pfa-defense-ready-fallback-telemetry`).

Each section states the mechanism, what was measured, what follows, and — where it applies —
what was deliberately **not** done and why.

---

## 1. The covariance input is calendar-distorted on the mixed universe

**Mechanism.** Casablanca closes hours before New York, and `align_calendars` forward-fills
BVC prices across Moroccan holidays. A daily BVC return can therefore reflect information
that US markets priced the day before.

**Measured** (`docs/NONSYNC_COVARIANCE.md`, `data/gold/nonsync_covariance.json`):

| group | corr *t* | corr *t−1* | ratio |
|---|---:|---:|---:|
| Non-US (4 BVC names) | 0.0041 | 0.0779 | **19.1×** |
| US-listed, same universe | 0.4902 | −0.0287 | −0.06 |
| `etf_2017` control (single calendar) | 0.3762 | −0.0373 | −0.099 |

Yesterday's SPY explains a BVC return ~19× better than today's does. BVC zero-return days
run 13–21% against 0.6% for the ETFs. `CIH.CS` has AR(1) −0.26 and a 5-day variance ratio of
0.443 — its daily-annualised volatility overstates its weekly-scaled volatility by ~48%, so a
minimum-variance optimizer under-weights it for a microstructure reason rather than an
economic one.

Re-estimating the covariance three ways through the unmodified engine changes the allocation
at **100% of rebalances** on `full_2021`, requiring ~17–18% of the portfolio to be traded to
move between them, with single weights shifting up to 20pp against a 25% cap.

**What follows.** Ledoit-Wolf, EWMA and DCC-GARCH were compared on how precisely they estimate
a *daily* covariance, while that daily covariance is itself calendar-distorted on the universe
containing BVC assets. This **limits what can be inferred from the covariance-model ladder**
on `full_2021`. `etf_2017` is single-calendar and shows none of it.

**What it is not.** No bound on achievable performance. No such quantity is estimated, and a
paired test does not establish that any of the three estimators outperforms another
(`weekly_lw` p = 0.304, `dimson_lw` p = 0.442).

---

## 2. Every reported Sharpe is excess-of-zero, and the numéraire-matched rate differs per universe

**Mechanism.** `params.yaml` carries `risk_free_annual: 0.0`, flagged "revisit in Phase 5";
the project is at Phase 8. Sharpe falls by `rf / σ`, which penalises **low-volatility**
strategies more, so the ranking is not invariant to the choice.

**Measured** (`data/gold/risk_free_sensitivity.json`). The project already ingests the Bank
Al-Maghrib policy rate, so this uses the realised rate rather than a chosen one: over
`full_2021`'s out-of-sample window `TAUX_DIR` averaged **2.5505%** (range 1.50–3.00%, 100%
coverage).

`full_2021` (MAD — `TAUX_DIR` is the numéraire-consistent rate):

| strategy | ann. return | ann. vol | rf = 0 | rf = 2.55% *(matched)* | rf = 3.00% |
|---|---:|---:|---:|---:|---:|
| `max_sharpe` | 11.95% | 11.18% | 1.0690 | **0.8408** | 0.8006 |
| `regime_conditional` | 9.45% | 9.87% | 0.9571 | **0.6988** | 0.6533 |
| `equal_weight` | 9.55% | 10.02% | 0.9528 | 0.6983 | 0.6534 |
| `min_variance_lw` | 8.39% | 9.56% | 0.8782 | 0.6112 | 0.5642 |

The regime-vs-best-classical gap runs **−10.47%** at rf = 0 and **−16.89%** at the matched
rate. Under the zero-rate assumption the relative gap is the **least unfavourable to
`regime_conditional`** that the grid produces; the matched rate makes it more unfavourable.
The ranking also changes at rf = 3.00%, where `equal_weight` overtakes `regime_conditional` —
the reviewer's rank-invariance point, made concrete.

**The direction is not universal, and that is the instructive part.** On `etf_2017` the gap
*narrows* as the rate rises (−1.62% → −1.11%), because there `regime_conditional` has the
**higher** volatility (11.22% vs `min_variance_lw`'s 10.89%), so the rate penalises its
competitor more. The ordering there is unchanged across the whole grid.

**Numéraire discipline.** `etf_2017` is USD and this project ingests no USD risk-free series.
`TAUX_DIR` is **not** substituted into it: subtracting a MAD policy rate from USD returns
would be the same class of currency error the base-currency correction existed to remove. For
that universe the grid is a sensitivity curve, explicitly labelled as such in the artifact,
not a corrected Sharpe.

**Deliberately not done.** `risk_free_annual` also enters the `max_sharpe` **objective**, so
genuinely adopting a non-zero rate means re-optimising, not re-scoring — rate timing,
annual-to-daily convention, instrument choice, and a full rebuild. That is a methodology
change and is recorded here as a scope limitation rather than attempted late.

---

## 3. The regime feature is smoothed inside the training window

**Mechanism.** `regime.predict_regime_posterior_series` calls `hmmlearn`'s
`model.predict_proba`, which returns **smoothed** posteriors — γ_t = P(state_t | x_1…x_T) from
the forward–backward algorithm.

For the bull/bear dispatch this is harmless: only the last row is read, and at t = T the
smoothed posterior equals the filtered one. For `ml_signals.attach_regime_feature` it is not.
Every historical row's `REGIME_BULL_PROB` at date *t* was computed using observations after
*t* (up to τ), while at inference the same column is a filtered posterior.

**What this is, precisely.** An in-window lookahead and a **train/serve mismatch**: the model
trains on a cleaner version of a feature than the one it is scored with.

**Measured** (`tests/test_regime_feature_smoothing.py`). Holding the fitted model **fixed**
and varying only the window length handed to `predict_proba` — so the difference is
attributable to the backward pass and nothing else — the effect is real but modest and highly
localised: **max drift ~1×10⁻³** in probability units, **fewer than 10 of 141 dates move at
all**, and the movement sits in the final rows of the window (last-10 mean 2.2×10⁻⁴ against a
first-100 mean of 9×10⁻¹⁶). That is the signature of forward-backward smoothing: the backward
pass carries most information where fewest future observations exist, while older rows are
already pinned by everything that followed them.

The isolation matters, and getting it wrong was the first attempt at this measurement. The
obvious diagnostic — call `attach_regime_feature` on a short window and a long one — conflates
smoothing with the **refit**, since that function re-estimates the HMM. A refit on more data
moves the posteriors by ~0.7 in probability units, some of it merely relabelled states. That
number would have overstated this defect by nearly three orders of magnitude.

**What it is not.** It does **not** inflate the out-of-sample results. Nothing after τ enters
the evaluation — the engine's slicing guarantees that, and the guarantee is tested. If
anything it degrades live performance, because the tree learns to over-trust a variable that
is noisier at inference time.

**Honest gap.** The project's causality tests stopped just short of this.
`test_future_returns_do_not_change_past_asset_features` covers `build_asset_features` — pure
rolling windows — and does not extend to where the non-causal column is attached. The regime
feature is therefore described as **exploratory** and this is not presented as a passing
causality guarantee. The fix is filtered posteriors (a rolling `predict_proba` over expanding
prefixes) or dropping the column from training rows; both change F7 results and neither is
attempted this late.

**How it is now marked.** `tests/test_regime_feature_smoothing.py` is a *diagnostic*, not an
`xfail`. Its five tests pass by measuring the dependency, showing it is concentrated where
the backward pass predicts, bounding its magnitude, proving the measurement is not an artefact
of recomputation, and keeping the confinement result adjacent. An expected-failure test would
have gone green while the defect stayed unmeasured; this one **fails if the defect is ever
fixed**, and says so — a test that must be deleted when a bug is closed is a more honest
marker than one permitted to fail forever.

---

## 4. DCC-GARCH is one step stale, and its horizon does not match the holding period

Two separate defects in `dcc_garch._dcc_covariance_uncached`, both confirmed by inspection.

**Off-by-one.** `sigma_t = sigmas[-1]` is `arch`'s `conditional_volatility[T-1]`, which is
conditional on information through τ−1 and does not incorporate the most recent shock.
Likewise the DCC recursion stops at Q_τ. The object wanted at rebalance τ is the one-step-ahead
forecast σ²_{τ+1|τ} = ω + α r²_τ + β σ²_τ, i.e. `result.forecast(horizon=1)`; `grep` confirms
no `forecast(` call exists anywhere in the module.

This introduces **no lookahead** — it is conservative — but it discards the one property GARCH
exists for: reacting to yesterday's shock. The most sophisticated risk model in the project is
deliberately one day deaf.

**Horizon mismatch, the larger of the two.** Rebalancing is monthly (`ME`) with a ~21-day hold,
but the optimizer is fed a *1-day* conditional covariance scaled by 252. GARCH mean-reverts
toward its unconditional level over 21 days, so this systematically over-weights the current
volatility state relative to what the portfolio actually experiences. The correct object is the
average of the 1…21-step forecast variances. The same critique applies more mildly to
`MinVarianceEWMA`.

**Status.** Diagnosed, **not quantified**. Both are candidate explanations for why the DCC rung
never paid, and neither has been measured — correcting them changes `dcc_garch` results and
requires a re-run. Stated as an open question rather than a finding.

---

## 5. Statistical power, stated up front rather than discovered

`test_frac: 0.35` leaves ~460 frozen-test days on `full_2021`. Intervals containing zero were
the predictable consequence of the design, not a discovery about the models. The project
addresses this directly — `docs/ETF_DEEP_HISTORY_EXPERIMENT.md` extends the ETF window to 2004
and `docs/NESTED_WALKFORWARD_EXPERIMENT.md` narrows the intervals by 28.6% — but a
minimum-detectable-effect calculation stated in advance would have framed the negative result
as **designed** rather than as a disappointment.

The standing wording rule follows from the same place: a point estimate is an *observed
difference*, an interval is *uncertainty quantification*, and neither becomes "superior" or
"equivalent" without a paired test of the difference.

---

## What was fixed rather than documented

The review's remaining finding was a correctness defect, not a limitation: the published
"0 fallbacks" integrity claim rested on a counter that could not increment. Six fallbacks were
found once the degradation paths were instrumented — all `regime_conditional` in its warm-up,
6.2% of that comparator's `full_2021` out-of-sample days. See `docs/MODEL_INTEGRITY.md`.

---

## Reproduce

```bash
.venv/bin/python experiments/nonsync_covariance.py
.venv/bin/python experiments/risk_free_sensitivity.py
```

Both are DVC stages with declared dependencies, so a rebuild of the underlying Gold data marks
them stale rather than leaving a silently outdated number behind.
