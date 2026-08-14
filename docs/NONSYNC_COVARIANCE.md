# Non-synchronous trading and the covariance input

**Experiment:** `experiments/nonsync_covariance.py`
**Artifact:** `data/gold/nonsync_covariance.json`
**Pre-registered outcome:** **A**
**Date:** 2026-08-11

---

## Why this exists

The covariance matrix is the one input every strategy in this project consumes —
`min_variance`, `min_variance_lw`, `min_variance_ewma`, `dcc_garch`, and the Sharpe
objective behind `max_sharpe` and both F7 signal models. The entire Phase 4 ablation
ladder is an argument about how best to **estimate** it.

That argument assumes the quantity being estimated is the one we mean. Casablanca closes
hours before New York, and `align_calendars` forward-fills BVC prices across Moroccan
holidays. Both push the same way: a daily BVC return may reflect information that US
markets priced the **day before**.

An external review raised this. Nothing in the repo had ever measured it. `grep` found no
mention of Epps, Dimson, Scholes–Williams or asynchrony anywhere.

---

## Stage A — the signature is unambiguous, and the control is clean

Cross-correlation against `SPY`, on `full_2021` (MAD, 1 301 rows):

| asset | corr *t* | **corr *t−1*** | corr *t+1* | AR(1) | % zero days | VR(5d) |
|---|---:|---:|---:|---:|---:|---:|
| QQQ | 0.9502 | −0.0573 | −0.0518 | −0.0564 | 0.6% | 0.787 |
| EEM | 0.6799 | −0.0591 | −0.0525 | −0.0997 | 0.7% | 0.766 |
| GLD | 0.1834 | 0.0065 | −0.0429 | −0.0561 | 0.6% | 0.683 |
| TLT | 0.1474 | −0.0050 | −0.0543 | −0.0985 | 0.6% | 0.823 |
| **IAM.CS** | 0.0090 | **0.0897** | 0.0301 | 0.0893 | **13.3%** | 1.097 |
| **ATW.CS** | 0.0287 | **0.0844** | 0.0301 | −0.0927 | **14.1%** | 0.820 |
| **CIH.CS** | −0.0183 | **0.0865** | 0.0155 | **−0.2555** | **20.8%** | **0.443** |
| **BCP.CS** | −0.0031 | **0.0510** | −0.0042 | −0.0786 | **20.2%** | 0.985 |

Grouped:

| group | mean corr *t* | mean corr *t−1* | ratio |
|---|---:|---:|---:|
| Non-US (4 BVC names) | **0.0041** | **0.0779** | **19.1×** |
| US-listed, same universe | 0.4902 | −0.0287 | −0.06 |

**Yesterday's SPY explains a BVC return roughly nineteen times better than today's does.**
Same-day correlation is essentially zero (0.004); the co-movement has moved into the lag.
For the US-listed assets *in the same universe, over the same days*, the ratio is small and
**negative** — the ordinary sign.

**The control.** `etf_2017` is five US-listed ETFs on one calendar. Its ratio is **−0.099**,
matching the US-listed block above. The signature does not appear where it should not, which
is what rules out "this is just noise" and fires outcome A rather than the pre-registered
refutation D.

Two supporting readings:

- **Zero-return days: 13–21% for BVC against 0.6% for the ETFs.** A fifth of `CIH.CS` and
  `BCP.CS` observations are mechanically flat — a price that did not move because nothing
  traded, not because nothing happened.
- **`CIH.CS` variance ratio 0.443** with AR(1) −0.26. This is the textbook bid–ask bounce:
  its daily-annualised volatility is 0.263 while its weekly-scaled volatility is 0.178, so
  the daily figure **overstates** its risk by roughly 48%. A minimum-variance optimizer
  under-weights `CIH.CS` for a reason that is microstructure, not economics.

---

## Stage B — it propagates to the allocation, on every single rebalance

Three covariance estimators through the **unmodified** engine — same objective, same 25% cap,
same optimizer, same cost model, same rebalance calendar. Only the covariance function varies.

`full_2021`, net of costs, 48 rebalances:

| estimator | net Sharpe | 90% CI | turnover | PSD projections |
|---|---:|:---:|---:|---:|
| `daily_lw` *(production)* | 0.8782 | [+0.12, +1.62] | 0.074 | 0 |
| `weekly_lw` | 0.9516 | [+0.25, +1.65] | 0.079 | 0 |
| `dimson_lw` | 0.8942 | [+0.16, +1.59] | 0.100 | 0 |

Allocation distance from production:

| estimator | mean turnover to switch | max | max single weight change | rebalances differing |
|---|---:|---:|---:|---:|
| `weekly_lw` | 0.1659 | 0.2271 | 0.1631 | **100%** |
| `dimson_lw` | 0.1806 | 0.2367 | 0.1989 | **100%** |

**Every rebalance changes.** Moving between the production estimator and either alternative
would require trading roughly **17–18% of the portfolio**, with individual asset weights
shifting by up to **20 percentage points** — against a 25% cap. That is not a rounding
difference; it is a different portfolio.

**The control, again.** On `etf_2017` all three estimators return **byte-identical** results
(0.9525, 0% of rebalances differing). This is the cap degeneracy already documented in
AGENTS.md §10.1 — with 5 assets and a 25% cap, `5 × 0.25 = 1.25` forces at least four
positions to the cap, so the constraint picks the portfolio and the covariance model has
nowhere to express a view. It is worth being explicit that **the control's Stage B is
therefore uninformative about the estimator**; the control's weight rests entirely on Stage A.

---

## Stage C — the allocation moves; the *performance* difference does not

Paired moving-block bootstrap of the difference against `daily_lw`, on `full_2021`:

| estimator | Δ Sharpe | 90% CI of the difference | p | P(Δ>0) |
|---|---:|:---:|---:|---:|
| `weekly_lw` | +0.0735 | [−0.1443, +0.2965] | 0.304 | 0.710 |
| `dimson_lw` | +0.0161 | [−0.2114, +0.2620] | 0.442 | 0.527 |

**No difference is established.** The point estimates favour the synchronisation-corrected
arms, and both probabilities lean the same way, but neither clears any reasonable bar. Per
this project's standing rule, that is **not** evidence the estimators are equivalent either —
it is an absence of a demonstrated difference, and the correct instrument was used to look.

> **Disclosure.** Stages A and B and the outcome rule were fixed before any number existed.
> Stage C was added *after* Stage A and B results had been seen, prompted by this project's
> own standing note that a paired test is the missing instrument (§12I.1). Its result was not
> known when it was added, and it does not feed the outcome letter — outcome A turns on the
> allocation sensitivity, which is measured directly. Recorded here rather than folded in.

---

## What this establishes, and what it does not

**Establishes.** The covariance input on `full_2021` is materially sensitive to
market-calendar and liquidity effects. The co-movement between Moroccan equities and US
markets sits predominantly at a one-day lag; a fifth of BVC observations are mechanically
flat; one name's daily volatility overstates its weekly-scaled volatility by ~48%; and
switching to either synchronisation-aware estimator changes the allocation at **100%** of
rebalances by ~17–18% of the portfolio.

**Therefore.** Conclusions drawn from the covariance-model ladder are conditioned on a
measurement choice that was never made deliberately. Ledoit-Wolf, EWMA and DCC-GARCH were
compared on how precisely they estimate a daily covariance — while the daily covariance is
itself calendar-distorted on the universe that contains BVC assets. **This limits what can be
inferred from the ladder** on `full_2021`.

**Does NOT establish.** That any estimator here is better than another — no paired test
supports that. And **nothing here is a bound on achievable performance.** This experiment
estimates no such quantity, and none may be quoted from it. It is a statement about the
sensitivity of an input, not about a ceiling on what a model could do.

**Also worth separating:** the `etf_2017` universe is single-calendar and shows none of this.
The finding is specific to the mixed universe, which is the one the EURAFRIC brief actually
targets.

---

## Honest limitations

1. **The weekly arm trades one bias for another.** Weekly returns over the same window carry
   one fifth the observations, so `weekly_lw` swaps synchronisation bias for estimation
   variance. It is a diagnostic, not a proposed replacement estimator, and is not offered as
   one. `dimson_lw` is the arm that keeps the daily sample size, which is why both are run.
2. **`dimson_lw` is truncated at one lag.** The hypothesis is a one-session offset between two
   exchanges, not long-memory dependence. A longer truncation is a different experiment.
3. **The PSD projection never fired** (0 projections in 48 rebalances on the target, 0 on the
   control), so the lead-lag correction stayed positive semi-definite on this data without
   needing rescue. The guard is retained because it is not guaranteed in general, and the
   count is reported in the artifact so a future run cannot hide it.
4. **Only `min_variance` was tested.** It isolates the covariance cleanly because covariance
   is its *only* input — no expected-return estimate is involved. Whether the same
   sensitivity propagates through `max_sharpe` or the F7 models, where a μ estimate also
   enters, is not measured here.
5. **`full_2021` is 48 rebalances.** The allocation-distance result is a direct measurement
   and does not depend on sample size; the Sharpe comparison does, and its intervals are
   correspondingly wide.

---

## Reproduce

```bash
.venv/bin/python experiments/nonsync_covariance.py
```

Deterministic — seeded bootstrap, no network. Writes `data/gold/nonsync_covariance.json`
with a full provenance block (numéraire, source-artifact hashes, git revision).
