# Crisis-window behaviour — the P3 evidence the project was missing

*2026-07-30. `experiments/crisis_windows.py` → `data/gold/crisis_windows.json`.*

## Why

**P3** — *"cross-asset correlations spike during crises, eliminating diversification exactly when
it is most needed"* — is one of the four problems this project exists to address, and it had the
least direct evidence. Every phase reported whole-period Sharpe, drawdown and Calmar. None reported
what happened **during the crises themselves**.

That is a real gap for the brief's *"pertinence financière des résultats"*: for an institution
allocating capital, behaviour in a drawdown is the financially relevant question, arguably more so
than an average Sharpe. The data to answer it was already sitting in `dashboard_equity.parquet`
and `dashboard_regime.parquet` — no re-running of anything was required.

## Method

Five crisis windows, defined by **external, published S&P 500 peak-to-trough dates**, fixed before
any result was inspected:

| Window | Dates | S&P drawdown |
|---|---|---|
| Global Financial Crisis | 2007-10-09 → 2009-03-09 | −56.8% |
| EU sovereign debt crisis | 2011-04-29 → 2011-10-03 | −19.4% |
| Q4 2018 selloff | 2018-09-20 → 2018-12-24 | −19.8% |
| COVID-19 crash | 2020-02-19 → 2020-03-23 | −33.9% |
| 2022 rate shock | 2022-01-03 → 2022-10-12 | −25.4% |

Deriving windows from *our own* portfolios' drawdowns would be circular — it selects the periods
where a strategy happens to look good and calls that a finding. Drawdown is measured against the
running **all-time** peak, not the within-window peak, because that is what an investor experiences.
All four strategies are reported for every window.

## Result A — the optimizers protect; 1/N does not

`etf_2017`, net of costs. Ordered by cumulative return within each window.

| Crisis | Strategy | Return | Max DD | Recovery |
|---|---|---:|---:|---:|
| **GFC 2008** | `regime_conditional` | **−20.9%** | **−26.9%** | 185 d |
| | `min_variance_lw` | −21.2% | −26.9% | 186 d |
| | `max_sharpe` | −23.5% | −32.8% | 184 d |
| | `equal_weight` | **−30.2%** | **−36.2%** | 219 d |
| **EU debt 2011** | `regime_conditional` / `min_variance_lw` | **+1.6%** | −4.4% | 1 d |
| | `max_sharpe` | +0.7% | −4.6% | 1 d |
| | `equal_weight` | **−5.4%** | −6.7% | 11 d |
| **Q4 2018** | all three optimizers | −7.6% | **−9.0%** | **53 d** |
| | `equal_weight` | −7.9% | **−12.4%** | **105 d** |
| **COVID 2020** | all three optimizers | −13.5% | **−16.2%** | **37 d** |
| | `equal_weight` | −16.9% | **−19.1%** | **71 d** |
| **2022 shock** | all three optimizers | −24.6% | −25.0% | 506 d |
| | `equal_weight` | −25.3% | −25.9% | 545 d |

**The consistent finding is optimizer-vs-1/N, in all five windows.** The constrained optimizers
lose less, draw down less, and recover faster than equal weighting every single time. The clearest
cases are the two deepest crises: in the GFC the optimizers saved **9.3 percentage points** of
return and **9.3 points** of drawdown against 1/N; through the EU debt crisis they were *positive*
(+1.6%) while 1/N lost 5.4%. Recovery is the most consistent margin of all — 53 d vs 105 d in 2018,
37 d vs 71 d in COVID: roughly **half the time underwater**.

### The honest caveat: `regime_conditional` adds little over `min_variance_lw`

In three of five windows the three optimizers are **identical to the decimal**. Two mechanisms
explain it, and both are already documented elsewhere in this project:

1. **`regime_conditional` *becomes* `min_variance_lw` in a bear regime** — that is its
   `bear_strategy` by construction. Since the HMM flags 92% of crisis rebalances as bear (below),
   during a crisis the two strategies are largely the *same portfolio*, not two competing ones.
2. **Cap degeneracy** (CLAUDE.md §10.1): with 5 assets and a 25% cap, `5 × 0.25 = 1.25` forces
   ≥4 assets to the cap, so `max_sharpe` collapses onto the same allocation too.

So the correct claim is **"constrained optimization protects in crises; 1/N does not"** — a P1/P3
result about the *constraint and the covariance model*, not a win for the regime layer. The one
window where the regime layer does distinguish itself is the GFC, where it beats both
`min_variance_lw` (−20.9% vs −21.2%) and `max_sharpe` (−23.5%) — a real but small margin over
17 rebalances.

## Result B — the unsupervised HMM detected every crisis ⭐

This is the stronger half, and it is the kind of claim the project has not been able to make
anywhere else.

The regime detector is **unsupervised**. It has never been shown a crisis date, a recession label,
or any external event — it sees only `MARKET_RETURN`, `MARKET_VOL_SHORT` and `AVG_PAIRWISE_CORR`,
and it labels each rebalance **causally**, from past data only, without knowing a crisis is under
way.

| | bear rate | rebalances |
|---|---:|---:|
| **Inside** crisis windows | **91.7%** | 36 |
| Outside | 29.2% | 212 |

**Risk ratio 3.13×**, and **all 5 of 5 crises** exceed the calm-period base rate:

| Crisis | bear rebalances |
|---|---|
| Global Financial Crisis | 15/17 (88%) |
| EU sovereign debt | 6/6 (100%) |
| Q4 2018 | 2/3 (67%) |
| COVID-19 | 1/1 (100%) |
| 2022 rate shock | 9/9 (100%) |

### Significance — and this project can finally claim some

Two tests, bracketing the serial-dependence problem:

- **Conservative (lead with this): sign test, each crisis as ONE observation, n=5 → p = 0.031.**
  This discards all within-crisis information and asks only whether each distinct episode beat the
  base rate. Serial correlation inside a window cannot inflate it.
- Liberal: Fisher exact over all 248 rebalances → p = 7.8 × 10⁻¹³, odds ratio 26.6. This treats
  serially-correlated monthly regimes as independent draws and is **optimistic — do not quote it
  alone**.

Even at the conservative bound this clears 5%. **It is, as far as I can tell, the first
statistically significant finding in the project** — every Sharpe comparison to date has had
overlapping confidence intervals.

### The caveat that must travel with it

"Bear" is *defined* as the HMM state with the lower fitted mean `MARKET_RETURN`, and crises are
low-return periods by construction. Some association is therefore definitional, and it would be
dishonest to present this as if the model had discovered crises from nothing.

The non-trivial content is that the detection is **real-time and causal**: at each rebalance the
model has only past data, has no idea a crisis is beginning, and still assigns bear at 3× the base
rate — while *not* crying wolf constantly (29.2% baseline, not 80%). That is a statement about
timing and specificity, not about labelling hindsight.

`full_2021` is too short to say anything: 4 crisis rebalances, one window, sign-test p = 0.5.

## What this changes

1. **P3 now has direct evidence** rather than an inference from whole-period drawdown.
2. **The defensible claim is about constrained optimization, not the regime layer** — 1/N is
   materially worse in every crisis, and that is a P1/P3 result worth leading with.
3. **The regime detector has independent validation.** Its portfolio *contribution* remains
   statistically indistinguishable from the baselines (Phase 5, nested walk-forward), but its
   *detection* is now demonstrably real. Those are separate claims and should be stated separately:
   the model sees what it claims to see; whether acting on it pays is a different, harder question
   the project has answered honestly in the negative.
4. **Recovery time is the most consistent margin** and appears nowhere in the current deliverable —
   roughly half the time underwater versus 1/N, in the two windows where all optimizers otherwise
   tie. Worth surfacing to a business audience, for whom time-to-recover is intuitive in a way
   Sharpe is not.

## Limitations

- Five windows is five observations. The portfolio comparisons (Result A) carry **no significance
  test at all** and are descriptive; only the regime-detection frequency (Result B) is tested.
- `etf_2017` only. `full_2021` starts mid-2022 and covers one partial crisis.
- The strategy set is the dashboard's four; F7 signal models are not included because
  `dashboard_equity.parquet` does not carry them.
- Crisis windows are S&P 500-defined, which is the right reference for a majority-ETF universe but
  not for the BVC assets in `full_2021`.
