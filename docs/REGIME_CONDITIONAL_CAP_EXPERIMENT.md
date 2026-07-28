# Regime-conditional weight cap — a well-motivated hypothesis, falsified

*2026-07-28. `experiments/regime_conditional_cap.py` → `data/gold/regime_conditional_cap.json`.*

## The hypothesis, and why it was worth testing

This project's own evidence says the weight cap does more estimation-error control than any
covariance model it has tried. Sweeping `max_weight` alone on `etf_2017` moves the best classical
net Sharpe 0.9525 → 0.8650 — a 10.1% swing, larger than the gap between any two models on that
universe (CLAUDE.md §10.1). Jagannathan & Ma (2003) explains the mechanism: a binding long-only
weight constraint is mathematically equivalent to shrinking the covariance matrix.

Phase 4 conditions the **covariance model** on the detected regime. If the cap is the stronger
regularizer, the obvious untested move is to condition **the cap** on the regime instead — tighten
it in a detected bear (more shrinkage exactly when correlations spike and estimation error hurts
most, P1 and P3 together), relax it in a bull (let the optimizer express a view when it is likelier
to be right).

It required **no new production code**. `RegimeConditionalStrategy` already accepts sub-strategy
instances and each baseline owns its `max_weight`, so the whole experiment is
`MaxSharpe(max_weight=bull_cap)` + `MinVarianceLW(max_weight=bear_cap)`, with the engine handed the
looser of the two so its trust-boundary check still binds.

## The control that makes it honest

"Tighter bear cap wins" would be a weak finding alone — a tighter cap might simply be better
everywhere, with the regime label contributing nothing. So the grid includes:

- **fixed caps** at both endpoints (isolates *"the cap level helped"*), and
- an **INVERTED** variant — loose in bear, tight in bull, the deliberately *wrong* direction. If
  inverted also beats the baseline, there is no regime effect, only cap sensitivity.

**Pre-registered outcomes:** (A) a regime-conditional cap beats the 0.25 baseline materially AND
beats both the fixed caps AND the inverted control → real value; (B) it beats the baseline but so
does a fixed cap or the control → cap-level sensitivity, not a regime effect; (C) nothing beats the
baseline materially → the lever is spent.

**Materiality, disclosed.** The first draft tested "beats baseline" with a bare `>`. A smoke run
returned a candidate at **+0.0016** Sharpe against intervals ~1.16 wide, which that rule would have
reported as outcome A — a meaningless difference dressed as a finding. `MATERIAL_MARGIN = 0.05` was
added **before** the real run. The change makes the test *stricter*, and it is recorded here and in
the module docstring rather than quietly folded in, because adjusting a decision rule after seeing
output is exactly what pre-registration exists to prevent.

## Result — outcome (C) on both universes

### `full_2021` (9 assets, cap floor 0.111, baseline 1.2363)

| Variant | Sharpe | 90% CI | lift | turnover | |
|---|---:|:---:|---:|---:|---|
| `both_40_15` | 1.2663 | [0.51, 2.03] | **+0.030** | 0.341 | best candidate |
| `aggressive_40_25` | 1.2656 | [0.52, 2.03] | +0.029 | 0.312 | |
| `defensive_25_15` | 1.2379 | [0.47, 2.03] | +0.002 | 0.313 | |
| **`baseline_25_25`** | **1.2363** | [0.46, 2.01] | — | 0.293 | shipped |
| `defensive_25_125` | 1.2163 | [0.47, 2.00] | −0.020 | 0.319 | |
| `INVERTED_15_40` | 1.1430 | [0.36, 1.96] | −0.093 | 0.265 | **control** |
| `fixed_maxsharpe` @0.15 | 1.2333 | [0.51, 1.99] | −0.003 | 0.093 | best fixed |

### `etf_2017` (5 assets, cap floor 0.200, baseline 0.9371)

| Variant | Sharpe | 90% CI | lift | |
|---|---:|:---:|---:|---|
| `fixed_minvarlw` @0.25 | 0.9525 | [0.60, 1.32] | +0.015 | best fixed |
| **`baseline_25_25`** | **0.9371** | [0.59, 1.30] | — | shipped |
| `INVERTED_20_40` | 0.8899 | [0.54, 1.25] | −0.047 | **control** |
| `aggressive_40_25` | 0.8525 | [0.50, 1.22] | **−0.085** | best candidate |
| `defensive_25_20` | 0.8200 | [0.48, 1.17] | −0.117 | |
| `both_40_20` | 0.7496 | [0.40, 1.11] | −0.188 | |

**No regime-conditional cap clears the materiality margin on either universe.** On `etf_2017` every
candidate is *worse* than the fixed 0.25 baseline, several substantially.

## What the result actually says

**1. The hypothesised mechanism is not supported.** The whole idea was that *tightening in a bear*
would help. It does essentially nothing: `defensive_25_15` gains **+0.0016** on `full_2021` and
`defensive_25_20` **loses 0.117** on `etf_2017`. What little positive movement exists on `full_2021`
comes from *loosening the bull cap* (`aggressive_40_25`, +0.029) — the opposite half of the idea,
and still below materiality.

**2. The control gives a split verdict, which is itself informative.** On `full_2021` the INVERTED
variant is clearly the worst candidate (−0.093), consistent with the regime label carrying real
directional information. But on `etf_2017` **INVERTED (−0.047) beats the best "correct-direction"
candidate (−0.085)**. A regime effect that reverses sign between universes is not a regime effect.
Had only `full_2021` been run, the −0.093 control would have looked like supporting evidence; the
second universe is what prevents that misreading.

**3. The 0.25 cap is at or near its optimum on both universes.** Which retro-justifies a choice
originally made as a business rule rather than an optimization — a pleasant result, but note it was
*not* tuned to be optimal, so this is luck confirmed after the fact, not design.

**4. A clean confirmation of the cap-degeneracy arithmetic.** On `etf_2017` at cap 0.20,
`fixed_minvarlw_200` and `fixed_maxsharpe_200` return the **identical** 0.7694 — exactly
`equal_weight`'s Sharpe. With 5 assets, `5 × 0.20 = 1.0` forces every long-only portfolio to be 1/N,
so both optimizers collapse onto the same allocation regardless of their objective. This is the
degeneracy of CLAUDE.md §10.1 reproduced as an exact numerical identity.

**5. Regime switching itself still pays on `full_2021`.** Every fixed-cap reference underperforms
the regime baseline there (best fixed 1.2333 vs 1.2363, and `min_variance_lw` variants all ≈1.05–
1.07). That is the *existing* Phase 4 finding, unchanged — this experiment does not disturb it.

**Nothing here is statistically significant.** Every variant shares a window and the intervals
overlap almost completely (`full_2021`: 1.492 of ~1.53; `etf_2017`: 0.627 of ~0.71). The JSON
records `statistically_significant: false` explicitly for each universe.

## Where this leaves the search

A well-motivated hypothesis, derived from the project's own strongest empirical finding,
pre-registered, controlled, and **falsified**. That is a good outcome for the deliverable: it closes
a plausible avenue cheaply and on the record, rather than leaving it as an untested "we could
have…".

It also extends the pattern. The project has now tested five distinct routes to beating the
regime + dynamic-covariance baseline — F7 return prediction (Phase 5), 5× more price history
(deep-Morocco), point-in-time fundamentals, honest per-fold re-selection (nested walk-forward), and
regime-conditional constraints (here). **None produces a statistically significant improvement.**
The consistent finding is not that any single idea failed, but that on this universe and at this
sample size the evaluation cannot resolve differences of the size these ideas produce.
