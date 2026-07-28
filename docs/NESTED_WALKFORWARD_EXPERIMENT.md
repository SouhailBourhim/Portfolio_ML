# Nested walk-forward on `full_2021` — buying statistical power

*2026-07-28. `experiments/nested_walkforward.py` → `data/gold/nested_walkforward_results.json`.*

## Why

Phase 5's own limitations section (CLAUDE.md §12D) named this as the honest next step:

> the `full_2021` test window is short (~1.75 yr / ~455 rows) — its DSR of 0.67 and very wide CIs
> partly reflect that; a nested walk-forward with periodic re-selection (the documented stretch)
> would use the whole window as OOS and is the honest next step if a tighter `full_2021` verdict
> is wanted.

Every `full_2021` comparison this project has made sits inside intervals ~2.2 Sharpe wide. That is
not a modelling problem and no better model fixes it — it is a **sample-size** problem. This
experiment attacks it directly.

**Scoped to `full_2021` deliberately.** `etf_2017`'s frozen test segment grew from 3.3 to 7.6 years
when the deep window was adopted, and its intervals narrowed 28% in the 2026-07-27 rerun. It is no
longer the constrained case; running it again would double the cost to re-answer a question it has
stopped asking.

## Design

Instead of selecting once and testing once on the final 35%, re-select the F7 configuration at each
of six outer boundaries and test on the segment immediately after it, then concatenate every
out-of-sample segment into one continuous series.

```
single split : |------------ select ------------|===== test (455 rows) =====|
nested       : |-- select --|== t1 ==|
                |------ select -------|== t2 ==|
                 |---------- select ---------|== t3 ==|  ...  concatenated (793 rows)
```

Selection never sees its own evaluation window: each fold's hyperparameters (purged K-Fold, scored
by information coefficient) and portfolio levers (validation net Sharpe) come strictly from data
before that fold begins. The concatenated series is therefore honestly out-of-sample **of
selection**, while covering 60% of the universe instead of 35%.

Six folds of ~126 trading days, OOS running **2023-07-12 → 2026-07-24 = 793 rows** against the
single split's 455 (**1.74×**). Baselines are evaluated on the identical concatenated dates.

**Pre-registered outcomes** (fixed in the module docstring before the run):

- **(A)** intervals narrow materially AND the ranking is unchanged → Phase 5's verdict was sound and
  is now better powered.
- **(B)** intervals narrow materially AND the ranking changes → the single-split verdict was a
  window artefact.
- **(C)** intervals do not narrow → the limitation is not sample length.

## Result — outcome (B)

| Strategy | Single split (455 rows) | width | **Nested (793 rows)** | width | narrowing |
|---|---:|---:|---:|---:|---:|
| `regime_conditional` | 1.213 [0.18, 2.35] | 2.167 | **1.672** [0.89, 2.51] | **1.612** | **25.6%** |
| `xgb_signal_tuned` | **1.308** [0.22, 2.48] | 2.260 | 1.436 [0.70, 2.26] | 1.558 | 31.1% |
| `min_variance_lw` | — | — | 1.417 [0.59, 2.29] | 1.703 | — |
| `max_sharpe` | — | — | 1.398 [0.66, 2.23] | 1.577 | — |
| `equal_weight` | 1.003 [−0.09, 2.24] | 2.333 | 1.284 [0.48, 2.18] | 1.705 | 26.9% |
| `rf_signal_tuned` | 1.040 [0.05, 2.17] | 2.125 | 1.256 [0.57, 2.04] | 1.472 | 30.7% |

**Mean interval width 2.221 → 1.587, a 28.6% reduction** — beating the 24.3% that pure √n scaling
predicts, because the concatenated segments span more distinct market conditions than one
contiguous tail does.

Deflated Sharpe of the winner against the whole search: **0.835 over 198 configurations** (the
single split's was 0.67 over 36). More configurations tried AND a higher DSR is the good direction:
the search got broader and the winner still survived deflation.

### Three things worth separating

**1. The interval narrowing is the robust result.** It is a property of the design, not of the
period, and it is the thing the experiment was built to deliver.

**2. Every lower bound is now positive.** In the single split, `equal_weight`'s 90% interval ran to
**−0.090** and `rf_signal_tuned`'s to 0.048 — neither could be called reliably profitable. Nested,
the lower bounds run 0.478 to 0.893. That is a genuine change in what can be claimed: on this
universe, every strategy examined is now credibly positive out-of-sample.

**3. The level shift is a PERIOD effect, not an improvement — do not quote it as one.** Every
strategy's point Sharpe rose (regime 1.213→1.672, xgb 1.308→1.436, rf 1.040→1.256, EW 1.003→1.284).
The nested window starts 2023-07 and the single split starts 2024-10, so the extra 15 months were
simply good ones for this portfolio. Nothing got better; the measurement covers a different span.
**Only within-run comparisons are meaningful.**

### The ranking flipped back — outcome (B)

Within the nested run, `regime_conditional` leads at **1.672**, ahead of `xgb_signal_tuned` at
1.436. The single split had the opposite order (xgb 1.308 > regime 1.213).

That is now **three different orderings observed on `full_2021`** across three evaluations:

| Evaluation | Leader |
|---|---|
| Phase 5, pre-dividend-correction | `regime_conditional` |
| Phase 5, corrected, single 35% split | `xgb_signal_tuned` |
| **Nested walk-forward (most OOS data)** | **`regime_conditional`** |

The honest reading is not "regime wins after all". It is that **the point ordering on this universe
is unstable to evaluation design**, which is the same conclusion Phase 5 reached, now demonstrated
a third time. The nested estimate deserves the most weight — 74% more out-of-sample data, honest
per-fold re-selection, the narrowest intervals — but `regime_conditional` [0.89, 2.51] and
`xgb_signal_tuned` [0.70, 2.26] still overlap across almost their entire range. **No significance
claim is available in either direction.**

What *has* improved is the defensibility of the project's headline choice: the dashboard presents
`regime_conditional` as "our system", and the best-powered evaluation available now puts it first
rather than second.

### Per-fold detail

| Fold | OOS window | RF levers | RF Sharpe | XGB levers | XGB Sharpe |
|---|---|---|---:|---|---:|
| 1 | 2023-07-12 → 2024-01-03 | shrink 0.25, λ 0.5 | +1.00 | shrink 0.25, λ 2.0 | +1.02 |
| 2 | 2024-01-04 → 2024-06-27 | shrink 0.75, λ 2.0 | +2.34 | shrink 0.25, λ 1.0 | +2.75 |
| 3 | 2024-06-28 → 2024-12-20 | shrink 0.50, λ 1.0 | +1.35 | shrink 0.25, λ 1.0 | +2.01 |
| 4 | 2024-12-23 → 2025-06-16 | shrink 0.25, λ 0.5 | +1.64 | shrink 0.25, λ 1.0 | +1.71 |
| 5 | 2025-06-17 → 2025-12-09 | shrink 0.25, λ 0.5 | +2.33 | shrink 0.25, λ 2.0 | +2.61 |
| 6 | 2025-12-10 → 2026-07-24 | shrink 0.25, λ 0.5 | **−0.00** | shrink 0.25, λ 0.5 | **+0.23** |

Two observations. The **selected levers move between folds** — RF's shrinkage ran 0.25/0.75/0.50/
0.25/0.25/0.25 and its turnover penalty 0.5/2.0/1.0/0.5/0.5/0.5 — which is Phase 4C's premise
confirmed again: a single global λ is wrong, and re-selection is doing real work rather than
converging on one answer. And **fold 6 is where both models collapse** (RF −0.00, XGB +0.23) on the
most recent seven months, while the concatenated total stays strong. A single-split evaluation
landing on that stretch would have reported a very different verdict — which is precisely the
fragility this design exists to average over.

## Limitations

- Six folds is a coarse re-selection cadence (~6 months). More folds would be more honest about
  drift but cost proportionally more; each fold is a full purged-CV grid plus a 16-point lever grid
  for both model families.
- Baselines (`regime_conditional`, `equal_weight`, …) have no hyperparameters and so are not
  re-selected per fold. That is not a handicap to them — but it does mean the F7 models pay a
  selection-noise cost the baselines don't. Arguably that cost is real and belongs in the
  comparison: needing to be tuned is a property of the model.
- The concatenated series stitches six segments; the block bootstrap treats it as one series, which
  slightly understates uncertainty at the seams. With 126-row segments and 21-day blocks the effect
  is small, but it is not zero.
- `full_2021` only. `etf_2017` was excluded on the reasoning above, so this says nothing about it.
