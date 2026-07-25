# Dividend Bias — The Headline Number Is Overstated

*Research note, 2026-07-25. Companion to `experiments/dividend_bias.py`.*
**Status: affects a shipped deliverable. Requires action before EURAFRIC review.**

## What was found

While investigating whether the investing.com deep BVC history could be spliced into production —
a step expected to be blocked by a dividend-adjustment mismatch — it turned out the mismatch
**already exists in the committed pipeline**:

| Asset class | Source | What the price series contains |
|---|---|---|
| ETFs (SPY, QQQ, EEM, GLD, TLT) | `yfinance`, `auto_adjust=True` | **Total return** — dividends reinvested |
| BVC (IAM, ATW, CIH, BCP) | `BVCscrap`, `feature="Value"` | **Price only** — dividends discarded |

Verified two independent ways:

1. `src/ingest.py:275` requests `feature="Value"`, the raw closing price, and no dividend
   adjustment is applied anywhere downstream in `clean.py` or `features.py`.
2. Over the 2021–2024 overlap, the committed BVC returns and the raw (definitely unadjusted)
   investing.com prices produce the **same CAGR to within 0.02%** — they are the same price-only
   series. For IAM the two agree to four significant figures (−10.81% both).

So `full_2021` — the flagship universe, and the one behind the stakeholder dashboard's headline —
systematically understates its four Moroccan assets by their dividend yield. Per
stockanalysis.com's ratios page, those yields are:

| Asset | Annual dividend yield | Source |
|---|---:|---|
| `ATW.CS` | 3.8% | published |
| `CIH.CS` | 4.5% | published |
| `BCP.CS` | 4.1% | published |
| `IAM.CS` | 5.5% | **estimate** — not published on stockanalysis.com |

## Why it does not cancel out

The instinct is that a common bias affects all strategies equally and washes out of a relative
comparison. It does not, because the strategies differ in how much they are *allowed* to avoid the
affected assets:

- **`equal_weight` is forced** to hold ~1/9 in each understated asset (44.4% BVC on average).
- **The optimizers are free** to underweight them — and do, precisely because the missing dividend
  makes those assets look worse than they really are.

So the optimizers get credit for dodging an artefact of our own data handling. The bias inflates
every optimizer's measured advantage over `equal_weight` — which is exactly the comparison the
dashboard headlines.

The mechanism is visible in the weights: once dividends are restored, `max_sharpe` moves from
42.5% to **50.2%** BVC exposure, and `regime_conditional` from 40.9% to 45.0%. They were avoiding
those assets for a reason that was not real.

## The measured impact

Correction method: add each BVC asset's dividend yield back as a constant daily log accrual,
`log(1+y)/252`. Log-returns are additive, so this reproduces the annual yield exactly over the
window. It is an approximation of timing (real dividends are lumpy ex-date events) but unbiased in
total return — the right approximation for measuring a systematic drift.

| Strategy | As committed | Dividend-corrected | Δ |
|---|---:|---:|---:|
| `equal_weight` | 0.9805 | 1.1755 | **+0.1950** |
| `min_variance_lw` | 0.9032 | 1.0933 | +0.1901 |
| `max_sharpe` | 0.9549 | 1.1319 | +0.1770 |
| `regime_conditional` | 1.1206 | 1.2581 | **+0.1375** |

`equal_weight` gains the most and `regime_conditional` the least — exactly the asymmetry predicted
above.

**Headline claim:**

| | Lift of `regime_conditional` over best classical |
|---|---:|
| As published (dashboard, README, CLAUDE.md, Phase 4 deliverable) | **+14.3%** |
| Dividend-corrected | **+7.0%** |

**The published figure is roughly double the defensible one.**

## Robustness — the conclusion does not rest on the estimated IAM yield

IAM's yield is the one number that had to be estimated, and IAM is a large holding, so the
correction was swept across the plausible range:

| Assumed IAM yield | Resulting headline lift |
|---:|---:|
| 0.0% (correct only the three *published* yields) | **+8.2%** |
| 3.0% | +8.2% |
| 5.5% (best estimate) | **+7.0%** |
| 7.0% | +6.3% |

**The published +14.3% is not reproduced at any plausible IAM yield** — not even at zero, where
only the three independently-published yields are corrected. The finding is robust to the one
judgement call it contains.

## What this does and does not change

**The core conclusion survives.** The regime + dynamic-covariance system still beats classical
Markowitz on `full_2021` after correction — by **+7.0%** instead of +14.3%. This is a smaller,
honest number, not a reversal. It is also still well inside the confidence intervals reported in
Phase 5, so it changes no significance claim (there were none to change — all CIs already
overlapped).

**What must change** is every place the +14.3% (or the earlier "+15%") figure appears:

- `dashboard/pages/1_Histoire_de_valeur.py` — the headline metric and narrative
- `data/gold/dashboard_showcase.json` — regenerated automatically once the fix lands
- `README.md` — the suite section
- `CLAUDE.md` — §5 status table, §13
- `docs/Livrable_Phase6-7_Suite_Portfolio_ML.docx` and the Phase 4 deliverable

Because the dashboard derives its numbers from Gold artifacts rather than hardcoding them (enforced
by `tests/test_run_dashboard_data.py`), fixing the *pipeline* fixes the dashboard automatically.
That design decision is what makes this a one-place fix instead of a hunt.

## Recommended fix, in order

1. **Correct the ingestion, not the presentation.** Add dividend handling to `src/ingest.py` so BVC
   prices become total-return series consistent with the ETFs. Options, best first:
   - **Per-period dividend-per-share from a published source**, applied on the ex-date. Most
     correct; needs a BVC dividend calendar (AMMC filings, or `casablanca-bourse.com`'s corporate
     actions section).
   - **Constant-yield accrual** as used in this experiment. Materially better than the status quo,
     approximate in timing, trivially implementable, fully documented.
   - **Switch the ETFs to price-only** (`auto_adjust=False`) so both sides are consistently
     price-only. Internally consistent and simple — but it throws away real ETF returns and makes
     every absolute Sharpe understated. Not recommended; listed for completeness.
2. **Re-run** `phase2_hurdle`, `phase4_compare`, `phase5_compare`, `dashboard_data`.
3. **Restate the headline** everywhere, and record the correction in the phase notes rather than
   silently swapping the number.
4. **Then** revisit the BVC deep-history splice — which is no longer blocked by a *mismatch*, since
   both sources are price-only, but is blocked by the same underlying gap: neither is total return.

## Note on how this was found

This was not on the plan. It surfaced because the ETF deep-history experiment required checking
whether investing.com prices were comparable to the committed series — and the answer, that they
were *identical*, was the tell that the committed series was not what it was assumed to be.

Two lessons worth keeping:

- **"Adjusted close" is a property of a specific request, not of a data source.** The pipeline
  docstring says "Downloads raw adjusted-close prices", and that is true of the yfinance path and
  false of the BVCscrap path. The docstring was written once and generalised silently.
- **A common bias across assets does not imply a common bias across strategies.** Anything that
  changes what the optimizer *wants to hold* affects constrained and unconstrained strategies
  differently.

## Reproducing

```bash
.venv/bin/python experiments/dividend_bias.py   # ~2 min, artifact: data/gold/dividend_bias.json
```
