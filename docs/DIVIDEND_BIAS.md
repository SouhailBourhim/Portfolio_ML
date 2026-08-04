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

## RESOLVED — the fix, and the final numbers

*Updated 2026-07-25, same day.* The correction is implemented and the whole pipeline
re-baselined. The approximation above has been replaced by the real thing.

**Dividend source.** `BVCscrap.getDividend` exists but is **broken** — it targets the legacy
`casablanca-bourse.com/bourseweb/Societe-Cote.aspx` endpoint, which now 307-redirects to a
redesigned site. `src/dividends.py` reads the modern per-issuer page instead, whose dividend table
is server-rendered into the HTML: per-share amount, type (Ordinaire / Exceptionnel), and — the part
that matters — the **ex-dividend date**. History reaches back to 2011–2013 per issuer.

**Cross-validation against the independent stockanalysis.com yields:**

| Ticker | Scraped, realised over 2021–2026 | Published (stockanalysis.com) |
|---|---:|---:|
| `ATW.CS` | 3.64% | 3.8% |
| `BCP.CS` | 4.17% | 4.1% |
| `CIH.CS` | 4.25% | 4.5% |
| `IAM.CS` | **3.03%** | not published (I had estimated 5.5%) |

The IAM estimate was **too high** — it has cut its dividend hard (9.26 in 2012 → 1.43 in 2025).
This is exactly why the sensitivity sweep was run rather than trusting one guess.

**Implementation.** `clean.compute_log_returns(prices, dividends=...)` computes
`r_t = ln((P_t + D_t) / P_{t-1})`, applying each payment on its ex-date — the same convention
`auto_adjust=True` already applies to the ETFs. `silver_pipeline(adjust_dividends=True)` is the
default; `False` reproduces the old numbers exactly for A/B. 61 dividends applied, 1 correctly
skipped as falling beyond the price window. 15 new tests in `tests/test_dividends.py`.

**Final corrected results (`full_2021`, net of costs, out-of-sample):**

| | Best classical | `regime_conditional` | Lift |
|---|---|---:|---:|
| As published | `equal_weight` 0.981 | 1.121 | **+14.3%** |
| **Corrected** | `max_sharpe` **1.163** | **1.238** | **+6.5%** |

Landing at +6.5% against the +7.0% the constant-yield approximation predicted, inside the
+6.3–8.2% sensitivity band. The approximation was sound; the exact treatment is now in the pipeline.

**A second finding fell out of the correction.** Once dividends are counted, **`max_sharpe`
(1.163) overtakes `equal_weight` (1.152)** on `full_2021`. The project has cited the DeMiguel et al.
(2009) "1/N beats the optimizers" result as reproduced on our own data since Phase 2 — **that
reproduction was an artefact of the missing dividends.** It does not survive the correction, and
every place it is claimed needs the same restatement.

**Propagation was automatic.** The dashboard and API picked up +6.47% with no code change, because
they derive from Gold artifacts and `tests/test_run_dashboard_data.py` forbids hardcoded figures.
Only English/French prose needed a manual edit.

## What this does and does not change

**The core conclusion survives.** The regime + dynamic-covariance system still beats classical
Markowitz on `full_2021` after correction — by **+7.0%** instead of +14.3%. This is a smaller,
honest number, not a reversal. It is also still well inside the confidence intervals reported in
Phase 5, so it changes no significance claim (there were none to change — all CIs already
overlapped).

**What must change** is every place the +14.3% (or the earlier "+15%") figure appears:

- `dashboard/pages/1_Resultats_recherche.py` — the headline metric and narrative
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
