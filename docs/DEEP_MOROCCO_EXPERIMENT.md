# Deep Moroccan Data Experiment — Was the ML Signal Starved?

*Research note, 2026-07-23. Companion to `notebooks/deep_morocco_data_expansion.ipynb` and
`experiments/deep_morocco_starvation.py`.*

## Motivation

Phase 5 (see `CLAUDE.md` §12D) evaluated the F7 return-prediction models out-of-sample with leak-free
tuning, a frozen test set, and block-bootstrap confidence intervals, and found them **statistically
indistinguishable** from a regime-switching baseline on both universes. That result has two possible
readings:

1. **The signal is genuinely absent** — the market is efficient, ML return prediction cannot add value.
2. **The signal was *starved*** — the `full_2021` universe has only 9 assets and ~1,300 rows, and the
   Phase 5 test window was ~1.7 years, so the confidence intervals were far too wide to detect a real
   effect.

Reading (2) is testable, and it points at a concrete fix: **more data**. This experiment runs that test.

## Data sourcing (what we found and used)

The binding constraint has always been that the free BVC source (`BVCscrap` / medias24) only reaches
back to ~mid-2021. A web search for deeper Moroccan equity data established:

- **No free *API*** serves the Casablanca banks' pre-2021 history — `yfinance` has none of the `.CS`
  names (only Maroc Telecom via its Euronext Paris cross-listing, `IAM.PA`, back to 2004), and the one
  affordable API candidate (**EODHD**) does not cover the Casablanca exchange.
- **investing.com** offers free, per-stock daily historical downloads (CSV, free account) covering the
  full BVC universe back to each name's IPO — capped at **5,000 rows per download**.

On 2026-07-22 the team downloaded daily histories for **17 Casablanca-listed stocks plus the MASI
index** from investing.com. The raw CSVs live under `data/bronze/morocco_investing/` (gitignored, like
all of `data/`).

**Data caveats (documented, not hidden):**
- Prices are investing.com **unadjusted** close in MAD (dividend/split adjustment is a production
  concern; it does not affect the starvation question).
- The **5,000-row free cap** truncates the oldest names around 2024, so the assembled window ends
  **2024-05**. Splicing to today via `BVCscrap` (the two sources overlap 2021–2024) is a production
  follow-up, not needed here.
- Trading calendars differ slightly across the 17 files; they are aligned to a business-day reference
  with capped forward-fill, exactly as `clean.py` handles the existing BVC data.

## The universe

From the 17 stocks, the **DEEP** universe keeps the **12 names with continuous history from 2005**,
spanning banking (ATW, BCP, CIH, BOA), telecom (IAM), cement (LHM, CMA), mining (MNG), steel (SID),
consumer (CSR), energy (GAZ) and insurance (WAA):

| Metric | Current (`full_2021`) | Deep Morocco |
|---|---|---|
| Assets | 9 | **12** |
| History | 2021-07 → today (~5 yr) | **2005 → 2024 (~20 yr)** |
| Pooled panel rows | ~11,700 | **56,184** (≈ 5×) |
| OOS test window | ~1.7 yr | **6.75 yr** (2017-08 → 2024-05) |
| Contains 2008 crisis | no | **yes** |

## Methodology

The *exact* Phase 5 honest-evaluation machinery, reused unchanged, so the bar is identical:

- **Stage A (the headline, no backtest):** purged + embargoed K-Fold cross-validation
  (`purged_kfold.py`) selecting RF/XGB hyperparameters scored by **information coefficient** (Spearman
  rank-correlation of predicted vs. realized next-period returns). This directly measures predictive
  skill without any portfolio or cost assumptions — the cleanest test of "was the model under-fed?".
- **Stage B:** a frozen held-out test (final 35%), evaluating the tuned RF/XGB (with **fixed** levers
  `shrink=0.5, penalty=1.0` — deliberately *no* lever grid, which is what made an earlier run
  intractable) alongside `regime_conditional`, `equal_weight`, and `max_sharpe`, each with a
  block-bootstrap 90% Sharpe CI.

Deterministic (all seeds fixed); `experiments/deep_morocco_starvation.py` reproduces the numbers and
writes `data/gold/deep_morocco_results.json` + `deep_morocco_equity.parquet`.

## Results

**Stage A — the model got sharper.** The information coefficient rose from Phase 5's **0.015–0.036** to
**~0.068 (RF) / ~0.074 (XGB)** — a ~2–4× increase, confirmed independently by both algorithms. An IC
around 0.07 is genuinely "usable" territory in cross-sectional equity ML. *The signal was, in part,
starved.*

**Stage B — the portfolio did not.** On the held-out 2017→2024 window (net of 30bps costs):

| Strategy | Test Sharpe | 90% CI |
|---|---:|---|
| RF (tuned) | +0.34 | [−0.41, +1.14] |
| XGB (tuned) | +0.28 | [−0.51, +1.10] |
| 1/N | +0.25 | [−0.56, +1.15] |
| Markowitz (max-Sharpe) | +0.18 | [−0.53, +0.93] |
| regime-switching | +0.07 | [−0.65, +0.82] |

The tuned ML strategies are the **best point-estimate performers** — they edge classical Markowitz and
the regime baseline — but every Sharpe is low and **every CI is wide and straddles zero**: nothing is
statistically significant. And the ranking *flips* versus Phase 5 (where regime led and ML trailed),
the signature of noise rather than a durable edge.

## Verdict

> Giving the models ~5× the data made the **model measurably smarter** (IC roughly tripled) but did
> **not** produce a statistically significant **portfolio** edge. The ceiling is not data *quantity* —
> it is the fundamental difficulty of converting weak, real return-predictability into a significant,
> stable, net-of-cost edge in a small, illiquid, high-cost single market. *Prediction accuracy ≠
> portfolio performance*, even with deep data and a fair, rigorous test.

## Why this is useful, and what's next

- **Rules out** "just get more price history" as the fix, with evidence — no more effort wasted there.
- **Redirects** the alpha search to the one input never tried: **data *quality*** — *fundamentals*
  (P/E, earnings, book value, sector), the actual drivers of cross-sectional returns.
- **Strengthens the honest-win story**: the ML's point-estimate advantage over classical Markowitz now
  holds across the original universe *and* 20 years of deep Moroccan data.

**Next experiment:** fundamentals (data quality, not quantity).

---

## ⚠️ CORRECTION 2026-09-15 — re-run with the Phase 1 instruments

Everything above is kept as the honest record of what this experiment concluded in July. Two
of its claims do not survive re-examination, and the wording predates the 2026-08-02 reframing
(AGENTS.md §5.2). Read this section as authoritative where the two conflict.

### It mostly reproduces — but not entirely

Re-run 2026-09-15 from the restored raw CSVs (`experiments/deep_morocco_starvation.py`,
unchanged). The universe rebuilds exactly: 12 tickers x 4,682 days, 2005-01-06 -> 2024-05-31,
zero NaNs. The ICs match (RF 0.0684, XGB 0.0751), and so do 1/N (0.2525), Markowitz (0.1782)
and regime (0.0669).

**XGB does not.** It came out at **0.2238**, against **+0.28** above. That is large enough to
overturn a qualitative claim: at 0.2238, tuned XGB sits *below* 1/N (0.2525), so "the tuned ML
strategies are the best point-estimate performers" holds for RF alone. The likely cause is
library drift (xgboost 3.2.0 now), which is exactly what a hashed input artifact would let us
confirm or exclude — and neither the raw CSVs nor the assembled panel is hashed anywhere.

### The "more data is ruled out" claim is withdrawn

The bullet "**Rules out** 'just get more price history' as the fix, with evidence" is **too
strong, and is retracted.**

It rested on CIs straddling zero, which says nothing about whether the design could have
detected the effect it was looking for. Measuring that (Ledoit-Wolf HAC standard error of the
Sharpe *difference*, then the minimum detectable effect at 80% power, alpha = 0.10) on the
frozen test window, n = 1,638:

| Comparison | Observed | MDE (80%) | Observed / MDE | p |
|---|---:|---:|---:|---:|
| `rf_tuned` vs `regime_conditional` | 0.263 | 0.461 | **0.57** | 0.142 |
| `rf_tuned` vs `max_sharpe` | 0.159 | 0.483 | 0.33 | 0.343 |
| `rf_tuned` vs `equal_weight` | 0.082 | 0.410 | 0.20 | 0.584 |
| `xgb_tuned` vs `equal_weight` | −0.025 | 0.419 | 0.06 | 0.875 |

On the shallow `full_2021` window the same ratio runs 0.01-0.24
(`docs/EVALUATION_LIMITS.md` §5). Here it reaches **0.57**. More data did not fail to help — it
more than doubled the evidence relative to the detection threshold. It simply did not carry it
across. "Did not cross the threshold" and "ruled out" are different statements, and only the
first is supported.

Holding the observed effect fixed, crossing needs the MDE down to ≈ 0.263, a factor of
(0.461/0.263)^2 ≈ 3.1 more data: roughly **5,045 out-of-sample days, about 20 years**. A large
ask, not an impossibility — which is precisely what "ruled out" denies.

### What can now be asserted, and what still cannot

Model Confidence Set (size 0.10) retains **all five** strategies; nothing is eliminated.
Romano-Wolf StepM names **0 of 4** superior to `equal_weight`. So the headline conclusion is
unchanged — no outperformance is established — but it is now a statement about resolving power
rather than an inference from overlapping marginal intervals.

**The probability of backtest overfitting cannot be computed from this experiment.** CSCV needs
the out-of-sample series of every searched configuration; `deep_morocco_equity.parquet` retains
five strategies, not the 15 CV configurations (6 RF + 9 XGB). Obtaining it means re-running with
per-configuration series kept. This matters, because PBO on the shallow data showed a
degradation slope near −1, and whether that milder on a 20-year panel is the open question this
experiment is best placed to answer.

**A related understatement.** `n_search_trials` records **5**, taken from the DSR ledger, while
the search actually evaluated 15 CV configurations. The reported DSR of 0.6841 is therefore
corrected for a multiplicity three times smaller than the real one — the same understatement
`docs/MULTIPLE_TESTING.md` exists to avoid.

### Wording

Per AGENTS.md §5.2, "statistically significant" is retracted project-wide and older text is to
be fixed when touched. In the Results and Verdict above, read "nothing is statistically
significant" and "did not produce a statistically significant portfolio edge" as **no candidate
was established as superior in a paired test of the difference** — which is now literally true,
and measured, rather than inferred from CI overlap.

## Limitations

Unadjusted prices; 5,000-row cap (window ends 2024, splice-to-today deferred); levers fixed (this run
answers the data question, not the tuning one); a single held-out window. None change the structural
significance finding, which is consistent with Phase 5 on entirely different data.
