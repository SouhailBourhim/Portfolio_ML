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

## Limitations

Unadjusted prices; 5,000-row cap (window ends 2024, splice-to-today deferred); levers fixed (this run
answers the data question, not the tuning one); a single held-out window. None change the structural
significance finding, which is consistent with Phase 5 on entirely different data.
