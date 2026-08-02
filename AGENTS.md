# Portfolio ML — Project Context for Codex

> **Read this file entirely before writing any code, suggesting any change, or answering any
> question about this project.** It is the authoritative record of every decision made and why.
> It does **not** embed source code — the repo (`src/`, `tests/`) is the source of truth for
> code; this file is the source of truth for *decisions, status, and scope*. This version
> supersedes both the original code-embedding AGENTS.md and the interim `Codex.v2.md` (now
> stale — do not edit it further; it is kept only as history). Last synced against the actual
> repo state on **2026-08-02** (see §5 for how that sync was verified).

---

## Table of Contents

1. [Project Overview and Official Brief](#1-project-overview)
2. [Problem Statement — P1 to P4](#2-problem-statement)
3. [ML Architecture and Interventions](#3-ml-architecture)
4. [Team, Supervisor, and Scope](#4-team-and-constraints)
5. [Current Status by Phase](#5-current-status)
6. [Technology Stack](#6-technology-stack)
7. [Data Architecture — Medallion Pattern](#7-data-architecture)
8. [Asset Universe and Known Data Gaps](#8-asset-universe)
9. [Phase 1 — Data Infrastructure (Complete)](#9-phase-1)
10. [Phase 2 — Markowitz Baseline + Backtesting (Complete)](#10-phase-2)
11. [Phase 3 — ML Feature Engineering (Complete)](#11-phase-3)
12. [Phase 4 — ML Portfolio Models (Active — starting now)](#12-phase-4)
13. [Phases 5–6 — Forward Plan](#13-phases-5-6)
14. [File Structure](#14-file-structure)
15. [Non-Negotiable Technical Decisions](#15-non-negotiable-decisions)
16. [Coding and Testing Conventions](#16-conventions)
17. [Maintenance Duties (Recurring)](#17-maintenance-duties)
18. [Supervisor Requirements](#18-supervisor-requirements)

---

## 1. Project Overview

**Project name:** ML-Based Portfolio Optimization System
**Client:** EURAFRIC Information — financial technology company, Bouskoura, Morocco
**Institution:** INPT (Institut National des Postes et Télécommunications)
**Type:** PFA — Projet de Fin d'Année (end-of-studies internship project)
**Supervisor:** Abdelmouttalib
**Team:** Souhail Bourhim (repo author), EL WALI Zakarya, BOUAJINE Yasmine

### Official EURAFRIC brief (verbatim, French)

> Dans le cadre de ce challenge, les étudiants travailleront sur une problématique centrale de la
> finance quantitative : comment allouer intelligemment un capital entre plusieurs actifs
> financiers afin de maximiser le rendement tout en maîtrisant le risque. Ils concevront et
> implémenteront un système d'optimisation de portefeuille piloté par le Machine Learning, capable
> d'analyser des séries temporelles financières, d'apprendre les corrélations dynamiques entre
> actifs et de proposer automatiquement des stratégies d'allocation robustes, **sous des
> contraintes réalistes de gestion**. Le prototype devra être évalué dans un cadre rigoureux de
> backtesting sur des données hors échantillon, avec une attention particulière portée à la
> robustesse, au risque de surapprentissage et à la pertinence financière des résultats.

Every phrase of that brief maps to our problem framing (§2): dynamic correlations → P1,
time-series analysis → P2, robust allocation → P3, out-of-sample rigor / overfitting → P4.
*Realistic management constraints* became a first-class Phase 2 requirement (§10.1) — long-only,
per-asset weight cap, and net-of-cost reporting are not optional metrics, they are constraints the
optimizer and backtest operate under.

The final deliverable is a complete, documented, reproducible pipeline a professional portfolio
manager at EURAFRIC could use — not a research notebook. The binding constraint throughout every
phase is **out-of-sample robustness**: a model that performs well in-sample but fails
out-of-sample is worse than useless.

---

## 2. Problem Statement

Classical MPT fails in production due to four structural problems. Every function in this
codebase must trace back to one or more of them (docstring format: `Addresses: P1, P2 — ...`).

| ID | Problem | Core Issue | Consequence |
|----|---------|------------|-------------|
| **P1** | Noisy covariance estimation | Sample covariance amplifies estimation error, especially with many assets and short history | Concentrated, unstable portfolios that look optimal in-sample and collapse out-of-sample |
| **P2** | Non-stationarity | Returns are not IID; volatility clusters, regimes shift, fat tails violate Gaussian assumptions | Parameters estimated in one period are invalid in another |
| **P3** | Diversification breakdown in crises | Cross-asset correlations spike during crises, eliminating diversification exactly when it is most needed | Portfolio concentrates risk at the worst possible moment |
| **P4** | Backtest overfitting | Lookahead bias and repeated testing on the same data inflate apparent performance | Published Sharpe ratios are not reproducible live |

If you cannot identify which problem a piece of code addresses, that code is wrong or unnecessary.

---

## 3. ML Architecture

Four ML interventions layered on the leakage-free backtesting framework built in Phase 2:

### 3.1 Regime detection — HMM → P2, P3 (Phase 4, starting now)
Hidden Markov Models (`hmmlearn`) detect latent market regimes (bull/bear/crisis) from returns
without lookahead; portfolio weights are conditioned on the detected regime.

### 3.2 Dynamic covariance → P1, P2 (Phase 4, starting now)
Time-varying covariance replaces the static sample matrix. **Ablation ladder for P1** — each rung
isolates how much of the improvement comes from which intervention, and each rung already has a
working baseline to compare against from Phase 2:
1. Sample covariance (`MinVariance` — Phase 2, done)
2. **Ledoit-Wolf shrinkage** (`MinVarianceLW` — Phase 2, done, scikit-learn built-in)
3. EWMA / RiskMetrics covariance — Phase 4 MVP target
4. DCC-GARCH — Phase 4 stretch goal. ⚠️ Known implementation risk: `arch` provides *univariate*
   GARCH only — the multivariate DCC step must be hand-written (two-stage estimation) or sourced
   from a less-maintained package. Do not assume `arch` ships DCC out of the box.

### 3.3 Purged K-Fold Cross-Validation → P4 (Phase 5)
Purge gap between train and test folds eliminates leakage in hyperparameter selection.
**Decision:** custom implementation (~50 lines from López de Prado 2018, Ch. 7). Do **not** depend
on `mlfinlab` — it moved to a restricted/paid model and is a dependency risk.

### 3.4 Walk-forward backtesting → P4 (built in Phase 2, reused by every later phase)
Expanding-window retraining; at each rebalancing date the model trains only on data available at
that date. Implemented once in `src/backtest.py` behind the `Strategy` interface (§10.3) — Phase 4
models plug into this engine without touching it.

### Pipeline of phases

```
Raw prices + macro (Phase 1 ✅) → Markowitz + backtest framework (Phase 2 ✅)
→ Causal feature engineering (Phase 3 ✅) → ML models: HMM + dynamic covariance (Phase 4 ◀ NOW)
→ OOS evaluation: purged CV + walk-forward + deflated Sharpe (Phase 5)
→ Production: REST API + dashboard (Phase 6)
```

Phase 2 was the most structurally critical phase — its backtesting framework is what every Phase 4
result will be judged against. It is done, tested (no-lookahead suite green), and the Phase 3
feature seam has been proven to plug into it without leaking (§11).

---

## 4. Team and Constraints

**Supervisor style — shapes everything:**
- Agile, MVP-first: a complete end-to-end pipeline with simple models beats a half-finished
  sophisticated one.
- Traceability non-negotiable: every function states which problem(s) it addresses.
- Deep understanding over copy-paste: every team member must be able to defend every line.
- Deliverables in French; code and technical documentation in English.

**Hard out-of-scope:** real-time data feeds; cloud deployment; live trading or order execution.

**Scope updates (chronological):**
- *2026-06-29:* Orchestration back in scope, limited to scheduling the existing pipeline locally
  with Dagster (chosen over Airflow/Prefect: asset model maps onto medallion layers, single local
  process). Does not reopen cloud/multi-service infrastructure.
- *2026-07-03:* "Contraintes réalistes de gestion" promoted from metric to requirement (§10.1);
  dual-universe backtesting adopted (§10.2).
- *2026-07-20:* Phase 3 finalized and merged; a structural Dagster bug (the scheduled job never
  ran the ETF-only universe or Phase 3 features) found and fixed same-day (§17.7). Phase 4 begins.

---

## 5. Current Status

| Phase | Name | Status | Notes |
|-------|------|--------|-------|
| Phase 1 | Data Infrastructure | ✅ **Complete** (2026-07-02) | Gold layer validated on live data; walkthrough doc delivered |
| Phase 2 | Markowitz Baseline + Backtesting | ✅ **Complete** (merged PR #3, 2026-07-10) | Engine + 4 baselines + metrics + DSR + notebook; no-lookahead suite green. Net-of-cost hurdle: **etf_2017 → max_sharpe, Sharpe 0.936**; **full_2021 → equal_weight, Sharpe 0.975** (EW beats optimizers net on Universe B — the DeMiguel result, reproduced on our data) |
| Phase 3 | ML Feature Engineering | ✅ **Complete** (merged PR #4, #5, #6; last 2026-07-20) | 6 causal return features + 7 lagged macro signals, both universes (0 NaN when built; `etf_2017` gained 3,659 leading macro NaN on 2026-07-25 with the deep window — §11); Phase 2↔3 seam proven leak-free by `tests/test_phase3_integration.py`; French deliverable + notebook shipped |
| Phase 4 | ML Portfolio Models | ✅ **Core complete** (branch `feature/phase4-regime-covariance`, 2026-07-20) | HMM regime dispatch (2-state) + full covariance ablation ladder (Ledoit-Wolf → EWMA → DCC-GARCH) built, tested, and benchmarked against the Phase 2 hurdle on live data. Result: **beats the hurdle on full_2021** (`regime_conditional`, net Sharpe **1.122** vs. hurdle 0.975), **does not beat it on etf_2017** (best stays `max_sharpe` at 0.936 — no Phase 4 strategy cleared it there). Hardening pass shipped (PR #8) |
| Phase 4B | Adaptive ML Signal Models (F7) | ✅ **Core complete** (branch `feature/phase4b-adaptive-ml-signals`, 2026-07-21) | RandomForest + XGBoost pooled cross-sectional return-prediction signals (regime-conditioned via one input feature), feeding the existing Sharpe objective. Benchmarked against the Phase 4 hurdle on live data — **honest negative result: neither strategy beats it on either universe.** ⚠️ REFRESHED 2026-07-29 on corrected data — the conclusion survives and the margin WIDENED: `etf_2017` best stays `min_variance` 0.9525 (`rf_signal` 0.904, was 0.727; `xgb_signal` 0.694); `full_2021` best stays `regime_conditional` 1.2363 (`rf_signal` 1.123, was 1.062 — gap to hurdle grew from −0.060 to −0.113 because the correction lifted the baseline more than the signal; `xgb_signal` 0.873 with turnover still >1.0). LSTM built + tested standalone, dropped from the live comparison after a torch+xgboost segfault — deferred, not abandoned. Details — §12B |
| Phase 4C | Cost-Aware Optimization + μ Regularization | ✅ **Complete** (merged to `main` via PRs #11/#13, 2026-07-21) | Turnover-penalized objective + μ shrinkage/rank tilt + cov-estimator selection on the F7 strategies, attacking Phase 4B's diagnosed failure (rf_signal best gross Sharpe 1.240 lost to 0.885 turnover). ⚠️ **The "near-miss" did NOT survive the dividend correction.** Refreshed 2026-07-30: `rf_signal_shrunk` sits at **1.006 vs a 1.2363 hurdle — a 0.230 gap**, not the 0.4% near-tie originally reported (1.117 vs 1.1215). No variant beats the hurdle on either universe; the best F7 variant on `full_2021` is plain `rf_signal` at 1.123. Key finding: shrinkage works (Chopra-Ziemba confirmed), the turnover penalty backfired on the good model (`rf_signal_cost` collapsed to 0.704) yet helped the over-trader (`xgb_signal` 0.743→0.863) — proving a single global λ is wrong, a Phase 5 tuning task. Details — §12C |
| Phase 5 | Out-of-Sample Evaluation | ✅ **Core complete** (branch `feature/phase5-oos-evaluation`, 2026-07-22) | Custom purged+embargoed K-Fold CV (leak-free ML selection, IC-scored) + validation-segment lever selection + frozen held-out test + block-bootstrap Sharpe CIs + DSR accumulated over the whole search. **The punchline result: on held-out data with proper error bars, the honestly-tuned F7 models and `regime_conditional` show differences far smaller than the uncertainty on both universes** (⚠️ wording reframed 2026-08-02 — this was written as "statistically indistinguishable"; see §5.2 rule 2) — the earlier point-Sharpe comparisons were within the noise. The verdict even *flips* by universe (tuned XGB edges the hurdle on `etf_2017` test, 1.449 vs 1.373; regime clearly leads on `full_2021` test, 1.199 vs 0.886) but every bootstrap CI is wide and overlapping. DSR-vs-search: 0.92 (`etf_2017`), 0.67 (`full_2021`). Details — §12D |
| Phase 6+7 | Portfolio ML Suite (dashboard + API) | ✅ **Complete** (branch `feature/phase6-7-suite`, 2026-07-25) | **Merged into one deliverable** (user's call): a two-page Streamlit app under `dashboard/` + a thin FastAPI service under `src/api/`, sharing one data layer so the two pages can never disagree about a number. Page 1 (file `1_Histoire_de_valeur.py`) is the stakeholder page — headlines the **+6.2%** net-Sharpe *observed difference* between `regime_conditional` and classical Markowitz on `full_2021` (see §5.1), with Phase 5 CIs shown and the `etf_2017` **-1.6% loss** displayed, not hidden. ⭐ **Restructured 2026-08-01** — Page 1 now LEADS with crisis behaviour (§12I) and the Sharpe delta is demoted to supporting evidence, because the delta is not demonstrated and is metric-fragile. ⭐ **Reworded 2026-08-02 (§5.2)** — Page 1 is now *« Résultats de recherche »*, Page 2 *« Explorateur de stratégies »*, both carrying explicit research-prototype / non-advisory disclaimers; every claim of significance or of indistinguishability was removed. Numbers are structurally un-fakeable: `tests/test_run_dashboard_data.py` verifies every displayed figure derives from Gold artifacts, including a source-inspection test forbidding hardcoded Sharpe literals. 390 tests as of 2026-08-02. Details — §13 |
| ~~Phase 7~~ | ~~Stakeholder Value Dashboard~~ | ✅ **Merged into Phase 6+7 above** (2026-07-25) | Shipped as Page 1 of the unified suite rather than a separate app — same "honest win" framing as originally locked on 2026-07-22, same integrity constraints, but sharing Phase 6's data layer. |

### 5.1 ⚠️ DATA CORRECTION 2026-07-25 — supersedes the Sharpe figures in every row above

**Every net-Sharpe number in the phase rows above, and in §10.4/§12/§12B/§12C/§12D, was computed on
data that understated the four BVC assets by their dividend yield.** They are kept as the honest
record of what each phase found *at the time*; they are **not** the current numbers.

**The bug.** The pipeline mixed two return definitions: ETFs arrived dividend-adjusted
(`yfinance auto_adjust=True`, total return) while BVC arrived price-only
(`BVCscrap feature="Value"`), discarding 3.0–4.3%/yr of real dividends. Found while checking
whether investing.com's deep BVC history was comparable to the committed series — the answer, that
they were *identical*, was the tell.

**Why it mattered asymmetrically.** `equal_weight` is forced to hold the understated assets; the
optimizers are free to underweight them, and did — earning credit for dodging an artefact of our
own data handling. The bias inflated every optimizer's measured edge.

**The fix (shipped).** `src/dividends.py` scrapes per-share amounts *and ex-dates* from
casablanca-bourse.com (`BVCscrap.getDividend` is broken — dead endpoint), and
`clean.compute_log_returns(prices, dividends=...)` computes `ln((P_t + D_t)/P_{t-1})` on the
ex-date. Scraped yields cross-validate against stockanalysis.com within ~0.3pp. 15 new tests.

**Current, corrected headline numbers (`full_2021`, net, OOS):**

| | Best classical | `regime_conditional` | Lift |
|---|---|---:|---:|
| Pre-correction (all rows above) | `equal_weight` 0.975 → 1.121 | 1.121 | +14.3% |
| **Corrected — use these** | **`max_sharpe` 1.1644** | **1.2363** | **+6.2%** |

New Phase 2 hurdle: **etf_2017 0.928, full_2021 1.163**.

**Two consequences beyond the number:**
1. **The DeMiguel reproduction does not survive.** `max_sharpe` (1.163) now beats `equal_weight`
   (1.152) on `full_2021`. The "1/N beats the optimizers net-of-cost" claim made since Phase 2
   (§10.4, §12) was an artefact of the missing dividends. Do not repeat it without re-checking.
2. **The ML advantage is real but half the size.** +6.2%, not +14.3% (it was briefly reported as +6.5% between the dividend fix and the deep-ETF window; the window moved it again). Still inside the Phase 5
   confidence intervals, so no significance claim changes — there were none to change.

Full analysis: `docs/DIVIDEND_BIAS.md`. Related: `docs/ETF_DEEP_HISTORY_EXPERIMENT.md` documents a
second confound (the 25% cap is near-fully determining the `etf_2017` allocation, making "no ML
benefit there" unfalsifiable as stated).

### 5.2 ⚠️ CLAIM REFRAMING 2026-08-02 — how results may be WORDED, everywhere

The numbers in this file are unchanged and correct. **What may be claimed from them changed**, and
the change was applied across `README.md`, both dashboard pages, `src/api/main.py`,
`docs/CRISIS_WINDOWS_EXPERIMENT.md` and the whole report (`docs/rapport/`). Three rules now hold —
apply them to any new text you write, and fix any older text you touch:

1. **Nothing in this project is "statistically significant."** The crisis-detection sign test
   (p = 0.03125) is a *diagnostic on five retrospectively-chosen windows* with a partly
   definitional outcome — see §12I. The phrase "the only statistically significant result" is
   **retracted**; it appeared in the README, dashboard, API, report and AGENTS.md and is gone from
   all of them.
2. **Nor is anything "statistically indistinguishable."** Overlapping *marginal* CIs are not a
   test of the *difference*; failing to reject is not accepting the null. Every phase that
   concluded "indistinguishable" (§12D–§12H) over-claimed in the opposite direction. Say **"no
   paired test of the difference was run"** — and note that running one (paired block bootstrap on
   return differences) is now the highest-value outstanding evaluation task.
3. **The deliverable is a research prototype, not a production or advisory tool.** No investment
   advice, no client recommendation, no order execution. The README, both pages and the API
   description now say so explicitly. Do not reintroduce "système de qualité production",
   "outil du gestionnaire" or "notre système ML" framing.

Standing wording: *observed difference* for point estimates, *uncertainty quantification* for
intervals; never "superior" or "equivalent" without a paired test.

**Delivered and approved:** Project Alignment Note (P1–P4 traceability), C4 Level 1 context
diagram, functional swimlane diagram (French, F1–F11), and one French `.docx` supervisor
deliverable per completed phase (`docs/Livrable_Phase*_*.docx`). Supervisor: "Best team.
Congratulations."

### Phase 4 result — first real run against live Gold data, 2026-07-20

`python src/run_phase4.py` (also wired as the `phase4_compare` DVC stage) reran all 4 Phase 2
baselines alongside the 3 new Phase 4 strategies (`min_variance_ewma`, `dcc_garch`,
`regime_conditional`) — 7 strategies, one shared DSR trial pool per universe, same live snapshot
`phase2_hurdle.json` was regenerated from:

| Universe | Winner | Net Sharpe | vs. Phase 2 hurdle | Notes |
|----------|--------|-----------:|---------------------|-------|
| `etf_2017` | `max_sharpe` (Phase 2) | 0.936 | **Not beaten** (best Phase 4: `regime_conditional` 0.894) | 2,489 rows of history already gives the simple optimizer most of the value; regime/covariance ML adds turnover without a payoff here |
| `full_2021` | `regime_conditional` (Phase 4) | **1.122** | **Beaten** (hurdle 0.975) | `dcc_garch` (1.046) and `min_variance_ewma` (1.041) also both beat the hurdle on this universe — the shorter, more volatile 9-asset BVC-inclusive universe is where regime/dynamic-covariance ML earns its complexity |

A real, honest, mixed result — not spun as a universal win. `DCCGarchStrategy`'s Ledoit-Wolf
fallback fired exactly once in this run (`IAM.CS`, `full_2021`, one rebalance window) — the
non-convergence safety net working as designed, not a bug. `RegimeConditionalStrategy`'s
non-convergence fallback (neutral posterior → defensive `bear_strategy`) also fired for the first
2–3 rebalances of each universe, before `min_regime_train_days=252` clears — expected and by
design. Full run (`dvc repro`, both new stages) took **~5.5 minutes** end-to-end, well under the
~10–15 min conservatively estimated during planning.

### Phase 4 readiness — verified 2026-07-20, immediately before kickoff

This was a full audit, not an assumption. What was checked and confirmed:
- `main` in sync with `origin/main`; PRs #1–#6 all merged; PR #6 CI green.
- **126/126 tests pass**, including the 5-test no-lookahead suite and the Phase 2↔3 seam
  integration tests (`tests/test_phase3_integration.py`) — the future-corruption test specifically
  proves past rebalance weights are unaffected by corrupting future feature rows, *and* that the
  strategy genuinely consumes the features (so the guarantee isn't vacuous).
- `dagster definitions validate` → all 8 assets resolve, no cycles. *(9 as of 2026-07-27 —
  `bvc_dividends` added, §17.8.)*
- Both universes' Gold data end on the same date (`2026-07-20`); both Phase 3 feature matrices
  have **0 NaN**. *(True when checked; `etf_2017` gained 3,659 leading macro NaN on 2026-07-25
  when the deep window was adopted — see §11.)*
- `requirements.txt` already lists `hmmlearn` and `arch`.

What was found stale and fixed same-day:
- `data/gold/phase2_hurdle.json` (the net-Sharpe benchmark Phase 4 must beat) had drifted one
  universe behind after a data resync, because it is **not** a DVC-tracked pipeline output —
  see §17.1 for the ongoing duty this creates.
- The Dagster asset graph never wired the ETF-only universe or Phase 3 features into the scheduled
  job at all (fixed via PR #6, §17.7).

Known, accepted, not blocking: `EEM` is stationarity-AMBIGUOUS (§8.4) — already flagged in Phase 1
as something to revisit when Phase 4 starts; `dcc_garch.py`'s Ledoit-Wolf fallback is the concrete
answer that materialized on real data (§8.4, §12).

---

## 6. Technology Stack

```bash
pip install yfinance fredapi pandas numpy pyarrow duckdb \
            pandera statsmodels hmmlearn arch \
            matplotlib seaborn plotly mlflow dvc \
            pytest scikit-learn scipy python-dotenv \
            dagster dagster-webserver BVCscrap lxml
```

| Tool | Purpose | Notes |
|------|---------|-------|
| `yfinance` | ETF prices + MAD FX rates | `auto_adjust=True` always (§15.3) |
| `BVCscrap` | BVC prices via medias24 | free tier starts ~2021-06/07, rolling window (§8.2) |
| `fredapi` | FRED macro series | ICE BofA series capped to rolling window; replaced by BAA10Y (§8.3) |
| Parquet / `pyarrow` | Columnar storage, all layers | |
| DuckDB | Analytical SQL on Gold Parquet | use instead of pandas groupby (§15.7) |
| Pandera | Data contracts, Silver + Gold writes | |
| DVC | Data versioning | `dvc.yaml` covers ingest→clean→features→ml_features; `phase2_hurdle.json` is deliberately outside it (§17.1) |
| MLflow | Experiment tracking (results) | Dagster tracks *execution*; MLflow tracks *results* — not redundant |
| Dagster | Scheduling + lineage | local only; launchd setup in `scripts/setup_launchd.sh`; 8 assets (§9, §11) |
| `statsmodels` | ADF/KPSS | |
| `hmmlearn` | HMM regimes | **Phase 4 — active now** |
| `arch` | Univariate GARCH | **Phase 4 — active now.** ⚠️ no built-in multivariate DCC (§3.2) |
| `scikit-learn` | Ledoit-Wolf shrinkage (used, Phase 2), CV base classes (Phase 5) | |
| `scipy` | Constrained portfolio optimization (SLSQP, used in `src/strategies.py`) | |
| FastAPI + Streamlit | Phase 6 API + dashboard | proposed; Streamlit over hand-built Plotly for speed |

**Not used, deliberately:** `mlfinlab` (license/dependency risk — custom purged CV instead);
Airflow/Prefect (heavier than needed); `pct_change()` for returns (see §15.1).

---

## 7. Data Architecture

Medallion pattern; raw data never overwritten; each layer a deterministic transformation of the
previous one.

```
Sources → [Bronze data/bronze/  immutable, as-received Parquet]
        → [Silver data/silver/  aligned, validated log-returns × 2 universes + validation reports]
        → [Gold   data/gold/    log_returns × 2, macro_features, stationarity_report,
                                 ml_features × 2 + manifest (Phase 3), phase2_hurdle.json (Phase 2)]
        → Phase 2+ modeling (reads data/gold/ ONLY)
```

- **Bronze:** zero transformation; files immutable; source changes → new versioned file
  (`bvc_prices.parquet` is the one exception — see §8.2, it's an intentionally-merged rolling file).
- **Silver:** calendar alignment (BVC ≠ NYSE holidays), forward-fill (capped, `ffill_limit=5`),
  initial-NaN window dropped, Pandera validation on every write, illiquidity flagging. Produced for
  **both** universes (`log_returns.parquet` 9-asset, `log_returns_etf.parquet` 5-ETF).
- **Gold:** log-returns matrices (Phase 2 input), lagged/standardized macro features (Phase 1),
  ADF+KPSS stationarity report, causal ML features + manifest (Phase 3, `src/ml_features.py`), and
  the Phase 2 net-of-cost hurdle (`phase2_hurdle.json`, not pipeline-tracked — §17.1).

---

## 8. Asset Universe and Known Data Gaps

### 8.1 Assets and macro series

**BVC equities:** `IAM.CS` (Maroc Telecom), `ATW.CS` (Attijariwafa Bank), `CIH.CS` (CIH Bank),
`BCP.CS` (Banque Centrale Populaire).
**International ETFs:** `SPY`, `QQQ`, `EEM`, `GLD`, `TLT`.
**FRED macro:** `VIX` (VIXCLS), `US10Y` (DGS10), `DXY` (DTWEXBGS), `CREDIT_SPREAD` (BAA10Y —
replaced BAMLH0A0HYM2, see §8.3).
**BAM macro:** `USDMAD`, `EURMAD` (Yahoo FX), `TAUX_DIR` (hand-maintained decision list — §17.1).

```python
START_DATE = "2017-01-01"   # requested; see §8.2 for what's actually available per universe
END_DATE   = None            # always download up to today
```

### 8.2 Gap: BVC history starts ~mid-2021, not 2017

medias24/BVCscrap's free tier has nothing before mid-2021; pre-2021 BVC requires a paid vendor.
The 9-asset universe (`log_returns.parquet`) therefore runs **2021-07 → today (~1,300 rows)**,
dropping the 2020 COVID crash. `align_calendars` warns about the truncation and
`validation_report.json` records it explicitly — this cannot happen silently.

**Mitigation: dual-universe backtesting (§10.2)** recovers COVID-crash evidence via the ETF-only
universe (`log_returns_etf.parquet`, 2017 → today, 5 assets) at zero data cost. Every Phase 2+
strategy runs on **both** universes. State the gap explicitly in anything shown to Abdelmouttalib;
never imply full 2017–today coverage for the 9-asset universe.

**Operational note:** the two universes are refreshed by different Dagster assets
(`log_returns` vs. `log_returns_etf`) that share Bronze inputs but run independently. They must be
regenerated together after any Bronze refresh or they drift apart — this happened twice (once
before Dagster wiring existed, once because the wiring itself was incomplete) and is now fixed at
the graph level (§17.7). `dvc status` after any manual pipeline run is the way to catch a repeat.

### 8.3 Resolved: HY_SPREAD replaced by CREDIT_SPREAD (BAA10Y) — 2026-07-05

FRED restricts licensed ICE BofA series (incl. `BAMLH0A0HYM2`) to roughly the last 3 years, which
had silently truncated HY_SPREAD to 2023-07+. Replaced with `CREDIT_SPREAD` = FRED `BAA10Y`
(Moody's Baa corporate yield minus 10Y Treasury) — unrestricted, ~2,478 observations from
2017-01-02, same `fredapi` path. Tradeoff: investment-grade Baa moves less dramatically than
high-yield OAS, but widens in the same credit-stress episodes.

### 8.4 Caveats on the data itself

- **EEM is stationarity-AMBIGUOUS** (ADF rejects unit root, KPSS also rejects stationarity at
  p≈0.033) — plausibly a 2022 EM-sell-off structural break. Treated as stationary through Phase 1–3;
  **explicitly flagged to revisit now that Phase 4 chooses regime/covariance model inputs** — an
  HMM or GARCH model fit on a series with a live structural-break question deserves a second look
  before being trusted.
- **CIH.CS / BCP.CS** show multi-day zero-return runs (illiquidity) — a fact of the BVC, flagged by
  the pipeline, handled via transaction-cost weighting in Phase 2 (30bps BVC vs. 10bps ETF).
- **Currency:** BVC returns are MAD-denominated, ETFs USD. Returns are unitless so portfolio
  arithmetic is valid, but every backtest result embeds an **unhedged USD/MAD exposure** —
  documented in `src/backtest.py`'s module docstring and the Phase 2 notebook §1; never "fixed" by
  converting returns (FX-hedging is out of scope).

---

## 9. Phase 1 — Data Infrastructure (✅ Complete, 2026-07-02)

**Deliverable achieved:** validated Gold layer (log-returns Parquet, dates × 9 assets, zero NaN,
Pandera-clean, 8/9 assets STATIONARY, EEM ambiguous and flagged) + evidentiary EDA notebook.

| Component | File | Role |
|-----------|------|------|
| Bronze ingestion | `src/ingest.py` | yfinance ETFs, FRED (partial-failure tolerant), BVCscrap, BAM (FX + hand-maintained TAUX_DIR) |
| Silver cleaning | `src/clean.py` | BVC merge, calendar alignment, log-returns, illiquidity flag, dual-universe output (§8.2) |
| Gold features | `src/features.py` | ADF+KPSS per asset, macro features (ffill-align → diff → z-score → lag ≥ 1 enforced) |
| Data contracts | `src/schemas.py` | log-returns schema (±50% bound, ≥500 rows, sorted index, no NaN) |
| Entry point | `src/pipeline.py` | Bronze→Silver→Gold under one MLflow run |
| Scheduling | `src/orchestration/` + `workspace.yaml` | Dagster assets wrapping the same functions; daily schedule; launchd for unattended runs |
| Analytics helper | `src/utils.py` | `query_gold()` — DuckDB over Gold Parquet; `load_params()` |

**EDA notebook** (`notebooks/phase1_eda.ipynb`) is *evidentiary*: fat tails + QQ, rolling vol, ACF
of squared returns, rolling correlations, 2022-shock diversification breakdown, stationarity table.

---

## 10. Phase 2 — Markowitz Baseline + Backtesting (✅ Complete, merged PR #3, 2026-07-10)

### 10.1 Realistic management constraints (from the EURAFRIC brief — a requirement, not a metric)

- **Long-only**, **max weight per asset** (25% cap, `params.yaml: backtest.max_weight`).
- **Transaction costs inside the backtest:** 10 bps ETFs, 30 bps BVC, deducted from net returns at
  every rebalance based on turnover. Both gross and net are always reported — a strategy that wins
  gross and loses net is a finding, not noise.
- **Illiquidity-aware:** the higher BVC cost bps is the mechanism; CIH.CS/BCP.CS are penalized
  through cost, not excluded.

#### ⭐ The 25% cap is not neutral scaffolding — on `etf_2017` it is the single strongest driver of performance

Promoted here from a research note on 2026-07-27, because it belongs in the constraint discussion
rather than a footnote. The cap was chosen in Phase 2 as a *realistic management constraint*, i.e.
a business rule, not a modelling device. Sweeping it on the deep `etf_2017` window (248 rebalances,
20.7 years, everything else fixed — `experiments/etf_cap_verdict.py`) shows it behaves like the
most powerful regularizer in the project:

| `max_weight` | Best classical net Sharpe | `min_variance_lw` distinct allocations / 248 |
|---|---:|---:|
| **0.25 (ours)** | **0.9525** | **1** |
| 0.30 | 0.9394 | 171 |
| 0.35 | 0.9325 | 248 |
| 0.40 | 0.9122 | 248 |
| 1.00 (uncapped) | 0.8650 | 248 |

Two things follow, and both are results in their own right:

1. **A 10.1% Sharpe swing driven purely by the constraint** (0.9525 → 0.8650), *monotonically
   decreasing as the cap loosens*. That is larger than the gap between ANY two models measured on
   this universe — Ledoit-Wolf, EWMA, DCC-GARCH and the HMM regime switch combined move it less.
   **This is Jagannathan & Ma (2003) reproduced on our own data:** a binding long-only weight
   constraint is mathematically equivalent to shrinking the covariance matrix, so the cap is doing
   the estimation-error control (P1) that the ML was introduced to do. The constraint is not a
   handicap we worked around; it is the best-performing risk control we have.
2. **At 0.25 the cap is very nearly the whole allocation decision.** With 5 assets,
   `5 × 0.25 = 1.25`, so every feasible long-only portfolio must hold ≥4 assets *at the cap*.
   `min_variance_lw` therefore emits **one** allocation across all 248 rebalances; at 0.30 it emits
   171. `min_variance`, `min_variance_lw` and `regime_conditional` returned byte-identical weights
   post-2018 (`max|diff| 4.4e-16`). This is why `phase2_hurdle.json` and `dashboard_showcase.json`
   can name *different* winners on `etf_2017` at the same Sharpe (0.9525) — the tie is real, not a
   bug.

**Scope of the claim.** `full_2021` is not affected: with 9 assets `9 × 0.25 = 2.25`, so only 4 of
9 need be at the cap and the optimizer keeps real freedom. And the cap is NOT why Phase 4's
`etf_2017` conclusion held — that was re-tested at every non-binding cap and regime ML lost at all
of them (§12, "SETTLED"). The cap made the conclusion *unfalsifiable as originally stated*; it did
not make it wrong.

**How to present it:** as a finding, not an apology. "Our most effective risk control on the ETF
universe turned out to be the position limit a real mandate would impose anyway" is a strong,
defensible, supervisor-facing statement — and it is exactly the kind of result the brief's
*contraintes réalistes de gestion* framing invites. Full analysis:
`docs/ETF_DEEP_HISTORY_EXPERIMENT.md`.

### 10.2 Dual-universe backtesting

Every strategy runs on both universes, reported side by side:
- **etf_2017** — 5 ETFs, 2017 → today: includes COVID crash + 2022 rate shock.
- **full_2021** — 9 assets, 2021 → today: the full EURAFRIC-relevant portfolio.

### 10.3 The strategy interface (the seam Phase 4 plugs into)

`src/strategies.py` defines the `Strategy` ABC: `fit(train_returns, extras=None) -> pd.Series`
(weights, long-only, sum to 1). `src/backtest.py`'s `run_backtest()` engine slices `returns` **and
every `extras` frame** to `:τ` before each fit — a strategy physically cannot see the future. This
is the exact mechanism Phase 3's feature frame plugs into (`extras={"features": ...}`), and it is
proven leak-free end-to-end, not just asserted (§11, `tests/test_phase3_integration.py`).

Implemented baselines (`src/strategies.py`): `EqualWeight`, `MinVariance` (sample covariance),
`MinVarianceLW` (Ledoit-Wolf shrinkage), `MaxSharpe`. These are rungs 1–2 of the Phase 4 covariance
ablation ladder (§3.2) — already built, already benchmarked.

### 10.4 Metrics and the Phase 4 hurdle

`src/metrics.py` implements annualized Sharpe (OOS only), max drawdown, Calmar, turnover,
information ratio vs. equal-weight benchmark, and **Deflated Sharpe Ratio** (López de Prado),
logged per-run to MLflow with `n_trials` for auditability.

`python src/run_backtest.py` runs all 4 baselines × 2 universes, logs everything to MLflow, and
writes `data/gold/phase2_hurdle.json` — the net-of-cost Sharpe **Phase 4 models must beat to
justify their added complexity**. Current hurdle (regenerated 2026-07-20 against fresh data):

| Universe | Best baseline | Net Sharpe | OOS window |
|----------|---------------|-----------:|------------|
| etf_2017 | max_sharpe | 0.936 | 2018-01-01 → 2026-07-20 |
| full_2021 | equal_weight | 0.975 | 2022-07-01 → 2026-07-20 |

Equal-weight beating the optimizers net-of-cost on `full_2021` reproduces the DeMiguel et al.
(2009) result on our own data — 1/N is the honest baseline it's always claimed to be, once costs
are counted. **This file is not DVC-tracked (§17.1) — rerun `run_backtest.py` after any data
refresh before quoting these numbers.**

**No-lookahead property is itself tested**, not just assumed: `tests/test_backtest.py::TestNoLookahead`
(5 tests — includes a "perfect foresight" strategy fed the full dataset by hand, which must collapse
to ≈0 Sharpe when routed through the engine).

**Notebook:** `notebooks/phase2_backtest.ipynb` — engine explanation, equity curves gross/net with
COVID-2020 + 2022 zoom panels, capped-vs-uncapped concentration (P1 evidence), turnover/cost
analysis, full metrics table with DSR.

---

## 11. Phase 3 — ML Feature Engineering (✅ Complete, merged PRs #4/#5/#6, 2026-07-17 → 2026-07-20)

**Deliverable:** `src/ml_features.py` — causal features for both universes, feeding Phase 4 through
the exact same `extras` seam Phase 2 built (§10.3).

**6 core return features** (not lagged — causal because the engine fits at τ and earns from τ+1):
`MARKET_RETURN`, `MARKET_VOL_SHORT` (21d), `MARKET_VOL_LONG` (63d), `AVG_PAIRWISE_CORR` (63d,
min 42 periods — this is the direct P3 signal), `CROSS_SECTIONAL_DISPERSION`, `MARKET_DRAWDOWN`.

**7 lagged macro signals** (differenced, standardized, lagged ≥1 day — publication-timing risk
makes same-day macro data lookahead): `VIX_DIFF_L1`, `US10Y_DIFF_L1`, `DXY_DIFF_L1`,
`CREDIT_SPREAD_DIFF_L1`, `EURMAD_DIFF_L1`, `USDMAD_DIFF_L1`, `TAUX_DIR_DIFF_L1`.

**Key design decisions:**
- **No global standardization.** Each Phase 4 model must fit its own scaler inside its own
  walk-forward training window — recorded in the manifest as `global_standardization: false`.
  Standardizing once over the full history before backtesting would itself be a lookahead leak.
- **Causal asymmetry is intentional**, not an oversight: return features derive from the same-day
  close the engine already fits on; macro features carry real publication delay and are lagged.
- **Multi-source macro alignment fix (2026-07-20):** FRED and BAM publish on different calendars;
  concatenating them left interior NaN gaps on dates only one source published. Fixed by
  forward-filling *all* gaps post-concat (`build_lagged_macro_signals`), not just the leading
  window — no INTERIOR gaps remain in either matrix. Locked in by
  `test_interior_nan_from_multi_source_calendars_is_forward_filled`.
- ⚠️ **`ml_features_etf.parquet` is NO LONGER fully dense (corrected 2026-07-27).** This section
  claimed "0 NaN for both universes" through 2026-07-25; adopting the deep ETF window (2004-11)
  pushed the matrix ~12 years behind the Moroccan macro series, so `etf_2017` now carries **3,659
  LEADING NaN**: `TAUX_DIR_DIFF_L1` 3,101, `EURMAD_DIFF_L1` 327, `DXY_DIFF_L1` 231. Its first
  fully-dense row is **2017-01-04** (2,493 of 5,594 rows). `ml_features_full.parquet` is still 0
  NaN.
  - **Not currently a live bug, and not silent:** the manifest records it (`max_leading_nan`,
    `fully_complete_rows`) exactly as §15.13 requires, and nothing consumes those columns —
    `regime.REGIME_FEATURES` is the three dense market features, and F7 reads only
    `REGIME_BULL_PROB`. It is a **trap for the next model** that reaches for a macro column on
    `etf_2017`, not a defect in today's results.
  - **The warm-up check below was triggered and not performed.** Its own instruction says to
    re-check when the window changes; the window changed on 2026-07-25 and this was caught two
    days later during review. Any strategy consuming a macro feature on `etf_2017` must either
    start after 2017-01-04 or handle the gap explicitly.
- **Warm-up is manifest-reported, not silently absorbed:** `leading_nan_by_column` and
  `max_leading_nan` per universe let Phase 4 verify `min_train_days` clears every feature's
  warm-up before the first fit. At `min_train_days=252` this holds for `full_2021` and for
  `etf_2017`'s six return features, but **NOT** for `etf_2017`'s macro signals (see above) —
  re-check whenever either universe's start date or `min_train_days` changes.

**The seam is proven, not just unit-tested per side:** `tests/test_phase3_integration.py` feeds a
real Phase 3 feature matrix through the real Phase 2 engine and asserts (a) a feature-consuming
strategy never sees a feature row dated after its decision date, and (b) corrupting the *future* of
the feature frame cannot change *any past* rebalance weight — while also proving the strategy
genuinely used the features (so the guarantee isn't vacuous). This is the exact mechanism Phase 4's
HMM/covariance strategies will depend on.

**Deliverables:** `docs/Livrable_Phase3_Feature_Engineering_ML.docx` (French), refreshed
`notebooks/phase3_features.ipynb` (explains the Phase 4 destination and the causal asymmetry, not
just the feature math).

---

## 12. Phase 4 — ML Portfolio Models (✅ Core complete, 2026-07-20 — hardening pass pending)

**Delivered:** HMM regime detection (§3.1) conditioning portfolio weights, plus the final two rungs
of the covariance ablation ladder (§3.2: EWMA, DCC-GARCH) — all three plugged into the unmodified
Phase 2 engine via the `Strategy` ABC, fed by Phase 3's `extras={"features": ...}` seam. Benchmarked
against the Phase 2 hurdle on live data — result and full context in §5's "Phase 4 result" section.

**Decisions locked before implementation (2026-07-20, with the user):**
1. **HMM: 2 states (bull/bear), not 3.** `full_2021` has only ~1,255 rows; a 3rd "crisis" state
   risked thin, seed-sensitive samples on that universe. `src/regime.py` trains on exactly
   `MARKET_RETURN`, `MARKET_VOL_SHORT`, `AVG_PAIRWISE_CORR` — the direct P2/P3 candidates — with its
   own `StandardScaler` fit fresh per window (never global, per §15.14) and 5 deterministic-seed EM
   restarts, keeping the highest-log-likelihood converged fit. `label_regimes()` maps hmmlearn's
   unordered state indices to "bull"/"bear" by ranking fitted `MARKET_RETURN` mean — recomputed
   every refit, since hmmlearn does not guarantee stable state ordering across fits.
2. **Composition: hard discrete regime dispatch**, not a continuous posterior blend.
   `RegimeConditionalStrategy` fits the HMM, labels the latest row's regime, and hands the *entire*
   decision to an already-tested Phase 2 baseline: `bull_strategy` (default `MaxSharpe`) or
   `bear_strategy` (default `MinVarianceLW`) — configurable via `params.yaml: regime.bull_strategy` /
   `bear_strategy`. On a non-converging/thin window (below `min_regime_train_days=252`), the neutral
   50/50 posterior resolves to the **defensive** `bear_strategy`, not an arbitrary tie-break — when
   uncertain, don't guess bullish. Known accepted trade-off: hard switching can spike turnover right
   at a regime boundary (observed in the real run — `regime_conditional` has the highest turnover of
   any Phase 4 strategy on both universes); a continuous-blend fast-follow remains a documented
   option if this proves costly, not built in this pass.
3. **DCC-GARCH was built in this pass** (not deferred as originally scoped in planning) —
   `src/dcc_garch.py` implements the full Engle (2002) two-stage estimator: per-asset univariate
   `GARCH(1,1)` via `arch` (which ships no multivariate DCC — the DCC recursion and its (a, b)
   QMLE fit are hand-written, not from any dependency), then `Q_t/R_t` correlation dynamics with
   variance-targeted `Q̄`. Falls back to Ledoit-Wolf shrinkage, with a logged `WARNING`, on any
   asset's GARCH non-convergence — fired once for real on `IAM.CS` in the first live run (§5),
   confirming the safety net works, not just in tests.
4. **A DVC stage was added for both `phase2_hurdle.json` and `phase4_results.json`**
   (`dvc.yaml`: `phase2_hurdle`, `phase4_compare` stages) — closes the exact staleness gap that hit
   `phase2_hurdle.json` once already (§17.1/§17.7). `dvc repro` now regenerates both automatically
   whenever their upstream Gold data or source code changes.

**New modules:** `src/regime.py` (HMM fit/label/predict, ~180 lines), `src/dcc_garch.py`
(two-stage DCC-GARCH, ~200 lines), `src/run_phase4.py` (7-strategy × 2-universe MLflow runner,
mirrors `run_backtest.py`). New `Strategy` subclasses in `src/strategies.py`: `MinVarianceEWMA`,
`DCCGarchStrategy`, `RegimeConditionalStrategy`. New tests: `tests/test_regime.py`,
`tests/test_dcc_garch.py`, `tests/test_phase4_integration.py` (the real no-lookahead gate, using
the actual `RegimeConditionalStrategy` — not a toy probe, mirrors `test_phase3_integration.py`'s
future-corruption pattern), `tests/test_run_phase4.py`. Full suite: 159 tests, offline/synthetic,
~20s (`python -m pytest tests/ -q`).

**Hardening pass shipped (PR #8, 2026-07-20):** `notebooks/phase4_regime_covariance.ipynb`
(executed live, 8 sections, zero errors), `docs/Livrable_Phase4_Regime_Covariance.docx` (French),
README updates. The notebook's dedicated investigation into why `etf_2017` didn't respond to
regime/covariance ML found: `etf_2017` has ~2× `full_2021`'s history and an already-strong Phase 2
hurdle (`max_sharpe` 0.936) — less mispricing left to correct; `full_2021`'s hurdle
(`equal_weight` 0.975, the DeMiguel result) is a weaker bar precisely because naive diversification
is doing the work there — exactly where regime/covariance ML has room to add value. Not a
hyperparameter artifact — a real, defensible negative result.

> ⚠️ **SUPERSEDED IN PART, 2026-07-25.** That explanation is incomplete, and the DeMiguel premise
> inside it is wrong. Two corrections:
> 1. **`full_2021`'s "weak bar" was a dividend artefact.** Once BVC total returns are used,
>    `max_sharpe` (1.163) beats `equal_weight` (1.152) — the DeMiguel result does not reproduce
>    (§5.1). The reasoning above rests on a premise that no longer holds.
> 2. **The `etf_2017` non-response is largely a CONSTRAINT artefact, not a mispricing story.** With
>    5 assets and a 25% cap, `5 × 0.25 = 1.25`, so any feasible long-only portfolio must hold ≥4
>    assets *at the cap*: the constraint, not the covariance model, picks the portfolio. Measured
>    directly — at cap 0.25 `min_variance_lw` produces **one** allocation across 248 rebalances; at
>    0.30 it produces 169; and `min_variance_lw`/`max_sharpe`/`regime_conditional` returned
>    byte-identical weights (`max|diff| 4.4e-16`) post-2018. "Regime/covariance ML adds no value on
>    `etf_2017`" is therefore **not falsifiable as stated** — the optimizer had nowhere to express a
>    view. `full_2021` (9 assets) is unaffected.
>
> Incidental but citable: the capped portfolio has the *highest* Sharpe in the cap sweep (0.953 vs
> 0.846 uncapped) — Jagannathan & Ma (2003) reproduced on our data; the constraint is doing the
> estimation-error control the ML was introduced for.
>
> **SETTLED same day** (`experiments/etf_cap_verdict.py`, outcomes pre-registered before the run).
> Sweeping the cap 0.25 → 1.00 on the deep 20.7-year window: **regime ML loses at every cap where
> the optimizer is free** (−0.9% at 0.30, −3.8% at 0.35, −3.2% at 0.40, −3.8% uncapped). So the
> Phase 4 `etf_2017` conclusion was CORRECT — it is now **defensible rather than merely
> unfalsifiable**, which is a better position than before even though the number didn't move. The
> lift is *least* negative at the binding 0.25, i.e. the cap was partly masking the gap.
>
> **The bigger finding is the cap itself.** Best classical Sharpe runs 0.953 at cap 0.25
> monotonically down to 0.865 unconstrained — a **10% swing driven purely by the constraint**,
> larger than any modelling difference measured on this universe. On `etf_2017` the 25% cap, chosen
> in Phase 2 as a *realistic management constraint* rather than a modelling device, is doing more
> estimation-error control than Ledoit-Wolf, EWMA, DCC-GARCH or the HMM regime switch combined.
> That belongs in the deliverable, not a footnote.
>
> Full analysis: `docs/ETF_DEEP_HISTORY_EXPERIMENT.md`.

---

## 12B. Phase 4B — Adaptive ML Signal Models (F7) (✅ Core complete, 2026-07-21)

**What F7 actually is, and why it's distinct from Phase 4:** an EURAFRIC functional diagram
(F1–F11) named F7 — *"Adaptive ML Models (RF/XGBoost/LSTM) — HMM regime-conditional return
prediction, signals for portfolio optimization"* — a genuinely different function from Phase 4's
F4 (regime *detection* used to switch between two existing Markowitz optimizers). Phase 4 never
predicts a return; F7 does. Confirmed by direct exploration before building anything: no per-asset
feature matrix existed anywhere in this codebase (`ml_features.py`'s features are all market-level
cross-sectional aggregates), and none of `xgboost`/`torch`/`tensorflow`/`keras` were installed.

**Decisions locked with the user (2026-07-20), each with reasoning:**
1. **RandomForest + XGBoost**, both pooled cross-sectional panel models (asset identity as a
   feature, not one model per asset) — multiplies effective training rows by the asset count
   (5–9×), the practical answer to `full_2021`'s ~1,255-row history.
2. **Regime posterior as one input feature** to a single model, not two dispatched sub-models
   (mirroring `RegimeConditionalStrategy`'s hard-switch design) — data-efficient; splitting an
   already-thin dataset further for a harder learning problem (regression vs. HMM's classification)
   was judged not worth the fragmentation.
3. **Predicted `mu` replaces the naive sample-mean** already fed into the existing Sharpe objective,
   via new thin `Strategy` subclasses (`RandomForestSignalStrategy`, `XGBoostSignalStrategy`) —
   zero new optimizer code, covariance held fixed at Ledoit-Wolf; the same "swap one moment into
   the unmodified engine" pattern every prior Phase 4 addition used.
4. **LSTM was originally in scope too** (the user's explicit initial choice — the full diagram, not
   a reduced MVP) and was fully built: `src/lstm_signal.py` (per-asset rolling sequences, a small
   1-layer `ReturnLSTM`, `torch.manual_seed`-deterministic training), a new `LSTMSignalStrategy`,
   and 10 passing standalone tests (`tests/test_lstm_signal.py`, ~6s, tiny synthetic fixtures).
   **Dropped from this pass** after wiring it into the shared strategy-invariant test suite (which
   exercises `RandomForestSignalStrategy`/`XGBoostSignalStrategy`/`LSTMSignalStrategy` back-to-back
   in one process on the full 599-row/9-asset fixture) caused a **segfault** — `torch` and
   `xgboost` both load native/OpenMP-linked libraries, and having both loaded in the same process
   crashed on this development machine. Not a defect in either library's own code; a real,
   reproducible environment conflict. All LSTM code, the `torch` dependency, and the `lstm:`
   params.yaml block were removed from this branch rather than shipped unstable — deferred to a
   run on more capable/isolated hardware, not abandoned.

**New modules:** `src/ml_signals.py` (~330 lines) — `build_asset_features` (per-asset trailing-
return/volatility/price-relative-to-moving-average panel, deliberately small, causal by the same
convention `ml_features.py` uses), `melt_to_panel` (wide → pooled `(Date, ASSET)` panel),
`attach_regime_feature` (reuses `regime.fit_hmm`/a new `regime.predict_regime_posterior_series`
factored out for this — broadcasts one bull-probability per date across every asset that date),
`build_supervised_dataset` (the highest-risk correctness point: the current rebalance date is
**structurally** excluded from training, by date, not by a NaN filter that could be bypassed —
covers even a hypothetical caller error where `log_returns` extends past the panel's last date),
`fit_predict_expected_returns` (orchestrates the above; RF/XGBoost lazy-imported; deterministic
`random_state=0` default, since both estimators are otherwise stochastic and would have broken the
"extras accepted and ignored" strategy invariant every prior addition satisfies), and
`run_ml_signal_features` (Gold-persistence for auditability only — the strategies recompute
features fresh from `train_returns` at every rebalance; they do not read this file). New
`_neg_sharpe` module-level helper in `src/strategies.py`, extracted from `MaxSharpe`'s inline
closure so every mu-swapping strategy shares one objective instead of duplicating it.

**Result — first real run against live Gold data, 2026-07-21** (`python src/run_phase4b.py`, also
the `phase4b_compare` DVC stage; 9 strategies — the existing 7 plus `rf_signal`/`xgb_signal` — one
shared DSR trial pool per universe, same live snapshot):

| Universe | Winner (unchanged from Phase 4) | Net Sharpe | `rf_signal` | `xgb_signal` |
|----------|----------------------------------|-----------:|-------------:|-------------:|
| `etf_2017` | `max_sharpe` | 0.936 | 0.727 | 0.747 (turnover 0.356) |
| `full_2021` | `regime_conditional` | 1.122 | 1.062 (close, doesn't clear it) | 0.743 (turnover 1.076) |

**An honest negative result — neither F7 strategy beats its universe's hurdle.** Not spun as a win.
`xgb_signal`'s very high turnover on `full_2021` (>1.0, meaning close to a full portfolio turnover
every rebalance) is the standout finding — a pooled tree ensemble retrained monthly on a noisy
return-prediction target appears to overfit month-to-month noise enough to trade aggressively
against it, net-of-cost. `rf_signal` came closest to the `full_2021` hurdle (1.062 vs. 1.122),
suggesting the signal isn't worthless, just not (yet) worth its transaction cost at this
configuration. Worth investigating in the hardening pass: feature importances (does the model
actually use the regime feature, or ignore it?), and whether a stronger `min_train_rows` floor or
shrunk hyperparameters reduce the turnover/overfitting problem.

**Tests:** `tests/test_ml_signals.py` (30), `tests/test_phase4b_integration.py` (8 — parametrized
over both strategies; mirrors `test_phase4_integration.py`'s future-corruption pattern but proves
**label construction**, not just feature slicing, doesn't leak — F7's labels are derived from the
same `returns` matrix its features come from, exactly where a subtle leak would hide),
`tests/test_run_phase4b.py` (2). Full suite: 213 tests, offline/synthetic, ~55s
(`python -m pytest tests/ -q`).

**Hardening pass shipped** (`feature/phase4b-hardening`, merged to `main` via PR #10→#12):
`notebooks/phase4b_adaptive_ml_signals.ipynb` (executed live), `docs/Livrable_Phase4B_Adaptive_ML_Signals.docx`
(French), README updates.

---

## 12C. Phase 4C — Cost-Aware Optimization + μ Regularization (✅ Complete, 2026-07-21)

**Why this phase exists.** Phase 4B's negative result was *diagnosable*, not mysterious: on
`full_2021`, `rf_signal` had the **best gross Sharpe of the whole comparison** (1.240, above the
Phase 4 winner's 1.204) and lost 0.178 of it to a 0.885 turnover. The signal was informative;
acting on every revision was not affordable — a portfolio-*construction* failure, not a
*prediction* one. This fits Chopra & Ziemba (1993): estimation error in expected returns hurts a
mean-variance optimizer ~10× more than equivalent covariance error, which is exactly why Phase 4
(better cov) beat its hurdle and Phase 4B (better μ) did not.

**Four levers added to the F7 strategies, all defaulting OFF** (so Phase 4B stays bit-reproducible
as the honest floor):
1. **Turnover penalty** — `_optimize_weights` gains `w_prev`/`turnover_penalty`, using a SMOOTH L1
   surrogate `√((w−w_prev)²+ε)` because SLSQP is gradient-based and raw `|·|` is non-differentiable
   exactly where a penalized optimum sits (trade nothing).
2. **μ shrinkage** toward the naive sample mean, and **3. rank tilt** (keep only the ordering) —
   `ml_signals.apply_mu_transform`. `shrinkage_weight=0` reproduces `MaxSharpe`'s naive μ exactly.
4. **`cov_estimator` selection** on the F7 strategies (`strategies.estimate_covariance`) — pairs a
   predicted μ with any covariance rung, so "best μ + best cov" is config, not a new class.

**New engine seam (opt-in, leak-free):** `backtest.CURRENT_WEIGHTS_KEY` injects the portfolio's
drifted weights at τ into `extras` — ONLY for strategies setting `wants_current_weights=True`, as a
one-row frame indexed by τ so it satisfies the identical `index.max() ≤ τ` invariant the
no-lookahead suite enforces on every other extras frame. Portfolio state is a function of past
`fit()` outputs and returns strictly before τ; the future-corruption gate is re-run for the new
channel (`test_future_corruption_cannot_change_past_weights_with_portfolio_state`).

**Real result (live Gold, `python src/run_phase4c.py` / `phase4c_compare` DVC stage, 14 strategies,
shared DSR pool):** an honest **near-miss, not a win**.

| Universe | Phase 4 hurdle | Best F7 4C variant | Beats? |
|----------|----------------|--------------------|--------|
| `full_2021` | `regime_conditional` 1.1215 | `rf_signal_shrunk` **1.117** (turnover 0.885→0.488) | **No — within 0.4%, untuned** |
| `etf_2017` | `max_sharpe` 0.936 | `rf_signal_shrunk` 0.855 | No |

Three findings: (1) **shrinkage was the right lever** — kept the gross signal (1.240→1.223) while
halving turnover, near-tying the hurdle untuned (Chopra-Ziemba confirmed); (2) **the turnover
penalty backfired on the good model** — `rf_signal_cost` crushed turnover to 0.175 but collapsed to
net 0.704 (λ=1.0 refused to trade a good signal); (3) **yet the same penalty *helped* the
over-trader** — `xgb_signal` 0.743→0.863 (turnover 1.076→0.606). A single global λ is too blunt for
the well-behaved model and about right for the pathological one → **the right λ is model-specific, a
Phase 5 tuning task.** The hyperparameters (`turnover_penalty=1.0`, `shrinkage_weight=0.5`) are
CHOSEN, not tuned — tuning them by looking at this run would be the P4 overfitting the project
forbids.

> ⚠️ **REFRESHED 2026-07-30 on dividend-corrected data — the headline finding of this phase is
> RETRACTED.** Every number above predates the correction. Re-running both stages
> (`phase4b_compare`, `phase4c_compare`) against the committed 07-24 Gold snapshot gives:
>
> | `full_2021` | gross | net | turnover | was (net) |
> |---|---:|---:|---:|---:|
> | `regime_conditional` (hurdle) | 1.313 | **1.2363** | 0.293 | 1.1215 |
> | `rf_signal` | 1.274 | 1.123 | 0.779 | 1.062 |
> | `rf_signal_shrunk` | 1.118 | **1.006** | 0.516 | **1.117** |
> | `rf_signal_cost` | **0.857** | 0.822 | 0.171 | 0.704 |
>
> 1. **The near-miss is gone.** `rf_signal_shrunk` was the phase's headline at 1.117 against a
>    1.1215 hurdle — within 0.4%, and the basis for "shrinkage works". Corrected, it is **1.006 vs
>    1.2363, a 0.230 gap**. The correction lifted the regime baseline (+0.115) while *lowering*
>    the shrunk variant (−0.111), so the two moved apart in both directions at once. Do not quote
>    the 0.4% figure again.
> 2. **The turnover penalty destroys the SIGNAL, not just the trading — now visible in gross.**
>    `rf_signal_cost`'s gross Sharpe is 0.857 against plain `rf_signal`'s 1.274. It is not buying
>    the same alpha more cheaply; refusing to trade means it cannot express the view at all.
>    Pre-correction this was only inferable from the net number (0.704).
> 3. **What DOES survive:** the diagnosis that founded the phase. `rf_signal` still has a strong
>    gross Sharpe (1.274, 2nd of 14) and still loses 0.151 of it to 0.779 turnover; `xgb_signal`
>    still turns over >1.0 (1.061) and still loses 0.199. The per-model asymmetry also survives —
>    the penalty helps the over-trader (`xgb_signal` 0.874→`xgb_signal_cost` 0.877, and on
>    `etf_2017` it *hurts* `rf_signal` 0.904→0.879) which is still evidence a single global λ is
>    wrong.
> 4. **Provenance note.** An earlier attempt produced a TORN artifact: the Dagster nightly job
>    succeeded for the first time in days (the module-level import fix in §17.9 repaired it) and
>    rebuilt Gold through 07-29 *mid-run*, so `etf_2017` read the 07-24 snapshot and `full_2021`
>    read a newer one — every `full_2021` baseline mismatched the dashboard by exactly the amount
>    5 extra days explains. Gold was restored from the DVC cache, the schedule STOPPED, and the run
>    redone end-to-end with a pre-launch assertion on the snapshot date. Lesson: **a long
>    modelling run and a live scheduler must not share a data directory.**

**New/changed:** `src/run_phase4c.py` (14-strategy runner), `strategies.estimate_covariance` +
`_extract_current_weights` + `_smooth_turnover` + `_MLSignalStrategy` extended with all four levers
+ per-instance `name`, `ml_signals.apply_mu_transform` + `_validate_mu_transform` +
`VALID_MU_TRANSFORMS`, `backtest.CURRENT_WEIGHTS_KEY`, `params.yaml phase4c` block, `dvc.yaml` stage
8. **Latent cap-breach bug fixed** in `_as_weight_series` (renormalization inflated a weight resting
on the cap past the engine's tolerance; a turnover penalty made the boundary case routine) via
water-filling + a feasibility guard that raises on infeasible caps — the fix stayed in the producer,
not by loosening the engine's trust-boundary check. New tests: `tests/test_phase4c_cost_aware.py`
(all four levers, cap-projection edge cases, the new no-lookahead gate). Full suite: **266 tests**,
offline/synthetic, ~1m45. Hardening: `notebooks/phase4c_cost_aware.ipynb` (reads the committed
`phase4c_results.json` artifact — the full run is ~85 min), `docs/Livrable_Phase4C_Optimisation_Sensible_aux_Couts.docx`
(French), README/AGENTS.md updates.

---

## 12D. Phase 5 — Out-of-Sample Evaluation (✅ Core complete, 2026-07-22)

> ⚠️ **WORDING SUPERSEDED 2026-08-02 (§5.2, rule 2).** This section — and §12E, §12F, §12G, §12H
> after it — concludes "STATISTICALLY INDISTINGUISHABLE" from overlapping marginal CIs. That is an
> over-claim: failing to reject is not accepting the null, and marginal intervals are not a test of
> the difference. The measurements are correct; the licensed conclusion is **"no paired test of the
> difference was run"**. A paired block bootstrap on the return differences is the missing
> instrument and the highest-value outstanding evaluation task. Read every "indistinguishable" and
> "no significant edge" below with that substitution.

**Why this phase exists.** Every prior phase reported a *point* Sharpe from a *single* full-window
pass. Phase 4C ended on a near-tie (`rf_signal_shrunk` 1.117 vs. 1.1215) with hyperparameters
CHOSEN, not tuned. Phase 5 is the honest evaluator: it (1) selects the F7 knobs leak-free, (2)
reports the verdict on data the selection never saw, and (3) attaches an error bar to every number.

**The honest-selection architecture (a strict time-ordered split):** final `test_frac=35%` of each
universe is a **frozen test segment** neither selector touches. On the train+validation portion:
**ML hyperparameters** (RF/XGB depth/leaf/lr) are selected by **purged+embargoed K-Fold** scored by
**information coefficient** (Spearman rank-corr of predicted vs. realized returns — the right metric
for a signal); **portfolio levers** (`shrinkage_weight`, per-model `turnover_penalty`) by
**validation-segment net Sharpe** through the real engine. Two objectives, two tools — using K-Fold
for the net-Sharpe levers would be a category error. Then FREEZE and evaluate on the test segment
alongside `regime_conditional` (the hurdle) and `equal_weight` re-evaluated on the *same* test dates.

**New modules:** `src/purged_kfold.py` (custom `PurgedKFold`, López de Prado Ch.7, NOT `mlfinlab`;
purges by DATE across the `(Date, ASSET)` panel; the leakage gate `tests/test_purged_kfold.py` must
be green before anything consumes it), `src/model_selection.py` (`information_coefficient`,
`select_ml_hyperparameters`, `select_portfolio_levers`), `src/run_phase5.py`. `src/metrics.py`
gained `block_bootstrap_sharpe_ci` (circular block bootstrap, monthly blocks, seeded) and
`DSRTrialLedger` (closes the §17.1 DSR N-accumulation gap: N = configs evaluated in the search that
produced the reported strategy). `params.yaml` `purged_cv`/`phase5` blocks; `dvc.yaml` stage 9.
Full suite: **297 tests**, offline/synthetic.

**The real result (live Gold, `python src/run_phase5.py`, ~43 min, zero fallbacks) — the project's
punchline.** On held-out data with block-bootstrap 90% CIs:

| Universe (frozen test window) | tuned RF | tuned XGB | `regime_conditional` | `equal_weight` | Verdict |
|---|---|---|---|---|---|
| `etf_2017` (2023-03→2026-07) | 1.438 [0.70, 2.31] | **1.449** [0.72, 2.31] | 1.373 [0.63, 2.20] | 1.407 [0.67, 2.28] | tuned XGB edges hurdle |
| `full_2021` (2024-10→2026-07) | 0.785 [−0.12, 1.76] | 0.886 [−0.06, 1.90] | **1.199** [0.19, 2.28] | 0.866 [−0.20, 2.00] | hurdle clearly leads |

**The honest conclusion: out-of-sample, the tuned F7 models and the regime baseline are
STATISTICALLY INDISTINGUISHABLE on both universes.** The point-estimate winner even *flips* by
universe, but every CI is wide (~1.5–2.4 Sharpe span on 1.7–3.3-year test windows) and heavily
overlapping — none of the differences (in either direction) is significant. The earlier phases'
tidy point comparisons (1.117 vs. 1.1215) were within the noise all along, which is exactly what a
rigorous OOS evaluation exists to reveal. The Deflated Sharpe of the best tuned strategy against the
whole 36-configuration search is **0.92** on `etf_2017` (credible) and **0.67** on `full_2021`
(quite possibly a selection artifact). The CV information coefficients are tiny (0.015–0.036) —
honestly weak, as financial return prediction usually is.

**What this means for the deliverable (and it is a strong result, not a disappointment):** the brief
demanded "une attention particulière portée à la robustesse, au risque de surapprentissage". Phase 5
is that attention made real — leak-free tuning, held-out testing, confidence intervals, deflated
Sharpe — and its finding is that the ML stack adds **no statistically-significant edge** over a
well-constructed regime-switching baseline out-of-sample. The evaluation machinery that *proves*
this rigorously is itself the phase's deliverable. The tuning also empirically confirmed Phase 4C's
premise: it selected different levers per model AND universe (`etf_2017` RF penalty 0.0 vs. XGB 1.0;
shrink 0.5 vs. 0.75), so a single global value was indeed wrong.

**Limitations (explicit):** the `full_2021` test window is short (~1.75 yr / ~455 rows) — its DSR of
0.67 and very wide CIs partly reflect that; a nested walk-forward with periodic re-selection (the
documented stretch) would use the whole window as OOS and is the honest next step if a tighter
`full_2021` verdict is wanted. **✅ DONE 2026-07-28 — see §12G.** DSR N counts this search's configs,
not all experiments ever (a deliberately bounded, defensible definition). Hardening pass (notebook,
French deliverable, README) pending as a follow-up PR.

---

## 12G. Nested Walk-Forward — the CI-width problem, attacked (2026-07-28)

**Why.** §12D's own limitation, executed: every `full_2021` comparison sat inside ~2.2-Sharpe
intervals, which is a *sample-size* problem no better model fixes. `experiments/nested_walkforward.py`
re-selects the F7 configuration at each of **6 outer boundaries** and concatenates every OOS segment
— **793 OOS rows vs the single split's 455 (1.74×)**, with selection never seeing its own evaluation
window. Scoped to `full_2021`; `etf_2017`'s test segment is already 7.6 yr after the deep window.

**Result — pre-registered outcome (B): intervals narrowed AND the ranking changed.**

| Strategy | Single split | width | **Nested** | width | narrowing |
|---|---:|---:|---:|---:|---:|
| `regime_conditional` | 1.213 | 2.167 | **1.672** [0.89, 2.51] | **1.612** | 25.6% |
| `xgb_signal_tuned` | **1.308** | 2.260 | 1.436 [0.70, 2.26] | 1.558 | 31.1% |
| `min_variance_lw` | — | — | 1.417 | 1.703 | — |
| `max_sharpe` | — | — | 1.398 | 1.577 | — |
| `equal_weight` | 1.003 | 2.333 | 1.284 | 1.705 | 26.9% |
| `rf_signal_tuned` | 1.040 | 2.125 | 1.256 | 1.472 | 30.7% |

**Mean CI width 2.221 → 1.587 (−28.6%)**, beating the 24.3% pure-√n prediction. DSR **0.835 over 198
configs** (vs 0.67 over 36) — broader search, higher deflated Sharpe.

Three things to keep separate, and the third is the one most likely to be misquoted:
1. **The narrowing is the robust result** — a property of the design, not the period.
2. **Every lower bound is now positive** (0.478–0.893). In the single split `equal_weight` ran to
   **−0.090**. On this universe every strategy examined is now credibly positive OOS — a real change
   in what can be claimed.
3. ⚠️ **The level shift is a PERIOD effect, not an improvement.** EVERY strategy's point Sharpe rose,
   because the nested window starts 2023-07 and the single split starts 2024-10. The extra 15 months
   were simply good ones. **Only within-run comparisons mean anything.**

**The ranking flipped back to `regime_conditional`** — making three different orderings across three
evaluations of `full_2021` (pre-correction: regime; corrected single split: xgb; nested: regime).
The honest reading is *not* "regime wins after all" but that **the point ordering is unstable to
evaluation design** — Phase 5's conclusion, demonstrated a third time. The nested estimate deserves
most weight (74% more OOS data, narrowest CIs), and it does make the dashboard's `ML_STRATEGY =
regime_conditional` choice the better-supported one — but [0.89, 2.51] vs [0.70, 2.26] overlap
almost entirely. **No significance claim in either direction.**

Two mechanism findings: the **selected levers move between folds** (RF shrinkage 0.25/0.75/0.50/
0.25/0.25/0.25) — Phase 4C's "one global λ is wrong" confirmed again, re-selection is doing real
work; and **fold 6 (2025-12→2026-07) is where both F7 models collapse** (RF −0.00, XGB +0.23) while
the concatenated total stays strong — a single split landing there would have reported a very
different verdict, which is exactly the fragility this design averages over.

Full analysis: `docs/NESTED_WALKFORWARD_EXPERIMENT.md`.

---

## 12H. Regime-Conditional Weight Cap — hypothesis falsified (2026-07-28)

**Why.** §10.1's finding says the cap out-regularizes every covariance model tried. Phase 4
conditions the *covariance* on the regime; the untested move is to condition *the cap* — tighten in
a detected bear (more shrinkage exactly when correlations spike: P1+P3), loosen in a bull. Required
**zero production code**: `RegimeConditionalStrategy` already takes sub-strategy instances, so it is
`MaxSharpe(max_weight=bull_cap)` + `MinVarianceLW(max_weight=bear_cap)`, engine given the looser cap.

**Design included the control that could kill it:** fixed-cap references (isolating "the level
helped") and an **INVERTED** variant — loose in bear, tight in bull, deliberately wrong. Outcomes
pre-registered. `MATERIAL_MARGIN = 0.05` was added after a smoke run returned +0.0016 and the
original bare `>` would have called it a win — the change makes the test STRICTER and is disclosed
in the script and the doc rather than folded in silently.

**Result — outcome (C) on BOTH universes.** No regime-conditional cap clears materiality.

| | baseline (25/25) | best candidate | INVERTED control | best fixed |
|---|---:|---:|---:|---:|
| `full_2021` | **1.2363** | `both_40_15` 1.2663 (**+0.030**) | 1.1430 (−0.093) | 1.2333 (−0.003) |
| `etf_2017` | **0.9371** | `aggressive_40_25` 0.8525 (**−0.085**) | 0.8899 (−0.047) | 0.9525 (+0.015) |

Five readings, and the second is the one that settles it:
1. **The hypothesised mechanism fails.** Tightening in bear does nothing (+0.0016 on `full_2021`,
   −0.117 on `etf_2017`). The small positive movement comes from *loosening the bull cap* — the
   opposite half of the idea, still immaterial.
2. **The control splits by universe.** INVERTED is worst on `full_2021` (−0.093, consistent with a
   real regime signal) but **beats the best correct-direction candidate on `etf_2017`** (−0.047 vs
   −0.085). A regime effect that reverses sign between universes is not a regime effect. Running
   only `full_2021` would have made the control look like supporting evidence.
3. **The 0.25 cap is at/near optimum on both** — retro-justifying a choice made as a business rule.
   Luck confirmed after the fact, not design.
4. **Cap degeneracy reproduced as an exact identity:** on `etf_2017` at cap 0.20,
   `fixed_minvarlw_200` and `fixed_maxsharpe_200` both return **0.7694** = `equal_weight` exactly,
   because `5 × 0.20 = 1.0` forces 1/N regardless of objective (§10.1).
5. **Regime switching itself still pays on `full_2021`** — every fixed-cap reference underperforms
   the regime baseline there. The Phase 4 finding is untouched.

Nothing significant: CI overlap 1.492 of ~1.53 (`full_2021`), 0.627 of ~0.71 (`etf_2017`); the JSON
records `statistically_significant: false` per universe.

**Five routes now tested** to beating regime + dynamic covariance — F7 prediction (Phase 5), 5×
price history (deep-Morocco), fundamentals, per-fold re-selection (§12G), regime-conditional
constraints (here). None significant. The finding is not that each idea failed individually but
that **at this sample size the evaluation cannot resolve differences of the size these ideas
produce** — which is itself the project's most defensible scientific claim.

Full analysis: `docs/REGIME_CONDITIONAL_CAP_EXPERIMENT.md`.

---

## 12E. Deep Moroccan Data Experiment — was the ML starved? (research, 2026-07-23)

**Why.** Phase 5's null result had two readings: the signal is absent, OR it was starved (9 assets,
~1.7-yr test window, huge CIs). Testable → this experiment. On 2026-07-22 the team hand-downloaded
20-year daily histories (2005–2024) for **17 BVC stocks + MASI** from investing.com (the free BVC
source only reaches 2021; no free *API* covers pre-2021 Casablanca — EODHD doesn't cover Morocco,
`yfinance` has only `IAM.PA` via Euronext). Raw CSVs under `data/bronze/morocco_investing/`
(gitignored).

**Universe.** The DEEP 12-asset panel (names with continuous 2005 history, 8 sectors), calendar-aligned
like `clean.py`: **12 stocks × 4,682 days = 56,184 pooled rows (≈5× `full_2021`), 6.75-yr held-out test
(2017-08→2024-05), includes the 2008 crisis.** Reuses the *exact* Phase 5 machinery (purged-CV IC
selection + frozen test + block-bootstrap CIs), so the bar is identical.

**Result (honest, the key finding).**
- **Stage A — the model got sharper:** purged-CV information coefficient rose from Phase 5's 0.015–0.036
  to **~0.068 (RF) / ~0.074 (XGB)** — ~2–4×, confirmed by both algorithms. The signal *was* partly
  starved.
- **Stage B — the portfolio did not:** held-out net Sharpes — RF **+0.34** [−0.41,+1.14], XGB +0.28,
  1/N +0.25, Markowitz +0.18, regime +0.07. The tuned ML are the best *point* performers (beat
  Markowitz and regime), but **every CI is wide and straddles zero → nothing significant**, and the
  ranking *flips* vs. Phase 5 (regime led there) — the signature of noise.

**Verdict:** more data made the model measurably smarter but produced **no statistically significant
portfolio edge**. The ceiling is **data QUALITY, not quantity** — prediction accuracy ≠ portfolio
performance, even with 5× the data. Rules out "more price history"; redirects the alpha search to
**fundamentals** (the next experiment). Strengthens the honest-win story: ML beats classical Markowitz
in point terms across the original universe *and* 20 years of deep Moroccan data.

**Artifacts:** `experiments/deep_morocco_starvation.py` (deterministic runner →
`data/gold/deep_morocco_results.json` + `deep_morocco_equity.parquet`),
`notebooks/deep_morocco_data_expansion.ipynb` (executed), `docs/DEEP_MOROCCO_EXPERIMENT.md`.
**Deferred production follow-ups:** splice investing.com deep history + BVCscrap recent (overlap
2021–2024) → continuous 2004→today; dividend adjustment; formalize into the medallion pipeline if the
deep universe is adopted. **Branch note:** built on `feature/phase5-oos-evaluation` (needs
`model_selection`/`purged_kfold`/`metrics` extensions not yet on `main`).

---

## 12F. Fundamentals Experiment — the third confirmation of the signal ceiling (research, 2026-07-23)

**Why.** The deep-Morocco experiment closed off "more price history" as the missing ingredient for
F7 and explicitly redirected the alpha search to **data quality → fundamentals**. This experiment
runs that follow-up: add point-in-time valuation ratios (P/E, P/B, P/S, D/E) as new per-asset
features on the existing F7 pipeline, then measure whether purged-CV IC and portfolio Sharpe both
lift, or just the first.

**Source & causal seam.** `stockanalysis.com` (S&P Global-sourced, free, `robots.txt` permits `*`)
publishes semi-annual financials as an inline JavaScript object literal in the initial HTML —
`src/fundamentals.py` parses it with a small tokenizer, no browser needed. Only 4 fields survived
across all 4 BVC tickers (IAM/ATW/CIH/BCP): `pe`, `pb`, `ps`, `debtequity` — profitability ratios
(ROE, margins) are annual-only for Moroccan issuers, an honest data limit recorded up-front. **Causal
discipline** is enforced by `apply_publication_lag(publication_lag_days=90)` — every period-end is
shifted forward by 90 business days to produce an `available_from` date, since stockanalysis.com
doesn't expose filing dates and AMMC gives issuers 60–90 days. Locked in by
`test_future_value_corruption_cannot_change_any_past_panel_row`, mirroring
`test_phase3_integration.py`'s guarantee for market features.

**Wiring** (5 lines net addition to production code). `ml_signals.attach_fundamentals_features`
adds per-asset fundamentals columns onto the F7 `(Date, ASSET)` panel — BVC assets get their own
values, ETFs get the cross-sectional BVC-median + `HAS_FUND=0` indicator so trees can gate. The
`extras["fundamentals"]` channel is opt-in in `fit_predict_expected_returns` — Phase 5's committed
pipeline stays bit-identical when the key is absent (existing tests unchanged, 319 pass).

**Stage A — IC lift on `full_2021` (strict apples-to-apples, same `(date, asset)` rows):**
- RandomForest: baseline mean IC **0.0281 ± 0.0196 → treatment 0.0581 ± 0.0379, lift +0.0301** —
  the highest ML IC this project has measured on `full_2021`, and it nearly doubles the baseline.
- XGBoost: 0.0291 → 0.0398, lift +0.0107 (much smaller but still positive).
- Feature importance: `FUND_pb` alone carries 15.75% of the RF's total importance (2nd feature
  overall); fundamentals collectively 21.7%. The tree really uses them, not noise-fitting.
- `HAS_FUND` importance = 0.0000 — the median-filled ETF rows don't confuse the tree, so the
  splits happen on the fundamentals themselves.

**Stage B — portfolio verdict on the frozen Phase 5 test window (2024-10-14 → 2026-07-20, 1.8 yr,
461 rows), 1 config each, no lever grid** (block-bootstrap 90% CIs):

| Strategy | Net Sharpe | 90% CI | Avg turnover |
|---|---:|:---:|---:|
| `equal_weight` | 0.866 | [−0.20, 2.00] | 0.035 |
| `regime_conditional` (Phase 4 hurdle) | 1.199 | [0.19, 2.28] | 0.321 |
| `rf_signal_baseline` (F7 prices only) | **1.214** | [0.27, 2.20] | 0.774 |
| `rf_signal_fundamentals` (F7 + fundamentals) | 0.882 | [−0.09, 1.89] | 0.392 |

**The honest verdict — `rf_signal` LIFT from adding fundamentals: −0.331 Sharpe.** Not statistically
significant (CIs overlap heavily), but every point-estimate goes the wrong way — of a feature that
doubled the CV IC. Fundamentals cut turnover roughly in half (0.774 → 0.392) as expected of a
slower-moving signal, but the stabilization killed the alpha in equal measure.

**The pattern is now three-for-three.** Phase 5 (F7 on prices, tuned honestly) → indistinguishable
from regime baseline; deep-Morocco (5× data) → IC ×2–4, no portfolio edge; fundamentals (this run) →
IC ×2, Sharpe **drops**. Three independent tests of "prediction accuracy ≠ portfolio performance"
all confirming the same ceiling — an empirical finding of this project, not a one-off. The valuable
takeaway for the EURAFRIC deliverable: the **regime + dynamic-covariance system** (Phase 4) is what
genuinely and reproducibly beats classical Markowitz on this universe by ~15% net Sharpe; the F7
return-prediction layer does not add statistically-significant value on top of it, and adding
richer features hasn't changed that.

**Notable within-run observation** (documented, not spun). `rf_signal_baseline` — my F7 without
Phase 5's `shrinkage_weight=0.5`/`turnover_penalty=1.0` — posted the highest point Sharpe of every
strategy in this comparison at **1.214**, edging even `regime_conditional` (1.199). Phase 5's own
*tuned* `rf_signal_shrunk` on the identical test window scored 0.785. All three (baseline F7, tuned
F7, regime) have wide overlapping CIs on the 1.8-year window, so the ranking is not to be trusted —
but Phase 5's aggressive regularization was arguably too much for this specific window. This is a
small-sample fragility, not a discovery, and it reinforces Phase 5's own conclusion that any point
comparison in this regime is within the noise.

**Artifacts:** `src/fundamentals.py` (scraper + causal panel), `tests/test_fundamentals.py` (15
tests including the future-corruption gate), `experiments/fundamentals_ic_lift.py`,
`experiments/fundamentals_portfolio.py`, `data/gold/fundamentals_features.parquet` +
`_manifest.json` + `_ic_lift.json` + `_portfolio.json`, `docs/FUNDAMENTALS_EXPERIMENT.md`.
319 tests total (20 new). `HAS_FUND` intentionally not used by the tree — kept for auditability, not
for feature reduction. **Branch note:** built on `feature/fundamentals-experiment` off `main`;
production code touched is minimal (`ml_signals.attach_fundamentals_features` +
`fit_predict_expected_returns` opt-in `extras["fundamentals"]` key + one params.yaml block); the
experiment scripts live under `experiments/`, not `src/`, matching the deep-Morocco pattern.

---

## 12I. Crisis-Window Behaviour — the P3 evidence, and an exploratory detection result (2026-07-30)

> ⚠️ **REFRAMED 2026-08-02 — this section previously called Result B "the ONLY significant
> result" and instructed you to lead with p = 0.03125. That framing was retracted** across the
> report, README, dashboard, API and `docs/CRISIS_WINDOWS_EXPERIMENT.md`. The numbers below are
> unchanged and correct; what changed is what may be claimed from them. See "Result B" for the
> current position, and §12I.1 for the two statistical arguments behind the retraction.

**Why.** P3 (diversification breakdown in crises) had the least direct evidence of the four
problems: every phase reported whole-period Sharpe/drawdown, none reported behaviour *during*
crises. The brief asks for "pertinence financière" — for an allocator, drawdown behaviour is that.
Cost: **zero compute**; `dashboard_equity.parquet` and `dashboard_regime.parquet` already held it.

**Method.** Five windows on **external published S&P 500 peak-to-trough dates**, fixed before any
result was inspected (deriving them from our own drawdowns would select the periods where we look
good). Drawdown vs the running all-time peak. `experiments/crisis_windows.py`.

**Result A — constrained optimization protects; 1/N does not, in all 5 windows.**

| Crisis | optimizers | `equal_weight` |
|---|---:|---:|
| GFC 2008 | -21.2% ret, -26.9% DD | **-30.2%, -36.2%** |
| EU debt 2011 | **+1.6%** (positive) | -5.4% |
| Q4 2018 / COVID / 2022 | lower DD, ~half the recovery time | — |

Recovery is the most consistent margin (37d vs 71d in COVID; 53d vs 105d in 2018) and appears
nowhere else in the deliverable.

⚠️ **Attribution: this credits the CONSTRAINT and covariance model (P1/P3), NOT the regime layer.**
In 3 of 5 windows the three optimizers are identical to the decimal — in a bear regime
`regime_conditional` *is* `min_variance_lw` by construction, and the 25% cap on 5 assets pins the
allocation (§10.1). Page 1 computes that tie-count from the artifact rather than asserting it.

**Result B — an EXPLORATORY association between the HMM's "bear" state and crisis windows.**

Bear rate **91.7% inside** crisis windows vs **29.2% outside** — risk ratio
**3.13×**, **5/5** crises above base rate.

- Sign test, each crisis = 1 observation, n=5: **p = 0.03125**. Reported as a *diagnostic*, kept
  in the artifact for transparency. It avoids serial dependence inside a window, but five events
  examined retrospectively do not support a confirmatory claim.
- Liberal Fisher exact over 248 rebalances: p = 7.78e-13 — treats serially
  correlated monthly regimes as independent; **optimistic, never quote alone**

**Do NOT describe this as statistically significant, as the project's only significant finding, or
as a validated crisis detector.** The defensible statement is narrower: *the detector is more often
in its lower-mean-return state inside these five externally-dated windows than outside them, and it
reaches that state causally, using only past data.*

*Caveat that must travel with it:* "bear" is DEFINED as the lower-mean-return state and crises are
low-return by construction, so part of the association is definitional. What is not definitional is
that detection is **causal and real-time** — the model has only past data, does not know a crisis is
beginning, and still flags it at 3× base rate *without* crying wolf (29% baseline, not 80%).

**Keep these two claims in different sentences:** the detector produces a causal reading of market
state associated with these windows; whether **acting** on it pays has not been established
(Phase 5, §12G).

Full analysis: `docs/CRISIS_WINDOWS_EXPERIMENT.md`.

### 12I.1 The two statistical arguments behind the 2026-08-02 retraction

Both are worth understanding, because they generalize to every result in this file.

1. **n=5, retrospectively chosen windows, partly definitional outcome.** A p-value below 0.05 on
   five post-hoc events with a state defined by the very quantity that characterizes the events is
   a diagnostic, not a test of generalization. The threshold was cleared; the inference it would
   license was not earned.
2. **⭐ Overlapping confidence intervals are NOT a test of difference — and this cuts both ways.**
   Every earlier phase (§12D, §12E, §12F, §12G, §12H) concluded "statistically indistinguishable"
   from overlapping CIs. That is an over-claim in the *opposite* direction: failing to reject is
   not accepting the null, and marginal (per-strategy) intervals say little about the distribution
   of the *difference*. The correct statement is **"no paired test of the difference was run"**,
   and the missing instrument is a **paired block bootstrap on the return differences**, which is
   now the single most valuable outstanding evaluation task in the project (§12D's stretch list).

**Rule going forward:** report point estimates as *observed differences*, report intervals as
*uncertainty quantification*, and never convert either into "superior" or "equivalent" without a
paired test. This is now the wording used in `README.md`, both dashboard pages, `src/api/main.py`,
`docs/CRISIS_WINDOWS_EXPERIMENT.md` and the report (`docs/rapport/`).

---

## 13. Phase 6+7 — Portfolio ML Suite (✅ Complete, 2026-07-25)

**Merged, not sequenced.** Phases 6 (production tool) and 7 (stakeholder dashboard) ship as ONE
Streamlit multi-page app under `dashboard/` (singular — user's call), plus a FastAPI service under
`src/api/`. Decided with the user 2026-07-25. Rationale: both surfaces read the same Gold artifacts
and need the same plot vocabulary, so a shared data layer saves ~40% of the work AND eliminates a
real drift risk — two separate dashboards quoting different Sharpes for the same strategy is the
exact §17.1 failure class this project already hit once with `phase2_hurdle.json`. The *pages* stay
distinct because the audiences and integrity rules genuinely differ.

**M1 — data layer + stakeholder page.**
- `src/run_dashboard_data.py` runs the 4 headline strategies (`equal_weight`, `min_variance_lw`,
  `max_sharpe`, `regime_conditional`) on both universes through the unmodified `run_backtest`
  engine, persisting `dashboard_equity.parquet`, `dashboard_weights.parquet`,
  `dashboard_regime.parquet` (the HMM timeline), and `dashboard_showcase.json`. Zero new modelling;
  every number derived, none typed. `dvc.yaml` stage 10 (`dashboard_data`).
- **The honesty gate — `tests/test_run_dashboard_data.py` (10 tests).** Beyond shape checks it
  asserts the headline lift is arithmetically consistent with the metrics the same run produced,
  that `best_classical`/`best_ml` are genuine argmaxes, that Phase 5 CIs are copied verbatim rather
  than recomputed, and — by inspecting the runner's own source — that **no Sharpe-like literal is
  hardcoded**. A stakeholder-facing surface is the worst place for a stale number, so the invariant
  is a test, not a convention.
- `dashboard/pages/1_Histoire_de_valeur.py` — French, no controls, 6 sections. Title since
  2026-08-02: **« Résultats de recherche — allocation assistée par ML »** (the *file* is still
  `1_Histoire_de_valeur.py`, so the Streamlit sidebar still shows the old name — a known
  inconsistency, §17.11). Opens with a `st.warning` stating it is a research prototype, not an
  investment tool. **Integrity constraints enforced in code:** `ML_STRATEGY` is a single constant
  = `regime_conditional` (F7 is never presented as our value-add); every headline is described as
  an *observed* point estimate with its interval; the `etf_2017` case where the system **loses**
  is shown, not omitted.

**M2 — API + strategy explorer.**
- `src/api/main.py` — 6 endpoints (`/health`, `/strategies`, `/metrics`, `/equity`, `/weights`,
  `/compare`). Deliberately THIN: serves committed artifacts, never refits on request (a
  `regime_conditional` walk-forward is ~4-13s — unusable in a handler, and it would burn CPU on
  every interaction). **`/compare` ships the CI and the caveat inseparably with the lift** — the
  machine-readable counterpart of Page 1's integrity rule; quoting the gain dishonestly requires
  actively discarding fields. `tests/test_api.py` (12 tests, TestClient, no live server).
- `dashboard/pages/2_Outil_gestionnaire.py` — titled **« Explorateur de stratégies »** since
  2026-08-02 (file name unchanged, §17.11). Universe/strategy pickers, date range, net-vs-gross
  toggle, metrics table, latest allocation + CSV export, full rebalance history. Shows ALL
  strategies including F7, with the "no paired test establishes a difference" caveat attached and
  an explicit non-advisory disclaimer.

**Real numbers this pass produced** (live Gold, one run, ~19s):

| Universe | Best classical | Best ML | Lift |
|---|---|---:|---:|
| `full_2021` | `equal_weight` 0.981 | `regime_conditional` **1.121** | **+14.3%** ⚠️ |
| `full_2021` *(current, post-correction)* | `max_sharpe` **1.1644** | **1.2363** | **+6.2%** |
| `etf_2017` | `max_sharpe` 0.930 | `regime_conditional` 0.893 | **−4.0%** (honest loss, displayed) |

**Verification was real, not assumed.** Rendered in a browser; found and fixed two layout bugs that
way (a title/legend collision; CI whiskers clipping the plot edge with value labels underneath
them). Page 1 fully verified visually. Page 2's rendering verified visually on `etf_2017`; its
`full_2021` path verified **programmatically** (the Streamlit combobox wouldn't open in the headless
browser) — asserting weights sum to 1, the 25% cap holds, CSV export works, and all 9 BVC+ETF assets
are present. Stated plainly rather than papered over.

**341 tests** (22 new). *(390 as of 2026-08-01.)* `requirements.txt` gained `streamlit`, `fastapi`, `uvicorn`, `httpx`.
**Explicitly out of scope, documented not dropped:** on-demand re-optimisation at arbitrary as-of
dates, custom constraints beyond `max_weight`, auth, cloud hosting.

---

## 14. File Structure

```
portfolio_ml/
├── AGENTS.md                      ← this file (gitignored; local context)
├── Codex.v2.md                   ← superseded, kept as history only — do not edit
├── README.md                      ← French, for EURAFRIC
├── requirements.txt / params.yaml / pytest.ini / dvc.yaml / workspace.yaml
├── .env                           ← FRED_API_KEY (never commit)
├── docs/
│   ├── Livrable_Phase1_*.docx / Livrable_Phase2_*.docx / Livrable_Phase3_*.docx  (French, committed)
│   └── PHASE1_WALKTHROUGH.md      ← zero-context team walkthrough (committed)
├── scripts/setup_launchd.sh       ← unattended Dagster (macOS LaunchAgents)
├── data/                          ← DVC-managed, gitignored (except dvc.lock, which is git-tracked)
│   ├── bronze/  raw_prices | raw_macro | bvc_prices | raw_bam_macro .parquet
│   ├── silver/  log_returns[_etf].parquet | validation_report[_log_returns_etf].json
│   └── gold/    log_returns[_etf] | macro_features | stationarity_report | ml_features_etf |
│                ml_features_full | ml_features_manifest.json .parquet/.json
│                + phase2_hurdle.json (NOT DVC-tracked — §17.1)
├── notebooks/   phase1_eda.ipynb  phase2_backtest.ipynb  phase3_features.ipynb
├── src/
│   ├── pipeline.py  ingest.py  clean.py  features.py  schemas.py  utils.py     (Phase 1)
│   ├── strategies.py  backtest.py  metrics.py  run_backtest.py                (Phase 2)
│   ├── ml_features.py                                                          (Phase 3)
│   └── orchestration/ assets.py  definitions.py     (8 assets, all wired — §17.7)
├── tests/  (126 tests, offline, ~4 s — `python -m pytest tests/ -q`)
└── mlruns/                        ← MLflow (auto-generated)
```

---

## 15. Non-Negotiable Technical Decisions

Do not change without raising the issue and explaining why.

1. **Log-returns, not simple returns.** `np.log(p/p.shift(1))`, never `pct_change()`.
2. **ADF and KPSS together, never alone.** Opposite null hypotheses; disagreement = AMBIGUOUS =
   investigate manually, never assume (see EEM, §8.4).
3. **`auto_adjust=True` in every yfinance call.** Splits/dividends otherwise create fake returns.
4. **Forward-fill only for calendar gaps, then drop the initial NaN window.** Backfill and
   interpolation use future information — banned.
5. **Bronze is immutable** (except `bvc_prices.parquet`, which is an intentional append-merge —
   see §8.2 — not a violation).
6. **FRED key via `FRED_API_KEY` env var.** Never hardcoded.
7. **DuckDB for analytical queries on Gold.** Not pandas groupby/apply.
8. **Macro features lagged ≥ 1 day — enforced by a raised exception**, not convention (Phase 1
   Gold macro *and* Phase 3 ML macro signals, independently configured in `params.yaml`).
9. **Every function docstring states its problem(s):** `Addresses: P1, P2 — ...`.
10. **Dagster schedules the pipeline; it does not contain pipeline logic.** Assets call existing
    functions unchanged (approved 2026-06-29). Every medallion output that should refresh on a
    schedule must be a wired asset — the 2026-07-20 incident (§17.7) was exactly this rule being
    violated silently for two outputs.
11. **Custom purged K-Fold; no `mlfinlab` dependency.**
12. **Every Phase ≥ 2 strategy runs on both universes (§10.2) and under constraints (§10.1);
    headline results are net-of-cost, out-of-sample.**
13. **Silent data loss is a bug, always.** Any code path that discards rows/columns/history must
    log what it dropped and why.
14. **No global standardization of ML features.** Every model fits its own scaler inside its own
    walk-forward training window (§11) — standardizing once over full history is a lookahead leak.
15. **The `Strategy.fit(train_returns, extras=None)` interface is the only way Phase 4 code may
    touch backtest data.** The engine — not the strategy — is responsible for slicing `extras` to
    `:τ`; do not bypass it by reading feature Parquet files directly inside a strategy.

---

## 16. Coding and Testing Conventions

- Python ≥ 3.10; type hints on all signatures; Google-style docstrings that explain *why*.
- `logging` throughout, never `print()`. INFO = pipeline steps, DEBUG = internals,
  WARNING = data-quality events a human should read.
- Explicit, descriptive exceptions; never `except: pass`.
- **Every function in `src/` has at least one test.** Small synthetic fixtures, offline (network
  mocked), suite finishes in seconds (126 tests, ~4 s total).
- Name tests after the rule they lock in (`test_forward_fill_not_backfill`,
  `test_future_feature_values_cannot_change_past_weights`), not the function they call — the suite
  doubles as documentation of *why* rules exist.
- When claiming a fact in docs/notebooks (a correlation, a coverage range, a Sharpe number),
  verify it against current data first — see §17.1 for exactly this failure mode with the Phase 4
  hurdle.

---

## 17. Maintenance Duties (Recurring)

Things that rot silently if nobody acts.

1. ~~`data/gold/phase2_hurdle.json` is not DVC-tracked.~~ **Resolved 2026-07-20** (same day it was
   found stale): `dvc.yaml` gained `phase2_hurdle` and `phase4_compare` stages
   (`python src/run_backtest.py` / `python src/run_phase4.py`), so both `phase2_hurdle.json` and
   `data/gold/phase4_results.json` now regenerate automatically via `dvc repro` whenever their
   upstream Gold parquets or source code change — no more manual-rerun discipline required.
2. **`_TAUX_DIRECTEUR_DECISIONS` (src/ingest.py).** BAM meets ~quarterly; append every decision
   (even holds). Runtime warning fires at >100 days stale.
3. **BVC pre-2021 data.** Periodically reassess whether a paid/alternate vendor is worth it — it
   would restore COVID to the 9-asset universe and unify the dual-universe design.
4. **Yahoo Finance fragility.** BVC tickers routinely fail on yfinance (BVCscrap is the real
   source). Diagnosis order: check VPN → `curl` Yahoo directly → wait out 429s. Never "fix" this by
   loosening pipeline validation.
5. **DVC snapshots after data changes.** After any pipeline run that modifies `data/`, run
   `dvc commit` and commit the updated `dvc.lock`. No DVC remote is configured — cache is
   local-only (`.dvc/cache/`); a shared remote is a team decision for later.
6. **Dagster daemon health on a laptop.** The daemon *process* can stay alive while its internal
   threads (scheduler included) are dead — zero ticks fire, silently. Check daemon health via the
   UI (Overview → Daemons) or `curl 127.0.0.1:3000/graphql` → `daemonHealth`, not just
   `launchctl list`/the process table.
7. **2026-07-20 incident — Dagster asset graph gap.** `log_returns_etf` and the Phase 3
   `ml_features_layer` were never wired into the scheduled asset graph at all (only the 9-asset
   `log_returns` was) — so the ETF universe and all Phase 3 features silently went stale relative
   to Bronze on every scheduled run, undetected until a manual `dvc status` caught it. Fixed in
   PR #6 (`src/orchestration/assets.py`, `definitions.py`); the graph now has 8 assets, validated
   via `dagster definitions validate`. Lesson for Phase 4: **any new Gold-layer output Phase 4
   introduces must be added as a Dagster asset in the same PR that creates it**, not as a follow-up.
8. **2026-07-27 — the dividend fix was one scrape away from silently undoing itself.** Found in
   review, in three connected pieces, all now closed:
   - The BVC dividend cache (`data/bronze/bvc_dividends/`) was **not** a declared dep of the
     `clean` DVC stage and **not** a Dagster asset — the single input that makes BVC returns
     total-return was invisible to both lineage graphs. Now `dvc.yaml`'s `clean` deps include it
     (plus `src/dividends.py`), and `bvc_dividends` is a Bronze asset upstream of `log_returns`
     (9 assets total).
   - `silver_pipeline` caught **any** scrape failure, logged one WARNING and continued with
     price-only returns. On the unattended schedule nobody reads WARNINGs, so the ~3.0–4.3%/yr
     understatement would have returned undetected. `require_dividends=True` (passed by
     `pipeline.py` and the Dagster asset) now raises `DividendDataUnavailable` instead, and an
     **empty** scrape counts as failure — a zero-row result is the same corruption wearing a
     success costume.
   - **The test suite was reaching the live network.** `data/bronze/` is gitignored, so the cache
     is not in the repo; on a fresh clone `test_pipeline.py` hit casablanca-bourse.com for real,
     contradicting §16 and the README's offline promise. Proven by hiding the cache: the suite
     stayed **green** while writing price-only returns — the broad `except` swallowed the failure.
     `tests/conftest.py::_no_network` (autouse) now blocks outbound connections, allowing loopback
     so FastAPI's TestClient still works; `allow_network` is the explicit opt-out.
   Lesson, generalizing §17.7 beyond Dagster: **an input that changes a number must be visible to
   every graph that claims to track it** — DVC deps, Dagster assets, and the test suite's
   hermeticity are three such graphs, and this input was missing from all three.
9. **2026-07-27 — the nightly job had been failing silently, and we found it by accident.**
   While checking DVC status mid-review, Bronze parquets showed a 23:00 mtime nobody had
   triggered. The scheduled run (`0 22 * * 1-5` UTC = 23:00 local) had fired and **failed**:
   ```
   ModuleNotFoundError: No module named 'dividends'
     src/clean.py:278  from dividends import load_bvc_dividends   # lazy, inside silver_pipeline
   ```
   **Root cause:** Dagster's gRPC code server (pid started **2026-07-24 22:58**) long-outlived the
   creation of `src/dividends.py` (**2026-07-25**). A fresh process imports the module fine; that
   three-day-old server never could. `--lazy-load-user-code` reloads *definitions*, not a stale
   interpreter's import state.
   **What it cost:** `log_returns` (9-asset) died while `log_returns_etf` succeeded, so Bronze and
   the ETF Silver universe advanced to 07-27 while the 9-asset Silver universe stayed at 07-25 —
   the §8.2 dual-universe drift, again. Gold was NOT rebuilt (it depends on the failed asset), so
   **every committed result remains internally consistent**; the damage was confined to Silver.
   **Also found:** a run from **2026-07-24 23:22 is still in `STARTED` state** — a 3-day zombie.
   This is §17.6's "daemon alive, internals dead" hazard in the flesh: `pgrep` showed healthy
   processes the whole time.
   **Fixed here:** `from dividends import ...` promoted to a module-level import in `clean.py` and
   `assets.py`, so this class of breakage surfaces when the code location LOADS (which Dagster
   reports immediately and `dagster definitions validate` catches) instead of 30 s into a run.
   Plus `tests/test_orchestration.py` — **nothing in the suite imported the orchestration package
   at all**, so 365 tests passed against a pipeline that could not start. It now asserts the code
   location imports and the asset graph matches expectations, on every commit.
   **Still outstanding (needs a human):** the running daemon is still serving stale code. Restart
   it with
   ```
   launchctl kickstart -k gui/$(id -u)/com.portfolioml.dagster-daemon
   ```
   Deliberately NOT done automatically: a restart may launch a catch-up run that rebuilds
   Bronze→Silver→Gold, and refreshing Gold invalidates `phase5_results.json` (the new freshness
   guard would then correctly refuse to publish the dashboard) — i.e. it silently commits you to
   another ~3.5 h Phase 5 rerun. That is a scheduling decision, not a cleanup step.

10. **2026-08-01 — the last two convention-only rules became executable.** A full review found
   the project enforced almost every rule it cared about in code (engine-side weight cap,
   no-lookahead slicing, water-filling rather than loosening a check, the no-hardcoded-Sharpe
   test, the Phase 5 freshness guard, `_no_network`, the orchestration import test) — but the two
   that had caused the most actual damage were still discipline:
   - **Cross-artifact consistency.** `tests/test_artifact_consistency.py` asserts every committed
     result artifact agrees on shared strategies, OOS windows and hurdles. Verified by replaying
     the real incidents against it on a copy: it catches the 2026-07-30 torn `phase4c` (both via
     the OOS-window check and the baseline check), the 2026-07-25 stale Phase 5 CIs, and
     dual-universe end-date drift. All four previously required a human noticing an mtime.
   - **P1–P4 traceability** (§18's first requirement). `tests/test_traceability.py` enforces it,
     counting function-, class- *and* module-level `Addresses:` (class-level is how
     `strategies.py` legitimately documents it). 107/110 public functions pass; 3 infrastructure
     functions are grandfathered with written reasons and the list may only shrink. It also
     rejects `Addresses:` lines that cite no P-number — which caught two on its first run
     (`api/main.py`, `run_dashboard_data.py` both had the label without a problem; both now name
     P4 and say why).
   Also fixed in the same pass: the API's `lru_cache(maxsize=1)` pinned artifacts for the process
   lifetime — the **third** instance of the stale-cache class after Dagster's code server and
   Streamlit's module cache — now keyed on `(path, mtime)`.

11. **2026-08-02 — the claim reframing (§5.2) has FIVE known leftovers.** The sweep covered the
   README, both dashboard pages, the API, `docs/CRISIS_WINDOWS_EXPERIMENT.md` and the report; 390
   tests still pass and `tests/test_api.py` was updated (not deleted) to assert the *new* caveat
   keyword. Outstanding, in priority order:
   - **`docs/Livrable_Phase8_Etudes_Robustesse.docx` still carries the retracted wording**
     ("seul résultat" ×1, "statistiquement significatif" ×3, "exploratoire" ×0). It is a committed
     supervisor deliverable that now contradicts the report. Regenerate via
     `scripts/build_livrable_phase8.py` after editing the source strings.
   - **The artifact tells its own readers to over-claim.** `data/gold/crisis_windows.json` carries
     `regime_detection.*.significance.note = "Lead with the conservative sign test…"`, and
     `/crisis` serves that block verbatim. The string lives in `experiments/crisis_windows.py`
     (~line 224); re-running the experiment regenerates it unchanged.
   - **Page names are inconsistent in three places.** Titles are now *Résultats de recherche* and
     *Explorateur de stratégies*, but the FILENAMES (`1_Histoire_de_valeur.py`,
     `2_Outil_gestionnaire.py`) drive the Streamlit sidebar, so the nav still shows the old names;
     page 1's `st.set_page_config(page_title=...)` was also left as "Histoire de valeur" while its
     `st.title` changed. `docs/rapport/chapters/Chapter5.tex` §"Le produit livré" still names the
     pages the old way. Renaming the files changes the URLs — decide deliberately.
   - **Chapter 5's taxonomy no longer matches its own summary.** Its intro promises three
     categories (observe / explore / réfute); the *Synthèse* still uses five labels (Exploré /
     Observé-non-démontré / Observé-robuste / Découvert / Réfuté).
   - **`docs/rapport/main.pdf` is stale** relative to the reworded chapters — rebuild with
     `cd docs/rapport && tectonic -X compile main.tex`.
   Lesson, and it is the same one as §17.8 in a new dress: **a claim is an artifact too.** It
   lived in six surfaces plus a binary deliverable plus a JSON field that the API re-serves, and
   nothing in the test suite could tell they had drifted apart — `test_artifact_consistency.py`
   compares *numbers*, not *wording*. Editing prose in one place is exactly as unsafe as editing a
   number in one place.

---

## 18. Supervisor Requirements

Abdelmouttalib's review process, non-negotiable:

- **Traceability first.** Every component states which of P1–P4 it addresses; untraceable code
  fails review.
- **MVP over perfection.** Ship end-to-end first; sophistication second.
- **Defensibility.** Each team member must explain every design decision without notes.
- **No lookahead bias.** Any path where future data influences past decisions is an automatic
  failure. This is the single most important correctness criterion, and it is now tested
  end-to-end across Phase 2 and Phase 3 (§11), not just per-module.
- **Honesty about limitations.** State data gaps and open questions proactively (§8, §12) rather
  than letting them be discovered.

**Primary references:**
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley — Ch. 7 (Purged
  K-Fold), Ch. 8 (Feature Importance); Deflated Sharpe Ratio.
- Tsay, R. S. (2010). *Analysis of Financial Time Series*. Wiley — GARCH and multivariate models.
- DeMiguel, Garlappi & Uppal (2009). "Optimal Versus Naive Diversification" — why 1/N is the
  honest benchmark (reproduced on our data, §10.4).
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed., otexts.com/fpp3) —
  stationarity background.

---

*Updated 2026-07-20 (later same day): Phase 4 core marked complete — HMM regime dispatch (2-state,
hard switch) + the full covariance ablation ladder (Ledoit-Wolf → EWMA → DCC-GARCH, all four rungs
now built) shipped and benchmarked against the Phase 2 hurdle on live data (§5, §12). Real result:
beats the hurdle on `full_2021` (`regime_conditional`, 1.122 vs. 0.975), does not on `etf_2017`
(stays `max_sharpe`, 0.936) — an honest mixed finding, not spun as a universal win. `phase2_hurdle.
json`'s DVC-tracking gap (§17.1) closed same-day via two new `dvc.yaml` stages. 159 tests green.
Hardening pass (notebook, French deliverable, README) pending as a follow-up PR.

*Updated 2026-07-21 (later): Phase 4C (cost-aware optimization + μ regularization) complete and
merged to `main` (PRs #11 core, #13 the cap-fix + Sourcery follow-ups; §12C). It attacks Phase 4B's
*diagnosed* failure — `rf_signal` had the best gross Sharpe of the whole comparison and lost it to
turnover — with a turnover-penalized objective, μ shrinkage/rank tilt, and cov-estimator selection.
Honest **near-miss**: no variant beats the Phase 4 hurdle, but `rf_signal_shrunk` reaches 1.117 vs.
1.1215 on `full_2021` (within 0.4%, untuned), confirming the Chopra-Ziemba diagnosis; the turnover
penalty backfired on the good model yet helped the over-trader, proving a single global λ is wrong
and handing Phase 5 a concrete tuning job. A latent renormalization cap-breach bug (surfaced by the
first live 4C run, `xgb_signal_cost` on GLD) was root-caused and fixed in the producer via
water-filling, not by loosening the engine's check. 266 tests. Notebook + French deliverable shipped.

Updated 2026-07-21: Phase 4B (F7 — adaptive ML signal models) core complete — RandomForest and
XGBoost pooled cross-sectional return-prediction strategies built, tested (213 tests total), and
benchmarked live against the Phase 4 hurdle (§12B). Real result: an honest NEGATIVE finding —
neither strategy beats its universe's hurdle, with `xgb_signal`'s turnover on `full_2021` (>1.0)
the standout red flag. LSTM (the diagram's third F7 model family) was fully built and tested
standalone, then dropped after a torch+xgboost segfault when run together in the same process —
deferred to more capable hardware, not abandoned; all LSTM code removed from this branch rather
than shipped unstable. Phase 4's own hardening pass (notebook, French deliverable, README) shipped
same-day via PR #8, closing out that phase's pending item from the entry below.

Earlier 2026-07-20: full resync with actual repo state before Phase 4 kickoff. Phases 2 and 3
marked complete with their real merged-PR history, test counts, and hurdle numbers (previously this
file still showed Phase 1 "Active" and Phases 2–3 "Pending", which was stale by several weeks).
Consolidated the code-embedding original AGENTS.md and the interim Codex.v2.md into this single
file going forward — Codex.v2.md is now history only. Documented the phase2_hurdle.json staleness
finding and fix and the Dagster asset-graph gap incident discovered the same day.*

*Updated 2026-08-02: claim reframing (§5.2), applied by the team across README, both dashboard
pages, `src/api/main.py`, `docs/CRISIS_WINDOWS_EXPERIMENT.md` and the whole report. Two statistical
arguments drive it (§12I.1): a p-value on five retrospectively-chosen windows with a partly
definitional outcome is a diagnostic, not a significance result — so "the project's only
statistically significant finding" is RETRACTED; and overlapping marginal CIs are not a test of a
difference, so "statistically indistinguishable" was an over-claim in the opposite direction and
the honest statement is "no paired test was run". The deliverable is now described everywhere as a
research prototype, explicitly not an advisory or production tool. 390 tests still pass;
`tests/test_api.py` was updated to assert the new caveat rather than dropped. FIVE leftovers are
tracked in §17.11 — the Phase 8 .docx, the `crisis_windows.json` note the API re-serves, the
dashboard page-name mismatch, Chapter 5's taxonomy, and the stale `main.pdf`.*
