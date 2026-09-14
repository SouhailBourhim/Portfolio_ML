# Portfolio ML — Robust, auditable portfolio optimisation

[![CI — main](https://github.com/SouhailBourhim/Portfolio_ML/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SouhailBourhim/Portfolio_ML/actions/workflows/ci.yml?query=branch%3Amain)
![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![DVC](https://img.shields.io/badge/Data-DVC%20%2B%20R2-945DD6)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-891-success)

**Final-year project (PFA) — INPT × EURAFRIC Information**<br>
**Team:** Souhail Bourhim · Zakarya El Wali · Yasmine Bouajine<br>
**Supervisor:** M. Abdelmouttalib Maqil

*[Version française](README.fr.md)*

## The project in one minute

Portfolio ML is a research prototype in portfolio management. It studies how to allocate capital
across Casablanca Stock Exchange (BVC) equities and international ETFs under realistic management
constraints: long-only positions, a per-asset weight cap, and transaction costs.

The system pits classical Markowitz allocations against data-driven extensions — dynamic
covariance, HMM market regimes, and Random Forest / XGBoost challengers. The question is not
"which model shows the best backtest?" but "which result survives strictly temporal evaluation,
costs, multiple testing, and data lineage?".

This is an academic research demonstrator: not financial advice, not a client recommendation, not
order execution.

> A reproducible research chain for determining whether ML complexity actually improves a portfolio
> allocation once constraints, costs, statistical uncertainty and model selection are accounted for.

[Final report — PDF](output/pdf/Rapport_PFA_Final_2026.pdf) ·
[Defense presentation — PowerPoint](output/presentation/Soutenance_PFA_Portfolio_ML_INPT_EURAFRIC.pptx) ·
[Executed notebook — `global_2004` evidence](notebooks/phase10_global_2004_evidence.ipynb) ·
[Model governance](docs/MODEL_GOVERNANCE.md) ·
[Explainability](docs/EXPLAINABILITY.md)

**Final research state.** All three evaluations are complete. Across both release universes and the
pre-registered `global_2004` extension, **no ML layer establishes outperformance against its primary
comparator**. The central result is therefore not a "winning" model but an evidence chain able to
tell a seductive backtest maximum apart from an edge that survives costs, time and data snooping.

![The four problems this project addresses](docs/rapport_final/assets/figures/quatre_problemes.png)

## Why this project is different

The repository does not try to manufacture Machine Learning as the winner. It builds the controls
needed for a result to be **refutable**: strictly temporal validation, paired comparison, data
snooping correction, data provenance, fallback observability, and automated consistency between
artifacts, API, dashboard, model cards and report.

The most important correction illustrates this. After converting the mixed universe's ETFs to MAD
using the official Bank Al-Maghrib rate, the observed edge of the regime strategy **flipped sign**.
The release keeps that negative result instead of rebuilding the narrative around a friendlier
benchmark.

### What is delivered

- **Bronze → Silver → Gold** data pipeline, contract-validated and DVC-versioned.
- Long-only walk-forward backtest, 25% per-asset cap, transaction costs deducted.
- Classical baselines: `equal_weight`, minimum variance, Ledoit–Wolf, maximum Sharpe.
- Dynamic covariance: EWMA and DCC-GARCH.
- Market regimes: two-state HMM with conditional allocation.
- Supervised challengers: Random Forest and XGBoost on a causal per-asset panel.
- Forward-only selection with purge, embargo and a frozen final test.
- Paired bootstrap, White Reality Check and Hansen SPA.
- Exact explainability, per-fit telemetry, model cards and a challenger policy.
- Streamlit dashboard, read-only FastAPI service, Docker, CI and release gates.
- **891 automated tests** in the repository's final state.

## Headline results

| Question | Current result | Permitted interpretation |
|---|---|---|
| Does the regime system beat Markowitz on `full_2021`? | Net Sharpe **0.9571** vs **1.0690**, observed gap **−10.47%** | Descriptive result unfavourable to ML; no paired test of this difference establishes the reverse superiority. |
| Does the regime system beat the best ETF reference? | **0.9371** vs **0.9525**, observed gap **−1.62%** | ML produces no observed gain on this universe. |
| Does any challenger win after 240 trials? | **Not established** by White RC or Hansen SPA against the pre-specified primary comparator | The benchmark choice is not rewritten after seeing the result. |
| Does the regime system win when the universe can actually express the allocation? | On `global_2004`: **0.8923** vs **0.9785**, ΔSharpe **−0.0862**, 90% CI **[−0.2133, +0.0414]** | Q1 establishes no Sharpe outperformance in a 10-ETF universe producing 249 distinct allocations out of 249. Nor does the interval demonstrate the reverse. |
| Is the best result from a fresh 240-challenger search credible? | Raw maximum **+0.0930**, but White RC **p = 0.9045** and Hansen SPA **p = 0.8656** | Q2 shows why the best observed candidate must not be turned into a post-selection result. |
| Is more Moroccan data enough? | The CI widens by **2× to 4×** across 12 equities, 2005–2024, with no established portfolio gain | The limit is not only quantity: quality, economic coverage and the signal → allocation transformation dominate. |
| Which intervention is most robust on ETFs? | The **25% cap**: Sharpe 0.9525 vs 0.8650 uncapped | The management constraint acts as a powerful regulariser of estimation error. |
| Do the labels hide degraded models? | See "Model integrity" in **Published facts** below | Figure generated from `data/gold/fit_report_summary.json`, never typed by hand; the count is cross-checked against a second artifact (`dashboard_regime.parquet`) by `TestFallbackCountsAgree`. |

![Out-of-sample results, net of costs](docs/rapport_final/assets/figures/courbes_equity.png)

## Data and numéraires

| Universe | Composition | Window | Numéraire | Use |
|---|---|---|---|---|
| `full_2021` | 4 BVC equities + 5 ETFs | 2021-07-29 → 2026-07 | **MAD**, causal conversion at the official BAM rate | Main mixed universe; unhedged USD exposure. |
| `etf_2017` | SPY, QQQ, EEM, GLD, TLT | 2004-11 → 2026-07 | **USD** | Deep history covering 2008, 2020 and 2022. |
| `global_2004` | 10 multi-asset US ETFs | 2004-11 → 2026-08 | **USD** | Pre-registered research extension; outside the API and dashboard. |

The universes are evaluated separately. Their Sharpe levels are **not** directly comparable to each
other: currency, window, composition and asset count all differ. The unhedged USD/MAD exposure of
`full_2021` is a material economic risk — stated in the governance surfaces with the pinned
wording **`risque économique matériel`** — and it is embedded in realised performance, with no
hedging contract and no roll cost modelled.

BVC equities use total returns with dividends at ex-dates. ETFs are downloaded adjusted. The **deep
Morocco** experiment adds a research panel of 12 equities over 2005–2024, but it is not integrated
into the canonical release: source reconciliation, dividends, corporate actions, recent splicing
and redistribution rights all remain to be industrialised.

## Architecture

```mermaid
flowchart LR
    S[Sources\nBVC · Yahoo/FRED · BAM] --> B[Bronze\nraw and persistent]
    B --> V[Silver\ncalendars · MAD · contracts]
    V --> G[Gold\nreturns · features · evidence]
    G --> BT[Causal backtest\nconstraints + costs]
    BT --> M[Models\nMarkowitz · HMM · RF/XGB]
    M --> E[Evaluation\nwalk-forward · bootstrap · RC/SPA]
    E --> P[Publication\nAPI · dashboard · report]
    P --> GOV[Governance\nmanifest · cards · monitoring]
```

### Models evaluated

```text
Classical references
  └─ 1/N · MinVariance · Ledoit-Wolf · MaxSharpe
      └─ Dynamic covariance
          └─ EWMA · DCC-GARCH
              └─ HMM + conditional allocation
                  └─ RF/XGBoost + cost-aware optimisation
```

Complexity is added in tiers. A model that does not justify its cost out-of-sample stays an
exploratory challenger.

## Validation protocol

1. Causal features: no information later than the decision date.
2. Selection by expanding/rolling walk-forward, with purge and embargo.
3. Final period frozen for the final comparison.
4. **Paired** block bootstrap on return differences.
5. Correction across the 240 reachable configurations via White Reality Check and Hansen SPA.
6. Publication only from Gold artifacts sharing the same provenance.

The eight published paired comparisons all contain zero in their interval. Against
`regime_conditional`, the pre-specified primary comparator, White RC and Hansen SPA establish no
outperformance across both universes and both statistics. Results against equal-weight remain
exploratory and do not prove added value from the ML layer.

## Research extension — the `global_2004` universe

> ⚠️ **Research experiment, outside the delivered system.** `global_2004` is wired to neither the
> API, nor the dashboard, nor any production-bound allocation, and no result published above
> depends on it. This section describes an extension run *after* the system closed, to remove an
> identification weakness common to both published universes.

**The problem.** On `etf_2017`, minimum variance emits **a single distinct allocation across 248
rebalances**: with five assets under a 25% cap, it is empirically the constraint — not the model —
that picks the portfolio. On `full_2021`, daily covariance is biased by non-overlapping sessions
and stale prices. In both cases it was impossible to distinguish "the model adds nothing" from
"the setup does not let it express anything".

**The setup.** Ten USD-denominated US ETFs over 21.7 years, under a **strictly identical
constraint** (same 25% cap, same costs, same engine), specified in a timestamped pre-registration
*before* any ingestion, then validated against ten data-readiness criteria.

| | `etf_2017` | `global_2004` |
|---|---:|---:|
| distinct allocations (minimum variance) | **1 / 248** | **249 / 249** |

**The result.** Neither the regime layer (Q1) nor the 240-configuration RF/XGBoost family (Q2)
establishes an edge. The best raw candidate did beat the reference by **+0.093 Sharpe** — and
White's correction (*p* = 0.905) and Hansen's (*p* = 0.866) refuse that value, because it is the
maximum of a 240-configuration search.

The challengers actually selected without access to the final segment also show a lower observed
net Sharpe: **−0.0657** for RF and **−0.0898** for XGBoost relative to `regime_conditional`. That
is a descriptive operational diagnostic, **not** statistical proof of underperformance: no
individual paired test of those two differences had been pre-specified.

> Once both identification defects were removed, the complex models finally got a fair test — and
> still established no edge. The contribution is the auditable proof showing why the raw winner,
> seductive as it was, must not be believed.

📄 **[Detailed results — `GLOBAL_2004_RESULTS.md`](docs/GLOBAL_2004_RESULTS.md)**
(generated from artifacts) ·
[pre-registered protocol](docs/GLOBAL_UNIVERSE_PREREGISTRATION.md) ·
[executed synthesis notebook](notebooks/phase10_global_2004_evidence.ipynb) ·
chapter 8 of the final report · five slides in the defense deck.

⚠️ `etf_2017` and `global_2004` share five instruments and largely overlapping periods: these are
**two distinct but statistically overlapping evaluations**, not independent confirmations.

## Dashboard and API

![Dashboard — research results](docs/rapport_final/assets/figures/dashboard_page1.png)

- **Research results** — published facts, currencies, results, crises and limitations.
- **Strategy explorer** — metrics, trajectories, allocations and CSV export.
- **FastAPI service** — read-only over the same Gold artifacts, with no training at request time.
- Published contracts require `base_currency` and `hedge_status`; a missing currency is never
  silently replaced by a default.

```bash
# Requires the DVC artifact bundle
streamlit run dashboard/streamlit_app.py
uvicorn api.main:app --app-dir src
```

Interactive API documentation: `http://127.0.0.1:8000/docs`.

## Reproduce and verify

### Local install

**macOS / Linux**

```bash
git clone https://github.com/SouhailBourhim/Portfolio_ML.git
cd Portfolio_ML
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
pytest -q
```

**Windows**

```powershell
git clone https://github.com/SouhailBourhim/Portfolio_ML.git
cd Portfolio_ML
.\scripts\bootstrap_windows.ps1
```

The command above is not a convenience wrapper around the four lines opposite.
`requirements.lock.txt` was frozen on macOS and pins **uvloop**, which has no
Windows build at all, so a plain `pip install -r requirements.lock.txt` aborts
partway and leaves a half-populated environment. The lock is a SHA-256 input to
the release manifest and is therefore never edited to accommodate a platform;
the bootstrap script filters the four POSIX-only distributions at install time
instead. It also handles the `.venv\Scripts` layout and UTF-8 defaults.

Read **[docs/WINDOWS_SETUP.md](docs/WINDOWS_SETUP.md)** before the first run —
particularly §1.1, because Git's default `core.autocrlf` on Windows breaks
snapshot verification and `dvc status` in a way that looks like data corruption
rather than a line-ending setting.

The final synthesis notebook is already executed and re-runs neither ingestion, selection nor
backtest:

```bash
jupyter notebook notebooks/phase10_global_2004_evidence.ipynb
```

Its source is rebuilt by `scripts/build_visualization_notebooks.py`. Tests verify that it reads
only versioned Gold artifacts, that every cell has been executed, and that no network access or
training slips into this presentation layer.

The test suite is offline. Market data is not distributed in Git. The DVC remote is private
because of data licensing; authorised members configure their R2 credentials locally and then run:

```bash
./scripts/dvc.sh pull
./scripts/dvc.sh status
./.venv/bin/python src/snapshot.py verify
```

```powershell
# Windows equivalent
.\scripts\dvc.ps1 pull
.\scripts\dvc.ps1 status
python src/snapshot.py verify
```

Use the wrapper rather than `dvc` directly on either platform: DVC runs the commands in `dvc.yaml`
through a shell, where `python` would otherwise resolve to the system interpreter.

### Docker

```bash
docker compose run --rm test       # offline tests
docker compose up api              # API, if the DVC bundle is present
docker compose up dashboard        # Streamlit app on http://localhost:8501
docker compose up notebook         # local Jupyter
```

`api` and `dashboard` read the same Gold artifacts, mounted read-only: neither surface writes or
re-estimates. Both run the same `scripts/check_artifacts.py` preflight before starting and
**refuse to launch** on an incomplete bundle — serving a subset a reader might mistake for the
whole is worse than serving nothing.

The `api` service health probe requires `status == "ok"`. Its scope is deliberately narrow: since
the preflight already blocks startup, a container listening on 8000 has necessarily satisfied the
stricter guard. The probe therefore covers ordinary liveness and the case where the bundle changes
*underneath* an already-running container — the mount is read-only for the container, not for the
host.

The `test` service does **not** mount `data/`: the suite must pass on a fresh clone. Artifact
consistency checks are therefore skipped. To exercise those too:

```bash
docker compose run --rm -v "$PWD/data:/app/data:ro" test     # PowerShell: "${PWD}/data:/app/data:ro"
```

### Release gates

```bash
./scripts/release_gates.sh          # macOS / Linux
```

```powershell
.\scripts\release_gates.ps1         # Windows — same six gates, same order
```

These check DVC state, snapshot checksums, bundle completeness, model card regeneration, Git
cleanliness and the test suite. The `release-gates` CI job fetches the bundle from R2 with a
read-only token, on trusted events only.

## Repository layout

```text
src/                 pipeline, backtest, models, evaluation, API
dashboard/           two-view Streamlit application
data/                Bronze/Silver/Gold artifacts managed by DVC
experiments/         robustness experiments kept separate from the release
notebooks/           executed notebooks, rebuilt from final artifacts
tests/               891 unit, integration and governance tests
docs/                deliverables, model cards and documentation
docs/rapport_final/  MAINTAINED SOURCE of the submitted report
docs/rapport/        historical short version — archive, not maintained
output/pdf/          final PFA report (copy of docs/rapport_final/main.pdf)
output/presentation/ final defense presentation
scripts/             build, release and environment entry points (.sh + .ps1 twins)
dvc.yaml             reproducible production graph
params.yaml          data, model and validation parameters
compose.yaml         containerised API, pipeline, tests and notebooks
.gitattributes       LF line endings on every platform — a release-gate prerequisite
```

## Key documentation

- [Final PFA report](output/pdf/Rapport_PFA_Final_2026.pdf) (French)
- [Defense presentation](output/presentation/Soutenance_PFA_Portfolio_ML_INPT_EURAFRIC.pptx) (French)
- [Executed notebook — `global_2004` synthesis](notebooks/phase10_global_2004_evidence.ipynb)
- [Generated `global_2004` results](docs/GLOBAL_2004_RESULTS.md)
- [Pre-registered `global_2004` protocol](docs/GLOBAL_UNIVERSE_PREREGISTRATION.md)
- [Model governance](docs/MODEL_GOVERNANCE.md)
- [Model integrity](docs/MODEL_INTEGRITY.md)
- [Explainability](docs/EXPLAINABILITY.md)
- [Data governance](docs/DATA_GOVERNANCE.md)
- [Multiple testing](docs/MULTIPLE_TESTING.md)
- [Deep Morocco experiment](docs/DEEP_MOROCCO_EXPERIMENT.md)
- [ETF deep-history experiment](docs/ETF_DEEP_HISTORY_EXPERIMENT.md)
- [Inference contract](docs/INFERENCE_CONTRACT.md)
- [Running on Windows](docs/WINDOWS_SETUP.md)

## Limitations

- The canonical universe containing BVC remains short; the deep experiment improves the signal but
  is not yet a reconciled, contractual production source.
- One BVC dividend yield remains estimated and documented via sensitivity analysis.
- The multiple-testing correction covers the defined family of 240 configurations; the eight
  external comparisons constitute a separate level of multiplicity.
- On `global_2004`, Q2 simultaneously changes the asset cut and the macro-variable policy; its
  result cannot be attributed to universe widening alone. White RC and Hansen SPA correct the
  strategy search, not the external decision to build this third universe after diagnosing the
  first two.
- There is no order execution, no capacity/market-impact model, and no USD/MAD hedging strategy.
- Monitoring is instrumented **offline** but deliberately inactive until atomic publication,
  release locking and rollback have been exercised.
- Validation remains internal to the team; independent validation is required before any
  institutional use.

## Positioning

This repository is a **research prototype**, not an advisory, discretionary-management or execution
tool. No strategy is recommended. The project's value lies in the evidence chain: data, models,
decisions, fallbacks and published claims are all versioned and testable.

> **The block below is reproduced verbatim in French, and deliberately so.** It is generated by
> `scripts/build_release_facts.py` from `src/release_facts.py`, and the same sentences — word for
> word — appear in the report, the dashboard and the API.
> `tests/test_release_facts.py::TestSurfacesQuoteTheGeneratedStringsExactly` fails if any surface
> paraphrases them, so translating them here would break the guarantee they exist to provide.
> An English rendering is in [`docs/MODEL_GOVERNANCE.md`](docs/MODEL_GOVERNANCE.md).

<!-- BEGIN RELEASE FACTS — generated by scripts/build_release_facts.py -->

### Faits publiés — ce que ce dépôt établit, et ce qu'il n'établit pas

> Bloc généré depuis `src/release_facts.py`. Les mêmes phrases, mot pour mot,
> figurent dans le rapport, le tableau de bord et l'API ; un test échoue si
> une surface s'en écarte.

1. `full_2021` est libellé en MAD, converti au taux de référence officiel de Bank Al-Maghrib (USDMAD, MAD par USD). Le portefeuille reste **non couvert** : la variation de change réalisée est incluse dans la performance, aucun contrat à terme ni coût de roulement n'est modélisé.
2. `etf_2017` est libellé en USD : cet univers ne contient que des ETF mono-devise, n'a donc jamais présenté de défaut de numéraire, et est **inchangé** par la correction.
3. Les niveaux de Sharpe ne sont pas comparables d'un univers à l'autre : les deux univers sont libellés dans des devises différentes, sur des fenêtres différentes et avec un nombre d'actifs différent.
4. Sur `full_2021`, l'écart ponctuel entre `regime_conditional` et `max_sharpe` est de **-10,47 %** (0,9571 contre 1,0690 de Sharpe net). L'écart est également défavorable sur `etf_2017` : **-1,6 %**.
5. Ces écarts ne démontrent pas une supériorité de l'approche classique : aucun test pairé de cette différence n'est présenté.
6. `regime_conditional` demeure le comparateur primaire pré-spécifié des tests White/SPA. Ce choix a été fixé avant l'observation des résultats et n'est pas réécrit maintenant que le signe de l'écart a changé.
7. Correction pour tests multiples (White 2000, Hansen 2005) sur les 240 configurations atteignables : aucun candidat n'établit de surperformance face au comparateur primaire pré-spécifié.
8. Walk-forward imbriqué (fenêtre OOS 2023-07-28 → 2026-07-24, 781 lignes) : le classement est sensible au protocole d'évaluation et à la fenêtre hors échantillon associée. Ratio de Sharpe dégonflé (DSR) = 0,6707 sur 198 configurations.
9. Intégrité des modèles : 6 ajustements sur 1 184 ont emprunté un repli d'estimateur (4 stratégies évaluées, 296 dates de rééquilibrage). Sur ces rééquilibrages, le résultat a été produit par un estimateur de substitution et non par le modèle que son étiquette désigne : les séries concernées sont des HYBRIDES. La portée de cette mesure est exactement l'ensemble compté ci-dessus : elle n'affirme pas qu'aucun repli n'est possible sur un autre instantané.
10. Aucune stratégie n'est recommandée, déployée, ni présentée comme une valeur ajoutée établie. Ce livrable est un prototype de recherche.

<!-- END RELEASE FACTS -->

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- DeMiguel, V., Garlappi, L. & Uppal, R. (2009). *Optimal Versus Naive Diversification*.
- Jagannathan, R. & Ma, T. (2003). *Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps*.
- White, H. (2000). *A Reality Check for Data Snooping*.
- Hansen, P. R. (2005). *A Test for Superior Predictive Ability*.
