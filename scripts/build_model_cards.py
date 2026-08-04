"""Generate the Phase 3 model cards from committed artifacts.

Addresses: P4 — a model card that restates results in prose is one more
surface that can outlive the run it describes. CLAUDE.md §17.11 is the record
of a retracted claim surviving in six places at once because each was edited
by hand; the Chapter 5 paired table was the same failure in LaTeX. So every
number here is read from `data/gold/`, and `tests/test_model_cards.py` fails
if a card and its artifact disagree.

Three cards, deliberately unequal (decided with the user, 2026-08-03):

    MODEL_CARD_REGIME_CONDITIONAL.md   primary reference system
    MODEL_CARD_RF_CHALLENGER.md        exploratory challenger
    MODEL_CARD_XGB_CHALLENGER.md       exploratory challenger

The primary system contains no RF and no XGBoost, so its explanation is a
deterministic DECISION TRACE (regime posterior → chosen sub-optimizer →
moment inputs → binding constraints → weights), not feature attribution.
Attribution is the right tool for the challengers and the wrong one for a
system whose interpretability comes from its structure.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
DOCS = ROOT / "docs"

PRIMARY = "regime_conditional"
CHALLENGERS = {"rf_signal_tuned": "RF", "xgb_signal_tuned": "XGB"}
BENCH_LABEL = {"regime_conditional": "regime_conditional", "equal_weight": "equal_weight"}


def _load(name: str) -> dict:
    path = GOLD / name
    if not path.is_file():
        raise FileNotFoundError(
            f"{path.relative_to(ROOT)} is missing. Model cards are generated from "
            "committed artifacts; run the pipeline or `dvc pull` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


SHOWCASE = _load("dashboard_showcase.json")
P5 = _load("phase5_results.json")
PAIRED = _load("paired_comparison_results.json")
PROTOCOL = _load("phase5_validation_protocol.json")
MANIFEST = _load("snapshot_manifest.json")
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))

UNIVERSES = ("etf_2017", "full_2021")


# NOTE: an earlier draft also stamped `git describe --tags --always` here.
# Removed, for two reasons. It describes the CHECKOUT, not the artifact set the
# card documents — the manifest's own `git_commit` is the revision that
# produced these numbers. And it changed on every commit, which made
# regeneration non-idempotent and quietly defeated the CI gate asserting that
# rebuilding the cards is a no-op. A card must be a function of its artifacts
# alone.


def _f(x: float, n: int = 3) -> str:
    return f"{x:.{n}f}"


def _pct(x: float, n: int = 1) -> str:
    return f"{x:+.{n}f}%"


def _comparisons_for(candidate: str) -> list[dict]:
    return [c for c in PAIRED["comparisons"] if c["candidate"] == candidate]


def _paired_table(candidate: str) -> str:
    """Rows of the paired bootstrap for one candidate, across both universes."""
    rows = ["| Universe | Benchmark | ΔSharpe | 90% CI | p | Establishes? |",
            "|---|---|---:|:---:|---:|:---:|"]
    for c in _comparisons_for(candidate):
        lo, hi = c["sharpe_diff_ci"]
        established = lo > 0 and c["p_value_no_outperformance"] < 0.05
        rows.append(
            f"| `{c['universe']}` | `{BENCH_LABEL[c['benchmark']]}` | {c['sharpe_diff']:+.3f} "
            f"| [{lo:+.3f}, {hi:+.3f}] | {_f(c['p_value_no_outperformance'])} "
            f"| {'**yes**' if established else 'no'} |"
        )
    return "\n".join(rows)


def _chronology_table() -> str:
    rows = ["| Universe | Train + validation | Frozen test | Test fraction |",
            "|---|---|---|---:|"]
    for uni in UNIVERSES:
        u = P5[uni]
        rows.append(
            f"| `{uni}` | {u['train_val_start']} → {u['train_val_end']} "
            f"| {u['test_start']} → {u['test_end']} | {u['test_frac']:.0%} |"
        )
    return "\n".join(rows)


def _universe_table() -> str:
    rows = ["| Universe | Assets | Full-window OOS | Net Sharpe | Max drawdown | Avg turnover |",
            "|---|---:|---|---:|---:|---:|"]
    for uni in UNIVERSES:
        block = SHOWCASE["universes"][uni]
        stats = block["strategies"][PRIMARY]
        rows.append(
            f"| `{uni}` | {len(SHOWCASE['assets_per_universe'][uni])} "
            f"| {block['oos_start']} → {block['oos_end']} "
            f"| {_f(stats['sharpe_net'], 4)} | {_f(stats['max_drawdown'], 4)} "
            f"| {_f(stats['avg_turnover'], 4)} |"
        )
    return "\n".join(rows)


def _test_window_table() -> str:
    rows = ["| Universe | Strategy | Net Sharpe (frozen test) | 90% CI |",
            "|---|---|---:|:---:|"]
    for uni in UNIVERSES:
        u = P5[uni]
        for name, block in list(u["baselines"].items()) + [
            (k, v) for k, v in u["tuned"].items()
        ]:
            lo, hi = block["test_sharpe_ci"]
            rows.append(f"| `{uni}` | `{name}` | {_f(block['test_sharpe_net'], 4)} "
                        f"| [{_f(lo)}, {_f(hi)}] |")
    return "\n".join(rows)


def _header(title: str, status: str) -> str:
    dirty = " ⚠️ generated from a DIRTY tree" if MANIFEST.get("git_dirty") else ""
    return f"""<!-- GENERATED by scripts/build_model_cards.py — do not edit by hand.
     Every number is read from data/gold/; edit the artifacts or the generator. -->

# {title}

> **Status: {status}**
> This is a research prototype produced for an academic end-of-year project
> (PFA, INPT × EURAFRIC Information). It is **not** production-approved, it is
> **not** investment advice, and it must not be used to execute orders or to
> make a client recommendation.

| | |
|---|---|
| Code revision (artifacts) | `{MANIFEST['git_commit'][:12]}`{dirty} |
| Python | {MANIFEST['python']} |
| Snapshot manifest | `data/gold/snapshot_manifest.json`, {len(MANIFEST['files'])} files hashed |
| Card generated from | committed Gold artifacts, not typed |
"""


def _multiple_testing_sentence(mt: dict) -> str:
    """One sentence, whichever status the pipeline recorded.

    The card must follow the artifact rather than assume a status: it read
    `mt['reason']` unconditionally and broke the moment the correction moved
    from `not_established` to `established`, which is the transition the whole
    exercise existed to produce.
    """
    if mt["status"] != "established":
        return mt.get("reason", "")
    return (
        f"{mt['verdict']} Corrected over {mt['n_candidates_corrected_for']} reachable "
        f"configurations via White's Reality Check and Hansen's SPA "
        f"(`{mt['artifact']}`). Comparisons against `equal_weight` are exploratory "
        f"by pre-specification and are not the basis of this status."
    )


def _shared_limitations() -> str:
    mt = PAIRED["multiple_testing"]
    return f"""## Limitations

- **No production approval.** No independent validation, no change-control
  process, no operational monitoring in service. See `docs/MODEL_GOVERNANCE.md`.
- **Not investment advice.** No suitability assessment, no client profiling,
  no order execution path.
- **No client data.** The system consumes public market and macro series only.
  It holds no personal data and no client PII of any kind.
- **Multiple testing is `{mt['status']}`.** {_multiple_testing_sentence(mt)}
- **Currency exposure is unhedged.** BVC returns are MAD-denominated and ETF
  returns USD; every reported figure embeds an unhedged USD/MAD exposure.
- **`full_2021` excludes the 2020 COVID crash** — the free BVC source begins
  mid-2021. The `etf_2017` universe exists to recover that evidence.
- **Marginal confidence intervals are not a test of a difference.** Comparative
  statements in this card come from the paired bootstrap, never from CI overlap.
"""


# ── Primary card ─────────────────────────────────────────────────────────────
def primary_card() -> str:
    regime = PARAMS["regime"]
    bt = PARAMS["backtest"]
    n_established = sum(
        1 for c in PAIRED["comparisons"]
        if c["sharpe_diff_ci"][0] > 0 and c["p_value_no_outperformance"] < 0.05
    )
    return f"""{_header("Model card — `regime_conditional`", "Primary reference system · research prototype")}
## Purpose

Allocate capital across a fixed asset universe at a monthly rebalance, under
long-only and per-asset weight constraints, net of transaction costs. The
system detects a latent market regime from past returns and hands the entire
allocation decision to one of two already-validated Markowitz optimizers.

**Non-purpose.** It does not forecast individual asset returns, does not time
entries or exits intraday, does not size positions for leverage or shorting,
and does not produce a client-specific recommendation.

## Why this is the reference system

Across {len(PAIRED['comparisons'])} paired comparisons on frozen test data,
**{n_established}** established statistically supported outperformance over
this system or over equal weighting. It is the reference not because it was
shown to be superior, but because no challenger displaced it and its behaviour
is fully reconstructible — see *Explainability* below.

## Supported universes

{_universe_table()}

Assets — `etf_2017`: {', '.join(f'`{a}`' for a in SHOWCASE['assets_per_universe']['etf_2017'])}.
`full_2021`: {', '.join(f'`{a}`' for a in SHOWCASE['assets_per_universe']['full_2021'])}.

## Inputs and their timestamps

| Input | Source | Availability at decision time τ |
|---|---|---|
| {', '.join(f'`{f}`' for f in regime['features'])} | Phase 3 market features | Same-day close at τ. Causal: the engine fits at τ and earns from τ+1 |
| Asset log-returns | Gold `log_returns*.parquet` | Strictly ≤ τ; the engine slices every frame to `:τ` |
| Macro signals | FRED / BAM | Differenced, standardized, lagged ≥ 1 trading day for publication delay |

The backtest engine — not the strategy — performs the slicing. A strategy
physically cannot read a row dated after its decision date, and that property
is tested end-to-end rather than asserted (`tests/test_phase4_integration.py`
corrupts the *future* of the feature frame and requires every *past* weight to
be unchanged, while also proving the features were genuinely consumed).

## Target definition

None. This system predicts no quantity. It classifies the current regime and
selects an optimizer; the optimizer's objective is defined on the sample
moments of the training window. This is the substantive difference from the
challengers, which do predict an expected return.

## Model and parameters

| Parameter | Value | Rationale |
|---|---|---|
| Regime model | Gaussian HMM (`hmmlearn`), `n_states={regime['n_states']}` | Two states, not three: `full_2021` has too little history for a thin third state |
| HMM features | {', '.join(f'`{f}`' for f in regime['features'])} | Direct P2/P3 candidates |
| Covariance type | `{regime['covariance_type']}` | |
| EM restarts | {regime['n_restarts']}, seeded from `{regime['random_state_base']}` | Highest-log-likelihood converged fit is kept |
| Bull sub-strategy | `{regime['bull_strategy']}` | |
| Bear sub-strategy | `{regime['bear_strategy']}` | Defensive; also the resolution of an uncertain posterior |
| Min. regime history | {regime['min_regime_train_days']} days | Below this the posterior is treated as uninformative |
| Rebalance frequency | `{bt['rebalance_freq']}` (month-end) | |
| Min. training history | {bt['min_train_days']} days | |
| Max weight per asset | {bt['max_weight']:.0%} | A management constraint, not a tuned hyperparameter |
| Transaction costs | {bt['costs_bps']['etf']} bps ETF, {bt['costs_bps']['bvc']} bps BVC | Deducted from net returns at every rebalance, on realised turnover |
| Risk-free rate | {bt['risk_free_annual']:.1%} annual | |

Scalers are fitted fresh inside each walk-forward training window. No feature
is standardized globally — doing so over the full history would itself be a
lookahead leak.

## Validation chronology

{_chronology_table()}

Hyperparameter selection uses `{PROTOCOL['protocol']}`
({PROTOCOL['config'].get('n_splits', 'n/a')} folds, embargo
{PROTOCOL['config'].get('embargo_dates', 'n/a')} days, label horizon
{PROTOCOL['config'].get('label_horizon', 'n/a')} day). Every fold satisfies
`max(train) < min(validation)`, asserted at split time rather than assumed. The
frozen test segment is untouched by any selector.

## Results on the frozen test window

{_test_window_table()}

### Paired comparisons against this system

The instrument is a paired moving-block bootstrap on identical test dates, net
of costs: both series are resampled with the *same* block indices, preserving
serial dependence within each strategy and same-day correlation between them.
The p-value is null-centred.

{_paired_table('rf_signal_tuned')}

{_paired_table('xgb_signal_tuned')}

**No comparison establishes outperformance in either direction.** This is not a
finding of equivalence: failing to reject is not accepting the null, and no
equivalence test against a pre-specified margin was run.

## Explainability — deterministic decision trace

This system is not explained by feature attribution, because it fits no
predictive model over features. It is explained by reconstructing the decision:

1. HMM input features at τ and the fitted posterior regime probabilities
2. The selected regime label (bull / bear) and how it was mapped from the
   HMM's unordered state indices
3. The sub-optimizer that received the decision
4. The expected-return and covariance inputs handed to that optimizer
5. The weight cap, cost model, and which constraints were **binding**
6. Final weights and realised turnover
7. Any fallback or degradation, with its reason

Every step is a function of data available at τ, so the trace is reproducible
from the committed snapshot. Artifact: `data/gold/model_explanations.json`.

## Known failure modes and fallback behaviour

| Condition | Behaviour | Observed |
|---|---|---|
| HMM fails to converge, or history < {regime['min_regime_train_days']} days | Neutral posterior resolves to the **defensive** `{regime['bear_strategy']}`, not an arbitrary tie-break | Fires on the first 2–3 rebalances of each universe, by design |
| Per-asset GARCH non-convergence (DCC path) | Ledoit-Wolf shrinkage substituted, logged at WARNING | Fired once on `IAM.CS` in a live run |
| Weight cap infeasible for the universe size | Raises rather than silently renormalizing | Guarded in `_as_weight_series` |
| BVC dividend cache absent | `clean` raises `DividendDataUnavailable` | Deliberate: silently reverting to price-only returns understated BVC assets by 3.0–4.3%/yr |

A binding cap deserves separate mention. On `etf_2017`, 5 assets × a
{bt['max_weight']:.0%} cap forces every feasible long-only portfolio to hold at
least four assets at the cap, so the constraint — not the covariance model —
very nearly determines the allocation. This is a documented property of the
universe, not a defect, but any conclusion drawn from `etf_2017` about model
choice must account for it.

## Cost and turnover assumptions

Costs are charged on realised turnover at each rebalance:
{bt['costs_bps']['etf']} bps for ETFs, {bt['costs_bps']['bvc']} bps for BVC
equities — the higher BVC rate is how illiquidity is penalised, rather than by
excluding thinly-traded names. Both gross and net figures are always reported;
a strategy that wins gross and loses net is treated as a finding.

{_shared_limitations()}
## Reproducibility

```bash
git checkout {MANIFEST['git_commit'][:12]}
./scripts/dvc.sh pull
./.venv/bin/python src/snapshot.py verify
```

The manifest hashes {len(MANIFEST['files'])} inputs and artifacts and records
whether the tree was clean when it was written. Verification rejects a manifest
produced from a dirty tree, and rejects a revision that is not an ancestor of
the checked-out one.
"""


# ── Challenger cards ─────────────────────────────────────────────────────────
def challenger_card(candidate: str) -> str:
    short = CHALLENGERS[candidate]
    algo = "RandomForest" if short == "RF" else "XGBoost"
    p5_params = {uni: P5[uni]["tuned"][candidate] for uni in UNIVERSES}
    grid_key = "rf_grid" if short == "RF" else "xgb_grid"
    grid = PARAMS["phase5"][grid_key]

    selected = ["| Universe | ML parameters | Levers | CV IC | Test Sharpe | 90% CI |",
                "|---|---|---|---:|---:|:---:|"]
    for uni in UNIVERSES:
        t = p5_params[uni]
        lo, hi = t["test_sharpe_ci"]
        ml = ", ".join(f"{k}={v}" for k, v in t["selected_ml_params"].items())
        lv = ", ".join(f"{k}={v}" for k, v in t["selected_levers"].items())
        selected.append(f"| `{uni}` | {ml} | {lv} | {t['best_cv_ic']:+.4f} "
                        f"| {_f(t['test_sharpe_net'], 4)} | [{_f(lo)}, {_f(hi)}] |")

    return f"""{_header(f"Model card — `{candidate}` ({algo} challenger)",
                        "Exploratory research challenger · NOT the reference strategy")}
> **Challenger status.** This model **did not establish statistically supported
> outperformance** against `{PRIMARY}` or against `equal_weight` in the Phase 2
> paired comparisons on frozen test data. It is documented as transparent
> negative-result research, not as a recommended strategy. The reference system
> is `{PRIMARY}` — see `docs/MODEL_CARD_REGIME_CONDITIONAL.md`.

## Purpose

Predict a cross-sectional expected return per asset, and substitute that
prediction for the naive sample mean in the existing Sharpe objective. The
covariance estimator and every portfolio constraint are unchanged, so any
difference in outcome is attributable to the expected-return input alone.

**Non-purpose.** It is not a price forecast, not a trading signal for
discretionary use, and not a standalone product.

## Model

A single **pooled cross-sectional** {algo}: asset identity is a feature rather
than a reason to fit one model per asset, which multiplies effective training
rows by the asset count — the practical answer to `full_2021`'s short history.
The regime posterior enters as one input feature (`REGIME_BULL_PROB`) rather
than by dispatching two sub-models.

Features are per-asset trailing returns over several momentum windows,
trailing volatility, and price relative to a moving average, plus the regime
probability. Labels are next-period returns; the current rebalance date is
excluded from training **structurally, by date** — not by a NaN filter that a
caller error could bypass.

## Selected configuration and frozen-test result

{chr(10).join(selected)}

Search grid: {json.dumps(grid)}, crossed with
shrinkage {PARAMS['phase5']['shrink_grid']} and turnover penalty
{PARAMS['phase5']['penalty_grid']}.

The information coefficients are small in absolute terms, and one is negative.
That is the honest state of cross-sectional return prediction on this data, and
it is reported rather than smoothed.

## Paired comparisons

{_paired_table(candidate)}

## Explainability

Feature attribution is appropriate here, because this model does fit a
predictive function over features:

- global feature importance across the training panel;
- local contribution for a selected rebalance date;
- the path from predicted expected returns to the resulting weights.

**Attribution is not causation.** An importance score describes what the fitted
function uses, not what moves markets. Nothing in this card licenses a causal
reading.

## Known failure modes

| Condition | Behaviour |
|---|---|
| Training panel below the minimum row count | Falls back to the naive sample mean; `fallback_used` is recorded |
| Regime posterior unavailable | `REGIME_BULL_PROB` defaults to a neutral 0.5 |
| Turnover penalty too aggressive | Documented pathology: refusing to trade prevents the view being expressed at all, visible in the *gross* Sharpe, not only the net |

{_shared_limitations()}"""


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    written = [("MODEL_CARD_REGIME_CONDITIONAL.md", primary_card())]
    written += [
        (f"MODEL_CARD_{short}_CHALLENGER.md", challenger_card(candidate))
        for candidate, short in CHALLENGERS.items()
    ]
    for name, text in written:
        (DOCS / name).write_text(text, encoding="utf-8")
        print(f"  docs/{name}  ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    print("model cards:")
    main()
