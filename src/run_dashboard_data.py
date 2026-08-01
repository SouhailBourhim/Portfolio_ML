"""
run_dashboard_data.py — Persist headline artifacts consumed by the Streamlit dashboard.

Addresses: P4 — reporting integrity, which is backtest overfitting's last mile.
A stakeholder-facing surface quoting a number that has drifted from what the
pipeline currently produces is indistinguishable, to a reader, from a number
that was overfitted; both are unreproducible. This runner is the single source
of what the dashboard shows, regenerated deterministically from committed Gold
via a DVC stage (§16's "verify claims against current data" rule made
mechanical). This runner is the single source of truth for what the dashboard
sees, and its output is regenerated deterministically from committed Gold
inputs via a DVC stage — so the "did I remember to refresh?" failure class
(that hit `phase2_hurdle.json` once already, §17.1) cannot happen here.

Runs the FOUR headline strategies:

  1. equal_weight       — the honest baseline (1/N, DeMiguel 2009)
  2. min_variance_lw    — Markowitz with Ledoit-Wolf shrinkage (Phase 2 baseline)
  3. max_sharpe         — Markowitz max-Sharpe (Phase 2 winner on etf_2017)
  4. regime_conditional — the ML system that BEAT Markowitz on full_2021
                          (Phase 4 winner, the headline of the pitch page)

on BOTH universes (etf_2017, full_2021), via the SAME `run_backtest` engine
every prior phase used — zero new modelling. F7 strategies (rf_signal /
xgb_signal) are deliberately NOT rerun here: Phase 5 proved they don't add
statistically-significant value; the pitch page never mentions them, and the
tool page (M2) will consume their Phase 5-committed numbers when it needs
to compare, not rerun them here.

Outputs (all under `data/gold/`, DVC-tracked):

  * `dashboard_equity.parquet`      — long-form (Date, universe, strategy, gross, net)
  * `dashboard_weights.parquet`     — long-form (Date, universe, strategy, asset, weight)
  * `dashboard_regime.parquet`      — long-form (Date, universe='full_2021',
                                       strategy='regime_conditional', bull_prob, bear_prob)
  * `dashboard_showcase.json`       — metadata + per-(universe, strategy) metrics
                                       table + headline comparison + Phase 5 CIs

The corresponding test (`tests/test_run_dashboard_data.py`) asserts that
every number this runner emits matches its committed Gold source exactly —
if a Phase 4 rerun ever changes `regime_conditional`'s Sharpe, this runner's
output changes with it, and the dashboard automatically stays honest.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from backtest import build_cost_vector, run_backtest
from metrics import (
    annualized_return,
    annualized_sharpe,
    calmar_ratio,
    max_drawdown,
)
from run_phase4 import load_features, load_universe
from strategies import (
    EqualWeight,
    MaxSharpe,
    MinVarianceLW,
    RegimeConditionalStrategy,
)
from utils import load_params

log = logging.getLogger("run_dashboard_data")

# What the dashboard reads. Kept flat and boring by design.
HEADLINE_STRATEGIES = ("equal_weight", "min_variance_lw", "max_sharpe", "regime_conditional")


def _build_headline_strategies(params: dict) -> dict[str, object]:
    """Fresh instantiations only — RegimeConditionalStrategy holds per-instance state
    (regime_log) that must never mix universes; a shared instance across two runs would
    leak the first universe's decisions into the second's log."""
    bt = params["backtest"]
    rf = bt["risk_free_annual"]
    max_weight = bt["max_weight"]
    regime_params = params["regime"]
    return {
        "equal_weight": EqualWeight(),
        "min_variance_lw": MinVarianceLW(max_weight=max_weight),
        "max_sharpe": MaxSharpe(max_weight=max_weight, risk_free_annual=rf),
        "regime_conditional": RegimeConditionalStrategy(
            bull_strategy=MaxSharpe(max_weight=max_weight, risk_free_annual=rf),
            bear_strategy=MinVarianceLW(max_weight=max_weight),
            n_states=regime_params["n_states"],
            n_restarts=regime_params["n_restarts"],
            random_state_base=regime_params["random_state_base"],
            covariance_type=regime_params["covariance_type"],
            min_regime_train_days=regime_params["min_regime_train_days"],
            features=regime_params.get("features"),
        ),
    }


def _metrics_for(net_returns: pd.Series, gross_returns: pd.Series,
                 turnover: pd.Series, rf: float) -> dict:
    return {
        "sharpe_net": round(float(annualized_sharpe(net_returns, rf)), 4),
        "sharpe_gross": round(float(annualized_sharpe(gross_returns, rf)), 4),
        "annualized_return_net": round(float(annualized_return(net_returns)), 4),
        "max_drawdown": round(float(max_drawdown(net_returns)), 4),
        "calmar": round(float(calmar_ratio(net_returns)), 4),
        "avg_turnover": round(float(turnover.mean()), 4),
    }


def run_universe(universe: str, params: dict) -> tuple[pd.DataFrame, pd.DataFrame,
                                                       pd.DataFrame | None, dict]:
    """Run all four headline strategies on one universe. Returns (equity_long,
    weights_long, regime_long_or_None, metrics_per_strategy)."""
    bt = params["backtest"]
    returns_path = bt["universes"][universe]
    features_path = params["ml_features"]["outputs"][universe]

    returns = load_universe(returns_path)
    features = load_features(features_path)

    cost_vector = build_cost_vector(
        list(returns.columns),
        etf_cost_bps=bt["costs_bps"]["etf"],
        bvc_cost_bps=bt["costs_bps"]["bvc"],
    )

    strategies = _build_headline_strategies(params)
    equity_rows, weights_rows, regime_rows = [], [], None
    metrics = {}

    for name, strat in strategies.items():
        t0 = time.time()
        extras = {"features": features} if name == "regime_conditional" else None
        result = run_backtest(
            returns, strat,
            rebalance_freq=bt["rebalance_freq"],
            min_train_days=bt["min_train_days"],
            cost_bps=cost_vector,
            extras=extras,
            universe_name=universe,
            max_weight=bt["max_weight"],
        )
        elapsed = time.time() - t0
        log.info("  %s / %s ✓ (%.1fs)  net Sharpe %.3f",
                 universe, name, elapsed,
                 float(annualized_sharpe(result.net_returns, bt["risk_free_annual"])))

        for date, gross, net in zip(result.gross_returns.index,
                                    result.gross_returns.values,
                                    result.net_returns.values):
            equity_rows.append({"Date": date, "universe": universe, "strategy": name,
                                "gross_return": float(gross), "net_return": float(net)})
        # target_weights is a DataFrame indexed by rebalance date, cols=assets
        for date, row in result.target_weights.iterrows():
            for asset, weight in row.items():
                weights_rows.append({"Date": date, "universe": universe, "strategy": name,
                                     "asset": asset, "weight": float(weight)})
        metrics[name] = _metrics_for(result.net_returns, result.gross_returns,
                                     result.turnover, bt["risk_free_annual"])

        # HMM regime timeline — only captured for regime_conditional. Each
        # entry has {date, regime, posterior: {'bull': p, 'bear': 1-p}, converged}.
        if name == "regime_conditional" and strat.regime_log:
            regime_rows_this = []
            for entry in strat.regime_log:
                posterior = entry.get("posterior") or {}
                regime_rows_this.append({
                    "Date": entry.get("date"),
                    "universe": universe,
                    "strategy": name,
                    "bull_prob": float(posterior.get("bull", 0.5)),
                    "bear_prob": float(posterior.get("bear", 0.5)),
                    "regime": entry.get("regime", "unknown"),
                    "converged": bool(entry.get("converged", False)),
                })
            if regime_rows is None:
                regime_rows = regime_rows_this
            else:
                regime_rows.extend(regime_rows_this)

    equity = pd.DataFrame(equity_rows)
    weights = pd.DataFrame(weights_rows)
    regime = pd.DataFrame(regime_rows) if regime_rows else None
    return equity, weights, regime, metrics


class StalePhase5Results(RuntimeError):
    """Raised when `phase5_results.json` predates the Gold data it must describe."""


# Filesystem mtime granularity, and the fact that a single `dvc repro` writes
# several artifacts within the same second, make an exact ordering test
# flaky. One second of slack catches the failure that actually matters (a
# results file DAYS older than its inputs) without failing on same-run writes.
_STALENESS_TOLERANCE_SECONDS = 1.0


def _assert_phase5_describes_current_gold(
    phase5_path: Path,
    gold_inputs: list[Path],
    tolerance_seconds: float = _STALENESS_TOLERANCE_SECONDS,
) -> None:
    """
    Refuse to publish if the Phase 5 CIs are older than the Gold data.

    Addresses: P4 — this runner COPIES Phase 5's held-out CIs verbatim into
    `dashboard_showcase.json`, where Page 1 renders them directly beneath the
    headline Sharpes this run just computed. The two must therefore describe
    the SAME data, or the page presents a validation of numbers that no longer
    exist while asserting the opposite ("le chiffre ci-dessus a été revalidé").

    This is not hypothetical. On 2026-07-25 the BVC dividend correction
    (commit 7ec3626) and the deep ETF window regenerated every Gold return
    series, but `phase5_results.json` was left at its 2026-07-22 run — so the
    dashboard showed corrected point estimates under pre-correction intervals
    for three days, undetected. The existing verbatim-copy test could not see
    it: it verifies copy FIDELITY against a fixture, which is a different
    property from FRESHNESS.

    mtime is a deliberately coarse proxy — it cannot prove the two runs used
    identical inputs, only catch the ordering violation that actually occurs
    in this workflow (regenerate Gold, forget the ~45-minute Phase 5 rerun).
    A content hash would be stronger and is worth doing if the artifacts ever
    grow a provenance header; ordering is what this failure class looks like
    today.

    Raises:
        StalePhase5Results: if `phase5_path` is older than any Gold input by
            more than `tolerance_seconds`, naming the offending file and the
            command that fixes it — a silent stale CI is exactly the failure
            this project treats as a bug (§15.13).
    """
    if not phase5_path.exists():
        raise StalePhase5Results(
            f"{phase5_path} is missing — run `dvc repro phase5_compare` before "
            f"regenerating the dashboard; Page 1 cannot show held-out intervals "
            f"without it."
        )

    phase5_mtime = phase5_path.stat().st_mtime
    stale_against = [
        (p, p.stat().st_mtime)
        for p in gold_inputs
        if p.exists() and p.stat().st_mtime - phase5_mtime > tolerance_seconds
    ]
    if stale_against:
        newest, newest_mtime = max(stale_against, key=lambda pair: pair[1])
        lag_hours = (newest_mtime - phase5_mtime) / 3600.0
        raise StalePhase5Results(
            f"{phase5_path.name} is {lag_hours:.1f}h OLDER than {newest.name}. "
            f"Its confidence intervals were computed on superseded data, and the "
            f"dashboard renders them directly beneath this run's Sharpe ratios — "
            f"publishing would present a validation of numbers that no longer "
            f"exist. Run `dvc repro phase5_compare` (~45 min), then rerun this."
        )


def build_showcase(all_metrics: dict, params: dict, equity: pd.DataFrame,
                   weights: pd.DataFrame) -> dict:
    """Assemble the compact JSON summary the dashboard's Page 1 headlines from.

    The single most important field is `headline` — for each universe, the best
    Markowitz baseline (net Sharpe from the classical strategies) and the best
    ML strategy (regime_conditional here), with the difference expressed as a
    percentage. This IS the "our system beats Markowitz by X%" claim, computed
    from the numbers this run just produced — no hardcoded numbers anywhere.
    """
    phase5_path = ROOT / params["phase5"]["results_path"]
    _assert_phase5_describes_current_gold(
        phase5_path,
        [
            ROOT / "data" / "gold" / "log_returns.parquet",
            ROOT / "data" / "gold" / "log_returns_etf.parquet",
        ],
    )
    p5 = json.loads(phase5_path.read_text())

    showcase = {
        "universes": {},
        "generated_from_committed_gold": True,
        "rebalance_freq": params["backtest"]["rebalance_freq"],
        "max_weight": params["backtest"]["max_weight"],
        "cost_bps": params["backtest"]["costs_bps"],
    }
    for universe, metrics in all_metrics.items():
        classical = ("equal_weight", "min_variance_lw", "max_sharpe")
        ml = ("regime_conditional",)
        best_classical_name = max(classical, key=lambda s: metrics[s]["sharpe_net"])
        best_ml_name = max(ml, key=lambda s: metrics[s]["sharpe_net"])
        base = metrics[best_classical_name]["sharpe_net"]
        top = metrics[best_ml_name]["sharpe_net"]
        pct = (top - base) / abs(base) * 100 if base != 0 else float("nan")

        # Phase 5 test-window CIs (frozen held-out, block-bootstrap 90%)
        p5_u = p5.get(universe, {})
        p5_baselines = p5_u.get("baselines", {})
        test_ci = {
            "regime_conditional": p5_baselines.get("regime_conditional", {}),
            "equal_weight": p5_baselines.get("equal_weight", {}),
            "test_start": p5_u.get("test_start"),
            "test_end": p5_u.get("test_end"),
        }

        universe_rows = equity[equity["universe"] == universe]
        oos_start = universe_rows["Date"].min()
        oos_end = universe_rows["Date"].max()

        showcase["universes"][universe] = {
            "strategies": metrics,
            "best_classical": {"name": best_classical_name, "sharpe_net": base},
            "best_ml": {"name": best_ml_name, "sharpe_net": top},
            "headline_lift_pct": round(pct, 2),
            "headline_lift_absolute_sharpe": round(top - base, 4),
            "oos_start": str(oos_start.date()) if oos_start is not None else None,
            "oos_end": str(oos_end.date()) if oos_end is not None else None,
            "phase5_test_window": test_ci,
        }

    # Number of assets per universe — useful for the pitch narrative.
    assets_per_universe = {
        u: sorted(weights[weights["universe"] == u]["asset"].unique().tolist())
        for u in all_metrics
    }
    showcase["assets_per_universe"] = assets_per_universe
    return showcase


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    params = load_params()

    equities, weightss, regimes, all_metrics = [], [], [], {}
    for universe in ("etf_2017", "full_2021"):
        log.info("=== running headline strategies on %s ===", universe)
        eq, w, rg, metrics = run_universe(universe, params)
        equities.append(eq)
        weightss.append(w)
        if rg is not None:
            regimes.append(rg)
        all_metrics[universe] = metrics

    equity = pd.concat(equities, ignore_index=True)
    weights = pd.concat(weightss, ignore_index=True)
    regime = pd.concat(regimes, ignore_index=True) if regimes else None
    showcase = build_showcase(all_metrics, params, equity, weights)

    out_dir = ROOT / "data" / "gold"
    out_dir.mkdir(parents=True, exist_ok=True)
    equity.to_parquet(out_dir / "dashboard_equity.parquet", index=False)
    weights.to_parquet(out_dir / "dashboard_weights.parquet", index=False)
    if regime is not None and not regime.empty:
        regime.to_parquet(out_dir / "dashboard_regime.parquet", index=False)
    (out_dir / "dashboard_showcase.json").write_text(json.dumps(showcase, indent=2, default=str))

    # Report — proves the numbers on every page derive from this one run.
    log.info("=" * 70)
    log.info("HEADLINES (dashboard will show):")
    for universe, u in showcase["universes"].items():
        log.info(
            "  %-10s  best classical: %-18s Sharpe %.3f    "
            "best ML: %-20s Sharpe %.3f    lift: %+.1f%%",
            universe, u["best_classical"]["name"], u["best_classical"]["sharpe_net"],
            u["best_ml"]["name"], u["best_ml"]["sharpe_net"], u["headline_lift_pct"],
        )
    log.info("=" * 70)
    log.info("wrote %d files under %s", 3 + (1 if regime is not None else 0), out_dir)


if __name__ == "__main__":
    main()
