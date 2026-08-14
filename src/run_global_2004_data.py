"""
run_global_2004_data.py — Checkpoint 1 for the `global_2004` experiment.

Addresses: P4 — establishes that the universe is SOUND before any performance
is computed on it. A negative result from an unsound universe is
unattributable: you cannot tell whether the models failed or the data did.
`etf_2017` is the cautionary case — its "no ML benefit" conclusion stood for
weeks before anyone measured that the constraint left the optimizer nowhere to
express a view.

WHAT THIS SCRIPT DOES AND DOES NOT DO. It builds Bronze -> Silver -> Gold for
the frozen instrument set and writes a data-readiness artifact. It computes NO
Sharpe ratio, NO portfolio return, NO drawdown and NO turnover. The
allocation-freedom gate is derived from WEIGHTS ONLY — whether the optimizer
varies its allocation, never whether that allocation earned anything.

That separation is deliberate and is the reason this is its own checkpoint: if
the readiness gates fail, the right move is to amend the protocol in the open,
not to discover the problem afterwards in a Sharpe table and rationalise it.

Usage:
    python src/run_global_2004_data.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from global_universe import (  # noqa: E402
    build_global_gold,
    build_global_silver,
    ingest_global_prices,
    load_global_config,
    load_retained_macro,
    measure_allocation_freedom,
    measure_synchrony,
    verify_no_lookahead,
)
from provenance import _sha256, git_revision  # noqa: E402
from utils import load_params  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("run_global_2004_data")


def _provenance(returns: pd.DataFrame, sources: list[str]) -> dict:
    """Provenance for the readiness artifact, with a SELF-DECLARED numéraire.

    `provenance.build_provenance` is not reused here because it resolves the
    base currency through `read_numeraire`, which raises for any universe
    absent from `currency_manifest.json`. Adding `global_2004` to that manifest
    means editing `src/features.py`, a dependency of the `features` stage, which
    would cascade a rebuild through the entire released Gold layer — an absurd
    price for a checkpoint-1 diagnostic.

    Declaring USD here is not a shortcut around the manifest's purpose: every
    instrument in this universe is US-listed and USD-denominated BY ELIGIBILITY
    RULE E1, so unlike `full_2021` there is no FX conversion to get wrong and
    no mixed-numéraire hazard to detect. Registering the universe in the
    canonical manifest is the right move when and if it is adopted, and is
    recorded as a follow-up rather than done here.
    """
    return {
        "universe": "global_2004",
        "base_currency": "USD",
        "base_currency_source": (
            "SELF-DECLARED from eligibility rule E1 (US-listed, USD-denominated). "
            "Not yet registered in data/gold/currency_manifest.json — see the "
            "docstring of _provenance() for why, and the follow-up note."
        ),
        "hedge_status": "unhedged",
        "currency_converted": False,
        "data_range": {
            "start": str(returns.index.min().date()),
            "end": str(returns.index.max().date()),
            "n_rows": int(len(returns)),
            "n_assets": int(returns.shape[1]),
        },
        "git_revision": git_revision(ROOT),
        "generated_at": pd.Timestamp.now().isoformat(),
        "source_artifacts": {rel: _sha256(ROOT / rel) for rel in sources},
    }


def _evaluate_gates(gates: dict, window: dict, coverage: dict, manifest: dict,
                    synchrony: dict, freedom: dict, lookahead: dict) -> list[dict]:
    """Score every pre-registered readiness gate. Thresholds come from config.

    Each gate records its own threshold and observed value so the artifact is
    auditable without re-reading this file.
    """
    checks: list[dict] = []

    def check(name: str, passed: bool, observed, threshold, note: str) -> None:
        checks.append({
            "gate": name, "passed": bool(passed),
            "observed": observed, "threshold": threshold, "note": note,
        })

    # (a) the 2004 window is genuinely preserved
    check(
        "window_start_preserved",
        window["start"] <= gates["max_start_date"],
        window["start"], gates["max_start_date"],
        "The universe exists to provide ~21 years; a later start means an "
        "instrument silently truncated the common history.",
    )
    check(
        "sufficient_trading_days",
        coverage["silver_rows"] >= gates["min_trading_days"],
        coverage["silver_rows"], gates["min_trading_days"],
        "Row count after calendar alignment and the initial-NaN drop.",
    )

    # (b) coverage and missingness
    check(
        "forward_fill_share_acceptable",
        coverage["forward_filled_share"] <= gates["max_ffill_share"],
        round(coverage["forward_filled_share"], 5), gates["max_ffill_share"],
        "Liquid US-listed ETFs should need almost no carrying-forward.",
    )
    check(
        "zero_return_share_acceptable",
        coverage["max_zero_return_share"] <= gates["max_zero_return_share"],
        round(coverage["max_zero_return_share"], 5), gates["max_zero_return_share"],
        "full_2021's BVC block ran 17.1% zero-return days — the stale-price "
        "signature this universe must not reproduce.",
    )
    check(
        "feature_warmup_bounded",
        manifest["max_leading_nan"] <= gates["max_feature_leading_nan"],
        manifest["max_leading_nan"], gates["max_feature_leading_nan"],
        "With the BAM block excluded this should be the DXY warm-up (~231), "
        "not TAUX_DIR's 3,101.",
    )

    # (c) synchrony
    ratio = synchrony.get("max_abs_lag1_over_same_day")
    check(
        "synchronous_trading",
        ratio is not None and ratio <= gates["max_lag1_over_same_day"],
        ratio, gates["max_lag1_over_same_day"],
        "Non-synchronous pairs show lag-1 correlation exceeding same-day. "
        "full_2021's BVC block measured 19.1x; a shared NYSE session should "
        "be far below 1.",
    )

    # (d) no lookahead — the project's automatic-failure criterion (§18)
    check(
        "no_lookahead_past_weights_unchanged",
        lookahead["past_weights_unchanged_by_future_corruption"],
        lookahead["past_weights_unchanged_by_future_corruption"], True,
        "Corrupting the FUTURE of the feature frame must not change ANY past "
        f"rebalance weight. Probe tilts on {lookahead['probe_feature']}, which "
        f"carries {lookahead['probe_feature_leading_nan']} leading NaN — the "
        "warm-up trap §11 warns a new universe springs on the next model.",
    )
    check(
        "no_lookahead_guarantee_non_vacuous",
        lookahead["guarantee_is_non_vacuous"],
        lookahead["future_weights_did_change"], True,
        "Future weights MUST move under corruption, else the probe ignores the "
        "feature and the unchanged-past result proves nothing.",
    )

    # (e) allocation freedom — the reason the universe exists
    mvlw = freedom["min_variance_lw"]
    check(
        "allocation_expressive",
        mvlw["distinct_share"] >= gates["min_distinct_allocation_share"],
        mvlw["distinct_share"], gates["min_distinct_allocation_share"],
        "etf_2017 produces 1 distinct min_variance_lw allocation in 248 "
        "rebalances (share 0.004). This is the defect the universe exists to "
        "remove, under the SAME 25% cap.",
    )
    check(
        "cap_not_dominating",
        mvlw["mean_assets_at_cap"] <= gates["max_mean_assets_at_cap"],
        mvlw["mean_assets_at_cap"], gates["max_mean_assets_at_cap"],
        "How many positions sit pinned at the cap on average.",
    )
    return checks


def run() -> dict:
    """Build the global_2004 data layer and write the readiness artifact."""
    cfg = load_global_config()
    params = load_params()
    bp = params["backtest"]
    paths = {k: ROOT / v for k, v in cfg["paths"].items()}

    log.info("=== global_2004 checkpoint 1: data readiness (NO performance) ===")

    # Bronze — reuse the cached download if present, so a rerun is cheap and
    # a rate-limited retry does not re-hit ten endpoints.
    if paths["bronze_prices"].exists():
        log.info("Reusing Bronze at %s", paths["bronze_prices"])
        prices = pd.read_parquet(paths["bronze_prices"])
    else:
        prices = ingest_global_prices(
            cfg["tickers"], cfg["start_date"], paths["bronze_prices"]
        )

    missing = [t for t in cfg["tickers"] if t not in prices.columns]
    if missing:
        raise ValueError(f"Frozen instruments absent from Bronze: {missing}")
    prices = prices[cfg["tickers"]]

    per_ticker_start = {
        col: str(prices[col].first_valid_index().date()) for col in prices.columns
    }

    # Silver
    log_returns, coverage = build_global_silver(
        prices, ffill_limit=int(params["clean"]["ffill_limit"]),
        out_path=paths["silver_returns"],
    )

    # Gold
    macro = load_retained_macro(cfg["macro_exclude"])
    features, manifest = build_global_gold(
        log_returns, macro, params["ml_features"],
        returns_out=paths["gold_returns"],
        features_out=paths["gold_features"],
        manifest_out=paths["gold_manifest"],
    )

    window = {
        "start": str(log_returns.index.min().date()),
        "end": str(log_returns.index.max().date()),
        "years": round((log_returns.index.max() - log_returns.index.min()).days / 365.25, 2),
        "per_ticker_first_date": per_ticker_start,
        "binding_instrument": max(per_ticker_start, key=per_ticker_start.get),
    }

    log.info("Measuring synchrony (Stage-A diagnostic)...")
    synchrony = measure_synchrony(log_returns)

    log.info("Running the no-lookahead future-corruption gate on real data...")
    lookahead = verify_no_lookahead(
        log_returns, features,
        min_train_days=int(bp["min_train_days"]),
        rebalance_freq=bp["rebalance_freq"],
    )

    log.info("Measuring allocation freedom under the SAME 25%% cap (weights only)...")
    freedom = measure_allocation_freedom(
        log_returns,
        max_weight=float(bp["max_weight"]),
        rebalance_freq=bp["rebalance_freq"],
        min_train_days=int(bp["min_train_days"]),
    )

    checks = _evaluate_gates(cfg["readiness_gates"], window, coverage, manifest,
                             synchrony, freedom, lookahead)
    passed = all(c["passed"] for c in checks)

    artifact = {
        "provenance": _provenance(log_returns, [
            "params_global_2004.yaml",
            "src/global_universe.py",
            "src/run_global_2004_data.py",
            str(paths["gold_returns"].relative_to(ROOT)),
            str(paths["gold_features"].relative_to(ROOT)),
        ]),
        "checkpoint": "1 — DATA READINESS",
        "scope_note": (
            "NO performance quantity is computed at this checkpoint. No Sharpe, "
            "no portfolio return, no drawdown, no turnover. The allocation-"
            "freedom block is derived from WEIGHTS ONLY and measures whether "
            "the optimizer can express a view, not whether it earned anything."
        ),
        "frozen_protocol": "docs/GLOBAL_UNIVERSE_PREREGISTRATION.md @ c20e606",
        "tickers": cfg["tickers"],
        "window": window,
        "coverage": coverage,
        "feature_manifest": manifest,
        "synchrony": synchrony,
        "no_lookahead": lookahead,
        "allocation_freedom": freedom,
        "reference_comparison": {
            "etf_2017_min_variance_lw_distinct": 1,
            "etf_2017_n_rebalances": 248,
            "etf_2017_distinct_share": 0.004,
            "source": "data/gold/etf_cap_verdict.json (cap 0.25)",
        },
        "gates": checks,
        "verdict": "READY" if passed else "NOT READY",
        "n_gates_passed": sum(c["passed"] for c in checks),
        "n_gates": len(checks),
    }

    paths["readiness"].parent.mkdir(parents=True, exist_ok=True)
    paths["readiness"].write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    log.info("Readiness artifact -> %s", paths["readiness"])

    log.info("")
    for c in checks:
        log.info("  [%s] %-32s observed=%s threshold=%s",
                 "PASS" if c["passed"] else "FAIL", c["gate"], c["observed"], c["threshold"])
    log.info("")
    log.info("  VERDICT: %s (%d/%d gates)", artifact["verdict"],
             artifact["n_gates_passed"], artifact["n_gates"])
    log.info("")
    log.info("  Checkpoint 1 complete. NO performance was computed. Q1/Q2 are")
    log.info("  a separate, later step and must not be run until this is reviewed.")
    return artifact


if __name__ == "__main__":
    run()
