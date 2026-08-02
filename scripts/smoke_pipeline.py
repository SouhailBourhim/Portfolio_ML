"""smoke_pipeline.py — fast wiring check on a SYNTHETIC universe. Not a result.

WHAT THIS IS FOR. The real evaluation is slow by necessity: `phase4c_compare`
runs 14 strategies over 248 monthly rebalances of a 21-year window and takes
~45 minutes, and Phase 5's purged-CV search is longer still. That is the right
cost for a final research number and the wrong cost for CI or for checking
that an import still resolves.

This target runs the REAL engine, the REAL strategy classes and the REAL
`Strategy.fit` seam over a small deterministic synthetic universe, so it fails
when the wiring breaks — a renamed parameter, a broken import, a strategy that
stops returning valid weights, a cap violation — in seconds rather than in an
hour.

WHAT THIS IS EMPHATICALLY NOT. It is not an evaluation, not a hurdle, and not
comparable to anything in the report. The data is generated from a fixed seed
and has no economic content whatsoever, so any Sharpe it prints is an artifact
of the random number generator. It writes to `data/smoke/`, never to
`data/gold/`, so it cannot contaminate a published artifact, and it is not an
input to `snapshot_manifest`. Never substitute it for `phase4c_compare` or
`phase5_compare` when producing results.

Addresses: P4 — a slow verification loop is a correctness risk in itself: it
encourages skipping the check before a change. Making the wiring cheap to test
protects the expensive evaluation from being run on broken code.

Usage:
    python scripts/smoke_pipeline.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest import build_cost_vector, run_backtest  # noqa: E402
from strategies import (  # noqa: E402
    EqualWeight,
    MaxSharpe,
    MinVariance,
    MinVarianceEWMA,
    MinVarianceLW,
    RandomForestSignalStrategy,
    RegimeConditionalStrategy,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("smoke")

OUT_PATH = ROOT / "data" / "smoke" / "smoke_results.json"

N_DAYS = 900
ASSETS = ["SPY", "QQQ", "GLD", "TLT", "IAM.CS"]   # names only, for the cost vector split
MAX_WEIGHT = 0.25
MIN_TRAIN_DAYS = 252


def synthetic_universe(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A deterministic synthetic market with a regime shift, and its features.

    Two volatility regimes so the HMM has something to find; no attempt at
    realism beyond that, because nothing here is interpreted as a result.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2019-01-02", periods=N_DAYS, name="Date")

    vol = np.where(np.arange(N_DAYS) < N_DAYS // 2, 0.008, 0.020)
    drift = np.where(np.arange(N_DAYS) < N_DAYS // 2, 0.0005, -0.0002)
    returns = pd.DataFrame(
        rng.normal(drift[:, None], vol[:, None], size=(N_DAYS, len(ASSETS))),
        index=index, columns=ASSETS,
    )

    market = returns.mean(axis=1)
    features = pd.DataFrame(
        {
            "MARKET_RETURN": market,
            "MARKET_VOL_SHORT": market.rolling(21, min_periods=21).std(),
            "MARKET_VOL_LONG": market.rolling(63, min_periods=63).std(),
            "AVG_PAIRWISE_CORR": returns.rolling(63, min_periods=42).corr()
            .groupby(level=0).mean().mean(axis=1),
        }
    ).bfill()
    return returns, features


def build_strategies() -> list:
    """A representative slice of the ladder — one strategy per mechanism."""
    return [
        EqualWeight(),
        MinVariance(max_weight=MAX_WEIGHT),
        MinVarianceLW(max_weight=MAX_WEIGHT),
        MinVarianceEWMA(max_weight=MAX_WEIGHT),
        MaxSharpe(max_weight=MAX_WEIGHT),
        # The cap lives on the sub-strategies this one dispatches to, not on
        # the dispatcher itself — it delegates the whole decision.
        RegimeConditionalStrategy(
            bull_strategy=MaxSharpe(max_weight=MAX_WEIGHT),
            bear_strategy=MinVarianceLW(max_weight=MAX_WEIGHT),
        ),
        RandomForestSignalStrategy(
            max_weight=MAX_WEIGHT,
            model_params={"n_estimators": 25, "max_depth": 4},
            min_train_rows=100,
            momentum_windows=[5, 21],
        ),
    ]


def main() -> int:
    returns, features = synthetic_universe()
    costs = build_cost_vector(returns.columns, etf_cost_bps=10, bvc_cost_bps=30)

    started = time.perf_counter()
    summary: dict[str, dict] = {}
    for strategy in build_strategies():
        result = run_backtest(
            returns, strategy,
            rebalance_freq="ME",
            min_train_days=MIN_TRAIN_DAYS,
            cost_bps=costs,
            extras={"features": features},
            universe_name="smoke_synthetic",
            max_weight=MAX_WEIGHT,
        )

        weights = result.target_weights
        # The invariants the engine promises. A violation here is a wiring
        # defect, and it is the entire point of this target.
        assert not weights.isna().any().any(), f"{strategy.name}: NaN weight"
        assert np.allclose(weights.sum(axis=1), 1.0), f"{strategy.name}: weights must sum to 1"
        assert (weights >= -1e-9).all().all(), f"{strategy.name}: long-only violated"
        assert weights.max().max() <= MAX_WEIGHT + 1e-9, f"{strategy.name}: cap violated"
        assert len(result.net_returns) > 0, f"{strategy.name}: produced no OOS returns"

        summary[strategy.name] = {
            "n_rebalances": int(len(result.rebalance_dates)),
            "n_oos_days": int(len(result.net_returns)),
            "avg_turnover": round(float(result.turnover.mean()), 6),
        }

    elapsed = time.perf_counter() - started
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "WARNING": (
                    "SYNTHETIC WIRING CHECK — NOT A RESULT. Generated from a fixed random "
                    "seed with no economic content. Never cite, compare, or publish these "
                    "numbers; they exist only to prove the engine and strategies still run."
                ),
                "seed": 7,
                "universe": "smoke_synthetic",
                "n_days": N_DAYS,
                "assets": ASSETS,
                "elapsed_seconds": round(elapsed, 2),
                "strategies": summary,
            },
            indent=2, sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"smoke OK — {len(summary)} strategies in {elapsed:.1f}s → "
          f"{OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
