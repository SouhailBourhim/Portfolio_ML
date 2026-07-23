"""
fundamentals_ic_lift.py — Do FUNDAMENTALS lift the F7 signal beyond prices?

The deep-Morocco experiment (2005-2024, 12 stocks, 56k pooled rows) ruled out
"more price history" as the missing ingredient: purged-CV IC rose 2-4x but the
portfolio edge did not. Its conclusion pointed at DATA QUALITY, not quantity —
which in practice means fundamentals. This experiment answers the follow-up:

    Add point-in-time fundamentals (P/E, P/B, P/S, D/E from stockanalysis.com,
    causally lagged 90 business days) on top of Phase 5's exact pipeline. Does
    the purged-CV information coefficient rise, or is it the same signal ceiling?

Ablation design — cheapest-decisive-first, same idiom Phase 5 uses:

    BASELINE  = price features + regime posterior     (Phase 5 exact)
    TREATMENT = price features + regime posterior + FUNDAMENTALS

For each universe (etf_2017, full_2021) × algorithm (RandomForest, XGBoost):
run `PurgedKFold` with the SAME grid Phase 5 used, score by IC. Report the
mean-IC lift as (treatment_best - baseline_best) with the CV std of each.

If |lift| is meaningful (say ≥ 0.01 above Phase 5's ~0.03 baseline), a
downstream portfolio run is justified (task #69). If not, stop — no portfolio
Sharpe hunt is going to salvage a signal the CV can't detect.

Notes on scope, honest limitations:
- Fundamentals cover only 4 BVC tickers (IAM, ATW, CIH, BCP). etf_2017 has NO
  BVC assets, so the treatment on that universe is a NEGATIVE CONTROL — the
  fundamentals columns are entirely cross-sectional-median fill, so IC there
  should not RISE. If it does, something's wrong (median-fill leaking a market
  timing signal). If it stays roughly the same, the wiring is clean.
- `full_2021` starts 2021-07; first fundamental available_from is ~2021-11
  (2021-06-30 + 90 bdays). The panel is NaN-dropped in build_supervised_dataset,
  so a few months of training rows are lost. Cost accepted; reported.
- Deterministic (all seeds fixed) so re-running reproduces the numbers.

Usage:
    .venv/bin/python experiments/fundamentals_ic_lift.py
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from ml_signals import (
    attach_fundamentals_features,
    attach_regime_feature,
    build_asset_features,
    build_supervised_dataset,
    melt_to_panel,
)
from model_selection import _grid_points, _instantiate, information_coefficient
from purged_kfold import PurgedKFold
from utils import load_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fundamentals_ic_lift")


# etf_2017 is DELIBERATELY excluded. It has zero BVC assets, so every
# fundamentals column would be pure cross-sectional-median fill — a nonsense
# experiment where any measured "IC lift" is a confound of the shortened
# treatment window (fundamentals only exist from 2021-11, so the training
# set silently shrinks from 2017-2023 to 2021-2023), not a real signal from
# the added feature. Running it once (this experiment's first pass) surfaced
# a spurious +0.06 to +0.08 IC lift on etf_2017 that vanished the moment
# baseline was restricted to the same dates. Documented and excluded, not
# silently dropped, so the choice is auditable.
UNIVERSES = {
    "full_2021": "data/gold/log_returns.parquet",
}


def build_xy(
    train_returns: pd.DataFrame,
    market_features: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
    fund_assets: list[str],
    short_window: int,
    long_window: int,
    momentum_windows,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the pooled (X, y) exactly like Phase 5, plus optional fundamentals.

    The only line that differs from `model_selection._build_panel_xy` is the
    conditional `attach_fundamentals_features` call. Everything else is by
    definition the Phase 5 baseline — apples to apples.
    """
    wide = build_asset_features(
        train_returns,
        short_window=short_window,
        long_window=long_window,
        momentum_windows=momentum_windows,
    )
    panel = melt_to_panel(wide, list(train_returns.columns))
    panel = attach_regime_feature(panel, market_features)
    if fundamentals is not None:
        panel = attach_fundamentals_features(
            panel, fundamentals, fund_assets=fund_assets
        )
    X, y, _ = build_supervised_dataset(panel, train_returns)
    return X, y


def score_grid(
    X: pd.DataFrame,
    y: pd.Series,
    grid: dict,
    model_type: str,
    n_splits: int,
    embargo_frac: float,
) -> list[dict]:
    """Purged-CV grid search, IC-scored. Same code path Phase 5 uses."""
    cv = PurgedKFold(n_splits=n_splits, embargo_frac=embargo_frac, label_horizon=1)
    splits = list(cv.split(X))
    rows = []
    for params in _grid_points(grid):
        fold_ics = []
        for train_idx, test_idx in splits:
            model = _instantiate(model_type, params)
            model.fit(X.iloc[train_idx].to_numpy(), y.iloc[train_idx].to_numpy())
            pred = model.predict(X.iloc[test_idx].to_numpy())
            fold_ics.append(
                information_coefficient(y.iloc[test_idx].to_numpy(), pred)
            )
        rows.append({
            **params,
            "mean_ic": float(np.mean(fold_ics)),
            "std_ic": float(np.std(fold_ics)),
            "n_folds": len(fold_ics),
        })
    return sorted(rows, key=lambda r: -r["mean_ic"])


def run_one_universe(
    universe: str,
    returns_path: Path,
    params: dict,
    fundamentals: pd.DataFrame,
    fund_assets: list[str],
) -> dict:
    log.info("=== %s ===", universe)
    returns = pd.read_parquet(returns_path)
    features_path = (
        ROOT / "data" / "gold" / f"ml_features_{'etf' if 'etf' in universe else 'full'}.parquet"
    )
    market_features = pd.read_parquet(features_path)

    p5 = params["phase5"]
    ms = params["ml_signals"]
    test_frac = p5["test_frac"]
    n_split_dates = int(len(returns) * (1 - test_frac))
    train_val_returns = returns.iloc[:n_split_dates]
    train_val_features = market_features.reindex(train_val_returns.index).ffill()

    log.info(
        "train+val window: %s → %s (%d rows)",
        train_val_returns.index[0].date(),
        train_val_returns.index[-1].date(),
        len(train_val_returns),
    )

    kwargs = dict(
        short_window=ms["short_window"],
        long_window=ms["long_window"],
        momentum_windows=ms["momentum_windows"],
    )

    log.info("[%s] BASELINE (price + regime only) ...", universe)
    X_b_full, y_b_full = build_xy(
        train_val_returns,
        train_val_features,
        fundamentals=None,
        fund_assets=fund_assets,
        **kwargs,
    )
    log.info("[%s] baseline X shape %s, y shape %s", universe, X_b_full.shape, y_b_full.shape)

    log.info("[%s] TREATMENT (+ fundamentals) ...", universe)
    X_t, y_t = build_xy(
        train_val_returns,
        train_val_features,
        fundamentals=fundamentals,
        fund_assets=fund_assets,
        **kwargs,
    )
    log.info("[%s] treatment X shape %s, y shape %s", universe, X_t.shape, y_t.shape)

    # Strict apples-to-apples: baseline restricted to the EXACT SAME
    # (date, asset) rows the treatment kept. Adding fundamentals shrinks the
    # training set (the first months of full_2021 have no published report
    # yet — `build_supervised_dataset` NaN-drops those (date, asset) pairs).
    # Comparing full baseline vs shrunken treatment would confound "extra
    # feature" with "different sample"; using .reindex on treatment's exact
    # MultiIndex removes both the date confound and the (date, asset)
    # subset confound in one step.
    common_idx = X_b_full.index.intersection(X_t.index)
    X_b = X_b_full.loc[common_idx]
    y_b = y_b_full.loc[common_idx]
    log.info(
        "[%s] baseline reindexed to treatment (date, asset) rows: %s → %s (%d rows dropped)",
        universe, X_b_full.shape, X_b.shape, len(X_b_full) - len(X_b),
    )

    cv = params["purged_cv"]
    result = {"universe": universe, "baseline": {}, "treatment": {}}

    for algo, grid_key in [("random_forest", "rf_grid"), ("xgboost", "xgb_grid")]:
        grid = p5[grid_key]
        log.info("[%s / %s] baseline grid search ...", universe, algo)
        b_rows = score_grid(X_b, y_b, grid, algo, cv["n_splits"], cv["embargo_frac"])
        log.info("[%s / %s] treatment grid search ...", universe, algo)
        t_rows = score_grid(X_t, y_t, grid, algo, cv["n_splits"], cv["embargo_frac"])

        b_best = b_rows[0]
        t_best = t_rows[0]
        lift = t_best["mean_ic"] - b_best["mean_ic"]
        log.info(
            "[%s / %s] BASELINE mean IC = %.4f ± %.4f  |  TREATMENT %.4f ± %.4f  |  LIFT %.4f",
            universe, algo,
            b_best["mean_ic"], b_best["std_ic"],
            t_best["mean_ic"], t_best["std_ic"],
            lift,
        )
        result["baseline"][algo] = {
            "best": b_best,
            "top3": b_rows[:3],
            "X_shape": list(X_b.shape),
        }
        result["treatment"][algo] = {
            "best": t_best,
            "top3": t_rows[:3],
            "X_shape": list(X_t.shape),
            "ic_lift_vs_baseline": lift,
        }
    return result


def main() -> None:
    params = load_params()
    fund_cfg = params["fundamentals"]
    fund_path = ROOT / fund_cfg["output_path"]
    if not fund_path.exists():
        raise SystemExit(
            f"Fundamentals Gold panel missing at {fund_path}. Run:\n"
            f"  .venv/bin/python src/fundamentals.py\n"
        )

    fundamentals = pd.read_parquet(fund_path)
    fund_assets = list(fund_cfg["tickers"])

    # In BVC universes the tickers carry a `.CS` suffix; strip when matching.
    # (Actually check the returns columns first — they might already be "IAM"
    # not "IAM.CS" depending on the universe.)
    for u_name in UNIVERSES:
        cols = pd.read_parquet(ROOT / UNIVERSES[u_name], columns=None).columns.tolist()
        log.info("universe %s columns: %s", u_name, cols)

    # Universe columns are `IAM.CS`, `ATW.CS`, `BCP.CS`, `CIH.CS`. Fundamentals
    # columns are `IAM__FUND_pe` etc. Rewrite fundamentals columns to match the
    # `.CS` suffix so `fund_assets` and `attach_fundamentals_features` align
    # with the ASSET index level of the pooled panel.
    rename = {}
    for c in fundamentals.columns:
        if "__FUND_" in c:
            ticker, rest = c.split("__FUND_")
            rename[c] = f"{ticker}.CS__FUND_{rest}"
    fundamentals = fundamentals.rename(columns=rename)
    fund_assets = [f"{t}.CS" for t in fund_assets]

    results = {}
    for u_name, path in UNIVERSES.items():
        results[u_name] = run_one_universe(
            u_name, ROOT / path, params, fundamentals, fund_assets
        )

    # ── Report ─────────────────────────────────────────────────────────────
    log.info("=" * 72)
    log.info("SUMMARY — fundamentals IC lift over Phase 5 baseline")
    log.info("=" * 72)
    for u_name, r in results.items():
        for algo in ("random_forest", "xgboost"):
            b = r["baseline"][algo]["best"]
            t = r["treatment"][algo]["best"]
            log.info(
                "%s  |  %-13s  |  baseline %.4f ± %.4f  →  treatment %.4f ± %.4f  |  lift %+.4f",
                u_name, algo,
                b["mean_ic"], b["std_ic"],
                t["mean_ic"], t["std_ic"],
                t["mean_ic"] - b["mean_ic"],
            )
    log.info("=" * 72)

    out = ROOT / "data" / "gold" / "fundamentals_ic_lift.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
