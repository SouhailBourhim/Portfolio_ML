"""
deep_morocco_pbo.py — Was the deep-panel search overfit, or just underpowered?

WHY THIS EXPERIMENT EXISTS. `docs/EVALUATION_LIMITS.md` §6 reports a CSCV
degradation slope near -1 on the shallow universes: the configuration that wins
in-sample does not merely fail to win out-of-sample, its ranking inverts. That
was measured over a 240-configuration search on 455 frozen-test days, where
`docs/REACHABLE_CLAIMS.md` §2 shows almost nothing is resolvable. Two readings
survive that, and they call for opposite responses:

  1. Selection overfitting is intrinsic to this problem -> shrink every search.
  2. It is an artefact of selecting on a sample too short to rank anything ->
     more out-of-sample data should temper it.

`deep_morocco_starvation.py` has the window that separates them: 1,638 frozen
test days, 3.6x the shallow one. It cannot answer the question itself, because
CSCV needs the out-of-sample series of EVERY searched configuration and that
experiment retains only the two winners. This runs the same 15 configurations
its Stage A actually evaluated -- 6 RF, 9 XGB -- through the unchanged backtest,
keeps all 15 series, and computes the PBO.

Nothing here selects anything. It is a diagnostic on a search already run, so
the grids are imported from that module rather than restated, and drift between
the two is impossible by construction.

ON PARALLELISM, because it changes a number if done carelessly. Fitting is left
single-threaded for XGBoost, exactly as `src/ml_signals.py` pins it, since its
histogram builder's thread reduction is not demonstrated bit-reproducible here
and 9 XGB configurations cost minutes anyway. RandomForest IS parallelised, via
an explicit `model_params["n_jobs"]` which that module documents as
authoritative. That is safe for a measured reason rather than a seed argument:
on this machine and this data, n_jobs of 1, 8 and -1 return bit-identical
predictions (0.002553829799) while fitting in 27.25s, 3.88s and 1.62s. Without
it the six RF configurations alone cost about 2.7 hours.
"""
from __future__ import annotations

import importlib.util
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
from sklearn.model_selection import ParameterGrid

from backtest import run_backtest
from inference import probability_of_backtest_overfitting
from strategies import RandomForestSignalStrategy, XGBoostSignalStrategy

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("deep_morocco_pbo")

OUT_JSON = ROOT / "data" / "gold" / "deep_morocco_pbo.json"
N_SPLITS = 10          # C(10,5) = 252 balanced partitions
RF_N_JOBS = -1         # bit-identical to 1; see the module docstring


def _starvation():
    """Import the sibling experiment for its universe, split and grids."""
    path = ROOT / "experiments" / "deep_morocco_starvation.py"
    spec = importlib.util.spec_from_file_location("deep_morocco_starvation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> dict:
    dm = _starvation()
    log_ret, features = dm.load_universe()
    split = int(round(len(log_ret) * (1 - dm.TEST_FRAC)))
    test_start = log_ret.index[split]
    log.info("UNIVERSE: %d x %d | frozen test from %s (%d days)",
             log_ret.shape[1], log_ret.shape[0], test_start.date(), len(log_ret) - split)

    common = dict(mu_transform="shrink", shrinkage_weight=0.5, turnover_penalty=1.0,
                  max_weight=dm.MAX_W, risk_free_annual=dm.RF, min_train_rows=504,
                  short_window=21, long_window=63, momentum_windows=(5, 21, 63),
                  condition_on_regime=True, **dm.REGIME_KW)

    configs = []
    for params in ParameterGrid(dm.RF_GRID):
        configs.append(("rf", RandomForestSignalStrategy, {**params, "n_jobs": RF_N_JOBS}, params))
    for params in ParameterGrid(dm.XGB_GRID):
        configs.append(("xgb", XGBoostSignalStrategy, dict(params), params))
    log.info("CONFIGURATIONS: %d (%d RF + %d XGB) -- the same grids Stage A evaluated",
             len(configs), len(list(ParameterGrid(dm.RF_GRID))), len(list(ParameterGrid(dm.XGB_GRID))))

    series, meta = {}, {}
    for i, (fam, cls, model_params, recorded) in enumerate(configs, 1):
        name = f"{fam}_{i:02d}"
        strat = cls(name=name, model_params=model_params, **common)
        r = run_backtest(log_ret, strat, rebalance_freq="ME", min_train_days=252,
                         cost_bps=dm.BVC_COST_BPS, extras={"features": features},
                         universe_name="deep_morocco_pbo", max_weight=dm.MAX_W)
        test_net = r.net_returns.loc[r.net_returns.index >= test_start]
        series[name] = test_net
        sharpe = float(test_net.mean() / test_net.std(ddof=0) * np.sqrt(252))
        meta[name] = {"family": fam, "params": recorded, "test_sharpe_net": round(sharpe, 4)}
        log.info("[%2d/%2d] %s %s -> test Sharpe %.4f", i, len(configs), name, recorded, sharpe)

    pbo = probability_of_backtest_overfitting(series, n_splits=N_SPLITS)
    out = {
        "experiment": "deep_morocco_pbo",
        "question": "does 3.6x more out-of-sample data temper the -1 degradation slope?",
        "window": {"test_start": str(test_start.date()),
                   "test_days": int(len(log_ret) - split), "n_configurations": len(configs)},
        "pbo": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in pbo.items()
                if not isinstance(v, (list, dict))},
        "configurations": meta,
        "reference_shallow": {
            "source": "docs/EVALUATION_LIMITS.md section 6",
            "full_2021": {"pbo": 0.333333, "slope": -1.082482},
            "etf_2017": {"pbo": 0.714286, "slope": -0.993645},
            "n_trials": 240, "test_days": 455,
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote %s", OUT_JSON.name)
    log.info("VERDICT: PBO=%.4f slope=%.4f (shallow: 0.333/-1.082 and 0.714/-0.994)",
             pbo["pbo"], pbo["performance_degradation_slope"])
    return out


if __name__ == "__main__":
    main()
