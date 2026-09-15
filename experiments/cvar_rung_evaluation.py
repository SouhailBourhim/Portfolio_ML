"""
cvar_rung_evaluation.py — does the mean-CVaR rung earn its place?

WHY THIS EXPERIMENT EXISTS. `docs/LITERATURE_IMPROVEMENTS.md` item 2.5
(Rockafellar & Uryasev 2000) observes that every optimizer in `strategies.py`
targets variance or the Sharpe ratio, while `metrics` reports drawdown and
Calmar for strategies that were never trying to achieve them. `MinCVaR` closes
that gap. This evaluates it.

IT RUNS THE CONTROL FIRST, which is the protocol `docs/REACHABLE_CLAIMS.md` §6
adopted after that note twice explained a null with a mechanism it had not
measured. Two controls apply here and both are checked before any verdict is
read:

  PANEL control -- `max_sharpe` vs `min_variance_lw` on log volatility, true by
  construction on any panel. On `full_2021` it is |t| = 6.10; on the deep
  Moroccan panel it is 2.14, at 0.86 of its own detection threshold, which is
  why that panel is not used here. If a panel cannot see a tautology, a null
  from it says nothing about a new rung.

  RUNG control -- `min_cvar` must attain lower IN-SAMPLE CVaR than every other
  rung, since that is its objective. `tests/test_min_cvar.py` pins this on
  synthetic data; it is re-checked here on the real panel, because a rung that
  fails its own objective in sample cannot be interpreted out of sample.

Only then does it ask the out-of-sample question, and it asks it of the
estimand the rung is FOR. Adding a downside optimizer and then ranking it on
Sharpe would repeat the error §1 of that note diagnoses: the estimand decides
what is reachable, not the model.

Nothing here is selected or tuned. `confidence` is fixed at the conventional
0.95 and `return_weight` at 0.0 -- the pure minimum-CVaR rung, the downside
analogue of `min_variance_lw` -- so this answers whether the OBJECTIVE changes
anything, not which of its settings looks best. Tuning them would be a search,
and `docs/EVALUATION_LIMITS.md` §6 measures what searches cost here.
"""
from __future__ import annotations

import itertools
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

from backtest import build_cost_vector, run_backtest
from inference import family_maxt_correction, paired_estimand_mde
from metrics import annualized_sharpe, certainty_equivalent, max_drawdown
from strategies import EqualWeight, MaxSharpe, MinVarianceLW, RegimeConditionalStrategy
from strategies_research import MinCVaR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("cvar_rung")

GOLD = ROOT / "data" / "gold"
OUT_JSON = GOLD / "cvar_rung_evaluation.json"

UNIVERSE = "full_2021"
RETURNS = GOLD / "log_returns.parquet"
# `RegimeConditionalStrategy` falls back to EQUAL WEIGHT when
# `extras["features"]` is absent, silently and by design, so a caller that
# forgets this wires a second equal-weight strategy into the family under a
# different label. The first run of this script did exactly that: it returned
# Sharpe, CVaR and turnover identical to `equal_weight` to four decimals, which
# is the only reason it was caught. The engine slices this frame to each train
# window itself.
FEATURES = GOLD / "ml_features_full.parquet"
MAX_W, RF, BLOCK_LEN, N_BOOT, SEED = 0.25, 0.0, 21, 4000, 0
ETF_BPS, BVC_BPS = 10.0, 30.0
CVAR_BETA = 0.95
TEST_FRAC = 0.35


def _cvar(x: np.ndarray, beta: float = CVAR_BETA) -> float:
    """Conditional value at risk: mean loss in the worst (1-beta) tail.
    Positive numbers are losses, so LOWER is better."""
    losses = -np.asarray(x, dtype=float)
    var = np.quantile(losses, beta)
    tail = losses[losses >= var]
    return float(tail.mean()) if tail.size else float(var)


def _sharpe(x: np.ndarray) -> float:
    return annualized_sharpe(pd.Series(x), risk_free_annual=RF)


def _ceq(x: np.ndarray) -> float:
    return certainty_equivalent(pd.Series(x), risk_aversion=1.0, risk_free_annual=RF)


def _mdd(x: np.ndarray) -> float:
    return max_drawdown(pd.Series(x))


def _log_vol(x: np.ndarray) -> float:
    sd = float(np.std(x, ddof=1))
    return float(np.log(sd)) if sd > 0 else float("-inf")


ESTIMANDS = {"sharpe": _sharpe, "ceq": _ceq, "max_drawdown": _mdd,
             "log_vol_ratio": _log_vol, "cvar": _cvar}


def main() -> dict:
    rets = pd.read_parquet(RETURNS)
    split = int(round(len(rets) * (1 - TEST_FRAC)))
    test_start = rets.index[split]
    costs = build_cost_vector(rets.columns, ETF_BPS, BVC_BPS)
    features = pd.read_parquet(FEATURES)
    log.info("PANEL %s: %d assets x %d days | frozen test from %s (%d rows)",
             UNIVERSE, rets.shape[1], len(rets), test_start.date(), len(rets) - split)

    strategies = {
        "min_cvar": MinCVaR(max_weight=MAX_W, confidence=CVAR_BETA, return_weight=0.0),
        "min_variance_lw": MinVarianceLW(max_weight=MAX_W),
        "max_sharpe": MaxSharpe(max_weight=MAX_W, risk_free_annual=RF),
        "equal_weight": EqualWeight(),
        "regime_conditional": RegimeConditionalStrategy(
            bull_strategy=MaxSharpe(max_weight=MAX_W),
            bear_strategy=MinVarianceLW(max_weight=MAX_W)),
    }

    # ---- RUNG CONTROL: the objective must win on itself, in sample ---------
    # `regime_conditional` is excluded: it needs regime `extras` the engine
    # supplies at backtest time, and it is not an optimizer with an objective
    # of its own to check.
    train = rets.iloc[:split]
    in_sample = {}
    for name, strat in strategies.items():
        if name == "regime_conditional":
            continue
        w = strat.fit(train)
        in_sample[name] = _cvar(np.expm1(train.to_numpy()) @ w.to_numpy())
    rung_ok = all(in_sample["min_cvar"] <= v + 1e-9 for v in in_sample.values())
    log.info("RUNG CONTROL (in-sample CVaR, lower is better): %s -> %s",
             {k: round(v, 5) for k, v in in_sample.items()},
             "PASS" if rung_ok else "FAIL")
    if not rung_ok:
        raise AssertionError("min_cvar does not minimise in-sample CVaR; do not read further")

    # ---- Backtests --------------------------------------------------------
    results, equity = {}, {}
    for name, strat in strategies.items():
        r = run_backtest(rets, strat, rebalance_freq="ME", min_train_days=252,
                         cost_bps=costs, extras={"features": features},
                         universe_name=f"{UNIVERSE}_cvar", max_weight=MAX_W)
        if r.fallback_rate > 0.0:
            log.warning("%s: %.1f%% of rebalances were produced by a FALLBACK, not the "
                        "requested model", name, 100 * r.fallback_rate)
        net = r.net_returns.loc[r.net_returns.index >= test_start]
        equity[name] = net
        results[name] = {
            "test_sharpe_net": round(_sharpe(net.to_numpy()), 4),
            "test_cvar_95": round(_cvar(net.to_numpy()), 5),
            "test_max_drawdown": round(float(_mdd(net.to_numpy())), 4),
            "annual_vol": round(float(net.std(ddof=1) * np.sqrt(252)), 4),
            "avg_turnover": round(float(r.turnover.mean()), 4),
            "fallback_rate": round(float(r.fallback_rate), 4),
        }
        log.info("%-20s Sharpe %+.4f | CVaR95 %.5f | maxDD %+.4f | vol %.4f | turn %.4f",
                 name, results[name]["test_sharpe_net"], results[name]["test_cvar_95"],
                 results[name]["test_max_drawdown"], results[name]["annual_vol"],
                 results[name]["avg_turnover"])

    # ---- PANEL CONTROL, on the frozen test window itself -------------------
    ctrl = paired_estimand_mde(equity["max_sharpe"], equity["min_variance_lw"],
                               _log_vol, block_len=BLOCK_LEN, n_boot=N_BOOT, seed=SEED)
    ctrl_ratio = float(ctrl["observed"] / ctrl["mde"]) if ctrl["mde"] else float("nan")
    panel_ok = abs(ctrl_ratio) > 1.0
    log.info("PANEL CONTROL max_sharpe vs min_variance_lw log vol: obs %+.4f MDE %.4f "
             "ratio %.2f -> %s", ctrl["observed"], ctrl["mde"], ctrl_ratio,
             "panel can resolve risk contrasts" if panel_ok else "PANEL CANNOT RESOLVE")

    # ---- Is the CVaR estimand reachable for this rung at all? --------------
    reach = {}
    for other in ("min_variance_lw", "max_sharpe", "equal_weight"):
        m = paired_estimand_mde(equity["min_cvar"], equity[other], _cvar,
                                block_len=BLOCK_LEN, n_boot=N_BOOT, seed=SEED)
        reach["min_cvar vs " + other] = {
            "observed": round(float(m["observed"]), 6),
            "mde_80": round(float(m["mde"]), 6),
            "obs_over_mde": round(float(m["observed"] / m["mde"]), 4) if m["mde"] else None,
        }
        log.info("REACHABILITY cvar  min_cvar vs %-18s obs %+.6f MDE %.6f ratio %+.2f",
                 other, m["observed"], m["mde"],
                 m["observed"] / m["mde"] if m["mde"] else float("nan"))

    # ---- Corrected family, now including the CVaR estimand ----------------
    names = sorted(equity)
    family = {}
    for a, b in itertools.combinations(names, 2):
        for est_name, fn in ESTIMANDS.items():
            family[est_name + ": " + a + " vs " + b] = (equity[a], equity[b], fn)
    n_pairs = len(list(itertools.combinations(names, 2)))
    log.info("FAMILY: %d pairs x %d estimands = %d hypotheses",
             n_pairs, len(ESTIMANDS), len(family))

    levels = {}
    for alpha in (0.05, 0.10):
        res = family_maxt_correction(family, block_len=BLOCK_LEN, n_boot=N_BOOT,
                                     alpha=alpha, seed=SEED)
        rows = sorted(res["results"].items(), key=lambda kv: -abs(kv[1]["t"]))
        surv = [k for k, v in rows if v["survives"]]
        levels["alpha_" + str(alpha)] = {
            "critical_value": round(float(res["critical_value"]), 4),
            "n_survivors": len(surv), "survivors": surv,
            "ranked": [{"hypothesis": k, "observed": round(float(v["observed"]), 5),
                        "t": round(float(v["t"]), 4), "survives": bool(v["survives"])}
                       for k, v in rows],
        }
        log.info("alpha=%.2f | critical |t|=%.3f | %d of %d survive",
                 alpha, res["critical_value"], len(surv), len(family))
        for k, v in rows[:6]:
            log.info("   %-56s |t|=%.2f %s", k, abs(v["t"]),
                     "SURVIVES" if v["survives"] else "")

    out = {
        "experiment": "cvar_rung_evaluation",
        "universe": UNIVERSE,
        "n_test_days": int(len(equity["min_cvar"])),
        "settings": {"confidence": CVAR_BETA, "return_weight": 0.0,
                     "max_weight": MAX_W,
                     "note": "fixed, not tuned — tuning would be a search"},
        "rung_control_in_sample_cvar": {k: round(v, 6) for k, v in in_sample.items()},
        "rung_control_pass": bool(rung_ok),
        "panel_control": {"observed": round(float(ctrl["observed"]), 4),
                          "mde_80": round(float(ctrl["mde"]), 4),
                          "obs_over_mde": round(ctrl_ratio, 4),
                          "pass": bool(panel_ok)},
        "strategies": results,
        "cvar_reachability": reach,
        "n_hypotheses": len(family),
        "levels": levels,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote %s", OUT_JSON.name)
    return out


if __name__ == "__main__":
    main()
