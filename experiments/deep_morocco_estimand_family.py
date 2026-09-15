"""
deep_morocco_estimand_family.py — which estimands survive on the 20-year panel?

WHY THIS EXPERIMENT EXISTS. `docs/REACHABLE_CLAIMS.md` §5 reports that zero of
forty hypotheses survive the Westfall-Young correction on the deep panel, and
names one surviving explanation for why the volatility advantage collapsed
there: the deep family contains no variance minimiser, so the contrast that
carried §2's two survivors -- `min_variance_lw`, directly and through
`regime_conditional` -- simply is not in it. §5 states that explanation is NOT
demonstrated, and says what would demonstrate it: put `min_variance_lw` in the
comparison set and re-run.

`deep_morocco_starvation.py` now carries that sixth strategy, so its equity
artifact has the missing column. This runs the family over it.

The family is every pair x every estimand computable from a net-equity curve:
C(6,2) = 15 pairs x 4 estimands = 60 hypotheses, against §5's 40. Cost drag and
turnover are absent for the same reason as in §5 -- the artifact stores net
equity only, so gross-vs-net and weight-change quantities cannot be recovered
from it. Adding a strategy therefore RAISES the correction's bar; a survivor
here has cleared a stricter test than §5's non-survivors faced, not a looser one.

Nothing here fits or selects anything. It reads one committed parquet and
applies the committed correction, so it is a diagnostic on a search already run
and costs no recompute.
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

from inference import family_maxt_correction, paired_estimand_mde
from metrics import annualized_sharpe, certainty_equivalent, max_drawdown

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("deep_morocco_family")

IN_EQUITY = ROOT / "data" / "gold" / "deep_morocco_equity.parquet"
OUT_JSON = ROOT / "data" / "gold" / "deep_morocco_estimand_family.json"

# Must match the run that produced the equity curve, or the Sharpe and CEQ
# rows would be computed against a different risk-free rate from every other
# number reported on this panel. `deep_morocco_starvation.py` sets RF = 0.0.
RF_ANNUAL = 0.0
RISK_AVERSION = 1.0
BLOCK_LEN = 21
N_BOOT = 4000
SEED = 0

# The four estimands a net-equity curve supports. Each maps a return array to a
# scalar; the tested quantity is always statistic(a) - statistic(b).
#
# The first three delegate to `src/metrics.py` rather than restating its
# formulae. That is not fastidiousness: the house Sharpe converts the risk-free
# rate GEOMETRICALLY and CEQ annualises its moments ARITHMETICALLY on excess
# returns, both for documented reasons, and a plausible-looking local
# reimplementation would silently report a different quantity from every other
# number in the project. The bootstrap hands each statistic an ndarray, so the
# only work here is restoring the Series those functions take.
def _sharpe(x: np.ndarray) -> float:
    return annualized_sharpe(pd.Series(x), risk_free_annual=RF_ANNUAL)


def _ceq(x: np.ndarray) -> float:
    return certainty_equivalent(pd.Series(x), risk_aversion=RISK_AVERSION,
                                risk_free_annual=RF_ANNUAL)


def _max_drawdown(x: np.ndarray) -> float:
    return max_drawdown(pd.Series(x))


def _log_vol(x: np.ndarray) -> float:
    """Log volatility. `src/metrics.py` exposes no standalone volatility
    function to delegate to, and the log is what makes the contrast a RATIO
    rather than a difference of levels. ddof=1 matches the pandas default the
    house Sharpe inherits from `returns.std()`."""
    sd = float(np.std(x, ddof=1))
    return float(np.log(sd)) if sd > 0 else float("-inf")


ESTIMANDS = {
    "sharpe": _sharpe,
    "ceq": _ceq,
    "max_drawdown": _max_drawdown,
    "log_vol_ratio": _log_vol,
}


def main(in_equity: Path | None = None, out_json: Path | None = None) -> dict:
    """Args default to the committed artifacts. They are overridable so the
    driver can be pointed at the pre-`min_variance_lw` equity curve and checked
    against `docs/REACHABLE_CLAIMS.md` §5's published 40-hypothesis numbers --
    a driver that cannot reproduce the result it extends is not trustworthy on
    the extension."""
    in_equity = in_equity or IN_EQUITY
    out_json = out_json or OUT_JSON
    equity = pd.read_parquet(in_equity)
    returns = equity.pct_change().dropna()
    names = sorted(returns.columns)
    log.info("PANEL: %d strategies x %d test days [%s -> %s]", len(names),
             len(returns), returns.index.min().date(), returns.index.max().date())
    log.info("STRATEGIES: %s", ", ".join(names))

    pairs = list(itertools.combinations(names, 2))
    family = {}
    for a, b in pairs:
        for est_name, fn in ESTIMANDS.items():
            family[f"{est_name}: {a} vs {b}"] = (returns[a], returns[b], fn)
    log.info("FAMILY: %d pairs x %d estimands = %d hypotheses",
             len(pairs), len(ESTIMANDS), len(family))

    out = {
        "experiment": "deep_morocco_estimand_family",
        "universe": "deep_morocco",
        "n_observations": int(len(returns)),
        "strategies": names,
        "n_hypotheses": len(family),
        "block_len": BLOCK_LEN, "n_boot": N_BOOT, "seed": SEED,
        "levels": {},
    }

    for alpha in (0.05, 0.10):
        res = family_maxt_correction(family, block_len=BLOCK_LEN, n_boot=N_BOOT,
                                     alpha=alpha, seed=SEED)
        crit = float(res["critical_value"])
        rows = sorted(res["results"].items(), key=lambda kv: -abs(kv[1]["t"]))
        survivors = [k for k, v in rows if v["survives"]]
        out["levels"][f"alpha_{alpha}"] = {
            "critical_value": round(crit, 4),
            "n_survivors": len(survivors),
            "survivors": survivors,
            "ranked": [{"hypothesis": k,
                        "observed": round(float(v["observed"]), 4),
                        "t": round(float(v["t"]), 4),
                        "survives": bool(v["survives"])} for k, v in rows],
        }
        log.info("alpha=%.2f | critical |t|=%.3f | %d of %d survive",
                 alpha, crit, len(survivors), len(family))
        for k, v in rows[:5]:
            log.info("   %-58s |t|=%.2f %s", k, abs(v["t"]),
                     "SURVIVES" if v["survives"] else "")

    # The specific contrast §5 predicts should reappear now the minimiser is in.
    if "min_variance_lw" in names and "regime_conditional" in names:
        for a, b in [("regime_conditional", "max_sharpe"),
                     ("min_variance_lw", "max_sharpe"),
                     ("min_variance_lw", "regime_conditional")]:
            if a in names and b in names:
                m = paired_estimand_mde(returns[a], returns[b], _log_vol,
                                        block_len=BLOCK_LEN, n_boot=N_BOOT, seed=SEED)
                out.setdefault("log_vol_reachability", {})[f"{a} vs {b}"] = {
                    "observed": round(float(m["observed"]), 4),
                    "standard_error": round(float(m["standard_error"]), 4),
                    "mde_80": round(float(m["mde"]), 4),
                    "obs_over_mde": round(float(m["observed"] / m["mde"]), 4)
                    if m["mde"] else None,
                }
                log.info("REACHABILITY log vol %-40s obs=%+.4f MDE=%.4f ratio=%.2f",
                         f"{a} vs {b}", m["observed"], m["mde"],
                         m["observed"] / m["mde"] if m["mde"] else float("nan"))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json.name)
    return out


if __name__ == "__main__":
    _in = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    _out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    main(_in, _out)
