"""
backtest.py — Leakage-free walk-forward backtesting engine.

The engine, not the strategy, is the trust boundary: at every rebalance it
slices `returns.loc[:τ]` (and every extras frame likewise) before calling
`strategy.fit`, so a strategy PHYSICALLY cannot see the future, and it
validates the returned weights rather than trusting them. This is the
mechanism that makes every Phase 2–5 result credible.

Addresses: P4 — no code path lets future data influence a past decision;
the timing convention (fit at close of τ, earn from τ+1, costs deducted at
τ+1) exists because applying new weights same-day would trade at prices the
decision already used.
Addresses: P2 — the expanding window re-estimates parameters at every
rebalance, so strategies adapt as regimes shift instead of assuming one
stationary world.

Currency caveat: BVC returns are MAD-denominated, ETF returns USD. Returns
are unitless so the arithmetic is valid, but portfolio results embed an
unhedged USD/MAD exposure. This is stated, not fixed — an FX-hedging model
is out of scope (also documented in the Phase 2 notebook and CLAUDE.md §8.4).
"""

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from schemas import BVC_ASSETS, ETF_ASSETS
from strategies import Strategy

log = logging.getLogger("backtest")


@dataclass(frozen=True)
class BacktestResult:
    """Everything the metrics layer and notebook need, nothing recomputable."""

    strategy_name: str
    universe: str
    rebalance_dates: pd.DatetimeIndex
    target_weights: pd.DataFrame    # rebalance date × asset — what fit() returned
    drifted_weights: pd.DataFrame   # rebalance date × asset — weights just BEFORE trading
    gross_returns: pd.Series        # daily simple returns, OOS dates only
    net_returns: pd.Series          # gross minus transaction-cost drag
    turnover: pd.Series             # per rebalance: Σᵢ |w_target,i − w_drifted,i|
    costs: pd.Series                # per rebalance: Σᵢ cᵢ·|Δwᵢ|, as fraction of NAV


def build_cost_vector(
    assets: Iterable[str],
    etf_cost_bps: float,
    bvc_cost_bps: float,
    overrides: Mapping[str, float] | None = None,
) -> pd.Series:
    """
    Per-asset one-way transaction costs in basis points.

    Addresses: P1/P3 realism ("contraintes réalistes de gestion") — BVC
    names trade with wider spreads and thinner books than US ETFs (CIH.CS
    and BCP.CS carry standing illiquidity flags from the Silver layer), so
    a single flat cost would understate the price of holding Morocco.

    Raises:
        ValueError: for any asset not classifiable as ETF or BVC and not
            covered by an override — a silent default here would be a
            silent cost misstatement.
    """
    overrides = dict(overrides or {})
    costs: dict[str, float] = {}
    for asset in assets:
        if asset in overrides:
            costs[asset] = overrides[asset]
        elif asset in ETF_ASSETS:
            costs[asset] = etf_cost_bps
        elif asset in BVC_ASSETS:
            costs[asset] = bvc_cost_bps
        else:
            raise ValueError(
                f"Unknown asset '{asset}' — not in ETF_ASSETS or BVC_ASSETS and "
                f"no override provided. Refusing to guess its trading cost."
            )
    return pd.Series(costs, dtype=float)


def _validate_weights(weights: pd.Series, assets: pd.Index, strategy_name: str) -> pd.Series:
    """
    Engine-side weight validation — the engine never trusts a strategy.

    Addresses: P4 — a strategy returning malformed weights (wrong assets,
    shorts, leverage) would corrupt every downstream number; failing loudly
    at the source is the only acceptable behavior.
    """
    if not weights.index.equals(assets):
        raise ValueError(
            f"{strategy_name}: weights index {list(weights.index)} != universe assets "
            f"{list(assets)} — strategies must weight exactly the assets they were given."
        )
    if weights.isna().any():
        raise ValueError(f"{strategy_name}: weights contain NaN.")
    if (weights < -1e-9).any():
        raise ValueError(f"{strategy_name}: negative weights (short selling) not allowed.")
    if abs(weights.sum() - 1.0) > 1e-6:
        raise ValueError(f"{strategy_name}: weights sum to {weights.sum():.8f}, expected 1.")
    return weights.clip(lower=0.0)


def _rebalance_schedule(
    index: pd.DatetimeIndex, rebalance_freq: str, min_train_days: int
) -> pd.DatetimeIndex:
    """Last actual trading date of each period, with ≥ min_train_days of
    history behind it and at least one tradable day after it."""
    last_per_period = index.to_series().resample(rebalance_freq).last().dropna()
    positions = index.get_indexer(pd.DatetimeIndex(last_per_period.values))
    keep = [
        index[p] for p in positions
        if p >= min_train_days - 1 and p < len(index) - 1
    ]
    return pd.DatetimeIndex(keep)


def run_backtest(
    returns: pd.DataFrame,
    strategy: Strategy,
    rebalance_freq: str = "ME",
    min_train_days: int = 252,
    cost_bps: pd.Series | float = 0.0,
    extras: Mapping[str, pd.DataFrame] | None = None,
    universe_name: str = "",
) -> BacktestResult:
    """
    Expanding-window walk-forward backtest of one strategy.

    Mechanics (each rebalance date τ):
      1. train = returns.loc[:τ]  (engine slices; strategy sees ONLY this)
      2. w_target = strategy.fit(train, extras sliced to :τ) — then validated
      3. Turnover = Σ|w_target − w_drifted| where w_drifted is the portfolio
         as it stands at the close of τ after drifting with returns since the
         last rebalance (first rebalance: from cash → turnover = 1)
      4. Cost = Σᵢ (cᵢ_bps/10⁴)·|Δwᵢ|, deducted from the FIRST day's return
         under the new weights (τ+1)
      5. From τ+1 until the next rebalance, daily portfolio gross return is
         Σᵢ wᵢ(e^{rᵢ}−1) — log-returns are converted to simple returns
         because a weighted sum of LOG-returns is not a portfolio return —
         and weights drift: w_t = w_{t−1}(1+R_t)/Σⱼ w_{t−1,j}(1+R_{t,j})

    Args:
        returns: Gold-layer LOG-returns matrix (DatetimeIndex × assets).
        strategy: Any Strategy implementation.
        rebalance_freq: pandas offset alias ("ME" = month-end).
        min_train_days: Minimum history before the first rebalance.
        cost_bps: Per-asset one-way costs (pd.Series from build_cost_vector)
            or a flat scalar for all assets. 0.0 → gross == net.
        extras: Auxiliary frames for Phase 4 strategies; the ENGINE slices
            each one to the train window before every fit call.
        universe_name: Label carried into the result for reporting.

    Addresses: P4, P2 — see module docstring.
    """
    if isinstance(cost_bps, (int, float)):
        cost_vector = pd.Series(float(cost_bps), index=returns.columns)
    else:
        cost_vector = cost_bps.reindex(returns.columns)
        if cost_vector.isna().any():
            missing = cost_vector[cost_vector.isna()].index.tolist()
            raise ValueError(f"cost_bps missing entries for assets: {missing}")

    rebalance_dates = _rebalance_schedule(returns.index, rebalance_freq, min_train_days)
    if len(rebalance_dates) == 0:
        raise ValueError(
            f"No valid rebalance dates: {len(returns)} rows with min_train_days="
            f"{min_train_days} and freq={rebalance_freq}. Need more history."
        )

    simple_returns = np.exp(returns) - 1.0   # the classic log-vs-simple conversion
    assets = returns.columns
    cost_frac = cost_vector.to_numpy() / 10_000.0

    target_rows: list[pd.Series] = []
    drifted_rows: list[pd.Series] = []
    turnover_vals: list[float] = []
    cost_vals: list[float] = []
    gross_out: dict[pd.Timestamp, float] = {}
    net_out: dict[pd.Timestamp, float] = {}

    w_current = np.zeros(len(assets))   # start from cash

    for k, tau in enumerate(rebalance_dates):
        train = returns.loc[:tau].copy()
        extras_slice = (
            {name: frame.loc[:tau].copy() for name, frame in extras.items()}
            if extras else None
        )
        w_target = _validate_weights(
            strategy.fit(train, extras_slice), assets, strategy.name
        ).to_numpy()

        w_drifted = w_current.copy()
        turnover = float(np.abs(w_target - w_drifted).sum())
        cost = float((cost_frac * np.abs(w_target - w_drifted)).sum())

        target_rows.append(pd.Series(w_target, index=assets, name=tau))
        drifted_rows.append(pd.Series(w_drifted, index=assets, name=tau))
        turnover_vals.append(turnover)
        cost_vals.append(cost)

        # Simulate τ+1 .. next rebalance (or end of data)
        tau_pos = returns.index.get_loc(tau)
        end_pos = (
            returns.index.get_loc(rebalance_dates[k + 1])
            if k + 1 < len(rebalance_dates) else len(returns) - 1
        )
        w = w_target.copy()
        for pos in range(tau_pos + 1, end_pos + 1):
            date = returns.index[pos]
            r_simple = simple_returns.iloc[pos].to_numpy()
            gross = float(w @ r_simple)
            gross_out[date] = gross
            net_out[date] = gross - (cost if pos == tau_pos + 1 else 0.0)
            growth = w * (1.0 + r_simple)
            w = growth / growth.sum() if growth.sum() > 0 else np.zeros_like(w)
        w_current = w

    result = BacktestResult(
        strategy_name=strategy.name,
        universe=universe_name,
        rebalance_dates=rebalance_dates,
        target_weights=pd.DataFrame(target_rows),
        drifted_weights=pd.DataFrame(drifted_rows),
        gross_returns=pd.Series(gross_out, name="gross"),
        net_returns=pd.Series(net_out, name="net"),
        turnover=pd.Series(turnover_vals, index=rebalance_dates, name="turnover"),
        costs=pd.Series(cost_vals, index=rebalance_dates, name="cost"),
    )
    log.info(
        "Backtest %s/%s: %d rebalances, OOS %s → %s (%d days), avg turnover %.3f",
        strategy.name, universe_name or "unnamed", len(rebalance_dates),
        result.gross_returns.index.min().date(), result.gross_returns.index.max().date(),
        len(result.gross_returns), result.turnover.mean(),
    )
    return result
