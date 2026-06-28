"""
conftest.py — Shared fixtures for all test modules.

All fixtures use small synthetic data so tests run without internet access
and complete in milliseconds.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ASSETS = ["IAM.CS", "ATW.CS", "CIH.CS", "BCP.CS", "SPY", "QQQ", "EEM", "GLD", "TLT"]
MACRO_SERIES = ["VIX", "US10Y", "DXY", "HY_SPREAD"]


@pytest.fixture()
def synthetic_prices() -> pd.DataFrame:
    """600 business days of synthetic adjusted close prices, wide format."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-02", periods=600)
    # GBM-like prices starting at 100
    returns = rng.normal(0.0003, 0.012, size=(600, len(ASSETS)))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=ASSETS)


@pytest.fixture()
def synthetic_log_returns(synthetic_prices) -> pd.DataFrame:
    """Log-returns derived from synthetic prices (599 rows)."""
    import numpy as np
    log_ret = np.log(synthetic_prices / synthetic_prices.shift(1)).dropna()
    log_ret.index.name = "Date"
    return log_ret


@pytest.fixture()
def synthetic_macro() -> pd.DataFrame:
    """Synthetic macro data aligned to a daily calendar, wide format."""
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2020-01-02", periods=600)
    data = {
        "VIX":       20 + rng.normal(0, 3, 600),
        "US10Y":     1.5 + rng.normal(0, 0.1, 600),
        "DXY":       95 + rng.normal(0, 1, 600),
        "HY_SPREAD": 400 + rng.normal(0, 20, 600),
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df
