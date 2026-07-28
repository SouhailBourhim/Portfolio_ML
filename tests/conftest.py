"""
conftest.py — Shared fixtures for all test modules.

All fixtures use small synthetic data so tests run without internet access
and complete in milliseconds. `_no_network` below turns that from a
convention into a guarantee.
"""

import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class NetworkAccessAttempted(RuntimeError):
    """A test tried to open a socket. Tests must be hermetic (§16)."""


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """
    Fail any test that tries to reach the network, instead of letting it
    quietly succeed on whatever happens to be cached locally.

    Addresses: P4 (reproducibility) — §16 requires the suite to be offline,
    and the README promises it needs "aucune clé API, aucune donnée requise".
    Both were true by convention and false in fact: `data/bronze/` is
    gitignored, so the BVC dividend HTML cache is NOT in the repository, and
    on a fresh clone `test_pipeline.py`'s `silver_pipeline()` calls reached
    `casablanca-bourse.com` for real. The suite passed on developer machines
    purely because the cache existed there — the worst kind of green.

    Blocking at the socket layer catches every client library at once
    (urllib, requests, yfinance, fredapi) rather than stubbing each call
    site and hoping the list stays complete.

    It blocks CONNECTING OUTWARD, not socket creation: FastAPI's TestClient
    and anyio's portal open loopback sockets in-process and are legitimate.
    Loopback is therefore allowed and everything else raises. A test that
    genuinely needs a real transport can request `allow_network`.
    """
    real_connect = socket.socket.connect
    real_create_connection = socket.create_connection

    def _is_loopback(address) -> bool:
        if not isinstance(address, tuple) or not address:
            return True          # AF_UNIX / socketpair — never remote
        host = address[0]
        return host in ("127.0.0.1", "::1", "localhost", "", None)

    def _guarded_connect(self, address, *args, **kwargs):
        if not _is_loopback(address):
            raise NetworkAccessAttempted(
                f"This test tried to connect to {address!r}. The suite must be "
                f"hermetic: stub the fetch (e.g. monkeypatch "
                f"dividends.fetch_issuer_page) or use a synthetic fixture. If a "
                f"real transport is genuinely required, request the "
                f"`allow_network` fixture explicitly."
            )
        return real_connect(self, address, *args, **kwargs)

    def _guarded_create_connection(address, *args, **kwargs):
        if not _is_loopback(address):
            raise NetworkAccessAttempted(
                f"This test tried to connect to {address!r}. See _no_network."
            )
        return real_create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket, "create_connection", _guarded_create_connection)


@pytest.fixture()
def allow_network(monkeypatch):
    """Opt back out of `_no_network` for a test that truly needs a socket."""
    monkeypatch.undo()

ASSETS = ["IAM.CS", "ATW.CS", "CIH.CS", "BCP.CS", "SPY", "QQQ", "EEM", "GLD", "TLT"]
MACRO_SERIES = ["VIX", "US10Y", "DXY", "CREDIT_SPREAD"]


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
        "CREDIT_SPREAD": 2.0 + rng.normal(0, 0.1, 600),
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df
