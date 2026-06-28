"""
ingest.py — Bronze layer.

Downloads raw adjusted-close prices and macro indicators, then writes
immutable Parquet files to data/bronze/. No transformation happens here.

Addresses: P1, P2, P3 — provides the raw return series from which covariance
estimation, regime detection, and crisis-correlation analysis will operate.

Usage:
    python src/ingest.py
"""

import logging
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf
from fredapi import Fred

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("ingest")

ROOT = Path(__file__).resolve().parents[1]
BRONZE_DIR = ROOT / "data" / "bronze"

# ── Asset universe ───────────────────────────────────────────────────────────

EQUITY_TICKERS = [
    "IAM.CS",  # Maroc Telecom
    "ATW.CS",  # Attijariwafa Bank
    "CIH.CS",  # CIH Bank
    "BCP.CS",  # Banque Centrale Populaire
]

ETF_TICKERS = [
    "SPY",  # S&P 500
    "QQQ",  # Nasdaq 100
    "EEM",  # MSCI Emerging Markets
    "GLD",  # Gold
    "TLT",  # 20+ Year US Treasury
]

ALL_TICKERS = EQUITY_TICKERS + ETF_TICKERS

# ── Macro series (FRED) ──────────────────────────────────────────────────────

FRED_SERIES = {
    "VIX":       "VIXCLS",        # Global fear gauge — key regime signal
    "US10Y":     "DGS10",         # 10-year US Treasury yield
    "DXY":       "DTWEXBGS",      # US Dollar index
    "HY_SPREAD": "BAMLH0A0HYM2",  # High-yield credit spread
}

START_DATE = "2017-01-01"


# ── Price ingestion ──────────────────────────────────────────────────────────

def _download_single(ticker: str, start: str, retries: int = 3) -> pd.Series | None:
    """Download one ticker, retrying on rate-limit errors. Returns a Series or None."""
    import time
    for attempt in range(1, retries + 1):
        try:
            raw = yf.Ticker(ticker).history(start=start, auto_adjust=True)
            if not raw.empty and raw["Close"].notna().any():
                s = raw["Close"].copy()
                s.name = ticker
                s.index = pd.to_datetime(s.index).tz_localize(None)
                return s
        except Exception as exc:
            log.debug("Attempt %d for %s failed: %s", attempt, ticker, exc)

        if attempt < retries:
            wait = 8 * attempt
            log.warning("No data for %s — retrying in %ds (attempt %d/%d)",
                        ticker, wait, attempt, retries)
            time.sleep(wait)

    log.warning("Could not download %s after %d attempts.", ticker, retries)
    return None


def _download_batch(tickers: list[str], start: str) -> pd.DataFrame:
    """Download a list of tickers individually and concatenate into a wide DataFrame."""
    import time
    series_list = []
    for i, ticker in enumerate(tickers):
        s = _download_single(ticker, start)
        if s is not None:
            series_list.append(s)
        if i < len(tickers) - 1:
            time.sleep(2)   # brief pause between tickers to avoid rate limits
    return pd.concat(series_list, axis=1) if series_list else pd.DataFrame()


def ingest_prices(start: str = START_DATE) -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers via yfinance.

    Addresses: P1, P2 — raw return series for covariance estimation and
    regime detection.

    Downloads ETFs first, then BVC equities separately — BVC tickers are
    more prone to rate-limiting because Yahoo Finance serves them from a
    different endpoint.

    Returns:
        Wide DataFrame — DatetimeIndex named "Date", one column per ticker.
    """
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading prices for %d tickers from %s", len(ALL_TICKERS), start)

    # Download ETFs and BVC equities separately to manage rate limits
    etf_prices  = _download_batch(ETF_TICKERS,    start)
    bvc_prices  = _download_batch(EQUITY_TICKERS, start)

    frames = [df for df in [etf_prices, bvc_prices] if not df.empty]
    if not frames:
        raise ValueError("yfinance returned no data. Check tickers and connection.")

    prices = pd.concat(frames, axis=1)
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"

    missing = [t for t in ALL_TICKERS if t not in prices.columns or prices[t].isna().all()]
    if missing:
        log.warning("Tickers with no data from Yahoo Finance: %s", missing)

    # Supplement with manually downloaded BVC data if available
    bvc_csv = BRONZE_DIR / "bvc_prices.csv"
    if bvc_csv.exists():
        log.info("Loading BVC prices from manual CSV: %s", bvc_csv)
        bvc = pd.read_csv(bvc_csv, index_col="Date", parse_dates=True)
        bvc.index.name = "Date"
        for col in bvc.columns:
            prices[col] = bvc[col]
        log.info("BVC columns merged: %s", list(bvc.columns))

    out_path = BRONZE_DIR / "raw_prices.parquet"
    pq.write_table(pa.Table.from_pandas(prices), out_path)
    log.info("Bronze prices written: %d rows × %d columns → %s", *prices.shape, out_path)
    return prices


# ── Macro ingestion ──────────────────────────────────────────────────────────

def ingest_macro(start: str = START_DATE) -> pd.DataFrame:
    """
    Download macro indicators from FRED via fredapi.

    Addresses: P2, P3 — macro indicators drive regime transitions and
    correlation spikes.

    Requires:
        FRED_API_KEY environment variable.

    Returns:
        Wide DataFrame — DatetimeIndex named "Date", one column per series name.
    """
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY environment variable not set. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    fred = Fred(api_key=api_key)
    log.info("Downloading %d FRED series from %s", len(FRED_SERIES), start)

    frames: dict[str, pd.Series] = {}
    for name, series_id in FRED_SERIES.items():
        try:
            frames[name] = fred.get_series(series_id, observation_start=start)
            log.info("  %s (%s): %d observations", name, series_id, len(frames[name]))
        except Exception as exc:
            log.warning("Could not fetch %s (%s): %s", name, series_id, exc)

    if not frames:
        raise RuntimeError("No macro series downloaded. Check API key and connection.")

    macro = pd.DataFrame(frames)
    macro.index = pd.to_datetime(macro.index)
    macro.index.name = "Date"

    out_path = BRONZE_DIR / "raw_macro.parquet"
    pq.write_table(pa.Table.from_pandas(macro), out_path)
    log.info("Bronze macro written: %d rows × %d columns → %s", *macro.shape, out_path)
    return macro


# ── BVC ingestion (BVCscrap / medias24) ─────────────────────────────────────

# Mapping from BVCscrap display names → our canonical column names
BVC_NAME_MAP = {
    "Maroc Telecom": "IAM.CS",
    "Attijariwafa":  "ATW.CS",
    "CIH":           "CIH.CS",
    "BCP":           "BCP.CS",
}

BVC_START = "2021-06-24"   # earliest date available from medias24 / BVCscrap


def ingest_bvc(start: str = BVC_START) -> pd.DataFrame:
    """
    Download BVC (Bourse de Casablanca) closing prices via BVCscrap.

    Addresses: P1, P2, P3 — Moroccan equities are the core local asset class.
    Without them, the portfolio has no direct exposure to the Moroccan economy
    that EURAFRIC / BMCE / RMA Assurance actually operate in.

    Data source: medias24.com via the BVCscrap library.
    Coverage:    June 2021 → today (earliest available from free source).
    Limitation:  Pre-2021 BVC data requires a paid data vendor or manual
                 download from casablanca-bourse.com.

    Args:
        start: Start date. Clamped to BVC_START (2021-06-24) since no earlier
               data is available from the free source.

    Returns:
        Wide DataFrame — DatetimeIndex named "Date", columns IAM.CS / ATW.CS /
        CIH.CS / BCP.CS, values in MAD.
    """
    try:
        import BVCscrap as bvc
    except ImportError:
        raise ImportError("BVCscrap not installed. Run: pip install BVCscrap lxml")

    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    effective_start = max(start, BVC_START)
    log.info("Downloading BVC prices from %s via BVCscrap", effective_start)

    names = list(BVC_NAME_MAP.keys())
    raw = bvc.loadmany(*names, start=effective_start, feature="Value")

    if raw.empty:
        raise RuntimeError("BVCscrap returned no data. Check network connection.")

    # Parse DD/MM/YYYY index → datetime
    raw.index = pd.to_datetime(raw.index, format="%d/%m/%Y", dayfirst=True)
    raw.index.name = "Date"
    raw = raw.sort_index()

    # Rename to canonical ticker symbols
    raw = raw.rename(columns=BVC_NAME_MAP)

    log.info(
        "BVC prices downloaded: %d rows × %d columns (%s → %s)",
        *raw.shape, raw.index.min().date(), raw.index.max().date(),
    )
    log.info("  Coverage note: data starts %s (medias24 free-tier limit)", BVC_START)

    out_path = BRONZE_DIR / "bvc_prices.parquet"
    pq.write_table(pa.Table.from_pandas(raw), out_path)
    log.info("Bronze BVC prices written → %s", out_path)
    return raw


# ── BAM macro ingestion ──────────────────────────────────────────────────────

# Taux directeur decisions scraped from bkam.ma/Politique-monetaire —
# quarterly rate, applied from the decision date forward (step function).
# Source: https://www.bkam.ma/Politique-monetaire/Cadre-strategique/
#         Decision-de-la-politique-monetaire/Historique-des-decisions
_TAUX_DIRECTEUR_DECISIONS = [
    ("2017-01-01", 2.25),  # baseline entry for series start
    ("2020-03-17", 2.00),
    ("2020-06-18", 1.50),
    ("2022-06-21", 1.50),
    ("2022-09-27", 2.00),
    ("2022-12-22", 2.50),
    ("2023-03-23", 3.00),
    ("2024-03-21", 3.00),
    ("2024-09-26", 2.75),
    ("2024-12-19", 2.50),
    ("2025-03-20", 2.25),
    ("2026-03-19", 2.25),
]


def ingest_bam_macro(start: str = START_DATE) -> pd.DataFrame:
    """
    Download and assemble Bank Al-Maghrib macro indicators.

    Addresses: P2, P3 — Moroccan monetary policy and currency dynamics are
    the primary local macro drivers for BVC equities and for any portfolio
    held by a Moroccan institution like EURAFRIC / RMA Assurance.

    Series included:
      - USDMAD : USD / Moroccan Dirham daily rate (Yahoo Finance)
      - EURMAD : EUR / Moroccan Dirham daily rate (Yahoo Finance)
      - TAUX_DIR : Bank Al-Maghrib policy rate — step function from
                   quarterly decision history scraped from bkam.ma

    Returns:
        Wide DataFrame — DatetimeIndex named "Date", one column per series.
    """
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading BAM macro indicators from %s", start)

    # ── MAD exchange rates (Yahoo Finance) ───────────────────────────────────
    raw_fx = yf.download(
        ["USDMAD=X", "EURMAD=X"],
        start=start,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw_fx.columns, pd.MultiIndex):
        fx = raw_fx["Close"].rename(columns={"USDMAD=X": "USDMAD", "EURMAD=X": "EURMAD"})
    else:
        fx = raw_fx[["Close"]].rename(columns={"Close": "USDMAD"})

    fx.index = pd.to_datetime(fx.index).tz_localize(None)
    fx.index.name = "Date"
    log.info("  USDMAD: %d rows | EURMAD: %d rows",
             fx["USDMAD"].notna().sum(), fx["EURMAD"].notna().sum())

    # ── Taux directeur (step function from BAM decisions) ────────────────────
    decisions = pd.DataFrame(_TAUX_DIRECTEUR_DECISIONS, columns=["date", "rate"])
    decisions["date"] = pd.to_datetime(decisions["date"])
    decisions = decisions.set_index("date").sort_index()

    # Build a daily series: forward-fill rate from each decision date
    full_index = pd.bdate_range(start=start, end=pd.Timestamp.today())
    taux = decisions["rate"].reindex(full_index, method="ffill")
    taux.name = "TAUX_DIR"
    taux.index.name = "Date"

    # ── Merge ─────────────────────────────────────────────────────────────────
    bam = fx.join(taux, how="outer")
    bam = bam[bam.index >= pd.Timestamp(start)]
    bam.index.name = "Date"

    out_path = BRONZE_DIR / "raw_bam_macro.parquet"
    pq.write_table(pa.Table.from_pandas(bam), out_path)
    log.info("Bronze BAM macro written: %d rows × %d columns → %s", *bam.shape, out_path)
    return bam


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ingest_prices()
    ingest_macro()
    ingest_bvc()
    ingest_bam_macro()
