"""
fundamentals.py — Point-in-time fundamental features for BVC equities.

Addresses: P4 — the deep-Morocco experiment (2005-2024, 12 stocks, 56k pooled
rows) ruled out "more price history" as the missing ingredient: purged-CV IC
improved 2-4x, portfolio edge did not. That result explicitly redirected the
alpha search to a different data class — fundamentals — which this module
provides as a new causal feature block for F7.

Source: stockanalysis.com (S&P Global-sourced, free tier, robots.txt permits
crawling for User-agent: *). The site embeds the financials into the initial
HTML as a JavaScript object literal (SvelteKit hydration payload), so a
browser is NOT required — a small tokenizer extracts the block and reshapes
it into a tidy (period_end, ticker, metric, value) frame.

The load-bearing decision here is CAUSAL, not scraping:

  * The site exposes each row's fiscal PERIOD-END date but not its actual
    FILING date. Feeding period-end-dated values into a t-indexed model
    would be a lookahead leak (a 2023-06-30 semi-annual is not visible on
    2023-07-01; Moroccan issuers file 60-90 days late by AMMC rule).
  * `apply_publication_lag()` therefore shifts every period-end forward by
    `publication_lag_days` (default 90) to produce an `available_from` date,
    the earliest business day the model may see that report.
  * `build_point_in_time_panel()` forward-fills each ticker's fundamentals
    from `available_from` onward, so a query at date t returns the most
    recently AVAILABLE report at t, never a future one.
  * `test_fundamentals.py`'s future-corruption gate proves this end-to-end
    the same way `test_phase3_integration.py` did for market features:
    changing a future period's value cannot alter a past feature row.

The Bronze/Gold discipline of the repo is preserved. Raw HTML pages are
cached under `data/bronze/fundamentals/{TICKER}_{page}.html` — immutable,
byte-for-byte reproducible offline runs (§7 medallion pattern). Gold output
is a wide (Date × ASSET × FIELD) parquet.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

log = logging.getLogger("fundamentals")

USER_AGENT = "Mozilla/5.0 (Portfolio_ML fundamentals scout)"
BASE_URL = "https://stockanalysis.com/quote/cbse/{ticker}/financials/{page_suffix}"
# stockanalysis.com's URL scheme: "" for annual income statement, "?p=quarterly"
# for the semi-annual view Moroccan issuers actually report (H1/H2, not Q1-Q4).
# "ratios/" is annual ratios; add "?p=quarterly" for the semi-annual ratios cut.
_PAGES = {
    "income_semi": "?p=quarterly",
    "ratios_semi": "ratios/?p=quarterly",
}


@dataclass(frozen=True)
class FundamentalsManifest:
    """Small stamp written next to the Gold parquet — audit trail, not code."""
    tickers: tuple[str, ...]
    ratio_fields: tuple[str, ...]
    publication_lag_days: int
    period_ends: dict[str, list[str]]         # ticker → sorted list of period_ends seen
    output_path: str

    def to_dict(self) -> dict:
        return {
            "tickers": list(self.tickers),
            "ratio_fields": list(self.ratio_fields),
            "publication_lag_days": self.publication_lag_days,
            "period_ends": {k: sorted(v) for k, v in self.period_ends.items()},
            "output_path": self.output_path,
        }


# ─── Parsing (pure, deterministic, no I/O) ──────────────────────────────────

def _js_object_to_dict(text: str) -> dict:
    """Convert a JavaScript object literal to a Python dict via JSON.

    stockanalysis.com embeds the financials as an inline JS object with
    unquoted identifier keys, leading-dot numeric literals (`-.0049`), and
    JS specials (`void 0`, `undefined`, `NaN`). All are legal JS but not
    legal JSON, so each is normalized to the JSON equivalent, then parsed.
    """
    # 1. Quote bare identifier keys: {foo: → {"foo":  and  ,bar: → ,"bar":
    quoted = re.sub(
        r'([{,])\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*:',
        r'\1"\2":',
        text,
    )
    # 2. Leading-dot numeric literals: -.00049 → -0.00049, .5 → 0.5
    quoted = re.sub(r'([\[,\s:])-\.', r'\g<1>-0.', quoted)
    quoted = re.sub(r'([\[,\s:])\.', r'\g<1>0.', quoted)
    # 3. JS specials that mean "no value here"
    quoted = re.sub(r'\bvoid\s+0\b', 'null', quoted)
    quoted = re.sub(r'\bundefined\b', 'null', quoted)
    quoted = re.sub(r'\bNaN\b', 'null', quoted)
    return json.loads(quoted)


def extract_financial_data(html: str, key: str = "financialData") -> dict | None:
    """Extract the `{key}:{ ... }` block from an HTML page's inline JS payload.

    Brace-walks the payload while respecting string literals — a naive
    `re.search(r'\\{.*?\\}')` would clip on any embedded `}` inside a quoted
    field. Returns None if the key isn't found (a page for a delisted ticker
    or a stockanalysis.com structural change would surface here rather than
    silently producing empty features).
    """
    match = re.search(rf'\b{re.escape(key)}:\{{', html)
    if not match:
        return None

    start = match.end() - 1
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(html)):
        c = html[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return _js_object_to_dict(html[start:i + 1])
    return None


def financial_data_to_frame(
    fd: dict,
    ticker: str,
    fields: Iterable[str],
) -> pd.DataFrame:
    """Reshape a parsed financialData dict into tidy (period_end, ticker, metric, value).

    The site's `datekey` array occasionally holds a "TTM" placeholder as its
    first element (trailing twelve months, not a real fiscal period). We drop
    it here, upstream of every user — no downstream code should have to know
    the source ever emitted it.
    """
    dates = fd.get("datekey", [])
    if not dates:
        return pd.DataFrame(columns=["period_end", "ticker", "metric", "value"])

    rows: list[dict] = []
    for field in fields:
        values = fd.get(field)
        if values is None or len(values) != len(dates):
            continue
        for date_str, val in zip(dates, values):
            if date_str == "TTM":
                continue
            try:
                pe = pd.Timestamp(date_str)
            except (ValueError, TypeError):
                continue
            rows.append({
                "period_end": pe,
                "ticker": ticker,
                "metric": field,
                "value": val,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["period_end", "ticker", "metric"])
    return df


# ─── I/O (cached, offline-friendly) ─────────────────────────────────────────

def _cache_path(cache_dir: Path, ticker: str, page: str) -> Path:
    return cache_dir / f"{ticker}_{page}.html"


def fetch_html(
    ticker: str,
    page: str,
    cache_dir: Path,
    force: bool = False,
    timeout: int = 30,
) -> str:
    """Fetch a stockanalysis.com page for a ticker, honouring the Bronze cache.

    Bronze rule (§15.5): raw as-received data is immutable and reused across
    runs. `force=True` re-downloads; the default silently reuses the cached
    HTML, so a full experiment can run offline once the cache is warm.
    """
    if page not in _PAGES:
        raise ValueError(f"Unknown page: {page!r} (known: {list(_PAGES)})")

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, ticker, page)
    if path.exists() and not force:
        return path.read_text(encoding="utf-8")

    url = BASE_URL.format(ticker=ticker, page_suffix=_PAGES[page])
    log.info("fetch %s (%s) from %s", ticker, page, url)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} for {ticker}/{page}: {e.reason}") from e

    path.write_text(data, encoding="utf-8")
    return data


def scrape_ticker(
    ticker: str,
    cache_dir: Path,
    ratio_fields: Iterable[str],
    force: bool = False,
) -> pd.DataFrame:
    """Scrape one ticker's semi-annual income + ratios, return a tidy frame.

    Raises RuntimeError if either page has no `financialData` block — a
    silent empty-frame return would let a bad ticker sneak into the panel
    as all-NaN, violating §15.13 "silent data loss is a bug".
    """
    ratios_html = fetch_html(ticker, "ratios_semi", cache_dir, force=force)
    ratios_fd = extract_financial_data(ratios_html)
    if ratios_fd is None:
        raise RuntimeError(
            f"stockanalysis.com returned a page for {ticker} with no "
            "financialData block — the ticker may be delisted or the site's "
            "HTML structure may have changed."
        )
    return financial_data_to_frame(ratios_fd, ticker, ratio_fields)


def fetch_all_fundamentals(
    tickers: Iterable[str],
    cache_dir: Path,
    ratio_fields: Iterable[str],
    force: bool = False,
) -> pd.DataFrame:
    """Scrape every ticker; return one long tidy frame keyed by period + ticker + metric."""
    frames = [scrape_ticker(t, cache_dir, ratio_fields, force=force) for t in tickers]
    tidy = pd.concat(frames, ignore_index=True)
    return tidy.sort_values(["ticker", "period_end", "metric"]).reset_index(drop=True)


# ─── Causal transform (the load-bearing correctness step) ──────────────────

def apply_publication_lag(tidy: pd.DataFrame, publication_lag_days: int) -> pd.DataFrame:
    """Add an `available_from` column = period_end + lag business days.

    This is the causal seam: no downstream consumer sees a period_end
    directly. If publication_lag_days is negative, raise — that would be
    a lookahead leak, structural, not a hyperparameter judgement.
    """
    if publication_lag_days < 0:
        raise ValueError(
            f"publication_lag_days must be ≥ 0 (negative = lookahead), "
            f"got {publication_lag_days}."
        )
    out = tidy.copy()
    # Business-day offset — matches the same trading calendar prices live on.
    offset = pd.tseries.offsets.BDay(publication_lag_days)
    out["available_from"] = out["period_end"] + offset
    return out


def build_point_in_time_panel(
    tidy_with_available_from: pd.DataFrame,
    date_index: pd.DatetimeIndex,
    tickers: Iterable[str],
) -> pd.DataFrame:
    """Turn tidy fundamentals into a wide (Date × TICKER__METRIC) panel.

    For each ticker and metric, the value on date t is the most recent
    period whose `available_from` is ≤ t — a strict-inequality forward-fill
    keyed on `available_from`, not `period_end`. NaN before the first
    `available_from`. This IS the guarantee proven by the integration test.

    Columns are named `{TICKER}__FUND_{METRIC}` — same naming style the F7
    pooled panel already uses for its per-asset price features
    (`{TICKER}__RET_5D`, etc.), so the melt-to-panel step in `ml_signals`
    can pick them up with the same regex.
    """
    tickers = list(tickers)
    if "available_from" not in tidy_with_available_from.columns:
        raise ValueError(
            "Panel construction expects an 'available_from' column — call "
            "apply_publication_lag() first."
        )

    idx = pd.DatetimeIndex(date_index).sort_values()
    metrics = sorted(tidy_with_available_from["metric"].unique())
    cols = [f"{t}__FUND_{m}" for t in tickers for m in metrics]
    out = pd.DataFrame(index=idx, columns=cols, dtype="float64")

    grouped = tidy_with_available_from.groupby(["ticker", "metric"])
    for (ticker, metric), grp in grouped:
        if ticker not in tickers:
            continue
        col = f"{ticker}__FUND_{metric}"
        # Sort by available_from; the LAST report whose available_from ≤ t wins.
        events = grp.sort_values("available_from")[["available_from", "value"]]
        # Reindex events onto the trading calendar via searchsorted for O(n log n).
        af = events["available_from"].values
        val = events["value"].values
        # For each date d in idx: position of first available_from > d, minus 1
        positions = pd.Series(af).searchsorted(idx.values, side="right") - 1
        col_series = pd.Series(
            [val[p] if p >= 0 else float("nan") for p in positions],
            index=idx,
        )
        out[col] = col_series

    return out


# ─── Orchestration ─────────────────────────────────────────────────────────

def run_fundamentals_features(
    config: dict,
    date_index: pd.DatetimeIndex,
    project_root: Path,
    force_refetch: bool = False,
) -> tuple[pd.DataFrame, FundamentalsManifest]:
    """End-to-end Gold builder: scrape → causal-lag → point-in-time panel + manifest.

    `date_index` is the trading calendar the returned panel will be indexed
    on — pass in the union of the universes' return indices so the panel
    aligns 1:1 with them. Returns (panel, manifest); persistence is the
    caller's job (parquet + json), mirroring `ml_features.run_phase3`'s
    return-then-write pattern.
    """
    cache_dir = project_root / config["cache_dir"]
    tickers = list(config["tickers"])
    ratio_fields = list(config["ratio_fields"])
    lag = int(config["publication_lag_days"])

    tidy = fetch_all_fundamentals(tickers, cache_dir, ratio_fields, force=force_refetch)
    tidy = apply_publication_lag(tidy, lag)
    panel = build_point_in_time_panel(tidy, date_index, tickers)

    period_ends = {
        t: tidy[tidy["ticker"] == t]["period_end"].dt.strftime("%Y-%m-%d").unique().tolist()
        for t in tickers
    }
    manifest = FundamentalsManifest(
        tickers=tuple(tickers),
        ratio_fields=tuple(ratio_fields),
        publication_lag_days=lag,
        period_ends=period_ends,
        output_path=config["output_path"],
    )
    log.info(
        "fundamentals panel: %d dates × %d columns, %d ticker-periods scraped",
        panel.shape[0], panel.shape[1], len(tidy) // max(1, len(ratio_fields)),
    )
    return panel, manifest


if __name__ == "__main__":  # pragma: no cover — thin CLI, unit-tested via calls
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

    from utils import load_params

    parser = argparse.ArgumentParser(description="Build the fundamentals Gold layer.")
    parser.add_argument("--force-refetch", action="store_true",
                        help="Re-download HTML instead of reusing the Bronze cache.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    params = load_params()
    cfg = params["fundamentals"]

    # Build a date index by taking the union of both universes' log-returns
    # indices — same "cover both universes" convention every prior Gold builder
    # uses (ml_features.run_phase3, etc.).
    ret_full = pd.read_parquet(root / "data/gold/log_returns.parquet")
    ret_etf = pd.read_parquet(root / "data/gold/log_returns_etf.parquet")
    idx = ret_full.index.union(ret_etf.index).sort_values()

    panel, manifest = run_fundamentals_features(cfg, idx, root, force_refetch=args.force_refetch)

    out_path = root / cfg["output_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path)
    (root / cfg["manifest_path"]).write_text(json.dumps(manifest.to_dict(), indent=2))
    log.info("wrote %s (%d rows × %d cols) and manifest", out_path,
             panel.shape[0], panel.shape[1])
