"""
dividends.py — BVC dividend history, and price-only → total-return conversion.

Addresses: P1, P4 — corrects a systematic bias discovered on 2026-07-25 that
affects every `full_2021` result in the project (see `docs/DIVIDEND_BIAS.md`).

The bug being fixed
-------------------
The committed pipeline mixed two different return definitions:

    ETFs  -> yfinance auto_adjust=True  => TOTAL RETURN (dividends reinvested)
    BVC   -> BVCscrap feature="Value"   => PRICE ONLY   (dividends discarded)

Moroccan bank and telecom yields run 3.5-5.5%/yr, so the four BVC assets were
systematically understated by that much. Critically the bias does NOT cancel
across strategies: `equal_weight` is forced to hold the understated assets
while the optimizers are free to underweight them — and did, earning credit
for avoiding an artefact of our own data handling. Correcting it moves the
project's headline claim from +14.3% to roughly +7%.

Why this module scrapes rather than uses BVCscrap
-------------------------------------------------
`BVCscrap.getDividend` exists but is BROKEN: it targets the legacy
`casablanca-bourse.com/bourseweb/Societe-Cote.aspx` endpoint, which now
307-redirects to the redesigned site (verified 2026-07-25). Rather than depend
on a dead path, this module reads the modern per-issuer page, whose dividend
table is server-rendered into the HTML — no browser or API key needed.

The table gives exactly what a correct adjustment needs: the per-share amount
AND the ex-dividend date ("Date de détachement"), per year. Cross-validated
against stockanalysis.com's independent `dps` series for IAM — the amounts
agree exactly.

Causality
---------
Applying a dividend on its EX-DATE is not lookahead. The ex-date is precisely
the day the price mechanically drops by (approximately) the dividend, so
adding it back on that day reconstructs the total return an actual holder
earned. This is the standard construction, and it is what `auto_adjust=True`
already does for the ETF side — which is the whole point: both sides must
mean the same thing.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger("dividends")

ROOT = Path(__file__).resolve().parents[1]

ISSUER_URL = "https://www.casablanca-bourse.com/fr/live-market/emetteurs/{code}"

# Canonical ticker -> Casablanca Stock Exchange issuer code.
# ATW's code is BCM130843, a legacy artefact of the pre-merger BMCE listing;
# verified 2026-07-25 by matching "ATTIJARIWAFA" on the instruments index.
BVC_ISSUER_CODES = {
    "IAM.CS": "IAM131204",
    "ATW.CS": "BCM130843",
    "CIH.CS": "CIH230667",
    "BCP.CS": "BCP060704",
}

CACHE_DIR = ROOT / "data" / "bronze" / "bvc_dividends"


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.html"


def _ssl_context_with_aia_intermediate(host: str = "www.casablanca-bourse.com"):
    """Build a verifying TLS context that tolerates the server's missing intermediate.

    Verified 2026-07-25: casablanca-bourse.com sends ONLY its leaf certificate
    and omits the Sectigo intermediate that signs it. Python's `ssl` module
    therefore cannot build a chain and raises CERTIFICATE_VERIFY_FAILED — and
    `certifi` does not help, because the missing link is the server's to send,
    not the trust store's to know. Browsers and macOS's system `curl` succeed
    only because they silently fetch the intermediate from the certificate's
    Authority Information Access (AIA) extension.

    This does the same thing explicitly: read the leaf's AIA `CA Issuers` URL,
    download the intermediate, and add it to an otherwise-normal verifying
    context.

    Deliberately NOT `verify=False` / `curl -k`, which would "fix" the error by
    accepting any certificate for a host we ingest financial data from. Also
    deliberately not shelling out to the system `curl`, which works on this
    machine but would fail on Linux/CI with a different CA bundle.

    Falls back to the default context (which will fail loudly, not silently)
    if the intermediate cannot be retrieved.
    """
    import ssl
    import urllib.request

    import certifi

    context = ssl.create_default_context(cafile=certifi.where())
    try:
        # Fetch the leaf without verifying, ONLY to read its AIA extension.
        # Nothing from this connection is trusted: the certificate it yields is
        # then validated by the real, verifying context built below.
        probe = ssl.create_default_context()
        probe.check_hostname = False
        probe.verify_mode = ssl.CERT_NONE
        with ssl.create_connection((host, 443), timeout=15) as raw:
            with probe.wrap_socket(raw, server_hostname=host) as sock:
                leaf_der = sock.getpeercert(binary_form=True)

        leaf_pem = ssl.DER_cert_to_PEM_cert(leaf_der)
        aia = re.search(r"CA Issuers - URI:(http[^\s]+)", _cert_text(leaf_pem))
        if not aia:
            log.warning("no AIA CA-Issuers URL on the leaf certificate for %s", host)
            return context

        with urllib.request.urlopen(aia.group(1), timeout=15) as resp:
            intermediate_der = resp.read()
        context.load_verify_locations(
            cadata=ssl.DER_cert_to_PEM_cert(intermediate_der)
        )
        log.debug("loaded missing intermediate from %s", aia.group(1))
    except Exception as exc:  # noqa: BLE001 — best-effort; the caller still verifies
        log.warning("could not complete the certificate chain via AIA (%s); "
                    "the request will fail loudly if the chain is incomplete", exc)
    return context


def _cert_text(pem: str) -> str:
    """Human-readable dump of a PEM certificate, for reading its AIA extension."""
    import subprocess

    completed = subprocess.run(
        ["openssl", "x509", "-noout", "-text"],
        input=pem, capture_output=True, text=True,
    )
    return completed.stdout


def fetch_issuer_page(ticker: str, force: bool = False, timeout: int = 30) -> str:
    """Fetch (or reuse the Bronze cache of) an issuer's page.

    See `_ssl_context_with_aia_intermediate` for why this needs a custom TLS
    context. Verification stays ON throughout.
    """
    if ticker not in BVC_ISSUER_CODES:
        raise ValueError(
            f"Unknown BVC ticker {ticker!r}. Known: {sorted(BVC_ISSUER_CODES)}."
        )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker)
    if path.exists() and not force:
        return path.read_text(encoding="utf-8")

    import urllib.request

    url = ISSUER_URL.format(code=BVC_ISSUER_CODES[ticker])
    log.info("fetching dividends for %s from %s", ticker, url)
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Portfolio_ML dividend ingest)"}
    )
    with urllib.request.urlopen(
        req, timeout=timeout, context=_ssl_context_with_aia_intermediate()
    ) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    path.write_text(html, encoding="utf-8")
    return html


def parse_dividend_table(html: str, ticker: str) -> pd.DataFrame:
    """Extract (ex_date, amount, kind) rows from an issuer page.

    The table is server-rendered inside the `emetteur_dividendes` section with
    columns: Année | Montant Dividende | Type dividende | Date de détachement
    | Date de paiement. Amounts use a comma decimal separator (French locale).

    Returns an EMPTY frame (never raises) when the section is absent — some
    issuers simply have no dividend history — but logs it, because a silently
    empty dividend series would reintroduce the exact bias this module fixes.
    """
    start = html.find("emetteur_dividendes")
    if start == -1:
        log.warning("%s: no dividend section found on the issuer page", ticker)
        return pd.DataFrame(columns=["ex_date", "ticker", "amount", "kind"])

    segment = html[start:start + 20000]
    text = re.sub(r"<[^>]+>", "|", segment)
    text = re.sub(r"(\s*\|\s*)+", "|", text)

    # year | amount | kind | ex-date | payment-date
    pattern = re.compile(
        r"\|(\d{4})\|([\d\s]*[\d],[\d]{2})\|([^|]{3,30}?)\|(\d{2}/\d{2}/\d{4})\|"
    )
    rows = []
    for year, amount, kind, ex_date in pattern.findall(text):
        value = float(amount.replace(" ", "").replace(",", "."))
        rows.append({
            "ex_date": pd.to_datetime(ex_date, format="%d/%m/%Y"),
            "ticker": ticker,
            "amount": value,
            "kind": kind.strip(),
            "year": int(year),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("%s: dividend section present but no rows parsed", ticker)
        return pd.DataFrame(columns=["ex_date", "ticker", "amount", "kind"])

    df = df.drop_duplicates(subset=["ex_date", "amount"]).sort_values("ex_date")
    log.info("%s: %d dividends parsed (%s -> %s)", ticker, len(df),
             df["ex_date"].min().date(), df["ex_date"].max().date())
    return df.reset_index(drop=True)


def load_bvc_dividends(tickers=None, force: bool = False) -> pd.DataFrame:
    """Tidy (ex_date, ticker, amount, kind) for every requested BVC ticker."""
    tickers = list(tickers or BVC_ISSUER_CODES)
    frames = [
        parse_dividend_table(fetch_issuer_page(t, force=force), t) for t in tickers
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["ex_date", "ticker", "amount", "kind"])
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "ex_date"])


def to_total_return(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    log_returns: bool = True,
) -> pd.DataFrame:
    """Convert price-only series to total return using per-share dividends.

    On each ex-date the holder receives `amount` per share while the price
    drops by roughly that much, so the correct one-period return is

        r_t = log((P_t + D_t) / P_{t-1})

    rather than `log(P_t / P_{t-1})`. Columns with no dividend history pass
    through unchanged — that is correct for the ETFs, whose yfinance series is
    already total return.

    Args:
        prices: Wide price matrix, DatetimeIndex, one column per asset.
        dividends: Tidy frame from `load_bvc_dividends`.
        log_returns: Return log-returns (project convention, §15.1) when True.

    Returns:
        Return matrix aligned to `prices.index[1:]`.
    """
    adjusted = prices.copy().astype("float64")
    dividend_column = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    if not dividends.empty:
        for ticker, group in dividends.groupby("ticker"):
            if ticker not in prices.columns:
                continue
            for ex_date, amount in zip(group["ex_date"], group["amount"]):
                # Snap the ex-date onto the trading calendar: the first
                # session on or after it. A dividend whose ex-date falls
                # outside the price window is dropped, and said so — silent
                # loss here would be the original bug in miniature.
                position = prices.index.searchsorted(ex_date)
                if position >= len(prices.index):
                    log.debug("%s: dividend on %s is after the price window; skipped",
                              ticker, ex_date.date())
                    continue
                dividend_column.iloc[position, dividend_column.columns.get_loc(ticker)] += amount

    import numpy as np

    ratio = (adjusted + dividend_column) / adjusted.shift(1)
    ratio = ratio.dropna(how="all")
    return np.log(ratio).dropna() if log_returns else (ratio - 1.0).dropna()


def dividend_yield_summary(prices: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker realised annual dividend yield — a sanity check on the scrape.

    A parsed amount off by a factor (decimal-separator slips are the classic
    failure) shows up immediately as an implausible yield, so this is worth
    printing whenever the dividend data is refreshed.
    """
    rows = []
    for ticker, group in dividends.groupby("ticker"):
        if ticker not in prices.columns:
            continue
        window = prices[ticker].dropna()
        if window.empty:
            continue
        in_window = group[(group["ex_date"] >= window.index.min())
                          & (group["ex_date"] <= window.index.max())]
        years = max((window.index.max() - window.index.min()).days / 365.25, 1e-9)
        rows.append({
            "ticker": ticker,
            "n_dividends": len(in_window),
            "total_paid": round(float(in_window["amount"].sum()), 2),
            "avg_price": round(float(window.mean()), 2),
            "annual_yield_pct": round(
                float(in_window["amount"].sum() / years / window.mean() * 100), 2),
        })
    return pd.DataFrame(rows)
