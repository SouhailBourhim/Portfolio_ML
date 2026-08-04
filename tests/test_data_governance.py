"""test_data_governance.py — the claims in docs/DATA_GOVERNANCE.md, as checks.

Two statements in that document do real work, and both are the kind that stay
true only until someone extends the system without noticing:

1. **No personal data is processed.** The Law 09-08 and GDPR positions rest
   entirely on this. If a client identifier ever entered a Gold artifact, the
   document would be wrong and nothing would say so.
2. **API logs record no caller content.** The most likely way this project
   acquires data-protection obligations by accident is a log line.

Cheap checks, deliberately. They cannot prove absence of personal data; they
fire on the shapes it would actually arrive in.
"""

from __future__ import annotations

import ast
import glob
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
API_DIR = ROOT / "src" / "api"

# Substrings that would appear in a column naming a person, an account, or a
# client. "asset" and "strategy" are the vocabulary this project should have.
PERSONAL_DATA_MARKERS = (
    "name", "email", "phone", "address", "client", "customer", "account",
    "ssn", "passport", "birth", "gender", "user", "person", "holder", "iban",
)

# Things a log call must never receive.
FORBIDDEN_LOG_SOURCES = (
    "request.body", "request.headers", "request.query_params", "request.cookies",
    "request.client", "await request", ".json()",
)


class TestNoPersonalDataInArtifacts:
    """The premise the entire legal section rests on."""

    def test_no_gold_column_names_a_person_account_or_client(self):
        parquets = sorted(glob.glob(str(GOLD / "*.parquet")))
        if not parquets:
            pytest.skip("Gold artifacts not present — run the pipeline or `dvc pull`.")

        offenders = []
        for path in parquets:
            for column in pd.read_parquet(path).columns:
                lowered = str(column).lower()
                for marker in PERSONAL_DATA_MARKERS:
                    if marker in lowered:
                        offenders.append(f"{Path(path).name}:{column} (~{marker})")
        assert not offenders, (
            "Gold artifacts contain column(s) suggesting personal or client data: "
            f"{offenders}. docs/DATA_GOVERNANCE.md states that none is processed, and "
            "the Law 09-08 / GDPR positions rest on that. A fresh assessment is "
            "required BEFORE such data enters the system."
        )

    def test_the_asset_universe_is_instruments_not_people(self):
        path = GOLD / "log_returns.parquet"
        if not path.is_file():
            pytest.skip("log_returns.parquet not present.")
        columns = [str(c) for c in pd.read_parquet(path).columns]
        assert columns, "empty universe"
        # Tickers: uppercase alphanumerics, optionally suffixed (IAM.CS, SPY).
        for column in columns:
            assert column.replace(".", "").replace("-", "").isalnum(), (
                f"{column!r} does not look like an instrument ticker."
            )


class TestApiLogsNoCallerContent:
    """A log line is the likeliest accidental route to a data-protection duty."""

    def _api_sources(self) -> list[Path]:
        sources = sorted(API_DIR.glob("*.py"))
        if not sources:
            pytest.skip("API sources not present.")
        return sources

    def test_no_log_call_receives_a_request_body_header_or_raw_query(self):
        offenders = []
        for path in self._api_sources():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                target = ast.unparse(node.func)
                if not (target.startswith(("log.", "logger.", "logging."))
                        or target.endswith(("print",))):
                    continue
                rendered = ast.unparse(node)
                for forbidden in FORBIDDEN_LOG_SOURCES:
                    if forbidden in rendered:
                        offenders.append(f"{path.name}: {rendered[:90]}")
        assert not offenders, (
            f"API logging touches caller-supplied content: {offenders}. "
            "docs/DATA_GOVERNANCE.md §4 forbids request bodies, headers, cookies, "
            "client addresses and raw query values in logs, in every environment."
        )

    def test_the_api_does_not_import_a_request_object_it_could_log(self):
        """Defence in depth: the current endpoints take validated params only.

        Accepting a raw `Request` is not wrong by itself, but it is the door
        through which caller content reaches a log, so its arrival should be a
        deliberate decision rather than a quiet import.
        """
        for path in self._api_sources():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                for alias in node.names
            }
            assert "Request" not in imported, (
                f"{path.name} imports fastapi.Request. If that is intended, extend "
                f"docs/DATA_GOVERNANCE.md §4 and this test deliberately."
            )


class TestCredentialHygiene:
    """The controls that keep secrets out of Git and out of the image."""

    def test_dockerignore_excludes_every_credential_file(self):
        path = ROOT / ".dockerignore"
        if not path.is_file():
            pytest.skip(".dockerignore not present.")
        content = path.read_text(encoding="utf-8")
        for secret in (".env", ".dvc/config.local"):
            assert secret in content, (
                f"{secret} is not excluded from the Docker build context. Gitignoring "
                f"a secret does NOT keep it out of an image — COPY reads the "
                f"filesystem, not the index."
            )

    def test_no_source_file_contains_a_literal_api_key_assignment(self):
        offenders = []
        for path in sorted((ROOT / "src").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                targets = " ".join(ast.unparse(t).lower() for t in node.targets)
                if not any(k in targets for k in ("api_key", "secret", "token", "password")):
                    continue
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    if node.value.value:
                        offenders.append(f"{path.name}:{node.lineno}")
        assert not offenders, (
            f"Literal credential assignment(s) found: {offenders}. Secrets come from "
            f"the environment (FRED_API_KEY) or from .dvc/config.local, never source."
        )


class TestCurrencyExposureTravelsWithEveryPortfolioFigure:
    """An unhedged FX exposure is economic risk, not a formality.

    Every portfolio number this project publishes embeds an unhedged USD/MAD
    position — BVC assets are MAD-denominated, the ETFs USD. It previously
    appeared in exactly one place: a bullet near the bottom of one dashboard
    page. These tests keep it beside the headline on every surface, so a reader
    who sees a Sharpe also sees what it is exposed to.
    """

    def test_the_readme_states_it_where_the_scope_is_set(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "USD/MAD" in readme
        assert "risque économique matériel" in readme, (
            "the README must call it a material economic risk, not a footnote"
        )

    def test_the_report_executive_summary_states_it(self):
        for name in ("resume.tex",):
            path = ROOT / "docs" / "rapport" / "frontmatter" / name
            if not path.is_file():
                pytest.skip(f"{name} not present.")
            text = path.read_text(encoding="utf-8")
            assert "USD/MAD" in text, f"{name} omits the currency exposure"

    def test_the_api_ships_it_with_the_headline_comparison(self):
        main_py = (ROOT / "src" / "api" / "main.py").read_text(encoding="utf-8")
        assert "CURRENCY_CAVEAT" in main_py
        assert '"currency_exposure": CURRENCY_CAVEAT' in main_py, (
            "/compare must return the exposure alongside the lift, so quoting the "
            "lift without it requires actively discarding a field"
        )

    def test_the_published_allocation_contract_requires_it(self):
        contracts = (ROOT / "src" / "api" / "contracts.py").read_text(encoding="utf-8")
        assert "currency_exposure: str" in contracts, (
            "an allocation quoted without the exposure omits a material risk; the "
            "field must be required, not optional"
        )

    def test_the_stakeholder_page_shows_it_above_the_results(self):
        page = ROOT / "dashboard" / "pages" / "1_Resultats_recherche.py"
        if not page.is_file():
            pytest.skip("dashboard page not present.")
        source = page.read_text(encoding="utf-8")
        headline = source.index("st.title(")
        first_metric = source.find("st.metric(")
        exposure = source.index("Exposition de change USD/MAD non couverte")
        assert headline < exposure, "the exposure must follow the title"
        if first_metric != -1:
            assert exposure < first_metric, (
                "the exposure must appear BEFORE the first performance figure, not "
                "in a caveat list below it"
            )
