"""
test_release_facts.py — every public surface must make the SAME claims.

Not "contains some similar numbers": the README and the report are checked
against the exact strings `release_facts.py` generates. A near-match is how
surfaces drift — §17.11 records the 2026-08-02 reframing being applied by hand
to six surfaces plus a binary deliverable plus a JSON field the API re-served,
with nothing in the suite able to tell they had come apart, because
`test_artifact_consistency.py` compares NUMBERS and not WORDING.

Two rules carry most of the weight here:

  * There is NO global base currency. `full_2021` is MAD (official Bank
    Al-Maghrib reference rate, unhedged); `etf_2017` is single-currency USD and
    was never converted. A surface that states one exposure for both is wrong
    for each, which is precisely what the API's old shared CURRENCY_CAVEAT did.
  * `regime_conditional` is the PRE-SPECIFIED primary comparator for White/SPA.
    It is not a reference system and not a recommended candidate. Relabelling it
    now that the observed difference has changed sign would be the exact
    multiple-testing abuse the correction exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from release_facts import load_facts, numeraire_for, statement_list

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def facts():
    if not (ROOT / "data" / "gold" / "dashboard_showcase.json").is_file():
        pytest.skip("Gold artifacts absent — run `dvc pull`.")
    return load_facts(ROOT)


@pytest.fixture(scope="module")
def statements():
    if not (ROOT / "data" / "gold" / "dashboard_showcase.json").is_file():
        pytest.skip("Gold artifacts absent — run `dvc pull`.")
    return statement_list(ROOT)


class TestNumbersAreDerivedNotTyped:
    def test_the_point_difference_matches_the_showcase_artifact(self, facts):
        showcase = json.loads(
            (ROOT / "data" / "gold" / "dashboard_showcase.json").read_text()
        )["universes"]
        for u in ("full_2021", "etf_2017"):
            assert facts["universes"][u]["point_difference_pct"] == \
                showcase[u]["headline_lift_pct"]

    def test_the_numeraire_matches_the_currency_manifest(self, facts):
        manifest = json.loads(
            (ROOT / "data" / "gold" / "currency_manifest.json").read_text()
        )["universes"]
        for u in ("full_2021", "etf_2017"):
            assert facts["universes"][u]["base_currency"] == manifest[u]["base_currency"]

    def test_no_sharpe_literal_is_typed_into_the_facts_module(self):
        """Same rule as tests/test_run_dashboard_data.py: a released claim must
        not be able to go stale independently of the artifact behind it."""
        import ast

        src = (ROOT / "src" / "release_facts.py").read_text(encoding="utf-8")
        offenders = [
            node.value for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.Constant) and isinstance(node.value, float)
            and 0.3 < abs(node.value) < 5.0
        ]
        assert not offenders, f"Sharpe-like literals in release_facts.py: {offenders}"


class TestEveryUniverseCarriesItsOwnNumeraire:
    """There is no global base currency, and no surface may imply one."""

    def test_full_2021_is_mad_unhedged(self, facts):
        n = numeraire_for("full_2021", ROOT)
        assert n["base_currency"] == "MAD"
        assert n["converted"] is True
        assert n["hedge_status"] == "unhedged"
        assert "Bank Al-Maghrib" in n["statement"]
        assert "non couvert" in n["statement"]

    def test_etf_2017_is_usd_and_unconverted(self, facts):
        n = numeraire_for("etf_2017", ROOT)
        assert n["base_currency"] == "USD"
        assert n["converted"] is False
        assert "inchangé" in n["statement"]

    def test_an_unknown_universe_raises_rather_than_defaulting(self, facts):
        """A wrong numéraire is worse than a missing one."""
        with pytest.raises(KeyError):
            numeraire_for("nonexistent", ROOT)

    def test_the_two_universes_do_not_share_a_currency_string(self, facts):
        a = numeraire_for("full_2021", ROOT)["statement"]
        b = numeraire_for("etf_2017", ROOT)["statement"]
        assert a != b, (
            "a single shared caveat describes a mixed exposure for both universes "
            "and is therefore wrong for each — the defect the correction removed"
        )

    def test_the_api_has_no_global_currency_constant(self):
        src = (ROOT / "src" / "api" / "main.py").read_text(encoding="utf-8")
        assert "CURRENCY_CAVEAT = (" not in src, (
            "the API must not carry a single global currency string; every "
            "universe-scoped response carries the numéraire of its own universe"
        )
        assert "numeraire_for" in src

    def test_the_allocation_contract_requires_a_base_currency(self):
        src = (ROOT / "src" / "api" / "contracts.py").read_text(encoding="utf-8")
        assert "base_currency: str" in src
        assert "hedge_status: str" in src
        for field in ("base_currency", "hedge_status"):
            assert f"{field}: str | None" not in src, (
                f"{field} must be REQUIRED; optional fields get dropped by consumers"
            )


class TestSurfacesQuoteTheGeneratedStringsExactly:
    """String equality, not number-spotting. A paraphrase is drift."""

    def _text(self, *rel: str) -> str:
        return "\n".join(
            (ROOT / r).read_text(encoding="utf-8") for r in rel if (ROOT / r).is_file()
        )

    def test_the_readme_carries_every_canonical_statement(self, statements):
        readme = self._text("README.md")
        missing = [s for s in statements if s not in readme]
        assert not missing, (
            f"{len(missing)} canonical statement(s) absent from README.md, verbatim. "
            f"Regenerate the release-facts block rather than paraphrasing.\n"
            f"First missing: {missing[0][:160]}…"
        )

    def test_the_report_carries_every_canonical_statement(self, statements):
        """The TeX target holds LaTeX-ESCAPED forms (``**x**`` becomes
        ``\\textbf{x}``), so the comparison runs through the same escaper the
        builder uses. Comparing raw markdown here would fail for a formatting
        reason and teach the next reader to loosen the check."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "brf", ROOT / "scripts" / "build_release_facts.py")
        brf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(brf)

        report = self._text("docs/rapport/chapters/faits_publies.tex")
        missing = [s for s in statements if brf._tex_escape(s) not in report]
        assert not missing, (
            f"{len(missing)} canonical statement(s) absent from the report, verbatim. "
            f"First missing: {missing[0][:160]}…"
        )

    def test_the_dashboard_renders_them_from_the_module(self):
        page = (ROOT / "dashboard" / "pages" / "1_Resultats_recherche.py").read_text(
            encoding="utf-8"
        )
        assert "from release_facts import" in page
        assert 'FACTS["statements"]' in page


class TestRegimeConditionalIsDescribedAsAComparator:
    """It is the pre-specified White/SPA benchmark — not a reference system, not
    a recommended candidate. Its role was fixed before the results were seen."""

    def test_the_multiple_testing_benchmark_is_unchanged(self, facts):
        assert facts["multiple_testing"]["primary_benchmark"] == "regime_conditional"

    def test_the_benchmark_note_says_it_was_not_rewritten_after_the_result(self, facts):
        note = facts["statements"]["primary_benchmark"]
        assert "pré-spécifié" in note
        assert "n'est pas réécrit" in note

    def test_no_surface_recommends_or_deploys_a_strategy(self, statements):
        banned = ("stratégie recommandée", "nous recommandons", "à déployer",
                  "valeur ajoutée établie", "système de qualité production")
        surfaces = [
            "README.md",
            "dashboard/pages/1_Resultats_recherche.py",
            "dashboard/pages/2_Explorateur_strategies.py",
            "src/api/main.py",
        ]
        # The canonical statements legitimately contain these phrases as
        # NEGATIONS ("ni presentee comme une valeur ajoutee etablie"). Strip the
        # generated block first so the scan tests the hand-written prose around
        # it, which is where a recommendation would actually creep back in.
        offenders = []
        for rel in surfaces:
            path = ROOT / rel
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            if "<!-- BEGIN RELEASE FACTS" in text and "<!-- END RELEASE FACTS -->" in text:
                head, rest = text.split("<!-- BEGIN RELEASE FACTS", 1)
                text = head + rest.split("<!-- END RELEASE FACTS -->", 1)[1]
            for canonical in statements:
                text = text.replace(canonical, "")
            offenders += [f"{rel}: {p}" for p in banned if p in text]
        assert not offenders, f"recommendation framing found: {offenders}"

    def test_the_model_cards_call_it_a_comparator(self):
        card = ROOT / "docs" / "MODEL_CARD_REGIME_CONDITIONAL.md"
        if not card.is_file():
            pytest.skip("model card not generated yet")
        text = card.read_text(encoding="utf-8")
        assert "comparateur primaire pré-spécifié" in text or \
               "pre-specified primary comparator" in text, (
            "the card must describe regime_conditional as the pre-specified "
            "White/SPA comparator, not as a reference system or recommended candidate"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Semantic checks — INDEPENDENT of the generator.
#
# The blind spot these close: every test above compares surfaces against the
# strings release_facts.py produces. If the GENERATOR ever emitted a wrong
# claim, README, report, dashboard and API would all be wrong *and perfectly
# synchronised*, and every equality test would pass.
#
# So these assert what the statements MEAN, not that they match each other. In
# particular the banned-phrase scanner is allowed to skip the generated block
# ONLY because the block is checked here: without this, deleting a single "ni"
# would turn a negation into an assertion invisibly.
# ─────────────────────────────────────────────────────────────────────────────

NEGATORS = ("aucun", "aucune", "ni ", "n'est", "ne sont", "pas ", "non ", "n'établit")


def _negated(text: str, phrase: str, window: int = 90) -> bool:
    """True if EVERY occurrence of `phrase` sits inside a negation."""
    lowered = text.lower()
    needle = phrase.lower()
    start = 0
    found = False
    while (idx := lowered.find(needle, start)) != -1:
        found = True
        context = lowered[max(0, idx - window): idx + len(needle)]
        if not any(n in context for n in NEGATORS):
            return False
        start = idx + len(needle)
    return found


class TestTheCanonicalFactsSayTheRightThing:
    """What the statements MEAN, checked without reference to any surface."""

    def test_the_value_add_phrase_appears_only_under_negation(self, statements):
        """The exact failure mode the scanner's block-skip would otherwise hide:
        drop the "ni" from statement 9 and it becomes a claim of established
        added value, identical in every surface and invisible to equality tests."""
        joined = "\n".join(statements)
        assert _negated(joined, "valeur ajoutée établie"), (
            "'valeur ajoutée établie' must appear only inside a negation"
        )

    def test_no_statement_recommends_or_deploys(self, statements):
        joined = "\n".join(statements)
        for phrase in ("recommandée", "déployée", "recommandé", "déployé"):
            if phrase.lower() in joined.lower():
                assert _negated(joined, phrase), (
                    f"'{phrase}' appears without a negation in the canonical facts"
                )

    def test_no_statement_claims_superiority_or_equivalence(self, statements):
        """Neither direction. AGENTS.md §5.2: never 'superior' or 'equivalent'
        without a paired test, and no paired test of this difference is shown."""
        joined = "\n".join(statements)
        for phrase in ("supériorité", "surperformance", "équivalent", "équivalence"):
            if phrase.lower() in joined.lower():
                assert _negated(joined, phrase), (
                    f"'{phrase}' is asserted rather than denied in the canonical facts"
                )

    def test_a_paired_test_is_explicitly_declared_absent(self, statements):
        joined = "\n".join(statements)
        assert "aucun test pairé" in joined.lower()

    def test_full_2021_is_declared_mad_and_unhedged(self, facts, statements):
        u = facts["universes"]["full_2021"]
        assert u["base_currency"] == "MAD"
        assert u["hedge_status"] == "unhedged"
        text = "\n".join(statements).lower()
        assert "mad" in text and "non couvert" in text

    def test_etf_2017_is_declared_usd_and_not_applicable(self, facts, statements):
        u = facts["universes"]["etf_2017"]
        assert u["base_currency"] == "USD"
        assert u["converted"] is False
        assert "not applicable" in u["hedge_status"].lower(), (
            f"etf_2017 hedge status must be not-applicable (single-currency); "
            f"got {u['hedge_status']!r}"
        )
        assert "usd" in "\n".join(statements).lower()

    def test_the_statements_declare_the_universes_non_comparable(self, statements):
        assert any("ne sont pas comparables" in s for s in statements)

    def test_regime_conditional_is_the_prespecified_white_spa_comparator(
        self, facts, statements
    ):
        assert facts["multiple_testing"]["primary_benchmark"] == "regime_conditional"
        joined = "\n".join(statements)
        assert "pré-spécifié" in joined
        assert "n'est pas réécrit" in joined, (
            "the facts must state the comparator was not relabelled after the result"
        )

    def test_the_dsr_is_reported_without_an_editorial_qualifier(self, statements):
        """No threshold was pre-specified, so the value is reported bare."""
        joined = "\n".join(statements).lower()
        assert "0,6707" in joined
        for adjective in ("crédible", "faible", "insuffisant", "solide", "peu fiable"):
            assert adjective not in joined, (
                f"DSR carries the qualifier '{adjective}' but no threshold was "
                f"pre-specified to judge it against"
            )


SUPERSEDED_MARK = "SUPERSEDED"
RETIRED_NUMBERS = ("+6,2 %", "+6.2%", "+6,2%", "6,2~\\%", "6.2\\%",
                   "1,2363", "1.2363", "1,1644", "1.1644")


class TestRetiredNumbersDoNotSurvive:
    """The pre-correction headline (regime 1.2363 vs max_sharpe 1.1644, +6.2%)
    was computed on a mixed-currency universe. It may appear only inside a block
    explicitly marked SUPERSEDED, never as a live figure."""

    @staticmethod
    def _surfaces() -> list[str]:
        """Enumerated, not hard-coded. An earlier fixed list omitted the report
        chapters and front matter, and `+6,2 %` survived into three pages of the
        built PDF — caught only by reading the PDF. A surface list that must be
        maintained by hand is a surface list that goes stale."""
        rels = [
            "README.md",
            "dashboard/streamlit_app.py",
            "docs/MODEL_CARD_REGIME_CONDITIONAL.md",
        ]
        for pattern in ("dashboard/pages/*.py", "src/api/*.py",
                        "docs/rapport/chapters/*.tex",
                        "docs/rapport/frontmatter/*.tex"):
            rels += sorted(p.relative_to(ROOT).as_posix() for p in ROOT.glob(pattern))
        return rels

    def test_no_surface_quotes_a_retired_number_outside_a_superseded_block(self):
        offenders = []
        for rel in self._surfaces():
            path = ROOT / rel
            if not path.is_file():
                continue
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if SUPERSEDED_MARK in line:
                    continue
                for number in RETIRED_NUMBERS:
                    if number in line:
                        offenders.append(f"{rel}:{lineno}  {number}  |  {line.strip()[:90]}")
        assert not offenders, (
            "pre-correction figures appear as live values (mark the block "
            f"{SUPERSEDED_MARK} if they are shown as history):\n  "
            + "\n  ".join(offenders)
        )


class TestTheReportIncludesTheFactsChapterExactlyOnce:
    def test_faits_publies_is_included_once(self):
        main = ROOT / "docs" / "rapport" / "main.tex"
        if not main.is_file():
            pytest.skip("main.tex absent")
        n = main.read_text(encoding="utf-8").count("faits_publies")
        assert n == 1, (
            f"faits_publies.tex is included {n} times in main.tex; it must appear "
            f"exactly once — zero means the canonical facts are not in the report, "
            f"more than one duplicates them."
        )


class TestPreCommitSurfaceChecks:
    """Four properties agreed before the source commit."""

    # Sections allowed to carry a SUPERSEDED marker. Kept SHORT and explicit:
    # the marker silences the retired-number scan, so every entry is a place
    # where old figures are deliberately shown as history.
    #   README.md        — the cap-experiment table (pre-correction full_2021)
    #   Chapter3.tex     — the revision chain +14,3 % -> +6,2 % -> -10,47 %,
    #                      unrolled in full so no old value resembles the current one
    HISTORICAL_SECTIONS = ("README.md", "docs/rapport/chapters/Chapter3.tex")

    def test_superseded_appears_only_in_authorised_historical_sections(self):
        """A SUPERSEDED marker suppresses the retired-number scan, so it must not
        be usable as a blanket escape hatch on a live surface."""
        offenders = []
        for path in list(ROOT.glob("dashboard/**/*.py")) + \
                    list(ROOT.glob("src/api/*.py")) + \
                    list(ROOT.glob("docs/rapport/chapters/*.tex")):
            rel = path.relative_to(ROOT).as_posix()
            if rel in self.HISTORICAL_SECTIONS:
                continue
            if SUPERSEDED_MARK in path.read_text(encoding="utf-8"):
                offenders.append(rel)
        assert not offenders, (
            f"{SUPERSEDED_MARK} used outside the authorised historical sections "
            f"{self.HISTORICAL_SECTIONS}: {offenders}. It silences the retired-number "
            f"check, so it must not appear on a live surface."
        )

    def test_every_dashboard_view_shows_the_selected_universe_numeraire(self):
        for rel in ("dashboard/pages/1_Resultats_recherche.py",
                    "dashboard/pages/2_Explorateur_strategies.py"):
            src = (ROOT / rel).read_text(encoding="utf-8")
            assert "numeraire_for" in src or 'FACTS["statements"]' in src, (
                f"{rel} must render the numéraire of the universe it displays"
            )

    def test_no_strategy_is_preselected_or_called_recommended(self, statements):
        """A default of ALL strategies is neutral; a default of ONE would
        privilege it. And no surface may name a strategy 'recommandée'."""
        page = (ROOT / "dashboard" / "pages" / "2_Explorateur_strategies.py").read_text(
            encoding="utf-8"
        )
        assert "default=available" in page, (
            "the comparison multiselect must default to every strategy, not to one"
        )
        for rel in ("dashboard/pages/1_Resultats_recherche.py",
                    "dashboard/pages/2_Explorateur_strategies.py",
                    "src/api/main.py"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            for canonical in statements:
                text = text.replace(canonical, "")
            assert "recommandée" not in text.lower() or _negated(text, "recommandée"), \
                f"{rel} names a strategy as recommended"

    def test_the_nested_conclusion_names_protocol_and_oos_window(self):
        """The reversal is confounded: the nested OOS window differs from the
        single split's, so it cannot be attributed to protocol design alone."""
        tex = (ROOT / "docs" / "rapport" / "chapters" / "Chapter5.tex").read_text(
            encoding="utf-8"
        )
        assert "sensible au protocole" in tex
        assert "fenêtre hors échantillon associée" in tex
        assert "confondus" in tex, (
            "the report must say protocol and period are confounded here, not that "
            "the protocol alone explains the change of sign"
        )
        assert "0,6707" in tex
