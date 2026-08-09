"""
release_facts.py — the claims every public surface must make, in one place.

Addresses: P4 — this project has twice shipped surfaces that disagreed with each
other. `phase2_hurdle.json` drifted a universe behind and nobody noticed until a
manual `dvc status`; the 2026-08-02 claim reframing had to be applied by hand to
six surfaces plus a binary deliverable plus a JSON field the API re-served, and
§17.11 records that nothing in the test suite could tell they had drifted apart
because `test_artifact_consistency.py` compares NUMBERS, not WORDING.

So the wording lives here, once, with its numbers derived from Gold rather than
typed. The dashboard and the API import it; a test asserts the README and the
report carry the same claims. A surface that disagrees now fails a test instead
of being spotted by a reader.

WHAT MAY AND MAY NOT BE SAID. Two rules do the most work:

  * An observed difference is not a demonstrated one. No paired test of the
    regime-minus-Markowitz difference is presented, so neither direction may be
    called superior (AGENTS.md §5.2).
  * `regime_conditional` is the PRE-SPECIFIED primary comparator for the
    White/SPA correction. It was fixed before the results were seen and is not
    rewritten now that the sign has changed. Relabelling the benchmark after
    the fact is exactly the multiple-testing abuse that correction exists to
    prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"

# Universes are evaluated separately and never pooled. Their Sharpe LEVELS are
# not comparable: they are denominated in different currencies, over different
# windows, on different asset counts.
NOT_COMPARABLE = (
    "Les niveaux de Sharpe ne sont pas comparables d'un univers à l'autre : "
    "les deux univers sont libellés dans des devises différentes, sur des "
    "fenêtres différentes et avec un nombre d'actifs différent."
)

NO_RECOMMENDATION = (
    "Aucune stratégie n'est recommandée, déployée, ni présentée comme une "
    "valeur ajoutée établie. Ce livrable est un prototype de recherche."
)

NO_PAIRED_TEST = (
    "Ces écarts ne démontrent pas une supériorité de l'approche classique : "
    "aucun test pairé de cette différence n'est présenté."
)

PRIMARY_BENCHMARK_NOTE = (
    "`regime_conditional` demeure le comparateur primaire pré-spécifié des "
    "tests White/SPA. Ce choix a été fixé avant l'observation des résultats et "
    "n'est pas réécrit maintenant que le signe de l'écart a changé."
)


def _load(name: str, root: Path = ROOT) -> dict:
    return json.loads((root / "data" / "gold" / name).read_text())


def load_facts(root: Path = ROOT) -> dict:
    """
    Assemble every released claim from the canonical Gold artifacts.

    Addresses: P4 — numbers are DERIVED here, never typed, for the same reason
    `tests/test_run_dashboard_data.py` forbids a Sharpe literal in the dashboard
    runner: a stakeholder-facing surface is the worst place for a stale figure.

    Returns:
        Dict of numbers and pre-rendered French sentences. Consumers should
        render these rather than compose their own, so the surfaces cannot
        drift apart in wording as they did on 2026-08-02.
    """
    showcase = _load("dashboard_showcase.json", root)["universes"]
    currency = _load("currency_manifest.json", root)["universes"]
    paired = _load("paired_comparison_results.json", root)
    nested = _load("nested_walkforward_results.json", root)

    facts: dict = {"universes": {}}
    for key in ("full_2021", "etf_2017"):
        u, c = showcase[key], currency[key]
        facts["universes"][key] = {
            "base_currency": c.get("base_currency"),
            "converted": c.get("converted"),
            "hedge_status": c.get("hedge_status"),
            "fx_series": c.get("fx_series"),
            "best_classical": u["best_classical"],
            "regime": u["best_ml"],
            "point_difference_pct": u["headline_lift_pct"],
        }

    mt = paired["multiple_testing"]
    facts["multiple_testing"] = {
        "primary_benchmark": mt["primary_benchmark"],
        "n_candidates": mt["n_candidates_corrected_for"],
        "verdict": mt["verdict"],
    }
    facts["nested"] = {
        "best": nested["best"],
        "dsr": nested["best_dsr_vs_search"],
        "n_trials": nested["n_search_trials"],
        "oos_start": nested["provenance"]["oos_range"]["start"],
        "oos_end": nested["provenance"]["oos_range"]["end"],
        "n_oos_rows": nested["provenance"]["oos_range"]["n_rows"],
        "base_currency": nested["provenance"]["base_currency"],
    }
    facts["statements"] = _statements(facts)
    return facts


def _fr(x: float, n: int = 2) -> str:
    """French decimal convention. The deliverable is in French (§4); a report
    mixing `-10.47 %` with `1,0690` reads as machine output rather than prose."""
    return f"{x:.{n}f}".replace(".", ",")


def _currency_statement(universe: str, payload: dict) -> str:
    """Render one universe's numeraire claim from the currency manifest alone.

    Addresses: P4 — API endpoints that only need the currency contract must not
    depend on the entire evaluation bundle. Besides being unnecessarily deep,
    that coupling made hermetic API tests reach for unrelated Gold artifacts.
    """
    if universe == "full_2021":
        return (
            f"`full_2021` est libellé en {payload['base_currency']}, converti au "
            f"taux de référence officiel de Bank Al-Maghrib ({payload['fx_series']}, "
            f"MAD par USD). Le portefeuille reste **non couvert** : la variation "
            f"de change réalisée est incluse dans la performance, aucun contrat "
            f"à terme ni coût de roulement n'est modélisé."
        )
    if universe == "etf_2017":
        return (
            f"`etf_2017` est libellé en {payload['base_currency']} : cet univers ne "
            f"contient que des ETF mono-devise, n'a donc jamais présenté de "
            f"défaut de numéraire, et est **inchangé** par la correction."
        )
    raise KeyError(universe)


def _statements(f: dict) -> dict:
    """Render the canonical sentences. Every number interpolated, none typed."""
    full, etf = f["universes"]["full_2021"], f["universes"]["etf_2017"]
    nested = f["nested"]
    return {
        "full_2021_currency": _currency_statement("full_2021", full),
        "etf_2017_currency": _currency_statement("etf_2017", etf),
        "not_comparable": NOT_COMPARABLE,
        "point_difference": (
            f"Sur `full_2021`, l'écart ponctuel entre `regime_conditional` et "
            f"`{full['best_classical']['name']}` est de "
            f"**{_fr(full['point_difference_pct'])} %** "
            f"({_fr(full['regime']['sharpe_net'], 4)} contre "
            f"{_fr(full['best_classical']['sharpe_net'], 4)} de Sharpe net). "
            f"L'écart est également défavorable sur `etf_2017` : "
            f"**{_fr(etf['point_difference_pct'], 1)} %**."
        ),
        "no_paired_test": NO_PAIRED_TEST,
        "primary_benchmark": PRIMARY_BENCHMARK_NOTE,
        "multiple_testing": (
            f"Correction pour tests multiples (White 2000, Hansen 2005) sur les "
            f"{f['multiple_testing']['n_candidates']} configurations atteignables : "
            f"aucun candidat n'établit de surperformance face au comparateur "
            f"primaire pré-spécifié."
        ),
        "nested": (
            f"Walk-forward imbriqué (fenêtre OOS {nested['oos_start']} → "
            f"{nested['oos_end']}, {nested['n_oos_rows']} lignes) : le classement "
            f"est sensible au protocole d'évaluation et à la fenêtre hors "
            f"échantillon associée. Ratio de Sharpe dégonflé (DSR) = "
            f"{_fr(nested['dsr'], 4)} sur {nested['n_trials']} configurations."
        ),
        "no_recommendation": NO_RECOMMENDATION,
    }


def numeraire_for(universe: str, root: Path = ROOT) -> dict:
    """
    The numéraire of ONE universe, for surfaces that serve one at a time.

    Addresses: P1, P4 — there is no global base currency in this project and a
    surface must never imply one. `full_2021` is MAD (converted at the official
    Bank Al-Maghrib reference rate, unhedged); `etf_2017` is USD and was never
    converted. A single shared caveat string, as the API previously used, states
    a mixed exposure for both and is therefore wrong for each.

    Raises:
        KeyError: on an unknown universe, rather than defaulting to either
            currency — a wrong numéraire is worse than a missing one.
    """
    manifest = _load("currency_manifest.json", root)["universes"]
    u = manifest[universe]
    return {
        "universe": universe,
        "base_currency": u["base_currency"],
        "converted": u["converted"],
        "hedge_status": u["hedge_status"],
        "fx_series": u.get("fx_series"),
        "statement": _currency_statement(universe, u),
        "cross_universe_note": NOT_COMPARABLE,
    }


def statement_list(root: Path = ROOT) -> list[str]:
    """The canonical claims, in the order every surface should present them."""
    s = load_facts(root)["statements"]
    return [s[k] for k in (
        "full_2021_currency", "etf_2017_currency", "not_comparable",
        "point_difference", "no_paired_test", "primary_benchmark",
        "multiple_testing", "nested", "no_recommendation",
    )]
