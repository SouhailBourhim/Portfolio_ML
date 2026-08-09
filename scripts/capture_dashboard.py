"""
capture_dashboard.py — regenerate the dashboard plates used in the report.

Addresses: P4 — the report embeds screenshots of the dashboard, and a screenshot
is an artifact like any other: it can go stale, and a stale one is worse than a
stale number because a reader believes their own eyes. The plates shipped before
this script showed the pre-correction page, which headlined a POSITIVE ML
value promise that the base-currency correction reversed. Nothing could detect
that, because they were taken by hand.

Making the capture a script means the plates are reproducible from the running
app rather than remembered, and can be regenerated whenever the page changes.

Streamlit scrolls an internal container (`section.stMain`), not the document, so
a naive full-page screenshot returns only the first viewport. This scrolls that
element explicitly and captures overlapping plates.

Usage:
    streamlit run dashboard/streamlit_app.py --server.port 8502 &
    python scripts/capture_dashboard.py --port 8502
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "rapport" / "assets" / "figures"

# (url path, output stem, number of plates the report includes)
PAGES = [
    ("Resultats_recherche", "dashboard_page1", 4),
    ("Explorateur_strategies", "dashboard_page2", 3),
]
VIEWPORT = {"width": 1440, "height": 1750}
SCROLLER = "section.stMain"


def capture(port: int) -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is required: pip install playwright && "
              "python -m playwright install chromium", file=sys.stderr)
        return 2

    FIGURES.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
        for path, stem, n_plates in PAGES:
            page.goto(f"http://localhost:{port}/{path}", wait_until="networkidle")
            # Streamlit renders charts after the initial network settle.
            page.wait_for_timeout(4000)

            total = page.evaluate(
                f"() => document.querySelector('{SCROLLER}').scrollHeight"
            )
            view = VIEWPORT["height"]
            # Overlapping offsets so no band of content falls between plates.
            span = max(total - view, 0)
            offsets = [round(span * i / (n_plates - 1)) for i in range(n_plates)] \
                if n_plates > 1 else [0]

            for i, offset in enumerate(offsets, 1):
                page.evaluate(
                    f"() => {{ document.querySelector('{SCROLLER}').scrollTop = {offset}; }}"
                )
                page.wait_for_timeout(900)
                out = FIGURES / f"{stem}_full_{i:02d}.png"
                page.screenshot(path=str(out))
                written.append(f"{out.name}  (scrollTop={offset}/{total})")

            # The single-plate variants the report also references.
            page.evaluate(f"() => {{ document.querySelector('{SCROLLER}').scrollTop = 0; }}")
            page.wait_for_timeout(600)
            out = FIGURES / f"{stem}.png"
            page.screenshot(path=str(out))
            written.append(out.name)
        browser.close()

    print("captures régénérées :")
    for w in written:
        print(f"  {w}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8502)
    raise SystemExit(capture(ap.parse_args().port))
