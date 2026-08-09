"""Capture the read-only research API documentation for the final report.

Run the service first with::

    python -m uvicorn api.main:app --app-dir src --port 8765

The screenshot is a reproducible presentation artifact. It contains the public
OpenAPI contract only; no credential, subscription key, or live client data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "rapport_final" / "assets" / "figures" / "api_swagger.png"


def capture(port: int) -> None:
    from playwright.sync_api import sync_playwright

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1200}, device_scale_factor=2)
        page.goto(f"http://127.0.0.1:{port}/docs", wait_until="networkidle")
        page.wait_for_selector(".swagger-ui")
        page.screenshot(path=str(OUT), full_page=True)
        browser.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    capture(parser.parse_args().port)
