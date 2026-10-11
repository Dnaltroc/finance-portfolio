"""Capture a screenshot of every app page into docs/screenshots/.

    pip install -e ".[app,screenshots]" && python -m playwright install chromium
    streamlit run app/Home.py --server.port 8599 &
    python scripts/screenshots.py --url http://localhost:8599
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGES = [("", "home"), ("comps", "comps"), ("variance", "variance"),
         ("lbo", "lbo"), ("sourcing", "sourcing"), ("kpis", "dashboard")]
OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--height", type=int, default=2200, help="viewport height; Streamlit scrolls inside the app")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": args.height}, device_scale_factor=1)
        for path, name in PAGES:
            page.goto(f"{args.url.rstrip('/')}/{path}", wait_until="networkidle")
            page.wait_for_selector("[data-testid='stMain']", timeout=60_000)
            # Wait until the script finishes and every Plotly chart has drawn.
            page.wait_for_function(
                "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=120_000)
            page.wait_for_timeout(2500)
            # Crop to the rendered content so the image has no empty tail.
            box = page.locator("[data-testid='stMainBlockContainer']").bounding_box()
            height = min(args.height, int(box["y"] + box["height"]) + 24) if box else args.height
            page.screenshot(path=str(OUT / f"{name}.png"), clip={"x": 0, "y": 0, "width": 1440, "height": height})
            print(f"wrote {OUT / f'{name}.png'}")
        browser.close()


if __name__ == "__main__":
    main()
