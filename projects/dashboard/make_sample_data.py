"""Synthetic monthly KPIs for 8 companies over 24 months with seasonality, noise and a few stories.
Run: python -m projects.dashboard.make_sample_data"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pandas as pd

random.seed(42)
OUT = Path(__file__).resolve().parent / "sample_data"

# name, starting monthly revenue, monthly growth, gross margin, opex ratio start, cash, headcount, story
COMPANIES = [
    ("Bayou Logistics",  820_000, 0.028, 0.68, 1.05, 9_500_000, 58, "steady"),
    ("Gulfstream Robotics", 310_000, 0.045, 0.52, 1.60, 6_200_000, 41, "burning"),
    ("Crescent Health",  540_000, 0.022, 0.74, 0.95, 4_100_000, 36, "profitable"),
    ("Pelican Fintech",  1_250_000, 0.018, 0.61, 0.80, 15_000_000, 92, "profitable"),
    ("Magnolia Battery", 190_000, 0.060, 0.35, 2.40, 7_800_000, 44, "burning"),
    ("Saltwater Analytics", 95_000, 0.075, 0.80, 2.80, 2_100_000, 14, "runway-risk"),
    ("Redfish Marine",   430_000, 0.012, 0.44, 0.62, 3_300_000, 31, "flat"),
    ("Cypress Edge",     660_000, 0.035, 0.70, 1.20, 8_900_000, 52, "steady"),
]
MONTHS = pd.period_range("2024-10", periods=24, freq="M").strftime("%Y-%m").tolist()


def main() -> None:
    rows = []
    for name, rev0, g, gm, opex_ratio, cash, heads, story in COMPANIES:
        cash_bal = cash
        for i, m in enumerate(MONTHS):
            season = 1 + 0.06 * math.sin(2 * math.pi * (i % 12) / 12)
            shock = 1.0
            if story == "flat" and i > 12:
                g_eff = 0.002
            else:
                g_eff = g
            if story == "runway-risk" and i == 17:
                shock = 0.82  # churn event
            revenue = rev0 * (1 + g_eff) ** i * season * shock * random.uniform(0.97, 1.03)
            cogs = revenue * (1 - gm) * random.uniform(0.97, 1.03)
            # opex ratio drifts down as companies scale
            ratio = opex_ratio * (0.985 ** i)
            opex = revenue * ratio * random.uniform(0.96, 1.04)
            if story == "profitable":
                opex = min(opex, revenue * (gm - 0.08))
            net = revenue - cogs - opex
            cash_bal += net
            if story == "burning" and i == 10:
                cash_bal += 12_000_000  # Series B
            if story == "runway-risk" and i == 7:
                cash_bal += 6_500_000  # seed extension
            heads = int(heads * (1 + g_eff * 0.6))
            rows.append(dict(company=name, month=m, revenue=round(revenue), cogs=round(cogs),
                             opex=round(opex), cash=round(cash_bal), headcount=heads))
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "kpis.csv", index=False)
    print(f"wrote {len(rows)} rows to {OUT / 'kpis.csv'}")


if __name__ == "__main__":
    main()
