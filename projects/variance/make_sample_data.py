"""Generate two months of synthetic GL actuals and budget. Run: python -m projects.variance.make_sample_data"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

random.seed(7)
OUT = Path(__file__).resolve().parent / "sample_data"

DEPTS = {
    "Sales":       {"revenue": [("Product revenue", 1_800_000), ("Services revenue", 420_000)],
                    "expense": [("Salaries", 310_000), ("Commissions", 140_000), ("Travel", 38_000)]},
    "Marketing":   {"revenue": [],
                    "expense": [("Salaries", 180_000), ("Paid media", 230_000), ("Events", 45_000), ("Software", 22_000)]},
    "Engineering": {"revenue": [],
                    "expense": [("Salaries", 760_000), ("Cloud hosting", 165_000), ("Contractors", 90_000), ("Software", 41_000)]},
    "G&A":         {"revenue": [("Interest income", 12_000)],
                    "expense": [("Salaries", 240_000), ("Rent", 95_000), ("Legal", 30_000), ("Insurance", 18_000)]},
}
PERIODS = ["2026-08", "2026-09"]

# Planted stories so the commentary has something to say
SHOCKS = {
    ("2026-09", "Marketing", "Paid media"): 1.28,   # overspend
    ("2026-09", "Sales", "Product revenue"): 0.93,  # revenue miss
    ("2026-09", "Engineering", "Contractors"): 0.55, # underspend (hiring slipped)
    ("2026-09", "G&A", "Legal"): 2.4,               # one-off matter
    ("2026-08", "Sales", "Services revenue"): 1.15,
}


def main() -> None:
    a_rows, b_rows = [], []
    for p in PERIODS:
        for dept, groups in DEPTS.items():
            for kind, lines in groups.items():
                for acct, base in lines:
                    b_rows.append(dict(period=p, department=dept, account=acct, account_type=kind, amount=base))
                    noise = random.uniform(0.96, 1.04)
                    shock = SHOCKS.get((p, dept, acct), 1.0)
                    a_rows.append(dict(period=p, department=dept, account=acct, account_type=kind,
                                       amount=round(base * noise * shock, 2)))
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(a_rows).to_csv(OUT / "actuals.csv", index=False)
    pd.DataFrame(b_rows).to_csv(OUT / "budget.csv", index=False)
    print(f"wrote {len(a_rows)} actual rows and {len(b_rows)} budget rows to {OUT}")


if __name__ == "__main__":
    main()
