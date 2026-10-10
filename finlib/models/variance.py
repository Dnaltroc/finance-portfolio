"""Budget-vs-actual variance math. Pure pandas, no I/O.

Sign convention (the thing every FP&A reviewer checks first):
    Revenue accounts: actual > budget is FAVORABLE.
    Expense accounts: actual > budget is UNFAVORABLE.
`variance_amt` is always actual - budget. `fav_unfav` carries the judgment.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

REVENUE_TYPES = {"revenue", "income", "sales"}
EXPENSE_TYPES = {"expense", "cogs", "opex", "capex"}
KEYS = ["period", "department", "account", "account_type"]


class LedgerRow(BaseModel):
    period: str = Field(pattern=r"^\d{4}-\d{2}$", description="YYYY-MM")
    department: str
    account: str
    account_type: str
    amount: float

    @field_validator("account_type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in REVENUE_TYPES | EXPENSE_TYPES:
            raise ValueError(f"account_type must be one of {sorted(REVENUE_TYPES | EXPENSE_TYPES)}")
        return v

    @field_validator("department", "account")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


@dataclass
class Thresholds:
    abs_amount: float = 25_000  # flag if |variance| >= this
    pct: float = 0.10           # OR if |variance %| >= this
    top_n: int = 15             # rows on the Flags sheet


def validate_ledger(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Validate every row with pydantic; raise one error listing every bad row."""
    required = {"period", "department", "account", "account_type", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    errors, rows = [], []
    for i, rec in enumerate(df.to_dict("records"), start=2):  # +2: header row and 1-index
        try:
            rows.append(LedgerRow(**rec).model_dump())
        except ValidationError as e:
            errors.append(f"  {name} row {i}: {e.errors()[0]['msg']} ({rec})")
    if errors:
        raise ValueError(f"{name}: {len(errors)} invalid row(s)\n" + "\n".join(errors[:20]))
    out = pd.DataFrame(rows)
    dupes = out.duplicated(KEYS, keep=False)
    if dupes.any():
        out = out.groupby(KEYS, as_index=False)["amount"].sum()  # roll duplicate lines together
    return out


def compute_variance(actuals: pd.DataFrame, budget: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    """Outer-join actuals to budget on the four keys and compute variance columns."""
    a = actuals.rename(columns={"amount": "actual"})
    b = budget.rename(columns={"amount": "budget"})
    df = a.merge(b, on=KEYS, how="outer")
    df["actual"] = df["actual"].fillna(0.0)
    df["budget"] = df["budget"].fillna(0.0)
    df["variance_amt"] = df["actual"] - df["budget"]
    df["variance_pct"] = df.apply(
        lambda r: (r["variance_amt"] / abs(r["budget"])) if r["budget"] != 0 else None, axis=1
    )
    is_rev = df["account_type"].isin(REVENUE_TYPES)
    # favorable_amt > 0 means good for the business regardless of account type
    df["favorable_amt"] = df["variance_amt"].where(is_rev, -df["variance_amt"])
    df["fav_unfav"] = df["favorable_amt"].apply(lambda x: "F" if x > 0 else ("U" if x < 0 else "-"))
    df["flag"] = (df["variance_amt"].abs() >= th.abs_amount) | (
        df["variance_pct"].fillna(0).abs() >= th.pct
    )
    df["driver"] = ""  # analyst fills in
    cols = KEYS + ["actual", "budget", "variance_amt", "variance_pct", "favorable_amt",
                   "fav_unfav", "flag", "driver"]
    return df[cols].sort_values(["department", "account_type", "account"]).reset_index(drop=True)


def rollup(detail: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = detail.groupby(by, as_index=False)[["actual", "budget", "variance_amt", "favorable_amt"]].sum()
    g["variance_pct"] = g.apply(
        lambda r: (r["variance_amt"] / abs(r["budget"])) if r["budget"] != 0 else None, axis=1
    )
    g["fav_unfav"] = g["favorable_amt"].apply(lambda x: "F" if x > 0 else ("U" if x < 0 else "-"))
    return g


def pnl_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Revenue, expenses, and net by period: the three lines a manager reads first."""
    is_rev = detail["account_type"].isin(REVENUE_TYPES)
    rev = detail[is_rev].groupby("period")[["actual", "budget"]].sum()
    exp = detail[~is_rev].groupby("period")[["actual", "budget"]].sum()
    out = pd.DataFrame({
        "line": ["Revenue", "Expenses", "Net"],
        "actual": [rev["actual"].sum(), exp["actual"].sum(), rev["actual"].sum() - exp["actual"].sum()],
        "budget": [rev["budget"].sum(), exp["budget"].sum(), rev["budget"].sum() - exp["budget"].sum()],
    })
    out["variance_amt"] = out["actual"] - out["budget"]
    out["variance_pct"] = out.apply(
        lambda r: r["variance_amt"] / abs(r["budget"]) if r["budget"] != 0 else None, axis=1
    )
    out["favorable_amt"] = [out.loc[0, "variance_amt"], -out.loc[1, "variance_amt"], out.loc[2, "variance_amt"]]
    out["fav_unfav"] = out["favorable_amt"].apply(lambda x: "F" if x > 0 else ("U" if x < 0 else "-"))
    return out


def top_flags(detail: pd.DataFrame, n: int) -> pd.DataFrame:
    f = detail[detail["flag"]].copy()
    f["abs_var"] = f["variance_amt"].abs()
    return f.sort_values("abs_var", ascending=False).drop(columns="abs_var").head(n).reset_index(drop=True)


def commentary(flags: pd.DataFrame, period: str) -> str:
    """One bullet per flagged line in plain FP&A language, with a blank for the driver."""
    lines = [f"# Variance commentary: {period}", "",
             "Draft generated from flagged lines. Fill in the driver after each 'driven by'.", ""]
    for _, r in flags.iterrows():
        direction = "over" if r["variance_amt"] > 0 else "under"
        pct = f" ({abs(r['variance_pct']):.0%})" if pd.notna(r["variance_pct"]) else ""
        fu = "favorable" if r["fav_unfav"] == "F" else "unfavorable"
        lines.append(
            f"- **{r['department']} / {r['account']}**: ${abs(r['variance_amt']):,.0f}{pct} "
            f"{direction} budget, {fu}, driven by ___"
        )
    if len(lines) == 4:
        lines.append("- No lines exceeded the flag thresholds this period.")
    return "\n".join(lines) + "\n"
