"""Variance workbook: Summary, By Department, Detail, Flags, Inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter

from .styles import (
    BODY_FONT, CENTER, HEADER_FILL, HEADER_FONT, INPUT_FONT, PCT_FMT, RIGHT,
    SUMMARY_FONT, TOP_BORDER, autosize,
)

RED = PatternFill("solid", fgColor="FFC7CE")
GREEN = PatternFill("solid", fgColor="C6EFCE")
MONEY = '#,##0;(#,##0);"-"'


def _header(ws, row: int, headers: list[str]) -> None:
    for j, h in enumerate(headers, start=1):
        c = ws.cell(row=row, column=j, value=h)
        c.fill, c.font, c.alignment = HEADER_FILL, HEADER_FONT, CENTER
    ws.row_dimensions[row].height = 28


def _frame(ws, df: pd.DataFrame, start_row: int, fmts: dict[str, str], font=BODY_FONT) -> int:
    """Write a DataFrame starting at start_row (header there). Returns the last row written."""
    _header(ws, start_row, list(df.columns))
    r = start_row + 1
    for rec in df.itertuples(index=False):
        for j, (col, v) in enumerate(zip(df.columns, rec), start=1):
            if isinstance(v, float) and pd.isna(v):
                v = None
            c = ws.cell(row=r, column=j, value=v)
            c.font = font
            if col in fmts:
                c.number_format, c.alignment = fmts[col], RIGHT
        r += 1
    return r - 1


def _fu_colors(ws, col_idx: int, first: int, last: int) -> None:
    letter = get_column_letter(col_idx)
    rng = f"{letter}{first}:{letter}{last}"
    ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"U"'], fill=RED))
    ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"F"'], fill=GREEN))


def write_variance_workbook(
    period: str,
    summary: pd.DataFrame,
    by_dept: pd.DataFrame,
    detail: pd.DataFrame,
    flags: pd.DataFrame,
    actuals: pd.DataFrame,
    budget: pd.DataFrame,
    thresholds_text: str,
    out_dir: Path,
) -> Path:
    wb = Workbook()

    # Summary
    ws = wb.active
    ws.title = "Summary"
    ws["A1"] = f"Budget vs Actual: {period}"
    ws["A1"].font = SUMMARY_FONT
    ws["A2"] = f"Flag rule: {thresholds_text}. F = favorable, U = unfavorable. Negatives in parentheses."
    ws["A2"].font = BODY_FONT
    s = summary[["line", "actual", "budget", "variance_amt", "variance_pct", "fav_unfav"]].rename(
        columns={"line": "P&L line", "actual": "Actual", "budget": "Budget",
                 "variance_amt": "Variance $", "variance_pct": "Variance %", "fav_unfav": "F/U"})
    last = _frame(ws, s, 4, {"Actual": MONEY, "Budget": MONEY, "Variance $": MONEY, "Variance %": PCT_FMT},
                  font=SUMMARY_FONT)
    for j in range(1, 7):
        ws.cell(row=last, column=j).border = TOP_BORDER
    _fu_colors(ws, 6, 5, last)
    ws.cell(row=last + 2, column=1, value="By department (contribution: revenue minus expense)").font = SUMMARY_FONT
    d = by_dept[["department", "actual", "budget", "variance_amt", "variance_pct", "fav_unfav"]].rename(
        columns={"department": "Department", "actual": "Actual contribution", "budget": "Budget contribution",
                 "variance_amt": "Variance $", "variance_pct": "Variance %", "fav_unfav": "F/U"})
    start = last + 3
    last2 = _frame(ws, d, start, {"Actual contribution": MONEY, "Budget contribution": MONEY,
                                  "Variance $": MONEY, "Variance %": PCT_FMT})
    _fu_colors(ws, 6, start + 1, last2)
    autosize(ws)

    # Flags
    ws = wb.create_sheet("Flags")
    ws["A1"] = f"Top {len(flags)} variances by absolute $. Fill in the Driver column."
    ws["A1"].font = SUMMARY_FONT
    f = flags[["department", "account", "account_type", "actual", "budget", "variance_amt",
               "variance_pct", "fav_unfav", "driver"]].rename(columns=str.title)
    f = f.rename(columns={"Variance_Amt": "Variance $", "Variance_Pct": "Variance %", "Fav_Unfav": "F/U",
                          "Account_Type": "Type"})
    last = _frame(ws, f, 3, {"Actual": MONEY, "Budget": MONEY, "Variance $": MONEY, "Variance %": PCT_FMT})
    _fu_colors(ws, 8, 4, last)
    ws.column_dimensions["I"].width = 48
    ws.freeze_panes = "A4"
    autosize(ws)
    ws.column_dimensions["I"].width = 48

    # Detail
    ws = wb.create_sheet("Detail")
    dd = detail.drop(columns=["driver"]).rename(columns={
        "variance_amt": "variance $", "variance_pct": "variance %", "favorable_amt": "favorable $"})
    last = _frame(ws, dd, 1, {"actual": MONEY, "budget": MONEY, "variance $": MONEY,
                              "favorable $": MONEY, "variance %": PCT_FMT})
    _fu_colors(ws, list(dd.columns).index("fav_unfav") + 1, 2, last)
    ws.auto_filter.ref = f"A1:{get_column_letter(len(dd.columns))}{last}"
    ws.freeze_panes = "A2"
    autosize(ws)

    # Inputs (blue): the raw files as loaded
    ws = wb.create_sheet("Inputs")
    ws["A1"] = "Actuals as loaded"
    ws["A1"].font = SUMMARY_FONT
    last = _frame(ws, actuals, 2, {"amount": MONEY}, font=INPUT_FONT)
    ws.cell(row=last + 2, column=1, value="Budget as loaded").font = SUMMARY_FONT
    _frame(ws, budget, last + 3, {"amount": MONEY}, font=INPUT_FONT)
    autosize(ws)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"variance_{period}.xlsx"
    wb.save(path)
    return path
