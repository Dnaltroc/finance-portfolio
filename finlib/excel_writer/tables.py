"""Comps workbook: Comps (formatted table), Inputs (raw pulls + sources), Notes."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter

from finlib.models.ratios import CompsRow, summary_rows
from finlib.providers.edgar import Fundamentals
from finlib.providers.market import Quote

from .styles import (
    BODY_FONT, CENTER, HEADER_FILL, HEADER_FONT, INPUT_FONT, NUM_FMT, PCT_FMT,
    PRICE_FMT, RIGHT, SUMMARY_FILL, SUMMARY_FONT, TOP_BORDER, X_FMT, autosize,
)

COLUMNS = [
    # (attribute, header, number format)
    ("ticker", "Ticker", None),
    ("company", "Company", None),
    ("price", "Price ($)", PRICE_FMT),
    ("market_cap_m", "Mkt Cap ($M)", NUM_FMT),
    ("ev_m", "EV ($M)", NUM_FMT),
    ("revenue_m", "LTM Revenue ($M)", NUM_FMT),
    ("ebitda_m", "LTM EBITDA ($M)", NUM_FMT),
    ("net_income_m", "LTM Net Inc ($M)", NUM_FMT),
    ("ev_revenue", "EV / Revenue", X_FMT),
    ("ev_ebitda", "EV / EBITDA", X_FMT),
    ("pe", "P / E", X_FMT),
    ("gross_margin", "Gross Margin", PCT_FMT),
    ("ebitda_margin", "EBITDA Margin", PCT_FMT),
    ("revenue_growth", "Rev Growth YoY", PCT_FMT),
    ("flags", "Flags", None),
]


def write_comps_workbook(
    rows: list[CompsRow],
    fundamentals: list[Fundamentals],
    quotes: list[Quote],
    peerset: str,
    out_dir: Path,
) -> Path:
    wb = Workbook()
    _comps_sheet(wb.active, rows, peerset)
    _inputs_sheet(wb.create_sheet("Inputs"), fundamentals, quotes)
    _notes_sheet(wb.create_sheet("Notes"), fundamentals, quotes)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"comps_{peerset}_{date.today().isoformat()}.xlsx"
    wb.save(path)
    return path


def _comps_sheet(ws, rows: list[CompsRow], peerset: str) -> None:
    ws.title = "Comps"
    ws["A1"] = f"Comparable Company Analysis: {peerset}"
    ws["A1"].font = SUMMARY_FONT
    ws["A2"] = f"As of {date.today().isoformat()}. $ in millions except per-share. NM = not meaningful."
    ws["A2"].font = BODY_FONT

    header_row = 4
    for j, (_, header, _) in enumerate(COLUMNS, start=1):
        c = ws.cell(row=header_row, column=j, value=header)
        c.fill, c.font, c.alignment = HEADER_FILL, HEADER_FONT, CENTER
    ws.row_dimensions[header_row].height = 30

    r = header_row + 1
    for row in rows:
        for j, (attr, _, fmt) in enumerate(COLUMNS, start=1):
            v = getattr(row, attr)
            c = ws.cell(row=r, column=j, value=v)
            c.font = BODY_FONT
            if fmt:
                c.number_format = fmt
                c.alignment = RIGHT
        r += 1
    last_data_row = r - 1

    # Median / mean rows on the multiples and margins only.
    for s in summary_rows(rows):
        for j, (attr, _, fmt) in enumerate(COLUMNS, start=1):
            v = s.get(attr) if attr != "company" else s["company"]
            c = ws.cell(row=r, column=j, value=v)
            c.font, c.fill, c.border = SUMMARY_FONT, SUMMARY_FILL, TOP_BORDER
            if fmt and attr in s:
                c.number_format = fmt
                c.alignment = RIGHT
        r += 1

    # Color scale on EV/EBITDA and P/E across the peer rows (low = green, high = red).
    if last_data_row > header_row:
        for attr in ("ev_ebitda", "pe"):
            col = [a for a, _, _ in COLUMNS].index(attr) + 1
            letter = get_column_letter(col)
            rng = f"{letter}{header_row + 1}:{letter}{last_data_row}"
            ws.conditional_formatting.add(
                rng,
                ColorScaleRule(start_type="min", start_color="C6EFCE",
                               mid_type="percentile", mid_value=50, mid_color="FFFFFF",
                               end_type="max", end_color="FFC7CE"),
            )

    ws.freeze_panes = ws.cell(row=header_row + 1, column=3)
    autosize(ws)


def _inputs_sheet(ws, fundamentals: list[Fundamentals], quotes: list[Quote]) -> None:
    ws["A1"] = ("Raw inputs pulled by the tool, in dollars and shares. Blue = hardcoded from source. "
                "The Comps sheet holds computed values; change inputs here and re-run the CLI to refresh.")
    ws["A1"].font = BODY_FONT
    headers = ["Ticker", "Concept", "Basis", "Value", "Source"]
    for j, h in enumerate(headers, start=1):
        c = ws.cell(row=3, column=j, value=h)
        c.fill, c.font = HEADER_FILL, HEADER_FONT
    r = 4
    qmap = {q.ticker: q for q in quotes}
    for f in fundamentals:
        q = qmap[f.ticker]
        _input_row(ws, r, f.ticker, "price", f"spot {q.as_of}", q.price, q.source); r += 1
        _input_row(ws, r, f.ticker, "shares_outstanding", "spot", q.shares_outstanding,
                   q.shares_source or q.source); r += 1
        for concept in ("revenue", "cogs", "operating_income", "da", "net_income"):
            _input_row(ws, r, f.ticker, concept, f"LTM {f.ltm_period}", f.ltm(concept),
                       f.sources.get(concept, "") + (" | " + f.basis[concept] if concept in f.basis else "")); r += 1
            if f.prior_ltm(concept) is not None:
                _input_row(ws, r, f.ticker, concept + " (prior LTM)", "LTM one year earlier",
                           f.prior_ltm(concept), f.sources.get(concept, "")); r += 1
        for concept in ("cash", "total_debt"):
            _input_row(ws, r, f.ticker, concept, f"as of {f.balance_date}",
                       f.latest(concept), f.sources.get(concept, "")); r += 1
    autosize(ws)


def _input_row(ws, r, ticker, concept, basis, value, source):
    ws.cell(row=r, column=1, value=ticker).font = BODY_FONT
    ws.cell(row=r, column=2, value=concept).font = BODY_FONT
    ws.cell(row=r, column=3, value=basis).font = BODY_FONT
    c = ws.cell(row=r, column=4, value=value)
    c.font = INPUT_FONT
    c.number_format = "#,##0.00" if concept == "price" else "#,##0"
    ws.cell(row=r, column=5, value=source).font = BODY_FONT


def _notes_sheet(ws, fundamentals: list[Fundamentals], quotes: list[Quote]) -> None:
    fixture = all("fixture" in b for f in fundamentals for b in f.basis.values())
    ltm_line = ("LTM = sum of the four most recent fixture quarters per concept (sample data)." if fixture else
                "LTM = the 10-K twelve-month value, or after a 10-Q: prior fiscal year + year-to-date "
                "- prior year-to-date (SEC companyfacts). The arithmetic per value is on the Inputs sheet.")
    lines = [
        "Methodology",
        ltm_line,
        "EBITDA = operating income + depreciation and amortization (not adjusted).",
        "EV = market cap + total debt - cash. No minority interest, preferred, or leases.",
        "Multiples on a zero or negative denominator are shown as NM and excluded from median/mean.",
        "",
        "Periods used",
    ]
    for f in fundamentals:
        lines.append(f"{f.ticker}: LTM {f.ltm_period}; balance sheet as of {f.balance_date}")
        if f.missing:
            lines.append(f"  missing concepts: {', '.join(f.missing)}")
    lines += ["", "Market data"]
    for q in quotes:
        lines.append(f"{q.ticker}: price {q.price} as of {q.as_of} ({q.source})")
    for i, line in enumerate(lines, start=1):
        c = ws.cell(row=i, column=1, value=line)
        c.font = SUMMARY_FONT if line in ("Methodology", "Periods used", "Market data") else BODY_FONT
    ws.column_dimensions["A"].width = 100
