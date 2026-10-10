# LBO Returns Engine

A sponsor-style LBO in Python, exported to an Excel workbook where every computed cell is a live formula. Change an input or pick a scenario from the dropdown and IRR moves. The Python engine is the reference; tests recalc the workbook in LibreOffice and assert Excel's `IRR()` matches Python within 0.1%.

```bash
python -m projects.lbo run --deal projects/lbo/deal.yaml      # writes output/lbo_<deal>.xlsx
python -m projects.lbo summary --deal projects/lbo/deal.yaml  # all scenarios + sensitivity grid in the terminal
```

Sample deal (`deal.yaml`): $80M EBITDA at 9.0x, 5.5x leverage across a Term Loan B (sweep) and Senior Notes, five-year hold, three scenarios.

```
  Base      MOIC 2.55x  IRR 20.6%
  Upside    MOIC 3.96x  IRR 31.7%
  Downside  MOIC 1.02x  IRR  0.4%
```

## Workbook

| Sheet | What is there |
| --- | --- |
| Assumptions | Deal inputs in blue; scenario dropdown in C4; scenario table with an **Active** column driven by `INDEX/MATCH`; tranche table; sources and uses |
| Operating | Revenue → EBITDA → EBIT → taxes → capex → ΔNWC → FCF, years 0–N across columns, every cell a formula |
| Debt | Per-tranche beginning / mandatory amortization / cash sweep / ending / interest, cash roll, total and net debt, leverage |
| Returns | Exit bridge (EV − debt + cash), sponsor equity, MOIC, equity cash flow row, `=IRR()` |
| Sensitivity | IRR grid, entry multiple × exit multiple, computed by Python at export (openpyxl cannot write Excel data tables) |

## Modeling choices, and why

- **Interest on beginning balance.** Avoids a circular reference so the workbook opens clean with no iteration setting. Average-balance interest is more precise; switch to it if the audience expects it and turn on iterative calc.
- **Cash sweep senior-first.** After mandatory amortization and the minimum cash balance, all remaining cash repays sweep-enabled tranches in table order. The sweep is capped at the tranche balance, so nothing over-repays.
- **ΔNWC = NWC % × change in revenue.** Standard simplification; replace with DSO/DIO/DPO drivers if the deal needs it.
- **Taxes on EBIT less interest, floored at zero.** No NOL carryforward.
- **No management rollover, dividends, or option pool.** Add them to the equity bridge on the Returns sheet if needed.

## Tests

`pytest tests/test_lbo.py`: closed-form IRR, sources = uses, debt roll-forward identities, no over-repayment, cash never below minimum, MOIC and IRR consistency, sensitivity monotonic in both axes, scenario validation, and (with LibreOffice present) Excel IRR = Python IRR for both the base case and after flipping the dropdown to Downside.
