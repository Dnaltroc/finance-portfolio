# Variance Report Generator

Two CSV exports in, a budget-vs-actual Excel pack and a draft commentary out. Built for the monthly close: validate the GL export, compute variances with the right sign, rank what matters, hand the analyst a Flags sheet with a blank Driver column.

```bash
python -m projects.variance demo --period 2026-09                       # bundled sample data
python -m projects.variance run --actuals a.csv --budget b.csv --period 2026-09 --config projects/variance/config.yaml
```

## Input format

Both files share one schema. Validation runs every row through pydantic and reports the file, row number and reason for each failure before anything else happens.

| column | rule |
| --- | --- |
| period | `YYYY-MM` |
| department | free text |
| account | free text |
| account_type | one of `revenue, income, sales, expense, cogs, opex, capex` |
| amount | number |

Duplicate key rows (same period, department, account, type) are summed. A line in actuals with no budget row shows budget 0 and the whole amount as variance.

## Sign convention

`variance_amt` is always actual − budget. `fav_unfav` applies the judgment: revenue over budget is **F**, expense over budget is **U**. The `favorable_amt` column flips expense variances so department rollups add up in business terms.

## Flag rule

`config.yaml`: a line is flagged when |variance| ≥ `abs_amount` **or** |variance %| ≥ `pct`. Top `top_n` by absolute dollars go to the Flags sheet and the commentary.

## Output

| File | Contents |
| --- | --- |
| `variance_<period>.xlsx` | **Summary** (Revenue / Expenses / Net, then by department, F/U colored), **Flags** (top variances with an empty Driver column), **Detail** (every line, filterable), **Inputs** (both files as loaded, blue) |
| `commentary_<period>.md` | One bullet per flagged line: "Marketing / Paid media: $56,022 (24%) over budget, unfavorable, driven by ___" |

## Tests

`FINLIB_OFFLINE=1 pytest tests/test_variance.py` covers sign logic, either-rule thresholds, rollups tying to detail, validation error messages, unbudgeted lines, commentary count, and both sample months end to end.

Sample data is synthetic (`make_sample_data.py`) with four planted stories so the commentary has something to say.
