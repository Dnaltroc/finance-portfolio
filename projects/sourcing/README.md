# Deal Sourcing Pipeline (SEC Form D)

Every company raising private capital under Reg D files a Form D within 15 days of first sale. That filing hits EDGAR weeks before a press release, and it is the primary record behind what PitchBook and FilingFlow sell. This pipeline reads yesterday's filings, drops the funds, scores the operating-company raises, and writes a ranked digest.

```bash
FINLIB_OFFLINE=1 python -m projects.sourcing run --day 2026-10-09        # fixture day, no network
python -m projects.sourcing run                                          # yesterday, live (needs SEC_IDENTITY)
python -m projects.sourcing run --email you@example.com                  # also send via SMTP_* env vars
```

Sample digest line:

```
| 1 | 99 | Bayou Logistics Software Inc | LA | Technology | $40,000,000 | indefinite | 31 | yes | [0001900001-26-000001](https://www.sec.gov/...) |
```

## Pipeline

1. **Index.** Fetch `daily-index/<year>/QTR<q>/form.<YYYYMMDD>.idx`, keep rows with form type `D` or `D/A`. A 404 means weekend or holiday.
2. **Fetch.** For each row, GET `edgar/data/<cik>/<accession>/primary_doc.xml`. User-Agent carries `SEC_IDENTITY`; requests are capped at 8 per second with backoff on 429.
3. **Parse.** Namespace-agnostic parse of the `edgarSubmission` schema: issuer, CIK, entity type, state, year of incorporation, industry group, amounts, investor count, exemption, amendment flag, security types. Related-person (officer/director) data is never parsed or stored.
4. **Filter.** Drop pooled investment funds by industry group, fund-shaped names (`LP`, `Fund`, `SPV`, `Feeder`, …) when the industry is blank or "Other", amendments (configurable), and raises below `min_amount_sold`.
5. **Score 0–100** from `config.yaml` weights: amount sold (log-scaled to a ceiling), investor count, first-time filer (CIK not in our database), target state, target industry.
6. **Store.** Upsert into `sourcing.sqlite` keyed on accession number, so re-running a day never duplicates and the first-time flag stays honest.
7. **Digest.** `output/digest_<day>.md` with the top N and a link to each filing; optional SMTP send.

## Schedule

`.github/workflows/sourcing.yml` runs weekdays at 12:17 UTC. With a `SEC_IDENTITY` repo secret it runs live and caches the SQLite file between runs; without one it does a dry run on the fixture day. It can also run as a Claude scheduled task or a cron line.

## Limits worth knowing

- Form D has **no business description**. Industry targeting uses the issuer-chosen industry group, which is coarse.
- Amounts are self-reported by the issuer.
- Coverage is US filings only; a company raising entirely under a non-Reg-D exemption will not appear.

## Tests

`FINLIB_OFFLINE=1 pytest tests/test_sourcing.py`: index parsing ignores non-D forms, parser fields and "Indefinite" handling, fund filter by industry and by name, keep rules, score bounds and caps, ranking order, digest row count, dedupe across two runs, and the end-to-end digest file. The fixture day in `tests/fixtures/formd/2026-10-09/` holds 36 synthetic filings shaped like real `edgarSubmission` XML.
