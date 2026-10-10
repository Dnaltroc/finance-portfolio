"""Filter and score Form D filings into a ranked sourcing list. Pure functions."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from finlib.providers.formd import FormD

FUND_INDUSTRY = {"pooled investment fund"}
FUND_NAME_PATTERNS = [
    r"\bL\.?P\.?$", r"\bFund\b", r"\bCapital Partners\b", r"\bInvestors\b", r"\bHoldings\b",
    r"\bSPV\b", r"\bOpportunit(y|ies)\b", r"\bVentures?\b\s*(I|II|III|IV|V|VI)?$", r"\bPortfolio\b",
    r"\bTrust\b", r"\bREIT\b", r"\bFeeder\b", r"\bMaster\b", r"\bCo-?Invest",
]
FUND_RE = re.compile("|".join(FUND_NAME_PATTERNS), re.IGNORECASE)


@dataclass
class ScoringConfig:
    target_states: list[str] = field(default_factory=lambda: ["LA", "TX", "FL"])
    target_industries: list[str] = field(default_factory=lambda: ["Technology", "Other Technology", "Business Services"])
    min_amount_sold: float = 250_000
    weights: dict = field(default_factory=lambda: {
        "amount": 45, "investors": 15, "first_time": 15, "state": 15, "industry": 10,
    })
    amount_log_ceiling: float = 50_000_000  # full amount score at or above this
    include_amendments: bool = False


def is_fund(f: FormD) -> bool:
    if f.industry_group.strip().lower() in FUND_INDUSTRY:
        return True
    if f.entity_type.lower() in ("limited partnership",) and FUND_RE.search(f.issuer):
        return True
    return bool(FUND_RE.search(f.issuer)) and f.industry_group.strip().lower() in ("", "other")


def keep(f: FormD, cfg: ScoringConfig) -> tuple[bool, str]:
    if is_fund(f):
        return False, "fund"
    if f.is_amendment and not cfg.include_amendments:
        return False, "amendment"
    if (f.total_sold or 0) < cfg.min_amount_sold:
        return False, "below min amount"
    return True, ""


def score(f: FormD, cfg: ScoringConfig, first_time: bool) -> tuple[float, dict]:
    w = cfg.weights
    parts = {}
    sold = f.total_sold or 0.0
    if sold > 0:
        parts["amount"] = w["amount"] * min(1.0, math.log10(sold) / math.log10(cfg.amount_log_ceiling))
    else:
        parts["amount"] = 0.0
    inv = f.investors or 0
    parts["investors"] = w["investors"] * min(1.0, inv / 25)
    parts["first_time"] = w["first_time"] if first_time else 0.0
    parts["state"] = w["state"] if f.state.upper() in {s.upper() for s in cfg.target_states} else 0.0
    parts["industry"] = w["industry"] if f.industry_group in cfg.target_industries else 0.0
    return round(sum(parts.values()), 1), parts


def rank(filings: list[FormD], cfg: ScoringConfig, seen_ciks: set[str]) -> list[dict]:
    rows = []
    for f in filings:
        ok, why = keep(f, cfg)
        first_time = f.cik not in seen_ciks
        s, parts = score(f, cfg, first_time) if ok else (0.0, {})
        rows.append({**f.as_dict(), "kept": ok, "drop_reason": why, "first_time": first_time,
                     "score": s, "score_parts": parts})
    rows.sort(key=lambda r: (-r["kept"], -r["score"], -(r["total_sold"] or 0)))
    return rows


def digest_markdown(rows: list[dict], day: str, top_n: int, cfg: ScoringConfig) -> str:
    kept = [r for r in rows if r["kept"]][:top_n]
    dropped = [r for r in rows if not r["kept"]]
    lines = [f"# Form D sourcing digest: {day}", "",
             f"{len(rows)} Form D filings; {len([r for r in rows if r['kept']])} operating-company raises kept, "
             f"{len(dropped)} dropped (funds, amendments, below ${cfg.min_amount_sold:,.0f}).",
             f"Target states: {', '.join(cfg.target_states)}. Target industries: {', '.join(cfg.target_industries)}.", "",
             "| # | Score | Issuer | State | Industry | Sold | Offering | Investors | First-time | Filing |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for i, r in enumerate(kept, start=1):
        sold = f"${r['total_sold']:,.0f}" if r["total_sold"] else "-"
        off = f"${r['total_offering']:,.0f}" if r["total_offering"] else "indefinite"
        lines.append(f"| {i} | {r['score']:.0f} | {r['issuer']} | {r['state']} | {r['industry_group']} | {sold} | {off} | "
                     f"{r['investors'] if r['investors'] is not None else '-'} | {'yes' if r['first_time'] else ''} | "
                     f"[{r['accession']}]({r['url']}) |")
    if not kept:
        lines.append("| - | - | No filings passed the filters today | | | | | | | |")
    lines += ["", "Score = amount (log-scaled) + investor count + first-time filer + target state + target industry. "
              "Source: SEC EDGAR Form D filings. Issuer-reported figures; no business description is available on Form D."]
    return "\n".join(lines) + "\n"
