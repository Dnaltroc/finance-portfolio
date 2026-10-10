"""Deal Sourcing Pipeline CLI.

    FINLIB_OFFLINE=1 python -m projects.sourcing run --day 2026-10-09      # fixture day
    python -m projects.sourcing run                                        # yesterday, live
    python -m projects.sourcing run --day 2026-10-09 --email you@x.com     # also send via SMTP env vars
"""

from __future__ import annotations

import os
import smtplib
from datetime import date, timedelta
from email.message import EmailMessage
from pathlib import Path

import typer
import yaml

from finlib.models.sourcing import ScoringConfig, digest_markdown, rank
from finlib.providers.formd import FormDProvider
from finlib.store.sourcing_db import SourcingDB

HERE = Path(__file__).resolve().parent
app = typer.Typer(add_completion=False, help="Daily SEC Form D sourcing digest.")


@app.callback()
def _root():
    """Daily SEC Form D sourcing digest."""


def load_config(path: Path) -> tuple[ScoringConfig, int]:
    raw = yaml.safe_load(path.read_text()) or {}
    top_n = int(raw.pop("top_n", 25))
    return ScoringConfig(**raw), top_n


def run_day(day: date, config: Path, db_path: Path, out_dir: Path) -> tuple[Path, dict]:
    cfg, top_n = load_config(config)
    db = SourcingDB(db_path)
    filings = FormDProvider().filings_for(day)
    rows = rank(filings, cfg, db.seen_ciks())
    new = db.upsert(rows)
    md = digest_markdown(rows, day.isoformat(), top_n, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"digest_{day.isoformat()}.md"
    path.write_text(md)
    stats = {"filings": len(rows), "kept": sum(r["kept"] for r in rows), "new": new, "db_total": db.count()}
    return path, stats


def send_email(to: str, subject: str, body_md: str) -> None:
    host, user, pwd = os.getenv("SMTP_HOST"), os.getenv("SMTP_USER"), os.getenv("SMTP_PASS")
    if not (host and user and pwd):
        raise RuntimeError("Set SMTP_HOST, SMTP_USER, SMTP_PASS to email the digest")
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = user, to, subject
    msg.set_content(body_md)
    with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587"))) as s:
        s.starttls()
        s.login(user, pwd)
        s.send_message(msg)


@app.command()
def run(
    day: str = typer.Option(None, help="YYYY-MM-DD; default is yesterday"),
    config: Path = typer.Option(HERE / "config.yaml"),
    db: Path = typer.Option(Path("sourcing.sqlite")),
    out: Path = typer.Option(Path("output")),
    email: str = typer.Option(None, help="Send the digest to this address via SMTP env vars"),
):
    """Fetch the day's Form D filings, score them, write the digest, update the database."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    d = date.fromisoformat(day) if day else date.today() - timedelta(days=1)
    path, stats = run_day(d, config, db, out)
    typer.echo(f"{d}: {stats['filings']} filings, {stats['kept']} kept, {stats['new']} new, "
               f"{stats['db_total']} in database")
    typer.echo(f"Wrote {path}")
    if email:
        send_email(email, f"Form D sourcing digest {d}", path.read_text())
        typer.echo(f"Emailed {email}")
