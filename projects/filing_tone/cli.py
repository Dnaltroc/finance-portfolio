"""Filing Tone CLI: train sentiment models on Financial PhraseBank, score real 10-Ks.

    python -m projects.filing_tone data                              # download PhraseBank, show sizes
    python -m projects.filing_tone train --model baseline            # TF-IDF + logistic regression
    python -m projects.filing_tone train --model cnn                 # from-scratch PyTorch TextCNN
    python -m projects.filing_tone train --model distilbert          # fine-tuned transformer
    python -m projects.filing_tone evaluate --model distilbert       # test metrics + all-agree ablation
    python -m projects.filing_tone results                           # results.json + README table
    python -m projects.filing_tone score --text "Revenue fell 20%."  # probabilities for one sentence
    python -m projects.filing_tone snapshot                          # score AEHR + peers' latest 10-Ks
    python -m projects.filing_tone summary                           # print the committed snapshot

Everything heavy (torch, transformers, scikit-learn) is imported inside the command that
needs it, so `summary` and the Streamlit page run without the ml extra. The committed
artifacts are tone_snapshot.json (the app's data) and results.json (the README table).
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import median
from typing import Callable

import numpy as np
import typer

from finlib.nlp.data import LABELS, Example, label_counts, load_fixture, load_phrasebank, stratified_split
from finlib.nlp.evaluate import summary
from finlib.nlp.sec_text import TenKFetcher, extract_sections, filing_tone, html_to_text, split_sentences

HERE = Path(__file__).resolve().parent
OUT = Path("output") / "filing_tone"
MODELS = OUT / "models"
RUNS = OUT / "runs"
SNAPSHOT = HERE / "tone_snapshot.json"
RESULTS = HERE / "results.json"
README = HERE / "README.md"
DEFAULT_TICKERS = ["AEHR", "FORM", "KLIC", "COHU", "ONTO", "TER", "CAMT"]  # the pitch universe
MODEL_NAMES = ("baseline", "cnn", "distilbert")
# Picked on the validation split only (30-run sweep, 2026-10-16; val macro-F1 0.765 vs 0.705 at the Kim 2014 defaults).
CNN_ARCH = dict(embed_dim=200, channels=200, dropout=0.3)
Scorer = Callable[[list[str]], np.ndarray]

app = typer.Typer(add_completion=False, help="Financial sentiment in PyTorch, applied to SEC 10-K text.")


@app.callback()
def _root():
    """Filing Tone."""


# ---------------------------------------------------------------- data ---------------
def load_examples(fixture: bool, config: str = "sentences_50agree") -> list[Example]:
    return load_fixture() if fixture else load_phrasebank(config)


def splits(fixture: bool, seed: int):
    return stratified_split(load_examples(fixture), seed=seed)


@app.command()
def data(fixture: bool = typer.Option(False, help="Use the 90-sentence synthetic fixture instead of PhraseBank.")):
    """Download Financial PhraseBank (once) and print split sizes and the label mix."""
    ex = load_examples(fixture)
    train, val, test = stratified_split(ex)
    typer.echo(f"{'fixture (synthetic)' if fixture else 'Financial PhraseBank, sentences_50agree'}: {len(ex)} sentences")
    typer.echo(f"label mix: {label_counts(ex)}")
    typer.echo(f"train {len(train)} / val {len(val)} / test {len(test)}")
    if not fixture:
        allagree = load_phrasebank("sentences_allagree")
        typer.echo(f"sentences_allagree: {len(allagree)} (the ablation evaluates the test sentences in this set)")


# ---------------------------------------------------------------- training -----------
def _fit_cnn(train, val, cfg, device):
    import torch

    from finlib.nlp.batches import make_loader, vocab_encoder
    from finlib.nlp.models import TextCNN
    from finlib.nlp.tokenize import RegexTokenizer, Vocab
    from finlib.nlp.train import RunLog, train_loop

    vocab = Vocab.build([e.text for e in train], RegexTokenizer(), min_freq=1)
    enc = vocab_encoder(vocab, cfg.max_len)
    model = TextCNN(len(vocab), **CNN_ARCH)
    log = RunLog.start(cfg, device=str(device))
    log.vocab_size = len(vocab)
    model = train_loop(model, make_loader(train, enc, cfg.batch_size, True, seed=cfg.seed),
                       make_loader(val, enc, 128, False), cfg, device, log)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "itos": vocab.itos, "config": cfg.to_dict(), "arch": CNN_ARCH},
               MODELS / "cnn.pt")
    return model, enc, log


def _fit_distilbert(train, val, cfg, device):
    from finlib.nlp.batches import hf_encoder, make_loader
    from finlib.nlp.models import build_distilbert
    from finlib.nlp.train import RunLog, train_loop

    tok, model = build_distilbert()
    enc = hf_encoder(tok, cfg.max_len)
    log = RunLog.start(cfg, device=str(device))
    model = train_loop(model, make_loader(train, enc, cfg.batch_size, True, seed=cfg.seed),
                       make_loader(val, enc, 128, False), cfg, device, log)
    MODELS.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(MODELS / "distilbert")
    tok.save_pretrained(MODELS / "distilbert")
    return model, enc, log


def _fit_baseline(train, cfg):
    import joblib

    from finlib.nlp.baseline import fit_baseline
    from finlib.nlp.train import RunLog

    log = RunLog.start(cfg, device="cpu")
    t0 = datetime.now()
    pipe = fit_baseline([e.text for e in train], [e.label for e in train], seed=cfg.seed)
    log.wall_seconds = (datetime.now() - t0).total_seconds()
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS / "baseline.joblib")
    return pipe, log


def load_scorer(model: str, device: str | None = None) -> Scorer:
    """A function sentences -> (N, 3) probabilities, from a saved checkpoint."""
    if model == "baseline":
        import joblib

        from finlib.nlp.baseline import baseline_probs

        pipe = joblib.load(MODELS / "baseline.joblib")
        return lambda texts: baseline_probs(pipe, texts)
    import torch

    from finlib.nlp.train import choose_device, predict_texts

    dev = choose_device(device)
    if model == "cnn":
        from finlib.nlp.batches import vocab_encoder
        from finlib.nlp.models import TextCNN
        from finlib.nlp.tokenize import Vocab

        ck = torch.load(MODELS / "cnn.pt", map_location=dev, weights_only=False)
        vocab = Vocab(ck["itos"][2:])
        net = TextCNN(len(vocab), **ck.get("arch", {}))
        net.load_state_dict(ck["state"])
        net.to(dev)
        enc = vocab_encoder(vocab, ck["config"]["max_len"])
    elif model == "distilbert":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from finlib.nlp.batches import hf_encoder
        from finlib.nlp.models import HFClassifier

        tok = AutoTokenizer.from_pretrained(MODELS / "distilbert")
        net = HFClassifier(AutoModelForSequenceClassification.from_pretrained(MODELS / "distilbert")).to(dev)
        enc = hf_encoder(tok, 128)
    else:
        raise typer.BadParameter(f"model must be one of {MODEL_NAMES}")
    return lambda texts: predict_texts(net, enc, texts, dev)


def best_available_model() -> str:
    for m in ("distilbert", "cnn", "baseline"):
        if (MODELS / ("distilbert" if m == "distilbert" else f"{m}.{'pt' if m == 'cnn' else 'joblib'}")).exists():
            return m
    raise FileNotFoundError(f"No trained model under {MODELS}. Run `train --model cnn` first.")


def _evaluate(scorer: Scorer, test: list[Example]) -> dict:
    probs = scorer([e.text for e in test])
    return summary([e.label for e in test], probs.argmax(1))


def _ablation(scorer: Scorer, test: list[Example]) -> dict | None:
    """The test sentences that every annotator agreed on. Easier, and how the paper reports."""
    try:
        agreed = {e.text for e in load_phrasebank("sentences_allagree")}
    except Exception as exc:  # no network, no cache
        typer.echo(f"ablation skipped: {exc}")
        return None
    subset = [e for e in test if e.text in agreed]
    return {"subset": "test ∩ sentences_allagree", **_evaluate(scorer, subset)} if subset else None


def _merge_results(model: str, log) -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    results[model] = {**log.test, "run_id": log.run_id, "wall_seconds": log.wall_seconds, "device": log.device,
                      "epochs_run": len(log.history), "ablation": log.ablation, "train_size": log.train_size,
                      "test_size": log.test_size}
    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


@app.command()
def train(model: str = typer.Option("cnn", help="baseline | cnn | distilbert"),
          epochs: int = typer.Option(None), batch_size: int = typer.Option(None), lr: float = typer.Option(None),
          seed: int = 42, device: str = typer.Option(None, help="cuda | cpu; default: cuda if available"),
          fixture: bool = typer.Option(False, help="Train on the synthetic fixture (tests and CI smoke)."),
          no_ablation: bool = typer.Option(False, help="Skip the all-agree ablation.")):
    """Train one model tier, evaluate on the held-out test split, save checkpoint, log and results."""
    if model not in MODEL_NAMES:
        raise typer.BadParameter(f"model must be one of {MODEL_NAMES}")
    from finlib.nlp.train import TrainConfig, choose_device, set_seed

    defaults = {"baseline": dict(epochs=0, batch_size=0, lr=0.0),
                "cnn": dict(epochs=30, batch_size=32, lr=2e-3, max_len=64, patience=4),
                "distilbert": dict(epochs=4, batch_size=32, lr=3e-5, max_len=128, patience=2)}[model]
    cfg = TrainConfig(model=model, seed=seed, device=device, **defaults)
    for k, v in (("epochs", epochs), ("batch_size", batch_size), ("lr", lr)):
        if v is not None:
            setattr(cfg, k, v)
    set_seed(seed)
    tr, va, te = splits(fixture, seed)
    dev = choose_device(device)
    typer.echo(f"{model}: train {len(tr)} / val {len(va)} / test {len(te)} on {dev}")
    if model == "baseline":
        _, log = _fit_baseline(tr, cfg)
    elif model == "cnn":
        _, _, log = _fit_cnn(tr, va, cfg, dev)
    else:
        _, _, log = _fit_distilbert(tr, va, cfg, dev)
    log.train_size, log.val_size, log.test_size = len(tr), len(va), len(te)
    scorer = load_scorer(model, device)
    log.test = _evaluate(scorer, te)
    if not fixture and not no_ablation:
        log.ablation = _ablation(scorer, te)
    path = log.save(RUNS / f"{log.run_id}.json")
    for h in log.history:
        typer.echo(f"  epoch {h['epoch']:>2}  loss {h['train_loss']:.3f}  val acc {h['val_accuracy']:.3f}  "
                   f"val F1 {h['val_macro_f1']:.3f}  {h['seconds']:.0f}s")
    typer.echo(f"test accuracy {log.test['accuracy']:.3f}  macro-F1 {log.test['macro_f1']:.3f}  "
               f"({log.wall_seconds:.0f}s on {log.device}); log {path}")
    if log.ablation:
        typer.echo(f"all-agree subset (n={log.ablation['n']}): accuracy {log.ablation['accuracy']:.3f}  "
                   f"macro-F1 {log.ablation['macro_f1']:.3f}")
    if not fixture:
        _merge_results(model, log)


@app.command()
def evaluate(model: str = typer.Option(None, help="Defaults to the best trained model present."),
             seed: int = 42, device: str = typer.Option(None)):
    """Re-score the test split (and the all-agree subset) from a saved checkpoint."""
    model = model or best_available_model()
    _, _, te = splits(False, seed)
    scorer = load_scorer(model, device)
    s = _evaluate(scorer, te)
    typer.echo(f"{model}: test n={s['n']} accuracy {s['accuracy']:.3f} macro-F1 {s['macro_f1']:.3f}")
    for name, m in s["per_class"].items():
        typer.echo(f"  {name:<9} P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} (n={m['support']})")
    typer.echo(f"  confusion (rows true, cols pred, order {LABELS}): {s['confusion']}")
    ab = _ablation(scorer, te)
    if ab:
        typer.echo(f"all-agree subset n={ab['n']}: accuracy {ab['accuracy']:.3f} macro-F1 {ab['macro_f1']:.3f}")


# ---------------------------------------------------------------- results ------------
def load_results() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}


def results_table(results: dict) -> str:
    rows = ["| Model | Accuracy | Macro-F1 | F1 negative | F1 neutral | F1 positive | Train time |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for m in MODEL_NAMES:
        if m not in results:
            continue
        r = results[m]
        pc = r["per_class"]
        secs = r.get("wall_seconds", 0.0)
        t = f"{secs:.0f}s" if secs < 90 else f"{secs / 60:.1f} min"
        rows.append(f"| {m} | {r['accuracy'] * 100:.1f}% | {r['macro_f1']:.3f} | {pc['negative']['f1']:.3f} | "
                    f"{pc['neutral']['f1']:.3f} | {pc['positive']['f1']:.3f} | {t} |")
    return "\n".join(rows)


def ablation_table(results: dict) -> str:
    rows = ["| Model | Accuracy, all agree | Macro-F1, all agree | n |", "| --- | --- | --- | --- |"]
    for m in MODEL_NAMES:
        ab = results.get(m, {}).get("ablation")
        if ab:
            rows.append(f"| {m} | {ab['accuracy'] * 100:.1f}% | {ab['macro_f1']:.3f} | {ab['n']} |")
    return "\n".join(rows) if len(rows) > 2 else ""


_BLOCK = re.compile(r"(<!-- results:start -->)\n?.*?\n?(<!-- results:end -->)", re.S)  # empty block allowed


def update_readme_results(readme: Path, table: str) -> None:
    txt = readme.read_text(encoding="utf-8")
    if not _BLOCK.search(txt):
        raise ValueError(f"{readme} has no <!-- results:start --> / <!-- results:end --> block")
    readme.write_text(_BLOCK.sub(lambda m: m.group(1) + "\n" + table + "\n" + m.group(2), txt), encoding="utf-8")


@app.command()
def results():
    """Print the model comparison from results.json and write it into the README block."""
    r = load_results()
    if not r:
        raise typer.Exit("No results.json yet; run `train` on PhraseBank first.")
    table = results_table(r)
    ab = ablation_table(r)
    block = table + (f"\n\nOn the sentences every annotator agreed on (test split only):\n\n{ab}" if ab else "")
    typer.echo(block)
    if README.exists():
        update_readme_results(README, block)
        typer.echo(f"README block updated: {README}")


# ---------------------------------------------------------------- scoring ------------
@app.command()
def score(text: str = typer.Option(..., help="One sentence."), model: str = typer.Option(None)):
    """Class probabilities for one sentence from the best (or named) trained model."""
    model = model or best_available_model()
    p = load_scorer(model)([text])[0]
    typer.echo(f"{model}: " + "  ".join(f"{name} {v:.3f}" for name, v in zip(LABELS, p)) + f"  -> {LABELS[int(p.argmax())]}")


def build_snapshot(tickers: list[str], scorer: Scorer, fetcher: TenKFetcher, model_name: str, run_id: str) -> dict:
    """Pure assembly: fetch each 10-K, cut the two sections, score the sentences, aggregate."""
    companies, skipped = {}, {}
    for t in tickers:
        t = t.upper()
        try:
            f = fetcher.latest_10k(t)
        except (LookupError, FileNotFoundError) as exc:
            skipped[t] = str(exc)
            continue
        sections, warnings = extract_sections(html_to_text(f.html))
        scored = {}
        for key, body in sections.items():
            sents = split_sentences(body)
            scored[key] = filing_tone(scorer(sents) if sents else np.zeros((0, 3)), sents)
        companies[t] = {"company": f.company, "cik": f.cik, "form": f.form, "accession": f.accession,
                        "report_date": f.report_date, "filed": f.filed, "document_url": f.url,
                        "sections": scored, "warnings": warnings}
    return {"date": date.today().isoformat(), "model": model_name, "run_id": run_id,
            "pulled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "training_data": "Financial PhraseBank (Malo et al., 2014), sentences_50agree",
            "net_tone_definition": "share of sentences classified positive minus share classified negative",
            "companies": companies, "skipped": skipped}


def load_snapshot(path: Path = SNAPSHOT) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def peer_median(snap: dict, section: str, exclude: str = "AEHR") -> float | None:
    vals = [c["sections"][section]["net_tone"] for t, c in snap["companies"].items()
            if t != exclude and c["sections"][section]["n"] > 0]
    return median(vals) if vals else None


@app.command()
def snapshot(tickers: str = typer.Option(",".join(DEFAULT_TICKERS)), model: str = typer.Option(None),
             out: Path = typer.Option(SNAPSHOT)):
    """Score the latest 10-K of each ticker and write the snapshot the app reads (live unless FINLIB_OFFLINE=1)."""
    model = model or best_available_model()
    runs = sorted(RUNS.glob(f"{model}-*.json"))
    run_id = runs[-1].stem if runs else model
    snap = build_snapshot(tickers.split(","), load_scorer(model), TenKFetcher(), model_name=model, run_id=run_id)
    out.write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_snapshot(snap)
    typer.echo(f"wrote {out}")


def _print_snapshot(snap: dict) -> None:
    typer.echo(f"Filing tone, {snap['date']}, model {snap['model']} ({snap['run_id']})")
    typer.echo(f"{'ticker':<7}{'fiscal year end':<17}{'MD&A tone':>10}{'n':>6}{'risk tone':>11}{'n':>6}")
    for t, c in snap["companies"].items():
        m, r = c["sections"]["mdna"], c["sections"]["risk_factors"]
        typer.echo(f"{t:<7}{c['report_date']:<17}{m['net_tone']:>+10.2f}{m['n']:>6}{r['net_tone']:>+11.2f}{r['n']:>6}")
        for w in c["warnings"]:
            typer.echo(f"        warning: {w}")
    for t, why in snap["skipped"].items():
        typer.echo(f"{t:<7}skipped: {why}")


@app.command(name="summary")
def summary_cmd(path: Path = typer.Option(SNAPSHOT)):
    """Print the committed snapshot."""
    _print_snapshot(load_snapshot(path))


if __name__ == "__main__":
    app()
