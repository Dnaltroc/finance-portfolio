"""Labeled sentences in, three lists out.

A *label* is the answer we want the model to learn. Here it is whether a finance sentence
reads positive, negative, or neutral. Financial PhraseBank (Malo et al., 2014) holds about
4,840 such sentences from financial news, each labeled by people with finance backgrounds,
published at four agreement levels (50%, 66%, 75%, 100% of annotators agree). Offline, a
90-sentence synthetic fixture stands in so tests never download anything.

The three splits:
    train       what the model learns from
    validation  checked after every epoch, decides when to stop; never trained on
    test        touched once, at the end, for the number we report
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

LABELS = ["negative", "neutral", "positive"]  # class id = position in this list (matches the HF dataset)
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
FIXTURE_CSV = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "filing_tone" / "sentences.csv"
PHRASEBANK_CACHE = Path("output") / "filing_tone" / "phrasebank"
PHRASEBANK_ID = "takala/financial_phrasebank"


@dataclass(frozen=True)
class Example:
    text: str
    label: int


def load_fixture(path: Path = FIXTURE_CSV) -> list[Example]:
    """The synthetic fixture: sentence,label,source with source == 'synthetic' on every row."""
    with path.open(encoding="utf-8", newline="") as f:
        return [Example(r["sentence"], LABEL_TO_ID[r["label"]]) for r in csv.DictReader(f)]


def load_phrasebank(config: str = "sentences_50agree", cache_dir: Path = PHRASEBANK_CACHE) -> list[Example]:
    """Financial PhraseBank from the Hugging Face hub (downloaded once, cached under output/).
    HF label ids are 0 negative, 1 neutral, 2 positive, the same order as LABELS."""
    from datasets import load_dataset  # imported here so the module works without the ml extra

    ds = load_dataset(PHRASEBANK_ID, config, split="train", cache_dir=str(cache_dir))
    return [Example(r["sentence"], int(r["label"])) for r in ds]


def label_counts(examples) -> dict[str, int]:
    c = Counter(LABELS[e.label] for e in examples)
    return {name: c.get(name, 0) for name in LABELS}


def stratified_split(examples, seed: int = 42, fracs=(0.8, 0.1, 0.1)):
    """Shuffle within each class, then cut each class by the same fractions, so every split
    has the same label mix as the whole. Deterministic for a given seed."""
    rng = random.Random(seed)
    parts: list[list[Example]] = [[], [], []]
    for cls in range(len(LABELS)):
        rows = [e for e in examples if e.label == cls]
        rng.shuffle(rows)
        n = len(rows)
        n_train, n_val = round(n * fracs[0]), round(n * fracs[1])
        parts[0] += rows[:n_train]
        parts[1] += rows[n_train:n_train + n_val]
        parts[2] += rows[n_train + n_val:]
    for p in parts:
        rng.shuffle(p)
    return tuple(parts)
