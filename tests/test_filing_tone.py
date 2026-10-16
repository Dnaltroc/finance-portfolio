"""Filing Tone: the PyTorch financial sentiment tool.

Pure-Python tests (data, tokenizer, metrics, 10-K text, snapshot) always run. Tests that need
torch or scikit-learn skip when the `ml` extra is not installed; .github/workflows/ml.yml runs
them. Everything here is offline: the 90-sentence fixture is synthetic, and the fixture 10-K is
a made-up document shaped like a real one."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from finlib.nlp.data import LABELS, Example, label_counts, load_fixture, stratified_split

FIX = Path(__file__).parent / "fixtures" / "filing_tone"


# ---------------------------------------------------------------- data ---------------
def test_fixture_is_balanced_and_labeled_synthetic():
    ex = load_fixture()
    assert len(ex) == 90 and label_counts(ex) == {"negative": 30, "neutral": 30, "positive": 30}
    assert all(isinstance(e, Example) and e.label in (0, 1, 2) for e in ex)
    assert LABELS == ["negative", "neutral", "positive"]
    assert "synthetic" in (FIX / "sentences.csv").read_text(encoding="utf-8").splitlines()[1]


def test_stratified_split_keeps_class_balance_and_is_deterministic():
    ex = load_fixture()
    a = stratified_split(ex, seed=1)
    b = stratified_split(ex, seed=1)
    assert [e.text for e in a[0]] == [e.text for e in b[0]]
    assert [len(s) for s in a] == [72, 9, 9]
    assert label_counts(a[2]) == {"negative": 3, "neutral": 3, "positive": 3}
    assert not {e.text for e in a[0]} & {e.text for e in a[2]}
