# Filing Tone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A PyTorch financial-sentiment tool that trains three model tiers on Financial PhraseBank, applies the best one to real 10-K text, and ships as tool #8 with a Streamlit page, tests, CI, and a learner's walkthrough.

**Architecture:** New `finlib/nlp/` package holds data loading, tokenizer/vocab, models, a shared training loop, hand-written metrics, a TF-IDF baseline, and SEC 10-K text extraction. `projects/filing_tone/cli.py` orchestrates runs and writes a committed `tone_snapshot.json` that the torch-free Streamlit page reads.

**Tech Stack:** Python 3.11+, PyTorch 2.x, Hugging Face `transformers` + `datasets`, scikit-learn (baseline and test oracle), NumPy, Typer, Streamlit, Plotly.

**Spec:** `docs/superpowers/specs/2026-10-16-filing-tone-design.md`

## Global Constraints

- `finlib/models/` and `finlib/providers/` are not modified.
- New dependencies only in an optional extra `ml = ["torch>=2.3", "transformers>=4.44", "datasets>=2.20", "scikit-learn>=1.4"]`; never in `requirements.txt` or the Dockerfile.
- No torch import under `app/`.
- Every test runs with `FINLIB_OFFLINE=1`, no network; torch tests use `pytest.importorskip("torch")`.
- Labels are `["negative", "neutral", "positive"]`, ids 0/1/2 (matches the HF dataset).
- Fixtures are labeled synthetic in their header.
- Never fabricate a figure: README results come from `results.json`, written by code.
- Follow DESIGN.md for the page: eyebrow `06 · Filing tone`, gray provenance badge, exhibit labels, source captions.
- Commit per task with a clear message ending in the Co-Authored-By line.

## Review Focus

1. A 10-K whose Item 1A heading appears only in the table of contents and whose body says "Item 1A. Risk Factors" with a non-breaking space: extraction must still find the body section (Task 6 test `test_extract_sections_prefers_the_body_over_the_toc`).
2. A sentence longer than the CNN `max_len` or DistilBERT `max_length`: it must be truncated, not crash (Task 4 test `test_vocab_encoder_truncates_to_max_len`).
3. An empty sentence list handed to `filing_tone` (a filing with no MD&A found): return zeros and empty drivers, not a ZeroDivisionError (Task 6 test `test_filing_tone_handles_no_sentences`).
4. A ticker not in the SEC table or a company with no 10-K (CAMT files 20-F): `latest_10k` raises a clear `LookupError` and `snapshot` skips it with a note in the JSON (Task 7 test `test_latest_10k_offline_unknown_ticker_raises`, Task 8 `snapshot` loop).
5. Training on a machine without CUDA: `choose_device()` falls back to CPU and the run log records it (Task 5 test `test_choose_device_falls_back_to_cpu`).

---

### Task 1: Package scaffold, dependencies, fixtures, data loading

**Files:**
- Create: `finlib/nlp/__init__.py`, `finlib/nlp/data.py`
- Create: `tests/fixtures/filing_tone/sentences.csv`, `tests/fixtures/filing_tone/sample_10k.html`
- Create: `tests/test_filing_tone.py`
- Modify: `pyproject.toml` (add `ml` extra)

**Interfaces:**
- Produces: `LABELS: list[str]`, `LABEL_TO_ID: dict[str,int]`, `Example(text: str, label: int)`, `load_fixture(path=FIXTURE_CSV) -> list[Example]`, `load_phrasebank(config: str = "sentences_50agree", cache_dir: Path = PHRASEBANK_CACHE) -> list[Example]`, `stratified_split(examples, seed=42, fracs=(0.8, 0.1, 0.1)) -> tuple[list, list, list]`, `label_counts(examples) -> dict[str,int]`.

- [ ] **Step 1: Add the extra to pyproject.toml**

```toml
ml = ["torch>=2.3", "transformers>=4.44", "datasets>=2.20", "scikit-learn>=1.4"]
```

- [ ] **Step 2: Write the fixture CSV** (`tests/fixtures/filing_tone/sentences.csv`, first line a comment-style header row is not allowed in CSV, so put the synthetic label in a column):

```csv
sentence,label,source
"Revenue for the quarter rose 18% to $412 million, ahead of guidance.",positive,synthetic
...30 positive, 30 negative, 30 neutral, all written for this fixture...
```

- [ ] **Step 3: Write the fixture 10-K** (`sample_10k.html`): a small HTML document with a table of contents listing `Item 1A. Risk Factors` and `Item 7. Management's Discussion`, then body sections with the same headings (one using `&#160;`), 8 to 10 sentences each, including "Inc.", "U.S." and "3.5%" to exercise the splitter, then `Item 1B`, `Item 7A`, `Item 8` headings to end the sections.

- [ ] **Step 4: Write failing tests**

```python
from finlib.nlp.data import LABELS, Example, label_counts, load_fixture, stratified_split

def test_fixture_is_balanced_and_labeled_synthetic():
    ex = load_fixture()
    assert len(ex) == 90 and label_counts(ex) == {"negative": 30, "neutral": 30, "positive": 30}
    assert all(isinstance(e, Example) and e.label in (0, 1, 2) for e in ex)

def test_stratified_split_keeps_class_balance_and_is_deterministic():
    ex = load_fixture()
    a = stratified_split(ex, seed=1)
    b = stratified_split(ex, seed=1)
    assert [e.text for e in a[0]] == [e.text for e in b[0]]
    assert [len(s) for s in a] == [72, 9, 9]
    assert label_counts(a[2]) == {"negative": 3, "neutral": 3, "positive": 3}
    assert not {e.text for e in a[0]} & {e.text for e in a[2]}
```

- [ ] **Step 5: Run, expect ImportError**: `pytest tests/test_filing_tone.py -q`

- [ ] **Step 6: Implement `finlib/nlp/data.py`**

```python
"""Labeled sentences in, three lists out.

A *label* is the answer we want the model to learn: here, whether a finance sentence reads
positive, negative, or neutral. Financial PhraseBank (Malo et al., 2014) has ~4,840 such
sentences labeled by finance professionals. Offline, a 90-sentence synthetic fixture stands
in so tests never download anything.
"""
from __future__ import annotations
import csv, random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

LABELS = ["negative", "neutral", "positive"]      # class id = position in this list
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
FIXTURE_CSV = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "filing_tone" / "sentences.csv"
PHRASEBANK_CACHE = Path("output") / "filing_tone" / "phrasebank"
PHRASEBANK_ID = "takala/financial_phrasebank"

@dataclass(frozen=True)
class Example:
    text: str
    label: int

def load_fixture(path: Path = FIXTURE_CSV) -> list[Example]:
    with path.open(encoding="utf-8", newline="") as f:
        return [Example(r["sentence"], LABEL_TO_ID[r["label"]]) for r in csv.DictReader(f)]

def load_phrasebank(config: str = "sentences_50agree", cache_dir: Path = PHRASEBANK_CACHE) -> list[Example]:
    from datasets import load_dataset   # imported here so the module works without it
    ds = load_dataset(PHRASEBANK_ID, config, split="train", cache_dir=str(cache_dir))
    return [Example(r["sentence"], int(r["label"])) for r in ds]   # HF ids: 0 neg, 1 neu, 2 pos

def label_counts(examples) -> dict[str, int]:
    c = Counter(LABELS[e.label] for e in examples)
    return {name: c.get(name, 0) for name in LABELS}

def stratified_split(examples, seed: int = 42, fracs=(0.8, 0.1, 0.1)):
    """Shuffle within each class, then cut each class by the same fractions, so every split
    has the same label mix. Deterministic for a given seed."""
    rng = random.Random(seed)
    parts: list[list[Example]] = [[], [], []]
    for cls in range(len(LABELS)):
        rows = [e for e in examples if e.label == cls]
        rng.shuffle(rows)
        n = len(rows); n_train = round(n * fracs[0]); n_val = round(n * fracs[1])
        parts[0] += rows[:n_train]; parts[1] += rows[n_train:n_train + n_val]; parts[2] += rows[n_train + n_val:]
    for p in parts:
        rng.shuffle(p)
    return tuple(parts)
```

- [ ] **Step 7: Run tests, expect pass.** Then `ruff check .`
- [ ] **Step 8: Commit** `Filing Tone: package scaffold, ml extra, synthetic fixtures, data split`

### Task 2: Tokenizer and vocabulary

**Files:** Create `finlib/nlp/tokenize.py`; Test `tests/test_filing_tone.py`

**Interfaces:** `RegexTokenizer.__call__(text) -> list[str]`; `Vocab(tokens: list[str])` with `PAD_ID = 0`, `UNK_ID = 1`, `Vocab.build(texts, tokenizer, min_freq=1, max_size=None) -> Vocab`, `encode(text) -> list[int]`, `decode(ids) -> list[str]`, `__len__`, `itos: list[str]`.

- [ ] **Step 1: Tests**

```python
from finlib.nlp.tokenize import RegexTokenizer, Vocab

def test_tokenizer_lowercases_splits_punctuation_keeps_numbers():
    tok = RegexTokenizer()
    assert tok("Revenue rose 18.5% to $412M, Inc.") == ["revenue", "rose", "18.5%", "to", "$", "412", "m", ",", "inc", "."]

def test_vocab_round_trips_and_maps_unknowns():
    tok = RegexTokenizer()
    v = Vocab.build(["profit rose", "profit fell"], tok, min_freq=1)
    assert v.itos[:2] == ["<pad>", "<unk>"] and Vocab.PAD_ID == 0 and Vocab.UNK_ID == 1
    assert v.decode(v.encode("profit rose")) == ["profit", "rose"]
    assert v.encode("profit soared") == [v.stoi["profit"], Vocab.UNK_ID]
    assert len(v) == 5

def test_vocab_min_freq_and_max_size_cap_the_vocab():
    tok = RegexTokenizer()
    v = Vocab.build(["a a a b b c"], tok, min_freq=2)
    assert "c" not in v.stoi and "b" in v.stoi
    v2 = Vocab.build(["a a a b b c"], tok, max_size=1)
    assert v2.itos == ["<pad>", "<unk>", "a"]
```

- [ ] **Step 2: Run, expect ImportError.**
- [ ] **Step 3: Implement**

```python
"""Text to integers.

A neural network cannot read letters; it reads numbers. A *tokenizer* chops a sentence into
pieces (tokens) and a *vocabulary* gives each distinct token an integer id. Two ids are
reserved: 0 pads short sentences out to a common length inside a batch, 1 stands for any
token the training data never showed us.
"""
from __future__ import annotations
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?%?|[^\sa-z\d]")

class RegexTokenizer:
    def __call__(self, text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())

class Vocab:
    PAD, UNK = "<pad>", "<unk>"
    PAD_ID, UNK_ID = 0, 1

    def __init__(self, tokens: list[str]):
        self.itos = [self.PAD, self.UNK] + list(tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.tokenizer = RegexTokenizer()

    @classmethod
    def build(cls, texts, tokenizer: RegexTokenizer, min_freq: int = 1, max_size: int | None = None) -> "Vocab":
        counts = Counter(t for text in texts for t in tokenizer(text))
        ranked = sorted((t for t, c in counts.items() if c >= min_freq), key=lambda t: (-counts[t], t))
        return cls(ranked[:max_size] if max_size else ranked)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(t, self.UNK_ID) for t in self.tokenizer(text)]

    def decode(self, ids) -> list[str]:
        return [self.itos[i] for i in ids]

    def __len__(self) -> int:
        return len(self.itos)
```

- [ ] **Step 4: Run tests, pass. Commit** `Filing Tone: regex tokenizer and vocabulary`

### Task 3: Metrics by hand, checked against scikit-learn

**Files:** Create `finlib/nlp/evaluate.py`; Test `tests/test_filing_tone.py`

**Interfaces:** `confusion_matrix(y_true, y_pred, n_classes=3) -> np.ndarray` (rows = true), `precision_recall_f1(cm) -> tuple[np.ndarray, np.ndarray, np.ndarray]`, `summary(y_true, y_pred, labels=LABELS) -> dict` with keys `accuracy`, `macro_f1`, `per_class: {name: {precision, recall, f1, support}}`, `confusion: list[list[int]]`, `n`.

- [ ] **Step 1: Tests**

```python
import numpy as np
from finlib.nlp.evaluate import confusion_matrix, precision_recall_f1, summary

Y_TRUE = [0, 0, 1, 1, 1, 2, 2, 2, 2, 0]
Y_PRED = [0, 1, 1, 1, 2, 2, 2, 0, 2, 0]

def test_confusion_matrix_rows_are_truth():
    cm = confusion_matrix(Y_TRUE, Y_PRED)
    assert cm.tolist() == [[2, 1, 0], [0, 2, 1], [1, 0, 3]]

def test_metrics_match_sklearn_oracle():
    sk = pytest.importorskip("sklearn.metrics")
    s = summary(Y_TRUE, Y_PRED)
    assert s["accuracy"] == pytest.approx(sk.accuracy_score(Y_TRUE, Y_PRED))
    assert s["macro_f1"] == pytest.approx(sk.f1_score(Y_TRUE, Y_PRED, average="macro"))
    p, r, f, _ = sk.precision_recall_fscore_support(Y_TRUE, Y_PRED)
    for i, name in enumerate(LABELS):
        assert s["per_class"][name]["f1"] == pytest.approx(f[i])
        assert s["per_class"][name]["precision"] == pytest.approx(p[i])

def test_f1_is_zero_not_nan_for_a_class_never_predicted():
    cm = confusion_matrix([0, 1, 2], [0, 0, 0])
    _, _, f = precision_recall_f1(cm)
    assert f[1] == 0.0 and f[2] == 0.0
```

- [ ] **Step 2: Implement**

```python
"""Scores from scratch so the numbers are understood, not just imported.

Accuracy: share of sentences the model got right. It hides problems when one class
dominates (PhraseBank is ~60% neutral), so we also report *macro-F1*: F1 for each class,
averaged with equal weight. F1 balances precision (of the sentences we called positive, how
many were) and recall (of the truly positive sentences, how many we caught).
"""
from __future__ import annotations
import numpy as np
from finlib.nlp.data import LABELS

def confusion_matrix(y_true, y_pred, n_classes: int = 3) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def precision_recall_f1(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    pred_pos = cm.sum(axis=0).astype(float); true_pos = cm.sum(axis=1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(pred_pos > 0, tp / pred_pos, 0.0)
        r = np.where(true_pos > 0, tp / true_pos, 0.0)
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return p, r, f

def summary(y_true, y_pred, labels=LABELS) -> dict:
    cm = confusion_matrix(y_true, y_pred, len(labels))
    p, r, f = precision_recall_f1(cm)
    n = int(cm.sum())
    return {
        "n": n,
        "accuracy": float(np.trace(cm) / n) if n else 0.0,
        "macro_f1": float(f.mean()),
        "per_class": {name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                             "support": int(cm[i].sum())} for i, name in enumerate(labels)},
        "confusion": cm.tolist(),
    }
```

- [ ] **Step 3: Run, pass. Commit** `Filing Tone: confusion matrix and macro-F1 by hand, tested against scikit-learn`

### Task 4: Datasets, batch encoders, and the TextCNN

**Files:** Create `finlib/nlp/models.py`, `finlib/nlp/batches.py`; Test `tests/test_filing_tone.py` (torch section, guarded by `torch = pytest.importorskip("torch")` inside each test or a module-level helper).

**Interfaces:**
- `batches.py`: `SentenceDataset(examples) -> (text, label)`; `vocab_encoder(vocab, max_len=64) -> Callable[[list[str]], dict[str, Tensor]]` returning `input_ids` (B,T) long and `attention_mask` (B,T) long; `hf_encoder(tokenizer, max_len=128)` same contract; `make_collate(encode_batch) -> collate_fn` adding `labels` (B,) long; `make_loader(examples, encode_batch, batch_size, shuffle, seed=None) -> DataLoader`.
- `models.py`: `TextCNN(vocab_size, embed_dim=100, num_classes=3, kernel_sizes=(3,4,5), channels=100, dropout=0.5, pad_id=0)` with `forward(input_ids, attention_mask=None) -> logits (B, C)`; `HFClassifier(model)` wrapping a HF sequence classifier with the same `forward` contract; `build_distilbert(num_labels=3, name="distilbert-base-uncased") -> (tokenizer, HFClassifier)`.

- [ ] **Step 1: Tests**

```python
def test_vocab_encoder_pads_to_longest_and_masks():
    torch = pytest.importorskip("torch")
    from finlib.nlp.batches import make_collate, vocab_encoder
    v = Vocab.build(["profit rose sharply", "loss"], RegexTokenizer())
    collate = make_collate(vocab_encoder(v, max_len=10))
    b = collate([("profit rose sharply", 2), ("loss", 0)])
    assert b["input_ids"].shape == (2, 3) and b["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert b["input_ids"][1].tolist() == [v.stoi["loss"], 0, 0] and b["labels"].tolist() == [2, 0]

def test_vocab_encoder_truncates_to_max_len():
    pytest.importorskip("torch")
    from finlib.nlp.batches import vocab_encoder
    v = Vocab.build(["a b c d e f"], RegexTokenizer())
    out = vocab_encoder(v, max_len=4)(["a b c d e f"])
    assert out["input_ids"].shape == (1, 4)

def test_textcnn_forward_shape_and_short_inputs():
    torch = pytest.importorskip("torch")
    from finlib.nlp.models import TextCNN
    m = TextCNN(vocab_size=50)
    ids = torch.randint(2, 50, (4, 2))          # shorter than the widest kernel (5)
    assert m(ids).shape == (4, 3)
```

- [ ] **Step 2: Implement `batches.py`**

```python
"""From a list of sentences to the tensors a model consumes.

A *Dataset* hands out one example at a time. A *DataLoader* groups them into *batches*
(the model updates its weights once per batch, not once per sentence) and calls a *collate*
function to stack them into one tensor. Sentences differ in length, so the short ones are
padded with id 0 and an *attention mask* records which positions are real.
"""
from __future__ import annotations
from typing import Callable
import torch
from torch.utils.data import DataLoader, Dataset
from finlib.nlp.data import Example
from finlib.nlp.tokenize import Vocab

Encoder = Callable[[list[str]], dict[str, torch.Tensor]]

class SentenceDataset(Dataset):
    def __init__(self, examples: list[Example]):
        self.examples = examples
    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i].text, self.examples[i].label

def vocab_encoder(vocab: Vocab, max_len: int = 64) -> Encoder:
    def encode_batch(texts: list[str]) -> dict[str, torch.Tensor]:
        ids = [vocab.encode(t)[:max_len] or [Vocab.UNK_ID] for t in texts]
        width = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), width), Vocab.PAD_ID, dtype=torch.long)
        mask = torch.zeros((len(ids), width), dtype=torch.long)
        for row, x in enumerate(ids):
            input_ids[row, :len(x)] = torch.tensor(x); mask[row, :len(x)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}
    return encode_batch

def hf_encoder(tokenizer, max_len: int = 128) -> Encoder:
    def encode_batch(texts: list[str]) -> dict[str, torch.Tensor]:
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    return encode_batch

def make_collate(encode_batch: Encoder):
    def collate(items):
        texts, labels = zip(*items)
        batch = encode_batch(list(texts))
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch
    return collate

def make_loader(examples, encode_batch, batch_size: int, shuffle: bool, seed: int | None = None) -> DataLoader:
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(SentenceDataset(examples), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=make_collate(encode_batch), generator=gen)
```

- [ ] **Step 3: Implement `models.py`**

```python
"""Two classifiers with one calling convention: forward(input_ids, attention_mask) -> logits.

TextCNN, built from scratch: an *embedding* turns each token id into a learned vector
(words used alike end up near each other). 1-D *convolutions* slide windows of 3, 4 and 5
tokens over the sentence and light up on phrases like "fell short of"; max-pooling keeps the
strongest signal per filter wherever it occurred; a linear layer maps those to three scores
(*logits*), one per class. Dropout randomly zeroes activations during training so no single
feature is relied on too much.

DistilBERT, pre-trained by others on a large text corpus, already knows English. We attach
a fresh 3-way head and *fine-tune* every weight on our 4k sentences.
"""
from __future__ import annotations
import torch
from torch import nn

class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 100, num_classes: int = 3,
                 kernel_sizes=(3, 4, 5), channels: int = 100, dropout: float = 0.5, pad_id: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        # padding=k-1 so a sentence shorter than the window still yields one output position
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, channels, k, padding=k - 1) for k in kernel_sizes])
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(channels * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embed(input_ids).transpose(1, 2)                      # (B, T, E) -> (B, E, T)
        pooled = [torch.relu(conv(x)).max(dim=2).values for conv in self.convs]   # each (B, channels)
        return self.out(self.drop(torch.cat(pooled, dim=1)))

class HFClassifier(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

def build_distilbert(num_labels: int = 3, name: str = "distilbert-base-uncased"):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)
    return tok, HFClassifier(model)
```

- [ ] **Step 4: Run, pass, ruff. Commit** `Filing Tone: datasets, padding collate, TextCNN and DistilBERT wrapper`

### Task 5: Training loop, run log, baseline

**Files:** Create `finlib/nlp/train.py`, `finlib/nlp/baseline.py`; Test `tests/test_filing_tone.py`

**Interfaces:**
- `TrainConfig(model: str, epochs=20, batch_size=32, lr=1e-3, weight_decay=0.01, warmup_frac=0.1, patience=3, max_grad_norm=1.0, seed=42, class_weights=True, max_len=64, device=None)` dataclass with `to_dict()`.
- `set_seed(seed)`, `choose_device(pref: str | None = None) -> torch.device`, `class_weight_tensor(labels: list[int], n_classes=3) -> Tensor`.
- `RunLog` dataclass: `model, config, seed, device, started, history: list[dict], test: dict | None, ablation: dict | None, wall_seconds, train_size, val_size, test_size, vocab_size, run_id`; `save(path)`, `load(path)`.
- `train_loop(model, train_loader, val_loader, cfg, device, log) -> nn.Module` (best val macro-F1 weights restored).
- `predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]` labels, probs (N, C).
- `predict_texts(model, encode_batch, texts, device, batch_size=64) -> np.ndarray` probs.
- `baseline.py`: `fit_baseline(texts, labels, seed=42) -> Pipeline`, `baseline_probs(pipe, texts) -> np.ndarray`.

- [ ] **Step 1: Tests**

```python
def test_choose_device_falls_back_to_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    from finlib.nlp.train import choose_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert choose_device().type == "cpu"
    assert choose_device("cpu").type == "cpu"

def test_class_weights_are_inverse_frequency_normalised():
    torch = pytest.importorskip("torch")
    from finlib.nlp.train import class_weight_tensor
    w = class_weight_tensor([0, 1, 1, 1, 2, 2])          # counts 1, 3, 2
    assert w.tolist() == pytest.approx([2.0, 2 / 3, 1.0])   # n / (C * count)

def test_textcnn_learns_the_fixture(tmp_path):
    torch = pytest.importorskip("torch")
    from finlib.nlp.batches import make_loader, vocab_encoder
    from finlib.nlp.models import TextCNN
    from finlib.nlp.train import RunLog, TrainConfig, predict, set_seed, train_loop
    ex = load_fixture(); train, val, _ = stratified_split(ex, seed=0)
    cfg = TrainConfig(model="cnn", epochs=40, batch_size=16, lr=3e-3, patience=40, device="cpu")
    set_seed(cfg.seed)
    vocab = Vocab.build([e.text for e in train], RegexTokenizer())
    enc = vocab_encoder(vocab, cfg.max_len)
    model = TextCNN(len(vocab), embed_dim=32, channels=16)
    log = RunLog.start(cfg, device="cpu")
    model = train_loop(model, make_loader(train, enc, 16, True, seed=0), make_loader(val, enc, 32, False),
                       cfg, torch.device("cpu"), log)
    y, probs = predict(model, make_loader(train, enc, 32, False), torch.device("cpu"))
    assert (probs.argmax(1) == y).mean() > 0.95           # it can memorize 72 sentences
    assert probs.shape == (72, 3) and np.allclose(probs.sum(1), 1)
    assert len(log.history) >= 1 and "val_macro_f1" in log.history[0]
    log.test = {"accuracy": 1.0}; log.save(tmp_path / "run.json")
    assert RunLog.load(tmp_path / "run.json").config["epochs"] == 40

def test_baseline_fits_and_predicts_probabilities():
    pytest.importorskip("sklearn")
    from finlib.nlp.baseline import baseline_probs, fit_baseline
    ex = load_fixture()
    pipe = fit_baseline([e.text for e in ex], [e.label for e in ex])
    p = baseline_probs(pipe, ["Revenue rose 18% to a record."])
    assert p.shape == (1, 3) and p.sum() == pytest.approx(1.0)
```

- [ ] **Step 2: Implement `train.py`**

```python
"""The training loop, shared by the CNN and DistilBERT.

One *epoch* is one pass over the training set. For each batch: run the model (forward),
measure how wrong it was with *cross-entropy loss*, compute the gradient of that loss with
respect to every weight (backward), and nudge the weights a small step against the gradient
(the optimizer, AdamW). The *learning rate* is the step size: it warms up from zero so the
first noisy batches do not throw the weights around, then decays linearly.

After each epoch we score the *validation* set, which the model never trains on. If
validation macro-F1 stops improving for `patience` epochs we stop (*early stopping*) and
restore the best weights: training longer would only memorize the training set.
"""
from __future__ import annotations
import copy, json, random, time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from finlib.nlp.evaluate import summary

@dataclass
class TrainConfig:
    model: str
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    patience: int = 3
    max_grad_norm: float = 1.0
    seed: int = 42
    class_weights: bool = True
    max_len: int = 64
    device: str | None = None
    def to_dict(self) -> dict: return asdict(self)

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def choose_device(pref: str | None = None) -> torch.device:
    if pref: return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def class_weight_tensor(labels, n_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(float)
    return torch.tensor(len(labels) / (n_classes * np.maximum(counts, 1)), dtype=torch.float)

@dataclass
class RunLog:
    run_id: str; model: str; config: dict; seed: int; device: str; started: str
    train_size: int = 0; val_size: int = 0; test_size: int = 0; vocab_size: int | None = None
    history: list[dict] = field(default_factory=list)
    test: dict | None = None; ablation: dict | None = None; wall_seconds: float = 0.0
    @classmethod
    def start(cls, cfg: TrainConfig, device: str) -> "RunLog":
        now = datetime.now(timezone.utc)
        return cls(run_id=f"{cfg.model}-{now:%Y%m%d-%H%M%S}", model=cfg.model, config=cfg.to_dict(),
                   seed=cfg.seed, device=device, started=now.isoformat(timespec="seconds"))
    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8"); return path
    @classmethod
    def load(cls, path: Path) -> "RunLog": return cls(**json.loads(path.read_text(encoding="utf-8")))

def _to(batch: dict, device: torch.device) -> dict: return {k: v.to(device) for k, v in batch.items()}

@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device):
    model.eval(); ys, ps = [], []
    for batch in loader:
        b = _to(batch, device); labels = b.pop("labels")
        ps.append(torch.softmax(model(**b), dim=1).cpu().numpy()); ys.append(labels.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)

@torch.no_grad()
def predict_texts(model: nn.Module, encode_batch, texts: list[str], device: torch.device, batch_size: int = 64) -> np.ndarray:
    model.eval(); out = []
    for i in range(0, len(texts), batch_size):
        b = _to(encode_batch(texts[i:i + batch_size]), device)
        out.append(torch.softmax(model(**b), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 3))

def train_loop(model: nn.Module, train_loader, val_loader, cfg: TrainConfig, device: torch.device, log: RunLog) -> nn.Module:
    model.to(device)
    labels = [int(l) for _, l in train_loader.dataset]
    weight = class_weight_tensor(labels).to(device) if cfg.class_weights else None
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.epochs * len(train_loader); warm = max(1, int(cfg.warmup_frac * total))
    sched = LambdaLR(opt, lambda s: s / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    best_f1, best_state, since_best, t0 = -1.0, copy.deepcopy(model.state_dict()), 0, time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train(); running, seen = 0.0, 0
        for batch in train_loader:
            b = _to(batch, device); y = b.pop("labels")
            loss = loss_fn(model(**b), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step(); sched.step()
            running += loss.item() * len(y); seen += len(y)
        y_val, p_val = predict(model, val_loader, device)
        s = summary(y_val, p_val.argmax(1))
        log.history.append({"epoch": epoch, "train_loss": running / max(seen, 1), "val_accuracy": s["accuracy"],
                            "val_macro_f1": s["macro_f1"], "lr": sched.get_last_lr()[0], "seconds": time.time() - t0})
        if s["macro_f1"] > best_f1:
            best_f1, best_state, since_best = s["macro_f1"], copy.deepcopy(model.state_dict()), 0
        else:
            since_best += 1
            if since_best >= cfg.patience: break
    model.load_state_dict(best_state); log.wall_seconds = time.time() - t0
    return model
```

- [ ] **Step 3: Implement `baseline.py`**

```python
"""The floor: bag-of-words TF-IDF features into logistic regression. No neural network.
If the PyTorch models cannot beat this, they are not earning their complexity."""
from __future__ import annotations
import numpy as np

def fit_baseline(texts, labels, seed: int = 42):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                     ("lr", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=seed))])
    return pipe.fit(list(texts), list(labels))

def baseline_probs(pipe, texts) -> np.ndarray:
    return pipe.predict_proba(list(texts))
```

- [ ] **Step 4: Run, pass, ruff. Commit** `Filing Tone: training loop with warmup, early stopping and run log; TF-IDF baseline`

### Task 6: 10-K text: HTML to text, section extraction, sentence splitting, tone

**Files:** Create `finlib/nlp/sec_text.py`; Test `tests/test_filing_tone.py` (pure Python, always runs)

**Interfaces:** `html_to_text(html: str) -> str`; `extract_sections(text: str) -> tuple[dict[str, str], list[str]]` with keys `risk_factors`, `mdna`; `split_sentences(text: str, min_words=6, max_words=120) -> list[str]`; `filing_tone(probs: np.ndarray, sentences: list[str], top_k=5) -> dict` with `n, share_positive, share_negative, share_neutral, net_tone, top_positive: [{sentence, p}], top_negative`.

- [ ] **Step 1: Tests**

```python
from finlib.nlp.sec_text import extract_sections, filing_tone, html_to_text, split_sentences
FIX = Path(__file__).parent / "fixtures" / "filing_tone"

def test_html_to_text_drops_tags_scripts_and_entities():
    t = html_to_text("<p>Item&#160;1A.<script>x()</script> Risk <b>Factors</b></p><p>Next</p>")
    assert "x()" not in t and "Item 1A. Risk Factors" in t and "\nNext" in t

def test_extract_sections_prefers_the_body_over_the_toc():
    sections, warnings = extract_sections(html_to_text((FIX / "sample_10k.html").read_text(encoding="utf-8")))
    assert "wafer-level" in sections["risk_factors"] and "Risk Factors" not in sections["risk_factors"][:5]
    assert "Net sales" in sections["mdna"] and len(sections["mdna"]) > 300 and warnings == []

def test_extract_sections_warns_instead_of_guessing():
    sections, warnings = extract_sections("Item 1A. Risk Factors\nSome risk here.\nItem 1B. Unresolved")
    assert sections["risk_factors"] and sections["mdna"] == "" and any("Item 7" in w for w in warnings)

def test_split_sentences_keeps_abbreviations_and_decimals():
    s = split_sentences("Aehr Test Systems, Inc. sells in the U.S. and abroad. Margin was 3.5% last year, up from 2.1%. We expect growth. Short one.")
    assert s == ["Aehr Test Systems, Inc. sells in the U.S. and abroad.", "Margin was 3.5% last year, up from 2.1%."]

def test_filing_tone_net_and_drivers():
    probs = np.array([[0.1, 0.1, 0.8], [0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.05, 0.05, 0.9]])
    t = filing_tone(probs, ["a", "b", "c", "d"], top_k=1)
    assert t["n"] == 4 and t["share_positive"] == 0.5 and t["share_negative"] == 0.25
    assert t["net_tone"] == pytest.approx(0.25)
    assert t["top_positive"] == [{"sentence": "d", "p": 0.9}] and t["top_negative"][0]["sentence"] == "b"

def test_filing_tone_handles_no_sentences():
    t = filing_tone(np.zeros((0, 3)), [])
    assert t["n"] == 0 and t["net_tone"] == 0.0 and t["top_positive"] == []
```

- [ ] **Step 2: Implement**

```python
"""Real 10-K text in, a tone number out.

A 10-K is an HTML document. We flatten it to text, cut out Item 1A (Risk Factors) and
Item 7 (MD&A), split those into sentences, score each sentence with the classifier, and
summarize: net tone = share of sentences positive minus share negative. The sentences that
the model was most sure about are kept so a reader can see what drove the number.
"""
from __future__ import annotations
import html as html_lib, re
from html.parser import HTMLParser
import numpy as np

BLOCK_TAGS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "td", "th"}

class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True); self.parts: list[str] = []; self._skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"): self._skip += 1
        elif tag in BLOCK_TAGS: self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag in ("script", "style"): self._skip = max(0, self._skip - 1)
        elif tag in BLOCK_TAGS: self.parts.append("\n")
    def handle_data(self, data):
        if not self._skip: self.parts.append(data)

def html_to_text(html: str) -> str:
    p = _Text(); p.feed(html); p.close()
    text = html_lib.unescape("".join(p.parts)).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()

SECTIONS = {
    "risk_factors": (re.compile(r"item\s*1a\.?\s*[\-–—:]?\s*risk\s+factors", re.I), re.compile(r"item\s*1b\.?|item\s*2\.?\s*[\-–—:]?\s*propert", re.I), "Item 1A"),
    "mdna": (re.compile(r"item\s*7\.?\s*[\-–—:]?\s*management['’]?s\s+discussion", re.I), re.compile(r"item\s*7a\.?|item\s*8\.?\s*[\-–—:]?\s*financial\s+statements", re.I), "Item 7"),
}

def extract_sections(text: str) -> tuple[dict[str, str], list[str]]:
    """Each heading appears at least twice (table of contents, then body). Take, among every
    start→next-end span, the longest one: the body. Missing section → "" plus a warning."""
    out, warnings = {}, []
    for key, (start_re, end_re, label) in SECTIONS.items():
        best = ""
        for m in start_re.finditer(text):
            e = end_re.search(text, m.end())
            span = text[m.end(): e.start() if e else len(text)].strip()
            if len(span) > len(best): best = span
        if len(best) < 200:
            warnings.append(f"{label} not found or under 200 characters; section left empty"); best = ""
        out[key] = best
    return out, warnings

_ABBR = ["Inc.", "Corp.", "Ltd.", "Co.", "LLC.", "U.S.", "U.K.", "No.", "vs.", "Mr.", "Ms.", "Dr.", "e.g.", "i.e.", "approx.", "Jan.", "Feb.", "Aug.", "Sept.", "Oct.", "Nov.", "Dec."]
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")

def split_sentences(text: str, min_words: int = 6, max_words: int = 120) -> list[str]:
    protected = text
    for i, a in enumerate(_ABBR):
        protected = protected.replace(a, a[:-1] + f"\x00{i}\x00")
    out = []
    for raw in _SPLIT.split(re.sub(r"\s+", " ", protected)):
        s = raw
        for i, a in enumerate(_ABBR):
            s = s.replace(a[:-1] + f"\x00{i}\x00", a)
        s = s.strip(); words = s.split()
        if not (min_words <= len(words) <= max_words): continue
        letters = sum(ch.isalpha() for ch in s)
        if letters < 0.5 * len(s.replace(" ", "")): continue     # tables and page furniture
        out.append(s)
    return out

def filing_tone(probs: np.ndarray, sentences: list[str], top_k: int = 5) -> dict:
    n = len(sentences)
    if n == 0:
        return {"n": 0, "share_positive": 0.0, "share_negative": 0.0, "share_neutral": 0.0, "net_tone": 0.0, "top_positive": [], "top_negative": []}
    pred = probs.argmax(1)
    share = {c: float((pred == c).mean()) for c in range(3)}
    def top(col):
        idx = np.argsort(-probs[:, col])[:top_k]
        return [{"sentence": sentences[i], "p": round(float(probs[i, col]), 4)} for i in idx]
    return {"n": n, "share_positive": share[2], "share_negative": share[0], "share_neutral": share[1],
            "net_tone": share[2] - share[0], "top_positive": top(2), "top_negative": top(0)}
```

- [ ] **Step 3: Run, pass, ruff. Commit** `Filing Tone: 10-K text extraction, sentence splitting, tone aggregation`

### Task 7: Fetching the latest 10-K from EDGAR (offline fixture path)

**Files:** Modify `finlib/nlp/sec_text.py`; Test `tests/test_filing_tone.py`

**Interfaces:** `Filing(ticker, company, cik, accession, form, report_date, filed, url, html)` dataclass; `TenKFetcher(identity=None, fixture_dir=FIXTURE_DIR, cache_dir=Path("output/filing_tone/cache"))` with `latest_10k(ticker) -> Filing`. Offline: any ticker listed in `fixture_dir/filings.json` maps to `sample_10k.html`; others raise `LookupError`.

- [ ] **Step 1: Fixture** `tests/fixtures/filing_tone/filings.json`:

```json
{"AEHR": {"company": "Aehr Test Systems (synthetic fixture)", "cik": "0001040470", "accession": "0000000000-26-000001", "form": "10-K", "report_date": "2026-05-31", "filed": "2026-08-15", "document": "sample_10k.html"}}
```

- [ ] **Step 2: Tests**

```python
def test_latest_10k_offline_reads_fixture():
    from finlib.nlp.sec_text import TenKFetcher
    f = TenKFetcher().latest_10k("aehr")
    assert f.ticker == "AEHR" and f.form == "10-K" and "Risk Factors" in f.html and f.url.startswith("fixture:")

def test_latest_10k_offline_unknown_ticker_raises():
    from finlib.nlp.sec_text import TenKFetcher
    with pytest.raises(LookupError):
        TenKFetcher().latest_10k("ZZZZ")
```

- [ ] **Step 3: Implement** (append to `sec_text.py`)

```python
import json, os, time
from dataclasses import dataclass
from pathlib import Path
from finlib import offline
from finlib.store.cache import JsonCache

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "filing_tone"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/{doc}"

@dataclass
class Filing:
    ticker: str; company: str; cik: str; accession: str; form: str; report_date: str; filed: str; url: str; html: str

class TenKFetcher:
    """Same SEC etiquette as finlib.providers.edgar: identity User-Agent, <10 req/s, JSON cache."""
    def __init__(self, identity: str | None = None, fixture_dir: Path = FIXTURE_DIR, cache_dir: Path = Path("output/filing_tone/cache")):
        self.identity = identity or os.getenv("SEC_IDENTITY", "")
        self.fixture_dir = fixture_dir; self.cache_dir = cache_dir
        self.cache = JsonCache("filing_tone", ttl_seconds=24 * 3600); self._last = 0.0

    def latest_10k(self, ticker: str) -> Filing:
        ticker = ticker.upper()
        return self._fixture(ticker) if offline() else self._live(ticker)

    def _fixture(self, ticker: str) -> Filing:
        table = json.loads((self.fixture_dir / "filings.json").read_text(encoding="utf-8"))
        if ticker not in table: raise LookupError(f"No fixture filing for {ticker}")
        r = table[ticker]
        return Filing(ticker, r["company"], r["cik"], r["accession"], r["form"], r["report_date"], r["filed"],
                      f"fixture:{r['document']}", (self.fixture_dir / r["document"]).read_text(encoding="utf-8"))

    def _get(self, url: str, as_json: bool):
        import requests
        if not self.identity: raise RuntimeError("SEC_IDENTITY is required for live EDGAR calls (see .env.example)")
        if as_json and (c := self.cache.get(url)) is not None: return c
        wait = self._last + 0.125 - time.monotonic()
        if wait > 0: time.sleep(wait)
        r = requests.get(url, headers={"User-Agent": self.identity, "Accept-Encoding": "gzip, deflate"}, timeout=60)
        self._last = time.monotonic(); r.raise_for_status()
        if as_json:
            data = r.json(); self.cache.set(url, data); return data
        return r.text

    def _live(self, ticker: str) -> Filing:
        table = self._get(TICKERS_URL, True)
        row = next((v for v in table.values() if v["ticker"].upper() == ticker), None)
        if row is None: raise LookupError(f"{ticker} not in SEC company_tickers.json")
        cik = str(row["cik_str"]).zfill(10)
        sub = self._get(SUBMISSIONS.format(cik=cik), True)
        rec = sub["filings"]["recent"]
        for i, form in enumerate(rec["form"]):
            if form == "10-K":
                acc = rec["accessionNumber"][i]; doc = rec["primaryDocument"][i]
                url = ARCHIVE.format(cik_int=int(cik), acc=acc.replace("-", ""), doc=doc)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                cached = self.cache_dir / f"{acc}.html"
                html = cached.read_text(encoding="utf-8") if cached.exists() else self._get(url, False)
                if not cached.exists(): cached.write_text(html, encoding="utf-8")
                return Filing(ticker, row["title"], cik, acc, form, rec["reportDate"][i], rec["filingDate"][i], url, html)
        raise LookupError(f"{ticker} ({row['title']}) has no 10-K in recent filings (foreign filers use 20-F)")
```

- [ ] **Step 4: Run, pass, ruff. Commit** `Filing Tone: latest 10-K fetcher with fixture path and SEC etiquette`

### Task 8: CLI

**Files:** Create `projects/filing_tone/__init__.py`, `__main__.py`, `cli.py`; Test `tests/test_filing_tone.py`

**Interfaces:** Typer app with commands `data`, `train --model {baseline,cnn,distilbert} [--epochs --batch-size --lr --seed --device --fixture]`, `evaluate --model`, `results`, `score --text`, `snapshot --tickers`, `summary`. Module-level helpers used by tests and the page: `OUT = Path("output/filing_tone")`, `HERE`, `SNAPSHOT = HERE / "tone_snapshot.json"`, `RESULTS = HERE / "results.json"`, `load_snapshot() -> dict`, `load_results() -> dict`, `results_table(results: dict) -> str` (markdown), `update_readme_results(readme: Path, table: str) -> None` (replaces the block between `<!-- results:start -->` and `<!-- results:end -->`), `build_snapshot(tickers, scorer, fetcher) -> dict` (pure: scorer is `Callable[[list[str]], np.ndarray]`), `peer_median(snapshot, section) -> float | None`.

Checkpoint layout under `OUT/models/`: `baseline.joblib`, `cnn.pt` (`{"state": state_dict, "itos": vocab.itos, "config": cfg}`), `distilbert/` (`save_pretrained`).

- [ ] **Step 1: Tests** (pure Python; no torch)

```python
from projects.filing_tone import cli as ft

def test_build_snapshot_uses_scorer_and_skips_missing_filers(monkeypatch):
    from finlib.nlp.sec_text import TenKFetcher
    def scorer(sentences):  # every sentence positive with p=0.9
        return np.tile([0.05, 0.05, 0.9], (len(sentences), 1))
    snap = ft.build_snapshot(["AEHR", "CAMT"], scorer, TenKFetcher(), model_name="test", run_id="t")
    a = snap["companies"]["AEHR"]
    assert a["sections"]["mdna"]["net_tone"] == 1.0 and a["sections"]["risk_factors"]["n"] > 3
    assert snap["skipped"] == {"CAMT": "No fixture filing for CAMT"} and snap["model"] == "test"

def test_results_table_and_readme_block(tmp_path):
    results = {"baseline": {"accuracy": 0.8, "macro_f1": 0.75, "per_class": {"negative": {"f1": .7}, "neutral": {"f1": .8}, "positive": {"f1": .75}}, "run_id": "b-1"},
               "cnn": {"accuracy": 0.85, "macro_f1": 0.8, "per_class": {"negative": {"f1": .8}, "neutral": {"f1": .85}, "positive": {"f1": .75}}, "run_id": "c-1"}}
    table = ft.results_table(results)
    assert "| cnn | 85.0% | 0.800 |" in table
    readme = tmp_path / "README.md"; readme.write_text("x\n<!-- results:start -->\nold\n<!-- results:end -->\ny", encoding="utf-8")
    ft.update_readme_results(readme, table)
    txt = readme.read_text(encoding="utf-8"); assert "old" not in txt and "| cnn |" in txt and txt.endswith("y")
```

- [ ] **Step 2: Implement `cli.py`** with the helpers above and the commands. `train` builds the split (fixture or PhraseBank), fits the chosen model, evaluates on test, writes the run log, saves the checkpoint, and merges `test` metrics plus `run_id` into `results.json`. `evaluate` reloads a checkpoint and also computes the all-agree ablation when PhraseBank is present. `score` loads the best available checkpoint (distilbert, else cnn, else baseline) and prints class probabilities for `--text`. `snapshot` uses `build_snapshot` with a scorer closed over the loaded model and writes `tone_snapshot.json`. `summary` prints the snapshot table. Write real code, not stubs; the `load_scorer(model_name) -> Callable` helper is the one place torch is imported.

- [ ] **Step 3: Run tests, pass. Run `python -m projects.filing_tone train --model cnn --fixture --epochs 5` and `--model baseline --fixture` as a smoke test. Commit** `Filing Tone: CLI (data, train, evaluate, results, score, snapshot, summary)`

### Task 9: Real training runs and the results table

- [ ] **Step 1:** `python -m projects.filing_tone data` (downloads PhraseBank, prints sizes and label mix).
- [ ] **Step 2:** `train --model baseline`, `train --model cnn --epochs 30`, `train --model distilbert --epochs 4 --lr 3e-5 --batch-size 32`. Record wall times.
- [ ] **Step 3:** `evaluate --model distilbert` and `--model cnn` for the all-agree ablation.
- [ ] **Step 4:** `results` writes `results.json` and updates the README block (README created in Task 12 with the markers; create the README skeleton here if needed). Commit `Filing Tone: results from baseline, CNN and DistilBERT runs`.

### Task 10: Live snapshot

- [ ] **Step 1:** `python -m projects.filing_tone snapshot --tickers AEHR,FORM,KLIC,COHU,ONTO,TER,CAMT` (live, SEC_IDENTITY from .env). Inspect warnings; if a filer's section extraction fails, note it in `skipped` rather than hand-patching.
- [ ] **Step 2: Test** that the committed snapshot is consistent:

```python
def test_committed_snapshot_is_internally_consistent():
    snap = ft.load_snapshot()
    assert snap["companies"] and snap["model"] and snap["date"]
    for t, c in snap["companies"].items():
        for sec in ("risk_factors", "mdna"):
            s = c["sections"][sec]
            assert s["net_tone"] == pytest.approx(s["share_positive"] - s["share_negative"])
            assert all(0 <= d["p"] <= 1 for d in s["top_positive"] + s["top_negative"])
```

- [ ] **Step 3: Commit** `Filing Tone: tone snapshot of AEHR and peers from the latest 10-Ks`

### Task 11: Streamlit page

**Files:** Create `app/pages/6_Filing_Tone.py`; Modify `app/Home.py` (add page), `app/pages/0_Overview.py` (add row if it lists tools); Test `tests/test_app.py`.

- [ ] **Step 1: Test**

```python
def test_filing_tone_page_shows_snapshot_numbers():
    from projects.filing_tone import cli as ft
    snap = ft.load_snapshot(); aehr = snap["companies"]["AEHR"]["sections"]
    at = run_page("pages/6_Filing_Tone.py")
    m = metrics(at)
    assert m["AEHR net tone, MD&A"] == f"{aehr['mdna']['net_tone']:+.2f}"
    assert m["AEHR net tone, risk factors"] == f"{aehr['risk_factors']['net_tone']:+.2f}"
    assert not any("torch" in mod for mod in sys.modules if mod.startswith("torch"))
```

- [ ] **Step 2: Implement the page** per DESIGN.md: `ui.page_setup(..., number="06", section="Filing tone")`, gray badge via `st.markdown(":gray-badge[:material/description: Snapshot · SEC EDGAR 10-Ks] ...")`, KPI strip (`st.metric` in a bordered horizontal container: AEHR MD&A tone, AEHR risk-factor tone, peer median MD&A, model macro-F1 from `results.json`), Exhibit 1 horizontal bar chart of MD&A net tone by company (`go.Bar`, `ui.SERIES[0]`, AEHR highlighted in `ui.NAVY`), Exhibit 2 risk factors, Exhibit 3 the model comparison table (`st.dataframe` from `results.json`), then one `st.expander` per company listing the top positive and negative sentences with their probabilities. `ui.source(...)` under every exhibit. A short "How to read this" paragraph in plain words.

- [ ] **Step 3: Register in `Home.py`**: `st.Page("pages/6_Filing_Tone.py", title="Filing tone", url_path="tone")`.
- [ ] **Step 4: Run `FINLIB_OFFLINE=1 pytest tests/test_app.py -q`, pass. Commit** `Filing Tone: Streamlit page on the committed snapshot`

### Task 12: Docs, CI, Makefile, README rows, walkthrough

**Files:** Create `.github/workflows/ml.yml`, `projects/filing_tone/README.md`, `docs/filing_tone/WALKTHROUGH.md`; Modify `Makefile`, `README.md`, `CLAUDE.md`, `AGENTS.md`, `app/README.md` if it has a page list.

- [ ] **Step 1: `ml.yml`**

```yaml
name: ML tests
on: [push, pull_request, workflow_dispatch]
jobs:
  ml:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install torch --index-url https://download.pytorch.org/whl/cpu
      - run: pip install -e ".[dev,ml]"
      - name: Filing Tone tests (torch, offline fixtures)
        env: { FINLIB_OFFLINE: "1" }
        run: pytest -q -rs tests/test_filing_tone.py
      - name: Train the CNN on the fixture (smoke)
        env: { FINLIB_OFFLINE: "1" }
        run: python -m projects.filing_tone train --model cnn --fixture --epochs 3 --device cpu
```

- [ ] **Step 2: Makefile** targets `demo-filing-tone` (fixture CNN train, 3 epochs) and `train-filing-tone` (full three-model run).
- [ ] **Step 3: README.md** row 8, `CLAUDE.md`/`AGENTS.md` table rows for `finlib/nlp/` and `projects/filing_tone/`, test count note.
- [ ] **Step 4: `projects/filing_tone/README.md`**: what it is, the results block (markers), resume line, how it is built (table like the pitch README), choices to defend, limits, how to run.
- [ ] **Step 5: `docs/filing_tone/WALKTHROUGH.md`**: the learner's walkthrough in the data-flow order from the spec, each section naming file and function.
- [ ] **Step 6: Full suite** `FINLIB_OFFLINE=1 pytest -q` and `ruff check .`; pass. Commit `Filing Tone: README, walkthrough, ML CI job, Makefile targets`.
