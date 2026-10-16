"""Text to integers, for the from-scratch model.

A neural network cannot read letters; it reads numbers. A *tokenizer* chops a sentence into
pieces (tokens) and a *vocabulary* gives each distinct token an integer id. Two ids are
reserved: 0 (`<pad>`) fills short sentences out to a common length inside a batch, and
1 (`<unk>`) stands for any token the training data never showed us.

DistilBERT brings its own tokenizer (WordPiece, 30k sub-word pieces), so this module is
only used by the TextCNN.
"""

from __future__ import annotations

import re
from collections import Counter

# Words, numbers (with decimals and a trailing %), and every other single character.
TOKEN_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?%?|[^\sa-z\d]")


class RegexTokenizer:
    def __call__(self, text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())


class Vocab:
    PAD, UNK = "<pad>", "<unk>"
    PAD_ID, UNK_ID = 0, 1

    def __init__(self, tokens: list[str]):
        self.itos = [self.PAD, self.UNK] + list(tokens)  # id -> token
        self.stoi = {t: i for i, t in enumerate(self.itos)}  # token -> id
        self.tokenizer = RegexTokenizer()

    @classmethod
    def build(cls, texts, tokenizer: RegexTokenizer, min_freq: int = 1, max_size: int | None = None) -> "Vocab":
        """Built from the training split only, so the vocabulary never sees test sentences.
        Most frequent first; ties broken alphabetically so the result is deterministic."""
        counts = Counter(t for text in texts for t in tokenizer(text))
        ranked = sorted((t for t, c in counts.items() if c >= min_freq), key=lambda t: (-counts[t], t))
        return cls(ranked[:max_size] if max_size else ranked)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(t, self.UNK_ID) for t in self.tokenizer(text)]

    def decode(self, ids) -> list[str]:
        return [self.itos[i] for i in ids]

    def __len__(self) -> int:
        return len(self.itos)
