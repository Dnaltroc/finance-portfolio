"""Two classifiers with one calling convention: forward(input_ids, attention_mask) -> logits.

*Logits* are three raw scores, one per class. Softmax turns them into probabilities that sum
to one; the largest is the prediction.

TextCNN, built from scratch (Kim, 2014):
    embedding     each token id becomes a learned vector of 100 numbers. Training moves words
                  that behave alike ("fell", "declined", "dropped") close together.
    convolutions  filters slide windows of 3, 4 and 5 tokens along the sentence and fire on
                  phrases like "fell short of" or "ahead of guidance", wherever they occur.
    max-pool      keep only the strongest response of each filter: did its phrase appear?
    dropout       during training, randomly zero half the features so the model cannot lean
                  on any single one; off at prediction time.
    linear        map the pooled features to three logits.

DistilBERT, pre-trained by Hugging Face on English Wikipedia and books, already carries a
usable representation of English. We attach a fresh 3-way head and *fine-tune* every weight
on our few thousand sentences: a small amount of labeled data goes a long way when the
model already knows the language.
"""

from __future__ import annotations

import torch
from torch import nn


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 100, num_classes: int = 3,
                 kernel_sizes=(3, 4, 5), channels: int = 100, dropout: float = 0.5, pad_id: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)  # pad vector stays zero
        # padding=k-1 on both sides, so a sentence shorter than the window still yields an output
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, channels, k, padding=k - 1) for k in kernel_sizes])
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(channels * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embed(input_ids).transpose(1, 2)  # (batch, tokens, embed) -> (batch, embed, tokens)
        pooled = [torch.relu(conv(x)).max(dim=2).values for conv in self.convs]  # each (batch, channels)
        return self.out(self.drop(torch.cat(pooled, dim=1)))


class HFClassifier(nn.Module):
    """Hugging Face models return an object with .logits; unwrap it to match TextCNN."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def build_distilbert(num_labels: int = 3, name: str = "distilbert-base-uncased"):
    """Downloads the pretrained weights once (about 260 MB, cached by Hugging Face)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)
    return tok, HFClassifier(model)
