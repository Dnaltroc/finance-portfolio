"""From a list of sentences to the tensors a model consumes.

A *Dataset* hands out one example at a time. A *DataLoader* groups examples into *batches*
(the model updates its weights once per batch, not once per sentence; 32 at a time is a
common size) and calls a *collate* function to stack them into one tensor. Sentences differ
in length, so the short ones in a batch are padded with id 0 out to the longest, and an
*attention mask* of 1s and 0s records which positions are real words.

Two encoders share one contract, texts -> {"input_ids", "attention_mask"}:
    vocab_encoder   our own Vocab, for the TextCNN
    hf_encoder      DistilBERT's WordPiece tokenizer, for the transformer
so the training loop never needs to know which model it is driving.
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

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i].text, self.examples[i].label


def vocab_encoder(vocab: Vocab, max_len: int = 64) -> Encoder:
    def encode_batch(texts: list[str]) -> dict[str, torch.Tensor]:
        ids = [vocab.encode(t)[:max_len] or [Vocab.UNK_ID] for t in texts]  # never an empty row
        width = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), width), Vocab.PAD_ID, dtype=torch.long)
        mask = torch.zeros((len(ids), width), dtype=torch.long)
        for row, x in enumerate(ids):
            input_ids[row, :len(x)] = torch.tensor(x)
            mask[row, :len(x)] = 1
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


def make_loader(examples, encode_batch: Encoder, batch_size: int, shuffle: bool,
                seed: int | None = None) -> DataLoader:
    """Shuffling the training set each epoch keeps batches from repeating in the same order;
    a seeded generator makes that shuffle reproducible."""
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(SentenceDataset(examples), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=make_collate(encode_batch), generator=gen)
