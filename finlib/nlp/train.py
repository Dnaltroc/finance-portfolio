"""The training loop, shared by the TextCNN and DistilBERT.

One *epoch* is one pass over the training set. For every batch:
    forward    run the model, get logits
    loss       cross-entropy: how far the predicted probabilities are from the true label.
               Zero when the model puts all its probability on the right class.
    backward   autograd computes the gradient of the loss with respect to every weight:
               which direction each weight should move to lower the loss
    step       the optimizer (AdamW) nudges every weight a small distance against its
               gradient. The *learning rate* is that distance; it warms up from zero so
               the first, noisy batches do not throw the weights around, then decays to zero.

Class weights make a mistake on a rare class (negative, ~12% of PhraseBank) cost more than a
mistake on a common one, so the model cannot win by over-predicting neutral.

After each epoch we score the *validation* set, which the model never trains on. When
validation macro-F1 stops improving for `patience` epochs we stop (*early stopping*) and
restore the best weights. Training longer would only memorize the training sentences
(*overfitting*): training loss keeps falling while validation gets worse.

Everything is seeded so a run can be repeated exactly, and every run writes a JSON log.
"""

from __future__ import annotations

import copy
import json
import random
import time
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

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(pref: str | None = None) -> torch.device:
    """A GPU if torch can see one, else the CPU. Both give the same answers; the GPU is faster."""
    if pref:
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def class_weight_tensor(labels, n_classes: int = 3) -> torch.Tensor:
    """n / (C * count_c): a class with half the examples gets twice the weight."""
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(float)
    return torch.tensor(len(labels) / (n_classes * np.maximum(counts, 1)), dtype=torch.float)


@dataclass
class RunLog:
    run_id: str
    model: str
    config: dict
    seed: int
    device: str
    started: str
    train_size: int = 0
    val_size: int = 0
    test_size: int = 0
    vocab_size: int | None = None
    history: list[dict] = field(default_factory=list)
    test: dict | None = None
    ablation: dict | None = None
    wall_seconds: float = 0.0

    @classmethod
    def start(cls, cfg: TrainConfig, device: str) -> "RunLog":
        now = datetime.now(timezone.utc)
        return cls(run_id=f"{cfg.model}-{now:%Y%m%d-%H%M%S}", model=cfg.model, config=cfg.to_dict(),
                   seed=cfg.seed, device=device, started=now.isoformat(timespec="seconds"))

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "RunLog":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def _to(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()  # no gradients needed when only predicting: faster and uses less memory
def predict(model: nn.Module, loader, device: torch.device):
    """Labels and class probabilities, (N,) and (N, 3), in loader order."""
    model.eval()  # turns dropout off
    ys, ps = [], []
    for batch in loader:
        b = _to(batch, device)
        labels = b.pop("labels")
        ps.append(torch.softmax(model(**b), dim=1).cpu().numpy())
        ys.append(labels.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


@torch.no_grad()
def predict_texts(model: nn.Module, encode_batch, texts: list[str], device: torch.device,
                  batch_size: int = 64) -> np.ndarray:
    """Probabilities for raw sentences (the path the 10-K scorer uses)."""
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        b = _to(encode_batch(texts[i:i + batch_size]), device)
        out.append(torch.softmax(model(**b), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 3))


def train_loop(model: nn.Module, train_loader, val_loader, cfg: TrainConfig, device: torch.device,
               log: RunLog) -> nn.Module:
    model.to(device)
    labels = [int(label) for _, label in train_loader.dataset]
    weight = class_weight_tensor(labels).to(device) if cfg.class_weights else None
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.epochs * len(train_loader)
    warm = max(1, int(cfg.warmup_frac * total))
    sched = LambdaLR(opt, lambda s: s / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))

    best_f1, best_state, since_best, t0 = -1.0, copy.deepcopy(model.state_dict()), 0, time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()  # dropout on
        running, seen = 0.0, 0
        for batch in train_loader:
            b = _to(batch, device)
            y = b.pop("labels")
            loss = loss_fn(model(**b), y)
            opt.zero_grad()  # gradients accumulate by default; clear last batch's
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)  # no single huge step
            opt.step()
            sched.step()
            running += loss.item() * len(y)
            seen += len(y)
        y_val, p_val = predict(model, val_loader, device)
        s = summary(y_val, p_val.argmax(1))
        log.history.append({"epoch": epoch, "train_loss": running / max(seen, 1), "val_accuracy": s["accuracy"],
                            "val_macro_f1": s["macro_f1"], "lr": sched.get_last_lr()[0],
                            "seconds": round(time.time() - t0, 1)})
        if s["macro_f1"] > best_f1:
            best_f1, best_state, since_best = s["macro_f1"], copy.deepcopy(model.state_dict()), 0
        else:
            since_best += 1
            if since_best >= cfg.patience:
                break
    model.load_state_dict(best_state)
    log.wall_seconds = round(time.time() - t0, 1)
    return model
