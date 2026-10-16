"""Scores from scratch, so the numbers are understood rather than imported.

*Accuracy* is the share of sentences the model got right. It hides problems when one class
dominates: PhraseBank is about 60% neutral, so always answering "neutral" scores 60%. We
therefore also report *macro-F1*: the F1 of each class, averaged with equal weight, so the
rarer negative class counts as much as neutral.

F1 balances two things for one class, say positive:
    precision  of the sentences we called positive, how many really were
    recall     of the truly positive sentences, how many we caught
    F1         their harmonic mean; high only when both are high

The *confusion matrix* shows every mistake: row = true label, column = predicted label.

scikit-learn computes the same numbers and is used in the tests as an oracle, not here.
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
    """Per-class arrays. A class never predicted gets precision 0 and F1 0, not NaN."""
    tp = np.diag(cm).astype(float)
    pred_pos = cm.sum(axis=0).astype(float)  # column sums: how often we said each class
    true_pos = cm.sum(axis=1).astype(float)  # row sums: how often each class was the truth
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
