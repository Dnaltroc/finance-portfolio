"""The floor: bag-of-words TF-IDF features into logistic regression. No neural network.

TF-IDF counts which words and word pairs appear in a sentence, down-weighting words that
appear everywhere ("the", "company"). Logistic regression learns one weight per feature per
class. It cannot see word order beyond pairs, which is exactly what the CNN and the
transformer add. If they cannot beat this, they are not earning their complexity.
"""

from __future__ import annotations

import numpy as np


def fit_baseline(texts, labels, seed: int = 42):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("lr", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=seed)),
    ])
    return pipe.fit(list(texts), list(labels))


def baseline_probs(pipe, texts) -> np.ndarray:
    return pipe.predict_proba(list(texts))
