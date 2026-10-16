"""Filing Tone: financial sentence sentiment in PyTorch, applied to SEC 10-K text.

Data flow, one module per step:
    data.py       labeled sentences (Financial PhraseBank, or the synthetic fixture) -> splits
    tokenize.py   sentence -> tokens -> integer ids (for the from-scratch model)
    batches.py    ids -> padded tensors, grouped into batches
    models.py     TextCNN (from scratch) and a DistilBERT wrapper, same forward() contract
    train.py      the training loop: loss, gradient step, validation, early stopping, run log
    evaluate.py   accuracy, macro-F1, confusion matrix, written by hand
    baseline.py   TF-IDF + logistic regression, the floor the neural models must beat
    sec_text.py   10-K HTML -> Item 1A / Item 7 text -> sentences -> net tone

Nothing here touches finlib/models or finlib/providers.
"""
