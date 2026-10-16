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


# ---------------------------------------------------------------- tokenizer ----------
def test_tokenizer_lowercases_splits_punctuation_keeps_numbers():
    from finlib.nlp.tokenize import RegexTokenizer
    tok = RegexTokenizer()
    assert tok("Revenue rose 18.5% to $412M, Inc.") == ["revenue", "rose", "18.5%", "to", "$", "412", "m",
                                                        ",", "inc", "."]


def test_vocab_round_trips_and_maps_unknowns():
    from finlib.nlp.tokenize import RegexTokenizer, Vocab
    tok = RegexTokenizer()
    v = Vocab.build(["profit rose", "profit fell"], tok, min_freq=1)
    assert v.itos[:2] == ["<pad>", "<unk>"] and Vocab.PAD_ID == 0 and Vocab.UNK_ID == 1
    assert v.decode(v.encode("profit rose")) == ["profit", "rose"]
    assert v.encode("profit soared") == [v.stoi["profit"], Vocab.UNK_ID]
    assert len(v) == 5


def test_vocab_min_freq_and_max_size_cap_the_vocab():
    from finlib.nlp.tokenize import RegexTokenizer, Vocab
    tok = RegexTokenizer()
    v = Vocab.build(["a a a b b c"], tok, min_freq=2)
    assert "c" not in v.stoi and "b" in v.stoi
    v2 = Vocab.build(["a a a b b c"], tok, max_size=1)
    assert v2.itos == ["<pad>", "<unk>", "a"]


# ---------------------------------------------------------------- metrics ------------
Y_TRUE = [0, 0, 1, 1, 1, 2, 2, 2, 2, 0]
Y_PRED = [0, 1, 1, 1, 2, 2, 2, 0, 2, 0]


def test_confusion_matrix_rows_are_truth():
    from finlib.nlp.evaluate import confusion_matrix
    assert confusion_matrix(Y_TRUE, Y_PRED).tolist() == [[2, 1, 0], [0, 2, 1], [1, 0, 3]]


def test_metrics_match_sklearn_oracle():
    sk = pytest.importorskip("sklearn.metrics")
    from finlib.nlp.evaluate import summary
    s = summary(Y_TRUE, Y_PRED)
    assert s["accuracy"] == pytest.approx(sk.accuracy_score(Y_TRUE, Y_PRED))
    assert s["macro_f1"] == pytest.approx(sk.f1_score(Y_TRUE, Y_PRED, average="macro"))
    p, r, f, _ = sk.precision_recall_fscore_support(Y_TRUE, Y_PRED)
    for i, name in enumerate(LABELS):
        assert s["per_class"][name]["f1"] == pytest.approx(f[i])
        assert s["per_class"][name]["precision"] == pytest.approx(p[i])
        assert s["per_class"][name]["recall"] == pytest.approx(r[i])
    assert s["n"] == 10 and s["confusion"] == [[2, 1, 0], [0, 2, 1], [1, 0, 3]]


def test_f1_is_zero_not_nan_for_a_class_never_predicted():
    from finlib.nlp.evaluate import confusion_matrix, precision_recall_f1
    _, _, f = precision_recall_f1(confusion_matrix([0, 1, 2], [0, 0, 0]))
    assert f[1] == 0.0 and f[2] == 0.0 and not np.isnan(f).any()


# ---------------------------------------------------------------- batches + model ----
def test_vocab_encoder_pads_to_longest_and_masks():
    pytest.importorskip("torch")
    from finlib.nlp.batches import make_collate, vocab_encoder
    from finlib.nlp.tokenize import RegexTokenizer, Vocab
    v = Vocab.build(["profit rose sharply", "loss"], RegexTokenizer())
    collate = make_collate(vocab_encoder(v, max_len=10))
    b = collate([("profit rose sharply", 2), ("loss", 0)])
    assert b["input_ids"].shape == (2, 3) and b["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert b["input_ids"][1].tolist() == [v.stoi["loss"], 0, 0] and b["labels"].tolist() == [2, 0]


def test_vocab_encoder_truncates_to_max_len():
    pytest.importorskip("torch")
    from finlib.nlp.batches import vocab_encoder
    from finlib.nlp.tokenize import RegexTokenizer, Vocab
    v = Vocab.build(["a b c d e f"], RegexTokenizer())
    assert vocab_encoder(v, max_len=4)(["a b c d e f"])["input_ids"].shape == (1, 4)


def test_textcnn_forward_shape_and_short_inputs():
    torch = pytest.importorskip("torch")
    from finlib.nlp.models import TextCNN
    m = TextCNN(vocab_size=50)
    ids = torch.randint(2, 50, (4, 2))  # shorter than the widest kernel (5)
    assert m(ids).shape == (4, 3)


# ---------------------------------------------------------------- training -----------
def test_choose_device_falls_back_to_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    from finlib.nlp.train import choose_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert choose_device().type == "cpu" and choose_device("cpu").type == "cpu"


def test_class_weights_are_inverse_frequency_normalised():
    pytest.importorskip("torch")
    from finlib.nlp.train import class_weight_tensor
    w = class_weight_tensor([0, 1, 1, 1, 2, 2])  # counts 1, 3, 2 -> n / (C * count)
    assert w.tolist() == pytest.approx([2.0, 2 / 3, 1.0])


def test_textcnn_learns_the_fixture(tmp_path):
    torch = pytest.importorskip("torch")
    from finlib.nlp.batches import make_loader, vocab_encoder
    from finlib.nlp.models import TextCNN
    from finlib.nlp.tokenize import RegexTokenizer, Vocab
    from finlib.nlp.train import RunLog, TrainConfig, predict, set_seed, train_loop
    train, val, _ = stratified_split(load_fixture(), seed=0)
    cfg = TrainConfig(model="cnn", epochs=40, batch_size=16, lr=3e-3, patience=40, device="cpu")
    set_seed(cfg.seed)
    vocab = Vocab.build([e.text for e in train], RegexTokenizer())
    enc = vocab_encoder(vocab, cfg.max_len)
    model = TextCNN(len(vocab), embed_dim=32, channels=16)
    log = RunLog.start(cfg, device="cpu")
    cpu = torch.device("cpu")
    # Validate on the training set itself: the check is that the loop can drive the loss down and
    # memorize 72 sentences, and best-checkpoint restore would otherwise hand back an early epoch.
    model = train_loop(model, make_loader(train, enc, 16, True, seed=0), make_loader(train, enc, 32, False),
                       cfg, cpu, log)
    y, probs = predict(model, make_loader(train, enc, 32, False), cpu)
    assert (probs.argmax(1) == y).mean() > 0.95
    assert log.history[-1]["train_loss"] < 0.5 * log.history[0]["train_loss"]
    assert probs.shape == (72, 3) and np.allclose(probs.sum(1), 1)
    assert len(log.history) >= 1 and {"epoch", "train_loss", "val_macro_f1", "lr"} <= set(log.history[0])
    log.test = {"accuracy": 1.0}
    log.save(tmp_path / "run.json")
    assert RunLog.load(tmp_path / "run.json").config["epochs"] == 40


def test_baseline_fits_and_predicts_probabilities():
    pytest.importorskip("sklearn")
    from finlib.nlp.baseline import baseline_probs, fit_baseline
    ex = load_fixture()
    pipe = fit_baseline([e.text for e in ex], [e.label for e in ex])
    p = baseline_probs(pipe, ["Revenue rose 18% to a record."])
    assert p.shape == (1, 3) and p.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------- 10-K text ----------
def test_html_to_text_drops_tags_scripts_and_entities():
    from finlib.nlp.sec_text import html_to_text
    t = html_to_text("<p>Item&#160;1A.<script>x()</script> Risk <b>Factors</b></p><p>Next</p>")
    assert "x()" not in t and "Item 1A. Risk Factors" in t and "\nNext" in t


def test_extract_sections_prefers_the_body_over_the_toc():
    from finlib.nlp.sec_text import extract_sections, html_to_text
    sections, warnings = extract_sections(html_to_text((FIX / "sample_10k.html").read_text(encoding="utf-8")))
    assert "wafer-level" in sections["risk_factors"] and not sections["risk_factors"].startswith("Risk")
    assert "Net sales" in sections["mdna"] and len(sections["mdna"]) > 300 and warnings == []
    assert "Unresolved" not in sections["risk_factors"] and "market risk" not in sections["mdna"]


def test_extract_sections_warns_instead_of_guessing():
    from finlib.nlp.sec_text import extract_sections
    body = "Item 1A. Risk Factors\n" + "Some risk here. " * 20 + "\nItem 1B. Unresolved"
    sections, warnings = extract_sections(body)
    assert sections["risk_factors"] and sections["mdna"] == "" and any("Item 7" in w for w in warnings)


def test_split_sentences_keeps_abbreviations_and_decimals():
    from finlib.nlp.sec_text import split_sentences
    s = split_sentences("Aehr Test Systems, Inc. sells in the U.S. and abroad. Margin was 3.5% last year, "
                        "up from 2.1%. We expect growth. Short one.")
    assert s == ["Aehr Test Systems, Inc. sells in the U.S. and abroad.", "Margin was 3.5% last year, up from 2.1%."]


def test_split_sentences_drops_table_rows():
    from finlib.nlp.sec_text import split_sentences
    assert split_sentences("2026 2025 2024 84.2 57.7 41.0 48.5% 44.1% 39.0% 9.6 (2.3) (4.1) totals.") == []


def test_filing_tone_net_and_drivers():
    from finlib.nlp.sec_text import filing_tone
    probs = np.array([[0.1, 0.1, 0.8], [0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.05, 0.05, 0.9]])
    t = filing_tone(probs, ["a", "b", "c", "d"], top_k=1)
    assert t["n"] == 4 and t["share_positive"] == 0.5 and t["share_negative"] == 0.25
    assert t["net_tone"] == pytest.approx(0.25)
    assert t["top_positive"] == [{"sentence": "d", "p": 0.9}] and t["top_negative"][0]["sentence"] == "b"


def test_filing_tone_handles_no_sentences():
    from finlib.nlp.sec_text import filing_tone
    t = filing_tone(np.zeros((0, 3)), [])
    assert t["n"] == 0 and t["net_tone"] == 0.0 and t["top_positive"] == []


def test_latest_10k_offline_reads_fixture():
    from finlib.nlp.sec_text import TenKFetcher
    f = TenKFetcher().latest_10k("aehr")
    assert f.ticker == "AEHR" and f.form == "10-K" and "Risk Factors" in f.html and f.url.startswith("fixture:")
    assert "synthetic" in f.company.lower()


def test_latest_10k_offline_unknown_ticker_raises():
    from finlib.nlp.sec_text import TenKFetcher
    with pytest.raises(LookupError):
        TenKFetcher().latest_10k("ZZZZ")
