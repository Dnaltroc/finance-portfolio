# Filing Tone: financial sentiment in PyTorch, applied to 10-Ks

Three sentence-sentiment models, trained on the same expert-labeled finance dataset and scored on the same held-out split, then the best one reads the Risk Factors and MD&A of real SEC 10-Ks and turns each filing into a tone number with the sentences that drove it.

New to this? Read [the walkthrough](../../docs/filing_tone/WALKTHROUGH.md) first. It explains every step in plain language, in the order the data flows, and names the file that does it.

## Resume line

> Built a PyTorch financial sentiment pipeline: a from-scratch text CNN and a fine-tuned DistilBERT (85% accuracy, 0.82 macro-F1 on Financial PhraseBank) against a TF-IDF baseline, then applied it to SEC 10-K risk factors and MD&A to score filing tone for a semiconductor peer group.

## Results

Held-out test split: 486 sentences, 10% of Financial PhraseBank (50% agreement level), stratified by label, seed 42, never used for any decision. Every number below is written by `python -m projects.filing_tone results` from `results.json`, which the `train` command writes.

<!-- results:start -->
| Model | Accuracy | Macro-F1 | F1 negative | F1 neutral | F1 positive | Train time |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 77.6% | 0.711 | 0.595 | 0.847 | 0.692 | 1s |
| cnn | 78.0% | 0.719 | 0.613 | 0.844 | 0.700 | 6s |
| distilbert | 85.0% | 0.815 | 0.734 | 0.888 | 0.824 | 26s |

On the sentences every annotator agreed on (test split only):

| Model | Accuracy, all agree | Macro-F1, all agree | n |
| --- | --- | --- | --- |
| baseline | 89.7% | 0.845 | 232 |
| cnn | 89.7% | 0.844 | 232 |
| distilbert | 95.3% | 0.920 | 232 |
<!-- results:end -->

What the table says:

- **The transformer wins by a wide margin.** DistilBERT already knows English; fine-tuning it on 3,876 sentences for four epochs (26 seconds on an RTX 5070) beats everything else on every class.
- **The from-scratch CNN is level with the baseline.** On 4k sentences a model that must learn word meanings from zero has little advantage over counting words. That is the honest finding, and the reason transfer learning dominates applied NLP. Its settings came from a 30-run sweep on the validation split only; on the GPU its test macro-F1 moves by a few hundredths between otherwise identical runs (cuDNN is not bit-deterministic, and the validation set is 484 sentences), so read it as roughly level with the baseline.
- **Negative is the hard class.** It is 12% of the data; macro-F1 weights it equally with neutral, which is why macro-F1 sits below accuracy for every model.
- **All-agree is easier.** The sentences every annotator agreed on are the clear-cut ones. The original paper reports on this subset, so the ablation makes the numbers comparable.

## What it does to a 10-K

```
python -m projects.filing_tone snapshot        # live: SEC_IDENTITY in .env
python -m projects.filing_tone summary         # print the committed snapshot
```

For AEHR and its pitch peers, the latest 10-K is found through the SEC submissions API, Item 1A (Risk Factors) and Item 7 (MD&A) are cut out of the HTML, split into sentences, and each sentence is classified. Net tone = share positive − share negative. The five most confident positive and negative sentences per section are kept as evidence. The result is `tone_snapshot.json`, which the Streamlit page reads so the app never needs torch.

<!-- tone:start -->
| Ticker | Fiscal year end | MD&A net tone | Risk-factor net tone | Sentences (MD&A / risk) |
| --- | --- | --- | --- | --- |
| AEHR | 2026-05-29 | -0.08 | -0.31 | 145 / 258 |
| FORM | 2025-12-27 | +0.10 | -0.42 | 193 / 286 |
| KLIC | 2025-10-04 | -0.06 | -0.30 | 179 / 345 |
| COHU | 2025-12-27 | +0.01 | -0.33 | 307 / 395 |
| ONTO | 2026-01-03 | +0.08 | -0.37 | 167 / 417 |
| TER | 2025-12-31 | +0.04 | -0.34 | 225 / 257 |
| CAMT | — | skipped | skipped | CAMT (CAMTEK LTD) has no 10-K in recent filings (foreign filers use 20-F) |

Snapshot 2026-10-16, model `distilbert` (run distilbert-20261016-084102).
<!-- tone:end -->

CAMT files a 20-F, not a 10-K, and is skipped with a note rather than guessed. The table is written by `snapshot`; `summary` prints it.

AEHR's MD&A is the most negative in the group, in the same filing where it guided revenue up 180%, while its risk factors read mildest of the six (every risk-factor section is negative; AEHR's is the least so). Taken with the [pitch](../pitch/README.md): management's narrative is cautious about the year just ended, the risk section is boilerplate-light for a company with 58% customer concentration, and the price assumes neither.

The first version of this table said the opposite about AEHR's risk factors. A review found that an in-text cross-reference ("see Item 1A. Risk Factors") was opening the section early on five of six filings, so peers' numbers included Item 1 or the financial statements. Fixed with a test; the committed snapshot is from the corrected extractor.

## How it is built

| Step | Where | What it does |
| --- | --- | --- |
| Data | `finlib/nlp/data.py` | Financial PhraseBank from the Hugging Face hub (original zip, parsed directly); stratified 80/10/10 split; a 90-sentence synthetic fixture for tests |
| Tokens | `finlib/nlp/tokenize.py` | Regex tokenizer and vocabulary with pad and unknown ids (the CNN path; DistilBERT brings its own) |
| Batches | `finlib/nlp/batches.py` | `Dataset`, padding collate with attention masks, `DataLoader` |
| Models | `finlib/nlp/models.py` | `TextCNN` (embedding, 3/4/5-token convolutions, max-pool, dropout, linear) and a DistilBERT wrapper with the same `forward` |
| Training | `finlib/nlp/train.py` | AdamW, linear warmup and decay, class weights, gradient clipping, early stopping on validation macro-F1, seeded, JSON run log |
| Metrics | `finlib/nlp/evaluate.py` | Confusion matrix, precision, recall, F1, macro-F1 by hand; scikit-learn is the oracle in the tests |
| Baseline | `finlib/nlp/baseline.py` | TF-IDF (1–2 grams) + logistic regression |
| Filings | `finlib/nlp/sec_text.py` | Latest 10-K via the submissions API, HTML to text, section cut, sentence split, net tone |
| CLI | `cli.py` | `data`, `train`, `evaluate`, `results`, `score`, `snapshot`, `summary` |
| Page | `app/pages/6_Filing_Tone.py` | Reads the snapshot and results; no torch |

```bash
pip install -e ".[dev,ml]"                               # torch, transformers, scikit-learn
python -m projects.filing_tone data                      # download PhraseBank once, show the split
python -m projects.filing_tone train --model baseline
python -m projects.filing_tone train --model cnn
python -m projects.filing_tone train --model distilbert  # 26s on a GPU, ~10 min on CPU
python -m projects.filing_tone results                   # results.json -> the table above
python -m projects.filing_tone score --text "Revenue fell 22% as the largest customer paused orders."
make demo-filing-tone                                    # offline: the CNN on the synthetic fixture
```

Checkpoints and run logs land in `output/filing_tone/` (gitignored). GPU is used when torch can see one; CPU gives the same answers.

## Choices to be ready to defend

- **Sentence level, then aggregate.** Only sentence labels exist. Aggregating transparently (shares of each class) keeps the filing score explainable: the driver sentences are the audit trail.
- **Net tone, not average probability.** A share is readable ("19% of MD&A sentences read negative"). Averaging probabilities would hide the neutral mass that dominates every filing.
- **DistilBERT, not FinBERT.** FinBERT was fine-tuned on PhraseBank itself. Starting from it would be grading on the training set. Starting from a general model shows the transfer step honestly.
- **TF-IDF baseline, not the Loughran-McDonald dictionary.** The LM word lists carry a non-commercial license and a separate download; TF-IDF is reproducible from the data alone. LM is the natural next baseline.
- **Metrics by hand.** Twenty lines of NumPy, tested against scikit-learn on a hand-built confusion matrix, so the numbers are understood rather than imported.
- **Section ends must start a line.** A real MD&A says "see Part II, Item 8" mid-paragraph; the first version lost 90% of KLIC's MD&A to that. Fixed with a test on a synthetic case.
- **No torch in the app.** The committed snapshot is the app's data, as the pitch snapshot is for the memo. Streamlit Cloud and the Docker image stay small.

## Limits

- Trained on news sentences; 10-K prose is longer and more hedged. The model is calibrated on the test split, not on filings. Read the driver sentences before trusting a score.
- Neutral dominates (60% of training, more of a filing), so small changes in net tone are within noise. Compare companies, not decimals.
- One filing per company. A time series per company (tone by year) is the obvious extension; the fetcher already returns the accession list order.
- Financial PhraseBank is CC BY-NC-SA 3.0: fine for a portfolio, not for a commercial product.

## Tests

`tests/test_filing_tone.py`: the fixture is balanced and labeled synthetic; the split is stratified and deterministic; the tokenizer and vocab round-trip; the metrics equal scikit-learn; batches pad and mask correctly and truncate to `max_len`; the CNN forward shape is right on inputs shorter than its widest kernel; the training loop can memorize the fixture and halves its loss; the device falls back to CPU; the baseline predicts probabilities; 10-K extraction prefers the body over the table of contents, ignores inline cross-references, warns instead of guessing, keeps abbreviations and decimals intact, drops table rows and page furniture; net tone and drivers match a hand calculation and survive an empty section; the fetcher reads the fixture and raises on an unknown ticker; the committed snapshot is internally consistent. `tests/test_app.py` checks the page shows the snapshot's numbers without importing torch.

Torch tests skip when the `ml` extra is absent; `.github/workflows/ml.yml` installs it and runs them, plus a 3-epoch fixture training smoke.

Student project, prepared for a portfolio. Not investment advice.
