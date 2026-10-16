# Filing Tone: design spec

Date: 2026-10-16. Status: approved in conversation, awaiting implementation plan.

## Purpose

A machine-learning tool, built in PyTorch, that shows ML proficiency to the finance and
data audience of this portfolio. It trains sentence-level financial sentiment models on a
public expert-labeled benchmark, compares three model tiers honestly, and then applies the
best model to real SEC 10-K text to produce a "filing tone" series for Aehr Test Systems
and its peers, tying into the existing equity pitch.

The owner is learning deep learning tooling. Code, docs, and the final walkthrough must
teach as well as work: every module explains what it does and why in plain language.

## Success criteria

1. `python -m projects.filing_tone train --model cnn` trains a from-scratch PyTorch text
   CNN on Financial PhraseBank and reports accuracy and macro-F1 on a held-out split.
2. `--model distilbert` fine-tunes DistilBERT with the same harness and beats the CNN;
   `--model baseline` (TF-IDF + logistic regression) is the floor.
3. The README has a results table produced by the code, not typed by hand, plus the
   50%-agreement vs 100%-agreement ablation.
4. `snapshot` scores the latest 10-K Risk Factors (Item 1A) and MD&A (Item 7) for AEHR and
   its 10-K-filing peers, writes a committed `tone_snapshot.json`, and the Streamlit page
   renders it without torch installed.
5. `FINLIB_OFFLINE=1 pytest -q` passes with and without torch installed; the ML tests skip
   cleanly when torch is absent and run in a dedicated CI job when it is present.
6. `docs/filing_tone/WALKTHROUGH.md` explains the pipeline to a reader seeing PyTorch for
   the first time.

## Data

**Training set: Financial PhraseBank** (Malo et al., 2014). About 4,840 English sentences
from financial news, each labeled positive, negative, or neutral by 16 annotators with
finance backgrounds. Published in four agreement levels (50%, 66%, 75%, 100% of annotators
agree). License CC BY-NC-SA 3.0; fine for a portfolio. Fetched through the Hugging Face
`datasets` library (`takala/financial_phrasebank`) and cached under `output/` (gitignored).

Split: stratified 80/10/10 train/validation/test with a fixed seed. The test split is
only touched by `evaluate`. Ablation: train on `sentences_50agree`, evaluate on the subset
that is also in `sentences_allagree`.

**Offline fixture:** `tests/fixtures/filing_tone/sentences.csv`, about 90 synthetic
finance sentences I write (30 per class), labeled "synthetic" in the file header, and
`tests/fixtures/filing_tone/sample_10k.html`, a synthetic 10-K-shaped document with
Item 1A and Item 7 headings. Tests and the CI smoke run never download anything.

**Application data:** the latest 10-K per company from SEC EDGAR, found through the
submissions API (`https://data.sec.gov/submissions/CIK##########.json`) and fetched from
the archive. Companies: AEHR and the pitch peers that file 10-Ks (FORM, KLIC, COHU, ONTO,
TER). CAMT files a 20-F and is excluded with a note.

## Models

All three share one evaluation harness so the comparison is apples to apples.

| Tier | What | Why it is there |
| --- | --- | --- |
| baseline | TF-IDF unigrams + bigrams, logistic regression (scikit-learn) | The honest floor; a reviewer should see the neural models beat it |
| cnn | From-scratch PyTorch text CNN: own regex tokenizer, vocab built from the train split, embedding layer, three parallel 1-D convolutions (widths 3/4/5), max-pool over time, dropout, linear head | Shows the fundamentals: `Dataset`, `DataLoader`, padding collate, `nn.Module`, explicit loop |
| distilbert | `distilbert-base-uncased` with a 3-class head via Hugging Face `transformers`, fine-tuned end to end | Shows transfer learning, the thing that actually wins on this task |

**Training loop** (shared by cnn and distilbert): AdamW, linear warmup then decay,
cross-entropy with class weights for the neutral-heavy label distribution, gradient
clipping, early stopping on validation macro-F1 with patience, best-checkpoint restore,
fixed seeds for Python, NumPy, and torch. Device: CUDA if available, else CPU.

**Metrics:** accuracy, macro-F1, per-class precision/recall/F1, confusion matrix. Computed
in `finlib/nlp/evaluate.py` from scratch with NumPy and tested against a hand-built
confusion matrix (scikit-learn is used in a test as an oracle, not in the metric code).

**Run log:** every training run writes `output/filing_tone/runs/<model>-<timestamp>.json`
holding config, seed, device, epoch history, test metrics, and wall time. The README
results table is rendered from the latest run per model by `python -m projects.filing_tone
results`.

## SEC application

`finlib/nlp/sec_text.py`:

1. `latest_10k(ticker)`: ticker to CIK via the SEC company_tickers table (same convention
   as `finlib/providers/edgar.py`: `SEC_IDENTITY` User-Agent, under 10 requests/second,
   JSON cache), then the submissions index for the newest 10-K accession and primary
   document URL. Offline: reads the fixture HTML.
2. `extract_sections(html)`: strips tags, normalizes whitespace, finds Item 1A and Item 7
   by heading regex, and returns the text of each. Returns an empty string with a warning
   when a section is not found instead of guessing.
3. `split_sentences(text)`: regex sentence splitter tuned for filings (keeps "Inc.",
   "U.S.", decimals intact). Drops fragments under 6 words and boilerplate like page numbers.
4. `score_sentences(model, sentences)`: batched inference, returns class probabilities.
5. `filing_tone(probs)`: net tone = (share positive − share negative), plus the five
   sentences with the highest positive and highest negative probability.

`tone_snapshot.json` holds, per company: ticker, company name, fiscal year end, accession,
document URL, sentence counts per section, net tone per section, and the driver sentences.
It carries the model name, run id, and snapshot date. Every number in the app is read from
this file.

## Code layout

```
finlib/nlp/
  __init__.py
  data.py        load_phrasebank(), load_fixture(), stratified_split(), label names
  tokenize.py    RegexTokenizer, Vocab (build, encode, decode, pad id)
  models.py      TextCNN(nn.Module); build_distilbert() wrapper
  train.py       TrainConfig, train_loop(), set_seed(), choose_device(), RunLog
  evaluate.py    confusion_matrix(), precision_recall_f1(), summary()
  baseline.py    tfidf_logreg() fit/predict
  sec_text.py    latest_10k(), extract_sections(), split_sentences(), filing_tone()
projects/filing_tone/
  __init__.py, __main__.py
  cli.py         data | train | evaluate | results | score | snapshot | summary
  README.md      what it is, results table, resume line, how to run
  tone_snapshot.json
app/pages/6_Filing_Tone.py
docs/filing_tone/WALKTHROUGH.md
tests/test_filing_tone.py, tests/test_filing_tone_app.py (in test_app.py pattern)
tests/fixtures/filing_tone/sentences.csv, sample_10k.html
.github/workflows/ml.yml
```

`finlib/models/` and `finlib/providers/` are not touched. `finlib/nlp/` is a new package.

Dependencies: a new optional extra `ml = ["torch>=2.3", "transformers>=4.44",
"datasets>=2.20", "scikit-learn>=1.4"]`. Not added to `requirements.txt` or the Docker
image; the app page only needs the committed snapshot.

## Streamlit page

`app/pages/6_Filing_Tone.py`, registered in `Home.py`, eyebrow `06 · Filing tone`.
Follows DESIGN.md anatomy: masthead, gray provenance badge ("Snapshot · SEC EDGAR 10-Ks,
scored by DistilBERT fine-tuned on Financial PhraseBank, <date>"), KPI strip (AEHR net
tone for Risk Factors and MD&A, peer median), Exhibit 1 bar chart of net MD&A tone by
company, Exhibit 2 the same for Risk Factors, Exhibit 3 the model comparison table from the
run log, then an expander per company with its driver sentences. Source captions under
every exhibit. No torch import anywhere in `app/`.

## Tests

`tests/test_filing_tone.py` (`pytest.importorskip("torch")` at top for the model tests;
pure-Python tests in a separate module-level section so they always run):

- Tokenizer: lowercases, splits punctuation, keeps numbers; vocab round-trips encode/decode;
  unknown word maps to `<unk>`; pad id is 0.
- Collate: a batch of unequal lengths pads to the longest, mask shape matches.
- Metrics: confusion matrix and per-class F1 equal scikit-learn on a hand-built example.
- TextCNN: forward pass shape is (batch, 3); training 30 epochs on the 90-sentence fixture
  reaches train accuracy above 0.95 (it can memorize), proving the loop learns.
- Baseline: fits and predicts on the fixture with the same label encoding.
- Section extraction: finds Item 1A and Item 7 in the fixture HTML; a document without
  Item 7 yields an empty string and a warning, never a guess.
- Sentence splitter: does not split on "Inc." or "U.S." or "3.5%".
- Filing tone: net tone from hand-built probabilities equals the hand calculation; driver
  sentences are the top-k by probability.
- Snapshot: the committed `tone_snapshot.json` is well-formed and every company's net tone
  equals `filing_tone()` recomputed from the stored per-section counts.
- App page: renders on the snapshot and its KPI strip shows the snapshot's AEHR numbers.

CI: `ml.yml` installs `.[dev,ml]` (CPU torch) on Python 3.12 and runs
`tests/test_filing_tone.py`. The existing `test.yml` is unchanged and skips torch tests.

## Documentation for learning

`docs/filing_tone/WALKTHROUGH.md`, written for someone meeting PyTorch for the first time,
in the order the data flows: what a label is and why PhraseBank; tokens and vocab; why
sentences become padded integer tensors; what an embedding is; what the CNN's convolutions
look for; what a loss is, what a gradient step does, what an epoch is; why we hold out
validation and test data and what early stopping prevents; what fine-tuning a pretrained
transformer means and why it wins; how the trained classifier becomes a filing tone number;
how to read the results table. Each section names the file and function that does it.

The README is the reviewer's view: results, resume line, choices to defend (why sentence
level, why net tone, why DistilBERT not FinBERT, why no LM dictionary baseline, what
the ablation shows), and limits (news sentences vs filing prose, neutral dominance,
NC license).

## Out of scope

Volatility or return prediction, risk-factor topic labeling, model serving as an API,
FinBERT or larger models, GPU-specific optimizations, labeling any filing sentences by hand.

## Decisions

| Decision | Rationale |
| --- | --- |
| PyTorch over TensorFlow | Industry and research default; an explicit loop shows more understanding than `fit()` |
| Sentence-level model applied to filings, not a filing-level model | Only sentence labels exist; aggregation is transparent and explainable |
| TF-IDF baseline instead of Loughran-McDonald dictionary | The LM lists carry a non-commercial license and a download step; TF-IDF is reproducible from the data alone. LM is noted as a follow-up |
| DistilBERT rather than FinBERT | FinBERT was trained on PhraseBank itself, so fine-tuning it would be a weaker demonstration; DistilBERT shows the transfer step honestly |
| Committed snapshot powers the app | Keeps torch out of the Streamlit Cloud and Docker images; same pattern as the pitch |
| Metrics implemented by hand, tested against scikit-learn | Shows understanding; scikit-learn stays an oracle |
| Separate CI job for ML tests | CPU torch is a 200 MB install; keeps the main job fast and unchanged |
