# Filing Tone, explained from the data up

This is the learner's guide to `projects/filing_tone`. It follows the data from a labeled sentence to a number on the Streamlit page, and names the file and function that does each step. Read it with the code open; every module's docstring repeats the idea in two sentences.

## 1. The problem, and what a label is

We want a program that reads a finance sentence and says whether it is good news, bad news, or neither. Nobody can write that program by hand, so we *learn* it from examples: sentences paired with the right answer, called a **label**.

Financial PhraseBank (Malo et al., 2014) is 4,846 sentences from financial news, each read by several people with finance backgrounds who voted positive, negative or neutral. It comes in four versions by how many annotators agreed (50%, 66%, 75%, 100%). We train on the 50% set (all sentences) and also report on the 100% set, the clear-cut ones.

Labels are integers because models work on numbers: `negative = 0, neutral = 1, positive = 2` (`finlib/nlp/data.py`, `LABELS`).

## 2. Three splits, and why the test set is sacred

`stratified_split()` cuts the data 80/10/10:

- **train**: what the model learns from.
- **validation**: checked after each pass through the training data to decide when to stop. The model never trains on it.
- **test**: touched once, at the very end, for the number we report. Any decision made by looking at it (which settings, which model) leaks information and inflates the score.

"Stratified" means each split has the same mix of labels. The seed makes the split repeatable.

The CNN's settings were chosen by a sweep that looked only at validation scores. That is the rule: tune on validation, report on test.

## 3. From words to numbers: tokens and a vocabulary

A model cannot read letters. `tokenize.py` turns a sentence into **tokens** (words, numbers, punctuation) and a **vocabulary** gives each distinct token an integer id:

```
"Revenue rose 18.5%."  ->  ["revenue", "rose", "18.5%", "."]  ->  [412, 97, 2051, 5]
```

Two ids are reserved. `0` is **padding**, explained next. `1` is **unknown**, for any token the training data never contained. The vocabulary is built from the training split only, so the test sentences stay unseen.

DistilBERT has its own tokenizer that splits rare words into pieces ("burn-in" becomes "burn", "-", "in"), with 30,000 pieces learned from a huge corpus. We use it as is.

## 4. Batches, padding and masks

Updating the model after every single sentence is slow and noisy. Instead, `batches.py` groups 32 sentences into a **batch** and the model updates once per batch. A `Dataset` hands out one example; a `DataLoader` groups them and shuffles the order each pass.

Sentences in a batch have different lengths, but a tensor is rectangular. Short sentences are **padded** with id 0 out to the longest one in the batch, and an **attention mask** of 1s and 0s records which positions are real. The model's embedding for id 0 is held at zero so padding carries no signal.

## 5. The model from scratch: TextCNN

`models.py`, class `TextCNN`. Four layers, each a few lines:

1. **Embedding.** A table with one row of 200 numbers per vocabulary id. Looking up a sentence's ids gives a 200-wide vector per token. These numbers start random and are learned; by the end, words that behave alike ("fell", "declined", "dropped") have similar rows.
2. **Convolutions.** A filter is a small pattern over 3, 4 or 5 consecutive token vectors. It slides along the sentence and produces a high value wherever its pattern matches, so one filter might learn to fire on "ahead of guidance" and another on "fell short of". There are 200 filters of each width.
3. **Max-pooling.** For each filter, keep only its strongest response anywhere in the sentence. The question becomes "did this phrase occur?", not "where?".
4. **Dropout and the linear head.** During training, half the pooled features are randomly zeroed each batch so no single feature can carry the prediction (**dropout**). A linear layer then maps the 600 features to three scores, the **logits**. Softmax turns logits into probabilities that sum to one.

This is the architecture from Kim (2014). It is small (about 2 million weights), trains in seconds, and is a good first neural text model to read because every piece is visible.

## 6. The pretrained model: DistilBERT and fine-tuning

DistilBERT is a transformer trained by Hugging Face on English Wikipedia and books to predict masked-out words. Doing that well requires a working model of English. We take those weights, attach a fresh three-way head, and continue training the whole network on our 3,876 sentences. That is **fine-tuning**, and it is why 4k examples are enough: the model is not learning English, only which English is good or bad news.

`HFClassifier` wraps the Hugging Face model so it has the same `forward(input_ids, attention_mask) -> logits` signature as `TextCNN`. That one shared contract is what lets a single training loop drive both.

## 7. The training loop

`train.py`, `train_loop()`. One **epoch** is one pass over the training set. For each batch:

1. **Forward.** Run the model, get logits.
2. **Loss.** Cross-entropy measures how far the predicted probabilities are from the true label. It is zero if the model puts all its probability on the right class and grows as it puts probability elsewhere. Class weights make a mistake on a rare class (negative, 12% of the data) cost more, so the model cannot score well by over-predicting neutral.
3. **Backward.** `loss.backward()` computes the **gradient**: for every one of the millions of weights, the direction that would reduce the loss. PyTorch does this automatically (autograd).
4. **Step.** The optimizer (AdamW) moves every weight a small distance against its gradient. The size of that step is the **learning rate**. It warms up from zero over the first 10% of steps and then decays linearly, a standard schedule for transformers that also helps the CNN. Gradient clipping caps the size of any single step.

After each epoch, the loop scores the validation set. If macro-F1 has not improved for `patience` epochs it stops (**early stopping**) and restores the best epoch's weights. Without this, training loss keeps falling while validation gets worse: the model is memorizing the training sentences (**overfitting**). You can see this in any run log under `output/filing_tone/runs/`: `train_loss` goes down every epoch, `val_macro_f1` peaks and then drifts.

Every run writes a JSON log with the config, seed, device, per-epoch history and test metrics, so a result can be reproduced and compared.

## 8. Measuring: accuracy, F1 and the confusion matrix

`evaluate.py`, written in NumPy without a library, and tested against scikit-learn.

- **Accuracy** is the share of sentences predicted correctly. It is misleading when one class dominates: a model that always says "neutral" scores 60% on this data.
- For one class, say positive: **precision** is "of the sentences we called positive, how many were", and **recall** is "of the truly positive sentences, how many we caught". **F1** is their harmonic mean, high only when both are.
- **Macro-F1** averages F1 across the three classes with equal weight, so the rare negative class counts as much as neutral. It is the number the training loop watches and the one to compare models on.
- The **confusion matrix** is a 3×3 table of true label (rows) against predicted label (columns). Every mistake is visible in it.

## 9. The baseline

`baseline.py`. TF-IDF counts which words and word pairs appear in a sentence, down-weighting words that appear everywhere, and logistic regression learns one weight per word per class. No neural network, one second to train. The neural models have to beat it to justify their complexity; in this project the transformer does by a wide margin and the from-scratch CNN only matches it. That is the expected shape of the result on a small dataset, and it is reported as found.

## 10. From a classifier to a filing tone

`sec_text.py`, used by the `snapshot` command:

1. **Find the filing.** The SEC's submissions API lists every filing by a company; the newest form "10-K" gives an accession number and primary document. The fetcher uses the same etiquette as the rest of this repo (identity header, under 10 requests a second, cached).
2. **HTML to text.** Strip tags, drop scripts, put every block element on its own line.
3. **Cut the sections.** Item 1A (Risk Factors) and Item 7 (MD&A) each appear at least twice: in the table of contents and in the body. Among all heading-to-next-heading spans, the longest is the body. A heading only counts if it starts a line, because filings say "see Item 8" mid-sentence.
4. **Split sentences.** On `.`, `!`, `?` followed by a capital, after protecting "Inc.", "U.S." and the like. Keep sentences of 6 to 120 words that are mostly letters, which drops tables and page furniture.
5. **Score and aggregate.** The model gives each sentence three probabilities. Net tone = share of sentences whose top class is positive − share negative, in [−1, 1]. The five most confident positive and negative sentences are kept as evidence.

The page does not compute anything. It reads `tone_snapshot.json` and `results.json`, which is why it runs without torch.

## 11. Reading the results table

- Compare **macro-F1** between models, and **accuracy** only as a sanity check.
- Per-class F1 shows where a model struggles: negative is lowest for every model because it is rare.
- The **all-agree** column is the easier subset the original paper reports on; it makes our numbers comparable to published ones.
- Train time shows the trade: DistilBERT is 10 times slower than the CNN and far better; on CPU it is minutes, not seconds.

## 12. Where to go next

Things this project deliberately leaves out, each a weekend: a tone-by-year series per company (the fetcher already sees every 10-K), the Loughran-McDonald dictionary as a second baseline, calibration on hand-labeled filing sentences, and a larger model (RoBERTa-base) to see how much the extra size buys.

## References

- Malo, P., Sinha, A., Korhonen, P., Wallenius, J., Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology*.
- Kim, Y. (2014). Convolutional neural networks for sentence classification. *EMNLP*.
- Sanh, V., Debut, L., Chaumond, J., Wolf, T. (2019). DistilBERT, a distilled version of BERT. *arXiv:1910.01108*.
