# Filing Tone: a self-paced course

Ten lessons that teach the machine learning behind `projects/filing_tone`, using the real code and the real numbers from this repo. Each lesson has the same shape:

- **The idea**, in plain words.
- **A picture**, drawn in text so it renders anywhere.
- **In the code**, the file and function that does it.
- **Try it**, one command you can run in the repo.
- **Check yourself**, two or three questions. Answers are hidden under "Answer"; think before you open them.

Read it top to bottom the first time. The lessons build on each other. Budget about 15 minutes per lesson, and run the "Try it" commands: the project is small enough that every number here can be reproduced on your machine in under a minute.

Companion files: `docs/filing_tone/WALKTHROUGH.md` (the same ideas, written as a reference), `finlib/nlp/` (the code, every module starts with a docstring that explains its concept), `projects/filing_tone/README.md` (results and design choices).

---

## Lesson 1. The job: classify one sentence

### The idea

Everything in this project reduces to one question, asked thousands of times:

> Given one sentence from a financial document, is its tone **negative**, **neutral**, or **positive**?

A program that answers a multiple-choice question is a **classifier**. The choices are the **classes**. Nobody can write the rules for this by hand ("fell" is negative, unless it is "costs fell"), so instead we *learn* the rules from examples that humans have already answered. Each human answer is a **label**.

Our labeled examples come from the Financial PhraseBank: 4,846 sentences from financial news, each read by several people with finance training who voted on the tone. The model's job is to imitate those votes. It is not discovering truth. If the annotators were biased, the model inherits the bias.

The model never sees the words "negative", "neutral", "positive". It sees 0, 1, 2. The order is arbitrary, but once chosen it must never change, because every number the model outputs is read back through it.

### A picture

```
          sentence                                   label   id
  ─────────────────────────────────────────────────  ───────  ──
  "Revenue fell 22% as the largest customer paused   negative  0
   orders."
  "The company's fiscal year ends on the last Friday neutral   1
   in May."
  "The board approved a 15% increase in the          positive  2
   quarterly dividend."

  PhraseBank label mix (all 4,846 sentences):

  negative  ████████                              12%
  neutral   ████████████████████████████████████  60%
  positive  █████████████████                     28%
```

The imbalance in that bar chart matters later (Lessons 6 and 7). A model that always says "neutral" is right 60% of the time while knowing nothing.

### In the code

- `finlib/nlp/data.py`: `LABELS = ["negative", "neutral", "positive"]`, the `Example` dataclass (one sentence plus its integer label), `load_phrasebank()` which reads the zip from the Hugging Face hub.
- `tests/fixtures/filing_tone/sentences.csv`: 90 synthetic sentences, 30 per class, so the tests run offline without the real dataset.

### Try it

```bash
python -m projects.filing_tone.cli data
```

It prints the label counts per split. Compare them to the bar chart above.

### Check yourself

**Q1.** Why can't the model's output be the word "negative" directly?

<details><summary>Answer</summary>
A neural network is arithmetic: multiplications and additions on numbers. Its output is three numbers, one per class. We pick the largest and look up its position in the LABELS list. The list is the only place the words exist.
</details>

**Q2.** Neutral is the hardest class for every model in this project. Why might that be, given how the classes are defined?

<details><summary>Answer</summary>
Negative and positive are defined by presence: certain words and phrases ("collapsed", "ahead of guidance") signal them. Neutral is defined by absence: nothing in the sentence pushes either way. A model has to learn "none of the signals fired", which is harder than learning any one signal. Neutral is also the class annotators disagree on most.
</details>

**Q3.** A colleague says the model is 60% accurate. Is that good?

<details><summary>Answer</summary>
No. Always answering "neutral" gets 60% on this data. Accuracy has to be compared to that floor, and Lesson 7 introduces a metric (macro-F1) that does not reward the lazy answer.
</details>

---

## Lesson 2. Three splits, and why the test set is sacred

### The idea

If you study the exact questions that will be on the exam, your exam score says nothing about what you learned. The same applies to a model. We cut the labeled sentences into three piles and use each for exactly one purpose:

- **Train** (80%): the model learns from these, seeing each sentence and its label many times.
- **Validation** (10%): the model never learns from these. After each pass over the training pile we score the model here to decide *when to stop* and *which settings are best*. Humans make decisions from this number.
- **Test** (10%): touched once, at the very end, to report the final number. Nobody changes anything after seeing it.

Why three and not two? Every time you look at a score and change something in response, a little information about that pile leaks into your choices. After enough looks, the validation score is no longer an honest estimate of how the model handles sentences it has never seen. The test set stays honest by staying untouched.

**Stratified** splitting keeps the label mix the same in all three piles. With only 12% negatives, a random cut could leave the test pile with too few negatives to measure anything. The **seed** (42) makes the shuffle repeatable, so the same sentences land in the same piles every run and two models are compared on identical test sentences.

### A picture

```
  4,846 labeled sentences, shuffled once with seed 42
  ┌──────────────────────────────────────────────────┬──────┬──────┐
  │                   train  3,876                   │ val  │ test │
  │                                                  │  484 │  486 │
  └──────────────────────────────────────────────────┴──────┴──────┘
         model learns here                             ▲        ▲
                                                       │        │
                                  "stop now?" "which settings?"  final score,
                                                                 looked at once

  each pile keeps the 12 / 60 / 28 label mix
```

### In the code

- `finlib/nlp/data.py`: `stratified_split(examples, seed=42)` returns the three lists. It groups by label, shuffles each group, and cuts each group 80/10/10 so the mix is preserved.

### Try it

```bash
python -m projects.filing_tone.cli data
```

Check that each split's percentages match within a point.

### Check yourself

**Q1.** You train on all 4,846 sentences and report accuracy on those same sentences: 99%. What is wrong with that number?

<details><summary>Answer</summary>
It measures memory, not understanding. The model has seen every one of those sentences with its answer. The number says nothing about how it will do on a 10-K sentence it has never seen, which is the only thing we care about. This is why the test pile is kept out of training.
</details>

**Q2.** You try 30 different model settings and for each one you check the test score, keeping the best. Why is the best score now optimistic?

<details><summary>Answer</summary>
With 30 tries, some settings will score well on the test pile by luck, fitting its particular quirks. Picking the maximum selects for that luck. The true performance on new sentences is lower than the number you kept. This is exactly what the validation pile is for: do the 30 tries there, then score the single winner on test once.
</details>

**Q3.** Honest disclosure: the CNN settings in this repo came from a 30-run sweep on the validation split, but the sweep was run *after* one CNN test score had already been seen. Is the CNN's test number compromised, and by how much?

<details><summary>Answer</summary>
Mildly. The sweep itself only looked at validation, so no test information drove the choice of settings directly. But knowing one test score beforehand could have nudged which settings seemed worth trying. The effect is small and is disclosed in the README. The DistilBERT headline number was never part of a sweep.
</details>

---

## Lesson 3. From words to numbers: tokens and a vocabulary

### The idea

A model cannot read letters. Two steps turn a sentence into integers:

1. A **tokenizer** chops the sentence into pieces called **tokens**: words, numbers, punctuation.
2. A **vocabulary** assigns each distinct token an integer id.

Two ids are reserved. Id 0 is `<pad>`, used in Lesson 4 to fill short sentences out to a common length. Id 1 is `<unk>`, "unknown", for any token the training pile never contained. If a test sentence says "amortization" and no training sentence did, the model sees `<unk>` there.

The vocabulary is built from the **training pile only**. If test sentences contributed words, the model would have a tiny bit of information about the test set before it was supposed to.

DistilBERT (Lesson 6) brings its own tokenizer that splits rare words into sub-word pieces ("burn-in" becomes "burn", "-", "in"), with 30,000 pieces learned from a huge corpus. Only the from-scratch model uses ours.

### A picture

```
  "Revenue rose 18.5%."
        │  RegexTokenizer: lowercase, then words | numbers(+%) | single symbols
        ▼
  ["revenue", "rose", "18.5%", "."]
        │  Vocab.stoi  (string -> integer)
        ▼
  [ 412,  97,  2051,  5 ]

  Vocab built from the 3,876 training sentences: 9,562 distinct tokens

  id   token        why it is here
  ──   ──────────   ───────────────────────────────
   0   <pad>        reserved: filler
   1   <unk>        reserved: never seen in training
   2   the          most frequent token
   3   .
   4   ,
   ...
  9561 (rarest)     appeared once in training
```

### In the code

- `finlib/nlp/tokenize.py`: `RegexTokenizer` (one regular expression), `Vocab.build(texts, tokenizer)` (count tokens, sort by frequency, assign ids), `Vocab.encode(text)` and `Vocab.decode(ids)`.

### Try it

```bash
python -c "
from finlib.nlp.tokenize import RegexTokenizer, Vocab
tok = RegexTokenizer()
v = Vocab.build(['Revenue rose 18.5%.', 'Revenue fell.'], tok)
print(v.itos)
print(v.encode('Revenue collapsed 40%.'))
"
```

Notice which tokens come back as id 1.

### Check yourself

**Q1.** The training vocabulary has 9,562 tokens. A 10-K sentence contains the word "amortization", which never appeared in PhraseBank. What does the CNN see for that word, and what does that imply?

<details><summary>Answer</summary>
It sees id 1, `<unk>`. The CNN learns one shared meaning for "a word I don't know", so every unknown word looks the same to it. On documents whose vocabulary differs from the training news (10-Ks are more technical), this is a real weakness, and one reason DistilBERT's sub-word tokenizer does better: it almost never hits an unknown.
</details>

**Q2.** Why does `Vocab.build` sort tokens by frequency and then alphabetically before assigning ids?

<details><summary>Answer</summary>
So the ids are deterministic. If two tokens have the same count, alphabetical order breaks the tie the same way every run. A vocabulary whose ids changed between runs would make a saved model unusable, since the model's first layer is indexed by those ids.
</details>

---

## Lesson 4. Batches, padding and masks

### The idea

Updating the model after every single sentence is slow and noisy. Instead we group sentences into a **batch** (32 here) and update once per batch. The GPU processes a batch as one rectangular block of numbers, which is why it is fast.

The catch: a rectangle needs every row the same length, and sentences are not. So we **pad** every sentence in the batch with id 0 out to the length of the longest one, and we hand the model an **attention mask**: a 1 where there is a real token, a 0 where there is padding. The mask lets the model ignore the filler.

Two PyTorch objects do this. A `Dataset` answers "give me example number *i*". A `DataLoader` pulls 32 of them, calls a **collate** function to pad them into a rectangle, and shuffles the order every pass over the data so the model does not learn anything from sequence.

### A picture

```
  three sentences, different lengths:

  [ 412,  97, 2051,  5 ]
  [  58, 310,   4 ]
  [ 412,  14,  73, 220, 19, 5 ]

  padded to the longest (6)          attention mask
  ┌────┬────┬────┬────┬────┬────┐    ┌───┬───┬───┬───┬───┬───┐
  │412 │ 97 │2051│  5 │  0 │  0 │    │ 1 │ 1 │ 1 │ 1 │ 0 │ 0 │
  │ 58 │310 │  4 │  0 │  0 │  0 │    │ 1 │ 1 │ 1 │ 0 │ 0 │ 0 │
  │412 │ 14 │ 73 │220 │ 19 │  5 │    │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │
  └────┴────┴────┴────┴────┴────┘    └───┴───┴───┴───┴───┴───┘
     tensor shape (3, 6)                 tensor shape (3, 6)
```

### In the code

- `finlib/nlp/batches.py`: `SentenceDataset`, `vocab_encoder(vocab, max_len)` and `hf_encoder(tokenizer, max_len)` (two ways to turn text into ids, one per model), `make_collate` (the padding), `make_loader` (wires the DataLoader).
- `max_len` caps a sentence at 64 tokens for the CNN and 128 for DistilBERT. Longer sentences are cut.

### Try it

```bash
python -c "
import torch
from finlib.nlp.tokenize import RegexTokenizer, Vocab
from finlib.nlp.batches import vocab_encoder
texts = ['Revenue rose 18.5%.', 'Costs fell.', 'Backlog at year end was a record.']
v = Vocab.build(texts, RegexTokenizer())
b = vocab_encoder(v, 64)(texts)
print(b['input_ids']); print(b['attention_mask'])
"
```

### Check yourself

**Q1.** A short sentence lands in a batch with a very long one and gets 40 padding tokens. Should its predicted tone change compared to when it sits in a batch of short sentences?

<details><summary>Answer</summary>
No. The prediction for a sentence must depend only on that sentence. The mask exists to guarantee this. The repo had exactly this bug in an early version: the CNN's pooling step was reading padding positions, so a sentence's score shifted with the length of its batch-mates. The fix was to zero out any window that touches no real token before pooling (see the comment in `TextCNN.forward`).
</details>

**Q2.** Why shuffle the training data every pass, but never the validation or test data?

<details><summary>Answer</summary>
Shuffling training data prevents the model from seeing, say, all the negatives in a row and drifting toward "negative" for a while. It also makes each batch a different mix, which gives a noisier but less biased update. Validation and test are only scored, not learned from, so order does not matter; keeping it fixed makes the predictions line up with the labels.
</details>

---

## Lesson 5. The model from scratch: TextCNN

### The idea

This is the model built with raw PyTorch pieces, so you can see every part. It has four stages.

**Embedding.** Each token id becomes a vector of 200 numbers that the model is free to adjust. Before training the vectors are random. Training moves words that behave alike close together: "fell", "declined", "dropped" end up near each other because they predict the same label. A sentence of 20 tokens is now a 20 × 200 grid.

**Convolutions.** A **filter** is a small window of 3, 4 or 5 tokens wide that slides along the sentence and computes one number at each position: how strongly this stretch of text matches the pattern the filter has learned. One filter might learn to fire on "ahead of guidance", another on "fell short of". There are 200 filters of each width, and they find their phrase anywhere in the sentence. That position-independence is the reason convolutions suit text.

**Max-pooling.** For each filter, keep only its single strongest response. The question becomes "did this phrase appear anywhere?" rather than "where?". After pooling, every sentence, whatever its length, is a vector of 600 numbers (3 widths × 200 filters).

**Dropout, then linear.** During training, randomly zero 30% of those 600 numbers each time, so the model cannot lean on any one feature. At prediction time dropout is off. A final linear layer turns the 600 numbers into 3 scores, called **logits**. Softmax turns the logits into probabilities that sum to 1; the largest is the prediction.

### A picture

```
  ids     [ 412 , 97 , 2051 , 5 , 0 , 0 ]
             │     │     │     │
  embed    ┌─┴─┐ ┌─┴─┐ ┌─┴─┐ ┌─┴─┐          each column: 200 numbers
           │   │ │   │ │   │ │   │ ...
           └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘
             │     │     │     │
  conv     [=====width 3=====]              a filter slides across,
                 [=====width 3=====]        one score per position
                       [=====width 3=====]
             ... 200 filters each for widths 3, 4, 5 ...

  max-pool   for each filter keep its biggest score
             ▼
           [ 600 numbers ]   (3 widths × 200 filters)
             │  dropout (training only): zero 30% at random
             ▼
  linear   [ 600 ] ──► [ 3 logits ]  e.g. [ -1.2 , 0.4 , 2.9 ]
             │  softmax
             ▼
           [ 0.01 , 0.08 , 0.91 ]   negative / neutral / positive
```

Parameters, roughly: embedding 9,562 × 200 ≈ 1.9 M numbers, convolutions ≈ 0.5 M, linear 600 × 3. Almost all the capacity is in the embedding.

### In the code

- `finlib/nlp/models.py`: `TextCNN.__init__` builds the four stages (`nn.Embedding`, three `nn.Conv1d`, `nn.Dropout`, `nn.Linear`). `TextCNN.forward` runs them. Read the comment about `touches`: that is the mask from Lesson 4 applied to pooling.
- `projects/filing_tone/cli.py`: `CNN_ARCH = dict(embed_dim=200, channels=200, dropout=0.3)`.

### Try it

```bash
python -c "
import torch
from finlib.nlp.models import TextCNN
m = TextCNN(vocab_size=100, embed_dim=8, channels=4)
ids = torch.tensor([[5, 9, 2, 0, 0]]); mask = torch.tensor([[1, 1, 1, 0, 0]])
print(m(ids, mask))                       # 3 logits, untrained so meaningless
print(sum(p.numel() for p in m.parameters()), 'parameters')
"
```

### Check yourself

**Q1.** Why does max-pooling make the model indifferent to *where* a phrase appears?

<details><summary>Answer</summary>
A filter produces one score per position. Max-pooling throws away all but the largest, so the output records "the strongest match was this strong" and forgets its position. "Revenue fell" at the start of a sentence and at the end produce the same pooled value.
</details>

**Q2.** Dropout zeroes 30% of features during training. Why turn it off at prediction time, and what does `model.eval()` have to do with it?

<details><summary>Answer</summary>
Dropout is a training trick: forcing the model to work with random pieces missing stops it from relying on any one feature. At prediction time we want the full model, deterministic, using everything it learned. `model.eval()` switches dropout (and similar training-only layers) off; `model.train()` switches them back on. The training loop calls both at the right moments.
</details>

**Q3.** Look at the logits in the picture, `[-1.2, 0.4, 2.9]`. Which class is predicted, and what would change if all three logits had 10 added to them?

<details><summary>Answer</summary>
Positive (index 2, the largest). Nothing would change: softmax only depends on differences between logits, so adding the same constant to all three gives identical probabilities. Logits are relative scores, not absolute ones.
</details>

---

## Lesson 6. The pretrained model: DistilBERT and fine-tuning

### The idea

The TextCNN starts knowing nothing about English and has to learn it from 3,876 sentences. That is not much. DistilBERT starts already knowing English.

Hugging Face trained DistilBERT on English Wikipedia and a large book corpus with a fill-in-the-blank task: hide a word, predict it. To do that well, the network had to learn grammar, word meaning and context. We download those weights (about 260 MB) and get a model that already represents a sentence as a rich set of numbers.

We then **fine-tune**: attach a fresh 3-way output layer and continue training *every* weight on our labeled sentences, but gently (a learning rate about 100 times smaller than the CNN's) and briefly (4 passes instead of up to 30). The pretrained knowledge is preserved; it just gets steered toward our three classes. This is why a few thousand examples go a long way, and why DistilBERT wins by 7 points of accuracy here.

The cost is size: about 66 million parameters versus the CNN's 2.4 million, and a dependency on the `transformers` library and the download. That is why the Streamlit page never loads either model. It reads numbers the CLI wrote to JSON, so the app stays torch-free.

### A picture

```
  TextCNN                              DistilBERT
  ───────────────────────              ──────────────────────────────────
  random weights                       weights pretrained on Wikipedia +
       │                                   books (fill-in-the-blank)
       │ learn English AND tone             │
       │ from 3,876 sentences               │ attach new 3-way head
       ▼                                    │ fine-tune ALL weights gently
  2.4 M parameters                          ▼
  6 s on GPU                           66 M parameters, 26 s on GPU
  78.0% / F1 0.719                     85.0% / F1 0.815

  shared interface:  forward(input_ids, attention_mask) -> 3 logits
```

### In the code

- `finlib/nlp/models.py`: `build_distilbert()` downloads tokenizer and model; `HFClassifier` wraps it so it returns logits exactly like `TextCNN`, which is what lets one training loop serve both.
- `projects/filing_tone/cli.py`: `_fit_distilbert` with epochs 4, lr 3e-5, max_len 128, patience 2.

### Try it

```bash
python -m projects.filing_tone.cli score --model distilbert --text "Orders slumped 40% as customers delayed capital spending."
python -m projects.filing_tone.cli score --model cnn --text "Orders slumped 40% as customers delayed capital spending."
```

Run both and compare. On this machine DistilBERT says negative at 0.99, while the CNN says **neutral** (0.65) even though "slumped" is in its vocabulary. Change "slumped" to "fell" and the CNN flips to negative at 1.00. The CNN learned "fell" from many training examples and "slumped" from a handful; DistilBERT already knew the two words mean the same thing before it ever saw PhraseBank. That is fine-tuning's advantage in one sentence.

### Check yourself

**Q1.** Why is the DistilBERT learning rate (3e-5) so much smaller than the CNN's (2e-3)?

<details><summary>Answer</summary>
Its weights already encode something valuable. Large steps would overwrite that knowledge in the first few batches, a failure called catastrophic forgetting. Small steps adjust the representation toward our task while keeping what it learned about English. The CNN's weights are random, so there is nothing to protect.
</details>

**Q2.** Which design decision lets `train_loop` train both models without an `if model_type == ...` branch anywhere?

<details><summary>Answer</summary>
Both models expose the same calling convention: `forward(input_ids, attention_mask)` returning a tensor of 3 logits. `HFClassifier` exists only to unwrap Hugging Face's output object into that shape. The loop never knows which model it holds.
</details>

---

## Lesson 7. The training loop

### The idea

Training is a loop with four steps per batch, repeated over the data for several **epochs** (one epoch is one full pass over the training pile).

1. **Forward.** Run the batch through the model, get logits.
2. **Loss.** Compare the logits to the true labels with **cross-entropy**: a single number that is zero when the model puts all its probability on the right class and grows as it puts probability elsewhere. Lower is better.
3. **Backward.** PyTorch's autograd computes, for every one of the millions of weights, the **gradient**: the direction and amount by which nudging that weight would change the loss.
4. **Step.** The **optimizer** (AdamW) moves every weight a small distance against its gradient. The size of that step is the **learning rate**.

Three refinements make this work in practice:

- **Class weights.** With only 12% negatives, the loss would barely notice the model ignoring them. Each class's mistakes are weighted by the inverse of its frequency, so a missed negative costs about five times a missed neutral.
- **Learning-rate schedule.** The rate starts at zero, rises over the first 10% of steps (warm-up, so random early gradients do not fling the weights around), then decays linearly to zero (so the final steps settle rather than bounce).
- **Early stopping.** After every epoch, score the validation pile. Keep a copy of the weights whenever the score is a new best. If it has not improved for `patience` epochs (4 for the CNN), stop and restore that best copy.

Early stopping guards against **overfitting**, the most important failure in machine learning. Watch the CNN's actual training curve below: training loss keeps falling all the way to 0.016 (it is memorizing the training sentences), but validation F1 peaks at epoch 13 and then wanders. Past that point the model is getting better at the training pile and no better at anything else.

### A picture

The CNN run that produced the committed results, from `output/filing_tone/runs/`:

```
  epoch  train_loss  val_F1    learning rate
  ─────  ──────────  ──────    ─────────────
    1      1.051     0.536     0.00067  ◄ warm-up: rate rising
    2      0.754     0.652     0.00133
    3      0.537     0.550     0.00200  ◄ peak rate, then decay
    4      0.440     0.594     0.00193
    5      0.296     0.710     0.00185
    6      0.217     0.721     0.00178
    7      0.162     0.692     0.00170
    8      0.127     0.669     0.00163
    9      0.100     0.701     0.00156
   10      0.125     0.730     0.00148
   11      0.051     0.700     0.00141
   12      0.057     0.711     0.00133
   13      0.044     0.745     0.00126  ◄ best validation F1: weights saved
   14      0.038     0.723     0.00119
   15      0.017     0.742     0.00111
   16      0.030     0.736     0.00104
   17      0.016     0.712     0.00096  ◄ 4 epochs without a new best: stop,
                                           restore epoch-13 weights

  train loss  ▇▆▅▄▃▂▂▁▁▁▁▁▁▁▁▁▁    keeps falling (memorizing)
  val F1      ▃▅▃▄▆▆▅▅▆▆▆▆▇▆▇▇▆    plateaus around epoch 10-15
```

### In the code

- `finlib/nlp/train.py`: `train_loop` is 40 lines and contains every step above, in order. Read it alongside this lesson. `class_weight_tensor` computes the weights. The `LambdaLR` line is the warm-up/decay schedule. `RunLog` writes the table above to JSON.
- `TrainConfig` holds every knob: epochs, batch size, learning rate, weight decay, warm-up fraction, patience, gradient clip, seed.

### Try it

Train the CNN on the small fixture (a few seconds, CPU is fine) and read the history:

```bash
python -m projects.filing_tone.cli train --model cnn --fixture --epochs 15
python -c "
import json, glob
p = sorted(glob.glob('output/filing_tone/fixture/runs/cnn-*.json'))[-1]
for h in json.load(open(p))['history']: print(h['epoch'], round(h['train_loss'],3), round(h['val_macro_f1'],3))
"
```

### Check yourself

**Q1.** In the table, training loss at epoch 17 is 0.016, the lowest ever. Why do we *not* keep the epoch-17 weights?

<details><summary>Answer</summary>
Training loss measures fit to the training pile, which the model has now largely memorized. Validation F1, which measures sentences the model has not learned from, was higher at epoch 13. We want the model that generalizes, so we keep epoch 13. The gap between the two curves is overfitting made visible.
</details>

**Q2.** What would happen without class weights on a dataset that is 60% neutral?

<details><summary>Answer</summary>
The cheapest way to lower average loss is to get the common class right, so the model drifts toward predicting neutral and under-predicts negatives. Accuracy could look fine (high 70s) while negative recall collapses. Class weights make each negative mistake expensive enough that the model has to learn the class.
</details>

**Q3.** `opt.zero_grad()` is called before every `loss.backward()`. What goes wrong if you forget it?

<details><summary>Answer</summary>
PyTorch accumulates gradients by default: each `backward()` adds to whatever is already stored. Without clearing, batch 2's step would use batch 1 + batch 2's gradients, batch 3's would use all three, and the steps grow without bound. Clearing makes each step reflect only the current batch.
</details>

**Q4.** Why is the random seed set at the start of every run?

<details><summary>Answer</summary>
Weight initialization, data shuffling and dropout all draw random numbers. Fixing the seed makes a run repeatable, so a changed score can be attributed to a changed setting rather than to luck. On a GPU some operations are still not bit-for-bit deterministic, which is why the README notes that CNN results drift by a few hundredths between runs.
</details>

---

## Lesson 8. Measuring: accuracy, the confusion matrix, and F1

### The idea

**Accuracy** is the fraction of sentences the model got right. It hides a lot. The **confusion matrix** shows everything: rows are the true class, columns are the predicted class, and each cell counts sentences. The diagonal is the correct ones.

From the matrix, for each class:

- **Precision**: of the sentences the model *called* this class, what fraction really were? (Column-wise.) Low precision means false alarms.
- **Recall**: of the sentences that *really are* this class, what fraction did the model catch? (Row-wise.) Low recall means misses.
- **F1**: the harmonic mean of the two, so a model cannot score well by maximizing one and ignoring the other.

**Macro-F1** is the plain average of the three per-class F1 scores. Every class counts equally, regardless of size. That is why it is the number we optimize and report: the 12% negatives matter as much as the 60% neutrals. Accuracy would let a model hide a bad negative class behind a good neutral one.

The **ablation** row in the results reports the same metrics on only the test sentences where every annotator agreed on the label. Scores are much higher there (95% for DistilBERT), which tells you how much of the remaining error is on sentences humans themselves found ambiguous.

### A picture

DistilBERT on the 486 test sentences:

```
                         predicted
                    neg    neu    pos    │ row total  recall
                 ┌──────┬──────┬──────┐  │
   true  neg     │  47  │  12  │   2  │  │    61      47/61  = 0.77
         neu     │  13  │ 254  │  21  │  │   288     254/288 = 0.88
         pos     │   7  │  18  │ 112  │  │   137     112/137 = 0.82
                 └──────┴──────┴──────┘
   column total     67    284    135
   precision      47/67  254/284 112/135
                  =0.70   =0.89   =0.83

   accuracy  = (47 + 254 + 112) / 486 = 0.850
   F1 per class:  neg 0.73   neu 0.89   pos 0.82
   macro-F1  = (0.73 + 0.89 + 0.82) / 3 = 0.815
```

Read the off-diagonal cells: the biggest errors are between neutral and the two others, never negative-for-positive. The model rarely gets the sign wrong; it struggles with how strong a statement has to be before it stops being neutral.

For comparison, the same matrix for the TF-IDF baseline has 33 in the top-left cell instead of 47: it misses almost half the negatives.

### In the code

- `finlib/nlp/evaluate.py`: `confusion_matrix`, `precision_recall_f1`, `summary`. All written by hand with NumPy, and tested against scikit-learn's versions so you can trust them and read them.
- `projects/filing_tone/results.json`: the committed numbers above.

### Try it

```bash
python -m projects.filing_tone.cli results
python -c "
from finlib.nlp.evaluate import summary
print(summary([0,0,1,1,2,2], [0,1,1,1,2,0]))
"
```

Work the second one by hand first: six sentences, which are wrong, what are precision and recall per class?

### Check yourself

**Q1.** A model predicts "neutral" for all 486 test sentences. Compute its accuracy and macro-F1.

<details><summary>Answer</summary>
Accuracy = 288/486 = 0.59. Neutral: precision 288/486 = 0.59, recall 1.0, F1 ≈ 0.75. Negative and positive: recall 0, so F1 = 0. Macro-F1 = (0 + 0.75 + 0) / 3 = 0.25. Accuracy makes it look passable; macro-F1 exposes it.
</details>

**Q2.** For the negative class, DistilBERT's precision is 0.70 and recall is 0.77. Which kind of mistake is it making more often: calling a non-negative sentence negative, or missing a true negative?

<details><summary>Answer</summary>
Precision is the lower number, so false alarms are the slightly larger problem: 20 sentences were called negative that were not (13 neutral + 7 positive), versus 14 true negatives missed (12 + 2). Both are small; the class is just hard with only 61 examples.
</details>

**Q3.** Why is the all-agree ablation reported alongside the full test score instead of replacing it?

<details><summary>Answer</summary>
The all-agree subset is easier and smaller (232 sentences). Reporting only that would overstate the model. Reporting both says: on clear-cut sentences the model is right 19 times in 20; on the ambiguous ones, less so, and humans disagreed on those too. The pair is more honest than either alone.
</details>

---

## Lesson 9. The baseline, and why it exists

### The idea

Before trusting a neural network, ask what the simplest reasonable method achieves. Here it is **TF-IDF + logistic regression**:

- **TF-IDF** turns a sentence into a long sparse vector with one slot per word and word-pair in the training vocabulary. The value is high when the word is frequent in this sentence and rare across sentences, so "collapsed" scores higher than "the".
- **Logistic regression** learns one weight per slot and adds them up per class. It is a linear model: no phrases in context, no word order beyond pairs.

It trains in about one second on a CPU and scores 77.6% accuracy, macro-F1 0.711. The TextCNN, with 2.4 million parameters and a GPU, scores 78.0% and 0.719. That is a half-point gain for a lot of machinery, and it is the honest finding of the project: **on 4,800 sentences, a from-scratch neural network barely beats bag-of-words**. The pretrained model is what moves the needle, because it brings knowledge the data cannot supply.

A baseline turns "the model got 78%" into "the model got 78%, which is worth about nothing over a one-second linear model". The second statement is the one a reviewer wants.

### A picture

```
                     accuracy   macro-F1   train time   parameters
  ─────────────────  ────────   ────────   ──────────   ───────────
  always "neutral"     59.3%      0.25        0            0
  TF-IDF + logreg      77.6%      0.711       1 s          ~30 k weights
  TextCNN              78.0%      0.719       6 s GPU      2.4 M
  DistilBERT           85.0%      0.815      26 s GPU     66 M

          0.25 ──────────────────── 0.71 0.72 ──────── 0.815
  macro   │ lazy                      │ baseline, CNN      │ pretrained
  F1      ▼                           ▼                    ▼
```

### In the code

- `finlib/nlp/baseline.py`: `fit_baseline` (a scikit-learn pipeline: `TfidfVectorizer` with 1- and 2-grams, then `LogisticRegression` with balanced class weights), `baseline_probs`.

### Try it

```bash
python -m projects.filing_tone.cli train --model baseline
```

### Check yourself

**Q1.** The CNN beats the baseline by 0.4 points of accuracy, and the README says CNN results vary by a few hundredths of F1 between GPU runs. What should you conclude about the CNN versus the baseline?

<details><summary>Answer</summary>
That they are tied within noise. A difference smaller than run-to-run variation is not evidence of a better model. The honest claim is "the CNN matches a linear baseline", and the project says so.
</details>

**Q2.** Why does the baseline use 2-grams (word pairs) and not just single words?

<details><summary>Answer</summary>
Single words lose negation and direction: "not profitable" and "profitable" look the same. Pairs capture "fell short", "ahead of", "not sustain". It is the cheapest way to give a bag-of-words model a little phrase awareness, and part of why it is competitive with the CNN.
</details>

---

## Lesson 10. From a classifier to a filing tone

### The idea

Everything so far scores one sentence. The product scores a company. The bridge has four steps, all in `finlib/nlp/sec_text.py`:

1. **Fetch** the latest 10-K from SEC EDGAR (ticker → CIK → submissions → the filing's HTML), with the required identity header and a polite request rate.
2. **Extract two sections** from the HTML: Item 1A (Risk Factors) and Item 7 (MD&A, management's own discussion of results). These are where tone lives. Everything else is tables and boilerplate.
3. **Split into sentences**, keeping only ones of 6 to 120 words that are mostly letters, so table fragments and page furniture are dropped.
4. **Score every sentence** with the classifier and summarize: **net tone = (share positive − share negative)**, between −1 and +1. Keep the five most confident sentences in each direction as the **drivers**, so a reader can check what moved the number.

Then compare across peers. The snapshot in this repo scores AEHR against six test-equipment peers. Risk Factors are negative for everyone by design (they are a list of things that could go wrong), so the interesting comparisons are *relative*: AEHR's MD&A is the most negative of the group, and its risk factors are the mildest.

One lesson from building it: the first extractor opened Item 1A at the first mention of "Item 1A" anywhere in the text, including sentences like "see Item 1A, Risk Factors" in the business overview. Five of six peers had their risk-factor scores computed on the wrong text. Anchoring the heading match to the start of a line fixed it, and reversed the AEHR risk-factor conclusion. The number was always computed correctly; the *input* was wrong, and nothing in the metrics could have caught it. Read the drivers.

### A picture

```
  ticker "AEHR"
     │  company_tickers.json  →  CIK
     │  submissions API       →  latest 10-K accession number
     │  archive URL           →  HTML (cached under output/filing_tone/cache/)
     ▼
  html_to_text ─► extract_sections ─► { "Item 1A": "...", "Item 7": "..." }
                                             │
                                  split_sentences (6–120 words, mostly letters)
                                             │
                             258 risk sentences      145 MD&A sentences
                                             │
                                  classifier → (p_neg, p_neu, p_pos) each
                                             │
          net tone = share_pos − share_neg;  top-5 drivers each way
                                             ▼
             risk  −0.31          MD&A  −0.08           (AEHR, Oct 2026)

  Peers, MD&A:     FORM +0.10   ONTO +0.08   TER +0.04   COHU +0.01   KLIC −0.06   AEHR −0.08
                   ─────────────────────────────────────────────────────────────────────────►
                   more positive                                            more negative
```

### In the code

- `finlib/nlp/sec_text.py`: `html_to_text`, `extract_sections` (note `re.M` and the `^\s*item\s*1a` anchor), `split_sentences`, `filing_tone`, `TenKFetcher.latest_10k`.
- `projects/filing_tone/cli.py`: `build_snapshot` writes `tone_snapshot.json`; the page reads it.
- `app/pages/6_Filing_Tone.py`: the view. It imports no torch.

### Try it

Offline, against the synthetic fixture 10-K:

```bash
FINLIB_OFFLINE=1 python -m projects.filing_tone.cli snapshot --tickers AEHR
FINLIB_OFFLINE=1 python -m projects.filing_tone.cli summary
```

Then open `tests/fixtures/filing_tone/sample_10k.html` and predict which sentences will be the drivers before you look.

### Check yourself

**Q1.** Why is the risk-factor tone negative for every company, and what does that imply about how to read it?

<details><summary>Answer</summary>
Item 1A is a legally motivated list of everything that could go wrong, so most of its sentences are negative by construction. The absolute number means little. The useful readings are relative (AEHR versus peers) and over time (this year versus last). A company whose risk section is unusually mild, or suddenly harsher, is the signal.
</details>

**Q2.** The extractor bug produced plausible-looking numbers for five companies. Which part of the pipeline could have caught it, and which could not?

<details><summary>Answer</summary>
No metric could: the classifier was scoring real English sentences correctly, they were just the wrong sentences. What catches it is looking at the drivers (the top sentences) and noticing they were about business operations or financial statements, not risks, plus a test that feeds HTML containing an in-text "see Item 1A" and asserts the section starts at the real heading. Model quality and input quality are separate problems.
</details>

**Q3.** Net tone is share-positive minus share-negative. What information does that throw away, and when might it mislead?

<details><summary>Answer</summary>
It ignores confidence (a 51% negative counts the same as a 99% negative) and the neutral share (a section that is 90% neutral and one that is 10% neutral can have the same net tone). A section full of hedged, mildly negative sentences and one with a few blunt ones can score the same. The drivers and the sentence counts on the page are there to fill that gap.
</details>

---

## Final check: ten questions across the course

Try these without looking back. Each answer is one or two sentences.

1. What is a label, and who produced ours?
2. Name the three splits and the one purpose of each.
3. What do ids 0 and 1 mean in the vocabulary?
4. What is an attention mask for?
5. In the TextCNN, what does max-pooling discard, and why is that useful?
6. What is fine-tuning, and why does it need a small learning rate?
7. Describe one training step in four verbs.
8. Why does early stopping look at validation F1 and not training loss?
9. Why report macro-F1 instead of accuracy on this data?
10. Why does the project include a baseline that is almost as good as the CNN?

<details><summary>Answers</summary>

1. A human-decided correct class for one sentence. Finance-trained annotators of the Financial PhraseBank.
2. Train (learn from), validation (decide when to stop and which settings), test (report once, untouched).
3. 0 is padding, filler to make a batch rectangular. 1 is unknown, any token absent from the training vocabulary.
4. To tell the model which positions are real tokens and which are padding, so predictions do not depend on batch-mates.
5. Position. A phrase's presence matters for tone; where it sits in the sentence mostly does not.
6. Continuing to train a pretrained model on a new task. Large steps would overwrite the pretrained knowledge.
7. Forward (compute logits), loss (compare to labels), backward (compute gradients), step (move the weights).
8. Training loss keeps falling as the model memorizes; validation F1 measures what generalizes, which is the goal.
9. 60% of sentences are neutral, so accuracy rewards ignoring the minority classes. Macro-F1 weights all three equally.
10. To show what the neural network adds over the simplest method. Here the answer is "almost nothing", which is the honest finding, and it makes DistilBERT's gain meaningful.
</details>

---

## Glossary

| Term | One line |
| --- | --- |
| Classifier | A model that picks one of a fixed set of classes. |
| Label | The human-decided correct class for one example. |
| Train / validation / test | Learn from / tune on / report on once. |
| Stratified split | A split that preserves the class mix in every piece. |
| Token | A piece of text (word, number, symbol) the model treats as a unit. |
| Vocabulary | The token-to-integer mapping, built from training data only. |
| Padding, mask | Filler to make a batch rectangular, and the flags that mark it. |
| Embedding | A learned vector per token; similar words end up close. |
| Convolution (1-D) | A small filter slid along the sequence, firing on phrases. |
| Max-pooling | Keep each filter's strongest response, discard position. |
| Dropout | Randomly zero features in training so no one feature dominates. |
| Logits, softmax | Raw per-class scores, and the function turning them into probabilities. |
| Pretraining, fine-tuning | Learn general language on huge text; then adjust gently to a task. |
| Epoch | One full pass over the training data. |
| Cross-entropy | The loss: zero when all probability is on the right class. |
| Gradient | How the loss changes when each weight moves; computed by autograd. |
| Learning rate | Step size per update; warmed up then decayed here. |
| Class weights | Make mistakes on rare classes cost more. |
| Early stopping | Stop when validation stops improving; keep the best weights. |
| Overfitting | Fitting the training data's quirks; training improves, validation does not. |
| Confusion matrix | True class by predicted class, counts in each cell. |
| Precision, recall, F1 | False-alarm rate, miss rate, and their harmonic mean. |
| Macro-F1 | Average of per-class F1, every class equal. |
| Baseline | The simplest reasonable method, to show what the model adds. |
| Net tone | Share of positive sentences minus share of negative, per section. |
