# 01 — Joint probability and smoothing in a supervised HMM

**Question.** Does a hidden Markov model fitted by counting assign higher
joint probability to a real word/tag annotation than to a random one, and
how should the smoothing constant be chosen?

## The model

A supervised HMM over observed words `x_t` and hidden POS tags `y_t` factors
into three categorical distributions: the initial-state vector `π`, the
transition matrix `A` and the emission matrix `B`.

```
P(x_1:T, y_1:T) = π[y_1] · B[y_1, x_1] · Π_{t=2..T} A[y_{t-1}, y_t] · B[y_t, x_t]
```

Because the tags are observed during training, the maximum-likelihood
estimate is just normalised counts, and no EM is required. What the model
has to learn is which tags start sentences, which tags follow which, and
which words each tag emits.

This is the **joint** probability of the words together with a given tag
sequence. It is not `P(words)`, which would require summing over all `K^T`
tag sequences, and it is not `P(tags | words)`, which is what chapter 02
decodes. Keeping those three apart is the main conceptual point of this
experiment.

## Implementation

1. Parse each training line into an aligned word sequence and tag sequence,
   validating that the lengths match.
2. Initialise the initial, transition and emission count arrays at `k`, the
   smoothing constant, rather than at zero.
3. Count the first tag of each sentence, every adjacent tag pair, and every
   tag-word pair including the final word.
4. Row-normalise. A row that is still entirely zero, which can only happen
   at `k = 0` for a tag never observed, falls back to uniform; that is a
   deliberate choice and not the maximum-likelihood estimate, so it is
   marked as such in the code.
5. Sweep `k ∈ {10⁻⁵, 10⁻⁴, 10⁻³, 10⁻², 10⁻¹, 1}` and pick the value with
   the highest mean per-sentence joint log probability on the development
   split. `k = 0` is excluded from this objective because a single unseen
   event sends the whole corpus score to `-inf`.
6. Score the held-out test split once, together with a seeded random-tag
   control over the same words.

Code: [`pgm_portfolio/hmm.py`](../../pgm_portfolio/hmm.py)
(`fit_supervised`, `joint_log_probability`, `corpus_log_probability`) and
[`pgm_portfolio/cli.py`](../../pgm_portfolio/cli.py) with
`--objective joint`.

## Results

On the Penn Treebank splits, from
[`results/real-joint/metrics.json`](../../results/real-joint/metrics.json):

| Quantity | Value |
|---|---:|
| Smoothing `k` selected on development data | 0.01 |
| Test mean joint log probability, real tags | **−129.43555** |
| Test mean joint log probability, random tags (seed 2024) | **−366.48646** |

A less negative log probability means higher probability. The real
annotation scores roughly 237 log units per sentence above a random one,
which is an enormous ratio, and it is the model saying that the tag
sequences humans actually produce are not interchangeable with arbitrary
ones. A model with no transition structure could not express that.

On a short worked example, `I LIKE THE BOOK` scores **−24.37** under
`PRP IN DT NN` and **−34.50** under `PRP IN DT VB`, so the grammatical
reading wins by about ten log units on four tokens.

### What the smoothing sweep shows

Development scores are close to each other around the optimum, so the
selected `k = 0.01` is a shallow maximum rather than a sharp one. That is
worth stating plainly: a plot of this curve is not precise enough to read a
best value off by eye, which is why the selection is done numerically and
every trial is written to `validation.csv`.

The choice is also **objective-dependent**. The `k` that maximises
likelihood is answering a different question from the `k` that maximises
tagging accuracy, and there is no reason in general for them to coincide.
They happen to agree on this corpus, which is a fact about this corpus.

### A caution about comparing log scores

These are per-sentence sums, so they grow more negative with sentence
length. Comparing two corpora with different length distributions by mean
joint log probability would mostly measure the length difference. The
comparison here is valid because the random-tag control uses exactly the
same sentences, changing only the tags.

## Reproduce

```bash
python -m pgm_portfolio.cli \
  --train data/coursework/train_preprocessed.csv \
  --dev   data/coursework/dev_preprocessed.csv \
  --test  data/coursework/test_preprocessed.csv \
  --objective joint --seed 2024 --output results/my-joint
```

Substitute `data/examples/{train,dev,test}.tsv` for a fast run on the
synthetic corpus. The development split fixes `k` before the test split is
touched.

## Techniques demonstrated

Generative modelling and the factorisation of a directed graphical model;
categorical maximum likelihood with add-k smoothing; log-space numerical
computation; the distinction between joint, marginal and conditional
quantities; validation-driven hyperparameter selection with a held-out test
set; and explicit handling of zero-probability events instead of an epsilon
that hides them.
