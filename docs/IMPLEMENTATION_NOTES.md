# Implementation notes

Design decisions behind the package in `pgm_portfolio/`, and how each one is
verified. Written for a reader deciding whether to trust the numbers, or
building something similar and wanting the reasoning rather than just the
code.

---

## 1. Impossible events stay at negative infinity

Probabilities are converted to logs with `np.log` and nothing else. An event
the training data never saw has probability zero at `k = 0`, so its log is
`-inf`, and any sequence containing it scores `-inf`.

The tempting alternative is `log(p + 1e-10)`, which keeps every score finite
and avoids warnings. It also silently changes the experiment: an impossible
sequence becomes merely a very unlikely one, the smoothing sweep stops being
able to distinguish "this model cannot explain the data at all" from "this
model explains it badly", and the epsilon becomes an undocumented smoothing
parameter of its own.

Keeping the `-inf` makes the `k = 0` row of the tagging sweep genuinely
informative: it reports that some test sentences have no decodable path,
which is the actual consequence of refusing to smooth.

`test_impossible_joint_stays_negative_infinity` and
`test_no_possible_viterbi_sequence` cover both paths.

JSON cannot represent infinity, so `metrics.json` writes the string
`"-Infinity"` and the writer runs with `allow_nan=False`. That keeps the
file valid JSON while preserving the distinction, rather than substituting a
plausible-looking finite number.

## 2. Vocabulary is built from the training split only

`data.encode` collects word types from training data alone and assigns index
0 to `<UNK>`. Development and test words that were never seen map to that
index.

Building the vocabulary from all three splits is a quiet form of leakage:
the model gets a dedicated emission parameter for a word it could not have
known about, and the held-out score stops being held out. Gold **tags** are
treated more strictly still: an evaluation tag outside the training tag set
raises rather than being folded into an unknown bucket, because that
indicates a corpus problem rather than a modelling situation.

`test_train_only_vocabulary_unknown` and
`test_unseen_tag_not_silently_accepted` cover this.

## 3. Three accuracy definitions, never conflated

- `token_accuracy` (micro): correct tokens over all tokens.
- `mean_sentence_token_accuracy` (macro): the mean over sentences of each
  sentence's own token accuracy.
- `sentence_accuracy`: fraction of sentences where every tag is correct.

Macro and micro differ whenever sentence lengths differ, because macro gives
a one-word sentence the same weight as a forty-word one. On the Penn
Treebank test split the HMM scores 0.93668 macro and 0.94142 micro: close,
but not the same number, and quoting one while computing the other is an
easy way to publish a figure nobody can reproduce.

All three are emitted on every run, and `metrics.json` records which one was
used for model selection via the `selection_metric` field.
`test_token_micro_accuracy_not_sentence_macro` constructs a case where macro
and micro deliberately disagree.

## 4. Model selection never sees the test labels

The smoothing constant is chosen on the development split; the selected
model is then evaluated once on test. `--select-metric` makes the choice of
development criterion explicit rather than implicit, because the macro and
micro criteria could in principle disagree. On this corpus both select
`k = 0.01`, and so does the likelihood objective.

Ties break towards the smallest `k`: `K_VALUES` is ascending and `max`
returns the first maximum, so the reported choice is the least smoothing
that achieves the best score.

## 5. Viterbi is checked against brute force

On three tokens with two tags there are only eight possible tag sequences,
so the test enumerates all of them, scores each with
`joint_log_probability`, and asserts that the dynamic program returns both
the best score and a path achieving it. That is the only check that actually
exercises the back-pointers; comparing against a second implementation of
the same idea would reproduce the same mistake.

`test_viterbi_against_brute_force`.

## 6. Log-parameters are computed once per model

`log_parameters` is a `cached_property`, so the log of the emission matrix is
taken once rather than once per sentence. With 38,217 training sentences and
a 37,101-word vocabulary, recomputing a 36 × 37,101 log per sentence
dominates everything else. The full sweep now runs in about 11 seconds.

## 7. Zero-count rows at k = 0

A tag that never appears as a source of transitions, or never emits
anything, leaves an all-zero count row that cannot be normalised. Those rows
fall back to uniform. This is a defined behaviour rather than the
maximum-likelihood estimate, which is undefined there, and it is marked as
such at the point it happens so nobody mistakes it for an estimate.

## 8. Reproducibility

Every run takes an explicit `--seed`, used for the random-tag control, and
records it in `metrics.json`. The deterministic parts of the pipeline are
seed-independent and reproduce exactly: `-129.43554751738608` for the test
joint score and `0.9366800859818618` for macro word accuracy, on repeated
runs.

Nothing depends on a machine-specific path. The corpora live under `data/`
and are passed as command-line arguments.

## 9. On `vocab.ezx`

The original pipeline loaded a serialised vocabulary from an HDF5 file.
It is kept in `data/coursework/` for provenance but is not loaded by this
package: it stores pickled Python objects, and unpickling an opaque binary
is not something a reader should have to trust in order to run the
experiments. The vocabulary is rebuilt from the training split instead, and
the reported figures are reproduced that way.

## 10. Scope of the results

The measurements are on one POS corpus, with one tag set, in one language.
They show that modelling tag transitions beats independent per-word tagging
on this data, by a margin that is much larger at the sentence level than at
the word level. They are not a claim about other domains, and the synthetic
corpus in `data/examples/` supports no claim at all: both methods score 100%
on it because it is too easy to separate them, and it exists only to
exercise the command-line interface quickly.
