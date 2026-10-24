# Probabilistic Graphical Models: exact inference and sequence learning

Supervised hidden Markov models for part-of-speech tagging, built from first
principles in NumPy: maximum-likelihood parameter estimation with add-k
smoothing, exact joint log-likelihood scoring, and Viterbi max-product
decoding, with development-set model selection and a tested baseline
comparison.

This is part of my assignments for CMPUT 651 (Probabilistic Graphical Models) at the
University of Alberta, and then rewrote it as a tested, installable package
so that anyone can run the experiments end to end and get the same numbers.

**Headline result on the Penn Treebank POS corpus (3,506 test sentences,
65,843 tokens):** my HMM with Viterbi decoding tags **93.67%** of words
correctly and gets **40.19%** of whole sentences exactly right, against
**90.09%** and **23.16%** for a most-frequent-tag baseline trained on the
same counts. Modelling tag-to-tag transitions is worth 3.6 points per word
and 17.0 points per sentence.

```bash
python -m pip install -e '.[plots]'
python -m unittest discover -s tests -v          # 12 tests, under a second

python -m pgm_portfolio.cli \
  --train data/coursework/train_preprocessed.csv \
  --dev   data/coursework/dev_preprocessed.csv \
  --test  data/coursework/test_preprocessed.csv \
  --objective token --seed 2024 --output results/my-viterbi
```

That run takes about 11 seconds and writes `metrics.json`, `validation.csv`
and a smoothing curve as `validation.png`.

## What I built, and what it demonstrates

| Area | What is implemented here |
|---|---|
| Directed graphical models | First-order HMM factorisation: initial-state, transition and emission distributions, estimated by counting aligned word/tag sequences |
| Parameter estimation | Categorical maximum likelihood with add-k smoothing, row normalisation, explicit handling of zero-count rows |
| Exact inference | Viterbi max-product dynamic programming in log space, `O(TK²)` time and `O(TK)` memory, with back-pointers and path reconstruction |
| Numerical method | Log-space arithmetic throughout; impossible events stay at `-inf` rather than being papered over with an epsilon |
| Model selection | Smoothing chosen on a development split, never on test; the selection metric is an explicit, recorded option |
| Evaluation | Three accuracy definitions kept separate (macro, micro, exact-match) plus a baseline that uses the same training counts |
| Verification | Viterbi checked against exhaustive enumeration of every tag sequence on short inputs |
| Software engineering | Installable package, command-line experiment runner, deterministic seeds, JSON and CSV outputs, 12 unit tests, no machine-specific paths |

The two experiments are written up separately, each with the question it
answers, the algorithm, and the result:

- **[01 — Joint probability and smoothing](projects/01-hmm-probability/README.md)**
  Can a learned HMM assign higher joint probability to a real annotation than
  to a random one, and how should the smoothing constant be chosen?
- **[02 — Viterbi decoding and accuracy](projects/02-hmm-viterbi/README.md)**
  Does exact sequence decoding beat tagging each word independently?

## Results on the Penn Treebank corpus

38,217 training, 3,251 development and 3,506 test sentences; 37,101
vocabulary types and 36 tags. The figures below come from running the code
in this repository, and the full machine-readable output is in
[`results/real-viterbi/metrics.json`](results/real-viterbi/metrics.json) and
[`results/real-joint/metrics.json`](results/real-joint/metrics.json).

| Quantity | Value |
|---|---:|
| Smoothing `k` selected on development data (both objectives) | 0.01 |
| Test mean joint log probability, real tags | −129.43555 |
| Test mean joint log probability, random tags (seed 2024) | −366.48646 |
| HMM word accuracy, macro over sentences | **0.93668** |
| HMM word accuracy, micro over tokens | 0.94142 |
| HMM whole-sentence accuracy | **0.40188** |
| Baseline word accuracy, macro over sentences | 0.90088 |
| Baseline word accuracy, micro over tokens | 0.90549 |
| Baseline whole-sentence accuracy | 0.23160 |

### Reading these numbers

**A fitted generative model separates real annotation from noise by a wide
margin.** Real tag sequences score −129.4 against −366.5 for uniformly
random tags on the same words. That gap is the model expressing that most
tag sequences are not plausible English, which is the whole point of putting
a transition distribution in the model.

**Sequence decoding helps far more at the sentence level than at the word
level.** The HMM gains 3.6 points of word accuracy over the baseline but
17.0 points of exact-sentence accuracy. The baseline already gets most
individual words right, because most word types have one dominant tag. What
it cannot do is make neighbouring decisions consistent with each other, and
a sentence is only correct when every tag in it is correct. That is exactly
the structure a transition matrix adds.

**Sentence accuracy is much lower than word accuracy for both methods**, and
that is arithmetic rather than a defect: at 94% per token, a twenty-token
sentence is fully correct only about 29% of the time if errors were
independent. Errors compound with length, which is why exact-match accuracy
is the honest metric for anything downstream that needs a whole parse.

### Two accuracy definitions, kept apart

Word accuracy can be averaged two ways, and they are not the same number:

- **macro**, the mean of each sentence's own token accuracy, which gives a
  one-word sentence the same weight as a forty-word one. This is the
  definition I used in the coursework and it is the headline figure above.
- **micro**, correct tokens divided by all tokens.

On this corpus they are 0.93668 and 0.94142 for the HMM. The code emits both
in every run, and `metrics.json` records which one drove model selection, so
the denominator is never ambiguous. Conflating them is an easy way to report
a number that cannot be reproduced.

## How to use this for your own work

The package is small and the pieces are independent, so it is usable as a
reference implementation rather than only as a finished result.

**The corpus format** is one sentence per line, words and tags separated by
a single tab, tokens separated by spaces:

```
PIERRE VINKEN 61 YEARS OLD WILL JOIN THE BOARD<TAB>NNP NNP CD NNS JJ MD VB DT NN
```

`pgm_portfolio/data.py` parses it strictly: a line with the wrong number of
tabs, or with tags that do not align with words, raises an error naming the
file and line number rather than silently producing a shorter sequence.

**Swapping in your own data** needs nothing but paths:

```bash
python -m pgm_portfolio.cli --train mine/train.tsv --dev mine/dev.tsv \
  --test mine/test.tsv --objective token --output results/mine
```

**The two objectives** differ in how `k` is chosen. `--objective joint`
selects the `k` that maximises mean joint log probability on the development
set, which is the right criterion when you care about the model as a density
over sequences. `--objective token` selects the `k` that maximises
development tagging accuracy, which is the right criterion when you care
about decoding. They need not agree, and the distinction between a
likelihood objective and a decision objective is worth being deliberate
about; on this corpus both happen to choose 0.01.

**Points worth copying if you implement this yourself:**

- Build the vocabulary from the training split only, and map unseen
  evaluation words to an explicit `<UNK>` index. Building it from all three
  splits leaks information and inflates the result.
- Keep `log 0 = -inf` instead of `log(p + 1e-10)`. The epsilon turns an
  impossible sequence into a merely unlikely one, which quietly changes what
  the smoothing sweep is measuring; with a true `-inf` you can see directly
  that `k = 0` leaves some test sequences undecodable.
- Cache the log-parameters once per model instead of recomputing the
  emission matrix per sentence. On a 38k-sentence corpus that is the
  difference between seconds and minutes.
- Test Viterbi against brute force. On sequences of three tokens and two
  tags there are only eight tag sequences, so you can enumerate all of them
  and assert that the dynamic program found the best one. That test is in
  `tests/test_hmm.py` and it is the only way to be sure the back-pointers
  are right.

## Repository map

```
pgm_portfolio/hmm.py    HMM estimation, joint score, Viterbi, metrics, baseline
pgm_portfolio/data.py   corpus parsing, train-only vocabulary, OOV encoding
pgm_portfolio/cli.py    experiment runner: selection, evaluation, outputs
projects/               the two experiments, written up with their results
data/coursework/        the Penn Treebank POS splits used above
data/examples/          a small synthetic corpus for smoke-testing the CLI
results/                executed output: metrics.json, validation.csv, PNG
tests/                  12 unit tests including the brute-force check
legacy/                 my original coursework scripts, kept for provenance
docs/IMPLEMENTATION_NOTES.md   design decisions and verification
```

The synthetic corpus in `data/examples/` exists so the CLI can be exercised
without the full treebank. Both methods reach 100% on it, which says only
that the code runs: the toy data is too easy to separate the algorithms, and
no conclusion about tagging quality should be drawn from it.

`legacy/` holds the scripts as I originally wrote them, including the
hard-coded paths they ran with. They are kept so the provenance of the work
is visible; the package in `pgm_portfolio/` is the version to read and run.

## Requirements

Python 3.9+ and NumPy. Matplotlib is optional and only used to draw the
smoothing curve.

## Scope

The CMPUT 651 course also covered Bayesian networks, Markov random fields,
conditional random fields, variable elimination, factor-graph message
passing and Bayesian parameter estimation. What is implemented here is the
supervised HMM and exact max-product inference. The results are measured on
one POS corpus and are not a claim about other domains or languages.
