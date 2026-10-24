# 02 — Viterbi decoding against independent word tagging

**Question.** Does exact sequence decoding recover better POS tags than
choosing each word's most likely tag independently, and where does the
benefit actually appear?

## The algorithm

The goal is the most probable whole tag sequence given the sentence:

```
ŷ_1:T = argmax_y P(y_1:T | x_1:T) = argmax_y P(x_1:T, y_1:T)
```

The conditional denominator `P(x)` is the same for every candidate, so
maximising the joint is equivalent and avoids computing it. Enumerating all
candidates costs `O(K^T)`, which for 36 tags and a twenty-word sentence is
about 10³¹ sequences.

Viterbi, the max-product dynamic program, does it in `O(TK²)` time and
`O(TK)` memory by exploiting the Markov property: the best path ending in
tag `j` at time `t` extends exactly one of the best paths ending at time
`t-1`.

```
δ_1(j) = log π_j + log B[j, x_1]
δ_t(j) = max_i { δ_{t-1}(i) + log A[i, j] } + log B[j, x_t]
```

Storing `argmax_i` at each cell and backtracking from the best final state
reconstructs the sequence. Everything is in log space, so the inner
operation is addition rather than multiplication and nothing underflows on
long sentences. `log 0` is exactly `-inf`, and if no path has positive
probability the decoder raises an informative error rather than returning an
arbitrary path.

Code: [`pgm_portfolio/hmm.py`](../../pgm_portfolio/hmm.py) (`viterbi`,
`tag_accuracy`, `MostFrequentTagBaseline`) and
[`pgm_portfolio/cli.py`](../../pgm_portfolio/cli.py) with
`--objective token`.

## Experiment

1. Fit the supervised HMM from aligned POS training data; unseen evaluation
   words map to a training-defined `<UNK>`.
2. Sweep `k ∈ {0, 10⁻⁵, 10⁻⁴, 10⁻³, 10⁻², 10⁻¹, 1}`. Unlike the likelihood
   objective, `k = 0` is kept here because its failure is informative: some
   test sentences become undecodable and the sweep records that explicitly.
3. Decode the development split and choose `k` by macro word accuracy, the
   mean of each sentence's own token accuracy. The selected metric is
   recorded in `metrics.json`; `--select-metric micro` switches it.
4. Decode the untouched test split and report macro word accuracy, micro
   token accuracy and exact-sentence accuracy.
5. Compare against a most-frequent-training-tag baseline, with the globally
   most frequent tag as the fallback for unseen words.
6. Verify the dynamic program against exhaustive enumeration on short
   sequences in the unit tests.

The baseline matters because it uses **the same supervised counts** as the
HMM and differs only in making each decision independently. The difference
between the two therefore isolates the contribution of the transition
matrix, rather than measuring the HMM against an unrelated method.

## Results

| Method | Word accuracy (macro) | Word accuracy (micro) | Sentence accuracy |
|---|---:|---:|---:|
| HMM + Viterbi | **93.668%** | 94.142% | **40.188%** |
| Most-frequent-tag baseline | 90.088% | 90.549% | 23.160% |
| Improvement | **+3.580 pts** | +3.593 pts | **+17.028 pts** |

Full output in
[`results/real-viterbi/metrics.json`](../../results/real-viterbi/metrics.json).

### Where the benefit comes from

The gain is five times larger at the sentence level than at the word level,
and that ratio is the real finding.

Most word types in English have one dominant POS tag, so a lookup table that
ignores context already gets 90% of individual tokens right. What it cannot
do is keep neighbouring decisions consistent. The transition matrix is what
rules out tag sequences that are locally plausible but jointly impossible,
and consistency is precisely what exact-match accuracy measures. A method
that improves each word a little, independently, would not move sentence
accuracy by 17 points.

Sentence accuracy is far below word accuracy for both methods, and that is
arithmetic: a sentence is correct only if every tag is correct, so the error
rate compounds with length. At 94% per token, independent errors would make
a twenty-token sentence fully correct about 29% of the time. The measured
40% is better than that, which is itself evidence that the remaining errors
are concentrated in particular sentences rather than scattered uniformly.

### Choosing k

Development accuracy is 0.9397, 0.9397, 0.9396, **0.9398**, 0.9371, 0.9165
for `k` from 10⁻⁵ to 1, with `k = 0` undecodable. The maximum at `k = 0.01`
is shallow, and both the macro and micro criteria select it. Ties go to the
smallest `k`, so the reported choice is the least smoothing that achieves
the best score.

## Reproduce

```bash
python -m pgm_portfolio.cli \
  --train data/coursework/train_preprocessed.csv \
  --dev   data/coursework/dev_preprocessed.csv \
  --test  data/coursework/test_preprocessed.csv \
  --objective token --seed 2024 --output results/my-viterbi
```

About 11 seconds. Use `data/examples/*.tsv` for a fast smoke test; both
methods score 100% there, which shows only that the pipeline runs.

## Techniques demonstrated

Exact inference by dynamic programming; vectorised `K×K` score updates;
back-pointer reconstruction of an optimal path; log-space numerical
stability; correctness verification against brute-force enumeration;
careful separation of three accuracy definitions; a baseline chosen to
isolate one modelling choice; and objective-specific hyperparameter
selection on held-out data.
