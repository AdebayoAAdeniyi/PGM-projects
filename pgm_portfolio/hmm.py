"""Discrete supervised Hidden Markov Model and exact log-space inference.

No machine-learning framework required; NumPy is used for numerical arrays.
"""
from dataclasses import dataclass
from functools import cached_property
import numpy as np


def _log_prob(p):
    """Preserve impossible events as -inf, instead of adding artificial epsilon."""
    with np.errstate(divide='ignore'):
        return np.log(p)


@dataclass(frozen=True)
class HiddenMarkovModel:
    initial: np.ndarray
    transition: np.ndarray
    emission: np.ndarray

    def __post_init__(self):
        pi, a, b = (np.asarray(x, dtype=float) for x in (self.initial, self.transition, self.emission))
        n = pi.size
        if pi.ndim != 1 or a.shape != (n, n) or b.ndim != 2 or b.shape[0] != n:
            raise ValueError('Inconsistent probability dimensions')
        for p in (pi, a, b):
            if not np.all(np.isfinite(p)) or np.any(p < 0):
                raise ValueError('Probabilities must be finite and nonnegative')
        if not np.isclose(pi.sum(), 1) or not np.allclose(a.sum(axis=1), 1) or not np.allclose(b.sum(axis=1), 1):
            raise ValueError('Probability rows must sum to one')

    @property
    def num_tags(self):
        return len(self.initial)

    @property
    def num_words(self):
        return self.emission.shape[1]

    def _check_words(self, words):
        w = np.asarray(words, dtype=int)
        if w.ndim != 1 or w.size == 0 or (w < 0).any() or (w >= self.num_words).any():
            raise ValueError('words must be nonempty valid vocabulary indices')
        return w

    def joint_log_probability(self, words, tags):
        """log P(words, tags) for a supervised pair; this is NOT log P(words).

        The marginal P(words) would require summing over all tag sequences.
        What is scored here is the joint probability of the words together
        with the tags that were handed in, which is what the smoothing
        experiment compares between a real annotation and a random one.

        Evaluated as a sum of logs over the factorisation
        pi[y_1] B[y_1, x_1] prod_t A[y_{t-1}, y_t] B[y_t, x_t].
        An impossible event contributes -inf and the whole sequence becomes
        -inf, which is the honest answer; adding an epsilon inside the log
        would silently turn a zero-probability sequence into a finite score.
        """
        w = self._check_words(words)
        t = np.asarray(tags, dtype=int)
        if t.ndim != 1 or t.size != w.size or (t < 0).any() or (t >= self.num_tags).any():
            raise ValueError('tags must match words and contain valid indices')
        log_pi, log_a, log_b = self.log_parameters
        total = log_pi[t[0]] + log_b[t, w].sum()
        if t.size > 1:
            total += log_a[t[:-1], t[1:]].sum()
        return float(total)

    @cached_property
    def log_parameters(self):
        return tuple(_log_prob(p) for p in (self.initial, self.transition, self.emission))

    def viterbi(self, words):
        """Max-product dynamic programming with back-pointers; O(T*K^2)."""
        w = self._check_words(words)
        k, length = self.num_tags, len(w)
        log_pi, log_a, log_b = self.log_parameters
        scores = np.full((length, k), -np.inf)
        back = np.zeros((length, k), dtype=int)
        scores[0] = log_pi + log_b[:, w[0]]
        for i in range(1, length):
            possibilities = scores[i - 1, :, None] + log_a
            back[i] = np.argmax(possibilities, axis=0)
            scores[i] = possibilities[back[i], np.arange(k)] + log_b[:, w[i]]
        end = int(np.argmax(scores[-1]))
        best_score = float(scores[-1, end])
        if not np.isfinite(best_score):
            raise ValueError('No possible tag sequence for these observations; use smoothing or handle OOV')
        path = [end]
        for i in range(length - 1, 0, -1):
            path.append(int(back[i, path[-1]]))
        path.reverse()
        return path, best_score


def fit_supervised(samples, num_tags, num_words, smoothing=0.01):
    """Fit pi, transitions, emissions by normalized counts and add-k smoothing."""
    if num_tags < 1 or num_words < 1 or not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError('Invalid HMM dimensions or smoothing parameter')
    initial = np.full(num_tags, float(smoothing))
    trans = np.full((num_tags, num_tags), float(smoothing))
    emit = np.full((num_tags, num_words), float(smoothing))
    seen = 0
    for words, tags in samples:
        w, t = np.asarray(words, dtype=int), np.asarray(tags, dtype=int)
        if w.ndim != 1 or t.ndim != 1 or w.size == 0 or w.size != t.size:
            raise ValueError('Every sequence must be nonempty and aligned')
        if (w < 0).any() or (w >= num_words).any() or (t < 0).any() or (t >= num_tags).any():
            raise ValueError('Input IDs outside vocabulary/tag range')
        seen += 1
        initial[t[0]] += 1
        np.add.at(emit, (t, w), 1)
        if len(t) > 1:
            np.add.at(trans, (t[:-1], t[1:]), 1)
    if not seen:
        raise ValueError('Training corpus is empty')
    for counts in (trans, emit):
        empty = counts.sum(axis=1) == 0
        # k=0 and unseen state => undefined MLE row; use uniform fallback,
        # explicitly documented instead of NaN/zero-division.
        counts[empty] = 1.0
    return HiddenMarkovModel(initial / initial.sum(), trans / trans.sum(axis=1, keepdims=True), emit / emit.sum(axis=1, keepdims=True))


def corpus_log_probability(model, samples):
    scores = [model.joint_log_probability(w, t) for w, t in samples]
    if not scores:
        raise ValueError('Evaluation corpus is empty')
    return float(np.mean(scores))


def tag_accuracy(model, samples):
    """Decode every sentence with Viterbi and report three distinct accuracies.

    `token_accuracy`                micro: correct tokens / all tokens.
    `mean_sentence_token_accuracy`  macro: the mean over sentences of each
                                    sentence's own token accuracy.
    `sentence_accuracy`             exact match: a sentence counts only if
                                    every tag in it is right.

    The first two differ whenever sentence lengths differ, because the macro
    version gives a one-word sentence the same weight as a forty-word one.
    Both are reported so that the denominator is never ambiguous.
    """
    correct_tokens = total_tokens = correct_sentences = total_sentences = 0
    sentence_token_accuracies = []
    for words, tags in samples:
        pred, _ = model.viterbi(words)
        true = list(tags)
        n_correct = sum(a == b for a, b in zip(pred, true))
        correct_tokens += n_correct
        sentence_token_accuracies.append(n_correct / len(true))
        total_tokens += len(true)
        correct_sentences += int(pred == true)
        total_sentences += 1
    if not total_sentences:
        raise ValueError('Evaluation corpus is empty')
    return {'token_accuracy': correct_tokens / total_tokens,
            'mean_sentence_token_accuracy': float(np.mean(sentence_token_accuracies)),
            'sentence_accuracy': correct_sentences / total_sentences,
            'sentences': total_sentences, 'tokens': total_tokens}


class MostFrequentTagBaseline:
    """Per-word most frequent training tag, with a global fallback for OOV.

    The point of comparison for Viterbi: it uses exactly the same supervised
    counts but makes each decision independently, so the difference between
    the two isolates the contribution of modelling tag-to-tag transitions.
    """

    def __init__(self, samples, num_tags):
        counts = np.zeros(num_tags, dtype=int)
        per_word = {}
        for words, tags in samples:
            for word, tag in zip(words, tags):
                counts[tag] += 1
                if word not in per_word:
                    per_word[word] = np.zeros(num_tags, dtype=int)
                per_word[word][tag] += 1
        if counts.sum() == 0:
            raise ValueError('No training tokens')
        self.fallback = int(counts.argmax())
        self.per_word = {word: int(c.argmax()) for word, c in per_word.items()}

    def predict(self, words):
        return [self.per_word.get(w, self.fallback) for w in words]

    def evaluate(self, samples):
        tokens = correct = sentences = exact = 0
        sentence_token_accuracies = []
        for w, t in samples:
            p = self.predict(w)
            tokens += len(t)
            n_correct = sum(a == b for a, b in zip(p, t))
            correct += n_correct
            sentence_token_accuracies.append(n_correct / len(t))
            sentences += 1
            exact += int(p == list(t))
        if not sentences:
            raise ValueError('Evaluation corpus is empty')
        return {'token_accuracy': correct / tokens,
                'mean_sentence_token_accuracy': float(np.mean(sentence_token_accuracies)),
                'sentence_accuracy': exact / sentences,
                'sentences': sentences, 'tokens': tokens}
