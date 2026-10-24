import itertools
import math
import unittest
import numpy as np
from pgm_portfolio.hmm import (fit_supervised, corpus_log_probability,
                               tag_accuracy, MostFrequentTagBaseline)
from pgm_portfolio.data import encode


class HMMTests(unittest.TestCase):
    def setUp(self):
        self.samples = [([0, 1, 2], [0, 1, 0]), ([0, 2], [0, 0]),
                        ([1, 2], [1, 0]), ([1, 1], [1, 1])]
        self.model = fit_supervised(self.samples, 2, 3, .1)

    def test_normalization(self):
        self.assertAlmostEqual(self.model.initial.sum(), 1)
        np.testing.assert_allclose(self.model.transition.sum(axis=1), 1)
        np.testing.assert_allclose(self.model.emission.sum(axis=1), 1)

    def test_exact_counting(self):
        model = fit_supervised([([0, 1], [0, 1])], 2, 2, 0)
        self.assertEqual(model.initial.tolist(), [1, 0])
        self.assertEqual(model.transition[0, 1], 1)
        self.assertEqual(model.emission[0, 0], 1)
        self.assertEqual(model.emission[1, 1], 1)

    def test_joint_factorization(self):
        w, t = [0, 1, 2], [0, 1, 0]
        value = self.model.initial[0] * self.model.emission[0, 0] * self.model.transition[0, 1] * self.model.emission[1, 1] * self.model.transition[1, 0] * self.model.emission[0, 2]
        self.assertAlmostEqual(self.model.joint_log_probability(w, t), math.log(value))

    def test_viterbi_against_brute_force(self):
        for words in ([0], [1, 2], [2, 1, 0]):
            scores = [(self.model.joint_log_probability(words, tags), tags)
                      for tags in itertools.product(range(2), repeat=len(words))]
            best_score = max(x[0] for x in scores)
            tags, score = self.model.viterbi(words)
            self.assertAlmostEqual(score, best_score)
            self.assertAlmostEqual(self.model.joint_log_probability(words, tags), best_score)

    def test_impossible_joint_stays_negative_infinity(self):
        model = fit_supervised([([0], [0])], 2, 2, 0)
        self.assertEqual(model.joint_log_probability([1], [0]), -math.inf)

    def test_no_possible_viterbi_sequence(self):
        model = fit_supervised([([0], [0])], 2, 2, 0)
        with self.assertRaisesRegex(ValueError, 'No possible'):
            model.viterbi([1])

    def test_input_validation(self):
        for words, tags in [([], []), ([0], []), ([9], [0]), ([0], [8])]:
            with self.assertRaises(ValueError):
                self.model.joint_log_probability(words, tags)
        with self.assertRaises(ValueError):
            fit_supervised([], 2, 3)
        with self.assertRaises(ValueError):
            fit_supervised(self.samples, 2, 3, -.1)

    def test_baseline(self):
        baseline = MostFrequentTagBaseline(self.samples, 2)
        self.assertEqual(len(baseline.predict([0, 12])), 2)
        outcome = baseline.evaluate(self.samples)
        self.assertTrue(0 <= outcome['token_accuracy'] <= 1)

    def test_token_micro_accuracy_not_sentence_macro(self):
        class Fake:
            def viterbi(self, w):
                return ([0] * len(w), 0)
        m = Fake()
        # long sentence 4/4 correct, short sentence 0/1 correct => .8
        result = tag_accuracy(m, [([1] * 4, [0] * 4), ([1], [1])])
        self.assertAlmostEqual(result['token_accuracy'], .8)
        self.assertEqual(result['sentence_accuracy'], .5)

    def test_train_only_vocabulary_unknown(self):
        (_, dev, test), words, tags = encode([(['seen'], ['N'])], [(['unseen'], ['N'])], [(['novel'], ['N'])])
        self.assertEqual(dev[0][0], [0])
        self.assertEqual(test[0][0], [0])
        self.assertNotIn('unseen', words)

    def test_train_dataset_corpus_score(self):
        manual = np.mean([self.model.joint_log_probability(*example) for example in self.samples])
        self.assertAlmostEqual(corpus_log_probability(self.model, self.samples), manual)

    def test_unseen_tag_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            encode([(['one'], ['N'])], [(['one'], ['V'])], [(['one'], ['N'])])


if __name__ == '__main__':
    unittest.main()
