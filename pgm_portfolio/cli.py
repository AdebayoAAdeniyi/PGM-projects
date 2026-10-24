"""Run development-set smoothing selection and held-out HMM evaluation.

The smoothing constant k is chosen on the development split and only then is
the selected model scored on test, so no test label ever influences a
modelling decision.  Which development metric drives that choice is an
explicit option rather than an implicit one, because the macro and micro
accuracies have different denominators and can in principle disagree.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .hmm import fit_supervised, corpus_log_probability, tag_accuracy, MostFrequentTagBaseline
from .data import load_tsv, encode

K_VALUES = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True)
    parser.add_argument('--dev', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--objective', choices=['joint', 'token'], default='token',
                        help="'joint' selects k by mean joint log probability, "
                             "'token' selects k by development tagging accuracy")
    parser.add_argument('--select-metric', choices=['macro', 'micro'], default='macro',
                        dest='select_metric',
                        help="for --objective token, the development accuracy used to "
                             "choose k: 'macro' is the mean of per-sentence token "
                             "accuracies, 'micro' is correct tokens over all tokens "
                             "(default: macro)")
    parser.add_argument('--output', default='results/example')
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args(argv)
    raw = [load_tsv(p) for p in (args.train, args.dev, args.test)]
    (train, dev, test), words, tags = encode(*raw)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    accuracy_key = 'mean_sentence_token_accuracy' if args.select_metric == 'macro' else 'token_accuracy'
    metric_name = ('mean joint log probability' if args.objective == 'joint'
                   else f'{args.select_metric} tagging accuracy ({accuracy_key})')
    # k=0 leaves genuinely unseen events at probability zero, so the joint
    # score of any corpus containing one is -inf.  It is skipped for the
    # joint objective and kept for the tagging objective, where it is
    # informative that decoding fails outright.
    trials = []
    for k in (K_VALUES[1:] if args.objective == 'joint' else K_VALUES):
        model = fit_supervised(train, len(tags), len(words), k)
        try:
            score = (corpus_log_probability(model, dev) if args.objective == 'joint'
                     else tag_accuracy(model, dev)[accuracy_key])
        except ValueError:  # no decodable path at k=0
            score = float('-inf')
        trials.append({'k': k, 'metric': metric_name, 'validation_score': score})
    # `max` keeps the first maximum and K_VALUES is ascending, so ties go to
    # the smallest k: the least smoothing that achieves the best score.
    best = max(trials, key=lambda x: x['validation_score'])
    model = fit_supervised(train, len(tags), len(words), best['k'])
    rng = np.random.default_rng(args.seed)
    randomized = [(w, rng.integers(len(tags), size=len(t)).tolist()) for w, t in test]
    report = {
        'dataset': {'training_sentences': len(train), 'development_sentences': len(dev), 'test_sentences': len(test), 'vocabulary_size': len(words), 'tag_count': len(tags)},
        'objective': args.objective,
        'selection_metric': metric_name,
        'validation_trials': trials,
        'selected_k': best['k'],
        'test_mean_joint_log_probability': corpus_log_probability(model, test),
        'random_tag_mean_joint_log_probability': corpus_log_probability(model, randomized),
        'test_hmm': tag_accuracy(model, test),
        'test_frequency_baseline': MostFrequentTagBaseline(train, len(tags)).evaluate(test),
        'random_seed': args.seed,
        'provenance': ('Computed on the named input corpus. The vocabulary is rebuilt '
                       'from the training split alone, so unseen evaluation words map '
                       'to <UNK> and no test information reaches the model.')
    }
    # JSON requires finite numeric values. Impossible sequences have log P=-inf.
    # Preserve that distinction as a textual marker instead of writing invalid JSON.
    def json_safe(value):
        if isinstance(value, float) and not np.isfinite(value):
            return '-Infinity' if value < 0 else 'Infinity'
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value
    report = json_safe(report)
    (out / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with (out / 'validation.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, ['k', 'metric', 'validation_score'])
        writer.writeheader()
        writer.writerows(trials)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        x = [t['k'] for t in trials]
        y = [t['validation_score'] for t in trials]
        fig, ax = plt.subplots(figsize=(6, 4))
        finite = [i for i, v in enumerate(y) if np.isfinite(v)]
        ax.plot(finite, [y[i] for i in finite], marker='o')
        # A -inf score cannot be drawn, and leaving a silent gap in the
        # curve invites the reader to assume the point is simply missing.
        # Label it instead: at k=0 some sequences have no decodable path.
        for i, v in enumerate(y):
            if not np.isfinite(v):
                ax.annotate('no decodable\npath', xy=(i, min(y[j] for j in finite)),
                            ha='center', va='bottom', fontsize=8, color='#b00')
        ax.set_xticks(range(len(x)), [str(v) for v in x], rotation=35)
        ax.set_xlabel('Add-k smoothing')
        ax.set_ylabel('Development-set mean joint log P' if args.objective == 'joint'
                      else f'Development-set {args.select_metric} tagging accuracy')
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(out / 'validation.png', dpi=150)
        plt.close(fig)
    except ImportError:
        pass
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    main()
