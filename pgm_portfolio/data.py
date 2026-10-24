"""Portable TSV dataset reader: WORD1 WORD2...<tab>TAG1 TAG2..."""
from pathlib import Path


def load_tsv(path):
    corpus = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split('\t')
        if len(fields) != 2:
            raise ValueError(f'{path}:{number}: expected one tab between words and tags')
        words, tags = fields[0].split(), fields[1].split()
        if not words or len(words) != len(tags):
            raise ValueError(f'{path}:{number}: empty or unaligned sequence')
        corpus.append((words, tags))
    if not corpus:
        raise ValueError(f'Empty corpus: {path}')
    return corpus


def encode(train, dev, test):
    """Fit vocabulary on TRAIN ONLY; reserve UNK for unseen validation/test words."""
    word_set = sorted({word for sentence, _ in train for word in sentence} - {'<UNK>'})
    tag_set = sorted({tag for _, tags in train for tag in tags})
    words = {'<UNK>': 0, **{w: i + 1 for i, w in enumerate(word_set) if w != '<UNK>'}}
    tags = {tag: i for i, tag in enumerate(tag_set)}
    def convert(corpus):
        out = []
        for word_seq, tag_seq in corpus:
            unknown_tags = set(tag_seq) - tags.keys()
            if unknown_tags:
                raise ValueError(f'Unseen gold label in evaluation: {unknown_tags}')
            out.append(([words.get(w, 0) for w in word_seq], [tags[t] for t in tag_seq]))
        return out
    return (convert(train), convert(dev), convert(test)), words, tags
