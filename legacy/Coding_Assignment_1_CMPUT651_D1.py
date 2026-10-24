import sys
import numpy as np
from C1.lib.strLib import cleanText, normalizedText
from C1.lib.easyX import easyX


path = "/media/adebayo/Windows/UofACanada/study/codes/CMPUT651_PGM"
FILENAMES = [[f"{path}/C1/data/train_preprocessed.csv", "train"],
            [f"{path}/C1/data/test_preprocessed.csv", "test"],
            [f"{path}/C1/data/dev_preprocessed.csv", "dev"]]

FILE_TRAIN = f"{path}/C1/data/train_preprocessed.csv"
FILE_Test = f"{path}/C1/data/test_preprocessed_mine.csv"
FILE_VOCAB = f"{path}/C1/data/vocab.ezx"

OUTPREFIX = f"{path}/results/HMM_single_probability_"
OUTLISTS = f"{path}/results/HMM_single_probability_meta.ezx"

ezx = easyX()
vocab = ezx.load(FILE_VOCAB)

id2word = vocab["id2word"]
word2id = vocab["word2id"]
id2tag  = vocab["id2tag"]
tag2id  = vocab["tag2id"]



def get_data(fname):
    fheader = open(fname, 'r')
    Lines = fheader.readlines()

    samples = []
    for linenum, line in enumerate(Lines):
        (words, tags) = line.split('\t')
        wordlist = words.strip().split(' ')
        taglist = tags.strip().split(' ')

        try:
            words_ids = [word2id[w] for w in wordlist]
            tags_ids = [tag2id[t] for t in taglist]
        except:
            print(linenum, line, wordlist)
            exit(1)
        samples.append((words_ids, tags_ids))

    fheader.close()
    return samples


def train(samples, k):
    """
    Train the HMM using supervised learning by counting transitions and emissions.
    """
    num_words = len(word2id)
    num_tags = len(tag2id)

    # Initialize counts with add-k smoothing
    P_init = np.ones(num_tags) * k
    P_transition = np.ones((num_tags, num_tags)) * k
    P_emission = np.ones((num_tags, num_words)) * k

    # Count occurrences of tags (initial, transition) and emissions (tag-word pairs)
    for words, tags in samples:
        P_init[tags[0]] += 1  # Initial POS tag count
        

        for i in range(len(tags) - 1):
            P_transition[tags[i], tags[i + 1]] += 1  # Transition from tags[i] to tags[i+1]
            P_emission[tags[i], words[i]] += 1  # Emission of word words[i] by POS tag tags[i]

        # Last word emission
        P_emission[tags[-1], words[-1]] += 1

    # Normalize the counts to get probabilities
    P_init /= P_init.sum()  # Normalize initial probabilities
    P_transition /= P_transition.sum(axis=1, keepdims=True)  # Normalize transitions
    P_emission /= P_emission.sum(axis=1, keepdims=True)  # Normalize emissions
    

    return P_init, P_transition, P_emission



def inf_logprob(samples, HMM):
    """
    Compute the log-probability of a sequence of words given the HMM.
    Penalize incorrect POS tag sequences.
    """
    P_init, P_transition, P_emission = HMM

    sum_log_prob = 0  # To sum up the log probabilities
    cnt = 0

    for words, tags in samples:
        n = len(tags)
        log_prob = 0

        # Initial probability for the first tag
        log_prob += np.log(P_init[tags[0]])

        # Emission probability for the first word
        log_prob += np.log(P_emission[tags[0], words[0]] + 1e-10)

        # Compute the log probability of the rest of the sequence
        for i in range(1, n):
            # Transition from tag[i-1] to tag[i]
            log_prob += np.log(P_transition[tags[i-1], tags[i]] + 1e-10)

            # Emission of the word by the current POS tag
            log_prob += np.log(P_emission[tags[i], words[i]] + 1e-10)

        sum_log_prob += log_prob
        cnt += 1

    # Return the average log probability
    return sum_log_prob / cnt if cnt > 0 else float('-inf')



if __name__ == '__main__':
    # Load training data and train the model
    samples = get_data(FILE_TRAIN)
    HMM = train(samples, 0)

     # Example test sentence
    #test_sent = 'I LIKE THE BOOK'
    #test_POS_1 = 'PPRP IN DT NN'
    #test_POS_2 = 'PRP IN DT VB'
    #test_sent = 'HE IS A GOOD LECTURER'
    #test_POS_1 = 'PRP VBZ DT JJ NN'
    #test_POS_2 = 'PRP VBZ DT JJ VB'
    test_sent = 'I LOVE MACHINE LEARNING'
    test_POS_1 = 'PRP VB NN NNP'
    test_POS_2 = 'PRP VB NN VBP'

    # Convert the sentence and POS tags to numerical indices
    X = [word2id[w] for w in test_sent.split()]
    Y1 = [tag2id[t] for t in test_POS_1.split()]
    Y2 = [tag2id[t] for t in test_POS_2.split()]

    # Run inference with both correct and incorrect POS tags
    print("Log Probability for POS 1: ", inf_logprob([(X, Y1)], HMM))
    print("Log Probability for POS 2: ", inf_logprob([(X, Y2)], HMM))


