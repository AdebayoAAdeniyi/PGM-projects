import sys
import numpy as np
from C1.lib.strLib import cleanText, normalizedText
from C1.lib.easyX import easyX

path = "/media/adebayo/Windows/UofACanada/study/codes/CMPUT651_PGM"
FILE_TRAIN = f"{path}/C1/data/train_preprocessed.csv"
FILE_Test = f"{path}/C1/data/test_preprocessed_mine.csv"
FILE_VOCAB = f"{path}/C1/data/vocab.ezx"

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
    num_words = len(word2id)
    num_tags = len(tag2id)

    # Initialize counts with add-k smoothing
    P_init = np.ones(num_tags) * k
    P_transition = np.ones((num_tags, num_tags)) * k
    P_emission = np.ones((num_tags, num_words)) * k

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

def max_product_inference(words, HMM):
    P_init, P_transition, P_emission = HMM
    n = len(words)
    num_tags = len(P_init)
    
    # Initialize the max-product (Viterbi) table and back-pointer table
    V = np.zeros((num_tags, n))
    B = np.zeros((num_tags, n), dtype=int)

    # Initial step: initialize the first column
    for t in range(num_tags):
        V[t, 0] = np.log(P_init[t] + 1e-10) + np.log(P_emission[t, words[0]] + 1e-10)
    
    # Forward pass: populate the max-product values with back-pointers
    for i in range(1, n):
        for t in range(num_tags):
            max_prob, best_prev_tag = max(
                (V[prev_tag, i - 1] + np.log(P_transition[prev_tag, t] + 1e-10), prev_tag)
                for prev_tag in range(num_tags)
            )
            V[t, i] = max_prob + np.log(P_emission[t, words[i]] + 1e-10)
            B[t, i] = best_prev_tag

    # Backtracking to get the most likely sequence of states
    best_last_tag = np.argmax(V[:, n - 1])
    best_tags = [best_last_tag]
    for i in range(n - 1, 0, -1):
        best_tags.insert(0, B[best_tags[0], i])

    return best_tags, np.max(V[:, n - 1])

if __name__ == '__main__':
    # Load training data and train the model
    samples = get_data(FILE_TRAIN)
    HMM = train(samples, 0.1)

    # Example test sentence
    #test_sent = 'I READ THE BOOK'
    #test_sent = 'I READ BOOK FLIGHTS'
    #test_sent = 'HE IS A GOOD LECTURER'
    test_sent = 'HE IS GOOD LECTURER  TEACHING'
    test_words = [word2id[w] for w in test_sent.split()]

    # Run max-product inference
    best_tags, best_score = max_product_inference(test_words, HMM)

    # Convert tags to words
    predicted_tags = [id2tag[t] for t in best_tags]
    print("Predicted POS tags:", predicted_tags)
    print("Best score (log-probability):", best_score)
