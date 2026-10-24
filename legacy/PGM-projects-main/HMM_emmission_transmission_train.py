import sys
import numpy as np
import matplotlib.pyplot as plt
from C1.lib.strLib import cleanText, normalizedText
from C1.lib.easyX import easyX

# Paths and file definitions
path = "/media/adebayo/Windows/UofACanada/study/codes/CMPUT651_PGM"
FILE_TRAIN = f"{path}/C1/data/train_preprocessed.csv"
FILE_TEST = f"{path}/C1/data/test_preprocessed.csv"
FILE_DEV = f"{path}/C1/data/dev_preprocessed.csv"
FILE_VOCAB = f"{path}/C1/data/vocab.ezx"

ezx = easyX()
vocab = ezx.load(FILE_VOCAB)

id2word = vocab["id2word"]
word2id = vocab["word2id"]
id2tag  = vocab["id2tag"]
tag2id  = vocab["tag2id"]

def get_data(fname):
    with open(fname, 'r') as fheader:
        samples = []
        for line in fheader:
            words, tags = line.split('\t')
            wordlist = words.strip().split(' ')
            taglist = tags.strip().split(' ')
            words_ids = [word2id[w] for w in wordlist]
            tags_ids = [tag2id[t] for t in taglist]
            samples.append((words_ids, tags_ids))
    return samples



def train(samples, k, num_topics):
    num_words = len(word2id)
    num_tags = len(tag2id)

    # Initialize counts with Dirichlet priors (alpha)
    alpha = np.ones(num_tags) * k  # Dirichlet prior for tag distribution
    beta = np.ones((num_topics, num_words)) * k  # Adjusted to num_topics

    # Count occurrences
    topic_word_counts = np.zeros((num_topics, num_words))
    topic_counts = np.zeros(num_topics)
    
    for words, tags in samples:
        for i in range(len(tags)):
            topic = tags[i] % num_topics  # Assign topics based on tags (you can customize this)
            topic_word_counts[topic, words[i]] += 1
            topic_counts[topic] += 1

    # Normalize to get probabilities
    phi = (topic_word_counts + beta) / (topic_counts[:, np.newaxis] + beta.sum(axis=1)[:, np.newaxis])
    
    return alpha, phi


def analyze_topics(phi):
    # Analyze the topics based on the learned distributions
    for topic_id, topic_dist in enumerate(phi):
        top_words = np.argsort(topic_dist)[-5:][::-1]  # Get top 5 words for each topic
        print(f"Topic {topic_id}: " + ", ".join([id2word[word] for word in top_words]))

def max_product_inference(words, HMM):
    P_init, P_transition, P_emission = HMM
    n = len(words)
    num_tags = len(P_init)
    
    V = np.zeros((num_tags, n))
    B = np.zeros((num_tags, n), dtype=int)

    for t in range(num_tags):
        V[t, 0] = np.log(P_init[t] + 1e-10) + np.log(P_emission[t, words[0]] + 1e-10)
    
    for i in range(1, n):
        for t in range(num_tags):
            max_prob, best_prev_tag = max(
                (V[prev_tag, i - 1] + np.log(P_transition[prev_tag, t] + 1e-10), prev_tag)
                for prev_tag in range(num_tags)
            )
            V[t, i] = max_prob + np.log(P_emission[t, words[i]] + 1e-10)
            B[t, i] = best_prev_tag

    best_last_tag = np.argmax(V[:, n - 1])
    best_tags = [best_last_tag]
    for i in range(n - 1, 0, -1):
        best_tags.insert(0, B[best_tags[0], i])

    return best_tags

def calculate_accuracy(pred_tags, true_tags):
    correct_words = sum(1 for pt, tt in zip(pred_tags, true_tags) if pt == tt)
    word_level_acc = correct_words / len(true_tags)
    sentence_level_acc = int(pred_tags == true_tags)
    return word_level_acc, sentence_level_acc

def evaluate(samples, HMM):
    word_accs = []
    sent_accs = []
    for words, true_tags in samples:
        pred_tags = max_product_inference(words, HMM)
        word_acc, sent_acc = calculate_accuracy(pred_tags, true_tags)
        word_accs.append(word_acc)
        sent_accs.append(sent_acc)
    return np.mean(word_accs), np.mean(sent_accs)

def most_frequent_baseline(samples, train_samples):
    tag_counts = np.zeros(len(tag2id), dtype=int)
    word_tag_counts = {word: np.zeros(len(tag2id), dtype=int) for word in word2id.values()}
    
    for words, tags in train_samples:
        for word, tag in zip(words, tags):
            tag_counts[tag] += 1
            word_tag_counts[word][tag] += 1

    most_freq_tag = np.argmax(tag_counts)
    baseline_preds = []
    
    for words, true_tags in samples:
        pred_tags = [np.argmax(word_tag_counts[word]) if word in word_tag_counts else most_freq_tag for word in words]
        word_acc, sent_acc = calculate_accuracy(pred_tags, true_tags)
        baseline_preds.append((word_acc, sent_acc))
    
    word_accs, sent_accs = zip(*baseline_preds)
    return np.mean(word_accs), np.mean(sent_accs)

def plot_word_level_accuracy(ks, word_accuracies):
    plt.plot(ks, word_accuracies, marker='o')
    plt.xlabel("Smoothing factor \( k \)")
    plt.ylabel("Validation Word-level Accuracy")
    plt.title("Validation Word-level Accuracy for Different Smoothing Factors")
    plt.grid(True)
    plt.show()

def analyze_trending_topics(phi, num_topics=5):
    topic_distribution = np.sum(phi, axis=1)  # Sum across words to get topic distributions
    trending_topics = sorted(range(num_topics), key=lambda i: topic_distribution[i], reverse=True)
    
    print("Trending Topics and Their Percentages:")
    for topic in trending_topics:
        percentage = (topic_distribution[topic] / np.sum(topic_distribution)) * 100
        print(f"Topic {topic}: {percentage:.2f}%")

if __name__ == '__main__':
    train_samples = get_data(FILE_TRAIN)
    test_samples = get_data(FILE_TEST)
    dev_samples = get_data(FILE_DEV)

    # Experiment with different k values and find the best k
    ks = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    word_accuracies = []

    for k in ks:
        alpha, phi = train(train_samples, k, num_topics=5)  # Set num_topics to the desired number of topics
        
        # Initialize HMM parameters
        num_tags = len(tag2id)  # Number of unique tags
        num_words = len(word2id)  # Number of unique words
        P_init = np.ones(num_tags) / num_tags
        P_transition = np.ones((num_tags, num_tags)) / num_tags
        P_emission = np.ones((num_tags, num_words)) / num_words
        HMM = (P_init, P_transition, P_emission)

        # Evaluate with HMM
        word_acc, _ = evaluate(dev_samples, HMM)
        word_accuracies.append(word_acc)

    plot_word_level_accuracy(ks, word_accuracies)

    # Select the best k value
    print(f"ks: {ks}")
    print(f"word_accuracies: {word_accuracies}")
    best_k = ks[np.argmax(word_accuracies)]
    print(f"best k: {best_k}")

    # Train with the best k value
    alpha_best, phi_best = train(train_samples, best_k, num_topics=5)

    # Initialize best HMM parameters
    P_init_best = np.ones(num_tags) / num_tags
    P_transition_best = np.ones((num_tags, num_tags)) / num_tags
    P_emission_best = np.ones((num_tags, num_words)) / num_words
    HMM_best = (P_init_best, P_transition_best, P_emission_best)

    # Analyze the topics
    analyze_topics(phi_best)

    # Evaluate on test set
    test_word_acc, test_sent_acc = evaluate(test_samples, HMM_best)
    print(f"Test Word-level Accuracy with Best k: {test_word_acc}")
    print(f"Test Sentence-level Accuracy with Best k: {test_sent_acc}")

    # Baseline: Most frequent tag
    baseline_word_acc, baseline_sent_acc = most_frequent_baseline(test_samples, train_samples)
    print(f"Baseline Word-level Accuracy: {baseline_word_acc}")
    print(f"Baseline Sentence-level Accuracy: {baseline_sent_acc}")
