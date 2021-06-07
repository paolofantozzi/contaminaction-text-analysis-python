from collections import Counter
from collections import defaultdict
from collections import deque
from pathlib import Path

import numpy as np
from spacy.lang.en import English
from tabulate import tabulate
from tqdm import tqdm

nlp = English()
tokenizer = nlp.tokenizer

def load_reviews(path):
    reviews = []
    with open(path, 'r') as reviews_file:
        for line in reviews_file:
            starting, *review_parts = line.strip().split(':')
            label, *_ = starting.split()
            review = ':'.join(review_parts).strip().lower()
            reviews.append((label, review))
    return reviews

def ngrams(tokens, size):
    ngram = deque([], maxlen=size)
    for token in tokens:
        ngram.append(str(token))
        if len(ngram) == size:
            yield tuple(ngram)

def recursive_defaultdict(deep):
    if deep < 1:
        return int
    return lambda: defaultdict(recursive_defaultdict(deep - 1))


class NgramModel:
    def __init__(self, ngrams_size):
        self.model = recursive_defaultdict(ngrams_size)()
        self.size = ngrams_size
        self.all_ngrams = set()

    def _with_special_tokens(self, tokens):
        return (('<s>',) * (self.size - 1)) + tuple(tokens) + (('</s>',) * (self.size - 1))

    def add_text(self, tokens):
        tokens = self._with_special_tokens(tokens)
        for ngram in ngrams(tokens, self.size):
            self.all_ngrams.add(ngram)
            counter = self.model
            token, *remaining = ngram
            while remaining:
                counter = counter[token]
                token, *remaining = remaining
            counter[token] += 1

    def _recursive_count(self, counter, deep):
        if deep >= self.size:
            return counter
        total = 0
        for _, inner_counter in counter.items():
            total += self._recursive_count(inner_counter, deep+1)
        return total

    def _count(self, tokens):
        assert len(tokens) <= self.size
        deep = 0
        counter = self.model
        for token in tokens:
            deep += 1
            counter = counter[token]
        result = self._recursive_count(counter, deep)
        try:
            result += len(counter)
        except TypeError:
            result += 1
        return result

    def probability_of_sentence(self, tokens):
        tokens = self._with_special_tokens(tokens)
        probabilities = [self._count(ngram) / self._count(ngram[:-1]) for ngram in ngrams(tokens, self.size) if ngram in self.all_ngrams]
        prob_array = np.array(probabilities)
        log_prob = np.sum(np.log(prob_array))
        return np.exp(log_prob)


def main():
    train_path = Path('works') / 'train.txt'
    model_pos = NgramModel(2)
    model_neg = NgramModel(2)
    tot_docs = 0
    pos_docs = 0
    for label, review in tqdm(load_reviews(train_path)):
        tot_docs += 1
        if label == '__label__1':
            model = model_neg
        else:
            model = model_pos
            pos_docs += 1
        tokens = tuple(tokenizer(review))
        model.add_text(tokens)

    pos_docs_prob = pos_docs / tot_docs
    neg_docs_prob = 1 - pos_docs_prob

    test_path = Path('works') / 'test.txt'
    predictions = []
    labels = []
    for label, review in tqdm(load_reviews(test_path)[:1000]):
        tokens = tuple(tokenizer(review))
        this_review_pos_prob = model_pos.probability_of_sentence(tokens) * pos_docs_prob
        this_review_neg_prob = model_neg.probability_of_sentence(tokens) * neg_docs_prob
        predictions.append(1 * (this_review_pos_prob > this_review_neg_prob))
        labels.append(1 * (label == '__label__2'))

    predictions = np.array(predictions)
    labels = np.array(labels)

    true_positive = np.sum(1 * ((predictions == labels) & (predictions == 1)))
    true_negative = np.sum(1 * ((predictions == labels) & (predictions == 0)))
    false_positive = np.sum(1 * ((predictions != labels) & (predictions == 1)))
    false_negative = np.sum(1 * ((predictions != labels) & (predictions == 0)))

    confusion_matrix = [
        ['0', true_negative, false_positive],
        ['1', false_negative, true_positive],
    ]
    print(tabulate(confusion_matrix, headers=['0', '1']))

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1:', (2 * precision * recall) / (precision + recall))



if __name__ == '__main__':
    main()

