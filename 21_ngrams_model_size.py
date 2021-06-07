from collections import defaultdict
from collections import deque
from pathlib import Path

import numpy as np
from spacy.lang.en import English
from tqdm import tqdm

nlp = English()
tokenizer = nlp.tokenizer

def load_reviews(path):
    reviews = []
    with open(path, 'r') as reviews_file:
        for line in reviews_file:
            _, *review_parts = line.strip().split(':')
            review = ':'.join(review_parts).strip().lower()
            reviews.append(review)
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
        probabilities = [self._count(ngram) / self._count(ngram[:-1]) for ngram in ngrams(tokens, self.size)]
        prob_array = np.array(probabilities)
        log_prob = np.sum(np.log(prob_array))
        return np.exp(log_prob)


def main():
    train_path = Path('works') / 'train.txt'
    test_path = Path('works') / 'test.txt'
    probabilities = []
    for size in range(2, 20):
        print(f'Model size: {size}')
        model = NgramModel(size)
        for review in tqdm(load_reviews(train_path)[:10000]):
            tokens = tuple(tokenizer(review))
            model.add_text(tokens)

        test_reviews = [tuple(tokenizer(review)) for review in tqdm(load_reviews(test_path)[:1000])]
        probs = np.array([model.probability_of_sentence(tokens) for tokens in tqdm(test_reviews)])
        mean_prob = np.mean(probs)
        probabilities.append((f'Size {size}:', mean_prob))
        print(mean_prob)
        print('===================')

    for desc, prob in probabilities:
        print(desc, prob)

if __name__ == '__main__':
    main()

