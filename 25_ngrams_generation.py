from collections import defaultdict
from collections import deque
from pathlib import Path
from random import choice

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
        self.all_words = set()

    def _with_special_tokens(self, tokens, start_only=False):
        first_part = (('<s>',) * (self.size - 1)) + tuple(tokens)
        if start_only:
            return first_part
        return first_part + (('</s>',) * (self.size - 1))

    def add_text(self, tokens):
        tokens = self._with_special_tokens(tokens)
        for ngram in ngrams(tokens, self.size):
            self.all_ngrams.add(ngram)
            counter = self.model
            token, *remaining = ngram
            self.all_words.add(token)
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

    def _most_frequent_next(self, tokens):
        assert len(tokens) <= (self.size - 1)
        counter = self.model
        for token in tokens:
            counter = counter[token]
        max_count = 0
        max_token = ''
        for next_token in counter:
            next_count = self._recursive_count(counter[next_token], len(tokens)+1)
            if next_count > max_count:
                max_count = next_count
                max_token = next_token
        return max_token

    def probability_of_sentence(self, tokens):
        tokens = self._with_special_tokens(tokens)
        probabilities = [self._count(ngram) / self._count(ngram[:-1]) for ngram in ngrams(tokens, self.size) if ngram in self.all_ngrams]
        prob_array = np.array(probabilities)
        log_prob = np.sum(np.log(prob_array))
        return np.exp(log_prob)

    def generate_sentence(self, tokens):
        all_words = list(self.all_words)
        tokens = list(self._with_special_tokens(tokens, start_only=True))
        while tokens[-1] != '</s>' and len(tokens) < 100:
            last_part = tokens[-(self.size - 1):]
            new_token = self._most_frequent_next(last_part)
            if not new_token or new_token == tokens[-1]:
                new_token = choice(all_words)
            tokens.append(new_token)
        return tokens[self.size-1:-1]


def main():
    train_path = Path('works') / 'train.txt'
    model = NgramModel(3)
    for review in tqdm(load_reviews(train_path)):
        tokens = tuple(tokenizer(review))
        model.add_text(tokens)

    new_review = input('Inserisci la prima parte di una recensione.\n')
    while new_review:
        tokens = [str(token) for token in tokenizer(new_review)]
        new_review_tokens = model.generate_sentence(tokens)
        print(' '.join(new_review_tokens), '\n')
        new_review = input('Inserisci la prima parte di una recensione.\n')


if __name__ == '__main__':
    main()

