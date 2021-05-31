import numpy as np

from doc_loader import load_docs


class Term:
    def __init__(self, array):
        self.array = array

    def __and__(self, other):
        return Term(self.array & other.array)

    def __or__(self, other):
        return Term(self.array | other.array)

    def __invert__(self):
        return Term(1 - self.array)


class TDMatrix:
    def __init__(self, docs):
        self.docs = docs

        all_words_set = set()
        for doc in docs:
            all_words_set.update(doc['words'])
        all_words_lst = sorted(all_words_set)
        self.all_words_lst = all_words_lst
        self.word_idx_map = {word: idx for idx, word in enumerate(self.all_words_lst)}

        matrix = []
        for term in self.all_words_lst:
            row = [1 * (term in doc['words']) for doc in self.docs]
            matrix.append(row)
        self.matrix = np.array(matrix)

    def term_from_word(self, word):
        negated = False
        if word.startswith('!'):
            word = word[1:]
            negated = True
        try:
            array = self.matrix[self.word_idx_map[word]]
        except KeyError:
            array = np.zeros(self.matrix.shape[1], dtype=np.int8)
        return ~Term(array) if negated else Term(array)

    def query(self, q):
        if not q:
            return []
        words = q.split()
        word, *words = words
        term = self.term_from_word(word)
        while words:
            op, word, *words = words
            word_term = self.term_from_word(word)
            if op == '&':
                term = term & word_term
            elif op == '|':
                term = term | word_term
        indexes = list(np.argwhere(term.array == 1).flatten())
        return [self.docs[idx] for idx in indexes]


if __name__ == '__main__':
    docs = load_docs()
    tdmatrix = TDMatrix(docs)
    query = input('Cosa vuoi cercare?\n')
    print('Risultati:')
    results = tdmatrix.query(query.lower())
    print(f'{len(results)}/{len(docs)}')
    for doc in results:
        print(f"{doc['title']} - {doc['year']}")

