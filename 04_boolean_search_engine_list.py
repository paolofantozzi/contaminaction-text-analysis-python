from collections import defaultdict

from doc_loader import load_docs


class InvertedIndex:
    def __init__(self, all_docs):
        self.all_words = sorted(set().union(*[d['words'] for d in all_docs]))
        self.all_docs = all_docs

        class Term:
            all_docs_idxs = list(range(len(all_docs)))

            def __init__(self, doc_ids=None):
                self.docs = sorted(doc_ids or [])

            def __and__(self, other):
                return Term(set(self.docs) & set(other.docs))

            def __or__(self, other):
                return Term(set(self.docs) | set(other.docs))

            def __invert__(self):
                return Term(set(self.all_docs_idxs) - set(self.docs))

        self.Term = Term
        self.index = defaultdict(self.Term)

        for word in self.all_words:
            word_docs = [idx for idx, doc in enumerate(self.all_docs) if word in doc['words']]
            self.index[word] = self.Term(word_docs)

    def term_from_word(self, word):
        negated = False
        if word.startswith('!'):
            word = word[1:]
            negated = True
        term = self.index[word]
        return ~term if negated else term

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
        return [self.all_docs[idx] for idx in set(term.docs)]


if __name__ == '__main__':
    docs = load_docs()
    iindex = InvertedIndex(docs)
    query = input('Cosa vuoi cercare?\n')
    print('Risultati:')
    results = iindex.query(query)
    print(f'{len(results)}/{len(docs)}')
    for doc in results:
        print(f"{doc['title']} - {doc['year']}")

