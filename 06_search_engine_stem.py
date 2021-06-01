import re
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer

from doc_loader import load_docs

stemmer = SnowballStemmer(language='english')


class InvertedIndex:
    def __init__(self, all_docs):
        for doc in all_docs:
            doc['stemmed_words'] = set(map(stemmer.stem, doc['words']))
        self.all_words = sorted(set().union(*[doc['stemmed_words'] for doc in all_docs]))
        self.all_docs = all_docs

        class Term:
            all_docs_idxs = list(range(len(all_docs)))

            def __init__(self, doc_ids=None):
                self.docs = sorted(doc_ids or [])
                self.pos = {idx: set() for idx in self.docs}

            def __and__(self, other):
                return Term(set(self.docs) & set(other.docs))

            def __or__(self, other):
                return Term(set(self.docs) | set(other.docs))

            def __invert__(self):
                return Term(set(self.all_docs_idxs) - set(self.docs))

        self.Term = Term
        self.index = defaultdict(self.Term)

        for word in self.all_words:
            word_docs = [idx for idx, doc in enumerate(self.all_docs) if word in doc['stemmed_words']]
            self.index[word] = self.Term(word_docs)

        for doc_idx, doc in enumerate(self.all_docs):
            words_list = re.split(r'\W+', doc['content'].lower())
            for word_idx, word in enumerate(words_list):
                word = stemmer.stem(word)
                self.index[word].pos[doc_idx].add(word_idx)

    def term_from_word(self, word):
        negated = False
        if word.startswith('!'):
            word = word[1:]
            negated = True
        word = stemmer.stem(word)
        term = self.index[word]
        return ~term if negated else term

    def bool_query(self, q):
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

    def phrase_query(self, q):
        if not q:
            return []
        words = q.split()
        terms = [self.index[stemmer.stem(word)] for word in words]
        if not terms:
            return []
        possible_docs = set(terms[0].docs).intersection(*[set(t.docs) for t in terms])
        results_ids = set()
        for doc_idx in possible_docs:
            first_word_pos = terms[0].pos[doc_idx]
            for pos in first_word_pos:
                if all([(pos + i) in t.pos[doc_idx] for i, t in enumerate(terms)]):
                    results_ids.add(doc_idx)
        return [self.all_docs[idx] for idx in results_ids]


if __name__ == '__main__':
    docs = load_docs()
    iindex = InvertedIndex(docs)
    query = input('Cosa vuoi cercare?\n')
    print('Risultati:')
    if set(query) & {'!', '&', '|'}:
        results = iindex.bool_query(query.lower())
    else:
        results = iindex.phrase_query(query.lower())
    print(f'{len(results)}/{len(docs)}')
    for doc in results:
        print(f"{doc['title']} - {doc['year']}")

