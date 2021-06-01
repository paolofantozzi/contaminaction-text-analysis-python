import json
import re
from collections import defaultdict
from pathlib import Path

import spacy

spacy.require_gpu()
nlp = spacy.load("en_core_web_sm")

def clean_token(token):
    return re.sub('[^0-9a-zA-Z]+', '', token.lower())


def load_docs():
    works_dir = Path('works') / 'splitted'
    docs = []
    for work in works_dir.iterdir():
        with open(work, 'r') as work_file:
            doc = json.load(work_file)
        doc['tokens'] = [clean_token(token.lemma_) for token in nlp(doc['content']) if not token.is_punct]
        doc['lemmas'] = set(doc['tokens'])
        docs.append(doc)
    return docs


class InvertedIndex:
    def __init__(self, all_docs):
        self.all_words = sorted(set().union(*[d['lemmas'] for d in all_docs]))
        self.all_docs = all_docs

        class Term:
            all_docs_idxs = list(range(len(all_docs)))

            def __init__(self, doc_ids=None):
                actual_doc_ids = doc_ids or []
                self.docs = sorted(actual_doc_ids)
                self.pos = {idx: set() for idx in actual_doc_ids}

            def __and__(self, other):
                return Term(set(self.docs) & set(other.docs))

            def __or__(self, other):
                return Term(set(self.docs) | set(other.docs))

            def __invert__(self):
                return Term(set(self.all_docs_idxs) - set(self.docs))

        self.Term = Term
        self.index = defaultdict(self.Term)

        for word in self.all_words:
            word_docs = [idx for idx, doc in enumerate(self.all_docs) if word in doc['lemmas']]
            self.index[word] = self.Term(word_docs)

        for doc_idx, doc in enumerate(self.all_docs):
            for word_idx, word in enumerate(doc['tokens']):
                self.index[word].pos[doc_idx].add(word_idx)

    def term_from_word(self, word):
        negated = False
        if word.startswith('!'):
            word = word[1:]
            negated = True
        lemma, *_ = [t.lemma_ for t in nlp(word)]
        term = self.index[lemma]
        return (~term, lemma) if negated else (term, lemma)

    def bool_query(self, q):
        if not q:
            return []
        words = q.split()
        word, *words = words
        term, _ = self.term_from_word(word)
        while words:
            op, word, *words = words
            word_term, _ = self.term_from_word(word)
            if op == '&':
                term = term & word_term
            elif op == '|':
                term = term | word_term
        return [self.all_docs[idx] for idx in set(term.docs)]

    def phrase_query(self, q):
        if not q:
            return []
        terms = [self.index[token.lemma_] for token in nlp(q)]
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
