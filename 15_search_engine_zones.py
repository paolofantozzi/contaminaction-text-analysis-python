import json
import math
import re
from collections import Counter
from collections import defaultdict
from collections import namedtuple
from itertools import chain
from pathlib import Path

import numpy as np
import spacy
from Levenshtein import distance
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

def clean_token(token):
    return re.sub('[^0-9a-zA-Z]+', '', token.lower())


def load_docs():
    works_dir = Path('works') / 'weir'
    docs = []
    file_paths = list(works_dir.iterdir())
    for work in tqdm(file_paths[:1000]):
        with open(work, 'r') as work_file:
            doc = json.load(work_file)
        doc['content_tokens_text'] = [l[0] for l in doc['lemmas']]
        doc['content_tokens_lemma'] = [l[1] for l in doc['lemmas']]
        doc['content_tokens_positions'] = [(l[2], l[3]) for l in doc['lemmas']]
        doc['content_words_set'] = set(doc['content_tokens_text'])
        doc['content_lemmas_set'] = set(doc['content_tokens_lemma'])
        doc['title_tokens_text'] = [l[0] for l in doc['title_lemmas']]
        doc['title_tokens_lemma'] = [l[1] for l in doc['title_lemmas']]
        doc['title_tokens_positions'] = [(0, 0) for l in doc['title_lemmas']]
        doc['title_words_set'] = set(doc['title_tokens_text'])
        doc['title_lemmas_set'] = set(doc['title_tokens_lemma'])
        if doc['paragraphs']:
            doc['pars_tokens_text'] = [text for text, _ in chain.from_iterable(p[1] for p in doc['paragraphs'])]
            doc['pars_tokens_lemma'] = [lemma for _, lemma in chain.from_iterable(p[1] for p in doc['paragraphs'])]
        else:
            doc['pars_tokens_text'] = []
            doc['pars_tokens_lemma'] = []
        doc['pars_tokens_positions'] = [(0, 0) for l in doc['pars_tokens_lemma']]
        doc['pars_words_set'] = set(doc['pars_tokens_text'])
        doc['pars_lemmas_set'] = set(doc['pars_tokens_lemma'])
        docs.append(doc)
    return docs


Result = namedtuple('Result', ['doc', 'pos', 'score'])


class InvertedIndex:
    def __init__(self, all_docs, zone_prefix):
        self.all_words = sorted(set().union(*[d[f'{zone_prefix}_lemmas_set'] for d in all_docs]))
        self.all_docs = all_docs
        self.zone_prefix = zone_prefix

        class Term:
            all_docs_idxs = set(range(len(all_docs)))
            _log_n = math.log(len(all_docs))

            def __init__(self, doc_ids=None):
                actual_doc_ids = doc_ids or []
                self.docs = set(actual_doc_ids)
                self.pos = defaultdict(set)
                self.idf = 0

            def __and__(self, other):
                return Term(self.docs & other.docs)

            def __or__(self, other):
                return Term(self.docs | other.docs)

            def __invert__(self):
                return Term(self.all_docs_idxs - self.docs)

            def add_pos(self, doc_id, pos):
                self.docs.add(doc_id)
                self.pos[doc_id].add(pos)
                self.idf = self._log_n - math.log(len(self.docs))

            def tf(self, doc_id):
                return len(self.pos[doc_id])

            def tfidf(self, doc_ids=None):
                idf = self.idf
                doc_ids = doc_ids or sorted(self.docs)
                return [self.tf(doc_id) * idf for doc_id in doc_ids]

        self.Term = Term
        self.index = defaultdict(self.Term)

        for doc_idx, doc in enumerate(self.all_docs):
            for word_idx, word in enumerate(doc[f'{self.zone_prefix}_tokens_lemma']):
                self.index[word].add_pos(doc_idx, word_idx)

    def term_from_word(self, word):
        negated = False
        if word.startswith('!'):
            word = word[1:]
            negated = True
        lemma, *_ = [t.lemma_ for t in nlp(word)]
        term = self.index[lemma]
        return (~term, lemma) if negated else (term, lemma)

    def tfidf_vec(self, lemmas_freqs):
        return [lemmas_freqs[lemma] * self.index[lemma].idf for lemma in self.all_words]

    def cosine_scores(self, query_counts, doc_ids):
        query_vec = np.array(self.tfidf_vec(query_counts)).reshape(1, -1)
        sorted_ids = sorted(doc_ids)
        tfidf_matrix = [self.tfidf_vec(Counter(self.all_docs[doc_id][f'{self.zone_prefix}_tokens_lemma'])) for doc_id in sorted_ids]
        scores = cosine_similarity(
            tfidf_matrix,
            query_vec,
        )
        return dict(zip(sorted_ids, scores[:, 0]))

    def bool_query(self, q):
        if not q:
            return []
        words = q.split()
        word, *words = words
        counter = Counter()
        term, lemma = self.term_from_word(word)
        counter[lemma] += 1
        terms = [term]
        while words:
            op, word, *words = words
            word_term, lemma = self.term_from_word(word)
            counter[lemma] += 1
            terms.append(word_term)
            if op == '&':
                term = term & word_term
            elif op == '|':
                term = term | word_term

        scores = self.cosine_scores(counter, term.docs)
        return {doc_id: Result(self.all_docs[doc_id], (0, 0), scores[doc_id]) for doc_id in term.docs}

    def phrase_query(self, q):
        if not q:
            return []
        lemmas = [token.lemma_ for token in nlp(q)]
        counter = Counter(lemmas)
        terms = [self.index[lemma] for lemma in lemmas]
        if not terms:
            return []
        possible_docs = set(terms[0].docs).intersection(*[set(t.docs) for t in terms])
        results_ids = {}
        for doc_idx in possible_docs:
            first_word_pos = terms[0].pos[doc_idx]
            for pos in first_word_pos:
                if all([(pos + i) in t.pos[doc_idx] for i, t in enumerate(terms)]):
                    the_doc = self.all_docs[doc_idx]
                    first_pos_start = the_doc[f'{self.zone_prefix}_tokens_positions'][pos][0]
                    last_pos_end = the_doc[f'{self.zone_prefix}_tokens_positions'][pos+len(terms)-1][1]
                    results_ids[doc_idx] = (max(0, first_pos_start - 50), last_pos_end + 50)

        scores = self.cosine_scores(counter, results_ids.keys())
        return {idx: Result(self.all_docs[idx], snippet_pos, scores[idx]) for idx, snippet_pos in results_ids.items()}


class SpellingCorrection:
    def __init__(self, all_docs):
        self.all_words = set().union(*[doc['content_words_set'] for doc in all_docs])
        self.all_words = self.all_words.union(*[doc['title_words_set'] for doc in all_docs])
        self.all_words = self.all_words.union(*[doc['pars_words_set'] for doc in all_docs])

    def nearest(self, word):
        distances = [(distance(word, voc_word), voc_word) for voc_word in self.all_words]
        distances = sorted(distances, key=lambda e: e[0])
        return distances[0][1]

    def correct(self, query):
        words = query.split()
        correct_words = []
        for word in words:
            if word in self.all_words:
                correct_words.append(word)
                continue
            correct_words.append(self.nearest(word))
        return ' '.join(correct_words)

def merge_results(content_results, pars_results, title_results):
    all_doc_ids = set().union(content_results.keys(), pars_results.keys(), title_results.keys())
    results = []
    for doc_id in all_doc_ids:
        the_doc = (content_results.get(doc_id, None) or pars_results.get(doc_id, None) or title_results[doc_id]).doc
        empty_result = Result(the_doc, (0, 0), 0)
        content_result = content_results.get(doc_id, empty_result)
        title_result = title_results.get(doc_id, empty_result)
        pars_result = pars_results.get(doc_id, empty_result)
        tot_score = (title_result.score * 3) + (pars_result.score * 2) + content_result.score
        pos = content_result.pos
        results.append(Result(the_doc, pos, tot_score))
    return results

if __name__ == '__main__':
    docs = load_docs()
    corrector = SpellingCorrection(docs)
    content_iindex = InvertedIndex(docs, 'content')
    title_iindex = InvertedIndex(docs, 'title')
    pars_iindex = InvertedIndex(docs, 'pars')
    query = input('Cosa vuoi cercare?\n')
    if set(query) & {'!', '&', '|'}:
        content_results = content_iindex.bool_query(query)
        title_results = title_iindex.bool_query(query)
        pars_results = pars_iindex.bool_query(query)
    else:
        correct_query = corrector.correct(query)
        if query != correct_query:
            print('Risultati per:')
            print(correct_query)
        content_results = content_iindex.phrase_query(correct_query)
        title_results = title_iindex.phrase_query(correct_query)
        pars_results = pars_iindex.phrase_query(correct_query)
    results = merge_results(content_results, pars_results, title_results)
    print('Risultati:')
    print(f'{len(results)}/{len(docs)}')
    for result in sorted(results, key=lambda r: r.score, reverse=True)[:10]:
        doc = result.doc
        score = result.score
        snippet_start, snippet_end = result.pos
        print(f"{doc['title']} - {doc['url']}: {score}")
        print('-' * 30)
        print(doc['content'][snippet_start:snippet_end])
        print()

