import json
import math
import re
from collections import Counter
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm


def clean_token(token):
    return re.sub('[^0-9a-zA-Z]+', '', token.lower())


def load_docs():
    works_dir = Path('works') / 'weir'
    docs = []
    file_paths = sorted(list(works_dir.iterdir()))
    domains = set()
    for work in tqdm(file_paths):
        with open(work, 'r') as work_file:
            doc = json.load(work_file)
        domain = urlparse(doc['url']).netloc
        if domain in domains:
            continue
        domains.add(domain)
        lemmas = [l[1] for l in doc['lemmas']]
        docs.append({
            'domain': domain,
            'content_tokens_lemma': lemmas,
            'content_lemmas_set': set(lemmas),
        })
    return docs


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

    def tfidf_vec(self, lemmas_freqs):
        return [lemmas_freqs[lemma] * self.index[lemma].idf for lemma in self.all_words]

    def cosine_matrix(self):
        sorted_ids = sorted(list(range(len(self.all_docs))))
        tfidf_matrix = [self.tfidf_vec(Counter(self.all_docs[doc_id][f'{self.zone_prefix}_tokens_lemma'])) for doc_id in sorted_ids]
        return tfidf_matrix

def main():
    docs = load_docs()
    content_iindex = InvertedIndex(docs, 'content')
    matrix = np.array(content_iindex.cosine_matrix())
    X_norm = normalize(matrix)
    classifications = KMeans(n_clusters=10).fit_predict(X_norm)
    clusters = dict([(docs[doc_id]['domain'], int(cluster_id)) for doc_id, cluster_id in enumerate(classifications)])
    out_path = Path('works') / 'clusters.json'
    with open(out_path, 'w') as out_file:
        json.dump(clusters, out_file)

if __name__ == '__main__':
    main()
