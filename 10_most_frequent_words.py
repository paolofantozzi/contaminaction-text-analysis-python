import json
import re
from collections import Counter
from pathlib import Path

import spacy

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

OLD_STOP_WORDS = frozenset(['thou', 'thy', 'thee', 'th'])

def clean_token(token):
    return re.sub('[^0-9a-zA-Z]+', '', token.lower())

def lemmas_non_stop(tokens):
    for token in tokens:
        if token.is_punct or token.is_stop:
            continue
        lemma = clean_token(token.lemma_)
        if len(lemma) < 2:
            continue
        if lemma in OLD_STOP_WORDS:
            continue
        yield lemma

def load_docs():
    works_dir = Path('works') / 'splitted'
    meta_counter = Counter()
    for work in works_dir.iterdir():
        with open(work, 'r') as work_file:
            doc = json.load(work_file)
        print(f"{doc['title']} - {doc['year']}")
        counter = Counter(lemmas_non_stop(nlp(doc['content'])))
        print(counter.most_common(10))
        meta_counter.update(dict(counter.most_common()))
    print('Tutti i docs:')
    print(meta_counter.most_common(10))

if __name__ == '__main__':
    load_docs()
