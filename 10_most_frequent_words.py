import json
import re
from collections import Counter
from pathlib import Path

import spacy

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

def clean_token(token):
    return re.sub('[^0-9a-zA-Z]+', '', token.lower())

def load_docs():
    works_dir = Path('works') / 'splitted'
    meta_counter = Counter()
    for work in works_dir.iterdir():
        with open(work, 'r') as work_file:
            doc = json.load(work_file)
        print(f"{doc['title']} - {doc['year']}")
        lemmas = [clean_token(token.lemma_) for token in nlp(doc['content']) if not token.is_punct]
        counter = Counter([l for l in lemmas if l])
        print(counter.most_common(10))
        meta_counter.update(dict(counter.most_common()))
    print('Tutti i docs:')
    print(meta_counter.most_common(10))

if __name__ == '__main__':
    load_docs()
