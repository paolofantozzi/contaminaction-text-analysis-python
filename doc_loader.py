import json
import re
from pathlib import Path


def load_docs():
    works_dir = Path('works') / 'splitted'

    docs = []
    for work in works_dir.iterdir():
        with open(work, 'r') as work_file:
            doc = json.load(work_file)
        doc['words'] = set(re.split(r'\W+', doc['content'].lower()))
        docs.append(doc)

    return docs

if __name__ == '__main__':
    load_docs()
