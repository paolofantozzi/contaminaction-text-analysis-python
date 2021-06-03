import codecs
import json
from pathlib import Path
from urllib.parse import urljoin

import spacy
from bs4 import BeautifulSoup
from tqdm import tqdm

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

def html_to_dict(path):
    domain = path.parent.name
    file_name = path.stem
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as html_file:
        html = html_file.read()
    soup = BeautifulSoup(html, 'html.parser')
    try:
        title = str(soup.title.string)
    except Exception:
        title = ''
    title_lemmas = [(token.text, token.lemma_) for token in nlp(title) if not token.is_punct]
    this_url = f'http://{domain}/{path.name}'
    links = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if (not href) or href.startswith('javascript'):
            continue
        anchor = str(link.get_text())
        anchor_lemmas = [(token.text, token.lemma_) for token in nlp(anchor) if not token.is_punct]
        url = urljoin(this_url, href)
        links.append((url, anchor, anchor_lemmas))
    paragraphs = []
    for paragraph in soup.find_all('h1'):
        par_text = str(paragraph.get_text())
        par_lemmas = [(token.text, token.lemma_) for token in nlp(par_text) if not token.is_punct]
        paragraphs.append((par_text, par_lemmas))
    content = soup.get_text()
    lemmas = []
    for token in nlp(content):
        if token.is_punct:
            continue
        lemmas.append((token.text, token.lemma_, token.sent.start_char, token.sent.end_char))
    return {
        'domain': domain,
        'title': title,
        'title_lemmas': title_lemmas,
        'links': links,
        'content': content,
        'lemmas': lemmas,
        'paragraphs': paragraphs,
        'file_name': file_name,
        'url': this_url,
    }


def main():
    base_dir = Path('works') / 'weir'
    base_dir.mkdir(parents=True, exist_ok=True)

    html_dir = Path('weir-dataset-pages')

    pages = []
    for idx, html_path in tqdm(enumerate(html_dir.glob('**/*.html'))):
        domain = html_path.parent.name
        file_name = html_path.stem
        out_path = base_dir / f'{domain}-{file_name}.json'
        if out_path.exists():
            continue
        page = html_to_dict(html_path)
        with open(out_path, 'w') as out_file:
            json.dump(page, out_file, indent=4)

if __name__ == '__main__':
    main()
