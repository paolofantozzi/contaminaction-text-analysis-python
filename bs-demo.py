import codecs
from pathlib import Path

from bs4 import BeautifulSoup

base_dir = Path('test_folder')
for path in base_dir.glob('**/*.html'):
    print(path.parent.name, path.name, path.stem)

    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as html_file:
        html = html_file.read()
        soup = BeautifulSoup(html, 'html.parser')

        print(soup.title.string)
        for link in soup.find_all('a'):
            print(link.string, link.get('href'))

        for par in soup.find_all('h1'):
            print(par.string)

        content = soup.get_text()
        print(content)
