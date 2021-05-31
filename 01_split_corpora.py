import json
from itertools import islice
from pathlib import Path


def write_to_json(lines, base_dir):
    lines_it = iter(lines)

    line = next(lines_it).strip()
    while not line:
        line = next(lines_it).strip()
    year = int(line)

    line = next(lines_it).strip()
    while not line:
        line = next(lines_it).strip()
    title = line

    line = next(lines_it).strip()
    while not line:
        line = next(lines_it).strip()
    author = line.lstrip('by').strip()

    content = ''.join(list(lines_it))

    filepath = base_dir / f'{year}-{title}-{author}.json'
    with open(filepath, 'w') as json_file:
        json.dump(
            {'year': year, 'title': title, 'author': author, 'content': content},
            json_file,
        )

def main():
    base_dir = Path('works')
    base_dir.mkdir(parents=True, exist_ok=True)

    shake_txt = Path('shakespeare.txt')

    tmp_dir = base_dir / 'splitted'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with open(shake_txt, 'r') as shake_file:
        go_writing = True
        doc = []
        for line in islice(shake_file, 244, None):
            if '<<' in line:
                go_writing = False

            if go_writing:
                doc.append(line)

            if 'THE END' in line:
                write_to_json(doc, tmp_dir)
                doc = []

            if '>>' in line:
                go_writing = True

if __name__ == '__main__':
    main()
