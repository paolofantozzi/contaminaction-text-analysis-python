from itertools import combinations
from pathlib import Path

from tqdm import tqdm

def load_reviews(path):
    reviews = []
    with open(path, 'r') as reviews_file:
        for line in reviews_file:
            _, *review_parts = line.split(':')
            review = ':'.join(review_parts)
            reviews.append(review)
    return reviews

def shingles(text, size):
    return {text[start:start+size] for start in range(len(text)-size)}

def jaccard_sim(set_1, set_2):
    return len(set_1 & set_2) / len(set_1 | set_2)

def most_similar(reviews, sizes):
    results = []
    for size in sizes:
        reviews_shingles = [shingles(review, size) for review in reviews]
        scores = []
        for rev_idx_1, rev_idx_2 in tqdm(combinations(range(len(reviews)), 2)):
            set_1 = reviews_shingles[rev_idx_1]
            set_2 = reviews_shingles[rev_idx_2]
            score = jaccard_sim(set_1, set_2)
            scores.append((reviews[rev_idx_1], reviews[rev_idx_2], score))
        scores = sorted(scores, key=lambda x: x[2], reverse=True)
        results.append((scores[0], size))
    return results

def main():
    train_path = Path('works') / 'test.txt'
    reviews = load_reviews(train_path)[:5000]
    for (rev_1, rev_2, score), size in most_similar(reviews, [3]):
        print(rev_1, rev_2, score, size)

if __name__ == '__main__':
    main()

