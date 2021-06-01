#Definizione dello stemmer

from nltk.stem import PorterStemmer
ps = PorterStemmer()

#Scelta di parole con stem simili:

example_words =["python","pythoner","pythoning","pythoned","pythonly"]

#Processo di stemming:

for w in example_words:
    print(ps.stem(w))
