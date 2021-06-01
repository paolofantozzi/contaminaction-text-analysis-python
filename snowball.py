#Definizione dello stemmer
import nltk
from nltk.stem.snowball import SnowballStemmer

# Parametri di linguaggio
snow_stemmer = SnowballStemmer(language='english')

# Lista di parole
words = ['cared','university','fairly','easily','singing',
         'sings','sung','singer','sportingly']

# Stemming
stem_words = []
for w in words:
    x = snow_stemmer.stem(w)
    stem_words.append(x)

# Print
for e1,e2 in zip(words,stem_words):
    print(e1+' ----> '+e2)
