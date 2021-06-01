import re

import spacy

# Inizializzazione
nlp = spacy.load('en_core_web_sm')

sentence = "The striped bats, are hanging on their- feet for best"

# Analisi della frase
doc = nlp(sentence)

# Estrazione del lemma per ogni parola e unificazione
lemmas = [token.lemma_ for token in doc if not token.is_punct]
lemmas = [re.sub(r'[^A-Za-z0-9]', '', lemma) for lemma in lemmas]
print(" ".join(lemmas))
