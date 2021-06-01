import spacy

# Inizializzazione
nlp = spacy.load('en_core_web_sm')

sentence = "The striped bats are hanging on their feet for best"

# Analisi della frase
doc = nlp(sentence)

# Estrazione del lemma per ogni parola e unificazione
print(" ".join([token.lemma_ for token in doc]))
