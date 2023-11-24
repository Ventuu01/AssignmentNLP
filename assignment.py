import requests
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import re
from bs4 import BeautifulSoup

# Funzione per ottenere il testo da Wikipedia tramite l'API
def get_wikipedia_text(title):
    endpoint = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "exintro": True,
    }

    response = requests.get(endpoint, params=params)
    data = response.json()

    # Estrazione testo dalla risposta JSON
    page = next(iter(data.get("query", {}).get("pages", {}).values()), {})
    return page.get("extract", None)


# Funzione per il pre-processing del testo
def preprocess_text(text):
    if text is None:
        return "" 
    # Rimozione dei tag HTML
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenizzazione
    try:
        tokens = word_tokenize(text)
    except (TypeError, ValueError):
        # Assegno lista vuota in caso di errori durante tokenizzazione
        tokens = []

    # Rimozione delle stopwords (con punteggiatura)
    stop_words = set(stopwords.words('english')+ list(string.punctuation))
    tokens = [word for word in tokens if word.lower() not in stop_words]

    # Lemmatizzazione 
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

# Dati di addestramento
data = [
    {"title": "Heart disease", "label": "medical"},
    {"title": "Technology", "label": "non-medical"},
    {"title": "Medicine", "label": "medical"},
    {"title": "Basketball", "label": "non-medical"},
    {"title": "Disease", "label": "medical"},
    {"title": "World literature", "label": "non-medical"},
    {"title": "Therapy", "label": "medical"},
    {"title": "Music", "label": "non-medical"},
    {"title": "Myocardial infarction", "label": "medical"},
    {"title": "Aristotle", "label": "non-medical"},
    {"title": "Mental disorder", "label": "medical"},
    {"title": "Camera obscura", "label": "non-medical"},
    {"title": "Evolutionary medicine", "label": "medical"},
    {"title": "Olympic Games", "label": "non-medical"},
    {"title": "Hospital medicine", "label": "medical"},
    {"title": "Mathematics", "label": "non-medical"},
    {"title": "Gastrointestinal disease", "label": "medical"},
    {"title": "Endocrine system", "label": "medical"},
    {"title": "Cancer treatment", "label": "medical"},
    {"title": "Respiratory diseases", "label": "medical"},
    {"title": "Neurological disorders", "label": "medical"},
    {"title": "Brain", "label": "medical"},
    {"title": "Surgery", "label": "medical"},
    {"title": "Genetic disorder", "label": "medical"},
    {"title": "Malaria", "label": "medical"},
    {"title": "PC game", "label": "non-medical"},
    {"title": "Fashion", "label": "non-medical"},
    {"title": "New York University", "label": "non-medical"},
    {"title": "Emmy Awards", "label": "non-medical"},
    {"title": "Albert_Einstein", "label": "non-medical"},
    {"title": "Italy", "label": "non-medical"},
    {"title": "Natural environment", "label": "non-medical"}
    
]

# Aggiunta testo wikipedia ai dati
for entry in data:
    entry["text"] = get_wikipedia_text(entry["title"])

# Preprocessing dei dati
for entry in data:
    entry["processed_text"] = preprocess_text(entry["text"])

# Creazione di feature usando Bag of Words (CountVectorizer)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([entry["processed_text"] for entry in data])
y = [entry["label"] for entry in data]

# Suddivisione dei dati in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Addestramento di un classificatore Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Valutazione del modello
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_rep)

# Lista di titoli da testare
title_test = [
    "Myocardial infarction",
    "Computer programming",
    "Cancer research",
    "Artificial intelligence",
    "Space exploration",
    "Literary criticism",
    "Syndrome",
    "Fashion design",
    "Alternative medicine",
    "Placebo",
    "Parkinson's disease",
    "Car"
]

# Preparazione dati per la previsione
data_pred = []
for title in title_test:
    text = get_wikipedia_text(title)
    if text:
        proc_text = preprocess_text(text)
        data_pred.append(proc_text)

# Trasforma i dati con il vettorizzatore 
X_previ = vectorizer.transform(data_pred)
print("PREVISIONE: ",X_previ)

#Previsione modello
previsions = nb_classifier.predict(X_previ)

# Stampa risultati
for title, prev in zip(title_test, previsions):
    print(f"Titolo: {title} - Previsto: {prev}")




