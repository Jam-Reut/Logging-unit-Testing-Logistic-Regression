
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Logging konfigurieren
logging.basicConfig(filename='logistic_model.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        logging.info("Daten erfolgreich geladen.")
        return data
    except FileNotFoundError:
        logging.error("Datei nicht gefunden.")
        raise

def train_model(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        logging.info("Modell erfolgreich trainiert.")
        return model, X_test, y_test
    except Exception as e:
        logging.error(f"Fehler beim Training: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    logging.info("Modell erfolgreich evaluiert.")
    return report
