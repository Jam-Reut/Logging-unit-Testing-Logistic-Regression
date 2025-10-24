import pandas as pd
import numpy as np
import logging
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------------------------------------------
# Dekorator für Zeitmessung
# ------------------------------------------------------------
timing_info = {}

def mytimer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        logging.info(f"Started '{func.__name__}'")
        logging.info(f"Running '{func.__name__}' ...")
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        timing_info[func.__name__] = elapsed
        logging.info(f"'{func.__name__}' executed in {elapsed:.4f} sec")
        logging.info(f"Completed '{func.__name__}' successfully.")
        return result
    return wrapper

def get_last_timing(func_name):
    return timing_info.get(func_name, None)

# ------------------------------------------------------------
# Funktionen für Daten, Training und Bewertung
# ------------------------------------------------------------
@mytimer
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

@mytimer
def train_model(df):
    X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
    y = df['Clicked on Ad']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_test, y_test

@mytimer
def evaluate_model(model, X_test, y_test):
    from logging import getLogger
    plain_logger = getLogger("plain")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Ausgabe im gewünschten Design:
    #plain_logger.info("\n[TEST 1 LOGGING: Vorhersageprüfung]\n")
    plain_logger.info(f"Genauigkeit (Accuracy): {acc:.2f}")
    plain_logger.info("Confusion Matrix:")
    plain_logger.info(f"{conf_matrix}\n")
    plain_logger.info("Klassifikationsbericht (Auszug):")
    plain_logger.info(report)
    plain_logger.info(f"\nFinal Accuracy: {acc:.2f}\n")

    return acc
