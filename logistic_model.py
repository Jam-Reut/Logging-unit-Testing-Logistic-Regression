import pandas as pd
import numpy as np
import logging
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------------------------------------------
# Technischer Logger mit Zeitstempel
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Timer-Dekorator für Messungen
# ------------------------------------------------------------
timings = {}

def mytimer(func):
    def wrapper(*args, **kwargs):
        logger.info(f"Started '{func.__name__}'")
        logger.info(f"Running '{func.__name__}' ...")
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        timings[func.__name__] = elapsed
        logger.info(f"'{func.__name__}' executed in {elapsed:.4f} sec")
        logger.info(f"Completed '{func.__name__}' successfully.")
        return result
    return wrapper


def get_last_timing(func_name: str):
    return timings.get(func_name, None)

# ------------------------------------------------------------
# Hauptfunktionen
# ------------------------------------------------------------

@mytimer
def load_data(path: str):
    df = pd.read_csv(path)
    return df


@mytimer
def train_model(df: pd.DataFrame):
    X = df[['Daily Time Spent on Site', 'Age', 'Area Income',
            'Daily Internet Usage', 'Male']]
    y = df['Clicked on Ad']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_test, y_test


@mytimer
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    # Text an Tests zurückgeben, nicht drucken
    metrics_text = (
        f"Genauigkeit (Accuracy): {acc:.2f}\n"
        f"Confusion Matrix:\n{cm}\n\n"
        f"Klassifikationsbericht (Auszug):\n{report}\n"
        f"Final Accuracy: {acc:.2f}\n"
    )
    return acc, metrics_text
