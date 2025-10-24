"""
logistic_model.py

Implementierung einer ML-Pipeline mit Logging und Zeitmessung
nach den Prinzipien von Ori Cohen (2019):
- Einheitliches Logging mit dem logging-Modul
- Nutzung von @my_logger und @my_timer Dekoratoren
- Keine Vermischung von Logging und Testergebnissen
"""

import logging
import time
import pandas as pd
from functools import wraps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ================================================================
# EINHEITLICHER LOGGER (Ori Cohen Prinzip)
# ================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

# ================================================================
# DEKORATOREN
# ================================================================
def my_logger(func):
    """Protokolliert Start und Ende der Funktion."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Started '{func.__name__}'")
        logger.info(f"Running '{func.__name__}' ...")
        result = func(*args, **kwargs)
        logger.info(f"Completed '{func.__name__}' successfully.")
        return result
    return wrapper


def my_timer(func):
    """Misst Laufzeit der Funktion und speichert sie im Funktionsobjekt."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"'{func.__name__}' executed in {elapsed:.4f} sec")
        wrapper.last_timing = elapsed
        return result
    return wrapper


def get_last_timing(func_name: str):
    """
    Gibt die zuletzt gemessene Laufzeit einer mit @my_timer dekorierten Funktion zurück.
    """
    func = globals().get(func_name)
    if hasattr(func, "last_timing"):
        return func.last_timing
    return None


# ================================================================
# PIPELINE-FUNKTIONEN
# ================================================================
@my_logger
@my_timer
def load_data(csv_path: str):
    """Lädt die Daten aus einer CSV-Datei."""
    df = pd.read_csv(csv_path)
    return df


@my_logger
@my_timer
def train_model(df: pd.DataFrame):
    """Trainiert das Logistische Regressionsmodell."""
    X = df[["Daily Time Spent on Site", "Age", "Area Income",
            "Daily Internet Usage", "Male"]]
    y = df["Clicked on Ad"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_test, y_test


@my_logger
@my_timer
def evaluate_model(model, X_test, y_test):
    """Bewertet das trainierte Modell auf Testdaten."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    metrics_text = (
        f"\nGenauigkeit (Accuracy): {acc:.2f}\n"
        f"Confusion Matrix:\n{cm}\n\n"
        f"Klassifikationsbericht (Auszug):\n{cr}\n"
        f"Final Accuracy: {acc:.2f}\n"
    )

    # Ori Cohen: technische Logs in logger, fachliche Ausgaben via print()
    print(metrics_text)
    return acc, metrics_text


# ================================================================
# MANUELLER START (wird nur bei direktem Aufruf ausgeführt)
# ================================================================
if __name__ == "__main__":
    print("=== Starte logistic_model.py ===\n")
    # Nur ein Beispiel-Durchlauf, keine Tests
    try:
        df = load_data("advertising.csv")
        model, X_test, y_test = train_model(df)
        evaluate_model(model, X_test, y_test)
    except FileNotFoundError:
        print("⚠️  Datei 'advertising.csv' nicht gefunden. Bitte sicherstellen, dass sie im selben Verzeichnis liegt.")
