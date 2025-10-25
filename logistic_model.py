import pandas as pd
import numpy as np
import logging
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =============================================================================
# LOGGING-KONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

# Für jede Funktion wird die Laufzeit zwischengespeichert
_last_timings = {}

# =============================================================================
# DEKORATOREN
# =============================================================================

def my_logger(func):
    """Decorator: Loggt Start, Ende und Erfolg von Funktionsaufrufen."""
    def wrapper(*args, **kwargs):
        name = func.__name__
        logger.info(f"Started '{name}'")
        logger.info(f"Running '{name}' ...")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Completed '{name}' successfully.")
            return result
        except Exception as e:
            logger.error(f"Error in '{name}': {e}")
            raise
    return wrapper


def my_timer(func):
    """Decorator: Misst die Ausführungszeit und speichert sie im globalen Timing-Dict."""
    def wrapper(*args, **kwargs):
        name = func.__name__
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        _last_timings[name] = elapsed
        logger.info(f"'{name}' executed in {elapsed:.4f} sec")
        return result
    return wrapper


def get_last_timing(name):
    """Gibt die letzte gemessene Laufzeit für eine Funktion zurück."""
    return _last_timings.get(name, None)

# =============================================================================
# FUNKTIONSDEFINITIONEN
# =============================================================================

@my_logger
@my_timer
def load_data(csv_path="advertising.csv"):
    """Lädt die Datendatei und gibt ein Pandas-DataFrame zurück."""
    df = pd.read_csv(csv_path)
    return df


@my_logger
@my_timer
def train_model(df):
    """Trainiert ein logistisches Regressionsmodell."""
    X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
    y = df['Clicked on Ad']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Letzte Laufzeit für Trainingsfunktion speichern
    _last_timings['train_model'] = get_last_timing('train_model')
    return model


@my_logger
@my_timer
def evaluate_model(model):
    """Bewertet das Modell mit Testdaten und gibt Accuracy, Confusion Matrix und Report zurück."""
    # Da der Test Daten selbst lädt, muss dieser Schritt separat geschehen
    df = pd.read_csv("advertising.csv")
    X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
    y = df['Clicked on Ad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, cm, report


# =============================================================================
# MAIN – optional für Debug-Zwecke
# =============================================================================
if __name__ == "__main__":
    print("=== Starte logistic_model.py ===\n")
    df = load_data()
    model = train_model(df)
    acc, cm, report = evaluate_model(model)

    print(f"\nGenauigkeit (Accuracy): {acc:.2f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nKlassifikationsbericht:")
    print(report)
