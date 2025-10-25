import logging
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==============================================================
# Logging-Konfiguration (Ori Cohen Stil)
# ==============================================================
logger = logging.getLogger("MLPipeline")
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.propagate = False

# Globale Variable zur Speicherung der letzten Laufzeiten
_last_timings = {}


# ==============================================================
# Timer-Dekorator (für Performance-Messung)
# ==============================================================
def my_timer(func):
    """Misst die Ausführungszeit einer Funktion und speichert sie in _last_timings."""
    def wrapper(*args, **kwargs):
        start = time.time()
        logger.info(f"Started '{func.__name__}'")
        logger.info(f"Running '{func.__name__}' ...")
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        _last_timings[func.__name__] = elapsed
        logger.info(f"'{func.__name__}' executed in {elapsed:.4f} sec")
        logger.info(f"Completed '{func.__name__}' successfully.")
        return result
    return wrapper


# ==============================================================
# Logger-Dekorator (für Funktionsstruktur)
# ==============================================================
def my_logger(func):
    """Dekorator für konsistentes Funktionslogging."""
    def wrapper(*args, **kwargs):
        logger.info(f"--- Start {func.__name__} ---")
        result = func(*args, **kwargs)
        logger.info(f"--- Ende {func.__name__} ---")
        return result
    return wrapper


# ==============================================================
# Funktionen des ML-Pipelinesystems
# ==============================================================
@my_timer
@my_logger
def load_data(path="advertising.csv"):
    """Lädt Datensatz und gibt DataFrame zurück."""
    df = pd.read_csv(path)
    return df


@my_timer
@my_logger
def train_model(df):
    """Trainiert ein logistisches Regressionsmodell."""
    X = df[["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage"]]
    y = df["Clicked on Ad"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, X_test, y_test


@my_timer
@my_logger
def evaluate_model(model, X_test, y_test):
    """Berechnet Metriken zur Modellbewertung und gibt Accuracy zurück."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Konsolidierte Textausgabe (wird in den Tests gezeigt)
    report = (
        f"\nGenauigkeit (Accuracy): {acc:.2f}\n"
        f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n\n"
        "Klassifikationsbericht (Auszug):\n"
        f"{classification_report(y_test, y_pred)}\n"
        f"Final Accuracy: {acc:.2f}\n"
    )
    print(report)
    return acc


# ==============================================================
# Zugriffsfunktion für letzte Laufzeit
# ==============================================================
def get_last_timing(func_name):
    """Gibt die letzte gemessene Laufzeit einer Funktion zurück."""
    return _last_timings.get(func_name, None)


# ==============================================================
# Manueller Direktaufruf
# ==============================================================
if __name__ == "__main__":
    print("=== Starte logistic_model.py ===\n")
    df = load_data("advertising.csv")
    model, X_test, y_test = train_model(df)
    evaluate_model(model, X_test, y_test)
