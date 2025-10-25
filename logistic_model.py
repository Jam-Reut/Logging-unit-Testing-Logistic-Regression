import time
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================================================
# Logger-Konfiguration (Ori Cohen Style)
# ==============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

_last_timings = {}

# ==============================================================
# Dekorator 1: my_logger – Verantwortlich für Funktionsverfolgung
# ==============================================================
def my_logger(func):
    """Protokolliert Start, Lauf und erfolgreichen Abschluss einer Funktion."""
    def wrapper(*args, **kwargs):
        logger.info(f"Started '{func.__name__}'")
        logger.info(f"Running '{func.__name__}' ...")
        result = func(*args, **kwargs)
        logger.info(f"Completed '{func.__name__}' successfully.")
        return result
    wrapper.__name__ = func.__name__
    return wrapper

# ==============================================================
# Dekorator 2: my_timer – Misst und speichert Ausführungszeit
# ==============================================================
def my_timer(func):
    """Misst die Ausführungszeit einer Funktion und speichert sie in _last_timings."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        _last_timings[func.__name__] = elapsed
        logger.info(f"'{func.__name__}' executed in {elapsed:.4f} sec")
        return result
    wrapper.__name__ = func.__name__
    return wrapper

# ==============================================================
# Funktionsdefinitionen mit beiden Dekoratoren (Ori Cohen Prinzip)
# ==============================================================
@my_logger
@my_timer
def load_data(path="advertising.csv"):
    df = pd.read_csv(path)
    return df

@my_logger
@my_timer
def train_model(df):
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
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    metrics_text = (
        f"\nGenauigkeit (Accuracy): {acc:.2f}\n"
        f"Confusion Matrix:\n{cm}\n\n"
        f"Klassifikationsbericht (Auszug):\n{report}\n"
        f"Final Accuracy: {acc:.2f}\n"
    )
    print(metrics_text)
    return acc, metrics_text

# ==============================================================
# Timing-Abfrage
# ==============================================================
def get_last_timing(func_name: str):
    """Gibt die zuletzt gemessene Laufzeit einer Funktion zurück."""
    return _last_timings.get(func_name)

# ==============================================================
# Direkter Start
# ==============================================================
if __name__ == "__main__":
    print("=== Starte logistic_model.py ===\n")
    df = load_data("advertising.csv")
    model, X_test, y_test = train_model(df)
    evaluate_model(model, X_test, y_test)
