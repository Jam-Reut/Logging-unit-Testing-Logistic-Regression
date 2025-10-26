import pandas as pd
import numpy as np
import logging
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================================================
# LOGGING-KONFIGURATION (Ori Kohen Stil)
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

_last_timings = {}

# ============================================================================
# DEKORATOREN
# ============================================================================
def my_logger(func):
    """Decorator: Loggt Start, Ausführung und Erfolg der Funktion."""
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
    """Decorator: Misst die Laufzeit einer Funktion und speichert sie."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        _last_timings[func.__name__] = elapsed
        logger.info(f"'{func.__name__}' executed in {elapsed:.4f} sec")
        return result
    return wrapper


def get_last_timing(name):
    """Gibt die letzte gemessene Laufzeit einer Funktion zurück."""
    return _last_timings.get(name, None)

# ============================================================================
# FUNKTIONSDEFINITIONEN
# ============================================================================
@my_logger
@my_timer
def load_data(csv_path="advertising.csv"):
    """Lädt die CSV-Datei."""
    df = pd.read_csv(csv_path)
    return df


@my_logger
@my_timer
def train_model(df):
    """Trainiert das Modell."""
    X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
    y = df['Clicked on Ad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


@my_logger
@my_timer
def evaluate_model(model):
    """Bewertet das Modell mit Testdaten."""
    df = pd.read_csv("advertising.csv")
    X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
    y = df['Clicked on Ad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, cm, report
