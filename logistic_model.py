import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import logging
from functools import wraps

# ------------------------------------------------
# Logging konfigurieren
# ------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

_last_timings = {}

# ------------------------------------------------
# Dekoratoren
# ------------------------------------------------
def my_timer(func):
    """Misst die Laufzeit einer Funktion und speichert sie."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        _last_timings[func.__name__] = elapsed
        logger.info(f"'{func.__name__}' executed in {elapsed:.4f} sec")
        return result
    return wrapper


def my_logger(func):
    """Protokolliert Start, Ende und Status einer Funktion."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Started '{func.__name__}'")
        logger.info(f"Running '{func.__name__}' ...")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Completed '{func.__name__}' successfully.")
            return result
        except Exception as e:
            logger.error(f"Error in '{func.__name__}': {e}")
            raise
    return wrapper


def get_last_timing(func_name):
    """Gibt die zuletzt gemessene Laufzeit zurück."""
    return _last_timings.get(func_name, None)


# ------------------------------------------------
# Hauptfunktionen
# ------------------------------------------------
@my_logger
@my_timer
def load_data(file_path):
    """CSV-Datensatz laden."""
    df = pd.read_csv(file_path)
    return df


@my_logger
@my_timer
def train_model(df):
    """Trainiert das logistische Regressionsmodell."""
    X = df[["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage"]]
    y = df["Clicked on Ad"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


@my_logger
@my_timer
def evaluate_model(model, X_test, y_test):
    """Bewertet das Modell mit Testdaten."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # --- Fachliche Ergebnisse zuerst ---
    print()
	print(f"Genauigkeit (Accuracy): {acc:.2f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nKlassifikationsbericht (Auszug):")
    print(classification_report(y_test, y_pred))
    print(f"\nFinal Accuracy: {acc:.2f}")

    return acc


# ------------------------------------------------
# Hauptprogramm
# ------------------------------------------------
if __name__ == "__main__":
    #print("\n=== Starte logistic_model.py ===\n")

    df = load_data("advertising.csv")
    model, X_test, y_test = train_model(df)
    acc = evaluate_model(model, X_test, y_test)
