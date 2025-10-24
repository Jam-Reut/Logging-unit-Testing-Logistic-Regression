# logistic_model.py

import pandas as pd
import logging
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- technischer Logger (mit Zeitstempel) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

_TIMINGS = {}
_LAST_METRICS_TEXT = ""  # wird von evaluate_model gesetzt


# === Bezeichnungen beibehalten ===
def mytimer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        logger.info(f"Started '{func.__name__}'")
        logger.info(f"Running '{func.__name__}' ...")
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        _TIMINGS[func.__name__] = elapsed
        logger.info(f"'{func.__name__}' executed in {elapsed:.4f} sec")
        logger.info(f"Completed '{func.__name__}' successfully.")
        return result
    return wrapper

def get_last_timing(func_name: str):
    return _TIMINGS.get(func_name, None)

def get_last_metrics_text() -> str:
    return _LAST_METRICS_TEXT


@mytimer
def load_data(csv_path: str):
    return pd.read_csv(csv_path)


@mytimer
def train_model(df: pd.DataFrame):
    X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
    y = df['Clicked on Ad']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """
    Logging und Metriken werden NICHT vermischt:
    - Technische Logs (Started/Running/Executed/Completed) erscheinen zusammen.
    - Metriken werden gesammelt und vom Test NACH dem Log-Block ausgegeben.
    """
    global _LAST_METRICS_TEXT

    # Technische Logs manuell (ohne Decorator), damit der gesamte Log-Block zusammenbleibt
    start = time.time()
    logger.info("Started 'evaluate_model'")
    logger.info("Running 'evaluate_model' ...")

    # --- Metriken berechnen, Text puffern (noch nicht drucken) ---
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    _LAST_METRICS_TEXT = (
        "Genauigkeit (Accuracy): {:.2f}\n"
        "Confusion Matrix:\n"
        "{}\n\n"
        "Klassifikationsbericht (Auszug):\n"
        "{}\n\n"
        "Final Accuracy: {:.2f}\n"
    ).format(acc, cm, report, acc)

    # Abschluss-Logs
    elapsed = time.time() - start
    _TIMINGS["evaluate_model"] = elapsed
    logger.info(f"'evaluate_model' executed in {elapsed:.4f} sec")
    logger.info("Completed 'evaluate_model' successfully.")

    return acc
