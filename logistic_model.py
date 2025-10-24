# logistic_model.py

import pandas as pd
import logging
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- technischer Logger mit Zeitstempel (für @mytimer) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

# Laufzeiten speichern
_timing_info = {}

# === Bezeichnungen wie gefordert beibehalten ===
def mytimer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        logger.info(f"Started '{func.__name__}'")
        logger.info(f"Running '{func.__name__}' ...")
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        _timing_info[func.__name__] = elapsed
        logger.info(f"'{func.__name__}' executed in {elapsed:.4f} sec")
        logger.info(f"Completed '{func.__name__}' successfully.")
        return result
    return wrapper

def get_last_timing(func_name: str):
    return _timing_info.get(func_name, None)

# === Hauptfunktionen ===
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

# ⚠️ KEIN Dekorator hier – sonst würden Zeitstempel vor den Metriken erscheinen
def evaluate_model(model, X_test, y_test):
    """Gibt Metriken im gewünschten Design über den plain-Logger aus und liefert Accuracy zurück."""
    plain_logger = logging.getLogger("plain")  # zeitstempelloser Logger

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Ausgabeformat exakt wie gefordert (ohne Zeitstempel)
    plain_logger.info(f"Genauigkeit (Accuracy): {acc:.2f}")
    plain_logger.info("Confusion Matrix:")
    plain_logger.info(f"{cm}\n")
    plain_logger.info("Klassifikationsbericht (Auszug):")
    plain_logger.info(report)
    plain_logger.info(f"\nFinal Accuracy: {acc:.2f}\n")

    return acc
