# logistic_model.py

import pandas as pd
import logging
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- technischer Logger (mit Zeitstempel) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

_TIMINGS = {}

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

# HYBRID: @mytimer für technische Logs; Metriken selbst ohne Zeitstempel via "plain"
@mytimer
def evaluate_model(model, X_test, y_test):
    plain = logging.getLogger("plain")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # kompaktes, zusammenhängendes Ausgabe-Layout
    plain.info("")  # Abstand
    plain.info("Genauigkeit (Accuracy): {:.2f}".format(acc))
    plain.info("Confusion Matrix:")
    plain.info(f"{cm}\n")
    plain.info("Klassifikationsbericht (Auszug):")
    plain.info(report)
    plain.info("\nFinal Accuracy: {:.2f}\n".format(acc))

    return acc
