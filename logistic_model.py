import logging
import pandas as pd
import time
from functools import wraps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ------------------------------------------------------------
# Technisches Logging (mit Zeitstempel)
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)

plain_logger = logging.getLogger("plain")
plain_handler = logging.StreamHandler()
plain_handler.setFormatter(logging.Formatter("%(message)s"))
plain_logger.addHandler(plain_handler)
plain_logger.propagate = False


# ------------------------------------------------------------
# Timer-Decorator für technische Logs
# ------------------------------------------------------------
def mytimer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Started '{func.__name__}'")
        logging.info(f"Running '{func.__name__}' ...")
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        duration = end - start
        logging.info(f"'{func.__name__}' executed in {duration:.4f} sec")
        logging.info(f"Completed '{func.__name__}' successfully.")
        wrapper.last_timing = duration
        return result
    wrapper.last_timing = 0
    return wrapper


# ------------------------------------------------------------
# Modellfunktionen
# ------------------------------------------------------------
@mytimer
def load_data(filename: str):
    df = pd.read_csv(filename)
    return df


@mytimer
def train_model(df: pd.DataFrame):
    X = df[["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage"]]
    y = df["Clicked on Ad"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_test, y_test


@mytimer
def evaluate_model(model, X_test, y_test):
    """Berechnet die Modellmetriken und gibt sie als formatierten Text zurück."""
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

    return acc, metrics_text


# ------------------------------------------------------------
# Helper für Unit-Test
# ------------------------------------------------------------
def get_last_timing(func_name: str):
    for f in [load_data, train_model, evaluate_model]:
        if f.__name__ == func_name:
            return getattr(f, "last_timing", 0)
    return 0


if __name__ == "__main__":
    #plain_logger.info("=== Starte logistic_model.py ===\n")
