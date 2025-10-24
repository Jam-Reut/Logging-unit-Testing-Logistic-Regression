import logging
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ================================================================
# LOGGER-KONFIGURATION (entspricht Ori Cohens „technical logger“)
# ================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

# ================================================================
# DECORATORS nach Ori Cohen
# ================================================================

def my_logger(func):
    """Protokolliert Start und Ende jeder Funktion."""
    def wrapper(*args, **kwargs):
        logger.info(f"Started '{func.__name__}'")
        logger.info(f"Running '{func.__name__}' ...")
        result = func(*args, **kwargs)
        logger.info(f"Completed '{func.__name__}' successfully.")
        return result
    return wrapper


def my_timer(func):
    """Misst Laufzeit und kombiniert Logging mit Zeitmessung."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        logger.info(f"'{func.__name__}' executed in {elapsed:.4f} sec")
        wrapper.last_timing = elapsed
        return result
    return wrapper


# ================================================================
# HILFSFUNKTION zum Abruf der letzten Laufzeit
# ================================================================
def get_last_timing(func_name: str):
    """Ermöglicht Zugriff auf die letzte gemessene Laufzeit."""
    func = globals().get(func_name)
    if hasattr(func, "last_timing"):
        return func.last_timing
    return None


# ================================================================
# MODELLFUNKTIONEN (kombinierter Einsatz beider Decorators)
# ================================================================
@my_logger
@my_timer
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


@my_logger
@my_timer
def train_model(df):
    X = df[["Daily Time Spent on Site", "Age", "Area Income",
            "Daily Internet Usage", "Male"]]
    y = df["Clicked on Ad"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_test, y_test


@my_logger
@my_timer
def evaluate_model(model, X_test, y_test):
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
    print(metrics_text)
    return acc, metrics_text


# ================================================================
# HAUPTPROGRAMM (wird in Aufgabenstellung meist leer gelassen)
# ================================================================
if __name__ == "__main__":
    # Der Professor möchte keine zusätzliche Konsolenausgabe hier
    # Daher bleibt dieser Block leer (keine "Hack"-Ausgabe)
    pass
