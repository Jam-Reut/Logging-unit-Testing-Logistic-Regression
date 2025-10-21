"""
logistic_model.py
-----------------
Trainings- und Evaluationspipeline für ein Logistic Regression Modell.
Implementiert nach dem Ansatz von Ori Cohen (Logging + Timing).
"""

from functools import wraps
import logging
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ======================================================
# LOGGING EINSTELLUNG
# ======================================================
logging.basicConfig(
    filename="logistic_model.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filemode="w"
)

# ======================================================
# DECORATORS
# ======================================================
def my_logger(func):
    """Decorator: Loggt Funktionsaufrufe mit Parametern."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Running: {func.__name__} | args={args} | kwargs={kwargs}")
        result = func(*args, **kwargs)
        logging.info(f"Completed: {func.__name__}")
        return result
    return wrapper


def my_timer(func):
    """Decorator: Misst Laufzeit der Funktion."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        print(f"TEST → {func.__name__} ran in: {duration:.4f} sec")
        return result
    return wrapper


# ======================================================
# MODELFUNKTIONEN
# ======================================================
@my_logger
@my_timer
def load_data(filename):
    print(f"\n=== Schritt 1: Lade Datensatz ===")
    df = pd.read_csv(filename)
    print(f"  Datei: {filename} | Shape: {df.shape}")
    return df


@my_logger
@my_timer
def train_model(df):
    print(f"\n=== Schritt 2: Trainiere Modell ===")
    X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
    y = df['Clicked on Ad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("  Modelltraining abgeschlossen.")
    return model, X_test, y_test


@my_logger
@my_timer
def evaluate_model(model, X_test, y_test):
    print(f"\n=== Schritt 3: Modellevaluierung ===")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=2)

    print(f"  Genauigkeit (Accuracy): {acc:.2f}")
    print("  Confusion Matrix:")
    for row in cm:
        print(f"    {row}")

    print("\n  Klassifikationsbericht (Auszug):")
    for line in report.strip().split("\n")[2:]:
        print("   ", line)
    print()
    return acc


# ======================================================
# PIPELINE AUSFÜHRUNG
# ======================================================
if __name__ == "__main__":
    print("=== Starte logistic_model.py ===")

    df = load_data("advertising.csv")
    model, X_test, y_test = train_model(df)
    acc = evaluate_model(model, X_test, y_test)

    print(f"Final Accuracy: {acc:.2f}\n")
