"""
logistic_model.py
-----------------
Pipeline zur Ausführung eines einfachen Logistic-Regression-Modells
inklusive Logging und Zeitmessung (nach Ori Cohen).
"""

from functools import wraps
import logging
import time
import inspect
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ======================================================
# LOGGING SETUP
# ======================================================
logging.basicConfig(
    filename="ml_system.log",
    level=logging.INFO,
    format="%(message)s",
    filemode="w"
)


# ------------------------------------------------------
# Testfall-Erkennung (ermöglicht Logging im Kontext)
# ------------------------------------------------------
def get_current_testcase():
    for frame in inspect.stack():
        if "test_predict" in frame.function or "test_1_predict_function" in frame.function:
            return "TESTFALL 1 (Vorhersage)"
        if "test_fit_runtime" in frame.function or "test_2_fit_runtime" in frame.function:
            return "TESTFALL 2 (Laufzeit)"
    return "MANUELLER LAUF"


# ------------------------------------------------------
# Decorators: Logging und Zeitmessung (nach Ori Cohen)
# ------------------------------------------------------
def my_logger(func):
    """Decorator: schreibt Funktionsaufrufe ins Logfile."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        context = get_current_testcase()
        logging.info(f"{context} → {func.__name__} gestartet")
        result = func(*args, **kwargs)
        logging.info(f"{context} → {func.__name__} abgeschlossen")
        return result
    return wrapper


def my_timer(func):
    """Decorator: misst Laufzeit einer Funktion."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"  → {func.__name__} ran in: {duration:.4f} sec")
        logging.info(f"{func.__name__} ran in: {duration:.4f} sec")
        return result
    return wrapper


# ======================================================
# FUNKTIONEN DER PIPELINE
# ======================================================

@my_logger
@my_timer
def load_data(filename):
    """Schritt 1: Daten laden."""
    print("\n=== Schritt 1: Lade Datensatz ===")
    df = pd.read_csv(filename)
    print(f"  Datei: {filename}")
    return df


@my_logger
@my_timer
def train_model(df):
    """Schritt 2: Modell trainieren."""
    print("\n=== Schritt 2: Trainiere Modell ===")
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
    """Schritt 3: Modell evaluieren."""
    print("\n=== Schritt 3: Modellevaluierung ===")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=2)
    print(f"  Genauigkeit (Accuracy): {acc:.2f}\n")
    print("  Confusion Matrix:")
    for row in cm:
        print(f"    {row}")
    print("\n  Klassifikationsbericht (Auszug):")
    for line in report.strip().split("\n")[2:]:
        print("   ", line)
    return acc


# ======================================================
# PIPELINE-AUSFÜHRUNG (ohne Laufzeitanalyse)
# ======================================================
if __name__ == "__main__":
  
    # Schritt 1: Daten laden
    df = load_data("advertising.csv")

    # Schritt 2: Zwei Trainingsdurchläufe (zur Demonstration)
    model, X_test, y_test = train_model(df)
    _ = train_model(df)

    # Schritt 3: Modellevaluierung
    acc = evaluate_model(model, X_test, y_test)

    # Finale Ausgabe
    print(f"\nFinal Accuracy: {acc:.2f}\n")
