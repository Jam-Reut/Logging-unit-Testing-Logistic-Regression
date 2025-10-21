"""
logistic_model.py
-----------------
Pipeline zur Ausführung eines einfachen Logistic-Regression-Modells
inklusive Logging und Zeitmessung (nach Ori Cohen).
"""

from functools import wraps
import logging
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ======================================================
# LOGGING EINSTELLUNGEN
# ======================================================
logging.basicConfig(
    filename="ml_system.log",
    level=logging.INFO,
    format="%(message)s",
    filemode="w"
)


# ======================================================
# DECORATORS
# ======================================================
def my_logger(func):
    """Protokolliert Funktionsaufrufe im Logfile."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Running: {func.__name__} | args={args} | kwargs={kwargs}")
        result = func(*args, **kwargs)
        logging.info(f"Completed: {func.__name__}")
        return result
    return wrapper


def my_timer(func):
    """Misst und gibt Laufzeit der Funktion aus."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"  → {func.__name__} ran in: {duration:.4f} sec")
        return result
    return wrapper


# ======================================================
# FUNKTIONEN
# ======================================================
@my_logger
@my_timer
def load_data(filename):
    print("\n=== Schritt 1: Lade Datensatz ===")
    df = pd.read_csv(filename)
    print(f"  Datei: {filename}")
    return df


@my_logger
@my_timer
def train_model(df):
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
    print("\n=== Schritt 3: Modellevaluierung ===")
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

    print()  # genau eine Leerzeile nach Bericht
    return acc


# ======================================================
# PIPELINE START
# ======================================================
if __name__ == "__main__":
    print("=== Starte logistic_model.py ===")

    df = load_data("advertising.csv")

    model, X_test, y_test = train_model(df)
    _ = train_model(df)  # zweiter Lauf für Zeitmessung / Logging

    acc = evaluate_model(model, X_test, y_test)

    print(f"Final Accuracy: {acc:.2f}\n")
