import pandas as pd
import numpy as np
import time
import logging
from functools import wraps
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===============================
# Dekoratoren (nach Ori Cohen)
# ===============================
def my_logger(func):
    logging.basicConfig(filename=f"{func.__name__}.log", level=logging.INFO)

    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Aufruf: {func.__name__} | args: {args} | kwargs: {kwargs}")
        result = func(*args, **kwargs)
        logging.info(f"Beendet: {func.__name__}")
        return result

    return wrapper


def my_timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time() - start_time
        print(f"TEST → {func.__name__} ran in: {end_time:.4f} sec")
        return result

    return wrapper


# ===============================
# ML-Pipeline Funktionen
# ===============================
@my_logger
@my_timer
def load_data(file_path):
    print("\n=== Schritt 1: Lade Datensatz ===")
    df = pd.read_csv(file_path)
    print(f"  Datei: {file_path} | Shape: {df.shape}")
    return df


@my_logger
@my_timer
def train_model(df):
    print("\n=== Schritt 2: Trainiere Modell ===")
    X = df[["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage"]]
    y = df["Clicked on Ad"]
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
    print(f"  Genauigkeit (Accuracy): {acc:.2f}")
    print("  Confusion Matrix:")
    print(f"    {cm}")
    print("\n  Klassifikationsbericht (Auszug):")
    print(classification_report(y_test, y_pred))
    return acc


# ===============================
# Hauptausführung (Pipeline)
# ===============================
if __name__ == "__main__":
    print("=== Starte logistic_model.py ===")
    df = load_data("advertising.csv")
    model, X_test, y_test = train_model(df)
    final_accuracy = evaluate_model(model, X_test, y_test)
    print(f"Final Accuracy: {final_accuracy:.2f}")
