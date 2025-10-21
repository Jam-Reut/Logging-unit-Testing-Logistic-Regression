# ======================================================
# Logistic Regression Example
# Mit erweiterten Ausgaben & Ori Cohen–Style Logging
# ======================================================

import logging
import time
from functools import wraps
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ======================================================
# Decorators (Logging + Timing)
# ======================================================
def my_logger(func):
    """Schreibt Funktionsaufrufe in eine separate Logdatei."""
    logging.basicConfig(
        filename=f"{func.__name__}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"▶️ Starte Funktion '{func.__name__}' mit args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        logging.info(f"✅ Funktion '{func.__name__}' erfolgreich beendet.")
        return result
    return wrapper


def my_timer(func):
    """Misst die Laufzeit, gibt sie aus und loggt sie."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} ran in: {elapsed:.4f} sec\n")
        logging.info(f"⏱ '{func.__name__}' Laufzeit: {elapsed:.4f} sec")
        return result
    return wrapper


# ======================================================
# ML Workflow mit erweiterten Ausgaben
# ======================================================
@my_logger
@my_timer
def load_data(file_path: str):
    print("\n=== Schritt 1: Lade Datensatz ===")
    df = pd.read_csv(file_path)
    print(f"📂 Datei: {file_path}")
    print(f"📏 Form: {df.shape[0]} Zeilen × {df.shape[1]} Spalten\n")
    logging.info(f"Daten erfolgreich geladen: {file_path} | Shape={df.shape}")
    return df


@my_logger
@my_timer
def train_model(df):
    print("=== Schritt 2: Trainiere Modell ===")
    X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
    y = df['Clicked on Ad']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"Trainingsgröße: {len(X_train)} | Testgröße: {len(X_test)}")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("✅ Modelltraining abgeschlossen.\n")
    logging.info("Training abgeschlossen.")
    logging.info(f"Trainingsgröße={len(X_train)}, Testgröße={len(X_test)}")
    return model, X_test, y_test


@my_logger
@my_timer
def evaluate_model(model, X_test, y_test):
    print("=== Schritt 3: Modellevaluierung ===")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    print(f"🎯 Genauigkeit (Accuracy): {acc:.2f}")
    print("🧮 Confusion Matrix:")
    print(cm)
    print("\n📄 Klassifikationsbericht:")
    print(report)

    logging.info("📊 Modellevaluierung abgeschlossen.")
    logging.info(f"Accuracy: {acc:.3f}")
    logging.info(f"Confusion Matrix:\n{cm}")
    logging.info("Classification Report:\n" + report)
    return acc


# ======================================================
# Main-Ausführung
# ======================================================
if __name__ == "__main__":
    df = load_data("advertising.csv")
    model, X_test, y_test = train_model(df)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"🏁 Final Accuracy: {accuracy:.2f}")
