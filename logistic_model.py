# ======================================================
# Logistic Regression Pipeline
# Ori Cohen Decorator Stil, aber mit bestehender Ausgabe
# ======================================================

from functools import wraps
import logging
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------------------------------------
# Logging Setup (einheitlich für alle Funktionen)
# ------------------------------------------------------
logging.basicConfig(
    filename="ml_system.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    filemode="w"
)

# ------------------------------------------------------
# Decorators (Ori Cohen Stil, aber angepasst)
# ------------------------------------------------------
def my_logger(func):
    """Decorator: Loggt Funktionsaufrufe & Parameter."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Running: {func.__name__} | args={args} | kwargs={kwargs}")
        result = func(*args, **kwargs)
        logging.info(f"Completed: {func.__name__}")
        return result
    return wrapper


def my_timer(func):
    """Decorator: Misst und gibt Laufzeit aus (Ori Cohen Stil)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        # Ori Cohen Stil – aber Format bleibt wie dein bisheriges Output
        print(f"{func.__name__} ran in: {duration:.4f} sec")
        logging.info(f"{func.__name__} ran in: {duration:.4f} sec")
        return result
    return wrapper


# ------------------------------------------------------
# ML-Funktionen mit Ori Cohen Decorators
# ------------------------------------------------------
@my_logger
@my_timer
def load_data(filename):
    """Lädt den CSV-Datensatz."""
    print("\n=== Schritt 1: Lade Datensatz ===")
    logging.info(f"Lade Daten aus {filename}")
    df = pd.read_csv(filename)
    print(f"📂 Datei: {filename}")
    print(f"📏 Form: {df.shape[0]} Zeilen × {df.shape[1]} Spalten\n")
    logging.info(f"Daten geladen mit Shape {df.shape}")
    return df


@my_logger
@my_timer
def train_model(df):
    """Trainiert das logistische Regressionsmodell."""
    print("=== Schritt 2: Trainiere Modell ===")
    X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
    y = df['Clicked on Ad']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Trainingsgröße: {X_train.shape[0]} | Testgröße: {X_test.shape[0]}")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("✅ Modelltraining abgeschlossen.\n")

    logging.info(f"Trainingsgröße: {X_train.shape[0]} | Testgröße: {X_test.shape[0]}")
    logging.info("Modelltraining abgeschlossen.")
    return model, X_test, y_test


@my_logger
@my_timer
def evaluate_model(model, X_test, y_test):
    """Bewertet das trainierte Modell."""
    print("=== Schritt 3: Modellevaluierung ===")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"🎯 Genauigkeit (Accuracy): {acc:.2f}")
    print("🧮 Confusion Matrix:")
    print(cm)
    print("\n📄 Klassifikationsbericht:")
    print(report)

    logging.info(f"Accuracy: {acc:.2f}")
    logging.info(f"Confusion Matrix:\n{cm}")
    logging.info(report)

    return acc


# ------------------------------------------------------
# Hauptlauf (zum manuellen Testen)
# ------------------------------------------------------
if __name__ == "__main__":
    print("=== Starte logistic_model.py ===")

    df = load_data("advertising.csv")
    model, X_test, y_test = train_model(df)
    acc = evaluate_model(model, X_test, y_test)
    print(f"🏁 Final Accuracy: {acc:.2f}")
