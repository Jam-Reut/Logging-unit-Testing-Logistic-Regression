from functools import wraps
import logging
import time
import inspect
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------------------------------------
# Logging Setup (Ori Cohen Stil)
# ------------------------------------------------------
logging.basicConfig(
    filename="ml_system.log",
    level=logging.INFO,
    format="%(message)s",
    filemode="w"
)

# ------------------------------------------------------
# Testfall-Erkennung
# ------------------------------------------------------
def get_current_testcase():
    for frame in inspect.stack():
        if "test_predict" in frame.function or "test_predict_function" in frame.function:
            return "TESTFALL 1 (Vorhersage)"
        if "test_fit_runtime" in frame.function:
            return "TESTFALL 2 (Laufzeit)"
    return "MANUELLER LAUF"

# ------------------------------------------------------
# Decorators (Ori Cohen Stil)
# ------------------------------------------------------
def my_logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        context = get_current_testcase()
        logging.info("=" * 70)
        logging.info(f"{context} → {func.__name__} gestartet")
        logging.info(f"Running: {func.__name__} | args={args} | kwargs={kwargs}")
        result = func(*args, **kwargs)
        logging.info(f"{context} → {func.__name__} abgeschlossen")
        return result
    return wrapper

def my_timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        context = get_current_testcase()
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"  → {func.__name__} ran in: {duration:.4f} sec")
        logging.info(f"{context} → {func.__name__} ran in: {duration:.4f} sec")
        return result
    return wrapper

# ------------------------------------------------------
# ML Funktionen (YAML-Style Output)
# ------------------------------------------------------
@my_logger
@my_timer
def load_data(filename):
    """Lädt den CSV-Datensatz."""
    print("\nSchritt 1: Lade Datensatz")
    df = pd.read_csv(filename)
    print(f"  📂 Datei: {filename}")
    print(f"  📏 Form: {df.shape[0]} Zeilen × {df.shape[1]} Spalten")
    return df

@my_logger
@my_timer
def train_model(df):
    """Trainiert das Modell."""
    print("Schritt 2: Trainiere Modell")
    X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
    y = df['Clicked on Ad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"  Trainingsgröße: {X_train.shape[0]} | Testgröße: {X_test.shape[0]}")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("  ✅ Modelltraining abgeschlossen.")
    return model, X_test, y_test

@my_logger
@my_timer
def evaluate_model(model, X_test, y_test):
    """Bewertet das Modell."""
    print("Schritt 3: Modellevaluierung")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"  🎯 Genauigkeit (Accuracy): {acc:.2f}")
    print(f"  🧮 Confusion Matrix:")
    for row in cm:
        print(f"    {row}")
    print("  📄 Klassifikationsbericht:")
    for line in report.strip().split("\n"):
        print("    " + line)
    return acc

# ------------------------------------------------------
# Hauptprogramm
# ------------------------------------------------------
if __name__ == "__main__":
    print("# ======================================================")
    print("# AUSFÜHRUNG DER PIPELINE (logistic_model.py)")
    print("# ======================================================\n")

    print("Starte logistic_model.py\n")

    df = load_data("advertising.csv")
    model, X_test, y_test = train_model(df)
    acc = evaluate_model(model, X_test, y_test)

    print(f"\n🏁 Final Accuracy: {acc:.2f}\n")
