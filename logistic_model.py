import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from functools import wraps
import time

# Globale Variable zum Speichern der letzten Laufzeit jeder Funktion
last_timing = {}

# ------------------------------------------------
# Decorators: my_logger & my_timer (nach Ori Cohen)
# ------------------------------------------------
def my_logger(orig_func):
    """Einfaches Logging der Funktionsaufrufe und Parameter"""
    import logging
    logging.basicConfig(
        filename=f"{orig_func.__name__}.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.info(f"Running: {orig_func.__name__} | args: {args} | kwargs: {kwargs}")
        return orig_func(*args, **kwargs)
    return wrapper


def my_timer(orig_func):
    """Misst und gibt die Laufzeit einer Funktion aus"""
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = orig_func(*args, **kwargs)
        end = time.time() - start
        last_timing[orig_func.__name__] = end
        print(f"→ {orig_func.__name__} ran in: {end:.4f} sec")
        return result
    return wrapper


def get_last_timing(func_name):
    """Gibt die letzte gemessene Laufzeit einer Funktion zurück"""
    return last_timing.get(func_name, 0.0)


# ------------------------------------------------
# ML PIPELINE
# ------------------------------------------------

@my_logger
@my_timer
def load_data(file_path="advertising.csv"):
    """Lädt den Datensatz"""
    print("\n=== Datensatz laden ===")
    df = pd.read_csv(file_path)
    print(f"  Datei: {file_path} | Shape: {df.shape}")
    return df


@my_logger
@my_timer
def train_model(df):
    """Trainiert das logistische Regressionsmodell"""
    print("\n=== Modelltraining ===")
    X = df.drop("Clicked_on_Ad", axis=1)
    y = df["Clicked_on_Ad"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("  Modelltraining abgeschlossen.")
    return model, X_test, y_test


@my_logger
@my_timer
def evaluate_model(model, X_test, y_test):
    """Bewertet das Modell mit Accuracy, Confusion Matrix und Klassifikationsbericht"""
    print("\n=== Modellevaluierung ===")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    print(f"  Genauigkeit (Accuracy): {accuracy:.2f}")
    print("  Confusion Matrix:")
    print(f"    {cm}")
    print("\n  Klassifikationsbericht (Auszug):")
    print(cr)
    return accuracy


# ------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------
if __name__ == "__main__":
    print("\n=== Starte logistic_model.py ===")

    df = load_data("advertising.csv")
    model, X_test, y_test = train_model(df)
    acc = evaluate_model(model, X_test, y_test)

    print(f"Final Accuracy: {acc:.2f}\n")
