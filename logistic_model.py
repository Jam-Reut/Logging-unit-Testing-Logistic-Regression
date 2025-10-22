import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

# Timer dictionary
_last_timings = {}

def timed(func):
    """Dekorator für Zeitmessung"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        _last_timings[func.__name__] = elapsed
        print(f"→ {func.__name__} ran in: {elapsed:.4f} sec")
        return result
    return wrapper

def get_last_timing(func_name):
    """Letzte gemessene Laufzeit abrufen"""
    return _last_timings.get(func_name, None)


@timed
def load_data(file_path):
    print("\n=== Datensatz laden ===")
    df = pd.read_csv(file_path)
    print(f"  Datei: {file_path} | Shape: {df.shape}")
    return df


@timed
def train_model(df):
    print("\n=== Modelltraining ===")
    X = df[["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage"]]
    y = df["Clicked on Ad"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_test, y_test


@timed
def evaluate_model(model, X_test, y_test):
    print("\n=== Modellevaluierung ===")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Genauigkeit (Accuracy): {acc:.2f}")
    print("  Confusion Matrix:")
    print(f"    {cm}")
    print("\n  Klassifikationsbericht (Auszug):")
    print(classification_report(y_test, y_pred))
    return acc


if __name__ == "__main__":
    print("=== Starte logistic_model.py ===")
    pass  # Hauptlauf deaktiviert, um doppelte Ausgabe zu vermeiden

