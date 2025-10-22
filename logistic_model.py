import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

# Laufzeiten speichern
_last_timings = {}

def my_timer(func):
    """Zeit messen & speichern (kein print hier!)."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        _last_timings[func.__name__] = time.time() - start
        return result
    return wrapper

def get_last_timing(func_name):
    return _last_timings.get(func_name, None)

@my_timer
def load_data(file_path):
    print("\n=== Datensatz laden ===")
    df = pd.read_csv(file_path)
    print(f"  Datei: {file_path} | Shape: {df.shape}")
    return df

@my_timer
def train_model(df):
    #print("\n=== Modelltraining ===")
    X = df[["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage"]]
    y = df["Clicked on Ad"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

@my_timer
def evaluate_model(model, X_test, y_test):
    print("\n=== Modellevaluierung ===")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Genauigkeit (Accuracy): {acc:.2f}")
    print("  Confusion Matrix:")
    print(cm)
    print("\n  Klassifikationsbericht (Auszug):")
    print(classification_report(y_test, y_pred))
    return acc

if __name__ == "__main__":
    #print("=== Starte logistic_model.py ===")
    df = load_data("advertising.csv")
    print(f"→ load_data ran in: {get_last_timing('load_data'):.4f} sec")

    model, X_test, y_test = train_model(df)
    print(f"→ train_model ran in: {get_last_timing('train_model'):.4f} sec")

    acc = evaluate_model(model, X_test, y_test)
    print(f"→ evaluate_model ran in: {get_last_timing('evaluate_model'):.4f} sec")
    print(f"Final Accuracy: {acc:.2f}")
	
