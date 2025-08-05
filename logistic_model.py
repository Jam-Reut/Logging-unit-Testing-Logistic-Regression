import pandas as pd
import logging
import time
from functools import wraps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Decorators ---
def my_logger(orig_func):
    """Loggt vor und nach Ausführung einer Funktion."""
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.info(f"Running '{orig_func.__name__}'")
        result = orig_func(*args, **kwargs)
        logging.info(f"Finished '{orig_func.__name__}'")
        return result
    return wrapper

def my_timer(orig_func):
    """Misst und loggt die Laufzeit einer Funktion."""
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = orig_func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logging.info(f"{orig_func.__name__} executed in {elapsed_time:.4f} sec")
        return result
    return wrapper

# --- Core Funktionen ---
@my_logger
def load_data(file_path):
    """CSV-Daten laden."""
    return pd.read_csv(file_path)

@my_logger
@my_timer
def train_model(df):
    """Logistische Regression trainieren."""
    X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
    y = df['Clicked on Ad']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_test, y_test

@my_logger
def evaluate_model(model, X_test, y_test):
    """Genauigkeit und Confusion Matrix berechnen."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    logging.info(f"Accuracy: {accuracy:.2f}")
    logging.info(f"Confusion Matrix:\n{cm}")
    return accuracy, cm

# --- Main ---
if __name__ == "__main__":
    df = load_data("advertising.csv")
    model, X_test, y_test = train_model(df)
    acc, cm = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {acc:.2f}")
