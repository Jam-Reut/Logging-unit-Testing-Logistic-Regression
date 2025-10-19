import logging
import time
import functools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Logging-Konfiguration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# internes Dictionary für Laufzeiten
__timings = {}


def my_logger(func):
    """Dekorator für Logging"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Running '{func.__name__}'")
        result = func(*args, **kwargs)
        logging.info(f"Finished '{func.__name__}'")
        return result
    return wrapper


def my_timer(func):
    """Dekorator für Zeitmessung – speichert Laufzeit im internen Timing-Dict"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            __timings[func.__name__] = elapsed
            logging.info(f"{func.__name__} executed in {elapsed:.4f} sec")
    return wrapper


def get_last_timing(func_name: str):
    """Liefert die letzte gemessene Laufzeit (Sekunden) oder None"""
    return __timings.get(func_name)


@my_logger
@my_timer
def load_data(file_path: str):
    logging.info(f"Lade Daten aus {file_path}")
    df = pd.read_csv(file_path)
    logging.info(f"Daten geladen mit Shape {df.shape}")
    return df


@my_logger
@my_timer
def train_model(df):
    # Features und Ziel definieren
    X = df.drop("Clicked on Ad", axis=1)
    y = df["Clicked on Ad"]
    
    # Alle nicht-numerischen Spalten automatisch in numerische Spalten umwandeln
    X = pd.get_dummies(X, drop_first=True)  # drop_first=True verhindert Multikollinearität
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Modell trainieren
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    logging.info("Training abgeschlossen")
    return model, X_test, y_test



@my_logger
@my_timer
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    logging.info(f"Accuracy: {acc:.2f}")
    logging.info(f"Confusion Matrix:\n{cm}")
    return acc


if __name__ == "__main__":
    df = load_data("advertising.csv")
    model, X_test, y_test = train_model(df)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")
