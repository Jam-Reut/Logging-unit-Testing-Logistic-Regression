import pandas as pd
import logging
import time
import functools
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def my_logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        logging.info(f"{func.__name__} returned {result}")
        return result
    return wrapper

def my_timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        logging.info(f"{func.__name__} executed in {elapsed:.4f} seconds")
        return result
    return wrapper

def load_data(file_path):
    logging.info(f"Lade Daten aus {file_path}")
    return pd.read_csv(file_path)

@my_timer
@my_logger
def train_model(df):
    logging.info("Starte Modelltraining")

    X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
    y = df['Clicked on Ad']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    logging.info("Training abgeschlossen")
    return model, X_test, y_test

@my_logger
def evaluate_model(model, X_test, y_test):
    logging.info("Starte Modellbewertung")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logging.info(f"Modellgenauigkeit: {accuracy:.2f}")
    return accuracy

if __name__ == "__main__":
    df = load_data("advertising.csv")
    model, X_test, y_test = train_model(df)
    acc = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {acc:.2f}")
