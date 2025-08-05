import pandas as pd
import logging
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def my_timer(func):
    """Decorator zum Messen und Loggen der Laufzeit"""
    def wrapper(*args, **kwargs):
        logging.info(f"Running '{func.__name__}'")
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        logging.info(f"{func.__name__} executed in {elapsed:.4f} sec")
        logging.info(f"Finished '{func.__name__}'")
        return result
    return wrapper

@my_timer
def load_data(file_path):
    logging.info(f"Lade Daten aus {file_path}")
    df = pd.read_csv(file_path)
    logging.info(f"Daten geladen mit Shape {df.shape}")
    return df

@my_timer
def train_model(df):
    # Features und Zielspalte definieren
    X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
    y = df['Clicked on Ad']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    logging.info("Training abgeschlossen")

    # Rückgabe des Modells und Testdaten für spätere Evaluation
    return model, X_test, y_test

@my_timer
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    logging.info(f"Accuracy: {acc:.2f}")
    logging.info("Confusion Matrix:")
    logging.info(f"\n{cm}")

    return acc

if __name__ == "__main__":
    df = load_data("advertising.csv")
    model, X_test, y_test = train_model(df)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")
