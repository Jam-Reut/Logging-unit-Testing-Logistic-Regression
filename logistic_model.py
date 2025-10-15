import pandas as pd
import logging
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def my_timer(func):
    """Decorator zum Messen und Loggen der Laufzeit"""
    def wrapper(*args, **kwargs):
        logger.info(f"Running '{func.__name__}'")
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        logger.info(f"{func.__name__} executed in {elapsed:.4f} sec")
        logger.info(f"Finished '{func.__name__}'")
        return result
    return wrapper

@my_timer
def load_data(file_path):
    logger.info(f"Lade Daten aus {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Daten geladen mit Shape {df.shape}")
    return df

def train_model(df):
    """
    Trainiert ein LogisticRegression-Modell mit StandardScaler.
    Gibt zurück: model, X_test, y_test, duration_seconds
    """
    start = time.perf_counter()

    # --- Spalten prüfen ---
    required_columns = [
        'Daily Time Spent on Site',
        'Age',
        'Area Income',
        'Daily Internet Usage',
        'Clicked on Ad'
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten im DataFrame: {missing}")

    # Referenzen auf die Spalten (kein Kopieren)
    X = df.loc[:, ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
    y = df.loc[:, 'Clicked on Ad']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Features skalieren
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modell trainieren – hoher Accuracy-Solver
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train_scaled, y_train)

    duration = time.perf_counter() - start
    logger.info(f"Training abgeschlossen in {duration:.4f} sec")

    # Rückgabe: Modell, Testdaten (skaliert), Zielwerte, Dauer
    return model, X_test_scaled, y_test, duration

@my_timer
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    logger.info(f"Accuracy: {acc:.2f}")
    logger.info("Confusion Matrix:")
    logger.info(f"\n{cm}")

    return acc

if __name__ == "__main__":
    df = load_data("advertising.csv")
    model, X_test, y_test, duration = train_model(df)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}, Training Time: {duration:.4f} sec")
