import logging
import time
import pandas as pd
from functools import wraps
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# ===========================================
# Decorators (nach Medium-Artikel)
# ===========================================

def my_logger(func):
    """Decorator: loggt Funktionsaufruf und Parameter in eine Logdatei."""
    logging.basicConfig(
        filename=f"{func.__name__}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Ran with args: {args}, and kwargs: {kwargs}")
        return func(*args, **kwargs)
    return wrapper


def my_timer(func):
    """Decorator: misst und gibt die Laufzeit einer Funktion aus."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time() - t1
        print(f"{func.__name__} ran in: {t2:.4f} sec")
        return result
    return wrapper


# ===========================================
# ML Workflow: Daten laden, trainieren, evaluieren
# ===========================================

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
    """Trainiert ein logistisches Regressionsmodell und gibt Modell + Testdaten zurück."""
    X = df[["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage"]]
    y = df["Clicked on Ad"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    logging.info("Training abgeschlossen")
    return model, X_test, y_test


@my_logger
@my_timer
def evaluate_model(model, X_test, y_test):
    """Berechnet Accuracy und Confusion Matrix."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    logging.info(f"Accuracy: {acc:.2f}")
    logging.info(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {acc:.2f}")
    print("Confusion Matrix:\n", cm)
    return acc, cm


if __name__ == "__main__":
    df = load_data("advertising.csv")
    model, X_test, y_test = train_model(df)
    acc, cm = evaluate_model(model, X_
