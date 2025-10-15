# logistic_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import logging

# Logger konfigurieren
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_data(filename):
    """Lädt die CSV-Daten und loggt die Ausführungszeit."""
    logger.info("Running 'load_data'")
    start = time.perf_counter()
    df = pd.read_csv(filename)
    end = time.perf_counter()
    logger.info(f"load_data executed in {end - start:.4f} sec")
    logger.info("Finished 'load_data'")
    return df


def train_model(df):
    """
    Trainiert ein logistisches Regressionsmodell.
    Gibt (model, X_test, y_test, duration_seconds) zurück.
    """
    logger.info("Running 'train_model'")
    start = time.perf_counter()

    # Beispiel-Datensatzaufbereitung
    X = df.drop("Clicked_on_Ad", axis=1)
    y = df["Clicked_on_Ad"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    end = time.perf_counter()
    duration = end - start

    logger.info("Training abgeschlossen")
    logger.info(f"train_model executed in {duration:.4f} sec")
    logger.info("Finished 'train_model'")

    return model, X_test, y_test, duration


def evaluate_model(model, X_test, y_test):
    """Bewertet das Modell mit Accuracy und Confusion Matrix."""
    logger.info("Running 'evaluate_model'")
    start = time.perf_counter()

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    end = time.perf_counter()

    logger.info(f"Accuracy: {acc:.2f}")
    logger.info("Confusion Matrix:")
    logger.info(f"\n{cm}")
    logger.info(f"evaluate_model executed in {end - start:.4f} sec")
    logger.info("Finished 'evaluate_model'")

    print(f"Accuracy: {acc:.2f}")
    return acc
