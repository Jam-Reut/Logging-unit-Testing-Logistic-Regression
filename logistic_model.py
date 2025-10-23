import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import logging
from functools import wraps

# ------------------------------------------------
# LOGGING-KONFIGURATION
# ------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

_last_timings = {}

# ------------------------------------------------
# @my_timer – Laufzeitmessung
# ------------------------------------------------
def my_timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        logger.info(f"Running '{func.__name__}' ...")
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        _last_timings[func.__name__] = duration
        logger.info(f"'{func.__name__}' executed in {duration:.4f} sec")
        return result
    return wrapper

# ------------------------------------------------
# @my_logger – Start, Ende, Fehler
# ------------------------------------------------
def my_logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Started '{func.__name__}'")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Completed '{func.__name__}' successfully.")
            return result
        except Exception as e:
            logger.error(f"Error in '{func.__name__}': {e}")
            raise
    return wrapper

def get_last_timing(func_name):
    return _last_timings.get(func_name, None)

# ------------------------------------------------
# FUNKTIONEN
# ------------------------------------------------
@my_logger
@my_timer
def load_data(file_path):
    df = pd.read_csv(file_path)
    #logger.info(f"=== Datensatz geladen ===")
    logger.info(f"Datei: {file_path} | Shape: {df.shape}")
    return df

@my_logger
@my_timer
def train_model(df):
    X = df[["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage"]]
    y = df["Clicked on Ad"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

@my_logger
@my_timer
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    #logger.info("=== Modellevaluierung ===")
    logger.info(f"Genauigkeit (Accuracy): {acc:.2f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info("\n" + classification_report(y_test, y_pred))
    return acc

# ------------------------------------------------
# HAUPTLAUF (optional)
# ------------------------------------------------
if __name__ == "__main__":
    logger.info("=== Starte logistic_model.py ===")

    df = load_data("advertising.csv")
    logger.info(f"→ load_data ran in: {get_last_timing('load_data'):.4f} sec")

    model, X_test, y_test = train_model(df)
    logger.info(f"→ train_model ran in: {get_last_timing('train_model'):.4f} sec")

    acc = evaluate_model(model, X_test, y_test)
    logger.info(f"→ evaluate_model ran in: {get_last_timing('evaluate_model'):.4f} sec")
    logger.info(f"Final Accuracy: {acc:.2f}")
