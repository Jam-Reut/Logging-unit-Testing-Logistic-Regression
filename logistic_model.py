# ======================================================
# Logistic Regression Example
# Implements Ori Cohen's my_logger & my_timer decorators
# Fulfills assignment requirements for ML transparency
# ======================================================

import logging
import time
from functools import wraps
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# ======================================================
# Decorators (EXACTLY as in Cohen’s article)
# ======================================================
def my_logger(func):
    """Decorator: log function calls into <function>.log"""
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
    """Decorator: measure and print runtime"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time() - start
        print(f"{func.__name__} ran in: {end:.4f} sec")
        # optional: also log runtime in same logfile
        logging.info(f"{func.__name__} ran in {end:.4f} sec")
        return result
    return wrapper


# ======================================================
# ML Pipeline Functions
# ======================================================
@my_logger
@my_timer
def load_data(file_path: str):
    df = pd.read_csv(file_path)
    logging.info(f"Data loaded successfully with shape {df.shape}")
    return df


@my_logger
@my_timer
def train_model(df):
    """Train logistic regression and return model/test sets"""
    X = df[["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage"]]
    y = df["Clicked on Ad"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    logging.info("Training completed successfully.")
    return model, X_test, y_test


@my_logger
@my_timer
def evaluate_model(model, X_test, y_test):
    """Evaluate the model and log results"""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    logging.info(f"Model evaluation: Accuracy={acc:.3f}, ConfusionMatrix={cm.tolist()}")
    print(f"Accuracy: {acc:.2f}")
    print("Confusion Matrix:\n", cm)
    return acc, cm


if __name__ == "__main__":
    df = load_data("advertising.csv")
    model, X_test, y_test = train_model(df)
    acc, cm = evaluate_model(model, X_test, y_test)
