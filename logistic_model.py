# ======================================================
# Logistic Regression Example (extended logging version)
# Based on Ori Cohen's article + assignment requirements
# ======================================================

import logging
import time
from functools import wraps
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ======================================================
# Logging Configuration
# ======================================================
logging.basicConfig(
    filename="ml_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ======================================================
# Decorators (enhanced with detailed logging)
# ======================================================
def my_logger(func):
    """Logs call details, arguments and results in detail."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"▶️ Start function '{func.__name__}'")
        logging.info(f"    Args: {args}")
        logging.info(f"    Kwargs: {kwargs}")
        result = func(*args, **kwargs)
        logging.info(f"✅ End function '{func.__name__}'")
        return result
    return wrapper


def my_timer(func):
    """Measures runtime, prints and logs it."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        runtime = time.time() - start
        print(f"{func.__name__} ran in: {runtime:.4f} sec")
        logging.info(f"⏱ '{func.__name__}' ran in {runtime:.4f} sec")
        return result
    return wrapper


# ======================================================
# Machine Learning Workflow
# ======================================================
@my_logger
@my_timer
def load_data(file_path: str):
    df = pd.read_csv(file_path)
    logging.info(f"📂 Data loaded from {file_path} with shape {df.shape}")
    logging.info(f"Columns: {list(df.columns)}")
    return df


@my_logger
@my_timer
def train_model(df):
    """Train logistic regression and return model/test sets."""
    X = df[["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage"]]
    y = df["Clicked on Ad"]

    logging.info("Preparing train/test split (70/30)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    logging.info("Fitting Logistic Regression model...")
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    logging.info(f"Training completed. Model score on test set: {score:.3f}")
    logging.info("Training samples: %d | Test samples: %d", len(X_train), len(X_test))
    return model, X_test, y_test


@my_logger
@my_timer
def evaluate_model(model, X_test, y_test):
    """Evaluate model and log full performance report."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    print(f"Accuracy: {acc:.2f}")
    print("Confusion Matrix:\n", cm)

    logging.info(f"📊 Evaluation metrics for model '{type(model).__name__}':")
    logging.info(f"Accuracy: {acc:.3f}")
    logging.info(f"Confusion Matrix:\n{cm}")
    logging.info("Detailed Classification Report:\n" + report)

    return acc, cm


if __name__ == "__main__":
    df = load_data("advertising.csv")
    model, X_test, y_test = train_model(df)
    acc, cm = evaluate_model(model, X_test, y_test)
