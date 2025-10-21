# ======================================================
# Logistic Regression Example
# Ori Cohen–style logging & timing decorators
# Matching assignment requirements (predict & fit tests)
# ======================================================

import logging
import time
from functools import wraps
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ======================================================
# Decorators (as in Ori Cohen's article)
# ======================================================
def my_logger(func):
    """Logs each function call to a separate logfile."""
    logging.basicConfig(
        filename=f"{func.__name__}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"▶️ Running '{func.__name__}' with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        logging.info(f"✅ Finished '{func.__name__}' successfully.")
        return result
    return wrapper


def my_timer(func):
    """Measures runtime, prints and logs it."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} ran in: {elapsed:.4f} sec")
        logging.info(f"⏱ '{func.__name__}' executed in {elapsed:.4f} sec")
        return result
    return wrapper


# ======================================================
# ML Workflow
# ======================================================
@my_logger
@my_timer
def load_data(file_path: str):
    """Load dataset and log metadata."""
    df = pd.read_csv(file_path)
    logging.info(f"📂 Data loaded from {file_path} with shape {df.shape}")
    logging.info(f"Columns: {list(df.columns)}")
    return df


@my_logger
@my_timer
def train_model(df):
    """Train logistic regression model."""
    X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
    y = df['Clicked on Ad']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    logging.info("Training completed.")
    logging.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    return model, X_test, y_test


@my_logger
@my_timer
def evaluate_model(model, X_test, y_test):
    """Evaluate model with accuracy and confusion matrix."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    print(f"Accuracy: {acc:.2f}")
    print("Confusion Matrix:\n", cm)

    logging.info(f"📊 Model Evaluation:")
    logging.info(f"Accuracy: {acc:.3f}")
    logging.info(f"Confusion Matrix:\n{cm}")
    logging.info("Classification Report:\n" + report)
    return acc


if __name__ == "__main__":
    df = load_data("advertising.csv")
    model, X_test, y_test = train_model(df)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Final Accuracy: {accuracy:.2f}")
