import pandas as pd
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_data(file_path):
    """CSV-Daten laden"""
    logging.info(f"Lade Daten aus {file_path}")
    return pd.read_csv(file_path)

def train_model(df):
    """Logistische Regression trainieren"""
    logging.info("Starte Modelltraining")

    # Features und Zielspalte
    X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
    y = df['Clicked on Ad']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    logging.info("Training abgeschlossen")
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Genauigkeit berechnen"""
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