# ======================================================
# Unit Tests für Logistic Regression Pipeline
# Strukturierte Ausgaben (ohne Emojis, mit #-Kennzeichnungen)
# ======================================================

import unittest
import time
import logging
from logistic_model import load_data, train_model, evaluate_model

# Logging-Konfiguration
logging.basicConfig(
    filename="test_results.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class TestLogisticModel(unittest.TestCase):
    """Automatisierte Tests mit klarer Trennung der Phasen."""

    @classmethod
    def setUpClass(cls):
        print("\n# =====================================")
        print("# SETUP: Initiales Modelltraining")
        print("# =====================================")
        logging.info("SETUP: Lade Daten und trainiere Modell einmalig.")
        cls.df = load_data("advertising.csv")
        cls.model, cls.X_test, cls.y_test = train_model(cls.df)
        print("# Setup abgeschlossen.\n")

    def test_predict_accuracy(self):
        """Test 1: Prüft Modellgenauigkeit & Confusion Matrix."""
        print("# =====================================")
        print("# TEST 1: Modellevaluierung (Accuracy & Confusion Matrix)")
        print("# =====================================")
        logging.info("Starte Test 1: Modellevaluierung")
        accuracy = evaluate_model(self.model, self.X_test, self.y_test)
        print(f"# Ergebnis: Accuracy = {accuracy:.3f}\n")
        logging.info(f"Accuracy erreicht: {accuracy:.3f}")
        self.assertGreaterEqual(
            accuracy, 0.9, "Accuracy unter 0.9 – Modell unzureichend."
        )
        logging.info("Test 1 bestanden.\n")

    def test_fit_runtime(self):
        """Test 2: Prüft Trainingslaufzeit ≤ 120 % der Referenz."""
        print("# =====================================")
        print("# TEST 2: Laufzeitanalyse des Trainings")
        print("# =====================================")
        logging.info("Starte Test 2: Laufzeitanalyse")

        # 1️⃣ Referenzlauf
        print("\n# --- Referenzlauf: Messe Trainingsdauer ---")
        start = time.perf_counter()
        _ = train_model(self.df)
        baseline = time.perf_counter() - start
        logging.info(f"Referenzlaufzeit: {baseline:.4f} sec")

        # 2️⃣ Testlauf
        print("\n# --- Testlauf: Zweiter Trainingsdurchlauf ---")
        start = time.perf_counter()
        _ = train_model(self.df)
        runtime = time.perf_counter() - start
        logging.info(f"Aktuelle Laufzeit: {runtime:.4f} sec")

        # 3️⃣ Vergleich
        limit = baseline * 1.2
        logging.info(f"Erlaubtes Limit (120 %): {limit:.4f} sec")

        print("\n# --- Laufzeit-Analyse ---")
        print(f"# Referenzlaufzeit (erster Lauf): {baseline:.4f} sec")
        print(f"# Aktuelle Laufzeit (zweiter Lauf): {runtime:.4f} sec")
        print(f"# Erlaubtes Limit (120 %): {limit:.4f} sec\n")

        self.assertLessEqual(
            runtime,
            limit,
            f"Laufzeit {runtime:.4f}s überschreitet 120 % der Referenzzeit ({baseline:.4f}s)"
        )
        logging.info("Test 2 bestanden.\n")


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
